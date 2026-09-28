//! NVIDIA's TitaNet speaker embedding network, read from the directory
//! `scripts/titanet_convert.py` writes from a NeMo ONNX export, and run on the crate's own
//! matrix kernel.
//!
//! An embedding is a resumable computation, [`Embedding`]: the front end, every layer and the
//! pooling run as stages over frame ranges, and the sums that squeeze-excitation and pooling
//! take over the whole span are accumulated in frame order. The matrix kernel computes a row
//! the same way whichever rows it is given with it, so a span embedded in slices of any size
//! gives the bytes of one [`TitaNet::embed`] call.
// [[rr:TD-3#Accumulation order is part of the contract]]

// Frame, channel and tensor counts are far below the ranges these casts could lose.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use crate::fbank::{self, BandStats, Fbank, NUM_BANDS};
use crate::gemm::gemm_abt;
use crate::kaldi_io::err;
use crate::resample::LinearResample;
use std::collections::HashMap;
use std::io::Result;
use std::path::Path;

const MAGIC: &[u8; 8] = b"UTTERTN\0";
const VERSION: u32 = 1;
const ALIGN: u64 = 64;
const MAX_NAME: usize = 256;
const MAX_RANK: usize = 3;
/// Sub-blocks per encoder block; the middle blocks carry a residual.
const BLOCKS: [usize; 5] = [1, 3, 3, 3, 1];
const STAT_FLOOR: f32 = 1e-10;
/// The longest span [`TitaNet::embed`] takes, in frames.
/// `[[rr:TD-15#Any span can be embedded from any thread]]`
pub const MAX_FRAMES: usize = 3000;
/// The longest span a recognizer embeds, in frames; a longer one carries no evidence.
/// `[[rr:TD-15#A span is embedded whole at its exact length]]`
pub const RECOGNIZER_MAX_FRAMES: usize = 120;
/// TitaNet-small's shape: each encoder block's kernel widths, the encoder's width before its
/// last block and after it, and the attention's and the embedding's widths.
/// `[[rr:TD-15#Only TitaNet-small is accepted]]`
const SMALL: ([&[usize]; 5], usize, usize, usize, usize) = (
    [&[3], &[7, 7, 7], &[11, 11, 11], &[15, 15, 15], &[1]],
    256,
    3072,
    128,
    192,
);

/// The front end `titanet.conf` must name, the one [`crate::fbank`] implements.
const FRONT_END: [(&str, &str); 7] = [
    ("architecture", "titanet"),
    ("sample_rate", "16000"),
    ("feature_normalize_type", "per_feature"),
    ("window_size_ms", "25"),
    ("window_stride_ms", "10"),
    ("window_type", "hann"),
    ("feat_dim", "80"),
];

struct Tensor {
    dims: Vec<usize>,
    data: Vec<f32>,
}

/// The tensor table of `weights.bin`, every length and offset checked against the file before
/// any tensor is copied out of it.
fn read_tensors(buf: &[u8]) -> Result<HashMap<String, Tensor>> {
    let bad = |what: &str| err(&format!("weights.bin: {what}"));
    if buf.len() < 16 || &buf[..8] != MAGIC {
        return Err(bad("not a converted TitaNet"));
    }
    let u32_at = |at: usize| -> Result<u32> {
        buf.get(at..at + 4)
            .map(|b| u32::from_le_bytes(b.try_into().expect("four bytes")))
            .ok_or_else(|| bad("truncated table"))
    };
    if u32_at(8)? != VERSION {
        return Err(bad("unsupported version"));
    }
    let count = u32_at(12)? as usize;
    // The smallest entry: a name length, a one-byte name, a rank, one dimension, an offset.
    if count > (buf.len() - 16) / 21 {
        return Err(bad("more tensors than the file can hold"));
    }
    let mut at = 16;
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        let len = u32_at(at)? as usize;
        at += 4;
        if len == 0 || len > MAX_NAME {
            return Err(bad("bad tensor name length"));
        }
        let name = buf
            .get(at..at + len)
            .and_then(|b| std::str::from_utf8(b).ok())
            .ok_or_else(|| bad("bad tensor name"))?
            .to_string();
        at += len;
        let rank = u32_at(at)? as usize;
        at += 4;
        if rank == 0 || rank > MAX_RANK {
            return Err(bad("bad tensor rank"));
        }
        let mut dims = Vec::with_capacity(rank);
        let mut size = 1usize;
        for _ in 0..rank {
            let d = u32_at(at)? as usize;
            at += 4;
            size = size
                .checked_mul(d)
                .filter(|_| d > 0)
                .ok_or_else(|| bad("bad tensor dimension"))?;
            dims.push(d);
        }
        let offset = buf
            .get(at..at + 8)
            .map(|b| u64::from_le_bytes(b.try_into().expect("eight bytes")))
            .ok_or_else(|| bad("truncated table"))?;
        at += 8;
        entries.push((name, dims, size, offset));
    }
    let mut regions = Vec::with_capacity(count);
    for (name, _, size, offset) in &entries {
        let bytes = size.checked_mul(4).ok_or_else(|| bad("tensor too large"))?;
        let end = usize::try_from(*offset)
            .ok()
            .and_then(|o| o.checked_add(bytes).map(|e| (o, e)))
            .filter(|&(o, e)| offset % ALIGN == 0 && o >= at && e <= buf.len())
            .ok_or_else(|| bad(&format!("{name} lies outside the data")))?;
        regions.push(end);
    }
    let mut sorted = regions.clone();
    sorted.sort_unstable();
    if sorted.windows(2).any(|w| w[0].1 > w[1].0) {
        return Err(bad("tensors overlap"));
    }
    let mut out = HashMap::with_capacity(count);
    for ((name, dims, _, _), (s, e)) in entries.into_iter().zip(regions) {
        let data = buf[s..e]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().expect("four bytes")))
            .collect();
        if out.insert(name.clone(), Tensor { dims, data }).is_some() {
            return Err(bad(&format!("{name} appears twice")));
        }
    }
    Ok(out)
}

struct Tensors {
    map: HashMap<String, Tensor>,
}

impl Tensors {
    fn take(&mut self, name: &str, dims: &[usize]) -> Result<Vec<f32>> {
        match self.map.remove(name) {
            Some(t) if t.dims == dims => Ok(t.data),
            Some(t) => Err(err(&format!(
                "weights.bin: {name} is {:?}, not {dims:?}",
                t.dims
            ))),
            None => Err(err(&format!("weights.bin: no {name}"))),
        }
    }

    fn dims(&self, name: &str) -> Result<&[usize]> {
        self.map
            .get(name)
            .map(|t| t.dims.as_slice())
            .ok_or_else(|| err(&format!("weights.bin: no {name}")))
    }
}

/// A depthwise convolution with zero padding either side, its kernel `[K x Cin]`, and the
/// pointwise one after it, `[Cout x Cin]`.
struct SepConv {
    k: usize,
    dw: Vec<f32>,
    pw: Vec<f32>,
    pw_bias: Vec<f32>,
    cin: usize,
    cout: usize,
}

struct Block {
    convs: Vec<SepConv>,
    /// Squeeze-excitation, as the graph's MatMuls hold it: `[C x R]` then `[R x C]`.
    se_down: Vec<f32>,
    se_up: Vec<f32>,
    se_width: usize,
    /// `[C x Cin]` and its bias.
    residual: Option<(Vec<f32>, Vec<f32>)>,
}

impl Block {
    fn cin(&self) -> usize {
        self.convs[0].cin
    }

    fn cout(&self) -> usize {
        self.convs[self.convs.len() - 1].cout
    }
}

/// Inference batchnorm as ONNX defines it: `(x - mean) / sqrt(var + epsilon) * scale + bias`.
struct BatchNorm {
    mean: Vec<f32>,
    root: Vec<f32>,
    scale: Vec<f32>,
    bias: Vec<f32>,
}

impl BatchNorm {
    fn read(t: &mut Tensors, path: &str, c: usize) -> Result<BatchNorm> {
        let scale = t.take(&format!("{path}.weight"), &[c])?;
        let bias = t.take(&format!("{path}.bias"), &[c])?;
        let mean = t.take(&format!("{path}.running_mean"), &[c])?;
        let var = t.take(&format!("{path}.running_var"), &[c])?;
        let eps = t.take(&format!("{path}.epsilon"), &[1])?[0];
        let root = var.iter().map(|v| (v + eps).sqrt()).collect();
        Ok(BatchNorm {
            mean,
            root,
            scale,
            bias,
        })
    }

    fn apply(&self, x: &mut [f32]) {
        for (j, v) in x.iter_mut().enumerate() {
            *v = (*v - self.mean[j]) / self.root[j] * self.scale[j] + self.bias[j];
        }
    }
}

/// A TitaNet model directory: `titanet.conf` and `weights.bin`, as
/// `scripts/titanet_convert.py` writes them.
pub struct TitaNet {
    blocks: Vec<Block>,
    /// The attention's first layer, split by what it reads: `[A x C]` over each frame, and
    /// `[A x 2C]` over the mean and deviation, which are the same for every frame.
    att_frame: Vec<f32>,
    att_stats: Vec<f32>,
    att_bias: Vec<f32>,
    att_bn: BatchNorm,
    /// `[C x A]`.
    att_out: Vec<f32>,
    att_out_bias: Vec<f32>,
    emb_bn: BatchNorm,
    /// `[E x 2C]`.
    emb: Vec<f32>,
    emb_bias: Vec<f32>,
    channels: usize,
    att_width: usize,
    dim: usize,
}

impl TitaNet {
    pub fn open(dir: &Path) -> Result<TitaNet> {
        let conf = std::fs::read_to_string(dir.join("titanet.conf"))?;
        let weights = std::fs::read(dir.join("weights.bin"))?;
        Self::from_bytes(&conf, &weights).map_err(|e| err(&format!("{}: {e}", dir.display())))
    }

    /// The model from the two files' contents, refused unless it is TitaNet-small.
    pub fn from_bytes(conf: &str, weights: &[u8]) -> Result<TitaNet> {
        let m = Self::parse(conf, weights)?;
        let (kernels, width, last, att, dim) = SMALL;
        let small = m.blocks.iter().zip(kernels).enumerate().all(|(i, (b, k))| {
            let cout = if i + 1 == kernels.len() { last } else { width };
            b.convs.iter().map(|c| c.k).eq(k.iter().copied()) && b.cout() == cout
        }) && m.att_width == att
            && m.dim == dim;
        if !small {
            return Err(err(&format!(
                "not TitaNet-small (encoder width {}, embedding {}): only TitaNet-small is accepted",
                m.blocks[0].convs[0].cout, m.dim
            )));
        }
        Ok(m)
    }

    /// [`from_bytes`](Self::from_bytes) without the check for TitaNet-small's shape, so the
    /// fuzz target's seed models, a few channels wide, reach [`embed`](Self::embed).
    #[doc(hidden)]
    pub fn from_bytes_any_shape(conf: &str, weights: &[u8]) -> Result<TitaNet> {
        Self::parse(conf, weights)
    }

    fn parse(conf: &str, weights: &[u8]) -> Result<TitaNet> {
        let keys: HashMap<&str, &str> = conf
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.trim(), v.trim()))
            .collect();
        for (k, v) in FRONT_END {
            if keys.get(k) != Some(&v) {
                return Err(err(&format!("titanet.conf: {k} is not {v}")));
            }
        }
        let mut t = Tensors {
            map: read_tensors(weights)?,
        };
        let mut blocks = Vec::with_capacity(BLOCKS.len());
        let mut width = NUM_BANDS;
        for (b, &subs) in BLOCKS.iter().enumerate() {
            let base = format!("encoder.encoder.{b}");
            let block_in = width;
            let mut convs = Vec::with_capacity(subs);
            for s in 0..subs {
                let dw_name = format!("{base}.mconv.{}.conv.weight", 5 * s);
                let k = t.dims(&dw_name)?[0];
                if k % 2 == 0 {
                    return Err(err(&format!("weights.bin: {dw_name} has an even kernel")));
                }
                let dw = t.take(&dw_name, &[k, width])?;
                let pw_name = format!("{base}.mconv.{}.conv.weight", 5 * s + 1);
                let cout = t.dims(&pw_name)?[0];
                let pw = t.take(&pw_name, &[cout, width])?;
                let pw_bias = t.take(&format!("{base}.mconv.{}.conv.bias", 5 * s + 1), &[cout])?;
                convs.push(SepConv {
                    k,
                    dw,
                    pw,
                    pw_bias,
                    cin: width,
                    cout,
                });
                width = cout;
            }
            let se = format!("{base}.mconv.{}.fc", 5 * (subs - 1) + 3);
            let se_width = t
                .dims(&format!("{se}.0.weight"))?
                .get(1)
                .copied()
                .unwrap_or(0);
            let se_down = t.take(&format!("{se}.0.weight"), &[width, se_width])?;
            let se_up = t.take(&format!("{se}.2.weight"), &[se_width, width])?;
            let residual = if b > 0 && b + 1 < BLOCKS.len() {
                let r = format!("{base}.res.0.0.conv");
                Some((
                    t.take(&format!("{r}.weight"), &[width, block_in])?,
                    t.take(&format!("{r}.bias"), &[width])?,
                ))
            } else {
                None
            };
            blocks.push(Block {
                convs,
                se_down,
                se_up,
                se_width,
                residual,
            });
        }
        let c = width;
        let att0 = "decoder._pooling.attention_layer.0.conv_layer";
        let a = t.dims(&format!("{att0}.weight"))?[0];
        let att = t.take(&format!("{att0}.weight"), &[a, 3 * c])?;
        let (mut att_frame, mut att_stats) =
            (Vec::with_capacity(a * c), Vec::with_capacity(a * 2 * c));
        for row in att.chunks_exact(3 * c) {
            att_frame.extend_from_slice(&row[..c]);
            att_stats.extend_from_slice(&row[c..]);
        }
        let att_bias = t.take(&format!("{att0}.bias"), &[a])?;
        let att_bn = BatchNorm::read(&mut t, "decoder._pooling.attention_layer.0.bn", a)?;
        let att2 = "decoder._pooling.attention_layer.2";
        let att_out = t.take(&format!("{att2}.weight"), &[c, a])?;
        let att_out_bias = t.take(&format!("{att2}.bias"), &[c])?;
        let emb_bn = BatchNorm::read(&mut t, "decoder.emb_layers.0.0", 2 * c)?;
        let fc = "decoder.emb_layers.0.1";
        let dim = t.dims(&format!("{fc}.weight"))?[0];
        let emb = t.take(&format!("{fc}.weight"), &[dim, 2 * c])?;
        let emb_bias = t.take(&format!("{fc}.bias"), &[dim])?;
        if let Some(extra) = t.map.keys().next() {
            return Err(err(&format!("weights.bin: unexpected tensor {extra}")));
        }
        if keys.get("dim") != Some(&dim.to_string().as_str()) {
            return Err(err("titanet.conf: dim does not match the embedding layer"));
        }
        Ok(TitaNet {
            blocks,
            att_frame,
            att_stats,
            att_bias,
            att_bn,
            att_out,
            att_out_bias,
            emb_bn,
            emb,
            emb_bias,
            channels: c,
            att_width: a,
            dim,
        })
    }

    /// Length of the embedding.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The embedding of 16 kHz samples in [-1, 1]. Fails on fewer samples than one 25 ms
    /// frame or more than [`MAX_FRAMES`] frames.
    pub fn embed(&self, samples: &[f32]) -> Result<Vec<f32>> {
        let mut e = Embedding::new();
        e.begin(self, samples)?;
        e.advance(self, u64::MAX);
        Ok(e.take())
    }

    /// The embedding of samples at `rate` Hz, brought to 16 kHz first as sherpa-onnx does.
    pub fn embed_at_rate(&self, samples: &[f32], rate: f32) -> Result<Vec<f32>> {
        let mut e = Embedding::new();
        e.begin_at_rate(self, samples, rate)?;
        e.advance(self, u64::MAX);
        Ok(e.take())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Stage {
    Features,
    BandMean,
    BandDeviation,
    BandNormalise,
    /// A separable convolution: its depthwise then its pointwise layer, then ReLU unless it is
    /// the block's last.
    Conv(usize, usize),
    SeSum(usize),
    SeGate(usize),
    /// The gate, the residual where the block has one, and ReLU.
    SeApply(usize),
    PoolMean,
    PoolDeviation,
    PoolStats,
    Attention,
    SoftmaxMax,
    SoftmaxExp,
    WeightedMean,
    WeightedDeviation,
    Embed,
    Done,
}

/// The per-frame cost of the front end, in the multiply-adds the stage costs are counted in.
const FEATURE_COST: u64 = 8_000;

/// One embedding in progress, and the scratch it reuses from one span to the next. The model
/// passed to [`Embedding::advance`] must be the one passed to [`Embedding::begin`].
pub struct Embedding {
    stage: Stage,
    at: usize,
    frames: usize,
    samples: Vec<f32>,
    fbank: Fbank,
    bands: BandStats,
    /// Three `[frames x C]` buffers: a block's input, and two its layers alternate between.
    bufs: [Vec<f32>; 3],
    input: usize,
    scratch: Vec<f32>,
    residual: Vec<f32>,
    sum: Vec<f32>,
    gate: Vec<f32>,
    squeeze: Vec<f32>,
    mean: Vec<f32>,
    var: Vec<f32>,
    stats: Vec<f32>,
    att_bias: Vec<f32>,
    max: Vec<f32>,
    out: Vec<f32>,
}

impl Default for Embedding {
    fn default() -> Self {
        Self::new()
    }
}

impl Embedding {
    pub fn new() -> Embedding {
        Embedding {
            stage: Stage::Done,
            at: 0,
            frames: 0,
            samples: Vec::new(),
            fbank: Fbank::new(),
            bands: BandStats::new(),
            bufs: [Vec::new(), Vec::new(), Vec::new()],
            input: 0,
            scratch: Vec::new(),
            residual: Vec::new(),
            sum: Vec::new(),
            gate: Vec::new(),
            squeeze: Vec::new(),
            mean: Vec::new(),
            var: Vec::new(),
            stats: Vec::new(),
            att_bias: Vec::new(),
            max: Vec::new(),
            out: Vec::new(),
        }
    }

    /// Starts embedding 16 kHz samples in [-1, 1], dropping any embedding in progress.
    pub fn begin(&mut self, model: &TitaNet, samples: &[f32]) -> Result<()> {
        self.samples.clear();
        self.samples.extend_from_slice(samples);
        self.start(model)
    }

    /// Starts embedding samples at `rate` Hz; see [`TitaNet::embed_at_rate`].
    pub fn begin_at_rate(&mut self, model: &TitaNet, samples: &[f32], rate: f32) -> Result<()> {
        let want = fbank::SAMPLE_RATE as f32;
        if rate == want {
            return self.begin(model, samples);
        }
        if !(rate > want && rate.fract() == 0.0 && rate < 1e7) {
            return Err(err(&format!(
                "TitaNet needs audio at {want} Hz or faster, in whole hertz; got {rate}"
            )));
        }
        let cutoff = (0.99 * 0.5 * f64::from(fbank::SAMPLE_RATE)) as f32;
        let mut r = LinearResample::new(rate as i64, i64::from(fbank::SAMPLE_RATE), cutoff, 6);
        self.samples = r.resample(samples, false);
        self.start(model)
    }

    /// Starts the network on features already computed and normalised, `[T x 80]`: the
    /// front end is skipped.
    pub fn begin_features(&mut self, model: &TitaNet, features: &[f32]) -> Result<()> {
        if !features.len().is_multiple_of(NUM_BANDS) {
            return Err(err("features are not rows of 80"));
        }
        self.samples.clear();
        self.set_frames(model, features.len() / NUM_BANDS)?;
        self.bufs[0][..features.len()].copy_from_slice(features);
        self.stage = Stage::Conv(0, 0);
        self.enter(model);
        Ok(())
    }

    fn start(&mut self, model: &TitaNet) -> Result<()> {
        self.set_frames(model, fbank::num_frames(self.samples.len()))
    }

    fn set_frames(&mut self, model: &TitaNet, t: usize) -> Result<()> {
        self.stage = Stage::Done;
        if t == 0 || t > MAX_FRAMES {
            return Err(err(&format!(
                "TitaNet embeds 1 to {MAX_FRAMES} frames of 10 ms; the span has {t}"
            )));
        }
        self.frames = t;
        self.stage = Stage::Features;
        self.at = 0;
        self.input = 0;
        let buf = &mut self.bufs[0];
        buf.clear();
        buf.resize(self.frames * NUM_BANDS, 0.0);
        self.bands.clear();
        let c = model.channels;
        self.mean.clear();
        self.mean.resize(c, 0.0);
        self.var.clear();
        self.var.resize(c, 0.0);
        Ok(())
    }

    /// Frames of the span begun last.
    pub fn frames(&self) -> usize {
        self.frames
    }

    pub fn is_done(&self) -> bool {
        self.stage == Stage::Done
    }

    /// The finished embedding, leaving the scratch for the next span. Empty before the
    /// embedding is done.
    pub fn take(&mut self) -> Vec<f32> {
        if self.is_done() {
            std::mem::take(&mut self.out)
        } else {
            Vec::new()
        }
    }

    /// The stage's cost per frame and its frames; a stage over no frames counts as one.
    fn cost(&self, m: &TitaNet) -> (u64, usize) {
        let c = m.channels as u64;
        let a = m.att_width as u64;
        match self.stage {
            Stage::Features => (FEATURE_COST, self.frames),
            Stage::BandMean | Stage::BandDeviation | Stage::BandNormalise => {
                (NUM_BANDS as u64, self.frames)
            }
            Stage::Conv(b, s) => {
                let sc = &m.blocks[b].convs[s];
                ((sc.k * sc.cin + sc.cin * sc.cout) as u64, self.frames)
            }
            Stage::SeSum(b) => (m.blocks[b].cout() as u64, self.frames),
            Stage::SeGate(b) => (2 * (m.blocks[b].cout() * m.blocks[b].se_width) as u64, 1),
            Stage::SeApply(b) => {
                let bl = &m.blocks[b];
                let res = bl.residual.as_ref().map_or(0, |_| bl.cin() * bl.cout());
                ((bl.cout() + res) as u64, self.frames)
            }
            Stage::PoolMean | Stage::PoolDeviation | Stage::SoftmaxMax => (c, self.frames),
            Stage::WeightedMean | Stage::WeightedDeviation => (c, self.frames),
            Stage::SoftmaxExp => (8 * c, self.frames),
            Stage::PoolStats => (2 * c * a, 1),
            Stage::Attention => (2 * c * a, self.frames),
            Stage::Embed => (2 * c * m.dim as u64, 1),
            Stage::Done => (1, 0),
        }
    }

    fn next(&self, m: &TitaNet) -> Stage {
        let last = m.blocks.len() - 1;
        match self.stage {
            Stage::Features => Stage::BandMean,
            Stage::BandMean => Stage::BandDeviation,
            Stage::BandDeviation => Stage::BandNormalise,
            Stage::BandNormalise => Stage::Conv(0, 0),
            Stage::Conv(b, s) if s + 1 < m.blocks[b].convs.len() => Stage::Conv(b, s + 1),
            Stage::Conv(b, _) => Stage::SeSum(b),
            Stage::SeSum(b) => Stage::SeGate(b),
            Stage::SeGate(b) => Stage::SeApply(b),
            Stage::SeApply(b) if b < last => Stage::Conv(b + 1, 0),
            Stage::SeApply(_) => Stage::PoolMean,
            Stage::PoolMean => Stage::PoolDeviation,
            Stage::PoolDeviation => Stage::PoolStats,
            Stage::PoolStats => Stage::Attention,
            Stage::Attention => Stage::SoftmaxMax,
            Stage::SoftmaxMax => Stage::SoftmaxExp,
            Stage::SoftmaxExp => Stage::WeightedMean,
            Stage::WeightedMean => Stage::WeightedDeviation,
            Stage::WeightedDeviation => Stage::Embed,
            Stage::Embed | Stage::Done => Stage::Done,
        }
    }

    /// Runs the embedding on until `budget` units of work are spent or it is done, and
    /// returns the units spent. Work is counted in whole frames of a stage at that stage's
    /// multiply-adds per frame, never in time, and each call runs at least one frame or step,
    /// so the same budgets give the same slices on any machine and repeated calls finish.
    pub fn advance(&mut self, m: &TitaNet, budget: u64) -> u64 {
        let mut spent = 0u64;
        while self.stage != Stage::Done {
            let (per_frame, total) = self.cost(m);
            let afford = usize::try_from((budget - spent) / per_frame).unwrap_or(usize::MAX);
            let n = if spent == 0 { afford.max(1) } else { afford }.min(total - self.at);
            if n == 0 {
                break;
            }
            self.run(m, self.at, self.at + n);
            self.at += n;
            spent = spent.saturating_add(per_frame * n as u64);
            if self.at == total {
                self.finish(m);
                self.stage = self.next(m);
                self.at = 0;
                self.enter(m);
            }
            if spent >= budget {
                break;
            }
        }
        spent
    }

    /// The layer buffers of block `b`'s sub-block `s`: what it reads and what it writes.
    fn conv_bufs(&self, s: usize) -> (usize, usize) {
        let (p, q) = ((self.input + 1) % 3, (self.input + 2) % 3);
        match s {
            0 => (self.input, p),
            s if s % 2 == 1 => (p, q),
            _ => (q, p),
        }
    }

    /// Where block `b`'s output is: its last sub-block's destination.
    fn block_out(&self, m: &TitaNet, b: usize) -> usize {
        self.conv_bufs(m.blocks[b].convs.len() - 1).1
    }

    /// Sizes the stage's buffers and clears its sums, when it begins.
    fn enter(&mut self, m: &TitaNet) {
        let c = m.channels;
        match self.stage {
            Stage::Conv(b, s) => {
                let dst = self.conv_bufs(s).1;
                let buf = &mut self.bufs[dst];
                buf.clear();
                buf.resize(self.frames * m.blocks[b].convs[s].cout, 0.0);
            }
            Stage::SeSum(b) => {
                self.sum.clear();
                self.sum.resize(m.blocks[b].cout(), 0.0);
            }
            Stage::PoolMean | Stage::WeightedMean => self.mean.fill(0.0),
            Stage::PoolDeviation | Stage::WeightedDeviation => self.var.fill(0.0),
            Stage::Attention => {
                let free = (self.input + 1) % 3;
                self.bufs[free].clear();
                self.bufs[free].resize(self.frames * c, 0.0);
            }
            Stage::SoftmaxMax => {
                self.max.clear();
                self.max.resize(c, f32::NEG_INFINITY);
            }
            Stage::SoftmaxExp => {
                self.sum.clear();
                self.sum.resize(c, 0.0);
            }
            _ => {}
        }
    }

    /// Completes the stage once its last frame has run.
    fn finish(&mut self, m: &TitaNet) {
        match self.stage {
            Stage::BandMean => self.bands.end_mean(self.frames),
            Stage::BandDeviation => self.bands.end_deviation(self.frames),
            Stage::SeApply(b) => self.input = self.block_out(m, b),
            _ => {}
        }
    }

    fn run(&mut self, m: &TitaNet, f0: usize, f1: usize) {
        let c = m.channels;
        let weight = 1.0 / self.frames as f32;
        match self.stage {
            Stage::Features => {
                let rows = &mut self.bufs[0][f0 * NUM_BANDS..f1 * NUM_BANDS];
                for (i, row) in rows.chunks_exact_mut(NUM_BANDS).enumerate() {
                    self.fbank.frame(&self.samples, f0 + i, row);
                }
            }
            Stage::BandMean => self
                .bands
                .add_mean(&self.bufs[0][f0 * NUM_BANDS..f1 * NUM_BANDS]),
            Stage::BandDeviation => self
                .bands
                .add_deviation(&self.bufs[0][f0 * NUM_BANDS..f1 * NUM_BANDS]),
            Stage::BandNormalise => self
                .bands
                .normalise(&mut self.bufs[0][f0 * NUM_BANDS..f1 * NUM_BANDS]),
            Stage::Conv(b, s) => {
                let block = &m.blocks[b];
                let sc = &block.convs[s];
                let (src, dst) = self.conv_bufs(s);
                depthwise(&self.bufs[src], self.frames, sc, f0, f1, &mut self.scratch);
                let out = &mut self.bufs[dst][f0 * sc.cout..f1 * sc.cout];
                gemm_abt(
                    &self.scratch,
                    f1 - f0,
                    sc.cin,
                    &sc.pw,
                    sc.cout,
                    Some(&sc.pw_bias),
                    out,
                );
                if s + 1 < block.convs.len() {
                    relu(out);
                }
            }
            Stage::SeSum(b) => {
                let w = m.blocks[b].cout();
                let h = &self.bufs[self.block_out(m, b)][f0 * w..f1 * w];
                for row in h.chunks_exact(w) {
                    for (s, v) in self.sum.iter_mut().zip(row) {
                        *s += v;
                    }
                }
            }
            Stage::SeGate(b) => self.se_gate(m, b),
            Stage::SeApply(b) => self.se_apply(m, b, f0, f1),
            Stage::PoolMean => {
                let h = &self.bufs[self.input][f0 * c..f1 * c];
                for row in h.chunks_exact(c) {
                    for (s, v) in self.mean.iter_mut().zip(row) {
                        *s += weight * v;
                    }
                }
            }
            Stage::PoolDeviation => {
                let h = &self.bufs[self.input][f0 * c..f1 * c];
                for row in h.chunks_exact(c) {
                    for ((s, v), mu) in self.var.iter_mut().zip(row).zip(&self.mean) {
                        let d = v - mu;
                        *s += weight * (d * d);
                    }
                }
            }
            Stage::PoolStats => {
                self.stats.clear();
                self.stats.extend_from_slice(&self.mean);
                self.stats
                    .extend(self.var.iter().map(|v| v.max(STAT_FLOOR).sqrt()));
                self.att_bias.clear();
                self.att_bias.resize(m.att_width, 0.0);
                gemm_abt(
                    &self.stats,
                    1,
                    2 * c,
                    &m.att_stats,
                    m.att_width,
                    Some(&m.att_bias),
                    &mut self.att_bias,
                );
            }
            Stage::Attention => self.attention(m, f0, f1),
            Stage::SoftmaxMax => {
                let e = &self.bufs[(self.input + 1) % 3][f0 * c..f1 * c];
                for row in e.chunks_exact(c) {
                    for (mx, v) in self.max.iter_mut().zip(row) {
                        *mx = mx.max(*v);
                    }
                }
            }
            Stage::SoftmaxExp => {
                let e = &mut self.bufs[(self.input + 1) % 3][f0 * c..f1 * c];
                for row in e.chunks_exact_mut(c) {
                    for ((v, mx), s) in row.iter_mut().zip(&self.max).zip(self.sum.iter_mut()) {
                        *v = (*v - mx).exp();
                        *s += *v;
                    }
                }
            }
            Stage::WeightedMean => {
                let (h, e) = two(&mut self.bufs, self.input, (self.input + 1) % 3);
                let rows = h[f0 * c..f1 * c]
                    .chunks_exact(c)
                    .zip(e[f0 * c..f1 * c].chunks_exact_mut(c));
                for (hr, er) in rows {
                    for (((a, v), s), mu) in er
                        .iter_mut()
                        .zip(hr)
                        .zip(&self.sum)
                        .zip(self.mean.iter_mut())
                    {
                        *a /= s;
                        *mu += *a * v;
                    }
                }
            }
            Stage::WeightedDeviation => {
                let (h, e) = (&self.bufs[self.input], &self.bufs[(self.input + 1) % 3]);
                let rows = h[f0 * c..f1 * c]
                    .chunks_exact(c)
                    .zip(e[f0 * c..f1 * c].chunks_exact(c));
                for (hr, er) in rows {
                    for (((s, a), v), mu) in self.var.iter_mut().zip(er).zip(hr).zip(&self.mean) {
                        let d = v - mu;
                        *s += a * (d * d);
                    }
                }
            }
            Stage::Embed => {
                self.stats.clear();
                self.stats.extend_from_slice(&self.mean);
                self.stats
                    .extend(self.var.iter().map(|v| v.max(STAT_FLOOR).sqrt()));
                m.emb_bn.apply(&mut self.stats);
                self.out.clear();
                self.out.resize(m.dim, 0.0);
                gemm_abt(
                    &self.stats,
                    1,
                    2 * c,
                    &m.emb,
                    m.dim,
                    Some(&m.emb_bias),
                    &mut self.out,
                );
            }
            Stage::Done => {}
        }
    }

    fn se_gate(&mut self, m: &TitaNet, b: usize) {
        let block = &m.blocks[b];
        let (w, r) = (block.cout(), block.se_width);
        let count = self.frames as f32;
        self.squeeze.clear();
        self.squeeze.resize(r, 0.0);
        for (ch, s) in self.sum.iter().enumerate() {
            let mean = s / count;
            for (q, wt) in self
                .squeeze
                .iter_mut()
                .zip(&block.se_down[ch * r..(ch + 1) * r])
            {
                *q += mean * wt;
            }
        }
        relu(&mut self.squeeze);
        self.gate.clear();
        self.gate.resize(w, 0.0);
        for (j, q) in self.squeeze.iter().enumerate() {
            for (g, wt) in self.gate.iter_mut().zip(&block.se_up[j * w..(j + 1) * w]) {
                *g += q * wt;
            }
        }
        for g in &mut self.gate {
            *g = 1.0 / (1.0 + (-*g).exp());
        }
    }

    fn se_apply(&mut self, m: &TitaNet, b: usize, f0: usize, f1: usize) {
        let block = &m.blocks[b];
        let (cin, w) = (block.cin(), block.cout());
        let n = f1 - f0;
        if let Some((rw, rb)) = &block.residual {
            self.residual.clear();
            self.residual.resize(n * w, 0.0);
            gemm_abt(
                &self.bufs[self.input][f0 * cin..f1 * cin],
                n,
                cin,
                rw,
                w,
                Some(rb),
                &mut self.residual,
            );
        }
        let out = self.block_out(m, b);
        let h = &mut self.bufs[out][f0 * w..f1 * w];
        for (i, row) in h.chunks_exact_mut(w).enumerate() {
            for (j, (v, g)) in row.iter_mut().zip(&self.gate).enumerate() {
                let mut y = *v * g;
                if block.residual.is_some() {
                    y += self.residual[i * w + j];
                }
                *v = y.max(0.0);
            }
        }
    }

    fn attention(&mut self, m: &TitaNet, f0: usize, f1: usize) {
        let (c, a) = (m.channels, m.att_width);
        let n = f1 - f0;
        self.scratch.clear();
        self.scratch.resize(n * a, 0.0);
        let (h, e) = two(&mut self.bufs, self.input, (self.input + 1) % 3);
        gemm_abt(
            &h[f0 * c..f1 * c],
            n,
            c,
            &m.att_frame,
            a,
            Some(&self.att_bias),
            &mut self.scratch,
        );
        for row in self.scratch.chunks_exact_mut(a) {
            relu(row);
            m.att_bn.apply(row);
            for v in row.iter_mut() {
                *v = v.tanh();
            }
        }
        gemm_abt(
            &self.scratch,
            n,
            a,
            &m.att_out,
            c,
            Some(&m.att_out_bias),
            &mut e[f0 * c..f1 * c],
        );
    }
}

/// Two distinct buffers of three, the first shared and the second mutable.
fn two(bufs: &mut [Vec<f32>; 3], i: usize, j: usize) -> (&[f32], &mut [f32]) {
    assert_ne!(i, j);
    let [b0, b1, b2] = bufs;
    let (x, y): (&Vec<f32>, &mut Vec<f32>) = match (i, j) {
        (0, 1) => (b0, b1),
        (0, 2) => (b0, b2),
        (1, 0) => (b1, b0),
        (1, 2) => (b1, b2),
        (2, 0) => (b2, b0),
        _ => (b2, b1),
    };
    (x, y)
}

fn relu(x: &mut [f32]) {
    for v in x {
        *v = v.max(0.0);
    }
}

/// Frames `f0..f1` of a same-length depthwise convolution over `x`, `[t x Cin]`, into `out`.
/// Plain products and sums: `f32::mul_add` without the fma target feature is a call to the
/// software `fmaf`.
fn depthwise(x: &[f32], t: usize, sc: &SepConv, f0: usize, f1: usize, out: &mut Vec<f32>) {
    let c = sc.cin;
    let p = sc.k / 2;
    out.clear();
    out.resize((f1 - f0) * c, 0.0);
    for (i, o) in (f0..f1).zip(out.chunks_exact_mut(c)) {
        for j in 0..sc.k {
            let Some(src) = (i + j).checked_sub(p).filter(|&s| s < t) else {
                continue;
            };
            let s = &x[src * c..(src + 1) * c];
            let w = &sc.dw[j * c..(j + 1) * c];
            for ((o, s), w) in o.iter_mut().zip(s).zip(w) {
                *o += s * w;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Named = (String, Vec<usize>, Vec<f32>);

    fn write(tensors: &[Named]) -> Vec<u8> {
        let mut out = MAGIC.to_vec();
        out.extend(VERSION.to_le_bytes());
        out.extend((tensors.len() as u32).to_le_bytes());
        let mut slots = Vec::new();
        for (name, dims, _) in tensors {
            out.extend((name.len() as u32).to_le_bytes());
            out.extend(name.as_bytes());
            out.extend((dims.len() as u32).to_le_bytes());
            for d in dims {
                out.extend((*d as u32).to_le_bytes());
            }
            slots.push(out.len());
            out.extend([0u8; 8]);
        }
        for ((_, _, data), slot) in tensors.iter().zip(slots) {
            while !out.len().is_multiple_of(ALIGN as usize) {
                out.push(0);
            }
            let at = out.len() as u64;
            out[slot..slot + 8].copy_from_slice(&at.to_le_bytes());
            for v in data {
                out.extend(v.to_le_bytes());
            }
        }
        out
    }

    /// Values from a fixed generator, so the tiny model is the same on every run.
    struct Values(u32);

    impl Values {
        fn take(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n)
                .map(|_| {
                    self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((self.0 >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * scale
                })
                .collect()
        }

        fn tensor(&mut self, t: &mut Vec<Named>, name: String, dims: Vec<usize>, scale: f32) {
            let data = self.take(dims.iter().product(), scale);
            t.push((name, dims, data));
        }

        fn batchnorm(&mut self, t: &mut Vec<Named>, path: &str, n: usize) {
            self.tensor(t, format!("{path}.weight"), vec![n], 1.0);
            self.tensor(t, format!("{path}.bias"), vec![n], 0.2);
            self.tensor(t, format!("{path}.running_mean"), vec![n], 0.2);
            let var = self.take(n, 0.5).into_iter().map(|x| x + 0.5).collect();
            t.push((format!("{path}.running_var"), vec![n], var));
            t.push((format!("{path}.epsilon"), vec![1], vec![1e-5]));
        }
    }

    /// A model of TitaNet's shape at a few channels.
    fn tiny() -> Vec<Named> {
        let mut v = Values(7);
        let mut t: Vec<Named> = Vec::new();
        let (c, last, r, a, e) = (8, 16, 2, 4, 3);
        let kernels = [[3, 0, 0], [3, 5, 7], [3, 5, 7], [3, 5, 7], [1, 0, 0]];
        let mut width = NUM_BANDS;
        for (b, &subs) in BLOCKS.iter().enumerate() {
            let base = format!("encoder.encoder.{b}");
            let block_in = width;
            let cout = if b + 1 == BLOCKS.len() { last } else { c };
            for s in 0..subs {
                let k = kernels[b][s];
                let dw = format!("{base}.mconv.{}.conv", 5 * s);
                let pw = format!("{base}.mconv.{}.conv", 5 * s + 1);
                v.tensor(&mut t, format!("{dw}.weight"), vec![k, width], 0.8);
                v.tensor(&mut t, format!("{pw}.weight"), vec![cout, width], 0.6);
                v.tensor(&mut t, format!("{pw}.bias"), vec![cout], 0.2);
                width = cout;
            }
            let se = format!("{base}.mconv.{}.fc", 5 * (subs - 1) + 3);
            v.tensor(&mut t, format!("{se}.0.weight"), vec![width, r], 1.0);
            v.tensor(&mut t, format!("{se}.2.weight"), vec![r, width], 1.0);
            if b > 0 && b + 1 < BLOCKS.len() {
                let res = format!("{base}.res.0.0.conv");
                v.tensor(&mut t, format!("{res}.weight"), vec![width, block_in], 0.6);
                v.tensor(&mut t, format!("{res}.bias"), vec![width], 0.2);
            }
        }
        let att0 = "decoder._pooling.attention_layer.0.conv_layer";
        v.tensor(&mut t, format!("{att0}.weight"), vec![a, 3 * last], 0.4);
        v.tensor(&mut t, format!("{att0}.bias"), vec![a], 0.2);
        v.batchnorm(&mut t, "decoder._pooling.attention_layer.0.bn", a);
        let att2 = "decoder._pooling.attention_layer.2";
        v.tensor(&mut t, format!("{att2}.weight"), vec![last, a], 1.0);
        v.tensor(&mut t, format!("{att2}.bias"), vec![last], 0.2);
        v.batchnorm(&mut t, "decoder.emb_layers.0.0", 2 * last);
        let fc = "decoder.emb_layers.0.1";
        v.tensor(&mut t, format!("{fc}.weight"), vec![e, 2 * last], 0.4);
        v.tensor(&mut t, format!("{fc}.bias"), vec![e], 0.2);
        t
    }

    fn conf(dim: usize) -> String {
        let mut s: String = FRONT_END
            .iter()
            .map(|(k, v)| format!("{k}={v}\n"))
            .collect();
        s.push_str(&format!("dim={dim}\n"));
        s
    }

    fn signal(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / 16_000.0;
                0.3 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                    + 0.1 * (2.0 * std::f32::consts::PI * 1_730.0 * t * (1.0 + t)).sin()
            })
            .collect()
    }

    fn tiny_model() -> TitaNet {
        TitaNet::parse(&conf(3), &write(&tiny())).unwrap()
    }

    #[test]
    fn only_titanet_small_opens() {
        let e = TitaNet::from_bytes(&conf(3), &write(&tiny()))
            .err()
            .unwrap();
        assert!(e.to_string().contains("only TitaNet-small"), "{e}");
    }

    #[test]
    fn models_and_jobs_cross_threads() {
        fn both<T: Send + Sync>() {}
        both::<TitaNet>();
        both::<Embedding>();
    }

    #[test]
    // Miri perturbs float intrinsics by a few ULP, so the exact-bytes compare below only holds natively.
    #[cfg_attr(miri, ignore)]
    fn a_tiny_model_embeds_the_same_bytes_in_any_slices() {
        let m = tiny_model();
        assert_eq!(m.dim(), 3);
        for n in [400, 400 + 160 * 24, 16_000] {
            let x = signal(n);
            let whole = m.embed(&x).unwrap();
            assert!(whole.iter().all(|v| v.is_finite()));
            let mut job = Embedding::new();
            let mut seed = 0x2545_f491u64;
            for fixed in [Some(1u64), Some(700), Some(50_000), None] {
                job.begin(&m, &x).unwrap();
                while !job.is_done() {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    job.advance(&m, fixed.unwrap_or(1 + seed % 20_000));
                }
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&job.take()), bits(&whole));
            }
        }
    }

    #[test]
    fn work_is_counted_in_frames_not_time() {
        let m = tiny_model();
        let x = signal(8_000);
        let mut job = Embedding::new();
        job.begin(&m, &x).unwrap();
        let (mut total, mut calls) = (0, 0);
        while !job.is_done() {
            let spent = job.advance(&m, 5_000);
            assert!(spent > 0);
            total += spent;
            calls += 1;
        }
        assert!(calls > 10);
        job.begin(&m, &x).unwrap();
        assert!(job.take().is_empty());
        assert_eq!(job.advance(&m, u64::MAX), total);
        assert_eq!(job.take().len(), 3);
    }

    #[test]
    // Miri perturbs float intrinsics by a few ULP, so the exact-bytes compare below only holds natively.
    #[cfg_attr(miri, ignore)]
    fn a_job_starts_over_on_begin() {
        let m = tiny_model();
        let (a, b) = (signal(6_000), signal(9_000));
        let mut job = Embedding::new();
        job.begin(&m, &a).unwrap();
        job.advance(&m, 30_000);
        job.begin(&m, &b).unwrap();
        job.advance(&m, u64::MAX);
        assert_eq!(job.take(), m.embed(&b).unwrap());
    }

    #[test]
    fn malformed_weights_are_refused() {
        let good = write(&tiny());
        let c = conf(3);
        assert!(TitaNet::parse(&c, &good).is_ok());
        for cut in [0, 8, 15, 16, 40, good.len() / 2, good.len() - 1] {
            assert!(TitaNet::parse(&c, &good[..cut]).is_err(), "cut at {cut}");
        }
        let mut bad = good.clone();
        bad[0] = b'X';
        assert!(TitaNet::parse(&c, &bad).is_err());
        let mut bad = good.clone();
        bad[8] = 2;
        assert!(TitaNet::parse(&c, &bad).is_err());
        // A tensor count the file cannot hold is refused before anything is allocated.
        let mut bad = good.clone();
        bad[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(TitaNet::parse(&c, &bad).is_err());
        // The first tensor's offset: misaligned, inside the table, past the end, overflowing.
        let first = &tiny()[0];
        let slot = 16 + 4 + first.0.len() + 4 + 4 * first.1.len();
        let off = u64::from_le_bytes(good[slot..slot + 8].try_into().unwrap());
        for o in [off + 4, 0, good.len() as u64, u64::MAX - 63] {
            let mut bad = good.clone();
            bad[slot..slot + 8].copy_from_slice(&o.to_le_bytes());
            assert!(TitaNet::parse(&c, &bad).is_err(), "offset {o}");
        }
        // Two tensors over the same bytes.
        let second = &tiny()[1];
        let slot2 = slot + 8 + 4 + second.0.len() + 4 + 4 * second.1.len();
        let mut bad = good.clone();
        bad.copy_within(slot..slot + 8, slot2);
        assert!(TitaNet::parse(&c, &bad).is_err());
        // A zero dimension, a wrong shape, a tensor missing, one too many, one twice.
        let mut z = tiny();
        z[0].1[0] = 0;
        z[0].2.clear();
        assert!(TitaNet::parse(&c, &write(&z)).is_err());
        let mut w = tiny();
        w[1].1.reverse();
        assert!(TitaNet::parse(&c, &write(&w)).is_err());
        let mut gone = tiny();
        gone.remove(5);
        assert!(TitaNet::parse(&c, &write(&gone)).is_err());
        let mut extra = tiny();
        extra.push(("classifier".into(), vec![1], vec![0.0]));
        assert!(TitaNet::parse(&c, &write(&extra)).is_err());
        let mut twice = tiny();
        twice.push(twice[0].clone());
        assert!(TitaNet::parse(&c, &write(&twice)).is_err());
    }

    #[test]
    fn another_front_end_or_dimension_is_refused() {
        let w = write(&tiny());
        assert!(TitaNet::parse(&conf(4), &w).is_err());
        assert!(TitaNet::parse(&conf(3).replace("hann", "povey"), &w).is_err());
        assert!(TitaNet::parse(&conf(3).replace("16000", "8000"), &w).is_err());
        assert!(TitaNet::parse("", &w).is_err());
    }
}
