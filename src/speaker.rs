//! Speaker evidence: a Kaldi x-vector network run frame by frame beside the decoder, whose
//! frame-level rows are pooled over any span of the utterance into a speaker vector.
// [[rr:TD-14#Decision outcome]]

use crate::frontend::OnlineMfcc;
use crate::gemm::gemm_abt;
use crate::kaldi_io::{err, find, parse_text_matrix, KaldiReader};
use crate::mfcc::MfccOptions;
use crate::nnet3::{InputView, Nnet3, Streamer};
use crate::resample::LinearResample;
use std::collections::VecDeque;
use std::io::Result;
use std::path::Path;

/// Pooled frames below which a span carries no evidence.
/// `[[rr:TD-14#A span needs a quarter second]]`
pub const MIN_FRAMES: usize = 25;
/// `[[rr:TD-14#Mean normalisation looks back only]]`
const CMN_WINDOW: usize = 300;
/// `[[rr:TD-14#Pooling reads the frames it keeps]]`
const MAX_ROWS: usize = 3000;

/// A speaker model directory as vosk's `SpkModel` reads it: `mfcc.conf`, the x-vector network
/// `final.ext.raw` ending in an affine over pooled statistics, and the centring vector
/// `mean.vec` and whitening matrix `transform.mat` applied to its output.
pub struct SpeakerModel {
    net: Nnet3,
    mfcc: MfccOptions,
    /// The embedding affine, `[embed_rows, 2 * frame_dim]` row-major, and its bias.
    embed: Vec<f32>,
    embed_bias: Vec<f32>,
    embed_rows: usize,
    mean: Vec<f32>,
    /// `[dim, embed_rows]` row-major.
    transform: Vec<f32>,
    dim: usize,
    frame_dim: usize,
    variance_floor: f32,
}

/// The component named `name` in a raw network: its bytes after the name, to the next one.
fn component_bytes<'a>(buf: &'a [u8], name: &str) -> Option<&'a [u8]> {
    let tag = format!("<ComponentName> {name} ");
    let s = find(buf, tag.as_bytes(), 0)? + tag.len();
    let e = find(buf, b"<ComponentName> ", s).unwrap_or(buf.len());
    Some(&buf[s..e])
}

fn field<T>(
    region: &[u8],
    token: &str,
    read: impl Fn(&mut KaldiReader<&[u8]>) -> Result<T>,
) -> Option<T> {
    let tag = format!("{token} ");
    let at = find(region, tag.as_bytes(), 0)? + tag.len();
    read(&mut KaldiReader::new(&region[at..])).ok()
}

/// A Kaldi vector or matrix file in either form, binary (`\0B` then `FV`/`FM`/`DV`/`DM`) or
/// text (`[ ... ]`), as `(rows, cols, data)`; a vector is one row.
#[allow(clippy::cast_possible_truncation)]
fn read_kaldi_matrix(path: &Path) -> Result<(usize, usize, Vec<f32>)> {
    let buf = std::fs::read(path)?;
    if buf.starts_with(b"\0B") {
        let mut r = KaldiReader::new(&buf[2..]);
        let kind = r.read_token()?;
        let mut r = KaldiReader::new(&buf[2..]);
        return match kind.as_str() {
            "FV" => r.read_float_vec().map(|v| (1, v.len(), v)),
            "FM" => r.read_float_matrix(),
            "DV" => r
                .read_double_vec()
                .map(|v| (1, v.len(), v.into_iter().map(|x| x as f32).collect())),
            "DM" => r
                .read_double_matrix()
                .map(|(rows, cols, d)| (rows, cols, d.into_iter().map(|x| x as f32).collect())),
            other => Err(err(&format!(
                "{}: unexpected object {other}",
                path.display()
            ))),
        };
    }
    let txt = String::from_utf8_lossy(&buf);
    let (rows, cols, d) = parse_text_matrix(&txt)?;
    Ok((rows, cols, d.into_iter().map(|x| x as f32).collect()))
}

impl SpeakerModel {
    /// Open a speaker model directory. Fails on a missing file, a network that is not an affine
    /// over mean and standard-deviation pooling of one frame-level node, or a component before
    /// the pooling this crate does not implement.
    pub fn open(dir: &Path) -> Result<SpeakerModel> {
        let mut mfcc = MfccOptions::from_conf(&std::fs::read_to_string(dir.join("mfcc.conf"))?);
        // [[rr:TD-14#The front end is the model's own]]
        mfcc.dither = 0.0;
        let buf = std::fs::read(dir.join("final.ext.raw"))?;
        let mut net = Nnet3::parse(&buf);
        let refused = |what: &str| err(&format!("{}: {what}", dir.display()));
        let head = net.pooling_head().ok_or_else(|| {
            refused("the network's output is not an affine over pooled statistics")
        })?;
        let (embed_comp, frames) = (head.embed.to_string(), head.frames.to_string());
        let pooling =
            component_bytes(&buf, head.pooling).ok_or_else(|| refused("no pooling component"))?;
        let extraction = component_bytes(&buf, head.extraction)
            .ok_or_else(|| refused("no extraction component"))?;
        let int = |r: &[u8], t: &str| field(r, t, |k| k.read_i32());
        let flag = |r: &[u8], t: &str| field(r, t, |k| k.read_bool());
        // Kaldi writes the extraction's variance flag under this spelling.
        let with_variance = flag(extraction, "<IncludeVarinance>");
        let stddevs = flag(pooling, "<OutputStddevs>");
        let log_counts = int(pooling, "<NumLogCountFeatures>");
        let periods = [
            int(extraction, "<InputPeriod>"),
            int(extraction, "<OutputPeriod>"),
            int(pooling, "<InputPeriod>"),
        ];
        if with_variance != Some(true)
            || stddevs != Some(true)
            || log_counts != Some(0)
            || periods.iter().any(|p| *p != Some(1))
        {
            return Err(refused(
                "pooling other than the mean and standard deviation of every frame is not supported",
            ));
        }
        let variance_floor = field(pooling, "<VarianceFloor>", |k| k.read_f32()).unwrap_or(1e-10);
        let (w, b) = net
            .affine(&embed_comp)
            .map(|(w, b)| (w.clone(), b.to_vec()))
            .ok_or_else(|| refused("no embedding affine"))?;
        net.retarget_output(&frames);
        let missing = net.unsupported();
        if !missing.is_empty() {
            return Err(refused(&format!(
                "network components not implemented: {}",
                missing.join(", ")
            )));
        }
        let frame_dim = net.output_dim;
        if w.c != 2 * frame_dim || net.input_dim != mfcc.num_ceps {
            return Err(refused(
                "the network's dimensions do not agree with its pooling and features",
            ));
        }
        let (_, mean_len, mean) = read_kaldi_matrix(&dir.join("mean.vec"))?;
        let (dim, t_cols, transform) = read_kaldi_matrix(&dir.join("transform.mat"))?;
        if mean_len != w.r || t_cols != w.r {
            return Err(refused(
                "mean.vec and transform.mat do not match the embedding",
            ));
        }
        Ok(SpeakerModel {
            net,
            mfcc,
            embed: w.d,
            embed_bias: b,
            embed_rows: w.r,
            mean,
            transform,
            dim,
            frame_dim,
            variance_floor,
        })
    }

    /// Length of the speaker vector.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Milliseconds between frames of the model's features.
    pub(crate) fn frame_shift_ms(&self) -> f32 {
        self.mfcc.frame_shift_ms
    }

    /// The vector for pooled statistics: the embedding affine over their mean and standard
    /// deviation, centred, whitened, and scaled to the norm a standard normal vector of its
    /// length has in expectation, as vosk's `GetSpkVector` leaves it.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn embed(&self, count: usize, sum: &[f64], sum_sq: &[f64]) -> Vec<f32> {
        let n = count as f64;
        let d = self.frame_dim;
        let mut stats = vec![0.0f32; 2 * d];
        for i in 0..d {
            let mean = sum[i] / n;
            let var = (sum_sq[i] / n - mean * mean).max(f64::from(self.variance_floor));
            stats[i] = mean as f32;
            stats[d + i] = var.sqrt() as f32;
        }
        let mut y = vec![0.0f32; self.embed_rows];
        gemm_abt(
            &stats,
            1,
            2 * d,
            &self.embed,
            self.embed_rows,
            Some(&self.embed_bias),
            &mut y,
        );
        for (v, m) in y.iter_mut().zip(&self.mean) {
            *v -= m;
        }
        let mut z = vec![0.0f32; self.dim];
        gemm_abt(
            &y,
            1,
            self.embed_rows,
            &self.transform,
            self.dim,
            None,
            &mut z,
        );
        let norm = z.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            let ratio = norm / (self.dim as f32).sqrt();
            z.iter_mut().for_each(|v| *v /= ratio);
        }
        z
    }
}

/// A span's speaker vector and the frames it pooled.
#[doc(hidden)]
#[derive(Clone)]
pub struct Evidence {
    pub vector: Vec<f32>,
    pub frames: usize,
    /// First and one past the last pooled frame, in frames of the stream.
    pub first: usize,
    pub end: usize,
}

/// The network over one stream of audio: the model's features at its own rate, normalised by a
/// mean that looks back only, and the frame-level rows the pooling reads.
#[doc(hidden)]
pub struct SpeakerStream<'m> {
    model: &'m SpeakerModel,
    resampler: Option<LinearResample>,
    mfcc: OnlineMfcc,
    cmn_sum: Vec<f64>,
    normalized: Vec<Vec<f32>>,
    streamer: Streamer<'m>,
    finished: bool,
    rows: VecDeque<Vec<f32>>,
    /// Stream frame of `rows[0]`.
    rows_first: usize,
}

impl<'m> SpeakerStream<'m> {
    /// Audio at `sample_rate` is resampled to the model's rate, as Kaldi's online features do
    /// when the audio is faster. Slower audio is refused, as Kaldi refuses it by default.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn new(model: &'m SpeakerModel, sample_rate: f32) -> Result<Self> {
        let (rate, want) = (
            sample_rate.round() as i64,
            model.mfcc.sample_rate.round() as i64,
        );
        if rate < want {
            return Err(err(&format!(
                "audio at {rate} Hz is below the speaker model's {want} Hz"
            )));
        }
        let resampler =
            (rate != want).then(|| LinearResample::new(rate, want, (rate.min(want) / 2) as f32, 6));
        Ok(SpeakerStream {
            model,
            resampler,
            mfcc: OnlineMfcc::new(&model.mfcc),
            cmn_sum: vec![0.0; model.mfcc.num_ceps],
            normalized: Vec::new(),
            streamer: model.net.streamer(),
            finished: false,
            rows: VecDeque::new(),
            rows_first: 0,
        })
    }

    /// Feed 16-bit samples at the stream's rate.
    pub fn accept(&mut self, samples: &[i16]) {
        if self.finished {
            return;
        }
        let x: Vec<f32> = samples.iter().map(|&s| f32::from(s)).collect();
        match self.resampler.as_mut() {
            Some(r) => {
                let y = r.resample(&x, false);
                self.mfcc.accept(&y);
            }
            None => self.mfcc.accept(&x),
        }
        self.advance();
    }

    /// The input has ended: the resampler flushes, the last frame is repeated for the right
    /// context, and every frame's row is computed.
    pub fn finish(&mut self) {
        if self.finished {
            return;
        }
        if let Some(r) = self.resampler.as_mut() {
            let y = r.resample(&[], true);
            self.mfcc.accept(&y);
        }
        self.finished = true;
        self.mfcc.input_finished();
        self.advance();
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    fn advance(&mut self) {
        while self.normalized.len() < self.mfcc.frames.len() {
            let t = self.normalized.len();
            for (s, v) in self.cmn_sum.iter_mut().zip(&self.mfcc.frames[t]) {
                *s += f64::from(*v);
            }
            if t >= CMN_WINDOW {
                for (s, v) in self
                    .cmn_sum
                    .iter_mut()
                    .zip(&self.mfcc.frames[t - CMN_WINDOW])
                {
                    *s -= f64::from(*v);
                }
            }
            let count = (t + 1).min(CMN_WINDOW) as f64;
            let row = self.mfcc.frames[t]
                .iter()
                .zip(&self.cmn_sum)
                .map(|(v, s)| (f64::from(*v) - s / count) as f32)
                .collect();
            self.normalized.push(row);
        }
        let ready = self.normalized.len();
        let (left, right) = self.model.net.context;
        let limit = if self.finished { ready + right } else { ready };
        if ready == 0 || limit.saturating_sub(right) <= self.rows_end() {
            return;
        }
        let view = InputView {
            frames: &self.normalized,
            ready,
            finished: self.finished,
            limit: limit as i64,
            first: -(left as i64),
        };
        let mut next = self.rows_first + self.rows.len();
        let (first, rows, cols) = self.streamer.advance(&view, &[]);
        for (i, row) in rows.chunks_exact(cols).enumerate() {
            let t = first + i as i64;
            if t < 0 || t as usize >= ready {
                continue;
            }
            assert_eq!(t as usize, next, "speaker rows out of order");
            self.rows.push_back(row.to_vec());
            next += 1;
        }
        while self.rows.len() > MAX_ROWS {
            self.rows.pop_front();
            self.rows_first += 1;
        }
    }

    /// The model's features so far, before and after mean normalisation.
    pub fn features(&self) -> (&[Vec<f32>], &[Vec<f32>]) {
        (&self.mfcc.frames, &self.normalized)
    }

    /// One past the last frame with a row.
    pub fn rows_end(&self) -> usize {
        self.rows_first + self.rows.len()
    }

    /// Rows before `frame` will not be pooled again.
    pub fn forget_before(&mut self, frame: usize) {
        while self.rows_first < frame && !self.rows.is_empty() {
            self.rows.pop_front();
            self.rows_first += 1;
        }
    }

    /// The vector over `frames`, stream frames in ascending order; those without a row are
    /// skipped. `None` below [`MIN_FRAMES`].
    pub fn pool(&self, frames: impl Iterator<Item = usize>) -> Option<Evidence> {
        let d = self.model.frame_dim;
        let (mut sum, mut sum_sq) = (vec![0.0f64; d], vec![0.0f64; d]);
        let (mut count, mut first, mut end) = (0usize, usize::MAX, 0usize);
        for k in frames {
            if k < self.rows_first || k >= self.rows_end() {
                continue;
            }
            for ((s, q), v) in sum
                .iter_mut()
                .zip(sum_sq.iter_mut())
                .zip(&self.rows[k - self.rows_first])
            {
                let v = f64::from(*v);
                *s += v;
                *q += v * v;
            }
            count += 1;
            first = first.min(k);
            end = k + 1;
        }
        (count >= MIN_FRAMES).then(|| Evidence {
            vector: self.model.embed(count, &sum, &sum_sq),
            frames: count,
            first,
            end,
        })
    }
}
