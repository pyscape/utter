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
use std::sync::{Mutex, MutexGuard};

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

#[cfg(test)]
thread_local! {
    /// Streams made on this thread compute every row as soon as the input allows.
    pub(crate) static EAGER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// A gap in the frames asked for wider than this starts the network afresh past it. Starting
/// afresh recomputes the lower layers' left context, about one frame's worth for this network.
const RESTART_GAP: usize = 8;

/// The network's rows over one stream, computed as spans ask for them.
struct Net<'m> {
    streamer: Streamer<'m>,
    /// The input frame the streamer's timeline starts at, and the frame it produces next;
    /// `None` before it first runs.
    origin: i64,
    next: Option<usize>,
    /// A frame's row, or `None` where no span has asked for it.
    rows: VecDeque<Option<Vec<f32>>>,
    /// Stream frame of `rows[0]`.
    rows_first: usize,
}

impl Net<'_> {
    fn has(&self, k: usize) -> bool {
        k >= self.rows_first
            && self
                .rows
                .get(k - self.rows_first)
                .is_some_and(Option::is_some)
    }

    /// Drop the rows before `frame`.
    fn forget_before(&mut self, frame: usize) {
        while self.rows_first < frame && !self.rows.is_empty() {
            self.rows.pop_front();
            self.rows_first += 1;
        }
        self.rows_first = self.rows_first.max(frame);
    }
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
    finished: bool,
    /// Behind a lock so a result, which only reads the recognizer, can compute the rows it pools.
    net: Mutex<Net<'m>>,
    /// Every row as soon as the input allows, as the network ran before rows were computed on
    /// demand: the reference the tests hold the on-demand rows to.
    #[cfg(test)]
    eager: bool,
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
            finished: false,
            net: Mutex::new(Net {
                streamer: model.net.streamer(),
                origin: 0,
                next: None,
                rows: VecDeque::new(),
                rows_first: 0,
            }),
            #[cfg(test)]
            eager: EAGER.with(std::cell::Cell::get),
        })
    }

    fn net(&self) -> MutexGuard<'_, Net<'m>> {
        self.net
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
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
        self.normalize();
    }

    /// The input has ended: the resampler flushes, and the last frame is repeated for the right
    /// context of the rows computed from here on.
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
        self.normalize();
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn normalize(&mut self) {
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
        #[cfg(test)]
        if self.eager {
            self.compute(&[(0, usize::MAX)]);
        }
    }

    /// The model's features so far, before and after mean normalisation.
    pub fn features(&self) -> (&[Vec<f32>], &[Vec<f32>]) {
        (&self.mfcc.frames, &self.normalized)
    }

    /// One past the last frame whose row the input allows: the network's right context behind
    /// the input, or the input's end once it has finished.
    pub fn rows_end(&self) -> usize {
        let ready = self.normalized.len();
        if self.finished || ready == 0 {
            ready
        } else {
            ready.saturating_sub(self.model.net.context.1)
        }
    }

    /// The first frame a span can pool: the last [`MAX_ROWS`] are kept.
    fn rows_start(&self) -> usize {
        self.rows_end().saturating_sub(MAX_ROWS)
    }

    /// Rows before `frame` will not be pooled again.
    pub fn forget_before(&mut self, frame: usize) {
        self.net().forget_before(frame);
    }

    /// Compute the rows of `spans`, stream frames ascending, as far as the input allows.
    pub fn compute(&self, spans: &[(usize, usize)]) {
        let mut net = self.net();
        self.compute_into(&mut net, spans);
    }

    fn compute_into(&self, net: &mut Net<'m>, spans: &[(usize, usize)]) {
        let (lo, hi) = (self.rows_start(), self.rows_end());
        net.forget_before(lo);
        for &(a, b) in spans {
            let (a, b) = (a.max(net.rows_first), b.min(hi));
            if a < b {
                self.fill(net, a, b);
            }
        }
    }

    /// Every row in `[a, b)`, where `b` is at most [`rows_end`](Self::rows_end).
    #[allow(clippy::cast_possible_wrap)]
    fn fill(&self, net: &mut Net<'m>, a: usize, b: usize) {
        let left = self.model.net.context.0 as i64;
        let next = net.next.unwrap_or(0);
        // Holes behind the streamer, left where it started afresh past a gap.
        let mut k = a;
        while k < b.min(next) {
            if net.has(k) {
                k += 1;
                continue;
            }
            let mut e = k + 1;
            while e < b.min(next) && !net.has(e) {
                e += 1;
            }
            let rows_first = net.rows_first;
            let mut streamer = self.model.net.streamer();
            self.run(&mut streamer, k as i64 - left, e, &mut net.rows, rows_first);
            k = e;
        }
        if b <= next && net.next.is_some() {
            return;
        }
        let start = match net.next {
            Some(n) if a <= n + RESTART_GAP => n,
            _ if a <= RESTART_GAP => 0,
            _ => a,
        };
        if net.next != Some(start) {
            net.streamer = self.model.net.streamer();
            net.origin = start as i64 - left;
        }
        let Net {
            streamer,
            origin,
            rows,
            rows_first,
            ..
        } = net;
        self.run(streamer, *origin, b, rows, *rows_first);
        net.next = Some(b);
    }

    /// Run `streamer`, whose timeline starts at input frame `origin`, until its rows reach `to`,
    /// and keep in `rows`, from stream frame `rows_first`, those not kept already.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn run(
        &self,
        streamer: &mut Streamer<'m>,
        origin: i64,
        to: usize,
        rows: &mut VecDeque<Option<Vec<f32>>>,
        rows_first: usize,
    ) {
        let ready = self.normalized.len();
        let view = InputView {
            frames: &self.normalized,
            ready,
            finished: self.finished,
            limit: (to + self.model.net.context.1) as i64,
            first: origin,
        };
        let (first, out, cols) = streamer.advance(&view, &[]);
        for (i, row) in out.chunks_exact(cols).enumerate() {
            let t = first + i as i64;
            if t < rows_first as i64 || t as usize >= ready {
                continue;
            }
            let at = t as usize - rows_first;
            if rows.len() <= at {
                rows.resize(at + 1, None);
            }
            if rows[at].is_none() {
                rows[at] = Some(row.to_vec());
            }
        }
    }

    /// The vector over the frames of `spans`, stream frames ascending and apart, that `keep`
    /// admits; frames beyond the kept rows are skipped. `None` below [`MIN_FRAMES`].
    pub fn pool(&self, spans: &[(usize, usize)], keep: impl Fn(usize) -> bool) -> Option<Evidence> {
        let mut net = self.net();
        self.compute_into(&mut net, spans);
        let hi = self.rows_end();
        let d = self.model.frame_dim;
        let (mut sum, mut sum_sq) = (vec![0.0f64; d], vec![0.0f64; d]);
        let (mut count, mut first, mut end) = (0usize, usize::MAX, 0usize);
        for &(a, b) in spans {
            for k in a.max(net.rows_first)..b.min(hi) {
                if !keep(k) {
                    continue;
                }
                let row = net.rows[k - net.rows_first]
                    .as_deref()
                    .expect("a span's rows are computed before it pools");
                for ((s, q), v) in sum.iter_mut().zip(sum_sq.iter_mut()).zip(row) {
                    let v = f64::from(*v);
                    *s += v;
                    *q += v * v;
                }
                count += 1;
                first = first.min(k);
                end = k + 1;
            }
        }
        drop(net);
        (count >= MIN_FRAMES).then(|| Evidence {
            vector: self.model.embed(count, &sum, &sum_sq),
            frames: count,
            first,
            end,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::path::PathBuf;

    /// The stock speaker model, `vosk-model-spk-0.4`, named by `UTTER_TEST_SPK_MODEL`.
    pub(crate) fn spk_model() -> Option<SpeakerModel> {
        let dir = PathBuf::from(std::env::var_os("UTTER_TEST_SPK_MODEL")?);
        SpeakerModel::open(&dir).ok()
    }

    pub(crate) fn clip(name: &str) -> Vec<i16> {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data");
        crate::wav::read_wav(&dir.join(format!("{name}.wav")))
            .unwrap()
            .samples
    }

    fn stream<'m>(model: &'m SpeakerModel, audio: &[i16], eager: bool) -> SpeakerStream<'m> {
        EAGER.with(|e| e.set(eager));
        let mut s = SpeakerStream::new(model, 16000.0).unwrap();
        EAGER.with(|e| e.set(false));
        for b in audio.chunks(640) {
            s.accept(b);
        }
        s
    }

    type Pooled = Option<(Vec<u32>, usize, usize, usize)>;

    fn pooled(s: &SpeakerStream, spans: &[(usize, usize)]) -> Pooled {
        s.pool(spans, |k| k % 5 != 0).map(|e| {
            (
                e.vector.iter().map(|v| v.to_bits()).collect(),
                e.frames,
                e.first,
                e.end,
            )
        })
    }

    #[test]
    fn rows_asked_for_in_any_order_pool_as_every_row_does() {
        let Some(m) = spk_model() else { return };
        let audio = [clip("yes"), clip("seven"), clip("no")].concat();
        let (mut eager, mut asked) = (stream(&m, &audio, true), stream(&m, &audio, false));
        let n = asked.rows_end();
        // A late span first, so the network starts past a gap; then spans reaching back into
        // the gap, across it, and to the end.
        for spans in [
            vec![(200, 260)],
            vec![(20, 300)],
            vec![(3, 40), (100, 150)],
            vec![(0, n)],
        ] {
            assert_eq!(pooled(&eager, &spans), pooled(&asked, &spans), "{spans:?}");
        }
        eager.finish();
        asked.finish();
        let n = asked.rows_end();
        assert_eq!(
            pooled(&eager, &[(n - 40, n)]),
            pooled(&asked, &[(n - 40, n)])
        );
        assert!(pooled(&asked, &[(n - 40, n)]).is_some());
    }
}
