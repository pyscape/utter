//! How the recognizer rates the sound it decodes, and what the sound outside words is like:
//! the acoustic model's certainty per output frame, and the front end's mel bands per feature
//! frame against each band's floor.
// [[rr:TD-17]]

/// The acoustic model's certainty on one output frame: one minus the entropy of the softmax over
/// the network's output, divided by the entropy of the uniform distribution over as many pdfs.
/// `row` is the row the decoder reads, `scale` the acoustic scale it was multiplied by.
// [[rr:TD-17#The rating is the acoustic model's certainty]]
#[allow(clippy::cast_precision_loss)]
pub fn certainty(row: &[f32], scale: f32) -> f64 {
    if row.len() < 2 {
        return 1.0;
    }
    let inv = 1.0 / scale;
    #[cfg(target_arch = "x86_64")]
    let (z, t) = if crate::gemm::have_avx2() {
        // SAFETY: the CPU has AVX2.
        unsafe {
            let max = avx2::row_max(row);
            avx2::softmax_sums(row, max, inv)
        }
    } else {
        softmax_sums(row, row_max(row), inv)
    };
    #[cfg(not(target_arch = "x86_64"))]
    let (z, t) = softmax_sums(row, row_max(row), inv);
    let entropy = z.ln() - t / z;
    (1.0 - entropy / (row.len() as f64).ln()).clamp(0.0, 1.0)
}

/// Lanes the softmax's sums run in. Fixed, and with no fused multiply-add in Rust unless asked
/// for, so the AVX2 build adds in the same order as any other and gives the same bits.
const LANES: usize = 8;

#[inline(always)]
fn row_max(row: &[f32]) -> f32 {
    row.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// `sum e^d` and `sum e^d d` over `d = (x - max) * inv`, the softmax's normaliser and the
/// numerator of its mean log.
#[inline(always)]
fn softmax_sums(row: &[f32], max: f32, inv: f32) -> (f64, f64) {
    let mut z = [0.0f32; LANES];
    let mut t = [0.0f32; LANES];
    let chunks = row.chunks_exact(LANES);
    let rest = chunks.remainder();
    for c in chunks {
        for k in 0..LANES {
            let d = (c[k] - max) * inv;
            let e = exp_nonpositive(d);
            z[k] += e;
            t[k] += e * d;
        }
    }
    finish_sums(z, t, rest, max, inv)
}

/// The row's last, partial chunk into the first lanes, then the lanes' sums in `f64`.
#[inline(always)]
fn finish_sums(
    mut z: [f32; LANES],
    mut t: [f32; LANES],
    rest: &[f32],
    max: f32,
    inv: f32,
) -> (f64, f64) {
    for (k, &x) in rest.iter().enumerate() {
        let d = (x - max) * inv;
        let e = exp_nonpositive(d);
        z[k] += e;
        t[k] += e * d;
    }
    let sum = |v: [f32; LANES]| v.iter().map(|&x| f64::from(x)).sum::<f64>();
    (sum(z), sum(t))
}

/// [`softmax_sums`] and [`row_max`] eight lanes to a register, each lane's operations those of
/// the portable path in its order: a division where it divides, no fused multiply-add.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{LANES, ROUND};
    use std::arch::x86_64::*;

    /// The maximum is exact in any order, but for the sign of a zero, which no sum can see.
    #[target_feature(enable = "avx2")]
    pub unsafe fn row_max(row: &[f32]) -> f32 {
        let chunks = row.chunks_exact(LANES);
        let rest = chunks.remainder();
        let mut m = _mm256_set1_ps(f32::NEG_INFINITY);
        // The running maximum second: `max_ps` returns its second operand against a NaN, so a
        // NaN in the row loses, as it does to `f32::max`.
        for c in chunks {
            // SAFETY: the chunk holds eight floats; the caller checked for AVX2.
            m = _mm256_max_ps(unsafe { _mm256_loadu_ps(c.as_ptr()) }, m);
        }
        let mut lanes = [0.0f32; LANES];
        // SAFETY: the array holds eight floats.
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), m) };
        super::row_max(&lanes).max(super::row_max(rest))
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn softmax_sums(row: &[f32], max: f32, inv: f32) -> (f64, f64) {
        let chunks = row.chunks_exact(LANES);
        let rest = chunks.remainder();
        let (maxv, invv) = (_mm256_set1_ps(max), _mm256_set1_ps(inv));
        let mut z = _mm256_setzero_ps();
        let mut t = _mm256_setzero_ps();
        for c in chunks {
            // SAFETY: the chunk holds eight floats; the caller checked for AVX2.
            let x = unsafe { _mm256_loadu_ps(c.as_ptr()) };
            let d = _mm256_mul_ps(_mm256_sub_ps(x, maxv), invv);
            let e = exp(d);
            z = _mm256_add_ps(z, e);
            t = _mm256_add_ps(t, _mm256_mul_ps(e, d));
        }
        let (mut zs, mut ts) = ([0.0f32; LANES], [0.0f32; LANES]);
        // SAFETY: each array holds eight floats.
        unsafe {
            _mm256_storeu_ps(zs.as_mut_ptr(), z);
            _mm256_storeu_ps(ts.as_mut_ptr(), t);
        }
        super::finish_sums(zs, ts, rest, max, inv)
    }

    /// [`exp_nonpositive`](super::exp_nonpositive) on eight lanes.
    #[target_feature(enable = "avx2")]
    fn exp(x: __m256) -> __m256 {
        let round = _mm256_set1_ps(ROUND);
        let x = _mm256_max_ps(x, _mm256_set1_ps(-87.0));
        let y = _mm256_add_ps(
            _mm256_mul_ps(x, _mm256_set1_ps(std::f32::consts::LOG2_E)),
            round,
        );
        let n = _mm256_sub_ps(y, round);
        let r = _mm256_add_ps(
            _mm256_sub_ps(x, _mm256_mul_ps(n, _mm256_set1_ps(0.693_359_4))),
            _mm256_mul_ps(n, _mm256_set1_ps(2.121_944_4e-4)),
        );
        let p = _mm256_add_ps(
            _mm256_set1_ps(1.0 / 720.0),
            _mm256_div_ps(r, _mm256_set1_ps(5040.0)),
        );
        let p = _mm256_add_ps(_mm256_set1_ps(1.0 / 120.0), _mm256_mul_ps(r, p));
        let p = _mm256_add_ps(_mm256_set1_ps(1.0 / 24.0), _mm256_mul_ps(r, p));
        let p = _mm256_add_ps(_mm256_set1_ps(1.0 / 6.0), _mm256_mul_ps(r, p));
        let p = _mm256_add_ps(_mm256_set1_ps(0.5), _mm256_mul_ps(r, p));
        let p = _mm256_add_ps(_mm256_set1_ps(1.0), _mm256_mul_ps(r, p));
        let p = _mm256_add_ps(_mm256_set1_ps(1.0), _mm256_mul_ps(r, p));
        let two_n = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(
            _mm256_sub_epi32(
                _mm256_castps_si256(y),
                _mm256_set1_epi32(ROUND.to_bits().cast_signed()),
            ),
            _mm256_set1_epi32(127),
        )));
        _mm256_mul_ps(p, two_n)
    }
}

/// Adding 1.5 * 2^23 leaves the nearest integer in the low mantissa bits.
const ROUND: f32 = 12_582_912.0;

/// `e^x` for `x <= 0` within about one ulp, in operations a compiler can run across lanes: `2^n
/// e^r` with `n` the nearest integer to `x / ln 2`, `e^r` by its Taylor series to the seventh power,
/// Cephes' split of `ln 2` for `r`. Below `-87` it is `e^-87`, which no sum of the softmax's
/// terms can tell from zero.
#[inline(always)]
fn exp_nonpositive(x: f32) -> f32 {
    let x = x.max(-87.0);
    let y = x * std::f32::consts::LOG2_E + ROUND;
    let n = y - ROUND;
    let r = x - n * 0.693_359_4 + n * 2.121_944_4e-4;
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (1.0 / 6.0
                    + r * (1.0 / 24.0 + r * (1.0 / 120.0 + r * (1.0 / 720.0 + r / 5040.0))))));
    let two_n = f32::from_bits(y.to_bits().wrapping_sub(ROUND.to_bits()).wrapping_add(127) << 23);
    p * two_n
}

/// Decibels per natural-log unit of power.
const DB_PER_NEPER: f64 = 10.0 / std::f64::consts::LN_10;

/// Feature frames per hop of the band floor's history.
const FLOOR_HOP_FRAMES: usize = 5;
/// Hops the band floor looks back over: ten seconds.
const FLOOR_HOPS: usize = 200;
/// How far under the loudest frame the loudest rise extends.
// [[rr:TD-17#The loudest rise]]
pub const RISE_DB: f32 = 6.0;

/// The front end's mel bands of the current utterance, one feature frame at a time, and the
/// floor of each band over the last ten seconds, across utterances.
pub struct SoundTrack {
    bands: usize,
    /// Pipeline feature frame of the utterance's first frame, and of the next frame taken.
    first: usize,
    next: usize,
    /// Running sums of each band's level in dB, and of its square, `bands` per frame from the
    /// utterance's first, with a row of zeros before it.
    sum: Vec<f64>,
    sum_sq: Vec<f64>,
    /// The frame's broadband level in dB: the power of all bands together.
    broad: Vec<f32>,
    /// Sums of the open hop's band levels and its broadband level, and its frame count.
    open: Vec<f64>,
    open_n: usize,
    open_silent: bool,
    /// Mean levels of the last `FLOOR_HOPS` hops, `FLOOR_HOPS` per band and then as many for the
    /// broadband, each band's a ring written at `hops_taken % FLOOR_HOPS`; a hop of digital
    /// silence has no level and sorts below every hop that has one.
    hops: Vec<f32>,
    /// The same levels, each band's in ascending order, as many in use as the ring holds.
    sorted: Vec<f32>,
    /// Hops taken since the track began, the floor was last computed at, and the floor.
    hops_taken: u64,
    floor_at: Option<u64>,
    floor: Vec<f32>,
}

impl SoundTrack {
    pub fn new(bands: usize) -> Self {
        SoundTrack {
            bands,
            first: 0,
            next: 0,
            sum: vec![0.0; bands],
            sum_sq: vec![0.0; bands],
            broad: Vec::new(),
            open: vec![0.0; bands + 1],
            open_n: 0,
            open_silent: true,
            hops: vec![0.0; (bands + 1) * FLOOR_HOPS],
            sorted: vec![0.0; (bands + 1) * FLOOR_HOPS],
            hops_taken: 0,
            floor_at: None,
            floor: Vec::new(),
        }
    }

    pub fn bands(&self) -> usize {
        self.bands
    }

    /// The pipeline feature frame the next frame taken must be.
    pub fn next_frame(&self) -> usize {
        self.next
    }

    /// Begin an utterance at pipeline feature frame `first`, which may be a new pipeline's `0`;
    /// the floor's history carries on.
    pub fn start_utterance(&mut self, first: usize) {
        self.first = first;
        self.next = first;
        self.sum.truncate(self.bands);
        self.sum_sq.truncate(self.bands);
        self.broad.clear();
    }

    /// Take the next feature frame's log mel energies, natural log of power.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn push(&mut self, logmel: &[f32]) {
        debug_assert_eq!(logmel.len(), self.bands);
        let at = self.sum.len() - self.bands;
        let mut power = 0.0f64;
        for (b, &l) in logmel.iter().enumerate() {
            let db = f64::from(l) * DB_PER_NEPER;
            self.sum.push(self.sum[at + b] + db);
            self.sum_sq.push(self.sum_sq[at + b] + db * db);
            self.open[b] += db;
            power += f64::from(l).exp();
        }
        // The front end's log of a band with no power: digital silence.
        let silent = f32::EPSILON.ln();
        self.open_silent &= logmel.iter().all(|&l| l <= silent);
        let broad = power.ln() * DB_PER_NEPER;
        self.broad.push(broad as f32);
        self.open[self.bands] += broad;
        self.open_n += 1;
        if self.open_n == FLOOR_HOP_FRAMES {
            let n = self.open_n as f64;
            let at = (self.hops_taken % FLOOR_HOPS as u64) as usize;
            let held = self.hops_taken.min(FLOOR_HOPS as u64) as usize;
            for (b, s) in self.open.iter().enumerate() {
                let v = if self.open_silent {
                    f32::NEG_INFINITY
                } else {
                    (s / n) as f32
                };
                let ring = b * FLOOR_HOPS;
                let run = &mut self.sorted[ring..ring + FLOOR_HOPS];
                if held == FLOOR_HOPS {
                    replace_sorted(run, self.hops[ring + at], v);
                } else {
                    insert_sorted(&mut run[..=held], v);
                }
                self.hops[ring + at] = v;
            }
            self.open.fill(0.0);
            self.open_n = 0;
            self.open_silent = true;
            self.hops_taken += 1;
        }
        self.next += 1;
    }

    /// Frames of the current utterance taken so far.
    pub fn frames(&self) -> usize {
        self.broad.len()
    }

    /// Bring [`floor`](Self::floor) up to the hops taken.
    pub fn refresh_floor(&mut self) {
        if self.hops_taken == 0 || self.floor_at == Some(self.hops_taken) {
            return;
        }
        #[allow(clippy::cast_possible_truncation)]
        let n = self.hops_taken.min(FLOOR_HOPS as u64) as usize;
        let rank = n * 5 / 100;
        self.floor.clear();
        for b in 0..=self.bands {
            let v = self.sorted[b * FLOOR_HOPS + rank];
            if v == f32::NEG_INFINITY {
                self.floor.clear();
                break;
            }
            self.floor.push(v);
        }
        self.floor_at = Some(self.hops_taken);
    }

    /// Which floor [`floor`](Self::floor) is: it changes whenever the floor may have.
    pub fn floor_version(&self) -> Option<u64> {
        self.floor_at
    }

    /// Each band's floor and then the broadband floor, in dB: the level of the hop at nearest
    /// rank `n * 5 / 100` over the hops of the last ten seconds, ascending, as the room's floor
    /// is ranked. `None` before a hop has closed, and when a band's rank falls on digital
    /// silence. Current as of [`refresh_floor`](Self::refresh_floor).
    // [[rr:TD-17#Each band is measured against its own floor]]
    pub fn floor(&self) -> Option<&[f32]> {
        (!self.floor.is_empty()).then_some(self.floor.as_slice())
    }

    /// The evidence over the utterance's feature frames in `spans`, each `[start, end)` in units
    /// of `unit` feature frames, in order and not overlapping, clipped to the frames taken; `None`
    /// when they hold no frame.
    #[allow(clippy::cast_precision_loss)]
    pub fn evidence(&self, spans: &[(usize, usize)], unit: usize) -> Option<Evidence> {
        let floor = self.floor();
        let taken = self.frames();
        let bands = self.bands;
        let frames = || {
            spans
                .iter()
                .map(move |&(a, b)| (a * unit, (b * unit).min(taken)))
                .filter(|&(a, b)| a < b)
        };
        let mut n = 0usize;
        let mut loudest: Option<(usize, f32)> = None;
        for (a, b) in frames() {
            n += b - a;
            for (t, &l) in self.broad[a..b].iter().enumerate() {
                if loudest.is_none_or(|(_, m)| l > m) {
                    loudest = Some((a + t, l));
                }
            }
        }
        let (peak, peak_db) = loudest?;
        let nf = n as f64;
        let mut level = floor.map(|_| Vec::with_capacity(bands));
        let mut sd = Vec::with_capacity(bands);
        for k in 0..bands {
            let (mut sum, mut sum_sq) = (0.0f64, 0.0f64);
            for (a, b) in frames() {
                sum += self.sum[b * bands + k] - self.sum[a * bands + k];
                sum_sq += self.sum_sq[b * bands + k] - self.sum_sq[a * bands + k];
            }
            let mean = sum / nf;
            if let (Some(level), Some(f)) = (level.as_mut(), floor) {
                level.push(mean - f64::from(f[k]));
            }
            sd.push((sum_sq / nf - mean * mean).max(0.0).sqrt());
        }
        // The rise runs through the spans' frames that touch the loudest without a gap.
        let (lo, hi) = frames()
            .find(|&(a, b)| a <= peak && peak < b)
            .expect("the loudest frame lies in a span");
        let within = |t: usize| self.broad[t] >= peak_db - RISE_DB;
        let mut start = peak;
        while start > lo && within(start - 1) {
            start -= 1;
        }
        let mut end = peak + 1;
        while end < hi && within(end) {
            end += 1;
        }
        Some(Evidence {
            level,
            sd,
            rise_start: start,
            rise_frames: end - start,
            rise_db: floor.map(|f| f64::from(peak_db) - f64::from(f[bands])),
        })
    }

    /// The pipeline feature frame of the utterance's frame `t`.
    pub fn pipeline_frame(&self, t: usize) -> usize {
        self.first + t
    }
}

/// Put `new` in place of `old`, which `run` holds, keeping `run` ascending.
fn replace_sorted(run: &mut [f32], old: f32, new: f32) {
    let i = run.partition_point(|x| x.total_cmp(&old).is_lt());
    let j = run.partition_point(|x| x.total_cmp(&new).is_lt());
    debug_assert_eq!(run[i].to_bits(), old.to_bits());
    if j <= i {
        run.copy_within(j..i, j + 1);
        run[j] = new;
    } else {
        run.copy_within(i + 1..j, i);
        run[j - 1] = new;
    }
}

/// Add `new` to the ascending `run[..run.len() - 1]`, keeping it ascending.
fn insert_sorted(run: &mut [f32], new: f32) {
    let held = run.len() - 1;
    let j = run[..held].partition_point(|x| x.total_cmp(&new).is_lt());
    run.copy_within(j..held, j + 1);
    run[j] = new;
}

/// What the sound over a set of frames is like.
// [[rr:TD-17#The sound outside words]]
pub struct Evidence {
    /// Each band's mean level in dB over its floor; `None` before the floor exists.
    pub level: Option<Vec<f64>>,
    /// Each band's standard deviation in dB across the frames.
    pub sd: Vec<f64>,
    /// The loudest rise's first frame, in the utterance's feature frames, and its length.
    pub rise_start: usize,
    pub rise_frames: usize,
    /// The loudest frame's broadband level over the broadband floor.
    pub rise_db: Option<f64>,
}

fn push_db_array(out: &mut String, values: &[f64]) {
    out.push('[');
    for (i, &v) in values.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        push_fixed(out, v, 1);
    }
    out.push(']');
}

/// `v` to `places` decimals, at most three, as `{v:.places$}` writes it, a sign on a negative
/// zero included. The formatter's exact rounding is only needed near a tie; elsewhere the
/// integer nearest `v * 10^places` is the formatter's digits, and writing it directly is several
/// times faster.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn push_fixed(out: &mut String, v: f64, places: usize) {
    use std::fmt::Write;
    const SCALE: [f64; 4] = [1.0, 10.0, 100.0, 1000.0];
    const UNIT: [u64; 4] = [1, 10, 100, 1000];
    let scaled = v * SCALE[places];
    let n = scaled.round();
    if n.is_nan() || n.abs() >= 1e9 || (scaled - n).abs() > 0.499_99 {
        let _ = write!(out, "{v:.places$}");
        return;
    }
    if v.is_sign_negative() {
        out.push('-');
    }
    let n = n.abs() as u64;
    let _ = write!(out, "{}", n / UNIT[places]);
    if places > 0 {
        out.push('.');
        for d in (0..places).rev() {
            out.push(char::from(b'0' + (n / UNIT[d] % 10) as u8));
        }
    }
}

impl Evidence {
    /// The evidence's keys, each preceded by `, `; `rise_start_sample` is the sample of the
    /// rise's first frame, `frame_ms` a feature frame's shift.
    pub fn write(&self, out: &mut String, rise_start_sample: u64, frame_ms: f64) {
        use std::fmt::Write;
        out.push_str(", \"band_db\": ");
        match &self.level {
            Some(l) => push_db_array(out, l),
            None => out.push_str("null"),
        }
        out.push_str(", \"band_sd_db\": ");
        push_db_array(out, &self.sd);
        #[allow(clippy::cast_precision_loss)]
        let ms = self.rise_frames as f64 * frame_ms;
        let _ = write!(
            out,
            ", \"rise_start_sample\": {rise_start_sample}, \"rise_ms\": "
        );
        push_fixed(out, ms, 0);
        out.push_str(", \"rise_db\": ");
        match self.rise_db {
            Some(db) => push_fixed(out, db, 1),
            None => out.push_str("null"),
        }
    }

    /// The keys with no frames to measure.
    pub fn write_none(out: &mut String) {
        out.push_str(
            ", \"band_db\": null, \"band_sd_db\": null, \"rise_start_sample\": null, \
             \"rise_ms\": null, \"rise_db\": null",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mfcc::{Mfcc, MfccOptions};

    const RATE: f32 = 16000.0;

    /// White noise of standard deviation `sd`, int16 range, the same every run.
    #[allow(clippy::cast_precision_loss)]
    fn noise(n: usize, sd: f32, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                let mut u = 0.0f32;
                for _ in 0..12 {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    u += (s >> 40) as f32 / (1u64 << 24) as f32;
                }
                (u - 6.0) * sd
            })
            .collect()
    }

    /// The stock small models' front end, dither off as the recognizer runs it.
    fn front_end() -> Mfcc {
        let mut o = MfccOptions::from_conf(
            "--use-energy=false\n--num-mel-bins=40\n--num-ceps=40\n--low-freq=20\n--high-freq=7600\n",
        );
        o.dither = 0.0;
        Mfcc::new(&o)
    }

    /// Push every frame of `samples` onto `track`, returning the utterance frames they took.
    fn push(track: &mut SoundTrack, mfcc: &mut Mfcc, samples: &[f32]) -> (usize, usize) {
        let from = track.frames();
        let mut row = vec![0.0; mfcc.dim()];
        for t in 0..mfcc.num_frames(samples.len()) {
            mfcc.compute_frame(&samples[t * 160..t * 160 + 400], &mut row);
            track.push(mfcc.last_logmel());
        }
        (from, track.frames())
    }

    /// Samples in the window of a result read at each decoder advance: eight 30 ms frames.
    const WINDOW: usize = 3840;

    /// Ten seconds of a quiet room, then a result's window of `event` over the same room.
    fn over_room(event: impl Fn(usize) -> f32) -> Evidence {
        let mut mfcc = front_end();
        let mut track = SoundTrack::new(mfcc.num_bins());
        let room = noise(10 * 16000, 30.0, 7);
        push(&mut track, &mut mfcc, &room);
        track.refresh_floor();
        let mut window = noise(WINDOW, 30.0, 11);
        for (i, s) in window.iter_mut().enumerate() {
            *s += event(i);
        }
        let span = push(&mut track, &mut mfcc, &window);
        track.evidence(&[span], 1).expect("frames in the span")
    }

    #[allow(clippy::cast_precision_loss)]
    fn rise_ms(e: &Evidence) -> f64 {
        e.rise_frames as f64 * 10.0
    }

    fn max(v: &[f64]) -> f64 {
        v.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }

    fn min(v: &[f64]) -> f64 {
        v.iter().copied().fold(f64::INFINITY, f64::min)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    #[allow(clippy::cast_precision_loss)]
    fn the_exponential_is_within_two_ulps() {
        for i in 0..=870_000 {
            let x = -(i as f32) * 1e-4;
            let want = f64::from(x).exp();
            let got = f64::from(exp_nonpositive(x));
            assert!(
                (got - want).abs() <= 2.0 * f64::from(f32::EPSILON) * want,
                "{x}: {got} for {want}"
            );
        }
    }

    #[test]
    // Miri perturbs float intrinsics by a few ULP, so the exact-bytes compare below only holds natively.
    #[cfg_attr(miri, ignore)]
    #[allow(clippy::cast_precision_loss)]
    fn the_certainty_is_the_normalised_entropy() {
        let mut s = 3u64;
        for len in [2, 7, 8, 9, 2192] {
            let row: Vec<f32> = (0..len)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    (s >> 40) as f32 / (1u64 << 24) as f32 * 30.0 - 25.0
                })
                .collect();
            let scale = 0.7;
            let max = row
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, |m: f64, x| m.max(f64::from(x)));
            let p: Vec<f64> = row
                .iter()
                .map(|&x| ((f64::from(x) - max) / f64::from(scale)).exp())
                .collect();
            let z: f64 = p.iter().sum();
            let h: f64 = -p.iter().map(|&q| q / z * (q / z).ln()).sum::<f64>();
            let want = 1.0 - h / (len as f64).ln();
            let got = certainty(&row, scale);
            assert!((got - want).abs() < 1e-5, "{len}: {got} for {want}");
            assert_eq!(
                got,
                softmax_certainty_portable(&row, scale),
                "every build adds alike"
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    #[cfg_attr(miri, ignore)]
    #[allow(clippy::cast_precision_loss)]
    fn the_avx2_sums_are_the_portable_sums_bit_for_bit() {
        if !crate::gemm::have_avx2() {
            return;
        }
        let mut s = 17u64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32
        };
        for (len, spread) in (1..=41)
            .map(|n| (n, 30.0))
            .chain([(2192, 30.0), (2192, 300.0)])
        {
            for _ in 0..50 {
                let row: Vec<f32> = (0..len).map(|_| next() * spread - spread).collect();
                for scale in [0.1, 0.7, 1.0] {
                    let max = row_max(&row);
                    let want = softmax_sums(&row, max, 1.0 / scale);
                    // SAFETY: the CPU has AVX2.
                    let (got_max, got) = unsafe {
                        (
                            avx2::row_max(&row),
                            avx2::softmax_sums(&row, max, 1.0 / scale),
                        )
                    };
                    assert_eq!(got_max.to_bits(), max.to_bits(), "{len}");
                    assert_eq!(got.0.to_bits(), want.0.to_bits(), "{len} {scale}");
                    assert_eq!(got.1.to_bits(), want.1.to_bits(), "{len} {scale}");
                }
            }
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn softmax_certainty_portable(row: &[f32], scale: f32) -> f64 {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let (z, t) = softmax_sums(row, max, 1.0 / scale);
        (1.0 - (z.ln() - t / z) / (row.len() as f64).ln()).clamp(0.0, 1.0)
    }

    // [[rr:TD-17#The sound outside words]]
    #[test]
    #[cfg_attr(miri, ignore)]
    #[allow(clippy::cast_precision_loss)]
    fn hiss_hum_and_click_are_told_apart() {
        let hiss_noise = noise(WINDOW, 300.0, 13);
        let hiss = over_room(|i| hiss_noise[i]);
        let hum =
            over_room(|i| 3000.0 * (2.0 * std::f32::consts::PI * 120.0 * i as f32 / RATE).sin());
        let click = over_room(|i| if i == WINDOW / 2 { 20000.0 } else { 0.0 });
        let level = |e: &Evidence| e.level.clone().expect("a floor");

        // Hiss: every band well over its floor, and loud for most of the window.
        let l = level(&hiss);
        assert!(min(&l) > 12.0, "hiss bands {l:?}");
        assert!(rise_ms(&hiss) >= 150.0, "hiss rise {} ms", rise_ms(&hiss));

        // Hum: the lowest bands far over their floors and steady there; the upper half untouched.
        let l = level(&hum);
        let loudest = l
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert!(loudest < 6 && l[loudest] > 30.0, "hum bands {l:?}");
        assert!(max(&l[20..]) < 6.0, "hum bands {l:?}");
        assert!(hum.sd[loudest] < 1.5, "hum steadiness {:?}", hum.sd);

        // Click: a rise of a few frames far over the floor, and the least steady bands of the three.
        assert!(rise_ms(&click) <= 30.0, "click rise {} ms", rise_ms(&click));
        assert!(
            click.rise_db.unwrap() > 20.0,
            "click rise {:?} dB",
            click.rise_db
        );
        assert!(
            max(&click.sd) > max(&hiss.sd) + 3.0,
            "click {:?} hiss {:?}",
            click.sd,
            hiss.sd
        );
        assert!(
            min(&level(&click)) < min(&level(&hiss)) - 6.0,
            "a click raises a band's mean less than a hiss does"
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    #[allow(clippy::cast_precision_loss)]
    fn fixed_decimals_are_written_as_the_formatter_writes_them() {
        let mut s = 9u64;
        let mut values = vec![
            0.0,
            -0.0,
            0.05,
            -0.05,
            0.25,
            0.35,
            -0.25,
            0.0005,
            0.0625,
            0.9995,
            1.0,
            1e-12,
            -1e-12,
            99.95,
            -99.95,
            1e12,
            f64::NAN,
            f64::INFINITY,
        ];
        for _ in 0..50_000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 11) as f64 / (1u64 << 53) as f64;
            values.push(u);
            values.push((u - 0.5) * 300.0);
            // Halfway between two values at each number of places, and the doubles either side.
            for halves in [2.0, 20.0, 200.0, 2000.0] {
                let tie = ((u - 0.5) * 3000.0).round() / halves;
                values.push(tie);
                values.push(f64::from_bits(tie.to_bits() + 1));
                values.push(f64::from_bits(tie.to_bits().wrapping_sub(1)));
            }
        }
        for places in 0..=3 {
            for &v in &values {
                let mut out = String::new();
                push_fixed(&mut out, v, places);
                assert_eq!(out, format!("{v:.places$}"), "{v:e} to {places}");
            }
        }
    }

    // [[rr:TD-17#Each band is measured against its own floor]]
    #[test]
    // Miri perturbs ln() by a few ULP on each call, so a silent frame's level no longer equals the
    // one the track takes for digital silence.
    #[cfg_attr(miri, ignore)]
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn each_floor_is_its_bands_hop_at_the_rank() {
        let bands = 3;
        let mut track = SoundTrack::new(bands);
        let mut s = 5u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32
        };
        let silent = f32::EPSILON.ln();
        let mut hops: Vec<Vec<f32>> = Vec::new();
        for hop in 0..(3 * FLOOR_HOPS + 37) {
            // Single hops of digital silence, fewer than the rank, and a run of more.
            let quiet = hop % 29 == 0 || (250..263).contains(&hop);
            let mut sum = vec![0.0f64; bands + 1];
            for _ in 0..FLOOR_HOP_FRAMES {
                // A band at any level, a band at eight levels, and a band at one.
                let l = if quiet {
                    vec![silent; bands]
                } else {
                    vec![next() * 8.0 + 2.0, (next() * 8.0).floor() * 0.5 + 2.0, 3.0]
                };
                track.push(&l);
                let mut power = 0.0f64;
                for (b, &x) in l.iter().enumerate() {
                    sum[b] += f64::from(x) * DB_PER_NEPER;
                    power += f64::from(x).exp();
                }
                sum[bands] += power.ln() * DB_PER_NEPER;
            }
            hops.push(if quiet {
                vec![f32::NEG_INFINITY; bands + 1]
            } else {
                sum.iter()
                    .map(|v| (v / FLOOR_HOP_FRAMES as f64) as f32)
                    .collect()
            });
            track.refresh_floor();
            let kept = &hops[hops.len().saturating_sub(FLOOR_HOPS)..];
            let want: Vec<f32> = (0..=bands)
                .map(|b| {
                    let mut col: Vec<f32> = kept.iter().map(|h| h[b]).collect();
                    col.sort_by(f32::total_cmp);
                    col[col.len() * 5 / 100]
                })
                .collect();
            let want = (!want.contains(&f32::NEG_INFINITY)).then_some(want);
            assert_eq!(track.floor(), want.as_deref(), "after hop {hop}");
        }
    }

    // [[rr:TD-17#Each band is measured against its own floor]]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn digital_silence_has_no_floor() {
        let mut mfcc = front_end();
        let mut track = SoundTrack::new(mfcc.num_bins());
        push(&mut track, &mut mfcc, &vec![0.0; 16000]);
        track.refresh_floor();
        assert!(track.floor().is_none());
        push(&mut track, &mut mfcc, &noise(10 * 16000, 30.0, 7));
        track.refresh_floor();
        assert!(
            track.floor().is_some(),
            "ten seconds of room push the silence out of rank"
        );
    }
}
