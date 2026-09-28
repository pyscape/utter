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
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    #[cfg(target_arch = "x86_64")]
    let (z, t) = if crate::gemm::have_avx2() {
        // SAFETY: the CPU has AVX2 and FMA.
        unsafe { softmax_sums_avx2(row, max, inv) }
    } else {
        softmax_sums(row, max, inv)
    };
    #[cfg(not(target_arch = "x86_64"))]
    let (z, t) = softmax_sums(row, max, inv);
    let entropy = z.ln() - t / z;
    (1.0 - entropy / (row.len() as f64).ln()).clamp(0.0, 1.0)
}

/// Lanes the softmax's sums run in. Fixed, and with no fused multiply-add in Rust unless asked
/// for, so the AVX2 build adds in the same order as any other and gives the same bits.
const LANES: usize = 8;

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
    for (k, &x) in rest.iter().enumerate() {
        let d = (x - max) * inv;
        let e = exp_nonpositive(d);
        z[k] += e;
        t[k] += e * d;
    }
    let sum = |v: [f32; LANES]| v.iter().map(|&x| f64::from(x)).sum::<f64>();
    (sum(z), sum(t))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn softmax_sums_avx2(row: &[f32], max: f32, inv: f32) -> (f64, f64) {
    softmax_sums(row, max, inv)
}

/// `e^x` for `x <= 0` within about one ulp, in operations a compiler can run across lanes: `2^n
/// e^r` with `n` the nearest integer to `x / ln 2`, `e^r` by its Taylor series to the seventh power,
/// Cephes' split of `ln 2` for `r`. Below `-87` it is `e^-87`, which no sum of the softmax's
/// terms can tell from zero.
#[inline(always)]
fn exp_nonpositive(x: f32) -> f32 {
    // Adding 1.5 * 2^23 leaves the nearest integer in the low mantissa bits.
    const ROUND: f32 = 12_582_912.0;
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
            for (b, s) in self.open.iter().enumerate() {
                self.hops[b * FLOOR_HOPS + at] = if self.open_silent {
                    f32::NEG_INFINITY
                } else {
                    (s / n) as f32
                };
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
        // The rank-th smallest of each band by a sorted run of the rank + 1 smallest seen, far
        // cheaper at this rank than a selection over a copy of the band.
        let mut lowest: Vec<f32> = Vec::with_capacity(rank + 2);
        self.floor.clear();
        for b in 0..=self.bands {
            lowest.clear();
            for &v in &self.hops[b * FLOOR_HOPS..b * FLOOR_HOPS + n] {
                if lowest.len() <= rank || v < lowest[rank] {
                    let at = lowest.partition_point(|&x| x <= v);
                    lowest.insert(at, v);
                    lowest.truncate(rank + 1);
                }
            }
            let v = lowest[rank];
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

    /// The evidence over the utterance's feature frames in `spans`, each `[start, end)`, in
    /// order and not overlapping, clipped to the frames taken; `None` when they hold no frame.
    #[allow(clippy::cast_precision_loss)]
    pub fn evidence(&self, spans: &[(usize, usize)]) -> Option<Evidence> {
        let floor = self.floor();
        let taken = self.frames();
        let bands = self.bands;
        let mut n = 0usize;
        let mut sum = vec![0.0f64; bands];
        let mut sum_sq = vec![0.0f64; bands];
        let mut loudest: Option<(usize, f32)> = None;
        for &(a, b) in spans {
            let b = b.min(taken);
            if a >= b {
                continue;
            }
            n += b - a;
            for k in 0..bands {
                sum[k] += self.sum[b * bands + k] - self.sum[a * bands + k];
                sum_sq[k] += self.sum_sq[b * bands + k] - self.sum_sq[a * bands + k];
            }
            for (t, &l) in self.broad[a..b].iter().enumerate() {
                if loudest.is_none_or(|(_, m)| l > m) {
                    loudest = Some((a + t, l));
                }
            }
        }
        let (peak, peak_db) = loudest?;
        let nf = n as f64;
        let mut level = Vec::with_capacity(bands);
        let mut sd = Vec::with_capacity(bands);
        for k in 0..bands {
            let mean = sum[k] / nf;
            level.push(floor.map(|f| mean - f64::from(f[k])));
            sd.push((sum_sq[k] / nf - mean * mean).max(0.0).sqrt());
        }
        // The rise runs through the spans' frames that touch the loudest without a gap.
        let (lo, hi) = spans
            .iter()
            .find(|&&(a, b)| a <= peak && peak < b)
            .map(|&(a, b)| (a, b.min(taken)))
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
            level: level.into_iter().collect(),
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
        push_tenths(out, v);
    }
    out.push(']');
}

/// `v` to one decimal, as `{v:.1}` writes it, a sign on a negative zero included. The
/// formatter's exact rounding is only needed near a tie; elsewhere the nearest tenth of `v * 10`
/// is the nearest tenth of `v`, and writing it directly is several times faster.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn push_tenths(out: &mut String, v: f64) {
    use std::fmt::Write;
    let scaled = v * 10.0;
    let n = scaled.round();
    if n.is_nan() || n.abs() >= 1e9 || (scaled - n).abs() > 0.499_99 {
        let _ = write!(out, "{v:.1}");
        return;
    }
    if v.is_sign_negative() {
        out.push('-');
    }
    let n = n.abs() as u64;
    let _ = write!(out, "{}.", n / 10);
    out.push(char::from(b'0' + (n % 10) as u8));
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
            ", \"rise_start_sample\": {rise_start_sample}, \"rise_ms\": {ms:.0}, \"rise_db\": "
        );
        match self.rise_db {
            Some(db) => {
                push_tenths(out, db);
            }
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
        track.evidence(&[span]).expect("frames in the span")
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

    #[allow(clippy::cast_precision_loss)]
    fn softmax_certainty_portable(row: &[f32], scale: f32) -> f64 {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let (z, t) = softmax_sums(row, max, 1.0 / scale);
        (1.0 - (z.ln() - t / z) / (row.len() as f64).ln()).clamp(0.0, 1.0)
    }

    // [[rr:TD-17#The sound outside words]]
    #[test]
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
    #[allow(clippy::cast_precision_loss)]
    fn tenths_are_written_as_the_formatter_writes_them() {
        let mut s = 9u64;
        let mut values = vec![
            0.0,
            -0.0,
            0.05,
            -0.05,
            0.25,
            0.35,
            -0.25,
            1e-12,
            -1e-12,
            99.95,
            -99.95,
            1e12,
            f64::NAN,
        ];
        for _ in 0..200_000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 11) as f64 / (1u64 << 53) as f64;
            values.push((u - 0.5) * 300.0);
            values.push(((u - 0.5) * 3000.0).round() / 20.0);
        }
        for v in values {
            let mut out = String::new();
            push_tenths(&mut out, v);
            assert_eq!(out, format!("{v:.1}"), "{v:e}");
        }
    }

    // [[rr:TD-17#Each band is measured against its own floor]]
    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn each_floor_is_its_bands_hop_at_the_rank() {
        let bands = 3;
        let mut track = SoundTrack::new(bands);
        let mut s = 5u64;
        let mut hops: Vec<Vec<f32>> = Vec::new();
        for hop in 0..(FLOOR_HOPS + 37) {
            let mut sum = vec![0.0f64; bands + 1];
            for _ in 0..FLOOR_HOP_FRAMES {
                let l: Vec<f32> = (0..bands)
                    .map(|_| {
                        s ^= s << 13;
                        s ^= s >> 7;
                        s ^= s << 17;
                        (s >> 40) as f32 / (1u64 << 24) as f32 * 8.0 + 2.0
                    })
                    .collect();
                track.push(&l);
                let mut power = 0.0f64;
                for (b, &x) in l.iter().enumerate() {
                    sum[b] += f64::from(x) * DB_PER_NEPER;
                    power += f64::from(x).exp();
                }
                sum[bands] += power.ln() * DB_PER_NEPER;
            }
            hops.push(
                sum.iter()
                    .map(|v| (v / FLOOR_HOP_FRAMES as f64) as f32)
                    .collect(),
            );
            track.refresh_floor();
            let kept = &hops[hops.len().saturating_sub(FLOOR_HOPS)..];
            let want: Vec<f32> = (0..=bands)
                .map(|b| {
                    let mut col: Vec<f32> = kept.iter().map(|h| h[b]).collect();
                    col.sort_by(f32::total_cmp);
                    col[col.len() * 5 / 100]
                })
                .collect();
            assert_eq!(track.floor().unwrap(), want.as_slice(), "after hop {hop}");
        }
    }

    // [[rr:TD-17#Each band is measured against its own floor]]
    #[test]
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
