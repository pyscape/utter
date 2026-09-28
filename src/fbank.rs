//! The filterbank front end sherpa-onnx gives NeMo speaker models, computed as
//! kaldi-native-fbank computes it: 16 kHz, 25 ms frames every 10 ms that fit wholly in the
//! input, pre-emphasis 0.97 without DC removal, a periodic Hann window, the power spectrum of a
//! 512-point FFT, 80 Slaney-normalised mel bands on the Slaney scale from 0 to 7,600 Hz, and the
//! log floored at `f32::EPSILON`. Then each band is normalised over the span.

// Frame and bin counts are small; the casts below are the reference's own int-to-float steps.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use crate::fft::RealFft;

pub const SAMPLE_RATE: u32 = 16_000;
pub const FRAME_LENGTH: usize = 400;
pub const FRAME_SHIFT: usize = 160;
pub const NUM_BANDS: usize = 80;
const FFT_SIZE: usize = 512;
const PREEMPH: f32 = 0.97;
const HIGH_FREQ_OFFSET: f32 = -400.0;
const NORM_FLOOR: f64 = 1e-5;

/// Frames in `samples` samples: every frame lies wholly inside the input.
pub fn num_frames(samples: usize) -> usize {
    if samples < FRAME_LENGTH {
        0
    } else {
        1 + (samples - FRAME_LENGTH) / FRAME_SHIFT
    }
}

fn mel_slaney(hz: f32) -> f32 {
    if hz <= 1000.0 {
        return hz * 3.0 / 200.0;
    }
    15.0 + 14.545_078_f32 * (hz / 1000.0).ln()
}

fn hz_slaney(mel: f32) -> f32 {
    if mel <= 15.0 {
        return 200.0f32 / 3.0 * mel;
    }
    1000.0 * ((mel - 15.0) * 0.068_751_775_f32).exp()
}

/// The window, the mel bank and the FFT, built once and shared by every frame.
pub struct Fbank {
    window: Vec<f32>,
    /// Per band, its first FFT bin and weights.
    bands: Vec<(usize, Vec<f32>)>,
    fft: RealFft,
    frame: Vec<f32>,
    power: Vec<f32>,
}

impl Default for Fbank {
    fn default() -> Self {
        Self::new()
    }
}

impl Fbank {
    pub fn new() -> Fbank {
        let a = std::f64::consts::TAU / FRAME_LENGTH as f64;
        let window = (0..FRAME_LENGTH)
            .map(|i| (0.5 - 0.5 * (a * i as f64).cos()) as f32)
            .collect();
        let rate = SAMPLE_RATE as f32;
        let bin_width = rate / FFT_SIZE as f32;
        let high = 0.5 * rate + HIGH_FREQ_OFFSET;
        let (mel_low, mel_high) = (mel_slaney(0.0), mel_slaney(high));
        let delta = (mel_high - mel_low) / (NUM_BANDS + 1) as f32;
        let bands = (0..NUM_BANDS)
            .map(|b| {
                let left = hz_slaney(mel_low + b as f32 * delta);
                let center = hz_slaney(mel_low + (b + 1) as f32 * delta);
                let right = hz_slaney(mel_low + (b + 2) as f32 * delta);
                let mut first = None;
                let mut weights = Vec::new();
                for i in 0..=FFT_SIZE / 2 {
                    let hz = bin_width * i as f32;
                    if hz > left && hz < right {
                        let w = if hz <= center {
                            (hz - left) / (center - left)
                        } else {
                            (right - hz) / (right - center)
                        };
                        first.get_or_insert(i);
                        weights.push(w * (2.0 / (right - left)));
                    } else if first.is_some() {
                        break;
                    }
                }
                (first.expect("every band covers a bin"), weights)
            })
            .collect();
        Fbank {
            window,
            bands,
            fft: RealFft::new(FFT_SIZE),
            frame: vec![0.0; FRAME_LENGTH],
            power: vec![0.0; FFT_SIZE / 2 + 1],
        }
    }

    /// The log mel energies of frame `f` of `samples` into `out`, `NUM_BANDS` wide.
    pub fn frame(&mut self, samples: &[f32], f: usize, out: &mut [f32]) {
        let s = &samples[f * FRAME_SHIFT..f * FRAME_SHIFT + FRAME_LENGTH];
        self.frame.copy_from_slice(s);
        let d = &mut self.frame;
        for i in (1..FRAME_LENGTH).rev() {
            d[i] -= PREEMPH * d[i - 1];
        }
        d[0] -= PREEMPH * d[0];
        for (x, w) in d.iter_mut().zip(&self.window) {
            *x *= w;
        }
        self.fft.power_spectrum(&self.frame, &mut self.power);
        for (o, (first, w)) in out.iter_mut().zip(&self.bands) {
            let mut e = 0.0f32;
            for (k, wk) in w.iter().enumerate() {
                e += wk * self.power[first + k];
            }
            *o = e.max(f32::EPSILON).ln();
        }
    }
}

/// Per-band sums over frames, accumulated in frame order in double precision, so a span
/// normalised in pieces gives the bytes of one pass.
pub struct BandStats {
    sum: Vec<f64>,
    mean: Vec<f64>,
    denom: Vec<f64>,
}

impl Default for BandStats {
    fn default() -> Self {
        Self::new()
    }
}

impl BandStats {
    pub fn new() -> BandStats {
        BandStats {
            sum: vec![0.0; NUM_BANDS],
            mean: vec![0.0; NUM_BANDS],
            denom: vec![0.0; NUM_BANDS],
        }
    }

    pub fn clear(&mut self) {
        self.sum.fill(0.0);
    }

    pub fn add_mean(&mut self, rows: &[f32]) {
        for row in rows.chunks_exact(NUM_BANDS) {
            for (s, &x) in self.sum.iter_mut().zip(row) {
                *s += f64::from(x);
            }
        }
    }

    /// Ends the mean pass over `frames` frames and starts the deviation pass.
    pub fn end_mean(&mut self, frames: usize) {
        for (m, s) in self.mean.iter_mut().zip(&mut self.sum) {
            *m = *s / frames as f64;
            *s = 0.0;
        }
    }

    pub fn add_deviation(&mut self, rows: &[f32]) {
        for row in rows.chunks_exact(NUM_BANDS) {
            for ((s, &x), m) in self.sum.iter_mut().zip(row).zip(&self.mean) {
                let d = f64::from(x) - m;
                *s += d * d;
            }
        }
    }

    /// Ends the deviation pass: each band is then divided by its population deviation plus 1e-5.
    pub fn end_deviation(&mut self, frames: usize) {
        for (d, s) in self.denom.iter_mut().zip(&self.sum) {
            *d = (*s / frames as f64).sqrt() + NORM_FLOOR;
        }
    }

    pub fn normalise(&self, rows: &mut [f32]) {
        for row in rows.chunks_exact_mut(NUM_BANDS) {
            for ((x, m), d) in row.iter_mut().zip(&self.mean).zip(&self.denom) {
                *x = ((f64::from(*x) - m) / d) as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_fit_wholly() {
        assert_eq!(num_frames(399), 0);
        assert_eq!(num_frames(400), 1);
        assert_eq!(num_frames(559), 1);
        assert_eq!(num_frames(560), 2);
        assert_eq!(num_frames(16_000), 98);
    }

    #[test]
    fn bands_are_slaney_triangles_to_7600_hz() {
        let f = Fbank::new();
        // The Slaney scale is linear below 1 kHz, so the first band starts at 0 Hz and the
        // bands are unit-area triangles: each one's weights, times the bin width, sum near 1.
        assert_eq!(f.bands[0].0, 1);
        let last = &f.bands[NUM_BANDS - 1];
        assert!((last.0 + last.1.len() - 1) as f32 * 31.25 < 7600.0);
        for (first, w) in &f.bands {
            assert!(*first > 0 && !w.is_empty() && w.iter().all(|&x| x > 0.0));
        }
        let area: f32 = f.bands[40].1.iter().sum::<f32>() * 31.25;
        assert!((area - 1.0).abs() < 0.1, "{area}");
    }

    #[test]
    // Miri perturbs ln() by a few ULP, so the exact-bytes compare below only holds natively.
    #[cfg_attr(miri, ignore)]
    fn a_zero_frame_is_the_log_floor() {
        let mut f = Fbank::new();
        let mut out = vec![0.0; NUM_BANDS];
        f.frame(&[0.0; 400], 0, &mut out);
        assert!(out.iter().all(|&x| x == f32::EPSILON.ln()));
    }

    #[test]
    fn a_constant_band_normalises_to_zero() {
        let mut rows = vec![0.0f32; 3 * NUM_BANDS];
        for (i, x) in rows.iter_mut().enumerate() {
            *x = if i % NUM_BANDS == 0 {
                -15.942385
            } else {
                (i / NUM_BANDS) as f32
            };
        }
        let mut s = BandStats::new();
        s.add_mean(&rows);
        s.end_mean(3);
        s.add_deviation(&rows);
        s.end_deviation(3);
        s.normalise(&mut rows);
        assert_eq!(rows[0], 0.0);
        let sd = (2.0f64 / 3.0).sqrt();
        assert!((f64::from(rows[1]) + 1.0 / (sd + 1e-5)).abs() < 1e-6);
    }
}
