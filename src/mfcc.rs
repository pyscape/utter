// Vendored from Vosk-Rust (Apache-2.0), see third_party/Vosk-Rust/NOTICE.
//! Kaldi MFCC front end: 25 ms Povey window every 10 ms, DC removal, pre-emphasis 0.97, FFT
//! rounded to a power of two, mel filter bank, log, DCT, cepstral lifter, no energy, snip edges.
//! Input samples are in Kaldi's int16 range. Dither is applied when `dither` is non-zero, with
//! Kaldi's `RandGauss` replaced by a plain Box-Muller on a fixed-seed generator.
// [[rr:TD-2#Front end: MFCC]]

use crate::fft::RealFft;
use crate::nnet3::Mat;
use std::f32::consts::PI;

fn mel(f: f32) -> f32 {
    1127.0 * (1.0 + f / 700.0).ln()
}

#[derive(Clone, Debug)]
pub struct MfccOptions {
    pub sample_rate: f32,
    pub num_mel_bins: usize,
    pub num_ceps: usize,
    pub low_freq: f32,
    pub high_freq: f32,
    pub preemph: f32,
    pub cepstral_lifter: f32,
    pub dither: f32,
    pub frame_length_ms: f32,
    pub frame_shift_ms: f32,
}

impl Default for MfccOptions {
    fn default() -> Self {
        MfccOptions {
            sample_rate: 16000.0,
            num_mel_bins: 23,
            num_ceps: 13,
            low_freq: 20.0,
            high_freq: 0.0,
            preemph: 0.97,
            cepstral_lifter: 22.0,
            dither: 1.0,
            frame_length_ms: 25.0,
            frame_shift_ms: 10.0,
        }
    }
}

impl MfccOptions {
    /// Parse Kaldi's `--key=value` lines from `conf/mfcc.conf`; unknown keys are ignored.
    pub fn from_conf(txt: &str) -> MfccOptions {
        let mut o = MfccOptions::default();
        for line in txt.lines() {
            let line = line.trim();
            let Some(rest) = line.strip_prefix("--") else {
                continue;
            };
            let Some((k, v)) = rest.split_once('=') else {
                continue;
            };
            let f = |d: f32| v.trim().parse::<f32>().unwrap_or(d);
            match k.trim() {
                "sample-frequency" => o.sample_rate = f(o.sample_rate),
                "num-mel-bins" => o.num_mel_bins = f(o.num_mel_bins as f32) as usize,
                "num-ceps" => o.num_ceps = f(o.num_ceps as f32) as usize,
                "low-freq" => o.low_freq = f(o.low_freq),
                "high-freq" => o.high_freq = f(o.high_freq),
                "preemphasis-coefficient" => o.preemph = f(o.preemph),
                "cepstral-lifter" => o.cepstral_lifter = f(o.cepstral_lifter),
                "dither" => o.dither = f(o.dither),
                "frame-length" => o.frame_length_ms = f(o.frame_length_ms),
                "frame-shift" => o.frame_shift_ms = f(o.frame_shift_ms),
                _ => {}
            }
        }
        o
    }
}

pub struct Mfcc {
    pub frame_len: usize,
    pub frame_shift: usize,
    num_ceps: usize,
    win: Vec<f32>,
    filt: Vec<Vec<f32>>,
    dct: Vec<Vec<f32>>,
    lift: Vec<f32>,
    preemph: f32,
    dither: f32,
    fft: RealFft,
    rng: u64,
}

impl Mfcc {
    pub fn new(o: &MfccOptions) -> Mfcc {
        let sr = o.sample_rate;
        let frame_len = (o.frame_length_ms * 0.001 * sr).round() as usize;
        let frame_shift = (o.frame_shift_ms * 0.001 * sr).round() as usize;
        let mut fft_size = 1;
        while fft_size < frame_len {
            fft_size <<= 1;
        }
        let nbins = fft_size / 2 + 1;
        let a = 2.0 * PI / (frame_len as f32 - 1.0);
        let win = (0..frame_len)
            .map(|i| (0.5 - 0.5 * (a * i as f32).cos()).powf(0.85))
            .collect();

        let high = if o.high_freq > 0.0 {
            o.high_freq
        } else {
            sr / 2.0 + o.high_freq
        };
        let (mel_low, mel_high) = (mel(o.low_freq), mel(high));
        let num_mel = o.num_mel_bins;
        let delta = (mel_high - mel_low) / (num_mel as f32 + 1.0);
        let mut filt = vec![vec![0.0f32; nbins]; num_mel];
        for (m, f) in filt.iter_mut().enumerate() {
            let (l, c, r) = (
                mel_low + m as f32 * delta,
                mel_low + (m + 1) as f32 * delta,
                mel_low + (m + 2) as f32 * delta,
            );
            for (i, wgt) in f.iter_mut().enumerate() {
                let ml = mel(sr / fft_size as f32 * i as f32);
                if ml > l && ml < r {
                    *wgt = if ml <= c {
                        (ml - l) / (c - l)
                    } else {
                        (r - ml) / (r - c)
                    };
                }
            }
        }
        let mut dct = vec![vec![0.0f32; num_mel]; o.num_ceps];
        for (k, row) in dct.iter_mut().enumerate() {
            for (n, v) in row.iter_mut().enumerate() {
                *v = if k == 0 {
                    (1.0 / num_mel as f32).sqrt()
                } else {
                    (2.0 / num_mel as f32).sqrt()
                        * (PI / num_mel as f32 * k as f32 * (n as f32 + 0.5)).cos()
                };
            }
        }
        let lifter = o.cepstral_lifter;
        let lift = (0..o.num_ceps)
            .map(|k| {
                if lifter == 0.0 {
                    1.0
                } else {
                    1.0 + 0.5 * lifter * (PI * k as f32 / lifter).sin()
                }
            })
            .collect();
        Mfcc {
            frame_len,
            frame_shift,
            num_ceps: o.num_ceps,
            win,
            filt,
            dct,
            lift,
            preemph: o.preemph,
            dither: o.dither,
            fft: RealFft::new(fft_size),
            rng: 0x9E3779B97F4A7C15,
        }
    }

    pub fn dim(&self) -> usize {
        self.num_ceps
    }

    /// Number of complete frames in `num_samples` samples (snip edges).
    pub fn num_frames(&self, num_samples: usize) -> usize {
        if num_samples < self.frame_len {
            0
        } else {
            (num_samples - self.frame_len) / self.frame_shift + 1
        }
    }

    fn gauss(&mut self) -> f32 {
        // xorshift64* for two uniforms, then Box-Muller.
        let mut next = || {
            self.rng ^= self.rng >> 12;
            self.rng ^= self.rng << 25;
            self.rng ^= self.rng >> 27;
            (self.rng.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64
        };
        let u1 = next().max(1e-12);
        let u2 = next();
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }

    /// One frame of `frame_len` samples to `num_ceps` coefficients.
    pub fn compute_frame(&mut self, frame: &[f32], out: &mut [f32]) {
        assert_eq!(frame.len(), self.frame_len);
        let mut w: Vec<f32> = frame.to_vec();
        if self.dither != 0.0 {
            for x in w.iter_mut() {
                *x += self.dither * self.gauss();
            }
        }
        let mean: f32 = w.iter().sum::<f32>() / self.frame_len as f32;
        for x in w.iter_mut() {
            *x -= mean;
        }
        for i in (1..self.frame_len).rev() {
            w[i] -= self.preemph * w[i - 1];
        }
        w[0] -= self.preemph * w[0];
        for i in 0..self.frame_len {
            w[i] *= self.win[i];
        }
        let nbins = self.fft.size() / 2 + 1;
        let mut power = vec![0.0f32; nbins];
        self.fft.power_spectrum(&w, &mut power);
        let num_mel = self.filt.len();
        let mut logmel = vec![0.0f32; num_mel];
        for (m, f) in self.filt.iter().enumerate() {
            let e: f32 = f.iter().zip(&power).map(|(a, b)| a * b).sum();
            logmel[m] = e.max(f32::EPSILON).ln();
        }
        for k in 0..self.num_ceps {
            let c: f32 = self.dct[k].iter().zip(&logmel).map(|(a, b)| a * b).sum();
            out[k] = c * self.lift[k];
        }
    }

    /// All complete frames of `samples` (int16 range) as [num_frames x num_ceps].
    pub fn compute(&mut self, samples: &[f32]) -> Mat {
        let n = self.num_frames(samples.len());
        let mut out = Mat::new(n, self.num_ceps);
        for t in 0..n {
            let start = t * self.frame_shift;
            let frame = samples[start..start + self.frame_len].to_vec();
            let row = &mut out.d[t * self.num_ceps..(t + 1) * self.num_ceps];
            self.compute_frame(&frame, row);
        }
        out
    }
}
