//! Kaldi's `LinearResample`, the streaming resampler `OnlineGenericBaseFeature` puts in front of a
//! feature computer whose configured rate is below the audio's: a Hann-windowed sinc with
//! `num_zeros` zero crossings each side, weights formed in single precision as Kaldi forms them,
//! and output held back until the window's right half has arrived unless the input is flushed.
// [[rr:TD-14#The front end is the model's own]]

// Rates are integer hertz and sample counts fit an i64; the casts below are Kaldi's.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

pub struct LinearResample {
    rate_in: i64,
    rate_out: i64,
    cutoff: f32,
    num_zeros: i64,
    input_samples_in_unit: i64,
    output_samples_in_unit: i64,
    first_index: Vec<i64>,
    weights: Vec<Vec<f32>>,
    input_sample_offset: i64,
    output_sample_offset: i64,
    input_remainder: Vec<f32>,
}

impl LinearResample {
    /// Kaldi asserts `cutoff * 2` is at most both rates; so does this.
    pub fn new(rate_in: i64, rate_out: i64, cutoff: f32, num_zeros: i64) -> Self {
        assert!(rate_in > 0 && rate_out > 0 && cutoff > 0.0 && num_zeros > 0);
        assert!(
            f64::from(cutoff) * 2.0 <= rate_in as f64 && f64::from(cutoff) * 2.0 <= rate_out as f64
        );
        let base = gcd(rate_in, rate_out);
        let mut r = LinearResample {
            rate_in,
            rate_out,
            cutoff,
            num_zeros,
            input_samples_in_unit: rate_in / base,
            output_samples_in_unit: rate_out / base,
            first_index: Vec::new(),
            weights: Vec::new(),
            input_sample_offset: 0,
            output_sample_offset: 0,
            input_remainder: Vec::new(),
        };
        r.set_indexes_and_weights();
        r
    }

    fn window_width(&self) -> f64 {
        self.num_zeros as f64 / (2.0 * f64::from(self.cutoff))
    }

    fn filter(&self, t: f32) -> f32 {
        let (t64, fc, nz) = (f64::from(t), f64::from(self.cutoff), self.num_zeros as f64);
        let window = if t64.abs() < self.window_width() {
            (0.5 * (1.0 + (std::f64::consts::TAU * fc / nz * t64).cos())) as f32
        } else {
            0.0
        };
        let filter = if t == 0.0 {
            (2.0 * fc) as f32
        } else {
            ((std::f64::consts::TAU * fc * t64).sin() / (std::f64::consts::PI * t64)) as f32
        };
        filter * window
    }

    fn set_indexes_and_weights(&mut self) {
        let width = self.window_width();
        for i in 0..self.output_samples_in_unit {
            let output_t = i as f64 / self.rate_out as f64;
            let min_t = output_t - width;
            let max_t = output_t + width;
            let lo = (min_t * self.rate_in as f64).ceil() as i64;
            let hi = (max_t * self.rate_in as f64).floor() as i64;
            let w: Vec<f32> = (lo..=hi)
                .map(|j| {
                    let delta = j as f64 / self.rate_in as f64 - output_t;
                    self.filter(delta as f32) / self.rate_in as f32
                })
                .collect();
            self.first_index.push(lo);
            self.weights.push(w);
        }
    }

    fn num_output_samples(&self, input_num_samp: i64, flush: bool) -> i64 {
        let tick_freq = self.rate_in / gcd(self.rate_in, self.rate_out) * self.rate_out;
        let ticks_per_input = tick_freq / self.rate_in;
        let mut interval = input_num_samp * ticks_per_input;
        if !flush {
            let width = (self.window_width() as f32 * tick_freq as f32).floor() as i64;
            interval -= width;
        }
        if interval <= 0 {
            return 0;
        }
        let ticks_per_output = tick_freq / self.rate_out;
        let mut last = interval / ticks_per_output;
        if last * ticks_per_output == interval {
            last -= 1;
        }
        last + 1
    }

    /// Every output sample the input so far determines, or with `flush` every one it touches,
    /// after which the resampler starts over.
    pub fn resample(&mut self, input: &[f32], flush: bool) -> Vec<f32> {
        let n = input.len() as i64;
        let tot_in = self.input_sample_offset + n;
        let tot_out = self.num_output_samples(tot_in, flush);
        let mut out = Vec::with_capacity((tot_out - self.output_sample_offset).max(0) as usize);
        for s in self.output_sample_offset..tot_out {
            let unit = s / self.output_samples_in_unit;
            let wrapped = (s - unit * self.output_samples_in_unit) as usize;
            let first_in = self.first_index[wrapped] + unit * self.input_samples_in_unit;
            let w = &self.weights[wrapped];
            let first = first_in - self.input_sample_offset;
            let v = if first >= 0 && first + w.len() as i64 <= n {
                let at = first as usize;
                w.iter()
                    .zip(&input[at..at + w.len()])
                    .map(|(a, b)| a * b)
                    .sum()
            } else {
                let rem = self.input_remainder.len() as i64;
                let mut acc = 0.0f32;
                for (i, &wt) in w.iter().enumerate() {
                    let idx = first + i as i64;
                    if idx < 0 && rem + idx >= 0 {
                        acc += wt * self.input_remainder[(rem + idx) as usize];
                    } else if idx >= 0 && idx < n {
                        acc += wt * input[idx as usize];
                    }
                }
                acc
            };
            out.push(v);
        }
        if flush {
            self.input_sample_offset = 0;
            self.output_sample_offset = 0;
            self.input_remainder.clear();
        } else {
            self.set_remainder(input);
            self.input_sample_offset = tot_in;
            self.output_sample_offset = tot_out;
        }
        out
    }

    fn set_remainder(&mut self, input: &[f32]) {
        let old = std::mem::take(&mut self.input_remainder);
        let need =
            (self.rate_in as f64 * self.num_zeros as f64 / f64::from(self.cutoff)).ceil() as i64;
        let mut rem = vec![0.0f32; need as usize];
        let n = input.len() as i64;
        for index in -need..0 {
            let at = (index + need) as usize;
            let input_index = index + n;
            if input_index >= 0 {
                rem[at] = input[input_index as usize];
            } else if input_index + old.len() as i64 >= 0 {
                rem[at] = old[(input_index + old.len() as i64) as usize];
            }
        }
        self.input_remainder = rem;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signal(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / 16000.0;
                3000.0 * (std::f32::consts::TAU * 440.0 * t).sin()
                    + 800.0 * (std::f32::consts::TAU * 3100.0 * t).sin()
                    + 500.0 * (std::f32::consts::TAU * 6000.0 * t).sin()
            })
            .collect()
    }

    #[test]
    fn blocks_do_not_change_the_output() {
        let x = signal(4000);
        let mut whole = LinearResample::new(16000, 8000, 4000.0, 6);
        let a = whole.resample(&x, false);
        let mut parts = LinearResample::new(16000, 8000, 4000.0, 6);
        let mut b = Vec::new();
        for chunk in x.chunks(37) {
            b.extend(parts.resample(chunk, false));
        }
        assert_eq!(a, b);
    }

    #[test]
    fn output_waits_for_the_window_and_flush_completes_it() {
        // 16 kHz to 8 kHz: 25 taps around every other input sample, 12 each side.
        let mut r = LinearResample::new(16000, 8000, 4000.0, 6);
        assert_eq!(r.weights[0].len(), 25);
        assert_eq!(r.resample(&signal(12), false).len(), 0);
        let mut r = LinearResample::new(16000, 8000, 4000.0, 6);
        let held = r.resample(&signal(1000), false).len();
        assert_eq!(held, (1000 - 12 - 1) / 2 + 1);
        let mut r = LinearResample::new(16000, 8000, 4000.0, 6);
        assert_eq!(r.resample(&signal(1000), true).len(), 500);
    }

    #[test]
    fn passes_the_low_band_and_stops_the_high() {
        let rms = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt();
        let tone = |f: f32| -> Vec<f32> {
            (0..4000)
                .map(|i| 1000.0 * (std::f32::consts::TAU * f * i as f32 / 16000.0).sin())
                .collect()
        };
        let mut r = LinearResample::new(16000, 8000, 4000.0, 6);
        let low = r.resample(&tone(440.0), false);
        let mut r = LinearResample::new(16000, 8000, 4000.0, 6);
        let high = r.resample(&tone(6000.0), false);
        assert!((rms(&low[100..]) - 707.0).abs() < 20.0);
        assert!(rms(&high[100..]) < 20.0);
    }
}
