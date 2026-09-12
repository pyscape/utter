//! Real FFT for the MFCC front end.
//!
//! The current kernel is an iterative radix-2 complex FFT over the zero-padded real frame; the
//! split-radix real FFT written after Kaldi's `SplitRadixRealFft`, for matching summation order,
//! is the G1 refinement.
// [[rr:TD-2#Dependency policy]]
// [[rr:TD-2#Front end: MFCC]]

use std::f64::consts::PI;

pub struct RealFft {
    n: usize,
    twiddle_re: Vec<f32>,
    twiddle_im: Vec<f32>,
    bitrev: Vec<usize>,
    buf_re: Vec<f32>,
    buf_im: Vec<f32>,
}

impl RealFft {
    pub fn new(n: usize) -> RealFft {
        assert!(n.is_power_of_two() && n >= 2);
        let twiddle_re = (0..n / 2)
            .map(|k| (-2.0 * PI * k as f64 / n as f64).cos() as f32)
            .collect();
        let twiddle_im = (0..n / 2)
            .map(|k| (-2.0 * PI * k as f64 / n as f64).sin() as f32)
            .collect();
        let bits = n.trailing_zeros();
        let bitrev = (0..n)
            .map(|i| i.reverse_bits() >> (usize::BITS - bits))
            .collect();
        RealFft {
            n,
            twiddle_re,
            twiddle_im,
            bitrev,
            buf_re: vec![0.0; n],
            buf_im: vec![0.0; n],
        }
    }

    pub fn size(&self) -> usize {
        self.n
    }

    /// Power spectrum of `input` (length <= n, zero padded) into `power` (length n/2 + 1).
    pub fn power_spectrum(&mut self, input: &[f32], power: &mut [f32]) {
        let n = self.n;
        assert!(input.len() <= n && power.len() == n / 2 + 1);
        for i in 0..n {
            let src = self.bitrev[i];
            self.buf_re[i] = if src < input.len() { input[src] } else { 0.0 };
            self.buf_im[i] = 0.0;
        }
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let step = n / len;
            for start in (0..n).step_by(len) {
                for k in 0..half {
                    let wr = self.twiddle_re[k * step];
                    let wi = self.twiddle_im[k * step];
                    let a = start + k;
                    let b = a + half;
                    let tr = self.buf_re[b] * wr - self.buf_im[b] * wi;
                    let ti = self.buf_re[b] * wi + self.buf_im[b] * wr;
                    self.buf_re[b] = self.buf_re[a] - tr;
                    self.buf_im[b] = self.buf_im[a] - ti;
                    self.buf_re[a] += tr;
                    self.buf_im[a] += ti;
                }
            }
            len *= 2;
        }
        for (k, p) in power.iter_mut().enumerate() {
            *p = self.buf_re[k] * self.buf_re[k] + self.buf_im[k] * self.buf_im[k];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_naive_dft() {
        let n = 16;
        let input: Vec<f32> = (0..12).map(|i| ((i * 7) % 5) as f32 - 1.5).collect();
        let mut fft = RealFft::new(n);
        let mut power = vec![0.0; n / 2 + 1];
        fft.power_spectrum(&input, &mut power);
        for k in 0..=n / 2 {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (t, &x) in input.iter().enumerate() {
                let ang = -2.0 * PI * (k * t) as f64 / n as f64;
                re += x as f64 * ang.cos();
                im += x as f64 * ang.sin();
            }
            let want = (re * re + im * im) as f32;
            assert!(
                (power[k] - want).abs() < 1e-3 * (1.0 + want.abs()),
                "bin {k}: {} vs {want}",
                power[k]
            );
        }
    }
}
