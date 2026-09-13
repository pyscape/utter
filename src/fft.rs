//! Real FFT for the MFCC front end.
//!
//! The current kernel is an iterative radix-2 complex FFT over the zero-padded real frame; the
//! split-radix real FFT written after Kaldi's `SplitRadixRealFft`, for matching summation order,
//! is the G1 refinement.
// [[rr:TD-2#Dependency policy]]
// [[rr:TD-2#Front end: MFCC]]

// The twiddle factors are computed in f64 over bin indices and stored as f32.
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::f64::consts::PI;

pub struct RealFft {
    n: usize,
    /// The same twiddle factors a single table would hold, gathered per stage so that a
    /// stage's factors sit next to each other instead of `n / len` apart: the butterfly then
    /// reads its factor and its two rows in step, and the k loop vectorises.
    tw_re: Vec<f32>,
    tw_im: Vec<f32>,
    bitrev: Vec<usize>,
    buf_re: Vec<f32>,
    buf_im: Vec<f32>,
}

/// Every stage of the transform in place, then the power of each bin `power` asks for.
fn transform(re: &mut [f32], im: &mut [f32], tw_re: &[f32], tw_im: &[f32], power: &mut [f32]) {
    let n = re.len();
    let mut half = 1;
    while half <= n / 2 {
        let len = half * 2;
        let (wr, wi) = (
            &tw_re[half - 1..half * 2 - 1],
            &tw_im[half - 1..half * 2 - 1],
        );
        for start in (0..n).step_by(len) {
            let (ra, rest) = re[start..start + len].split_at_mut(half);
            let (ia, imb) = im[start..start + len].split_at_mut(half);
            // Every slice the loop indexes is cut to `half` first, so the bound is one the
            // compiler can see and the k loop carries no check.
            let (rb, ib, wi) = (&mut rest[..half], &mut imb[..half], &wi[..half]);
            for k in 0..half {
                let tr = rb[k] * wr[k] - ib[k] * wi[k];
                let ti = rb[k] * wi[k] + ib[k] * wr[k];
                rb[k] = ra[k] - tr;
                ib[k] = ia[k] - ti;
                ra[k] += tr;
                ia[k] += ti;
            }
        }
        half = len;
    }
    for (p, (&r, &i)) in power.iter_mut().zip(re.iter().zip(im.iter())) {
        *p = r * r + i * i;
    }
}

impl RealFft {
    pub fn new(n: usize) -> RealFft {
        assert!(n.is_power_of_two() && n >= 2);
        let twiddle_re: Vec<f32> = (0..n / 2)
            .map(|k| (-2.0 * PI * k as f64 / n as f64).cos() as f32)
            .collect();
        let twiddle_im: Vec<f32> = (0..n / 2)
            .map(|k| (-2.0 * PI * k as f64 / n as f64).sin() as f32)
            .collect();
        let (mut tw_re, mut tw_im) = (Vec::with_capacity(n - 1), Vec::with_capacity(n - 1));
        let mut half = 1;
        while half <= n / 2 {
            let step = n / (2 * half);
            for k in 0..half {
                tw_re.push(twiddle_re[k * step]);
                tw_im.push(twiddle_im[k * step]);
            }
            half *= 2;
        }
        let bits = n.trailing_zeros();
        let bitrev = (0..n)
            .map(|i| i.reverse_bits() >> (usize::BITS - bits))
            .collect();
        RealFft {
            n,
            tw_re,
            tw_im,
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
        self.buf_im.fill(0.0);
        for i in 0..n {
            let src = self.bitrev[i];
            self.buf_re[i] = if src < input.len() { input[src] } else { 0.0 };
        }
        transform(
            &mut self.buf_re,
            &mut self.buf_im,
            &self.tw_re,
            &self.tw_im,
            power,
        );
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
