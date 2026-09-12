//! Single-precision matrix products for the network: `C[m x n] = A[m x k] * B[n x k]^T`, both
//! operands contiguous along k, which is how Kaldi stores affine parameters (rows are outputs).
//!
//! Own kernel: an 8-wide FMA tile over four B rows at a time, with an explicit AVX2+FMA path
//! behind runtime detection and a portable path the compiler vectorizes for the rest.
// [[rr:TD-2#Dependency policy]]

#[inline(always)]
fn dot4_portable(a: &[f32], b0: &[f32], b1: &[f32], b2: &[f32], b3: &[f32]) -> [f32; 4] {
    let k = a.len();
    let mut acc = [[0.0f32; 8]; 4];
    let mut i = 0;
    while i + 8 <= k {
        for l in 0..8 {
            let x = a[i + l];
            acc[0][l] += x * b0[i + l];
            acc[1][l] += x * b1[i + l];
            acc[2][l] += x * b2[i + l];
            acc[3][l] += x * b3[i + l];
        }
        i += 8;
    }
    let mut out = [0.0f32; 4];
    for j in 0..4 {
        out[j] = acc[j].iter().sum();
    }
    while i < k {
        out[0] += a[i] * b0[i];
        out[1] += a[i] * b1[i];
        out[2] += a[i] * b2[i];
        out[3] += a[i] * b3[i];
        i += 1;
    }
    out
}

#[inline(always)]
fn dot1_portable(a: &[f32], b: &[f32]) -> f32 {
    let k = a.len();
    let mut acc = [0.0f32; 8];
    let mut i = 0;
    while i + 8 <= k {
        for l in 0..8 {
            acc[l] += a[i + l] * b[i + l];
        }
        i += 8;
    }
    let mut s: f32 = acc.iter().sum();
    while i < k {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot4(a: &[f32], b0: &[f32], b1: &[f32], b2: &[f32], b3: &[f32]) -> [f32; 4] {
        let k = a.len();
        let mut c0 = _mm256_setzero_ps();
        let mut c1 = _mm256_setzero_ps();
        let mut c2 = _mm256_setzero_ps();
        let mut c3 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= k {
            let x = _mm256_loadu_ps(a.as_ptr().add(i));
            c0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(b0.as_ptr().add(i)), c0);
            c1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(b1.as_ptr().add(i)), c1);
            c2 = _mm256_fmadd_ps(x, _mm256_loadu_ps(b2.as_ptr().add(i)), c2);
            c3 = _mm256_fmadd_ps(x, _mm256_loadu_ps(b3.as_ptr().add(i)), c3);
            i += 8;
        }
        let mut out = [hsum(c0), hsum(c1), hsum(c2), hsum(c3)];
        while i < k {
            out[0] += a[i] * b0[i];
            out[1] += a[i] * b1[i];
            out[2] += a[i] * b2[i];
            out[3] += a[i] * b3[i];
            i += 1;
        }
        out
    }

    /// Three rows of A against four rows of B: twelve accumulators, seven loads per step.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot3x4(a: [&[f32]; 3], b: [&[f32]; 4], out: &mut [[f32; 4]; 3]) {
        let k = a[0].len();
        let mut acc = [[_mm256_setzero_ps(); 4]; 3];
        let mut i = 0;
        while i + 8 <= k {
            let b0 = _mm256_loadu_ps(b[0].as_ptr().add(i));
            let b1 = _mm256_loadu_ps(b[1].as_ptr().add(i));
            let b2 = _mm256_loadu_ps(b[2].as_ptr().add(i));
            let b3 = _mm256_loadu_ps(b[3].as_ptr().add(i));
            for r in 0..3 {
                let x = _mm256_loadu_ps(a[r].as_ptr().add(i));
                acc[r][0] = _mm256_fmadd_ps(x, b0, acc[r][0]);
                acc[r][1] = _mm256_fmadd_ps(x, b1, acc[r][1]);
                acc[r][2] = _mm256_fmadd_ps(x, b2, acc[r][2]);
                acc[r][3] = _mm256_fmadd_ps(x, b3, acc[r][3]);
            }
            i += 8;
        }
        for r in 0..3 {
            for c in 0..4 {
                out[r][c] = hsum(acc[r][c]);
            }
        }
        while i < k {
            for r in 0..3 {
                for c in 0..4 {
                    out[r][c] += a[r][i] * b[c][i];
                }
            }
            i += 1;
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot1(a: &[f32], b: &[f32]) -> f32 {
        let k = a.len();
        let mut c = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= k {
            c = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i)),
                _mm256_loadu_ps(b.as_ptr().add(i)),
                c,
            );
            i += 8;
        }
        let mut s = hsum(c);
        while i < k {
            s += a[i] * b[i];
            i += 1;
        }
        s
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_hadd_ps(s, s);
        let s = _mm_hadd_ps(s, s);
        _mm_cvtss_f32(s)
    }
}

fn have_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use std::sync::OnceLock;
        static FLAG: OnceLock<bool> = OnceLock::new();
        *FLAG.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// `c[m x n] = a[m x k] * b[n x k]^T`, then `+ bias` per column when given.
pub fn gemm_abt(
    a: &[f32],
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    bias: Option<&[f32]>,
    c: &mut [f32],
) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert_eq!(c.len(), m * n);
    let avx = have_avx2();
    // B rows in blocks that stay in L2 while every A row streams past.
    let block_rows = (256 * 1024 / (k.max(1) * 4)).clamp(4, 512) & !3;
    let mut j0 = 0;
    while j0 < n {
        let j1 = (j0 + block_rows).min(n);
        let mut i = 0;
        #[cfg(target_arch = "x86_64")]
        if avx {
            // row triples of A against row quads of B
            while i + 3 <= m {
                let ra = [
                    &a[i * k..(i + 1) * k],
                    &a[(i + 1) * k..(i + 2) * k],
                    &a[(i + 2) * k..(i + 3) * k],
                ];
                let mut j = j0;
                while j + 4 <= j1 {
                    let rb = [
                        &b[j * k..(j + 1) * k],
                        &b[(j + 1) * k..(j + 2) * k],
                        &b[(j + 2) * k..(j + 3) * k],
                        &b[(j + 3) * k..(j + 4) * k],
                    ];
                    let mut out = [[0.0f32; 4]; 3];
                    unsafe { avx2::dot3x4(ra, rb, &mut out) };
                    for r in 0..3 {
                        c[(i + r) * n + j..(i + r) * n + j + 4].copy_from_slice(&out[r]);
                    }
                    j += 4;
                }
                while j < j1 {
                    let br = &b[j * k..(j + 1) * k];
                    for r in 0..3 {
                        c[(i + r) * n + j] = unsafe { avx2::dot1(ra[r], br) };
                    }
                    j += 1;
                }
                i += 3;
            }
        }
        while i < m {
            let ar = &a[i * k..(i + 1) * k];
            let cr = &mut c[i * n..(i + 1) * n];
            let mut j = j0;
            while j + 4 <= j1 {
                let (b0, b1, b2, b3) = (
                    &b[j * k..(j + 1) * k],
                    &b[(j + 1) * k..(j + 2) * k],
                    &b[(j + 2) * k..(j + 3) * k],
                    &b[(j + 3) * k..(j + 4) * k],
                );
                let d = if avx {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        avx2::dot4(ar, b0, b1, b2, b3)
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    dot4_portable(ar, b0, b1, b2, b3)
                } else {
                    dot4_portable(ar, b0, b1, b2, b3)
                };
                cr[j..j + 4].copy_from_slice(&d);
                j += 4;
            }
            while j < j1 {
                let br = &b[j * k..(j + 1) * k];
                cr[j] = if avx {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        avx2::dot1(ar, br)
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    dot1_portable(ar, br)
                } else {
                    dot1_portable(ar, br)
                };
                j += 1;
            }
            i += 1;
        }
        j0 = j1;
    }
    if let Some(bias) = bias {
        assert_eq!(bias.len(), n);
        for i in 0..m {
            let cr = &mut c[i * n..(i + 1) * n];
            for j in 0..n {
                cr[j] += bias[j];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_naive() {
        let (m, k, n) = (5, 37, 11);
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 31) % 17) as f32 * 0.1 - 0.7)
            .collect();
        let b: Vec<f32> = (0..n * k)
            .map(|i| ((i * 13) % 19) as f32 * 0.05 - 0.4)
            .collect();
        let bias: Vec<f32> = (0..n).map(|j| j as f32).collect();
        let mut c = vec![0.0; m * n];
        gemm_abt(&a, m, k, &b, n, Some(&bias), &mut c);
        for i in 0..m {
            for j in 0..n {
                let want: f32 = (0..k).map(|l| a[i * k + l] * b[j * k + l]).sum::<f32>() + bias[j];
                assert!(
                    (c[i * n + j] - want).abs() < 1e-3,
                    "{i},{j}: {} vs {want}",
                    c[i * n + j]
                );
            }
        }
    }
}
