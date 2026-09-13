//! Single-precision matrix products for the network: `C[m x n] = A[m x k] * B[n x k]^T`, both
//! operands contiguous along k, which is how Kaldi stores affine parameters (rows are outputs).
//!
//! Own kernel: an 8-wide FMA tile of four A rows by three B rows, with an explicit AVX2+FMA
//! path behind runtime detection and a portable path the compiler vectorizes for the rest.
// [[rr:TD-2#Dependency policy]]

/// The fold the AVX2 `hsum` performs, so the portable kernels land on the same bits: the two
/// halves lane-wise, then pairwise. `[[rr:TD-3#Accumulation order is part of the contract]]`
#[inline(always)]
fn fold8(acc: &[f32; 8]) -> f32 {
    ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]))
}

#[inline(always)]
fn dot4_portable(a: &[f32], b0: &[f32], b1: &[f32], b2: &[f32], b3: &[f32]) -> [f32; 4] {
    let k = a.len();
    let mut acc = [[0.0f32; 8]; 4];
    let mut i = 0;
    while i + 8 <= k {
        for l in 0..8 {
            let x = a[i + l];
            acc[0][l] = x.mul_add(b0[i + l], acc[0][l]);
            acc[1][l] = x.mul_add(b1[i + l], acc[1][l]);
            acc[2][l] = x.mul_add(b2[i + l], acc[2][l]);
            acc[3][l] = x.mul_add(b3[i + l], acc[3][l]);
        }
        i += 8;
    }
    let mut out = [
        fold8(&acc[0]),
        fold8(&acc[1]),
        fold8(&acc[2]),
        fold8(&acc[3]),
    ];
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
            acc[l] = a[i + l].mul_add(b[i + l], acc[l]);
        }
        i += 8;
    }
    let mut s = fold8(&acc);
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
        assert!(b0.len() == k && b1.len() == k && b2.len() == k && b3.len() == k);
        let mut c0 = _mm256_setzero_ps();
        let mut c1 = _mm256_setzero_ps();
        let mut c2 = _mm256_setzero_ps();
        let mut c3 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= k {
            // SAFETY: every slice holds k floats and i + 8 <= k, so each unaligned load stays in
            // bounds; the caller checked for AVX2 and FMA.
            unsafe {
                let x = _mm256_loadu_ps(a.as_ptr().add(i));
                c0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(b0.as_ptr().add(i)), c0);
                c1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(b1.as_ptr().add(i)), c1);
                c2 = _mm256_fmadd_ps(x, _mm256_loadu_ps(b2.as_ptr().add(i)), c2);
                c3 = _mm256_fmadd_ps(x, _mm256_loadu_ps(b3.as_ptr().add(i)), c3);
            }
            i += 8;
        }
        // SAFETY: the caller checked for AVX2 and FMA.
        let mut out = unsafe { [hsum(c0), hsum(c1), hsum(c2), hsum(c3)] };
        while i < k {
            out[0] += a[i] * b0[i];
            out[1] += a[i] * b1[i];
            out[2] += a[i] * b2[i];
            out[3] += a[i] * b3[i];
            i += 1;
        }
        out
    }

    /// `hsum` over four accumulators at once, same tree, so the results do not move: the
    /// pairwise lane fold, then `(a0+a1) + (a2+a3)`. Lanes 0, 4, 1, 5 hold the four sums.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn hsum4(u: __m256, v: __m256, w: __m256, x: __m256) -> __m256 {
        let uv = _mm256_add_ps(
            _mm256_permute2f128_ps(u, v, 0x20),
            _mm256_permute2f128_ps(u, v, 0x31),
        );
        let wx = _mm256_add_ps(
            _mm256_permute2f128_ps(w, x, 0x20),
            _mm256_permute2f128_ps(w, x, 0x31),
        );
        let t = _mm256_add_ps(
            _mm256_shuffle_ps(uv, wx, 0x88),
            _mm256_shuffle_ps(uv, wx, 0xdd),
        );
        _mm256_add_ps(_mm256_shuffle_ps(t, t, 0x88), _mm256_shuffle_ps(t, t, 0xdd))
    }

    /// `[[rr:TD-3#A tile fits the register file]]`
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot4x3(a: [&[f32]; 4], b: [&[f32]; 3], out: &mut [[f32; 3]; 4]) {
        let k = a[0].len();
        assert!(a.iter().all(|r| r.len() == k) && b.iter().all(|r| r.len() == k));
        let mut acc = [[_mm256_setzero_ps(); 3]; 4];
        let mut i = 0;
        while i + 8 <= k {
            // SAFETY: every slice holds k floats and i + 8 <= k, so each unaligned load stays in
            // bounds; the caller checked for AVX2 and FMA.
            unsafe {
                let b0 = _mm256_loadu_ps(b[0].as_ptr().add(i));
                let b1 = _mm256_loadu_ps(b[1].as_ptr().add(i));
                let b2 = _mm256_loadu_ps(b[2].as_ptr().add(i));
                for r in 0..4 {
                    let x = _mm256_loadu_ps(a[r].as_ptr().add(i));
                    acc[r][0] = _mm256_fmadd_ps(x, b0, acc[r][0]);
                    acc[r][1] = _mm256_fmadd_ps(x, b1, acc[r][1]);
                    acc[r][2] = _mm256_fmadd_ps(x, b2, acc[r][2]);
                }
            }
            i += 8;
        }
        let mut st = [0.0f32; 24];
        let p = st.as_mut_ptr();
        // SAFETY: three unaligned stores of eight floats fill the 24-float scratch exactly; the
        // caller checked for AVX2 and FMA.
        unsafe {
            _mm256_storeu_ps(p, hsum4(acc[0][0], acc[0][1], acc[0][2], acc[1][0]));
            _mm256_storeu_ps(p.add(8), hsum4(acc[1][1], acc[1][2], acc[2][0], acc[2][1]));
            _mm256_storeu_ps(p.add(16), hsum4(acc[2][2], acc[3][0], acc[3][1], acc[3][2]));
        }
        const LANE: [usize; 4] = [0, 4, 1, 5];
        for r in 0..4 {
            for c in 0..3 {
                let f = r * 3 + c;
                out[r][c] = st[(f / 4) * 8 + LANE[f % 4]];
            }
        }
        while i < k {
            for r in 0..4 {
                for c in 0..3 {
                    out[r][c] += a[r][i] * b[c][i];
                }
            }
            i += 1;
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot1(a: &[f32], b: &[f32]) -> f32 {
        let k = a.len();
        assert_eq!(b.len(), k);
        let mut c = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= k {
            // SAFETY: both slices hold k floats and i + 8 <= k, so each unaligned load stays in
            // bounds; the caller checked for AVX2 and FMA.
            unsafe {
                c = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a.as_ptr().add(i)),
                    _mm256_loadu_ps(b.as_ptr().add(i)),
                    c,
                );
            }
            i += 8;
        }
        // SAFETY: the caller checked for AVX2 and FMA.
        let mut s = unsafe { hsum(c) };
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
    // B rows in blocks that stay in L2 while every A row streams past, a whole number of tiles.
    let block_rows = (256 * 1024 / (k.max(1) * 4)).clamp(3, 510);
    let block_rows = block_rows - block_rows % 3;
    let mut j0 = 0;
    while j0 < n {
        let j1 = (j0 + block_rows).min(n);
        let mut i = 0;
        #[cfg(target_arch = "x86_64")]
        if avx {
            while i + 4 <= m {
                let ra = [
                    &a[i * k..(i + 1) * k],
                    &a[(i + 1) * k..(i + 2) * k],
                    &a[(i + 2) * k..(i + 3) * k],
                    &a[(i + 3) * k..(i + 4) * k],
                ];
                let mut j = j0;
                while j + 3 <= j1 {
                    let rb = [
                        &b[j * k..(j + 1) * k],
                        &b[(j + 1) * k..(j + 2) * k],
                        &b[(j + 2) * k..(j + 3) * k],
                    ];
                    let mut out = [[0.0f32; 3]; 4];
                    // SAFETY: `avx` is the runtime AVX2 and FMA check; the rows are k long.
                    unsafe { avx2::dot4x3(ra, rb, &mut out) };
                    for r in 0..4 {
                        c[(i + r) * n + j..(i + r) * n + j + 3].copy_from_slice(&out[r]);
                    }
                    j += 3;
                }
                while j < j1 {
                    let br = &b[j * k..(j + 1) * k];
                    for r in 0..4 {
                        // SAFETY: `avx` is the runtime AVX2 and FMA check; the rows are k long.
                        c[(i + r) * n + j] = unsafe { avx2::dot1(ra[r], br) };
                    }
                    j += 1;
                }
                i += 4;
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
                    // SAFETY: `avx` is the runtime AVX2 and FMA check; the rows are k long.
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
                    // SAFETY: `avx` is the runtime AVX2 and FMA check; the rows are k long.
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

    /// The accumulation order `[[rr:TD-3#Accumulation order is part of the contract]]` fixes:
    /// eight lanes over k, folded `(a0+a4 + a1+a5) + (a2+a6 + a3+a7)`, then a scalar tail.
    fn lane_tree(a: &[f32], b: &[f32]) -> f32 {
        let k = a.len();
        let mut acc = [0.0f32; 8];
        let mut i = 0;
        while i + 8 <= k {
            for (l, acc) in acc.iter_mut().enumerate() {
                *acc = a[i + l].mul_add(b[i + l], *acc);
            }
            i += 8;
        }
        let mut s =
            ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
        while i < k {
            s += a[i] * b[i];
            i += 1;
        }
        s
    }

    #[test]
    // Interpreted, the 24x1280x96 product runs for many minutes; the test below holds the same
    // contract on the portable kernels, the only ones Miri reaches.
    #[cfg_attr(miri, ignore)]
    fn accumulation_order_is_the_contract() {
        // shapes that exercise the 4x3 tile, its column tail, and a k that is not a multiple
        // of eight
        for &(m, k, n) in &[
            (24usize, 1280usize, 96usize),
            (24, 150, 640),
            (8, 96, 11),
            (5, 37, 7),
        ] {
            let a: Vec<f32> = (0..m * k)
                .map(|i| (((i * 2654435761) % 65521) as f32 / 32760.0) - 1.0)
                .collect();
            let b: Vec<f32> = (0..n * k)
                .map(|i| (((i * 40503 + 7) % 65519) as f32 / 32759.0) - 1.0)
                .collect();
            let mut c = vec![0.0f32; m * n];
            gemm_abt(&a, m, k, &b, n, None, &mut c);
            for i in 0..m {
                for j in 0..n {
                    let want = lane_tree(&a[i * k..(i + 1) * k], &b[j * k..(j + 1) * k]);
                    assert_eq!(
                        c[i * n + j].to_bits(),
                        want.to_bits(),
                        "{m}x{k}x{n} at {i},{j}: {} vs {want}",
                        c[i * n + j]
                    );
                }
            }
        }
    }

    /// The portable kernels are the whole path off x86, and never run on a CI host with AVX2.
    #[test]
    fn the_portable_kernels_keep_the_contract() {
        for &k in &[1280usize, 150, 37, 8, 3] {
            let a: Vec<f32> = (0..k)
                .map(|i| (((i * 7919) % 251) as f32 / 125.0) - 1.0)
                .collect();
            let b: Vec<f32> = (0..4 * k)
                .map(|i| (((i * 104729 + 3) % 257) as f32 / 128.0) - 1.0)
                .collect();
            let rows: Vec<&[f32]> = (0..4).map(|j| &b[j * k..(j + 1) * k]).collect();
            let want: Vec<f32> = rows.iter().map(|r| lane_tree(&a, r)).collect();
            let got = dot4_portable(&a, rows[0], rows[1], rows[2], rows[3]);
            for j in 0..4 {
                assert_eq!(got[j].to_bits(), want[j].to_bits(), "k={k} dot4 row {j}");
                let one = dot1_portable(&a, rows[j]);
                assert_eq!(one.to_bits(), want[j].to_bits(), "k={k} dot1 row {j}");
            }
        }
    }

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
