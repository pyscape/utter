//! The i-vector branch of the front end, after Kaldi's `OnlineIvectorFeature`: online CMVN
//! (mean only) primed from the global stats, splice, LDA, Gaussian selection on the diagonal
//! UBM, posterior pruning and scaling, online statistics with prior scaling past `max_count`,
//! and the i-vector estimated by conjugate gradient, most recent estimate for every frame.
// [[rr:TD-2#Front end: the i-vector branch]]

use crate::kaldi_io::{err, parse_text_matrix, KaldiReader};
use std::io::Result;

pub struct DiagGmm {
    pub dim: usize,
    pub num_gauss: usize,
    gconsts: Vec<f32>,
    means_invvars: Vec<f32>,
    inv_vars: Vec<f32>,
}

impl DiagGmm {
    pub fn parse(bytes: &[u8]) -> Result<DiagGmm> {
        let mut r = KaldiReader::new(bytes);
        r.expect_binary()?;
        r.expect_token("<DiagGMM>")?;
        let mut tok = r.read_token()?;
        if tok == "<GCONSTS>" {
            r.read_float_vec()?;
            tok = r.read_token()?;
        }
        if tok != "<WEIGHTS>" {
            return Err(err("DiagGmm: expected <WEIGHTS>"));
        }
        let weights = r.read_float_vec()?;
        r.expect_token("<MEANS_INVVARS>")?;
        let (g, d, means_invvars) = r.read_float_matrix()?;
        r.expect_token("<INV_VARS>")?;
        let (g2, d2, inv_vars) = r.read_float_matrix()?;
        if g != g2 || d != d2 || weights.len() != g {
            return Err(err("DiagGmm: dimension mismatch"));
        }
        let offset = -0.5 * (2.0 * std::f64::consts::PI).ln() * d as f64;
        let gconsts = (0..g)
            .map(|m| {
                let mut gc = (weights[m] as f64).ln() + offset;
                for k in 0..d {
                    let miv = means_invvars[m * d + k] as f64;
                    let iv = inv_vars[m * d + k] as f64;
                    gc += 0.5 * iv.ln() - 0.5 * miv * miv / iv;
                }
                gc as f32
            })
            .collect();
        Ok(DiagGmm { dim: d, num_gauss: g, gconsts, means_invvars, inv_vars })
    }

    pub fn loglikes(&self, x: &[f32], out: &mut [f32]) {
        let d = self.dim;
        let sq: Vec<f32> = x.iter().map(|v| v * v).collect();
        for m in 0..self.num_gauss {
            let mi = &self.means_invvars[m * d..(m + 1) * d];
            let iv = &self.inv_vars[m * d..(m + 1) * d];
            let mut a = 0.0f32;
            let mut b = 0.0f32;
            for k in 0..d {
                a += mi[k] * x[k];
                b += iv[k] * sq[k];
            }
            out[m] = self.gconsts[m] + a - 0.5 * b;
        }
    }
}

/// Kaldi's `VectorToPosteriorEntry`: the `num_gselect` most likely Gaussians as posteriors,
/// pruned below `min_post` of the total, renormalized to one.
pub fn select_posteriors(loglikes: &[f32], num_gselect: usize, min_post: f32) -> Vec<(usize, f32)> {
    let max_like = loglikes.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut temp: Vec<(usize, f32)> = Vec::new();
    if min_post != 0.0 {
        let cutoff = max_like + min_post.ln();
        for (g, &l) in loglikes.iter().enumerate() {
            if l > cutoff {
                temp.push((g, (l - max_like).exp()));
            }
        }
    }
    if temp.is_empty() {
        temp = loglikes.iter().enumerate().map(|(g, &l)| (g, (l - max_like).exp())).collect();
    }
    temp.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    temp.truncate(num_gselect.min(temp.len()));
    let mut tot: f32 = temp.iter().map(|p| p.1).sum();
    let cutoff = min_post * tot;
    while temp.len() > 1 && temp.last().unwrap().1 < cutoff {
        tot -= temp.pop().unwrap().1;
    }
    let inv = 1.0 / tot;
    for p in temp.iter_mut() {
        p.1 *= inv;
    }
    temp
}

#[inline]
fn packed_index(i: usize, j: usize) -> usize {
    if i >= j {
        i * (i + 1) / 2 + j
    } else {
        j * (j + 1) / 2 + i
    }
}

/// y = A x for a packed symmetric A of dimension n.
fn sp_mat_vec(a: &[f64], n: usize, x: &[f64], y: &mut [f64]) {
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += a[packed_index(i, j)] * x[j];
        }
        y[i] = s;
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Solve A x = b by Cholesky, A packed symmetric positive definite.
fn cholesky_solve(a: &[f64], n: usize, b: &[f64], x: &mut [f64]) -> bool {
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[packed_index(i, j)];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    return false;
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }
        y[i] = s / l[i * n + i];
    }
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in i + 1..n {
            s -= l[k * n + i] * x[k];
        }
        x[i] = s / l[i * n + i];
    }
    true
}

/// Kaldi's `LinearCgd` with its default options: residual recomputed when it shrinks a
/// hundredfold, exact solve when the residual got worse.
fn linear_cgd(a: &[f64], n: usize, b: &[f64], x: &mut [f64], max_iters: i32) {
    let mut r = vec![0.0f64; n];
    let mut p = vec![0.0f64; n];
    let mut ap = vec![0.0f64; n];
    let x_orig = x.to_vec();
    sp_mat_vec(a, n, x, &mut ap);
    for i in 0..n {
        p[i] = b[i] - ap[i];
        r[i] = -p[i];
    }
    let mut r_cur = dot(&r, &r);
    let r_initial = r_cur;
    let mut r_recompute = r_cur;
    let max_error_sq = f64::MIN_POSITIVE;
    let residual_factor = 0.01f64 * 0.01;
    let inv_residual_factor = 1.0 / residual_factor;
    let mut k = 0i32;
    while (k as usize) < n + 5 && k != max_iters {
        sp_mat_vec(a, n, &p, &mut ap);
        let alpha = -dot(&p, &r) / dot(&p, &ap);
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] += alpha * ap[i];
        }
        let mut r_next = dot(&r, &r);
        if r_next < residual_factor * r_recompute || r_next > inv_residual_factor * r_recompute {
            sp_mat_vec(a, n, x, &mut r);
            for i in 0..n {
                r[i] -= b[i];
            }
            r_next = dot(&r, &r);
            r_recompute = r_next;
        }
        k += 1;
        if r_next <= max_error_sq {
            r_cur = r_next;
            break;
        }
        let beta = r_next / r_cur;
        for i in 0..n {
            p[i] = beta * p[i] - r[i];
        }
        r_cur = r_next;
    }
    if r_cur > r_initial && r_cur > r_initial + 1.0e-10 * dot(b, b) {
        let mut exact = vec![0.0f64; n];
        if cholesky_solve(a, n, b, &mut exact) {
            x.copy_from_slice(&exact);
        } else {
            x.copy_from_slice(&x_orig);
        }
    }
}

pub struct IvectorExtractor {
    pub feat_dim: usize,
    pub ivector_dim: usize,
    pub num_gauss: usize,
    pub prior_offset: f64,
    /// Per Gaussian: Sigma_i^{-1} M_i, feat_dim x ivector_dim row-major.
    sigma_inv_m: Vec<Vec<f64>>,
    /// Per Gaussian: M_i^T Sigma_i^{-1} M_i, packed symmetric.
    u: Vec<Vec<f64>>,
}

impl IvectorExtractor {
    pub fn parse(bytes: &[u8]) -> Result<IvectorExtractor> {
        let mut r = KaldiReader::new(bytes);
        r.expect_binary()?;
        r.expect_token("<IvectorExtractor>")?;
        r.expect_token("<w>")?;
        let (w_rows, _, _) = r.read_double_matrix()?;
        if w_rows != 0 {
            return Err(err("i-vector-dependent weights are not supported"));
        }
        r.expect_token("<w_vec>")?;
        let _w_vec = r.read_double_vec()?;
        r.expect_token("<M>")?;
        let size = r.read_i32()? as usize;
        let mut m = Vec::with_capacity(size);
        for _ in 0..size {
            m.push(r.read_double_matrix()?);
        }
        r.expect_token("<SigmaInv>")?;
        let mut sigma_inv = Vec::with_capacity(size);
        for _ in 0..size {
            sigma_inv.push(r.read_packed_double()?);
        }
        r.expect_token("<IvectorOffset>")?;
        let prior_offset = r.read_f64()?;
        r.expect_token("</IvectorExtractor>")?;
        let (d, s) = (m[0].0, m[0].1);
        let mut sigma_inv_m = Vec::with_capacity(size);
        let mut u = Vec::with_capacity(size);
        for i in 0..size {
            let (_, _, mi) = &m[i];
            let (sd, si) = &sigma_inv[i];
            if *sd != d {
                return Err(err("SigmaInv dimension mismatch"));
            }
            // sim = Sigma^{-1} M : d x s
            let mut sim = vec![0.0f64; d * s];
            for a in 0..d {
                for c in 0..s {
                    let mut acc = 0.0;
                    for b in 0..d {
                        acc += si[packed_index(a, b)] * mi[b * s + c];
                    }
                    sim[a * s + c] = acc;
                }
            }
            // U = M^T sim : s x s, packed lower
            let mut ui = vec![0.0f64; s * (s + 1) / 2];
            for p in 0..s {
                for q in 0..=p {
                    let mut acc = 0.0;
                    for b in 0..d {
                        acc += mi[b * s + p] * sim[b * s + q];
                    }
                    ui[packed_index(p, q)] = acc;
                }
            }
            sigma_inv_m.push(sim);
            u.push(ui);
        }
        Ok(IvectorExtractor { feat_dim: d, ivector_dim: s, num_gauss: size, prior_offset, sigma_inv_m, u })
    }
}

/// Kaldi's `OnlineIvectorEstimationStats`.
pub struct OnlineIvectorStats {
    dim: usize,
    prior_offset: f64,
    max_count: f64,
    num_frames: f64,
    linear: Vec<f64>,
    quadratic: Vec<f64>,
}

impl OnlineIvectorStats {
    pub fn new(dim: usize, prior_offset: f64, max_count: f64) -> Self {
        let mut s = OnlineIvectorStats {
            dim,
            prior_offset,
            max_count,
            num_frames: 0.0,
            linear: vec![0.0; dim],
            quadratic: vec![0.0; dim * (dim + 1) / 2],
        };
        s.linear[0] += prior_offset;
        for i in 0..dim {
            s.quadratic[packed_index(i, i)] += 1.0;
        }
        s
    }

    /// Accumulate frames with their pruned, scaled posteriors.
    pub fn acc(&mut self, ext: &IvectorExtractor, feats: &[&[f32]], posts: &[Vec<(usize, f32)>]) {
        let d = ext.feat_dim;
        let s = self.dim;
        // Per Gaussian: total weight and weighted feature sum, in first-seen order.
        let mut order: Vec<usize> = Vec::new();
        let mut slot = vec![usize::MAX; ext.num_gauss];
        let mut tot_w: Vec<f64> = Vec::new();
        let mut wsum: Vec<Vec<f64>> = Vec::new();
        for (t, post) in posts.iter().enumerate() {
            for &(g, w) in post {
                if slot[g] == usize::MAX {
                    slot[g] = order.len();
                    order.push(g);
                    tot_w.push(0.0);
                    wsum.push(vec![0.0; d]);
                }
                let k = slot[g];
                tot_w[k] += w as f64;
                for (acc, &x) in wsum[k].iter_mut().zip(feats[t]) {
                    *acc += w as f64 * x as f64;
                }
            }
        }
        let mut tot_weight = 0.0;
        for (k, &g) in order.iter().enumerate() {
            let sim = &ext.sigma_inv_m[g];
            for c in 0..s {
                let mut acc = 0.0;
                for a in 0..d {
                    acc += sim[a * s + c] * wsum[k][a];
                }
                self.linear[c] += acc;
            }
            let ug = &ext.u[g];
            for (q, &v) in self.quadratic.iter_mut().zip(ug) {
                *q += tot_w[k] * v;
            }
            tot_weight += tot_w[k];
        }
        if self.max_count > 0.0 {
            let old = self.num_frames;
            let new = self.num_frames + tot_weight;
            let old_scale = old.max(self.max_count) / self.max_count;
            let new_scale = new.max(self.max_count) / self.max_count;
            let change = new_scale - old_scale;
            if change != 0.0 {
                self.linear[0] += self.prior_offset * change;
                for i in 0..s {
                    self.quadratic[packed_index(i, i)] += change;
                }
            }
        }
        self.num_frames += tot_weight;
    }

    pub fn get_ivector(&self, num_cg_iters: i32, ivector: &mut [f64]) {
        if self.num_frames > 0.0 {
            if ivector[0] == 0.0 {
                ivector[0] = self.prior_offset;
            }
            linear_cgd(&self.quadratic, self.dim, &self.linear, ivector, num_cg_iters);
        } else {
            for v in ivector.iter_mut() {
                *v = 0.0;
            }
            ivector[0] = self.prior_offset;
        }
    }

    pub fn num_frames(&self) -> f64 {
        self.num_frames
    }
}

#[derive(Clone, Debug)]
pub struct IvectorOptions {
    pub ivector_period: usize,
    pub num_gselect: usize,
    pub min_post: f32,
    pub posterior_scale: f32,
    pub max_count: f64,
    pub num_cg_iters: i32,
    pub max_remembered_frames: usize,
    pub cmn_window: usize,
    pub speaker_frames: usize,
    pub global_frames: usize,
    pub left_context: usize,
    pub right_context: usize,
}

impl Default for IvectorOptions {
    fn default() -> Self {
        IvectorOptions {
            ivector_period: 10,
            num_gselect: 5,
            min_post: 0.025,
            posterior_scale: 0.1,
            max_count: 100.0,
            num_cg_iters: 15,
            max_remembered_frames: 1000,
            cmn_window: 600,
            speaker_frames: 600,
            global_frames: 200,
            left_context: 3,
            right_context: 3,
        }
    }
}

fn conf_int(txt: &str, key: &str) -> Option<usize> {
    for line in txt.lines() {
        if let Some(v) = line.trim().strip_prefix(&format!("--{key}=")) {
            return v.trim().parse().ok();
        }
    }
    None
}

/// Everything read once per model for the i-vector branch.
pub struct IvectorInfo {
    pub opts: IvectorOptions,
    pub gmm: DiagGmm,
    pub extractor: IvectorExtractor,
    /// LDA rows x (spliced dim + 1): linear part then the offset column.
    lda: Vec<f32>,
    lda_rows: usize,
    lda_cols: usize,
    /// Global CMVN stats, row 0: sums then count.
    global_mean_stats: Vec<f64>,
}

impl IvectorInfo {
    pub fn open(dir: &std::path::Path) -> Result<IvectorInfo> {
        let mut opts = IvectorOptions::default();
        if let Ok(t) = std::fs::read_to_string(dir.join("splice.conf")) {
            if let Some(v) = conf_int(&t, "left-context") {
                opts.left_context = v;
            }
            if let Some(v) = conf_int(&t, "right-context") {
                opts.right_context = v;
            }
        }
        if let Ok(t) = std::fs::read_to_string(dir.join("online_cmvn.conf")) {
            if let Some(v) = conf_int(&t, "cmn-window") {
                opts.cmn_window = v;
            }
            if let Some(v) = conf_int(&t, "global-frames") {
                opts.global_frames = v;
            }
            if let Some(v) = conf_int(&t, "speaker-frames") {
                opts.speaker_frames = v;
            }
        }
        let gmm = DiagGmm::parse(&std::fs::read(dir.join("final.dubm"))?)?;
        let extractor = IvectorExtractor::parse(&std::fs::read(dir.join("final.ie"))?)?;
        let mat = std::fs::read(dir.join("final.mat"))?;
        let mut r = KaldiReader::new(&mat[..]);
        r.expect_binary()?;
        let (lda_rows, lda_cols, lda) = r.read_float_matrix()?;
        let stats_txt = std::fs::read_to_string(dir.join("global_cmvn.stats"))?;
        let (rows, cols, stats) = parse_text_matrix(&stats_txt)?;
        if rows != 2 {
            return Err(err("global_cmvn.stats must have two rows"));
        }
        let global_mean_stats = stats[..cols].to_vec();
        let base_dim = cols - 1;
        let spliced = base_dim * (opts.left_context + 1 + opts.right_context);
        if lda_cols != spliced && lda_cols != spliced + 1 {
            return Err(err("LDA columns do not match the spliced dimension"));
        }
        if lda_rows != gmm.dim || gmm.dim != extractor.feat_dim {
            return Err(err("i-vector feature dimensions disagree"));
        }
        Ok(IvectorInfo { opts, gmm, extractor, lda, lda_rows, lda_cols, global_mean_stats })
    }

    pub fn ivector_dim(&self) -> usize {
        self.extractor.ivector_dim
    }
}

/// The per-stream state: MFCC frames appended as they arrive, statistics up to the last frame
/// asked for, and the current estimate.
pub struct IvectorStream<'a> {
    info: &'a IvectorInfo,
    frames: Vec<Vec<f32>>,
    /// Running sums of the raw frames, for the CMVN window.
    prefix: Vec<Vec<f64>>,
    stats: OnlineIvectorStats,
    num_frames_stats: usize,
    current: Vec<f64>,
    input_finished: bool,
    /// Pending weight changes per input frame, lowest frame first.
    delta_weights: std::collections::BinaryHeap<std::cmp::Reverse<(usize, u32)>>,
    delta_values: Vec<f32>,
    delta_weights_provided: bool,
    most_recent_frame_with_weight: i64,
}

impl<'a> IvectorStream<'a> {
    pub fn new(info: &'a IvectorInfo) -> Self {
        let dim = info.extractor.ivector_dim;
        let mut current = vec![0.0; dim];
        current[0] = info.extractor.prior_offset;
        IvectorStream {
            info,
            frames: Vec::new(),
            prefix: vec![vec![0.0; info.global_mean_stats.len() - 1]],
            stats: OnlineIvectorStats::new(dim, info.extractor.prior_offset, info.opts.max_count),
            num_frames_stats: 0,
            current,
            input_finished: false,
            delta_weights: std::collections::BinaryHeap::new(),
            delta_values: Vec::new(),
            delta_weights_provided: false,
            most_recent_frame_with_weight: -1,
        }
    }

    /// Kaldi's `UpdateFrameWeights`: weight changes for input frames, applied as their frames
    /// are reached.
    pub fn update_frame_weights(&mut self, deltas: &[(usize, f32)]) {
        for &(frame, w) in deltas {
            let idx = self.delta_values.len() as u32;
            self.delta_values.push(w);
            self.delta_weights.push(std::cmp::Reverse((frame, idx)));
            if frame as i64 > self.most_recent_frame_with_weight {
                self.most_recent_frame_with_weight = frame as i64;
            }
        }
        self.delta_weights_provided = true;
    }

    /// Kaldi's `GetMinPost`.
    fn min_post_for(&self, weight: f32) -> f32 {
        let abs = weight.abs();
        if abs == 0.0 {
            return 0.99;
        }
        (self.info.opts.min_post / abs).min(0.99)
    }

    /// Kaldi's `UpdateStatsForFrames`: frames with weights, duplicates summed.
    fn update_stats_for_frames(&mut self, frame_weights: &[(usize, f32)]) {
        if frame_weights.is_empty() {
            return;
        }
        let mut merged: Vec<(usize, f32)> = frame_weights.to_vec();
        merged.sort_by_key(|p| p.0);
        let mut out: Vec<(usize, f32)> = Vec::with_capacity(merged.len());
        for (f, w) in merged {
            match out.last_mut() {
                Some(last) if last.0 == f => last.1 += w,
                _ => out.push((f, w)),
            }
        }
        let o = &self.info.opts;
        let fd = self.info.extractor.feat_dim;
        let mut ll = vec![0.0f32; self.info.gmm.num_gauss];
        let mut raw: Vec<Vec<f32>> = Vec::with_capacity(out.len());
        let mut posts = Vec::with_capacity(out.len());
        for &(t, weight) in &out {
            let mut post = Vec::new();
            if weight != 0.0 {
                let mut f = vec![0.0f32; fd];
                self.lda_frame(t, true, &mut f);
                self.info.gmm.loglikes(&f, &mut ll);
                post = select_posteriors(&ll, o.num_gselect, self.min_post_for(weight));
                for p in post.iter_mut() {
                    p.1 *= o.posterior_scale * weight;
                }
            }
            posts.push(post);
            let mut g = vec![0.0f32; fd];
            self.lda_frame(t, false, &mut g);
            raw.push(g);
        }
        let refs: Vec<&[f32]> = raw.iter().map(|v| v.as_slice()).collect();
        self.stats.acc(&self.info.extractor, &refs, &posts);
    }

    fn update_stats_until_frame_weighted(&mut self, frame: usize) {
        assert!(frame as i64 <= self.most_recent_frame_with_weight, "i-vector frame has no weight yet");
        let mut frame_weights: Vec<(usize, f32)> = Vec::new();
        while self.num_frames_stats <= frame {
            let t = self.num_frames_stats;
            while let Some(std::cmp::Reverse((f, idx))) = self.delta_weights.peek().copied() {
                if f > t {
                    break;
                }
                self.delta_weights.pop();
                frame_weights.push((f, self.delta_values[idx as usize]));
            }
            if t == frame {
                self.update_stats_for_frames(&frame_weights);
                frame_weights.clear();
                let mut cur = std::mem::take(&mut self.current);
                self.stats.get_ivector(self.info.opts.num_cg_iters, &mut cur);
                self.current = cur;
            }
            self.num_frames_stats += 1;
        }
        self.update_stats_for_frames(&frame_weights);
    }

    pub fn push_frame(&mut self, frame: &[f32]) {
        let last = self.prefix.last().unwrap();
        let next: Vec<f64> = last.iter().zip(frame).map(|(a, &b)| a + b as f64).collect();
        self.prefix.push(next);
        self.frames.push(frame.to_vec());
    }

    pub fn set_input_finished(&mut self) {
        self.input_finished = true;
    }

    /// Frames the i-vector branch can serve: the spliced view needs the right context.
    pub fn num_frames_ready(&self) -> usize {
        if self.input_finished {
            self.frames.len()
        } else {
            self.frames.len().saturating_sub(self.info.opts.right_context)
        }
    }

    fn cmvn_frame(&self, t: usize, out: &mut [f32]) {
        let o = &self.info.opts;
        let dim = out.len();
        let lo = (t + 1).saturating_sub(o.cmn_window);
        let count = (t + 1 - lo) as f64;
        let mut sums: Vec<f64> = (0..dim).map(|d| self.prefix[t + 1][d] - self.prefix[lo][d]).collect();
        let mut cur_count = count;
        if cur_count < o.cmn_window as f64 {
            let global_count = self.info.global_mean_stats[dim];
            let mut from_global = o.cmn_window as f64 - cur_count;
            if from_global > o.global_frames as f64 {
                from_global = o.global_frames as f64;
            }
            if from_global > 0.0 {
                let scale = from_global / global_count;
                for d in 0..dim {
                    sums[d] += scale * self.info.global_mean_stats[d];
                }
                cur_count += from_global;
            }
        }
        let src = &self.frames[t];
        for d in 0..dim {
            out[d] = src[d] - (sums[d] / cur_count) as f32;
        }
    }

    /// Spliced then LDA-transformed frame t, from raw frames or from CMVN frames.
    fn lda_frame(&self, t: usize, normalized: bool, out: &mut [f32]) {
        let o = &self.info.opts;
        let dim = self.frames[0].len();
        let total = self.frames.len();
        let width = o.left_context + 1 + o.right_context;
        let mut spliced = vec![0.0f32; dim * width];
        let mut tmp = vec![0.0f32; dim];
        for (n, t2) in (t as i64 - o.left_context as i64..=t as i64 + o.right_context as i64).enumerate() {
            let t2 = t2.clamp(0, total as i64 - 1) as usize;
            if normalized {
                self.cmvn_frame(t2, &mut tmp);
                spliced[n * dim..(n + 1) * dim].copy_from_slice(&tmp);
            } else {
                spliced[n * dim..(n + 1) * dim].copy_from_slice(&self.frames[t2]);
            }
        }
        let (rows, cols) = (self.info.lda_rows, self.info.lda_cols);
        let lin = cols.min(spliced.len());
        for r in 0..rows {
            let row = &self.info.lda[r * cols..(r + 1) * cols];
            let mut acc = if cols > lin { row[lin] } else { 0.0 };
            for c in 0..lin {
                acc += row[c] * spliced[c];
            }
            out[r] = acc;
        }
    }

    fn update_stats_until(&mut self, frame: usize) {
        let o = &self.info.opts;
        let fd = self.info.extractor.feat_dim;
        let mut pending: Vec<usize> = Vec::new();
        let flush = |this: &mut Self, pending: &mut Vec<usize>| {
            if pending.is_empty() {
                return;
            }
            let mut normalized: Vec<Vec<f32>> = Vec::with_capacity(pending.len());
            let mut raw: Vec<Vec<f32>> = Vec::with_capacity(pending.len());
            let mut ll = vec![0.0f32; this.info.gmm.num_gauss];
            let mut posts = Vec::with_capacity(pending.len());
            for &t in pending.iter() {
                let mut f = vec![0.0f32; fd];
                this.lda_frame(t, true, &mut f);
                this.info.gmm.loglikes(&f, &mut ll);
                let mut post = select_posteriors(&ll, o.num_gselect, o.min_post);
                for p in post.iter_mut() {
                    p.1 *= o.posterior_scale;
                }
                posts.push(post);
                normalized.push(f);
                let mut g = vec![0.0f32; fd];
                this.lda_frame(t, false, &mut g);
                raw.push(g);
            }
            let refs: Vec<&[f32]> = raw.iter().map(|v| v.as_slice()).collect();
            this.stats.acc(&this.info.extractor, &refs, &posts);
            pending.clear();
        };
        while self.num_frames_stats <= frame {
            let t = self.num_frames_stats;
            pending.push(t);
            if t == frame {
                flush(self, &mut pending);
                let mut cur = std::mem::take(&mut self.current);
                self.stats.get_ivector(o.num_cg_iters, &mut cur);
                self.current = cur;
            }
            self.num_frames_stats += 1;
        }
        flush(self, &mut pending);
    }

    /// The i-vector feature for `frame` (must be below `num_frames_ready`): statistics are
    /// brought up to that frame and the estimate refreshed, the prior mean removed from
    /// dimension 0.
    pub fn get_frame(&mut self, frame: usize, out: &mut [f32]) {
        assert!(frame < self.num_frames_ready(), "i-vector frame not ready");
        if frame >= self.num_frames_stats {
            if self.delta_weights_provided {
                self.update_stats_until_frame_weighted(frame);
            } else {
                self.update_stats_until(frame);
            }
        }
        for (o, &v) in out.iter_mut().zip(&self.current) {
            *o = v as f32;
        }
        out[0] -= self.info.extractor.prior_offset as f32;
    }

    pub fn num_frames_stats(&self) -> usize {
        self.num_frames_stats
    }
}
