// Vendored from Vosk-Rust (Apache-2.0), see third_party/Vosk-Rust/NOTICE.
//! Forward pass for a Kaldi nnet3 chain network read from `am/final.mdl`.
//!
//! The descriptor graph is executed over a window of input frames as dense [T, dim] matrices.
//! Identity components (Dropout/NoOp/SpecAugment) pass through; the xent branch is never
//! evaluated. `Offset` descriptors clamp at the window edges, which is Kaldi's behaviour at an
//! utterance start (the first frame repeated) and at an utterance end after input finished (the
//! last frame repeated); a caller who wants exact interior frames passes a window that carries
//! the network's context, see `Nnet3::context`.
// [[rr:TD-2#The network]]

// Row windows are i64 and matrix extents usize; every index below is non-negative and within
// the window by the clamp or the assert above it.
#![allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]

use crate::gemm::gemm_abt;

use crate::kaldi_io::{find, KaldiReader};
use std::collections::HashMap;

/// Row-major dense matrix [rows x cols].
#[derive(Clone)]
pub struct Mat {
    pub r: usize,
    pub c: usize,
    pub d: Vec<f32>,
}

impl Mat {
    pub fn new(r: usize, c: usize) -> Self {
        Mat {
            r,
            c,
            d: vec![0.0; r * c],
        }
    }
    #[inline]
    pub fn row(&self, i: usize) -> &[f32] {
        &self.d[i * self.c..(i + 1) * self.c]
    }
    /// y[T, out] = self[T, in] * w[out, in]^T (+ bias)
    fn affine(&self, w: &Mat, bias: Option<&[f32]>) -> Mat {
        assert_eq!(self.c, w.c, "affine dim mismatch");
        let mut y = Mat::new(self.r, w.r);
        gemm_abt(&self.d, self.r, self.c, &w.d, w.r, bias, &mut y.d);
        y
    }
}

pub enum Comp {
    Affine {
        w: Mat,
        b: Vec<f32>,
    },
    Linear {
        w: Mat,
    },
    Tdnn {
        offsets: Vec<i32>,
        w: Mat,
        b: Vec<f32>,
    },
    BatchNorm {
        mean: Vec<f32>,
        /// `target_rms / sqrt(var + epsilon)` per dimension; it does not depend on the frame,
        /// so it is formed once here rather than per chunk.
        scale: Vec<f32>,
    },
    Relu,
    Identity,
    /// A component type this forward pass does not implement. Held rather than refused at
    /// parse time because a model carries components the output never reaches, the xent
    /// branch among them; only one the output depends on is fatal.
    Unsupported(String),
}

pub enum Desc {
    Ref(String),
    Offset(Box<Desc>, i32),
    Scale(f32, Box<Desc>),
    Sum(Box<Desc>, Box<Desc>),
    Append(Vec<Desc>),
    ReplaceIndex(Box<Desc>),
}

/// Names from [`Nnet3::pooling_head`].
pub(crate) struct PoolingHead<'a> {
    pub embed: &'a str,
    pub pooling: &'a str,
    pub extraction: &'a str,
    pub frames: &'a str,
}

enum Node {
    Input,
    DimRange { src: String, off: usize, dim: usize },
    Component { comp: String, desc: Desc },
    Output { desc: Desc },
}

pub struct Nnet3 {
    comps: HashMap<String, Comp>,
    nodes: HashMap<String, Node>,
    /// dim of the `ivector` input node; 0 if none.
    pub ivector_dim: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    /// (left, right) input frames the output at a frame depends on; an upper bound.
    pub context: (usize, usize),
}

fn mat_at(buf: &[u8], pos: usize) -> Mat {
    let (r, c, d) = KaldiReader::new(&buf[pos..]).read_float_matrix().unwrap();
    Mat { r, c, d }
}
fn vec_at(buf: &[u8], pos: usize) -> Vec<f32> {
    KaldiReader::new(&buf[pos..]).read_float_vec().unwrap()
}
fn named_mat(buf: &[u8], s: usize, e: usize, tok: &[u8]) -> Option<Mat> {
    find(&buf[s..e], tok, 0).map(|off| mat_at(buf, s + off + tok.len()))
}
fn named_vec(buf: &[u8], s: usize, e: usize, tok: &[u8]) -> Option<Vec<f32>> {
    find(&buf[s..e], tok, 0).map(|off| vec_at(buf, s + off + tok.len()))
}
fn named_f32(buf: &[u8], s: usize, e: usize, tok: &[u8]) -> Option<f32> {
    find(&buf[s..e], tok, 0).map(|off| {
        KaldiReader::new(&buf[s + off + tok.len()..])
            .read_f32()
            .unwrap()
    })
}

impl Nnet3 {
    pub fn parse(buf: &[u8]) -> Nnet3 {
        let comps = parse_components(buf);
        let (nodes, input_dim, ivector_dim) = parse_graph(buf);
        let mut net = Nnet3 {
            comps,
            nodes,
            ivector_dim,
            input_dim,
            output_dim: 0,
            context: (0, 0),
        };
        net.output_dim = match net.comps.get("output.affine") {
            Some(Comp::Affine { w, .. }) => w.r,
            _ => 0,
        };
        net.context = net.node_context("output");
        net
    }

    /// The chain a statistics-pooling embedding hangs from, read back from the output node:
    /// the affine component the output takes, the pooling component that affine reads, the
    /// extraction component the pooling reads, and the node whose frames the extraction takes.
    /// `None` for a network whose output is not an affine over pooled statistics.
    pub(crate) fn pooling_head(&self) -> Option<PoolingHead<'_>> {
        let component = |node: &str| match self.nodes.get(node)? {
            Node::Component {
                comp,
                desc: Desc::Ref(src),
            } => Some((comp.as_str(), src.as_str())),
            _ => None,
        };
        let Some(Node::Output {
            desc: Desc::Ref(embed_node),
        }) = self.nodes.get("output")
        else {
            return None;
        };
        let (embed, pooling_node) = component(embed_node)?;
        let (pooling, extraction_node) = component(pooling_node)?;
        let (extraction, frames) = component(extraction_node)?;
        let unsupported =
            |c: &str, t: &str| matches!(self.comps.get(c), Some(Comp::Unsupported(u)) if u == t);
        (matches!(self.comps.get(embed), Some(Comp::Affine { .. }))
            && unsupported(pooling, "<StatisticsPoolingComponent>")
            && unsupported(extraction, "<StatisticsExtractionComponent>"))
        .then_some(PoolingHead {
            embed,
            pooling,
            extraction,
            frames,
        })
    }

    /// An affine component's weights, `[out, in]`, and its bias.
    pub(crate) fn affine(&self, comp: &str) -> Option<(&Mat, &[f32])> {
        match self.comps.get(comp)? {
            Comp::Affine { w, b } => Some((w, b)),
            _ => None,
        }
    }

    /// Point the output node at `node`, so the network ends there.
    pub(crate) fn retarget_output(&mut self, node: &str) {
        self.nodes.insert(
            "output".into(),
            Node::Output {
                desc: Desc::Ref(node.into()),
            },
        );
        self.context = self.node_context("output");
        self.output_dim = self.node_dim("output");
    }

    /// Component types the output depends on and this forward pass does not implement, in the
    /// order met. Empty for a model that can be decoded. A component the output never reaches
    /// is not reported: the xent branch is never evaluated.
    pub fn unsupported(&self) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        self.walk("output", &mut seen, &mut out);
        out
    }

    fn walk(
        &self,
        name: &str,
        seen: &mut std::collections::HashSet<String>,
        out: &mut Vec<String>,
    ) {
        if !seen.insert(name.to_string()) {
            return;
        }
        match self.nodes.get(name) {
            None | Some(Node::Input) => {}
            Some(Node::DimRange { src, .. }) => self.walk(src, seen, out),
            Some(Node::Output { desc }) => self.walk_desc(desc, seen, out),
            Some(Node::Component { comp, desc }) => {
                if let Some(Comp::Unsupported(t)) = self.comps.get(comp) {
                    if !out.contains(t) {
                        out.push(t.clone());
                    }
                }
                self.walk_desc(desc, seen, out);
            }
        }
    }

    fn walk_desc(
        &self,
        d: &Desc,
        seen: &mut std::collections::HashSet<String>,
        out: &mut Vec<String>,
    ) {
        match d {
            Desc::Ref(n) => self.walk(n, seen, out),
            Desc::Offset(a, _) | Desc::Scale(_, a) | Desc::ReplaceIndex(a) => {
                self.walk_desc(a, seen, out)
            }
            Desc::Sum(a, b) => {
                self.walk_desc(a, seen, out);
                self.walk_desc(b, seen, out);
            }
            Desc::Append(parts) => {
                for x in parts {
                    self.walk_desc(x, seen, out)
                }
            }
        }
    }

    fn node_context(&self, name: &str) -> (usize, usize) {
        match self.nodes.get(name) {
            None | Some(Node::Input) => (0, 0),
            Some(Node::DimRange { src, .. }) => self.node_context(src),
            Some(Node::Component { comp, desc }) => {
                let (l, r) = self.desc_context(desc);
                match self.comps.get(comp) {
                    Some(Comp::Tdnn { offsets, .. }) => {
                        let lo = offsets
                            .iter()
                            .copied()
                            .min()
                            .unwrap_or(0)
                            .min(0)
                            .unsigned_abs() as usize;
                        let hi = offsets.iter().copied().max().unwrap_or(0).max(0) as usize;
                        (l + lo, r + hi)
                    }
                    _ => (l, r),
                }
            }
            Some(Node::Output { desc }) => self.desc_context(desc),
        }
    }

    fn desc_context(&self, d: &Desc) -> (usize, usize) {
        match d {
            Desc::Ref(n) => self.node_context(n),
            Desc::Offset(a, n) => {
                let (l, r) = self.desc_context(a);
                if *n < 0 {
                    (l + n.unsigned_abs() as usize, r)
                } else {
                    (l, r + *n as usize)
                }
            }
            Desc::Scale(_, a) | Desc::ReplaceIndex(a) => self.desc_context(a),
            Desc::Sum(a, b) => {
                let (la, ra) = self.desc_context(a);
                let (lb, rb) = self.desc_context(b);
                (la.max(lb), ra.max(rb))
            }
            Desc::Append(parts) => parts
                .iter()
                .map(|p| self.desc_context(p))
                .fold((0, 0), |(l, r), (l2, r2)| (l.max(l2), r.max(r2))),
        }
    }

    /// Network output for every input frame of `feats` (no subsampling applied). `ivector` is
    /// the single i-vector the whole window uses.
    pub fn forward_window(&self, feats: Mat, ivector: &[f32]) -> Mat {
        let t = feats.r;
        let mut cache: HashMap<String, Mat> = HashMap::new();
        cache.insert("input".into(), feats);
        let mut iv = Mat::new(1, self.ivector_dim);
        iv.d.copy_from_slice(&ivector[..self.ivector_dim]);
        cache.insert("ivector".into(), iv);
        self.eval_node("output", &mut cache, t)
    }

    /// Whole-utterance forward with one i-vector: chain log-likelihoods at frames 0, 3, 6, ...
    pub fn forward(&self, feats: Mat, ivector: &[f32], subsampling: usize) -> Mat {
        let out = self.forward_window(feats, ivector);
        let cols = out.c;
        let rows = out.r.div_ceil(subsampling);
        let mut y = Mat::new(rows, cols);
        for (j, i) in (0..out.r).step_by(subsampling).enumerate() {
            y.d[j * cols..(j + 1) * cols].copy_from_slice(out.row(i));
        }
        y
    }

    fn eval_node(&self, name: &str, cache: &mut HashMap<String, Mat>, t: usize) -> Mat {
        if let Some(m) = cache.get(name) {
            return m.clone();
        }
        let r = match self
            .nodes
            .get(name)
            .unwrap_or_else(|| panic!("no node {name}"))
        {
            Node::Input => panic!("input {name} not pre-populated"),
            Node::DimRange { src, off, dim } => {
                let s = self.eval_node(src, cache, t);
                let mut m = Mat::new(s.r, *dim);
                for i in 0..s.r {
                    m.d[i * dim..(i + 1) * dim].copy_from_slice(&s.row(i)[*off..off + dim]);
                }
                m
            }
            Node::Component { comp, desc } => {
                let x = self.eval_desc(desc, cache, t);
                self.apply(comp, x)
            }
            Node::Output { desc } => self.eval_desc(desc, cache, t),
        };
        cache.insert(name.into(), r.clone());
        r
    }

    fn eval_desc(&self, d: &Desc, cache: &mut HashMap<String, Mat>, t: usize) -> Mat {
        match d {
            Desc::Ref(n) => self.eval_node(n, cache, t),
            Desc::Offset(a, n) => clamp_offset(&self.eval_desc(a, cache, t), *n),
            Desc::Scale(s, a) => {
                let mut m = self.eval_desc(a, cache, t);
                m.d.iter_mut().for_each(|v| *v *= *s);
                m
            }
            Desc::Sum(a, b) => {
                let mut m = self.eval_desc(a, cache, t);
                let n = self.eval_desc(b, cache, t);
                m.d.iter_mut().zip(&n.d).for_each(|(x, y)| *x += *y);
                m
            }
            Desc::Append(parts) => {
                let mats: Vec<Mat> = parts.iter().map(|p| self.eval_desc(p, cache, t)).collect();
                let rows = mats[0].r;
                let cols: usize = mats.iter().map(|m| m.c).sum();
                let mut out = Mat::new(rows, cols);
                for i in 0..rows {
                    let mut off = 0;
                    for m in &mats {
                        out.d[i * cols + off..i * cols + off + m.c].copy_from_slice(m.row(i));
                        off += m.c;
                    }
                }
                out
            }
            Desc::ReplaceIndex(a) => {
                let m = self.eval_desc(a, cache, t);
                let mut out = Mat::new(t, m.c);
                for i in 0..t {
                    out.d[i * m.c..(i + 1) * m.c].copy_from_slice(m.row(0));
                }
                out
            }
        }
    }

    fn apply(&self, comp: &str, x: Mat) -> Mat {
        match self
            .comps
            .get(comp)
            .unwrap_or_else(|| panic!("no comp {comp}"))
        {
            Comp::Identity => x,
            Comp::Unsupported(t) => panic!("component type {t} is not implemented"),
            Comp::Relu => {
                let mut m = x;
                m.d.iter_mut().for_each(|v| *v = v.max(0.0));
                m
            }
            Comp::BatchNorm { mean, scale } => {
                let mut m = x;
                let (mean, scale) = (&mean[..m.c], &scale[..m.c]);
                for row in m.d.chunks_exact_mut(m.c) {
                    for ((v, mu), s) in row.iter_mut().zip(mean).zip(scale) {
                        *v = (*v - *mu) * *s;
                    }
                }
                m
            }
            Comp::Affine { w, b } => x.affine(w, Some(b)),
            Comp::Linear { w } => x.affine(w, None),
            Comp::Tdnn { offsets, w, b } => {
                let inn = x.c;
                let sp = offsets.len();
                let mut spliced = Mat::new(x.r, inn * sp);
                for (k, &o) in offsets.iter().enumerate() {
                    for i in 0..x.r {
                        let src = (i as i32 + o).clamp(0, x.r as i32 - 1) as usize;
                        let at = i * inn * sp + k * inn;
                        spliced.d[at..at + inn].copy_from_slice(x.row(src));
                    }
                }
                spliced.affine(w, if b.is_empty() { None } else { Some(b) })
            }
        }
    }
}

fn clamp_offset(m: &Mat, n: i32) -> Mat {
    let mut out = Mat::new(m.r, m.c);
    for i in 0..m.r {
        let src = (i as i32 + n).clamp(0, m.r as i32 - 1) as usize;
        out.d[i * m.c..(i + 1) * m.c].copy_from_slice(m.row(src));
    }
    out
}

fn parse_components(buf: &[u8]) -> HashMap<String, Comp> {
    let tag = b"<ComponentName> ";
    let mut starts = vec![];
    let mut p = 0;
    while let Some(i) = find(buf, tag, p) {
        starts.push(i);
        p = i + tag.len();
    }
    starts.push(buf.len());
    let mut comps = HashMap::new();
    for k in 0..starts.len() - 1 {
        let (s, e) = (starts[k], starts[k + 1]);
        let mut r = KaldiReader::new(&buf[s + tag.len()..]);
        let name = r.read_token().unwrap();
        let ctype = r.read_token().unwrap();
        let comp = match ctype.as_str() {
            "<NaturalGradientAffineComponent>" | "<FixedAffineComponent>" | "<AffineComponent>" => {
                Comp::Affine {
                    w: named_mat(buf, s, e, b"<LinearParams> ").unwrap(),
                    b: named_vec(buf, s, e, b"<BiasParams> ").unwrap_or_default(),
                }
            }
            "<LinearComponent>" => Comp::Linear {
                w: named_mat(buf, s, e, b"<Params> ").unwrap(),
            },
            "<TdnnComponent>" => {
                let off =
                    find(&buf[s..e], b"<TimeOffsets> ", 0).unwrap() + s + b"<TimeOffsets> ".len();
                let offsets = KaldiReader::new(&buf[off..]).read_i32_vec().unwrap();
                Comp::Tdnn {
                    offsets,
                    w: named_mat(buf, s, e, b"<LinearParams> ").unwrap(),
                    b: named_vec(buf, s, e, b"<BiasParams> ").unwrap_or_default(),
                }
            }
            "<BatchNormComponent>" => {
                let var = named_vec(buf, s, e, b"<StatsVar> ").unwrap();
                let eps = named_f32(buf, s, e, b"<Epsilon> ").unwrap_or(1e-3);
                let trms = named_f32(buf, s, e, b"<TargetRms> ").unwrap_or(1.0);
                Comp::BatchNorm {
                    mean: named_vec(buf, s, e, b"<StatsMean> ").unwrap(),
                    scale: var.iter().map(|v| trms / (v + eps).sqrt()).collect(),
                }
            }
            "<RectifiedLinearComponent>" => Comp::Relu,
            // Trained-time only: at inference each passes its input through unchanged.
            "<NoOpComponent>"
            | "<DropoutComponent>"
            | "<GeneralDropoutComponent>"
            | "<SpecAugmentTimeMaskComponent>" => Comp::Identity,
            other => Comp::Unsupported(other.to_string()),
        };
        comps.insert(name, comp);
    }
    comps
}

fn kv<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!(" {key}=");
    let l = format!(" {line}");
    let i = l.find(&pat)? + pat.len();
    let rest = &line[i - 1..];
    rest.split_whitespace().next()
}

fn parse_graph(buf: &[u8]) -> (HashMap<String, Node>, usize, usize) {
    let a = find(buf, b"<Nnet3>", 0).unwrap();
    let b = find(buf, b"<NumComponents>", 0).unwrap();
    let txt = String::from_utf8_lossy(&buf[a..b]);
    let mut nodes = HashMap::new();
    let (mut input_dim, mut ivector_dim) = (0, 0);
    for line in txt.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("input-node") {
            let name = kv(rest, "name").unwrap().to_string();
            let dim: usize = kv(rest, "dim").and_then(|d| d.parse().ok()).unwrap_or(0);
            match name.as_str() {
                "input" => input_dim = dim,
                "ivector" => ivector_dim = dim,
                _ => {}
            }
            nodes.insert(name, Node::Input);
        } else if let Some(rest) = line.strip_prefix("dim-range-node") {
            let name = kv(rest, "name").unwrap().to_string();
            nodes.insert(
                name,
                Node::DimRange {
                    src: kv(rest, "input-node").unwrap().to_string(),
                    off: kv(rest, "dim-offset").unwrap().parse().unwrap(),
                    dim: kv(rest, "dim").unwrap().parse().unwrap(),
                },
            );
        } else if let Some(rest) = line.strip_prefix("component-node") {
            let name = kv(rest, "name").unwrap().to_string();
            let comp = kv(rest, "component").unwrap().to_string();
            let desc = &line[line.find("input=").unwrap() + 6..];
            nodes.insert(
                name,
                Node::Component {
                    comp,
                    desc: parse_desc(desc.trim()),
                },
            );
        } else if let Some(rest) = line.strip_prefix("output-node") {
            let name = kv(rest, "name").unwrap().to_string();
            let di = line.find("input=").unwrap() + 6;
            let desc = &line[di..line[di..]
                .find(" objective=")
                .map(|x| x + di)
                .unwrap_or(line.len())];
            nodes.insert(
                name,
                Node::Output {
                    desc: parse_desc(desc.trim()),
                },
            );
        }
    }
    (nodes, input_dim, ivector_dim)
}

fn parse_desc(s: &str) -> Desc {
    let toks = tokenize(s);
    let mut pos = 0;
    parse_expr(&toks, &mut pos)
}
fn tokenize(s: &str) -> Vec<String> {
    let mut out = vec![];
    let mut cur = String::new();
    for ch in s.chars() {
        if ch == '(' || ch == ')' || ch == ',' {
            if !cur.trim().is_empty() {
                out.push(cur.trim().to_string());
            }
            cur.clear();
            out.push(ch.to_string());
        } else {
            cur.push(ch);
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}
fn parse_expr(toks: &[String], pos: &mut usize) -> Desc {
    let head = toks[*pos].clone();
    *pos += 1;
    let is_op = matches!(
        head.as_str(),
        "Offset" | "Scale" | "Sum" | "Append" | "ReplaceIndex" | "Round" | "IfDefined"
    );
    if is_op && *pos < toks.len() && toks[*pos] == "(" {
        *pos += 1;
        let mut args = vec![parse_expr(toks, pos)];
        while toks[*pos] == "," {
            *pos += 1;
            args.push(parse_expr(toks, pos));
        }
        assert_eq!(toks[*pos], ")");
        *pos += 1;
        return build_op(&head, args);
    }
    Desc::Ref(head)
}
fn build_op(op: &str, mut args: Vec<Desc>) -> Desc {
    match op {
        "Offset" => {
            let n = leaf_int(&args[1]);
            Desc::Offset(Box::new(args.remove(0)), n)
        }
        "Scale" => {
            let s = leaf_f32(&args[0]);
            Desc::Scale(s, Box::new(args.remove(1)))
        }
        "Sum" => {
            let b = args.remove(1);
            let a = args.remove(0);
            Desc::Sum(Box::new(a), Box::new(b))
        }
        "Append" => Desc::Append(args),
        "ReplaceIndex" => Desc::ReplaceIndex(Box::new(args.remove(0))),
        "IfDefined" => args.remove(0),
        // Rounding the time index down to a multiple of one leaves it where it is.
        "Round" if leaf_int(&args[1]) == 1 => args.remove(0),
        _ => panic!("unhandled desc op {op}"),
    }
}
fn leaf_int(d: &Desc) -> i32 {
    if let Desc::Ref(s) = d {
        s.parse().unwrap()
    } else {
        panic!("expected int leaf")
    }
}
fn leaf_f32(d: &Desc) -> f32 {
    if let Desc::Ref(s) = d {
        s.parse().unwrap()
    } else {
        panic!("expected f32 leaf")
    }
}

/// Streaming evaluation after Kaldi's looped computation: every node keeps the rows it has
/// computed on one timeline of input frames, and each advance computes, node by node, every
/// row whose inputs exist. Frames before the first are copies of it and, once input has
/// finished, frames after the last are copies of the last, so a node never clamps: the padding
/// is at the input only, as in Kaldi.
///
/// A node's rows live contiguously, so a consumer reads them as a slice and a producer writes
/// its product straight into them: nothing between two matrix products is copied.
pub struct Streamer<'n> {
    net: &'n Nnet3,
    steps: Vec<Step>,
    stores: Vec<Store>,
    /// Every component node, including the ones that share another's rows, to its store.
    store_of: HashMap<String, usize>,
    /// Rows a node keeps behind its frontier for consumers with negative offsets.
    history: usize,
    ivector: Vec<f32>,
    /// Reused between advances for the one matrix a splice has to materialise.
    scratch: Vec<f32>,
    /// The store the output node's rows end up in.
    out: usize,
    out_cols: usize,
}

struct Step {
    name: String,
    store: usize,
    comp: Option<String>,
    dim: usize,
    /// The extreme time offsets the component reads its input at.
    lo: i64,
    hi: i64,
}

/// One node's computed rows, oldest first and contiguous. Rows retire from the front; the
/// buffer slides back to the start only when the back runs out, which the slack at the back
/// makes rare.
#[derive(Default)]
struct Store {
    dim: usize,
    /// Frame index of the row at `data[off * dim ..]`.
    first: i64,
    off: usize,
    len: usize,
    cap: usize,
    data: Vec<f32>,
    /// False until the node has produced anything, when it has no frame index yet.
    started: bool,
}

impl Store {
    fn end(&self) -> i64 {
        self.first + self.len as i64
    }

    fn rows(&self, a: i64, b: i64) -> &[f32] {
        let s = a - self.first;
        let e = b - self.first;
        assert!(
            s >= 0 && e >= s && e <= self.len as i64,
            "rows {a}..{b} wanted of a node holding {}..{}",
            self.first,
            self.end()
        );
        let (s, e) = (self.off + s as usize, self.off + e as usize);
        &self.data[s * self.dim..e * self.dim]
    }

    /// Retire all but `keep` rows, make room for `n` more, and return the row the new ones
    /// start at.
    fn reserve(&mut self, n: usize, keep: usize) -> usize {
        let retire = self.len.saturating_sub(keep);
        self.off += retire;
        self.len -= retire;
        self.first += retire as i64;
        if self.cap < keep + n {
            // slack at the back so one slide covers several advances
            self.cap = keep + n + 64;
            let mut d = vec![0.0f32; self.cap * self.dim];
            d[..self.len * self.dim]
                .copy_from_slice(&self.data[self.off * self.dim..(self.off + self.len) * self.dim]);
            self.data = d;
            self.off = 0;
        } else if self.off + self.len + n > self.cap {
            self.data
                .copy_within(self.off * self.dim..(self.off + self.len) * self.dim, 0);
            self.off = 0;
        }
        self.off + self.len
    }
}

/// A descriptor's rows: borrowed from the producing node's store where the descriptor is a
/// reference to one, and materialised only where it combines several.
enum Rows<'a> {
    Slice(&'a [f32], usize),
    Own(Vec<f32>, usize),
}

impl Rows<'_> {
    fn data(&self) -> &[f32] {
        match self {
            Rows::Slice(d, _) => d,
            Rows::Own(d, _) => d,
        }
    }
    fn cols(&self) -> usize {
        match self {
            Rows::Slice(_, c) | Rows::Own(_, c) => *c,
        }
    }
    fn row(&self, i: usize) -> &[f32] {
        let c = self.cols();
        &self.data()[i * c..(i + 1) * c]
    }
}

impl Nnet3 {
    /// A node whose component is an identity over a plain reference has exactly the rows it
    /// refers to, so it shares that node's store and never runs.
    fn alias<'a>(&'a self, name: &'a str) -> &'a str {
        if let Some(Node::Component { comp, desc }) = self.nodes.get(name) {
            if let (Some(Comp::Identity), Desc::Ref(src)) = (self.comps.get(comp), desc) {
                if matches!(self.nodes.get(src), Some(Node::Component { .. })) {
                    return self.alias(src);
                }
            }
        }
        if let Some(Node::Output {
            desc: Desc::Ref(src),
        }) = self.nodes.get(name)
        {
            if matches!(self.nodes.get(src), Some(Node::Component { .. })) {
                return self.alias(src);
            }
        }
        name
    }

    fn node_dim(&self, name: &str) -> usize {
        match self.nodes.get(name) {
            None => 0,
            Some(Node::Input) => {
                if name == "ivector" {
                    self.ivector_dim
                } else {
                    self.input_dim
                }
            }
            Some(Node::DimRange { dim, .. }) => *dim,
            Some(Node::Component { comp, desc }) => match self.comps.get(comp) {
                Some(Comp::Affine { w, .. } | Comp::Linear { w } | Comp::Tdnn { w, .. }) => w.r,
                _ => self.desc_dim(desc),
            },
            Some(Node::Output { desc }) => self.desc_dim(desc),
        }
    }

    fn desc_dim(&self, d: &Desc) -> usize {
        match d {
            Desc::Ref(n) => self.node_dim(n),
            Desc::Offset(a, _) | Desc::Scale(_, a) | Desc::ReplaceIndex(a) => self.desc_dim(a),
            Desc::Sum(a, _) => self.desc_dim(a),
            Desc::Append(v) => v.iter().map(|p| self.desc_dim(p)).sum(),
        }
    }

    pub fn streamer(&self) -> Streamer<'_> {
        // topological order over component, dim-range and output nodes
        let mut order = Vec::new();
        let mut seen = std::collections::HashSet::new();
        fn visit(
            net: &Nnet3,
            name: &str,
            seen: &mut std::collections::HashSet<String>,
            order: &mut Vec<String>,
        ) {
            if seen.contains(name) {
                return;
            }
            seen.insert(name.to_string());
            match net.nodes.get(name) {
                None | Some(Node::Input) => return,
                Some(Node::DimRange { src, .. }) => visit(net, src, seen, order),
                Some(Node::Component { desc, .. }) | Some(Node::Output { desc }) => {
                    let mut refs = Vec::new();
                    desc_refs(desc, &mut refs);
                    for r in refs {
                        visit(net, &r, seen, order);
                    }
                }
            }
            order.push(name.to_string());
        }
        visit(self, "output", &mut seen, &mut order);

        let mut steps: Vec<Step> = Vec::new();
        let mut store_of: HashMap<String, usize> = HashMap::new();
        let mut stores: Vec<Store> = Vec::new();
        for name in &order {
            if matches!(self.nodes.get(name), Some(Node::DimRange { .. })) {
                continue;
            }
            let owner = self.alias(name);
            if owner != name {
                let id = store_of[owner];
                store_of.insert(name.clone(), id);
                continue;
            }
            let id = stores.len();
            stores.push(Store::default());
            store_of.insert(name.clone(), id);
            let comp = match self.nodes.get(name) {
                Some(Node::Component { comp, .. }) => Some(comp.clone()),
                _ => None,
            };
            let (lo, hi) = match comp.as_deref().and_then(|c| self.comps.get(c)) {
                Some(Comp::Tdnn { offsets, .. }) => (
                    offsets.iter().copied().min().unwrap_or(0) as i64,
                    offsets.iter().copied().max().unwrap_or(0) as i64,
                ),
                _ => (0, 0),
            };
            steps.push(Step {
                name: name.clone(),
                store: id,
                comp,
                dim: self.node_dim(name),
                lo,
                hi,
            });
        }
        let out = store_of[self.alias("output")];
        Streamer {
            net: self,
            steps,
            stores,
            store_of,
            history: 16,
            ivector: vec![0.0; self.ivector_dim],
            scratch: Vec::new(),
            out,
            out_cols: self.node_dim("output"),
        }
    }
}

fn desc_refs(d: &Desc, out: &mut Vec<String>) {
    match d {
        Desc::Ref(n) => out.push(n.clone()),
        Desc::Offset(a, _) | Desc::Scale(_, a) | Desc::ReplaceIndex(a) => desc_refs(a, out),
        Desc::Sum(a, b) => {
            desc_refs(a, out);
            desc_refs(b, out);
        }
        Desc::Append(v) => v.iter().for_each(|p| desc_refs(p, out)),
    }
}

/// The input rows available to a pass: frames `[0, ready)`, padded before the first and, when
/// finished, after the last up to `limit`.
pub struct InputView<'a> {
    pub frames: &'a [Vec<f32>],
    pub ready: usize,
    pub finished: bool,
    /// Rows at or beyond this frame index are not available yet (exclusive frontier).
    pub limit: i64,
    /// Earliest frame index the timeline starts at (negative: left context padding).
    pub first: i64,
}

impl Streamer<'_> {
    fn node_first(&self, name: &str, input: &InputView) -> i64 {
        if let Some(&id) = self.store_of.get(name) {
            if self.stores[id].started {
                return self.stores[id].first;
            }
        }
        match self.net.nodes.get(name) {
            None | Some(Node::Input) => input.first,
            Some(Node::DimRange { src, .. }) => self.node_first(src, input),
            Some(Node::Component { comp, desc }) => {
                let lo = match self.net.comps.get(comp) {
                    Some(Comp::Tdnn { offsets, .. }) => offsets.iter().copied().min().unwrap_or(0),
                    _ => 0,
                };
                self.desc_first(desc, input) - lo as i64
            }
            Some(Node::Output { desc }) => self.desc_first(desc, input),
        }
    }

    fn desc_first(&self, d: &Desc, input: &InputView) -> i64 {
        match d {
            Desc::Ref(n) => self.node_first(n, input),
            Desc::Offset(a, n) => self.desc_first(a, input) - *n as i64,
            Desc::Scale(_, a) => self.desc_first(a, input),
            Desc::ReplaceIndex(_) => i64::MIN / 4,
            Desc::Sum(a, b) => self.desc_first(a, input).max(self.desc_first(b, input)),
            Desc::Append(v) => v
                .iter()
                .map(|p| self.desc_first(p, input))
                .max()
                .unwrap_or(input.first),
        }
    }

    /// Exclusive frontier of rows a node can serve.
    fn node_avail(&self, name: &str, input: &InputView) -> i64 {
        if let Some(&id) = self.store_of.get(name) {
            return if self.stores[id].started {
                self.stores[id].end()
            } else {
                self.node_first(name, input)
            };
        }
        match self.net.nodes.get(name) {
            Some(Node::DimRange { src, .. }) => self.node_avail(src, input),
            _ => input.limit,
        }
    }

    fn desc_avail(&self, d: &Desc, input: &InputView) -> i64 {
        match d {
            Desc::Ref(n) => self.node_avail(n, input),
            Desc::Offset(a, n) => self.desc_avail(a, input) - *n as i64,
            Desc::Scale(_, a) => self.desc_avail(a, input),
            Desc::ReplaceIndex(_) => i64::MAX / 4,
            Desc::Sum(a, b) => self.desc_avail(a, input).min(self.desc_avail(b, input)),
            Desc::Append(v) => v
                .iter()
                .map(|p| self.desc_avail(p, input))
                .min()
                .unwrap_or(input.limit),
        }
    }

    fn node_rows(&self, name: &str, a: i64, b: i64, input: &InputView) -> Rows<'_> {
        if let Some(&id) = self.store_of.get(name) {
            return Rows::Slice(self.stores[id].rows(a, b), self.stores[id].dim);
        }
        match self.net.nodes.get(name) {
            Some(Node::DimRange { src, off, dim }) => {
                let s = self.node_rows(src, a, b, input);
                let rows = (b - a) as usize;
                let mut d = vec![0.0f32; rows * dim];
                for i in 0..rows {
                    d[i * dim..(i + 1) * dim].copy_from_slice(&s.row(i)[*off..off + dim]);
                }
                Rows::Own(d, *dim)
            }
            _ => {
                let dim = self.net.input_dim;
                let rows = (b - a) as usize;
                let mut d = vec![0.0f32; rows * dim];
                for (i, t) in (a..b).enumerate() {
                    let src = t.clamp(0, input.ready as i64 - 1) as usize;
                    d[i * dim..(i + 1) * dim].copy_from_slice(&input.frames[src]);
                }
                Rows::Own(d, dim)
            }
        }
    }

    fn eval_desc(&self, d: &Desc, a: i64, b: i64, input: &InputView) -> Rows<'_> {
        match d {
            Desc::Ref(n) => self.node_rows(n, a, b, input),
            Desc::Offset(x, n) => self.eval_desc(x, a + *n as i64, b + *n as i64, input),
            _ => {
                let cols = self.net.desc_dim(d);
                let mut out = vec![0.0f32; (b - a) as usize * cols];
                self.eval_desc_into(d, a, b, input, &mut out);
                Rows::Own(out, cols)
            }
        }
    }

    /// A descriptor written straight into the rows that will hold it.
    fn eval_desc_into(&self, d: &Desc, a: i64, b: i64, input: &InputView, out: &mut [f32]) {
        match d {
            Desc::Ref(_) => {
                let x = self.eval_desc(d, a, b, input);
                out.copy_from_slice(x.data());
            }
            Desc::Offset(x, n) => self.eval_desc_into(x, a + *n as i64, b + *n as i64, input, out),
            Desc::Scale(s, x) => {
                let x = self.eval_desc(x, a, b, input);
                for (o, v) in out.iter_mut().zip(x.data()) {
                    *o = *v * *s;
                }
            }
            // the skip connections: one pass over both operands rather than a scaled copy
            // and then a sum
            Desc::Sum(x, y) => {
                let q = self.eval_desc(y, a, b, input);
                if let Desc::Scale(s, xs) = &**x {
                    let p = self.eval_desc(xs, a, b, input);
                    for ((o, u), v) in out.iter_mut().zip(p.data()).zip(q.data()) {
                        *o = *u * *s + *v;
                    }
                } else {
                    let p = self.eval_desc(x, a, b, input);
                    for ((o, u), v) in out.iter_mut().zip(p.data()).zip(q.data()) {
                        *o = *u + *v;
                    }
                }
            }
            Desc::Append(parts) => {
                let rows = (b - a) as usize;
                let cols = self.net.desc_dim(d);
                let mut off = 0;
                for p in parts {
                    let m = self.eval_desc(p, a, b, input);
                    let c = m.cols();
                    for i in 0..rows {
                        out[i * cols + off..i * cols + off + c].copy_from_slice(m.row(i));
                    }
                    off += c;
                }
            }
            Desc::ReplaceIndex(_) => {
                let dim = self.ivector.len();
                for row in out.chunks_exact_mut(dim) {
                    row.copy_from_slice(&self.ivector);
                }
            }
        }
    }

    /// Compute every row every node can now compute; the output node's new rows are returned
    /// with the frame index of the first, as one `[rows, output_dim]` block.
    pub fn advance(&mut self, input: &InputView, ivector: &[f32]) -> (i64, &[f32], usize) {
        let n = self.ivector.len();
        self.ivector.copy_from_slice(&ivector[..n]);
        let steps = std::mem::take(&mut self.steps);
        let mut scratch = std::mem::take(&mut self.scratch);
        let (mut out_first, mut out_rows) = (0i64, 0usize);
        for st in &steps {
            let desc = match &self.net.nodes[&st.name] {
                Node::Component { desc, .. } | Node::Output { desc } => desc,
                _ => continue,
            };
            let avail = self.desc_avail(desc, input) - st.hi;
            let next = if self.stores[st.store].started {
                self.stores[st.store].end()
            } else {
                self.node_first(&st.name, input)
            };
            if avail <= next {
                continue;
            }
            let nrows = (avail - next) as usize;
            let mut dst = std::mem::take(&mut self.stores[st.store]);
            if !dst.started {
                dst.started = true;
                dst.first = next;
                dst.dim = st.dim;
            }
            let keep = self.history;
            let start = dst.reserve(nrows, keep);
            let out = &mut dst.data[start * st.dim..(start + nrows) * st.dim];
            match st.comp.as_deref().and_then(|c| self.net.comps.get(c)) {
                Some(Comp::Tdnn { offsets, w, b }) => {
                    let x = self.eval_desc(desc, next + st.lo, avail + st.hi, input);
                    let inn = x.cols();
                    let sp = offsets.len();
                    scratch.clear();
                    scratch.resize(nrows * inn * sp, 0.0);
                    for i in 0..nrows {
                        for (k, &o) in offsets.iter().enumerate() {
                            let src = i + (o as i64 - st.lo) as usize;
                            scratch[i * inn * sp + k * inn..i * inn * sp + (k + 1) * inn]
                                .copy_from_slice(x.row(src));
                        }
                    }
                    let bias = if b.is_empty() { None } else { Some(&b[..]) };
                    gemm_abt(&scratch, nrows, inn * sp, &w.d, w.r, bias, out);
                }
                Some(Comp::Affine { w, b }) => {
                    let x = self.eval_desc(desc, next, avail, input);
                    gemm_abt(x.data(), nrows, x.cols(), &w.d, w.r, Some(b), out);
                }
                Some(Comp::Linear { w }) => {
                    let x = self.eval_desc(desc, next, avail, input);
                    gemm_abt(x.data(), nrows, x.cols(), &w.d, w.r, None, out);
                }
                Some(Comp::Relu) => {
                    let x = self.eval_desc(desc, next, avail, input);
                    for (o, v) in out.iter_mut().zip(x.data()) {
                        *o = v.max(0.0);
                    }
                }
                Some(Comp::BatchNorm { mean, scale, .. }) => {
                    let x = self.eval_desc(desc, next, avail, input);
                    let (mean, scale) = (&mean[..st.dim], &scale[..st.dim]);
                    for (o, v) in out
                        .chunks_exact_mut(st.dim)
                        .zip(x.data().chunks_exact(st.dim))
                    {
                        for ((o, v), (m, s)) in o.iter_mut().zip(v).zip(mean.iter().zip(scale)) {
                            *o = (*v - *m) * *s;
                        }
                    }
                }
                _ => self.eval_desc_into(desc, next, avail, input, out),
            }
            dst.len += nrows;
            self.stores[st.store] = dst;
            if st.store == self.out {
                out_first = next;
                out_rows = nrows;
            }
        }
        self.steps = steps;
        self.scratch = scratch;
        let cols = self.out_cols;
        if out_rows == 0 {
            return (out_first, &[], cols);
        }
        let rows = self.stores[self.out].rows(out_first, out_first + out_rows as i64);
        (out_first, rows, cols)
    }
}
