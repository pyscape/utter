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
        Mat { r, c, d: vec![0.0; r * c] }
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
    Affine { w: Mat, b: Vec<f32> },
    Linear { w: Mat },
    Tdnn { offsets: Vec<i32>, w: Mat, b: Vec<f32> },
    BatchNorm { mean: Vec<f32>, var: Vec<f32>, eps: f32, trms: f32 },
    Relu,
    Identity,
}

pub enum Desc {
    Ref(String),
    Offset(Box<Desc>, i32),
    Scale(f32, Box<Desc>),
    Sum(Box<Desc>, Box<Desc>),
    Append(Vec<Desc>),
    ReplaceIndex(Box<Desc>),
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
    find(&buf[s..e], tok, 0).map(|off| KaldiReader::new(&buf[s + off + tok.len()..]).read_f32().unwrap())
}

impl Nnet3 {
    pub fn parse(buf: &[u8]) -> Nnet3 {
        let comps = parse_components(buf);
        let (nodes, input_dim, ivector_dim) = parse_graph(buf);
        let mut net = Nnet3 { comps, nodes, ivector_dim, input_dim, output_dim: 0, context: (0, 0) };
        net.output_dim = match net.comps.get("output.affine") {
            Some(Comp::Affine { w, .. }) => w.r,
            _ => 0,
        };
        net.context = net.node_context("output");
        net
    }

    fn node_context(&self, name: &str) -> (usize, usize) {
        match self.nodes.get(name) {
            None | Some(Node::Input) => (0, 0),
            Some(Node::DimRange { src, .. }) => self.node_context(src),
            Some(Node::Component { comp, desc }) => {
                let (l, r) = self.desc_context(desc);
                match self.comps.get(comp) {
                    Some(Comp::Tdnn { offsets, .. }) => {
                        let lo = offsets.iter().copied().min().unwrap_or(0).min(0).unsigned_abs() as usize;
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
            Desc::Append(parts) => parts.iter().map(|p| self.desc_context(p)).fold((0, 0), |(l, r), (l2, r2)| (l.max(l2), r.max(r2))),
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
        let rows = (out.r + subsampling - 1) / subsampling;
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
        let r = match self.nodes.get(name).unwrap_or_else(|| panic!("no node {name}")) {
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
        match self.comps.get(comp).unwrap_or_else(|| panic!("no comp {comp}")) {
            Comp::Identity => x,
            Comp::Relu => {
                let mut m = x;
                m.d.iter_mut().for_each(|v| *v = v.max(0.0));
                m
            }
            Comp::BatchNorm { mean, var, eps, trms } => {
                let mut m = x;
                let scale: Vec<f32> = var.iter().map(|v| trms / (v + eps).sqrt()).collect();
                for i in 0..m.r {
                    let row = &mut m.d[i * m.c..(i + 1) * m.c];
                    for j in 0..m.c {
                        row[j] = (row[j] - mean[j]) * scale[j];
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
                    let shifted = clamp_offset(&x, o);
                    for i in 0..x.r {
                        spliced.d[i * inn * sp + k * inn..i * inn * sp + k * inn + inn].copy_from_slice(shifted.row(i));
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
            "<NaturalGradientAffineComponent>" | "<FixedAffineComponent>" | "<AffineComponent>" => Comp::Affine {
                w: named_mat(buf, s, e, b"<LinearParams> ").unwrap(),
                b: named_vec(buf, s, e, b"<BiasParams> ").unwrap_or_default(),
            },
            "<LinearComponent>" => Comp::Linear { w: named_mat(buf, s, e, b"<Params> ").unwrap() },
            "<TdnnComponent>" => {
                let off = find(&buf[s..e], b"<TimeOffsets> ", 0).unwrap() + s + b"<TimeOffsets> ".len();
                let offsets = KaldiReader::new(&buf[off..]).read_i32_vec().unwrap();
                Comp::Tdnn {
                    offsets,
                    w: named_mat(buf, s, e, b"<LinearParams> ").unwrap(),
                    b: named_vec(buf, s, e, b"<BiasParams> ").unwrap_or_default(),
                }
            }
            "<BatchNormComponent>" => Comp::BatchNorm {
                mean: named_vec(buf, s, e, b"<StatsMean> ").unwrap(),
                var: named_vec(buf, s, e, b"<StatsVar> ").unwrap(),
                eps: named_f32(buf, s, e, b"<Epsilon> ").unwrap_or(1e-3),
                trms: named_f32(buf, s, e, b"<TargetRms> ").unwrap_or(1.0),
            },
            "<RectifiedLinearComponent>" => Comp::Relu,
            _ => Comp::Identity,
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
            nodes.insert(name, Node::Component { comp, desc: parse_desc(desc.trim()) });
        } else if let Some(rest) = line.strip_prefix("output-node") {
            let name = kv(rest, "name").unwrap().to_string();
            let di = line.find("input=").unwrap() + 6;
            let desc = &line[di..line[di..].find(" objective=").map(|x| x + di).unwrap_or(line.len())];
            nodes.insert(name, Node::Output { desc: parse_desc(desc.trim()) });
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
    let is_op = matches!(head.as_str(), "Offset" | "Scale" | "Sum" | "Append" | "ReplaceIndex" | "Round" | "IfDefined");
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
