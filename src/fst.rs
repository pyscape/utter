//! Weighted finite-state transducers over the tropical semiring: the in-memory form the decoder
//! walks, a reader for OpenFst's binary header, `ConstFst` body, symbol tables and the
//! `olabel_lookahead` add-on, and connect.
// [[rr:TD-2]]

use crate::kaldi_io::err;
use std::collections::HashMap;
use std::io::Result;

pub type StateId = u32;
pub type Label = i32;
pub const NO_STATE: StateId = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Arc {
    pub ilabel: Label,
    pub olabel: Label,
    pub weight: f32,
    pub nextstate: StateId,
}

#[derive(Clone, Default)]
pub struct State {
    /// Final weight; infinity when the state is not final.
    pub final_weight: f32,
    pub arcs: Vec<Arc>,
}

#[derive(Clone)]
pub struct VectorFst {
    pub start: StateId,
    pub states: Vec<State>,
}

impl Default for VectorFst {
    fn default() -> Self {
        VectorFst { start: NO_STATE, states: Vec::new() }
    }
}

impl VectorFst {
    pub fn add_state(&mut self) -> StateId {
        self.states.push(State { final_weight: f32::INFINITY, arcs: Vec::new() });
        (self.states.len() - 1) as StateId
    }
    pub fn add_arc(&mut self, s: StateId, arc: Arc) {
        self.states[s as usize].arcs.push(arc);
    }
    pub fn set_final(&mut self, s: StateId, w: f32) {
        self.states[s as usize].final_weight = w;
    }
    pub fn is_final(&self, s: StateId) -> bool {
        self.states[s as usize].final_weight.is_finite()
    }
    pub fn num_states(&self) -> usize {
        self.states.len()
    }
    pub fn num_arcs(&self) -> usize {
        self.states.iter().map(|s| s.arcs.len()).sum()
    }

    /// Remove states not reachable from the start or not able to reach a final state.
    pub fn connect(&mut self) {
        let n = self.states.len();
        if self.start == NO_STATE || n == 0 {
            self.states.clear();
            self.start = NO_STATE;
            return;
        }
        let mut accessible = vec![false; n];
        let mut stack = vec![self.start];
        accessible[self.start as usize] = true;
        while let Some(s) = stack.pop() {
            for a in &self.states[s as usize].arcs {
                if !accessible[a.nextstate as usize] {
                    accessible[a.nextstate as usize] = true;
                    stack.push(a.nextstate);
                }
            }
        }
        let mut preds: Vec<Vec<StateId>> = vec![Vec::new(); n];
        for (s, st) in self.states.iter().enumerate() {
            for a in &st.arcs {
                preds[a.nextstate as usize].push(s as StateId);
            }
        }
        let mut coaccessible = vec![false; n];
        let mut stack: Vec<StateId> =
            (0..n).filter(|&s| self.states[s].final_weight.is_finite()).map(|s| s as StateId).collect();
        for &s in &stack {
            coaccessible[s as usize] = true;
        }
        while let Some(s) = stack.pop() {
            for &p in &preds[s as usize] {
                if !coaccessible[p as usize] {
                    coaccessible[p as usize] = true;
                    stack.push(p);
                }
            }
        }
        let mut newid = vec![NO_STATE; n];
        let mut next = 0u32;
        for s in 0..n {
            if accessible[s] && coaccessible[s] {
                newid[s] = next;
                next += 1;
            }
        }
        let old = std::mem::take(&mut self.states);
        self.states.reserve(next as usize);
        for (s, mut st) in old.into_iter().enumerate() {
            if newid[s] == NO_STATE {
                continue;
            }
            st.arcs.retain(|a| newid[a.nextstate as usize] != NO_STATE);
            for a in st.arcs.iter_mut() {
                a.nextstate = newid[a.nextstate as usize];
            }
            self.states.push(st);
        }
        self.start = if newid[self.start as usize] == NO_STATE { NO_STATE } else { newid[self.start as usize] };
        if self.start == NO_STATE {
            self.states.clear();
        }
    }

    pub fn arc_sort_ilabel(&mut self) {
        for st in self.states.iter_mut() {
            st.arcs.sort_by_key(|a| a.ilabel);
        }
    }
}

pub const FST_MAGIC: i32 = 2125659606;
pub const SYMBOL_TABLE_MAGIC: i32 = 2125658996;
pub const ADD_ON_MAGIC: i32 = 446681434;
const FLAG_HAS_ISYMBOLS: i32 = 1;
const FLAG_HAS_OSYMBOLS: i32 = 2;
const FLAG_IS_ALIGNED: i32 = 4;
const FILE_ALIGN: usize = 16;

#[derive(Clone, Debug)]
pub struct FstHeader {
    pub fst_type: String,
    pub arc_type: String,
    pub version: i32,
    pub flags: i32,
    pub properties: u64,
    pub start: i64,
    pub num_states: i64,
    pub num_arcs: i64,
}

pub struct SymbolTable {
    pub name: String,
    pub symbols: Vec<(String, i64)>,
}

impl SymbolTable {
    /// `id -> symbol`, sized to the largest key plus one.
    pub fn id_to_symbol(&self) -> Vec<String> {
        let n = self.symbols.iter().map(|(_, k)| *k).max().unwrap_or(-1) + 1;
        let mut v = vec![String::new(); n.max(0) as usize];
        for (s, k) in &self.symbols {
            if *k >= 0 {
                v[*k as usize] = s.clone();
            }
        }
        v
    }
    pub fn symbol_to_id(&self) -> HashMap<String, i64> {
        self.symbols.iter().cloned().collect()
    }
    /// OpenFst's text form: `symbol id` per line.
    pub fn parse_text(txt: &str) -> SymbolTable {
        let mut symbols = Vec::new();
        for line in txt.lines() {
            let mut it = line.split_whitespace();
            if let (Some(w), Some(i)) = (it.next(), it.next()) {
                if let Ok(id) = i.parse::<i64>() {
                    symbols.push((w.to_string(), id));
                }
            }
        }
        SymbolTable { name: String::new(), symbols }
    }
}

/// The label reachability record `StdOLabelLookAheadFst` stores after its `ConstFst` body. The
/// relabeling map is only present when the writer kept it; Kaldi's graphs are written without
/// it, with the words table already in the relabeled space, so an empty map means identity.
pub struct LabelReachable {
    pub reach_input: bool,
    pub final_label: Label,
    /// Original output label -> the label the graph's arcs carry.
    pub label2index: HashMap<Label, Label>,
    /// Per state: the output labels reachable through output-epsilon paths, as half-open
    /// `[begin, end)` intervals; `final_label` stands in for a reachable final state.
    pub intervals: Vec<Vec<(Label, Label)>>,
}

pub struct FstFile {
    pub header: FstHeader,
    pub fst: VectorFst,
    pub isymbols: Option<SymbolTable>,
    pub osymbols: Option<SymbolTable>,
    pub addon: Option<LabelReachable>,
}

struct Cur<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.p + n > self.b.len() {
            return Err(err("unexpected end of FST file"));
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn bool(&mut self) -> Result<bool> {
        Ok(self.take(1)?[0] != 0)
    }
    fn string(&mut self) -> Result<String> {
        let n = self.i32()?;
        if n < 0 {
            return Err(err("negative string length"));
        }
        Ok(String::from_utf8_lossy(self.take(n as usize)?).into_owned())
    }
    fn align(&mut self) {
        let rem = self.p % FILE_ALIGN;
        if rem != 0 {
            self.p += FILE_ALIGN - rem;
        }
    }
}

fn read_header(c: &mut Cur) -> Result<FstHeader> {
    if c.i32()? != FST_MAGIC {
        return Err(err("bad FST magic"));
    }
    Ok(FstHeader {
        fst_type: c.string()?,
        arc_type: c.string()?,
        version: c.i32()?,
        flags: c.i32()?,
        properties: c.u64()?,
        start: c.i64()?,
        num_states: c.i64()?,
        num_arcs: c.i64()?,
    })
}

fn read_symbol_table(c: &mut Cur) -> Result<SymbolTable> {
    if c.i32()? != SYMBOL_TABLE_MAGIC {
        return Err(err("bad symbol table magic"));
    }
    let name = c.string()?;
    let _available_key = c.i64()?;
    let size = c.i64()?;
    let mut symbols = Vec::with_capacity(size.max(0) as usize);
    for _ in 0..size {
        let s = c.string()?;
        let k = c.i64()?;
        symbols.push((s, k));
    }
    Ok(SymbolTable { name, symbols })
}

fn read_const_body(c: &mut Cur, h: &FstHeader) -> Result<VectorFst> {
    let aligned = h.version == 1 || (h.flags & FLAG_IS_ALIGNED) != 0;
    if aligned {
        c.align();
    }
    let ns = h.num_states.max(0) as usize;
    let na = h.num_arcs.max(0) as usize;
    let mut fst = VectorFst { start: if h.start < 0 { NO_STATE } else { h.start as StateId }, states: Vec::with_capacity(ns) };
    let mut spans = Vec::with_capacity(ns);
    for _ in 0..ns {
        let w = c.f32()?;
        let pos = c.u32()? as usize;
        let narcs = c.u32()? as usize;
        let _niepsilons = c.u32()?;
        let _noepsilons = c.u32()?;
        spans.push((pos, narcs));
        fst.states.push(State { final_weight: w, arcs: Vec::with_capacity(narcs) });
    }
    if aligned {
        c.align();
    }
    let arcs_bytes = c.take(na * 16)?;
    for (s, &(pos, narcs)) in spans.iter().enumerate() {
        if pos + narcs > na {
            return Err(err("ConstFst arc span out of range"));
        }
        let st = &mut fst.states[s];
        for a in pos..pos + narcs {
            let b = &arcs_bytes[a * 16..a * 16 + 16];
            st.arcs.push(Arc {
                ilabel: i32::from_le_bytes(b[0..4].try_into().unwrap()),
                olabel: i32::from_le_bytes(b[4..8].try_into().unwrap()),
                weight: f32::from_le_bytes(b[8..12].try_into().unwrap()),
                nextstate: u32::from_le_bytes(b[12..16].try_into().unwrap()),
            });
        }
    }
    Ok(fst)
}

fn read_label_reachable(c: &mut Cur) -> Result<LabelReachable> {
    let reach_input = c.bool()?;
    let keep_relabel_data = c.bool()?;
    let mut label2index = HashMap::new();
    if keep_relabel_data {
        let n = c.i64()?;
        label2index.reserve(n.max(0) as usize);
        for _ in 0..n {
            let k = c.i32()?;
            let v = c.i32()?;
            label2index.insert(k, v);
        }
    }
    let final_label = c.i32()?;
    let num_interval_sets = c.i64()?.max(0) as usize;
    let mut intervals = Vec::with_capacity(num_interval_sets);
    for _ in 0..num_interval_sets {
        let m = c.i64()?.max(0) as usize;
        let mut set = Vec::with_capacity(m);
        for _ in 0..m {
            let begin = c.i32()?;
            let end = c.i32()?;
            set.push((begin, end));
        }
        let _count = c.i32()?;
        intervals.push(set);
    }
    Ok(LabelReachable { reach_input, final_label, label2index, intervals })
}

/// Read an OpenFst binary file: `const` and `vector` bodies, and `olabel_lookahead` wrapping a
/// `const` body. Any other type yields the header and symbol tables with an empty FST.
pub fn read_fst_bytes(bytes: &[u8]) -> Result<FstFile> {
    let mut c = Cur { b: bytes, p: 0 };
    let header = read_header(&mut c)?;
    let isymbols = if header.flags & FLAG_HAS_ISYMBOLS != 0 { Some(read_symbol_table(&mut c)?) } else { None };
    let osymbols = if header.flags & FLAG_HAS_OSYMBOLS != 0 { Some(read_symbol_table(&mut c)?) } else { None };
    if header.arc_type != "standard" {
        return Err(err(&format!("unsupported arc type {}", header.arc_type)));
    }
    let mut addon = None;
    let fst = match header.fst_type.as_str() {
        "const" => read_const_body(&mut c, &header)?,
        "vector" => read_vector_body(&mut c, &header)?,
        "olabel_lookahead" => {
            if c.i32()? != ADD_ON_MAGIC {
                return Err(err("bad add-on magic"));
            }
            let inner = read_header(&mut c)?;
            if inner.flags & FLAG_HAS_ISYMBOLS != 0 {
                read_symbol_table(&mut c)?;
            }
            if inner.flags & FLAG_HAS_OSYMBOLS != 0 {
                read_symbol_table(&mut c)?;
            }
            let fst = match inner.fst_type.as_str() {
                "const" => read_const_body(&mut c, &inner)?,
                "vector" => read_vector_body(&mut c, &inner)?,
                t => return Err(err(&format!("unsupported lookahead body type {t}"))),
            };
            // A MatcherFst add-on is an AddOnPair: data for matching on the input side, then
            // for the output side; an olabel_lookahead graph carries only the second.
            let have_addon = c.bool()?;
            if have_addon {
                let have_input_side = c.bool()?;
                if have_input_side {
                    addon = Some(read_label_reachable(&mut c)?);
                }
                let have_output_side = c.bool()?;
                if have_output_side {
                    addon = Some(read_label_reachable(&mut c)?);
                }
            }
            fst
        }
        _ => VectorFst::default(),
    };
    Ok(FstFile { header, fst, isymbols, osymbols, addon })
}

fn read_vector_body(c: &mut Cur, h: &FstHeader) -> Result<VectorFst> {
    let ns = h.num_states.max(0) as usize;
    let mut fst = VectorFst { start: if h.start < 0 { NO_STATE } else { h.start as StateId }, states: Vec::with_capacity(ns) };
    for _ in 0..ns {
        let w = c.f32()?;
        let narcs = c.i64()?.max(0) as usize;
        let mut arcs = Vec::with_capacity(narcs);
        for _ in 0..narcs {
            arcs.push(Arc { ilabel: c.i32()?, olabel: c.i32()?, weight: c.f32()?, nextstate: c.u32()? });
        }
        fst.states.push(State { final_weight: w, arcs });
    }
    Ok(fst)
}

pub fn read_fst_file(path: &std::path::Path) -> Result<FstFile> {
    read_fst_bytes(&std::fs::read(path)?)
}

/// Header and symbol tables only; the body is skipped.
pub fn read_fst_symbols(path: &std::path::Path) -> Result<(FstHeader, Option<SymbolTable>, Option<SymbolTable>)> {
    let bytes = std::fs::read(path)?;
    let mut c = Cur { b: &bytes, p: 0 };
    let header = read_header(&mut c)?;
    let isymbols = if header.flags & FLAG_HAS_ISYMBOLS != 0 { Some(read_symbol_table(&mut c)?) } else { None };
    let osymbols = if header.flags & FLAG_HAS_OSYMBOLS != 0 { Some(read_symbol_table(&mut c)?) } else { None };
    Ok((header, isymbols, osymbols))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connect_drops_dead_ends() {
        let mut f = VectorFst::default();
        let s0 = f.add_state();
        let s1 = f.add_state();
        let s2 = f.add_state();
        let _dead = f.add_state();
        f.start = s0;
        f.add_arc(s0, Arc { ilabel: 1, olabel: 1, weight: 0.0, nextstate: s1 });
        f.add_arc(s0, Arc { ilabel: 2, olabel: 2, weight: 0.0, nextstate: 3 });
        f.add_arc(s1, Arc { ilabel: 3, olabel: 3, weight: 0.0, nextstate: s2 });
        f.set_final(s2, 0.0);
        f.connect();
        assert_eq!(f.num_states(), 3);
        assert_eq!(f.states[0].arcs.len(), 1);
    }
}
