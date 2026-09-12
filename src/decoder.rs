//! Frame-synchronous token passing with the pruning semantics of Kaldi's
//! `LatticeFasterOnlineDecoder`: `GetCutoff` with beam, max-active, min-active and beam-delta,
//! the adaptive beam and the per-frame cost offset, an emitting pass then a non-emitting pass
//! per frame. No lattice: each token carries a reference-counted chain of records, one per word
//! emitted and per phone change, from which partials, alignments, trailing silence and
//! alternatives are read.
// [[rr:TD-2#The decoder: search]]
// [[rr:TD-2#The decoder: partials]]
// [[rr:TD-2#The decoder: endpointing]]

use crate::fst::{Label, StateId, VectorFst, NO_STATE};
use std::collections::HashMap;

/// The token maps key on a graph state id. `HashMap`'s default hash costs more than the probe
/// it protects and its quality buys nothing for a dense integer key, so the two per-frame maps
/// use a multiply-rotate over the id instead.
///
/// Iteration order decides which tokens a narrowing cutoff prunes, so it can move a partial
/// where two readings tie. The default hash is seeded afresh every run, which made that
/// order, and so those partials, differ between runs of the same audio; a fixed hash is what
/// makes them repeatable.
#[derive(Default)]
struct StateHasher(u64);

impl std::hash::Hasher for StateHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }
    fn write_u32(&mut self, v: u32) {
        self.write_u64(v as u64);
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = (self.0.rotate_left(5) ^ v).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

type StateMap<V> = HashMap<StateId, V, std::hash::BuildHasherDefault<StateHasher>>;
use std::sync::Arc as Rc;
use std::sync::Arc;

pub struct Link {
    pub prev: Option<Rc<Link>>,
    /// Output frame this record was created on: the frame an emitting arc consumed, or the
    /// frame count at a non-emitting arc.
    pub frame: u32,
    /// Word emitted here, 0 for none.
    pub word: Label,
    /// Phone that starts here, 0 for no phone change.
    pub phone: i32,
}

/// libvosk scales the graph half of a final path's cost by this before choosing which path
/// to emit, so a token carries its graph cost apart from its total.
/// `[[rr:TD-6#Decision outcome]]`
pub const GRAPH_SCALE: f32 = 0.9;

#[derive(Clone)]
pub struct Token {
    pub cost: f32,
    /// The arc weights along this path, without the acoustic likelihoods.
    pub graph: f32,
    pub link: Option<Rc<Link>>,
    /// Phone of the last emitting arc on this path, for change detection.
    pub phone: i32,
    /// Creation order within the frame; ties in cost go to the newest token, as Kaldi's token
    /// list, iterated newest first with a strict comparison, resolves them.
    pub seq: u32,
}

#[derive(Clone, Debug)]
pub struct DecoderConfig {
    pub beam: f32,
    pub max_active: usize,
    pub min_active: usize,
    pub beam_delta: f32,
}

/// A phone segment on the best path: `[start, end)` in output frames.
#[derive(Clone, Debug, PartialEq)]
pub struct PhoneSegment {
    pub phone: i32,
    pub start: usize,
    pub end: usize,
}

/// Everything a traceback yields.
#[derive(Clone, Debug, Default)]
pub struct Path {
    pub words: Vec<Label>,
    /// Frame each word was emitted on, parallel to `words`.
    pub word_frames: Vec<usize>,
    pub phones: Vec<PhoneSegment>,
    pub cost: f32,
}

pub struct Decoder<'g> {
    fst: Arc<VectorFst>,
    pub config: DecoderConfig,
    tid2pdf: &'g [i32],
    tid2phone: &'g [i32],
    cur: StateMap<Token>,
    num_frames_decoded: usize,
    /// Per state: whether it has input-epsilon arcs.
    has_eps: Vec<bool>,
    tmp_costs: Vec<f32>,
    queue: Vec<StateId>,
    seq: u32,
}

impl<'g> Decoder<'g> {
    pub fn new(
        fst: Arc<VectorFst>,
        tid2pdf: &'g [i32],
        tid2phone: &'g [i32],
        config: DecoderConfig,
    ) -> Self {
        let has_eps = fst
            .states
            .iter()
            .map(|s| s.arcs.iter().any(|a| a.ilabel == 0))
            .collect();
        let mut d = Decoder {
            fst,
            config,
            tid2pdf,
            tid2phone,
            cur: StateMap::default(),
            num_frames_decoded: 0,
            has_eps,
            tmp_costs: Vec::new(),
            queue: Vec::new(),
            seq: 0,
        };
        d.init_decoding();
        d
    }

    pub fn init_decoding(&mut self) {
        self.cur.clear();
        self.num_frames_decoded = 0;
        if self.fst.start == NO_STATE {
            return;
        }
        self.seq = 0;
        self.cur.insert(
            self.fst.start,
            Token {
                cost: 0.0,
                graph: 0.0,
                link: None,
                phone: 0,
                seq: 0,
            },
        );
        let beam = self.config.beam;
        self.process_nonemitting(beam);
    }

    pub fn num_frames_decoded(&self) -> usize {
        self.num_frames_decoded
    }

    pub fn num_active(&self) -> usize {
        self.cur.len()
    }

    fn get_cutoff(&mut self) -> (f32, f32, Option<StateId>) {
        let c = &self.config;
        let mut best = f32::INFINITY;
        let mut best_state = None;
        self.tmp_costs.clear();
        for (&s, t) in &self.cur {
            self.tmp_costs.push(t.cost);
            if t.cost < best {
                best = t.cost;
                best_state = Some(s);
            }
        }
        let beam_cutoff = best + c.beam;
        let mut max_active_cutoff = f32::INFINITY;
        let n = self.tmp_costs.len();
        if n > c.max_active {
            self.tmp_costs
                .select_nth_unstable_by(c.max_active, |a, b| a.partial_cmp(b).unwrap());
            max_active_cutoff = self.tmp_costs[c.max_active];
        }
        if max_active_cutoff < beam_cutoff {
            return (
                max_active_cutoff,
                max_active_cutoff - best + c.beam_delta,
                best_state,
            );
        }
        let mut min_active_cutoff = f32::INFINITY;
        if n > c.min_active {
            if c.min_active == 0 {
                min_active_cutoff = best;
            } else {
                let end = if n > c.max_active { c.max_active } else { n };
                self.tmp_costs[..end]
                    .select_nth_unstable_by(c.min_active, |a, b| a.partial_cmp(b).unwrap());
                min_active_cutoff = self.tmp_costs[c.min_active];
            }
        }
        if min_active_cutoff > beam_cutoff {
            (
                min_active_cutoff,
                min_active_cutoff - best + c.beam_delta,
                best_state,
            )
        } else {
            (beam_cutoff, c.beam, best_state)
        }
    }

    /// Decode one frame from `loglikes` (acoustic scale already applied, indexed by pdf).
    pub fn advance_frame(&mut self, loglikes: &[f32]) {
        let next_cutoff = self.process_emitting(loglikes);
        self.process_nonemitting(next_cutoff);
        self.num_frames_decoded += 1;
    }

    fn process_emitting(&mut self, loglikes: &[f32]) -> f32 {
        let frame = self.num_frames_decoded as u32;
        let (cur_cutoff, adaptive_beam, best_state) = self.get_cutoff();
        let prev = std::mem::take(&mut self.cur);
        let mut next: StateMap<Token> =
            StateMap::with_capacity_and_hasher(prev.len() * 2, Default::default());
        let mut next_cutoff = f32::INFINITY;
        let mut cost_offset = 0.0f32;
        if let Some(bs) = best_state {
            let tok = &prev[&bs];
            cost_offset = -tok.cost;
            for a in &self.fst.states[bs as usize].arcs {
                if a.ilabel != 0 {
                    let w = a.weight + cost_offset
                        - loglikes[self.tid2pdf[a.ilabel as usize] as usize]
                        + tok.cost;
                    if w + adaptive_beam < next_cutoff {
                        next_cutoff = w + adaptive_beam;
                    }
                }
            }
        }
        for (&s, tok) in &prev {
            if tok.cost > cur_cutoff {
                continue;
            }
            for a in &self.fst.states[s as usize].arcs {
                if a.ilabel == 0 {
                    continue;
                }
                let ac_cost = cost_offset - loglikes[self.tid2pdf[a.ilabel as usize] as usize];
                let tot = tok.cost + ac_cost + a.weight;
                if tot >= next_cutoff {
                    continue;
                }
                if tot + adaptive_beam < next_cutoff {
                    next_cutoff = tot + adaptive_beam;
                }
                let (better, seq) = match next.get(&a.nextstate) {
                    Some(t) => (tot < t.cost, t.seq),
                    None => {
                        self.seq += 1;
                        (true, self.seq)
                    }
                };
                if better {
                    let phone = self.tid2phone[a.ilabel as usize];
                    let link = if a.olabel != 0 || phone != tok.phone {
                        Some(Rc::new(Link {
                            prev: tok.link.clone(),
                            frame,
                            word: a.olabel,
                            phone: if phone != tok.phone { phone } else { 0 },
                        }))
                    } else {
                        tok.link.clone()
                    };
                    next.insert(
                        a.nextstate,
                        Token {
                            cost: tot,
                            graph: tok.graph + a.weight,
                            link,
                            phone,
                            seq,
                        },
                    );
                }
            }
        }
        self.cur = next;
        next_cutoff
    }

    fn process_nonemitting(&mut self, cutoff: f32) {
        let frame = self.num_frames_decoded as u32;
        self.queue.clear();
        for &s in self.cur.keys() {
            if self.has_eps[s as usize] {
                self.queue.push(s);
            }
        }
        while let Some(s) = self.queue.pop() {
            let tok = self.cur[&s].clone();
            if tok.cost >= cutoff {
                continue;
            }
            for a in &self.fst.states[s as usize].arcs {
                if a.ilabel != 0 {
                    continue;
                }
                let tot = tok.cost + a.weight;
                if tot >= cutoff {
                    continue;
                }
                let (better, seq) = match self.cur.get(&a.nextstate) {
                    Some(t) => (tot < t.cost, t.seq),
                    None => {
                        self.seq += 1;
                        (true, self.seq)
                    }
                };
                if better {
                    let link = if a.olabel != 0 {
                        Some(Rc::new(Link {
                            prev: tok.link.clone(),
                            frame,
                            word: a.olabel,
                            phone: 0,
                        }))
                    } else {
                        tok.link.clone()
                    };
                    self.cur.insert(
                        a.nextstate,
                        Token {
                            cost: tot,
                            graph: tok.graph + a.weight,
                            link,
                            phone: tok.phone,
                            seq,
                        },
                    );
                    self.queue.push(a.nextstate);
                }
            }
        }
    }

    /// Best token with or without final costs, Kaldi's `BestPathEnd`: with final costs, a
    /// non-final token counts only when no token is final.
    pub fn best_token(&self, use_final: bool) -> Option<(&Token, f32)> {
        let any_final = use_final && self.cur.keys().any(|&s| self.fst.is_final(s));
        let mut best: Option<(&Token, f32, f32)> = None;
        for (&s, t) in &self.cur {
            let fw = if any_final {
                self.fst.states[s as usize].final_weight
            } else {
                0.0
            };
            let c = t.cost + fw;
            let rank = if any_final {
                c - (1.0 - GRAPH_SCALE) * (t.graph + fw)
            } else {
                c
            };
            if c.is_finite()
                && best
                    .map(|(bt, b, _)| rank < b || (rank == b && t.seq > bt.seq))
                    .unwrap_or(true)
            {
                best = Some((t, rank, c));
            }
        }
        best.map(|(t, _, c)| (t, c))
    }

    /// Kaldi's `FinalRelativeCost`: infinity when no active token is final.
    pub fn final_relative_cost(&self) -> f32 {
        let mut best = f32::INFINITY;
        let mut best_final = f32::INFINITY;
        for (&s, t) in &self.cur {
            best = best.min(t.cost);
            best_final = best_final.min(t.cost + self.fst.states[s as usize].final_weight);
        }
        if best.is_infinite() && best_final.is_infinite() {
            f32::INFINITY
        } else {
            best_final - best
        }
    }

    /// Traceback of a token's records into words and phone segments.
    pub fn trace(&self, tok: &Token, cost: f32) -> Path {
        let mut records: Vec<&Link> = Vec::new();
        let mut l = tok.link.as_deref();
        while let Some(r) = l {
            records.push(r);
            l = r.prev.as_deref();
        }
        records.reverse();
        let mut path = Path {
            cost,
            ..Default::default()
        };
        let mut open: Option<(i32, usize)> = None;
        for r in &records {
            if r.phone != 0 {
                if let Some((p, start)) = open.take() {
                    path.phones.push(PhoneSegment {
                        phone: p,
                        start,
                        end: r.frame as usize,
                    });
                }
                open = Some((r.phone, r.frame as usize));
            }
            if r.word != 0 {
                path.words.push(r.word);
                path.word_frames.push(r.frame as usize);
            }
        }
        if let Some((p, start)) = open {
            path.phones.push(PhoneSegment {
                phone: p,
                start,
                end: self.num_frames_decoded,
            });
        }
        path
    }

    pub fn best_path(&self, use_final: bool) -> Option<Path> {
        self.best_token(use_final).map(|(t, c)| self.trace(t, c))
    }

    /// Kaldi's `TrailingSilenceLength` on the best path without final costs.
    pub fn trailing_silence_frames(&self, silence_phones: &[i32]) -> usize {
        let Some(path) = self.best_path(false) else {
            return 0;
        };
        let mut n = 0;
        for seg in path.phones.iter().rev() {
            if silence_phones.contains(&seg.phone) {
                n += seg.end - seg.start;
            } else {
                break;
            }
        }
        n
    }

    /// Surviving tokens grouped by word sequence, cheapest first; each group carries its
    /// cheapest token's path. With `use_final`, final costs are added as for `best_token`.
    pub fn alternatives(&self, use_final: bool, max: usize) -> Vec<Path> {
        let any_final = use_final && self.cur.keys().any(|&s| self.fst.is_final(s));
        let mut groups: HashMap<Vec<Label>, (f32, f32, &Token)> = HashMap::new();
        for (&s, t) in &self.cur {
            let fw = if any_final {
                self.fst.states[s as usize].final_weight
            } else {
                0.0
            };
            let c = t.cost + fw;
            if !c.is_finite() {
                continue;
            }
            // Grouped and ranked at the scale the reading is chosen by, not at the raw cost.
            let rank = if any_final {
                c - (1.0 - GRAPH_SCALE) * (t.graph + fw)
            } else {
                c
            };
            let mut words = Vec::new();
            let mut l = t.link.as_deref();
            while let Some(r) = l {
                if r.word != 0 {
                    words.push(r.word);
                }
                l = r.prev.as_deref();
            }
            words.reverse();
            match groups.get_mut(&words) {
                Some(g) if g.0 < rank || (g.0 == rank && g.2.seq >= t.seq) => {}
                Some(g) => *g = (rank, c, t),
                None => {
                    groups.insert(words, (rank, c, t));
                }
            }
        }
        let mut v: Vec<(f32, f32, &Token)> = groups.into_values().collect();
        v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(b.2.seq.cmp(&a.2.seq)));
        v.truncate(max);
        v.into_iter().map(|(_, c, t)| self.trace(t, c)).collect()
    }
}
