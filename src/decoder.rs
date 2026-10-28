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

/// A history a merge overwrote, and what it cost over the path that won. From the merge on the
/// two share one future, so the differences hold for every extension of the winner.
struct Kept {
    link: Option<Rc<Link>>,
    /// The record the losing arc would have added on top of `link`, held here rather than
    /// allocated: its word and the phone it starts, 0 for none.
    word: Label,
    starts: i32,
    cost: f32,
    graph: f32,
    /// The phone it was in at the merge, where the winner's may differ.
    phone: i32,
}

/// `[[rr:TD-18#A merge keeps what it overwrites within the margin]]`
struct Join {
    /// Held so that the slot stays this join's while any path still runs through it.
    node: Rc<Link>,
    /// The winner's phone at the merge.
    phone: i32,
    kept: Kept,
}

/// The joins by the slot their record names; a slot is reused once no path runs through its
/// record.
#[derive(Default)]
struct Joins {
    slots: Vec<Option<Join>>,
    free: Vec<u32>,
}

impl Joins {
    fn is_empty(&self) -> bool {
        self.slots.len() == self.free.len()
    }

    fn clear(&mut self) {
        self.slots.clear();
        self.free.clear();
    }

    /// The join a record's `join` names.
    fn get(&self, join: u32) -> &Join {
        self.slots[join as usize - 1]
            .as_ref()
            .expect("a record names a live join")
    }

    /// Put a join on `won`'s path that keeps `kept`.
    fn add(&mut self, won: &mut Token, kept: Kept, frame: u32) {
        let slot = self.free.pop().map_or(self.slots.len(), |s| s as usize);
        let node = Rc::new(Link {
            prev: won.link.take(),
            frame,
            word: 0,
            phone: 0,
            join: u32::try_from(slot + 1).expect("joins fit u32"),
        });
        let join = Some(Join {
            node: node.clone(),
            phone: won.phone,
            kept,
        });
        if slot == self.slots.len() {
            self.slots.push(join);
        } else {
            self.slots[slot] = join;
        }
        won.link = Some(node);
    }

    /// Free the joins only this map still holds.
    fn sweep(&mut self) {
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if slot
                .as_ref()
                .is_some_and(|j| Rc::strong_count(&j.node) == 1)
            {
                *slot = None;
                self.free.push(u32::try_from(i).expect("joins fit u32"));
            }
        }
    }
}

/// A path from a join down to the start: its words as a hash and a count, and what it costs
/// over the path the join sits on. It takes the join's kept history or its own path down to
/// the path `at` in the list of the next join, so its route and its words are read back only if
/// it is listed.
#[derive(Clone, Copy)]
struct Below {
    hash: u64,
    len: usize,
    cost: f32,
    graph: f32,
    takes_kept: bool,
    next: u32,
    at: u32,
}

const START: Below = Below {
    hash: 0,
    len: 0,
    cost: 0.0,
    graph: 0.0,
    takes_kept: false,
    next: 0,
    at: 0,
};

/// The stretch of a path from a join down to the next join, 0 at the start: the hash of its
/// words and the factor that shifts a hash past them, and what it costs.
#[derive(Clone, Copy)]
struct Way {
    next: u32,
    hash: u64,
    shift: u64,
    len: usize,
    cost: f32,
    graph: f32,
    takes_kept: bool,
}

/// A word sequence's hash is the polynomial of its labels, so the hash of a sequence joined to
/// another is the first's shifted past the second, plus the second's.
const HASH_BASE: u64 = 0x9e37_79b9_7f4a_7c15;

/// The hash of words given newest first, and the factor that shifts a hash past them.
fn hash_newest_first(words: impl Iterator<Item = Label>) -> (u64, u64) {
    words.fold((0, 1), |(h, shift), w| {
        (
            h.wrapping_add(u64::from(w.cast_unsigned()).wrapping_mul(shift)),
            shift.wrapping_mul(HASH_BASE),
        )
    })
}

pub struct Link {
    pub prev: Option<Rc<Link>>,
    /// Output frame this record was created on: the frame an emitting arc consumed, or the
    /// frame count at a non-emitting arc.
    pub frame: u32,
    /// Word emitted here, 0 for none.
    pub word: Label,
    /// Phone that starts here, 0 for no phone change.
    pub phone: i32,
    /// A join's slot in `Decoder::joins` plus one, 0 for every other record.
    join: u32,
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

/// The frame a link records. 2^32 frames is over a year of audio in one utterance.
fn frame_id(n: usize) -> u32 {
    u32::try_from(n).expect("frame count exceeds u32")
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

impl Path {
    /// Kaldi's `TrailingSilenceLength`.
    pub fn trailing_silence_frames(&self, silence_phones: &[i32]) -> usize {
        let mut n = 0;
        for seg in self.phones.iter().rev() {
            if silence_phones.contains(&seg.phone) {
                n += seg.end - seg.start;
            } else {
                break;
            }
        }
        n
    }
}

/// One reading, with the cheapest token that carries it.
/// `[[rr:TD-9#Every reading carries its lead's motion]]`
/// `rank` is what the readings are ordered by, `cost` what a traced path reports; without
/// final costs they are equal.
pub struct Group<'t> {
    pub words: Vec<Label>,
    pub rank: f32,
    pub cost: f32,
    pub token: &'t Token,
    /// The joins whose kept history the reading's path takes, empty for the token's own path.
    pub route: Vec<u32>,
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
    /// Off by default: it walks both chains where the costs are close.
    /// `[[rr:TD-9#No lattice is added for this feature]]`
    pub census: bool,
    /// The margin every close count of the census is taken at.
    pub census_nats: f32,
    /// Collisions at a state where the two chains' word sequences differ and their costs are
    /// within `census_nats`. The loser is dropped whichever way the comparison goes, so both
    /// directions count. A collision is not a reading lost: the loser's sequence may survive
    /// on another token, and neither path need be near the leader.
    pub merges_close: u64,
    /// Collisions of those whose loser was within `census_nats` of the frame's best token,
    /// counted at the merge, whatever becomes of the loser's other copies.
    pub overwritten_close: u64,
    /// The word sequences those collisions overwrote this utterance, with the first frame each
    /// was overwritten on.
    pub overwritten: HashMap<Vec<Label>, u32>,
    census_losers: Vec<(f32, Vec<Label>)>,
    /// `[[rr:TD-18#A merge keeps what it overwrites within the margin]]`
    keep: Option<f32>,
    joins: Joins,
    scratch: (Vec<Label>, Vec<Label>),
    requeued: Vec<StateId>,
}

pub const CENSUS_NATS: f32 = 2.0;

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
            census: false,
            census_nats: CENSUS_NATS,
            merges_close: 0,
            overwritten_close: 0,
            overwritten: HashMap::new(),
            census_losers: Vec::new(),
            keep: None,
            joins: Joins::default(),
            scratch: (Vec::new(), Vec::new()),
            requeued: Vec::new(),
        };
        d.init_decoding();
        d
    }

    /// Keep the histories a merge overwrites within `margin` nats of the path that wins it, and
    /// read them out beside the survivors. `None` stops both; what is kept already goes as its
    /// paths end.
    /// `[[rr:TD-18#A merge keeps what it overwrites within the margin]]`
    pub fn set_keep(&mut self, margin: Option<f32>) {
        self.keep = margin;
    }

    pub fn init_decoding(&mut self) {
        self.cur.clear();
        self.joins.clear();
        self.overwritten.clear();
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
        if !self.census_losers.is_empty() {
            self.count_overwritten();
        }
        // A join no path runs through any more is held by the map alone.
        if !self.joins.is_empty() && self.num_frames_decoded.is_multiple_of(8) {
            self.joins.sweep();
        }
        self.num_frames_decoded += 1;
    }

    fn count_overwritten(&mut self) {
        let best = self.cur.values().fold(f32::INFINITY, |b, t| b.min(t.cost));
        let frame = frame_id(self.num_frames_decoded);
        for (cost, words) in self.census_losers.drain(..) {
            if cost - best < self.census_nats {
                self.overwritten_close += 1;
                self.overwritten.entry(words).or_insert(frame);
            }
        }
    }

    /// Each pass is compiled with and without the margin. `[[rr:TD-18#What it costs]]`
    fn process_emitting(&mut self, loglikes: &[f32]) -> f32 {
        if self.keep.is_some() {
            self.emitting::<true>(loglikes)
        } else {
            self.emitting::<false>(loglikes)
        }
    }

    fn process_nonemitting(&mut self, cutoff: f32) {
        if self.keep.is_some() {
            self.nonemitting::<true>(cutoff);
        } else {
            self.nonemitting::<false>(cutoff);
        }
    }

    // Transition ids and pdf ids are non-negative; a negative one indexes past the slice and
    // panics either way.
    #[allow(clippy::cast_sign_loss)]
    fn emitting<const KEEP: bool>(&mut self, loglikes: &[f32]) -> f32 {
        let frame = frame_id(self.num_frames_decoded);
        let margin = self.keep.unwrap_or(0.0);
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
                let (better, seq, held) = match next.get(&a.nextstate) {
                    Some(t) => {
                        if self.census && (t.cost - tot).abs() < self.census_nats {
                            let mut mine = Self::words_on(tok.link.as_deref());
                            if a.olabel != 0 {
                                mine.push(a.olabel);
                            }
                            let theirs = Self::words_on(t.link.as_deref());
                            if mine != theirs {
                                self.merges_close += 1;
                                self.census_losers.push(if tot < t.cost {
                                    (t.cost, theirs)
                                } else {
                                    (tot, mine)
                                });
                            }
                        }
                        (tot < t.cost, t.seq, t.cost)
                    }
                    None => {
                        self.seq += 1;
                        (true, self.seq, f32::INFINITY)
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
                            join: 0,
                        }))
                    } else {
                        tok.link.clone()
                    };
                    let won = Token {
                        cost: tot,
                        graph: tok.graph + a.weight,
                        link,
                        phone,
                        seq,
                    };
                    if !KEEP {
                        next.insert(a.nextstate, won);
                    } else if let Some(lost) = next.insert(a.nextstate, won) {
                        if held - tot <= margin {
                            let won = next.get_mut(&a.nextstate).expect("the winner was inserted");
                            keep_lost(&mut self.joins, &mut self.scratch, margin, won, lost, frame);
                        }
                    }
                } else if KEEP && tot - held <= margin {
                    let won = next.get_mut(&a.nextstate).expect("the state holds a token");
                    let arc = (a.olabel, self.tid2phone[a.ilabel as usize], a.weight);
                    let joins = (&mut self.joins, &mut self.scratch);
                    keep_arrival(joins, margin, won, tok, arc, tot, frame);
                }
            }
        }
        self.cur = next;
        next_cutoff
    }

    #[allow(clippy::cast_sign_loss)]
    fn nonemitting<const KEEP: bool>(&mut self, cutoff: f32) {
        let frame = frame_id(self.num_frames_decoded);
        let margin = self.keep.unwrap_or(0.0);
        self.queue.clear();
        self.requeued.clear();
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
                let (better, seq, held) = match self.cur.get(&a.nextstate) {
                    Some(t) => {
                        if self.census && (t.cost - tot).abs() < self.census_nats {
                            let mut mine = Self::words_on(tok.link.as_deref());
                            if a.olabel != 0 {
                                mine.push(a.olabel);
                            }
                            let theirs = Self::words_on(t.link.as_deref());
                            if mine != theirs {
                                self.merges_close += 1;
                                self.census_losers.push(if tot < t.cost {
                                    (t.cost, theirs)
                                } else {
                                    (tot, mine)
                                });
                            }
                        }
                        (tot < t.cost, t.seq, t.cost)
                    }
                    None => {
                        self.seq += 1;
                        (true, self.seq, f32::INFINITY)
                    }
                };
                if better {
                    let link = if a.olabel != 0 {
                        Some(Rc::new(Link {
                            prev: tok.link.clone(),
                            frame,
                            word: a.olabel,
                            phone: 0,
                            join: 0,
                        }))
                    } else {
                        tok.link.clone()
                    };
                    let won = Token {
                        cost: tot,
                        graph: tok.graph + a.weight,
                        link,
                        phone: tok.phone,
                        seq,
                    };
                    if !KEEP {
                        self.cur.insert(a.nextstate, won);
                    } else if let Some(lost) = self.cur.insert(a.nextstate, won) {
                        if held - tot <= margin {
                            let won = self
                                .cur
                                .get_mut(&a.nextstate)
                                .expect("the winner was inserted");
                            keep_lost(&mut self.joins, &mut self.scratch, margin, won, lost, frame);
                        }
                    }
                    self.queue.push(a.nextstate);
                } else if KEEP && tot - held <= margin {
                    let won = self
                        .cur
                        .get_mut(&a.nextstate)
                        .expect("the state holds a token");
                    // An epsilon arc changes no phone.
                    let arc = (a.olabel, tok.phone, a.weight);
                    let joins = (&mut self.joins, &mut self.scratch);
                    // The winner may already have passed its path on; once more carries the
                    // join after it. Only once, or a cycle would pass joins round for ever.
                    if keep_arrival(joins, margin, won, &tok, arc, tot, frame)
                        && self.has_eps[a.nextstate as usize]
                        && !self.requeued.contains(&a.nextstate)
                    {
                        self.requeued.push(a.nextstate);
                        self.queue.push(a.nextstate);
                    }
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

    fn words_on(link: Option<&Link>) -> Vec<Label> {
        let mut words = Vec::new();
        let mut l = link;
        while let Some(r) = l {
            if r.word != 0 {
                words.push(r.word);
            }
            l = r.prev.as_deref();
        }
        words.reverse();
        words
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

    /// Surviving tokens grouped by word sequence, cheapest first; each group carries its
    /// cheapest token. With `use_final`, final costs are added as for `best_token`.
    /// `[[rr:TD-9#Readings are read once per decoding advance]]`
    pub fn grouped(&self, use_final: bool) -> Vec<Group<'_>> {
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
        let mut v: Vec<Group<'_>> = groups
            .into_iter()
            .map(|(words, (rank, cost, token))| Group {
                words,
                rank,
                cost,
                token,
                route: Vec::new(),
            })
            .collect();
        v.sort_by(|a, b| {
            a.rank
                .partial_cmp(&b.rank)
                .unwrap()
                .then(b.token.seq.cmp(&a.token.seq))
        });
        v
    }

    /// [`grouped`](Self::grouped) with the readings kept at joins beside the survivors', as many
    /// of them as could be among the first `listed`. The survivors' first group stays first,
    /// read off its own token; every other reading takes its cheapest copy, kept or surviving.
    /// `[[rr:TD-18#Kept readings are reported as readings]]`
    pub fn readings(&self, use_final: bool, listed: usize) -> Vec<Group<'_>> {
        let Some(margin) = self.keep else {
            return self.grouped(use_final);
        };
        if self.joins.is_empty() || listed == 0 {
            return self.grouped(use_final);
        }
        let any_final = use_final && self.cur.keys().any(|&s| self.fst.is_final(s));
        // Tokens whose paths reach the same join with the same words above it read the same
        // below it, so only the cheapest of them is expanded.
        let mut heads: HashMap<(u32, Vec<Label>), (f32, f32, &Token)> = HashMap::new();
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
            let rank = if any_final {
                c - (1.0 - GRAPH_SCALE) * (t.graph + fw)
            } else {
                c
            };
            let (join, above) = self.down_to_join(t.link.as_deref());
            match heads.entry((join, above)) {
                std::collections::hash_map::Entry::Occupied(mut e) => {
                    let g = e.get_mut();
                    if !(g.0 < rank || (g.0 == rank && g.2.seq >= t.seq)) {
                        *g = (rank, c, t);
                    }
                }
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert((rank, c, t));
                }
            }
        }
        // The survivors' readings, chosen as `grouped` chooses them.
        let mut own_words: StateMap<Vec<Label>> = StateMap::default();
        let mut survivors: HashMap<Vec<Label>, Group<'_>> = HashMap::new();
        for (&(join, ref above), &(rank, cost, token)) in &heads {
            let mut words = own_words
                .entry(join)
                .or_insert_with(|| self.own_words(join))
                .clone();
            words.extend(above.iter().rev());
            match survivors.get_mut(&words) {
                Some(g) if g.rank < rank || (g.rank == rank && g.token.seq >= token.seq) => {}
                Some(g) => {
                    *g = Group {
                        words: Vec::new(),
                        rank,
                        cost,
                        token,
                        route: Vec::new(),
                    };
                }
                None => {
                    survivors.insert(
                        words,
                        Group {
                            words: Vec::new(),
                            rank,
                            cost,
                            token,
                            route: Vec::new(),
                        },
                    );
                }
            }
        }
        let mut own: Vec<Group<'_>> = survivors
            .into_iter()
            .map(|(words, g)| Group { words, ..g })
            .collect();
        own.sort_by(|a, b| {
            a.rank
                .partial_cmp(&b.rank)
                .unwrap()
                .then(b.token.seq.cmp(&a.token.seq))
        });
        // A reading ranked past the listed-th survivor cannot be listed. On a partial a kept
        // reading costs more than the path it rides on, so neither can any kept below that path.
        let bound = own.get(listed - 1).map_or(f32::INFINITY, |g| g.rank);
        let mut below: StateMap<Vec<Below>> = StateMap::default();
        // Per word sequence, by hash, the cheapest kept copy: its rank and cost, its token, and
        // where in the lists below its token's first join its path is.
        let mut kept: HashMap<(u64, usize), (f32, f32, &Token, u32, u32)> = HashMap::new();
        for (&(join, ref above), &(rank, cost, token)) in &heads {
            if join == 0 || (!any_final && rank >= bound) {
                continue;
            }
            self.fill_below(join, margin, listed + 1, &mut below);
            let (above_hash, shift) = hash_newest_first(above.iter().copied());
            for (i, b) in below[&join].iter().enumerate().skip(1) {
                let scaled = if any_final {
                    (1.0 - GRAPH_SCALE) * b.graph
                } else {
                    0.0
                };
                let r = rank + b.cost - scaled;
                if r >= bound {
                    continue;
                }
                let key = (
                    b.hash.wrapping_mul(shift).wrapping_add(above_hash),
                    b.len + above.len(),
                );
                let at = u32::try_from(i).expect("paths below a join fit u32");
                match kept.get_mut(&key) {
                    Some(k) if k.0 < r || (k.0 == r && k.2.seq >= token.seq) => {}
                    Some(k) => *k = (r, cost + b.cost, token, join, at),
                    None => {
                        kept.insert(key, (r, cost + b.cost, token, join, at));
                    }
                }
            }
        }
        let mut own = own.into_iter();
        let mut v: Vec<Group<'_>> = own.next().into_iter().collect();
        let mut rest: HashMap<Vec<Label>, Group<'_>> = own.map(|g| (g.words.clone(), g)).collect();
        for (rank, cost, token, join, at) in kept.into_values() {
            let route = Self::route_of(&below, join, at);
            let mut g = Group {
                words: Vec::new(),
                rank,
                cost,
                token,
                route,
            };
            g.words = self.route_words(g.token, &g.route);
            if v.first().is_some_and(|f| f.words == g.words) {
                continue;
            }
            match rest.get_mut(&g.words) {
                Some(r) if read_order(r, &g).is_lt() => {}
                Some(r) => *r = g,
                None => {
                    rest.insert(g.words.clone(), g);
                }
            }
        }
        let mut rest: Vec<Group<'_>> = rest.into_values().collect();
        rest.sort_by(read_order);
        v.extend(rest);
        v
    }

    /// The words of a token's path along `route`, from the start.
    fn route_words(&self, tok: &Token, route: &[u32]) -> Vec<Label> {
        let mut words = Vec::new();
        let mut l = tok.link.as_deref();
        while let Some(r) = l {
            if r.word != 0 {
                words.push(r.word);
            }
            l = if r.join != 0 && route.contains(&r.join) {
                let kept = &self.joins.get(r.join).kept;
                if kept.word != 0 {
                    words.push(kept.word);
                }
                kept.link.as_deref()
            } else {
                r.prev.as_deref()
            };
        }
        words.reverse();
        words
    }

    /// The words of the path a join sits on, from the start to the join.
    fn own_words(&self, join: u32) -> Vec<Label> {
        let mut words = Vec::new();
        if join != 0 {
            let mut l = self.joins.get(join).node.prev.as_deref();
            while let Some(r) = l {
                if r.word != 0 {
                    words.push(r.word);
                }
                l = r.prev.as_deref();
            }
        }
        words.reverse();
        words
    }

    /// The `cap` cheapest paths from `root` down to the start through the joins beneath it, the
    /// join's own path first, for it and every join on the way.
    fn fill_below(&self, root: u32, margin: f32, cap: usize, memo: &mut StateMap<Vec<Below>>) {
        let mut pending: StateMap<[Option<Way>; 2]> = StateMap::default();
        let mut stack = vec![root];
        while let Some(&j) = stack.last() {
            if memo.contains_key(&j) {
                stack.pop();
                continue;
            }
            let ways = pending.remove(&j).unwrap_or_else(|| {
                let join = self.joins.get(j);
                let k = &join.kept;
                [
                    Some(self.way(0, join.node.prev.as_deref(), 0.0, 0.0, false)),
                    (k.cost <= margin)
                        .then(|| self.way(k.word, k.link.as_deref(), k.cost, k.graph, true)),
                ]
            });
            let mut missing = false;
            for w in ways.iter().flatten() {
                if w.next != 0 && !memo.contains_key(&w.next) {
                    stack.push(w.next);
                    missing = true;
                }
            }
            if missing {
                pending.insert(j, ways);
                continue;
            }
            stack.pop();
            let mut out: Vec<Below> = Vec::new();
            for w in ways.iter().flatten() {
                let from: &[Below] = if w.next == 0 {
                    &[START]
                } else {
                    &memo[&w.next]
                };
                for (i, b) in from.iter().enumerate() {
                    let cost = w.cost + b.cost;
                    if cost > margin {
                        continue;
                    }
                    out.push(Below {
                        hash: b.hash.wrapping_mul(w.shift).wrapping_add(w.hash),
                        len: b.len + w.len,
                        cost,
                        graph: w.graph + b.graph,
                        takes_kept: w.takes_kept,
                        next: w.next,
                        at: u32::try_from(i).expect("paths below a join fit u32"),
                    });
                }
            }
            // Stable, so on a tie the join's own path, pushed first, stays first.
            out.sort_by(|a, b| a.cost.total_cmp(&b.cost));
            let mut cheapest: Vec<Below> = Vec::with_capacity(out.len().min(cap));
            for b in out {
                if cheapest.len() == cap {
                    break;
                }
                if !cheapest.iter().any(|c| c.hash == b.hash && c.len == b.len) {
                    cheapest.push(b);
                }
            }
            memo.insert(j, cheapest);
        }
    }

    /// The way from `l`, with `word` on top when that is not 0, down to the next join.
    fn way(
        &self,
        word: Label,
        mut l: Option<&Link>,
        cost: f32,
        graph: f32,
        takes_kept: bool,
    ) -> Way {
        let (mut hash, mut shift, mut len) = (0u64, 1u64, 0);
        let mut add = |w: Label| {
            hash = hash.wrapping_add(u64::from(w.cast_unsigned()).wrapping_mul(shift));
            shift = shift.wrapping_mul(HASH_BASE);
            len += 1;
        };
        if word != 0 {
            add(word);
        }
        let mut next = 0;
        while let Some(r) = l {
            if r.word != 0 {
                add(r.word);
            } else if r.join != 0 {
                next = r.join;
                break;
            }
            l = r.prev.as_deref();
        }
        Way {
            next,
            hash,
            shift,
            len,
            cost,
            graph,
            takes_kept,
        }
    }

    /// The joins whose kept history the path `at` in `join`'s list takes.
    fn route_of(memo: &StateMap<Vec<Below>>, mut join: u32, mut at: u32) -> Vec<u32> {
        let mut route = Vec::new();
        while join != 0 {
            let b = memo[&join][at as usize];
            if b.takes_kept {
                route.push(join);
            }
            (join, at) = (b.next, b.at);
        }
        route
    }

    /// The first join on a path from `l` down, 0 when there is none, with the words before it,
    /// newest first.
    fn down_to_join(&self, mut l: Option<&Link>) -> (u32, Vec<Label>) {
        let mut words = Vec::new();
        while let Some(r) = l {
            if r.word != 0 {
                words.push(r.word);
            } else if r.join != 0 {
                return (r.join, words);
            }
            l = r.prev.as_deref();
        }
        (0, words)
    }

    /// [`trace`](Self::trace) along a route through kept histories.
    fn trace_route(&self, tok: &Token, route: &[u32], cost: f32) -> Path {
        if route.is_empty() {
            return self.trace(tok, cost);
        }
        // (frame, word, phone), newest first
        let mut records: Vec<(u32, Label, i32)> = Vec::new();
        let mut l = tok.link.as_deref();
        while let Some(r) = l {
            if r.join != 0 && route.contains(&r.join) {
                let join = self.joins.get(r.join);
                let kept = &join.kept;
                // From the next frame the kept history is in whatever phone the winner's path
                // is in, which records no change there when it stays in its own.
                let resumes = r.frame + 1;
                let changes_there = records
                    .iter()
                    .rev()
                    .find(|x| x.2 != 0)
                    .is_some_and(|x| x.0 == resumes);
                if kept.phone != join.phone
                    && (resumes as usize) < self.num_frames_decoded
                    && !changes_there
                {
                    records.push((resumes, 0, join.phone));
                }
                if kept.word != 0 || kept.starts != 0 {
                    records.push((r.frame, kept.word, kept.starts));
                }
                l = kept.link.as_deref();
                continue;
            }
            records.push((r.frame, r.word, r.phone));
            l = r.prev.as_deref();
        }
        records.reverse();
        let mut path = Path {
            cost,
            ..Default::default()
        };
        let mut open: Option<(i32, usize)> = None;
        for &(frame, word, phone) in &records {
            // Where a kept history meets the winner's path both may be in one phone.
            if phone != 0 && open.map(|o| o.0) != Some(phone) {
                if let Some((p, start)) = open.take() {
                    path.phones.push(PhoneSegment {
                        phone: p,
                        start,
                        end: frame as usize,
                    });
                }
                open = Some((phone, frame as usize));
            }
            if word != 0 {
                path.words.push(word);
                path.word_frames.push(frame as usize);
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

    pub fn trace_groups(&self, groups: &[Group<'_>], max: usize) -> Vec<Path> {
        groups
            .iter()
            .take(max)
            .map(|g| self.trace_route(g.token, &g.route, g.cost))
            .collect()
    }

    pub fn alternatives(&self, use_final: bool, max: usize) -> Vec<Path> {
        let groups = self.grouped(use_final);
        self.trace_groups(&groups, max)
    }
}

/// The order readings are listed in: by rank, a surviving copy before a kept one, the newest
/// token first, and by words where one token's kept readings tie.
fn read_order(a: &Group<'_>, b: &Group<'_>) -> std::cmp::Ordering {
    a.rank
        .partial_cmp(&b.rank)
        .unwrap()
        .then(b.route.is_empty().cmp(&a.route.is_empty()))
        .then(b.token.seq.cmp(&a.token.seq))
        .then(a.words.cmp(&b.words))
}

/// Keep `lost`, the token `won` replaced at its state, if it is worth keeping.
#[cold]
#[inline(never)]
fn keep_lost(
    joins: &mut Joins,
    scratch: &mut (Vec<Label>, Vec<Label>),
    margin: f32,
    won: &mut Token,
    lost: Token,
    frame: u32,
) {
    let cost = lost.cost - won.cost;
    if worth_keeping(won, lost.link.as_deref(), 0, margin - cost, joins, scratch) {
        let kept = Kept {
            link: lost.link,
            word: 0,
            starts: 0,
            cost,
            graph: lost.graph - won.graph,
            phone: lost.phone,
        };
        joins.add(won, kept, frame);
    }
}

/// Keep the path that lost to `won` by taking `arc` (its word, the phone it is in, its weight)
/// from `tok` at a cost of `tot`, if it is worth keeping; whether it was kept.
#[cold]
#[inline(never)]
fn keep_arrival(
    (joins, scratch): (&mut Joins, &mut (Vec<Label>, Vec<Label>)),
    margin: f32,
    won: &mut Token,
    tok: &Token,
    (word, phone, weight): (Label, i32, f32),
    tot: f32,
    frame: u32,
) -> bool {
    let cost = tot - won.cost;
    if !worth_keeping(
        won,
        tok.link.as_deref(),
        word,
        margin - cost,
        joins,
        scratch,
    ) {
        return false;
    }
    let kept = Kept {
        link: tok.link.clone(),
        word,
        starts: if phone != tok.phone { phone } else { 0 },
        cost,
        graph: tok.graph + weight - won.graph,
        phone,
    };
    joins.add(won, kept, frame);
    true
}

/// Whether `lost`, with `word` on top when that is not 0, reads a different word sequence from
/// `won`'s path, or carries a reading kept within `slack` of its own above the record the two
/// share. Below that record they are one history, and it is not walked.
fn worth_keeping(
    won: &Token,
    lost: Option<&Link>,
    word: Label,
    slack: f32,
    joins: &Joins,
    scratch: &mut (Vec<Label>, Vec<Label>),
) -> bool {
    let (above_won, above_lost) = scratch;
    above_won.clear();
    above_lost.clear();
    if word != 0 {
        above_lost.push(word);
    }
    let (mut a, mut b) = (won.link.as_deref(), lost);
    loop {
        let step_won = match (a, b) {
            (None, None) => break,
            (Some(x), Some(y)) if std::ptr::eq(x, y) => break,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            // On one frame either may lie below the other, and the one below may be shared.
            (Some(x), Some(y)) => {
                x.frame > y.frame || (x.frame == y.frame && !in_frame_below(y, x))
            }
        };
        if step_won {
            let x = a.expect("stepped side has a record");
            if x.word != 0 {
                above_won.push(x.word);
            }
            a = x.prev.as_deref();
        } else {
            let y = b.expect("stepped side has a record");
            if y.word != 0 {
                above_lost.push(y.word);
            } else if y.join != 0 && joins.get(y.join).kept.cost <= slack {
                return true;
            }
            b = y.prev.as_deref();
        }
    }
    above_won != above_lost
}

/// Whether `x` is `y` or a record below it on `y`'s frame.
fn in_frame_below(y: &Link, x: &Link) -> bool {
    let mut l = Some(y);
    while let Some(r) = l {
        if r.frame != x.frame {
            return false;
        }
        if std::ptr::eq(r, x) {
            return true;
        }
        l = r.prev.as_deref();
    }
    false
}
