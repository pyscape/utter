//! Batch best-path Viterbi over a composed graph, for gate G0: beam and max-active pruning, an
//! epsilon closure after every frame, final costs added at the end. The streaming decoder with
//! Kaldi's pruning semantics is a later milestone.

use crate::fst::{Label, StateId, VectorFst, NO_STATE};
use crate::nnet3::Mat;
use std::collections::HashMap;

#[derive(Clone, Copy)]
struct Tok {
    cost: f32,
    back: i32,
    word: Label,
}

pub struct BatchDecoder {
    pub beam: f32,
    pub acoustic_scale: f32,
    pub max_active: usize,
}

fn relax(
    arena: &mut Vec<Tok>,
    map: &mut HashMap<StateId, usize>,
    state: StateId,
    cost: f32,
    back: i32,
    word: Label,
) -> bool {
    match map.get(&state) {
        Some(&idx) if arena[idx].cost <= cost => false,
        _ => {
            let idx = arena.len();
            arena.push(Tok { cost, back, word });
            map.insert(state, idx);
            true
        }
    }
}

/// Back pointers are i32; an arena that large would not fit in memory.
fn tok_id(i: usize) -> i32 {
    i32::try_from(i).expect("token arena exceeds i32")
}

impl BatchDecoder {
    fn epsilon_closure(
        &self,
        fst: &VectorFst,
        arena: &mut Vec<Tok>,
        active: &mut HashMap<StateId, usize>,
    ) {
        let mut queue: Vec<StateId> = active.keys().copied().collect();
        while let Some(s) = queue.pop() {
            let Some(&ti) = active.get(&s) else { continue };
            let base = arena[ti].cost;
            let updates: Vec<(StateId, f32, Label)> = fst.states[s as usize]
                .arcs
                .iter()
                .filter(|a| a.ilabel == 0)
                .map(|a| (a.nextstate, base + a.weight, a.olabel))
                .collect();
            for (ns, cost, word) in updates {
                if relax(arena, active, ns, cost, tok_id(ti), word) {
                    queue.push(ns);
                }
            }
        }
    }

    fn cap_active(&self, arena: &[Tok], active: &mut HashMap<StateId, usize>) {
        if active.len() <= self.max_active {
            return;
        }
        let mut costs: Vec<f32> = active.values().map(|&i| arena[i].cost).collect();
        let k = self.max_active;
        costs.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap());
        let thresh = costs[k];
        active.retain(|_, &mut i| arena[i].cost < thresh);
    }

    /// Best word sequence and its cost with final weights; `loglikes` rows are output frames.
    // Transition ids and pdf ids are non-negative; a negative one indexes past the slice and
    // panics either way.
    #[allow(clippy::cast_sign_loss)]
    pub fn decode(&self, fst: &VectorFst, tid2pdf: &[i32], loglikes: &Mat) -> (Vec<Label>, f32) {
        if fst.start == NO_STATE {
            return (vec![], f32::INFINITY);
        }
        let mut arena: Vec<Tok> = vec![Tok {
            cost: 0.0,
            back: -1,
            word: 0,
        }];
        let mut active: HashMap<StateId, usize> = HashMap::new();
        active.insert(fst.start, 0);
        self.epsilon_closure(fst, &mut arena, &mut active);
        for t in 0..loglikes.r {
            let frame = loglikes.row(t);
            let mut next: HashMap<StateId, usize> = HashMap::new();
            let mut best = f32::INFINITY;
            for (&s, &ti) in active.iter() {
                let base = arena[ti].cost;
                for a in &fst.states[s as usize].arcs {
                    if a.ilabel == 0 {
                        continue;
                    }
                    let pdf = tid2pdf[a.ilabel as usize] as usize;
                    let cost = base + a.weight - frame[pdf] * self.acoustic_scale;
                    if cost < best {
                        best = cost;
                    }
                    relax(
                        &mut arena,
                        &mut next,
                        a.nextstate,
                        cost,
                        tok_id(ti),
                        a.olabel,
                    );
                }
            }
            let cutoff = best + self.beam;
            next.retain(|_, &mut ti| arena[ti].cost <= cutoff);
            active = next;
            self.epsilon_closure(fst, &mut arena, &mut active);
            self.cap_active(&arena, &mut active);
        }
        let mut best_ti: i32 = -1;
        let mut best_cost = f32::INFINITY;
        for (&s, &ti) in active.iter() {
            let fw = fst.states[s as usize].final_weight;
            if fw.is_finite() {
                let c = arena[ti].cost + fw;
                if c < best_cost {
                    best_cost = c;
                    best_ti = tok_id(ti);
                }
            }
        }
        if best_ti < 0 {
            for &ti in active.values() {
                if arena[ti].cost < best_cost {
                    best_cost = arena[ti].cost;
                    best_ti = tok_id(ti);
                }
            }
        }
        let mut words = Vec::new();
        let mut t = best_ti;
        while let Ok(i) = usize::try_from(t) {
            let tok = arena[i];
            if tok.word != 0 {
                words.push(tok.word);
            }
            t = tok.back;
        }
        words.reverse();
        (words, best_cost)
    }
}
