//! The runtime grammar as libvosk compiles it: a port of libvosk `src/language_model.cc`, an
//! absolute-discount n-gram estimator (order 2, discount 0.5) emitted as an acceptor over word
//! ids. Word 0 stands for the sentence end. libvosk determinizes the result; the estimator's
//! output is already deterministic (one arc per label per state), so only connect follows.
// [[rr:TD-2#The graph: the grammar as a bigram]]

use crate::fst::{Arc, StateId, VectorFst, NO_STATE};
use std::collections::{BTreeMap, HashMap};

#[derive(Default)]
struct LmState {
    history: Vec<i32>,
    counts: BTreeMap<i32, i32>,
    tot_count: i32,
    backoff: Option<usize>,
    fst_state: StateId,
}

pub struct BigramEstimator {
    order: usize,
    discount: f64,
    states: Vec<LmState>,
    index: HashMap<Vec<i32>, usize>,
}

impl BigramEstimator {
    pub fn new(order: usize, discount: f64) -> Self {
        assert!(order >= 2);
        BigramEstimator { order, discount, states: Vec::new(), index: HashMap::new() }
    }

    pub fn add_sentence(&mut self, sentence: &[i32]) {
        let mut history: Vec<i32> = Vec::new();
        for &w in sentence {
            assert!(w != 0, "word id 0 is the sentence end");
            self.increment(&history, w);
            history.push(w);
            if history.len() >= self.order {
                history.remove(0);
            }
        }
        self.increment(&history, 0);
    }

    fn increment(&mut self, history: &[i32], word: i32) {
        let s = self.find_or_create(history);
        let st = &mut self.states[s];
        *st.counts.entry(word).or_insert(0) += 1;
        st.tot_count += 1;
    }

    fn find_or_create(&mut self, history: &[i32]) -> usize {
        if let Some(&i) = self.index.get(history) {
            return i;
        }
        let i = self.states.len();
        self.states.push(LmState { history: history.to_vec(), fst_state: NO_STATE, ..Default::default() });
        self.index.insert(history.to_vec(), i);
        if !history.is_empty() {
            let b = self.find_or_create(&history[1..]);
            self.states[i].backoff = Some(b);
        }
        i
    }

    fn find_nonzero(&self, mut history: Vec<i32>) -> usize {
        loop {
            match self.index.get(&history) {
                Some(&l) if self.states[l].tot_count != 0 => return l,
                _ => {
                    assert!(!history.is_empty(), "no LM state for the empty history");
                    history.remove(0);
                }
            }
        }
    }

    pub fn estimate(mut self) -> VectorFst {
        // Every state's counts are added to all of its backoff ancestors.
        for l in 0..self.states.len() {
            let counts: Vec<(i32, i32)> = self.states[l].counts.iter().map(|(&k, &v)| (k, v)).collect();
            let mut b = self.states[l].backoff;
            while let Some(bi) = b {
                for &(w, c) in &counts {
                    *self.states[bi].counts.entry(w).or_insert(0) += c;
                    self.states[bi].tot_count += c;
                }
                b = self.states[bi].backoff;
            }
        }
        let mut fst = VectorFst::default();
        for st in self.states.iter_mut() {
            if st.tot_count != 0 {
                st.fst_state = fst.add_state();
            }
        }
        fst.start = self.states[self.find_nonzero(Vec::new())].fst_state;
        for l in 0..self.states.len() {
            let st = &self.states[l];
            if st.fst_state == NO_STATE {
                continue;
            }
            let state_count = st.tot_count as f64;
            for (&word, &count) in &st.counts {
                let logprob = (count as f64 * self.discount / state_count).ln();
                if word == 0 {
                    fst.set_final(st.fst_state, (-logprob) as f32);
                } else {
                    let mut next = st.history.clone();
                    next.push(word);
                    let dest = self.states[self.find_nonzero(next)].fst_state;
                    fst.add_arc(st.fst_state, Arc { ilabel: word, olabel: word, weight: (-logprob) as f32, nextstate: dest });
                }
            }
            if let Some(b) = st.backoff {
                let dest = self.states[b].fst_state;
                fst.add_arc(
                    st.fst_state,
                    Arc { ilabel: 0, olabel: 0, weight: (-(1.0 - self.discount).ln()) as f32, nextstate: dest },
                );
            }
        }
        fst.connect();
        fst.arc_sort_ilabel();
        fst
    }
}

/// libvosk's grammar compile: each string is one sentence, split on single spaces; words absent
/// from the symbol table are dropped with a warning through `warn`.
pub fn grammar_fst(strings: &[String], word_ids: &HashMap<String, i64>, mut warn: impl FnMut(String)) -> VectorFst {
    let mut est = BigramEstimator::new(2, 0.5);
    for line in strings {
        let mut sentence = Vec::new();
        for token in line.split(' ') {
            if token.is_empty() {
                continue;
            }
            match word_ids.get(token) {
                Some(&id) => sentence.push(id as i32),
                None => warn(format!("Ignoring word missing in vocabulary: '{token}'")),
            }
        }
        est.add_sentence(&sentence);
    }
    est.estimate()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_word_sentences_share_a_unigram_state() {
        let mut est = BigramEstimator::new(2, 0.5);
        est.add_sentence(&[5]);
        est.add_sentence(&[7]);
        let fst = est.estimate();
        // unigram state, [5], [7]
        assert_eq!(fst.num_states(), 3);
        let start = &fst.states[fst.start as usize];
        // two word arcs; no backoff from the empty history
        assert_eq!(start.arcs.len(), 2);
        // unigram counts: 5:1, 7:1, eos:2 -> tot 4; word arc -log(0.5/4)
        let w = start.arcs[0].weight;
        assert!((w - (-(0.5f64 / 4.0).ln()) as f32).abs() < 1e-6);
        assert!((start.final_weight - (-(2.0f64 * 0.5 / 4.0).ln()) as f32).abs() < 1e-6);
        let s5 = start.arcs.iter().find(|a| a.ilabel == 5).unwrap().nextstate;
        let st5 = &fst.states[s5 as usize];
        assert_eq!(st5.arcs.len(), 1);
        assert_eq!(st5.arcs[0].ilabel, 0);
        assert!((st5.arcs[0].weight - 0.6931472).abs() < 1e-6);
        assert!((st5.final_weight - 0.6931472).abs() < 1e-6);
    }
}
