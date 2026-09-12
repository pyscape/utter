//! Eager composition of the model's `HCLr` with the grammar `G`, the epsilon-sequencing filter of
//! OpenFst's default `ComposeFst`, after relabeling G's input side with the lookahead map the
//! graph carries; then disambiguation input labels are erased and the result connected.
//!
//! A determinized HCL delays word labels until the phone sequence disambiguates the word, so
//! most of the graph sits behind output epsilons and a plain eager composition copies it once per
//! grammar state. The label reachability sets the graph carries are used to skip a pair of states
//! from which no label the grammar state can accept is reachable; those pairs are exactly the
//! ones connect would remove, so the result is unchanged.
// [[rr:TD-2#The graph: composition]]

use crate::fst::{Arc, Label, StateId, VectorFst, NO_STATE};
use std::collections::HashMap;

pub struct ComposeError(pub String);

/// Compose `hcl` (output side) with `g` (input side). `relabel` maps G's labels into the label
/// space HCLr's output arcs carry, identity when empty; a G arc whose label has no image can
/// never match.
/// `max_states` bounds the result before the work is done.
pub fn compose(
    hcl: &VectorFst,
    g: &VectorFst,
    relabel: &HashMap<Label, Label>,
    reach: Option<(&[Vec<(Label, Label)>], Label)>,
    max_states: usize,
) -> Result<VectorFst, ComposeError> {
    let mut out = VectorFst::default();
    if hcl.start == NO_STATE || g.start == NO_STATE {
        return Ok(out);
    }
    // Per G state: every relabeled label on arcs from its input-epsilon closure, plus the final
    // label when that closure holds a final state; sorted, for the reachability test.
    let useful: Vec<Vec<Label>> = (0..g.num_states())
        .map(|s| {
            let mut labels = Vec::new();
            let mut seen = vec![false; g.num_states()];
            let mut stack = vec![s as StateId];
            seen[s] = true;
            while let Some(q) = stack.pop() {
                let st = &g.states[q as usize];
                if st.final_weight.is_finite() {
                    if let Some((_, fl)) = reach {
                        labels.push(fl);
                    }
                }
                for a in &st.arcs {
                    if a.ilabel == 0 {
                        if !seen[a.nextstate as usize] {
                            seen[a.nextstate as usize] = true;
                            stack.push(a.nextstate);
                        }
                    } else if relabel.is_empty() {
                        labels.push(a.ilabel);
                    } else if let Some(&l) = relabel.get(&a.ilabel) {
                        labels.push(l);
                    }
                }
            }
            labels.sort_unstable();
            labels.dedup();
            labels
        })
        .collect();
    let can_reach = |s1: StateId, s2: StateId| -> bool {
        let Some((sets, _)) = reach else { return true };
        let set = &sets[s1 as usize];
        let labels = &useful[s2 as usize];
        let mut i = 0;
        for &(begin, end) in set {
            while i < labels.len() && labels[i] < begin {
                i += 1;
            }
            if i < labels.len() && labels[i] < end {
                return true;
            }
        }
        false
    };
    // Per G state: ilabel (relabeled) -> arcs, plus the input-epsilon arcs.
    let mut g_by_label: Vec<HashMap<Label, Vec<Arc>>> = Vec::with_capacity(g.num_states());
    let mut g_eps: Vec<Vec<Arc>> = Vec::with_capacity(g.num_states());
    for st in &g.states {
        let mut m: HashMap<Label, Vec<Arc>> = HashMap::new();
        let mut eps = Vec::new();
        for a in &st.arcs {
            if a.ilabel == 0 {
                eps.push(*a);
            } else if relabel.is_empty() {
                m.entry(a.ilabel).or_default().push(*a);
            } else if let Some(&l) = relabel.get(&a.ilabel) {
                m.entry(l).or_default().push(*a);
            }
        }
        g_by_label.push(m);
        g_eps.push(eps);
    }

    let mut index: HashMap<(StateId, StateId, u8), StateId> = HashMap::new();
    let mut queue: Vec<(StateId, StateId, u8)> = Vec::new();
    fn intern(
        index: &mut HashMap<(StateId, StateId, u8), StateId>,
        key: (StateId, StateId, u8),
        out: &mut VectorFst,
        queue: &mut Vec<(StateId, StateId, u8)>,
    ) -> StateId {
        if let Some(&s) = index.get(&key) {
            return s;
        }
        let s = out.add_state();
        index.insert(key, s);
        queue.push(key);
        s
    }
    if !can_reach(hcl.start, g.start) {
        return Ok(out);
    }
    out.start = intern(&mut index, (hcl.start, g.start, 0), &mut out, &mut queue);

    while let Some(key) = queue.pop() {
        if out.num_states() > max_states {
            return Err(ComposeError(format!("composition exceeds {max_states} states")));
        }
        let (s1, s2, f) = key;
        let cur = index[&key];
        let st1 = &hcl.states[s1 as usize];
        let st2 = &g.states[s2 as usize];
        let num_oeps = st1.arcs.iter().filter(|a| a.olabel == 0).count();
        let all_eps1 = num_oeps == st1.arcs.len();
        let no_eps1 = num_oeps == 0;

        if st1.final_weight.is_finite() && st2.final_weight.is_finite() {
            out.set_final(cur, st1.final_weight + st2.final_weight);
        }
        for a1 in &st1.arcs {
            if a1.olabel == 0 {
                // fst1 moves alone on an output epsilon: only from filter state 0.
                if f == 0 && can_reach(a1.nextstate, s2) {
                    let ns = intern(&mut index, (a1.nextstate, s2, 0), &mut out, &mut queue);
                    out.add_arc(cur, Arc { ilabel: a1.ilabel, olabel: 0, weight: a1.weight, nextstate: ns });
                }
            } else if let Some(arcs2) = g_by_label[s2 as usize].get(&a1.olabel) {
                for a2 in arcs2 {
                    if !can_reach(a1.nextstate, a2.nextstate) {
                        continue;
                    }
                    let ns = intern(&mut index, (a1.nextstate, a2.nextstate, 0), &mut out, &mut queue);
                    out.add_arc(cur, Arc { ilabel: a1.ilabel, olabel: a2.olabel, weight: a1.weight + a2.weight, nextstate: ns });
                }
            }
        }
        // fst2 moves alone on an input epsilon: blocked when fst1 has only epsilon arcs; the
        // filter moves to 1 when fst1 has epsilon arcs it must not take afterwards.
        if !all_eps1 {
            let nf = if no_eps1 { 0 } else { 1 };
            for a2 in &g_eps[s2 as usize] {
                if !can_reach(s1, a2.nextstate) {
                    continue;
                }
                let ns = intern(&mut index, (s1, a2.nextstate, nf), &mut out, &mut queue);
                out.add_arc(cur, Arc { ilabel: 0, olabel: a2.olabel, weight: a2.weight, nextstate: ns });
            }
        }
    }
    Ok(out)
}

/// Map every input label in `labels` to epsilon.
pub fn erase_input_labels(fst: &mut VectorFst, labels: &[Label]) {
    let set: std::collections::HashSet<Label> = labels.iter().copied().collect();
    for st in fst.states.iter_mut() {
        for a in st.arcs.iter_mut() {
            if set.contains(&a.ilabel) {
                a.ilabel = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arc(i: Label, o: Label, w: f32, n: StateId) -> Arc {
        Arc { ilabel: i, olabel: o, weight: w, nextstate: n }
    }

    #[test]
    fn composes_a_word_path_through_backoff() {
        // HCL: 0 -(1:eps)-> 1 -(2:A)-> 2(final); 1 -(3:B)-> 3(final)
        let mut hcl = VectorFst::default();
        for _ in 0..4 {
            hcl.add_state();
        }
        hcl.start = 0;
        hcl.add_arc(0, arc(1, 0, 0.5, 1));
        hcl.add_arc(1, arc(2, 10, 0.0, 2));
        hcl.add_arc(1, arc(3, 11, 0.0, 3));
        hcl.set_final(2, 0.0);
        hcl.set_final(3, 0.0);
        // G: 0 -(A)-> 1(final); 1 -(eps)-> 0 (backoff); words A=100, B=101
        let mut g = VectorFst::default();
        g.add_state();
        g.add_state();
        g.start = 0;
        g.add_arc(0, arc(100, 100, 1.0, 1));
        g.add_arc(1, arc(0, 0, 0.7, 0));
        g.set_final(1, 0.2);
        let relabel: HashMap<Label, Label> = [(100, 10), (101, 11)].into_iter().collect();
        let c = compose(&hcl, &g, &relabel, None, 1000).ok().unwrap();
        let mut c = c;
        c.connect();
        // path: (0,0) -1:eps-> (1,0) -2:100-> (2,1) final 0.2
        assert_eq!(c.start, 0);
        let s = &c.states[c.start as usize];
        assert_eq!(s.arcs.len(), 1);
        let n1 = &c.states[s.arcs[0].nextstate as usize];
        assert_eq!(n1.arcs.len(), 1);
        assert_eq!(n1.arcs[0].olabel, 100);
        assert!((n1.arcs[0].weight - 1.0).abs() < 1e-6);
        let fin = &c.states[n1.arcs[0].nextstate as usize];
        assert!((fin.final_weight - 0.2).abs() < 1e-6);
    }
}
