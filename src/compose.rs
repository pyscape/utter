//! Composition of the model's `HCLr` with the grammar `G` exactly as libvosk's lazy
//! `ComposeFst` over an `olabel_lookahead` graph builds it, enumerated eagerly: the alternate
//! sequence filter (G's epsilons before HCLr's), label reachability over the interval sets the
//! graph carries, lookahead weights as the log-sum of the reachable G arcs pushed onto HCLr's
//! epsilon arcs and quantized in the filter state, and G's arc pushed whole onto the first
//! HCLr epsilon arc from which it is the only reachable one. States libvosk's decoder could
//! reach are all kept, so pruning sees the same graph; disambiguation input labels are erased.
// [[rr:TD-2#The graph: composition]]

use crate::fst::{Arc, Label, StateId, VectorFst, NO_STATE};
use std::collections::HashMap;

pub struct ComposeError(pub String);

/// OpenFst's `kDelta`, the quantum of the pushed-weight filter state.
const DELTA: f32 = 1.0 / 1024.0;

fn quantize(w: f32) -> f32 {
    if !w.is_finite() {
        w
    } else {
        (w / DELTA + 0.5).floor() * DELTA
    }
}

/// Tropical weights summed in the log semiring, OpenFst's `LogPlus` in double precision.
fn log_plus(w: f64, v: f64) -> f64 {
    if w.is_infinite() {
        return v;
    }
    let (f1, f2) = (w, v);
    if f1 > f2 {
        f2 - (1.0 + (-(f1 - f2)).exp()).ln()
    } else {
        f1 - (1.0 + (-(f2 - f1)).exp()).ln()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Key {
    s1: StateId,
    s2: StateId,
    alt: u8,
    fw_bits: u32,
    fl: Label,
}

/// What the lookahead over G's arcs from a state finds, through the reachability set of one
/// HCLr state. A function of that pair alone, and the pair recurs across the composed states
/// that share it, so it is worth keeping.
#[derive(Clone, Copy)]
struct Look {
    reachable: bool,
    /// The single reachable G arc, when there is exactly one and no final to reach.
    prefix: Option<Arc>,
    /// Log-sum of the reachable arcs, met with the final weight; read only without a prefix.
    lweight: f32,
}

fn member(set: &[(Label, Label)], label: Label) -> bool {
    // intervals are sorted and disjoint
    let mut lo = 0;
    let mut hi = set.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        let (b, e) = set[mid];
        if label < b {
            hi = mid;
        } else if label >= e {
            lo = mid + 1;
        } else {
            return true;
        }
    }
    false
}

/// `reach`: per HCLr state, the reachable output labels as sorted `[begin, end)` intervals,
/// and the label that stands for a reachable final state.
pub fn compose(
    hcl: &VectorFst,
    g: &VectorFst,
    reach: &[Vec<(Label, Label)>],
    final_label: Label,
    max_states: usize,
) -> Result<VectorFst, ComposeError> {
    let mut out = VectorFst::default();
    if hcl.start == NO_STATE || g.start == NO_STATE {
        return Ok(out);
    }
    if reach.len() != hcl.num_states() {
        return Err(ComposeError("graph carries no reachability data".into()));
    }
    let mut index: HashMap<Key, StateId> = HashMap::new();
    let mut looks: HashMap<u64, Look> = HashMap::new();
    let mut queue: Vec<(Key, StateId)> = Vec::new();
    fn intern(
        index: &mut HashMap<Key, StateId>,
        key: Key,
        out: &mut VectorFst,
        queue: &mut Vec<(Key, StateId)>,
    ) -> StateId {
        if let Some(&s) = index.get(&key) {
            return s;
        }
        let s = out.add_state();
        index.insert(key, s);
        queue.push((key, s));
        s
    }
    let start_key = Key {
        s1: hcl.start,
        s2: g.start,
        alt: 0,
        fw_bits: 0f32.to_bits(),
        fl: 0,
    };
    out.start = intern(&mut index, start_key, &mut out, &mut queue);

    while let Some((key, cur)) = queue.pop() {
        if out.num_states() > max_states {
            return Err(ComposeError(format!(
                "composition exceeds {max_states} states"
            )));
        }
        let Key {
            s1,
            s2,
            alt,
            fw_bits,
            fl,
        } = key;
        let fw = f32::from_bits(fw_bits);
        let st1 = &hcl.states[s1 as usize];
        let st2 = &g.states[s2 as usize];
        let g_final = st2.final_weight.is_finite();
        let ne2 = st2.arcs.iter().filter(|a| a.ilabel == 0).count();
        let alleps2 = ne2 == st2.arcs.len() && !g_final;
        let noeps2 = ne2 == 0;

        // Final weight: the pushed weight is refunded; a pending label forbids finality.
        if fl == 0 && st1.final_weight.is_finite() && g_final {
            out.set_final(cur, (st1.final_weight - fw) + st2.final_weight);
        }

        if fl != 0 {
            // A pushed label is outstanding: G stands still until HCLr delivers it.
            for a1 in &st1.arcs {
                if a1.olabel == fl {
                    let ns = intern(
                        &mut index,
                        Key {
                            s1: a1.nextstate,
                            s2,
                            alt: 0,
                            fw_bits: 0f32.to_bits(),
                            fl: 0,
                        },
                        &mut out,
                        &mut queue,
                    );
                    out.add_arc(
                        cur,
                        Arc {
                            ilabel: a1.ilabel,
                            olabel: 0,
                            weight: a1.weight,
                            nextstate: ns,
                        },
                    );
                } else if a1.olabel == 0 {
                    let ok = st1.arcs.len() == 1 || member(&reach[a1.nextstate as usize], fl);
                    if ok {
                        let ns = intern(
                            &mut index,
                            Key {
                                s1: a1.nextstate,
                                s2,
                                alt,
                                fw_bits,
                                fl,
                            },
                            &mut out,
                            &mut queue,
                        );
                        out.add_arc(
                            cur,
                            Arc {
                                ilabel: a1.ilabel,
                                olabel: 0,
                                weight: a1.weight,
                                nextstate: ns,
                            },
                        );
                    }
                }
            }
            continue;
        }

        for a1 in &st1.arcs {
            if a1.olabel == 0 {
                // HCLr moves alone on an output epsilon.
                if alleps2 {
                    continue;
                }
                let new_alt: u8 = if noeps2 { 0 } else { 1 };
                let look_key = ((a1.nextstate as u64) << 32) | (s2 as u64);
                let look = match looks.get(&look_key) {
                    Some(&l) => l,
                    None => {
                        let set = &reach[a1.nextstate as usize];
                        let mut first: Option<usize> = None;
                        let mut last: usize = 0;
                        let mut lsum = f64::INFINITY;
                        for (pos, a2) in st2.arcs.iter().enumerate() {
                            if a2.ilabel != 0 && member(set, a2.ilabel) {
                                if first.is_none() {
                                    first = Some(pos);
                                }
                                last = pos + 1;
                                lsum = log_plus(lsum, a2.weight as f64);
                            }
                        }
                        let reach_final = g_final && member(set, final_label);
                        let reach_arc = first.is_some();
                        let prefix = match first {
                            Some(b) if last - b == 1 && !reach_final => Some(st2.arcs[b]),
                            _ => None,
                        };
                        let mut lweight = if reach_arc {
                            lsum as f32
                        } else {
                            f32::INFINITY
                        };
                        if reach_final {
                            lweight = if reach_arc {
                                lweight.min(st2.final_weight)
                            } else {
                                st2.final_weight
                            };
                        }
                        let l = Look {
                            reachable: reach_arc || reach_final,
                            prefix,
                            lweight,
                        };
                        looks.insert(look_key, l);
                        l
                    }
                };
                if !look.reachable {
                    continue;
                }
                if let Some(larc) = look.prefix {
                    // Weight lookahead is skipped when a prefix is found: the lookahead weight
                    // is One, the earlier pushed weight is refunded, and G's arc is charged.
                    let weight = a1.weight + (0.0 - fw) + larc.weight;
                    let ns = intern(
                        &mut index,
                        Key {
                            s1: a1.nextstate,
                            s2: larc.nextstate,
                            alt: new_alt,
                            fw_bits: 0f32.to_bits(),
                            fl: larc.ilabel,
                        },
                        &mut out,
                        &mut queue,
                    );
                    out.add_arc(
                        cur,
                        Arc {
                            ilabel: a1.ilabel,
                            olabel: larc.olabel,
                            weight,
                            nextstate: ns,
                        },
                    );
                } else {
                    let lweight = look.lweight;
                    let weight = a1.weight + lweight - fw;
                    let ns = intern(
                        &mut index,
                        Key {
                            s1: a1.nextstate,
                            s2,
                            alt: new_alt,
                            fw_bits: quantize(lweight).to_bits(),
                            fl: 0,
                        },
                        &mut out,
                        &mut queue,
                    );
                    out.add_arc(
                        cur,
                        Arc {
                            ilabel: a1.ilabel,
                            olabel: 0,
                            weight,
                            nextstate: ns,
                        },
                    );
                }
            } else {
                // A real match: no lookahead, the pushed weight is refunded against G's arc.
                for a2 in st2.arcs.iter().filter(|a2| a2.ilabel == a1.olabel) {
                    let ns = intern(
                        &mut index,
                        Key {
                            s1: a1.nextstate,
                            s2: a2.nextstate,
                            alt: 0,
                            fw_bits: 0f32.to_bits(),
                            fl: 0,
                        },
                        &mut out,
                        &mut queue,
                    );
                    out.add_arc(
                        cur,
                        Arc {
                            ilabel: a1.ilabel,
                            olabel: a2.olabel,
                            weight: a1.weight + a2.weight - fw,
                            nextstate: ns,
                        },
                    );
                }
            }
        }
        // G moves alone on an input epsilon, only in alternate-sequence state 0.
        if alt == 0 {
            for a2 in st2.arcs.iter().filter(|a2| a2.ilabel == 0) {
                let ns = intern(
                    &mut index,
                    Key {
                        s1,
                        s2: a2.nextstate,
                        alt: 0,
                        fw_bits: 0f32.to_bits(),
                        fl: 0,
                    },
                    &mut out,
                    &mut queue,
                );
                out.add_arc(
                    cur,
                    Arc {
                        ilabel: 0,
                        olabel: a2.olabel,
                        weight: a2.weight - fw,
                        nextstate: ns,
                    },
                );
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
        Arc {
            ilabel: i,
            olabel: o,
            weight: w,
            nextstate: n,
        }
    }

    /// HCL: 0 -(1:eps)-> 1 -(2:A)-> 2(final); 1 -(3:B)-> 3(final). G accepts A (weight 1.0)
    /// then is final (0.2). The word label and G's weight are pushed onto the first arc.
    #[test]
    fn pushes_label_and_weight_onto_the_epsilon_arc() {
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
        let mut g = VectorFst::default();
        g.add_state();
        g.add_state();
        g.start = 0;
        g.add_arc(0, arc(10, 10, 1.0, 1));
        g.set_final(1, 0.2);
        let fl = 100;
        let reach = vec![
            vec![(10, 12)],
            vec![(10, 12)],
            vec![(fl, fl + 1)],
            vec![(fl, fl + 1)],
        ];
        let c = compose(&hcl, &g, &reach, fl, 1000).ok().unwrap();
        let s = &c.states[c.start as usize];
        assert_eq!(s.arcs.len(), 1);
        assert_eq!(s.arcs[0].olabel, 10);
        assert!((s.arcs[0].weight - 1.5).abs() < 1e-6);
        let n1 = &c.states[s.arcs[0].nextstate as usize];
        // only the A arc survives while the pushed label is outstanding, and it is now epsilon
        assert_eq!(n1.arcs.len(), 1);
        assert_eq!(n1.arcs[0].olabel, 0);
        assert_eq!(n1.arcs[0].ilabel, 2);
        let fin = &c.states[n1.arcs[0].nextstate as usize];
        assert!((fin.final_weight - 0.2).abs() < 1e-6);
    }
}
