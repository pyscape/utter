# Reading a partial's trust

Run 2026-09-12 16:11:10Z, utterpy over the Speech Commands testing split, 9230 clips whose first word appeared in a partial that carried a runner-up, 92-entry grammar, 40 ms blocks, 5 alternatives. Measured by `scripts/partial_trust.py`.

The first word a host sees is revised before the utterance ends on 1147 of 9230 (12.4%). A host that wants to act early needs to know, at the moment a word appears, whether it is one of those. The signals the partial carries at that moment, each scored by rank-AUC against whether the word survived:

| signal on the partial | rank-AUC |
|---|---|
| gap from the top `confidence` to the next | 0.12 |
| the top reading's own `confidence` | 0.32 |
| `energy_dbfs` under the word | 0.55 |

0.5 is no information; below 0.5 the signal is inverted (a smaller value means a revision is coming). Only the gap carries the answer. The word's own confidence barely moves it, because it is a raw accumulated path cost that grows with the utterance and is not a probability; the energy is the silence signal, not the trust one. `stable_ms` is zero the instant a word appears, so it cannot rank a first sighting, and is the hold a host waits out afterward instead.

## The gap is a probability the API already gives you

`confidence` is the negation of a reading's best-token cost, and the decoder's costs are negative log-likelihoods in nats. So the gap between the top two readings,

    gap = confidence[0] - confidence[1] = cost[1] - cost[0]  >= 0,

is how many nats of likelihood the runner-up gives up to the leader. Treating the two as a choice between them, the leader's share of the mass is a softmax over the two costs,

    P(leader) = e^-cost0 / (e^-cost0 + e^-cost1) = 1 / (1 + e^-gap) = sigmoid(gap).

That is a Viterbi approximation over the two leading beam readings, not a lattice posterior, so it is only roughly calibrated; the table below is how roughly. Buckets of the first appearances by gap, against the survival `sigmoid(gap)` predicts:

| gap (nats) | first appearances | survived | predicted `sigmoid(mean gap)` |
|---|---|---|---|
| 0 to 0.5 | 1145 | 614 / 1145 (54%) | 56% |
| 0.5 to 1 | 947 | 660 / 947 (70%) | 68% |
| 1 to 2 | 1368 | 1147 / 1368 (84%) | 81% |
| 2 to 4 | 1949 | 1848 / 1949 (95%) | 95% |
| 4 to 8 | 2446 | 2439 / 2446 (100%) | 100% |
| 8 to 8+ | 1375 | 1375 / 1375 (100%) | 100% |

Survival climbs with the gap and tracks the softmax, so the sigmoid is a usable trust score and a threshold on it is a trust gate. In terms a host acts on, holding a word until the gap clears a threshold:

| hold until gap over | words held | revisions caught | good words delayed |
|---|---|---|---|
| 0.5 | 1145 (12%) | 531 / 1147 (46%) | 614 / 8083 (8%) |
| 1.0 | 2092 (23%) | 818 / 1147 (71%) | 1274 / 8083 (16%) |
| 1.5 | 2861 (31%) | 969 / 1147 (84%) | 1892 / 8083 (23%) |
| 2.0 | 3460 (37%) | 1039 / 1147 (91%) | 2421 / 8083 (30%) |
| 3.0 | 4545 (49%) | 1108 / 1147 (97%) | 3437 / 8083 (43%) |
| 5.0 | 6065 (66%) | 1143 / 1147 (100%) | 4922 / 8083 (61%) |

So the number to read is not a word's own confidence but its lead over the next reading. A host that waits for `sigmoid(gap)` to clear its bar trades a little latency on the words it holds for catching most of the revisions it would otherwise have acted on. The Python is in the README, under the partial example.
