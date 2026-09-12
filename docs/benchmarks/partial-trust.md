# Reading a partial's trust

Run 2026-09-12 23:04:00Z, utter 9aa6a83+dirty, utterpy 0.0.1, model vosk-model-small-en-us-0.15, 12th Gen Intel(R) Core(TM) i7-12700F, Python 3.14.4. Decoded by the binding /home/jared/Repos/utterpy/.venv/lib/python3.14/site-packages/utterpy/__init__.py built from utter 96e272c0ea28c53f022fbbff8bdcfa66aa30abd4-dirty, **not** the checkout as it stands (9aa6a83+dirty).

utterpy over the Speech Commands testing split, 9230 clips whose first word appeared in a partial that carried a runner-up, 92-entry grammar, 40 ms blocks, 5 alternatives, every coefficient and bar fitted on the validation split. Measured by `scripts/partial_trust.py`.

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
| 1 to 2 | 1366 | 1145 / 1366 (84%) | 81% |
| 2 to 4 | 1951 | 1850 / 1951 (95%) | 95% |
| 4 to 8 | 2446 | 2439 / 2446 (100%) | 100% |
| 8 to 8+ | 1375 | 1375 / 1375 (100%) | 100% |

Survival climbs with the gap and tracks the softmax, so the sigmoid is a usable trust score and a threshold on it is a trust gate. In terms a host acts on, holding a word until the gap clears a threshold:

| hold until gap over | words held | revisions caught | good words delayed |
|---|---|---|---|
| 0.5 | 1145 (12%) | 531 / 1147 (46%) | 614 / 8083 (8%) |
| 1.0 | 2092 (23%) | 818 / 1147 (71%) | 1274 / 8083 (16%) |
| 1.5 | 2861 (31%) | 969 / 1147 (84%) | 1892 / 8083 (23%) |
| 2.0 | 3458 (37%) | 1039 / 1147 (91%) | 2419 / 8083 (30%) |
| 3.0 | 4545 (49%) | 1108 / 1147 (97%) | 3437 / 8083 (43%) |
| 5.0 | 6065 (66%) | 1143 / 1147 (100%) | 4922 / 8083 (61%) |

So the number to read is not a word's own confidence but its lead over the next reading. A host that waits for `sigmoid(gap)` to clear its bar trades a little latency on the words it holds for catching most of the revisions it would otherwise have acted on. The Python is in the README, under the partial example.


## What the readings have done by the time a word appears

The gap above is a level: one snapshot of a two-way Viterbi softmax. It is an approximation, not a posterior, so its errors need not be independent between readings of the beam, and how a reading's standing *moves* could carry survival information the level does not. What follows measures that. The readings only change when the decoder advances a frame, which at 40 ms blocks is one block in six (`[[rr:TD-7#Decision outcome]]`), so motion is measured per advance, not per block. An advance is a block whose readings differ from the block before; a reading is its whole text, `[sil]` included; and a reading's lead is its confidence less the best of the others, which is the gap for rank 0 and a deficit for the rest.

Advances arrive every 240 / 240 ms (median / p90), and on this corpus that is exact: 21315 of the 21315 intervals measured 240 ms and none measured anything else, one network chunk at 40 ms blocks. The first advance lands at 400 / 400 ms, the audio that first chunk needs. A one-second clip therefore holds 3 / 3 advances, and a first sighting has almost no history behind it.

The two moments a rule is read at, and the two places the figure could come from. The harness sees the readings the runtime reports, five of them here, and derives the motion from its own series; the runtime keeps the history over every surviving group, so a reading entering the list carries motion the harness has no record of (`[[rr:TD-9#Readings are read once per decoding advance]]`). The runtime column is **pending**: it is filled when the fields are read from the JSON instead of derived, and until then this page's availability is a lower bound on the runtime's.

| figure | defined at the sighting | defined one advance later | runtime, over every group |
|---|---|---|---|
| the rank-0 reading was there at the previous advance (`had_history`) | 3760 / 9230 (40.7%) | 4930 / 5045 (97.7%) | pending |
| `lead_delta` of rank 0 is defined | 3760 / 9230 (40.7%) | 4930 / 5045 (97.7%) | pending |
| `lead_delta` of rank 1 is defined | 3698 / 9230 (40.1%) | 1921 / 5045 (38.1%) | pending |
| `lead_delta` of the reading that led before is defined | 3468 / 9230 (37.6%) | 4729 / 5045 (93.7%) | pending |
| `entropy_delta` is defined (there was a previous advance) | 8473 / 9230 (91.8%) | 5045 / 5045 (100.0%) | pending |

The rank-0 reading has stood 1 / 2 advances (median / p90) and is at its first on 59.3% of first sightings. That is the shape of the problem: a word first appears when its reading appears, so at that moment there is usually nothing to have moved.

## Every signal at the first sighting, against a revision

The same rank-AUC as above, revision as the positive label, extended with the motion signals. A signal that is undefined at some first sightings is scored on the ones where it is defined, and that subset is given:

| signal at the first sighting | first sightings scored | rank-AUC |
|---|---|---|
| gap from the top `confidence` to the next | 9230 | 0.12 |
| the top reading's own `confidence` | 9230 | 0.32 |
| `energy_dbfs` under the word | 9230 | 0.55 |
| advances the top reading has stood | 9230 | 0.42 |
| `lead_delta` of rank 0 | 3760 | 0.18 |
| `lead_delta` of rank 1 | 3698 | 0.48 |
| `lead_delta` of the reading that led at the last advance | 3468 | 0.66 |
| rank-0 changes so far | 9230 | 0.47 |
| entropy over the readings | 9230 | 0.88 |
| `entropy_delta` | 8473 | 0.85 |
| readings in contention that vanished, by now | 9230 | 0.37 |

Below 0.5 the signal is inverted, as with the gap. The subset column is half the answer: a signal with information on a tenth of the first sightings is a signal a host cannot read most of the time.

## The same signals at equal gap

AUC inside each gap bucket, so the level is held roughly fixed and what is left is whatever the motion adds. A dash is a bucket with no revisions, one class only, or no clip where the signal is defined:

| signal | 0-0.5 | 0.5-1 | 1-2 | 2-4 | 4-8 | 8-8+ |
|---|---|---|---|---|---|---|
| advances the top reading has stood | 0.54 | 0.54 | 0.54 | 0.55 | 0.56 | - |
| `lead_delta` of rank 0 | 0.62 | 0.55 | 0.57 | 0.51 | 0.65 | - |
| `lead_delta` of rank 1 | 0.43 | 0.47 | 0.41 | 0.46 | 0.52 | - |
| `lead_delta` of the reading that led at the last advance | 0.47 | 0.54 | 0.56 | 0.55 | 0.63 | - |
| rank-0 changes so far | 0.50 | 0.51 | 0.51 | 0.51 | 0.51 | - |
| entropy over the readings | 0.59 | 0.61 | 0.66 | 0.69 | 0.66 | - |
| `entropy_delta` | 0.59 | 0.60 | 0.61 | 0.66 | 0.70 | - |
| readings in contention that vanished, by now | 0.51 | 0.50 | 0.50 | 0.50 | 0.51 | - |
| first sightings in the bucket | 1145 | 947 | 1366 | 1951 | 2446 | 1375 |

## Gap alone against gap plus one motion signal

A two-feature logistic regression of P(revision) on the gap and one motion signal, fit by Newton steps with a 1e-3 ridge, scored **in sample** by AUC against the gap alone on the same clips. The held-out version of this question is the three rules below; this table is the per-signal ceiling. First on the clips where the signal is defined:

| signal | clips | AUC, gap | AUC, gap + signal | coefficient on gap | on the signal |
|---|---|---|---|---|---|
| advances the top reading has stood | 9230 | 0.878 | 0.880 | -1.137 | +0.4907 |
| `lead_delta` of rank 0 | 3760 | 0.935 | 0.937 | -1.244 | +0.1646 |
| `lead_delta` of rank 1 | 3698 | 0.872 | 0.874 | -1.000 | -0.0411 |
| `lead_delta` of the reading that led at the last advance | 3468 | 0.828 | 0.828 | -1.130 | +0.0257 |
| rank-0 changes so far | 9230 | 0.878 | 0.878 | -1.103 | +0.0671 |
| entropy over the readings | 9230 | 0.878 | 0.884 | -0.687 | +1.5468 |
| `entropy_delta` | 8473 | 0.884 | 0.887 | -0.865 | +0.7395 |
| readings in contention that vanished, by now | 9230 | 0.878 | 0.878 | -1.101 | +0.0059 |

Then on all 9230 first sightings, the undefined values set to zero beside an indicator column that says they were undefined, which is what a host that reads the field unconditionally would be doing. The gap alone scores 0.878 on this set:

| signal | AUC, gap + signal + missing | coefficient on gap | on the signal | on missing |
|---|---|---|---|---|
| advances the top reading has stood | 0.880 | -1.137 | +0.4907 | - |
| `lead_delta` of rank 0 | 0.881 | -1.185 | +0.1504 | +0.332 |
| `lead_delta` of rank 1 | 0.879 | -1.097 | -0.0410 | +0.271 |
| `lead_delta` of the reading that led at the last advance | 0.878 | -1.097 | +0.0277 | -0.132 |
| rank-0 changes so far | 0.878 | -1.103 | +0.0671 | - |
| entropy over the readings | 0.884 | -0.687 | +1.5468 | - |
| `entropy_delta` | 0.881 | -0.918 | +0.6791 | +0.444 |
| readings in contention that vanished, by now | 0.878 | -1.101 | +0.0059 | - |

## Holding by the fit instead of by the gap, at equal words held

The best of those fits is gap + entropy over the readings (AUC 0.884 against 0.878 for the gap alone, both in sample). A host holds a word when the fit's P(revision) clears a bar; the bar is set here to hold exactly as many words as each gap threshold in the table above holds, so the two rules are compared at the same count of words held:

| hold until gap over | words held | revisions caught, gap | revisions caught, fit | good words delayed, gap | good words delayed, fit |
|---|---|---|---|---|---|
| 0.5 | 1145 (12%) | 531 / 1147 (46%) | 543 / 1147 (47%) | 614 / 8083 (8%) | 602 / 8083 (7%) |
| 1.0 | 2092 (23%) | 818 / 1147 (71%) | 845 / 1147 (74%) | 1274 / 8083 (16%) | 1247 / 8083 (15%) |
| 1.5 | 2861 (31%) | 969 / 1147 (84%) | 985 / 1147 (86%) | 1892 / 8083 (23%) | 1876 / 8083 (23%) |
| 2.0 | 3458 (37%) | 1039 / 1147 (91%) | 1045 / 1147 (91%) | 2419 / 8083 (30%) | 2413 / 8083 (30%) |
| 3.0 | 4545 (49%) | 1108 / 1147 (97%) | 1116 / 1147 (97%) | 3437 / 8083 (43%) | 3429 / 8083 (42%) |
| 5.0 | 6065 (66%) | 1143 / 1147 (100%) | 1143 / 1147 (100%) | 4922 / 8083 (61%) | 4922 / 8083 (61%) |

Both columns are fit and scored on the same clips, so the fit's is a ceiling rather than a forecast; the held-out figures are below.

## Holding one advance

A host that waits for the next advance before acting waits 240 / 240 ms (median / p90) and only gets the chance on 5045 of 9230 first sightings (54.7%): the rest are at the last advance the clip has. On 85% of those the next advance is the clip's own last, so on this corpus holding one advance is close to waiting for the final. The same signals read at that advance:

| signal at the next advance | clips scored | rank-AUC |
|---|---|---|
| gap from the top `confidence` to the next | 5045 | 0.24 |
| the top reading's own `confidence` | 5045 | 0.42 |
| `energy_dbfs` under the word | 5032 | 0.55 |
| advances the top reading has stood | 5045 | 0.32 |
| `lead_delta` of rank 0 | 4930 | 0.71 |
| `lead_delta` of rank 1 | 1921 | 0.33 |
| `lead_delta` of the reading that led at the last advance | 4729 | 0.02 |
| rank-0 changes so far | 5045 | 0.89 |
| entropy over the readings | 5045 | 0.75 |
| `entropy_delta` | 5045 | 0.24 |
| readings in contention that vanished, by now | 5045 | 0.69 |

The revision is often already on the page: rank 0 leads with a different first word at that advance on 687 of the 711 revisions (97%), against 6 of the 4334 words that survive (0.1%); rank 0 had fallen back to `[sil]` on 13. So the hold is nearly a decision by itself, before any signal is read off it.

Calibration of the gap read at that advance, against whether the word first shown one advance earlier is the one the final settles on:

| gap at the next advance (nats) | clips | survived | predicted `sigmoid(mean gap)` |
|---|---|---|---|
| 0 to 0.5 | 86 | 41 / 86 (48%) | 56% |
| 0.5 to 1 | 87 | 46 / 87 (53%) | 68% |
| 1 to 2 | 155 | 78 / 155 (50%) | 82% |
| 2 to 4 | 360 | 226 / 360 (63%) | 96% |
| 4 to 8 | 1524 | 1260 / 1524 (83%) | 100% |
| 8 to 8+ | 2833 | 2683 / 2833 (95%) | 100% |

Expected calibration error of `sigmoid(gap)` read one advance late, against survival to the final: 0.118. It is worse than at the first sighting because the gap keeps growing while the word's fate is already decided.

## Readings that vanish while still in contention

A reading within 2 nats of the leader at one advance and absent at the next did not decay out of the list, it disappeared from it. Vosk's alternatives group tokens by word sequence, so two alignments of one reading merge and the loser's evidence is dropped rather than added (`[[rr:TD-2#The decoder: partial alternatives]]`); beam pruning also removes readings, so this count is an upper bound on those merges and a proxy, not a measurement.

Over a clip there are 0.96 such events on average, 1 / 2 by median / p90, and 6481 of 9230 clips (70.2%) have at least one. Counting only the ones visible by the first sighting, 0.67 per clip and 5207 of 9230 clips (56.4%) have one.

| clips | revised |
|---|---|
| at least one vanishing by the first sighting | 400 / 5207 (7.7%) |
| none | 747 / 4023 (18.6%) |

## Six rules on the same clips, every one fitted off them

The sections above are in sample. Here are the rules a host could run, with every coefficient and every operating point fitted on the validation split and every figure scored on the testing split:

- **R0**, the README's rule: `trust = sigmoid(gap)` at the first sighting, and 1.0 when the list holds one reading, exactly as the README's `trust()` reads it. Nothing is fit.
- **B1**, the same gap through a logistic fit, which is the recalibration alone and the baseline any new field has to beat.
- **B2**, the gap and the entropy of the readings' softmax: the strongest second reading a host already has, and no runtime change.
- **R1**, the gap and the motion available at the first sighting: the displaced reading's `lead_delta` and rank 0's, each missing value zeroed beside an indicator.
- **R1e**, the same with the entropy beside it, which is the strongest rule on this page that a host could run at the sighting.
- **R2**, R1 read at the first advance after the sighting, charged the wait.
- **H**, the wait with no reading at all: release at the first advance where rank 0 still leads with the word. It separates what the delay buys from what the motion buys.

| rule | clips scored | AUC against revision |
|---|---|---|
| R0, `sigmoid(gap)`, the README's rule | 9230 | 0.878 |
| B1, the gap recalibrated | 9230 | 0.878 |
| B2, the gap and the readings' entropy | 9230 | 0.884 |
| R1, the gap and the motion | 9230 | 0.881 |
| R1e, the gap, the entropy and the motion | 9230 | 0.889 |
| R2, R1 one advance later, as the rule runs | 5045 | 0.998 |
| H, hold one advance, reading nothing | 5045 | 0.982 |
| R2, the later reading alone | 5045 | 0.779 |

B1 orders the clips exactly as R0 does, a logistic of one column being monotone in it, so its AUC difference is zero by construction and everything it buys is in the calibration and the decisions below. Scores are oriented as P(revision), so higher is better and 0.5 is no information. R2 and H are scored only on the clips that have another advance, and both are read as they run: a word rank 0 no longer leads with is a revision called with certainty, which is where nearly all of their information is. H carries no score of its own, so its AUC is that call and nothing else. Scoring R2's later reading on its own gap and motion, as if the word were still on offer, is worse than R0 because the gap keeps growing after the word's fate is settled. The paired bootstrap resamples clips 1000 times, both rules seeing the same resample:

| difference | clips | AUC difference | 95% interval |
|---|---|---|---|
| B1 - R0, the recalibration alone | 9230 | 0.0000 | 0.0000 to 0.0000 |
| B2 - B1, the entropy over it | 9230 | 0.0062 | 0.0035 to 0.0091 |
| R1 - R0, the motion against the README | 9230 | 0.0033 | 0.0015 to 0.0053 |
| R1 - B1, the motion over the recalibration | 9230 | 0.0033 | 0.0015 to 0.0053 |
| R1 - B2, the motion alone against the entropy alone | 9230 | -0.0029 | -0.0063 to 0.0004 |
| R1e - B2, the motion added to the entropy | 9230 | 0.0047 | 0.0026 to 0.0068 |
| R2 - R0, as the rule runs | 5045 | 0.1337 | 0.1218 to 0.1455 |
| H - R0, the wait alone | 5045 | 0.1177 | 0.1059 to 0.1299 |

Where the R1 coefficients come from moves it little: fit on the validation split it scores 0.881 on all 9230 testing clips; fit on the even-indexed half of testing it scores 0.878 on the odd half; fit and scored on testing itself, 0.881. The fitted coefficients, each in the order named:

| rule | order | coefficients |
|---|---|---|
| gap | intercept, gap | -0.0706, -1.0816 |
| gap + entropy | intercept, gap, entropy | -1.7228, -0.7114, +1.3611 |
| gap + motion | intercept, gap, displaced `lead_delta`, its missing flag, rank-0 `lead_delta`, its missing flag | -0.2352, -1.1447, +0.0381, -0.2573, +0.1437, +0.4156 |
| gap + entropy + motion | intercept, gap, entropy, displaced `lead_delta`, its missing flag, rank-0 `lead_delta`, its missing flag | -2.0163, -0.7568, +1.5334, +0.0023, +0.0844, +0.1076, +0.0169 |

### Calibration, and the error in it

Each rule's predicted trust against the survival observed, in bins of the prediction. Expected calibration error is the bin-weighted distance between the two columns:

| rule | bin | clips | mean predicted | survived |
|---|---|---|---|---|
| R0 | 0.5 to 0.7 | 1831 | 60% | 58% |
| R0 | 0.7 to 0.8 | 867 | 75% | 80% |
| R0 | 0.8 to 0.9 | 970 | 85% | 88% |
| R0 | 0.9 to 0.95 | 820 | 93% | 94% |
| R0 | 0.95 to 1 | 4742 | 99% | 99% |
| R0 | **expected calibration error** | 9230 | | **0.012** |
| B1 | 0.5 to 0.7 | 1583 | 61% | 57% |
| B1 | 0.7 to 0.8 | 867 | 75% | 76% |
| B1 | 0.8 to 0.9 | 972 | 85% | 85% |
| B1 | 0.9 to 0.95 | 777 | 93% | 92% |
| B1 | 0.95 to 1 | 5031 | 99% | 99% |
| B1 | **expected calibration error** | 9230 | | **0.009** |
| B2 | 0 to 0.5 | 257 | 46% | 35% |
| B2 | 0.5 to 0.7 | 1244 | 61% | 59% |
| B2 | 0.7 to 0.8 | 873 | 75% | 74% |
| B2 | 0.8 to 0.9 | 940 | 85% | 87% |
| B2 | 0.9 to 0.95 | 739 | 93% | 91% |
| B2 | 0.95 to 1 | 5177 | 99% | 99% |
| B2 | **expected calibration error** | 9230 | | **0.011** |
| R1 | 0 to 0.5 | 114 | 45% | 35% |
| R1 | 0.5 to 0.7 | 1485 | 61% | 58% |
| R1 | 0.7 to 0.8 | 799 | 75% | 75% |
| R1 | 0.8 to 0.9 | 1002 | 85% | 85% |
| R1 | 0.9 to 0.95 | 727 | 93% | 95% |
| R1 | 0.95 to 1 | 5103 | 99% | 99% |
| R1 | **expected calibration error** | 9230 | | **0.010** |
| R1e | 0 to 0.5 | 328 | 44% | 33% |
| R1e | 0.5 to 0.7 | 1168 | 61% | 61% |
| R1e | 0.7 to 0.8 | 836 | 75% | 73% |
| R1e | 0.8 to 0.9 | 935 | 85% | 86% |
| R1e | 0.9 to 0.95 | 720 | 93% | 93% |
| R1e | 0.95 to 1 | 5243 | 99% | 99% |
| R1e | **expected calibration error** | 9230 | | **0.008** |
| R2 | 0 to 0.5 | 693 | 0% | 1% |
| R2 | 0.5 to 0.7 | 44 | 63% | 91% |
| R2 | 0.7 to 0.8 | 46 | 75% | 85% |
| R2 | 0.8 to 0.9 | 59 | 85% | 92% |
| R2 | 0.9 to 0.95 | 81 | 93% | 94% |
| R2 | 0.95 to 1 | 4122 | 100% | 100% |
| R2 | **expected calibration error** | 5045 | | **0.006** |

The same calibration read in the gap's own buckets, the ones the table at the top of the page uses, so a rule that is better ordered but worse calibrated shows as one:

| gap at the sighting (nats) | clips | survived | R0 predicts | B1 predicts | B2 predicts | R1 predicts | R1e predicts | R2 predicts |
|---|---|---|---|---|---|---|---|---|
| 0 to 0.5 | 1145 | 614 / 1145 (54%) | 56% | 58% | 58% | 57% | 57% | 52% |
| 0.5 to 1 | 947 | 660 / 947 (70%) | 68% | 70% | 70% | 70% | 70% | 67% |
| 1 to 2 | 1366 | 1145 / 1366 (84%) | 81% | 83% | 83% | 84% | 84% | 83% |
| 2 to 4 | 1951 | 1850 / 1951 (95%) | 94% | 96% | 96% | 96% | 96% | 95% |
| 4 to 8 | 2446 | 2439 / 2446 (100%) | 100% | 100% | 100% | 100% | 100% | 100% |
| 8 to 8+ | 1375 | 1375 / 1375 (100%) | 100% | 100% | 100% | 100% | 100% | 100% |

Expected calibration error over those buckets, each rule weighted by the clips it scores: R0 0.011 on 9230 clips, B1 0.008 on 9230 clips, B2 0.009 on 9230 clips, R1 0.007 on 9230 clips, R1e 0.008 on 9230 clips, R2 0.007 on 5045 clips. R2's bucket is the gap at the sighting, not at the advance it reads, so its column is what a host holding that word would have been told one advance later.

### The delay each rule charges

A rule holds the word it was shown until its trust clears the bar, re-reading the partial at every advance; it can only release the word it is holding, so an advance where rank 0 leads with something else never clears. A revision the rule never releases is caught. A good word it never releases is delayed to the final, and charged the audio from the sighting to the end of the clip. The delay columns are milliseconds of audio from the first sighting, over the good words only:

| rule | trust bar | revisions caught | good words: mean ms | median ms | p90 ms | never released |
|---|---|---|---|---|---|---|
| R0 | 0.50 | 0 / 1147 (0%) | 0 | 0 | 0 | 0 / 8083 (0%) |
| R0 | 0.70 | 763 / 1147 (67%) | 26 | 0 | 120 | 452 / 8083 (6%) |
| R0 | 0.80 | 933 / 1147 (81%) | 43 | 0 | 240 | 760 / 8083 (9%) |
| R0 | 0.90 | 1051 / 1147 (92%) | 65 | 0 | 240 | 1136 / 8083 (14%) |
| R0 | 0.95 | 1103 / 1147 (96%) | 85 | 0 | 240 | 1542 / 8083 (19%) |
| R0 | 0.99 | 1143 / 1147 (100%) | 118 | 120 | 240 | 2370 / 8083 (29%) |
| B1 | 0.50 | 0 / 1147 (0%) | 0 | 0 | 0 | 0 / 8083 (0%) |
| B1 | 0.70 | 681 / 1147 (59%) | 22 | 0 | 120 | 376 / 8083 (5%) |
| B1 | 0.80 | 883 / 1147 (77%) | 39 | 0 | 240 | 666 / 8083 (8%) |
| B1 | 0.90 | 1031 / 1147 (90%) | 59 | 0 | 240 | 1031 / 8083 (13%) |
| B1 | 0.95 | 1093 / 1147 (95%) | 78 | 0 | 240 | 1388 / 8083 (17%) |
| B1 | 0.99 | 1140 / 1147 (99%) | 110 | 120 | 240 | 2168 / 8083 (27%) |
| B2 | 0.50 | 164 / 1147 (14%) | 2 | 0 | 0 | 29 / 8083 (0%) |
| B2 | 0.70 | 665 / 1147 (58%) | 21 | 0 | 120 | 317 / 8083 (4%) |
| B2 | 0.80 | 895 / 1147 (78%) | 37 | 0 | 240 | 592 / 8083 (7%) |
| B2 | 0.90 | 1020 / 1147 (89%) | 57 | 0 | 240 | 977 / 8083 (12%) |
| B2 | 0.95 | 1087 / 1147 (95%) | 74 | 0 | 240 | 1301 / 8083 (16%) |
| B2 | 0.99 | 1141 / 1147 (99%) | 112 | 120 | 240 | 2206 / 8083 (27%) |
| R1 | 0.50 | 73 / 1147 (6%) | 1 | 0 | 0 | 16 / 8083 (0%) |
| R1 | 0.70 | 689 / 1147 (60%) | 22 | 0 | 120 | 388 / 8083 (5%) |
| R1 | 0.80 | 887 / 1147 (77%) | 37 | 0 | 240 | 646 / 8083 (8%) |
| R1 | 0.90 | 1044 / 1147 (91%) | 58 | 0 | 240 | 1017 / 8083 (13%) |
| R1 | 0.95 | 1081 / 1147 (94%) | 76 | 0 | 240 | 1360 / 8083 (17%) |
| R1 | 0.99 | 1141 / 1147 (99%) | 110 | 120 | 240 | 2154 / 8083 (27%) |
| R1e | 0.50 | 219 / 1147 (19%) | 3 | 0 | 0 | 34 / 8083 (0%) |
| R1e | 0.70 | 671 / 1147 (59%) | 20 | 0 | 59 | 320 / 8083 (4%) |
| R1e | 0.80 | 897 / 1147 (78%) | 35 | 0 | 240 | 584 / 8083 (7%) |
| R1e | 0.90 | 1031 / 1147 (90%) | 55 | 0 | 240 | 959 / 8083 (12%) |
| R1e | 0.95 | 1081 / 1147 (94%) | 72 | 0 | 240 | 1262 / 8083 (16%) |
| R1e | 0.99 | 1140 / 1147 (99%) | 110 | 120 | 240 | 2158 / 8083 (27%) |
| R2 | 0.50 | 1123 / 1147 (98%) | 183 | 240 | 240 | 3755 / 8083 (46%) |
| R2 | 0.70 | 1127 / 1147 (98%) | 184 | 240 | 240 | 3794 / 8083 (47%) |
| R2 | 0.80 | 1134 / 1147 (99%) | 184 | 240 | 240 | 3832 / 8083 (47%) |
| R2 | 0.90 | 1139 / 1147 (99%) | 185 | 240 | 240 | 3885 / 8083 (48%) |
| R2 | 0.95 | 1144 / 1147 (100%) | 186 | 240 | 240 | 3956 / 8083 (49%) |
| R2 | 0.99 | 1147 / 1147 (100%) | 190 | 240 | 240 | 4135 / 8083 (51%) |

At the README's operating point, trust 0.9 (a gap of about 2.2 nats for R0), the same rules and an exact McNemar on the clips they handle differently. A clip is handled correctly when a revision is held to the final or a good word is released before it. H has no bar; it is the wait:

| rule | clips | correct | revisions caught | good words: mean ms | median ms | p90 ms | never released |
|---|---|---|---|---|---|---|---|
| R0 | 9230 | 7998 / 9230 (86.7%) | 1051 / 1147 (92%) | 65 | 0 | 240 | 1136 / 8083 (14%) |
| B1 | 9230 | 8083 / 9230 (87.6%) | 1031 / 1147 (90%) | 59 | 0 | 240 | 1031 / 8083 (13%) |
| B2 | 9230 | 8126 / 9230 (88.0%) | 1020 / 1147 (89%) | 57 | 0 | 240 | 977 / 8083 (12%) |
| R1 | 9230 | 8110 / 9230 (87.9%) | 1044 / 1147 (91%) | 58 | 0 | 240 | 1017 / 8083 (13%) |
| R1e | 9230 | 8155 / 9230 (88.4%) | 1031 / 1147 (90%) | 55 | 0 | 240 | 959 / 8083 (12%) |
| R2 | 9230 | 5337 / 9230 (57.8%) | 1139 / 1147 (99%) | 185 | 240 | 240 | 3885 / 8083 (48%) |
| H | 9230 | 5451 / 9230 (59.1%) | 1123 / 1147 (98%) | 183 | 240 | 240 | 3755 / 8083 (46%) |
| R2, where it can act | 5045 | 4901 / 5045 (97.1%) | 703 / 711 (99%) | 244 | 240 | 240 | 136 / 4334 (3%) |

| pair | R0 right, other wrong | other right, R0 wrong | exact McNemar p |
|---|---|---|---|
| R0 against B1 | 20 | 105 | 4.13e-15 |
| R0 against B2 | 33 | 161 | 2.01e-21 |
| R0 against R1 | 36 | 148 | 2.51e-17 |
| R0 against R1e | 36 | 193 | 3.76e-27 |
| R0 against R2 | 2784 | 123 | <1e-300 |
| B1 against R1, the motion's own share | 46 | 73 | 0.0168 |

### The frozen operating points

R0 at the README's bar charges a mean 64 ms on the validation split's good words and catches 838 of its 910 revisions there. Each other rule's bar is the one that comes closest to *that* on *that* split, frozen, and then read on the testing clips. The intervals are 95% over clip resamples (1000 of them):

| rule | matched on | bar frozen on the fit | revisions caught | 95% interval | good words: mean ms | 95% interval |
|---|---|---|---|---|---|---|
| B1 | equal delay | 0.920 | 1051 / 1147 (91.6%) | 90.0% to 93.2% | 65 | 63 to 67 |
| B1 | equal catch | 0.920 | 1051 / 1147 (91.6%) | 90.0% to 93.2% | 65 | 63 to 67 |
| B2 | equal delay | 0.930 | 1055 / 1147 (92.0%) | 90.3% to 93.6% | 66 | 63 to 68 |
| B2 | equal catch | 0.925 | 1051 / 1147 (91.6%) | 89.9% to 93.2% | 64 | 62 to 66 |
| R1 | equal delay | 0.925 | 1066 / 1147 (92.9%) | 91.3% to 94.5% | 66 | 63 to 68 |
| R1 | equal catch | 0.920 | 1061 / 1147 (92.5%) | 90.9% to 94.0% | 64 | 62 to 66 |
| R1e | equal delay | 0.935 | 1066 / 1147 (92.9%) | 91.3% to 94.5% | 66 | 64 to 68 |
| R1e | equal catch | 0.925 | 1054 / 1147 (91.9%) | 90.2% to 93.5% | 62 | 60 to 65 |
| R2 | equal delay | 0.005 | 1123 / 1147 (97.9%) | 97.1% to 98.7% | 183 | 181 to 184 |
| R2 | equal catch | 0.005 | 1123 / 1147 (97.9%) | 97.1% to 98.7% | 183 | 181 to 184 |
| R0 | the reference, trust 0.9 | - | 1051 / 1147 (91.6%) | 90.0% to 93.2% | 65 | 63 to 68 |

For comparison, and **descriptive only**, the same two matches made by scanning R1's bar on the testing clips themselves, which is the curve and not an operating point a host could have chosen: against R0's 1051 of 1147 for a mean 65 ms, R1 at 0.925 catches 1066 for 66 ms, and at 0.910 catches 1050 for 61 ms. A bar chosen where it is judged flatters itself; the frozen table above is the one to read.

R2 looks near-perfect where it can act, and the corpus is why: on 85% of the clips that have another advance at all, that advance is the clip's last, so rank 0 there is all but the final and the rule is reading the answer rather than predicting it. H is the same wait with nothing read off it, which is how much of R2 is the wait. What R2 measures on one-second clips is the value of waiting, not the value of the motion, and the wait it charges is the whole remainder of the utterance.

## What the motion figures say

The motion is mostly not there to read. A word's first sighting is the birth of its reading: rank 0 was present at the previous advance on 40.7% of first sightings, and every delta needs that. The deltas the hypothesis is about are therefore undefined on the majority of the moments a host would want them, which is a fact about the corpus as much as about the decoder: at 240 ms an advance and a second of audio there are three advances in a clip, and a word tends to appear at the second or the third.

Where a signal is defined it ranks revisions: the strongest is entropy over the readings at 0.88 on 9230 first sightings, against the gap's own 0.12. But the level is most of what those signals are reading. Held at equal gap the columns fall towards 0.5, the largest average distance left being entropy over the readings at 0.15 over the buckets with 50 clips or more, and one signal at a time in sample the AUC moves from 0.878 to at best 0.884 (+0.006).

Held out, the whole motion set moves the AUC from 0.878 for the README's rule to 0.881, and from 0.878 for the same gap recalibrated and 0.884 for the gap with the readings' entropy beside it; the bootstrap table above gives an interval on each of those differences. Most of the distance from the README's rule is the recalibration, which costs nothing and needs no field: at the README's bar the recalibrated gap alone decides 8083 of 9230 clips correctly where the README's rule decides 7998, and the motion decides 8110. Against the recalibration the motion is +27 clips, 73 to 46 on the clips they disagree about, exact McNemar 0.017, and it catches +13 revisions in the exchange. That is a trade made at one bar, not a safety gain: the frozen table above is where the two are compared at the same delay and the same catch, each bar chosen on the fitting split.

What buys more is the delay, not a new field. Waiting for the next advance costs 240 / 240 ms, is available on 54.7% of first sightings, and on the ones where it is available rank 0 has already changed its first word on 97% of the revisions. Read as a rule that is R2: 0.998 AUC on the clips it can score, 1139 revisions caught at the operating point, for a median 240 ms on the good words. That number is flattered by the corpus: on 85% of those clips the next advance is the last one the clip has, so the hold is nearly the wait for the final.

The vanishing proxy is common: a reading inside two nats of the leader is gone at the next advance on 56.4% of clips before the first word even appears. So the alternatives are a bounded sample of the beam whose mass is not conserved, which is the reason to expect the gap to be approximate rather than a posterior. It is not a warning sign for the word that follows: those clips revise at 7.7% against 18.6% for clips with none, the opposite direction to the obvious guess, because a leader decisive enough to drop its rivals is a leader that holds.

Caveats. Every advance interval on the corpus measured 240 ms, as `[[rr:TD-7#Decision outcome]]` and the chunk of `[[rr:TD-2#The network]]` say it should. 0 blocks after a clip's first advance carried no readings at all, and 0 first sightings fell on a block that was not an advance. The merge count is a proxy: beam pruning removes readings too, and nothing in the output distinguishes the two. One clip decoded eight times in fresh processes gave identical readings and identical finals, so these figures are a property of the build and not of a run; the build is the one the page was measured with, and an older wheel that ordered two readings of equal cost by hash moved a few of the finals. A corpus of sentences would give the motion the room a one-second clip does not, and is where this should be measured again.
