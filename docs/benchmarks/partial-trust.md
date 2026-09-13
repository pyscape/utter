# Reading a partial's trust

Run 2026-09-13 00:21:31Z, utter 71e57bf, utterpy 0.0.1, model vosk-model-small-en-us-0.15, 12th Gen Intel(R) Core(TM) i7-12700F, Python 3.14.4. Decoded by the binding /home/jared/Repos/utterpy/.venv/lib/python3.14/site-packages/utterpy/__init__.py built from utter 71e57bf779540c3eb4c11e05bc38644206c2789c, the checkout's HEAD.

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

The two moments a rule is read at, and the two places the figure comes from. The harness sees the readings the runtime reports, five of them here, and derives the motion from its own series; the runtime keeps the history over every surviving group, so a reading entering the list carries motion the harness has no record of (`[[rr:TD-9#Readings are read once per decoding advance]]`). Every figure below is the runtime's; the harness columns are what a host that differenced its own partials would have had instead, and where both are defined the two agree, which is this page's cross-check on the fields.

| figure | harness top-n, at the sighting | runtime, at the sighting | harness top-n, one advance later | runtime, one advance later |
|---|---|---|---|---|
| the rank-0 reading was there at the previous advance (`had_history`) | 3760 / 9230 (40.7%) | 3760 / 9230 (40.7%) | 4930 / 5045 (97.7%) | 4930 / 5045 (97.7%) |
| `lead_delta` of rank 0 is defined | 3760 / 9230 (40.7%) | 8433 / 9230 (91.4%) | 4930 / 5045 (97.7%) | 5041 / 5045 (99.9%) |
| `lead_delta` of rank 1 is defined | 3698 / 9230 (40.1%) | 7737 / 9230 (83.8%) | 1921 / 5045 (38.1%) | 4029 / 5045 (79.9%) |
| `lead_delta` of the reading that led before is defined | 3468 / 9230 (37.6%) | 3468 / 9230 (37.6%) | 4729 / 5045 (93.7%) | 4729 / 5045 (93.7%) |
| `entropy_delta` is defined (there was a previous advance) | 8473 / 9230 (91.8%) | 8473 / 9230 (91.8%) | 5045 / 5045 (100.0%) | 5045 / 5045 (100.0%) |

The cross-check behind those columns: over every advance of every clip decoded, 41902 readings carried a `lead_delta` from both the runtime and the harness's own derivation and all 41902 agreed, the largest disagreement 2.0e-06 nats against a tolerance of 2e-05 (the confidences print to six decimals and the runtime's leads are single precision); 53043 more carried one from the runtime alone, a reading entering the reported list with a history the harness never saw; and there was no reading the harness could difference and the runtime could not.

The rank-0 reading has stood 1 / 2 advances (median / p90) and is at its first on 59.3% of first sightings. That is the shape of the problem: a word first appears when its reading appears, so at that moment there is usually nothing to have moved.

## Every signal at the first sighting, against a revision

The same rank-AUC as above, revision as the positive label, extended with the motion signals. A signal that is undefined at some first sightings is scored on the ones where it is defined, and that subset is given:

| signal at the first sighting | first sightings scored | rank-AUC |
|---|---|---|
| gap from the top `confidence` to the next | 9230 | 0.12 |
| the top reading's own `confidence` | 9230 | 0.32 |
| `energy_dbfs` under the word | 9230 | 0.55 |
| advances the top reading has stood | 9230 | 0.42 |
| `lead_delta` of rank 0 | 8433 | 0.32 |
| `lead_delta` of rank 1 | 7737 | 0.64 |
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
| `lead_delta` of rank 0 | 0.49 | 0.46 | 0.46 | 0.47 | 0.56 | - |
| `lead_delta` of rank 1 | 0.53 | 0.51 | 0.49 | 0.52 | 0.49 | - |
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
| `lead_delta` of rank 0 | 8433 | 0.885 | 0.885 | -1.088 | -0.0297 |
| `lead_delta` of rank 1 | 7737 | 0.872 | 0.872 | -1.091 | +0.0009 |
| `lead_delta` of the reading that led at the last advance | 3468 | 0.828 | 0.828 | -1.130 | +0.0257 |
| rank-0 changes so far | 9230 | 0.878 | 0.878 | -1.103 | +0.0671 |
| entropy over the readings | 9230 | 0.878 | 0.884 | -0.687 | +1.5468 |
| `entropy_delta` | 8473 | 0.884 | 0.887 | -0.865 | +0.7395 |
| readings in contention that vanished, by now | 9230 | 0.878 | 0.878 | -1.101 | +0.0059 |

Then on all 9230 first sightings, the undefined values set to zero beside an indicator column that says they were undefined, which is what a host that reads the field unconditionally would be doing. The gap alone scores 0.878 on this set:

| signal | AUC, gap + signal + missing | coefficient on gap | on the signal | on missing |
|---|---|---|---|---|
| advances the top reading has stood | 0.880 | -1.137 | +0.4907 | - |
| `lead_delta` of rank 0 | 0.878 | -1.089 | -0.0297 | -0.221 |
| `lead_delta` of rank 1 | 0.878 | -1.102 | +0.0008 | -0.047 |
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
| `lead_delta` of rank 0 | 5041 | 0.73 |
| `lead_delta` of rank 1 | 4029 | 0.35 |
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

### The decoder's own count

The proxy above reads the readings that were reported. The decoder can count what it dropped, and `stream --census` does, without touching a decision. Three quantities, kept apart because they are not the same event: `merges_close`, two paths meeting at one state with different word sequences and costs within 2 nats, counted whichever of the two is dropped; `readings_lost`, word sequences present among the surviving groups at one chunk and gone at the next; and `readings_lost_close`, those of them that were within 2 nats of the leader when last seen. A collision is not a reading lost: the loser's word sequence usually survives elsewhere in the beam, and neither path need be anywhere near the leader.

Over 9230 clips, per clip: 815 close collisions (780 / 1180 by median / p90), 165.5 readings lost, and 0.53 of those lost from within 2 nats of the leader. Collisions are not rare events to be correlated with anything: 9230 of 9230 clips have at least one, so the question is how many, not whether.

Revision rate of the first word by how many close collisions the clip's decode recorded, in quartiles:

| close collisions in the decode | clips | revised |
|---|---|---|
| 0 to 616 | 2299 | 196 / 2299 (8.5%) |
| 616 to 780 | 2310 | 237 / 2310 (10.3%) |
| 780 to 980 | 2309 | 290 / 2309 (12.6%) |
| 980 and up | 2312 | 424 / 2312 (18.3%) |

And by the quantity the proxy above was reaching for, a reading lost from within 2 nats of the leader, which the decoder sees over every surviving group and the proxy sees only among the five reported:

| clips | revised |
|---|---|
| at least one close reading lost | 556 / 4090 (13.6%) |
| none | 591 / 5140 (11.5%) |

Read this as an observation and not as a verdict. It counts collisions and disappearances; what a retained history or a summed posterior would have been worth is a measurement of its own, which the record defers rather than closes.

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
| R1, the gap and the motion | 9230 | 0.877 |
| R1e, the gap, the entropy and the motion | 9230 | 0.885 |
| R2, R1 one advance later, as the rule runs | 5045 | 0.998 |
| H, hold one advance, reading nothing | 5045 | 0.982 |
| R2, the later reading alone | 5045 | 0.728 |

B1 orders the clips exactly as R0 does, a logistic of one column being monotone in it, so its AUC difference is zero by construction and everything it buys is in the calibration and the decisions below. Scores are oriented as P(revision), so higher is better and 0.5 is no information. R2 and H are scored only on the clips that have another advance, and both are read as they run: a word rank 0 no longer leads with is a revision called with certainty, which is where nearly all of their information is. H carries no score of its own, so its AUC is that call and nothing else. Scoring R2's later reading on its own gap and motion, as if the word were still on offer, is worse than R0 because the gap keeps growing after the word's fate is settled. The paired bootstrap resamples clips 1000 times, both rules seeing the same resample:

| difference | clips | AUC difference | 95% interval |
|---|---|---|---|
| B1 - R0, the recalibration alone | 9230 | 0.0000 | 0.0000 to 0.0000 |
| B2 - B1, the entropy over it | 9230 | 0.0062 | 0.0035 to 0.0091 |
| R1 - R0, the motion against the README | 9230 | -0.0008 | -0.0022 to 0.0006 |
| R1 - B1, the motion over the recalibration | 9230 | -0.0008 | -0.0022 to 0.0006 |
| R1 - B2, the motion alone against the entropy alone | 9230 | -0.0071 | -0.0101 to -0.0040 |
| R1e - B2, the motion added to the entropy | 9230 | 0.0013 | -0.0004 to 0.0031 |
| R2 - R0, as the rule runs | 5045 | 0.1337 | 0.1218 to 0.1455 |
| H - R0, the wait alone | 5045 | 0.1177 | 0.1059 to 0.1299 |

Where the R1 coefficients come from moves it little: fit on the validation split it scores 0.877 on all 9230 testing clips; fit on the even-indexed half of testing it scores 0.876 on the odd half; fit and scored on testing itself, 0.878. The fitted coefficients, each in the order named:

| rule | order | coefficients |
|---|---|---|
| gap | intercept, gap | -0.0706, -1.0816 |
| gap + entropy | intercept, gap, entropy | -1.7228, -0.7114, +1.3611 |
| gap + motion | intercept, gap, displaced `lead_delta`, its missing flag, rank-0 `lead_delta`, its missing flag | +0.3324, -1.0797, +0.0242, -0.0978, -0.0260, -0.5460 |
| gap + entropy + motion | intercept, gap, entropy, displaced `lead_delta`, its missing flag, rank-0 `lead_delta`, its missing flag | -1.6712, -0.6442, +1.6384, -0.0281, +0.3606, -0.0723, -1.0547 |

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
| R1 | 0 to 0.5 | 72 | 49% | 35% |
| R1 | 0.5 to 0.7 | 1508 | 61% | 58% |
| R1 | 0.7 to 0.8 | 835 | 75% | 75% |
| R1 | 0.8 to 0.9 | 999 | 85% | 85% |
| R1 | 0.9 to 0.95 | 761 | 93% | 93% |
| R1 | 0.95 to 1 | 5055 | 99% | 99% |
| R1 | **expected calibration error** | 9230 | | **0.008** |
| R1e | 0 to 0.5 | 286 | 45% | 34% |
| R1e | 0.5 to 0.7 | 1188 | 61% | 61% |
| R1e | 0.7 to 0.8 | 877 | 75% | 72% |
| R1e | 0.8 to 0.9 | 934 | 85% | 86% |
| R1e | 0.9 to 0.95 | 722 | 93% | 93% |
| R1e | 0.95 to 1 | 5223 | 99% | 99% |
| R1e | **expected calibration error** | 9230 | | **0.010** |
| R2 | 0 to 0.5 | 717 | 2% | 4% |
| R2 | 0.5 to 0.7 | 76 | 60% | 87% |
| R2 | 0.7 to 0.8 | 42 | 75% | 98% |
| R2 | 0.8 to 0.9 | 75 | 86% | 89% |
| R2 | 0.9 to 0.95 | 65 | 93% | 98% |
| R2 | 0.95 to 1 | 4070 | 100% | 100% |
| R2 | **expected calibration error** | 5045 | | **0.012** |

The same calibration read in the gap's own buckets, the ones the table at the top of the page uses, so a rule that is better ordered but worse calibrated shows as one:

| gap at the sighting (nats) | clips | survived | R0 predicts | B1 predicts | B2 predicts | R1 predicts | R1e predicts | R2 predicts |
|---|---|---|---|---|---|---|---|---|
| 0 to 0.5 | 1145 | 614 / 1145 (54%) | 56% | 58% | 58% | 58% | 58% | 51% |
| 0.5 to 1 | 947 | 660 / 947 (70%) | 68% | 70% | 70% | 70% | 70% | 66% |
| 1 to 2 | 1366 | 1145 / 1366 (84%) | 81% | 83% | 83% | 83% | 83% | 82% |
| 2 to 4 | 1951 | 1850 / 1951 (95%) | 94% | 96% | 96% | 96% | 96% | 95% |
| 4 to 8 | 2446 | 2439 / 2446 (100%) | 100% | 100% | 100% | 100% | 100% | 99% |
| 8 to 8+ | 1375 | 1375 / 1375 (100%) | 100% | 100% | 100% | 100% | 100% | 100% |

Expected calibration error over those buckets, each rule weighted by the clips it scores: R0 0.011 on 9230 clips, B1 0.008 on 9230 clips, B2 0.009 on 9230 clips, R1 0.008 on 9230 clips, R1e 0.010 on 9230 clips, R2 0.012 on 5045 clips. R2's bucket is the gap at the sighting, not at the advance it reads, so its column is what a host holding that word would have been told one advance later.

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
| R1 | 0.50 | 47 / 1147 (4%) | 1 | 0 | 0 | 13 / 8083 (0%) |
| R1 | 0.70 | 675 / 1147 (59%) | 22 | 0 | 120 | 422 / 8083 (5%) |
| R1 | 0.80 | 878 / 1147 (77%) | 38 | 0 | 240 | 691 / 8083 (9%) |
| R1 | 0.90 | 1036 / 1147 (90%) | 59 | 0 | 240 | 1079 / 8083 (13%) |
| R1 | 0.95 | 1089 / 1147 (95%) | 77 | 0 | 240 | 1425 / 8083 (18%) |
| R1 | 0.99 | 1141 / 1147 (99%) | 111 | 120 | 240 | 2206 / 8083 (27%) |
| R1e | 0.50 | 188 / 1147 (16%) | 2 | 0 | 0 | 35 / 8083 (0%) |
| R1e | 0.70 | 654 / 1147 (57%) | 20 | 0 | 16 | 353 / 8083 (4%) |
| R1e | 0.80 | 900 / 1147 (78%) | 36 | 0 | 240 | 622 / 8083 (8%) |
| R1e | 0.90 | 1030 / 1147 (90%) | 56 | 0 | 240 | 997 / 8083 (12%) |
| R1e | 0.95 | 1081 / 1147 (94%) | 73 | 0 | 240 | 1312 / 8083 (16%) |
| R1e | 0.99 | 1141 / 1147 (99%) | 111 | 120 | 240 | 2222 / 8083 (27%) |
| R2 | 0.50 | 1125 / 1147 (98%) | 183 | 240 | 240 | 3777 / 8083 (47%) |
| R2 | 0.70 | 1135 / 1147 (99%) | 184 | 240 | 240 | 3841 / 8083 (48%) |
| R2 | 0.80 | 1136 / 1147 (99%) | 185 | 240 | 240 | 3881 / 8083 (48%) |
| R2 | 0.90 | 1144 / 1147 (100%) | 186 | 240 | 240 | 3944 / 8083 (49%) |
| R2 | 0.95 | 1145 / 1147 (100%) | 187 | 240 | 240 | 4005 / 8083 (50%) |
| R2 | 0.99 | 1147 / 1147 (100%) | 191 | 240 | 240 | 4205 / 8083 (52%) |

At the README's operating point, trust 0.9 (a gap of about 2.2 nats for R0), the same rules and an exact McNemar on the clips they handle differently. A clip is handled correctly when a revision is held to the final or a good word is released before it. H has no bar; it is the wait:

| rule | clips | correct | revisions caught | good words: mean ms | median ms | p90 ms | never released |
|---|---|---|---|---|---|---|---|
| R0 | 9230 | 7998 / 9230 (86.7%) | 1051 / 1147 (92%) | 65 | 0 | 240 | 1136 / 8083 (14%) |
| B1 | 9230 | 8083 / 9230 (87.6%) | 1031 / 1147 (90%) | 59 | 0 | 240 | 1031 / 8083 (13%) |
| B2 | 9230 | 8126 / 9230 (88.0%) | 1020 / 1147 (89%) | 57 | 0 | 240 | 977 / 8083 (12%) |
| R1 | 9230 | 8040 / 9230 (87.1%) | 1036 / 1147 (90%) | 59 | 0 | 240 | 1079 / 8083 (13%) |
| R1e | 9230 | 8116 / 9230 (87.9%) | 1030 / 1147 (90%) | 56 | 0 | 240 | 997 / 8083 (12%) |
| R2 | 9230 | 5283 / 9230 (57.2%) | 1144 / 1147 (100%) | 186 | 240 | 240 | 3944 / 8083 (49%) |
| H | 9230 | 5451 / 9230 (59.1%) | 1123 / 1147 (98%) | 183 | 240 | 240 | 3755 / 8083 (46%) |
| R2, where it can act | 5045 | 4847 / 5045 (96.1%) | 708 / 711 (100%) | 246 | 240 | 240 | 195 / 4334 (4%) |

| pair | R0 right, other wrong | other right, R0 wrong | exact McNemar p |
|---|---|---|---|
| R0 against B1 | 20 | 105 | 4.13e-15 |
| R0 against B2 | 33 | 161 | 2.01e-21 |
| R0 against R1 | 26 | 68 | 1.73e-05 |
| R0 against R1e | 29 | 147 | 3.36e-20 |
| R0 against R2 | 2808 | 93 | <1e-300 |
| B1 against R1, the motion's own share | 65 | 22 | 4.35e-06 |

### The frozen operating points

R0 at the README's bar charges a mean 64 ms on the validation split's good words and catches 838 of its 910 revisions there. Each other rule's bar is the one that comes closest to *that* on *that* split, frozen, and then read on the testing clips. The intervals are 95% over clip resamples (1000 of them):

| rule | matched on | bar frozen on the fit | revisions caught | 95% interval | good words: mean ms | 95% interval |
|---|---|---|---|---|---|---|
| B1 | equal delay | 0.920 | 1051 / 1147 (91.6%) | 90.0% to 93.2% | 65 | 63 to 67 |
| B1 | equal catch | 0.920 | 1051 / 1147 (91.6%) | 90.0% to 93.2% | 65 | 63 to 67 |
| B2 | equal delay | 0.930 | 1055 / 1147 (92.0%) | 90.3% to 93.6% | 66 | 63 to 68 |
| B2 | equal catch | 0.925 | 1051 / 1147 (91.6%) | 89.9% to 93.2% | 64 | 62 to 66 |
| R1 | equal delay | 0.920 | 1058 / 1147 (92.2%) | 90.6% to 93.8% | 65 | 63 to 67 |
| R1 | equal catch | 0.920 | 1058 / 1147 (92.2%) | 90.6% to 93.8% | 65 | 63 to 67 |
| R1e | equal delay | 0.935 | 1065 / 1147 (92.9%) | 91.2% to 94.4% | 67 | 64 to 69 |
| R1e | equal catch | 0.925 | 1054 / 1147 (91.9%) | 90.3% to 93.5% | 63 | 61 to 65 |
| R2 | equal delay | 0.005 | 1123 / 1147 (97.9%) | 97.1% to 98.7% | 183 | 181 to 184 |
| R2 | equal catch | 0.005 | 1123 / 1147 (97.9%) | 97.1% to 98.7% | 183 | 181 to 184 |
| R0 | the reference, trust 0.9 | - | 1051 / 1147 (91.6%) | 90.0% to 93.2% | 65 | 63 to 68 |

For comparison, and **descriptive only**, the same two matches made by scanning R1's bar on the testing clips themselves, which is the curve and not an operating point a host could have chosen: against R0's 1051 of 1147 for a mean 65 ms, R1 at 0.920 catches 1058 for 65 ms, and at 0.910 catches 1051 for 62 ms. A bar chosen where it is judged flatters itself; the frozen table above is the one to read.

R2 looks near-perfect where it can act, and the corpus is why: on 85% of the clips that have another advance at all, that advance is the clip's last, so rank 0 there is all but the final and the rule is reading the answer rather than predicting it. H is the same wait with nothing read off it, which is how much of R2 is the wait. What R2 measures on one-second clips is the value of waiting, not the value of the motion, and the wait it charges is the whole remainder of the utterance.

## What the motion figures say

The motion is mostly not there to read. A word's first sighting is the birth of its reading: rank 0 was present at the previous advance on 40.7% of first sightings, and every delta needs that. The deltas the hypothesis is about are therefore undefined on the majority of the moments a host would want them, which is a fact about the corpus as much as about the decoder: at 240 ms an advance and a second of audio there are three advances in a clip, and a word tends to appear at the second or the third.

Where a signal is defined it ranks revisions: the strongest is entropy over the readings at 0.88 on 9230 first sightings, against the gap's own 0.12. But the level is most of what those signals are reading. Held at equal gap the columns fall towards 0.5, the largest average distance left being entropy over the readings at 0.15 over the buckets with 50 clips or more, and one signal at a time in sample the AUC moves from 0.878 to at best 0.884 (+0.006).

Held out, the whole motion set moves the AUC from 0.878 for the README's rule to 0.877, and from 0.878 for the same gap recalibrated and 0.884 for the gap with the readings' entropy beside it; the bootstrap table above gives an interval on each of those differences. Most of the distance from the README's rule is the recalibration, which costs nothing and needs no field: at the README's bar the recalibrated gap alone decides 8083 of 9230 clips correctly where the README's rule decides 7998, and the motion decides 8040. Against the recalibration the motion is -43 clips, 22 to 65 on the clips they disagree about, exact McNemar 4.3e-06, and it catches +5 revisions in the exchange. That is a trade made at one bar, not a safety gain: the frozen table above is where the two are compared at the same delay and the same catch, each bar chosen on the fitting split.

What buys more is the delay, not a new field. Waiting for the next advance costs 240 / 240 ms, is available on 54.7% of first sightings, and on the ones where it is available rank 0 has already changed its first word on 97% of the revisions. Read as a rule that is R2: 0.998 AUC on the clips it can score, 1144 revisions caught at the operating point, for a median 240 ms on the good words. That number is flattered by the corpus: on 85% of those clips the next advance is the last one the clip has, so the hold is nearly the wait for the final.

The vanishing proxy is common: a reading inside two nats of the leader is gone at the next advance on 56.4% of clips before the first word even appears. So the alternatives are a bounded sample of the beam whose mass is not conserved, which is the reason to expect the gap to be approximate rather than a posterior. It is not a warning sign for the word that follows: those clips revise at 7.7% against 18.6% for clips with none, the opposite direction to the obvious guess, because a leader decisive enough to drop its rivals is a leader that holds.

Caveats. Every advance interval on the corpus measured 240 ms, as `[[rr:TD-7#Decision outcome]]` and the chunk of `[[rr:TD-2#The network]]` say it should. 0 blocks after a clip's first advance carried no readings at all, and 0 first sightings fell on a block that was not an advance. The merge count is a proxy: beam pruning removes readings too, and nothing in the output distinguishes the two; the decoder's own count is beside it above. The whole split was decoded twice, each pass in a process started for it, comparing every partial with its readings block by block and every final, and no clip read differently the second time (`scripts/speech_commands.py` `determinism_pass`). Two processes are what the comparison needs: a hash seed is drawn once per process, so a reading that depends on a map's iteration order is perfectly stable within a run and moves only between runs, and a repeat inside one process cannot see it. These figures are therefore a property of the build, which is the build this page names; an older wheel that ordered two readings of equal cost by hash moved a few of the finals. A corpus of sentences would give the motion the room a one-second clip does not, and is where this should be measured again.
