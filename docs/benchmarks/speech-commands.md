# Speech Commands v2, testing split, 11005 clips, 40 ms blocks

Grammar: the dataset's 35 words plus 26 letters, 26 NATO words and 5 colours, 92 entries.

Run 2026-09-12 03:36:20Z, utter 1e88e2c, vosk 0.3.45, utterpy 0.0.1, model vosk-model-small-en-us-0.15, 12th Gen Intel(R) Core(TM) i7-12700F, Python 3.14.4.

## Full grammar: accuracy

| engine | correct | accuracy | never in a partial | first appearance after the clip's energy end p50 / p90 | first word shown later changed |
|---|---|---|---|---|---|
| vosk | 10070 / 11005 | 91.50% | 2445 | 40 / 250 ms over 8560 | 1138 / 9220 (12.3%) |
| utterpy | 10102 / 11005 | 91.79% | 2432 | 40 / 250 ms over 8573 | 1149 / 9230 (12.4%) |

Compute, one recognizer per clip:

| engine | first construction, ms | every later one, ms | mean, ms | RTF, decode only | RTF with construction |
|---|---|---|---|---|---|
| vosk | 0.38 | 0.286 | 0.286 | 0.0167 | 0.0170 |
| utterpy | 2.30 | 0.024 | 0.024 | 0.0197 | 0.0197 |

| word | vosk | utterpy |
|---|---|---|
| yes | 387 / 419 | 387 / 419 |
| no | 395 / 405 | 395 / 405 |
| up | 347 / 425 | 348 / 425 |
| down | 378 / 406 | 377 / 406 |
| left | 387 / 412 | 389 / 412 |
| right | 360 / 396 | 361 / 396 |
| on | 363 / 396 | 362 / 396 |
| off | 341 / 402 | 341 / 402 |
| stop | 401 / 411 | 402 / 411 |
| go | 377 / 402 | 382 / 402 |
| zero | 382 / 418 | 388 / 418 |
| one | 368 / 399 | 368 / 399 |
| two | 399 / 424 | 403 / 424 |
| three | 355 / 405 | 356 / 405 |
| four | 361 / 400 | 360 / 400 |
| five | 423 / 445 | 425 / 445 |
| six | 367 / 394 | 368 / 394 |
| seven | 388 / 406 | 389 / 406 |
| eight | 315 / 408 | 322 / 408 |
| nine | 392 / 408 | 394 / 408 |
| backward | 152 / 165 | 152 / 165 |
| forward | 133 / 155 | 135 / 155 |
| follow | 163 / 172 | 163 / 172 |
| learn | 137 / 161 | 137 / 161 |
| visual | 141 / 165 | 138 / 165 |
| bed | 190 / 207 | 191 / 207 |
| bird | 175 / 185 | 175 / 185 |
| cat | 177 / 194 | 177 / 194 |
| dog | 206 / 220 | 206 / 220 |
| happy | 192 / 203 | 192 / 203 |
| house | 181 / 191 | 181 / 191 |
| marvin | 180 / 195 | 180 / 195 |
| sheila | 198 / 212 | 198 / 212 |
| tree | 171 / 193 | 171 / 193 |
| wow | 188 / 206 | 189 / 206 |

Most frequent confusions, vosk: up -> (nothing) (67); eight -> a (50); off -> o (34); three -> tree (27); eight -> (nothing) (25); four -> forward (20); forward -> four (17); down -> (nothing) (16); cat -> (nothing) (15); on -> (nothing) (14)
Most frequent confusions, utterpy: up -> (nothing) (65); eight -> a (48); off -> o (31); three -> tree (29); eight -> (nothing) (25); four -> forward (20); on -> (nothing) (18); down -> (nothing) (16); cat -> (nothing) (15); off -> (nothing) (14)

## Where the engines differ

Paired over the same clips: `vosk only` counts clips vosk got right and utterpy did not, `utterpy only` the reverse. The p-value is a two-sided exact McNemar test on those two counts; a large one means the split is what chance would produce.

| scope | vosk only | utterpy only | p | Holm |
|---|---|---|---|---|
| all words | 19 | 51 | 0.000 | |
| zero | 0 | 6 | 0.031 | 1.000 |
| eight | 1 | 8 | 0.039 | 1.000 |

35 words were tested, so about two would fall below 0.05 by chance; the last column is the Holm-Bonferroni adjustment for that. No word survives it.

## Compute on continuous audio

One recognizer over 295 s of the clips joined end to end.

| engine | RTF | per-block compute ms p50 / p95 / p99 |
|---|---|---|
| vosk | 0.0123 | 0.049 / 2.85 / 3.29 |
| utterpy | 0.0182 | 0.107 / 4.10 / 4.55 |

## Accuracy against noise

800 clips of the testing split, the dataset's own background recordings laid under each at the stated signal-to-noise ratio, the same mixed samples to both engines.

| engine | 20 dB | 10 dB | 5 dB | 0 dB | clean |
|---|---|---|---|---|---|
| vosk | 699 (87.4%) | 634 (79.2%) | 558 (69.8%) | 399 (49.9%) | 730 (91.2%) |
| utterpy | 702 (87.8%) | 634 (79.2%) | 560 (70.0%) | 400 (50.0%) | 730 (91.2%) |

Paired at each level, as above: `vosk only`, `utterpy only`, and the exact McNemar p.

| level | vosk only | utterpy only | p |
|---|---|---|---|
| 20 dB | 2 | 5 | 0.453 |
| 10 dB | 6 | 6 | 1.000 |
| 5 dB | 6 | 8 | 0.791 |
| 0 dB | 7 | 8 | 1.000 |
| clean | 2 | 2 | 1.000 |

## Grammar size

400 clips, the 35 dataset words plus filler up to each size, so every clip stays decidable at every size. The runtime refuses 300 distinct words or more; the sweep stops at 246, the filler this model's table knows.

| entries | vosk accuracy | vosk first ms | vosk later ms | vosk RTF | utterpy accuracy | utterpy first ms | utterpy later ms | utterpy RTF |
|---|---|---|---|---|---|---|---|---|
| 35 | 94.0% | 0.23 | 0.191 | 0.0153 | 94.5% | 0.84 | 0.015 | 0.0201 |
| 60 | 92.5% | 0.29 | 0.227 | 0.0164 | 92.8% | 1.44 | 0.019 | 0.0204 |
| 92 | 92.0% | 0.35 | 0.288 | 0.0169 | 92.2% | 2.31 | 0.024 | 0.0197 |
| 150 | 91.0% | 0.45 | 0.388 | 0.0175 | 92.0% | 3.59 | 0.036 | 0.0199 |
| 200 | 90.5% | 0.57 | 0.467 | 0.0181 | 91.5% | 5.19 | 0.044 | 0.0200 |
| 246 | 88.8% | 0.63 | 0.527 | 0.0186 | 89.8% | 6.45 | 0.063 | 0.0201 |

## Twelve-class: ten commands plus the unknown-word symbol

| engine | commands correct | fillers and digits read as unknown | fillers and digits read as a command | noise seconds | noise seconds with a word |
|---|---|---|---|---|---|
| vosk | 3916 / 4074 (96.12%) | 5767 / 6931 (83.21%) | 678 (9.78%) | 398 | 0 |
| utterpy | 3920 / 4074 (96.22%) | 5815 / 6931 (83.90%) | 633 (9.13%) | 398 | 0 |

## Background noise under the full grammar

| engine | grammar | noise minutes | partial blocks | blocks with a word at rank 0 | phantom final words per minute | word blocks with the silence reading among the rivals |
|---|---|---|---|---|---|---|
| vosk | full | 6.6 | 9950 | 0 (0.00%) | 0.0 | no alternatives |
| vosk | full + [unk] | 6.6 | 9950 | 0 (0.00%) | 0.0 | no alternatives |
| utterpy | full | 6.6 | 9950 | 0 (0.00%) | 0.0 | 0 / 0 |
| utterpy | full + [unk] | 6.6 | 9950 | 0 (0.00%) | 0.0 | 0 / 0 |
## Reading

Accuracy and first-appearance latency are at parity with the stock
wheel, and the finals agree on 99.15% of clips. Where they differ, the
runtime is now the more accurate of the two by 32 clips, which a paired
test calls real. That is a divergence from the oracle, in the direction
of being right: `[[rr:TD-6]]` reproduces libvosk's rescaling of graph
cost over token chains rather than over a determinized lattice, so it
recovers the readings libvosk rescues from silence without reproducing
all of the word-against-word errors the same rescaling costs libvosk.
Closing that gap needs the lattice `[[rr:TD-2#Decision outcome]]` puts
out of scope.

Before TD-6 the runtime read 458 clips as silence against libvosk's
414, a one-way excess of 46; it now reads 412. No word's difference
survives correction for having tested all 35.

Compute is the standing gap: 1.5 times libvosk on continuous audio,
4.10 ms against 2.85 at the 95th percentile. Construction is the other
way around, 0.024 ms against 0.286, because a compiled grammar is kept
(`[[rr:TD-4]]`); the first construction is dearer and grows with the
grammar, 0.84 ms at 35 entries to 6.45 at 246.

Two figures a host should read before choosing a grammar. Accuracy
falls about five points from 35 entries to 246, so distractors are not
free. And one first partial in eight is later revised, on both
engines, so a host that acts on the first word it sees acts wrongly
about 12% of the time.

Noise costs both engines the same. Accuracy halves between clean and
0 dB, and the paired test finds no difference between the engines at
any level. An earlier run showed the runtime 2.3 points behind at
0 dB; that was TD-6's bug, not the front end.
