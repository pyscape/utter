# Speech Commands v2, vosk-model-small-en-us-0.15, testing split, 11005 clips, 40 ms blocks

Grammar: the dataset's 35 words plus 26 letters, 26 NATO words and 5 colours, 92 entries.

Run 2026-09-12 06:33:41Z, utter 5cee590, vosk 0.3.45, utterpy 0.0.1, model vosk-model-small-en-us-0.15, 12th Gen Intel(R) Core(TM) i7-12700F, Python 3.14.4.

## Engines

Each engine opened the model and decoded a clip before the run began; a model an engine cannot decode stops the run here rather than part-way through a pass.

| engine | model load, s | first recognizer, ms |
|---|---|---|
| vosk | 0.18 | 0.4 |
| utterpy | 0.13 | 22.8 |

## Full grammar: accuracy

| engine | correct | accuracy | never in a partial | first appearance after the clip's energy end p50 / p90 | first word shown later changed |
|---|---|---|---|---|---|
| vosk | 10069 / 11005 | 91.49% | 2451 | 40 / 250 ms over 8554 | 1137 / 9218 (12.3%) |
| utterpy | 10102 / 11005 | 91.79% | 2432 | 40 / 250 ms over 8573 | 1147 / 9230 (12.4%) |

Compute, one recognizer per clip:

| engine | first construction, ms | every later one, ms | mean, ms | RTF, decode only | RTF with construction |
|---|---|---|---|---|---|
| vosk | 0.32 | 0.288 | 0.288 | 0.0167 | 0.0170 |
| utterpy | 2.29 | 0.026 | 0.027 | 0.0140 | 0.0140 |

| word | vosk | utterpy |
|---|---|---|
| yes | 387 / 419 | 387 / 419 |
| no | 395 / 405 | 395 / 405 |
| up | 348 / 425 | 348 / 425 |
| down | 377 / 406 | 377 / 406 |
| left | 387 / 412 | 389 / 412 |
| right | 360 / 396 | 361 / 396 |
| on | 363 / 396 | 362 / 396 |
| off | 340 / 402 | 341 / 402 |
| stop | 401 / 411 | 402 / 411 |
| go | 377 / 402 | 382 / 402 |
| zero | 382 / 418 | 388 / 418 |
| one | 368 / 399 | 368 / 399 |
| two | 400 / 424 | 403 / 424 |
| three | 354 / 405 | 356 / 405 |
| four | 361 / 400 | 360 / 400 |
| five | 424 / 445 | 425 / 445 |
| six | 367 / 394 | 368 / 394 |
| seven | 388 / 406 | 389 / 406 |
| eight | 313 / 408 | 322 / 408 |
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
| wow | 189 / 206 | 189 / 206 |

Most frequent confusions, vosk: up -> (nothing) (66); eight -> a (51); off -> o (35); three -> tree (28); eight -> (nothing) (25); four -> forward (21); forward -> four (17); down -> (nothing) (16); cat -> (nothing) (15); on -> (nothing) (14)
Most frequent confusions, utterpy: up -> (nothing) (65); eight -> a (48); off -> o (31); three -> tree (29); eight -> (nothing) (25); four -> forward (20); on -> (nothing) (18); down -> (nothing) (16); cat -> (nothing) (15); off -> (nothing) (14)

## Agreement with the oracle

The contract is to reproduce vosk's output, which the accuracy above cannot measure: two engines are at parity when they are right equally often, and that is compatible with their being right about different clips. This counts the clips whose finals are identical word for word. Every clip is in `speech-commands.clips.jsonl` with both readings, so the ones that differ can be listed rather than only counted.

| pair | finals identical | differ |
|---|---|---|
| vosk vs utterpy | 10913 / 11005 (99.16%) | 92 |

Of the 92 that differ, vosk reads the label and utterpy does not on 17, utterpy and not vosk on 50, and neither on 25.

| engine | empty finals | empty here, a word from the other |
|---|---|---|
| vosk | 411 (3.73%) | 5 |
| utterpy | 412 (3.74%) | 6 |

The last column is the directional one. Equal totals in the middle column can still hide two engines falling silent on different clips, and an excess in one direction is the signature of a decoder dropping a reading the other keeps.

## Where the engines differ

Paired over the same clips: `vosk only` counts clips vosk got right and utterpy did not, `utterpy only` the reverse. The p-value is a two-sided exact McNemar test on those two counts; a large one means the split is what chance would produce.

| scope | vosk only | utterpy only | p | Holm |
|---|---|---|---|---|
| all words | 17 | 50 | 0.000 | |
| eight | 0 | 9 | 0.004 | 0.137 |
| zero | 0 | 6 | 0.031 | 1.000 |

35 words were tested, so about two would fall below 0.05 by chance; the last column is the Holm-Bonferroni adjustment for that. No word survives it.

## Determinism

200 clips decoded twice in this process and once in a process started for the purpose, comparing the whole partial trace and the final. The fresh process is the half that matters: a hash seed drawn once per process gives a map the same iteration order for every repeat within a run, so a reading that depends on it is perfectly stable until the next run. Any clip that moves is named in the JSON.

| engine | clips | same process: partials | same process: finals | fresh process: partials | fresh process: finals |
|---|---|---|---|---|---|
| vosk | 200 | 0 | 0 | 0 | 0 |
| utterpy | 200 | 0 | 0 | 0 | 0 |

Every clip read the same way every time.

## Endpoint latency

800 clips, each followed by 2000 ms of silence, measured to the first block at which the engine closes a segment itself. Reported against the clip's own energy end, the same reference the first-appearance figure uses. Neither engine endpoints within the clip alone, so without the silence there is nothing here to measure.

| engine | endpointed | after the clip's energy end p50 / p90 / p99 | never within the silence |
|---|---|---|---|
| vosk | 781 / 800 | 870 / 1010 / 1150 ms over 781 | 19 |
| utterpy | 781 / 800 | 870 / 1010 / 1150 ms over 781 | 19 |

Both engines endpointed on 781 of the 800, at the same block on 764 (97.82%). Endpointing is a second surface the contract covers, and one the accuracy figures do not touch: a host that agreed on every word would still feel a difference here.

## Block size

400 clips at each block size, the same clips throughout. A word can only appear at a block boundary, but the decoder only revises a partial when the network has consumed a whole chunk, so the block is the coarser of the two grids only when it fails to land on the chunk. A size that does not divide the chunk's spacing is included deliberately: without one, the sweep cannot tell a decoder that ignores block size from a sweep that happened to pick sizes it agrees with. The accuracy column is the one that should not move.

| block ms | engine | accuracy | first appearance p50 / p90 | same instant as the finest block | RTF, decode only |
|---|---|---|---|---|---|
| 10 | vosk | 368 / 400 (92.0%) | 30 / 240 ms | 317 / 317 | 0.0171 |
| 10 | utterpy | 369 / 400 (92.2%) | 30 / 240 ms | 319 / 319 | 0.0143 |
| 20 | vosk | 368 / 400 (92.0%) | 30 / 240 ms | 317 / 317 | 0.0170 |
| 20 | utterpy | 369 / 400 (92.2%) | 30 / 240 ms | 319 / 319 | 0.0143 |
| 40 | vosk | 368 / 400 (92.0%) | 30 / 240 ms | 317 / 317 | 0.0168 |
| 40 | utterpy | 369 / 400 (92.2%) | 30 / 240 ms | 319 / 319 | 0.0142 |
| 80 | vosk | 368 / 400 (92.0%) | 30 / 240 ms | 317 / 317 | 0.0168 |
| 80 | utterpy | 369 / 400 (92.2%) | 30 / 240 ms | 319 / 319 | 0.0141 |
| 100 | vosk | 368 / 400 (92.0%) | 80 / 280 ms | 26 / 317, up to 60 ms later | 0.0168 |
| 100 | utterpy | 368 / 400 (92.0%) | 80 / 280 ms | 26 / 319, up to 60 ms later | 0.0142 |

The column that carries the result is the per-clip one, not the quantiles: two block sizes can produce the same p50 by luck, and only a clip-by-clip comparison against the finest block (10 ms) shows whether the word actually appeared at the same instant.

## Compute on continuous audio

One recognizer over 295 s of the clips joined end to end.

| engine | RTF | per-block compute ms p50 / p95 / p99 |
|---|---|---|
| vosk | 0.0123 | 0.049 / 2.84 / 3.28 |
| utterpy | 0.0115 | 0.090 / 2.43 / 2.69 |

## Accuracy against noise

800 clips of the testing split, the dataset's own background recordings laid under each at the stated signal-to-noise ratio, the same mixed samples to both engines.

| engine | 20 dB | 10 dB | 5 dB | 0 dB | clean |
|---|---|---|---|---|---|
| vosk | 699 (87.4%) | 634 (79.2%) | 558 (69.8%) | 399 (49.9%) | 729 (91.1%) |
| utterpy | 702 (87.8%) | 634 (79.2%) | 560 (70.0%) | 400 (50.0%) | 730 (91.2%) |

Paired at each level, as above: `vosk only`, `utterpy only`, and the exact McNemar p.

| level | vosk only | utterpy only | p |
|---|---|---|---|
| 20 dB | 2 | 5 | 0.453 |
| 10 dB | 6 | 6 | 1.000 |
| 5 dB | 6 | 8 | 0.791 |
| 0 dB | 7 | 8 | 1.000 |
| clean | 1 | 2 | 1.000 |

## Grammar size

400 clips, the 35 dataset words plus filler up to each size, so every clip stays decidable at every size. The runtime refuses 300 distinct words or more; the sweep stops at 246, the filler this model's table knows.

| entries | vosk accuracy | vosk first ms | vosk later ms | vosk RTF | utterpy accuracy | utterpy first ms | utterpy later ms | utterpy RTF |
|---|---|---|---|---|---|---|---|---|
| 35 | 94.0% | 0.23 | 0.190 | 0.0152 | 94.5% | 0.92 | 0.014 | 0.0139 |
| 60 | 92.5% | 0.29 | 0.224 | 0.0163 | 92.8% | 1.49 | 0.019 | 0.0137 |
| 92 | 92.0% | 0.40 | 0.286 | 0.0168 | 92.2% | 2.37 | 0.027 | 0.0139 |
| 150 | 91.0% | 0.44 | 0.387 | 0.0174 | 92.0% | 3.69 | 0.041 | 0.0139 |
| 200 | 90.5% | 0.56 | 0.458 | 0.0179 | 91.5% | 5.04 | 0.048 | 0.0140 |
| 246 | 89.0% | 0.61 | 0.518 | 0.0184 | 89.8% | 6.07 | 0.058 | 0.0140 |

## Twelve-class: ten commands plus the unknown-word symbol

| engine | commands correct | fillers and digits read as unknown | fillers and digits read as a command | noise seconds | noise seconds with a word |
|---|---|---|---|---|---|
| vosk | 3916 / 4074 (96.12%) | 5773 / 6931 (83.29%) | 671 (9.68%) | 398 | 0 |
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
wheel, and the finals agree on 99.16% of clips. Where they differ, the
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

Compute is no longer a gap. On continuous audio the runtime is at
0.0115 against the wheel's 0.0123, and ahead where it counts, 2.43 ms
against 2.84 at the 95th percentile, the blocks that run a network
chunk. It is behind on the blocks that do not: 0.090 ms against 0.049,
which is the one figure left to chase. Construction is the other way
around again, 0.026 ms against 0.288, because a compiled grammar is
kept (`[[rr:TD-4]]`); the first is dearer and grows with the grammar.

Endpointing agrees less closely than the words do: both engines close
781 of 800, at the same block on 97.8%, against 99.2% agreement on
finals. A host that agreed on every word would still feel that.

Block size is not a latency dial. The network consumes a fixed chunk,
so a partial is revised only at multiples of 80 ms and a word appears
at the same instant whether it is fed in 10 ms blocks or 80. At 100
ms, which does not divide that, it is late by up to 60 ms on all but
26 clips of 400. Choose a block that divides 80 ms; the size buys
compute, not latency. Both engines behave alike, so this is libvosk
reproduced rather than a limitation of this one.

Two figures a host should read before choosing a grammar. Accuracy
falls about five points from 35 entries to 246, so distractors are not
free. And one first partial in eight is later revised, on both
engines, so a host that acts on the first word it sees acts wrongly
about 12% of the time.

Noise costs both engines the same. Accuracy halves between clean and
0 dB, and the paired test finds no difference between the engines at
any level. An earlier run showed the runtime 2.3 points behind at
0 dB; that was TD-6's bug, not the front end.
