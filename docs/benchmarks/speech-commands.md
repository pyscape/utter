# Speech Commands v2, vosk-model-small-en-us-0.15, testing split, 11005 clips, 40 ms blocks

Grammar: the dataset's 35 words plus 26 letters, 26 NATO words and 5 colours, 92 entries.

Run 2026-09-14 00:03:39Z, utter 38bfcb3, vosk 0.3.45, utterpy 0.0.1, model vosk-model-small-en-us-0.15, 12th Gen Intel(R) Core(TM) i7-12700F, Python 3.14.4. Decoded by the binding `/home/jared/Repos/utterpy/python/utterpy/__init__.py` built from utter 38bfcb34508b80fc65794c4d73c01eb3337da36a, the checkout's HEAD.

## Engines

Each engine opened the model and decoded a clip before the run began; a model an engine cannot decode stops the run here rather than part-way through a pass.

| engine | model load, s | first recognizer, ms |
|---|---|---|
| vosk | 0.19 | 0.4 |
| utterpy | 0.14 | 23.4 |

## Full grammar: accuracy

| engine | correct | accuracy | never in a partial | first appearance after the clip's energy end p50 / p90 | first word shown later changed |
|---|---|---|---|---|---|
| vosk | 10069 / 11005 | 91.49% | 2451 | 40 / 250 ms over 8554 | 1137 / 9218 (12.3%) |
| utterpy | 10102 / 11005 | 91.79% | 2432 | 40 / 250 ms over 8573 | 1147 / 9230 (12.4%) |

The last column counts the first word shown against the final, which a host feels as two different things. A partial overtaken by another partial is a flicker on the screen; a partial that stands through every later partial and is replaced only by the final is not a flicker at all, and no amount of waiting on partials would have caught it. The two columns below do not sum to that one: a partial can be overtaken and then come back, which the last column does not count and the first of these does. The lag is the gap between the two partials, and it decides whether a flicker is visible or too brief to see.

| engine | first word shown | overtaken by a later partial | stood, then changed by the final | revision lag p50 / p90 | lag min / max |
|---|---|---|---|---|---|
| vosk | 9218 | 688 (7.5%) | 456 (4.9%) | 240 / 240 ms | 240 / 480 ms |
| utterpy | 9230 | 691 (7.5%) | 462 (5.0%) | 240 / 240 ms | 240 / 480 ms |

Compute, one recognizer per clip:

| engine | first construction, ms | every later one, ms | mean, ms | RTF, decode only | RTF with construction |
|---|---|---|---|---|---|
| vosk | 0.32 | 0.293 | 0.293 | 0.0170 | 0.0173 |
| utterpy | 2.29 | 0.028 | 0.029 | 0.0137 | 0.0137 |

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

When one engine reads a clip the other misses, the miss is almost always a different word, not silence. The same paired wins, split by what the losing engine returned:

| direction | other read a different word | other read silence |
|---|---|---|
| vosk only | 14 | 3 |
| utterpy only | 47 | 3 |

Silence is 3 clips one way and 3 the other, so neither engine falls silent where the other reads a word more than the reverse; the difference is word against word, the near ties `[[rr:TD-6]]` leaves in the acoustics. Over the whole corpus the two read almost the same number of clips as silence, 411 and 412.

## Determinism

11005 clips decoded in two processes started for the purpose, comparing every partial with its 5 readings, block by block, and the final. A process is what matters: a hash seed drawn once per process gives a map the same iteration order for every repeat within a run, so a reading that depends on it is perfectly stable until the next run. The partial column compares a digest of the whole trace; any clip whose final moves is named in the JSON.

| engine | clips | partials and readings | finals |
|---|---|---|---|
| vosk | 11005 | 0 | 0 |
| utterpy | 11005 | 0 | 0 |

Every clip read the same way both times.

## Endpoint latency

800 clips, each followed by 2000 ms of silence, measured to the first block at which the engine closes a segment itself. Reported against the clip's own energy end, the same reference the first-appearance figure uses. Neither engine endpoints within the clip alone, so without the silence there is nothing here to measure.

The rows spelled `utterpy@MS` and `utterpy@MS/NATS` carry a host endpoint bound of their own beside the model's rules, `[[rr:TD-8#A host may add one endpoint bound of its own, off by default]]`: a final once the trailing silence reaches MS, and where a margin follows, no final while a reading extending the partial is within that many nats of the leader.

A bound shorter than one decoding advance does not thereby fire at the first advance that sees silence. The rule is read once per advance, 240 ms at this block size, but the trailing silence it compares against is counted in the model's own subsampled frames of 30 ms (`[[rr:endpoint_detected]]`), and at the first advance holding any trailing silence that span already stands anywhere from 30 ms to a full advance: 100 ms fires at that advance on the clips where it does and one advance later on the rest, so it neither coincides with every shorter bound nor with 300.

| engine | endpointed | after the clip's energy end p50 / p90 / p99 | never within the silence |
|---|---|---|---|
| vosk | 781 / 800 | 870 / 1010 / 1150 ms over 781 | 19 |
| utterpy | 781 / 800 | 870 / 1010 / 1150 ms over 781 | 19 |
| utterpy@100 | 781 / 800 | 470 / 620 / 790 ms over 781 | 19 |
| utterpy@300 | 781 / 800 | 660 / 800 / 930 ms over 781 | 19 |
| utterpy@300/4 | 781 / 800 | 660 / 810 / 940 ms over 781 | 19 |
| utterpy@300/8 | 781 / 800 | 670 / 820 / 960 ms over 781 | 19 |

Both engines endpointed on 781 of the 800, at the same block on 765 (97.95%). Endpointing is a second surface the contract covers, and one the accuracy figures do not touch: a host that agreed on every word would still feel a difference here.

### What the bound costs in words

The same 800 padded clips, every word each engine finally reported, paired against stock `utterpy` clip by clip. *Right* is the clip's label and nothing else; the three columns after it are how a reading that differs from the stock engine's differs. The test is exact McNemar over the clips where exactly one of the two was right.

| engine | right | differs from utterpy | said nothing | said more words | said another word | utterpy right only / bound right only | exact p |
|---|---|---|---|---|---|---|---|
| utterpy@100 | 736 / 800 (92.00%) | 0 | 0 | 0 | 0 | 0 / 0 | 1 |
| utterpy@300 | 736 / 800 (92.00%) | 0 | 0 | 0 | 0 | 0 / 0 | 1 |
| utterpy@300/4 | 736 / 800 (92.00%) | 0 | 0 | 0 | 0 | 0 / 0 | 1 |
| utterpy@300/8 | 736 / 800 (92.00%) | 0 | 0 | 0 | 0 | 0 / 0 | 1 |

Stock `utterpy` was right on 736 of the 800 under the same grammar and padding, which is the row every bound is paired against.

Nothing here separates one bound from another or from the stock rules, the bounds below 300 ms included: a clip carries one word and every bound fires in the silence after it, so no word is lost, none is decoded twice, and the paired test has nothing to weigh. Where a short bound can fall inside an utterance is the built streams, and the states page measures it there.

## Block size

400 clips at each block size, the same clips throughout. A word can only appear at a block boundary, but the decoder only revises a partial when the network has consumed a whole chunk, so the block is the coarser of the two grids only when it fails to land on the chunk. A size that does not divide the chunk's spacing is included deliberately: without one, the sweep cannot tell a decoder that ignores block size from a sweep that happened to pick sizes it agrees with. The accuracy column is the one that should not move.

| block ms | engine | accuracy | first appearance p50 / p90 | same instant as the finest block | RTF, decode only |
|---|---|---|---|---|---|
| 10 | vosk | 368 / 400 (92.0%) | 30 / 240 ms | 317 / 317 | 0.0174 |
| 10 | utterpy | 369 / 400 (92.2%) | 30 / 240 ms | 319 / 319 | 0.0141 |
| 20 | vosk | 368 / 400 (92.0%) | 30 / 240 ms | 317 / 317 | 0.0171 |
| 20 | utterpy | 369 / 400 (92.2%) | 30 / 240 ms | 319 / 319 | 0.0139 |
| 40 | vosk | 368 / 400 (92.0%) | 30 / 240 ms | 317 / 317 | 0.0173 |
| 40 | utterpy | 369 / 400 (92.2%) | 30 / 240 ms | 319 / 319 | 0.0137 |
| 80 | vosk | 368 / 400 (92.0%) | 30 / 240 ms | 317 / 317 | 0.0171 |
| 80 | utterpy | 369 / 400 (92.2%) | 30 / 240 ms | 319 / 319 | 0.0137 |
| 100 | vosk | 368 / 400 (92.0%) | 80 / 280 ms | 26 / 316, up to 60 ms later | 0.0170 |
| 100 | utterpy | 368 / 400 (92.0%) | 80 / 280 ms | 26 / 319, up to 60 ms later | 0.0143 |

The column that carries the result is the per-clip one, not the quantiles: two block sizes can produce the same p50 by luck, and only a clip-by-clip comparison against the finest block (10 ms) shows whether the word actually appeared at the same instant.

## Compute on continuous audio

One recognizer over 295 s of the clips joined end to end.

| engine | RTF | per-block compute ms p50 / p95 / p99 |
|---|---|---|
| vosk | 0.0124 | 0.049 / 2.83 / 3.38 |
| utterpy | 0.0113 | 0.042 / 2.59 / 2.84 |

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
| 35 | 94.0% | 0.23 | 0.194 | 0.0154 | 94.5% | 0.96 | 0.016 | 0.0136 |
| 60 | 92.5% | 0.30 | 0.232 | 0.0166 | 92.8% | 1.48 | 0.021 | 0.0140 |
| 92 | 92.0% | 0.38 | 0.292 | 0.0170 | 92.2% | 2.51 | 0.028 | 0.0138 |
| 150 | 91.0% | 0.46 | 0.393 | 0.0177 | 92.0% | 3.96 | 0.042 | 0.0138 |
| 200 | 90.2% | 0.55 | 0.472 | 0.0182 | 91.5% | 5.11 | 0.049 | 0.0145 |
| 246 | 89.0% | 0.63 | 0.534 | 0.0186 | 89.8% | 6.19 | 0.059 | 0.0145 |

## Twelve-class: ten commands plus the unknown-word symbol

| engine | commands correct | fillers and digits read as unknown | fillers and digits read as a command | noise minutes | silence finals per minute |
|---|---|---|---|---|---|
| vosk | 3917 / 4074 (96.15%) | 5769 / 6931 (83.23%) | 679 (9.80%) | 6.7 | 0.0 |
| utterpy | 3920 / 4074 (96.22%) | 5815 / 6931 (83.90%) | 633 (9.13%) | 6.7 | 0.0 |

## Background noise under the full grammar

| engine | grammar | noise minutes | partial blocks | blocks with a word at rank 0 | silence finals per minute | word blocks with a wordless reading among the rivals |
|---|---|---|---|---|---|---|
| vosk | full | 6.7 | 9932 | 0 (0.00%) | 1.1 | no alternatives |
| vosk | full + [unk] | 6.7 | 9931 | 0 (0.00%) | 0.9 | no alternatives |
| utterpy | full | 6.7 | 9932 | 0 (0.00%) | 0.9 | 0 / 0 |
| utterpy | full + [unk] | 6.7 | 9932 | 0 (0.00%) | 0.9 | 0 / 0 |
| utterpy@100 | full | 6.7 | 9932 | 0 (0.00%) | 0.9 | 0 / 0 |
| utterpy@100 | full + [unk] | 6.7 | 9932 | 0 (0.00%) | 0.9 | 0 / 0 |
| utterpy@300 | full | 6.7 | 9932 | 0 (0.00%) | 0.9 | 0 / 0 |
| utterpy@300 | full + [unk] | 6.7 | 9932 | 0 (0.00%) | 0.9 | 0 / 0 |
| utterpy@300/4 | full | 6.7 | 9932 | 0 (0.00%) | 0.9 | 0 / 0 |
| utterpy@300/4 | full + [unk] | 6.7 | 9932 | 0 (0.00%) | 0.9 | 0 / 0 |
| utterpy@300/8 | full | 6.7 | 9932 | 0 (0.00%) | 0.9 | 0 / 0 |
| utterpy@300/8 | full + [unk] | 6.7 | 9932 | 0 (0.00%) | 0.9 | 0 / 0 |

## Words on wordless audio

Every figure here is counted on audio after the last thing anybody said: the background recordings alone, and 105 clips followed by 3 to 5 s of one recording, from the moment each clip's own energy ends. A word at rank 0 is counted when the partial holds more words than were spoken so far in the segment, so a word still held from the clip is not one, nor is a misrecognition of it, and a word beyond it is. The after-a-word takes are the quietest, median and loudest clip of every dataset word, chosen for the level of the word that precedes the silence. Rates are per minute of non-speech, and both engines see the same samples.

### By floor level

The recordings scaled so their floor sits at each level, and digital silence of the same length beside them, which the floor's percentile treats apart from a room.

| floor | condition | engine | non-speech minutes | blocks | word at rank 0 /min | word among the rivals /min | finals with a word /min | longest run of blocks |
|---|---|---|---|---|---|---|---|---|
| digital silence | cold | vosk | 6.7 | 9911 | 0.00 | no alternatives | 0.00 | 0 |
| digital silence | after a word | vosk | 7.2 | 10787 | 0.56 | no alternatives | 0.00 | 4 |
| -70 dBFS | cold | vosk | 6.7 | 9953 | 0.00 | no alternatives | 2.25 | 0 |
| -70 dBFS | after a word | vosk | 7.2 | 10786 | 0.56 | no alternatives | 0.00 | 4 |
| -60 dBFS | cold | vosk | 6.7 | 9955 | 0.00 | no alternatives | 2.25 | 0 |
| -60 dBFS | after a word | vosk | 7.2 | 10788 | 0.56 | no alternatives | 0.00 | 4 |
| -50 dBFS | cold | vosk | 6.7 | 9954 | 0.00 | no alternatives | 2.25 | 0 |
| -50 dBFS | after a word | vosk | 7.2 | 10788 | 0.56 | no alternatives | 0.00 | 4 |
| -40 dBFS | cold | vosk | 6.7 | 9950 | 0.00 | no alternatives | 1.95 | 0 |
| -40 dBFS | after a word | vosk | 7.2 | 10788 | 0.56 | no alternatives | 0.00 | 4 |
| digital silence | cold | utterpy | 6.7 | 9911 | 0.00 | 0.0 | 0.00 | 0 |
| digital silence | after a word | utterpy | 7.2 | 10787 | 0.56 | 0.6 | 0.00 | 4 |
| -70 dBFS | cold | utterpy | 6.7 | 9954 | 0.00 | 0.0 | 2.25 | 0 |
| -70 dBFS | after a word | utterpy | 7.2 | 10786 | 0.56 | 0.6 | 0.00 | 4 |
| -60 dBFS | cold | utterpy | 6.7 | 9955 | 0.00 | 0.0 | 2.40 | 0 |
| -60 dBFS | after a word | utterpy | 7.2 | 10788 | 0.56 | 0.6 | 0.00 | 4 |
| -50 dBFS | cold | utterpy | 6.7 | 9954 | 0.00 | 0.0 | 2.25 | 0 |
| -50 dBFS | after a word | utterpy | 7.2 | 10788 | 0.56 | 0.6 | 0.00 | 4 |
| -40 dBFS | cold | utterpy | 6.7 | 9950 | 0.90 | 0.9 | 1.95 | 6 |
| -40 dBFS | after a word | utterpy | 7.2 | 10788 | 0.56 | 0.6 | 0.00 | 4 |

Paired over the same samples at -50 dBFS, cold, a sample carrying any rank-0 word: vosk only 0, utterpy only 0, exact two-sided p = 1.
Paired over the same samples at -50 dBFS, after a word, a sample carrying any rank-0 word: vosk only 0, utterpy only 0, exact two-sided p = 1.

### By grammar size

At a -50 dBFS floor, the 35 dataset words plus filler up to each size.

| entries | condition | engine | non-speech minutes | blocks | word at rank 0 /min | word among the rivals /min | finals with a word /min | longest run of blocks |
|---|---|---|---|---|---|---|---|---|
| 35 | cold | vosk | 6.7 | 9954 | 0.00 | no alternatives | 2.25 | 0 |
| 35 | after a word | vosk | 7.2 | 10786 | 0.00 | no alternatives | 0.14 | 0 |
| 35 | cold | utterpy | 6.7 | 9954 | 0.00 | 0.0 | 2.25 | 0 |
| 35 | after a word | utterpy | 7.2 | 10787 | 0.00 | 0.0 | 0.14 | 0 |
| 60 | cold | vosk | 6.7 | 9954 | 0.00 | no alternatives | 2.25 | 0 |
| 60 | after a word | vosk | 7.2 | 10787 | 0.56 | no alternatives | 0.00 | 4 |
| 60 | cold | utterpy | 6.7 | 9954 | 0.00 | 0.0 | 2.25 | 0 |
| 60 | after a word | utterpy | 7.2 | 10787 | 0.56 | 0.6 | 0.00 | 4 |
| 92 | cold | vosk | 6.7 | 9954 | 0.00 | no alternatives | 2.25 | 0 |
| 92 | after a word | vosk | 7.2 | 10788 | 0.56 | no alternatives | 0.00 | 4 |
| 92 | cold | utterpy | 6.7 | 9954 | 0.00 | 0.0 | 2.25 | 0 |
| 92 | after a word | utterpy | 7.2 | 10788 | 0.56 | 0.6 | 0.00 | 4 |
| 150 | cold | vosk | 6.7 | 9954 | 0.00 | no alternatives | 2.25 | 0 |
| 150 | after a word | vosk | 7.2 | 10788 | 3.06 | no alternatives | 0.14 | 22 |
| 150 | cold | utterpy | 6.7 | 9954 | 0.00 | 0.0 | 2.25 | 0 |
| 150 | after a word | utterpy | 7.2 | 10788 | 3.06 | 3.1 | 0.14 | 22 |
| 200 | cold | vosk | 6.7 | 9954 | 0.00 | no alternatives | 2.25 | 0 |
| 200 | after a word | vosk | 7.2 | 10788 | 3.06 | no alternatives | 0.14 | 22 |
| 200 | cold | utterpy | 6.7 | 9954 | 0.00 | 0.0 | 2.25 | 0 |
| 200 | after a word | utterpy | 7.2 | 10788 | 3.06 | 3.1 | 0.14 | 22 |
| 246 | cold | vosk | 6.7 | 9954 | 0.00 | no alternatives | 2.25 | 0 |
| 246 | after a word | vosk | 7.2 | 10788 | 4.72 | no alternatives | 0.28 | 22 |
| 246 | cold | utterpy | 6.7 | 9954 | 2.70 | 2.7 | 2.40 | 18 |
| 246 | after a word | utterpy | 7.2 | 10789 | 3.89 | 3.1 | 0.28 | 22 |

### The unknown-word symbol

At the same floor and the full grammar, `[unk]` admitted at each cost the private sweep uses, `[[rr:The unknown-word symbol against silence phantoms]]`. The stock wheel takes no cost, so it appears once, at its own default.

| `[unk]` cost | condition | engine | non-speech minutes | blocks | word at rank 0 /min | word among the rivals /min | finals with a word /min | longest run of blocks |
|---|---|---|---|---|---|---|---|---|
| stock | cold | vosk | 6.7 | 9954 | 0.00 | no alternatives | 2.25 | 0 |
| stock | after a word | vosk | 7.2 | 10788 | 0.56 | no alternatives | 0.00 | 4 |
| stock | cold | utterpy | 6.7 | 9954 | 0.00 | 0.0 | 2.25 | 0 |
| stock | after a word | utterpy | 7.2 | 10789 | 0.00 | 0.0 | 0.00 | 0 |
| 8 | cold | utterpy | 6.7 | 9954 | 0.00 | 0.0 | 2.25 | 0 |
| 8 | after a word | utterpy | 7.2 | 10788 | 0.56 | 0.6 | 0.00 | 4 |
| 4 | cold | utterpy | 6.7 | 9954 | 0.00 | 0.0 | 2.25 | 0 |
| 4 | after a word | utterpy | 7.2 | 10788 | 0.56 | 0.6 | 0.00 | 4 |
| 2 | cold | utterpy | 6.7 | 9954 | 0.00 | 0.0 | 2.25 | 0 |
| 2 | after a word | utterpy | 7.2 | 10789 | 0.56 | 0.6 | 0.00 | 4 |
| 0 | cold | utterpy | 6.7 | 9954 | 0.00 | 0.0 | 2.25 | 0 |
| 0 | after a word | utterpy | 7.2 | 10789 | 0.00 | 0.0 | 0.00 | 0 |
| -2 | cold | utterpy | 6.7 | 9954 | 0.00 | 0.0 | 2.25 | 0 |
| -2 | after a word | utterpy | 7.2 | 10790 | 0.00 | 0.0 | 0.00 | 0 |

### The floor gate a host applies

The same finals with the runtime's own reads applied host-side: a final word is silence when its `energy_dbfs` is within the margin of the `floor_dbfs` that final carries, and the runtime never applies the margin. Only the runtime reports the two fields, so the stock wheel has no row here. The McNemar pairs each sample before and after the gate.

| floor | condition | margin | finals with a word /min | after the gate /min | words caught | b / c / p |
|---|---|---|---|---|---|---|
| digital silence | cold | 4 dB | 0.00 | 0.00 | 0 / 0 | 0 / 0 / 1 |
| digital silence | cold | 8 dB | 0.00 | 0.00 | 0 / 0 | 0 / 0 / 1 |
| digital silence | cold | 12 dB | 0.00 | 0.00 | 0 / 0 | 0 / 0 / 1 |
| digital silence | after a word | 4 dB | 0.00 | 0.00 | 0 / 0 | 0 / 0 / 1 |
| digital silence | after a word | 8 dB | 0.00 | 0.00 | 0 / 0 | 0 / 0 / 1 |
| digital silence | after a word | 12 dB | 0.00 | 0.00 | 0 / 0 | 0 / 0 / 1 |
| -70 dBFS | cold | 4 dB | 2.25 | 0.30 | 13 / 15 | 5 / 0 / 0.0625 |
| -70 dBFS | cold | 8 dB | 2.25 | 0.15 | 14 / 15 | 5 / 0 / 0.0625 |
| -70 dBFS | cold | 12 dB | 2.25 | 0.15 | 14 / 15 | 5 / 0 / 0.0625 |
| -70 dBFS | after a word | 4 dB | 0.00 | 0.00 | 0 / 0 | 0 / 0 / 1 |
| -70 dBFS | after a word | 8 dB | 0.00 | 0.00 | 0 / 0 | 0 / 0 / 1 |
| -70 dBFS | after a word | 12 dB | 0.00 | 0.00 | 0 / 0 | 0 / 0 / 1 |
| -60 dBFS | cold | 4 dB | 2.40 | 0.15 | 15 / 16 | 5 / 0 / 0.0625 |
| -60 dBFS | cold | 8 dB | 2.40 | 0.15 | 15 / 16 | 5 / 0 / 0.0625 |
| -60 dBFS | cold | 12 dB | 2.40 | 0.15 | 15 / 16 | 5 / 0 / 0.0625 |
| -60 dBFS | after a word | 4 dB | 0.00 | 0.00 | 0 / 0 | 0 / 0 / 1 |
| -60 dBFS | after a word | 8 dB | 0.00 | 0.00 | 0 / 0 | 0 / 0 / 1 |
| -60 dBFS | after a word | 12 dB | 0.00 | 0.00 | 0 / 0 | 0 / 0 / 1 |
| -50 dBFS | cold | 4 dB | 2.25 | 0.15 | 14 / 15 | 5 / 0 / 0.0625 |
| -50 dBFS | cold | 8 dB | 2.25 | 0.15 | 14 / 15 | 5 / 0 / 0.0625 |
| -50 dBFS | cold | 12 dB | 2.25 | 0.15 | 14 / 15 | 5 / 0 / 0.0625 |
| -50 dBFS | after a word | 4 dB | 0.00 | 0.00 | 0 / 0 | 0 / 0 / 1 |
| -50 dBFS | after a word | 8 dB | 0.00 | 0.00 | 0 / 0 | 0 / 0 / 1 |
| -50 dBFS | after a word | 12 dB | 0.00 | 0.00 | 0 / 0 | 0 / 0 / 1 |
| -40 dBFS | cold | 4 dB | 1.95 | 0.15 | 12 / 13 | 4 / 0 / 0.125 |
| -40 dBFS | cold | 8 dB | 1.95 | 0.15 | 12 / 13 | 4 / 0 / 0.125 |
| -40 dBFS | cold | 12 dB | 1.95 | 0.15 | 12 / 13 | 4 / 0 / 0.125 |
| -40 dBFS | after a word | 4 dB | 0.00 | 0.00 | 0 / 0 | 0 / 0 / 1 |
| -40 dBFS | after a word | 8 dB | 0.00 | 0.00 | 0 / 0 | 0 / 0 / 1 |
| -40 dBFS | after a word | 12 dB | 0.00 | 0.00 | 0 / 0 | 0 / 0 / 1 |

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

Words on wordless audio put a number on the complaint the runtime
exists to answer, and the number is small and it is not the decoder's.
Fed a quiet room cold under the full grammar, neither engine puts a
word at rank 0 on a single block of ten thousand, at any floor from
digital silence to -50 dBFS, and the words that do appear appear in
finals: about 2.25 a minute at every floor a room has, and none at all
on digital zeros, which is where a silent
stretch's word comes from, `[[rr:TD-8#A silent stretch closes as
libvosk closes it]]`, and not a partial a host could have read first.

The dial that moves it is the grammar, not the microphone. From 35 to
92 entries the stretch after a spoken word carries at most 0.56 words a
minute; at 150 it is 3.06 and the longest run a word holds jumps from 4
blocks to 22, near a second of a host seeing a word nobody said. Both
engines move together, so this is libvosk's behaviour reproduced
rather than this runtime's defect, and it is the strongest argument on
the page for keeping a command grammar small.

The floor gate works and the unknown-word symbol does not. At 8 dB
above the reported floor the gate takes 14 of the 15 wordless final
words at -50 dBFS and 12 of 13 at -40, in every case leaving one
behind, and `[unk]` at any cost from 8 down to -2 changes the rank-0
count by nothing. That is the private sweep's finding
(`[[rr:The unknown-word symbol against silence phantoms]]`) reproduced
on public audio.

Two cells in the whole census differ between the engines, and both are
one event rather than a rate: a six-block run at a -40 dBFS floor and
an eighteen-block run at 246 entries, each seen by the runtime and not
by the stock wheel, each inside a single recording. They are recorded
here because a one-way excess is what a decoder dropping a reading
looks like; at two events in some twenty thousand paired blocks there
is nothing to test, and the paired tests over samples find no
difference at all.
