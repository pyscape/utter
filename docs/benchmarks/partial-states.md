# Silence direction and word transitions on a built stream

Run 2026-09-13 17:12:30Z, utter 11fbe89, utterpy 0.0.1, vosk 0.3.45, model vosk-model-small-en-us-0.15, 12th Gen Intel(R) Core(TM) i7-12700F, Python 3.14.4. Decoded by the binding /home/jared/Repos/utterpy/.venv/lib/python3.14/site-packages/utterpy/__init__.py built from utter 11fbe8924c91fae89e2ccf7984bc2d4a93a234a4, the checkout's HEAD.

Grammar: the dataset's 35 words plus 26 letters, 26 NATO words and 5 colours, 92 entries. 40 ms blocks, 8 partial alternatives, partial words on, seed 20260912. Measured by `scripts/partial_states.py`.

Every reading's `relation` and `lead_delta` below are the runtime's own, read off the partial. The harness derives them again from its series as a check: over the streams decoded here 220372 readings carried a delta from both and all 220372 agreed, the largest disagreement 3.0e-06 nats against a tolerance of 2e-05; 79875 more carried one from the runtime alone, a reading entering the reported list whose history the harness never saw (`[[rr:TD-9#Readings are read once per decoding advance]]`).

The reading with no word on it is `[sil]` on silence phones and `[speech]` inside a word's phones (`[[rr:TD-10#Decision outcome]]`); every figure below keys it as one reading, and its history runs across the label. Over every partial read here it stood at rank 0 as `[speech]` on 88030 of 305669 blocks, and a `[speech]` entry closed the word list on 89698.

A Speech Commands clip holds one word, so it holds no transition between words and no finish. The streams here are built from the clips: utterances of one to five words, pauses of 100-800 ms between the words of an utterance, finishes of 2-3 s between utterances, and every gap cut in rotation from the dataset's own six background recordings, scaled to about -50 dBFS RMS rather than written as digital zeros, which the floor tracker treats apart (`[[rr:TD-8#The runtime reports the floor]]`). Every word's position and every gap's length are therefore exact, and each word's onset and offset are the first and last 10 ms frame within 20 dB of its clip's peak, which owes nothing to the decoder.

| split | streams | minutes | utterances | words | pauses | finishes | blocks | advances | finals |
|---|---|---|---|---|---|---|---|---|---|
| validation (thresholds fitted) | 72 | 77.2 | 720 | 2204 | 1484 | 720 | 115858 | 17224 | 2069 |
| testing (every figure below) | 75 | 78.6 | 750 | 2208 | 1458 | 750 | 117893 | 17562 | 2071 |

The readings change only when the decoder advances a chunk: over the testing streams 15491 intervals between advances, median / p90 240 / 240 ms, mean 240.0 ms, which is the 240 ms `[[rr:TD-9#Readings are read once per decoding advance]]` states at this block size. Between advances every partial repeats, so a rule reading the partial gives the same call; the calls below are therefore held flat between advances and scored at every block.

## The rules

R0 is TD-8's reading of the fields, and nothing else: a word arriving at rank 0 is speech beginning (a detection, no lead), the trailing `[sil]` entry's span past a bound is speech ending, a span that goes on growing is silence being kept, and a word at rank 0 with no trailing span is speech continuing. R1 is R0 plus the readings' motion: at an utterance's start the empty reading `[sil]` losing its lead means a word is coming, and mid-utterance a reading that extends rank 0 by a word and is gaining on the field means the next word is forming.

Fitted on the validation streams, at the 240 ms horizon, by the macro F1 over the four states: first the bound on the trailing span (500 ms chosen), then the two motion thresholds at that bound (`[sil]` lead_delta at or below -8.00 nats, an extending reading's lead_delta at or above +0.25 nats). TD-8's own bound, 300 ms, is reported beside them because the record quotes its private figures there.

| bound (ms) | macro F1 | F1 of *toward* | F1 of *away* | false *away* calls/min in finishes |
|---|---|---|---|---|
| 200 | 0.175 | 0.108 | 0.025 | 11.4 |
| 300 | 0.185 | 0.143 | 0.025 | 11.4 |
| 400 | 0.192 | 0.141 | 0.025 | 11.4 |
| 500 | 0.207 | 0.176 | 0.025 | 11.4 |
| 600 | 0.201 | 0.150 | 0.025 | 11.4 |
| 700 | 0.195 | 0.127 | 0.025 | 11.4 |
| 800 | 0.191 | 0.109 | 0.025 | 11.4 |
| 1000 | 0.185 | 0.084 | 0.025 | 11.4 |

The motion thresholds, over the same validation streams at the chosen bound. A pair that calls *away* everywhere buys recall on that state and loses the other three, which is why the choice is the macro F1 and not the *away* F1; the last column is what each pair costs inside the finish gaps, where the truth is never *away*.

| `[sil]` lead_delta at or below | extending lead_delta at or above | macro F1 | F1 of *away* | F1 of *toward* | false *away* calls/min in finishes |
|---|---|---|---|---|---|
| +0.00 | +0.00 | 0.152 | 0.150 | 0.094 | 72.7 |
| +0.00 | +0.25 | 0.154 | 0.141 | 0.094 | 72.1 |
| +0.00 | +0.50 | 0.155 | 0.134 | 0.094 | 71.4 |
| +0.00 | +1.00 | 0.154 | 0.127 | 0.094 | 70.5 |
| +0.00 | +2.00 | 0.153 | 0.122 | 0.094 | 67.6 |
| +0.00 | +4.00 | 0.151 | 0.112 | 0.094 | 63.1 |
| -0.25 | +0.00 | 0.190 | 0.131 | 0.098 | 46.6 |
| -0.25 | +0.25 | 0.191 | 0.118 | 0.098 | 46.0 |
| -0.25 | +0.50 | 0.191 | 0.108 | 0.098 | 45.3 |
| -0.25 | +1.00 | 0.189 | 0.097 | 0.098 | 44.4 |
| -0.25 | +2.00 | 0.187 | 0.088 | 0.098 | 41.5 |
| -0.25 | +4.00 | 0.182 | 0.069 | 0.098 | 37.0 |
| -0.50 | +0.00 | 0.198 | 0.127 | 0.096 | 32.0 |
| -0.50 | +0.25 | 0.199 | 0.113 | 0.096 | 31.3 |
| -0.50 | +0.50 | 0.198 | 0.101 | 0.096 | 30.7 |
| -0.50 | +1.00 | 0.197 | 0.089 | 0.096 | 29.7 |
| -0.50 | +2.00 | 0.194 | 0.077 | 0.096 | 26.9 |
| -0.50 | +4.00 | 0.188 | 0.055 | 0.096 | 22.4 |
| -1.00 | +0.00 | 0.204 | 0.126 | 0.106 | 25.0 |
| -1.00 | +0.25 | 0.204 | 0.110 | 0.106 | 24.4 |
| -1.00 | +0.50 | 0.203 | 0.098 | 0.106 | 23.8 |
| -1.00 | +1.00 | 0.201 | 0.085 | 0.106 | 22.8 |
| -1.00 | +2.00 | 0.198 | 0.072 | 0.106 | 19.9 |
| -1.00 | +4.00 | 0.192 | 0.047 | 0.106 | 15.5 |
| -2.00 | +0.00 | 0.208 | 0.128 | 0.121 | 24.6 |
| -2.00 | +0.25 | 0.209 | 0.112 | 0.121 | 24.0 |
| -2.00 | +0.50 | 0.208 | 0.100 | 0.121 | 23.3 |
| -2.00 | +1.00 | 0.206 | 0.086 | 0.121 | 22.4 |
| -2.00 | +2.00 | 0.202 | 0.073 | 0.121 | 19.5 |
| -2.00 | +4.00 | 0.196 | 0.047 | 0.121 | 15.0 |
| -4.00 | +0.00 | 0.218 | 0.133 | 0.148 | 24.5 |
| -4.00 | +0.25 | 0.218 | 0.117 | 0.148 | 23.9 |
| -4.00 | +0.50 | 0.217 | 0.103 | 0.148 | 23.2 |
| -4.00 | +1.00 | 0.215 | 0.089 | 0.148 | 22.3 |
| -4.00 | +2.00 | 0.211 | 0.075 | 0.148 | 19.4 |
| -4.00 | +4.00 | 0.205 | 0.048 | 0.148 | 14.9 |
| -8.00 | +0.00 | 0.223 | 0.135 | 0.162 | 24.5 |
| -8.00 | +0.25 | 0.223 | 0.119 | 0.162 | 23.9 |
| -8.00 | +0.50 | 0.222 | 0.105 | 0.162 | 23.2 |
| -8.00 | +1.00 | 0.220 | 0.091 | 0.162 | 22.3 |
| -8.00 | +2.00 | 0.216 | 0.077 | 0.162 | 19.4 |
| -8.00 | +4.00 | 0.209 | 0.048 | 0.162 | 14.9 |

## A. Silence direction

At every block the ground truth says what the next W ms do: *toward* silence (speech now, silence inside the window), *away* from silence (silence now, speech inside the window), *maintaining silence*, *maintaining speech*. The predictor reads only the partial at that block.

The *away* rows below are exploratory in the same way the onset section is: they are measured on spliced words, and a rule that calls *away* inside a built gap may be reading the gap. They are scored here against R0 and read as a comparison, never as evidence that speech can be foreseen.

Read the *away* row first. R0's *away* is a detection with no lead by construction: the word is already at rank 0 when it calls, and by then the truth has usually moved on to *maintaining speech*, so a low precision there is the shape of the problem rather than a fault in the rule. The question this page asks is whether the motion moves that row.

### W = 240 ms

| state | blocks (truth) | R0 precision | R0 recall | R1 precision | R1 recall |
|---|---|---|---|---|---|
| away | 10633 | 2.3% | 2.8% | 7.6% | 24.2% |
| toward | 11715 | 29.7% | 13.1% | 28.9% | 12.3% |
| maintaining silence | 73619 | 63.9% | 48.7% | 63.9% | 48.7% |
| maintaining speech | 9275 | 5.0% | 16.7% | 5.7% | 6.3% |

F1 of *away*: R0 0.025, R1 0.116; paired bootstrap over utterances (1000 resamples) of R1 - R0, +0.090 with a 95% interval [+0.083, +0.098].
F1 of *toward*: R0 0.182, R1 0.173; paired bootstrap over utterances (1000 resamples) of R1 - R0, -0.009 with a 95% interval [-0.013, -0.005].

### W = 500 ms

| state | blocks (truth) | R0 precision | R0 recall | R1 precision | R1 recall |
|---|---|---|---|---|---|
| away | 22636 | 10.2% | 5.9% | 21.0% | 31.6% |
| toward | 18223 | 40.1% | 11.4% | 39.0% | 10.7% |
| maintaining silence | 61616 | 53.8% | 48.9% | 53.8% | 48.9% |
| maintaining speech | 2767 | 2.4% | 26.4% | 2.3% | 8.4% |

F1 of *away*: R0 0.075, R1 0.253; paired bootstrap over utterances (1000 resamples) of R1 - R0, +0.178 with a 95% interval [+0.167, +0.189].
F1 of *toward*: R0 0.177, R1 0.168; paired bootstrap over utterances (1000 resamples) of R1 - R0, -0.010 with a 95% interval [-0.014, -0.006].

### Lead time per true transition

Every word's onset is a true move away from silence and every word's offset a move toward it. Per transition, the milliseconds from the first block the rule calls that state, inside the window from the previous transition to the next, to the transition itself; negative is before it. Handled means the call lands within 240 ms of the truth, which is the paired unit for McNemar.

| rule | transition | transitions | called | called before | p10 / p50 / p90 ms |
|---|---|---|---|---|---|
| R0 | away | 2208 | 1416 / 2208 (64.1%) | 1003 / 1416 (70.8%) | -976.7 / -642.2 / 431.3 |
| R0 | toward | 2208 | 565 / 2208 (25.6%) | 486 / 565 (86.0%) | -398.2 / -178.7 / 45.0 |
| R1 | away | 2208 | 1672 / 2208 (75.7%) | 1375 / 1672 (82.2%) | -985.9 / -778.1 / 364.8 |
| R1 | toward | 2208 | 535 / 2208 (24.2%) | 458 / 535 (85.6%) | -397.1 / -177.7 / 59.3 |

A call before the transition is not the same thing as foresight: a rule that calls *away* on a phantom word inside the gap has called it before the word, and the first call is what this table takes. The false alarm rates below are the other half of the figure, and the two should be read together.

McNemar on *away* handled within the horizon: both 1416, R0 only 0, R1 only 256, neither 536, exact two-sided p = 1.73e-77.
McNemar on *toward* handled within the horizon: both 535, R0 only 30, R1 only 0, neither 1643, exact two-sided p = 1.86e-09.

### False alarms

*away* where the truth cannot be *away*: inside a finish gap once the word has ended, and anywhere on the six background recordings fed continuously through one recognizer. Counted as blocks and as distinct runs of the call.

| audio | minutes | rule | away blocks/min | away calls/min |
|---|---|---|---|---|
| finish gaps | 31.4 | R0 | 40.7 | 10.46 |
| finish gaps | 31.4 | R1 | 230.7 | 24.07 |
| recordings alone | 6.7 | R0 | 0.0 | 0.00 |
| recordings alone | 6.7 | R1 | 0.0 | 0.00 |

The recordings alone also produced 61 finals over 6.7 minutes, 6 of them carrying a word, which is the figure the benchmark's noise pass reports; the rest are the 5 s silence rule closing a wordless stretch. Of the worded ones 6 carried words no partial had shown, and by the rule that closed them: flush 1, rule5 5.

### The end-of-speech confusion

TD-8's core problem in public form. Among the silent stretches, at each millisecond of trailing `[sil]` span: the share of inter-word pauses whose span reaches it before the next word begins (a pause taken for a finish) against the share of finishes whose span reaches it at all. The motion columns refuse the call at an advance where a reading extending rank 0 is gaining by at least +0.25 nats. The last column is how far ahead of the decoder's own endpoint final the call arrived.

| trailing span (ms) | pauses mistaken (1458) | with motion | finishes called (750) | with motion | called before the endpoint | median ms early |
|---|---|---|---|---|---|---|
| 100 | 1386 / 1458 (95.1%) | 919 / 1458 (63.0%) | 750 / 750 (100.0%) | 738 / 750 (98.4%) | 730 / 750 (97.3%) | 480.0 |
| 200 | 1330 / 1458 (91.2%) | 898 / 1458 (61.6%) | 750 / 750 (100.0%) | 737 / 750 (98.3%) | 730 / 750 (97.3%) | 240.0 |
| 300 | 1185 / 1458 (81.3%) | 660 / 1458 (45.3%) | 722 / 750 (96.3%) | 483 / 750 (64.4%) | 651 / 750 (86.8%) | 240.0 |
| 400 | 717 / 1458 (49.2%) | 430 / 1458 (29.5%) | 600 / 750 (80.0%) | 373 / 750 (49.7%) | 295 / 750 (39.3%) | 240.0 |
| 500 | 187 / 1458 (12.8%) | 70 / 1458 (4.8%) | 321 / 750 (42.8%) | 225 / 750 (30.0%) | 38 / 750 (5.1%) | 720.0 |
| 600 | 175 / 1458 (12.0%) | 70 / 1458 (4.8%) | 316 / 750 (42.1%) | 225 / 750 (30.0%) | 32 / 750 (4.3%) | 960.0 |
| 800 | 109 / 1458 (7.5%) | 27 / 1458 (1.9%) | 249 / 750 (33.2%) | 194 / 750 (25.9%) | 19 / 750 (2.5%) | 720.0 |
| 1000 | 91 / 1458 (6.2%) | 20 / 1458 (1.4%) | 213 / 750 (28.4%) | 168 / 750 (22.4%) | 15 / 750 (2.0%) | 720.0 |
| 1200 | 89 / 1458 (6.1%) | 20 / 1458 (1.4%) | 213 / 750 (28.4%) | 168 / 750 (22.4%) | 15 / 750 (2.0%) | 720.0 |
| 1500 | 73 / 1458 (5.0%) | 11 / 1458 (0.8%) | 132 / 750 (17.6%) | 104 / 750 (13.9%) | 14 / 750 (1.9%) | 720.0 |

That share of pauses is a fact about this stream's pauses, which are uniform over 100-800 ms by construction, so it is given again by the length of the pause that was built, at TD-8's 300 ms bound and at the fitted 500 ms one:

| pause built (ms) | pauses | mistaken at 300 ms | with motion | mistaken at 500 ms | with motion |
|---|---|---|---|---|---|
| 100-200 | 205 | 115 / 205 (56.1%) | 41 / 205 (20.0%) | 26 / 205 (12.7%) | 2 / 205 (1.0%) |
| 200-300 | 205 | 133 / 205 (64.9%) | 68 / 205 (33.2%) | 22 / 205 (10.7%) | 4 / 205 (2.0%) |
| 300-400 | 202 | 160 / 202 (79.2%) | 69 / 202 (34.2%) | 24 / 202 (11.9%) | 5 / 202 (2.5%) |
| 400-500 | 213 | 185 / 213 (86.9%) | 97 / 213 (45.5%) | 18 / 213 (8.5%) | 4 / 213 (1.9%) |
| 500-600 | 215 | 193 / 215 (89.8%) | 117 / 215 (54.4%) | 20 / 215 (9.3%) | 9 / 215 (4.2%) |
| 600-700 | 221 | 212 / 221 (95.9%) | 130 / 221 (58.8%) | 33 / 221 (14.9%) | 20 / 221 (9.0%) |
| 700-800 | 197 | 187 / 197 (94.9%) | 138 / 197 (70.1%) | 44 / 197 (22.3%) | 26 / 197 (13.2%) |

### The same stretches, from the finals the runtime emitted

The table above reads the trailing `[sil]` span a host would apply its own bound to. This one reads what the recognizer did: a final landing between a word's energy offset and the next word's onset is a pause the runtime ended, a final inside a finish gap is a finish it called. The bound rows are the same streams decoded again with the recognizer's own host bound set (`[[rr:TD-8#A host may add one endpoint bound of its own, off by default]]`), `MS` of trailing silence and, after the slash, the margin in nats within which a reading extending the partial vetoes the final.

| engine | pauses ended | finishes called | ms after the word's energy end p50 / p90 | finals | with a word | never shown | per minute | words lost | words decoded twice |
|---|---|---|---|---|---|---|---|---|---|
| utterpy | 942 / 1458 (64.6%) | 732 / 750 (97.6%) | 866.2 / 1000.9 | 2071 | 1996 | 3 | 26.4 | 52 / 2208 (2.4%) | 6 / 2208 (0.3%) |
| utterpy@100 | 1330 / 1458 (91.2%) | 727 / 750 (96.9%) | 481.7 / 623.6 | 2232 | 2156 | ? | 28.4 | 47 / 2208 (2.1%) | 6 / 2208 (0.3%) |
| utterpy@300 | 1208 / 1458 (82.9%) | 732 / 750 (97.6%) | 659.8 / 799.0 | 2200 | 2125 | ? | 28.0 | 49 / 2208 (2.2%) | 6 / 2208 (0.3%) |
| utterpy@300/4 | 1207 / 1458 (82.8%) | 732 / 750 (97.6%) | 659.8 / 799.0 | 2192 | 2117 | ? | 27.9 | 49 / 2208 (2.2%) | 6 / 2208 (0.3%) |
| utterpy@300/8 | 1159 / 1458 (79.5%) | 732 / 750 (97.6%) | 680.7 / 870.1 | 2139 | 2064 | ? | 27.2 | 48 / 2208 (2.2%) | 6 / 2208 (0.3%) |

*Never shown* counts the worded finals none of whose words had held in a partial: every word's `stable_ms` is zero, `[[rr:TD-11#Decision outcome]]`. A `?` is a row recorded before finals carried the field. By the rule that closed them, the worded finals of the first row: flush 1, rule2 1995.

The first row is the model's own endpointing (`[[rr:TD-2#Inputs: configuration]]`), which on this stream ends most pauses already. A bound is a gain only where it calls the finishes sooner than that without ending more pauses and without costing words, and the last two columns are where a bound firing inside a word shows up: a word no final covers, or one that two of them each decode.

Below 300 ms the price is not in the words. Against the 52 of 2208 the stock rules leave no final covering, the bounds leave 47 to 49, and the count two finals each decode is the same 6 at every row, the stock one included: what a bound shorter than 300 ms gives up is finishes called, not words.

A bound shorter than one decoding advance does not thereby fire at the first advance that sees silence. The rule is read once per advance, 240 ms at this block size, but the trailing silence it compares against is counted in the model's own subsampled frames of 30 ms (`[[rr:endpoint_detected]]`), and at the first advance holding any trailing silence that span already stands anywhere from 30 ms to a full advance: 100 ms fires at that advance on the stretches where it does and one advance later on the rest, so it neither coincides with every shorter bound nor with 300.

By the length of the pause that was built, as the series table above gives it:

| pause built (ms) | utterpy | utterpy@100 | utterpy@300 | utterpy@300/4 | utterpy@300/8 |
|---|---|---|---|---|---|
| 100-200 | 59 / 205 (28.8%) | 153 / 205 (74.6%) | 106 / 205 (51.7%) | 105 / 205 (51.2%) | 93 / 205 (45.4%) |
| 200-300 | 78 / 205 (38.0%) | 174 / 205 (84.9%) | 136 / 205 (66.3%) | 136 / 205 (66.3%) | 127 / 205 (62.0%) |
| 300-400 | 119 / 202 (58.9%) | 183 / 202 (90.6%) | 163 / 202 (80.7%) | 163 / 202 (80.7%) | 155 / 202 (76.7%) |
| 400-500 | 134 / 213 (62.9%) | 205 / 213 (96.2%) | 192 / 213 (90.1%) | 192 / 213 (90.1%) | 181 / 213 (85.0%) |
| 500-600 | 174 / 215 (80.9%) | 210 / 215 (97.7%) | 204 / 215 (94.9%) | 204 / 215 (94.9%) | 201 / 215 (93.5%) |
| 600-700 | 200 / 221 (90.5%) | 216 / 221 (97.7%) | 218 / 221 (98.6%) | 218 / 221 (98.6%) | 214 / 221 (96.8%) |
| 700-800 | 178 / 197 (90.4%) | 189 / 197 (95.9%) | 189 / 197 (95.9%) | 189 / 197 (95.9%) | 188 / 197 (95.4%) |

## B. Transitioning between words

Per ground-truth transition into a word, inside a window reaching 1000 ms back from its onset and 1500 ms forward: when the rank-0 word list grew, which is the transition a host sees today; when the word itself led; when a reading extending rank 0 first appeared and what its extra word was. A transition into the first word of an utterance is a different thing from one mid-utterance: when rank 0 is the empty reading every word-carrying reading extends it by definition (`[[rr:TD-9#Every reading names its relation to the partial]]`), so the two are reported apart.

### Preview lead

| position | transitions | rank 0 grew | the word that arrived was right | an extending reading first | zero lead | lead ms p50 |
|---|---|---|---|---|---|---|
| first in utterance | 750 | 741 / 750 (98.8%) | 629 / 741 (84.9%) | 741 / 741 (100.0%) | 0 / 741 (0.0%) | 1200.0 |
| mid utterance | 1458 | 1437 / 1458 (98.6%) | 730 / 1437 (50.8%) | 1437 / 1437 (100.0%) | 236 / 1437 (16.4%) | 960.0 |

The *an extending reading first* column is 100% by definition and not by measurement: wherever rank 0 is the empty reading, which is most of a gap, every word-carrying reading extends it. The column that carries information is the next one, and the right-word figure below it.

Over all 2208 transitions rank 0 grew on 2178 and the word itself led on 2061. An extending reading was there before the growth on 2178 / 2178 (100.0%), and that lead was zero advances on 236 / 2178 (10.8%); where it was not zero it ran 4.0 / 6.0 advances (median / p90), 1200.0 / 1440.0 ms. A reading extending rank 0 *by the word that was coming* appeared first on 1143 / 2178 (52.5%), by a median 240.0 ms and a p90 720.0 ms.

### Preview accuracy by the lead still to run

This is a measured result of its own, and separate from the question of whether a word can be foreseen: given that a reading extending rank 0 is present, how often its extra word is the word that arrives. The *all* group is candidate presence alone, the *gaining* group is presence plus the motion, and the two columns beside each other are the baseline and the field.

Every advance in those windows that carried an extending reading, 7497 of them, scored by whether the top extending reading's extra word is the word that was coming (top-1) and whether any of the top three extending readings has it (top-3), against the milliseconds still to run before rank 0 grew. The second group keeps only the advances where rank 0 already held a word, which is a continuation being previewed rather than a word being guessed at from silence; the third keeps the advances carrying a gaining one-word extender and ranks only those, which is the reading the rule uses.

| lead still to run (ms) | advances | top-1 | top-3 | over a word | top-1 | top-3 | gaining | top-1 | top-3 |
|---|---|---|---|---|---|---|---|---|---|
| 0-240 | 0 | - | - | 0 | - | - | 0 | - | - |
| 240-480 | 1634 | 256 / 1634 (15.7%) | 510 / 1634 (31.2%) | 55 | 8 / 55 (14.5%) | 25 / 55 (45.5%) | 1293 | 227 / 1293 (17.6%) | 430 / 1293 (33.3%) |
| 480-720 | 1398 | 19 / 1398 (1.4%) | 63 / 1398 (4.5%) | 106 | 3 / 106 (2.8%) | 6 / 106 (5.7%) | 604 | 4 / 604 (0.7%) | 15 / 604 (2.5%) |
| 720-1000 | 2526 | 40 / 2526 (1.6%) | 100 / 2526 (4.0%) | 696 | 7 / 696 (1.0%) | 34 / 696 (4.9%) | 967 | 13 / 967 (1.3%) | 34 / 967 (3.5%) |
| 1000-inf | 1939 | 29 / 1939 (1.5%) | 59 / 1939 (3.0%) | 795 | 5 / 795 (0.6%) | 23 / 795 (2.9%) | 857 | 6 / 857 (0.7%) | 18 / 857 (2.1%) |

### False transitions, and what the `[sil]` entries measure

Readings that extend rank 0 by one word where no word follows, and never reach rank 0 themselves, per minute. Split by whether rank 0 held a word: over a word this is the beam offering a continuation that never comes, over the empty reading it is a word being offered on silence, which is the rival `[[rr:TD-8#A word on a quiet block was spoken and is reported]]` describes.

| audio | minutes | per minute over a word | per minute over silence |
|---|---|---|---|
| finish gaps | 31.4 | 147.7 | 264.9 |
| recordings alone | 6.7 | 0.0 | 20.7 |

The inter-word `[sil]` entry against the pause that was built, read at the advance the next word first leads: 157 pauses, mean error -143.9 ms, median -138.1 ms, p90 absolute error 274.3 ms. It is readable only where the decoder did not close the utterance inside the pause, since a final starts the entries again.

The trailing `[sil]` span per advance inside a gap, against two clocks: the silence elapsed since the word's energy offset, and the silence since the decoder's own end of that word. The second is the span's own lag, the first adds the difference between the two alignments.

| gap | advances | mean error ms | median error ms | p90 absolute error ms | against the decoder's own end: mean | p10 / p50 / p90 |
|---|---|---|---|---|---|---|
| finish | 3802 | -521.6 | -744.9 | 979.2 | -161.1 | -160.0 / -160.0 / -160.0 |
| pause | 3568 | -277.8 | -264.2 | 908.9 | -160.9 | -160.0 / -160.0 / -160.0 |

### Transition latency on continuous audio

From a word's energy onset to the block the word itself leads the partial: median / p90 467.0 / 733.8 ms over the 2061 of 2208 transitions where it led inside the window, and to the block its `stable_ms` clears 200 ms, 667.0 / 933.8 ms over 2061. Mid-utterance alone the first figure is 465.2 ms. The decoder's own endpoint closed the utterance inside the pause on 942 / 1458 (64.6%) of the mid-utterance transitions: at this model's rules (`[[rr:TD-2#Inputs: configuration]]`) a pause of half a second is already an endpoint, so most of this page's mid-utterance transitions are the decoder starting again rather than continuing.

### Before and after, on the transitions

R0 is the host that sees a transition when rank 0 grows; R1 is the host that reads the extending reading's extra word where one appeared first, and R0 where none did. Paired on the transitions where rank 0 grew inside the window.

R1 comes in two forms: the first extending reading of any kind, and the first that is *gaining* by at least +0.25 nats, which is what reading the motion means. The gaining form was available on 2167 / 2178 (99.5%) of the transitions, with a lead of 720.0 / 1440.0 ms (median / p90).

| comparison | both right | R0 only | R1 only | neither | exact p |
|---|---|---|---|---|---|
| R0 against R1, any extender | 27 | 1332 | 11 | 808 | < 1e-308 |
| R0 against R1, gaining extender | 47 | 1312 | 14 | 805 | < 1e-308 |

Milliseconds from the onset to the call: R0 380.6 / 610.2 (median / p90), R1 any -806.1 / -290.3, R1 gaining -511.7 / 192.5. Paired bootstrap over transitions (1000 resamples) of the mean difference in call time: any extender -863.5 ms, 95% interval [-890.2, -836.9]; gaining -583.8 ms, [-612.9, -554.2]. Earlier is better only if the word called is right, which the table above answers.

## C. The readings' motion over many advances

A clip of one word holds three or four advances, so a reading's motion can be read once there and not followed. On a stream a reading lives through a pause and a finish, and these are the questions that needs.

### Does a lead's motion carry to the next advance

Pearson r and sign agreement between a reading's `lead_delta` at one advance and at the next, by the band its lead sits in (the absolute value, so a rival's deficit and a leader's gap fall in the same bands) and by what the reading is to rank 0. Each pair needs three advances, since a delta is itself a difference.

| readings | band (nats) | pairs | r | sign agreement |
|---|---|---|---|---|
| all | all | 73557 | 0.115 | 52.4% |
| all | <1 | 1454 | 0.223 | 62.0% |
| all | 1-4 | 14433 | 0.216 | 45.6% |
| all | >=4 | 57670 | 0.053 | 53.8% |
| [sil] | all | 7552 | 0.124 | 49.1% |
| [sil] | <1 | 211 | 0.525 | 81.5% |
| [sil] | 1-4 | 3653 | 0.449 | 48.2% |
| [sil] | >=4 | 3688 | -0.211 | 48.2% |
| extends | all | 56248 | 0.076 | 53.2% |
| extends | <1 | 350 | 0.082 | 62.6% |
| extends | 1-4 | 8300 | 0.088 | 43.1% |
| extends | >=4 | 47598 | 0.082 | 54.9% |
| differs | all | 4684 | -0.235 | 37.6% |
| differs | <1 | 397 | -0.108 | 35.3% |
| differs | 1-4 | 1469 | -0.231 | 30.7% |
| differs | >=4 | 2818 | -0.110 | 42.9% |

### What the readings do in silence that is being kept

Inside a finish gap, a second after the word has ended and half a second before anything begins, and on the recordings from one second in: the leading reading's lead over its best rival, that lead's motion, and how old the rival is. A lead that hovers while the rival stays young is a bound set by word hypotheses that keep being started and pruned, not a lead that is being won.

| audio | advances | `[sil]` at rank 0 | lead mean / sd | lead p10 / p50 / p90 | velocity mean / sd | velocity p10 / p50 / p90 | rival age ms p50 / p90 | rival one advance old |
|---|---|---|---|---|---|---|---|---|
| finish gaps, interior | 3909 | 3907 / 3909 (99.9%) | 4.22 / 0.79 | 3.69 / 4.05 / 5.12 | -0.067 / 0.309 | -0.42 / -0.05 / 0.20 | 480.0 / 1200.0 | 1367 / 3909 (35.0%) |
| recordings alone | 1584 | 1584 / 1584 (100.0%) | 5.16 / 1.36 | 3.61 / 5.25 / 6.79 | 0.051 / 0.317 | -0.19 / 0.01 / 0.39 | 2880.0 / 13440.0 | 128 / 1584 (8.1%) |

### The motion around a true transition

Every ground-truth onset and offset aligned at the advance that first covers it, with the median `lead_delta` at the advances around it: the empty reading's where rank 0 carries no word, the leading reading's, and the best one-word extending reading's where rank 0 carries a word. An advance is 240 ms.

| transition | advance | `[sil]` velocity (n) | leader velocity (n) | extending velocity (n) |
|---|---|---|---|---|
| away | -3 | -0.00 (805) | 0.01 (914) | 4.22 (9) |
| away | -2 | -0.01 (795) | 0.01 (991) | 3.92 (81) |
| away | -1 | 0.04 (957) | 0.05 (1179) | 1.78 (177) |
| away | +0 | -0.15 (1242) | -0.09 (1494) | 1.13 (209) |
| away | +1 | -2.04 (654) | 7.47 (2144) | 5.61 (142) |
| away | +2 | -0.00 (160) | 4.57 (2144) | 2.66 (1100) |
| toward | -3 | 0.01 (898) | 0.02 (1063) | 2.23 (103) |
| toward | -2 | 0.11 (1139) | 0.15 (1387) | 1.78 (136) |
| toward | -1 | -1.23 (984) | 0.32 (1777) | 3.70 (154) |
| toward | +0 | -2.65 (241) | 7.17 (2101) | 4.08 (525) |
| toward | +1 | -0.00 (49) | 1.48 (2167) | 1.80 (1665) |
| toward | +2 | -0.00 (54) | 0.00 (2005) | 0.29 (1816) |

### Calling a word before it arrives: exploratory

**Everything in this section is exploratory.** The streams are isolated read words spliced together with built gaps, and a rule that fires in a gap can be reading the stream's construction rather than the word that follows. Nothing here is a recommended read, and the record says so (`[[rr:TD-9#Every reading carries its lead's motion]]`).

The rule under test is the README's, and exactly it: a reading that is rank 0 plus one word and is gaining on the field, which before a first word is any word-carrying reading, since every one of them extends the empty reading. A run of consecutive advances where the rule holds is one call. Each call takes at most one word and each word at most one call: calls in time order claim the earliest unclaimed word whose onset falls between one advance before the call and 1000 ms after it. Every other call is a false alarm, counted against all the non-speech in the streams, the audio right after a word included, and every unclaimed word is a miss.

Four baselines are scored the same way, because a rule that calls more words by calling more often has shown nothing about its signal: the same candidate merely *existing*, the candidate's *lead* rather than its motion, the trailing `[sil]` span, and a clock the host keeps itself since it last saw rank 0 grow. Each threshold is fitted on the validation streams twice, once for its own best F1 and once to spend the motion rule's false alarms (32.05 a non-speech minute there), and read on the testing streams:

| rule | threshold | calls | words called | precision | F1 | false alarms/min | at the motion rule's alarm rate: threshold | words called | F1 |
|---|---|---|---|---|---|---|---|---|---|
| a one-word extending reading gaining by at least (nats) | 0.50 | 3836 | 1840 / 2208 (83.3%) | 48.0% | 0.609 | 31.90 | 0.50 | 1840 / 2208 (83.3%) | 0.609 |
| a one-word extending reading exists at all | 0.00 | 3142 | 1312 / 2208 (59.4%) | 41.8% | 0.490 | 29.24 | 0.00 | 1312 / 2208 (59.4%) | 0.490 |
| the best one-word extending reading's lead is at least (nats) | -8.00 | 2657 | 1245 / 2208 (56.4%) | 46.9% | 0.512 | 22.56 | -12.00 | 1336 / 2208 (60.5%) | 0.490 |
| the trailing `[sil]` span has reached (ms) | 300.00 | 3379 | 1529 / 2208 (69.2%) | 45.3% | 0.547 | 29.56 | 300.00 | 1529 / 2208 (69.2%) | 0.547 |
| this long since the host last saw rank 0 grow (ms) | 300.00 | 3168 | 1444 / 2208 (65.4%) | 45.6% | 0.537 | 27.55 | 200.00 | 1474 / 2208 (66.8%) | 0.499 |

Then the half of the figure that matters most, and the reason the earlier reading of it is withdrawn. A call before a word's onset is not evidence about that word unless some of the word has been fed. Splitting the motion rule's calls by what the decoder had heard when it fired:

| the call landed | motion | the candidate merely existing |
|---|---|---|
| before any sample of the coming word's clip | 1065 / 1840 (57.9%) | 699 / 1312 (53.3%) |
| inside the clip, before its energy onset | 307 / 1840 (16.7%) | 394 / 1312 (30.0%) |
| at or after the energy onset, a detection | 468 / 1840 (25.4%) | 219 / 1312 (16.7%) |

The lead of the calls that do come first is -910.5 / -594.8 / -142.8 ms (p10 / p50 / p90), but a lead measured over calls that mostly precede the word's clip entirely is not an acoustic warning of that word: it is a call made on the silence before it, which on this stream is a built gap of known length. That figure is reported here and **is not** to be read as the decoder hearing a word coming.

Because the pauses are built, the alarm rate and the calls are partly a fact about the distribution they were built from. The same rule at the same threshold on testing streams built with other pause ranges:

| pauses built (ms) | words | non-speech minutes | words called | precision | false alarms/min | called before the clip |
|---|---|---|---|---|---|---|
| 100-800 (the page's streams) | 2208 | 62.6 | 1840 / 2208 (83.3%) | 48.0% | 31.90 | 1065 / 1840 (57.9%) |
| 100-300 | 606 | 15.2 | 512 / 606 (84.5%) | 54.1% | 28.55 | 308 / 512 (60.2%) |
| 400-1600 | 606 | 20.6 | 497 / 606 (82.0%) | 39.3% | 37.28 | 212 / 497 (42.7%) |

What this section establishes is a comparison, not a forecast: whether the motion beats candidate presence, candidate level, elapsed silence and a clock, at the alarm rate it costs, on spliced words. Whether any of it survives on continuous recorded commands is not measured here and is the work the record leaves open.

### The two crossings, as fitted signals

The same two motions read as raw crossings rather than as the README's rule, kept for the `[sil]` half: a crossing is one advance where the empty reading's lead falls past a threshold, or a one-word extending reading's lead rises past one, scored by the older many-to-one rule (a crossing is a hit when any onset follows within the lookback). The `[sil]` row is why the README's read is *an extends reading gaining* and never *`[sil]` falling*.

| signal | threshold (nats) | crossings | precision | onsets called | lead p10 / p50 / p90 ms | alarms/min in kept silence | alarms/min on the recordings |
|---|---|---|---|---|---|---|---|
| `[sil]` lead falling past | +0.00 | 4909 | 40.2% | 1091 / 2208 (49.4%) | -964.4 / -778.2 / -303.4 | 131.9 | 109.8 |
| extending lead rising past | +0.00 | 3745 | 52.4% | 1189 / 2208 (53.8%) | -918.6 / -644.5 / -244.9 | 0.1 | 0.0 |

### Leading but losing

Rank 0's lead is never negative, so the reversal `[[rr:TD-9#Every reading carries its lead's motion]]` describes, a reading whose lead and whose motion disagree in sign, is rank 0 with a negative `lead_delta`. Scored against rank 0 changing at the next advance, and against rank 0's words already differing from the final that closes the segment, beside the rank-AUC of the level and of the motion on the same advances. 0.5 is no information and below 0.5 the signal is inverted, as on the trust page: a lead that is small, or a motion that is negative, goes with the change.

| band (nats) | target | advances | positives | precision | recall | AUC of the lead | AUC of `lead_delta` |
|---|---|---|---|---|---|---|---|
| all | rank 0 changes next | 13406 | 1901 | 20.5% | 62.3% | 0.272 | 0.297 |
| all | rank 0 unlike the final | 13406 | 8617 | 82.7% | 55.5% | 0.297 | 0.220 |
| <1 | rank 0 changes next | 857 | 360 | 61.0% | 53.9% | 0.445 | 0.308 |
| <1 | rank 0 unlike the final | 857 | 522 | 94.3% | 57.5% | 0.506 | 0.239 |
| 1-4 | rank 0 changes next | 5077 | 900 | 24.8% | 80.1% | 0.208 | 0.188 |
| 1-4 | rank 0 unlike the final | 5077 | 4018 | 92.9% | 67.2% | 0.719 | 0.154 |
| >=4 | rank 0 changes next | 7472 | 641 | 10.6% | 42.1% | 0.375 | 0.372 |
| >=4 | rank 0 unlike the final | 7472 | 4077 | 69.7% | 43.6% | 0.151 | 0.262 |

### The stock wheel on the same streams

The stock wheel has partial text and no readings to move. Over the same testing streams its partial grew a word inside the window on 2177 / 2208 (98.6%) transitions, and that word was the right one on 1357 / 2177 (62.3%), at 381.8 / 610.2 ms from the onset (median / p90). The word itself led on 2060 / 2208 (93.3%) at 467.9 / 733.8 ms. It returned 2068 finals where utter returned 2071. That is the detection baseline: the moment R0 reads, with nothing beside it.

Its partial word times could not be used at all. With partial words on, 24 of 1374 blocks of the first testing stream carried partial text, against 510 with partial words off: the wheel drops the word in progress from a partial when word times are requested. utter's partial carried text on 510 blocks with them on and 510 with them off. The baseline above is therefore the wheel with partial words off.

## Caveat

The words are isolated read commands joined with built gaps. They carry no coarticulation across the join and no sentence prosody, and every pause is a splice rather than a speaker drawing breath, so the transitions here are cleaner than dictation and the pauses are more uniform. TD-8's figures come from the first consumer's private gate corpus; this page is the reproducible counterpart, not a replacement. The gaps are the dataset's own recordings scaled to one floor, so the floor does not move within a stream as it does between rooms.

Every figure above is in `partial-states.json`, and `partial-states.streams.jsonl` carries, per stream, the ground truth, the finals and the whole advance series with each reading's confidence, lead delta and relation, so another rule can be scored on the same streams without decoding them again.

## Words in the stream's own silences

The census of `[[rr:Words on wordless audio]]` read off the built streams instead of the recordings: every gap between words and after the last one, and the recordings fed cold beside them. A word at rank 0 counts when the partial holds more words than the stream has finished saying since the last final, so a word still held is not one. Rates are per minute of the gaps themselves.

### By the length of the pause or the finish

The page's own testing streams, gaps built at -50 dBFS.

| gap | engine | non-speech minutes | blocks | word at rank 0 /min | word among the rivals /min | finals with a word /min | longest run of blocks |
|---|---|---|---|---|---|---|---|
| finish, under 2400 ms | vosk | 10.8 | 16176 | 1.95 | no alternatives | 0.19 | 21 |
| finish, under 2700 ms | vosk | 9.3 | 13938 | 0.00 | no alternatives | 0.00 | 0 |
| finish, 2700 ms or more | vosk | 11.3 | 16992 | 1.15 | no alternatives | 0.09 | 13 |
| pause, under 200 ms | vosk | 0.5 | 768 | 3.90 | no alternatives | 1.95 | 2 |
| pause, under 400 ms | vosk | 2.0 | 3035 | 9.85 | no alternatives | 0.49 | 20 |
| pause, under 600 ms | vosk | 3.6 | 5342 | 3.37 | no alternatives | 0.28 | 12 |
| pause, under 800 ms | vosk | 4.8 | 7272 | 0.62 | no alternatives | 0.00 | 3 |
| finish, under 2400 ms | utterpy | 10.8 | 16176 | 1.95 | 1.95 | 0.19 | 21 |
| finish, under 2700 ms | utterpy | 9.3 | 13938 | 0.00 | 0.00 | 0.00 | 0 |
| finish, 2700 ms or more | utterpy | 11.3 | 16992 | 1.15 | 1.15 | 0.18 | 13 |
| pause, under 200 ms | utterpy | 0.5 | 768 | 3.90 | 3.90 | 1.95 | 2 |
| pause, under 400 ms | utterpy | 2.0 | 3035 | 9.85 | 9.85 | 0.49 | 20 |
| pause, under 600 ms | utterpy | 3.6 | 5342 | 3.37 | 3.37 | 0.28 | 12 |
| pause, under 800 ms | utterpy | 4.8 | 7272 | 0.62 | 0.00 | 0.00 | 3 |

### The recordings fed cold, by floor level

The same recordings the gaps are cut from, fed whole through one recognizer, scaled so their floor sits at each level.

| floor | engine | non-speech minutes | blocks | word at rank 0 /min | word among the rivals /min | finals with a word /min | longest run of blocks |
|---|---|---|---|---|---|---|---|
| digital silence | vosk | 6.7 | 9987 | 0.00 | no alternatives | 0.00 | 0 |
| digital silence | utterpy | 6.7 | 9987 | 0.00 | 0.00 | 0.00 | 0 |
| -70 dBFS | vosk | 6.7 | 9987 | 0.00 | no alternatives | 2.25 | 0 |
| -70 dBFS | utterpy | 6.7 | 9987 | 0.00 | 0.00 | 2.25 | 0 |
| -60 dBFS | vosk | 6.7 | 9987 | 0.00 | no alternatives | 2.25 | 0 |
| -60 dBFS | utterpy | 6.7 | 9987 | 0.00 | 0.00 | 2.40 | 0 |
| -50 dBFS | vosk | 6.7 | 9987 | 0.00 | no alternatives | 2.25 | 0 |
| -50 dBFS | utterpy | 6.7 | 9987 | 0.00 | 0.00 | 2.25 | 0 |
| -40 dBFS | vosk | 6.7 | 9987 | 0.00 | no alternatives | 1.95 | 0 |
| -40 dBFS | utterpy | 6.7 | 9987 | 0.90 | 0.90 | 1.95 | 6 |

### The finish gaps, by the floor they were built at

Testing streams rebuilt with the gaps scaled to each floor, so the stretch after a spoken word is measured at the level a new microphone would put it.

| gap floor | engine | non-speech minutes | blocks | word at rank 0 /min | word among the rivals /min | finals with a word /min | longest run of blocks |
|---|---|---|---|---|---|---|---|
| -70 dBFS | vosk | 8.3 | 12475 | 0.00 | no alternatives | 0.00 | 0 |
| -70 dBFS | utterpy | 8.3 | 12475 | 0.00 | 0.00 | 0.12 | 0 |
| -60 dBFS | vosk | 8.3 | 12475 | 0.00 | no alternatives | 0.00 | 0 |
| -60 dBFS | utterpy | 8.3 | 12475 | 0.00 | 0.00 | 0.12 | 0 |
| -50 dBFS | vosk | 31.4 | 47106 | 1.08 | no alternatives | 0.10 | 21 |
| -50 dBFS | utterpy | 31.4 | 47106 | 1.08 | 1.08 | 0.13 | 21 |
| -40 dBFS | vosk | 8.3 | 12475 | 0.00 | no alternatives | 0.00 | 0 |
| -40 dBFS | utterpy | 8.3 | 12475 | 0.00 | 0.00 | 0.12 | 0 |

Paired over the 75 testing streams, a stream carrying any word at rank 0 in a gap: utterpy only 0, vosk only 0, exact two-sided p = 1.

Three things to read off these tables. The rate is not flat across a gap's length: it is highest in the short pauses, where the decoder is still inside the utterance and no endpoint has fired, and it falls as the gap lengthens and the rules close the stretch, which is the mechanism `[[rr:TD-8#A word on a quiet block was spoken and is reported]]` describes rather than a second one. The floor the gap is built at moves the finals and barely moves rank 0, so a host that gates on level is gating the right half. And the two engines are read on the same blocks here, so any cell where they differ is a one-way excess and is reported as such; the paired test over streams is the test of it.

The gap-floor table's own rows are not one sample: the page's own streams carry the whole testing split and each rebuilt set carries fewer words, so read the rates and not the block counts across it.

