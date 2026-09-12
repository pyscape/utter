# Speech Commands v2

Script: `scripts/speech_commands.py`. Dataset: Google Speech Commands
v2, CC BY 4.0, the testing split of 11,005 one-second clips, one word
each, 16 kHz. Both engines fed the same 40 ms blocks under the same
grammar, the partial read after every block, the final at the end.
Model: `vosk-model-small-en-us-0.15`, stock configuration, default
dither. Engines: utter through `utterpy`, and the `vosk` 0.3.45 wheel.

## Full grammar: 92 entries

The dataset's 35 words, the letters a to z, the NATO alphabet with
"x ray" as two words, and red, yellow, blue, black and white. Every clip
must decode to exactly its word.

| engine | correct | accuracy | never in a partial before the final | first appearance after vosk's word end, p50 / p90 | RTF |
|---|---|---|---|---|---|
| vosk | 10,070 / 11,005 | 91.50% | 2,445 | -20 / 130 ms over 8,511 | 0.019 |
| utterpy | 10,075 / 11,005 | 91.55% | 2,432 | -20 / 130 ms over 8,516 | 0.024 |

Most frequent confusions, vosk: up to nothing (67), eight to a (50),
off to o (34), three to tree (27), eight to nothing (25), four to
forward (20), forward to four (17). utterpy: up to nothing (77), eight to
a (48), off to o (30), three to tree (29), eight to nothing (25), on to
nothing (23), four to forward (20). The lists are the same words in
nearly the same order: the distractor letters take the short words
(eight to a, off to o), and "up" is lost to silence on both engines.

"Never in a partial" counts clips whose word appears only in the final:
a one-second clip often ends inside the network's right context, so the
word is decoded on the flush, not on a block.

Per word, correct / clips, vosk then utterpy:

| word | vosk | utterpy |
|---|---|---|
| yes | 387 / 419 | 388 / 419 |
| no | 395 / 405 | 395 / 405 |
| up | 347 / 425 | 338 / 425 |
| down | 378 / 406 | 377 / 406 |
| left | 387 / 412 | 388 / 412 |
| right | 360 / 396 | 359 / 396 |
| on | 363 / 396 | 361 / 396 |
| off | 341 / 402 | 340 / 402 |
| stop | 401 / 411 | 401 / 411 |
| go | 377 / 402 | 381 / 402 |
| zero | 382 / 418 | 388 / 418 |
| one | 368 / 399 | 368 / 399 |
| two | 399 / 424 | 398 / 424 |
| three | 355 / 405 | 356 / 405 |
| four | 361 / 400 | 360 / 400 |
| five | 423 / 445 | 420 / 445 |
| six | 367 / 394 | 368 / 394 |
| seven | 388 / 406 | 389 / 406 |
| eight | 315 / 408 | 322 / 408 |
| nine | 392 / 408 | 394 / 408 |
| backward | 152 / 165 | 152 / 165 |
| forward | 133 / 155 | 136 / 155 |
| follow | 163 / 172 | 163 / 172 |
| learn | 137 / 161 | 137 / 161 |
| visual | 141 / 165 | 139 / 165 |
| bed | 190 / 207 | 191 / 207 |
| bird | 175 / 185 | 174 / 185 |
| cat | 177 / 194 | 176 / 194 |
| dog | 206 / 220 | 205 / 220 |
| happy | 192 / 203 | 192 / 203 |
| house | 181 / 191 | 181 / 191 |
| marvin | 180 / 195 | 180 / 195 |
| sheila | 198 / 212 | 198 / 212 |
| tree | 171 / 193 | 171 / 193 |
| wow | 188 / 206 | 189 / 206 |

## Twelve-class: ten commands plus the unknown-word symbol

Grammar: yes, no, up, down, left, right, on, off, stop, go, and `[unk]`.
The 25 other words should read as unknown; the background-noise
recordings, cut into one-second pieces, should read as nothing.

| engine | commands correct | others read as unknown | others read as a command | noise seconds | noise seconds with a word |
|---|---|---|---|---|---|
| vosk | 3,916 / 4,074 (96.12%) | 5,770 / 6,931 (83.25%) | 675 (9.74%) | 398 | 0 |
| utterpy | 3,915 / 4,074 (96.10%) | 5,790 / 6,931 (83.54%) | 620 (8.95%) | 398 | 0 |

## Reading

Parity with the stock wheel on a public set: accuracy within 0.05
points, the same confusions, the same first-appearance latency
distribution, no phantom words on noise for either engine. The
real-time factor is measured in-process through Python, one clip at a
time with recognizer construction included, so it is a ceiling.
