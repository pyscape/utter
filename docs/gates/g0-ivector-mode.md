# G0, 2026-09-11: zero against faithful i-vector mode

Gate: `[[rr:TD-2#Verification and acceptance]]`, G0.

## What ran

- Model: `vosk-model-small-en-us-0.15`, stock configuration (beam 10,
  max-active 3000, frames per chunk 24 by Kaldi's default).
- Grammar: the first consumer's grammar, a few dozen single-word entries, composed offline
  through `Model::compile_grammar` (12,281 states, 32,537 arcs, 8 ms).
- Corpus: a consumer replay set of 33 takes, 16 kHz mono,
  about 29 minutes of audio.
- Oracle: `vosk` 0.3.45 wheel through `scripts/g0.py oracle`, fed 40 ms
  blocks, dither 0 through a `conf/mfcc.conf` sibling; every final's
  words per take concatenated.
- Runtime: `g0` binary, dither 0, batch decode over each whole take with
  the network run in 24-frame chunks, the i-vector per chunk chosen as
  libvosk's streaming schedule would at 40 ms steps.

## Figures

| Mode | WER against libvosk finals | Takes exact |
|---|---|---|
| faithful | 15.50% (73 / 471 reference words) | 8 of 33 |
| zero | 17.83% (84 / 471 reference words) | 6 of 33 |

Zero minus faithful: +2.34 WER points.

## Ruling

Zero mode is not within one point of faithful mode, so the faithful
online i-vector is required and is not deferred.

The absolute figures are not a decoder-parity measurement: libvosk
resets its decoder at every endpoint and emits a final per segment,
while G0 decodes each take as one utterance. Most of the remaining
difference is isolated one-word finals libvosk emits for short noises
between commands. Parity is G2's measurement.
