# The unknown-word symbol against silence phantoms

Ruling: `[[rr:TD-2#Silence and unknown speech announce themselves]]`, the
`[unk]` option. Question: does admitting the model's unknown-word symbol
remove the vocabulary words a closed grammar reads into silence, and
what does it cost in real words?

## What ran

- Model: `vosk-model-small-en-us-0.15`; the first consumer's grammar.
- Replay corpus of 32 takes, 40 ms blocks, dither 0, partial words on,
  `stream --unknown-cost` at each cost, scored by
  `scripts/unk_sweep.py`: a block is quiet when its last 400 ms sit
  below -40 dBFS; a final word is quiet when its own interval does.
- Two scripted recordings, 679 script words, run by the consumer's
  benchmark through `utterpy` with the same costs, finals against the
  script.

## Replay corpus

| run | quiet blocks | rank 0 a vocabulary word on a quiet block | final words | quiet final words | `[unk]` finals | segments equal to libvosk |
|---|---|---|---|---|---|---|
| no `[unk]` | 30,943 | 851 | 467 | 52 | 0 | 191 / 205 |
| cost 8 | 30,943 | 851 | 467 | 52 | 0 | 191 / 205 |
| cost 4 | 30,944 | 841 | 467 | 52 | 0 | 189 / 205 |
| cost 2 | 30,945 | 831 | 464 | 52 | 2 | 189 / 205 |
| cost 0 | 30,949 | 786 | 454 | 52 | 6 | 184 / 205 |
| cost -2 | 30,951 | 770 | 445 | 54 | 12 | 179 / 205 |

## Scripted recordings

| run | recording 1: WER, insertions, deletions, substitutions, words | recording 2 |
|---|---|---|
| no `[unk]` | 35.6%, 6, 228, 8, 457 | 35.6%, 0, 225, 17, 454 |
| cost 2 | 35.6%, 6, 228, 8, 457 | 35.6%, 0, 226, 16, 453 |
| cost 0 | 35.6%, 6, 228, 8, 457 | 35.6%, 0, 226, 16, 453 |
| cost -2 | 37.7%, 6, 242, 8, 443 | 35.9%, 0, 228, 16, 451 |

The deletions are the recordings against the script and are the same
on every Kaldi engine; only the changes between rows matter here.

## Reading

Admitting `[unk]` takes at most a tenth of the quiet-block phantoms and
none of the quiet final words before it starts deleting real words: one
script word at costs 0 and 2, fourteen more at a bonus of 2, with the
corpus finals dropping from 467 to 445 words. On this model the
unknown-word path does not compete with short vocabulary words read into
silence. The silence fix a host should use is the energy under the word
and the `[sil]` entries; `[unk]` stays an option, off by default.
