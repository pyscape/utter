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
silence. `[unk]` stays an option, off by default.

## libvosk reads the same 851

Run afterwards on the 2026-08-30 corpus, both engines fed the same 40 ms
blocks at dither 0 under the same grammar, quiet as defined above:
libvosk carries a vocabulary word at rank 0 on **851** quiet blocks, the
same count, and on the 30,941 quiet blocks where both emitted a partial
there is **not one** where only one of them shows a word. libvosk ends
with slightly more quiet final words than the runtime, 55 against 52.

So a word on a quiet block is not a defect in this runtime, and emitting
fewer of them would be a divergence from the oracle rather than a fix.
Silence is a host's gate to apply, not the decoder's to pre-empt.

## The 851 counts the wrong thing

Of the 851, **596 (70%) are carry-over**: no rank-0 word's own span
reaches the quiet tail at all, and the partial is still holding words
spoken earlier in the segment because the decoder has not endpointed
yet. Of the remaining 255, the words straddle the boundary, with a
median energy of -27 dBFS under a median span of 450 ms: the block is
quiet, the word is not.

Counted as a word whose own span carries no speech, the corpus has
**61 such finals, 2.13 a minute**, and 44 of those are one artefact:
a single word whose alignment has absorbed a silent region, median span
19.7 s against 360 ms for a real word. libvosk produces it too.

A gate on the energy under the word works, but only relative to the
take's own noise floor, not at an absolute level that will not carry
between microphones: at the floor plus 4 dB it catches 69% of the 61
and suppresses 1.5% of real words; at plus 12 dB, 95% for 4.4%. A
duration guard is independent of level and nearly as good on its own,
since real words reach 750 ms at the 95th percentile and the artefacts
start around five seconds.

The `[sil]` rival does not help here. It appears on **none** of 210
no-speech blocks, and on 5.5% of speech blocks: over sustained silence
the silence path is already pruned from the beam, so no rival is
offered to rank. The ruling that put it there,
`[[rr:TD-2#Silence and unknown speech announce themselves]]`, holds for
a pause inside speech and not for a quiet room.
