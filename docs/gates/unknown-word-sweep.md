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

## What the 851 are

Of the 851, **596 (70%) are carry-over**: no rank-0 word's own span
reaches the quiet tail, and the partial still holds words spoken earlier
in the utterance because no endpoint has fired. The other 255 straddle
the block: the word has just ended, at a median -27 dBFS over a median
450 ms span. Against the take's floor, the 5th percentile of RMS over
100 ms windows at a 50 ms hop, the last word shown on a quiet block is at
least 8.5 dB above it and on nine blocks in ten 18 dB or more. The stream
shows no word that was not spoken.

Take floors on this corpus run from -51.7 to -37.2 dBFS, 14 dB apart on
one microphone, and the public dataset's six background recordings sit
between -62.5 and -9.8 dBFS by the same measure. The -40 dBFS above is
the wrong instrument; the figures below are relative to the floor.

## Where the silence finals come from

Of 209 segments, 22 close with no word after five seconds of silence
phones on the best path, rule 1, and 51 close at the 20 s cap, rule 5.
On 45 of those 51 the final is one word spanning the whole utterance,
and three grammar words account for all 45. libvosk emits the same word
on every one, and no partial in any of these segments shows it.

The mechanism: on a quiet room the best path sits in the opening phones
of a word without reaching its label, so no silence phone is on the
path. Of the 30,092 quiet blocks with no word at rank 0, 26,665 (89%)
carry an empty word list and 3,427 (11%) a trailing `[sil]` entry, whose
span reaches 4.8 s, the next block being rule 1. Trailing silence stays
at zero, rules 1 to 4 cannot fire, rule 5 closes the stretch, and the
final, which must end in a final state, completes the word.

## Gates a host can apply to a final

The 45 rule-5 finals sit a median 2.3 dB above the floor, at most 9.1;
the other 424 final words a median 23.4 dB above it, 11.2 at the 5th
percentile. Catch rate on the 45 against words suppressed among the
424:

| gate | catches | suppresses |
|---|---|---|
| mean energy below floor + 4 dB | 39 / 45 (87%) | 7 / 424 (1.7%) |
| mean energy below floor + 8 dB | 44 / 45 (98%) | 11 / 424 (2.6%) |
| mean energy below floor + 12 dB | 45 / 45 (100%) | 29 / 424 (6.8%) |
| mean energy below floor + 16 dB | 45 / 45 (100%) | 56 / 424 (13.2%) |
| span longer than 2 s | 45 / 45 (100%) | 16 / 424 (3.8%) |

The sixteen other words longer than two seconds are spoken words whose
last phone absorbed the silence after them, up to 18 s inside a
multi-word segment, and 20 s closures on a louder room. A span
corroborates; energy against the floor decides. A 100 ms peak is a worse
gate than the mean: a third of the 45 hold a breath or a click that puts
a peak 12 dB over the floor, and the last 200 ms is coarser than the
mean.

## Two `[sil]` mechanisms

`[[rr:TD-2#Silence and unknown speech announce themselves]]` puts `[sil]`
in two places. As a ranked rival: on a quiet room `[sil]` leads and there
is nothing to rival it; when a word leads a quiet block it was spoken
and the empty path is long gone, so a `[sil]` rival appears on 6 of the
851. As an entry in the word list: after a word, on 742 of the 851
(87%); on a quiet room with no word, on the 11% above. The entry is the
clock after a word. Nothing announces a quiet room but the absence of
words and the energy.

## Reading the end of speech

Over the 108 segments that end in a word, the trailing `[sil]` entry
appears a median 240 ms after the last word appeared (95th percentile
480), and the endpoint a median 760 ms after it (5th percentile 520,
95th 1000). Over the 289 intervals between two words of one segment,
the next word appears a median 480 ms after the previous (95th 720,
maximum 1200), and the decoder's own pause between them, the longest
trailing `[sil]` entry seen before the next word, is 270 ms at the 95th
percentile, 480 at the 99th, 630 at most.

| host rule | calls a finish inside a pause | calls it before the endpoint | lead, median |
|---|---|---|---|
| trailing `[sil]` span at least 200 ms | 25 / 289 (8.7%) | 105 / 108 (97%) | 280 ms |
| trailing `[sil]` span at least 300 ms | 13 / 289 (4.5%) | 94 / 108 (87%) | 280 ms |
| trailing `[sil]` span at least 400 ms | 6 / 289 (2.1%) | 45 / 108 (42%) | 280 ms |
| last word held 400 ms, no new word | 156 / 289 (54%) | 106 / 108 (98%) | 360 ms |
| last word held 500 ms, no new word | 43 / 289 (15%) | 106 / 108 (98%) | 260 ms |
| last word held 800 ms, no new word | 11 / 289 (3.8%) | 48 / 108 (44%) | 200 ms |

The hold on the last word is what a pause between words looks like too;
the trailing entry is measured on the decoded path, where a word still
being spoken shows as non-silence phones before its label appears.
