# G2 and G5: the stream against the stock wheel

Gates: `[[rr:TD-2#Verification and acceptance]]`, G2 and G5, with the
compute half of G3.

## What ran

- Model: `vosk-model-small-en-us-0.15`, stock configuration.
- Grammar: the first consumer's, a few dozen single-word entries.
- Corpus: a consumer replay set of 33 takes, about 29 minutes.
- Oracle: `vosk` 0.3.45 through `scripts/g2.py oracle`, 40 ms blocks,
  dither 0 through a `conf/mfcc.conf` sibling, the partial read after
  every block, `Result` on every endpoint, `FinalResult` at the end.
- Runtime: `stream`, the same blocks, dither 0, four takes in parallel
  on one machine, compute timed per block.
- Texts compared with bracketed tokens removed.

## Figures

| Quantity | Value | Gate |
|---|---|---|
| partial text equal per block | 42,578 / 42,697 (99.72%) | 95% |
| word first appears in the same block as libvosk's partial | 442 / 444, 2 earlier, 0 later | G3, same p50 and p90 within a block |
| segment word sequence equal | 193 / 205 (94.15%) | 99% |
| word times within one output frame, equal segments | 438 / 444 (98.65%) | 95% |
| endpoints within 0.2 s | 167 / 172 (97.09%) | 95% |
| compute per 40 ms block, p50 / p95 / p99 | 0.04 / 3.83 / 4.21 ms | p95 5 ms |
| real-time factor | 0.017 | 0.05 |

Word disagreement over 471 reference words: 5 words only in libvosk's
finals, 3 only in the runtime's, 6 substitutions.

## Reading

Partials, word times and endpoints pass. Segment equality misses the 99%
mark by 12 segments whose difference is a near tie on the acoustic side
(a rank word against a file letter, one colour word against another);
the split is symmetric, so neither side is emitting noise words. The
front-end differences that remain, 7.8e-4 on MFCC and an unmeasured
i-vector difference, are the plausible source, and the i-vector oracle
is the open leg of G1. Compute is inside the budget with the stock chunk
of 24 frames: a block that computes a chunk costs about 4 ms, of which
the network is the larger part and the i-vector update most of the rest,
and every other block well under 0.1 ms. Two runs of the same audio
produce the same output; ties in token cost go to the newest token.

These figures are a later run than the reading below, after the matrix
kernel was retiled and after `[[rr:TD-6]]` scaled the graph cost on
finals. The retile is arithmetically exact and moved nothing but
compute; TD-6 moved segment equality from 191 to 193 and the words
present only in libvosk's finals from 7 to 5, and left partials
identical at 99.72%, which is what a change confined to the choice of
final should do.

## What decided the partial figure

The first streaming runs agreed with libvosk on 93% of blocks and
surfaced words a chunk late, because the graph placed word labels where
the determinized lexicon disambiguates a word among the whole
vocabulary. libvosk's lookahead composition pushes each grammar arc's
label and weight to the first arc from which the word is the only
grammar word reachable. Reproducing that filter stack in
`[[rr:TD-2#The graph: composition]]` moved the figure to 99.7%.
