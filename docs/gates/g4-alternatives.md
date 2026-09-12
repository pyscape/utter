# G4: partial alternatives

Gate: `[[rr:TD-2#Verification and acceptance]]`, G4, under the ruling
`[[rr:TD-2#Silence and unknown speech announce themselves]]`.

## What ran

- Model: `vosk-model-small-en-us-0.15`, stock configuration; the same
  grammar as G2.
- Corpus: a consumer census set of 61 short takes, mostly silence
  between commands, 20,914 partial blocks.
- Oracle: `vosk` 0.3.45 through `scripts/g2.py oracle`, 40 ms blocks,
  dither 0.
- Runtime: `stream --alternatives 5 --partial-words`, scored by
  `scripts/g4.py`.

## Figures

| Quantity | Value | Gate |
|---|---|---|
| rank 0 text equals the partial | 20,914 / 20,914 blocks | every block |
| best path without a word offered as `[sil]` at rank 0 | 18,814 / 18,814 | every such block |
| blocks whose libvosk partial was empty | 18,819 | |
| of those, rank 0 a vocabulary word | 5 (0.027%) | 0 |
| partials with a word whose rank 1 is one word away | 1,962 / 2,100 (93.4%) | census, recorded |
| `[sil]` entries in partial word lists | 4,087 | |

The same run against the oracle: partial text equal on 99.67% of
blocks, segment word sequences equal on 138 of 144 (95.8%), word times
within one frame on 97.6%, endpoints within 0.2 s on 82 of 83.

## Reading

The five blocks where libvosk's partial was empty and the runtime's
rank 0 carried a word are blocks where the two decodes differ, the
0.3% of G2, not a silence reading forced into a word: on every block
where the runtime's own best path carries no word, rank 0 reads `[sil]`.
The contest census says that when a word is showing, a rival one word
away is alive in the beam on 93% of blocks; the fork's lattice n-best
had no current figure to compare, which is why the census is recorded
rather than gated.
