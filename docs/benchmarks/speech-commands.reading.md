## Reading

Accuracy and first-appearance latency are at parity with the stock
wheel, and the finals agree on 99.15% of clips. Where they differ, the
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

Compute is no longer the gap it was. On continuous audio the runtime
is at 0.0118 against the wheel's 0.0124, and ahead where it counts,
2.50 ms against 2.86 at the 95th percentile, which is the blocks that
run a network chunk. It is behind on the blocks that do not: 0.091 ms
against 0.048. Construction is the other way around again, 0.026 ms
against 0.288, because a compiled grammar is kept (`[[rr:TD-4]]`); the
first is dearer and grows with the grammar.

Two figures a host should read before choosing a grammar. Accuracy
falls about five points from 35 entries to 246, so distractors are not
free. And one first partial in eight is later revised, on both
engines, so a host that acts on the first word it sees acts wrongly
about 12% of the time.

Noise costs both engines the same. Accuracy halves between clean and
0 dB, and the paired test finds no difference between the engines at
any level. An earlier run showed the runtime 2.3 points behind at
0 dB; that was TD-6's bug, not the front end.
