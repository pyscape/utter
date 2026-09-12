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
