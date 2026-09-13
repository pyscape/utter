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

Words on wordless audio put a number on the complaint the runtime
exists to answer, and the number is small and it is not the decoder's.
Fed a quiet room cold under the full grammar, neither engine puts a
word at rank 0 on a single block of ten thousand, at any floor from
digital silence to -50 dBFS, and the words that do appear appear in
finals: about 2.25 a minute at every floor a room has, and none at all
on digital zeros, which is where a silent
stretch's word comes from, `[[rr:TD-8#A silent stretch closes as
libvosk closes it]]`, and not a partial a host could have read first.

The dial that moves it is the grammar, not the microphone. From 35 to
92 entries the stretch after a spoken word carries at most 0.56 words a
minute; at 150 it is 3.06 and the longest run a word holds jumps from 4
blocks to 22, near a second of a host seeing a word nobody said. Both
engines move together, so this is libvosk's behaviour reproduced
rather than this runtime's defect, and it is the strongest argument on
the page for keeping a command grammar small.

The floor gate works and the unknown-word symbol does not. At 8 dB
above the reported floor the gate takes 14 of the 15 wordless final
words at -50 dBFS and 12 of 13 at -40, in every case leaving one
behind, and `[unk]` at any cost from 8 down to -2 changes the rank-0
count by nothing. That is the private sweep's finding
(`[[rr:The unknown-word symbol against silence phantoms]]`) reproduced
on public audio.

Two cells in the whole census differ between the engines, and both are
one event rather than a rate: a six-block run at a -40 dBFS floor and
an eighteen-block run at 246 entries, each seen by the runtime and not
by the stock wheel, each inside a single recording. They are recorded
here because a one-way excess is what a decoder dropping a reading
looks like; at two events in some twenty thousand paired blocks there
is nothing to test, and the paired tests over samples find no
difference at all.
