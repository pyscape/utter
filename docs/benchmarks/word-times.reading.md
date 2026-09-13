## Reading

The word times on finals are the wheel's to within one output frame.
Of 10,521 words paired on the 10,505 clips whose finals agree, the
start is exact on 97.8% and within one 30 ms frame on 99.9%; the end
is exact on 88.8% and within a frame on 98.9%. Taking a word as
matched only when both its start and its end are within a frame, 98.8%
match and 99.5% are within two frames.

The 54 words beyond two frames are, on 53 of them, single-word
finals. On 44 the start agrees and the end differs by three frames or
more, up to 330 ms: where the word's tail stops and the trailing
silence begins. On the other ten the start moves too, once by 510 ms.
Over all paired words the runtime puts the end later on 620 and earlier
on 563, so there is no bias, only a boundary the two searches settle
differently on a word in two hundred.

Every time the runtime reports is a multiple of 30 ms. The wheel
reports 25 that are not, on 17 two-word finals, each the boundary
between the two words, which its lattice alignment places between
frames. Ten of those finals are paired here; on them the runtime's
boundary is the nearest frame.

The claim on the Vosk reference page is stated from this table: word
times match the wheel's to within one frame, on the public split, on
the finals the two engines agree about.
