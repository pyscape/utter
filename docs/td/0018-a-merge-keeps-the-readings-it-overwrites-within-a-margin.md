# TD-18: A merge keeps the readings it overwrites within a margin of the path that wins

- Tags: decoder, readings, alternatives, interface, measurement

## Context

The decoder keeps one token per graph state,
`[[rr:TD-2#The decoder: search]]`. Where two paths reach one state on
one frame, the cheaper carries on and the other's history is gone. The
readings are the survivors grouped by word sequence,
`[[rr:TD-2#The decoder: partial alternatives]]`, so a reading whose
close copies all lost merges is offered only through a copy that lost
none, further behind, or not at all. A host that settles close readings
by knowledge of its own can only choose among the readings offered,
`[[rr:5. What the application needs from this library]]`.

Under a grammar of two phrases, "go seven" and "no seven", the
"seven" clip in `tests/data` reads both ways 0.19 nats apart: the two
first words meet in the states of "seven", and "go seven" overwrites
"no seven" there. The partials offer "no seven" through a copy that
lost no merge, 6.6 nats behind, for six of them, and then not at all.
No beam brings such a reading back, as no beam moves a final under a
grammar, `[[rr:The finals over a range of beams]]`; a lattice would
keep it.

The census `[[rr:TD-9#No lattice is added for this feature]]` left this
open: it counts collisions and readings that leave the beam, so a
reading overwritten while close, whose last copy dies far behind, is
counted by neither.

## Considered options

- **Kaldi's lattice: every arc kept, determinized, n-best read off.**
  Out of scope, `[[rr:TD-2#Scope]]`, and paid on every arc of every
  frame for readings no host asked about. Rejected.
- **k tokens per state.** Every arc is taken k times to keep what a few
  merges a frame drop. Rejected.
- **A join on the winning path.** Taken. A record on the winner's path
  holds the loser's history; the search does nothing else, and nothing
  at all without a margin.
- **A margin against the frame's best token.** A kept reading's gap to
  the path that beat it never changes after the merge, and that path
  may lead later. The margin is taken against the winner.
- **Keeping only where the words differ.** A loser that reads as the
  winner may carry kept readings the winner's path lacks; it is kept
  when it does.
- **A key marking a kept reading.** A reading is its word sequence,
  `[[rr:TD-9#Every reading carries its lead's motion]]`; whether its
  cheapest copy lost a merge says nothing about it. No key.
- **Letting a kept reading take rank 0, or veto the host's bound.** The
  setting reports and does not decide. Rejected.
- **Reading out every kept reading at every advance.** Kept readings
  multiply along a long utterance: 20 s into one, the readout listed
  3,237 readings and took 4.7 ms, against 0.07 ms for the survivors
  alone. Rejected for reading out only what a list could show.
- **On by default.** A host that does not settle close readings itself
  pays for readings it never reads. The margin is a setting, off by
  default.

## Decision outcome

### A merge keeps what it overwrites within the margin

With a margin set, a merge whose loser is within the margin of the
winner, and reads a different word sequence above the history the two
share or carries a kept reading there, puts a join on the winner's
path. The join holds the loser's history and its cost and graph cost
over the winner. From the merge on the two share one future, so both
differences hold for every extension of the winner, and a kept reading
lives as long as the winner's path. What a loser kept is kept again at
the sum of the differences. A winner that has already passed its path
on by epsilon passes it on once more with the join. With no margin the
decoder is the one before this record.

### Kept readings are reported as readings

The partial's readings and the final's alternatives are the survivors'
word sequences and every kept reading within the margin of the path it
rides on, each at its cheapest copy, ranked as before: by cost on a
partial, by the scaled cost of `[[rr:TD-6#Decision outcome]]` on a
final. A kept reading carries `confidence`, `relation`, `lead_delta`
and word entries like any other, and its history is the one every
reading has, `[[rr:TD-9#Readings are read once per decoding advance]]`.
Its words and phones are the kept history's up to the merge and the
winner's after it. Where the kept history was in another phone at the
merge, it takes the winner's phone from the next frame.

Kept readings are read out only as far as the list asked for could
show them. One ranked past the n-th surviving reading, for a list of n,
is not read out, nor, on a partial, any kept on a path ranked past it;
below a join no more paths are followed than the list holds. Every
surviving reading keeps its history as before; a kept reading has one
from the advance it was first read out.

### Rank 0 and every decision stay the survivors'

Rank 0 is the best path, read off its own token as without a margin,
even on a final where a kept reading ranks better at the scaled cost;
such a reading is listed second. The endpoint rules, the host's bound
and its veto, the best path, the silence weighting and every key of a
result outside its readings are read off the survivors.

### The census counts at the merge

`stream --census` counts a merge that overwrites a different word
sequence within the census margin of the frame's best token when it
happens, whatever becomes of the reading's other copies, and lists each
word sequence so overwritten once per utterance, with the sample it
was first overwritten at. The census margin applies to every close
count the census takes.

### What it costs

The cost falls on the merges inside the margin and on the readout; with
no margin set there is none. Each pass of the search is compiled with
and without the margin, because one branch on it at every merge cost
the search 2% of its time with no margin set.

On the compute gate's streams of `[[rr:TD-17#What it costs]]`, 400
test clips in 40 streams, 571 s of audio in 40 ms blocks with words on
partials, the fastest of five repetitions per block on one core, each
cell 0.0.6 / no margin / margin 1 / margin 3:

| Configuration | p99 block, ms | Slowest block, ms | Real-time factor |
|---|---|---|---|
| plain | 2.506 / 2.530 / 2.568 / 2.619 | 4.34 / 4.58 / 4.43 / 4.67 | 0.01041 / 0.01045 / 0.01060 / 0.01073 |
| three readings | 2.606 / 2.623 / 2.685 / 2.723 | 4.55 / 4.66 / 4.72 / 4.85 | 0.01083 / 0.01089 / 0.01110 / 0.01117 |
| three readings, ten alternatives | 2.589 / 2.610 / 2.654 / 2.696 | 4.56 / 4.57 / 4.69 / 5.06 | 0.01077 / 0.01084 / 0.01098 / 0.01109 |

A margin of 1 costs 7.5 to 10.9 us a block and a margin of 3 about 13,
against 0.0.6. Without a margin a block costs 1.5 to 2.6 us more than
0.0.6's, inside the spread of the gate itself, whose 0.0.6 column moved
by up to 8 us a block between two runs of one configuration; the search
alone over 295 s of joined clips, the fastest of seven passes over five
runs, takes 24.40 us a frame against 0.0.6's 24.57. With the margin
off every result of the gate is byte for byte 0.0.6's, 14,334 in each
configuration, and with it on rank 0 moved in none.

The search alone with a margin of 3 takes about 4.7 us a frame more.
The readout, at the end of one utterance, against the survivors alone:
0.071 against 0.033 ms 5 s in, 0.209 against 0.064 ms 20 s in, which is
as long as the model's last endpoint rule lets an utterance run.

## Consequences

- A host that sets a margin sees, among the readings, the ones the
  search would have dropped while they were close. On the clip of the
  context, with a margin of 3 nats, "no seven" is offered 0.19 nats
  behind "go seven" at every advance from its first sighting, where
  without it the partials carry a copy 6.6 nats behind for six advances
  and then nothing.
- Over the 11,005 clips of the Speech Commands testing split under the
  92-entry grammar, ten readings and ten alternatives, a margin of 3
  nats gave 509 of 281,912 results a reading within 3 nats of rank 0
  that the survivors did not list, 171 of them finals. The census at
  that margin counts 1,286 merges overwriting a reading near the
  frame's best token, 934 distinct readings; 28 of those were offered
  by no result of their utterance without the margin, and 20 of them
  are with it. With the margin off every result is byte for byte
  0.0.6's, and with it on rank 0 moved in none.
- The margin offers what was close, not what was right. The clip's
  word among a final's first n alternatives is as often found as
  without it at n = 1 (10,109 clips), and a few clips less often at
  n = 2, 5 and 10 (10,504 against 10,509, 10,741 against 10,747,
  10,826 against 10,828), where a kept reading closer than the right
  word takes its place in the list.
- A surviving reading whose kept copy is cheaper is reported at the
  kept copy's cost; its place in the list moves with it.
- A kept reading is as exact as the survivors are: its cost is the one
  its own path had at the merge plus what the winner paid after it. It
  is not a lattice's posterior, and a reading that lost a merge by more
  than the margin, or was pruned, is still gone.
- `[[rr:TD-9#No lattice is added for this feature]]` is answered for
  retained histories; posterior mass stays deferred.

## Implemented by

- `[[rr:worth_keeping]]` and `[[rr:src/decoder.rs#add]]`: the merge.
- `[[rr:src/decoder.rs#readings]]`, `[[rr:fill_below]]` and `[[rr:trace_route]]`: the
  readout.
- `[[rr:count_overwritten]]`: the census at the merge.
