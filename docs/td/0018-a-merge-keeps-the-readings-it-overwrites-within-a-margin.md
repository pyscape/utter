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
no margin set there is none. Decoding alone, over 295 s of the testing
split's clips joined into one utterance, one core, the median of five
passes: 0.0258 ms a frame without a margin and 0.0305 ms with a margin
of 3 nats. The readout, at the end of one utterance, against the
survivors alone: 0.071 against 0.033 ms 5 s in, 0.209 against 0.064 ms
20 s in, which is as long as the model's last endpoint rule lets an
utterance run.

## Consequences

- A host that sets a margin sees, among the readings, the ones the
  search would have dropped while they were close. On the clip of the
  context, with a margin of 3 nats, "no seven" is offered 0.19 nats
  behind "go seven" at every advance from its first sighting, where
  without it the partials carry a copy 6.6 nats behind for six advances
  and then nothing.
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
