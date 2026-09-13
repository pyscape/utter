# TD-6: The graph cost is scaled before a final is chosen

- Tags: decoder, parity, finals

## Context

libvosk rescores before it emits a final: `recognizer.cc:827` applies
`fst::ScaleLattice(fst::GraphLatticeScale(0.9))` to the lattice, above
every output branch. `PartialResult` does not.

This runtime took the raw best path at scale 1.0. Scaling the graph
half by 0.9 favours whichever path carries more graph cost. A path
with a word carries about `ln(2N)` more than the empty path, so at a
92-entry grammar a word gained about 0.52 over silence that we never
granted: 46 clips of Speech Commands read as silence where libvosk
read a word, against 2 the other way.

## Considered options

- **Carry the graph cost on the token and scale it on finals.** Taken.
- **Scale during the search.** Changes which tokens survive the beam,
  which libvosk does not do; its search runs at 1.0.
- **Leave it.** The divergence is one-way and grows with the grammar.

## Decision outcome

A token carries the arc weights along its path apart from its total.
A final reading, in `best_token` and in the grouping and ranking of
`alternatives`, is chosen by `(cost + final) - 0.1 x (graph + final)`.
A partial is chosen at the raw cost, as libvosk's is.

## Consequences

On the gate corpus, segments equal to libvosk rise from 191/205 to
193/205 and the words present only in libvosk's finals fall from 7 to
5. Partials are untouched: 99.72% before and after.

Accuracy on a public set may fall, because libvosk's errors are
reproduced along with its successes: it keeps the graph-costlier word
on some word-against-word ties that this runtime otherwise reads
correctly. Parity is the contract.

The match is first-order only. libvosk rescales a determinized lattice
and can then choose a path no single-best token chain represents, so
some disagreement remains by the same limit as
`[[rr:TD-2#Decision outcome]]`, which keeps lattices out of scope.

## Implemented by

- `[[rr:GRAPH_SCALE]]`
