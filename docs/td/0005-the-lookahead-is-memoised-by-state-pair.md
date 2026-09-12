# TD-5: The composition's lookahead is a function of the state pair

- Status: Proposed
- Date: 2026-09-11
- Tags: graph, grammar, performance, algorithm

## Context and problem statement

Composing the model's graph with the grammar, as
`[[rr:TD-2#The graph: composition]]` requires, looks ahead from every
output-epsilon arc of every graph state reached: which of the
grammar's arcs are reachable through that state's reachability
intervals, their log-sum weight, and whether exactly one is reachable,
which is the case that pushes the grammar's arc whole onto the
epsilon.

Done arc by arc, that scan walks the grammar's entire arc list each
time. The grammar's arc list grows with the vocabulary, and so does
the number of composed states that scan it, so the cost grew as
roughly the square of the grammar. Measured on one model: 1.0 ms to
compose a 35-entry grammar, 16.6 ms for 246 entries, while the graph
itself grew only 4.6-fold. A host with a few hundred command words
paid tens of milliseconds to open a recognizer.

## Decision drivers

- The composed graph must not change: it is what
  `[[rr:TD-2#The graph: composition]]` specifies, arc for arc.
- A grammar of a few hundred words is the size the library is for, so
  the cost must be reasonable across that range and not merely at the
  small end.
- The fix must be in the composition, not a cache around it;
  `[[rr:TD-4]]` already spares repeat constructions and leaves the
  first one untouched.

## Considered options

- **Keep the lookahead's result per state pair.** Taken. The scan
  reads only the grammar state it starts from and the reachability
  set of the graph state being entered, so its result is a function of
  that pair and nothing else, and the pair recurs across every
  composed state that shares it.
- **Walk the sorted arc list against the sorted intervals.** The
  grammar's arcs are sorted by label and the intervals are disjoint
  and sorted, so the two can be intersected instead of testing every
  arc. It helps where few words are reachable and not at all near the
  root, where everything is; it treats the scan's cost rather than its
  repetition.
- **Prune the composed graph first.** Changes what the decoder sees.
- **Compose lazily.** Ruled out by `[[rr:TD-2#The graph: composition]]`.

## Decision outcome

The lookahead is computed once per pair of grammar state and entered
graph state, and reused for every composed state that shares the pair.
Anything the lookahead reads beyond that pair would break the identity
that makes this sound, so a change to what it reads is a change to the
key.

## Consequences

Composition becomes linear in the graph it produces rather than
quadratic in the vocabulary: 0.98 to 0.76 ms at 35 entries, 4.15 to
2.06 at 92, and 16.63 to 5.46 at 246, so seven times the entries costs
seven times the work rather than seventeen. The composed graph is
identical state for state and arc for arc, and the same corpus decodes
to byte-identical partials and segments over 42,878 blocks.

The memo holds one entry per pair actually visited, which is bounded
by the composition it is already doing, and is dropped with the
composition.

The first construction on a grammar is still the slower one; what this
record removes is the growth, not the cost. A grammar near the
supported limit still takes a few milliseconds to compose, once.

## Implemented by

- `[[rr:Look]]`
