# TD-5: The composition's lookahead is a function of the state pair

- Tags: graph, grammar, performance, algorithm

## Context and problem statement

Composing the graph with the grammar, per
`[[rr:TD-2#The graph: composition]]`, looks ahead from every
output-epsilon arc of every graph state reached: which grammar arcs
are reachable through that state's intervals, their log-sum weight,
and whether exactly one is reachable, the case that pushes the
grammar's arc whole onto the epsilon.

Arc by arc, that scan walks the grammar's whole arc list. The list
grows with the vocabulary and so does the number of composed states
scanning it, so cost grew as roughly the square of the grammar: 1.0 ms
for 35 entries, 16.6 ms for 246, while the graph itself grew 4.6-fold.

## Decision drivers

- The composed graph does not change; it is what
  `[[rr:TD-2#The graph: composition]]` specifies, arc for arc.
- A few hundred command words is the size the library is for.
- The fix is in the composition. `[[rr:TD-4]]` spares repeat
  constructions and leaves the first untouched.

## Considered options

- **Keep the result per state pair.** Taken.
- **Intersect the sorted arc list with the sorted intervals.** Helps
  where few words are reachable, not at the root where all are. Treats
  the scan's cost, not its repetition.
- **Prune the composed graph first.** Changes what the decoder sees.
- **Compose lazily.** Ruled out by
  `[[rr:TD-2#The graph: composition]]`.

## Decision outcome

The lookahead reads only the grammar state it starts from and the
reachability set of the graph state being entered, so its result is a
function of that pair. It is computed once per pair and reused by
every composed state sharing it. Anything it reads beyond that pair is
a change to the key.

## Consequences

Composition becomes linear in the graph it produces rather than
quadratic in the vocabulary: 0.98 to 0.76 ms at 35 entries, 4.15 to
2.06 at 92, 16.63 to 5.46 at 246. The graph is identical state for
state and arc for arc; the same corpus decodes to byte-identical
partials and segments over 42,878 blocks.

The memo holds one entry per pair visited, bounded by the composition
already being done, and is dropped with it.

The first construction on a grammar is still the slower one. This
removes the growth, not the cost.

## Implemented by

- `[[rr:Look]]`
