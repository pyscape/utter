# TD-4: Compiled grammars are kept on the model

- Tags: graph, grammar, performance, api

## Context

`[[rr:TD-2#The graph: composition]]` composes eagerly, to reproduce
libvosk's lookahead filter stack arc for arc. libvosk composes lazily
while decoding, so its recognizer costs almost nothing to build.

Eager composition moves that cost to construction. The stock API,
which this library mirrors, invites a recognizer per utterance: 4.8 ms
a clip under a 92-entry grammar against the wheel's 0.4 ms, for an
identical graph every time.

## Considered options

- **Bounded, least recently used, held on the model.** Taken.
- **Compose every time.** The state before this record.
- **Hand the graph to the caller.** Adds public surface, and every
  host solves the same problem. The stock API has no such shape, so a
  host porting from it would not find it.
- **Unbounded cache.** Grows without limit for a host building
  grammars from user input.
- **Compose lazily.** Ruled out by
  `[[rr:TD-2#The graph: composition]]`.

## Decision outcome

### The model keeps the grammars most recently asked of it

Keyed by the grammar with the unknown-word cost and the state bound
that produced it, evicting the least recently used beyond a fixed
count. A recognizer on a grammar already compiled takes the graph as
it is. The count is a constant, not an option.

### The lock spans the composition

Held across the composition, not only the lookup, so several streams
opened on one grammar compose it once between them.

## Consequences

Construction after the first falls to about 0.03 ms, below the wheel's
0.3 ms. The first is unchanged; that cost is `[[rr:TD-5]]`.

A model retains up to the bound for its lifetime, so its resident size
depends on the grammars asked of it. A host alternating between more
than the bound recompiles on every switch, which is the behaviour
before this record, not a new penalty.

A thread building on a second, uncached grammar waits for a first
composition. Composition is milliseconds and construction is off the
audio path, so the simpler rule beats a lock per grammar.

## Implemented by

- `[[rr:grammar_graph]]`
