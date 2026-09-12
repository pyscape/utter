# TD-4: Compiled grammars are kept on the model

- Status: Proposed
- Date: 2026-09-11
- Tags: graph, grammar, performance, api

## Context and problem statement

`[[rr:TD-2#The graph: composition]]` composes the grammar with the
model's graph eagerly, because the label placement that gives the
partial its latency depends on reproducing libvosk's lookahead filter
stack arc for arc. libvosk itself composes lazily while it decodes, so
building one of its recognizers costs almost nothing and the work is
spread over the decode.

Eager composition puts that cost at construction instead. The stock
API, which this library mirrors, invites a host to build a recognizer
per utterance, and a benchmark that does exactly that paid the whole
composition on every clip. Under a 92-entry grammar that was 4.8 ms a
clip against the wheel's 0.4 ms, for a graph identical every time.

## Decision drivers

- The eager composition itself is not negotiable; it is what
  `[[rr:TD-2#The graph: composition]]` buys.
- A host must not be punished for the construction pattern the stock
  API teaches.
- Memory is bounded and predictable: a decoder that holds graphs for
  grammars nobody will ask for again is a leak.
- Two streams opened on one grammar at once must not each compose it.

## Considered options

- **Keep the compiled graphs on the model, bounded, least recently
  used evicted.** Taken.
- **Compose every time.** The state before this record; correct, and
  it charges a per-utterance host the full composition forever.
- **Hand the graph to the caller and let a host hold it.** Puts the
  cost where the host can see it, but adds public surface and makes
  every host solve the same problem. The stock API has no such shape,
  so a host porting from it would not find it.
- **An unbounded cache.** Trivial, and it grows without limit for a
  host that builds grammars from user input.
- **Compose lazily, as libvosk does.** Ruled out by
  `[[rr:TD-2#The graph: composition]]`.

## Decision outcome

A model holds the compiled graphs of the grammars most recently asked
of it, keyed by the grammar together with the unknown-word cost and
the state bound that produced it, and evicts the least recently used
beyond a fixed count. A recognizer built on a grammar the model has
already compiled takes the graph as it is.

The compiling lock is held across the composition, not merely across
the lookup, so that several streams opened on one grammar compose it
once between them rather than each in parallel.

The count is a constant in the source, not an option: a host that
needs a different one is a host whose grammar set is better solved by
a different design, and that is a decision to record, not a knob.

## Consequences

Construction after the first on a grammar falls to about 0.03 ms, from
4.8 ms, and is then cheaper than the wheel's 0.3 ms. The first
construction is unchanged; making the composition itself affordable is
`[[rr:TD-5]]`.

A model now retains up to the bounded number of graphs for its
lifetime, so its resident size depends on the grammars it has been
asked for and not on the model files alone. A host that alternates
between more grammars than the bound recompiles on every switch, which
is the old behaviour rather than a new penalty.

Because the lock spans the composition, a thread building a recognizer
on a second, uncached grammar waits for a first composition to finish.
Composition is milliseconds and construction is not on the audio path,
so the simpler rule is preferred to a lock per grammar.

## Implemented by

- `[[rr:grammar_graph]]`
