# Technical decision template

Copy the block below into `docs/td/NNNN-short-slug.md`. Keep the
heading shape exactly; rr keys the record on it. Decision outcome is
the only required section: it is the record. Write the others when
they carry something a reader cannot get from the outcome, and leave
them out when they do not.

```
# TD-N: Title in one line

- Tags: comma, separated

## Context

What is true today and why it forces a choice.

## Decision drivers

- Only when the options are judged against something the context does
  not already state.

## Considered options

- **Option.** One line on why it was rejected or taken.

## Decision outcome

The ruling, stated so a reader can tell whether code obeys it.

## Consequences

What becomes easier, what becomes harder, what must change to comply.

## Implemented by (optional)

- Only when the record names the thing that carries it out: a marker
  minted with rr at on a line inside it, backticked. The reverse
  question, what cites this record, is rr search, not a list.
```
