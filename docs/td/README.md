# Technical decision log

One file per decision, numbered once and never renumbered. A record
fixes how the library is built: architecture, algorithms, tooling, and
engineering policy. What the library must do for a consumer is stated
in `usecases/`; a record answers a use case, it never restates one.
TD-1 is the description of this system.

## File and heading

File: `docs/td/NNNN-short-slug.md`, four-digit zero-padded number.
First line: `# TD-N: Title`. rr indexes that heading as the record
anchor, so a marker on the record ID cites the whole record and a marker
on the ID plus a section heading cites that section alone:

```
[[rr:TD-N]]
[[rr:TD-N#Decision outcome]]
```

Mint markers with `rr at`, never by hand. `rr index && rr verify` from
the repository root must pass before a change to `docs/` or `src/` is
committed.

## Changing a record

A record states what is. When a decision changes, the record changes
with it, or a later record replaces it and says so in its own text.
Before either, `rr search 'TD-N'` lists every comment and document
that leans on the record; each one is revisited in the same change.

## Sections

Copy `TEMPLATE.md`. Every section is required; "none" is an acceptable
body for Considered options when the decision was forced. Implemented
by is optional: code cites the record with a bare marker, and
`rr search 'TD-N'` lists every citation, so no list has to be kept.

## Index

| ID | Title |
|---|---|
| TD-1 | The technical decision record and the references that carry it |
| TD-2 | The streaming runtime for Vosk models - scope, stages, interface, dependencies and gates |
| TD-3 | The matrix kernel's tile is bounded by the register file |
| TD-4 | Compiled grammars are kept on the model |
| TD-5 | The composition's lookahead is a function of the state pair |
