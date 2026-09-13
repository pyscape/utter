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

Copy `TEMPLATE.md`. Decision outcome is the record; the rest are
written when they carry something it does not. Considered options
usually does, and is where a record earns its length: the next reader
reaches for the rejected option. Implemented by is optional, because
`rr search 'TD-N'` lists every citation and no list has to be kept.

## Index

| ID | Title |
|---|---|
| TD-1 | The technical decision record and the references that carry it |
| TD-2 | The streaming runtime for Vosk models - scope, stages, interface, dependencies and gates |
| TD-3 | The matrix kernel's tile is bounded by the register file |
| TD-4 | Compiled grammars are kept on the model |
| TD-5 | The composition's lookahead is a function of the state pair |
| TD-6 | The graph cost is scaled before a final is chosen |
| TD-7 | The best path is read once per decoding advance |
| TD-8 | Silence is the host's gate, on a floor the runtime reports |
| TD-9 | Each reading carries its relation to the partial and the motion of its lead |
| TD-10 | A wordless reading says whether a word has begun |
