# TD-1: The technical decision record and the references that carry it

- Tags: documentation, references, tooling, process

## Context and problem statement

This library starts from a specification written in another repository
and from a use case that names, in detail, what its first consumer
needs. Both are prose. Prose that is copied into a second place drifts
from the first, and a bare path or `file:line` pointer breaks silently
on the next rename. Before any code exists, the repository needs one
place where an engineering ruling is stated, one way to cite it from
code and from other documents, and a mechanical check that every
citation still resolves.

## Decision drivers

- A ruling is written once and cited everywhere else; a comment or
  document references a rule, it never restates it.
- A reference is checked by a tool, so drift is a finding, not a
  surprise.
- A record's identity is stable for the life of the repository:
  numbers are never reused and records are never deleted.
- A record describes what is, not what someone hopes to be allowed to
  do: it is written by whoever takes the decision, and it is in force
  because the code is built that way.
- What the library must do is kept apart from how it is built: a use
  case states a need, a record answers it.

## Considered options

- **Design notes in the README and in code comments.** The first
  version of every note is fine and the third is wrong. Rejected.
- **Architecture decision records with a conventional template, cited
  by path.** Right shape, but the citations rot. Rejected.
- **Numbered decision records with rr markers as the only way to cite
  them, and `rr verify` as the gate.** Taken. It is the system the
  first consumer's repository runs, so the two repositories agree on
  what a record is and how it is cited.

## Decision outcome

### One log, one record per decision

`docs/td/` holds the technical decision log. A record is one file named
`docs/td/NNNN-short-slug.md` whose first line is `# TD-N: Title`. The
number is assigned once from the next free number in `docs/td/README.md`
and is the record's identity: a record is never renumbered. A record
that stops applying is changed or replaced, not left standing.

### The sections

Every record carries, in this order: the tags line;
Context and problem statement; Decision drivers; Considered options;
Decision outcome; Consequences; and, when the record names what carries
it out, Implemented by. `docs/td/TEMPLATE.md` is copied, never
retyped. Decision outcome is written in subsections whose headings
state the ruling, so that a citation can point at one ruling rather than
at the whole record.

### A record carries no ceremony

A record has no status field and no date. It describes the decision
the library is built on, so the only question a reader asks of it is
whether the code still matches, and `rr verify` plus the gates answer
that. A record that has stopped being true is changed or replaced in
the same commit as the code that stopped matching it, never left
standing with a label saying to disregard it.

Before a record is changed or replaced, `rr search 'TD-N'` lists
everything that cites it, and each citation is revisited in the same
change.

### Citation is by marker

Every reference to a record, from code, from another record, from a use
case or from the README, is an rr marker, on the record for the whole
of it and on the record plus a section heading for one ruling:

```
[[rr:TD-N]]
[[rr:TD-N#Decision outcome]]
```

Markers are minted with `rr at` on the line they point to, never typed
by hand, and read with `rr read` or `rrcat`. A path, a line number, or
a paraphrase is not a citation.

### The gate

The repository carries a `.rr.toml` profile: Markdown and Rust comments
are in scope, fixture and build directories are excluded, and the four
marker rules are enforced. `rr index && rr verify` must pass before a
change under `docs/` or `src/` is committed. The index lives in
`.ref-cache/`, which is ignored.

### Use cases are the other side of the log

`usecases/` holds what a consumer needs, in the consumer's terms and
without its name. A use case is not a decision;
it is the thing a record answers. A record that answers a use case
cites the use case's section by marker, and a use case that a record
now satisfies is not edited to say so: `rr search` on the record tells
the reader.

### Comments in code

A comment in `src/` is either a bare rr marker citing the record or
section the code obeys, or a statement of something a competent reader
of the code would get wrong without it. A comment that restates the
code is deleted at review.

## Consequences

- The specification the library starts from becomes TD-2, cited from
  code as it lands, rather than a document that code silently departs
  from.
- Adding a record costs a file and a line in the index; changing one
  costs a visit to everything that cites it, which is the point.
- Contributors need rr installed to pass the gate; the skill under
  `.agents/skills/rr/` tells an agent how to use it.

## Implemented by

- `.rr.toml`: the profile the gate runs under.
- `docs/td/README.md` and `docs/td/TEMPLATE.md`: numbering, sections,
  and the index.
