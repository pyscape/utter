---
name: rr
description: Cite code and prose by stable anchor with rr (ripref) - mint a marker with rr at, confirm with rr read, print a marker's content with rrcat, gate with rr verify. Read before citing a decision, heading, scenario, or symbol from a comment, plan, finding, or design document. Repo conventions (prefixes, where rules live, gates) are in the repo's reference doc.
---

# rr

A `file:line` breaks on the next insert; a rule restated in prose forks
silently. A marker names WHAT is cited and `rr verify` reports when it
stops resolving. Conventions per repo: ripref `README.md` + `doc/ad/`;
ChessAmis `docs/dev/references.md`; elsewhere `.rr.toml` and the defaults.

## Anchors

| Kind | Defined by | Identity |
|---|---|---|
| record | heading opening `ID:`, ID = `LETTERS-DIGITS` | `TD-3` |
| heading | any other Markdown heading | full title |
| scenario | Gherkin `Scenario:` / `Feature:` | title |
| symbol | code definition (Rust, Python) | name |

A list item or bold run is not an anchor; promote it to a heading.
Hidden paths (`.rr.toml`, `.claude/`) are not indexed; cite by path.

## Making a file citable

You never WRITE an anchor. The definitions already in the file ARE the
anchors, so making something citable means giving it a definition worth
naming. `rr at --all <file>:<line>` prints the nest covering a line,
outermost first.

| File | What is an anchor | Nest |
|---|---|---|
| `.py` | `class`, `def` - methods and inner defs too | class, then method |
| `.rs` | `fn`, `struct`, `enum`, `mod` | mod, then item |
| `.feature` | `Feature:`, `Scenario:` | feature, then scenario |
| `.md` | headings; a record's `ID:` heading outranks a plain one | outer, then inner |

Cite BARE when the name is unique tree-wide, `path#name` when it is
not. `rr at` prints the shortest form that resolves, so paste what it
gave you rather than choosing: `[[rr:VoiceController]]` and
`[[rr:HeapEntry]]` are bare, while ChessAmis has `create_chapter` in
both Rust and Python, so the Rust one only ever resolves as
`[[rr:rust/src/store/chapter.rs#create_chapter]]`.

NOTHING AT MODULE LEVEL IS AN ANCHOR, which is the commonest surprise:
a constant, an import, a module docstring, a bare dict. `rr at` answers
"no anchor covers" and there is no marker to paste. Three ways out, in
order: move the lines inside a `def`/`class`/`mod` that deserves a name;
cite the nearest enclosing symbol and name the detail in prose; or, for
prose, promote it to a heading. Naming a thing to make it citable is
the point, not a workaround.

A rename is a breaking change to every citation. `rr search '<name>'`
before renaming a symbol, exactly as for a record.

## Verbs

```bash
rr index                  # after any edit
rr at <file>:<line>       # location -> marker (--all: the whole nest)
rr read '<anchor>'        # marker -> file:start-end; exit 1 = dangling
rrcat '<anchor>'...       # markers -> the lines they span, in order, -- between
rr search '<anchor>'      # who cites it; --markers all, --mentions bare paths
rr verify                 # the gate: 0 findings
```

## Choosing a verb

Stop at the first case that resolves the reference; a marker already
returned by one command is not re-confirmed with another.

1. Marker or anchor already in hand -> `rrcat` it directly.
2. Only a `file:line` is known -> `rr at`, then `rrcat` the result.
3. Only the subject's words are known -> `rr search`, take the
   narrowest matching marker, then `rrcat` it.
4. Diagnosing why a marker won't resolve -> `rr read` (locations and
   ambiguity; `rrcat` is what prints the content).

Cite the narrowest anchor whose span holds the fact - a record's
section over the whole record, a symbol over the file that defines
it. Never invent a marker from a title or ID; resolve it first.

## Following a marker

`rrcat '[[rr:TD-3#Decision outcome]]' '[[rr:TD-4]]'` prints each
definition's text in argument order, `--` between them: one call for
every marker you hold, instead of `rr read` plus a Read per marker.
Bare ids (`TD-3`) work too. A stale index is rebuilt once (`rrcat:
rebuilt the index` on stderr) and the read retried. Exit is the worst
`rr read` gave (1 dangling, still printing the rest; 2 a usage or
read failure; 3 still stale after the rebuild);
an ambiguous anchor prints every definition, `--` separated. Use it
whenever markers are the input and the content is what you need: a
cited rule, a spec section, a finding's source. Handing off, send the
marker plus only the case-specific status beyond it; the recipient
runs `rrcat` for the body rather than being handed a paste of it.

## Minting

1. Read the lines that state the thing.
2. `rr at --all <file>:<line>`; take the innermost anchor covering them.
   Prefer `TD-3#Decision outcome` over `TD-3`, and a rule's home over
   code that obeys it.
3. `rr read` it. Paste as printed. Never from memory.
4. If the title is unique only for now (every record has a "Decision
   outcome"), qualify by record yourself and `rr read` it.
5. `rr index && rr verify` before handing over.

## Verifying

`rr index -q` first - the gate only sees what got indexed. Scope to
what changed (`rr verify <path>...`) or run repo-wide (`rr verify`,
no path) when a record moved or the question is whether the whole
tree complies. `0 findings` is clean; anything else is a failure to
fix or report - name each finding's marker and location, do not wave
a nonzero count through.

## Form

- Python `#`, Rust `//`, commit message: bare, alone on its line.
- Rust `///` `//!`, Markdown: backticked (rustdoc and renderers eat
  the brackets; a code span survives reflow).
- Gherkin: bare, on a `#` line above the scenario.
- Prose beside a marker only when it says what the marker cannot.

## Before changing a record

`rr search 'TD-N'` lists every citation; revisit each in the same
change. Records keep no list of implementers; the index is the list.

## What it looks like

```python
# [[rr:TD-1#Decision outcome]]
def leg_references(rr, runner=None):
```

```rust
/// `[[rr:AD-1#Decision outcome]]`
pub fn parse_reference(s: &str) -> Reference {
```

```gherkin
# [[rr:PD-2]]
Scenario: Turning the board round shows it from the other side
```

Markdown: "the gate exits 1 on findings (`[[rr:AD-3]]`)". The common
case is the first one: marker alone, nothing else.

## Traps

Fenced code is invisible to rr (right for examples, wrong for a
citation). Never rewrite markers in string literals or fixtures. Only
`[verify]` and `[scan.*]` are configurable; anchor kinds are fixed.

A Gherkin `#` comment above a scenario sits inside the PREVIOUS
scenario's span, so `rr at` on the line you are writing a marker on
answers with the scenario before it. Write the marker there anyway (see
Form); just do not read that answer as the anchor you meant.

A plain heading is unique until someone adds the same one elsewhere,
and then every bare citation of it breaks at once. `rr verify` reports
it as "ambiguous marker"; the fix is to make the new heading
subject-naming, not to path-qualify the old citation.
