# TD-10: A wordless reading says whether a word has begun

- Tags: partials, silence, evidence, parity

## Context

`[[rr:TD-2#Silence and unknown speech announce themselves]]` gives the
reading of a best path that carries no word one name, `[sil]`, and the
README glossed it as the decoder having heard nothing. The word list
under the same partial was built by `[[rr:entries]]` from the path's
phones, and wrote a `[sil]` entry only over silence phones. The two
disagreed whenever the best path had entered a word and not yet named
it.

That is not rare. The composition places a word's label as soon as the
lookahead can tell the word from its neighbours, so a path inside a
phone that several grammar words share carries no label. On a room's
noise floor the acoustic model can prefer such a phone to silence: on
the first consumer's microphone the aspirated `HH` that begins four
grammar words scores below `SIL` from about two seconds into a quiet
stretch, the all-silence path leaves the beam near twelve seconds, and
at the 20 s length rule the only complete paths are those four words.
Stock libvosk emits the same final. The runtime's partial read `[sil]`
for the whole stretch with no `[sil]` entry under it, and a host that
took the text at its word saw silence turn into a word spanning twenty
seconds, `[[rr:TD-8#A silent stretch closes as libvosk closes it]]`.

The decoder is not at fault: it reads what libvosk reads,
`[[rr:TD-8#The decoder reads silence as libvosk does]]`, and the final
carries the energy that lets a host discard it,
`[[rr:TD-8#The gate is the host's and is relative to the floor]]`. The
label was.

## Considered options

- **Name the two states.** Taken. The phones are already on the path
  and `word_boundary.int` says which belong to words.
- **Print libvosk's empty string for the unlabelled case.** Reverses
  `[[rr:TD-2#Silence and unknown speech announce themselves]]`, and an
  empty string still says nothing about which state it is.
- **A flag beside the text.** Leaves `[sil]` in the text over a word's
  phones; every host that reads the text keeps the wrong reading.
- **Correct the README's gloss only.** True and useless: the host still
  cannot tell the two states apart.

## Decision outcome

A reading whose best path carries no word is `[speech]` when the path's
last phone belongs to a word by `word_boundary.int`, and `[sil]`
otherwise. The rule applies wherever a reading is named: the `partial`
text, each reading of `partial_alternatives`, the group trace, and the
text of a final and of its alternatives. A reading is still never the
empty string.

In a word list, the word phones after the last aligned word are one
`[speech]` entry, last in the list, with the same span, energy and hold
fields as a `[sil]` entry. Silence runs after the last word keep their
`[sil]` entries and precede it. A path with words lists words only in
its text, as before.

The rule names the path; it changes no cost, no rank and no final. A
host that removes bracketed tokens before comparing with libvosk sees
the same words.

## Consequences

- `[sil]` now means what the README said it meant. Its lead and the
  trailing `[sil]` entry's span keep their readings in
  `[[rr:TD-8#End of speech is the endpoint, and the trailing silence entry is its clock]]`.
  A `[speech]` entry after a trailing silence is the best path leaving
  silence, which is where the rules stop counting, as Kaldi's do on the
  same path. The `[sil]` entry stays, with the span it reached, so a
  host that reads the last `[sil]` entry reads what it read before;
  whether the `[speech]` entry is a word or the room is its energy
  against the floor, `[[rr:TD-8#The gate is the host's and is relative to the floor]]`.
  The runtime's floor on a stream built from clips that carry digital
  silence sits well below a gap's noise, so the states page reads the
  `[sil]` entry alone and its figures do not move.
- On a noise floor the acoustic model reads as speech, a stream shows
  `[speech]` where it showed `[sil]`, and a host's quiet-room test must
  read the floor, `[[rr:TD-8#The runtime reports the floor]]`, not the
  label alone. The benchmark pages that key the empty reading by its
  text fold the two labels into one reading and count the flips.
- The parity gates are untouched: G2 and G4 compare texts with
  bracketed tokens removed.

## Implemented by

- `[[rr:src/recognizer.rs#text_of]]` and `[[rr:is_word_phone]]`: the
  rule on a reading's text.
- `[[rr:entries]]` and `[[rr:EntryWord]]`: the entry.
