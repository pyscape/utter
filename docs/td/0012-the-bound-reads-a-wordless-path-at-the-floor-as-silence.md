# TD-12: The host's bound reads a wordless path at the floor as silence

- Tags: endpointing, silence, evidence, parity

## Context

On a quiet room the acoustic model reads the hiss as a word's first
phone within a few seconds, and the best path then carries no word and
ends inside one: the `[speech]` reading of
`[[rr:TD-10#Decision outcome]]`. The model's silence rules count
silence phones at the end of the path and see none, so nothing closes
the stretch until the length cap, whose final must complete a path and
so carries the cheapest grammar word, with a hold of zero and its
endpoint named by `[[rr:TD-11#Decision outcome]]`. The host's bound,
`[[rr:TD-8#A host may add one endpoint bound of its own, off by default]]`,
does no better: it counts the same silence phones, and where it carries
a veto, the readings that extend the wordless partial by a grammar word
sit four to eight nats behind it over hiss and veto every fire.

On the six public background recordings brought to a -50 dBFS floor,
6.7 minutes, the stock rules, a 300 ms bound and the same bound vetoed
at 8 nats all close the stretches the same way: 15 worded finals, 11 of
them forced by the length cap, a median 5 s per stretch.

The first consumer drops the length-cap final whose word never held,
`[[rr:5. What the application needs from this library]]`, and still
learns of the silence only when the cap fires, 20 s in. It asks that
the runtime close such a stretch at the bound with an empty word list,
never a manufactured word. `[[rr:TD-11#Considered options]]` already
refused to read a length-rule final over a wordless path as wordless:
that suppresses a word libvosk emits, against
`[[rr:TD-8#The decoder reads silence as libvosk does]]`, and the raw
path cannot tell hiss from a quiet word's first phone. The energy under
the path can, against the floor the runtime reports, and the margin
that reads it is the host's, `[[rr:TD-8#The gate is the host's and is relative to the floor]]`.

## Considered options

- **A runtime rule that closes a wordless path at the floor.** A
  threshold in the decoder that fires with the bound unset, which
  moves the silence finals away from the oracle. Rejected.
- **A host-side drop of the length-cap final.** What the first
  consumer does. It leaves the stretch open for 20 s and the two
  silence rules that fire on a floor-level path before the cap still
  hand the parser a word. Rejected as the only answer.
- **The bound reads the path against the floor with a margin the host
  names.** Off by default, so parity holds; on, the host has already
  chosen a bound and a margin, and the runtime applies them where the
  host would. Taken.

## Decision outcome

A host may give the bound a floor margin in dB,
`[[rr:set_endpoint_floor_margin]]`. With it set and a floor reported:

- The bound's trailing silence is the wordless span that closes the
  best path, contiguous back from the last decoded frame: its `[sil]`
  entries, and its `[speech]` entry when the mean energy under that
  entry is within the margin of the floor. A word, a gap, or a
  `[speech]` entry above the margin ends the span. The model's rules
  read the path as before.
- A reading that extends the partial by a word whose mean energy is
  within the margin of the floor does not veto the bound.
- A final the bound reaches only through the margin is the best path
  as it stood, not the best completed path, so it carries the words the
  partial showed and no other; its text is `[sil]` or `[speech]` when
  there were none, its word list is empty, and its `endpoint` is
  `floor`. A final the bound reaches on silence phones alone is
  unchanged and stays `bound`.

With the margin unset the recognizer is byte-identical to the one
without it.

## Consequences

- On the same six recordings at -50 dBFS, the bound at 300 ms vetoed at
  8 nats with an 8 dB margin closes the stretches at a median 0.72 s as
  wordless `floor` finals, and one length-cap final with a word remains
  in 6.7 minutes where there were 11. On 250 words spliced into gaps at
  that floor, 12 minutes, it decodes the same 240 words the stock rules
  do; the pages carry the figures on the full streams.
- A quiet word whose first phone sits within the margin of the floor
  for the length of the bound is cut where the hiss is: the price of
  the margin, measured on the pages as words lost, and the host's to
  set against the silence it buys.
- A floor-level word on the best path, as opposed to a floor-level
  `[speech]` entry, is not silence to the bound. It is the silence word
  of `[[rr:TD-8#Measurements count a word whose own span carries no speech]]`,
  reported with its energy for the host's gate, and its stretch closes
  as before. Under a grammar of twenty words the hiss reads as such a
  word; under the ninety-two of the public pages it reads as `[speech]`.
- The `endpoint` value set gains `floor`.

## Implemented by

- `[[rr:set_endpoint_floor_margin]]`
