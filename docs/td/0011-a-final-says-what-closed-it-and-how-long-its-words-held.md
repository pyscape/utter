# TD-11: A final says what closed it and how long its words held

- Tags: finals, endpointing, evidence, parity

## Context

`[[rr:TD-10#Decision outcome]]` names the partial's state over a
stretch of room noise the acoustic model reads as a word's first phone.
The final that ends that stretch is untouched: at the 20 s length rule
the only complete paths are grammar words, the decoder picks one, and
libvosk picks the same one. On the first consumer's microphone that is
a worded final about every 21 s of silence, each carrying a word the
player did not say, and it reaches the parser as any other final does.
The first consumer asks for that final to be tellable at the final, not
only by a floor gate, `[[rr:5. What the application needs from this library]]`.

The energy under the word already sits at the floor,
`[[rr:TD-8#A final word carries its energy]]`, and a host's margin
removes every such final on the gate corpus. It is a threshold. Two
facts the decoder holds are not: no partial ever showed the word, and
the rule that closed the utterance was the length rule.

## Considered options

- **Report the two facts.** Taken.
- **Read a length-rule final over a wordless raw path as wordless.**
  Suppresses a word libvosk emits, which
  `[[rr:TD-8#The decoder reads silence as libvosk does]]` rules out,
  and drops a real word spoken just before the length rule fires whose
  label the composition had not yet placed. The raw path cannot tell
  those apart; the hold can.
- **A boolean flag for the floor-level word.** A threshold in the
  runtime, which `[[rr:TD-8#The gate is the host's and is relative to the floor]]`
  keeps out.

## Decision outcome

Every final word carries `stable_ms`: how long it had held its place
in the partial when the final was cut, read off the same list a partial
word's hold is read off, by position, and zero where the list held
something else. A word a final carries that no partial showed has a
hold of zero.

Every final carries `endpoint`: what closed the utterance. `rule1` to
`rule5` name the model's rules in the order libvosk tests them and the
first that fired; `bound` is the host's rule of
`[[rr:TD-8#A host may add one endpoint bound of its own, off by default]]`;
`flush` is `final_result`; `host` is `result` called with no rule
fired.

Both keys follow the keys already emitted, after `text` and before
`floor_dbfs`, and on each word after `energy_dbfs`. No cost, rank,
word or endpoint moves.

## Consequences

- A host's settle rule, act on a word once it has held, applies to
  finals as it does to partials, instead of a final bypassing it. A
  length-rule final over the floor carries a word with a hold of zero;
  a spoken word closed by a silence rule carries the hundreds of
  milliseconds it stood in the partial.
- A real word spoken just before the length rule fires, whose label
  was not yet placed, also carries a hold of zero. The host treats it
  as it treats a partial word it has not yet seen hold, which is the
  rule it already has.
- The states page counts the worded finals none of whose words had
  held, and the worded finals by the rule that closed them.

## Implemented by

- `[[rr:endpoint_reason]]` and `[[rr:Endpoint]]`: the rule that fired.
- `[[rr:holds]]` and `[[rr:write_final_words]]`: the hold on each
  final word.
