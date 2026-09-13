# Gate results

One file per gate, named by the gate and what it measures. A file records what was
run, against which oracle, and the figures; the gates themselves are
defined in `[[rr:TD-2#Verification and acceptance]]`. Corpora and models
live in the first consumer's private checkout and are named here only by
date and by the stock model's directory name; no take identifiers, audio
or transcripts are copied in.

`unknown-word-sweep.md` is not a gate but the measurement behind the
`[unk]` option's default.

`words-on-wordless-audio.md` is not a gate either but the private half of
the Speech Commands page's wordless pass: how often either engine puts a
word at rank 0, among the rivals or in a final on a block the audio says
carries no speech, over the replay corpus, by grammar and by the take
property a word that is neither held nor straddling goes with.
