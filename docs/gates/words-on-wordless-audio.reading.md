## Reading

Under the grammar a host actually ships, the two engines put a word at
rank 0 on wordless blocks at exactly the same rate, and the split of
those words is the one the unknown-word sweep already found by another
route: nine in ten still held from earlier in the utterance, the rest
just ended and straddling the block. This script labels its blocks from
the audio and computes each word's own energy from the WAV, sharing no
code with that sweep, so the agreement is a second measurement rather
than the same one repeated.

Nothing here is a word that was never spoken. The words that are neither
held nor straddling are real words the partial is still showing six to
nine blocks after their own span went quiet, and every one of them
followed speech in the same take: all of them with a word before them,
more than half within half a second of it. They do not sort by floor,
which spans three levels among so few, and they do not appear at all
until a word has been said. That is the case in question, and it is a stale
word rather than an invented one: the runtime reports it with its span
and its energy and withholds nothing (`[[rr:TD-8#A word on a quiet block
was spoken and is reported]]`), and the margin that removes it is the
host's.

The grammar rows are the surprise and they are not a size effect to act
on. A leading fraction of a grammar is not a smaller grammar a host
would ship: it is a vocabulary with whole classes of word missing, and
under one the decoder sits in a label instead of in the opening phones
of a word it cannot complete, so almost every wordless block carries a
word. Both engines do it, within two per cent of each other, so it is
the model's behaviour under a degenerate vocabulary and not this
runtime's. The figure worth carrying from those rows is only that the
count is a property of the vocabulary and not of the silence; the public
page sweeps grammar size properly, on a vocabulary that stays decidable
at every size.

The wordless final words move barely at all across the three rows, 57 to
73 of some 470, which says again that the finals and the partials are
two different mechanisms and only the finals need the gate.
