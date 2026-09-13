# Words on wordless audio, on the consumer's replay corpus

Ruling: `[[rr:TD-8#Measurements count a word whose own span carries no speech]]`. The private half of the public page's wordless pass, on the recordings the complaint came from.

## What ran

- Model: `vosk-model-small-en-us-0.15`; the first consumer's grammar, and the leading fractions of it.
- Run 2026-09-13 05:11:02Z, utter ec82d1c, vosk 0.3.45, utterpy 0.0.1, model vosk-model-small-en-us-0.15, 12th Gen Intel(R) Core(TM) i7-12700F, Python 3.14.4. Decoded by the binding /home/jared/Repos/utterpy/.venv/lib/python3.14/site-packages/utterpy/__init__.py built from utter ec82d1cffd8f9ca50cfc6901693d44d36376e97c, the checkout's HEAD.
- Corpus: the 2026-08-30 replay set, 33 takes, 28.6 minutes, of which 24.8 minutes are wordless blocks.
- 40 ms blocks, both engines fed identically, scored by `scripts/wordless_census.py`.
- A block is wordless by the audio alone: its last 400 ms within 8 dB of the floor at that moment, the floor computed host-side by the runtime's own definition so the stock wheel is judged by the same rule. Neither engine's reading enters the labelling, so both are scored on one set of blocks. The block counts still differ by a handful between the engines, which is the blocks each consumed as a final rather than a partial.

## The census

Rates are per wordless minute. **silence** is the bucket the complaint falls in: a word at rank 0 whose own span carries no speech, counted apart from a held word and a straddling one rather than as a phantom.

| grammar | engine | wordless blocks | rank 0 a word | rank 0 /min | rival word /min | held | straddling | silence | silence /min | final words | silence final words | longest silence run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 of it | vosk | 37099 | 30306 | 1220.47 | - | - | - | - | - | 488 | 73 | - |
| 0.25 of it | utterpy | 37094 | 29744 | 1197.84 | 1197.8 | 827 | 965 | 27952 | 1125.67 | 480 | 69 | 486 |
| 0.5 of it | vosk | 37087 | 28736 | 1157.25 | - | - | - | - | - | 470 | 60 | - |
| 0.5 of it | utterpy | 37085 | 28507 | 1148.03 | 1148.0 | 921 | 997 | 26589 | 1070.78 | 468 | 60 | 480 |
| all of it | vosk | 37098 | 1419 | 57.15 | - | - | - | - | - | 471 | 57 | - |
| all of it | utterpy | 37092 | 1419 | 57.15 | 57.1 | 938 | 458 | 23 | 0.93 | 469 | 58 | 9 |

A dash means the engine showed a different partial when asked for the words under it, so that pass is not a split of the count beside it: the stock wheel's partial-word path is a lattice one that trails the audio.

Paired over the same takes, a take carrying any word at rank 0 on a wordless block:
- the grammar, 0.25 of it: vosk only 0, utterpy only 0, exact two-sided p = 1.
- the grammar, 0.5 of it: vosk only 0, utterpy only 0, exact two-sided p = 1.
- the grammar, all of it: vosk only 0, utterpy only 0, exact two-sided p = 1.

## What a silence word goes with

Every silence word at rank 0 under the full grammar, split by the take property it fell in.

| utterpy: floor | silence words |
|---|---|
| -40 dBFS | 11 |
| -50 dBFS | 6 |
| -30 dBFS | 6 |

| utterpy: a word came first | silence words |
|---|---|
| yes | 23 |

| utterpy: since speech | silence words |
|---|---|
| under 0.5 s | 13 |
| under 5 s | 6 |
| under 1 s | 4 |

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
until a word has been said. That is the owner's case, and it is a stale
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

