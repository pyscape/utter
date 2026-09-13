# Word times on finals, Speech Commands v2, vosk-model-small-en-us-0.15, testing split, 11005 clips, 40 ms blocks

Grammar: the Speech Commands page's 92 entries. Both engines with `SetWords(True)`; each word of each final with its `start` and `end` in seconds. Times are paired on the clips where both finals carry the same word sequence; utterpy is measured against vosk.

Run 2026-09-13 17:57:01Z, utter 608d4f7, vosk 0.3.45, utterpy 0.0.1, model vosk-model-small-en-us-0.15, 12th Gen Intel(R) Core(TM) i7-12700F, Python 3.14.4. Decoded by the binding `/home/jared/Repos/utterpy/.venv/lib/python3.14/site-packages/utterpy/__init__.py` built from utter fafb7f5e4d6bab8dea51c2ad7dd578ebd14ffbd4, **not** the checkout as it stands (608d4f7).

## Clips

| clips | same word sequence | sequences differ | both empty | words paired |
|---|---|---|---|---|
| 11005 | 10505 | 93 | 407 | 10521 |

## Frame grid

A word time that is not a multiple of 30 ms is not one of Kaldi's output frame times.

| engine | words | off the 30 ms grid |
|---|---|---|
| vosk | 10612 | 25 |
| utterpy | 10614 | 0 |

## Agreement

Each paired word's utterpy time against its vosk time. A word counts as within a frame when both its start and its end are.

| time | exact | within one frame (30 ms) | within two frames | beyond |
|---|---|---|---|---|
| start | 10287 (97.78%) | 10510 (99.90%) | 10515 (99.94%) | 6 (0.06%) |
| end | 9338 (88.76%) | 10401 (98.86%) | 10473 (99.54%) | 48 (0.46%) |
| word | 9132 (86.80%) | 10390 (98.75%) | 10467 (99.49%) | 54 (0.51%) |

| time | abs difference p50 / p90 / p99 / max, ms | utterpy earlier | utterpy later |
|---|---|---|---|
| start | 0 / 0 / 30 / 510 | 119 | 115 |
| end | 0 / 30 / 60 / 330 | 563 | 620 |

## Reading

The word times on finals are the wheel's to within one output frame.
Of 10,521 words paired on the 10,505 clips whose finals agree, the
start is exact on 97.8% and within one 30 ms frame on 99.9%; the end
is exact on 88.8% and within a frame on 98.9%. Taking a word as
matched only when both its start and its end are within a frame, 98.8%
match and 99.5% are within two frames.

The 54 words beyond two frames are, on 53 of them, single-word
finals. On 44 the start agrees and the end differs by three frames or
more, up to 330 ms: where the word's tail stops and the trailing
silence begins. On the other ten the start moves too, once by 510 ms.
Over all paired words the runtime puts the end later on 620 and earlier
on 563, so there is no bias, only a boundary the two searches settle
differently on a word in two hundred.

Every time the runtime reports is a multiple of 30 ms. The wheel
reports 25 that are not, on 17 two-word finals, each the boundary
between the two words, which its lattice alignment places between
frames. Ten of those finals are paired here; on them the runtime's
boundary is the nearest frame.

The claim on the Vosk reference page is stated from this table: word
times match the wheel's to within one frame, on the public split, on
the finals the two engines agree about.
