# Decoder diagnostics, vosk-model-small-en-us-0.15, testing split, 11005 clips, 40 ms blocks

Grammar: the dataset's 35 words plus 26 letters, 26 NATO words and 5 colours, 92 entries. 10 readings asked for on the partial and on the final.

Run 2026-09-24 07:22:16Z, utter 765482d, vosk 0.3.45, utterpy 0.0.5, model vosk-model-small-en-us-0.15, 12th Gen Intel(R) Core(TM) i7-12700F, Python 3.14.4. Decoded by the binding `/home/jared/Repos/utter-bench-data/site-765482d/utterpy/__init__.py` built from utter 765482d7b70a6d3f5c4071e56e280b7d5df7b598, the checkout's HEAD. Pages under `docs` were modified at the time of the run.

## Engines

Each engine opened the model and decoded a clip before the run began; a model an engine cannot decode stops the run here rather than part-way through a pass.

| engine | model load, s | first recognizer, ms |
|---|---|---|
| vosk | 0.17 | 0.4 |
| utterpy | 0.14 | 21.3 |

## Oracle in the beam: was the word there at all

At the first partial that carried a word, and in the final, with n-best asked for on both. A reading counts at the sighting when the word leads it, because a partial is read for the word at its front; in the final it counts when the whole reading is the word, which is what the accuracy pass calls correct. An engine whose partial carries one reading has no rate above n = 1: readings on a partial are utter's, by `[[rr:TD-2#The decoder: partial alternatives]]`.

### The word among the readings at the first sighting

| engine | n = 1 | n = 5 | n = 10 | rank 0 | rank 1 | rank 2 and below | median rank |
|---|---|---|---|---|---|---|---|
| vosk | 7921 / 11005 (72.0%) | - | - | 7921 | - | - | - |
| utterpy | 7936 / 11005 (72.1%) | 8978 / 11005 (81.6%) | 9117 / 11005 (82.8%) | 7936 | 539 | 642 | 0 |

| n | vosk | utterpy | only vosk | only utterpy | McNemar p |
|---|---|---|---|---|---|
| 1 | 7921 | 7936 | 41 | 56 | 0.155 |

Paired over the 11005 clips both engines read, with an exact McNemar test on the clips they differ about. Only n = 1 is paired here, because vosk offers one reading on a partial.

### The word among the readings in the final

| engine | n = 1 | n = 5 | n = 10 | rank 0 | rank 1 | rank 2 and below | median rank |
|---|---|---|---|---|---|---|---|
| vosk | 10071 / 11005 (91.5%) | 10328 / 11005 (93.8%) | 10329 / 11005 (93.9%) | 10071 | 211 | 47 | 0 |
| utterpy | 10102 / 11005 (91.8%) | 10526 / 11005 (95.6%) | 10558 / 11005 (95.9%) | 10102 | 292 | 164 | 0 |

| n | vosk | utterpy | only vosk | only utterpy | McNemar p |
|---|---|---|---|---|---|
| 1 | 10071 | 10102 | 18 | 49 | 0.000194 |
| 5 | 10328 | 10526 | 7 | 205 | 1.09e-51 |
| 10 | 10329 | 10558 | 7 | 236 | 1.33e-60 |

Paired over the 11005 clips both engines read, with an exact McNemar test on the clips they differ about.

### The n-best did not move the answer

Rank 0 at the stock search against the finals `speech-commands.clips.jsonl` carries, over the clips in both.

| engine | rank 0 equals the recorded final |
|---|---|
| vosk | 10984 / 11005 (99.81%) |
| utterpy | 11005 / 11005 (100.00%) |

## Beam sensitivity: the clips a wider search moves

The same clips through a sibling model directory: `am/`, `graph/` and `ivector/` are symlinks to the stock model, `conf/` is a copy with two lines changed. The model's own `conf/model.conf` carries `beam` 10.0 and `max-active` 3000; the sibling asks for 18.0 and 20000 instead. Nothing under `src/` differs between the two columns.

- vosk stock reports at model load: `beam=10 max-active=3000 lattice-beam=2`
- vosk wide reports at model load: `beam=18 max-active=20000 lattice-beam=2`

| engine | final changed | became right | became wrong | wrong to wrong | accuracy stock | accuracy wide |
|---|---|---|---|---|---|---|
| vosk | 16 / 11005 (0.15%) | 7 | 5 | 4 | 10071 / 11005 (91.51%) | 10073 / 11005 (91.53%) |
| utterpy | 0 / 11005 (0.00%) | 0 | 0 | 0 | 10102 / 11005 (91.79%) | 10102 / 11005 (91.79%) |

A changed final is the search reaching a different answer, not the search having erred before.

### What the wider search costs

| engine | RTF stock | RTF wide | per-block ms p50 stock / wide | p95 stock / wide |
|---|---|---|---|---|
| vosk | 0.0122 | 0.0152 | 0.048 / 0.053 | 2.81 / 3.57 |
| utterpy | 0.0103 | 0.0118 | 0.042 / 0.043 | 2.35 / 2.73 |

One recognizer over 295 s of clips joined end to end at each setting, the same pass the Speech Commands page reports compute with.

### The finals over a range of beams

Every beam over the same 800 clips, each in its own sibling directory at `max-active` 20000, counted against the stock search.

| engine | beam 0.5 | beam 2 | beam 5 | beam 13 | beam 18 |
|---|---|---|---|---|---|
| vosk | 1 / 800 | 1 / 800 | 1 / 800 | 0 / 800 | 0 / 800 |
| utterpy | 0 / 800 | 0 / 800 | 0 / 800 | 0 / 800 | 0 / 800 |

### The sibling directory is read

A sibling built the same way but with `acoustic-scale` 1.0 to 0.1, over 200 clips. The finals have to move; an engine whose finals stood would be one not reading the file, and the beam figures above would then be measuring nothing.

| engine | finals changed against stock |
|---|---|
| vosk | 193 / 200 (96%) |
| utterpy | 192 / 200 (96%) |

### The word among the readings at the wider search

| engine | sighting, n = 10 | final, n = 1 | final, n = 10 |
|---|---|---|---|
| vosk | - | 10073 / 11005 (91.5%) | 10331 / 11005 (93.9%) |
| utterpy | 9117 / 11005 (82.8%) | 10102 / 11005 (91.8%) | 10559 / 11005 (95.9%) |

## Where the replacing word was when the first word appeared

Every first word that was later revised, from `partial-trust.clips.jsonl`: 1147 of 9230 sightings (12.4%). No audio is decoded. On 436 of them (38.0%) the sighting was the clip's last advance, so the word was replaced by the final with no partial in between and the series holds nothing to place it by. The other 711 split as follows, the replacing word being the word leading the clip's last advance and the buckets tried in order.

| where the replacing word was | revisions | share of the 711 | it was the clip's own word |
|---|---|---|---|
| the next advance settled it | 684 | 96.2% | 619 / 684 (90%) |
| present at the sighting, ranked below | 4 | 0.6% | 4 / 4 (100%) |
| absent at the sighting, arrived later | 14 | 2.0% | 14 / 14 (100%) |
| no word led the last advance | 9 | 1.3% | 0 / 9 (0%) |

The last advance's leader and the emitted final agree about the first word on 8771 of 9230 sightings (95.03%); where they do not, a bucket is about the last partial a host saw and not about the final.

## Reading

Every figure on this page is an observation about where a word was, and none of
them names a cause. The three measurements are read as follows.

**The word among the readings at the sighting** is whether the clip's own word
led any reading the beam held at the first partial that carried a word. It does
not say the word was decodable then: the sighting is early in the utterance and
the audio that settles the word may not have been fed yet, which is what the
revision split below is about. utter's partial carries a ranked n-best of the
beam's surviving groups, `[[rr:TD-2#The decoder: partial alternatives]]`;
the stock wheel measured here exposes no such call, so its partial is one
reading, it has no rate above n = 1, and the two engines are comparable at
n = 1 only. Absence at the sighting is not evidence of
pruning, and presence below rank 0 is not evidence of a mis-scored cost; both are
the position, not the reason.

**The word among the readings in the final** at n = 1 is the accuracy the
Speech Commands page reports, arrived at the same way. Above n = 1 it is the
depth of the n-best, which the two engines reach differently: utter's is the
beam's own n-best and libvosk's is read off a lattice whose width is
`lattice-beam`, a setting this page leaves at the model's stock value in both
columns. A word at rank 3 is not a word a host could have had: nothing here says
which of the n to choose, and no figure on this page is a selection rule.

**Neither oracle rate is a ceiling.** The rate at the wider search is what this
search kept under this grammar at this beam, and reading it as the limit of the
acoustic model would be reading a search result as a model property. It bounds
nothing.

**The clips a wider search moves** is search sensitivity. A final that changes
when the beam is widened tells you the answer was not settled by a margin the
stock beam covers; it does not establish that the stock search erred, and a clip
that became right is not a clip the stock search should have got. The split into
became-right, became-wrong and wrong-to-wrong is there because a wider search
moves clips in both directions, and a net accuracy difference hides that.

**The range of beams** is there because one wider setting cannot tell an
insensitive final from a lucky pair of settings. It is counted on a sample of
the split rather than all of it, and it counts finals that differ, not finals
that are wrong. A row of zeros says the final under this grammar does not depend
on the beam over that range; it says nothing about a larger grammar, about
another model, or about the partials along the way, which are not compared here.

**That the sibling directory is read** is established by a third sibling, not
assumed from the beam figures: libvosk prints the beam and max-active it parsed
as it opens a model, which the page quotes, and both engines are then shown to
return different finals when a key with an undoubted effect is changed. Without
that, a row of zeros would be indistinguishable from a configuration file that
was never opened.

**The compute column** is one steady-state pass per setting on the same joined
audio, the pass the Speech Commands page uses, so the two columns are comparable
with each other and with that page. It is a single run on one machine, not a
minimum over repeats, so read the p50 and treat small differences in the tails as
noise.

**The revision split** is where the word that replaced a first word stood when
that first word appeared, read off a recorded series. It decodes nothing, so it
inherits that series' terms: a record's word is the first word a host was shown,
and the replacing word is taken as the word leading the clip's last advance,
which is the last partial a host saw and not always the final the engine emitted
(the page gives the rate at which those two agree). Revisions whose sighting was
also the clip's last advance are set aside rather than bucketed: the series holds
no later reading to place the replacing word by.

Of the buckets, one has a lever already named elsewhere and one has a lever with
a price. A replacing word that stood among the readings at the sighting but below
rank 0 is a question about cost, where the graph scale and the bigram cost sit,
and `[[rr:TD-6]]` settled that the runtime reproduces libvosk's choices there
rather than improving on them, so the lever exists and parity is what it costs.
A revision the next advance settles is the audio arriving at the decoding
cadence, and the only thing that moves the cadence is the chunk
(`[[rr:TD-2#Inputs: configuration]]`), which is read from `conf/model.conf` and
is part of the byte identity the runtime is gated on. The remaining buckets have
no lever named on this page and none is guessed for them.
