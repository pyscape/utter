# TD-17: Every result rates what the acoustic model hears and describes the sound outside words

- Tags: evidence, streaming, interface, front end

## Context

A host that gates commands wants two things no result carries,
`[[rr:5. What the application needs from this library]]`: how well the
acoustic model hears the audio just decoded, on a scale that holds
still, and what the sound outside the words is like, so a door, a hum
and a hiss can be told apart without a second detector.

Nothing today serves the first. A reading's `confidence` and a final
alternative's are the path's cost, a raw log-likelihood that ranks
readings against each other and moves with the grammar, the path's
length and the input level. A final word's `conf` is always `1.0`, for
want of a lattice. `energy_dbfs` and `floor_dbfs` are levels,
`[[rr:TD-8#The runtime reports the floor]]`: they say how loud, not
what kind of sound.

The pieces exist. The acoustic network gives, per 30 ms output frame, a
row with one score per pdf, which the decoder reads after multiplying
by the acoustic scale; its length is the model's pdf count. The front
end computes, per 10 ms feature frame, the log energy of each mel band
before the DCT turns them into cepstra; the stock small models have 40.

## Considered options

- **The path's cost, normalised by its length.** Still moves with the
  grammar, since the graph cost is in it, and with the path chosen.
  Rejected; the cost keeps its meaning and its keys.
- **The posterior of the path's own pdf per frame.** A rating of the
  path, not of the audio: two readings of the same frames would rate
  them differently. Rejected; the path only splits the frames.
- **One minus the entropy of each frame's output, normalised.** Depends
  on the network and the audio alone, is 0 for a flat output and 1 for
  a certain one, and is fixed per model. Taken.
- **A filter bank of its own for the sound outside words.** Finer
  bands cost a second transform per frame. Rejected for the front end's
  own mel bands, which are computed anyway.
- **Naming the sound: noise, tone, click.** Which sound a host cares
  about and where it draws the line is the host's, as silence is,
  `[[rr:TD-8#The gate is the host's and is relative to the floor]]`.
  Rejected; the runtime reports the measures.
- **Counts over the samples: zeros, clipping, peak.** The host holds
  the samples it fed. Not added.
- **Sound evidence over `[sil]` and a closed `[speech]` alone.** The
  need's own wording, but the decoder holds steady noise as an open
  `[speech]`: a pink-noise room at a -60 dBFS floor reads as one
  `[speech]` entry for fifteen seconds and more, and white or pink
  noise over it got no evidence at any level until a rule closed the
  utterance. Rejected; the evidence covers every frame in no named
  word.
- **A setting to turn it on.** The keys are cheap enough to be always
  on, and a host that would switch them on pays the same. No setting.

## Decision outcome

### The rating is the acoustic model's certainty

A frame's certainty is `1 - H / ln D`, where `H` is the entropy of the
softmax over the network's output row for the frame, the row the
decoder reads divided by the acoustic scale, and `D` is the row's
length, the model's pdf count. It lies in 0 to 1, clamped. It is
independent of the grammar, the path and the readings, and changes
with the input only as the network's output does: 20 dB quieter, the
test clips' certainty falls 9 to 15%. Between grammars it moves only
through the i-vector, whose statistics are weighted by the decode's
silence, by 0.0001 on the test clips. Chain models give flat outputs,
so on the stock small models it sits well under 0.1; only differences
between frames of the same model mean anything.

### Words and outside words

A path's entries split the frames. A frame is in words when it lies
inside a word entry, or inside a `[speech]` entry of a partial or a
reading, where the word is still open. Every other frame is outside
words: leading silence, `[sil]` entries, and on a final a `[speech]`
entry, which closed with no word. The certainty is split so.

The sound evidence covers every frame in no named word: the frames
outside words, and on a partial or a reading the open `[speech]`'s
too. A word's first phones, before the decoder names it, enter the
evidence of a window or two; noise the decoder holds as `[speech]` is
described while it lasts.

### The window is since the previous result

A result's top-level keys are over the frames decoded since the
previous result of any kind was read: partial or final, on the same
recognizer. A result read with no frame decoded since carries the keys
with null values and zero counts. A new utterance starts a new window.

### The sound outside words

Over a set of feature frames, the evidence is:

- each band's mean level in dB over the frames, less that band's floor;
- each band's standard deviation in dB across the frames, the
  steadiness: a steady noise or tone has a small one, an impulse a
  large one in every band it reaches;
- the loudest rise.

Levels are the front end's log mel energies, the natural log of power
before the DCT, converted to dB. A tone raises one or two low bands
over their floors; broadband noise raises all of them.

### Each band is measured against its own floor

Each band, and the broadband level, the dB of the bands' summed power,
has a floor: the mean level of each 50 ms hop of five feature frames,
ranked over the last 200 hops, 10 s, and taken at nearest rank
`n * 5 / 100` ascending, as the room's floor is ranked. A hop of
digital silence has no level and ranks below every hop that has one;
when a band's rank falls on one, there is no floor, as there is no
`floor_dbfs`. The floor runs on across utterances and pipeline
rebuilds, like `floor_dbfs`, and is null until the first hop has
closed.

### The loudest rise

The rise is the loudest frame by broadband level in the set, extended
both ways through contiguous frames of the set within 6 dB of it. It
reports its first frame's start in samples, its length in ms, and the
loudest frame's broadband level over the broadband floor. A click is
one or two frames long and far over the floor; a steady noise's rise
covers most of the set, at the noise's own level.

### Where the keys appear

Every partial and final carries, after every key it already has:

- `certainty_words` and `certainty_outside`, the mean certainty over
  the window's frames in words and outside them, three decimals, null
  with no frames;
- `words_frames` and `outside_frames`, those frames' counts;
- `band_db`, `band_sd_db`, `rise_start_sample`, `rise_ms` and
  `rise_db`, the sound over the window's frames in no named word, null
  with none, and `band_db` and `rise_db` null before a floor.

The split is by the object's own path: the partial's best path, each
reading's, each final alternative's, and the final's. The alternatives
shape carries the first alternative's at its top. The empty final,
`{"text": ""}`, carries none; the empty partial carries them as null.

### What a word entry carries

Every entry of every word list carries `certainty`, the mean over its
own span, whether or not the span lies in the window. `[sil]` and
`[speech]` entries of `partial_result` also carry the sound evidence
over their own span.

### Computed when a result is read

The network's rows for decoded frames are held until a result is read,
and the certainty of each is computed then, once, into a running sum
per utterance, so a span's mean is a difference of two sums. When 200
decoded frames, 6 s, wait with no result read, they are computed on the
block that decodes them, so the rows held stay bounded. The front end
keeps each feature frame's band levels until then; the track keeps
running sums of each band's dB and its square per frame of the
utterance, so a band's mean and deviation over a span are differences.
A `[sil]` or `[speech]` entry's keys are kept by span until the floor
moves or the utterance ends, so the partials between two advances
write them once. With these keys taken out, every result is byte for
byte the one the runtime gave before them.

### What it costs

On the compute gate's streams, 400 test clips in 40 streams, 571 s of
audio in 40 ms blocks, five repetitions on one core, against 0.0.5 and
the runtime before this record, each cell this record / 0.0.5 / before:

| Configuration | p99 block, ms | Slowest block, ms | Real-time factor |
|---|---|---|---|
| plain | 2.499 / 2.495 / 2.467 | 4.28 / 4.46 / 4.20 | 0.01036 / 0.01028 / 0.01020 |
| three readings | 2.570 / 2.587 / 2.537 | 4.42 / 4.71 / 4.40 | 0.01071 / 0.01070 / 0.01052 |
| three readings, ten alternatives | 2.618 / 2.612 / 2.574 | 4.60 / 4.88 / 4.50 | 0.01091 / 0.01084 / 0.01070 |
| margin 8 | 2.497 / 2.482 / 2.456 | 4.25 / 4.46 / 4.22 | 0.01036 / 0.01027 / 0.01017 |

The keys cost 6 to 8.5 us a block over the runtime before them. Against
0.0.5 the real-time factor is up to 0.9% higher and the 99th percentile
block up to 0.015 ms slower; the slowest block is faster in every
configuration. With the keys taken out, every result is byte for byte
0.0.5's, 14,334 results in each configuration.

### Verification

The unit tests check the exponential within two ulps, the certainty
against the entropy in f64 and the AVX2 sums against the portable sums
bit for bit, each floor against a plain sort of its band's hops, the
decimals against the formatter, and that a hiss, a hum and a click over
one advance's window are told apart and digital silence has no floor.
The model tests check that the windows tile each utterance and a second
read is empty, that a word entry's certainty does not depend on when
results are read, that the alternatives carry their first
alternative's rating at the top, that the certainty moves with the
input level and the grammar within their tolerances, and that noise
the decoder holds as an open `[speech]` carries the sound. The
benchmark page, docs/benchmarks/sound-rating.md, measures what the keys
tell apart on public audio.

## Consequences

- A partial with the sound evidence carries 80 numbers more at its
  top, and each `[sil]` or `[speech]` entry of `partial_result` 80
  more; results grow about fourfold on the benchmark streams.
- The certainty's value depends on the model: a threshold fitted on one
  model does not carry to another.
- Reading a result now changes what the next one covers. Two hosts
  reading at different moments see different windows over the same
  audio; per-entry certainty does not depend on when results are read.
