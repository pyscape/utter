# TD-14: Every word carries the speaker evidence of its own span

- Tags: speaker, evidence, streaming, interface

## Context

The first consumer gates what a command does on who spoke it, and the
gate decides word by word as the partial forms: a word that changes
state acts only when an enrolled player said it, and fails closed when
there is no evidence, `[[rr:5. What the application needs from this library]]`.
Until now the runtime left speaker vectors to the stock wheel,
`[[rr:TD-2#Scope]]`, and the consumer ran them on a second recognizer:
vosk's `SpkModel` beside a fresh `KaldiRecognizer` per clip, one
`AcceptWaveform` over the clip, the vector read off the final.

That path fits neither the gate nor this runtime. libvosk computes the
vector only for a final, in `GetSpkVector`: it keeps the frames its
best path puts on non-silence phones, only within the last 500 decoder
frames, refuses fewer than 50, normalises the kept frames by a mean
over 300 of them centred on each, runs the network over them joined
end to end, and centres, whitens and rescales the output. A partial has
no vector, and a final's covers the utterance, so a word said by a
second speaker inherits the first speaker's frames.

The model the consumer ships, `vosk-model-spk-0.4`, is Kaldi's
callhome diarization x-vector: 23 MFCCs at 8 kHz up to 3.7 kHz, five
frame-level TDNN layers seeing seven frames either side, mean and
standard-deviation pooling, and a 128-dimensional embedding.
Everything before the pooling is local in time and the pooling is a
sum, so the network can compute each frame's row once, when a span
first needs it, and any span's vector follows from the rows of its
frames.

## Considered options

- **A stateless call over a whole clip, as the wheel was used.** The
  evidence arrives after the words were acted on and covers the clip,
  not the word decided. Rejected; enrolment reads the same vector off a
  final instead.
- **Utterance evidence on partials alone.** Early, but a second
  speaker's word inherits the first speaker's frames. Kept beside the
  per-word evidence, not instead of it.
- **Scores against profiles registered with the recognizer.** Smaller
  results, but who is enrolled, how a profile is built and how a score
  is read are the host's, as silence is,
  `[[rr:TD-8#The gate is the host's and is relative to the floor]]`.
  Rejected: the runtime reports the vector.
- **Kaldi's centred normalisation, recomputed over each span.** The
  model was trained on frames normalised over a window centred on each,
  so a frame's features would change as later audio arrives and every
  partial would rerun the network over the utterance. Rejected for the
  live path; the benchmark page runs it as the offline reference.
- **A 16 kHz model such as TitaNet or CAM++.** Their encoders average
  over the whole input inside every block, so each partial reruns the
  network from the start, and they ship as ONNX, which this crate does
  not read. Not taken; the benchmark page measures them beside the
  x-vector on the same words.

## Decision outcome

A host gives a recognizer a speaker model, `[[rr:set_spk_model]]`,
opened from a directory in the layout vosk's `SpkModel` reads:
`mfcc.conf`, `final.ext.raw`, `mean.vec` and `transform.mat`. The
network must end in an affine over the mean and standard deviation of
one frame-level node; any other is refused when opened. From then on:

### Evidence is pooled over a span

- Every entry of every word list that is a word or `[speech]` carries
  the evidence of its own span: `partial_result`, each reading's
  `result`, the final's `result`, and each final alternative's
  `result`. A `[sil]` entry carries none.
- Every partial and final carries at its top level the evidence over
  its best path's speech: the words and `[speech]` of a partial, the
  words of a final, of the first alternative in the alternatives
  shape.
- Evidence is four keys. `spk` is the vector; `spk_frames` the 10 ms
  frames pooled; `spk_start` and `spk_end` the first pooled frame's
  start and the last one's end, in samples fed since the recognizer
  was built, the clock of `start_sample`. On the plain final `spk` and
  `spk_frames` stand before `text`, where libvosk puts them; everywhere
  else the four follow the keys already there.
- The vector is the embedding affine over the mean and standard
  deviation of the rows pooled, centred by `mean.vec`, whitened by
  `transform.mat` and scaled to a norm of the square root of its
  length, as libvosk leaves its own. Cosine is the natural score; the
  vector's scale is not a probability.
- With a floor margin set, `[[rr:set_endpoint_floor_margin]]`, a frame
  whose own energy is within the margin of the floor is not pooled.
- No key names a speaker. The runtime reports evidence; the host
  decides.

### The front end is the model's own

The features are the model's `mfcc.conf` at the model's rate. Faster
audio is brought down by Kaldi's online `LinearResample`, its cutoff at
half the lower rate and six zero crossings, as libvosk's speaker
features are; slower audio is refused. The first coefficient is the
frame's log energy where the configuration keeps Kaldi's default.
Dither is off, so the same audio in the same blocks gives the same
vectors. The model's frames must be the decoder's 10 ms, so a speaker
frame and a decoder frame cover the same samples.

### Mean normalisation looks back only

Each frame is normalised by the mean of itself and the 299 frames
before it, over all audio since the stream began, silence included,
and across the finals of one stream. Kaldi's recipe normalises over 300
frames centred on each frame, before selecting speech; here a frame
never waits for audio after it, so a frame's row is computed at most
once and a word's rows never change. The price is paid in the first three
seconds of a stream, normalised over fewer frames, and in a mean that
carries the previous speaker's frames into the next speaker's first
words. The benchmark page measures both against the centred reference.

A frame's row exists once the network's right context has arrived,
seven frames later, so a span's last 70 ms join its evidence on a later
partial; `final_result` flushes the stream and pools every frame.

A row is computed when a span needs it, or earlier when a span is
likely to. In the block before the decoder next advances, found by
taking the next block to be the size of this one, the rows the advance
is likely to pool are computed: while the best path or a reading ends
in speech, every frame from where that speech began to the last the
advance will decode, and otherwise the frames more than 20 dB over the
floor, or over -40 dBFS before there is a floor, with 16 frames either
side. The readings are the host's; when it asks for none, the
look-ahead reads up to three of its own, those within 10 of the best's
cost. Between advances, while the best path ends in speech, its rows
are computed once 20 of them can be. When the decoder advances, the
rows of its best path's words and `[speech]` not yet computed are
computed once those spans could pool 25 frames; a reading, an
alternative or a final that pools a frame not yet computed computes it
then, and the network starts afresh past a gap of more than eight
frames. Silence the look-ahead does not take for speech and no span
covers never runs the network. The model's features are computed only
when rows or a result need them, over the audio since they last were.
A row does not depend on which frames are computed with it or when, so
every result is byte for byte the one computing every frame as it
arrives gives,
`[[rr:rows_on_demand_give_every_result_as_rows_computed_eagerly]]`,
`[[rr:rows_ahead_of_the_chunk_give_every_result_as_rows_computed_eagerly]]`.

Over 400 Speech Commands test clips in streams of ten with half-second
gaps, results read after every 40 ms block, the fastest of five runs
per block put the 99th percentile block at 2.97, 3.03 and 3.05 ms with
the speaker model set, for no readings, three readings, and three
readings with ten final alternatives; the real-time factor at 0.0175,
0.0176 and 0.0176; the median at 0.057, 0.060 and 0.060 ms. When every
frame ran as audio arrived these were 3.29, 3.42 and 3.46 ms, 0.0190,
0.0197 and 0.0199, and 0.308, 0.345 and 0.346 ms; without the speaker
model 2.66, 2.63 and 2.64 ms, 0.0108 and 0.0109, and 0.052 to 0.055
ms. The slowest block, a final that flushes the stream, ran from
0.5 ms under to 0.2 ms over its time when every frame ran, across runs.
Rows computed only when needed put the 99th percentile at 4.19 ms, the
network running on the decoder's own block; 16 frames either side took
it to 2.88 ms where 5 took it to 3.24 and 10 to 2.96, at the same
real-time factor. 20 dB and -40 dBFS were chosen among 3 to 25 dB and
-40 to -30 dBFS in single runs, where the 99th percentile moved less
than it does between runs. With readings asked for, following the best
path alone left the words readings put on quiet audio to the advance's
block, 23 rows on its 99th percentile blocks, which stood at 4.23 ms
with three readings; following the readings brought it to 3.03, and
with the keeping up below the real-time factor rose by 0.0020. Without
readings asked for, the look-ahead's own within 10 of the best's cost
brought the slowest block other than a final from 6.1 to 4.4 ms, for
0.0015; within 5 left it at 6.1. Keeping up with the best path's
speech in batches of 20 took the slowest final from 6.24 to 5.27 ms;
batches of 12 took it to 5.50 for 0.002 ms more median, and of 8 put
the median at 0.24 ms. The speaker page's latency pass,
`[[rr:latency_pass]]`, measures the cost on its game streams.

### A span needs a quarter second

A span with fewer than 25 pooled frames carries no evidence: its keys
are absent, and absent keys are no evidence. libvosk's floor of 50
frames is not kept, because a command word is 300 to 600 ms of speech
and would mostly carry none. What a span of a given length is worth is
measured on the benchmark page by word length.

### Pooling reads the frames it keeps

The rows of the current utterance are kept, up to 3,000 frames, 30 s;
a span reaching further back pools only what is kept. Rows of earlier
utterances are dropped at each final. A span pools every frame it
covers: the model's own extraction graph pools at most 401 frames,
which only a span over four seconds would reach, and its vector
differs from libvosk's there.

### Verification

`scripts/speaker_oracle.py` checks the stages against Kaldi on 70
public clips, the loudest and the quietest test clip of every Speech
Commands word. The model's MFCC through the online resampler agree with
`online2-wav-dump-features` within 4.6e-3, and the energy within 7e-6:
the decoder's own MFCC show the same take-dependent bound. The vector
over the runtime's normalised features agrees with
`nnet3-xvector-compute` without padding within 1.6e-3, cosine at least
0.9999998. The model tests check that a word's evidence does not
depend on the block size, or on when the speaker model was set. The
benchmark page measures what the evidence identifies.

## Consequences

- Enrolment is a recognizer over the enrolment audio: one vector per
  final, and the host's mean over them. Enrolment recordings serve as
  they are; thresholds fitted on the wheel's vectors do not carry over,
  because the normalisation and the frames pooled differ.
- A result grows by about 1.3 KB of text per word that carries
  evidence. The network's rows cost compute on speech frames; the
  speaker page's latency section gives the cost per block and in all.
- `[[rr:TD-2#Scope]]` no longer excludes speaker vectors; this record
  is where they are specified.
- A model of another architecture needs its own record: this one fixes
  the pooling the runtime computes.
