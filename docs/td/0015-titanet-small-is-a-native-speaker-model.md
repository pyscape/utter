# TD-15: TitaNet-small is a native speaker model that embeds each word's span whole

- Tags: speaker, evidence, streaming, model-format, interface

## Context

Every word carries the evidence of its own span,
`[[rr:TD-14#Evidence is pooled over a span]]`, and the only model that
supplies it is the callhome x-vector. On the speaker page a gate at 1%
false accept passes 40.1% of an enrolled speaker's words on that
evidence; NVIDIA's TitaNet-small, embedding the span the wheel reports
for each word, passes 84.9%, and 64.9% of words under 400 ms against
29.2%.

TD-14 turned TitaNet down for two reasons,
`[[rr:TD-14#Considered options]]`, and both still hold. Its
squeeze-excitation layers and its attentive pooling take means over the
whole input, so no row of it is computed once per frame and pooled
later: a span's embedding is a run of the whole network over that span
alone. And NeMo ships it as ONNX, which the crate does not read and,
under `[[rr:TD-2#Dependency policy]]`, will not.

What makes it fit a streaming recognizer anyway is the network's shape.
It runs in stages: the first block, three blocks of three separable
convolutions with residuals, the last block, and the pooling. The means
over the span are needed only at a stage's own boundary, and a sum over
time can be accumulated a frame range at a time in frame order. A
matrix product's rows are independent and their accumulation order is
fixed, `[[rr:TD-3#Accumulation order is part of the contract]]`. So one
embedding can be cut into slices of any size between the recognizer's
decoder advances, and the slices give the bytes of one call.

Over the speaker page's streams a word's span runs 510 ms at the median
of finals and 870 ms at their 99th percentile. The longest real word is
1,050 ms. Longer spans all belong to words opening on /h/ or /w/ whose
span reaches back over the background noise before them, where floor
audio reads as the opening phone, `[[rr:TD-10#Decision outcome]]`: finals
to 3.6 s and readings to 6.1 s.

## Considered options

- **Read the ONNX file in the crate.** An ONNX reader is a protobuf
  parser and an operator set the crate would own for one model.
  Rejected for a converter script beside the crate, run once.
- **NeMo's PyTorch preprocessing.** The 84.9% was measured through
  sherpa-onnx, whose front end for NeMo models is kaldi-native-fbank's.
  Rejected for that one.
- **Pad the span to a multiple of 16 frames.** The ONNX graph masks
  padded frames, and sherpa-onnx resizes its feature buffer to a
  multiple of 16. But the tensor sherpa-onnx gives onnxruntime has the
  span's own frame count, so it never padded, and the 84.9% is an
  exact-length score. Padded, the same protocol passes 83.6%, 63.1%
  under 400 ms, and exact length is ahead in every policy and length
  bin but one. Rejected.
- **Fold the pooling batchnorms, run in f16 or int8, or prune.** Each
  changes the vector the accuracy was measured on. Rejected.
- **A second thread in the crate.** Evidence would arrive early and
  cost the recognizer's thread nothing, but when it arrives would depend
  on the machine, and the crate owns no thread. Rejected.
- **TitaNet-large, or any size but small.** Large passes fewer of the
  page's words, 82.0%, at 3.4 times the cost. Rejected, as is CAM++.
- **Embed a word on the block that closes it.** 2.2 to 4.1 ms more on
  that block, which puts the 99th percentile at 6 to 9 ms. Rejected.
- **Embed at the final.** 10 to 20 ms on the final's block, and evidence
  about 1.4 s after the word. Rejected.
- **Embed again at every change of a span.** Earliest evidence, about
  0.0245 more real-time factor through sherpa-onnx. Rejected.
- **Only a stateless call, the host scheduling it.** Costs the crate
  nothing and fits hosts that embed audio the recognizer never saw.
  Taken beside the recognizer's own schedule, not instead of it.

## Decision outcome

A host gives a recognizer TitaNet-small through the same speaker model
call as the x-vector, `[[rr:set_spk_model]]`. What TD-14 fixes about
which entries carry evidence and how the four keys are written holds,
except where the rulings below say otherwise.

### The model directory is converted once

The ONNX export is converted by a script beside the crate into a
directory of two files: titanet.conf, key=value lines naming the
architecture, the front end, the embedding's dimension, the source
file's name and SHA-256, the parameter count and the count of weights
zeroed; and weights.bin, a magic string, a version, a table of tensor
name, dimensions and offset, then 64-byte-aligned little-endian f32
data. Each initializer is named by its NeMo module path, not by its
place in the graph; the topology is checked; everything after the
embedding layer, the classifier, is dropped; weights with magnitude
below 1e-15 are zeroed. The encoder's batchnorms arrive folded into its
convolutions by the export; the two pooling batchnorms are kept and run
as ONNX defines them.

Opening a speaker model tells the two layouts apart by their files.
Every length, count, offset and shape in weights.bin is checked against
the bytes behind it before anything is allocated or copied, and an
unknown, missing or duplicated tensor is refused.

### Only TitaNet-small is accepted

A converted model whose tensor shapes are not TitaNet-small's is refused
when opened. The converter converts TitaNet-large as well; the runtime
does not open it.

### The front end is sherpa-onnx's for NeMo models

The features are kaldi-native-fbank's as sherpa-onnx configures it for
a NeMo speaker model, on samples scaled to [-1, 1]:

- 16 kHz. Faster audio is resampled per span by Kaldi's
  LinearResample, built fresh for the span, cut off at 99% of 8 kHz
  with six zero crossings and not flushed, as sherpa-onnx does. Slower
  audio is refused.
- Frames of 25 ms every 10 ms, snip edges: every frame lies wholly in
  the span, and a span of n samples has 1 + (n - 400) / 160 frames.
- Pre-emphasis 0.97, the first sample taken against itself, with no DC
  removal and no dither.
- The periodic Hann window, 0.5 - 0.5 cos(2πi/400), not Povey's.
- The power spectrum of a 512-point FFT.
- 80 triangular mel bands on the Slaney scale, linear below 1 kHz and
  logarithmic above, from 0 Hz to 7,600 Hz, 400 Hz under Nyquist. Each
  triangle is laid out in hertz and scaled by 2 over its width in hertz,
  Slaney's area normalisation, so every band has about unit area.
- The natural log of each band's energy, floored at f32's epsilon.
- Each band normalised over the span: less its mean, divided by its
  population standard deviation plus 1e-5.

The normalisation is computed in double precision, where both
references compute it in single. That is the one deliberate difference
from them. Where a band sits at the log floor in every frame of a span,
as on quiet clips, the exact result is 0, while the single-precision
mean is an ulp off and the division by 1e-5 turns that into about
±0.087 across the band. On the speaker page's protocol the two score
the same, 84.9%.

### A span is embedded whole at its exact length

The network runs over exactly the span's frames, with no padding. A
span of fewer than 25 frames carries no evidence, as under
`[[rr:TD-14#A span needs a quarter second]]`. A span of more than 120
frames, 1,205 ms of audio, carries none either: it is above the
longest real word on the benchmark streams, 1,050 ms, and bounds the
scratch a recognizer keeps for one embedding at about 3.5 MB.

### Any span can be embedded from any thread

The model embeds any span it is given, without a recognizer: in Rust
SpeakerModel::embed(&[f32]) -> Result<Vec<f32>> on 16 kHz samples in
[-1, 1], and in C

```c
int utter_spk_model_embed(const UtterSpkModel *model, const float *samples,
                          int n, float sample_rate, float *out, int cap);
int utter_spk_model_dim(const UtterSpkModel *model);
```

which returns the dimension written, or -1 for an x-vector model or on
an error. A model is shared between threads. The call's scratch lives
for the call, so it takes spans up to 3,000 frames, 30 s, for
enrolment recordings.

### The recognizer embeds a word once its span has closed

A word is queued at the decoder advance whose best path first shows
another entry after it, and queued again only if a later best path
gives it a different span. Its evidence is kept by (start sample, end
sample) and read from there by every result that shows that span;
results only read it. The kept evidence is dropped at each final. A
model set mid-stream queues the current partial's words.

### Embedding runs in slices between advances

Each accept call that does not advance the decoder runs the queued
embeddings for a fixed budget of work, counted in frames of a stage
times that stage's multiply-adds per frame, never in time, so the same
audio in the same blocks gives the same results on any machine. A call
that advances the decoder runs none, and neither does reading a result
or a final. A word's evidence appears on the first result after its
embedding completes, and its bytes are the stateless call's. The budget
is set by what a block may cost, below.

### What carries evidence

- An entry that carries evidence under
  `[[rr:TD-14#Evidence is pooled over a span]]` carries the four keys
  once its span is embedded, and none before. A final word whose
  embedding has not finished carries none: absent keys are no evidence,
  and a gate on them fails closed.
- No partial and no final carries evidence at its top level.
- spk is the raw embedding, 192 values as the network's last layer
  gives them, neither scaled to unit length nor to the square root of
  its length. Cosine is the natural score.
- spk_frames is the filterbank frames embedded; spk_start and spk_end
  are the first frame's start and the last frame's end, on the clock of
  start_sample, in the order `[[rr:TD-14#Evidence is pooled over a span]]`
  sets.

### Readings and alternatives: decided by measurement

Readings and final alternatives show spans the best path may never
hold. Two rules are built behind a switch:

- **Already embedded.** A reading's or an alternative's entry carries
  evidence only when its span is one the best path queued and its
  embedding has completed. Nothing is queued for it.
- **Own jobs.** A reading's or an alternative's closed word is queued
  like a best-path word, after them.

The speaker page's run on the recognizer's pull request decides: the
share of reading and alternative words that carry evidence, the
acceptance at 1% false accept of the words a reading puts forward, and
the 99th percentile and worst block and real-time factor of each rule
against the baseline below. The rule the figures favour is kept and the
switch removed.

### The floor margin: decided by measurement

With a floor margin set, `[[rr:set_endpoint_floor_margin]]`, the
x-vector leaves out frames within the margin of the floor. TitaNet
embeds a span whole, so there is no frame to leave out inside it. Two
rules are built behind a switch:

- **The span as reported.** The margin does not touch the embedding.
- **Floor-level edges trimmed.** Frames within the margin of the floor
  at either edge of the span are cut before it is embedded; a span that
  falls under 25 frames carries no evidence.

The speaker page decides, by acceptance at 1% false accept per word
length bin with and without the margin set.

### What a block may cost

A recognizer with TitaNet set is held to 0.0.5's with the x-vector set:
over the speaker-cost streams, a 99th percentile block of 3.26 ms and a
real-time factor of 0.0197 are not exceeded, nor is the slowest block.
The recognizer's pull request measures, on the harness that holds those
figures: the 99th percentile, worst block and real-time factor in each
configuration; the 99th percentile of the advancing blocks alone with
and without slices, which is where running 27 MB of weights between
advances would show by evicting the acoustic model from cache; how long
after a word's end its evidence arrives; the share of final words that
carry it; and result bytes per block.

### Verification

scripts/titanet_oracle.py compares the stages against the references on
210 spans of Speech Commands test clips, the loudest and the quietest
clip of every word, each whole, cut to a multiple of 16 frames and cut
to a seeded random length.

- Log mel features against kaldi-native-fbank 1.22.3 agree within
  1.3e-3, all of it in the lowest bands of loud clips, where the two
  FFTs round differently at a bin within about 1e-7 of the spectrum's
  peak. Normalised features, both sides in double precision, agree
  within 4.1e-4.
- The network against onnxruntime on identical features, over the 210
  and the research's 300 probe word spans: 1 - cosine at most 3.7e-12,
  maximum absolute difference 1.1e-6.
- The whole path against kaldi-native-fbank, single-precision
  normalisation and onnxruntime: minimum cosine 0.899, and against
  sherpa-onnx 0.964, both from the 22 spans with a band at the floor
  throughout. Given the runtime's normalisation, onnxruntime agrees on
  all 210 to cosine 0.99999976.
- At 48 kHz against sherpa-onnx fed at 48 kHz, 62 clips: minimum cosine
  0.9999921.
- Slices under budgets from one unit to random, and at 10, 20, 40 and
  200 ms blocks, are byte-identical to one call.

The model tests hold committed reference vectors to cosine 0.9999998.

## Consequences

- A result grows by about 2 KB of text per word that carries evidence,
  against the x-vector's 1.3 KB.
- A model is 27 MB of weights, shared by every recognizer; each
  recognizer with it set keeps about 3.5 MB of scratch.
- Evidence arrives after a word's span closes, not while it forms: a
  gate on it fails closed until then.
- Enrolment is the stateless call over the enrolment recordings, and
  profiles fitted on the x-vector do not carry over.
- The crate ships no weights. The converter needs onnx and numpy, which
  are the script's dependencies, never the crate's. TitaNet-small is
  CC-BY-4.0, and a host that ships it carries the attribution.
- `exp` and `tanh` differ between platforms' maths libraries, so vectors
  are byte-identical within a platform, not across them.
- The x-vector stays. Whether it is kept as the light fallback, for 8
  kHz audio and evidence while a word forms, is decided at the release
  that measures TitaNet in the crate.
