# Roadmap

Direction for roughly the next year. It is intent, not a commitment to
dates; the decision records under [docs/td](docs/td) fix what has been
settled, and the [use cases](usecases) state what the library must do
for a consumer.

## Where the project is

Released at 0.0.1: a streaming decoder for Vosk models with no
dependencies, a C ABI and a Python binding, matching the stock Vosk
wheel on the public Speech Commands benchmark. The English small model
is the reference.

## Planned

- **Close the open parity gap.** Segment-for-segment agreement with
  libvosk on near-tie acoustics sits just under the mark the gate asks
  for; the aim is to reach it.
- **Reach a stable 0.1 API.** The surface keeps the Vosk wheel's shape;
  breaking changes are expected before 0.1 and reported through
  cargo-semver-checks. After 0.1 the API settles.
- **Verify more small models.** The German, French, Spanish and Russian
  small models share the layout and are expected to work; each is to be
  checked against the wheel and its result recorded.

### Direct audio input from a speaker extractor

We will accept the companion library's reconstructed waveform directly
as borrowed normalized `f32` samples, so it can feed utter without
serializing audio or rounding it to 16-bit integers. The boundary and
compatibility requirements are settled in `[[rr:TD-13]]`; the companion
library's model and training plan are described in the
[target speaker extraction design](docs/design/target-speaker-extraction.md).

- [ ] Add `Recognizer::accept_f32(&[f32]) -> Result<Step, AudioInputError>`
  with explicit sample-rate and full-scale conventions, whole-block
  validation, and the existing integer API preserved.
- [ ] Share input advancement between integer and float calls, converting
  into the front end's existing amplitude units with reusable scratch.
  Preserve the current decoder, endpoint and sample-accounting order.
- [ ] Retain waveform history as floats and update per-word energy and
  floor tracking to preserve fractional samples. Verify identical results
  for integer-origin audio and measure the additional history memory.
- [ ] Add a Rust composition example that consumes borrowed float slices
  directly or groups them into 40 ms blocks. Cover continuous silence,
  source offsets, capture discontinuities, and draining the upstream
  processor before `final_result`.
- [ ] Add float/integer equivalence, fractional-amplitude, invalid-input
  and stream-lifecycle tests; run the existing stock-Vosk parity gates.
  Use a synthetic upstream processor so these tests need no trained
  separator model or new runtime dependency.
- [ ] Benchmark 10, 20 and 40 ms float delivery, including compute,
  allocations, retained memory and result timing. Once extraction weights
  are available, measure target-command accuracy, interfering-speaker
  false commands, silence behavior and end-to-end latency separately.

### Speaker evidence from a stronger model

Every word carries speaker evidence from vosk-model-spk-0.4, the
x-vector the stock wheel uses, `[[rr:TD-14]]`. On the [speaker evidence
benchmark](docs/benchmarks/speaker-evidence.md) a gate at 1% false
accept passes 40.1% of an enrolled speaker's words on that evidence.
NVIDIA's TitaNet-small, run on one CPU thread over the span utter
reports for each word, passes 84.9%, and 64.9% of words under 400 ms
against 29.2%. It needs no GPU: through an ONNX runtime a word costs
about 6 ms on one core. Embedded again whenever a word's span changes
on a partial, its evidence 200 ms after the word ends passes 77.7% of
words, where the x-vector's passes 33.1%. TitaNet-large passes fewer of
these words, 82.0%, at three to five times the cost.

TD-14 did not take TitaNet for two reasons that still hold. Its
squeeze-excitation layers and attentive pooling average over the whole
input, so a span's embedding cannot be pooled from rows computed once
per frame: each word's span is embedded on its own, at its exact
length. And it ships as ONNX, which the crate does not read. A host can
embed utter's word spans itself today from start_sample and
end_sample; the aim here is the same evidence from the crate with no
runtime dependency. A prototype of the network on the crate's own
matrix kernel reproduces the ONNX graph to a cosine of at least
0.9999997 on 300 word spans, and embeds a half-second word in 3.6 ms on
one core.

- [ ] Record the model in its own TD, as TD-14 requires of a model that
  pools differently: the 16 kHz front end, a span embedded whole at its
  exact length, a call that embeds any span from any thread, when the
  recognizer embeds a word for its four keys (on the partial that first
  shows another entry after it, and again only if a later span
  differs), and what that block may cost.
- [ ] Convert the ONNX weights once, by a script beside the model, into
  a layout the crate reads, without the training classifier: 6.8M
  parameters, 27 MB as f32. Weights that training decayed below 1e-15
  are zeroed; kept, their subnormal products made an embedding two to
  three times slower.
- [ ] Implement the front end (25 ms Hann window, 80 bands on the
  librosa mel scale to 7.6 kHz, per-band normalisation over the span),
  the depthwise and pointwise convolutions, squeeze-excitation and
  attentive statistics pooling on the existing matrix kernel. Verify
  the network against onnxruntime on identical features, the front end
  against kaldi-native-fbank, and the whole against sherpa-onnx.
- [ ] Add a row to the speaker page: accuracy beside the host-side one,
  the cost of the block that embeds, and the evidence against time
  after a word's end.
- [ ] Run the benchmark on a set recorded through one microphone, such
  as VCTK, where a profile cannot lean on the device and the room.
- [ ] Decide the x-vector's place once TitaNet is measured in the
  crate: the lighter model beside it, or retired before 0.1.

### Overlapping speech

The per-word gate assumes one voice in each word. When a second voice
talks over the enrolled speaker, the goal is that the gate never acts
on a word the other voice spoke, and loses as few of the enrolled
speaker's own words as it can. That is two jobs. Detection says that a
word's span holds two voices, so the gate fails closed on it.
Separation recovers the enrolled speaker's voice from the mixture, so
the word can still be recognised and admitted; that is the companion
library's [target speaker extraction](docs/design/target-speaker-extraction.md).
Diarization, labelling who spoke when across a session, is not the
goal: the gate needs a verdict on one word within a few hundred
milliseconds of its end, not a session's turns.

On two-speaker mixtures of Speech Commands words, with the shorter word
wholly inside the other, utter decodes both words in 5 to 7% of
mixtures, and a TitaNet-small gate at 1% false accept admits 7.9% of
the words it does decode as the speaker who did not say them, against
1.5 to 1.7% when the words only meet or overlap by a quarter; 9 of 51
words of a quieter speaker said under a louder one were admitted as the
louder. A threshold strict enough to hold that to 1% cuts the clean
words it passes from 81.5% to 57.0%. pyannote segmentation-3.0 flags 36
to 67% of fully overlapped frames and catches about half of the wrong
admissions while turning away 7 to 24% of clean words; NVIDIA's
Streaming Sortformer v2 on CPU flags 14 to 35% and catches at most a
third, 750 ms or more after the word. Both see seconds of cross-talk: 65% and
83% of those frames.

Open questions:

- What a detector must deliver: which words it flags, how soon after a
  word's end, and at what share of clean words lost.
- Whether a detector that runs on CPU within that budget can see one
  word said over another, or one must be trained for it.
- Whether detection belongs in the crate, as a key beside a word's
  speaker evidence, or stays with the host.
- How separation and the gate compose: whether the gate scores the raw
  or the extracted audio, and whether TitaNet on extracted audio still
  scores as its speaker.
- Whether real recorded overlaps and a same-microphone set move these
  figures.

Candidates:

- pyannote segmentation-3.0 (MIT; ONNX through sherpa-onnx; 1.5M
  parameters; about 6% of one core): cheap and available now, catches
  about half of the wrong admissions.
- NVIDIA Streaming Sortformer 4spk v2 (CC BY 4.0; 117M parameters; CPU
  through NeMo-Speech.cpp, Apache 2.0): within real time on one thread
  only with 1.6 s chunks, at 0.36 to 0.61 of a core; the model card's
  1.04 s setting takes 2.4 cores. Its successor, Nemotron-3-Diarization
  (OpenMDW 1.1), is not measured.
- NeMo MSDD (NGC terms; TitaNet-large at five scales, clustered over
  the whole session; PyTorch): offline, not a live candidate.
- A detector trained on word-length overlaps, or the extractor's own
  activity evidence once it is trained.

## Not planned

- **General large-vocabulary or free-form transcription.** The runtime
  decodes under a grammar only, by design; word-error benchmarks such
  as LibriSpeech do not apply.
- **Model architectures the runtime does not implement**, such as the
  CNN-TDNN of the larger English model, which is refused when opened.
  Support is added only if a consumer needs it.
- **Runtime dependencies.** The zero-dependency constraint stays; it is
  the point of the crate.
