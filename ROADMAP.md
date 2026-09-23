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

## Not planned

- **General large-vocabulary or free-form transcription.** The runtime
  decodes under a grammar only, by design; word-error benchmarks such
  as LibriSpeech do not apply.
- **Model architectures the runtime does not implement**, such as the
  CNN-TDNN of the larger English model, which is refused when opened.
  Support is added only if a consumer needs it.
- **Runtime dependencies.** The zero-dependency constraint stays; it is
  the point of the crate.
