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

## Not planned

- **General large-vocabulary or free-form transcription.** The runtime
  decodes under a grammar only, by design; word-error benchmarks such
  as LibriSpeech do not apply.
- **Model architectures the runtime does not implement**, such as the
  CNN-TDNN of the larger English model, which is refused when opened.
  Support is added only if a consumer needs it.
- **Runtime dependencies.** The zero-dependency constraint stays; it is
  the point of the crate.
