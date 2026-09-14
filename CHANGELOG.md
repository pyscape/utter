# Changelog

Notable changes to the crate, newest first. The project follows semantic
versioning; release dates are recorded by the git tags and the GitHub
releases.

## 0.0.4

### Fixed

- The floor margin on the bound reads no span shorter than the bound
  itself. A quiet word's first frames after a pause sit near the floor
  on any microphone, and 0.0.3 let them waive the veto and count as
  silence, cutting a two-word command in two; on the first consumer's
  recordings that lost 29 words for 39 gained across 13 of 33 takes.
  A span at least the bound's length is seconds on a cold room and a
  few frames at an onset. TD-12 amended.

## 0.0.3

### Added

- A floor margin on the host's endpoint bound, `set_endpoint_floor_margin`,
  `SetEndpointFloorMargin`, `utter_recognizer_set_endpoint_floor_margin`:
  within it the bound reads a wordless `[speech]` path as trailing
  silence, a rival word at the floor cannot veto, and the final is the
  path as it stood with the new endpoint value `floor`. Off by default
  and byte-identical when unset. TD-12.
- `utter_version` and `utter_revision` in the C ABI: the crate version
  and the git commit the library was built from, so a host can record
  which runtime produced a result.

### Fixed

- Digital silence no longer reads as -999 dBFS. A word entry's
  `energy_dbfs` is `null` when there is no signal under the word, and
  `floor_dbfs` is absent while the quietest windows of the last ten
  seconds are digital silence, so a host gate relative to the floor is
  not handed a number every word sits above.
- `REVISION` reports the commit in a crate built from crates.io. The
  build script reads the commit `cargo package` records when there is
  no checkout to ask; 0.0.1 and 0.0.2 report "unknown" from the registry.

### Distribution

- The release attaches the Sigstore bundle of its build-provenance
  attestation, `utter-vX.Y.Z.sigstore.json`, so an archive verifies
  offline with `gh attestation verify --bundle`.

## 0.0.2

### Security

- The model-file and audio readers reject malformed input as an error
  rather than panicking or allocating from an unchecked length field. A
  length or count is checked against the bytes behind it before it sizes
  an allocation, and an out-of-range id or state is rejected. Six fuzz
  targets cover the readers and the audio path and run in CI.

### Changed

- Narrowing numeric conversions are checked. A negative or out-of-range
  dimension, word id or state read from a corrupt model file is now an
  error instead of a wrapped value.

There is no change to the decoding API or its output; the benchmark
per-clip records are unchanged from 0.0.1.

### Distribution

- Prebuilt release archives for Linux, Windows and macOS, each carrying
  the shared library, the `stream` tool and the C header. The macOS
  archive is one universal binary; the Windows archives are zips; the
  Linux libraries link against glibc 2.28. Every archive carries a
  build-provenance attestation, verifiable with `gh attestation verify`,
  and the release tag is signed.

## 0.0.1

- First release: a pure-Rust streaming decoder for Vosk models with no
  dependencies, a C ABI and a Python binding. Matches the stock Vosk
  wheel on the public Speech Commands benchmark.
