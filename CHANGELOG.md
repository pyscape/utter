# Changelog

Notable changes to the crate, newest first. The project follows semantic
versioning; release dates are recorded by the git tags and the GitHub
releases.

## Unreleased

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
