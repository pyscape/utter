# Assurance case

Why the security requirements in [SECURITY.md](../SECURITY.md) are met.

## What is trusted, and what is not

utter is an in-process library. It runs with the privileges of the host
that links it; it opens no network connection, spawns no process, and
uses no cryptography, so its attack surface is the data it reads.

- **Untrusted:** a model directory and its files (the Kaldi model, the
  OpenFst graphs and symbol tables, the transition model and i-vector
  extractor), the grammar as a JSON string, and the audio samples. Any
  of these may be malformed, truncated or hostile.
- **Trusted:** the host process, its memory, and the build and release
  pipeline.

The trust boundary is the point where the library parses those bytes:
the readers in `kaldi_io`, `fst`, `transition_model`, `ivector` and
`json`, and the C ABI in `capi`, which turns caller-supplied pointers
and lengths into Rust slices. Everything downstream operates on values
those readers have already validated.

## Secure design

- **Memory-safe language.** The crate is Rust; the only `unsafe` is the
  C ABI and the AVX2 kernels, each site carrying a SAFETY comment that
  states its precondition, enforced by the `undocumented_unsafe_blocks`
  and `unsafe_op_in_unsafe_fn` lints.
- **Validate before use.** A length or count read from a file is checked
  against the bytes remaining before it sizes an allocation; an id or
  state index is checked against its table before it is used. Invalid
  input becomes an error, not a panic or an out-of-bounds access.
- **Least privilege and small surface.** No network, no subprocess, no
  cryptography, no runtime dependencies. The release and CI workflows
  request least-privilege tokens and pin every action by commit hash.

## Common weaknesses, and how each is countered

- **Out-of-bounds read (CWE-125).** The readers reject an arc or start
  state that names a state the graph does not have, and a symbol or word
  id outside its table; the C ABI bounds every buffer by the length the
  caller passes. Miri runs the unit tests to catch a stray access the
  checks miss.
- **Integer overflow (CWE-190).** Size arithmetic uses checked or
  saturating operations; the crate denies the truncating and
  sign-losing cast lints, so a narrowing conversion is a checked one.
- **Unbounded allocation / uncontrolled resource use (CWE-789/400).** A
  count from a file must fit the bytes behind it before it sizes a
  vector, so a large length field yields an error rather than an
  allocation the input never backs.
- **Panic on malformed input.** The parsers return `Result`; a
  malformed file is an error the host can handle, not a crash.

## Evidence

- Six fuzz targets over the readers and the audio path run on every push
  and for longer weekly; the crashes they first found are fixed and each
  has a regression test.
- Miri runs the unit tests on every push.
- clippy with every warning denied, CodeQL, and cargo-deny run in CI;
  line coverage is over ninety percent.
- The reasoning above is checked against the code by the reviews and the
  decision records under [td](td).
