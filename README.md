# utter

Pure-Rust streaming speech decoder for Vosk (Kaldi nnet3) models, with
zero dependencies. Feed 16 kHz audio and get the best-path partial after
every block, the distinct hypotheses alive in the search beam, word
times, runtime word-list grammars, and Kaldi endpointing. No C
toolchain, no Docker, no cloud.

## Status

Specification stage. There is no code in this repository yet. The
design is settled to the level of the Kaldi semantics each stage must
reproduce and the gates that decide when it is done. Decisions live in
[`docs/td/`](docs/td/): TD-1 describes the record system, TD-2 is the
runtime specification. The first use case it is built for is in
[`usecases/`](usecases/).

## Why

Vosk is the best small offline recognizer for a closed vocabulary, and
its partials are fast: under a grammar of a few dozen words a spoken word shows up in
the partial a median 40 ms after it ends, against about half a second
for every ONNX streaming model tried under the same grammar. But Vosk
is a C++ wrapper around Kaldi, and Kaldi does not build on Windows
without a Docker cross-compile, so anyone who needs one more field out
of the decoder than the stock wheel exposes ends up maintaining a fork
they cannot easily ship.

utter reimplements only the parts of that stack a partials-first
application uses, in Rust, from the model files a stock Vosk model
directory already contains. It is not a port of libvosk and it is not
a general Kaldi. It decodes, it tells you what it thinks so far and
what else it is still considering, and it tells you when the speaker
stopped.

## What it does

- Loads a stock Vosk model directory: the nnet3 chain acoustic model,
  the i-vector extractor, and the `HCLr.fst` lookahead graph.
- Takes a grammar at construction as a list of strings, builds the same
  bigram libvosk builds, and composes it with the graph.
- Runs Kaldi-exact MFCC, online CMVN, splice, LDA and the online
  i-vector estimate, then the network in streaming chunks.
- Decodes frame-synchronously with Kaldi's beam, max-active and
  min-active semantics.
- After every audio block returns the best path as the partial, with
  each word's start and end in samples of the audio you fed.
- Returns the distinct word sequences still alive in the beam, ranked
  by cost, the empty sequence included when it is a contender. These
  are current to the last decoded frame, not to a lattice that trails
  the audio.
- Applies Kaldi's endpoint rules and reports the end of speech as a
  sample position.
- Reports per-frame energy and the silence state of the best path from
  the same loop that decodes, so a caller does not need a second voice
  activity detector on a second clock.

## What it does not do

- Lattices, lattice determinization, minimum Bayes risk confidences, or
  lattice n-best. Beam n-best replaces them.
- Decoding against the model's full language model (`Gr.fst`). A
  grammar is required.
- RNNLM or ARPA rescoring, speaker vectors, batch or GPU decoding.

## Dependencies

None at runtime. `cargo build` fetches nothing. The standard library is
the foundation and `std::arch` supplies the SIMD paths. Every piece the
decoder needs is small enough to own: the Kaldi binary readers, a
split-radix real FFT written after Kaldi's, a blocked single-precision
GEMM, Cholesky and conjugate gradient for the i-vector, an OpenFst
`ConstFst` reader with the `olabel_lookahead` add-on, composition and
trimming, and JSON. An optional `gemm` feature can swap in
`matrixmultiply` if the own kernel misses its budget on a target; it is
off by default. Dev-dependencies never reach a consumer's build.

## Interface

A Rust library, plus a `cdylib` with a C ABI that mirrors it so hosts
without Rust get the same surface. Sketch:

```rust
let model = utter::Model::open("vosk-model-small-en-us-0.15")?;
let mut rec = utter::Recognizer::new(&model, 16_000, &["alpha", "bravo", "seven"])?;
rec.set_alternatives(4);

loop {
    let block: &[i16] = capture.next_block();          // 40 ms, mono
    let step = rec.accept(block);                       // decodes, checks endpoint
    let partial = rec.partial();                        // best path, word spans in samples
    for reading in rec.alternatives() { /* ranked, distinct, may be empty */ }
    if let Some(end) = step.end_of_speech { /* sample position */ }
}
```

The JSON produced by the C ABI keeps the key layout of libvosk's
`PartialResult` and `Result`, extended with the alternatives and the
sample-indexed spans, so a host that already parses Vosk output keeps
parsing.

## Model compatibility

Any Vosk small model with the standard layout: `am/final.mdl`,
`graph/HCLr.fst`, `graph/words.txt`, `graph/disambig_tid.int`,
`graph/phones/word_boundary.int`, `ivector/`, `conf/mfcc.conf`,
`conf/model.conf`. The English small model is the reference; the
German, French, Spanish and Russian small models share the layout and
are checked before they are claimed.

## Verification

The stock `vosk` wheel is the oracle. The gates, in order:

1. Batch decode of a replay corpus through the vendored acoustic and
   graph code, scored against libvosk's finals, with and without
   i-vectors: the i-vector decision.
2. Front-end parity: MFCC within 1e-3, i-vectors within 1e-2 relative.
3. Decoder parity: partial text after each block equals libvosk's on at
   least 95% of blocks; finals on at least 99% of segments; word times
   within one output frame.
4. Latency: the same median and 90th percentile first-appearance
   figures as the stock wheel, within one block, on the same corpus.
5. Alternatives: rank 0 always equals the partial; the empty reading is
   offered as empty when it leads.
6. Endpoints within one 0.2 s step of libvosk's on 95% of segments.

## Third-party code

The Kaldi model reader, MFCC and nnet3 forward pass start from
[Vosk-Rust](https://github.com/Reza2kn/Vosk-Rust), Apache-2.0, vendored
as source with its notices kept. Kaldi and OpenFst were read for the
semantics this crate reproduces; no code from either is included.

## License

MIT. See [LICENSE](LICENSE). Vendored Apache-2.0 files keep their own
headers.
