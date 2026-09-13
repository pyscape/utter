# utter

[![CI](https://github.com/pyscape/utter/actions/workflows/ci.yml/badge.svg)](https://github.com/pyscape/utter/actions/workflows/ci.yml)
[![docs](https://github.com/pyscape/utter/actions/workflows/docs.yml/badge.svg)](https://github.com/pyscape/utter/actions/workflows/docs.yml)
[![crates.io](https://img.shields.io/crates/v/utter)](https://crates.io/crates/utter)
[![PyPI](https://img.shields.io/pypi/v/utterpy)](https://pypi.org/project/utterpy/)
[![License: MIT AND Apache-2.0](https://img.shields.io/badge/license-MIT%20AND%20Apache--2.0-blue.svg)](LICENSE)

A pure-Rust streaming speech recognizer for [Vosk](https://alphacephei.com/vosk/)
models, built for applications that act on partial results. Feed it
16 kHz audio in small blocks and after every block it tells you the
words it believes so far, the rival readings it is still weighing, how
loud and how stable each word is, and when the speaker stopped.

Zero dependencies. No C toolchain, no Docker, no cloud. It loads a stock
Vosk model directory as-is.

```python
import json, utterpy

model = utterpy.Model("vosk-model-small-en-us-0.15")
rec = utterpy.KaldiRecognizer(model, 16000, json.dumps(["alpha", "bravo", "seven"]))
rec.SetWords(True)
rec.SetPartialWords(True)
rec.SetPartialAlternatives(4)

for block in microphone():                 # 40 ms of 16-bit mono PCM
    if rec.AcceptWaveform(block):          # the speaker stopped
        print(rec.Result())
    else:
        print(rec.PartialResult())         # best guess, rivals, word spans
```

## Contents

- [Why utter](#why-utter)
- [Status](#status)
- [Quick start](#quick-start)
- [What a partial tells you](#what-a-partial-tells-you)
- [Where next](#where-next)
- [Features](#features)
- [Models](#models)
- [Benchmarks](#benchmarks)
- [Building and testing](#building-and-testing)
- [Repository layout](#repository-layout)
- [Third-party code and license](#third-party-code-and-license)

## Why utter

Vosk is the best small offline recognizer for a closed vocabulary, and
it is fast: under a grammar of a few dozen words, a spoken word appears
in the partial result a median 40 ms after the word ends. Streaming
ONNX models tried under the same grammar took about half a second.

But Vosk is a C++ wrapper around Kaldi, and Kaldi does not build on
Windows without a Docker cross-compile. Anyone who needs one more field
out of the decoder than the stock wheel exposes ends up maintaining a
fork they cannot easily ship.

utter reimplements the part of that stack a partials-first application
uses, in Rust, from the files a stock Vosk model directory already
contains. It is not a port of libvosk and not a general Kaldi. It
decodes, it tells you what it thinks so far and what else it is still
considering, and it tells you when the speaker stopped. Its output is
checked block-for-block against the stock Vosk wheel on the same audio.

## Status

**0.0.1**, the first release: `utter` on crates.io, `utterpy` on PyPI.
The streaming path is complete and measured: front end, i-vector
adaptation, chunked neural network, decoder, partials with alternatives
and word times, endpointing, a C ABI and a Python binding. The API
keeps the vosk wheel's shape; the keys it adds are in the
[results reference](docs/reference/results.md). Expect breaking changes
before 0.1.

On the public Speech Commands benchmark it matches the stock Vosk
wheel: 91.8% against 91.5% accuracy on 11,005 clips, finals agreeing
on 99.2% of them, the same 40 ms median first-appearance latency. See
[Benchmarks](#benchmarks).

One parity gap is open: segment-for-segment agreement with libvosk on
the private test corpus sits just under the mark the gate asks for, on
near ties in the acoustics. Every gate result is in
[docs/gates/](docs/gates/).

Supported today: the English small model. Other small models share the
layout and should work but have not been checked.

## Quick start

### Python

The Python package is [utterpy](https://github.com/pyscape/utterpy).
Its API mirrors the `vosk` package so existing code keeps working, and
the wheel ships type stubs:

```bash
pip install utterpy
```

```python
import json, wave, utterpy

model = utterpy.Model("vosk-model-small-en-us-0.15")
rec = utterpy.KaldiRecognizer(model, 16000, json.dumps(["alpha", "bravo", "seven"]))
rec.SetWords(True)
rec.SetPartialWords(True)
rec.SetPartialAlternatives(4)

with wave.open("alpha_seven.wav") as w:          # 16 kHz, 16-bit, mono
    while block := w.readframes(640):            # 40 ms
        if rec.AcceptWaveform(block):
            print("final:", rec.Result())
        else:
            print("partial:", rec.PartialResult())
print("final:", rec.FinalResult())
```

### Rust

```toml
[dependencies]
utter = "0.0.1"
```

```rust
use std::path::Path;

let model = utter::Model::open(Path::new("vosk-model-small-en-us-0.15"))?;
let grammar: Vec<String> = ["alpha", "bravo", "seven"]
    .iter()
    .map(|s| s.to_string())
    .collect();
let mut rec = utter::Recognizer::new(&model, 16_000.0, &grammar)?;
rec.set_words(true);            // word spans on finals
rec.set_partial_words(true);    // and on partials
rec.set_alternatives(4);        // rivals to carry alongside the partial

while let Some(block) = capture.next_block() {  // 40 ms of mono PCM
    let step = rec.accept(block);               // decodes, applies the endpoint rules
    if step.endpoint {
        println!("{}", rec.result());           // the segment that just ended
    } else {
        println!("{}", rec.partial());          // best path, rivals, word spans
    }
    let _decoded_through = step.sample;         // samples fed, as of the last decoded frame
}
println!("{}", rec.final_result());
```

This is [examples/readme_sketch.rs](examples/readme_sketch.rs), which
CI builds, so it cannot drift from the API.

### C

`cargo build --release` also produces a shared library, `libutter.so`,
`libutter.dylib` or `utter.dll` under `target/release`. The header is
[include/utter.h](include/utter.h):

```c
UtterModel *model = utter_model_new("vosk-model-small-en-us-0.15");
UtterRecognizer *rec = utter_recognizer_new_grm(model, 16000.0f, "[\"alpha\", \"bravo\", \"seven\"]");
utter_recognizer_set_words(rec, 1);
utter_recognizer_set_partial_words(rec, 1);
utter_recognizer_set_alternatives(rec, 4);

while (read_block(pcm, 640)) {
    if (utter_recognizer_accept_waveform_s(rec, pcm, 640))
        puts(utter_recognizer_result(rec));
    else
        puts(utter_recognizer_partial_result(rec));
}
puts(utter_recognizer_final_result(rec));
```

All three return the same JSON strings.

Build a C host against the header and the library, and keep the
library on the loader's path at run time:

```bash
cc host.c -I include -L target/release -lutter -o host
LD_LIBRARY_PATH=target/release ./host
```

## What a partial tells you

The stock Vosk partial is a line of text:

```json
{"partial": "alpha seven"}
```

utter returns the same text, and beside it the evidence an application
otherwise has to reconstruct with its own detectors:

```json
{
  "partial": "alpha seven",
  "partial_alternatives": [
    {"text": "alpha seven",  "confidence": 1.94,  "result": [ ... ],
     "relation": "same",    "lead_delta": 0.41},
    {"text": "alpha eleven", "confidence": -3.32, "result": [ ... ],
     "relation": "differs", "lead_delta": -0.41}
  ],
  "partial_result": [
    {"word": "alpha", "start_sample": 196800, "end_sample": 202080, "energy_dbfs": -23.2, "stable_ms": 960},
    {"word": "seven", "start_sample": 202080, "end_sample": 207680, "energy_dbfs": -22.8, "stable_ms": 240},
    {"word": "[sil]",  "start_sample": 207680, "end_sample": 208320, "energy_dbfs": -48.6, "stable_ms": 0}
  ],
  "floor_dbfs": -51.7
}
```

Each word also carries Vosk's `start` and `end` in seconds, left out
here for room. The key layout matches libvosk's, so a parser written
for Vosk keeps working; the new keys are added after the old ones.
Every key, its type, when it is present and whether Vosk has it is in
the [results reference](docs/reference/results.md).

In one line each: the gap between the first two confidences predicts
whether the leading word will be revised; a reading that `extends` the
partial and is gaining lead is the next word forming; `stable_ms` is
the hold to wait out before acting on a word; `energy_dbfs` against
`floor_dbfs` tells a spoken word from one read into a quiet room;
`[sil]` and `[speech]` say whether the decoder heard nothing or a word
it cannot yet name; and `endpoint` on a final says what closed it.

## Where next

| You are | Start with |
|---|---|
| Using the `vosk` package today | [Coming from Vosk](docs/reference/vosk.md): every method, where it stands, and what is added. |
| A Rust engineer | `cargo doc --open`. The quick start is on the crate page and each method that returns JSON links to the results reference. |
| Reading the JSON | [Results](docs/reference/results.md): every document the recognizer returns and every key in it. |
| Building on the partials | [Reading a partial](docs/guide/reading-a-partial.md): the trust read as a calibrated probability, the next-word preview, silence and the doubtful last word, each a line of Python. Then [Ending speech sooner](docs/guide/ending-speech-sooner.md): the endpoint bound and its veto, with what they buy and cost. |

The [docs index](docs/README.md) lists every directory.

## Features

- Loads a stock Vosk model directory: the nnet3 chain acoustic model,
  the i-vector extractor and the `HCLr.fst` lookahead graph.
- Takes a grammar at construction as a list of words or phrases, builds
  the same bigram libvosk builds, and composes it with the graph.
- Kaldi-exact MFCC, online CMVN, splice, LDA and online i-vector
  estimation, then the network in streaming chunks.
- Frame-synchronous decoding with Kaldi's beam, max-active and
  min-active semantics.
- After every block: the best path as the partial, each word's span in
  samples, its loudness and how long it has held.
- The distinct word sequences still alive in the beam, ranked, with
  each one's relation to the partial and the motion of its lead.
- Kaldi's endpoint rules, plus an optional bound of your own.
- The room's noise floor, so silence detection needs no second detector
  on a second clock.
- A C ABI and a Python binding with the `vosk` package's API.

Not included, by design:

- Lattices, lattice determinization, minimum Bayes risk confidences or
  lattice n-best. Beam n-best takes their place.
- Decoding against the model's full language model. A grammar is
  required.
- RNNLM or ARPA rescoring, speaker vectors, batch or GPU decoding.

## Models

Models are downloaded from
[alphacephei.com/vosk/models](https://alphacephei.com/vosk/models) and
unpacked into a directory the examples open by name:

```bash
curl -LO https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
```

Any Vosk small model with the standard layout: `am/final.mdl`,
`graph/HCLr.fst`, `graph/words.txt`, `graph/disambig_tid.int`,
`graph/phones/word_boundary.int`, `ivector/`, `conf/mfcc.conf`,
`conf/model.conf`. The English small model
`vosk-model-small-en-us-0.15` is the reference. The German, French,
Spanish and Russian small models share the layout and are expected to
work; they have not been checked against the wheel yet.

A model that uses a network component this runtime does not implement
is refused when opened, naming the component. The larger English model
`vosk-model-en-us-0.22-lgraph` is refused for that reason: it is a
CNN-TDNN and convolution is not implemented.

## Benchmarks

Reproducible measurements on the public
[Google Speech Commands v2](https://arxiv.org/abs/1804.03209) dataset,
each paired against the stock Vosk wheel on identical audio, live in
[docs/benchmarks/](docs/benchmarks/):

| Page | What it measures |
|---|---|
| [speech-commands.md](docs/benchmarks/speech-commands.md) | Accuracy under a command grammar, agreement with Vosk, first-appearance and endpoint latency, block size, compute, noise, grammar size, and words appearing on audio where nothing was said |
| [partial-trust.md](docs/benchmarks/partial-trust.md) | Whether a partial word will hold: the gap as a calibrated probability, against entropy, energy and the motion of each reading's lead, every rule fitted on the validation split and charged in milliseconds of delay |
| [partial-states.md](docs/benchmarks/partial-states.md) | What the readings say about silence and the next word, on words spliced into streams: end of speech, the preview an extending reading gives, and what the endpoint bound does to real finals |
| [word-times.md](docs/benchmarks/word-times.md) | Whether the word times on finals are the wheel's: every word paired across the engines, exact and within one 30 ms frame |

Headline figures from the Speech Commands page:

| | Vosk 0.3.45 | utter |
|---|---|---|
| Accuracy, 11,005 clips, full grammar | 91.49% | 91.79% |
| First word shown after the clip ends, median / p90 | 40 / 250 ms | 40 / 250 ms |
| First word shown later revised | 12.3% | 12.4% |
| Real-time factor, decode only | 0.018 | 0.016 |

The private gates, run on the first consumer's recordings against the
wheel, are recorded in [docs/gates/](docs/gates/): front-end parity,
per-block partial agreement, word times, first-appearance latency,
alternatives, and endpoints.

## Building and testing

Rust stable, pinned in `rust-toolchain.toml`.

```bash
cargo build --release          # library, cdylib, and the `stream` tool
cargo test                     # unit tests and fixtures, no model needed
UTTER_TEST_MODEL=path/to/vosk-model-small-en-us-0.15 cargo test   # plus decoding tests
```

There are no dependencies to fetch, at runtime, for the build or for
the tests, and CI fails if the lock file ever grows one. The pieces the
decoder needs are small enough to own: the Kaldi binary readers, an
FFT, a blocked single-precision GEMM with AVX2 paths, Cholesky and
conjugate gradient for the i-vector, an OpenFst `ConstFst` reader with
lookahead composition, and JSON.

`target/release/stream` decodes WAV files or a corpus directory and
prints one JSON line per file with every partial and final; it is what
the gates and benchmarks are built on. `stream --help` lists its
options, including diagnostic traces of the decoder's readings.

The benchmark scripts under `scripts/` need Python 3 with `numpy` and
the `vosk` package for the oracle; each script's header says how to run
it.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | The runtime. `recognizer.rs` is the streaming API and JSON, `decoder.rs` the search, `frontend.rs` and `ivector.rs` the features, `compose.rs` the grammar composition, `capi.rs` the C ABI. |
| `src/bin/stream.rs` | The command-line decoder the gates and benchmarks use. |
| `include/utter.h` | The C header. |
| `examples/` | The Rust sketch from this README, built by CI. |
| `docs/` | Reference, guides, benchmarks, gates and decision records; [docs/README.md](docs/README.md) is the index. |
| `tests/` | Integration tests; the decoding ones run when `UTTER_TEST_MODEL` is set. |
| `scripts/` | Gate and benchmark scripts. |
| `docs/benchmarks/` | Public benchmark pages, regenerated by the scripts. |
| `docs/gates/` | Results of the parity gates on private audio. |
| `docs/td/` | Technical decision records: why the runtime is built the way it is. TD-2 is the specification. |
| `usecases/` | The application the runtime was built for. |
| `third_party/` | Vendored code and its notices. |

Design decisions are recorded before code lands, one file per decision
in `docs/td/`. Comments and documents cite those records by anchor
rather than restating them; `docs/td/README.md` explains the
conventions if you want to contribute.

## Third-party code and license

The Kaldi model reader, MFCC and nnet3 forward pass started from
[Vosk-Rust](https://github.com/Reza2kn/Vosk-Rust), Apache-2.0, vendored
as source with its notices kept. Kaldi and OpenFst were read for the
semantics this crate reproduces; no code from either is included.

utter is MIT licensed, see [LICENSE](LICENSE). The crate declares
MIT AND Apache-2.0 because the vendored files keep their own headers
and license.
