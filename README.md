# utter

[![CI](https://github.com/pyscape/utter/actions/workflows/ci.yml/badge.svg)](https://github.com/pyscape/utter/actions/workflows/ci.yml)
[![docs.rs](https://img.shields.io/docsrs/utter)](https://docs.rs/utter)
[![crates.io](https://img.shields.io/crates/v/utter)](https://crates.io/crates/utter)
[![PyPI](https://img.shields.io/pypi/v/utterpy)](https://pypi.org/project/utterpy/)
[![MSRV](https://img.shields.io/crates/msrv/utter)](Cargo.toml)
[![Coverage](https://img.shields.io/coverallsCoverage/github/pyscape/utter?branch=main)](https://coveralls.io/github/pyscape/utter?branch=main)
[![CodeQL](https://github.com/pyscape/utter/actions/workflows/codeql.yml/badge.svg)](https://github.com/pyscape/utter/actions/workflows/codeql.yml)
[![Scorecard](https://api.scorecard.dev/projects/github.com/pyscape/utter/badge)](https://scorecard.dev/viewer/?uri=github.com/pyscape/utter)
[![Dependencies](https://deps.rs/repo/github/pyscape/utter/status.svg)](https://deps.rs/repo/github/pyscape/utter)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/14627/badge)](https://www.bestpractices.dev/projects/14627)
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
- [Quick start](#quick-start)
- [Where next](#where-next)
- [Benchmarks](#benchmarks)
- [Building and testing](#building-and-testing)
- [License](#license)

## Why utter

### Fast command recognition

Vosk is the best small offline recognizer for a closed vocabulary, and
it is fast: under a grammar of a few dozen words, a spoken word appears
in the partial result a median 40 ms after the word ends. The [Speech
Commands measurements](docs/benchmarks/speech-commands.md#full-grammar-accuracy)
show that latency; the [design measurements](docs/td/0002-the-streaming-runtime-for-vosk-models.md#context-and-problem-statement)
put streaming ONNX models under the same grammar at about half a second.

### Know when speech falls outside your phrase list

Vosk phrase lists can include `[unk]`, giving speech outside the list a
named reading instead of the closest listed phrase. The stock Vosk
recognizer API does not let a host tune how strongly that path competes.
utter exposes that control in Rust as `RecognizerOptions::unknown_cost`:
lower it to return `[unk]` more often, raise it to prefer commands, or
leave it unset to omit `[unk]`. See the [API reference](docs/reference/vosk.md#what-utter-adds)
and [unknown-word benchmark](docs/benchmarks/speech-commands.md#twelve-class-ten-commands-plus-the-unknown-word-symbol).

### Tell silence from speech

utter reports `[sil]` when the best path is silence and `[speech]` when
it has entered word phones without reaching a word label. It also reports
trailing-silence duration and the room's noise floor, so a host can tell
a pause or quiet room from a command without running a second detector.
See the [result fields](docs/reference/results.md#the-partial-result) and
[silence measurements](docs/benchmarks/speech-commands.md#the-floor-gate-a-host-applies).

### Build and ship on every platform

But Vosk is a C++ wrapper around Kaldi, and Kaldi does not build on
Windows without a Docker cross-compile. Anyone who needs one more field
out of the decoder than the stock wheel exposes ends up maintaining a
fork they cannot easily ship. See the [first consumer's build
constraint](usecases/live-command-dictation.md#4-what-goes-wrong-today).

### Make streaming decisions from the decoder

utter runs stock Vosk models in Rust for applications that must act before
speech ends. It reports the current reading and alternatives still in
contention; word timing, energy, and stability; silence, unlabelled
speech, and the room's noise floor; and the end of speech. The [results
reference](docs/reference/results.md#the-partial-result) describes every
field, and the [partial-state benchmarks](docs/benchmarks/partial-states.md)
measure how those readings behave over time.

### Verified against Vosk

Our public Speech Commands benchmark shows that utter matches the stock
Vosk wheel: 91.8% against 91.5% accuracy on 11,005 clips, finals
agreeing on 99.2% of them, and the same 40 ms median first-appearance
latency. We run both engines on the same audio and record every partial
and final so that we can demonstrate parity at each decoding step. See
the [benchmarks](#benchmarks).

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

The [releases](https://github.com/pyscape/utter/releases) carry the
library, the `stream` tool and the header for Linux, Windows and
macOS. The macOS build is universal and signed ad hoc, not notarized;
a copy that came through a browser is quarantined until
`xattr -d com.apple.quarantine libutter.dylib` clears it. Every archive
carries a build provenance attestation:

```bash
gh attestation verify utter-linux-x86_64.tar.gz -R pyscape/utter
```

The release also attaches the attestation's Sigstore bundle,
`utter-vX.Y.Z.sigstore.json`, which names every archive and verifies
without reaching GitHub:

```bash
gh attestation verify utter-linux-x86_64.tar.gz --bundle utter-v0.0.3.sigstore.json --owner pyscape
```

## Where next

| You are | Start with |
|---|---|
| Using the `vosk` package today | [Coming from Vosk](docs/reference/vosk.md): every method, where it stands, and what is added. |
| A Rust engineer | `cargo doc --open`. The quick start is on the crate page and each method that returns JSON links to the results reference. |
| Reading the JSON | [Results](docs/reference/results.md): every document the recognizer returns and every key in it. |
| Building on the partials | [Reading a partial](docs/guide/reading-a-partial.md): the trust read as a calibrated probability, the next-word preview, silence and the doubtful last word, each a line of Python. Then [Ending speech sooner](docs/guide/ending-speech-sooner.md): the endpoint bound and its veto, with what they buy and cost. |

The [docs index](docs/README.md) lists every directory.

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
| [quiet-onsets.md](docs/benchmarks/quiet-onsets.md) | What an endpoint bound does to a word whose first frames sit near the floor, and what the floor margin costs there |
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

## License

utter is MIT licensed. Vendored code retains its own Apache-2.0 headers
and notices; see [LICENSE](LICENSE).
