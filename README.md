# utter

Pure-Rust streaming speech decoder for Vosk (Kaldi nnet3) models, with
zero dependencies. Feed 16 kHz audio and get the best-path partial after
every block, the distinct hypotheses alive in the search beam, word
times, runtime word-list grammars, and Kaldi endpointing. No C
toolchain, no Docker, no cloud.

## Status

Pre-alpha, and not published anywhere yet. The streaming path is built
and measured: front end, i-vector, chunked network, decoder, partials
with alternatives and word times, endpointing, and the C ABI. It
decodes the reference model against the stock wheel.

What the gates found is recorded in [`docs/gates/`](docs/gates/), one
file per gate, and what a public dataset says is in
[`docs/benchmarks/`](docs/benchmarks/). One thing is open:
segment-for-segment agreement with libvosk sits below the mark the gate
asks for, on near ties in the acoustics. The front end, i-vector
included, matches the wheel's Kaldi within the gate, quiet speech
included.

Decisions live in [`docs/td/`](docs/td/): TD-1 describes the record
system, TD-2 is the runtime specification. The first use case it is
built for is in [`usecases/`](usecases/).

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

## What a partial carries

The stock wheel's partial is a line of text. A host that needs to know
when a word is safe to act on, which rivals the decoder is still
weighing, or whether a sound was speech at all runs its own detectors
beside the decoder. utter puts that evidence in the partial. Fed the
same audio at the same block, where the stock wheel returns

```json
{"partial": "alpha seven"}
```

utter returns the same text and, beside it, the readings still alive in
the beam, the span and loudness and hold of each word, and the room's
own noise floor:

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

The `start` and `end` seconds libvosk emits sit on each word too, left
out here for room.

| field | stock wheel | what a host does with it |
|---|---|---|
| `partial_alternatives` | absent | the distinct readings still alive, ranked; the gap from the first `confidence` to the second is the strongest single sign the word is about to be revised |
| `confidence` | absent | orders the readings; a raw path cost, never a probability |
| `relation` | absent | how the reading stands against the partial as a word sequence: `same`, `prefix`, `extends`, `differs`; which readings to look at for a competing phrase, a missing tail or a possible next word |
| `lead_delta` | absent | how much the reading gained on the field since the previous decoding advance, in nats; null when it was absent then, so a null is a broken history and not a zero |
| `stable_ms` | absent | how long the word has held its place, the hold to wait out before acting on it |
| `energy_dbfs` | absent | the loudness under the word, for telling a spoken word from one read into a quiet room |
| `floor_dbfs` | absent | the room's own noise floor, so a silence gate travels between microphones rather than being a fixed dBFS |
| `start_sample`, `end_sample` | absent | the word's span in samples of the audio fed: one clock, no drift against a second detector |
| `[sil]` | empty string | a best path carrying no word says so, instead of vanishing or being forced to the nearest word |

Stock `partial_result`, where it is turned on, carries `word`, `start`,
`end` and `conf` in seconds; utter's carries the sample span, the energy
and the hold, and adds the ranked readings the stock partial never had.
Turning stock partial words on also costs the stock partial its text on
most blocks: libvosk builds that path from a lattice, which trails the
audio, and over the first stream of
`[[rr:Silence direction and word transitions on a built stream]]` its
partial carried text on 24 blocks of 1374 with partial words on against
510 without, where utter's carried text on 510 either way.

Finals gain the same `energy_dbfs` on each word and `floor_dbfs` beside
the text. What each field means exactly is `[[rr:TD-2#Interface]]`; how a
host reads them for silence and for end of speech is
`[[rr:TD-8#Decision outcome]]`. The trailing `[sil]` entry ends at the
last frame decoded rather than at the last sample fed, so its span lags
the audio by the chunk, a constant on the built stream; a host adding
its own bound to that span is adding it to a clock that runs behind.

**How much to trust a partial word.** A word in a partial may still be
revised as more audio arrives; on the Speech Commands split the first
word shown is later revised 12% of the time. What predicts it is not the
word's own `confidence`, which is a raw path cost, but its lead over the
next reading, and the API already returns that lead in
`partial_alternatives`. The decoder's costs are log-likelihoods in nats,
so the gap between the top two confidences is a two-way softmax, and its
sigmoid is a usable probability that the leading word will hold:

```python
import json, math

def trust(partial_json):
    """P that the top reading holds, from its lead over the next reading."""
    alts = json.loads(partial_json).get("partial_alternatives") or []
    if len(alts) < 2:
        return 1.0                                    # nothing else is close
    gap = alts[0]["confidence"] - alts[1]["confidence"]   # nats, >= 0
    return 1.0 / (1.0 + math.exp(-gap))                   # sigmoid(gap)

# act on the leading word only once the reading is clear of its rival
p = rec.PartialResult()
if trust(p) >= 0.9:                                   # a lead of about 2.2 nats
    act(json.loads(p)["partial"])
```

That sigmoid is a Viterbi approximation over the two leading beam
readings, not a lattice posterior, but it is well calibrated in practice:
the derivation and a benchmark that holds the predicted survival against
the measured survival over the testing split are in
[`docs/benchmarks/partial-trust.md`](docs/benchmarks/partial-trust.md).

The second read is the entropy of the readings' softmax, `-sum(p*log p)`
over `softmax(confidences)`, and it costs nothing beyond the partial in
hand: it ranks first sightings as well as the gap does and keeps ranking
them inside every bucket of the gap, where the gap itself is held fixed,
`[[rr:The same signals at equal gap]]`. It is always defined, where the
motion below usually is not. The runtime does not report it, because it
is a function of the confidences one partial already carries.

The third read, the motion `lead_delta`, is not a trust read. Measured
on the runtime's history over every surviving group, which carries a
delta for the leading reading on nine sightings in ten, the motion does
not improve on the gap at the first sighting or at the bar above, and
within a bucket of the gap it barely ranks at all
`[[rr:What the motion figures say]]`. What does buy a decision here is
waiting: one advance costs 240 ms and by then the revision is usually
already on the page `[[rr:Holding one advance]]`. Read the motion for
the state reads below instead, and for the one trust-shaped case it does
answer: a lead and its motion disagreeing in sign, a reading that leads
but is losing, is the case to wait out `[[rr:Leading but losing]]`.

## Five reads off one partial

Each is a line, and the bar in it is the host's. The figures behind
them are on the two benchmark pages, which is also where the reads that
did not survive measurement are recorded.

```python
alts = json.loads(p)["partial_alternatives"]
top  = alts[0]

# 1. trust: will the leading word hold? (sigmoid(gap), above)
trust = 1 / (1 + math.exp(-(top["confidence"] - alts[1]["confidence"])))

# 2. a word is coming: a reading that extends the partial and is gaining
coming = [a for a in alts if a["relation"] == "extends"
          and (a["lead_delta"] or 0) > 0]
# its extra word is the text past the partial's:
next_word = coming[0]["text"][len(top["text"]):].split()[:1] if coming else []

# 3. kept silence: the empty reading leads and its lead is not moving
quiet = top["text"] == "[sil]" and abs(top["lead_delta"] or 0) < 0.5

# 4. the last word may not be there: the best reading without it
prefix = next((a for a in alts if a["relation"] == "prefix"), None)
doubt  = prefix["confidence"] - top["confidence"] if prefix else None

# 5. end of speech: the trailing [sil] entry's span, against your bound
tail = json.loads(p)["partial_result"][-1]
ended = tail["word"] == "[sil]" and \
        (tail["end_sample"] - tail["start_sample"]) / 16000 > 0.5
```

Read 2 before a first word is every word-carrying reading, since they
all extend the empty one; the empty reading's *own* velocity is not the
read, because it costs two alarms a second in silence that is merely
being kept, where the extending reading costs none
`[[rr:The two crossings, as fitted signals]]`. Whether either foretells
a word at all is exploratory and measured on spliced words, not on
recorded speech `[[rr:Calling a word before it arrives: exploratory]]`;
what *is* measured is the preview: the coming word's reading is often in
the beam an advance before the partial grows, and its identity is right
about one time in six `[[rr:Preview accuracy by the lead still to run]]`.

Read 3 is silence held in the beam as a lead that hovers rather than
grows `[[rr:What the readings do in silence that is being kept]]`.
Read 4 is null evidence and nothing more: a `prefix` reading competes by
omitting the partial's tail, it is not a posterior for a null at a word
position, and a competitor that is absent from the list is unknown
evidence rather than evidence of absence
`[[rr:TD-9#Every reading names its relation to the partial]]`.
Read 5 is TD-8's clock, and it stays the clock: the motion trades
mistaken pauses against finishes missed rather than beating it
`[[rr:The end-of-speech confusion]]`. A null `lead_delta` in any of
these is a history the reading does not have; treat it as unknown, never
as a zero.

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

None today, at runtime or for the build or for the tests: `cargo
build` fetches nothing and CI fails if the lock file grows a second
crate. The standard library is the foundation and `std::arch` supplies
the SIMD paths. Every piece the decoder needs is small enough to own:
the Kaldi binary readers, a real FFT, a blocked
single-precision GEMM, Cholesky and conjugate
gradient for the i-vector, an OpenFst `ConstFst` reader with the
`olabel_lookahead` add-on, composition and trimming, and JSON. The
policy and the two exceptions it reserves are `[[rr:TD-2#Dependency
policy]]`; neither exception is built, so the crate stands at none of
any kind.

## Interface

A Rust library, plus a `cdylib` with a C ABI that mirrors it so hosts
without Rust get the same surface. Sketch:

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

That sketch is [`examples/readme_sketch.rs`](examples/readme_sketch.rs),
built by CI, so it cannot drift from the interface.

Results are JSON strings, the same from the library and the C ABI. They
keep the key layout of libvosk's `PartialResult` and `Result`, so a host
that already parses Vosk output keeps parsing, and add what a
partials-first host asks for: `partial_alternatives` ranked by
confidence, `start_sample` and `end_sample` beside Kaldi's seconds,
`energy_dbfs` under each word of a partial and of a final, `stable_ms`
for how long a partial word has held, `floor_dbfs` for the noise floor
of the audio fed, and `[sil]` where the reading carries no word.

What those added fields are and how a host reads them for trust, for
silence and for end of speech is in "What a partial carries" above and
in `[[rr:TD-8#Decision outcome]]`.

## Model compatibility

Any Vosk small model with the standard layout: `am/final.mdl`,
`graph/HCLr.fst`, `graph/words.txt`, `graph/disambig_tid.int`,
`graph/phones/word_boundary.int`, `ivector/`, `conf/mfcc.conf`,
`conf/model.conf`. The English small model is the reference; the
German, French, Spanish and Russian small models are believed to share
the layout and are unchecked until each is decoded against the wheel.

A model whose network uses a component this runtime does not implement
is refused when it is opened, naming the component, rather than
decoding with that layer passed through. The larger English model,
`vosk-model-en-us-0.22-lgraph`, is refused on that rule: it is a
CNN-TDNN and the convolution is not implemented.

## Verification

The stock `vosk` wheel is the oracle, fed the same blocks from the same
audio. Each gate has a file under [`docs/gates/`](docs/gates/) recording
what ran, against which oracle, and the figures. The gates, in order:

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

The gate corpora are the first consumer's private recordings, so the
files name them only by date. Anything reproducible by a reader is a
benchmark instead, under [`docs/benchmarks/`](docs/benchmarks/), on a
public dataset with a published licence.

## Third-party code

The Kaldi model reader, MFCC and nnet3 forward pass start from
[Vosk-Rust](https://github.com/Reza2kn/Vosk-Rust), Apache-2.0, vendored
as source with its notices kept. Kaldi and OpenFst were read for the
semantics this crate reproduces; no code from either is included.

## License

MIT. See [LICENSE](LICENSE). Vendored Apache-2.0 files keep their own
headers.
