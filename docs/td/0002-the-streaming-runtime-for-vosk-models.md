# TD-2: The streaming runtime for Vosk models - scope, stages, interface, dependencies and gates

- Status: Proposed
- Date: 2026-09-11
- Tags: decoder, kaldi, vosk, partials, alternatives, grammar, endpointing, dependencies

## Context and problem statement

The first consumer of this library (`usecases/live-command-dictation.md`)
dictates short commands from a closed vocabulary and acts on each word
while the speaker is still talking. It runs the stock Vosk wheel and a
fork of libvosk that exposes what the stock partial throws away:
alternative readings, the null hypothesis, sample-indexed word
intervals, energy under each word, and readiness. The fork costs a
Kaldi cross-compile in a container to reach Windows, or a remote host
over SSH, and its alternatives came from a lattice that trailed the
audio by about a second.

Measured on the consumer's recordings, Vosk's partials are the right
thing to keep: a spoken word appears in the partial a median 40 ms after
its acoustic end and at the 90th percentile 190 ms, against about half a
second for every ONNX streaming model tried under the same grammar. The
decoder architecture is worth keeping; the build and the missing
outputs are not. What is needed is a runtime that reads a stock Vosk
model directory, reproduces Kaldi's partial behaviour, exposes the
beam, and builds with `cargo build` alone.

Two facts bound the design. Vosk-Rust (Apache-2.0) already
reads the Kaldi model, runs the chain network verified at 7e-6 against a
numpy reference, does Kaldi-exact MFCC, and decodes best path over a
graph, but only on whole utterances. And a Vosk small model's graph is
stored as OpenFst type `olabel_lookahead`, whose add-on carries the
relabeling the grammar side must be transformed with.

## Decision drivers

- Partials as fast as libvosk's: the same median and tail on the same
  recordings.
- Alternatives current to the last decoded frame, distinct by word
  sequence, the silent reading preserved when it leads and announced as
  silence rather than delivered as an empty string or a forced
  vocabulary word.
- Word intervals, energy and the silence state from the one loop that
  decodes; no second detector on a second clock.
- One decode: the stream is the answer, and end of speech is an event
  with a sample position, not a second result to reconcile.
- The stock Vosk model directory as the only input; five shipped
  languages, no conversion step.
- Zero runtime dependencies and no C toolchain, so the build is one
  `cargo build` on every platform for the life of the library.
- The stock wheel as oracle: every stage is checked against libvosk on
  the consumer's recordings before the next stage is built.

## Considered options

- **Keep the libvosk fork and finish its evidence serialiser.** Every
  change is a container cross-compile, and the lattice n-best it builds
  on is a second late. Rejected.
- **A different engine on ONNX models with a compiled grammar graph.**
  Works, and grammar-constrained decoding was proven on the consumer's
  audio, but partials arrive about half a second later by design, and
  grammar-capable streaming models exist for a fraction of the shipped
  languages. Rejected for this consumer.
- **A full port of libvosk.** libvosk is 2,000 lines of glue over
  Kaldi; a port is a port of Kaldi, most of which this consumer never
  exercises. Rejected.
- **A runtime that reimplements only the partials path, starting from
  Vosk-Rust, with beam n-best in place of lattices.** Taken.

## Decision outcome

### Scope

In: Kaldi-exact feature extraction (MFCC, online CMVN, splice, LDA) and
the online i-vector extractor with a zero-i-vector mode, including
libvosk's silence weighting of its statistics; the nnet3 chain
acoustic model in streaming chunks; the runtime grammar compiled to the
bigram libvosk compiles and composed with the model's graph; a
frame-synchronous token-passing decoder with Kaldi's pruning semantics;
partials as the best-path traceback after every block, with word
intervals; partial alternatives as the distinct word sequences alive in
the beam; per-word energy and silence state; Kaldi's endpoint rules;
finals for hosts that want them; the output shapes libvosk emits,
extended.

Out: lattices, lattice determinization, minimum Bayes risk confidences
and lattice n-best; decoding against the full language model
`graph/Gr.fst`; RNNLM and constant-ARPA rescoring; speaker vectors;
batch and GPU recognizers.

### Inputs: the model directory

A Vosk small model, about 68 MB, with the layout every Vosk small model
shares:

| File | Format | Used for |
|---|---|---|
| `am/final.mdl` | Kaldi binary: TransitionModel then nnet3 Nnet | phone and pdf per transition id; the network |
| `graph/HCLr.fst` | OpenFst binary, type `olabel_lookahead`, arc `standard` | H, C, L composed, word side relabeled, plus the lookahead add-on |
| `graph/Gr.fst` | OpenFst binary, type `ngram` | its arcs are not read; its attached output symbol table is the word table, 152,217 symbols, already in the relabeled label space of `HCLr.fst` |
| `graph/disambig_tid.int` | text, integers | transition-id-space disambiguation symbols erased after composition |
| `graph/words.txt` | absent in the small models | libvosk falls back to the table attached to `Gr.fst`, and so does the runtime |
| `graph/phones/word_boundary.int` | `phone type` per line | word alignment on the best path: begin, end, internal, singleton, nonword |
| `ivector/final.dubm`, `final.ie`, `final.mat`, `global_cmvn.stats` | Kaldi binary | Gaussian selection, i-vector estimation, LDA, CMVN start-up |
| `ivector/online_cmvn.conf`, `splice.conf` | Kaldi option files | CMVN options (defaults) and splice context 3 and 3 |
| `conf/mfcc.conf`, `conf/model.conf` | Kaldi option files | below |

### Inputs: configuration

From `conf/mfcc.conf` of the reference model: sample frequency 16000,
no energy, 40 mel bins, 40 cepstra, low frequency 20 Hz, high frequency
7600 Hz. Every other frame option is Kaldi's default: 25 ms window,
10 ms shift, pre-emphasis 0.97, DC removal, Povey window, FFT rounded to
512 points, snip edges, cepstral lifter 22, dither 1.0. Dither is an
option and parity tests run both sides at 0.

From `conf/model.conf`: min-active, max-active, beam, acoustic scale,
frame subsampling factor, frames per chunk, endpoint silence phones and
endpoint rule thresholds are read from the file; the stock file carries
min-active 200, max-active 3000, beam 10.0, acoustic scale 1.0,
subsampling 3, silence phones 1 to 10, and rules 2, 3 and 4 at 0.5, 1.0
and 2.0 s, and does not set frames per chunk, so the stock model runs
Kaldi's default of 24 input frames per chunk; a host may ship a sibling configuration with a smaller chunk and shorter trailing-silence rules, and the runtime takes whatever the file says. Kaldi defaults fill the rest: rule 1 at 5.0 s with
no speech required, rule 5 at 20 s of utterance, rule 2 relative cost
2.0, rule 3 relative cost 8.0, rule 4 unbounded.

Kaldi defaults libvosk never overrides and the runtime reproduces:
online CMVN window 600 frames, speaker frames 600, global frames 200,
mean normalization only; i-vector period 10, 5 selected Gaussians,
minimum posterior 0.025, posterior scale 0.1, maximum remembered frames
1000, 15 conjugate-gradient iterations, most recent i-vector used for
every frame; libvosk sets max count 100.

### Inputs: the grammar

A list of strings, bounded at fewer than 300 distinct words; a larger
grammar is refused at construction. libvosk tokenizes each on spaces, drops words absent
from the word table with a warning, and treats each string as one
sentence for the bigram estimator. The runtime does the same, including the
warning. A grammar may include the model's unknown-word symbol, and
silence is always a reading that names itself (see "Silence and unknown
speech announce themselves").

### Interface

A Rust library and a `cdylib` with a C ABI mirroring it. The surface,
named so that a host written against libvosk's C API or its Python
bindings maps one to one:

- `Model::open(path)`; `Recognizer::new(&model, sample_rate, grammar)`.
- `set_words`, `set_partial_words`, `set_alternatives(n)`,
  `set_max_alternatives(n)`, `set_log_callback`.
- `accept(&[i16]) -> Step`: 16-bit mono PCM at the recognizer's rate;
  `Step` carries whether an endpoint fired and at which sample.
- `partial()`, `alternatives()`, `result()`, `final_result()`: typed
  in Rust, JSON text through the C ABI.

JSON, partial, keys in this order:

```json
{"partial": "alpha seven",
 "partial_alternatives": [
   {"text": "alpha seven", "confidence": -1234.5,
    "result": [{"word": "alpha", "start": 12.30, "end": 12.63,
                "start_sample": 196800, "end_sample": 202080}, ...]},
   {"text": "[sil]", "confidence": -1239.1,
    "result": [{"word": "[sil]", "start": 11.90, "end": 12.98,
                "start_sample": 190400, "end_sample": 207680}]}],
 "partial_result": [{"word": "alpha", "start": 12.30, "end": 12.63,
                     "start_sample": 196800, "end_sample": 202080,
                     "energy_dbfs": -31.2, "stable_ms": 120},
                    {"word": "[sil]", "start": 12.63, "end": 12.98,
                     "start_sample": 202080, "end_sample": 207680,
                     "energy_dbfs": -58.4, "stable_ms": 350}, ...]}
```

`partial_alternatives` is present when alternatives were requested; rank
0 is always the best path and its text equals `partial`.
`partial_result` is present when partial words are on. A reading is
never an empty string: a best path with no word on it reads `[sil]`
(see "Silence and unknown speech announce themselves"). Final and
`alternatives` shapes keep libvosk's key layout; `conf` on final words
is emitted as 1.0 and documented as a constant, since no lattice
posterior exists.

Time base: `start` and `end` in seconds are `samples_round_start /
sample_rate + (frame_offset + output_frame) * 0.03`, libvosk's formula;
`start_sample` and `end_sample` are integer samples of the audio fed
since the recognizer was created and are the values a host should use.

Lifecycle: construction compiles the grammar and composes the graph.
`accept` feeds the feature pipeline in steps of 0.2 s of samples,
advancing the decoder after each step, then evaluates the endpoint
rules. After an endpoint the next `accept` starts a new utterance on the
same feature pipeline, so i-vector statistics carry across utterances,
with `frame_offset` advanced. The feature pipeline is rebuilt, and
`samples_round_start` advanced, after `final_result` or once
`frame_offset` exceeds 20000 output frames, as libvosk does. No
synthetic silence is ever fed by the runtime.

### Front end: MFCC

Kaldi's `OnlineMfcc` over the configuration above. Reference: Vosk-Rust
`src/mfcc.rs`, verified at a maximum delta of 1e-3 against
`torchaudio.compliance.kaldi`. Frames are emitted as soon as their 25 ms
window is complete; the last partial window is dropped. The 512-point
real FFT is the runtime's own, written after Kaldi's
`SplitRadixRealFft` so that summation order matches.

### Front end: the i-vector branch

Kaldi's `OnlineIvectorFeature` on the MFCC frames: online CMVN (mean
only) primed from `global_cmvn.stats`; splice 3 and 3, then LDA
`final.mat`; every 10 frames, the 5 best Gaussians of `final.dubm` per
frame, posteriors floored at 0.025 and scaled by 0.1, zeroth and first
order statistics accumulated, scaled down once their count passes 100,
frames beyond 1000 forgotten; the i-vector estimated from the statistics
with `final.ie` by 15 conjugate-gradient iterations, the most recent
estimate handed to every network chunk. The network input itself is the
raw MFCC frame; online CMVN applies only inside this branch.

The statistics are silence-weighted as libvosk configures Kaldi's
`OnlineSilenceWeighting`: weight 1e-3 for the frames the current
best-path traceback labels with a silence phone, 1.0 for the rest,
`max_state_duration` -1 (off). Before every decoder advance the
traceback is recomputed and delta weights are produced over the frames
ready, with `frame_offset * 3` as the first input frame of the
utterance, and handed to the i-vector feature, which applies them as
its statistics reach each frame; a frame already counted is corrected
by the difference, retroactively over the last hundred decoder frames
when the traceback changes. The weighting object is fresh for every
utterance, so the frames of a new utterance carry the silence weight
until a traceback exists. Frames newer than the traceback take the most
recent weight.

Vosk-Rust implements the batch extractor at 0.999 correlation to Kaldi's
`ivector-extract` and reports that Kaldi's online tool has an
extraction-order behaviour it did not reproduce. The runtime carries a
mode switch: `faithful` (the above) and `zero` (a zero vector of the
extractor's dimension). Gate G0 (`docs/gates/g0-ivector-mode.md`)
decides it: faithful beats zero by 2.34 word error points against
libvosk's finals on the consumer's corpus, so faithful ships and zero
remains a measurement mode.

### The network

Reader for Kaldi's binary `final.mdl`: the TransitionModel followed by
the nnet3 `Nnet` with its components and descriptor graph. Reference:
Vosk-Rust `src/kaldi_io.rs`, `src/transition_model.rs`, `src/nnet3.rs`.

Streaming execution follows Kaldi's `DecodableNnetSimpleLooped`: left
and right context are read from the descriptor graph (26 left and 14
right input frames for the reference model, whose output is 2,192 pdfs
over 10,014 transition ids); input frames are consumed in chunks of
frames-per-chunk; output frame t needs input frames up to 3t plus the
right context and is not emitted before those frames exist; the output
is the chain network's log-likelihood per output frame scaled by the
acoustic scale, indexed by pdf.

The i-vector a chunk uses is the newest estimate available at the
moment the chunk is computed, as `DecodableNnetLoopedOnlineBase::AdvanceChunk`
takes it: the i-vector feature's frame at the smaller of the most
recent input frame ready and the last i-vector frame ready, where
i-vector frames ready is MFCC frames ready minus the splice right
context of 3. Which estimate that is therefore depends on the feed
step: 0.2 s inside libvosk's `AcceptWaveform`, but a host that calls
`accept` per 40 ms block advances in 40 ms steps, and the runtime
reproduces the schedule of whichever step the host uses.

Matrix multiplication is the cost centre: 40-dimensional input, a
15 MiB network, one forward per 30 ms of audio. The kernel is the
runtime's own: a blocked single-precision GEMM with the inner product
unrolled over eight accumulators so the compiler vectorizes it on
stable Rust, and explicit AVX2 with FMA and NEON paths behind runtime
feature detection through `std::arch`, scalar fallback everywhere else.

### The graph: reading HCLr.fst

The header names type `olabel_lookahead`, OpenFst's
`StdOLabelLookAheadFst`: a `ConstFst` body followed by an `AddOnPair`
whose input-side entry is absent and whose output-side entry is a
`LabelReachableData` record: the reach-input flag, the final label,
the label-to-index map, and the interval sets per state. The runtime
carries its own reader for the OpenFst binary header and the `ConstFst`
body and its own parser for the add-on per OpenFst's
`label-reachable.h`.

The graph was written by OpenFst 1.6.7, whose `olabel_lookahead` flags
omit `kLookAheadKeepRelabelData`, so the label-to-index map in the
record is empty: the relabeling was applied to the graph's output
labels when it was built and not kept. The word table that matters is
the symbol table attached to `Gr.fst`, which is already in that
relabeled space and matches `HCLr.fst`'s output labels one to one, and
it is what libvosk uses for its word symbols when no `words.txt` is
present. The runtime therefore reads `Gr.fst`'s header and symbol table
and never relabels G. The interval sets are the part of the add-on the
runtime uses, for composition (next section).

### The graph: the grammar as a bigram

A port of libvosk's `src/language_model.cc` (211 lines, a modification
of Kaldi's `chain/language-model.cc`): order 2, discount 0.5. Each
grammar string is a sentence of word ids bounded by sentence start and
end; counts per history state; each seen bigram gets an arc weighted
`-log(count * discount / state_count)`; each state with a backoff
history gets an epsilon arc weighted `-log(1 - discount)` to the backoff
state; sentence end goes to a final state. An exact port: the arc
weights decide ties between grammar words, and the parity gates compare
against libvosk.

### The graph: composition

libvosk composes with OpenFst's default `ComposeFst` over the lookahead
matcher the graph type carries, which selects the lookahead compose
filter with weight and label pushing, then maps every input label in
`disambig_tid.int` to epsilon, lazily with a 32 MiB cache. Lookahead
exists to keep a composition from expanding paths that can never reach
a G arc, and the determinized HCL makes that expansion the common case:
word labels are delayed, so nearly the whole graph sits behind output
epsilons and a plain eager composition copies it once per G state,
which passed two million states on a grammar of a few dozen words before
it was stopped.

The runtime therefore composes eagerly at construction with the
standard epsilon-sequencing filter (G's backoff arcs are epsilons on its
input side) and prunes with the add-on's interval sets: a state pair is
not expanded when no output label the G state accepts is reachable from
the HCLr state. This is the reachability test the lookahead matcher
performs, applied at build time. The disambiguation labels are erased
and states that cannot reach a final state are trimmed; the connected
result is identical to the unpruned composition. A grammar of a few dozen
words composes to about twelve thousand states and thirty thousand arcs
in single-digit milliseconds; the supported bound is fewer than 300
words. The result
is a plain vector FST the decoder walks directly.

This yields the same paths and path weights as libvosk's graph, not the
same weight placement: the lookahead filter also pushes G's weights and
labels earlier along a word's arcs, which lets the beam discard a losing
word sooner. Best paths are identical; pruning near the beam edge can
differ, which gate G2 measures. Should G2 fail on that account, weight
pushing after eager composition is the fallback, a single pass over the
trimmed graph. The runtime refuses a grammar whose composition exceeds a
configured state count rather than silently taking seconds; a
large-vocabulary G, which would need lazy composition, is out of scope.

### The decoder: search

A frame-synchronous token-passing Viterbi over the composed graph,
reproducing the pruning semantics of Kaldi's `LatticeFasterOnlineDecoder`
without lattice links. Per output frame: emitting arcs add the graph
weight plus `-acoustic_scale * loglike[pdf(transition_id)]`; then
non-emitting arcs propagate within the frame in cost order until no
token improves. Pruning follows Kaldi's `GetCutoff`: the beam around the
best token, tightened when more than max-active tokens survive and
relaxed to admit at least min-active, with beam-delta 0.5 and the
adaptive beam Kaldi carries between frames; costs are stored relative
to a per-frame offset. Each token carries its accumulated cost and a
pointer to a word link record (the last word emitted on its path, the
output frame of that emission, the previous record), reference counted,
so a traceback is a walk over words. Transition ids are kept per frame
on the token chain for alignment. Final costs are added only when a
final is requested: a token in a final state pays the state's final
weight; if no active token is final, the best non-final token wins.

### The decoder: partials

After each `accept` step, the partial is the word sequence on the best
token's word links, without final costs, exactly libvosk's
`PartialResult` without partial words. This is the behaviour the
consumer measured and the runtime must reproduce (G3).

### The decoder: word intervals

libvosk's word times come from minimum Bayes risk on a word-aligned
lattice. The runtime aligns the best path instead: the phone sequence
from the transition ids, segmented by `word_boundary.int` (a word runs
from a `begin` or `singleton` phone to the next `end` or `singleton`;
`nonword` phones belong to no word), so each word gets the first and
last output frame of its phones. Intervals are reported in integer
samples of the fed audio and in libvosk's seconds. This is the
alignment `WordAlignLattice` performs on a linear lattice.

### The decoder: per-word energy and silence

From the same loop: the RMS energy in dBFS of the PCM under each word's
interval, computed from the samples the runtime was fed, and the run of
trailing silence phones on the best path. A word also carries
`stable_ms`, how long its position in the partial has held unchanged. A
host that wants readiness (a hold per word and a quiet floor) evaluates
it from these fields; the runtime supplies the evidence and does not
withhold words on its own.

### The decoder: partial alternatives

After each step, with alternatives requested:

1. Take every surviving token at the current frame after pruning.
2. Group tokens by word sequence read from their word links; the empty
   sequence is a group, and a sequence ending in the unknown-word
   symbol is a group when the grammar included it.
3. Score each group by its cheapest token's accumulated cost; the
   confidence reported is the negation of that cost, the fork's
   definition, so a host that reads the fork's output reads this
   unchanged.
4. Rank ascending by cost, emit up to n groups, rank 0 being the best
   token's group and therefore equal to `partial`.
5. With partial words on, each group carries its words aligned from its
   cheapest token's path.

Two properties distinguish this from the fork's lattice n-best: the
hypotheses are current to the last decoded frame, and grouping is by
word sequence, so two alignments of the same words never appear as two
readings. The beam is a bounded sample of the search space, not a
lattice: hypotheses pruned before the current frame are gone. The
consumer's census scripts are rerun on the runtime to state how many
partials carry a one-word contest.

### Silence and unknown speech announce themselves

The runtime may emit during silence provided what it emits says it is
silence. Two facts the decoder already holds make that exact. Kaldi's graph carries silence
as phones of the silence set (`endpoint.silence-phones`, 1 to 10 in
the reference model) that belong to no word, and the model's word table
carries an unknown-word symbol, `[unk]` in Vosk models, that a grammar
may include so out-of-vocabulary speech decodes to it instead of to the
closest word. They are different things and are reported differently.

- **`[sil]` is the reading of a best path that carries no word.** The
  `partial` text is then `[sil]`, never the empty string and never a
  vocabulary word the beam happened to prefer. In the alternatives, the
  group whose word sequence is empty is labelled `[sil]` and ranked on
  its cost like any other group, so on silence rank 0 reads `[sil]` and
  the rivals stand behind it.
- **Every run of silence phones on the best path that lies between
  words or after the last word is an entry in `partial_result`** with
  the token `[sil]`, its interval in samples and seconds, its energy in
  dBFS and its `stable_ms`, so a pause has a duration the host can
  read. The `partial` text lists words only, plus `[sil]` when there is
  no word; silence entries between words appear in the word list, not
  in the text, so a host that parses the text as words sees exactly the
  words.
- **`[unk]` is a word.** When the grammar includes the model's
  unknown-word symbol it decodes, ranks, aligns and reports like any
  other word, with its interval and energy; a reading that is only
  `[unk]` reads `[unk]`, not `[sil]`. `Recognizer::new` offers an
  option to add the symbol to the grammar with a cost offset; it is off
  by default so that the parity gates run on grammars identical to
  libvosk's, and the host decides whether to turn it on.
- **Bracketed tokens are never words to the host.** A host that acts on
  words treats `[sil]` as nothing offered and `[unk]` as a sound that
  was not a command. The parity gates G2 and G4 compare texts with
  bracketed tokens removed, since libvosk emits none.

### The decoder: endpointing

Kaldi's `EndpointDetected` over the best-path traceback without final
costs: trailing silence is the run of frames at the end of the path
whose transition id maps to a silence phone, times 0.03 s; the utterance
length is frames decoded times 0.03 s; relative cost is the best
final-state token's cost minus the best token's cost, infinite when no
active token is final. The five rules are evaluated after every step,
and any rule firing is reported in `Step` with the sample position of
the last decoded frame.

### The decoder: finals

`result` after an endpoint and `final_result` at stream end take the
best path with final costs, align its words, and emit libvosk's final
shape. With `set_max_alternatives(n)`, n > 1, the `alternatives` array
comes from the beam grouping at the final frame with final costs added:
a beam n-best, not libvosk's lattice n-best. A host that acts on the
stream need never read a final; nothing is reported only there.

### Packaging

The library is this repository. It offers a `cdylib` target with the C
ABI. No C, C++ or system library anywhere in the build. The Vosk-Rust
modules the runtime starts from are vendored as source under their
Apache-2.0 licence with attribution, not pulled as a crate. `accept`
never holds a lock a caller can observe; one recognizer is used from
one thread at a time, as libvosk requires. A `Model` parses `final.mdl`
and `HCLr.fst` once; a `Recognizer` shares them and owns only its
composed graph and decoder state.

### Dependency policy

The crate has zero runtime dependencies: `cargo build` of the library
fetches nothing, and `Cargo.lock` lists the crate alone. The standard
library is the only foundation; `std::arch` provides the SIMD
intrinsics. Every piece the decoder needs is small enough to own, and
owning it keeps the build a single `cargo build` on every platform for
the life of the library.

| Need | Would have been | Instead |
|---|---|---|
| Kaldi binary model and matrix reading | Vosk-Rust's `kaldi_io.rs` | vendored, already dependency-free |
| 512-point real FFT for MFCC | `rustfft` | own split-radix real FFT after Kaldi's |
| single-precision GEMM for the network | `matrixmultiply` | own blocked kernel, `std::arch` SIMD |
| Cholesky and conjugate gradient for the i-vector | `nalgebra` or `ndarray` | own, on plain `Vec<f32>` and `Vec<f64>` |
| OpenFst binary reading, composition, trimming | `rustfst` | own `ConstFst` reader, add-on parser, vector FST, epsilon-sequencing compose, connect |
| JSON output and grammar input | `serde_json` | own writer with string escaping, own parser for an array of strings |
| logging | `log` | a caller-supplied callback, off by default |

Two exceptions, both off by default: a `gemm` feature that swaps in
`matrixmultiply` (itself dependency-free apart from `rawpointer`) if the
own kernel misses the budget on a target, and dev-dependencies for
tests, which never reach a consumer's build. Adding a runtime dependency
later is a decision recorded in this log, not a convenience.

### Performance budgets

Measured on the consumer's benchmark machine against the figures the
stock wheel produced there:

| Quantity | Budget | Stock libvosk measured |
|---|---|---|
| decode compute per 40 ms block, p95 | 5 ms | 2 ms |
| real-time factor over the corpus | 0.05 | 0.019 |
| `Model::open` | 2 s | not measured |
| `Recognizer::new` with a grammar under 300 words | 250 ms | not measured |
| resident memory with one recognizer | 300 MB | not measured |
| partial first appearance after word end, p50 and p90 | 40 ms and 190 ms | 40 ms and 190 ms |

### Verification and acceptance

Oracle: the stock `vosk` wheel, driven by the consumer's benchmark on
its replay corpora with identical 40 ms blocks. Both sides run at
dither 0 for parity tests, libvosk through a `conf/mfcc.conf` sibling
that adds `--dither=0`.

Gates, in order; each is cheap enough to run before the next stage is
built:

- G0, before any streaming code: batch-decode the replay corpus through
  the vendored acoustic and graph code with the grammar composed
  offline, in zero and in faithful i-vector mode, and score word error
  against libvosk's finals on the same takes. If zero mode is within one
  point of faithful under the grammar, faithful mode is deferred.
- G1, front end: MFCC frames within 1e-3 and i-vector estimates within
  1e-2 relative of Kaldi's on ten takes.
- G2, decoder: the partial text after each block equals libvosk's on at
  least 95% of blocks; the final word sequence equals libvosk's on at
  least 99% of segments; word times within one output frame (30 ms) on
  95% of matched words.
- G3, latency: the consumer's benchmark with the runtime added as an
  engine reports the same p50 and p90 first-appearance latency as the
  stock wheel within one block, inside the compute budget.
- G4, alternatives: rank 0 equals the partial on every block; a silent
  primary is offered as `[sil]` and never as a vocabulary word; on the
  consumer's silence census takes, no block whose libvosk raw partial
  was empty yields a vocabulary word at rank 0; the contest census is
  rerun and recorded.
- G5, endpointing: endpoint times equal libvosk's within one step
  (0.2 s) on 95% of segments.

## Consequences

- The consumer selects this runtime the way it selects its other
  backends today and keeps the stock wheel selectable until every gate
  has passed; a recording names which runtime produced it.
- Line dictation, the one consumer mode that asks for ten alternatives
  on finals, receives beam n-best; its behaviour is measured before
  that mode leaves the stock wheel.
- Speaker vectors stay on the stock wheel for the consumer's enrolment
  path until the play path is done.
- The other shipped languages are assumed to share the reference
  model's layout and are checked by loading each under G2 before the
  runtime is offered for them.
- The zero-dependency policy puts the GEMM kernel and the FFT in the
  runtime's own hands; the kernel is the one place performance can
  miss its budget, and the `gemm` feature is the bounded escape.
- Eager composition without lookahead places G's weights later along a
  word than libvosk's graph does; G2 measures the pruning consequence,
  weight pushing is the fallback.
- Floating-point summation order in the network differs from Kaldi's
  BLAS; chunked execution is checked at G1 against the same reference
  Vosk-Rust used.

## Implemented by

- Nothing yet. Milestones, in order: vendor and build (G0); the graph
  reader, bigram and composition; the streaming front end and chunked
  network (G1); the decoder with partials, intervals and endpointing
  (G2, G3, G5); alternatives and finals (G4); the C ABI.
