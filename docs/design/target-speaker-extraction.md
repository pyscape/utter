# Streaming target speaker extraction for utter

This is a Rust library that takes microphone audio and a previously
enrolled speaker, then emits that speaker's estimated waveform.  utter
consumes the resulting PCM directly. Alice's speech should remain
recognizable while Bob talks over her; when Alice stops, the output should
be quiet even if Bob continues.

This is an implementation proposal for a new library. `speaker_filter` is
the working crate name in examples. Its APIs and model do not exist yet.
The numerical performance and quality requirements below are acceptance
targets, not measurements. We must train and evaluate the proposed model
before claiming that it solves the task.

## Library boundary and scope

The deployed library owns enrollment inference, speaker-conditioned
separation, waveform reconstruction, and sample accounting. The application
owns microphone capture, conversion to 16 kHz mono, device recovery, and
delivery to utter. utter owns recognition, phrase lists, partials, word
timing, and endpointing. Neither library depends on the other at runtime.

```mermaid
flowchart LR
    E["Alice enrollment PCM"] --> S["Speaker encoder: run once"]
    S --> P["Reusable speaker profile"]
    M["Microphone: 16 kHz mono PCM"] --> X["Stateful speaker extractor"]
    P --> X
    X --> B["Borrow f32 samples, optionally collect 640"]
    B --> U["utter recognizer"]
    U --> A["Application: partials and finals"]
```

The first version supports one microphone channel, one selected speaker,
and one continuous output stream. Multiple simultaneous target speakers
require separate extractor instances. Enrollment is explicit: the caller
provides recordings of Alice speaking alone. We do not discover Alice by
name or learn her identity from the mixed live stream.

The initial task is speech separation with incidental noise suppression.
Microphone capture, echo cancellation, diarization, source localization,
automatic speaker switching, and transcription stay outside this library.
Single-channel extraction is an estimate: clipping, nearly identical
voices, very quiet target speech, and heavy overlap can defeat it.

## Reference systems and our implementation choice

[VoiceFilter](https://google.github.io/speaker-id/publications/VoiceFilter/)
establishes the enrollment-embedding and spectrogram-mask approach.
[VoiceFilter-Lite](https://research.google/pubs/voicefilter-lite-streaming-targeted-voice-separation-for-on-device-speech-recognition/)
addresses streaming recognition and the need to preserve recognition on
uncontaminated speech. These inform the task and evaluation; we implement
our own model, training recipe, and runtime. Our output is a reconstructed
waveform, so utter needs no knowledge of the separator's feature space.

[SpeakerBeam-SS](https://www.isca-archive.org/interspeech_2024/sato24_interspeech.pdf)
provides a useful alternative architecture and latency reference: its
causal configuration uses a 320-sample analysis window and 160-sample
shift at 16 kHz. Its reported CPU results belong to its model and hardware;
they are not predictions for ours.
[WeSep](https://github.com/wenet-e2e/WeSep) provides reference recipes for
speaker conditioning and mixture generation. We do not ship either
toolkit as a dependency or assume their checkpoints fit our architecture.

We start with a short-time Fourier transform (STFT), two small recurrent
layers, and a complex spectral mask. This gives us a completely specified
baseline using FFTs, matrix-vector products, and elementwise operations.
The FFT represents the mixture; the trained network decides which parts
belong to Alice. A frequency filter or voice activity detector alone
cannot make that decision during overlapping speech.

We choose this baseline for its limited operator set and straightforward
streaming state. It may need more capacity to meet the quality gates.
Failure of those gates is a model-design problem, not something a smaller
input block can fix. A SpeakerBeam-style learned analysis/synthesis model
is the next architecture to evaluate if the spectral baseline fails.

## Audio format and sample contract

| Property | First implementation |
|---|---|
| Preferred input and output | Mono normalized `f32` PCM, 16,000 samples/s |
| Integer capture convenience | `i16` input divided by 32768; output remains `f32` |
| Internal waveform and weights | `f32` |
| Analysis window, `W` | 320 samples, 20 ms |
| Internal hop, `H` | 160 samples, 10 ms |
| FFT length, `F` | 512; 257 nonredundant frequency bins |
| Suggested capture delivery | 160 or 320 samples, 10 or 20 ms |
| Suggested utter delivery | 640 samples, 40 ms |
| Neural future-frame context | Zero |
| Pending output in steady state | One hop, plus an incomplete input hop |

The direct connection passes a borrowed `&[f32]` waveform to utter's
planned float entry point. Full-scale amplitudes are `-1.0` and `1.0`;
samples must be finite and within that range. No waveform serialization,
integer quantization, or feature exchange occurs at the library boundary.
The utter-side integration contract is `[[rr:TD-13]]` and must be
implemented before this preferred connection is available. Its work is
tracked in the [utter roadmap](../../ROADMAP.md).

Input calls may contain any number of samples, including fewer than one
hop. Their boundaries do not change the model's frame boundaries or
results. The extractor processes each completed hop immediately and
retains the remainder. A 20 ms call normally causes two 10 ms updates.

An output sample keeps its input sample index. Suppression changes its
value, never its position. We emit quiet samples for suppressed speech
and preserve silence, pauses, and the duration of the recording. After
`finish`, total output samples equal total real input samples exactly.

We do **not** promise 320 output samples from every 320-sample input call.
The first call produces only 160 output samples because reconstruction
holds one hop. Subsequent full calls produce 320. `finish` releases the
remaining real samples. Zero-length input produces no output and does
not advance time.

The 40 ms downstream block is our application choice. utter's `accept`
accepts arbitrary lengths (`[[rr:src/recognizer.rs#accept]]`), and the
streaming tool defaults to 40 ms (`[[rr:src/bin/stream.rs#main]]`).
[Vosk's input loop](https://github.com/alphacep/vosk-api/blob/master/src/recognizer.cc)
also accepts shorter blocks: its 200 ms step limits slices of a large
call, rather than requiring 200 ms to accumulate before processing.
The acoustic model's separate inference cadence is not an alignment
requirement for this extractor.

## Exact analysis and reconstruction

### Frame indexing and windows

Let the stream start at relative sample zero and define `x[n] = 0` for
negative indices. At each hop boundary `t = H, 2H, 3H, ...`, analyze
`x[t-W .. t]`, with an exclusive upper bound. Use the periodic square-root
Hann window:

```text
w[j] = sqrt(0.5 - 0.5*cos(2*pi*j/W)),  0 <= j < W
X_t  = FFT_512(concat(w * x[t-W .. t], 192 zeros))
```

The forward transform is unnormalized. Reconstruction uses an inverse
transform divided by 512. Use the same window definition in Python and
Rust; a symmetric Hann with denominator `W-1` is a different operation.

Only bins `0..=256` enter the mask calculation. Restore conjugate symmetry
before the inverse FFT, force DC and Nyquist imaginary components to zero,
take the first 320 inverse-transform samples, and multiply by `w` again.
Add those samples into the synthesis accumulator at `[t-W, t)`.

Maintain a parallel accumulator of `w[j]^2` and divide by its accumulated
weight when emitting samples. With these windows and a half-window hop,
the complete overlap has unit weight. Treat an unexpectedly tiny divisor
on a real output sample as an implementation error, rather than amplifying
it. RustFFT leaves normalization to the caller, so the inverse scaling
must be explicit ([RustFFT documentation](https://docs.rs/rustfft/latest/rustfft/)).

### When an output interval becomes final

After processing frame `t`, emit `[t-W, t-W+H)`. No later frame overlaps
that interval. Discard its negative-index portion but retain the frame's
contribution to later, nonnegative samples.

| Real input available | Frame analyzed | Output finalized |
|---|---|---|
| 10 ms | -10 to 10 ms, with a zero prefix | Negative interval discarded |
| 20 ms | 0 to 20 ms | 0 to 10 ms |
| 30 ms | 10 to 30 ms | 10 to 20 ms |
| 50 ms | 30 to 50 ms | 30 to 40 ms; first utter block complete |

This is causal with respect to the latest captured input, but output for
an earlier interval needs the rest of its analysis window. Zero future
frames in the network does not mean zero waveform delay.

### End of input

For `N > 0` real input samples, pad internally to the next hop boundary,
then process one further all-zero hop. Emit only sample indices in
`[0, N)`. For an exact multiple of `H`, only the further hop is needed.
Keep the recurrent state while draining. These zeros are reconstruction
context; they are not appended to the user's recording.

`finish` produces at most `2H-1` real samples, possibly in two callbacks,
and marks the stream finished. A repeated `finish` emits nothing. A new
input call after finishing returns `Finished`. For `N = 0`, finishing
does no model work and emits nothing.

There is no leading silence inserted into the output, no loss of real
samples in the last hop, and no timestamp correction hidden in utter. The final
callback may have fewer than 160 samples, as required to preserve `N`.

## Speaker enrollment

### Enrollment recordings and profile lifetime

Collect roughly 5–10 seconds of Alice speaking alone, preferably across
several phrases. The initial API accepts up to 32 segments of at least
0.5 seconds each, with 3–30 seconds total. These bounds limit work and
reject clearly insufficient input; they do not certify that the recording
contains only Alice.

Run each segment independently through the speaker encoder. Average its
normalized embedding with the other segments using equal segment weights,
then normalize the average to unit length. Equal weighting avoids making
the longest recording the entire profile. Do this once, off the live
processing thread.

Return duration, RMS level, clipped-sample fraction, and pairwise embedding
similarities with the profile. Reject invalid lengths, all-zero audio, and
a near-zero final embedding. Surface poor recording quality as diagnostic
information. Any future automatic quality cutoff must be selected using
held-out enrollment data; duration and energy cannot prove speaker identity.

The profile contains the 128-element embedding, its format version, and
the identity of the exact model package that produced it. Reusing a
profile with a different model package is an error. The application owns
profile storage; enrollment audio is not needed during live inference.
We keep the profile fixed during a stream so that Bob cannot gradually
replace Alice through automatic adaptation.

### Compact speaker encoder

Use the same 320-sample window, 160-sample hop, and FFT convention as the
extractor. For enrollment, use complete frames within each segment and
omit an incomplete final frame. Convert squared FFT magnitudes into 80
mel bands between 20 and 7,600 Hz. Generate triangular filters using
`mel(f) = 2595*log10(1+f/700)`, unit peak, evaluated at FFT bin frequencies;
export the actual `80 x 257` filter matrix with the weights.

Take `ln(max(band_power, 1e-10))` and normalize each band with fixed
training-set mean and standard deviation, floored at `1e-4`. These are
separate from the separator's feature statistics.

| Layer | Shape and operation |
|---|---|
| Temporal convolution 1 | 80 to 128 channels, kernel 5, dilation 1 |
| Temporal convolution 2 | 128 to 128 channels, kernel 3, dilation 2 |
| Temporal convolution 3 | 128 to 128 channels, kernel 3, dilation 3 |
| After each convolution | Channel layer normalization, epsilon `1e-5`, then ReLU |
| Statistics pooling | Per-channel mean and population standard deviation over real frames; concatenate to 256 values |
| Projection | Affine 256 to 128 |
| Embedding | Divide by L2 norm, with a `1e-8` rejection floor |

All convolutions use stride one, bias, and left zero padding; no right
padding. Statistics pooling includes the resulting outputs for real
frames, including the initial frames with left context padding. Pooling
may see the entire enrollment segment because enrollment is offline.
Compute standard deviations as `sqrt(max(E[x^2]-E[x]^2, 0)+1e-5)`.
Convolution weights use `[out_channel, in_channel, kernel_position]`
layout and cross-correlation semantics, matching the training implementation.

This encoder is approximately 184,000 parameters before its training-only
speaker classifier. Its usefulness on unseen speakers is a required
experiment, not a consequence of the embedding having 128 dimensions.

## Streaming separation model

### Inputs, layers, and outputs

At each 10 ms hop, form 257 features:

```text
a[k] = ln(max(abs(X_t[k]), 1e-5))
f[k] = clamp((a[k] - mean[k]) / max(std[k], 1e-4), -8, 8)
v    = concat(f, enrolled_embedding)  # 257 + 128 = 385 values
```

`mean` and `std` are fixed statistics from training mixtures. Do not
normalize over the live recording or future frames. The waveform path
retains its original amplitude; feature normalization does not change
output gain.

| Layer | Shape and operation |
|---|---|
| Input projection | Affine 385 to 256, channel layer normalization with epsilon `1e-5`, ReLU |
| Recurrent layer 1 | Unidirectional GRU, input 256, hidden state 256 |
| Recurrent layer 2 | Unidirectional GRU, input 256, hidden state 256 |
| Mask head | Affine 256 to 514; split into 257 real and 257 imaginary values |
| Activity head | Affine 256 to 1, then sigmoid |

Both hidden states start at zero and persist through every hop, including
silence. There is no dropout in the reference training baseline and no
recurrent-state reset at microphone-call or utterance boundaries.

The mask is `M_re = 2*tanh(o_re)` and `M_im = 2*tanh(o_im)`. Multiply
the mixture by that complex mask:

```text
Y_re = M_re*X_re - M_im*X_im
Y_im = M_re*X_im + M_im*X_re
```

This can adjust phase and amplify a bin within the mask's bounds. It
cannot reconstruct every cancellation in a mixture; the bounds trade
reconstruction freedom for numerical stability. Initialize mask-head
weights to zero, real biases to `atanh(0.5)`, and imaginary biases to
zero, giving an identity mask at the start of separator training.

### Exact recurrent operation

For each GRU layer, use these equations and export gates in `r, z, n`
order. All products below are matrix-vector products except `*`, which
is elementwise:

```text
r = sigmoid(W_input_reset x + b_input_reset + W_hidden_reset h_prev + b_hidden_reset)
z = sigmoid(W_input_update x + b_input_update + W_hidden_update h_prev + b_hidden_update)
n = tanh(W_input_new x + b_input_new + r * (W_hidden_new h_prev + b_hidden_new))
h = (1-z) * n + z * h_prev
```

This follows [PyTorch's GRU convention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.modules.rnn.GRU.html),
including where the reset gate applies. Store separate input and hidden
biases. Substituting the other common GRU reset convention would invalidate
the checkpoint. Layer normalization uses population variance across the
current vector's channels, with learned scale and bias.

The separator has about 1.02 million parameters and performs approximately
102 million multiply-accumulate operations per second, excluding FFTs and
elementwise work. These are architecture counts, not CPU timing estimates.
The encoder and separator together contain about 1.2 million parameters,
or roughly 4.6 MiB of FP32 weights before feature tables and metadata.

### Suppression control and activity evidence

Default output uses the extracted waveform directly. A stream option
`strength` in `[0, 1]` blends it with the **sample-aligned** original:

```text
output[n] = strength*extracted[n] + (1-strength)*original[n]
```

At zero, return the original normalized samples exactly while retaining
the same delay and block accounting. At one, use the full model estimate. Intermediate
values preserve more of both Alice and Bob. This is a waveform tradeoff;
it does not guarantee a monotonic change in recognition errors. Keep
strength fixed for a stream in version one. Clamp the final blended
waveform to `[-1, 1]` and count saturated samples. This explicit output
policy permits direct use of utter's checked float interface. Training
includes that clamp in the waveform used to calculate reconstruction loss.

The activity head reports a model score for Alice's activity in the
analyzed window. It does not multiply the audio by another gate and is
not an authentication result. Return the window's real sample interval
with the score; it is not necessarily the same interval as the emitted
hop. Calibration, missed onsets, and false activity on Bob are measured
separately. Do not label the score a probability without calibration data.

No automatic bypass activates when the model is uncertain, and no
threshold deletes PCM. `[unk]` remains a recognition result owned by utter;
it is not a speaker-identification signal for this library.

## Rust runtime and model packaging

Implement the fixed network directly in Rust. The initial dependency
budget allows an FFT crate, JSON parsing, and checksum support. It does
not require a tensor framework, Python, ONNX Runtime, libtorch, libvosk,
or a GPU at deployment. Training can use Python and PyTorch independently.
An optional ONNX export can help reference comparisons, but it is not
the runtime's interchange format.

Start with scalar matrix-vector kernels and cross-check them against the
Python reference. Add SIMD only when the measured budget requires it.
Model loading creates FFT plans, validates dimensions, packs weights if
needed, and allocates all buffers. The steady-state extractor allocates
nothing, performs no I/O, takes no locks, and starts no threads. Work done
inside the caller's output callback is outside that guarantee.

The runtime state consists of an incomplete input hop, the previous
analysis hop, synthesis values and weights, a short original-waveform delay
buffer, FFT scratch, feature and activation scratch, two GRU states,
and input/output counters. All are bounded independently of session
length. Weights are immutable and shared through `Arc<Model>`; each
`Extractor` owns its state and has a single mutating owner.

The distributable model directory contains:

| File | Required contents |
|---|---|
| `model.json` | Format version; architecture identifier; rate/window/hop/FFT constants; tensor names, dimensions and byte offsets; weights checksum; training provenance |
| `weights.bin` | Little-endian, contiguous FP32 tensor data, including feature statistics and mel filters |
| `model-card.md` | Training and evaluation splits, supported conditions, measured hardware, quality and latency results, known failure cases |
| `LICENSE` and notices | Terms for the implementation, weights, and any incorporated material |

The first architecture identifier fixes every layer above. This is a
checked tensor bundle, not an arbitrary executable graph. Require unique
tensor names, exact required dimensions, four-byte alignment, disjoint
in-bounds byte ranges, finite values, and positive normalization scales.
Reject files over the initial 8 MiB tensor budget, integer overflow,
unsupported architecture constants, missing or extra tensors, and
checksum mismatch before constructing an extractor.

Export model identity as a hash of the exact manifest bytes and weight
checksum. Persist profiles in a versioned file containing that identity,
128 finite normalized values, and enrollment diagnostics. Profile loading
validates identity and dimensions before accepting the embedding.

We must record source-data terms and permission to redistribute trained
weights separately from a reference repository's code license. We build
our own checkpoint; no compatible production weights exist for this
proposed architecture yet.

## Proposed Rust API

The following signatures define the intended surface. Constructors may
allocate and fail. Processing uses a borrowed output callback so that
input length cannot force an unbounded output allocation.

```rust
pub const SAMPLE_RATE: u32 = 16_000;
pub const HOP_SAMPLES: usize = 160;

pub struct StreamOptions {
    pub source_start: u64,
    pub strength: f32, // Default: 1.0; validated in [0, 1].
}

pub struct FrameEvidence {
    pub source_start: u64,
    pub source_end: u64, // Real, observed part of the analysis window.
    pub target_activity_score: f32,
    pub used_end_padding: bool,
}

pub struct OutputChunk<'a> {
    pub source_start: u64,
    pub samples: &'a [f32], // Usually 160 samples; shorter only at EOF.
    pub evidence: FrameEvidence,
}

pub struct ProcessReport {
    pub accepted_samples: usize,
    pub emitted_samples: usize,
    pub next_input_sample: u64,
    pub next_output_sample: u64,
    pub clipped_output_samples: usize,
}

impl Model {
    pub fn open(path: &std::path::Path) -> Result<Self, Error>;
    pub fn enroll(&self, segments: &[&[f32]]) -> Result<Enrollment, Error>;
}

impl Extractor {
    pub fn new(
        model: std::sync::Arc<Model>,
        profile: &SpeakerProfile,
        options: StreamOptions,
    ) -> Result<Self, Error>;

    pub fn process_f32<F>(
        &mut self,
        first_sample: u64,
        samples: &[f32],
        emit: &mut F,
    ) -> Result<ProcessReport, Error>
    where
        F: for<'a> FnMut(OutputChunk<'a>);

    pub fn process_i16<F>(
        &mut self,
        first_sample: u64,
        pcm: &[i16],
        emit: &mut F,
    ) -> Result<ProcessReport, Error>
    where
        F: for<'a> FnMut(OutputChunk<'a>);

    pub fn finish<F>(&mut self, emit: &mut F) -> Result<ProcessReport, Error>
    where
        F: for<'a> FnMut(OutputChunk<'a>);
}
```

`Enrollment` contains a `SpeakerProfile` and its quality report. Profiles
copy their embedding into a new extractor; the caller may release the
profile afterward. Both input methods emit normalized `f32` chunks; the
integer method is convenience for capture APIs, not a different model
path. A chunk's waveform is borrowed only until its callback
returns. Callbacks execute synchronously on the processing thread and
must not reenter the extractor.

Require `first_sample == next_input_sample`. Validate that condition,
stream state, counter overflow, and the entire float slice's finite values
and `[-1, 1]` range before consuming a call. Enrollment uses the same
float validity check. A rejected call leaves state unchanged. Each emitted chunk must begin exactly
where the previous chunk ended. `finish` accepts zero new real samples;
its report counts only real output and leaves the input counter unchanged.

Emit the reconstructed, blended and clamped floats without rounding to
integer PCM. Nonfinite runtime values return a numerical error and poison the
stream; include the accepted and emitted counters so the caller can
identify any prefix already delivered. Do not substitute raw microphone
audio after a processing failure. A callback panic also invalidates the
instance for reuse after unwinding.

Errors distinguish invalid model, incompatible profile, invalid options or samples,
unexpected sample position, finished stream, counter overflow, and
numerical failure. Device discontinuities and speaker changes create a
new extractor and recognizer; there is no ambiguous reset that silently
renumbers an existing stream.

## Direct composition with utter

### Normal stream and shutdown

The application can deliver each borrowed 160-sample float chunk directly
to `rec.accept_f32(chunk.samples)`. This requires no staging copy. The
initial comparison profile groups four hops to match existing 40 ms
recognizer measurements; it holds a fixed `[f32; 640]` staging array and a fill count.
Every emitted 160-sample chunk goes into it. As soon as it contains 640
samples, call utter immediately. Do not wait for a separate 40 ms timer.
The same adapter must handle the final shorter chunk.

The processing sequence below uses the proposed extractor and planned
utter float entry point. `Reblock640` is application adapter code with a fixed
array; `push` emits full arrays and `finish` emits a nonempty remainder.
`publish` copies or consumes utter's borrowed JSON before the next call.

```rust
let mut rec = utter::Recognizer::new(&vosk_model, 16_000.0, &grammar)?;
let mut blocks = Reblock640::new();
let mut next_output_sample = stream_source_start;

let mut recognize = |samples: &[f32]| {
    // Invalid input is a programming error here: the extractor guarantees
    // finite, normalized output and the adapter preserves its samples.
    let step = rec.accept_f32(samples).expect("valid extractor output");
    if step.endpoint {
        publish(rec.result());
    } else {
        publish(rec.partial());
    }
};

{
    let mut emit = |chunk: speaker_filter::OutputChunk<'_>| {
        assert_eq!(chunk.source_start, next_output_sample);
        next_output_sample += chunk.samples.len() as u64;
        blocks.push(chunk.samples, &mut recognize);
    };
    while let Some(packet) = capture.next_packet() {
        extractor.process_i16(packet.first_sample, &packet.pcm, &mut emit)?;
    }
    extractor.finish(&mut emit)?;
}

blocks.finish(&mut recognize);
drop(recognize);
publish(rec.final_result());
```

For a consumer of today's integer-only utter interface, a temporary adapter
can convert each float using
`clamp(round_away_from_zero(y * 32768), -32768, 32767)` into a reusable
`[i16; 640]` and call `accept`. This is quantization, not serialization,
and must be benchmarked separately from the preferred float connection.

The required shutdown order is extractor drain, adapter drain, then utter
flush (`[[rr:final_result]]`). Neither `result` nor an ordinary utter
endpoint resets the extractor. Alice's speaker profile and separation
history persist across phrases; utter starts its next utterance through
its existing lifecycle (`[[rr:result]]`).

Shipping an example that depends on both crates is sufficient for initial
integration. An optional companion adapter crate can follow if applications
repeat the same code. Keep microphone and utter dependencies out of the
extractor's core dependency graph.

### Timestamps and silence

For a stream beginning at source sample `S`, output sample `j` corresponds
to original sample `S+j`. Feed each output sample exactly once. utter's
sample timestamps are in this audio sequence; the application adds `S`
when displaying absolute capture positions. Do not add or subtract
algorithmic latency from word positions: latency affects when the result
arrives, not the source interval it describes.

Capture packets carry both sample position and a monotonic timestamp.
Preserve the device's sample-to-time mapping separately from the worker's
wall clock so scheduling jitter does not move words on the recording.
If the device runs at 48 kHz, resample once upstream and account for the
resampler's delay there. A 44.1 kHz source likewise needs a stateful
resampler; splitting it into conveniently sized chunks is insufficient.

Keep sending the waveform during Alice's pauses and Bob-only intervals.
utter needs that elapsed audio for endpointing and its energy/floor
measurements. Those measurements now describe the filtered signal;
suppression may change their distributions. Test this explicitly rather
than carrying over raw-microphone threshold claims. Suppression also
does not prove that utter will never produce a word on quiet audio.

### Threads, backpressure, and discontinuities

The capture callback copies PCM and timestamp metadata into a preallocated
single-producer/single-consumer ring. A worker runs the extractor and
utter. It publishes application events away from capture. Start with
one worker and measure the combined workload; separate workers are an
optimization with an additional bounded queue, not a prerequisite.

Provision the input ring for 100 ms as an initial maximum capacity,
but process new hops immediately. Capacity is not a target queue depth.
Measure queue age and trigger a discontinuity if it exceeds the configured
latency limit or the ring overflows. The capture callback never blocks
waiting for recognition and never overwrites unread audio unnoticed.

On a missing packet, device change, or deliberate target switch, terminate
the current segment explicitly. If its accepted prefix is complete, it
may be drained and finalized with a discontinuity marker. Start both
libraries fresh at the next available absolute sample and retain that
segment's source offset. Do not concatenate across missing samples,
replace missing audio with unmarked silence, or reset only one library.

## Latency and resource budget

### Separate throughput from delay

Completing a 20 ms block within 20 ms prevents a growing compute backlog.
It does not remove that computation from latency. In this design,
reconstruction also holds a hop. The earlier estimate of effectively no
added latency from aligned blocks is therefore not a design guarantee.

For a finalized 10 ms output interval, the next 10 ms of input is needed
after its end. Relative to the interval's beginning, capture wait is
20 ms. Relative to its final sample, it is approximately 10 ms. Report
which convention a measurement uses.

For the end `T` of a 40 ms output block, define:

```text
t_raw(T)      = earliest delivery time of raw audio through T
t_filtered(T) = delivery time of filtered audio through the same T
added_delay  = t_filtered(T) - t_raw(T)
```

With regular 10 ms capture delivery, the first raw block `[0,40)` is
available at 40 ms. Its filtered counterpart needs the frame ending at
50 ms, so delivery is approximately `50 ms + C + Q`, where `C` is that
hop's compute and `Q` is scheduling or queue delay. Added delay is
approximately `10 ms + C + Q`.

With 20 ms capture deliveries at 20, 40, and 60 ms, the frame ending at
50 ms cannot start until the 60 ms packet arrives. Emit its output before
processing the packet's next hop. Added delay for the same block is then
approximately `20 ms + C + Q`, assuming earlier work has finished.
For illustration, `C = 3 ms` and `Q = 0` give 13 or 23 ms respectively.
These are scheduling calculations, not measured model performance.

Additional device buffering, upstream resampling, operating-system jitter,
and recognition work remain outside those examples. A future model with
lookahead must declare its extra context and change this calculation.

### Initial targets and measurement

Use a single worker on a desktop CPU as the provisional performance
target, with x86-64 and Apple silicon measured independently. Record the
exact hardware before treating any target as a supported configuration.

| Metric | Initial acceptance target |
|---|---|
| Extractor mean real-time factor | At most 0.20, including analysis and reconstruction |
| Extractor compute per 10 ms hop | p99 at most 5 ms |
| Filtered versus raw delivery of identical 40 ms intervals | p99 added delay at most 30 ms with 20 ms capture delivery |
| Combined extractor + utter throughput | Mean real-time factor at most 0.50 on the same worker |
| Soak | No dropped input and no sustained queue growth over 30 minutes |
| Tensor package | At most 8 MiB in FP32 |
| Per-stream mutable state | At most 1 MiB, excluding shared model and caller buffers |

Report p50, p90, p99, maximum, deadline-miss count, allocations, resident
memory, and queue age. Include cold enrollment/model startup separately.
Benchmark a stripped consumer binary against the same consumer without
the library to report deployment size honestly.

Run both an unpaced compute benchmark and a replay paced by capture
timestamps. Only the latter measures delivery delay and contention.
Measure first-word and endpoint arrival against the original target
speech times as an additional end-to-end test: recognition decisions can
change after filtering, so adding the transport delay to an old utter
latency figure is not a prediction of user-visible latency.

## Training and export plan

### Data and split rules

Use [LibriMix's published data and mixture machinery](https://github.com/JorisCos/LibriMix)
as the first reproducible separation benchmark, then add our own streaming
mixture generator. A mixture alone is insufficient for target extraction:
each item also needs a different recording of the selected speaker for
enrollment. Never use the target test utterance as its own enrollment.

Split by speaker before generating enrollment/mixture pairs. Training,
development, and test speakers must be disjoint. Include a separate
microphone-recorded evaluation with different rooms and devices, because
synthetic mixtures do not establish microphone performance.

Generate 8-second training sequences with the following initial sampling
weights. These are recipe settings to evaluate, not corpus statistics.

| Condition | Share |
|---|---|
| Alice + one interfering speaker, optional noise | 40% |
| Alice alone or Alice + nonspeech noise | 25% |
| Interfering speaker(s), Alice absent but enrolled | 20% |
| Silence or nonspeech noise, Alice absent | 10% |
| Alice + two interfering speakers | 5% |

Randomize speaker entrances, exits, pauses, and overlap fraction. Cover
0%, 25%, 50%, 75%, and 100% overlap; target-to-interferer ratios from
-10 to +10 dB; noise SNRs from 0 to 30 dB; and independent recording
gain/channel changes for enrollment. Use activity-region RMS to set source
ratios so long silences do not determine a speaker's gain. Retain explicit
source-activity labels from generation.

For reverberant examples, convolve each source with its own impulse
response in the same simulated room. The target label is Alice's
microphone-position waveform, including that response. This first model
does not also promise dereverberation. Align labels to the mixture sample
clock and apply the same final gain to mixture and target. Reject or
rescale mixtures that would clip before PCM conversion.

Keep target-absent examples, short onsets, and long Bob-only stretches in
every development run. Otherwise a model can appear successful while
always returning some speaker or clipping the start of Alice's command.

### Objectives and an initial reproducible recipe

First train the speaker encoder using speaker classification: unit-norm
classifier weights and embeddings, logits `30*cos(theta)`, ordinary
cross-entropy. Use batches of 32 speakers with two independently augmented
enrollment crops each. Evaluate same-speaker/different-speaker cosine
separation on held-out speakers and recordings. Freeze the encoder for
the first separator experiment; exclude the classifier from export.

Train the separator against its reconstructed waveform, running the exact
streaming window schedule and end drain. For an 8-second example with
target `s`, estimate `y`, and mixture `x`, use the following initial losses:

```text
d          = max(RMS(x), 1e-3)
L_wave     = mean(abs(y-s)) / d
compress(z)= z * max(abs(z), 1e-5)^(-0.7)
L_spec     = mean(abs(compress(STFT(y/d)) - compress(STFT(s/d))))
L_activity = binary_cross_entropy(activity_score, activity_label)
L_total    = L_wave + 0.5*L_spec + 0.1*L_activity
```

Use the same STFT definition for `L_spec` and include only frames with real
audio support. Label a frame active when at least half its real samples
overlap a generated Alice speech interval. At EOF, exclude synthetic
padding from that denominator. Apply a factor of two to `L_wave` for
Alice-only examples to penalize damage to already usable speech.

When Alice is absent, `s` is exactly zero and the waveform/spectral losses
remain defined. Do not apply SI-SDR to a zero target. Report SI-SDR for
target-present evaluation with its definition fixed; the
[SI-SDR paper](https://arxiv.org/abs/1811.02508) explains why metric choice
matters. Scale-sensitive waveform loss also discourages arbitrary output
gain that could disturb utter's energy measurements.

Use AdamW, learning rate `3e-4`, betas `(0.9, 0.999)`, weight decay
`1e-4`, global gradient norm clip 5, and deterministic data-generator
seeds. Train the encoder for an initial 100,000 updates, evaluating
held-out speaker verification every 5,000 updates and retaining the
checkpoint with the lowest equal-error rate. The separator's effective batch is 16 sequences, using gradient
accumulation if necessary. Start with 200,000 separator updates; evaluate
every 5,000 and retain the checkpoint with the lowest development loss
among those satisfying the clean-speech regression check. Record all
training configuration and selected checkpoints, including failed runs.

Carry recurrent state across each entire sequence. Introduce 32-second
sequences for 10% of batches during the final quarter of training to expose
long silences and changing turns. Backpropagate in one-second segments,
detaching hidden state at segment boundaries while preserving its values
and the analysis/synthesis buffers. Do not reset every segment as though
it were a new microphone stream. Weight segment losses by real sample
count before batch aggregation.

These hyperparameters make the first experiment reproducible; they do
not establish adequate training duration or accuracy. Training requires
an appropriate GPU environment and the licensed source audio. The first
work item is to demonstrate quality with the reference model before
investing in optimized native inference.

### Export and numerical agreement

Export explicit tensors, dimensions, feature statistics, mel filters,
architecture constants, and a model checksum. The exporter generates
reference fixtures for enrollment embeddings, each GRU state, mask
outputs, activity scores, reconstructed FP32 audio, final clamped float
output, and optional integer-adapter output.

Compare Rust and Python one hop at a time using the same initial state.
Initial tolerances are `atol=1e-5, rtol=1e-4` for a single layer and
RMS error at most `1e-4` full scale over a 60-second end-to-end clip.
Report maximum error and integer-adapter differences as well. Long-running drift
must be investigated rather than hidden by repeatedly resetting state.
Use the reference backend to localize differences before changing a
tolerance. Quantization or approximate nonlinearities require a separate
quality and latency evaluation after FP32 passes.

## Verification and acceptance gates

### Transport and DSP correctness

These gates do not require successful speaker separation:

1. **Identity reconstruction:** force a unit complex mask; recover impulses,
   tones, noise, full-scale PCM, and arbitrary EOF lengths with maximum
   FP32 error at most `1e-5` full scale and at most one PCM count after rounding.
2. **Chunk independence:** replay identical samples in 1-, 159-, 160-,
   161-, 320-, 640-sample and randomized calls. Output samples and source
   indices agree exactly within a fixed backend. Delivery times may differ.
3. **Sample conservation:** for every length from zero through several
   windows, and long randomized streams, finishing emits exactly `N`
   samples with no duplicate, negative, or skipped source position.
4. **Declared dependency:** two inputs identical through hop boundary `t`
   produce identical output through `t-H` and identical model state at
   `t`, irrespective of later samples. Perturb future input to detect
   accidental centered padding or normalization across time.
5. **State isolation:** interleave different streams and profiles; results
   match independent runs. Repeat finish, rejection, and construction paths.
6. **Resource bounds:** a 30-minute run has fixed state size and no runtime
   allocations inside the extractor. Exercise truncated/corrupt weights,
   bad profiles, overflow, discontinuities, and nonfinite computations.

### Acoustic and speaker-selection quality

Evaluate target-only, overlap, target-absent, noise-only, same-sex similar
voices, wrong enrollment, noisy enrollment, quiet Alice/loud Bob, and
three-speaker stress cases separately. Swap the enrolled profile on the
same mixture: the output should change to the newly selected speaker.
Never score quality only on intervals where the model reports Alice active.

Initial quality gates on a held-out, speaker-disjoint corpus are:

| Condition | Gate |
|---|---|
| Two-speaker mixture, target present | Median SI-SDR improvement at least 5 dB over the mixture; publish lower-tail results |
| Bob-only audio | Median output attenuation at least 20 dB; publish p10 attenuation and residual output dBFS |
| Alice-only command audio | Paired accuracy loss no more than 1 percentage point against raw Alice audio |
| Overlapping command audio | At least 20% relative reduction in target-command errors against raw mixture |
| Bob-only command audio | At least 80% relative reduction in false command events, with the application rule held fixed |

Record absolute numerators and denominators for every relative target.
If the baseline has no errors or false events, that relative gate is
undefined and must be reported as such. Report paired 95% confidence
intervals using speaker-level resampling. A release claim needs adequate
sample size; a point estimate alone does not establish it. Publish failure
slices even when an aggregate passes.

For Alice-absent input, report
`10*log10((mean(x^2)+1e-10)/(mean(y^2)+1e-10))` attenuation alongside
`10*log10(mean(y^2)+1e-10)` output dBFS. Do not count already-silent
mixtures as successful speaker suppression. Speech distortion,
interferer leakage, and recognition errors are complementary measurements.

### Recognition with utter and stock Vosk

Run these four streams through the same recognizer configuration:

| Stream | Purpose |
|---|---|
| Original Alice waveform | Recognition reference for the source |
| Alice/Bob mixture | Baseline without extraction |
| Unit-mask reconstructed mixture | Detect DSP, timing, or conversion regressions |
| Learned extracted Alice waveform | Measure the actual benefit and harm |

Use utter as the primary downstream consumer. Replay the exact same saved
filtered PCM through stock Vosk as a second check, keeping grammar,
unknown-word setting, endpoint options, input blocks, and finalization
identical. Preserve normalized float samples in the benchmark artifact;
feed stock Vosk's float entry point samples multiplied by 32768, as its
input convention requires. A wrapper that accepts only integer PCM needs
a separate quantized comparison, not an implicit cast in the float test.
File encoding for a saved benchmark artifact is outside the live library
connection. Existing utter/Vosk comparisons establish recognizer behavior
on their corpus (`[[rr:Full grammar: accuracy]]`); they provide no evidence
for the new extraction stage until we run these experiments.

Build a command corpus with Alice and Bob issuing both identical and
different commands, including Bob speaking while Alice is absent. For
Speech Commands-derived examples, obtain enrollment from other recordings
of the same speaker, exclude the target clip, and report exclusions where
the speaker lacks sufficient enrollment. Keep speaker-disjoint splits
across training and evaluation. Do not evaluate utter on unrestricted
LibriSpeech transcripts by forcing them into an unsuitable phrase list.

Record command accuracy, Bob-derived insertions, false command events
per hour and per trial, missing target commands, first partial appearance,
partial revisions, endpoints, word sample positions, and `[unk]` counts.
An unchanged held partial is not a new false command on every block;
define event boundaries in the benchmark manifest. Report both raw decoder
outputs and the application's command-acceptance rule.

Use the same application rule across baseline and extraction variants,
with thresholds selected only on development data. If filtering needs
different endpoint settings, report that as a separate experiment.
Always include long silence/Bob-only sessions and commands after those
sessions to test floor tracking and stale recurrent state.

The benchmark manifest records audio hashes, enrollment IDs, speakers,
mixture offsets and gains, model/weight hashes, code revisions, runtime
backend, hardware, thread count, recognizer options, delivery schedule,
and padding policy. Save per-example metrics and failures so the report
can be reproduced and disagreements inspected.

## Implementation sequence and repository shape

The new repository can start with this structure:

```text
src/
  lib.rs          Public API and error types
  model.rs        Fixed tensor loader and validation
  enrollment.rs   Speaker encoder and profiles
  stft.rs         Analysis, overlap-add, and exact drain
  network.rs      GRU layers, masks, and activity head
  kernels.rs      Checked scalar inference operations
  stream.rs       Buffers, counters, and processing lifecycle
examples/
  filter_wav.rs   PCM extraction with profile and model arguments
  utter_stream.rs 640-sample adapter and recognizer composition
training/
  model.py        Matching reference encoder and separator
  data.py         Speaker-disjoint mixture/enrollment generation
  train.py        Reproducible training entry point
  export.py       Tensor bundle and cross-runtime fixtures
benchmarks/
  replay.rs       Paced and unpaced streaming measurements
  score.py        Acoustic, command, and latency reports
tests/
  fixtures/      Small synthetic DSP and numerical reference cases
```

Implement in this order, with a reviewable artifact at each step:

1. **Signal path:** implement the reference windows and reconstruction,
   streaming sample contract, and unit-mask tests. Demonstrate the 10/20 ms
   latency accounting with timestamped replay before adding a network.
2. **Model feasibility:** train the encoder and separator reference;
   produce target-present, target-absent, and clean-speech results, including
   downstream utter command accuracy. If this baseline cannot meet the
   quality gates, revise its architecture before native optimization.
3. **Rust inference:** implement the fixed operators, loader, enrollment,
   and stateful extraction. Pass exported numerical fixtures and the
   chunk-independence tests.
4. **Composition:** deliver the WAV and utter examples with correct
   shutdown, timestamps, silence, and discontinuity handling.
5. **Measured deployment:** run paced workloads, acoustic and recognition
   gates, and long sessions on each intended CPU. Optimize the measured
   bottleneck, then repeat affected quality and numerical gates.

The unresolved questions are whether this small model preserves Alice
well enough under overlap, how much enrollment is sufficient on the
intended microphones, and which CPUs meet the latency target. The design
makes those measurable without tying utter to a particular separator
architecture or neural-network runtime.
