# TD-13: Preprocessors feed borrowed waveform samples directly into utter

- Tags: streaming, audio-input, integration, precision, parity

## Context

A companion library will extract an enrolled speaker's waveform from
mixed microphone audio. Its proposed architecture is described in
`[[rr:Library boundary and scope]]`. utter should consume that waveform
without acquiring a dependency on the extractor or its neural model.

The current Rust entry point takes `&[i16]` and converts it into floating
point before feature extraction (`[[rr:src/recognizer.rs#accept]]`). A
borrowed integer slice already needs no serialization. However, using
it for a floating-point extractor would round the waveform to integers
only for utter to convert it back to floats. Word energy and the floor
also currently read integer samples (`[[rr:energy_dbfs]]`,
`[[rr:src/recognizer.rs#feed]]`), so adding a float signature alone would
not remove that loss of precision.

This record fixes the interface and implementation requirements for the
planned integration. The roadmap tracks the work; the float entry point
is not implemented by this documentation change.

## Considered options

- **Borrowed 16-bit waveform slices.** Already supported and sufficient
  for an immediate compatibility adapter. No encoding is necessary,
  but a floating-point extractor would quantize its output. Retained
  for existing consumers; not the preferred companion interface.
- **WAV, byte packets, or a pipe between processes.** Useful when the
  application needs files or process isolation. They introduce framing,
  ownership and buffering work that in-process composition does not need.
  Rejected as the default connection between the libraries.
- **FFT bins, masks, mel features, or MFCCs.** Would couple the extractor
  to utter's feature configuration and bypass waveform-dependent energy,
  normalization and adaptation. The extractor's STFT is not utter's MFCC
  front end: this design uses 20 ms square-root Hann windows, while the
  reference Vosk model uses 25 ms Povey windows with pre-emphasis and
  different feature processing (`[[rr:TD-2#Inputs: configuration]]`). Matching hop sizes does not make their
  spectra interchangeable. Rejected for this integration.
- **Borrowed floating-point waveform slices.** Preserve the extractor's
  sample precision and keep the connection independent of both models.
  Taken, with amplitude, lifetime and clock conventions defined below.

## Decision outcome

### The connection carries normalized waveform samples

Add this Rust entry point while keeping `accept(&[i16]) -> Step`:

```rust
pub fn accept_f32(&mut self, samples: &[f32]) -> Result<Step, AudioInputError>;
```

Input is contiguous mono time-domain PCM at the recognizer's configured
sample rate. The companion uses 16,000 samples/s and constructs utter at
that rate. Each `f32` is an amplitude, not a serialized byte sequence or
a feature vector. Negative full scale is `-1.0`, positive full scale is
`1.0`, and zero is digital silence. Valid samples are finite and in the
inclusive range `[-1.0, 1.0]`.

The extractor reconstructs and clamps its waveform to that range before
handoff, reporting saturation in its diagnostics. Clamping is an explicit
output policy; integer quantization is absent from this connection. An
`i16` source maps exactly through `sample as f32 / 32768.0`.

Validate the entire slice before mutating recognizer state. A nonfinite
or out-of-range sample returns `AudioInputError`, with its index within
the call and the error kind. Do not silently normalize, clip, replace,
or partially accept invalid input. Empty input follows the existing
integer entry point's lifecycle without advancing the sample clock.

The caller lends the slice for the duration of the call and may reuse its
buffer afterward. utter must not retain a pointer into it. Same-thread
composition requires only a slice borrow; cross-thread composition needs
owned reusable buffers or a bounded ring managed by the application.

This removes serialization and integer conversion at the library boundary.
It does not promise that utter retains no samples, allocates nothing, or
performs no internal copies.

### A shared waveform path preserves scale and precision

Keep the MFCC implementation's existing amplitude convention internally:
one raw PCM count is one float unit. Convert normalized input with
`raw = normalized * 32768.0` using a reusable bounded scratch buffer.
The integer entry point supplies `raw = sample as f32` to the same
internal acceptance path. Both transformations are exact for values
originating from `i16`.

The shared path owns the existing input slicing, feature advancement,
silence weighting, PCM accounting, stability updates and endpoint check.
Do not add an integer cast anywhere on the normalized float path.
Preserve the old call and decoder-advance order so an equivalent integer
input does not change recognition behavior.

Retain waveform history as raw-scale `f32` rather than `i16`. The floor
tracker and per-word energy accumulate its squared values in `f64` and
use the existing full-scale reference of 32768 when producing dBFS.
This preserves the arithmetic for integer-origin samples and measures
fractional samples that an integer buffer would erase. Digital silence
keeps the existing absent-energy/floor behavior, rather than introducing
a new finite minimum.

The two public input methods may be interleaved on the same recognizer
without resetting it. Units are converted at entry; all retained samples
use the one internal convention. Measure the extra history memory from
four-byte samples instead of two-byte samples. Reuse conversion scratch;
do not allocate a new temporary vector for every upstream block.

### Sample count defines time and delivery cadence

Feed every output sample, including suppressed speech and silence, once
and in source order. Do not remove Bob-only intervals or use an activity
score to stop advancing utter's audio clock. The application checks source
continuity before accepting a chunk; utter continues to count the samples
it is actually given.

The extractor may output 160-sample hops. The application may pass each
hop immediately or collect two/four hops for 20/40 ms delivery. A 40 ms
adapter uses a fixed `[f32; 640]` and calls utter as soon as it is full.
Copying into that staging array is not serialization. Direct hop delivery
avoids the staging copy; its decoder/query overhead and partial cadence
must be measured against the established 40 ms baseline.

Do not align extractor windows to utter's acoustic-model chunks or insert
a fixed 200 ms wait. Calls may have arbitrary lengths. Grouping calls can
change when partials and endpoints are observed, so the benchmark records
the actual delivery schedule rather than assuming every block size is
observationally equivalent.

For a new recognizer whose first source sample is `S`, utter sample `j`
refers to source sample `S+j`. Buffering and computation delay delivery;
they do not move the sample's source position. The application owns this
offset and the capture sample-to-wall-clock mapping. No extractor-specific
timestamp, speaker identifier, or activity-score field is added to the
recognizer's input or results for this connection.

### The host composes two independent stream lifecycles

The extractor owns enrollment, separation state, windowing and draining.
utter owns feature extraction, recognition and endpointing. An ordinary
utter endpoint and `result` call do not reset the extractor. The next
phrase continues through the same enrolled profile and separation state.

At true EOF, finish the extractor, deliver all remaining real samples,
flush a partially filled delivery block, then call `final_result` on utter.
Do not feed the extractor's internal padding as additional real silence.
The existing utter lifecycle remains the downstream contract
(`[[rr:result]]`, `[[rr:final_result]]`).

On a capture gap or target change, the application closes the old segment
explicitly and starts fresh instances with the next source offset. A
speech recognizer that has adapted to Alice should not silently continue
as if a new selected speaker were the same stream. The host controls
which incomplete result, if any, to publish at that discontinuity.

Energy and floor results describe the waveform utter receives. When that
waveform has been filtered, raw-microphone thresholds are not automatically
valid. The recognizer applies its configured endpoint rules as before;
speaker rejection policy stays with the extractor and application.

### Compatibility is verified at the input and result boundaries

The implementation is accepted only after these checks:

- Feed identical `i16` samples through the old path and through exact
  normalized floats, using the same blocks and deterministic front-end
  settings. Compare features, per-block results, word positions, energy,
  floor, endpoint reasons and final flushes. Existing integer parity gates
  must still pass against stock Vosk.
- Feed quiet fractional floats that would round to zero in `i16`. Verify
  their nonzero feature and energy input so a hidden integer conversion
  cannot pass the test. Verify scaling with known-amplitude signals and
  check digital silence separately.
- Reject NaN, infinities and out-of-range values before any state change,
  including an invalid value at the end of a long slice. Exercise empty
  calls, mixed integer/float calls and endpoint transitions.
- Replay a synthetic preprocessing stream with startup delay, partial
  final hops, silence, and discontinuities. Verify sample counts, source
  mapping and extractor-before-recognizer shutdown without requiring a
  trained speaker model in utter's tests.
- Benchmark normalized float input at 10, 20 and 40 ms delivery, recording
  allocations, retained memory, compute and result arrival. Publish a
  separate end-to-end report once trained extraction weights are available.

An integer-origin float test requires equality on a fixed backend. A
learned filter changes the waveform and therefore may change recognized
words; recognizer parity is measured by feeding both engines the same
waveform. Quality improvement against unfiltered input is a different
measurement.

## Consequences

The companion can hand utter its reconstructed waveform without integer
rounding or an interchange protocol. utter remains a standalone,
zero-dependency recognizer, and other Rust audio processors can use the
same entry point.

The required utter work is a checked float entry point, shared ingestion,
float waveform history and energy handling, a composition example, and
the verification above. Existing integer and C callers retain their
current interface. A new C float ABI or Python convenience wrapper can
be designed when a consumer needs it; neither is required to connect the
two Rust libraries.
