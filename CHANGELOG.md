# Changelog

Notable changes to the crate, newest first. The project follows semantic
versioning; release dates are recorded by the git tags and the GitHub
releases.

## 0.0.6

### Added

- TitaNet-small speaker evidence. SpeakerModel::open and
  utter_spk_model_new also open a TitaNet-small directory converted by
  scripts/titanet_convert.py; set on a recognizer, it embeds each
  word's span on a thread of the recognizer's own, the best path's
  open word included, and the word's entry carries spk, spk_frames,
  spk_start and spk_end from the first result read after one more
  40 ms of audio, on every machine. Partials and finals carry no
  top-level evidence with TitaNet. Only TitaNet-small opens. Needs
  audio at 16 kHz or faster. The recognizer's own thread costs about
  what it costs with no speaker model, and a word's evidence arrives
  a median 200 ms after it ends. set_spk_threads(false),
  utter_recognizer_set_spk_threads, embeds closed spans in budgeted
  slices between decoder advances instead: the 99th percentile block
  is 2.60 ms against 2.91 for 0.0.5 with the x-vector, the real-time
  factor 0.0134 against 0.0171, and evidence arrives a median 420 ms
  after the word; a host that advances the decoder on every accept
  gets none. On the speaker page, one word clears a gate at 1% false
  accept 81.2% of the time, against 41.6% with the x-vector. TD-15,
  TD-16.
- The acoustic model's certainty and the sound outside words, on every
  partial and final, always on. certainty_words and
  certainty_outside are 1 minus the normalised entropy of the
  network's output over the frames decoded since the previous result,
  split by the path into frames in words and outside them, counted in
  words_frames and outside_frames; neither moves with the grammar or
  the path. band_db, band_sd_db, rise_start_sample, rise_ms and
  rise_db describe the frames in no named word: each mel band's level
  over its own floor, how steady it is, and the loudest rise. Every
  word entry carries certainty over its own span, and partial_result's
  [sil] and [speech] entries carry the sound over theirs. With the
  keys taken out every result is 0.0.5's byte for byte. Against 0.0.5
  they cost up to 0.9% in real-time factor and 0.015 ms at the 99th
  percentile block on TD-17's streams; on the Speech Commands steady
  state, three runs a side, the median block is 3.2% slower and the
  99th percentile 0.068 ms, 2.5%, with the real-time factor within the
  spread between runs. A sound-rating benchmark page measures what
  they tell apart. TD-17.
- Embedding any span from a speaker model of either layout:
  SpeakerModel embed, utter_spk_model_embed and utter_spk_model_dim.
- Normalized float input: Recognizer::accept_f32 takes samples in
  -1.0..=1.0 and refuses a block with any other value whole, leaving
  the recognizer as it was. A 16-bit sample s gives what accept gives
  for s as f32 / 32768.0, and the two calls may be interleaved. Rust
  only. TD-13.
- A small-models gate page: German, French, Spanish and Russian small
  models against the stock wheel on Common Voice single words. Each
  passes: partials per block 98.9 to 99.9%, segments equal 95.7 to
  98.8%.

### Fixed

- A stream's heap no longer grows with its length. Every per-frame and
  per-sample buffer is kept back to the current utterance: at 20
  minutes without a speaker model, 17.9 MiB where 0.0.5 held 246 MiB.
  A speaker model set mid-stream starts at the current utterance, its
  mean normalisation afresh there. TD-14 amended.
- The benchmark preflight opens each engine in a process of its own.
  The engine opened second no longer shows about 20 ms on its first
  recognizer that was not its own.
- speech_commands.py sends grammars as unescaped JSON, so the stock
  wheel keeps non-ASCII words.

## 0.0.5

### Added

- Speaker evidence on every word. With a speaker model set,
  set_spk_model, SetSpkModel, utter_recognizer_set_spk_model, opened
  from a vosk speaker model directory such as vosk-model-spk-0.4 by
  SpeakerModel::open or utter_spk_model_new, every word entry of every
  word list carries spk, spk_frames, spk_start and spk_end over its own
  span, and every partial and final carries them over its speech. Off
  by default; without a speaker model the output is identical to 0.0.4.
  The network runs only on frames a word's evidence pools or is likely
  to, so it moves no result to a later block; on the speaker page's
  streams it takes the real-time factor from 0.0117 to 0.0197 and the
  99th percentile block from 2.86 to 3.26 ms. TD-14.
- A speaker evidence benchmark page: how often one word names its
  speaker, utter's evidence beside the wheel's vector, the same
  x-vector through Kaldi, TitaNet-small and CAM++, by word length and
  as speech accumulates, and how soon a partial carries it.

## 0.0.4

### Fixed

- The floor margin on the bound reads no span shorter than the bound
  itself. A quiet word's first frames after a pause sit near the floor
  on any microphone, and 0.0.3 let them waive the veto and count as
  silence. A span at least the bound's length is seconds on a cold
  room and a few frames at an onset. On the public rooms the wordless
  closes stay, at a median 1.2 s instead of 0.72; on the first
  consumer's recordings, at the finals, 65 phantoms leave, 9 spoken
  words come back and 2 short ones are lost to a boundary in hiss.
  TD-12 amended with the rule and the price.

### Added

- A quiet-onsets benchmark page: the states page's pause streams with
  every word's onset held a few dB over the gap floor, decoded by the
  stock rules, the vetoed bound and the same bound with the margin.

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
