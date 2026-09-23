# Use case: live dictation of short commands from a closed vocabulary

The application this library is first built for takes short spoken
commands from a closed vocabulary of about sixty words, in five
languages, and acts on each word as soon as it is stable, while the
speaker is still talking. A command is one to four words, such as
"alpha seven" or "bravo takes charlie two", spoken in a quiet room by
one known speaker, often with long silences between commands while the
speaker thinks. Every session is recorded and replayed offline for
diagnosis, so the decoder's behaviour must be reproducible from the
audio alone.

This document names every way the application uses Vosk today, both the
stock wheel and the fork it had to maintain, states what goes wrong,
and states what it needs instead. It deliberately leaves out one thing
the application does today: acting on the recognizer's final result.
That path is being removed, for reasons given in section 4.

## 1. Audio and clocks

- Capture is 16 kHz, mono, 16-bit, in 40 ms blocks from the microphone
  thread. Every block is fed to the recognizer and, separately, to a
  voice activity detector that keeps a three-second ring of speech
  probabilities stamped with the wall clock.
- The recognizer's word times are seconds of audio fed, from the start
  of its stream. The detector's times are wall clock. Reconciling the
  two is a standing source of bugs: the application subtracts an
  accumulated drift each time it primes the decoder with a synthetic
  silence block (section 2), and any mismatch gates a word on the wrong
  stretch of audio.

## 2. How the stock library is used

Call sequence, per session:

1. `Model(path)` once per language. The model directory is a copy of
   the stock small model whose `conf/model.conf` sets
   `--frames-per-chunk=12` and endpoint rules 2, 3 and 4 to 0.4 s of
   trailing silence.
2. `KaldiRecognizer(model, 16000, grammar_json)` with the grammar as a
   JSON array of strings: the vocabulary as single words, today 62 of
   them, plus any aliases the speaker has enrolled. The recognizer is
   rebuilt whenever the grammar changes.
3. `SetWords(True)` so results carry word times. In one dictation mode
   `SetMaxAlternatives(10)`; otherwise no alternatives are requested
   from the stock wheel.
4. On every rebuild, one 0.25 s block of digital silence is fed to
   prime the decoder, and the last real block is fed again so a word
   split across the rebuild is not lost. The synthetic silence shifts
   every later word time, hence the drift bookkeeping in section 1.
5. `AcceptWaveform(block)` for every 40 ms block; `PartialResult()`
   immediately after each one. The partial's text is split into words
   and each word is fed to a settle rule: a word is acted on after it
   has appeared unchanged in a fixed number of consecutive partials,
   the number depending on the word (one return for most content
   words, three for the words most often retracted), measured at a
   50 ms cadence. Words are ranked against the application's current
   state before one is chosen, but only among readings the decoder
   actually offered; the application never repairs a word.
6. When `AcceptWaveform` returns true, `Result()` is read. This is the
   path being removed.
7. Every partial and result is written to the session recording with
   its offset in milliseconds, the backend that produced it, and the
   model path, so an offline tool can replay the WAV through the same
   recognizer and reproduce the session.
8. Five languages ship, each a stock small model with the same layout.
9. The same library's speaker model extracts speaker embeddings for
   enrolment and speaker gating, from a second recognizer per clip whose
   final carries the vector. Gating decides per word, so that evidence
   is wanted from the decoding loop itself: need 9 below.

## 3. What the fork added, and why every piece is critical

The stock partial is one string. Everything the application needs to
decide when a word is real, and which of two near-homophones was said,
is inside the decoder and thrown away. The fork exposed it. Each item
below is a hard requirement of this use case.

### 3.1 Partial alternatives

`SetPartialAlternatives(n)`: up to n candidate readings alongside the
partial, not in place of it, taken from the same mid-utterance state
without ending the utterance. The application ranks the readings and
chooses one. Requirements learned the hard way:

- Readings are distinct word sequences. The fork extracted shortest
  paths from a lattice, which returned alignment variants of the same
  words as separate readings; on a 3,387-partial census only one
  partial in the whole session carried two readings that differed in
  one word.
- Rank 0 is always the decoder's own best reading, even when it is
  empty. An empty best reading must be delivered as an empty reading,
  never dropped so that the first non-empty rival takes its place. That
  exact defect put phantom words on screen during silence.
- The readings must be current. The fork's came from an incrementally
  determinized lattice that trailed the fed audio by a median 0.965 s,
  so by the time a rival was visible the decoder had already resolved
  it. Readings drawn from the live search beam at the last decoded
  frame are what is wanted.
- Each reading carries a raw score (the fork used the negated total
  cost). The application uses scores only to order readings, never as
  probabilities.

### 3.2 Null evidence: the confusion network with the empty hypothesis

`SetPartialConfusion(1)`: per word position, every rival the decoder
weighed and its posterior, including the null hypothesis. Without it a
word suppressed for a low posterior looks simply absent, and absence
cannot be told from silence. The rule the application built on it: a
word is eligible only if its evidence is present in its position and
exceeds the null evidence; a tie favours absence; missing null evidence
is unknown, not zero.

### 3.3 The evidence snapshot

`SetEvidenceConfig(json)` and `EvidenceResult()`: a versioned snapshot
published beside every partial, read without advancing decoding or
changing any result. Its contents, all required:

- The epoch and the frontiers: how many samples were accepted, how far
  decoding has reached, and how far the lattice had reached, each in
  samples, so a consumer can tell a fresh reading from a stale one.
- Candidates: the ordered readings, each flagged primary or not and
  empty or not, with a likelihood when known and a `ready_prefix`
  count.
- Per word in a candidate: the text; start and end in integer samples
  of the fed audio, or explicitly unknown; the confusion-network bin it
  aligns to; the word posterior and the null posterior in that bin;
  an energy verdict, quiet or non-quiet, from the PCM under the word's
  interval against a configured dBFS floor; how many milliseconds the
  word has stood unchanged; whether it is ready; and a reason code
  when it is not.
- A `truncated` flag when the distinct-reading bound was hit.

The configuration: a mode (`shadow` computes and publishes but never
withholds a word; `enforce` withholds words that are not ready), a
per-word map of hold durations in milliseconds with a default, the
quiet floor in dBFS or null to disable it, and a profile identifier so
a recording names the profile that ran. A rejected configuration leaves
the previous one in force; an accepted one takes effect at the next
utterance; before any configuration the output is byte-identical to
stock.

### 3.4 What the snapshot was for

Readiness is the settle rule moved into the decoder, where the
evidence is: a word becomes ready when it has stood unchanged for its
hold, has word evidence above null evidence in its bin, and the audio
under its interval was not quiet. Zero ready words means no word is
dispatched; it is not an end-of-speech event. The application's
acceptance gates for this mechanism were: no ready word during labelled
room-only intervals, no loss of correct words against the stock
baseline, and a 95th-percentile delay from acoustic word end to ready
no more than 120 ms above baseline.

The fork never finished it: its snapshot serialiser emitted no
candidates, so none of 3.3 was ever measured on real audio. The
requirement stands; the implementation is this library's.

## 4. What goes wrong today

- **Silence produces words.** With a closed grammar and no way to say
  "nothing", the decoder maps room noise onto the nearest vocabulary
  word. Measured on two sessions: 373 of 414 and 138 of 173 words that
  the application acted on came from a partial whose raw text was
  empty, promoted by the empty-primary defect above. Separately, both
  the stock wheel and the fork emit a one-word result on pure silence
  about twenty seconds into every silent stretch, at each decoder
  rebuild.
- **There is no unknown-word option.** Vosk's vocabulary has an
  unknown-word symbol that a grammar may include so out-of-vocabulary
  speech decodes to it instead of to the closest word. The application
  never enabled it, and a closed grammar without it cannot express
  "that was a sound, not a command". A hypothesis for silence and a
  hypothesis for unknown speech must both be available in every
  reading list, with their own scores, so absence is a reading the
  application can rank rather than an accident of filtering. Emitting
  during silence is acceptable as long as what is emitted says it is
  silence, with a bracketed token such as `[unk]` or `[sil]`, never an
  empty string and never a vocabulary word.
- **Partials and finals disagree.** On one session, the words acted on
  from the stream contradicted the utterance's own final on 53 of 59
  takes. The application had to reconcile two decodes of the same
  audio, retracting words it had already acted on. The fix is to have
  one decode: act on the stream, and treat end of speech as an event
  that says where the stream ended, not as a second answer.
- **Two clocks.** Word times in fed-audio seconds versus a detector on
  the wall clock, with drift from synthetic silence, as in section 1.
- **The build.** Any change to what the decoder exposes meant a Kaldi
  cross-compile in a Docker container inside WSL2, or running the fork
  on a Linux host over SSH with audio streamed across. The application
  needs the decoder to build with `cargo build` on Windows.

## 5. What the application needs from this library

Per 40 ms block, after `accept`:

1. **The partial**: the best path as words, each with start and end in
   integer samples of the fed audio, and therefore a duration.
   Samples, not seconds, and never the wall clock.
2. **Readings**: the distinct word sequences alive in the beam, ranked
   by cost, rank 0 equal to the partial, each with its raw score,
   current to the last decoded frame. A silent reading announces
   itself with a bracketed token and is never an empty string; the
   unknown-word sequence is a reading whenever it is a contender.
3. **Per-word evidence** from the same loop: the energy in dBFS under
   the word's interval, the speech probability under it, how long the
   word has stood unchanged, and, in enforce mode, whether it is ready
   under a configured per-word hold and quiet floor. No second detector
   and no second clock.
4. **End of speech** as an event carrying the sample position at which
   the endpoint rules fired and the reading that stood at that moment.
   Nothing else is decoded at that point; the stream is the answer.
5. **Grammar changes** without a synthetic silence prime and without a
   time shift: constructing a recognizer for a new grammar is cheap
   enough to do on every change, and word positions stay in the samples
   of the audio actually fed.
6. **Reproducibility**: the same audio in the same blocks yields the
   same partials and the same readings, so a recorded session replays
   exactly. Dither off by default.
7. **Five languages** from the stock small models, no conversion step.
8. **The stock output shapes** where they exist, so the recording and
   replay tools keep parsing: the partial's key layout, the word
   objects, and the alternatives array, extended rather than replaced.
9. **Speaker evidence per word**: every word of the partial, of each
   reading, and of each final alternative carries a speaker vector over
   its own span, with that span in samples on the words' clock, absent
   when the span holds too little speech. The application scores it
   against its own profiles and decides, word by word, failing closed
   when it is absent. Enrolment reads one vector per final. The
   computation never delays the partial it rides on, and replays
   exactly.

Explicitly not needed: the final result. The application will not read
one. If the library offers a final for other consumers it must not be
the only place word times or the end of speech are reported.

## 6. Measurements the application already has

- A latency benchmark that feeds identical 40 ms blocks to two decoders
  and, for every word, records when it first appeared in a partial and
  when it stopped changing, relative to the word's acoustic end; the
  stock wheel scores a median 40 ms and a 90th percentile of 190 ms on
  32 recorded takes. This library must match it.
- A census script that counts, per session, how many partials carried
  more than one reading and how many of those differed in exactly one
  word position; the fork's figures were 12 of 3,387 and 1 of 3,387.
- Replay tooling that re-feeds a recorded WAV and diffs the decoder's
  events against the recording.

These run unchanged against this library once it presents the stock
shapes, which is why section 5 asks for them.
