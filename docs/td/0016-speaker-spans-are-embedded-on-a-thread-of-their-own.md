# TD-16: Speaker spans are embedded on a thread of their own, published at a deadline counted in audio

- Tags: speaker, evidence, streaming, threads, determinism, interface

## Context

Under `[[rr:TD-15#Embedding runs in slices between advances]]`, a word's
TitaNet evidence reaches a result a median 430 ms after the word ends,
540 ms at the 90th percentile, against 40 and 220 ms for the x-vector.
Measured on the speaker page's streams, the 430 ms is:

| part | median | p90 |
|---|---|---|
| word end until the decoder closes its span | 310 ms | 402 ms |
| span closed until a slice starts on it | 40 ms | 40 ms |
| slices until the embedding completes | 80 ms | 120 ms |
| embedding complete until the next result | 0 ms | 0 ms |

The second row is one block, always: the accept that queues a span
advances the decoder, and an advancing accept runs no slice. The third
is two or three idle accepts. Both come from keeping the embedding off
the decoder's blocks, which is what holds a TitaNet recognizer under
0.0.5's x-vector figures, `[[rr:TD-15#What a block may cost]]`.

Of the words with no evidence on that page, 66 of 128 are the last word
before a FinalResult: their span closes during the final's own flush,
and reading a final runs no embedding.

The first row is the decoder's rhythm, one advance in about six
accepts. Only embedding the open word on the audio already seen can
shorten it, and on the recognizer's thread that roughly doubles
TitaNet's compute, a real-time factor near 0.020 against the 0.0197
limit.

TD-15 rejected a second thread because when evidence arrived would
depend on the machine, and because the crate owns no thread. This
record keeps the first property and gives up the second.

## Decision drivers

- The same audio in the same blocks gives the same results, speaker
  keys and the result that first carries them included, on any machine.
- The thread is the standard library's, so the crate keeps zero
  dependencies, `[[rr:TD-2#Dependency policy]]`.
- A host can still choose TD-15's single thread by configuration, and
  gets exactly what TD-15 gives.

## Considered options

- **A worker the host runs on a thread of its own.** The crate would
  own no thread, and a host with a real-time audio thread or many
  recognizers would choose the core. But it could not be the default:
  a host that does nothing would get no second thread. Rejected for a
  thread the crate spawns, with single-threaded a configuration away.
- **Publish evidence as soon as the thread finishes.** Earliest, but
  which result first carries a word's vector would depend on the
  machine and its load. Rejected for a deadline counted in audio.
- **Publish at the next decoder advance.** Deterministic, but that
  advance comes about 245 ms after the one that queued the span, later
  than today's 120 ms. Rejected.
- **A pool shared by many recognizers.** Fewer threads for a host with
  many recognizers, but the crate would own a scheduler, and one slow
  stream would delay another's deadlines. Not taken here.

## Decision outcome

### Multi-threaded by default, single-threaded by configuration

A recognizer with a TitaNet model set embeds on a thread of its own,
spawned with `std::thread` when the model is set and joined when the
recognizer is dropped or the model is removed. Its jobs are the queued
spans of `[[rr:TD-15#The recognizer embeds a word once its span has closed]]`,
embedded with the stateless call's arithmetic, so their bytes are the
stateless call's, `[[rr:TD-15#Any span can be embedded from any thread]]`.
One thread serves one recognizer.

A host switches it off by configuration: `set_spk_threads(false)` in
Rust, `utter_recognizer_set_spk_threads` in C, before or after the model
is set. Single-threaded, the recognizer runs no thread and
`[[rr:TD-15#Embedding runs in slices between advances]]` applies
unchanged. Multi-threaded, the recognizer runs no slices.

The x-vector's evidence is computed as audio arrives and is not moved
to the thread.

### Evidence is published at a deadline counted in audio

A span queued at the accept that brings the audio fed to sample t is
published on the results read after the first accept that brings it
to t + D. At that accept, if the thread has not finished the span, the
accept waits for it. So a result carries a word's evidence at the same
point in the stream on every machine, and a slow thread costs the
decoder's thread only the wait.

D is decided by measurement, from one block of 40 ms upward: the
smallest D at which, on the speaker-cost streams fed in real time on
one spare core, no accept waits at the 99th percentile.

### A final waits for its words: decided by measurement

Two rules are built behind a switch:

- **As now.** A span still pending at a final is dropped, and its word
  carries no evidence.
- **The final waits.** A final queues the spans its flush closes and
  waits for every pending span before it is returned.

The final waits is kept if the final's block stays within the worst
block TD-15 allows on the speaker-cost streams multi-threaded.

### The open word: decided by measurement

Two rules are built behind a switch:

- **Closed spans only.** As TD-15: a word is queued once another entry
  follows it on the best path.
- **The open word too.** At each advance, the best path's last word is
  queued over the audio its span holds so far, and its evidence is
  published by the same deadline on that entry, keyed by its span as
  `[[rr:TD-15#The recognizer embeds a word once its span has closed]]`
  keys it. When the span closes it is queued again as a new span.

The open word too is kept if, multi-threaded, it gives evidence
on a partial earlier by a median of at least 150 ms, and the thread
keeps up at the deadline D above.

### What a block may cost

Multi-threaded, the recognizer's thread is held to 0.0.5 with
no speaker model, not with the x-vector: a 99th percentile block of
2.86 ms, a real-time factor of 0.0117, and its slowest block, on the
speaker-cost streams, waits included. The thread's own compute is
reported beside them, per second of audio and per core.

### Verification

- Multi-threaded, every result's bytes equal those with slices at the
  same publication points; a test forces the thread to lag and to lead
  and compares the results byte for byte.
- The same stream fed twice, one run with the thread on an idle core
  and one with the thread slowed by a sleep per job, gives identical
  results.
- The speaker page gains a row per rule above, with the evidence
  latency table TD-15's page carries.

## Consequences

- By default a TitaNet recognizer takes a second core's time and gives
  earlier evidence. A host that sets single-threaded gets TD-15's
  schedule and figures.
- The two modes publish evidence at different points in the stream, so
  their results differ in which result first carries a word's keys,
  never in the keys' bytes. Each mode is deterministic on its own.
- The recognizer and its thread share the queue behind a lock the host
  never observes; `accept` waits on it only at a deadline, which
  changes `[[rr:TD-2#Packaging]]`'s "never holds a lock a caller can
  observe" in the letter: the wait is a caller-visible stall, bounded
  by one embedding.
- Feeding faster than real time, as a batch host does, makes every
  deadline arrive early, and the accepts wait: the stream's wall time
  approaches the recognizer's plus the thread's. Such a host may prefer
  single-threaded.
- A recognizer is `Send` still; the thread holds only the model and
  the shared queue.
- utterpy exposes the switch.
- TD-15's rejected option "A second thread in the crate" is answered by
  this record.
