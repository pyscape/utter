# TD-8: Silence is the host's gate, on a floor the runtime reports

- Tags: silence, endpointing, evidence, finals, parity, measurement

## Context

The first consumer's complaint is that silence produces words:
`[[rr:4. What goes wrong today]]`. `[[rr:TD-2#Silence and unknown speech
announce themselves]]` fixed the shape of what the runtime says on
silence, and `[[rr:TD-2#The decoder: per-word energy and silence]]` ruled
that the runtime supplies evidence and withholds nothing. What was left
open is what the decoder actually does on a quiet room, which of the
signals it emits a host can act on, and how a host gates a word without
a threshold that fails on the next microphone.

The gate corpus answers the first question, `[[rr:What the 851 are]]`
and `[[rr:Where the silence finals come from]]`:

- **The stream shows no phantom.** Every word at rank 0 on a quiet block
  was spoken: seven in ten are still held from earlier in the utterance
  because no endpoint has fired, the rest have just ended and straddle
  the block. The quietest of them sits 8.5 dB above the take's floor.
  libvosk shows a word on the same blocks, and on no block does only one
  engine show one.
- **The phantom lives in finals.** On a quiet room the best path sits in
  the opening phones of some word without reaching its label. No silence
  phone is on the path, trailing silence stays at zero, and the rules
  that wait for it never fire. The 20 s utterance rule closes the
  stretch, and the final, which must end in a final state, completes the
  word: one word spanning the whole utterance, at the floor, on no
  partial. libvosk emits the same word on every such segment. When the
  path does sit in silence phones the 5 s rule closes the stretch with no
  word.
- **The floor moves.** The quietest 100 ms of a take ranges over 14 dB
  across one corpus from one microphone, and the public dataset's
  background recordings run from 12 dB below that corpus's room to 41 dB
  above it. A level that separates a word from silence on one take is
  wrong on the next.
- **An endpoint needs a clock.** After the last word appears its
  `stable_ms` grows while the speaker pauses between words exactly as it
  grows when the speaker has finished, and on this corpus the gaps
  between words overlap the trailing silences the endpoint rules wait
  for. The trailing `[sil]` entry is the decoder's own trailing-silence
  measure and separates the two cases.

### Prior art on silence and end of speech

Nobody puts the gate in a Kaldi decoder. Vosk carries no voice activity
detection by design (vosk-api #836), its maintainers answer the
silence-word complaint (#837) with training data or a newer model, and
they say of the unknown-word symbol that it guarantees nothing (#319).
Online CMVN and the i-vector normalise room tone up to speech level
before the network sees it, which is why energy cannot protect the
decoder from inside. Where a fix exists it is outside: sherpa-onnx opens
a fresh stream per detector segment, the Whisper front ends run Silero
ahead of the model, keyword spotters score a word against a filler or
background path and reject on the ratio.

Every energy gate that travels is relative to something tracked.
Rabiner and Sambur set the threshold at four times the opening 100 ms,
Kaldi's `compute-vad` at half the mean log energy plus a constant,
WebRTC's detector keeps a running minimum over the last second and
updates its noise model only on frames it judged noise. The absolute
thresholds in the field are on model scores, not levels.

Every production endpointer pairs a silence clock with a second signal
and none endpoints on hypothesis stability alone. Stability commits a
partial: Google's `stability`, Azure's stable-partial count, AWS's
partial-result stabilisation, LocalAgreement in the Whisper streamers.
Deepgram's word-gap `UtteranceEnd` is a fallback behind its silence
endpointer with a one-second minimum. Maas et al. (2018) measured the
reason: hypothesis-unchanged features have no clock of their own and
need a pause bound, or the ninetieth percentile lands at 1.5 s. Decoder
side evidence, trailing blanks or silence on the traceback, beats an
acoustic detector by one to two hundred milliseconds in every paper
that reports both, which is the evidence Kaldi's rules already read.

## Considered options

- **A voice activity detector ahead of the decoder, feeding it speech
  only.** What sherpa-onnx and the Whisper front ends do. Rejected: word
  positions leave the audio's clock and a replay no longer reproduces a
  session, the two-clock defect of `[[rr:4. What goes wrong today]]`, and
  libvosk decodes the silence, so the output diverges from the oracle.
- **Fewer words on quiet blocks from the decoder.** Rejected: the oracle
  shows the same words on the same blocks, and every one was spoken.
- **The unknown-word symbol.** Vosk's standard answer. Measured in
  `[[rr:The unknown-word symbol against silence phantoms]]`: at most a
  tenth of the quiet-block words and none of the silence finals before
  real words go. Stays an option, off.
- **The `[sil]` rival as the silence signal.** On a quiet room `[sil]`
  leads and there is no rival to read; when a word leads a quiet block it
  was spoken and the empty path lost long ago, six blocks in 851. The
  rival is the contest at the moment a word appears,
  `[[rr:TD-2#The decoder: partial alternatives]]`.
- **The trailing `[sil]` entry as the quiet-room signal.** On nine quiet
  blocks in ten with no word the path holds no silence phone, so there is
  no entry. It is the after-a-word clock, present on 87% of quiet blocks
  that hold a word, and that is the use it is put to.
- **Hypothesis stability as the endpoint.** A partial whose words stop
  changing while `stable_ms` grows is what a finished utterance looks
  like and also what a pause between two words looks like: a 500 ms hold
  mistakes 15% of this corpus's inter-word gaps for a finish, a 400 ms
  hold half of them. Rejected, with the literature above.
- **A peak or tail energy beside the mean.** Rejected by measurement: the
  mean over the span is the sharpest gate on silence finals. A third of
  them hold a breath or a click that puts a 100 ms peak 12 dB over the
  floor, and the last 200 ms is coarser than the mean.
- **An absolute quiet floor in dBFS.** The fork's configuration, and the
  shape of the Whisper and Silero thresholds. Rejected: it does not
  travel.
- **The host tracks the floor.** Rejected: every host writes the same
  tracker over PCM it has already handed over, and the runtime holds that
  PCM on the one clock.
- **A duration guard in the runtime, or a report of which endpoint rule
  fired.** Rejected: a spoken word can carry a span that absorbed the
  silence after it, sixteen of 424 on the corpus, so a bound alone
  decides wrongly, and the rule number says nothing the span and the
  floor do not.

## Decision outcome

### The decoder reads silence as libvosk does

No detector, no threshold and no suppression in the decoder. The measure
of silence handling is agreement with the oracle on quiet blocks and on
silence finals, `[[rr:libvosk reads the same 851]]`. A change that lowers
the count of words on quiet blocks or the count of silence finals moves
away from the oracle and is a record of its own, as
`[[rr:TD-6#Decision outcome]]` is, never a fix.

### A word on a quiet block was spoken and is reported

The partial reads `[sil]` with an empty word list while the best path
holds neither a word nor a silence phone, which on a quiet room is most
of the time. A word at rank 0 on a quiet block is held from earlier in
the utterance or has just ended; it is reported with its span, its
energy and its hold, and is never withheld. `[sil]` entries between and
after words carry the pause's span and energy, `[[rr:entries]]`; the
rival is the contest for a word at the moment it appears.

### A silent stretch closes as libvosk closes it

Kaldi's rules in `[[rr:TD-2#Inputs: configuration]]` close a silent
stretch either after 5 s of silence phones with no word or at the 20 s
cap with one word spanning the utterance, `[[rr:endpoint_detected]]`.
Both stay. The wordless final's text reads `[sil]` under
`[[rr:TD-2#Silence and unknown speech announce themselves]]` where
libvosk's reads the empty string; a host that compares finals to
libvosk's removes bracketed tokens first, as the gates do.

### A final word carries its energy

With words on, each final word carries `energy_dbfs`, the mean RMS in
dBFS of the PCM under its span, as a partial word does,
`[[rr:energy_dbfs]]`. A final that closes a silent stretch thereby says
so itself.

### The runtime reports the floor

Every partial and every final carries `floor_dbfs`: the 5th percentile
of RMS over 100 ms windows at a 50 ms hop across the last 10 s of audio
fed, absent until 100 ms has been fed. A window of 100 ms is longer than
a pitch period and shorter than a word; 10 s spans the pauses between
commands and follows a change of room within seconds; the 5th percentile
survives a quarter second of digital silence or a click where a minimum
does not. The floor is a property of the audio, not of the decode, so it
runs on across utterances and across pipeline rebuilds, and libvosk has
no such field, so it is additive.

### The gate is the host's and is relative to the floor

A host treats a word as silence when `energy_dbfs` is within its margin
of `floor_dbfs`. The margin is the host's; the corpus gives the operating
points, `[[rr:Gates a host can apply to a final]]`: 8 dB catches 98% of
the silence finals for 2.6% of other final words, 4 dB 87% for 1.7%,
12 dB all of them for 6.8%. A span longer than two seconds corroborates,
since every silence final has one, and does not decide, since spoken
words absorb silence too. The runtime never applies the margin.

### End of speech is the endpoint, and the trailing silence entry is its clock

End of speech is the endpoint the model's rules fire, reported in `Step`
with its sample, `[[rr:TD-2#The decoder: endpointing]]`. A host that
wants it sooner than the rules allow reads the span of the trailing
`[sil]` entry, the same trailing silence the rules read, and applies its
own bound, `[[rr:Reading the end of speech]]`: at 300 ms it calls 87% of
finishes before the endpoint, by a median 280 ms, and mistakes 4.5% of
the corpus's inter-word pauses for a finish. `stable_ms` is the hold a
host applies before acting on a word, `[[rr:update_stable]]`, and is not
an end-of-speech signal.

### A host may add one endpoint bound of its own, off by default

Beside the model's rules the recognizer takes one bound of the host's,
`[[rr:set_endpoint_bound]]`: a final once the trailing silence reaches
it. Unset, the runtime is byte-identical to the stock rules, so parity
holds; set, it calls a final where the rule above says a host would
read the trailing `[sil]` entry, and pays for it in G5 and in the
pauses it mistakes for a finish. The span cannot see a word beginning:
the word's label is not yet on the best path while the silence before
it still counts, so the span is highest as the next word starts and a
bare bound fires into it. The beam can see it, in the readings that
extend the partial by a further word, and the bound takes a veto: no
final while such a reading is within a margin the host names. The
benchmarks run the bound, the vetoed bound and the stock rules side by
side (`utterpy@300` and `utterpy@300/8` in the harness). On the built
streams the stock rules end 942 utterances inside a pause, the bare
300 ms bound 1,208, the bound vetoed at 4 nats 1,207 and at 8 nats
1,159: the readings that extend the partial sit four to eight nats
behind the leader as a word begins, so the margin has to reach them.
Both numbers are the host's; the runtime ships neither.

### Measurements count a word whose own span carries no speech

A silence word, in a partial or in a final, is a word whose span's mean
energy is within the margin of the floor; a held word and a straddling
word are counted apart from it, not as phantoms. The public benchmark's
noise pass feeds each recording continuously through one recognizer, so
the 5 s and 20 s rules fire as they do in use, and reports silence finals
per minute, on which both engines are expected equal.

## Consequences

- The recognizer gains `energy_dbfs` on final words and `floor_dbfs` on
  every result. Both are additions to libvosk's shapes, and byte
  identity of partial text, segments and word times over the gate corpus
  holds before and after, with `--partial-words` as a second
  configuration so the new fields are exercised.
- `scripts/unk_sweep.py` counts by the definitions above and the
  benchmark's noise pass runs continuously; the benchmark page is rerun.
  The README says where a host reads the gate and the clock once the
  fields exist.
- A host gets one margin that travels between microphones, one clock for
  end of speech that it can shorten at a measured cost, and evidence on
  finals equal to the evidence on partials. The 851 stops reading as a
  defect.
- A host must hold a margin. The floor drifts upward through continuous
  speech longer than the history. A final that closes a silent stretch
  still carries a word to a host that reads finals, as libvosk's does.

## Implemented by

- `[[rr:entries]]`, `[[rr:endpoint_detected]]` and `[[rr:energy_dbfs]]`:
  the entries, the closures and the energy as they stand.
- `[[rr:final_json]]`: the energy on each final word.
- `[[rr:FloorTracker]]`: the floor, on every partial and every final.
- `[[rr:scripts/unk_sweep.py#main]]` and `[[rr:noise_pass]]`: the
  measurements.
