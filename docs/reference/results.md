# Results

Every method that returns a result returns a JSON string, and the
Python, Rust and C bindings return the same string for the same audio.
This page lists each document the recognizer produces and every key in
it. The guide pages say what to do with them; the decision records
under [docs/td](../td/) say why each added key exists.

The key layout is libvosk's. Every key stock Vosk emits is emitted
here with the same name and type, so a parser written for Vosk keeps
working, and the keys utter adds come after the keys Vosk has. A key
marked *Vosk* below is in the stock wheel's output; a key marked
*added* is new.

Contents:

- [The partial result](#the-partial-result)
- [A reading](#a-reading)
- [A word entry](#a-word-entry)
- [The final result](#the-final-result)
- [The endpoint value](#the-endpoint-value)
- [Speaker evidence](#speaker-evidence)
- [Clocks and units](#clocks-and-units)
- [The empty results](#the-empty-results)

## The partial result

Returned by `PartialResult` in Python, `partial` in Rust and
`utter_recognizer_partial_result` in C. Current to the last decoded
frame, after every `AcceptWaveform` that did not end the utterance.

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
    {"end": 12.63, "start": 12.30, "word": "alpha",
     "start_sample": 196800, "end_sample": 202080, "energy_dbfs": -23.2, "stable_ms": 960},
    {"end": 12.98, "start": 12.63, "word": "seven",
     "start_sample": 202080, "end_sample": 207680, "energy_dbfs": -22.8, "stable_ms": 240},
    {"end": 13.02, "start": 12.98, "word": "[sil]",
     "start_sample": 207680, "end_sample": 208320, "energy_dbfs": -48.6, "stable_ms": 0}
  ],
  "floor_dbfs": -51.7
}
```

| Key | Type | Present | Origin | Meaning |
|---|---|---|---|---|
| `partial` | string | always | Vosk | The words on the best path, space separated. Where Vosk gives an empty string, utter gives `[sil]` when the path sits on silence and `[speech]` when it has entered a word the grammar cannot yet name. |
| `partial_alternatives` | array of [readings](#a-reading) | `SetPartialAlternatives(n)` with n above 0 | added | The distinct word sequences still alive in the search, best first, at most n of them. |
| `partial_result` | array of [word entries](#a-word-entry) | `SetPartialWords(True)` | Vosk | The best path as word entries, including its `[sil]` and `[speech]` entries. Each entry carries the added evidence keys. |
| `floor_dbfs` | number | once enough audio has been fed to measure it | added | The room's noise floor over the last ten seconds, in dBFS. Absent while the quietest windows of those seconds are digital silence, which has no level: a gate relative to the floor has nothing to compare against. |
| `spk`, `spk_frames`, `spk_start`, `spk_end` | see [speaker evidence](#speaker-evidence) | `SetSpkModel`, and enough speech on the path | added | The speaker evidence over the best path's words and `[speech]`. |

Turning on partial words in stock Vosk switches it to a lattice-based
partial that trails the audio. utter's partial is the same either way.

## A reading

One entry of `partial_alternatives`.

| Key | Type | Present | Origin | Meaning |
|---|---|---|---|---|
| `text` | string | always | added | The reading's words, or `[sil]` or `[speech]` as for `partial`. |
| `confidence` | number | always | added | The reading's path score in nats: a raw log-likelihood, not a probability. Compare readings to each other; never read one alone. |
| `result` | array of [word entries](#a-word-entry) | `SetPartialWords(True)` | added | The reading's words with their spans. These entries carry the span keys and the speaker evidence, not the energy or the hold. |
| `relation` | string | always | added | How the reading relates to `partial` as a word sequence: `same`, `prefix` when it lacks the partial's tail, `extends` when it has more words, `differs` otherwise. |
| `lead_delta` | number or null | always | added | How much the reading's lead over the field moved since the last decoding advance, in nats. `null` when the reading has no history yet. Treat `null` as unknown, not as zero. |

The readings are current to the last decoded frame, not to a lattice
that trails the audio. A competitor absent from the list is unknown,
not disproved.

## A word entry

One entry of `partial_result`, of a reading's `result`, or of a
final's `result`. Which keys an entry carries depends on where it sits.

| Key | Type | Present | Origin | Meaning |
|---|---|---|---|---|
| `word` | string | always | Vosk | The word, or `[sil]` for a run of silence phones, or `[speech]` for the phones of a word the path has entered and not yet named. Finals carry words only. |
| `start`, `end` | number | always | Vosk | The word's span in seconds since the recognizer was built. |
| `start_sample`, `end_sample` | integer | always | added | The same span in samples of the audio you fed, on the clock your own code keeps. |
| `conf` | number | finals only | Vosk | Always `1.0`. Vosk computes this from a lattice; utter has no lattice, and keeps the key so parsers do not break. Use the alternatives' `confidence` instead. |
| `energy_dbfs` | number or null | `partial_result` and finals | added | The loudness of the audio under the word, in dBFS. Tells a spoken word from one the decoder read into a quiet room. `null` when there is no signal under the word at all, digital silence or no samples: nothing was said there. |
| `stable_ms` | integer | `partial_result` and finals | added | How long the word has held its place in the partial. On a final, how long it had held when the final was cut; zero means no partial ever showed it. |
| `spk`, `spk_frames`, `spk_start`, `spk_end` | see [speaker evidence](#speaker-evidence) | every list, words and `[speech]`, with `SetSpkModel` and enough speech in the span | added | The speaker evidence over this entry's own span. `[sil]` entries carry none. |

The trailing `[sil]` entry ends at the last frame the decoder has
processed, not at the last sample you fed, so it runs one chunk behind
the audio. A `[speech]` entry after it is the decoder leaving silence
for a word's first phone, where its own endpoint rules stop counting;
the `[sil]` entry keeps its span and stops growing.

## The final result

Returned by `Result` and `FinalResult` in Python, `result` and
`final_result` in Rust, `utter_recognizer_result` and
`utter_recognizer_final_result` in C. `Result` closes the utterance
decoded so far and the next `AcceptWaveform` starts a new one.
`FinalResult` flushes the audio still in the pipeline first.

The plain shape, with `SetWords(True)`:

```json
{
  "result": [
    {"conf": 1.0, "end": 12.63, "start": 12.30, "word": "alpha",
     "start_sample": 196800, "end_sample": 202080, "energy_dbfs": -23.2, "stable_ms": 960},
    {"conf": 1.0, "end": 12.98, "start": 12.63, "word": "seven",
     "start_sample": 202080, "end_sample": 207680, "energy_dbfs": -22.8, "stable_ms": 580}
  ],
  "text": "alpha seven",
  "endpoint": "rule2",
  "floor_dbfs": -51.7
}
```

| Key | Type | Present | Origin | Meaning |
|---|---|---|---|---|
| `text` | string | always | Vosk | The words of the utterance. Empty when nothing was said. |
| `result` | array of [word entries](#a-word-entry) | `SetWords(True)` | Vosk | The words with their spans and evidence. `[sil]` and `[speech]` never appear here. |
| `endpoint` | string | always | added | What closed the utterance. See [the endpoint value](#the-endpoint-value). |
| `floor_dbfs` | number | once enough audio has been fed to measure it | added | As on the partial. |
| `spk`, `spk_frames` | see [speaker evidence](#speaker-evidence) | `SetSpkModel`, and enough speech in the words | Vosk | The speaker evidence over the utterance's words, before `text` as in Vosk. Vosk's comes from its own frame selection and normalisation and differs in value. |
| `spk_start`, `spk_end` | integer | with `spk` | added | The span `spk` pools, after the other keys. |

The alternatives shape, with `SetMaxAlternatives(n)` and n above 1,
replaces `text` and `result` with a list, as Vosk does:

```json
{
  "alternatives": [
    {"confidence": 1.94,  "result": [ ... ], "text": "alpha seven"},
    {"confidence": -3.32, "result": [ ... ], "text": "alpha eleven"}
  ],
  "endpoint": "rule2",
  "floor_dbfs": -51.7
}
```

| Key | Type | Present | Origin | Meaning |
|---|---|---|---|---|
| `alternatives` | array | `SetMaxAlternatives(n)` with n above 1 | Vosk | Up to n readings of the utterance, best first. |
| `alternatives[].confidence` | number | always | Vosk | The path score in nats. Vosk's figure comes from a lattice; utter's from the beam. |
| `alternatives[].result` | array of [word entries](#a-word-entry) | `SetWords(True)` | Vosk | The reading's words, with the same keys as the plain shape's `result`. |
| `alternatives[].text` | string | always | Vosk | The reading's words. |

With `SetSpkModel` the alternatives shape also carries the four
speaker keys after `floor_dbfs`, over the first alternative's words.
Vosk carries none in this shape.

## The endpoint value

The `endpoint` key on a final names what closed it.

| Value | What closed the utterance |
|---|---|
| `rule1` to `rule4` | One of the model's silence rules from `conf/model.conf`: trailing silence of a length each rule sets, with a demand on the best path's cost that loosens as the silence grows. `rule1` fires on a long silence with no word yet. |
| `rule5` | The model's length cap, 20 s of audio in the stock models. A word in a `rule5` final with `stable_ms` of zero was forced onto the stretch by the cap and never shown in a partial. |
| `bound` | Your own bound from `SetEndpointBound`. |
| `floor` | Your own bound, reached through the floor margin of `SetEndpointFloorMargin`: the path ended in silence or in a `[speech]` entry within the margin of the floor. The final is the path as it stood, so a wordless one has an empty `result` and the text `[sil]` or `[speech]`. |
| `flush` | `FinalResult`: the stream was flushed. |
| `host` | `Result` was called with no rule fired. |

The rules and the bound are described in the guide page
[Ending speech sooner](../guide/ending-speech-sooner.md).

## Speaker evidence

With a speaker model set, `SetSpkModel` in Python, `set_spk_model` in
Rust and `utter_recognizer_set_spk_model` in C, a speaker vector rides
every partial and final and every word entry, each over its own span.
A span with less than a quarter second of speech carries none: the
keys are absent, and absent keys mean no evidence, not a stranger.
Why it is built this way is `[[rr:TD-14]]`.

```json
{"word": "seven", "start_sample": 202080, "end_sample": 207680,
 "spk": [1.12, -0.48, ...], "spk_frames": 35,
 "spk_start": 202080, "spk_end": 207680}
```

| Key | Type | Meaning |
|---|---|---|
| `spk` | array of numbers | The speaker vector, as long as the model's embedding: 128 for `vosk-model-spk-0.4`. Read its length from the array. Compare two vectors by cosine; the scale is not a probability. |
| `spk_frames` | integer | The 10 ms frames pooled into the vector. More frames, steadier evidence. |
| `spk_start`, `spk_end` | integer | The first pooled frame's start and the last one's end, in samples on the clock of `start_sample`. A word's last 70 ms reach its evidence one partial later, so on a partial the span can end before the word does. |

To enrol a speaker, run their recordings through a recognizer with the
same speaker model and average the vectors of the finals; to identify
one, score a word's vector against each enrolled average. utter reports
the vector and never names a speaker. A floor margin set with
`SetEndpointFloorMargin` also keeps frames within it of the floor out
of the vectors. The
[speaker evidence page](../benchmarks/speaker-evidence.md) measures how
well the stock model identifies speakers from one word and from more.

## Clocks and units

- `start` and `end` are seconds since the recognizer was built,
  Vosk's clock. `start_sample` and `end_sample` are samples fed since
  the recognizer was built. They are the same instants in two units.
- `confidence`, `lead_delta` and the veto in `SetEndpointBound` are
  nats, the natural-log unit of the decoder's costs. A gap of 2.2 nats
  between two readings is a likelihood ratio of about nine.
- `energy_dbfs` and `floor_dbfs` are decibels relative to full scale
  of 16-bit audio, so both are negative and a louder sound is nearer
  zero. Set thresholds on their difference, which travels between
  microphones.
- `stable_ms` is milliseconds of audio fed, not wall-clock time.
- `AcceptWaveform` in Python and C returns true when an endpoint rule
  fired. Rust's `accept` returns a `Step` whose `endpoint` field is
  that flag and whose `sample` field is the sample position of the
  last decoded frame's end, the same figure `DecodedSample` and
  `utter_recognizer_decoded_sample` return.

## The empty results

Before the first frame has been decoded the partial is
`{"partial": "[sil]"}`, with `floor_dbfs` once it is known. A final
with nothing decoded, a second call to `Result` or `FinalResult` with
no audio between them, and the result after `Reset` are all
`{"text": ""}`, as in Vosk.
