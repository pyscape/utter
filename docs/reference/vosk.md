# Coming from Vosk

utter's bindings keep the `vosk` package's names, so a program written
against the stock wheel runs against utter with the import changed.
This page lists every method Vosk has and where it stands in utter,
then the methods utter adds. What the results contain is on the
[results](results.md) page.

The Python column is [utterpy](https://github.com/pyscape/utterpy).
The Rust column is the `utter` crate. The C column is
[include/utter.h](../../include/utter.h), which mirrors `vosk_api.h`
with the `vosk_` prefix replaced by `utter_`.

## Vosk's methods

| vosk | utterpy | Rust | C | Status |
|---|---|---|---|---|
| `Model(path)` | `Model(path)` | `Model::open(&Path)` | `utter_model_new` | Same. Any small model with the stock layout; a model using an unimplemented network component is refused when opened. |
| `Model.vosk_model_find_word(word)` | `Model.FindWord(word)` | `model.word_ids.get(word)` | `utter_model_find_word` | Same: the word's id, or -1 when the model does not know it. |
| `KaldiRecognizer(model, rate)` | | | | Not included. Decoding against the model's full language model is out of scope; a grammar is required. |
| `KaldiRecognizer(model, rate, grammar)` | `KaldiRecognizer(model, rate, grammar, unknown_cost=None)` | `Recognizer::new(&model, rate, &grammar)` and `Recognizer::with_options` | `utter_recognizer_new_grm` and `utter_recognizer_new_grm_unk` | Same. The grammar is a JSON array of words or phrases in Python and C, a slice of strings in Rust. Fewer than 300 distinct words. |
| `SetWords(on)` | `SetWords(on)` | `set_words(on)` | `utter_recognizer_set_words` | Same: word entries on finals. |
| `SetPartialWords(on)` | `SetPartialWords(on)` | `set_partial_words(on)` | `utter_recognizer_set_partial_words` | Same key. Vosk switches to a lattice partial that trails the audio; utter's partial does not change. |
| `SetMaxAlternatives(n)` | `SetMaxAlternatives(n)` | `set_max_alternatives(n)` | `utter_recognizer_set_max_alternatives` | Same shape on finals. The confidences come from the beam, not a lattice. |
| `AcceptWaveform(bytes)` | `AcceptWaveform(bytes)` | `accept(&[i16]) -> Step` | `utter_recognizer_accept_waveform_s` | Same: 16-bit mono PCM at the recognizer's rate, true when an endpoint rule fired. Rust returns a `Step` with that flag and the decoded sample position. C has only the 16-bit form, not Vosk's float or byte forms. |
| `PartialResult()` | `PartialResult()` | `partial()` | `utter_recognizer_partial_result` | Same keys, [extended](results.md#the-partial-result). |
| `Result()` | `Result()` | `result()` | `utter_recognizer_result` | Same keys, [extended](results.md#the-final-result). |
| `FinalResult()` | `FinalResult()` | `final_result()` | `utter_recognizer_final_result` | Same keys, extended. |
| `Reset()` | `Reset()` | `reset()` | `utter_recognizer_reset` | Same. |
| `SetGrammar(grammar)` | | | | Not included. Build a new recognizer; the model caches the grammars it has compiled, so the second recognizer on the same grammar is cheap. |
| `SetSpkModel`, `SpkModel` | | | | Not included. No speaker vectors. |
| `SetNLSML(on)` | | | | Not included. |
| `SrtResult(stream)` | | | | Not included. |
| `SetLogLevel(level)` | `SetLogLevel(level)`, does nothing | `set_log_callback(f)` | | Differs. Rust takes a callback for the decoder's messages; the Python function is kept so scripts run unchanged. |
| `BatchModel`, `BatchRecognizer`, `GpuInit` | | | | Not included. No batch or GPU decoding. |

Word times, on by `SetWords` and `SetPartialWords`, are Kaldi's output
frame times, multiples of 30 ms. On finals they match the wheel's to
within one frame on 98.8% of words over the public Speech Commands
split, the start on 99.9%; the
[word-times](../benchmarks/word-times.md) page has the measurement. A
recognizer is not shared between threads in either library.

## What utter adds

| utterpy | Rust | C | What it does |
|---|---|---|---|
| `SetPartialAlternatives(n)` | `set_alternatives(n)` | `utter_recognizer_set_alternatives` | Carry up to n [readings](results.md#a-reading) beside every partial. Off at 0. |
| `SetEndpointBound(trailing_ms, extending_veto_nats=None)` | `set_endpoint_bound(Option<f32>, Option<f32>)` | `utter_recognizer_set_endpoint_bound` | One endpoint rule of your own beside the model's; see [Ending speech sooner](../guide/ending-speech-sooner.md). Off by default. |
| `DecodedSample()` | `decoded_sample()`, also `Step.sample` | `utter_recognizer_decoded_sample` | The sample position, in audio fed since construction, of the last decoded frame's end. |
| `KaldiRecognizer(..., unknown_cost=c)` | `RecognizerOptions::unknown_cost` | `utter_recognizer_new_grm_unk` | Add the model's `[unk]` symbol to the grammar at a cost, so out-of-grammar speech has somewhere to go. |
| | `Recognizer::endpoint_reason()` | | Which rule would close the utterance now, the value the final's `endpoint` key will carry. |

Every added key in the results is listed on the
[results](results.md) page with its origin column set to *added*.

## What stays the same in the output

With none of the added setters used, the `partial`, `text`, `result`,
`alternatives` keys and every word entry's `conf`, `start`, `end` and
`word` are the stock wheel's, on the same audio, block for block; the
gates under [docs/gates](../gates/) measure the agreement. The added
keys follow the stock keys in every document, so a parser that reads
by name never sees them and a parser that reads by position sees the
stock keys first.

Two visible differences remain. Where Vosk's partial text is empty,
utter's is `[sil]` or `[speech]`. And `conf` on a final's words is a
constant `1.0`, because there is no lattice to compute it from.
