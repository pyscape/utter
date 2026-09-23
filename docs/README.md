# Documentation

Where to start depends on where you are coming from.

| You are | Start with |
|---|---|
| Using the `vosk` package today | [Coming from Vosk](reference/vosk.md): every method, where it stands, and what is added. |
| A Rust engineer | `cargo doc --open`, or the crate docs once published. The quick start is on the crate page; each method that returns JSON links to the results reference. |
| Reading the JSON | [Results](reference/results.md): every document the recognizer returns and every key in it. |
| Building on the partials | [Reading a partial](guide/reading-a-partial.md), then [Ending speech sooner](guide/ending-speech-sooner.md). |

| Directory | Contents |
|---|---|
| [reference/](reference/) | What the API is: the results and the Vosk method map. |
| [guide/](guide/) | What to do with it: reading partials, ending speech sooner. |
| [benchmarks/](benchmarks/) | Public measurements on Speech Commands, each paired against the stock wheel. |
| [gates/](gates/) | The private parity gates and their results. |
| [td/](td/) | The technical decision records: why each part is built as it is. |
| [design/](design/target-speaker-extraction.md) | Proposed companion systems: streaming target speaker extraction with utter as its downstream consumer. |

The reference and guide pages describe behaviour. When they say why,
they link to a decision record rather than repeating it.
