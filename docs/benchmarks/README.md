# Benchmarks

Public material the runtime is measured on, apart from the gates under
`docs/gates`, which use the first consumer's private recordings. A
benchmark here is reproducible by anyone with the dataset and a stock
Vosk model; each page names the script, the grammar and the figures.

General word-error benchmarks (LibriSpeech, TED-LIUM, Common Voice) do
not apply: the runtime decodes under a grammar only, by
`[[rr:TD-2#Scope]]`. Closed-vocabulary sets do.

| Page | Dataset | What it measures |
|---|---|---|
| `speech-commands.md` | Google Speech Commands v2 | accuracy under a command grammar with distractors, first-appearance latency, the unknown-word symbol and silence |
