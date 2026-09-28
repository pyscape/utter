# Small models against the stock wheel

Gates: `[[rr:TD-2#Verification and acceptance]]`, the G2 and G5 bars as `g2-g5-streaming.md` applies them to the English model, on each language's small model.

## What ran

- Run 2026-09-27 22:30:52Z, utter 36b0a5b, vosk 0.3.45, utterpy 0.0.5, model vosk-model-small-de-0.15, vosk-model-small-fr-0.22, vosk-model-small-es-0.42, vosk-model-small-ru-0.22, 12th Gen Intel(R) Core(TM) i7-12700F, Python 3.14.4. Decoded by the binding `site-765482d/utterpy/__init__.py` built from utter 765482d7b70a6d3f5c4071e56e280b7d5df7b598, **not** the checkout as it stands (36b0a5b).
- Models: the latest small model per language from alphacephei (Apache 2.0), stock configuration, both engines reading one copy whose `mfcc.conf` adds `--dither=0`.
- Corpus: Common Voice 7.0, Single Word Target segment (CC0), every validated clip of the language, converted from mp3 to 16 kHz mono by sox without dither. A take is up to 20 of one speaker's clips back to back, so a segment closes on the pauses the recordings carry.
- Grammar: the language's distinct prompts (the digits, yes, no, and the segment's wake words) that the model's vocabulary holds, no unknown-word symbol.
- 40 ms blocks, the partial read after every block, `Result` on every endpoint, `FinalResult` at the end, scored by the rules of `scripts/g2.py score`.

## Figures

Segment equality carries no bar here: it is set beside the English model's 193 / 205 (94.15%), which misses G2's 99% itself. The other three rows are pass or fail at 95%.

| | de | fr | es | ru |
|---|---|---|---|---|
| model | vosk-model-small-de-0.15 | vosk-model-small-fr-0.22 | vosk-model-small-es-0.42 | vosk-model-small-ru-0.22 |
| takes, clips, audio | 1,558, 13,648, 9.6 h | 2,449, 20,017, 14.1 h | 3,702, 22,982, 17.5 h | 206, 2,034, 1.2 h |
| grammar words | 15 | 14 | 14 | 13 |
| not in the model | none | none | none | firefox |
| partial text equal per block | 848,433 / 849,695 (99.85%), pass | 1,233,590 / 1,246,762 (98.94%), pass | 1,547,656 / 1,551,354 (99.76%), pass | 110,081 / 110,300 (99.80%), pass |
| segment word sequence equal | 14,689 / 14,875 (98.75%) | 22,083 / 23,066 (95.74%) | 25,849 / 26,258 (98.44%) | 2,006 / 2,033 (98.67%) |
| word times within 30 ms, equal segments | 13,437 / 13,505 (99.50%), pass | 20,200 / 20,596 (98.08%), pass | 22,544 / 22,672 (99.44%), pass | 1,850 / 1,857 (99.62%), pass |
| endpoints within 0.2 s | 13,155 / 13,317 (98.78%), pass | 19,919 / 20,617 (96.61%), pass | 22,178 / 22,556 (98.32%), pass | 1,803 / 1,827 (98.69%), pass |
| segments, vosk / utterpy | 14,875 / 14,872 | 23,066 / 23,100 | 26,258 / 26,254 | 2,033 / 2,034 |
| words only in vosk / only in utterpy / substituted | 160 / 156 / 16 | 975 / 867 / 75 | 386 / 375 / 6 | 25 / 24 / 0 |
| word error against the prompts, vosk / utterpy | 2.92% / 2.89% | 11.95% / 11.20% | 2.66% / 2.69% | 8.46% / 8.46% |

## Reading

Every model opens in both engines, and each passes the three bars with
room: partials at 98.9% or better, word times at 98.1% or better,
endpoints at 96.6% or better. Segment equality is above the English
model's 94.15% in all four languages. That comparison is loose, because
a take here is a speaker saying one short word per clip with the
recording's own pauses between clips, and G2's corpus is continuous
command speech.

French is the weakest on every row. Its disagreement is split between
words only in libvosk's finals (975) and words only in the runtime's
(867), with few substitutions, which is the near-tie pattern G2 found
in English rather than one side inventing words. Against the prompts,
the two engines' word error is within 0.8 points in every language.

The Russian model has no Latin-script words, so the segment's Firefox
prompt is left out of its grammar; its clips are still decoded.

Two traps for anyone re-running this. The release's German TSV spells
the prompt "null" as "nan", a missing value written back out. The Speech
Commands harness builds its grammar with an ASCII-escaped JSON dump,
which the stock wheel does not unescape: it drops every non-ASCII word
with a warning, all of Russian and German "fünf" among them, while
utterpy reads the grammar correctly. The script sends the grammar
unescaped to both engines.

The binding is the v0.0.5 build. The runtime's sources (src, build.rs,
Cargo.toml, Cargo.lock) are unchanged between it and the revision the
script ran at, so its figures stand for that revision.
