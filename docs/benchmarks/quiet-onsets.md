# Quiet onsets: what a bound does to a word it can barely see

Run 2026-09-24 07:14:33Z, utter 765482d, utterpy 0.0.5, model vosk-model-small-en-us-0.15, 12th Gen Intel(R) Core(TM) i7-12700F, Python 3.14.4. Decoded by the binding `/home/jared/Repos/utter-bench-data/site-765482d/utterpy/__init__.py` built from utter 765482d7b70a6d3f5c4071e56e280b7d5df7b598, the checkout's HEAD. Pages under `docs` were modified at the time of the run.

The [states page](partial-states.md) measures the endpoint bound on words that sit 15 dB and more over the gap floor, so their first frames clear any floor margin at once. A microphone's do not. Here its pause streams are built again with every word's onset held a few dB over the -50 dBFS gap floor for its first 200 ms and rising to the word's own level over the next 200, pauses of 400-800 ms between words, and decoded by each engine in turn. The bound rows are spelled as the states page spells them, `MS/NATS` and, after a second slash, the floor margin in dB (`[[rr:TD-12#Decision outcome]]`).

*Lost* is a spoken word no final's entry covers and *decoded twice* one that two cover, as on the states page. *Misread* is a spoken word covered only by another; *gained* a final word covering no spoken one; *late* a covering entry starting more than 500 ms after the onset; *cut* a final landing between a word's onset and its offset. A rule that reads a forming onset as hiss shows in the last four.

| onset | engine | words | right | lost | decoded twice | misread | gained | late | cut | floor finals | pauses ended | finishes called |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| floor + 3 dB | utterpy | 606 | 487 / 606 (80.4%) | 71 | 2 | 48 | 0 | 11 | 75 | 0 | 283 / 406 (69.7%) | 181 / 200 (90.5%) |
| floor + 3 dB | utterpy@300/8 | 606 | 488 / 606 (80.5%) | 71 | 2 | 46 | 0 | 12 | 60 | 0 | 300 / 406 (73.9%) | 180 / 200 (90.0%) |
| floor + 3 dB | utterpy@300/8/8 | 606 | 488 / 606 (80.5%) | 71 | 2 | 46 | 0 | 12 | 60 | 0 | 300 / 406 (73.9%) | 180 / 200 (90.0%) |
| floor + 6 dB | utterpy | 606 | 509 / 606 (84.0%) | 53 | 3 | 43 | 0 | 14 | 83 | 0 | 287 / 406 (70.7%) | 185 / 200 (92.5%) |
| floor + 6 dB | utterpy@300/8 | 606 | 509 / 606 (84.0%) | 55 | 3 | 41 | 0 | 14 | 62 | 0 | 308 / 406 (75.9%) | 184 / 200 (92.0%) |
| floor + 6 dB | utterpy@300/8/8 | 606 | 510 / 606 (84.2%) | 55 | 3 | 40 | 0 | 14 | 62 | 2 | 308 / 406 (75.9%) | 184 / 200 (92.0%) |
| floor + 12 dB | utterpy | 606 | 543 / 606 (89.6%) | 27 | 5 | 34 | 0 | 15 | 86 | 0 | 302 / 406 (74.4%) | 191 / 200 (95.5%) |
| floor + 12 dB | utterpy@300/8 | 606 | 542 / 606 (89.4%) | 27 | 4 | 35 | 0 | 14 | 64 | 0 | 324 / 406 (79.8%) | 190 / 200 (95.0%) |
| floor + 12 dB | utterpy@300/8/8 | 606 | 544 / 606 (89.8%) | 26 | 3 | 35 | 0 | 14 | 64 | 5 | 325 / 406 (80.0%) | 190 / 200 (95.0%) |
