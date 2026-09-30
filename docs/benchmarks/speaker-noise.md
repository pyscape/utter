# Speaker evidence in a noisy room

Run 2026-09-30 09:18:40Z, utter 0.0.6 from acbbefd7acbe09185ee6b0dd5d5cd63e35ad9057 through its C ABI (checkout ca5450e), model vosk-model-small-en-us-0.15, speaker model titanet-small as scripts/titanet_convert.py writes it, 12th Gen Intel(R) Core(TM) i7-12700F, Python 3.14.4, blocks of 40 ms.

Speech Commands speakers each recorded on their own device, so the room a word was spoken in is part of what a profile learns. This page lays a room under the probe words and asks how far the speaker evidence moves when the profile was enrolled in the quiet. Speakers of the test split with at least 20 clips enrol from one clean stream of 10 clips; 10 other clips each are a probe stream, decoded clean and again with a background recording laid under each clip at 20 dB, 10 dB, 5 dB SNR, the SNR over the clip's 10 ms frames within 20 dB of its loudest. Each clip draws one recording and offset, the same at every SNR. The recordings: doing_the_dishes (95 s), dude_miaowing (62 s), exercise_bike (61 s), pink_noise (60 s), running_tap (61 s), white_noise (60 s); pink and white noise are generated. Streams decode under the Speech Commands page's grammar.

210 of 210 speakers enrolled under the recognizer row and 210 under embed(), from 1835 of 2037 enrolment words with evidence. Probe words with evidence: clean 1833 of 2043, 20 dB 1758 of 1981, 10 dB 1481 of 1722, 5 dB 1097 of 1330.

Each stream runs through a recognizer with TitaNet-small set. Every spoken word entry of its finals is a word, and one with evidence is a probe whatever its label; *no evidence* is the share of words without. The *recognizer* row scores the entry's own vector; the *embed()* row scores utter_spk_model_embed over the entry's spk_start to spk_end, as a host that embeds the spans itself would. Each row enrols from its own vectors. Beside single words, a speaker's probe words are taken, 3 times in shuffled orders, until their evidence has pooled 0.5 s, 1 s, 2 s: the recognizer row averages their vectors weighted by frames, the embed() row embeds their spans joined. The last word taken may carry the pool past the length.

*Can't tell* is a score above the threshold that turns away 1% of own-profile scores and not above the one that lets 1% of other-profile scores through: a host accepts above the band and rejects below it, each at a 1% error, and asks for more speech inside it. When the two thresholds cross the band is empty. The cell gives the share of own-profile scores in the band, then of other-profile scores. *Accepted at 1% false accept* is at the row's own threshold; the gate and the band are the clean row's at the same length, as a host sets them in the quiet.

| speech | condition | row | items | no evidence | equal error rate | accepted at 1% false accept | accepted at the gate | false accept at the gate | can't tell |
|---|---|---|---|---|---|---|---|---|---|
| one word | clean | recognizer | 2043 | 10.3% | 4.8% | 77.7% | 77.7% | 1.0% | 12.4%, 20.6% |
| one word | clean | embed() | 2043 | 10.3% | 4.8% | 77.7% | 77.7% | 1.0% | 12.4%, 20.6% |
| one word | 20 dB | recognizer | 1981 | 11.3% | 7.8% | 63.6% | 60.5% | 0.8% | 28.8%, 17.5% |
| one word | 20 dB | embed() | 1981 | 11.3% | 7.8% | 63.6% | 60.5% | 0.8% | 28.8%, 17.5% |
| one word | 10 dB | recognizer | 1722 | 14.0% | 11.4% | 48.4% | 40.1% | 0.5% | 44.6%, 14.7% |
| one word | 10 dB | embed() | 1722 | 14.0% | 11.4% | 48.4% | 40.1% | 0.5% | 44.6%, 14.7% |
| one word | 5 dB | recognizer | 1330 | 17.5% | 13.6% | 35.9% | 24.8% | 0.3% | 55.3%, 12.6% |
| one word | 5 dB | embed() | 1330 | 17.5% | 13.6% | 35.9% | 24.8% | 0.3% | 55.2%, 12.6% |
| 0.5 s | clean | recognizer | 627 | 0.0% | 2.9% | 93.0% | 93.0% | 1.0% | 5.9%, 5.5% |
| 0.5 s | clean | embed() | 627 | 0.0% | 2.7% | 94.3% | 94.3% | 1.0% | 4.6%, 5.1% |
| 0.5 s | 20 dB | recognizer | 624 | 0.0% | 4.8% | 82.9% | 81.6% | 0.8% | 14.1%, 4.6% |
| 0.5 s | 20 dB | embed() | 624 | 0.0% | 3.7% | 88.5% | 84.1% | 0.8% | 13.1%, 3.9% |
| 0.5 s | 10 dB | recognizer | 612 | 0.0% | 7.0% | 77.1% | 68.3% | 0.5% | 20.9%, 3.5% |
| 0.5 s | 10 dB | embed() | 612 | 0.0% | 5.8% | 78.6% | 68.5% | 0.5% | 22.2%, 3.0% |
| 0.5 s | 5 dB | recognizer | 576 | 0.0% | 8.9% | 65.5% | 50.0% | 0.4% | 30.6%, 2.9% |
| 0.5 s | 5 dB | embed() | 576 | 0.0% | 7.8% | 68.2% | 51.2% | 0.4% | 30.7%, 2.3% |
| 1 s | clean | recognizer | 624 | 0.0% | 1.6% | 97.9% | 97.9% | 1.0% | 0.8%, 2.5% |
| 1 s | clean | embed() | 624 | 0.0% | 1.6% | 97.9% | 97.9% | 1.0% | 1.0%, 0.8% |
| 1 s | 20 dB | recognizer | 621 | 0.0% | 2.3% | 93.4% | 92.1% | 0.8% | 5.6%, 2.0% |
| 1 s | 20 dB | embed() | 621 | 0.0% | 2.1% | 96.8% | 94.0% | 0.7% | 3.1%, 0.4% |
| 1 s | 10 dB | recognizer | 606 | 0.0% | 4.5% | 85.0% | 78.7% | 0.5% | 12.4%, 1.5% |
| 1 s | 10 dB | embed() | 606 | 0.0% | 3.8% | 88.1% | 78.5% | 0.4% | 6.6%, 0.3% |
| 1 s | 5 dB | recognizer | 540 | 0.0% | 5.2% | 76.7% | 60.4% | 0.3% | 20.7%, 1.1% |
| 1 s | 5 dB | embed() | 540 | 0.0% | 4.3% | 82.4% | 68.1% | 0.3% | 7.6%, 0.2% |
| 2 s | clean | recognizer | 618 | 0.0% | 0.6% | 99.8% | 99.8% | 1.0% | 0.0%, 0.0% |
| 2 s | clean | embed() | 618 | 0.0% | 1.0% | 99.0% | 99.0% | 1.0% | 0.0%, 0.0% |
| 2 s | 20 dB | recognizer | 606 | 0.0% | 1.3% | 98.2% | 97.9% | 0.8% | 0.0%, 0.0% |
| 2 s | 20 dB | embed() | 606 | 0.0% | 1.5% | 97.7% | 96.5% | 0.6% | 0.0%, 0.0% |
| 2 s | 10 dB | recognizer | 528 | 0.0% | 1.6% | 95.5% | 91.5% | 0.5% | 0.0%, 0.0% |
| 2 s | 10 dB | embed() | 528 | 0.0% | 1.9% | 95.8% | 90.7% | 0.4% | 0.0%, 0.0% |
| 2 s | 5 dB | recognizer | 318 | 0.0% | 3.1% | 88.1% | 77.4% | 0.3% | 0.0%, 0.0% |
| 2 s | 5 dB | embed() | 318 | 0.0% | 1.9% | 95.6% | 80.5% | 0.3% | 0.0%, 0.0% |
