# Speaker evidence on a shared microphone

Run 2026-09-30 09:21:17Z, utter 0.0.6 from acbbefd7acbe09185ee6b0dd5d5cd63e35ad9057 through its C ABI (checkout ca5450e), model vosk-model-small-en-us-0.15, speaker model titanet-small as scripts/titanet_convert.py writes it, 12th Gen Intel(R) Core(TM) i7-12700F, Python 3.14.4, blocks of 40 ms.

Speech Commands speakers each recorded on their own device, so a profile carries the microphone as well as the voice. In the CSTR VCTK Corpus 0.92 (CC BY 4.0) every speaker read into the same microphones in the same room; this page uses the DPA 4035 channel (mic1), resampled from 48 kHz to 16 kHz. Each speaker enrols from one stream of 10 utterances, and 10 other utterances are the probe stream. utter decodes under a grammar only, so a stream's grammar is the words of its own utterances' prompts; the model's full vocabulary is not open to it.

109 of 109 speakers enrolled under the recognizer row and 109 under embed(), from 3438 of 7411 enrolment words with evidence. Probe words with evidence: clean 3565 of 7491.

Each stream runs through a recognizer with TitaNet-small set. Every spoken word entry of its finals is a word, and one with evidence is a probe whatever its label; *no evidence* is the share of words without. The *recognizer* row scores the entry's own vector; the *embed()* row scores utter_spk_model_embed over the entry's spk_start to spk_end, as a host that embeds the spans itself would. Each row enrols from its own vectors. Beside single words, a speaker's probe words are taken, 3 times in shuffled orders, until their evidence has pooled 0.5 s, 1 s, 2 s: the recognizer row averages their vectors weighted by frames, the embed() row embeds their spans joined. The last word taken may carry the pool past the length.

*Can't tell* is a score above the threshold that turns away 1% of own-profile scores and not above the one that lets 1% of other-profile scores through: a host accepts above the band and rejects below it, each at a 1% error, and asks for more speech inside it. When the two thresholds cross the band is empty. The cell gives the share of own-profile scores in the band, then of other-profile scores. The gate and the band are each row's own.

| speech | condition | row | items | no evidence | equal error rate | accepted at 1% false accept | accepted at the gate | false accept at the gate | can't tell |
|---|---|---|---|---|---|---|---|---|---|
| one word | clean | recognizer | 7491 | 52.4% | 6.5% | 37.5% | 37.5% | 1.0% | 20.3%, 28.4% |
| one word | clean | embed() | 7491 | 52.4% | 6.5% | 37.5% | 37.5% | 1.0% | 20.3%, 28.4% |
| 0.5 s | clean | recognizer | 327 | 0.0% | 2.4% | 93.9% | 93.9% | 1.0% | 4.9%, 2.8% |
| 0.5 s | clean | embed() | 327 | 0.0% | 1.6% | 97.9% | 97.9% | 1.0% | 0.9%, 1.2% |
| 1 s | clean | recognizer | 327 | 0.0% | 0.6% | 99.7% | 99.7% | 1.0% | 0.0%, 0.0% |
| 1 s | clean | embed() | 327 | 0.0% | 0.4% | 100.0% | 100.0% | 1.0% | 0.0%, 0.0% |
| 2 s | clean | recognizer | 327 | 0.0% | 0.3% | 100.0% | 100.0% | 1.0% | 0.0%, 0.0% |
| 2 s | clean | embed() | 327 | 0.0% | 0.0% | 100.0% | 100.0% | 1.0% | 0.0%, 0.0% |
