# Speaker evidence on short commands

Run 2026-09-30 04:48:11Z, utter 0.0.5 from 2fd0366ba052f516037344ecf4f8e3a04d480690-dirty through its C ABI (checkout 2fd0366), vosk 0.3.45, sherpa-onnx 1.13.4, numpy 2.5.3, model vosk-model-small-en-us-0.15, speaker models vosk-model-spk-0.4, nemo_en_titanet_small.onnx and wespeaker_en_voxceleb_CAM++.onnx, 12th Gen Intel(R) Core(TM) i7-12700F, Python 3.14.4, blocks of 40 ms.

A host that gates commands per word needs to know, word by word, whose voice it is, and whether the evidence is there when the word is. This page measures that on the test split of Speech Commands: 205 speakers with at least 20 clips (5 more had no profile under some engine but the wheel), each enrolled from 10 clips, with 1980 probe words scored against every profile (probe clips whose final had no word are left out: 70). The wheel's half-second floor left 14 of the speakers with no profile at all: their words count as no evidence for the wheel, and they are no rival to anyone else's.

## Engines

| engine | what runs | audio | speech it is given |
|---|---|---|---|
| utter | utter's speaker evidence through its C ABI, continuous streams | 8 kHz | each word entry's own span, at least a quarter second |
| utter, 8 dB floor margin | the same with a floor margin of 8 dB set, enrollment included | 8 kHz | the span's frames more than the margin over the floor |
| utter, new recognizer per word | the same, each probe clip through a recognizer built for it, as a host that builds one at every grammar change does; enrolled as the first utter row | 8 kHz | the word's span, its normalisation seeing only the clip |
| utter, TitaNet-small, single-threaded | utter's TitaNet-small through the same recognizer, set single-threaded: each closed word embedded in slices between decoder advances | 16 kHz | each word entry's own span, 25 to 120 frames |
| utter, TitaNet-small, multi-threaded | the same, multi-threaded, the default: each closed word, and at each advance the open word over its span so far, embedded on the recognizer's TitaNet thread and published 40 ms of audio after it was queued | 16 kHz | each word entry's own span, 25 to 120 frames |
| utter, TitaNet-small, single-threaded, 8 dB floor margin, span as reported | the single-threaded row with a floor margin of 8 dB set, enrollment included | 16 kHz | the span as reported |
| utter, TitaNet-small, multi-threaded, 8 dB floor margin, span as reported | the multi-threaded row with the same margin | 16 kHz | the span as reported |
| utter's embed(), TitaNet-small | utter's stateless embed() through its C ABI, no recognizer | 16 kHz | the wheel's word span |
| vosk | the vosk wheel with vosk-model-spk-0.4 set, one clip per recognizer | 8 kHz | the frames its decode puts on speech phones, at least half a second |
| x-vector (Kaldi) | the same network through Kaldi's binaries, centred mean over the span | 8 kHz | the wheel's word span, at least a quarter second |
| TitaNet-small | NVIDIA NeMo TitaNet small through sherpa-onnx | 16 kHz | the wheel's word span |
| CAM++ | WeSpeaker CAM++ (VoxCeleb) through sherpa-onnx, less the enrollment set's mean embedding, as WeSpeaker scores it | 16 kHz | the wheel's word span |

*Top-1 of K* is how often the probe's own profile scores highest among K enrolled speakers drawn at random (20 draws per probe, the same draws for every engine). *Accepted at 1% false accept* is the share of probes whose own-profile score clears the threshold that lets 1% of other-profile scores through. A probe with no evidence is neither identified nor accepted. The equal error rate is over probes with evidence only.

## One word

| engine | probes | no evidence | top-1 of 2 | top-1 of 4 | equal error rate | accepted at 1% false accept |
|---|---|---|---|---|---|---|
| utter | 1980 | 2.0% | 96.2% | 93.4% | 12.4% | 41.6% |
| utter, 8 dB floor margin | 1980 | 3.5% | 93.8% | 90.3% | 14.0% | 39.1% |
| utter, new recognizer per word | 1980 | 1.7% | 96.4% | 93.5% | 13.5% | 39.8% |
| utter, TitaNet-small, single-threaded | 1980 | 6.3% | 93.3% | 92.8% | 4.7% | 80.7% |
| utter, TitaNet-small, multi-threaded | 1980 | 5.8% | 93.8% | 93.3% | 4.7% | 81.2% |
| utter, TitaNet-small, single-threaded, 8 dB floor margin, span as reported | 1980 | 6.3% | 93.3% | 92.8% | 4.7% | 80.7% |
| utter, TitaNet-small, multi-threaded, 8 dB floor margin, span as reported | 1980 | 5.8% | 93.8% | 93.3% | 4.7% | 81.2% |
| utter's embed(), TitaNet-small | 1980 | 0.0% | 99.5% | 98.8% | 5.2% | 85.8% |
| vosk | 1980 | 49.3% | 48.6% | 45.7% | 14.3% | 20.2% |
| x-vector (Kaldi) | 1980 | 1.6% | 93.8% | 87.2% | 15.3% | 37.4% |
| TitaNet-small | 1980 | 0.0% | 99.5% | 98.8% | 5.2% | 85.7% |
| CAM++ | 1980 | 0.0% | 78.6% | 59.9% | 28.8% | 11.7% |

## By word length

The length is the wheel's word span. Each cell is accepted at 1% false accept, then top-1 of 2.

| engine | under 400 ms (403) | 400 to 600 ms (1124) | 600 ms and over (453) |
|---|---|---|---|
| utter | 30.5%, 90.0% | 42.3%, 97.7% | 49.9%, 98.0% |
| utter, 8 dB floor margin | 27.0%, 84.3% | 40.0%, 95.8% | 47.7%, 97.2% |
| utter, new recognizer per word | 33.0%, 89.5% | 40.6%, 98.2% | 44.2%, 98.0% |
| utter, TitaNet-small, single-threaded | 62.3%, 87.9% | 83.8%, 94.7% | 89.2%, 94.7% |
| utter, TitaNet-small, multi-threaded | 62.5%, 88.4% | 84.5%, 95.3% | 89.4%, 95.0% |
| utter, TitaNet-small, single-threaded, 8 dB floor margin, span as reported | 62.3%, 87.9% | 83.8%, 94.7% | 89.2%, 94.7% |
| utter, TitaNet-small, multi-threaded, 8 dB floor margin, span as reported | 62.5%, 88.4% | 84.5%, 95.3% | 89.4%, 95.0% |
| utter's embed(), TitaNet-small | 67.0%, 98.4% | 89.5%, 99.7% | 93.2%, 99.8% |
| vosk | 0.0%, 0.2% | 17.1%, 46.9% | 45.7%, 95.8% |
| x-vector (Kaldi) | 19.9%, 84.7% | 38.3%, 95.5% | 51.0%, 97.6% |
| TitaNet-small | 66.7%, 98.3% | 89.5%, 99.7% | 93.2%, 99.8% |
| CAM++ | 8.9%, 79.4% | 12.5%, 77.9% | 11.9%, 79.6% |

## As speech accumulates

Each speaker's probe words joined into 3 streams in shuffled orders and cut at each length of speech. The offline engines embed the cut whole; utter's is its own partial's evidence at the first partial that has pooled that much speech. Top-1 of 4, then accepted at 1% false accept at that length's own threshold.

| engine | 250 ms | 500 ms | 750 ms | 1000 ms | 1500 ms | 2000 ms | 3000 ms |
|---|---|---|---|---|---|---|---|
| utter | 88.9%, 25.5% | 94.1%, 36.1% | 96.5%, 46.6% | 98.1%, 51.6% | 98.8%, 60.9% | 98.7%, 70.3% | 96.1%, 81.1% |
| utter, 8 dB floor margin | 86.9%, 28.0% | 93.5%, 40.0% | 96.5%, 46.4% | 97.2%, 55.2% | 96.1%, 64.2% | 92.3%, 69.5% | 71.3%, 61.4% |
| utter's embed(), TitaNet-small | 90.0%, 54.1% | 99.2%, 87.8% | 99.9%, 95.9% | 100.0%, 97.7% | 100.0%, 99.3% | 100.0%, 99.8% | 100.0%, 99.8% |
| vosk | 0.0%, 0.0% | 0.0%, 0.0% | 52.8%, 17.5% | 78.3%, 34.8% | 85.7%, 43.4% | 88.0%, 53.2% | 91.2%, 58.0% |
| x-vector (Kaldi) | 0.0%, 0.0% | 87.8%, 35.6% | 93.5%, 42.0% | 95.5%, 52.4% | 97.5%, 61.2% | 98.4%, 69.3% | 99.4%, 75.9% |
| TitaNet-small | 90.0%, 54.3% | 99.2%, 87.8% | 99.9%, 95.8% | 100.0%, 97.7% | 100.0%, 99.3% | 100.0%, 99.8% | 100.0%, 99.8% |
| CAM++ | 44.2%, 5.7% | 58.3%, 12.0% | 42.7%, 5.1% | 29.3%, 1.6% | 25.7%, 1.0% | 25.7%, 1.0% | 26.8%, 1.4% |

## After a word's end

When a probe word's evidence shows on the game streams, counted in audio fed after the final's end of the word, on any partial or final. Each cell is the share of probe words whose evidence has shown by then, then the share whose final evidence clears the row's gate at 1% false accept and has shown by then.

| engine | median, 90th percentile | 0 ms | 100 ms | 200 ms | 300 ms | 400 ms | 500 ms | 600 ms | 800 ms | 1000 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| utter | 40, 220 ms | 35.8%, 16.6% | 68.0%, 31.5% | 87.2%, 38.2% | 95.7%, 41.1% | 98.3%, 41.6% | 98.3%, 41.6% | 98.3%, 41.6% | 98.3%, 41.6% | 98.3%, 41.6% |
| utter, TitaNet-small, single-threaded | 430, 546 ms | 0.0%, 0.0% | 0.0%, 0.0% | 0.1%, 0.1% | 2.1%, 1.6% | 35.5%, 30.2% | 72.8%, 62.1% | 92.3%, 79.4% | 93.8%, 80.6% | 93.8%, 80.7% |
| utter, TitaNet-small, multi-threaded | 80, 260 ms | 26.5%, 25.0% | 50.8%, 47.0% | 79.9%, 70.4% | 91.7%, 78.7% | 96.1%, 81.2% | 96.5%, 81.2% | 96.5%, 81.2% | 96.5%, 81.2% | 96.5%, 81.2% |
| utter, TitaNet-small, single-threaded, 8 dB floor margin, span as reported | 430, 546 ms | 0.0%, 0.0% | 0.0%, 0.0% | 0.1%, 0.1% | 2.1%, 1.6% | 35.5%, 30.2% | 72.8%, 62.1% | 92.3%, 79.4% | 93.8%, 80.6% | 93.8%, 80.7% |
| utter, TitaNet-small, multi-threaded, 8 dB floor margin, span as reported | 80, 260 ms | 26.5%, 25.0% | 50.8%, 47.0% | 79.9%, 70.4% | 91.7%, 78.7% | 96.1%, 81.2% | 96.5%, 81.2% | 96.5%, 81.2% | 96.5%, 81.2% | 96.5%, 81.2% |

## On utter's partials

Of 1980 probe words, 1913 carried evidence on a partial before their final. It first appeared a median 40 ms after the word's end (90th percentile 220 ms), counting the audio fed, with the network's 70 ms of right context and the block included. Top-1 of 2 on that first evidence: 93.7%; on the final's: 96.2%.

## Agreement

utter's word vector against the wheel's clip vector: median cosine 0.718 over 1019 probes both score. utter normalises by a mean that looks back over the stream and pools the word's own span; the wheel normalises by a centred mean over the frames it keeps.
The Kaldi row against the wheel: median cosine 0.993 over 1022.

## Latency and compute

The game streams again, 2004 s of audio, through each engine and setup in rotating order from one stream to the next, blocks of 40 ms with words on partials. A block's compute runs from the call that takes its audio to the return of the partial or final read after it; a closing block is one that returned a final. Parsing is the host's json.loads of that result in Python. A multi-threaded TitaNet row is fed in real time, each block when its audio would have arrived, and its block times are the recognizer's thread, waits for the TitaNet thread included; the TitaNet thread's own compute, its on-CPU time over the audio, is the next column. Every other row is fed as fast as it decodes and runs on the caller's thread alone.

| engine | recognizer's thread, real-time factor | per block, ms p50 / p95 / p99 / max | closing blocks, ms p50 / max | TitaNet thread, real-time factor | result text per block | parsing, ms p50 / p99 |
|---|---|---|---|---|---|---|
| utter | 0.0114 | 0.062 / 2.49 / 2.74 / 5.0 | 2.20 / 5.0 |  | 1.70 KiB | 0.009 / 0.055 |
| utter, x-vector set | 0.0191 | 0.081 / 2.85 / 3.14 / 6.7 | 2.42 / 6.7 |  | 5.16 KiB | 0.031 / 0.140 |
| utter, TitaNet-small, single-threaded | 0.0158 | 0.066 / 2.59 / 2.81 / 5.0 | 2.37 / 5.0 |  | 4.49 KiB | 0.022 / 0.162 |
| utter, TitaNet-small, multi-threaded | 0.0149 | 0.139 / 2.94 / 3.18 / 7.0 | 2.69 / 5.4 | 0.0078 | 5.10 KiB | 0.025 / 0.168 |
| utter, 3 readings | 0.0117 | 0.067 / 2.53 / 2.79 / 5.1 | 2.29 / 5.1 |  | 3.90 KiB | 0.019 / 0.092 |
| utter, 3 readings, x-vector set | 0.0194 | 0.092 / 2.93 / 3.27 / 6.7 | 2.55 / 6.7 |  | 14.56 KiB | 0.069 / 0.419 |
| utter, TitaNet-small, single-threaded, 3 readings | 0.0162 | 0.074 / 2.65 / 2.89 / 5.2 | 2.46 / 5.2 |  | 14.81 KiB | 0.070 / 0.521 |
| utter, TitaNet-small, multi-threaded, 3 readings | 0.0153 | 0.149 / 3.01 / 3.26 / 5.7 | 2.78 / 5.7 | 0.0078 | 16.69 KiB | 0.072 / 0.554 |
| utter, 8 dB floor margin | 0.0114 | 0.063 / 2.49 / 2.73 / 4.9 | 2.20 / 4.9 |  | 1.70 KiB | 0.010 / 0.054 |
| utter, 8 dB floor margin, x-vector set | 0.0197 | 0.106 / 2.88 / 3.21 / 6.6 | 2.46 / 6.6 |  | 5.10 KiB | 0.031 / 0.140 |
| utter, TitaNet-small, single-threaded, 8 dB floor margin, span as reported | 0.0158 | 0.066 / 2.59 / 2.81 / 5.2 | 2.36 / 5.2 |  | 4.49 KiB | 0.022 / 0.163 |
| utter, TitaNet-small, multi-threaded, 8 dB floor margin, span as reported | 0.0148 | 0.137 / 2.95 / 3.20 / 6.0 | 2.69 / 6.0 | 0.0078 | 5.10 KiB | 0.025 / 0.171 |
| vosk | 0.0130 | 0.055 / 2.93 / 3.40 / 8.6 | 3.05 / 6.9 |  | 0.12 KiB | 0.001 / 0.005 |
| vosk, speaker model set | 0.0192 | 0.082 / 3.03 / 15.25 / 42.5 | 15.80 / 42.5 |  | 0.13 KiB | 0.001 / 0.012 |

With a speaker model set, none of utter's 452052 results differ from the same setup's results without one once the four speaker keys are removed: the evidence moves no word, partial or final to a later block.

Compute per second of speech embedded by the offline engines: TitaNet-small 9.9 ms, CAM++ 12.7 ms, utter's embed(), TitaNet-small 6.9 ms, x-vector (Kaldi) 5.9 ms (the Kaldi figure is a batch through three processes). The utter's embed(), TitaNet-small row and every in-crate TitaNet row run titanet-small as scripts/titanet_convert.py writes it.

## Caveats

Speech Commands speakers recorded on their own devices, so a profile carries the microphone and the room as well as the voice, which flatters every engine against a table where everyone speaks into one microphone. A same-microphone set such as VCTK (CC BY 4.0) is the harder check and is not run here. Enrollment here is ten single words; a host that enrolls from minutes of speech has more to average.

## Reading

The figures on this page predate a change to how the utter rows enrol,
and stand until the page is rerun. They were measured with one vector
per final: a speaker's ten enrolment clips run together into a median
of four finals, so each profile rested on about four vectors. The
script now enrols every utter row with one vector per spoken word that
carries evidence, as the other rows enrol one per clip. Replaying this
run's draws with that enrolment, TitaNet-small in the crate passes
81.4% at 1% false accept instead of 73.9%, with or without the floor
margin; the x-vector 41.4% instead of 40.1%, and 37.4% instead of
37.0% with the margin.

One word through the stock speaker model is evidence, but not much of
it, and utter makes it available on nearly every word. Over 1,990 probe
words from 206 speakers, utter's word evidence names the speaker among
two enrolled speakers on 95.4% of words and among four on 92.3%, and it
is missing on 2.6%, most of them words under 400 ms, where the quarter
second floor leaves 9.8% without. The wheel gives no vector for half of
the words: its 50-frame floor drops every word under half a second, and
15 speakers enrolled nothing at all. As a gate at 1% false accept, utter
passes 40.1% of words and the wheel 20.1%. A host that gates one word
at a time on this model will turn away more than half of an enrolled
speaker's words at that operating point.

The normalisation that looks back costs nothing measurable on single
words. The Kaldi row runs the same network with the centred mean the
model was trained with, over the word alone: 93.8% top-1 of 2, 15.3%
equal error rate and 37.5% accepted, against utter's 95.4%, 13.1% and
40.1%. It is ahead only on words of 600 ms and over at the gate, 51.1%
against 46.0%, and gives evidence on one word more in a hundred. Of the
two prices in `[[rr:TD-14#Mean normalisation looks back only]]`, the new
recognizer per word row pays the first, a mean over the clip's few
frames, and the continuous game streams pay the second, a mean carrying
the other speaker's turns. The two rows are within 1.7
points of each other on every figure in the one-word table.

The 8 dB floor margin costs a little: evidence missing on 4.2% of words
instead of 2.6%, and 37.0% accepted instead of 40.1%. The frames it
drops as too close to the floor carry some of the speaker.

TitaNet-small is far ahead of every 8 kHz row: 99.5% top-1 of 2, a 5.1%
equal error rate, and 85.7% of words accepted at 1% false accept,
including 67.1% of words under 400 ms, where utter's is 29.2%. It is the
only engine here whose single word would pass most of a speaker's words
through a strict gate. CAM++, trained on VoxCeleb interviews, is the
weakest on these words, 11.6% accepted.

As speech accumulates, utter's partial evidence at 1% false accept
rises from 26.4% at 250 ms to 52.5% at one second and 81.4% at three.
TitaNet-small is at 87.4% by half a second. The utter rows dip at the
longest lengths only because some joined streams never pooled that much
speech: at 3 s, 4.1% of streams have no evidence, and 29.4% with the
margin, which pools fewer frames of the same audio. Among the streams
with evidence, top-1 of 4 is 99.8% on both rows. The Kaldi row has no
vector at 250 ms because a 250 ms cut makes 23 frames, under its
25-frame minimum. The wheel has none until 750 ms for the same reason
at 50.

The evidence is there when the word is. Of 1,990 words, 1,923 carried
evidence on a partial before their final, first a median 40 ms after
the word's end (90th percentile 220 ms), counting the audio fed. That
first evidence names the speaker among two on 93.2% of words, against
95.4% on the final's.

utter's vectors are not the wheel's. Their median cosine is 0.721, while
the same network through Kaldi agrees with the wheel at 0.993. The
difference is the normalisation and the frames pooled, not the network.
Profiles and thresholds fitted on the wheel's vectors do not carry over,
`[[rr:TD-14#Consequences]]`.

The speaker model costs compute on the frames words pool, not on the
whole stream. On the game streams, clips back to back with no pause
between them, it takes utter's real-time factor from 0.0117 to 0.0197,
the median block from 0.060 to 0.078 ms and the 99th percentile from
2.86 to 3.26 ms, and it moves no result to a later block. The wheel with
its own speaker model costs about as much in total, 0.0190, but most of
it on the block that closes an utterance: a median 15.5 ms there and
42.9 ms at worst, against utter's 2.55 and 7.0. The results grow from
0.54 to 3.99 KiB a block, and parsing one in Python from 3 to 21 µs at
the median.

TitaNet-small in the crate passes 73.9% of words at 1% false accept on
the game streams, against 40.1% for the x-vector in the same
recognizer, and 55.3% of words under 400 ms against 29.2%. It stays
behind the same network on the wheel's span, 85.8% through utter's
embed() and 85.7% through sherpa-onnx: the recognizer embeds its own
word spans, and 6.4% of words carry none. embed() and
sherpa-onnx agree on every figure to a tenth of a point.

Its evidence comes after the word, not during it: a median 430 ms after
the word's end and 540 ms at the 90th percentile, against the
x-vector's 40 and 220 ms. By 600 ms after the end, 92.1% of words carry
it and 72.8% have passed the gate. It costs a real-time factor of
0.0164 against the x-vector's 0.0196, and a 99th percentile block of
2.99 ms against 3.28.

Of TitaNet's two rules for a floor margin, embedding the span as
reported passes 73.9% and trimming its floor-level edges 68.8%, behind
in every length bin. Of its two rules for readings, queueing their
words as jobs of their own gives evidence to 21.3% of the spans
readings and alternatives put forward and passes 15.1% of them, for a
real-time factor of 0.0201 against 0.0168 with three readings, and
changes nothing on the best path. `[[rr:TD-15#Decision outcome]]` keeps
the span as reported and spans already embedded, and removes the other
two rules with their switches. The script no longer produces the
trimmed rows, the readings table or the own-jobs latency row, and it
names the remaining readings row "utter, TitaNet-small, 3 readings";
the figures here stand as this run recorded them. Without the trimmed
engine, its next run keeps the speakers with a profile under the rest.
