# Speaker evidence on short commands

Run 2026-09-24 06:37:28Z, utter 0.0.5 from 765482d7b70a6d3f5c4071e56e280b7d5df7b598 through its C ABI (checkout 765482d), vosk 0.3.45, sherpa-onnx 1.13.4, numpy 2.5.3, model vosk-model-small-en-us-0.15, speaker models vosk-model-spk-0.4, nemo_en_titanet_small.onnx and wespeaker_en_voxceleb_CAM++.onnx, 12th Gen Intel(R) Core(TM) i7-12700F, Python 3.14.4, blocks of 40 ms.

A host that gates commands per word needs to know, word by word, whose voice it is, and whether the evidence is there when the word is. This page measures that on the test split of Speech Commands: 206 speakers with at least 20 clips (4 more had no profile under some engine but the wheel), each enrolled from 10 clips, with 1990 probe words scored against every profile (probe clips whose final had no word are left out: 70). The wheel's half-second floor left 15 of the speakers with no profile at all: their words count as no evidence for the wheel, and they are no rival to anyone else's.

## Engines

| engine | what runs | audio | speech it is given |
|---|---|---|---|
| vosk | the vosk wheel with vosk-model-spk-0.4 set, one clip per recognizer | 8 kHz | the frames its decode puts on speech phones, at least half a second |
| utter | utter's speaker evidence through its C ABI, continuous streams | 8 kHz | each word entry's own span, at least a quarter second |
| utter, 8 dB floor margin | the same with a floor margin of 8 dB set, enrollment included | 8 kHz | the span's frames more than the margin over the floor |
| utter, new recognizer per word | the same, each probe clip through a recognizer built for it, as a host that builds one at every grammar change does; enrolled as the first utter row | 8 kHz | the word's span, its normalisation seeing only the clip |
| x-vector (Kaldi) | the same network through Kaldi's binaries, centred mean over the span | 8 kHz | the wheel's word span, at least a quarter second |
| TitaNet-small | NVIDIA NeMo TitaNet small through sherpa-onnx | 16 kHz | the wheel's word span |
| CAM++ | WeSpeaker CAM++ (VoxCeleb) through sherpa-onnx, less the enrollment set's mean embedding, as WeSpeaker scores it | 16 kHz | the wheel's word span |

*Top-1 of K* is how often the probe's own profile scores highest among K enrolled speakers drawn at random (20 draws per probe, the same draws for every engine). *Accepted at 1% false accept* is the share of probes whose own-profile score clears the threshold that lets 1% of other-profile scores through. A probe with no evidence is neither identified nor accepted. The equal error rate is over probes with evidence only.

## One word

| engine | probes | no evidence | top-1 of 2 | top-1 of 4 | equal error rate | accepted at 1% false accept |
|---|---|---|---|---|---|---|
| vosk | 1990 | 49.5% | 48.4% | 45.5% | 14.3% | 20.1% |
| utter | 1990 | 2.6% | 95.4% | 92.3% | 13.1% | 40.1% |
| utter, 8 dB floor margin | 1990 | 4.2% | 93.2% | 89.9% | 14.6% | 37.0% |
| utter, new recognizer per word | 1990 | 1.7% | 96.0% | 92.5% | 14.4% | 38.4% |
| x-vector (Kaldi) | 1990 | 1.6% | 93.8% | 87.5% | 15.3% | 37.5% |
| TitaNet-small | 1990 | 0.0% | 99.5% | 98.7% | 5.1% | 85.7% |
| CAM++ | 1990 | 0.0% | 78.9% | 59.7% | 28.8% | 11.6% |

## By word length

The length is the wheel's word span. Each cell is accepted at 1% false accept, then top-1 of 2.

| engine | under 400 ms (407) | 400 to 600 ms (1129) | 600 ms and over (454) |
|---|---|---|---|
| vosk | 0.0%, 0.2% | 17.0%, 46.7% | 45.6%, 96.0% |
| utter | 29.2%, 87.7% | 41.6%, 97.0% | 46.0%, 98.3% |
| utter, 8 dB floor margin | 26.3%, 81.5% | 38.3%, 95.7% | 43.4%, 97.8% |
| utter, new recognizer per word | 31.2%, 89.3% | 39.2%, 97.7% | 43.0%, 97.9% |
| x-vector (Kaldi) | 19.9%, 84.9% | 38.4%, 95.4% | 51.1%, 97.6% |
| TitaNet-small | 67.1%, 98.5% | 89.5%, 99.8% | 93.2%, 99.8% |
| CAM++ | 8.8%, 80.2% | 12.5%, 78.3% | 11.9%, 79.4% |

## As speech accumulates

Each speaker's probe words joined into 3 streams in shuffled orders and cut at each length of speech. The offline engines embed the cut whole; utter's is its own partial's evidence at the first partial that has pooled that much speech. Top-1 of 4, then accepted at 1% false accept at that length's own threshold.

| engine | 250 ms | 500 ms | 750 ms | 1000 ms | 1500 ms | 2000 ms | 3000 ms |
|---|---|---|---|---|---|---|---|
| vosk | 0.0%, 0.0% | 0.0%, 0.0% | 53.7%, 21.1% | 78.2%, 34.8% | 86.0%, 43.2% | 88.0%, 50.3% | 90.9%, 55.0% |
| utter | 86.6%, 26.4% | 93.6%, 36.4% | 96.4%, 43.6% | 97.9%, 52.5% | 98.9%, 65.7% | 98.6%, 74.1% | 95.8%, 81.4% |
| utter, 8 dB floor margin | 85.6%, 26.4% | 93.8%, 39.5% | 97.2%, 49.9% | 97.7%, 59.2% | 95.9%, 69.5% | 91.5%, 73.9% | 70.4%, 63.1% |
| x-vector (Kaldi) | 0.0%, 0.0% | 87.9%, 35.4% | 93.9%, 43.4% | 96.1%, 51.1% | 97.8%, 61.1% | 98.3%, 69.6% | 99.2%, 74.6% |
| TitaNet-small | 90.7%, 54.0% | 99.4%, 87.4% | 99.8%, 95.1% | 99.9%, 97.4% | 100.0%, 98.9% | 100.0%, 99.7% | 100.0%, 99.8% |
| CAM++ | 44.2%, 6.5% | 58.7%, 13.1% | 45.1%, 4.7% | 28.7%, 1.3% | 26.2%, 1.0% | 25.1%, 1.0% | 26.3%, 1.4% |

## On utter's partials

Of 1990 probe words, 1923 carried evidence on a partial before their final. It first appeared a median 40 ms after the word's end (90th percentile 220 ms), counting the audio fed, with the network's 70 ms of right context and the block included. Top-1 of 2 on that first evidence: 93.2%; on the final's: 95.4%.

## Agreement

utter's word vector against the wheel's clip vector: median cosine 0.721 over 1019 probes both score. utter normalises by a mean that looks back over the stream and pools the word's own span; the wheel normalises by a centred mean over the frames it keeps.
The Kaldi row against the wheel: median cosine 0.993 over 1023.

## Latency and compute

The game streams again, 2014 s of audio, through each engine with and without its speaker model, the four in alternating order, blocks of 40 ms with words on partials. A block's compute runs from the call that takes its audio to the return of the partial or final read after it; a closing block is one that returned a final. Parsing is the host's json.loads of that result in Python.

| engine | real-time factor | per block, ms p50 / p95 / p99 / max | closing blocks, ms p50 / max | result text per block | parsing, ms p50 / p99 |
|---|---|---|---|---|---|
| vosk | 0.0129 | 0.053 / 2.92 / 3.45 / 6.8 | 3.04 / 6.8 | 0.11 KiB | 0.001 / 0.004 |
| vosk, speaker model set | 0.0190 | 0.080 / 3.01 / 14.77 / 42.9 | 15.47 / 42.9 | 0.13 KiB | 0.001 / 0.012 |
| utter | 0.0117 | 0.060 / 2.56 / 2.86 / 8.7 | 2.31 / 8.7 | 0.54 KiB | 0.003 / 0.011 |
| utter, speaker model set | 0.0197 | 0.078 / 2.94 / 3.26 / 7.0 | 2.55 / 7.0 | 3.99 KiB | 0.021 / 0.091 |

With the speaker model set, none of utter's 50478 results differ from its results without it once the four speaker keys are removed: the evidence moves no word, partial or final to a later block.

Compute per second of speech embedded by the offline engines: TitaNet-small 10.0 ms, CAM++ 12.8 ms, x-vector (Kaldi) 5.9 ms (the Kaldi figure is a batch through three processes).

## Caveats

Speech Commands speakers recorded on their own devices, so a profile carries the microphone and the room as well as the voice, which flatters every engine against a table where everyone speaks into one microphone. A same-microphone set such as VCTK (CC BY 4.0) is the harder check and is not run here. Enrollment here is ten single words; a host that enrolls from minutes of speech has more to average.

## Reading

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
