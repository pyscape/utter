## Reading

One word through the stock speaker model is evidence, but not much of
it, and utter makes it available on nearly every word. Over 1,980 probe
words from 205 speakers, utter's word evidence names the speaker among
two enrolled speakers on 96.2% of words and among four on 93.4%, and it
is missing on 2.0%, most of them words under 400 ms, where the quarter
second floor leaves 7.4% without. The wheel gives no vector for half of
the words: its 50-frame floor drops every word under half a second, and
14 speakers enrolled nothing at all. As a gate at 1% false accept, utter
passes 41.6% of words and the wheel 20.2%. A host that gates one word
at a time on this model will turn away more than half of an enrolled
speaker's words at that operating point.

The normalisation that looks back costs nothing measurable on single
words. The Kaldi row runs the same network with the centred mean the
model was trained with, over the word alone: 93.8% top-1 of 2, 15.3%
equal error rate and 37.4% accepted, against utter's 96.2%, 12.4% and
41.6%. It is ahead only on words of 600 ms and over at the gate, 51.0%
against 49.9%, and gives evidence on four words more in a thousand. Of the
two prices in `[[rr:TD-14#Mean normalisation looks back only]]`, the new
recognizer per word row pays the first, a mean over the clip's few
frames, and the continuous game streams pay the second, a mean carrying
the other speaker's turns. The two rows are within 1.8
points of each other on every figure in the one-word table.

The 8 dB floor margin costs a little: evidence missing on 3.5% of words
instead of 2.0%, and 39.1% accepted instead of 41.6%. The frames it
drops as too close to the floor carry some of the speaker.

TitaNet-small is far ahead of every 8 kHz row: 99.5% top-1 of 2, a 5.2%
equal error rate, and 85.7% of words accepted at 1% false accept,
including 66.7% of words under 400 ms, where utter's is 30.5%. It is the
only engine here whose single word would pass most of a speaker's words
through a strict gate. CAM++, trained on VoxCeleb interviews, is the
weakest on these words, 11.7% accepted.

As speech accumulates, utter's partial evidence at 1% false accept
rises from 25.5% at 250 ms to 51.6% at one second and 81.1% at three.
TitaNet-small is at 87.8% by half a second. The utter rows dip at the
longest lengths only because some joined streams never pooled that much
speech: at 3 s, 3.7% of streams have no evidence, and 28.6% with the
margin, which pools fewer frames of the same audio. Among the streams
with evidence, top-1 of 4 is 99.9% and 99.8%. The Kaldi row has no
vector at 250 ms because a 250 ms cut makes 23 frames, under its
25-frame minimum. The wheel has none until 750 ms for the same reason
at 50.

The evidence is there when the word is. Of 1,980 words, 1,913 carried
evidence on a partial before their final, first a median 40 ms after
the word's end (90th percentile 220 ms), counting the audio fed. That
first evidence names the speaker among two on 93.7% of words, against
96.2% on the final's.

utter's vectors are not the wheel's. Their median cosine is 0.718, while
the same network through Kaldi agrees with the wheel at 0.993. The
difference is the normalisation and the frames pooled, not the network.
Profiles and thresholds fitted on the wheel's vectors do not carry over,
`[[rr:TD-14#Consequences]]`.

The speaker model costs compute on the frames words pool, not on the
whole stream. On the game streams, clips back to back with no pause
between them, it takes utter's real-time factor from 0.0114 to 0.0191,
the median block from 0.062 to 0.081 ms and the 99th percentile from
2.74 to 3.14 ms, and it moves no result to a later block. The wheel with
its own speaker model costs about as much in total, 0.0192, but most of
it on the block that closes an utterance: a median 15.8 ms there and
42.5 ms at worst, against utter's 2.42 and 6.7. The results grow from
1.70 to 5.16 KiB a block, and parsing one in Python from 9 to 31 µs at
the median.

TitaNet-small in the crate passes 81.2% of words at 1% false accept on
the game streams, against 41.6% for the x-vector in the same
recognizer, and 62.5% of words under 400 ms against 30.5%. It stays
behind the same network on the wheel's span, 85.8% through utter's
embed() and 85.7% through sherpa-onnx: the recognizer embeds its own
word spans, and 5.8% of words carry none. embed() and
sherpa-onnx agree on every figure to within 0.3 points. With the 8 dB
floor margin set, both TitaNet rows are the same figure for figure: the
margin does not touch a span TitaNet embeds,
`[[rr:TD-15#The floor margin: decided by measurement]]`.

Multi-threaded, the default, it embeds the open word as it grows on a
thread of its own, `[[rr:TD-16#The open word: decided by measurement]]`, and its
evidence comes a median 80 ms after the word's end and 260 ms at the
90th percentile, 40 ms behind the x-vector's 40 and 220 ms at both. By
300 ms after the end, 78.7% of words have passed the gate, against the
x-vector's 41.1%. The recognizer's thread runs at a real-time factor of
0.0149 and the TitaNet thread at 0.0078 beside it; the median block,
waits for that thread included, is 0.139 ms and the 99th percentile
3.18 ms, against the x-vector's 0.081 and 3.14.

Single-threaded, its evidence comes after the word, not during it: a
median 430 ms after the word's end and 546 ms at the 90th percentile.
By 600 ms after the end, 92.3% of words carry it and 79.4% have passed
the gate. It passes 80.7% of words in all, with 6.3% carrying none, for
a real-time factor of 0.0158 against the x-vector's 0.0191 and a 99th
percentile block of 2.81 ms against 3.14.
