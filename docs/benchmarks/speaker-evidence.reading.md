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
