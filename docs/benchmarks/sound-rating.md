# Certainty and the sound outside words

Run 2026-09-28 22:46:57Z, utter 0.0.5 from 4313af5781777702cd67b1c6e1214643dc328a21 through its C ABI (checkout 4313af5), model vosk-model-small-en-us-0.15, 12th Gen Intel(R) Core(TM) i7-12700F, Python 3.14.4, blocks of 40 ms, the Speech Commands page's grammar of 92 words.

Every partial and final carries the acoustic model's certainty over the frames decoded since the previous result, in words and outside them, and what the sound outside words is like; every word entry carries the certainty over its own span, `[[rr:TD-17]]`. This page asks whether they do what they are for, on public audio.

## Spoken words and words decoded out of a room

Each of the 11005 test clips through a recognizer of its own, and each of the dataset's 6 background recordings streamed whole with its floor at -60, -50, -40 dBFS, where nothing is said and every final word came out of the room. A word is right when it is the clip's own word.

| Words | Count | Certainty p10 | Median | p90 |
|---|---|---|---|---|
| spoken, right | 10109 | 0.034 | 0.042 | 0.053 |
| spoken, misread | 505 | 0.029 | 0.036 | 0.048 |
| room | 44 | 0.005 | 0.009 | 0.015 |

How well each figure separates right words from room words: the chance a right word scores above a room word (AUC), and the room words it rejects at the threshold that keeps 95.0% of the right words. `energy over floor` is the word's `energy_dbfs` less its final's `floor_dbfs`, the gate a host already has, `[[rr:TD-8#The gate is the host's and is relative to the floor]]`. `both` takes each at the threshold that keeps 97.5% of the right words alone.

| Figure | AUC, right words | AUC, every spoken word | Threshold | Right words kept | Room words rejected |
|---|---|---|---|---|---|
| certainty | 1.000 | 1.000 | 0.032 | 96.0% | 100.0% |
| energy over floor | 0.995 | 0.995 | 15.9 | 95.0% | 100.0% |
| both | | | 0.031 and 11.7 dB | 95.2% | 100.0% |

Room words by floor, and the share the certainty's threshold above rejects:

| Floor | Room words | Median certainty | Rejected |
|---|---|---|---|
| -60 dBFS | 16 | 0.006 | 100.0% |
| -50 dBFS | 15 | 0.010 | 100.0% |
| -40 dBFS | 13 | 0.010 | 100.0% |

| Recording | Seconds | Room words | Median certainty |
|---|---|---|---|
| doing_the_dishes at -60 dBFS | 95 | 1 | 0.012 |
| dude_miaowing at -60 dBFS | 62 | 3 | 0.015 |
| exercise_bike at -60 dBFS | 61 | 3 | 0.006 |
| pink_noise at -60 dBFS | 60 | 3 | 0.005 |
| running_tap at -60 dBFS | 61 | 3 | 0.009 |
| white_noise at -60 dBFS | 60 | 3 | 0.005 |
| doing_the_dishes at -50 dBFS | 95 | 1 | 0.013 |
| dude_miaowing at -50 dBFS | 62 | 2 | 0.013 |
| exercise_bike at -50 dBFS | 61 | 3 | 0.007 |
| pink_noise at -50 dBFS | 60 | 3 | 0.005 |
| running_tap at -50 dBFS | 61 | 3 | 0.014 |
| white_noise at -50 dBFS | 60 | 3 | 0.007 |
| doing_the_dishes at -40 dBFS | 95 | 0 | - |
| dude_miaowing at -40 dBFS | 62 | 1 | 0.011 |
| exercise_bike at -40 dBFS | 61 | 3 | 0.011 |
| pink_noise at -40 dBFS | 60 | 3 | 0.007 |
| running_tap at -40 dBFS | 61 | 3 | 0.015 |
| white_noise at -40 dBFS | 60 | 3 | 0.009 |

## Input level and grammar

1000 test clips drawn at random, each decoded with no result read until the end, so the finals' windows cover every frame; the clip's certainty is the mean over them. Each figure is against the same clip at 0 dB under the page's grammar.

| Condition | Median certainty | Median change | p10 | p90 | p90 of the size of the change |
|---|---|---|---|---|---|
| 0 dB | 0.039 | 0.0% | 0.0% | 0.0% | 0.0% |
| -10 dB | 0.036 | -6.6% | -14.1% | 0.0% | 14.2% |
| -20 dB | 0.034 | -13.9% | -26.8% | -2.8% | 26.8% |
| -30 dB | 0.030 | -22.4% | -37.5% | -9.2% | 37.5% |
| page grammar | 0.039 | 0.0% | 0.0% | 0.0% | 0.0% |
| the 35 dataset words | 0.039 | 0.0% | 0.0% | 0.0% | 0.0% |
| the 10 commands | 0.039 | 0.0% | -1.4% | 1.1% | 2.2% |

## The sound outside words

A room of pink noise with its floor at -60 dBFS: 10 s of it, then each sound in turn for 2 s with 3 s of the room after, the room running on under it. A steady sound's level is its RMS over the room's floor; an impulse's is its loudest 10 ms frame's, laid at the middle of its two seconds. The dishes clink is the loudest 20 ms of that recording. Partials are read after every block. A result's window is scored when it lies wholly inside a steady sound, holds an impulse, or lies in the room half a second clear of every sound.

The features, medians over the scored windows with evidence: `broad` the median of `band_db`; `low` the loudest of the lowest eight bands over the median of the upper 24; `sd` the median of `band_sd_db`; the rise's length and height. A window has evidence when it holds a frame in no named word and the band floors are known; a sound the decoder reads as a word has none.

| Sound | Over floor | Windows | With evidence | broad | low | sd | Rise ms | Rise dB | Example rule says |
|---|---|---|---|---|---|---|---|---|---|
| room | in the 10 dB stream | 135 | 135 | 1.9 | 2.5 | 2.3 | 240 | 1.3 | impulse 1, room 134 |
| room | in the 20 dB stream | 135 | 135 | 1.9 | 2.5 | 2.3 | 240 | 1.2 | impulse 1, room 134 |
| room | in the 30 dB stream | 135 | 135 | 1.9 | 2.5 | 2.3 | 240 | 1.2 | impulse 1, room 134 |
| white noise | 10 dB | 8 | 8 | 13.6 | -5.7 | 2.4 | 240 | 17.2 | steady broadband 8 |
| white noise | 20 dB | 8 | 8 | 23.2 | -6.8 | 2.4 | 240 | 27.1 | steady broadband 8 |
| white noise | 30 dB | 8 | 8 | 33.2 | -6.7 | 2.4 | 240 | 37.0 | steady broadband 8 |
| pink noise | 10 dB | 7 | 7 | 9.1 | 2.6 | 2.4 | 240 | 8.4 | steady broadband 7 |
| pink noise | 20 dB | 7 | 7 | 18.4 | 2.3 | 2.2 | 240 | 17.6 | steady broadband 7 |
| pink noise | 30 dB | 7 | 7 | 28.2 | 2.3 | 2.2 | 240 | 27.5 | steady broadband 7 |
| 50 Hz tone | 10 dB | 7 | 7 | 1.9 | 19.3 | 2.0 | 240 | 1.3 | low tone 7 |
| 50 Hz tone | 20 dB | 7 | 7 | 1.9 | 29.1 | 2.0 | 240 | 1.9 | low tone 7 |
| 50 Hz tone | 30 dB | 7 | 7 | 1.9 | 39.2 | 2.0 | 240 | 5.4 | low tone 7 |
| 60 Hz tone | 10 dB | 7 | 7 | 2.1 | 20.1 | 2.2 | 240 | 1.3 | low tone 7 |
| 60 Hz tone | 20 dB | 7 | 7 | 2.2 | 30.1 | 2.1 | 240 | 2.0 | low tone 7 |
| 60 Hz tone | 30 dB | 7 | 7 | 2.1 | 40.1 | 2.1 | 240 | 6.2 | low tone 7 |
| 100 Hz tone | 10 dB | 8 | 8 | 2.0 | 21.5 | 2.2 | 240 | 1.2 | low tone 8 |
| 100 Hz tone | 20 dB | 8 | 8 | 2.0 | 31.4 | 2.2 | 240 | 2.6 | low tone 8 |
| 100 Hz tone | 30 dB | 8 | 8 | 2.0 | 41.4 | 2.1 | 240 | 8.4 | low tone 8 |
| 120 Hz tone | 10 dB | 8 | 8 | 2.1 | 22.3 | 2.2 | 240 | 1.5 | low tone 8 |
| 120 Hz tone | 20 dB | 8 | 8 | 2.2 | 32.3 | 2.1 | 240 | 3.1 | low tone 8 |
| 120 Hz tone | 30 dB | 8 | 8 | 2.2 | 42.2 | 2.1 | 240 | 9.4 | low tone 8 |
| click | 10 dB | 1 | 1 | 2.9 | 3.2 | 3.7 | 10 | 15.6 | impulse 1 |
| click | 20 dB | 1 | 1 | 3.7 | 2.7 | 5.5 | 10 | 25.3 | impulse 1 |
| click | 30 dB | 1 | 1 | 4.5 | 2.7 | 8.0 | 10 | 35.3 | impulse 1 |
| 5 ms burst | 10 dB | 1 | 1 | 3.0 | 2.5 | 3.6 | 10 | 16.3 | impulse 1 |
| 5 ms burst | 20 dB | 1 | 1 | 3.9 | 2.0 | 5.0 | 10 | 26.1 | impulse 1 |
| 5 ms burst | 30 dB | 1 | 1 | 5.0 | 1.7 | 7.7 | 10 | 36.1 | impulse 1 |
| dishes clink | 10 dB | 1 | 1 | 3.0 | 1.7 | 3.1 | 20 | 17.1 | impulse 1 |
| dishes clink | 20 dB | 1 | 1 | 4.0 | 0.5 | 4.0 | 20 | 27.0 | impulse 1 |
| dishes clink | 30 dB | 1 | 1 | 5.1 | 0.2 | 6.4 | 20 | 36.9 | impulse 1 |
| running tap | 10 dB | 7 | 7 | 10.8 | -4.8 | 2.7 | 240 | 18.6 | steady broadband 7 |
| running tap | 20 dB | 7 | 7 | 20.2 | -5.2 | 2.7 | 240 | 28.5 | steady broadband 7 |
| running tap | 30 dB | 7 | 7 | 30.2 | -5.2 | 2.7 | 240 | 38.5 | steady broadband 7 |
| exercise bike | 10 dB | 8 | 7 | 13.2 | -1.5 | 2.5 | 240 | 17.8 | steady broadband 7 |
| exercise bike | 20 dB | 8 | 8 | 22.9 | -1.1 | 2.5 | 240 | 27.8 | steady broadband 8 |
| exercise bike | 30 dB | 8 | 8 | 32.8 | -1.2 | 2.5 | 240 | 37.8 | steady broadband 8 |
| dishes | 10 dB | 8 | 8 | 8.1 | 1.2 | 4.8 | 55 | 22.7 | steady broadband 3, impulse 3, room 2 |
| dishes | 20 dB | 8 | 8 | 16.0 | 1.6 | 5.8 | 40 | 32.7 | steady broadband 3, impulse 5 |
| dishes | 30 dB | 8 | 8 | 25.7 | 1.7 | 6.0 | 40 | 42.7 | steady broadband 3, impulse 5 |

The example rule, the page's and not the runtime's: an impulse when the rise lasts at most 40 ms and stands 10 dB over the floor; otherwise a low tone when `low` is at least 10 dB; otherwise steady broadband when `broad` is at least 5 dB; otherwise the room. Over every level, what it says of each kind of sound's windows with evidence:

| Laid in | steady broadband | low tone | impulse | room |
|---|---|---|---|---|
| steady broadband | 45 | 0 | 0 | 0 |
| low tone | 0 | 90 | 0 | 0 |
| impulse | 0 | 0 | 9 | 0 |
| room | 0 | 0 | 3 | 402 |

The rise's onset against the impulse's own sample, over 9 windows: median -10 ms, p10 -15, p90 -10; within 10 ms on 66.7%.

## Reading

**Certainty is what the acoustic model heard, not whether the word is right.**
Misread spoken words sit only a little below right ones, since the model heard
speech either way. What it tells apart is speech from a room: every word decoded
out of a background recording scored below the lowest tenth of the right ones.
That is 44 room words over six recordings at three floors, so the 100% it rejects
is a count on this page and not a rate to carry elsewhere.

**Energy over the floor does nearly as well on these recordings**, and taking
both does not reject more room words here, because the room words are already
all rejected. What certainty adds is a figure that needs no floor.

**A threshold on certainty depends on the input level.** Turning a clip down
lowers its certainty, by a seventh at -20 dB and a fifth at -30 dB in the median,
while the grammar hardly moves it. A host that sets a threshold sets it for the
level it runs at.

**The sound outside words keeps the kinds of sound apart** on synthetic sounds
laid over a synthetic room: steady noise raises `broad`, a hum raises `low`, an
impulse shows as a short rise, and the example rule sorts every such window. The
recorded dishes, a mix of clatter and running water, fall between the kinds, and
three room windows read as impulses; the rule is the page's, and a host writes its
own. The rise starts a frame before the impulse's own sample in the median, which
is the frame grid and not a lead.
