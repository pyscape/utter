# Ending speech sooner

The `endpoint` key on a final names what closed it; its values are in
the [results reference](../reference/results.md#the-endpoint-value).

Kaldi's endpoint rules wait for the silence after a word to reach their
own bounds, half a second at the earliest, so a final lands a median
870 ms after the word ends. utter keeps those rules and lets you add
one bound of your own, off by default:

```python
rec.SetEndpointBound(300, 8)   # ms of trailing silence; veto margin in nats
```

```rust
rec.set_endpoint_bound(Some(300.0), Some(8.0));
```

The first number ends the segment once the trailing silence reaches
that many milliseconds. The second is a veto: no final while a reading
that extends the partial by another word is within that many nats of
the leader. The veto exists because the silence clock cannot see a word
beginning, the next word is not on the best path while the pause before
it still counts, but the beam already holds it as a rival.

What it buys and costs, measured on the recognizer's own finals with
the bound at 300 ms:

- Every finish arrives about 200 ms sooner: median 866 ms to 660 ms on
  multi-word streams, 870 ms to 660 ms on single-word clips.
- The same finishes are called, 732 of 750, and no word is lost or split
  on 800 single-word clips.
- The cost is pauses inside a phrase being taken for its end. This does
  not drop words: the words so far arrive in one final and the next word
  in the next, so "alpha seven" can reach you as two finals. On streams
  built with pauses of 100 to 800 ms, the stock rules end 65% of pauses
  and the bound ends 83%, the difference concentrated on pauses shorter
  than the half second the stock rule waits.

If your application acts on single words, the bound costs nothing. If
it needs whole phrases, either join consecutive finals across a short
gap or leave the bound unset, which is byte-identical to stock Vosk.
On the first consumer's recordings, an 8 nat veto let bounds as low as
10 ms run without losing commands, where the same bounds without the
veto did; measure on your own audio before going below 300.

In C, `utter_recognizer_set_endpoint_bound(rec, 300.0f, 8.0f)`; a
bound of zero or less removes it, and a veto of zero or less turns the
veto off. In Rust, `None` for either argument does the same.

## Closing a quiet room

On a quiet room the model reads the hiss as a word's first phone within
a few seconds. The partial then says `[speech]`, no silence rule sees
trailing silence, and the bound's veto is held by grammar words a few
nats behind, so nothing closes the stretch until the 20 s cap forces
the cheapest word onto it, with `stable_ms` of zero. A margin over the
floor lets the bound read that path for what it is:

```python
rec.SetEndpointBound(300, 8)
rec.SetEndpointFloorMargin(8)   # dB over floor_dbfs
```

With the margin set, a trailing `[speech]` entry whose energy is within
it counts as trailing silence, a rival word within it cannot veto, and
the final the bound reaches that way is the path as it stood: an empty
word list, the text `[sil]` or `[speech]`, and the endpoint `floor`.
Neither span is read until it is at least as long as the bound, so the
first frames of a quiet word after a pause, which sit near the floor
on any microphone, are not taken for hiss.
On the public background recordings at a -50 dBFS floor that closes a
wordless stretch at a median 0.72 s instead of 20, and the forced word
is gone. The cost is a word spoken so quietly that its first phone sits
within the margin for the length of the bound, which is cut where the
hiss is; the [states page](../benchmarks/partial-states.md) counts the
words lost on its streams. The margin is yours, as the gate on a final
word's energy is; 8 dB is the operating point the silence record,
TD-8, measured for that gate. In C,
`utter_recognizer_set_endpoint_floor_margin(rec, 8.0f)`, zero or less
removes it; in Rust, `set_endpoint_floor_margin(Some(8.0))`.
