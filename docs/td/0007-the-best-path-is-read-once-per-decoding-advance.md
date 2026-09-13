# TD-7: The best path is read once per decoding advance

- Tags: decoder, streaming, performance, algorithm

## Context

Several readers want the decoder's best path without final costs on
every `accept` step: the silence weighting of
`[[rr:TD-2#Front end: the i-vector branch]]` labels decoded frames from
it, the word list behind the hold time of
`[[rr:TD-2#The decoder: partial alternatives]]` reads its word sequence,
`[[rr:TD-2#The decoder: endpointing]]` measures trailing silence on it,
and `[[rr:TD-2#The decoder: partials]]` reads the partial off it again.

A traceback is not a lookup. It walks the best token's whole record
chain, one record per word emitted and per phone change since the
utterance began, and rebuilds the phone segmentation from it.
`[[rr:TD-2#The network]]` computes a chunk of 24 input frames at a time,
so at 40 ms a block only every sixth block decodes a frame. On the other
five, each reader walked the same chain to the same answer.

## Considered options

- **Hold the path against the decoded frame count.** Taken.
- **Let each reader keep its own last answer.** The same walks once
  each, and the invariant that licenses them stated four times.
- **Trace incrementally over the new frames.** Pruning can move the best
  token, so one decoded frame can replace the whole path. There is
  nothing to extend.
- **Skip the readers on a block that decodes nothing.** Endpointing's
  answer is a function of the decoded frames alone and could be cached
  whole, but the silence weighting's delta weights advance with the
  frames *ready*, which every block moves. One path serves both halves.
- **Hand out a borrowed path.** Removes the allocation, not the walk,
  and the walk is the cost.

## Decision outcome

The recognizer holds the best path without final costs together with the
decoded frame count it was read at, and every reader takes it from
there. Only advancing a frame changes the token set a traceback walks,
and advancing a frame always moves the count, so the count is the key: a
reader that depends on anything the count does not determine is a change
to the key, not a new reader. Restarting the decoder returns the count
to zero, so the memo is dropped where that happens rather than trusted
to the count alone.

The silence weighting's frame labels and the word list behind the hold
time are functions of that path. They carry the same count and are left
as they are while it does not move.

A final reads the path *with* final costs, which is a different
traceback and is not this memo. Finals are once per utterance.

## Consequences

On 295 s of clips joined end to end through one recognizer at 40 ms
blocks and dither 0, the traceback and what hangs off it fell from 39 to
1.4 of the microseconds an average block costs, and the block's p50 from
0.059 to 0.021 ms. The decoder is traced once per chunk instead of three
times per block, so the saving is the five blocks in six, not the one.

The same corpus decodes to byte-identical partials and segments over
42,878 blocks, with partial words, with partial alternatives, with an
unknown-word cost and at the model's own dither as well as at zero.

A reader added to `accept` gets the path for nothing. A reader that
wants a path the frame count does not determine has to say so by
changing the key.

## Implemented by

- `[[rr:refresh_best_path]]`
