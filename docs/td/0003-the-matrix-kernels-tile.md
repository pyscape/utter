# TD-3: The matrix kernel's tile is bounded by the register file

- Tags: performance, network, simd

## Context

`[[rr:TD-2#Dependency policy]]` puts the matrix product in the
library's own hands; `[[rr:TD-2#Performance budgets]]` gives it a
budget. The network is the larger part of a block's compute and the
matrix product the larger part of the network.

A tile needs one accumulator per output element plus its operand
registers. AVX2 has sixteen YMM; a tile wanting more spills one every
step. Nothing in the code states the count, so the limit is invisible
where it is broken.

## Considered options

- **4 A rows x 3 B rows.** 12 accumulators, 3 B, 1 A: sixteen. Taken.
- **3 x 4.** Same 12 accumulators, same 12:7 arithmetic-to-load ratio,
  seventeen registers. Spills every step. This shipped first;
  measuring it against 4x3 produced this record.
- **3 x 3, 4 x 2.** Fit, but the ratio falls to 1.5 or below and the
  kernel becomes load-bound.
- **The same reduction tree, four accumulators at a time.** The
  horizontal sum is most of the distance to peak at small k, and a
  lane fold across pairs of accumulators followed by two shuffled
  adds reaches the same tree `(a0+a4 + a1+a5) + (a2+a6 + a3+a7)` at
  about two shuffle-port uops per output rather than five. Same
  order, same float. Taken.
- **Broadcast over pre-packed weights.** Removes the horizontal sums
  altogether. Accumulates each output sequentially over k where this
  kernel uses eight lanes, so it changes summation order and moves the
  near-ties G2 measures. Prototyped and measured, not taken; see
  Consequences. Available if the budget is missed, behind a G2 re-run.
- **`matrixmultiply`.** The escape `[[rr:TD-2#Dependency policy]]`
  already reserves. Not needed.

## Decision outcome

### A tile fits the register file

The AVX2 kernel computes 4 A rows by 3 B rows. A tile's accumulators
and operand registers together do not exceed sixteen, and a change to
the tile states the count it implies.

### Accumulation order is part of the contract

A tile change keeps the eight-lane accumulation over k and the
horizontal sum per output element, so the result does not move. A
change that alters summation order is a change to the acoustics and is
ruled on separately.

## Consequences

Over the stock small model's chunk shapes, on one 12th-generation Core
P-core against a peak near 153 GFLOP/s, each step byte-identical over
42,878 blocks: 1.82 ms of matrix product per 24-frame chunk at 3 x 4,
1.46 at 4 x 3, 1.30 once the reduction is batched. The loss is the
per-tile epilogue and not the k loop, and it lands on the small-k
shapes: a tile costs k/8 iterations of six cycles plus a fixed
epilogue, so efficiency runs as k/(k+c). At 4 x 3 with c near eighty
the model predicts 94% at k=1280, 71% at k=192 and 55% at k=96 against
measured 94%, 71% and 54%.

The broadcast kernel was prototyped against the same shapes. It is
worth 1.11 ms, a further 15%, but only at 6 A rows by 2 YMM: 4 x 3
wants seventeen registers here too and spills to 1.71 ms, and 4 x 2
fits but leaves its eight accumulators' four-cycle chains exactly
equal to FMA throughput with no slack, reaching only 1.40. Its cost is
that 98% of log-likelihoods move, by at most 5.7e-6 against a
first-percentile per-frame best-minus-second margin of 1.5e-2. Over
the G2 corpus not one partial or segment moved, but that is a
measurement on one corpus and not the guarantee the section above
states.

Odd rows and columns fall to the single-row path, correct and slower,
so a model whose shapes do not suit a four-row tile is measured before
it is claimed to meet the budget.

## Implemented by

- `[[rr:dot4x3]]`
- `[[rr:hsum4]]`
