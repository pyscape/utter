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
- **Broadcast over pre-packed weights.** Removes the horizontal sums,
  which is the rest of the distance to peak. Accumulates each output
  sequentially over k where this kernel uses eight lanes, so it
  changes summation order and moves the near-ties G2 measures.
  Available if the budget is missed, behind a G2 re-run.
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

Over the stock small model's chunk shapes: 1.82 to 1.46 ms of matrix
product per 24-frame chunk, 85 to 106 GFLOP/s on one 12th-generation
Core P-core against a peak near 153, byte-identical over 42,878
blocks. The rest of the distance to peak is the horizontal sums, which
only the rejected broadcast kernel removes.

Odd rows and columns fall to the single-row path, correct and slower,
so a model whose shapes do not suit a four-row tile is measured before
it is claimed to meet the budget.

## Implemented by

- `[[rr:dot4x3]]`
