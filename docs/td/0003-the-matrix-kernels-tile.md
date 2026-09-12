# TD-3: The matrix kernel's tile is bounded by the register file

- Status: Proposed
- Date: 2026-09-11
- Tags: performance, network, simd

## Context and problem statement

`[[rr:TD-2#Dependency policy]]` puts the single-precision matrix
product in the library's own hands rather than a crate's, and
`[[rr:TD-2#Performance budgets]]` gives it a budget to meet. The
network is the larger part of a block's compute, and the matrix
product is the larger part of the network, so the shape of the inner
tile decides whether the budget is met.

A tile holds one accumulator per output element it computes, plus the
operand registers it loads each step. AVX2 gives sixteen YMM
registers. A tile that wants more spills an accumulator to the stack
and reloads it every step, and the spill costs more than the wider
tile earns. Nothing in the code says how many registers a tile wants,
so the constraint is invisible at the point where it is violated.

## Decision drivers

- The budget in `[[rr:TD-2#Performance budgets]]` is the bar.
- Parity with libvosk is the contract; a faster kernel that changes
  what the decoder reads is a regression, not an improvement.
- The kernel is the one place the zero-dependency policy puts
  performance at risk, and `[[rr:TD-2#Dependency policy]]` reserves
  the `gemm` feature as the escape if it cannot be met.
- A tile's shape must be justified by a number, not by taste.

## Considered options

- **Four A rows by three B rows.** Twelve accumulators, three B
  registers, one A register: exactly the sixteen the instruction set
  has. Taken.
- **Three A rows by four B rows.** The same twelve accumulators and
  the same twelve-FMA-to-seven-load ratio, but seventeen registers.
  One accumulator spills every eight-element step. This was the shape
  that shipped first, and measuring it against 4x3 is what produced
  this record.
- **Three by three, or four by two.** Fit comfortably, but the ratio
  of arithmetic to loads falls to 1.5 or below and the kernel becomes
  load-bound.
- **A broadcast kernel over pre-packed weights.** What a tuned BLAS
  does, and it would remove the horizontal sums entirely. Rejected for
  now: it accumulates each output sequentially over k where this
  kernel accumulates in eight lanes, so it changes floating-point
  summation order, and the near-ties that decide segment agreement
  would move with it. It is available if the budget is ever missed,
  behind a re-run of G2.
- **`matrixmultiply` behind the `gemm` feature.** The escape
  `[[rr:TD-2#Dependency policy]]` already reserves; not needed.

## Decision outcome

The AVX2 kernel computes a tile of four A rows by three B rows, and a
tile is sized so that its accumulators and operand registers together
do not exceed the sixteen the instruction set provides. A change to
the tile states the register count it implies.

Accumulation order is part of the contract, not an implementation
detail: a tile change keeps the eight-lane accumulation over k and the
horizontal sum per output element, so that the result is unchanged bit
for bit. A change that does not is a change to the acoustics and is
ruled on separately.

## Consequences

Over the stock small model's chunk shapes the retile moved the matrix
product from 1.82 ms to 1.46 ms per 24-frame chunk, 85 to 106 GFLOP/s
on one 12th-generation Core P-core, against a peak near 153. The
decode is unchanged: the same corpus gives byte-identical partials and
segments over 42,878 blocks.

The remaining distance to peak is the horizontal sums, which only the
rejected broadcast kernel removes. A tile of four rows suits a network
run in chunks whose row count is a multiple of four; the odd rows and
columns fall to the single-row path, which is correct but slower, so a
model whose shapes are unfriendly to the tile should be measured
before it is claimed to meet the budget.

## Implemented by

- `[[rr:dot4x3]]`
