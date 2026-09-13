# G1: front end against Kaldi

Gate: `[[rr:TD-2#Verification and acceptance]]`, G1.

## MFCC: passed

- Reference: kaldi-native-fbank 1.22.3, a C++ port of Kaldi's feature
  code, `OnlineMfcc` with the model's `conf/mfcc.conf` and dither 0.
- Runtime: `mfcc_dump`, the online front end fed 40 ms blocks, dither 0.
- Ten takes of a consumer replay set, 16 kHz mono, 65,845 frames in all.

| Quantity | Value | Gate |
|---|---|---|
| frame counts | equal on every take | |
| worst absolute difference over all frames and coefficients | 7.8e-4 | 1e-3 |
| mean absolute difference | 5e-5 to 7e-5 per take | |

The residual is float summation order in the filter bank and DCT; Kaldi
sums in single precision in BLAS order, the runtime in its own.

### MFCC against Kaldi itself

The same twelve takes as the i-vector leg below, through Kaldi's
`compute-mfcc-feats` at dither 0: frame counts equal on every take,
16,610 frames; worst absolute difference 1.34e-2, worst relative
1.02e-4. The absolute figure is over the 1e-3 the gate states and is
not a regression: kaldi-native-fbank differs from Kaldi by 1.30e-2 on
the same takes and from the runtime by 1.08e-2, and on four takes the
runtime is the nearer of the two ports. The large cells sit in the
loudest frames, the log of a filter-bank energy in a spectral null
amplifying single-precision FFT error, the mechanism named above. The
1e-3 bound holds on the consumer's ten takes and not on these, so it is
a property of the take set; a relative bound would hold on both, and
TD-2 states the absolute one.

## i-vector: against the wheel's Kaldi, passed once its quiet-frame rule was matched

- Reference: `ivector-extract-online2` from Kaldi built on this machine
  at 93ef0, the commit the stock 0.3.45 wheel's `libvosk.so` names in
  its version string `5.5.1082~1-93ef0` (alphacep's `vosk` branch; the
  branch tip bc5baf14 is byte-identical to it in `src/feat`,
  `src/ivector`, `src/online2`, `src/nnet3`, `src/decoder`, `src/lat`
  and `src/fstext`, and its output on every take here is byte-identical
  too). Fed `compute-mfcc-feats` output at dither 0 with the model's
  `conf/mfcc.conf`, under an `ivector_extractor.conf` naming the
  model's `splice.conf`, `online_cmvn.conf`, `final.mat`,
  `global_cmvn.stats`, `final.dubm` and `final.ie` by absolute path.
- Runtime: `ivector_dump` at `--period 10`, the feature pipeline alone.
  Neither side applies silence weighting; `stream --silence-weight 1.0`
  turns it off on the decoding path for a cross-check.

Two take sets, because the first was unrepresentative.

| Takes | Before the rule was matched | After | Gate |
|---|---|---|---|
| first testing clip of ten words, and two recordings (1,662 rows) | 7.24e-4 | 3.49e-4 | 1e-2 |
| the quietest testing clip of the same ten words | 1.86e-1, ten of ten over | **1.41e-5** | 1e-2 |
| the same ten, reference run with `--cmn-min-energy=-1000` | 1.04e-5 | | |

The cause is one commit of the fork, `ecb4b4715` "Avoid CMVN update for
quiet frames" (2022-11-14, an ancestor of 93ef0, not in upstream
Kaldi): `OnlineCmvn::GetFrame` computes the sliding-window statistics
only for a frame whose raw c0 is above `--cmn-min-energy`, default
50.0, and for a quieter frame re-smooths the previous frame's
already-smoothed statistics instead. The runtime's `cmvn_frame`
computes the window for every frame, which is upstream Kaldi's
behaviour. On a clip the difference is present at the first row and
grows with the quiet prefix (`left/f2e59fea_nohash_4`: 7.8e-3 at row 0
to 1.86e-1; cosine 0.9992 to 0.9749); it is not a lag, a scale or an
offset.

How much audio the rule touches, by `compute-mfcc-feats` over the whole
testing split and the six recordings:

| Quantity | testing split (11,005 clips) | recordings |
|---|---|---|
| frames | 1,059,699 | 39,928 |
| frames with c0 at or below 50 | 105,679 (9.97%) | 9 |
| of which at the digital-zero floor | 1,278 | 8 |
| clips with at least one | 3,735 (33.9%) | 4 of 6 |
| runs, and runs of ten frames or more | 8,679; 3,776 | 4 |
| runs at a clip's start / end / interior | 2,721 / 1,718 / 4,240 | |

Quiet speech, not edge zeros: the longest run is 93 frames of a
98-frame clip. The first take set had none of it in its 980 word-clip
frames, which is why it passed before the rule was matched.

The runtime now carries the rule, `[[rr:cmvn_frame]]`: the window is
recomputed only above the threshold, a quieter frame reuses and
re-smooths what the previous call left, and the weighted update fetches
every frame's normalized view before it reads the weights, as Kaldi's
`GetFrames` does, since the state depends on the order of the fetches.
On the gate corpus the rule never fires: a room's floor keeps c0 above
50, and the stream is byte-identical before and after on all 33 takes,
42,878 blocks and 209 segments, so G2's figures stand to the digit and
its twelve missed segments are not this. What changes is the i-vector on
quiet audio, a tenth of the public benchmark's frames; what that
reaches is almost nothing: on the Speech Commands testing split not
one final moved (10,913 of 11,005 identical to the wheel before and
after, the same 92 disagreements), and the trust and states pages
moved by two clips in a bucket and the third decimal of a figure. The
rule matters to a host whose input is quiet or padded, where the
i-vector would otherwise drift from the wheel's; it does not move
this decoder's words.

The command, with `--num-cg-iters` dropped: the tool does not register
it and its value is fixed at 15. `--use-most-recent-ivector=true` is
accepted and then set to false by the binary, the sequence that lines
the rows up.

```sh
compute-mfcc-feats --config=$M/conf/mfcc.conf --dither=0 \
  "scp:echo take $W|" ark:mfcc.ark
ivector-extract-online2 --config=ivector_extractor.conf \
  --ivector-period=10 --num-gselect=5 --min-post=0.025 \
  --posterior-scale=0.1 --max-count=100 \
  --use-most-recent-ivector=true --max-remembered-frames=1000 \
  ark:spk2utt ark:mfcc.ark ark,t:ivectors.txt
cargo build --release
python scripts/g1.py --model $M --ivector-dump target/release/ivector_dump \
  --kaldi ivectors.txt $W...
```

`spk2utt` maps each take to itself, its utterance key the take's folder
and stem. Kaldi builds from the `vosk` branch in about thirteen minutes
by the steps of vosk-api's `kaldi-build.ipynb`: `make openfst` in
`tools`, OpenBLAS by `extras/install_openblas.sh`, then `make depend`
and `make featbin online2bin` in `src`; the extractor lives in
`online2bin`. A release-matched worktree at 93ef0 configures against
the tip's tools with `--fst-root` and `--openblas-root` and builds in
two minutes. Under gcc 15 OpenBLAS's link test fails on implicit
declarations, and re-running its `make` with
`CFLAGS="-Wno-implicit-function-declaration -Wno-implicit-int
-Wno-int-conversion"` completes it. The wheel's Kaldi commit is read
from its binary: `strings libvosk.so | grep -oE '5\.5\.[0-9]+~1-[0-9a-f]+'`.
