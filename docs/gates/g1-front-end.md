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

## i-vector: no oracle on this machine

The i-vector estimate has no reference here: no Kaldi build, no
container runtime, and neither the vosk wheel nor kaldi-native-fbank
exposes i-vectors. Evidence short of the gate: faithful mode beats zero
mode by 2.34 word error points (G0), and the streaming decode agrees
with libvosk's partials on 99.7% of blocks (G2), which an i-vector far
from Kaldi's would not allow.

The gate needs an owner-run leg on a machine with a Kaldi build, for
example the container images the consumer's fork recipes build. For ten
takes, with the model directory as `$M` and a take as `$W`:

```
compute-mfcc-feats --config=$M/conf/mfcc.conf --dither=0 \
  "scp:echo take $W|" ark:mfcc.ark
ivector-extract-online2 --config=$M/ivector/ivector_extractor.conf \
  --ivector-period=10 --num-gselect=5 --min-post=0.025 \
  --posterior-scale=0.1 --max-count=100 --num-cg-iters=15 \
  --use-most-recent-ivector=true --max-remembered-frames=1000 \
  ark:spk2utt ark:mfcc.ark ark,t:ivectors.txt
```

where `ivector_extractor.conf` names the model's `splice.conf`,
`online_cmvn.conf`, `final.mat`, `global_cmvn.stats`, `final.dubm` and
`final.ie`, and `spk2utt` maps each take to itself, its utterance key the
WAV's stem.

That tool has no decoder to take a traceback from, so it applies no
silence weighting. The `ivector_dump` binary runs the feature pipeline
alone and so applies none either, and writes the estimate every
`--period` frames in the shape `--ivector-period` produces; `stream
--silence-weight 1.0` turns the weighting off on the decoding path for a
cross-check. With the reference written as `ark,t:ivectors.txt`:

```
cargo build --release
python scripts/g1.py --model $M --ivector-dump target/release/ivector_dump \
  --kaldi ivectors.txt $W...
```

which prints the per-take table and the worst relative difference.
Acceptance stays as TD-2 states it: within 1e-2 relative on ten takes.
