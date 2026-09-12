#!/usr/bin/env python3
"""Gate G1: utter's front end against Kaldi.

MFCC half, against kaldi-native-fbank's OnlineMfcc (a C++ port of Kaldi's feature code) on a
set of WAV takes, both at dither 0:

    python scripts/g1.py --conf MODEL/conf/mfcc.conf --dump target/release/mfcc_dump WAV...

kaldi_native_fbank must be importable (pip install kaldi-native-fbank, into a --target dir if
the environment is frozen, then PYTHONPATH).

i-vector half, against `ivector-extract-online2` run on a machine with a Kaldi build. Write
its output as text (ark,t:ivectors.txt), with one utterance key per take named as the WAV
stem, then:

    python scripts/g1.py --model MODEL --ivector-dump target/release/ivector_dump \
        --kaldi ivectors.txt WAV...

The Kaldi tool has no decoder to take a traceback from, so it applies no silence weighting;
the dump binary runs the feature pipeline alone and so applies none either. The invocation
that produces the reference is recorded in docs/gates/g1-front-end.md.
"""

import argparse
import struct
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np


def knf_mfcc(conf_text, samples):
    # Imported here so the i-vector half runs on a machine without it.
    import kaldi_native_fbank as knf

    opts = knf.MfccOptions()
    opts.frame_opts.dither = 0.0
    for line in conf_text.splitlines():
        line = line.strip()
        if not line.startswith("--") or "=" not in line:
            continue
        k, v = line[2:].split("=", 1)
        if k == "sample-frequency":
            opts.frame_opts.samp_freq = float(v)
        elif k == "use-energy":
            opts.use_energy = v == "true"
        elif k == "num-mel-bins":
            opts.mel_opts.num_bins = int(v)
        elif k == "num-ceps":
            opts.num_ceps = int(v)
        elif k == "low-freq":
            opts.mel_opts.low_freq = float(v)
        elif k == "high-freq":
            opts.mel_opts.high_freq = float(v)
    m = knf.OnlineMfcc(opts)
    m.accept_waveform(opts.frame_opts.samp_freq, samples.tolist())
    m.input_finished()
    n = m.num_frames_ready
    return np.array([m.get_frame(i) for i in range(n)], dtype=np.float32)


def read_matrix_dump(path):
    data = Path(path).read_bytes()
    r, c = struct.unpack_from("<ii", data, 0)
    return np.frombuffer(data[8:], dtype="<f4").reshape(r, c)


def read_kaldi_text_ark(path):
    """The matrices in a `ark,t:` file, by utterance key."""
    out, key, rows = {}, None, []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if key is None:
            key, _, line = line.partition("[")
            key = key.strip()
            line = line.strip()
            if not line:
                continue
        closed = line.endswith("]")
        row = [float(v) for v in line.rstrip("]").split()]
        if row:
            rows.append(row)
        if closed:
            out[key] = np.array(rows, dtype=np.float32)
            key, rows = None, []
    return out


def ivector_half(args):
    ref = read_kaldi_text_ark(args.kaldi)
    rows = []
    worst = 0.0
    for wav_path in args.wavs:
        stem = Path(wav_path).stem
        if stem not in ref:
            raise SystemExit(f"{args.kaldi} has no utterance '{stem}'")
        with tempfile.NamedTemporaryFile(suffix=".bin") as tmp:
            subprocess.run(
                [args.ivector_dump, args.model, wav_path, tmp.name, "--period", str(args.period)],
                check=True,
            )
            ours = read_matrix_dump(tmp.name)
        theirs = ref[stem]
        n = min(len(ours), len(theirs))
        diff = np.abs(ours[:n] - theirs[:n])
        scale = max(float(np.abs(theirs[:n]).max()), 1e-9)
        rel = float(diff.max()) / scale
        worst = max(worst, rel)
        rows.append((stem, len(theirs), len(ours), float(diff.max()), rel))
    print("| take | kaldi rows | utter rows | max abs diff | max rel diff |")
    print("|---|---|---|---|---|")
    for row in rows:
        print(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]:.2e} | {row[4]:.2e} |")
    print(f"\nworst relative difference {worst:.2e} [G1 asks 1e-2]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", help="MODEL/conf/mfcc.conf, for the MFCC half")
    ap.add_argument("--dump", help="path to the mfcc_dump binary")
    ap.add_argument("--model", help="model directory, for the i-vector half")
    ap.add_argument("--ivector-dump", help="path to the ivector_dump binary")
    ap.add_argument("--kaldi", help="ivector-extract-online2 output as ark,t text")
    ap.add_argument("--period", type=int, default=10, help="--ivector-period the reference used")
    ap.add_argument("wavs", nargs="+")
    args = ap.parse_args()
    if args.kaldi:
        if not (args.model and args.ivector_dump):
            ap.error("--kaldi needs --model and --ivector-dump")
        ivector_half(args)
        return
    if not (args.conf and args.dump):
        ap.error("the MFCC half needs --conf and --dump")
    conf = Path(args.conf).read_text()
    worst = 0.0
    rows = []
    for wav_path in args.wavs:
        with wave.open(wav_path) as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
        ref = knf_mfcc(conf, pcm)
        with tempfile.NamedTemporaryFile(suffix=".bin") as tmp:
            subprocess.run([args.dump, args.conf, wav_path, tmp.name], check=True)
            ours = read_matrix_dump(tmp.name)
        n = min(len(ref), len(ours))
        diff = np.abs(ref[:n] - ours[:n])
        worst = max(worst, float(diff.max()))
        rows.append((Path(wav_path).stem, len(ref), len(ours), float(diff.max()), float(diff.mean())))
    print("| take | kaldi frames | utter frames | max abs diff | mean abs diff |")
    print("|---|---|---|---|---|")
    for row in rows:
        print(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]:.2e} | {row[4]:.2e} |")
    print(f"\nworst max abs diff {worst:.2e} [G1 asks 1e-3]")


if __name__ == "__main__":
    main()
