#!/usr/bin/env python3
"""Gate G1, MFCC half: utter's online MFCC frames against kaldi-native-fbank's OnlineMfcc (a
C++ port of Kaldi's feature code) on a set of WAV takes, both at dither 0.

    python scripts/g1.py --conf MODEL/conf/mfcc.conf --dump target/release/mfcc_dump WAV...

kaldi_native_fbank must be importable (pip install kaldi-native-fbank, into a --target dir if
the environment is frozen, then PYTHONPATH).
"""

import argparse
import struct
import subprocess
import tempfile
import wave
from pathlib import Path

import kaldi_native_fbank as knf
import numpy as np


def knf_mfcc(conf_text, samples):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True)
    ap.add_argument("--dump", required=True, help="path to the mfcc_dump binary")
    ap.add_argument("wavs", nargs="+")
    args = ap.parse_args()
    conf = Path(args.conf).read_text()
    worst = 0.0
    rows = []
    for wav_path in args.wavs:
        with wave.open(wav_path) as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
        ref = knf_mfcc(conf, pcm)
        with tempfile.NamedTemporaryFile(suffix=".bin") as tmp:
            subprocess.run([args.dump, args.conf, wav_path, tmp.name], check=True)
            data = Path(tmp.name).read_bytes()
        r, c = struct.unpack_from("<ii", data, 0)
        ours = np.frombuffer(data[8:], dtype="<f4").reshape(r, c)
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
