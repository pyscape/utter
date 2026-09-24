#!/usr/bin/env python3
"""The speaker front end and network against Kaldi, stage by stage, on public clips.

    python scripts/speaker_oracle.py --data DIR --spk-model DIR --kaldi KALDI_SRC [--clips N]

For each clip `spk_dump` streams the audio through the runtime's speaker path and writes its
features; Kaldi then computes the same stages from the same input:

- the model's MFCC through the online resampler, against `online2-wav-dump-features` with the
  model's `mfcc.conf` and dither off, which runs Kaldi's `LinearResample` in front of its MFCC
  as vosk's speaker features do;
- the network, pooling and embedding over the runtime's own normalised features, against
  `nnet3-xvector-compute` without padding, which pools the frames the network's context lets it
  compute; the runtime pools the same frames, and the vector is centred, whitened and scaled on
  both sides as vosk does.

The mean normalisation between the two is the runtime's own and has no Kaldi counterpart to
compare with: `[[rr:TD-14#Mean normalisation looks back only]]`. The clips are chosen by the
property under test: the loudest and the quietest of the test split's clips for every word, so
both ends of the feature range are exercised.
"""

import argparse
import json
import math
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import numpy.typing as npt


def ark(path: Path) -> npt.NDArray[np.float64]:
    """One utterance of a Kaldi text archive, as a matrix, or a vector's values as one row."""
    txt = path.read_text()
    body = txt[txt.index("[") + 1 : txt.rindex("]")]
    rows = [[float(x) for x in line.split()] for line in body.strip().split("\n") if line.strip()]
    return np.array(rows, dtype=np.float64)


def post_process(raw: npt.NDArray[np.float64], spk: Path) -> npt.NDArray[np.float64]:
    """vosk's `GetSpkVector` after the network: centre, whiten, scale to the expected norm."""
    mean = np.array([float(x) for x in (spk / "mean.vec").read_text().replace("[", " ").replace("]", " ").split()])
    b = (spk / "transform.mat").read_bytes()
    rows, cols = struct.unpack("<i", b[6:10])[0], struct.unpack("<i", b[11:15])[0]
    transform = np.frombuffer(b, dtype="<f4", count=rows * cols, offset=15).reshape(rows, cols)
    z = transform.astype(np.float64) @ (raw.reshape(-1) - mean)
    return np.asarray(z * math.sqrt(len(z)) / np.linalg.norm(z))


def rms(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)
    return float(np.sqrt(np.mean(a * a))) if len(a) else 0.0


def pick(data: Path, per_word: int) -> list[Path]:
    """The loudest and the quietest test clips of every word."""
    by_word: dict[str, list[Path]] = {}
    for line in (data / "testing_list.txt").read_text().split():
        by_word.setdefault(line.split("/")[0], []).append(data / line)
    out: list[Path] = []
    for word in sorted(by_word):
        ranked = sorted(by_word[word], key=rms)
        out += ranked[:per_word] + ranked[-per_word:]
    return out


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Speech Commands v2, extracted")
    ap.add_argument("--spk-model", required=True, help="vosk-model-spk-0.4")
    ap.add_argument("--kaldi", required=True, help="a built Kaldi src directory")
    ap.add_argument("--dump", default="target/release/spk_dump", help="the runtime's spk_dump tool")
    ap.add_argument("--per-word", type=int, default=1, help="loudest and quietest clips per word")
    args = ap.parse_args()
    spk, kaldi = Path(args.spk_model), Path(args.kaldi)
    clips = pick(Path(args.data), args.per_word)
    worst: dict[str, float] = {"mfcc_abs": 0.0, "mfcc_energy_abs": 0.0, "vector_abs": 0.0}
    min_cos = 1.0
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        conf = t / "mfcc.conf"
        conf.write_text((spk / "mfcc.conf").read_text() + "\n--dither=0\n")
        for i, clip in enumerate(clips):
            p = t / f"c{i}"
            out = subprocess.run(
                [args.dump, str(spk), str(clip), str(p)], check=True, capture_output=True, text=True
            ).stdout
            ours = json.loads(out)["unpadded"]
            run(
                [
                    str(kaldi / "online2bin/online2-wav-dump-features"),
                    "--feature-type=mfcc",
                    f"--mfcc-config={conf}",
                    "ark:echo utt utt|",
                    f"scp:echo utt {clip}|",
                    f"ark,t:{p}.kaldi.mfcc.ark",
                ]
            )
            k, u = ark(Path(f"{p}.kaldi.mfcc.ark")), ark(Path(f"{p}.mfcc.ark"))
            if k.shape != u.shape:
                sys.exit(f"{clip}: {k.shape} frames from Kaldi, {u.shape} from the runtime")
            worst["mfcc_abs"] = max(worst["mfcc_abs"], float(np.abs(k[:, 1:] - u[:, 1:]).max()))
            worst["mfcc_energy_abs"] = max(worst["mfcc_energy_abs"], float(np.abs(k[:, 0] - u[:, 0]).max()))
            if ours is None:
                continue
            run(
                [
                    str(kaldi / "nnet3bin/nnet3-xvector-compute"),
                    "--use-gpu=no",
                    "--pad-input=false",
                    "--min-chunk-size=25",
                    str(spk / "final.ext.raw"),
                    f"ark,t:{p}.feats.ark",
                    f"ark,t:{p}.kaldi.xvec.ark",
                ]
            )
            z = post_process(ark(Path(f"{p}.kaldi.xvec.ark")), spk)
            v = np.array(ours["vector"], dtype=np.float64)
            worst["vector_abs"] = max(worst["vector_abs"], float(np.abs(z - v).max()))
            min_cos = min(min_cos, float(z @ v / np.linalg.norm(z) / np.linalg.norm(v)))
    print(json.dumps({"clips": len(clips), **{k: float(f"{v:.3g}") for k, v in worst.items()}, "min_cosine": min_cos}))


if __name__ == "__main__":
    main()
