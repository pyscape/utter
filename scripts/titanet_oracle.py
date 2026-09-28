#!/usr/bin/env python3
"""The TitaNet front end and network against their references, on spans of public clips.

    python scripts/titanet_oracle.py --data DIR --onnx nemo_en_titanet_small.onnx --model DIR
    python scripts/titanet_oracle.py --test-vectors tests/data/titanet-small.txt --onnx ONNX

DIR is Speech Commands v2 extracted; --model is the directory scripts/titanet_convert.py wrote
from the same ONNX file. Needs numpy, kaldi-native-fbank, onnxruntime and sherpa-onnx.

The spans: for the loudest and the quietest test clip of every word, the whole clip, a span of
a multiple of 16 frames and a span of a seeded random length, at seeded offsets. For each,
titanet_dump computes the runtime's log mel features and embedding, and the script checks:

- the log mel features against kaldi-native-fbank configured as sherpa-onnx configures it for
  NeMo models, and the normalised features against the same normalisation done in numpy;
- the network against onnxruntime on identical features: the runtime's own normalised
  features, given to both;
- the whole path against kaldi-native-fbank then onnxruntime at the span's exact length, and
  against sherpa-onnx, which embeds at exact length too; its features are padded to a multiple
  of 16 frames, but the tensor it hands onnxruntime is the span's own length. The spans whose
  frame count is a multiple of 16 are reported apart;
- each whole clip brought to 48 kHz, embedded by the runtime through its resampler and by
  sherpa-onnx at 48 kHz.

--test-vectors writes the model tests' references instead, for the clips in tests/data: each
whole clip and a 40-frame span of it through kaldi-native-fbank, the runtime's double-precision
normalisation and onnxruntime, and each clip held to 48 kHz (every sample three times) through
sherpa-onnx.
"""

import argparse
import json
import random
import struct
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

type F32 = npt.NDArray[np.float32]
type F64 = npt.NDArray[np.float64]
RATE = 16000
BANDS = 80
SEED = 15


def samples_of(frames: int) -> int:
    return 400 + (frames - 1) * 160


def rms(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)
    return float(np.sqrt(np.mean(a * a))) if len(a) else 0.0


def clip(path: Path) -> F32:
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def pick(data: Path) -> list[Path]:
    by_word: dict[str, list[Path]] = {}
    for line in (data / "testing_list.txt").read_text().split():
        by_word.setdefault(line.split("/")[0], []).append(data / line)
    out: list[Path] = []
    for word in sorted(by_word):
        ranked = sorted(by_word[word], key=rms)
        out += [ranked[0], ranked[-1]]
    return out


def spans(clips: list[Path]) -> list[F32]:
    rng = random.Random(SEED)
    out: list[F32] = []
    for i, p in enumerate(clips):
        x = clip(p)
        most = (len(x) - 400) // 160 + 1
        if most < 16:
            continue
        out.append(x)
        for frames in (16 * (1 + i % (most // 16)), rng.randint(1, most)):
            start = rng.randint(0, len(x) - samples_of(frames))
            out.append(x[start : start + samples_of(frames)])
    return out


def knf_fbank(x: F32) -> F32:
    import kaldi_native_fbank as knf

    o = knf.FbankOptions()
    o.frame_opts.samp_freq = RATE
    o.frame_opts.frame_shift_ms = 10
    o.frame_opts.frame_length_ms = 25
    o.frame_opts.dither = 0.0
    o.frame_opts.preemph_coeff = 0.97
    o.frame_opts.remove_dc_offset = False
    o.frame_opts.window_type = "hann"
    o.frame_opts.round_to_power_of_two = True
    o.frame_opts.snip_edges = True
    o.mel_opts.num_bins = BANDS
    o.mel_opts.low_freq = 0
    o.mel_opts.high_freq = -400
    o.mel_opts.is_librosa = True
    fb = knf.OnlineFbank(o)
    fb.accept_waveform(RATE, x.tolist())
    fb.input_finished()
    return np.stack([fb.get_frame(i) for i in range(fb.num_frames_ready)]).astype(np.float32)


def normalise(f: F32) -> F32:
    """Per band over the span, in double precision as the runtime does it."""
    g = f.astype(np.float64)
    mu = g.mean(axis=0)
    sd = np.sqrt(((g - mu) ** 2).mean(axis=0))
    return ((g - mu) / (sd + 1e-5)).astype(np.float32)


def normalise_f32(f: F32) -> F32:
    """Per band over the span in single precision, as the research reference did it."""
    mu = f.mean(axis=0)
    sd = np.sqrt(((f - mu) ** 2).mean(axis=0))
    return ((f - mu) / (sd + 1e-5)).astype(np.float32)


def write_matrices(path: Path, mats: list[F32]) -> None:
    with open(path, "wb") as fh:
        fh.write(struct.pack("<I", len(mats)))
        for m in mats:
            fh.write(struct.pack("<II", *m.shape) + np.ascontiguousarray(m, dtype="<f4").tobytes())


def read_matrices(path: Path) -> list[F32]:
    b = path.read_bytes()
    (n,), at, out = struct.unpack_from("<I", b), 4, []
    for _ in range(n):
        t, d = struct.unpack_from("<II", b, at)
        at += 8
        out.append(np.frombuffer(b, dtype="<f4", count=t * d, offset=at).reshape(t, d))
        at += 4 * t * d
    return out


def write_spans(path: Path, xs: list[tuple[int, F32]]) -> None:
    with open(path, "wb") as fh:
        fh.write(struct.pack("<I", len(xs)))
        for rate, x in xs:
            fh.write(struct.pack("<II", rate, len(x)) + np.ascontiguousarray(x, dtype="<f4").tobytes())


def cosines(a: F64, b: F64) -> F64:
    return np.asarray(np.sum(a * b, axis=1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1))


def compare(ours: F32, ref: F32) -> dict[str, float]:
    a, b = ours.astype(np.float64), ref.astype(np.float64)
    return dict(
        spans=len(a),
        min_cosine=float(cosines(a, b).min()),
        max_abs=float(np.abs(a - b).max()),
        max_rel=float((np.abs(a - b).max(axis=1) / np.abs(b).max(axis=1)).max()),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="Speech Commands v2, extracted")
    ap.add_argument("--onnx", required=True, help="the NeMo TitaNet export")
    ap.add_argument("--model", help="the directory titanet_convert.py wrote from it")
    ap.add_argument("--test-vectors", help="write the model tests' reference vectors here instead")
    ap.add_argument("--dump", default="target/release/titanet_dump", help="the runtime's titanet_dump tool")
    args = ap.parse_args()
    import onnxruntime as ort
    import sherpa_onnx

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])

    def onnx_embed(f: F32) -> F32:
        (e,) = sess.run(["embs"], {"audio_signal": f.T[None], "length": np.asarray([len(f)], dtype=np.int64)})
        return np.asarray(e[0], dtype=np.float32)

    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=args.onnx, num_threads=1, provider="cpu")
    ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

    def sherpa_embed(x: F32, rate: int) -> F32:
        st = ex.create_stream()
        st.accept_waveform(rate, x)
        st.input_finished()
        return np.asarray(ex.compute(st), dtype=np.float32)

    if args.test_vectors:
        lines = []
        for name in ("yes", "no", "seven"):
            x = clip(Path(__file__).resolve().parents[1] / "tests/data" / f"{name}.wav")
            for start, end in ((0, len(x)), (1600, 1600 + samples_of(40))):
                v = onnx_embed(normalise(knf_fbank(x[start:end])))
                lines.append(" ".join([name, str(RATE), str(start), str(end)] + [repr(float(f)) for f in v]))
            v = sherpa_embed(np.repeat(x, 3), 3 * RATE)
            lines.append(" ".join([name, str(3 * RATE), "0", str(3 * len(x))] + [repr(float(f)) for f in v]))
        Path(args.test_vectors).write_text("\n".join(lines) + "\n")
        return
    if not args.data or not args.model:
        ap.error("--data and --model are needed unless --test-vectors is given")
    xs = spans(pick(Path(args.data)))
    wholes = [x for x in xs if len(x) == RATE]
    up = [np.fft.irfft(np.fft.rfft(x.astype(np.float64)), n=3 * len(x)).astype(np.float32) * 3 for x in wholes]
    report: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        write_spans(t / "spans", [(RATE, x) for x in xs] + [(3 * RATE, x) for x in up])
        subprocess.run([args.dump, args.model, "audio", str(t / "spans"), str(t / "out")], check=True)
        emb = np.fromfile(t / "out.emb", dtype="<f4").reshape(len(xs) + len(up), -1)
        fb = read_matrices(t / "out.fbank")
        ours_norm = [normalise(f) for f in fb]
        write_matrices(t / "feats", ours_norm)
        subprocess.run([args.dump, args.model, "net", str(t / "feats"), str(t / "net")], check=True)
        net = np.fromfile(t / "net", dtype="<f4").reshape(len(xs), -1)

    knf = [knf_fbank(x) for x in xs]
    frames = np.asarray([len(f) for f in knf])
    report["frames"] = dict(spans=len(xs), min=int(frames.min()), max=int(frames.max()))
    report["log_mel"] = dict(max_abs=max(float(np.abs(a - b).max()) for a, b in zip(fb, knf, strict=True)))
    report["normalised"] = dict(
        max_abs_same_arithmetic=max(
            float(np.abs(normalise(a) - normalise(b)).max()) for a, b in zip(fb, knf, strict=True)
        ),
        max_abs_against_f32_reference=max(
            float(np.abs(normalise(a) - normalise_f32(b)).max()) for a, b in zip(fb, knf, strict=True)
        ),
    )
    ref_net = np.stack([onnx_embed(f) for f in ours_norm])
    report["network_on_identical_features"] = compare(net, ref_net)
    ref_whole = np.stack([onnx_embed(normalise_f32(f)) for f in knf])
    report["whole_against_knf_onnxruntime"] = compare(emb[: len(xs)], ref_whole)
    sh = np.stack([sherpa_embed(x, RATE) for x in xs])
    report["whole_against_sherpa"] = compare(emb[: len(xs)], sh)
    m16 = frames % 16 == 0
    report["whole_against_sherpa_multiple_of_16"] = compare(emb[: len(xs)][m16], sh[m16])
    sh48 = np.stack([sherpa_embed(x, 3 * RATE) for x in up])
    report["48k_against_sherpa"] = compare(emb[len(xs) :], sh48)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
