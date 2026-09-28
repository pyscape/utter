#!/usr/bin/env python3
"""Speaker evidence on short commands: how often a speaker model names the right speaker from one
spoken word, how much speech it needs, and when utter's partials first carry the evidence.
Google Speech Commands v2 (CC BY 4.0), whose file names carry an anonymous speaker id.

    python scripts/speaker_evidence.py --data DIR --model MODEL_DIR --speaker-models DIR \\
        [--kaldi KALDI_SRC] [--lib target/release/libutter.so] [--out docs/benchmarks/speaker-evidence]

MODEL_DIR is the stock small English model. The vosk wheel decodes every clip under the Speech
Commands page's grammar with the stock speaker model set: its final gives the wheel's own speaker
vector and the clip's word span, the speech every other model except utter is given. DIR holds
the speaker models: vosk-model-spk-0.4 unpacked, and nemo_en_titanet_small.onnx and
wespeaker_en_voxceleb_CAM++.onnx from sherpa-onnx's speaker recognition model release, run
through the sherpa-onnx package. KALDI_SRC is a built Kaldi src directory; with it the same
x-vector network also runs through compute-mfcc-feats, apply-cmvn-sliding and
nnet3-xvector-compute over the span alone, the network as it was trained to be used, without the
wheel's half-second floor. utter runs through its C ABI over continuous streams, as a host feeds
it: one stream of each speaker's enrollment clips, and games of two speakers taking turns; with
the x-vector, and with TitaNet-small as --titanet holds it converted, both as a recognizer's
speaker model and through the stateless embed() on the wheel's span.

Speakers of the test split with at least --min-clips clips take part. Each enrolls from --enroll
clips, one vector per clip (per spoken word for utter), normalised and averaged into a profile. Up to
--probes other clips per speaker are probes, scored by cosine against every profile:

- top-1 of K: among K enrolled speakers drawn at random, the probe's own profile scores highest;
  the draws are shared by every engine;
- equal error rate over own-profile against other-profile scores, among probes with evidence;
- accepted at 1% false accept: the share of probes whose own-profile score clears the threshold
  that lets 1% of other-profile scores through, a gate's operating point;
- no evidence: probes an engine gave no vector, which are neither identified nor accepted.

The same figures by the word's length, then over each speaker's probe words joined into one
stream and cut at a growing length of speech, which is how evidence accumulates as a command goes
on; utter's figure there is its own partial's evidence at the moment it has pooled that much.
How long after a word's end its evidence shows. Last, the game streams are decoded again by both engines with and without their speaker models,
for per-block compute, result size, the host's parse, and whether the evidence moves any result.

Writes `<out>.md` and `<out>.json`; what a run means goes in `<out>.reading.md`, which the page
appends to itself, so the prose survives the next run.
"""

import argparse
import bisect
import ctypes
import json
import random
import subprocess
import sys
import tempfile
import time
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent))
import speech_commands as sc  # noqa: E402

Json = dict[str, Any]
type Vec = NDArray[np.float64]
RATE = sc.RATE
LENGTHS_MS = (250, 500, 750, 1000, 1500, 2000, 3000)
WORD_BINS = ((0, 400, "under 400 ms"), (400, 600, "400 to 600 ms"), (600, 1_000_000, "600 ms and over"))
TABLE_SIZES = (2, 4)
TABLE_DRAWS = 20
FALSE_ACCEPT = 0.01
STREAMS_PER_SPEAKER = 3
VOSK, KALDI, UTTER, TITANET, CAMPP = "vosk", "x-vector (Kaldi)", "utter", "TitaNet-small", "CAM++"
UTTER_MARGIN, UTTER_COLD = "utter, 8 dB floor margin", "utter, new recognizer per word"
UTTER_TN = "utter, TitaNet-small"
UTTER_TN_MARGIN = "utter, TitaNet-small, 8 dB floor margin, span as reported"
UTTER_TN_EMBED = "utter's embed(), TitaNet-small"
MARGIN_DB = 8.0
# The readings and final alternatives the latency pass's readings rows carry.
READINGS = 3
AFTER_END_MS = (0, 100, 200, 300, 400, 500, 600, 800, 1000)
TITANET_DIR = "titanet-small"
SHERPA_FILES = {TITANET: "nemo_en_titanet_small.onnx", CAMPP: "wespeaker_en_voxceleb_CAM++.onnx"}
SPK_DIR = "vosk-model-spk-0.4"


@dataclass
class Clip:
    key: str
    speaker: str
    pcm: bytes
    # The word span in samples from the wheel's final; None when the final had no word.
    span: tuple[int, int] | None = None


@dataclass
class Probe:
    """One scored item: a probe clip's word, or a prefix of a speaker's joined words."""

    key: str
    speaker: str
    pcm: bytes
    length_ms: float
    vectors: dict[str, Vec | None] = field(default_factory=dict)


def unit(v: Vec) -> Vec:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def pct(x: float) -> str:
    return "-" if x != x else f"{100.0 * x:.1f}%"


# --- engines --------------------------------------------------------------------------------------


class Wheel:
    """The vosk wheel with the stock speaker model set, one fresh recognizer per item."""

    def __init__(self, model_dir: str, spk_dir: Path, grammar: Sequence[str]) -> None:
        import vosk

        vosk.SetLogLevel(-1)
        self.vosk = vosk
        self.model = vosk.Model(model_dir)
        self.spk = vosk.SpkModel(str(spk_dir))
        self.grammar = json.dumps(list(grammar))

    def decode(self, pcm: bytes) -> Json:
        rec = self.vosk.KaldiRecognizer(self.model, float(RATE), self.grammar)
        rec.SetWords(True)
        rec.SetSpkModel(self.spk)
        rec.AcceptWaveform(pcm)
        out: Json = json.loads(rec.FinalResult())
        return out

    def timed(self, pcm: bytes, block_ms: int, speaker: bool) -> "Timed":
        """A stream through one recognizer, words on partials, each block timed as utter's is."""
        rec = self.vosk.KaldiRecognizer(self.model, float(RATE), self.grammar)
        rec.SetWords(True)
        rec.SetPartialWords(True)
        if speaker:
            rec.SetSpkModel(self.spk)
        out: list[tuple[float, bool, str]] = []
        for b in blocks(pcm, block_ms):
            t = time.perf_counter()
            closed = bool(rec.AcceptWaveform(b))
            text = rec.Result() if closed else rec.PartialResult()
            out.append((time.perf_counter() - t, closed, text))
        t = time.perf_counter()
        text = rec.FinalResult()
        out.append((time.perf_counter() - t, True, text))
        return [(dt, closed, s.encode()) for dt, closed, s in out]

    @staticmethod
    def vector(res: Mapping[str, Any]) -> Vec | None:
        spk = res.get("spk")
        return np.asarray(spk, dtype=np.float64) if spk else None

    @staticmethod
    def span(res: Mapping[str, Any], n_samples: int) -> tuple[int, int] | None:
        words = res.get("result") or []
        if not words:
            return None
        a = max(0, round(min(w["start"] for w in words) * RATE))
        b = min(n_samples, round(max(w["end"] for w in words) * RATE))
        return (a, b) if b > a else None


class Sherpa:
    """A sherpa-onnx speaker embedding model over whole items."""

    def __init__(self, path: Path) -> None:
        import sherpa_onnx

        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(path), num_threads=1, provider="cpu")
        self.ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self.seconds = 0.0
        self.audio = 0.0

    def embed(self, pcm: bytes) -> Vec | None:
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        t = time.perf_counter()
        st = self.ex.create_stream()
        st.accept_waveform(RATE, x)
        st.input_finished()
        v = np.asarray(self.ex.compute(st), dtype=np.float64) if self.ex.is_ready(st) else None
        self.seconds += time.perf_counter() - t
        self.audio += len(x) / RATE
        return v


def kaldi_matrix(path: Path) -> NDArray[np.float64]:
    """A Kaldi float matrix or vector file, binary (`\\0B` then `FM`/`FV`) or text (`[ ... ]`)."""
    b = path.read_bytes()
    if b.startswith(b"\0BFM "):
        rows = int.from_bytes(b[6:10], "little", signed=True)
        cols = int.from_bytes(b[11:15], "little", signed=True)
        return np.frombuffer(b, dtype="<f4", count=rows * cols, offset=15).reshape(rows, cols).astype(np.float64)
    if b.startswith(b"\0BFV "):
        dim = int.from_bytes(b[6:10], "little", signed=True)
        return np.frombuffer(b, dtype="<f4", count=dim, offset=10).astype(np.float64)
    body = b.decode().replace("[", " ").replace("]", " ")
    return np.array([float(x) for x in body.split()], dtype=np.float64)


class KaldiXvector:
    """The x-vector network through Kaldi's own binaries, batched: the span's features, the
    centred sliding mean over them, the network without padding, then the wheel's centring,
    whitening and length scaling."""

    def __init__(self, kaldi: Path, spk_dir: Path) -> None:
        self.kaldi = kaldi
        self.spk = spk_dir
        self.mean = kaldi_matrix(spk_dir / "mean.vec")
        self.transform = kaldi_matrix(spk_dir / "transform.mat")
        self.seconds = 0.0
        self.audio = 0.0

    def post(self, raw: Vec) -> Vec:
        z: Vec = self.transform @ (raw - self.mean)
        n = float(np.linalg.norm(z))
        return z * (np.sqrt(len(z)) / n) if n > 0 else z

    def embed_all(self, items: Mapping[str, bytes]) -> dict[str, Vec | None]:
        out: dict[str, Vec | None] = {k: None for k in items}
        if not items:
            return out
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            names = {}
            with open(d / "wav.scp", "w") as scp:
                for i, (k, pcm) in enumerate(items.items()):
                    name = f"u{i:07d}"
                    names[name] = k
                    p = d / f"{name}.wav"
                    with wave.open(str(p), "wb") as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)
                        w.setframerate(RATE)
                        w.writeframes(pcm)
                    scp.write(f"{name} {p}\n")
                    self.audio += len(pcm) / 2 / RATE
            feat, nnet = self.kaldi / "featbin", self.kaldi / "nnet3bin"
            cmd = (
                f"{feat}/compute-mfcc-feats --config={self.spk}/mfcc.conf --dither=0 scp:{d}/wav.scp ark:- | "
                f"{feat}/apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- | "
                f"{nnet}/nnet3-xvector-compute --use-gpu=no --pad-input=false --min-chunk-size=25 "
                f"{self.spk}/final.ext.raw ark:- ark,t:{d}/xvectors.txt"
            )
            t = time.perf_counter()
            # nnet3-xvector-compute exits nonzero when no item reached its minimum length, which
            # leaves every item without a vector rather than failing the run.
            done = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], capture_output=True, text=True)
            self.seconds += time.perf_counter() - t
            xv = d / "xvectors.txt"
            if done.returncode != 0 and not (xv.exists() and xv.read_text().strip()):
                sc.note(f"kaldi: no vectors for {len(items)} items: {done.stderr.strip().splitlines()[-1:]}")
                return out
            for line in xv.read_text().splitlines():
                name, _, rest = line.partition(" ")
                vals = rest.replace("[", " ").replace("]", " ").split()
                if name in names and vals:
                    out[names[name]] = self.post(np.array([float(x) for x in vals], dtype=np.float64))
        return out


class UtterLib:
    """utter's C ABI through ctypes."""

    def __init__(self, path: str) -> None:
        lib = ctypes.CDLL(path)
        vp, cp, ci, cf = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_float
        for name, res, args in [
            ("utter_version", cp, []),
            ("utter_revision", cp, []),
            ("utter_model_new", vp, [cp]),
            ("utter_spk_model_new", vp, [cp]),
            ("utter_recognizer_new_grm", vp, [vp, cf, cp]),
            ("utter_recognizer_set_words", None, [vp, ci]),
            ("utter_recognizer_set_partial_words", None, [vp, ci]),
            ("utter_recognizer_set_alternatives", None, [vp, ci]),
            ("utter_recognizer_set_max_alternatives", None, [vp, ci]),
            ("utter_recognizer_set_spk_model", ci, [vp, vp]),
            ("utter_recognizer_set_endpoint_floor_margin", None, [vp, cf]),
            ("utter_recognizer_accept_waveform_s", ci, [vp, cp, ci]),
            ("utter_recognizer_partial_result", cp, [vp]),
            ("utter_recognizer_result", cp, [vp]),
            ("utter_recognizer_final_result", cp, [vp]),
            ("utter_recognizer_free", None, [vp]),
            ("utter_spk_model_dim", ci, [vp]),
            ("utter_spk_model_embed", ci, [vp, ctypes.POINTER(cf), ci, cf, ctypes.POINTER(cf), ci]),
        ]:
            fn = getattr(lib, name)
            fn.restype = res
            fn.argtypes = args
        self.lib = lib
        self.version = str(lib.utter_version().decode())
        self.revision = str(lib.utter_revision().decode())
        self.model: int | None = None
        self.spk: int | None = None
        self.tn: int | None = None
        self.seconds = 0.0
        self.audio = 0.0

    def open(self, model_dir: str, spk_dir: Path, tn_dir: Path) -> None:
        self.model = self.lib.utter_model_new(model_dir.encode())
        self.spk = self.lib.utter_spk_model_new(str(spk_dir).encode())
        self.tn = self.lib.utter_spk_model_new(str(tn_dir).encode())
        if not self.model or not self.spk or not self.tn:
            raise SystemExit("utter could not open the model or a speaker model")

    def embed(self, pcm: bytes) -> Vec | None:
        """TitaNet's stateless embedding of 16-bit samples, as a host scheduling it would call it."""
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        dim = self.lib.utter_spk_model_dim(self.tn)
        out = (ctypes.c_float * dim)()
        t = time.perf_counter()
        n = self.lib.utter_spk_model_embed(
            self.tn, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(x), float(RATE), out, dim
        )
        self.seconds += time.perf_counter() - t
        self.audio += len(x) / RATE
        return np.asarray(out[:n], dtype=np.float64) if n > 0 else None


@dataclass(frozen=True)
class Setup:
    """How a stream's recognizer is set: which speaker model, a floor margin, and readings with
    final alternatives."""

    speaker: str | None = "x-vector"
    margin: float | None = None
    readings: int = 0


PLAIN, BARE = Setup(), Setup(speaker=None)
SETUPS = {
    UTTER: PLAIN,
    UTTER_MARGIN: Setup(margin=MARGIN_DB),
    UTTER_TN: Setup(speaker="titanet"),
    UTTER_TN_MARGIN: Setup(speaker="titanet", margin=MARGIN_DB),
}


class UtterRec:
    """One stream through utter: word entries on, the speaker model set unless told not to."""

    def __init__(self, u: UtterLib, grammar: Sequence[str], setup: Setup = PLAIN) -> None:
        self.u = u
        self.h = new_recognizer(u, grammar, setup)
        self.fed = 0

    def accept(self, pcm: bytes) -> bool:
        self.fed += len(pcm) // 2
        return bool(self.u.lib.utter_recognizer_accept_waveform_s(self.h, pcm, len(pcm) // 2) == 1)

    def _json(self, fn: Any) -> Json:
        out: Json = json.loads(fn(self.h))
        return out

    def partial(self) -> Json:
        return self._json(self.u.lib.utter_recognizer_partial_result)

    def closed(self) -> Json:
        return self._json(self.u.lib.utter_recognizer_result)

    def final(self) -> Json:
        out = self._json(self.u.lib.utter_recognizer_final_result)
        self.u.lib.utter_recognizer_free(self.h)
        return out


def new_recognizer(u: UtterLib, grammar: Sequence[str], setup: Setup) -> int:
    lib = u.lib
    h: int = lib.utter_recognizer_new_grm(u.model, float(RATE), json.dumps(list(grammar)).encode())
    lib.utter_recognizer_set_words(h, 1)
    lib.utter_recognizer_set_partial_words(h, 1)
    if setup.margin is not None:
        lib.utter_recognizer_set_endpoint_floor_margin(h, setup.margin)
    if setup.readings:
        lib.utter_recognizer_set_alternatives(h, setup.readings)
        lib.utter_recognizer_set_max_alternatives(h, setup.readings)
    spk = {None: None, "x-vector": u.spk, "titanet": u.tn}[setup.speaker]
    if spk and lib.utter_recognizer_set_spk_model(h, spk) != 0:
        raise SystemExit("utter refused the speaker model")
    return h


def evidence(entry: Mapping[str, Any]) -> Vec | None:
    spk = entry.get("spk")
    return np.asarray(spk, dtype=np.float64) if spk else None


def blocks(pcm: bytes, block_ms: int) -> list[bytes]:
    n = RATE * block_ms // 1000 * 2
    return [pcm[i : i + n] for i in range(0, len(pcm), n)]


# --- the passes -----------------------------------------------------------------------------------


def speakers_of(data: Path, min_clips: int) -> dict[str, list[str]]:
    by: dict[str, list[str]] = {}
    for line in (data / "testing_list.txt").read_text().split():
        by.setdefault(line.split("/")[1].split("_nohash_")[0], []).append(line)
    return {s: sorted(c) for s, c in sorted(by.items()) if len(c) >= min_clips}


def utter_enroll(u: UtterLib, grammar: Sequence[str], clips: Sequence[Clip], block_ms: int, setup: Setup) -> list[Vec]:
    """One stream of a speaker's enrollment clips: the evidence of every spoken final word that has
    one. The clips run together into a few finals, so one vector per final would leave a profile
    resting on about four."""
    rec = UtterRec(u, grammar, setup)
    finals: list[Json] = []
    for c in clips:
        for b in blocks(c.pcm, block_ms):
            if rec.accept(b):
                finals.append(rec.closed())
    finals.append(rec.final())
    return [v for f in finals for e in f.get("result", []) if spoken(e) and (v := evidence(e)) is not None]


def word_entries(res: Mapping[str, Any], closed: bool) -> tuple[list[Json], list[Json]]:
    """A result's best-path word entries, and its readings' or alternatives' word entries."""
    if closed:
        alts = res.get("alternatives")
        if alts is not None:
            best = alts[0].get("result", []) if alts else []
            return best, [e for a in alts[1:] for e in a.get("result", [])]
        return res.get("result", []), []
    return res.get("partial_result", []), [e for a in res.get("partial_alternatives", []) for e in a.get("result", [])]


def spoken(e: Mapping[str, Any]) -> bool:
    return str(e["word"]) not in ("[sil]", "[speech]")


def utter_game(
    u: UtterLib, grammar: Sequence[str], order: Sequence[Clip], block_ms: int, setup: Setup = PLAIN
) -> dict[str, Json]:
    """A game's stream, clips back to back. For each clip: the final word entry with evidence
    inside its range that pooled most frames, and the first best-path entry with evidence inside
    it on a partial and on any result, with the samples fed when it appeared."""
    starts, at = [], 0
    for c in order:
        starts.append(at)
        at += len(c.pcm) // 2

    def clip_of(e: Mapping[str, Any]) -> str | None:
        mid = (int(e["start_sample"]) + int(e["end_sample"])) // 2
        i = bisect.bisect_right(starts, mid) - 1
        return order[i].key if 0 <= i < len(order) else None

    rec = UtterRec(u, grammar, setup)
    seen: dict[str, Json] = {}
    seen_any: dict[str, Json] = {}
    finals: list[Json] = []

    def look(res: Json, closed: bool) -> None:
        for e in word_entries(res, closed)[0]:
            k, v = clip_of(e), evidence(e)
            if k is None or v is None or not spoken(e):
                continue
            for got in (seen_any,) if closed else (seen, seen_any):
                got.setdefault(k, dict(fed=rec.fed, vector=v, end_sample=int(e["end_sample"])))

    for c in order:
        for b in blocks(c.pcm, block_ms):
            if rec.accept(b):
                finals.append(rec.closed())
                look(finals[-1], True)
            else:
                look(rec.partial(), False)
    finals.append(rec.final())
    look(finals[-1], True)
    best: dict[str, Json] = {}
    for f in finals:
        for e in word_entries(f, True)[0]:
            k = clip_of(e)
            v = evidence(e)
            if k is None or v is None:
                continue
            if k not in best or e["spk_frames"] > best[k]["frames"]:
                best[k] = dict(vector=v, frames=int(e["spk_frames"]), end_sample=int(e["end_sample"]))
    out = {k: dict(final=b, first=seen.get(k), first_any=seen_any.get(k)) for k, b in best.items()}
    for k, s in seen_any.items():
        out.setdefault(k, dict(final=None, first=seen.get(k), first_any=s))
    return out


def utter_cold(u: UtterLib, grammar: Sequence[str], clip: Clip, block_ms: int) -> Vec | None:
    """A clip through a recognizer built for it alone, as a host that builds a new decoder at
    every grammar change does: the word with evidence that pooled most frames."""
    rec = UtterRec(u, grammar)
    finals: list[Json] = []
    for b in blocks(clip.pcm, block_ms):
        if rec.accept(b):
            finals.append(rec.closed())
    finals.append(rec.final())
    words = [e for f in finals for e in f.get("result", []) if e.get("spk")]
    best = max(words, key=lambda e: int(e["spk_frames"]), default=None)
    return evidence(best) if best else None


def utter_accumulating(
    u: UtterLib,
    grammar: Sequence[str],
    pcm: bytes,
    block_ms: int,
    lengths: Sequence[int],
    setup: Setup = PLAIN,
) -> dict[int, Vec | None]:
    """A speaker's joined words through one stream: for each length, the partial's top-level
    evidence at the first partial that has pooled that much speech."""
    rec = UtterRec(u, grammar, setup)
    got: dict[int, Vec | None] = {n: None for n in lengths}

    def take(res: Mapping[str, Any]) -> None:
        v = evidence(res)
        if v is None:
            return
        pooled_ms = 10 * int(res["spk_frames"])
        for n in lengths:
            if got[n] is None and pooled_ms >= n:
                got[n] = v

    for b in blocks(pcm, block_ms):
        if rec.accept(b):
            take(rec.closed())
        else:
            take(rec.partial())
    take(rec.final())
    return got


SPK_KEYS = frozenset(("spk", "spk_frames", "spk_start", "spk_end"))
type Timed = list[tuple[float, bool, bytes]]


def utter_timed(u: UtterLib, grammar: Sequence[str], pcm: bytes, block_ms: int, setup: Setup) -> Timed:
    """A stream through utter's C ABI, each block timed from the call that takes its audio to the
    return of the result read after it."""
    lib = u.lib
    h = new_recognizer(u, grammar, setup)
    out: Timed = []
    for b in blocks(pcm, block_ms):
        t = time.perf_counter()
        closed = lib.utter_recognizer_accept_waveform_s(h, b, len(b) // 2) == 1
        text = lib.utter_recognizer_result(h) if closed else lib.utter_recognizer_partial_result(h)
        out.append((time.perf_counter() - t, closed, bytes(text)))
    t = time.perf_counter()
    text = lib.utter_recognizer_final_result(h)
    out.append((time.perf_counter() - t, True, bytes(text)))
    lib.utter_recognizer_free(h)
    return out


def without_evidence(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: without_evidence(v) for k, v in x.items() if k not in SPK_KEYS}
    if isinstance(x, list):
        return [without_evidence(v) for v in x]
    return x


def latency_pass(u: UtterLib, wheel: "Wheel", grammar: Sequence[str], streams: Sequence[bytes], block_ms: int) -> Json:
    """The game streams again through each engine and setup, in rotating order from one stream to
    the next. Each utter setup with a speaker model is checked against the same setup without one,
    its speaker keys removed."""
    variants: dict[str, tuple[str, Setup]] = {
        UTTER: (UTTER, BARE),
        f"{UTTER}, x-vector set": (UTTER, PLAIN),
        UTTER_TN: (UTTER, SETUPS[UTTER_TN]),
        f"{UTTER}, {READINGS} readings": (UTTER, Setup(speaker=None, readings=READINGS)),
        f"{UTTER}, {READINGS} readings, x-vector set": (UTTER, Setup(readings=READINGS)),
        f"{UTTER_TN}, {READINGS} readings": (UTTER, Setup(speaker="titanet", readings=READINGS)),
        f"{UTTER}, 8 dB floor margin": (UTTER, Setup(speaker=None, margin=MARGIN_DB)),
        f"{UTTER}, 8 dB floor margin, x-vector set": (UTTER, SETUPS[UTTER_MARGIN]),
        UTTER_TN_MARGIN: (UTTER, SETUPS[UTTER_TN_MARGIN]),
        VOSK: (VOSK, BARE),
        f"{VOSK}, speaker model set": (VOSK, PLAIN),
    }
    bare_of = {s: k for k, (e, s) in variants.items() if e == UTTER and s.speaker is None}
    acc: dict[str, dict[str, list[float]]] = {k: dict(t=[], closing=[], size=[], parse=[]) for k in variants}
    documents = differ = 0
    names = list(variants)
    for i, pcm in enumerate(streams):
        runs: dict[str, Timed] = {}
        r0 = i % len(names)
        for k in names[r0:] + names[:r0]:
            e, s = variants[k]
            runs[k] = (
                utter_timed(u, grammar, pcm, block_ms, s)
                if e == UTTER
                else wheel.timed(pcm, block_ms, s.speaker is not None)
            )
        for k, run_ in runs.items():
            a = acc[k]
            for dt, closed, text in run_:
                a["t"].append(dt)
                a["size"].append(len(text))
                if closed:
                    a["closing"].append(dt)
                t = time.perf_counter()
                json.loads(text)
                a["parse"].append(time.perf_counter() - t)
        for k, (e, s) in variants.items():
            if e != UTTER or s.speaker is None:
                continue
            bare = [json.loads(x[2]) for x in runs[bare_of[Setup(speaker=None, margin=s.margin, readings=s.readings)]]]
            spk = [without_evidence(json.loads(x[2])) for x in runs[k]]
            documents += len(bare)
            differ += sum(x != y for x, y in zip(bare, spk, strict=False)) + abs(len(bare) - len(spk))
    audio = sum(len(p) for p in streams) / 2 / RATE

    def ms(v: Sequence[float], q: float) -> float:
        return float(np.percentile(v, q)) * 1000

    rows = {
        k: dict(
            rtf=sum(a["t"]) / audio,
            blocks=len(a["t"]),
            p50=ms(a["t"], 50),
            p95=ms(a["t"], 95),
            p99=ms(a["t"], 99),
            max=max(a["t"]) * 1000,
            closing_p50=ms(a["closing"], 50),
            closing_max=max(a["closing"]) * 1000,
            kib_per_block=float(np.mean(a["size"])) / 1024,
            parse_p50=ms(a["parse"], 50),
            parse_p99=ms(a["parse"], 99),
        )
        for k, a in acc.items()
    }
    return dict(audio_seconds=audio, rows=rows, documents=documents, differ=differ)


# --- scoring --------------------------------------------------------------------------------------


def eer(targets: Sequence[float], nontargets: Sequence[float]) -> float:
    if not targets or not nontargets:
        return float("nan")
    t, n = np.sort(np.asarray(targets)), np.sort(np.asarray(nontargets))
    thr = np.unique(np.concatenate([t, n]))
    frr = np.searchsorted(t, thr, side="left") / len(t)
    far = 1.0 - np.searchsorted(n, thr, side="left") / len(n)
    i = int(np.argmin(np.abs(frr - far)))
    return float((frr[i] + far[i]) / 2)


def draw_tables(
    probes: Sequence[Probe], names: Sequence[str], sizes: Sequence[int], rng: random.Random
) -> dict[int, list[list[list[str]]]]:
    """For each table size, per probe, the other speakers of each draw."""
    return {
        k: [[rng.sample([n for n in names if n != p.speaker], k - 1) for _ in range(TABLE_DRAWS)] for p in probes]
        for k in sizes
    }


def score(
    probes: Sequence[Probe],
    engine: str,
    profiles: Mapping[str, Vec],
    tables: Mapping[int, list[list[list[str]]]],
    bins: Sequence[tuple[int, int, str]] = (),
    threshold: float | None = None,
) -> Json:
    """threshold, when given, is the gate's in place of the one these probes' own scores set."""
    # A speaker the engine could not enrol has no evidence of its own and is no contender for
    # anyone else's.
    names = sorted(profiles)
    idx = {n: i for i, n in enumerate(names)}
    mat = np.stack([unit(profiles[n]) for n in names])
    rows: list[Vec | None] = []
    targets: list[float] = []
    nontargets: list[float] = []
    for p in probes:
        v = p.vectors.get(engine)
        if v is None or p.speaker not in idx:
            rows.append(None)
            continue
        s: Vec = mat @ unit(v)
        rows.append(s)
        targets.append(float(s[idx[p.speaker]]))
        nontargets.extend(float(x) for j, x in enumerate(s) if j != idx[p.speaker])
    thr = threshold if threshold is not None else own_threshold(nontargets)

    def top1(i: int, k: int) -> float:
        s = rows[i]
        if s is None:
            return 0.0
        own = s[idx[probes[i].speaker]]
        rivals = [[s[idx[o]] for o in draw if o in idx] for draw in tables[k][i]]
        return float(np.mean([own > max(r, default=-np.inf) for r in rivals]))

    def accepted(i: int) -> bool:
        s = rows[i]
        return s is not None and float(s[idx[probes[i].speaker]]) > thr

    def summary(which: Sequence[int]) -> Json:
        n = len(which)
        return dict(
            probes=n,
            no_evidence=(sum(rows[i] is None for i in which) / n) if n else float("nan"),
            top1={str(k): (sum(top1(i, k) for i in which) / n) if n else float("nan") for k in tables},
            accepted=(sum(accepted(i) for i in which) / n) if n else float("nan"),
        )

    out = summary(range(len(probes)))
    out.update(eer=eer(targets, nontargets), threshold=thr)
    out["bins"] = {
        label: summary([i for i, p in enumerate(probes) if lo <= p.length_ms < hi]) for lo, hi, label in bins
    }
    return out


def own_threshold(nontargets: Sequence[float]) -> float:
    return float(np.quantile(nontargets, 1.0 - FALSE_ACCEPT)) if nontargets else float("inf")


def after_end(
    probes: Sequence[Probe], engine: str, words: Mapping[str, Json], profiles: Mapping[str, Vec], threshold: float
) -> Json:
    """By time after each probe word's end: the share of words whose evidence has shown on some
    result, and the share whose final evidence clears the gate and has shown by then."""
    shown: list[float] = []
    passed: list[float] = []
    for p in probes:
        w = words.get(p.key, {})
        first, fin = w.get("first_any"), w.get("final")
        if not first or p.speaker not in profiles:
            continue
        end = fin["end_sample"] if fin else first["end_sample"]
        lag = (first["fed"] - end) * 1000.0 / RATE
        shown.append(lag)
        v = p.vectors.get(engine)
        if v is not None and float(unit(profiles[p.speaker]) @ unit(v)) > threshold:
            passed.append(lag)
    n = len(probes)
    return dict(
        shown={str(t): sum(x <= t for x in shown) / n for t in AFTER_END_MS},
        accepted={str(t): sum(x <= t for x in passed) / n for t in AFTER_END_MS},
        lag_ms=dict(
            median=float(np.median(shown)) if shown else float("nan"),
            p90=float(np.quantile(shown, 0.9)) if shown else float("nan"),
        ),
    )


def profile_of(vectors: Sequence[Vec | None]) -> Vec | None:
    got = [unit(v) for v in vectors if v is not None]
    return unit(np.mean(np.stack(got), axis=0)) if got else None


# --- the run --------------------------------------------------------------------------------------


def run(a: argparse.Namespace) -> Json:
    data, models = Path(a.data), Path(a.speaker_models)
    spk_dir = models / SPK_DIR
    grammar = sc.DATASET_WORDS + sc.LETTERS + sc.NATO + sc.COLOURS
    rng = random.Random(a.seed)
    speakers = speakers_of(data, a.min_clips)
    names = sorted(speakers)
    if a.limit_speakers:
        names = sorted(rng.sample(names, min(a.limit_speakers, len(names))))
    enroll: dict[str, list[Clip]] = {}
    probe_clips: dict[str, list[Clip]] = {}
    for s in names:
        keys = speakers[s][:]
        rng.shuffle(keys)
        enroll[s] = [Clip(k, s, sc.read_pcm(data / k)) for k in keys[: a.enroll]]
        probe_clips[s] = [Clip(k, s, sc.read_pcm(data / k)) for k in keys[a.enroll : a.enroll + a.probes]]
    sc.note(f"{len(names)} speakers, {a.enroll} enrollment and up to {a.probes} probe clips each")

    engines = (
        [UTTER, UTTER_MARGIN, UTTER_COLD, UTTER_TN, UTTER_TN_MARGIN, UTTER_TN_EMBED, VOSK]
        + ([KALDI] if a.kaldi else [])
        + [TITANET, CAMPP]
    )
    wheel = Wheel(a.model, spk_dir, grammar)
    sherpa = {e: Sherpa(models / f) for e, f in SHERPA_FILES.items()}
    kaldi = KaldiXvector(Path(a.kaldi), spk_dir) if a.kaldi else None
    u = UtterLib(a.lib)
    u.open(a.model, spk_dir, Path(a.titanet))

    # The wheel decodes every clip: its vector, and the span the offline engines are given.
    clip_vecs: dict[str, dict[str, Vec | None]] = {e: {} for e in engines}
    all_clips = [c for s in names for c in enroll[s] + probe_clips[s]]
    for c in all_clips:
        r = wheel.decode(c.pcm)
        c.span = Wheel.span(r, len(c.pcm) // 2)
        clip_vecs[VOSK][c.key] = Wheel.vector(r)
    sc.note(f"wheel decoded {len(all_clips)} clips")
    spans = {c.key: c.pcm[c.span[0] * 2 : c.span[1] * 2] for c in all_clips if c.span}
    for e, sh in sherpa.items():
        for k, pcm in spans.items():
            clip_vecs[e][k] = sh.embed(pcm)
    for k, pcm in spans.items():
        clip_vecs[UTTER_TN_EMBED][k] = u.embed(pcm)
    if kaldi:
        clip_vecs[KALDI].update(kaldi.embed_all(spans))
    sc.note("offline engines embedded the spans")

    # WeSpeaker scores CAM++ after subtracting a mean embedding; the enrollment set's stands in
    # for its training set's. The other engines are scored as they come.
    centre: dict[str, Vec] = {}
    enrolled = [v for s in names for c in enroll[s] if (v := clip_vecs[CAMPP].get(c.key)) is not None]
    if enrolled:
        centre[CAMPP] = np.mean(np.stack(enrolled), axis=0)
        clip_vecs[CAMPP] = {k: None if v is None else v - centre[CAMPP] for k, v in clip_vecs[CAMPP].items()}

    # Profiles: one vector per enrollment clip, per utter word.
    profiles: dict[str, dict[str, Vec]] = {e: {} for e in engines}
    for s in names:
        for e in engines:
            if e == UTTER_COLD:
                continue
            vecs = (
                utter_enroll(u, grammar, enroll[s], a.block_ms, SETUPS[e])
                if e in SETUPS
                else [clip_vecs[e].get(c.key) for c in enroll[s] if c.span]
            )
            p = profile_of(vecs)
            if p is not None:
                profiles[e][s] = p
    # A host that builds a recognizer per grammar change still enrolls from whole recordings.
    profiles[UTTER_COLD] = dict(profiles[UTTER])
    profile_counts = {e: len(profiles[e]) for e in engines}
    sc.note(f"profiles per engine: {profile_counts}")
    # The wheel's half-second floor leaves some speakers with no profile at all; they stay, as no
    # evidence for the wheel, rather than being dropped in its favour.
    kept = [s for s in names if all(s in profiles[e] for e in engines if e != VOSK)]
    dropped = len(names) - len(kept)
    profiles = {e: {s: profiles[e][s] for s in kept if s in profiles[e]} for e in engines}
    sc.note(f"profiles for {len(kept)} speakers under every engine but the wheel, {dropped} dropped")

    # Games: two speakers taking turns, one continuous stream through utter.
    order = kept[:]
    rng.shuffle(order)
    games = [order[i : i + 2] for i in range(0, len(order) - 1, 2)]
    if len(order) % 2 and games:
        games[-1].append(order[-1])
    game_words: dict[str, dict[str, Json]] = {e: {} for e in SETUPS}
    streams: list[bytes] = []
    for g in games:
        lists = [probe_clips[s] for s in g]
        turns = [x[i] for i in range(max(len(x) for x in lists)) for x in lists if i < len(x)]
        for e, setup in SETUPS.items():
            game_words[e].update(utter_game(u, grammar, turns, a.block_ms, setup))
        streams.append(b"".join(c.pcm for c in turns))
    utter_words = game_words[UTTER]
    sc.note(f"{len(games)} games through utter")
    latency = latency_pass(u, wheel, grammar, streams, a.block_ms)
    sc.note("latency pass decoded")

    probes: list[Probe] = []
    no_span = 0
    for s in kept:
        for c in probe_clips[s]:
            if not c.span:
                no_span += 1
                continue
            p = Probe(c.key, s, c.pcm, (c.span[1] - c.span[0]) * 1000.0 / RATE)
            for e in engines:
                if e in SETUPS:
                    w = game_words[e].get(c.key, {}).get("final")
                    p.vectors[e] = w["vector"] if w else None
                elif e == UTTER_COLD:
                    p.vectors[e] = utter_cold(u, grammar, c, a.block_ms)
                else:
                    p.vectors[e] = clip_vecs[e].get(c.key)
            probes.append(p)
    tables = draw_tables(probes, kept, TABLE_SIZES, rng)
    word = {e: score(probes, e, profiles[e], tables, WORD_BINS) for e in engines}
    after = {
        e: after_end(probes, e, game_words[e], profiles[e], word[e]["threshold"])
        for e in (UTTER, UTTER_TN, UTTER_TN_MARGIN)
    }

    # Utter's partials: when a probe word's evidence first appears, and whether it names the
    # speaker then as the final does.
    firsts = [Probe(p.key, p.speaker, b"", p.length_ms) for p in probes]
    lags: list[float] = []
    for fp in firsts:
        w = utter_words.get(fp.key, {})
        first, fin = w.get("first"), w.get("final")
        fp.vectors[UTTER] = first["vector"] if first else None
        if first:
            end = fin["end_sample"] if fin else first["end_sample"]
            lags.append((first["fed"] - end) * 1000.0 / RATE)
    at_first = score(firsts, UTTER, profiles[UTTER], tables)
    partials = dict(
        words=len(probes),
        with_partial_evidence=len(lags),
        lag_ms=dict(
            median=float(np.median(lags)) if lags else float("nan"),
            p90=float(np.quantile(lags, 0.9)) if lags else float("nan"),
        ),
        top1_2_first=at_first["top1"]["2"],
        top1_2_final=word[UTTER]["top1"]["2"],
    )

    # Evidence as speech accumulates: each speaker's probe words joined, cut at each length.
    curve_engines = [e for e in engines if e not in (UTTER_COLD, UTTER_TN, UTTER_TN_MARGIN)]
    curve: dict[str, dict[str, Json]] = {e: {} for e in curve_engines}
    joined: dict[int, list[Probe]] = {n: [] for n in LENGTHS_MS}
    for s in kept:
        pieces = [c.pcm[c.span[0] * 2 : c.span[1] * 2] for c in probe_clips[s] if c.span]
        for j in range(STREAMS_PER_SPEAKER):
            rng.shuffle(pieces)
            stream = b"".join(pieces)
            ut = utter_accumulating(u, grammar, stream, a.block_ms, LENGTHS_MS)
            um = utter_accumulating(u, grammar, stream, a.block_ms, LENGTHS_MS, SETUPS[UTTER_MARGIN])
            for n in LENGTHS_MS:
                if len(stream) // 2 < n * RATE // 1000:
                    continue
                p = Probe(f"{s}/{j}/{n}", s, stream[: n * RATE // 1000 * 2], float(n))
                p.vectors[UTTER] = ut[n]
                p.vectors[UTTER_MARGIN] = um[n]
                joined[n].append(p)
    for n, items in joined.items():
        for p in items:
            p.vectors[VOSK] = Wheel.vector(wheel.decode(p.pcm))
            p.vectors[UTTER_TN_EMBED] = u.embed(p.pcm)
            for e, sh in sherpa.items():
                v = sh.embed(p.pcm)
                p.vectors[e] = v - centre[e] if v is not None and e in centre else v
        if kaldi:
            cut = kaldi.embed_all({p.key: p.pcm for p in items})
            for p in items:
                p.vectors[KALDI] = cut[p.key]
        t = draw_tables(items, kept, (4,), rng)
        for e in curve_engines:
            curve[e][str(n)] = score(items, e, profiles[e], t)
    sc.note("accumulation curve scored")

    both = [
        float(unit(p.vectors[UTTER]) @ unit(p.vectors[VOSK]))
        for p in probes
        if p.vectors.get(UTTER) is not None and p.vectors.get(VOSK) is not None
    ]
    kaldi_both = [
        float(unit(p.vectors[KALDI]) @ unit(p.vectors[VOSK]))
        for p in probes
        if p.vectors.get(KALDI) is not None and p.vectors.get(VOSK) is not None
    ]
    speech_s = sum(len(v) for v in spans.values()) / 2 / RATE
    cost = {e: sh.seconds / sh.audio for e, sh in sherpa.items() if sh.audio}
    if u.audio:
        cost[UTTER_TN_EMBED] = u.seconds / u.audio
    if kaldi and kaldi.audio:
        cost[KALDI] = kaldi.seconds / kaldi.audio
    return dict(
        speakers=len(names),
        kept=len(kept),
        dropped=dropped,
        profiles_per_engine=profile_counts,
        probes=len(probes),
        probes_without_span=no_span,
        games=len(games),
        engines=engines,
        word=word,
        after_end=after,
        partials=partials,
        curve=curve,
        agreement=dict(
            utter_vosk=dict(n=len(both), median=float(np.median(both)) if both else float("nan")),
            kaldi_vosk=dict(n=len(kaldi_both), median=float(np.median(kaldi_both)) if kaldi_both else float("nan")),
        ),
        cost=dict(
            seconds_per_speech_second=cost,
            speech_seconds=speech_s,
        ),
        latency=latency,
        utter=dict(version=u.version, revision=u.revision),
    )


# --- the page -------------------------------------------------------------------------------------


def header(report: Json, a: argparse.Namespace) -> str:
    prov = report["provenance"]
    engines = ", ".join(f"{k} {v}" for k, v in prov["engines"].items())
    ut = report["result"]["utter"]
    return (
        f"Run {prov['date']}, utter {ut['version']} from {ut['revision']} through its C ABI "
        f"(checkout {prov['utter_revision']}), {engines}, model {prov['model']}, speaker models "
        f"{SPK_DIR}, {SHERPA_FILES[TITANET]} and {SHERPA_FILES[CAMPP]}, {prov['cpu']}, "
        f"Python {prov['python']}, blocks of {a.block_ms} ms."
    )


def page(report: Json, a: argparse.Namespace) -> list[str]:
    r = report["result"]
    engines: list[str] = r["engines"]
    L = [
        "# Speaker evidence on short commands",
        "",
        header(report, a),
        "",
        "A host that gates commands per word needs to know, word by word, whose voice it is, and "
        "whether the evidence is there when the word is. This page measures that on the test "
        f"split of Speech Commands: {r['kept']} speakers with at least {a.min_clips} clips "
        f"({r['dropped']} more had no profile under some engine but the wheel), each enrolled "
        f"from {a.enroll} clips, with {r['probes']} probe words scored against every profile "
        f"(probe clips whose final had no word are left out: {r['probes_without_span']}). The "
        f"wheel's half-second floor left {r['kept'] - r['profiles_per_engine'][VOSK]} of the "
        "speakers with no profile at all: their words count as no evidence for the wheel, and "
        "they are no rival to anyone else's.",
        "",
        "## Engines",
        "",
        "| engine | what runs | audio | speech it is given |",
        "|---|---|---|---|",
        f"| {UTTER} | utter's speaker evidence through its C ABI, continuous streams | 8 kHz | each "
        "word entry's own span, at least a quarter second |",
        f"| {UTTER_MARGIN} | the same with a floor margin of {MARGIN_DB:g} dB set, enrollment included "
        "| 8 kHz | the span's frames more than the margin over the floor |",
        f"| {UTTER_COLD} | the same, each probe clip through a recognizer built for it, as a host "
        "that builds one at every grammar change does; enrolled as the first utter row | 8 kHz | "
        "the word's span, its normalisation seeing only the clip |",
        f"| {UTTER_TN} | utter's TitaNet-small through the same recognizer, embedding each closed word "
        "between decoder advances | 16 kHz | each word entry's own span, 25 to 120 frames |",
        f"| {UTTER_TN_MARGIN} | the same with a floor margin of {MARGIN_DB:g} dB set, enrollment included "
        "| 16 kHz | the span as reported |",
        f"| {UTTER_TN_EMBED} | utter's stateless embed() through its C ABI, no recognizer | 16 kHz | "
        "the wheel's word span |",
        f"| {VOSK} | the vosk wheel with {SPK_DIR} set, one clip per recognizer | 8 kHz | the frames "
        "its decode puts on speech phones, at least half a second |",
    ]
    if KALDI in engines:
        L.append(
            f"| {KALDI} | the same network through Kaldi's binaries, centred mean over the span | "
            "8 kHz | the wheel's word span, at least a quarter second |"
        )
    L += [
        f"| {TITANET} | NVIDIA NeMo TitaNet small through sherpa-onnx | 16 kHz | the wheel's word span |",
        f"| {CAMPP} | WeSpeaker CAM++ (VoxCeleb) through sherpa-onnx, less the enrollment set's mean "
        "embedding, as WeSpeaker scores it | 16 kHz | the wheel's word span |",
        "",
        "*Top-1 of K* is how often the probe's own profile scores highest among K enrolled "
        f"speakers drawn at random ({TABLE_DRAWS} draws per probe, the same draws for every "
        "engine). *Accepted at 1% false accept* is the share of probes whose own-profile score "
        "clears the threshold that lets 1% of other-profile scores through. A probe with no "
        "evidence is neither identified nor accepted. The equal error rate is over probes with "
        "evidence only.",
        "",
        "## One word",
        "",
        "| engine | probes | no evidence | top-1 of 2 | top-1 of 4 | equal error rate | accepted at 1% false accept |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in engines:
        w = r["word"][e]
        L.append(
            f"| {e} | {w['probes']} | {pct(w['no_evidence'])} | {pct(w['top1']['2'])} | "
            f"{pct(w['top1']['4'])} | {pct(w['eer'])} | {pct(w['accepted'])} |"
        )
    labels = [label for _, _, label in WORD_BINS]
    counts = [r["word"][engines[0]]["bins"][label]["probes"] for label in labels]
    L += [
        "",
        "## By word length",
        "",
        "The length is the wheel's word span. Each cell is accepted at 1% false accept, then top-1 of 2.",
        "",
        "| engine | " + " | ".join(f"{label} ({n})" for label, n in zip(labels, counts, strict=True)) + " |",
        "|---|" + "---|" * len(labels),
    ]
    for e in engines:
        cells = [
            f"{pct(r['word'][e]['bins'][label]['accepted'])}, {pct(r['word'][e]['bins'][label]['top1']['2'])}"
            for label in labels
        ]
        L.append(f"| {e} | " + " | ".join(cells) + " |")
    lengths = [str(n) for n in LENGTHS_MS if str(n) in r["curve"][engines[0]]]
    L += [
        "",
        "## As speech accumulates",
        "",
        f"Each speaker's probe words joined into {STREAMS_PER_SPEAKER} streams in shuffled orders "
        "and cut at each length of speech. The offline engines embed the cut whole; utter's is "
        "its own partial's evidence at the first partial that has pooled that much speech. "
        "Top-1 of 4, then accepted at 1% false accept at that length's own threshold.",
        "",
        "| engine | " + " | ".join(f"{n} ms" for n in lengths) + " |",
        "|---|" + "---|" * len(lengths),
    ]
    for e in engines:
        if e not in r["curve"]:
            continue
        cells = [f"{pct(r['curve'][e][n]['top1']['4'])}, {pct(r['curve'][e][n]['accepted'])}" for n in lengths]
        L.append(f"| {e} | " + " | ".join(cells) + " |")
    ends = [str(t) for t in AFTER_END_MS]
    L += [
        "",
        "## After a word's end",
        "",
        "When a probe word's evidence shows on the game streams, counted in audio fed after the "
        "final's end of the word, on any partial or final. Each cell is the share of probe words "
        "whose evidence has shown by then, then the share whose final evidence clears the "
        "row's gate at 1% false accept and has shown by then.",
        "",
        "| engine | median, 90th percentile | " + " | ".join(f"{t} ms" for t in ends) + " |",
        "|---|---|" + "---|" * len(ends),
    ]
    for e, x in r["after_end"].items():
        cells = [f"{pct(x['shown'][t])}, {pct(x['accepted'][t])}" for t in ends]
        L.append(f"| {e} | {x['lag_ms']['median']:.0f}, {x['lag_ms']['p90']:.0f} ms | " + " | ".join(cells) + " |")
    pa = r["partials"]
    ag = r["agreement"]
    co = r["cost"]
    L += [
        "",
        "## On utter's partials",
        "",
        f"Of {pa['words']} probe words, {pa['with_partial_evidence']} carried evidence on a "
        "partial before their final. It first appeared a median "
        f"{pa['lag_ms']['median']:.0f} ms after the word's end (90th percentile "
        f"{pa['lag_ms']['p90']:.0f} ms), counting the audio fed, with the network's 70 ms of right "
        f"context and the block included. Top-1 of 2 on that first evidence: "
        f"{pct(pa['top1_2_first'])}; on the final's: {pct(pa['top1_2_final'])}.",
        "",
        "## Agreement",
        "",
        f"utter's word vector against the wheel's clip vector: median cosine "
        f"{ag['utter_vosk']['median']:.3f} over {ag['utter_vosk']['n']} probes both score. utter "
        "normalises by a mean that looks back over the stream and pools the word's own span; the "
        "wheel normalises by a centred mean over the frames it keeps.",
    ]
    if KALDI in engines:
        L.append(
            f"The Kaldi row against the wheel: median cosine {ag['kaldi_vosk']['median']:.3f} over "
            f"{ag['kaldi_vosk']['n']}."
        )
    la = r["latency"]
    L += [
        "",
        "## Latency and compute",
        "",
        f"The game streams again, {la['audio_seconds']:.0f} s of audio, through each engine and setup "
        f"in rotating order from one stream to the next, blocks of {a.block_ms} ms with words on "
        "partials. A block's compute runs from the call that takes its audio to the return "
        "of the partial or final read after it; a closing block is one that returned a final. "
        "Parsing is the host's json.loads of that result in Python.",
        "",
        "| engine | real-time factor | per block, ms p50 / p95 / p99 / max | closing blocks, ms p50 / "
        "max | result text per block | parsing, ms p50 / p99 |",
        "|---|---|---|---|---|---|",
    ]
    for k, x in la["rows"].items():
        L.append(
            f"| {k} | {x['rtf']:.4f} | {x['p50']:.3f} / {x['p95']:.2f} / {x['p99']:.2f} / {x['max']:.1f} "
            f"| {x['closing_p50']:.2f} / {x['closing_max']:.1f} | {x['kib_per_block']:.2f} KiB "
            f"| {x['parse_p50']:.3f} / {x['parse_p99']:.3f} |"
        )
    moved = ": the evidence moves no word, partial or final to a later block." if la["differ"] == 0 else "."
    L += [
        "",
        f"With a speaker model set, {la['differ'] or 'none'} of utter's {la['documents']} results "
        "differ from the same setup's results without one once the four speaker keys are removed"
        f"{moved}",
        "",
        "Compute per second of speech embedded by the offline engines: "
        + ", ".join(f"{e} {v * 1000:.1f} ms" for e, v in co["seconds_per_speech_second"].items())
        + f" (the Kaldi figure is a batch through three processes). The {UTTER_TN_EMBED} row "
        f"and every in-crate TitaNet row run {Path(a.titanet).name} as scripts/titanet_convert.py "
        "writes it.",
        "",
        "## Caveats",
        "",
        "Speech Commands speakers recorded on their own devices, so a profile carries the "
        "microphone and the room as well as the voice, which flatters every engine against a "
        "table where everyone speaks into one microphone. A same-microphone set such as VCTK "
        "(CC BY 4.0) is the harder check and is not run here. Enrollment here is ten single "
        "words; a host that enrolls from minutes of speech has more to average.",
        "",
    ]
    reading = Path(a.out).with_suffix(".reading.md")
    if reading.exists():
        L += ["", reading.read_text()]
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "Repos/utter-bench-data/speech_commands_v0.02"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--speaker-models", default=str(Path.home() / "Repos/utter-bench-data/speaker-models"))
    ap.add_argument(
        "--titanet",
        default=str(Path.home() / "Repos/utter-bench-data/release-0.0.6" / TITANET_DIR),
        help="TitaNet-small as scripts/titanet_convert.py writes it",
    )
    ap.add_argument("--kaldi", default=None, help="a built Kaldi src directory; the Kaldi row is skipped without it")
    ap.add_argument("--lib", default="target/release/libutter.so")
    ap.add_argument("--out", default="docs/benchmarks/speaker-evidence")
    ap.add_argument("--min-clips", type=int, default=20)
    ap.add_argument("--enroll", type=int, default=10)
    ap.add_argument("--probes", type=int, default=10)
    ap.add_argument("--limit-speakers", type=int, default=0)
    ap.add_argument("--block-ms", type=int, default=40)
    ap.add_argument("--seed", type=int, default=14)
    a = ap.parse_args()
    t = time.monotonic()
    result = run(a)
    report: Json = dict(
        provenance=sc.provenance(a, ["vosk", "sherpa-onnx", "numpy"]),
        args=vars(a),
        seconds=time.monotonic() - t,
        result=result,
    )
    out = Path(a.out)
    out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    text = sc.write_page(out.with_suffix(".md"), page(report, a))
    print(text)
    sc.note(f"wrote {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
