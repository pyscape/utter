#!/usr/bin/env python3
"""The acoustic model's certainty and the sound outside words, on public audio: whether a word's
certainty tells a spoken word from one decoded out of a room, whether it holds across input level
and grammar, and whether the sound evidence tells steady broadband noise, a low tone and a short
impulse apart. Google Speech Commands v2 (CC BY 4.0) and its background recordings.

    python scripts/sound_rating.py --model MODEL_DIR [--data DIR] [--lib target/release/libutter.so] \\
        [--out docs/benchmarks/sound-rating]

MODEL_DIR is the stock small English model; utter runs through its C ABI under the Speech
Commands page's grammar. Three passes:

- words: every test clip through a recognizer of its own, and every background recording streamed
  whole at three floor levels, where nothing is said and every final word is decoded out of the
  room. Each final word's `certainty` against its `energy_dbfs` over the final's `floor_dbfs`, the
  gate a host already has: how well each separates spoken words from room words, and the room
  words each rejects while keeping 95% of the spoken ones.
- level and grammar: a sample of clips again, each read once at the end so the final's window is
  every frame decoded, at lower gains and under smaller grammars; how far the clip's certainty
  moves from its own at 0 dB under the page's grammar.
- sounds: a stream in a room of pink noise at a -60 dBFS floor with sounds laid in: white and
  pink noise, tones of 50 to 120 Hz, a click, a 5 ms burst and the loudest 20 ms of the dishes
  recording, and two seconds of the other recordings, each at 10, 20 and 30 dB over the floor.
  Partials are read after every block, as a host reads them, and each result's window is scored
  against what was laid under it: the evidence's features by sound, and how an example rule over
  them, the page's and not the runtime's, sorts the windows.

Writes `<out>.md`, `<out>.json` and `<out>.clips.jsonl`; what a run means goes in
`<out>.reading.md`, which the page appends to itself.
"""

import argparse
import ctypes
import json
import random
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent))
import speech_commands as sc  # noqa: E402

Json = dict[str, Any]
type Samples = NDArray[np.float64]
RATE = sc.RATE
PHANTOM_FLOORS = (-60.0, -50.0, -40.0)
KEEP_SPOKEN = 0.95
GAINS_DB = (0.0, -10.0, -20.0, -30.0)
ROOM_FLOOR_DBFS = -60.0
ROOM_SECONDS = 10.0
EVENT_SECONDS = 2.0
GAP_SECONDS = 3.0
LEVELS_DB = (10.0, 20.0, 30.0)
# Feature frames of 10 ms: an impulse's level is its loudest frame's.
FRAME = RATE // 100
BROADBAND, TONE, IMPULSE, ROOM, OTHER = "steady broadband", "low tone", "impulse", "room", "other"
CLASSES = (BROADBAND, TONE, IMPULSE, ROOM)
# The example rule's thresholds, in dB and ms.
RULE_IMPULSE_MS, RULE_IMPULSE_DB, RULE_TONE_DB, RULE_BROAD_DB = 40.0, 10.0, 10.0, 5.0


# --- utter ----------------------------------------------------------------------------------------


class Utter:
    """utter's C ABI through ctypes."""

    def __init__(self, path: str, model_dir: str) -> None:
        lib = ctypes.CDLL(path)
        vp, cp, ci, cf, u64 = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_float, ctypes.c_uint64
        for name, res, args in [
            ("utter_version", cp, []),
            ("utter_revision", cp, []),
            ("utter_model_new", vp, [cp]),
            ("utter_recognizer_new_grm", vp, [vp, cf, cp]),
            ("utter_recognizer_set_words", None, [vp, ci]),
            ("utter_recognizer_set_partial_words", None, [vp, ci]),
            ("utter_recognizer_accept_waveform_s", ci, [vp, cp, ci]),
            ("utter_recognizer_partial_result", cp, [vp]),
            ("utter_recognizer_result", cp, [vp]),
            ("utter_recognizer_final_result", cp, [vp]),
            ("utter_recognizer_decoded_sample", u64, [vp]),
            ("utter_recognizer_free", None, [vp]),
        ]:
            fn = getattr(lib, name)
            fn.restype = res
            fn.argtypes = args
        self.lib = lib
        self.version = str(lib.utter_version().decode())
        self.revision = str(lib.utter_revision().decode())
        self.model = lib.utter_model_new(model_dir.encode())
        if not self.model:
            raise SystemExit(f"utter could not open {model_dir}")

    def run(self, grammar: Sequence[str], pcm: bytes, block_ms: int, partials: bool) -> list[tuple[str, int, Json]]:
        """Every result of one stream, as (kind, decoded sample, result): a partial after each
        block that closed nothing when `partials`, each final an endpoint closed, and the flush."""
        lib = self.lib
        h = lib.utter_recognizer_new_grm(self.model, float(RATE), json.dumps(list(grammar)).encode())
        lib.utter_recognizer_set_words(h, 1)
        lib.utter_recognizer_set_partial_words(h, 1)
        out: list[tuple[str, int, Json]] = []
        n = RATE * block_ms // 1000 * 2
        for i in range(0, len(pcm), n):
            b = pcm[i : i + n]
            if lib.utter_recognizer_accept_waveform_s(h, b, len(b) // 2) == 1:
                r = lib.utter_recognizer_result(h)
                out.append(("final", lib.utter_recognizer_decoded_sample(h), json.loads(r)))
            elif partials:
                r = lib.utter_recognizer_partial_result(h)
                out.append(("partial", lib.utter_recognizer_decoded_sample(h), json.loads(r)))
        r = lib.utter_recognizer_final_result(h)
        out.append(("final", lib.utter_recognizer_decoded_sample(h), json.loads(r)))
        lib.utter_recognizer_free(h)
        return out


def final_words(results: Sequence[tuple[str, int, Json]]) -> list[Json]:
    """Every word entry of every final, each with the final's floor beside it."""
    out = []
    for kind, _, r in results:
        if kind != "final":
            continue
        for w in r.get("result", []):
            out.append(dict(w, floor_dbfs=r.get("floor_dbfs")))
    return out


def window_certainty(results: Sequence[tuple[str, int, Json]]) -> float | None:
    """The mean certainty over every frame the results' windows cover."""
    total = frames = 0.0
    for _, _, r in results:
        for c, n in (("certainty_words", "words_frames"), ("certainty_outside", "outside_frames")):
            if r.get(n):
                total += float(r[c]) * r[n]
                frames += r[n]
    return total / frames if frames else None


# --- audio ----------------------------------------------------------------------------------------


def samples(pcm: bytes) -> Samples:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float64)


def pcm16(x: Samples) -> bytes:
    """Saturates rather than wraps, as an ADC would."""
    return bytes(np.clip(np.round(x), -32768, 32767).astype(np.int16).tobytes())


def dbfs_rms(level_db: float) -> float:
    return float(32768.0 * 10.0 ** (level_db / 20.0))


def rms(x: Samples) -> float:
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0


def loudest_frame_rms(x: Samples) -> float:
    n = len(x) // FRAME * FRAME
    if n == 0:
        return rms(x)
    return float(np.sqrt(np.max(np.mean(x[:n].reshape(-1, FRAME) ** 2, axis=1))))


def auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """The chance a spoken word scores above a room word, ties counted half."""
    if not pos or not neg:
        return float("nan")
    n = np.sort(np.asarray(neg, dtype=np.float64))
    p = np.asarray(pos, dtype=np.float64)
    below = np.searchsorted(n, p, side="left")
    ties = np.searchsorted(n, p, side="right") - below
    return float(np.mean((below + 0.5 * ties) / len(n)))


def q(v: Sequence[float], p: float) -> float:
    return float(np.quantile(np.asarray(v, dtype=np.float64), p)) if len(v) else float("nan")


# --- the passes -----------------------------------------------------------------------------------


def test_clips(data: Path) -> list[tuple[str, Path]]:
    return [(k.split("/")[0], data / k) for k in sorted((data / "testing_list.txt").read_text().split())]


def words_pass(
    u: Utter, a: argparse.Namespace, grammar: Sequence[str], clips: Sequence[tuple[str, Path]], rooms: Sequence[Path]
) -> tuple[Json, list[Json]]:
    rows: list[Json] = []
    spoken: dict[str, list[Json]] = {"right": [], "misread": []}
    for i, (label, path) in enumerate(clips):
        words = final_words(u.run(grammar, sc.read_pcm(path), a.block_ms, partials=False))
        for w in words:
            spoken["right" if w["word"] == label else "misread"].append(w)
        rows.append(dict(clip=sc.recorded_path(path, 2), label=label, words=[word_row(w) for w in words]))
        if i % 2000 == 0:
            sc.note(f"words: clip {i} of {len(clips)}")
    phantoms: list[Json] = []
    by_room: dict[str, Json] = {}
    for floor in PHANTOM_FLOORS:
        for p in rooms:
            pcm = sc.scaled_to_floor(sc.read_pcm(p), floor)
            words = final_words(u.run(grammar, pcm, a.block_ms, partials=False))
            for w in words:
                w["room"] = p.stem
                w["room_floor"] = floor
            phantoms += words
            by_room[f"{p.stem} at {floor:g} dBFS"] = dict(
                seconds=len(pcm) / 2 / RATE, words=len(words), certainty=[w.get("certainty") for w in words]
            )
            rows.append(dict(room=p.stem, floor_dbfs=floor, words=[word_row(w) for w in words]))
        sc.note(f"words: rooms at {floor:g} dBFS")

    def cert(ws: Sequence[Json]) -> list[float]:
        return [w["certainty"] for w in ws if w.get("certainty") is not None]

    def margin(ws: Sequence[Json]) -> list[float]:
        return [
            w["energy_dbfs"] - w["floor_dbfs"]
            for w in ws
            if w.get("energy_dbfs") is not None and w.get("floor_dbfs") is not None
        ]

    said = spoken["right"] + spoken["misread"]
    features: Mapping[str, Callable[[Sequence[Json]], list[float]]] = {"certainty": cert, "energy over floor": margin}
    sep: Json = {}
    for name, f in features.items():
        pos, neg = f(spoken["right"]), f(phantoms)
        t = q(pos, 1 - KEEP_SPOKEN)
        sep[name] = dict(
            auc_right=auc(pos, neg),
            auc_said=auc(f(said), neg),
            threshold=t,
            spoken_kept=float(np.mean(np.asarray(pos) >= t)) if pos else float("nan"),
            room_rejected=float(np.mean(np.asarray(neg) < t)) if neg else float("nan"),
        )
    # Both gates, each at the threshold that keeps 97.5% of the right words alone.
    both_keep = 1 - (1 - KEEP_SPOKEN) / 2
    tc = q(cert(spoken["right"]), 1 - both_keep)
    tm = q(margin(spoken["right"]), 1 - both_keep)

    def passes(w: Json) -> bool:
        c, e, fl = w.get("certainty"), w.get("energy_dbfs"), w.get("floor_dbfs")
        return c is not None and c >= tc and (e is None or fl is None or e - fl >= tm)

    sep["both"] = dict(
        threshold_certainty=tc,
        threshold_margin=tm,
        spoken_kept=float(np.mean([passes(w) for w in spoken["right"]])) if spoken["right"] else float("nan"),
        room_rejected=float(np.mean([not passes(w) for w in phantoms])) if phantoms else float("nan"),
    )
    dist = {
        k: dict(words=len(ws), p10=q(cert(ws), 0.1), p50=q(cert(ws), 0.5), p90=q(cert(ws), 0.9))
        for k, ws in (("spoken, right", spoken["right"]), ("spoken, misread", spoken["misread"]), ("room", phantoms))
    }
    by_floor = {
        f"{floor:g} dBFS": dict(
            words=len(ws := [w for w in phantoms if w["room_floor"] == floor]),
            p50=q(cert(ws), 0.5),
            rejected=float(np.mean(np.asarray(cert(ws)) < sep["certainty"]["threshold"])) if ws else float("nan"),
        )
        for floor in PHANTOM_FLOORS
    }
    rooms_out = {
        k: dict(seconds=v["seconds"], words=v["words"], p50=q([c for c in v["certainty"] if c is not None], 0.5))
        for k, v in by_room.items()
    }
    return dict(clips=len(clips), distribution=dist, separation=sep, by_floor=by_floor, by_room=rooms_out), rows


def word_row(w: Mapping[str, Any]) -> Json:
    return {k: w.get(k) for k in ("word", "start_sample", "end_sample", "certainty", "energy_dbfs", "floor_dbfs")}


def level_pass(u: Utter, a: argparse.Namespace, grammar: Sequence[str], clips: Sequence[tuple[str, Path]]) -> Json:
    grammars = {
        "page grammar": list(grammar),
        "the 35 dataset words": list(sc.DATASET_WORDS),
        "the 10 commands": list(sc.COMMANDS),
    }
    base: list[float | None] = []
    gains: dict[str, list[float | None]] = {f"{g:g} dB": [] for g in GAINS_DB}
    grams: dict[str, list[float | None]] = {k: [] for k in grammars}
    for i, (_, path) in enumerate(clips):
        x = samples(sc.read_pcm(path))
        for g in GAINS_DB:
            c = window_certainty(u.run(grammar, pcm16(x * 10.0 ** (g / 20.0)), a.block_ms, partials=False))
            gains[f"{g:g} dB"].append(c)
            if g == 0.0:
                base.append(c)
        for k, gr in grammars.items():
            grams[k].append(
                base[-1] if k == "page grammar" else window_certainty(u.run(gr, pcm16(x), a.block_ms, partials=False))
            )
        if i % 250 == 0:
            sc.note(f"level and grammar: clip {i} of {len(clips)}")

    def moved(vals: Sequence[float | None]) -> Json:
        r = [v / b - 1.0 for v, b in zip(vals, base, strict=True) if v is not None and b]
        m = [abs(x) for x in r]
        return dict(
            clips=len(r),
            median_certainty=q([v for v in vals if v is not None], 0.5),
            median_change=q(r, 0.5),
            p10_change=q(r, 0.1),
            p90_change=q(r, 0.9),
            p90_abs_change=q(m, 0.9),
        )

    return dict(
        clips=len(clips),
        gains={k: moved(v) for k, v in gains.items()},
        grammars={k: moved(v) for k, v in grams.items()},
    )


@dataclass
class Event:
    name: str
    kind: str
    level_db: float
    # The event's samples in the stream, and for an impulse the sample it peaks at.
    start: int = 0
    end: int = 0
    peak: int | None = None


def sound_stream(data: Path, level_db: float, seed: int) -> tuple[bytes, list[Event]]:
    """Ten seconds of the room, then each sound in turn, EVENT_SECONDS of it and GAP_SECONDS of
    the room after; the room runs on under every sound."""
    bg = data / "_background_noise_"
    rng = random.Random(seed)
    room_src = samples(sc.scaled_to_floor(sc.read_pcm(bg / "pink_noise.wav"), ROOM_FLOOR_DBFS))
    floor_rms = dbfs_rms(ROOM_FLOOR_DBFS)
    target = floor_rms * 10.0 ** (level_db / 20.0)
    ev_n, gap_n = int(EVENT_SECONDS * RATE), int(GAP_SECONDS * RATE)

    def excerpt(name: str, n: int) -> Samples:
        x = samples(sc.read_pcm(bg / f"{name}.wav"))
        o = rng.randrange(len(x) - n)
        return x[o : o + n]

    def steady(x: Samples) -> Samples:
        out: Samples = x * (target / rms(x))
        return out

    def impulse(x: Samples, at: int) -> Samples:
        out = np.zeros(ev_n)
        out[at : at + len(x)] = x * (target / loudest_frame_rms(np.concatenate([x, np.zeros(FRAME)])))
        return out

    t = np.arange(ev_n) / RATE
    dishes = samples(sc.read_pcm(bg / "doing_the_dishes.wav"))
    loud = int(np.argmax(np.abs(dishes)))
    clink = dishes[max(0, loud - RATE // 200) : loud + RATE // 100 + RATE // 200]
    mid = ev_n // 2
    sounds: list[tuple[str, str, Samples, int | None]] = [
        ("white noise", BROADBAND, steady(excerpt("white_noise", ev_n)), None),
        ("pink noise", BROADBAND, steady(excerpt("pink_noise", ev_n)), None),
        *[(f"{hz} Hz tone", TONE, steady(np.sin(2 * np.pi * hz * t)), None) for hz in (50, 60, 100, 120)],
        ("click", IMPULSE, impulse(np.array([1.0]), mid), mid),
        ("5 ms burst", IMPULSE, impulse(samples(pcm16(np.asarray(excerpt("white_noise", RATE // 200)))), mid), mid),
        ("dishes clink", IMPULSE, impulse(clink, mid), mid + (loud - max(0, loud - RATE // 200))),
        ("running tap", OTHER, steady(excerpt("running_tap", ev_n)), None),
        ("exercise bike", OTHER, steady(excerpt("exercise_bike", ev_n)), None),
        ("dishes", OTHER, steady(excerpt("doing_the_dishes", ev_n)), None),
    ]
    total = int(ROOM_SECONDS * RATE) + len(sounds) * (ev_n + gap_n)
    room = np.resize(room_src, total)
    out = room.copy()
    events = []
    at = int(ROOM_SECONDS * RATE)
    for name, kind, x, peak in sounds:
        out[at : at + ev_n] += x
        events.append(Event(name, kind, level_db, at, at + ev_n, None if peak is None else at + peak))
        at += ev_n + gap_n
    return pcm16(out), events


def window_label(w0: int, w1: int, events: Sequence[Event]) -> Event | str | None:
    """What a window from sample w0 to w1 lies over: an impulse it holds, a steady sound it lies
    wholly inside, the room clear of every sound by half a second, or nothing it can be scored as."""
    for e in events:
        if e.peak is not None and w0 <= e.peak < w1:
            return e
        if e.peak is None and e.start <= w0 and w1 <= e.end:
            return e
    settle = RATE // 2
    if w0 >= int(ROOM_SECONDS * RATE // 2) and all(w1 <= e.start or w0 >= e.end + settle for e in events):
        return ROOM
    return None


def features(r: Mapping[str, Any]) -> Json | None:
    band = r.get("band_db")
    if band is None:
        return None
    b = np.asarray(band, dtype=np.float64)
    sd = np.asarray(r["band_sd_db"], dtype=np.float64)
    return dict(
        broad=float(np.median(b)),
        low=float(np.max(b[:8]) - np.median(b[16:])),
        sd=float(np.median(sd)),
        rise_ms=float(r["rise_ms"]),
        rise_db=float(r["rise_db"]) if r.get("rise_db") is not None else float("nan"),
        rise_start=int(r["rise_start_sample"]),
    )


def rule(f: Mapping[str, float]) -> str:
    """The page's example of reading the evidence; the runtime names no sound."""
    if f["rise_ms"] <= RULE_IMPULSE_MS and f["rise_db"] >= RULE_IMPULSE_DB:
        return IMPULSE
    if f["low"] >= RULE_TONE_DB:
        return TONE
    if f["broad"] >= RULE_BROAD_DB:
        return BROADBAND
    return ROOM


def sounds_pass(u: Utter, a: argparse.Namespace, grammar: Sequence[str]) -> Json:
    per_event: dict[str, dict[str, Json]] = {}
    confusion: dict[str, dict[str, int]] = {c: {d: 0 for d in CLASSES} for c in CLASSES}
    onsets: list[float] = []
    for level in LEVELS_DB:
        pcm, events = sound_stream(Path(a.data), level, a.seed)
        results = u.run(grammar, pcm, a.block_ms, partials=True)
        prev = 0
        seen: dict[str, list[Json | None]] = {}
        for _, decoded, r in results:
            w0, w1 = prev, decoded
            prev = decoded
            if w1 <= w0:
                continue
            lab = window_label(w0, w1, events)
            if lab is None:
                continue
            f = features(r)
            name = lab if isinstance(lab, str) else lab.name
            seen.setdefault(name, []).append(f)
            kind = lab if isinstance(lab, str) else lab.kind
            if f is not None and kind in CLASSES:
                confusion[kind][rule(f)] += 1
            if f is not None and isinstance(lab, Event) and lab.peak is not None:
                onsets.append((f["rise_start"] - lab.peak) / RATE * 1000.0)
        for name, fs in seen.items():
            got = [f for f in fs if f is not None]
            row: Json = dict(windows=len(fs), with_evidence=len(got))
            for k in ("broad", "low", "sd", "rise_ms", "rise_db"):
                row[k] = q([f[k] for f in got], 0.5)
            kind = ROOM if name == ROOM else next(e.kind for e in events if e.name == name)
            row["kind"] = kind
            row["rule"] = {c: sum(rule(f) == c for f in got) for c in CLASSES} if got else {c: 0 for c in CLASSES}
            per_event.setdefault(name, {})[f"{level:g} dB"] = row
        sc.note(f"sounds: {level:g} dB over the floor")
    return dict(
        per_event=per_event,
        confusion=confusion,
        onset_ms=dict(
            n=len(onsets),
            p10=q(onsets, 0.1),
            p50=q(onsets, 0.5),
            p90=q(onsets, 0.9),
            within_10=float(np.mean(np.abs(np.asarray(onsets)) <= 10.0)) if onsets else float("nan"),
        ),
    )


def run(a: argparse.Namespace) -> tuple[Json, list[Json]]:
    data = Path(a.data)
    u = Utter(a.lib, a.model)
    grammar = sc.DATASET_WORDS + sc.LETTERS + sc.NATO + sc.COLOURS
    clips = test_clips(data)
    if a.limit:
        clips = clips[: a.limit]
    rooms = sorted((data / "_background_noise_").glob("*.wav"))
    words, rows = words_pass(u, a, grammar, clips, rooms)
    chosen = random.Random(a.seed).sample(clips, min(a.level_clips, len(clips)))
    level = level_pass(u, a, grammar, chosen)
    sounds = sounds_pass(u, a, grammar)
    return dict(utter=dict(version=u.version, revision=u.revision), words=words, level=level, sounds=sounds), rows


# --- the page -------------------------------------------------------------------------------------


def pct(x: float) -> str:
    return "-" if x != x else f"{100.0 * x:.1f}%"


def num(x: float, d: int = 3) -> str:
    return "-" if x != x else f"{x:.{d}f}"


def page(report: Json, a: argparse.Namespace) -> list[str]:
    prov, r = report["provenance"], report["result"]
    ut, w, lv, s = r["utter"], r["words"], r["level"], r["sounds"]
    L = [
        "# Certainty and the sound outside words",
        "",
        f"Run {prov['date']}, utter {ut['version']} from {ut['revision']} through its C ABI "
        f"(checkout {prov['utter_revision']}), model {prov['model']}, {prov['cpu']}, Python "
        f"{prov['python']}, blocks of {a.block_ms} ms, the Speech Commands page's grammar of "
        f"{len(sc.DATASET_WORDS + sc.LETTERS + sc.NATO + sc.COLOURS)} words.",
        "",
        "Every partial and final carries the acoustic model's certainty over the frames decoded "
        "since the previous result, in words and outside them, and what the sound outside words "
        "is like; every word entry carries the certainty over its own span, `[[rr:TD-17]]`. This "
        "page asks whether they do what they are for, on public audio.",
        "",
        "## Spoken words and words decoded out of a room",
        "",
        f"Each of the {w['clips']} test clips through a recognizer of its own, and each of the "
        f"dataset's {len(w['by_room']) // len(PHANTOM_FLOORS)} background recordings streamed whole "
        f"with its floor at {', '.join(f'{f:g}' for f in PHANTOM_FLOORS)} dBFS, where nothing is "
        "said and every final word came out of the room. A word is right when it is the clip's "
        "own word.",
        "",
        "| Words | Count | Certainty p10 | Median | p90 |",
        "|---|---|---|---|---|",
    ]
    for k, d in w["distribution"].items():
        L.append(f"| {k} | {d['words']} | {num(d['p10'])} | {num(d['p50'])} | {num(d['p90'])} |")
    L += [
        "",
        f"How well each figure separates right words from room words: the chance a right word "
        f"scores above a room word (AUC), and the room words it rejects at the threshold that keeps "
        f"{pct(KEEP_SPOKEN)} of the right words. `energy over floor` is the word's `energy_dbfs` less "
        "its final's `floor_dbfs`, the gate a host already has, `[[rr:TD-8#The gate is the host's "
        "and is relative to the floor]]`. `both` takes each at the threshold that keeps 97.5% of "
        "the right words alone.",
        "",
        "| Figure | AUC, right words | AUC, every spoken word | Threshold | Right words kept | Room words rejected |",
        "|---|---|---|---|---|---|",
    ]
    for k in ("certainty", "energy over floor"):
        d = w["separation"][k]
        L.append(
            f"| {k} | {num(d['auc_right'])} | {num(d['auc_said'])} | {num(d['threshold'], 3 if k == 'certainty' else 1)} "
            f"| {pct(d['spoken_kept'])} | {pct(d['room_rejected'])} |"
        )
    b = w["separation"]["both"]
    L += [
        f"| both | | | {num(b['threshold_certainty'])} and {num(b['threshold_margin'], 1)} dB "
        f"| {pct(b['spoken_kept'])} | {pct(b['room_rejected'])} |",
        "",
        "Room words by floor, and the share the certainty's threshold above rejects:",
        "",
        "| Floor | Room words | Median certainty | Rejected |",
        "|---|---|---|---|",
    ]
    for k, d in w["by_floor"].items():
        L.append(f"| {k} | {d['words']} | {num(d['p50'])} | {pct(d['rejected'])} |")
    L += ["", "| Recording | Seconds | Room words | Median certainty |", "|---|---|---|---|"]
    for k, d in w["by_room"].items():
        L.append(f"| {k} | {d['seconds']:.0f} | {d['words']} | {num(d['p50'])} |")
    L += [
        "",
        "## Input level and grammar",
        "",
        f"{lv['clips']} test clips drawn at random, each decoded with no result read until the "
        "end, so the finals' windows cover every frame; the clip's certainty is the mean over "
        "them. Each figure is against the same clip at 0 dB under the page's grammar.",
        "",
        "| Condition | Median certainty | Median change | p10 | p90 | p90 of the size of the change |",
        "|---|---|---|---|---|---|",
    ]
    for group in ("gains", "grammars"):
        for k, d in lv[group].items():
            L.append(
                f"| {k} | {num(d['median_certainty'])} | {pct(d['median_change'])} | {pct(d['p10_change'])} "
                f"| {pct(d['p90_change'])} | {pct(d['p90_abs_change'])} |"
            )
    L += [
        "",
        "## The sound outside words",
        "",
        f"A room of pink noise with its floor at {ROOM_FLOOR_DBFS:g} dBFS: {ROOM_SECONDS:g} s of it, "
        f"then each sound in turn for {EVENT_SECONDS:g} s with {GAP_SECONDS:g} s of the room after, "
        "the room running on under it. A steady sound's level is its RMS over the room's floor; an "
        "impulse's is its loudest 10 ms frame's, laid at the middle of its two seconds. The dishes "
        "clink is the loudest 20 ms of that recording. Partials are read after every block. A "
        "result's window is scored when it lies wholly inside a steady sound, holds an impulse, or "
        "lies in the room half a second clear of every sound.",
        "",
        "The features, medians over the scored windows with evidence: `broad` the median of "
        "`band_db`; `low` the loudest of the lowest eight bands over the median of the upper 24; "
        "`sd` the median of `band_sd_db`; the rise's length and height. A window has evidence when "
        "it holds a frame in no named word and the band floors are known; a sound the decoder "
        "reads as a word has none.",
        "",
        "| Sound | Over floor | Windows | With evidence | broad | low | sd | Rise ms | Rise dB | Example rule says |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, levels in s["per_event"].items():
        for lvl, d in levels.items():
            said = ", ".join(f"{c} {n}" for c, n in d["rule"].items() if n)
            L.append(
                f"| {name} | {lvl if name != ROOM else f'in the {lvl} stream'} | {d['windows']} | {d['with_evidence']} "
                f"| {num(d['broad'], 1)} | {num(d['low'], 1)} | {num(d['sd'], 1)} | {num(d['rise_ms'], 0)} "
                f"| {num(d['rise_db'], 1)} | {said or '-'} |"
            )
    L += [
        "",
        f"The example rule, the page's and not the runtime's: an impulse when the rise lasts at most "
        f"{RULE_IMPULSE_MS:g} ms and stands {RULE_IMPULSE_DB:g} dB over the floor; otherwise a low "
        f"tone when `low` is at least {RULE_TONE_DB:g} dB; otherwise steady broadband when `broad` "
        f"is at least {RULE_BROAD_DB:g} dB; otherwise the room. Over every level, what it says of "
        "each kind of sound's windows with evidence:",
        "",
        "| Laid in | " + " | ".join(CLASSES) + " |",
        "|---|" + "---|" * len(CLASSES),
    ]
    for c, row in s["confusion"].items():
        L.append(f"| {c} | " + " | ".join(str(row[d]) for d in CLASSES) + " |")
    o = s["onset_ms"]
    L += [
        "",
        f"The rise's onset against the impulse's own sample, over {o['n']} windows: median "
        f"{num(o['p50'], 0)} ms, p10 {num(o['p10'], 0)}, p90 {num(o['p90'], 0)}; within 10 ms on "
        f"{pct(o['within_10'])}.",
    ]
    reading = Path(a.out).with_suffix(".reading.md")
    if reading.exists():
        L += ["", reading.read_text()]
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "Repos/utter-bench-data/speech_commands_v0.02"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--lib", default="target/release/libutter.so")
    ap.add_argument("--out", default="docs/benchmarks/sound-rating")
    ap.add_argument("--limit", type=int, default=0, help="test clips for the words pass, 0 for all")
    ap.add_argument("--level-clips", type=int, default=1000)
    ap.add_argument("--block-ms", type=int, default=40)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    t = time.monotonic()
    result, rows = run(a)
    report: Json = dict(
        provenance=sc.provenance(a, ["numpy"]),
        args=sc.recorded_args(vars(a)),
        seconds=time.monotonic() - t,
        result=result,
    )
    out = Path(a.out)
    out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    with open(out.with_suffix(".clips.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    text = sc.write_page(out.with_suffix(".md"), page(report, a))
    print(text)
    sc.note(f"wrote {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
