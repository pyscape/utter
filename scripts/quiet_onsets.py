"""Quiet onsets: what an endpoint bound does to a word whose first frames sit near the floor.

    python scripts/quiet_onsets.py --data DIR --model DIR [--out docs/benchmarks/quiet-onsets]

The states page's pause streams are built again with every word's onset held a few dB over the
gap floor and rising to the word's own level, and decoded by each engine in turn: the stock
rules, the bound vetoed at 8 nats, and the same bound with a floor margin. The public words sit
15 dB and more over the room floor, so their first frames clear any margin at once; a
microphone's do not, and a rule that reads a forming onset as hiss shows here as a final cut
into the word, the word lost, decoded twice or recovered late, and grammar words gained from its
first frames. A few minutes for the default sweep; the states page takes an hour.
"""

import argparse
import array
import json
import math
import random
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import partial_states as ps  # noqa: E402
import speech_commands as sc  # noqa: E402

Json = dict[str, Any]
RATE = sc.RATE
LATE_MS = 500.0


def quiet_onset(pcm: bytes, target_dbfs: float, hold_ms: float, ramp_ms: float) -> bytes:
    """The clip with its onset brought down to `target_dbfs`: from where the energy span starts
    the gain holds for `hold_ms` at the ratio that puts the span's mean RMS at the target, then
    rises to unity over `ramp_ms`, and is unity after. The body of the word keeps its level."""
    a = array.array("h")
    a.frombytes(pcm)
    start = ps.energy_start(pcm)
    end = sc.energy_end(pcm)
    lo = int((start or 0.0) * RATE)
    hi = int((end if end is not None else len(a) / RATE) * RATE)
    seg = a[lo:hi] if hi > lo else a
    rms = math.sqrt(sum(float(v) * v for v in seg) / len(seg)) if len(seg) else 0.0
    if rms <= 0.0:
        return pcm
    g0 = min(1.0, 32768.0 * 10.0 ** (target_dbfs / 20.0) / rms)
    hold = int(hold_ms * ps.SAMPLES_PER_MS)
    ramp = max(1, int(ramp_ms * ps.SAMPLES_PER_MS))
    out = array.array("h", pcm)
    for i in range(lo, min(len(a), lo + hold + ramp)):
        gain = g0 if i < lo + hold else g0 + (1.0 - g0) * (i - lo - hold) / ramp
        out[i] = int(round(a[i] * gain))
    return out.tobytes()


def run_stream_finals(eng: sc.Engine, pcm: bytes, block_ms: int) -> list[Json]:
    """The finals alone, with their word entries, as the states page scores them."""
    rec = eng.new()
    block = RATE * block_ms // 1000 * 2
    finals: list[Json] = []
    fed = 0

    def keep(f: Json, endpoint: str | None = None) -> None:
        finals.append(
            dict(fed=fed, text=f.get("text", ""), result=f.get("result", []), endpoint=endpoint or f.get("endpoint"))
        )

    for i in range(0, len(pcm), block):
        chunk = pcm[i : i + block]
        fed += len(chunk) // 2
        if rec.AcceptWaveform(chunk):
            keep(json.loads(rec.Result()))
    keep(json.loads(rec.FinalResult()), "flush")
    return finals


def score(words: Sequence[Json], gap_rows: Sequence[Json], finals: Sequence[Json], samples: int) -> Counter[str]:
    """The states page's finals scoring plus what a cut onset leaves: a spoken word covered only
    by another (misread), a final word covering no spoken one (gained), a covering entry that
    starts more than LATE_MS after the onset (late), and a final landing inside a word (cut)."""
    c: Counter[str] = Counter()
    r = ps.runtime_finals([dict(words=list(words), gaps=list(gap_rows), finals=list(finals), samples=samples)])
    for k in ("words", "lost", "restarted", "finals"):
        c[k] += r[k]
    c["pauses"] += r["totals"]["pause"]
    c["pauses_ended"] += r["pauses_mistaken"]
    c["finishes"] += r["totals"]["finish"]
    c["finishes_called"] += r["finishes_called"]
    c["floor_finals"] += sum(1 for f in finals if f.get("endpoint") == "floor")
    spans = [
        (e["start"] * RATE, e["end"] * RATE, e["word"]) for f in finals for e in f["result"] if sc.words_of(e["word"])
    ]
    for lo, hi, _ in spans:
        c["gained"] += not any(lo < t["offset"] and hi > t["onset"] for t in words)
    for t in words:
        cov = [(lo, w) for lo, hi, w in spans if lo < t["offset"] and hi > t["onset"]]
        c["right"] += len(cov) == 1 and cov[0][1] == t["label"]
        c["misread"] += bool(cov) and t["label"] not in [w for _, w in cov]
        c["late"] += any(w == t["label"] and lo > t["onset"] + LATE_MS * ps.SAMPLES_PER_MS for lo, w in cov)
        c["cut"] += any(t["onset"] < f["fed"] < t["offset"] for f in finals)
    return c


def build(
    data: Path, gaps_paths: Sequence[Path], a: argparse.Namespace, level: float
) -> list[tuple[bytes, list[Json], list[Json]]]:
    rng = random.Random(a.seed)
    clips = ps.split_clips(data, "testing", rng)
    gaps = ps.GapSource(gaps_paths, ps.GAP_FLOOR_DBFS)
    lo_ms, hi_ms = (float(x) for x in a.pauses.split("-"))
    gain = partial(quiet_onset, target_dbfs=ps.GAP_FLOOR_DBFS + level, hold_ms=a.hold_ms, ramp_ms=a.ramp_ms)
    streams: list[tuple[bytes, list[Json], list[Json]]] = []
    n_words = 0
    while n_words < a.words and len(clips) > ps.UTTERANCES_PER_STREAM * ps.WORDS_PER_UTTERANCE[1]:
        built = ps.build_stream(rng, clips, gaps, a.utterances, (lo_ms, hi_ms), word_gain=gain)
        streams.append(built)
        n_words += len(built[1])
    return streams


def sweep(engines: Mapping[str, sc.Engine], data: Path, gaps_paths: Sequence[Path], a: argparse.Namespace) -> Json:
    out: Json = {}
    for level in [float(x) for x in a.levels.split(",") if x.strip()]:
        streams = build(data, gaps_paths, a, level)
        minutes = sum(len(p) // 2 for p, _, _ in streams) / RATE / 60.0
        sc.note(f"onsets at floor + {level:g} dB: {sum(len(w) for _, w, _ in streams)} words, {minutes:.1f} min")
        per: Json = {}
        for name, eng in engines.items():
            c: Counter[str] = Counter()
            for pcm, words, gap_rows in streams:
                c.update(score(words, gap_rows, run_stream_finals(eng, pcm, a.block_ms), len(pcm) // 2))
            per[name] = dict(c)
        out[f"{level:g}"] = dict(level=level, streams=len(streams), minutes=minutes, per_engine=per)
    return out


def page(report: Json, a: argparse.Namespace) -> list[str]:
    q = report["sweep"]
    L = [
        "# Quiet onsets: what a bound does to a word it can barely see",
        "",
        sc.provenance_line(report["provenance"]),
        "",
        "The [states page](partial-states.md) measures the endpoint bound on words that sit "
        "15 dB and more over the gap floor, so their first frames clear any floor margin at once. "
        "A microphone's do not. Here its pause streams are built again with every word's onset "
        f"held a few dB over the {ps.GAP_FLOOR_DBFS:g} dBFS gap floor for its first {a.hold_ms:g} ms "
        f"and rising to the word's own level over the next {a.ramp_ms:g}, pauses of {a.pauses} ms "
        "between words, and decoded by each engine in turn. The bound rows are spelled as the "
        "states page spells them, `MS/NATS` and, after a second slash, the floor margin in dB "
        "(`[[rr:TD-12#Decision outcome]]`).",
        "",
        "*Lost* is a spoken word no final's entry covers and *decoded twice* one that two cover, "
        "as on the states page. *Misread* is a spoken word covered only by another; *gained* a "
        f"final word covering no spoken one; *late* a covering entry starting more than {LATE_MS:g} ms "
        "after the onset; *cut* a final landing between a word's onset and its offset. A rule that "
        "reads a forming onset as hiss shows in the last four.",
        "",
        "| onset | engine | words | right | lost | decoded twice | misread | gained | late | cut | "
        "floor finals | pauses ended | finishes called |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for d in q.values():
        for name, c in d["per_engine"].items():
            L.append(
                f"| floor + {d['level']:g} dB | {name} | {c['words']} | {ps.share(c['right'], c['words'])} | "
                f"{c['lost']} | {c['restarted']} | {c['misread']} | {c['gained']} | {c['late']} | {c['cut']} | "
                f"{c['floor_finals']} | {ps.share(c['pauses_ended'], c['pauses'])} | "
                f"{ps.share(c['finishes_called'], c['finishes'])} |"
            )
    L.append("")
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="docs/benchmarks/quiet-onsets")
    ap.add_argument("--engines", default="utterpy,utterpy@300/8,utterpy@300/8/8")
    ap.add_argument("--levels", default="3,6,12", help="dB over the gap floor the onsets are held at")
    ap.add_argument("--hold-ms", type=float, default=200.0)
    ap.add_argument("--ramp-ms", type=float, default=200.0)
    ap.add_argument("--pauses", default="400-800", help="ms between words, uniform over the range")
    ap.add_argument("--words", type=int, default=600, help="words per level")
    ap.add_argument("--utterances", type=int, default=ps.UTTERANCES_PER_STREAM)
    ap.add_argument("--block-ms", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260914)
    a = ap.parse_args()

    import utterpy

    data = Path(a.data)
    gaps_paths = sorted((data / "_background_noise_").glob("*.wav"))
    grammar = sc.DATASET_WORDS + sc.LETTERS + sc.NATO + sc.COLOURS
    names = [x for x in a.engines.split(",") if x.strip()]
    engines = {n: sc.Engine(n, utterpy, a.model, grammar) for n in names}
    report: Json = dict(
        provenance=sc.provenance(a, ["utterpy"]),
        args=vars(a),
        grammar_size=len(grammar),
        sweep=sweep(engines, data, gaps_paths, a),
    )
    out = Path(a.out)
    out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    sc.write_page(out.with_suffix(".md"), page(report, a))
    sc.note(f"wrote {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
