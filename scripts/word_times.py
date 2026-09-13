#!/usr/bin/env python3
"""Public benchmark on Google Speech Commands v2 (CC BY 4.0): the word times on finals, utter
(through utterpy) against the stock vosk wheel, same blocks, same grammar.

    python scripts/word_times.py --data DIR --model MODEL_DIR --out docs/benchmarks/word-times

Both engines decode every clip of the split under the Speech Commands page's 92-entry grammar
with `SetWords(True)`, and each word of each final is taken with its `start` and `end`. On the
clips where the two finals carry the same word sequence, every word's start and end are paired
and the difference is reported: exact, within one output frame of 30 ms, within two, and the
distribution. Clips where the sequences differ are counted and left out, because there is no
word to pair; the Speech Commands page measures that disagreement.

The frame grid is checked too: a time that is not a multiple of 30 ms would mean one engine
is not reporting Kaldi's frame times.

Writes `<out>.md`, `<out>.json`, and `<out>.clips.jsonl`, a line per clip with each engine's
words and times. `<out>.reading.md`, if present, is appended to the page.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from speech_commands import (  # noqa: E402
    COLOURS,
    DATASET_WORDS,
    LETTERS,
    NATO,
    Engine,
    feed,
    note,
    pct,
    provenance,
    provenance_line,
    quantile,
    read_pcm,
    write_page,
)

FRAME_S = 0.03
EPS = 1e-6


def timed_words(finals):
    """(word, start, end) for every word of every final, in order."""
    out = []
    for f in finals:
        for e in f.get("result") or []:
            w = e["word"]
            if w.startswith("[") and w.endswith("]"):
                continue
            out.append((w, float(e["start"]), float(e["end"])))
    return out


def on_grid(t):
    k = round(t / FRAME_S)
    return abs(t - k * FRAME_S) < EPS


def bucket(deltas_ms):
    """Counts of |delta| exact, within one frame, within two, and beyond."""
    frame_ms = FRAME_S * 1000
    b = {"exact": 0, "one_frame": 0, "two_frames": 0, "beyond": 0}
    for d in deltas_ms:
        a = abs(d)
        if a < EPS * 1000:
            b["exact"] += 1
        elif a <= frame_ms + EPS:
            b["one_frame"] += 1
        elif a <= 2 * frame_ms + EPS:
            b["two_frames"] += 1
        else:
            b["beyond"] += 1
    return b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="output prefix; writes .md, .json and .clips.jsonl")
    ap.add_argument("--split", default="testing", choices=["testing", "validation"])
    ap.add_argument("--limit", type=int, default=0, help="clips per word, 0 for all")
    ap.add_argument("--block-ms", type=int, default=40)
    ap.add_argument("--engines", default="vosk,utterpy", help="two engines; the first is the reference")
    args = ap.parse_args()

    names = [n for n in args.engines.split(",") if n.strip()]
    if len(names) != 2:
        ap.error("--engines takes exactly two engines")
    data = Path(args.data)
    listed = [line.strip() for line in (data / f"{args.split}_list.txt").read_text().splitlines() if line.strip()]
    by_word = defaultdict(list)
    for rel in listed:
        by_word[rel.split("/")[0]].append(rel)
    clips = []
    for w in DATASET_WORDS:
        items = by_word.get(w, [])
        if args.limit:
            items = items[: args.limit]
        clips.extend((w, data / rel) for rel in items)

    modules = {}
    for name in names:
        modules[name] = __import__(name.split("@")[0])
        if name == "vosk":
            modules[name].SetLogLevel(-1)
    grammar = DATASET_WORDS + LETTERS + NATO + COLOURS
    prov = provenance(args, modules)
    engines = {n: Engine(n, modules[n], args.model, grammar) for n in names}
    ref, other = names

    rows = []
    off_grid = {n: 0 for n in names}
    words_seen = {n: 0 for n in names}
    starts, ends = [], []
    same_sequence = 0
    differ = 0
    empty_both = 0
    for i, (label, clip) in enumerate(clips):
        pcm = read_pcm(clip)
        row = {"clip": str(clip), "label": label}
        got = {}
        for n, eng in engines.items():
            got[n] = timed_words(feed(eng.new(), pcm, args.block_ms))
            words_seen[n] += len(got[n])
            off_grid[n] += sum(1 for _, s, e in got[n] if not (on_grid(s) and on_grid(e)))
            row[n] = [{"word": w, "start": s, "end": e} for w, s, e in got[n]]
        rows.append(row)
        a, b = got[ref], got[other]
        if [w for w, _, _ in a] != [w for w, _, _ in b]:
            differ += 1
            continue
        if not a:
            empty_both += 1
            continue
        same_sequence += 1
        for (_, sa, ea), (_, sb, eb) in zip(a, b):
            starts.append((sb - sa) * 1000.0)
            ends.append((eb - ea) * 1000.0)
        if (i + 1) % 1000 == 0:
            note(f"{i + 1} / {len(clips)} clips")

    paired = len(starts)
    both = [max(abs(s), abs(e)) for s, e in zip(starts, ends)]
    report = {
        "split": args.split,
        "clips": len(clips),
        "block_ms": args.block_ms,
        "grammar_size": len(grammar),
        "frame_ms": FRAME_S * 1000,
        "reference": ref,
        "compared": other,
        "provenance": prov,
        "words": words_seen,
        "off_grid": off_grid,
        "same_sequence": same_sequence,
        "differ": differ,
        "empty_both": empty_both,
        "paired_words": paired,
        "start": bucket(starts),
        "end": bucket(ends),
        "word": bucket(both),
        "abs_ms": {
            k: {q: quantile([abs(x) for x in v], float(q)) for q in ("0.5", "0.9", "0.99")}
            | {"max": max((abs(x) for x in v), default=float("nan"))}
            for k, v in (("start", starts), ("end", ends))
        },
        "signed_ms": {
            k: {"earlier": sum(1 for x in v if x < -EPS), "later": sum(1 for x in v if x > EPS)}
            for k, v in (("start", starts), ("end", ends))
        },
    }

    with open(args.out + ".clips.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    lines = [
        f"# Word times on finals, Speech Commands v2, {prov['model']}, {args.split} split, {len(clips)} clips, {args.block_ms} ms blocks",
        "",
        f"Grammar: the Speech Commands page's {len(grammar)} entries. Both engines with `SetWords(True)`; each word of each final with its `start` and `end` in seconds. Times are paired on the clips where both finals carry the same word sequence; {other} is measured against {ref}.",
        "",
        provenance_line(prov),
        "",
        "## Clips",
        "",
        "| clips | same word sequence | sequences differ | both empty | words paired |",
        "|---|---|---|---|---|",
        f"| {len(clips)} | {same_sequence} | {differ} | {empty_both} | {paired} |",
        "",
        "## Frame grid",
        "",
        "A word time that is not a multiple of 30 ms is not one of Kaldi's output frame times.",
        "",
        "| engine | words | off the 30 ms grid |",
        "|---|---|---|",
    ]
    for n in names:
        lines.append(f"| {n} | {words_seen[n]} | {off_grid[n]} |")
    lines += [
        "",
        "## Agreement",
        "",
        f"Each paired word's {other} time against its {ref} time. A word counts as within a frame when both its start and its end are.",
        "",
        "| time | exact | within one frame (30 ms) | within two frames | beyond |",
        "|---|---|---|---|---|",
    ]
    for k in ("start", "end", "word"):
        b = report[k]
        cum1 = b["exact"] + b["one_frame"]
        cum2 = cum1 + b["two_frames"]
        lines.append(
            f"| {k} | {b['exact']} ({pct(b['exact'], paired):.2f}%) | {cum1} ({pct(cum1, paired):.2f}%) | {cum2} ({pct(cum2, paired):.2f}%) | {b['beyond']} ({pct(b['beyond'], paired):.2f}%) |"
        )
    lines += [
        "",
        f"| time | abs difference p50 / p90 / p99 / max, ms | {other} earlier | {other} later |",
        "|---|---|---|---|",
    ]
    for k in ("start", "end"):
        q = report["abs_ms"][k]
        s = report["signed_ms"][k]
        lines.append(
            f"| {k} | {q['0.5']:.0f} / {q['0.9']:.0f} / {q['0.99']:.0f} / {q['max']:.0f} | {s['earlier']} | {s['later']} |"
        )
    lines.append("")
    reading = Path(args.out + ".reading.md")
    if reading.exists():
        lines.append(reading.read_text().strip())
        lines.append("")
    text = write_page(args.out + ".md", lines)
    Path(args.out + ".json").write_text(json.dumps(report, indent=1))
    note(f"written: {args.out}.md, .json, .clips.jsonl")
    print(text)


if __name__ == "__main__":
    main()
