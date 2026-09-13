#!/usr/bin/env python3
"""Diff two runs of scripts/speech_commands.py at the clip.

    python scripts/diff_clips.py OLD.clips.jsonl NEW.clips.jsonl

Totals move by a handful of clips between runs and say nothing about which clips moved, or
whether a change fixed one set and broke another of the same size. This reports, per engine,
the clips that went right-to-wrong and wrong-to-right, and what each engine read instead.
"""

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def load(path: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["clip"]] = row
    return rows


def engines_of(rows: Mapping[str, Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    for row in rows.values():
        for k, v in row.items():
            if k not in ("clip", "label") and isinstance(v, dict) and k not in names:
                names.append(k)
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--show", type=int, default=12, help="clips to list per direction")
    args = ap.parse_args()
    old, new = load(args.old), load(args.new)

    shared = [c for c in new if c in old]
    print(f"{len(old)} clips in {Path(args.old).name}, {len(new)} in {Path(args.new).name}, {len(shared)} in both")
    only_old = len(old) - len(shared)
    only_new = len(new) - len(shared)
    if only_old or only_new:
        print(f"  {only_old} only in the old run, {only_new} only in the new one; compared on the rest")

    for name in engines_of(new):
        broke: list[tuple[str, list[str]]] = []
        fixed: list[tuple[str, list[str]]] = []
        reread = 0
        per_word: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
        for clip in shared:
            a, b = old[clip].get(name), new[clip].get(name)
            if a is None or b is None:
                continue
            if a["words"] != b["words"]:
                reread += 1
            if a["correct"] and not b["correct"]:
                broke.append((old[clip]["label"], b["words"]))
                per_word[old[clip]["label"]][0] += 1
            elif b["correct"] and not a["correct"]:
                fixed.append((old[clip]["label"], a["words"]))
                per_word[old[clip]["label"]][1] += 1
        ok_a = sum(1 for c in shared if old[c].get(name, {}).get("correct"))
        ok_b = sum(1 for c in shared if new[c].get(name, {}).get("correct"))
        print(f"\n## {name}: {ok_a} -> {ok_b} correct ({ok_b - ok_a:+d}), {reread} clips read differently")
        print(f"   {len(broke)} went wrong, {len(fixed)} came right")
        for title, items in (("went wrong", broke), ("came right", fixed)):
            if not items:
                continue
            print(f"   {title}:")
            for label, words in items[: args.show]:
                print(f"     {label} -> {' '.join(words) or '(nothing)'}")
            if len(items) > args.show:
                print(f"     ... and {len(items) - args.show} more")
        moved = {w: v for w, v in per_word.items() if v[0] != v[1]}
        if moved:
            print("   by word (wrong/right):", ", ".join(f"{w} {v[0]}/{v[1]}" for w, v in sorted(moved.items())))


if __name__ == "__main__":
    main()
