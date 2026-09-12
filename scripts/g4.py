#!/usr/bin/env python3
"""Gate G4: partial alternatives.

    python scripts/g4.py --oracle ORACLE.json --hyp STREAM.jsonl

From a `stream --alternatives N --partial-words` run and the g2 oracle on the same corpus: rank 0
equals the partial on every block; a best path without a word is offered as [sil] at rank 0;
blocks whose libvosk partial was empty never yield a vocabulary word at rank 0; and the contest
census: how many partials carrying a word have a rival one word away.
"""
import argparse
import json
from pathlib import Path


def words_of(text):
    return [w for w in text.split() if not (w.startswith("[") and w.endswith("]"))]


def one_word_apart(a, b):
    if a == b:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    if abs(len(a) - len(b)) != 1:
        return False
    s, l = (a, b) if len(a) < len(b) else (b, a)
    for i in range(len(l)):
        if l[:i] + l[i + 1:] == s:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ref = json.loads(Path(args.oracle).read_text())
    blocks = rank0_eq = 0
    empty_best = empty_as_sil = 0
    vosk_empty = vosk_empty_utter_word = 0
    with_word = contest = 0
    alt_counts = {}
    sil_entries = 0
    for line in Path(args.hyp).read_text().splitlines():
        if not line.strip():
            continue
        h = json.loads(line)
        r = ref.get(h["take"])
        for i, p in enumerate(h["partials"]):
            if p is None:
                continue
            alts = p.get("partial_alternatives")
            if alts is None:
                continue
            blocks += 1
            alt_counts[len(alts)] = alt_counts.get(len(alts), 0) + 1
            rank0 = alts[0]["text"] if alts else None
            if rank0 == p["partial"]:
                rank0_eq += 1
            if not words_of(p["partial"]):
                empty_best += 1
                if rank0 == "[sil]":
                    empty_as_sil += 1
            for e in p.get("partial_result", []):
                if e["word"] == "[sil]":
                    sil_entries += 1
            if r is not None and i < len(r["partials"]) and r["partials"][i] is not None:
                if not r["partials"][i].get("partial", ""):
                    vosk_empty += 1
                    if words_of(p["partial"]):
                        vosk_empty_utter_word += 1
            w0 = words_of(p["partial"])
            if w0:
                with_word += 1
                if len(alts) > 1 and one_word_apart(w0, words_of(alts[1]["text"])):
                    contest += 1
    pct = lambda a, b: 100.0 * a / b if b else float("nan")
    lines = ["# G4: partial alternatives", ""]
    lines.append(f"- rank 0 equals the partial on {rank0_eq} / {blocks} blocks ({pct(rank0_eq, blocks):.2f}%) [asks every block]")
    lines.append(f"- best path without a word on {empty_best} blocks, offered as [sil] at rank 0 on {empty_as_sil} ({pct(empty_as_sil, empty_best):.2f}%)")
    lines.append(f"- blocks whose libvosk partial was empty: {vosk_empty}; rank 0 a vocabulary word on {vosk_empty_utter_word} ({pct(vosk_empty_utter_word, vosk_empty):.3f}%) [asks 0]")
    lines.append(f"- contest census: {contest} / {with_word} partials with a word have a rival one word away ({pct(contest, with_word):.2f}%)")
    lines.append(f"- [sil] entries in partial word lists: {sil_entries}")
    lines.append("- alternatives per block: " + ", ".join(f"{k}: {v}" for k, v in sorted(alt_counts.items())))
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
