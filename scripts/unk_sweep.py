#!/usr/bin/env python3
"""Silence phantoms against the unknown-word cost.

    python scripts/unk_sweep.py --corpus DIR RUN.jsonl...

For each `stream` run (one per unknown-word cost, `--partial-words` on): blocks whose last 400 ms
of audio sit below -40 dBFS and whose rank-0 partial carries a vocabulary word; final words whose
interval sits below -40 dBFS; total final words; total [unk] finals.
"""
import argparse
import json
import math
import wave
from pathlib import Path

RATE = 16000
QUIET_DBFS = -40.0
TAIL_MS = 400


def words_of(text):
    return [w for w in text.split() if not (w.startswith("[") and w.endswith("]"))]


def dbfs(samples):
    if not samples:
        return -999.0
    acc = 0
    for s in samples:
        acc += s * s
    rms = math.sqrt(acc / len(samples))
    return 20 * math.log10(rms / 32768.0) if rms > 0 else -999.0


def load_pcm(path):
    import array
    w = wave.open(str(path))
    a = array.array("h")
    a.frombytes(w.readframes(w.getnframes()))
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--label", action="append", default=[], help="label per run, in order")
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    pcm_cache = {}
    print("| run | quiet blocks | rank 0 a word on quiet block | final words | quiet final words | [unk] finals |")
    print("|---|---|---|---|---|---|")
    for idx, run in enumerate(args.runs):
        label = args.label[idx] if idx < len(args.label) else Path(run).stem
        quiet_blocks = quiet_word = final_words = quiet_finals = unk_finals = 0
        for line in Path(run).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            take = r["take"]
            if take not in pcm_cache:
                pcm_cache[take] = load_pcm(Path(args.corpus) / f"{take}.wav")
            pcm = pcm_cache[take]
            block = RATE * r["block_ms"] // 1000
            tail = RATE * TAIL_MS // 1000
            for i, p in enumerate(r["partials"]):
                if p is None:
                    continue
                end = (i + 1) * block
                if dbfs(pcm[max(0, end - tail):end]) < QUIET_DBFS:
                    quiet_blocks += 1
                    if words_of(p.get("partial", "")):
                        quiet_word += 1
            for s in r["segments"]:
                res = s["result"]
                for e in res.get("result", []):
                    if e["word"] == "[unk]":
                        unk_finals += 1
                        continue
                    if e["word"].startswith("["):
                        continue
                    final_words += 1
                    if dbfs(pcm[e["start_sample"]:e["end_sample"]]) < QUIET_DBFS:
                        quiet_finals += 1
        print(f"| {label} | {quiet_blocks} | {quiet_word} | {final_words} | {quiet_finals} | {unk_finals} |")


if __name__ == "__main__":
    main()
