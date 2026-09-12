#!/usr/bin/env python3
"""Silence words against the unknown-word cost.

    python scripts/unk_sweep.py --corpus DIR [--margin DB] RUN.jsonl...

One row per `stream` run (one per unknown-word cost, `--partial-words` on, so the runtime reports
the floor and the energy under every word). A block is quiet when the RMS of its last 400 ms of
audio, read from the WAV, is within the margin of the floor the partial carries; a quiet block
whose rank-0 partial holds a word is split by what that last word is doing there. Final words are
counted, and among them those within the margin of the floor, those spanning more than two
seconds, and the [unk] readings.
"""

import argparse
import json
import math
import wave
from pathlib import Path

RATE = 16000
TAIL_MS = 400
LONG_S = 2.0


def words_of(text):
    return [w for w in text.split() if not (w.startswith("[") and w.endswith("]"))]


def spoken_entries(entries):
    return [e for e in entries if not (e["word"].startswith("[") and e["word"].endswith("]"))]


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

    with wave.open(str(path)) as w:
        a = array.array("h")
        a.frombytes(w.readframes(w.getnframes()))
    return a


# [[rr:TD-8#Measurements count a word whose own span carries no speech]]
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--margin", type=float, default=8.0, help="dB above the floor that is still silence")
    ap.add_argument("--label", action="append", default=[], help="label per run, in order")
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    pcm_cache = {}
    print(
        "| run | quiet blocks | rank 0 a word on a quiet block | held | straddling | silence | "
        "final words | silence final words | final words over 2 s | [unk] finals |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for idx, run in enumerate(args.runs):
        label = args.label[idx] if idx < len(args.label) else Path(run).stem
        quiet_blocks = word_blocks = held = straddling = silence_blocks = 0
        finals = silence_finals = long_finals = unk_finals = 0
        floors = 0
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
                # [[rr:TD-8#The runtime reports the floor]]
                if p is None or "floor_dbfs" not in p:
                    continue
                floors += 1
                quiet = p["floor_dbfs"] + args.margin
                end = (i + 1) * block
                if dbfs(pcm[max(0, end - tail) : end]) > quiet:
                    continue
                quiet_blocks += 1
                if not words_of(p.get("partial", "")):
                    continue
                word_blocks += 1
                if "partial_result" not in p:
                    raise SystemExit(f"{run}: partials carry no word list; rerun with --partial-words")
                said = spoken_entries(p["partial_result"])
                if not any(e["end_sample"] > end - tail for e in said):
                    held += 1
                elif said[-1]["energy_dbfs"] > quiet:
                    straddling += 1
                else:
                    silence_blocks += 1
            for s in r["segments"]:
                res = s["result"]
                if not res.get("result"):
                    continue
                quiet = res["floor_dbfs"] + args.margin
                for e in res["result"]:
                    if e["word"] == "[unk]":
                        unk_finals += 1
                        continue
                    if e["word"].startswith("["):
                        continue
                    finals += 1
                    if e["energy_dbfs"] <= quiet:
                        silence_finals += 1
                    if e["end"] - e["start"] > LONG_S:
                        long_finals += 1
        if not floors:
            raise SystemExit(f"{run}: no floor_dbfs on any partial; rerun with a stream that reports it")
        print(
            f"| {label} | {quiet_blocks} | {word_blocks} | {held} | {straddling} | {silence_blocks} | "
            f"{finals} | {silence_finals} | {long_finals} | {unk_finals} |"
        )


if __name__ == "__main__":
    main()
