#!/usr/bin/env python3
"""Gate G0 oracle and scorer.

  oracle: run the stock vosk wheel over a corpus of 16 kHz mono WAV takes with a grammar,
          feeding fixed blocks, and record every final's words per take.
  score:  word error rate of utter's batch decodes (g0 JSON lines) against the oracle.

Paths to the model, the corpus and the grammar are arguments; nothing here names a consumer.
"""
import argparse
import json
import os
import sys
import tempfile
import wave
from pathlib import Path


def dither0_model(model_dir):
    """A sibling model directory whose mfcc.conf adds --dither=0; everything else is linked."""
    tmp = Path(tempfile.mkdtemp(prefix="utter-model-"))
    for entry in Path(model_dir).iterdir():
        if entry.name == "conf":
            (tmp / "conf").mkdir()
            for f in entry.iterdir():
                text = f.read_text()
                if f.name == "mfcc.conf":
                    text = text.rstrip("\n") + "\n--dither=0\n"
                (tmp / "conf" / f.name).write_text(text)
        else:
            os.symlink(entry.resolve(), tmp / entry.name)
    return tmp


def oracle(args):
    import vosk
    vosk.SetLogLevel(-1)
    model_dir = dither0_model(args.model) if args.dither0 else Path(args.model)
    model = vosk.Model(str(model_dir))
    grammar = json.loads(Path(args.grammar).read_text())
    block = 16000 * args.block_ms // 1000 * 2
    out = {}
    for wav_path in sorted(Path(args.corpus).glob("*.wav")):
        w = wave.open(str(wav_path))
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2), wav_path
        pcm = w.readframes(w.getnframes())
        rec = vosk.KaldiRecognizer(model, 16000, json.dumps(grammar))
        rec.SetWords(True)
        segments = []
        fed = 0
        for i in range(0, len(pcm), block):
            chunk = pcm[i:i + block]
            fed += len(chunk) // 2
            if rec.AcceptWaveform(chunk):
                res = json.loads(rec.Result())
                segments.append({"end_sample": fed, "text": res.get("text", ""),
                                 "result": res.get("result", [])})
        res = json.loads(rec.FinalResult())
        segments.append({"end_sample": fed, "text": res.get("text", ""), "result": res.get("result", [])})
        words = [w for s in segments for w in s["text"].split()]
        out[wav_path.stem] = {"words": words, "segments": segments, "samples": fed}
        print(wav_path.stem, len(words), " ".join(words), file=sys.stderr)
    Path(args.out).write_text(json.dumps(out, indent=1))


def edit_distance(ref, hyp):
    d = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(hyp) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev = cur
    return d[len(hyp)]


def score(args):
    ref = json.loads(Path(args.oracle).read_text())
    rows = [json.loads(l) for l in Path(args.hyp).read_text().splitlines() if l.strip()]
    modes = sorted({r["mode"] for r in rows})
    lines = ["# G0: batch decode against libvosk finals", ""]
    lines.append("| take | ref words | " + " | ".join(f"{m} err" for m in modes) + " |")
    lines.append("|---|---|" + "---|" * len(modes))
    totals = {m: [0, 0] for m in modes}
    by_take = {}
    for r in rows:
        by_take.setdefault(r["take"], {})[r["mode"]] = r
    exact = {m: 0 for m in modes}
    for take in sorted(by_take):
        if take not in ref:
            continue
        ref_words = ref[take]["words"]
        cells = []
        for m in modes:
            r = by_take[take].get(m)
            if r is None:
                cells.append("-")
                continue
            e = edit_distance(ref_words, r["words"])
            totals[m][0] += e
            totals[m][1] += len(ref_words)
            exact[m] += int(e == 0)
            cells.append(str(e))
        lines.append(f"| {take} | {len(ref_words)} | " + " | ".join(cells) + " |")
    lines.append("")
    for m in modes:
        e, n = totals[m]
        wer = 100.0 * e / n if n else float("nan")
        lines.append(f"- {m}: WER {wer:.2f}% ({e} errors / {n} reference words), {exact[m]} takes exact")
    if "zero" in totals and "faithful" in totals and totals["zero"][1]:
        gap = 100.0 * (totals["zero"][0] - totals["faithful"][0]) / totals["zero"][1]
        lines.append(f"- zero minus faithful: {gap:+.2f} WER points")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("oracle")
    o.add_argument("--model", required=True)
    o.add_argument("--corpus", required=True)
    o.add_argument("--grammar", required=True, help="JSON array of strings")
    o.add_argument("--out", required=True)
    o.add_argument("--block-ms", type=int, default=40)
    o.add_argument("--dither0", action="store_true")
    o.set_defaults(fn=oracle)
    s = sub.add_parser("score")
    s.add_argument("--oracle", required=True)
    s.add_argument("--hyp", required=True, help="g0 JSON lines")
    s.add_argument("--out", default=None)
    s.set_defaults(fn=score)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
