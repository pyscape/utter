#!/usr/bin/env python3
"""Gates G2, G3 and G5: the stock vosk wheel and utter fed identical blocks.

  oracle: run vosk over a corpus with a grammar, recording the partial after every block and
          every final (Result on endpoint, FinalResult at the end) with word times.
  score:  compare a `stream` JSON-lines run against the oracle: partial text per block, final
          word sequence per segment with the disagreement split the consumer cares about, word
          times, endpoint times, and compute per block.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g0 import dither0_model, edit_distance  # noqa: E402

RATE = 16000


def oracle(args):
    import vosk
    vosk.SetLogLevel(-1)
    model_dir = dither0_model(args.model) if args.dither0 else Path(args.model)
    model = vosk.Model(str(model_dir))
    grammar = json.loads(Path(args.grammar).read_text())
    block = RATE * args.block_ms // 1000 * 2
    out = {}
    import wave
    for wav_path in sorted(Path(args.corpus).glob("*.wav")):
        w = wave.open(str(wav_path))
        pcm = w.readframes(w.getnframes())
        rec = vosk.KaldiRecognizer(model, RATE, json.dumps(grammar))
        rec.SetWords(True)
        partials, segments, fed = [], [], 0
        for i in range(0, len(pcm), block):
            chunk = pcm[i:i + block]
            fed += len(chunk) // 2
            if rec.AcceptWaveform(chunk):
                segments.append({"end_sample": fed, "result": json.loads(rec.Result())})
                partials.append(None)
            else:
                partials.append(json.loads(rec.PartialResult()))
        segments.append({"end_sample": fed, "result": json.loads(rec.FinalResult())})
        out[wav_path.stem] = {"block_ms": args.block_ms, "samples": fed, "partials": partials,
                              "segments": segments}
        print(wav_path.stem, len(segments), file=sys.stderr)
    Path(args.out).write_text(json.dumps(out))


def words_of(text):
    """Words of a text with bracketed tokens ([sil], [unk]) removed, as G2 and G4 compare."""
    return [w for w in text.split() if not (w.startswith("[") and w.endswith("]"))]


def align_ops(ref, hyp):
    """Levenshtein alignment ops: counts of (ref only, hyp only, substitutions)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, dele, ins, sub = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            sub += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return dele, ins, sub


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def score(args):
    ref = json.loads(Path(args.oracle).read_text())
    hyp_rows = {}
    for line in Path(args.hyp).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            hyp_rows[r["take"]] = r
    blocks = eq_blocks = 0
    seg_total = seg_equal = seg_matched = 0
    only_vosk = only_utter = subs = ref_words_total = 0
    vosk_segments = utter_segments = 0
    word_pairs = word_within = 0
    endpoints_ref = endpoints_hyp = endpoints_matched = endpoints_within = 0
    compute = []
    per_take = []
    step_ms = None
    for take, r in sorted(ref.items()):
        h = hyp_rows.get(take)
        if h is None:
            continue
        step_ms = r["block_ms"]
        rp, hp = r["partials"], h["partials"]
        tb = te = 0
        for a, b in zip(rp, hp):
            if a is None or b is None:
                continue
            tb += 1
            te += words_of(a.get("partial", "")) == words_of(b.get("partial", ""))
        blocks += tb
        eq_blocks += te
        compute.extend(h.get("compute_us", []))
        # segments: pair by end sample within one step
        rs = [s for s in r["segments"]]
        hs = [s for s in h["segments"]]
        vosk_segments += len(rs)
        utter_segments += len(hs)
        step_samples = RATE * r["block_ms"] // 1000
        used = set()
        take_eq = take_tot = 0
        for s in rs:
            best = None
            for k, t in enumerate(hs):
                if k in used:
                    continue
                if abs(t["end_sample"] - s["end_sample"]) <= step_samples * 5:
                    if best is None or abs(t["end_sample"] - s["end_sample"]) < abs(hs[best]["end_sample"] - s["end_sample"]):
                        best = k
            rw = words_of(s["result"].get("text", ""))
            ref_words_total += len(rw)
            seg_total += 1
            take_tot += 1
            if best is None:
                only_vosk += len(rw)
                continue
            used.add(best)
            seg_matched += 1
            hw = words_of(hs[best]["result"].get("text", ""))
            if rw == hw:
                seg_equal += 1
                take_eq += 1
            dele, ins, sub = align_ops(rw, hw)
            only_vosk += dele
            only_utter += ins
            subs += sub
            # word times on matched words
            if rw == hw:
                for a, b in zip(s["result"].get("result", []), hs[best]["result"].get("result", [])):
                    word_pairs += 1
                    if abs(a["start"] - b["start"]) <= 0.03 + 1e-6 and abs(a["end"] - b["end"]) <= 0.03 + 1e-6:
                        word_within += 1
        for k, t in enumerate(hs):
            if k not in used:
                only_utter += len(words_of(t["result"].get("text", "")))
        # endpoints: every segment but the last is an endpoint
        re_ = [s["end_sample"] for s in rs[:-1]]
        he = [s["end_sample"] for s in hs[:-1]]
        endpoints_ref += len(re_)
        endpoints_hyp += len(he)
        for e in re_:
            near = [x for x in he if abs(x - e) <= step_samples * 5]
            if near:
                endpoints_matched += 1
                if min(abs(x - e) for x in near) <= RATE // 5:
                    endpoints_within += 1
        per_take.append((take, tb, te, take_tot, take_eq))
    lines = [f"# G2, G3, G5: stream against the stock wheel, {step_ms} ms blocks", ""]
    lines.append(f"- Partial text equal on {eq_blocks} / {blocks} blocks ({pct(eq_blocks, blocks):.2f}%) [G2 asks 95%]")
    lines.append(f"- Segments: vosk {vosk_segments}, utter {utter_segments}; paired within 5 blocks: {seg_matched}; "
                 f"word sequence equal on {seg_equal} / {seg_total} vosk segments ({pct(seg_equal, seg_total):.2f}%) [G2 asks 99%]")
    lines.append(f"- Word disagreement over {ref_words_total} vosk words: {only_vosk} only in vosk, {only_utter} only in utter, {subs} substitutions")
    lines.append(f"- Word times within 30 ms on {word_within} / {word_pairs} words of equal segments ({pct(word_within, word_pairs):.2f}%) [G2 asks 95%]")
    lines.append(f"- Endpoints: vosk {endpoints_ref}, utter {endpoints_hyp}; within 0.2 s: {endpoints_within} / {endpoints_ref} ({pct(endpoints_within, endpoints_ref):.2f}%) [G5 asks 95%]")
    if compute:
        c = sorted(compute)
        p = lambda q: c[min(len(c) - 1, int(q * len(c)))] / 1000.0
        total_s = sum(compute) / 1e6
        audio_s = sum(r["samples"] for r in ref.values() if r) / RATE
        lines.append(f"- Compute per block: p50 {p(0.5):.2f} ms, p95 {p(0.95):.2f} ms, p99 {p(0.99):.2f} ms, max {c[-1]/1000:.1f} ms; RTF {total_s/audio_s:.4f} [budget p95 5 ms, RTF 0.05]")
    lines.append("")
    lines.append("| take | blocks | equal partials | vosk segments | equal |")
    lines.append("|---|---|---|---|---|")
    for take, tb, te, tt, teq in per_take:
        lines.append(f"| {take} | {tb} | {te} | {tt} | {teq} |")
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
    o.add_argument("--grammar", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--block-ms", type=int, default=40)
    o.add_argument("--dither0", action="store_true")
    o.set_defaults(fn=oracle)
    s = sub.add_parser("score")
    s.add_argument("--oracle", required=True)
    s.add_argument("--hyp", required=True)
    s.add_argument("--out", default=None)
    s.set_defaults(fn=score)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
