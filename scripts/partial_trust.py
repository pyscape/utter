#!/usr/bin/env python3
"""Whether the gap to the next reading predicts that a partial's first word will hold.

    python scripts/partial_trust.py --data DIR --model DIR [--out FILE] [--stride N]

Feeds the testing split through utterpy with alternatives and partial words on, and for every
clip records, at the block where the first word appears, what the runtime reported then: the
gap from the top reading's confidence to the next, the top reading's own confidence, and the
energy under the word. The target is whether that first-shown word survives to the final. The
note reports each signal's rank-AUC, a calibration of the gap against a two-reading softmax,
and the cost of holding a word until the gap clears a threshold: the trust rule the README's
partial example and docs/benchmarks/partial-trust.md describe.
"""

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import speech_commands as sc  # noqa: E402

RATE = 16000


def first_word(text):
    ws = sc.words_of(text)
    return ws[0] if ws else None


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def auc(pairs):
    """Rank-AUC of a score against a boolean label; 0.5 is no information, below 0.5 inverted."""
    xs = [(s, y) for s, y in pairs if s is not None]
    pos = [s for s, y in xs if y]
    neg = [s for s, y in xs if not y]
    if not pos or not neg:
        return None
    xs.sort(key=lambda t: t[0])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j < len(xs) and xs[j][0] == xs[i][0]:
            j += 1
        for k in range(i, j):
            ranks[k] = (i + j - 1) / 2 + 1
        i = j
    rank_sum = sum(r for r, (_, y) in zip(ranks, xs) if y)
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def trace(clips, eng, block_ms, alternatives):
    """Per clip, the top-two confidences and the leading word's energy at its first appearance,
    and whether that word is the one the final settles on."""
    feats = []
    t0 = time.monotonic()
    for n, path in enumerate(clips):
        if n and n % 2000 == 0:
            print(f"{n}/{len(clips)} {time.monotonic() - t0:.0f}s", file=sys.stderr, flush=True)
        shown = {}

        def on_partial(fed, p, shown=shown):
            if shown or not sc.words_of(p.get("partial", "")):
                return
            alts = p.get("partial_alternatives") or []
            if len(alts) < 2:
                return
            lead = [e for e in (p.get("partial_result") or []) if e["word"] == first_word(p["partial"])]
            shown["gap"] = alts[0]["confidence"] - alts[1]["confidence"]
            shown["conf0"] = alts[0]["confidence"]
            shown["energy"] = lead[-1]["energy_dbfs"] if lead else None
            shown["word"] = first_word(p["partial"])

        rec = eng.new(alternatives=alternatives)
        if hasattr(rec, "SetPartialWords"):
            rec.SetPartialWords(True)
        finals = sc.feed(rec, sc.read_pcm(path), block_ms, on_partial)
        if not shown:
            continue
        final = first_word(" ".join(f.get("text", "") for f in finals))
        feats.append(
            dict(gap=shown["gap"], conf0=shown["conf0"], energy=shown["energy"], survived=shown["word"] == final)
        )
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="docs/benchmarks/partial-trust.md")
    ap.add_argument("--block-ms", type=int, default=40)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--alternatives", type=int, default=5)
    a = ap.parse_args()

    import utterpy

    data = Path(a.data)
    listed = [x.strip() for x in (data / "testing_list.txt").read_text().splitlines() if x.strip()]
    by = defaultdict(list)
    for rel in listed:
        by[rel.split("/")[0]].append(rel)
    clips = [data / rel for w in sc.DATASET_WORDS for rel in by.get(w, [])][:: a.stride]

    grammar = sc.DATASET_WORDS + sc.LETTERS + sc.NATO + sc.COLOURS
    eng = sc.Engine("utterpy", utterpy, a.model, grammar)
    feats = trace(clips, eng, a.block_ms, a.alternatives)

    n = len(feats)
    revised = sum(1 for f in feats if not f["survived"])
    good = n - revised

    lines = [
        "# Reading a partial's trust",
        "",
        f"Run {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}, utterpy over the Speech "
        f"Commands testing split, {n} clips whose first word appeared in a partial that carried a "
        f"runner-up, 92-entry grammar, {a.block_ms} ms blocks, {a.alternatives} alternatives. "
        "Measured by `scripts/partial_trust.py`.",
        "",
        f"The first word a host sees is revised before the utterance ends on {revised} of {n} "
        f"({100 * revised / n:.1f}%). A host that wants to act early needs to know, at the moment a "
        "word appears, whether it is one of those. The signals the partial carries at that moment, "
        "each scored by rank-AUC against whether the word survived:",
        "",
        "| signal on the partial | rank-AUC |",
        "|---|---|",
        f"| gap from the top `confidence` to the next | {auc([(f['gap'], not f['survived']) for f in feats]):.2f} |",
        f"| the top reading's own `confidence` | {auc([(f['conf0'], not f['survived']) for f in feats]):.2f} |",
        f"| `energy_dbfs` under the word | {auc([(f['energy'], not f['survived']) for f in feats]):.2f} |",
        "",
        "0.5 is no information; below 0.5 the signal is inverted (a smaller value means a revision "
        "is coming). Only the gap carries the answer. The word's own confidence barely moves it, "
        "because it is a raw accumulated path cost that grows with the utterance and is not a "
        "probability; the energy is the silence signal, not the trust one. `stable_ms` is zero the "
        "instant a word appears, so it cannot rank a first sighting, and is the hold a host waits "
        "out afterward instead.",
        "",
        "## The gap is a probability the API already gives you",
        "",
        "`confidence` is the negation of a reading's best-token cost, and the decoder's costs are "
        "negative log-likelihoods in nats. So the gap between the top two readings,",
        "",
        "    gap = confidence[0] - confidence[1] = cost[1] - cost[0]  >= 0,",
        "",
        "is how many nats of likelihood the runner-up gives up to the leader. Treating the two as a "
        "choice between them, the leader's share of the mass is a softmax over the two costs,",
        "",
        "    P(leader) = e^-cost0 / (e^-cost0 + e^-cost1) = 1 / (1 + e^-gap) = sigmoid(gap).",
        "",
        "That is a Viterbi approximation over the two leading beam readings, not a lattice "
        "posterior, so it is only roughly calibrated; the table below is how roughly. Buckets of "
        "the first appearances by gap, against the survival `sigmoid(gap)` predicts:",
        "",
        "| gap (nats) | first appearances | survived | predicted `sigmoid(mean gap)` |",
        "|---|---|---|---|",
    ]
    edges = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, float("inf"))]
    for lo, hi in edges:
        b = [f for f in feats if lo <= f["gap"] < hi]
        if not b:
            continue
        surv = sum(1 for f in b if f["survived"])
        mean_gap = sum(f["gap"] for f in b) / len(b)
        hi_label = "8+" if hi == float("inf") else f"{hi:g}"
        lines.append(
            f"| {lo:g} to {hi_label} | {len(b)} | {surv} / {len(b)} ({100 * surv / len(b):.0f}%) | "
            f"{100 * sigmoid(mean_gap):.0f}% |"
        )
    lines += [
        "",
        "Survival climbs with the gap and tracks the softmax, so the sigmoid is a usable trust "
        "score and a threshold on it is a trust gate. In terms a host acts on, holding a word until "
        "the gap clears a threshold:",
        "",
        "| hold until gap over | words held | revisions caught | good words delayed |",
        "|---|---|---|---|",
    ]
    for thr in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        held = [f for f in feats if f["gap"] < thr]
        caught = sum(1 for f in held if not f["survived"])
        delayed = sum(1 for f in held if f["survived"])
        lines.append(
            f"| {thr:.1f} | {len(held)} ({100 * len(held) / n:.0f}%) | "
            f"{caught} / {revised} ({100 * caught / (revised or 1):.0f}%) | "
            f"{delayed} / {good} ({100 * delayed / good:.0f}%) |"
        )
    lines += [
        "",
        "So the number to read is not a word's own confidence but its lead over the next reading. A "
        "host that waits for `sigmoid(gap)` to clear its bar trades a little latency on the words it "
        "holds for catching most of the revisions it would otherwise have acted on. The Python is "
        "in the README, under the partial example.",
        "",
    ]
    Path(a.out).write_text("\n".join(lines))
    print(f"wrote {a.out} from {n} clips", file=sys.stderr)


if __name__ == "__main__":
    main()
