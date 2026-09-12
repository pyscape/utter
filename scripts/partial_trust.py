#!/usr/bin/env python3
"""Whether the gap to the next reading, and how that gap moves, predict that a word will hold.

    python scripts/partial_trust.py --data DIR --model DIR [--out FILE] [--clips FILE] [--stride N]
    python scripts/partial_trust.py --from-clips FILE [--fit-clips FILE] [--out FILE] [--json FILE]

Feeds the testing split through utterpy with alternatives and partial words on, and for every
clip records, at the block where the first word appears, what the runtime reported then: the
gap from the top reading's confidence to the next, the top reading's own confidence, and the
energy under the word. The target is whether that first-shown word survives to the final. The
note reports each signal's rank-AUC, a calibration of the gap against a two-reading softmax,
and the cost of holding a word until the gap clears a threshold: the trust rule the README's
partial example and docs/benchmarks/partial-trust.md describe.

The gap is a snapshot of a two-way Viterbi softmax rather than a posterior, and an
approximation's errors are autocorrelated where a posterior's increments would not be, so the
whole per-advance series of readings from the first sighting on is kept and the note asks a
second question: does how each reading's standing moves between decoder advances say anything
the level does not. Those sections measure what motion is even defined at a first sighting,
each motion signal's AUC alone and at equal gap, and how often a reading still in contention
vanishes; then three rules are put on the same clips and charged for their delay: the README's
sigmoid of the gap, that plus the motion, and that read one advance later.

The decode and the figures are separate runs. A decode writes one JSON line per clip, with the
per-advance series from the first sighting to the end of the clip, so every figure can be
recomputed, and any rule's delay to clear rescored, without decoding again. `--from-clips`
reads those records instead of a model, and `--fit-clips` takes the coefficients from a second
file, which is how the validation split fits a rule the testing split scores.
"""

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import speech_commands as sc  # noqa: E402

RATE = 16000
SAMPLES_PER_MS = RATE // 1000
IN_CONTENTION = -2.0  # nats of deficit to the leader, for the merge-loss proxy
OPERATING_TRUST = 0.9  # the README's bar: sigmoid(gap) >= 0.9, a lead of about 2.2 nats
BARS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
BOOTSTRAP = 1000

GAP_EDGES = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, float("inf"))]
TRUST_EDGES = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.01)]

LEVEL = [
    ("gap", "gap from the top `confidence` to the next"),
    ("conf0", "the top reading's own `confidence`"),
    ("energy", "`energy_dbfs` under the word"),
]
MOTION = [
    ("advances_alive", "advances the top reading has stood"),
    ("lead_delta0", "`lead_delta` of rank 0"),
    ("lead_delta1", "`lead_delta` of rank 1"),
    ("displaced_delta", "`lead_delta` of the reading that led at the last advance"),
    ("churn", "rank-0 changes so far"),
    ("entropy", "entropy over the readings"),
    ("entropy_delta", "`entropy_delta`"),
    ("vanished", "readings in contention that vanished, by now"),
]


def first_word(text):
    ws = sc.words_of(text)
    return ws[0] if ws else None


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, x))))


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


# ---------------------------------------------------------------- the decode


def readings_of(p):
    return tuple((e["text"], e["confidence"]) for e in (p.get("partial_alternatives") or []))


def leads_of(readings):
    """Each reading's confidence less the best of the others: the gap for rank 0, the deficit to
    the leader for the rest. Undefined where a reading stands alone."""
    if len(readings) < 2:
        return {t: None for t, _ in readings}
    return {t: c - max(d for u, d in readings if u != t) for t, c in readings}


def relation_of(top, text):
    """A reading's word sequence set against rank 0's, as
    `[[rr:TD-9#Every reading names its relation to the partial]]` defines it: a sequence relation
    from the labels, and nothing about which word differs."""
    a, b = sc.words_of(top), sc.words_of(text)
    if a == b:
        return "same"
    if a[: len(b)] == b:
        return "prefix"
    if b[: len(a)] == a:
        return "extends"
    return "differs"


def entropy_of(readings):
    """Shannon entropy in nats of the softmax over the confidences."""
    if not readings:
        return None
    top = max(c for _, c in readings)
    w = [math.exp(c - top) for _, c in readings]
    z = sum(w)
    return -sum((x / z) * math.log(x / z) for x in w if x > 0)


def energy_of(p):
    lead = [e for e in (p.get("partial_result") or []) if e["word"] == first_word(p["partial"])]
    return lead[-1]["energy_dbfs"] if lead else None


def advance_states(series):
    """The blocks where the readings changed, each with what the readings had done by then.

    A block that carries no readings at all is not an advance: the decoder has not run a chunk
    yet, so there is no beam to report. Past that, `[[rr:TD-7#Decision outcome]]` is why a change
    marks an advance: the readings are a function of the decoded frame count, so between advances
    every block repeats the last one."""
    states = []
    alive = {}
    churn = 0
    prev = None
    for i, (fed, readings, p) in enumerate(series):
        if not readings or (prev is not None and readings == series[i - 1][1]):
            continue
        texts = [t for t, _ in readings]
        alive = {t: alive.get(t, 0) + 1 for t in texts}
        lead = leads_of(readings)
        ent = entropy_of(readings)
        top = texts[0]
        if prev is not None and prev["top"] != top:
            churn += 1
        delta = {}
        if prev is not None:
            for t in texts:
                before = prev["lead"].get(t)
                if before is not None and lead[t] is not None:
                    delta[t] = lead[t] - before
        st = dict(
            block=i,
            fed=fed,
            p=p,
            readings=readings,
            top=top,
            lead=lead,
            delta=delta,
            alive=dict(alive),
            churn=churn,
            entropy=ent,
            entropy_delta=None if prev is None else ent - prev["entropy"],
            prev_top=None if prev is None else prev["top"],
            prev_texts=frozenset() if prev is None else frozenset(prev["lead"]),
        )
        states.append(st)
        prev = st
    return states


def vanishings(states):
    """Per advance, the readings that were within `IN_CONTENTION` nats of the leader and are gone
    at the next advance. An upper bound on Viterbi merges: beam pruning drops readings too."""
    out = []
    for k in range(len(states) - 1):
        here, nxt = states[k], states[k + 1]
        gone = [
            t for t, lead in here["lead"].items() if lead is not None and lead >= IN_CONTENTION and t not in nxt["lead"]
        ]
        out.append((nxt["block"], len(gone)))
    return out


def features_at(st, vanished):
    """What a host could read off one advance."""
    r = st["readings"]
    rank1 = r[1][0] if len(r) >= 2 else None
    prev_top = st["prev_top"]
    return dict(
        ms=st["fed"] / SAMPLES_PER_MS,
        top_word=first_word(st["top"]),
        gap=r[0][1] - r[1][1] if len(r) >= 2 else None,
        conf0=r[0][1],
        energy=energy_of(st["p"]),
        advances_alive=st["alive"][st["top"]],
        lead_delta0=st["delta"].get(st["top"]),
        lead_delta1=st["delta"].get(rank1),
        displaced_delta=st["delta"].get(prev_top),
        churn=st["churn"],
        entropy=st["entropy"],
        entropy_delta=st["entropy_delta"],
        vanished=vanished,
        had_history=st["top"] in st["prev_texts"],
        readings=[
            dict(text=t, conf=c, relation=relation_of(st["top"], t), lead=st["lead"][t], lead_delta=st["delta"].get(t))
            for t, c in r
        ],
    )


def decode(clips, eng, block_ms, alternatives):
    """Per clip, the readings at the first block whose partial carries a word and a runner-up, and
    every advance from there to the end of the clip."""
    recs = []
    diag = dict(clips=0, intervals=[], advances=[], first_advance_ms=[], gaps_after_start=0, sighting_off_advance=0)
    t0 = time.monotonic()
    for n, path in enumerate(clips):
        if n and n % 2000 == 0:
            print(f"{n}/{len(clips)} {time.monotonic() - t0:.0f}s", file=sys.stderr, flush=True)
        series = []
        sight = []

        def on_partial(fed, p, series=series, sight=sight):
            series.append((fed, readings_of(p), p))
            if not sight and len(p.get("partial_alternatives") or []) >= 2 and sc.words_of(p.get("partial", "")):
                sight.append(len(series) - 1)

        rec = eng.new(alternatives=alternatives)
        if hasattr(rec, "SetPartialWords"):
            rec.SetPartialWords(True)
        finals = sc.feed(rec, sc.read_pcm(path), block_ms, on_partial)

        states = advance_states(series)
        diag["clips"] += 1
        diag["advances"].append(len(states))
        if states:
            diag["first_advance_ms"].append(states[0]["fed"] / SAMPLES_PER_MS)
            diag["gaps_after_start"] += sum(1 for _, r, _ in series[states[0]["block"] :] if not r)
        for a, b in zip(states, states[1:]):
            diag["intervals"].append((b["fed"] - a["fed"]) / SAMPLES_PER_MS)
        if not sight:
            continue
        block = sight[0]
        at = [k for k, st in enumerate(states) if st["block"] == block]
        if not at:
            diag["sighting_off_advance"] += 1
            continue
        k = at[0]
        gone = vanishings(states)
        word = first_word(series[block][2]["partial"])
        final = first_word(" ".join(f.get("text", "") for f in finals))
        entries = [features_at(st, sum(c for b, c in gone if b <= st["block"])) for st in states[k:]]
        recs.append(
            dict(
                entries[0],
                clip=f"{path.parent.name}/{path.name}",
                word=word,
                survived=word == final,
                sighting_ms=entries[0]["ms"],
                end_ms=series[-1][0] / SAMPLES_PER_MS,
                advance_index=k,
                n_advances=len(states),
                advance_ms=[st["fed"] / SAMPLES_PER_MS for st in states],
                vanished_all=sum(c for _, c in gone),
                series=entries,
            )
        )
    return recs, diag


def promote(rec):
    """The record as the figures read it: the sighting's own features at the top level, and the
    next advance beside them with the wait it costs."""
    f = dict(rec)
    if len(rec["series"]) > 1:
        f["hold"] = dict(rec["series"][1], wait_ms=rec["series"][1]["ms"] - rec["series"][0]["ms"])
        f["hold"]["word_same"] = rec["series"][1]["top_word"] == rec["word"]
        f["hold"]["word_gone"] = rec["series"][1]["top_word"] is None
    else:
        f["hold"] = None
    return f


def load(path):
    with open(path) as fh:
        return [promote(json.loads(line)) for line in fh if line.strip()]


# ---------------------------------------------------------------- statistics


def defined(feats, key):
    return [f for f in feats if f.get(key) is not None]


def fmt(x, places=2):
    return "-" if x is None else f"{x:.{places}f}"


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def dist(values, places=0):
    if not values:
        return "-"
    return f"{sc.quantile(values, 0.5):.{places}f} / {sc.quantile(values, 0.9):.{places}f}"


def solve(A, b):
    """Gaussian elimination with partial pivoting; None where the system is singular."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-12:
            return None
        M[c], M[piv] = M[piv], M[c]
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / M[c][c]
            for j in range(c, n + 1):
                M[r][j] -= f * M[c][j]
    return [M[i][n] / M[i][i] for i in range(n)]


def logistic_fit(rows, ys, l2=1e-3, iters=60):
    """Newton steps on the log-likelihood. The ridge is there because some columns separate the
    labels outright, and an unpenalised fit walks off to infinity on those."""
    k = len(rows[0])
    b = [0.0] * k
    for _ in range(iters):
        g = [-l2 * v for v in b]
        H = [[l2 if i == j else 0.0 for j in range(k)] for i in range(k)]
        for x, y in zip(rows, ys):
            z = sum(bi * xi for bi, xi in zip(b, x))
            p = sigmoid(z)
            w = p * (1 - p)
            r = (1.0 if y else 0.0) - p
            for i in range(k):
                g[i] += r * x[i]
                for j in range(k):
                    H[i][j] += w * x[i] * x[j]
        step = solve(H, g)
        if step is None:
            break
        b = [bi + si for bi, si in zip(b, step)]
        if max(abs(s) for s in step) < 1e-9:
            break
    return b


def fit(cols, ys):
    """Coefficients and the in-sample AUC of a logistic fit; the columns get an intercept here."""
    rows = [[1.0] + list(c) for c in cols]
    b = logistic_fit(rows, ys)
    scores = [sum(bi * xi for bi, xi in zip(b, x)) for x in rows]
    return b, auc(list(zip(scores, ys)))


def ece(pairs, edges):
    """Expected calibration error: how far the predicted probability sits from the observed rate,
    averaged over bins by the clips in them."""
    rows = [(p, y) for p, y in pairs if p is not None]
    if not rows:
        return None, []
    out = []
    err = 0.0
    for lo, hi in edges:
        b = [(p, y) for p, y in rows if lo <= p < hi]
        if not b:
            continue
        mean_p = sum(p for p, _ in b) / len(b)
        obs = sum(1 for _, y in b if y) / len(b)
        err += len(b) / len(rows) * abs(mean_p - obs)
        out.append(((lo, hi), len(b), mean_p, obs))
    return err, out


def sorted_pairs(scores, labels):
    idx = [i for i, s in enumerate(scores) if s is not None]
    idx.sort(key=lambda i: scores[i])
    return [(i, scores[i], labels[i]) for i in idx]


def weighted_auc(pairs, w):
    """Mann-Whitney U over pre-sorted scores with each clip counted `w[i]` times: the bootstrap
    resamples clips, so the sort is done once and the weights change."""
    below = acc = pos = neg = 0.0
    i = 0
    n = len(pairs)
    while i < n:
        j = i
        s = pairs[i][1]
        tp = tn = 0.0
        while j < n and pairs[j][1] == s:
            k, _, y = pairs[j]
            if y:
                tp += w[k]
            else:
                tn += w[k]
            j += 1
        acc += tp * (below + 0.5 * tn)
        below += tn
        pos += tp
        neg += tn
        i = j
    return acc / (pos * neg) if pos and neg else None


def paired_bootstrap(a_scores, b_scores, labels, n=BOOTSTRAP, seed=0):
    """The AUC difference between two rules and its 95% interval, resampling clips in pairs so
    the two rules always see the same clips."""
    keep = [i for i in range(len(labels)) if a_scores[i] is not None and b_scores[i] is not None]
    both = set(keep)
    a = sorted_pairs([a_scores[i] if i in both else None for i in range(len(labels))], labels)
    b = sorted_pairs([b_scores[i] if i in both else None for i in range(len(labels))], labels)
    rng = random.Random(seed)
    m = len(labels)
    diffs = []
    for _ in range(n):
        w = [0] * m
        for _ in range(len(keep)):
            w[keep[rng.randrange(len(keep))]] += 1
        x, y = weighted_auc(a, w), weighted_auc(b, w)
        if x is not None and y is not None:
            diffs.append(y - x)
    diffs.sort()
    if not diffs:
        return None, None, None, len(keep)
    ones = [1] * m
    base = weighted_auc(b, ones) - weighted_auc(a, ones)
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[min(len(diffs) - 1, int(0.975 * len(diffs)))]
    return base, lo, hi, len(keep)


# ---------------------------------------------------------------- the rules


def rule_cols(e):
    """The motion a host can read at one advance, missing values as zero beside the flag that says
    they were missing, which is what reading the field unconditionally amounts to."""
    return [
        e["gap"],
        e["displaced_delta"] or 0.0,
        0.0 if e["displaced_delta"] is not None else 1.0,
        e["lead_delta0"] or 0.0,
        0.0 if e["lead_delta0"] is not None else 1.0,
    ]


def gap_cols(e):
    return [e["gap"]]


def gap_entropy_cols(e):
    return [e["gap"], e["entropy"] if e["entropy"] is not None else 0.0]


def entropy_motion_cols(e):
    return gap_entropy_cols(e) + rule_cols(e)[1:]


COEF_ORDER = {
    "gap": "intercept, gap",
    "gap + entropy": "intercept, gap, entropy",
    "gap + motion": ("intercept, gap, displaced `lead_delta`, its missing flag, rank-0 `lead_delta`, its missing flag"),
    "gap + entropy + motion": (
        "intercept, gap, entropy, displaced `lead_delta`, its missing flag, rank-0 `lead_delta`, its missing flag"
    ),
}


def fit_cols(entries, ys, cols):
    rows = [[1.0] + cols(e) for e in entries if e["gap"] is not None]
    ok = [y for e, y in zip(entries, ys) if e["gap"] is not None]
    return logistic_fit(rows, ok)


def fit_rule(entries, ys):
    return fit_cols(entries, ys, rule_cols)


def gap_trust(e):
    """The README's rule: sigmoid of the lead, and 1.0 where nothing else is in the list."""
    return 1.0 if e["gap"] is None else sigmoid(e["gap"])


def fitted_trust(cols, coef):
    """A fitted rule's trust: one minus its P(revision), so every rule is read on one scale."""

    def trust(e):
        if e["gap"] is None:
            return 1.0
        x = [1.0] + cols(e)
        return 1.0 - sigmoid(sum(c * v for c, v in zip(coef, x)))

    return trust


def rule_trust(e, coef):
    return fitted_trust(rule_cols, coef)(e)


def always_trust(e):
    """The hold-only rule: no score at all, so it releases at the first advance it is read on."""
    return 1.0


def clears_at(f, trust_fn, bar, start):
    """The ms at which a rule first releases the word it is holding. Rank 0 has to still lead with
    that word: a host is offered the word it acts on, and a partial that has moved on is not
    offering it."""
    for e in f["series"][start:]:
        if e["top_word"] == f["word"] and trust_fn(e) >= bar:
            return e["ms"]
    return None


def rule_run(feats, trust_fn, bar, start):
    """Per clip, when the rule released the word and what that costs: a revision the rule never
    released is caught, a good word it never released is delayed to the final."""
    caught = 0
    delays = []
    never = 0
    correct = []
    per_clip = []
    for f in feats:
        at = clears_at(f, trust_fn, bar, start)
        if at is None:
            never += 1
        delay = None
        if f["survived"]:
            delay = (at - f["sighting_ms"]) if at is not None else (f["end_ms"] - f["sighting_ms"])
            delays.append(delay)
        elif at is None:
            caught += 1
        correct.append((f["survived"] and at is not None) or (not f["survived"] and at is None))
        per_clip.append((f["survived"], at is None, delay))
    return dict(caught=caught, delays=delays, never=never, correct=correct, per_clip=per_clip)


def run_interval(run, n=BOOTSTRAP, seed=1):
    """The realized catch and delay with a 95% interval, resampling clips with replacement. A
    threshold frozen on one split lands where the other split's clips put it, and that is a
    sample."""
    rows = run["per_clip"]
    if not rows:
        return {}
    rng = random.Random(seed)
    caught, delay = [], []
    m = len(rows)
    for _ in range(n):
        pick = [rows[rng.randrange(m)] for _ in range(m)]
        rev = [held for survived, held, _ in pick if not survived]
        good = [d for survived, _, d in pick if survived and d is not None]
        if rev:
            caught.append(sum(rev) / len(rev))
        if good:
            delay.append(sum(good) / len(good))
    caught.sort()
    delay.sort()

    def band(v):
        return (v[int(0.025 * len(v))], v[min(len(v) - 1, int(0.975 * len(v)))]) if v else (None, None)

    rev_n = sum(1 for survived, _, _ in rows if not survived)
    return dict(
        caught_share=run["caught"] / rev_n if rev_n else None,
        caught_lo=band(caught)[0],
        caught_hi=band(caught)[1],
        delay_mean=mean(run["delays"]),
        delay_lo=band(delay)[0],
        delay_hi=band(delay)[1],
    )


def frozen_bar(fit_feats, trust_fn, start, target, kind, grid=None):
    """The bar a rule is frozen at, chosen on the fitting split alone: the one whose delay, or
    whose revisions caught, comes closest to the reference rule's on that same split."""
    grid = grid or [i / 200 for i in range(1, 200)]
    runs = [(bar, rule_run(fit_feats, trust_fn, bar, start)) for bar in grid]
    if kind == "delay":
        return min(runs, key=lambda t: abs(mean(t[1]["delays"]) - target))[0]
    return min(runs, key=lambda t: (abs(t[1]["caught"] / max(1, len(fit_feats)) - target), mean(t[1]["delays"])))[0]


def rule_score(feats, trust_fn, start, as_run=False):
    """Each clip's P(revision) under a rule, or None where the rule has no reading to score. As the
    rule runs, a word rank 0 has already dropped is a revision the rule calls with certainty; the
    reading alone does not say that, so both orderings are scored."""
    out = []
    for f in feats:
        e = f["series"][start] if len(f["series"]) > start else None
        if e is None:
            out.append(None)
        elif as_run and e["top_word"] != f["word"]:
            out.append(1.0)
        else:
            out.append(1.0 - trust_fn(e))
    return out


# ---------------------------------------------------------------- the page


def availability_section(feats, diag, lines, fig):
    n = len(feats)
    interval = sc.quantile(diag["intervals"], 0.5) if diag["intervals"] else float("nan")
    exact = sum(1 for v in diag["intervals"] if v == interval)
    lines += [
        "",
        "## What the readings have done by the time a word appears",
        "",
        "The gap above is a level: one snapshot of a two-way Viterbi softmax. It is an "
        "approximation, not a posterior, so its errors need not be independent between readings "
        "of the beam, and how a reading's standing *moves* could carry survival information the "
        "level does not. What follows measures that. The readings only change when the decoder "
        "advances a frame, which at 40 ms blocks is one block in six "
        "(`[[rr:TD-7#Decision outcome]]`), so motion is measured per advance, not per block. An "
        "advance is a block whose readings differ from the block before; a reading is its whole "
        "text, `[sil]` included; and a reading's lead is its confidence less the best of the "
        "others, which is the gap for rank 0 and a deficit for the rest.",
        "",
        f"Advances arrive every {dist(diag['intervals'])} ms (median / p90), and on this corpus "
        f"that is exact: {exact} of the {len(diag['intervals'])} intervals measured "
        f"{interval:.0f} ms and none measured anything else, one network chunk at 40 ms blocks. The "
        f"first advance lands at {dist(diag['first_advance_ms'])} ms, the audio that first chunk "
        f"needs. A one-second clip therefore holds {dist(diag['advances'])} advances, and a first "
        "sighting has almost no history behind it.",
        "",
        "The two moments a rule is read at, and the two places the figure could come from. The "
        "harness sees the readings the runtime reports, five of them here, and derives the motion "
        "from its own series; the runtime keeps the history over every surviving group, so a "
        "reading entering the list carries motion the harness has no record of "
        "(`[[rr:TD-9#Readings are read once per decoding advance]]`). The runtime column is "
        "**pending**: it is filled when the fields are read from the JSON instead of derived, and "
        "until then this page's availability is a lower bound on the runtime's.",
        "",
        "| figure | defined at the sighting | defined one advance later | runtime, over every group |",
        "|---|---|---|---|",
    ]
    held = [f for f in feats if f["hold"]]
    rows = [
        (
            "history",
            "the rank-0 reading was there at the previous advance (`had_history`)",
            sum(1 for f in feats if f["had_history"]),
            sum(1 for f in held if f["hold"]["had_history"]),
        ),
        (
            "lead_delta0",
            "`lead_delta` of rank 0 is defined",
            len(defined(feats, "lead_delta0")),
            sum(1 for f in held if f["hold"]["lead_delta0"] is not None),
        ),
        (
            "lead_delta1",
            "`lead_delta` of rank 1 is defined",
            len(defined(feats, "lead_delta1")),
            sum(1 for f in held if f["hold"]["lead_delta1"] is not None),
        ),
        (
            "displaced_delta",
            "`lead_delta` of the reading that led before is defined",
            len(defined(feats, "displaced_delta")),
            sum(1 for f in held if f["hold"]["displaced_delta"] is not None),
        ),
        (
            "entropy_delta",
            "`entropy_delta` is defined (there was a previous advance)",
            len(defined(feats, "entropy_delta")),
            sum(1 for f in held if f["hold"]["entropy_delta"] is not None),
        ),
    ]
    for _, label, c, h in rows:
        hold_cell = f"{h} / {len(held)} ({100 * h / len(held):.1f}%)" if held else "-"
        lines.append(f"| {label} | {c} / {n} ({100 * c / n:.1f}%) | {hold_cell} | pending |")
    alive = [f["advances_alive"] for f in feats]
    alive1 = sum(1 for v in alive if v == 1)
    lines += [
        "",
        f"The rank-0 reading has stood {dist(alive, 0)} advances (median / p90) and is at its "
        f"first on {100 * alive1 / n:.1f}% of first sightings. That is the shape of the problem: a "
        "word first appears when its reading appears, so at that moment there is usually nothing "
        "to have moved.",
    ]
    fig["availability"] = dict(
        first_sightings=n,
        clips_decoded=diag["clips"],
        defined={k: c for k, _, c, _ in rows},
        defined_at_hold={k: h for k, _, _, h in rows},
        hold_available=len(held),
        runtime_all_groups="pending",
        advances_alive_p50=sc.quantile(alive, 0.5),
        advances_alive_p90=sc.quantile(alive, 0.9),
        born_at_sighting=alive1,
        advance_interval_ms_p50=interval,
        advance_intervals=len(diag["intervals"]),
        advance_intervals_at_p50=exact,
        advance_interval_ms_p90=sc.quantile(diag["intervals"], 0.9) if diag["intervals"] else None,
        first_advance_ms_p50=sc.quantile(diag["first_advance_ms"], 0.5) if diag["first_advance_ms"] else None,
        advances_per_clip_p50=sc.quantile(diag["advances"], 0.5) if diag["advances"] else None,
        blocks_without_readings_after_start=diag["gaps_after_start"],
        sightings_off_advance=diag["sighting_off_advance"],
    )


def auc_section(feats, lines, fig):
    lines += [
        "",
        "## Every signal at the first sighting, against a revision",
        "",
        "The same rank-AUC as above, revision as the positive label, extended with the motion "
        "signals. A signal that is undefined at some first sightings is scored on the ones where "
        "it is defined, and that subset is given:",
        "",
        "| signal at the first sighting | first sightings scored | rank-AUC |",
        "|---|---|---|",
    ]
    fig["auc_at_sighting"] = {}
    for key, label in LEVEL + MOTION:
        sub = defined(feats, key)
        a = auc([(f[key], not f["survived"]) for f in sub])
        lines.append(f"| {label} | {len(sub)} | {fmt(a)} |")
        fig["auc_at_sighting"][key] = dict(n=len(sub), auc=a)
    lines += [
        "",
        "Below 0.5 the signal is inverted, as with the gap. The subset column is half the answer: "
        "a signal with information on a tenth of the first sightings is a signal a host cannot "
        "read most of the time.",
    ]


def stratified_section(feats, lines, fig):
    lines += [
        "",
        "## The same signals at equal gap",
        "",
        "AUC inside each gap bucket, so the level is held roughly fixed and what is left is "
        "whatever the motion adds. A dash is a bucket with no revisions, one class only, or no "
        "clip where the signal is defined:",
        "",
        "| signal | "
        + " | ".join(f"{lo:g}-{'8+' if hi == float('inf') else f'{hi:g}'}" for lo, hi in GAP_EDGES)
        + " |",
        "|---|" + "---|" * len(GAP_EDGES),
    ]
    strat = []
    fig["auc_by_gap_bucket"] = {}
    for key, label in MOTION:
        cells = []
        read = []
        fig["auc_by_gap_bucket"][key] = []
        for lo, hi in GAP_EDGES:
            b = [f for f in feats if lo <= f["gap"] < hi and f.get(key) is not None]
            a = auc([(f[key], not f["survived"]) for f in b])
            cells.append(fmt(a))
            fig["auc_by_gap_bucket"][key].append(dict(lo=lo, hi=None if hi == float("inf") else hi, n=len(b), auc=a))
            if a is not None and len(b) >= 50:
                read.append((len(b), abs(a - 0.5)))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
        if read:
            strat.append((sum(w * d for w, d in read) / sum(w for w, _ in read), key, label))
    counts = [len([f for f in feats if lo <= f["gap"] < hi]) for lo, hi in GAP_EDGES]
    lines.append("| first sightings in the bucket | " + " | ".join(str(c) for c in counts) + " |")
    strat.sort(reverse=True)
    return strat


def logistic_section(feats, lines, fig):
    """Gap alone against gap plus one motion signal, in sample, one signal at a time."""
    ys = [not f["survived"] for f in feats]
    lines += [
        "",
        "## Gap alone against gap plus one motion signal",
        "",
        "A two-feature logistic regression of P(revision) on the gap and one motion signal, fit "
        "by Newton steps with a 1e-3 ridge, scored **in sample** by AUC against the gap alone on "
        "the same clips. The held-out version of this question is the three rules below; this "
        "table is the per-signal ceiling. First on the clips where the signal is defined:",
        "",
        "| signal | clips | AUC, gap | AUC, gap + signal | coefficient on gap | on the signal |",
        "|---|---|---|---|---|---|",
    ]
    fig["incremental_in_sample"] = {}
    for key, label in MOTION:
        sub = defined(feats, key)
        y = [not f["survived"] for f in sub]
        if not sub or len(set(y)) < 2:
            continue
        _, a0 = fit([[f["gap"]] for f in sub], y)
        b, a1 = fit([[f["gap"], f[key]] for f in sub], y)
        lines.append(f"| {label} | {len(sub)} | {fmt(a0, 3)} | {fmt(a1, 3)} | {b[1]:+.3f} | {b[2]:+.4f} |")
        fig["incremental_in_sample"][key] = dict(n=len(sub), auc_gap=a0, auc_both=a1, coef=b)
    _, base = fit([[f["gap"]] for f in feats], ys)
    lines += [
        "",
        f"Then on all {len(feats)} first sightings, the undefined values set to zero beside an "
        "indicator column that says they were undefined, which is what a host that reads the "
        f"field unconditionally would be doing. The gap alone scores {fmt(base, 3)} on this set:",
        "",
        "| signal | AUC, gap + signal + missing | coefficient on gap | on the signal | on missing |",
        "|---|---|---|---|---|",
    ]
    best = (base, None, None)
    fig["incremental_filled"] = {}
    for key, label in MOTION:
        cols = [[f["gap"], f[key] or 0.0, 0.0 if f.get(key) is not None else 1.0] for f in feats]
        if all(c[2] == 0.0 for c in cols):
            cols = [c[:2] for c in cols]
        b, a1 = fit(cols, ys)
        miss = f"{b[3]:+.3f}" if len(b) > 3 else "-"
        lines.append(f"| {label} | {fmt(a1, 3)} | {b[1]:+.3f} | {b[2]:+.4f} | {miss} |")
        fig["incremental_filled"][key] = dict(auc=a1, coef=b)
        if a1 is not None and a1 > best[0]:
            best = (a1, key, label)
    fig["incremental_filled"]["gap_alone_auc"] = base
    return best, base


def hold_section(feats, best, base, lines, fig):
    """The existing hold-until-gap table, matched on words held against the best in-sample fit."""
    n = len(feats)
    revised = sum(1 for f in feats if not f["survived"])
    good = n - revised
    ys = [not f["survived"] for f in feats]
    key, label = best[1], best[2]
    lines += ["", "## Holding by the fit instead of by the gap, at equal words held", ""]
    if key is None:
        lines.append(
            "No motion signal beat the gap alone on the full set, so there is no combined rule to "
            "compare at equal words held."
        )
        return
    cols = [[f["gap"], f[key] or 0.0, 0.0 if f.get(key) is not None else 1.0] for f in feats]
    if all(c[2] == 0.0 for c in cols):
        cols = [c[:2] for c in cols]
    b = logistic_fit([[1.0] + c for c in cols], ys)
    score = [sum(bi * xi for bi, xi in zip(b, [1.0] + c)) for c in cols]
    order = sorted(range(n), key=lambda i: -score[i])
    lines += [
        f"The best of those fits is gap + {label} (AUC {fmt(best[0], 3)} against {fmt(base, 3)} "
        "for the gap alone, both in sample). A host holds a word when the fit's P(revision) clears "
        "a bar; the bar is set here to hold exactly as many words as each gap threshold in the "
        "table above holds, so the two rules are compared at the same count of words held:",
        "",
        "| hold until gap over | words held | revisions caught, gap | revisions caught, fit | "
        "good words delayed, gap | good words delayed, fit |",
        "|---|---|---|---|---|---|",
    ]
    fig["equal_words_held"] = []
    for thr in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        held = [f for f in feats if f["gap"] < thr]
        caught = sum(1 for f in held if not f["survived"])
        by_fit = [feats[i] for i in order[: len(held)]]
        caught_fit = sum(1 for f in by_fit if not f["survived"])
        lines.append(
            f"| {thr:.1f} | {len(held)} ({100 * len(held) / n:.0f}%) | "
            f"{caught} / {revised} ({100 * caught / (revised or 1):.0f}%) | "
            f"{caught_fit} / {revised} ({100 * caught_fit / (revised or 1):.0f}%) | "
            f"{len(held) - caught} / {good} ({100 * (len(held) - caught) / good:.0f}%) | "
            f"{len(by_fit) - caught_fit} / {good} ({100 * (len(by_fit) - caught_fit) / good:.0f}%) |"
        )
        fig["equal_words_held"].append(dict(gap=thr, held=len(held), caught_gap=caught, caught_fit=caught_fit))
    lines += [
        "",
        "Both columns are fit and scored on the same clips, so the fit's is a ceiling rather than "
        "a forecast; the held-out figures are below.",
    ]


def next_advance_section(feats, lines, fig):
    held = [f for f in feats if f["hold"]]
    n = len(feats)
    lines += ["", "## Holding one advance", ""]
    if not held:
        lines.append("No clip ran to another advance after its first sighting.")
        return
    revised = [f for f in held if not f["survived"]]
    changed = sum(1 for f in held if not f["hold"]["word_same"])
    changed_rev = sum(1 for f in revised if not f["hold"]["word_same"])
    changed_good = changed - changed_rev
    silent = sum(1 for f in held if f["hold"]["word_gone"])
    last = sum(1 for f in held if f["advance_index"] + 2 == f["n_advances"])
    waits = [f["hold"]["wait_ms"] for f in held]
    lines += [
        f"A host that waits for the next advance before acting waits {dist(waits)} ms (median / "
        f"p90) and only gets the chance on {len(held)} of {n} first sightings "
        f"({100 * len(held) / n:.1f}%): the rest are at the last advance the clip has. On "
        f"{100 * last / len(held):.0f}% of those the next advance is the clip's own last, so on "
        "this corpus holding one advance is close to waiting for the final. The same signals read "
        "at that advance:",
        "",
        "| signal at the next advance | clips scored | rank-AUC |",
        "|---|---|---|",
    ]
    fig["auc_at_next_advance"] = {}
    for key, label in LEVEL + MOTION:
        sub = [f for f in held if f["hold"].get(key) is not None]
        a = auc([(f["hold"][key], not f["survived"]) for f in sub])
        lines.append(f"| {label} | {len(sub)} | {fmt(a)} |")
        fig["auc_at_next_advance"][key] = dict(n=len(sub), auc=a)
    lines += [
        "",
        "The revision is often already on the page: rank 0 leads with a different first word at "
        f"that advance on {changed_rev} of the {len(revised)} revisions "
        f"({100 * changed_rev / (len(revised) or 1):.0f}%), against {changed_good} of the "
        f"{len(held) - len(revised)} words that survive "
        f"({100 * changed_good / ((len(held) - len(revised)) or 1):.1f}%); rank 0 had fallen back "
        f"to `[sil]` on {silent}. So the hold is nearly a decision by itself, before any signal is "
        "read off it.",
        "",
        "Calibration of the gap read at that advance, against whether the word first shown one "
        "advance earlier is the one the final settles on:",
        "",
        "| gap at the next advance (nats) | clips | survived | predicted `sigmoid(mean gap)` |",
        "|---|---|---|---|",
    ]
    fig["next_advance_calibration"] = []
    for lo, hi in GAP_EDGES:
        b = [f for f in held if f["hold"]["gap"] is not None and lo <= f["hold"]["gap"] < hi]
        if not b:
            continue
        surv = sum(1 for f in b if f["survived"])
        mean_gap = sum(f["hold"]["gap"] for f in b) / len(b)
        hi_label = "8+" if hi == float("inf") else f"{hi:g}"
        lines.append(
            f"| {lo:g} to {hi_label} | {len(b)} | {surv} / {len(b)} ({100 * surv / len(b):.0f}%) | "
            f"{100 * sigmoid(mean_gap):.0f}% |"
        )
        fig["next_advance_calibration"].append(dict(lo=lo, n=len(b), survived=surv, predicted=sigmoid(mean_gap)))
    e, _ = ece(
        [(sigmoid(f["hold"]["gap"]) if f["hold"]["gap"] is not None else 1.0, f["survived"]) for f in held], TRUST_EDGES
    )
    lines += [
        "",
        f"Expected calibration error of `sigmoid(gap)` read one advance late, against survival to "
        f"the final: {fmt(e, 3)}. It is worse than at the first sighting because the gap keeps "
        "growing while the word's fate is already decided.",
    ]
    fig["next_advance"] = dict(
        available=len(held),
        of=n,
        wait_ms_p50=sc.quantile(waits, 0.5),
        wait_ms_p90=sc.quantile(waits, 0.9),
        next_is_last=last,
        revisions=len(revised),
        revisions_visible=changed_rev,
        survivors_changed=changed_good,
        silent=silent,
        ece=e,
    )


def vanishing_section(feats, lines, fig):
    n = len(feats)
    all_ev = [f["vanished_all"] for f in feats]
    before = [f["vanished"] for f in feats]
    with_ev = [f for f in feats if f["vanished"] > 0]
    without = [f for f in feats if f["vanished"] == 0]

    def rate(g):
        r = sum(1 for f in g if not f["survived"])
        return f"{r} / {len(g)} ({100 * r / (len(g) or 1):.1f}%)"

    lines += [
        "",
        "## Readings that vanish while still in contention",
        "",
        f"A reading within {-IN_CONTENTION:g} nats of the leader at one advance and absent at the "
        "next did not decay out of the list, it disappeared from it. Vosk's alternatives group "
        "tokens by word sequence, so two alignments of one reading merge and the loser's evidence "
        "is dropped rather than added (`[[rr:TD-2#The decoder: partial alternatives]]`); beam "
        "pruning also removes readings, so this count is an upper bound on those merges and a "
        "proxy, not a measurement.",
        "",
        f"Over a clip there are {sum(all_ev) / n:.2f} such events on average, "
        f"{dist(all_ev, 0)} by median / p90, and {sum(1 for v in all_ev if v):d} of {n} clips "
        f"({100 * sum(1 for v in all_ev if v) / n:.1f}%) have at least one. Counting only the ones "
        f"visible by the first sighting, {sum(before) / n:.2f} per clip and {len(with_ev)} of {n} "
        f"clips ({100 * len(with_ev) / n:.1f}%) have one.",
        "",
        "| clips | revised |",
        "|---|---|",
        f"| at least one vanishing by the first sighting | {rate(with_ev)} |",
        f"| none | {rate(without)} |",
    ]
    fig["vanishings"] = dict(
        per_clip=sum(all_ev) / n,
        any_clip=sum(1 for v in all_ev if v),
        before_sighting_per_clip=sum(before) / n,
        before_sighting_any=len(with_ev),
        revised_with=sum(1 for f in with_ev if not f["survived"]),
        with_n=len(with_ev),
        revised_without=sum(1 for f in without if not f["survived"]),
        without_n=len(without),
    )


def rules_section(feats, fit_feats, fit_name, lines, fig):
    """The before and the after, every coefficient and every bar fitted on one split and scored on
    another: the README's rule, the two cheaper baselines it is worth beating, a hold that reads
    nothing, the motion, and the motion read one advance later."""
    ys = [not f["survived"] for f in feats]
    fit_ys = [not f["survived"] for f in fit_feats]
    fit_entries = [f["series"][0] for f in fit_feats]
    n = len(feats)
    revised = sum(ys)
    good = n - revised

    coefs = {
        name: fit_cols(fit_entries, fit_ys, cols)
        for name, cols in (
            ("gap", gap_cols),
            ("gap + entropy", gap_entropy_cols),
            ("gap + motion", rule_cols),
            ("gap + entropy + motion", entropy_motion_cols),
        )
    }
    coef = coefs["gap + motion"]
    trust = {
        "R0": gap_trust,
        "B1": fitted_trust(gap_cols, coefs["gap"]),
        "B2": fitted_trust(gap_entropy_cols, coefs["gap + entropy"]),
        "R1": fitted_trust(rule_cols, coef),
        "R1e": fitted_trust(entropy_motion_cols, coefs["gap + entropy + motion"]),
        "R2": fitted_trust(rule_cols, coef),
        "H": always_trust,
    }
    start = {"R0": 0, "B1": 0, "B2": 0, "R1": 0, "R1e": 0, "R2": 1, "H": 1}
    labels = {
        "R0": "R0, `sigmoid(gap)`, the README's rule",
        "B1": "B1, the gap recalibrated",
        "B2": "B2, the gap and the readings' entropy",
        "R1": "R1, the gap and the motion",
        "R1e": "R1e, the gap, the entropy and the motion",
        "R2": "R2, R1 one advance later, as the rule runs",
        "H": "H, hold one advance, reading nothing",
    }
    even = [f for i, f in enumerate(feats) if i % 2 == 0]
    odd = [f for i, f in enumerate(feats) if i % 2 == 1]
    coef_half = fit_rule([f["series"][0] for f in even], [not f["survived"] for f in even])
    coef_in = fit_rule([f["series"][0] for f in feats], ys)

    scores = {k: rule_score(feats, trust[k], start[k], as_run=(k in ("R2", "H"))) for k in trust}
    scores["R2 reading alone"] = rule_score(feats, trust["R2"], 1)
    aucs = {k: auc(list(zip(v, ys))) for k, v in scores.items()}
    a0, a1, a2, a2r = aucs["R0"], aucs["R1"], aucs["R2"], aucs["R2 reading alone"]

    lines += [
        "",
        "## Six rules on the same clips, every one fitted off them",
        "",
        "The sections above are in sample. Here are the rules a host could run, with every "
        f"coefficient and every operating point fitted on {fit_name} and every figure scored on "
        "the testing split:",
        "",
        "- **R0**, the README's rule: `trust = sigmoid(gap)` at the first sighting, and 1.0 when "
        "the list holds one reading, exactly as the README's `trust()` reads it. Nothing is fit.",
        "- **B1**, the same gap through a logistic fit, which is the recalibration alone and the "
        "baseline any new field has to beat.",
        "- **B2**, the gap and the entropy of the readings' softmax: the strongest second reading "
        "a host already has, and no runtime change.",
        "- **R1**, the gap and the motion available at the first sighting: the displaced reading's "
        "`lead_delta` and rank 0's, each missing value zeroed beside an indicator.",
        "- **R1e**, the same with the entropy beside it, which is the strongest rule on this page "
        "that a host could run at the sighting.",
        "- **R2**, R1 read at the first advance after the sighting, charged the wait.",
        "- **H**, the wait with no reading at all: release at the first advance where rank 0 still "
        "leads with the word. It separates what the delay buys from what the motion buys.",
        "",
        "| rule | clips scored | AUC against revision |",
        "|---|---|---|",
    ]
    for k in ("R0", "B1", "B2", "R1", "R1e", "R2", "H"):
        lines.append(f"| {labels[k]} | {sum(1 for x in scores[k] if x is not None)} | {fmt(aucs[k], 3)} |")
    lines.append(
        f"| R2, the later reading alone | {sum(1 for x in scores['R2 reading alone'] if x is not None)} | "
        f"{fmt(a2r, 3)} |"
    )
    fig["rules"] = dict(
        fit_on=fit_name,
        coefficients=coef,
        coefficients_by_rule=coefs,
        coefficients_split_half=coef_half,
        coefficients_in_sample=coef_in,
    )
    fig["rules"]["auc"] = dict(aucs)
    lines += [
        "",
        "B1 orders the clips exactly as R0 does, a logistic of one column being monotone in it, so "
        "its AUC difference is zero by construction and everything it buys is in the calibration "
        "and the decisions below. Scores are oriented as P(revision), so higher is better and 0.5 "
        "is no information. R2 and "
        "H are scored only on the clips that have another advance, and both are read as they run: "
        "a word rank 0 no longer leads with is a revision called with certainty, which is where "
        "nearly all of their information is. H carries no score of its own, so its AUC is that "
        "call and nothing else. Scoring R2's later reading on its own gap and motion, as if the "
        "word were still on offer, is worse than R0 because the gap keeps growing after the word's "
        f"fate is settled. The paired bootstrap resamples clips {BOOTSTRAP} times, both rules "
        "seeing the same resample:",
        "",
        "| difference | clips | AUC difference | 95% interval |",
        "|---|---|---|---|",
    ]
    fig["rules"]["bootstrap"] = {}
    pairs = (
        ("B1 - R0, the recalibration alone", "R0", "B1"),
        ("B2 - B1, the entropy over it", "B1", "B2"),
        ("R1 - R0, the motion against the README", "R0", "R1"),
        ("R1 - B1, the motion over the recalibration", "B1", "R1"),
        ("R1 - B2, the motion alone against the entropy alone", "B2", "R1"),
        ("R1e - B2, the motion added to the entropy", "B2", "R1e"),
        ("R2 - R0, as the rule runs", "R0", "R2"),
        ("H - R0, the wait alone", "R0", "H"),
    )
    for name, a, b in pairs:
        d, lo, hi, m = paired_bootstrap(scores[a], scores[b], ys)
        lines.append(f"| {name} | {m} | {fmt(d, 4)} | {fmt(lo, 4)} to {fmt(hi, 4)} |")
        fig["rules"]["bootstrap"][name] = dict(n=m, diff=d, lo=lo, hi=hi)
    half_auc = auc(list(zip(rule_score(odd, fitted_trust(rule_cols, coef_half), 0), [not f["survived"] for f in odd])))
    in_auc = auc(list(zip(rule_score(feats, fitted_trust(rule_cols, coef_in), 0), ys)))
    lines += [
        "",
        f"Where the R1 coefficients come from moves it little: fit on {fit_name} it scores "
        f"{fmt(a1, 3)} on all {n} testing clips; fit on the even-indexed half of testing it scores "
        f"{fmt(half_auc, 3)} on the odd half; fit and scored on testing itself, {fmt(in_auc, 3)}. "
        "The fitted coefficients, each in the order named:",
        "",
        "| rule | order | coefficients |",
        "|---|---|---|",
    ]
    for name, c in coefs.items():
        lines.append(f"| {name} | {COEF_ORDER[name]} | " + ", ".join(f"{x:+.4f}" for x in c) + " |")
    lines += [
        "",
        "### Calibration, and the error in it",
        "",
        "Each rule's predicted trust against the survival observed, in bins of the prediction. "
        "Expected calibration error is the bin-weighted distance between the two columns:",
        "",
        "| rule | bin | clips | mean predicted | survived |",
        "|---|---|---|---|---|",
    ]
    fig["rules"]["calibration"] = {}
    for name in ("R0", "B1", "B2", "R1", "R1e", "R2"):
        pr = [(1.0 - p, f["survived"]) for p, f in zip(scores[name], feats) if p is not None]
        e, bins = ece(pr, TRUST_EDGES)
        for (lo, hi), m, mean_p, obs in bins:
            hi_label = "1" if hi > 1 else f"{hi:g}"
            lines.append(f"| {name} | {lo:g} to {hi_label} | {m} | {100 * mean_p:.0f}% | {100 * obs:.0f}% |")
        lines.append(f"| {name} | **expected calibration error** | {len(pr)} | | **{fmt(e, 3)}** |")
        fig["rules"]["calibration"][name] = dict(
            ece=e, bins=[[lo, hi, m, mean_p, obs] for (lo, hi), m, mean_p, obs in bins]
        )
    lines += [
        "",
        "The same calibration read in the gap's own buckets, the ones the table at the top of the "
        "page uses, so a rule that is better ordered but worse calibrated shows as one:",
        "",
        "| gap at the sighting (nats) | clips | survived | R0 predicts | B1 predicts | B2 predicts | R1 predicts | R1e predicts | R2 predicts |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    shown = ("R0", "B1", "B2", "R1", "R1e", "R2")
    errs = {k: 0.0 for k in shown}
    scored = {k: sum(1 for x in scores[k] if x is not None) for k in shown}
    fig["rules"]["gap_bucket_calibration"] = []
    for lo, hi in GAP_EDGES:
        idx = [i for i, f in enumerate(feats) if lo <= f["gap"] < hi]
        if not idx:
            continue
        row = dict(lo=lo, hi=None if hi == float("inf") else hi, n=len(idx))
        cells = []
        for name in shown:
            have = [i for i in idx if scores[name][i] is not None]
            if not have:
                cells.append("-")
                continue
            pred = sum(1.0 - scores[name][i] for i in have) / len(have)
            obs = sum(1 for i in have if feats[i]["survived"]) / len(have)
            errs[name] += len(have) / scored[name] * abs(pred - obs)
            cells.append(f"{100 * pred:.0f}%")
            row[name] = dict(n=len(have), predicted=pred, observed=obs)
        surv = sum(1 for i in idx if feats[i]["survived"])
        hi_label = "8+" if hi == float("inf") else f"{hi:g}"
        lines.append(
            f"| {lo:g} to {hi_label} | {len(idx)} | {surv} / {len(idx)} ({100 * surv / len(idx):.0f}%) | "
            + " | ".join(cells)
            + " |"
        )
        fig["rules"]["gap_bucket_calibration"].append(row)
    lines += [
        "",
        "Expected calibration error over those buckets, each rule weighted by the clips it "
        "scores: "
        + ", ".join(f"{k} {v:.3f} on {scored[k]} clips" for k, v in errs.items())
        + ". R2's bucket is the gap at the sighting, not at the advance it reads, so its column is "
        "what a host holding that word would have been told one advance later.",
        "",
        "### The delay each rule charges",
        "",
        "A rule holds the word it was shown until its trust clears the bar, re-reading the "
        "partial at every advance; it can only release the word it is holding, so an advance where "
        "rank 0 leads with something else never clears. A revision the rule never releases is "
        "caught. A good word it never releases is delayed to the final, and charged the audio from "
        "the sighting to the end of the clip. The delay columns are milliseconds of audio from the "
        "first sighting, over the good words only:",
        "",
        "| rule | trust bar | revisions caught | good words: mean ms | median ms | p90 ms | never released |",
        "|---|---|---|---|---|---|---|",
    ]
    fig["rules"]["delay"] = {}
    runs = {}
    for name in ("R0", "B1", "B2", "R1", "R1e", "R2"):
        fig["rules"]["delay"][name] = []
        for bar in BARS:
            r = rule_run(feats, trust[name], bar, start[name])
            runs[(name, bar)] = r
            good_never = sum(1 for f, ok in zip(feats, r["correct"]) if f["survived"] and not ok)
            lines.append(
                f"| {name} | {bar:.2f} | {r['caught']} / {revised} ({100 * r['caught'] / revised:.0f}%) | "
                f"{mean(r['delays']):.0f} | {sc.quantile(r['delays'], 0.5):.0f} | "
                f"{sc.quantile(r['delays'], 0.9):.0f} | "
                f"{good_never} / {good} ({100 * good_never / good:.0f}%) |"
            )
            fig["rules"]["delay"][name].append(
                dict(
                    bar=bar,
                    caught=r["caught"],
                    revised=revised,
                    delay_mean=mean(r["delays"]),
                    delay_p50=sc.quantile(r["delays"], 0.5),
                    delay_p90=sc.quantile(r["delays"], 0.9),
                    good_never_released=good_never,
                    good=good,
                    correct=sum(1 for c in r["correct"] if c),
                )
            )
    runs[("H", 1.0)] = rule_run(feats, always_trust, 1.0, 1)
    lines += [
        "",
        f"At the README's operating point, trust {OPERATING_TRUST:g} (a gap of about 2.2 nats for "
        "R0), the same rules and an exact McNemar on the clips they handle differently. A clip is "
        "handled correctly when a revision is held to the final or a good word is released before "
        "it. H has no bar; it is the wait:",
        "",
        "| rule | clips | correct | revisions caught | good words: mean ms | median ms | p90 ms | never released |",
        "|---|---|---|---|---|---|---|---|",
    ]
    base = runs[("R0", OPERATING_TRUST)]
    fig["rules"]["operating_point"] = dict(trust=OPERATING_TRUST)
    can_act = [f for f in feats if f["hold"]]
    rows = [(name, feats, runs[(name, OPERATING_TRUST)]) for name in ("R0", "B1", "B2", "R1", "R1e", "R2")]
    rows.append(("H", feats, runs[("H", 1.0)]))
    rows.append(("R2, where it can act", can_act, rule_run(can_act, trust["R2"], OPERATING_TRUST, 1)))
    for name, on, r in rows:
        ok = sum(1 for c in r["correct"] if c)
        rev = sum(1 for f in on if not f["survived"])
        good_on = len(on) - rev
        never_good = sum(1 for f, c in zip(on, r["correct"]) if f["survived"] and not c)
        lines.append(
            f"| {name} | {len(on)} | {ok} / {len(on)} ({100 * ok / len(on):.1f}%) | "
            f"{r['caught']} / {rev} ({100 * r['caught'] / (rev or 1):.0f}%) | "
            f"{mean(r['delays']):.0f} | {sc.quantile(r['delays'], 0.5):.0f} | "
            f"{sc.quantile(r['delays'], 0.9):.0f} | "
            f"{never_good} / {good_on} ({100 * never_good / (good_on or 1):.0f}%) |"
        )
        fig["rules"]["operating_point"][name] = dict(
            clips=len(on),
            correct=ok,
            caught=r["caught"],
            revised=rev,
            delay_mean=mean(r["delays"]),
            delay_p50=sc.quantile(r["delays"], 0.5),
            delay_p90=sc.quantile(r["delays"], 0.9),
            good_never_released=never_good,
        )
    lines += [
        "",
        "| pair | R0 right, other wrong | other right, R0 wrong | exact McNemar p |",
        "|---|---|---|---|",
    ]
    for name in ("B1", "B2", "R1", "R1e", "R2"):
        r = runs[(name, OPERATING_TRUST)]
        b = sum(1 for x, y in zip(base["correct"], r["correct"]) if x and not y)
        c = sum(1 for x, y in zip(base["correct"], r["correct"]) if y and not x)
        pv = sc.mcnemar(b, c)
        lines.append(f"| R0 against {name} | {b} | {c} | {'<1e-300' if pv == 0.0 else f'{pv:.3g}'} |")
        fig["rules"]["operating_point"][f"mcnemar_R0_{name}"] = dict(b=b, c=c, p=pv)
    r1_op = runs[("R1", OPERATING_TRUST)]
    b1_op = runs[("B1", OPERATING_TRUST)]
    b = sum(1 for x, y in zip(b1_op["correct"], r1_op["correct"]) if x and not y)
    c = sum(1 for x, y in zip(b1_op["correct"], r1_op["correct"]) if y and not x)
    pv = sc.mcnemar(b, c)
    lines.append(f"| B1 against R1, the motion's own share | {b} | {c} | {'<1e-300' if pv == 0.0 else f'{pv:.3g}'} |")
    fig["rules"]["operating_point"]["mcnemar_B1_R1"] = dict(b=b, c=c, p=pv)

    # The frozen operating points: the targets come from the fitting split, the figures from the
    # scored one, so the bar is never chosen on the clips that judge it.
    fit_base = rule_run(fit_feats, gap_trust, OPERATING_TRUST, 0)
    fit_target_delay = mean(fit_base["delays"])
    fit_target_caught = fit_base["caught"] / max(1, len(fit_feats))
    frozen = {}
    lines += [
        "",
        "### The frozen operating points",
        "",
        f"R0 at the README's bar charges a mean {fit_target_delay:.0f} ms on {fit_name}'s good "
        f"words and catches {fit_base['caught']} of its {sum(1 for f in fit_feats if not f['survived'])} "
        "revisions there. Each other rule's bar is the one that comes closest to *that* on *that* "
        "split, frozen, and then read on the testing clips. The intervals are 95% over clip "
        f"resamples ({BOOTSTRAP} of them):",
        "",
        "| rule | matched on | bar frozen on the fit | revisions caught | 95% interval | good words: mean ms | 95% interval |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in ("B1", "B2", "R1", "R1e", "R2"):
        for kind, target in (("delay", fit_target_delay), ("catch", fit_target_caught)):
            bar = frozen_bar(fit_feats, trust[name], start[name], target, kind)
            r = rule_run(feats, trust[name], bar, start[name])
            iv = run_interval(r)
            frozen[f"{name}|{kind}"] = dict(bar=bar, caught=r["caught"], **iv)
            lines.append(
                f"| {name} | equal {kind} | {bar:.3f} | {r['caught']} / {revised} "
                f"({100 * iv['caught_share']:.1f}%) | {100 * iv['caught_lo']:.1f}% to "
                f"{100 * iv['caught_hi']:.1f}% | {iv['delay_mean']:.0f} | {iv['delay_lo']:.0f} to "
                f"{iv['delay_hi']:.0f} |"
            )
    r0_iv = run_interval(base)
    lines.append(
        f"| R0 | the reference, trust {OPERATING_TRUST:g} | - | {base['caught']} / {revised} "
        f"({100 * r0_iv['caught_share']:.1f}%) | {100 * r0_iv['caught_lo']:.1f}% to "
        f"{100 * r0_iv['caught_hi']:.1f}% | {r0_iv['delay_mean']:.0f} | {r0_iv['delay_lo']:.0f} to "
        f"{r0_iv['delay_hi']:.0f} |"
    )
    fig["rules"]["frozen"] = dict(
        fit_delay_mean=fit_target_delay, fit_caught_share=fit_target_caught, R0=r0_iv, points=frozen
    )
    grid = [i / 200 for i in range(1, 200)]
    fine = [(bar, rule_run(feats, trust["R1"], bar, 0)) for bar in grid]
    same_delay = min(fine, key=lambda t: abs(mean(t[1]["delays"]) - mean(base["delays"])))
    same_caught = min(fine, key=lambda t: (abs(t[1]["caught"] - base["caught"]), mean(t[1]["delays"])))
    lines += [
        "",
        "For comparison, and **descriptive only**, the same two matches made by scanning R1's bar "
        "on the testing clips themselves, which is the curve and not an operating point a host "
        f"could have chosen: against R0's {base['caught']} of {revised} for a mean "
        f"{mean(base['delays']):.0f} ms, R1 at {same_delay[0]:.3f} catches "
        f"{same_delay[1]['caught']} for {mean(same_delay[1]['delays']):.0f} ms, and at "
        f"{same_caught[0]:.3f} catches {same_caught[1]['caught']} for "
        f"{mean(same_caught[1]['delays']):.0f} ms. A bar chosen where it is judged flatters itself; "
        "the frozen table above is the one to read.",
    ]
    fig["rules"]["matched"] = dict(
        descriptive=True,
        R0_caught=base["caught"],
        R0_delay_mean=mean(base["delays"]),
        equal_delay=dict(bar=same_delay[0], caught=same_delay[1]["caught"], delay_mean=mean(same_delay[1]["delays"])),
        equal_caught=dict(
            bar=same_caught[0], caught=same_caught[1]["caught"], delay_mean=mean(same_caught[1]["delays"])
        ),
    )
    last = sum(1 for f in can_act if f["advance_index"] + 2 == f["n_advances"])
    lines += [
        "",
        "R2 looks near-perfect where it can act, and the corpus is why: on "
        f"{100 * last / (len(can_act) or 1):.0f}% of the clips that have another advance at all, "
        "that advance is the clip's last, so rank 0 there is all but the final and the rule is "
        "reading the answer rather than predicting it. H is the same wait with nothing read off "
        "it, which is how much of R2 is the wait. What R2 measures on one-second clips is the "
        "value of waiting, not the value of the motion, and the wait it charges is the whole "
        "remainder of the utterance.",
    ]
    return coef, a0, a1, a2, a2r, runs


def reading_section(feats, best, base, strat, diag, rules, lines, fig):
    n = len(feats)
    held = [f for f in feats if f["hold"]]
    revised = [f for f in held if not f["survived"]]
    caught = sum(1 for f in revised if not f["hold"]["word_same"])
    last = sum(1 for f in held if f["advance_index"] + 2 == f["n_advances"])
    hist = 100 * sum(1 for f in feats if f["had_history"]) / n
    gain = (best[0] - base) if best[1] else 0.0
    gap_dev = abs(auc([(f["gap"], not f["survived"]) for f in feats]) - 0.5)
    with_ev = [f for f in feats if f["vanished"] > 0]
    without = [f for f in feats if f["vanished"] == 0]
    rev_ev = 100 * sum(1 for f in with_ev if not f["survived"]) / (len(with_ev) or 1)
    rev_no = 100 * sum(1 for f in without if not f["survived"]) / (len(without) or 1)
    _, a0, a1, a2, a2r, runs = rules
    op0 = runs[("R0", OPERATING_TRUST)]
    op1 = runs[("R1", OPERATING_TRUST)]
    op2 = runs[("R2", OPERATING_TRUST)]
    op_r0 = sum(1 for c in op0["correct"] if c)
    op_r1 = sum(1 for c in op1["correct"] if c)
    op_b1 = sum(1 for c in runs[("B1", OPERATING_TRUST)]["correct"] if c)
    op_b1_caught = runs[("B1", OPERATING_TRUST)]["caught"]
    top = []
    for key, label in MOTION:
        sub = defined(feats, key)
        a = auc([(f[key], not f["survived"]) for f in sub])
        if a is not None:
            top.append((abs(a - 0.5), a, len(sub), label))
    top.sort(reverse=True)
    lines += [
        "",
        "## What the motion figures say",
        "",
        "The motion is mostly not there to read. A word's first sighting is the birth of its "
        f"reading: rank 0 was present at the previous advance on {hist:.1f}% of first sightings, "
        "and every delta needs that. The deltas the hypothesis is about are therefore undefined on "
        "the majority of the moments a host would want them, which is a fact about the corpus as "
        "much as about the decoder: at 240 ms an advance and a second of audio there are three "
        "advances in a clip, and a word tends to appear at the second or the third.",
        "",
        f"Where a signal is defined it ranks revisions: the strongest is {top[0][3]} at "
        f"{top[0][1]:.2f} on {top[0][2]} first sightings, against the gap's own "
        f"{0.5 - gap_dev:.2f}. But the level is most of what those signals are reading. Held at "
        "equal gap the columns fall towards 0.5"
        + (
            f", the largest average distance left being {strat[0][2]} at {strat[0][0]:.2f} over the "
            "buckets with 50 clips or more"
            if strat
            else ""
        )
        + f", and one signal at a time in sample the AUC moves from {fmt(base, 3)} to at best "
        f"{fmt(best[0], 3)} ({gain:+.3f}).",
        "",
        f"Held out, the whole motion set moves the AUC from {fmt(a0, 3)} for the README's rule to "
        f"{fmt(a1, 3)}, and from {fmt(fig['rules']['auc']['B1'], 3)} for the same gap recalibrated "
        f"and {fmt(fig['rules']['auc']['B2'], 3)} for the gap with the readings' entropy beside it; "
        "the bootstrap table above gives an interval on each of those differences. Most of the "
        "distance from the README's rule is the recalibration, which costs nothing and needs no "
        f"field: at the README's bar the recalibrated gap alone decides {op_b1} of {n} clips "
        f"correctly where the README's rule decides {op_r0}, and the motion decides {op_r1}. "
        f"Against the recalibration the motion is {op_r1 - op_b1:+d} clips, "
        f"{fig['rules']['operating_point']['mcnemar_B1_R1']['c']} to "
        f"{fig['rules']['operating_point']['mcnemar_B1_R1']['b']} on the clips they disagree "
        f"about, exact McNemar {fig['rules']['operating_point']['mcnemar_B1_R1']['p']:.2g}, and it "
        f"catches {op1['caught'] - op_b1_caught:+d} revisions in the exchange. That is a trade "
        "made at one bar, not a safety gain: the frozen table above is where the two are compared "
        "at the same delay and the same catch, each bar chosen on the fitting split.",
        "",
        "What buys more is the delay, not a new field. Waiting for the next advance costs "
        f"{dist([f['hold']['wait_ms'] for f in held]) if held else '-'} ms, is available on "
        f"{100 * len(held) / n:.1f}% of first sightings, and on the ones where it is available "
        f"rank 0 has already changed its first word on {100 * caught / (len(revised) or 1):.0f}% of "
        f"the revisions. Read as a rule that is R2: {fmt(a2, 3)} AUC on the clips it can score, "
        f"{op2['caught']} revisions caught at the operating point, for a median "
        f"{sc.quantile(op2['delays'], 0.5):.0f} ms on the good words. That number is flattered by "
        f"the corpus: on {100 * last / (len(held) or 1):.0f}% of those clips the next advance is "
        "the last one the clip has, so the hold is nearly the wait for the final.",
        "",
        "The vanishing proxy is common: a reading inside two nats of the leader is gone at the next "
        f"advance on {100 * len(with_ev) / n:.1f}% of clips before the first word even appears. So "
        "the alternatives are a bounded sample of the beam whose mass is not conserved, which is "
        "the reason to expect the gap to be approximate rather than a posterior. It is not a "
        f"warning sign for the word that follows: those clips revise at {rev_ev:.1f}% against "
        f"{rev_no:.1f}% for clips with none"
        + (
            ", the opposite direction to the obvious guess, because a leader decisive enough to "
            "drop its rivals is a leader that holds."
            if rev_ev < rev_no
            else "."
        ),
        "",
        f"Caveats. Every advance interval on the corpus measured {sc.quantile(diag['intervals'], 0.5):.0f} ms, "
        "as `[[rr:TD-7#Decision outcome]]` and the chunk of `[[rr:TD-2#The network]]` say it "
        f"should. {diag['gaps_after_start']} blocks after a clip's first advance carried no "
        f"readings at all, and {diag['sighting_off_advance']} first sightings fell on a block that "
        "was not an advance. The merge count is a proxy: beam pruning removes readings too, and "
        "nothing in the output distinguishes the two. One clip decoded eight times in fresh "
        "processes gave identical readings and identical finals, so these figures are a property "
        "of the build and not of a run; the build is the one the page was measured with, and an "
        "older wheel that ordered two readings of equal cost by hash moved a few of the finals. A "
        "corpus of sentences would give the motion the room a one-second clip does not, and is "
        "where this should be measured again.",
        "",
    ]


def build_page(feats, diag, fit_feats, fit_name, split, block_ms, alternatives, fig):
    n = len(feats)
    revised = sum(1 for f in feats if not f["survived"])
    good = n - revised
    lines = [
        "# Reading a partial's trust",
        "",
        sc.provenance_line(diag["provenance"]),
        "",
        f"utterpy over the Speech Commands {split} split, {n} clips whose first word appeared in a "
        f"partial that carried a runner-up, 92-entry grammar, {block_ms} ms blocks, "
        f"{alternatives} alternatives, every coefficient and bar fitted on {fit_name}. Measured by "
        "`scripts/partial_trust.py`.",
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
    fig["calibration"] = []
    for lo, hi in GAP_EDGES:
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
        fig["calibration"].append(dict(lo=lo, n=len(b), survived=surv, predicted=sigmoid(mean_gap)))
    lines += [
        "",
        "Survival climbs with the gap and tracks the softmax, so the sigmoid is a usable trust "
        "score and a threshold on it is a trust gate. In terms a host acts on, holding a word until "
        "the gap clears a threshold:",
        "",
        "| hold until gap over | words held | revisions caught | good words delayed |",
        "|---|---|---|---|",
    ]
    fig["hold_until_gap"] = []
    for thr in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        held = [f for f in feats if f["gap"] < thr]
        caught = sum(1 for f in held if not f["survived"])
        delayed = sum(1 for f in held if f["survived"])
        lines.append(
            f"| {thr:.1f} | {len(held)} ({100 * len(held) / n:.0f}%) | "
            f"{caught} / {revised} ({100 * caught / (revised or 1):.0f}%) | "
            f"{delayed} / {good} ({100 * delayed / good:.0f}%) |"
        )
        fig["hold_until_gap"].append(dict(gap=thr, held=len(held), caught=caught, delayed=delayed))
    lines += [
        "",
        "So the number to read is not a word's own confidence but its lead over the next reading. A "
        "host that waits for `sigmoid(gap)` to clear its bar trades a little latency on the words it "
        "holds for catching most of the revisions it would otherwise have acted on. The Python is "
        "in the README, under the partial example.",
        "",
    ]
    fig["first_sightings"] = n
    fig["revised"] = revised
    fig["auc_headline"] = {k: auc([(f[k], not f["survived"]) for f in feats]) for k, _ in LEVEL}

    availability_section(feats, diag, lines, fig)
    auc_section(feats, lines, fig)
    strat = stratified_section(feats, lines, fig)
    best, base = logistic_section(feats, lines, fig)
    hold_section(feats, best, base, lines, fig)
    next_advance_section(feats, lines, fig)
    vanishing_section(feats, lines, fig)
    rules = rules_section(feats, fit_feats, fit_name, lines, fig)
    reading_section(feats, best, base, strat, diag, rules, lines, fig)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--model")
    ap.add_argument(
        "--engine", default="utterpy", help="utterpy, or utterpy@MS for a host endpoint rule at MS of trailing silence"
    )
    ap.add_argument("--out", default="docs/benchmarks/partial-trust.md")
    ap.add_argument("--clips", default=None, help="write the per-clip records as JSON lines")
    ap.add_argument("--from-clips", default=None, help="read the records instead of decoding")
    ap.add_argument("--fit-clips", default=None, help="records to fit the rules on; default is the scored set")
    ap.add_argument("--json", default=None, help="write every figure as JSON")
    ap.add_argument("--split", default="testing", choices=("testing", "validation"))
    ap.add_argument("--block-ms", type=int, default=40)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--alternatives", type=int, default=5)
    a = ap.parse_args()

    if a.from_clips:
        feats = load(a.from_clips)
        diag = json.loads(Path(a.from_clips + ".diag.json").read_text())
    else:
        import utterpy

        data = Path(a.data)
        listed = [x.strip() for x in (data / f"{a.split}_list.txt").read_text().splitlines() if x.strip()]
        by = defaultdict(list)
        for rel in listed:
            by[rel.split("/")[0]].append(rel)
        clips = [data / rel for w in sc.DATASET_WORDS for rel in by.get(w, [])][:: a.stride]
        grammar = sc.DATASET_WORDS + sc.LETTERS + sc.NATO + sc.COLOURS
        eng = sc.Engine(a.engine, utterpy, a.model, grammar)
        recs, diag = decode(clips, eng, a.block_ms, a.alternatives)
        diag["provenance"] = sc.provenance(a, ["utterpy"])
        feats = [promote(r) for r in recs]
        if a.clips:
            with open(a.clips, "w") as fh:
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
            Path(a.clips + ".diag.json").write_text(json.dumps(diag))

    fit_feats = load(a.fit_clips) if a.fit_clips else feats
    fit_name = "the testing split itself, in sample"
    if a.fit_clips:
        name = Path(a.fit_clips).name
        fit_name = "the validation split" if "validation" in name else name
    fig = dict(
        run=time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        split=a.split,
        block_ms=a.block_ms,
        alternatives=a.alternatives,
        clips_file=a.from_clips or a.clips,
        fit_on=fit_name,
        fit_clips=len(fit_feats),
        provenance=diag.get("provenance"),
    )
    lines = build_page(feats, diag, fit_feats, fit_name, a.split, a.block_ms, a.alternatives, fig)
    Path(a.out).write_text("\n".join(lines))
    if a.json:
        Path(a.json).write_text(json.dumps(fig, indent=2, default=str))
    print(f"wrote {a.out} from {len(feats)} clips of {diag['clips']}", file=sys.stderr)


if __name__ == "__main__":
    main()
