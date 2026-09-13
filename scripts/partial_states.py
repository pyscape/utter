#!/usr/bin/env python3
"""Silence direction and word transitions, read off the partial's readings, on a built stream.

    python scripts/partial_states.py --data DIR --model DIR [--out FILE] [--words N]

A Speech Commands clip carries one word, so it carries no transition and no finish. This script
builds a stream instead: utterances of one to five clips, pauses of 100-800 ms between the words
of an utterance, finishes of 2-3 s between utterances, every gap cut in rotation from the
dataset's own six background recordings and scaled to about -50 dBFS RMS rather than to digital
zeros, which the floor tracker treats apart (`[[rr:TD-8#The runtime reports the floor]]`). The
stream is built, so every word's position and every gap's length are exact, and each word's
onset and offset come from the clip's own energy envelope, owing nothing to either engine.

Two questions, both about what the readings say before rank 0 moves:

A. Silence direction. At every block the ground truth says whether the next W ms cross into
   silence, out of it, or neither. R0 is TD-8's reading of the fields, the trailing `[sil]`
   span past a bound and a word arriving at rank 0
   (`[[rr:TD-8#End of speech is the endpoint, and the trailing silence entry is its clock]]`);
   R1 is R0 plus the readings' motion, the empty reading's `lead_delta` at an utterance's start
   and an extending reading's mid-utterance
   (`[[rr:TD-9#Every reading carries its lead's motion]]`). Scored per state at each W, by lead
   time per true transition, by false alarms per minute on the recordings and on finishes, and
   by the end-of-speech confusion curve TD-8's private figures give the shape of.
B. Word transitions. Per ground-truth transition, whether a reading extending rank 0 by one
   word appears before that word reaches rank 0, by how many advances and milliseconds, how
   often its extra word is the right one, and what the `[sil]` entries measure meanwhile.
C. Whether any of that calls a word before it is spoken. Exploratory: the README's rule against
   candidate presence, candidate level, elapsed silence and a clock the host keeps, each fitted
   to its own best F1 and again to the same false alarm rate, with every call matched to at most
   one word and the calls made before any sample of that word's clip reported apart, since those
   carry no acoustic evidence of it.

Held out as `[[rr:TD-9#The benchmark is paired, held out, and charged in milliseconds]]` asks:
every threshold is fitted on streams built from the validation split and every reported figure
comes from streams built from the testing split, with the same seed so the streams are
reproducible. Writes `<out>.md`, `<out>.json` (every figure) and `<out>.streams.jsonl` (per
stream the ground truth, the finals and the whole advance series, so a rule can be rescored
without decoding again).
"""

import argparse
import array
import bisect
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import speech_commands as sc  # noqa: E402

RATE = sc.RATE
SAMPLES_PER_MS = RATE // 1000

GAP_FLOOR_DBFS = -50.0
PAUSE_MS = (100, 800)
FINISH_MS = (2000, 3000)
WORDS_PER_UTTERANCE = (1, 5)
UTTERANCES_PER_STREAM = 10

HORIZONS = (240, 500)
STATES = ("away", "toward", "maintaining silence", "maintaining speech")

BOUND_GRID = (200, 300, 400, 500, 600, 700, 800, 1000)
SIL_DELTA_GRID = (0.0, -0.25, -0.5, -1.0, -2.0, -4.0, -8.0)
EXT_DELTA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
TD8_BOUND_MS = 300  # the bound TD-8 quotes its private figures at

EOS_GRID = (100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1500)
STABLE_BOUND_MS = 200
BOOTSTRAPS = 1000


# --- ground truth -----------------------------------------------------------------------------


def energy_start(pcm, frame_ms=10, drop_db=20.0):
    """Start, in seconds, of the first frame within `drop_db` of the clip's loudest frame: the
    mirror of sc.energy_end, which gives the last."""
    a = array.array("h")
    a.frombytes(pcm)
    n = RATE * frame_ms // 1000
    levels = []
    for i in range(0, len(a) - n + 1, n):
        acc = 0
        for v in a[i : i + n]:
            acc += v * v
        levels.append(10 * math.log10(acc / n + 1e-9))
    if not levels:
        return None
    peak = max(levels)
    first = min(i for i, level in enumerate(levels) if level >= peak - drop_db)
    return first * n / RATE


class GapSource:
    """Background audio for the gaps, taken from the recordings in rotation and scaled to a fixed
    floor. Each recording keeps its own cursor, so no two gaps cut from one recording overlap
    until it wraps."""

    def __init__(self, paths, floor_dbfs=GAP_FLOOR_DBFS):
        self.names = [p.name for p in paths]
        self.audio = []
        for p in paths:
            a = array.array("h")
            a.frombytes(sc.read_pcm(p))
            self.audio.append(a)
        self.cursor = [0] * len(paths)
        self.turn = 0
        self.target = 32768.0 * 10.0 ** (floor_dbfs / 20.0)

    def take(self, n):
        i = self.turn % len(self.audio)
        self.turn += 1
        a = self.audio[i]
        c = self.cursor[i]
        seg = [a[(c + k) % len(a)] for k in range(n)]
        self.cursor[i] = (c + n) % len(a)
        rms = math.sqrt(sum(float(v) * v for v in seg) / len(seg)) if seg else 0.0
        gain = self.target / rms if rms > 0 else 0.0
        out = array.array("h", [0]) * n
        for k in range(n):
            v = int(round(gain * seg[k]))
            out[k] = -32768 if v < -32768 else (32767 if v > 32767 else v)
        return out.tobytes(), self.names[i]


def build_stream(rng, clips, gaps, utterances, pause_ms=PAUSE_MS):
    """One stream and its truth: the PCM, each word with its position and energy span, each gap
    with its kind and length. The stream ends with a finish gap so the last word has one."""
    parts = []
    words = []
    gap_rows = []
    pos = 0
    for u in range(utterances):
        n = rng.randint(*WORDS_PER_UTTERANCE)
        for k in range(n):
            label, path = clips.pop()
            pcm = sc.read_pcm(path)
            start = energy_start(pcm)
            end = sc.energy_end(pcm)
            n_samples = len(pcm) // 2
            words.append(
                dict(
                    label=label,
                    clip=f"{path.parent.name}/{path.name}",
                    utt=u,
                    index=k,
                    pos=pos,
                    samples=n_samples,
                    onset=pos + (int(start * RATE) if start is not None else 0),
                    offset=pos + (int(end * RATE) if end is not None else n_samples),
                )
            )
            parts.append(pcm)
            pos += n_samples
            last = k == n - 1
            ms = rng.uniform(*(FINISH_MS if last else pause_ms))
            g = int(ms * SAMPLES_PER_MS)
            audio, src = gaps.take(g)
            parts.append(audio)
            gap_rows.append(
                dict(kind="finish" if last else "pause", after=len(words) - 1, pos=pos, samples=g, source=src)
            )
            pos += g
    return b"".join(parts), words, gap_rows


class Truth:
    """Speech is the union of the words' energy spans; everything else, the gaps and the quiet
    heads and tails of the clips themselves, is silence."""

    def __init__(self, words):
        self.iv = [(w["onset"], w["offset"]) for w in words]
        self.starts = [a for a, _ in self.iv]

    def speech_at(self, s):
        i = bisect.bisect_right(self.starts, s) - 1
        return i >= 0 and s < self.iv[i][1]

    def any_speech(self, a, b):
        i = bisect.bisect_right(self.starts, b) - 1
        return i >= 0 and self.iv[i][1] > a

    def any_silence(self, a, b):
        i = bisect.bisect_right(self.starts, a) - 1
        return not (i >= 0 and self.iv[i][0] <= a and self.iv[i][1] >= b)

    def forecast(self, fed, window):
        """The state of the next `window` samples after the audio fed, from the truth alone."""
        a = fed - 1
        b = a + window
        if self.speech_at(a):
            return "toward" if self.any_silence(a, b) else "maintaining speech"
        return "away" if self.any_speech(a, b) else "maintaining silence"


# --- the readings -----------------------------------------------------------------------------


# The two labels of the reading with no word on it, [[rr:TD-10#Decision outcome]]. The harness
# keys the empty reading by one label so its history runs across the flip, as the runtime's
# does, and counts the flips here.
WORDLESS = ("[sil]", "[speech]")
SPEECH_LABELS = {"advances": 0, "rank0": 0, "entries": 0}


def canon(text):
    return "[sil]" if text in WORDLESS else text


def beam_readings(p):
    alts = p.get("partial_alternatives") or []
    SPEECH_LABELS["advances"] += 1
    SPEECH_LABELS["rank0"] += bool(alts) and alts[0]["text"] == "[speech]"
    SPEECH_LABELS["entries"] += any(e["word"] == "[speech]" for e in p.get("partial_result") or [])
    return tuple((canon(e["text"]), e["confidence"]) for e in alts)


def word_list(text):
    return [w for w in text.split() if w not in WORDLESS]


def reading_leads(readings):
    """Each reading's confidence less the best among the others: the gap for rank 0, a deficit
    for the rest, undefined where a reading stands alone."""
    if len(readings) < 2:
        return {t: None for t, _ in readings}
    return {t: c - max(d for u, d in readings if u != t) for t, c in readings}


def relation(w0, w):
    if w == w0:
        return "same"
    if len(w) < len(w0) and w0[: len(w)] == w:
        return "prefix"
    if len(w) > len(w0) and w[: len(w0)] == w0:
        return "extends"
    return "differs"


def runtime_motion(p):
    """`relation` and `lead_delta` as the runtime reports them, by reading text, or None from a
    build that predates the keys. The runtime's history runs over every surviving group, so it
    reports a delta for a reading entering the list that this harness has no record of
    (`[[rr:TD-9#Readings are read once per decoding advance]]`)."""
    alts = p.get("partial_alternatives") or []
    if not alts or "lead_delta" not in alts[0]:
        return None
    return {canon(e["text"]): (e["relation"], e["lead_delta"]) for e in alts}


# Four printed confidences enter a reconstructed delta, each rounded to six decimals, and the
# runtime's leads and their difference are single precision: a few times 1e-6 either way.
RUNTIME_TOL = 2e-5


def check_motion(rt, derived, rel, parity):
    """The harness's derivation is the cross-check on the runtime's fields, over the readings it
    can see: a reading reported at both advances has its lead from the same two confidences
    either way."""
    for text, (r, delta) in rt.items():
        if r != rel[text]:
            raise AssertionError(f"relation {r!r} from the runtime, {rel[text]!r} derived, on {text!r}")
        mine = derived.get(text)
        if mine is not None and delta is None:
            raise AssertionError(f"lead_delta {mine} derived, null from the runtime, on {text!r}")
        if mine is None:
            parity["runtime_only" if delta is not None else "neither"] += 1
            continue
        if abs(delta - mine) > RUNTIME_TOL:
            raise AssertionError(f"lead_delta {delta} from the runtime, {mine} derived, on {text!r}")
        parity["agreed"] += 1
        parity["max_diff"] = max(parity["max_diff"], abs(delta - mine))


def empty_parity():
    return dict(agreed=0, runtime_only=0, neither=0, max_diff=0.0)


def trailing_sil(p):
    """The trailing silence a host reads: the `[sil]` entry after the last word. A `[speech]`
    entry after it (`[[rr:TD-10#Decision outcome]]`) is the best path entering a word's phones;
    the silence entry keeps its span and stops growing there."""
    e = p.get("partial_result") or []
    if e and e[-1]["word"] == "[speech]":
        e = e[:-1]
    if e and e[-1]["word"] == "[sil]":
        return (e[-1]["end_sample"] - e[-1]["start_sample"]) / SAMPLES_PER_MS, e[-1]["start_sample"]
    return None, None


def last_word_entry(p):
    for e in reversed(p.get("partial_result") or []):
        if e["word"] not in WORDLESS:
            return e
    return None


def inner_sil_spans(p):
    """The `[sil]` entries with a word on both sides, by the index of the word before them: the
    pause the decoder measured between two words."""
    e = p.get("partial_result") or []
    out = {}
    seen = 0
    for k, entry in enumerate(e):
        if entry["word"] not in WORDLESS:
            seen += 1
        elif 0 < k < len(e) - 1 and e[k + 1]["word"] not in WORDLESS and seen:
            out[seen - 1] = (entry["end_sample"] - entry["start_sample"]) / SAMPLES_PER_MS
    return out


def build_state(fed, seg, p, readings, prev, first_seen, parity=None):
    texts = [t for t, _ in readings]
    for t in texts:
        first_seen.setdefault(t, fed)
    lead = reading_leads(readings)
    derived = {}
    if prev is not None:
        for t in texts:
            before = prev["lead"].get(t)
            if before is not None and lead[t] is not None:
                derived[t] = lead[t] - before
    words = {t: word_list(t) for t in texts}
    w0 = words[texts[0]]
    rel = {t: relation(w0, words[t]) for t in texts}
    delta = derived
    rt = runtime_motion(p)
    if rt is not None:
        check_motion(rt, derived, rel, parity if parity is not None else empty_parity())
        delta = {t: d for t, (_, d) in rt.items() if d is not None}
        rel = {t: r for t, (r, _) in rt.items()}
    extends = [
        dict(rank=i, text=t, extra=words[t][len(w0) :], lead=lead[t], lead_delta=delta.get(t))
        for i, t in enumerate(texts)
        if rel[t] == "extends"
    ]
    span, sil_start = trailing_sil(p)
    entry = last_word_entry(p)
    return dict(
        fed=fed,
        seg=seg,
        readings=readings,
        words0=w0,
        lead=lead,
        delta=delta,
        # For the hover measurement alone; no field carries it.
        # [[rr:TD-9#Considered options]]
        age={t: (fed - first_seen[t]) / SAMPLES_PER_MS for t in texts},
        relation=rel,
        extends=extends,
        sil_span=span,
        sil_start=sil_start,
        last_word=entry["word"] if entry else None,
        last_word_end=entry["end_sample"] if entry else None,
        inner_sil=inner_sil_spans(p),
        grew=False,
    )


def run_stream(eng, pcm, block_ms, alternatives, words_on=True, parity=None):
    """Feed one stream in blocks, keeping every partial's readings with the position fed and
    every final with its own. A final clears the decoder's history, so the series is cut there:
    ages and lead deltas start again in the next segment."""
    rec = eng.new(alternatives=alternatives)
    if words_on and hasattr(rec, "SetPartialWords"):
        rec.SetPartialWords(True)
    block = RATE * block_ms // 1000 * 2
    states, blocks, finals = [], [], []
    fed = 0
    seg = 0
    prev = None
    prev_readings = None
    first_seen = {}
    prev_words0 = []
    for i in range(0, len(pcm), block):
        chunk = pcm[i : i + block]
        fed += len(chunk) // 2
        if rec.AcceptWaveform(chunk):
            f = json.loads(rec.Result())
            finals.append(dict(fed=fed, seg=seg, text=f.get("text", ""), result=f.get("result") or []))
            blocks.append(dict(fed=fed, adv=None, final=True, stable=None, word=None))
            seg += 1
            prev, prev_readings, first_seen, prev_words0 = None, None, {}, []
            continue
        p = json.loads(rec.PartialResult())
        readings = beam_readings(p)
        adv = None
        if readings and readings != prev_readings:
            st = build_state(fed, seg, p, readings, prev, first_seen, parity)
            st["i"] = len(states)
            st["grew"] = len(st["words0"]) > len(prev_words0)
            prev_words0 = st["words0"]
            states.append(st)
            prev = st
            adv = len(states) - 1
        elif readings and states:
            adv = len(states) - 1
        prev_readings = readings
        entry = last_word_entry(p)
        blocks.append(
            dict(
                fed=fed,
                adv=adv,
                final=False,
                stable=entry["stable_ms"] if entry and "stable_ms" in entry else None,
                word=entry["word"] if entry else None,
            )
        )
    f = json.loads(rec.FinalResult())
    finals.append(dict(fed=fed, seg=seg, text=f.get("text", ""), result=f.get("result") or []))
    return states, blocks, finals


def run_stream_text(eng, pcm, block_ms, words_on=False):
    """The stock wheel's half: rank-0 text per block and the finals, no readings to move."""
    rec = eng.new()
    if words_on and hasattr(rec, "SetPartialWords"):
        rec.SetPartialWords(True)
    block = RATE * block_ms // 1000 * 2
    blocks, finals = [], []
    fed = 0
    for i in range(0, len(pcm), block):
        chunk = pcm[i : i + block]
        fed += len(chunk) // 2
        if rec.AcceptWaveform(chunk):
            finals.append(dict(fed=fed, text=json.loads(rec.Result()).get("text", "")))
            blocks.append(dict(fed=fed, words=[], final=True))
        else:
            p = json.loads(rec.PartialResult())
            blocks.append(dict(fed=fed, words=word_list(p.get("partial", "")), final=False))
    finals.append(dict(fed=fed, text=json.loads(rec.FinalResult()).get("text", "")))
    return blocks, finals


# --- the rules --------------------------------------------------------------------------------


def gaining_extender(st, ext_delta):
    """A reading that is rank 0 plus exactly one word and is gaining on the field."""
    return any(
        len(e["extra"]) == 1 and e["lead_delta"] is not None and e["lead_delta"] >= ext_delta for e in st["extends"]
    )


def gaining_extras(st, ext_delta):
    """The extra words of the readings that are rank 0 plus one word and gaining, in rank order."""
    return [
        e["extra"][0]
        for e in st["extends"]
        if len(e["extra"]) == 1 and e["lead_delta"] is not None and e["lead_delta"] >= ext_delta
    ]


def calls(states, bound_ms, sil_delta=None, ext_delta=None):
    """One forecast per advance. R0 is sil_delta being None: a word arriving at rank 0 is speech
    beginning, the trailing span crossing the bound is speech ending, and a span that goes on
    growing is silence being kept. R1 reads the motion first: at an utterance's start the empty
    reading losing its lead means a word is coming, and mid-utterance an extending reading
    gaining on the field means the next word is forming."""
    out = []
    crossed = False
    seg = None
    for st in states:
        if st["seg"] != seg:
            seg, crossed = st["seg"], False
        w0 = st["words0"]
        span = st["sil_span"]
        call = None
        if sil_delta is not None and not st["grew"]:
            if not w0:
                d = st["delta"].get("[sil]")
                if d is not None and d <= sil_delta:
                    call = "away"
            elif gaining_extender(st, ext_delta):
                call = "away"
        if call is None:
            if st["grew"]:
                call, crossed = "away", False
            elif span is not None and span >= bound_ms:
                call = "maintaining silence" if crossed else "toward"
                crossed = True
            elif w0:
                call = "maintaining speech"
            else:
                call = "maintaining silence"
        out.append(call)
    return out


def block_calls(blocks, per_advance):
    """The call a host reads at every block: between advances the partial repeats, so the call
    does."""
    return [None if b["adv"] is None else per_advance[b["adv"]] for b in blocks]


def tally(counts, truth, call):
    if call == truth:
        counts[truth]["tp"] += 1
    else:
        counts[call]["fp"] += 1
        counts[truth]["fn"] += 1


def prf(c):
    p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else float("nan")
    r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else float("nan")
    f = 2 * p * r / (p + r) if p == p and r == r and p + r else 0.0
    return p, r, f


def add_counts(a, b):
    for s in STATES:
        for k in ("tp", "fp", "fn"):
            a[s][k] += b[s][k]
    return a


def empty_counts():
    return {s: dict(tp=0, fp=0, fn=0) for s in STATES}


# --- benchmark A ------------------------------------------------------------------------------


def label_blocks(stream):
    """The ground-truth forecast at every block, and the utterance each block belongs to, both
    fixed once so a grid of thresholds is scored against the same labels."""
    truth, words = stream["truth"], stream["words"]
    pos = [w["pos"] for w in words]
    stream["labels"] = {
        w_ms: [truth.forecast(b["fed"], w_ms * SAMPLES_PER_MS) for b in stream["blocks"]] for w_ms in HORIZONS
    }
    stream["utt"] = [
        words[max(0, bisect.bisect_right(pos, b["fed"]) - 1)]["utt"] if words else 0 for b in stream["blocks"]
    ]


def group_counts(stream, per_advance, w_ms):
    """Per-utterance confusion counts, so a bootstrap can resample utterances."""
    labels = stream["labels"][w_ms]
    out = defaultdict(empty_counts)
    for utt, label, call in zip(stream["utt"], labels, block_calls(stream["blocks"], per_advance)):
        if call is not None:
            tally(out[utt], label, call)
    return out


def transitions_of(stream, lookback_ms):
    """Every ground-truth transition with the window a rule may call it in. The window reaches
    back lookback_ms and no further: a stream's gaps run to three seconds, and a call that
    early is not a forecast of the word that eventually arrives."""
    words = stream["words"]
    back = lookback_ms * SAMPLES_PER_MS
    rows = []
    for i, w in enumerate(words):
        prev_end = words[i - 1]["offset"] if i else 0
        nxt = words[i + 1]["onset"] if i + 1 < len(words) else stream["samples"]
        rows.append(
            dict(
                kind="away",
                at=w["onset"],
                lo=max(prev_end, w["onset"] - back),
                hi=min(w["offset"], w["onset"] + back),
                word=i,
                label=w["label"],
            )
        )
        rows.append(
            dict(
                kind="toward",
                at=w["offset"],
                lo=max(w["onset"], w["offset"] - back),
                hi=min(nxt, w["offset"] + back),
                word=i,
                label=w["label"],
            )
        )
    return rows


def lead_times(streams, per_advance_by_stream, tolerance):
    """Per true transition, the milliseconds from the first block a rule calls that state to the
    transition itself, negative before it. A transition is 'handled' when the call falls inside
    the tolerance window around the truth, which is the paired unit for McNemar."""
    out = {"away": [], "toward": []}
    handled = {"away": [], "toward": []}
    for s in streams:
        cb = block_calls(s["blocks"], per_advance_by_stream[s["key"]])
        feds = [b["fed"] for b in s["blocks"]]
        for tr in s["transitions"]:
            lo = bisect.bisect_left(feds, tr["lo"])
            hi = bisect.bisect_right(feds, tr["hi"])
            first = None
            for k in range(lo, hi):
                if cb[k] == tr["kind"]:
                    first = feds[k]
                    break
            ms = None if first is None else (first - tr["at"]) / SAMPLES_PER_MS
            out[tr["kind"]].append(ms)
            ok = ms is not None and -tolerance <= ms <= tolerance
            handled[tr["kind"]].append(bool(ok))
    return out, handled


def alarms(streams, per_advance_by_stream, kind, block_ms):
    """False calls where the truth cannot be that state: `away` inside a finish gap after the
    word has ended, or anywhere on the recordings. Counted as blocks and as distinct runs."""
    blocks = runs = total_blocks = 0
    for s in streams:
        cb = block_calls(s["blocks"], per_advance_by_stream[s["key"]])
        spans = (
            [(g["pos"], g["pos"] + g["samples"]) for g in s["gaps"] if g["kind"] == "finish"]
            if kind == "finish"
            else [(0, s["samples"])]
        )
        prev = None
        for b, call in zip(s["blocks"], cb):
            if not any(lo <= b["fed"] < hi for lo, hi in spans):
                prev = None
                continue
            total_blocks += 1
            if call == "away":
                blocks += 1
                if prev != "away":
                    runs += 1
            prev = call
    minutes = total_blocks * block_ms / 1000 / 60.0 if total_blocks else float("nan")
    return dict(blocks=blocks, runs=runs, minutes=minutes, per_min_blocks=blocks / minutes, per_min=runs / minutes)


def eos_curve(streams, bound_grid, ext_delta):
    """Among the silent stretches, at each millisecond of trailing silence: the share of pauses
    whose span reaches it before the next word (a pause taken for a finish) and the share of
    finishes whose span reaches it at all. The motion variant refuses the call at an advance
    where an extending reading is gaining."""
    rows = {x: dict(pause=0, pause_motion=0, finish=0, finish_motion=0) for x in bound_grid}
    totals = Counter()
    early = {x: [] for x in bound_grid}
    for s in streams:
        by_seg = defaultdict(list)
        for st in s["states"]:
            by_seg[st["seg"]].append(st)
        finals = sorted(f["fed"] for f in s["finals"])
        for g in s["gaps"]:
            w = s["words"][g["after"]]
            lo = w["offset"]
            hi = g["pos"] + g["samples"]
            if g["kind"] == "pause":
                nxt = g["after"] + 1
                hi = s["words"][nxt]["onset"] if nxt < len(s["words"]) else hi
            totals[g["kind"]] += 1
            spans = [st for st in s["states"] if lo <= st["fed"] < hi and st["sil_span"] is not None]
            endpoint = next((f for f in finals if lo <= f < hi), None)
            for x in bound_grid:
                hit = next((st for st in spans if st["sil_span"] >= x), None)
                if hit is not None:
                    rows[x][g["kind"]] += 1
                    if g["kind"] == "finish" and endpoint is not None and hit["fed"] <= endpoint:
                        early[x].append((endpoint - hit["fed"]) / SAMPLES_PER_MS)
                quiet = next((st for st in spans if st["sil_span"] >= x and not gaining_extender(st, ext_delta)), None)
                if quiet is not None:
                    rows[x][g["kind"] + "_motion"] += 1
    return rows, totals, early


# --- benchmark B ------------------------------------------------------------------------------


def eos_by_gap_length(streams, bound_ms, ext_delta, edges=(100, 200, 300, 400, 500, 600, 700, 800)):
    """At one bound, the share of pauses taken for a finish by the length of the pause that was
    built, since the headline share is a function of that distribution and nothing else."""
    rows = {}
    for lo, hi in zip(edges, edges[1:]):
        rows[f"{lo}-{hi}"] = dict(n=0, mistaken=0, mistaken_motion=0)
    for s_ in streams:
        for g in s_["gaps"]:
            if g["kind"] != "pause":
                continue
            ms = g["samples"] / SAMPLES_PER_MS
            key = next((f"{lo}-{hi}" for lo, hi in zip(edges, edges[1:]) if lo <= ms < hi), None)
            if key is None:
                continue
            w = s_["words"][g["after"]]
            nxt = g["after"] + 1
            lo_s = w["offset"]
            hi_s = s_["words"][nxt]["onset"] if nxt < len(s_["words"]) else g["pos"] + g["samples"]
            spans = [st for st in s_["states"] if lo_s <= st["fed"] < hi_s and st["sil_span"] is not None]
            rows[key]["n"] += 1
            if any(st["sil_span"] >= bound_ms for st in spans):
                rows[key]["mistaken"] += 1
            if any(st["sil_span"] >= bound_ms and not gaining_extender(st, ext_delta) for st in spans):
                rows[key]["mistaken_motion"] += 1
    return rows


def built_alike(a, b):
    """Whether two runs' `stream_sizes` describe the same built streams. The counts the decode
    produces, finals and advances among them, are what a bound is there to change."""
    keys = ("streams", "words", "utterances", "pauses", "finishes", "transitions", "blocks", "minutes")
    return all(a[k] == b[k] for k in keys)


def runtime_finals(streams, edges=(100, 200, 300, 400, 500, 600, 700, 800)):
    """The same question as `eos_curve`, asked of the finals the runtime emitted rather than of
    the spans the series recorded: over the same silent stretches, how many pauses got a final
    before the next word and how many finishes got one at all. With a host bound set this is what
    the bound did; with the stock rules it is what they did.

    The word counts are the price. A final's word entries carry spans measured from the start of
    the audio, so a truth word no entry covers was lost and one two entries cover was decoded
    twice, which is what a bound firing inside a word leaves behind."""
    tot = Counter()
    hits = Counter()
    lead = []
    by_pause = {f"{lo}-{hi}": dict(n=0, mistaken=0) for lo, hi in zip(edges, edges[1:])}
    finals = words_in_finals = 0
    samples = 0
    words_total = lost = restarted = 0
    for s_ in streams:
        feds = sorted(f["fed"] for f in s_["finals"])
        for g in s_["gaps"]:
            w = s_["words"][g["after"]]
            lo_s = w["offset"]
            hi_s = g["pos"] + g["samples"]
            if g["kind"] == "pause":
                nxt = g["after"] + 1
                hi_s = s_["words"][nxt]["onset"] if nxt < len(s_["words"]) else hi_s
            tot[g["kind"]] += 1
            hit = next((f for f in feds if lo_s <= f < hi_s), None)
            if hit is not None:
                hits[g["kind"]] += 1
                if g["kind"] == "finish":
                    lead.append((hit - lo_s) / SAMPLES_PER_MS)
            if g["kind"] == "pause":
                ms = g["samples"] / SAMPLES_PER_MS
                key = next((k for k, (lo, hi) in zip(by_pause, zip(edges, edges[1:])) if lo <= ms < hi), None)
                if key is not None:
                    by_pause[key]["n"] += 1
                    by_pause[key]["mistaken"] += hit is not None
        spans = [
            (e["start"] * RATE, e["end"] * RATE) for f in s_["finals"] for e in f["result"] if sc.words_of(e["word"])
        ]
        for w in s_["words"]:
            covering = sum(1 for lo_s, hi_s in spans if lo_s < w["offset"] and hi_s > w["onset"])
            words_total += 1
            lost += covering == 0
            restarted += covering > 1
        finals += len(s_["finals"])
        words_in_finals += sum(1 for f in s_["finals"] if sc.words_of(f["text"]))
        samples += s_["samples"]
    minutes = samples / RATE / 60.0
    return dict(
        totals={k: tot[k] for k in ("pause", "finish")},
        pauses_mistaken=hits["pause"],
        finishes_called=hits["finish"],
        finish_ms_p50=sc.quantile(lead, 0.5),
        finish_ms_p90=sc.quantile(lead, 0.9),
        by_pause=by_pause,
        finals=finals,
        finals_with_a_word=words_in_finals,
        finals_per_min=finals / minutes if minutes else float("nan"),
        minutes=minutes,
        words=words_total,
        lost=lost,
        restarted=restarted,
    )


def transition_records(streams, ext_delta, lookback_ms, forward_ms=1500):
    """Per ground-truth transition into a word, inside a bounded window around its onset: when
    the rank-0 word list grew (the detection a host has today), when the word itself led, when a
    reading first extended rank 0 and what its extra word was, the decoder's own measure of the
    pause, and the latency from the onset. Also every advance in the window that carried an
    extending reading, so preview accuracy can be read against the lead still to run."""
    rows = []
    previews = []
    back = lookback_ms * SAMPLES_PER_MS
    fwd = forward_ms * SAMPLES_PER_MS
    for s_ in streams:
        feds = [b["fed"] for b in s_["blocks"]]
        for i, w in enumerate(s_["words"]):
            prev = s_["words"][i - 1] if i and s_["words"][i - 1]["utt"] == w["utt"] else None
            prev_end = s_["words"][i - 1]["offset"] if i else 0
            nxt = s_["words"][i + 1]["onset"] if i + 1 < len(s_["words"]) else s_["samples"]
            lo = max(prev_end, w["onset"] - back)
            hi = min(nxt, w["onset"] + fwd)
            inside = [st for st in s_["states"] if lo <= st["fed"] < hi]
            arrive = next((st for st in inside if st["grew"]), None)
            leads = next((st for st in inside if st["words0"] and st["words0"][-1] == w["label"]), None)
            ext = next((st for st in inside if st["extends"]), None)
            gext = next((st for st in inside if gaining_extender(st, ext_delta)), None)
            correct_ext = next(
                (st for st in inside if any(e["extra"] and e["extra"][0] == w["label"] for e in st["extends"])),
                None,
            )
            gap = next((g for g in s_["gaps"] if prev is not None and g["after"] == i - 1), None)
            row = dict(
                stream=s_["key"],
                word=i,
                utt=w["utt"],
                label=w["label"],
                first_in_utterance=prev is None,
                onset_ms=w["onset"] / SAMPLES_PER_MS,
                arrive_ms=None if arrive is None else arrive["fed"] / SAMPLES_PER_MS,
                arrive_word=None if arrive is None else (arrive["words0"][-1] if arrive["words0"] else None),
                arrive_index=None if arrive is None else arrive["i"],
                leads_ms=None if leads is None else leads["fed"] / SAMPLES_PER_MS,
                ext_ms=None if ext is None else ext["fed"] / SAMPLES_PER_MS,
                ext_index=None if ext is None else ext["i"],
                ext_top=None
                if ext is None
                else (ext["extends"][0]["extra"][0] if ext["extends"][0]["extra"] else None),
                ext_top3=None if ext is None else [e["extra"][0] for e in ext["extends"][:3] if e["extra"]],
                ext_rank0_empty=None if ext is None else not ext["words0"],
                ext_gaining=None if ext is None else gaining_extender(ext, ext_delta),
                gext_ms=None if gext is None else gext["fed"] / SAMPLES_PER_MS,
                gext_index=None if gext is None else gext["i"],
                gext_top=None if gext is None else gaining_extras(gext, ext_delta)[0],
                gext_top3=None if gext is None else gaining_extras(gext, ext_delta)[:3],
                correct_ext_ms=None if correct_ext is None else correct_ext["fed"] / SAMPLES_PER_MS,
                gap_ms=None if gap is None else gap["samples"] / SAMPLES_PER_MS,
                gap_kind=None if gap is None else gap["kind"],
                elapsed_ms=None if prev is None else (w["onset"] - prev["offset"]) / SAMPLES_PER_MS,
                inner_sil_ms=None,
                latency_ms=None,
                stable_ms_from_onset=None,
                preview_advances=None,
                preview_ms=None,
                preview_correct_ms=None,
                gaining_preview_ms=None,
                final_between=None
                if prev is None
                else any(prev["offset"] <= f["fed"] < w["onset"] for f in s_["finals"]),
            )
            if arrive is not None and ext is not None:
                row["preview_advances"] = row["arrive_index"] - row["ext_index"]
                row["preview_ms"] = row["arrive_ms"] - row["ext_ms"]
                if correct_ext is not None:
                    row["preview_correct_ms"] = row["arrive_ms"] - row["correct_ext_ms"]
            if arrive is not None and gext is not None:
                row["gaining_preview_ms"] = row["arrive_ms"] - row["gext_ms"]
            if leads is not None:
                row["latency_ms"] = (leads["fed"] - w["onset"]) / SAMPLES_PER_MS
                if prev is not None and len(leads["words0"]) >= 2:
                    # the [sil] entry before the word that has just led, which is the last one
                    row["inner_sil_ms"] = leads["inner_sil"].get(len(leads["words0"]) - 2)
                k = bisect.bisect_left(feds, leads["fed"])
                stable = next(
                    (
                        b["fed"]
                        for b in s_["blocks"][k:]
                        if not b["final"] and b["word"] == w["label"] and (b["stable"] or 0) >= STABLE_BOUND_MS
                    ),
                    None,
                )
                row["stable_ms_from_onset"] = None if stable is None else (stable - w["onset"]) / SAMPLES_PER_MS
            rows.append(row)
            if arrive is None:
                continue
            for st in inside:
                if st["fed"] >= arrive["fed"] or not st["extends"]:
                    continue
                extra = [e["extra"][0] for e in st["extends"] if e["extra"]]
                gain = gaining_extras(st, ext_delta)
                previews.append(
                    dict(
                        stream=s_["key"],
                        word=i,
                        label=w["label"],
                        utt=w["utt"],
                        lead_ms=(arrive["fed"] - st["fed"]) / SAMPLES_PER_MS,
                        rank0_empty=not st["words0"],
                        top1=bool(extra and extra[0] == w["label"]),
                        top3=w["label"] in extra[:3],
                        gaining=bool(gain),
                        gain_top1=bool(gain and gain[0] == w["label"]),
                        gain_top3=w["label"] in gain[:3],
                    )
                )
    return rows, previews


def span_errors(streams):
    """The trailing `[sil]` span, per advance inside a gap, against two clocks: the silence that
    has elapsed since the word's energy offset, and the silence since the decoder's own end of
    that word. The second isolates the span's own lag from the difference between the two
    alignments."""
    out = defaultdict(list)
    decoded = defaultdict(list)
    for s_ in streams:
        for g in s_["gaps"]:
            w = s_["words"][g["after"]]
            lo, hi = w["offset"], g["pos"] + g["samples"]
            if g["kind"] == "pause":
                nxt = g["after"] + 1
                hi = s_["words"][nxt]["onset"] if nxt < len(s_["words"]) else hi
            for st in s_["states"]:
                if lo <= st["fed"] < hi and st["sil_span"] is not None:
                    out[g["kind"]].append(st["sil_span"] - (st["fed"] - lo) / SAMPLES_PER_MS)
                    if st["last_word_end"] is not None:
                        decoded[g["kind"]].append(st["sil_span"] - (st["fed"] - st["last_word_end"]) / SAMPLES_PER_MS)
    return out, decoded


def dying_extenders(streams, spans, one_word_only=True):
    """Extending readings that appear where no word follows and never reach rank 0: the beam
    guessing at a word that is not coming. Counted per minute of the audio in spans, and
    split by whether rank 0 held words: over an empty rank 0 every word-carrying reading
    extends it, so those are a word being guessed at from silence, not a continuation."""
    out = {True: 0, False: 0}
    total = 0
    for s_ in streams:
        for lo, hi in spans(s_):
            total += hi - lo
            seen = {True: set(), False: set()}
            reached = set()
            for st in s_["states"]:
                if not (lo <= st["fed"] < hi):
                    continue
                for e in st["extends"]:
                    if e["extra"] and (len(e["extra"]) == 1 or not one_word_only):
                        seen[bool(st["words0"])].add(tuple(word_list(e["text"])))
                if st["words0"]:
                    reached.add(tuple(st["words0"]))
            for k in (True, False):
                out[k] += len([t for t in seen[k] if t not in reached])
    minutes = total / RATE / 60.0 if total else float("nan")

    def per(c):
        return c / minutes if minutes == minutes and minutes else float("nan")

    return dict(
        over_words=out[True],
        over_silence=out[False],
        minutes=minutes,
        per_min_over_words=per(out[True]),
        per_min_over_silence=per(out[False]),
    )


# --- the readings' momentum -------------------------------------------------------------------

BANDS = (("<1", 0.0, 1.0), ("1-4", 1.0, 4.0), (">=4", 4.0, float("inf")))
MOMENTUM_SUBSETS = ("all", "[sil]", "extends", "differs")
ALIGN_OFFSETS = (-3, -2, -1, 0, 1, 2)
SIL_CROSS_GRID = (0.0, -0.25, -0.5, -1.0, -2.0, -4.0)
EXT_CROSS_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def band_of(lead):
    a = abs(lead)
    return next(name for name, lo, hi in BANDS if lo <= a < hi)


def rank_auc(pairs):
    """Rank-AUC of a score against a boolean label, ties at the mid-rank; 0.5 is no information
    and below 0.5 the signal is inverted."""
    xs = [(v, y) for v, y in pairs if v is not None]
    pos = sum(1 for _, y in xs if y)
    neg = len(xs) - pos
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
    rank_sum = sum(rk for rk, (_, y) in zip(ranks, xs) if y)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


class Corr:
    """Pearson r and sign agreement accumulated in sums, since the pairs run to six figures."""

    def __init__(self):
        self.n = self.sx = self.sy = self.sxx = self.syy = self.sxy = 0.0
        self.signed = self.agree = 0

    def add(self, x, y):
        self.n += 1
        self.sx += x
        self.sy += y
        self.sxx += x * x
        self.syy += y * y
        self.sxy += x * y
        if x and y:
            self.signed += 1
            self.agree += (x > 0) == (y > 0)

    def read(self):
        n = self.n
        if n < 2:
            return dict(n=int(n), r=None, sign_agreement=None, signed=self.signed)
        cov = self.sxy / n - (self.sx / n) * (self.sy / n)
        vx = self.sxx / n - (self.sx / n) ** 2
        vy = self.syy / n - (self.sy / n) ** 2
        r = cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else None
        return dict(
            n=int(n),
            r=r,
            sign_agreement=self.agree / self.signed if self.signed else None,
            signed=self.signed,
        )


def momentum(streams):
    """Does a lead's motion carry from one advance to the next: the correlation between a
    reading's lead_delta now and at the next advance, by the band its lead sits in and by what
    the reading is to rank 0. Three advances are needed for one pair, since each delta is itself
    a difference."""
    acc = defaultdict(Corr)
    for s_ in streams:
        states = s_["states"]
        for k in range(len(states) - 1):
            a, b = states[k], states[k + 1]
            if a["seg"] != b["seg"]:
                continue
            for t, lead in a["lead"].items():
                da, db = a["delta"].get(t), b["delta"].get(t)
                if lead is None or da is None or db is None:
                    continue
                band = band_of(lead)
                subsets = ["all", "[sil]"] if t == "[sil]" else ["all", a["relation"][t]]
                for subset in subsets:
                    if subset not in MOMENTUM_SUBSETS:
                        continue
                    acc[(subset, band)].add(da, db)
                    acc[(subset, "all")].add(da, db)
    return {f"{subset}|{band}": c.read() for (subset, band), c in acc.items()}


def spread(values):
    v = [x for x in values if x is not None]
    if not v:
        return dict(n=0, mean=None, sd=None, p10=None, p50=None, p90=None)
    mean = sum(v) / len(v)
    var = sum((x - mean) ** 2 for x in v) / len(v)
    return dict(
        n=len(v),
        mean=mean,
        sd=math.sqrt(var),
        p10=sc.quantile(v, 0.1),
        p50=sc.quantile(v, 0.5),
        p90=sc.quantile(v, 0.9),
    )


def kept_silence_spans(stream, after_ms=1000, before_ms=500):
    """The interior of each finish gap: long after the word has ended and well before anything
    begins, which is silence being kept rather than a transition."""
    out = []
    for g in stream["gaps"]:
        if g["kind"] != "finish":
            continue
        w = stream["words"][g["after"]]
        lo = w["offset"] + after_ms * SAMPLES_PER_MS
        hi = g["pos"] + g["samples"] - before_ms * SAMPLES_PER_MS
        if hi > lo:
            out.append((lo, hi))
    return out


def whole_stream_spans(stream, after_ms=1000):
    return [(after_ms * SAMPLES_PER_MS, stream["samples"])]


def silence_signature(streams, spans):
    """In steady silence: the leading reading's lead over its best rival, that lead's motion, and
    how old the rival is. A lead that hovers while the rival stays young is a bound set by word
    hypotheses that are started and pruned, not a lead that is being won."""
    leads, deltas, ages, rank1_ages = [], [], [], []
    sil_leads = advances = fresh = 0
    for s_ in streams:
        windows = spans(s_)
        for st in s_["states"]:
            if not any(lo <= st["fed"] < hi for lo, hi in windows):
                continue
            advances += 1
            top = st["readings"][0][0]
            sil_leads += top == "[sil]"
            leads.append(st["lead"].get(top))
            deltas.append(st["delta"].get(top))
            ages.append(st["age"][top])
            if len(st["readings"]) >= 2:
                rival = st["readings"][1][0]
                rank1_ages.append(st["age"][rival])
                fresh += st["age"][rival] <= 240
    return dict(
        advances=advances,
        sil_at_rank0=sil_leads,
        lead=spread(leads),
        velocity=spread(deltas),
        leader_age_ms=spread(ages),
        rival_age_ms=spread(rank1_ages),
        rival_fresh=fresh,
        rival_n=len(rank1_ages),
    )


def best_extends_delta(st):
    for e in st["extends"]:
        if len(e["extra"]) == 1:
            return e["lead_delta"]
    return None


def aligned_velocity(streams):
    """Every ground-truth onset and offset aligned at the advance that first covers it, with the
    motion of three readings at the advances around it."""
    series = defaultdict(lambda: defaultdict(list))
    for s_ in streams:
        states = s_["states"]
        feds = [st["fed"] for st in states]
        for tr in s_["transitions"]:
            i0 = bisect.bisect_left(feds, tr["at"])
            if i0 >= len(states):
                continue
            for off in ALIGN_OFFSETS:
                k = i0 + off
                if not (0 <= k < len(states)) or states[k]["seg"] != states[i0]["seg"]:
                    continue
                st = states[k]
                kind = tr["kind"]
                series[(kind, off)]["leader"].append(st["delta"].get(st["readings"][0][0]))
                if not st["words0"]:
                    series[(kind, off)]["sil"].append(st["delta"].get("[sil]"))
                else:
                    series[(kind, off)]["extends"].append(best_extends_delta(st))
    return {f"{kind}|{off}": {name: spread(v) for name, v in d.items()} for (kind, off), d in series.items()}


def crossings(stream, signal, theta):
    """The advances where one reading's motion crosses a threshold: the empty reading's lead
    falling, or a one-word extending reading's lead rising."""
    out = []
    for st in stream["states"]:
        if signal == "sil":
            if st["words0"]:
                continue
            d = st["delta"].get("[sil]")
            if d is not None and d <= theta:
                out.append(st["fed"])
        else:
            if not st["words0"]:
                continue
            d = best_extends_delta(st)
            if d is not None and d >= theta:
                out.append(st["fed"])
    return out


def crossing_scores(streams, signal, theta, lookback_ms, block_ms):
    """A crossing counts as a hit when a word's onset follows it within the lookback; recall is
    over the onsets. False alarms are crossings in kept silence and, on the recordings, anywhere.
    """
    back = lookback_ms * SAMPLES_PER_MS
    hits = total = 0
    onsets = called = 0
    leads = []
    alarm_events = 0
    alarm_samples = 0
    for s_ in streams:
        fires = crossings(s_, signal, theta)
        starts = [w["onset"] for w in s_["words"]]
        total += len(fires)
        for f in fires:
            j = bisect.bisect_left(starts, f)
            hits += j < len(starts) and starts[j] - f <= back
        for w in s_["words"]:
            onsets += 1
            before = [f for f in fires if w["onset"] - back <= f < w["onset"]]
            if before:
                called += 1
                leads.append((before[0] - w["onset"]) / SAMPLES_PER_MS)
        windows = kept_silence_spans(s_) if s_["words"] else whole_stream_spans(s_)
        for lo, hi in windows:
            alarm_samples += hi - lo
            alarm_events += sum(1 for f in fires if lo <= f < hi)
    precision = hits / total if total else float("nan")
    recall = called / onsets if onsets else float("nan")
    minutes = alarm_samples / RATE / 60.0
    return dict(
        theta=theta,
        crossings=total,
        hits=hits,
        precision=precision,
        recall=recall,
        f1=2 * precision * recall / (precision + recall) if total and called else 0.0,
        onsets=onsets,
        called=called,
        lead_p10=sc.quantile(leads, 0.1),
        lead_p50=sc.quantile(leads, 0.5),
        lead_p90=sc.quantile(leads, 0.9),
        alarms_per_min=alarm_events / minutes if minutes else float("nan"),
        alarm_minutes=minutes,
    )


# --- the onset question, exploratory ------------------------------------------------------------

ONSET_BACK_MS = 1000  # how long after a call a word may arrive and still be the word it called
ONSET_DETECT_MS = 240  # one advance: a call this soon after the onset is a detection, not a miss

ONSET_GRIDS = {
    "motion": (0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
    "presence": (0.0,),
    "level": (-16.0, -12.0, -8.0, -6.0, -4.0, -2.0, -1.0, 0.0),
    "silence": (200.0, 300.0, 400.0, 600.0, 800.0, 1200.0, 1600.0, 2000.0),
    "timing": (200.0, 300.0, 400.0, 600.0, 800.0, 1200.0, 1600.0, 2000.0),
}
ONSET_LABELS = {
    "motion": "a one-word extending reading gaining by at least (nats)",
    "presence": "a one-word extending reading exists at all",
    "level": "the best one-word extending reading's lead is at least (nats)",
    "silence": "the trailing `[sil]` span has reached (ms)",
    "timing": "this long since the host last saw rank 0 grow (ms)",
}


def one_word_extenders(st):
    """The readings that are rank 0 plus exactly one word. Where rank 0 is the empty reading every
    word-carrying reading is one of these, which is the README's rule before a first word."""
    return [e for e in st["extends"] if len(e["extra"]) == 1]


def onset_fires(stream, rule, theta):
    """The advances at which a rule first calls a word coming. A run of consecutive advances where
    the rule holds is one call, not one per advance, so the calls are events a host acts on."""
    out = []
    holding = False
    clock = None
    seg = None
    for st in stream["states"]:
        if st["seg"] != seg:
            seg, clock, holding = st["seg"], st["fed"], False
        if st["grew"]:
            clock = st["fed"]
        cand = one_word_extenders(st)
        if rule == "motion":
            hit = any(e["lead_delta"] is not None and e["lead_delta"] >= theta for e in cand)
        elif rule == "presence":
            hit = bool(cand)
        elif rule == "level":
            hit = any(e["lead"] is not None and e["lead"] >= theta for e in cand)
        elif rule == "silence":
            hit = st["sil_span"] is not None and st["sil_span"] >= theta
        else:
            hit = (st["fed"] - clock) / SAMPLES_PER_MS >= theta
        if hit and not holding:
            out.append(st["fed"])
        holding = hit
    return out


def nonspeech_samples(stream):
    speech = sum(w["offset"] - w["onset"] for w in stream["words"])
    return max(0, stream["samples"] - speech)


def match_calls(fires, words, back_ms=ONSET_BACK_MS, detect_ms=ONSET_DETECT_MS):
    """One call to one word and no more. Each call in time order takes the earliest word still
    unclaimed whose onset falls between one advance before it and `back_ms` after it; every other
    call is a false alarm and every unclaimed word a miss."""
    back = back_ms * SAMPLES_PER_MS
    detect = detect_ms * SAMPLES_PER_MS
    claimed = [False] * len(words)
    matched = []
    alarms = []
    for f in fires:
        take = None
        for i, w in enumerate(words):
            if claimed[i]:
                continue
            if f - detect <= w["onset"] <= f + back:
                take = i
                break
            if w["onset"] > f + back:
                break
        if take is None:
            alarms.append(f)
        else:
            claimed[take] = True
            matched.append((f, take))
    return matched, alarms, claimed


def onset_scores(streams, rule, theta):
    """One rule at one threshold over a set of streams, with the calls that could carry acoustic
    evidence of the word kept apart from the calls that could not."""
    calls = matched = onsets = 0
    before_clip = in_clip = after_onset = 0
    leads = []
    alarm_calls = 0
    quiet_samples = 0
    for s_ in streams:
        words = s_["words"]
        fires = onset_fires(s_, rule, theta)
        hits, false_calls, claimed = match_calls(fires, words)
        calls += len(fires)
        matched += len(hits)
        onsets += len(words)
        alarm_calls += len(false_calls)
        quiet_samples += nonspeech_samples(s_)
        for f, i in hits:
            w = words[i]
            leads.append((f - w["onset"]) / SAMPLES_PER_MS)
            if f < w["pos"]:
                before_clip += 1
            elif f < w["onset"]:
                in_clip += 1
            else:
                after_onset += 1
    minutes = quiet_samples / RATE / 60.0
    precision = matched / calls if calls else float("nan")
    recall = matched / onsets if onsets else float("nan")
    before = [x for x in leads if x < 0]
    return dict(
        rule=rule,
        theta=theta,
        calls=calls,
        matched=matched,
        onsets=onsets,
        precision=precision,
        recall=recall,
        f1=2 * precision * recall / (precision + recall) if matched else 0.0,
        before_clip=before_clip,
        in_clip_before_onset=in_clip,
        after_onset=after_onset,
        lead_p10=sc.quantile(before, 0.1),
        lead_p50=sc.quantile(before, 0.5),
        lead_p90=sc.quantile(before, 0.9),
        alarms=alarm_calls,
        nonspeech_minutes=minutes,
        alarms_per_min=alarm_calls / minutes if minutes else float("nan"),
    )


def onset_study(val, test, noise):
    """Every rule fitted twice on the validation streams, once for its own best F1 and once to
    spend the same false alarms as the motion rule, and both read on the testing streams. The
    equal-alarm column is the attribution: a rule that calls more words by calling more often has
    not shown that its signal is what did it."""
    fitted = {r: {t: onset_scores(val, r, t) for t in ONSET_GRIDS[r]} for r in ONSET_GRIDS}
    best = {r: max(g, key=lambda t: g[t]["f1"]) for r, g in fitted.items()}
    budget = fitted["motion"][best["motion"]]["alarms_per_min"]
    equal = {r: min(g, key=lambda t: (abs(g[t]["alarms_per_min"] - budget), -g[t]["f1"])) for r, g in fitted.items()}
    out = dict(
        alarm_budget_per_min=budget,
        grid={r: {str(t): v for t, v in g.items()} for r, g in fitted.items()},
        best_theta={r: best[r] for r in best},
        equal_alarm_theta={r: equal[r] for r in equal},
        testing={r: onset_scores(test, r, best[r]) for r in best},
        testing_equal_alarm={r: onset_scores(test, r, equal[r]) for r in equal},
        recordings={r: onset_scores(noise, r, best[r]) for r in best},
    )
    return out


def reversal(streams):
    """The leading reading leading but losing: its lead is positive and its motion negative.
    Scored against rank 0 changing at the next advance and against rank 0's words differing from
    the final that closes the segment."""
    rows = []
    for s_ in streams:
        finals = {f["seg"]: sc.words_of(f["text"]) for f in s_["finals"]}
        states = s_["states"]
        for k in range(len(states) - 1):
            st, nxt = states[k], states[k + 1]
            if st["seg"] != nxt["seg"]:
                continue
            top = st["readings"][0][0]
            lead, d = st["lead"].get(top), st["delta"].get(top)
            if lead is None or d is None:
                continue
            rows.append(
                dict(
                    band=band_of(lead),
                    lead=lead,
                    delta=d,
                    reversing=d < 0,
                    changes=nxt["readings"][0][0] != top,
                    unlike_final=st["words0"] != finals.get(st["seg"], []),
                )
            )
    out = {}
    for band in ("all",) + tuple(name for name, _, _ in BANDS):
        b = rows if band == "all" else [x for x in rows if x["band"] == band]
        if not b:
            continue
        for target in ("changes", "unlike_final"):
            tp = sum(1 for x in b if x["reversing"] and x[target])
            fp = sum(1 for x in b if x["reversing"] and not x[target])
            fn = sum(1 for x in b if not x["reversing"] and x[target])
            out[f"{band}|{target}"] = dict(
                advances=len(b),
                positives=tp + fn,
                precision=tp / (tp + fp) if tp + fp else float("nan"),
                recall=tp / (tp + fn) if tp + fn else float("nan"),
                auc_lead=rank_auc([(x["lead"], x[target]) for x in b]),
                auc_delta=rank_auc([(x["delta"], x[target]) for x in b]),
            )
    return out


# --- statistics -------------------------------------------------------------------------------


def bootstrap_diff(groups, metric, reps=BOOTSTRAPS, seed=7):
    """Paired bootstrap over the groups (utterances, or transitions) of metric(b) - metric(a).
    Each group carries whatever the metric needs for both rules, so the pairing is kept."""
    if not groups:
        return None
    rng = random.Random(seed)
    base = metric(groups)
    draws = []
    n = len(groups)
    for _ in range(reps):
        sample = [groups[rng.randrange(n)] for _ in range(n)]
        draws.append(metric(sample))
    draws.sort()
    lo = draws[int(0.025 * reps)]
    hi = draws[min(reps - 1, int(0.975 * reps))]
    return dict(diff=base, lo=lo, hi=hi)


def f1_metric(state):
    def m(groups):
        a, b = empty_counts(), empty_counts()
        for ga, gb in groups:
            add_counts(a, ga)
            add_counts(b, gb)
        return prf(b[state])[2] - prf(a[state])[2]

    return m


def mean_metric(index):
    def m(groups):
        vals = [g[index] for g in groups if g[index] is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    return m


def mcnemar_pair(a, b):
    """Discordant counts and the exact two-sided p: only_b is where the second rule alone is
    right."""
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    return dict(
        both=sum(1 for x, y in zip(a, b) if x and y),
        only_a=only_a,
        only_b=only_b,
        neither=sum(1 for x, y in zip(a, b) if not x and not y),
        p=sc.mcnemar(only_a, only_b),
    )


def pval(x):
    """An exact p over a thousand discordant pairs underflows a float, where 0 would read as a
    measurement rather than as the bottom of the type."""
    return "< 1e-308" if x == 0 else f"{x:.3g}"


def num(x, places=1):
    if x is None:
        return "-"
    if isinstance(x, float) and x != x:
        return "-"
    return f"{x:.{places}f}"


def share(a, b):
    return f"{a} / {b} ({100.0 * a / b:.1f}%)" if b else "-"


# --- the run ----------------------------------------------------------------------------------


def split_clips(data, name, rng):
    listed = [x.strip() for x in (data / f"{name}_list.txt").read_text().splitlines() if x.strip()]
    keep = set(sc.DATASET_WORDS)
    rows = [(rel.split("/")[0], data / rel) for rel in listed if rel.split("/")[0] in keep]
    rng.shuffle(rows)
    return rows


def decode_split(
    eng,
    data,
    split,
    args,
    gaps_paths,
    vosk_eng=None,
    pause_ms=PAUSE_MS,
    word_target=None,
    tag=None,
    parity=None,
    gap_floor=GAP_FLOOR_DBFS,
):
    """Build and decode this split's streams, one at a time, keeping the truth and the series but
    not the audio."""
    rng = random.Random(args.seed + (0 if split == "validation" else 1))
    clips = split_clips(data, split, rng)
    gaps = GapSource(gaps_paths, gap_floor)
    streams = []
    n_words = 0
    want = args.words if word_target is None else word_target
    name = tag or split
    vosk = dict(rows=[], partial_words=None)
    while n_words < want and len(clips) > UTTERANCES_PER_STREAM * WORDS_PER_UTTERANCE[1]:
        pcm, words, gap_rows = build_stream(rng, clips, gaps, args.utterances, pause_ms)
        states, blocks, finals = run_stream(eng, pcm, args.block_ms, args.alternatives, parity=parity)
        s = dict(
            key=f"{name}/{len(streams)}",
            split=name,
            words=words,
            gaps=gap_rows,
            samples=len(pcm) // 2,
            states=states,
            blocks=blocks,
            finals=finals,
            truth=Truth(words),
        )
        s["transitions"] = transitions_of(s, args.lookback_ms)
        label_blocks(s)
        streams.append(s)
        n_words += len(words)
        if vosk_eng is not None:
            vb, vf = run_stream_text(vosk_eng, pcm, args.block_ms)
            vosk["rows"].append(dict(key=s["key"], blocks=vb, finals=vf))
            if vosk["partial_words"] is None:
                vosk["partial_words"] = dict(
                    vosk=partial_words_check(vosk_eng, pcm, args.block_ms),
                    utter=partial_words_check(eng, pcm, args.block_ms),
                )
        sc.note(f"{name}: stream {len(streams)}, {n_words} words, {s['samples'] / RATE:.0f} s")
    return streams, vosk


def advance_intervals(streams):
    out = []
    for s in streams:
        for a, b in zip(s["states"], s["states"][1:]):
            if a["seg"] == b["seg"]:
                out.append((b["fed"] - a["fed"]) / SAMPLES_PER_MS)
    return out


def fit_thresholds(streams, args):
    """The bound first, on the state it decides; then the two motion thresholds, on theirs.
    Fitted on the validation streams only, at the shorter horizon, by the macro F1 over the four
    states, which a rule that calls one state everywhere cannot win; the per-state figures and
    the false alarms each choice costs are measured on the same streams and reported beside
    it."""

    w = HORIZONS[0]

    def scores_of(per_advance):
        c = empty_counts()
        for s_ in streams:
            for g in group_counts(s_, per_advance[s_["key"]], w).values():
                add_counts(c, g)
        f1 = {state: prf(c[state])[2] for state in STATES}
        return dict(
            f1_away=f1["away"],
            f1_toward=f1["toward"],
            macro_f1=sum(f1.values()) / len(STATES),
            finish_alarms_per_min=alarms(streams, per_advance, "finish", args.block_ms)["per_min"],
        )

    bound_scores = {}
    for b in BOUND_GRID:
        bound_scores[b] = scores_of({s_["key"]: calls(s_["states"], b) for s_ in streams})
    bound = max(bound_scores, key=lambda k: bound_scores[k]["macro_f1"])
    motion = {}
    for d in SIL_DELTA_GRID:
        for e in EXT_DELTA_GRID:
            motion[(d, e)] = scores_of({s_["key"]: calls(s_["states"], bound, d, e) for s_ in streams})
    sil_delta, ext_delta = max(motion, key=lambda k: motion[k]["macro_f1"])
    return dict(
        bound_ms=bound,
        sil_delta=sil_delta,
        ext_delta=ext_delta,
        bound_scores={str(k): v for k, v in bound_scores.items()},
        motion_scores={f"{k[0]},{k[1]}": v for k, v in motion.items()},
    )


def stream_sizes(streams):
    words = sum(len(s["words"]) for s in streams)
    pauses = sum(1 for s in streams for g in s["gaps"] if g["kind"] == "pause")
    finishes = sum(1 for s in streams for g in s["gaps"] if g["kind"] == "finish")
    return dict(
        streams=len(streams),
        words=words,
        utterances=sum(len({w["utt"] for w in s["words"]}) for s in streams),
        pauses=pauses,
        finishes=finishes,
        transitions=sum(len(s["words"]) for s in streams),
        blocks=sum(len(s["blocks"]) for s in streams),
        advances=sum(len(s["states"]) for s in streams),
        finals=sum(len(s["finals"]) for s in streams),
        minutes=sum(s["samples"] for s in streams) / RATE / 60.0,
    )


def recordings_pass(eng, paths, args, bound, sil_delta, ext_delta):
    """The recordings alone, fed continuously through one recognizer as the benchmark's noise
    pass does: every away call here is a false alarm."""
    out = []
    for p in paths:
        pcm = sc.read_pcm(p)
        states, blocks, finals = run_stream(eng, pcm, args.block_ms, args.alternatives)
        s = dict(
            key=f"noise/{p.name}",
            words=[],
            gaps=[],
            samples=len(pcm) // 2,
            states=states,
            blocks=blocks,
            finals=finals,
        )
        out.append(s)
        sc.note(f"noise: {p.name}, {s['samples'] / RATE:.0f} s, {len(states)} advances, {len(finals)} finals")
    pa0 = {s["key"]: calls(s["states"], bound) for s in out}
    pa1 = {s["key"]: calls(s["states"], bound, sil_delta, ext_delta) for s in out}
    return out, pa0, pa1


def wordless_recordings(eng, vosk_eng, paths, args, level):
    """The recordings alone at one floor level, through both engines. `level` None is digital
    silence of the same length, which the floor's percentile treats apart from a room."""
    out = []
    rows = []
    for p in paths:
        pcm = sc.read_pcm(p)
        pcm = bytes(len(pcm)) if level is None else sc.scaled_to_floor(pcm, level)
        states, blocks, finals = run_stream(eng, pcm, args.block_ms, args.alternatives)
        key = f"noise/{p.name}"
        out.append(dict(key=key, words=[], gaps=[], samples=len(pcm) // 2, states=states, blocks=blocks, finals=finals))
        if vosk_eng is not None:
            vb, vf = run_stream_text(vosk_eng, pcm, args.block_ms)
            rows.append(dict(key=key, blocks=vb, finals=vf))
    return out, rows


def write_streams(path, streams):
    with open(path, "w") as fh:
        for s in streams:
            row = dict(
                key=s["key"],
                split=s.get("split"),
                samples=s["samples"],
                words=s["words"],
                gaps=s["gaps"],
                finals=[dict(fed=f["fed"], text=f["text"]) for f in s["finals"]],
                advances=[
                    dict(
                        fed=st["fed"],
                        seg=st["seg"],
                        sil_span=st["sil_span"],
                        words0=st["words0"],
                        grew=st["grew"],
                        readings=[[t, c, st["delta"].get(t), st["relation"][t]] for t, c in st["readings"]],
                    )
                    for st in s["states"]
                ],
                stable=[[b["fed"], b["word"], b["stable"]] for b in s["blocks"] if b["stable"]],
            )
            fh.write(json.dumps(row) + "\n")


GAP_FLOOR_GRID = (-70.0, -60.0, -50.0, -40.0)
GAP_LENGTH_EDGES = (200, 400, 600, 800, 2400, 2700)


def spoken_before(stream, finals):
    """How many of the stream's words have ended since the last final, per block and per final. The
    finals are the deciding engine's own, since its history is what a final of its clears. A
    word the partial still holds is not a phantom; a word beyond them is. A final is judged against
    what was spoken before it closed, since closing is what clears the count."""
    ends = sorted(w["offset"] for w in stream["words"])
    marks = [f["fed"] for f in finals]

    def ended_by(fed):
        return bisect.bisect_right(ends, fed)

    per_block = []
    base = 0
    fi = 0
    for b in stream["blocks"]:
        while fi < len(marks) and marks[fi] <= b["fed"]:
            base = ended_by(marks[fi])
            fi += 1
        per_block.append(ended_by(b["fed"]) - base)
    per_final = []
    base = 0
    for m in marks:
        per_final.append(ended_by(m) - base)
        base = ended_by(m)
    return per_block, per_final


def wordless_spans(stream, kind):
    """The stretches a word would be a phantom in: a built stream's gaps, by kind, or the whole of
    a recording fed cold."""
    if not stream["gaps"]:
        return [(0, stream["samples"], "cold", stream["samples"])] if kind in ("cold", "all") else []
    out = []
    for g in stream["gaps"]:
        if kind not in ("all", g["kind"]):
            continue
        out.append((g["pos"], g["pos"] + g["samples"], g["kind"], g["samples"]))
    return out


GAP_BUCKETS = [f"under {e} ms" for e in GAP_LENGTH_EDGES] + [f"{GAP_LENGTH_EDGES[-1]} ms or more"]


def gap_bucket(samples):
    ms = samples / SAMPLES_PER_MS
    for i, e in enumerate(GAP_LENGTH_EDGES):
        if ms < e:
            return GAP_BUCKETS[i]
    return GAP_BUCKETS[-1]


def census_words(stream, spans, rank0, readings, finals):
    """One stream's census over `spans`, in one walk of the blocks. `rank0(i)` is the rank-0 word
    list at block i, `readings(i)` the rivals there or None where the engine offers none. Keyed by
    the span's kind and length, so a rate can be read against the span's own clock."""
    spoken, spoken_at_final = spoken_before(stream, finals)
    order = sorted(spans, key=lambda x: x[0])
    starts = [x[0] for x in order]
    rows = defaultdict(Counter)
    for _, _, kind, samples in order:
        r = rows[(kind, gap_bucket(samples))]
        r["spans"] += 1
        r["samples"] += samples

    def key_at(fed):
        j = bisect.bisect_left(starts, fed) - 1
        if j < 0:
            return None
        s0, e0, kind, samples = order[j]
        return (kind, gap_bucket(samples)) if s0 < fed <= e0 else None

    run = 0
    last = None
    for i, b in enumerate(stream["blocks"]):
        key = key_at(b["fed"])
        if key is None:
            run, last = 0, None
            continue
        if key != last:
            run, last = 0, key
        r = rows[key]
        r["blocks"] += 1
        if readings is not None:
            r["with_rivals"] += 1
        said = spoken[i]
        if len(rank0(i)) <= said:
            run = 0
            continue
        r["word_blocks"] += 1
        run += 1
        r["longest_run"] = max(r["longest_run"], run)
        rv = readings(i) if readings is not None else None
        if rv is not None and any(len(w) > said for w in rv[1:]):
            r["rival_blocks"] += 1
    for j, f in enumerate(finals):
        key = key_at(f["fed"])
        if key is not None and len(word_list(f["text"])) > spoken_at_final[j]:
            rows[key]["phantom_finals"] += 1
    return rows


def utter_rank0(stream):
    def rank0(i):
        adv = stream["blocks"][i]["adv"]
        return stream["states"][adv]["words0"] if adv is not None else []

    def readings(i):
        adv = stream["blocks"][i]["adv"]
        if adv is None:
            return []
        return [word_list(t) for t, _ in stream["states"][adv]["readings"]]

    return rank0, readings


def vosk_rank0(row):
    return (lambda i: row["blocks"][i]["words"] if i < len(row["blocks"]) else []), None


def census_totals(rows):
    out = Counter()
    for v in rows.values():
        for k, n in v.items():
            if k == "longest_run":
                out[k] = max(out[k], n)
            else:
                out[k] += n
    return out


def census_row(label, engine, c):
    minutes = c["samples"] / RATE / 60.0
    rival = f"{c['rival_blocks'] / minutes:.2f}" if c.get("with_rivals") else "no alternatives"
    return (
        f"| {label} | {engine} | {minutes:.1f} | {c['blocks']} | "
        f"{c['word_blocks'] / minutes if minutes else float('nan'):.2f} | {rival} | "
        f"{c['phantom_finals'] / minutes if minutes else float('nan'):.2f} | {c['longest_run']} |"
    )


def census_header(lines, what):
    lines.append("")
    lines.append(
        f"| {what} | engine | non-speech minutes | blocks | word at rank 0 /min | "
        "word among the rivals /min | finals with a word /min | longest run of blocks |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")


def census_set(streams, vosk_rows, kind):
    """Both engines over one set of streams, keyed by span kind and length bucket."""
    out = {}
    by_key = {r["key"]: r for r in (vosk_rows or [])}
    for s in streams:
        spans = wordless_spans(s, kind)
        if not spans:
            continue
        rank0, readings = utter_rank0(s)
        for key, c in census_words(s, spans, rank0, readings, s["finals"]).items():
            out.setdefault("utterpy", {}).setdefault(key, Counter()).update(c)
        row = by_key.get(s["key"])
        if row is not None:
            vr, _ = vosk_rank0(row)
            for key, c in census_words(s, spans, vr, None, row["finals"]).items():
                out.setdefault("vosk", {}).setdefault(key, Counter()).update(c)
    return out


def wordless_section(report, a, test, noise_streams, vosk_rows, variants, lines):
    """The states page's half of the wordless coverage: a word standing where the stream says
    nothing was said, read off the runtime's own readings rather than off the audio, by the length
    of the pause or finish it stood in and by the floor the gap was built at.
    `[[rr:TD-8#Measurements count a word whose own span carries no speech]]`"""
    lines += [
        "",
        "## Words in the stream's own silences",
        "",
        "The census of `[[rr:Words on wordless audio]]` read off the built streams instead of the "
        "recordings: every gap between words and after the last one, and the recordings fed cold "
        "beside them. A word at rank 0 counts when the partial holds more words than the stream "
        "has finished saying since the last final, so a word still held is not one. Rates are per "
        "minute of the gaps themselves.",
    ]
    out = {}

    lines += ["", "### By the length of the pause or the finish", ""]
    lines.append(f"The page's own testing streams, gaps built at {GAP_FLOOR_DBFS:g} dBFS.")
    census_header(lines, "gap")
    built = census_set(test, vosk_rows, "all")
    for engine in ("vosk", "utterpy"):
        for key in sorted(built.get(engine, {}), key=lambda k: (k[0], GAP_BUCKETS.index(k[1]))):
            c = built[engine][key]
            lines.append(census_row(f"{key[0]}, {key[1]}", engine, c))
            out.setdefault("built", {}).setdefault(engine, {})[f"{key[0]}/{key[1]}"] = dict(c)

    lines += ["", "### The recordings fed cold, by floor level", ""]
    lines.append(
        "The same recordings the gaps are cut from, fed whole through one recognizer, scaled so "
        "their floor sits at each level."
    )
    census_header(lines, "floor")
    for level, streams, rows in noise_streams:
        got = census_set(streams, rows, "cold")
        for engine in ("vosk", "utterpy"):
            if engine not in got:
                continue
            c = census_totals(got[engine])
            lines.append(census_row("digital silence" if level is None else f"{level:g} dBFS", engine, c))
            out.setdefault("recordings", {}).setdefault(engine, {})[str(level)] = dict(c)

    if variants:
        lines += ["", "### The finish gaps, by the floor they were built at", ""]
        lines.append(
            "Testing streams rebuilt with the gaps scaled to each floor, so the stretch after a "
            "spoken word is measured at the level a new microphone would put it."
        )
        census_header(lines, "gap floor")
        for level, streams, rows in variants:
            got = census_set(streams, rows, "finish")
            for engine in ("vosk", "utterpy"):
                if engine not in got:
                    continue
                c = census_totals(got[engine])
                lines.append(census_row(f"{level:g} dBFS", engine, c))
                out.setdefault("gap_floor", {}).setdefault(engine, {})[f"{level:g}"] = dict(c)

    pairs = []
    for s in test:
        spans = wordless_spans(s, "all")
        rank0, readings = utter_rank0(s)
        u = census_totals(census_words(s, spans, rank0, readings, s["finals"]))
        row = {r["key"]: r for r in (vosk_rows or [])}.get(s["key"])
        if row is None:
            continue
        vr, _ = vosk_rank0(row)
        v = census_totals(census_words(s, spans, vr, None, row["finals"]))
        pairs.append((u["word_blocks"] > 0, v["word_blocks"] > 0))
    if pairs:
        b = sum(1 for x, y in pairs if x and not y)
        c = sum(1 for x, y in pairs if y and not x)
        out["mcnemar_engines"] = dict(b=b, c=c, p=sc.mcnemar(b, c))
        lines += [
            "",
            f"Paired over the {len(pairs)} testing streams, a stream carrying any word at rank 0 in "
            f"a gap: utterpy only {b}, vosk only {c}, exact two-sided p = {sc.mcnemar(b, c):.3g}.",
        ]
    lines += [
        "",
        "Three things to read off these tables. The rate is not flat across a gap's length: it is "
        "highest in the short pauses, where the decoder is still inside the utterance and no "
        "endpoint has fired, and it falls as the gap lengthens and the rules close the stretch, "
        "which is the mechanism `[[rr:TD-8#A word on a quiet block was spoken and is reported]]` "
        "describes rather than a second one. The floor the gap is built at moves the finals and "
        "barely moves rank 0, so a host that gates on level is gating the right half. And the two "
        "engines are read on the same blocks here, so any cell where they differ is a one-way "
        "excess and is reported as such; the paired test over streams is the test of it.",
        "",
        "The gap-floor table's own rows are not one sample: the page's own streams carry the whole "
        "testing split and each rebuilt set carries fewer words, so read the rates and not the "
        "block counts across it.",
        "",
    ]
    report["wordless"] = out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--engine", default="utterpy", help="utterpy, or utterpy@MS for a host endpoint rule at MS of trailing silence"
    )
    ap.add_argument("--out", default="docs/benchmarks/partial-states")
    ap.add_argument("--words", type=int, default=2000, help="words per split, at least")
    ap.add_argument("--utterances", type=int, default=UTTERANCES_PER_STREAM)
    ap.add_argument("--block-ms", type=int, default=40)
    ap.add_argument("--alternatives", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument(
        "--lookback-ms",
        type=int,
        default=1000,
        help="how far before a transition a rule's call still counts as forecasting it",
    )
    ap.add_argument("--with-vosk", action="store_true", help="run the stock wheel over the same streams")
    ap.add_argument(
        "--pause-variants",
        default="100-300,400-1600",
        help="extra testing streams built with these pause ranges in ms, for the onset rule alone; empty to skip",
    )
    ap.add_argument("--variant-words", type=int, default=600, help="words per pause-variant stream set")
    ap.add_argument(
        "--wordless-floors",
        default="silence,-70,-60,-50,-40",
        help="floor levels the recordings are scaled to for the wordless census, empty to skip it",
    )
    ap.add_argument(
        "--wordless-gap-floors",
        default="-70,-60,-40",
        help="extra testing streams built with the gaps at these floors, beside the page's own; empty to skip",
    )
    ap.add_argument("--wordless-words", type=int, default=600, help="words per gap-floor variant stream set")
    ap.add_argument("--no-streams", action="store_true", help="skip the per-stream JSON lines")
    ap.add_argument(
        "--bound-runs",
        default="",
        help="JSON written by earlier runs of this script under a host endpoint bound, reported "
        "beside this run's finals; the streams must be the ones this run builds",
    )
    a = ap.parse_args()

    import utterpy

    data = Path(a.data)
    gaps_paths = sorted((data / "_background_noise_").glob("*.wav"))
    grammar = sc.DATASET_WORDS + sc.LETTERS + sc.NATO + sc.COLOURS
    eng = sc.Engine(a.engine, utterpy, a.model, grammar)
    if eng.missing:
        sc.note(f"grammar words the model lacks: {eng.missing}")
    vosk_eng = None
    if a.with_vosk:
        import vosk

        vosk.SetLogLevel(-1)
        vosk_eng = sc.Engine("vosk", vosk, a.model, grammar)

    report = dict(
        provenance=sc.provenance(a, ["utterpy", "vosk"] if a.with_vosk else ["utterpy"]),
        # The stream set is built from the seed and these: a page cannot be reproduced without them.
        args=vars(a),
        wheel=utterpy.__file__,
        grammar_size=len(grammar),
        seed=a.seed,
        alternatives=a.alternatives,
        build=dict(
            pause_ms=PAUSE_MS,
            finish_ms=FINISH_MS,
            words_per_utterance=WORDS_PER_UTTERANCE,
            gap_floor_dbfs=GAP_FLOOR_DBFS,
            gap_sources=[p.name for p in gaps_paths],
            utterances_per_stream=a.utterances,
        ),
    )

    t0 = time.monotonic()
    parity = empty_parity()
    sc.note("validation streams (thresholds are fitted here)")
    val, _ = decode_split(eng, data, "validation", a, gaps_paths, parity=parity)
    sc.note("testing streams (every reported figure comes from here)")
    test, vosk = decode_split(eng, data, "testing", a, gaps_paths, vosk_eng, parity=parity)
    report["parity"] = parity
    report["sizes"] = dict(validation=stream_sizes(val), testing=stream_sizes(test))
    report["decode_seconds"] = time.monotonic() - t0

    sc.note("fitting on validation")
    chosen = fit_thresholds(val, a)
    report["fit"] = chosen
    bound, sil_delta, ext_delta = chosen["bound_ms"], chosen["sil_delta"], chosen["ext_delta"]

    intervals = advance_intervals(test)
    report["advance_ms"] = dict(
        n=len(intervals),
        p50=sc.quantile(intervals, 0.5),
        p90=sc.quantile(intervals, 0.9),
        mean=sum(intervals) / len(intervals) if intervals else float("nan"),
        histogram={str(k): v for k, v in sorted(Counter(round(x) for x in intervals).items())},
    )

    pa = {
        "R0": {s["key"]: calls(s["states"], bound) for s in test},
        "R1": {s["key"]: calls(s["states"], bound, sil_delta, ext_delta) for s in test},
        "R0@300": {s["key"]: calls(s["states"], TD8_BOUND_MS) for s in test},
    }

    # A1: per state precision and recall at each horizon.
    a1 = {}
    groups = {}
    for w_ms in HORIZONS:
        per_rule = {}
        for rule in ("R0", "R1"):
            total = empty_counts()
            g = {}
            for s in test:
                for utt, c in group_counts(s, pa[rule][s["key"]], w_ms).items():
                    add_counts(total, c)
                    g[(s["key"], utt)] = c
            per_rule[rule] = total
            groups[(rule, w_ms)] = g
        a1[str(w_ms)] = {}
        for rule, counts in per_rule.items():
            a1[str(w_ms)][rule] = {}
            for state in STATES:
                p_, r_, f_ = prf(counts[state])
                a1[str(w_ms)][rule][state] = dict(counts[state], precision=p_, recall=r_, f1=f_)
    report["A1"] = a1

    # A1 paired bootstrap on the F1 of the two transition states.
    boot = {}
    for w_ms in HORIZONS:
        keys = sorted(set(groups[("R0", w_ms)]) & set(groups[("R1", w_ms)]))
        pairs = [(groups[("R0", w_ms)][k], groups[("R1", w_ms)][k]) for k in keys]
        for state in ("away", "toward"):
            boot[f"{state}@{w_ms}"] = bootstrap_diff(pairs, f1_metric(state))
    report["A1_bootstrap"] = boot

    # A2: lead time per true transition, and McNemar at the horizon's tolerance.
    a2 = {}
    handled = {}
    for rule in ("R0", "R1"):
        lt, hd = lead_times(test, pa[rule], HORIZONS[0] * SAMPLES_PER_MS)
        handled[rule] = hd
        a2[rule] = {
            kind: dict(
                n=len(v),
                called=sum(1 for x in v if x is not None),
                before=sum(1 for x in v if x is not None and x < 0),
                p10=sc.quantile([x for x in v if x is not None], 0.1),
                p50=sc.quantile([x for x in v if x is not None], 0.5),
                p90=sc.quantile([x for x in v if x is not None], 0.9),
            )
            for kind, v in lt.items()
        }
        a2[rule]["_lead"] = {k: [x for x in v] for k, v in lt.items()}
    report["A2"] = {rule: {k: v for k, v in d.items() if k != "_lead"} for rule, d in a2.items()}
    report["A2_lead_ms"] = {rule: d["_lead"] for rule, d in a2.items()}
    report["A2_mcnemar"] = {kind: mcnemar_pair(handled["R0"][kind], handled["R1"][kind]) for kind in ("away", "toward")}

    # A3: false alarms.
    report["A3"] = dict(
        finish={r: alarms(test, pa[r], "finish", a.block_ms) for r in ("R0", "R1")},
    )
    sc.note("noise recordings")
    noise_streams, npa0, npa1 = recordings_pass(eng, gaps_paths, a, bound, sil_delta, ext_delta)
    report["A3"]["noise"] = dict(
        R0=alarms(noise_streams, npa0, "all", a.block_ms),
        R1=alarms(noise_streams, npa1, "all", a.block_ms),
        finals=sum(len(s_["finals"]) for s_ in noise_streams),
        finals_with_a_word=sum(1 for s_ in noise_streams for f in s_["finals"] if sc.words_of(f["text"])),
        minutes=sum(s["samples"] for s in noise_streams) / RATE / 60.0,
    )

    # A4: the end-of-speech confusion curve.
    rows, totals, early = eos_curve(test, EOS_GRID, ext_delta)
    report["A4"] = dict(
        totals=dict(totals),
        curve={
            str(x): dict(
                pauses_mistaken=rows[x]["pause"],
                pauses_mistaken_motion=rows[x]["pause_motion"],
                finishes_called=rows[x]["finish"],
                finishes_called_motion=rows[x]["finish_motion"],
                early_n=len(early[x]),
                early_ms_p50=sc.quantile(early[x], 0.5),
            )
            for x in EOS_GRID
        },
    )

    report["A4_by_pause"] = dict(
        bound_ms=bound,
        td8_bound=eos_by_gap_length(test, TD8_BOUND_MS, ext_delta),
        chosen=eos_by_gap_length(test, bound, ext_delta),
    )

    # A5: the same stretches, read off the finals the runtime emitted rather than off the series.
    report["A5"] = runtime_finals(test)
    report["A5_bound"] = {}
    for path in [x for x in a.bound_runs.split(",") if x.strip()]:
        other = json.loads(Path(path).read_text())
        report["A5_bound"][other["args"]["engine"]] = dict(
            other["A5"],
            wheel_revision=other["provenance"]["wheel_revision"],
            same_streams=built_alike(other["sizes"]["testing"], report["sizes"]["testing"]),
        )

    # B: the transitions.
    rec, previews = transition_records(test, ext_delta, a.lookback_ms)
    mid = [r for r in rec if not r["first_in_utterance"]]
    arrived = [r for r in rec if r["arrive_ms"] is not None]
    led = [r for r in rec if r["leads_ms"] is not None]
    with_ext = [r for r in arrived if r["preview_ms"] is not None]
    report["B1"] = dict(
        transitions=len(rec),
        mid_utterance=len(mid),
        arrived=len(arrived),
        led=len(led),
        with_extender=len(with_ext),
        zero_lead=sum(1 for r in with_ext if r["preview_advances"] == 0),
        advances_p50=sc.quantile([r["preview_advances"] for r in with_ext], 0.5),
        advances_p90=sc.quantile([r["preview_advances"] for r in with_ext], 0.9),
        ms_p50=sc.quantile([r["preview_ms"] for r in with_ext], 0.5),
        ms_p90=sc.quantile([r["preview_ms"] for r in with_ext], 0.9),
        rank0_empty=sum(1 for r in with_ext if r["ext_rank0_empty"]),
        correct_lead_n=sum(1 for r in arrived if r["preview_correct_ms"] is not None),
        correct_lead_ms_p50=sc.quantile(
            [r["preview_correct_ms"] for r in arrived if r["preview_correct_ms"] is not None], 0.5
        ),
        correct_lead_ms_p90=sc.quantile(
            [r["preview_correct_ms"] for r in arrived if r["preview_correct_ms"] is not None], 0.9
        ),
    )
    by_position = {}
    for name, rows_ in (("first in utterance", [r for r in rec if r["first_in_utterance"]]), ("mid utterance", mid)):
        w = [r for r in rows_ if r["preview_ms"] is not None]
        by_position[name] = dict(
            n=len(rows_),
            arrived=sum(1 for r in rows_ if r["arrive_ms"] is not None),
            with_extender=len(w),
            zero_lead=sum(1 for r in w if r["preview_advances"] == 0),
            ms_p50=sc.quantile([r["preview_ms"] for r in w], 0.5),
            top1=sum(1 for r in w if r["ext_top"] == r["label"]),
            top3=sum(1 for r in w if r["label"] in (r["ext_top3"] or [])),
            arrive_correct=sum(1 for r in rows_ if r["arrive_word"] == r["label"]),
        )
    report["B2"] = by_position
    report["B2_advances"] = len(previews)
    report["B2_by_lead"] = {}
    subsets = (
        ("all", previews, "top1", "top3"),
        ("over a word", [q for q in previews if not q["rank0_empty"]], "top1", "top3"),
        ("gaining", [q for q in previews if q["gaining"]], "gain_top1", "gain_top3"),
    )
    for lo, hi in ((0, 240), (240, 480), (480, 720), (720, 1000), (1000, float("inf"))):
        for where, rows_, k1, k3 in subsets:
            b = [q for q in rows_ if lo <= q["lead_ms"] < hi]
            report["B2_by_lead"].setdefault(f"{lo:g}-{hi:g}", {})[where] = dict(
                n=len(b),
                top1=sum(1 for q in b if q[k1]),
                top3=sum(1 for q in b if q[k3]),
            )
    report["B3"] = dict(
        finish=dying_extenders(
            test, lambda s_: [(g["pos"], g["pos"] + g["samples"]) for g in s_["gaps"] if g["kind"] == "finish"]
        ),
        noise=dying_extenders(noise_streams, lambda s_: [(0, s_["samples"])]),
    )
    inner = [(r["inner_sil_ms"], r["elapsed_ms"]) for r in mid if r["inner_sil_ms"] is not None]
    err = [x - y for x, y in inner]
    span, decoded = span_errors(test)
    report["B4"] = dict(
        inner_n=len(inner),
        inner_mean_err=sum(err) / len(err) if err else float("nan"),
        inner_p50_err=sc.quantile(err, 0.5),
        inner_p90_abs_err=sc.quantile([abs(e) for e in err], 0.9),
        trailing={
            k: dict(
                n=len(v),
                mean_err=sum(v) / len(v) if v else float("nan"),
                p50_err=sc.quantile(v, 0.5),
                p90_abs_err=sc.quantile([abs(x) for x in v], 0.9),
                decoded_mean_err=sum(decoded[k]) / len(decoded[k]) if decoded[k] else float("nan"),
                decoded_p50_err=sc.quantile(decoded[k], 0.5),
                decoded_p10_err=sc.quantile(decoded[k], 0.1),
                decoded_p90_err=sc.quantile(decoded[k], 0.9),
            )
            for k, v in span.items()
        },
    )
    lat = [r["latency_ms"] for r in led]
    stab = [r["stable_ms_from_onset"] for r in led if r["stable_ms_from_onset"] is not None]
    report["B5"] = dict(
        n=len(lat),
        transitions=len(rec),
        rank0_p50=sc.quantile(lat, 0.5),
        rank0_p90=sc.quantile(lat, 0.9),
        stable_n=len(stab),
        stable_p50=sc.quantile(stab, 0.5),
        stable_p90=sc.quantile(stab, 0.9),
        mid_rank0_p50=sc.quantile([r["latency_ms"] for r in mid if r["latency_ms"] is not None], 0.5),
        mid_n=len(mid),
        finals_in_pause=sum(1 for r in mid if r["final_between"]),
    )
    # B before and after: R0 reads the transition when rank 0 changes, R1 reads the extending
    # reading's extra word where there is one, and falls back to R0 where there is not.
    unit = [r for r in rec if r["arrive_ms"] is not None]
    ok0 = [r["arrive_word"] == r["label"] for r in unit]
    ok1 = [
        (r["ext_top"] == r["label"]) if r["preview_ms"] is not None else (r["arrive_word"] == r["label"]) for r in unit
    ]
    okg = [
        (r["gext_top"] == r["label"]) if r["gaining_preview_ms"] is not None else (r["arrive_word"] == r["label"])
        for r in unit
    ]
    report["B_mcnemar"] = mcnemar_pair(ok0, ok1)
    report["B_mcnemar_gaining"] = mcnemar_pair(ok0, okg)
    call0 = [r["arrive_ms"] - r["onset_ms"] for r in unit]
    call1 = [(r["ext_ms"] - r["onset_ms"]) if r["preview_ms"] is not None else c for r, c in zip(unit, call0)]
    callg = [(r["gext_ms"] - r["onset_ms"]) if r["gaining_preview_ms"] is not None else c for r, c in zip(unit, call0)]
    report["B_bootstrap_call_ms"] = bootstrap_diff(
        list(zip(call0, call1)), lambda g: mean_metric(1)(g) - mean_metric(0)(g)
    )
    report["B_bootstrap_call_ms_gaining"] = bootstrap_diff(
        list(zip(call0, callg)), lambda g: mean_metric(1)(g) - mean_metric(0)(g)
    )
    report["B_gaining"] = dict(
        with_gaining=sum(1 for r in unit if r["gaining_preview_ms"] is not None),
        ms_p50=sc.quantile([r["gaining_preview_ms"] for r in unit if r["gaining_preview_ms"] is not None], 0.5),
        ms_p90=sc.quantile([r["gaining_preview_ms"] for r in unit if r["gaining_preview_ms"] is not None], 0.9),
    )
    report["B_call_ms"] = dict(
        n=len(unit),
        R0_p50=sc.quantile(call0, 0.5),
        R1_p50=sc.quantile(call1, 0.5),
        Rg_p50=sc.quantile(callg, 0.5),
        R0_p90=sc.quantile(call0, 0.9),
        R1_p90=sc.quantile(call1, 0.9),
        Rg_p90=sc.quantile(callg, 0.9),
    )

    # C: the readings' momentum, all of it off the series already recorded.
    report["C1_momentum"] = momentum(test)
    report["C2_silence"] = dict(
        finish_interiors=silence_signature(test, kept_silence_spans),
        recordings=silence_signature(noise_streams, whole_stream_spans),
    )
    report["C3_aligned"] = aligned_velocity(test)
    cross = {}
    for signal, grid in (("sil", SIL_CROSS_GRID), ("extends", EXT_CROSS_GRID)):
        fitted = {t: crossing_scores(val, signal, t, a.lookback_ms, a.block_ms) for t in grid}
        theta = max(fitted, key=lambda t: fitted[t]["f1"])
        cross[signal] = dict(
            grid={str(t): v for t, v in fitted.items()},
            theta=theta,
            testing=crossing_scores(test, signal, theta, a.lookback_ms, a.block_ms),
            recordings=crossing_scores(noise_streams, signal, theta, a.lookback_ms, a.block_ms),
        )
    report["C3_crossing"] = cross
    report["C5_onset"] = onset_study(val, test, noise_streams)
    variants = {}
    for spec in [x for x in a.pause_variants.split(",") if x.strip()]:
        lo, hi = (float(x) for x in spec.split("-"))
        sc.note(f"pause variant {spec} ms")
        vstreams, _ = decode_split(
            eng,
            data,
            "testing",
            a,
            gaps_paths,
            pause_ms=(lo, hi),
            word_target=a.variant_words,
            tag=f"pause{spec}",
            parity=parity,
        )
        theta = report["C5_onset"]["best_theta"]["motion"]
        variants[spec] = dict(
            sizes=stream_sizes(vstreams),
            motion=onset_scores(vstreams, "motion", theta),
            presence=onset_scores(vstreams, "presence", 0.0),
        )
    report["C5_onset"]["pause_variants"] = variants
    report["C4_reversal"] = reversal(test)

    if vosk["rows"]:
        report["vosk"] = vosk_baseline(test, vosk["rows"], a.lookback_ms)
        report["vosk"]["partial_words"] = vosk["partial_words"]

    out = Path(a.out)
    if not a.no_streams:
        write_streams(out.with_suffix(".streams.jsonl"), test + noise_streams)
    lines = page(report, a, chosen)
    floors = [None if v.strip() == "silence" else float(v) for v in a.wordless_floors.split(",") if v.strip()]
    if floors:
        sc.note("wordless census: the recordings at each floor")
        noise_sets = [(level, *wordless_recordings(eng, vosk_eng, gaps_paths, a, level)) for level in floors]
        variants = [(GAP_FLOOR_DBFS, test, vosk["rows"])]
        for spec in [x for x in a.wordless_gap_floors.split(",") if x.strip()]:
            level = float(spec)
            sc.note(f"wordless census: streams with the gaps at {level:g} dBFS")
            vs, vk = decode_split(
                eng,
                data,
                "testing",
                a,
                gaps_paths,
                vosk_eng,
                word_target=a.wordless_words,
                tag=f"gapfloor{level:g}",
                gap_floor=level,
            )
            variants.append((level, vs, vk["rows"]))
        variants.sort(key=lambda x: x[0])
        wordless_section(report, a, test, noise_sets, vosk["rows"], variants, lines)
    reading = out.with_suffix(".reading.md")
    if reading.exists():
        lines += [reading.read_text().strip(), ""]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    out.with_suffix(".json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    sc.note(f"wrote {out.with_suffix('.md')}, {out.with_suffix('.json')}")


def vosk_baseline(streams, vosk_rows, lookback_ms, forward_ms=1500):
    """The stock wheel over the same streams: when a word first leads its partial, whether it is
    the right one, and how many finals the same audio produces. It has no readings, so it has no
    motion, and the same window bounds the search."""
    by_key = {r["key"]: r for r in vosk_rows}
    back = lookback_ms * SAMPLES_PER_MS
    fwd = forward_ms * SAMPLES_PER_MS
    lat = []
    label_lat = []
    correct = 0
    n = 0
    for s_ in streams:
        r = by_key.get(s_["key"])
        if r is None:
            continue
        grew = []
        led = []
        prev = 0
        for b in r["blocks"]:
            if b["final"]:
                prev = 0
                continue
            if len(b["words"]) > prev:
                grew.append((b["fed"], b["words"][-1]))
            if b["words"]:
                led.append((b["fed"], b["words"][-1]))
            prev = len(b["words"])
        for i, w in enumerate(s_["words"]):
            prev_end = s_["words"][i - 1]["offset"] if i else 0
            nxt = s_["words"][i + 1]["onset"] if i + 1 < len(s_["words"]) else s_["samples"]
            lo = max(prev_end, w["onset"] - back)
            hi = min(nxt, w["onset"] + fwd)
            n += 1
            hit = next((g for g in grew if lo <= g[0] < hi), None)
            if hit is not None:
                lat.append((hit[0] - w["onset"]) / SAMPLES_PER_MS)
                correct += hit[1] == w["label"]
            right = next((g for g in led if lo <= g[0] < hi and g[1] == w["label"]), None)
            if right is not None:
                label_lat.append((right[0] - w["onset"]) / SAMPLES_PER_MS)
    return dict(
        transitions=n,
        arrived=len(lat),
        correct=correct,
        rank0_p50=sc.quantile(lat, 0.5),
        rank0_p90=sc.quantile(lat, 0.9),
        led=len(label_lat),
        label_p50=sc.quantile(label_lat, 0.5),
        label_p90=sc.quantile(label_lat, 0.9),
        finals=sum(len(by_key[s_["key"]]["finals"]) for s_ in streams if s_["key"] in by_key),
    )


def partial_words_check(eng, pcm, block_ms):
    """How many blocks carry partial text with partial words on and off. The stock wheel drops
    the in-progress word from its partial when words are requested, so its word times cannot be
    read on this audio at all; utter's partial is unchanged either way."""
    out = {}
    for words_on in (False, True):
        blocks, _ = run_stream_text(eng, pcm, block_ms, words_on)
        out["on" if words_on else "off"] = dict(blocks=len(blocks), with_text=sum(1 for b in blocks if b["words"]))
    return out


# --- the page ---------------------------------------------------------------------------------


def momentum_section(r, chosen):
    """What a reading's motion does over many advances, which a one-second clip cannot show."""
    o = r["C5_onset"]
    L = [
        "## C. The readings' motion over many advances",
        "",
        "A clip of one word holds three or four advances, so a reading's motion can be read once "
        "there and not followed. On a stream a reading lives through a pause and a finish, and "
        "these are the questions that needs.",
        "",
        "### Does a lead's motion carry to the next advance",
        "",
        "Pearson r and sign agreement between a reading's `lead_delta` at one advance and at the "
        "next, by the band its lead sits in (the absolute value, so a rival's deficit and a "
        "leader's gap fall in the same bands) and by what the reading is to rank 0. Each pair "
        "needs three advances, since a delta is itself a difference.",
        "",
        "| readings | band (nats) | pairs | r | sign agreement |",
        "|---|---|---|---|---|",
    ]
    for subset in MOMENTUM_SUBSETS:
        for band in ("all",) + tuple(name for name, _, _ in BANDS):
            d = r["C1_momentum"].get(f"{subset}|{band}")
            if not d or not d["n"]:
                continue
            L.append(
                f"| {subset} | {band} | {d['n']} | {num(d['r'], 3) if d['r'] is not None else '-'} | "
                f"{num(100 * d['sign_agreement']) + '%' if d['sign_agreement'] is not None else '-'} |"
            )
    sig = r["C2_silence"]
    L += [
        "",
        "### What the readings do in silence that is being kept",
        "",
        "Inside a finish gap, a second after the word has ended and half a second before "
        "anything begins, and on the recordings from one second in: the leading reading's lead "
        "over its best rival, that lead's motion, and how old the rival is. A lead that hovers "
        "while the rival stays young is a bound set by word hypotheses that keep being started "
        "and pruned, not a lead that is being won.",
        "",
        "| audio | advances | `[sil]` at rank 0 | lead mean / sd | lead p10 / p50 / p90 | velocity mean / sd | velocity p10 / p50 / p90 | rival age ms p50 / p90 | rival one advance old |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, key in (("finish gaps, interior", "finish_interiors"), ("recordings alone", "recordings")):
        d = sig[key]
        lead, vel, age = d["lead"], d["velocity"], d["rival_age_ms"]
        L.append(
            f"| {name} | {d['advances']} | {share(d['sil_at_rank0'], d['advances'])} | "
            f"{num(lead['mean'], 2)} / {num(lead['sd'], 2)} | "
            f"{num(lead['p10'], 2)} / {num(lead['p50'], 2)} / {num(lead['p90'], 2)} | "
            f"{num(vel['mean'], 3)} / {num(vel['sd'], 3)} | "
            f"{num(vel['p10'], 2)} / {num(vel['p50'], 2)} / {num(vel['p90'], 2)} | "
            f"{num(age['p50'])} / {num(age['p90'])} | {share(d['rival_fresh'], d['rival_n'])} |"
        )
    L += [
        "",
        "### The motion around a true transition",
        "",
        "Every ground-truth onset and offset aligned at the advance that first covers it, with "
        "the median `lead_delta` at the advances around it: the empty reading's where rank 0 "
        "carries no word, the leading reading's, and the best one-word extending reading's where "
        "rank 0 carries a word. An advance is 240 ms.",
        "",
        "| transition | advance | `[sil]` velocity (n) | leader velocity (n) | extending velocity (n) |",
        "|---|---|---|---|---|",
    ]
    for kind in ("away", "toward"):
        for off in ALIGN_OFFSETS:
            d = r["C3_aligned"].get(f"{kind}|{off}")
            if not d:
                continue
            cells = []
            for name in ("sil", "leader", "extends"):
                x = d.get(name)
                cells.append(f"{num(x['p50'], 2)} ({x['n']})" if x and x["n"] else "-")
            L.append(f"| {kind} | {off:+d} | " + " | ".join(cells) + " |")
    L += [
        "",
        "### Calling a word before it arrives: exploratory",
        "",
        "**Everything in this section is exploratory.** The streams are isolated read words "
        "spliced together with built gaps, and a rule that fires in a gap can be reading the "
        "stream's construction rather than the word that follows. Nothing here is a recommended "
        "read, and the record says so "
        "(`[[rr:TD-9#Every reading carries its lead's motion]]`).",
        "",
        "The rule under test is the README's, and exactly it: a reading that is rank 0 plus one "
        "word and is gaining on the field, which before a first word is any word-carrying reading, "
        "since every one of them extends the empty reading. A run of consecutive advances where "
        "the rule holds is one call. Each call takes at most one word and each word at most one "
        "call: calls in time order claim the earliest unclaimed word whose onset falls between one "
        f"advance before the call and {ONSET_BACK_MS} ms after it. Every other call is a false "
        "alarm, counted against all the non-speech in the streams, the audio right after a word "
        "included, and every unclaimed word is a miss.",
        "",
        "Four baselines are scored the same way, because a rule that calls more words by calling "
        "more often has shown nothing about its signal: the same candidate merely *existing*, the "
        "candidate's *lead* rather than its motion, the trailing `[sil]` span, and a clock the "
        "host keeps itself since it last saw rank 0 grow. Each threshold is fitted on the "
        "validation streams twice, once for its own best F1 and once to spend the motion rule's "
        f"false alarms ({num(o['alarm_budget_per_min'], 2)} a non-speech minute there), and read "
        "on the testing streams:",
        "",
        "| rule | threshold | calls | words called | precision | F1 | false alarms/min | at the motion rule's alarm rate: threshold | words called | F1 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rule in ("motion", "presence", "level", "silence", "timing"):
        t = o["testing"][rule]
        e = o["testing_equal_alarm"][rule]
        L.append(
            f"| {ONSET_LABELS[rule]} | {num(float(o['best_theta'][rule]), 2)} | {t['calls']} | "
            f"{share(t['matched'], t['onsets'])} | {num(100 * t['precision'])}% | {num(t['f1'], 3)} | "
            f"{num(t['alarms_per_min'], 2)} | {num(float(o['equal_alarm_theta'][rule]), 2)} | "
            f"{share(e['matched'], e['onsets'])} | {num(e['f1'], 3)} |"
        )
    m = o["testing"]["motion"]
    pres = o["testing"]["presence"]
    L += [
        "",
        "Then the half of the figure that matters most, and the reason the earlier reading of it "
        "is withdrawn. A call before a word's onset is not evidence about that word unless some of "
        "the word has been fed. Splitting the motion rule's calls by what the decoder had heard "
        "when it fired:",
        "",
        "| the call landed | motion | the candidate merely existing |",
        "|---|---|---|",
        f"| before any sample of the coming word's clip | {share(m['before_clip'], m['matched'])} | "
        f"{share(pres['before_clip'], pres['matched'])} |",
        f"| inside the clip, before its energy onset | {share(m['in_clip_before_onset'], m['matched'])} | "
        f"{share(pres['in_clip_before_onset'], pres['matched'])} |",
        f"| at or after the energy onset, a detection | {share(m['after_onset'], m['matched'])} | "
        f"{share(pres['after_onset'], pres['matched'])} |",
        "",
        f"The lead of the calls that do come first is {num(m['lead_p10'])} / {num(m['lead_p50'])} / "
        f"{num(m['lead_p90'])} ms (p10 / p50 / p90), but a lead measured over calls that mostly "
        "precede the word's clip entirely is not an acoustic warning of that word: it is a call "
        "made on the silence before it, which on this stream is a built gap of known length. That "
        "figure is reported here and **is not** to be read as the decoder hearing a word coming.",
        "",
        "Because the pauses are built, the alarm rate and the calls are partly a fact about the "
        "distribution they were built from. The same rule at the same threshold on testing streams "
        "built with other pause ranges:",
        "",
        "| pauses built (ms) | words | non-speech minutes | words called | precision | false alarms/min | called before the clip |",
        "|---|---|---|---|---|---|---|",
        f"| {PAUSE_MS[0]:g}-{PAUSE_MS[1]:g} (the page's streams) | {r['sizes']['testing']['words']} | "
        f"{num(m['nonspeech_minutes'])} | {share(m['matched'], m['onsets'])} | "
        f"{num(100 * m['precision'])}% | {num(m['alarms_per_min'], 2)} | "
        f"{share(m['before_clip'], m['matched'])} |",
    ]
    for spec, d in (o.get("pause_variants") or {}).items():
        v = d["motion"]
        L.append(
            f"| {spec} | {d['sizes']['words']} | {num(v['nonspeech_minutes'])} | "
            f"{share(v['matched'], v['onsets'])} | {num(100 * v['precision'])}% | "
            f"{num(v['alarms_per_min'], 2)} | {share(v['before_clip'], v['matched'])} |"
        )
    L += [
        "",
        "What this section establishes is a comparison, not a forecast: whether the motion beats "
        "candidate presence, candidate level, elapsed silence and a clock, at the alarm rate it "
        "costs, on spliced words. Whether any of it survives on continuous recorded commands is "
        "not measured here and is the work the record leaves open.",
        "",
        "### The two crossings, as fitted signals",
        "",
        "The same two motions read as raw crossings rather than as the README's rule, kept for "
        "the `[sil]` half: a crossing is one advance where the empty reading's lead falls past a "
        "threshold, or a one-word extending reading's lead rises past one, scored by the older "
        "many-to-one rule (a crossing is a hit when any onset follows within the lookback). The "
        "`[sil]` row is why the README's read is *an extends reading gaining* and never *`[sil]` "
        "falling*.",
        "",
        "| signal | threshold (nats) | crossings | precision | onsets called | lead p10 / p50 / p90 ms | alarms/min in kept silence | alarms/min on the recordings |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for signal, label in (("sil", "`[sil]` lead falling past"), ("extends", "extending lead rising past")):
        c = r["C3_crossing"][signal]
        t, rec = c["testing"], c["recordings"]
        L.append(
            f"| {label} | {float(c['theta']):+.2f} | {t['crossings']} | {num(100 * t['precision'])}% | "
            f"{share(t['called'], t['onsets'])} | {num(t['lead_p10'])} / {num(t['lead_p50'])} / "
            f"{num(t['lead_p90'])} | {num(t['alarms_per_min'])} | {num(rec['alarms_per_min'])} |"
        )

    L += [
        "",
        "### Leading but losing",
        "",
        "Rank 0's lead is never negative, so the reversal `[[rr:TD-9#Every reading carries its "
        "lead's motion]]` describes, a reading whose lead and whose motion disagree in sign, is "
        "rank 0 with a negative `lead_delta`. Scored against rank 0 changing at the next advance, "
        "and against rank 0's words already differing from the final that closes the segment, "
        "beside the rank-AUC of the level and of the motion on the same advances. 0.5 is no "
        "information and below 0.5 the signal is inverted, as on the trust page: a lead that is "
        "small, or a motion that is negative, goes with the change.",
        "",
        "| band (nats) | target | advances | positives | precision | recall | AUC of the lead | AUC of `lead_delta` |",
        "|---|---|---|---|---|---|---|---|",
    ]
    names = {"changes": "rank 0 changes next", "unlike_final": "rank 0 unlike the final"}
    for band in ("all",) + tuple(name for name, _, _ in BANDS):
        for target, label in names.items():
            d = r["C4_reversal"].get(f"{band}|{target}")
            if not d:
                continue
            L.append(
                f"| {band} | {label} | {d['advances']} | {d['positives']} | "
                f"{num(100 * d['precision'])}% | {num(100 * d['recall'])}% | "
                f"{num(d['auc_lead'], 3) if d['auc_lead'] is not None else '-'} | "
                f"{num(d['auc_delta'], 3) if d['auc_delta'] is not None else '-'} |"
            )
    L.append("")
    return L


def _bound_word_cost(rows):
    """What the bound rows cost in words, read off the rows themselves so the sentence cannot
    drift from the table above it."""
    base = rows[0][1]
    lost = [d["lost"] for _, d in rows[1:]]
    twice = sorted({d["restarted"] for _, d in rows})
    span = f"{min(lost)}" if min(lost) == max(lost) else f"{min(lost)} to {max(lost)}"
    same = f"the same {twice[0]}" if len(twice) == 1 else f"{twice[0]} to {twice[-1]}"
    return (
        f"Below 300 ms the price is not in the words. Against the {base['lost']} of "
        f"{base['words']} the stock rules leave no final covering, the bounds leave {span}, and "
        f"the count two finals each decode is {same} at every row, the stock one included: what a "
        "bound shorter than 300 ms gives up is finishes called, not words."
    )


def runtime_finals_section(r, a):
    """The finals the runtime emitted over the stretches the curve above measures, for this run's
    engine and for any bound run passed beside it."""
    rows = [(a.engine, r["A5"])] + sorted(r.get("A5_bound", {}).items())
    L = [
        "",
        "### The same stretches, from the finals the runtime emitted",
        "",
        "The table above reads the trailing `[sil]` span a host would apply its own bound to. "
        "This one reads what the recognizer did: a final landing between a word's energy offset "
        "and the next word's onset is a pause the runtime ended, a final inside a finish gap is a "
        "finish it called. The bound rows are the same streams decoded again with the "
        "recognizer's own host bound set (`[[rr:TD-8#A host may add one endpoint bound of its "
        "own, off by default]]`), `MS` of trailing silence and, after the slash, the margin in "
        "nats within which a reading extending the partial vetoes the final.",
        "",
        "| engine | pauses ended | finishes called | ms after the word's energy end p50 / p90 | "
        "finals | with a word | per minute | words lost | words decoded twice |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, d in rows:
        tot = d["totals"]
        L.append(
            f"| {name} | {share(d['pauses_mistaken'], tot.get('pause', 0))} | "
            f"{share(d['finishes_called'], tot.get('finish', 0))} | "
            f"{num(d['finish_ms_p50'])} / {num(d['finish_ms_p90'])} | {d['finals']} | "
            f"{d['finals_with_a_word']} | {d['finals_per_min']:.1f} | "
            f"{share(d['lost'], d['words'])} | {share(d['restarted'], d['words'])} |"
        )
    stale = [name for name, d in r.get("A5_bound", {}).items() if not d["same_streams"]]
    if stale:
        L += ["", f"**{', '.join(stale)} ran over a different stream set; its row is not paired.**"]
    L += [
        "",
        "The first row is the model's own endpointing (`[[rr:TD-2#Inputs: configuration]]`), "
        "which on this stream ends most pauses already. A bound is a gain only where it calls the "
        "finishes sooner than that without ending more pauses and without costing words, and the "
        "last two columns are where a bound firing inside a word shows up: a word no final "
        "covers, or one that two of them each decode.",
        "",
        _bound_word_cost(rows),
        "",
        "A bound shorter than one decoding advance does not thereby fire at the first advance "
        "that sees silence. The rule is read once per advance, 240 ms at this block size, but the "
        "trailing silence it compares against is counted in the model's own subsampled frames of "
        "30 ms (`[[rr:endpoint_detected]]`), and at the first advance holding any trailing "
        "silence that span already stands anywhere from 30 ms to a full advance: 100 ms fires at "
        "that advance on the stretches where it does and one advance later on the rest, so it "
        "neither coincides with every shorter bound nor with 300.",
        "",
        "By the length of the pause that was built, as the series table above gives it:",
        "",
        "| pause built (ms) | " + " | ".join(name for name, _ in rows) + " |",
        "|---|" + "---|" * len(rows),
    ]
    for key in r["A5"]["by_pause"]:
        cells = []
        for _, d in rows:
            c = d["by_pause"].get(key, dict(n=0, mistaken=0))
            cells.append(share(c["mistaken"], c["n"]))
        L.append(f"| {key} | " + " | ".join(cells) + " |")
    return L


def page(r, a, chosen):
    p = r["provenance"]
    sz = r["sizes"]["testing"]
    vz = r["sizes"]["validation"]
    adv = r["advance_ms"]
    par = r["parity"]
    L = [
        "# Silence direction and word transitions on a built stream",
        "",
        sc.provenance_line(p),
        "",
        f"Grammar: the dataset's 35 words plus 26 letters, 26 NATO words and 5 colours, "
        f"{r['grammar_size']} entries. {p['block_ms']} ms blocks, {r['alternatives']} partial "
        f"alternatives, partial words on, seed {r['seed']}. Measured by "
        "`scripts/partial_states.py`.",
        "",
        "Every reading's `relation` and `lead_delta` below are the runtime's own, read off the "
        "partial. The harness derives them again from its series as a check: over the streams "
        f"decoded here {par['agreed']} readings carried a delta from both and all {par['agreed']} "
        f"agreed, the largest disagreement {par['max_diff']:.1e} nats against a tolerance of "
        f"{RUNTIME_TOL:.0e}; {par['runtime_only']} more carried one from the runtime alone, a "
        "reading entering the reported list whose history the harness never saw "
        "(`[[rr:TD-9#Readings are read once per decoding advance]]`).",
        "",
        "The reading with no word on it is `[sil]` on silence phones and `[speech]` inside a "
        "word's phones (`[[rr:TD-10#Decision outcome]]`); every figure below keys it as one "
        f"reading, and its history runs across the label. Over every partial read here it stood "
        f"at rank 0 as `[speech]` on {SPEECH_LABELS['rank0']} of {SPEECH_LABELS['advances']} "
        f"blocks, and a `[speech]` entry closed the word list on {SPEECH_LABELS['entries']}.",
        "",
        "A Speech Commands clip holds one word, so it holds no transition between words and no "
        "finish. The streams here are built from the clips: utterances of one to five words, "
        f"pauses of {PAUSE_MS[0]}-{PAUSE_MS[1]} ms between the words of an utterance, finishes of "
        f"{FINISH_MS[0] / 1000:g}-{FINISH_MS[1] / 1000:g} s between utterances, and every gap cut in "
        "rotation from the dataset's own six background recordings, scaled to about "
        f"{GAP_FLOOR_DBFS:g} dBFS RMS rather than written as digital zeros, which the floor "
        "tracker treats apart (`[[rr:TD-8#The runtime reports the floor]]`). Every word's "
        "position and every gap's length are therefore exact, and each word's onset and offset "
        "are the first and last 10 ms frame within 20 dB of its clip's peak, which owes nothing "
        "to the decoder.",
        "",
        "| split | streams | minutes | utterances | words | pauses | finishes | blocks | advances | finals |",
        "|---|---|---|---|---|---|---|---|---|---|",
        f"| validation (thresholds fitted) | {vz['streams']} | {vz['minutes']:.1f} | {vz['utterances']} | "
        f"{vz['words']} | {vz['pauses']} | {vz['finishes']} | {vz['blocks']} | {vz['advances']} | {vz['finals']} |",
        f"| testing (every figure below) | {sz['streams']} | {sz['minutes']:.1f} | {sz['utterances']} | "
        f"{sz['words']} | {sz['pauses']} | {sz['finishes']} | {sz['blocks']} | {sz['advances']} | {sz['finals']} |",
        "",
        f"The readings change only when the decoder advances a chunk: over the testing streams "
        f"{adv['n']} intervals between advances, median / p90 {adv['p50']:.0f} / {adv['p90']:.0f} ms, "
        f"mean {adv['mean']:.1f} ms, which is the 240 ms "
        "`[[rr:TD-9#Readings are read once per decoding advance]]` states at this block size. "
        "Between advances every partial repeats, so a rule reading the partial gives the same "
        "call; the calls below are therefore held flat between advances and scored at every "
        "block.",
        "",
        "## The rules",
        "",
        "R0 is TD-8's reading of the fields, and nothing else: a word arriving at rank 0 is "
        "speech beginning (a detection, no lead), the trailing `[sil]` entry's span past a bound "
        "is speech ending, a span that goes on growing is silence being kept, and a word at rank "
        "0 with no trailing span is speech continuing. R1 is R0 plus the readings' motion: at an "
        "utterance's start the empty reading `[sil]` losing its lead means a word is coming, and "
        "mid-utterance a reading that extends rank 0 by a word and is gaining on the field means "
        "the next word is forming.",
        "",
        f"Fitted on the validation streams, at the {HORIZONS[0]} ms horizon, by the macro F1 over "
        f"the four states: first the bound on the trailing span ({chosen['bound_ms']} ms chosen), "
        f"then the two motion thresholds at that bound (`[sil]` lead_delta at or below "
        f"{chosen['sil_delta']:+.2f} nats, an extending reading's lead_delta at or above "
        f"{chosen['ext_delta']:+.2f} nats). TD-8's own bound, 300 ms, is reported beside them "
        "because the record quotes its private figures there.",
        "",
        "| bound (ms) | macro F1 | F1 of *toward* | F1 of *away* | false *away* calls/min in finishes |",
        "|---|---|---|---|---|",
    ]
    for k, v in r["fit"]["bound_scores"].items():
        L.append(
            f"| {k} | {v['macro_f1']:.3f} | {v['f1_toward']:.3f} | {v['f1_away']:.3f} | "
            f"{v['finish_alarms_per_min']:.1f} |"
        )
    L += [
        "",
        "The motion thresholds, over the same validation streams at the chosen bound. A pair "
        "that calls *away* everywhere buys recall on that state and loses the other three, which "
        "is why the choice is the macro F1 and not the *away* F1; the last column is what each "
        "pair costs inside the finish gaps, where the truth is never *away*.",
        "",
        "| `[sil]` lead_delta at or below | extending lead_delta at or above | macro F1 | F1 of *away* | F1 of *toward* | false *away* calls/min in finishes |",
        "|---|---|---|---|---|---|",
    ]
    for key, v in r["fit"]["motion_scores"].items():
        d, e = key.split(",")
        L.append(
            f"| {float(d):+.2f} | {float(e):+.2f} | {v['macro_f1']:.3f} | {v['f1_away']:.3f} | "
            f"{v['f1_toward']:.3f} | {v['finish_alarms_per_min']:.1f} |"
        )
    L += [
        "",
        "## A. Silence direction",
        "",
        "At every block the ground truth says what the next W ms do: *toward* silence (speech "
        "now, silence inside the window), *away* from silence (silence now, speech inside the "
        "window), *maintaining silence*, *maintaining speech*. The predictor reads only the "
        "partial at that block.",
        "",
        "The *away* rows below are exploratory in the same way the onset section is: they are "
        "measured on spliced words, and a rule that calls *away* inside a built gap may be reading "
        "the gap. They are scored here against R0 and read as a comparison, never as evidence "
        "that speech can be foreseen.",
        "",
        "Read the *away* row first. R0's *away* is a detection with no lead by construction: the "
        "word is already at rank 0 when it calls, and by then the truth has usually moved on to "
        "*maintaining speech*, so a low precision there is the shape of the problem rather than "
        "a fault in the rule. The question this page asks is whether the motion moves that row.",
        "",
    ]
    for w_ms in HORIZONS:
        d = r["A1"][str(w_ms)]
        L += [
            f"### W = {w_ms} ms",
            "",
            "| state | blocks (truth) | R0 precision | R0 recall | R1 precision | R1 recall |",
            "|---|---|---|---|---|---|",
        ]
        for s in STATES:
            c0, c1 = d["R0"][s], d["R1"][s]
            L.append(
                f"| {s} | {c0['tp'] + c0['fn']} | {num(100 * c0['precision'])}% | {num(100 * c0['recall'])}% | "
                f"{num(100 * c1['precision'])}% | {num(100 * c1['recall'])}% |"
            )
        L.append("")
        for state in ("away", "toward"):
            b = r["A1_bootstrap"].get(f"{state}@{w_ms}")
            if b:
                L.append(
                    f"F1 of *{state}*: R0 {d['R0'][state]['f1']:.3f}, R1 {d['R1'][state]['f1']:.3f}; paired "
                    f"bootstrap over utterances ({BOOTSTRAPS} resamples) of R1 - R0, "
                    f"{b['diff']:+.3f} with a 95% interval [{b['lo']:+.3f}, {b['hi']:+.3f}]."
                )
        L.append("")
    L += [
        "### Lead time per true transition",
        "",
        "Every word's onset is a true move away from silence and every word's offset a move "
        "toward it. Per transition, the milliseconds from the first block the rule calls that "
        "state, inside the window from the previous transition to the next, to the transition "
        f"itself; negative is before it. Handled means the call lands within {HORIZONS[0]} ms of "
        "the truth, which is the paired unit for McNemar.",
        "",
        "| rule | transition | transitions | called | called before | p10 / p50 / p90 ms |",
        "|---|---|---|---|---|---|",
    ]
    for rule in ("R0", "R1"):
        for kind in ("away", "toward"):
            d = r["A2"][rule][kind]
            L.append(
                f"| {rule} | {kind} | {d['n']} | {share(d['called'], d['n'])} | "
                f"{share(d['before'], d['called'])} | {num(d['p10'])} / {num(d['p50'])} / {num(d['p90'])} |"
            )
    L.append("")
    L += [
        "A call before the transition is not the same thing as foresight: a rule that calls "
        "*away* on a phantom word inside the gap has called it before the word, and the first "
        "call is what this table takes. The false alarm rates below are the other half of the "
        "figure, and the two should be read together.",
        "",
    ]
    for kind in ("away", "toward"):
        m = r["A2_mcnemar"][kind]
        L.append(
            f"McNemar on *{kind}* handled within the horizon: both {m['both']}, R0 only "
            f"{m['only_a']}, R1 only {m['only_b']}, neither {m['neither']}, exact two-sided p = "
            f"{pval(m['p'])}."
        )
    n = r["A3"]["noise"]
    L += [
        "",
        "### False alarms",
        "",
        "*away* where the truth cannot be *away*: inside a finish gap once the word has ended, "
        "and anywhere on the six background recordings fed continuously through one recognizer. "
        "Counted as blocks and as distinct runs of the call.",
        "",
        "| audio | minutes | rule | away blocks/min | away calls/min |",
        "|---|---|---|---|---|",
    ]
    for rule in ("R0", "R1"):
        d = r["A3"]["finish"][rule]
        L.append(f"| finish gaps | {d['minutes']:.1f} | {rule} | {d['per_min_blocks']:.1f} | {d['per_min']:.2f} |")
    for rule in ("R0", "R1"):
        d = n[rule]
        L.append(f"| recordings alone | {d['minutes']:.1f} | {rule} | {d['per_min_blocks']:.1f} | {d['per_min']:.2f} |")
    L += [
        "",
        f"The recordings alone also produced {n['finals']} finals over {n['minutes']:.1f} minutes, "
        f"{n['finals_with_a_word']} of them carrying a word, which is the figure the benchmark's "
        "noise pass reports; the rest are the 5 s silence rule closing a wordless stretch.",
        "",
        "### The end-of-speech confusion",
        "",
        "TD-8's core problem in public form. Among the silent stretches, at each millisecond of "
        "trailing `[sil]` span: the share of inter-word pauses whose span reaches it before the "
        "next word begins (a pause taken for a finish) against the share of finishes whose span "
        "reaches it at all. The motion columns refuse the call at an advance where a reading "
        f"extending rank 0 is gaining by at least {chosen['ext_delta']:+g} nats. The last column "
        "is how far ahead of the decoder's own endpoint final the call arrived.",
        "",
        f"| trailing span (ms) | pauses mistaken ({r['A4']['totals'].get('pause', 0)}) | with motion | "
        f"finishes called ({r['A4']['totals'].get('finish', 0)}) | with motion | called before the endpoint | median ms early |",
        "|---|---|---|---|---|---|---|",
    ]
    tot = r["A4"]["totals"]
    for x in EOS_GRID:
        c = r["A4"]["curve"][str(x)]
        L.append(
            f"| {x} | {share(c['pauses_mistaken'], tot.get('pause', 0))} | "
            f"{share(c['pauses_mistaken_motion'], tot.get('pause', 0))} | "
            f"{share(c['finishes_called'], tot.get('finish', 0))} | "
            f"{share(c['finishes_called_motion'], tot.get('finish', 0))} | "
            f"{share(c['early_n'], tot.get('finish', 0))} | {num(c['early_ms_p50'])} |"
        )
    L += [
        "",
        f"That share of pauses is a fact about this stream's pauses, which are uniform over "
        f"{PAUSE_MS[0]}-{PAUSE_MS[1]} ms by construction, so it is given again by the length of "
        f"the pause that was built, at TD-8's 300 ms bound and at the fitted "
        f"{r['A4_by_pause']['bound_ms']} ms one:",
        "",
        "| pause built (ms) | pauses | mistaken at 300 ms | with motion | mistaken at "
        f"{r['A4_by_pause']['bound_ms']} ms | with motion |",
        "|---|---|---|---|---|---|",
    ]
    for key, d in r["A4_by_pause"]["td8_bound"].items():
        c = r["A4_by_pause"]["chosen"][key]
        L.append(
            f"| {key} | {d['n']} | {share(d['mistaken'], d['n'])} | {share(d['mistaken_motion'], d['n'])} | "
            f"{share(c['mistaken'], c['n'])} | {share(c['mistaken_motion'], c['n'])} |"
        )
    L += runtime_finals_section(r, a)
    b1 = r["B1"]
    b2 = r["B2"]
    b3 = r["B3"]
    b4 = r["B4"]
    b5 = r["B5"]
    L += [
        "",
        "## B. Transitioning between words",
        "",
        "Per ground-truth transition into a word, inside a window reaching "
        f"{a.lookback_ms} ms back from its onset and 1500 ms forward: when the rank-0 word list "
        "grew, which is the transition a host sees today; when the word itself led; when a "
        "reading extending rank 0 first appeared and what its extra word was. A transition into "
        "the first word of an utterance is a different thing from one mid-utterance: when rank 0 "
        "is the empty reading every word-carrying reading extends it by definition "
        "(`[[rr:TD-9#Every reading names its relation to the partial]]`), so the two are reported "
        "apart.",
        "",
        "### Preview lead",
        "",
        "| position | transitions | rank 0 grew | the word that arrived was right | an extending "
        "reading first | zero lead | lead ms p50 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in ("first in utterance", "mid utterance"):
        d = b2[name]
        L.append(
            f"| {name} | {d['n']} | {share(d['arrived'], d['n'])} | {share(d['arrive_correct'], d['arrived'])} | "
            f"{share(d['with_extender'], d['arrived'])} | {share(d['zero_lead'], d['with_extender'])} | "
            f"{num(d['ms_p50'])} |"
        )
    L += [
        "",
        "The *an extending reading first* column is 100% by definition and not by measurement: "
        "wherever rank 0 is the empty reading, which is most of a gap, every word-carrying "
        "reading extends it. The column that carries information is the next one, and the "
        "right-word figure below it.",
        "",
        f"Over all {b1['transitions']} transitions rank 0 grew on {b1['arrived']} and the word "
        f"itself led on {b1['led']}. An extending reading was there before the growth on "
        f"{share(b1['with_extender'], b1['arrived'])}, and that lead was zero advances on "
        f"{share(b1['zero_lead'], b1['with_extender'])}; where it was not zero it ran "
        f"{num(b1['advances_p50'])} / {num(b1['advances_p90'])} advances (median / p90), "
        f"{num(b1['ms_p50'])} / {num(b1['ms_p90'])} ms. A reading extending rank 0 *by the word "
        f"that was coming* appeared first on {share(b1['correct_lead_n'], b1['arrived'])}, by a "
        f"median {num(b1['correct_lead_ms_p50'])} ms and a p90 "
        f"{num(b1['correct_lead_ms_p90'])} ms.",
        "",
        "### Preview accuracy by the lead still to run",
        "",
        "This is a measured result of its own, and separate from the question of whether a word "
        "can be foreseen: given that a reading extending rank 0 is present, how often its extra "
        "word is the word that arrives. The *all* group is candidate presence alone, the *gaining* "
        "group is presence plus the motion, and the two columns beside each other are the "
        "baseline and the field.",
        "",
        f"Every advance in those windows that carried an extending reading, {r['B2_advances']} of "
        "them, scored by whether the top extending reading's extra word is the word that was "
        "coming (top-1) and whether any of the top three extending readings has it (top-3), "
        "against the milliseconds still to run before rank 0 grew. The second group keeps only "
        "the advances where rank 0 already held a word, which is a continuation being previewed "
        "rather than a word being guessed at from silence; the third keeps the advances carrying "
        "a gaining one-word extender and ranks only those, which is the reading the rule uses.",
        "",
        "| lead still to run (ms) | advances | top-1 | top-3 | over a word | top-1 | top-3 | gaining | top-1 | top-3 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k, d in r["B2_by_lead"].items():
        allr, over, gain = d["all"], d["over a word"], d["gaining"]
        L.append(
            f"| {k} | {allr['n']} | {share(allr['top1'], allr['n'])} | {share(allr['top3'], allr['n'])} | "
            f"{over['n']} | {share(over['top1'], over['n'])} | {share(over['top3'], over['n'])} | "
            f"{gain['n']} | {share(gain['top1'], gain['n'])} | {share(gain['top3'], gain['n'])} |"
        )
    L += [
        "",
        "### False transitions, and what the `[sil]` entries measure",
        "",
        "Readings that extend rank 0 by one word where no word follows, and never reach rank 0 "
        "themselves, per minute. Split by whether rank 0 held a word: over a word this is the "
        "beam offering a continuation that never comes, over the empty reading it is a word "
        "being offered on silence, which is the rival "
        "`[[rr:TD-8#A word on a quiet block was spoken and is reported]]` describes.",
        "",
        "| audio | minutes | per minute over a word | per minute over silence |",
        "|---|---|---|---|",
        f"| finish gaps | {b3['finish']['minutes']:.1f} | {b3['finish']['per_min_over_words']:.1f} | "
        f"{b3['finish']['per_min_over_silence']:.1f} |",
        f"| recordings alone | {b3['noise']['minutes']:.1f} | {b3['noise']['per_min_over_words']:.1f} | "
        f"{b3['noise']['per_min_over_silence']:.1f} |",
        "",
        f"The inter-word `[sil]` entry against the pause that was built, read at the advance the "
        f"next word first leads: {b4['inner_n']} pauses, mean error {num(b4['inner_mean_err'])} ms, "
        f"median {num(b4['inner_p50_err'])} ms, p90 absolute error "
        f"{num(b4['inner_p90_abs_err'])} ms. It is readable only where the decoder did not close "
        "the utterance inside the pause, since a final starts the entries again.",
        "",
        "The trailing `[sil]` span per advance inside a gap, against two clocks: the silence "
        "elapsed since the word's energy offset, and the silence since the decoder's own end of "
        "that word. The second is the span's own lag, the first adds the difference between the "
        "two alignments.",
        "",
        "| gap | advances | mean error ms | median error ms | p90 absolute error ms | against the decoder's own end: mean | p10 / p50 / p90 |",
        "|---|---|---|---|---|---|---|",
    ]
    for kind, d in sorted(b4["trailing"].items()):
        L.append(
            f"| {kind} | {d['n']} | {num(d['mean_err'])} | {num(d['p50_err'])} | {num(d['p90_abs_err'])} | "
            f"{num(d['decoded_mean_err'])} | {num(d['decoded_p10_err'])} / {num(d['decoded_p50_err'])} / "
            f"{num(d['decoded_p90_err'])} |"
        )
    L += [
        "",
        "### Transition latency on continuous audio",
        "",
        f"From a word's energy onset to the block the word itself leads the partial: median / p90 "
        f"{num(b5['rank0_p50'])} / {num(b5['rank0_p90'])} ms over the {b5['n']} of "
        f"{b5['transitions']} transitions where it led inside the window, and to the block its "
        f"`stable_ms` clears {STABLE_BOUND_MS} ms, {num(b5['stable_p50'])} / "
        f"{num(b5['stable_p90'])} ms over {b5['stable_n']}. Mid-utterance alone the first figure "
        f"is {num(b5['mid_rank0_p50'])} ms. The decoder's own endpoint closed the utterance "
        f"inside the pause on {share(b5['finals_in_pause'], b5['mid_n'])} of the mid-utterance "
        "transitions: at this model's rules (`[[rr:TD-2#Inputs: configuration]]`) a pause of half "
        "a second is already an endpoint, so most of this page's mid-utterance transitions are "
        "the decoder starting again rather than continuing.",
        "",
        "### Before and after, on the transitions",
        "",
        "R0 is the host that sees a transition when rank 0 grows; R1 is the host that reads the "
        "extending reading's extra word where one appeared first, and R0 where none did. Paired "
        "on the transitions where rank 0 grew inside the window.",
        "",
    ]
    m = r["B_mcnemar"]
    mg = r["B_mcnemar_gaining"]
    cm = r["B_call_ms"]
    bb = r["B_bootstrap_call_ms"]
    bg = r["B_bootstrap_call_ms_gaining"]
    bgn = r["B_gaining"]
    L += [
        f"R1 comes in two forms: the first extending reading of any kind, and the first that is "
        f"*gaining* by at least {chosen['ext_delta']:+.2f} nats, which is what reading the motion "
        f"means. The gaining form was available on {share(bgn['with_gaining'], cm['n'])} of the "
        f"transitions, with a lead of {num(bgn['ms_p50'])} / {num(bgn['ms_p90'])} ms (median / "
        "p90).",
        "",
        "| comparison | both right | R0 only | R1 only | neither | exact p |",
        "|---|---|---|---|---|---|",
        f"| R0 against R1, any extender | {m['both']} | {m['only_a']} | {m['only_b']} | "
        f"{m['neither']} | {pval(m['p'])} |",
        f"| R0 against R1, gaining extender | {mg['both']} | {mg['only_a']} | {mg['only_b']} | "
        f"{mg['neither']} | {pval(mg['p'])} |",
        "",
        f"Milliseconds from the onset to the call: R0 {num(cm['R0_p50'])} / {num(cm['R0_p90'])} "
        f"(median / p90), R1 any {num(cm['R1_p50'])} / {num(cm['R1_p90'])}, R1 gaining "
        f"{num(cm['Rg_p50'])} / {num(cm['Rg_p90'])}. Paired bootstrap over transitions "
        f"({BOOTSTRAPS} resamples) of the mean difference in call time: any extender "
        f"{bb['diff']:+.1f} ms, 95% interval [{bb['lo']:+.1f}, {bb['hi']:+.1f}]; gaining "
        f"{bg['diff']:+.1f} ms, [{bg['lo']:+.1f}, {bg['hi']:+.1f}]. Earlier is better only if the "
        "word called is right, which the table above answers.",
        "",
    ]
    L += momentum_section(r, chosen)
    if "vosk" in r:
        v = r["vosk"]
        pw = v.get("partial_words") or {}
        L += [
            "### The stock wheel on the same streams",
            "",
            f"The stock wheel has partial text and no readings to move. Over the same testing "
            f"streams its partial grew a word inside the window on "
            f"{share(v['arrived'], v['transitions'])} transitions, and that word was the right one "
            f"on {share(v['correct'], v['arrived'])}, at {num(v['rank0_p50'])} / "
            f"{num(v['rank0_p90'])} ms from the onset (median / p90). The word itself led on "
            f"{share(v['led'], v['transitions'])} at {num(v['label_p50'])} / {num(v['label_p90'])} "
            f"ms. It returned {v['finals']} finals where utter returned "
            f"{r['sizes']['testing']['finals']}. That is the detection baseline: the moment R0 "
            "reads, with nothing beside it.",
            "",
        ]
        if pw:
            L += [
                f"Its partial word times could not be used at all. With partial words on, "
                f"{pw['vosk']['on']['with_text']} of {pw['vosk']['on']['blocks']} blocks of the "
                f"first testing stream carried partial text, against "
                f"{pw['vosk']['off']['with_text']} with partial words off: the wheel drops the "
                "word in progress from a partial when word times are requested. utter's partial "
                f"carried text on {pw['utter']['on']['with_text']} blocks with them on and "
                f"{pw['utter']['off']['with_text']} with them off. The baseline above is "
                "therefore the wheel with partial words off.",
                "",
            ]
    L += [
        "## Caveat",
        "",
        "The words are isolated read commands joined with built gaps. They carry no "
        "coarticulation across the join and no sentence prosody, and every pause is a splice "
        "rather than a speaker drawing breath, so the transitions here are cleaner than "
        "dictation and the pauses are more uniform. TD-8's figures come from the first "
        "consumer's private gate corpus; this page is the reproducible counterpart, not a "
        "replacement. The gaps are the dataset's own recordings scaled to one floor, so the "
        "floor does not move within a stream as it does between rooms.",
        "",
        "Every figure above is in `partial-states.json`, and `partial-states.streams.jsonl` "
        "carries, per stream, the ground truth, the finals and the whole advance series with "
        "each reading's confidence, lead delta and relation, so another rule can be scored on the "
        "same streams without decoding them again.",
    ]
    return L


if __name__ == "__main__":
    main()
