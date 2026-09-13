#!/usr/bin/env python3
"""Words on wordless audio, over the first consumer's replay corpus.

    python scripts/wordless_census.py --corpus DIR --model MODEL_DIR --grammar G.json --out PAGE.md

The private half of the public page's wordless pass: the same census, on the recordings a host
actually failed on. Both engines are fed identical blocks under identical grammars, and a block
is wordless by the audio alone, never by either engine's reading, so the two are scored on one
set of blocks. A wordless block is one whose last 400 ms sit within `--margin` of the floor at
that moment, the floor being the 5th percentile of RMS over 100 ms windows at a 50 ms hop across
the last 10 s fed, which is the measure the runtime reports and this script reproduces host-side
so the stock wheel can be judged by it too.

On those blocks a word at rank 0 is split as the private sweep splits it: **held** when no word's
span reaches the block's last 400 ms, **straddling** when one does and its own span carries
speech, and **silence** when the last word's own span does not, which is the bucket the owner's
complaint falls in. Where a silence word appears it is reported against the properties of the
take it appeared in, so a cause can be named.

Paths to the corpus, the model and the grammar are arguments; nothing here names a consumer, and
the page this writes carries counts and rates only.
"""

import argparse
import array
import json
import math
import sys
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import speech_commands as sc  # noqa: E402

RATE = 16000
TAIL_MS = 400
FLOOR_WINDOW_MS = 100
FLOOR_HOP_MS = 50
FLOOR_HISTORY_S = 10
FLOOR_Q = 0.05
SINCE_SPEECH_EDGES = (0.5, 1.0, 2.0, 5.0)


def note(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def words_of(text):
    return [w for w in text.split() if not (w.startswith("[") and w.endswith("]"))]


def read_pcm(path):
    with wave.open(str(path)) as w:
        a = array.array("h")
        a.frombytes(w.readframes(w.getnframes()))
    return a


def dbfs(samples):
    if not samples:
        return -999.0
    acc = 0
    for v in samples:
        acc += v * v
    rms = math.sqrt(acc / len(samples))
    return 20 * math.log10(rms / 32768.0) if rms > 0 else -999.0


def quantile(v, q):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))]


def floor_series(pcm, block):
    """The floor as it stands at the end of every block, by the runtime's own definition.
    [[rr:TD-8#The runtime reports the floor]]"""
    n = RATE * FLOOR_WINDOW_MS // 1000
    hop = RATE * FLOOR_HOP_MS // 1000
    keep = RATE * FLOOR_HISTORY_S // hop
    levels = []
    out = []
    windows = 0
    for end in range(block, len(pcm) + block, block):
        while (windows + 1) * hop + n <= min(end, len(pcm)):
            i = windows * hop
            levels.append(dbfs(pcm[i : i + n]))
            windows += 1
        out.append(quantile(levels[-keep:], FLOOR_Q) if levels else None)
    return out


def wordless_blocks(pcm, block, margin):
    """One label per block from the audio alone, so both engines are scored on the same blocks."""
    floors = floor_series(pcm, block)
    tail = RATE * TAIL_MS // 1000
    out = []
    for i, floor in enumerate(floors):
        end = min((i + 1) * block, len(pcm))
        level = dbfs(pcm[max(0, end - tail) : end])
        out.append((floor is not None and level <= floor + margin, floor, level))
    return out


def span_energy(pcm, start_s, end_s):
    a = int(start_s * RATE)
    b = int(end_s * RATE)
    return dbfs(pcm[a:b]) if b > a else -999.0


def engines(names, model_dir, grammar):
    out = {}
    for name in names:
        mod = __import__(name)
        if name == "vosk":
            mod.SetLogLevel(-1)
        out[name] = (mod, mod.Model(str(model_dir)), json.dumps(grammar))
    return out


def recognizer(entry, alternatives, partial_words):
    mod, model, grammar = entry
    rec = mod.KaldiRecognizer(model, RATE, grammar)
    rec.SetWords(True)
    if partial_words and hasattr(rec, "SetPartialWords"):
        rec.SetPartialWords(True)
    if alternatives and hasattr(rec, "SetPartialAlternatives"):
        rec.SetPartialAlternatives(alternatives)
    return rec


def classify(entries, pcm, end, margin, floor):
    """A rank-0 word on a wordless block, split as the private sweep splits it. The word's own
    energy is measured here from the WAV, so the stock wheel is classified by the same rule.
    [[rr:TD-8#Measurements count a word whose own span carries no speech]]"""
    said = [e for e in entries if not e["word"].startswith("[")]
    if not said:
        return None
    tail = RATE * TAIL_MS // 1000
    last = said[-1]
    if int(last["end"] * RATE) <= end - tail:
        return "held"
    if span_energy(pcm, last["start"], last["end"]) > floor + margin:
        return "straddling"
    return "silence"


def run_take(entry, pcm, labels, block_ms, margin, alternatives, partial_words=False):
    """One take through one engine, counted only on the blocks the audio called wordless. Asking
    for the words under a partial can change which partial an engine shows: the stock wheel's
    partial-word path is a lattice one that trails the audio, so the census is read off the plain
    partial for both engines and the split below off a second pass, which is trusted only where
    the two passes agree on the count."""
    block = RATE * block_ms // 1000
    counts = Counter()
    events = []
    since_speech = None
    spoken_before = 0
    run = longest = 0
    rec = recognizer(entry, alternatives, partial_words)
    raw = pcm.tobytes()
    for i, (quiet, floor, _level) in enumerate(labels):
        end = min((i + 1) * block, len(pcm))
        chunk = raw[2 * (i * block) : 2 * end]
        if not chunk:
            break
        if not quiet:
            since_speech = end
        if rec.AcceptWaveform(chunk):
            f = json.loads(rec.Result())
            spoken_before += len(words_of(f.get("text", "")))
            counts["finals"] += 1
            for e in f.get("result") or []:
                if e["word"].startswith("["):
                    continue
                counts["final_words"] += 1
                if span_energy(pcm, e["start"], e["end"]) <= (floor if floor is not None else -999.0) + margin:
                    counts["silence_final_words"] += 1
            run = 0
            continue
        if not quiet:
            run = 0
            continue
        p = json.loads(rec.PartialResult())
        counts["quiet_blocks"] += 1
        if not words_of(p.get("partial", "")):
            run = 0
            continue
        counts["rank0_word"] += 1
        kind = classify(p.get("partial_result") or [], pcm, end, margin, floor)
        counts[kind or "held"] += 1
        alts = p.get("partial_alternatives")
        if alts is not None:
            counts["rival_word"] += any(words_of(a["text"]) for a in alts[1:])
        if kind == "silence":
            run += 1
            longest = max(longest, run)
            events.append(
                dict(
                    floor=floor,
                    after_a_word=spoken_before > 0,
                    since_speech_s=(end - since_speech) / RATE if since_speech is not None else None,
                )
            )
        else:
            run = 0
    f = json.loads(rec.FinalResult())
    counts["finals"] += 1
    for e in f.get("result") or []:
        if e["word"].startswith("["):
            continue
        counts["final_words"] += 1
        last_floor = next((fl for _, fl, _ in reversed(labels) if fl is not None), -999.0)
        if span_energy(pcm, e["start"], e["end"]) <= last_floor + margin:
            counts["silence_final_words"] += 1
    counts["longest_silence_run"] = longest
    return counts, events


def bucket_since(v):
    if v is None:
        return "no speech yet"
    for e in SINCE_SPEECH_EDGES:
        if v < e:
            return f"under {e:g} s"
    return f"{SINCE_SPEECH_EDGES[-1]:g} s or more"


def rate(n, minutes):
    return n / minutes if minutes else float("nan")


def census(names, corpus, model_dir, grammar, block_ms, margin, alternatives, fractions):
    takes = sorted(Path(corpus).glob("*.wav"))
    note(f"{len(takes)} takes, {len(names)} engines, {len(fractions)} grammar fractions")
    audio = {}
    labels = {}
    minutes = 0.0
    for p in takes:
        pcm = read_pcm(p)
        audio[p] = pcm
        labels[p] = wordless_blocks(pcm, RATE * block_ms // 1000, margin)
        minutes += len(pcm) / RATE / 60.0
    quiet_minutes = sum(sum(1 for q, _, _ in labels[p] if q) for p in takes) * block_ms / 1000.0 / 60.0
    out = dict(
        takes=len(takes),
        minutes=minutes,
        quiet_minutes=quiet_minutes,
        block_ms=block_ms,
        margin_db=margin,
        by_fraction={},
        by_property={},
        paired={},
        split_trusted={},
    )
    for frac in fractions:
        g = grammar[: max(1, round(len(grammar) * frac))]
        label = "all of it" if frac >= 1.0 else f"{frac:g} of it"
        eng = engines(names, model_dir, g)
        for name in names:
            note(f"  grammar {label}, {name}")
            plain = Counter()
            split = Counter()
            events = []
            per_take = {}
            for p in takes:
                c, _ = run_take(eng[name], audio[p], labels[p], block_ms, margin, alternatives)
                plain.update(c)
                d, ev = run_take(eng[name], audio[p], labels[p], block_ms, margin, alternatives, True)
                longest = d.pop("longest_silence_run")
                split.update(d)
                split["longest_silence_run"] = max(split["longest_silence_run"], longest)
                events += ev
                per_take[p.name] = c["rank0_word"]
            # The split describes the partial its own pass showed; where that pass counted a
            # different number of rank-0 words, it is not a split of the census above.
            trusted = split["rank0_word"] == plain["rank0_word"]
            out["split_trusted"].setdefault(label, {})[name] = trusted
            out["by_fraction"].setdefault(label, {})[name] = dict(plain, **{f"split_{k}": v for k, v in split.items()})
            out["paired"].setdefault(label, {})[name] = [per_take[p.name] > 0 for p in takes]
            if frac == 1.0 and trusted:
                by = defaultdict(Counter)
                for e in events:
                    by["floor"][f"{10 * round(e['floor'] / 10):g} dBFS" if e["floor"] is not None else "none"] += 1
                    by["a word came first"]["yes" if e["after_a_word"] else "no"] += 1
                    by["since speech"][bucket_since(e["since_speech_s"])] += 1
                out["by_property"][name] = {k: dict(v) for k, v in by.items()}
    return out


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in range(k + 1)]
    top = max(terms)
    log_tail = top + math.log(sum(math.exp(t - top) for t in terms)) - n * math.log(2.0)
    return min(1.0, 2.0 * math.exp(log_tail)) if log_tail > -700.0 else 0.0


def recorded_args(a):
    """The arguments a rerun needs, with the paths that name a consumer replaced by the labels the
    gates already use. A page here carries counts and rates and nothing that identifies a take."""
    out = dict(vars(a))
    out["corpus"] = f"the {a.corpus_label} replay set"
    out["model"] = Path(a.model).name
    out["grammar"] = "the first consumer's grammar, exported as a JSON array"
    return out


def page(r, names, corpus_label, model_name):
    q = r["quiet_minutes"]
    lines = [
        "# Words on wordless audio, on the consumer's replay corpus",
        "",
        "Ruling: `[[rr:TD-8#Measurements count a word whose own span carries no speech]]`. The "
        "private half of the public page's wordless pass, on the recordings the complaint came "
        "from.",
        "",
        "## What ran",
        "",
        f"- Model: `{model_name}`; the first consumer's grammar, and the leading fractions of it.",
        f"- {r['provenance_line']}",
        f"- Corpus: the {corpus_label} replay set, {r['takes']} takes, "
        f"{r['minutes']:.1f} minutes, of which {q:.1f} minutes are wordless blocks.",
        f"- {r['block_ms']} ms blocks, both engines fed identically, scored by `scripts/wordless_census.py`.",
        "- A block is wordless by the audio alone: its last 400 ms within "
        f"{r['margin_db']:.0f} dB of the floor at that moment, the floor computed host-side by the "
        "runtime's own definition so the stock wheel is judged by the same rule. Neither engine's "
        "reading enters the labelling, so both are scored on one set of blocks. The block counts "
        "still differ by a handful between the engines, which is the blocks each consumed as a "
        "final rather than a partial.",
        "",
        "## The census",
        "",
        "Rates are per wordless minute. **silence** is the bucket the complaint falls in: a word "
        "at rank 0 whose own span carries no speech, counted apart from a held word and a "
        "straddling one rather than as a phantom.",
        "",
        "| grammar | engine | wordless blocks | rank 0 a word | rank 0 /min | rival word /min | "
        "held | straddling | silence | silence /min | final words | silence final words | "
        "longest silence run |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for frac, per in r["by_fraction"].items():
        for name, c in per.items():
            rival = f"{rate(c.get('rival_word', 0), q):.1f}" if c.get("rival_word") is not None else "-"
            if r["split_trusted"][frac][name]:
                sil = c.get("split_silence", 0)
                split = f"{c.get('split_held', 0)} | {c.get('split_straddling', 0)} | {sil} | {rate(sil, q):.2f} | "
                longest = c.get("split_longest_silence_run", 0)
            else:
                split = "- | - | - | - | "
                longest = "-"
            lines.append(
                f"| {frac} | {name} | {c.get('quiet_blocks', 0)} | {c.get('rank0_word', 0)} | "
                f"{rate(c.get('rank0_word', 0), q):.2f} | {rival} | {split}"
                f"{c.get('final_words', 0)} | {c.get('silence_final_words', 0)} | {longest} |"
            )
    lines += [
        "",
        "A dash means the engine showed a different partial when asked for the words under it, so "
        "that pass is not a split of the count beside it: the stock wheel's partial-word path is a "
        "lattice one that trails the audio.",
    ]
    lines += ["", "Paired over the same takes, a take carrying any word at rank 0 on a wordless block:", ""]
    for frac, per in r["paired"].items():
        if len(names) == 2:
            a, b = per[names[0]], per[names[1]]
            bb = sum(1 for x, y in zip(a, b) if x and not y)
            cc = sum(1 for x, y in zip(a, b) if y and not x)
            lines.append(
                f"- the grammar, {frac}: {names[0]} only {bb}, {names[1]} only {cc}, exact "
                f"two-sided p = {mcnemar(bb, cc):.3g}."
            )
    lines += ["", "## What a silence word goes with", ""]
    if not any(r["by_property"].get(n) for n in names):
        lines.append("No silence word appeared at rank 0 under any grammar, so there is nothing to split.")
    else:
        lines.append("Every silence word at rank 0 under the full grammar, split by the take property it fell in.")
        for name in names:
            props = r["by_property"].get(name) or {}
            for prop, counts in props.items():
                lines += ["", f"| {name}: {prop} | silence words |", "|---|---|"]
                for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
                    lines.append(f"| {k} | {v} |")
    lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--grammar", required=True, help="JSON array of strings")
    ap.add_argument("--out", required=True, help="page written here; .json beside it")
    ap.add_argument("--corpus-label", required=True, help="the date label the gates already use")
    ap.add_argument("--engines", default="vosk,utterpy")
    ap.add_argument("--block-ms", type=int, default=40)
    ap.add_argument("--margin", type=float, default=8.0, help="dB above the floor that is still silence")
    ap.add_argument("--alternatives", type=int, default=4)
    ap.add_argument("--grammar-fractions", default="0.25,0.5,1.0")
    a = ap.parse_args()
    names = [n for n in a.engines.split(",") if n.strip()]
    grammar = json.loads(Path(a.grammar).read_text())
    fractions = [float(v) for v in a.grammar_fractions.split(",") if v.strip()]
    prov = sc.provenance(argparse.Namespace(model=a.model, block_ms=a.block_ms), names)
    r = census(names, a.corpus, a.model, grammar, a.block_ms, a.margin, a.alternatives, fractions)
    r["provenance"] = prov
    r["provenance_line"] = sc.provenance_line(prov)
    r["args"] = recorded_args(a)
    lines = page(r, names, a.corpus_label, Path(a.model).name)
    # The page is overwritten every run, so what a run means lives beside it and is appended.
    reading = Path(a.out).with_suffix(".reading.md")
    if reading.exists():
        lines += [reading.read_text().strip(), ""]
    sc.write_page(a.out, lines)
    Path(a.out).with_suffix(".json").write_text(json.dumps(r, indent=1) + "\n")
    note(f"wrote {a.out}")


if __name__ == "__main__":
    main()
