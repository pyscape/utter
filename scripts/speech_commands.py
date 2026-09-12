#!/usr/bin/env python3
"""Public benchmark on Google Speech Commands v2 (CC BY 4.0): closed-vocabulary accuracy and
word latency, utter (through utterpy) against the stock vosk wheel, same blocks, same grammar.

    python scripts/speech_commands.py --data DIR --model MODEL_DIR --out docs/benchmarks/speech-commands

DIR holds the extracted dataset (one directory per word, testing_list.txt, _background_noise_).
The grammar is the dataset's 35 words plus distractors that a command host is likely to carry:
the letters a to z, the NATO alphabet, and red, yellow, blue, black and white. Three runs:

- full grammar: every test clip must decode to its word;
- twelve-class: the ten command words plus the unknown-word symbol, where the other 25 words
  must decode to [unk] and background noise to nothing;
- background noise under the full grammar, with and without the unknown-word symbol: phantom
  words per minute, and whether the silence reading is still offered as a rival when rank 0
  is a word.

Latency is the fed-audio time at which the label first appears in a partial, minus the word's
end from the clip's own energy envelope (the last 10 ms frame within 20 dB of the clip's peak),
a reference that owes nothing to either engine. Also reported: how often the first word shown is
later changed.
"""

import argparse
import array
import json
import math
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path

RATE = 16000
COMMANDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
FILLERS = [
    "backward",
    "forward",
    "follow",
    "learn",
    "visual",
    "bed",
    "bird",
    "cat",
    "dog",
    "happy",
    "house",
    "marvin",
    "sheila",
    "tree",
    "wow",
]
LETTERS = list("abcdefghijklmnopqrstuvwxyz")
NATO = [
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "kilo",
    "lima",
    "mike",
    "november",
    "oscar",
    "papa",
    "quebec",
    "romeo",
    "sierra",
    "tango",
    "uniform",
    "victor",
    "whiskey",
    "yankee",
    "zulu",
] + ["x ray"]
COLOURS = ["red", "yellow", "blue", "black", "white"]
DATASET_WORDS = COMMANDS + DIGITS + FILLERS


def words_of(text):
    return [w for w in text.split() if not (w.startswith("[") and w.endswith("]"))]


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def quantile(v, q):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))]


def read_pcm(path):
    with wave.open(str(path)) as w:
        return w.readframes(w.getnframes())


def energy_end(pcm, frame_ms=10, drop_db=20.0):
    """End, in seconds, of the last frame within `drop_db` of the clip's loudest frame."""
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
    last = max(i for i, level in enumerate(levels) if level >= peak - drop_db)
    return (last + 1) * n / RATE


class Engine:
    def __init__(self, name, module, model_dir, grammar, unknown_cost=None):
        self.name = name
        self.mod = module
        self.model = module.Model(str(model_dir))
        self.grammar = json.dumps(grammar)
        self.unknown_cost = unknown_cost
        find = getattr(self.model, "FindWord", None) or getattr(self.model, "vosk_model_find_word", None)
        self.missing = [w for e in grammar for w in e.split() if find is not None and find(w) < 0]

    def new(self, alternatives=0):
        if self.unknown_cost is not None:
            rec = self.mod.KaldiRecognizer(self.model, RATE, self.grammar, self.unknown_cost)
        else:
            rec = self.mod.KaldiRecognizer(self.model, RATE, self.grammar)
        rec.SetWords(True)
        if alternatives and hasattr(rec, "SetPartialAlternatives"):
            rec.SetPartialAlternatives(alternatives)
        return rec


def feed(rec, pcm, block_ms, on_partial=None):
    """Feed a clip in blocks; returns the finals. `on_partial(fed_samples, partial_json)` per block."""
    block = RATE * block_ms // 1000 * 2
    fed = 0
    finals = []
    for i in range(0, len(pcm), block):
        chunk = pcm[i : i + block]
        fed += len(chunk) // 2
        if rec.AcceptWaveform(chunk):
            finals.append(json.loads(rec.Result()))
        elif on_partial is not None:
            on_partial(fed, json.loads(rec.PartialResult()))
    finals.append(json.loads(rec.FinalResult()))
    return finals


def final_words(finals):
    return [w for f in finals for w in words_of(f.get("text", ""))]


def run_clip(engine, pcm, block_ms, label):
    """Returns (final words, first-appearance second of the label or None, first word shown or None, wall seconds)."""
    state = {"first": None, "first_word": None}

    def on_partial(fed, p):
        partial = words_of(p.get("partial", ""))
        if partial and state["first_word"] is None:
            state["first_word"] = partial[0]
        if state["first"] is None and partial[:1] == [label]:
            state["first"] = fed / RATE

    t0 = time.perf_counter()
    finals = feed(engine.new(), pcm, block_ms, on_partial)
    wall = time.perf_counter() - t0
    return final_words(finals), state["first"], state["first_word"], wall


def full_grammar_pass(modules, model, clips, block_ms, grammar, lines, report):
    results = {}
    ref_end = {}
    for name, mod in modules.items():
        eng = Engine(name, mod, model, grammar)
        if eng.missing:
            lines.append(f"- {name}: words absent from the model's table: {eng.missing}")
        correct = never = shown = changed = 0
        conf = Counter()
        per_word = Counter()
        per_word_ok = Counter()
        lat = []
        wall = audio = 0.0
        for label, path in clips:
            pcm = read_pcm(path)
            audio += len(pcm) / 2 / RATE
            if path not in ref_end:
                ref_end[path] = energy_end(pcm)
            words, first, first_word, t = run_clip(eng, pcm, block_ms, label)
            wall += t
            per_word[label] += 1
            if words == [label]:
                correct += 1
                per_word_ok[label] += 1
            else:
                conf[(label, " ".join(words) or "(nothing)")] += 1
            if first_word is not None:
                shown += 1
                changed += words[:1] != [first_word]
            if first is None:
                never += 1
            elif ref_end[path] is not None:
                lat.append((first - ref_end[path]) * 1000.0)
        results[name] = dict(
            correct=correct,
            total=len(clips),
            confusions=conf.most_common(15),
            per_word={w: (per_word_ok[w], per_word[w]) for w in DATASET_WORDS},
            latency_ms_p50=quantile(lat, 0.5),
            latency_ms_p90=quantile(lat, 0.9),
            latency_n=len(lat),
            never_in_partial=never,
            first_shown=shown,
            first_changed=changed,
            rtf=wall / audio if audio else None,
        )
    lines.append("## Full grammar: accuracy")
    lines.append("")
    lines.append(
        "| engine | correct | accuracy | never in a partial | first appearance after the clip's energy end p50 / p90 | first word shown later changed | RTF |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['correct']} / {r['total']} | {pct(r['correct'], r['total']):.2f}% | {r['never_in_partial']} | "
            f"{r['latency_ms_p50']:.0f} / {r['latency_ms_p90']:.0f} ms over {r['latency_n']} | "
            f"{r['first_changed']} / {r['first_shown']} ({pct(r['first_changed'], r['first_shown']):.1f}%) | {r['rtf']:.4f} |"
        )
    lines.append("")
    lines.append("| word | " + " | ".join(results) + " |")
    lines.append("|---|" + "---|" * len(results))
    for w in DATASET_WORDS:
        lines.append(
            f"| {w} | " + " | ".join(f"{r['per_word'][w][0]} / {r['per_word'][w][1]}" for r in results.values()) + " |"
        )
    lines.append("")
    for name, r in results.items():
        lines.append(
            f"Most frequent confusions, {name}: " + "; ".join(f"{a} -> {b} ({n})" for (a, b), n in r["confusions"][:10])
        )
    lines.append("")
    report["full"] = results


def twelve_class_pass(modules, model, clips, noise, block_ms, lines, report):
    lines.append("## Twelve-class: ten commands plus the unknown-word symbol")
    lines.append("")
    lines.append(
        "| engine | commands correct | fillers and digits read as unknown | fillers and digits read as a command | noise seconds | noise seconds with a word |"
    )
    lines.append("|---|---|---|---|---|---|")
    twelve = {}
    for name, mod in modules.items():
        eng = Engine(name, mod, model, COMMANDS + ["[unk]"])
        cmd_total = cmd_ok = other_total = other_unk = other_cmd = 0
        for label, path in clips:
            finals = feed(eng.new(), read_pcm(path), block_ms)
            raw = [w for f in finals for w in f.get("text", "").split()]
            words = words_of(" ".join(raw))
            if label in COMMANDS:
                cmd_total += 1
                cmd_ok += words == [label]
            else:
                other_total += 1
                if not words and "[unk]" in raw:
                    other_unk += 1
                elif words:
                    other_cmd += 1
        noise_seconds = noise_words = 0
        for path in noise:
            pcm = read_pcm(path)
            sec = RATE * 2
            for i in range(0, len(pcm) - sec + 1, sec):
                finals = feed(eng.new(), pcm[i : i + sec], block_ms)
                noise_seconds += 1
                noise_words += bool(final_words(finals))
        twelve[name] = dict(
            cmd_ok=cmd_ok,
            cmd_total=cmd_total,
            other_unk=other_unk,
            other_cmd=other_cmd,
            other_total=other_total,
            noise_seconds=noise_seconds,
            noise_words=noise_words,
        )
        lines.append(
            f"| {name} | {cmd_ok} / {cmd_total} ({pct(cmd_ok, cmd_total):.2f}%) | {other_unk} / {other_total} ({pct(other_unk, other_total):.2f}%) | "
            f"{other_cmd} ({pct(other_cmd, other_total):.2f}%) | {noise_seconds} | {noise_words} |"
        )
    report["twelve"] = twelve


def noise_pass(modules, model, noise, block_ms, grammar, lines, report):
    lines.append("")
    lines.append("## Background noise under the full grammar")
    lines.append("")
    lines.append(
        "| engine | grammar | noise minutes | partial blocks | blocks with a word at rank 0 | phantom final words per minute | word blocks with the silence reading among the rivals |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    out = {}
    for name, mod in modules.items():
        for variant, g in [("full", grammar), ("full + [unk]", grammar + ["[unk]"])]:
            eng = Engine(name, mod, model, g)
            blocks = word_blocks = sil_rival = finals_words = seconds = 0
            for path in noise:
                pcm = read_pcm(path)
                sec = RATE * 2
                for i in range(0, len(pcm) - sec + 1, sec):
                    counts = {"blocks": 0, "words": 0, "rival": 0}

                    def on_partial(fed, p, counts=counts):
                        counts["blocks"] += 1
                        if words_of(p.get("partial", "")):
                            counts["words"] += 1
                            alts = p.get("partial_alternatives") or []
                            if any(a["text"] == "[sil]" for a in alts[1:]):
                                counts["rival"] += 1

                    finals = feed(eng.new(alternatives=4), pcm[i : i + sec], block_ms, on_partial)
                    blocks += counts["blocks"]
                    word_blocks += counts["words"]
                    sil_rival += counts["rival"]
                    finals_words += len(final_words(finals))
                    seconds += 1
            minutes = seconds / 60.0
            out[f"{name}/{variant}"] = dict(
                minutes=minutes, blocks=blocks, word_blocks=word_blocks, sil_rival=sil_rival, final_words=finals_words
            )
            rival = f"{sil_rival} / {word_blocks}" if name != "vosk" else "no alternatives"
            lines.append(
                f"| {name} | {variant} | {minutes:.1f} | {blocks} | {word_blocks} ({pct(word_blocks, blocks):.2f}%) | "
                f"{finals_words / minutes if minutes else float('nan'):.1f} | {rival} |"
            )
    report["noise"] = out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="output prefix; writes .md and .json")
    ap.add_argument("--split", default="testing", choices=["testing", "validation"])
    ap.add_argument("--limit", type=int, default=0, help="clips per word, 0 for all")
    ap.add_argument("--block-ms", type=int, default=40)
    ap.add_argument("--engines", default="vosk,utterpy")
    args = ap.parse_args()
    data = Path(args.data)
    listed = [line.strip() for line in (data / f"{args.split}_list.txt").read_text().splitlines() if line.strip()]
    by_word = defaultdict(list)
    for rel in listed:
        by_word[rel.split("/")[0]].append(rel)
    clips = []
    for w in DATASET_WORDS:
        items = by_word.get(w, [])
        if args.limit:
            items = items[: args.limit]
        clips.extend((w, data / rel) for rel in items)
    noise = sorted((data / "_background_noise_").glob("*.wav"))

    modules = {}
    for name in args.engines.split(","):
        modules[name] = __import__("vosk" if name == "vosk" else name)
        if name == "vosk":
            modules[name].SetLogLevel(-1)

    full_grammar = DATASET_WORDS + LETTERS + NATO + COLOURS
    report = {"split": args.split, "clips": len(clips), "block_ms": args.block_ms, "grammar_size": len(full_grammar)}
    lines = [f"# Speech Commands v2, {args.split} split, {len(clips)} clips, {args.block_ms} ms blocks", ""]
    lines.append(
        f"Grammar: the dataset's 35 words plus {len(LETTERS)} letters, {len(NATO)} NATO words and {len(COLOURS)} colours, {len(full_grammar)} entries."
    )
    lines.append("")
    full_grammar_pass(modules, args.model, clips, args.block_ms, full_grammar, lines, report)
    twelve_class_pass(modules, args.model, clips, noise, args.block_ms, lines, report)
    noise_pass(modules, args.model, noise, args.block_ms, full_grammar, lines, report)
    text = "\n".join(lines) + "\n"
    Path(args.out + ".md").write_text(text)
    Path(args.out + ".json").write_text(json.dumps(report, indent=1))
    print(text)


if __name__ == "__main__":
    main()
