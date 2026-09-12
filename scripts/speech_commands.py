#!/usr/bin/env python3
"""Public benchmark on Google Speech Commands v2 (CC BY 4.0): closed-vocabulary accuracy and
word latency, utter (through utterpy) against the stock vosk wheel, same blocks, same grammar.

    python scripts/speech_commands.py --data DIR --model MODEL_DIR --out docs/benchmarks/speech-commands

DIR holds the extracted dataset (one directory per word, testing_list.txt, _background_noise_).
The grammar is the dataset's 35 words plus distractors that a command host is likely to carry:
the letters a to z, the NATO alphabet, and red, yellow, blue, black and white. Two runs:

- full grammar: every test clip must decode to its word;
- twelve-class: the ten command words plus the unknown-word symbol, where the other 25 words
  must decode to [unk] and background noise to nothing.

Latency is the fed-audio time at which the label first appears in a partial, minus the word's
end from vosk's own alignment on that clip (SetWords), the reference both engines share.
"""

import argparse
import json
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


class Engine:
    def __init__(self, name, module, model_dir, grammar):
        self.name = name
        self.mod = module
        self.model = module.Model(str(model_dir))
        self.grammar = json.dumps(grammar)
        find = getattr(self.model, "FindWord", None) or getattr(self.model, "vosk_model_find_word", None)
        self.missing = [w for e in grammar for w in e.split() if find is not None and find(w) < 0]

    def new(self):
        rec = self.mod.KaldiRecognizer(self.model, RATE, self.grammar)
        rec.SetWords(True)
        return rec


def run_clip(engine, pcm, block_ms, label):
    """Returns (final words, first-appearance second of the label or None, vosk-style word end, wall seconds)."""
    rec = engine.new()
    block = RATE * block_ms // 1000 * 2
    fed = 0
    first = None
    finals = []
    t0 = time.perf_counter()
    for i in range(0, len(pcm), block):
        chunk = pcm[i : i + block]
        fed += len(chunk) // 2
        if rec.AcceptWaveform(chunk):
            finals.append(json.loads(rec.Result()))
        else:
            partial = words_of(json.loads(rec.PartialResult()).get("partial", ""))
            if first is None and label is not None and partial[: len(label)] == label:
                first = fed / RATE
    finals.append(json.loads(rec.FinalResult()))
    wall = time.perf_counter() - t0
    words = [w for f in finals for w in words_of(f.get("text", ""))]
    ends = [e["end"] for f in finals for e in f.get("result", []) if not e["word"].startswith("[")]
    return words, first, (ends[-1] if ends else None), wall


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def quantile(v, q):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="output prefix; writes .md and .json")
    ap.add_argument("--split", default="testing", choices=["testing", "validation"])
    ap.add_argument("--limit", type=int, default=0, help="clips per word, 0 for all")
    ap.add_argument("--block-ms", type=int, default=40)
    ap.add_argument("--engines", default="vosk,utterpy")
    ap.add_argument("--unknown-cost", type=float, default=0.0)
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
    twelve_grammar = COMMANDS + ["[unk]"]
    report = {
        "split": args.split,
        "clips": len(clips),
        "block_ms": args.block_ms,
        "grammar_size": len(full_grammar),
        "engines": {},
    }
    lines = [f"# Speech Commands v2, {args.split} split, {len(clips)} clips, {args.block_ms} ms blocks", ""]
    lines.append(
        f"Grammar: the dataset's 35 words plus {len(LETTERS)} letters, {len(NATO)} NATO words and {len(COLOURS)} colours, {len(full_grammar)} entries."
    )
    lines.append("")

    # pass 1: full grammar, vosk first so its word ends serve as the latency reference
    ref_end = {}
    results = {}
    for name in ["vosk"] + [n for n in modules if n != "vosk"]:
        if name not in modules:
            continue
        eng = Engine(name, modules[name], args.model, full_grammar)
        if eng.missing:
            lines.append(f"- {name}: words absent from the model's table: {eng.missing}")
        correct = 0
        conf = Counter()
        per_word = Counter()
        per_word_ok = Counter()
        lat = []
        never = 0
        wall = 0.0
        audio = 0.0
        for label, path in clips:
            with wave.open(str(path)) as w:
                pcm = w.readframes(w.getnframes())
            audio += len(pcm) / 2 / RATE
            words, first, end, t = run_clip(eng, pcm, args.block_ms, [label])
            wall += t
            per_word[label] += 1
            if words == [label]:
                correct += 1
                per_word_ok[label] += 1
            else:
                conf[(label, " ".join(words) or "(nothing)")] += 1
            if name == "vosk" and end is not None and words == [label]:
                ref_end[path] = end
            if first is None:
                never += 1
            elif path in ref_end:
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
            rtf=wall / audio if audio else None,
        )
    lines.append("## Full grammar: accuracy")
    lines.append("")
    lines.append(
        "| engine | correct | accuracy | never in a partial | first appearance after vosk's word end p50 / p90 | RTF |"
    )
    lines.append("|---|---|---|---|---|---|")
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['correct']} / {r['total']} | {pct(r['correct'], r['total']):.2f}% | {r['never_in_partial']} | {r['latency_ms_p50']:.0f} / {r['latency_ms_p90']:.0f} ms over {r['latency_n']} | {r['rtf']:.4f} |"
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
    report["engines"]["full"] = results

    # pass 2: twelve-class with the unknown-word symbol
    lines.append("## Twelve-class: ten commands plus the unknown-word symbol")
    lines.append("")
    lines.append(
        "| engine | commands correct | fillers and digits read as unknown | fillers and digits read as a command | noise seconds | noise seconds with a word |"
    )
    lines.append("|---|---|---|---|---|---|")
    twelve = {}
    for name, mod in modules.items():
        grammar = twelve_grammar
        eng = Engine(name, mod, args.model, grammar)
        if name != "vosk" and args.unknown_cost:
            eng.new = lambda eng=eng: (lambda r: (r.SetWords(True), r)[1])(
                eng.mod.KaldiRecognizer(eng.model, RATE, json.dumps(COMMANDS), args.unknown_cost)
            )
        cmd_total = cmd_ok = other_total = other_unk = other_cmd = 0
        for label, path in clips:
            with wave.open(str(path)) as w:
                pcm = w.readframes(w.getnframes())
            rec = eng.new()
            block = RATE * args.block_ms // 1000 * 2
            finals = []
            for i in range(0, len(pcm), block):
                if rec.AcceptWaveform(pcm[i : i + block]):
                    finals.append(json.loads(rec.Result()))
            finals.append(json.loads(rec.FinalResult()))
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
            with wave.open(str(path)) as w:
                pcm = w.readframes(w.getnframes())
            sec = RATE * 2
            for i in range(0, len(pcm) - sec + 1, sec):
                rec = eng.new()
                piece = pcm[i : i + sec]
                finals = []
                block = RATE * args.block_ms // 1000 * 2
                for j in range(0, len(piece), block):
                    if rec.AcceptWaveform(piece[j : j + block]):
                        finals.append(json.loads(rec.Result()))
                finals.append(json.loads(rec.FinalResult()))
                noise_seconds += 1
                noise_words += bool(words_of(" ".join(f.get("text", "") for f in finals)))
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
            f"| {name} | {cmd_ok} / {cmd_total} ({pct(cmd_ok, cmd_total):.2f}%) | {other_unk} / {other_total} ({pct(other_unk, other_total):.2f}%) | {other_cmd} ({pct(other_cmd, other_total):.2f}%) | {noise_seconds} | {noise_words} |"
        )
    report["engines"]["twelve"] = twelve
    text = "\n".join(lines) + "\n"
    Path(args.out + ".md").write_text(text)
    Path(args.out + ".json").write_text(json.dumps(report, indent=1))
    print(text)


if __name__ == "__main__":
    main()
