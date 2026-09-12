#!/usr/bin/env python3
"""Public benchmark on Google Speech Commands v2 (CC BY 4.0): closed-vocabulary accuracy and
word latency, utter (through utterpy) against the stock vosk wheel, same blocks, same grammar.

    python scripts/speech_commands.py --data DIR --model MODEL_DIR --out docs/benchmarks/speech-commands

DIR holds the extracted dataset (one directory per word, testing_list.txt, _background_noise_).
The grammar is the dataset's 35 words plus distractors that a command host is likely to carry:
the letters a to z, the NATO alphabet, and red, yellow, blue, black and white. The passes:

- full grammar: every test clip must decode to its word, with a paired exact McNemar test on
  the clips the two engines disagree about, overall and per word;
- agreement: the clips whose finals are identical, and the empty finals each engine returns
  split by direction, because reproducing the oracle and scoring like it are different claims;
- determinism: a sample decoded twice in this process and once in a fresh one, comparing the
  whole partial trace, because a hash seed drawn per process is invisible to a repeat inside it;
- endpoint latency: the clip followed by silence, to the block at which the engine closes a
  segment itself, which is the pause a speaker waits through and nothing else here measures;
- block size: 10 to 100 ms against accuracy, first-appearance latency and RTF, comparing the
  instant a word appears clip by clip against the finest block rather than only in aggregate;
- compute: recognizer construction split into the first and every later one, decode-only RTF,
  and per-block compute over the clips joined into one continuous stream;
- noise: the dataset's own background recordings mixed under the clips at each --snr-db, the
  same mixed samples fed to both engines;
- grammar size: the 35 dataset words plus filler up to each --grammar-sizes entry, which is
  the dial a host turns, against accuracy, construction and RTF;
- twelve-class: the ten command words plus the unknown-word symbol, where the other 25 words
  must decode to [unk] and background noise to nothing;
- background noise under the full grammar, with and without the unknown-word symbol: phantom
  words per minute, and whether the silence reading is still offered as a rival when rank 0
  is a word.

Latency is the fed-audio time at which the label first appears in a partial, minus the word's
end from the clip's own energy envelope (the last 10 ms frame within 20 dB of the clip's peak),
a reference that owes nothing to either engine. Also reported: how often the first word shown is
later changed.

Writes `<out>.md`, `<out>.json`, and `<out>.clips.jsonl`, a line per clip with each engine's
reading so two runs can be diffed rather than only compared in aggregate. `<out>.md` is
generated and overwritten every run; what a run *means* goes in `<out>.reading.md`, which the
report appends to itself, so the prose survives the next run. The header of the
report records the utter revision, the engine versions and the machine, without which the
compute figures mean nothing.
"""

import argparse
import array
import json
import math
import platform
import subprocess
import sys
import tempfile
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path

RATE = 16000
START = time.monotonic()


def note(msg):
    """Progress to stderr, so a twenty-minute run is distinguishable from a hung one. The
    report itself goes to stdout and the files, and is not disturbed by this."""
    print(f"[{time.monotonic() - START:6.1f}s] {msg}", file=sys.stderr, flush=True)


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

# Filler for the grammar-size sweep, in the order added. Any word the model's table lacks is
# dropped, so the sweep reports the sizes it reached rather than the ones asked for.
SWEEP_EXTRA = [
    "north",
    "south",
    "east",
    "west",
    "up",
    "down",
    "open",
    "close",
    "start",
    "stop",
    "pause",
    "resume",
    "next",
    "previous",
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "last",
    "begin",
    "end",
    "enter",
    "exit",
    "select",
    "cancel",
    "confirm",
    "accept",
    "reject",
    "yes",
    "no",
    "maybe",
    "please",
    "repeat",
    "louder",
    "quieter",
    "faster",
    "slower",
    "bigger",
    "smaller",
    "lighter",
    "darker",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "today",
    "tomorrow",
    "yesterday",
    "morning",
    "evening",
    "night",
    "noon",
    "midnight",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "spring",
    "summer",
    "autumn",
    "winter",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "million",
    "quarter",
    "half",
    "double",
    "triple",
    "add",
    "remove",
    "insert",
    "delete",
    "copy",
    "paste",
    "undo",
    "redo",
    "save",
    "load",
    "new",
    "clear",
    "reset",
    "apply",
    "play",
    "record",
    "mute",
    "unmute",
    "volume",
    "channel",
    "screen",
    "window",
    "menu",
    "button",
    "switch",
    "toggle",
    "left",
    "right",
    "centre",
    "middle",
    "top",
    "bottom",
    "front",
    "back",
    "inside",
    "outside",
    "above",
    "below",
    "near",
    "far",
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "brown",
    "grey",
    "silver",
    "gold",
    "phone",
    "table",
    "chair",
    "door",
    "floor",
    "wall",
    "light",
    "lamp",
    "clock",
    "radio",
    "camera",
    "mouse",
    "keyboard",
    "water",
    "coffee",
    "music",
    "movie",
    "picture",
    "message",
    "number",
    "letter",
    "word",
    "line",
    "page",
    "file",
    "folder",
]


def words_of(text):
    return [w for w in text.split() if not (w.startswith("[") and w.endswith("]"))]


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def quantile(v, q):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))]


def mcnemar(b, c):
    """Two-sided exact McNemar on the discordant counts: the chance of a split at least this
    lopsided if each engine were equally likely to win one. Returns 1.0 when neither differs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, in the order given. Testing one word at a time and
    reading every p below 0.05 as a finding would turn two of thirty-five words into findings
    by chance alone; this is the correction for having asked the question that many times."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    out = [1.0] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        adjusted = min(1.0, (len(pvalues) - rank) * pvalues[i])
        running = max(running, adjusted)  # adjusted p must not decrease with rank
        out[i] = running
    return out


def provenance(args, modules):
    """What a reader needs to judge the figures: the build, the engines, the machine."""

    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None

    here = Path(__file__).resolve().parent
    rev = run(["git", "-C", str(here), "rev-parse", "--short", "HEAD"])
    dirty = run(["git", "-C", str(here), "status", "--porcelain"])
    cpu = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    versions = {}
    for name in modules:
        try:
            import importlib.metadata as md

            versions[name] = md.version(name)
        except Exception:
            versions[name] = "?"
    return dict(
        date=time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        utter_revision=(rev + ("+dirty" if dirty else "")) if rev else "?",
        engines=versions,
        model=Path(args.model).name,
        cpu=cpu or platform.processor() or platform.machine(),
        python=platform.python_version(),
        block_ms=args.block_ms,
    )


def write_clip_records(path, clips, results):
    """One line per clip with every engine's reading, so two runs can be diffed."""
    with open(path, "w") as f:
        for label, clip in clips:
            row = {"clip": str(clip), "label": label}
            for name, r in results.items():
                o = r["outcome"].get(str(clip))
                if o is not None:
                    row[name] = o
            f.write(json.dumps(row) + "\n")


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
        # The wheel and the binding spell the vocabulary lookup differently.
        self.find = getattr(self.model, "FindWord", None) or getattr(self.model, "vosk_model_find_word", None)
        self.missing = [w for e in grammar for w in e.split() if not self.knows(w)]

    def knows(self, word):
        return self.find is None or all(self.find(t) >= 0 for t in word.split())

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
    """Returns (final words, first-appearance second of the label or None, first word shown or
    None, (construction seconds, decode seconds))."""
    state = {"first": None, "first_word": None}

    def on_partial(fed, p):
        partial = words_of(p.get("partial", ""))
        if partial and state["first_word"] is None:
            state["first_word"] = partial[0]
        if state["first"] is None and partial[:1] == [label]:
            state["first"] = fed / RATE

    t0 = time.perf_counter()
    rec = engine.new()
    t1 = time.perf_counter()
    finals = feed(rec, pcm, block_ms, on_partial)
    t2 = time.perf_counter()
    return final_words(finals), state["first"], state["first_word"], (t1 - t0, t2 - t1)


def agreement_table(clips, results, lines, report):
    """How often the two engines return the same final, which accuracy cannot say: both can be
    right equally often and still read different clips differently, and the contract is to
    reproduce the oracle rather than to match its score. The empty-final counts are split by
    direction because a one-way excess of them is what a decoder losing tokens looks like, and
    no accuracy or latency figure on this page would show it."""
    names = list(results)
    if len(names) != 2:
        return
    a, b = names
    oa, ob = results[a]["outcome"], results[b]["outcome"]
    same = 0
    a_right = b_right = neither = 0
    empty = {a: 0, b: 0}
    only_empty = {a: 0, b: 0}
    for label, path in clips:
        wa = oa[str(path)]["words"]
        wb = ob[str(path)]["words"]
        for name, w in ((a, wa), (b, wb)):
            empty[name] += not w
        if not wa and wb:
            only_empty[a] += 1
        if not wb and wa:
            only_empty[b] += 1
        if wa == wb:
            same += 1
        elif wa == [label]:
            a_right += 1
        elif wb == [label]:
            b_right += 1
        else:
            neither += 1
    total = len(clips)
    differ = total - same
    lines.append("")
    lines.append("## Agreement with the oracle")
    lines.append("")
    lines.append(
        f"The contract is to reproduce {a}'s output, which the accuracy above cannot measure: two "
        "engines are at parity when they are right equally often, and that is compatible with their "
        "being right about different clips. This counts the clips whose finals are identical word "
        f"for word. Every clip is in `{Path(report['out']).name}.clips.jsonl` with both readings, so "
        "the ones that differ can be listed rather than only counted."
    )
    lines.append("")
    lines.append("| pair | finals identical | differ |")
    lines.append("|---|---|---|")
    lines.append(f"| {a} vs {b} | {same} / {total} ({pct(same, total):.2f}%) | {differ} |")
    lines.append("")
    lines.append(
        f"Of the {differ} that differ, {a} reads the label and {b} does not on {a_right}, "
        f"{b} and not {a} on {b_right}, and neither on {neither}."
    )
    lines.append("")
    lines.append("| engine | empty finals | empty here, a word from the other |")
    lines.append("|---|---|---|")
    for name in (a, b):
        lines.append(f"| {name} | {empty[name]} ({pct(empty[name], total):.2f}%) | {only_empty[name]} |")
    lines.append("")
    lines.append(
        "The last column is the directional one. Equal totals in the middle column can still hide "
        "two engines falling silent on different clips, and an excess in one direction is the "
        "signature of a decoder dropping a reading the other keeps."
    )
    report["agreement"] = dict(
        pair=[a, b],
        total=total,
        identical=same,
        differ=differ,
        differ_a_right=a_right,
        differ_b_right=b_right,
        differ_neither=neither,
        empty=empty,
        empty_one_way=only_empty,
    )


def significance_table(clips, results, lines, report):
    """The engines decode the same clips, so the comparison is paired: only the clips they
    disagree on carry information. Aggregate counts alone cannot say whether a gap is real."""
    names = list(results)
    if len(names) != 2:
        return
    a, b = names
    oa, ob = results[a]["outcome"], results[b]["outcome"]
    out = {}
    lines.append("")
    lines.append("## Where the engines differ")
    lines.append("")
    lines.append(
        f"Paired over the same clips: `{a} only` counts clips {a} got right and {b} did not, "
        f"`{b} only` the reverse. The p-value is a two-sided exact McNemar test on those two "
        "counts; a large one means the split is what chance would produce."
    )
    lines.append("")
    lines.append(f"| scope | {a} only | {b} only | p | Holm |")
    lines.append("|---|---|---|---|---|")
    rows = [("all words", [c for _, c in clips])]
    by_word = defaultdict(list)
    for label, clip in clips:
        by_word[label].append(clip)
    for w in DATASET_WORDS:
        rows.append((w, by_word[w]))
    counted = []
    for scope, paths in rows:
        only_a = only_b = 0
        for path in paths:
            ca = oa.get(str(path), {}).get("correct")
            cb = ob.get(str(path), {}).get("correct")
            if ca and not cb:
                only_a += 1
            elif cb and not ca:
                only_b += 1
        counted.append((scope, only_a, only_b, mcnemar(only_a, only_b)))
    words = [c for c in counted if c[0] != "all words"]
    adjusted = holm([c[3] for c in words])
    overall = next(c for c in counted if c[0] == "all words")
    out["all words"] = dict(only_a=overall[1], only_b=overall[2], p=overall[3], holm=None)
    lines.append(f"| all words | {overall[1]} | {overall[2]} | {overall[3]:.3f} | |")
    shown = 0
    for (scope, only_a, only_b, p), q in sorted(zip(words, adjusted), key=lambda z: z[0][3]):
        out[scope] = dict(only_a=only_a, only_b=only_b, p=p, holm=q)
        if p < 0.05:
            lines.append(f"| {scope} | {only_a} | {only_b} | {p:.3f} | {q:.3f} |")
            shown += 1
    if not shown:
        lines.append("")
        lines.append("No single word's difference reaches p < 0.05 before correction.")
    survivors = [w for w in out if w != "all words" and out[w]["holm"] is not None and out[w]["holm"] < 0.05]
    lines.append("")
    lines.append(
        f"{len(words)} words were tested, so about two would fall below 0.05 by chance; the last "
        "column is the Holm-Bonferroni adjustment for that. "
        + ("Surviving it: " + ", ".join(sorted(survivors)) + "." if survivors else "No word survives it.")
    )
    report["significance"] = out


def full_grammar_pass(modules, model, clips, block_ms, grammar, lines, report):
    results = {}
    ref_end = {}
    note(f"full grammar: {len(clips)} clips x {len(modules)} engines")
    for name, mod in modules.items():
        eng = Engine(name, mod, model, grammar)
        note(f"  {name}")
        if eng.missing:
            lines.append(f"- {name}: words absent from the model's table: {eng.missing}")
        correct = never = shown = changed = 0
        conf = Counter()
        per_word = Counter()
        per_word_ok = Counter()
        lat = []
        outcome = {}
        ctors = []
        ctor = decode = audio = 0.0
        for done, (label, path) in enumerate(clips):
            if done and done % 2500 == 0:
                note(f"    {done} / {len(clips)}")
            pcm = read_pcm(path)
            audio += len(pcm) / 2 / RATE
            if path not in ref_end:
                ref_end[path] = energy_end(pcm)
            words, first, first_word, (t_new, t_decode) = run_clip(eng, pcm, block_ms, label)
            ctor += t_new
            ctors.append(t_new)
            decode += t_decode
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
            outcome[str(path)] = dict(
                words=words,
                correct=words == [label],
                first=first,
                first_word=first_word,
            )
        results[name] = dict(
            outcome=outcome,
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
            rtf=(ctor + decode) / audio if audio else None,
            decode_rtf=decode / audio if audio else None,
            ctor_ms=1000.0 * ctor / len(clips) if clips else None,
            ctor_first_ms=1000.0 * ctors[0] if ctors else None,
            ctor_rest_ms=1000.0 * sum(ctors[1:]) / max(1, len(ctors) - 1) if ctors else None,
        )
    lines.append("## Full grammar: accuracy")
    lines.append("")
    lines.append(
        "| engine | correct | accuracy | never in a partial | first appearance after the clip's energy end p50 / p90 | first word shown later changed |"
    )
    lines.append("|---|---|---|---|---|---|")
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['correct']} / {r['total']} | {pct(r['correct'], r['total']):.2f}% | {r['never_in_partial']} | "
            f"{r['latency_ms_p50']:.0f} / {r['latency_ms_p90']:.0f} ms over {r['latency_n']} | "
            f"{r['first_changed']} / {r['first_shown']} ({pct(r['first_changed'], r['first_shown']):.1f}%) |"
        )
    lines.append("")
    lines.append("Compute, one recognizer per clip:")
    lines.append("")
    lines.append(
        "| engine | first construction, ms | every later one, ms | mean, ms | RTF, decode only | RTF with construction |"
    )
    lines.append("|---|---|---|---|---|---|")
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['ctor_first_ms']:.2f} | {r['ctor_rest_ms']:.3f} | {r['ctor_ms']:.3f} | "
            f"{r['decode_rtf']:.4f} | {r['rtf']:.4f} |"
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
    agreement_table(clips, results, lines, report)
    significance_table(clips, results, lines, report)
    lines.append("")
    report["full"] = results


def trace_clip(engine, pcm, block_ms):
    """Every partial the engine showed, in order, and the final. The trace is kept whole because
    a final can be stable while the path to it is not, and the path is what a host renders."""
    partials = []
    finals = feed(engine.new(), pcm, block_ms, lambda fed, p: partials.append(p.get("partial", "")))
    return [partials, final_words(finals)]


def determinism_traces(modules, model, clips, block_ms, grammar):
    """One pass over the clips per engine, a recognizer each, as a host would run them."""
    out = {}
    for name, mod in modules.items():
        eng = Engine(name, mod, model, grammar)
        out[name] = [trace_clip(eng, read_pcm(path), block_ms) for _, path in clips]
    return out


def trace_worker(job_path):
    """The fresh process's half of the determinism pass: the same clips, a process of its own,
    the traces back over stdout."""
    job = json.loads(Path(job_path).read_text())
    modules = {}
    for name in job["engines"]:
        modules[name] = __import__("vosk" if name == "vosk" else name)
        if name == "vosk":
            modules[name].SetLogLevel(-1)
    clips = [(None, Path(p)) for p in job["clips"]]
    json.dump(determinism_traces(modules, job["model"], clips, job["block_ms"], job["grammar"]), sys.stdout)


def fresh_process_traces(model, clips, block_ms, grammar, engines):
    """The same decode in a process started for it. A repeat inside one process cannot see a
    per-process seed: it draws the same one. Rust's default hasher is seeded once per process, so
    a hash map's iteration order — and with it which tokens a narrowing cutoff reaches first and
    prunes — is fixed for a run and free to move between runs. Only a second process can show it."""
    job = dict(
        model=str(model),
        block_ms=block_ms,
        grammar=grammar,
        engines=list(engines),
        clips=[str(p) for _, p in clips],
    )
    path = Path(tempfile.mkdtemp()) / "trace-job.json"
    path.write_text(json.dumps(job))
    try:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--trace-job", str(path)],
            capture_output=True,
            text=True,
        )
    finally:
        path.unlink()
        path.parent.rmdir()
    if proc.returncode != 0:
        note(f"  fresh process failed ({proc.returncode}): {proc.stderr.strip().splitlines()[-1:]}")
        return {}
    return json.loads(proc.stdout)


def determinism_pass(modules, model, clips, block_ms, grammar, lines, report, count):
    """Whether the same bytes give the same reading twice. Nothing else on this page would
    notice if they did not: every other pass decodes each clip once, so a reading that moves
    between runs is invisible to it and shows up only as noise in next week's comparison."""
    step = max(1, len(clips) // count)
    chosen = clips[::step][:count]
    note(f"determinism: {len(chosen)} clips x {len(modules)} engines, twice here and once in a fresh process")
    first = determinism_traces(modules, model, chosen, block_ms, grammar)
    second = determinism_traces(modules, model, chosen, block_ms, grammar)
    fresh = fresh_process_traces(model, chosen, block_ms, grammar, modules)
    lines.append("## Determinism")
    lines.append("")
    lines.append(
        f"{len(chosen)} clips decoded twice in this process and once in a process started for the "
        "purpose, comparing the whole partial trace and the final. The fresh process is the half "
        "that matters: a hash seed drawn once per process gives a map the same iteration order for "
        "every repeat within a run, so a reading that depends on it is perfectly stable until the "
        "next run. Any clip that moves is named in the JSON."
    )
    lines.append("")
    lines.append(
        "| engine | clips | same process: partials | same process: finals | fresh process: partials | fresh process: finals |"
    )
    lines.append("|---|---|---|---|---|---|")
    out = {}
    for name in modules:
        row = {}
        for tag, other in [("same_process", second[name]), ("fresh_process", fresh.get(name) or [])]:
            moved = {"partials": [], "finals": []}
            for (_, path), x, y in zip(chosen, first[name], other):
                if x[0] != y[0]:
                    moved["partials"].append(str(path))
                if x[1] != y[1]:
                    moved["finals"].append(str(path))
            moved["compared"] = min(len(first[name]), len(other))
            row[tag] = moved
        out[name] = row
        s, f = row["same_process"], row["fresh_process"]
        cells = [len(s["partials"]), len(s["finals"])]
        cells += ["n/a", "n/a"] if not f["compared"] else [len(f["partials"]), len(f["finals"])]
        lines.append(f"| {name} | {len(chosen)} | " + " | ".join(str(c) for c in cells) + " |")
    lines.append("")
    unstable = [
        f"{name}, {tag.replace('_', ' ')}: {len(m['partials'])} partial traces and {len(m['finals'])} finals"
        for name, row in out.items()
        for tag, m in row.items()
        if m["partials"] or m["finals"]
    ]
    lines.append(
        "Every clip read the same way every time."
        if not unstable
        else "**Not reproducible.** " + "; ".join(unstable) + "."
    )
    lines.append("")
    report["determinism"] = out


def endpoint_time(engine, pcm, block_ms):
    """The fed-audio second at which the engine first closes a segment of its own accord, or None
    if it never does within the audio given."""
    rec = engine.new()
    block = RATE * block_ms // 1000 * 2
    fed = 0
    for i in range(0, len(pcm), block):
        chunk = pcm[i : i + block]
        fed += len(chunk) // 2
        if rec.AcceptWaveform(chunk):
            return fed / RATE
    return None


def endpoint_pass(modules, model, clips, block_ms, grammar, lines, report, count, pad_ms):
    """How long after the speech stops the final arrives. First appearance says when a word
    becomes visible; this says when the engine commits, which is the pause a speaker sits through
    before the turn comes back and the thing a host tunes endpointing for.

    A Speech Commands clip ends when the word does, and neither engine endpoints inside one, so
    the clip is followed by silence. Without it the only final is the one end-of-audio forces,
    whose arrival time is the clip's length and says nothing about the decoder."""
    step = max(1, len(clips) // count)
    chosen = clips[::step][:count]
    pad = b"\x00\x00" * (RATE * pad_ms // 1000)
    note(f"endpoint: {len(chosen)} clips x {len(modules)} engines, {pad_ms} ms of silence appended")
    lines.append("## Endpoint latency")
    lines.append("")
    lines.append(
        f"{len(chosen)} clips, each followed by {pad_ms} ms of silence, measured to the first block "
        "at which the engine closes a segment itself. Reported against the clip's own energy end, "
        "the same reference the first-appearance figure uses. Neither engine endpoints within the "
        "clip alone, so without the silence there is nothing here to measure."
    )
    lines.append("")
    lines.append("| engine | endpointed | after the clip's energy end p50 / p90 / p99 | never within the silence |")
    lines.append("|---|---|---|---|")
    out = {}
    at_by_engine = {}
    for name, mod in modules.items():
        eng = Engine(name, mod, model, grammar)
        delays = []
        never = 0
        at_by_engine[name] = {}
        for _, path in chosen:
            pcm = read_pcm(path)
            end = energy_end(pcm)
            at = endpoint_time(eng, pcm + pad, block_ms)
            at_by_engine[name][str(path)] = at
            if at is None:
                never += 1
            elif end is not None:
                delays.append((at - end) * 1000.0)
        out[name] = dict(
            endpointed=len(chosen) - never,
            never=never,
            ms_p50=quantile(delays, 0.5),
            ms_p90=quantile(delays, 0.9),
            ms_p99=quantile(delays, 0.99),
            n=len(delays),
        )
        r = out[name]
        lines.append(
            f"| {name} | {r['endpointed']} / {len(chosen)} | "
            f"{r['ms_p50']:.0f} / {r['ms_p90']:.0f} / {r['ms_p99']:.0f} ms over {r['n']} | {r['never']} |"
        )
    lines.append("")
    names = list(modules)
    agree = None
    if len(names) == 2:
        a, b = names
        both = same = 0
        for _, path in chosen:
            x, y = at_by_engine[a][str(path)], at_by_engine[b][str(path)]
            if x is not None and y is not None:
                both += 1
                same += x == y
        agree = dict(both=both, same=same)
        lines.append(
            f"Both engines endpointed on {both} of the {len(chosen)}, at the same block on {same} "
            f"({pct(same, both):.2f}%). Endpointing is a second surface the contract covers, and one "
            "the accuracy figures do not touch: a host that agreed on every word would still feel a "
            "difference here."
        )
        lines.append("")
    report["endpoint"] = dict(clips=len(chosen), pad_ms=pad_ms, engines=out, agreement=agree)


def block_size_pass(modules, model, clips, grammar, lines, report, sizes, count):
    """Block size is the host's latency-against-compute dial, and the page measures one setting
    of it. A smaller block shows a word sooner and pays the per-block cost more often per second
    of audio; whether it also changes what is decoded is the question worth asking, because a
    block boundary is not supposed to be audible in the result."""
    step = max(1, len(clips) // count)
    chosen = clips[::step][:count]
    pcms = [(label, read_pcm(path)) for label, path in chosen]
    ends = [energy_end(pcm) for _, pcm in pcms]
    audio = sum(len(p) / 2 / RATE for _, p in pcms)
    note(f"block size: {len(sizes)} sizes x {len(chosen)} clips x {len(modules)} engines")
    lines.append("## Block size")
    lines.append("")
    lines.append(
        f"{len(chosen)} clips at each block size, the same clips throughout. A word can only appear "
        "at a block boundary, but the decoder only revises a partial when the network has consumed "
        "a whole chunk, so the block is the coarser of the two grids only when it fails to land on "
        "the chunk. A size that does not divide the chunk's spacing is included deliberately: "
        "without one, the sweep cannot tell a decoder that ignores block size from a sweep that "
        "happened to pick sizes it agrees with. The accuracy column is the one that should not move."
    )
    lines.append("")
    lines.append(
        "| block ms | engine | accuracy | first appearance p50 / p90 | same instant as the finest block | RTF, decode only |"
    )
    lines.append("|---|---|---|---|---|---|")
    out = {}
    finest = min(sizes)
    at_finest = {}
    for ms in sizes:
        out[ms] = {}
        for name, mod in modules.items():
            eng = Engine(name, mod, model, grammar)
            ok = 0
            lat = []
            decode = 0.0
            firsts = []
            for (label, pcm), end in zip(pcms, ends):
                words, first, _, (_, t_dec) = run_clip(eng, pcm, ms, label)
                ok += words == [label]
                decode += t_dec
                firsts.append(first)
                if first is not None and end is not None:
                    lat.append((first - end) * 1000.0)
            if ms == finest:
                at_finest[name] = firsts
            both = [(x, y) for x, y in zip(firsts, at_finest.get(name, [])) if x is not None and y is not None]
            shifted = [round((x - y) * 1000.0) for x, y in both]
            out[ms][name] = dict(
                correct=ok,
                accuracy=pct(ok, len(pcms)),
                latency_ms_p50=quantile(lat, 0.5),
                latency_ms_p90=quantile(lat, 0.9),
                latency_n=len(lat),
                same_as_finest=sum(1 for d in shifted if d == 0),
                compared_to_finest=len(shifted),
                max_shift_ms=max(shifted) if shifted else 0,
                decode_rtf=decode / audio if audio else None,
            )
            r = out[ms][name]
            same = f"{r['same_as_finest']} / {r['compared_to_finest']}"
            if r["max_shift_ms"]:
                same += f", up to {r['max_shift_ms']} ms later"
            lines.append(
                f"| {ms} | {name} | {r['correct']} / {len(pcms)} ({r['accuracy']:.1f}%) | "
                f"{r['latency_ms_p50']:.0f} / {r['latency_ms_p90']:.0f} ms | {same} | {r['decode_rtf']:.4f} |"
            )
    lines.append("")
    lines.append(
        "The column that carries the result is the per-clip one, not the quantiles: two block "
        "sizes can produce the same p50 by luck, and only a clip-by-clip comparison against the "
        f"finest block ({finest} ms) shows whether the word actually appeared at the same instant."
    )
    lines.append("")
    report["block_size"] = dict(clips=len(chosen), finest_ms=finest, sizes=out)


def twelve_class_pass(modules, model, clips, noise, block_ms, lines, report):
    lines.append("## Twelve-class: ten commands plus the unknown-word symbol")
    lines.append("")
    lines.append(
        "| engine | commands correct | fillers and digits read as unknown | fillers and digits read as a command | noise seconds | noise seconds with a word |"
    )
    lines.append("|---|---|---|---|---|---|")
    twelve = {}
    note(f"twelve-class: {len(clips)} clips x {len(modules)} engines")
    for name, mod in modules.items():
        note(f"  {name}")
        eng = Engine(name, mod, model, COMMANDS + ["[unk]"])
        cmd_total = cmd_ok = other_total = other_unk = other_cmd = 0
        for done, (label, path) in enumerate(clips):
            if done and done % 2500 == 0:
                note(f"    {done} / {len(clips)}")
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
    note(f"background noise: {len(modules)} engines x 2 grammars over the noise recordings")
    for name, mod in modules.items():
        for variant, g in [("full", grammar), ("full + [unk]", grammar + ["[unk]"])]:
            note(f"  {name}, {variant}")
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


def mix_at_snr(pcm, noise, offset, snr_db):
    """The clip with `noise` laid under it at `snr_db`, scaling the noise to the clip's own
    speech level. Saturates rather than wraps, as an ADC would."""
    a = array.array("h")
    a.frombytes(pcm)
    n = array.array("h")
    n.frombytes(noise)
    if not a or not n:
        return pcm
    sig = math.sqrt(sum(float(v) * v for v in a) / len(a))
    seg = [n[(offset + i) % len(n)] for i in range(len(a))]
    nse = math.sqrt(sum(float(v) * v for v in seg) / len(seg))
    if sig == 0 or nse == 0:
        return pcm
    gain = (sig / nse) / (10.0 ** (snr_db / 20.0))
    out = array.array("h", [0]) * len(a)
    for i in range(len(a)):
        v = int(a[i] + gain * seg[i])
        out[i] = -32768 if v < -32768 else (32767 if v > 32767 else v)
    return out.tobytes()


def snr_pass(modules, model, clips, noise_paths, block_ms, grammar, lines, report, levels, count):
    """Accuracy against noise. Clean one-second clips are the most forgiving regime there is;
    a front-end difference shows itself here first. Both engines are fed the same mixed bytes."""
    step = max(1, len(clips) // count)
    chosen = clips[::step][:count]
    noise = b"".join(read_pcm(p) for p in noise_paths)
    mixed = {}
    for level in levels:
        mixed[level] = [
            (label, mix_at_snr(read_pcm(path), noise, (i * 7919) % max(1, len(noise) // 2), level))
            for i, (label, path) in enumerate(chosen)
        ]
    out = {}
    outcome = {}
    note(f"noise: {len(chosen)} clips x {len(levels)} levels + clean x {len(modules)} engines")
    lines.append("## Accuracy against noise")
    lines.append("")
    lines.append(
        f"{len(chosen)} clips of the testing split, the dataset's own background recordings laid "
        "under each at the stated signal-to-noise ratio, the same mixed samples to both engines."
    )
    lines.append("")
    lines.append("| engine | " + " | ".join(f"{level} dB" for level in levels) + " | clean |")
    lines.append("|---|" + "---|" * (len(levels) + 1))
    for name, mod in modules.items():
        eng = Engine(name, mod, model, grammar)
        row = {}
        outcome[name] = {}
        for level in levels:
            ok = 0
            marks = []
            for label, pcm in mixed[level]:
                words, _, _, _ = run_clip(eng, pcm, block_ms, label)
                got = words == [label]
                marks.append(got)
                ok += got
            row[level] = ok
            outcome[name][level] = marks
        clean = 0
        marks = []
        for label, path in chosen:
            words, _, _, _ = run_clip(eng, read_pcm(path), block_ms, label)
            got = words == [label]
            marks.append(got)
            clean += got
        row["clean"] = clean
        outcome[name]["clean"] = marks
        out[name] = {str(k): v for k, v in row.items()}
        cells = " | ".join(f"{row[level]} ({pct(row[level], len(chosen)):.1f}%)" for level in levels)
        lines.append(f"| {name} | {cells} | {clean} ({pct(clean, len(chosen)):.1f}%) |")
    lines.append("")
    # [[rr:Benchmarks]]
    paired = {}
    names = list(modules)
    if len(names) == 2:
        a, b = names
        lines.append(f"Paired at each level, as above: `{a} only`, `{b} only`, and the exact McNemar p.")
        lines.append("")
        lines.append("| level | " + f"{a} only | {b} only | p |")
        lines.append("|---|---|---|---|")
        for level in [*levels, "clean"]:
            ma, mb = outcome[a][level], outcome[b][level]
            only_a = sum(1 for x, y in zip(ma, mb) if x and not y)
            only_b = sum(1 for x, y in zip(ma, mb) if y and not x)
            pv = mcnemar(only_a, only_b)
            paired[str(level)] = dict(only_a=only_a, only_b=only_b, p=pv)
            label = f"{level} dB" if level != "clean" else "clean"
            lines.append(f"| {label} | {only_a} | {only_b} | {pv:.3f} |")
        lines.append("")
    report["snr"] = dict(clips=len(chosen), levels=levels, results=out, paired=paired)


def grammar_size_pass(modules, model, clips, block_ms, lines, report, sizes, count):
    """How the figures move with the grammar, the one dial a host turns. The dataset's own 35
    words are always in, so every clip stays decidable; the rest is filler."""
    step = max(1, len(clips) // count)
    chosen = clips[::step][:count]
    pcms = [(label, read_pcm(path)) for label, path in chosen]
    audio = sum(len(p) / 2 / RATE for _, p in pcms)
    probe = Engine("probe", next(iter(modules.values())), model, DATASET_WORDS)
    pool = []
    for w in LETTERS + NATO + COLOURS + SWEEP_EXTRA:
        if w in DATASET_WORDS or w in pool:
            continue
        if probe.knows(w):
            pool.append(w)
    out = {}
    note(f"grammar size: {len(sizes)} sizes x {len(chosen)} clips x {len(modules)} engines")
    lines.append("## Grammar size")
    lines.append("")
    lines.append(
        f"{len(chosen)} clips, the 35 dataset words plus filler up to each size, so every clip "
        f"stays decidable at every size. The runtime refuses 300 distinct words or more; the "
        f"sweep stops at {len(DATASET_WORDS) + len(pool)}, the filler this model's table knows."
    )
    lines.append("")
    lines.append(
        "| entries | " + " | ".join(f"{n} accuracy | {n} first ms | {n} later ms | {n} RTF" for n in modules) + " |"
    )
    lines.append("|---|" + "---|" * (4 * len(modules)))
    done = set()
    for size in sizes:
        extra = pool[: max(0, size - len(DATASET_WORDS))]
        grammar = DATASET_WORDS + extra
        if len(grammar) in done:
            continue
        done.add(len(grammar))
        cells = []
        row = {}
        for name, mod in modules.items():
            eng = Engine(name, mod, model, grammar)
            ok = 0
            ctors = []
            decode = 0.0
            for label, pcm in pcms:
                words, _, _, (t_new, t_dec) = run_clip(eng, pcm, block_ms, label)
                ok += words == [label]
                ctors.append(t_new)
                decode += t_dec
            row[name] = dict(
                correct=ok,
                accuracy=pct(ok, len(pcms)),
                ctor_first_ms=1000.0 * ctors[0],
                ctor_rest_ms=1000.0 * sum(ctors[1:]) / max(1, len(ctors) - 1),
                decode_rtf=decode / audio,
            )
            r = row[name]
            cells += [
                f"{r['accuracy']:.1f}%",
                f"{r['ctor_first_ms']:.2f}",
                f"{r['ctor_rest_ms']:.3f}",
                f"{r['decode_rtf']:.4f}",
            ]
        out[len(grammar)] = row
        lines.append(f"| {len(grammar)} | " + " | ".join(cells) + " |")
    lines.append("")
    report["grammar_size"] = dict(clips=len(chosen), sizes=out)


def steady_state_pass(modules, model, clips, block_ms, grammar, lines, report, seconds):
    """Compute on continuous audio: one recognizer over the clips joined end to end, which is
    how a host runs, rather than the fixed cost of a clip at a time."""
    step = max(1, len(clips) // seconds)
    joined = b"".join(read_pcm(p) for _, p in clips[::step][:seconds])
    audio = len(joined) / 2 / RATE
    block = RATE * block_ms // 1000 * 2
    out = {}
    note(f"continuous audio: {audio:.0f} s x {len(modules)} engines")
    lines.append("## Compute on continuous audio")
    lines.append("")
    lines.append(f"One recognizer over {audio:.0f} s of the clips joined end to end.")
    lines.append("")
    lines.append("| engine | RTF | per-block compute ms p50 / p95 / p99 |")
    lines.append("|---|---|---|")
    for name, mod in modules.items():
        eng = Engine(name, mod, model, grammar)
        feed(eng.new(), joined[: block * 50], block_ms)  # warm the code paths
        rec = eng.new()
        per = []
        for i in range(0, len(joined), block):
            t = time.perf_counter()
            rec.AcceptWaveform(joined[i : i + block])
            per.append((time.perf_counter() - t) * 1000.0)
        rec.FinalResult()
        total = sum(per) / 1000.0
        out[name] = dict(
            rtf=total / audio,
            ms_p50=quantile(per, 0.5),
            ms_p95=quantile(per, 0.95),
            ms_p99=quantile(per, 0.99),
            seconds=audio,
        )
        r = out[name]
        lines.append(f"| {name} | {r['rtf']:.4f} | {r['ms_p50']:.3f} / {r['ms_p95']:.2f} / {r['ms_p99']:.2f} |")
    lines.append("")
    report["steady_state"] = out


def preflight(modules, model, grammar, block_ms, clips, lines, report):
    """Open the model and decode one clip on every engine before the run commits twenty minutes
    to it. A model an engine cannot decode is a result worth having, but it is worth having in
    the first ten seconds and in one piece: without this the run ends part-way through a pass,
    having written nothing. A Rust panic reaches Python as a BaseException, so the catch is wide.

    The run stops rather than dropping the engine, because every comparison on this page is
    paired. A page with one engine's columns silently missing is worse than no page."""
    out = {}
    failed = []
    for name, mod in modules.items():
        t = time.perf_counter()
        try:
            eng = Engine(name, mod, model, grammar)
            load = time.perf_counter() - t
            t = time.perf_counter()
            rec = eng.new()
            ctor = time.perf_counter() - t
            feed(rec, read_pcm(clips[0][1]), block_ms)
        except BaseException as e:  # noqa: BLE001 - a panic in the engine is the finding
            failed.append(f"{name}: {type(e).__name__}: {str(e).splitlines()[0] if str(e) else '(no message)'}")
            out[name] = dict(ok=False, error=f"{type(e).__name__}: {e}")
            continue
        out[name] = dict(ok=True, model_load_s=load, first_ctor_ms=1000.0 * ctor)
        note(f"preflight {name}: model {load:.2f} s, first recognizer {1000.0 * ctor:.1f} ms")
    report["preflight"] = out
    if failed:
        for line in failed:
            note(f"preflight FAILED {line}")
        raise SystemExit(
            f"{len(failed)} of {len(modules)} engines cannot decode {Path(model).name}:\n  "
            + "\n  ".join(failed)
            + "\nEvery comparison on this page is paired, so the run stops here rather than "
            "writing a page with an engine missing. Use --engines to measure one deliberately."
        )
    lines.append("## Engines")
    lines.append("")
    lines.append(
        "Each engine opened the model and decoded a clip before the run began; a model an engine "
        "cannot decode stops the run here rather than part-way through a pass."
    )
    lines.append("")
    lines.append("| engine | model load, s | first recognizer, ms |")
    lines.append("|---|---|---|")
    for name, r in out.items():
        lines.append(f"| {name} | {r['model_load_s']:.2f} | {r['first_ctor_ms']:.1f} |")
    lines.append("")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--trace-job":
        return trace_worker(sys.argv[2])
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="output prefix; writes .md and .json")
    ap.add_argument("--split", default="testing", choices=["testing", "validation"])
    ap.add_argument("--limit", type=int, default=0, help="clips per word, 0 for all")
    ap.add_argument("--block-ms", type=int, default=40)
    ap.add_argument("--engines", default="vosk,utterpy")
    ap.add_argument("--steady-state-clips", type=int, default=300, help="clips joined for the compute pass")
    ap.add_argument("--snr-db", default="20,10,5,0", help="signal-to-noise ratios to mix, empty to skip")
    ap.add_argument("--snr-clips", type=int, default=800, help="clips per noise level")
    ap.add_argument("--grammar-sizes", default="35,60,92,150,200,246", help="grammar sweep, empty to skip")
    ap.add_argument("--grammar-clips", type=int, default=400, help="clips per grammar size")
    ap.add_argument("--determinism-clips", type=int, default=200, help="clips decoded three times, 0 to skip")
    ap.add_argument("--endpoint-clips", type=int, default=800, help="clips for endpoint latency, 0 to skip")
    ap.add_argument("--endpoint-pad-ms", type=int, default=2000, help="silence appended so an endpoint can fire")
    ap.add_argument("--block-sizes", default="10,20,40,80,100", help="block-size sweep, empty to skip")
    ap.add_argument("--block-clips", type=int, default=400, help="clips per block size")
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
    prov = provenance(args, modules)
    report = {
        "split": args.split,
        "clips": len(clips),
        "block_ms": args.block_ms,
        "grammar_size": len(full_grammar),
        "out": args.out,
        "provenance": prov,
    }
    lines = [
        f"# Speech Commands v2, {prov['model']}, {args.split} split, {len(clips)} clips, {args.block_ms} ms blocks",
        "",
    ]
    lines.append(
        f"Grammar: the dataset's 35 words plus {len(LETTERS)} letters, {len(NATO)} NATO words and {len(COLOURS)} colours, {len(full_grammar)} entries."
    )
    lines.append("")
    lines.append(
        f"Run {prov['date']}, utter {prov['utter_revision']}, "
        + ", ".join(f"{k} {v}" for k, v in prov["engines"].items())
        + f", model {prov['model']}, {prov['cpu']}, Python {prov['python']}."
    )
    lines.append("")
    preflight(modules, args.model, full_grammar, args.block_ms, clips, lines, report)
    full_grammar_pass(modules, args.model, clips, args.block_ms, full_grammar, lines, report)
    write_clip_records(args.out + ".clips.jsonl", clips, report["full"])
    for r in report["full"].values():
        del r["outcome"]
    if args.determinism_clips:
        determinism_pass(modules, args.model, clips, args.block_ms, full_grammar, lines, report, args.determinism_clips)
    if args.endpoint_clips:
        endpoint_pass(
            modules,
            args.model,
            clips,
            args.block_ms,
            full_grammar,
            lines,
            report,
            args.endpoint_clips,
            args.endpoint_pad_ms,
        )
    block_sizes = [int(v) for v in args.block_sizes.split(",") if v.strip()]
    if block_sizes:
        block_size_pass(modules, args.model, clips, full_grammar, lines, report, block_sizes, args.block_clips)
    steady_state_pass(modules, args.model, clips, args.block_ms, full_grammar, lines, report, args.steady_state_clips)
    levels = [int(v) for v in args.snr_db.split(",") if v.strip()]
    if levels:
        snr_pass(modules, args.model, clips, noise, args.block_ms, full_grammar, lines, report, levels, args.snr_clips)
    sizes = [int(v) for v in args.grammar_sizes.split(",") if v.strip()]
    if sizes:
        grammar_size_pass(modules, args.model, clips, args.block_ms, lines, report, sizes, args.grammar_clips)
    twelve_class_pass(modules, args.model, clips, noise, args.block_ms, lines, report)
    noise_pass(modules, args.model, noise, args.block_ms, full_grammar, lines, report)
    # [[rr:Benchmarks]]
    reading = Path(args.out + ".reading.md")
    if reading.exists():
        lines.append(reading.read_text().strip())
        lines.append("")
    text = "\n".join(lines) + "\n"
    Path(args.out + ".md").write_text(text)
    Path(args.out + ".json").write_text(json.dumps(report, indent=1))
    note(f"written: {args.out}.md, .json, .clips.jsonl")
    print(text)


if __name__ == "__main__":
    main()
