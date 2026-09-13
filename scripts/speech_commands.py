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
- determinism: the whole split decoded in each of two fresh processes, comparing every partial
  with its readings and the final, because a hash seed drawn per process is invisible to a
  repeat inside it and a tie broken by chance on one clip in two thousand is invisible in a
  sample of two hundred;
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
  must decode to [unk], and the background recordings for silence finals per minute;
- background noise under the full grammar, with and without the unknown-word symbol: each
  recording fed continuously through one recognizer, for silence finals per minute and for
  whether the silence reading is still offered as a rival when rank 0 is a word.

Latency is the fed-audio time at which the label first appears in a partial, minus the word's
end from the clip's own energy envelope (the last 10 ms frame within 20 dB of the clip's peak),
a reference that owes nothing to either engine. Also reported: how often the first word shown is
later changed, split into the partial that another partial overtakes and the partial that stands
until the final replaces it, with the lag between the two partials for the first kind.

Writes `<out>.md`, `<out>.json`, and `<out>.clips.jsonl`, a line per clip with each engine's
reading so two runs can be diffed rather than only compared in aggregate. `<out>.md` is
generated and overwritten every run; what a run *means* goes in `<out>.reading.md`, which the
report appends to itself, so the prose survives the next run. The header of the
report records the utter revision, the engine versions and the machine, without which the
compute figures mean nothing.
"""

import argparse
import array
import hashlib
import json
import math
import platform
import random
import re
import subprocess
import sys
import tempfile
import time
import wave
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, NotRequired, TypedDict, cast

RATE = 16000
START = time.monotonic()

Json = dict[str, Any]
Clip = tuple[str, Path]
OnPartial = Callable[[int, Json], None]
# A clip's partials, as a digest of the whole trace, beside its final words.
Trace = list[Any]
Take = tuple[str, Path, float]
# A sweep row: the label it is named by in the table, its grammar, its unknown-word cost.
SweepGrammar = tuple[object, Sequence[str], float | None]
RoomCache = dict[str, list[tuple[str, bytes]]]


class WordlessSample(TypedDict):
    key: str
    pcm: bytes
    speech_end: int
    spoken: list[str]
    samples: int
    source: NotRequired[str]


class GateRow(TypedDict):
    key: str
    energy: float
    floor: float


def note(msg: str) -> None:
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


def words_of(text: str) -> list[str]:
    return [w for w in text.split() if not (w.startswith("[") and w.endswith("]"))]


# The two readings of a path with no word on it. [[rr:TD-10#Decision outcome]]
WORDLESS = ("[sil]", "[speech]")


def pct(a: float, b: float) -> float:
    return 100.0 * a / b if b else float("nan")


def quantile(v: Sequence[float], q: float) -> float:
    if not v:
        return float("nan")
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))]


def mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar on the discordant counts: the chance of a split at least this
    lopsided if each engine were equally likely to win one. Returns 1.0 when neither differs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # In log space throughout: the trust and states pages reach thousands of discordant clips,
    # where 2.0**n and math.comb both overflow a float.
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in range(k + 1)]
    top = max(terms)
    log_tail = top + math.log(sum(math.exp(t - top) for t in terms)) - n * math.log(2.0)
    return min(1.0, 2.0 * math.exp(log_tail)) if log_tail > -700.0 else 0.0


def holm(pvalues: Sequence[float]) -> list[float]:
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


# Rooted pathspecs: git resolves a bare one against the caller's directory, which here is scripts.
SOURCE_PATHS = (":/src", ":/scripts", ":/build.rs", ":/Cargo.toml", ":/Cargo.lock")


def provenance(args: argparse.Namespace, modules: Iterable[str]) -> dict[str, Any]:
    """What a reader needs to judge the figures: the build, the engines, the machine."""

    def run(cmd: Sequence[str]) -> str | None:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None

    here = Path(__file__).resolve().parent
    rev = run(["git", "-C", str(here), "rev-parse", "--short", "HEAD"])
    # A page is written under docs/benchmarks by this script, so a run that writes one page and
    # then decodes the next finds the checkout dirty by its own output. Only the sources the
    # runtime and the harness are built from bear on what a figure means.
    dirty = run(["git", "-C", str(here), "status", "--porcelain", "--"] + list(SOURCE_PATHS))
    docs_dirty = run(["git", "-C", str(here), "status", "--porcelain", "--", ":/docs"])
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
    head = (rev + ("+dirty" if dirty else "")) if rev else "?"
    wheel_path, wheel_rev = wheel_build()
    return dict(
        date=time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        utter_revision=head,
        source_dirty=bool(dirty),
        docs_dirty=bool(docs_dirty),
        wheel_path=wheel_path,
        wheel_revision=wheel_rev,
        wheel_matches_head=same_revision(wheel_rev, rev, bool(dirty)),
        engines=versions,
        model=Path(args.model).name,
        cpu=cpu or platform.processor() or platform.machine(),
        python=platform.python_version(),
        block_ms=args.block_ms,
    )


def same_revision(wheel_rev: str | None, head: str | None, source_dirty: bool) -> bool:
    """Whether the binding was built from the sources as they stand. A dirty build never matches:
    it was compiled from edits, and nothing names which."""
    if wheel_rev is None or head is None:
        return False
    built_dirty = wheel_rev.endswith("-dirty")
    return wheel_rev.removesuffix("-dirty").startswith(head) and not built_dirty and not source_dirty


def wheel_build() -> tuple[str | None, str | None]:
    """The binding's file and the utter revision it was compiled from. A page's `git rev-parse`
    names the checkout the script was read from, which is not evidence about the runtime that
    decoded: the binding is a compiled artifact and can be any age."""
    try:
        import utterpy
    except Exception:
        return None, None
    rev = getattr(utterpy, "UTTER_REVISION", None)
    return getattr(utterpy, "__file__", None), rev


def write_page(path: str | Path, lines: Sequence[str]) -> str:
    """Sections and appended readings each bring their own blank lines; markdownlint wants one."""
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip("\n") + "\n"
    Path(path).write_text(text)
    return text


def provenance_line(prov: Mapping[str, Any]) -> str:
    """The header sentence every page opens with."""
    engines = ", ".join(f"{k} {v}" for k, v in prov["engines"].items())
    if prov.get("wheel_revision"):
        agree = (
            "the checkout's HEAD"
            if prov["wheel_matches_head"]
            else f"**not** the checkout as it stands ({prov['utter_revision']})"
        )
        wheel = f"binding `{prov['wheel_path']}` built from utter {prov['wheel_revision']}, {agree}"
    else:
        wheel = (
            f"binding `{prov['wheel_path']}`, which does not report the utter revision it was built "
            "from, so the revision above is the checkout's and not the runtime's"
        )
    docs = " Pages under `docs` were modified at the time of the run." if prov.get("docs_dirty") else ""
    return (
        f"Run {prov['date']}, utter {prov['utter_revision']}, {engines}, model {prov['model']}, "
        f"{prov['cpu']}, Python {prov['python']}. Decoded by the {wheel}.{docs}"
    )


def write_clip_records(path: str | Path, clips: Sequence[Clip], results: Mapping[str, Any]) -> None:
    """One line per clip with every engine's reading, so two runs can be diffed."""
    with open(path, "w") as f:
        for label, clip in clips:
            row = {"clip": str(clip), "label": label}
            for name, r in results.items():
                o = r["outcome"].get(str(clip))
                if o is not None:
                    row[name] = o
            f.write(json.dumps(row) + "\n")


def read_pcm(path: str | Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        return w.readframes(w.getnframes())


def energy_end(pcm: bytes, frame_ms: int = 10, drop_db: float = 20.0) -> float | None:
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
    def __init__(
        self,
        name: str,
        module: ModuleType,
        model_dir: str | Path,
        grammar: Sequence[str],
        unknown_cost: float | None = None,
    ) -> None:
        # `utterpy@300`: utterpy with a host endpoint bound at 300 ms of trailing silence;
        # `utterpy@300/4`: the same, vetoed while a reading extending the partial is within 4 nats
        self.name = name
        self.endpoint_bound: tuple[float, float | None] | None = None
        if "@" in name:
            parts = name.split("@", 1)[1].split("/")
            self.endpoint_bound = (float(parts[0]), float(parts[1]) if len(parts) > 1 else None)
        self.mod = module
        self.model = module.Model(str(model_dir))
        self.grammar = json.dumps(grammar)
        self.unknown_cost = unknown_cost
        # The wheel and the binding spell the vocabulary lookup differently.
        self.find = getattr(self.model, "FindWord", None) or getattr(self.model, "vosk_model_find_word", None)
        self.missing = [w for e in grammar for w in e.split() if not self.knows(w)]

    def knows(self, word: str) -> bool:
        return self.find is None or all(self.find(t) >= 0 for t in word.split())

    def new(self, alternatives: int = 0) -> Any:
        if self.unknown_cost is not None:
            rec = self.mod.KaldiRecognizer(self.model, RATE, self.grammar, self.unknown_cost)
        else:
            rec = self.mod.KaldiRecognizer(self.model, RATE, self.grammar)
        rec.SetWords(True)
        if self.endpoint_bound is not None:
            rec.SetEndpointBound(*self.endpoint_bound)
        if alternatives and hasattr(rec, "SetPartialAlternatives"):
            rec.SetPartialAlternatives(alternatives)
        return rec


def feed(rec: Any, pcm: bytes, block_ms: int, on_partial: OnPartial | None = None) -> list[Json]:
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


def final_words(finals: Iterable[Mapping[str, Any]]) -> list[str]:
    return [w for f in finals for w in words_of(f.get("text", ""))]


def finals_with_a_word(finals: Iterable[Mapping[str, Any]]) -> int:
    return sum(1 for f in finals if words_of(f.get("text", "")))


def run_clip(
    engine: Engine, pcm: bytes, block_ms: int, label: str
) -> tuple[list[str], float | None, str | None, float | None, tuple[float, float]]:
    """Returns (final words, first-appearance second of the label or None, first word shown or
    None, milliseconds from that first showing to the first partial that led with a different
    word or None, (construction seconds, decode seconds))."""
    state: dict[str, Any] = {"first": None, "first_word": None, "shown_at": None, "revised_ms": None}

    def on_partial(fed: int, p: Json) -> None:
        partial = words_of(p.get("partial", ""))
        if partial and state["first_word"] is None:
            state["first_word"] = partial[0]
            state["shown_at"] = fed
        elif partial and state["revised_ms"] is None and partial[0] != state["first_word"]:
            state["revised_ms"] = 1000.0 * (fed - state["shown_at"]) / RATE
        if state["first"] is None and partial[:1] == [label]:
            state["first"] = fed / RATE

    t0 = time.perf_counter()
    rec = engine.new()
    t1 = time.perf_counter()
    finals = feed(rec, pcm, block_ms, on_partial)
    t2 = time.perf_counter()
    return final_words(finals), state["first"], state["first_word"], state["revised_ms"], (t1 - t0, t2 - t1)


def agreement_table(clips: Sequence[Clip], results: Mapping[str, Any], lines: list[str], report: Json) -> None:
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


def significance_table(clips: Sequence[Clip], results: Mapping[str, Any], lines: list[str], report: Json) -> None:
    """The engines decode the same clips, so the comparison is paired: only the clips they
    disagree on carry information. Aggregate counts alone cannot say whether a gap is real."""
    names = list(results)
    if len(names) != 2:
        return
    a, b = names
    oa, ob = results[a]["outcome"], results[b]["outcome"]
    out: dict[str, Json] = {}
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
    by_word: defaultdict[str, list[Path]] = defaultdict(list)
    for label, clip in clips:
        by_word[label].append(clip)
    for w in DATASET_WORDS:
        rows.append((w, by_word[w]))
    counted: list[tuple[str, int, int, float]] = []
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
    for (scope, only_a, only_b, p), q in sorted(zip(words, adjusted, strict=True), key=lambda z: z[0][3]):
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

    # A directional win is either the loser reading a different word or reading nothing; only the
    # second is a silence divergence, and the two engines' silence behaviour is the contract.
    # [[rr:TD-8#The decoder reads silence as libvosk does]]
    split = {a: dict(word=0, silence=0), b: dict(word=0, silence=0)}
    for _, path in clips:
        wa = oa.get(str(path), {}).get("words")
        wb = ob.get(str(path), {}).get("words")
        ca = oa.get(str(path), {}).get("correct")
        cb = ob.get(str(path), {}).get("correct")
        if ca and not cb:
            split[a]["silence" if not wb else "word"] += 1
        elif cb and not ca:
            split[b]["silence" if not wa else "word"] += 1
    empty = report["agreement"]["empty"]
    lines.append("")
    lines.append(
        "When one engine reads a clip the other misses, the miss is almost always a different "
        "word, not silence. The same paired wins, split by what the losing engine returned:"
    )
    lines.append("")
    lines.append("| direction | other read a different word | other read silence |")
    lines.append("|---|---|---|")
    lines.append(f"| {a} only | {split[a]['word']} | {split[a]['silence']} |")
    lines.append(f"| {b} only | {split[b]['word']} | {split[b]['silence']} |")
    lines.append("")
    lines.append(
        f"Silence is {split[a]['silence']} clips one way and {split[b]['silence']} the other, so "
        "neither engine falls silent where the other reads a word more than the reverse; the "
        f"difference is word against word, the near ties `[[rr:TD-6]]` leaves in the acoustics. "
        f"Over the whole corpus the two read almost the same number of clips as silence, "
        f"{empty[a]} and {empty[b]}."
    )
    for name in (a, b):
        out["all words"][name + "_split"] = split[name]
    report["significance"] = out


def full_grammar_pass(
    modules: Mapping[str, ModuleType],
    model: str | Path,
    clips: Sequence[Clip],
    block_ms: int,
    grammar: Sequence[str],
    lines: list[str],
    report: Json,
) -> None:
    results: dict[str, Json] = {}
    ref_end: dict[Path, float | None] = {}
    note(f"full grammar: {len(clips)} clips x {len(modules)} engines")
    for name, mod in modules.items():
        eng = Engine(name, mod, model, grammar)
        note(f"  {name}")
        if eng.missing:
            lines.append(f"- {name}: words absent from the model's table: {eng.missing}")
        correct = never = shown = changed = 0
        revised = flushed = 0
        rev_lag: list[float] = []
        conf: Counter[tuple[str, str]] = Counter()
        per_word: Counter[str] = Counter()
        per_word_ok: Counter[str] = Counter()
        lat: list[float] = []
        outcome: dict[str, Json] = {}
        ctors: list[float] = []
        ctor = decode = audio = 0.0
        for done, (label, path) in enumerate(clips):
            if done and done % 2500 == 0:
                note(f"    {done} / {len(clips)}")
            pcm = read_pcm(path)
            audio += len(pcm) / 2 / RATE
            if path not in ref_end:
                ref_end[path] = energy_end(pcm)
            end = ref_end[path]
            words, first, first_word, revised_ms, (t_new, t_decode) = run_clip(eng, pcm, block_ms, label)
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
                if revised_ms is not None:
                    revised += 1
                    rev_lag.append(revised_ms)
                elif words[:1] != [first_word]:
                    flushed += 1
            if first is None:
                never += 1
            elif end is not None:
                lat.append((first - end) * 1000.0)
            outcome[str(path)] = dict(
                words=words,
                correct=words == [label],
                first=first,
                first_word=first_word,
                revised_ms=revised_ms,
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
            first_revised=revised,
            first_flushed=flushed,
            rev_lag_ms_p50=quantile(rev_lag, 0.5),
            rev_lag_ms_p90=quantile(rev_lag, 0.9),
            rev_lag_ms_min=min(rev_lag) if rev_lag else None,
            rev_lag_ms_max=max(rev_lag) if rev_lag else None,
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
    lines.append(
        "The last column counts the first word shown against the final, which a host feels as two "
        "different things. A partial overtaken by another partial is a flicker on the screen; a "
        "partial that stands through every later partial and is replaced only by the final is not a "
        "flicker at all, and no amount of waiting on partials would have caught it. The two columns "
        "below do not sum to that one: a partial can be overtaken and then come back, which the last "
        "column does not count and the first of these does. The lag is the gap between the two "
        "partials, and it decides whether a flicker is visible or too brief to see."
    )
    lines.append("")
    lines.append(
        "| engine | first word shown | overtaken by a later partial | stood, then changed by the final | revision lag p50 / p90 | lag min / max |"
    )
    lines.append("|---|---|---|---|---|---|")
    for name, r in results.items():
        lag = f"{r['rev_lag_ms_p50']:.0f} / {r['rev_lag_ms_p90']:.0f} ms" if r["first_revised"] else "-"
        span = f"{r['rev_lag_ms_min']:.0f} / {r['rev_lag_ms_max']:.0f} ms" if r["first_revised"] else "-"
        lines.append(
            f"| {name} | {r['first_shown']} | {r['first_revised']} ({pct(r['first_revised'], r['first_shown']):.1f}%) | "
            f"{r['first_flushed']} ({pct(r['first_flushed'], r['first_shown']):.1f}%) | {lag} | {span} |"
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


def trace_clip(engine: Engine, pcm: bytes, block_ms: int, alternatives: int = 0) -> Trace:
    """Every partial the engine showed with its readings, in order, and the final. The trace is
    kept whole because a final can be stable while the path to it is not, and the path is what a
    host renders; it is compared as a digest because the whole split's traces are too large to
    carry between processes."""
    partials = []

    def keep(fed: int, p: Json) -> None:
        partials.append([p.get("partial", ""), p.get("partial_alternatives") or []])

    finals = feed(engine.new(alternatives=alternatives), pcm, block_ms, keep)
    digest = hashlib.sha256(json.dumps(partials, sort_keys=True).encode()).hexdigest()
    return [digest, final_words(finals)]


def determinism_traces(
    modules: Mapping[str, ModuleType],
    model: str | Path,
    clips: Sequence[tuple[str | None, Path]],
    block_ms: int,
    grammar: Sequence[str],
    alternatives: int = 0,
) -> dict[str, list[Trace]]:
    """One pass over the clips per engine, a recognizer each, as a host would run them."""
    out = {}
    for name, mod in modules.items():
        eng = Engine(name, mod, model, grammar)
        out[name] = [trace_clip(eng, read_pcm(path), block_ms, alternatives) for _, path in clips]
    return out


def trace_worker(job_path: str) -> None:
    """The fresh process's half of the determinism pass: the same clips, a process of its own,
    the traces back over stdout."""
    job = json.loads(Path(job_path).read_text())
    modules = {}
    for name in job["engines"]:
        modules[name] = __import__(name.split("@")[0])
        if name == "vosk":
            modules[name].SetLogLevel(-1)
    clips = [(None, Path(p)) for p in job["clips"]]
    traces = determinism_traces(
        modules, job["model"], clips, job["block_ms"], job["grammar"], job.get("alternatives", 0)
    )
    json.dump(traces, sys.stdout)


def fresh_process_traces(
    model: str | Path,
    clips: Sequence[Clip],
    block_ms: int,
    grammar: Sequence[str],
    engines: Iterable[str],
    alternatives: int = 0,
) -> dict[str, list[Trace]]:
    """The same decode in a process started for it. A repeat inside one process cannot see a
    per-process seed: it draws the same one. Rust's default hasher is seeded once per process, so
    a hash map's iteration order — and with it which tokens a narrowing cutoff reaches first and
    prunes — is fixed for a run and free to move between runs. Only a second process can show it."""
    job = dict(
        model=str(model),
        block_ms=block_ms,
        grammar=grammar,
        engines=list(engines),
        alternatives=alternatives,
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
    return cast("dict[str, list[Trace]]", json.loads(proc.stdout))


def determinism_pass(
    modules: Mapping[str, ModuleType],
    model: str | Path,
    clips: Sequence[Clip],
    block_ms: int,
    grammar: Sequence[str],
    lines: list[str],
    report: Json,
    count: int,
    alternatives: int,
) -> None:
    """Whether the same bytes give the same reading twice. Nothing else on this page would
    notice if they did not: every other pass decodes each clip once, so a reading that moves
    between runs is invisible to it and shows up only as noise in next week's comparison.

    Both passes are processes started for the purpose. A repeat inside one process cannot see a
    per-process seed: it draws the same one, so a reading that depends on a hash order is
    perfectly stable within a run and moves only between runs. The whole split is compared
    because a tie broken by chance on one clip in two thousand is invisible in a sample of two
    hundred; `--determinism-clips N` takes a sample of N when a run has to be cheap."""
    chosen = clips
    if count > 0 and count < len(clips):
        step = max(1, len(clips) // count)
        chosen = clips[::step][:count]
    note(f"determinism: {len(chosen)} clips x {len(modules)} engines, in two fresh processes")
    first = fresh_process_traces(model, chosen, block_ms, grammar, modules, alternatives)
    second = fresh_process_traces(model, chosen, block_ms, grammar, modules, alternatives)
    lines.append("## Determinism")
    lines.append("")
    lines.append(
        f"{len(chosen)} clips decoded in two processes started for the purpose, comparing every "
        f"partial with its {alternatives} readings, block by block, and the final. A process is "
        "what matters: a hash seed drawn once per process gives a map the same iteration order "
        "for every repeat within a run, so a reading that depends on it is perfectly stable until "
        "the next run. The partial column compares a digest of the whole trace; any clip whose "
        "final moves is named in the JSON."
    )
    lines.append("")
    lines.append("| engine | clips | partials and readings | finals |")
    lines.append("|---|---|---|---|")
    out: dict[str, Json] = {}
    for name in modules:
        moved: Json = {"partials": [], "finals": []}
        a, b = first.get(name) or [], second.get(name) or []
        for (_, path), x, y in zip(chosen, a, b, strict=False):  # a failed process leaves no traces
            if x[0] != y[0]:
                moved["partials"].append(str(path))
            if x[1] != y[1]:
                moved["finals"].append(str(path))
        moved["compared"] = min(len(a), len(b))
        out[name] = {"fresh_processes": moved}
        cells = ["n/a", "n/a"] if not moved["compared"] else [len(moved["partials"]), len(moved["finals"])]
        lines.append(f"| {name} | {moved['compared']} | " + " | ".join(str(c) for c in cells) + " |")
    lines.append("")
    unstable = [
        f"{name}: {len(m['partials'])} partial traces and {len(m['finals'])} finals"
        for name, row in out.items()
        for m in row.values()
        if m["partials"] or m["finals"]
    ]
    lines.append(
        "Every clip read the same way both times."
        if not unstable
        else "**Not reproducible.** " + "; ".join(unstable) + "."
    )
    lines.append("")
    report["determinism"] = out


def endpoint_run(engine: Engine, pcm: bytes, block_ms: int) -> tuple[float | None, list[str]]:
    """The fed-audio second at which the engine first closes a segment of its own accord, or None
    if it never does within the audio given, and every word it reports over the whole of it. The
    audio runs on past the first close because a bound that fires early leaves the rest of the
    clip to a second segment, and a word cut in two shows up only in the words."""
    rec = engine.new()
    block = RATE * block_ms // 1000 * 2
    fed = 0
    at = None
    finals = []
    for i in range(0, len(pcm), block):
        chunk = pcm[i : i + block]
        fed += len(chunk) // 2
        if rec.AcceptWaveform(chunk):
            finals.append(json.loads(rec.Result()))
            if at is None:
                at = fed / RATE
    finals.append(json.loads(rec.FinalResult()))
    return at, final_words(finals)


def bound_engines(modules: Mapping[str, ModuleType], names: Iterable[str]) -> list[tuple[str, ModuleType]]:
    """`utterpy@300` and the rest run on the module their name is spelled over."""
    return [(name, modules[name.split("@")[0]]) for name in names]


def endpoint_pass(
    modules: Mapping[str, ModuleType],
    model: str | Path,
    clips: Sequence[Clip],
    block_ms: int,
    grammar: Sequence[str],
    lines: list[str],
    report: Json,
    count: int,
    pad_ms: int,
    bound: Sequence[str] = (),
) -> None:
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
    if bound:
        lines.append(
            "The rows spelled `utterpy@MS` and `utterpy@MS/NATS` carry a host endpoint bound of "
            "their own beside the model's rules, `[[rr:TD-8#A host may add one endpoint bound of "
            "its own, off by default]]`: a final once the trailing silence reaches MS, and where a "
            "margin follows, no final while a reading extending the partial is within that many "
            "nats of the leader."
        )
        lines.append("")
        lines.append(
            "A bound shorter than one decoding advance does not thereby fire at the first advance "
            "that sees silence. The rule is read once per advance, 240 ms at this block size, but "
            "the trailing silence it compares against is counted in the model's own subsampled "
            "frames of 30 ms (`[[rr:endpoint_detected]]`), and at the first advance holding any "
            "trailing silence that span already stands anywhere from 30 ms to a full advance: "
            "100 ms fires at that advance on the clips where it does and one advance later on the "
            "rest, so it neither coincides with every shorter bound nor with 300."
        )
        lines.append("")
    lines.append("| engine | endpointed | after the clip's energy end p50 / p90 / p99 | never within the silence |")
    lines.append("|---|---|---|---|")
    out: dict[str, Json] = {}
    at_by_engine: dict[str, dict[str, float | None]] = {}
    words_by_engine: dict[str, dict[str, list[str]]] = {}
    for name, mod in list(modules.items()) + bound_engines(modules, bound):
        eng = Engine(name, mod, model, grammar)
        delays: list[float] = []
        never = 0
        at_by_engine[name] = {}
        words_by_engine[name] = {}
        for _, path in chosen:
            pcm = read_pcm(path)
            end = energy_end(pcm)
            at, words = endpoint_run(eng, pcm + pad, block_ms)
            at_by_engine[name][str(path)] = at
            words_by_engine[name][str(path)] = words
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
    if bound:
        bound_words_section(chosen, words_by_engine, bound, lines, report)


def bound_words_section(
    chosen: Sequence[Clip],
    words_by_engine: Mapping[str, Mapping[str, list[str]]],
    bound: Sequence[str],
    lines: list[str],
    report: Json,
) -> None:
    """What the host's bound costs in words on the clips it was timed over. A bound that fires
    inside a word leaves the rest of it to the next segment, so the loss is not only a word gone:
    it is also a word arriving twice."""
    base = "utterpy"
    lines.append("### What the bound costs in words")
    lines.append("")
    lines.append(
        f"The same {len(chosen)} padded clips, every word each engine finally reported, paired "
        f"against stock `{base}` clip by clip. *Right* is the clip's label and nothing else; the "
        "three columns after it are how a reading that differs from the stock engine's differs. "
        "The test is exact McNemar over the clips where exactly one of the two was right."
    )
    lines.append("")
    lines.append(
        f"| engine | right | differs from {base} | said nothing | said more words | said another word | "
        f"{base} right only / bound right only | exact p |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    out: dict[str, Json] = {}
    for name in bound:
        right = differs = empty = longer = other = b = c = 0
        for label, path in chosen:
            key = str(path)
            mine, theirs = words_by_engine[name][key], words_by_engine[base][key]
            ok, ok_base = mine == [label], theirs == [label]
            right += ok
            b += ok_base and not ok
            c += ok and not ok_base
            if mine != theirs:
                differs += 1
                if not mine:
                    empty += 1
                elif len(mine) > len(theirs):
                    longer += 1
                else:
                    other += 1
        p = mcnemar(b, c)
        out[name] = dict(
            right=right,
            base_right=sum(1 for label, path in chosen if words_by_engine[base][str(path)] == [label]),
            differs=differs,
            said_nothing=empty,
            said_more=longer,
            said_another=other,
            base_only=b,
            bound_only=c,
            p=p,
        )
        lines.append(
            f"| {name} | {right} / {len(chosen)} ({pct(right, len(chosen)):.2f}%) | {differs} | {empty} | "
            f"{longer} | {other} | {b} / {c} | {p:.3g} |"
        )
    lines.append("")
    lines.append(
        f"Stock `{base}` was right on {out[bound[0]]['base_right']} of the {len(chosen)} under the "
        "same grammar and padding, which is the row every bound is paired against."
    )
    lines.append("")
    worst = max((o["differs"] for o in out.values()), default=0)
    if worst:
        lines.append(
            f"The bounds differ from stock `{base}` on up to {worst} of the {len(chosen)}; a word "
            "lost shows in *said nothing* and a word decoded twice in *said more words*."
        )
    else:
        lines.append(
            "Nothing here separates one bound from another or from the stock rules, the bounds "
            "below 300 ms included: a clip carries one word and every bound fires in the silence "
            "after it, so no word is lost, none is decoded twice, and the paired test has nothing "
            "to weigh. Where a short bound can fall inside an utterance is the built streams, and "
            "the states page measures it there."
        )
    lines.append("")
    report["endpoint"]["bound_words"] = out


def block_size_pass(
    modules: Mapping[str, ModuleType],
    model: str | Path,
    clips: Sequence[Clip],
    grammar: Sequence[str],
    lines: list[str],
    report: Json,
    sizes: Sequence[int],
    count: int,
) -> None:
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
    out: dict[int, dict[str, Json]] = {}
    finest = min(sizes)
    at_finest: dict[str, list[float | None]] = {}
    for ms in sizes:
        out[ms] = {}
        for name, mod in modules.items():
            eng = Engine(name, mod, model, grammar)
            ok = 0
            lat = []
            decode = 0.0
            firsts: list[float | None] = []
            for (label, pcm), end in zip(pcms, ends, strict=True):
                words, first, _, _, (_, t_dec) = run_clip(eng, pcm, ms, label)
                ok += words == [label]
                decode += t_dec
                firsts.append(first)
                if first is not None and end is not None:
                    lat.append((first - end) * 1000.0)
            if ms == finest:
                at_finest[name] = firsts
            # Empty until the finest size has run.
            both = [
                (x, y) for x, y in zip(firsts, at_finest.get(name, []), strict=False) if x is not None and y is not None
            ]
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


def twelve_class_pass(
    modules: Mapping[str, ModuleType],
    model: str | Path,
    clips: Sequence[Clip],
    noise: Sequence[Path],
    block_ms: int,
    lines: list[str],
    report: Json,
) -> None:
    lines.append("## Twelve-class: ten commands plus the unknown-word symbol")
    lines.append("")
    lines.append(
        "| engine | commands correct | fillers and digits read as unknown | fillers and digits read as a command | noise minutes | silence finals per minute |"
    )
    lines.append("|---|---|---|---|---|---|")
    twelve: dict[str, Json] = {}
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
        noise_samples = silence_finals = 0
        for path in noise:
            pcm = read_pcm(path)
            # [[rr:TD-8#Measurements count a word whose own span carries no speech]]
            finals = feed(eng.new(), pcm, block_ms)
            noise_samples += len(pcm) // 2
            silence_finals += finals_with_a_word(finals)
        noise_minutes = noise_samples / RATE / 60.0
        twelve[name] = dict(
            cmd_ok=cmd_ok,
            cmd_total=cmd_total,
            other_unk=other_unk,
            other_cmd=other_cmd,
            other_total=other_total,
            noise_minutes=noise_minutes,
            silence_finals=silence_finals,
        )
        lines.append(
            f"| {name} | {cmd_ok} / {cmd_total} ({pct(cmd_ok, cmd_total):.2f}%) | {other_unk} / {other_total} ({pct(other_unk, other_total):.2f}%) | "
            f"{other_cmd} ({pct(other_cmd, other_total):.2f}%) | {noise_minutes:.1f} | "
            f"{silence_finals / noise_minutes if noise_minutes else float('nan'):.1f} |"
        )
    report["twelve"] = twelve


def noise_pass(
    modules: Mapping[str, ModuleType],
    model: str | Path,
    noise: Sequence[Path],
    block_ms: int,
    grammar: Sequence[str],
    lines: list[str],
    report: Json,
    bound: Sequence[str] = (),
) -> None:
    lines.append("")
    lines.append("## Background noise under the full grammar")
    lines.append("")
    lines.append(
        "| engine | grammar | noise minutes | partial blocks | blocks with a word at rank 0 | silence finals per minute | word blocks with a wordless reading among the rivals |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    out: dict[str, Json] = {}
    engines = list(modules.items()) + bound_engines(modules, bound)
    note(f"background noise: {len(engines)} engines x 2 grammars over the noise recordings")
    for name, mod in engines:
        for variant, g in [("full", list(grammar)), ("full + [unk]", [*grammar, "[unk]"])]:
            note(f"  {name}, {variant}")
            eng = Engine(name, mod, model, g)
            blocks = word_blocks = sil_rival = silence_finals = samples = 0
            for path in noise:
                pcm = read_pcm(path)
                counts = {"blocks": 0, "words": 0, "rival": 0}

                def on_partial(fed: int, p: Json, counts: dict[str, int] = counts) -> None:
                    counts["blocks"] += 1
                    if words_of(p.get("partial", "")):
                        counts["words"] += 1
                        alts = p.get("partial_alternatives") or []
                        if any(a["text"] in WORDLESS for a in alts[1:]):
                            counts["rival"] += 1

                # [[rr:TD-8#Measurements count a word whose own span carries no speech]]
                finals = feed(eng.new(alternatives=4), pcm, block_ms, on_partial)
                blocks += counts["blocks"]
                word_blocks += counts["words"]
                sil_rival += counts["rival"]
                silence_finals += finals_with_a_word(finals)
                samples += len(pcm) // 2
            minutes = samples / RATE / 60.0
            out[f"{name}/{variant}"] = dict(
                minutes=minutes,
                blocks=blocks,
                word_blocks=word_blocks,
                sil_rival=sil_rival,
                silence_finals=silence_finals,
            )
            rival = f"{sil_rival} / {word_blocks}" if name != "vosk" else "no alternatives"
            lines.append(
                f"| {name} | {variant} | {minutes:.1f} | {blocks} | {word_blocks} ({pct(word_blocks, blocks):.2f}%) | "
                f"{silence_finals / minutes if minutes else float('nan'):.1f} | {rival} |"
            )
    report["noise"] = out


def mix_at_snr(pcm: bytes, noise: bytes, offset: int, snr_db: float) -> bytes:
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


def snr_pass(
    modules: Mapping[str, ModuleType],
    model: str | Path,
    clips: Sequence[Clip],
    noise_paths: Sequence[Path],
    block_ms: int,
    grammar: Sequence[str],
    lines: list[str],
    report: Json,
    levels: Sequence[int],
    count: int,
) -> None:
    """Accuracy against noise. Clean one-second clips are the most forgiving regime there is;
    a front-end difference shows itself here first. Both engines are fed the same mixed bytes."""
    step = max(1, len(clips) // count)
    chosen = clips[::step][:count]
    noise = b"".join(read_pcm(p) for p in noise_paths)
    mixed: dict[int, list[tuple[str, bytes]]] = {}
    for level in levels:
        mixed[level] = [
            (label, mix_at_snr(read_pcm(path), noise, (i * 7919) % max(1, len(noise) // 2), level))
            for i, (label, path) in enumerate(chosen)
        ]
    out: dict[str, dict[str, int]] = {}
    outcome: dict[str, dict[int | str, list[bool]]] = {}
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
        row: dict[int | str, int] = {}
        outcome[name] = {}
        for level in levels:
            ok = 0
            marks: list[bool] = []
            for label, pcm in mixed[level]:
                words, _, _, _, _ = run_clip(eng, pcm, block_ms, label)
                got = words == [label]
                marks.append(got)
                ok += got
            row[level] = ok
            outcome[name][level] = marks
        clean = 0
        marks = []
        for label, path in chosen:
            words, _, _, _, _ = run_clip(eng, read_pcm(path), block_ms, label)
            got = words == [label]
            marks.append(got)
            clean += got
        row["clean"] = clean
        outcome[name]["clean"] = marks
        out[name] = {str(k): v for k, v in row.items()}
        cells = " | ".join(f"{row[level]} ({pct(row[level], len(chosen)):.1f}%)" for level in levels)
        lines.append(f"| {name} | {cells} | {clean} ({pct(clean, len(chosen)):.1f}%) |")
    lines.append("")
    # [[rr:docs/benchmarks/README.md#Benchmarks]]
    paired: dict[str, Json] = {}
    names = list(modules)
    if len(names) == 2:
        a, b = names
        lines.append(f"Paired at each level, as above: `{a} only`, `{b} only`, and the exact McNemar p.")
        lines.append("")
        lines.append("| level | " + f"{a} only | {b} only | p |")
        lines.append("|---|---|---|---|")
        scopes: list[int | str] = [*levels, "clean"]
        for scope in scopes:
            ma, mb = outcome[a][scope], outcome[b][scope]
            only_a = sum(1 for x, y in zip(ma, mb, strict=True) if x and not y)
            only_b = sum(1 for x, y in zip(ma, mb, strict=True) if y and not x)
            pv = mcnemar(only_a, only_b)
            paired[str(scope)] = dict(only_a=only_a, only_b=only_b, p=pv)
            label = f"{scope} dB" if scope != "clean" else "clean"
            lines.append(f"| {label} | {only_a} | {only_b} | {pv:.3f} |")
        lines.append("")
    report["snr"] = dict(clips=len(chosen), levels=levels, results=out, paired=paired)


def filler_pool(modules: Mapping[str, ModuleType], model: str | Path) -> list[str]:
    """Distractors this model's table knows, in sweep order: what a grammar is padded to size
    with, so every clip stays decidable at every size."""
    probe = Engine("probe", next(iter(modules.values())), model, DATASET_WORDS)
    pool = []
    for w in LETTERS + NATO + COLOURS + SWEEP_EXTRA:
        if w in DATASET_WORDS or w in pool:
            continue
        if probe.knows(w):
            pool.append(w)
    return pool


def grammar_size_pass(
    modules: Mapping[str, ModuleType],
    model: str | Path,
    clips: Sequence[Clip],
    block_ms: int,
    lines: list[str],
    report: Json,
    sizes: Sequence[int],
    count: int,
) -> None:
    """How the figures move with the grammar, the one dial a host turns. The dataset's own 35
    words are always in, so every clip stays decidable; the rest is filler."""
    step = max(1, len(clips) // count)
    chosen = clips[::step][:count]
    pcms = [(label, read_pcm(path)) for label, path in chosen]
    audio = sum(len(p) / 2 / RATE for _, p in pcms)
    pool = filler_pool(modules, model)
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
                words, _, _, _, (t_new, t_dec) = run_clip(eng, pcm, block_ms, label)
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


def steady_state_pass(
    modules: Mapping[str, ModuleType],
    model: str | Path,
    clips: Sequence[Clip],
    block_ms: int,
    grammar: Sequence[str],
    lines: list[str],
    report: Json,
    seconds: int,
) -> None:
    """Compute on continuous audio: one recognizer over the clips joined end to end, which is
    how a host runs, rather than the fixed cost of a clip at a time."""
    if seconds <= 0:
        return
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


def preflight(
    modules: Mapping[str, ModuleType],
    model: str | Path,
    grammar: Sequence[str],
    block_ms: int,
    clips: Sequence[Clip],
    lines: list[str],
    report: Json,
) -> None:
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


WORDLESS_BASE_FLOOR = -50.0


def floor_of(samples: Sequence[int], window_ms: int = 100, hop_ms: int = 50, q: float = 0.05) -> float:
    """The floor the runtime reports, computed here over a whole recording so a sweep can be
    built at a named level. [[rr:TD-8#The runtime reports the floor]]"""
    n = RATE * window_ms // 1000
    hop = RATE * hop_ms // 1000
    levels = []
    for i in range(0, len(samples) - n + 1, hop):
        acc = 0
        for v in samples[i : i + n]:
            acc += v * v
        rms = math.sqrt(acc / n)
        levels.append(20 * math.log10(rms / 32768.0) if rms > 0 else -999.0)
    return quantile(levels, q) if levels else -999.0


def scaled_to_floor(pcm: bytes, target_dbfs: float) -> bytes:
    """The same recording with its floor moved to `target_dbfs`, so a sweep over floor level is a
    sweep over one property of one piece of audio. Saturates rather than wraps, as an ADC would."""
    a = array.array("h")
    a.frombytes(pcm)
    have = floor_of(a)
    if have <= -999.0:
        return pcm
    gain = 10.0 ** ((target_dbfs - have) / 20.0)
    out = array.array("h", bytes(2 * len(a)))
    for i, v in enumerate(a):
        x = int(round(v * gain))
        out[i] = -32768 if x < -32768 else (32767 if x > 32767 else x)
    return out.tobytes()


def wordless_rooms(noise: Sequence[Path], level: float | None, cache: RoomCache) -> list[tuple[str, bytes]]:
    """The background recordings at one floor level; `level` None is digital silence of the same
    length, which the floor's percentile treats apart from a room."""
    key = "silence" if level is None else f"{level:g}"
    if key not in cache:
        raw = [(p.name, read_pcm(p)) for p in noise]
        cache[key] = [(n, bytes(len(pcm)) if level is None else scaled_to_floor(pcm, level)) for n, pcm in raw]
    return cache[key]


def wordless_takes(clips: Sequence[Clip], per_word: int = 3) -> list[Take]:
    """Clips chosen by the property the after-a-word condition turns on, the level of the word
    that precedes the silence: the quietest, the median and the loudest clip of each dataset
    word, never the first file of each."""
    by_word = defaultdict(list)
    for label, path in clips:
        by_word[label].append(path)
    out = []
    for label in sorted(by_word):
        rows = []
        for path in by_word[label]:
            a = array.array("h")
            a.frombytes(read_pcm(path))
            peak = max((abs(v) for v in a), default=0)
            rows.append((20 * math.log10(peak / 32768.0) if peak else -999.0, str(path), label, path))
        rows.sort()
        for i in sorted({0, len(rows) // 2, len(rows) - 1})[:per_word]:
            out.append((rows[i][2], rows[i][3], rows[i][0]))
    return out


def wordless_samples(
    kind: str,
    noise: Sequence[Path],
    takes: Sequence[Take],
    level: float | None,
    seed: int,
    tail_ms: tuple[float, ...],
    cache: RoomCache,
) -> list[WordlessSample]:
    """The audio a wordless pass is measured on. `cold` is each recording alone; `after a word` is
    one spoken clip followed by a stretch of recording, so the phantom counted there is a word
    beyond the one that was said."""
    rooms = wordless_rooms(noise, level, cache)
    if kind == "cold":
        return [WordlessSample(key=f"cold/{n}", pcm=p, speech_end=0, spoken=[], samples=len(p) // 2) for n, p in rooms]
    rng = random.Random(seed)
    out: list[WordlessSample] = []
    for i, (label, path, _peak) in enumerate(takes):
        clip = read_pcm(path)
        end = energy_end(clip)
        name, room = rooms[i % len(rooms)]
        n = int(rng.uniform(*tail_ms) * RATE / 1000)
        start = rng.randrange(max(1, len(room) // 2 - n))
        pcm = clip + room[2 * start : 2 * (start + n)]
        out.append(
            WordlessSample(
                key=f"after/{path.parent.name}/{path.name}",
                pcm=pcm,
                speech_end=int((end if end is not None else len(clip) / 2 / RATE) * RATE),
                spoken=[label],
                samples=len(pcm) // 2,
                source=name,
            )
        )
    return out


def wordless_run(eng: Engine, samples: Sequence[WordlessSample], block_ms: int, alternatives: int = 4) -> Json:
    """One engine over one sample set, counting only what falls after the audio's own speech. A
    phantom is rank 0 holding more words than were spoken so far in the segment: a word still held
    from the clip is not one, and neither is a misrecognition of it, which is an accuracy figure
    and is measured elsewhere. A final ends a segment and what was spoken resets with it, the
    decoder's history going too."""
    tot: Counter[str] = Counter()
    longest = 0
    gate: list[GateRow] = []
    per_sample: list[Json] = []
    block = RATE * block_ms // 1000 * 2
    for s in samples:
        # Asking for the words under a partial changes which partial the stock wheel shows, to a
        # lattice one that trails the audio, so the census is read off the plain partial on both
        # engines. The gate below needs only what a final carries.
        rec = eng.new(alternatives=alternatives)
        spoken = list(s["spoken"])
        fed = run = 0
        row: Counter[str] = Counter()
        for i in range(0, len(s["pcm"]), block):
            chunk = s["pcm"][i : i + block]
            fed += len(chunk) // 2
            if rec.AcceptWaveform(chunk):
                spoken = wordless_final(json.loads(rec.Result()), spoken, fed, s, row, gate)
                run = 0
                continue
            if fed <= s["speech_end"]:
                continue
            p = json.loads(rec.PartialResult())
            row["blocks"] += 1
            if len(words_of(p.get("partial", ""))) <= len(spoken):
                run = 0
                continue
            row["word_blocks"] += 1
            run += 1
            longest = max(longest, run)
            alts = p.get("partial_alternatives") or []
            if any(len(words_of(a["text"])) > len(spoken) for a in alts[1:]):
                row["rival_blocks"] += 1
        wordless_final(json.loads(rec.FinalResult()), spoken, fed, s, row, gate)
        row["nonspeech"] = max(0, s["samples"] - s["speech_end"])
        tot.update(row)
        per_sample.append(dict(key=s["key"], word_blocks=row["word_blocks"], phantom_finals=row["phantom_finals"]))
    return dict(
        blocks=tot["blocks"],
        word_blocks=tot["word_blocks"],
        rival_blocks=tot["rival_blocks"],
        finals=tot["finals"],
        phantom_finals=tot["phantom_finals"],
        longest_run=longest,
        minutes=tot["nonspeech"] / RATE / 60.0,
        gate=gate,
        per_sample=per_sample,
    )


def wordless_final(
    f: Mapping[str, Any],
    spoken: Sequence[str],
    fed: int,
    s: WordlessSample,
    row: Counter[str],
    gate: list[GateRow],
) -> list[str]:
    """One final that closed after the audio's speech: whether it carries a word nobody said, and
    the evidence TD-8 gives a host to judge that itself. Returns what is spoken from here on."""
    if fed <= s["speech_end"]:
        return list(spoken)
    words = words_of(f.get("text", ""))
    row["finals"] += 1
    if len(words) > len(spoken):
        row["phantom_finals"] += 1
        floor = f.get("floor_dbfs")
        said = [e for e in (f.get("result") or []) if not e["word"].startswith("[")]
        for e in said[len(spoken) :]:
            if floor is not None and "energy_dbfs" in e:
                gate.append(GateRow(key=s["key"], energy=e["energy_dbfs"], floor=floor))
    return []


def wordless_header(lines: list[str], what: str) -> None:
    lines.append("")
    lines.append(
        f"| {what} | condition | engine | non-speech minutes | blocks | word at rank 0 /min | "
        "word among the rivals /min | finals with a word /min | longest run of blocks |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")


def wordless_row(name: str, condition: str, engine: str, tot: Mapping[str, Any]) -> str:
    m = tot["minutes"]
    rival = "no alternatives" if engine == "vosk" else f"{tot['rival_blocks'] / m if m else float('nan'):.1f}"
    return (
        f"| {name} | {condition} | {engine} | {m:.1f} | {tot['blocks']} | "
        f"{tot['word_blocks'] / m if m else float('nan'):.2f} | {rival} | "
        f"{tot['phantom_finals'] / m if m else float('nan'):.2f} | {tot['longest_run']} |"
    )


def engine_takes_cost(eng: Engine, alternatives: int) -> bool:
    """Whether this engine's recognizer accepts an unknown-word cost at all."""
    try:
        eng.new(alternatives=alternatives)
    except TypeError:
        return False
    return True


def wordless_sweep(
    modules: Mapping[str, ModuleType],
    model: str | Path,
    sets: Mapping[tuple[float | None, str], list[WordlessSample]],
    levels: Sequence[float | None],
    kinds: Sequence[str],
    block_ms: int,
    alternatives: int,
    lines: list[str],
    grammars: Sequence[SweepGrammar],
) -> Json:
    """One table: every engine over every sample set under each named grammar, rows in the order
    the table reads. The engine is built once per grammar, which loads the model once."""
    out: Json = {}
    for name, grammar, cost in grammars:
        for eng_name, mod in modules.items():
            eng = Engine(eng_name, mod, model, grammar, unknown_cost=cost)
            if cost is not None and not engine_takes_cost(eng, alternatives):
                # The stock wheel takes no unknown-word cost, so it has a row only at the default.
                continue
            for level in levels:
                for kind in kinds:
                    tot = wordless_run(eng, sets[(level, kind)], block_ms, alternatives)
                    label = (
                        str(name) if len(levels) == 1 else ("digital silence" if level is None else f"{level:g} dBFS")
                    )
                    out.setdefault(label, {})[f"{kind}/{eng_name}"] = {
                        k: v for k, v in tot.items() if k not in ("gate", "per_sample")
                    }
                    out.setdefault("_paired", {})[(label, kind, eng_name)] = tot
                    lines.append(wordless_row(label, kind, eng_name, tot))
    return out


def wordless_pass(
    modules: Mapping[str, ModuleType],
    model: str | Path,
    clips: Sequence[Clip],
    noise: Sequence[Path],
    block_ms: int,
    grammar: Sequence[str],
    lines: list[str],
    report: Json,
    args: argparse.Namespace,
) -> None:
    """Words on wordless audio: how often a word appears where nothing was said, swept over the
    property that decides it on the next microphone (the floor), over the dial a host turns
    (grammar size), and under the reads a host is given,
    `[[rr:TD-8#The gate is the host's and is relative to the floor]]`."""
    cache: RoomCache = {}
    takes = wordless_takes(clips)[: args.wordless_clips]
    tail = tuple(float(v) for v in args.wordless_tail_ms.split(","))
    levels = [None if v.strip() == "silence" else float(v) for v in args.wordless_floors.split(",") if v.strip()]
    base = args.wordless_base_floor
    kinds = ("cold", "after a word")
    alts = args.wordless_alternatives
    note(f"wordless: {len(levels)} floor levels x {len(kinds)} conditions, {len(takes)} takes after a word")
    sets = {
        (level, kind): wordless_samples(kind, noise, takes, level, args.wordless_seed, tail, cache)
        for level in levels
        for kind in kinds
    }
    out: Json = {"takes": len(takes), "base_floor_dbfs": base}

    lines.append("")
    lines.append("## Words on wordless audio")
    lines.append("")
    lines.append(
        f"Every figure here is counted on audio after the last thing anybody said: the background "
        f"recordings alone, and {len(takes)} clips followed by "
        f"{tail[0] / 1000:g} to {tail[1] / 1000:g} s of one recording, from the moment each clip's own "
        f"energy ends. A word at rank 0 is counted when the partial holds more words than were "
        f"spoken so far in the segment, so a word still held from the clip is not one, nor is a "
        f"misrecognition of it, and a word beyond it is. The after-a-word takes are the quietest, "
        f"median and loudest clip of "
        f"every dataset word, chosen for the level of the word that precedes the silence. Rates are "
        f"per minute of non-speech, and both engines see the same samples."
    )

    lines.append("")
    lines.append("### By floor level")
    lines.append("")
    lines.append(
        "The recordings scaled so their floor sits at each level, and digital silence of the same "
        "length beside them, which the floor's percentile treats apart from a room."
    )
    wordless_header(lines, "floor")
    by_floor = wordless_sweep(modules, model, sets, levels, kinds, block_ms, alts, lines, [(None, grammar, None)])
    paired = by_floor.pop("_paired")
    out["floor"] = by_floor

    engines = list(modules)
    if len(engines) == 2:
        lines.append("")
        base_label = "digital silence" if base is None else f"{base:g} dBFS"
        mc: Json = {}
        for kind in kinds:
            a = [r["word_blocks"] > 0 for r in paired[(base_label, kind, engines[0])]["per_sample"]]
            b = [r["word_blocks"] > 0 for r in paired[(base_label, kind, engines[1])]["per_sample"]]
            bb = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
            cc = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
            mc[kind] = dict(b=bb, c=cc, p=mcnemar(bb, cc))
            lines.append(
                f"Paired over the same samples at {base_label}, {kind}, a sample carrying any rank-0 "
                f"word: {engines[0]} only {bb}, {engines[1]} only {cc}, exact two-sided "
                f"p = {mc[kind]['p']:.3g}."
            )
        out["mcnemar_engines"] = mc

    lines.append("")
    lines.append("### By grammar size")
    lines.append("")
    lines.append(f"At a {base:g} dBFS floor, the 35 dataset words plus filler up to each size.")
    wordless_header(lines, "entries")
    pool = filler_pool(modules, model)
    grammars: list[SweepGrammar] = []
    for size in [int(v) for v in args.grammar_sizes.split(",") if v.strip()]:
        g = DATASET_WORDS + pool[: max(0, size - len(DATASET_WORDS))]
        if all(len(g) != len(prev) for _, prev, _ in grammars):
            grammars.append((len(g), g, None))
    by_grammar = wordless_sweep(modules, model, sets, [base], kinds, block_ms, alts, lines, grammars)
    by_grammar.pop("_paired")
    out["grammar"] = by_grammar

    lines.append("")
    lines.append("### The unknown-word symbol")
    lines.append("")
    lines.append(
        "At the same floor and the full grammar, `[unk]` admitted at each cost the private sweep "
        "uses, `[[rr:The unknown-word symbol against silence phantoms]]`. The stock wheel takes no "
        "cost, so it appears once, at its own default."
    )
    wordless_header(lines, "`[unk]` cost")
    costs: list[SweepGrammar] = [("stock", [*grammar, "[unk]"], None)]
    costs += [(f"{float(v):g}", [*grammar, "[unk]"], float(v)) for v in args.wordless_unk_costs.split(",") if v.strip()]
    by_cost = wordless_sweep(modules, model, sets, [base], kinds, block_ms, alts, lines, costs)
    by_cost.pop("_paired")
    out["unknown_cost"] = by_cost

    lines.append("")
    lines.append("### The floor gate a host applies")
    lines.append("")
    lines.append(
        "The same finals with the runtime's own reads applied host-side: a final word is silence "
        "when its `energy_dbfs` is within the margin of the `floor_dbfs` that final carries, and "
        "the runtime never applies the margin. Only the runtime reports the two fields, so the "
        "stock wheel has no row here. The McNemar pairs each sample before and after the gate."
    )
    lines.append("")
    lines.append(
        "| floor | condition | margin | finals with a word /min | after the gate /min | words caught | b / c / p |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    margins = [float(v) for v in args.wordless_margins.split(",") if v.strip()]
    gate: Json = {}
    for level in levels:
        label = "digital silence" if level is None else f"{level:g} dBFS"
        for kind in kinds:
            for eng_name in modules:
                if eng_name == "vosk":
                    continue
                tot = paired[(label, kind, eng_name)]
                m = tot["minutes"]
                for margin in margins:
                    kept = [r for r in tot["gate"] if r["energy"] > r["floor"] + margin]
                    by_key = Counter(r["key"] for r in kept)
                    before = [r["phantom_finals"] > 0 for r in tot["per_sample"]]
                    after = [by_key.get(r["key"], 0) > 0 for r in tot["per_sample"]]
                    bb = sum(1 for x, y in zip(before, after, strict=True) if x and not y)
                    cc = sum(1 for x, y in zip(before, after, strict=True) if y and not x)
                    p = mcnemar(bb, cc)
                    gate.setdefault(label, {}).setdefault(kind, {})[f"{margin:g}"] = dict(
                        words=len(tot["gate"]), caught=len(tot["gate"]) - len(kept), minutes=m, b=bb, c=cc, p=p
                    )
                    lines.append(
                        f"| {label} | {kind} | {margin:g} dB | "
                        f"{tot['phantom_finals'] / m if m else float('nan'):.2f} | "
                        f"{len(kept) / m if m else float('nan'):.2f} | "
                        f"{len(tot['gate']) - len(kept)} / {len(tot['gate'])} | {bb} / {cc} / {p:.3g} |"
                    )
    out["gate"] = gate
    lines.append("")
    report["wordless"] = out


def main() -> None:
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
    ap.add_argument(
        "--determinism-clips",
        type=int,
        default=0,
        help="clips decoded in each of two fresh processes; 0 is the whole split",
    )
    ap.add_argument("--no-determinism", action="store_true", help="skip the determinism pass")
    ap.add_argument(
        "--determinism-alternatives",
        type=int,
        default=5,
        help="partial alternatives carried in the compared trace",
    )
    ap.add_argument("--endpoint-clips", type=int, default=800, help="clips for endpoint latency, 0 to skip")
    ap.add_argument(
        "--bound-engines",
        default="utterpy@300,utterpy@300/4,utterpy@300/8",
        help="host endpoint bounds run beside the stock engines in the endpoint and noise passes, "
        "spelled MS or MS/NATS; empty to skip",
    )
    ap.add_argument("--endpoint-pad-ms", type=int, default=2000, help="silence appended so an endpoint can fire")
    ap.add_argument(
        "--wordless-clips", type=int, default=105, help="clips for the after-a-word condition, 0 to skip the pass"
    )
    ap.add_argument(
        "--wordless-floors",
        default="silence,-70,-60,-50,-40",
        help="floor levels the recordings are scaled to, in dBFS",
    )
    ap.add_argument(
        "--wordless-base-floor",
        type=float,
        default=WORDLESS_BASE_FLOOR,
        help="the floor the grammar, [unk] and paired tables are read at",
    )
    ap.add_argument("--wordless-tail-ms", default="3000,5000", help="recording appended after a spoken clip, in ms")
    ap.add_argument("--wordless-unk-costs", default="8,4,2,0,-2", help="unknown-word costs swept, empty to skip")
    ap.add_argument("--wordless-margins", default="4,8,12", help="dB above the floor a host still calls silence")
    ap.add_argument(
        "--wordless-alternatives", type=int, default=4, help="partial alternatives carried, for the rival column"
    )
    ap.add_argument("--wordless-seed", type=int, default=20260912, help="seed for the tail lengths and offsets")
    ap.add_argument("--block-sizes", default="10,20,40,80,100", help="block-size sweep, empty to skip")
    ap.add_argument("--block-clips", type=int, default=400, help="clips per block size")
    args = ap.parse_args()
    data = Path(args.data)
    listed = [line.strip() for line in (data / f"{args.split}_list.txt").read_text().splitlines() if line.strip()]
    by_word: defaultdict[str, list[str]] = defaultdict(list)
    for rel in listed:
        by_word[rel.split("/")[0]].append(rel)
    clips: list[Clip] = []
    for w in DATASET_WORDS:
        items = by_word.get(w, [])
        if args.limit:
            items = items[: args.limit]
        clips.extend((w, data / rel) for rel in items)
    noise = sorted((data / "_background_noise_").glob("*.wav"))

    modules: dict[str, ModuleType] = {}
    for name in args.engines.split(","):
        modules[name] = __import__(name.split("@")[0])
        if name == "vosk":
            modules[name].SetLogLevel(-1)

    bound = [n for n in args.bound_engines.split(",") if n.strip()]
    absent = sorted({n.split("@")[0] for n in bound} - set(modules))
    if absent:
        ap.error(f"--bound-engines names {absent}, which --engines does not carry")

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
    lines.append(provenance_line(prov))
    lines.append("")
    preflight(modules, args.model, full_grammar, args.block_ms, clips, lines, report)
    full_grammar_pass(modules, args.model, clips, args.block_ms, full_grammar, lines, report)
    write_clip_records(args.out + ".clips.jsonl", clips, report["full"])
    for r in report["full"].values():
        del r["outcome"]
    if not args.no_determinism:
        determinism_pass(
            modules,
            args.model,
            clips,
            args.block_ms,
            full_grammar,
            lines,
            report,
            args.determinism_clips,
            args.determinism_alternatives,
        )
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
            bound,
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
    noise_pass(modules, args.model, noise, args.block_ms, full_grammar, lines, report, bound)
    if args.wordless_clips:
        wordless_pass(modules, args.model, clips, noise, args.block_ms, full_grammar, lines, report, args)
    # [[rr:docs/benchmarks/README.md#Benchmarks]]
    reading = Path(args.out + ".reading.md")
    if reading.exists():
        lines.append(reading.read_text().strip())
        lines.append("")
    text = write_page(args.out + ".md", lines)
    Path(args.out + ".json").write_text(json.dumps(report, indent=1))
    note(f"written: {args.out}.md, .json, .clips.jsonl")
    print(text)


if __name__ == "__main__":
    main()
