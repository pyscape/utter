#!/usr/bin/env python3
"""Run benchmark pages against a baseline build and a candidate build, and rank what drifted.

    python scripts/bench_compare.py run [--pages speech-commands,word-times] [--only PAGE:PATH] [--quick] \\
        [--baseline v0.0.4] [--reps 3] [--no-timing] [--publish] [--dry-run]
    python scripts/bench_compare.py list [--pages ...] [--only PAGE:PATH] [--depth N]
    python scripts/bench_compare.py report OLD_DIR NEW_DIR [--pages ...] [--only PAGE:PATH]
    python scripts/bench_compare.py check

The candidate is this checkout as it stands, uncommitted edits included. Each page's baseline is
the release docs/benchmarks/drift.toml names for it, or --baseline for every page. Both sides run
the candidate's scripts, so a figure that moves was moved by the runtime, not the harness.

A page's measured figures are deterministic, so a baseline run is cached and reused while the
scripts and the page's arguments are unchanged. Its timing figures are not: with timing on, the
baseline is run again beside the candidate, alternating, --reps times each, and a timing slip
counts only beyond the spread of those runs. --no-timing reuses the cache and leaves timing out.

--only names figures, speech-commands:steady_state.utterpy.rtf say, and ranks only those; list
prints the names. A page whose script can skip passes (speech-commands, decoder-diagnostics' beam
sweep) runs only the passes the figures need.

--quick runs each page on a fixed subset, minutes rather than an hour; a quick run is compared
only with a quick baseline. Outputs go under ~/Repos/utter-bench-data/compare; docs/benchmarks
is written only by --publish, which takes a full run from a clean checkout.

The ranking and every weight are in docs/benchmarks/drift.toml. check holds that file to the pages
(every section listed, every figure under a rule, every rule matching); CI runs it, and run --publish
runs it before and after.
"""

import argparse
import fnmatch
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import time
import tomllib
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from speech_commands import PASSES as SPEECH_COMMANDS_PASSES

REPO = Path(__file__).resolve().parent.parent
BENCH = Path.home() / "Repos/utter-bench-data"
UTTERPY = Path.home() / "Repos/utterpy"
PYTHON = UTTERPY / ".venv/bin/python"
DATA = BENCH / "speech_commands_v0.02"
MODEL = BENCH / "models/vosk-model-small-en-us-0.15"
KALDI = Path.home() / "Repos/kaldi/src"
DRIFT = REPO / "docs/benchmarks/drift.toml"
DOCS = REPO / "docs/benchmarks"

MAX_LOAD = 1.5
BOUNDS = ("100", "300", "300/4", "300/8", "300/8/8")
STATES_ARGS = [
    "--words", "2200", "--utterances", "10", "--block-ms", "40", "--alternatives", "8",
    "--seed", "20260912", "--lookback-ms", "1000", "--pause-variants", "100-300,400-1600",
    "--variant-words", "600", "--wordless-floors", "silence,-70,-60,-50,-40",
    "--wordless-gap-floors", "-70,-60,-40", "--wordless-words", "600",
]  # fmt: skip
STATES_QUICK = ["--words", "500", "--variant-words", "150", "--wordless-words", "150"]


@dataclass
class Side:
    """One build the pages run against: a source tree and the bindings made from it."""

    label: str
    tree: Path
    revision: str
    site: Path | None = None
    lib: Path | None = None


@dataclass
class Step:
    script: str
    argv: list[str]
    log: str


@dataclass
class Page:
    name: str
    binding: str  # "wheel" or "lib"
    minutes: tuple[float, float]  # full, quick
    steps: Callable[["Ctx"], list[Step]]
    json: Callable[[Path], Path] = lambda p: p.with_suffix(".json")
    clips: Callable[[Path], Path] = lambda p: p.with_suffix(".clips.jsonl")
    # The pass that writes each top-level JSON key, for pages whose script can skip passes; a key
    # not named here is written by a pass that always runs.
    passes: Mapping[str, str] = field(default_factory=dict)


@dataclass
class Ctx:
    prefix: Path  # <dir>/<page>, what --out is given
    work: Path  # the side's run directory, for runs a page is built from
    side: Side
    quick: bool
    sibling_inputs: Path  # where this side's other pages are, for pages that read them
    passes: frozenset[str] | None = None  # None runs every pass


def common() -> list[str]:
    return ["--data", str(DATA), "--model", str(MODEL)]


def pick(quick: bool, args: list[str]) -> list[str]:
    return args if quick else []


def speech_commands(c: Ctx) -> list[Step]:
    argv = common() + ["--out", str(c.prefix), "--bound-engines", ",".join(f"utterpy@{b}" for b in BOUNDS)]
    argv += pick(
        c.quick,
        ["--limit", "20", "--steady-state-clips", "100", "--snr-clips", "200", "--grammar-clips", "100",
         "--endpoint-clips", "200", "--wordless-clips", "30", "--block-clips", "100", "--determinism-clips", "300"],
    )  # fmt: skip
    if c.passes is not None:
        argv += ["--passes", ",".join(sorted(c.passes))]
    return [Step("speech_commands.py", argv, "speech-commands")]


def partial_trust(c: Ctx) -> list[Step]:
    p = c.prefix
    argv = common() + [
        "--engine", "utterpy", "--out", str(p.with_suffix(".md")), "--clips", str(p.with_suffix(".clips.jsonl")),
        "--fit-clips", str(BENCH / "partial-trust/validation.clips.jsonl"), "--json", str(p.with_suffix(".json")),
        "--split", "testing", "--census", str(DOCS / "partial-trust.census.json"), "--block-ms", "40",
        "--stride", "10" if c.quick else "1", "--alternatives", "5",
    ]  # fmt: skip
    return [Step("partial_trust.py", argv, "partial-trust")]


def word_times(c: Ctx) -> list[Step]:
    return [Step("word_times.py", common() + ["--out", str(c.prefix)] + pick(c.quick, ["--limit", "20"]), "word-times")]


def quiet_onsets(c: Ctx) -> list[Step]:
    argv = common() + [
        "--out", str(c.prefix), "--engines", "utterpy,utterpy@300/8,utterpy@300/8/8", "--levels", "3,6,12",
        "--hold-ms", "200", "--ramp-ms", "200", "--pauses", "400-800", "--utterances", "10", "--block-ms", "40",
        "--seed", "20260914", "--words", "150" if c.quick else "600",
    ]  # fmt: skip
    return [Step("quiet_onsets.py", argv, "quiet-onsets")]


def own_or_docs(c: Ctx, name: str) -> Path:
    """A page another page reads: this side's own run of it when there is one, else the published copy."""
    own = c.sibling_inputs / name
    return own if own.exists() else DOCS / name


def decoder_diagnostics(c: Ctx) -> list[Step]:
    argv = common() + [
        "--out", str(c.prefix), "--work", str(BENCH / "sibling"),
        "--series", str(own_or_docs(c, "partial-trust.clips.jsonl")),
        "--compare", str(own_or_docs(c, "speech-commands.clips.jsonl")),
    ]  # fmt: skip
    argv += pick(
        c.quick, ["--limit", "20", "--probe-clips", "50", "--sweep-clips", "200", "--steady-state-clips", "100"]
    )
    if c.passes is not None and "beam_sweep" not in c.passes:
        argv += ["--beam-sweep", ""]
    return [Step("decoder_diagnostics.py", argv, "decoder-diagnostics")]


def partial_states(c: Ctx) -> list[Step]:
    base = common() + STATES_ARGS + pick(c.quick, STATES_QUICK)
    runs = c.work / "boundruns"
    steps = []
    for b in BOUNDS:
        name = "b" + b.replace("/", "-")
        argv = base + ["--engine", f"utterpy@{b}", "--out", str(runs / name), "--no-streams"]
        steps.append(Step("partial_states.py", argv, f"bound-{name}"))
    bound_runs = ",".join(str(runs / ("b" + b.replace("/", "-") + ".json")) for b in BOUNDS)
    argv = base + ["--engine", "utterpy", "--out", str(c.prefix), "--with-vosk", "--bound-runs", bound_runs]
    steps.append(Step("partial_states.py", argv, "partial-states"))
    return steps


def speaker_evidence(c: Ctx) -> list[Step]:
    assert c.side.lib is not None
    argv = ["--model", str(MODEL), "--lib", str(c.side.lib), "--kaldi", str(KALDI), "--out", str(c.prefix)]
    return [Step("speaker_evidence.py", argv + pick(c.quick, ["--limit-speakers", "40"]), "speaker-evidence")]


def sound_rating(c: Ctx) -> list[Step]:
    assert c.side.lib is not None
    argv = ["--model", str(MODEL), "--lib", str(c.side.lib), "--out", str(c.prefix)]
    return [Step("sound_rating.py", argv + pick(c.quick, ["--limit", "2000", "--level-clips", "200"]), "sound-rating")]


# In the order a full run takes them: decoder-diagnostics reads the two pages before it.
PAGES = {
    p.name: p
    for p in (
        Page(
            "speech-commands",
            "wheel",
            (36, 6),
            speech_commands,
            passes={k: k for k in SPEECH_COMMANDS_PASSES} | {"agreement": "full", "significance": "full"},
        ),  # fmt: skip
        Page("partial-trust", "wheel", (5, 1), partial_trust),
        Page("word-times", "wheel", (7, 1), word_times),
        Page("quiet-onsets", "wheel", (3, 1), quiet_onsets),
        Page("decoder-diagnostics", "wheel", (15, 3), decoder_diagnostics, passes={"beam_sweep": "beam_sweep"}),
        Page("partial-states", "wheel", (36, 10), partial_states),
        Page("speaker-evidence", "lib", (27, 6), speaker_evidence),
        Page("sound-rating", "lib", (5, 1), sound_rating),
    )
}


# ---------------------------------------------------------------- running


def sh(cmd: Sequence[str], cwd: Path | None = None, env: Mapping[str, str] | None = None) -> str:
    r = subprocess.run(list(cmd), cwd=cwd, env=dict(env) if env else None, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"{' '.join(cmd)} failed ({r.returncode}):\n{r.stdout[-2000:]}{r.stderr[-4000:]}")
    return r.stdout.strip()


def git(*args: str, cwd: Path = REPO) -> str:
    return sh(["git", "-C", str(cwd), *args])


def tree_revision(tree: Path) -> str:
    """HEAD, with -dirty when tracked files differ, as utter's build.rs names the build."""
    head = git("rev-parse", "HEAD", cwd=tree)
    dirty = git("status", "--porcelain", "--untracked-files=no", cwd=tree)
    return head + ("-dirty" if dirty else "")


def short(rev: str) -> str:
    return rev[:7] + ("-dirty" if rev.endswith("-dirty") else "")


def quiet_window_block(minutes: float, now: datetime | None = None) -> str | None:
    """Mon-Fri 06:15-12:01 local starts nothing, and nothing starts that would still run at 06:00."""
    now = now or datetime.now()
    weekday = now.weekday() < 5
    t = now.time()
    if weekday and (6, 15) <= (t.hour, t.minute) < (12, 1):
        return "inside the weekday quiet window (06:15-12:01)"
    end = now + timedelta(minutes=minutes)
    six = now.replace(hour=6, minute=0, second=0, microsecond=0)
    if six <= now:
        six += timedelta(days=1)
    if six.weekday() < 5 and end >= six:
        return f"would likely still run at 06:00 (about {minutes:.0f} min from {now:%H:%M})"
    return None


def wait_for_load(limit: float) -> None:
    said = False
    while (load := os.getloadavg()[0]) >= limit:
        if not said:
            print(f"  waiting for load {load:.2f} to fall under {limit}", flush=True)
            said = True
        time.sleep(30)


def worktree(rev: str, into: Path, repo: Path) -> Path:
    """A detached worktree of rev, made once and reused."""
    sha = git("rev-parse", f"{rev}^{{commit}}", cwd=repo)
    path = into / sha[:12]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        git("worktree", "add", "--detach", str(path), sha, cwd=repo)
    return path


def crate_version(tree: Path) -> str:
    return str(tomllib.loads((tree / "Cargo.toml").read_text())["package"]["version"])


def build_lib(side: Side, work: Path) -> Path:
    key = side.revision if not side.revision.endswith("-dirty") else "candidate"
    target = work / "target" / ("lib-" + key[:12])
    env = dict(os.environ, CARGO_TARGET_DIR=str(target))
    print(f"  building libutter at {short(side.revision)}", flush=True)
    sh(["cargo", "build", "--release", "--lib", "--manifest-path", str(side.tree / "Cargo.toml")], env=env)
    out = work / "libs" / f"libutter-{short(side.revision)}.so"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target / "release/libutter.so", out)
    return out


def utterpy_ref(baseline_tag: str | None) -> str:
    """The binding's source: the release's own tag for a release baseline, else utterpy's main."""
    if baseline_tag:
        tags = git("tag", "--list", baseline_tag, cwd=UTTERPY)
        if tags:
            return baseline_tag
    return "origin/main"


def build_wheel(side: Side, work: Path, ref: str) -> Path:
    """The binding built from utterpy at ref with its utter dependency patched to side.tree.

    utterpy pins utter to one 0.0.x version and cargo drops a path patch whose version the pin does
    not match, silently building the registry crate instead; the pin is set to the tree's version
    and the lock is checked for an unused patch, and the built binding must name side.revision."""
    src = worktree(ref, work / "utterpy", UTTERPY)
    git("checkout", "--", ".", cwd=src)
    version = crate_version(side.tree)
    manifest = src / "Cargo.toml"
    text = manifest.read_text()
    lines = [f'utter = "={version}"' if ln.startswith("utter = ") else ln for ln in text.splitlines()]
    manifest.write_text("\n".join(lines) + "\n")
    (src / ".cargo").mkdir(exist_ok=True)
    (src / ".cargo/config.toml").write_text(f'[patch.crates-io]\nutter = {{ path = "{side.tree}" }}\n')
    key = side.revision[:12] if not side.revision.endswith("-dirty") else "candidate"
    target = work / "target" / ("wheel-" + key)
    wheels = work / "wheels" / key
    shutil.rmtree(wheels, ignore_errors=True)
    env = dict(os.environ, CARGO_TARGET_DIR=str(target))
    print(f"  building utterpy {ref} against utter {short(side.revision)}", flush=True)
    sh([str(PYTHON), "-m", "maturin", "build", "--release", "--out", str(wheels)], cwd=src, env=env)
    if "[[patch.unused]]" in (src / "Cargo.lock").read_text():
        raise SystemExit(f"utterpy {ref} did not take the utter patch; the binding would be the registry crate")
    site = work / "site" / key
    shutil.rmtree(site, ignore_errors=True)
    (wheel,) = wheels.glob("*.whl")
    sh([str(PYTHON), "-m", "pip", "install", "--quiet", "--no-deps", "--target", str(site), str(wheel)])
    probe = "import utterpy; print(utterpy.UTTER_REVISION)"
    built = sh([str(PYTHON), "-c", probe], env=dict(os.environ, PYTHONPATH=str(site)))
    if built != side.revision:
        raise SystemExit(f"the binding names utter {built}, not {side.revision}")
    return site


def scripts_digest() -> str:
    h = hashlib.sha256()
    for p in sorted((REPO / "scripts").glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def run_page(page: Page, ctx: Ctx, logs: Path) -> tuple[bool, float, str]:
    """Every step of a page, in order; false and the log's tail when a step fails."""
    env = dict(os.environ, TMPDIR=str(BENCH / "tmp"))
    if ctx.side.site:
        env["PYTHONPATH"] = str(ctx.side.site)
    (BENCH / "tmp").mkdir(exist_ok=True)
    ctx.prefix.parent.mkdir(parents=True, exist_ok=True)
    spent = 0.0
    for step in page.steps(ctx):
        wait_for_load(MAX_LOAD)
        start = time.monotonic()
        log = logs / f"{ctx.side.label}-{step.log}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as f:
            f.write(f"# {' '.join([str(PYTHON), 'scripts/' + step.script, *step.argv])}\n")
            f.flush()
            rc = subprocess.run(
                [str(PYTHON), str(REPO / "scripts" / step.script), *step.argv],
                cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT,
            ).returncode  # fmt: skip
        spent += time.monotonic() - start
        if rc:
            tail = "\n".join(log.read_text().splitlines()[-15:])
            return False, spent, f"{step.log} exited {rc} ({log}):\n{tail}"
    return True, spent, ""


def parse_only(specs: Sequence[str]) -> list[tuple[str, list[str]]]:
    """--only's page:path patterns; a path is dotted segments, each a glob, and names its subtree."""
    out = []
    for spec in specs:
        for item in spec.split(","):
            if not item.strip():
                continue
            page, _, path = item.strip().rpartition(":")
            out.append((page or "*", [seg for seg in path.split(".") if seg]))
    return out


def only_for(only: Sequence[tuple[str, list[str]]], page: str) -> list[list[str]] | None:
    """The patterns that apply to a page, or None when --only was not given."""
    if not only:
        return None
    return [path for p, path in only if fnmatch.fnmatchcase(page, p)]


def needed_passes(page: Page, patterns: list[list[str]] | None) -> frozenset[str] | None:
    """The passes a page must run for the figures asked for; None when every pass is needed."""
    if patterns is None or not page.passes:
        return None
    needed: set[str] = set()
    for path in patterns:
        if not path or not literal(path[0]):
            return None
        needed.update(page.passes[o] for o in options(path[0]) if o in page.passes)
    return frozenset(needed)


def run_side(
    page: Page, side: Side, rep: int, run_dir: Path, quick: bool, publish: bool, passes: frozenset[str] | None
) -> tuple[Path | None, str]:
    """One run of a page on one side: where its outputs are, or why there are none."""
    label = f"{side.label}-{rep}"
    prefix, sib = run_dir / label / page.name, run_dir / label
    if publish and side.label == "candidate":
        prefix, sib = DOCS / page.name, DOCS
    ctx = Ctx(prefix, run_dir / label, side, quick, sib, passes)
    ok, secs, err = run_page(page, ctx, run_dir / "logs" / str(rep))
    print(f"  {label} {'done' if ok else 'FAILED'} in {secs / 60:.1f} min", flush=True)
    return (prefix, "") if ok else (None, f"{side.label} run failed: {err}")


# ---------------------------------------------------------------- comparing


@dataclass
class Rule:
    index: int
    page: str
    path: list[str]
    kind: str
    better: str
    tol: float | None
    rel: float | None
    weight: float
    line: str | float | None
    note: str


def load_rules(path: Path = DRIFT) -> tuple[list[Rule], dict[str, Any], dict[str, str]]:
    cfg = tomllib.loads(path.read_text())
    return parse_rules(cfg)


def parse_rules(cfg: Mapping[str, Any]) -> tuple[list[Rule], dict[str, Any], dict[str, str]]:
    d = cfg.get("defaults", {})
    rules = []
    for i, r in enumerate(cfg.get("rule", [])):
        tol = r.get("tol")
        rel = r.get("rel", d.get("rel") if tol is None else None)
        rules.append(
            Rule(
                i,
                r["page"],
                list(r["path"]),
                r.get("kind", d.get("kind", "exact")),
                r.get("better", d.get("better", "either")),
                tol,
                rel,
                float(r.get("weight", d.get("weight", 1))),
                r.get("line"),
                r.get("note", ""),
            )  # fmt: skip
        )
    default = Rule(-1, "*", ["**"], d.get("kind", "exact"), d.get("better", "either"), None,
                   d.get("rel", 0.1), float(d.get("weight", 1)), None, "no rule; the default")  # fmt: skip
    baseline = dict(cfg.get("baseline", {}))
    return rules, {"default": default}, baseline


def options(seg: str) -> list[str]:
    return seg[1:-1].split(",") if seg.startswith("{") and seg.endswith("}") else [seg]


def segment(name: str, pattern: str) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in options(pattern))


def match(pattern: Sequence[str], path: Sequence[str]) -> bool:
    if not pattern:
        return not path
    if pattern[0] == "**":
        return any(match(pattern[1:], path[i:]) for i in range(len(path) + 1))
    return bool(path) and segment(path[0], pattern[0]) and match(pattern[1:], path[1:])


def rule_for(rules: Sequence[Rule], default: Rule, page: str, path: Sequence[str]) -> Rule:
    for r in rules:
        if fnmatch.fnmatchcase(page, r.page) and match(r.path, path):
            return r
    return default


def flatten(o: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], float]]:
    """Every numeric figure by path. A list's elements are keyed by index; a list of anything but
    numbers and objects (clip names, say) is one figure, its length."""
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flatten(v, path + (str(k),))
    elif isinstance(o, list):
        if not o or not all(isinstance(x, (dict, list, int, float)) and not isinstance(x, bool) for x in o):
            yield path, float(len(o))
            return
        for i, v in enumerate(o):
            yield from flatten(v, path + (str(i),))
    elif isinstance(o, (int, float)) and not isinstance(o, bool) and math.isfinite(o):
        yield path, float(o)


def figures(doc: Mapping[str, Any]) -> dict[tuple[str, ...], float]:
    body = {k: v for k, v in doc.items() if k not in ("provenance", "args", "run")}
    return dict(flatten(body))


@dataclass
class Finding:
    page: str
    path: tuple[str, ...]
    rule: Rule
    base: float
    cand: float
    slip: float  # in the figure's unit, the part that counts
    unit: float  # the tolerance it is measured against
    severity: float
    line: bool
    spread: float | None = None
    folded: int = 0

    @property
    def name(self) -> str:
        return ".".join(self.path)


@dataclass
class PageResult:
    page: str
    baseline: str
    findings: list[Finding] = field(default_factory=list)
    improvements: list[Finding] = field(default_factory=list)
    moved_inputs: list[str] = field(default_factory=list)
    compared: int = 0
    unchanged: int = 0
    ignored: int = 0
    timing_reps: int = 0
    clips_moved: tuple[int, int, list[str]] | None = None
    error: str = ""


def judge(page: str, base_runs: Sequence[Mapping[str, Any]], cand_runs: Sequence[Mapping[str, Any]],
          rules: Sequence[Rule], default: Rule, timing: bool,
          only: list[list[str]] | None = None) -> PageResult:  # fmt: skip
    res = PageResult(page, "")
    bases = [figures(d) for d in base_runs]
    cands = [figures(d) for d in cand_runs]
    res.timing_reps = min(len(bases), len(cands))
    for path in sorted(set(bases[0]) | set(cands[0])):
        rule = rule_for(rules, default, page, path)
        # Inputs and the oracle are checked whatever --only names: they say whether the rest compares.
        if only is not None and rule.kind not in ("input", "oracle") and not any(match(p + ["**"], path) for p in only):
            continue
        if rule.kind == "ignore" or (rule.kind == "timing" and not timing):
            res.ignored += 1
            continue
        b_all = [f[path] for f in bases if path in f]
        c_all = [f[path] for f in cands if path in f]
        if not b_all or not c_all:
            if rule.kind in ("input", "oracle"):
                res.moved_inputs.append(f"{'.'.join(path)} only in the {'candidate' if c_all else 'baseline'}")
            else:
                res.ignored += 1
            continue
        res.compared += 1
        b, c = statistics.median(b_all), statistics.median(c_all)
        if rule.kind in ("input", "oracle"):
            if b != c:
                res.moved_inputs.append(f"{'.'.join(path)} {fmt(b)} -> {fmt(c)} ({rule.kind})")
            else:
                res.unchanged += 1
            continue
        spread = None
        if rule.kind == "timing" and len(b_all) > 1 and len(c_all) > 1:
            spread = max(max(b_all) - min(b_all), max(c_all) - min(c_all))
        delta = c - b
        worse = {"up": -delta, "down": delta}.get(rule.better, abs(delta))
        unit = rule.tol if rule.tol is not None else (rule.rel or 0.1) * (abs(b) or 1.0)
        if delta == 0 or (spread is not None and abs(delta) <= spread):
            res.unchanged += 1
            continue
        counted = max(0.0, abs(worse) - (spread or 0.0))
        if worse <= 0:
            res.improvements.append(
                Finding(page, path, rule, b, c, counted, unit, rule.weight * counted / unit, False, spread)
            )
            continue
        crossed = crossed_line(rule, b, c, spread)
        res.findings.append(
            Finding(page, path, rule, b, c, counted, unit, rule.weight * counted / unit, crossed, spread)
        )
    return res


def crossed_line(rule: Rule, b: float, c: float, spread: float | None) -> bool:
    if rule.line is None:
        return False
    if rule.line == "any":
        return True
    if rule.line == "noise":
        return spread is not None
    bound = float(rule.line)
    return {"up": c < bound, "down": c > bound}.get(rule.better, c != bound)


def fold(findings: Sequence[Finding]) -> list[Finding]:
    """One line per rule a page's figures fell under: its worst, with how many more it stands for."""
    groups: dict[tuple[str, int, str], list[Finding]] = {}
    for f in findings:
        key = (f.page, f.rule.index, ".".join(f.path) if f.rule.index < 0 else "")
        groups.setdefault(key, []).append(f)
    out = []
    for g in groups.values():
        g.sort(key=lambda f: (f.line, f.severity), reverse=True)
        head = g[0]
        head.folded = len(g) - 1
        out.append(head)
    return sorted(out, key=lambda f: f.severity, reverse=True)


def clips_moved(base: Path, cand: Path) -> tuple[int, int, list[str]] | None:
    """How many clips' records differ, the oracle's readings left out, and the first few."""
    if not (base.exists() and cand.exists()):
        return None

    def load(p: Path) -> dict[str, Any]:
        rows = {}
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if "clip" in r:
                    # Older pages named a clip by its absolute path, newer ones by word/file.
                    clip = "/".join(Path(r["clip"]).parts[-2:])
                    rows[clip] = {k: v for k, v in r.items() if k != "clip" and not k.startswith("vosk")}
        return rows

    a, b = load(base), load(cand)
    shared = [k for k in b if k in a]
    moved = [k for k in shared if a[k] != b[k]]
    return len(moved), len(shared), moved[:5]


def fmt(x: float) -> str:
    if x == int(x) and abs(x) < 1e12:
        return str(int(x))
    return f"{x:.4g}"


def describe(f: Finding) -> str:
    why = f"weight {fmt(f.rule.weight)}, tol {fmt(f.unit)}"
    if f.rule.line is not None:
        why += f", line {f.rule.line}"
    if f.spread is not None:
        why += f", spread {fmt(f.spread)}"
    more = f"  (+{f.folded} more under this rule)" if f.folded else ""
    note = f"\n        {f.rule.note}" if f.rule.note else ""
    return f"[{f.severity:7.1f}] {f.page}  {f.name}  {fmt(f.base)} -> {fmt(f.cand)}  ({f.rule.kind}; {why}){more}{note}"


def render(results: Sequence[PageResult], header: str, timing: bool) -> str:
    out = [header, ""]
    bad = [r for r in results if r.moved_inputs or r.error]
    if bad:
        out.append("Not comparable")
        for r in bad:
            if r.error:
                out.append(f"  {r.page}: {r.error}")
            for m in r.moved_inputs[:8]:
                out.append(f"  {r.page}: {m}")
            if len(r.moved_inputs) > 8:
                out.append(f"  {r.page}: +{len(r.moved_inputs) - 8} more")
        out.append("")
    ok = [r for r in results if not r.moved_inputs and not r.error]
    findings = fold([f for r in ok for f in r.findings])
    lines = [f for f in findings if f.line]
    slips = [f for f in findings if not f.line]
    if lines:
        out.append("Lines crossed")
        out += ["  " + describe(f) for f in lines] + [""]
    out.append("Slips" if slips else "Slips: none")
    out += ["  " + describe(f) for f in slips]
    out.append("")
    gains = fold([f for r in ok for f in r.improvements])
    if gains:
        out.append("Improvements")
        out += ["  " + describe(f) for f in gains] + [""]
    out.append("Pages")
    for r in results:
        s = f"  {r.page} against {r.baseline}: {r.compared} figures compared, {r.unchanged} unchanged, {r.ignored} left out"
        if timing and r.timing_reps == 1 and not r.error:
            s += "; timing from one run a side, spread unknown, so no timing line can be crossed"
        if r.clips_moved:
            n, of, first = r.clips_moved
            s += f"; {n} of {of} clips' records differ" + (f" ({', '.join(first)}...)" if n else "")
        out.append(s)
    return "\n".join(out) + "\n"


def to_json(results: Sequence[PageResult]) -> list[dict[str, Any]]:
    def f(x: Finding) -> dict[str, Any]:
        return dict(
            page=x.page, path=list(x.path), kind=x.rule.kind, base=x.base, cand=x.cand, slip=x.slip, tol=x.unit,
            weight=x.rule.weight, severity=x.severity, line=x.line, spread=x.spread, rule=x.rule.index,
        )  # fmt: skip

    return [
        dict(
            page=r.page,
            baseline=r.baseline,
            error=r.error,
            moved_inputs=r.moved_inputs,
            compared=r.compared,
            unchanged=r.unchanged,
            ignored=r.ignored,
            clips_moved=r.clips_moved,
            findings=[f(x) for x in r.findings],
            improvements=[f(x) for x in r.improvements],
        )  # fmt: skip
        for r in results
    ]


# ---------------------------------------------------------------- checking


def section_of(path: Sequence[str]) -> str:
    return ".".join(path[:2] if path[0] == "result" and len(path) > 1 else path[:1])


def literal(seg: str) -> bool:
    return not any(ch in o for o in options(seg) for ch in "*?[")


def anchored(rule: Rule) -> bool:
    """Whether a rule names the section it scores, so a new section cannot fall under it unseen."""
    if rule.page == "*" or not rule.path:
        return False
    at = 1 if rule.path[0] == "result" else 0
    return len(rule.path) > at and all(literal(seg) for seg in rule.path[: at + 1])


def version_tag(ref: str) -> tuple[int, ...] | None:
    parts = ref.removeprefix("v").split(".")
    return tuple(int(p) for p in parts) if ref.startswith("v") and all(p.isdigit() for p in parts) else None


def check(pages_dir: Path = DOCS, drift: Path = DRIFT) -> list[str]:
    """What keeps drift.toml from describing the pages: every problem found, empty when it holds."""
    cfg = tomllib.loads(drift.read_text())
    rules, defaults, baselines = parse_rules(cfg)
    listed: dict[str, list[str]] = dict(cfg.get("sections", {}))
    problems = []

    on_disk = {p.name.removesuffix(".json") for p in pages_dir.glob("*.json") if p.name.count(".") == 1}
    for name in sorted(on_disk - set(PAGES)):
        problems.append(f"{name}.json is in {pages_dir.name} but bench_compare.py has no page for it")
    for name in sorted(set(PAGES) - on_disk):
        problems.append(f"page {name} has no {name}.json in {pages_dir.name}")
    for name in sorted(set(listed) - set(PAGES)):
        problems.append(f"[sections] lists {name}, which is not a page")

    used: set[int] = set()
    for name, page in PAGES.items():
        if name not in on_disk:
            continue
        figs = figures(json.loads(page.json(pages_dir / name).read_text()))
        sections = list(dict.fromkeys(section_of(k) for k in figs))
        want = listed.get(name)
        if want is None:
            problems.append(f"{name}: no [sections] entry; its sections are {sections}")
        else:
            for sec in sections:
                if sec not in want:
                    problems.append(f"{name}: section {sec} is not in [sections]; decide its rules and list it")
            for sec in want:
                if sec not in sections:
                    problems.append(f"{name}: [sections] lists {sec}, which the page no longer writes")
        unruled: dict[str, int] = {}
        for k in figs:
            rule = rule_for(rules, defaults["default"], name, k)
            if rule.index < 0:
                unruled[section_of(k)] = unruled.get(section_of(k), 0) + 1
            else:
                used.add(rule.index)
        for sec, n in unruled.items():
            problems.append(
                f"{name}: {n} figure(s) in {sec} have no rule (bench_compare.py list --only '{name}:{sec}')"
            )
        for field_key in page.passes:
            if field_key not in sections:
                problems.append(f"{name}: the pass map names {field_key}, which the page does not write")
        if name == "speech-commands" and set(page.passes.values()) != set(SPEECH_COMMANDS_PASSES):
            problems.append("speech-commands: the pass map and speech_commands.PASSES name different passes")

    for r in rules:
        where = f"rule {r.index} ({r.page}: {'.'.join(r.path)})"
        if r.index not in used:
            problems.append(f"{where} matches no figure")
        if r.kind in ("exact", "timing") and not anchored(r):
            problems.append(f"{where} scores figures without naming their section")
        if r.kind not in ("exact", "timing", "oracle", "input", "ignore"):
            problems.append(f"{where} has kind {r.kind}")
        if r.better not in ("up", "down", "either"):
            problems.append(f"{where} has better = {r.better}")

    tags = [t for t in (version_tag(x) for x in git("tag", "--list", "v*").split()) if t]
    newest = max(tags) if tags else ()
    for name in PAGES:
        ref = baselines.get(name, baselines.get("default"))
        if not ref:
            problems.append(f"{name}: no baseline and no default")
            continue
        try:
            git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
        except SystemExit:
            v = version_tag(ref)
            if v is None or v <= newest:
                problems.append(f"{name}: baseline {ref} does not resolve")
    return problems


# ---------------------------------------------------------------- commands


def selected(arg: str | None, only: Sequence[tuple[str, list[str]]] = ()) -> list[Page]:
    """--pages, or when it is not given, the pages --only names, or every page."""
    names = [n.strip() for n in (arg or "").split(",") if n.strip()]
    if not names:
        named = [page for page, _ in only if page != "*"]
        if not named or len(named) < len(only):
            return list(PAGES.values())
        names = [n for n in PAGES if any(fnmatch.fnmatchcase(n, pat) for pat in named)]
    unknown = [n for n in names if n not in PAGES]
    if unknown:
        raise SystemExit(f"no page {', '.join(unknown)}; the pages are {', '.join(PAGES)}")
    return [p for p in PAGES.values() if p.name in names]


def cmd_run(a: argparse.Namespace) -> None:
    only = parse_only(a.only or [])
    pages = selected(a.pages, only)
    rules, defaults, baselines = load_rules()
    work = Path(a.work)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = work / "runs" / stamp
    candidate = Side("candidate", REPO, tree_revision(REPO))
    if a.publish:
        if a.quick:
            raise SystemExit("--publish takes a full run")
        if candidate.revision.endswith("-dirty"):
            raise SystemExit("--publish takes a clean checkout; commit first")
        if only:
            raise SystemExit("--publish takes whole pages, not --only")
        if problems := check():
            raise SystemExit("drift.toml does not describe the pages; bench_compare.py check:\n" + "\n".join(problems))

    plan: list[tuple[Page, str | None]] = []
    missing: list[str] = []
    for p in pages:
        if only and not only_for(only, p.name):
            print(f"skipping {p.name} (--only names nothing on it)")
            continue
        rev = a.baseline or baselines.get(p.name, baselines.get("default", "v0.0.4"))
        try:
            git("rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}")
        except SystemExit:
            if a.publish:
                print(f"{p.name}: baseline {rev} does not exist yet; published without a comparison")
                plan.append((p, None))
            else:
                missing.append(f"{p.name} (baseline {rev} does not exist yet; --baseline names another)")
            continue
        plan.append((p, rev))
    for m in missing:
        print(f"skipping {m}")
    if not plan:
        return

    reps = a.reps if not a.no_timing else 1
    total = sum(p.minutes[1 if a.quick else 0] * (1 + (0 if a.no_timing else 1)) * reps for p, _ in plan)
    if not a.ignore_quiet_window and (why := quiet_window_block(total)):
        raise SystemExit(f"not starting: {why}")
    print(f"run {run_dir}; at most about {total:.0f} min if nothing is cached", flush=True)
    if a.dry_run:
        for page, rev in plan:
            placeholder = Side("candidate", REPO, candidate.revision, BENCH / "SITE", BENCH / "LIB.so")
            passes = needed_passes(page, only_for(only, page.name))
            ctx = Ctx(run_dir / "candidate-1" / page.name, run_dir / "candidate-1", placeholder, a.quick,
                      run_dir / "candidate-1", passes)  # fmt: skip
            shown = "every pass" if passes is None else "passes " + (", ".join(sorted(passes)) or "that always run")
            print(f"{page.name}: baseline {rev or 'none yet'}, binding {page.binding}, {shown}")
            for step in page.steps(ctx):
                print(f"  {step.script} {' '.join(step.argv)}")
        return

    sides: dict[str, Side] = {}

    def side_for(rev: str, binding: str) -> Side:
        if rev not in sides:
            tree = worktree(rev, work / "src", REPO)
            sides[rev] = Side(f"base-{rev.replace('/', '-')}", tree, tree_revision(tree))
        s = sides[rev]
        if binding == "wheel" and s.site is None:
            s.site = build_wheel(s, work, utterpy_ref(rev if git("tag", "--list", rev) else None))
        if binding == "lib" and s.lib is None:
            s.lib = build_lib(s, work)
        return s

    def ensure_candidate(binding: str) -> None:
        if binding == "wheel" and candidate.site is None:
            candidate.site = build_wheel(candidate, work, a.utterpy or "origin/main")
        if binding == "lib" and candidate.lib is None:
            candidate.lib = build_lib(candidate, work)

    git("fetch", "--quiet", "origin", cwd=UTTERPY)
    digest = scripts_digest()
    results: list[PageResult] = []
    mode = "quick" if a.quick else "full"
    for page, maybe_rev in plan:
        ensure_candidate(page.binding)
        patterns = only_for(only, page.name)
        passes = needed_passes(page, patterns)
        if maybe_rev is None:
            print(f"{page.name}: no baseline yet, {mode}", flush=True)
            res = PageResult(page.name, "no baseline yet")
            results.append(res)
            _, res.error = run_side(page, candidate, 1, run_dir, a.quick, a.publish, passes)
            continue
        rev = maybe_rev
        print(f"{page.name}: baseline {rev}, {mode}", flush=True)
        base = side_for(rev, page.binding)
        res = PageResult(page.name, f"{rev} ({short(base.revision)})")
        results.append(res)
        ran = "all" if passes is None else ",".join(sorted(passes))
        key = hashlib.sha256(f"{page.name}|{mode}|{ran}|{base.revision}|{digest}".encode()).hexdigest()[:16]
        cache = work / "cache" / key
        base_runs: list[dict[str, Any]] = []
        cand_runs: list[dict[str, Any]] = []
        base_prefix = cand_prefix = Path()

        if a.no_timing and (cache / f"{page.name}.json").exists():
            print("  baseline from cache", flush=True)
            base_prefix = cache / page.name
            base_runs.append(json.loads(page.json(base_prefix).read_text()))
            got, res.error = run_side(page, candidate, 1, run_dir, a.quick, a.publish, passes)
            if got:
                cand_prefix = got
                cand_runs.append(json.loads(page.json(got).read_text()))
        else:
            # A side that fails does not stop the other: with --publish the candidate's page is
            # wanted even when the baseline cannot run today's script.
            errors: list[str] = []
            for rep in range(1, reps + 1):
                order = (base, candidate) if rep % 2 else (candidate, base)
                for side in order:
                    if a.publish and side is candidate and rep > 1:
                        continue
                    got, err = run_side(page, side, rep, run_dir, a.quick, a.publish, passes)
                    if got is None:
                        errors.append(err)
                        continue
                    doc = json.loads(page.json(got).read_text())
                    if side is candidate:
                        cand_runs.append(doc)
                        cand_prefix = got
                    else:
                        base_runs.append(doc)
                        base_prefix = got
                if errors:
                    break
            res.error = "; ".join(errors)
            if base_runs and not res.error:
                shutil.rmtree(cache, ignore_errors=True)
                cache.mkdir(parents=True)
                for f in base_prefix.parent.glob(page.name + ".*"):
                    shutil.copy2(f, cache / f.name)
        if res.error or not base_runs or not cand_runs:
            continue
        judged = judge(page.name, base_runs, cand_runs, rules, defaults["default"], not a.no_timing, patterns)
        judged.baseline = res.baseline
        judged.clips_moved = clips_moved(page.clips(base_prefix), page.clips(cand_prefix))
        results[-1] = judged

    header = f"candidate {short(candidate.revision)}, {mode}, " + (
        "timing left out" if a.no_timing else f"{reps} run{'s' if reps > 1 else ''} a side"
    )
    text = render(results, header, not a.no_timing)
    if a.publish and (problems := check()):
        text += "\nThe published pages have figures drift.toml does not describe; fix it before the next run:\n"
        text += "".join(f"  {p}\n" for p in problems)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.txt").write_text(text)
    (run_dir / "report.json").write_text(json.dumps(to_json(results), indent=1) + "\n")
    print()
    print(text, end="")
    print(f"report in {run_dir}")


def cmd_report(a: argparse.Namespace) -> None:
    rules, defaults, _ = load_rules()
    old, new = Path(a.old), Path(a.new)
    results = []
    only = parse_only(a.only or [])
    for page in selected(a.pages, only):
        if only and not only_for(only, page.name):
            continue
        bo, bn = page.json(old / page.name), page.json(new / page.name)
        if not (bo.exists() and bn.exists()):
            continue
        r = judge(page.name, [json.loads(bo.read_text())], [json.loads(bn.read_text())], rules,
                  defaults["default"], a.timing, only_for(only, page.name))  # fmt: skip
        r.baseline = str(old)
        r.clips_moved = clips_moved(page.clips(old / page.name), page.clips(new / page.name))
        results.append(r)
    if not results:
        raise SystemExit(f"no page has a .json in both {old} and {new}")
    print(render(results, f"{old} -> {new}", a.timing), end="")


def cmd_check(a: argparse.Namespace) -> None:
    problems = check(Path(a.dir))
    for p in problems:
        print(p)
    if problems:
        raise SystemExit(f"{len(problems)} problem(s): drift.toml does not describe the pages")
    print(f"drift.toml describes every figure on the {len(PAGES)} pages")


def cmd_list(a: argparse.Namespace) -> None:
    """The figures a page reports, by the names --only takes, from the pages in docs/benchmarks."""
    rules, defaults, _ = load_rules()
    only = parse_only(a.only or [])
    for page in selected(a.pages, only):
        patterns = only_for(only, page.name)
        if only and not patterns:
            continue
        path = page.json(Path(a.dir) / page.name)
        if not path.exists():
            print(f"{page.name}: no {path}")
            continue
        figs = figures(json.loads(path.read_text()))
        if patterns is not None:
            figs = {k: v for k, v in figs.items() if any(match(p + ["**"], k) for p in patterns)}
        depth = a.depth or (0 if patterns else 1)
        print(f"{page.name}  ({len(figs)} figures)")
        if depth:
            groups: dict[tuple[str, ...], int] = {}
            for k in figs:
                groups[k[:depth]] = groups.get(k[:depth], 0) + 1
            for k, n in groups.items():
                one = n == 1 and k in figs
                print(f"  {page.name}:{'.'.join(k)}" + (f" = {fmt(figs[k])}" if one else f"  ({n})"))
            continue
        for k, v in figs.items():
            r = rule_for(rules, defaults["default"], page.name, k)
            how = (
                r.kind
                if r.kind in ("input", "oracle", "ignore")
                else f"{r.kind}, better {r.better}, weight {fmt(r.weight)}"
            )
            print(f"  {page.name}:{'.'.join(k)} = {fmt(v)}  ({how})")


EXAMPLES = """\
examples:
  find a figure's name
    bench_compare.py list --pages speech-commands
    bench_compare.py list --only speech-commands:steady_state

  iterate on decode speed against main, steady state only, three runs a side
    bench_compare.py run --only speech-commands:steady_state.utterpy --baseline origin/main --reps 3

  iterate on accuracy, no timing, cached baseline
    bench_compare.py run --pages speech-commands,word-times --quick --no-timing

  drift check of every page against its release baseline
    bench_compare.py run --reps 2

  rerun the pages for a release and write docs/benchmarks
    bench_compare.py run --publish

  see what a run would do without building or running anything
    bench_compare.py run --only word-times:start --dry-run
"""


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser(
        "run",
        help="build both sides, run the pages, rank the drift",
        description="Build the baseline and the candidate, run the pages on both, and rank what drifted.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    r.add_argument("--pages", help=f"comma-separated, from {', '.join(PAGES)}; all by default")
    r.add_argument(
        "--only",
        action="append",
        help="figures to rank, as page:path with dotted glob segments (speech-commands:steady_state.utterpy); "
        "repeatable or comma-separated; a page whose script can skip passes runs only those the figures need",
    )
    r.add_argument("--quick", action="store_true", help="each page on a fixed subset")
    r.add_argument("--baseline", help="one revision for every page instead of drift.toml's per-page releases")
    r.add_argument("--reps", type=int, default=1, help="runs a side, alternating; 2 or more gives timing a spread")
    r.add_argument("--no-timing", action="store_true", help="reuse a cached baseline and leave timing out")
    r.add_argument("--dry-run", action="store_true", help="print the plan and each page's commands, run nothing")
    r.add_argument("--publish", action="store_true", help="write the candidate's pages to docs/benchmarks")
    r.add_argument("--utterpy", help="utterpy revision the candidate's binding is built from (origin/main)")
    r.add_argument("--work", default=str(BENCH / "compare"), help="builds, cache and run outputs (%(default)s)")
    r.add_argument(
        "--ignore-quiet-window",
        action="store_true",
        help="start inside the weekday 06:15-12:01 window or past 06:00; the owner's call only",
    )
    r.set_defaults(func=cmd_run)
    c = sub.add_parser(
        "check",
        help="that drift.toml describes every page, section and figure",
        description="That drift.toml describes the pages: every page and section listed, every figure under a "
        "rule of its own, every scoring rule naming its section, every rule matching a figure, every baseline "
        "resolving. run --publish runs it before and after.",
    )
    c.add_argument("--dir", default=str(DOCS), help="the pages to check (docs/benchmarks)")
    c.set_defaults(func=cmd_check)
    ls = sub.add_parser(
        "list",
        help="name the figures a page reports, to hand to --only",
        description="Name the figures a page reports, as --only takes them: a page's sections first, "
        "a named section's figures with their value, kind and weight.",
    )
    ls.add_argument("--pages")
    ls.add_argument("--only", action="append", help="the figures to list, as for run")
    ls.add_argument(
        "--depth", type=int, default=0, help="group by this many path segments; 1 by default, leaves under --only"
    )
    ls.add_argument("--dir", default=str(DOCS), help="the pages to read (docs/benchmarks)")
    ls.set_defaults(func=cmd_list)
    p = sub.add_parser(
        "report",
        help="rank the drift between two directories of page outputs",
        description="Rank the drift between two directories holding <page>.json (and <page>.clips.jsonl), "
        "running nothing: two runs' outputs, or docs/benchmarks against a copy of it.",
    )
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--pages")
    p.add_argument("--only", action="append", help="figures to rank, as for run")
    p.add_argument("--timing", action="store_true", help="rank timing figures too; single runs, no spread")
    p.set_defaults(func=cmd_report)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
