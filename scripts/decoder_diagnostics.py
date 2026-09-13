#!/usr/bin/env python3
"""Diagnostics on Google Speech Commands v2: where a wrong word was, not why it was wrong.

    python scripts/decoder_diagnostics.py --data DIR --model MODEL_DIR --out docs/benchmarks/decoder-diagnostics

Three measurements, all observation:

- oracle-in-beam: with n-best asked for on the partial and on the final, is the clip's word
  among the readings at all, at n = 1, 5 and 10, and where in the order; both engines over the
  same clips, the split tested paired;
- beam sensitivity: the same clips again through a sibling model directory whose `conf/` asks
  for a wider beam and more active states, and the clips whose final moved, split by whether
  the move was toward the word or away from it, with the compute each search costs, a sweep
  over a range of beams beside it and a third sibling that shows the file is read at all;
- revision attribution: for every first word that was later revised, where the word that
  replaced it was at the sighting, read off the series `scripts/partial_trust.py` already
  recorded. No audio is decoded for this one.

Writes `<out>.md`, `<out>.json` and `<out>.clips.jsonl`. `<out>.reading.md` is appended to the
page, so what the figures mean survives the next run.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import speech_commands as sc  # noqa: E402

STOCK = "stock"
WIDE = "wide"


def read_conf(model_dir):
    """`conf/model.conf` as a dict of the `--key=value` lines it carries."""
    out = {}
    for line in (Path(model_dir) / "conf" / "model.conf").read_text().splitlines():
        line = line.strip()
        if line.startswith("--") and "=" in line:
            k, v = line[2:].split("=", 1)
            out[k.strip()] = v.strip()
    return out


def sibling_model(model_dir, dest, overrides):
    """A model directory that is the original except for `conf/model.conf`.

    The trick is the one `[[rr:TD-2#Inputs: configuration]]` describes for a host shipping its
    own chunk: the heavy parts are symlinked, so the sibling costs nothing and cannot drift
    from what it is a sibling of.
    """
    model_dir = Path(model_dir).resolve()
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for child in sorted(model_dir.iterdir()):
        if child.name != "conf":
            (dest / child.name).symlink_to(child)
    shutil.copytree(model_dir / "conf", dest / "conf")
    conf = dest / "conf" / "model.conf"
    kept = [line for line in conf.read_text().splitlines() if line.split("=", 1)[0].strip("- ") not in overrides]
    kept += [f"--{k}={v}" for k, v in overrides.items()]
    conf.write_text("\n".join(kept) + "\n")
    return dest


def vosk_decoding_params(module, model_dir):
    """What the wheel says it decodes with, from its own log line as it opens the model.

    libvosk parses `conf/model.conf` into the option objects it then decodes with and prints
    beam, max-active and lattice-beam at model load; capturing that line is the check that a
    sibling directory is honoured rather than assumed. The log is written by the C++ library to
    file descriptor 2, so the capture is at the descriptor, not at `sys.stderr`.
    """
    module.SetLogLevel(0)
    saved = os.dup(2)
    with tempfile.TemporaryFile() as tmp:
        os.dup2(tmp.fileno(), 2)
        try:
            module.Model(str(model_dir))
        finally:
            os.dup2(saved, 2)
            os.close(saved)
            module.SetLogLevel(-1)
        tmp.seek(0)
        text = tmp.read().decode("utf-8", "replace")
    for line in text.splitlines():
        if "Decoding params" in line:
            return line.split("Decoding params", 1)[1].strip()
    return None


def final_readings(finals):
    """The final's n-best as a list of word sequences, best first, and the words the engine read.

    `SetMaxAlternatives` replaces the plain `text` with an `alternatives` array; below 2 the
    shape is unchanged, so both are accepted. A clip the engine closed more than once has an
    n-best per segment: the words are every segment's best, as the accuracy pass reads them,
    and the ranked list is the last segment that carried a word, because a ranking across
    segments is not a thing the engine offered.
    """
    words = []
    ranked = []
    segments = 0
    for f in finals:
        alts = f.get("alternatives")
        if alts is None:
            alts = [{"text": f.get("text", "")}]
        best = sc.words_of(alts[0].get("text", ""))
        words.extend(best)
        if best:
            segments += 1
            ranked = [sc.words_of(a.get("text", "")) for a in alts]
    return words, ranked, segments


def sighting_readings(p):
    """The readings at a partial, best first, as word sequences."""
    alts = p.get("partial_alternatives") or []
    if alts:
        return [sc.words_of(a.get("text", "")) for a in alts]
    return [sc.words_of(p.get("partial", ""))]


def rank_of(ranked, hit):
    for i, words in enumerate(ranked):
        if hit(words):
            return i
    return None


def decode_clip(eng, pcm, block_ms, label, n):
    """One clip: the readings when its first word appeared, and the readings in its final."""
    sight = {}

    def on_partial(fed, p):
        if not sight and sc.words_of(p.get("partial", "")):
            sight["readings"] = sighting_readings(p)
            sight["ms"] = 1000.0 * fed / sc.RATE

    rec = eng.new(alternatives=n)
    rec.SetMaxAlternatives(n)
    finals = sc.feed(rec, pcm, block_ms, on_partial)
    words, ranked, segments = final_readings(finals)
    return dict(
        words=words,
        correct=words == [label],
        final_ranks=[" ".join(w) for w in ranked[:n]],
        final_rank=rank_of(ranked, lambda w: w == [label]),
        sighting_ranks=[" ".join(w) for w in sight.get("readings", [])[:n]],
        sighting_rank=rank_of(sight.get("readings", []), lambda w: w[:1] == [label]),
        sighting_ms=sight.get("ms"),
        segments=segments,
    )


def decode_pass(modules, model_dir, clips, block_ms, grammar, n, tag):
    """Every clip through every engine at one search setting, and whose partials carry a n-best."""
    out = {}
    nbest = {}
    for name, mod in modules.items():
        eng = sc.Engine(name, mod, model_dir, grammar)
        nbest[name] = hasattr(eng.new(), "SetPartialAlternatives")
        sc.note(f"{tag}: {name}, {len(clips)} clips")
        rows = {}
        for done, (label, path) in enumerate(clips):
            if done and done % 2500 == 0:
                sc.note(f"  {tag} {name} {done} / {len(clips)}")
            rows[str(path)] = decode_clip(eng, sc.read_pcm(path), block_ms, label, n)
        out[name] = rows
    return out, nbest


def oracle_counts(rows, key, ns):
    """How often the word was among the readings, at each n, and where it stood when it was."""
    rank = f"{key}_rank"
    out = {
        n: dict(
            present=sum(1 for r in rows.values() if r[rank] is not None and r[rank] < n),
            total=len(rows),
        )
        for n in ns
    }
    out["ranks"] = Counter(r[rank] for r in rows.values() if r[rank] is not None)
    return out


def oracle_section(results, nbest, ns, lines, report, key_prefix="oracle"):
    lines.append("## Oracle in the beam: was the word there at all")
    lines.append("")
    lines.append(
        "At the first partial that carried a word, and in the final, with n-best asked for on "
        "both. A reading counts at the sighting when the word leads it, because a partial is "
        "read for the word at its front; in the final it counts when the whole reading is the "
        "word, which is what the accuracy pass calls correct. An engine whose partial carries "
        "one reading has no rate above n = 1: readings on a partial are utter's, by "
        "`[[rr:TD-2#The decoder: partial alternatives]]`."
    )
    lines.append("")
    fig = {}
    for key, what in (("sighting", "at the first sighting"), ("final", "in the final")):
        limited = [name for name in results if key == "sighting" and not nbest.get(name, True)]
        lines.append(f"### The word among the readings {what}")
        lines.append("")
        lines.append(
            "| engine | " + " | ".join(f"n = {n}" for n in ns) + " | rank 0 | rank 1 | rank 2 and below | median rank |"
        )
        lines.append("|---|" + "---|" * (len(ns) + 4))
        for name, rows in results.items():
            counts = oracle_counts(rows, key, ns)
            one = name in limited
            cells = [
                "-"
                if one and n > 1
                else f"{counts[n]['present']} / {counts[n]['total']} ({sc.pct(counts[n]['present'], counts[n]['total']):.1f}%)"
                for n in ns
            ]
            r = counts["ranks"]
            tail = sum(v for k, v in r.items() if k >= 2)
            depth = ["-", "-"] if one else [str(r.get(1, 0)), str(tail)]
            median = sc.quantile(list(r.elements()), 0.5)
            lines.append(
                f"| {name} | "
                + " | ".join(cells)
                + f" | {r.get(0, 0)} | "
                + " | ".join(depth)
                + (" | - |" if one else f" | {median} |")
            )
            fig.setdefault(key, {})[name] = dict(
                at={str(n): counts[n] for n in ns if not (one and n > 1)},
                rank0=r.get(0, 0),
                rank1=None if one else r.get(1, 0),
                rank_2_and_below=None if one else tail,
                median_rank=None if one else median,
                one_reading_only=one,
            )
        lines.append("")
        names = list(results)
        if len(names) == 2:
            a, b = (results[n] for n in names)
            shared = sorted(set(a) & set(b))
            paired_ns = [n for n in ns if not (limited and n > 1)]
            lines.append("| n | " + " | ".join(names) + f" | only {names[0]} | only {names[1]} | McNemar p |")
            lines.append("|---|" + "---|" * (len(names) + 3))
            for n in paired_ns:
                ok_a = {c for c in shared if a[c][f"{key}_rank"] is not None and a[c][f"{key}_rank"] < n}
                ok_b = {c for c in shared if b[c][f"{key}_rank"] is not None and b[c][f"{key}_rank"] < n}
                only_a, only_b = len(ok_a - ok_b), len(ok_b - ok_a)
                p = sc.mcnemar(only_a, only_b)
                lines.append(f"| {n} | {len(ok_a)} | {len(ok_b)} | {only_a} | {only_b} | {p:.3g} |")
                fig.setdefault("paired", {})[f"{key}_{n}"] = dict(
                    both=len(ok_a & ok_b),
                    only_a=only_a,
                    only_b=only_b,
                    p=p,
                    engines=names,
                    clips=len(shared),
                )
            lines.append("")
            note = (
                f" Only n = 1 is paired here, because {' and '.join(limited)} offers one reading on a partial."
                if limited
                else ""
            )
            lines.append(
                f"Paired over the {len(shared)} clips both engines read, with an exact McNemar test on "
                f"the clips they differ about.{note}"
            )
            lines.append("")
    report[key_prefix] = fig


def finals_at(eng, clips, block_ms, n):
    return [decode_clip(eng, sc.read_pcm(p), block_ms, label, n)["words"] for label, p in clips]


def moved_against(base, got):
    return sum(1 for a, b in zip(base, got) if a != b)


def conf_read_check(modules, model_dir, probe_model, scale, clips, block_ms, grammar, n, lines, report):
    """That a sibling directory's `conf/` reaches the decoder, shown rather than assumed.

    A beam that moves no final cannot tell a file that was read from a file that was ignored, so
    the check is a key whose effect is not in doubt: the acoustic scale weighs every acoustic
    cost in the search, and an engine reading the file cannot return the same finals under it.
    """
    lines.append("### The sibling directory is read")
    lines.append("")
    lines.append(
        f"A sibling built the same way but with `acoustic-scale` {scale['from']} to {scale['to']}, over "
        f"{len(clips)} clips. The finals have to move; an engine whose finals stood would be one not reading "
        "the file, and the beam figures above would then be measuring nothing."
    )
    lines.append("")
    lines.append("| engine | finals changed against stock |")
    lines.append("|---|---|")
    fig = {}
    for name, mod in modules.items():
        base = finals_at(sc.Engine(name, mod, model_dir, grammar), clips, block_ms, n)
        got = finals_at(sc.Engine(name, mod, probe_model, grammar), clips, block_ms, n)
        moved = moved_against(base, got)
        lines.append(f"| {name} | {moved} / {len(clips)} ({sc.pct(moved, len(clips)):.0f}%) |")
        fig[name] = dict(changed=moved, clips=len(clips))
    lines.append("")
    report["conf_read"] = dict(acoustic_scale=scale, engines=fig)


def beam_sweep_section(
    modules,
    model_dir,
    work,
    beams,
    max_active,
    clips,
    block_ms,
    grammar,
    n,
    lines,
    report,
):
    """The same clips at a range of beams, so one wider point is not read as the whole story."""
    lines.append("### The finals over a range of beams")
    lines.append("")
    lines.append(
        f"Every beam over the same {len(clips)} clips, each in its own sibling directory at `max-active` "
        f"{max_active}, counted against the stock search."
    )
    lines.append("")
    lines.append("| engine | " + " | ".join(f"beam {b:g}" for b in beams) + " |")
    lines.append("|---|" + "---|" * len(beams))
    fig = {}
    siblings = [
        (
            b,
            sibling_model(model_dir, work / f"beam-{b:g}", {"beam": b, "max-active": max_active}),
        )
        for b in beams
    ]
    for name, mod in modules.items():
        base = finals_at(sc.Engine(name, mod, model_dir, grammar), clips, block_ms, n)
        cells = []
        for b, sib in siblings:
            moved = moved_against(base, finals_at(sc.Engine(name, mod, sib, grammar), clips, block_ms, n))
            cells.append(f"{moved} / {len(clips)}")
            fig.setdefault(name, {})[f"{b:g}"] = moved
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    report["beam_sweep"] = dict(
        beams=[float(b) for b in beams],
        max_active=max_active,
        clips=len(clips),
        engines=fig,
    )


def beam_section(stock, wide, conf, params, compute, lines, report):
    lines.append("## Beam sensitivity: the clips a wider search moves")
    lines.append("")
    lines.append(
        "The same clips through a sibling model directory: `am/`, `graph/` and `ivector/` are "
        "symlinks to the stock model, `conf/` is a copy with two lines changed. The model's own "
        "`conf/model.conf` carries "
        + " and ".join(f"`{k}` {conf['from'][k]}" for k in conf["to"])
        + "; the sibling asks for "
        + " and ".join(f"{conf['to'][k]}" for k in conf["to"])
        + " instead. Nothing under `src/` differs between the two columns."
    )
    lines.append("")
    for name, line in params.items():
        lines.append(
            f"- {name} reports at model load: `{line}`"
            if line
            else f"- {name} prints no decoding parameters at model load"
        )
    lines.append("")
    lines.append(
        "| engine | final changed | became right | became wrong | wrong to wrong | accuracy stock | accuracy wide |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    fig = {}
    for name in stock:
        a, b = stock[name], wide[name]
        shared = sorted(set(a) & set(b))
        moved = [c for c in shared if a[c]["words"] != b[c]["words"]]
        became_right = sum(1 for c in moved if b[c]["correct"] and not a[c]["correct"])
        became_wrong = sum(1 for c in moved if a[c]["correct"] and not b[c]["correct"])
        wrong_wrong = sum(1 for c in moved if not a[c]["correct"] and not b[c]["correct"])
        ok_a = sum(1 for c in shared if a[c]["correct"])
        ok_b = sum(1 for c in shared if b[c]["correct"])
        lines.append(
            f"| {name} | {len(moved)} / {len(shared)} ({sc.pct(len(moved), len(shared)):.2f}%) | {became_right} | "
            f"{became_wrong} | {wrong_wrong} | {ok_a} / {len(shared)} ({sc.pct(ok_a, len(shared)):.2f}%) | "
            f"{ok_b} / {len(shared)} ({sc.pct(ok_b, len(shared)):.2f}%) |"
        )
        fig[name] = dict(
            clips=len(shared),
            changed=len(moved),
            became_right=became_right,
            became_wrong=became_wrong,
            wrong_to_wrong=wrong_wrong,
            correct_stock=ok_a,
            correct_wide=ok_b,
            mcnemar=sc.mcnemar(
                sum(1 for c in shared if a[c]["correct"] and not b[c]["correct"]),
                sum(1 for c in shared if b[c]["correct"] and not a[c]["correct"]),
            ),
        )
    lines.append("")
    lines.append("A changed final is the search reaching a different answer, not the search having erred before.")
    lines.append("")
    lines.append("### What the wider search costs")
    lines.append("")
    lines.append("| engine | RTF stock | RTF wide | per-block ms p50 stock / wide | p95 stock / wide |")
    lines.append("|---|---|---|---|---|")
    for name in stock:
        s, w = compute[STOCK].get(name), compute[WIDE].get(name)
        if not s or not w:
            continue
        lines.append(
            f"| {name} | {s['rtf']:.4f} | {w['rtf']:.4f} | {s['ms_p50']:.3f} / {w['ms_p50']:.3f} | "
            f"{s['ms_p95']:.2f} / {w['ms_p95']:.2f} |"
        )
    lines.append("")
    lines.append(
        f"One recognizer over {compute[STOCK][list(stock)[0]]['seconds']:.0f} s of clips joined end to end at "
        "each setting, the same pass the Speech Commands page reports compute with."
    )
    lines.append("")
    report["beam"] = dict(clips=fig, conf=conf, vosk_params=params, compute=compute)


def wide_oracle_section(fig, ns, lines):
    """The oracle rate again with the search widened, which is what the beam is a lever over."""
    top, low = str(max(ns)), str(min(ns))
    lines.append("### The word among the readings at the wider search")
    lines.append("")
    lines.append(f"| engine | sighting, n = {max(ns)} | final, n = {min(ns)} | final, n = {max(ns)} |")
    lines.append("|---|---|---|---|")

    def cell(at, n):
        if n not in at:
            return "-"
        c = at[n]
        return f"{c['present']} / {c['total']} ({sc.pct(c['present'], c['total']):.1f}%)"

    for name in fig["final"]:
        lines.append(
            f"| {name} | {cell(fig['sighting'][name]['at'], top)} | {cell(fig['final'][name]['at'], low)} | "
            f"{cell(fig['final'][name]['at'], top)} |"
        )
    lines.append("")


def attribution(series_path, lines, report):
    """Where the replacing word stood when the word it replaced was first shown.

    Reads the series `scripts/partial_trust.py` recorded and decodes nothing. In that file a
    record's `word` is the first word a host was shown and `survived` says whether the final
    still led with it, so a revision is a record with `survived` false.
    """
    revised = []
    consistent = 0
    total = 0
    for line in Path(series_path).read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        total += 1
        series = rec["series"]
        settled = series[-1]["top_word"]
        consistent += (settled == rec["word"]) == bool(rec["survived"])
        if not rec["survived"]:
            revised.append((rec, settled))

    def leads(entry):
        return [r["text"].split()[0] for r in entry["readings"] if r["text"].split()]

    buckets = Counter()
    truth_right = Counter()
    alone = [r for r, _ in revised if len(r["series"]) == 1]
    followed = [(r, s) for r, s in revised if len(r["series"]) > 1]
    for rec, settled in followed:
        series = rec["series"]
        at_sighting = leads(series[0])
        if settled is None:
            name = "no word led the last advance"
        elif series[1]["top_word"] == settled:
            name = "the next advance settled it"
        elif settled in at_sighting[1:]:
            name = "present at the sighting, ranked below"
        elif settled not in at_sighting and any(settled in leads(e) for e in series[1:]):
            name = "absent at the sighting, arrived later"
        else:
            name = "the rest"
        buckets[name] += 1
        truth_right[name] += settled == rec["clip"].split("/")[0]
    order = [
        "the next advance settled it",
        "present at the sighting, ranked below",
        "absent at the sighting, arrived later",
        "the rest",
        "no word led the last advance",
    ]
    n = len(revised)
    m = len(followed)
    lines.append("## Where the replacing word was when the first word appeared")
    lines.append("")
    lines.append(
        f"Every first word that was later revised, from `{Path(series_path).name}`: {n} of {total} sightings "
        f"({sc.pct(n, total):.1f}%). No audio is decoded. On {len(alone)} of them ({sc.pct(len(alone), n):.1f}%) "
        "the sighting was the clip's last advance, so the word was replaced by the final with no partial in "
        f"between and the series holds nothing to place it by. The other {m} split as follows, the replacing "
        "word being the word leading the clip's last advance and the buckets tried in order."
    )
    lines.append("")
    lines.append(
        "| where the replacing word was | revisions | share of the " + str(m) + " | it was the clip's own word |"
    )
    lines.append("|---|---|---|---|")
    for name in order:
        if not buckets[name]:
            continue
        lines.append(
            f"| {name} | {buckets[name]} | {sc.pct(buckets[name], m):.1f}% | "
            f"{truth_right[name]} / {buckets[name]} ({sc.pct(truth_right[name], buckets[name]):.0f}%) |"
        )
    lines.append("")
    lines.append(
        f"The last advance's leader and the emitted final agree about the first word on "
        f"{consistent} of {total} sightings ({sc.pct(consistent, total):.2f}%); where they do not, a bucket "
        "is about the last partial a host saw and not about the final."
    )
    lines.append("")
    report["attribution"] = dict(
        source=Path(series_path).name,
        sightings=total,
        revisions=n,
        no_advance_followed=len(alone),
        with_an_advance=m,
        buckets={k: buckets[k] for k in order if buckets[k]},
        settled_was_the_clips_word={k: truth_right[k] for k in order if buckets[k]},
        last_advance_agrees_with_final=consistent,
    )


def asking_for_alternatives_check(stock, compare, lines, report):
    """Whether asking for an n-best moved the answer the engine gives without one.

    The figures above are only about the Speech Commands page's if the rank-0 reading is the
    final that page recorded, and an engine is free to reach the n-best a different way.
    """
    path = Path(compare)
    if not path.exists():
        return
    other = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            other[row["clip"]] = row
    lines.append("### The n-best did not move the answer")
    lines.append("")
    lines.append(f"Rank 0 at the stock search against the finals `{path.name}` carries, over the clips in both.")
    lines.append("")
    lines.append("| engine | rank 0 equals the recorded final |")
    lines.append("|---|---|")
    fig = {}
    for name, rows in stock.items():
        both = [(r, other[c][name]) for c, r in rows.items() if c in other and name in other[c]]
        same = sum(1 for mine, theirs in both if mine["words"] == theirs["words"])
        lines.append(f"| {name} | {same} / {len(both)} ({sc.pct(same, len(both)):.2f}%) |")
        fig[name] = dict(same=same, clips=len(both))
    lines.append("")
    report["nbest_check"] = dict(against=path.name, engines=fig)


def write_clips(path, clips, stock, wide):
    with open(path, "w") as f:
        for label, clip in clips:
            row = {"clip": f"{clip.parent.name}/{clip.name}", "label": label}
            for tag, res in ((STOCK, stock), (WIDE, wide)):
                for name, rows in res.items():
                    r = rows.get(str(clip))
                    if r is not None:
                        row[f"{name}.{tag}"] = r
            f.write(json.dumps(row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="output prefix; writes .md, .json and .clips.jsonl")
    ap.add_argument("--split", default="testing", choices=["testing", "validation"])
    ap.add_argument("--limit", type=int, default=0, help="clips per word, 0 for all")
    ap.add_argument("--block-ms", type=int, default=40)
    ap.add_argument("--engines", default="vosk,utterpy")
    ap.add_argument(
        "--alternatives",
        type=int,
        default=10,
        help="readings asked for on the partial and the final",
    )
    ap.add_argument("--oracle-n", default="1,5,10", help="the n the oracle rate is reported at")
    ap.add_argument("--wide-beam", type=float, default=18.0)
    ap.add_argument("--wide-max-active", type=int, default=20000)
    ap.add_argument(
        "--probe-acoustic-scale",
        type=float,
        default=0.1,
        help="the sibling that shows a conf is read",
    )
    ap.add_argument("--probe-clips", type=int, default=200, help="clips decoded for that check")
    ap.add_argument(
        "--beam-sweep",
        default="0.5,2,5,13,18",
        help="beams swept beside the stock one, empty to skip",
    )
    ap.add_argument("--sweep-clips", type=int, default=800, help="clips per swept beam")
    ap.add_argument(
        "--steady-state-clips",
        type=int,
        default=300,
        help="clips joined for the compute pass",
    )
    ap.add_argument("--series", default="docs/benchmarks/partial-trust.clips.jsonl")
    ap.add_argument("--compare", default="docs/benchmarks/speech-commands.clips.jsonl")
    ap.add_argument(
        "--work",
        default="",
        help="where the sibling model directory is built; a temporary one by default",
    )
    args = ap.parse_args()

    data = Path(args.data)
    listed = [line.strip() for line in (data / f"{args.split}_list.txt").read_text().splitlines() if line.strip()]
    by_word = {}
    for rel in listed:
        by_word.setdefault(rel.split("/")[0], []).append(rel)
    clips = []
    for w in sc.DATASET_WORDS:
        items = by_word.get(w, [])
        if args.limit:
            items = items[: args.limit]
        clips.extend((w, data / rel) for rel in items)

    modules = {}
    for name in args.engines.split(","):
        modules[name] = __import__(name)
        if name == "vosk":
            modules[name].SetLogLevel(-1)

    grammar = sc.DATASET_WORDS + sc.LETTERS + sc.NATO + sc.COLOURS
    ns = [int(v) for v in args.oracle_n.split(",") if v.strip()]
    prov = sc.provenance(args, modules)
    report = {
        "split": args.split,
        "clips": len(clips),
        "block_ms": args.block_ms,
        "alternatives": args.alternatives,
        "grammar_size": len(grammar),
        "out": args.out,
        "provenance": prov,
    }

    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="decoder-diagnostics-"))
    work.mkdir(parents=True, exist_ok=True)
    stock_conf = read_conf(args.model)
    overrides = {"beam": args.wide_beam, "max-active": args.wide_max_active}
    wide_model = sibling_model(args.model, work / (Path(args.model).name + "-wide-beam"), overrides)
    conf = dict(
        from_={k: stock_conf.get(k) for k in overrides},
        to={k: str(v) for k, v in overrides.items()},
        stock=stock_conf,
        wide=read_conf(wide_model),
    )
    conf["from"] = conf.pop("from_")
    scale = dict(**{"from": stock_conf.get("acoustic-scale")}, to=str(args.probe_acoustic_scale))
    probe_model = sibling_model(
        args.model,
        work / (Path(args.model).name + "-scale"),
        {"acoustic-scale": scale["to"]},
    )
    params = {}
    if "vosk" in modules:
        params["vosk stock"] = vosk_decoding_params(modules["vosk"], args.model)
        params["vosk wide"] = vosk_decoding_params(modules["vosk"], wide_model)

    lines = [
        f"# Decoder diagnostics, {prov['model']}, {args.split} split, {len(clips)} clips, {args.block_ms} ms blocks",
        "",
        f"Grammar: the dataset's 35 words plus {len(sc.LETTERS)} letters, {len(sc.NATO)} NATO words and "
        f"{len(sc.COLOURS)} colours, {len(grammar)} entries. {args.alternatives} readings asked for on the "
        "partial and on the final.",
        "",
        sc.provenance_line(prov),
        "",
    ]

    sc.preflight(modules, args.model, grammar, args.block_ms, clips, lines, report)
    stock, nbest = decode_pass(modules, args.model, clips, args.block_ms, grammar, args.alternatives, STOCK)
    oracle_section(stock, nbest, ns, lines, report)
    asking_for_alternatives_check(stock, args.compare, lines, report)
    wide, _ = decode_pass(modules, wide_model, clips, args.block_ms, grammar, args.alternatives, WIDE)
    wide_report = {}
    oracle_section(wide, nbest, ns, [], wide_report, key_prefix="oracle_wide")
    report["oracle_wide"] = wide_report["oracle_wide"]

    compute = {}
    for tag, model_dir in ((STOCK, args.model), (WIDE, wide_model)):
        held = {}
        sc.steady_state_pass(
            modules,
            model_dir,
            clips,
            args.block_ms,
            grammar,
            [],
            held,
            args.steady_state_clips,
        )
        compute[tag] = held.get("steady_state", {})
    beam_section(stock, wide, conf, params, compute, lines, report)
    beams = [float(v) for v in args.beam_sweep.split(",") if v.strip()]
    if beams:
        beam_sweep_section(
            modules,
            args.model,
            work,
            beams,
            args.wide_max_active,
            clips[:: max(1, len(clips) // args.sweep_clips)][: args.sweep_clips],
            args.block_ms,
            grammar,
            args.alternatives,
            lines,
            report,
        )
    conf_read_check(
        modules,
        args.model,
        probe_model,
        scale,
        clips[: args.probe_clips],
        args.block_ms,
        grammar,
        args.alternatives,
        lines,
        report,
    )

    wide_oracle_section(report["oracle_wide"], ns, lines)

    attribution(args.series, lines, report)

    write_clips(args.out + ".clips.jsonl", clips, stock, wide)
    # [[rr:docs/benchmarks/README.md#Benchmarks]]
    reading = Path(args.out + ".reading.md")
    if reading.exists():
        lines.append(reading.read_text().strip())
        lines.append("")
    text = "\n".join(lines) + "\n"
    Path(args.out + ".md").write_text(text)
    Path(args.out + ".json").write_text(json.dumps(report, indent=1))
    if not args.work:
        shutil.rmtree(work, ignore_errors=True)
    sc.note(f"written: {args.out}.md, .json, .clips.jsonl")
    print(text)


if __name__ == "__main__":
    main()
