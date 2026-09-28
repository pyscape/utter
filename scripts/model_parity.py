#!/usr/bin/env python3
"""Small models other than English against the stock vosk wheel, on Common Voice's Single Word
Target segment (CC0), scored by the thresholds G2 and G5 apply to the English model.

    python scripts/model_parity.py --corpus CV_DIR --wav-cache DIR \\
        --model de=MODEL_DIR --model fr=MODEL_DIR ... --out docs/gates/small-models

CV_DIR is the extracted cv-corpus-7.0-singleword; a language's validated.tsv lists the clips.
Its mp3 clips are converted once by sox into DIR. A take is one speaker's clips back to back,
fed in blocks to both engines through one model directory whose mfcc.conf carries --dither=0,
as the G2 oracle is run. The grammar is the language's distinct
prompts that the model's vocabulary holds.

Writes `<out>.md` and `<out>.json`, and per-take readings to `<clips-dir>/<lang>.jsonl`.
`<out>.md` is regenerated every run; `<out>.reading.md`, if present, is appended to it.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from multiprocessing import Pool
from pathlib import Path
from types import ModuleType
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g0 import dither0_model  # noqa: E402
from g2 import align_ops  # noqa: E402
from speech_commands import (  # noqa: E402
    RATE,
    Engine,
    Json,
    feed,
    note,
    pct,
    provenance,
    provenance_line,
    read_pcm,
    words_of,
    write_page,
)

ENGINES = ("vosk", "utterpy")
# G2's segment figure for vosk-model-small-en-us-0.15, docs/gates/g2-g5-streaming.md.
ENGLISH_SEGMENTS = (193, 205)
GATE = 95.0


class Tracked:
    """A recognizer that notes the fed sample at which each final closed, which feed() does not
    return and G2's segment and endpoint pairing is by."""

    def __init__(self, rec: Any) -> None:
        self.rec = rec
        self.fed = 0
        self.ends: list[int] = []

    def AcceptWaveform(self, chunk: bytes) -> bool:  # noqa: N802 - the wheel's spelling
        self.fed += len(chunk) // 2
        closed = bool(self.rec.AcceptWaveform(chunk))
        if closed:
            self.ends.append(self.fed)
        return closed

    def __getattr__(self, name: str) -> Any:
        return getattr(self.rec, name)


Prompted = tuple[str, str]
Take = tuple[str, list[Prompted]]


def prompts(corpus: Path, lang: str) -> list[tuple[str, str, str]]:
    """(speaker, clip stem, prompt) for every validated clip."""
    with open(corpus / lang / "validated.tsv", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    out = []
    for r in rows:
        text = r["sentence"].strip().lower()
        # German "null" was exported as a missing value and reads "nan" in the release's TSV.
        if lang == "de" and text == "nan":
            text = "null"
        out.append((r["client_id"], r["path"].removesuffix(".mp3"), text))
    return out


def takes(rows: Sequence[tuple[str, str, str]], per_take: int) -> list[Take]:
    """One speaker's clips back to back, per_take at a time, in the release's order: the pauses
    between words are the recordings' own, which a segment has to close on."""
    by_speaker: dict[str, list[Prompted]] = {}
    for speaker, stem, text in rows:
        by_speaker.setdefault(speaker, []).append((stem, text))
    out = []
    for clips in by_speaker.values():
        for i in range(0, len(clips), per_take):
            out.append((clips[i][0], clips[i : i + per_take]))
    return out


def wav_of(corpus: Path, cache: Path, lang: str, stem: str) -> Path:
    wav = cache / lang / f"{stem}.wav"
    if not wav.exists():
        wav.parent.mkdir(parents=True, exist_ok=True)
        part = wav.with_suffix(".part.wav")
        # -D: sox dithers by default when it narrows to 16 bits, and the cache should be reproducible.
        subprocess.run(
            [
                "sox",
                "-D",
                str(corpus / lang / "clips" / f"{stem}.mp3"),
                "-r",
                str(RATE),
                "-c",
                "1",
                "-b",
                "16",
                "-e",
                "signed",
                str(part),
            ],
            check=True,
        )
        part.rename(wav)
    return wav


STATE: dict[str, Any] = {}


def worker_init(model_dir: str, grammar: Sequence[str], corpus: str, cache: str, lang: str, block_ms: int) -> None:
    modules: dict[str, ModuleType] = {}
    for name in ENGINES:
        modules[name] = __import__(name)
    modules["vosk"].SetLogLevel(-1)
    engines = {n: Engine(n, m, model_dir, grammar) for n, m in modules.items()}
    STATE.update(
        engines=engines,
        corpus=Path(corpus),
        cache=Path(cache),
        lang=lang,
        block_ms=block_ms,
    )


def decode(job: Take) -> Json:
    take, clips = job
    pcm = b"".join(read_pcm(wav_of(STATE["corpus"], STATE["cache"], STATE["lang"], stem)) for stem, _ in clips)
    row: Json = {
        "take": take,
        "clips": [c for c, _ in clips],
        "prompts": [t for _, t in clips],
        "samples": len(pcm) // 2,
    }
    for name, eng in STATE["engines"].items():
        rec = Tracked(eng.new())
        partials: dict[int, list[str]] = {}

        def keep(fed: int, p: Json, into: dict[int, list[str]] = partials) -> None:
            into[fed] = words_of(p.get("partial", ""))

        finals = feed(rec, pcm, STATE["block_ms"], keep)
        ends = rec.ends + [rec.fed]
        row[name] = {
            "partials": {str(k): v for k, v in partials.items()},
            "segments": [{"end_sample": e, "result": f} for e, f in zip(ends, finals, strict=True)],
        }
    return row


def score(rows: Sequence[Json], block_ms: int) -> Json:
    """G2's rules, as scripts/g2.py score applies them to the English takes."""
    step = RATE * block_ms // 1000
    s: dict[str, int] = dict.fromkeys(
        [
            "blocks",
            "blocks_equal",
            "segments_vosk",
            "segments_utter",
            "segments_paired",
            "segments_equal",
            "vosk_words",
            "only_vosk",
            "only_utter",
            "substitutions",
            "word_pairs",
            "words_within",
            "endpoints_vosk",
            "endpoints_utter",
            "endpoints_within",
            "prompt_errors_vosk",
            "prompt_errors_utter",
            "prompts",
            "takes",
        ],
        0,
    )
    for row in rows:
        v, u = row["vosk"], row["utterpy"]
        s["takes"] += 1
        s["prompts"] += len(row["prompts"])
        for k, a in v["partials"].items():
            b = u["partials"].get(k)
            if b is not None:
                s["blocks"] += 1
                s["blocks_equal"] += a == b
        rs, hs = v["segments"], u["segments"]
        s["segments_vosk"] += len(rs)
        s["segments_utter"] += len(hs)
        used: set[int] = set()
        for seg in rs:
            best: int | None = None
            for k, t in enumerate(hs):
                gap = abs(t["end_sample"] - seg["end_sample"])
                if (
                    k not in used
                    and gap <= step * 5
                    and (best is None or gap < abs(hs[best]["end_sample"] - seg["end_sample"]))
                ):
                    best = k
            rw = words_of(seg["result"].get("text", ""))
            s["vosk_words"] += len(rw)
            if best is None:
                s["only_vosk"] += len(rw)
                continue
            used.add(best)
            s["segments_paired"] += 1
            hw = words_of(hs[best]["result"].get("text", ""))
            dele, ins, sub = align_ops(rw, hw)
            s["only_vosk"] += dele
            s["only_utter"] += ins
            s["substitutions"] += sub
            if rw == hw:
                s["segments_equal"] += 1
                for a, b in zip(seg["result"].get("result", []), hs[best]["result"].get("result", []), strict=False):
                    s["word_pairs"] += 1
                    s["words_within"] += (
                        abs(a["start"] - b["start"]) <= 0.03 + 1e-6 and abs(a["end"] - b["end"]) <= 0.03 + 1e-6
                    )
        s["only_utter"] += sum(len(words_of(t["result"].get("text", ""))) for k, t in enumerate(hs) if k not in used)
        re_ = [x["end_sample"] for x in rs[:-1]]
        he = [x["end_sample"] for x in hs[:-1]]
        s["endpoints_vosk"] += len(re_)
        s["endpoints_utter"] += len(he)
        for e in re_:
            near = [abs(x - e) for x in he if abs(x - e) <= step * 5]
            s["endpoints_within"] += bool(near) and min(near) <= RATE // 5
        for name, key in (("vosk", "prompt_errors_vosk"), ("utterpy", "prompt_errors_utter")):
            got = [w for seg in row[name]["segments"] for w in words_of(seg["result"].get("text", ""))]
            s[key] += sum(align_ops(row["prompts"], got))
    return s


def run_language(args: argparse.Namespace, lang: str, model: Path) -> Json:
    all_prompts = prompts(Path(args.corpus), lang)
    import vosk

    vosk.SetLogLevel(-1)
    vocab = Engine("vosk", vosk, model, sorted({t for _, _, t in all_prompts}))
    grammar = sorted({t for _, _, t in all_prompts} - set(vocab.missing))
    jobs = takes(all_prompts[: args.limit] if args.limit else all_prompts, args.per_take)
    sibling = dither0_model(model)
    note(f"{lang}: {len(jobs)} takes, grammar {grammar}, not in the model: {sorted(set(vocab.missing))}")
    rows: list[Json] = []
    with Pool(
        args.workers, worker_init, (str(sibling), grammar, args.corpus, args.wav_cache, lang, args.block_ms)
    ) as pool:
        for i, r in enumerate(pool.imap(decode, jobs)):
            rows.append(r)
            if i % 200 == 0:
                note(f"{lang}: {i}/{len(jobs)}")
    clips_dir = Path(args.clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)
    with open(clips_dir / f"{lang}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    s = score(rows, args.block_ms)
    s["audio_s"] = round(sum(r["samples"] for r in rows) / RATE, 1)
    return {"model": model.name, "grammar": grammar, "not_in_model": sorted(set(vocab.missing)), "figures": s}


def verdict(value: float) -> str:
    return "pass" if value >= GATE else "**fail**"


def page(results: Mapping[str, Json], prov: Mapping[str, Any], args: argparse.Namespace) -> list[str]:
    en = pct(*ENGLISH_SEGMENTS)
    lines = [
        "# Small models against the stock wheel",
        "",
        "Gates: `[[rr:TD-2#Verification and acceptance]]`, the G2 and G5 bars as "
        "`g2-g5-streaming.md` applies them to the English model, on each language's small model.",
        "",
        "## What ran",
        "",
        f"- {provenance_line(prov)}",
        "- Models: the latest small model per language from alphacephei (Apache 2.0), stock "
        "configuration, both engines reading one copy whose `mfcc.conf` adds `--dither=0`.",
        "- Corpus: Common Voice 7.0, Single Word Target segment (CC0), every validated clip of the "
        "language, converted from mp3 to 16 kHz mono by sox without dither. A take is up to "
        f"{args.per_take} of one speaker's clips back to back, so a segment closes on the pauses the "
        "recordings carry.",
        "- Grammar: the language's distinct prompts (the digits, yes, no, and the segment's "
        "wake words) that the model's vocabulary holds, no unknown-word symbol.",
        f"- {args.block_ms} ms blocks, the partial read after every block, `Result` on every "
        "endpoint, `FinalResult` at the end, scored by the rules of `scripts/g2.py score`.",
        "",
        "## Figures",
        "",
        "Segment equality carries no bar here: it is set beside the English model's "
        f"{ENGLISH_SEGMENTS[0]} / {ENGLISH_SEGMENTS[1]} ({en:.2f}%), which misses G2's 99% itself. "
        f"The other three rows are pass or fail at {GATE:.0f}%.",
        "",
        "| | " + " | ".join(results) + " |",
        "|---|" + "---|" * len(results),
    ]

    def row(title: str, cell: Any) -> None:
        lines.append(f"| {title} | " + " | ".join(cell(r["figures"], r) for r in results.values()) + " |")

    def ratio(a: str, b: str, gate: bool) -> Any:
        def cell(f: Json, _: Json) -> str:
            p = pct(f[a], f[b])
            return f"{f[a]:,} / {f[b]:,} ({p:.2f}%)" + (f", {verdict(p)}" if gate else "")

        return cell

    row("model", lambda f, r: r["model"])
    row("takes, clips, audio", lambda f, r: f"{f['takes']:,}, {f['prompts']:,}, {f['audio_s'] / 3600:.1f} h")
    row("grammar words", lambda f, r: str(len(r["grammar"])))
    row("not in the model", lambda f, r: ", ".join(r["not_in_model"]) or "none")
    row("partial text equal per block", ratio("blocks_equal", "blocks", True))
    row("segment word sequence equal", ratio("segments_equal", "segments_vosk", False))
    row("word times within 30 ms, equal segments", ratio("words_within", "word_pairs", True))
    row("endpoints within 0.2 s", ratio("endpoints_within", "endpoints_vosk", True))
    row("segments, vosk / utterpy", lambda f, r: f"{f['segments_vosk']:,} / {f['segments_utter']:,}")
    row(
        "words only in vosk / only in utterpy / substituted",
        lambda f, r: f"{f['only_vosk']} / {f['only_utter']} / {f['substitutions']}",
    )
    row(
        "word error against the prompts, vosk / utterpy",
        lambda f, r: (
            f"{pct(f['prompt_errors_vosk'], f['prompts']):.2f}% / {pct(f['prompt_errors_utter'], f['prompts']):.2f}%"
        ),
    )
    lines.append("")
    reading = Path(args.out + ".reading.md")
    if reading.exists():
        lines.append(reading.read_text().strip())
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="the extracted cv-corpus-7.0-singleword")
    ap.add_argument("--wav-cache", required=True)
    ap.add_argument("--clips-dir", required=True, help="where per-clip readings go")
    ap.add_argument("--model", action="append", required=True, help="LANG=MODEL_DIR, repeated")
    ap.add_argument("--out", required=True, help="output prefix; writes .md and .json")
    ap.add_argument("--block-ms", type=int, default=40)
    ap.add_argument("--per-take", type=int, default=20, help="clips joined into one take")
    ap.add_argument("--limit", type=int, default=0, help="clips per language, 0 for all")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()
    models = dict(m.split("=", 1) for m in args.model)
    results = {lang: run_language(args, lang, Path(d)) for lang, d in models.items()}
    args.model = ", ".join(Path(d).name for d in models.values())
    prov = provenance(args, ENGINES)
    report = {
        "provenance": prov,
        "english_segments": ENGLISH_SEGMENTS,
        "gate_percent": GATE,
        "per_take": args.per_take,
        "languages": results,
    }
    Path(args.out + ".json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(write_page(args.out + ".md", page(results, prov, args)))
    note("done")


if __name__ == "__main__":
    main()
