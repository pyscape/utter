#!/usr/bin/env python3
"""Speaker evidence away from the speaker's own device: TitaNet-small through utter's stateless
embed on words in a noisy room, and on words from speakers who all share one microphone.

    python scripts/speaker_conditions.py noise --model MODEL_DIR [--data DIR] \\
        [--lib target/release/libutter.so] [--out docs/benchmarks/speaker-noise]
    python scripts/speaker_conditions.py vctk --model MODEL_DIR [--vctk ZIP] \\
        [--lib target/release/libutter.so] [--out docs/benchmarks/speaker-vctk]

MODEL_DIR is the stock small English model; utter's recognizer finds the word spans, and
utter_spk_model_embed embeds them with TitaNet-small as scripts/titanet_convert.py writes it.

noise: the Speech Commands v2 (CC BY 4.0) test split as scripts/speaker_evidence.py draws it.
Each speaker enrols from --enroll clean words; the probe words are scored clean, then with one of
the dataset's _background_noise_ recordings laid under them at each SNR in SNRS_DB, the SNR taken
over the span embedded. Thresholds are the clean row's at the same length: a host sets its gate
in the quiet and meets the room afterwards.

vctk: the CSTR VCTK Corpus 0.92 (CC BY 4.0), read from its DataShare zip, the DPA 4035 channel
(mic1) at 48 kHz resampled to 16 kHz by sox. Each utterance is decoded under a grammar of its own
prompt's words, so the recognizer places the words it was told were said; one word of at least
--min-word-ms is taken per utterance. Each speaker enrols from --enroll such words from as many
utterances, and --probes words from other utterances are probes. Thresholds are the set's own.

In both, a probe is scored by cosine against every enrolled profile. Beside single words, each
speaker's probe words are joined, --streams times in shuffled orders, and cut at each length in
POOLED_MS, which is the evidence a host has once that much of a move has been spoken. Per row:

- equal error rate over own-profile against other-profile scores, among probes with evidence;
- accepted at 1% false accept: probes whose own score clears the threshold that lets 1% of
  other-profile scores through;
- can't tell: scores above the threshold that turns away 1% of own-profile scores and not above
  the 1% false accept one, as a share of own-profile scores and of other-profile scores. Above the
  band a host accepts, below it rejects, each at a 1% error; in it, it asks for more speech.

Writes `<out>.md` and `<out>.json`; what a run means goes in `<out>.reading.md`, which the page
appends to itself.
"""

import argparse
import ctypes
import json
import random
import re
import subprocess
import sys
import time
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent))
import speaker_evidence as se  # noqa: E402
import speech_commands as sc  # noqa: E402

Json = dict[str, Any]
type Vec = NDArray[np.float64]
type Samples = NDArray[np.float32]
RATE = sc.RATE
SNRS_DB = (20.0, 10.0, 5.0)
POOLED_MS = (500, 1000, 2000)
WORD = "one word"
CLEAN = "clean"
FALSE_ACCEPT = 0.01
FALSE_REJECT = 0.01


def samples(pcm: bytes) -> Samples:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


@dataclass
class Item:
    """One scored item: a probe word, or a cut of a speaker's joined probe words."""

    key: str
    speaker: str
    x: Samples
    vectors: dict[str, Vec | None] = field(default_factory=dict)


class Utter:
    """utter's C ABI: the model for word spans, TitaNet-small for embeddings."""

    def __init__(self, lib_path: str, model_dir: str, titanet: str) -> None:
        self.u = se.UtterLib(lib_path)
        lib = self.u.lib
        self.u.model = lib.utter_model_new(model_dir.encode())
        self.u.tn = lib.utter_spk_model_new(titanet.encode())
        if not self.u.model or not self.u.tn:
            raise SystemExit("utter could not open the model or TitaNet-small")
        self.dim = int(lib.utter_spk_model_dim(self.u.tn))

    def words(self, grammar: Sequence[str], pcm: bytes, block_ms: int) -> list[Json]:
        """The spoken word entries of every final, the audio fed in blocks as a host feeds it."""
        rec = se.UtterRec(self.u, grammar, se.BARE)
        finals: list[Json] = []
        for b in se.blocks(pcm, block_ms):
            if rec.accept(b):
                finals.append(rec.closed())
        finals.append(rec.final())
        return [e for f in finals for e in f.get("result", []) if se.spoken(e)]

    def embed(self, x: Samples) -> Vec | None:
        out = (ctypes.c_float * self.dim)()
        x = np.ascontiguousarray(x, dtype=np.float32)
        n = self.u.lib.utter_spk_model_embed(
            self.u.tn, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(x), float(RATE), out, self.dim
        )
        return np.asarray(out[:n], dtype=np.float64) if n > 0 else None


def span_of(entries: Sequence[Mapping[str, Any]], n: int) -> tuple[int, int] | None:
    if not entries:
        return None
    a = max(0, round(min(float(e["start"]) for e in entries) * RATE))
    b = min(n, round(max(float(e["end"]) for e in entries) * RATE))
    return (a, b) if b > a else None


# --- scoring --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Gate:
    accept: float
    reject: float


def trials(items: Sequence[Item], cond: str, profiles: Mapping[str, Vec]) -> tuple[list[float], list[float], int]:
    """Own-profile scores, other-profile scores, and the items with no vector."""
    names = sorted(profiles)
    mat = np.stack([se.unit(profiles[n]) for n in names])
    own: list[float] = []
    other: list[float] = []
    missing = 0
    for it in items:
        v = it.vectors.get(cond)
        if v is None or it.speaker not in profiles:
            missing += 1
            continue
        s = mat @ se.unit(v)
        for n, x in zip(names, s, strict=True):
            (own if n == it.speaker else other).append(float(x))
    return own, other, missing


def gate_of(own: Sequence[float], other: Sequence[float]) -> Gate:
    return Gate(
        accept=float(np.quantile(other, 1.0 - FALSE_ACCEPT)) if other else float("inf"),
        reject=float(np.quantile(own, FALSE_REJECT)) if own else float("-inf"),
    )


def row(items: Sequence[Item], cond: str, profiles: Mapping[str, Vec], gate: Gate | None) -> Json:
    """gate, when given, is set elsewhere; the row's own 1% false accept figure is kept beside it."""
    own, other, missing = trials(items, cond, profiles)
    own_gate = gate_of(own, other)
    g = gate or own_gate
    n = len(items)
    o, x = np.asarray(own), np.asarray(other)

    def share(a: NDArray[np.float64], lo: float, hi: float) -> float:
        return float(np.mean((a > lo) & (a <= hi))) if len(a) else float("nan")

    return dict(
        items=n,
        no_evidence=missing / n if n else float("nan"),
        eer=se.eer(own, other),
        accepted_own=float(np.sum(o > own_gate.accept)) / n if n else float("nan"),
        accepted=float(np.sum(o > g.accept)) / n if n else float("nan"),
        false_accept=float(np.mean(x > g.accept)) if len(x) else float("nan"),
        false_reject=float(np.mean(o <= g.reject)) if len(o) else float("nan"),
        band_own=share(o, g.reject, g.accept),
        band_other=share(x, g.reject, g.accept),
        gate=dict(accept=g.accept, reject=g.reject),
    )


def pooled_items(words: Mapping[str, Sequence[Samples]], streams: int, rng: random.Random) -> dict[int, list[Item]]:
    """Each speaker's words joined in shuffled orders and cut at each pooled length."""
    out: dict[int, list[Item]] = {n: [] for n in POOLED_MS}
    for s, pieces in words.items():
        pieces = list(pieces)
        for j in range(streams):
            rng.shuffle(pieces)
            x = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
            for n in POOLED_MS:
                m = n * RATE // 1000
                if len(x) >= m:
                    out[n].append(Item(f"{s}/{j}/{n}", s, x[:m]))
    return out


def by_length(words: Sequence[Item], pooled: Mapping[int, list[Item]]) -> dict[str, list[Item]]:
    return {WORD: list(words)} | {f"{n / 1000:g} s": pooled[n] for n in POOLED_MS}


# --- noise ----------------------------------------------------------------------------------------


def mixed(x: Samples, noise: Samples, offset: int, snr_db: float) -> Samples:
    """x with noise laid under it at snr_db over x's own span, saturating as an ADC would."""
    seg = np.take(noise, np.arange(offset, offset + len(x)), mode="wrap")
    sig = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    nse = float(np.sqrt(np.mean(np.square(seg, dtype=np.float64))))
    if sig == 0 or nse == 0:
        return x
    gain = sig / nse / 10.0 ** (snr_db / 20.0)
    return np.clip(x + gain * seg, -1.0, 1.0).astype(np.float32)


def snr_label(db: float) -> str:
    return f"{db:g} dB"


def run_noise(a: argparse.Namespace) -> Json:
    data = Path(a.data)
    rng = random.Random(a.seed)
    speakers = se.speakers_of(data, a.min_clips)
    names = sorted(speakers)
    if a.limit_speakers:
        names = sorted(rng.sample(names, min(a.limit_speakers, len(names))))
    rooms = {p.stem: samples(sc.read_pcm(p)) for p in sorted((data / "_background_noise_").glob("*.wav"))}
    room_names = sorted(rooms)
    ut = Utter(a.lib, a.model, a.titanet)
    grammar = sc.DATASET_WORDS + sc.LETTERS + sc.NATO + sc.COLOURS

    def span(key: str) -> Samples | None:
        pcm = sc.read_pcm(data / key)
        sp = span_of(ut.words(grammar, pcm, a.block_ms), len(pcm) // 2)
        return samples(pcm)[sp[0] : sp[1]] if sp else None

    profiles: dict[str, Vec] = {}
    probe_words: dict[str, list[Samples]] = {}
    no_span = 0
    for s in names:
        keys = speakers[s][:]
        rng.shuffle(keys)
        enrol = [x for k in keys[: a.enroll] if (x := span(k)) is not None]
        p = se.profile_of([ut.embed(x) for x in enrol])
        if p is None:
            continue
        profiles[s] = p
        probe_words[s] = []
        for k in keys[a.enroll : a.enroll + a.probes]:
            x = span(k)
            if x is None:
                no_span += 1
            else:
                probe_words[s].append(x)
    sc.note(f"{len(profiles)} of {len(names)} speakers enrolled, {sum(map(len, probe_words.values()))} probe words")

    words = [Item(f"{s}/{i}", s, x) for s, xs in probe_words.items() for i, x in enumerate(xs)]
    lengths = by_length(words, pooled_items(probe_words, a.streams, rng))
    conds = [CLEAN] + [snr_label(d) for d in SNRS_DB]
    draws: dict[str, list[str]] = {}
    for label, items in lengths.items():
        for it in items:
            room = rng.choice(room_names)
            off = rng.randrange(len(rooms[room]))
            draws.setdefault(room, []).append(it.key)
            it.vectors[CLEAN] = ut.embed(it.x)
            for d in SNRS_DB:
                it.vectors[snr_label(d)] = ut.embed(mixed(it.x, rooms[room], off, d))
        sc.note(f"{label}: {len(items)} items embedded clean and at {len(SNRS_DB)} SNRs")

    table: dict[str, dict[str, Json]] = {}
    for label, items in lengths.items():
        clean = row(items, CLEAN, profiles, None)
        g = Gate(clean["gate"]["accept"], clean["gate"]["reject"])
        table[label] = {c: clean if c == CLEAN else row(items, c, profiles, g) for c in conds}
    return dict(
        speakers=len(names),
        enrolled=len(profiles),
        probes=len(words),
        probes_without_span=no_span,
        rooms={r: len(v) / RATE for r, v in rooms.items()},
        room_draws={r: len(k) for r, k in sorted(draws.items())},
        conditions=conds,
        table=table,
        utter=dict(version=ut.u.version, revision=ut.u.revision),
    )


# --- VCTK -----------------------------------------------------------------------------------------


class Vctk:
    """The corpus zip, read in place: prompts under txt/, mic1 audio under wav48_silence_trimmed/."""

    def __init__(self, path: Path) -> None:
        self.z = zipfile.ZipFile(path)
        self.prompts: dict[str, dict[str, str]] = {}
        audio = set()
        for n in self.z.namelist():
            m = re.fullmatch(r"txt/(p\d+|s\d+)/(\1_\d+)\.txt", n)
            if m:
                self.prompts.setdefault(m[1], {})[m[2]] = n
            m = re.fullmatch(r"wav48_silence_trimmed/(p\d+|s\d+)/((\1_\d+)_mic1)\.flac", n)
            if m:
                audio.add(m[3])
        self.prompts = {s: {u: n for u, n in us.items() if u in audio} for s, us in sorted(self.prompts.items())}

    def words(self, utt: str) -> list[str]:
        spk = utt.split("_")[0]
        text = self.z.read(self.prompts[spk][utt]).decode("utf-8", "replace").lower()
        return re.findall(r"[a-z]+(?:'[a-z]+)?", text)

    def pcm(self, utt: str) -> bytes:
        spk = utt.split("_")[0]
        flac = self.z.read(f"wav48_silence_trimmed/{spk}/{utt}_mic1.flac")
        done = subprocess.run(
            ["sox", "-q", "-t", "flac", "-", "-t", "raw", "-r", str(RATE), "-b", "16", "-c", "1", "-e", "signed", "-"],
            input=flac,
            capture_output=True,
            check=True,
        )
        return done.stdout


def run_vctk(a: argparse.Namespace) -> Json:
    rng = random.Random(a.seed)
    corpus = Vctk(Path(a.vctk))
    names = [s for s, us in corpus.prompts.items() if len(us) >= a.enroll + a.probes]
    if a.limit_speakers:
        names = sorted(rng.sample(names, min(a.limit_speakers, len(names))))
    ut = Utter(a.lib, a.model, a.titanet)
    min_n = a.min_word_ms * RATE // 1000
    decoded = placed = 0
    word_ms: list[float] = []

    def one_word(utt: str) -> Samples | None:
        """A word the recognizer placed in the utterance, drawn among those long enough."""
        nonlocal decoded, placed
        told = corpus.words(utt)
        if not told:
            return None
        pcm = corpus.pcm(utt)
        decoded += 1
        entries = [e for e in ut.words(sorted(set(told)), pcm, a.block_ms) if str(e["word"]) in told]
        placed += len(entries)
        spans = [sp for e in entries if (sp := span_of([e], len(pcm) // 2)) and sp[1] - sp[0] >= min_n]
        if not spans:
            return None
        b, e = rng.choice(spans)
        word_ms.append((e - b) * 1000.0 / RATE)
        return samples(pcm)[b:e]

    profiles: dict[str, Vec] = {}
    probe_words: dict[str, list[Samples]] = {}
    for s in names:
        utts = sorted(corpus.prompts[s])
        rng.shuffle(utts)
        enrol: list[Samples] = []
        probes: list[Samples] = []
        for u in utts:
            if len(probes) >= a.probes:
                break
            x = one_word(u)
            if x is not None:
                (enrol if len(enrol) < a.enroll else probes).append(x)
        p = se.profile_of([ut.embed(x) for x in enrol]) if len(enrol) == a.enroll else None
        if p is None:
            continue
        profiles[s] = p
        probe_words[s] = probes
        sc.note(f"{s}: {len(enrol)} enrolment and {len(probes)} probe words")

    words = [Item(f"{s}/{i}", s, x) for s, xs in probe_words.items() for i, x in enumerate(xs)]
    lengths = by_length(words, pooled_items(probe_words, a.streams, rng))
    for items in lengths.values():
        for it in items:
            it.vectors[CLEAN] = ut.embed(it.x)
    table = {label: {CLEAN: row(items, CLEAN, profiles, None)} for label, items in lengths.items()}
    return dict(
        speakers=len(names),
        enrolled=len(profiles),
        probes=len(words),
        utterances_decoded=decoded,
        words_placed=placed,
        word_ms=dict(
            median=float(np.median(word_ms)) if word_ms else float("nan"),
            p10=float(np.quantile(word_ms, 0.1)) if word_ms else float("nan"),
            p90=float(np.quantile(word_ms, 0.9)) if word_ms else float("nan"),
        ),
        conditions=[CLEAN],
        table=table,
        utter=dict(version=ut.u.version, revision=ut.u.revision),
    )


# --- the pages ------------------------------------------------------------------------------------


def header(report: Json, a: argparse.Namespace) -> str:
    prov, ut = report["provenance"], report["result"]["utter"]
    return (
        f"Run {prov['date']}, utter {ut['version']} from {ut['revision']} through its C ABI "
        f"(checkout {prov['utter_revision']}), model {prov['model']}, speaker model "
        f"{Path(a.titanet).name} as scripts/titanet_convert.py writes it, {prov['cpu']}, "
        f"Python {prov['python']}, blocks of {a.block_ms} ms."
    )


def figures(r: Json, lines: list[str], threshold_note: str) -> None:
    lines += [
        "*Can't tell* is a score above the threshold that turns away 1% of own-profile scores "
        "and not above the one that lets 1% of other-profile scores through: a host accepts "
        "above the band and rejects below it, each at a 1% error, and asks for more speech "
        "inside it. When the two thresholds cross the band is empty. The cell gives the share of "
        f"own-profile scores in the band, then of other-profile scores. {threshold_note}",
        "",
        "| speech | condition | items | no evidence | equal error rate | accepted at 1% false accept "
        "| accepted at the gate | false accept at the gate | can't tell |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for label, conds in r["table"].items():
        for c, x in conds.items():
            lines.append(
                f"| {label} | {c} | {x['items']} | {se.pct(x['no_evidence'])} | {se.pct(x['eer'])} "
                f"| {se.pct(x['accepted_own'])} | {se.pct(x['accepted'])} | {se.pct(x['false_accept'])} "
                f"| {se.pct(x['band_own'])}, {se.pct(x['band_other'])} |"
            )


def noise_page(report: Json, a: argparse.Namespace) -> list[str]:
    r = report["result"]
    rooms = ", ".join(f"{k} ({v:.0f} s)" for k, v in r["rooms"].items())
    L = [
        "# Speaker evidence in a noisy room",
        "",
        header(report, a),
        "",
        "Speech Commands speakers each recorded on their own device, so the room a word was "
        "spoken in is part of what a profile learns. This page lays a room under the probe "
        "words and asks how far TitaNet-small's evidence through utter_spk_model_embed moves "
        f"when the profile was enrolled in the quiet. {r['enrolled']} speakers of the test split "
        f"with at least {a.min_clips} clips enrol from {a.enroll} clean words each; "
        f"{r['probes']} probe words ({r['probes_without_span']} probe clips gave utter's "
        "recognizer no word and are left out) are scored clean and with a background recording "
        f"laid under them at {', '.join(snr_label(d) for d in SNRS_DB)} SNR, the SNR taken over "
        "the span embedded. Each item draws one recording and offset, the same at every SNR. "
        f"The recordings: {rooms}; pink and white noise are generated.",
        "",
        "The word span is where utter's recognizer, under the Speech Commands page's grammar, "
        "puts the clip's spoken words, found on the clean clip. Beside single words, each "
        f"speaker's probe words are joined {a.streams} times in shuffled orders and cut at "
        f"{', '.join(f'{n / 1000:g} s' for n in POOLED_MS)} of speech, and the room laid under "
        "the cut.",
        "",
    ]
    figures(
        r,
        L,
        "*Accepted at 1% false accept* is at the row's own threshold; the gate and the band "
        "are the clean row's at the same length, as a host sets them in the quiet.",
    )
    return L


def vctk_page(report: Json, a: argparse.Namespace) -> list[str]:
    r = report["result"]
    wm = r["word_ms"]
    L = [
        "# Speaker evidence on a shared microphone",
        "",
        header(report, a),
        "",
        "Speech Commands speakers each recorded on their own device, so a profile carries the "
        "microphone as well as the voice. In the CSTR VCTK Corpus 0.92 (CC BY 4.0) every "
        "speaker read into the same microphones in the same room; this page scores "
        "TitaNet-small through utter_spk_model_embed on the DPA 4035 channel (mic1), resampled "
        "from 48 kHz to 16 kHz.",
        "",
        "VCTK has no word times. Each utterance is decoded by utter's recognizer under a grammar "
        "of its own prompt's words, which places the words it was told were said; one word of "
        f"at least {a.min_word_ms} ms is drawn per utterance ({r['words_placed']} words placed in "
        f"{r['utterances_decoded']} utterances decoded; the words drawn have a median of "
        f"{wm['median']:.0f} ms, 10th and 90th percentiles {wm['p10']:.0f} and {wm['p90']:.0f} ms). "
        f"{r['enrolled']} of {r['speakers']} speakers enrol from {a.enroll} words from as many "
        f"utterances; {r['probes']} probe words come from {a.probes} other utterances each. "
        f"Beside single words, each speaker's probe words are joined {a.streams} times in "
        f"shuffled orders and cut at {', '.join(f'{n / 1000:g} s' for n in POOLED_MS)} of speech.",
        "",
    ]
    figures(r, L, "The gate and the band are each row's own.")
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="set", required=True)
    for name, stem in (("noise", "speaker-noise"), ("vctk", "speaker-vctk")):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument(
            "--titanet",
            default=str(Path.home() / "Repos/utter-bench-data/release-0.0.6" / se.TITANET_DIR),
            help="TitaNet-small as scripts/titanet_convert.py writes it",
        )
        p.add_argument("--lib", default="target/release/libutter.so")
        p.add_argument("--out", default=f"docs/benchmarks/{stem}")
        p.add_argument("--enroll", type=int, default=10)
        p.add_argument("--probes", type=int, default=10)
        p.add_argument("--streams", type=int, default=se.STREAMS_PER_SPEAKER)
        p.add_argument("--limit-speakers", type=int, default=0)
        p.add_argument("--block-ms", type=int, default=40)
        p.add_argument("--seed", type=int, default=14)
        if name == "noise":
            p.add_argument("--data", default=str(Path.home() / "Repos/utter-bench-data/speech_commands_v0.02"))
            p.add_argument("--min-clips", type=int, default=20)
        else:
            p.add_argument("--vctk", default=str(Path.home() / "Repos/utter-bench-data/vctk/VCTK-Corpus-0.92.zip"))
            p.add_argument("--min-word-ms", type=int, default=250)
    a = ap.parse_args()
    t = time.monotonic()
    result = run_noise(a) if a.set == "noise" else run_vctk(a)
    report: Json = dict(
        provenance=sc.provenance(a, ["numpy"]),
        args=sc.recorded_args(vars(a)),
        seconds=time.monotonic() - t,
        result=result,
    )
    out = Path(a.out)
    out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    lines = noise_page(report, a) if a.set == "noise" else vctk_page(report, a)
    reading = out.with_suffix(".reading.md")
    if reading.exists():
        lines += ["", reading.read_text()]
    text = sc.write_page(out.with_suffix(".md"), lines)
    print(text)
    sc.note(f"wrote {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
