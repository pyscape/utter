#!/usr/bin/env python3
"""Speaker evidence away from the speaker's own device: the evidence utter's recognizer puts on
each word with TitaNet-small set, in a noisy room and from speakers who all share one microphone.

    python scripts/speaker_conditions.py noise --model MODEL_DIR [--data DIR] \\
        [--lib target/release/libutter.so] [--out docs/benchmarks/speaker-noise]
    python scripts/speaker_conditions.py vctk --model MODEL_DIR [--vctk ZIP] \\
        [--lib target/release/libutter.so] [--out docs/benchmarks/speaker-vctk]

MODEL_DIR is the stock small English model. Each speaker's audio runs through utter's C ABI as
continuous streams, as a host feeds it, to a recognizer with TitaNet-small set, as
scripts/titanet_convert.py writes it. Every spoken word entry of the finals is a word; one with
evidence is a probe whatever its label, and the words without evidence are counted. Two rows:

- recognizer: the entry's own spk vector;
- embed(): utter_spk_model_embed over the entry's spk_start to spk_end, as a host that embeds the
  spans itself would.

noise: the Speech Commands v2 (CC BY 4.0) test split as scripts/speaker_evidence.py draws it,
under that page's grammar. Each speaker's --enroll clips are one clean stream that sets the
profile; the --probes other clips are a second stream, decoded clean and again with one of the
dataset's _background_noise_ recordings laid under each clip at each SNR in SNRS_DB. The SNR is
over the clip's active speech, its 10 ms frames within ACTIVE_DB of its loudest. Thresholds are
the clean row's at the same length: a host sets its gate in the quiet and meets the room
afterwards.

vctk: the CSTR VCTK Corpus 0.92 (CC BY 4.0), read from its DataShare zip, the DPA 4035 channel
(mic1) at 48 kHz resampled to 16 kHz by sox. utter decodes under a grammar only, so each stream's
grammar is the words of its own utterances' prompts. Each speaker enrols from one stream of
--enroll utterances, and --probes other utterances are the probe stream. Thresholds are the set's
own.

Probes are scored by cosine against every enrolled profile, one per row. Beside single words, a
speaker's probe words are taken, --streams times in shuffled orders, until their evidence has
pooled each length in POOLED_MS: the recognizer row averages their vectors weighted by frames,
the embed() row embeds their spans joined. Per row:

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
ACTIVE_DB = 20.0
POOLED_MS = (500, 1000, 2000)
# TitaNet's frames are 10 ms.
FRAME_MS = 10
WORD = "one word"
CLEAN = "clean"
REC, EMB = "recognizer", "embed()"
ROWS = (REC, EMB)
FALSE_ACCEPT = 0.01
FALSE_REJECT = 0.01


def samples(pcm: bytes) -> Samples:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def pcm_of(x: Samples) -> bytes:
    return bytes((np.clip(x, -1.0, 32767 / 32768) * 32768.0).astype(np.int16).tobytes())


@dataclass
class Word:
    """A spoken word entry of a final, with its evidence and the span it pooled."""

    speaker: str
    rec: Vec | None
    frames: int
    span: Samples | None
    emb: Vec | None = None


@dataclass
class Item:
    """One scored item: a probe word, or the words pooled to a length."""

    key: str
    speaker: str
    vectors: dict[str, Vec | None] = field(default_factory=dict)


class Utter:
    """utter's C ABI: a recognizer with TitaNet-small set, and its stateless embed."""

    def __init__(self, lib_path: str, model_dir: str, titanet: str) -> None:
        self.u = se.UtterLib(lib_path)
        lib = self.u.lib
        self.u.model = lib.utter_model_new(model_dir.encode())
        self.u.tn = lib.utter_spk_model_new(titanet.encode())
        if not self.u.model or not self.u.tn:
            raise SystemExit("utter could not open the model or TitaNet-small")
        self.dim = int(lib.utter_spk_model_dim(self.u.tn))

    def stream(self, speaker: str, grammar: Sequence[str], x: Samples, block_ms: int) -> list[Word]:
        """The stream through one recognizer: every spoken word entry of its finals."""
        pcm = pcm_of(x)
        rec = se.UtterRec(self.u, grammar, se.SETUPS[se.UTTER_TN])
        finals: list[Json] = []
        for b in se.blocks(pcm, block_ms):
            if rec.accept(b):
                finals.append(rec.closed())
        finals.append(rec.final())
        out = []
        for e in (e for f in finals for e in f.get("result", []) if se.spoken(e)):
            v = se.evidence(e)
            w = Word(speaker, v, int(e.get("spk_frames", 0)), None)
            if v is not None:
                w.span = x[int(e["spk_start"]) : int(e["spk_end"])]
                w.emb = self.embed(w.span)
            out.append(w)
        return out

    def embed(self, x: Samples) -> Vec | None:
        out = (ctypes.c_float * self.dim)()
        x = np.ascontiguousarray(x, dtype=np.float32)
        n = self.u.lib.utter_spk_model_embed(
            self.u.tn, x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(x), float(RATE), out, self.dim
        )
        return np.asarray(out[:n], dtype=np.float64) if n > 0 else None


def profiles_of(words: Sequence[Word]) -> dict[str, Vec | None]:
    return {REC: se.profile_of([w.rec for w in words]), EMB: se.profile_of([w.emb for w in words])}


# --- scoring --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Gate:
    accept: float
    reject: float


def trials(items: Sequence[Item], row_: str, profiles: Mapping[str, Vec]) -> tuple[list[float], list[float], int]:
    """Own-profile scores, other-profile scores, and the items with no vector."""
    names = sorted(profiles)
    mat = np.stack([se.unit(profiles[n]) for n in names])
    own: list[float] = []
    other: list[float] = []
    missing = 0
    for it in items:
        v = it.vectors.get(row_)
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


def score(items: Sequence[Item], row_: str, profiles: Mapping[str, Vec], gate: Gate | None) -> Json:
    """gate, when given, is set elsewhere; the row's own 1% false accept figure is kept beside it."""
    own, other, missing = trials(items, row_, profiles)
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


def word_items(words: Mapping[str, Sequence[Word]]) -> list[Item]:
    """Every spoken word is an item; one without evidence counts as none for both rows."""
    return [Item(f"{s}/{i}", s, {REC: w.rec, EMB: w.emb}) for s, ws in words.items() for i, w in enumerate(ws)]


def pooled_items(
    ut: Utter, words: Mapping[str, Sequence[Word]], streams: int, rng: random.Random
) -> dict[int, list[Item]]:
    """A speaker's words with evidence, in shuffled orders, taken until they have pooled each length."""
    out: dict[int, list[Item]] = {n: [] for n in POOLED_MS}
    for s, ws in words.items():
        have = [w for w in ws if w.rec is not None and w.span is not None]
        for j in range(streams):
            rng.shuffle(have)
            for n in POOLED_MS:
                taken, ms = [], 0
                for w in have:
                    if ms >= n:
                        break
                    taken.append(w)
                    ms += w.frames * FRAME_MS
                if ms < n:
                    continue
                weights = np.array([w.frames for w in taken], dtype=np.float64)
                rec = np.average(
                    np.stack([se.unit(w.rec) for w in taken if w.rec is not None]), axis=0, weights=weights
                )
                spans = [w.span for w in taken if w.span is not None]
                out[n].append(Item(f"{s}/{j}/{n}", s, {REC: rec, EMB: ut.embed(np.concatenate(spans))}))
    return out


def table_of(
    ut: Utter,
    words: Mapping[str, Sequence[Word]],
    profiles: Mapping[str, Mapping[str, Vec]],
    streams: int,
    rng: random.Random,
    gates: Mapping[str, Mapping[str, Gate]] | None,
) -> dict[str, dict[str, Json]]:
    """Per length, per row. gates, when given, are per length and row."""
    pooled = pooled_items(ut, words, streams, rng)
    lengths = {WORD: word_items(words)} | {f"{n / 1000:g} s": pooled[n] for n in POOLED_MS}
    return {
        label: {r: score(items, r, profiles[r], gates[label][r] if gates else None) for r in ROWS}
        for label, items in lengths.items()
    }


def gates_of(table: Mapping[str, Mapping[str, Json]]) -> dict[str, dict[str, Gate]]:
    return {
        label: {r: Gate(x["gate"]["accept"], x["gate"]["reject"]) for r, x in rows.items()}
        for label, rows in table.items()
    }


def split_profiles(by_speaker: Mapping[str, Mapping[str, Vec | None]]) -> dict[str, dict[str, Vec]]:
    """Per row, the speakers it enrolled; a speaker a row could not enrol is no rival under it."""
    return {r: {s: v for s, p in by_speaker.items() if (v := p[r]) is not None} for r in ROWS}


def evidence_counts(words: Mapping[str, Sequence[Word]]) -> Json:
    ws = [w for x in words.values() for w in x]
    return dict(words=len(ws), with_evidence=sum(w.rec is not None for w in ws))


# --- noise ----------------------------------------------------------------------------------------


def active_rms(x: Samples) -> float:
    n = RATE * FRAME_MS // 1000
    frames = x[: len(x) // n * n].reshape(-1, n).astype(np.float64)
    if not len(frames):
        return 0.0
    power = np.mean(np.square(frames), axis=1)
    loud = power >= power.max() * 10.0 ** (-ACTIVE_DB / 10.0)
    return float(np.sqrt(np.mean(power[loud])))


def mixed(x: Samples, noise: Samples, offset: int, snr_db: float) -> Samples:
    """x with noise laid under it at snr_db over x's active speech, saturating as an ADC would."""
    seg = np.take(noise, np.arange(offset, offset + len(x)), mode="wrap")
    sig = active_rms(x)
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
    conds = [CLEAN] + [snr_label(d) for d in SNRS_DB]

    by_speaker: dict[str, dict[str, Vec | None]] = {}
    enrol_words: dict[str, list[Word]] = {}
    probe_words: dict[str, dict[str, list[Word]]] = {c: {} for c in conds}
    draws: dict[str, int] = {}
    for s in names:
        keys = speakers[s][:]
        rng.shuffle(keys)
        enrol = np.concatenate([samples(sc.read_pcm(data / k)) for k in keys[: a.enroll]])
        enrol_words[s] = ut.stream(s, grammar, enrol, a.block_ms)
        by_speaker[s] = profiles_of(enrol_words[s])
        clips = [samples(sc.read_pcm(data / k)) for k in keys[a.enroll : a.enroll + a.probes]]
        laid = []
        for x in clips:
            room = rng.choice(room_names)
            draws[room] = draws.get(room, 0) + 1
            laid.append((x, rooms[room], rng.randrange(len(rooms[room]))))
        probe_words[CLEAN][s] = ut.stream(s, grammar, np.concatenate(clips), a.block_ms)
        for d in SNRS_DB:
            noisy = np.concatenate([mixed(x, room, off, d) for x, room, off in laid])
            probe_words[snr_label(d)][s] = ut.stream(s, grammar, noisy, a.block_ms)
        sc.note(f"{s}: {len(enrol_words[s])} enrolment words, {len(probe_words[CLEAN][s])} clean probe words")
    profiles = split_profiles(by_speaker)

    clean = table_of(ut, probe_words[CLEAN], profiles, a.streams, rng, None)
    gates = gates_of(clean)
    tables = {CLEAN: clean} | {
        c: table_of(ut, probe_words[c], profiles, a.streams, rng, gates) for c in conds if c != CLEAN
    }
    table = {label: {c: tables[c][label] for c in conds} for label in clean}
    return dict(
        speakers=len(names),
        enrolled={r: len(p) for r, p in profiles.items()},
        enrolment=evidence_counts(enrol_words),
        probes={c: evidence_counts(probe_words[c]) for c in conds},
        rooms={r: len(v) / RATE for r, v in rooms.items()},
        room_draws=dict(sorted(draws.items())),
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

    def audio(self, utt: str) -> Samples:
        spk = utt.split("_")[0]
        flac = self.z.read(f"wav48_silence_trimmed/{spk}/{utt}_mic1.flac")
        done = subprocess.run(
            ["sox", "-q", "-t", "flac", "-", "-t", "raw", "-r", str(RATE), "-b", "16", "-c", "1", "-e", "signed", "-"],
            input=flac,
            capture_output=True,
            check=True,
        )
        return samples(done.stdout)


def run_vctk(a: argparse.Namespace) -> Json:
    rng = random.Random(a.seed)
    corpus = Vctk(Path(a.vctk))
    names = [s for s, us in corpus.prompts.items() if len(us) >= a.enroll + a.probes]
    if a.limit_speakers:
        names = sorted(rng.sample(names, min(a.limit_speakers, len(names))))
    ut = Utter(a.lib, a.model, a.titanet)

    def stream(s: str, utts: Sequence[str]) -> list[Word]:
        grammar = sorted({w for u in utts for w in corpus.words(u)})
        return ut.stream(s, grammar, np.concatenate([corpus.audio(u) for u in utts]), a.block_ms)

    by_speaker: dict[str, dict[str, Vec | None]] = {}
    enrol_words: dict[str, list[Word]] = {}
    probe_words: dict[str, list[Word]] = {}
    seconds = 0.0
    for s in names:
        utts = sorted(corpus.prompts[s])
        rng.shuffle(utts)
        enrol_words[s] = stream(s, utts[: a.enroll])
        by_speaker[s] = profiles_of(enrol_words[s])
        probe_words[s] = stream(s, utts[a.enroll : a.enroll + a.probes])
        seconds += sum(len(w.span) for w in enrol_words[s] + probe_words[s] if w.span is not None) / RATE
        sc.note(f"{s}: {len(enrol_words[s])} enrolment and {len(probe_words[s])} probe words")
    profiles = split_profiles(by_speaker)
    table = {label: {CLEAN: rows} for label, rows in table_of(ut, probe_words, profiles, a.streams, rng, None).items()}
    return dict(
        speakers=len(names),
        enrolled={r: len(p) for r, p in profiles.items()},
        enrolment=evidence_counts(enrol_words),
        probes={CLEAN: evidence_counts(probe_words)},
        evidence_seconds=seconds,
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


def rows_text(a: argparse.Namespace) -> list[str]:
    return [
        f"Each stream runs through a recognizer with TitaNet-small set. Every spoken word entry of "
        "its finals is a word, and one with evidence is a probe whatever its label; *no evidence* "
        f"is the share of words without. The *{REC}* row scores the entry's own vector; the "
        f"*{EMB}* row scores utter_spk_model_embed over the entry's spk_start to spk_end, as a "
        "host that embeds the spans itself would. Each row enrols from its own vectors. Beside "
        f"single words, a speaker's probe words are taken, {a.streams} times in shuffled orders, "
        f"until their evidence has pooled {', '.join(f'{n / 1000:g} s' for n in POOLED_MS)}: the "
        f"{REC} row averages their vectors weighted by frames, the {EMB} row embeds their spans "
        "joined. The last word taken may carry the pool past the length.",
        "",
    ]


def figures(r: Json, lines: list[str], threshold_note: str) -> None:
    lines += [
        "*Can't tell* is a score above the threshold that turns away 1% of own-profile scores "
        "and not above the one that lets 1% of other-profile scores through: a host accepts "
        "above the band and rejects below it, each at a 1% error, and asks for more speech "
        "inside it. When the two thresholds cross the band is empty. The cell gives the share of "
        f"own-profile scores in the band, then of other-profile scores. {threshold_note}",
        "",
        "| speech | condition | row | items | no evidence | equal error rate | accepted at 1% false accept "
        "| accepted at the gate | false accept at the gate | can't tell |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, conds in r["table"].items():
        for c, rows in conds.items():
            for rw, x in rows.items():
                lines.append(
                    f"| {label} | {c} | {rw} | {x['items']} | {se.pct(x['no_evidence'])} | {se.pct(x['eer'])} "
                    f"| {se.pct(x['accepted_own'])} | {se.pct(x['accepted'])} | {se.pct(x['false_accept'])} "
                    f"| {se.pct(x['band_own'])}, {se.pct(x['band_other'])} |"
                )


def counts_text(r: Json) -> str:
    e = r["enrolment"]
    return (
        f"{r['enrolled'][REC]} of {r['speakers']} speakers enrolled under the {REC} row and "
        f"{r['enrolled'][EMB]} under {EMB}, from {e['with_evidence']} of {e['words']} enrolment "
        "words with evidence. Probe words with evidence: "
        + ", ".join(f"{c} {x['with_evidence']} of {x['words']}" for c, x in r["probes"].items())
        + "."
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
        "words and asks how far the speaker evidence moves when the profile was enrolled in the "
        f"quiet. Speakers of the test split with at least {a.min_clips} clips enrol from one "
        f"clean stream of {a.enroll} clips; {a.probes} other clips each are a probe stream, "
        "decoded clean and again with a background recording laid under each clip at "
        f"{', '.join(snr_label(d) for d in SNRS_DB)} SNR, the SNR over the clip's 10 ms frames "
        f"within {ACTIVE_DB:g} dB of its loudest. Each clip draws one recording and offset, the "
        f"same at every SNR. The recordings: {rooms}; pink and white noise are generated. "
        "Streams decode under the Speech Commands page's grammar.",
        "",
        counts_text(r),
        "",
        *rows_text(a),
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
    L = [
        "# Speaker evidence on a shared microphone",
        "",
        header(report, a),
        "",
        "Speech Commands speakers each recorded on their own device, so a profile carries the "
        "microphone as well as the voice. In the CSTR VCTK Corpus 0.92 (CC BY 4.0) every "
        "speaker read into the same microphones in the same room; this page uses the DPA 4035 "
        "channel (mic1), resampled from 48 kHz to 16 kHz. Each speaker enrols from one stream "
        f"of {a.enroll} utterances, and {a.probes} other utterances are the probe stream. utter "
        "decodes under a grammar only, so a stream's grammar is the words of its own "
        "utterances' prompts; the model's full vocabulary is not open to it.",
        "",
        counts_text(r),
        "",
        *rows_text(a),
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
