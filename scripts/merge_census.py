"""The merge and reading-loss census over a Speech Commands split, as JSON keyed by clip.

`stream --census` counts three things per take and `[[rr:TD-9#No lattice is added for this
feature]]` is why they are kept apart. The binary takes its WAVs as arguments, so the split is
fed to it in batches; `scripts/partial_trust.py --census` joins the result to its first sightings.
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import speech_commands as sc  # noqa: E402

BATCH = 500


def clips_of(data, split):
    listed = [x.strip() for x in (data / f"{split}_list.txt").read_text().splitlines() if x.strip()]
    by = defaultdict(list)
    for rel in listed:
        by[rel.split("/")[0]].append(rel)
    return [data / rel for w in sc.DATASET_WORDS for rel in by.get(w, [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="testing", choices=("testing", "validation"))
    ap.add_argument("--block-ms", type=int, default=40)
    ap.add_argument("--alternatives", type=int, default=5)
    ap.add_argument("--stream", default="target/release/stream")
    a = ap.parse_args()

    data = Path(a.data)
    clips = clips_of(data, a.split)
    grammar = sc.DATASET_WORDS + sc.LETTERS + sc.NATO + sc.COLOURS
    gpath = Path(a.out + ".grammar.json")
    gpath.write_text(json.dumps(grammar))
    rows = {}
    try:
        for i in range(0, len(clips), BATCH):
            batch = clips[i : i + BATCH]
            cmd = [
                a.stream,
                "--model",
                str(a.model),
                "--grammar",
                str(gpath),
                "--block-ms",
                str(a.block_ms),
                "--alternatives",
                str(a.alternatives),
                "--partial-words",
                "--census",
            ] + [str(p) for p in batch]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            for path, line in zip(batch, proc.stdout.splitlines(), strict=True):
                take = json.loads(line)
                rows[f"{path.parent.name}/{path.name}"] = take["census"]
            sc.note(f"census: {i + len(batch)}/{len(clips)} clips")
    finally:
        gpath.unlink(missing_ok=True)
    Path(a.out).write_text(json.dumps(dict(split=a.split, block_ms=a.block_ms, clips=rows)))
    sc.note(f"wrote {a.out} over {len(rows)} clips")


if __name__ == "__main__":
    main()
