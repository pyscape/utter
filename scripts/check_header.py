"""The hand-written C header declares exactly the exported functions, with cbindgen's
signatures: names, parameter types and return types, so a change on either side that the
symbol-name diff would miss fails here.

Usage: scripts/check_header.py [repo root]
"""

import re
import subprocess
import sys
from pathlib import Path


def prototypes(text: str) -> dict[str, str]:
    # A declaration runs from the start of a line to its semicolon, possibly over several lines.
    out = {}
    for decl in re.findall(r"^[A-Za-z_][^;#{}]*?\butter_[a-z_]+\s*\([^;]*?\)\s*;", text, re.M | re.S):
        flat = re.sub(r"\s+", " ", decl).strip()
        flat = flat.replace("struct ", "")
        flat = re.sub(r"\s*\*\s*", " *", flat)
        flat = re.sub(r"\(\s*", "(", flat)
        flat = re.sub(r"\s*\)", ")", flat)
        flat = re.sub(r"\s*,\s*", ", ", flat)
        out[re.findall(r"\butter_[a-z_]+", flat)[0]] = flat
    return out


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    generated = subprocess.run(
        ["cbindgen", "--lang", "c", "--crate", "utter", str(root)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    want = prototypes(generated)
    have = prototypes((root / "include" / "utter.h").read_text())
    bad = 0
    for name in sorted(set(want) | set(have)):
        if want.get(name) != have.get(name):
            bad += 1
            print(f"{name}:\n  cbindgen: {want.get(name)}\n  header:   {have.get(name)}")
    print(f"{len(have)} declarations, {bad} differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
