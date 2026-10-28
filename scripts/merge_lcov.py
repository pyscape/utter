"""Merge lcov files from runs over the same build, and hold the merged line coverage to a floor.

Usage: merge_lcov.py OUT FLOOR IN...

Hits on a line or a function add up across the inputs. Prints the merged line coverage and
exits non-zero when it is under FLOOR percent.
"""

import sys
from collections import defaultdict


def merge(paths: list[str]) -> tuple[dict[str, dict[int, int]], dict[str, dict[str, tuple[int, int]]]]:
    lines: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    functions: dict[str, dict[str, tuple[int, int]]] = defaultdict(dict)
    for path in paths:
        source = ""
        with open(path, encoding="utf-8") as f:
            for raw in f:
                record = raw.rstrip("\n")
                tag, _, value = record.partition(":")
                if tag == "SF":
                    source = value
                elif tag == "DA":
                    line, hits = value.split(",")[:2]
                    lines[source][int(line)] += int(hits)
                elif tag == "FN":
                    start_text, name = value.split(",", 1)
                    _, calls = functions[source].get(name, (0, 0))
                    functions[source][name] = (int(start_text), calls)
                elif tag == "FNDA":
                    calls_text, name = value.split(",", 1)
                    start, calls = functions[source].get(name, (0, 0))
                    functions[source][name] = (start, calls + int(calls_text))
    return lines, functions


def main() -> int:
    out, floor, inputs = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
    lines, functions = merge(inputs)
    found = hit = 0
    with open(out, "w", encoding="utf-8") as f:
        for source in sorted(lines.keys() | functions.keys()):
            f.write(f"SF:{source}\n")
            fns = functions.get(source, {})
            for name, (start, _) in sorted(fns.items(), key=lambda kv: kv[1][0]):
                f.write(f"FN:{start},{name}\n")
            for name, (_, calls) in fns.items():
                f.write(f"FNDA:{calls},{name}\n")
            f.write(f"FNF:{len(fns)}\nFNH:{sum(1 for _, c in fns.values() if c)}\n")
            das = lines.get(source, {})
            for line in sorted(das):
                f.write(f"DA:{line},{das[line]}\n")
            covered = sum(1 for h in das.values() if h)
            f.write(f"LF:{len(das)}\nLH:{covered}\nend_of_record\n")
            found += len(das)
            hit += covered
    percent = 100.0 * hit / found if found else 0.0
    print(f"lines: {hit} of {found}, {percent:.2f}%")
    if percent < floor:
        print(f"under the floor of {floor}%", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
