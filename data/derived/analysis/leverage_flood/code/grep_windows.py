"""Print keyword windows from text dumps in the scratchpad.
Usage: py grep_windows.py <dump name> <window chars> <keyword1> [keyword2 ...]
"""
import os
import re
import sys

import pathlib as _pathlib
_REPO = _pathlib.Path(__file__).resolve().parents[5]  # 저장소 루트(파일 위치 기준)
SCRATCH = str(_REPO / "data/derived/analysis/cache/scratch/f45f5967/scratchpad")


def main():
    name, win = sys.argv[1], int(sys.argv[2])
    kws = sys.argv[3:]
    path = os.path.join(SCRATCH, f"{name}.txt")
    text = open(path, encoding="utf-8").read()
    for kw in kws:
        hits = [m.start() for m in re.finditer(re.escape(kw), text)]
        print(f"\n######## [{name}] '{kw}' hits={len(hits)}")
        last_end = -1
        for pos in hits[:6]:
            s, e = max(0, pos - win), min(len(text), pos + win)
            if s < last_end:
                continue
            last_end = e
            print(f"--- @{pos}")
            print(text[s:e].replace("\n", " "))


if __name__ == "__main__":
    main()
