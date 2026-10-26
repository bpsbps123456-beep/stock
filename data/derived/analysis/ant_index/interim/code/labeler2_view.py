"""라벨러 2 보기 도구: blind_samples.json을 묶음별로 사람이 읽기 좋게 출력만 한다(판정 없음).

쓰기: py labeler2_view.py <묶음|score200> [시작] [끝]
"""
import json
import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "results" / "blind_samples.json"


def flat(s):
    return (s or "").replace("\r", "").replace("\n", " ⏎ ")


def main():
    d = json.loads(SRC.read_text(encoding="utf-8"))
    b = sys.argv[1]
    lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    hi = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
    rows = d["score_samples"] if b == "score200" else [s for s in d["samples"] if s["bundle"] == b]
    if b != "score200":
        print("묶음", b, d["bundles"][b]["meaning"], "| 범위", d["bundles"][b]["unit_scope"])
    for i, s in enumerate(rows):
        if i < lo or i >= hi:
            continue
        hits = " ".join(
            f"{h['k']}:{h['word']}" + (f"[{h['dict_reading']}]" if h.get("dict_reading") else "")
            + (f"<{h['note']}>" if h.get("note") else "") + f"@{h['where']}"
            for h in s.get("hits", [])
        )
        head = f"#{i} {s['id']} {s['unit']}" + (" (발췌)" if s.get("excerpted") else "")
        print(head, "|", hits)
        if s["unit"] == "댓글":
            print("  부모:", flat(s.get("parent_title")))
            print("  댓글:", flat(s.get("text")))
        else:
            print("  제목:", flat(s.get("title")))
            print("  본문:", flat(s.get("body")))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
