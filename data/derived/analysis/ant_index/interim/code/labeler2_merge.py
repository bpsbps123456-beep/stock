"""라벨러 2: 손으로 적은 labels_2_raw.txt를 labels_2.json으로 옮긴다(판정 없음, 형식 변환·점검만).

raw 형식: '@묶음' 머리줄 뒤에 '<묶음 안 순번> <in_sense Y/N/-><irony Y/N> [B/R/Z] [kY kN ...]'.
순번은 labeler2_view.py가 보여 준 순서(blind_samples.json 안 순서)다.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
YN = {"Y": "예", "N": "아니오"}
POL = {"B": "강세", "R": "약세", "Z": "중립"}


def main():
    d = json.loads((RES / "blind_samples.json").read_text(encoding="utf-8"))
    groups = {}
    for s in d["samples"]:
        groups.setdefault(s["bundle"], []).append(s)
    groups["score200"] = d["score_samples"]

    raw = (RES / "labels_2_raw.txt").read_text(encoding="utf-8").splitlines()
    cur, seen, out, problems = None, set(), [], []
    for line in raw:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("@"):
            cur = line[1:]
            if cur in seen:
                problems.append(f"묶음 중복 {cur}")
            seen.add(cur)
            continue
        parts = line.split()
        i, flags = int(parts[0]), parts[1]
        rest = parts[2:]
        s = groups[cur][i]
        pol = None
        if rest and rest[0] in POL:
            pol = POL[rest[0]]
            rest = rest[1:]
        hits = {}
        if rest:
            for k, v in re.findall(r"(\d+)([YN])", rest[0]):
                hits[int(k)] = YN[v]
        rec = {
            "id": s["id"],
            "in_sense": None if flags[0] == "-" else YN[flags[0]],
            "irony": YN[flags[1]],
            "polarity": pol,
            "bundle": s.get("bundle", "score200"),
            "set": s["set"],
        }
        ks = [h["k"] for h in s.get("hits", [])]
        if len(ks) > 1:
            if sorted(hits) != sorted(ks):
                problems.append(f"{cur}#{i} 적중 번호 불일치 {sorted(hits)} vs {ks}")
            rec["hits_in_sense"] = {str(k): hits.get(k) for k in ks}
            anyy = any(v == "예" for v in hits.values())
            if (rec["in_sense"] == "예") != anyy:
                problems.append(f"{cur}#{i} 단위·적중 판정 불일치")
        elif hits:
            problems.append(f"{cur}#{i} 적중 1개인데 적중별 판정 있음")
        need_pol = cur in ("BULL", "BEAR", "score200")
        if need_pol and pol is None:
            problems.append(f"{cur}#{i} polarity 없음")
        if not need_pol and pol is not None:
            problems.append(f"{cur}#{i} polarity 불필요")
        out.append(rec)

    ids = [r["id"] for r in out]
    allids = [s["id"] for s in d["samples"]] + [s["id"] for s in d["score_samples"]]
    dup = [k for k, v in Counter(ids).items() if v > 1]
    missing = sorted(set(allids) - set(ids))
    if dup:
        problems.append(f"id 중복 {len(dup)}")
    if missing:
        problems.append(f"라벨 없는 id {len(missing)}")
    for b, rows in groups.items():
        n = sum(1 for r in out if (r["bundle"] == b))
        if n != len(rows):
            problems.append(f"{b} 개수 {n} != {len(rows)}")
    if problems:
        print("문제:", *problems, sep="\n  ")
        sys.exit(1)
    (RES / "labels_2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("저장", len(out))
    by = Counter((r["bundle"], r["in_sense"]) for r in out)
    for b in groups:
        y, n = by[(b, "예")], by[(b, "아니오")]
        irn = sum(1 for r in out if r["bundle"] == b and r["irony"] == "예")
        print(f"{b}\t예 {y}\t아니오 {n}\t반어·농담 {irn}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
