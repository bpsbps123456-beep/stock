"""독립 라벨러 1: 사람이 직접 읽고 적은 판정(results/labels_1_raw/*.txt)을 표본 id에 붙여 labels_1.json으로 묶는다.

판정 자체는 이 스크립트가 하지 않는다. raw 파일의 형식:
  no: 3 7 12        -> 그 묶음에서 '묶음 뜻 아님'인 표본 순번(묶음 안 1부터). 나머지는 '예'
  yes: 4 5          -> (no 대신) '묶음 뜻'인 표본 순번. 나머지는 '아니오'
  irony: 7 8        -> 반어·농담인 표본 순번
  hitno: 21:3 9:1,2 -> 단위는 '예'지만 그 적중(k)은 묶음 뜻이 아닌 경우
  pol: +0-...       -> BULL/BEAR/score200 표본의 강세(+)/약세(-)/중립(0), 순번대로
순번은 blind_samples.json 안에서 그 묶음 표본이 나오는 순서다(score200은 score_samples 순서).
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "results", "blind_samples.json")
RAW = os.path.join(BASE, "results", "labels_1_raw")
OUT = os.path.join(BASE, "results", "labels_1.json")
POL = {"+": "강세", "-": "약세", "0": "중립"}


def parse(path):
    spec = {"no": None, "yes": None, "irony": set(), "hitno": {}, "pol": None}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            key, val = key.strip(), val.strip()
            if key in ("no", "yes", "irony"):
                nums = {int(x) for x in val.split()} if val else set()
                spec[key] = nums
            elif key == "hitno":
                for tok in val.split():
                    idx, ks = tok.split(":")
                    spec["hitno"][int(idx)] = {int(k) for k in ks.split(",")}
            elif key == "pol":
                spec["pol"] = val
    return spec


def main():
    d = json.load(open(SRC, encoding="utf-8"))
    groups = {}
    for s in d["samples"]:
        groups.setdefault(s["bundle"], []).append(s)
    groups["SCORE"] = d["score_samples"]

    out, problems = [], []
    for bundle, lst in groups.items():
        path = os.path.join(RAW, f"{bundle}.txt")
        if not os.path.exists(path):
            problems.append(f"{bundle}: raw 파일 없음")
            continue
        spec = parse(path)
        n = len(lst)
        for key in ("no", "yes", "irony"):
            bad = [i for i in (spec[key] or set()) if not 1 <= i <= n]
            if bad:
                problems.append(f"{bundle}: {key} 범위 밖 {bad}")
        if spec["no"] is not None and spec["yes"] is not None:
            problems.append(f"{bundle}: no와 yes를 함께 씀")
        need_pol = bundle in ("BULL", "BEAR", "SCORE")
        if need_pol and (spec["pol"] is None or len(spec["pol"]) != n):
            problems.append(f"{bundle}: pol 길이 {len(spec['pol'] or '')} != {n}")
        for i, s in enumerate(lst, 1):
            rec = {"id": s["id"], "set": s["set"], "bundle": s.get("bundle", "score200")}
            if bundle == "SCORE":
                rec["in_sense"] = None
            elif spec["yes"] is not None:
                rec["in_sense"] = i in spec["yes"]
            else:
                rec["in_sense"] = i not in (spec["no"] or set())
            rec["irony"] = i in spec["irony"]
            rec["polarity"] = POL[spec["pol"][i - 1]] if need_pol else None
            hits = s.get("hits", [])
            if len(hits) > 1:
                hn = spec["hitno"].get(i, set())
                badk = hn - {h["k"] for h in hits}
                if badk:
                    problems.append(f"{bundle} #{i}: hitno k {sorted(badk)} 없음")
                rec["hits_in_sense"] = {str(h["k"]): (rec["in_sense"] and h["k"] not in hn) for h in hits}
            elif spec["hitno"].get(i):
                problems.append(f"{bundle} #{i}: 적중 1개인데 hitno 있음")
            if spec["hitno"].get(i) and not rec["in_sense"]:
                problems.append(f"{bundle} #{i}: 단위 '아니오'인데 hitno 있음")
            out.append(rec)

    if problems:
        print("문제:", *problems, sep="\n  ")
        sys.exit(1)
    ids = [r["id"] for r in out]
    assert len(ids) == len(set(ids)) == len(d["samples"]) + len(d["score_samples"])
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("저장", OUT, len(out))


if __name__ == "__main__":
    main()
