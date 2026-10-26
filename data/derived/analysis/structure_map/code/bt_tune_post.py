"""사후 점검(2026-10-23): 12시간 방향 필터를 어디에 거느냐에 따라 결과가 흔들리는지.

bt_tune.py에서 H1(4셋업 모두, 채널 4P는 선 확정 봉의 방향)은 통과했지만 H3(채널 4P는 닿은 봉의 방향)은 하반기에서 떨어졌다.
개선이 거짓 돌파 되돌림(A-TL)에서 나왔으므로, 필터를 되돌림에만 건 변형을 2025(보조)·2026 상·하반기·최근 60일로 본다. 결과를 본 뒤 만든 변형이다.
"""
from __future__ import annotations

import calendar
import datetime as dt
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402
import bt_2026 as y  # noqa: E402
import bt_tune as tu  # noqa: E402

P0 = calendar.timegm((2025, 2, 1, 0, 0, 0)) * 1000
P60 = calendar.timegm((2026, 7, 30, 0, 0, 0)) * 1000
V = {
    "기준": lambda c: True,
    "H1 4셋업 모두 (4P는 선 확정 봉)": lambda c: c["dir12"] == c["d"],
    "H3 4셋업 모두 (4P는 닿은 봉)": lambda c: (c["dir12f"] if c["kind"] == "E-CH4" else c["dir12"]) == c["d"],
    "H4 되돌림(A-TL·A-CH)에만": lambda c: c["kind"] not in tu.A or c["dir12"] == c["d"],
    "H5 추세선 되돌림(A-TL)에만": lambda c: c["kind"] != "A-TL" or c["dir12"] == c["d"],
}


def per(t):
    return "2025" if P0 <= t < y.P1 else "26H1" if y.P1 <= t < y.P2 else "26H2" if t >= y.P2 else None


def main():
    L = ["# 12시간 방향 필터 — 어디에 거느냐 (사후 점검)", "", "칸 = 평균 R (거래 수), 비용 0.", ""]
    for g, cfg in tu.GROUPS.items():
        rows = {k: [] for k in V}
        for sym in cfg["syms"]:
            try:
                res = tu.build(sym, cfg)
            except FileNotFoundError:
                continue
            bad = {dt.date.fromisoformat(x) for x in json.loads((bt.HIST / f"{sym}_bad.json").read_text())} if cfg["us"] else set()
            for nm, flt in V.items():
                rows[nm] += [x for x in y.follow([c for c in res["base"] if flt(c)], set(y.KINDS), 1.0) if per(x["t"]) and not y.crosses_bad(x, bad)]
        L += [f"## {g}", "", "| 변형 | 2025 (보조) | 2026 상반기 | 2026 하반기 | 최근 60일 |", "|---|---|---|---|---|"]
        for nm, xs in rows.items():
            cs = [f"{s['avg']:+.2f} ({s['n']})" for s in [y.st([x for x in xs if per(x["t"]) == p]) for p in ("2025", "26H1", "26H2")] + [y.st([x for x in xs if x["t"] >= P60])]]
            L.append(f"| {nm} | " + " | ".join(cs) + " |")
        L.append("")
    txt = "\n".join(L)
    (tu.OUT / "tune_post.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
