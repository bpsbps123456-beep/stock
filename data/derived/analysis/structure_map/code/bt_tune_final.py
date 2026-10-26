"""지표에 넣을 최종 기본값 점검 (2026-10-23): 코인 = 손절 여유 0.1 + 되돌림(A-TL·A-CH)은 12시간 추세 쪽만,
미국 주식 = 손절 여유 0.5 + 같은 필터. 2025(보조)·2026 상·하반기·최근 60일."""
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
H4 = lambda c: c["kind"] not in tu.A or c["dir12"] == c["d"]
SETS = {
    "코인": [("기준(여유 0.1)", "base", lambda c: True), ("최종: 여유 0.1 + 되돌림 12H 쪽만", "base", H4)],
    "미국AI115": [("기준(여유 0.1)", "base", lambda c: True), ("여유 0.5", "buf50", lambda c: True),
                 ("최종: 여유 0.5 + 되돌림 12H 쪽만", "buf50", H4)],
}


def per(t):
    return "2025" if P0 <= t < y.P1 else "26H1" if y.P1 <= t < y.P2 else "26H2" if t >= y.P2 else None


def main():
    L = ["# 지표 최종 기본값 점검", "", "칸 = 평균 R (거래 수) · 승률, 비용 0.", ""]
    for g, cfg in tu.GROUPS.items():
        rows = {nm: [] for nm, _, _ in SETS[g]}
        for sym in cfg["syms"]:
            try:
                res = tu.build(sym, cfg)
            except FileNotFoundError:
                continue
            bad = {dt.date.fromisoformat(x) for x in json.loads((bt.HIST / f"{sym}_bad.json").read_text())} if cfg["us"] else set()
            for nm, vk, flt in SETS[g]:
                rows[nm] += [x for x in y.follow([c for c in res[vk] if flt(c)], set(y.KINDS), 1.0) if per(x["t"]) and not y.crosses_bad(x, bad)]
        L += [f"## {g}", "", "| 설정 | 2025 (보조) | 2026 상반기 | 2026 하반기 | 최근 60일 |", "|---|---|---|---|---|"]
        for nm, xs in rows.items():
            cs = [f"{s['avg']:+.2f} ({s['n']}) · {s['win']*100:.0f}%" for s in
                  [y.st([x for x in xs if per(x["t"]) == p]) for p in ("2025", "26H1", "26H2")] + [y.st([x for x in xs if x["t"] >= P60])]]
            L.append(f"| {nm} | " + " | ".join(cs) + " |")
        L.append("")
    txt = "\n".join(L)
    (tu.OUT / "tune_final.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
