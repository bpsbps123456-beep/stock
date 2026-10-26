"""미국 AI 115 1시간봉 — 손절 여유 0.5 ATR(bt_tune.py S2) 점검 (2026-10-23).

2026 상·하반기를 통과한 미국 조건이라 2025(2~12월, 보조)·최근 60일·셋업별로 다시 본다. 코인에서는 하반기에 나빠졌다(+0.17 → −0.04).
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


def per(t):
    return "2025" if P0 <= t < y.P1 else "26H1" if y.P1 <= t < y.P2 else "26H2" if t >= y.P2 else None


def main():
    cfg = tu.GROUPS["미국AI115"]
    rows = {"기준 0.1": [], "손절 여유 0.5": [], "손절 여유 0.5 + Trap만": []}
    for sym in cfg["syms"]:
        try:
            res = tu.build(sym, cfg)
        except FileNotFoundError:
            continue
        bad = {dt.date.fromisoformat(x) for x in json.loads((bt.HIST / f"{sym}_bad.json").read_text())}
        for nm, vk, flt in (("기준 0.1", "base", lambda c: True), ("손절 여유 0.5", "buf50", lambda c: True),
                            ("손절 여유 0.5 + Trap만", "buf50", lambda c: c["kind"] not in tu.A or c["typ"] == "trap")):
            rows[nm] += [x for x in y.follow([c for c in res[vk] if flt(c)], set(y.KINDS), 1.0) if per(x["t"]) and not y.crosses_bad(x, bad)]
    L = ["# 미국 AI 115 1시간봉 — 손절 여유 점검", "", "칸 = 평균 R (거래 수) · 승률, 비용 0.", "",
         "| 구분 | 2025 (보조) | 2026 상반기 | 2026 하반기 | 최근 60일 (7/30~) |", "|---|---|---|---|---|"]
    for nm, xs in rows.items():
        cs = [f"{s['avg']:+.2f} ({s['n']}) · {s['win']*100:.0f}%" for s in
              [y.st([x for x in xs if per(x["t"]) == p]) for p in ("2025", "26H1", "26H2")] + [y.st([x for x in xs if x["t"] >= P60])]]
        L.append(f"| {nm} | " + " | ".join(cs) + " |")
    L += ["", "| 셋업 (2026) | 기준 0.1 | 손절 여유 0.5 |", "|---|---|---|"]
    for k in y.KINDS:
        a = y.st([x for x in rows["기준 0.1"] if x["kind"] == k and x["t"] >= y.P1])
        b = y.st([x for x in rows["손절 여유 0.5"] if x["kind"] == k and x["t"] >= y.P1])
        L.append(f"| {k} | {a['avg']:+.2f} ({a['n']}) | {b['avg']:+.2f} ({b['n']}) |")
    txt = "\n".join(L)
    (tu.OUT / "tune_us.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
