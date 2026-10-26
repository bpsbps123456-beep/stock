"""Fake와 Trap을 나누는 게 의미 있나 (2026-10-23 사용자 질문).

거짓 돌파 되돌림 매매(A-TL 추세선·A-CH 채널)를 판정 종류별로 나눠 2026 상·하반기 성적을 본다.
  꼬리 Fake: 종가는 선 안, 꼬리만 0.3 ATR 넘게 나갔다 들어옴 / Fake: 밖에서 마감 1~3봉 뒤 복귀 / Trap: 4~12봉 뒤 복귀
지표 기본값(강의 그대로: 손절 버퍼 0.1 ATR, 손익비 제한 없음), 1시간봉, 비용 0. 신호마다 따로 센다(한 포지션 규칙 없이).
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402
import bt_2026 as y  # noqa: E402
import bt_tune as tu  # noqa: E402

NAME = {"wick": "꼬리 Fake", "fake": "Fake (1~3봉)", "trap": "Trap (4~12봉)"}


def main():
    L = ["# Fake와 Trap — 거짓 돌파 되돌림 성적 (1시간봉, 2026)", "", "칸 = 평균 R (신호 수) · 승률, 비용 0, 손절 버퍼 0.1 ATR, 손익비 제한 없음.", "",
         "| 묶음 | 판정 | 상반기 | 하반기 | 손절 폭 중앙값(ATR) |", "|---|---|---|---|---|"]
    for g, cfg in tu.GROUPS.items():
        rows = {k: {1: [], 2: []} for k in NAME}
        risk = {k: [] for k in NAME}
        for sym in cfg["syms"]:
            try:
                res = tu.build(sym, cfg)
            except FileNotFoundError:
                continue
            bad = {dt.date.fromisoformat(x) for x in json.loads((bt.HIST / f"{sym}_bad.json").read_text())} if cfg["us"] else set()
            for c in res["base"]:
                if c["kind"] not in tu.A or c["typ"] not in NAME:
                    continue
                p = y.half(c["t"])
                if p and not y.crosses_bad(c, bad):
                    rows[c["typ"]][p].append(c)
                    if c["R0"] != c["R5"]:
                        risk[c["typ"]].append(2 * cfg["fee"] / (c["R0"] - c["R5"]))   # 손절 폭 ÷ 가격
        for k, nm in NAME.items():
            a, b = y.st(rows[k][1]), y.st(rows[k][2])
            rs = sorted(risk[k])
            L.append(f"| {g} | {nm} | {a['avg']:+.2f} ({a['n']}) · {a['win']*100:.0f}% | {b['avg']:+.2f} ({b['n']}) · {b['win']*100:.0f}% | 가격의 {rs[len(rs)//2]*100:.2f}% |" if rs else f"| {g} | {nm} | - | - | - |")
        print(g, flush=True)
    txt = "\n".join(L)
    (tu.OUT / "fake_trap.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
