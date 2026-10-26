"""'원래 지표(강의 그대로)'로 되돌릴 때의 2026 성적 (2026-10-23 사용자 "수익률 개선한다고 너무 보수적으로 갔다, 원래 지표로").

설정 셋을 같은 1시간봉·같은 기간으로 잰다(2026 상반기·하반기, 비용 0).
  10/22판: 4셋업(추세선·채널 거짓 돌파, 채널 4P, 돌파 뒤 리테스트), 손절 여유 0.1 ATR, 손익비 1 이상만
  강의 그대로: 같은 4셋업, 손익비 제한 없음
  강의 그대로 + 전부: 유동성 스윕 뒤 반대(A-LQ)·오더블럭·FVG 첫 닿음(B-OB·C-FVG)까지 켬
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

K4 = ["A-TL", "A-CH", "E-CH4", "D-RT"]
K7 = K4 + ["A-LQ", "B-OB", "C-FVG"]
V01 = bt.Var(buf=0.1, exit="full", minrr=0.0, ch_close=False)
SETS = [("10/22판 (손익비 1+)", K4, 1.0), ("강의 그대로", K4, 0.0), ("강의 그대로 + 전부", K7, 0.0)]


def cands(sym, cfg):
    eng = bt.Engine(sym, "1h")
    sub = bt.Sub(sym, cfg["sub"]) if cfg["sub"] else None
    cs = []
    for s in eng.setups:
        base, _, mk = s.kind.partition(":")
        if mk or base not in K7:
            continue
        f = bt.fill(eng, s)
        if f is None:
            continue
        r = bt.simulate(eng, s, V01, cfg["fee"], f, sub, cfg["path"])
        if r is None:
            continue
        cs.append(dict(kind=base, t_req=f[0] if base in ("E-CH4", "B-OB", "C-FVG") else s.i, fj=f[0], xj=f[0] + r["hold"], rr1=r["rr1"],
                       R0=r["Rg"], R5=r["R"], t=r["t_fill"], te=r["t_exit"]))
    cs.sort(key=lambda c: (c["t_req"], c["fj"]))
    return cs


def one(sym, days=180):
    """종목 하나의 최근 days일 매수·매도 수와 합계 R (사용자 차트 대조용). 오늘 기본값(손절 여유 0.5·손익비 1+)도 함께."""
    cfg = tu.GROUPS["미국AI115"]
    cs = cands(sym, cfg)
    start = max(c["t"] for c in cs) - days * 86400000
    res = tu.build(sym, cfg)["buf50"]
    rows = [("여유 0.5 + 손익비 1+ (10/23 오전 기본값)", y.follow(res, set(K4), 1.0))] + [(nm, y.follow(cs, set(k), rr)) for nm, k, rr in SETS]
    for nm, tr in rows:
        xs = [x for x in tr if x["t"] >= start]
        print(f"{sym} 최근 {days}일 {nm:36} {len(xs):3}건 합계 {sum(x['R0'] for x in xs):+.1f}R 승률 {sum(1 for x in xs if x['R0'] > 0) / max(1, len(xs)) * 100:.0f}%")


def main():
    L = ["# 원래 지표(강의 그대로)로 되돌릴 때 — 1시간봉 2026", "", "칸 = 평균 R (거래 수) · 승률, 비용 0. 손절 여유 0.1 ATR.", "",
         "| 묶음 | 설정 | 상반기 | 하반기 |", "|---|---|---|---|"]
    for g, cfg in tu.GROUPS.items():
        rows = {nm: {1: [], 2: []} for nm, _, _ in SETS}
        for sym in cfg["syms"]:
            try:
                cs = cands(sym, cfg)
            except FileNotFoundError:
                continue
            bad = {dt.date.fromisoformat(x) for x in json.loads((bt.HIST / f"{sym}_bad.json").read_text())} if cfg["us"] else set()
            for nm, kinds, rr in SETS:
                for x in y.follow(cs, set(kinds), rr):
                    p = y.half(x["t"])
                    if p and not y.crosses_bad(x, bad):
                        rows[nm][p].append(x)
        for nm, _, _ in SETS:
            a, b = y.st(rows[nm][1]), y.st(rows[nm][2])
            L.append(f"| {g} | {nm} | {a['avg']:+.2f} ({a['n']}) · {a['win']*100:.0f}% | {b['avg']:+.2f} ({b['n']}) · {b['win']*100:.0f}% |")
        print(g, flush=True)
    txt = "\n".join(L)
    (tu.OUT / "original_2026.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    if len(sys.argv) > 1:          # py code/bt_original.py IREN
        one(sys.argv[1])
    else:
        main()
