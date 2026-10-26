"""12시간 추세 방향 필터(bt_tune.py H1) 점검 — 코인 1시간봉 (2026-10-23).

2026 상·하반기를 통과한 유일한 조건이라 2025(2~12월, 보조 확인)·최근 60일·코인별·무작위 진입(같은 방향 필터)으로 다시 본다.
무작위 진입도 12시간 방향만 남겼을 때 같은 만큼 좋아지면, 개선은 셋업이 아니라 '추세 방향'의 몫이다.
"""
from __future__ import annotations

import calendar
import pathlib
import random
import sys
import zlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402
import bt_2026 as y  # noqa: E402
import bt_tune as tu  # noqa: E402

OUT = tu.OUT
P0 = calendar.timegm((2025, 2, 1, 0, 0, 0)) * 1000
P60 = calendar.timegm((2026, 7, 30, 0, 0, 0)) * 1000
VARR = bt.Var(buf=0.0, exit="full", minrr=1.0, ch_close=False)


def per(t):
    return "2025" if P0 <= t < y.P1 else "26H1" if y.P1 <= t < y.P2 else "26H2" if t >= y.P2 else None


def main():
    cfg = tu.GROUPS["코인"]
    keep = lambda c: c["dir12"] == c["d"]
    rows = {}
    rnd = []
    for sym in cfg["syms"]:
        res = tu.build(sym, cfg)
        for nm, flt in (("기준", lambda c: True), ("12H 방향만", keep)):
            xs = [x for x in y.follow([c for c in res["base"] if flt(c)], set(y.KINDS), 1.0) if per(x["t"])]
            rows.setdefault(nm, []).extend(xs)
        # 무작위 진입: 2025-02~ 아무 봉 다음 시가, 방향 무작위, 손절 폭은 실제 셋업 분포(ATR 배수)
        eng = bt.Engine(sym, "1h")
        sub = bt.Sub(sym, "5m")
        h12 = eng.htf["dir12"]
        pool = []
        for s in eng.setups:
            base, _, mk = s.kind.partition(":")
            if mk or base not in y.KINDS:
                continue
            f = bt.fill(eng, s)
            if f is not None and per(eng.b[f[0]]["t"]):
                ra = y.risk_atr(eng, s, f)
                if ra:
                    pool.append(ra[1])
        idx = [i for i in range(eng.n - 2) if per(eng.b[i + 1]["t"])]
        rng = random.Random(zlib.crc32(f"{sym}|tune".encode()))
        for _ in range(6 * len(pool)):
            i = rng.choice(idx)
            d = rng.choice((1, -1))
            k = rng.choice(pool)
            px = eng.b[i + 1]["o"]
            so = bt.Setup("rnd", i, d, None, px - d * k * eng.atr[i], i + 1, i + 1, -1, px)
            rr = bt.simulate(eng, so, VARR, 0.0, (i + 1, px, True), sub, "stop")
            if rr is not None:
                rnd.append(dict(t=rr["t_fill"], R0=rr["Rg"], sym=sym, ok=(h12[i] or 0) == d))
        print(sym, flush=True)
    L = ["# 12시간 추세 방향 필터 점검 — 코인 1시간봉", "", "칸 = 평균 R (거래 수) · 승률, 비용 0.", "",
         "| 구분 | 2025 (보조) | 2026 상반기 | 2026 하반기 | 최근 60일 (7/30~) |", "|---|---|---|---|---|"]
    for nm, xs in rows.items():
        cs = []
        for p in ("2025", "26H1", "26H2"):
            s = y.st([x for x in xs if per(x["t"]) == p])
            cs.append(f"{s['avg']:+.2f} ({s['n']}) · {s['win']*100:.0f}%")
        s = y.st([x for x in xs if x["t"] >= P60])
        cs.append(f"{s['avg']:+.2f} ({s['n']}) · {s['win']*100:.0f}%")
        L.append(f"| {nm} | " + " | ".join(cs) + " |")
    for nm, flt in (("무작위 전체", lambda x: True), ("무작위 12H 방향만", lambda x: x["ok"]), ("무작위 12H 반대", lambda x: not x["ok"])):
        cs = []
        for p in ("2025", "26H1", "26H2"):
            s = y.st([x for x in rnd if per(x["t"]) == p and flt(x)])
            cs.append(f"{s['avg']:+.2f} ({s['n']})")
        s = y.st([x for x in rnd if x["t"] >= P60 and flt(x)])
        cs.append(f"{s['avg']:+.2f} ({s['n']})")
        L.append(f"| {nm} | " + " | ".join(cs) + " |")
    L += ["", "| 코인 (2026) | 기준 | 12H 방향만 |", "|---|---|---|"]
    for sym in cfg["syms"]:
        a = y.st([x for x in rows["기준"] if x["sym"] == sym and x["t"] >= y.P1])
        b = y.st([x for x in rows["12H 방향만"] if x["sym"] == sym and x["t"] >= y.P1])
        L.append(f"| {sym} | {a['avg']:+.2f} ({a['n']}) | {b['avg']:+.2f} ({b['n']}) |")
    L += ["", "| 셋업 (2026, 12H 방향만) | 성적 |", "|---|---|"]
    for k in y.KINDS:
        L.append(f"| {k} | {y.cell(y.st([x for x in rows['12H 방향만'] if x['kind'] == k and x['t'] >= y.P1]))} |")
    txt = "\n".join(L)
    (OUT / "tune_check.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
