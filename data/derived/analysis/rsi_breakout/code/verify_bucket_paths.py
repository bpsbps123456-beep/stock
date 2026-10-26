"""감사 보조: verify_bucket_concentration.py에서 나온 이상치 세 가지를 파고든다(2026-10-10).
  A) 최선(T-sub-N5-eq)이 live를 가장 크게 이긴 날(7/30, 4/9, 3/23)에 둘이 무엇을 들었나
  B) 시작일 의존: 1/2 출발 경로의 1/12~5/29 수익 대 1/12 새 출발 수익(최선·live 각각), 첫 몇 주 보유 비교
  C) 버퍼 3/5/8에서 뒤 구간이 같은 이유: 6/1 이후 주간 보유가 같아지는가
실행: cd data/derived/analysis/rsi_breakout/code && PYTHONIOENCODING=utf-8 py verify_bucket_paths.py
"""
import os, sys
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import deskx  # noqa: E402
from bucket_main import BucketEngine, BParams, open_ret  # noqa: E402

PICK = BParams(universe="time", gran="sub", n=5, wmode="eq")
LIVE = BParams(gran="name", wmode="fund", n=7, semis_cap=0.50)


def daily(v):
    return (v.shift(-1) / v - 1).dropna()


def win(r, a, b):
    x = r[(r.index >= a) & (r.index < b)]
    return float((1 + x).prod() - 1)


def held(E, rows, day):
    row = next(r for r in rows if r["day"] == day)
    k = E.days.index(day) + 1
    return sorted(((E.ticker[x], round(wt, 3), round(open_ret(E, x, k), 3)) for x, wt in row["w"].items()),
                  key=lambda z: -z[1])


def main():
    E = BucketEngine()
    vp, rp_rows, lp = E.run_logged(PICK)
    vl, rl_rows, ll = E.run_logged(LIVE)
    rp, rl = daily(vp), daily(vl)

    print("A) 초과 상위 날 보유(티커, 비중, 그날 시가→다음 시가)")
    for d in ("2026-07-30", "2026-04-09", "2026-03-23", "2026-08-13"):
        print(f"  {d}: 최선 {rp[d]:+.3f} {held(E, rp_rows, d)}")
        print(f"  {' ' * 10}  live {rl[d]:+.3f} {held(E, rl_rows, d)}")

    print("\nB) 시작일 의존")
    saved = deskx.START
    try:
        for st in ("2026-01-12", "2026-01-20", "2026-02-02"):
            deskx.START = st
            vp2, rows_p2, lp2 = E.run_logged(PICK)
            vl2, rows_l2, ll2 = E.run_logged(LIVE)
            a0, b0 = win(rp, st, "2026-06-01"), win(rl, st, "2026-06-01")
            a1, b1 = win(daily(vp2), st, "2026-06-01"), win(daily(vl2), st, "2026-06-01")
            print(f"  {st}~5/29  최선: 1/2 경로 {a0:.3f} / 새 출발 {a1:.3f}   live: 1/2 경로 {b0:.3f} / 새 출발 {b1:.3f}")
            same_p = next((e["session"] for e, f in zip(lp[[x["session"] for x in lp].index(lp2[0]["session"]):], lp2)
                           if set(e["keep"]) == set(f["keep"])), None)
            same_l = next((e["session"] for e, f in zip(ll[[x["session"] for x in ll].index(ll2[0]["session"]):], ll2)
                           if set(e["keep"]) == set(f["keep"])), None)
            print(f"     보유가 1/2 경로와 처음 같아지는 주: 최선 {same_p}, live {same_l}")
            for e, f in list(zip(lp[[x["session"] for x in lp].index(lp2[0]["session"]):], lp2))[:3]:
                print(f"     {e['session']} 최선 1/2경로 {sorted(E.ticker[x] for x in e['keep'])} | 새 출발 {sorted(E.ticker[x] for x in f['keep'])}")
    finally:
        deskx.START = saved

    print("\nC) 버퍼별 주간 보유가 같아지는 시점(최선)")
    logs = {}
    for b in (3, 5, 8):
        _, _, lg = E.run_logged(replace(PICK, buffer=b))
        logs[b] = {e["session"]: frozenset(E.ticker[x] for x in e["keep"]) for e in lg}
    sess = sorted(logs[5])
    diff = [s for s in sess if not (logs[3][s] == logs[5][s] == logs[8][s])]
    print(f"  보유가 다른 주 {len(diff)}/{len(sess)}개, 마지막으로 다른 주 {diff[-1] if diff else None}")
    for s in diff[-3:]:
        print(f"   {s}: b3 {sorted(logs[3][s])} b5 {sorted(logs[5][s])} b8 {sorted(logs[8][s])}")
    # 앞 구간 차이를 만든 주
    vb = {b: daily(E.run_logged(replace(PICK, buffer=b))[0]) for b in (3, 8)}
    for b in (3, 8):
        dd = (vb[b] - rp)
        top = dd.abs().sort_values(ascending=False).index[:3]
        print(f"  버퍼 {b} 대 5: 일간 차 큰 날 {[(d, round(float(dd[d]), 3)) for d in top]}")

    print("\nD) 시작 주 쓸기(그 주 첫 거래일에 현금에서 시작): 최선 - live")
    mondays = sorted({e["session"] for e in ll})
    saved = deskx.START
    out = {}
    try:
        for lab, lo, hi, end in (("앞(~5/29)", "2026-01-02", "2026-03-30", "2026-06-01"),
                                 ("뒤(~10/9)", "2026-06-01", "2026-08-03", "2026-12-31")):
            gaps = []
            for st in [m for m in mondays if lo <= m <= hi]:
                deskx.START = st
                a = win(daily(E.run_logged(PICK)[0]), st, end)
                b = win(daily(E.run_logged(LIVE)[0]), st, end)
                gaps.append((st, round(a, 3), round(b, 3), round(a - b, 3)))
            g = np.array([x[3] for x in gaps])
            out[lab] = gaps
            print(f"  {lab}: 시작 {len(g)}개, 최선>live {int((g > 0).sum())}개, 차 중앙값 {np.median(g):+.3f}, "
                  f"최소 {g.min():+.3f}, 최대 {g.max():+.3f}")
            print("    ", gaps)
    finally:
        deskx.START = saved


if __name__ == "__main__":
    main()
