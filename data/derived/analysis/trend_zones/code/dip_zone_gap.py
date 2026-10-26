"""추세 단계마다 다른 이격 % (2026-10-11, 연구용, dip_by_zone·dip_gap_limit 다음 단계).

사용자(10/11): "큰 추세선(5단계)에 따라서 다른 이격 매수 %를 설정하는 것."
자료·규칙: dip_by_zone.py와 같다(136종목, 24시간, 전날 +7%면 정규장 쉼, 매도 = RSI>70 다음 봉 또는 아침 8시).
  창 1(고르기) '5분 근사' 7/20~9/8, 창 2(확인) '3분 정확' 9/9~10/9. 단계 = 매수 순간의 일봉 추세 5단계(trend_5zone.pine과 같음).
사전 등록
  단계 묶음 4개: 극상방 / 상방 / 애매(+계산 전) / 하방·극하방(극하방은 창 2에 0건이라 묶음).
  이격 후보: 1.5, 2.0, 2.5, 3.0, 3.5, 4.0%.
  고르기 목표(창 1): 묶음마다 Σ(수익 − c), c = 창 1에서 지금 규칙(2%)의 건당 평균. 자리가 한정된 계좌에서 '평균보다 나은 매매만
     가치가 있다'는 대리값이다(c = 0이면 얕은 이격, 건당 평균이면 가장 깊은 이격만 고르게 된다). 묶음별로 따로 고른다
     (같은 이격으로 돌린 매매를 매수 순간 단계로 나눠 계산).
  비교 규칙(정확히 다시 돌림): P0 지금(전부 2%) | P1 같은 목표로 고른 '모든 단계 같은 이격' | P2 단계별로 고른 이격 |
     P3 직관형(극상방 2.0·상방 2.0·애매 2.5·하방계 3.0, 극하방만 3.5).
  평가(두 창): 건수·합·건당·정규장 건당·최악, 동시 K자리(5·10·20) 자금 제약(매수 시각 순 선착, 자리당 같은 돈, 합/K).
실행: PYTHONIOENCODING=utf-8 py data/derived/analysis/trend_zones/code/dip_zone_gap.py
결과: results/dip_zone_gap.txt
"""
import os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import zones as Zm  # noqa: E402
import dip_by_zone as DZ  # noqa: E402
from dip_by_zone import bars, prep, hot_days, RAW, WIN, NY, sim  # noqa: E402

GAPS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
GROUPS = {"극상방": (2,), "상방": (1,), "애매": (0, -9), "하방·극하방": (-1, -2)}
W1, W2 = list(WIN)


def group_of(z):
    for g, zs in GROUPS.items():
        if z in zs:
            return g
    return "애매"


def load(S, ddates, dclose, Z):
    """창마다 종목별 (d, t, zone, hot)를 한 번만 만든다."""
    out = {}
    for wname, (mins, lo, hi) in WIN.items():
        ema_len, rsi_len = (60, 14) if mins == 3 else (36, 9)
        for j, s in enumerate(S):
            d = prep(bars(RAW, s, mins), ema_len, rsi_len)
            t = d.index.tz_convert(NY)
            k = np.searchsorted(dclose, t.as_unit("ns").asi8, side="right") - 1
            zone = np.where(k >= 0, Z[np.maximum(k, 0), j], -9)
            out[wname, s] = (d, t, zone, hot_days(RAW, s, d, 0.07), lo, hi)
    return out


def run(data, wname, gap_of_zone, no_after=False):
    """gap_of_zone: {단계값: 이격} → 매매 표. no_after: 애프터장(16~20시 ET) 매수 안 함(사용자 주문 시간 08~24시 KST와 거의 안 겹침)."""
    rows = []
    for (w, s), (d, t, zone, hot, lo, hi) in data.items():
        if w != wname:
            continue
        gap = np.array([gap_of_zone.get(int(z), gap_of_zone.get("기타")) for z in zone], float) / 100
        if no_after:
            gap = np.where(d.ses.values == "애프터", np.inf, gap)
        for ei, xi, ent, px, why, mae in sim(d, gap, hot):
            if lo <= t[ei].date() <= hi:
                rows.append((s, t[ei], t[xi], d.ses.iloc[ei], px / ent - 1, int(zone[ei])))
    return pd.DataFrame(rows, columns=["sym", "tin", "tout", "ses", "ret", "zone"])


def kslot(T, K):
    T = T.sort_values(["tin", "sym"])
    open_, tot = [], 0.0
    for tin, tout, r in zip(T.tin, T.tout, T.ret):
        open_ = [x for x in open_ if x > tin]
        if len(open_) < K:
            open_.append(tout)
            tot += r
    return tot / K


def uniform(g):
    return {z: g for z in (2, 1, 0, -1, -2, -9)}


def policy(gmap):
    """{묶음: 이격} → {단계값: 이격}."""
    return {z: gmap[g] for g, zs in GROUPS.items() for z in zs}


def main():
    S = DZ.syms()
    ddates, dclose, Z, _ = DZ.daily_context(S)
    data = load(S, ddates, dclose, Z)
    lines = []
    # 1) 고르기(창 1): 같은 이격으로 돌린 매매를 단계 묶음으로 나눈다
    T1 = {g: run(data, W1, uniform(g)) for g in GAPS}
    c = T1[2.0].ret.mean()
    lines += [f"== 고르기(창 1 {W1}) — 묶음별 Σ(수익 − c), c = 지금 규칙 건당 {c * 100:.2f}%  [건수 · 건당]"]
    best = {}
    for gname, zs in GROUPS.items():
        cells, sc = [], {}
        for g in GAPS:
            x = T1[g][T1[g].zone.isin(zs)]
            sc[g] = (x.ret - c).sum()
            cells.append(f"{g:.1f}%: {sc[g] * 100:+7.1f} [{len(x)}·{x.ret.mean() * 100:+.2f}]")
        best[gname] = max(sc, key=sc.get)
        lines.append(f"  {gname:<7} " + " | ".join(cells) + f"  → {best[gname]:.1f}%")
    usc = {g: (T1[g].ret - c).sum() for g in GAPS}
    ubest = max(usc, key=usc.get)
    lines.append(f"  모든 단계 같은 이격: " + " | ".join(f"{g:.1f}% {usc[g] * 100:+.1f}" for g in GAPS) + f"  → {ubest:.1f}%")
    P = {"P0 지금(전부 2%)": uniform(2.0),
         f"P1 모두 같은 이격({ubest:.1f}%)": uniform(ubest),
         "P2 단계별(" + "·".join(f"{k} {v:.1f}" for k, v in best.items()) + ")": policy(best),
         "P3 직관형(상방2·애매2.5·하방3·극하방3.5)": {2: 2.0, 1: 2.0, 0: 2.5, -9: 2.5, -1: 3.0, -2: 3.5}}
    P["P4 모두 3%(참고)"] = uniform(3.0)
    # 2) 정확히 다시 돌려 두 창에서 비교. 사후 추가(검증 지적 뒤): 애프터장 매수를 끈 판
    for na, wname in [(na, w) for na in (False, True) for w in (W1, W2)]:
        lines += ["", f"== {wname} {'(고른 창)' if wname == W1 else '(확인 창)'}{' · 애프터 매수 안 함(사후)' if na else ' · 24시간'}",
                  f"  {'규칙':<44} {'건수':>5} {'합':>9} {'건당':>7} {'정규장 건당':>11} {'최악':>7} | K5 / K10 / K20"]
        for pname, gm in P.items():
            T = run(data, wname, gm, no_after=na)
            reg = T[T.ses == "정규"]
            ks = " / ".join(f"{kslot(T, K) * 100:+6.1f}%" for K in (5, 10, 20))
            lines.append(f"  {pname:<44} {len(T):5d} {T.ret.sum() * 100:+8.1f}% {T.ret.mean() * 100:+6.2f}% "
                         f"{reg.ret.mean() * 100:+6.2f}%({len(reg)}) {T.ret.min() * 100:+6.1f}% | {ks}")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(Zm.OUT, "dip_zone_gap.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt)


if __name__ == "__main__":
    main()
