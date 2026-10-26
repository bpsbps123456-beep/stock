"""반박 검증 — 거래량(투매) 계열 최선안 'V1 투매봉 TOD k4 손실0% +60분안'을 정의 문장만 보고 다시 만든다(2026-10-07).

정의(연구자 보고): 봉 i 시가에 판다. 신호 봉 j=i-1(j >= ei)에서
  (1) j 정규장(ET 9:30~15:59) 음봉 c<o
  (2) v[j] >= k x TOD[j]. TOD[j] = 그 봉이 덮는 분마다 '같은 분의 지난 N거래일 평균'(오늘 제외, 최소 3일)의 합.
      분 단위 거래량: 1분봉이 있는 날은 1분봉, 그 밖은 5분봉 거래량/5를 5분에 고르게.
  (3) c[j] <= 매수가 x (1-g)
  (4) i - ei <= W분/봉 길이
이 파일은 분 표를 '봉 칸(slot) 표'로 바로 모아(칸 합 = 분 평균의 합) 지난 N일 평균을 낸다. 연구자 코드는 쓰지 않는다.
사용: py stops/verify_volume.py → ../results/stops/verify_volume.txt"""
import os, sys, time
from datetime import date
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from stop_lab import load, compare, stop_audit, base_trades, run, RAW, UNIV

OUT = os.path.join(HERE, "..", "..", "results", "stops", "verify_volume.txt")
U78 = UNIV["사용자78"]
_SLOT = {}


def day_minute_volume(sym, upto=None):
    """날짜 x 정규장 분(570..959) 거래량 표. upto(date)가 있으면 그날까지의 원자료만(미래 참조 검사용)."""
    def regular(df):
        mod = df.index.hour * 60 + df.index.minute
        df = df[(mod >= 570) & (mod < 960)]
        return pd.DataFrame({"day": df.index.date, "mod": df.index.hour * 60 + df.index.minute,
                             "v": df.Volume.to_numpy(dtype=float)})
    f5 = regular(pd.read_parquet(os.path.join(RAW, f"5m_{sym}.parquet")))
    f5["v"] = f5["v"] / 5.0
    spread = pd.concat([f5.assign(mod=f5["mod"] + q) for q in range(5)], ignore_index=True)
    spread = spread[spread["mod"] < 960]
    cols = list(range(570, 960))
    tab = spread.pivot_table(index="day", columns="mod", values="v", aggfunc="sum").reindex(columns=cols).fillna(0.0)
    p1 = os.path.join(RAW, f"1m_{sym}.parquet")
    if os.path.exists(p1):
        f1 = regular(pd.read_parquet(p1))
        if len(f1):
            t1 = f1.pivot_table(index="day", columns="mod", values="v", aggfunc="sum").reindex(columns=cols).fillna(0.0)
            tab = pd.concat([tab.drop(index=[d for d in t1.index if d in tab.index]), t1]).sort_index()
    if upto is not None:
        tab = tab[tab.index <= upto]
    return tab


def slot_baseline(sym, mins, N, upto=None):
    """날짜 x 봉 칸 기준선: 칸 거래량의 지난 N거래일 평균(오늘 제외, 3일 미만이면 NaN)."""
    key = (sym, mins, N, upto)
    if key in _SLOT:
        return _SLOT[key]
    tab = day_minute_volume(sym, upto)
    slot_of = (np.asarray(tab.columns) - 570) // mins
    sv = tab.T.groupby(slot_of).sum().T.to_numpy()          # 날짜 x 칸
    base = np.full(sv.shape, np.nan)
    for d in range(len(sv)):
        prev = sv[max(0, d - N):d]
        if len(prev) >= 3:
            base[d] = prev.mean(axis=0)
    out = (list(tab.index), base)
    if upto is None:
        _SLOT[key] = out
    return out


def bar_baseline(A, N=10, upto=None):
    days, base = slot_baseline(A["sym"], A["mins"], N, upto)
    pos = {d: k for k, d in enumerate(days)}
    mins = A["mins"]
    b = np.full(len(A["c"]), np.nan)
    for j in np.flatnonzero(A["reg"]):
        m = int(A["mod"][j])
        assert (m - 570) % mins == 0, (A["sym"], m)
        p = pos.get(A["day"][j])
        if p is not None:
            b[j] = base[p, (m - 570) // mins]
    b[~(b > 0)] = np.nan
    return b


def make(k=4.0, g=0.0, W=60, N=10):
    def make_rule(A):
        o, c, v, reg = A["o"], A["c"], A["v"], A["reg"]
        b = bar_baseline(A, N)
        sig = reg & (c < o) & np.where(np.isnan(b), False, v >= k * np.nan_to_num(b, nan=np.inf))
        lim = W // A["mins"]
        gate = 1.0 - g

        def rule(i, ei, ent):
            j = i - 1
            return j >= ei and (i - ei) <= lim and bool(sig[j]) and c[j] <= ent * gate
        return rule
    return make_rule


def day_split(mk, period, reg_only=False):
    S = load(period)
    t = run(S, mk, period, reg_only, syms=U78)
    b = base_trades(period, reg_only)
    b = b[b.sym.isin(U78)]
    days = sorted(set(t.day) | set(b.day))
    d = t.groupby("day").ret.sum().reindex(days, fill_value=0) - b.groupby("day").ret.sum().reindex(days, fill_value=0)
    return d


def main():
    t0 = time.time()
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    # 0) 미래 참조 검사: 원자료를 그날까지만 남기고 만든 기준선 == 전체 원자료 기준선(그날 봉에서)
    S_late = load("late")
    bad = tested = 0
    for sym in ("NVDA", "AAOI", "TEM", "IREN"):
        if sym not in S_late:
            continue
        A = S_late[sym]
        full = bar_baseline(A)
        for D in (date(2026, 9, 15), date(2026, 9, 24), date(2026, 10, 1)):
            cut = bar_baseline(A, upto=D)
            m = np.array([x == D for x in A["day"]])
            x, y = full[m], cut[m]
            ok = ~(np.isnan(x) & np.isnan(y))
            tested += int(ok.sum())
            bad += int((~np.isclose(x[ok], y[ok])).sum())
    out(f"미래 참조 검사(기준선, 그날까지 자른 원자료 vs 전체): {tested}개 비교, 불일치 {bad}개")

    # 1) 재현
    out("\n[재현] V1 투매봉 TOD k4 손실0% +60분안 (독립 구현)")
    best = make(4.0, 0.0, 60, 10)
    df = compare(best, "재현", show=False)
    show = df[["기간", "범위", "종목", "건수", "합", "차이", "t", "손절", "최악", "차이_10월6일"]].copy()
    out(show.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    for p in ("early", "late"):
        for reg in (False, True):
            au = stop_audit(best, p, reg)
            out(f"stop_audit {p} {'정규장만' if reg else '24시간'} 사용자78: " +
                ", ".join(f"{kk} {vv:.3f}" if isinstance(vv, float) else f"{kk} {vv}" for kk, vv in au.items()))

    # 2) 하루 의존
    out("\n[하루 의존] 사용자78 날짜별 차이(규칙 - 기준)")
    for p in ("early", "late"):
        for reg in (False, True):
            d = day_split(best, p, reg)
            top = d.sort_values(ascending=False)
            out(f"{p} {'정규장만' if reg else '24시간'}: 합 {d.sum():+.3f}, 최대 이득일 {top.index[0]} {top.iloc[0]:+.3f}, "
                f"그날 빼면 {d.sum() - top.iloc[0]:+.3f}, 상위2일 빼면 {d.sum() - top.iloc[:2].sum():+.3f}, "
                f"최대 손해일 {top.index[-1]} {top.iloc[-1]:+.3f}, 나은날 {(d > 1e-12).sum()} 나쁜날 {(d < -1e-12).sum()}")

    # 3) 이웃 매개변수
    out("\n[이웃 매개변수] 사용자78 차이(24시간 / 정규장만), 137 24시간")
    nb = [("k3", dict(k=3.0)), ("k5", dict(k=5.0)), ("k3.5", dict(k=3.5)), ("k4.5", dict(k=4.5)),
          ("손실1%(c<=0.99ent)", dict(g=0.01)), ("손실-1%(c<=1.01ent)", dict(g=-0.01)),
          ("30분안", dict(W=30)), ("45분안", dict(W=45)), ("90분안", dict(W=90)),
          ("TOD 5일", dict(N=5)), ("TOD 20일", dict(N=20))]
    rows = []
    for name, kw in nb:
        p = dict(k=4.0, g=0.0, W=60, N=10)
        p.update(kw)
        d2 = compare(make(**p), name, show=False)
        u = d2[d2.종목 == "사용자78"].set_index(["기간", "범위"])
        a = d2[d2.종목 == "전체137"].set_index(["기간", "범위"])
        r = dict(이웃=name, e24=u.loc[("early", "24시간"), "차이"], eReg=u.loc[("early", "정규장만"), "차이"],
                 l24=u.loc[("late", "24시간"), "차이"], lReg=u.loc[("late", "정규장만"), "차이"],
                 A_e24=a.loc[("early", "24시간"), "차이"], A_l24=a.loc[("late", "24시간"), "차이"],
                 e24_손절=u.loc[("early", "24시간"), "손절"], l24_손절=u.loc[("late", "24시간"), "손절"])
        rows.append(r)
        out(f"{name:<22} early24 {r['e24']:+.2f} earlyReg {r['eReg']:+.2f} | late24 {r['l24']:+.2f} lateReg {r['lReg']:+.2f} "
            f"| 137 e24 {r['A_e24']:+.2f} l24 {r['A_l24']:+.2f} (손절 e{r['e24_손절']:.0f} l{r['l24_손절']:.0f}) [{time.time() - t0:.0f}s]")
    R = pd.DataFrame(rows)
    out(f"이웃 {len(R)}개 중 early24 플러스 {(R.e24 > 0).sum()}개, earlyReg 플러스 {(R.eReg > 0).sum()}개, "
        f"late24 플러스 {(R.l24 > 0).sum()}개, lateReg 플러스 {(R.lReg > 0).sum()}개, 네 칸 모두 플러스 "
        f"{((R.e24 > 0) & (R.eReg > 0) & (R.l24 > 0) & (R.lReg > 0)).sum()}개")
    out(f"\n총 {time.time() - t0:.0f}초")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
