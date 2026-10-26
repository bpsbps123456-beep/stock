"""반박 검증 — rsipath 최선안 'F_프리만_rsi<30_60분' 독립 재구현(2026-10-07).

정의(연구자 보고 definition만 보고 다시 씀):
  매수 봉 ei가 프리장(ET 분 mod 240 <= mod < 570, 즉 04:00~09:30)일 때만.
  k = m // 봉 길이(m=60분: early 5분봉 12봉, late 3분봉 20봉). 봉 i = ei + k에서 한 번만 확인.
  max(RSI[ei..i-1]) < T (T=30, 종가 기준 RSI: early RSI9, late RSI14)이면 봉 i 시가에 판다.
구현: RSI 길이 k 창 최대값을 sliding_window_view로 미리 계산(wmax[ei] = max r[ei..ei+k-1] = max r[ei..i-1]).
이웃: T 25/35, m 30/45/75/90, 세션(프리+애프터, 전 세션).
사용: py stops/verify_rsipath.py   (결과: ../results/stops/verify_rsipath.txt)
"""
import os, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from stop_lab import compare, stop_audit, load, base_trades, run, UNIV  # noqa: E402

OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "stops", "verify_rsipath.txt"))
U78 = UNIV["사용자78"]


class RF:
    """매수 뒤 m분 동안 RSI가 T에 한 번도 못 닿으면(봉 ei..i-1 최대 < T) i = ei+k 시가에 판다."""

    def __init__(self, T=30.0, m=60, ses="pre"):
        self.T, self.m, self.ses = T, m, ses

    def __call__(self, A):
        k = self.m // int(A["mins"])
        r = np.asarray(A["r"], dtype=float)
        mod = A["mod"]
        if self.ses == "pre":
            ok = (mod >= 240) & (mod < 570)
        elif self.ses == "nonreg":
            ok = (mod < 570) | (mod >= 960)
        else:
            ok = np.ones(len(r), dtype=bool)
        wmax = np.full(len(r), np.nan)
        if len(r) >= k:
            wmax[:len(r) - k + 1] = sliding_window_view(r, k).max(axis=1)   # NaN 있으면 NaN -> 판단 안 함
        sell = ok & (wmax < self.T)

        def rule(i, ei, ent):
            return (i - ei == k) and bool(sell[ei])
        return rule


def label(x):
    s = {"pre": "프리만", "nonreg": "프리+애프터", "all": "전세션"}[x.ses]
    return f"{s}_rsi<{x.T:g}_{x.m}분"


def job(x):
    return label(x), compare(x, label(x), show=False)


def cellmap(df):
    return {(q.기간, q.범위, q.종목): q for q in df.itertuples()}


def day_dep(x, period, reg_only=False):
    t = run(load(period), x, period, reg_only, syms=U78)
    b = base_trades(period, reg_only)
    b = b[b.sym.isin(U78)]
    days = sorted(set(t.day) | set(b.day))
    d = t.groupby("day").ret.sum().reindex(days, fill_value=0) - b.groupby("day").ret.sum().reindex(days, fill_value=0)
    return d.sort_values(ascending=False)


def causal_check(x, period, n=300, seed=3):
    """배열을 [:i]로 자른 A로 만든 규칙의 i 판단 == 전체 배열 규칙의 i 판단인지(프리 매수 표본, 확인 봉 i=ei+k 포함)."""
    S = load(period)
    b = base_trades(period)
    b = b[b.ses == "프리"]
    rng = np.random.default_rng(seed)
    bad = tot = 0
    for q in b.sample(min(n, len(b)), random_state=seed).itertuples():
        if q.xi <= q.ei + 1:
            continue
        A = S[q.sym]
        k = x.m // int(A["mins"])
        i = q.ei + k if q.ei + k <= q.xi and rng.random() < 0.7 else int(rng.integers(q.ei + 1, q.xi + 1))
        want = x(A)(i, q.ei, q.ent)
        Tr = {kk: (A[kk][:i] if isinstance(A[kk], np.ndarray) else A[kk]) for kk in A}
        Tr["mins"] = A["mins"]
        try:
            got = x(Tr)(i, q.ei, q.ent)
        except IndexError:
            got = None
        tot += 1
        bad += (got != want)
    return bad, tot


if __name__ == "__main__":
    t0 = time.time()
    best = RF()
    nbrs = [RF(T=25), RF(T=35), RF(m=30), RF(m=45), RF(m=75), RF(m=90),
            RF(T=25, m=45), RF(T=35, m=75), RF(ses="nonreg"), RF(ses="all")]
    with Pool(6) as pool:
        res = pool.map(job, [best] + nbrs)
    lines = []
    w = lines.append
    w("반박 검증 — rsipath 최선안 독립 재구현 (verify_rsipath) 2026-10-07")
    w(f"시험한 규칙 수(이 검증): {len(res)} (최선안 1 + 이웃 {len(nbrs)})")
    w("")
    nm0, df0 = res[0]
    w(f"== 최선안 재현: {nm0}")
    show = df0[["기간", "범위", "종목", "건수", "합", "차이", "t", "손절", "최악", "차이_10월6일", "나은날", "나쁜날"]]
    w(show.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    w("")
    w("== 이웃 매개변수 (사용자78 차이 / t, 전체137 24시간 차이)")
    rows = []
    for nm, df in res:
        cm = cellmap(df)
        g = lambda p, s: cm[(p, s, "사용자78")]
        rows.append(dict(규칙=nm,
                         e24=g("early", "24시간").차이, e24t=g("early", "24시간").t, eReg=g("early", "정규장만").차이,
                         l24=g("late", "24시간").차이, l24t=g("late", "24시간").t, lReg=g("late", "정규장만").차이,
                         e24_137=cm[("early", "24시간", "전체137")].차이, l24_137=cm[("late", "24시간", "전체137")].차이,
                         손절e=g("early", "24시간").손절, 손절l=g("late", "24시간").손절))
    nb = pd.DataFrame(rows)
    w(nb.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    w(f"네 칸 모두 플러스(사용자78): {int(((nb.e24 > 0) & (nb.eReg > 0) & (nb.l24 > 0) & (nb.lReg > 0)).sum())}개, "
      f"early24 플러스: {int((nb.e24 > 0).sum())}개, late24 플러스: {int((nb.l24 > 0).sum())}개")
    w("")
    w("== stop_audit (사용자78)")
    for p in ("early", "late"):
        for reg in (False, True):
            a = stop_audit(best, p, reg_only=reg)
            w(f"  {p} {'정규장만' if reg else '24시간'}: " + ", ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in a.items()))
    w("")
    w("== 하루 의존 (사용자78, 날짜별 합 차이)")
    for p in ("early", "late"):
        for reg in (False, True):
            d = day_dep(best, p, reg)
            w(f"  {p} {'정규장만' if reg else '24시간'}: 합 {d.sum():+.3f}, 최고일 {d.index[0]} {d.iloc[0]:+.3f}, "
              f"뺀 뒤 {d.sum() - d.iloc[0]:+.3f}, 최악일 {d.index[-1]} {d.iloc[-1]:+.3f}, "
              f"나은날 {(d > 1e-12).sum()} 나쁜날 {(d < -1e-12).sum()}")
            w("     상위3일: " + ", ".join(f"{k}:{v:+.3f}" for k, v in d.head(3).items())
              + " / 하위3일: " + ", ".join(f"{k}:{v:+.3f}" for k, v in d.tail(3).items()))
    w("")
    w("== early 24시간 플러스 이웃의 하루 의존 (사용자78)")
    for x in [RF(m=75), RF(m=90), RF(T=35, m=75)]:
        for p in ("early", "late"):
            d = day_dep(x, p)
            w(f"  {label(x)} {p}: 합 {d.sum():+.3f}, 최고일 {d.index[0]} {d.iloc[0]:+.3f}, 뺀 뒤 {d.sum() - d.iloc[0]:+.3f}, "
              f"나은날 {(d > 1e-12).sum()} 나쁜날 {(d < -1e-12).sum()}")
    w("")
    for p in ("early", "late"):
        bad, tot = causal_check(best, p)
        w(f"== 미래 참조 확인 {p}: [:i] 잘린 배열 규칙 vs 전체 배열 규칙 불일치 {bad}/{tot}")
    w(f"{time.time() - t0:.0f}s")
    txt = "\n".join(lines)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)
