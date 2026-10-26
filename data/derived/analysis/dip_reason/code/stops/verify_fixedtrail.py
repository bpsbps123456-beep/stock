"""반박 검증 — fixedtrail 최선안 '본전+2.0%(종가무장)|깨어' 독립 재구현(2026-10-07).

정의(연구자 보고 definition만 보고 다시 씀):
  봉 i(i >= ei+1)에서 i-1까지만 보고, 셋 다 참이면 봉 i 시가에 판다.
  (1) 무장  max(c[ei..i-1]) >= ent*(1+arm)      arm=0.02, 종가 기준
  (2) 발동  c[i-1] <= ent*(1+trig)              trig=0 (본전)
  (3) 시각  봉 i 시작 시각의 KST 시(hour) in [h0, h1)   h0=8, h1=24
이웃: arm 1.5%/2.5%, trig -0.5%/+0.5%, h0 7/9, h1 23, 고가 무장.
사용: py stops/verify_fixedtrail.py   (결과: ../results/stops/verify_fixedtrail.txt)
"""
import os, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from stop_lab import compare, stop_audit, load, base_trades, run, UNIV  # noqa: E402

OUT = os.path.normpath(os.path.join(HERE, "..", "..", "results", "stops", "verify_fixedtrail.txt"))
U78 = UNIV["사용자78"]


class BE:
    """피클 가능한 규칙 생성기. 무장 여부는 trade(ei)별로 누적 최대로만 갱신(i-1까지)."""

    def __init__(self, arm=0.02, trig=0.0, h0=8, h1=24, src="c"):
        self.arm, self.trig, self.h0, self.h1, self.src = arm, trig, h0, h1, src

    def __call__(self, A):
        c = A["c"]
        top_src = A["c"] if self.src == "c" else A["h"]
        # KST 시 = (UTC 초 + 9시간) // 3600 % 24  (tz 라이브러리 없이 계산)
        kh = ((A["tk"] // 1_000_000_000 + 9 * 3600) // 3600) % 24
        ok_hour = (kh >= self.h0) & (kh < self.h1)
        arm, trig = self.arm, self.trig
        mem = {"ei": -1, "upto": -1, "top": -np.inf}

        def rule(i, ei, ent):
            if mem["ei"] != ei:
                mem["ei"], mem["upto"], mem["top"] = ei, ei - 1, -np.inf
            # i-1 까지 누적 최대를 갱신(호출이 건너뛰어도 안전하게)
            if mem["upto"] < i - 1:
                mem["top"] = max(mem["top"], top_src[mem["upto"] + 1:i].max())
                mem["upto"] = i - 1
            if not ok_hour[i]:
                return False
            return mem["top"] >= ent * (1 + arm) and c[i - 1] <= ent * (1 + trig)
        return rule


def label(r):
    return f"본전+{r.arm:.1%}({'종가' if r.src == 'c' else '고가'}무장) 발동{r.trig:+.1%} KST{r.h0}~{r.h1}시"


def job(r):
    return label(r), compare(r, label(r), show=False)


def cellmap(df):
    o = {}
    for x in df.itertuples():
        o[(x.기간, x.범위, x.종목)] = x
    return o


def day_dep(r, period, reg_only=False):
    t = run(load(period), r, period, reg_only, syms=U78)
    b = base_trades(period, reg_only)
    b = b[b.sym.isin(U78)]
    days = sorted(set(t.day) | set(b.day))
    d = t.groupby("day").ret.sum().reindex(days, fill_value=0) - b.groupby("day").ret.sum().reindex(days, fill_value=0)
    d = d.sort_values(ascending=False)
    return d


def causal_check(r, period, n=300, seed=1):
    """A의 모든 배열을 [:i]로 자르고(시각 tk만 i까지 포함) 만든 규칙과 전체 배열 규칙 비교."""
    S = load(period)
    b = base_trades(period)
    rng = np.random.default_rng(seed)
    bad = tot = 0
    for x in b.sample(min(n, len(b)), random_state=seed).itertuples():
        if x.xi <= x.ei + 1:
            continue
        A = S[x.sym]
        i = int(rng.integers(x.ei + 1, x.xi + 1))
        full = r(A)
        for j in range(x.ei + 1, i + 1):
            want = full(j, x.ei, x.ent)
        T = {k: (A[k][:i] if isinstance(A[k], np.ndarray) else A[k]) for k in A}
        T["tk"] = A["tk"][:i + 1]
        try:
            got = r(T)(i, x.ei, x.ent)
        except IndexError:
            got = None
        tot += 1
        bad += (got != want)
    return bad, tot


if __name__ == "__main__":
    t0 = time.time()
    best = BE()
    nbrs = [BE(arm=0.015), BE(arm=0.025), BE(trig=-0.005), BE(trig=0.005),
            BE(h0=7), BE(h0=9), BE(h1=23), BE(src="h"), BE(h0=0, h1=24)]
    with Pool(6) as pool:
        res = pool.map(job, [best] + nbrs)
    lines = []
    w = lines.append
    w("반박 검증 — fixedtrail 최선안 독립 재구현 (verify_fixedtrail) 2026-10-07")
    w(f"시험한 규칙 수(이 검증): {len(res)} (최선안 1 + 이웃 {len(nbrs)})")
    w("")
    nm0, df0 = res[0]
    w(f"== 최선안 재현: {nm0}")
    show = df0[["기간", "범위", "종목", "건수", "합", "차이", "t", "손절", "최악", "차이_10월6일", "나은날", "나쁜날"]]
    w(show.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    w("")
    w("== 이웃 매개변수 (사용자78 차이 / t)")
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
    w(f"네 칸 모두 플러스(사용자78): {int(((nb.e24 > 0) & (nb.eReg > 0) & (nb.l24 > 0) & (nb.lReg > 0)).sum())}개")
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
            top = d.index[0]
            w(f"  {p} {'정규장만' if reg else '24시간'}: 합 {d.sum():+.3f}, 최고일 {top} {d.iloc[0]:+.3f}, "
              f"뺀 뒤 {d.sum() - d.iloc[0]:+.3f}, 상위2일 뺀 뒤 {d.sum() - d.iloc[:2].sum():+.3f}, "
              f"최악일 {d.index[-1]} {d.iloc[-1]:+.3f}, 나은날 {(d > 1e-12).sum()} 나쁜날 {(d < -1e-12).sum()}")
            w("     상위5일: " + ", ".join(f"{k}:{v:+.3f}" for k, v in d.head(5).items()))
    w("")
    for p in ("early", "late"):
        bad, tot = causal_check(best, p)
        w(f"== 미래 참조 확인 {p}: 잘린 배열 규칙 vs 전체 배열 규칙 불일치 {bad}/{tot}")
    w(f"{time.time() - t0:.0f}s")
    txt = "\n".join(lines)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)
