"""반박 검증 — grind 계열 최선안 'F5 M30 max'를 정의 문장만 보고 다시 만든다(2026-10-07).

정의: 매수 봉 ei 세션이 '정규'가 아닌 매수(프리·애프터)만. 매수 뒤 처음 오는 정규장 봉 중 ET 분(mod) >= 570+M인 첫 봉 i에서
한 번만 본다. c[ei..i-1] 최고값(종가, 매수 봉 포함) < 매수가 x (1+Y) 이면 봉 i 시가에 판다. 봉 i가 정리 시각(dead[ei]) 이후면 안 본다.
최선안 M=30, Y=0.5%. 이웃: M 0/15/45/60, Y 0/0.25%/0.75%/1.0%.

사용: py stops/verify_grind.py   (code 폴더에서)   출력: ../results/stops/verify_grind.txt"""
import os, sys, time
import numpy as np, pandas as pd

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
from stop_lab import load, compare, stop_audit, run, base_trades, UNIV

OUT = os.path.normpath(os.path.join(CODE, "..", "results", "stops", "verify_grind.txt"))
U78 = UNIV["사용자78"]


def no_bounce_at_open(M=30, Y=0.005):
    def make_rule(A):
        c, tk, dead = A["c"], A["tk"], A["dead"]
        n = len(c)
        ok = A["reg"] & (A["mod"] >= 570 + M)            # 봉 i의 시각 정보(시가 시점에 앎)
        pos = np.flatnonzero(ok)
        k = np.searchsorted(pos, np.arange(n + 1))        # j 이상에서 처음 ok인 봉
        first_ok = np.where(k < len(pos), pos[np.minimum(k, len(pos) - 1)], -1)
        chk = first_ok[1:]                                 # chk[ei] = ei 뒤(ei+1 이상) 첫 검사 봉
        offhours = A["ses"] != "정규"

        def rule(i, ei, ent):
            if not offhours[ei] or i != chk[ei] or tk[i] >= dead[ei]:
                return False
            return c[ei:i].max() < ent * (1.0 + Y)        # i-1 봉 종가까지
        return rule
    return make_rule


def say(f, s=""):
    print(s)
    f.write(s + "\n")
    f.flush()


def cell(df, p, r, u):
    return df[(df.기간 == p) & (df.범위 == r) & (df.종목 == u)].iloc[0]


def day_dependence(mk, p):
    t = run(load(p), mk, p, False, syms=U78)
    b = base_trades(p, False)
    b = b[b.sym.isin(U78)]
    days = sorted(set(b.day) | set(t.day))
    d = t.groupby("day").ret.sum().reindex(days, fill_value=0) - b.groupby("day").ret.sum().reindex(days, fill_value=0)
    d = d.sort_values(ascending=False)
    return d


def future_wipe_check(mk, p):
    """손절 거래마다 봉 i 이후 h l c v r 와 봉 i+1 이후 o를 엉뚱한 값으로 바꿔도 결정이 같은지."""
    S = load(p)
    t = run(S, mk, p, False, syms=U78)
    st = t[t.why == "손절"]
    changed = 0
    rng = np.random.default_rng(0)
    for x in st.itertuples():
        A = dict(S[x.sym])
        for key in ("h", "l", "c", "v", "r"):
            a = A[key].copy()
            a[x.xi:] = rng.uniform(0, 1e6, len(a) - x.xi)
            A[key] = a
        o = A["o"].copy()
        o[x.xi + 1:] = np.nan
        A["o"] = o
        if not mk(A)(x.xi, x.ei, x.ent):
            changed += 1
    return len(st), changed


def main():
    f = open(OUT, "w", encoding="utf-8")
    t0 = time.time()
    say(f, __doc__)
    mk = no_bounce_at_open(30, 0.005)

    say(f, "\n=== 1) 재현: compare (최선안 M30 Y0.5%)")
    df = compare(mk, "verify F5 M30 max", show=False)
    show = df[["기간", "범위", "종목", "건수", "합", "차이", "t", "손절", "최악", "차이_10월6일", "나은날", "나쁜날"]]
    say(f, show.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    say(f, "\n=== 2) stop_audit 사용자78")
    for p in ("early", "late"):
        for reg in (False, True):
            a = stop_audit(mk, p, reg_only=reg)
            say(f, f"{p} {'정규장만' if reg else '24시간'}: " +
                ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in a.items()))

    say(f, "\n=== 3) 미래 참조 교란 점검(봉 i 이후 값을 난수로 바꿈)")
    for p in ("early", "late"):
        n, ch = future_wipe_check(mk, p)
        say(f, f"{p}: 손절 {n}건, 결정 바뀐 건 {ch}건")

    say(f, "\n=== 4) 이웃 매개변수 (사용자78 / 전체137)")
    grid = [(0, 0.005), (15, 0.005), (45, 0.005), (60, 0.005), (30, 0.0), (30, 0.0025), (30, 0.0075), (30, 0.01)]
    for M, Y in grid:
        d = compare(no_bounce_at_open(M, Y), f"M{M} Y{Y:.2%}", show=False)
        e, l = cell(d, "early", "24시간", "사용자78"), cell(d, "late", "24시간", "사용자78")
        ea, la = cell(d, "early", "24시간", "전체137"), cell(d, "late", "24시간", "전체137")
        er, lr = cell(d, "early", "정규장만", "사용자78"), cell(d, "late", "정규장만", "사용자78")
        say(f, f"M{M:>2} Y{Y:+.2%}: e24 {e.차이:+.2f} (t {e.t:+.2f}, 손절 {e.손절}) eR {er.차이:+.2f} | "
               f"l24 {l.차이:+.2f} (t {l.t:+.2f}, 손절 {l.손절}) lR {lr.차이:+.2f} | 137 e24 {ea.차이:+.2f} l24 {la.차이:+.2f}"
               f"  [{time.time() - t0:.0f}s]")

    say(f, "\n=== 5) 하루 의존 (사용자78 24시간, 날짜별 합 차이)")
    for p in ("early", "late"):
        d = day_dependence(mk, p)
        nz = d[d.abs() > 1e-12]
        say(f, f"{p}: 합 {d.sum():+.3f}, 나은날 {(d > 1e-12).sum()} 나쁜날 {(d < -1e-12).sum()}, "
               f"최고 하루 {d.index[0]} {d.iloc[0]:+.3f} → 빼면 {d.sum() - d.iloc[0]:+.3f}, "
               f"상위 2일 빼면 {d.sum() - d.iloc[:2].sum():+.3f}, 최악 하루 {d.index[-1]} {d.iloc[-1]:+.3f}")
        say(f, "  날짜별(0 아닌 날): " + ", ".join(f"{k:%m/%d} {v:+.3f}" for k, v in nz.items()))
    say(f, f"\n끝 {time.time() - t0:.0f}s")
    f.close()


if __name__ == "__main__":
    main()
