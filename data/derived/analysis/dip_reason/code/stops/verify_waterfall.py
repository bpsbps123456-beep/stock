"""반박 검증 — waterfall 계열 최선안 'G1 #22 종가 -5.0% 30분 안'을 정의 문장만 보고 다시 만든다(2026-10-07).

정의: 매수(봉 ei, 매수가 ent) 뒤 봉 i 시가에 팔지 정할 때, (i - ei) x 봉길이(분) <= 30 이고 c[i-1] <= ent x 0.95 이면 판다.
모든 세션 적용, 30분 지나면 손절 없음. 매도 순서·재매수 금지는 stop_lab.sim 그대로.
이웃: X 4/4.5/5.5/6%, T 15/25/35/45분(early 5분봉 기준 한 봉 단위 포함). 참고: 봉 수 대신 시계 시간 30분 창.

사용: py stops/verify_waterfall.py   (code 폴더에서)   출력: ../results/stops/verify_waterfall.txt"""
import os, sys, time
import numpy as np, pandas as pd

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
from stop_lab import load, compare, stop_audit, run, base_trades, UNIV

OUT = os.path.normpath(os.path.join(CODE, "..", "results", "stops", "verify_waterfall.txt"))
U78 = UNIV["사용자78"]


def early_kill(X=0.05, T=30, clock=False):
    """매수 뒤 T분 안, 직전 봉 종가가 매수가보다 X 넘게 낮으면 다음 봉 시가 매도."""
    def make_rule(A):
        c, tk, mins = A["c"], A["tk"], int(A["mins"])
        lim_ns = np.int64(T * 60 * 10**9)

        def rule(i, ei, ent):
            if clock:
                if tk[i] - tk[ei] > lim_ns:          # 봉 i 시각(시가 시점에 앎) - 매수 봉 시각
                    return False
            elif (i - ei) * mins > T:
                return False
            return c[i - 1] <= ent * (1.0 - X)       # i-1 봉 종가까지만
        return rule
    return make_rule


def say(f, s=""):
    print(s, flush=True)
    f.write(s + "\n")


def cells(df):
    g = lambda p, sc, un="사용자78": df[(df.기간 == p) & (df.범위 == sc) & (df.종목 == un)].iloc[0]
    return g


def day_dep(make, period, reg_only=False):
    S = load(period)
    t = run(S, make, period, reg_only, syms=U78)
    b = base_trades(period, reg_only)
    b = b[b.sym.isin(U78)]
    days = sorted(set(t.day) | set(b.day))
    d = t.groupby("day").ret.sum().reindex(days, fill_value=0) - b.groupby("day").ret.sum().reindex(days, fill_value=0)
    d = d[d.abs() > 1e-12].sort_values(ascending=False)
    return d, t


def main():
    t0 = time.time()
    with open(OUT, "w", encoding="utf-8") as f:
        say(f, "반박 검증: waterfall 최선안 '종가 -5.0% 30분 안' (독립 구현 verify_waterfall.py, 2026-10-07)")
        say(f, "")
        best = early_kill(0.05, 30)
        df = compare(best, "재현 X5% T30", show=False)
        g = cells(df)
        say(f, "== 1. 재현(compare) ==")
        say(f, df[["기간", "범위", "종목", "건수", "합", "차이", "t", "손절", "최악", "차이_10월6일"]].to_string(
            index=False, float_format=lambda v: f"{v:+.3f}"))
        rep = {("early", "24시간"): 0.146, ("early", "정규장만"): 0.041, ("late", "24시간"): -0.002, ("late", "정규장만"): 0.040}
        for (p, sc), v in rep.items():
            say(f, f"  사용자78 {p} {sc}: 재현 {g(p, sc).차이:+.4f} / 보고 {v:+.3f} -> 차 {g(p, sc).차이 - v:+.4f}")
        say(f, "")
        say(f, "== 2. stop_audit(사용자78) ==")
        for p in ("early", "late"):
            for reg in (False, True):
                a = stop_audit(best, p, reg)
                say(f, f"  {p} {'정규장만' if reg else '24시간'}: " + ", ".join(
                    f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in a.items()))
        say(f, "")
        say(f, "== 3. 하루 의존(사용자78, 날짜별 합 차이) ==")
        for p in ("early", "late"):
            for reg in (False, True):
                d, t = day_dep(best, p, reg)
                tot = d.sum()
                top = d.index[0] if len(d) else None
                say(f, f"  {p} {'정규장만' if reg else '24시간'}: 차이 {tot:+.4f}, 바뀐 날 {len(d)}, "
                       f"최대 이득일 {top} {d.iloc[0] if len(d) else 0:+.4f}, 그날 빼면 {tot - (d.iloc[0] if len(d) else 0):+.4f}; "
                       f"상위3 {', '.join(f'{k}:{v:+.3f}' for k, v in d.head(3).items())}; "
                       f"하위3 {', '.join(f'{k}:{v:+.3f}' for k, v in d.tail(3).items())}")
        # late 손절 내역
        d, t = day_dep(best, "late", False)
        b = base_trades("late", False)
        st = t[t.why == "손절"].merge(b[["sym", "ei", "ret", "why", "px"]], on=["sym", "ei"], how="left", suffixes=("", "_기준"))
        say(f, "  late 24시간 손절 내역(사용자78):")
        say(f, st[["sym", "t", "ses", "ent", "px", "ret", "ret_기준", "why_기준"]].to_string(
            index=False, float_format=lambda v: f"{v:+.4f}"))
        say(f, "")
        say(f, "== 4. 이웃 매개변수(사용자78 차이; late는 확인용) ==")
        neigh = [(0.04, 30, False), (0.045, 30, False), (0.055, 30, False), (0.06, 30, False),
                 (0.05, 15, False), (0.05, 25, False), (0.05, 35, False), (0.05, 45, False), (0.05, 30, True)]
        rows = []
        for X, T, ck in neigh:
            nm = f"X{X:.1%} T{T}" + (" 시계" if ck else "")
            dd = compare(early_kill(X, T, ck), nm, show=False)
            h = cells(dd)
            rows.append(dict(규칙=nm, e24=h("early", "24시간").차이, e정규=h("early", "정규장만").차이,
                             l24=h("late", "24시간").차이, l정규=h("late", "정규장만").차이,
                             e24손절=h("early", "24시간").손절, e24_137=h("early", "24시간", "전체137").차이,
                             l24_137=h("late", "24시간", "전체137").차이))
            print(f"   {nm} 끝 ({time.time() - t0:.0f}s)", flush=True)
        R = pd.DataFrame(rows)
        say(f, R.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
        say(f, f"  early24 플러스 {int((R.e24 > 0).sum())}/{len(R)}, 네 칸 플러스 {int(((R.e24 > 0) & (R.e정규 > 0) & (R.l24 > 0) & (R.l정규 > 0)).sum())}/{len(R)}")
        say(f, f"\n시험한 규칙 수: 최선안 1 + 이웃 {len(neigh)} = {1 + len(neigh)}  ({time.time() - t0:.0f}s)")
    print("저장:", OUT)


if __name__ == "__main__":
    main()
