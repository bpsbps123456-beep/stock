"""칼손절 연구 — 흘러내림 감지(시간 손절) 계열(2026-10-07).
매수 뒤 T분이 지나도 반등이 없으면 판다 등, '주루룩 흘러내리는' 매수를 일찍 자르는 규칙을 stop_lab 시험대로 비교한다.

규칙 공통: 봉 i 시가에 팔지 정하고 봉 i-1까지의 값만 쓴다. 경과 = (i - 매수 봉) x 봉 길이(분, early 5 / late 3) = 거래 시간 분.
T분 검사는 경과가 처음 T분 이상이 되는 봉 i(k = ceil(T / 봉 길이) 봉 뒤) 한 번만 한다(F4는 그 뒤에도 매 봉).
그 봉이 이미 원래 정리 시각(아침 8시 한국) 이후면 손절로 세지 않고 8시 정리에 맡긴다.

계열(시험 규칙 71개):
  F1 반등 없음   T분까지 매수 봉~i-1 봉 종가 최고 < 매수가 x (1+Y)                      T 15~240, Y 0/0.5/1%
  F2 T분 종가    T분 시점 i-1 봉 종가 < 매수가 x (1+Y)                                  T 30~240, Y 0/-1/-2%
  F3 고점 갱신 없음 매수 다음 봉~i-1 봉 종가 최고 <= 매수 봉 종가                         T 30/60/120
  F4 아래 머문 비율 T분 뒤 매 봉: 매수 봉~i-1 봉 종가 중 매수가 아래 비율 >= F            T 60/120/240, F 0.7/0.8/0.9
  F5 비정규장 매수 개장 뒤 M분 정규장 개장(9:30 ET) + M분 첫 봉에 i-1 종가 < 매수가(close) 또는 매수 뒤 종가 최고 < 매수가 x1.005(max)
  F6 세션별 T    F1(Y=F1 early 최선 Y)을 프리·정규·애프터 매수 하나에만 적용(12개),
                 조합A = 세션별 early 최선 T 묶음, 조합B = F5 early 최선 + 정규장 매수 F1(정규 최선 T)
  F7 반등 없음+깊이 F1 조건 그리고 i-1 종가 <= 매수가 x (1-D)                            T 60/120, D 1/2%

고르기(early만): 사용자78 24시간 차이가 가장 큰 규칙, 단 사용자78 정규장만 차이 >= 0인 규칙 안에서. late는 확인용.
사용: py stops/grind.py  (code 폴더에서)   출력: ../results/stops/grind.txt
     py stops/grind.py --why  대표 시간 손절의 손절 거래가 기준이면 어떻게 끝났는지 grind.txt 뒤에 붙임"""
import os, sys, time
import numpy as np, pandas as pd

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
from stop_lab import load, compare, stop_audit, run, base_trades, UNIV

OUT = os.path.normpath(os.path.join(CODE, "..", "results", "stops", "grind.txt"))


class Tee:
    def __init__(self, path):
        self.f, self.o = open(path, "w", encoding="utf-8"), sys.stdout

    def write(self, s):
        self.o.write(s); self.f.write(s); self.f.flush()

    def flush(self):
        self.o.flush(); self.f.flush()


def _k(T, mins):
    return max(1, int(np.ceil(T / mins)))


def f1(T, Y, sess=None, D=None):
    """반등 없음(+선택: 세션 한정, 깊이 D)."""
    def make_rule(A):
        c, tk, dead, ses = A["c"], A["tk"], A["dead"], A["ses"]
        kT = _k(T, A["mins"])

        def rule(i, ei, ent):
            if i - ei != kT or tk[i] >= dead[ei]:
                return False
            if sess is not None and ses[ei] not in sess:
                return False
            if D is not None and c[i - 1] > ent * (1 - D):
                return False
            return c[ei:i].max() < ent * (1 + Y)
        return rule
    return make_rule


def f1_combo(plan):
    """세션별로 다른 (T, Y). plan = {'프리': (T, Y), ...}; 없는 세션은 손절 안 함."""
    def make_rule(A):
        c, tk, dead, ses = A["c"], A["tk"], A["dead"], A["ses"]
        kT = {s: _k(T, A["mins"]) for s, (T, Y) in plan.items()}

        def rule(i, ei, ent):
            s = ses[ei]
            if s not in plan or i - ei != kT[s] or tk[i] >= dead[ei]:
                return False
            return c[ei:i].max() < ent * (1 + plan[s][1])
        return rule
    return make_rule


def f2(T, Y):
    def make_rule(A):
        c, tk, dead = A["c"], A["tk"], A["dead"]
        kT = _k(T, A["mins"])

        def rule(i, ei, ent):
            return i - ei == kT and tk[i] < dead[ei] and c[i - 1] < ent * (1 + Y)
        return rule
    return make_rule


def f3(T):
    def make_rule(A):
        c, tk, dead = A["c"], A["tk"], A["dead"]
        kT = max(2, _k(T, A["mins"]))

        def rule(i, ei, ent):
            return i - ei == kT and tk[i] < dead[ei] and c[ei + 1:i].max() <= c[ei]
        return rule
    return make_rule


def f4(T, F):
    def make_rule(A):
        c, tk, dead = A["c"], A["tk"], A["dead"]
        kT = _k(T, A["mins"])
        st = [-1, -1, 0]                                  # 매수 봉, 마지막 i, 매수가 아래 종가 수(매수 봉~i-1)

        def rule(i, ei, ent):
            if st[0] != ei or st[1] != i - 1:
                st[0], st[2] = ei, int((c[ei:i] < ent).sum())
            else:
                st[2] += int(c[i - 1] < ent)
            st[1] = i
            k = i - ei
            return k >= kT and tk[i] < dead[ei] and st[2] / k >= F
        return rule
    return make_rule


def f5(M, mode):
    """비정규장(프리·애프터) 매수가 정규장 개장 + M분 첫 봉에 아직 아래면 판다."""
    def make_rule(A):
        c, tk, dead, ses, reg, mod = A["c"], A["tk"], A["dead"], A["ses"], A["reg"], A["mod"]
        cut = 570 + M

        def rule(i, ei, ent):
            if ses[ei] == "정규" or not reg[i] or mod[i] < cut or tk[i] >= dead[ei]:
                return False
            if reg[i - 1] and mod[i - 1] >= cut and i - 1 > ei:   # 이미 지난 검사 시각(한 번만 본다)
                return False
            if mode == "close":
                return c[i - 1] < ent
            return c[ei:i].max() < ent * 1.005
        return rule
    return make_rule


def orr(*mks):
    """규칙 여러 개 중 하나라도 참이면 판다(상태가 있는 규칙도 매 봉 모두 부른다)."""
    def make_rule(A):
        rs = [m(A) for m in mks]

        def rule(i, ei, ent):
            out = False
            for r in rs:
                out = r(i, ei, ent) or out
            return out
        return rule
    return make_rule


def cells(df):
    """사용자78·전체137 주요 칸만 뽑는다."""
    g = df.set_index(["기간", "범위", "종목"])
    return {(p, r, u): g.loc[(p, r, u)] for p, r, u in g.index}


def line(name, df):
    c = cells(df)
    u = lambda p, r: c[(p, r, "사용자78")]
    a = lambda p, r: c[(p, r, "전체137")]
    e24, er, l24, lr = u("early", "24시간"), u("early", "정규장만"), u("late", "24시간"), u("late", "정규장만")
    return (f"{name:<34} | e24 {e24.차이:+6.2f} (t {e24.t:+5.2f}, 손절 {e24.손절:4d}) eR {er.차이:+6.2f} | "
            f"l24 {l24.차이:+6.2f} (t {l24.t:+5.2f}, 손절 {l24.손절:4d}) lR {lr.차이:+6.2f} | "
            f"137 e24 {a('early', '24시간').차이:+6.2f} l24 {a('late', '24시간').차이:+6.2f}")


def summarize(name, df):
    c = cells(df)
    row = dict(규칙=name)
    for p in ("early", "late"):
        for r, rk in (("24시간", "24"), ("정규장만", "R")):
            for un, uk in (("사용자78", ""), ("전체137", "_137")):
                x = c[(p, r, un)]
                row[f"{p[0]}{rk}{uk}"] = x.차이
                row[f"{p[0]}{rk}{uk}_t"] = x.t
                row[f"{p[0]}{rk}{uk}_손절"] = x.손절
    return row


def lookahead_check(make_rule, period, n_max=400):
    """손절된 거래마다 봉 i 이후(i 포함)의 고·저·종·거래량·RSI를 NaN으로 지워도 같은 결정이 나오는지."""
    S = load(period)
    t = run(S, make_rule, period, False, syms=UNIV["사용자78"])
    st = t[t.why == "손절"].head(n_max)
    bad = 0
    for x in st.itertuples():
        A = dict(S[x.sym])
        for k in ("h", "l", "c", "v", "r"):
            a = A[k].copy(); a[x.xi:] = np.nan; A[k] = a
        o = A["o"].copy(); o[x.xi + 1:] = np.nan; A["o"] = o
        if not make_rule(A)(x.xi, x.ei, x.ent):
            bad += 1
    return len(st), bad


def main():
    sys.stdout = Tee(OUT)
    t0 = time.time()
    print(__doc__)
    rows, tables = [], {}

    def test(name, mk):
        df = compare(mk, name, show=False)
        tables[name] = (mk, df)
        rows.append(summarize(name, df))
        print(line(name, df), f"[{time.time() - t0:5.0f}s]")

    print("\n=== F1 반등 없음: T분까지 종가 최고 < 매수가 x (1+Y)")
    for T in (15, 30, 60, 90, 120, 180, 240):
        for Y in (0.0, 0.005, 0.01):
            test(f"F1 T{T} Y{Y:+.1%}", f1(T, Y))
    print("\n=== F2 T분 시점 종가 < 매수가 x (1+Y)")
    for T in (30, 60, 120, 240):
        for Y in (0.0, -0.01, -0.02):
            test(f"F2 T{T} Y{Y:+.1%}", f2(T, Y))
    print("\n=== F3 매수 봉 종가 위 고점 갱신 없음")
    for T in (30, 60, 120):
        test(f"F3 T{T}", f3(T))
    print("\n=== F4 매수가 아래 머문 비율 >= F (T분 뒤 매 봉)")
    for T in (60, 120, 240):
        for F in (0.7, 0.8, 0.9):
            test(f"F4 T{T} F{F:.1f}", f4(T, F))
    print("\n=== F5 비정규장 매수, 정규장 개장 + M분에 아직 아래")
    for M in (0, 30, 60, 90):
        for mode in ("close", "max"):
            test(f"F5 M{M} {mode}", f5(M, mode))

    R = pd.DataFrame(rows).set_index("규칙")
    f1r = R[R.index.str.startswith("F1")]
    f1ok = f1r[f1r.eR >= 0]
    ystar = float((f1ok if len(f1ok) else f1r).e24.idxmax().split("Y")[1].rstrip("%")) / 100
    print(f"\n=== F6 세션 한정 F1 (Y = F1 early 최선의 Y {ystar:+.1%})")
    for s in ("프리", "정규", "애프터"):
        for T in (30, 60, 120, 240):
            test(f"F6 {s} T{T}", f1(T, ystar, sess=(s,)))
    R = pd.DataFrame(rows).set_index("규칙")
    plan = {}
    for s in ("프리", "정규", "애프터"):
        x = R[R.index.str.startswith(f"F6 {s} ")]
        x = x[x.eR >= 0] if s == "정규" else x
        if len(x) and x.e24.max() > 0:
            plan[s] = (int(x.e24.idxmax().split("T")[1]), ystar)
    print("  세션별 early 최선(early 24시간 차이 > 0인 세션만):", plan)
    # 조합 A: 세션별 최선 T 묶음(플러스 세션이 없으면 프리60·정규120으로 대신 시험)
    planA = plan if plan else {"프리": (60, ystar), "정규": (120, ystar)}
    test("F6 조합A " + ",".join(f"{s}{T}" for s, (T, _) in planA.items()), f1_combo(planA))
    # 조합 B: F5 early 최선(비정규장 매수) + 정규장 매수는 F6 정규 최선 T(없으면 120분)
    f5r = R[R.index.str.startswith("F5")]
    f5b = f5r.e24.idxmax()
    M5, mode5 = int(f5b.split()[1][1:]), f5b.split()[2]
    Treg = plan.get("정규", (120, ystar))[0]
    test(f"F6 조합B {f5b} + 정규{Treg}", orr(f5(M5, mode5), f1(Treg, ystar, sess=("정규",))))
    print("\n=== F7 반등 없음 + i-1 종가 <= 매수가 x (1-D)")
    for T in (60, 120):
        for D in (0.01, 0.02):
            test(f"F7 T{T} D{D:.0%}", f1(T, 0.005, D=D))

    R = pd.DataFrame(rows).set_index("규칙")
    n = len(R)
    four = R[(R.e24 > 0) & (R.eR > 0) & (R.l24 > 0) & (R.lR > 0)]
    print(f"\n시험 규칙 {n}개. 사용자78 네 칸(early/late x 24시간/정규장만) 모두 플러스: {len(four)}개 {list(four.index)}")
    print(f"early 사용자78 24시간 차이 > 0: {(R.e24 > 0).sum()}개, 그중 정규장만도 >= 0: {((R.e24 > 0) & (R.eR >= 0)).sum()}개")

    cand = R[R.eR >= 0] if (R.eR >= 0).any() else R
    order = cand.sort_values("e24", ascending=False)
    best = order.index[0]
    cols = ["e24", "e24_t", "e24_손절", "eR", "l24", "l24_t", "l24_손절", "lR", "e24_137", "l24_137", "eR_137", "lR_137"]
    print("\n=== early 기준 상위 12 (고르기: 사용자78 정규장만 차이 >= 0 안에서 24시간 차이 최대; l = late 확인용)")
    print(order[cols].head(12).to_string(float_format=lambda v: f"{v:+.2f}"))
    print("\n=== early 기준 하위 5")
    print(R.sort_values("e24")[cols].head(5).to_string(float_format=lambda v: f"{v:+.2f}"))

    mk, df = tables[best]
    print(f"\n=== 최선안(early로 고름): {best}  — 전체 표")
    f = df.copy()
    for k, fmt in (("평균", "{:+.2%}"), ("승률", "{:.0%}"), ("최악", "{:+.1%}"), ("하위5", "{:+.1%}"), ("합", "{:+.2f}"),
                   ("차이", "{:+.2f}"), ("t", "{:+.2f}"), ("손절평균", "{:+.2%}"), ("차이_10월6일", "{:+.2f}")):
        f[k] = f[k].map(lambda v: "" if pd.isna(v) else fmt.format(v))
    print(f.to_string(index=False))
    for p in ("early", "late"):
        for reg in (False, True):
            a = stop_audit(mk, p, reg_only=reg)
            print(f"stop_audit {p} {'정규장만' if reg else '24시간'} 사용자78: " +
                  ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in a.items()))
    for p in ("early", "late"):
        m, b = lookahead_check(mk, p)
        print(f"미래 참조 점검 {p}: 손절 {m}건을 봉 i 이후 값 지우고 다시 판정 → 결정 바뀐 건 {b}건")
    print(f"\n끝 {time.time() - t0:.0f}s")


def why_audit():
    """이미 시험한 대표 시간 손절의 손절 거래가 기준이면 어떻게 끝났는지(새 규칙 아님). grind.txt 뒤에 붙인다."""
    f = open(OUT, "a", encoding="utf-8")
    say = lambda s: (print(s), f.write(s + "\n"), f.flush())
    say("\n=== 왜 안 되나: 대표 시간 손절의 손절 거래 짝 비교(사용자78 24시간; 새 규칙 아님, 위에서 이미 센 규칙)")
    for name, mk in (("F1 T60 Y+0.0%", f1(60, 0.0)), ("F1 T120 Y+0.0%", f1(120, 0.0)), ("F2 T60 Y-2.0%", f2(60, -0.02)),
                     ("F7 T120 D2%", f1(120, 0.005, D=0.02))):
        for p in ("early", "late"):
            t = run(load(p), mk, p, False, syms=UNIV["사용자78"])
            b = base_trades(p, False)
            b = b[b.sym.isin(UNIV["사용자78"])].set_index(["sym", "ei"])
            s = t[t.why == "손절"].set_index(["sym", "ei"]).ret.to_frame("손절").join(b[["ret", "mae"]], how="inner")
            cls = np.where(s.ret > 0, "반등", np.where(s.ret <= -0.03, "큰 손실(-3% 이하)", "작은 손실"))
            g = s.assign(cls=cls).groupby("cls").agg(건수=("손절", "size"), 손절합=("손절", "sum"), 기준합=("ret", "sum"))
            g["아낀손실"] = g.손절합 - g.기준합
            say(f"{name} {p}: 손절 {len(s)}건\n" + g.to_string(float_format=lambda v: f"{v:+.2f}"))
    f.close()


if __name__ == "__main__":
    if "--why" in sys.argv:
        why_audit()
    else:
        main()
