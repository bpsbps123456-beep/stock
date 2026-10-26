"""일봉 지지 구역 '며칠 버팀' 연구(2026-10-09, 키 daily).

질문: 일봉 스윙 저점 구역에 닿은 뒤 종가로 깨지지 않고 버틴 일수 D가 길수록 위로 돌아서나(또는 오래 두드릴수록 결국 깨지나),
그리고 '버팀의 끝자락'(구역 위 4% 안 = 아직 안 오른 때)에 사면 처음 닿은 날 사는 것·무작위보다 나은가.

구역(사용자 트레이딩뷰 지표 sr_zones_ref.py의 묶음 규칙을 일봉에): 좌우 N봉(3·5) 스윙 저점, 최근 60개, 0.5% 안은 한 구역(lo~hi),
  구역 중심끼리 1.2% 미만이면 닿은 횟수 적은 쪽(같으면 오래된 쪽) 제외. 피벗 i는 i+N일 종가가 지나야 확정 → 날짜 t 판단에는 i+N <= t-1만.
지지 = 전날 종가 아래(중심 기준) 구역 중 가장 가까운 것.
에피소드: 전날 종가 > hi*(1+1%)이고 오늘 저가 <= hi*(1+1%)이면 '처음 닿음'(t0). 구역은 t0 값으로 고정.
  매일 종가 < lo*(1-1%) → 깨짐(끝), 종가 > hi*(1+4%) → 위로 떠남(끝), 그 밖 → 살아 있음, 버틴 일수 D += 1 (t0 종가가 버티면 D=1).
진입(전부 신호일 종가 뒤 다음 날 시가 체결):
  E1(d): 살아 있는 상태로 D == d가 된 날 → 다음 날 시가(= 끝자락: 아직 구역 위 4% 안).
  E2(d): D >= d로 살아 있던 다음 날, 종가가 직전 3일 고가를 넘고(돌파 폭 <= 3%), 구역 아래로 안 깨짐 → 다음 날 시가. 에피소드당 첫 번째만.
기준: A2 처음 닿은 날 다음 날 시가, A1 처음 닿은 순간(hi*(1+1%) 지정가, 체결 min(시가, 지정가)), 무작위(같은 종목·같은 달 모든 날 시가 진입 평균).
보유 h = 1·3·5·10일: 매수 봉 e의 시가(또는 지정가) → e+h 봉 시가.
기간: 고르기 = 매수일 < 2026-04-07이고 매도일도 그 전, 확인 = 매수일 >= 2026-04-07. 고르기로만 최선안을 정하고 확인 값으로 다시 고르지 않는다.
t: 날짜 군집(거래일 max(5,h)일 묶음) CR1 표준오차.
사용: py support/daily.py   (code 폴더에서)
"""
import os, sys
from datetime import date
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
sys.path.insert(0, CODE)
from fetch import universe, user_universe

RAW = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad/dip/raw"
OUT = os.path.normpath(os.path.join(CODE, "../results/support/daily.txt"))
KEEP, TOL, GAP = 60, 0.005, 0.012          # 사용자 지표 그대로
EPS, UP = 0.01, 0.04                       # 닿음·깨짐 허용 1%, 떠남 4%(미리 정함, 고르지 않음)
BK, CAP = 3, 0.03                          # E2: 직전 3일 고가 돌파, 폭 3% 이하
SPLIT = date(2026, 4, 7)
HS = (1, 3, 5, 10)
E1_D = (1, 2, 3, 4, 5, 6, 8, 10)
E2_D = (1, 2, 3, 5)
USER78 = set(user_universe()["합집합"])
ALL = sorted(USER78 | set(universe()))
UNIV = {"사용자78": USER78, "전체137": set(ALL)}
CAL = {d: i for i, d in enumerate(pd.read_parquet(f"{RAW}/1d_SPY.parquet").index.date)}

LINES = []


def out(s=""):
    print(s)
    LINES.append(str(s))


def pivots(h, l, N, src):
    """확정 순서(봉 순서)로 (봉 i, 가격). src='low'면 스윙 저점만, 'both'면 사용자 지표처럼 고점도."""
    n = len(l)
    if n < 2 * N + 1:
        return np.array([], int), np.array([])
    mid_l, mid_h = l[N:n - N], h[N:n - N]
    lowp = (mid_l == swv(l, 2 * N + 1).min(1)) & (swv(l, N)[:n - 2 * N] > mid_l[:, None]).all(1)
    bi, px = [], []
    if src == "both":
        highp = (mid_h == swv(h, 2 * N + 1).max(1)) & (swv(h, N)[:n - 2 * N] < mid_h[:, None]).all(1)
    for j in range(n - 2 * N):
        if lowp[j]:
            bi.append(j + N); px.append(mid_l[j])
        if src == "both" and highp[j]:
            bi.append(j + N); px.append(mid_h[j])
    return np.array(bi, int), np.array(px, float)


def zones(px, bi):
    px, bi = px[-KEEP:], bi[-KEEP:]
    Z = []
    for k in np.argsort(px, kind="stable"):
        p, q = px[k], bi[k]
        if Z and (p - Z[-1][0]) / Z[-1][0] <= TOL:
            Z[-1][1] = p; Z[-1][2] += 1; Z[-1][3] = max(Z[-1][3], q)
        else:
            Z.append([p, p, 1, q])
    keep = []
    for z in sorted(Z, key=lambda z: z[2] * 1e7 + z[3], reverse=True):
        mid = (z[0] + z[1]) / 2
        if all(abs(mid - (k[0] + k[1]) / 2) / ((k[0] + k[1]) / 2) >= GAP for k in keep):
            keep.append(z)
    return [(z[0], z[1], z[2]) for z in keep]


def scan(sym, N, src):
    d = pd.read_parquet(f"{RAW}/1d_{sym}.parquet")
    o, h, l, c = (d[k].values.astype(float) for k in ("Open", "High", "Low", "Close"))
    dates = np.asarray(d.index.date)
    n = len(c)
    bi, px = pivots(h, l, N, src)
    conf = bi + N
    cache, eps, touch, alive, brk = {}, [], [], [], []
    ep = None
    for t in range(1, n):
        if ep is None:
            k = int(np.searchsorted(conf, t - 1, side="right"))       # t-1 종가까지 확정된 피벗 수
            if k == 0:
                continue
            if k not in cache:
                cache[k] = zones(px[:k], bi[:k])
            Z = [z for z in cache[k] if (z[0] + z[1]) / 2 < c[t - 1]]
            if not Z:
                continue
            lo, hi, cnt = max(Z, key=lambda z: z[0] + z[1])
            top = hi * (1 + EPS)
            if not (c[t - 1] > top and l[t] <= top):
                continue
            ep = dict(id=len(eps), lo=lo, hi=hi, D=0, T=0)
            eps.append([t, n, "열림", cnt])
            touch.append((t, min(o[t], top), ep["id"], cnt))
        if l[t] <= ep["hi"] * (1 + EPS):
            ep["T"] += 1
        if ep["D"] >= 1 and t >= BK:
            ph = h[t - BK:t].max()
            if ph < c[t] <= ph * (1 + CAP) and c[t] >= ep["lo"] * (1 - EPS):
                brk.append((t, ep["D"], ep["id"]))
        if c[t] < ep["lo"] * (1 - EPS):
            eps[ep["id"]][1:3] = [t, "깨짐"]; ep = None; continue
        if c[t] > ep["hi"] * (1 + UP):
            eps[ep["id"]][1:3] = [t, "위로"]; ep = None; continue
        ep["D"] += 1
        alive.append((t, ep["D"], ep["T"], ep["id"]))
    return dict(o=o, dates=dates, n=n, eps=eps, touch=touch, alive=alive, brk=brk)


def per_bar_random(sym):
    """같은 종목·같은 달 무작위 진입(시가 → h일 뒤 시가) 평균과 전체 행."""
    d = pd.read_parquet(f"{RAW}/1d_{sym}.parquet")
    o = d.Open.values.astype(float)
    dates = np.asarray(d.index.date)
    rows = []
    for h in HS:
        e = np.arange(1, len(o) - h)
        r = o[e + h] / o[e] - 1
        rows.append(pd.DataFrame(dict(sym=sym, h=h, e=e, de=dates[e], dx=dates[e + h], ret=r)))
    R = pd.concat(rows)
    R["ym"] = [x.year * 100 + x.month for x in R.de]
    return R


def trades(S, sym, N, src, kinds):
    """kinds: [(이름, d)] → 거래 행."""
    o, dates, n = S["o"], S["dates"], S["n"]
    eps = S["eps"]
    ents = []                                    # (rule, e, 체결가, ep_id, 신호일 t)
    al = np.array(S["alive"], int).reshape(-1, 4)
    for kind, dd in kinds:
        if kind == "E1":
            for t, D, T, ei in al[al[:, 1] == dd]:
                ents.append((f"E1 D={dd}", t + 1, None, ei, t))
        elif kind == "E1all":
            for t, D, T, ei in al:
                ents.append((f"D{min(D, 11)}", t + 1, None, ei, t))
        elif kind == "E2":
            seen = set()
            for t, Dp, ei in S["brk"]:
                if Dp >= dd and ei not in seen:
                    seen.add(ei); ents.append((f"E2 D>={dd}", t + 1, None, ei, t))
        elif kind == "A2":
            for t, fill, ei, cnt in S["touch"]:
                ents.append(("A2 첫닿음 다음날", t + 1, None, ei, t))
        elif kind == "A1":
            for t, fill, ei, cnt in S["touch"]:
                ents.append(("A1 첫닿음 지정가", t, fill, ei, t))
        elif kind == "Tall":                     # 두드림 횟수(닿은 날 수) 설명용
            for t, D, T, ei in al:
                ents.append((f"T{min(T, 5)}", t + 1, None, ei, t))
    rows = []
    for rule, e, fill, ei, t in ents:
        if e >= n:
            continue
        px = o[e] if fill is None else fill
        t_end, kind_end = eps[ei][1], eps[ei][2]
        k = t_end - t                                  # 신호일 뒤 며칠 안에 끝나나
        for h in HS:
            if e + h >= n:
                continue
            rows.append((rule, sym, N, src, e, dates[e], dates[e + h], CAL.get(dates[e], -1), h, o[e + h] / px - 1,
                         kind_end if 0 <= k <= 5 and kind_end != "열림" else "버팀", kind_end))
    return rows


COLS = ["rule", "sym", "N", "src", "e", "de", "dx", "cal", "h", "ret", "end5", "end"]


def period(df):
    p1 = (df.de < SPLIT) & (df.dx < SPLIT)
    p2 = df.de >= SPLIT
    return np.where(p1, "고르기", np.where(p2, "확인", ""))


def ct(x, g):
    """평균의 날짜 군집 t(CR1)."""
    x = np.asarray(x, float); n = len(x)
    if n < 3:
        return np.nan
    s = pd.Series(x - x.mean()).groupby(np.asarray(g)).sum().values
    G = len(s)
    if G < 2:
        return np.nan
    se = np.sqrt(G / (G - 1) * (s ** 2).sum()) / n
    return x.mean() / se if se > 0 else np.nan


def stats(g, h):
    cl = g.cal // max(5, h)
    ex = g.ret - g.rnd
    return pd.Series(dict(n=len(g), 평균=g.ret.mean(), 승률=(g.ret > 0).mean(), 최악=g.ret.min(), 합=g.ret.sum(),
                          무작위=g.rnd.mean(), 초과=ex.mean(), t초과=ct(ex, cl), t=ct(g.ret, cl),
                          깨짐5=(g.end5 == "깨짐").mean(), 위로5=(g.end5 == "위로").mean()))


def fmt(df, cols=None):
    f = df.copy()
    for k, fm in (("평균", "{:+.2%}"), ("승률", "{:.0%}"), ("최악", "{:+.1%}"), ("합", "{:+.2f}"), ("무작위", "{:+.2%}"),
                  ("초과", "{:+.2%}"), ("t초과", "{:+.2f}"), ("t", "{:+.2f}"), ("깨짐5", "{:.0%}"), ("위로5", "{:.0%}"),
                  ("A2평균", "{:+.2%}"), ("A2대비", "{:+.2%}")):
        if k in f:
            f[k] = f[k].map(lambda v: "" if pd.isna(v) else fm.format(v))
    if "n" in f:
        f["n"] = f["n"].map(lambda v: "" if pd.isna(v) else f"{int(v)}")
    return f.to_string(index=False) if cols is None else f[cols].to_string(index=False)


def main():
    R = pd.concat([per_bar_random(s) for s in ALL])
    R["P"] = period(R)
    rnd_mean = R.groupby(["sym", "ym", "h"]).ret.mean()
    kinds = [("E1", d) for d in E1_D] + [("E2", d) for d in E2_D] + [("A2", 0), ("A1", 0), ("E1all", 0), ("Tall", 0)]
    rows, epsum = [], []
    for N in (3, 5):
        for s in ALL:
            S = scan(s, N, "low")
            rows += trades(S, s, N, "low", kinds)
            for t0, te, kind, cnt in S["eps"]:
                epsum.append((N, s, S["dates"][t0], kind, te - t0))
    T = pd.DataFrame(rows, columns=COLS)
    T["ym"] = [x.year * 100 + x.month for x in T.de]
    T["rnd"] = rnd_mean.reindex(pd.MultiIndex.from_arrays([T.sym, T.ym, T.h])).values
    T["P"] = period(T)
    T = T[T.P != ""]
    E = pd.DataFrame(epsum, columns=["N", "sym", "d0", "end", "len"])
    E["P"] = np.where(np.asarray([x < SPLIT for x in E.d0]), "고르기", "확인")

    out("일봉 지지 구역 '며칠 버팀' 연구 (2026-10-09, support/daily.py)")
    out(f"데이터: yfinance 일봉 2025-10-07~2026-10-06 정규장, {len(ALL)}종목(사용자78 ⊂ 전체137). 고르기 = 매수·매도 모두 2026-04-07 전, 확인 = 매수 2026-04-07 이후.")
    out(f"구역 = 일봉 스윙 저점(좌우 N=3·5) 최근 {KEEP}개, {TOL:.1%} 안 한 구역, 중심 {GAP:.1%} 미만이면 약한 쪽 제외(사용자 지표 규칙). 피벗은 i+N일 종가 뒤부터 사용.")
    out(f"닿음: 전날 종가 > hi×1.01, 오늘 저가 <= hi×1.01. 깨짐: 종가 < lo×0.99. 위로 떠남: 종가 > hi×1.04. 버틴 일수 D = 처음 닿은 날부터 살아 있는 날 수.")
    out("수익 = 다음 날 시가 매수 → h거래일 뒤 시가 매도(수수료 0). 초과 = 수익 - 같은 종목·같은 달 무작위 시가 진입 평균. t = 날짜 군집(max(5,h)거래일 묶음).")
    out("깨짐5/위로5 = 신호일 뒤 5거래일 안에 그 에피소드가 깨짐/위로 떠남으로 끝난 비율.")
    out()

    # 1) 에피소드 결과
    out("■ 1. 에피소드(처음 닿음) 수와 끝난 모양")
    g = E.groupby(["N", "P", "end"]).size().unstack(fill_value=0)
    out(g.to_string())
    out()

    # 2) 단조성: 버틴 일수 D별
    out("■ 2. 단조성 — 버틴 일수 D(그날 종가까지 살아 있음, 아직 구역 위 4% 안)별, 다음 날 시가 매수. D11 = 11일 이상")
    Dall = T[T.rule.str.fullmatch(r"D\d+")].copy()
    Dall["D"] = Dall.rule.str[1:].astype(int)
    for N in (3, 5):
        for un, us in UNIV.items():
            for P in ("고르기", "확인"):
                x = Dall[(Dall.N == N) & (Dall.P == P) & Dall.sym.isin(us)]
                tab = []
                for D, g in x.groupby("D"):
                    r = dict(D=D)
                    for h in HS:
                        gh = g[g.h == h]
                        st = stats(gh, h)
                        if h == 5:
                            r.update(n=st.n, 승률5=st.승률, 깨짐5=st.깨짐5, 위로5=st.위로5)
                        r[f"평균{h}"] = st.평균; r[f"초과{h}"] = st.초과
                        if h in (5, 10):
                            r[f"t초과{h}"] = st.t초과
                    tab.append(r)
                tb = pd.DataFrame(tab)
                for k in tb:
                    if k.startswith(("평균", "초과")):
                        tb[k] = tb[k].map(lambda v: f"{v:+.2%}")
                    elif k.startswith(("승률", "깨짐", "위로")):
                        tb[k] = tb[k].map(lambda v: f"{v:.0%}")
                    elif k.startswith("t초과"):
                        tb[k] = tb[k].map(lambda v: f"{v:+.2f}")
                    elif k == "n":
                        tb[k] = tb[k].map(lambda v: f"{int(v)}")
                out(f"[N={N} {un} {P}]")
                out(tb[["D", "n", "평균1", "초과1", "평균3", "초과3", "평균5", "초과5", "t초과5", "평균10", "초과10", "t초과10",
                        "승률5", "깨짐5", "위로5"]].to_string(index=False))
                # 순위 상관(D와 초과5)
                x5 = x[x.h == 5]
                ex5 = (x5.ret - x5.rnd).groupby(x5.D).mean()
                br5 = (x5.end5 == "깨짐").groupby(x5.D).mean()
                Dk = pd.Series(ex5.index, dtype=float)
                out(f"  D(1~11)↔초과5 스피어만 {Dk.corr(pd.Series(ex5.values), method='spearman'):+.2f}, "
                    f"D↔깨짐5 스피어만 {Dk.corr(pd.Series(br5.values), method='spearman'):+.2f}")
                out()

    # 2b) 구간으로 묶은 단조성(표본 작은 꼬리 정리)
    out("■ 2b. 단조성 요약 — D를 구간으로 묶음(1 / 2 / 3~4 / 5~7 / 8~10 / 11+), 사용자78·전체137, 5일·10일 보유")
    bins = [(1, 1, "1"), (2, 2, "2"), (3, 4, "3~4"), (5, 7, "5~7"), (8, 10, "8~10"), (11, 11, "11+")]
    for N in (3, 5):
        for un, us in UNIV.items():
            tab = []
            for a, b, lab in bins:
                r = dict(D=lab)
                for P in ("고르기", "확인"):
                    for h in (5, 10):
                        g = Dall[(Dall.N == N) & (Dall.P == P) & Dall.sym.isin(us) & (Dall.h == h) & Dall.D.between(a, b)]
                        st = stats(g, h)
                        r[f"{P[:2]}n{h}"] = f"{int(st.n)}"
                        r[f"{P[:2]}평균{h}"] = f"{st.평균:+.2%}"
                        r[f"{P[:2]}초과{h}"] = f"{st.초과:+.2%}"
                        if h == 5:
                            r[f"{P[:2]}깨짐5"] = f"{st.깨짐5:.0%}"
                tab.append(r)
            out(f"[N={N} {un}]")
            out(pd.DataFrame(tab).to_string(index=False))
            out()

    # 3) 두드림 횟수(닿은 날 수) 설명용
    out("■ 3. 두드림 횟수 T(에피소드 안에서 저가가 구역에 닿은 날 수, T5 = 5번 이상)별 — 오래 두드리면 깨지나")
    Tt = T[T.rule.str.fullmatch(r"T\d")]
    for N in (3, 5):
        for P in ("고르기", "확인"):
            x = Tt[(Tt.N == N) & (Tt.P == P) & Tt.sym.isin(USER78) & (Tt.h == 5)]
            tb = x.groupby("rule").apply(lambda g: stats(g, 5)).reset_index()
            out(f"[N={N} 사용자78 {P}, 5일 보유]")
            out(fmt(tb, ["rule", "n", "평균", "초과", "t초과", "승률", "깨짐5", "위로5"]))
            out()

    # 4) 규칙 격자(96개) — 고르기·확인, 기준 A2·무작위 대비
    out("■ 4. 시험 규칙 격자: N(3,5) × 진입 12개(E1 D=1,2,3,4,5,6,8,10 / E2 D>=1,2,3,5) × 보유 1·3·5·10일 = 96개")
    rules = [f"E1 D={d}" for d in E1_D] + [f"E2 D>={d}" for d in E2_D]
    base = ["A2 첫닿음 다음날", "A1 첫닿음 지정가"]
    G = T[T.rule.isin(rules + base)]
    res = []
    for (rule, N, h, P), g in G.groupby(["rule", "N", "h", "P"]):
        for un, us in UNIV.items():
            gg = g[g.sym.isin(us)]
            if len(gg):
                res.append(dict(rule=rule, N=N, h=h, P=P, 종목=un, **stats(gg, h)))
    M = pd.DataFrame(res)
    a2 = M[M.rule == "A2 첫닿음 다음날"].set_index(["N", "h", "P", "종목"]).평균.rename("A2평균")
    M = M.join(a2, on=["N", "h", "P", "종목"])
    M["A2대비"] = M.평균 - M.A2평균
    M["이김"] = (M.A2대비 > 0) & (M.초과 > 0)
    R96 = M[M.rule.isin(rules)]
    assert R96.groupby(["rule", "N", "h"]).ngroups == 96, R96.groupby(["rule", "N", "h"]).ngroups
    W = R96.pivot_table(index=["rule", "N", "h"], columns=["종목", "P"], values="이김", aggfunc="first").fillna(False)
    both_u = int((W[("사용자78", "고르기")] & W[("사용자78", "확인")]).sum())
    both_a = int((W[("전체137", "고르기")] & W[("전체137", "확인")]).sum())
    both_ua = int((W[("사용자78", "고르기")] & W[("사용자78", "확인")] & W[("전체137", "고르기")] & W[("전체137", "확인")]).sum())
    out(f"두 기간 모두 기준(A2 평균보다 높고 무작위 초과 > 0)을 이긴 규칙: 사용자78 {both_u}/96, 전체137 {both_a}/96, 둘 다 {both_ua}/96")
    pick_u = int(W[("사용자78", "고르기")].sum()); conf_u = int(W[("사용자78", "확인")].sum())
    out(f"  (사용자78 고르기 기간만 이긴 규칙 {pick_u}개, 확인 기간만 {conf_u}개)")
    wl = W[W[("사용자78", "고르기")] & W[("사용자78", "확인")]].index
    for rule, N, h in wl:
        a = R96[(R96.rule == rule) & (R96.N == N) & (R96.h == h) & (R96.종목 == "사용자78")].set_index("P")
        out(f"  사용자78 두 기간 이김: N={N} {rule} {h}일 — " + ", ".join(
            f"{P} n{int(a.loc[P].n)} 평균 {a.loc[P].평균:+.2%} A2대비 {a.loc[P].A2대비:+.2%} 초과 {a.loc[P].초과:+.2%}(t {a.loc[P].t초과:+.2f})"
            for P in ("고르기", "확인")))
    out()
    for un in UNIV:
        for P in ("고르기", "확인"):
            x = M[(M.종목 == un) & (M.P == P)].sort_values(["N", "h", "rule"])
            out(f"[{un} {P}] 규칙별 (A2대비 = 같은 N·보유의 '처음 닿은 날 다음 날 시가' 평균과의 차이)")
            out(fmt(x, ["N", "h", "rule", "n", "평균", "A2대비", "초과", "t초과", "승률", "최악", "합"]))
            out()

    # 5) 최선안: 고르기 기간 사용자78에서 n>=30, A2보다 평균 높음, t초과 최대
    cand = R96[(R96.종목 == "사용자78") & (R96.P == "고르기") & (R96.n >= 30) & (R96.A2대비 > 0)]
    if not len(cand):
        cand = R96[(R96.종목 == "사용자78") & (R96.P == "고르기") & (R96.n >= 30)]
    best = cand.sort_values("t초과", ascending=False).iloc[0]
    out(f"■ 5. 최선안(고르기 기간·사용자78에서 n>=30, A2보다 평균 높음 중 t초과 최대): {best.rule}, N={best.N}, 보유 {best.h}일")
    bx = M[(M.N == best.N) & (M.h == best.h) & M.rule.isin([best.rule] + base)].sort_values(["종목", "P", "rule"])
    out(fmt(bx, ["종목", "P", "rule", "n", "평균", "승률", "최악", "합", "무작위", "초과", "t초과", "t"]))
    out()
    out("같은 진입의 보유 기간별(참고, 고르기는 위 보유로만 정함)")
    bh = M[(M.N == best.N) & (M.rule == best.rule)].sort_values(["종목", "P", "h"])
    out(fmt(bh, ["종목", "P", "h", "n", "평균", "A2평균", "A2대비", "초과", "t초과", "승률", "최악"]))
    out()

    # 6) 민감도: 사용자 지표 그대로(스윙 고점 포함) 구역으로 같은 규칙
    out("■ 6. 민감도(규칙 1개 추가 = 97번째): 사용자 지표처럼 스윙 고점도 구역에 넣었을 때 같은 최선안")
    kind = ("E1", int(best.rule.split("=")[1])) if best.rule.startswith("E1") else ("E2", int(best.rule.split(">=")[1]))
    rows2 = []
    for s in ALL:
        S = scan(s, int(best.N), "both")
        rows2 += trades(S, s, int(best.N), "both", [kind, ("A2", 0)])
    T2 = pd.DataFrame(rows2, columns=COLS)
    T2["ym"] = [x.year * 100 + x.month for x in T2.de]
    T2["rnd"] = rnd_mean.reindex(pd.MultiIndex.from_arrays([T2.sym, T2.ym, T2.h])).values
    T2["P"] = period(T2)
    T2 = T2[(T2.P != "") & (T2.h == best.h)]
    res2 = []
    for (rule, P), g in T2.groupby(["rule", "P"]):
        for un, us in UNIV.items():
            res2.append(dict(rule=rule, P=P, 종목=un, **stats(g[g.sym.isin(us)], int(best.h))))
    out(fmt(pd.DataFrame(res2).sort_values(["종목", "P", "rule"]), ["종목", "P", "rule", "n", "평균", "승률", "최악", "합", "초과", "t초과"]))
    out()

    # 7) 무작위 전체(무조건) 기준
    out("■ 7. 무작위(무조건) 기준: 모든 종목·모든 날 다음 날 시가 진입")
    rr = []
    for (h, P), g in R[R.P != ""].groupby(["h", "P"]):
        for un, us in UNIV.items():
            x = g[g.sym.isin(us)].ret
            rr.append(dict(h=h, P=P, 종목=un, n=len(x), 평균=x.mean(), 승률=(x > 0).mean(), 최악=x.min()))
    out(fmt(pd.DataFrame(rr).sort_values(["종목", "P", "h"])))
    out()

    # 8) 지금 이격매수 지표(분봉) 참고값
    out("■ 8. 참고: 지금 이격매수 지표(3분봉/5분봉, 하루 안팎 보유) 거래 — 보유 기간이 달라 직접 비교 안 함")
    try:
        from stop_lab import base_trades
        for p in ("early", "late"):
            for reg in (False, True):
                b = base_trades(p, reg)
                for un, us in UNIV.items():
                    x = b[b.sym.isin(us)].ret
                    out(f"  {p} {'정규장만' if reg else '24시간'} {un}: {len(x)}건 평균 {x.mean():+.2%} 승률 {(x > 0).mean():.0%} 최악 {x.min():+.1%} 합 {x.sum():+.2f}")
    except Exception as ex:  # 캐시가 없으면 건너뜀
        out(f"  (불러오기 실패: {ex})")
    out()

    # 미래 참조 점검
    out("■ 미래 참조 점검")
    bad = 0
    for s in ALL[:20]:
        S = scan(s, 5, "low")
        d = pd.read_parquet(f"{RAW}/1d_{s}.parquet")
        l = d.Low.values
        bi, px = pivots(d.High.values, l, 5, "low")
        for t0, fill, ei, cnt in S["touch"]:
            k = int(np.searchsorted(bi + 5, t0 - 1, side="right"))
            bad += int((bi[:k] + 5 > t0 - 1).any())
    out(f"  처음 닿은 날 t0의 구역에 쓰인 피벗 중 확정일(i+N)이 t0-1 넘는 것: {bad}건(20종목 점검). 진입은 전부 신호일 다음 봉 시가 또는 미리 건 지정가.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES) + "\n")
    M.to_csv(OUT.replace(".txt", "_rules.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
