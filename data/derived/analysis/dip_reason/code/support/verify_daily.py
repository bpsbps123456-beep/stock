"""반박 검증: 일봉 지지 구역 '며칠 버팀' 최선안(E1 D=1, N=5, 3일 보유) 독립 재구현 (2026-10-09).

best_rule.definition 글만 보고 새로 짰다(support/daily.py 코드는 가져오지 않음).
  구역: 일봉 좌우 N봉 스윙 저점(l[i]가 i-N..i+N 최소, 왼쪽 N봉은 모두 더 높음), i+N <= t-1 인 피벗 중 최근 60개,
        가격순으로 구역 하단 대비 0.5% 안이면 한 구역, 중심끼리 1.2% 미만이면 닿은 횟수 적은 쪽(같으면 오래된 쪽) 제외.
  지지: 중심 < 전날 종가 중 가장 높은 구역. 처음 닿음 t0: 전날 종가 > hi*1.01, 당일 저가 <= hi*1.01. 구역 고정.
  버팀: 종가 < lo*0.99 깨짐, 종가 > hi*1.04 위로 떠남, 그 밖 D += 1. 진입 E1(d): D == d 된 날 다음 날 시가. 매도 e+h 시가.
  기준: A2 처음 닿은 날 다음 날 시가, A1 hi*1.01 지정가(min(시가, 지정가)), 무작위(같은 종목·같은 달, 그리고 무조건).
점검: 재현, 이웃 매개변수, 쏠림, 단조성(D 구간표).
사용: py support/verify_daily.py   (code 폴더에서)
"""
import os, sys
from datetime import date
import numpy as np, pandas as pd

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
from fetch import universe, user_universe

RAW = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad/dip/raw"
OUT = os.path.normpath(os.path.join(CODE, "../results/support/verify_daily.txt"))
SPLIT = pd.Timestamp("2026-04-07")
U78 = sorted(set(user_universe()["합집합"]))
ALL = sorted(set(U78) | set(universe()))
SPY_IDX = {d: i for i, d in enumerate(pd.read_parquet(f"{RAW}/1d_SPY.parquet").index)}
BASE = dict(N=5, eps=0.01, up=0.04, tol=0.005, gap=0.012, keep=60, tie="new")

DATA = {}
for s in ALL:
    d = pd.read_parquet(f"{RAW}/1d_{s}.parquet")
    DATA[s] = (d.index, d.Open.to_numpy(float), d.High.to_numpy(float), d.Low.to_numpy(float), d.Close.to_numpy(float))

LOG = []


def say(x=""):
    print(x); LOG.append(str(x))


def swing_lows(l, N):
    idx = []
    for i in range(N, len(l) - N):
        if l[i] <= l[i - N:i + N + 1].min() and (l[i - N:i] > l[i]).all():
            idx.append(i)
    return np.asarray(idx, int)


def make_zones(bars, l, P):
    bars = bars[-P["keep"]:]
    pr = l[bars]
    zs = []
    for k in np.argsort(pr, kind="mergesort"):
        p, b = pr[k], bars[k]
        if zs and p <= zs[-1]["lo"] * (1 + P["tol"]):
            z = zs[-1]; z["hi"] = p; z["n"] += 1; z["new"] = max(z["new"], b); z["old"] = min(z["old"], b)
        else:
            zs.append(dict(lo=p, hi=p, n=1, new=b, old=b))
    age = "new" if P["tie"] == "new" else "old"
    kept = []
    for z in sorted(zs, key=lambda z: (z["n"], z[age]), reverse=True):
        z["c"] = (z["lo"] + z["hi"]) / 2
        if all(abs(z["c"] - k["c"]) / k["c"] >= P["gap"] for k in kept):
            kept.append(z)
    return kept


def run_symbol(s, P):
    """에피소드 상태 기계. 반환: touches [(t0, 지정가)], alive [(t, D, ep)], ep_end {ep: (t_end, 끝 종류)}"""
    idx, o, h, l, c = DATA[s]
    n = len(c)
    piv = swing_lows(l, P["N"])
    conf = piv + P["N"]
    zc = {}
    touches, alive, ends = [], [], {}
    ep = None
    for t in range(1, n):
        if ep is None:
            k = int((conf <= t - 1).sum())
            if k == 0:
                continue
            if k not in zc:
                zc[k] = make_zones(piv[:k], l, P)
            below = [z for z in zc[k] if z["c"] < c[t - 1]]
            if not below:
                continue
            z = max(below, key=lambda z: z["c"])
            top = z["hi"] * (1 + P["eps"])
            if not (c[t - 1] > top and l[t] <= top):
                continue
            ep = dict(id=len(touches), lo=z["lo"], hi=z["hi"], D=0)
            touches.append((t, min(o[t], top)))
        if c[t] < ep["lo"] * (1 - P["eps"]):
            ends[ep["id"]] = (t, "깨짐"); ep = None; continue
        if c[t] > ep["hi"] * (1 + P["up"]):
            ends[ep["id"]] = (t, "위로"); ep = None; continue
        ep["D"] += 1
        alive.append((t, ep["D"], ep["id"]))
    return touches, alive, ends


def trades(P, H=(1, 3, 5, 10), Ds=(1,)):
    """규칙별 거래 표. rule: 'E1 D=d', 'A2', 'A1', 'Dk'(k=1..11 단조성용)."""
    rows = []
    for s in ALL:
        idx, o, h, l, c = DATA[s]
        n = len(o)
        touches, alive, ends = run_symbol(s, P)
        ents = [("A2", t0 + 1, None, i, t0) for i, (t0, lim) in enumerate(touches)]
        ents += [("A1", t0, lim, i, t0) for i, (t0, lim) in enumerate(touches)]
        for t, Dd, ei in alive:
            if Dd in Ds:
                ents.append((f"E1 D={Dd}", t + 1, None, ei, t))
            ents.append((f"D{min(Dd, 11)}", t + 1, None, ei, t))
        for rule, e, lim, ei, t in ents:
            if e >= n:
                continue
            px = o[e] if lim is None else lim
            te, kind = ends.get(ei, (10 ** 9, "열림"))
            brk5 = kind == "깨짐" and te - t <= 5
            for hh in H:
                if e + hh < n:
                    rows.append((rule, s, e, idx[e], idx[e + hh], hh, o[e + hh] / px - 1, brk5))
    T = pd.DataFrame(rows, columns=["rule", "sym", "e", "de", "dx", "h", "ret", "brk5"])
    T["P"] = np.where((T.de < SPLIT) & (T.dx < SPLIT), "고르기", np.where(T.de >= SPLIT, "확인", ""))
    T = T[T.P != ""].copy()
    T["ym"] = T.de.dt.year * 100 + T.de.dt.month
    T["cal"] = T.de.map(SPY_IDX).fillna(-1).astype(int)
    return T


def random_tables(H=(1, 3, 5, 10)):
    rows = []
    for s in ALL:
        idx, o, *_ = DATA[s]
        for hh in H:
            e = np.arange(0, len(o) - hh)
            rows.append(pd.DataFrame(dict(sym=s, h=hh, de=idx[e], dx=idx[e + hh], ret=o[e + hh] / o[e] - 1)))
    R = pd.concat(rows, ignore_index=True)
    R["ym"] = R.de.dt.year * 100 + R.de.dt.month
    R["P"] = np.where((R.de < SPLIT) & (R.dx < SPLIT), "고르기", np.where(R.de >= SPLIT, "확인", ""))
    return R


def cl_t(x, g):
    x = np.asarray(x, float)
    if len(x) < 3:
        return np.nan
    dev = pd.Series(x - x.mean()).groupby(np.asarray(g)).sum().to_numpy()
    G = len(dev)
    if G < 2:
        return np.nan
    se = np.sqrt(G / (G - 1) * (dev ** 2).sum()) / len(x)
    return x.mean() / se if se > 0 else np.nan


R = random_tables()
RM = R.groupby(["sym", "ym", "h"]).ret.mean()
RU = R[R.P != ""].groupby(["P", "h"]).apply(lambda g: pd.Series({u: g[g.sym.isin(set(us))].ret.mean() for u, us in (("U78", U78), ("ALL", ALL))}))


def summarize(T, rule, hh, P, uni):
    g = T[(T.rule == rule) & (T.h == hh) & (T.P == P) & T.sym.isin(set(U78 if uni == "U78" else ALL))]
    if not len(g):
        return dict(n=0)
    rnd = RM.reindex(pd.MultiIndex.from_arrays([g.sym, g.ym, g.h])).to_numpy()
    ex = g.ret.to_numpy() - rnd
    cl = g.cal // max(5, hh)
    return dict(n=len(g), avg=g.ret.mean(), win=(g.ret > 0).mean(), worst=g.ret.min(), sum=g.ret.sum(),
                ex=np.nanmean(ex), t_ex=cl_t(ex, cl), t=cl_t(g.ret, cl), brk5=g.brk5.mean(), g=g)


def pct(v):
    return "nan" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:+.2f}%"


# ---------------- 1. 재현 ----------------
say("■ 1. 독립 재현: E1 D=1, N=5, 3일 보유 (기준 A2·A1·무작위)")
TB = trades(BASE)
for uni in ("U78", "ALL"):
    for P in ("고르기", "확인"):
        b = summarize(TB, "E1 D=1", 3, P, uni)
        a2 = summarize(TB, "A2", 3, P, uni)
        a1 = summarize(TB, "A1", 3, P, uni)
        say(f"  [{uni} {P}] E1 n{b['n']} 평균 {pct(b['avg'])} 승률 {b['win']:.0%} 최악 {pct(b['worst'])} 합 {b['sum']:+.2f} "
            f"초과(같은달 무작위) {pct(b['ex'])} t초과 {b['t_ex']:+.2f} | A2 n{a2['n']} {pct(a2['avg'])} → 차이 {pct(b['avg'] - a2['avg'])}"
            f" | A1 {pct(a1['avg'])} | 무조건 무작위 3일 {pct(RU.loc[(P, 3), uni])}")
say()

# ---------------- 2. 이웃 매개변수 ----------------
say("■ 2. 이웃 매개변수 — 한 번에 하나씩 바꿈. 표시: E1 평균 / A2 평균 / 차이(E1-A2) / 같은달 무작위 초과, 사용자78")
variants = [("기본", {})] + [(f"N={v}", dict(N=v)) for v in (3, 4, 6, 7)] + \
           [(f"닿음·깨짐 {v:.1%}", dict(eps=v)) for v in (0.005, 0.02)] + \
           [(f"떠남 {v:.0%}", dict(up=v)) for v in (0.03, 0.06)] + \
           [(f"묶음 {v:.1%}", dict(tol=v)) for v in (0.003, 0.01)] + \
           [(f"간격 {v:.1%}", dict(gap=v)) for v in (0.008, 0.02)] + \
           [("동률=첫 피벗 기준", dict(tie="old")), ("최근 스윙 30개", dict(keep=30))]
NB = []
for name, ch in variants:
    P_ = dict(BASE, **ch)
    T = TB if not ch else trades(P_, H=(1, 2, 3, 4, 5), Ds=(1, 2, 3))
    hs = (3,) if ch else (3,)
    for uni in ("U78", "ALL"):
        cells = []
        for P in ("고르기", "확인"):
            b = summarize(T, "E1 D=1", 3, P, uni); a2 = summarize(T, "A2", 3, P, uni)
            cells.append((P, b["n"], b["avg"], a2["avg"], b["avg"] - a2["avg"], b["ex"]))
            NB.append(dict(변형=name, uni=uni, P=P, n=b["n"], E1=b["avg"], A2=a2["avg"], 차이=b["avg"] - a2["avg"], 초과=b["ex"]))
        say(f"  {name:14s} {uni}: " + " || ".join(f"{P} n{n} {pct(e)} / {pct(a)} / {pct(d)} / 초과 {pct(x)}" for P, n, e, a, d, x in cells))
say()
say("  같은 기본 구역에서 D·보유만 바꿈 (사용자78): E1 평균 / A2 평균 / 차이")
TH = trades(BASE, H=(1, 2, 3, 4, 5), Ds=(1, 2, 3))
for dd in (1, 2, 3):
    for hh in (1, 2, 3, 4, 5):
        cells = []
        for P in ("고르기", "확인"):
            b = summarize(TH, f"E1 D={dd}", hh, P, "U78"); a2 = summarize(TH, "A2", hh, P, "U78")
            cells.append(f"{P} n{b['n']} {pct(b['avg'])} / {pct(a2['avg'])} / {pct(b['avg'] - a2['avg'])}")
            NB.append(dict(변형=f"D={dd} h={hh}", uni="U78", P=P, n=b["n"], E1=b["avg"], A2=a2["avg"], 차이=b["avg"] - a2["avg"], 초과=b["ex"]))
        say(f"  D={dd} h={hh}: " + " || ".join(cells))
NBd = pd.DataFrame(NB)
for P in ("고르기", "확인"):
    x = NBd[NBd.P == P]
    say(f"  [{P}] 이웃 {len(x)}칸 중 E1 > A2: {(x.차이 > 0).sum()}칸, E1 평균 > 0: {(x.E1 > 0).sum()}칸, 같은달 무작위 초과 > 0: {(x.초과 > 0).sum()}칸")
say()

# ---------------- 3. 쏠림 ----------------
say("■ 3. 쏠림 — 기본 규칙(E1 D=1, N=5, 3일)")
for uni in ("U78", "ALL"):
    for P in ("고르기", "확인"):
        g = summarize(TB, "E1 D=1", 3, P, uni)["g"]
        bs = g.groupby("sym").ret.sum().sort_values(ascending=False)
        bd = g.groupby("de").ret.sum().sort_values(ascending=False)
        top = g.ret.max()
        wo1 = g[g.sym != bs.index[0]].ret.mean()
        wo3 = g[~g.sym.isin(bs.index[:3])].ret.mean()
        wod = g[g.de != bd.index[0]].ret.mean()
        wod5 = g[~g.de.isin(bd.index[:5])].ret.mean()
        trim = g.ret.sort_values().iloc[:-max(1, len(g) // 100)].mean()
        say(f"  [{uni} {P}] n{len(g)} 평균 {pct(g.ret.mean())} 중앙값 {pct(g.ret.median())} 합 {g.ret.sum():+.2f} | "
            f"1위 종목 {bs.index[0]} {bs.iloc[0]:+.2f}({bs.iloc[0] / g.ret.sum():.0%}) 빼면 {pct(wo1)}, 상위3 빼면 {pct(wo3)} | "
            f"1위 날 {bd.index[0]:%m-%d} {bd.iloc[0]:+.2f} 빼면 {pct(wod)}, 상위5일 빼면 {pct(wod5)} | 최대 1건 {pct(top)}, 상위1% 빼면 {pct(trim)}")
say()

# ---------------- 4. 단조성 ----------------
say("■ 4. 단조성 다시 만들기 — D 구간별, 다음 날 시가 매수, 같은달 무작위 초과(5일·10일)와 5일 안 깨짐 비율")
bins = [(1, 1, "1"), (2, 2, "2"), (3, 4, "3~4"), (5, 7, "5~7"), (8, 10, "8~10"), (11, 11, "11+")]
TB3 = trades(dict(BASE, N=3))
for N, T in ((3, TB3), (5, TB)):
    for uni in ("U78", "ALL"):
        rows = []
        for a, b, lab in bins:
            r = [lab]
            for P in ("고르기", "확인"):
                for hh in (5, 10):
                    parts = [summarize(T, f"D{k}", hh, P, uni) for k in range(a, b + 1)]
                    gs = [p["g"] for p in parts if p["n"]]
                    g = pd.concat(gs)
                    rnd = RM.reindex(pd.MultiIndex.from_arrays([g.sym, g.ym, g.h])).to_numpy()
                    r += [len(g), f"{np.nanmean(g.ret.to_numpy() - rnd) * 100:+.2f}"]
                    if hh == 5:
                        r.append(f"{g.brk5.mean():.0%}")
            rows.append(r)
        say(f"  [N={N} {uni}]")
        say(pd.DataFrame(rows, columns=["D", "고n5", "고초과5", "고깨짐5", "고n10", "고초과10", "확n5", "확초과5", "확깨짐5", "확n10", "확초과10"]).to_string(index=False))
        # 스피어만(D1~11 ↔ 5일 초과)
        for P in ("고르기", "확인"):
            ex = []
            for k in range(1, 12):
                p = summarize(T, f"D{k}", 5, P, uni)
                ex.append(p["ex"] if p["n"] else np.nan)
            sp = pd.Series(range(1, 12), dtype=float).corr(pd.Series(ex), method="spearman")
            say(f"    {P} D(1~11)↔초과5 스피어만 {sp:+.2f}")
say()

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(LOG) + "\n")
