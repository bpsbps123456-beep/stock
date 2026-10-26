"""반박 검증: daily_C 최선안 C_N60_수축_E1_H10 독립 재구현 (2026-10-09).
best_rule.definition 문장만 보고 종목별 pandas 루프로 다시 만든다(연구자 코드 복사 없음).
 - boxHi = max(High[t-N..t-1]), boxLo = min(Low[t-N..t-1])
 - width = boxHi/boxLo - 1, 수축 = width의 최근 PW봉(현재 포함, 최소 60) 백분위 <= thr
   pr_mode 'avg' = pandas 평균순위/개수(정의 문장), 'pine' = 직전 PW봉 중 현재 이하 비율(Pine ta.percentrank)
 - 신호 = Close[t] > boxHi[t] and Close[t-1] <= boxHi[t-1] and 수축
 - 체결 = Open[t+1], 매도 = Open[t+1+H]
 - [t-N, t+1+H] 안에 종가·시가(전일 종가 대비) -40% 이하 / +80% 이상 날이 있으면 제외
비교: R1 = 같은 묶음 전 종목 같은 날 같은 보유 평균, R2 = 같은 종목 같은 기간 아무 날 같은 보유 평균.
사용: py verify_daily_C.py → results/verify_daily_C.txt
"""
import os, sys
import numpy as np, pandas as pd

ROOT = r"C:\Users\geunt\OneDrive\Desktop\주식"
DD = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad/box/daily"
OUT = os.path.join(ROOT, r"data\derived\analysis\box_trade\results\verify_daily_C.txt")
sys.path.insert(0, os.path.join(ROOT, r"data\derived\analysis\dip_reason\code"))
from fetch import user_universe  # noqa: E402

LEV = {"SOXL", "SOXS", "TQQQ", "SQQQ", "TECL", "LABU", "NAIL", "DPST", "KORU", "SPXL", "TNA", "QQQ", "SPY", "IWM", "SMH"}
PERIODS = {"고르기": ("2026-01-02", "2026-06-30"), "확인": ("2026-07-01", "2026-10-08"), "참고25": ("2024-10-01", "2025-12-31")}

out_lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out_lines.append(s)


# ------------------------------------------------------------------ 로드
sp = set(pd.read_csv(f"{DD}/sp500_list.csv").iloc[:, 0].astype(str))
u78 = set(user_universe()["합집합"])
files = [f for f in os.listdir(DD) if f.startswith("1d_") and f.endswith(".parquet")]
DATA = {}
for f in files:
    t = f[3:-8]
    if t in LEV:
        continue
    if t not in sp and t not in u78:
        continue
    d = pd.read_parquet(f"{DD}/{f}")[["Open", "High", "Low", "Close"]].astype(float)
    d.index = pd.to_datetime(d.index).normalize()
    d = d[~d.index.duplicated()].sort_index()
    DATA[t] = d
say(f"로드: S&P500 {sum(t in sp for t in DATA)}개, 사용자78 {sum(t in u78 for t in DATA)}개")


def pct_rank_avg(x, win, minp):
    """현재 포함 최근 win개(유효값 minp 이상) 안 평균순위/개수."""
    v = x.to_numpy()
    res = np.full(len(v), np.nan)
    for i in range(len(v)):
        if np.isnan(v[i]):
            continue
        w = v[max(0, i - win + 1):i + 1]
        w = w[~np.isnan(w)]
        if len(w) < minp:
            continue
        less = (w < v[i]).sum()
        eq = (w == v[i]).sum()
        res[i] = (less + (eq + 1) / 2) / len(w)
    return res


def pct_rank_pine(x, win):
    """Pine ta.percentrank: 직전 win개 값 중 현재 이하 비율(현재 제외)."""
    v = x.to_numpy()
    res = np.full(len(v), np.nan)
    for i in range(win, len(v)):
        w = v[i - win:i]
        if np.isnan(v[i]) or np.isnan(w).any():
            continue
        res[i] = (w <= v[i]).mean()
    return res


PR_CACHE = {}


def per_ticker(t, N, PW, mode):
    key = (t, N, PW, mode)
    if key in PR_CACHE:
        return PR_CACHE[key]
    d = DATA[t]
    hi = d["High"].rolling(N, min_periods=N).max().shift(1)
    lo = d["Low"].rolling(N, min_periods=N).min().shift(1)
    wd = hi / lo - 1
    pr = pct_rank_avg(wd, PW, 60) if mode == "avg" else pct_rank_pine(wd, PW)
    PR_CACHE[key] = (hi.to_numpy(), lo.to_numpy(), pr)
    return PR_CACHE[key]


def badflag(d):
    pc = d["Close"].shift(1)
    rc = d["Close"] / pc - 1
    ro = d["Open"] / pc - 1
    return ((rc <= -0.4) | (rc >= 0.8) | (ro <= -0.4) | (ro >= 0.8)).to_numpy()


BAD = {t: badflag(d) for t, d in DATA.items()}
CBAD = {t: np.concatenate([[0], np.cumsum(b)]) for t, b in BAD.items()}


def trades(N=60, thr=0.30, PW=250, H=10, mode="avg", contr=True):
    rows = []
    for t, d in DATA.items():
        O = d["Open"].to_numpy(); C = d["Close"].to_numpy()
        hi, lo, pr = per_ticker(t, N, PW, mode)
        n = len(d)
        cb = CBAD[t]
        idx = d.index
        for i in range(1, n):
            if not (C[i] > hi[i]) or not (C[i - 1] <= hi[i - 1]):
                continue
            if contr and not (pr[i] <= thr):
                continue
            e = i + 1
            x = e + H
            if x >= n:
                continue
            a = max(i - N, 0)
            if cb[x + 1] - cb[a] > 0:
                continue
            if not (np.isfinite(O[e]) and np.isfinite(O[x])):
                continue
            rows.append((t, idx[i], idx[e], idx[x], O[e], O[x], O[x] / O[e] - 1))
    return pd.DataFrame(rows, columns=["tk", "sig", "ent", "exd", "px", "xp", "r"])


# ------------------------------------------------------------------ 비교 기준 (보유 H별 종목×날짜 수익표)
CAL = pd.read_parquet(f"{DD}/1d_SPY.parquet").index.normalize()


def ret_table(H):
    cols = {}
    for t, d in DATA.items():
        O = d["Open"].to_numpy()
        n = len(d)
        r = np.full(n, np.nan)
        cb = CBAD[t]
        for e in range(n - H):
            if cb[e + H + 1] - cb[e] > 0:
                continue
            r[e] = O[e + H] / O[e] - 1
        cols[t] = pd.Series(r, index=d.index)
    return pd.DataFrame(cols).reindex(CAL)


RT = {}


def baselines(H):
    if H not in RT:
        RT[H] = ret_table(H)
    return RT[H]


def ctstat(x, g):
    x = np.asarray(x, float)
    n = len(x)
    if n < 3:
        return np.nan
    dev = pd.Series(x - x.mean()).groupby(np.asarray(g)).sum().to_numpy()
    se = np.sqrt((dev ** 2).sum()) / n
    return x.mean() / se if se > 0 else np.nan


def evaluate(tr, uni, per, H):
    a, b = PERIODS[per]
    U = sp if uni == "S&P500" else u78
    rt = baselines(H)
    ucols = [c for c in rt.columns if c in U]
    sel = tr[tr.tk.isin(U) & (tr.ent >= a) & (tr.ent <= b)].copy()
    if len(sel) == 0:
        return dict(n=0), sel
    r1 = rt[ucols].mean(axis=1)
    pm = (rt.index >= a) & (rt.index <= b)
    r2 = rt[pm].mean(axis=0)
    sel["b1"] = sel.ent.map(r1)
    sel["b2"] = sel.tk.map(r2)
    sel = sel.dropna(subset=["b1", "b2"])
    sel["x1"] = sel.r - sel.b1
    sel["x2"] = sel.r - sel.b2
    st = dict(n=len(sel), nd=sel.ent.nunique(), avg=sel.r.mean() * 100, win=(sel.r > 0).mean() * 100,
              worst=sel.r.min() * 100, sum=sel.r.sum() * 100, t=ctstat(sel.r, sel.ent),
              x1=sel.x1.mean() * 100, t1=ctstat(sel.x1, sel.ent), x2=sel.x2.mean() * 100, t2=ctstat(sel.x2, sel.ent))
    return st, sel


def fmt(st):
    if st.get("n", 0) == 0:
        return "n=0"
    return (f"n={st['n']:>4} 날={st['nd']:>3} 평균={st['avg']:+6.2f}% 승률={st['win']:5.1f}% 최악={st['worst']:+6.1f}% "
            f"합={st['sum']:+7.0f}% t={st['t']:+5.2f} | R1초과={st['x1']:+5.2f}%(t{st['t1']:+5.2f}) R2초과={st['x2']:+5.2f}%(t{st['t2']:+5.2f})")


# ------------------------------------------------------------------ 1) 최선안 재현
say("== 1) 최선안 C_N60_수축(<=30%, 250봉, 평균순위)_E1_H10 독립 재현")
best = trades()
SEL = {}
for uni in ("S&P500", "사용자78"):
    for per in PERIODS:
        st, sel = evaluate(best, uni, per, 10)
        SEL[(uni, per)] = sel
        say(f"{uni:<7} {per:<5} {fmt(st)}")
say("")
say("-- Pine percentrank(직전 250봉, 현재 제외) 변형")
bp = trades(mode="pine")
for uni in ("S&P500", "사용자78"):
    for per in ("고르기", "확인"):
        st, _ = evaluate(bp, uni, per, 10)
        say(f"{uni:<7} {per:<5} {fmt(st)}")
say("")

# ------------------------------------------------------------------ 2) 이웃
say("== 2) 이웃 매개변수 (S&P500 / 사용자78, 고르기·확인)  — 평균 / R1초과 / R2초과 / n")
grid = []
for N in (40, 60, 80):
    grid.append(dict(N=N))
for thr in (0.2, 0.4, 1.01):
    grid.append(dict(thr=thr))
for PW in (125, 500):
    grid.append(dict(PW=PW))
for H in (5, 15):
    grid.append(dict(H=H))
for g in grid:
    H = g.get("H", 10)
    tr = trades(**g)
    parts = []
    for uni in ("S&P500", "사용자78"):
        for per in ("고르기", "확인"):
            st, _ = evaluate(tr, uni, per, H)
            if st.get("n", 0):
                parts.append(f"{uni[:3]}{per[:2]} {st['avg']:+5.2f}/{st['x1']:+5.2f}/{st['x2']:+5.2f}/{st['n']}")
            else:
                parts.append(f"{uni[:3]}{per[:2]} n=0")
    say(f"{str(g):<16} " + " | ".join(parts))
say("")

# ------------------------------------------------------------------ 3) 쏠림
say("== 3) 쏠림 (S&P500)")
for per in ("고르기", "확인"):
    s = SEL[("S&P500", per)]
    s = s.assign(m=s.ent.dt.to_period("M"))
    say(f"-- {per}: 월별 n / 평균 / R1초과")
    for m, g in s.groupby("m"):
        say(f"   {m} n={len(g):>3} 평균={g.r.mean()*100:+6.2f} R1초과={g.x1.mean()*100:+6.2f}")
    byd = s.groupby("ent").x1.agg(["size", "sum"]).sort_values("sum", key=abs, ascending=False)
    tot = s.x1.sum()
    say(f"   R1초과 합 {tot*100:+.1f}%p, 상위 1일 {byd['sum'].iloc[0]*100:+.1f}%p ({byd.index[0].date()}, {byd['size'].iloc[0]}건), "
        f"상위 3일 {byd['sum'].iloc[:3].sum()*100:+.1f}%p")
    byt = s.groupby("tk").x1.sum().sort_values(key=abs, ascending=False)
    say(f"   상위 종목 {byt.index[0]} {byt.iloc[0]*100:+.1f}%p, 상위 5종목 {byt.iloc[:5].sum()*100:+.1f}%p")
    # 최대 1일 제거 후
    top_day = byd.index[0]
    rest = s[s.ent != top_day]
    say(f"   최대 영향일 빼면 n={len(rest)} 평균={rest.r.mean()*100:+.2f} R1초과={rest.x1.mean()*100:+.2f}(t{ctstat(rest.x1, rest.ent):+.2f})")
    # 4월 제외(고르기)
    if per == "고르기":
        na = s[s.m != pd.Period("2026-04", "M")]
        say(f"   4월 빼면 n={len(na)} 평균={na.r.mean()*100:+.2f} R1초과={na.x1.mean()*100:+.2f}(t{ctstat(na.x1, na.ent):+.2f}) R2초과={na.x2.mean()*100:+.2f}")
say("")

# ------------------------------------------------------------------ 4) 확인 기간 하위 20 거래
s = SEL[("S&P500", "확인")]
say("== 4) 확인 기간 S&P500 거래 R1초과 하위·상위 5")
for _, r in pd.concat([s.nsmallest(5, "x1"), s.nlargest(5, "x1")]).iterrows():
    say(f"   {r.tk:<6} 신호 {r.sig.date()} 진입 {r.ent.date()} {r.px:.2f} → {r.exd.date()} {r.xp:.2f} 수익 {r.r*100:+.1f}% R1초과 {r.x1*100:+.1f}")
say("")

# ------------------------------------------------------------------ 5) NKE
say("== 5) NKE 2026-07-01 이후 C 신호 (N 10/20/40/60, 수축 없음)")
for N in (10, 20, 40, 60):
    tr = trades(N=N, contr=False, H=0)
    k = tr[(tr.tk == "NKE") & (tr.sig >= "2026-07-01")]
    say(f"   N{N}: {len(k)}건 " + ", ".join(str(x.date()) for x in k.sig))

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))
