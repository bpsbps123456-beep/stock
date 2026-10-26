"""반박 검증(키 verify_regime, 2026-10-09): regime.py의 최선안 B20[qbelow]/H1을 정의 문장만 보고 다시 만든다(코드 복사 없음).

정의(연구자 best_rule.definition):
  봉 일봉. Lmin[t] = min(Low[t-20..t-1]). 필터 qBelow[t] = QQQ Close[t-1] < SMA50(QQQ Close)[t-1].
  필터 참인 날 장 전에 Lmin 지정가 매수 → Low[t] < Lmin이면 체결, 체결가 min(Open[t], Lmin). 매도 Open[t+1]. 보유 중 새 진입 없음.
기간(신호일): 고르기 2026-01-02~06-30, 확인 2026-07-01~10-08, 참고 2024-11-01~2025-12-31.
기준:
  R1 = 같은 묶음·같은 날 전 종목 Open[t]→Open[t+1] 평균(연구자 주 기준과 같은 뜻).
  R2 = 같은 종목·같은 날 Open[t]→Open[t+1](같은 종목 시가 진입 = 같은 보유 기간 매수 후 보유).
  R3 = 같은 날·같은 깊이 무작위 지정가: 같은 묶음 다른 종목 k에 '전날 종가 × (Lmin_j / 전날 종가_j)' 지정가, 체결된 k만 평균.
       → '20일 최저가'라는 박스 정보가 '전날 종가 아래 같은 깊이 지정가'보다 나은지.
  R4 = 같은 종목·같은 달 다른 날 같은 깊이 지정가(체결된 날 평균).
이웃: 박스 N 15·20·25·(10·30), 이탈 깊이 0·0.5·1·2%, SMA 40·50·60, 매도 H1·H2, 재진입(매도 당일 재매수 허용/금지), 체결 관통 0·0.2·0.5%.
쏠림: 신호일 수·종목 수, 상위 1·3·5일/종목 제거 뒤 초과, 하루 빼기 최소값.
사용: py verify_regime.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "../results/verify_regime.txt"))
SRC = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad/box/daily"
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "../../dip_reason/code")))
from fetch import user_universe  # noqa: E402

PER = {"고르기": ("2026-01-02", "2026-06-30"), "확인": ("2026-07-01", "2026-10-08"), "참고": ("2024-11-01", "2025-12-31")}
LOG = []


def p(s=""):
    print(s)
    LOG.append(str(s))


# ------------------------------------------------------------ 데이터
files = sorted(glob.glob(os.path.join(SRC, "1d_*.parquet")))
raw = {os.path.basename(f)[3:-len(".parquet")]: pd.read_parquet(f) for f in files}
cal = raw["QQQ"].index
names = list(raw)
NT, NS = len(cal), len(names)
col = {nm: i for i, nm in enumerate(names)}


def mat(field):
    a = np.empty((NT, NS))
    for i, nm in enumerate(names):
        a[:, i] = raw[nm][field].reindex(cal).to_numpy(dtype=float)
    a[(a <= 0) | ~np.isfinite(a)] = np.nan
    return a


Op, Hi, Lo, Cl = mat("Open"), mat("High"), mat("Low"), mat("Close")
DS = np.array([str(x)[:10] for x in cal])
MO = np.array([x[:7] for x in DS])

sp = set(pd.read_csv(os.path.join(SRC, "sp500_list.csv")).iloc[:, 0].astype(str))
u78 = set(user_universe()["합집합"])
GRP = {"SP500": np.array([nm in sp for nm in names]), "U78": np.array([nm in u78 for nm in names])}

# 분할 의심: 하루 -40% 이하 또는 +80% 이상(종가·시가 모두 전날 종가 대비)
prevC = np.vstack([np.full((1, NS), np.nan), Cl[:-1]])
with np.errstate(divide="ignore", invalid="ignore"):
    jump = (Cl / prevC <= 0.6) | (Cl / prevC >= 1.8) | (Op / prevC <= 0.6) | (Op / prevC >= 1.8)
jump_cum = np.concatenate([np.zeros((1, NS), int), np.cumsum(jump, axis=0)])   # jump_cum[k] = jump[0..k-1] 합


def clean(a, b, j):
    """[a, b] 안(양끝 포함) 분할 의심 날 없으면 True."""
    a = np.clip(a, 0, NT - 1)
    b = np.clip(b, 0, NT - 1)
    return (jump_cum[b + 1, j] - jump_cum[a, j]) == 0


def lowest_prev(n):
    """Lmin[t] = min(Lo[t-n..t-1]); 하나라도 NaN이면 NaN (루프로 직접 계산)."""
    out = np.full((NT, NS), np.nan)
    for t in range(n, NT):
        w = Lo[t - n:t]
        ok = np.isfinite(w).all(axis=0)
        out[t, ok] = w[:, ok].min(axis=0)
    return out


q = Cl[:, col["QQQ"]]


def qbelow(n_sma):
    """qb[t] = QQQ 종가[t-1] < SMA_n[t-1]. 1/0/NaN."""
    qb = np.full(NT, np.nan)
    for t in range(n_sma, NT):
        s = q[t - n_sma:t].mean()          # SMA at t-1 (t-n..t-1)
        qb[t] = float(q[t - 1] < s)
    return qb


# 기준 R1: 같은 묶음·같은 날 시가→h봉 뒤 시가
def open_ret(h):
    r = np.full((NT, NS), np.nan)
    r[:NT - h] = Op[h:] / Op[:NT - h] - 1
    for t in range(NT):
        bad = ~clean(np.full(NS, t - 1), np.full(NS, t + h), np.arange(NS))
        r[t, bad] = np.nan
    return r


OR = {h: open_ret(h) for h in (1, 2, 3, 5)}
R1 = {(g, h): np.array([np.nanmean(OR[h][t, m]) if np.isfinite(OR[h][t, m]).any() else np.nan for t in range(NT)])
      for g, m in GRP.items() for h in OR}


def run(N=20, dep=0.0, sma=50, h=1, reentry=True, thru=0.0, filt=True, extra=None):
    Lm = LMIN.setdefault(N, lowest_prev(N))
    qb = QB.setdefault(sma, qbelow(sma))
    lim = Lm * (1 - dep)
    rows = []
    for j in range(NS):
        last_exit = -1
        for t in range(N, NT - h):
            if filt and qb[t] != 1:
                continue
            if extra is not None and not extra[t, j]:
                continue
            if not np.isfinite(lim[t, j]) or not np.isfinite(Lo[t, j]) or not np.isfinite(Op[t, j]):
                continue
            if not (Lo[t, j] < lim[t, j] * (1 - thru)):
                continue
            if (t < last_exit) or (not reentry and t == last_exit):
                continue
            x = Op[t + h, j]
            if not np.isfinite(x) or not clean(t - N, t + h, j):
                continue
            px = min(Op[t, j], lim[t, j])
            rows.append((t, j, px, x / px - 1, lim[t, j] / Cl[t - 1, j] if np.isfinite(Cl[t - 1, j]) else np.nan))
            last_exit = t + h
    d = pd.DataFrame(rows, columns=["t", "j", "px", "ret", "depth"])
    d["date"] = DS[d.t]
    d["tk"] = [names[j] for j in d.j]
    d["own"] = OR[h][d.t, d.j]
    for g in GRP:
        d["r1_" + g] = R1[(g, h)][d.t]
        d["in_" + g] = GRP[g][d.j]
    return d


LMIN, QB = {}, {}


def cl_t(x, g):
    """군집 t: 군집(날짜)별 편차 합으로 표준오차."""
    x = np.asarray(x, float)
    g = np.asarray(g)
    m = np.isfinite(x)
    x, g = x[m], g[m]
    n = len(x)
    if n < 3:
        return np.nan
    s = pd.Series(x - x.mean()).groupby(g).sum().to_numpy()
    G = len(s)
    if G < 3:
        return np.nan
    se = np.sqrt(G / (G - 1) * np.sum(s ** 2)) / n
    return x.mean() / se if se > 0 else np.nan


def cut(d, g, per):
    a, b = PER[per]
    return d[d["in_" + g] & (d.date >= a) & (d.date <= b)]


def summ(d, g):
    if len(d) == 0:
        return dict(n=0)
    ex = d.ret - d["r1_" + g]
    return dict(n=len(d), avg=d.ret.mean() * 100, win=(d.ret > 0).mean() * 100, worst=d.ret.min() * 100, sum=d.ret.sum(),
                t=cl_t(d.ret, d.t // 5), r1=d["r1_" + g].mean() * 100, ex=ex.mean() * 100, te=cl_t(ex, d.t // 5),
                te_day=cl_t(ex, d.t), ex_own=(d.ret - d.own).mean() * 100, days=d.t.nunique(), tks=d.j.nunique())


def fmt(s):
    if s["n"] == 0:
        return "n 0"
    return (f"n{s['n']:>5} 평균{s['avg']:+6.2f} 승{s['win']:3.0f}% 최악{s['worst']:+6.1f} 합{s['sum']:+7.2f} t{s['t']:+5.1f}"
            f" R1{s['r1']:+6.2f} 초과{s['ex']:+6.2f} te{s['te']:+5.1f}(일군집{s['te_day']:+5.1f}) 같은종목시가대비{s['ex_own']:+6.2f}"
            f" 신호일{s['days']:>3} 종목{s['tks']:>3}")


# ------------------------------------------------------------ R3·R4: 같은 깊이 무작위 지정가
def matched_limit(d, g, h=1):
    m = GRP[g]
    r3 = np.full(len(d), np.nan)
    r4 = np.full(len(d), np.nan)
    for t, idx in d.groupby("t").groups.items():
        sub = d.loc[idx]
        pc, lo, op = Cl[t - 1], Lo[t], Op[t]
        nxt = Op[t + h]
        okk = m & np.isfinite(pc) & np.isfinite(lo) & np.isfinite(op) & np.isfinite(nxt) & clean(
            np.full(NS, t - 1), np.full(NS, t + h), np.arange(NS))
        for i, row in zip(idx, sub.itertuples()):
            ok2 = okk.copy()
            ok2[row.j] = False
            lim = pc * row.depth
            f = ok2 & (lo < lim)
            if f.any():
                px = np.minimum(op[f], lim[f])
                r3[d.index.get_loc(i)] = np.mean(nxt[f] / px - 1)
    # R4: 같은 종목, 같은 달 다른 날
    for k, row in enumerate(d.itertuples()):
        j, t = row.j, row.t
        ss = np.where((MO == MO[t]) & (np.arange(NT) != t) & (np.arange(NT) >= 1) & (np.arange(NT) <= NT - 1 - h))[0]
        lim = Cl[ss - 1, j] * row.depth
        f = np.isfinite(lim) & (Lo[ss, j] < lim) & np.isfinite(Op[ss + h, j])
        f &= clean(ss - 1, ss + h, np.full(len(ss), j))
        if f.any():
            px = np.minimum(Op[ss[f], j], lim[f])
            r4[k] = np.mean(Op[ss[f] + h, j] / px - 1)
    return r3, r4


# ------------------------------------------------------------ 실행
p("반박 검증 verify_regime (2026-10-09) — 최선안 B20[qbelow]/H1 독립 재구현")
p(f"데이터 {DS[0]}~{DS[-1]} {NT}거래일, 종목 {NS}, S&P500 {GRP['SP500'].sum()}, 사용자78 {GRP['U78'].sum()}, 분할 의심 날 {int(np.nansum(jump))}")
qb50 = qbelow(50)
QB[50] = qb50
for per, (a, b) in PER.items():
    msk = (DS >= a) & (DS <= b)
    p(f"  {per} 거래일 {msk.sum()}, QQQ 50일선 아래(전날) 날 {int(np.nansum(qb50[msk]))}: "
      + ",".join(DS[msk & (qb50 == 1)][[0, -1]] if np.nansum(qb50[msk]) else []))
p()

best = run()
p("■ 1. 재현 (연구자 값: SP500 고르기 n3066 +0.25 초과+0.22 te3.3 / 확인 n1504 +0.64 초과+0.53 te4.1;"
  " U78 고르기 n421 +0.58 초과+0.38 / 확인 n289 +1.56 초과+1.24)")
for g in GRP:
    for per in PER:
        p(f"  {g:<5} {per}: " + fmt(summ(cut(best, g, per), g)))
p()

p("■ 2. 기준 비교 (%/거래, 괄호 일군집 t): R1 같은 묶음 같은 날 시가진입 / R2 같은 종목 같은 날 시가진입 /"
  " R3 같은 날 다른 종목 같은 깊이 지정가 / R4 같은 종목 같은 달 같은 깊이 지정가")
for g in GRP:
    for per in ("고르기", "확인"):
        d = cut(best, g, per).copy()
        r3, r4 = matched_limit(d, g)
        d["r3"], d["r4"] = r3, r4
        e1 = d.ret - d["r1_" + g]
        e2 = d.ret - d.own
        e3 = (d.ret - d.r3)
        e4 = (d.ret - d.r4)
        p(f"  {g:<5} {per}: B {d.ret.mean()*100:+.2f} | R1 {d['r1_'+g].mean()*100:+.2f} 초과 {e1.mean()*100:+.2f}({cl_t(e1, d.t):+.1f})"
          f" | R2 {d.own.mean()*100:+.2f} 초과 {e2.mean()*100:+.2f}({cl_t(e2, d.t):+.1f})"
          f" | R3 {np.nanmean(d.r3)*100:+.2f} 초과 {np.nanmean(e3)*100:+.2f}({cl_t(e3, d.t):+.1f}) n{np.isfinite(e3).sum()}"
          f" | R4 {np.nanmean(d.r4)*100:+.2f} 초과 {np.nanmean(e4)*100:+.2f}({cl_t(e4, d.t):+.1f}) n{np.isfinite(e4).sum()}")
p()

p("■ 3. 이웃 매개변수 (S&P500 | 사용자78, 칸: 고르기 초과(te) → 확인 초과(te), R1 기준)")
neigh = [("기준 N20 d0 S50 H1", {}),
         ("N15", dict(N=15)), ("N25", dict(N=25)), ("N10", dict(N=10)), ("N30", dict(N=30)),
         ("깊이 0.5%", dict(dep=0.005)), ("깊이 1%", dict(dep=0.01)), ("깊이 2%", dict(dep=0.02)),
         ("SMA40", dict(sma=40)), ("SMA60", dict(sma=60)),
         ("H2", dict(h=2)), ("재진입 금지", dict(reentry=False)),
         ("관통 0.2%", dict(thru=0.002)), ("관통 0.5%", dict(thru=0.005)),
         ("필터 없음", dict(filt=False))]
NB = {}
for nm, kw in neigh:
    d = best if not kw else run(**kw)
    NB[nm] = d
    cells = []
    for g in GRP:
        a, b = summ(cut(d, g, "고르기"), g), summ(cut(d, g, "확인"), g)
        cells.append(f"{a.get('ex', np.nan):+5.2f}({a.get('te', np.nan):+4.1f}) n{a['n']:>5} → {b.get('ex', np.nan):+5.2f}({b.get('te', np.nan):+4.1f}) n{b['n']:>5}")
    p(f"  {nm:<18} " + " | ".join(cells))
p()

p("■ 4. 쏠림 (R1 초과 기준)")
for g in GRP:
    for per in ("고르기", "확인"):
        d = cut(best, g, per).copy()
        d["ex"] = d.ret - d["r1_" + g]
        tot = d.ex.sum()
        byd = d.groupby("date").ex.agg(["sum", "count"]).sort_values("sum", ascending=False)
        byt = d.groupby("tk").ex.sum().sort_values(ascending=False)
        res = [f"{g} {per}: 초과합 {tot:+.2f}(거래당 {d.ex.mean()*100:+.2f}%), 신호일 {len(byd)}, 종목 {len(byt)}"]
        res.append("상위일 " + ", ".join(f"{i} {r['sum']:+.2f}(n{int(r['count'])})" for i, r in byd.head(5).iterrows()))
        for k in (1, 3, 5):
            dd = d[~d.date.isin(byd.index[:k])]
            res.append(f"상위{k}일 빼면 거래당 {dd.ex.mean()*100:+.2f}%(te{cl_t(dd.ex, dd.t):+.1f})")
        for k in (1, 5, 10):
            dd = d[~d.tk.isin(byt.index[:k])]
            res.append(f"상위{k}종목 빼면 {dd.ex.mean()*100:+.2f}%")
        loo = [d[d.date != x].ex.mean() * 100 for x in byd.index]
        res.append(f"하루 빼기 최소 {min(loo):+.2f}%")
        pos = (byd["sum"] > 0).mean()
        res.append(f"초과 양수 날 비율 {pos:.0%}")
        p("  " + " | ".join(res[:2]))
        p("     " + " | ".join(res[2:]))
        p("     날짜별(신호일 n, 초과%/거래): " + ", ".join(f"{i[5:]} {int(r['count'])} {r['sum']/r['count']*100:+.2f}"
                                               for i, r in byd.sort_index().iterrows()))
p()

# NKE
j = col["NKE"]
t = int(np.where(DS == "2026-10-02")[0][0])
Lm20 = LMIN[20]
p(f"■ 5. NKE 10/2: Lmin20 {Lm20[t, j]:.2f} 시가 {Op[t, j]:.2f} 저가 {Lo[t, j]:.2f} 체결 {min(Op[t, j], Lm20[t, j]):.2f}"
  f" qBelow={qb50[t]:.0f} → 최선안 신호 {'있음' if qb50[t] == 1 else '없음'}; 10/5 시가 {Op[t+1, j]:.2f}"
  f" ({Op[t+1, j]/min(Op[t, j], Lm20[t, j])-1:+.2%})")

# ------------------------------------------------------------ 6. 박스 정보 자체의 값: 같은 깊이 지정가 대조(박스 아닌 종목만) + 다른 B 형태
EC = pd.read_csv(os.path.normpath(os.path.join(HERE, "../results/earnings_cache.csv")))
EWIN = np.zeros((NT, NS), bool)
cal_d = np.array([np.datetime64(x) for x in DS])
for tk, ts in zip(EC.ticker, EC.earnings_dt.astype(str)):
    if tk not in col:
        continue
    day = np.datetime64(ts[:10])
    hr = int(ts[11:13])
    k = int(np.searchsorted(cal_d, day))
    if k >= NT:
        continue
    if cal_d[k] == day and hr >= 16:
        k += 1
    EWIN[k:k + 3, col[tk]] = True


def matched2(d, g, h, need_qb):
    """R3c = 같은 날 다른 종목 중 '자기 20일 최저가를 안 깬' 종목에 같은 깊이 지정가; R4q = 같은 종목 ±40일, (필터면 QQQ<SMA50 날만) 같은 깊이."""
    m = GRP[g]
    L20 = LMIN[20]
    r3c = np.full(len(d), np.nan)
    r4q = np.full(len(d), np.nan)
    pos = {ix: k for k, ix in enumerate(d.index)}
    for t, idx in d.groupby("t").groups.items():
        pc, lo, op, nxt = Cl[t - 1], Lo[t], Op[t], Op[t + h]
        okk = m & np.isfinite(pc) & np.isfinite(lo) & np.isfinite(op) & np.isfinite(nxt) & clean(
            np.full(NS, t - 1), np.full(NS, t + h), np.arange(NS))
        okk &= ~(lo < L20[t])            # 박스 이탈 종목 제외
        for i in idx:
            row = d.loc[i]
            lim = pc * row.depth
            f = okk & (lo < lim)
            f[int(row.j)] = False
            if f.any():
                r3c[pos[i]] = np.mean(nxt[f] / np.minimum(op[f], lim[f]) - 1)
    ar = np.arange(NT)
    for k, row in enumerate(d.itertuples()):
        j, t = row.j, row.t
        ss = ar[(np.abs(ar - t) <= 40) & (ar != t) & (ar >= 1) & (ar <= NT - 1 - h)]
        if need_qb:
            ss = ss[qb50[ss] == 1]
        if len(ss) == 0:
            continue
        lim = Cl[ss - 1, j] * row.depth
        f = np.isfinite(lim) & (Lo[ss, j] < lim) & np.isfinite(Op[ss + h, j]) & clean(ss - 1, ss + h, np.full(len(ss), j))
        if f.any():
            r4q[k] = np.mean(Op[ss[f] + h, j] / np.minimum(Op[ss[f], j], lim[f]) - 1)
    return r3c, r4q


p()
p("■ 6. 박스 정보의 값 — 같은 깊이 지정가 대조 (%/거래, 괄호 일군집 t)")
p("   R3c = 같은 날 다른 종목 중 20일 최저가를 안 깬 종목에 '전날 종가 × 같은 깊이' 지정가(체결분 평균) → 박스 이탈이란 정보가 더하는 값")
p("   R4q = 같은 종목 ±40거래일 다른 날(필터 규칙이면 QQQ<SMA50 날만) 같은 깊이 지정가")
for nm, kw, h, nq in (("B20[qbelow]/H1(최선안)", dict(), 1, True),
                      ("B20/H1(필터 없음)", dict(filt=False), 1, False),
                      ("B20/H5(표준)", dict(filt=False, h=5), 5, False),
                      ("B20[실적]/H5(나이키형)", dict(filt=False, h=5, extra=EWIN), 5, False)):
    d0 = run(**kw)
    for g in GRP:
        for per in ("고르기", "확인"):
            d = cut(d0, g, per).copy()
            if len(d) < 3:
                continue
            r3c, r4q = matched2(d, g, h, nq)
            e1 = d.ret - d["r1_" + g]
            e3 = d.ret.to_numpy() - r3c
            e4 = d.ret.to_numpy() - r4q
            p(f"   {nm:<22} {g:<5} {per}: n{len(d):>5} 신호일{d.t.nunique():>3} B{d.ret.mean()*100:+6.2f}"
              f" | R1초과{e1.mean()*100:+6.2f}({cl_t(e1, d.t):+4.1f})"
              f" | R3c{np.nanmean(r3c)*100:+6.2f} 초과{np.nanmean(e3)*100:+6.2f}({cl_t(e3, d.t):+4.1f})"
              f" | R4q{np.nanmean(r4q)*100:+6.2f} 초과{np.nanmean(e4)*100:+6.2f}({cl_t(e4, d.t):+4.1f}) n{np.isfinite(e4).sum()}")
    if nm.startswith("B20[실적]"):
        for g in GRP:
            d = cut(d0, g, "확인").copy()
            d["ex"] = d.ret - d["r1_" + g]
            byt = d.groupby("tk").ex.sum().sort_values(ascending=False)
            byd = d.groupby("date").ex.agg(["sum", "count"]).sort_values("sum", ascending=False)
            cc = [f"{nm} {g} 확인 쏠림: 종목 {len(byt)}, 신호일 {len(byd)}"]
            for k in (1, 3, 5):
                dd = d[~d.tk.isin(byt.index[:k])]
                cc.append(f"상위{k}종목 빼면 {dd.ex.mean()*100:+.2f}%({cl_t(dd.ex, dd.t):+.1f})")
            for k in (1, 3):
                dd = d[~d.date.isin(byd.index[:k])]
                cc.append(f"상위{k}일 빼면 {dd.ex.mean()*100:+.2f}%")
            cc.append("상위 종목 " + ", ".join(f"{i} {v*100:+.0f}%p" for i, v in byt.head(5).items()))
            p("   " + " | ".join(cc))

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(LOG) + "\n")
