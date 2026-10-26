"""이격매수매도 — 매수 순간의 '동종·시장 동반' 모양으로 반등을 미리 알 수 있나 (2026-10-10).

질문(사용자): 10/9 CRCL은 정규장 신호 뒤 빠르게 반등했고 MRVL은 계속 빠졌다. LITE·COHR는 광자 테마로 함께 급등했다가
되돌렸는데 LITE만 회복했다. 3분봉으로 미리 알 수 있었나. 이 파일은 그중 '동종·시장 동반' 갈래만 맡는다.

결과 변수(시험 전에 고정, 바꾸지 않음):
  Y1 = 거래 수익 ret (지금 매도 규칙: RSI14>70 다음 봉 시가 또는 매수 뒤 첫 한국 08시 이후 봉 시가)
  Y2 = 빠른 반등: 매수 뒤 20봉(60분) 안에 RSI 매도 (why == RSI and bars <= 20)
  사건 E(테마 급등 되돌림, 종목-일 단위, 정규장):
    그날 04:00부터 봉 k까지의 3분봉 종가 최고 H_k가 전일 정규장 종가 pc의 +5% 이상이고,
    정규장 봉 k 종가가 처음으로 상승폭의 절반 이상을 되돌림(c_k <= pc + 0.5*(H_k - pc)).
    진입 = 봉 k+1 시가(정규장 안). 고점은 꼬리 체결을 피하려고 고가가 아니라 종가로 잰다.
    EY1 = 그날 정규장 마지막 봉 종가 / 진입 - 1
    EY2 = 진입 뒤 그날 정규장 종가 최고가 진입 + 0.5*(H_k - 진입) 이상(되돌린 폭의 절반 회복) — 0/1
특성(모두 봉 i 시가에 알 수 있는 값: 자기 종목은 봉 i-1, 동종·ETF는 i 시작 시각 전에 끝난 봉, 일봉은 전날까지):
  동종 = watchlist.AI_UNIVERSE 같은 소분류의 미국 티커(데이터 있는 것 2개 이상, 아니면 같은 대분류).
         바이오 12 + TEM은 한 묶음, 유니버스 밖 사용자 종목은 코인(COIN CRCL HOOD MSTR)·전자상거래(EBAY MELI PDD)·
         GOOG(미국 빅테크, GOOGL 제외)·SKHY(메모리 대분류). SOXX는 동종 없음.
  pdev     동종 EMA60 대비 위치 평균
  rel_dev  자기 위치 - pdev (동종보다 더 이격됨)
  ptouch   직전 30분(10봉) 안에 자기 이격 선(EMA60[1]x0.98)에 닿은 동종 비율
  rel20    자기 직전 20봉(60분) 수익 - 동종 평균
  pday     동종 평균 그날 수익(전일 정규장 종가 대비)
  rel_day  자기 그날 수익 - pday
  phi      동종 평균 그날 최고(종가 기준, 04:00부터) 상승폭
  rel_hi   자기 그날 최고 상승폭 - phi (테마 급등을 앞에서 끌었나)
  pgive    동종 평균 그날 고점 대비 하락폭 (사건 시험 전용)
  rel_give 자기 고점 대비 하락폭 - pgive (사건 시험 전용)
  smh_dev  SMH EMA60 대비 위치
  lead5    전날까지 5거래일 수익 - 동종 평균(대장인가)
  lead20   전날까지 20거래일 수익 - 동종 평균
  sdev_q   자기 위치 - QQQ 위치(trades.csv, 직전 연구 '혼자 빠짐'; 정규장·깨끗한 체결로 다시 봄)
  통제: open30(09:30~09:59 매수), atr_d(전날까지 일봉 ATR14/종가), ivol(직전 60봉 3분 수익 표준편차), 날짜 고정효과
방법:
  정규장 거래가 주 결과다. 프리·애프터는 '깨끗한 체결'(처음 닿은 1분봉의 직전 1분 종가가 매수가보다 1% 넘게 위가 아닌 체결)만 보조로 본다.
  문턱(하위 1/3, 상위 1/3)과 방향은 앞 절반(9/10~9/23) 정규장에서 정하고 뒤 절반(9/24~10/8)에 그대로 쓴다.
  t = 날짜 군집(날짜마다 A평균 - B평균의 평균/표준오차). 앞에서 |t| 상위 특성 3개를 뒤에서 확인한다.
  확인: 날짜 고정효과 회귀(open30·log atr_d·log ivol 통제, 날짜 군집 표준오차), 개장 30분 안/밖 따로, 상위 1종목·1일 빼기,
        무작위 특성 같은 절차(선택편향 크기), 앞 기간 5분봉 근사(7/14~9/8, EMA36·RSI9) 보조.
입력: ../results/trades.csv, raw_1009(1분·일봉), stop_lab early 캐시. 다시 받지 않는다.
출력: ../results/pat_peers.txt, pat_peers_tests.csv, pat_peers_trades.csv, pat_peers_events.csv
사용: py pat_peers.py"""
import os, re, sys
from datetime import date
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "../../dip_reason/code"))
sys.path.insert(0, os.path.join(HERE, "../../../../.."))
from events import bars, prep
from fetch import BIO
import watchlist as w

SP = r"C:/Users/geunt/AppData/Local/Temp/claude/C--Users-geunt-OneDrive-Desktop---/4e686c73-74be-4337-8ccc-2f6136beaf7d/scratchpad/dip"
RAW = os.path.join(SP, "raw_1009")
RAW_OLD = os.path.join(SP, "raw")
RES = os.path.join(HERE, "../results")
MID = date(2026, 9, 24)
LEV = set("SOXL,SOXS,TQQQ,SQQQ,TECL,TECS,LABU,LABD,NAIL,DPST,KORU,SPXL,SPXS,UPRO,SPXU,TNA,TZA,FNGU,FNGD,YINN,YANG".split(","))
TRADE_F = ["pdev", "rel_dev", "ptouch", "rel20", "pday", "rel_day", "phi", "rel_hi", "smh_dev", "lead5", "lead20", "sdev_q"]
EVENT_F = ["pdev", "rel_dev", "ptouch", "rel20", "pday", "rel_day", "phi", "rel_hi", "pgive", "rel_give", "smh_dev", "lead5", "lead20"]
SPLITS = ("하위1/3", "상위1/3")
LINES = []
RNG = np.random.default_rng(20261010)


def say(*a):
    s = " ".join(str(x) for x in a)
    LINES.append(s)
    print(s)


def pct(x):
    return "nan" if pd.isna(x) else f"{100 * x:+.2f}%"


# ---------------- 동종 묶음 ----------------
def peer_map(have):
    sub = {}
    for big, subs in w.AI_UNIVERSE.items():
        for sn, ts in subs.items():
            for t in ts:
                if re.fullmatch(r"[A-Z]{1,5}", t):
                    sub[t] = (big, sn)
    for t in BIO + ["TEM"]:
        sub[t] = ("바이오·유전체", "바이오·유전체")
    for t in ["COIN", "CRCL", "HOOD", "MSTR"]:
        sub[t] = ("코인", "코인")
    for t in ["EBAY", "MELI", "PDD"]:
        sub[t] = ("전자상거래", "전자상거래")
    sub["GOOG"] = ("빅테크·AI 모델", "미국 빅테크")
    sub["SKHY"] = ("메모리·스토리지", "DRAM·HBM")
    dup = {"GOOG": "GOOGL", "GOOGL": "GOOG"}
    P, lvl = {}, {}
    for s in have:
        if s not in sub:
            P[s], lvl[s] = [], "없음"
            continue
        b, sn = sub[s]
        p = [x for x, (bb, ss) in sub.items() if bb == b and ss == sn and x != s and x in have and x != dup.get(s)]
        lvl[s] = "소분류"
        if len(p) < 2:
            p = [x for x, (bb, ss) in sub.items() if bb == b and x != s and x in have and x != dup.get(s)]
            lvl[s] = "대분류"
        P[s] = p
        if not p:
            lvl[s] = "없음"
    return P, lvl, sub


# ---------------- 종목별 봉 표 ----------------
def sym_table(raw, s, mins, el, rl):
    d = prep(bars(raw, s, mins), el, rl)
    dd = pd.read_parquet(os.path.join(raw, f"1d_{s}.parquet"))
    dd.index = pd.to_datetime(dd.index).date
    c1 = dd.Close
    prevc = c1.shift(1)
    pc = pd.Series(d.date.values).map(prevc).values.astype(float)
    gap = 0.03 if s in LEV else 0.02
    n20, win, ivw = 60 // mins, 30 // mins, 180 // mins
    T = pd.DataFrame(index=d.index)
    T["dev"] = d.dev.values
    tch = (d.l <= d.ema.shift(1) * (1 - gap)).astype(float)
    T["touch"] = tch.rolling(win, min_periods=1).max().values
    T["touch_s"] = tch.rolling(max(win // 2, 1), min_periods=1).max().values      # 15분 (견고성 전용)
    T["touch_l"] = tch.rolling(win * 2, min_periods=1).max().values               # 60분 (견고성 전용)
    T["r20"] = (d.c / d.c.shift(n20) - 1).values
    T["dret"] = d.c.values / pc - 1
    hc = d.c.groupby(d.date.values).cummax()
    T["dhi"] = hc.values / pc - 1
    T["ddh"] = T.dhi - T.dret
    T["ivol"] = np.log(d.c).diff().rolling(ivw, min_periods=ivw // 2).std().values
    # 일봉(전날까지): 5·20거래일 수익, ATR14/종가
    tr = pd.concat([dd.High - dd.Low, (dd.High - dd.Close.shift()).abs(), (dd.Low - dd.Close.shift()).abs()], axis=1).max(axis=1)
    daily = pd.DataFrame({"ret5": c1.shift(1) / c1.shift(6) - 1, "ret20": c1.shift(1) / c1.shift(21) - 1,
                          "atr_d": (tr / dd.Close).rolling(14).mean().shift(1), "pc": prevc})
    return d, T, daily


def build_all(raw, mins, el, rl, syms):
    D, T, DL = {}, {}, {}
    for s in syms:
        try:
            D[s], T[s], DL[s] = sym_table(raw, s, mins, el, rl)
        except FileNotFoundError:
            pass
    grid = pd.DatetimeIndex(sorted(set().union(*[t.index for t in T.values()])))
    gday = pd.Series(grid.date, index=grid)
    lim = 30 // mins
    G = {}
    for col in ["dev", "touch", "touch_s", "touch_l", "r20", "dret", "dhi", "ddh"]:
        f = pd.DataFrame({s: t[col].reindex(grid) for s, t in T.items()})
        f = f.groupby(gday.values).ffill(limit=lim)
        G[col] = f.shift(1)        # 행 t = t 시작 전에 끝난 봉까지의 값
    return D, T, DL, G, grid


def feats_at(s, pos, D, T, DL, G, grid, P):
    """종목 s의 봉 pos(매수 봉 i) 시가 시점 특성."""
    t = D[s].index[pos]
    own = T[s].iloc[pos - 1]
    dy = D[s].date.iloc[pos]
    pe = P.get(s, [])
    gi = grid.get_loc(t)
    out = dict(open30=float(570 <= D[s]["mod"].iloc[pos] < 600), ivol=own.ivol,
               atr_d=DL[s].atr_d.get(dy, np.nan), own_dev=own.dev, own_dhi=own.dhi, own_dret=own.dret)
    sm = G["dev"].iloc[gi]
    out["smh_dev"] = sm.get("SMH", np.nan)
    out["qqq_dev"] = sm.get("QQQ", np.nan)
    if not pe:
        for k in EVENT_F:
            out.setdefault(k, np.nan)
        return out
    row = {c: G[c].iloc[gi][pe] for c in G}
    out["pdev"] = row["dev"].mean()
    out["rel_dev"] = own.dev - out["pdev"]
    out["ptouch"] = row["touch"].mean()
    out["ptouch_s"] = row["touch_s"].mean()
    out["ptouch_l"] = row["touch_l"].mean()
    out["rel20"] = own.r20 - row["r20"].mean()
    out["pday"] = row["dret"].mean()
    out["rel_day"] = own.dret - out["pday"]
    out["phi"] = row["dhi"].mean()
    out["rel_hi"] = own.dhi - out["phi"]
    out["pgive"] = row["ddh"].mean()
    out["rel_give"] = own.ddh - out["pgive"]
    r5 = [DL[p].ret5.get(dy, np.nan) for p in pe]
    r20 = [DL[p].ret20.get(dy, np.nan) for p in pe]
    out["lead5"] = DL[s].ret5.get(dy, np.nan) - np.nanmean(r5) if np.isfinite(r5).any() else np.nan
    out["lead20"] = DL[s].ret20.get(dy, np.nan) - np.nanmean(r20) if np.isfinite(r20).any() else np.nan
    out["n_peer"] = len(pe)
    return out


# ---------------- 깨끗한 체결(1분봉) ----------------
def clean_flag(tr, raw):
    out = np.full(len(tr), np.nan)
    prv_all = np.full(len(tr), np.nan)
    for s, g in tr.groupby("sym"):
        m = pd.read_parquet(os.path.join(raw, f"1m_{s}.parquet"))
        idx = m.index
        for k, r in zip(g.index, g.itertuples()):
            t0 = r.ts
            a = idx.searchsorted(t0)
            b = idx.searchsorted(t0 + pd.Timedelta(minutes=3))
            w_ = m.iloc[a:b]
            hit = np.where(w_.Low.values <= r.ent * (1 + 1e-6))[0]
            if len(hit) and a + hit[0] > 0:
                p1 = a + hit[0]
                prv = m.Close.iloc[p1 - 1] / r.ent - 1
                prv_all[tr.index.get_loc(k)] = prv
                out[tr.index.get_loc(k)] = float(prv <= 0.01)
    return out, prv_all


# ---------------- 통계 ----------------
def day_t(day, y, a):
    """날짜마다 A평균 - B평균 → (평균, t, 날짜 수). day: 정수 코드."""
    ok = np.isfinite(y)
    day, y, a = day[ok], y[ok], a[ok]
    if len(day) == 0:
        return np.nan, np.nan, 0
    n = day.max() + 1
    sA = np.bincount(day[a], y[a], n); cA = np.bincount(day[a], None, n)
    sB = np.bincount(day[~a], y[~a], n); cB = np.bincount(day[~a], None, n)
    k = (cA > 0) & (cB > 0)
    d = sA[k] / cA[k] - sB[k] / cB[k]
    if len(d) < 3 or d.std(ddof=1) == 0:
        return (d.mean() if len(d) else np.nan), np.nan, len(d)
    return d.mean(), d.mean() / (d.std(ddof=1) / np.sqrt(len(d))), len(d)


def fe_ols(df, ycol, acol, ctrl):
    x = df[[ycol, acol] + ctrl + ["day"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 20:
        return np.nan, np.nan, 0
    g = x.day.values
    cols = [acol] + ctrl
    M = x[[ycol] + cols].astype(float)
    M = M - M.groupby(g).transform("mean")
    X, y = M[cols].values, M[ycol].values
    XtX = X.T @ X
    if np.linalg.matrix_rank(XtX) < len(cols):
        return np.nan, np.nan, 0
    inv = np.linalg.inv(XtX)
    b = inv @ X.T @ y
    e = y - X @ b
    meat = np.zeros_like(XtX)
    ug = np.unique(g)
    for v in ug:
        k = g == v
        u = X[k].T @ e[k]
        meat += np.outer(u, u)
    G_ = len(ug)
    V = inv @ meat @ inv * G_ / max(G_ - 1, 1)
    return b[0], b[0] / np.sqrt(V[0, 0]), G_


def pooled_t(day, y, a):
    """날짜 고정효과 없이 A평균 - B평균, 표준오차는 날짜 군집(그날 전체가 강했나 같은 하루 단위 특성용, 참고)."""
    ok = np.isfinite(y)
    day, y, a = day[ok], y[ok], a[ok].astype(float)
    if a.sum() < 2 or (1 - a).sum() < 2:
        return np.nan, np.nan
    X = np.column_stack([np.ones(len(y)), a])
    inv = np.linalg.inv(X.T @ X)
    b = inv @ X.T @ y
    e = y - X @ b
    meat = np.zeros((2, 2))
    ug = np.unique(day)
    for v in ug:
        k = day == v
        u = X[k].T @ e[k]
        meat += np.outer(u, u)
    V = inv @ meat @ inv * len(ug) / max(len(ug) - 1, 1)
    return b[1], b[1] / np.sqrt(V[1, 1])


def tod_bucket(mod):
    """정규장 시각 칸: 첫 3봉(09:30~09:38), 09:39~09:59, 10:00~11:59, 12:00~15:59."""
    return np.select([mod < 579, mod < 600, mod < 720], [0, 1, 2], 3)


def robust(R, row, ctrl):
    """뒤에서 살아남은 후보의 견고성: 시각 칸 통제, 첫 3봉 빼기, 이웃 문턱, (ptouch면) 창 길이, 시장 위치 통제, 결과 건수."""
    f, sp, yv = row.feat, row.split, row.y
    sgn = np.sign(row["앞_dd"])
    say(f"\n[견고성] {f} {sp} → {yv} (앞 방향 {'+' if sgn > 0 else '-'})")
    R = R.copy()
    b = tod_bucket(R["mod"].values)
    for j in range(1, 4):
        R[f"tb{j}"] = (b == j).astype(float)
    tb = ["tb1", "tb2", "tb3"]
    ctrl = [c for c in ctrl if c != "open30"]          # open30은 시각 칸과 겹친다
    f1 = R[R.half == "앞"]
    qs = [0.5, 0.6, 2 / 3, 0.75, 0.8] if sp == "상위1/3" else [0.2, 0.25, 1 / 3, 0.4, 0.5]
    feats = [f] + ([f + "_s", f + "_l"] if f == "ptouch" else [])
    for ff in feats:
        for q in qs:
            th = f1[ff].quantile(q)
            parts = []
            for h in ["앞", "뒤"]:
                x = R[R.half == h]
                v = x[ff].values
                fin = np.isfinite(v)
                a = (v > th) if sp == "상위1/3" else (v <= th)
                a = a & fin
                x = x[fin].assign(A=a[fin].astype(float))
                aa = a[fin]
                for y in ["ret", "fast"]:
                    m, t, nd = day_t(x.dayc.values, x[y].values.astype(float), aa)
                    parts.append(f"{h} {y} {pct(m)} t={t:+.2f}")
            say(f"  {ff} 문턱 앞 {q:.0%}분위({th:+.3f}) A비중 앞 {(f1[ff] > th).mean() if sp == '상위1/3' else (f1[ff] <= th).mean():.0%}: " + " | ".join(parts))
    th = (row.q33, row.q67)
    for h in ["앞", "뒤"]:
        x = R[R.half == h]
        a, fin = masks(x[f], th, sp)
        x = x[fin].assign(A=a[fin].astype(float))
        aa = a[fin]
        for y in ["ret", "fast"]:
            r1 = fe_ols(x, y, "A", ctrl + tb)
            r2 = fe_ols(x, y, "A", ctrl + tb + ["qqq_dev", "smh_dev", "pdev"])
            late = x["mod"].values >= 579
            m3, t3, n3 = day_t(x.dayc.values[late], x[y].values.astype(float)[late], aa[late])
            m4, t4, n4 = day_t(x.dayc.values[~late], x[y].values.astype(float)[~late], aa[~late])
            say(f"  {h} {y}: 시각4칸 통제 {pct(r1[0])} t={r1[1]:+.2f} | +QQQ·SMH·동종 위치 통제 {pct(r2[0])} t={r2[1]:+.2f}"
                f" | 첫 3봉(09:30~09:38) 빼고 {pct(m3)} t={t3:+.2f}({n3}일, A {aa[late].sum()}·B {(~aa[late]).sum()}건)"
                f" | 첫 3봉만 {pct(m4)} t={t4:+.2f}({n4}일, A {aa[~late].sum()}·B {(~aa[~late]).sum()}건)")
        say(f"  {h} 건수: A {aa.sum()}건 중 {yv}=1 {int(x[yv][aa].sum()) if yv == 'fast' else '-'} | B {(~aa).sum()}건 중 {int(x[yv][~aa].sum()) if yv == 'fast' else '-'}"
            f" | 수익 합 A {x.ret[aa].sum():+.2f} B {x.ret[~aa].sum():+.2f} | 첫 3봉 비중 A {(~late[aa]).mean():.0%} B {(~late[~aa]).mean():.0%}")


def masks(x, thr, split):
    q33, q67 = thr
    v = x.values
    fin = np.isfinite(v)
    a = (v <= q33) if split == "하위1/3" else (v > q67)
    return a & fin, fin


def run_tests(df, feats, outs, scope, thr=None, front_only_thr=True):
    """df: 'half','dayc','ret' 등. 문턱은 앞 절반에서. 반환: 시험 행 DataFrame."""
    rows = []
    f1 = df[df.half == "앞"]
    for f in feats:
        th = thr[f] if thr is not None else tuple(f1[f].quantile([1 / 3, 2 / 3]).values)
        for sp in SPLITS:
            for yv in outs:
                rec = dict(scope=scope, feat=f, split=sp, y=yv, q33=th[0], q67=th[1])
                for h in ["앞", "뒤"]:
                    x = df[df.half == h]
                    a, fin = masks(x[f], th, sp)
                    y = x[yv].values.astype(float)
                    dc = x.dayc.values
                    m, t, nd = day_t(dc[fin], y[fin], a[fin])
                    pm, ptt = pooled_t(dc[fin], y[fin], a[fin])
                    rec.update({f"{h}_tp": ptt,f"{h}_nA": int(a.sum()), f"{h}_nB": int((fin & ~a).sum()),
                                f"{h}_mA": np.nanmean(y[a]) if a.any() else np.nan,
                                f"{h}_mB": np.nanmean(y[fin & ~a]) if (fin & ~a).any() else np.nan,
                                f"{h}_dd": m, f"{h}_t": t, f"{h}_nd": nd})
                rows.append(rec)
    return pd.DataFrame(rows)


def pick3(tab):
    """앞 절반 |t| 최대인 특성 3개(특성마다 가장 강한 분할·결과 하나)."""
    x = tab.dropna(subset=["앞_t"]).copy()
    x["abs"] = x["앞_t"].abs()
    x = x.sort_values("abs", ascending=False)
    return x.drop_duplicates("feat").head(3)


def drop_one(x, f, th, sp, yv, sgn, col):
    best = None
    for v in x[col].unique():
        k = (x[col] != v).values
        xx = x[k]
        a, fin = masks(xx[f], th, sp)
        m, t, nd = day_t(xx.dayc.values[fin], xx[yv].values.astype(float)[fin], a[fin])
        if np.isnan(m):
            continue
        if best is None or sgn * m < sgn * best[1]:
            best = (v, m, t)
    return best


def confirm(df, row, label, ctrl):
    f, sp, yv = row.feat, row.split, row.y
    th = (row.q33, row.q67)
    sgn = np.sign(row["앞_dd"])
    say(f"\n[{label}] {f} {sp} (문턱 q33={th[0]:+.4f}, q67={th[1]:+.4f}) | 고른 결과 {yv} | 앞 방향 {'+' if sgn > 0 else '-'}")
    for h in ["앞", "뒤"]:
        x = df[df.half == h].copy()
        a, fin = masks(x[f], th, sp)
        x["A"] = a.astype(float)
        x = x[fin]
        a = a[fin]
        for y in ["ret", "fast"]:
            m, t, nd = day_t(x.dayc.values, x[y].values.astype(float), a)
            b, bt, g = fe_ols(x, y, "A", ctrl)
            o = x.open30.values.astype(bool)
            m1, t1, n1 = day_t(x.dayc.values[o], x[y].values.astype(float)[o], a[o])
            m2, t2, n2 = day_t(x.dayc.values[~o], x[y].values.astype(float)[~o], a[~o])
            pm, ptt = pooled_t(x.dayc.values, x[y].values.astype(float), a)
            ds = drop_one(x, f, th, sp, y, sgn, "sym")
            dd_ = drop_one(x, f, th, sp, y, sgn, "day")
            say(f"  {h} {y}: A {a.sum()}건 {pct(x[y][a].mean())} vs B {(~a).sum()}건 {pct(x[y][~a].mean())} | 날짜차 {pct(m)} t={t:+.2f}({nd}일)"
                f" | 합동(날짜군집SE) t={ptt:+.2f} | 통제회귀 {pct(b)} t={bt:+.2f} | 개장30분 안 {pct(m1)} t={t1:+.2f}({n1}) 밖 {pct(m2)} t={t2:+.2f}({n2})"
                f" | -{ds[0] if ds else '-'} {pct(ds[1]) if ds else 'nan'} t={ds[2] if ds else np.nan:+.2f}"
                f" | -{dd_[0] if dd_ else '-'} {pct(dd_[1]) if dd_ else 'nan'} t={dd_[2] if dd_ else np.nan:+.2f}")
        say(f"     {h} A의 개장30분 비중 {x.open30[a].mean():.0%} vs B {x.open30[~a].mean():.0%} | ivol 중앙 A {x.ivol[a].median():.4f} B {x.ivol[~a].median():.4f}"
            f" | atr_d 중앙 A {x.atr_d[a].median():.3f} B {x.atr_d[~a].median():.3f}")


def placebo(df, outs, nf, iters, nsplit=2):
    """무작위 특성 nf개로 같은 절차: 앞에서 |t| 상위 3개 → 뒤에서 같은 방향 t. 반환: 뒤 t 배열(고른 순서 1~3위)."""
    f1, f2 = df[df.half == "앞"], df[df.half == "뒤"]
    d1, d2 = f1.dayc.values, f2.dayc.values
    Y1 = {y: f1[y].values.astype(float) for y in outs}
    Y2 = {y: f2[y].values.astype(float) for y in outs}
    res = []
    for _ in range(iters):
        z1 = RNG.standard_normal((nf, len(f1)))
        z2 = RNG.standard_normal((nf, len(f2)))
        cand = []
        for j in range(nf):
            q33, q67 = np.quantile(z1[j], [1 / 3, 2 / 3])
            for sp in range(nsplit):
                a1 = z1[j] <= q33 if sp == 0 else z1[j] > q67
                a2 = z2[j] <= q33 if sp == 0 else z2[j] > q67
                for y in outs:
                    m, t, _ = day_t(d1, Y1[y], a1)
                    if np.isfinite(t):
                        cand.append((abs(t), j, np.sign(m), a2, y))
        cand.sort(key=lambda c: -c[0])
        seen, picked = set(), []
        for c in cand:
            if c[1] in seen:
                continue
            seen.add(c[1])
            m, t, _ = day_t(d2, Y2[c[4]], c[3])
            picked.append(c[2] * t if np.isfinite(t) else np.nan)
            if len(picked) == 3:
                break
        res.append(picked)
    return np.array(res)


# ---------------- 사건: 테마 급등 되돌림 ----------------
def events(D, T, DL, G, grid, P, syms):
    rows = []
    for s in syms:
        if s not in D:
            continue
        d, Tt = D[s], T[s]
        pcs = DL[s].pc
        c, o, mod, dts = d.c.values, d.o.values, d["mod"].values, d.date.values
        day_idx = pd.Series(np.arange(len(d))).groupby(dts).agg(["min", "max"])
        for dy, (lo, hi) in day_idx.iterrows():
            pc = pcs.get(dy, np.nan)
            if not np.isfinite(pc) or lo < 61:
                continue
            H = np.maximum.accumulate(c[lo:hi + 1])
            reg = (mod[lo:hi + 1] >= 570) & (mod[lo:hi + 1] < 960)
            ri = np.where(reg)[0]
            if len(ri) == 0:
                continue
            last = lo + ri[-1]
            full = mod[last] >= 954
            for j in ri:
                k = lo + j
                Hk = H[j]
                if Hk >= pc * 1.05 and c[k] <= pc + 0.5 * (Hk - pc):
                    if k + 1 > last:
                        break
                    ent = o[k + 1]
                    ey1 = c[last] / ent - 1 if full else np.nan
                    mx = c[k + 1:last + 1].max()
                    ey2 = float(mx >= ent + 0.5 * (Hk - ent)) if full else np.nan
                    f = feats_at(s, k + 1, D, T, DL, G, grid, P)
                    rows.append(dict(sym=s, day=dy, t=d.index[k + 1], ent=ent, runup=Hk / pc - 1, drop=(Hk - c[k]) / pc,
                                     ey1=ey1, ey2=ey2, full=full, **f))
                    break
    return pd.DataFrame(rows)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    tr = pd.read_csv(os.path.join(RES, "trades.csv"))
    tr["ts"] = pd.to_datetime(tr.t.str[:19]).dt.tz_localize("America/New_York")
    tr["day"] = pd.to_datetime(tr.day).dt.date
    tr["fast"] = ((tr.why == "RSI") & (tr.bars <= 20)).astype(float)
    syms = sorted(set(tr.sym))
    have_files = {f[3:-8] for f in os.listdir(RAW) if f.startswith("1m_")}
    P, lvl, sub = peer_map(set(syms) | (have_files - {"QQQ", "SMH", "IGV", "ARKG", "XBI", "IWM", "SPY"}))
    need = sorted(set(syms) | {p for s in syms for p in P.get(s, [])} | {"QQQ", "SMH"})
    D, T, DL, G, grid = build_all(RAW, 3, 60, 14, need)

    # ---- 거래 특성 ----
    F = []
    for r in tr.itertuples():
        d = D[r.sym]
        pos = d.index.get_loc(r.ts)
        F.append(feats_at(r.sym, pos, D, T, DL, G, grid, P))
    F = pd.DataFrame(F, index=tr.index)
    X = pd.concat([tr, F], axis=1)
    X["mod"] = X.ts.dt.hour * 60 + X.ts.dt.minute
    X["latr"] = np.log(X.atr_d)
    X["livol"] = np.log(X.ivol.clip(lower=1e-5))
    X["clean"], X["prv1m"] = clean_flag(X, RAW)
    days = sorted(X.day.unique())
    X["dayc"] = X.day.map({d_: i for i, d_ in enumerate(days)})
    X.drop(columns=["ts"]).to_csv(os.path.join(RES, "pat_peers_trades.csv"), index=False, encoding="utf-8-sig")

    say("# 이격매수매도 — 동종·시장 동반 모양으로 반등 미리 알기 (pat_peers, 2026-10-10)")
    say("## 정의(시험 전 고정)")
    say("Y1 = 거래 수익 ret(지금 매도 규칙). Y2 = fast = 매수 뒤 20봉(60분) 안 RSI 매도(why=RSI & bars<=20).")
    say("사건 E(테마 급등 되돌림, 정규장, 종목-일 첫 1회): 04:00부터 봉 k까지 3분봉 종가 최고 H_k >= 전일 정규장 종가 pc x1.05 이고"
        " 정규장 봉 k 종가가 처음으로 c_k <= pc + 0.5(H_k - pc). 진입 = 봉 k+1 시가."
        " EY1 = 그날 정규장 마지막 종가/진입 - 1. EY2 = 진입 뒤 그날 정규장 종가 최고 >= 진입 + 0.5(H_k - 진입) (되돌린 폭 절반 회복).")
    say("특성은 봉 i 시가에 아는 값(자기 i-1, 동종·ETF는 i 시작 전 끝난 봉, 일봉은 전날까지). 동종 = AI_UNIVERSE 소분류(2개 미만이면 대분류).")
    nl = pd.Series({s: lvl.get(s, "없음") for s in syms}).value_counts()
    say(f"동종 수준: {dict(nl)} | 예: LITE {P.get('LITE')} / CRCL {P.get('CRCL')} / MRVL({lvl.get('MRVL')}) {len(P.get('MRVL', []))}종목")

    R = X[X.ses == "정규"].copy()
    say(f"\n## 데이터: 거래 {len(X)}건, 정규장 {len(R)}건(앞 {sum(R.half == '앞')}·뒤 {sum(R.half == '뒤')}), 날짜 앞 {R[R.half == '앞'].day.nunique()}·뒤 {R[R.half == '뒤'].day.nunique()}")
    say(f"정규장 Y2 기저율: 앞 {R[R.half == '앞'].fast.mean():.1%}({int(R[R.half == '앞'].fast.sum())}건) 뒤 {R[R.half == '뒤'].fast.mean():.1%}({int(R[R.half == '뒤'].fast.sum())}건) — 드물어 Y2 시험은 힘이 약하다.")
    say(f"정규장 개장 30분 매수 비중: {R.open30.mean():.0%}. 특성 결측(동종 없음 등): {R.pdev.isna().sum()}건.")
    cl = X[X.ses != "정규"]
    for s_ in ["프리", "애프터"]:
        g = cl[cl.ses == s_]
        say(f"{s_}: {len(g)}건 중 깨끗한 체결 {int((g.clean == 1).sum())}건({(g.clean == 1).mean():.0%}), 평균 깨끗 {pct(g.ret[g.clean == 1].mean())} vs 꼬리 {pct(g.ret[g.clean == 0].mean())}")
    say(f"정규장 꼬리 체결 비율 {(R.clean == 0).mean():.0%} (정규장은 거래량이 실제라 그대로 쓴다)")

    # 특성 간 상관(앞 정규장)
    say("\n특성 간 순위상관(정규장 전체, |r|>=0.5만): " + ", ".join(
        f"{a}~{b} {v:+.2f}" for (a, b), v in R[TRADE_F].corr("spearman").stack().items() if a < b and abs(v) >= 0.5))

    # ---- 1) 주 시험: 정규장 ----
    ctrl = ["open30", "latr", "livol"]
    tab = run_tests(R, TRADE_F, ["ret", "fast"], "정규")
    say(f"\n## 1) 정규장 시험: 특성 {len(TRADE_F)} x 분할 2 x 결과 2 = {len(tab)}조합 (문턱·방향은 앞 절반)")
    say("특성 | 분할 | 결과 | 앞 A/B건 | 앞 A평균 vs B평균 | 앞 날짜차 t | 뒤 A/B건 | 뒤 날짜차 t(같은 문턱)")
    for _, q in tab.iterrows():
        say(f"{q.feat} | {q.split} | {q.y} | {int(q['앞_nA'])}/{int(q['앞_nB'])} | {pct(q['앞_mA'])} vs {pct(q['앞_mB'])} | "
            f"{pct(q['앞_dd'])} t={q['앞_t']:+.2f} | {int(q['뒤_nA'])}/{int(q['뒤_nB'])} | {pct(q['뒤_dd'])} t={q['뒤_t']:+.2f}")
    na2 = (tab["앞_t"].abs() >= 2).sum()
    say(f"앞 |t|>=2: {na2}/{len(tab)} (우연 기대 약 {0.05 * len(tab):.1f}개, 날짜 10개라 t 꼬리가 두꺼워 더 많을 수 있음)")
    same = ((np.sign(tab["앞_t"]) == np.sign(tab["뒤_t"])) & tab["뒤_t"].notna()).mean()
    say(f"앞·뒤 부호 일치 비율 {same:.0%} (우연 50%) | 앞 t와 뒤 t 상관 {tab['앞_t'].corr(tab['뒤_t']):+.2f}")

    pk = pick3(tab)
    say("\n## 2) 앞에서 고른 후보 3개 → 뒤 확인 (통제: 날짜FE 회귀[open30·log atr_d·log ivol], 개장 30분 안/밖, 상위 1종목·1일 빼기)")
    for i, (_, row) in enumerate(pk.iterrows()):
        confirm(R, row, f"후보 {i + 1}", ctrl)

    for _, row in pk.iterrows():
        if np.sign(row["앞_dd"]) * row["뒤_t"] >= 2:
            robust(R, row, ctrl)

    # 무작위 특성 같은 절차
    pl = placebo(R, ["ret", "fast"], len(TRADE_F), 300)
    obs = []
    for _, row in pk.iterrows():
        obs.append(np.sign(row["앞_dd"]) * row["뒤_t"])
    say(f"\n무작위 특성 {len(TRADE_F)}개 x 300회 같은 절차: 고른 3개의 뒤 t(같은 방향) 중앙 {np.nanmedian(pl):+.2f}, "
        f"95% 위 {np.nanpercentile(pl, 95):+.2f}, 뒤 t>=2 비율 {np.nanmean(pl >= 2):.1%}")
    for i, o in enumerate(obs):
        say(f"  후보 {i + 1} 뒤 t(같은 방향) {o:+.2f} → 무작위에서 이 이상 비율 {np.nanmean(pl[:, i] >= o) if np.isfinite(o) else np.nan:.1%}")

    # ---- 3) 프리·애프터 깨끗한 체결 보조 ----
    C = X[(X.ses != "정규") & (X.clean == 1)].copy()
    tabc =run_tests(C, TRADE_F, ["ret", "fast"], "프리애프터깨끗", thr=None)
    say(f"\n## 3) 프리·애프터 깨끗한 체결 보조({len(C)}건, 문턱은 이 묶음의 앞 절반): {len(tabc)}조합")
    for _, row in pk.iterrows():
        rr = tabc[(tabc.feat == row.feat) & (tabc.split == row.split)]
        for _, q in rr.iterrows():
            say(f"  {q.feat} {q.split} {q.y}: 앞 {pct(q['앞_dd'])} t={q['앞_t']:+.2f} | 뒤 {pct(q['뒤_dd'])} t={q['뒤_t']:+.2f}")
    say(f"  프리·애프터 깨끗 앞 |t|>=2: {(tabc['앞_t'].abs() >= 2).sum()}/{len(tabc)}, 앞→뒤 부호 일치 {((np.sign(tabc['앞_t']) == np.sign(tabc['뒤_t'])) & tabc['뒤_t'].notna()).mean():.0%}")
    # 직전 연구 sdev_q(혼자 빠짐) 정규·깨끗 다시
    say("\n## 4) 혼자 빠짐 sdev_q 다시(정규장 / 프리·애프터 깨끗한 체결, 문턱 각 묶음 앞 절반)")
    for nm, tb in [("정규", tab), ("프리애프터깨끗", tabc)]:
        for _, q in tb[tb.feat == "sdev_q"].iterrows():
            say(f"  {nm} {q.split} {q.y}: 앞 {pct(q['앞_dd'])} t={q['앞_t']:+.2f} | 뒤 {pct(q['뒤_dd'])} t={q['뒤_t']:+.2f}")
    # 앞 단계에서 나온 rel_dev(동종 대비 혼자 빠짐)도 같이
    for nm, tb in [("정규", tab), ("프리애프터깨끗", tabc)]:
        for _, q in tb[tb.feat == "rel_dev"].iterrows():
            say(f"  (동종 대비) rel_dev {nm} {q.split} {q.y}: 앞 {pct(q['앞_dd'])} t={q['앞_t']:+.2f} | 뒤 {pct(q['뒤_dd'])} t={q['뒤_t']:+.2f}")

    # ---- 5) 앞 기간 5분봉 근사 보조 ----
    try:
        import stop_lab
        E5 = stop_lab.run(stop_lab.load("early"), None, "early")
        E5 = E5[E5.ses == "정규"].copy()
        E5["fast"] = ((E5.why == "RSI") & (E5.bars <= 12)).astype(float)
        need5 = sorted(set(E5.sym) | {p for s in set(E5.sym) for p in P.get(s, [])} | {"QQQ", "SMH"})
        D5, T5, DL5, G5, grid5 = build_all(RAW_OLD, 5, 36, 9, need5)
        F5 = []
        for r in E5.itertuples():
            if r.sym not in D5:
                F5.append({})
                continue
            F5.append(feats_at(r.sym, int(r.ei), D5, T5, DL5, G5, grid5, P))
        E5 = pd.concat([E5.reset_index(drop=True), pd.DataFrame(F5)], axis=1)
        dd5 = sorted(E5.day.unique())
        E5["dayc"] = E5.day.map({d_: i for i, d_ in enumerate(dd5)})
        E5["half"] = "뒤"
        E5["sdev_q"] = E5.own_dev - E5.qqq_dev
        E5["latr"] = np.log(E5.atr_d)
        E5["livol"] = np.log(E5.ivol.clip(lower=1e-5))
        say(f"\n## 5) 앞 기간 5분봉 근사(7/14~9/8, EMA36·RSI9, Y2=12봉 안 RSI 매도) 정규장 {len(E5)}건·{len(dd5)}일 — 단위가 달라 보조")
        for _, row in pk.iterrows():
            f, sp = row.feat, row.split
            sgn = np.sign(row["앞_dd"])
            for lab, th in [("3분봉 앞 문턱", (row.q33, row.q67)), ("5분봉 자체 삼분위", tuple(E5[f].quantile([1 / 3, 2 / 3]).values))]:
                a, fin = masks(E5[f], th, sp)
                for y in ["ret", "fast"]:
                    m, t, nd = day_t(E5.dayc.values[fin], E5[y].values.astype(float)[fin], a[fin])
                    E5["A"] = a.astype(float)
                    b, bt, _ = fe_ols(E5[fin], y, "A", ctrl)
                    say(f"  {f} {sp} [{lab}] {y}: A {a.sum()}건 {pct(E5[y][a].mean())} vs B {(fin & ~a).sum()}건 {pct(E5[y][fin & ~a].mean())}"
                        f" | 날짜차 {pct(m)} t={t:+.2f}({nd}일) 앞 방향 기준 {sgn * t:+.2f} | 통제회귀 t={bt:+.2f}")
    except Exception as ex:  # 보조 확인 실패는 주 결과를 막지 않는다
        say(f"\n## 5) 앞 기간 5분봉 보조 실패: {ex!r}")

    # ---- 6) 사건 연구: 테마 급등 되돌림 ----
    EV = events(D, T, DL, G, grid, P, syms)
    EV["half"] = np.where(EV.day < MID, "앞", np.where(EV.day <= date(2026, 10, 8), "뒤", "10/9"))
    EV["latr"] = np.log(EV.atr_d)
    EV["livol"] = np.log(EV.ivol.clip(lower=1e-5))
    EV.to_csv(os.path.join(RES, "pat_peers_events.csv"), index=False, encoding="utf-8-sig")
    E = EV[EV.half.isin(["앞", "뒤"]) & EV.full].copy()
    edays = sorted(E.day.unique())
    E["dayc"] = E.day.map({d_: i for i, d_ in enumerate(edays)})
    E = E.rename(columns={"ey1": "ret", "ey2": "fast"})
    say(f"\n## 6) 사건 E(테마 급등 되돌림): {len(E)}건(앞 {sum(E.half == '앞')}·뒤 {sum(E.half == '뒤')}), 종목 {E.sym.nunique()}, 날짜 {len(edays)}")
    say(f"EY1 평균 앞 {pct(E[E.half == '앞'].ret.mean())} 뒤 {pct(E[E.half == '뒤'].ret.mean())} | EY2 비율 앞 {E[E.half == '앞'].fast.mean():.0%} 뒤 {E[E.half == '뒤'].fast.mean():.0%}")
    say(f"날짜별 건수 상위: {dict(E.day.value_counts().head(6))}")
    tabe = run_tests(E, EVENT_F, ["ret", "fast"], "사건E")
    say(f"사건 시험: 특성 {len(EVENT_F)} x 분할 2 x 결과 2 = {len(tabe)}조합 (ret=EY1, fast=EY2)")
    for r in tabe.itertuples():
        q = tabe.loc[r.Index]
        say(f"  {q.feat} | {q.split} | {q.y} | 앞 {int(q['앞_nA'])}/{int(q['앞_nB'])} {pct(q['앞_dd'])} t={q['앞_t']:+.2f}({int(q['앞_nd'])}일) 합동 t={q['앞_tp']:+.2f}"
            f" | 뒤 {int(q['뒤_nA'])}/{int(q['뒤_nB'])} {pct(q['뒤_dd'])} t={q['뒤_t']:+.2f}({int(q['뒤_nd'])}일) 합동 t={q['뒤_tp']:+.2f}")
    say(f"사건 앞 |t|>=2: {(tabe['앞_t'].abs() >= 2).sum()}/{len(tabe)}, 앞→뒤 부호 일치 {((np.sign(tabe['앞_t']) == np.sign(tabe['뒤_t'])) & tabe['뒤_t'].notna()).mean():.0%}")
    pke = pick3(tabe)
    E["open30"] = E.open30.astype(float)
    for i, (_, row) in enumerate(pke.iterrows()):
        confirm(E, row, f"사건 후보 {i + 1}", ctrl)
    ple = placebo(E, ["ret", "fast"], len(EVENT_F), 300)
    say(f"사건 무작위 특성 같은 절차: 고른 3개 뒤 t 95% 위 {np.nanpercentile(ple, 95):+.2f}, 뒤 t>=2 비율 {np.nanmean(ple >= 2):.1%}")
    for i, (_, row) in enumerate(pke.iterrows()):
        o = np.sign(row["앞_dd"]) * row["뒤_t"]
        say(f"  사건 후보 {i + 1} 뒤 t(같은 방향) {o:+.2f} → 무작위에서 이 이상 {np.nanmean(ple[:, i] >= o) if np.isfinite(o) else np.nan:.1%}")

    # ---- 7) 10/9 사례(설명용, 1분봉이 10:18 ET에서 끝나 결과는 그때까지만) ----
    say("\n## 7) 10/9 사례(설명용, 데이터 10:18 ET까지)")
    import build
    for s in ["CRCL", "MRVL", "LITE", "COHR"]:
        d, dead_of, hot = build.arrays(s)
        tt = [x for x in build.sim(d, dead_of, hot, 0.02) if d.date.iloc[x[0]] == date(2026, 10, 9)]
        # 마지막 거래가 아직 열려 있으면 sim이 내보내지 않으므로 직접 찾는다
        lo = np.where(d.date.values == date(2026, 10, 9))[0]
        ent_pos = None
        if tt:
            ent_pos = [x[0] for x in tt if d.ses.iloc[x[0]] == "정규"]
            ent_pos = ent_pos[0] if ent_pos else None
        if ent_pos is None:
            for i in lo:
                if d.ses.iloc[i] == "정규" and d.l.iloc[i] <= d.ema.iloc[i - 1] * 0.98 and not hot[i]:
                    ent_pos = i
                    break
        if ent_pos is None:
            say(f"  {s}: 10/9 정규장 매수 없음")
            continue
        D_ = D[s]
        pos = D_.index.get_loc(d.index[ent_pos])
        f = feats_at(s, pos, D, T, DL, G, grid, P)
        ent = min(d.o.iloc[ent_pos], d.ema.iloc[ent_pos - 1] * 0.98)
        aft = d.iloc[ent_pos:lo[-1] + 1]
        rx = aft.r.values
        hit = np.where(rx[:-1] > 70)[0]
        say(f"  {s} 매수 {d.index[ent_pos].strftime('%H:%M')} {ent:.2f} | 10:18까지 최고 {pct(aft.h.max() / ent - 1)} 마지막 {pct(aft.c.iloc[-1] / ent - 1)}"
            f" RSI>70 {'있음 ' + aft.index[hit[0]].strftime('%H:%M') if len(hit) else '없음'} | "
            + " ".join(f"{k}={f.get(k, np.nan):+.3f}" for k in TRADE_F if k != "sdev_q")
            + f" | 앞 문턱 대비: " + ", ".join(f"{k} {'하위' if f.get(k, np.nan) <= tab[tab.feat == k].q33.iloc[0] else ('상위' if f.get(k, np.nan) > tab[tab.feat == k].q67.iloc[0] else '중간')}"
                                         for k in pk.feat))
    e9 = EV[EV.day == date(2026, 10, 9)]
    say(f"  10/9 사건 E 해당: {', '.join(e9.sym.tolist())}")
    for s in ["LITE", "COHR"]:
        x = e9[e9.sym == s]
        if len(x):
            r = x.iloc[0]
            say(f"  {s} 사건 진입 {r.t.strftime('%H:%M')} 상승폭 {pct(r.runup)} | " + " ".join(f"{k}={r[k]:+.3f}" for k in EVENT_F)
                + " | 앞 문턱 대비: " + ", ".join(f"{k} {'하위' if r[k] <= tabe[tabe.feat == k].q33.iloc[0] else ('상위' if r[k] > tabe[tabe.feat == k].q67.iloc[0] else '중간')}" for k in pke.feat))
    # 과거 LITE·COHR 사건
    lc = E[E.sym.isin(["LITE", "COHR"])][["sym", "day", "runup", "rel_hi", "rel_give", "lead5", "ret", "fast"]]
    say("  9/10~10/8 LITE·COHR 사건: " + ("; ".join(f"{r.sym} {r.day} 상승 {pct(r.runup)} rel_hi {r.rel_hi:+.3f} lead5 {r.lead5:+.3f} EY1 {pct(r.ret)} EY2 {int(r.fast)}" for r in lc.itertuples()) or "없음"))

    # ---- 저장 ----
    allt = pd.concat([tab, tabc, tabe], ignore_index=True)
    allt.to_csv(os.path.join(RES, "pat_peers_tests.csv"), index=False, encoding="utf-8-sig")
    say(f"\n시험 조합 합계: 정규 {len(tab)} + 프리·애프터 깨끗 {len(tabc)} + 사건 {len(tabe)} = {len(allt)} (뒤 확인은 정규 3 + 사건 3, 5분봉 보조 3)")
    with open(os.path.join(RES, "pat_peers.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
