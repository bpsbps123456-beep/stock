"""8단계 반박 검증 V_RECON (2026-10-12): 갈래 RECON 대표 후보를 깨려고 시도한다.

후보: 재구성 F&G(z점수판) ≤10 첫날 → 126거래일 보유, S&P500 3배(UPRO 대리).
독립 재계산: p8_RECON.py·trade_lab·stock_lab·harness·lev5를 import하지 않는다. parquet·json 자료와 fg_data(자료 읽기)만 쓴다.
정의(과제 문장 그대로 새로 짬)
  C1 ^GSPC/125일선−1, C2 −^VIX/50일선, C3 S&P 20일 수익 − 국채(^TNX, 듀레이션 7: −7×20일 금리 변화 + 20일 이자),
  C4 VWEHX 20일 총수익 − 5년 듀레이션 국채 20일 수익(둘 다 하루 늦춤), C5 −CBOE 전체 풋콜 5일 평균(2020-09~ CNN put_call),
  C6 RSP−SPY 20일 로그수익 차(RSP 전 ^NYA−^GSPC). 요소마다 252일 z(그날 포함, 최소 126일) → ±2 자름 → 50+25z → 요소 3개 이상인 날 평균.
  사건: 값 ≤10 첫날(20 위로 올라간 뒤에만 새 사건). 그날 종가 확인 → 다음 날 종가 매수 → 126거래일 뒤 종가 매도.
  3배: 일일 3×수익 − 2×^IRX/252 − (0.95%+추가 1.2%)/252. 1배 이음: ^GSPC→SPY(1993-02-01 뒤), ^NDX→QQQ(1999-03-11 뒤), ^SOX→SOXX(2001-07-16 뒤).
  아무 날: 같은 창의 모든 매수일(다음 날 종가) 평균. 고르기 창(1990~2010)은 매도까지 창 안, 확인 창(2011~2025)은 10/9까지 자료로 매도.
실행: PYTHONIOENCODING=utf-8 py p8_V_RECON.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import fg_data  # 자료 읽기만

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RESULTS = HERE.parent / "results"
ID = "p8_V_RECON"

# ============================================================================ 사전 고정 상수(확인 구간을 본 뒤 바꾸지 않는다)
START, END = "1985-01-01", "2026-10-09"
BASE = dict(T=10.0, REARM=10.0, H=126, ZW=252, CLIP=2.0, MINC=3, LAG=1)
NEIGH = {"T": [6.0, 8.0, 10.0, 12.0, 14.0], "REARM": [6.0, 8.0, 10.0, 12.0, 14.0], "H": [84, 105, 126, 147, 168],
         "ZW": [126, 189, 252, 315, 378], "CLIP": [1.5, 1.75, 2.0, 2.25, 2.5], "MINC": [1, 2, 3, 4, 5]}
JOINT_T, JOINT_H = NEIGH["T"], NEIGH["H"]          # T×H 25칸
FEE3, EXTRA = 0.0095, {"GSPC": 0.012, "NDX": 0.013, "SOX": 0.015}
EXTRA_GRID = [0.006, 0.009, 0.012, 0.015, 0.018]
SLIP, EXTRA_DD = 0.005, 0.20
N_PLA, MIN_SHIFT, N_RAND, N_RAND_BIG, SEED = 20, 63, 20, 10000, 20261012
DIP_DD = 0.10                                       # '하락장 아무 날' 대조: S&P 252일 고점 대비 −10% 이하인 날
SIMPLE = {"S&P 고점 −20% 첫날(−10% 위 회복 뒤 재무장)": ("dd", 0.20, 0.10),
          "VIX ≥40 첫날(30 아래 뒤 재무장)": ("vix", 40.0, 30.0)}
# RECON 고르기 절차 재현용(그 갈래 상수와 같은 값, 새로 짬)
PROC_SIGS, PROC_E1, PROC_E2, PROC_H, PROC_MIN = ("R252", "R504", "RZ"), (10, 20, 25), (0, 10), (21, 63, 126, 252), 8

WIN = {"rsel": ("1990-01-01", "2010-12-31"), "rchk": ("2011-01-01", "2025-12-31"),
       "csel": ("2011-01-01", "2018-12-31"), "cchk": ("2019-01-01", "2025-12-31"), "c_all": ("2011-01-01", "2025-12-31"),
       "2026앞": ("2026-01-02", "2026-05-29"), "2026뒤": ("2026-06-01", END), "AI쇼크": ("2026-06-22", "2026-07-29"),
       "2025": ("2025-01-02", "2025-12-31")}
ERAS = {"1990": ("1990-01-02", "1990-12-31"), "1998": ("1998-07-01", "1998-12-31"), "2000~02": ("2000-03-01", "2002-10-31"),
        "2008": ("2007-10-01", "2009-03-31"), "2011": ("2011-04-01", "2011-12-30"), "2015~16": ("2015-07-01", "2016-06-30"),
        "2018 4분기": ("2018-10-01", "2018-12-31"), "2020": ("2020-01-02", "2020-12-31"), "2022": ("2022-01-03", "2022-12-30"),
        "2025": ("2025-01-02", "2025-12-31"), "2026앞": WIN["2026앞"], "2026뒤": WIN["2026뒤"], "AI쇼크": WIN["AI쇼크"]}
SELCAP = {"rsel", "csel"}
TRUNC = {"2026앞", "2026뒤", "AI쇼크"}


# ============================================================================ 자료
def load(f):
    d = pd.read_parquet(CACHE / f)
    d["date"] = d["date"].astype(str)
    return d


P7, YF, LEV = load("p7_idx.parquet"), load("p8_RECON_yf.parquet"), load("p5_lev_long.parquet")


def ser(df, sym, col="Adj Close"):
    g = df[df.sym == sym].sort_values("date").drop_duplicates("date", keep="last").dropna(subset=["Close"])
    g = g[g.Close > 0]
    return pd.Series(g[col].to_numpy(float), index=g.date.to_numpy())


CAL = ser(P7, "^GSPC").index
CAL = np.asarray(CAL[(CAL >= START) & (CAL <= END)]).astype(str)
N = len(CAL)


def al(s, limit=5):
    s = s[~s.index.duplicated(keep="last")]
    u = sorted(set(s.index) | set(CAL))
    s = s.reindex(u)
    return (s.ffill(limit=limit) if limit else s).reindex(CAL)


def pos(day, right=False):
    return int(np.searchsorted(CAL, day, side="right" if right else "left"))


def wix(w):
    a, z = WIN[w] if w in WIN else ERAS[w]
    return pos(a), pos(z, True) - 1


# ============================================================================ 재구성 지수
def comps():
    spx, vix = al(ser(P7, "^GSPC")), al(ser(P7, "^VIX"))
    tnx = al(ser(YF, "^TNX", "Close")) / 100.0
    C = pd.DataFrame(index=CAL)
    C["C1"] = spx / spx.rolling(125, min_periods=125).mean() - 1
    C["C2"] = -vix / vix.rolling(50, min_periods=50).mean()
    b10 = -7.0 * (tnx - tnx.shift(20)) + tnx.shift(20) * 20 / 252
    C["C3"] = spx / spx.shift(20) - 1 - b10
    hy = al(ser(YF, "VWEHX")).shift(1)
    b5 = (-5.0 * (tnx - tnx.shift(20)) + tnx.shift(20) * 20 / 252).shift(1)
    C["C4"] = hy / hy.shift(20) - 1 - b5
    pc = load("p8_RECON_pc.parquet")
    rank = {"pcratioarchive.csv": 0, "totalpcarchive.csv": 1, "totalpc.csv": 2}
    pc["o"] = pc.src.map(rank)
    pc = pc.sort_values(["date", "o"]).drop_duplicates("date", keep="last").set_index("date")["pc"]
    pc = pc[(pc > 0.1) & (pc < 5)].sort_index()
    pc5 = pc.rolling(5, min_periods=5).mean()
    cpc = fg_data.components()["put_call"].dropna()
    pcs = pd.concat([pc5[pc5.index < "2020-09-01"], cpc[cpc.index >= "2020-09-01"]])
    C["C5"] = -al(pcs, 3)
    rsp, spy, nya = al(ser(YF, "RSP")), al(ser(P7, "SPY")), al(ser(YF, "^NYA"))
    e1 = np.log(rsp / rsp.shift(20)) - np.log(spy / spy.shift(20))
    e0 = np.log(nya / nya.shift(20)) - np.log(spx / spx.shift(20))
    C["C6"] = e1.where(e1.notna(), e0)
    return C, pcs, pc


def zidx(C, W=252, clip=2.0, minc=3, drop=None):
    P = {}
    for k in C.columns:
        if k == drop:
            continue
        s = C[k]
        mu = s.rolling(W, min_periods=W // 2).mean()
        sd = s.rolling(W, min_periods=W // 2).std()
        P[k] = (50 + 25 * ((s - mu) / sd).clip(-clip, clip)).where(s.notna())
    P = pd.DataFrame(P)
    return P.mean(axis=1).where(P.notna().sum(axis=1) >= minc).to_numpy(float)


def pidx(C, W):
    P = {}
    for k in C.columns:
        s = C[k]
        r = s.rolling(W, min_periods=W // 2).rank(method="average", pct=True)
        n = s.rolling(W, min_periods=W // 2).count()
        P[k] = (100 * (r - 0.5 / n)).where(s.notna())
    P = pd.DataFrame(P)
    return P.mean(axis=1).where(P.notna().sum(axis=1) >= 3).to_numpy(float)


# ============================================================================ 대상
def splice(idx_sym, etf, cut, etf_df=P7):
    a = al(ser(P7, idx_sym)).to_numpy()
    b = al(ser(etf_df, etf)).to_numpy()
    ra = np.r_[np.nan, a[1:] / a[:-1] - 1]
    rb = np.r_[np.nan, b[1:] / b[:-1] - 1]
    r = np.where(np.arange(N) <= pos(cut), ra, rb)
    i0 = int(np.flatnonzero(np.isfinite(a))[0])
    c = np.full(N, np.nan)
    c[i0] = 100.0
    c[i0 + 1:] = 100 * np.cumprod(1 + np.nan_to_num(r[i0 + 1:]))
    return c


IRX = al(ser(P7, "^IRX", "Close") / 100, 10).fillna(0.0).to_numpy()


def lev3(c1, extra):
    r = np.r_[np.nan, c1[1:] / c1[:-1] - 1]
    r3 = 3 * r - 2 * IRX / 252 - (FEE3 + extra) / 252
    r3 = np.where(np.isfinite(r3), np.maximum(r3, -0.999), np.nan)
    i0 = int(np.flatnonzero(np.isfinite(c1))[0])
    c = np.full(N, np.nan)
    c[i0] = 100.0
    c[i0 + 1:] = 100 * np.cumprod(1 + np.nan_to_num(r3[i0 + 1:]))
    return c


def upro_cost(c1):
    up = al(ser(LEV, "UPRO")).to_numpy()
    s0 = lev3(c1, 0.0)
    i0 = int(np.flatnonzero(np.isfinite(up))[0]) + 1
    i1 = N - 1
    return float(-np.log((up[i1] / up[i0]) / (s0[i1] / s0[i0])) / ((i1 - i0) / 252)), str(CAL[i0])


# ============================================================================ 사건·평가
def events(F, T, rearm, mode="le"):
    """mode le: F≤T 첫날(F>T+rearm 뒤 재무장). ge: F≥T 첫날(F<rearm 뒤 재무장, rearm은 절대값)."""
    out, armed_off = [], True
    for i, v in enumerate(F):
        if not np.isfinite(v):
            continue
        if mode == "le":
            if armed_off and v <= T:
                out.append(i)
                armed_off = False
            elif not armed_off and v > T + rearm:
                armed_off = True
        else:
            if armed_off and v >= T:
                out.append(i)
                armed_off = False
            elif not armed_off and v < rearm:
                armed_off = True
    return np.array(out, dtype=int)


def events_e2(F, delta):
    """≤20에 있다가 20+δ 위로 올라선 첫날."""
    out, inside = [], False
    for i, v in enumerate(F):
        if not np.isfinite(v):
            continue
        if v <= 20:
            inside = True
        elif inside and v > 20 + delta:
            out.append(i)
            inside = False
    return np.array(out, dtype=int)


def evaluate(c, ev, H, w, lag=1, slip=0.0, entry=None, keep_list=False):
    """사건형 평가. entry: 시가 체결용 진입 가격 배열(signal day s → 가격), 없으면 c[s+lag]."""
    a, z = wix(w)
    i0 = int(np.flatnonzero(np.isfinite(c))[0])
    a = max(a, i0)
    if z < a:
        return {"n": 0}
    cap = z if w in SELCAP else N - 1
    g = (1 - slip) ** 2
    s = ev[(ev >= a) & (ev <= z)]
    days = np.arange(a, z + 1)

    def px_in(si):
        return entry[si] if entry is not None else c[si + lag]

    if w in TRUNC:
        rows = []
        for si in s:
            b = si + lag
            q = min(b + H, N - 1)
            if q <= b:
                continue
            L = q - b
            bb = days + lag
            bb = bb[bb + L <= N - 1]
            base = np.mean(c[bb + L] / (entry[bb - lag] if entry is not None else c[bb])) * g - 1
            ret = c[q] / px_in(si) * g - 1
            lo = np.min(c[b:q + 1])
            rows.append((si, b, q, ret, base, lo <= 0.8 * c[b]))
        if not rows:
            return {"n": 0}
        sidx = np.array([r[0] for r in rows]); b = np.array([r[1] for r in rows]); q = np.array([r[2] for r in rows])
        ret = np.array([r[3] for r in rows]); base_v = np.array([r[4] for r in rows]); hit = np.array([r[5] for r in rows])
        bmean, bhit = float(base_v.mean()), np.nan
        ex = ret - base_v
    else:
        s = s[(s + lag + H <= cap)]
        if len(s) == 0:
            return {"n": 0}
        sidx = s
        b = s + lag
        q = b + H
        ret = c[q] / np.array([px_in(si) for si in s]) * g - 1
        hit = np.array([np.min(c[bi:qi + 1]) <= (1 - EXTRA_DD) * c[bi] for bi, qi in zip(b, q)])
        bd = days[days + lag + H <= cap]
        bp = entry[bd] if entry is not None else c[bd + lag]
        br = c[bd + lag + H] / bp * g - 1
        ok = np.isfinite(br)
        br = br[ok]
        bhit_arr = np.array([np.min(c[x + lag:x + lag + H + 1]) <= (1 - EXTRA_DD) * c[x + lag] for x in bd[ok]])
        bmean, bhit = float(br.mean()), float(bhit_arr.mean())
        ex = ret - bmean
    nind, endp = 0, -1
    for bi, qi in zip(b, q):
        if bi > endp:
            nind += 1
            endp = qi
    k = len(ret)
    sd = float(np.std(ex, ddof=1)) if k > 1 else np.nan
    t = float(np.mean(ex) / (sd / np.sqrt(nind))) if k > 1 and sd > 0 else np.nan
    out = {"n": int(k), "n_ind": int(nind), "mean": float(ret.mean()), "ex": float(ex.mean()), "base": bmean,
           "hit20": float(hit.mean()), "base_hit20": bhit, "t": t, "pos": float((ret > 0).mean())}
    if keep_list:
        out["list"] = [{"신호일": str(CAL[si]), "매수일": str(CAL[bi]), "매도일": str(CAL[qi]), "수익": float(r), "초과": float(e),
                        "보유중최저": float(np.min(c[bi:qi + 1]) / c[bi] - 1)} for si, bi, qi, r, e in zip(sidx, b, q, ret, ex)]
    return out


# ============================================================================ 시점 감사
def timing_audit(C, pcraw, RZ):
    r = al(ser(P7, "^GSPC")).pct_change()
    out = {}

    def corr3(x, nm):
        d = pd.DataFrame({"x": x, "r0": r, "rm1": r.shift(1), "rp1": r.shift(-1)}).dropna()
        out[nm] = {"그날 수익": float(d.x.corr(d.r0)), "전날 수익": float(d.x.corr(d.rm1)), "다음 날 수익": float(d.x.corr(d.rp1)),
                   "일수": int(len(d))}

    fg = fg_data.fg()
    fgc = fg.reindex(CAL)
    dfg = fgc.diff()
    dfg_s = pd.Series(dfg.to_numpy(), index=CAL)
    corr3(dfg_s.where(np.asarray(CAL < "2021-02-01")), "실제 F&G 하루 변화(보관본 2011~2021-01)")
    corr3(dfg_s.where(np.asarray(CAL >= "2021-02-01")), "실제 F&G 하루 변화(CNN 공식 2021-02~)")
    pcd = al(pcraw, 0)
    corr3(pd.Series(pcd.to_numpy(), index=CAL).diff(), "CBOE 풋콜 하루 변화(1995~2019)")
    cpc = al(fg_data.components()["put_call"].dropna(), 0)
    cpc = cpc.where(np.asarray(CAL >= "2021-02-01"))
    corr3(cpc.diff(), "CNN put_call(5일 평균) 하루 변화(2021-02~)")
    hy = al(ser(YF, "VWEHX"), 0).pct_change()
    corr3(hy, "VWEHX 기준가 하루 수익(하루 늦추기 전)")
    corr3(pd.Series(RZ, index=CAL).diff(), "재구성 z점수판 하루 변화")
    return out


# ============================================================================ 실행
def main():
    t0 = time.time()
    J = {"id": ID, "상수": {"BASE": BASE, "NEIGH": NEIGH, "EXTRA_GRID": EXTRA_GRID, "SLIP": SLIP, "N_PLA": N_PLA, "N_RAND": N_RAND,
                           "N_RAND_BIG": N_RAND_BIG, "SEED": SEED, "DIP_DD": DIP_DD, "SIMPLE": {k: list(v) for k, v in SIMPLE.items()}}}
    C, pcs, pcraw = comps()
    RZ = zidx(C)
    R252, R504 = pidx(C, 252), pidx(C, 504)
    # 갈래 RECON 지수와 대조(읽기만)
    ref = pd.read_parquet(CACHE / "p8_RECON_index.parquet").reindex(CAL)
    J["지수대조"] = {k: {"상관": float(pd.Series(v).corr(pd.Series(ref[k].to_numpy()))),
                       "최대차": float(np.nanmax(np.abs(v - ref[k].to_numpy())))} for k, v in (("RZ", RZ), ("R252", R252), ("R504", R504))}
    s1 = {"GSPC": splice("^GSPC", "SPY", "1993-02-01"), "NDX": splice("^NDX", "QQQ", "1999-03-11"),
          "SOX": splice("^SOX", "SOXX", "2001-07-16", LEV)}
    up_cost, up0 = upro_cost(s1["GSPC"])
    J["UPRO추가비용_잰값"] = {"연": up_cost, "시작": up0}
    TG = {f"{k}1": v for k, v in s1.items()} | {f"{k}3": lev3(v, EXTRA[k]) for k, v in s1.items()}
    c3 = TG["GSPC3"]
    print("준비", f"{time.time() - t0:.0f}s", J["지수대조"], J["UPRO추가비용_잰값"], flush=True)

    ev0 = events(RZ, BASE["T"], BASE["REARM"])
    J["사건_1990~"] = [str(CAL[i]) for i in ev0 if CAL[i] >= "1990"]

    # ---------------- 1) 재현
    R = {}
    for tg in TG:
        for w in ("rsel", "rchk", "2025", "2026앞", "2026뒤", "AI쇼크"):
            R[f"{tg}|{w}"] = evaluate(TG[tg], ev0, 126, w)
    R["GSPC3|rsel|목록"] = evaluate(c3, ev0, 126, "rsel", keep_list=True)["list"]
    R["GSPC3|rchk|목록"] = evaluate(c3, ev0, 126, "rchk", keep_list=True)["list"]
    R["GSPC1|rchk|목록"] = evaluate(TG["GSPC1"], ev0, 126, "rchk", keep_list=True)["list"]
    R["GSPC1|rsel|목록"] = evaluate(TG["GSPC1"], ev0, 126, "rsel", keep_list=True)["list"]
    J["재현"] = R
    print("재현 GSPC3", {w: round(R[f'GSPC3|{w}'].get('ex', np.nan), 4) for w in ('rsel', 'rchk', '2025')}, flush=True)

    # 집중도: 2025 빼기, 중앙값, 겹친 사건 묶음(에피소드)
    conc = {}
    for w in ("rsel", "rchk"):
        L = R[f"GSPC3|{w}|목록"]
        ex = np.array([x["초과"] for x in L])
        conc[w] = {"평균": float(ex.mean()), "중앙값": float(np.median(ex)), "플러스비율": float((ex > 0).mean())}
        # 에피소드: 보유 기간이 겹치면 한 묶음
        eps, cur, endq = [], [], ""
        for x in L:
            if cur and x["매수일"] <= endq:
                cur.append(x)
                endq = max(endq, x["매도일"])
            else:
                if cur:
                    eps.append(cur)
                cur, endq = [x], x["매도일"]
        eps.append(cur)
        em = np.array([np.mean([y["초과"] for y in e]) for e in eps])
        conc[w]["에피소드"] = [{"첫날": e[0]["신호일"], "건수": len(e), "평균초과": float(np.mean([y["초과"] for y in e]))} for e in eps]
        conc[w]["에피소드평균"] = float(em.mean())
        conc[w]["에피소드_한개빼기_최소"] = float(min(np.delete(em, i).mean() for i in range(len(em))))
        conc[w]["에피소드_한개빼기_최대기여뺌"] = float(np.delete(em, int(np.argmax(em))).mean())
        rng = np.random.default_rng(SEED)
        bs = [rng.choice(em, len(em), replace=True).mean() for _ in range(5000)]
        conc[w]["에피소드_부트스트랩_5~95%"] = [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]
        conc[w]["에피소드_부트스트랩_양수확률"] = float(np.mean(np.array(bs) > 0))
    L = R["GSPC3|rchk|목록"]
    ex_wo = [x["초과"] for x in L if not x["신호일"].startswith("2025")]
    conc["rchk_2025빼면"] = {"n": len(ex_wo), "평균초과": float(np.mean(ex_wo))}
    J["집중도"] = conc

    # ---------------- 2) 시점 감사 + 체결 민감도
    J["시점_상관"] = timing_audit(C, pcraw, RZ)
    S = {}
    for w in ("rsel", "rchk", "2025"):
        S[f"미끄러짐0.5%|{w}"] = evaluate(c3, ev0, 126, w, slip=SLIP)
        S[f"하루늦춤(t+2 종가)|{w}"] = evaluate(c3, ev0, 126, w, lag=2)
        S[f"이틀늦춤(t+3 종가)|{w}"] = evaluate(c3, ev0, 126, w, lag=3)
    # 다음 날 시가 체결(SPY 시가, 1993-02 뒤만): 3배 진입가 = c3[s]×(1+3×(시가_{s+1}/종가_s−1))
    spy_o = al(ser(P7, "SPY", "Open") * ser(P7, "SPY", "Adj Close") / ser(P7, "SPY", "Close"), 0).to_numpy()
    spy_c = al(ser(P7, "SPY"), 0).to_numpy()
    on = np.full(N, np.nan)
    on[:-1] = spy_o[1:] / spy_c[:-1] - 1
    ent3 = c3 * (1 + 3 * on)
    ent1 = TG["GSPC1"] * (1 + on)
    ok_open = np.asarray(CAL >= "1993-02-01")
    for w in ("rchk", "2025"):
        S[f"다음날시가|{w}"] = evaluate(c3, ev0, 126 + 1, w, lag=0, entry=np.where(ok_open, ent3, np.nan))
        S[f"다음날시가_1배|{w}"] = evaluate(TG["GSPC1"], ev0, 126 + 1, w, lag=0, entry=np.where(ok_open, ent1, np.nan))
    for ex_ in EXTRA_GRID:
        cc = lev3(s1["GSPC"], ex_)
        S[f"추가비용{ex_:.1%}|rchk"] = evaluate(cc, ev0, 126, "rchk")
        S[f"추가비용{ex_:.1%}|rsel"] = evaluate(cc, ev0, 126, "rsel")
    J["민감도"] = S

    # ---------------- 3) 이웃(한 번에 한 파라미터 + T×H 25칸 + 요소 하나 빼기)
    NB = {}
    for p, vals in NEIGH.items():
        for v in vals:
            prm = dict(BASE)
            prm[p] = v
            F = RZ if p in ("T", "REARM", "H") else zidx(C, W=prm["ZW"], clip=prm["CLIP"], minc=prm["MINC"])
            ev = events(F, prm["T"], prm["REARM"])
            for tg in ("GSPC3", "GSPC1"):
                for w in ("rsel", "rchk"):
                    NB[f"{p}={v}|{tg}|{w}"] = evaluate(TG[tg], ev, prm["H"], w)
    for T_ in JOINT_T:
        ev = events(RZ, T_, BASE["REARM"])
        for H_ in JOINT_H:
            for w in ("rsel", "rchk"):
                NB[f"TxH|{T_}|{H_}|{w}"] = evaluate(c3, ev, H_, w)
    for d in C.columns:
        F = zidx(C, drop=d)
        ev = events(F, BASE["T"], BASE["REARM"])
        for w in ("rsel", "rchk"):
            NB[f"빼기{d}|GSPC3|{w}"] = evaluate(c3, ev, 126, w)
    J["이웃"] = NB
    print("이웃", f"{time.time() - t0:.0f}s", flush=True)

    # ---------------- 4) 위약
    PL = {}
    s0 = int(np.flatnonzero(np.isfinite(RZ))[0])
    nv = N - s0
    step = nv // (N_PLA + 1)
    rng = np.random.default_rng(SEED)
    shifts_a = [k * step for k in range(1, N_PLA + 1)]
    shifts_b = sorted(int(x) for x in rng.integers(MIN_SHIFT, nv - MIN_SHIFT, N_PLA))

    def roll(F, k):
        o = F.copy()
        o[s0:] = np.roll(F[s0:], k)
        return o

    for nm, shs in (("원형밀기_등간격", shifts_a), ("원형밀기_무작위", shifts_b)):
        for w in ("rsel", "rchk"):
            real = R[f"GSPC3|{w}"]["ex"]
            arr = []
            for k in shs:
                e = evaluate(c3, events(roll(RZ, k), BASE["T"], BASE["REARM"]), 126, w)
                arr.append(e["ex"] if e["n"] else np.nan)
            arr = np.array(arr)
            PL[f"{nm}|{w}"] = {"순위": int(1 + np.sum(arr[np.isfinite(arr)] > real)), "판수": int(np.isfinite(arr).sum() + 1),
                               "위약중앙": float(np.nanmedian(arr)), "위약최대": float(np.nanmax(arr)), "밀기": shs}
    # 무작위 날짜(같은 건수)
    for w in ("rsel", "rchk"):
        a, z = wix(w)
        cap = z if w in SELCAP else N - 1
        elig = np.arange(a, z + 1)
        elig = elig[elig + 1 + 126 <= cap]
        real = R[f"GSPC3|{w}"]
        k = real["n"]
        base = real["base"]
        fr = c3[elig + 1 + 126] / c3[elig + 1] - 1
        rr = np.random.default_rng(SEED + (1 if w == "rchk" else 0))
        small = np.array([fr[rr.choice(len(elig), k, replace=False)].mean() - base for _ in range(N_RAND)])
        big = np.array([fr[rr.choice(len(elig), k, replace=False)].mean() - base for _ in range(N_RAND_BIG)])
        # 하락장 아무 날(S&P 252일 고점 대비 −10% 이하)
        spx = al(ser(P7, "^GSPC")).to_numpy()
        hi = pd.Series(spx).rolling(252, min_periods=126).max().to_numpy()
        dip = elig[(spx[elig] / hi[elig] - 1) <= -DIP_DD]
        frd = c3[dip + 1 + 126] / c3[dip + 1] - 1
        bigd = np.array([frd[rr.choice(len(dip), k, replace=False)].mean() - base for _ in range(N_RAND_BIG)])
        PL[f"무작위날짜|{w}"] = {"순위": int(1 + np.sum(small > real["ex"])), "판수": N_RAND + 1, "위약중앙": float(np.median(small)),
                              "p값_1만판": float(np.mean(big >= real["ex"])), "하락장날_p값_1만판": float(np.mean(bigd >= real["ex"])),
                              "하락장날_평균초과": float(frd.mean() - base), "하락장날_수": int(len(dip))}
    # RECON 고르기 절차 전체(재구성 3판 × E1 3 × E2 2 × 보유 4 = 60 후보, 고르기 사건 8건 이상, 최대 → 확인), 3배 3대상 + 대표 고르기
    SIG = {"R252": R252, "R504": R504, "RZ": RZ}
    s0s = {k: int(np.flatnonzero(np.isfinite(v))[0]) for k, v in SIG.items()}

    def proc(sigs, tg):
        best = None
        cands = []
        for sg, F in sigs.items():
            evs = {f"E1_{t}": events(F, t, 10) for t in PROC_E1} | {f"E2_{d}": events_e2(F, d) for d in PROC_E2}
            for rn, ev in evs.items():
                for H in PROC_H:
                    e = evaluate(TG[tg], ev, H, "rsel")
                    if e["n"] >= PROC_MIN:
                        cands.append((e["ex"], sg, rn, H, ev))
        if not cands:
            return None
        ex, sg, rn, H, ev = max(cands, key=lambda x: x[0])
        chk = evaluate(TG[tg], ev, H, "rchk")
        return {"sig": sg, "rule": rn, "H": H, "고르기": ex, "확인": chk.get("ex") if chk["n"] else None}

    real_proc = {tg: proc(SIG, tg) for tg in ("GSPC3", "NDX3", "SOX3")}
    pla_proc = {tg: [] for tg in real_proc}
    pla_rep = []
    for j in range(N_PLA):
        sj = {}
        for sg, F in SIG.items():
            nv_ = N - s0s[sg]
            st_ = nv_ // (N_PLA + 1)
            o = F.copy()
            o[s0s[sg]:] = np.roll(F[s0s[sg]:], (j + 1) * st_)
            sj[sg] = o
        vals = {}
        for tg in real_proc:
            p = proc(sj, tg)
            vals[tg] = None if p is None else p["확인"]
            pla_proc[tg].append(vals[tg])
        pos_ = [v for v in vals.values() if v is not None and v > 0]
        pla_rep.append(max(pos_) if pos_ else None)
        print(f"  절차 위약 {j + 1}/{N_PLA}", f"{time.time() - t0:.0f}s", flush=True)
    PR = {}
    for tg, p in real_proc.items():
        arr = np.array([v for v in pla_proc[tg] if v is not None])
        PR[tg] = {"실제": p, "위약확인": pla_proc[tg],
                  "순위": None if p is None or p["확인"] is None else int(1 + np.sum(arr > p["확인"])), "판수": int(len(arr) + 1)}
    rp = [v for v in pla_rep if v is not None]
    real_rep = max(v["확인"] for v in [p for p in real_proc.values() if p] if v["확인"] is not None and v["확인"] > 0)
    PR["대표고르기까지"] = {"실제": real_rep, "위약": pla_rep, "순위": int(1 + sum(v > real_rep for v in rp)),
                       "판수": N_PLA + 1, "위약에서_통과후보없음": int(sum(v is None for v in pla_rep))}
    PL["절차"] = PR
    J["위약"] = PL
    print("위약", f"{time.time() - t0:.0f}s", flush=True)

    # ---------------- 단순 대안(같은 틀): S&P −20% 첫날, VIX≥40 첫날
    spx = al(ser(P7, "^GSPC")).to_numpy()
    hi = pd.Series(spx).rolling(252, min_periods=126).max().to_numpy()
    dd = spx / hi - 1
    vix = al(ser(P7, "^VIX")).to_numpy()
    ALT = {}
    for nm, (kind, x, y) in SIMPLE.items():
        if kind == "dd":
            # 고점 대비 ≤−20% 첫날, −10% 위로 회복 뒤 재무장
            out, armed = [], True
            for i in range(N):
                if not np.isfinite(dd[i]):
                    continue
                if armed and dd[i] <= -x:
                    out.append(i)
                    armed = False
                elif not armed and dd[i] > -y:
                    armed = True
            ev = np.array(out, dtype=int)
        else:
            ev = events(vix, x, y, mode="ge")
        for w in ("rsel", "rchk", "2025", "2026앞", "2026뒤"):
            ALT[f"{nm}|{w}"] = evaluate(c3, ev, 126, w)
        ALT[f"{nm}|날짜"] = [str(CAL[i]) for i in ev if CAL[i] >= "1990"]
    J["단순대안"] = ALT

    # ---------------- 5) 시대별
    ER = {}
    rz_s = pd.Series(RZ, index=CAL)
    fg = fg_data.fg().reindex(CAL)
    evc = events(fg.to_numpy(float), BASE["T"], BASE["REARM"])
    J["실제F&G_같은규칙_사건"] = [str(CAL[i]) for i in evc]
    for e in ERAS:
        a, z = wix(e)
        seg = rz_s.iloc[a:z + 1]
        d = {"재구성최저": float(seg.min()), "최저일": str(seg.idxmin()), "사건": [str(CAL[i]) for i in ev0 if a <= i <= z]}
        for tg in ("GSPC1", "GSPC3", "NDX3"):
            d[tg] = evaluate(TG[tg], ev0, 126, e, keep_list=True)
        fseg = fg.iloc[a:z + 1].dropna()
        if len(fseg):
            d["실제최저"] = float(fseg.min())
            d["실제F&G_GSPC3"] = evaluate(c3, evc, 126, e, keep_list=True)
        ER[e] = d
    J["시대별"] = ER
    CF = {}
    for w in ("csel", "cchk", "c_all", "2025", "2026앞", "2026뒤"):
        CF[w] = evaluate(c3, evc, 126, w, keep_list=True)
    J["실제F&G_같은규칙"] = CF

    J["실행초"] = round(time.time() - t0, 1)
    pd.DataFrame({"RZ": RZ, "R252": R252, "R504": R504}, index=CAL).to_parquet(CACHE / f"{ID}_index.parquet")
    (RESULTS / f"{ID}.json").write_text(json.dumps(clean(J), ensure_ascii=False, indent=1), encoding="utf-8")
    print("저장", J["실행초"], "s")
    return J


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [clean(v) for v in list(o)]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


if __name__ == "__main__":
    main()
