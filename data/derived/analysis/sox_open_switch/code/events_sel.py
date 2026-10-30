"""BRIEF3 사건 분석(sel 2026-01-02~06-30만, 잠금 켜지 않음). 사용: py events_sel.py → results/events_sel.json, results/events_sel_events.csv

사건: 결정 봉 09:30~10:44(마감 10:45까지)에서 종목 s(SOXL 또는 SOXS)의 1분 종가가
      L = 09:30 봉부터 지금 봉까지 s 저가의 최저 × (1 + x%) 이상이 된 봉. x = 2, 2.5, 3, 3.5, 4, 5.
  - first   : 종목·날·x마다 첫 사건.
  - re      : 청산 리셋 없이 '최저 갱신 뒤 다시'. 앞 사건 뒤 L(09:30부터 최저)이 앞 사건 때 L보다 낮아진 뒤 다시 조건을 채운 봉.
  - re_reset: (참고, 표 by_x만) 앞 사건 다음 봉부터 L을 새로 잰 두 번째 이후 사건(H0의 '청산 뒤 L 다시 재기'에 가까운 읽기).
체결(lib.Engine과 같은 규칙, engine_check로 대조): 신호 봉 i 마감 → 봉 i+1 종가 × (1 + 반 호가) 매수(사람 1분 지연).
  목표 +h%: 봉 i+2부터 고가 ≥ 진입가 × (1+h) × 1.0002면 진입가 × (1+h)에 지정가 청산(호가 비용 없음).
  창 끝: w90 10:59 봉 종가, std 09:59 봉 종가 × (1 − 반 호가). 손절 없음. std는 결정 봉 마감 09:45까지 사건만.
다리: opp = 반대 종목 매수, fol = 오른 종목 매수(추종), rand = 같은 분 두 종목 평균, diff = opp − rand = (opp − fol)/2.
손익 단위 %(1.0 = 1%). 구간: 날 묶음 부트스트랩 10,000회, seed 20261029, 2.5·97.5 백분위.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import (Data, Engine, Order, RTH0, TOUCH, INST, bucket, hm, idx, load_costs)  # noqa: E402

T0 = time.time()
SEED, NB = 20261029, 10000
XS = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
XS_SUP = [6.0, 7.0, 8.0, 10.0]   # 보충(결과 본 뒤 추가): re_reset 뒤 사건이 세션 최저 대비 6~12% 늘어난 자리라 first를 더 큰 x로도 잰다. by_x만
HS = [0.5, 1.0, 1.5]
WIN = {"w90": (idx("11:00"), idx("10:44")),      # (cut, 마지막 결정 봉)
       "std": (idx("10:00"), idx("09:44"))}
RSI_N = 14
OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))

D = Data()
SEL = D.days_in("sel")
COST = load_costs()
A = D.A
OTHER = {"SOXL": "SOXS", "SOXS": "SOXL"}
SIGN = {"SOXL": 1, "SOXS": -1}
LAST_DEC = WIN["w90"][1]


def xkey(x):
    return f"{x:g}"


# ------------------------------------------------------------------ 지표(신호 봉 마감 때 아는 값만)

def wilder_rsi(c, n=RSI_N):
    """t2_check.wilder_rsi와 같은 식(첫 n개 단순 평균으로 씨앗, 이후 Wilder)."""
    c = np.asarray(c, float)
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return out
    dl = np.diff(c)
    g, l = np.clip(dl, 0, None), np.clip(-dl, 0, None)
    ag = pd.Series(np.r_[g[:n].mean(), g[n:]]).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    al = pd.Series(np.r_[l[:n].mean(), l[n:]]).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100.0 - 100.0 / (1.0 + ag / al)
    out[n:] = np.where(al == 0, 100.0, r)
    return out


def chain_days():
    """연장 끔 지표를 날 넘어 잇는 사슬: sel 첫날 10거래일 전(2025-12, 예열용 가격만)부터 sel 끝까지."""
    return list(range(SEL[0] - 10, SEL[-1] + 1))


def rth_chain(sym):
    """정규장 봉만 날 넘어 이은 (날, 색인) 목록과 종가·거래량."""
    rows = []
    for d in chain_days():
        e = D.rth_end[d]
        has = D.has[sym][d, RTH0:e]
        ii = np.arange(RTH0, e)
        rows.append((np.full(len(ii), d), ii, has, A[sym]["Close"][d, RTH0:e], A[sym]["Volume"][d, RTH0:e]))
    dd = np.concatenate([r[0] for r in rows]); ii = np.concatenate([r[1] for r in rows])
    has = np.concatenate([r[2] for r in rows]); c = np.concatenate([r[3] for r in rows])
    v = np.concatenate([r[4] for r in rows])
    return dd, ii, has, c, v


IND = {}
for s in ["SOXL", "SOXS"]:
    dd, ii, has, c, v = rth_chain(s)
    # RSI: 실제 거래가 있는 봉만(앞 연구 기준 R과 같게), 빈 분은 앞 값 유지
    r = np.full(len(c), np.nan)
    r[has] = wilder_rsi(c[has])
    r = pd.Series(r).ffill().to_numpy()
    rsi = np.full((len(D.days), 960), np.nan)
    rsi[dd, ii] = r
    # 거래량 배수: 신호 봉 거래량 ÷ 직전 정규장 20봉 평균(날 넘어 이음)
    prev20 = pd.Series(v).shift(1).rolling(20).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        vr = v / prev20
    vratio = np.full((len(D.days), 960), np.nan)
    vratio[dd, ii] = vr
    # 세션 VWAP(09:30부터)
    pv = A[s]["VWAP"] * A[s]["Volume"]
    cpv = np.cumsum(np.where(np.arange(960) >= RTH0, pv, 0.0), axis=1)
    cv = np.cumsum(np.where(np.arange(960) >= RTH0, A[s]["Volume"], 0.0), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = cpv / cv
    IND[s] = {"rsi": rsi, "vratio": vratio, "vwap": vwap}

# 전날 SOXL 정규장 고저폭 3분위(sel 날들의 전날 값 분포로 나눔, 개장 전 아는 값)
_prev_rng = []
for d in SEL:
    p = D.daily.loc[("SOXL", D.days[d - 1])]
    _prev_rng.append(p["high"] / p["low"] - 1)
PR_Q = np.quantile(_prev_rng, [1 / 3, 2 / 3])
PREV_RNG = dict(zip(SEL, _prev_rng))


# ------------------------------------------------------------------ 사건 찾기

def find_events():
    rows = []
    for d in SEL:
        for s in ["SOXL", "SOXS"]:
            lo = A[s]["Low"][d, RTH0:LAST_DEC + 1]
            cl = A[s]["Close"][d, RTH0:LAST_DEC + 1]
            rmin = np.minimum.accumulate(lo)
            for x in XS + XS_SUP:
                cond = cl >= rmin * (1 + x / 100)
                if x in XS_SUP:
                    k = np.nonzero(cond)[0]
                    if len(k):
                        rows.append((d, s, x, "first", 0, RTH0 + k[0], rmin[k[0]]))
                    continue
                # first + re(최저 갱신 뒤 다시)
                k_prev, L_prev, n = None, None, 0
                for k in np.nonzero(cond)[0]:
                    if k_prev is None or rmin[k] < L_prev:
                        rows.append((d, s, x, "first" if n == 0 else "re", n, RTH0 + k, rmin[k]))
                        k_prev, L_prev, n = k, rmin[k], n + 1
                # re_reset: 앞 사건 다음 봉부터 L을 새로 잰다
                k0, n = 0, 0
                while k0 < len(cl):
                    rm = np.minimum.accumulate(lo[k0:])
                    hit = np.nonzero(cl[k0:] >= rm * (1 + x / 100))[0]
                    if not len(hit):
                        break
                    k = k0 + hit[0]
                    if n > 0:
                        rows.append((d, s, x, "re_reset", n, RTH0 + k, rm[hit[0]]))
                    n += 1
                    k0 = k + 1
    ev = pd.DataFrame(rows, columns=["d", "rise", "x", "kind", "seq", "i", "L"])
    return ev


def trade(sym, d, f, cut):
    """봉 f 종가 매수(+반 호가) → 목표 h(봉 f+1부터, 관통 1.0002) / 창 끝(cut−1 봉 종가 − 반 호가)."""
    e = A[sym]["Close"][d, f] * (1 + COST[sym][bucket(f)])
    hi = A[sym]["High"][d, f + 1:cut]
    lo = A[sym]["Low"][d, f + 1:cut]
    xend = A[sym]["Close"][d, cut - 1] * (1 - COST[sym][bucket(cut - 1)])
    end = (xend / e - 1) * 100
    o = {"mfe": (hi.max() / e - 1) * 100, "mae_end": (lo.min() / e - 1) * 100, "end": end}
    for h in HS:
        k = np.nonzero(hi >= e * (1 + h / 100) * TOUCH)[0]
        t = f"{int(h * 10):02d}"
        if len(k):
            k = k[0]
            o[f"hit{t}"], o[f"p{t}"], o[f"min{t}"] = 1.0, h, float(k + 1)
            o[f"maeh{t}"] = (lo[:k + 1].min() / e - 1) * 100
        else:
            o[f"hit{t}"], o[f"p{t}"], o[f"min{t}"], o[f"maeh{t}"] = 0.0, end, np.nan, np.nan
    return o


def bin_time(i):
    c = i + 1                                    # 봉 마감 색인
    return "09:31~09:45" if c <= idx("09:45") else ("09:46~10:15" if c <= idx("10:15") else "10:16~10:45")


def bins_speed(m):
    return "0~5분" if m <= 5 else "6~10분" if m <= 10 else "11~15분" if m <= 15 else "16~30분" if m <= 30 else "30분 초과"


def bins_v(v):
    return "nan" if not np.isfinite(v) else "<1" if v < 1 else "1~2" if v < 2 else "2~4" if v < 4 else "≥4"


def bins_rsi(r):
    return "nan" if not np.isfinite(r) else "<70" if r < 70 else "70~80" if r < 80 else "80~90" if r < 90 else "≥90"


def bins_vw(p):
    return "<0%" if p < 0 else "0~1%" if p < 1 else "1~2%" if p < 2 else "2~4%" if p < 4 else "≥4%"


def bins_gap(g):
    return "갭 하락(<−1%)" if g < -1 else "보합(±1%)" if g <= 1 else "갭 상승(>+1%)"


def build():
    ev = find_events()
    recs = []
    for r in ev.itertuples(index=False):
        d, s, x, i = r.d, r.rise, r.x, int(r.i)
        o = OTHER[s]
        lo = A[s]["Low"][d, RTH0:i + 1]
        jl = RTH0 + len(lo) - 1 - int(np.argmin(lo[::-1]))   # 지금 최저를 만든 마지막 봉
        cl = A[s]["Close"][d, i]
        prev_close = D.daily.loc[(s, D.days[d - 1]), "close"]
        gap = (A[s]["Open"][d, RTH0] / prev_close - 1) * 100
        if i < idx("09:34"):
            f5 = "미확정(신호 09:35 전)"
        else:
            f5 = "상승" if A[s]["Close"][d, idx("09:34")] >= A[s]["Open"][d, RTH0] else "하락"
        mk = 0
        for m in ["QQQ", "NVDA"]:
            if SIGN[s] > 0:
                ref = A[m]["Low"][d, RTH0:i + 1].min()
                mk += A[m]["Close"][d, i] >= ref * (1 + x / 300)
            else:
                ref = A[m]["High"][d, RTH0:i + 1].max()
                mk += A[m]["Close"][d, i] <= ref * (1 - x / 300)
        qq = A["QQQ"]["Close"][d, i] >= A["QQQ"]["Low"][d, RTH0:i + 1].min() * (1 + x / 300) if SIGN[s] > 0 else \
            A["QQQ"]["Close"][d, i] <= A["QQQ"]["High"][d, RTH0:i + 1].max() * (1 - x / 300)
        pr = PREV_RNG[d]
        base = {"day": str(D.days[d]), "d": d, "rise": s, "buy_opp": o, "x": x, "kind": r.kind, "seq": r.seq,
                "sig": hm(i + 1), "i": i, "L": r.L, "close": cl, "speed": i - jl,
                "vratio": IND[s]["vratio"][d, i], "rsi": IND[s]["rsi"][d, i],
                "vwap_pct": (cl / IND[s]["vwap"][d, i] - 1) * 100, "gap_pct": gap, "first5": f5,
                "mkt_n": int(mk), "qqq": bool(qq), "prev_rng": pr, "cum_pct": (cl / lo.min() - 1) * 100}
        for w, (cut, ld) in WIN.items():
            if i > ld:
                continue
            row = dict(base, win=w)
            to, tf = trade(o, d, i + 1, cut), trade(s, d, i + 1, cut)
            for k, v in to.items():
                row["opp_" + k] = v
            for k, v in tf.items():
                row["fol_" + k] = v
            row["win_ret_rise"] = (A[s]["Close"][d, cut - 1] / A[s]["Open"][d, RTH0] - 1) * 100
            recs.append(row)
    R = pd.DataFrame(recs)
    # 나눠 보기 칸(경계는 결과 보기 전 코드에 고정)
    R["b_time"] = [bin_time(i) for i in R["i"]]
    R["b_speed"] = [bins_speed(m) for m in R["speed"]]
    R["b_vratio"] = [bins_v(v) for v in R["vratio"]]
    R["b_rsi"] = [bins_rsi(v) for v in R["rsi"]]
    R["b_vwap"] = [bins_vw(v) for v in R["vwap_pct"]]
    R["b_gap"] = [bins_gap(v) for v in R["gap_pct"]]
    R["b_first5"] = R["first5"]
    R["b_mkt"] = R["mkt_n"].map({2: "QQQ·NVDA 둘 다", 1: "하나만", 0: "둘 다 아님(반도체만)"})
    R["b_side"] = R["rise"].map({"SOXL": "SOXL 오름 → SOXS 매수", "SOXS": "SOXS 오름 → SOXL 매수"})
    R["b_prevrng"] = ["하" if v <= PR_Q[0] else "중" if v <= PR_Q[1] else "상" for v in R["prev_rng"]]
    R["b_daydir"] = ["<0%" if v < 0 else "0~5%" if v < 5 else "≥5%" for v in R["win_ret_rise"]]
    # re_reset 전용 칸(결과 본 뒤 추가, 경계는 사건 분포만 보고 정함)
    R["b_seq"] = ["1번째" if q == 0 else "2번째" if q == 1 else "3번째 이상" for q in R["seq"]]
    R["b_cum"] = ["<5%" if v < 5 else "5~8%" if v < 8 else "8~11%" if v < 11 else "≥11%" for v in R["cum_pct"]]
    # rand·diff
    for t in ["05", "10", "15"]:
        R[f"rand_p{t}"] = (R[f"opp_p{t}"] + R[f"fol_p{t}"]) / 2
        R[f"rand_hit{t}"] = (R[f"opp_hit{t}"] + R[f"fol_hit{t}"]) / 2
        R[f"diff_p{t}"] = R[f"opp_p{t}"] - R[f"rand_p{t}"]
        R[f"diff_hit{t}"] = R[f"opp_hit{t}"] - R[f"rand_hit{t}"]
    for k in ["end", "mfe", "mae_end"]:
        R[f"rand_{k}"] = (R[f"opp_{k}"] + R[f"fol_{k}"]) / 2
        R[f"diff_{k}"] = R[f"opp_{k}"] - R[f"rand_{k}"]
    return R


SPLITS = [  # (열, 이름, 신호 때 아는가)
    ("b_time", "신호 시각대(봉 마감)", "신호 때 앎"),
    ("b_speed", "오름 속도: 세션 최저(09:30부터) 봉에서 신호 봉까지 분(re_reset도 세션 최저 기준, 다시 잰 L이 아님)", "신호 때 앎"),
    ("b_vratio", "신호 봉 거래량 ÷ 직전 정규장 20봉 평균(날 넘어 이음)", "신호 때 앎"),
    ("b_rsi", "오른 종목 1분 RSI14(연장 끔, 날 넘어 이음)", "신호 때 앎"),
    ("b_vwap", "오른 종목 종가의 세션 VWAP 위 거리", "신호 때 앎"),
    ("b_gap", "오른 종목 자기 갭(09:30 시가 ÷ 전날 정규장 종가)", "개장 때 앎"),
    ("b_first5", "오른 종목 첫 5분 방향(09:34 봉 종가 vs 09:30 시가)", "09:35부터 앎, 그 전 신호는 미확정"),
    ("b_mkt", "QQQ·NVDA가 같은 쪽으로 같은 기준 x/3% 이상 움직였나", "신호 때 앎"),
    ("b_side", "어느 쪽이 올랐나", "신호 때 앎"),
    ("b_prevrng", "전날 SOXL 고저폭 3분위(추가, BRIEF3 L3 축; 경계는 sel 날 분포)", "개장 전 앎(경계는 sel 분포라 표본 안)"),
    ("b_daydir", "오른 종목 창 수익(09:30 시가→창 끝 종가)", "사후(설명용, 매매에 못 씀)"),
]
SPLITS_RR = [  # re_reset에만 더하는 칸(결과 본 뒤 추가)
    ("b_seq", "그날 몇 번째 x% 오름인가(L을 앞 사건 다음 봉부터 다시 잼)", "신호 때 앎(결과 본 뒤 추가한 칸)"),
    ("b_cum", "세션 최저(09:30부터) 대비 누적 오름(신호 봉 종가)", "신호 때 앎(결과 본 뒤 추가한 칸)"),
]

FULL = []  # (이름, 분자 열, 분모 열 또는 None)
for leg in ["opp", "fol"]:
    for k in ["mfe", "end", "mae_end"]:
        FULL.append((f"{leg}_{k}", f"{leg}_{k}", None))
    for t in ["05", "10", "15"]:
        FULL += [(f"{leg}_hit{t}", f"{leg}_hit{t}", None), (f"{leg}_p{t}", f"{leg}_p{t}", None),
                 (f"{leg}_min{t}", f"{leg}_min{t}", f"{leg}_hit{t}"),
                 (f"{leg}_maeh{t}", f"{leg}_maeh{t}", f"{leg}_hit{t}")]
for k in ["mfe", "end", "mae_end"]:
    FULL += [(f"rand_{k}", f"rand_{k}", None), (f"diff_{k}", f"diff_{k}", None)]
for t in ["05", "10", "15"]:
    FULL += [(f"rand_hit{t}", f"rand_hit{t}", None), (f"rand_p{t}", f"rand_p{t}", None),
             (f"diff_hit{t}", f"diff_hit{t}", None), (f"diff_p{t}", f"diff_p{t}", None)]
FULL += [("diff_adj_p10", "diff_adj_p10", None), ("diff_adj_end", "diff_adj_end", None)]

SHORT_NAMES = ["opp_p05", "opp_p10", "opp_p15", "opp_hit05", "opp_hit10", "opp_hit15", "opp_end", "opp_mfe",
               "opp_mae_end", "opp_min10", "opp_maeh10", "fol_p10", "fol_hit10", "fol_end", "fol_mfe",
               "rand_p10", "rand_hit10", "rand_end", "rand_mfe",
               "diff_p05", "diff_p10", "diff_p15", "diff_hit10", "diff_end", "diff_mfe", "diff_adj_p10", "diff_adj_end"]
SHORT = [m for m in FULL if m[0] in SHORT_NAMES]
PCT_HIT = {"hit"}  # 적중률은 0~1 비율


def cmat(ncl):
    rng = np.random.default_rng(SEED)
    ix = rng.integers(0, ncl, (NB, ncl))
    flat = (np.arange(NB)[:, None] * ncl + ix).ravel()
    return np.bincount(flat, minlength=NB * ncl).reshape(NB, ncl).astype(float)


def _sums(df, inv, ncl, metrics, mask=None):
    Sn = np.zeros((ncl, len(metrics)))
    Sd = np.zeros((ncl, len(metrics)))
    mk = np.ones(len(df), bool) if mask is None else mask
    for j, (_, num, den) in enumerate(metrics):
        nv = df[num].to_numpy(float)
        dv = np.ones(len(df)) if den is None else df[den].to_numpy(float)
        ok = np.isfinite(nv) & mk
        np.add.at(Sn[:, j], inv[ok], (nv * dv)[ok])
        np.add.at(Sd[:, j], inv[ok], dv[ok])
    return Sn, Sd


def stats(df, metrics, C=None):
    """칸 하나: 사건 가중 평균과 날 묶음 부트스트랩 95% 구간. 반환 {이름: [평균, 하한, 상한]}, n, 날 수."""
    if len(df) == 0:
        return {"n": 0, "days": 0}
    u, inv = np.unique(df["d"].to_numpy(), return_inverse=True)
    ncl = len(u)
    C = cmat(ncl) if C is None else C
    Sn, Sd = _sums(df, inv, ncl, metrics)
    with np.errstate(divide="ignore", invalid="ignore"):
        pt = Sn.sum(0) / Sd.sum(0)
        bs = (C @ Sn) / (C @ Sd)
    out = {"n": int(len(df)), "days": int(ncl)}
    for j, (name, _, _) in enumerate(metrics):
        col = bs[:, j]
        if not np.isfinite(pt[j]):
            out[name] = None
            continue
        lo, hi = np.nanpercentile(col, [2.5, 97.5]) if np.isfinite(col).any() else (np.nan, np.nan)
        out[name] = [round(float(pt[j]), 4), round(float(lo), 4), round(float(hi), 4)]
        if name in ("diff_p10", "diff_end", "diff_adj_p10"):
            out[name + "_p_le0"] = round(float(np.nanmean(col <= 0)), 4)
    if "opp_min10" in [m[0] for m in metrics]:
        mm = df["opp_min10"].dropna()
        out["opp_min10_median"] = float(mm.median()) if len(mm) else None
    return out


def lift(df, mask, names=("diff_p10", "diff_end", "opp_p10", "diff_adj_p10")):
    """같은 칸 안에서 '이 조건' − '나머지' 차이(같은 날 다시 뽑기로 함께 부트스트랩)."""
    if mask.sum() == 0 or (~mask).sum() == 0:
        return None
    u, inv = np.unique(df["d"].to_numpy(), return_inverse=True)
    ncl = len(u)
    C = cmat(ncl)
    mets = [m for m in FULL if m[0] in names]
    Si, Di = _sums(df, inv, ncl, mets, mask)
    So, Do = _sums(df, inv, ncl, mets, ~mask)
    with np.errstate(divide="ignore", invalid="ignore"):
        pt = Si.sum(0) / Di.sum(0) - So.sum(0) / Do.sum(0)
        bs = (C @ Si) / (C @ Di) - (C @ So) / (C @ Do)
    out = {}
    for j, (name, _, _) in enumerate(mets):
        lo, hi = np.nanpercentile(bs[:, j], [2.5, 97.5])
        out[name] = [round(float(pt[j]), 4), round(float(lo), 4), round(float(hi), 4)]
    return out


def baseline_window_random():
    """창 무작위: sel 모든 날, 진입 가능한 모든 결정 분 × 두 종목, 같은 청산 규칙.
    함께 '시각 맞춘 방향 기준선' b(f) = 체결 분 f에서 (SOXS 매수 − SOXL 매수)/2의 sel 전체 날 평균(앞뒤 7분 이동 평균)을 만든다.
    사건의 diff에서 이 값을 빼면(diff_adj) sel 장초의 시각별 평균 방향(표본 안 값)을 걷어 낸 '되돌림 몫'이 남는다."""
    out, bmap = {}, {}
    for w, (cut, ld) in WIN.items():
        rows = []
        for d in SEL:
            for i in range(RTH0, ld + 1):
                for s in ["SOXL", "SOXS"]:
                    t = trade(s, d, i + 1, cut)
                    rows.append({"d": d, "sym": s, "f": i + 1, **{"opp_" + k: v for k, v in t.items()}})
        B = pd.DataFrame(rows)
        mets = [m for m in FULL if m[0].startswith("opp_")]
        st = stats(B, mets)
        out[w] = {k.replace("opp_", "any_"): v for k, v in st.items()}
        out[w]["note"] = "모든 결정 분(09:30 봉~마지막 결정 봉) × SOXL·SOXS 각각 매수, 분 가중 평균"
        for s in ["SOXL", "SOXS"]:
            st = stats(B[B.sym == s], [m for m in mets if m[0] in ("opp_p10", "opp_end", "opp_hit10", "opp_mfe")])
            out[w]["buy_" + s] = {k.replace("opp_", ""): v for k, v in st.items()}
        P = B.pivot_table(index=["d", "f"], columns="sym", values=["opp_p10", "opp_end"])
        dd = pd.DataFrame({"p10": (P[("opp_p10", "SOXS")] - P[("opp_p10", "SOXL")]) / 2,
                           "end": (P[("opp_end", "SOXS")] - P[("opp_end", "SOXL")]) / 2}).reset_index()
        dd["d"] = dd["d"].astype(int)
        st = stats(dd.rename(columns={"p10": "x_p10", "end": "x_end"}),
                   [("soxs_minus_soxl_half_p10", "x_p10", None), ("soxs_minus_soxl_half_end", "x_end", None)])
        out[w]["same_minute_soxs_minus_soxl_half"] = st
        bm = dd.groupby("f")[["p10", "end"]].mean().rolling(15, center=True, min_periods=1).mean()
        bmap[w] = bm
        out[w]["b_by_fill_minute_every5"] = {hm(int(f)): [round(float(a), 3), round(float(b), 3)]
                                             for f, (a, b) in bm.iloc[::5].iterrows()}
    return out, bmap


class FirstFade:
    """엔진 대조용: rise 종목의 x% 첫 사건에서 반대(또는 같은) 종목을 +1% 목표로 한 번만 산다."""
    def __init__(self, rise, x, last_dec, follow=False):
        self.rise, self.x, self.ld, self.done = rise, x, last_dec, False
        self.buy = SIGN[rise] if follow else -SIGN[rise]

    def on_bar(self, ctx, i):
        if self.done or i > self.ld:
            return None
        lo = ctx.bars(self.rise, "Low")[RTH0:]
        if ctx.c(self.rise)[-1] >= lo.min() * (1 + self.x / 100):
            self.done = True
            return Order(self.buy, target=0.01)
        return None


def engine_check(R):
    out = {}
    for w, (cut, ld) in WIN.items():
        E = Engine(D, scheme=w)
        for rise in ["SOXL", "SOXS"]:
            for follow in [False, True]:
                _, tr = E.run(lambda: FirstFade(rise, 3.0, ld, follow), period="sel")
                mine = R[(R.win == w) & (R.kind == "first") & (R.x == 3.0) & (R.rise == rise)]
                col = "fol_p10" if follow else "opp_p10"
                a = mine.set_index("day")[col]
                b = tr.set_index(tr["day"].astype(str))["ret"] * 100
                j = a.index.intersection(b.index)
                key = f"{w}_{rise}_{'fol' if follow else 'opp'}"
                out[key] = {"events": int(len(a)), "engine_trades": int(len(b)), "matched": int(len(j)),
                            "max_abs_diff_pct": float((a.loc[j] - b.loc[j]).abs().max()) if len(j) else None}
    return out


def main():
    R = build()
    base, bmap = baseline_window_random()
    for k in ["p10", "end"]:
        b = np.array([bmap[w].loc[i + 1, k] for w, i in zip(R["win"], R["i"])])
        sgn = np.where(R["buy_opp"] == "SOXS", 1.0, -1.0)
        R[f"diff_adj_{k}"] = R[f"diff_{k}"] - sgn * b
    os.makedirs(OUT, exist_ok=True)
    keep = [c for c in R.columns if not c.startswith("b_")]
    R[keep].to_csv(os.path.join(OUT, "events_sel_events.csv"), index=False, encoding="utf-8-sig")
    res = {"meta": {}, "engine_check": engine_check(R), "baseline_window_random": base,
           "by_x": {}, "splits": {}, "lift": {}, "rank": {}, "consistency": {}}
    n_cells = 0
    n_ci = 0

    # 1) x별 주 표(전체 지표)
    for kind in ["first", "re", "re_reset", "first+re"]:
        res["by_x"][kind] = {}
        for w in WIN:
            res["by_x"][kind][w] = {}
            for x in (XS + XS_SUP if kind == "first" else XS):
                if kind == "first+re":
                    sub = R[(R.win == w) & (R.x == x) & R.kind.isin(["first", "re"])]
                else:
                    sub = R[(R.win == w) & (R.x == x) & (R.kind == kind)]
                st = stats(sub, FULL)
                if len(sub):
                    st["by_side_n"] = sub["rise"].value_counts().to_dict()
                res["by_x"][kind][w][xkey(x)] = st
                n_cells += 1
                n_ci += sum(1 for k, v in st.items() if isinstance(v, list))

    # 2) 나눠 보기 + 조건 lift
    for kind in ["first", "re", "re_reset"]:
        res["splits"][kind], res["lift"][kind] = {}, {}
        for w in WIN:
            res["splits"][kind][w], res["lift"][kind][w] = {}, {}
            for x in XS:
                P = R[(R.win == w) & (R.x == x) & (R.kind == kind)]
                xs, xl = {}, {}
                for col, _, _ in (SPLITS + SPLITS_RR if kind == "re_reset" else SPLITS):
                    xs[col], xl[col] = {}, {}
                    for b in sorted(P[col].unique()):
                        m = (P[col] == b).to_numpy()
                        st = stats(P[m], SHORT)
                        xs[col][b] = st
                        n_cells += 1
                        n_ci += sum(1 for k, v in st.items() if isinstance(v, list))
                        lf = lift(P, m)
                        if lf is not None:
                            xl[col][b] = lf
                            n_ci += len(lf)
                res["splits"][kind][w][xkey(x)] = xs
                res["lift"][kind][w][xkey(x)] = xl

    # 3) 순위: 칸 안 '반대 − 같은 분 무작위'(+1% 목표 매매)가 큰 조건, 사건 15건·날 10일 이상
    for kind in ["first", "re", "re_reset"]:
        res["rank"][kind] = {}
        for w in WIN:
            res["rank"][kind][w] = {}
            for x in XS:
                rows = []
                bx = res["by_x"][kind][w][xkey(x)]
                for col, nm, when in (SPLITS + SPLITS_RR if kind == "re_reset" else SPLITS):
                    for b, st in res["splits"][kind][w][xkey(x)][col].items():
                        if st.get("n", 0) < 15 or st.get("days", 0) < 10 or st.get("diff_p10") is None:
                            continue
                        lf = res["lift"][kind][w][xkey(x)][col].get(b)
                        rows.append({"split": col, "bin": b, "when": when, "n": st["n"], "days": st["days"],
                                     "diff_p10": st["diff_p10"], "diff_adj_p10": st["diff_adj_p10"],
                                     "opp_p10": st["opp_p10"], "opp_hit10": st["opp_hit10"],
                                     "lift_diff_p10": lf["diff_p10"] if lf else None,
                                     "lift_diff_adj_p10": lf["diff_adj_p10"] if lf else None})
                rows.sort(key=lambda r: -r["diff_p10"][0])
                res["rank"][kind][w][xkey(x)] = {"all_events": {"n": bx.get("n"), "diff_p10": bx.get("diff_p10"),
                                                                "diff_adj_p10": bx.get("diff_adj_p10"),
                                                                "opp_p10": bx.get("opp_p10")},
                                                 "top": rows[:8], "bottom": rows[-5:]}

    # 4) x 6개에 걸친 일관성(first·re_reset): 조건별로 diff_p10 > 0인 x 수, lift > 0인 x 수, 구간이 0 위인 x 수(사건 15건·날 10일 이상인 x만)
    for kind, w in [(k, w) for k in ["first", "re_reset"] for w in WIN]:
        res["consistency"].setdefault(kind, {})[w] = {}
        for col, nm, when in (SPLITS + SPLITS_RR if kind == "re_reset" else SPLITS):
            bins = sorted({b for x in XS for b in res["splits"][kind][w][xkey(x)][col]})
            for b in bins:
                pos = lift_pos = ci_pos = lift_ci_pos = apos = aci = 0
                vals, lvals, ns, avals = [], [], [], []
                for x in XS:
                    st = res["splits"][kind][w][xkey(x)][col].get(b)
                    lf = res["lift"][kind][w][xkey(x)][col].get(b)
                    if not st or st.get("diff_p10") is None or st["n"] < 15 or st["days"] < 10:
                        continue                      # 사건 15건·날 10일 미만 x는 세지 않는다
                    v = st["diff_p10"]
                    vals.append(v[0]); ns.append(st["n"])
                    pos += v[0] > 0; ci_pos += v[1] > 0
                    av = st.get("diff_adj_p10")
                    if av:
                        avals.append(av[0]); apos += av[0] > 0; aci += av[1] > 0
                    if lf:
                        lvals.append(lf["diff_p10"][0])
                        lift_pos += lf["diff_p10"][0] > 0; lift_ci_pos += lf["diff_p10"][1] > 0
                res["consistency"][kind][w][f"{col}|{b}"] = {
                    "when": when, "x_count": len(vals), "n_by_x": ns,
                    "diff_p10_by_x": [round(v, 3) for v in vals], "x_diff_pos": int(pos), "x_diff_ci_above0": int(ci_pos),
                    "lift_by_x": [round(v, 3) for v in lvals], "x_lift_pos": int(lift_pos), "x_lift_ci_above0": int(lift_ci_pos),
                    "diff_adj_p10_by_x": [round(v, 3) for v in avals], "x_adj_pos": int(apos), "x_adj_ci_above0": int(aci)}

    nev = R.drop_duplicates(["d", "rise", "x", "kind", "seq"])
    res["meta"] = {
        "period": "sel 2026-01-02~2026-06-30", "sel_days": len(SEL),
        "lock": "SOS_UNLOCK 켜지 않음. 엔진 대조는 Engine.run(period='sel'). 지표 예열(RSI·20봉 거래량)과 전날 종가·고저폭에 sel 첫날 전 10거래일(2025-12) 가격을 썼다(결과가 아니라 신호 때 아는 과거 값).",
        "event_def": "결정 봉 09:30~10:44(마감 ≤10:45), 종가 ≥ (09:30부터 지금 봉까지 그 종목 저가 최저) × (1+x%). first = 종목·날·x 첫 사건, re = 최저 갱신 뒤 다시(청산 리셋 없음), re_reset = 앞 사건 다음 봉부터 L 새로 잼(참고). std 표는 결정 봉 마감 09:45까지 사건만",
        "fill": "신호 봉 i 마감 → 봉 i+1 종가 × (1+반 호가) 매수. 목표는 봉 i+2부터 고가 ≥ 목표 × 1.0002, 지정가(비용 0). 창 끝 = w90 10:59 / std 09:59 봉 종가 × (1−반 호가). 손절 없음. 반 호가 cost_table.json",
        "metrics": {"mfe": "창 끝까지 최대 유리 움직임(진입가=반 호가 포함 대비 고가 최대) %", "end": "창 끝 청산 손익(비용 후) %",
                    "mae_end": "창 끝까지 최저 저가 %", "hitHH": "+H% 목표 도달률(05=0.5%, 10=1%, 15=1.5%), 0~1",
                    "pHH": "+H% 목표 매매 손익(닿으면 +H, 못 닿으면 창 끝, 비용 후) %",
                    "minHH": "도달까지 분(체결 봉 다음부터, 도달한 매매만)", "maehHH": "도달 전 MAE(체결 다음 봉~도달 봉 저가 최저, 도달한 매매만) %",
                    "legs": "opp 반대 매수, fol 오른 종목 매수(추종), rand 같은 분 두 종목 평균, diff = opp − rand = (opp − fol)/2",
                    "diff_adj": "diff − (산 종목이 SOXS면 +b(f), SOXL이면 −b(f)). b(f) = 체결 분 f의 (SOXS − SOXL)/2 sel 모든 날 평균, 앞뒤 7분 이동 평균(표본 안 값, 설명용). sel 장초 시각별 평균 방향을 걷어 낸 몫",
                    "stat": "[평균, 95% 하한, 95% 상한], 사건 가중 평균, 날 묶음 부트스트랩 10,000회 seed 20261029. *_p_le0 = 부트스트랩 평균 ≤ 0 비율(단측)",
                    "lift": "같은 x·창·사건 종류 안에서 '그 칸' − '나머지 칸'의 차이, 같은 날 다시 뽑기로 함께 부트스트랩"},
        "splits": [{"col": c, "name": n, "when": w} for c, n, w in SPLITS],
        "splits_re_reset_extra": [{"col": c, "name": n, "when": w} for c, n, w in SPLITS_RR],
        "added_after_results": ["diff_adj(시각 맞춘 방향 기준선, 대조용)", "re_reset 나눠 보기 전체와 b_seq·b_cum 칸",
                                "first 보충 x 6·7·8·10%(by_x만)"],
        "split_bins_fixed_before_results": "SPLITS 11열의 칸 경계는 첫 실행 전에 코드에 고정. re_reset 전용 2열은 결과 본 뒤 추가",
        "prev_rng_tertiles_pct": [round(float(q) * 100, 3) for q in PR_Q],
        "events_unique": {k: {xkey(x): int(((nev.kind == k) & (nev.x == x)).sum()) for x in (XS + XS_SUP if k == "first" else XS)}
                          for k in ["first", "re", "re_reset"]},
        "n_cells": n_cells, "n_ci": n_ci,
        "n_cells_note": "칸 = (사건 종류 × 창 × x × [나눠 보기 열 × 칸]) 하나. by_x 칸 + splits 칸. n_ci = 구간을 붙인 평균 수(lift 포함)",
        "runtime_s": round(time.time() - T0, 1),
    }
    with open(os.path.join(OUT, "events_sel.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("cells", n_cells, "ci", n_ci, "events", len(R), "time", round(time.time() - T0, 1))


if __name__ == "__main__":
    main()
