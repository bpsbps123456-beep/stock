# -*- coding: utf-8 -*-
"""BRIEF4 사건 표(results/ev4_events.parquet) 독립 점검. ev4_build.py·lib.py는 import하지 않는다(2번 미래 참조 시험에서만 ev4_build.features를 부른다).

1) 원자료(cache/alp1m_long/*.parquet)로 960분 격자를 새로 만들고, 면 2 × 문턱 4 사건을 처음부터 다시 훑어 사건 목록과 결과
   (r_w90·r_follow·hit 등)를 전부 대조한다. 무작위 사건 60개(seed 20261102, 연도 × 면 × 문턱 32칸 층화 돌려 뽑기)는 BRIEF4 조건 28개를
   수치 1e-6, 수준 정확히 대조한다(보조 열 alt_* 3개도 수치 대조). 같은 계산을 사건 전체에도 돌려 보조로 둔다.
   수준은 내가 따로 정한다: 값이 경계와 1e-7 안이면 가격(소수 6자리 → 유리수)·거래량(정수)으로 정확한 값을 다시 재서 BRIEF4 글자대로
   배정한다(표의 EPS 맞추기를 따라 하지 않는다). RSI(조건 8)는 정확한 값을 못 내므로 경계 근처면 '미해결'로 센다.
   다른 것마다 어느 쪽이 BRIEF4 정의에 맞는지(verdict) 적는다.
2) 미래 참조 시험: 무작위 30개(같은 seed 흐름 이어서)와 경계 사건 몇 개에서, 신호 봉 다음 봉부터 그날 자료를 (A) 쓰레기 값으로 바꾸고
   뒤 날들도 쓰레기로, (B) NaN으로 지우고 뒤 날들을 잘라 낸 자료로 ev4_build.features를 다시 돌려 조건 값(보조 열 포함)이 비트 단위로 같은지
   본다. 내 재스캔의 신호 봉·s0·순번도 같은 자료로 다시 잰다.
3) 결과 열 분포 상식 점검(사건 수, 도달률, 평균, NaN, 불변식).
덧붙임: 경계값 점검(경계 1e-7 안 사건 전부의 정확한 값, 부동소수로 반대편에 떨어진 값, 표 EPS 맞추기가 틀리게 맞춘 값),
   해석 민감도(BRIEF4 문장이 두 가지로 읽히는 곳에서 수준이 바뀌는 사건 수), 자료 품질(거래량 0 봉),
   검출력(표에 2e-6·수준 한 칸 오류를 넣으면 잡는지, features에 미래 값 4종을 넣으면 미래 참조 시험이 잡는지).

내 격자 규칙(lib.py docstring에 적힌 관례를 따로 구현): 04:00~19:59 ET 960분, 빈 분은 같은 날 앞 종가(시가·고가·저가·VWAP도 그 값)·거래량 0.
거래일 = SOXL 09:30~15:59 봉이 있는 날. 정규장 끝 = NYSE 반일장 달력 8일(BRIEF4 '반일장 8일')은 13:00, 나머지 16:00.
사용: cd code && py ev4_audit.py → results/ev4_audit.json. 손익 %(1.0 = 1%).
"""
import json
import os
import sys
import time
from fractions import Fraction as Fr

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
CACHE = os.path.join(ROOT, "cache", "alp1m_long")
RES = os.path.join(ROOT, "results")
SEED = 20261102
SYMS = ["SOXL", "SOXS", "SOXX", "QQQ", "NVDA"]
FLDS = ["Open", "High", "Low", "Close", "Volume", "VWAP"]
NMIN = 960
NEAR = 1e-7                     # 이 안이면 정확한 값으로 경계 판정


def ix(hhmm):
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m - 240


def clock(i):
    m = i + 240
    return f"{m // 60:02d}:{m % 60:02d}"


B0930, B0934, B0800 = ix("09:30"), ix("09:34"), ix("08:00")
B1100, B1000, B1300, B1600 = ix("11:00"), ix("10:00"), ix("13:00"), ix("16:00")
LAST_DEC = ix("10:44")          # 결정 봉 마감 ≤ 10:45
LAST_STD = ix("09:44")          # r_std: 결정 마감 ≤ 09:45
E90 = (B0930, B1100)            # 장초 90분 = 09:30~10:59 봉
HALF = {"2023-07-03", "2023-11-24", "2024-07-03", "2024-11-29", "2024-12-24", "2025-07-03", "2025-11-28", "2025-12-24"}
SIDES = {"user": ("SOXL", "SOXS", 1), "mirror": ("SOXS", "SOXL", -1)}   # (오른 종목 R, 사는 종목 B, 부호)
XS = [0.025, 0.03, 0.035, 0.04]
ALT = ["alt_c17_qqq_0930", "alt_c18_nvda_0930", "alt_c19_soxx_0930"]
COST = json.load(open(os.path.join(ROOT, "cost_table.json"), encoding="utf-8"))


def log(*a):
    print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


# ================================================================== 격자

def build_grid():
    raw = {}
    for s in SYMS:
        f = pd.read_parquet(os.path.join(CACHE, f"{s}.parquet"))
        assert str(f.index.tz) == "America/New_York", s
        loc = f.index.tz_localize(None)
        raw[s] = (f, loc.normalize().values.astype("datetime64[D]"), (loc.hour * 60 + loc.minute).values)
    _, dd, mm = raw["SOXL"]
    days = np.unique(dd[(mm >= 570) & (mm < 960)])
    n = len(days)
    G, HAS, RAW, dup = {}, {}, {}, {}
    for s in SYMS:
        f, dd, mm = raw[s]
        di = np.searchsorted(days, dd)
        di_c = np.minimum(di, n - 1)
        m = mm - 240
        ok = (di < n) & (days[di_c] == dd) & (m >= 0) & (m < NMIN)
        dup[s] = int(pd.Series(di_c[ok] * NMIN + m[ok]).duplicated().sum())
        R = {}
        for fld in FLDS:
            a = np.full((n, NMIN), np.nan)
            a[di_c[ok], m[ok]] = f[fld].values[ok].astype(float)
            R[fld] = a
        has = np.isfinite(R["Close"])
        pos = np.where(has, np.arange(NMIN)[None, :], 0)
        np.maximum.accumulate(pos, axis=1, out=pos)
        seen = np.logical_or.accumulate(has, axis=1)
        cf = np.where(seen, R["Close"][np.arange(n)[:, None], pos], np.nan)
        g = {"Close": cf, "Volume": np.where(has, R["Volume"], 0.0)}
        for fld in ["Open", "High", "Low", "VWAP"]:
            g[fld] = np.where(has, R[fld], cf)
        G[s], HAS[s], RAW[s] = g, has, R
    dstr = [str(d) for d in days]
    rth_end = np.array([B1300 if ds in HALF else B1600 for ds in dstr])
    v = G["SOXL"]["Volume"]
    vol_rule = v[:, B1300:B1600].sum(1) < 0.15 * v[:, B0930:B1300].sum(1)
    daily = {}
    for s in SYMS:
        o = G[s]["Open"][:, B0930]
        hi = np.array([G[s]["High"][d, B0930:rth_end[d]].max() for d in range(n)])
        lo = np.array([G[s]["Low"][d, B0930:rth_end[d]].min() for d in range(n)])
        cl = np.array([G[s]["Close"][d, rth_end[d] - 1] for d in range(n)])
        daily[s] = {"open": o, "high": hi, "low": lo, "close": cl}
    return dict(days=days, dstr=dstr, n=n, G=G, HAS=HAS, RAW=RAW, rth_end=rth_end, daily=daily, dup=dup,
                half_cal=sorted(HALF & set(dstr)), half_vol=[dstr[k] for k in np.nonzero(vol_rule)[0]])


# ================================================================== 사건 재스캔·매매

def half_spread(sym, j, cm=1.0):
    b = "pre" if j < B0930 else "open5" if j < B0930 + 5 else "early" if j < B1000 else "late"
    return COST[sym][b] * cm


def trade(row, sym, dec, cut, stop=None, cm=1.0):
    """row(sym, fld) = 그날 배열. 결정 봉 dec → dec+1 봉 종가 + 반 호가 매수, +1% 지정가(고가 ≥ 목표 × 1.0002), 손절 우선, 못 닿으면 cut−1 봉 종가 − 반 호가."""
    O, H, L, C = (row(sym, k) for k in ("Open", "High", "Low", "Close"))
    f = dec + 1
    px = C[f] * (1 + half_spread(sym, f, cm))
    tgt = px * (1 + 0.01)
    stp = px * (1 - stop) if stop else None
    for j in range(f + 1, cut):
        if stp is not None and L[j] <= stp:
            out = min(O[j], stp) * (1 - half_spread(sym, j, cm))
            return out / px - 1, j, "stop", px, out
        if H[j] >= tgt * 1.0002:
            return tgt / px - 1, j, "target", px, tgt
    j = cut - 1
    out = C[j] * (1 - half_spread(sym, j, cm))
    return out / px - 1, j, "cutoff", px, out


def scan_day(row, R, Bsym, x, full=True):
    """한 날·한 면·한 문턱. L = s0~i R 저가 최저(달리는 최저), 종가 ≥ L × (1 + x)면 결정. 청산 봉 다음 봉부터 다시 잰다."""
    Lw, C = row(R, "Low"), row(R, "Close")
    out, s0, i, seq, Lrun = [], B0930, B0930, 0, np.inf
    while i <= LAST_DEC:
        Lrun = min(Lrun, Lw[i])
        c = C[i]
        if np.isfinite(c) and np.isfinite(Lrun) and Lrun > 0 and c >= Lrun * (1 + x):
            seq += 1
            ev = {"dec": i, "s0": s0, "seq": seq, "L": Lrun, "c": c}
            r, j, why, px, outpx = trade(row, Bsym, i, B1100)
            ev.update(r=r, j=j, why=why, px=px, outpx=outpx)
            if full:
                lo, hi = row(Bsym, "Low")[i + 2:j + 1], row(Bsym, "High")[i + 2:j + 1]
                ev["mae"] = min(lo.min() / px - 1, 0.0)
                ev["mfe"] = max(hi.max() / px - 1, 0.0)
                ev["r_f"], _, ev["why_f"], _, _ = trade(row, R, i, B1100)
                ev["r_s2"], _, ev["why_s2"], _, _ = trade(row, Bsym, i, B1100, stop=0.02)
                ev["r_c2"], _, _, _, _ = trade(row, Bsym, i, B1100, cm=2.0)
                if i <= LAST_STD:
                    ev["r_std"], _, ev["why_std"], _, _ = trade(row, Bsym, i, B1000)
                else:
                    ev["r_std"], ev["why_std"] = np.nan, None
            out.append(ev)
            if why == "cutoff":
                break
            s0 = i = j + 1
            Lrun = np.inf
            continue
        i += 1
    return out


def rescan(W):
    G = W["G"]
    rows = []
    for side, (R, Bs, _) in SIDES.items():
        for x in XS:
            for d in range(W["n"]):
                row = lambda s, k, d=d: G[s][k][d]          # noqa: E731
                for e in scan_day(row, R, Bs, x):
                    e.update(side=side, x=x, d=d)
                    rows.append(e)
    return pd.DataFrame(rows)


# ================================================================== 조건(독립 구현)

def wilder(c, n=14):
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return out
    dl = np.diff(c)
    g, lo = np.where(dl > 0, dl, 0.0), np.where(dl < 0, -dl, 0.0)
    ag, al = g[:n].mean(), lo[:n].mean()
    out[n] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    for k in range(n, len(dl)):
        ag = (ag * (n - 1) + g[k]) / n
        al = (al * (n - 1) + lo[k]) / n
        out[k + 1] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out


def chains(W, sym):
    """정규장 봉 사슬(자료 첫날부터 날 넘어 이음). RSI는 거래 있는 봉만(BRIEF 해석 a) / 빈 분 포함(해석 b), 거래량은 모든 분(빈 분 0)."""
    G, H, E = W["G"][sym], W["HAS"][sym], W["rth_end"]
    k_has, c_has, k_all, c_all, v_all = [], [], [], [], []
    for d in range(W["n"]):
        m = np.arange(B0930, E[d])
        mh = m[H[d, B0930:E[d]]]
        k_has.append(d * 1000 + mh)
        c_has.append(G["Close"][d, mh])
        k_all.append(d * 1000 + m)
        c_all.append(G["Close"][d, m])
        v_all.append(G["Volume"][d, m])
    k_has, k_all = np.concatenate(k_has), np.concatenate(k_all)
    return {"k_has": k_has, "rsi_has": wilder(np.concatenate(c_has)), "k_all": k_all,
            "rsi_all": wilder(np.concatenate(c_all)), "v_all": np.concatenate(v_all)}


def at_last(keys, vals, d, i):
    p = np.searchsorted(keys, d * 1000 + i, side="right") - 1
    return vals[p] if p >= 0 else np.nan


def prev20(ch, d, j):
    p = np.searchsorted(ch["k_all"], d * 1000 + j)
    assert ch["k_all"][p] == d * 1000 + j
    if p < 20:
        return np.nan
    m = ch["v_all"][p - 20:p].mean()
    return m if m > 0 else np.nan


def pre_p0(W, d):
    """프리장 08:00 가격: 07:59 봉까지 마지막 거래 종가, 없으면 08:00~09:29 첫 거래 봉 시가."""
    hs = W["HAS"]["SOXL"][d]
    before = np.nonzero(hs[:B0800])[0]
    if len(before):
        return W["RAW"]["SOXL"]["Close"][d, before[-1]]
    nxt = np.nonzero(hs[B0800:B0930])[0]
    return W["RAW"]["SOXL"]["Open"][d, B0800 + nxt[0]] if len(nxt) else np.nan


def day_table(W):
    dl, G, n = W["daily"]["SOXL"], W["G"]["SOXL"], W["n"]
    cl, op = dl["close"], dl["open"]
    rg = (dl["high"] - dl["low"]) / dl["open"]
    a, b = E90
    o90 = G["Open"][:, a]
    h90, l90, c90 = G["High"][:, a:b].max(1), G["Low"][:, a:b].min(1), G["Close"][:, b - 1]
    eff = np.where(h90 > l90, (c90 - o90) / np.where(h90 > l90, h90 - l90, 1.0), np.nan)
    e90r = (h90 - l90) / o90
    T = {k: np.full(n, np.nan) for k in ["gap", "prevret", "ret5", "ma20", "prevrange", "preveff", "preveff_s", "e90range", "pre"]}
    for d in range(n):
        if d >= 1:
            T["gap"][d] = (op[d] / cl[d - 1] - 1) * 100
            T["preveff"][d], T["preveff_s"][d] = abs(eff[d - 1]), eff[d - 1]
        if d >= 2:
            T["prevret"][d] = (cl[d - 1] / cl[d - 2] - 1) * 100
        if d >= 6:
            T["ret5"][d] = (cl[d - 1] / cl[d - 6] - 1) * 100
        if d >= 20:
            T["ma20"][d] = cl[d - 1] / cl[d - 20:d].mean()
        if d >= 60:
            T["prevrange"][d] = rg[d - 1] / np.median(rg[d - 60:d])
            T["e90range"][d] = e90r[d - 5:d].mean() / np.median(e90r[d - 60:d])
        T["pre"][d] = (G["Close"][d, B0930 - 1] / pre_p0(W, d) - 1) * 100
    return T


def mv(W, sym, d, s0, i, sg):
    g = W["G"][sym]
    c = g["Close"][d, i]
    if sg > 0:
        return (c / g["Low"][d, s0:i + 1].min() - 1) * 100
    return (1 - c / g["High"][d, s0:i + 1].max()) * 100


def feats(W, CH, T, side, x, d, i, s0, seq):
    """BRIEF4 '조건 목록'을 원자료 격자에서 바로 계산(결정 봉 i 마감까지의 값만). 보조 열 alt_*(표에도 있음)와 해석 민감도 열."""
    R, _, sg = SIDES[side]
    g, raw, has = W["G"][R], W["RAW"][R], W["HAS"][R]
    O, H, Lw, C, V = (g[k][d] for k in ("Open", "High", "Low", "Close", "Volume"))
    c = C[i]
    seg = Lw[s0:i + 1]
    L = seg.min()
    j_last = s0 + int(np.nonzero(seg == L)[0][-1])
    j_first = s0 + int(np.nonzero(seg == L)[0][0])
    dlR = W["daily"][R]
    f = {"L": L, "L_i": j_last, "trig_close": c, "ratio": c / L}
    f["c01_time"] = float(i + 1 - B0930)
    f["c02_speed"] = float(i - j_last)
    f["c03_vs_open"] = (c / O[B0930] - 1) * 100
    f["c04_vs_prevclose"] = (c / dlR["close"][d - 1] - 1) * 100 if d >= 1 else np.nan
    f["c05_gap"] = T["gap"][d] * sg
    f["c06_first5"] = (C[B0934] / O[B0930] - 1) * 100 if i >= B0934 else np.nan
    mk = has[d, B0930:i + 1]
    vw, vv = raw["VWAP"][d, B0930:i + 1][mk], raw["Volume"][d, B0930:i + 1][mk]
    f["c07_vwap"] = (c / ((vw * vv).sum() / vv.sum()) - 1) * 100 if vv.sum() > 0 else np.nan
    ch = CH[R]
    r_i = at_last(ch["k_has"], ch["rsi_has"], d, i)
    f["c08_rsi"] = r_i
    # 신고가 표시: 봉 k(09:30 뒤)의 고가 > 09:30~k−1 고가 최고
    hh = H[B0930:i + 1]
    newh = np.zeros(len(hh), bool)
    run = hh[0]
    for k in range(1, len(hh)):
        if hh[k] > run:
            newh[k] = True
            run = hh[k]
    if newh[-1]:
        prev = np.nonzero(newh[:-1])[0]
        kb = B0930 + (int(prev[-1]) if len(prev) else 0)
        r_k = at_last(ch["k_has"], ch["rsi_has"], d, kb)
        f["c09_rsidiv"] = float(r_i < r_k) if np.isfinite(r_i) and np.isfinite(r_k) else np.nan
    else:
        f["c09_rsidiv"] = 0.0
    rng_ = H[i] - Lw[i]
    f["c10_wick"] = (H[i] - max(O[i], c)) / rng_ if rng_ > 0 else 0.0
    f["c11_vol"] = V[i] / prev20(ch, d, i)
    f["c12_risevol"] = V[j_last:i + 1].mean() / prev20(ch, d, j_last)
    f["c13_newhighs"] = float(newh[max(0, i - 14 - B0930):].sum())
    f["c14_topclose"] = float(c >= C[B0930:i + 1].max())
    f["c15_vs_prevhigh"] = (c / dlR["high"][d - 1] - 1) * 100 if d >= 1 else np.nan
    ph = raw["High"][d, :B0930][has[d, :B0930]]
    f["c16_vs_prehigh"] = (c / ph.max() - 1) * 100 if len(ph) else np.nan
    f["c17_qqq"] = mv(W, "QQQ", d, s0, i, sg)
    f["c18_nvda"] = mv(W, "NVDA", d, s0, i, sg)
    f["c19_soxx"] = mv(W, "SOXX", d, s0, i, sg) / ((c / L - 1) * 100 / 3)
    f["c20_prevret"] = T["prevret"][d]
    f["c21_prevrange"] = T["prevrange"][d]
    f["c22_preveff"] = T["preveff"][d]
    f["c23_ret5"] = T["ret5"][d]
    f["c24_pre"] = T["pre"][d]
    f["c25_weekday"] = float(pd.Timestamp(W["days"][d]).weekday())
    f["c26_ma20"] = T["ma20"][d]
    f["c27_e90range"] = T["e90range"][d]
    f["c28_seq"] = float(seq)
    # 보조 열(표 alt_*와 같은 이름): 조건 17~19를 09:30 봉부터 창으로
    f["alt_c17_qqq_0930"] = mv(W, "QQQ", d, B0930, i, sg)
    f["alt_c18_nvda_0930"] = mv(W, "NVDA", d, B0930, i, sg)
    lo930 = Lw[B0930:i + 1].min()
    f["alt_c19_soxx_0930"] = mv(W, "SOXX", d, B0930, i, sg) / ((c / lo930 - 1) * 100 / 3)
    # 해석 민감도(수준이 바뀌는지만 센다)
    f["sens_c02_first"] = float(i - j_first)
    f["sens_c08_allmin"] = at_last(ch["k_all"], ch["rsi_all"], d, i)
    f["sens_c17_mirror_rise"] = mv(W, "QQQ", d, s0, i, +1)
    sx = W["G"]["SOXL"]["Close"][d, i]
    f["sens_c23_to_signal"] = (sx / W["daily"]["SOXL"]["close"][d - 5] - 1) * 100 if d >= 5 else np.nan
    return f


# ================================================================== 정확한 값(경계 판정용)

def q(p):
    """가격은 소수 6자리 이하(원자료 전수 확인) → 문자열 경유로 정확한 유리수."""
    return Fr(str(round(float(p), 6)))


def fmed(a):
    a = sorted(a)
    n = len(a)
    return a[n // 2] if n % 2 else (a[n // 2 - 1] + a[n // 2]) / 2


def exact_value(W, CH, e, col):
    """조건 col의 정확한 값(Fraction). 못 내면 None(RSI)."""
    d, i, R = int(e["d"]), int(e["dec_i"]), str(e["rise"])
    s0, sg = int(e["s0_i"]), SIDES[str(e["side"])][2]
    G, dl = W["G"], W["daily"]
    gR = G[R]
    c = q(gR["Close"][d, i])

    def mvx(sym, a):
        g = G[sym]
        cc = q(g["Close"][d, i])
        if sg > 0:
            return (cc / q(g["Low"][d, a:i + 1].min()) - 1) * 100
        return (1 - cc / q(g["High"][d, a:i + 1].max())) * 100

    def e90(k):
        g = G["SOXL"]
        o, cc = q(g["Open"][k, B0930]), q(g["Close"][k, B1100 - 1])
        h, lo = q(g["High"][k, B0930:B1100].max()), q(g["Low"][k, B0930:B1100].min())
        return o, h, lo, cc

    def p20(j):
        ch = CH[R]
        p = np.searchsorted(ch["k_all"], d * 1000 + j)
        if p < 20:
            return None
        s = sum(Fr(int(v)) for v in ch["v_all"][p - 20:p]) / 20
        return s if s > 0 else None

    cl = dl["SOXL"]["close"]
    if col == "c03_vs_open":
        return (c / q(gR["Open"][d, B0930]) - 1) * 100
    if col == "c04_vs_prevclose":
        return (c / q(dl[R]["close"][d - 1]) - 1) * 100 if d >= 1 else None
    if col in ("c05_gap", "soxl_gap"):
        g0 = (q(G["SOXL"]["Open"][d, B0930]) / q(cl[d - 1]) - 1) * 100 if d >= 1 else None
        return None if g0 is None else g0 * (sg if col == "c05_gap" else 1)
    if col == "c06_first5":
        return (q(gR["Close"][d, B0934]) / q(gR["Open"][d, B0930]) - 1) * 100 if i >= B0934 else None
    if col == "c07_vwap":
        mk = W["HAS"][R][d, B0930:i + 1]
        vw, vv = W["RAW"][R]["VWAP"][d, B0930:i + 1][mk], W["RAW"][R]["Volume"][d, B0930:i + 1][mk]
        den = sum(Fr(int(b)) for b in vv)
        if den <= 0:
            return None
        num = sum(Fr(float(a)) * Fr(int(b)) for a, b in zip(vw, vv))        # VWAP는 받은 float 그대로 정확히
        return (c / (num / den) - 1) * 100
    if col == "c10_wick":
        o, h, lo = (q(gR[k][d, i]) for k in ("Open", "High", "Low"))
        return (h - max(o, c)) / (h - lo) if h > lo else Fr(0)
    if col == "c11_vol":
        m = p20(i)
        return Fr(int(gR["Volume"][d, i])) / m if m else None
    if col == "c12_risevol":
        seg = gR["Low"][d, s0:i + 1]
        jl = s0 + int(np.nonzero(seg == seg.min())[0][-1])
        m = p20(jl)
        vs = [Fr(int(v)) for v in gR["Volume"][d, jl:i + 1]]
        return (sum(vs) / len(vs)) / m if m else None
    if col == "c15_vs_prevhigh":
        return (c / q(dl[R]["high"][d - 1]) - 1) * 100 if d >= 1 else None
    if col == "c16_vs_prehigh":
        ph = W["RAW"][R]["High"][d, :B0930][W["HAS"][R][d, :B0930]]
        return (c / q(ph.max()) - 1) * 100 if len(ph) else None
    if col == "c17_qqq":
        return mvx("QQQ", s0)
    if col == "c18_nvda":
        return mvx("NVDA", s0)
    if col == "c19_soxx":
        return mvx("SOXX", s0) / ((c / q(gR["Low"][d, s0:i + 1].min()) - 1) * 100 / 3)
    if col == "alt_c17_qqq_0930":
        return mvx("QQQ", B0930)
    if col == "alt_c18_nvda_0930":
        return mvx("NVDA", B0930)
    if col == "alt_c19_soxx_0930":
        return mvx("SOXX", B0930) / ((c / q(gR["Low"][d, B0930:i + 1].min()) - 1) * 100 / 3)
    if col == "c20_prevret":
        return (q(cl[d - 1]) / q(cl[d - 2]) - 1) * 100 if d >= 2 else None
    if col == "c21_prevrange":
        if d < 60:
            return None
        h_, l_, o_ = dl["SOXL"]["high"], dl["SOXL"]["low"], dl["SOXL"]["open"]
        rg = [(q(h_[k]) - q(l_[k])) / q(o_[k]) for k in range(d - 60, d)]
        return rg[-1] / fmed(rg)
    if col == "c22_preveff":
        if d < 1:
            return None
        o, h, lo, cc = e90(d - 1)
        return abs(cc - o) / (h - lo) if h > lo else None
    if col == "c23_ret5":
        return (q(cl[d - 1]) / q(cl[d - 6]) - 1) * 100 if d >= 6 else None
    if col == "c24_pre":
        p0 = pre_p0(W, d)
        return (q(G["SOXL"]["Close"][d, B0930 - 1]) / q(p0) - 1) * 100 if np.isfinite(p0) else None
    if col == "c26_ma20":
        if d < 20:
            return None
        return q(cl[d - 1]) / (sum(q(cl[k]) for k in range(d - 20, d)) / 20)
    if col == "c27_e90range":
        if d < 60:
            return None
        er = [(lambda t: (t[1] - t[2]) / t[0])(e90(k)) for k in range(d - 60, d)]
        return (sum(er[-5:]) / 5) / fmed(er)
    return None                    # c08 RSI 등


# ================================================================== 수준(BRIEF4 표 글자 그대로)

LV = {  # (col, 수준 수, 함수(v, x) → 0..k−1 / None, 경계값 목록(x 함수))
    "c01_time": (4, lambda v, x: None if not np.isfinite(v) else
                 (0 if 571 <= 570 + v <= 584 else 1 if 585 <= 570 + v <= 599 else 2 if 600 <= 570 + v <= 614
                  else 3 if 615 <= 570 + v <= 645 else None), lambda x: []),
    "c02_speed": (4, lambda v, x: None if not np.isfinite(v) else 0 if v <= 5 else 1 if v <= 15 else 2 if v <= 30 else 3,
                  lambda x: []),
    "c03_vs_open": (4, lambda v, x: None if not np.isfinite(v) else 0 if v < 0 else 1 if v < 2 else 2 if v <= 4 else 3,
                    lambda x: [0.0, 2.0, 4.0]),
    "c04_vs_prevclose": (2, lambda v, x: None if not np.isfinite(v) else 0 if v < 0 else 1, lambda x: [0.0]),
    "c05_gap": (3, lambda v, x: None if not np.isfinite(v) else 0 if v <= -1 else 2 if v >= 1 else 1, lambda x: [-1.0, 1.0]),
    "c06_first5": (2, lambda v, x: None if not np.isfinite(v) else 0 if v < 0 else 1, lambda x: [0.0]),
    "c07_vwap": (4, lambda v, x: None if not np.isfinite(v) else 0 if v < 1 else 1 if v < 2 else 2 if v < 3 else 3,
                 lambda x: [1.0, 2.0, 3.0]),
    "c08_rsi": (3, lambda v, x: None if not np.isfinite(v) else 0 if v < 70 else 1 if v < 80 else 2, lambda x: [70.0, 80.0]),
    "c09_rsidiv": (2, lambda v, x: None if not np.isfinite(v) else 0 if v == 1 else 1, lambda x: []),
    "c10_wick": (2, lambda v, x: None if not np.isfinite(v) else 0 if v < 0.3 else 1, lambda x: [0.3]),
    "c11_vol": (3, lambda v, x: None if not np.isfinite(v) else 0 if v < 1 else 1 if v < 2 else 2, lambda x: [1.0, 2.0]),
    "c12_risevol": (2, lambda v, x: None if not np.isfinite(v) else 0 if v < 1 else 1, lambda x: [1.0]),
    "c13_newhighs": (3, lambda v, x: None if not np.isfinite(v) else 0 if v <= 1 else 1 if v <= 3 else 2, lambda x: []),
    "c14_topclose": (2, lambda v, x: None if not np.isfinite(v) else 0 if v == 1 else 1, lambda x: []),
    "c15_vs_prevhigh": (3, lambda v, x: None if not np.isfinite(v) else 0 if v < -0.5 else 2 if v > 0.5 else 1,
                        lambda x: [-0.5, 0.5]),
    "c16_vs_prehigh": (2, lambda v, x: None if not np.isfinite(v) else 0 if v < 0 else 1, lambda x: [0.0]),
    "c17_qqq": (2, lambda v, x: None if not np.isfinite(v) else 0 if v < 0.5 else 1, lambda x: [0.5]),
    "c18_nvda": (2, lambda v, x: None if not np.isfinite(v) else 0 if v < x * 100 / 3 else 1, lambda x: [x * 100 / 3]),
    "c19_soxx": (2, lambda v, x: None if not np.isfinite(v) else 0 if v < 0.8 else 1, lambda x: [0.8]),
    "c20_prevret": (3, lambda v, x: None if not np.isfinite(v) else 0 if v <= -3 else 2 if v >= 3 else 1, lambda x: [-3.0, 3.0]),
    "c21_prevrange": (2, lambda v, x: None if not np.isfinite(v) else 0 if v < 1 else 1, lambda x: [1.0]),
    "c22_preveff": (2, lambda v, x: None if not np.isfinite(v) else 0 if v < 0.6 else 1, lambda x: [0.6]),
    "c23_ret5": (3, lambda v, x: None if not np.isfinite(v) else 0 if v <= -5 else 2 if v >= 5 else 1, lambda x: [-5.0, 5.0]),
    "c24_pre": (3, lambda v, x: None if not np.isfinite(v) else 0 if v <= -1 else 2 if v >= 1 else 1, lambda x: [-1.0, 1.0]),
    "c25_weekday": (5, lambda v, x: None if not np.isfinite(v) else int(v), lambda x: []),
    "c26_ma20": (2, lambda v, x: None if not np.isfinite(v) else 0 if v < 1 else 1, lambda x: [1.0]),
    "c27_e90range": (2, lambda v, x: None if not np.isfinite(v) else 0 if v < 1 else 1, lambda x: [1.0]),
    "c28_seq": (2, lambda v, x: None if not np.isfinite(v) else 0 if v <= 1 else 1, lambda x: []),
}
CN = list(LV)
# BRIEF4 표가 경계에서 겹치거나 말이 없는 곳(어느 쪽 배정도 '틀림'이 아님). 나는 위 칸(표와 같은 관례)에 둔다.
AMBIG = {("c03_vs_open", 2.0), ("c04_vs_prevclose", 0.0), ("c06_first5", 0.0), ("c07_vwap", 2.0), ("c16_vs_prehigh", 0.0)}
# 보조 열·C3 가설에서 고정 문턱과 바로 비교하는 값(수준 열이 없음)
EXTRA_EDGES = {"alt_c17_qqq_0930": lambda x: [0.5], "alt_c18_nvda_0930": lambda x: [x * 100 / 3], "alt_c19_soxx_0930": lambda x: [0.8],
               "c10_wick": lambda x: [0.5], "soxl_gap": lambda x: [-2.0], "c04_vs_prevclose": lambda x: [-0.5, 0.5]}
LABELS = {}


def bnd_exact(col, b, x):
    return Fr(str(x)) * 100 / 3 if col in ("c18_nvda", "alt_c18_nvda_0930") else Fr(str(b))


def sgn(a):
    return int(a > 0) - int(a < 0)


def resolve_levels(W, CH, ev, col, vals):
    """내 수준(BRIEF4 글자, 경계 1e-7 안은 정확한 값으로). 반환: 코드 배열(−1 = 결측), 경계 근처 기록."""
    k, fn, bnd = LV[col]
    xs = ev["x"].values
    codes = np.full(len(vals), -1)
    recs = []
    for r, (v, x) in enumerate(zip(vals, xs)):
        lv = fn(v, x)
        near = [b for b in bnd(x) if np.isfinite(v) and abs(v - b) < NEAR]
        if near:
            b = near[0]
            e = ev.iloc[r]
            ex = exact_value(W, CH, e, col)
            tl = e[col + "_lv"]
            rec = {"row": r, "col": col, "boundary": b, "day": str(e["day"].date()), "year": int(e["year"]), "side": str(e["side"]),
                   "x": float(x), "dec_close": e["dec_close"], "value_float": repr(float(v)), "within_1e-9": bool(abs(v - b) <= 1e-9),
                   "float_side": sgn(v - b), "ambiguous": (col, b) in AMBIG, "table_level": None if pd.isna(tl) else str(tl)}
            if ex is None:
                rec.update(exact_side=None, unresolved=True)
            else:
                s = sgn(ex - bnd_exact(col, b, x))
                lv = fn(b, x) if s == 0 else fn(b + s * 1e-6, x)
                rec.update(exact_side=s, unresolved=False, exact_value=str(ex) if len(str(ex)) < 60 else float(ex))
            rec["my_level"] = None if lv is None else LABELS[col][lv]
            rec["table_vs_mine_differ"] = rec["my_level"] != rec["table_level"]
            recs.append(rec)
        codes[r] = -1 if lv is None else lv
    return codes, recs


def same_num(a, b, tol=1e-6):
    a, b = float(a), float(b)
    if np.isnan(a) and np.isnan(b):
        return True
    if np.isnan(a) or np.isnan(b):
        return False
    if np.isinf(a) or np.isinf(b):
        return a == b
    return abs(a - b) <= tol


def compare_event(mine, mycode, e, near_rec):
    """mine = 내 수치, mycode = {col: 내 수준 코드}, e = 표의 한 행, near_rec = {col: 경계 근처 기록}. 반환: 다른 항목 목록(+ 어느 쪽이 맞는지)."""
    out = []
    for col in CN:
        mv_, tv = mine[col], e[col]
        if not same_num(mv_, tv):
            out.append({"col": col, "kind": "수치", "mine": float(mv_), "table": float(tv),
                        "verdict": "검토 필요: 두 계산이 다르다(정의 해석 차이인지 원인 확인)"})
        tl = e[col + "_lv"]
        tcode = -1 if pd.isna(tl) else LABELS[col].index(tl)
        if mycode[col] != tcode:
            rec = near_rec.get(col)
            if rec is None:
                vd = "검토 필요: 경계와 멀리 떨어진 값인데 수준이 다르다(수준 함수 차이)"
            elif rec["ambiguous"]:
                vd = "둘 다 가능: BRIEF4 글자가 이 경계에서 겹친다"
            elif rec["unresolved"]:
                vd = "미해결: 정확한 값을 못 낸다(RSI)"
            else:
                vd = "내 쪽: 정확한 값이 " + ("경계와 같다" if rec["exact_side"] == 0 else "경계 " +
                                           ("위" if rec["exact_side"] > 0 else "아래")) + ", BRIEF4 글자대로면 '" + str(rec["my_level"]) + "'"
            out.append({"col": col, "kind": "수준", "mine": None if mycode[col] < 0 else LABELS[col][mycode[col]],
                        "table": None if tcode < 0 else LABELS[col][tcode], "value": float(mv_), "verdict": vd})
    for a in ALT:
        if not same_num(mine[a], e[a]):
            out.append({"col": a, "kind": "보조 수치", "mine": float(mine[a]), "table": float(e[a]), "verdict": "검토 필요"})
    return out


# ================================================================== 2. 미래 참조 시험

def leak_test(W, ev, pick, rng, inject=None):
    """inject: 검출력 확인용으로 features 뒤에 일부러 미래 값을 섞는다(None = 정상 시험).
    'next_bar' 오늘 봉 i+1 종가, 'next_day' 다음 날 09:30 종가, 'today_daily' 오늘 정규장 일봉 고가, 'has_peek' 오늘 i+1~ 거래 봉 수."""
    sys.path.insert(0, HERE)
    import ev4_build as EB                        # 이 시험에서만(그 안에서 lib.Ctx를 쓴다)
    import datetime as dt
    days_py = [dt.date.fromisoformat(s) for s in W["dstr"]]

    def mk(A, has, days, daily_df):
        o = type("Dlike", (), {})()
        o.A, o.has, o.days, o.daily = A, has, days, daily_df
        return o

    def daily_frame(days, upto=None, garbage_from=None):
        rows = []
        for s in SYMS:
            dl = W["daily"][s]
            for k, day in enumerate(days):
                if upto is not None and k >= upto:
                    break
                vals = [dl[c][k] for c in ("open", "high", "low", "close")]
                if garbage_from is not None and k >= garbage_from:
                    vals = list(rng.uniform(1, 1e6, 4))
                rows.append((s, day, *vals))
        return pd.DataFrame(rows, columns=["sym", "day", "open", "high", "low", "close"]).set_index(["sym", "day"]).sort_index()

    base_daily = daily_frame(days_py)
    D0 = mk({s: dict(W["G"][s]) for s in SYMS}, dict(W["HAS"]), days_py, base_daily)

    def run(Dx, e):
        ctx = EB.Ctx(Dx, int(e["d"]))
        ctx.i = int(e["dec_i"])
        R = str(e["rise"])
        try:
            f = EB.features(ctx, EB.Past(), ctx.i, R, 1 if R == "SOXL" else -1, float(e["x"]), int(e["s0_i"]), int(e["c28_seq"]))
            if inject == "next_bar":
                f["c10_wick"] = float(Dx.A[R]["Close"][ctx.d, ctx.i + 1])
            elif inject == "next_day":
                f["c20_prevret"] = float(Dx.A["SOXL"]["Close"][ctx.d + 1, B0930])
            elif inject == "today_daily":
                f["c15_vs_prevhigh"] = float(Dx.daily.loc[("SOXL", Dx.days[ctx.d])]["high"])
            elif inject == "has_peek":
                f["c13_newhighs"] = float(Dx.has["SOXL"][ctx.d, ctx.i + 1:].sum())
            return f
        except Exception as ex:                    # 잘라 낸 자료에서 미래를 읽으면 색인·키 오류 = 검출
            return {"__error__": type(ex).__name__}

    keys = CN + ALT + ["L", "jL", "ratio", "trig_close", "rsidiv_gap", "preveff_signed", "soxl_gap"]

    def bit_same(a, b):
        a, b = float(a), float(b)
        return (np.isnan(a) and np.isnan(b)) or a == b

    def fdiff(f0, f1):
        if "__error__" in f0 or "__error__" in f1:
            return ["예외:" + f1.get("__error__", f0.get("__error__", ""))]
        return [k for k in keys if not bit_same(f0[k], f1[k])]

    res = []
    for _, e in pick.iterrows():
        d, i = int(e["d"]), int(e["dec_i"])
        f0 = run(D0, e)
        vs_table = [c for c in CN + ALT if "__error__" in f0 or not same_num(f0[c], e[c])]
        # (A) 쓰레기: 오늘 i+1~ 끝, 뒤 날 전부, 오늘·뒤 날 일봉 표
        A1, H1 = {}, {}
        for s in SYMS:
            A1[s] = {}
            for fld in FLDS:
                a = W["G"][s][fld].copy()
                a[d, i + 1:] = rng.uniform(1, 1e6, NMIN - i - 1)
                a[d + 1:] = rng.uniform(1, 1e6, a[d + 1:].shape)
                A1[s][fld] = a
            h = W["HAS"][s].copy()
            h[d, i + 1:] = rng.random(NMIN - i - 1) < 0.5
            h[d + 1:] = rng.random(h[d + 1:].shape) < 0.5
            H1[s] = h
        f1 = run(mk(A1, H1, days_py, daily_frame(days_py, garbage_from=d)), e)
        # (B) 잘라 냄: 오늘 i+1~ NaN·거래 없음, 뒤 날 삭제, 오늘부터 일봉 표 삭제
        A2, H2 = {}, {}
        for s in SYMS:
            A2[s] = {}
            for fld in FLDS:
                a = W["G"][s][fld][:d + 1].copy()
                a[d, i + 1:] = np.nan
                A2[s][fld] = a
            h = W["HAS"][s][:d + 1].copy()
            h[d, i + 1:] = False
            H2[s] = h
        f2 = run(mk(A2, H2, days_py[:d + 1], daily_frame(days_py, upto=d)), e)
        dA, dB = fdiff(f0, f1), fdiff(f0, f2)
        # 내 재스캔의 신호 봉·s0·순번(결정 봉 ≤ i)도 같은 자료로
        R, Bs, _ = SIDES[str(e["side"])]
        base = [(q_["dec"], q_["s0"], q_["seq"]) for q_ in scan_day(lambda s, k: W["G"][s][k][d], R, Bs, float(e["x"]), full=False)
                if q_["dec"] <= i]
        alt = [(q_["dec"], q_["s0"], q_["seq"]) for q_ in scan_day(lambda s, k: A1[s][k][d], R, Bs, float(e["x"]), full=False)
               if q_["dec"] <= i]
        res.append({"day": W["dstr"][d], "side": str(e["side"]), "x": float(e["x"]), "dec_close": clock(i + 1),
                    "seq": int(e["c28_seq"]), "f0_vs_table": vs_table, "diff_garbage": dA, "diff_cut": dB,
                    "trigger_same": base == alt and (i, int(e["s0_i"]), int(e["c28_seq"])) in base})
    return res


# ================================================================== 경계값·자료 품질·검출력

def boundary_summary(W, CH, ev, RECS):
    """경계 1e-7 안 사건 전부: 정확한 값이 경계 위·위·아래 어디인지, float가 반대편에 떨어졌는지, 표 수준이 BRIEF4대로인지."""
    R = list(RECS.values())
    by = {}
    for r in R:
        b = by.setdefault(f"{r['col']}={r['boundary']:g}", {"hits": 0, "exact_on": 0, "exact_above": 0, "exact_below": 0, "unresolved": 0,
                                                             "float_wrong_side": 0, "within_1e-9_but_exact_off": 0,
                                                             "table_vs_brief_wrong": 0, "ambiguous": r["ambiguous"]})
        b["hits"] += 1
        if r["unresolved"]:
            b["unresolved"] += 1
            continue
        b["exact_on" if r["exact_side"] == 0 else "exact_above" if r["exact_side"] > 0 else "exact_below"] += 1
        b["float_wrong_side"] += int(r["float_side"] != r["exact_side"])
        b["within_1e-9_but_exact_off"] += int(r["within_1e-9"] and r["exact_side"] != 0)
        b["table_vs_brief_wrong"] += int(r["table_vs_mine_differ"] and not r["ambiguous"])
    wrong = [r for r in R if r["table_vs_mine_differ"] and not r["ambiguous"]]
    # 수준 열이 없는 보조 값(alt_*)과 C3 가설 경계(H4 윗꼬리 ≥ 0.5, H8 갭 ≤ −2%·SOXL 전날 종가 ±0.5%): float 비교가 정확한 값과 반대가 되는 사건
    extra = {}
    for col, ef in EXTRA_EDGES.items():
        hits = []
        for r_, (v, x) in enumerate(zip(ev[col].values.astype(float), ev["x"].values)):
            for b_ in ef(x):
                if np.isfinite(v) and abs(v - b_) < NEAR:
                    e = ev.iloc[r_]
                    ex = exact_value(W, CH, e, col)
                    s = None if ex is None else sgn(ex - bnd_exact(col, b_, x))
                    hits.append({"boundary": b_, "day": str(e["day"].date()), "side": str(e["side"]), "x": float(x), "dec_close": e["dec_close"],
                                 "value_float": repr(float(v)), "exact_side": s, "float_side": sgn(v - b_)})
        extra[col] = {"hits": len(hits), "exact_on": sum(1 for h in hits if h["exact_side"] == 0),
                      "float_wrong_side": sum(1 for h in hits if h["exact_side"] is not None and h["float_side"] != h["exact_side"]),
                      "float_below_but_exact_on(≥ 비교에서 빠짐)": sum(1 for h in hits if h["exact_side"] == 0 and h["float_side"] < 0),
                      "float_above_but_exact_on(≤ 비교에 안 듦)": sum(1 for h in hits if h["exact_side"] == 0 and h["float_side"] > 0),
                      "user_x3": sum(1 for h in hits if h["side"] == "user" and h["x"] == 0.03),
                      "detail": hits[:12]}
    return {"near_window": NEAR, "hits": len(R), "by_col_boundary": by,
            "table_vs_brief_wrong": len(wrong), "table_vs_brief_wrong_detail": wrong,
            "float_wrong_side_total": int(sum(v["float_wrong_side"] for v in by.values())),
            "snap_misapplied(within_1e-9_but_exact_off)": int(sum(v["within_1e-9_but_exact_off"] for v in by.values())),
            "unresolved_total": int(sum(v["unresolved"] for v in by.values())),
            "extra_edges_no_level_column": extra}


def data_quality(W, ev):
    """정규장 거래 봉 중 거래량 0 비율(분할 조정 반올림 의심)과 사건 신호 봉 거래량 0 수."""
    out = {}
    yrs = np.array([int(s[:4]) for s in W["dstr"]])
    for s in ("SOXL", "SOXS"):
        h, v = W["HAS"][s][:, B0930:B1600], W["RAW"][s]["Volume"][:, B0930:B1600]
        out[f"{s}_rth_trade_bars_vol0_share"] = {str(y): round(float(((v == 0) & h)[yrs == y].sum() / max(h[yrs == y].sum(), 1)), 5)
                                                  for y in (2023, 2024, 2025, 2026)}
    v0 = np.array([W["G"][str(r)]["Volume"][d, i] == 0 for r, d, i in zip(ev["rise"], ev["d"], ev["dec_i"])])
    out["signal_bar_vol0_events"] = int(v0.sum())
    out["c11_median_by_side_year"] = {f"{s}_{y}": round(float(g["c11_vol"].median()), 3)
                                      for (s, y), g in ev.assign(side=ev["side"].astype(str)).groupby(["side", "year"])}
    return out


def power_checks(W, ev, FA, MYC, rng):
    """시험이 실제로 잡는지: (1) 표 사본의 수치에 2e-6을 더하고 수준을 한 칸 바꾸면 compare_event가 잡는가,
    (2) features에 일부러 미래 값을 섞으면 미래 참조 시험(쓰레기·잘라 냄)이 잡는가."""
    q0 = rng.choice(len(ev), size=len(CN), replace=False)
    caught_num = caught_lv = 0
    for col, q_ in zip(CN, q0):
        e = ev.iloc[q_].copy()
        v = float(e[col])
        e[col] = (v + 2e-6) if np.isfinite(v) else 0.0
        labs = LABELS[col]
        cur = e[col + "_lv"]
        e[col + "_lv"] = labs[(labs.index(cur) + 1) % len(labs)] if not pd.isna(cur) else labs[0]
        d_ = compare_event(FA.iloc[q_], {c: MYC[c][q_] for c in CN}, e, {})
        caught_num += any(x_["col"] == col and x_["kind"] == "수치" for x_ in d_)
        caught_lv += any(x_["col"] == col and x_["kind"] == "수준" for x_ in d_)
    pick = ev.iloc[rng.choice(len(ev), size=5, replace=False)]
    inj = {}
    for mode in ("next_bar", "next_day", "today_daily", "has_peek"):
        L = leak_test(W, ev, pick, rng, inject=mode)
        inj[mode] = {"n": len(L), "caught_garbage": int(sum(1 for r in L if r["diff_garbage"])),
                     "caught_cut": int(sum(1 for r in L if r["diff_cut"]))}
    return {"comparator": {"conditions": len(CN), "caught_numeric_2e-6": int(caught_num), "caught_level_shift": int(caught_lv)},
            "leak_injection": inj}


# ================================================================== main

def main():
    ev = pd.read_parquet(os.path.join(RES, "ev4_events.parquet"))
    for col in CN:
        LABELS[col] = list(ev[col + "_lv"].cat.categories)
    log(f"사건 표 {len(ev)}행 × {ev.shape[1]}열")
    W = build_grid()
    log(f"격자 {W['dstr'][0]}~{W['dstr'][-1]} {W['n']}일, 중복 봉 {W['dup']}")
    out = {"grid": {"days": W["n"], "first": W["dstr"][0], "last": W["dstr"][-1], "dup_bars": W["dup"],
                    "half_days_calendar": W["half_cal"], "half_days_volume_rule": W["half_vol"],
                    "half_days_same": W["half_cal"] == W["half_vol"]}}
    tday = ev.groupby("d")["day"].first()
    out["grid"]["day_index_mismatch"] = int(sum(str(pd.Timestamp(v).date()) != W["dstr"][k] for k, v in tday.items()))
    meta = json.load(open(os.path.join(RES, "ev4_events_meta.json"), encoding="utf-8"))
    out["build_selfchecks"] = {"code_sha256_16": meta.get("code_sha256_16"),
                               "h0_bitmatch_all": all(v["match"] for v in meta["checks"]["h0_bitmatch"].values()),
                               "engine_vs_sim_max": max(meta["checks"]["engine_vs_sim_max_abs_diff"].values()),
                               "leak_test": {k: meta["checks"]["leak_test"][k] for k in ("n", "mismatch_events")}}

    # ---------- 1a. 사건 전수 재스캔
    me = rescan(W)
    log(f"재스캔 사건 {len(me)}")
    key = ["side", "x", "d", "dec_i"]
    me = me.rename(columns={"dec": "dec_i"})
    me = me.rename(columns={c: "m_" + c for c in me.columns if c not in key})
    tb = ev.copy()
    tb["side"] = tb["side"].astype(str)
    mg = tb.merge(me, on=key, how="outer", indicator=True)
    only_t, only_m = mg[mg["_merge"] == "left_only"], mg[mg["_merge"] == "right_only"]
    both = mg[mg["_merge"] == "both"].copy()
    num_pairs = {"r_w90": ("m_r", 100), "r_follow": ("m_r_f", 100), "r_w90_s2": ("m_r_s2", 100), "r_w90_c2": ("m_r_c2", 100),
                 "r_std": ("m_r_std", 100), "mae": ("m_mae", 100), "mfe": ("m_mfe", 100), "entry_px": ("m_px", 1),
                 "exit_px": ("m_outpx", 1), "L": ("m_L", 1), "trig_close": ("m_c", 1)}
    full = {"events_table": int(len(ev)), "events_rescan": int(len(me)), "only_table": int(len(only_t)),
            "only_rescan": int(len(only_m)), "matched": int(len(both)), "max_abs_diff": {}, "mismatch_n": {}}
    for tcol, (mcol, sc) in num_pairs.items():
        a, b = both[tcol].astype(float).values, both[mcol].astype(float).values * sc
        nn = np.isnan(a) & np.isnan(b)
        dif = np.where(nn, 0.0, np.abs(a - b))
        full["max_abs_diff"][tcol] = float(np.nanmax(dif)) if len(dif) else 0.0
        full["mismatch_n"][tcol] = int(((dif > 1e-6) | (np.isnan(dif) & ~nn)).sum())
    for tcol, mcol in {"why": "m_why", "why_follow": "m_why_f", "why_s2": "m_why_s2", "why_std": "m_why_std"}.items():
        a = [None if pd.isna(v) else str(v) for v in both[tcol].values]
        b = [None if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v) for v in both[mcol].values]
        full["mismatch_n"][tcol] = int(sum(p != q_ for p, q_ in zip(a, b)))
    full["mismatch_n"]["hit"] = int((both["hit"].values != (both["m_why"].astype(str).values == "target")).sum())
    full["mismatch_n"]["exit_i"] = int((both["exit_i"].values != both["m_j"].values).sum())
    full["mismatch_n"]["s0_i"] = int((both["s0_i"].values != both["m_s0"].values).sum())
    full["mismatch_n"]["c28_seq"] = int((both["c28_seq"].values != both["m_seq"].values).sum())
    full["only_table_rows"] = only_t[key].astype(str).head(10).to_dict("records")
    full["only_rescan_rows"] = only_m[key].astype(str).head(10).to_dict("records")
    out["rescan_all"] = full
    log("재스캔 대조", json.dumps({k: full[k] for k in ["only_table", "only_rescan", "matched"]}), json.dumps(full["mismatch_n"]))

    # ---------- 1b. 조건 28개 + 보조 3개(사건 전체 + 무작위 60)
    CH = {s: chains(W, s) for s in ("SOXL", "SOXS")}
    T = day_table(W)
    log("RSI·거래량 사슬, 날 값 완료")
    FA = pd.DataFrame([feats(W, CH, T, str(r.side), float(r.x), int(r.d), int(r.dec_i), int(r.s0_i), int(r.c28_seq))
                       for r in ev.itertuples(index=False)])
    log("조건 재계산 완료")
    MYC, RECS = {}, {}
    for col in CN:
        MYC[col], recs = resolve_levels(W, CH, ev, col, FA[col].values.astype(float))
        for r in recs:
            RECS[(r["row"], col)] = r
    log(f"수준 배정(경계 1e-7 안 {len(RECS)}건은 정확한 값으로)")
    allcmp = {"numeric_mismatch": {}, "level_mismatch": {}, "max_abs_diff": {}, "examples": {}, "alt_numeric_mismatch": {},
              "alt_max_abs_diff": {}}
    for col in CN + ALT:
        a, b = FA[col].values.astype(float), ev[col].values.astype(float)
        nn = np.isnan(a) & np.isnan(b)
        dif = np.where(nn, 0.0, np.abs(a - b))
        bad = (dif > 1e-6) | (np.isnan(dif) & ~nn)
        mx = float(np.nanmax(np.where(np.isinf(dif), np.nan, dif))) if (~np.isnan(dif)).any() else None
        if col in ALT:
            allcmp["alt_numeric_mismatch"][col], allcmp["alt_max_abs_diff"][col] = int(bad.sum()), mx
            continue
        allcmp["numeric_mismatch"][col], allcmp["max_abs_diff"][col] = int(bad.sum()), mx
        tc = ev[col + "_lv"].cat.codes.values
        lbad = MYC[col] != tc
        allcmp["level_mismatch"][col] = int(lbad.sum())
        ex = np.nonzero(bad | lbad)[0][:5]
        if len(ex):
            allcmp["examples"][col] = [{"day": str(ev["day"].iloc[q_].date()), "side": str(ev["side"].iloc[q_]), "x": float(ev["x"].iloc[q_]),
                                        "dec_close": ev["dec_close"].iloc[q_], "mine": float(a[q_]), "table": float(b[q_]),
                                        "mine_lv": LABELS[col][MYC[col][q_]] if MYC[col][q_] >= 0 else None,
                                        "table_lv": LABELS[col][tc[q_]] if tc[q_] >= 0 else None,
                                        "near_boundary": RECS.get((int(q_), col))} for q_ in ex]
    out["conditions_all"] = allcmp
    log("조건 전체 대조: 수치 불일치", sum(allcmp["numeric_mismatch"].values()), "수준 불일치", sum(allcmp["level_mismatch"].values()),
        "보조 수치 불일치", sum(allcmp["alt_numeric_mismatch"].values()))

    # 해석 민감도(사건 전체에서 수준이 바뀌는 수, float 판정)
    def lvc(col, vals):
        fn = LV[col][1]
        return np.array([(-1 if fn(v, x) is None else fn(v, x)) for v, x in zip(vals, ev["x"].values)])
    yr, us3 = ev["year"].values == 2026, (ev["side"].astype(str).values == "user") & (ev["x"].values == 0.03)
    sens = {}
    for alt, col in [("sens_c02_first", "c02_speed"), ("sens_c08_allmin", "c08_rsi"), ("alt_c17_qqq_0930", "c17_qqq"),
                     ("alt_c18_nvda_0930", "c18_nvda"), ("alt_c19_soxx_0930", "c19_soxx"), ("sens_c23_to_signal", "c23_ret5")]:
        ch_ = lvc(col, FA[alt].values) != lvc(col, FA[col].values)
        sens[f"{col} ← {alt}"] = {"all": int(ch_.sum()), "2026": int((ch_ & yr).sum()), "user_x3_2026": int((ch_ & yr & us3).sum())}
    mir = ev["side"].astype(str).values == "mirror"
    ch_ = (lvc("c17_qqq", FA["sens_c17_mirror_rise"].values) != lvc("c17_qqq", FA["c17_qqq"].values)) & mir
    sens["c17_qqq mirror ← 부호 안 뒤집음(QQQ 저점 대비 오름)"] = {"all": int(ch_.sum()), "2026": int((ch_ & yr).sum())}
    sens["c06_first5 NaN(결정 마감 ≤ 09:34라 모름)"] = {"all": int(np.isnan(ev["c06_first5"].values).sum()),
                                                  "2026": int((np.isnan(ev["c06_first5"].values) & yr).sum())}
    out["interpretation_sensitivity"] = sens

    # 무작위 60(층화)
    rng = np.random.default_rng(SEED)
    grp = {k: rng.permutation(g.index.values) for k, g in ev.groupby(["year", "side", "x"], observed=True)}
    ks = sorted(grp)
    pick, rd = [], 0
    while len(pick) < 60:
        for o in rng.permutation(len(ks)):
            if len(pick) < 60 and rd < len(grp[ks[o]]):
                pick.append(int(grp[ks[o]][rd]))
        rd += 1
    s60 = []
    for q_ in pick:
        e = ev.loc[q_]
        diffs = compare_event(FA.loc[q_], {c: MYC[c][q_] for c in CN}, e, {c: RECS[(q_, c)] for c in CN if (q_, c) in RECS})
        mrow = me[(me["side"] == str(e["side"])) & (me["x"] == e["x"]) & (me["d"] == e["d"]) & (me["dec_i"] == e["dec_i"])]
        if len(mrow) == 1:
            m1 = mrow.iloc[0]
            oc = {"r_w90": [float(m1["m_r"] * 100), float(e["r_w90"])], "r_follow": [float(m1["m_r_f"] * 100), float(e["r_follow"])],
                  "hit": [int(m1["m_why"] == "target"), int(e["hit"])]}
            for k_, (a_, b_) in oc.items():
                if (k_ == "hit" and a_ != b_) or (k_ != "hit" and not same_num(a_, b_)):
                    diffs.append({"col": k_, "kind": "결과", "mine": a_, "table": b_, "verdict": "검토 필요"})
        else:
            diffs.append({"col": "event", "kind": "사건", "mine": "재스캔에 없음", "table": "있음", "verdict": "검토 필요"})
        s60.append({"row": int(q_), "day": str(e["day"].date()), "year": int(e["year"]), "side": str(e["side"]), "x": float(e["x"]),
                    "dec_close": e["dec_close"], "seq": int(e["c28_seq"]), "r_w90": round(float(e["r_w90"]), 6), "hit": int(e["hit"]),
                    "near_boundary": [RECS[(q_, c)]["col"] for c in CN if (q_, c) in RECS],
                    "n_checked": 28 * 2 + len(ALT) + 3, "diffs": diffs})
    cnt = pd.Series([(r["year"], r["side"], r["x"]) for r in s60]).value_counts()
    out["sample60"] = {"strata_counts": {"years": pd.Series([r["year"] for r in s60]).value_counts().sort_index().to_dict(),
                                         "sides": pd.Series([r["side"] for r in s60]).value_counts().to_dict(),
                                         "x": pd.Series([r["x"] for r in s60]).value_counts().sort_index().to_dict(),
                                         "cells_min_max": [int(cnt.min()), int(cnt.max())], "cells": int(len(cnt))},
                       "items_checked": int(sum(r["n_checked"] for r in s60)),
                       "events_with_diff": int(sum(1 for r in s60 if r["diffs"])),
                       "diff_total": int(sum(len(r["diffs"]) for r in s60)), "events": s60}
    log("무작위 60: 다른 사건", out["sample60"]["events_with_diff"], "다른 항목", out["sample60"]["diff_total"])

    # ---------- 2. 미래 참조 시험
    p30 = list(rng.choice(len(ev), size=30, replace=False))
    edge = {}
    evs = ev.sort_values(["d", "side", "x", "dec_i"])
    prev_half = {k + 1 for k, s in enumerate(W["dstr"]) if s in HALF}
    cands = {"첫 사건(자료 첫날 근처)": evs.index[0],
             "결정 마감 ≤ 09:34(c06 NaN)": evs[evs["dec_i"] < B0934].index[:1],
             "결정 봉 = 09:34 봉": evs[evs["dec_i"] == B0934].index[:1],
             "결정 봉 = 10:44 봉": evs[evs["dec_i"] == LAST_DEC].index[:1],
             "순번 ≥ 3": evs[evs["c28_seq"] >= 3].index[:1],
             "전날 반일장": evs[evs["d"].isin(prev_half)].index[:1],
             "거울 2023 순번 2": evs[(evs["side"] == "mirror") & (evs["year"] == 2023) & (evs["c28_seq"] == 2)].index[:1],
             "d < 60(c21·c27 NaN)": evs[(evs["d"] > 5) & (evs["d"] < 60)].index[:1],
             "마지막 날(2026-10-28)": evs[evs["d"] == W["n"] - 1].index[:1]}
    for nm, v in cands.items():
        v = list(np.atleast_1d(v))
        if v:
            edge[nm] = int(v[0])
    L30 = leak_test(W, ev, ev.loc[p30], rng)
    LE = leak_test(W, ev, ev.loc[list(edge.values())], rng)
    for nm, r in zip(edge, LE):
        r["edge"] = nm

    def lsum(L):
        return {"n": len(L), "changed_garbage": int(sum(1 for r in L if r["diff_garbage"])),
                "changed_cut": int(sum(1 for r in L if r["diff_cut"])),
                "trigger_changed": int(sum(1 for r in L if not r["trigger_same"])),
                "baseline_vs_table_mismatch": int(sum(1 for r in L if r["f0_vs_table"])),
                "values_compared_per_event": len(CN) + len(ALT) + 7}
    out["leak_test"] = {"random30": lsum(L30), "edge": lsum(LE), "detail_random30": L30, "detail_edge": LE,
                        "years_random30": pd.Series([r["day"][:4] for r in L30]).value_counts().sort_index().to_dict(),
                        "sides_random30": pd.Series([r["side"] for r in L30]).value_counts().to_dict()}
    log("미래 참조", json.dumps(out["leak_test"]["random30"]), json.dumps(out["leak_test"]["edge"]))

    # ---------- 3. 분포 상식 점검 + 경계값·자료 품질·검출력
    out["sanity"] = sanity(ev, meta)
    out["boundary"] = boundary_summary(W, CH, ev, RECS)
    out["data_quality"] = data_quality(W, ev)
    log("분포 점검 완료. 경계 근처", out["boundary"]["hits"], "표 BRIEF 반대편", out["boundary"]["table_vs_brief_wrong"],
        "float 반대편", out["boundary"]["float_wrong_side_total"], "EPS 잘못 맞춤", out["boundary"]["snap_misapplied(within_1e-9_but_exact_off)"])
    out["power"] = power_checks(W, ev, FA, MYC, rng)
    log("검출력", json.dumps(out["power"], ensure_ascii=False))
    with open(os.path.join(RES, "ev4_audit.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v))
    log("저장 results/ev4_audit.json")


def sanity(ev, meta):
    S = {}
    side = ev["side"].astype(str)
    S["rows"] = int(len(ev))
    S["dup_keys"] = int(ev.duplicated(["side", "x", "d", "dec_i"]).sum())
    S["n_by_side_x_year"] = {f"{s}_{x:g}": {str(y): int(((side == s) & (ev["x"] == x) & (ev["year"] == y)).sum())
                                             for y in (2023, 2024, 2025, 2026)} for s in SIDES for x in XS}
    S["n_matches_meta"] = all(S["n_by_side_x_year"][f"{s}_{x:g}"] == meta["n_events"][s][f"{x:g}"] for s in SIDES for x in XS)
    S["n_nested_by_threshold"] = {s: [int(((side == s) & (ev["x"] == x)).sum()) for x in XS] for s in SIDES}
    st = []
    for (s, x, y), g in ev.assign(side=side).groupby(["side", "x", "year"]):
        st.append({"side": s, "x": x, "year": int(y), "n": len(g), "days": int(g["d"].nunique()), "hit": round(float(g["hit"].mean()), 3),
                   "mean_r_w90": round(float(g["r_w90"].mean()), 4), "median_r_w90": round(float(g["r_w90"].median()), 4),
                   "mean_r_follow": round(float(g["r_follow"].mean()), 4), "hit_follow": round(float((g["why_follow"] == "target").mean()), 3),
                   "mean_mae": round(float(g["mae"].mean()), 3), "mean_mfe": round(float(g["mfe"].mean()), 3),
                   "stop_s2": round(float((g["why_s2"] == "stop").mean()), 3),
                   "mean_min_to_target": round(float(g["min_to_target"].mean()), 1) if g["hit"].any() else None})
    S["by_side_x_year"] = st
    inv = {}
    hit = ev["hit"].values == 1
    seq1 = ev["c28_seq"].values == 1
    inv["hit ≠ (why == target)"] = int((hit != (ev["why"].astype(str).values == "target")).sum())
    inv["적중인데 r_w90 ≠ 1.0"] = int((hit & (np.abs(ev["r_w90"].values - 1.0) > 1e-9)).sum())
    inv["r_w90 > 1.0202(창 끝 청산 상한 넘음)"] = int((ev["r_w90"].values > 1.0202 + 1e-9).sum())
    inv["r_follow > 1.0202"] = int((ev["r_follow"].values > 1.0202 + 1e-9).sum())
    inv["fill_i ≠ dec_i + 1"] = int((ev["fill_i"].values != ev["dec_i"].values + 1).sum())
    inv["dec_i ∉ [09:30, 10:44] 봉"] = int(((ev["dec_i"].values < B0930) | (ev["dec_i"].values > LAST_DEC)).sum())
    inv["exit_i ∉ [fill+1, 10:59 봉]"] = int(((ev["exit_i"].values < ev["fill_i"].values + 1) | (ev["exit_i"].values > B1100 - 1)).sum())
    inv["창 끝 청산인데 exit ≠ 10:59"] = int(((~hit) & (ev["exit_i"].values != B1100 - 1)).sum())
    inv["entry·exit·dec_close 표기 불일치"] = int(sum(1 for a, b, c, f, j, dd in zip(ev["entry"], ev["exit"], ev["dec_close"], ev["fill_i"],
                                                                         ev["exit_i"], ev["dec_i"])
                                                if a != clock(f) or b != clock(j) or c != clock(dd + 1)))
    inv["min_to_target ≠ exit − fill(적중)"] = int((hit & (ev["min_to_target"].values != ev["exit_i"].values - ev["fill_i"].values)).sum())
    inv["min_to_target 값 있음(미적중)"] = int(((~hit) & ev["min_to_target"].notna().values).sum())
    inv["mae > 0 또는 mfe < 0"] = int(((ev["mae"].values > 0) | (ev["mfe"].values < 0)).sum())
    inv["적중인데 mfe < 1.0202"] = int((hit & (ev["mfe"].values < (1.01 * 1.0002 - 1) * 100 - 1e-9)).sum())
    inv["미적중인데 mfe ≥ 1.0202"] = int(((~hit) & (ev["mfe"].values >= (1.01 * 1.0002 - 1) * 100 + 1e-9)).sum())
    inv["r_w90 ≠ exit_px ÷ entry_px − 1"] = int((np.abs((ev["exit_px"] / ev["entry_px"] - 1).values * 100 - ev["r_w90"].values) > 1e-9).sum())
    inv["ratio < 1 + x"] = int((ev["ratio"].values < 1 + ev["x"].values - 1e-12).sum())
    inv["ratio ≠ trig_close ÷ L"] = int((np.abs(ev["ratio"] - ev["trig_close"] / ev["L"]).values > 1e-12).sum())
    inv["L_i ∉ [s0_i, dec_i]"] = int(((ev["L_i"].values < ev["s0_i"].values) | (ev["L_i"].values > ev["dec_i"].values)).sum())
    inv["c02_speed ≠ dec_i − L_i"] = int((ev["c02_speed"].values != ev["dec_i"].values - ev["L_i"].values).sum())
    inv["r_std 있음 ⇔ dec ≤ 09:44 봉 어긋남"] = int((ev["r_std"].notna().values != (ev["dec_i"].values <= LAST_STD)).sum())
    inv["r_w90_c2 > r_w90"] = int((ev["r_w90_c2"].values > ev["r_w90"].values + 1e-9).sum())
    inv["why_s2 = stop인데 r_w90_s2 > −2%"] = int(((ev["why_s2"].astype(str).values == "stop") & (ev["r_w90_s2"].values > -2.0 + 1e-9)).sum())
    inv["순번 1인데 alt_* ≠ c17~c19"] = int((seq1 & ((ev["alt_c17_qqq_0930"].values != ev["c17_qqq"].values)
                                                 | (ev["alt_c18_nvda_0930"].values != ev["c18_nvda"].values)
                                                 | (ev["alt_c19_soxx_0930"].values != ev["c19_soxx"].values))).sum())
    inv["순번 1인데 s0 ≠ 09:30 봉"] = int((seq1 & (ev["s0_i"].values != B0930)).sum())
    inv["c06 NaN ⇔ dec < 09:34 봉 어긋남"] = int((np.isnan(ev["c06_first5"].values) != (ev["dec_i"].values < B0934)).sum())
    inv["c05 ≠ soxl_gap × 면 부호"] = int((np.abs(ev["c05_gap"].values - ev["soxl_gap"].values * np.where(side.values == "user", 1, -1)) > 1e-12).sum())
    inv["c09 = 1인데 c13 = 0(신고가 아님)"] = int(((ev["c09_rsidiv"].values == 1) & (ev["c13_newhighs"].values == 0)).sum())
    g = ev.assign(side=side).sort_values(["side", "x", "d", "dec_i"])
    seq_exp = g.groupby(["side", "x", "d"]).cumcount().values + 1
    inv["c28_seq ≠ 같은 날 순번"] = int((g["c28_seq"].values != seq_exp).sum())
    prev_exit = g.groupby(["side", "x", "d"])["exit_i"].shift(1).values
    first = np.isnan(prev_exit)
    inv["s0_i ≠ (첫 사건 09:30 / 앞 청산 + 1)"] = int(((first & (g["s0_i"].values != B0930))
                                                     | (~first & (g["s0_i"].values != prev_exit + 1))).sum())
    inv["dec_i ≤ 앞 사건 exit_i"] = int((~first & (g["dec_i"].values <= prev_exit)).sum())
    S["invariants"] = inv
    S["nan_counts"] = {c: int(ev[c].isna().sum()) for c in ev.columns if ev[c].isna().any()}
    S["nan_counts_meta_match"] = all(int(ev[c["col"] + "_lv"].isna().sum()) == c["null_events"] for c in meta["conditions"])
    num = [c for c in ev.columns if c.startswith("c") and not c.endswith("_lv") and c[1:3].isdigit()] + ALT
    S["inf_counts"] = {c: int(np.isinf(ev[c].values.astype(float)).sum()) for c in num if np.isinf(ev[c].values.astype(float)).any()}
    S["cond_ranges_min_p1_med_p99_max"] = {c: [round(float(np.nanmin(ev[c])), 4), round(float(np.nanpercentile(ev[c], 1)), 4),
                                               round(float(np.nanmedian(ev[c])), 4), round(float(np.nanpercentile(ev[c], 99)), 4),
                                               round(float(np.nanmax(ev[c])), 4)] for c in num}
    S["level_counts"] = {c: {str(k): int(v) for k, v in ev[c + "_lv"].value_counts(dropna=False).sort_index().items()}
                         for c in num if c + "_lv" in ev.columns}
    # 2026 사용자 x=3 하루 평균(207일, 안 산 날 0) — report3 기준 +0.0775와 같은지
    e3 = ev[(side == "user") & (ev["x"] == 0.03) & (ev["year"] == 2026)]
    per = e3.groupby("d")["r_w90"].apply(lambda r: (np.prod(1 + r.values / 100) - 1) * 100)
    S["user_x3_2026"] = {"events": int(len(e3)), "active_days": int(per.size), "mean_per_event": round(float(e3["r_w90"].mean()), 4),
                         "mean_daily_207": round(float(per.sum() / 207), 4), "hit": round(float(e3["hit"].mean()), 3)}
    # 2026 면 × 문턱 요약(C1 자격 판단에 쓰는 표본 크기)
    S["y2026_side_x"] = [{"side": s, "x": x, "n": int(len(g)), "days": int(g["d"].nunique()), "hit": round(float(g["hit"].mean()), 3),
                          "mean_r_w90": round(float(g["r_w90"].mean()), 4), "mean_r_follow": round(float(g["r_follow"].mean()), 4)}
                         for (s, x), g in ev[ev["year"] == 2026].assign(side=side).groupby(["side", "x"])]
    return S


if __name__ == "__main__":
    main()
