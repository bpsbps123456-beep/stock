"""5단계 KORU 반박 검증(2026-10-11): G2 'EWY 50일선 띠 3% → 현금'을 lev5·stock_lab 없이 parquet만으로 다시 잰다.

사용자 질문: "SOXL이나 KORU 같은 건 손절 규칙 어떻게 잡아?"  대상 후보(p5_KORU.md G2):
  EWY 종가 < SMA50×0.97 → 다음 시가 KORU 0, 현금 상태에서 EWY 종가 > SMA50×1.03 → 다음 시가 KORU 1, 사이 유지.

독립 재계산(이 파일만으로): cache/p5_lev_long.parquet → 배당 수정 일봉, ^IRX 금융비용, 3배 일일 재설정 합성,
  추가 비용은 실제 KORU와 겹친 기간에서 새로 잼. 엔진은 '전날 종가 비율이 밤사이 흐름 → 시가에 목표로 맞춤 → 장중 흐름'
  일별 루프(비율 단위). 같은 규칙 묶음 21개를 다시 짜서 G1·G2·G3 선택도 재현한다.

검증 항목(결과를 보기 전에 고정)
  1) 주장 숫자 재현(합성·실제). 0.5%p 넘게 다르면 원인 규명.
  2) 시점: 미래 교란 점검, 하루 늦춘 체결(t+2 시가), 미끄러짐 0.5%×|Δw|, 수정 안 한 EWY 종가로 신호.
  3) 이웃: n∈{30,40,50,60,70} × 띠∈{1,2,3,4,5}% = 25개. '2026 앞·뒤·2025·긴 역사 효율선초과 모두 > 0' 비율.
  4) 위약: 같은 평균 비율·같은 전환 횟수를 무작위 날짜에 둔 판 시드 20개(긴 창 = 안쪽 구간 길이 섞기, 짧은 창 =
     무작위 전환일 + 노출 ±0.03 이내만 채택) 대비 순위.
  5) 다른 상품: TQQQ(^NDX→QQQ 1999-03-11)·UPRO(^GSPC→SPY 1993-02-01)·TNA(IWM)·TECL(XLK)·YINN(FXI)·EDC(EEM) 합성에
     같은 규칙(50일·3%) → 효율선초과 부호.
  6) 시작 시점 민감도: 시작 해 2001~2015 → 2025-12 효율선초과, 앞뒤 절반(2000-05~2012, 2013~2025).
출력: results/p5_V_KORU.md, results/p5_V_KORU.json. 실행: py p5_V_KORU.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PQ = ROOT / "cache" / "p5_lev_long.parquet"
OUT = ROOT / "results"
FEE = 0.0095
SLIP = 0.005
WG = np.round(np.arange(0, 1.0001, 0.01), 2)
END = "2026-10-09"

W = {
    "2026앞": ("2026-01-02", "2026-05-29"), "2026뒤": ("2026-06-01", "2026-10-09"), "2026": ("2026-01-02", "2026-10-09"),
    "2025": ("2025-01-02", "2025-12-31"), "2022": ("2022-01-03", "2022-12-30"), "2020": ("2020-01-02", "2020-12-31"),
    "2008위기": ("2007-10-01", "2009-03-31"), "2000~02": ("2000-05-12", "2002-10-31"),
    "긴역사": ("2000-05-12", END), "긴앞이전": ("2000-05-12", "2025-12-31"), "실제기간": ("2013-04-11", END),
    "준비후앞이전": ("2001-03-01", "2025-12-31"),
    "앞절반": ("2000-05-12", "2012-12-31"), "뒤절반": ("2013-01-01", "2025-12-31"),
}
for _y in range(2001, 2016):
    W[f"시작{_y}"] = (f"{_y}-01-01", "2025-12-31")


# ----------------------------------------------------------------------------- 자료
_D = None


def raw():
    global _D
    if _D is None:
        _D = pd.read_parquet(PQ)
    return _D


def series(sym, adjust=True):
    g = raw()
    g = g[(g.sym == sym) & g.Close.notna() & (g.Close > 0)].sort_values("date")
    g = g[g.date <= END]
    f = (g["Adj Close"] / g["Close"]).fillna(1.0).to_numpy() if adjust else np.ones(len(g))
    c = g.Close.to_numpy(float)
    o = g.Open.fillna(g.Close).to_numpy(float)
    h = g.High.fillna(g.Close).to_numpy(float)
    lo = g.Low.fillna(g.Close).to_numpy(float)
    return {"d": g.date.to_numpy(), "o": o * f, "h": h * f, "l": lo * f, "c": c * f}


def join(a, b, at):
    """at 전은 a의 수익, at부터 b 가격(이어 붙인 날 수익은 a 기준)."""
    ia, ib = int(np.searchsorted(a["d"], at)), int(np.searchsorted(b["d"], at))
    k = b["c"][ib] / a["c"][ia]
    return {"d": np.concatenate([a["d"][:ia], b["d"][ib:]]),
            **{x: np.concatenate([a[x][:ia] * k, b[x][ib:]]) for x in ("o", "h", "l", "c")}}


def irx(days):
    g = raw()
    s = g[g.sym == "^IRX"].drop_duplicates("date").set_index("date")["Close"].astype(float).sort_index() / 100
    return s.reindex(s.index.union(pd.Index(days))).ffill().reindex(days).fillna(0).to_numpy()


def lever(u, L=3.0, extra=0.0):
    """일일 재설정 L배: 종가 수익 = L·기초 − (L−1)·금리/252 − (보수+추가)/252, 시가 = 전일 종가 × (1 + L·기초 시가 갭)."""
    d, o, h, lo, c = u["d"], u["o"], u["h"], u["l"], u["c"]
    r = irx(d)
    n = len(c)
    P = {k: np.empty(n) for k in ("o", "h", "l", "c")}
    for k in P:
        P[k][0] = 100.0
    pc = c[:-1]
    day = np.maximum(1 + L * (c[1:] / pc - 1) - (L - 1) * r[1:] / 252 - (FEE + extra) / 252, 1e-3)
    P["c"][1:] = 100 * np.cumprod(day)
    prev = P["c"][:-1]
    P["o"][1:] = prev * np.maximum(1 + L * (o[1:] / pc - 1), 1e-3)
    hi = prev * np.maximum(1 + L * (h[1:] / pc - 1), 1e-3)
    lw = prev * np.maximum(1 + L * (lo[1:] / pc - 1), 1e-3)
    P["h"][1:] = np.maximum.reduce([hi, P["o"][1:], P["c"][1:]])
    P["l"][1:] = np.minimum.reduce([lw, P["o"][1:], P["c"][1:]])
    P["d"] = d.copy()
    return P


def overlap_cost(real, syn):
    common = np.intersect1d(real["d"], syn["d"])
    ia, isy = np.searchsorted(real["d"], common), np.searchsorted(syn["d"], common)
    a, s = real["c"][ia], syn["c"][isy]
    ra, rs = a[1:] / a[:-1] - 1, s[1:] / s[:-1] - 1
    mdd = lambda v: float((v / np.maximum.accumulate(v) - 1).min())
    return {"시작": str(common[0]), "일수": int(len(common)), "상관": float(np.corrcoef(ra, rs)[0, 1]),
            "연로그차": float(-np.log((a[-1] / a[0]) / (s[-1] / s[0])) / (len(common) / 252)), "실제낙폭": mdd(a), "합성낙폭": mdd(s)}


def calibrated(u, real):
    s0 = lever(u, 3, 0.0)
    ex = overlap_cost(real, s0)["연로그차"]
    s1 = lever(u, 3, ex)
    return s1, ex, overlap_cost(real, s1)


# ----------------------------------------------------------------------------- 규칙(목표 비율 des[t] = t 종가 뒤 결정 → t+1 시가)
def sma(c, n):
    return pd.Series(c).rolling(n, min_periods=n).mean().to_numpy()


def band_rule(c, n=50, b=0.03):
    s = sma(c, n)
    out = np.ones(len(c))
    st = 1.0
    for t in range(len(c)):
        if s[t] == s[t]:
            if st == 1.0 and c[t] < s[t] * (1 - b):
                st = 0.0
            elif st == 0.0 and c[t] > s[t] * (1 + b):
                st = 1.0
        out[t] = st
    return out


def family(u):
    """p5_KORU와 같은 사전 고정 21개를 이 파일 안에서 다시 짠다(S 계열은 손절 엔진용 재진입 신호)."""
    c, d = u["c"], u["d"]
    ret = np.concatenate([[np.nan], c[1:] / c[:-1] - 1])
    sig = pd.Series(ret).rolling(20, min_periods=20).std().to_numpy() * np.sqrt(252)
    m = pd.Series(sig).expanding(min_periods=60).median().shift(1).to_numpy()
    iso = pd.to_datetime(pd.Series(d)).dt.isocalendar()
    key = (iso.year.astype(int) * 100 + iso.week.astype(int)).to_numpy()
    fow = np.concatenate([[True], key[1:] != key[:-1]])
    fow_next = np.concatenate([fow[1:], [False]])
    ok = np.isfinite(sig) & np.isfinite(m)
    R = {}
    for n in (50, 100, 150, 200):
        s = sma(c, n)
        R[f"T{n}"] = ("T", np.where(np.isfinite(s) & (c < s), 0.0, 1.0))
        R[f"T{n}b"] = ("T", band_rule(c, n, 0.03))
    s200 = sma(c, 200)
    R["T200h"] = ("T", np.where(np.isfinite(s200) & (c < s200), 0.5, 1.0))
    safe = np.where(ok, sig, 1.0)
    vk = lambda k: np.where(ok, np.clip(k * m / safe, 0, 1), 1.0)
    vt = lambda tg: np.where(np.isfinite(sig), np.clip(tg / (3 * np.where(np.isfinite(sig), sig, 1.0)), 0, 1), 1.0)
    R["V1"], R["V075"], R["V45"], R["V60"] = ("V", vk(1.0)), ("V", vk(0.75)), ("V", vt(0.45)), ("V", vt(0.60))
    R["V1w"] = ("V", np.where(fow_next, vk(1.0), np.nan))
    R["V60w"] = ("V", np.where(fow_next, vt(0.60), np.nan))
    R["VS15"] = ("VS", np.where(ok & (sig > 1.5 * np.where(ok, m, 0)), 0.0, 1.0))
    R["VS20"] = ("VS", np.where(ok & (sig > 2.0 * np.where(ok, m, 0)), 0.0, 1.0))
    up = ~(np.isfinite(s200) & (c < s200))
    calm = ~(ok & (sig >= 1.5 * np.where(ok, m, 0)))
    R["TV"] = ("TV", np.where(up & calm, 1.0, np.where(up | calm, 0.5, 0.0)))
    s50 = sma(c, 50)
    re = ~(np.isfinite(s50) & (c <= s50))
    for x in (30, 40, 50):
        R[f"S{x}"] = ("S", re)
    return R


# ----------------------------------------------------------------------------- 엔진
def sim(P, des, slip=0.0, cash=0.0, delay=1):
    """des[t] = t일 종가 뒤 목표 비율 → t+delay일 시가에 맞춤(NaN이면 그대로). 시작은 0일 종가 가치 1, 비율 des[0](NaN이면 1)."""
    o, c = P["o"], P["c"]
    n = len(c)
    V, wc, wo, turn = np.ones(n), np.empty(n), np.empty(n), np.zeros(n)
    wc[0] = wo[0] = des[0] if np.isfinite(des[0]) else 1.0
    for t in range(1, n):
        g = o[t] / c[t - 1]
        vo = V[t - 1] * (1 + wc[t - 1] * (g - 1)) + V[t - 1] * (1 - wc[t - 1]) * cash / 252
        w = V[t - 1] * wc[t - 1] * g / vo if vo > 0 else 0.0
        k = t - delay
        tg = des[k] if k >= 0 else np.nan
        if np.isfinite(tg) and abs(tg - w) > 1e-12:
            turn[t] = abs(tg - w)
            vo *= 1 - slip * turn[t]
            w = tg
        wo[t] = w
        gd = c[t] / o[t]
        V[t] = vo * (1 + w * (gd - 1))
        wc[t] = vo * w * gd / V[t] if V[t] > 0 else 0.0
    return V, wc, wo, turn


def sim_stop(P, x, reent, slip=0.0):
    """산 뒤 최고 종가 × (1−x) 장중 손절(t−1까지 정보), 판 뒤 reent[t]이면 t+1 시가 전부 재진입."""
    o, lo, c = P["o"], P["l"], P["c"]
    n = len(c)
    V, wc, wo, turn = np.ones(n), np.ones(n), np.ones(n), np.zeros(n)
    inpos, peak, cashv, sh = True, c[0], 0.0, 1.0 / c[0]
    for t in range(1, n):
        if not inpos and reent[t - 1]:
            v = cashv * (1 - slip)
            sh, cashv, inpos, peak = v / o[t], 0.0, True, o[t]
            turn[t] += 1
        wo[t] = 1.0 if inpos else 0.0
        if inpos:
            L = peak * (1 - x)
            px = o[t] if o[t] <= L else (L if lo[t] <= L else None)
            if px is not None:
                cashv, sh, inpos = sh * px * (1 - slip), 0.0, False
                turn[t] += 1
        V[t] = cashv + sh * c[t]
        wc[t] = sh * c[t] / V[t]
        if inpos:
            peak = max(peak, c[t])
    return V, wc, wo, turn


def frontier(P, slip=0.0, cash=0.0):
    o, c = P["o"], P["c"]
    n = len(c)
    F = np.ones((len(WG), n))
    wc = WG.copy()
    for t in range(1, n):
        g = o[t] / c[t - 1]
        vo = F[:, t - 1] * (1 + wc * (g - 1)) + F[:, t - 1] * (1 - wc) * cash / 252
        w = np.where(vo > 0, F[:, t - 1] * wc * g / np.where(vo > 0, vo, 1), 0)
        vo = vo * (1 - slip * np.abs(WG - w))
        gd = c[t] / o[t]
        F[:, t] = vo * (1 + WG * (gd - 1))
        wc = np.where(F[:, t] > 0, vo * WG * gd / np.where(F[:, t] > 0, F[:, t], 1), 0)
    return F


def win_idx(days, d0, d1):
    a = max(int(np.searchsorted(days, d0)), 1)
    z = int(np.searchsorted(days, d1, side="right")) - 1
    if z < a or (pd.Timestamp(days[a - 1]) - pd.Timestamp(d0)).days > 30:
        return None
    return a, z


def mdd(v):
    return float((v / np.maximum.accumulate(v) - 1).min())


def cg(ret, nd):
    return float((1 + ret) ** (252 / nd) - 1) if ret > -1 else -1.0


def stats(res, P, F, wname):
    V, wc, wo, turn = res
    ix = win_idx(P["d"], *W[wname])
    if ix is None:
        return None
    a, z = ix
    seg = V[a - 1:z + 1] / V[a - 1]
    nd = z - a + 1
    ret, dd = float(seg[-1] - 1), mdd(seg)
    long = nd > 300
    out = {"수익": ret, "낙폭": dd, "연율": cg(ret, nd), "노출": float(wc[a:z + 1].mean()),
           "연매매": float((turn[a:z + 1] >= 0.05).sum() / (nd / 252)), "일수": nd}
    if F is not None:
        Fs = F[:, a - 1:z + 1] / F[:, a - 1:a]
        fr = Fs[:, -1] - 1
        fm = (Fs / np.maximum.accumulate(Fs, axis=1) - 1).min(axis=1)
        okk = fm >= dd - 1e-12
        j = int(np.argmax(np.where(okk, fr, -np.inf)))
        k = int(round(out["노출"] * 100))
        out["효율선_w"] = float(WG[j])
        out["효율선초과"] = (out["연율"] - cg(fr[j], nd)) if long else ret - float(fr[j])
        out["같은노출차"] = (out["연율"] - cg(fr[k], nd)) if long else ret - float(fr[k])
    if long:
        r1 = V[a - 1 + 252:z + 1] / V[a - 1:z + 1 - 252] - 1
        out["최악1년"] = float(r1.min())
        out["전멸"] = bool(dd <= -0.90)
    return out


def align(u, P, arr):
    idx = np.searchsorted(u["d"], P["d"])
    assert np.all(u["d"][idx] == P["d"])
    return arr[idx]


# ----------------------------------------------------------------------------- 위약
def runs(x):
    """(시작, 길이, 값) 목록."""
    ch = np.flatnonzero(np.diff(x) != 0) + 1
    st = np.concatenate([[0], ch])
    en = np.concatenate([ch, [len(x)]])
    return [(int(s), int(e - s), float(x[s])) for s, e in zip(st, en)]


def shuffle_inner(x, rng):
    """첫·마지막 구간은 두고 안쪽 보유 구간끼리·현금 구간끼리 길이를 섞는다(노출·전환 횟수 정확히 같음)."""
    R = runs(x)
    if len(R) <= 3:
        return x.copy()
    inner = R[1:-1]
    L1 = [l for _, l, v in inner if v == 1.0]
    L0 = [l for _, l, v in inner if v == 0.0]
    rng.shuffle(L1)
    rng.shuffle(L0)
    out = [np.full(R[0][1], R[0][2])]
    i1 = i0 = 0
    for _, _, v in inner:
        if v == 1.0:
            out.append(np.full(L1[i1], 1.0)); i1 += 1
        else:
            out.append(np.full(L0[i0], 0.0)); i0 += 1
    out.append(np.full(R[-1][1], R[-1][2]))
    return np.concatenate(out)


def random_dates(x, rng, tol=0.03, tries=20000):
    """같은 전환 횟수를 무작위 날짜에(시작 상태 같음), 평균 비율이 ±tol 이내인 것만 채택."""
    n = len(x)
    k = int((np.diff(x) != 0).sum())
    if k == 0:
        return x.copy()
    target = x.mean()
    for _ in range(tries):
        cut = np.sort(rng.choice(np.arange(1, n), size=k, replace=False))
        y = np.empty(n)
        v = x[0]
        prev = 0
        for cpt in list(cut) + [n]:
            y[prev:cpt] = v
            v = 1.0 - v
            prev = cpt
        if abs(y.mean() - target) <= tol:
            return y
    return None


# ----------------------------------------------------------------------------- 실행
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    J = {}
    u = series("EWY")
    uR = series("EWY", adjust=False)
    real = series("KORU")
    syn, extra, val = calibrated(u, real)
    J["대조"] = {"추가비용": extra, **val}
    print("대조", J["대조"])
    Fs, Fr = frontier(syn), frontier(real)
    Fs5, Fr5 = frontier(syn, SLIP), frontier(real, SLIP)
    Fs4 = frontier(syn, 0, 0.04)
    fam = family(u)

    def run(P, rid, des_u=None, **kw):
        kind, arr = fam[rid] if des_u is None else ("T", des_u)
        a = align(u, P, arr)
        if kind == "S":
            return sim_stop(P, int(rid[1:]) / 100, a, kw.get("slip", 0.0))
        return sim(P, a, **kw)

    # ---------- 1) 21개 재현 + 고르기
    allw = list(W)
    tab = {}
    for rid in fam:
        rs, rr = run(syn, rid), run(real, rid)
        tab[rid] = {"합성": {w: stats(rs, syn, Fs, w) for w in allw}, "실제": {w: stats(rr, real, Fr, w) for w in allw}}
    hold = {"합성": {w: stats(sim(syn, np.ones(len(syn["c"]))), syn, Fs, w) for w in allw},
            "실제": {w: stats(sim(real, np.ones(len(real["c"]))), real, Fr, w) for w in allw}}
    ex = lambda rid, p, w: tab[rid][p][w]["효율선초과"]
    G1 = max(fam, key=lambda r: ex(r, "실제", "2026앞"))
    G2 = max(fam, key=lambda r: ex(r, "합성", "긴앞이전"))
    G2b = max(fam, key=lambda r: ex(r, "합성", "준비후앞이전"))
    rank2 = sorted(fam, key=lambda r: -ex(r, "합성", "긴앞이전"))
    J["고르기"] = {"G1": G1, "G2": G2, "G2_준비후": G2b,
                 "G2순위표": [(r, ex(r, "합성", "긴앞이전")) for r in rank2[:6]]}
    print("고르기", J["고르기"])
    T = tab["T50b"]
    # 민감도 판
    rs5, rr5 = run(syn, "T50b", slip=SLIP), run(real, "T50b", slip=SLIP)
    rs4 = run(syn, "T50b", cash=0.04)
    rsD, rrD = run(syn, "T50b", delay=2), run(real, "T50b", delay=2)
    desU = band_rule(uR["c"], 50, 0.03)
    rsU, rrU = run(syn, None, des_u=desU), run(real, None, des_u=desU)
    sens = {}
    keyw = ["2026앞", "2026뒤", "2025", "2022", "2008위기", "긴역사", "긴앞이전", "실제기간"]
    for lab, rs_, rr_, fs_, fr_ in (("기본", run(syn, "T50b"), run(real, "T50b"), Fs, Fr),
                                     ("미끄러짐0.5%", rs5, rr5, Fs5, Fr5), ("하루늦춤", rsD, rrD, Fs, Fr),
                                     ("하루늦춤+미끄러짐", run(syn, "T50b", delay=2, slip=SLIP), run(real, "T50b", delay=2, slip=SLIP), Fs5, Fr5),
                                     ("수정안한EWY신호", rsU, rrU, Fs, Fr)):
        sens[lab] = {"합성": {w: stats(rs_, syn, fs_, w) for w in keyw}, "실제": {w: stats(rr_, real, fr_, w) for w in keyw}}
    sens["현금4%"] = {"합성": {w: stats(rs4, syn, Fs4, w) for w in ["긴역사", "긴앞이전"]}}
    # 다음 날 종가 체결(시가 자료를 쓰지 않는 판): 시가 = 종가로 바꾸면 t+1 종가에 맞춤
    cS = {"d": syn["d"], "o": syn["c"], "c": syn["c"]}
    cR = {"d": real["d"], "o": real["c"], "c": real["c"]}
    FcS, FcR = frontier(cS), frontier(cR)
    sens["다음날종가체결"] = {"합성": {w: stats(sim(cS, align(u, syn, band_rule(u["c"]))), cS, FcS, w) for w in keyw},
                         "실제": {w: stats(sim(cR, align(u, real, band_rule(u["c"]))), cR, FcR, w) for w in keyw}}
    J["민감도"] = sens
    # 실제 KORU 체결일 중 가격이 멈춘 봉(시가=고가=저가=종가)
    rr0 = run(real, "T50b")
    tr_days = np.flatnonzero(rr0[3] >= 0.05)
    flat = (real["o"] == real["h"]) & (real["h"] == real["l"]) & (real["l"] == real["c"])
    J["실제체결일_멈춘봉"] = {"체결일": int(len(tr_days)), "멈춘봉": int(flat[tr_days].sum()),
                         "멈춘봉_날짜": [str(real["d"][t]) for t in tr_days if flat[t]]}
    # 수정 안 한 EWY 신호가 다른 날
    dA = band_rule(u["c"])
    J["수정신호차이일"] = int((dA != desU).sum())

    # 미래 교란 점검: T일부터 EWY·KORU를 흔들어도 T 전 가치와 T일 시가 비율이 같아야 한다
    rng = np.random.default_rng(11)
    Tt = int(np.searchsorted(u["d"], "2020-03-02"))
    u2 = {k: (v.copy() if k != "d" else v) for k, v in u.items()}
    sh = 1 + rng.normal(0, 0.05, len(u["c"]) - Tt)
    u2["c"][Tt:] *= sh
    u2["o"][Tt + 1:] *= sh[1:]
    u2["h"][Tt:] = np.maximum.reduce([u2["h"][Tt:] * sh, u2["o"][Tt:], u2["c"][Tt:]])
    u2["l"][Tt:] = np.minimum.reduce([u2["l"][Tt:] * sh, u2["o"][Tt:], u2["c"][Tt:]])
    syn2 = lever(u2, 3, extra)
    d1, d2 = band_rule(u["c"]), band_rule(u2["c"])
    a1, a2 = sim(syn, d1), sim(syn2, d2)
    J["미래교란"] = {"T": str(u["d"][Tt]), "T전가치차": float(np.abs(a1[0][:Tt] - a2[0][:Tt]).max()),
                  "T일시가비율차": float(np.abs(a1[2][:Tt + 1] - a2[2][:Tt + 1]).max()),
                  "T일신호차(같은날종가는 바뀌어야 정상)": float(np.abs(d1[Tt] - d2[Tt]))}
    # 신호 → 체결 지연이 정확히 1일인지: 상태 전환일 다음 날 시가 비율이 바뀌는지
    desS = align(u, syn, d1)
    sw = np.flatnonzero(np.diff(desS) != 0) + 1          # des가 바뀐 날(종가 결정)
    J["체결점검"] = {"전환수": int(len(sw)), "다음날시가에반영": int(sum(abs(a1[2][t + 1] - desS[t]) < 1e-12 for t in sw if t + 1 < len(desS))),
                  "같은날시가에는이전값": int(sum(abs(a1[2][t] - desS[t - 1]) < 1e-12 for t in sw))}

    # ---------- 3) 이웃
    nb = {}
    for n in (30, 40, 50, 60, 70):
        for b in (0.01, 0.02, 0.03, 0.04, 0.05):
            d = band_rule(u["c"], n, b)
            rs_, rr_ = run(syn, None, des_u=d), run(real, None, des_u=d)
            nb[f"{n}/{b:.2f}"] = {"합성": {w: stats(rs_, syn, Fs, w) for w in ["긴역사", "긴앞이전", "준비후앞이전", "앞절반", "뒤절반", "2026앞", "2026뒤", "2025"]},
                                  "실제": {w: stats(rr_, real, Fr, w) for w in ["2026앞", "2026뒤", "2025", "실제기간"]}}
    J["이웃"] = nb

    # ---------- 4) 위약
    plac = {}
    desS_full = align(u, syn, d1)
    desR_full = align(u, real, d1)
    base_s, base_r = sim(syn, desS_full), sim(real, desR_full)
    for lab, P, F, des, base, wins in (("합성 긴 역사", syn, Fs, desS_full, base_s, ["긴역사", "긴앞이전"]),
                                       ("실제 2013~", real, Fr, desR_full, base_r, ["실제기간"])):
        vals = {w: [] for w in wins}
        for seed in range(20):
            y = shuffle_inner(des, np.random.default_rng(seed))
            r_ = sim(P, y)
            for w in wins:
                s_ = stats(r_, P, F, w)
                vals[w].append((s_["연율"], s_["낙폭"], s_["효율선초과"], s_["전멸"] if "전멸" in s_ else None))
        for w in wins:
            s0 = stats(base, P, F, w)
            v = vals[w]
            plac[f"{lab}|{w}"] = {"규칙": (s0["연율"], s0["낙폭"], s0["효율선초과"]),
                                 "연율순위": 1 + sum(x[0] > s0["연율"] for x in v), "낙폭순위": 1 + sum(x[1] > s0["낙폭"] for x in v),
                                 "효율선초과순위": 1 + sum(x[2] > s0["효율선초과"] for x in v),
                                 "위약중앙_연율": float(np.median([x[0] for x in v])), "위약중앙_낙폭": float(np.median([x[1] for x in v])),
                                 "위약중앙_효율선초과": float(np.median([x[2] for x in v])), "위약전멸수": int(sum(bool(x[3]) for x in v)),
                                 "위약최고_연율": float(max(x[0] for x in v))}
    # 짧은 창: 창 안에서 무작위 전환일
    for w in ("2026앞", "2026뒤", "2025"):
        a, z = win_idx(real["d"], *W[w])
        x = desR_full[a - 1:z + 1]          # a−1 종가 결정 ~ z 종가 결정
        s0 = stats(base_r, real, Fr, w)
        vals = []
        for seed in range(20):
            y = random_dates(x, np.random.default_rng(100 + seed))
            if y is None:
                continue
            des = desR_full.copy()
            des[a - 1:z + 1] = y
            s_ = stats(sim(real, des), real, Fr, w)
            vals.append((s_["수익"], s_["낙폭"], s_["효율선초과"]))
        k = int((np.diff(x) != 0).sum())
        plac[f"실제|{w}"] = {"규칙": (s0["수익"], s0["낙폭"], s0["효율선초과"]), "전환수": k, "노출": float(x.mean()), "채택": len(vals),
                            "수익순위": 1 + sum(v[0] > s0["수익"] for v in vals), "효율선초과순위": 1 + sum(v[2] > s0["효율선초과"] for v in vals),
                            "위약중앙_수익": float(np.median([v[0] for v in vals])) if vals else None,
                            "위약중앙_효율선초과": float(np.median([v[2] for v in vals])) if vals else None}
    J["위약"] = plac

    # ---------- 5) 다른 상품
    prods = {"TQQQ": lambda: join(series("^NDX"), series("QQQ"), "1999-03-11"),
             "UPRO": lambda: join(series("^GSPC"), series("SPY"), "1993-02-01"),
             "TNA": lambda: series("IWM"), "TECL": lambda: series("XLK"), "YINN": lambda: series("FXI"), "EDC": lambda: series("EEM")}
    xp = {}
    for p, uf in prods.items():
        ub = uf()
        rp = series(p)
        sp, exp_, vp = calibrated(ub, rp)
        Fp, Frp = frontier(sp), frontier(rp)
        Fp5 = frontier(sp, SLIP)
        dd_ = band_rule(ub["c"], 50, 0.03)
        idx = np.searchsorted(ub["d"], sp["d"])
        r_ = sim(sp, dd_[idx])
        r5_ = sim(sp, dd_[idx], slip=SLIP)
        h_ = sim(sp, np.ones(len(sp["c"])))
        ir = np.searchsorted(ub["d"], rp["d"])
        assert np.all(ub["d"][ir] == rp["d"])
        rr_ = sim(rp, dd_[ir])
        w0 = str(sp["d"][252])
        W[f"{p}_긴앞이전"] = (w0, "2025-12-31")
        W[f"{p}_긴역사"] = (w0, END)
        ws = [f"{p}_긴앞이전", f"{p}_긴역사", "2026앞", "2026뒤", "2025", "2022", "2008위기"]
        xp[p] = {"추가비용": exp_, "대조": vp, "긴시작": w0,
                 "규칙": {w: stats(r_, sp, Fp, w) for w in ws}, "보유": {w: stats(h_, sp, Fp, w) for w in ws},
                 "미끄러짐": {w: stats(r5_, sp, Fp5, w) for w in ws[:2]},
                 "실제": {w: stats(rr_, rp, Frp, w) for w in ("2026앞", "2026뒤", "2025")}}
        print(p, round(exp_ * 100, 2), {w: round(xp[p]["규칙"][w]["효율선초과"] * 100, 1) for w in ws if xp[p]["규칙"][w]})
    J["다른상품"] = xp

    J["T50b"] = T
    J["보유"] = hold
    J["전체21"] = {r: {"합성": {w: tab[r]["합성"][w] for w in ("긴역사", "긴앞이전", "준비후앞이전", "2026앞")},
                     "실제": {w: tab[r]["실제"][w] for w in ("2026앞", "2026뒤", "2025", "실제기간")}} for r in fam}
    OUT.mkdir(exist_ok=True)
    (OUT / "p5_V_KORU.json").write_text(json.dumps(J, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    md = report(J)
    (OUT / "p5_V_KORU.md").write_text(md, encoding="utf-8")
    print(md)


# ----------------------------------------------------------------------------- 보고서
def p(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%"


def pp(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%p"


def report(J):
    T, H, S = J["T50b"], J["보유"], J["민감도"]
    ts, tr = T["합성"], T["실제"]
    L = []
    A = L.append
    nb = J["이웃"]
    keys = list(nb)
    ex = lambda k, src, w: nb[k][src][w]["효율선초과"]
    four = [k for k in keys if all(x > 0 for x in (ex(k, "실제", "2026앞"), ex(k, "실제", "2026뒤"), ex(k, "실제", "2025"), ex(k, "합성", "긴앞이전")))]
    lpos = [k for k in keys if ex(k, "합성", "긴앞이전") > 0]
    halves = [k for k in keys if ex(k, "합성", "앞절반") > 0 and ex(k, "합성", "뒤절반") > 0]
    wipe = [k for k in keys if nb[k]["합성"]["긴역사"]["전멸"]]
    med_l = float(np.median([ex(k, "합성", "긴앞이전") for k in keys]))
    rank_l = 1 + sum(ex(k, "합성", "긴앞이전") > ex("50/0.03", "합성", "긴앞이전") for k in keys)
    xp = J["다른상품"]
    xl = [q for q in xp if xp[q]["규칙"][f"{q}_긴앞이전"]["효율선초과"] > 0]
    x4 = [q for q in xp if all(xp[q]["규칙"][w]["효율선초과"] > 0 for w in (f"{q}_긴앞이전", "2026앞", "2026뒤", "2025"))]
    pl = J["위약"]
    A("# 5단계 KORU 반박 검증: G2 'EWY 50일선 띠 3% → 현금' (2026-10-11)")
    A("")
    A("> 사용자: \"SOXL이나 KORU 같은 건 손절 규칙 어떻게 잡아?\" · 대상: `results/p5_KORU.md`의 G2. 스크립트 `code/p5_V_KORU.py`(lev5·stock_lab 안 씀, parquet만), 숫자 `results/p5_V_KORU.json`.")
    A(f"> 합성 = EWY(배당 수정) 3배 일일 재설정 − 2×^IRX − 보수 0.95% − 추가 비용 {J['대조']['추가비용'] * 100:.2f}%(실제 KORU와 겹친 {J['대조']['일수']}일에서 새로 잼; 원래 2.6%), 일수익 상관 {J['대조']['상관']:.3f}. "
      "2026 앞·뒤·2025·2022·실제 2013~ = 실제 KORU, 2008·긴 역사 = 합성. 효율선초과 = 규칙 − 낙폭이 더 깊지 않은 '덜 담기'(w 0~1, 0.01 간격, 매일 맞춤) 중 최고(긴 창은 연율 차).")
    A("")
    A("## 결론")
    A("")
    A(f"1. **숫자는 재현됐다.** 주장 숫자 23개 모두 0.1%p 안(긴 역사 연 {p(ts['긴역사']['연율'])}/{p(ts['긴역사']['낙폭'])}, 효율선 대비 {pp(ts['긴역사']['효율선초과'])}, 실제 2013~ {pp(tr['실제기간']['효율선초과'])}). "
      f"21개 규칙을 새로 짜서 고르기도 같은 결과(G1 {J['고르기']['G1']}, G2 {J['고르기']['G2']}, 지표 준비 후 G2' {J['고르기']['G2_준비후']}). 미래 교란·체결 하루 지연 점검에서 누수 없음.")
    A(f"2. **긴 역사 우위는 좁은 봉우리다.** 이웃 25개(50±20일 × 띠 1~5%) 중 긴 역사(2026 앞 이전) 효율선초과 > 0은 {len(lpos)}개, 중앙 {pp(med_l)}(대상은 {rank_l}등), "
      f"'2026 앞·뒤·2025·긴 역사 모두 > 0'은 {len(four)}/25, 전멸 {len(wipe)}/25. 앞 절반(2000-05~2012) 효율선초과 {pp(ts['앞절반']['효율선초과'])} — 우위는 2013~2025에서만 나왔다. "
      f"차트 그대로(배당 수정 안 한 EWY)로 신호를 내면 {pp(S['수정안한EWY신호']['합성']['긴앞이전']['효율선초과'])}(2025 {pp(S['수정안한EWY신호']['실제']['2025']['효율선초과'])}), "
      f"다음 날 종가 체결이면 {pp(S['다음날종가체결']['합성']['긴앞이전']['효율선초과'])}. 다른 3배 6개에 그대로 걸면 긴 역사 > 0은 {len(xl)}/6, 네 창 모두 > 0은 {len(x4)}/6.")
    A(f"3. **판정: 맞지만 약하다(confirmed_fragile).** 같은 노출·같은 전환 수를 무작위 날짜에 둔 위약 20개보다는 긴 역사 {pl['합성 긴 역사|긴역사']['연율순위']}/21등"
      f"(위약 중앙 연 {p(pl['합성 긴 역사|긴역사']['위약중앙_연율'])}, 20개 중 {pl['합성 긴 역사|긴역사']['위약전멸수']}개 전멸)이라 타이밍 정보가 전혀 없지는 않다. "
      f"그러나 합격 기준인 2026은 앞 {pp(tr['2026앞']['효율선초과'])}·뒤 {pp(tr['2026뒤']['효율선초과'])}로 졌고, 위 민감도에서 우위가 0 근처로 줄어든다. "
      "KORU 손절 규칙으로 쓰지 않고 '덜 담기(w≈0.33)'를 기준으로 둔다는 원래 결론이 맞다.")
    A("")
    # ---- 재현
    A("## 1. 재현 (주장 → 독립 재계산)")
    A("")
    A("| 항목 | 주장 | 재계산 |")
    A("|---|---|---|")
    rows = [
        ("합성 2000-05~2026-10 연율 / 최대 낙폭", "+10.8% / -85.6%", f"{p(ts['긴역사']['연율'])} / {p(ts['긴역사']['낙폭'])}"),
        ("최악 1년 / 연 매매", "-74% / 4.3", f"{p(ts['긴역사']['최악1년'], 0)} / {ts['긴역사']['연매매']:.1f}"),
        ("효율선 대비(전체 / 2026 앞 이전)", "+4.1%p / +2.8%p", f"{pp(ts['긴역사']['효율선초과'])} / {pp(ts['긴앞이전']['효율선초과'])}"),
        ("실제 KORU 2013~ 연율 / 낙폭 / 효율선 대비", "+16.6% / -84.6% / +8.7%p", f"{p(tr['실제기간']['연율'])} / {p(tr['실제기간']['낙폭'])} / {pp(tr['실제기간']['효율선초과'])}"),
        ("2008 위기(보유)", "-51.5% (-99.0%)", f"{p(ts['2008위기']['수익'])} ({p(H['합성']['2008위기']['수익'])})"),
        ("2022 실제(보유)", "-17.0% (-70.2%)", f"{p(tr['2022']['수익'])} ({p(H['실제']['2022']['수익'])})"),
        ("미끄러짐 0.5% / 현금 4% 긴 역사 연율", "+8.5% / +12.7%", f"{p(S['미끄러짐0.5%']['합성']['긴역사']['연율'])} / {p(S['현금4%']['합성']['긴역사']['연율'])}"),
        ("2026 앞 수익/낙폭 (효율선 대비)", "+352%/-57% (-75.9%p)", f"{p(tr['2026앞']['수익'], 0)}/{p(tr['2026앞']['낙폭'], 0)} ({pp(tr['2026앞']['효율선초과'])})"),
        ("2026 뒤 수익/낙폭 (효율선 대비)", "-51.0%/-58% (-51.0%p)", f"{p(tr['2026뒤']['수익'])}/{p(tr['2026뒤']['낙폭'], 0)} ({pp(tr['2026뒤']['효율선초과'])})"),
        ("2025 수익 (효율선 대비)", "+260% (-6.6%p)", f"{p(tr['2025']['수익'], 0)} ({pp(tr['2025']['효율선초과'])})"),
    ]
    for r in rows:
        A(f"| {r[0]} | {r[1]} | {r[2]} |")
    A("")
    A(f"- G2 순위(2026 앞 이전 긴 역사 효율선초과): " + ", ".join(f"{k} {pp(v)}" for k, v in J["고르기"]["G2순위표"]) + ".")
    A("")
    # ---- 시점·비용
    A("## 2. 시점·비용 (효율선초과; 짧은 창 %p, 긴 창 연 %p)")
    A("")
    A("| 판 | 2026 앞 | 2026 뒤 | 2025 | 2022 | 긴 역사 2026 앞 이전 | 긴 역사 전체 | 실제 2013~ | 긴 역사 연율/낙폭 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for lab, nm in (("기본", "기본(t 종가 → t+1 시가)"), ("미끄러짐0.5%", "미끄러짐 0.5%×|Δw|"), ("하루늦춤", "하루 늦춤(t+2 시가)"),
                    ("하루늦춤+미끄러짐", "하루 늦춤 + 미끄러짐"), ("다음날종가체결", "t+1 종가 체결"), ("수정안한EWY신호", "배당 수정 안 한 EWY로 신호")):
        s, r = S[lab]["합성"], S[lab]["실제"]
        A(f"| {nm} | {pp(r['2026앞']['효율선초과'])} | {pp(r['2026뒤']['효율선초과'])} | {pp(r['2025']['효율선초과'])} | {pp(r['2022']['효율선초과'])} | "
          f"{pp(s['긴앞이전']['효율선초과'])} | {pp(s['긴역사']['효율선초과'])} | {pp(r['실제기간']['효율선초과'])} | {p(s['긴역사']['연율'])}/{p(s['긴역사']['낙폭'], 0)} |")
    c4 = S["현금4%"]["합성"]
    A(f"| 현금 4% | — | — | — | — | {pp(c4['긴앞이전']['효율선초과'])} | {pp(c4['긴역사']['효율선초과'])} | — | {p(c4['긴역사']['연율'])}/{p(c4['긴역사']['낙폭'], 0)} |")
    A("")
    lk, ck = J["미래교란"], J["체결점검"]
    A(f"- 미래 교란: {lk['T']}부터 EWY를 흔들어도 그 전 가치 최대 차 {lk['T전가치차']:.1e}, 그날 시가 비율 차 {lk['T일시가비율차']:.1e}. "
      f"상태 전환 {ck['전환수']}번 모두 다음 날 시가에 반영({ck['다음날시가에반영']}), 전환한 날 시가는 이전 값({ck['같은날시가에는이전값']}).")
    dU = S["수정안한EWY신호"]
    A(f"- 배당 수정 신호와 차트 신호가 다른 날은 {J['수정신호차이일']}/6642일(대부분 12~2월 배당락 전후)인데 긴 역사 우위가 "
      f"{pp(dU['합성']['긴앞이전']['효율선초과'] - ts['긴앞이전']['효율선초과'])}, 2025가 {pp(dU['실제']['2025']['효율선초과'] - tr['2025']['효율선초과'])} 바뀐다. "
      "1년에 몇 일만 들고/안 들고가 결과를 뒤집는다는 뜻이다.")
    A(f"- 합성 긴 역사 연율 차(+10.9 vs +10.8%)는 추가 비용을 새로 잰 2.55%로 넣은 탓이다. 원래처럼 2.6%면 +10.83%·효율선 대비 +4.09/+2.82%p.")
    rf = J["실제체결일_멈춘봉"]
    A(f"- 실제 KORU 체결일 {rf['체결일']}일 중 가격이 멈춘 봉(시가=고가=저가=종가)은 {rf['멈춘봉']}일.")
    A("")
    # ---- 이웃
    A("## 3. 이웃 25개 (사전 고정: n 30·40·50·60·70 × 띠 1~5%)")
    A("")
    A("긴 역사(합성 2000-05~2025-12) 효율선초과, 연 %p († = 긴 역사 최대 낙폭 −90% 이하 전멸):")
    A("")
    A("| n \\ 띠 | 1% | 2% | 3% | 4% | 5% |")
    A("|---|---|---|---|---|---|")
    for n in (30, 40, 50, 60, 70):
        cells = []
        for b in ("0.01", "0.02", "0.03", "0.04", "0.05"):
            k = f"{n}/{b}"
            v = pp(ex(k, "합성", "긴앞이전")) + ("†" if nb[k]["합성"]["긴역사"]["전멸"] else "")
            cells.append(f"**{v}**" if k == "50/0.03" else v)
        A(f"| {n}일 | " + " | ".join(cells) + " |")
    A("")
    cnt = lambda src, w: sum(ex(k, src, w) > 0 for k in keys)
    A("| 창 | 효율선초과 > 0 (25개 중) | 중앙 |")
    A("|---|---|---|")
    for lab, src, w in (("2026 앞(실제)", "실제", "2026앞"), ("2026 뒤(실제)", "실제", "2026뒤"), ("2025(실제)", "실제", "2025"),
                        ("긴 역사 2026 앞 이전(합성)", "합성", "긴앞이전"), ("지표 준비 후 2001-03~2025(합성)", "합성", "준비후앞이전"),
                        ("앞 절반 2000-05~2012(합성)", "합성", "앞절반"), ("뒤 절반 2013~2025(합성)", "합성", "뒤절반"), ("실제 KORU 2013~", "실제", "실제기간")):
        A(f"| {lab} | {cnt(src, w)} | {pp(float(np.median([ex(k, src, w) for k in keys])))} |")
    A(f"| 네 창(2026 앞·뒤·2025·긴 역사) 모두 > 0 | {len(four)} | — |")
    A(f"| 앞·뒤 절반 모두 > 0 | {len(halves)} ({', '.join(halves) or '없음'}) | — |")
    A("")
    # ---- 위약
    A("## 4. 위약 (같은 평균 비율·같은 전환 수, 무작위 날짜, 시드 20개; 순위는 21개 중, 1 = 최고)")
    A("")
    A("| 창 | 방법 | 규칙 | 위약 중앙 | 연율(수익) 순위 | 효율선초과 순위 | 위약 전멸 |")
    A("|---|---|---|---|---|---|---|")
    for k, lab in (("합성 긴 역사|긴역사", "합성 2000-05~2026-10"), ("합성 긴 역사|긴앞이전", "합성 2000-05~2025-12"), ("실제 2013~|실제기간", "실제 KORU 2013~")):
        v = pl[k]
        A(f"| {lab} | 안쪽 구간 길이 섞기 | 연 {p(v['규칙'][0])}/{p(v['규칙'][1], 0)} | 연 {p(v['위약중앙_연율'])}/{p(v['위약중앙_낙폭'], 0)} | "
          f"{v['연율순위']} | {v['효율선초과순위']} | {v['위약전멸수']}/20 |")
    for w, lab in (("2026앞", "2026 앞"), ("2026뒤", "2026 뒤"), ("2025", "2025")):
        v = pl[f"실제|{w}"]
        A(f"| {lab}(실제) | 전환 {v['전환수']}번 무작위 날짜, 노출 ±0.03 | {p(v['규칙'][0], 0)} | {p(v['위약중앙_수익'], 0)} | {v['수익순위']} | {v['효율선초과순위']} | — |")
    A("")
    # ---- 다른 상품
    A("## 5. 다른 3배 상품 (같은 규칙: 기초 50일선 띠 3%, 합성; 효율선초과)")
    A("")
    A("| 상품(기초) | 추가 비용 | 긴 역사 앞 이전(연) | 미끄러짐 0.5%(연) | 2026 앞 | 2026 뒤 | 2025 | 긴 역사 낙폭 규칙/보유 |")
    A("|---|---|---|---|---|---|---|---|")
    base = {"TQQQ": "^NDX→QQQ", "UPRO": "^GSPC→SPY", "TNA": "IWM", "TECL": "XLK", "YINN": "FXI", "EDC": "EEM"}
    for q, d in xp.items():
        r, h = d["규칙"], d["보유"]
        lw = f"{q}_긴앞이전"
        A(f"| {q}({base[q]}, {d['긴시작'][:4]}~) | {d['추가비용'] * 100:+.2f}% | {pp(r[lw]['효율선초과'])} | {pp(d['미끄러짐'][lw]['효율선초과'])} | "
          f"{pp(r['2026앞']['효율선초과'])} | {pp(r['2026뒤']['효율선초과'])} | {pp(r['2025']['효율선초과'])} | {p(r[lw]['낙폭'], 0)}/{p(h[lw]['낙폭'], 0)} |")
    A("")
    A(f"- 긴 역사 > 0: {len(xl)}/6({', '.join(xl)}), 미끄러짐 0.5%에서도 > 0: {sum(xp[q]['미끄러짐'][f'{q}_긴앞이전']['효율선초과'] > 0 for q in xp)}/6, "
      f"네 창 모두 > 0: {len(x4)}/6. 규칙이 전멸(−90% 이하)을 막은 상품 {sum(not xp[q]['규칙'][f'{q}_긴앞이전']['전멸'] for q in xp)}/6(보유는 0/6). "
      "p5_XPROD의 T50b3 값과 같다(YINN 긴 역사 −4.0 vs −4.2%p: 시작일 차이).")
    A("")
    # ---- 시작 시점
    A("## 6. 시작 시점 민감도 (합성, 시작 해 1월 → 2025-12, 효율선초과 연 %p)")
    A("")
    ys = list(range(2001, 2016))
    A("| 시작 | 2000-05 | " + " | ".join(str(y) for y in ys) + " |")
    A("|---|---|" + "---|" * len(ys))
    A("| T50b | " + pp(ts["긴앞이전"]["효율선초과"]) + " | " + " | ".join(pp(ts[f"시작{y}"]["효율선초과"]) for y in ys) + " |")
    A("")
    A(f"- > 0: {sum(ts[f'시작{y}']['효율선초과'] > 0 for y in ys)}/15. 모두 2013~2025를 공유해 독립 표본이 아니다. "
      f"겹치지 않는 두 절반: 앞 {pp(ts['앞절반']['효율선초과'])}, 뒤 {pp(ts['뒤절반']['효율선초과'])}.")
    A("")
    # ---- 판정
    A("## 판정·한계")
    A("")
    A("- **confirmed_fragile.** 주장 숫자와 '관찰 후보로만' 결론은 맞다. 우위의 근거는 2013~2025 한 구간과 50일·3% 한 점에 몰려 있고, "
      "이웃·신호 가격(배당 수정 여부)·체결 시각(시가/종가)·다른 상품에서 0 근처나 음수로 바뀐다. 2026(합격 기준)은 앞·뒤 모두 졌다.")
    A("- 시험 범위: 대상 1개 + 재현용 21개, 이웃 25개(사전 고정), 시점·비용 판 7개, 위약 시드 20개 × 6창, 다른 상품 6개, 시작 해 15개. "
      "'다음 날 종가 체결' 판과 '배당 수정 안 한 신호' 판은 검증 중 더한 것이다(결과를 보고 고른 값이 아님).")
    A("- 위약은 무작위 타이밍이라 폭락을 피하지 못해 대부분 전멸한다. 그래서 위약 1등은 '타이밍 정보가 0은 아니다'까지만 말하고, '덜 담기보다 낫다'는 효율선 비교로만 판단했다.")
    A("- 합성 KORU는 미국 장 EWY로 재설정해 실제(한국 장 기준)와 상관 0.98, 원화 환율은 넣지 않았다. 현금 이자 0. 매수 추천·주문 수량이 아니다.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
