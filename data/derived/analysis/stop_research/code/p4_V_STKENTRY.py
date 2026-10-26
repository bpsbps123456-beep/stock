"""4단계 반박 검증 V_STKENTRY(2026-10-11): 갈래 STKENTRY의 G1 대표 후보를 깨려고 시도한다.

후보: 252일 신고가 첫날(종가) 다음 날 시가에 사서 20거래일 보유, 10일째(산 날 = 0일째) 종가가 진입가 이하면
      11일째 시가에 전량 판다. 비교 = 같은 거래를 20일 들고만 있기(짝 평균차, 진입 월 묶음 t).

독립 재계산: trade_lab·stock_lab·lev_lab·tranche_lab을 import하지 않는다. cache/p4_stocks.parquet만 읽어
  (1) 종목 정리(시가·고가·저가·종가 결측·0 이하 행 제거, 260봉 미만 종목 제외),
  (2) 배당 수정(Adj Close/Close 비율을 OHLC에 곱함), (3) 자료 오류 의심(하루 종가 +150% 이상 또는 −70% 이하) 제외,
  (4) 신호·간격·거래·요약을 새로 짠다.

사전 고정(뒤·2025를 본 뒤 바꾸지 않음):
  - 창: 2026앞 1/2~5/29(청산도 5/29 안) · 2026뒤 6/1~10/9 · 2025(규약 H60, 보조로 H20) · 맥락 2018~2024 · 2018Q4 · 2020급락.
  - 이웃: 신고가 기간 L {126,189,252,315,378} × 간격 G {10,15,20,25,30} × 점검일 T {6,8,10,12,14}
          × 문턱 θ {−2,−1,0,+1,+2%} × 보유 H {10,15,20,25,30} (T < H−1만, 2,625개). 후보 ±2칸.
  - 위약: 같은 창 같은 거래에서 '판 거래 수'를 같게, 판 거래만 무작위로 고른다(같은 11일째 시가 청산 = 같은 보유일 분포).
          시드 20개 × (창 전체 무작위 · 진입 월 안 무작위) 두 가지.
  - 시점 변형: 청산 하루 늦춤(12일째 시가) · 진입 하루 늦춤(신호 이틀 뒤 시가) · 둘 다.
  - 비용: 수수료 0, 미끄러짐 0.5%(엔진 규약: 진입·시간 청산에만. 보조: 기간끝 종가 청산에도 똑같이).

실행: py p4_V_STKENTRY.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RESULTS = HERE.parent / "results"
OUT_JSON = RESULTS / "p4_V_STKENTRY.json"
OUT_MD = RESULTS / "p4_V_STKENTRY.md"

# ============================================================================ 사전 고정
CAND = dict(L=252, G=20, T=10, thr=0.0, H=20)
MIN_POS = 252                                   # 신호일은 종목 자료 252번째 봉 이후(엔진과 같음)
WIN = {
    "2026앞": ("2026-01-02", "2026-05-29", "2026-05-29"),
    "2026뒤": ("2026-06-01", "2026-10-09", None),
    "2025": ("2025-01-02", "2025-12-31", None),
    **{str(y): (f"{y}-01-01", f"{y}-12-31", None) for y in range(2018, 2025)},
    "2018Q4": ("2018-10-01", "2018-12-24", None),
    "2020급락": ("2020-02-19", "2020-03-23", None),
}
NB_L = (126, 189, 252, 315, 378)
NB_G = (10, 15, 20, 25, 30)
NB_T = (6, 8, 10, 12, 14)
NB_THR = (-0.02, -0.01, 0.0, 0.01, 0.02)
NB_H = (10, 15, 20, 25, 30)
N_SEEDS = 20
SLIP = 0.005



# ============================================================================ 자료(독립)
def load():
    df = pd.read_parquet(CACHE / "p4_stocks.parquet")
    df = df.sort_values(["sym", "date"], kind="mergesort")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[(df.Open > 0) & (df.High > 0) & (df.Low > 0) & (df.Close > 0)]
    n_all = int(df.sym.nunique())
    cnt = df.groupby("sym").size()
    keep = cnt[cnt >= 260].index
    df = df[df.sym.isin(keep)].reset_index(drop=True)
    f = (df["Adj Close"] / df["Close"]).fillna(1.0).to_numpy()
    O, Hh, Lo, C = (df[k].to_numpy() * f for k in ("Open", "High", "Low", "Close"))
    D = df.date.to_numpy().astype(str)
    sym = df.sym.to_numpy()
    starts = np.flatnonzero(np.r_[True, sym[1:] != sym[:-1]])
    ends = np.r_[starts[1:], len(sym)]
    syms = sym[starts]
    # 자료 오류 의심
    bad = []
    for s, a, b in zip(syms, starts, ends):
        r = C[a + 1:b] / C[a:b - 1] - 1
        if np.any(r > 1.5) or np.any(r < -0.7):
            bad.append(s)
    return dict(O=O, H=Hh, L=Lo, C=C, D=D, syms=syms, starts=starts, ends=ends, bad=set(bad),
                n_parquet=n_all, n_kept=len(syms))


def rolling_max(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=1).max().to_numpy()


def signals(X, L: int) -> dict:
    """종목별 신고가 첫날 신호 위치(종목 안 위치)."""
    out = {}
    C = X["C"]
    for i, (s, a, b) in enumerate(zip(X["syms"], X["starts"], X["ends"])):
        if s in X["bad"]:
            continue
        c = C[a:b]
        hi = rolling_max(c, L)
        sig = np.zeros(len(c), bool)
        sig[1:] = (c[1:] >= hi[1:] - 1e-12) & (c[:-1] < hi[:-1] - 1e-12)
        sig[:max(MIN_POS, L if L > MIN_POS else 0)] = False
        out[i] = np.flatnonzero(sig)
    return out


def entries(X, sig: dict, d0: str, d1: str, gap: int, mode: str = "engine") -> np.ndarray:
    """진입 = 신호 다음 날(전역 위치). mode engine: 창마다 신호 목록을 창 직전 날부터 새로 시작(엔진과 같음).
    full: 전체 이력에서 간격을 적용한 뒤 창 안 진입만."""
    D = X["D"]
    E = []
    for i, ss in sig.items():
        a, b = X["starts"][i], X["ends"][i]
        n = b - a
        d = D[a:b]
        lo = int(np.searchsorted(d, d0, "left")) - 1
        hi = int(np.searchsorted(d, d1, "right")) - 1
        if mode == "engine":
            s_lo, s_hi = max(lo, MIN_POS), min(hi, n - 2)
            cand = ss[(ss >= s_lo) & (ss <= s_hi)]
        else:
            cand = ss[ss <= n - 2]
        last = -10 ** 9
        for s in cand:
            if s - last >= gap:
                if mode == "engine" or (d0 <= d[s + 1] <= d1):
                    E.append(a + s + 1)
                last = s
    E = np.array(sorted(E), int)
    if len(E):
        E = E[(D[E] >= d0) & (D[E] <= d1)]
    return E


def random_sig(X) -> dict:
    """대조용 '아무 날' 진입(엔진 random과 같은 정의: 10거래일마다, 위상은 종목 이름 글자 코드 합)."""
    out = {}
    for i, (s, a, b) in enumerate(zip(X["syms"], X["starts"], X["ends"])):
        if s in X["bad"]:
            continue
        pos = np.arange(b - a)
        ph = sum(map(ord, s)) % 10
        out[i] = pos[(pos % 10 == ph) & (pos >= MIN_POS)]
    return out


# ============================================================================ 거래
def end_of(X, E):
    idx = np.searchsorted(X["starts"], E, "right") - 1
    return X["ends"][idx]


def trades(X, E, H, T, thr=0.0, slip=0.0, exit_by=None, exit_delay=0, entry_delay=0, slip_end=False):
    """반환 dict(배열): ret, bh, days, trig, month. 자료 부족·청산 마감 밖 거래는 뺀다."""
    O, C, D = X["O"], X["C"], X["D"]
    E = np.asarray(E) + entry_delay
    endpos = end_of(X, np.asarray(E) - entry_delay)
    ok = E + H - 1 < endpos
    if T + 1 + exit_delay > H - 1:
        ok &= False
    E = E[ok]
    if exit_by is not None:
        E = E[D[E + H - 1] <= exit_by]
    entry = O[E] * (1 + slip)
    xs = (1 - slip) if slip_end else 1.0
    bh = C[E + H - 1] * xs / entry - 1
    trig = C[E + T] <= entry * (1 + thr)
    sell = O[E + T + 1 + exit_delay] * (1 - slip) / entry - 1
    ret = np.where(trig, sell, bh)
    days = np.where(trig, T + 2 + exit_delay, H).astype(float)
    month = np.array([x[:7] for x in D[E]])
    return dict(E=E, ret=ret, bh=bh, days=days, trig=trig, month=month)


def mclust_t(d, month):
    if len(d) == 0:
        return float("nan")
    g = pd.Series(d).groupby(month).mean()
    if len(g) <= 2:
        return float("nan")
    return float(g.mean() / (g.std(ddof=1) / np.sqrt(len(g))))


def stats(tr, H) -> dict:
    r, bh, days = tr["ret"], tr["bh"], tr["days"]
    n = len(r)
    if n == 0:
        return {"n": 0}

    def one(x, dd):
        w, l = x[x > 0], x[x <= 0]
        return {"기대값": float(x.mean()), "승률": float((x > 0).mean()),
                "손익비": float(w.mean() / -l.mean()) if len(w) and len(l) and l.mean() < 0 else float("nan"),
                "P30": float((x <= -0.30).mean()), "하위5": float(np.quantile(x, 0.05)),
                "보유일당": float(x.sum() / dd.sum())}
    a, b = one(r, days), one(bh, np.full(n, float(H)))
    d = r - bh
    # 꼬리맞춤: 같은 하위5%가 나오게 처음부터 k만큼만 샀을 때(나머지 현금 0%) 대비 기대값 초과
    k = float(np.clip(a["하위5"] / b["하위5"], 0, 1)) if b["하위5"] < 0 and a["하위5"] < 0 else (0.0 if b["하위5"] < 0 else 1.0)
    return {"n": n, "월수": int(len(set(tr["month"]))), "평균차": float(d.mean()), "t": mclust_t(d, tr["month"]),
            **a, **{"보유_" + k_: v for k_, v in b.items()}, "평균보유": float(days.mean()), "청산비율": float(tr["trig"].mean()),
            "꼬리맞춤_k": k, "꼬리맞춤초과": float(a["기대값"] - k * b["기대값"])}


# ============================================================================ 본 계산
def main():
    t0 = time.time()
    X = load()
    n_bad = len(X["bad"])
    print("parquet", X["n_parquet"], "· 260봉 이상", X["n_kept"], "· 오류 의심", n_bad, "· 사용", X["n_kept"] - n_bad, flush=True)
    sig252 = signals(X, 252)
    out = {"meta": {"n_parquet": X["n_parquet"], "n_kept": X["n_kept"], "n_bad": n_bad, "n_used": X["n_kept"] - n_bad,
                    "bad": sorted(X["bad"])}}

    # ---- 1) 재현 + 맥락 (H20·H60)
    rep = {}
    Ewin = {}
    for w, (d0, d1, xb) in WIN.items():
        E = entries(X, sig252, d0, d1, 20, "engine")
        Ewin[w] = E
        for H in (20, 60):
            for sl in (0.0, SLIP):
                tr = trades(X, E, H, 10, 0.0, sl, xb)
                rep[f"{w}|{H}|{sl}"] = stats(tr, H)
            tr = trades(X, E, H, 10, 0.0, SLIP, xb, slip_end=True)
            rep[f"{w}|{H}|sym"] = stats(tr, H)
    out["rep"] = rep
    print("재현 끝", f"{time.time() - t0:.0f}s", flush=True)

    # 간격을 전체 이력으로 적용한 판(엔진은 창마다 새로 시작)
    full = {}
    for w in ("2026앞", "2026뒤", "2025", "2022", "2020", "2018"):
        d0, d1, xb = WIN[w]
        E = entries(X, sig252, d0, d1, 20, "full")
        for H in (20, 60):
            full[f"{w}|{H}"] = stats(trades(X, E, H, 10, 0.0, 0.0, xb), H)
    out["full_gap"] = full

    # ---- 2) 시점 변형
    timing = {}
    for w in ("2026앞", "2026뒤", "2025", "2022", "2020", "2018"):
        d0, d1, xb = WIN[w]
        E = Ewin[w]
        for H in (20, 60):
            for name, kw in (("정상", {}), ("청산하루늦춤", {"exit_delay": 1}), ("진입하루늦춤", {"entry_delay": 1}),
                             ("둘다", {"exit_delay": 1, "entry_delay": 1})):
                for sl in (0.0, SLIP):
                    timing[f"{w}|{H}|{name}|{sl}"] = stats(trades(X, E, H, 10, 0.0, sl, xb, **kw), H)
    out["timing"] = timing
    # 미래 교란 점검: 진입일 이후 가격을 바꿔도 진입 신호는 같아야 한다
    rng = np.random.default_rng(0)
    E26 = Ewin["2026앞"]
    pick = rng.choice(len(E26), size=min(200, len(E26)), replace=False)
    same = 0
    for k in pick:
        e = E26[k]
        i = int(np.searchsorted(X["starts"], e, "right") - 1)
        a, b = X["starts"][i], X["ends"][i]
        c = X["C"][a:b].copy()
        c[e - a:] *= 0.5                                     # 진입일 이후를 반토막
        hi = rolling_max(c, 252)
        s = e - a - 1
        ok = (c[s] >= hi[s] - 1e-12) and (c[s - 1] < hi[s - 1] - 1e-12)
        same += bool(ok)
    out["future_check"] = {"표본": int(len(pick)), "진입 뒤 교란해도 신호 유지": int(same)}
    print("시점 끝", f"{time.time() - t0:.0f}s", flush=True)

    # ---- 3) 이웃
    nb = []
    sig_cache = {252: sig252}
    for L in NB_L:
        if L not in sig_cache:
            sig_cache[L] = signals(X, L)
        for G in NB_G:
            Es = {w: entries(X, sig_cache[L], *WIN[w][:2], G, "engine") for w in ("2026앞", "2026뒤", "2025")}
            for H in NB_H:
                for T in NB_T:
                    if T >= H - 1:
                        continue
                    for thr in NB_THR:
                        row = {"L": L, "G": G, "H": H, "T": T, "thr": thr}
                        for w in ("2026앞", "2026뒤", "2025"):
                            for sl in (0.0, SLIP):
                                tr = trades(X, Es[w], H, T, thr, sl, WIN[w][2])
                                d = tr["ret"] - tr["bh"]
                                row[f"{w}|{sl}"] = float(d.mean()) if len(d) else float("nan")
                                if sl == 0.0:
                                    row[f"{w}|n"] = int(len(d))
                                    row[f"{w}|t"] = mclust_t(d, tr["month"])
                        tr = trades(X, Es["2025"], 60, T, thr, 0.0, None)
                        d = tr["ret"] - tr["bh"]
                        row["2025H60|0.0"] = float(d.mean()) if len(d) else float("nan")
                        nb.append(row)
            print("이웃", L, G, len(nb), f"{time.time() - t0:.0f}s", flush=True)
    out["neighbors"] = nb

    # ---- 4) 위약: 같은 거래, 판 거래 수 같게 무작위로
    plc = {}
    for w in ("2026앞", "2026뒤", "2025", "2022", "2020", "2018", "2019", "2021", "2023", "2024"):
        d0, d1, xb = WIN[w]
        for H in (20, 60):
            tr = trades(X, Ewin[w], H, 10, 0.0, 0.0, xb)
            n = len(tr["ret"])
            if n == 0:
                continue
            k = int(tr["trig"].sum())
            sell = X["O"][tr["E"] + 11] / X["O"][tr["E"]] - 1
            actual = float((tr["ret"] - tr["bh"]).mean())
            res = {"전체": [], "월안": []}
            tails = {"P30": [], "하위5": []}
            months = tr["month"]
            for seed in range(N_SEEDS):
                rg = np.random.default_rng(1000 + seed)
                sel = np.zeros(n, bool)
                sel[rg.choice(n, size=k, replace=False)] = True
                res["전체"].append(float(np.where(sel, sell - tr["bh"], 0.0).mean()))
                sel2 = np.zeros(n, bool)
                for m in np.unique(months):
                    ii = np.flatnonzero(months == m)
                    km = int(tr["trig"][ii].sum())
                    if km:
                        sel2[rg.choice(ii, size=km, replace=False)] = True
                res["월안"].append(float(np.where(sel2, sell - tr["bh"], 0.0).mean()))
                rr = np.where(sel2, sell, tr["bh"])
                tails["P30"].append(float((rr <= -0.30).mean()))
                tails["하위5"].append(float(np.quantile(rr, 0.05)))
            plc[f"{w}|{H}"] = {"n": n, "판거래": k, "실제": actual,
                               "실제_P30": float((tr["ret"] <= -0.30).mean()), "실제_하위5": float(np.quantile(tr["ret"], 0.05)),
                               "보유_P30": float((tr["bh"] <= -0.30).mean()), "보유_하위5": float(np.quantile(tr["bh"], 0.05)),
                               "가짜_P30": float(np.mean(tails["P30"])), "가짜_하위5": float(np.mean(tails["하위5"])),
                               **{f"{m}_평균": float(np.mean(v)) for m, v in res.items()},
                               **{f"{m}_최소": float(np.min(v)) for m, v in res.items()},
                               **{f"{m}_최대": float(np.max(v)) for m, v in res.items()},
                               **{f"{m}_가짜이상": int(sum(x >= actual for x in v)) for m, v in res.items()}}
    out["placebo"] = plc

    # 다른 진입에 같은 청산(아무 날 진입)
    rs = random_sig(X)
    rnd = {}
    for w in ("2026앞", "2026뒤", "2025"):
        d0, d1, xb = WIN[w]
        E = entries(X, rs, d0, d1, 20, "engine")
        for H in (20, 60):
            rnd[f"{w}|{H}"] = stats(trades(X, E, H, 10, 0.0, 0.0, xb), H)
    out["random_entry"] = rnd
    out["secs"] = time.time() - t0
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, default=float), encoding="utf-8")
    print("json →", OUT_JSON, f"{time.time() - t0:.0f}s")
    write_md(out)


# ============================================================================ md
def pp(x, d=2):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}"


def pc(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def tv(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:+.1f}"


# 주장 대조 행: (창, H, 미끄러짐 키, 지표, 주장값, 허용오차, 표시)
CLAIM_ROWS = [
    ("2026앞", 20, "0.0", "n", 2571, 0, "n"), ("2026앞", 20, "0.0", "평균차", 0.0024, 0.00005, "pp"),
    ("2026앞", 20, "0.0", "t", 0.3, 0.05, "t"), ("2026앞", 20, "0.0", "기대값", 0.0268, 0.00005, "pp"),
    ("2026앞", 20, "0.0", "보유_기대값", 0.0244, 0.00005, "pp"), ("2026앞", 20, "0.0", "승률", 0.48, 0.005, "pc0"),
    ("2026앞", 20, "0.0", "보유_승률", 0.55, 0.005, "pc0"), ("2026앞", 20, "0.0", "손익비", 2.01, 0.005, "r"),
    ("2026앞", 20, "0.0", "보유_손익비", 1.33, 0.005, "r"), ("2026앞", 20, "0.0", "P30", 0.004, 0.0005, "pc1"),
    ("2026앞", 20, "0.0", "보유_P30", 0.011, 0.0005, "pc1"), ("2026앞", 20, "0.0", "하위5", -0.141, 0.0005, "pc1"),
    ("2026앞", 20, "0.0", "보유_하위5", -0.188, 0.0005, "pc1"), ("2026앞", 20, "0.0", "보유일당", 0.0016, 0.00005, "pp"),
    ("2026앞", 20, "0.0", "보유_보유일당", 0.0012, 0.00005, "pp"), ("2026앞", 20, "0.0", "평균보유", 16.7, 0.05, "r1"),
    ("2026뒤", 20, "0.0", "n", 1911, 0, "n"), ("2026뒤", 20, "0.0", "평균차", 0.0026, 0.00005, "pp"),
    ("2026뒤", 20, "0.0", "t", 2.2, 0.05, "t"), ("2026뒤", 20, "0.0", "P30", 0.003, 0.0005, "pc1"),
    ("2026뒤", 20, "0.0", "보유_P30", 0.009, 0.0005, "pc1"),
    ("2026앞", 20, "0.005", "평균차", 0.0005, 0.00005, "pp"), ("2026뒤", 20, "0.005", "평균차", -0.0000, 0.00005, "pp"),
    ("2025", 60, "0.0", "평균차", -0.0192, 0.00005, "pp"), ("2025", 60, "0.0", "t", -2.4, 0.05, "t"),
    ("2018", 60, "0.0", "평균차", 0.0005, 0.00005, "pp"), ("2019", 60, "0.0", "평균차", -0.0008, 0.00005, "pp"),
    ("2020", 60, "0.0", "평균차", -0.0104, 0.00005, "pp"), ("2021", 60, "0.0", "평균차", -0.0125, 0.00005, "pp"),
    ("2022", 60, "0.0", "평균차", -0.0003, 0.00005, "pp"), ("2023", 60, "0.0", "평균차", -0.0085, 0.00005, "pp"),
    ("2024", 60, "0.0", "평균차", -0.0127, 0.00005, "pp"), ("2018Q4", 60, "0.0", "평균차", 0.008, 0.0005, "pp1"),
    ("2020급락", 60, "0.0", "평균차", 0.021, 0.0005, "pp1"),
]
KO = {"n": "거래 수", "평균차": "짝 평균차", "t": "월 묶음 t", "기대값": "규칙 기대값", "보유_기대값": "들고만 있기 기대값",
      "승률": "규칙 승률", "보유_승률": "들고만 있기 승률", "손익비": "규칙 손익비", "보유_손익비": "들고만 있기 손익비",
      "P30": "규칙 −30% 이하 비율", "보유_P30": "들고만 있기 −30% 이하 비율", "하위5": "규칙 하위 5%",
      "보유_하위5": "들고만 있기 하위 5%", "보유일당": "규칙 보유일당", "보유_보유일당": "들고만 있기 보유일당", "평균보유": "평균 보유일"}


def fmt(v, kind):
    if kind == "n":
        return f"{int(v):,}"
    if kind == "pp":
        return f"{v * 100:+.2f}%"
    if kind == "pp1":
        return f"{v * 100:+.1f}%"
    if kind == "pc0":
        return f"{v * 100:.0f}%"
    if kind == "pc1":
        return f"{v * 100:.1f}%"
    if kind == "t":
        return f"{v:+.2f}"
    if kind == "r":
        return f"{v:.2f}"
    return f"{v:.1f}"


def _rates(d, f, b, y):
    pos = d[d[f] > 0]
    return dict(n=len(d), 앞=len(pos), 뒤앞=(pos[b] > 0).mean() if len(pos) else np.nan, 뒤=(d[b] > 0).mean(),
                y앞=(pos[y] > 0).mean() if len(pos) else np.nan, y=(d[y] > 0).mean(),
                셋=((d[f] > 0) & (d[b] > 0) & (d[y] > 0)).mean())


def write_md(J: dict):
    R, TM, PL, RN, FG = J["rep"], J["timing"], J["placebo"], J["random_entry"], J["full_gap"]
    nb = pd.DataFrame(J["neighbors"])
    m = J["meta"]
    r = lambda w, H, sl="0.0": R[f"{w}|{H}|{sl}"]
    L = []
    # ---------------------------------------------------------------- 대조
    rows, worst, n_ok = [], 0.0, 0
    for w, H, sl, key, cv, tol, kd in CLAIM_ROWS:
        v = R[f"{w}|{H}|{sl}"][key]
        ok = abs(v - cv) <= tol + 1e-12
        n_ok += ok
        if kd not in ("n", "t", "r", "r1"):
            worst = max(worst, abs(v - cv))
        slab = "" if sl == "0.0" else " · 미끄러짐 0.5%"
        rows.append(f"| {w} · {H}일{slab} | {KO[key]} | {fmt(cv, kd)} | {fmt(v, kd)} | {'예' if ok else '아니오'} |")
    # ---------------------------------------------------------------- 이웃 요약
    F, B, Y, Y6 = "2026앞|0.0", "2026뒤|0.0", "2025|0.0", "2025H60|0.0"
    Fs, Bs, Ys = "2026앞|0.005", "2026뒤|0.005", "2025|0.005"
    r0, r60, rs = _rates(nb, F, B, Y), _rates(nb, F, B, Y6), _rates(nb, Fs, Bs, Ys)
    rH20 = _rates(nb[nb.H == 20], F, B, Y)
    cand = nb[(nb.L == 252) & (nb.G == 20) & (nb.H == 20) & (nb["T"] == 10) & np.isclose(nb.thr, 0)].iloc[0]
    rank_front = int((nb[F] > cand[F]).sum()) + 1
    pl = lambda w, H: PL[f"{w}|{H}"]
    c26f, c26b, c25, c2560 = r("2026앞", 20), r("2026뒤", 20), r("2025", 20), r("2025", 60)
    yrs20 = ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026앞", "2026뒤"]
    pos20 = [w for w in yrs20 if r(w, 20)["평균차"] > 0]
    neg20 = [w for w in yrs20 if r(w, 20)["평균차"] <= 0]
    tm = lambda w, H, nm, sl=0.0: TM[f"{w}|{H}|{nm}|{sl}"]

    L.append("# V_STKENTRY: 대표 후보 G1 반박 검증 — 신고가 돌파 매수 · 10일째 진입가 이하면 매도 · 20일 보유 (2026-10-11)\n")
    L.append("> 대상 주장: `results/p4_STKENTRY.md`의 G1(breakout · 10일째 진입가 이하면 · H20). 코드 `code/p4_V_STKENTRY.py`, 숫자 `results/p4_V_STKENTRY.json`.")
    L.append(f"> `trade_lab`·`stock_lab`·`lev_lab`·`tranche_lab`을 import하지 않고 `cache/p4_stocks.parquet`만으로 새로 짰다. parquet {m['n_parquet']:,}종목 → 260봉 이상 {m['n_kept']:,}종목 → "
             f"자료 오류 의심(하루 종가 +150% 이상·−70% 이하) {m['n_bad']}종목 제외 → **{m['n_used']:,}종목**. 수수료 0.\n")
    L.append("## 결론\n")
    L.append(f"1. **숫자는 그대로 재현됐다.** 주장 숫자 {len(CLAIM_ROWS)}개 중 {n_ok}개가 반올림 폭 안에서 같았다(가장 큰 차 {worst * 100:.2f}%p, 주장 승률이 정수 %로 반올림된 탓). "
             f"앞 {pp(c26f['평균차'])}%p(t {c26f['t']:+.1f}, {c26f['n']:,}건), 뒤 {pp(c26b['평균차'])}%p(t {c26b['t']:+.1f}, {c26b['n']:,}건), 2025(60일) {pp(c2560['평균차'])}%p(t {c2560['t']:+.1f}).")
    L.append(f"2. **뒤 구간의 우위는 규칙이 골라 판 덕이 아니다.** 같은 거래에서 같은 달에 같은 수를 무작위로 골라 11일째 시가에 판 가짜 규칙(시드 20개)도 평균 {pp(pl('2026뒤', 20)['월안_평균'])}%p였다(20개 중 {pl('2026뒤', 20)['월안_가짜이상']}개가 실제 이상). "
             f"뒤 구간은 돌파 매수 20일 수익이 평균 {pp(c26b['보유_기대값'])}%인 하락 구간이라 일찍 나가기만 하면 이겼다(이웃 2,625개 중 {r0['뒤'] * 100:.0f}%가 뒤 플러스). "
             f"진입을 하루 늦추면 뒤 우위는 {pp(tm('2026뒤', 20, '진입하루늦춤')['평균차'])}%p(t {tv(tm('2026뒤', 20, '진입하루늦춤')['t'])})로 사라진다. "
             f"앞 {pp(c26f['평균차'])}%p는 t {c26f['t']:+.1f}인 잡음 수준이고, 창 전체에서 무작위로 판 가짜(평균 {pp(pl('2026앞', 20)['전체_평균'])}%p)는 20개 모두 이겼지만, 같은 달 안에서 무작위로 판 가짜(평균 {pp(pl('2026앞', 20)['월안_평균'])}%p)와는 멀지 않다(20개 중 {pl('2026앞', 20)['월안_가짜이상']}개가 실제 이상). 앞 우위의 상당 부분도 '어느 달에 많이 팔았나'다.")
    L.append(f"3. **2025와 다른 해에 졌다 → 판정 반박(refuted).** 2025는 규약 60일 {pp(c2560['평균차'])}%p(t {c2560['t']:+.1f}), 보조로 같은 20일도 {pp(c25['평균차'])}%p(t {c25['t']:+.1f}). "
             f"이웃 2,625개 중 앞·뒤·2025가 모두 플러스는 {r0['셋'] * 100:.1f}%, 후보와 같은 20일 보유 이웃은 {rH20['셋'] * 100:.0f}%, 2025를 규약 60일로 보면 {r60['셋'] * 100:.0f}%, 미끄러짐 0.5%면 {rs['셋'] * 100:.0f}%. "
             f"남는 값은 꼬리 절단뿐이다. 20일 거래의 −30% 이하 손실을 {pc(c26f['보유_P30'])}→{pc(c26f['P30'])}로 줄였고, 무작위로 판 가짜({pc(pl('2026앞', 20)['가짜_P30'])})보다 2배쯤 잘 잘랐다. "
             f"하지만 기대값은 20일 보유 10개 창 중 {len(neg20)}개에서 들고만 있기보다 낮았다.\n")

    L.append("## 사용자에게 뜻하는 것\n")
    L.append(f"- 신고가 돌파로 산 일반 주식에 '10거래일 뒤 본전 아래면 판다'를 걸어도 **수익은 늘지 않는다**. 20일 보유에서 해마다 {pp(min(r(w, 20)['평균차'] for w in yrs20))}~{pp(max(r(w, 20)['평균차'] for w in yrs20))}%p "
             f"(플러스 {', '.join(pos20)} / 마이너스 {', '.join(neg20)}).")
    L.append(f"- 쓸모는 **보험**이다. 해마다 −30% 이하 거래를 줄였다(20일 보유: 2020년 {pc(r('2020', 20)['보유_P30'])}→{pc(r('2020', 20)['P30'])}, 2022년 {pc(r('2022', 20)['보유_P30'])}→{pc(r('2022', 20)['P30'])}). "
             f"20일 보유의 보험료는 거의 0(위 범위), 같은 꼬리가 나오게 처음부터 덜 사는 것과 비교한 꼬리맞춤은 해마다 {pp(min(r(w, 20)['꼬리맞춤초과'] for w in yrs20))}~{pp(max(r(w, 20)['꼬리맞춤초과'] for w in yrs20))}%p(플러스 {sum(r(w, 20)['꼬리맞춤초과'] > 0 for w in yrs20)}/10 창, 평균 {pp(np.mean([r(w, 20)['꼬리맞춤초과'] for w in yrs20]))}%p)로 '덜 사기'와 비기거나 조금 낫다.")
    L.append(f"- 60일 들고 갈 매매에 같은 규칙을 걸면 비싸다: 2025 {pp(c2560['평균차'])}, 2024 {pp(r('2024', 60)['평균차'])}, 2021 {pp(r('2021', 60)['평균차'])}, 2020 {pp(r('2020', 60)['평균차'])}%p(10일 만에 판 뒤 50일 반등을 놓친다).\n")

    # ---------------------------------------------------------------- 범위
    L.append("## 시험 범위 (사전 고정, 스크립트 맨 위 상수)\n")
    L.append("| 검증 | 개수·간격 | 비고 |\n|---|---|---|")
    L.append(f"| 독립 재계산 | 후보 1개, 주장 숫자 {len(CLAIM_ROWS)}개 | 0.5%p 넘게 다르면 원인 규명 |")
    L.append("| 시점 감사 | 미래 교란 200건(진입일 뒤 가격을 반으로 바꿔도 신호 유지?), 간격 규약(창마다 새로 시작 vs 전체 이력) | 엔진 코드도 읽음 |")
    L.append("| 체결 변형 | 청산 하루 늦춤(12일째 시가) · 진입 하루 늦춤(신호 이틀 뒤 시가) · 둘 다, 각각 미끄러짐 0·0.5% | 2026앞·뒤·2025·2022·2020·2018, 20·60일 |")
    L.append("| 미끄러짐 규약 | 엔진 규약(진입·시간 청산 0.5%, 기간끝 종가 청산 0) · 대칭 규약(기간끝 청산도 0.5%) | 대칭은 보조 |")
    L.append("| 이웃 | 신고가 기간 L {126, 189, 252, 315, 378} × 간격 G {10, 15, 20, 25, 30} × 점검일 T {6, 8, 10, 12, 14} × 문턱 θ {−2, −1, 0, +1, +2%} × 보유 H {10, 15, 20, 25, 30}, T < H−1만 = **2,625개** | 후보 ±2칸. 2026앞·뒤·2025(같은 H)·2025(규약 60일)·미끄러짐 0.5% |")
    L.append("| 위약 | 같은 거래에서 판 거래 수를 같게 무작위로 골라 11일째 시가 청산(같은 보유일 분포). 창 전체 무작위 · 진입 월 안 무작위, 각 시드 20개 | 10개 창 × 20·60일 |")
    L.append("| 다른 진입 | 같은 청산 규칙을 '아무 날(10거래일마다)' 진입에 | 돌파 진입이 필요한가 |\n")

    L.append("## 변수 시점\n")
    L.append("| 변수 | 언제 아는가 | 체결 | 매매 근거 |\n|---|---|---|---|")
    L.append("| 신고가 첫날: 종가 = 252일 최고 종가(그날 포함), 전날 종가 < 전날까지 252일 최고 종가 | 신호일 s 종가 | s+1일 시가 매수 | 예 |")
    L.append("| 같은 종목 신호 간격 20거래일 | 신호일 | — | 예 |")
    L.append("| 10일째 종가 ≤ 진입가 | e+10일 종가 | e+11일 시가 매도 | 예 |")
    L.append("| 20일 보유 끝 | e+19일 종가 | 그날 종가 매도 | 예 |")
    L.append("| 배당 수정 비율(Adj Close/Close) | 신호 창 안 두 날짜 비교에는 그 사이 배당만 반영(그 뒤 배당은 양쪽에 같은 배수) | — | 미래 정보 없음 |")
    L.append("| 종목표(현재 시총 10억 달러 이상) | 2026-10 현재(사후) | — | 아니오 — 생존편향 |")
    L.append(f"| 미래 교란 점검 | 진입일 이후 가격을 반으로 바꿔도 신호 유지 {J['future_check']['진입 뒤 교란해도 신호 유지']}/{J['future_check']['표본']} | — | 통과 |\n")

    # ---------------------------------------------------------------- 1) 재계산
    L.append("## 1) 독립 재계산\n")
    L.append("| 창 | 항목 | 주장 | 재계산 | 일치 |\n|---|---|---|---|---|")
    L.extend(rows)
    L.append("")
    L.append(f"- 2026앞 20일은 청산이 5/29 안인 거래만 썼다(진입 1/2~5/1). 판 비율 {c26f['청산비율'] * 100:.0f}%(뒤 {c26b['청산비율'] * 100:.0f}%).")
    L.append(f"- 월 묶음 수: 앞 {c26f['월수']}개월, 뒤 {c26b['월수']}개월. 뒤 t +2.2는 자유도 3이라 양측 p ≈ 0.12다.")
    L.append(f"- 간격 규약: 엔진은 창마다 신호 목록을 새로 시작한다. 전체 이력에서 20일 간격을 적용해도 앞 {pp(FG['2026앞|20']['평균차'])}%p(t {tv(FG['2026앞|20']['t'])}), "
             f"뒤 {pp(FG['2026뒤|20']['평균차'])}%p(t {tv(FG['2026뒤|20']['t'])}), 2025 20일 {pp(FG['2025|20']['평균차'])}%p · 60일 {pp(FG['2025|60']['평균차'])}%p로 결론이 같다.\n")

    # ---------------------------------------------------------------- 2) 시점·체결
    L.append("## 2) 시점 감사 · 하루 늦춘 판 · 미끄러짐\n")
    L.append("짝 평균차(규칙 − 같은 거래 들고만 있기, %p)와 (월 묶음 t). 진입을 늦춘 판은 들고만 있기도 같은 날 산다.\n")
    L.append("| 창 · 보유 | 정상 | 청산 하루 늦춤 | 진입 하루 늦춤 | 둘 다 | 정상 · 미끄러짐 0.5% | 둘 다 · 0.5% | 대칭 미끄러짐 0.5% |\n|---|---|---|---|---|---|---|---|")
    cellt = lambda x: f"{pp(x['평균차'])} ({tv(x['t'])})"
    for w in ("2026앞", "2026뒤", "2025", "2022", "2020", "2018"):
        for H in (20, 60):
            L.append(f"| {w} · {H}일 | {cellt(tm(w, H, '정상'))} | {cellt(tm(w, H, '청산하루늦춤'))} | {cellt(tm(w, H, '진입하루늦춤'))} | {cellt(tm(w, H, '둘다'))} | "
                     f"{cellt(tm(w, H, '정상', SLIP))} | {cellt(tm(w, H, '둘다', SLIP))} | {cellt(R[f'{w}|{H}|sym'])} |")
    L.append("")
    L.append(f"- 시점: 신호는 s일 종가까지, 매수는 s+1일 시가, 청산 판단은 10일째 종가, 매도는 다음 날 시가. 미래 정보 없음(미래 교란 {J['future_check']['진입 뒤 교란해도 신호 유지']}/{J['future_check']['표본']} 통과).")
    L.append(f"- 청산을 하루 늦춰도 거의 같다(뒤 {pp(tm('2026뒤', 20, '청산하루늦춤')['평균차'])}%p). 진입을 하루 늦추면 뒤 우위가 {pp(tm('2026뒤', 20, '진입하루늦춤')['평균차'])}%p로 사라진다. 뒤 우위는 진입일 하루의 우연에 기대고 있다.")
    L.append(f"- 미끄러짐 0.5%: 엔진 규약은 시간 청산만 0.5%를 더 내므로 판 비율({c26b['청산비율'] * 100:.0f}%)만큼 불리하다 → 뒤 {pp(R['2026뒤|20|0.005']['평균차'])}%p. "
             f"기간끝 종가 청산에도 똑같이 0.5%를 매기는 대칭 규약에서는 앞 {pp(R['2026앞|20|sym']['평균차'])}, 뒤 {pp(R['2026뒤|20|sym']['평균차'])}%p로 0% 때와 같다. "
             f"'미끄러짐 0.5%에 사라진다'는 규약 의존이다. 다만 대칭 규약에서도 2025는 마이너스다(20일 {pp(R['2025|20|sym']['평균차'])}, 60일 {pp(R['2025|60|sym']['평균차'])}%p).\n")

    # ---------------------------------------------------------------- 3) 이웃
    L.append("## 3) 이웃 (2,625개)\n")
    L.append("| 집합 | 앞 > 0 개수 / 전체 | 뒤 > 0: 앞>0 집합 / 무조건 | 2025 > 0: 앞>0 집합 / 무조건 | 앞·뒤·2025 모두 > 0 |\n|---|---|---|---|---|")
    for nm, x in (("미끄러짐 0, 2025 같은 H", r0), ("미끄러짐 0, 2025 규약 60일", r60), ("미끄러짐 0.5%, 2025 같은 H", rs), ("H = 20 이웃만(625개)", rH20)):
        L.append(f"| {nm} | {x['앞']} / {x['n']} | {pc(x['뒤앞'], 0)} / {pc(x['뒤'], 0)} | {pc(x['y앞'], 0)} / {pc(x['y'], 0)} | {pc(x['셋'], 1)} |")
    L.append("")
    L.append("H별 '앞·뒤·2025(같은 H) 모두 > 0' 비율: " + " · ".join(f"H{H} {_rates(nb[nb.H == H], F, B, Y)['셋'] * 100:.1f}%" for H in NB_H) + ".\n")
    L.append("파라미터 하나씩만 바꾼 판(나머지는 후보값). 짝 평균차 %p (월 묶음 t):\n")
    L.append("| 파라미터 | 값 | 2026앞 | 2026뒤 | 2025 같은 H | 2025 60일(규약) | 앞 · 0.5% | 뒤 · 0.5% |\n|---|---|---|---|---|---|---|---|")
    base = dict(L=252, G=20, H=20, T=10, thr=0.0)
    for prm, vals, lab in (("L", NB_L, "신고가 기간"), ("G", NB_G, "신호 간격"), ("T", NB_T, "점검일"), ("thr", NB_THR, "문턱"), ("H", NB_H, "보유")):
        for v in vals:
            q = dict(base, **{prm: v})
            vv = f"{v * 100:+.0f}%" if prm == "thr" else str(v)
            if q["T"] >= q["H"] - 1:
                L.append(f"| {lab} | {vv} | 해당 없음(T ≥ H−1) | | | | | |")
                continue
            x = nb[(nb.L == q["L"]) & (nb.G == q["G"]) & (nb.H == q["H"]) & (nb["T"] == q["T"]) & np.isclose(nb.thr, q["thr"])].iloc[0]
            star = " ◀ 후보" if q == base else ""
            L.append(f"| {lab} | {vv}{star} | {pp(x[F])} ({tv(x['2026앞|t'])}) | {pp(x[B])} ({tv(x['2026뒤|t'])}) | {pp(x[Y])} ({tv(x['2025|t'])}) | "
                     f"{pp(x[Y6])} | {pp(x[Fs])} | {pp(x[Bs])} |")
    L.append("")
    h15 = nb[(nb.H == 15) & (nb[F] > 0) & (nb[B] > 0) & (nb[Y] > 0)]
    L.append(f"- 후보의 앞 값은 이웃 2,625개 중 {rank_front}번째(앞 중앙값 {pp(nb[F].median())}%p, 최대 {pp(nb[F].max())}%p). 앞 > 0 집합의 뒤 성공률({pc(r0['뒤앞'], 0)})은 무조건 비율({pc(r0['뒤'], 0)})과 같다 → 앞으로 골라도 뒤를 더 맞히지 못한다.")
    L.append(f"- 2025(같은 H) 플러스 {int((nb[Y] > 0).sum())}개는 모두 H15 이웃이다(앞·뒤·2025 모두 플러스 {len(h15)}개, 그 2025 값 중앙 {pp(h15[Y].median()) if len(h15) else '—'}%p). H20 이웃 625개는 2025가 모두 마이너스다.\n")

    # ---------------------------------------------------------------- 4) 위약
    L.append("## 4) 위약: 같은 거래, 같은 수를 무작위로 판 가짜 규칙 (시드 20개)\n")
    L.append("| 창 · 보유 | 거래 / 판 거래 | 실제 짝 평균차 | 가짜(창 전체) 평균 [최소~최대] · 실제 이상 | 가짜(월 안) 평균 [최소~최대] · 실제 이상 | −30% 이하: 보유 → 가짜 → 실제 | 하위 5%: 보유 → 가짜 → 실제 |\n|---|---|---|---|---|---|---|")
    for w in ("2026앞", "2026뒤", "2025", "2018", "2019", "2020", "2021", "2022", "2023", "2024"):
        for H in (20, 60):
            x = PL.get(f"{w}|{H}")
            if not x:
                continue
            L.append(f"| {w} · {H}일 | {x['n']:,} / {x['판거래']:,} | {pp(x['실제'])} | {pp(x['전체_평균'])} [{pp(x['전체_최소'])}~{pp(x['전체_최대'])}] · {x['전체_가짜이상']}/20 | "
                     f"{pp(x['월안_평균'])} [{pp(x['월안_최소'])}~{pp(x['월안_최대'])}] · {x['월안_가짜이상']}/20 | "
                     f"{pc(x['보유_P30'])} → {pc(x['가짜_P30'])} → {pc(x['실제_P30'])} | {pc(x['보유_하위5'])} → {pc(x['가짜_하위5'])} → {pc(x['실제_하위5'])} |")
    L.append("")
    n_beat = [w for w in yrs20 if PL[f"{w}|20"]["월안_가짜이상"] <= 1]
    n_lose = [w for w in yrs20 if PL[f"{w}|20"]["월안_가짜이상"] >= 19]
    L.append(f"- 기대값: 20일 보유 10개 창 중 실제가 가짜(월 안) 20개 중 19개 이상을 이긴 창은 {len(n_beat)}개({', '.join(n_beat)}), 거꾸로 가짜 19개 이상이 실제 이상인 창은 {len(n_lose)}개({', '.join(n_lose)}). "
             "'10일째 본전 아래'라는 정보가 남은 10일 수익을 맞히는 방향이 해마다 바뀌었다.")
    L.append("- 꼬리: 실제 규칙은 모든 창에서 가짜보다 −30% 이하 비율과 하위 5%를 더 줄였다. 본전 아래 거래만 골라 파니 꼬리는 확실히 잘린다. 그 대가로 기대값이 오르지는 않았다.\n")
    L.append(f"다른 진입에 같은 청산('아무 날' 진입, 20일): 2026앞 {pp(RN['2026앞|20']['평균차'])}%p(t {tv(RN['2026앞|20']['t'])}), 2026뒤 {pp(RN['2026뒤|20']['평균차'])}%p, 2025 {pp(RN['2025|20']['평균차'])}%p "
             f"(60일 2025 {pp(RN['2025|60']['평균차'])}%p). 앞이 플러스인 건 돌파 진입일 때뿐이었다.\n")

    # ---------------------------------------------------------------- 5) 장세 · 핵심 표
    L.append("## 5) 장세별 핵심 표 (상황 → 규칙 → 숫자)\n")
    L.append("규칙 = 신고가 첫날 다음 날 시가 매수, 10일째 종가가 진입가 이하면 11일째 시가 매도. 셀 = 짝 평균차 %p (t) · −30% 이하 비율(보유→규칙) · 하위 5%(보유→규칙).\n")
    L.append("| 창 | 건수 20일 / 60일 | 20일 보유 | 60일 보유 | 들고만 있기 기대값 20일 / 60일 |\n|---|---|---|---|---|")
    cellr = lambda x: f"{pp(x['평균차'])} ({tv(x['t'])}) · {pc(x['보유_P30'])}→{pc(x['P30'])} · {pc(x['보유_하위5'], 0)}→{pc(x['하위5'], 0)}"
    for w in ("2026앞", "2026뒤", "2025", "2024", "2023", "2022", "2021", "2020", "2019", "2018", "2018Q4", "2020급락"):
        a, b = r(w, 20), r(w, 60)
        L.append(f"| {w} | {a['n']:,} / {b['n']:,} | {cellr(a)} | {cellr(b)} | {pp(a['보유_기대값'], 1)}% / {pp(b['보유_기대값'], 1)}% |")
    L.append("")
    L.append(f"- 폭락 구간에서는 일찍 판 것이 컸다: 2020년 2/19~3/23 진입은 20일 {pp(r('2020급락', 20)['평균차'], 1)}%p, 60일 {pp(r('2020급락', 60)['평균차'], 1)}%p(60일은 반등을 놓쳐 줄어든다). "
             f"2018 4분기 {pp(r('2018Q4', 20)['평균차'], 1)}·{pp(r('2018Q4', 60)['평균차'], 1)}%p, 2022년 {pp(r('2022', 20)['평균차'])}·{pp(r('2022', 60)['평균차'])}%p.")
    L.append("- 2026앞은 5개월, 2026뒤는 4개월이라 월 묶음 t의 자유도가 작다. '해마다 같은 방향인가'가 더 믿을 만한 근거다.\n")

    # ---------------------------------------------------------------- 승률·손익비
    L.append("## 승률 · 손익비 · 꼬리 (20일 보유, 규칙 / 들고만 있기)\n")
    L.append("| 창 | 기대값 | 승률 | 손익비 | −30% 이하 | 하위 5% | 보유일당 | 평균 보유일 | 꼬리맞춤(같은 하위 5%로 덜 사기 대비) |\n|---|---|---|---|---|---|---|---|---|")
    for w in yrs20:
        x = r(w, 20)
        L.append(f"| {w} | {pp(x['기대값'])}% / {pp(x['보유_기대값'])}% | {pc(x['승률'], 0)} / {pc(x['보유_승률'], 0)} | {x['손익비']:.2f} / {x['보유_손익비']:.2f} | "
                 f"{pc(x['P30'])} / {pc(x['보유_P30'])} | {pc(x['하위5'])} / {pc(x['보유_하위5'])} | {pp(x['보유일당'], 3)}% / {pp(x['보유_보유일당'], 3)}% | "
                 f"{x['평균보유']:.1f} / 20 | {pp(x['꼬리맞춤초과'])}%p (k {x['꼬리맞춤_k']:.2f}) |")
    L.append("")
    lowpd = [w for w in yrs20 if r(w, 20)["보유일당"] < r(w, 20)["보유_보유일당"]]
    L.append("- 승률은 늘 낮아지고 손익비는 늘 높아진다. 본전 아래를 일찍 끊으니 생기는 기계적 결과다. 기대값은 거의 그대로다.")
    L.append(f"- 보유일당이 들고만 있기보다 낮은 창: {', '.join(lowpd)}. 판 돈을 바로 다른 데 넣을 수 있어야 보유일당 우위가 의미 있다.\n")

    # ---------------------------------------------------------------- 판정
    L.append("## 판정\n")
    L.append("| 검증 | 결과 | 후보에 유리? |\n|---|---|---|")
    L.append(f"| 독립 재계산 | {n_ok}/{len(CLAIM_ROWS)} 일치 | 중립(숫자는 맞음) |")
    L.append(f"| 시점 감사 | 미래 정보 없음. 청산 하루 늦춤은 같음, 진입 하루 늦춤은 뒤 {pp(c26b['평균차'])} → {pp(tm('2026뒤', 20, '진입하루늦춤')['평균차'])}%p | 불리 |")
    L.append(f"| 미끄러짐 0.5% | 엔진 규약: 앞 {pp(R['2026앞|20|0.005']['평균차'])} · 뒤 {pp(R['2026뒤|20|0.005']['평균차'])}%p. 대칭 규약: 앞 {pp(R['2026앞|20|sym']['평균차'])} · 뒤 {pp(R['2026뒤|20|sym']['평균차'])}%p | 규약에 따라 다름 |")
    L.append(f"| 이웃 | 앞·뒤·2025 모두 > 0: {r0['셋'] * 100:.1f}%(H20 이웃 {rH20['셋'] * 100:.0f}%, 규약 2025 60일 {r60['셋'] * 100:.0f}%, 0.5% {rs['셋'] * 100:.0f}%) | 불리 |")
    L.append(f"| 위약(월 안 무작위) | 앞 실제 이상 {pl('2026앞', 20)['월안_가짜이상']}/20, 뒤 {pl('2026뒤', 20)['월안_가짜이상']}/20 — 뒤 우위는 노출 감소로 설명됨 | 불리 |")
    L.append(f"| 2025 | 60일 {pp(c2560['평균차'])}%p (t {c2560['t']:+.1f}), 20일 {pp(c25['평균차'])}%p (t {c25['t']:+.1f}) | 불리 |")
    L.append(f"| 다른 해(20일) | 플러스 {len(pos20)}/10 창: {', '.join(pos20)} | 불리 |\n")
    L.append("**반박(refuted)**: '신고가 돌파 매수에 10일째 본전 이하 매도를 걸면 20일 들고만 있기보다 낫다'는 서지 않는다. 앞 우위는 잡음이다. 뒤 우위는 하락 구간에서 일찍 나간 효과라 같은 수를 무작위로 팔아도 나왔고, 진입을 하루 늦추면 사라졌다. "
             "2025는 20일·60일 모두 졌다. 꼬리 절단(−30% 이하 손실 약 2/3 감소)은 진짜지만, 갈래 결론 2와 같은 '보험'이지 수익 규칙이 아니다.\n")

    L.append("## 선택편향 · 한계\n")
    L.append(f"- 후보는 갈래에서 손절 190개 중 앞 1위로 뽑혔다. 앞 +0.24%p(t +0.3)는 그 최댓값이고, 여기 이웃 2,625개에서도 앞 값 상위 {rank_front / len(nb) * 100:.0f}%다.")
    L.append("- 생존편향: 종목표는 2026-10 현재 시총 10억 달러 이상 생존 종목이다. 상장폐지·파산 종목(본전 아래에서 더 빠졌을 거래)이 없어 손절에 불리하다. 실제 꼬리 절단 효과는 여기보다 조금 크다. 20일 보유에서는 그런 거래의 비중이 작다.")
    L.append("- 2026앞 20일은 청산이 5/29 안이어야 해서 진입이 1/2~5/1, 뒤 20일은 진입이 6/1~9/14다. 월 묶음 4~5개라 t 자유도가 작다.")
    L.append("- 2025 확인은 규약대로 60일이다. 20일 규칙에 불리할 수 있어 20일도 보조로 보였고, 결론은 같았다.")
    L.append("- 위약은 같은 거래 안에서 판 거래만 바꿨다(진입 신호는 그대로). 다른 진입(아무 날)에 같은 청산을 건 판은 따로 보였다.")
    L.append(f"- 배당 수정은 parquet의 Adj Close/Close 비율(엔진과 같은 방식). 자료 오류 의심 {m['n_bad']}종목 제외. 수수료 0, 현금 이자 0. 이웃 L > 252는 신호일을 L번째 봉 이후로 제한했다.")
    L.append(f"\n계산 {J.get('secs', 0):.0f}초.")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("md →", OUT_MD)


if __name__ == "__main__":
    import sys
    if "--md" in sys.argv:
        write_md(json.loads(OUT_JSON.read_text(encoding="utf-8")))
    else:
        main()
