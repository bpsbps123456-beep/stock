"""4단계 반박 검증 V_STKTAIL(2026-10-11): 갈래 STKTAIL의 G1 대표 후보를 깨려고 시도한다.

후보(G1): 신호일 60일 실현변동성이 그날 단면 아래 1/3인 종목을 무작위 날짜(종목마다 10거래일에 한 번)에 다음 날 시가로 사서
          60거래일 보유, 산 뒤 t−1일까지 최고 종가(첫날은 진입가) × 0.75 아래로 장중에 내려가면 전량 판다(시가가 선 아래면 시가).
          비교 = 같은 거래를 60일 들고만 있기(짝 평균차, 진입 월 묶음 t).

독립 재계산: trade_lab·stock_lab·lev_lab·tranche_lab을 import하지 않는다. cache/p4_stocks.parquet(가격)와
  cache/p4_STKTAIL_splits.json(분할 이력, 100칸 고르기 재현의 복권형 명목 주가에만)만 읽어 새로 짠다.
  (1) 종목 정리: 시가·고가·저가·종가 결측·0 이하 행 제거, 260봉 미만 종목 제외.
  (2) 배당 수정: Adj Close/Close 비율을 시가·고가·저가·종가에 곱한다.
  (3) 자료 오류 의심: 하루 종가 변화 +150% 이상 또는 −70% 이하가 한 번이라도 있으면 뺀다.
  (4) 진입·묶음·거래·요약·고르기를 새로 짠다.

사전 고정(뒤·2025를 본 뒤 바꾸지 않음):
  - 창: 2026앞(진입 1/2~, 60일째가 5/29 안) · 2026뒤(진입 6/1~, 10/9까지 60일을 채운 거래) · 2025(보조) ·
        맥락 2015~2024 · 폭락 시기 진입 2018 4분기·2020 1분기·2022 상반기(사후 정의).
  - 이웃(후보 ±2칸): 추적 폭 p {15,20,25,30,35%} × 변동성 기간 lb {20,40,60,90,120일} × 묶음 문턱 q {1/5,1/4,1/3,2/5,1/2}
        × 보유 H {40,50,60,70,80일} = 625개. H 60 고정 3차원 125개도 따로.
  - 위약 2가지(시드 20개): A = 같은 묶음 거래에서 '나간 거래 수'를 같게 무작위 거래를 고르고, 실제 손절 청산일을 섞어 붙여
        그날 시가에 판다(같은 노출 시간·같은 나가기 횟수). B = 실제로 손절된 같은 거래들에 청산일만 서로 섞는다(같은 보유일 분포).
  - 시점 변형: 손절 체결 하루 늦춤(선에 닿은 다음 날 시가) · 종가 기준 손절(종가가 선 아래면 다음 날 시가) ·
        묶음 정보 하루 늦춤(s−1일 백분위) · 진입 하루 늦춤(신호 이틀 뒤 시가) · 모두.
  - 비용: 수수료 0, 미끄러짐 0.5%(엔진 규약: 진입·손절 체결에만. 보조: 기간끝 종가 청산에도 똑같이).

실행: PYTHONIOENCODING=utf-8 py p4_V_STKTAIL.py
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
RESULTS = HERE.parent / "results"
OUT_JSON = RESULTS / "p4_V_STKTAIL.json"
OUT_MD = RESULTS / "p4_V_STKTAIL.md"

# ============================================================================ 사전 고정
CAND = dict(p=0.25, lb=60, q=1 / 3, H=60)
EVERY = 10
MIN_POS = 252
NB_P = (0.15, 0.20, 0.25, 0.30, 0.35)
NB_LB = (20, 40, 60, 90, 120)
NB_Q = (0.20, 0.25, 1 / 3, 0.40, 0.50)
NB_H = (40, 50, 60, 70, 80)
N_SEEDS = 20
SLIP = 0.005
CUT_FRONT = "2026-05-29"
MIN_N = 30
WIN = {"2026앞": ("2026-01-02", "2026-05-29"), "2026뒤": ("2026-06-01", "2026-10-09"),
       "2025": ("2025-01-02", "2025-12-31")}
for _y in range(2015, 2025):
    WIN[str(_y)] = (f"{_y}-01-01", f"{_y}-12-31")
MAIN_W = ["2026앞", "2026뒤", "2025"]
CRASH = {"2018 4분기 진입": ("2018", "2018-10-01", "2018-12-31"), "2020 1분기 진입": ("2020", "2020-01-02", "2020-03-31"),
         "2022 상반기 진입": ("2022", "2022-01-03", "2022-06-30")}
H_SENS = {"2026앞": 20, "2026뒤": 20, "2025": 120, "2022": 120}

# 주장 숫자(갈래 STKTAIL md·json, 저변동 × 추적-25%)
CLAIM = {
    "2026앞": dict(n=3353, 짝차=-0.0026, 월t=-3.2, 발동률=0.082, 보유_P30=0.0048, 규칙_P30=0.0, 보유_하위5=-0.194,
                  규칙_하위5=-0.222, 미끄러짐_짝차=-0.0030),
    "2026뒤": dict(n=2581, 짝차=0.0002, 발동률=0.035),
    "2025": dict(n=19332, 짝차=-0.0032, 월t=-1.3, 발동률=0.068, 보유_P30=0.0049, 규칙_P30=0.0007),
    "2022": dict(n=18214, 짝차=-0.0019),
    "2025_H120": dict(n=19332, 짝차=-0.0097),
    "2022_H120": dict(n=18214, 짝차=-0.0069),
}
G1_CLAIM_CELL = ("저변동", "추적-25%")
STKTAIL_RULES = ["추적-20%", "추적-25%", "SMA50 이탈", "고정-15%", "추적-20% 절반"]


# ============================================================================ 자료
def load():
    raw = pd.read_parquet(CACHE / "p4_stocks.parquet")
    raw = raw.dropna(subset=["Open", "High", "Low", "Close"])
    raw = raw[(raw.Open > 0) & (raw.High > 0) & (raw.Low > 0) & (raw.Close > 0)]
    raw = raw.sort_values(["sym", "date"], kind="mergesort")
    assert not raw.duplicated(["sym", "date"]).any(), "같은 날 중복 행"
    out = {}
    for sym, g in raw.groupby("sym", sort=True):
        if len(g) < 260:
            continue
        f = (g["Adj Close"] / g["Close"]).fillna(1.0).to_numpy()
        out[sym] = {"days": g["date"].to_numpy().astype(str), "o": g.Open.to_numpy() * f, "h": g.High.to_numpy() * f,
                    "l": g.Low.to_numpy() * f, "c": g.Close.to_numpy() * f, "rc": g.Close.to_numpy(),
                    "v": g.Volume.to_numpy(float)}
    return out


def is_bad(c):
    r = c[1:] / c[:-1] - 1
    return bool((r > 1.5).any() or (r < -0.7).any())


def features(D, syms, splits):
    """종목별 지표(t일 종가까지) → 그날 단면 백분위. 반환 {키: DataFrame(날짜 × 종목)}"""
    cols = {f"rv{lb}": {} for lb in NB_LB}
    cols.update({k: {} for k in ("dv60", "mom120", "max20", "dist", "px")})
    for s in syms:
        d = D[s]
        c = d["c"]
        idx = pd.Index(d["days"])
        r = pd.Series(np.concatenate([[np.nan], c[1:] / c[:-1] - 1]))
        for lb in NB_LB:
            cols[f"rv{lb}"][s] = pd.Series(r.rolling(lb, min_periods=lb).std().to_numpy() * np.sqrt(252), idx)
        cols["dv60"][s] = pd.Series(pd.Series(d["rc"] * d["v"]).rolling(60, min_periods=60).mean().to_numpy(), idx)
        mom = np.full(len(c), np.nan)
        mom[120:] = c[120:] / c[:-120] - 1
        cols["mom120"][s] = pd.Series(mom, idx)
        cols["max20"][s] = pd.Series(r.rolling(20, min_periods=20).max().to_numpy(), idx)
        cols["dist"][s] = pd.Series(c / pd.Series(c).rolling(252, min_periods=1).max().to_numpy() - 1, idx)
        fac = np.ones(len(c))
        for day, ratio in splits.get(s, []):
            if ratio > 0:
                fac[d["days"] < day] *= ratio
        cols["px"][s] = pd.Series(d["rc"] * fac, idx)
    pct = {}
    for k, v in cols.items():
        pct[k] = pd.DataFrame(v).sort_index().rank(axis=1, pct=True)
    return pct


# ============================================================================ 진입·거래
def entries(days, n, sym, d0, d1, H, cut=None, lag_entry=0):
    """신호일 s(종목 자기 봉 번호, s % 10 == 이름 위상)는 d0 전날 ~ d1. 진입 e = s+1(+lag). 반환 (s, e)."""
    i0 = int(np.searchsorted(days, d0, "left"))
    i1 = int(np.searchsorted(days, d1, "right"))
    lo, hi = max(i0 - 1, MIN_POS), min(i1 - 1, n - 2)
    if hi < lo:
        return np.array([], int), np.array([], int)
    s = np.arange(lo, hi + 1)
    s = s[(s % EVERY) == (sum(map(ord, sym)) % EVERY)]
    e = s + 1 + lag_entry
    ok = e + H - 1 <= n - 1
    s, e = s[ok], e[ok]
    if cut is not None and len(e):
        ok = days[e + H - 1] <= cut
        s, e = s[ok], e[ok]
    return s, e


def first_hit(m):
    a = m.any(1)
    return a, np.where(a, m.argmax(1), -1)


def sim(d, e, H, ps, extra=False, slip=0.0, end_slip=0.0):
    """e일 시가 진입, H거래일. 반환 dict: bh, ret[p], trig[p], tday[p](체결일 번호), 그리고 extra면 다른 규칙·시점 변형."""
    k = len(e)
    I = e[:, None] + np.arange(H)[None, :]
    O, Hi, Lo, Cl = d["o"][I], d["h"][I], d["l"][I], d["c"][I]
    ent = O[:, 0] * (1 + slip)
    endpx = Cl[:, -1] * (1 - end_slip)
    bh = endpx / ent - 1
    peak = np.maximum.accumulate(np.concatenate([ent[:, None], Cl[:, :-1]], axis=1), axis=1)   # t−1까지 최고 종가
    rows = np.arange(k)
    out = {"bh": bh, "ret": {}, "trig": {}, "tday": {}}
    for p in ps:
        L = peak * (1 - p)
        trig, t = first_hit((O <= L) | (Lo <= L))
        tt = np.clip(t, 0, None)
        fill = np.where(O[rows, tt] <= L[rows, tt], O[rows, tt], L[rows, tt]) * (1 - slip)
        out["ret"][p] = np.where(trig, fill / ent - 1, bh)
        out["trig"][p] = trig
        out["tday"][p] = t
    if extra:
        p = CAND["p"]
        L = peak * (1 - p)
        # 손절 하루 늦춤: 선에 닿은 다음 날 시가(마지막 날 닿으면 기간끝 종가)
        trig, t = first_hit((O <= L) | (Lo <= L))
        t2 = np.clip(t + 1, 0, H - 1)
        late = trig & (t + 1 <= H - 1)
        out["ret_late"] = np.where(late, O[rows, t2] * (1 - slip) / ent - 1, bh)
        # 종가 기준: t일 종가 ≤ 선 → t+1 시가
        trigc, tc = first_hit(Cl[:, :-1] <= L[:, :-1])
        tc2 = np.clip(tc + 1, 0, H - 1)
        out["ret_close"] = np.where(trigc, O[rows, tc2] * (1 - slip) / ent - 1, bh)
        out["trig_close"] = trigc
        # 고정 −15%(진입가 대비, 장중)
        Lf = np.repeat((ent * 0.85)[:, None], H, axis=1)
        trf, tf = first_hit((O <= Lf) | (Lo <= Lf))
        tf_ = np.clip(tf, 0, None)
        fillf = np.where(O[rows, tf_] <= Lf[rows, tf_], O[rows, tf_], Lf[rows, tf_])
        out["ret_fixed15"] = np.where(trf, fillf / ent - 1, bh)
        # SMA50: t일 종가 < SMA50(t 포함) → t+1 시가(t ≤ H−2)
        c = d["c"]
        sma = pd.Series(c).rolling(50, min_periods=50).mean().to_numpy()
        S = sma[I]
        trs, ts = first_hit(Cl[:, :-1] < S[:, :-1])
        ts2 = np.clip(ts + 1, 0, H - 1)
        out["ret_sma50"] = np.where(trs, O[rows, ts2] / ent - 1, bh)
        out["open_rel"] = (O / ent[:, None]).astype(np.float32)
    return out


def run(D, syms, w, H, ps, extra=False, slip=0.0, end_slip=0.0, lag_entry=0):
    d0, d1 = WIN[w]
    cut = CUT_FRONT if w == "2026앞" else None
    recs = []
    for si, s in enumerate(syms):
        d = D[s]
        sig, e = entries(d["days"], len(d["c"]), s, d0, d1, H, cut, lag_entry)
        if not len(e):
            continue
        r = sim(d, e, H, ps, extra, slip, end_slip)
        rec = {"si": np.full(len(e), si), "e": e, "entry": d["days"][e], "sday": d["days"][sig],
               "sday1": d["days"][sig - 1], "bh": r["bh"]}
        for p in ps:
            rec[f"r{p}"] = r["ret"][p]
            rec[f"x{p}"] = r["trig"][p]
            rec[f"t{p}"] = r["tday"][p]
        if extra:
            for k in ("ret_late", "ret_close", "trig_close", "ret_fixed15", "ret_sma50"):
                rec[k] = r[k]
            rec["open_rel"] = list(r["open_rel"])
        recs.append(pd.DataFrame(rec))
    return pd.concat(recs, ignore_index=True)


def lookup(pctdf, days, syms_arr):
    ri = pctdf.index.get_indexer(days)
    ci = pctdf.columns.get_indexer(syms_arr)
    A = pctdf.to_numpy()
    ok = (ri >= 0) & (ci >= 0)
    v = np.full(len(days), np.nan)
    v[ok] = A[ri[ok], ci[ok]]
    return v


# ============================================================================ 통계
def mt(dif, entry):
    g = pd.Series(dif).groupby(pd.Series(entry).str[:7].to_numpy()).mean()
    if len(g) < 3 or g.std(ddof=1) == 0:
        return float("nan"), int(len(g))
    return float(g.mean() / (g.std(ddof=1) / np.sqrt(len(g)))), int(len(g))


def stats(x, bh, entry, trig, tday, H):
    n = len(x)
    if n == 0:
        return {"n": 0}
    dif = x - bh
    t, m = mt(dif, entry)
    days_in = np.where(trig, tday + 1, H)

    def wl(v):
        w, lo = v[v > 0], v[v <= 0]
        return float(len(w) / len(v)), (float(w.mean() / -lo.mean()) if len(w) and len(lo) and lo.mean() < 0 else float("nan"))

    wr, pf = wl(x)
    wrb, pfb = wl(bh)
    return {"n": int(n), "보유_기대값": float(bh.mean()), "규칙_기대값": float(x.mean()), "짝차": float(dif.mean()),
            "월t": t, "월수": m, "발동률": float(trig.mean()), "규칙이_나은비율": float((dif > 1e-12).mean()),
            "보유_P30": float((bh <= -0.3).mean()), "규칙_P30": float((x <= -0.3).mean()),
            "보유_P20": float((bh <= -0.2).mean()), "규칙_P20": float((x <= -0.2).mean()),
            "보유_하위5": float(np.quantile(bh, 0.05)), "규칙_하위5": float(np.quantile(x, 0.05)),
            "보유_하위1": float(np.quantile(bh, 0.01)), "규칙_하위1": float(np.quantile(x, 0.01)),
            "보유_승률": wrb, "규칙_승률": wr, "보유_손익비": pfb, "규칙_손익비": pf,
            "보유_보유일당": float(bh.sum() / (n * H)), "규칙_보유일당": float(x.sum() / days_in.sum())}


def cand_mask(df, pct, lb=60, q=1 / 3, lag_info=False, syms=None):
    days = df["sday1" if lag_info else "sday"].to_numpy()
    v = lookup(pct[f"rv{lb}"], days, np.asarray(syms)[df["si"].to_numpy()])
    return np.nan_to_num(v, nan=9) <= q + 1e-12


def cand_stats(df, mask, H, p=0.25, col=None):
    sub = df[mask]
    x = sub[col].to_numpy() if col else sub[f"r{p}"].to_numpy()
    return stats(x, sub["bh"].to_numpy(), sub["entry"].to_numpy(), sub[f"x{p}"].to_numpy(), sub[f"t{p}"].to_numpy(), H)


# ============================================================================ 100칸 고르기 재현
def stktail_cells(df, pct, syms):
    sa = np.asarray(syms)[df["si"].to_numpy()]
    days = df["sday"].to_numpy()

    def ter(key):
        v = lookup(pct[key], days, sa)
        return np.where(np.isnan(v), 0, np.where(v <= 1 / 3, 1, np.where(v <= 2 / 3, 2, 3)))

    liq, vol, mom, dist = ter("dv60"), ter("rv60"), ter("mom120"), ter("dist")
    mx, px = lookup(pct["max20"], days, sa), lookup(pct["px"], days, sa)
    lot = (mx > 0.8) & (px <= 1 / 3)
    B = {"전체": np.ones(len(df), bool)}
    for lab, arr in (("저유동", liq == 1), ("중유동", liq == 2), ("고유동", liq == 3), ("저변동", vol == 1), ("중변동", vol == 2),
                     ("고변동", vol == 3), ("약모멘텀", mom == 1), ("중모멘텀", mom == 2), ("강모멘텀", mom == 3),
                     ("고점먼", dist == 1), ("고점중간", dist == 2), ("고점가까운", dist == 3)):
        B[lab] = arr
    B["복권형"] = lot
    B["비복권"] = (~lot) & (vol > 0)
    B["고변동&복권"] = (vol == 3) & lot
    B["고변동&저유동"] = (vol == 3) & (liq == 1)
    B["고변동&약모멘텀"] = (vol == 3) & (mom == 1)
    B["고변동&강모멘텀"] = (vol == 3) & (mom == 3)
    B["고변동&고점먼"] = (vol == 3) & (dist == 1)
    R = {"추적-20%": df["r0.2"].to_numpy(), "추적-25%": df["r0.25"].to_numpy(), "SMA50 이탈": df["ret_sma50"].to_numpy(),
         "고정-15%": df["ret_fixed15"].to_numpy(), "추적-20% 절반": 0.5 * df["r0.2"].to_numpy() + 0.5 * df["bh"].to_numpy()}
    bh = df["bh"].to_numpy()
    out = {}
    for b, m in B.items():
        if m.sum() < MIN_N:
            continue
        for rn, x in R.items():
            out[f"{b}|{rn}"] = float((x[m] - bh[m]).mean())
    return out


# ============================================================================ 위약
def placebo(sub, H, seeds=N_SEEDS, p=0.25):
    """A: 같은 나가기 횟수·같은 청산일 분포를 무작위 거래에. B: 손절된 같은 거래에 청산일만 섞기. 청산가 = 그날 시가."""
    bh = sub["bh"].to_numpy()
    trig = sub[f"x{p}"].to_numpy()
    tday = sub[f"t{p}"].to_numpy()
    OR = np.vstack(sub["open_rel"].to_numpy())
    actual = float((sub[f"r{p}"].to_numpy() - bh).mean())
    K = int(trig.sum())
    td = tday[trig]
    idx_trig = np.flatnonzero(trig)
    A, Bv = [], []
    for sd in range(seeds):
        rng = np.random.default_rng(1000 + sd)
        pick = rng.choice(len(bh), K, replace=False)
        days = rng.permutation(td)
        ret = bh.copy()
        ret[pick] = OR[pick, days] - 1
        A.append(float((ret - bh).mean()))
        days_b = rng.permutation(td)
        ret = bh.copy()
        ret[idx_trig] = OR[idx_trig, days_b] - 1
        Bv.append(float((ret - bh).mean()))
    A, Bv = np.array(A), np.array(Bv)
    return {"실제": actual, "나간수": K, "A_중앙": float(np.median(A)), "A_최대": float(A.max()), "A_최소": float(A.min()),
            "A_실제이상비율": float((A >= actual).mean()), "A_순위": int((A > actual).sum()) + 1,
            "B_중앙": float(np.median(Bv)), "B_최대": float(Bv.max()), "B_최소": float(Bv.min()),
            "B_실제이상비율": float((Bv >= actual).mean()), "B_순위": int((Bv > actual).sum()) + 1}


# ============================================================================ 본체
def main():
    t0 = time.time()
    D = load()
    allsyms = sorted(D)
    bad = [s for s in allsyms if is_bad(D[s]["c"])]
    syms = [s for s in allsyms if s not in set(bad)]
    print("종목", len(allsyms), "자료 오류 의심", len(bad), "→", len(syms), flush=True)
    splits = json.loads((CACHE / "p4_STKTAIL_splits.json").read_text(encoding="utf-8"))
    pct = features(D, syms, splits)
    print("지표", round(time.time() - t0), "초", flush=True)
    res = {"meta": {"종목": len(allsyms), "자료오류": len(bad), "사용": len(syms),
                    "격자": {"p": NB_P, "lb": NB_LB, "q": NB_Q, "H": NB_H, "조합": len(NB_P) * len(NB_LB) * len(NB_Q) * len(NB_H)}}}

    # ---- 1) 재계산(H60, 창 13개) + 100칸 + 위약
    main_rows, cells100, plac = {}, {}, {}
    T60 = {}
    for w in WIN:
        df = run(D, syms, w, 60, NB_P, extra=True)
        T60[w] = df
        m = cand_mask(df, pct, syms=syms)
        st = cand_stats(df, m, 60)
        sub = df[m]
        for k, col in (("늦춤_손절", "ret_late"), ("종가기준", "ret_close")):
            st[f"짝차_{k}"] = float((sub[col] - sub["bh"]).mean())
        st["발동률_종가기준"] = float(sub["trig_close"].mean())
        m1 = cand_mask(df, pct, lag_info=True, syms=syms)
        st["짝차_정보하루늦춤"] = float((df["r0.25"][m1] - df["bh"][m1]).mean())
        st["n_정보하루늦춤"] = int(m1.sum())
        main_rows[w] = st
        if w in MAIN_W or w in ("2022", "2020", "2018"):
            cells100[w] = stktail_cells(df, pct, syms)
            plac[w] = placebo(sub, 60)
        print(w, "거래", len(df), "후보", int(m.sum()), "짝차", round(st["짝차"] * 100, 3), round(time.time() - t0), "초", flush=True)
    res["main"] = main_rows
    res["placebo"] = plac
    # 폭락 시기 진입
    crash = {}
    for cn, (w, a, b) in CRASH.items():
        df = T60[w]
        m = cand_mask(df, pct, syms=syms) & (df["entry"] >= a).to_numpy() & (df["entry"] <= b).to_numpy()
        crash[cn] = cand_stats(df, m, 60)
    res["crash"] = crash
    # 100칸 고르기
    c100 = pd.DataFrame(cells100)
    front = c100["2026앞"]
    res["cells100"] = {"칸수": int(front.notna().sum()), "앞_최대칸": front.idxmax(), "앞_최대값": float(front.max()),
                       "앞_플러스칸": int((front > 0).sum()), "앞_두번째": front.sort_values().index[-2],
                       "앞_두번째값": float(front.sort_values().iloc[-2]),
                       "뒤_플러스비율": float((c100["2026뒤"] > 0).mean()), "2025_플러스비율": float((c100["2025"] > 0).mean()),
                       "2022_플러스비율": float((c100["2022"] > 0).mean()),
                       "상위5": {k: {w: float(c100.loc[k, w]) for w in c100.columns} for k in front.sort_values().index[-5:][::-1]}}
    print("100칸 앞 최대", res["cells100"]["앞_최대칸"], res["cells100"]["앞_최대값"], flush=True)
    del T60

    # ---- 2) 시점·비용 변형(앞·뒤·2025·2022)
    var = {}
    for w in MAIN_W + ["2022"]:
        r = {}
        dfs = run(D, syms, w, 60, (0.25,), slip=SLIP)
        m = cand_mask(dfs, pct, syms=syms)
        r["미끄러짐(진입·손절)"] = float((dfs["r0.25"][m] - dfs["bh"][m]).mean())
        dfs2 = run(D, syms, w, 60, (0.25,), slip=SLIP, end_slip=SLIP)
        m2 = cand_mask(dfs2, pct, syms=syms)
        r["미끄러짐(모든 체결)"] = float((dfs2["r0.25"][m2] - dfs2["bh"][m2]).mean())
        dfl = run(D, syms, w, 60, (0.25,), extra=True, lag_entry=1)
        ml = cand_mask(dfl, pct, syms=syms)
        r["진입하루늦춤"] = float((dfl["r0.25"][ml] - dfl["bh"][ml]).mean())
        ml1 = cand_mask(dfl, pct, lag_info=True, syms=syms)
        r["모두늦춤(정보·진입·손절)"] = float((dfl["ret_late"][ml1] - dfl["bh"][ml1]).mean())
        r["n_진입하루늦춤"] = int(ml.sum())
        var[w] = r
        print("변형", w, {k: round(v * 100, 3) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    res["variants"] = var

    # ---- 보유 기간 민감도
    sens = {}
    for w, H in H_SENS.items():
        df = run(D, syms, w, H, (0.25,))
        m = cand_mask(df, pct, syms=syms)
        sens[f"{w}_H{H}"] = cand_stats(df, m, H)
    res["sens"] = sens

    # ---- 3) 이웃 625개(앞·뒤·2025)
    nb = []
    for w in MAIN_W:
        for H in NB_H:
            df = run(D, syms, w, H, NB_P)
            sa = np.asarray(syms)[df["si"].to_numpy()]
            pv = {lb: lookup(pct[f"rv{lb}"], df["sday"].to_numpy(), sa) for lb in NB_LB}
            bh = df["bh"].to_numpy()
            for p in NB_P:
                dif = df[f"r{p}"].to_numpy() - bh
                for lb in NB_LB:
                    v = np.nan_to_num(pv[lb], nan=9)
                    for q in NB_Q:
                        m = v <= q + 1e-12
                        nb.append({"창": w, "H": H, "p": p, "lb": lb, "q": round(q, 4), "n": int(m.sum()),
                                   "짝차": float(dif[m].mean()) if m.sum() else float("nan")})
            print("이웃", w, "H", H, round(time.time() - t0), "초", flush=True)
    NB = pd.DataFrame(nb)
    piv = NB.pivot_table(index=["H", "p", "lb", "q"], columns="창", values="짝차")
    pivn = NB.pivot_table(index=["H", "p", "lb", "q"], columns="창", values="n")
    ok = (pivn[MAIN_W] >= MIN_N).all(axis=1)
    allpos = (piv[MAIN_W] > 0).all(axis=1) & ok
    h60 = piv.index.get_level_values("H") == 60
    res["neighbors"] = {
        "조합": int(ok.sum()), "앞뒤2025_모두플러스": int(allpos.sum()), "비율": float(allpos.sum() / ok.sum()),
        "H60_조합": int((ok & h60).sum()), "H60_모두플러스": int((allpos & h60).sum()),
        "앞플러스": int(((piv["2026앞"] > 0) & ok).sum()), "뒤플러스": int(((piv["2026뒤"] > 0) & ok).sum()),
        "2025플러스": int(((piv["2025"] > 0) & ok).sum()),
        "앞_최대": float(piv["2026앞"][ok].max()), "앞_최대칸": [float(x) for x in piv["2026앞"][ok].idxmax()],
        "앞_중앙": float(piv["2026앞"][ok].median()), "2025_중앙": float(piv["2025"][ok].median()),
        "뒤_중앙": float(piv["2026뒤"][ok].median()),
        "1차원": {},
    }
    base = (60, 0.25, 60, round(1 / 3, 4))
    for name, vals, pos in (("p", NB_P, 1), ("lb", NB_LB, 2), ("q", NB_Q, 3), ("H", NB_H, 0)):
        row = {}
        for v in vals:
            key = list(base)
            key[pos] = round(v, 4) if name == "q" else v
            key = tuple(key)
            if key in piv.index:
                row[str(round(v, 3))] = {w: float(piv.loc[key, w]) for w in MAIN_W}
        res["neighbors"]["1차원"][name] = row
    # 앞 양수 집합 → 뒤·2025 성공(무조건과 나란히)
    fp = (piv["2026앞"] > 0) & ok
    res["neighbors"]["앞플러스→뒤"] = float((piv["2026뒤"][fp] > 0).mean()) if fp.sum() else float("nan")
    res["neighbors"]["앞플러스→2025"] = float((piv["2025"][fp] > 0).mean()) if fp.sum() else float("nan")
    res["neighbors"]["무조건_뒤"] = float((piv["2026뒤"][ok] > 0).mean())
    res["neighbors"]["무조건_2025"] = float((piv["2025"][ok] > 0).mean())
    res["meta"]["초"] = round(time.time() - t0)
    OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)),
                        encoding="utf-8")
    print("저장", OUT_JSON, round(time.time() - t0), "초")


# ============================================================================ 보조 점검(본 계산 뒤 추가, 판정에는 보조로만)
def supp():
    """(a) 2026뒤 절단판: 신호일 5/29~10/8 진입을 모두 넣고 10/9에 끊는다(보유 = min(60, 남은 날)).
    (b) 창 경계 겹침: 신호일이 d0 전날~d1이라 진입이 d1 다음 날일 수 있다(2025 창의 2026-01-02 진입).
    (c) 2026앞 진입 월별 짝차(월 묶음 t의 바탕)."""
    t0 = time.time()
    D = load()
    syms = [s for s in sorted(D) if not is_bad(D[s]["c"])]
    splits = json.loads((CACHE / "p4_STKTAIL_splits.json").read_text(encoding="utf-8"))
    pct = features(D, syms, splits)
    rows = []
    for si, s in enumerate(syms):
        d = D[s]
        n, days = len(d["c"]), d["days"]
        lo = max(int(np.searchsorted(days, "2026-06-01", "left")) - 1, MIN_POS)
        sg = np.arange(lo, n - 1)
        sg = sg[(sg % EVERY) == (sum(map(ord, s)) % EVERY)]
        for g in sg:
            e = g + 1
            H = min(60, n - e)
            r = sim(d, np.array([e]), H, (0.25,))
            rows.append((si, days[g], days[e], H, float(r["bh"][0]), float(r["ret"][0.25][0]), bool(r["trig"][0.25][0])))
    T = pd.DataFrame(rows, columns=["si", "sday", "entry", "H", "bh", "r", "x"])
    v = lookup(pct["rv60"], T["sday"].to_numpy(), np.asarray(syms)[T["si"].to_numpy()])
    m = np.nan_to_num(v, nan=9) <= 1 / 3 + 1e-12
    Q = T[m]
    out = {"절단_2026뒤": {"n": int(len(Q)), "평균보유일": float(Q["H"].mean()), "짝차": float((Q["r"] - Q["bh"]).mean()),
                         "발동률": float(Q["x"].mean()), "보유_기대값": float(Q["bh"].mean()), "규칙_기대값": float(Q["r"].mean()),
                         "보유_P30": float((Q["bh"] <= -0.3).mean()), "규칙_P30": float((Q["r"] <= -0.3).mean()),
                         "월별": {k: {"n": int(len(g)), "짝차": float((g["r"] - g["bh"]).mean())}
                                for k, g in Q.groupby(Q["entry"].str[:7])}}}
    df25 = run(D, syms, "2025", 60, (0.25,))
    m25 = cand_mask(df25, pct, syms=syms)
    df26 = run(D, syms, "2026앞", 60, (0.25,))
    m26 = cand_mask(df26, pct, syms=syms)
    s25 = df25[m25]
    out["경계"] = {"2025_진입이_2026": int((s25["entry"] >= "2026-01-01").sum()), "2025_후보": int(len(s25)),
                  "2025_겹침뺀_짝차": float((s25["r0.25"] - s25["bh"])[s25["entry"] < "2026-01-01"].mean()),
                  "2026앞_첫날진입": int((df26[m26]["entry"] == "2026-01-02").sum())}
    s26 = df26[m26]
    out["앞_월별"] = {k: {"n": int(len(g)), "짝차": float((g["r0.25"] - g["bh"]).mean()), "발동률": float(g["x0.25"].mean())}
                    for k, g in s26.groupby(s26["entry"].str[:7])}
    res = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    res["supp"] = out
    OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)),
                        encoding="utf-8")
    print("보조 점검", round(time.time() - t0), "초", json.dumps(out, ensure_ascii=False)[:1200])


# ============================================================================ 보고서
def _p(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%"


def _pp(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%p"


def _q(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def _t(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:+.1f}"


def render(res: dict) -> str:
    M, V, PL, NB, S, C1 = res["main"], res["variants"], res["placebo"], res["neighbors"], res["sens"], res["cells100"]
    SP = res.get("supp", {})
    a, b, y25, y22 = M["2026앞"], M["2026뒤"], M["2025"], M["2022"]
    L = []
    A = L.append
    A("# 4단계 반박 검증 V_STKTAIL: G1 저변동 × 추적 −25% (2026-10-11)")
    A("")
    A("> 후보: 신호일 60일 실현변동성이 그날 단면 아래 1/3인 종목을 무작위 날짜(종목마다 10거래일에 한 번)에 다음 날 시가로 사서 60거래일 보유, "
      "산 뒤 전날까지 최고 종가 × 0.75 아래로 장중에 내려가면 전량 판다. 비교는 같은 거래를 60일 그냥 들고 있기. "
      "갈래 요약 `results/p4_STKTAIL.md`. 검증 코드 `code/p4_V_STKTAIL.py`(엔진 import 없음), 숫자 `results/p4_V_STKTAIL.json`.")
    A("")
    A("## 판정: refuted(쓸 만한 규칙으로서) — 숫자는 그대로 재현, 손절이 보탠 값은 없음")
    A("")
    A("## 결론")
    A("")
    A(f"1. **숫자는 엔진 없이 다시 짜도 똑같이 나왔다.** 거래 수(앞 {a['n']:,}·뒤 {b['n']:,}·2025 {y25['n']:,})가 같고 짝차도 같다: "
      f"앞 {_pp(a['짝차'], 2)}, 뒤 {_pp(b['짝차'], 2)}, 2025 {_pp(y25['짝차'], 2)}, 2022 {_pp(y22['짝차'], 2)}. "
      f"100칸 고르기를 다시 해도 같은 칸이 1등이고 앞 플러스 칸은 {C1['앞_플러스칸']}개다. 시점을 하루씩 늦추거나 미끄러짐 0.5%를 넣어도 부호는 그대로다.")
    A(f"2. **이긴 규칙이 아니라 '거의 안 걸려서 덜 진' 칸이다.** 이웃 {NB['조합']}개 중 앞·뒤·2025가 모두 플러스인 조합은 {NB['앞뒤2025_모두플러스']}개다. "
      f"손절 폭을 넓힐수록 0에 다가간다(−15% {_pp(NB['1차원']['p']['0.15']['2026앞'], 2)} → −35% {_pp(NB['1차원']['p']['0.35']['2026앞'], 2)}). "
      f"같은 수의 거래를 같은 보유일 분포로 무작위로 판 위약 20판보다 실제 손절이 앞·2025·2022·2018에서 모두 나빴다(21판 중 꼴찌). "
      f"손절은 무작위로 파는 것보다 나쁜 때(빠진 뒤 되돌기 직전)에 판다.")
    A(f"3. **꼬리도 맨 끝만 줄었다.** −30% 이하 확률은 {_q(a['보유_P30'])} → {_q(a['규칙_P30'])}로 줄었다. 그러나 −20% 이하 확률은 {_q(a['보유_P20'])} → {_q(a['규칙_P20'])}, "
      f"하위 5%는 {_p(a['보유_하위5'])} → {_p(a['규칙_하위5'])}로 나빠졌다(2025도 {_q(y25['보유_P20'])} → {_q(y25['규칙_P20'])}, {_p(y25['보유_하위5'])} → {_p(y25['규칙_하위5'])}). "
      f"−25% 근처에서 끊고 나온 거래가 늘어서다. 보험으로 쳐도 하위 1% 밖(−25% 아래)만 막는다.")
    A("")
    A("## 격자(사전 고정, 뒤·2025를 본 뒤 바꾸지 않음)")
    A("")
    A("- 대상: 일반 주식 2,441종목 중 자료 오류 의심 71종목을 빼고 2,370종목(갈래와 같은 수).")
    A("- 이웃(후보 ±2칸): 추적 폭 {15, 20, 25, 30, 35%}(5%p 간격) × 변동성 기간 {20, 40, 60, 90, 120일} × 묶음 문턱 {아래 1/5, 1/4, 1/3, 2/5, 1/2} "
      "× 보유 {40, 50, 60, 70, 80일}(10일 간격) = **625개**. 보유 60일 고정 3차원 125개도 따로 셌다.")
    A("- 위약: 시드 20개 × 2가지. A = 같은 묶음 거래에서 실제로 손절된 수만큼 무작위 거래를 골라, 실제 손절일들을 섞어 붙이고 그날 시가에 판다. "
      "B = 실제로 손절된 같은 거래들에 손절일만 서로 섞는다(같은 보유일 분포).")
    A("- 시점 변형 5개(손절 체결 하루 늦춤, 종가 기준 손절, 묶음 정보 하루 늦춤, 진입 하루 늦춤, 모두) × 비용 2개(미끄러짐 0.5%: 엔진 규약, 모든 체결).")
    A("- 창: 2026 앞(진입 1/2~, 60일째가 5/29 안), 2026 뒤(진입 6/1~, 10/9까지 60일을 채운 거래), 2025(보조), 2015~2024(맥락), "
      "폭락 시기 진입 3개(사후 정의, 맥락).")
    A("")
    A("## 변수 시점")
    A("")
    A("| 변수 | 언제 아는가 | 감사 결과 |")
    A("|---|---|---|")
    A("| 60일 실현변동성(일수익 표준편차 × √252) | 신호일 s 종가 | 진입은 s+1 시가. 정보를 s−1로 늦춰도 짝차 같음(앞 "
      f"{_pp(a['짝차_정보하루늦춤'], 2)}) |")
    A("| 단면 백분위(그날 2,370종목 중 순위) | 신호일 s(그날 전 종목 종가) | 미래 가격 없음. 단, 종목표 자체가 2026-10 생존 종목(생존편향) |")
    A("| 손절선 = 산 뒤 t−1일까지 최고 종가 × 0.75 | t−1일 종가 | t일 시가가 선 아래면 시가, 장중 저가가 닿으면 선 가격. "
      f"하루 늦춤 {_pp(a['짝차_늦춤_손절'], 2)}, 종가 기준 {_pp(a['짝차_종가기준'], 2)} |")
    A("| 배당 수정 가격 | 2026-10 기준 역산 | 수익 계산에만 영향, 신호 시점과 무관 |")
    A("| 창 소속 | 진입일(2026 앞은 60일째 날짜도) | 60일째가 5/29 안인 거래만 앞에 넣음 → 앞 구간 판단에 6/1 이후 가격 없음 |")
    A("")
    # ---- 1 재현
    A("## 1. 독립 재계산(엔진 import 없이 parquet만)")
    A("")
    A("| 창 | 항목 | 주장 | 재계산 | 차이 |")
    A("|---|---|---|---|---|")
    for w, cl in CLAIM.items():
        src = S.get(w) if "_H" in w else M.get(w)
        for k, v in cl.items():
            mine = V[w]["미끄러짐(진입·손절)"] if k == "미끄러짐_짝차" else src.get(k)
            if k == "n":
                A(f"| {w} | 거래 수 | {v:,} | {mine:,} | {mine - v:+,} |")
            elif k == "월t":
                A(f"| {w} | 월 묶음 t | {v:+.1f} | {_t(mine)} | {mine - v:+.1f} |")
            elif k in ("짝차", "미끄러짐_짝차"):
                A(f"| {w} | {'짝차' if k == '짝차' else '짝차(미끄러짐 0.5%)'} | {_pp(v, 2)} | {_pp(mine, 2)} | {_pp(mine - v, 2)} |")
            else:
                lab = {"발동률": "발동률", "보유_P30": "P(−30%) 보유", "규칙_P30": "P(−30%) 규칙", "보유_하위5": "하위 5% 보유",
                       "규칙_하위5": "하위 5% 규칙"}[k]
                f = _p if "하위" in k else _q
                A(f"| {w} | {lab} | {f(v)} | {f(mine)} | {_pp(mine - v)} |")
    A("")
    A("- 주장 숫자는 갈래 md의 반올림 값이라 0.0x%p 차이는 반올림이다. 0.5%p를 넘는 차이는 없다.")
    A("")
    # ---- 2 핵심 표
    A("## 2. 핵심 표: 상황 → 규칙 → 숫자(짝차 = 같은 거래에서 손절 − 그냥 보유, 60거래일 거래 1건 평균)")
    A("")
    A("| 상황(진입 전날 알 수 있음) | 규칙 | 2026 앞 | 2026 뒤 | 2025 | 2022 | 2020 | 2018 | 2020 1분기 진입 | 2018 4분기 진입 | P(−30%) 앞 보유→규칙 | 하위5% 앞 보유→규칙 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    cr = res["crash"]
    A(f"| 60일 변동성 아래 1/3 | 추적 −25%(후보) | {_pp(a['짝차'])} | {_pp(b['짝차'])} | {_pp(y25['짝차'])} | {_pp(y22['짝차'])} | "
      f"{_pp(M['2020']['짝차'])} | {_pp(M['2018']['짝차'])} | {_pp(cr['2020 1분기 진입']['짝차'])} | {_pp(cr['2018 4분기 진입']['짝차'])} | "
      f"{_q(a['보유_P30'])}→{_q(a['규칙_P30'])} | {_p(a['보유_하위5'])}→{_p(a['규칙_하위5'])} |")
    for pk, lab in (("0.15", "추적 −15%"), ("0.2", "추적 −20%"), ("0.3", "추적 −30%"), ("0.35", "추적 −35%")):
        r = NB["1차원"]["p"][pk]
        A(f"| 같은 묶음 | {lab}(이웃) | {_pp(r['2026앞'])} | {_pp(r['2026뒤'])} | {_pp(r['2025'])} | — | — | — | — | — | — | — |")
    A("")
    A("- 손절을 넓힐수록 짝차가 0(그냥 보유)에 붙는다. 후보가 1등인 이유는 '손절이 가장 덜 걸린 칸'이기 때문이다.")
    A("")
    # ---- 3 변형
    A("## 3. 시점 감사·하루 늦춘 판·미끄러짐")
    A("")
    A("| 변형 | 2026 앞 | 2026 뒤 | 2025 | 2022 |")
    A("|---|---|---|---|---|")
    W4 = ["2026앞", "2026뒤", "2025", "2022"]
    A("| 기본(재계산) | " + " | ".join(_pp(M[w]["짝차"], 2) for w in W4) + " |")
    A("| 손절 체결 하루 늦춤(닿은 다음 날 시가) | " + " | ".join(_pp(M[w]["짝차_늦춤_손절"], 2) for w in W4) + " |")
    A("| 종가 기준 손절(종가가 선 아래 → 다음 날 시가) | " + " | ".join(_pp(M[w]["짝차_종가기준"], 2) for w in W4) + " |")
    A("| 묶음 정보 하루 늦춤(s−1 백분위) | " + " | ".join(_pp(M[w]["짝차_정보하루늦춤"], 2) for w in W4) + " |")
    A("| 진입 하루 늦춤(신호 이틀 뒤 시가) | " + " | ".join(_pp(V[w]["진입하루늦춤"], 2) for w in W4) + " |")
    A("| 모두 늦춤(정보·진입·손절) | " + " | ".join(_pp(V[w]["모두늦춤(정보·진입·손절)"], 2) for w in W4) + " |")
    A("| 미끄러짐 0.5%(진입·손절 체결, 엔진 규약) | " + " | ".join(_pp(V[w]["미끄러짐(진입·손절)"], 2) for w in W4) + " |")
    A("| 미끄러짐 0.5%(기간끝 매도까지 모든 체결) | " + " | ".join(_pp(V[w]["미끄러짐(모든 체결)"], 2) for w in W4) + " |")
    A("")
    A("- 미래 정보 누수는 보이지 않는다(누수가 있었다면 늦춘 판이 나빠졌을 텐데, 비슷하거나 오히려 낫다). 손절을 하루 늦추거나 종가로 판단하면 덜 손해 본다: "
      "장중 −25%에 닿은 뒤 다음 날 조금 되돌기 때문이다. 어느 판도 앞·2025에서 플러스가 아니다.")
    A("")
    # ---- 4 이웃
    A(f"## 4. 이웃 {NB['조합']}개(파라미터마다 ±2칸)")
    A("")
    A(f"- 앞·뒤·2025 모두 플러스: **{NB['앞뒤2025_모두플러스']}/{NB['조합']}**(보유 60일 고정 {NB['H60_모두플러스']}/{NB['H60_조합']}).")
    A(f"- 앞 플러스 {NB['앞플러스']}개 → 그중 뒤 플러스 {_q(NB['앞플러스→뒤'], 0)}, 2025 플러스 {_q(NB['앞플러스→2025'], 0)}"
      f"(무조건 비율: 뒤 {_q(NB['무조건_뒤'], 0)}, 2025 {_q(NB['무조건_2025'], 0)}). 2025 플러스는 625개 중 {NB['2025플러스']}개.")
    A(f"- 앞 최대 칸: 보유 {NB['앞_최대칸'][0]:.0f}일 × 추적 −{NB['앞_최대칸'][1] * 100:.0f}% × 변동성 {NB['앞_최대칸'][2]:.0f}일 × 아래 "
      f"{NB['앞_최대칸'][3] * 100:.0f}% = {_pp(NB['앞_최대'], 2)}(사실상 0). 중앙값: 앞 {_pp(NB['앞_중앙'], 2)}, 뒤 {_pp(NB['뒤_중앙'], 2)}, 2025 {_pp(NB['2025_중앙'], 2)}.")
    A("")
    A("한 번에 하나씩 바꾼 판(나머지는 후보 값):")
    A("")
    A("| 파라미터 | 값 | 2026 앞 | 2026 뒤 | 2025 |")
    A("|---|---|---|---|---|")
    names = {"p": "추적 폭", "lb": "변동성 기간(일)", "q": "묶음 문턱(아래 몇 %)", "H": "보유(일)"}
    for k, rows in NB["1차원"].items():
        for v, r in rows.items():
            vv = f"−{float(v) * 100:.0f}%" if k == "p" else (f"{float(v) * 100:.0f}%" if k == "q" else v)
            A(f"| {names[k]} | {vv} | {_pp(r['2026앞'], 2)} | {_pp(r['2026뒤'], 2)} | {_pp(r['2025'], 2)} |")
    A("")
    # ---- 5 위약
    A("## 5. 위약(같은 거래의 무작위 청산, 시드 20개)")
    A("")
    A("| 창 | 실제 짝차 | 나간 거래 | 위약 A 중앙(최소~최대) | A 대비 순위(21판 중) | 위약 B 중앙(최소~최대) | B 대비 순위 |")
    A("|---|---|---|---|---|---|---|")
    for w, p in PL.items():
        A(f"| {w} | {_pp(p['실제'], 2)} | {p['나간수']:,} | {_pp(p['A_중앙'], 2)} ({_pp(p['A_최소'], 2)}~{_pp(p['A_최대'], 2)}) | {p['A_순위']}등 | "
          f"{_pp(p['B_중앙'], 2)} ({_pp(p['B_최소'], 2)}~{_pp(p['B_최대'], 2)}) | {p['B_순위']}등 |")
    A("")
    A("- 1등이 가장 좋다. A(같은 나가기 수·같은 보유일 분포를 무작위 거래에)에서 실제 손절은 앞·2025·2022·2018에서 21판 중 21등이다. "
      f"2020(V자 폭락)에서만 위약 사이({PL['2020']['A_순위']}등)다. 2026 뒤는 {PL['2026뒤']['A_순위']}등.")
    A("- B는 '결국 손절된 거래'를 미리 알고 고른 판이라 위약에 유리하다(손절 전 날짜에 팔면 당연히 덜 빠진다). 참고로만 둔다. "
      "그래도 실제 손절가가 같은 거래의 무작위 날 시가보다 시드 20개 모두에서 평균으로 낮았다는 점은, 손절이 그 거래의 낮은 자리에서 판다는 뜻이다.")
    A("")
    # ---- 6 장세
    A("## 6. 장세: 진입 연도별(60거래일)")
    A("")
    A("| 창 | n | 보유 기대값 | 규칙 기대값 | 짝차 | 월 묶음 t | 발동률 | P(−30%) 보유→규칙 | 하위5% 보유→규칙 |")
    A("|---|---|---|---|---|---|---|---|---|")
    order = ["2026앞", "2026뒤", "2025", "2024", "2023", "2022", "2021", "2020", "2019", "2018", "2017", "2016", "2015"]
    for w in order:
        r = M[w]
        A(f"| {w} | {r['n']:,} | {_p(r['보유_기대값'])} | {_p(r['규칙_기대값'])} | {_pp(r['짝차'], 2)} | {_t(r['월t'])} | {_q(r['발동률'])} | "
          f"{_q(r['보유_P30'])}→{_q(r['규칙_P30'])} | {_p(r['보유_하위5'])}→{_p(r['규칙_하위5'])} |")
    pos = [w for w in order[2:] if M[w]["짝차"] > 0]
    A("")
    A(f"- 2015~2025 11개 해 중 짝차 플러스는 {len(pos)}개({', '.join(pos)})뿐이다.")
    A("")
    A("폭락 시기 진입(사후 정의, 맥락):")
    A("")
    A("| 시기 | n | 짝차 | 발동률 | P(−30%) 보유→규칙 | 하위5% 보유→규칙 |")
    A("|---|---|---|---|---|---|")
    for k, r in cr.items():
        A(f"| {k} | {r['n']:,} | {_pp(r['짝차'], 2)} | {_q(r['발동률'])} | {_q(r['보유_P30'])}→{_q(r['규칙_P30'])} | {_p(r['보유_하위5'])}→{_p(r['규칙_하위5'])} |")
    A("")
    A("- 2020 1분기 진입은 손절이 꼬리를 크게 줄였지만(−30% 이하 22%→2%) V자 반등을 놓쳐 평균은 더 잃었다.")
    A("")
    A("보유 기간 민감도:")
    A("")
    A("| 창·보유 | n | 짝차 | 월 묶음 t | 발동률 | P(−30%) 보유→규칙 | 하위5% 보유→규칙 |")
    A("|---|---|---|---|---|---|---|")
    for k, r in S.items():
        A(f"| {k} | {r['n']:,} | {_pp(r['짝차'], 2)} | {_t(r['월t'])} | {_q(r['발동률'])} | {_q(r['보유_P30'])}→{_q(r['규칙_P30'])} | {_p(r['보유_하위5'])}→{_p(r['규칙_하위5'])} |")
    A("")
    # ---- 7 승률
    A("## 7. 승률·손익비·꼬리")
    A("")
    A("| 창 | 승률 보유→규칙 | 손익비 보유→규칙 | P(−20%) 보유→규칙 | P(−30%) | 하위 5% | 하위 1% | 보유일당 보유→규칙 |")
    A("|---|---|---|---|---|---|---|---|")
    for w in ("2026앞", "2026뒤", "2025", "2022"):
        r = M[w]
        A(f"| {w} | {_q(r['보유_승률'])}→{_q(r['규칙_승률'])} | {r['보유_손익비']:.2f}→{r['규칙_손익비']:.2f} | {_q(r['보유_P20'])}→{_q(r['규칙_P20'])} | "
          f"{_q(r['보유_P30'])}→{_q(r['규칙_P30'])} | {_p(r['보유_하위5'])}→{_p(r['규칙_하위5'])} | {_p(r['보유_하위1'])}→{_p(r['규칙_하위1'])} | "
          f"{_p(r['보유_보유일당'], 3)}→{_p(r['규칙_보유일당'], 3)} |")
    A("")
    A("- 보유일당(손절로 생긴 현금을 다른 데 쓸 수 있을 때)도 규칙 쪽이 낮다. 갈래 표의 '+0.01%→+0.01%'는 소수 둘째 자리 반올림이 차이를 가린 것이다.")
    A("")
    # ---- 8 고르기
    A("## 8. 고르기 재현(갈래의 100칸: 묶음 20 × 규칙 5)")
    A("")
    A(f"- 앞 짝차 1등: {C1['앞_최대칸'].replace('|', ' × ')} {_pp(C1['앞_최대값'], 2)}, 2등: {C1['앞_두번째'].replace('|', ' × ')} {_pp(C1['앞_두번째값'], 2)}. 앞 플러스 칸 {C1['앞_플러스칸']}/{C1['칸수']}.")
    A(f"- 무조건 플러스 비율: 뒤 {_q(C1['뒤_플러스비율'], 0)}, 2025 {_q(C1['2025_플러스비율'], 0)}, 2022 {_q(C1['2022_플러스비율'], 0)}(갈래와 같음).")
    A("")
    A("| 앞 상위 칸 | 2026 앞 | 2026 뒤 | 2025 | 2022 | 2020 | 2018 |")
    A("|---|---|---|---|---|---|---|")
    for k, r in C1["상위5"].items():
        A(f"| {k.replace('|', ' × ')} | {_pp(r['2026앞'], 2)} | {_pp(r['2026뒤'], 2)} | {_pp(r['2025'], 2)} | {_pp(r['2022'], 2)} | {_pp(r['2020'], 2)} | {_pp(r['2018'], 2)} |")
    A("")
    A("- 상위 5칸 중 4칸이 저변동 묶음이다. 손절이 가장 덜 걸리는 묶음이 '가장 덜 지는' 자리를 차지한다.")
    A("")
    # ---- 9 보조
    if SP:
        A("## 9. 보조 점검(본 계산 뒤 추가)")
        A("")
        t = SP["절단_2026뒤"]
        A(f"- 2026 뒤 절단판(6/1~10/8 진입을 모두 넣고 10/9에 끊음, 평균 보유 {t['평균보유일']:.0f}일): n {t['n']:,}, 짝차 {_pp(t['짝차'], 2)}, "
          f"발동률 {_q(t['발동률'])}, P(−30%) {_q(t['보유_P30'])}→{_q(t['규칙_P30'])}. 월별: "
          + ", ".join(f"{k} {_pp(v['짝차'], 2)}(n {v['n']:,})" for k, v in t["월별"].items()) + ".")
        g = SP["경계"]
        A(f"- 창 경계: 엔진은 신호일을 '창 첫날 전날 ~ 창 끝날'로 잡아 진입이 창 끝 다음 날일 수 있다. 2025 후보 {g['2025_후보']:,}건 중 "
          f"{g['2025_진입이_2026']:,}건이 2026-01-02 진입이라 2026 앞 첫날 진입({g['2026앞_첫날진입']:,}건)과 겹친다. 빼도 2025 짝차 {_pp(g['2025_겹침뺀_짝차'], 2)}.")
        A("- 2026 앞 진입 월별: " + ", ".join(f"{k} {_pp(v['짝차'], 2)}(n {v['n']:,}, 발동 {_q(v['발동률'])})" for k, v in SP["앞_월별"].items())
          + f". 월 묶음 t {_t(a['월t'])}는 월 3개(자유도 2)로 낸 값이라 약하다.")
        A("")
    # ---- 불일치·한계
    A("## 발견한 불일치·선택편향·한계")
    A("")
    A("- **불일치(작음)**: ① 갈래 md는 위약에서 'G1 이상인 판 0%'라 했지만 json은 0.5%(200판 중 1판)다. ② 보유일당 '+0.01%→+0.01%'는 반올림이 "
      f"규칙 쪽 하락(앞 {_p(a['보유_보유일당'], 3)}→{_p(a['규칙_보유일당'], 3)}, 2025 {_p(y25['보유_보유일당'], 3)}→{_p(y25['규칙_보유일당'], 3)})을 가렸다. "
      "③ 꼬리 주장은 P(−30%)만 보였다. 같은 거래의 하위 5%와 P(−20%)는 나빠졌다. 숫자 자체(짝차·거래 수·발동률)는 모두 맞다.")
    A("- **고르기 구조**: 사전 격자에 '손절 없음'이 없어서, 앞 짝차 최대 = 가장 덜 걸리는 칸이 뽑힌다. 이웃 625개에서도 손절 폭이 넓을수록, "
      "변동성 문턱이 낮을수록 0에 붙는다. G1이 갈래 위약(무작위 종목 묶음) 95%보다 높았던 것도 같은 구조 때문이다.")
    A("- **생존편향**: 종목표는 2026-10에 살아 있고 시총 10억 달러 이상인 종목이다. 망한 종목이 빠져 보유 쪽이 유리하고 손절 값은 깎인다. "
      "저변동 묶음은 상장폐지가 드물어 영향이 작은 편이지만, 결론의 방향(손절이 못 이김)은 이 한계 안에서만 말할 수 있다.")
    A("- **표본 겹침**: 같은 종목을 10거래일마다 사서 60일 들어 거래가 6겹으로 겹친다. 2026 앞은 진입 월 3개, 뒤는 2개뿐이라 월 묶음 t가 약하다.")
    A("- **비용**: 수수료 0(사용자 매매 조건). 미끄러짐 0.5%는 엔진 규약(손절 체결에만)이면 손절 쪽에 0.5% × 발동률만큼 불리하고, "
      "모든 체결에 물려도 결론이 같다.")
    A("- **배당**: 손절선은 배당 수정 가격으로 계산했다. 실제 화면 가격(배당락 반영 안 됨)과 60일에 한 번 1% 안팎 차이가 날 수 있다.")
    A("")
    A("## 다시 돌리기")
    A("")
    A("```bash")
    A("cd data/derived/analysis/stop_research/code")
    A("PYTHONIOENCODING=utf-8 py p4_V_STKTAIL.py          # 본 계산(약 2분) + 보조 점검 + 보고서")
    A("PYTHONIOENCODING=utf-8 py p4_V_STKTAIL.py md       # 저장된 json에서 보고서만")
    A("```")
    return "\n".join(L) + "\n"


def write_md():
    res = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    OUT_MD.write_text(render(res), encoding="utf-8")
    print("보고서", OUT_MD)


if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg == "all":
        main()
        supp()
    elif arg == "supp":
        supp()
    write_md()
