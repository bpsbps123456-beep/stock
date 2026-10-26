"""BRIEF.md의 지표 9개와 결과 변수를 하루 한 줄 표로 만든다 → results/daily.parquet.

모든 지표는 t일 장 마감 뒤 값이다. 결과 변수(Y1·Y2)는 t+1 종가부터 잰다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]  # 저장소 루트(watchlist.py)
CACHE = HERE.parent / "cache"
RES = HERE.parent / "results"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import fetch  # noqa: E402
import watchlist  # noqa: E402

Q_DIV = 0.006   # QQQ 배당수익률(연) 고정
H = 20          # 결과 창(거래일)
WIN, MINP = 250, 120
# 방향: +1이면 값이 높을수록 경고, −1이면 낮을수록 경고(BRIEF 표)
DIRECTION = {"EQ_PCR5": -1, "AI_CPCR5": -1, "SKEW": -1, "UPVOL10": +1, "VIX6M": -1,
             "QQQ_LSKEW": -1, "SPX_PSH20": +1, "EQ_PSH20": +1}
CALLHOT_PARTS = ["EQ_PCR5", "AI_CPCR5", "SKEW", "UPVOL10"]


def ai_us_symbols() -> list[str]:
    out = set()
    for subs in watchlist.AI_UNIVERSE.values():
        for syms in subs.values():
            out.update(s for s in syms if "." not in s)
    return sorted(out)


def rolling_pct(s: pd.Series) -> pd.Series:
    """지난 250거래일(자기 포함) 안에서 자기 값 이하의 비율. 앞날 값은 쓰지 않는다."""
    v = s.dropna()
    p = v.rolling(WIN, min_periods=MINP).apply(lambda w: (w <= w[-1]).mean(), raw=True)
    return p.reindex(s.index)


# ---- 블랙-숄즈 내재변동성(벡터 이분법) ----
def _bs(kind, S, K, T, r, q, v):
    sq = v * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * v * v) * T) / sq
    d2 = d1 - sq
    call = S * np.exp(-q * T) * ndtr(d1) - K * np.exp(-r * T) * ndtr(d2)
    put = K * np.exp(-r * T) * ndtr(-d2) - S * np.exp(-q * T) * ndtr(-d1)
    return np.where(kind == "C", call, put)


def implied_vol(kind, price, S, K, T, r, q):
    lo, hi = np.full(len(price), 0.01), np.full(len(price), 3.0)
    for _ in range(80):
        mid = (lo + hi) / 2
        too_high = _bs(kind, S, K, T, r, q, mid) > price
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo, mid)
    iv = (lo + hi) / 2
    intrinsic_ok = price > _bs(kind, S, K, T, r, q, np.full(len(price), 0.011))  # 하한 근처(가격 < 내재가치 수준)는 버린다
    return np.where(intrinsic_ok & (iv < 2.9), iv, np.nan)


def _local_iv(k, iv, k0, band=0.05):
    m = np.abs(k - k0) <= band
    if m.sum() >= 3:
        b = np.polyfit(k[m] - k0, iv[m], 1)
        return b[1]
    return iv[m].mean() if m.any() else np.nan


def qqq_long_skew(px: pd.DataFrame) -> pd.DataFrame:
    """만기별 IV(0.9F)·IV(1.1F)·IV(F)를 구한 뒤 1년 고정 만기로 선형 보간(양쪽 만기가 있을 때만)."""
    S, r = px["QQQ"], px["^IRX"] / 100
    per = []
    for f in sorted((CACHE / "leaps").glob("QQQ_*.parquet")):
        e = pd.Timestamp(f.stem.split("_")[1])
        d = pd.read_parquet(f)
        d = d[(d["volume"] > 0) & (d["close"] > 0) & d["date"].isin(S.index)].copy()
        d["S"], d["r"] = d["date"].map(S), d["date"].map(r).fillna(0.04)
        d["T"] = (e - pd.to_datetime(d["date"])).dt.days / 365.0
        d = d[d["T"] > 0.05]
        F = d["S"] * np.exp((d["r"] - Q_DIV) * d["T"])
        d = d[((d["kind"] == "P") & (d["strike"] < F)) | ((d["kind"] == "C") & (d["strike"] > F))].copy()
        d["iv"] = implied_vol(d["kind"].to_numpy(), d["close"].to_numpy(), d["S"].to_numpy(), d["strike"].to_numpy(),
                              d["T"].to_numpy(), d["r"].to_numpy(), Q_DIV)
        d["k"] = np.log(d["strike"] / (d["S"] * np.exp((d["r"] - Q_DIV) * d["T"])))
        d = d.dropna(subset=["iv"])
        for day, g in d.groupby("date"):
            k, iv = g["k"].to_numpy(), g["iv"].to_numpy()
            t, rr = g["T"].iloc[0], g["r"].iloc[0]
            per.append({"date": day, "expiry": e, "T": t, "r": rr, "iv90": _local_iv(k, iv, np.log(0.9)),
                        "iv110": _local_iv(k, iv, np.log(1.1)), "atm": _local_iv(k, iv, 0.0, band=0.03),
                        "iv90s": _local_iv(k, iv, np.log(0.9) - (rr - Q_DIV) * t)})  # 현물 대비 0.9 행사가
    per = pd.DataFrame(per).dropna(subset=["iv90", "iv110"])
    per["skew"] = per["iv90"] - per["iv110"]
    out = {}
    for day, g in per.groupby("date"):
        lo, hi = g[g["T"] <= 1.0], g[g["T"] > 1.0]
        if lo.empty or hi.empty:
            continue
        a, b = lo.loc[lo["T"].idxmax()], hi.loc[hi["T"].idxmin()]
        w = (1.0 - a["T"]) / (b["T"] - a["T"])
        out[day] = {k: (1 - w) * a[k] + w * b[k] for k in ("skew", "iv90", "iv110", "iv90s", "r")}
    o = pd.DataFrame(out).T.sort_index().rename(columns={"skew": "lskew_raw", "iv90": "liv90", "iv110": "liv110"})
    # 1년 만기 현물 대비 10% 외가격 풋 값(현물 대비 %) — 먼 만기 보험료(설명용, 신호 아님)
    put = _bs(np.array(["P"] * len(o)), 1.0, 0.9, 1.0, o["r"].to_numpy(float), Q_DIV, o["iv90s"].to_numpy(float))
    o["hedge1y"] = put * 100
    o.to_csv(RES / "qqq_long_skew_raw.csv")
    per.to_csv(RES / "qqq_long_skew_by_expiry.csv", index=False)
    return o


def fwd_outcomes(c: pd.Series) -> pd.DataFrame:
    a = c.to_numpy()
    n = len(a)
    mdd, ret = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n - H - 1):
        w = a[i + 1:i + H + 2]  # t+1 … t+21
        mdd[i] = (w / np.maximum.accumulate(w) - 1).min() * 100
        ret[i] = (w[-1] / w[0] - 1) * 100
    return pd.DataFrame({"mdd": mdd, "ret": ret}, index=c.index)


def main() -> None:
    RES.mkdir(exist_ok=True)
    cb = fetch.load_cboe()
    days = cb.index  # CBOE가 있는 날 = 분석일
    full = pd.read_parquet(CACHE / "yahoo.parquet")
    full = full[full.index <= days[-1]]  # 장중(아직 안 끝난 날) Yahoo 행은 뺀다
    px = full.reindex(days)
    syms = ai_us_symbols()
    occ = fetch.load_occ(syms)
    ai = occ.groupby("date")[["C_C", "C_P"]].sum().reindex(days)
    missing = sorted(set(syms) - set(occ["underlying"]))

    df = pd.DataFrame(index=days)
    df["EQ_PCR5"] = cb["equity_pv"].rolling(5).sum() / cb["equity_cv"].rolling(5).sum()
    df["AI_CPCR5"] = ai["C_P"].rolling(5).sum() / ai["C_C"].rolling(5).sum()
    df["SKEW"] = px["^SKEW"]
    up = (full["QQQ"].diff() > 0) & (full["^VXN"].diff() > 0)
    df["UPVOL10"] = up.astype(float).rolling(10).sum().reindex(days)
    df["VIX6M"] = px["^VIX6M"]
    ls = qqq_long_skew(px)
    df["QQQ_LSKEW"] = (ls["lskew_raw"] * 100).reindex(days).rolling(5, min_periods=3).median()
    df["HEDGE1Y"] = ls["hedge1y"].astype(float).reindex(days).rolling(5, min_periods=3).median()  # 설명용
    df["SPX_PSH20"] = (cb["spx_poi"] / (cb["spx_poi"] + cb["spx_coi"])).diff(20) * 100
    df["EQ_PSH20"] = (cb["equity_poi"] / (cb["equity_poi"] + cb["equity_coi"])).diff(20) * 100
    for k, sgn in DIRECTION.items():  # 방향 맞춘 백분위(1 = 가장 경고)
        df[f"{k}_w"] = rolling_pct(df[k] * sgn)
    parts = df[[f"{k}_w" for k in CALLHOT_PARTS]]
    df["CALLHOT"] = parts.mean(axis=1).where(parts.notna().sum(axis=1) >= 3)
    df["CALLHOT_w"] = rolling_pct(df["CALLHOT"])
    for a in ("QQQ", "SOXX"):  # 결과 창은 Yahoo 전체 거래일로 센다(CBOE가 빠진 날이 있어도 20거래일)
        o = fwd_outcomes(full[a].dropna()).reindex(days)
        df[f"{a}_mdd"], df[f"{a}_ret"] = o["mdd"], o["ret"]
        df[a] = px[a]
    for k in ("^VIX", "^VXN", "^VIX3M", "^VVIX", "^DSPX"):
        df[k] = px[k]
    df.to_parquet(RES / "daily.parquet")
    print("분석일", days[0], "~", days[-1], len(days), "| AI 미국 종목", len(syms), "OCC에 없음", missing)
    print(df[[c for c in df.columns if not c.endswith("_w")]].describe().T.round(3).to_string())


if __name__ == "__main__":
    main()
