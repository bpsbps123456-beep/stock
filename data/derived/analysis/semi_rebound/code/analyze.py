"""2026년 반도체 반등 국면별 주도주 지속성 검증과 10/22 저점 이후 관찰 순위.

질문(2026-10-28 사용자): 반도체 본격 반등이 오는 것 같다. 7월 반등을 참고해 살 만한 상위 10개는?

사전 고정한 절차:
1. 국면: SOXX 종가 10% 지그재그의 2026 저점(3/30, 6/5, 7/29, 9/14 완료)과 진행 중인 10/22 저점.
   하락 = SOXX 직전 고점일 → 저점일 종목 수익률, 반등 = 저점일 → 다음 SOXX 고점일.
   국면 길이가 달라 고정 10거래일 반등도 함께 잰다.
2. 시험 요인(3개, 모두 저점 시점에 아는 값):
   - 직전 반등(앞 국면의 저점→고점 수익률): '7월 주도주를 다시 산다'
   - 이번 하락 깊이(고점→저점): 많이 빠진 종목이 많이 튄다
   - 베타(고점 전 60거래일 SOXX 대비)
3. 채택 기준: 완료된 2026 네 국면 모두 같은 부호이고 평균 |ρ| ≥ 0.20.
   직전 반등 = 앞 저점 → 이번 고점(9/14 국면이면 7/29→8/17, 곧 '7월 반등').
4. 10/22 관찰 순위 = 채택 요인의 백분위 평균. 채택 요인이 없으면 순위를 만들지 않는다.

가격: yfinance 수정 종가(분할·배당 반영). 결과: ../results/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

# watchlist.AI_UNIVERSE의 반도체 네 묶음(칩 설계·파운드리·후공정·메모리·스토리지·장비·소재·테스트) 중 미국 상장
GROUPS = {
    "칩 설계": ["NVDA", "AMD", "CBRS", "AVGO", "MRVL", "INTC", "ARM", "QCOM", "SWKS",
              "CRDO", "ALAB", "RMBS", "MTSI", "TXN", "ADI", "NXPI", "MCHP", "ON", "MPWR",
              "STM", "NVTS", "AOSL"],
    "파운드리·후공정": ["TSM", "UMC", "GFS", "TSEM", "ASX", "AMKR"],
    "메모리·스토리지": ["MU", "SNDK", "SIMO", "STX", "WDC"],
    "장비·소재·테스트": ["AMAT", "LRCX", "KLAC", "ASML", "TER", "KEYS", "AEHR", "FORM", "ENTG", "AXTI"],
}
TICKERS = [t for g in GROUPS.values() for t in g]
GROUP_OF = {t: g for g, ts in GROUPS.items() for t in ts}
ZIGZAG = 0.10
BETA_DAYS = 60
FIXED = 10
MIN_ABS_RHO = 0.20


def load_prices() -> pd.DataFrame:
    cache = OUT / "prices.csv"
    raw = yf.download(TICKERS + ["SOXX", "QQQ"], start="2025-06-01", auto_adjust=True,
                      progress=False)["Close"]
    raw = raw.dropna(how="all")
    raw.to_csv(cache)
    return raw


def zigzag(s: pd.Series, th: float) -> list[tuple[str, pd.Timestamp]]:
    piv, mode, ext, ext_i = [], None, s.iloc[0], s.index[0]
    for d, v in s.items():
        if mode in (None, "up"):
            if v > ext:
                ext, ext_i = v, d
            elif v < ext * (1 - th):
                piv.append(("peak", ext_i)); mode, ext, ext_i = "down", v, d
        else:
            if v < ext:
                ext, ext_i = v, d
            elif v > ext * (1 + th):
                piv.append(("trough", ext_i)); mode, ext, ext_i = "up", v, d
    piv.append(("peak" if mode == "up" else "trough", ext_i))
    return piv


def episodes(px: pd.DataFrame) -> list[dict]:
    piv = zigzag(px["SOXX"].dropna(), ZIGZAG)
    out = []
    for i, (kind, d) in enumerate(piv):
        if kind != "trough" or i == 0:
            continue
        nxt = piv[i + 1][1] if i + 1 < len(piv) else px.index[-1]
        prev_trough = next((piv[j][1] for j in range(i - 1, -1, -1) if piv[j][0] == "trough"), None)
        out.append({"peak": piv[i - 1][1], "trough": d, "next_peak": nxt,
                    "prev_trough": prev_trough, "complete": i + 1 < len(piv)})
    return out


def ret(px: pd.DataFrame, a, b) -> pd.Series:
    return px.loc[b] / px.loc[a] - 1


def beta(px: pd.DataFrame, end) -> pd.Series:
    r = px.pct_change().loc[:end].iloc[-BETA_DAYS:]
    m = r["SOXX"]
    return r.apply(lambda c: c.cov(m) / m.var())


def spearman(a: pd.Series, b: pd.Series) -> dict:
    j = pd.concat([a, b], axis=1).dropna()
    rho, p = stats.spearmanr(j.iloc[:, 0], j.iloc[:, 1])
    return {"rho": round(float(rho), 3), "p": round(float(p), 4), "n": int(len(j))}


def episode_table(px: pd.DataFrame, ep: dict) -> pd.DataFrame:
    idx = px.index
    t_pos = idx.get_loc(ep["trough"])
    fixed_end = idx[min(t_pos + FIXED, len(idx) - 1)]
    df = pd.DataFrame({
        "drop": ret(px, ep["peak"], ep["trough"]),
        "rebound": ret(px, ep["trough"], ep["next_peak"]),
        "rebound10": ret(px, ep["trough"], fixed_end),
        "beta": beta(px, ep["peak"]),
    })
    if ep["prev_trough"] is not None:
        prev_peak = ep["peak"]
        df["prior_rebound"] = ret(px, ep["prev_trough"], prev_peak)
    return df.drop(index=["SOXX", "QQQ"])


def resid_rank(y: pd.Series, x: pd.Series) -> pd.Series:
    j = pd.concat([y.rank(), x.rank()], axis=1).dropna()
    b = np.polyfit(j.iloc[:, 1], j.iloc[:, 0], 1)
    return j.iloc[:, 0] - np.polyval(b, j.iloc[:, 1])


ENTRY_UP = 0.07  # bottom_confirm 연구의 확인 문턱(7%·8% 구별 안 됨, 저점 20일 유지 99.3%). 여기서 조정하지 않음
TOP_N = 10
HOLD = 20


def entry_test(px: pd.DataFrame, ep: dict, df: pd.DataFrame) -> dict:
    """첫 결과를 본 뒤 추가한 점검: 저점을 모르고 SOXX가 저점 +7%를 처음 넘긴 날 종가에 산다면."""
    s = px["SOXX"].loc[ep["trough"]:]
    hit = s[s >= s.iloc[0] * (1 + ENTRY_UP)]
    if hit.empty:
        return {}
    e = hit.index[0]
    pos = px.index.get_loc(e)
    end = px.index[min(pos + HOLD, len(px) - 1)]
    fwd = ret(px, e, end)
    path = px.loc[e:end] / px.loc[e] - 1
    stock = df.index
    top = df["beta"].dropna().sort_values(ascending=False).index[:TOP_N]
    return {
        "entry": e.date().isoformat(), "exit": end.date().isoformat(),
        "beta→fwd": spearman(df["beta"], fwd[stock]),
        "prior_rebound→fwd": spearman(df["prior_rebound"], fwd[stock]) if "prior_rebound" in df else None,
        "prior_rebound_net_beta→fwd": spearman(resid_rank(df["prior_rebound"], df["beta"]), fwd[stock])
        if "prior_rebound" in df else None,
        "top_beta": list(top),
        "top_beta_fwd": round(float(fwd[top].mean()), 4),
        "all_fwd": round(float(fwd[stock].mean()), 4),
        "soxx_fwd": round(float(fwd["SOXX"]), 4),
        "top_beta_worst_path": round(float(path[top].mean(axis=1).min()), 4),
        "soxx_worst_path": round(float(path["SOXX"].min()), 4),
    }


def main() -> None:
    px = load_prices()
    eps = episodes(px)
    tables = {ep["trough"].date().isoformat(): (ep, episode_table(px, ep)) for ep in eps
              if ep["trough"] >= pd.Timestamp("2026-01-01")}

    tests = {}
    for key, (ep, df) in tables.items():
        if not ep["complete"]:
            continue
        tests[key] = {
            "window": f'{ep["peak"].date()}→{ep["trough"].date()}→{ep["next_peak"].date()}',
            "soxx_drop": round(float(ret(px, ep["peak"], ep["trough"])["SOXX"]), 4),
            "soxx_rebound": round(float(ret(px, ep["trough"], ep["next_peak"])["SOXX"]), 4),
            **{f"{f}→{y}": spearman(df[f], df[y]) for f in ("prior_rebound", "drop", "beta")
               if f in df for y in ("rebound", "rebound10")},
            "beta→drop": spearman(df["beta"], df["drop"]),
            "entry_plus7": entry_test(px, ep, df),
        }

    # 채택: 같은 부호(모든 국면) + 평균 |ρ| ≥ 0.20, 목표는 저점→다음 고점 반등
    verdict = {}
    for f in ("prior_rebound", "drop", "beta"):
        rhos = [t[f"{f}→rebound"]["rho"] for t in tests.values() if f"{f}→rebound" in t]
        same = len(rhos) >= 2 and (all(r > 0 for r in rhos) or all(r < 0 for r in rhos))
        verdict[f] = {"rhos": rhos, "adopted": bool(same and np.mean(np.abs(rhos)) >= MIN_ABS_RHO),
                      "sign": int(np.sign(np.mean(rhos))) if rhos else 0}

    # 진행 중 국면(10/22 저점)
    live_key = max(tables)
    ep, now = tables[live_key]
    july = tables["2026-07-29"][1]
    now["july_rebound"] = july["rebound"]
    now["july_drop"] = july["drop"]
    now["since_low"] = ret(px, ep["trough"], px.index[-1])
    now["group"] = [GROUP_OF[t] for t in now.index]
    hi = px.loc[px.index[-1] - pd.Timedelta(days=365):].max()
    now["from_52w_high"] = (px.iloc[-1] / hi - 1).drop(["SOXX", "QQQ"])

    adopted = [f for f, v in verdict.items() if v["adopted"]]
    if adopted:
        score = sum(now[f].rank(pct=True, ascending=verdict[f]["sign"] > 0) for f in adopted) / len(adopted)
        now["score"] = score
        now = now.sort_values("score", ascending=False)

    now.round(4).to_csv(OUT / "live_2026-10-22.csv")
    summary = {"asof_close": px.index[-1].date().isoformat(), "episodes": tests, "verdict": verdict,
               "adopted": adopted, "live_trough": live_key,
               "soxx_since_low": round(float(ret(px, ep["trough"], px.index[-1])["SOXX"]), 4),
               "n_tickers": len(TICKERS)}
    (OUT / "tests.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    cols = ["group", "drop", "july_rebound", "beta", "since_low", "from_52w_high"] + (["score"] if adopted else [])
    print(now[cols].round(3).to_string())


if __name__ == "__main__":
    main()
