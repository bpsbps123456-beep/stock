"""7월 급락 뒤 '전고점에 다시 붙은 종목'과 '아직 먼 종목'의 차이 (2026-10-05, Path-Q 연구).

대상: AI 유니버스 중 급락 전(5/1~7/17) 종가 고점이 역대 고가의 15% 안이었고, 그 뒤 8/21까지 15% 넘게
빠진 종목('급락 코호트'). 결과는 같은 폴더 report.json(사람이 읽는 정리는 대화와 report.md).

요인
  규모·유동성: 시총(USD), 1억 달러 거래당 가격 변동(Amihud 근사), 하루 거래대금/시총
  가격 경로: 연초→급락 전 고점 상승 배수, 급락 깊이
  실적·밸류: 최근 분기 매출 증가율, 영업이익률, 선행 PER
  인기·보유: 애널리스트 수, 기관 보유율, 공매도 비율(주로 미국 종목에만 있음)
  큰손: 두 펀드의 저점 이후 실매매, 국내 개인·외국인·기관(국내만)
  시장: 한국·일본·대만·기타 아시아 vs 미국
상관은 인과가 아니다. 같은 기간 수익과 같은 기간 수급은 서로를 설명할 수 있다.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import ai_ath_gap  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT = Path(__file__).resolve().parent
PEAK_WIN = ("2026-05-01", "2026-07-17")
LOW_END = "2026-08-21"
REBOUND_FROM = "2026-07-30"
FX = {"KRW": "KRW=X", "JPY": "JPY=X", "TWD": "TWD=X", "CNY": "CNY=X", "HKD": "HKD=X", "EUR": "EURUSD=X"}


def fx_rates():
    import yfinance as yf
    out = {"USD": 1.0}
    for cur, tk in FX.items():
        last = float(yf.Ticker(tk).history(period="5d")["Close"].iloc[-1])
        out[cur] = 1.0 / last if cur == "EUR" else last  # 통화 단위 / 1 USD
    return out


def info(symbol):
    import yfinance as yf
    try:
        i = yf.Ticker(symbol).info or {}
    except Exception:
        i = {}
    return symbol, {k: i.get(k) for k in ("currency", "marketCap", "numberOfAnalystOpinions",
                                          "heldPercentInstitutions", "shortPercentOfFloat")}


def path_measures(b, fx):
    c, hi, v = b["Close"], b["High"], b["Volume"].astype(float)
    win = c[(c.index >= PEAK_WIN[0]) & (c.index <= PEAK_WIN[1])]
    if win.empty:
        return None
    p_at, p = win.idxmax(), float(win.max())
    ath_then = float(hi[(hi.index >= "2010-01-01") & (hi.index <= p_at)].max())
    after = c[(c.index >= p_at) & (c.index <= LOW_END)]
    l_at, low = after.idxmin(), float(after.min())
    start = c[c.index <= "2026-01-02"]
    now = float(c.iloc[-1])
    reb = b[b.index >= REBOUND_FROM]
    dv = (reb["Close"] * reb["Volume"]) / fx  # USD 거래대금
    ret = reb["Close"].pct_change().abs()
    impact = (ret / dv.replace(0, np.nan)).median() * 1e8 * 100  # 1억 달러당 % 변동(중앙값)
    return {"peak_date": p_at.date().isoformat(), "low_date": l_at.date().isoformat(),
            "near_before": p / ath_then - 1, "crash": low / p - 1, "rebound": now / low - 1,
            "vs_peak": now / p - 1, "recovery": (now - low) / (p - low) if p > low else np.nan,
            "runup": p / float(start.iloc[-1]) if len(start) else np.nan,
            "dollar_vol": float(dv.median()), "impact": float(impact) if pd.notna(impact) else np.nan}


def ols_hc1(df, y, xs):
    d = df[[y] + xs].dropna()
    Z = (d[xs] - d[xs].mean()) / d[xs].std(ddof=0)
    X = np.column_stack([np.ones(len(d)), Z.values])
    yy = d[y].values
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ yy
    e = yy - X @ beta
    n, k = X.shape
    meat = (X * e[:, None] ** 2).T @ X
    cov = xtx_inv @ meat @ xtx_inv * n / (n - k)
    t = beta / np.sqrt(np.diag(cov))
    r2 = 1 - (e @ e) / ((yy - yy.mean()) @ (yy - yy.mean()))
    return {"n": n, "r2": round(float(r2), 2),
            "coef": {x: {"beta": round(float(beta[i + 1]) * 100, 2), "t": round(float(t[i + 1]), 2)} for i, x in enumerate(xs)}}


def main():
    rows = ai_ath_gap.universe_rows()
    syms = [s for _, _, s in rows]
    sub_of = {s: (g, sub) for g, sub, s in rows}
    fx = fx_rates()
    with ThreadPoolExecutor(max_workers=8) as pool:
        bars = {s: b for s, b in pool.map(lambda s: ai_ath_gap.fetch(s, None), syms) if b is not None}
        infos = dict(pool.map(info, syms))
    cache = json.loads((ROOT / "data" / "derived" / "desk" / "giant.json").read_text(encoding="utf-8"))["stocks"]
    crash_flows = {}
    vpath = ROOT / "data" / "derived" / "analysis" / "giant_validation" / "validation.json"
    for r in json.loads(vpath.read_text(encoding="utf-8"))["kr_flow_stocks"]["급락기 수급 → 반등기 수익"]:
        crash_flows[r["s"]] = r

    recs = []
    for s in syms:
        if s not in bars or s not in cache:
            continue
        inf = infos.get(s, {})
        cur = inf.get("currency") or "USD"
        rate = fx.get(cur, np.nan)
        pm = path_measures(bars[s], rate)
        if pm is None:
            continue
        g = cache[s]
        mcap_usd = (inf.get("marketCap") or g.get("mcap") or np.nan) / rate
        funds = g.get("funds") or {}
        fund_net = sum((f.get("since_low", {}).get("kind") in ("매수", "신규")) - (f.get("since_low", {}).get("kind") in ("매도", "편출"))
                       for f in funds.values())
        kr = g.get("kr_flow") or {}
        cf = crash_flows.get(s, {})
        recs.append({
            "symbol": s, "name": g["name"], "group": sub_of[s][0], "sub": sub_of[s][1], "market": g["market"],
            "gap_now": g["gap_pct"] / 100, **pm,
            "log_mcap": np.log10(mcap_usd) if mcap_usd and mcap_usd > 0 else np.nan, "mcap_b": mcap_usd / 1e9,
            "log_impact": np.log10(pm["impact"]) if pm["impact"] and pm["impact"] > 0 else np.nan,
            "turnover": pm["dollar_vol"] / mcap_usd * 100 if mcap_usd else np.nan,
            "log_runup": np.log(pm["runup"]) if pm["runup"] and pm["runup"] > 0 else np.nan,
            "rev_growth": g.get("rev_growth"), "op_margin": g.get("op_margin"),
            "log_pe": np.log(g["fwd_pe"]) if g.get("fwd_pe") else np.nan,
            "analysts": inf.get("numberOfAnalystOpinions"), "inst_own": inf.get("heldPercentInstitutions"),
            "short_float": inf.get("shortPercentOfFloat"), "fund_net": fund_net,
            "kr_foreign": kr.get("foreign_pct"), "kr_inst": kr.get("inst_pct"), "kr_indiv": kr.get("indiv_pct"),
            "kr_indiv_crash": cf.get("indiv"), "is_kr": float(g["market"] == "한국"),
            "is_asia_other": float(g["market"] in ("일본", "대만", "중국", "홍콩")), "is_memory": float(sub_of[s][0] == "메모리·스토리지"),
        })
    df = pd.DataFrame(recs)
    cohort = df[(df.near_before >= -0.15) & (df.crash <= -0.15)].copy()
    cohort["band"] = pd.cut(cohort.gap_now, [-9, -0.25, -0.10, 9], labels=["먼 종목(-25% 밑)", "중간(-25~-10%)", "가까운 종목(-10% 이내)"])

    factors = ["log_mcap", "log_impact", "turnover", "log_runup", "crash", "rev_growth", "op_margin", "log_pe",
               "analysts", "inst_own", "short_float", "fund_net", "is_kr", "is_memory"]
    bands = cohort.groupby("band", observed=True).agg(
        n=("symbol", "size"), gap_now=("gap_now", "median"), crash=("crash", "median"), rebound=("rebound", "median"),
        runup=("runup", "median"), mcap_b=("mcap_b", "median"), impact=("impact", "median"), turnover=("turnover", "median"),
        rev_growth=("rev_growth", "median"), op_margin=("op_margin", "median"), pe=("log_pe", lambda x: float(np.exp(x.median()))),
        analysts=("analysts", "median"), inst_own=("inst_own", "median"), fund_net=("fund_net", "mean"),
        kr_share=("is_kr", "mean"), memory_share=("is_memory", "mean"))
    corr = {}
    for sub_name, sub in (("전체", cohort), ("미국", cohort[cohort.market == "미국"]), ("한국", cohort[cohort.market == "한국"])):
        corr[sub_name] = {}
        for f in factors + ["kr_foreign", "kr_inst", "kr_indiv", "kr_indiv_crash"]:
            d = sub[[f, "gap_now", "recovery"]].dropna()
            if len(d) >= 12 and d[f].nunique() > 2:
                r1, p1 = spearmanr(d[f], d.gap_now)
                r2, p2 = spearmanr(d[f], d.recovery)
                corr[sub_name][f] = {"n": len(d), "rho_gap": round(float(r1), 2), "p_gap": round(float(p1), 3),
                                     "rho_recovery": round(float(r2), 2), "p_recovery": round(float(p2), 3)}
    base = ["log_mcap", "log_runup", "crash", "rev_growth", "log_pe", "is_kr", "is_asia_other"]
    models = {"gap_now": ols_hc1(cohort, "gap_now", base),
              "recovery": ols_hc1(cohort, "recovery", base),
              "gap_now+memory": ols_hc1(cohort, "gap_now", base + ["is_memory"]),
              "gap_now+impact": ols_hc1(cohort, "gap_now", [b for b in base if b != "log_mcap"] + ["log_impact"])}
    out = {"population": {"universe": len(df), "cohort": len(cohort),
                          "not_near_before": int((df.near_before < -0.15).sum()),
                          "mild_crash": int(((df.near_before >= -0.15) & (df.crash > -0.15)).sum())},
           "bands": bands.round(3).reset_index().to_dict("records"), "corr": corr, "models": models,
           "cohort": cohort.round(4).to_dict("records"), "fx": fx}
    (OUT / "report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(json.dumps(out["population"], ensure_ascii=False))
        print(bands.round(3).to_string())
        for k, v in corr.items():
            print(f"\n[{k}] Spearman (gap_now / recovery)")
            for f, r in sorted(v.items(), key=lambda kv: -abs(kv[1]["rho_gap"])):
                print(f"  {f:<15} n={r['n']:<4} gap ρ={r['rho_gap']:+.2f} (p={r['p_gap']:.3f})   recovery ρ={r['rho_recovery']:+.2f} (p={r['p_recovery']:.3f})")
        for name, m in models.items():
            print(f"\n[OLS {name}] n={m['n']} R²={m['r2']} (계수: 1표준편차당 %p, HC1 t)")
            for f, c in m["coef"].items():
                print(f"  {f:<15} {c['beta']:+7.2f}  t={c['t']:+.2f}")


if __name__ == "__main__":
    main()
