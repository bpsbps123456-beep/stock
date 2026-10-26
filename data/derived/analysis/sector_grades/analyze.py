"""AI 섹터 등급과 '다음 회계연도 이익 증가율' (2026-10-06, Path-Q 연구).

사용자 가설: AI 안에서도 회복탄력성 등급이 갈리고(예: 광통신은 상위), 상위 등급은 FY27·28 이익 증가율이
더 크다. 변수는 미리 알 수 있는 값만 쓴다(사용자 원칙: 예측할 수 없는 사후 값은 근거가 아니다).

대상: 급락 코호트(../recovery_factors 정의, 233종목 중 186)
등급: 소분류(코호트 3종목 이상)의 회복률 중앙값. S 90%↑, A 60~90%, B 40~60%, C 20~40%, D 20% 미만
미리 아는 값(7/8 무렵 = 오늘부터 90일 전, 저점 7/29 이전의 yfinance 컨센서스 eps_trend 90daysAgo)
  g1   = 다음 회계연도 EPS / 올해 회계연도 EPS - 1 (대부분 FY27/FY26, 회계연도가 이른 회사는 FY28/FY27)
  pe1  = 급락 전 고점 가격 / 다음 회계연도 EPS (급락 전 선행 PER), pe1_low = 저점 가격 / 같은 EPS
사후 값(설명용): 다음 회계연도 EPS 수정률(90일 전→지금), 최근 분기 매출 증가율(대부분 7~8월 발표)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import ai_ath_gap  # noqa: E402
from desk import giant  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("rf", OUT.parent / "recovery_factors" / "analyze.py")
rf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rf)

GRADES = [("S", 0.90), ("A", 0.60), ("B", 0.40), ("C", 0.20), ("D", -9.0)]
MIN_MEMBERS = 3


def grade_of(recovery: float) -> str:
    return next(g for g, lo in GRADES if recovery >= lo)


def estimates(symbol: str) -> tuple[str, dict]:
    import yfinance as yf

    t = yf.Ticker(symbol)
    out: dict = {}
    try:
        et = t.eps_trend
        if et is not None and len(et) and {"0y", "+1y"} <= set(et.index):
            out.update(eps0_pre=et.loc["0y", "90daysAgo"], eps1_pre=et.loc["+1y", "90daysAgo"],
                       eps0_now=et.loc["0y", "current"], eps1_now=et.loc["+1y", "current"],
                       est_cur=et["currency"].iloc[0] if "currency" in et else None)
    except Exception:
        pass
    try:
        ee = t.earnings_estimate
        if ee is not None and len(ee) and "+1y" in ee.index:
            out.update(analysts1=ee.loc["+1y", "numberOfAnalysts"], eps_last=ee.loc["0y", "yearAgoEps"],
                       eps1_avg_now=ee.loc["+1y", "avg"])
            out.setdefault("est_cur", ee["currency"].iloc[0] if "currency" in ee else None)
    except Exception:
        pass
    try:
        info = t.info or {}
        out.update(price_cur=info.get("currency"), fy_end=info.get("nextFiscalYearEnd"),
                   info_fwd_pe=info.get("forwardPE"), info_fwd_eps=info.get("forwardEps"),
                   price_now=info.get("currentPrice") or info.get("regularMarketPrice"))
    except Exception:
        pass
    return symbol, out


def positive_ratio(a, b):
    return a / b - 1 if a and b and a > 0 and b > 0 else np.nan


def main():
    rows = ai_ath_gap.universe_rows()
    syms = [s for _, _, s in rows]
    sub_of = {s: (g, sub) for g, sub, s in rows}
    cache = json.loads(giant.CACHE_PATH.read_text(encoding="utf-8"))["stocks"]
    fx = rf.fx_rates()
    with ThreadPoolExecutor(max_workers=8) as pool:
        bars = {s: b for s, b in pool.map(lambda s: ai_ath_gap.fetch(s, None), syms) if b is not None}
        est = dict(pool.map(estimates, syms))
    recs = []
    for s in syms:
        if s not in bars or s not in cache:
            continue
        e = est.get(s, {})
        price_cur = e.get("price_cur") or giant.MARKET_CURRENCY.get(cache[s]["market"], "USD")
        pm = rf.path_measures(bars[s], fx.get(price_cur, np.nan))
        if pm is None:
            continue
        c = bars[s]["Close"]
        peak_px = float(c.loc[pd.Timestamp(pm["peak_date"])])
        low_px = float(c.loc[pd.Timestamp(pm["low_date"])])
        # EPS를 가격 통화로 맞춘다(ADR처럼 통화가 다르면 환율로 바꾼다).
        conv = 1.0
        est_cur = e.get("est_cur")
        if est_cur and price_cur and est_cur != price_cur and est_cur in fx and price_cur in fx:
            conv = fx[price_cur] / fx[est_cur]
        eps1_pre = (e.get("eps1_pre") or np.nan) * conv
        g = cache[s]
        recs.append({
            "symbol": s, "name": g["name"], "group": sub_of[s][0], "sub": sub_of[s][1], "market": g["market"], **pm,
            "mcap_b": (g.get("mcap_usd") or np.nan) / 1e9, "rev_growth_q": g.get("rev_growth"),
            "g1": positive_ratio(e.get("eps1_pre"), e.get("eps0_pre")),
            "g2": (np.sqrt(e["eps1_pre"] / e["eps_last"]) - 1
                   if e.get("eps1_pre") and e.get("eps_last") and e["eps1_pre"] > 0 and e["eps_last"] > 0 else np.nan),
            "pe1": peak_px / eps1_pre if eps1_pre and eps1_pre > 0 else np.nan,
            "pe1_low": low_px / eps1_pre if eps1_pre and eps1_pre > 0 else np.nan,
            "revision1": positive_ratio(e.get("eps1_now"), e.get("eps1_pre")),
            "analysts1": e.get("analysts1"), "has_trend": e.get("eps1_pre") is not None,
            "info_fwd_pe": e.get("info_fwd_pe"),
            "pe1_now_calc": (e.get("price_now") / ((e.get("eps1_now") or np.nan) * conv)
                             if e.get("price_now") and e.get("eps1_now") and e["eps1_now"] > 0 else np.nan),
        })
    df = pd.DataFrame(recs)
    cohort = df[(df.near_before >= -0.15) & (df.crash <= -0.15)].copy()

    # 소분류 등급
    subs = []
    for (grp, sub), part in cohort.groupby(["group", "sub"]):
        if len(part) < MIN_MEMBERS:
            continue
        rec = float(part.recovery.median())
        subs.append({"group": grp, "sub": sub, "n": len(part), "foreign": int((part.market != "한국").sum()),
                     "recovery": rec, "grade": grade_of(rec), "crash": float(part.crash.median()),
                     "rebound": float(part.rebound.median()), "g1": float(part.g1.median()),
                     "pe1": float(part.pe1.median()), "revision1": float(part.revision1.median()),
                     "rev_growth_q": float(part.rev_growth_q.median()), "mcap_b": float(part.mcap_b.median()),
                     "n_est": int(part.g1.notna().sum()),
                     "best": part.sort_values("recovery", ascending=False).name.head(3).tolist()})
    sg = pd.DataFrame(subs).sort_values("recovery", ascending=False)
    cohort["grade"] = cohort["sub"].map(dict(zip(sg["sub"], sg["grade"])))

    feats = ["g1", "g2", "pe1", "pe1_low", "revision1", "rev_growth_q", "mcap_b"]
    corr = {}
    for label, part in (("all", cohort), ("foreign", cohort[cohort.market != "한국"]), ("us", cohort[cohort.market == "미국"])):
        corr[label] = {}
        for f in feats:
            for y in ("crash", "rebound", "recovery"):
                d = part[[f, y]].dropna()
                if len(d) >= 15:
                    r, p = spearmanr(d[f], d[y])
                    corr[label][f"{f}~{y}"] = {"n": len(d), "rho": round(float(r), 2), "p": round(float(p), 3)}
    sector_corr = {}
    for f in ("g1", "pe1", "revision1", "rev_growth_q", "mcap_b", "crash"):
        d = sg[[f, "recovery"]].dropna()
        r, p = spearmanr(d[f], d.recovery)
        sector_corr[f] = {"n": len(d), "rho": round(float(r), 2), "p": round(float(p), 3)}
    top, bottom = cohort[cohort.grade.isin(["S", "A"])], cohort[cohort.grade.isin(["C", "D"])]
    by_grade = {}
    for f in feats + ["crash", "rebound", "recovery"]:
        a, b = top[f].dropna(), bottom[f].dropna()
        by_grade[f] = {"SA": round(float(a.median()), 3), "CD": round(float(b.median()), 3), "n": [len(a), len(b)],
                       "p": round(float(mannwhitneyu(a, b).pvalue), 3) if len(a) > 4 and len(b) > 4 else None}

    # 데스크 선행 PER(yfinance info forwardPE) 점검: 다음 회계연도 EPS로 직접 계산한 값과 비교
    chk = df.dropna(subset=["info_fwd_pe", "pe1_now_calc"]).copy()
    chk["ratio"] = chk.info_fwd_pe / chk.pe1_now_calc
    off = chk[(chk.ratio > 1.5) | (chk.ratio < 1 / 1.5)]

    result = {"population": {"universe": len(df), "cohort": len(cohort), "with_estimates": int(cohort.g1.notna().sum()),
                             "with_estimates_foreign": int(cohort[cohort.market != "한국"].g1.notna().sum())},
              "grades": json.loads(sg.round(3).to_json(orient="records", force_ascii=False)),
              "corr": corr, "sector_corr": sector_corr, "by_grade": by_grade,
              "pe_check": {"compared": len(chk), "off": off[["symbol", "name", "market", "info_fwd_pe", "pe1_now_calc"]]
                           .round(1).to_dict("records")},
              "cohort": json.loads(cohort.round(4).to_json(orient="records", force_ascii=False))}
    (OUT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    print(json.dumps(result["population"], ensure_ascii=False))
    print("\nGRADES (sub-sector median of cohort members)")
    for r in result["grades"]:
        print(f"  {r['grade']} {r['recovery']:+.2f} {r['sub'][:16]:<16} n={r['n']:<2} (해외 {r['foreign']}) crash {r['crash']:+.2f} "
              f"rebound {r['rebound']:+.2f} g1 {r['g1']} pe1 {r['pe1']} rev1 {r['revision1']} revQ {r['rev_growth_q']} "
              f"mcap {r['mcap_b']:.0f}B n_est {r['n_est']} | {', '.join(r['best'])}")
    print("\nSA vs CD (stock medians):")
    for f, v in by_grade.items():
        print(f"  {f:<13} SA {v['SA']}  CD {v['CD']}  n {v['n']}  p {v['p']}")
    print("\nsector-level Spearman with recovery:", sector_corr)
    for label in ("all", "foreign", "us"):
        print(f"\n[{label}] stock-level Spearman")
        for k, v in corr[label].items():
            print(f"  {k:<22} ρ={v['rho']:+.2f} p={v['p']:.3f} n={v['n']}")
    print(f"\nforwardPE check: {len(chk)} compared, {len(off)} off by >1.5x")
    print(off[["symbol", "name", "market", "info_fwd_pe", "pe1_now_calc"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
