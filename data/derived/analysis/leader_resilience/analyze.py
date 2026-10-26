"""7월 급락 뒤 '회복탄력성'과 섹터 대장 (2026-10-05, Path-Q 연구).

사용자 가설: 회복이 강한 종목은 각 섹터의 대장이다(메모리·냉각·데이터센터·광통신의 대장 등).
해외(한국 외 상장) 종목을 중심으로 본다. 결과는 같은 폴더 result.json, 사람이 읽는 정리는 report.md.

측정(종목 자신의 경로, ../recovery_factors/analyze.py와 같은 정의)
  급락 코호트: 5/1~7/17 종가 고점이 그때까지 역대 고가의 -15% 안, 그 뒤 8/21까지 -15% 넘게 하락
  회복률 = (현재 - 저점) / (급락 전 고점 - 저점). 1이면 낙폭을 모두 되찾았다.
  급락 전 대비 = 현재 / 급락 전 고점 - 1, 전고점 대비 = 거인의 어깨 캐시의 gap_pct
대장 정의(결과를 보기 전에 고정)
  D1 사업 대장: AI 탐방 부서의 ★(desk/tour_map.json lead, 출처 있는 사업 1등)
  D2 시총 대장: 유니버스 소분류(3종목 이상) 안 달러 시총 1위
  D3 주가 대장: 유니버스 소분류(3종목 이상) 안 연초→급락 전 고점 상승 배수 1위(흔히 말하는 대장주)
  D4 병목 대장: D1 중 탐방이 공급 병목(bottleneck)을 적은 부서의 ★
상관은 인과가 아니고, 대장 표시는 오늘(10/5) 붙인 것이라 사후 판단이 섞였을 수 있다.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu, spearmanr

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import ai_ath_gap  # noqa: E402
from desk import giant, tour  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("rf", OUT.parent / "recovery_factors" / "analyze.py")
rf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rf)


def tour_symbols(m: dict, symbols) -> dict[str, str]:
    """탐방 지도 키 -> 유니버스 심볼 (SKHY -> 000660.KS, GOOG -> GOOGL 같은 별칭 포함)."""
    alias = {a: t for t, c in m["companies"].items() for a in c.get("aliases", [])}
    out = {}
    for sym in symbols:
        k = tour.universe_symbol(sym)
        out[alias.get(k, k)] = sym
    return out


def mark_leaders(df: pd.DataFrame, m: dict) -> pd.DataFrame:
    key_to_sym = tour_symbols(m, df.symbol)
    d1, d4, zones_of = set(), set(), {}
    for _, z in tour.zones(m):
        lead = key_to_sym.get(z.get("lead"))
        if lead:
            d1.add(lead)
            if z.get("bottleneck"):
                d4.add(lead)
        for mem in z["members"]:
            s = key_to_sym.get(mem["t"])
            if s:
                zones_of.setdefault(s, []).append(z["id"])
    df["d1"] = df.symbol.isin(d1)
    df["d4"] = df.symbol.isin(d4)
    df["zones"] = df.symbol.map(lambda s: zones_of.get(s, []))
    big_sub = df.groupby("sub").symbol.transform("size") >= 3
    top_cap = set(df.loc[df.groupby("sub").mcap_b.idxmax().dropna()].symbol)
    top_run = set(df.loc[df.groupby("sub").runup.idxmax().dropna()].symbol)
    df["d2"] = np.where(big_sub, df.symbol.isin(top_cap).astype(float), np.nan)
    df["d3"] = np.where(big_sub, df.symbol.isin(top_run).astype(float), np.nan)
    return df


def compare(c: pd.DataFrame, flag: str) -> dict:
    d = c.dropna(subset=[flag])
    yes, no = d[d[flag].astype(bool)], d[~d[flag].astype(bool)]
    out = {"n_leader": len(yes), "n_other": len(no)}
    for k in ("recovery", "vs_peak", "crash", "rebound", "gap_now", "mcap_b"):
        out[k] = [round(float(yes[k].median()), 3), round(float(no[k].median()), 3)]
    if len(yes) >= 3 and len(no) >= 3:
        out["p_recovery"] = round(float(mannwhitneyu(yes.recovery, no.recovery).pvalue), 3)
        out["p_vs_peak"] = round(float(mannwhitneyu(yes.vs_peak, no.vs_peak).pvalue), 3)
    out["leaders"] = yes.sort_values("recovery", ascending=False)[["name", "recovery", "vs_peak", "gap_now"]].round(3).values.tolist()
    return out


def within_zone(fc: pd.DataFrame, m: dict) -> dict:
    """같은 부서 안에서 ★의 회복률과 나머지 해외 코호트 종목의 중앙값을 비교한다."""
    key_to_sym = tour_symbols(m, fc.symbol)
    pairs = []
    for _, z in tour.zones(m):
        lead = key_to_sym.get(z.get("lead"))
        members = fc[fc.zones.map(lambda zs: z["id"] in zs)]
        if lead not in set(members.symbol):
            continue
        row = members[members.symbol == lead].iloc[0]
        peers = members[members.symbol != lead]
        if len(peers):
            pairs.append({"zone": z["name"], "lead": row["name"], "lead_recovery": round(float(row.recovery), 3),
                          "peer_median": round(float(peers.recovery.median()), 3), "peers": len(peers),
                          "peer_names": peers.name.tolist(), "bottleneck": bool(z.get("bottleneck"))})
    wins = sum(p["lead_recovery"] > p["peer_median"] for p in pairs)
    return {"pairs": pairs, "wins": wins, "n": len(pairs),
            "sign_p": round(float(binomtest(wins, len(pairs)).pvalue), 3) if pairs else None}


def main():
    rows = ai_ath_gap.universe_rows()
    syms = [s for _, _, s in rows]
    sub_of = {s: (g, sub) for g, sub, s in rows}
    cache = json.loads(giant.CACHE_PATH.read_text(encoding="utf-8"))["stocks"]
    fx = rf.fx_rates()
    with ThreadPoolExecutor(max_workers=8) as pool:
        bars = {s: b for s, b in pool.map(lambda s: ai_ath_gap.fetch(s, None), syms) if b is not None}
        infos = dict(pool.map(rf.info, syms))
    recs = []
    for s in syms:
        if s not in bars or s not in cache:
            continue
        inf = infos.get(s) or {}
        pm = rf.path_measures(bars[s], fx.get(inf.get("currency") or "USD", np.nan))
        if pm is None:
            continue
        g = cache[s]
        sh = g.get("short") or {}
        recs.append({"symbol": s, "name": g["name"], "group": sub_of[s][0], "sub": sub_of[s][1], "market": g["market"],
                     **pm, "gap_now": g["gap_pct"] / 100, "mcap_b": (g.get("mcap_usd") or np.nan) / 1e9,
                     "rev_growth": g.get("rev_growth"), "op_margin": g.get("op_margin"), "fwd_pe": g.get("fwd_pe"),
                     "short_float": sh.get("float_pct"), "short_cover": sh.get("cover_pct"),
                     "inst_own": inf.get("heldPercentInstitutions"), "analysts": inf.get("numberOfAnalystOpinions")})
    m = tour.load_map()
    df = mark_leaders(pd.DataFrame(recs), m)
    df["foreign"] = df.market != "한국"
    cohort = df[(df.near_before >= -0.15) & (df.crash <= -0.15)].copy()
    fc = cohort[cohort.foreign].copy()
    us = cohort[cohort.market == "미국"].copy()

    result = {"population": {"universe": len(df), "cohort": len(cohort), "foreign_cohort": len(fc), "us_cohort": len(us),
                             "foreign_not_near_before": int(((df.near_before < -0.15) & df.foreign).sum()),
                             "foreign_mild_crash": int(((df.near_before >= -0.15) & (df.crash > -0.15) & df.foreign).sum())}}
    result["compare_foreign"] = {f: compare(fc, f) for f in ("d1", "d2", "d3", "d4")}
    result["compare_us"] = {f: compare(us, f) for f in ("d1", "d2", "d3", "d4")}
    lead = fc[fc.d1]
    result["d1_bottleneck_split"] = {
        "bottleneck": [int(lead.d4.sum()), round(float(lead[lead.d4].recovery.median()), 3)],
        "other": [int((~lead.d4).sum()), round(float(lead[~lead.d4].recovery.median()), 3)],
        "p": (round(float(mannwhitneyu(lead[lead.d4].recovery, lead[~lead.d4].recovery).pvalue), 3)
              if lead.d4.sum() >= 3 and (~lead.d4).sum() >= 3 else None)}
    result["within_zone"] = within_zone(fc, m)

    factors = ["mcap_b", "crash", "runup", "rev_growth", "op_margin", "fwd_pe", "short_float", "short_cover",
               "impact", "inst_own", "analysts", "d1", "d2", "d3", "d4"]
    corr = {}
    for label, part in (("foreign", fc), ("us", us)):
        corr[label] = {}
        for f in factors:
            d = part[[f, "recovery", "vs_peak"]].dropna()
            if len(d) >= 12 and d[f].nunique() > 1:
                r1, p1 = spearmanr(d[f].astype(float), d.recovery)
                r2, p2 = spearmanr(d[f].astype(float), d.vs_peak)
                corr[label][f] = {"n": len(d), "rho_recovery": round(float(r1), 2), "p_recovery": round(float(p1), 3),
                                  "rho_vs_peak": round(float(r2), 2), "p_vs_peak": round(float(p2), 3)}
    result["corr"] = corr

    lo, hi = fc.recovery.quantile([0.2, 0.8]).values
    prof = {}
    for label, part in (("top20", fc[fc.recovery >= hi]), ("bottom20", fc[fc.recovery <= lo]), ("all", fc)):
        prof[label] = {"n": int(len(part)), **{k: round(float(part[k].median()), 3) for k in
                       ("recovery", "crash", "rebound", "runup", "mcap_b", "op_margin", "rev_growth", "fwd_pe", "short_float")},
                       "d1_share": round(float(part.d1.mean()), 2), "d4_share": round(float(part.d4.mean()), 2),
                       "names": part.sort_values("recovery", ascending=False).name.tolist()}
    result["profile"] = prof

    fc["log_mcap"] = np.log10(fc.mcap_b)
    fc["log_runup"] = np.log(fc.runup)
    fc["d1f"], fc["d4f"] = fc.d1.astype(float), fc.d4.astype(float)
    base = ["log_mcap", "crash", "log_runup", "op_margin"]
    result["ols"] = {"recovery~d1": rf.ols_hc1(fc, "recovery", ["d1f"]),
                     "recovery~d1+controls": rf.ols_hc1(fc, "recovery", ["d1f"] + base),
                     "recovery~d4+controls": rf.ols_hc1(fc, "recovery", ["d4f"] + base),
                     "recovery~controls+short": rf.ols_hc1(fc, "recovery", base + ["short_float"])}
    keep = ["symbol", "name", "group", "sub", "market", "near_before", "crash", "rebound", "recovery", "vs_peak", "gap_now",
            "runup", "mcap_b", "impact", "op_margin", "rev_growth", "fwd_pe", "short_float", "short_cover",
            "d1", "d2", "d3", "d4", "zones"]
    result["cohort_foreign"] = fc[keep].round(4).to_dict("records")
    (OUT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    print(json.dumps(result["population"], ensure_ascii=False))
    for label in ("compare_foreign", "compare_us"):
        print(f"\n[{label}] median leader vs other: recovery / vs_peak / crash / rebound / gap / mcap$B, Mann-Whitney p")
        for f, r in result[label].items():
            print(f"  {f}: n {r['n_leader']}/{r['n_other']}  rec {r['recovery']}  vs_peak {r['vs_peak']}  crash {r['crash']}  "
                  f"rebound {r['rebound']}  gap {r['gap_now']}  mcap {r['mcap_b']}  p_rec {r.get('p_recovery')}  p_vs {r.get('p_vs_peak')}")
    print("\nD1 split by bottleneck:", result["d1_bottleneck_split"])
    wz = result["within_zone"]
    print(f"\nwithin zone: lead beats foreign peers' median in {wz['wins']}/{wz['n']} zones (sign p={wz['sign_p']})")
    for p in sorted(wz["pairs"], key=lambda p: p["lead_recovery"] - p["peer_median"], reverse=True):
        print(f"  {'B' if p['bottleneck'] else ' '} {p['zone'][:16]:<16} {p['lead'][:12]:<12} lead {p['lead_recovery']:+.2f}"
              f" vs peers {p['peer_median']:+.2f} (n={p['peers']}: {', '.join(p['peer_names'][:5])})")
    for label in ("foreign", "us"):
        print(f"\n[{label}] Spearman with recovery / vs_peak")
        for f, r in sorted(corr[label].items(), key=lambda kv: -abs(kv[1]["rho_recovery"])):
            print(f"  {f:<12} n={r['n']:<3} rec ρ={r['rho_recovery']:+.2f} (p={r['p_recovery']:.3f})"
                  f"  vs_peak ρ={r['rho_vs_peak']:+.2f} (p={r['p_vs_peak']:.3f})")
    print("\nprofile:")
    for k, v in prof.items():
        print(" ", k, {kk: vv for kk, vv in v.items() if kk != "names"})
        print("    ", ", ".join(v["names"][:18]))
    for name, mdl in result["ols"].items():
        print(f"\n[OLS {name}] n={mdl['n']} R²={mdl['r2']}")
        for f, c in mdl["coef"].items():
            print(f"  {f:<12} {c['beta']:+7.2f}  t={c['t']:+.2f}")


if __name__ == "__main__":
    main()
