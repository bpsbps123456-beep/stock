"""저평가 표시 기준 연구 (2026-10-10, 사용자 요청: AI 산업 지도에 '저평가' 표시).

요청: 병목처럼 투자 지침이 되는 '저평가' 표시. 미래 매출 하나로 모든 회사를 묶으면 늘 메모리만
고르게 되므로 한 기준으로 묶지 않고, 순환매를 전제한다.

정의(결과를 보기 전에 정함). 기간 L 동안
  R = 다음 회계연도(+1y) EPS 컨센서스 변화율       (이익 전망)
  P = 같은 기간 주가 수익률(배당 조정)
  V = (1+P)/(1+R) - 1                               (같은 이익 기준 선행 PER의 변화)
V < 0이면 전망이 오른 만큼 주가가 따라가지 않아 선행 PER이 내려간 것이다. 비교는 같은 분야(탐방
지도의 부서) 안에서 하고(동료 대비), 분야끼리는 AI 유니버스 전체와 비교한다(순환매).

점-시점: yfinance eps_trend는 같은 회계연도 추정치의 current·7·30·60·90일 전 값을 준다. 오늘
(2026-10-10) 받으면 날짜는 10/10·10/3·9/10·8/11·7/12다. 그래서 검증은 두 달뿐이다.
  W1 신호 8/11 (L=30: 7/12→8/11) → 결과 8/11→9/10
  W2 신호 9/10 (L=30: 8/11→9/10, L=60: 7/12→9/10) → 결과 9/10→10/9
W1으로 고르고 W2로 확인한다(AGENTS.md 연구 보고 규칙). 변형은 아래 VARIANTS에 미리 적은 것뿐이다.

    py data/derived/analysis/value_signal/analyze.py [--refresh]
"""

from __future__ import annotations

import json
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import ai_ath_gap  # noqa: E402
import watchlist  # noqa: E402
from desk import tour  # noqa: E402

HERE = Path(__file__).resolve().parent
EST = HERE / "estimates.json"
PX = HERE / "closes.csv"
TODAY = pd.Timestamp("2026-10-10")
DATES = {"current": TODAY, "7daysAgo": TODAY - pd.Timedelta(days=7), "30daysAgo": TODAY - pd.Timedelta(days=30),
         "60daysAgo": TODAY - pd.Timedelta(days=60), "90daysAgo": TODAY - pd.Timedelta(days=90)}
MIN_ANALYSTS = 3
VARIANTS = ("A", "B", "C", "P", "D", "G", "Z")   # 미리 정한 회사 단위 변형 전부(결과와 함께 모두 보고)
CYCLE_ZONES = {"hbm", "flash", "disk"}     # 메모리·저장: 이익이 정점일 때 PER이 가장 낮아 보이는 업종


# ----------------------------------------------------------------------------- data
def _estimates(symbol: str) -> tuple[str, dict]:
    import yfinance as yf

    t, out = yf.Ticker(symbol), {}
    try:
        et = t.eps_trend
        if et is not None and len(et):
            for per in ("0y", "+1y"):
                if per in et.index:
                    out[per] = {c: (None if pd.isna(et.loc[per, c]) else float(et.loc[per, c]))
                                for c in DATES if c in et.columns}
    except Exception as exc:  # noqa: BLE001
        out["err_trend"] = type(exc).__name__
    try:
        ee = t.earnings_estimate
        if ee is not None and len(ee) and "+1y" in ee.index:
            row = ee.loc["+1y"]
            out["analysts"] = None if pd.isna(row.get("numberOfAnalysts")) else float(row["numberOfAnalysts"])
            out["growth1"] = None if pd.isna(row.get("growth")) else float(row["growth"])
    except Exception as exc:  # noqa: BLE001
        out["err_est"] = type(exc).__name__
    return symbol, out


def load(refresh: bool = False):
    rows = ai_ath_gap.universe_rows()
    symbols = [s for _, _, s in rows]
    if refresh or not EST.exists():
        with ThreadPoolExecutor(max_workers=8) as pool:
            est = dict(pool.map(_estimates, symbols))
        EST.write_text(json.dumps(est, ensure_ascii=False, indent=1), encoding="utf-8")
    est = json.loads(EST.read_text(encoding="utf-8"))
    if refresh or not PX.exists():
        import yfinance as yf

        px = yf.download(symbols + ["SOXX", "QQQ"], start="2025-06-01", end="2026-10-11", auto_adjust=True,
                         progress=False, threads=True)["Close"]
        px.to_csv(PX)
    px = pd.read_csv(PX, index_col=0, parse_dates=True)
    return rows, est, px


def zones_of():
    m = tour.load_map()
    rows = watchlist.ai_tour_sectors(m, tour.universe_symbol)
    zone_syms = {r["key"].split(":", 1)[1]: r["symbols"] for r in rows}
    main = {}   # symbol -> its representative zone (tour_map 'main', else first)
    aliases = {a: t for t, c in m["companies"].items() for a in c.get("aliases", [])}
    seen = {}
    for st in m["stops"]:
        for z in st["zones"]:
            for mem in z.get("members", []):
                seen.setdefault(mem["t"], []).append((z["id"], bool(mem.get("main"))))
    for zid, syms in zone_syms.items():
        for s in syms:
            key = tour.universe_symbol(s)
            key = aliases.get(key, key)
            places = seen.get(key, [])
            pick = next((z for z, is_main in places if is_main), places[0][0] if places else zid)
            main[s] = pick
    names = {z["id"]: (st["no"], st["name"], z["name"]) for st in m["stops"] for z in st["zones"]}
    return zone_syms, main, names


def px_on(px: pd.DataFrame, sym: str, day: pd.Timestamp):
    s = px[sym].dropna() if sym in px else pd.Series(dtype=float)
    s = s[s.index <= day]
    return float(s.iloc[-1]) if len(s) else None


def high_52w(px: pd.DataFrame, sym: str, day: pd.Timestamp):
    s = px[sym].dropna() if sym in px else pd.Series(dtype=float)
    s = s[(s.index <= day) & (s.index > day - pd.Timedelta(days=365))]
    return float(s.max()) if len(s) else None


# ----------------------------------------------------------------------------- metrics
GIANT = json.loads((ROOT / "data" / "derived" / "desk" / "giant.json").read_text(encoding="utf-8"))["stocks"]


def snapshot(est, px, symbols, start: str, end: str):
    """Signal metrics over [start → end] (eps_trend column names)."""
    d0, d1 = DATES[start], DATES[end]
    out = {}
    for s in symbols:
        e = est.get(s, {})
        y1, y0 = e.get("+1y") or {}, e.get("0y") or {}
        e0, e1 = y1.get(start), y1.get(end)
        p0, p1 = px_on(px, s, d0), px_on(px, s, d1)
        row = {"analysts": e.get("analysts"), "growth1": e.get("growth1"), "p0": p0, "p1": p1}
        if p0 and p1:
            row["P"] = p1 / p0 - 1
            hi = high_52w(px, s, d1)
            row["dd"] = p1 / hi - 1 if hi else None
        if e0 and e1 and e0 > 0 and e1 > 0:
            row["R"] = e1 / e0 - 1
            if "P" in row:
                row["V"] = (1 + row["P"]) / (1 + row["R"]) - 1
            # PER level at the signal date: Yahoo PE now (currency-consistent) scaled by price and EPS ratios
            pe_now, p_now, e_now = (GIANT.get(s) or {}).get("fwd_pe"), px_on(px, s, TODAY), y1.get("current")
            if pe_now and p_now and e_now and e_now > 0 and p1:
                row["pe"] = pe_now * (p1 / p_now) * (e_now / e1)
            g0 = y0.get(end)
            if g0 and g0 > 0:
                row["g"] = e1 / g0 - 1
        elif e0 is not None and e1 is not None:
            row["loss"] = True       # 적자(이익이 0 이하): PER로 판단하지 않는다
        out[s] = row
    return out


def forward(px, symbols, start: str, end: str):
    d0, d1 = DATES[start], DATES[end]
    out = {}
    for s in symbols:
        a, b = px_on(px, s, d0), px_on(px, s, d1)
        out[s] = b / a - 1 if a and b else None
    return out


def usable(r):
    return r.get("V") is not None and (r.get("analysts") or 0) >= MIN_ANALYSTS


def flags(snap, zone_syms, main, variant):
    """Company flags for one pre-registered variant."""
    peer_med = {}
    for zid, syms in zone_syms.items():
        vals = [snap[s]["V"] for s in syms if s in snap and usable(snap[s])]
        peer_med[zid] = float(np.median(vals)) if len(vals) >= 3 else None
    uni_v = [r["V"] for r in snap.values() if usable(r)]
    cut = np.percentile(uni_v, 33.3) if uni_v else None
    for r in snap.values():
        if r.get("pe") and r.get("g") and r["g"] > 0:
            r["peg"] = r["pe"] / (100 * r["g"])
    pe_med, peg_med = {}, {}
    for zid, syms in zone_syms.items():
        pes = [snap[s]["pe"] for s in syms if s in snap and usable(snap[s]) and snap[s].get("pe")]
        pegs = [snap[s]["peg"] for s in syms if s in snap and usable(snap[s]) and snap[s].get("peg")]
        pe_med[zid] = float(np.median(pes)) if len(pes) >= 3 else None
        peg_med[zid] = float(np.median(pegs)) if len(pegs) >= 3 else None
    cheap_zones = set(zone_flags(snap, zone_syms))
    out = set()
    for s, r in snap.items():
        if not usable(r) or r["R"] <= 0:
            continue
        z = main.get(s)
        pm = peer_med.get(z)
        if variant == "A":     # 전망 상향 + 선행 PER이 AI 전체 하위 1/3로 싸짐
            ok = r["V"] <= cut
        elif variant == "B":   # A + 같은 분야 중앙값보다 더 싸짐
            ok = r["V"] <= cut and pm is not None and r["V"] < pm
        elif variant == "C":   # B + 52주 고점 대비 10% 이상 눌림
            ok = r["V"] <= cut and pm is not None and r["V"] < pm and (r.get("dd") or 0) <= -0.10
        elif variant == "P":   # 같은 분야 안에서만: 분야 중앙값보다 5%p 이상 더 싸짐(분야 3종목 이상)
            ok = pm is not None and r["V"] <= pm - 0.05
        elif variant == "D":   # 지금 선행 PER이 같은 분야 중앙값 이하(수준, 동료 대비)
            ok = r.get("pe") is not None and pe_med.get(z) is not None and r["pe"] <= pe_med[z]
        elif variant == "G":   # 성장 대비 PER(PER ÷ 다음 회계연도 이익 증가율%)이 같은 분야 중앙값 이하
            ok = r.get("peg") is not None and peg_med.get(z) is not None and r["peg"] <= peg_med[z]
        elif variant == "Z":   # 분야가 '저평가 분야'(아래 zone_flags)이고 자기 V도 0 미만
            ok = z in cheap_zones and r["V"] < 0
        else:
            raise ValueError(variant)
        if ok:
            out.add(s)
    return out


def zone_flags(snap, zone_syms):
    """'저평가 분야': 분야 중앙값 V가 분야들 중 하위 1/3이고 분야 중앙값 R > 0."""
    zt, _, _ = zone_table(snap, zone_syms)
    order = sorted(zt, key=lambda z: zt[z]["dV"])
    third = max(1, len(order) // 3)
    return [z for z in order[:third] if zt[z]["R"] > 0]


def zone_table(snap, zone_syms):
    rows = {}
    uni_v = float(np.median([r["V"] for r in snap.values() if usable(r)]))
    uni_r = float(np.median([r["R"] for r in snap.values() if usable(r)]))
    for zid, syms in zone_syms.items():
        rs = [snap[s] for s in syms if s in snap and usable(snap[s])]
        if len(rs) < 3:
            continue
        rows[zid] = {"n": len(rs), "V": float(np.median([r["V"] for r in rs])), "R": float(np.median([r["R"] for r in rs])),
                     "P": float(np.median([r["P"] for r in rs])), "dV": float(np.median([r["V"] for r in rs])) - uni_v}
    return rows, uni_v, uni_r


def evaluate(snap, fwd, zone_syms, main, label):
    from scipy.stats import mannwhitneyu, spearmanr

    alive = [s for s in snap if fwd.get(s) is not None]
    med = float(np.median([fwd[s] for s in alive]))
    ex = {s: fwd[s] - med for s in alive}
    res = {"label": label, "universe_fwd_median": med, "n": len(alive)}
    xs = [s for s in alive if usable(snap[s])]
    rho = spearmanr([snap[s]["V"] for s in xs], [ex[s] for s in xs])
    res["rho_V_all"] = {"n": len(xs), "rho": round(float(rho.statistic), 3), "p": round(float(rho.pvalue), 4)}
    # within-zone: V and excess both minus the zone median (main zone)
    dv, dx = [], []
    for s in xs:
        z = main.get(s)
        peers = [t for t in zone_syms.get(z, []) if t in xs]
        if len(peers) < 3:
            continue
        dv.append(snap[s]["V"] - np.median([snap[t]["V"] for t in peers]))
        dx.append(ex[s] - np.median([ex[t] for t in peers]))
    rho = spearmanr(dv, dx)
    res["rho_V_within_zone"] = {"n": len(dv), "rho": round(float(rho.statistic), 3), "p": round(float(rho.pvalue), 4)}
    rx = [s for s in xs]
    rho = spearmanr([snap[s]["R"] for s in rx], [ex[s] for s in rx])
    res["rho_R_all"] = {"n": len(rx), "rho": round(float(rho.statistic), 3), "p": round(float(rho.pvalue), 4)}
    for v in VARIANTS:
        f = [s for s in flags(snap, zone_syms, main, v) if s in ex]
        rest = [s for s in xs if s not in f]
        if not f:
            res[v] = {"n": 0}
            continue
        p = mannwhitneyu([ex[s] for s in f], [ex[s] for s in rest]).pvalue if len(f) >= 3 and rest else None
        res[v] = {"n": len(f), "mean_ex": round(float(np.mean([ex[s] for s in f])) * 100, 2),
                  "median_ex": round(float(np.median([ex[s] for s in f])) * 100, 2),
                  "hit": round(float(np.mean([ex[s] > 0 for s in f])) * 100),
                  "rest_mean_ex": round(float(np.mean([ex[s] for s in rest])) * 100, 2),
                  "p": None if p is None else round(float(p), 4)}
    zt, uni_v, _ = zone_table(snap, zone_syms)
    zex = {z: float(np.median([ex[s] for s in zone_syms[z] if s in ex])) for z in zt}
    order = sorted(zt, key=lambda z: zt[z]["dV"])
    third = max(1, len(order) // 3)
    cheap = [z for z in order[:third] if zt[z]["R"] > 0]
    dear = order[-third:]
    rho = spearmanr([zt[z]["dV"] for z in order], [zex[z] for z in order])
    res["zones"] = {"n": len(order), "rho_dV": round(float(rho.statistic), 3), "p": round(float(rho.pvalue), 4),
                    "cheap_third_rising": {"n": len(cheap), "median_ex": round(float(np.median([zex[z] for z in cheap])) * 100, 2) if cheap else None,
                                           "hit": round(float(np.mean([zex[z] > 0 for z in cheap])) * 100) if cheap else None,
                                           "zones": cheap},
                    "dear_third": {"n": len(dear), "median_ex": round(float(np.median([zex[z] for z in dear])) * 100, 2),
                                   "hit": round(float(np.mean([zex[z] > 0 for z in dear])) * 100)}}
    return res


def main(argv):
    rows, est, px = load("--refresh" in argv)
    symbols = [s for _, _, s in rows]
    zone_syms, mains, names = zones_of()
    cover = {"universe": len(symbols),
             "with_trend": sum(1 for s in symbols if (est.get(s, {}).get("+1y") or {}).get("current") is not None),
             "analysts>=3": sum(1 for s in symbols if (est.get(s, {}).get("analysts") or 0) >= MIN_ANALYSTS)}
    out = {"coverage": cover, "windows": []}
    w1 = snapshot(est, px, symbols, "90daysAgo", "60daysAgo")
    out["windows"].append(evaluate(w1, forward(px, symbols, "60daysAgo", "30daysAgo"), zone_syms, mains, "W1 신호 8/11 (L30) → 9/10"))
    out["windows"].append(evaluate(w1, forward(px, symbols, "60daysAgo", "current"), zone_syms, mains, "W1 신호 8/11 (L30) → 10/9 (2개월)"))
    w2 = snapshot(est, px, symbols, "60daysAgo", "30daysAgo")
    out["windows"].append(evaluate(w2, forward(px, symbols, "30daysAgo", "current"), zone_syms, mains, "W2 신호 9/10 (L30) → 10/9"))
    w2b = snapshot(est, px, symbols, "90daysAgo", "30daysAgo")
    out["windows"].append(evaluate(w2b, forward(px, symbols, "30daysAgo", "current"), zone_syms, mains, "W2 신호 9/10 (L60) → 10/9"))
    now = snapshot(est, px, symbols, "90daysAgo", "current")
    zt, uni_v, uni_r = zone_table(now, zone_syms)
    out["now"] = {"uni_V": uni_v, "uni_R": uni_r,
                  "zones": {z: {**v, "name": names.get(z)} for z, v in sorted(zt.items(), key=lambda kv: kv[1]["dV"])}}
    (HERE / "result.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out["coverage"], ensure_ascii=False))
    for w in out["windows"]:
        print("\n==", w["label"], "| universe fwd median", round(w["universe_fwd_median"] * 100, 2), "%")
        print("  rho V(all)", w["rho_V_all"], "| V within zone", w["rho_V_within_zone"], "| R(all)", w["rho_R_all"])
        for v in VARIANTS:
            print("  variant", v, w[v])
        print("  zones", {k: w["zones"][k] for k in ("n", "rho_dV", "p")})
        print("    cheap third w/ rising est", w["zones"]["cheap_third_rising"])
        print("    dear third", w["zones"]["dear_third"])
    print("\n== now (7/12→10/10): uni V", round(uni_v * 100, 1), "% uni R", round(uni_r * 100, 1), "%")
    for z, v in out["now"]["zones"].items():
        print(f"  {z:16s} n={v['n']:2d} V={v['V']*100:6.1f}% R={v['R']*100:6.1f}% P={v['P']*100:6.1f}% dV={v['dV']*100:6.1f} {v['name']}")


if __name__ == "__main__":
    main(sys.argv[1:])


# ----------------------------------------------------------------------------- production rule check
def production_check(refresh: bool = False):
    """desk/value.py의 judge()를 지도 전체 상장 회사(유니버스 + 지도에만 있는 32곳)에 그대로 돌려
    두 달 검증을 다시 잰다. 화면의 CHECK 숫자는 이 결과다. 애널리스트 수는 지금 값을 쓴다."""
    from desk import value

    m = tour.load_map()
    cands = value.map_symbols(m)
    allsyms = sorted({s for c in cands.values() for s in c})
    pxf = HERE / "closes_map.csv"
    if refresh or not pxf.exists():
        import yfinance as yf

        yf.download(allsyms + ["SOXX"], start="2025-06-01", end="2026-10-11", auto_adjust=True,
                    progress=False, threads=True)["Close"].to_csv(pxf)
    px = pd.read_csv(pxf, index_col=0, parse_dates=True)
    sym_of = {t: next((s for s in c if s in px and px[s].notna().any()), None) for t, c in cands.items()}
    sym_of = {t: s for t, s in sym_of.items() if s}
    estf = HERE / "estimates_map.json"
    if refresh or not estf.exists():
        with ThreadPoolExecutor(max_workers=8) as pool:
            est = dict(pool.map(_estimates, sorted(set(sym_of.values()))))
        estf.write_text(json.dumps(est, ensure_ascii=False, indent=1), encoding="utf-8")
    est = json.loads(estf.read_text(encoding="utf-8"))
    zones, main = value.zone_index(m)
    out = {"population": {"listed_placed": len(cands), "resolved": len(sym_of)}, "windows": []}
    for label, start, end, fend in (("W1 신호 8/11 → 9/10", "90daysAgo", "60daysAgo", "30daysAgo"),
                                    ("W2 신호 9/10 → 10/9", "60daysAgo", "30daysAgo", "current"),
                                    ("W1 신호 8/11 → 10/9 (2개월)", "90daysAgo", "60daysAgo", "current")):
        rows = {}
        for t, s in sym_of.items():
            y1 = (est.get(s) or {}).get("+1y") or {}
            e = {"+1y": {"current": y1.get(end), "30daysAgo": y1.get(start)}, "analysts": (est.get(s) or {}).get("analysts")}
            rows[t] = value.measure(e, px_on(px, s, DATES[end]), px_on(px, s, DATES[start]))
        res = value.judge(rows, zones, main)
        fwd = {t: (lambda a, b: b / a - 1 if a and b else None)(px_on(px, s, DATES[end]), px_on(px, s, DATES[fend]))
               for t, s in sym_of.items()}
        alive = {t: f for t, f in fwd.items() if f is not None}
        med = float(np.median(list(alive.values())))
        ex = {t: f - med for t, f in alive.items()}
        zex = {z: float(np.median([ex[t] for t in dict.fromkeys(zones[z]) if t in ex]))
               for z in res["zones"] if any(t in ex for t in zones[z])}
        zr = res["zones"]
        order = sorted(zr, key=lambda z: zr[z]["dV"])
        third = max(1, int(len(order) / 3))
        flagged = [z for z in order if zr[z]["flag"]]
        dear = order[-third:]
        comp = [t for t, c in res["companies"].items() if c["flag"] and t in ex]
        rest = [t for t, c in res["companies"].items() if not c["flag"] and t in ex and value.usable(rows[t])]
        w = {"label": label, "universe_median": round(med * 100, 2), "zones_judged": len(order),
             "zone_flag": {"n": len(flagged), "zones": flagged,
                           "median_ex": round(float(np.median([zex[z] for z in flagged])) * 100, 2) if flagged else None,
                           "hit": round(float(np.mean([zex[z] > 0 for z in flagged])) * 100) if flagged else None},
             "zone_dear": {"n": len(dear), "median_ex": round(float(np.median([zex[z] for z in dear])) * 100, 2),
                           "hit": round(float(np.mean([zex[z] > 0 for z in dear])) * 100)},
             "company_flag": {"n": len(comp),
                              "mean_ex": round(float(np.mean([ex[t] for t in comp])) * 100, 2) if comp else None,
                              "median_ex": round(float(np.median([ex[t] for t in comp])) * 100, 2) if comp else None,
                              "hit": round(float(np.mean([ex[t] > 0 for t in comp])) * 100) if comp else None,
                              "rest_mean_ex": round(float(np.mean([ex[t] for t in rest])) * 100, 2)}}
        out["windows"].append(w)
    (HERE / "production_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__" and "--production" in sys.argv:
    print(json.dumps(production_check("--refresh" in sys.argv), ensure_ascii=False, indent=1))
