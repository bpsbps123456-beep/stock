"""Adversarial audit of deskx_timing.py (2026-10-10, claude). Research scratch: not committed.

Independent re-implementation from the frozen CSVs (../cache/deskx_prices) and the fund PDFs
(gbrc.etf.trades.load_fund_snapshots). Only the desk's holding-row classification is reused
(desk.strategies.us_weights / sector_of); ranking, buffer, near-high filter, cap fill, open-to-open
path and metrics are written here from the §3 spec, then compared with deskx.Engine.

Sections
  A  path equality: independent sim vs Engine for base/daily/lb60/nh80/nh90 (+ base vs desk run_model)
  B  two daily-variant decisions recomputed by hand (rule fixed before looking: first mid-week
     membership change in the front half and in the back half), one nh90 exclusion
  C  raw facts for the user's question (MU / PLTR in TIME vs KoAct, MU's would-be rank)
  D  robustness: monthly diffs, name attribution, membership-only daily refresh (diagnostic),
     every-m-session rebalancing over all phases, lookback / near-high neighbours, spread cost
Run: PYTHONIOENCODING=utf-8 py data/derived/analysis/rsi_breakout/code/verify_deskx_timing.py
"""
import os, sys, json
from datetime import date
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, *[".."] * 5))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from gbrc.etf.trades import load_fund_snapshots          # noqa: E402
from gbrc.etf.market import yahoo_ticker                 # noqa: E402
from desk.strategies import us_weights, sector_of        # noqa: E402  (row classification only)
from desk import config                                  # noqa: E402

FROZEN = os.path.join(HERE, "..", "cache", "deskx_prices")
OUT = os.path.join(HERE, "..", "results", "deskx", "verify_timing.json")
START = "2026-01-02"
PERIODS = (("앞", "2026-01-02", "2026-05-29"), ("뒤", "2026-06-01", "2026-10-09"),
           ("9/1~", "2026-09-01", "2026-10-09"), ("전체", "2026-01-02", "2026-10-09"))
MONTHS = [f"2026-{m:02d}" for m in range(1, 11)]
SEMIS = set(config.SEMIS_SECTORS)
UNK = "미분류"


# ------------------------------------------------------------------ raw inputs
def _load(name):
    df = pd.read_csv(os.path.join(FROZEN, name), index_col=0, parse_dates=True)
    df.index = df.index.strftime("%Y-%m-%d")
    return df


ADJ, OPN, CLS = _load("prices_adj_close.csv"), _load("prices_open.csv"), _load("prices_close.csv")
DAYS = list(ADJ["QQQ"].dropna().index)
ADJ, OPN, CLS = (f.reindex(DAYS) for f in (ADJ, OPN, CLS))
TSNAP = load_fund_snapshots("456600", "9999-12-31")
KSNAP = load_fund_snapshots("2ETFQ1", "9999-12-31")
TPDF = sorted(TSNAP)
KPDF = sorted(KSNAP)
TW = {d: us_weights(s) for d, s in TSNAP.items()}
KW = {d: us_weights(s) for d, s in KSNAP.items()}
KEYS = sorted({k for w in TW.values() for k in w} | {k for w in KW.values() for k in w})
TICK = {k: yahoo_ticker(k) for k in KEYS}
SECT = {k: sector_of(k, TICK[k]) for k in KEYS}
A = {k: ADJ[TICK[k]].to_numpy(float) for k in KEYS if TICK[k] in ADJ.columns}
O = {k: OPN[TICK[k]].to_numpy(float) for k in KEYS if TICK[k] in OPN.columns}
C = {k: CLS[TICK[k]].to_numpy(float) for k in KEYS if TICK[k] in CLS.columns}
K0 = DAYS.index([d for d in DAYS if d >= START][0])


def pdf_before(pdfs, day):
    prior = [d for d in pdfs if d < day]          # strictly before the US session date
    return prior[-1] if prior else None


def isoweek(day):
    return date.fromisoformat(day).isocalendar()[:2]


# ------------------------------------------------------------------ rule pieces (from §3 spec)
def caps(raw, stock=0.25, sect=0.60, semis=0.60):
    """Proportional progressive fill: every name grows at rate raw[k] until its stock cap, its sector
    cap or the semis(+unclassified) budget binds; leftover stays cash."""
    names = [k for k, v in raw.items() if v > 0]
    sec = {k: SECT.get(k, UNK) for k in names}
    semi = {k for k in names if sec[k] in SEMIS or sec[k] == UNK}
    a = dict.fromkeys(names, 0.0)
    goal = sum(raw[k] for k in names)
    for _ in range(100):
        rem = goal - sum(a.values())
        if rem <= 1e-12:
            break
        gsum = {}
        for k in names:
            gsum[sec[k]] = gsum.get(sec[k], 0.0) + a[k]
        ssum = sum(a[k] for k in semi)
        act = [k for k in names if stock - a[k] > 1e-12 and sect - gsum[sec[k]] > 1e-12
               and not (k in semi and semis - ssum <= 1e-12)]
        if not act:
            break
        t = rem / sum(raw[k] for k in act)
        for k in act:
            t = min(t, (stock - a[k]) / raw[k])
        for g in {sec[k] for k in act}:
            t = min(t, (sect - gsum[g]) / sum(raw[k] for k in act if sec[k] == g))
        sr = sum(raw[k] for k in act if k in semi)
        if sr > 0:
            t = min(t, (semis - ssum) / sr)
        t = max(t, 0.0)
        for k in act:
            a[k] += t * raw[k]
    return {k: v for k, v in a.items() if v > 1e-12}


def rank(k, day, L=120, nh=None, extra=()):
    pdf = pdf_before(TPDF, day)
    if pdf is None or k - 1 - L < 0:
        return pdf, []
    w = dict(TW[pdf])
    for x in extra:                                  # diagnostic only: inject a name (MU) with tiny weight
        w.setdefault(x, 1e-9)
    out = []
    for key in sorted(w):
        if w[key] <= 0 or key not in A:
            continue
        p1, p0 = A[key][k - 1], A[key][k - 1 - L]
        if not (np.isfinite(p1) and np.isfinite(p0) and p0 > 0):
            continue
        info = {"key": key, "t": TICK[key], "mom": p1 / p0 - 1, "w": w[key]}
        c = C[key]
        win = c[max(0, k - 252):k]
        info["hi"] = float(np.nanmax(win)) if np.isfinite(win).any() else np.nan
        info["nh"] = c[k - 1] / info["hi"] if np.isfinite(info["hi"]) and info["hi"] > 0 else np.nan
        if nh is not None and not (np.isfinite(info["nh"]) and info["nh"] >= nh):
            continue
        out.append(info)
    out.sort(key=lambda r: -r["mom"])
    return pdf, out


def decide(k, day, held, L=120, nh=None, n=7, buf=5):
    pdf, rk = rank(k, day, L, nh)
    if not rk:
        return pdf, rk, [], {}
    order = [r["key"] for r in rk]
    keep = [x for x in order[:n + buf] if x in held][:n]
    for x in order:
        if len(keep) >= n:
            break
        if x not in keep:
            keep.append(x)
    wm = {r["key"]: r["w"] for r in rk}
    tot = sum(wm[x] for x in keep)
    return pdf, rk, keep, caps({x: wm[x] / tot for x in keep})


def orets(keys, k):
    out = {}
    for key in keys:
        o = O.get(key)
        ok = o is not None and np.isfinite(o[k]) and np.isfinite(o[k - 1]) and o[k - 1] > 0
        out[key] = o[k] / o[k - 1] - 1 if ok else 0.0
    return out


def sim(mode="weekly", L=120, nh=None, every=None, phase=0, cost=0.0):
    """mode: weekly | daily | every (every m sessions from START, given phase) | weekly_swap
    (weekly full reset + mid-week trade only when the chosen 7 changes; then full reset to target)."""
    w, held, started = {}, set(), False
    val, rows, turn = [1.0], [], []
    for k in range(K0, len(DAYS)):
        day = DAYS[k]
        if k > K0:
            r = orets(w, k)
            gross = sum(wt * r[x] for x, wt in w.items())
            tot = sum(wt * (1 + r[x]) for x, wt in w.items()) + 1 - sum(w.values())
            w = {x: wt * (1 + r[x]) / tot for x, wt in w.items()}
            val.append(val[-1] * (1 + gross))
        weekly = (not started) or isoweek(day) != isoweek(DAYS[k - 1])
        if mode == "weekly":
            reb = weekly
        elif mode == "daily":
            reb = True
        elif mode == "every":
            reb = (not started) or ((k - K0 - phase) % every == 0)
        elif mode == "weekly_swap":
            reb = True
        else:
            raise ValueError(mode)
        if reb:
            pdf, rk, keep, tgt = decide(k, day, held, L, nh)
            if rk and (mode != "weekly_swap" or weekly or set(keep) != held):
                started = True
                tv = sum(abs(tgt.get(x, 0) - w.get(x, 0)) for x in set(tgt) | set(w))
                turn.append((day, tv))
                if cost:
                    val[-1] *= 1 - cost * tv
                w, held = dict(tgt), set(keep)
        rows.append({"day": day, "w": dict(w)})
    return pd.Series(val, index=DAYS[K0:]), rows, turn


def metrics(v):
    r = (v.shift(-1) / v - 1).dropna()
    out = {}
    for lab, a, b in PERIODS:
        x = r[(r.index >= a) & (r.index <= b)]
        cur = (1 + x).cumprod()
        out[lab] = round(float(cur.iloc[-1] - 1), 4)
        out[lab + "_mdd"] = round(float((cur / cur.cummax() - 1).min()), 4)
    out["월"] = {m[5:]: round(float((1 + r[r.index.str.startswith(m)]).prod() - 1), 4) for m in MONTHS}
    return out


def attribution(v, rows, a, b):
    """Dollar P&L by name over [a,b] in units of the period's starting value (additive, exact)."""
    pnl = {}
    days = list(v.index)
    base_i = next(i for i, d in enumerate(days) if d >= a)
    v0 = v.iloc[base_i]
    for i in range(base_i, len(days) - 1):
        if days[i] > b:
            break
        k = K0 + i + 1
        r = orets(rows[i]["w"], k)
        for x, wt in rows[i]["w"].items():
            pnl[TICK[x]] = pnl.get(TICK[x], 0.0) + v.iloc[i] * wt * r[x] / v0
    return pnl


# ------------------------------------------------------------------ main
def main():
    res = {}
    # A. equality with Engine and run_model
    from deskx import Engine, Params
    from desk.strategies import run_model
    E = Engine()
    VAR = {"base": ({}, Params()), "daily": ({"mode": "daily"}, Params(rebalance="daily")),
           "lb60": ({"L": 60}, Params(lookback=60)), "nh80": ({"nh": 0.80}, Params(min_near_high=0.80)),
           "nh90": ({"nh": 0.90}, Params(min_near_high=0.90))}
    mine, eng = {}, {}
    print("A. 독립 재계산 vs Engine (가치 경로 최대 차, 마지막 행 비중 최대 차)")
    for name, (kw, p) in VAR.items():
        v, rows, turn = sim(**kw)
        ev, erows = E.run(p)
        dv = float(np.max(np.abs(v.to_numpy() - ev.to_numpy())))
        dw = max(abs(rows[i]["w"].get(x, 0) - erows[i]["w"].get(x, 0))
                 for i in range(len(rows)) for x in set(rows[i]["w"]) | set(erows[i]["w"]))
        mine[name] = (v, rows, turn)
        eng[name] = (ev, erows)
        m = metrics(v)
        res.setdefault("A", {})[name] = {"path_maxdiff": dv, "weights_maxdiff_all_days": dw, **m,
                                         "trades": len(turn), "sum_abs_turnover": round(sum(t for _, t in turn), 3)}
        print(f"  {name:6} path {dv:.2e} w {dw:.2e} | 앞 {m['앞']:+.4f} 뒤 {m['뒤']:+.4f} 9/1~ {m['9/1~']:+.4f} "
              f"전체 {m['전체']:+.4f} MDD {m['전체_mdd']:+.4f} | 월 {m['월']}")
    ref = run_model(E.book, start=START)
    d_desk = float(np.max(np.abs(mine["base"][0].to_numpy() - np.asarray(ref.value))))
    res["A"]["base_vs_desk_run_model"] = d_desk
    print("  base(독립) vs desk.run_model 최대 차", d_desk)

    # B. hand-checked decisions
    print("\nB. 결정 재계산")
    vD, rD, _ = mine["daily"]
    picks = []
    for half_a, half_b in (("2026-01-02", "2026-05-29"), ("2026-06-01", "2026-10-09")):
        for i in range(1, len(rD)):
            d = rD[i]["day"]
            if half_a <= d <= half_b and isoweek(d) == isoweek(rD[i - 1]["day"]) and set(rD[i]["w"]) != set(rD[i - 1]["w"]):
                picks.append(i)
                break
    res["B"] = []
    for i in picks:
        k = K0 + i
        day = DAYS[k]
        prev = set(rD[i - 1]["w"])
        pdf, rk, keep, tgt = decide(k, day, prev)
        erow = eng["daily"][1][i]["w"]
        diff = max(abs(tgt.get(x, 0) - erow.get(x, 0)) for x in set(tgt) | set(erow))
        top = [(j + 1, r["t"], round(r["mom"], 4), round(r["w"], 4), SECT[r["key"]]) for j, r in enumerate(rk[:14])]
        out = {"session": day, "pdf_used": pdf, "close_used": DAYS[k - 1], "mom_base_close": DAYS[k - 121],
               "prev_held": sorted(TICK[x] for x in prev), "new_held": sorted(TICK[x] for x in keep),
               "in": sorted(TICK[x] for x in set(keep) - prev), "out": sorted(TICK[x] for x in prev - set(keep)),
               "target": {TICK[x]: round(t, 4) for x, t in sorted(tgt.items(), key=lambda kv: -kv[1])},
               "engine_vs_mine_maxdiff": diff, "top14": top}
        # why did the leaver leave? its rank today
        out["leaver_rank"] = {TICK[x]: next((j + 1 for j, r in enumerate(rk) if r["key"] == x), None) for x in prev - set(keep)}
        res["B"].append(out)
        print(json.dumps(out, ensure_ascii=False))
    # nh90: first rebalance where a base-held name is dropped by the filter
    vB, rB, _ = mine["base"]
    _, r90, _ = mine["nh90"]
    for i in range(1, len(r90)):
        k = K0 + i
        day = DAYS[k]
        if isoweek(day) == isoweek(DAYS[k - 1]):
            continue
        held = set(r90[i - 1]["w"])
        _, rk_all = rank(k, day, 120, None)
        _, rk_f = rank(k, day, 120, 0.90)
        dropped = [r for r in rk_all if r["key"] in held and r["key"] not in {q["key"] for q in rk_f}]
        if dropped:
            r0 = dropped[0]
            c = C[r0["key"]]
            j = int(np.nanargmax(c[max(0, k - 252):k])) + max(0, k - 252)
            nb = {"session": day, "name": r0["t"], "close_k-1": round(float(c[k - 1]), 3), "close_date": DAYS[k - 1],
                  "max_close_252": round(float(c[j]), 3), "max_date": DAYS[j], "ratio": round(float(r0["nh"]), 4),
                  "sold_in_engine": r0["key"] not in eng["nh90"][1][i]["w"], "window_start": DAYS[max(0, k - 252)]}
            res["B_nh90"] = nb
            print("  nh90 제외 예:", nb)
            break

    # B2. why daily differs: classify every mid-week membership change, and trace DELL entry dates
    print("\nB2. 주중 교체 분류(daily)")
    ev = []
    for i in range(1, len(rD)):
        k = K0 + i
        d = DAYS[k]
        if isoweek(d) != isoweek(DAYS[k - 1]):
            continue
        prev, cur = set(rD[i - 1]["w"]), set(rD[i]["w"])
        if prev == cur:
            continue
        pdf, rk = rank(k, d)
        pos = {r["key"]: j + 1 for j, r in enumerate(rk)}
        for x in prev - cur:
            why = "PDF에서 빠짐/가격 없음" if x not in pos else f"순위 {pos[x]}위(12위 밖)"
            ev.append({"day": d, "out": TICK[x], "why": why, "in": sorted(TICK[y] for y in cur - prev)})
    res["B2_midweek_swaps"] = ev
    n_pdf = sum(1 for e in ev if e["why"].startswith("PDF"))
    print(f"  주중 편출 {len(ev)}건 중 TIME PDF 편출 때문 {n_pdf}건")
    for e in ev:
        print("   ", e)
    dell = next(x for x in KEYS if TICK[x] == "DELL")
    tl = {}
    for nm, rows in (("base", rB), ("daily", rD), ("weekly_swap", None)):
        if rows is None:
            rows = sim("weekly_swap")[1]
        held = [r["day"] for r in rows if dell in r["w"] and "2026-04-01" <= r["day"] <= "2026-05-29"]
        tl[nm] = [held[0], held[-1], len(held)] if held else None
    res["B2_DELL_held_Apr_May"] = tl
    print("  DELL 보유(4~5월) 첫날·마지막날·일수:", tl)
    for i in range(len(rD)):
        d = rD[i]["day"]
        if "2026-04-27" <= d <= "2026-05-15":
            print("   ", d, "base", sorted(TICK[x] for x in rB[i]["w"]), "| daily", sorted(TICK[x] for x in rD[i]["w"]))

    # C. user's question facts
    print("\nC. MU / PLTR")
    def dates_with(snapw, pdfs, t):
        return [d for d in pdfs if any(TICK.get(x) == t and v > 0 for x, v in snapw[d].items())]
    mu_t, mu_k = dates_with(TW, TPDF, "MU"), dates_with(KW, KPDF, "MU")
    pl_t, pl_k = dates_with(TW, TPDF, "PLTR"), dates_with(KW, KPDF, "PLTR")
    res["C"] = {"TIME_pdfs": len(TPDF), "KoAct_pdfs": len(KPDF),
                "MU_in_TIME": [len(mu_t), mu_t[:1], mu_t[-1:]], "MU_in_KoAct": [len(mu_k), mu_k[:1], mu_k[-1:]],
                "PLTR_in_TIME": [len(pl_t), pl_t[:1], pl_t[-1:]], "PLTR_in_KoAct": [len(pl_k), pl_k[:1], pl_k[-1:]]}
    print(" ", res["C"])
    mukey = next(x for x in KEYS if TICK[x] == "MU")
    wr = []
    for k in range(K0, len(DAYS)):
        day = DAYS[k]
        if k > K0 and isoweek(day) == isoweek(DAYS[k - 1]):
            continue
        if day < "2026-06-01":
            continue
        _, rk = rank(k, day, 120, None, extra=(mukey,))
        pos = next((j + 1 for j, r in enumerate(rk) if r["key"] == mukey), None)
        mom = next((r["mom"] for r in rk if r["key"] == mukey), None)
        wr.append((day, pos, round(mom, 3) if mom is not None else None, len(rk)))
    res["C"]["MU_would_be_rank_weekly_back"] = wr
    print("  MU를 TIME 후보에 넣었다면 120일 순위(뒤 구간 주간 재조정일):", wr)
    # 10/5 ranking check
    k105 = DAYS.index("2026-10-05")
    _, rk105 = rank(k105, "2026-10-05")
    res["C"]["rank_2026-10-05_top14"] = [r["t"] for r in rk105[:14]]
    print("  10/5 순위 1~14:", res["C"]["rank_2026-10-05_top14"], "| base 보유:", sorted(TICK[x] for x in rB[k105 - K0]["w"]))

    # D. robustness
    print("\nD. 강건성")
    mb, md = metrics(mine["base"][0]), metrics(mine["daily"][0])
    res["D_month_diff_daily_minus_base"] = {m: round(md["월"][m] - mb["월"][m], 4) for m in md["월"]}
    print("  daily-base 월별 차:", res["D_month_diff_daily_minus_base"])
    for nm in ("base", "daily", "nh90"):
        v, rows, _ = mine[nm]
        for lab, a, b in (("앞", "2026-01-02", "2026-05-29"), ("뒤", "2026-06-01", "2026-10-09"), ("7월", "2026-07-01", "2026-07-31")):
            pn = attribution(v, rows, a, b)
            res.setdefault("D_attr", {}).setdefault(nm, {})[lab] = {t: round(x, 3) for t, x in sorted(pn.items(), key=lambda kv: -abs(kv[1]))[:8]}
    for lab in ("앞", "뒤", "7월"):
        b_, d_, n_ = (res["D_attr"][x][lab] for x in ("base", "daily", "nh90"))
        print(f"  [{lab}] base {b_}\n        daily {d_}\n        nh90 {n_}")
    # name-level difference daily - base
    for lab, a, b in (("앞", "2026-01-02", "2026-05-29"), ("뒤", "2026-06-01", "2026-10-09")):
        pb = attribution(*mine["base"][:2], a, b)
        pd_ = attribution(*mine["daily"][:2], a, b)
        dd = {t: round(pd_.get(t, 0) - pb.get(t, 0), 3) for t in set(pb) | set(pd_)}
        top = dict(sorted(dd.items(), key=lambda kv: -abs(kv[1]))[:6])
        res.setdefault("D_attr_diff_daily_minus_base", {})[lab] = top
        print(f"  daily-base 종목별 손익 차 [{lab}]:", top, "합", round(sum(dd.values()), 3))

    # D2 membership-only daily refresh (diagnostic, NOT a selectable candidate)
    vs, rs, ts = sim("weekly_swap")
    ms = metrics(vs)
    res["D_weekly_swap_diagnostic"] = {**ms, "trades": len(ts), "sum_abs_turnover": round(sum(t for _, t in ts), 3)}
    print("  [진단] 주간 재조정 + 주중엔 7종목이 바뀔 때만 교체:", {k: ms[k] for k in ("앞", "뒤", "9/1~", "전체", "전체_mdd")},
          "월", ms["월"], "거래일", len(ts))

    # D3 rebalance every m sessions, all phases
    every = {}
    for m in (1, 2, 3, 5, 10):
        fr, bk, fu = [], [], []
        for ph in range(m):
            v, _, _ = sim("every", every=m, phase=ph)
            mm = metrics(v)
            fr.append(mm["앞"]), bk.append(mm["뒤"]), fu.append(mm["전체"])
        every[m] = {"앞 평균": round(np.mean(fr), 4), "앞 범위": [round(min(fr), 4), round(max(fr), 4)],
                    "뒤 평균": round(np.mean(bk), 4), "뒤 범위": [round(min(bk), 4), round(max(bk), 4)],
                    "전체 평균": round(np.mean(fu), 4)}
        print(f"  {m}거래일마다(위상 {m}개):", every[m])
    res["D_every_m"] = every

    # D4 neighbours
    nbr = {}
    for L in (40, 60, 80, 100, 120, 140):
        mm = metrics(sim(L=L)[0])
        nbr[f"L{L}"] = {k: mm[k] for k in ("앞", "뒤", "9/1~", "전체", "전체_mdd")}
    for th in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        v, rows, turn = sim(nh=th)
        mm = metrics(v)
        nbr[f"nh{int(th * 100)}"] = {**{k: mm[k] for k in ("앞", "뒤", "9/1~", "전체", "전체_mdd")},
                                     "7월": mm["월"]["07"], "거래일": len(turn)}
    res["D_neighbours"] = nbr
    for k_, x in nbr.items():
        print(f"  이웃 {k_:5}", x)

    # D5 spread cost: c per unit of |Δw| (each side)
    sc = {}
    for c in (0.0005, 0.001):
        sc[c] = {}
        for name, kw in (("base", {}), ("daily", {"mode": "daily"}), ("nh80", {"nh": 0.80}), ("nh90", {"nh": 0.90}),
                         ("lb60", {"L": 60})):
            mm = metrics(sim(cost=c, **kw)[0])
            sc[c][name] = {k: mm[k] for k in ("앞", "뒤", "전체")}
        print(f"  스프레드 {c:.4f}/편도:", sc[c])
    res["D_spread"] = {str(k): v for k, v in sc.items()}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    print("저장:", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
