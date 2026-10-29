"""F5 개장 전 상태로 추세·되돌림 고르기 (BRIEF2.md 묶음 F5).

핵심 규칙: 개장 전에 아는 상태 하나 s가 sel 안 분위 문턱 thr보다 크면(hi) / 작거나 같으면(lo) 두 기본 규칙 중 하나를 쓴다.
- 추세(trend): 처음 N분(09:30~) 합친 봉이 양봉이면 SOXL, 음봉이면 SOXS(F1 양·음봉 규칙과 같은 꼴).
- 되돌림(rev): 그 반대 종목.
- 결정 = N번째 봉 마감(N=1 09:31, 5 09:35, 15 09:45), 체결 = 엔진 규칙(다음 봉 종가, 1분 지연).
- 손절 stop='rng'(F1 핵심): 첫 N분 SOXL 범위에서 보유 쪽이 지는 끝까지의 거리(SOXL 기준 %).
  SOXL 보유면 (C − 범위 저가)/C, SOXS 보유면 (범위 고가 − C)/C. C = 결정 봉 SOXL 종가.
  거리 바닥 FLOOR = 0.5%(장초 1분 |수익| 중앙값 0.39~0.50%, 구조 진단 f): 되돌림 쪽은 위 꼬리가 0이면 거리가 0이 되어
  손절이 체결 직후 걸리거나(엔진은 stop=0을 '손절 없음'으로 읽는다) 의미가 없어서다. stop='none'은 손절 없음.
- 마감: 엔진 주문 마감(actual: 서머타임 11:00·표준시 10:00, std: 10:00) 1분 전 봉 종가.
- 첫 N분 봉이 보합(시가 = 종가)이거나 상태 값이 없으면 그날은 현금.

상태 8개(모두 결정 전에 아는 값, ctx로만 계산):
  prev_range   전날 SOXL 정규장 (고가 − 저가) ÷ 시가
  abs_gap      |오늘 09:30 시가 ÷ 전날 정규장 종가 − 1|
  prev_ret     전날 정규장 종가 ÷ 그 전날 종가 − 1
  trend5       전날 종가 ÷ 6거래일 전 종가 − 1(5일 수익)
  prev_abs_eff |전날 09:30~11:00 방향 효율| = |11:00 직전 종가 − 09:30 시가| ÷ (창 고가 − 창 저가)
  weekday      요일(월 0 ~ 금 4)
  abs_QQQ_pre  |QQQ 08:00 가격 → 09:29 봉 종가|
  abs_NVDA_pre |NVDA 08:00 가격 → 09:29 봉 종가|
반일장(13:00 마감) 전날은 정규장을 09:30~13:00으로 자른다(lib Data.daily는 15:59 연장 종가를 쓰는 결함이 있어 쓰지 않는다).

짝 pair: 'hi_trend' = hi면 추세·lo면 되돌림, 'hi_rev' = hi면 되돌림·lo면 추세.
거울(mirror=True): 매일 보유 종목을 뒤집고, 손절은 같은 규칙(뒤집힌 보유 쪽이 지는 범위 끝)으로 다시 잰다.
  그래서 (…, hi_trend) 의 거울은 (…, hi_rev) 와 같다.

실행: py fam_F5.py → results/fam_F5_sel.csv, results/fam_F5_frozen.json (sel만 연다).
"""
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import (Data, Engine, Order, RTH0, RTH1, ROOT, idx, summarize, load_costs)   # noqa: E402

FLOOR = 0.005
I0800, I1100, I1300 = idx("08:00"), idx("11:00"), idx("13:00")
HALF_DAYS = {dt.date(2024, 12, 24), dt.date(2025, 7, 3), dt.date(2025, 11, 28), dt.date(2025, 12, 24)}  # 13:00 마감
STATES = ["prev_range", "abs_gap", "prev_ret", "trend5", "prev_abs_eff", "weekday", "abs_QQQ_pre", "abs_NVDA_pre"]
NS, QS, PAIRS, STOPS = [1, 5, 15], [33, 50, 67], ["hi_trend", "hi_rev"], ["rng", "none"]
SEED = 20261029


# ───────── 개장 전 상태(ctx만 사용) ─────────
def rth_ohlc(ctx, k, sym="SOXL"):
    """k거래일 전 정규장 시가·고가·저가·종가. 반일장은 13:00 마감. 없으면 None."""
    if ctx.d - k < 0:
        return None
    day = ctx._D.days[ctx.d - k]                  # 지난 날짜(가격 아님)만 읽는다
    e = I1300 if day in HALF_DAYS else RTH1
    O, H, L, C = (ctx.prev_bars(sym, f, k) for f in ("Open", "High", "Low", "Close"))
    return O[RTH0], np.nanmax(H[RTH0:e]), np.nanmin(L[RTH0:e]), C[e - 1]


def pre_ret(ctx, sym):
    """08:00 가격(07:59 봉 종가, 없으면 08:00~09:29 첫 봉 시가) → 09:29 봉 종가. ctx.i ≥ 09:29 봉일 때."""
    c = ctx.bars(sym, "Close")
    o = ctx.bars(sym, "Open")
    p0 = c[I0800 - 1]
    if not np.isfinite(p0):
        w = o[I0800:RTH0]
        f = np.where(np.isfinite(w))[0]
        p0 = w[f[0]] if len(f) else np.nan
    return c[RTH0 - 1] / p0 - 1


def state_value(ctx, name):
    """결정 봉(≥ 09:30 봉) 시점에 아는 상태 값."""
    if name == "weekday":
        return float(ctx.day.weekday())
    if name in ("abs_QQQ_pre", "abs_NVDA_pre"):
        return abs(pre_ret(ctx, name[4:-4]))
    p1 = rth_ohlc(ctx, 1)
    if p1 is None:
        return np.nan
    o1, h1, l1, c1 = p1
    if name == "prev_range":
        return (h1 - l1) / o1
    if name == "abs_gap":
        return abs(ctx.bars("SOXL", "Open")[RTH0] / c1 - 1)
    if name == "prev_ret":
        p2 = rth_ohlc(ctx, 2)
        return np.nan if p2 is None else c1 / p2[3] - 1
    if name == "trend5":
        p6 = rth_ohlc(ctx, 6)
        return np.nan if p6 is None else c1 / p6[3] - 1
    if name == "prev_abs_eff":
        O, H, L, C = (ctx.prev_bars("SOXL", f, 1) for f in ("Open", "High", "Low", "Close"))
        rng = np.nanmax(H[RTH0:I1100]) - np.nanmin(L[RTH0:I1100])
        return abs(C[I1100 - 1] - O[RTH0]) / rng if rng > 0 else np.nan
    raise KeyError(name)


# ───────── 전략 ─────────
class F5:
    start = "09:30"

    def __init__(self, p, mirror=False):
        self.p, self.mirror = p, mirror
        self.dec = RTH0 + int(p["N"]) - 1

    def on_bar(self, ctx, i):
        if i != self.dec:
            return None
        p = self.p
        o = ctx.bars("SOXL", "Open")[RTH0]
        c = ctx.c("SOXL")[i]
        if not (np.isfinite(o) and np.isfinite(c)) or c == o:
            return None
        first = 1 if c > o else -1
        if p.get("state") is None:                 # 참고판: 상태 없이 늘 같은 기본 규칙
            mode = p["mode"]
        else:
            s = state_value(ctx, p["state"])
            if not np.isfinite(s):
                return None
            hi = s > p["thr"]
            mode = "trend" if hi == (p["pair"] == "hi_trend") else "rev"
        pos = first if mode == "trend" else -first
        if self.mirror:
            pos = -pos
        stop = None
        if p["stop"] == "rng":
            hh = np.nanmax(ctx.bars("SOXL", "High")[RTH0:i + 1])
            ll = np.nanmin(ctx.bars("SOXL", "Low")[RTH0:i + 1])
            dist = (c - ll) / c if pos == 1 else (hh - c) / c
            stop = max(float(dist), FLOOR)
        return Order(pos, stop=stop)


def make(params, mirror=False):
    """Engine.run(make(params), ...)에 그대로 넣는다. params: state, thr, pair, N, stop (q는 기록용)."""
    return lambda: F5(params, mirror)


# ───────── 격자 ─────────
def sel_states(D, days):
    """sel 날마다 8개 상태 값(문턱 계산용). ctx.i = 09:30 봉(시가가 보이는 첫 봉)."""
    from lib import Ctx
    rows = []
    for d in days:
        ctx = Ctx(D, d)
        ctx.i = RTH0
        rows.append({s: state_value(ctx, s) for s in STATES} | {"day": D.days[d]})
    return pd.DataFrame(rows).set_index("day")


def cfg_id(p):
    if p.get("state") is None:
        return f"REF|{p['mode']}|N{p['N']}|{p['stop']}"
    return f"{p['state']}|q{p['q']}|N{p['N']}|{p['pair']}|{p['stop']}"


def neighbors(p):
    out = []
    for ax, vals in (("N", NS), ("q", QS)):
        k = vals.index(p[ax])
        for j in (k - 1, k + 1):
            if 0 <= j < len(vals):
                out.append(cfg_id({**p, ax: vals[j]}))
    out.append(cfg_id({**p, "stop": "rng" if p["stop"] == "none" else "none"}))
    return out


def main():
    t0 = time.time()
    D = Data()
    sel = D.days_in("sel")
    print(f"자료 {time.time() - t0:.1f}s, sel {len(sel)}일, 비용표 {load_costs()['SOXL']}")
    S = sel_states(D, sel)
    thr = {s: {q: float(np.nanquantile(S[s].values, q / 100)) for q in QS} for s in STATES}
    nan_n = {s: int(S[s].isna().sum()) for s in STATES}

    grid = []
    for s in STATES:
        for q in QS:
            for N in NS:
                for pair in PAIRS:
                    for st in STOPS:
                        grid.append({"state": s, "q": q, "thr": thr[s][q], "N": N, "pair": pair, "stop": st})
    refs = [{"state": None, "mode": m, "N": N, "stop": st} for m in ("trend", "rev") for N in NS for st in STOPS]
    print(f"격자 {len(grid)} + 참고 {len(refs)} = {len(grid) + len(refs)}")

    engines = {"actual": Engine(D), "std": Engine(D, scheme="std")}
    rows, dailies = [], {}
    for p in grid + refs:
        cid = cfg_id(p)
        for sch, E in engines.items():
            daily, tr = E.run(make(p), days=sel)
            sm = summarize(daily)
            dailies[(cid, sch)] = daily["ret"].values
            if p.get("state") is not None:
                hi = S[p["state"]].values > p["thr"]
                trend_share = float(np.mean(hi == (p["pair"] == "hi_trend")))
            else:
                trend_share = 1.0 if p["mode"] == "trend" else 0.0
            rows.append({"id": cid, "scheme": sch, "state": p.get("state") or "REF", "q": p.get("q"),
                         "thr": p.get("thr"), "N": p["N"], "pair": p.get("pair") or p["mode"], "stop": p["stop"],
                         "days": sm["days"], "active_days": sm["active_days"], "trades": sm["trades"],
                         "mean_pct": sm["mean_pct"], "sd_pct": sm["sd_pct"], "hit_days": sm["hit_days"], "t": sm["t"],
                         "ci95_lo": sm["ci95"][0], "ci95_hi": sm["ci95"][1], "p_one_sided": sm["p_one_sided"],
                         "max_dd_pct": sm["max_dd_pct"], "compound_pct": sm["compound_pct"],
                         "trend_day_share": trend_share,
                         "stop_exit_share": float((tr["why"] == "stop").mean()) if len(tr) else None,
                         "sym_soxl_share": float((tr["sym"] == "SOXL").mean()) if len(tr) else None})
        print(f"  {cid}: actual t={rows[-2]['t']:+.2f} std t={rows[-1]['t']:+.2f}  ({time.time() - t0:.0f}s)")
    R = pd.DataFrame(rows)

    # 이웃 t 중앙값(같은 상태·짝, N·q 한 칸, 손절 켜고 끄기)
    tmap = {(r.id, r.scheme): r.t for r in R.itertuples()}
    nb_med, nb_n = [], []
    pmap = {cfg_id(p): p for p in grid}
    for r in R.itertuples():
        if r.id in pmap:
            ts = [r.t] + [tmap[(n, r.scheme)] for n in neighbors(pmap[r.id])]
            nb_med.append(float(np.median(ts)))
            nb_n.append(len(ts))
        else:
            nb_med.append(np.nan)
            nb_n.append(0)
    R["neighbor_median_t"], R["n_with_neighbors"] = nb_med, nb_n
    out_dir = os.path.join(ROOT, "results")
    R.to_csv(os.path.join(out_dir, "fam_F5_sel.csv"), index=False, encoding="utf-8-sig")

    # 고르기: 활동일 ≥ 25, 이웃 t 중앙값 1위(같으면 자기 t)
    frozen, picked = [], {}
    for sch in ("actual", "std"):
        g = R[(R.scheme == sch) & R.id.isin(pmap) & (R.active_days >= 25)]
        top = g.sort_values(["neighbor_median_t", "t"], ascending=False).iloc[0]
        picked[sch] = top.id
    ids = list(dict.fromkeys([picked["actual"], picked["std"]]))
    for cid in ids:
        p = pmap[cid]
        sel_actual = R[(R.id == cid) & (R.scheme == "actual")].iloc[0].to_dict()
        sel_std = R[(R.id == cid) & (R.scheme == "std")].iloc[0].to_dict()
        mir = {}
        for sch, E in engines.items():
            dm, _ = E.run(make(p, mirror=True), days=sel)
            ms = summarize(dm)
            mir[sch] = {"mean_pct": ms["mean_pct"], "t": ms["t"], "ci95": ms["ci95"]}
        pair_id = cfg_id({**p, "pair": "hi_rev" if p["pair"] == "hi_trend" else "hi_trend"})
        mir_check = {sch: float(np.max(np.abs(engines[sch].run(make(p, mirror=True), days=sel)[0]["ret"].values
                                               - dailies[(pair_id, sch)]))) for sch in engines}
        schemes = [s for s in ("actual", "std") if picked[s] == cid]
        frozen.append({
            "id": f"F5_{'_'.join(schemes)}", "config": cid, "picked_for": schemes,
            "params": {k: p[k] for k in ("state", "q", "thr", "N", "pair", "stop")} | {"floor": FLOOR},
            "factory": f"fam_F5.make({json.dumps({k: p[k] for k in ('state', 'q', 'thr', 'N', 'pair', 'stop')})})",
            "mirror_factory": f"fam_F5.make(<params>, mirror=True)  # = {pair_id}",
            "code": os.path.join(ROOT, "code", "fam_F5.py"),
            "sel_actual": sel_actual, "sel_std": sel_std,
            "neighbor_median_t": {"actual": sel_actual["neighbor_median_t"], "std": sel_std["neighbor_median_t"]},
            "neighbors": neighbors(p),
            "mirror_sel": mir, "mirror_equals_pair_config_maxdiff": mir_check,
        })

    def land(sch):
        g = R[(R.scheme == sch) & R.id.isin(pmap)]
        return {"n": int(len(g)), "t_max": float(g.t.max()), "t_median": float(g.t.median()),
                "t_min": float(g.t.min()), "pos_share": float((g.t > 0).mean()),
                "t_gt2_n": int((g.t > 2).sum()), "nbmed_max": float(g.neighbor_median_t.max()),
                "nbmed_median": float(g.neighbor_median_t.median()),
                "mean_abs_t_minus_nbmed": float((g.t - g.neighbor_median_t).abs().mean())}
    meta = {
        "family": "F5", "period": "sel 2026-01-02~2026-06-30", "n_configs_grid": len(grid), "n_configs_ref": len(refs),
        "n_configs_total": len(grid) + len(refs),
        "grid": "상태 8 × 문턱 분위 3(33·50·67) × 첫 움직임 N 3(1·5·15분) × 짝 2(hi_trend·hi_rev) × 손절 2(rng·none) = 288; "
                "참고(상태 없이 늘 추세/늘 되돌림) 2 × N 3 × 손절 2 = 12",
        "engine": "lib.Engine fill=close, cost_table.json 1배, actual·std",
        "thresholds": thr, "state_nan_days": nan_n,
        "state_quantiles_sel": {s: {k: float(v) for k, v in S[s].describe().items()} for s in STATES},
        "landscape": {"actual": land("actual"), "std": land("std")},
        "picked": picked, "frozen": frozen,
        "refs": R[R.id.str.startswith("REF")].to_dict("records"),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, "fam_F5_frozen.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps({"picked": picked, "landscape": meta["landscape"]}, ensure_ascii=False, indent=1))
    print(f"끝 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
