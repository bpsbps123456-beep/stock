"""감사 3·4: 체결 현실성 민감도와 주문 가능 시간.
impl_a.py의 자료·RSI·신호 함수만 빌려 쓰고(파일은 고치지 않음), 매매 표는 변형 조건을 받는 자체 함수로 다시 계산한다.
기준 조건(v0)은 impl_a 주 칸과 같아야 한다(assert).
- 비용: 진입·시간 청산 한 쪽당 $0.02/$0.03/$0.05, 0.02%/0.05%/0.10%
- 목표 관통: +$0.02/$0.03/$0.05, +0.05%·0.10%, 종가 ≥ 목표
- 진입 봉 고가 제외, 진입 지연 E3·E4, 진입 봉 중간값 체결
- 거울 대조: Y의 도달 고가를 반대 ETF 같은 분 저가로 다시 만든 값으로도 목표 관통인가(한 종목만의 튐 거르기)
- 튐 봉(SOXS 2026-10-21 10:59 고가 47.10) 정리 뒤 32칸 다시 계산
- 스프레드 대략 추정(1분봉 고저폭·Abdi-Ranaldo·Roll), 장초 분당 거래대금
- 주문 시간: 서머타임 24:00 KST = 11:00 ET, 표준시 24:00 KST = 10:00 ET
사용: py code/audit_fill.py → results/audit_fill.json
"""
import copy
import json
import os
import sys

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import impl_a as A  # noqa: E402

ROOT = A.ROOT
OUT = os.path.join(ROOT, "results", "audit_fill.json")
T = A.TICKERS


def r6(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 6)


# ------------------------------------------------------------ 자료(impl_a와 같은 경로)
data = {t: A.load(t) for t in T}
days = sorted(set(data["SOXL"].index[(A.minute_of_day(data["SOXL"].index) >= 570)
                                     & (A.minute_of_day(data["SOXL"].index) <= 959)].date))
ND = len(days)
grids = {t: A.regular_grid(data[t], days) for t in T}
rsi = {t: A.rsi_series(data[t], "R") for t in T}
sig_w60 = {t: A.signals_for(rsi[t], days, 60)["T1"] for t in T}


# ------------------------------------------------------------ 변형 매매 표
def table(g, hold=30, entry_cost=("abs", 0.01), exit_cost=("abs", 0.01), through=("abs", 0.01),
          hit_mode="high", skip_entry_bar=False, entry_px="open"):
    O, H, L, C = g["O"], g["H"], g["L"], g["C"]
    ne = O.shape[1] - hold + 1
    base = O[:, :ne] if entry_px == "open" else ((H + L) / 2)[:, :ne]
    P = base + (entry_cost[1] if entry_cost[0] == "abs" else base * entry_cost[1] / 100)
    tgt = P * 1.01
    thr = tgt + (through[1] if through[0] == "abs" else tgt * through[1] / 100)
    Hw = sliding_window_view(H, hold, axis=1)
    Cw = sliding_window_view(C, hold, axis=1)
    Lw = sliding_window_view(L, hold, axis=1)
    up = (Hw >= thr[..., None]) if hit_mode == "high" else (Cw >= tgt[..., None])
    if skip_entry_bar:
        up[..., 0] = False
    hit = up.any(-1)
    j = np.where(hit, up.argmax(-1), hold - 1)
    last = C[:, hold - 1:hold - 1 + ne]
    xc = exit_cost[1] if exit_cost[0] == "abs" else last * exit_cost[1] / 100
    pnl = np.where(hit, 1.0, ((last - xc) / P - 1) * 100)
    cmin = np.minimum.accumulate(Lw, axis=-1)
    mae = (np.take_along_axis(cmin, j[..., None], -1)[..., 0] / P - 1) * 100
    # 도달 봉 저가를 뺀 MAE(도달 봉 안에서 저가가 목표 뒤였을 수 있음): 진입 봉 도달이면 0
    jm1 = np.maximum(j - 1, 0)
    mae_ex = np.where(hit & (j == 0), 0.0,
                      (np.take_along_axis(cmin, np.where(hit, jm1, j)[..., None], -1)[..., 0] / P - 1) * 100)
    return {"P": P, "hit": hit, "exit_off": j, "pnl": pnl, "mae": mae, "mae_ex_hitbar": mae_ex, "open": O[:, :ne]}


def run(tables, k, sig=sig_w60, mode="fade"):
    rows = []
    for X, sg in sig.items():
        for d, s in zip(sg["d"], sg["s"]):
            rows.append((d, s, X, A.OTHER[X] if mode == "fade" else X))
    rows.sort(key=lambda r: (r[0], r[3], r[1] + k, r[2]))
    recs, last = [], {}
    for d, s, X, Y in rows:
        e = s + k
        tb = tables[Y]
        if e >= tb["P"].shape[1] or np.isnan(tb["open"][d, e]):
            continue
        if e < last.get((d, Y), -10**9) + 1:
            continue
        last[(d, Y)] = e + int(tb["exit_off"][d, e])
        recs.append({"d": d, "s": s, "X": X, "Y": Y, "e": e, "hit": bool(tb["hit"][d, e]),
                     "pnl": float(tb["pnl"][d, e]), "mae": float(tb["mae"][d, e]),
                     "mae_ex": float(tb["mae_ex_hitbar"][d, e]), "exit_off": int(tb["exit_off"][d, e])})
    return pd.DataFrame(recs)


def b0(tables, k, nwin=60):
    parts = []
    for Y in T:
        tb = tables[Y]
        e = np.arange(nwin) + k
        dd, ee = np.meshgrid(np.arange(ND), e, indexing="ij")
        parts.append(pd.DataFrame({"d": dd.ravel(), "hit": tb["hit"][dd, ee].ravel(), "pnl": tb["pnl"][dd, ee].ravel()}))
    return pd.concat(parts, ignore_index=True)


def boot(fade, base0):
    rng = np.random.default_rng(A.SEED)
    idx = rng.integers(0, ND, size=(A.NBOOT, ND))

    def agg(df):
        n = np.bincount(df["d"], minlength=ND).astype(float)
        s = np.bincount(df["d"], weights=df["pnl"], minlength=ND)
        h = np.bincount(df["d"], weights=df["hit"].astype(float), minlength=ND)
        N = n[idx].sum(1)
        return s[idx].sum(1) / N, h[idx].sum(1) / N
    fe, fh = agg(fade)
    be, bh = agg(base0)
    q = lambda v: [r6(x) for x in np.nanpercentile(v, [2.5, 97.5])]
    return {"fade_ev_ci": q(fe), "fade_minus_b0_hit_ci": q(fh - bh), "fade_minus_b0_ev_ci": q(fe - be)}


first_half = np.array([dd <= A.HALF_SPLIT for dd in days])


def summarize(name, kw, k=2):
    tabs = {t: table(grids[t], **kw) for t in T}
    f = run(tabs, k)
    b = b0(tabs, k)
    bs = boot(f, b)
    h1 = f[first_half[f["d"]]]
    h2 = f[~first_half[f["d"]]]
    return {"variant": name, "fade_n": len(f), "fade_hit": r6(f["hit"].mean()), "fade_ev": r6(f["pnl"].mean()),
            "b0_hit": r6(b["hit"].mean()), "b0_ev": r6(b["pnl"].mean()),
            "fade_minus_b0_hit": r6(f["hit"].mean() - b["hit"].mean()),
            "fade_ev_ci": bs["fade_ev_ci"], "fade_minus_b0_hit_ci": bs["fade_minus_b0_hit_ci"],
            "half1_ev": r6(h1["pnl"].mean()), "half2_ev": r6(h2["pnl"].mean()),
            "mae_median": r6(f["mae"].median()), "mae_median_ex_hitbar": r6(f["mae_ex"].median())}, f


variants = [
    ("v0 기준(BRIEF 그대로)", {}),
    ("비용 한쪽 $0.02", {"entry_cost": ("abs", 0.02), "exit_cost": ("abs", 0.02)}),
    ("비용 한쪽 $0.03", {"entry_cost": ("abs", 0.03), "exit_cost": ("abs", 0.03)}),
    ("비용 한쪽 $0.05", {"entry_cost": ("abs", 0.05), "exit_cost": ("abs", 0.05)}),
    ("비용 한쪽 0.02%", {"entry_cost": ("pct", 0.02), "exit_cost": ("pct", 0.02)}),
    ("비용 한쪽 0.05%", {"entry_cost": ("pct", 0.05), "exit_cost": ("pct", 0.05)}),
    ("비용 한쪽 0.10%", {"entry_cost": ("pct", 0.10), "exit_cost": ("pct", 0.10)}),
    ("관통 +$0.02", {"through": ("abs", 0.02)}),
    ("관통 +$0.03", {"through": ("abs", 0.03)}),
    ("관통 +$0.05", {"through": ("abs", 0.05)}),
    ("관통 +0.05%", {"through": ("pct", 0.05)}),
    ("관통 +0.10%", {"through": ("pct", 0.10)}),
    ("도달 = 종가 ≥ 목표", {"hit_mode": "close"}),
    ("진입 봉 고가 제외", {"skip_entry_bar": True}),
    ("진입 = 진입 봉 (고+저)/2 + 0.01, 진입 봉 고가 제외", {"entry_px": "mid", "skip_entry_bar": True}),
    ("묶음: 비용 0.03%·관통 +$0.02·진입 봉 제외", {"entry_cost": ("pct", 0.03), "exit_cost": ("pct", 0.03),
                                         "through": ("abs", 0.02), "skip_entry_bar": True}),
]
res = []
base_f = None
for name, kw in variants:
    r, f = summarize(name, kw)
    res.append(r)
    if base_f is None:
        base_f = f
# 기준이 impl_a 주 칸과 같은지
m = json.load(open(os.path.join(ROOT, "results", "impl_a.json"), encoding="utf-8"))["cells"][A.MAIN]
assert res[0]["fade_n"] == m["fade"]["n"] and abs(res[0]["fade_ev"] - m["fade"]["ev_mean_pct"]) < 1e-5
assert abs(res[0]["b0_hit"] - m["b0"]["hit_rate"]) < 1e-5

# 진입 지연(E3·E4: 신호 봉 시작 + 3·4분 시가) — BRIEF 격자 밖, 지연 민감도만
delay = []
for k in [1, 2, 3, 4, 5]:
    tabs = {t: table(grids[t]) for t in T}
    f = run(tabs, k)
    b = b0(tabs, k)
    bs = boot(f, b)
    delay.append({"fill": f"s+{k}분 시가", "fade_n": len(f), "fade_hit": r6(f["hit"].mean()), "fade_ev": r6(f["pnl"].mean()),
                  "b0_hit": r6(b["hit"].mean()), "b0_ev": r6(b["pnl"].mean()), "fade_ev_ci": bs["fade_ev_ci"]})

# 손익분기 한쪽 비용(%) — 기준 관통 +$0.01 고정
be = None
for c in np.arange(0.0, 0.5001, 0.005):
    tabs = {t: table(grids[t], entry_cost=("pct", c), exit_cost=("pct", c)) for t in T}
    f = run(tabs, 2)
    if f["pnl"].mean() <= 0:
        be = round(float(c), 3)
        break

# ------------------------------------------------------------ 거울 대조: 도달 봉 고가를 반대 ETF 저가로 다시 만든 값
mirror = []
for _, r in base_f[base_f["hit"]].iterrows():
    d, e, j, Y = int(r["d"]), int(r["e"]), int(r["exit_off"]), r["Y"]
    t = e + j
    Z = A.OTHER[Y]
    gy, gz = grids[Y], grids[Z]
    prev_y = gy["C"][d, t - 1] if t > 0 else np.nan
    prev_z = gz["C"][d, t - 1] if t > 0 else np.nan
    # Y 수익 ≈ −(Z 수익) (둘 다 3배, 방향 반대). Y 고가 ↔ Z 저가
    implied_high = prev_y * (1 - (gz["L"][d, t] / prev_z - 1))
    P = gy["O"][d, e] + 0.01
    thr = P * 1.01 + 0.01
    mirror.append({"day": days[d].isoformat(), "buy": Y, "hit_bar": A.hhmm(t), "high": r6(gy["H"][d, t]),
                   "mirror_implied_high": r6(implied_high), "threshold": r6(thr),
                   "mirror_through": bool(implied_high >= thr),
                   "mirror_through_any_bar_to_hit": None})
    # 도달 봉까지 어느 봉이든 거울 고가가 관통했나
    any_ok = False
    for tt in range(e, t + 1):
        if tt == 0:
            continue
        ih = gy["C"][d, tt - 1] * (1 - (gz["L"][d, tt] / gz["C"][d, tt - 1] - 1))
        if ih >= thr:
            any_ok = True
    mirror[-1]["mirror_through_any_bar_to_hit"] = any_ok
mir = pd.DataFrame(mirror)

# ------------------------------------------------------------ 거울 불일치(한 종목만 튄 봉) 전수: 09:30~11:00
mm = []
for d in range(ND):
    for t in range(1, 91):
        L_, S_ = grids["SOXL"], grids["SOXS"]
        upL = L_["H"][d, t] / L_["C"][d, t - 1] - 1
        dnS = S_["L"][d, t] / S_["C"][d, t - 1] - 1
        upS = S_["H"][d, t] / S_["C"][d, t - 1] - 1
        dnL = L_["L"][d, t] / L_["C"][d, t - 1] - 1
        a, b = upL + dnS, upS + dnL      # 거울이면 ≈ 0(3배 ±, 같은 분)
        if abs(a) > 0.006 or abs(b) > 0.006:
            mm.append({"ts": f"{days[d]} {A.hhmm(t)}", "SOXL_up+SOXS_dn_pct": r6(a * 100), "SOXS_up+SOXL_dn_pct": r6(b * 100)})

# ------------------------------------------------------------ 튐 봉 정리 뒤 32칸 다시(impl_a 함수 그대로, 자료만 고침)
clean = copy.deepcopy(data)
ts_bad = pd.Timestamp("2026-10-21 10:59", tz="America/New_York")
bad_row = clean["SOXS"].loc[ts_bad].copy()
clean["SOXS"].loc[ts_bad, "High"] = max(bad_row["Open"], bad_row["Close"], 45.45)   # 앞 봉 고가 수준
cg = {t: A.regular_grid(clean[t], days) for t in T}
ctab = {t: A.trade_table(cg[t]) for t in T}
otab = {t: A.trade_table(grids[t]) for t in T}
changed_cells = []
for b_ in A.BASES:
    rs = {t: A.rsi_series(clean[t], b_) for t in T}
    for w, nw in A.WINDOWS.items():
        sg = {t: A.signals_for(rs[t], days, nw) for t in T}
        for Tn in A.SIGNALS:
            for e_, k in A.FILLS.items():
                key = f"{Tn}|{b_}|{e_}|{w}"
                sbx = {t: sg[t][Tn] for t in T}
                for mode in ["fade", "follow"]:
                    new, _, _ = A.run_trades(sbx, ctab, cg, k, mode)
                    old, _, _ = A.run_trades(sbx, otab, grids, k, mode)
                    if len(new) != len(old) or not np.allclose(new["pnl"].to_numpy(float), old["pnl"].to_numpy(float)):
                        changed_cells.append({"cell": key, "mode": mode, "old_ev": r6(old["pnl"].mean()),
                                              "new_ev": r6(new["pnl"].mean()), "old_hit": r6(old["hit"].mean()),
                                              "new_hit": r6(new["hit"].mean())})
b0_change = {}
for e_, k in A.FILLS.items():
    for w, nw in A.WINDOWS.items():
        o = A.b0_trades(otab, k, nw, ND)
        n = A.b0_trades(ctab, k, nw, ND)
        b0_change[f"{e_}|{w}"] = {"old_hit": r6(o["hit"].mean()), "new_hit": r6(n["hit"].mean()),
                                  "n_changed": int((o["pnl"].to_numpy() != n["pnl"].to_numpy()).sum())}

# ------------------------------------------------------------ 스프레드 대략 추정(1분봉만, 웹 검색 없음)
spread = {}
for t in T:
    d = data[t]
    raw = pd.read_parquet(os.path.join(ROOT, "cache", "1m", f"{t}.parquet"))
    raw.index = raw.index.tz_convert("America/New_York")
    mod = A.minute_of_day(raw.index)
    out = {}
    for nm, lo, hi in [("09:30-09:34", 570, 574), ("09:35-10:59", 575, 659), ("11:00-15:59", 660, 959)]:
        x = raw[(mod >= lo) & (mod <= hi)]
        rng_c = (x["High"] - x["Low"])
        rng_p = rng_c / x["Close"] * 100
        # Abdi-Ranaldo: s^2 = 4 E[(c_t − η_t)(c_t − η_{t+1})], 날 안에서, 로그
        ars, rolls = [], []
        for _, g in x.groupby(x.index.date):
            c = np.log(g["Close"].to_numpy())
            eta = (np.log(g["High"].to_numpy()) + np.log(g["Low"].to_numpy())) / 2
            if len(c) >= 3:
                v = 4 * np.mean((c[:-1] - eta[:-1]) * (c[:-1] - eta[1:]))
                ars.append(np.sqrt(max(v, 0)) * 100)
                dc = np.diff(c)
                if len(dc) >= 3:
                    cv = np.cov(dc[1:], dc[:-1])[0, 1]
                    rolls.append(2 * np.sqrt(-cv) * 100 if cv < 0 else 0.0)
        px = x["Close"].median()
        out[nm] = {"median_px": r6(px), "range_median_cents": r6(rng_c.median() * 100), "range_p10_cents": r6(rng_c.quantile(0.1) * 100),
                   "range_median_pct": r6(rng_p.median()), "abdi_ranaldo_median_pct": r6(np.median(ars)),
                   "roll_median_pct": r6(np.median(rolls)),
                   "one_tick_pct_of_price": r6(0.01 / px * 100),
                   "median_dollar_vol_per_min_musd": r6((x["Volume"] * x["Close"]).median() / 1e6),
                   "share_vol_median": int(x["Volume"].median())}
    spread[t] = out

# ------------------------------------------------------------ 주문 가능 시간
ev = pd.read_csv(os.path.join(ROOT, "results", "events_a.csv"))
ev["last_bar_end"] = pd.to_datetime(ev["day"] + " " + ev["exit_time"]) + pd.Timedelta(minutes=1)
ev["time_exit"] = ev["hit"] == 0
summer_cut = "11:00"
winter_cut = "10:00"
order_time = {
    "conversion": "서머타임(EDT, UTC−4): KST = ET + 13h → 24:00 KST = 11:00 ET. 표준시(EST, 2026-11-01부터): KST = ET + 14h → 24:00 KST = 10:00 ET(BRIEF의 10:30 ET는 틀림). desk_system.md 표 '한국 시각' 행과 같음",
    "main_cell_time_exits": int(ev["time_exit"].sum()),
    "main_cell_time_exit_after_summer_cut": int((ev["time_exit"] & (ev["last_bar_end"].dt.strftime("%H:%M") > summer_cut)).sum()),
    "main_cell_time_exit_at_summer_cut_exact": ev.loc[ev["time_exit"] & (ev["last_bar_end"].dt.strftime("%H:%M") == summer_cut),
                                                       ["day", "signal_time", "entry_time", "exit_time", "pnl_pct"]].to_dict("records"),
    "main_cell_entries_after_winter_cut": int((ev["entry_time"] >= winter_cut).sum()),
    "main_cell_time_exit_after_winter_cut": int((ev["time_exit"] & (ev["last_bar_end"].dt.strftime("%H:%M") > winter_cut)).sum()),
    "main_cell_hits_done_by_winter_cut": int(((ev["hit"] == 1) & (ev["last_bar_end"].dt.strftime("%H:%M") <= winter_cut)).sum()),
    "W60_E2_signal_minutes_whose_hold_ends_after_11:00": "s = 10:29 (마지막 봉 11:00 시작, 11:01 끝); s = 10:28은 11:00:00 정각 끝",
    "winter_any_window_feasible_30min_time_exit": "없음: E1·E2 진입이 09:31 이후라 30봉째 종가가 10:01 이후 → 표준시 24:00 KST(10:00 ET) 뒤 시장가 매도 필요",
}
# 0930 갭 신호 비중(R 기준: 9:30 봉 변화 = 전날 15:59 종가 → 9:30 종가, 밤사이 갭 포함)
ev["is0930"] = ev["signal_time"] == "09:30"
gap_split = {"n_0930": int(ev["is0930"].sum()), "ev_0930": r6(ev.loc[ev["is0930"], "pnl_pct"].mean()),
             "hit_0930": r6(ev.loc[ev["is0930"], "hit"].mean()),
             "n_later": int((~ev["is0930"]).sum()), "ev_later": r6(ev.loc[~ev["is0930"], "pnl_pct"].mean()),
             "hit_later": r6(ev.loc[~ev["is0930"], "hit"].mean())}
# B0 같은 시각(s = 09:30 → 09:32 진입) 대조
tabs0 = {t: table(grids[t]) for t in T}
b0930 = np.concatenate([tabs0[t]["pnl"][:, 2] for t in T])
b0930h = np.concatenate([tabs0[t]["hit"][:, 2] for t in T])
gap_split["b0_0932_entry_ev"] = r6(b0930.mean())
gap_split["b0_0932_entry_hit"] = r6(b0930h.mean())

out = {"variants_main_cell": res, "entry_delay": delay, "breakeven_cost_per_side_pct": be,
       "mirror_check_hits": mir.to_dict("records"),
       "mirror_check_summary": {"hits": len(mir), "hit_bar_mirror_through": int(mir["mirror_through"].sum()),
                                "any_bar_mirror_through": int(mir["mirror_through_any_bar_to_hit"].sum())},
       "mirror_mismatch_bars_0930_1100_gt_0.6pct": mm,
       "spike_clean_rerun": {"bar": "SOXS 2026-10-21 10:59 고가 47.10 → 45.45", "changed_fade_follow_cells": changed_cells,
                             "b0_change": b0_change},
       "spread_proxy": spread, "order_time": order_time, "gap_0930_split_main": gap_split}
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)

pd.set_option("display.width", 250)
print(pd.DataFrame(res).to_string())
print(pd.DataFrame(delay).to_string())
print("breakeven per-side pct", be)
print(mir.to_string())
print("mirror mismatches", len(mm))
for x in mm[:30]:
    print("  ", x)
print("spike clean", json.dumps(out["spike_clean_rerun"], ensure_ascii=False))
print(json.dumps(spread, ensure_ascii=False, indent=1))
print(json.dumps(order_time, ensure_ascii=False, indent=1, default=str))
print(gap_split)
