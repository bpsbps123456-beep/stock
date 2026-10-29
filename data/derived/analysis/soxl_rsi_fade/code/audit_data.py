"""감사 1: 야후 1분봉 품질(장초 09:30~11:00 빠진 봉·0 거래량·한 봉 튐·1분→5분 묶음 대조)과
주 칸 매매(events_a.csv) 도달 봉이 튀는 고가였는지 점검. impl_a.py를 쓰지 않고 원자료에서 직접 잰다.
사용: py code/audit_data.py  → results/audit_data.json
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(ROOT, "results", "audit_data.json")
T = ["SOXL", "SOXS"]


def load(sym, iv):
    d = pd.read_parquet(os.path.join(ROOT, "cache", iv, f"{sym}.parquet"))
    d.index = d.index.tz_convert("America/New_York")
    return d.sort_index()


def mod(idx):
    return idx.hour * 60 + idx.minute


def r6(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 6)


out = {}
one = {t: load(t, "1m") for t in T}
five = {t: load(t, "5m") for t in T}
days = sorted(set(one["SOXL"].index[(mod(one["SOXL"].index) >= 570) & (mod(one["SOXL"].index) <= 959)].date))
out["n_days"] = len(days)

# ---- 1) 빠진 봉 · 0 거래량 · 같은 가격 채움 봉 (09:30~10:59 90봉, 11:00 봉 포함 91봉)
miss = {}
for t in T:
    d = one[t]
    rows = []
    for day in days:
        x = d[(d.index.date == day) & (mod(d.index) >= 570) & (mod(d.index) <= 660)]
        have = set(mod(x.index))
        missing = [f"{m // 60:02d}:{m % 60:02d}" for m in range(570, 661) if m not in have]
        zero_vol = int((x["Volume"] == 0).sum())
        flat = int(((x["High"] == x["Low"])).sum())
        rows.append({"day": day.isoformat(), "bars": len(x), "missing": missing, "zero_vol": zero_vol,
                     "flat_bars": flat})
    reg = d[(mod(d.index) >= 570) & (mod(d.index) <= 959)]
    full = {day: int((reg.index.date == day).sum()) for day in days}
    miss[t] = {"open_90m_missing_total": sum(len(r["missing"]) for r in rows),
               "open_90m_zero_vol_total": sum(r["zero_vol"] for r in rows),
               "open_90m_flat_total": sum(r["flat_bars"] for r in rows),
               "regular_bar_count_min": min(full.values()), "regular_bar_count_max": max(full.values()),
               "days_short_of_390": {k.isoformat(): v for k, v in full.items() if v != 390},
               "per_day_with_issue": [r for r in rows if r["missing"] or r["zero_vol"] or r["flat_bars"]]}
out["missing"] = miss

# ---- 2) 한 봉 튐: 고가가 앞뒤 봉 고가·자기 몸통보다 크게 튀었다가 바로 돌아온 봉(장초 09:30~11:00)
spikes = {}
for t in T:
    d = one[t]
    rows = []
    for day in days:
        x = d[(d.index.date == day) & (mod(d.index) >= 570) & (mod(d.index) <= 959)].copy()
        rng = (x["High"] - x["Low"])
        med = rng.rolling(15, center=True, min_periods=5).median()
        body_hi = np.maximum(x["Open"], x["Close"])
        body_lo = np.minimum(x["Open"], x["Close"])
        nb_hi = np.maximum(x["High"].shift(1), x["High"].shift(-1))
        nb_lo = np.minimum(x["Low"].shift(1), x["Low"].shift(-1))
        up = (x["High"] - np.maximum(body_hi, nb_hi)) / med
        dn = (np.minimum(body_lo, nb_lo) - x["Low"]) / med
        x["up_z"], x["dn_z"] = up, dn
        w = x[(mod(x.index) <= 660)]
        for ts, r in w.iterrows():
            if r["up_z"] >= 3 or r["dn_z"] >= 3:
                rows.append({"ts": ts.strftime("%Y-%m-%d %H:%M"), "O": r["Open"], "H": r["High"], "L": r["Low"],
                             "C": r["Close"], "V": int(r["Volume"]), "up_z": r6(r["up_z"]), "dn_z": r6(r["dn_z"])})
    spikes[t] = {"rule": "튐 = (고가 − max(몸통, 앞뒤 봉 고가)) ÷ 앞뒤 15봉 고저폭 중앙 ≥ 3 (저가 쪽 대칭), 09:30~11:00",
                 "n": len(rows), "bars": rows}
out["spikes"] = spikes

# ---- 3) 1분 → 5분 묶음과 cache/5m 대조 (정규장 09:30~10:55 봉 시작, 겹치는 21일)
agg_cmp = {}
for t in T:
    d1 = one[t][(mod(one[t].index) >= 570) & (mod(one[t].index) <= 959)]
    g = d1.groupby(d1.index.floor("5min")).agg(Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"),
                                               Close=("Close", "last"), n=("Close", "size"))
    d5 = five[t]
    j = g.join(d5[["Open", "High", "Low", "Close"]], rsuffix="_5", how="inner")
    jw = j[(mod(j.index) >= 570) & (mod(j.index) <= 655)]
    res = {}
    for name, jj in [("open_0930_1055", jw), ("regular_all", j)]:
        r = {"n_bars": len(jj)}
        for c in ["Open", "High", "Low", "Close"]:
            diff = (jj[c] - jj[c + "_5"]).abs()
            rel = diff / jj[c + "_5"] * 100
            r[c] = {"share_gt_1c": r6((diff > 0.0101).mean()), "max_abs": r6(diff.max()),
                    "max_rel_pct": r6(rel.max()), "p99_abs": r6(diff.quantile(0.99))}
        r["1m_high_above_5m_high_gt1c"] = int(((jj["High"] - jj["High_5"]) > 0.0101).sum())
        r["1m_low_below_5m_low_gt1c"] = int(((jj["Low_5"] - jj["Low"]) > 0.0101).sum())
        r["incomplete_5m_groups"] = int((jj["n"] < 5).sum())
        res[name] = r
    worst = jw.assign(dh=(jw["High"] - jw["High_5"]), dl=(jw["Low"] - jw["Low_5"]), dc=(jw["Close"] - jw["Close_5"]))
    worst = worst.reindex(worst[["dh", "dl", "dc"]].abs().max(1).sort_values(ascending=False).index).head(8)
    res["worst_open"] = [{"ts": i.strftime("%Y-%m-%d %H:%M"), "H1": r["High"], "H5": r["High_5"], "L1": r["Low"],
                          "L5": r["Low_5"], "C1": r["Close"], "C5": r["Close_5"]} for i, r in worst.iterrows()]
    agg_cmp[t] = res
out["agg_1m_vs_5m"] = agg_cmp

# ---- 4) 주 칸 도달 매매: 도달 봉 고가가 튀었나, 목표를 얼마나 넘었나, 종가·다음 봉도 목표 위였나
ev = pd.read_csv(os.path.join(ROOT, "results", "events_a.csv"))
hits = []
for _, r in ev[ev["hit"] == 1].iterrows():
    d = one[r["buy"]]
    ts = pd.Timestamp(f"{r['day']} {r['exit_time']}").tz_localize("America/New_York")
    bar = d.loc[ts]
    P = r["entry_px"]
    tgt = P * 1.01
    x = d[(d.index.date == ts.date()) & (mod(d.index) >= 570) & (mod(d.index) <= 959)]
    k = x.index.get_loc(ts)
    nb_hi = max(x["High"].iloc[k - 1], x["High"].iloc[k + 1])
    nxt = x.iloc[k + 1]
    med = (x["High"] - x["Low"]).iloc[max(0, k - 7):k + 8].median()
    # 5분봉 고가로도 목표 관통인가
    t5 = ts.floor("5min")
    h5 = five[r["buy"]].loc[t5, "High"] if t5 in five[r["buy"]].index else np.nan
    # 목표를 몇 틱 넘었나, 몇 봉이 목표 이상에서 거래됐나(진입~보유 끝)
    e_ts = pd.Timestamp(f"{r['day']} {r['entry_time']}").tz_localize("America/New_York")
    hold = x[(x.index >= e_ts) & (x.index <= e_ts + pd.Timedelta(minutes=29))]
    n_bars_at_tgt = int((hold["High"] >= tgt + 0.01).sum())
    close_at_tgt = bool((hold["Close"] >= tgt).any())
    hits.append({"day": r["day"], "buy": r["buy"], "entry": r["entry_time"], "hit_bar": r["exit_time"],
                 "P": P, "target": round(tgt, 4), "hit_bar_H": bar["High"], "hit_bar_C": bar["Close"],
                 "hit_bar_V": int(bar["Volume"]),
                 "through_ticks": int(round((bar["High"] - tgt) / 0.01)),
                 "through_pct": r6((bar["High"] / tgt - 1) * 100),
                 "spike_z": r6((bar["High"] - max(bar["Open"], bar["Close"], nb_hi)) / med),
                 "next_bar_H": nxt["High"], "bars_in_hold_through_target": n_bars_at_tgt,
                 "any_close_ge_target": close_at_tgt, "same_bar_as_entry": r["entry_time"] == r["exit_time"],
                 "five_min_high": h5, "five_min_high_through": bool(h5 >= tgt + 0.01)})
out["hit_bars"] = hits
hb = pd.DataFrame(hits)
out["hit_bar_summary"] = {
    "n_hits": len(hb),
    "through_ticks_min_median_max": [int(hb["through_ticks"].min()), float(hb["through_ticks"].median()),
                                     int(hb["through_ticks"].max())],
    "hits_with_spike_z_ge_3": int((hb["spike_z"] >= 3).sum()),
    "hits_only_1_bar_through_target": int((hb["bars_in_hold_through_target"] == 1).sum()),
    "hits_without_any_close_ge_target": int((~hb["any_close_ge_target"]).sum()),
    "hits_in_entry_bar": int(hb["same_bar_as_entry"].sum()),
    "hits_not_through_on_5m": int((~hb["five_min_high_through"]).sum()),
}

# ---- 5) 분할·병합·날 사이 큰 틈(기준 R RSI가 날을 넘어 이어지므로)
gaps = {}
for t in T:
    reg = one[t][(mod(one[t].index) >= 570) & (mod(one[t].index) <= 959)]
    last = reg.groupby(reg.index.date)["Close"].last()
    first = reg.groupby(reg.index.date)["Open"].first()
    g = (first.shift(-1) / last - 1) * 100
    gaps[t] = {k.isoformat(): r6(v) for k, v in g.dropna().items()}
both = pd.DataFrame(gaps)
out["overnight_gap_pct"] = {"by_day_next_open_vs_close": gaps,
                            "soxl_plus_soxs_gap_corr": r6(both["SOXL"].corr(both["SOXS"])),
                            "max_abs_gap": {t: r6(both[t].abs().max()) for t in T}}

with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)

print(json.dumps({k: v for k, v in out["missing"].items()}, ensure_ascii=False, default=str)[:3000])
for t in T:
    print(t, "spikes", out["spikes"][t]["n"])
    for b in out["spikes"][t]["bars"][:40]:
        print("  ", b)
print(json.dumps(out["agg_1m_vs_5m"], ensure_ascii=False, default=str, indent=0)[:5000])
print(hb.to_string())
print(out["hit_bar_summary"])
print(json.dumps(out["overnight_gap_pct"], default=str)[:2000])
