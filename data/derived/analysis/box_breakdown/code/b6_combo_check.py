"""b6_combo 독립 대조: 가족마다 1칸을 하네스 쌓기·evaluate_cells 없이 긴 형식 일봉(cache/daily.parquet)과 저장 사건표로
다시 계산해 b6_combo_cells.csv의 켜짐 수·평균·차이와 맞춘다. 결과는 ../results/b6_combo_check.json.

대조 칸
1) D|N20|stock|ret5_pre≥-0.02387 & pc_equity5_p≥0.8447 & sq_brk=0 — ret5_pre·sq_brk는 일봉에서 직접, pc는 b4 저장 사건표.
2) b2bounce|N20|stock|nvdl_flow20_p≥0.6426 — bounce 진입·청산을 일봉에서 직접(갈래 2 정의), 꺼짐 = N20 종목 즉시 진입.
3) b8|N14|dbox_pos≥0.1937 & sq_brk20_prev=0 — SOXX·QQQ 20일 박스 상태(전 거래일)를 일봉에서 직접.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FRONT, BACK = ("2026-01-02", "2026-05-29"), ("2026-06-01", "2026-09-29")
COST = 0.001

d = pd.read_parquet(ROOT / "cache" / "daily.parquet")
d["date"] = pd.to_datetime(d["date"])
cal = pd.DatetimeIndex(sorted(d.loc[d["sym"] == "QQQ", "date"].unique()))
pos = {t: k for k, t in enumerate(cal)}
W = {f: d.pivot(index="date", columns="sym", values=f).reindex(cal) for f in ("Open", "High", "Low", "Close")}


def brk20(sym):
    low_min = W["Low"][sym].shift(1).rolling(20, min_periods=20).min()
    c = W["Close"][sym]
    return pd.Series(np.where(low_min.isna() | c.isna(), np.nan, (c < low_min).astype(float)), index=cal)


sqb = np.fmax(brk20("SOXX").to_numpy(), brk20("QQQ").to_numpy())
sqb[np.isnan(brk20("SOXX").to_numpy()) | np.isnan(brk20("QQQ").to_numpy())] = np.nan
cells = pd.read_csv(RES / "b6_combo_cells.csv").set_index("cell")
out = {}


def stats(df, on, y):
    r = {}
    for nm, per in (("f", FRONT), ("b", BACK)):
        m = (df["t"] >= per[0]) & (df["t"] <= per[1]) & df[y].notna() & on.notna()
        a, b = df.loc[m & (on == 1), y], df.loc[m & (on == 0), y]
        r[nm] = {"n_on": int(len(a)), "mean_on": float(a.mean()), "diff": float(a.mean() - b.mean())}
    return r


def compare(name, r):
    c = cells.loc[name]
    dif = {f"{p}_{k}": float(abs(r[p][k] - c[f"{p}_{k}"])) for p in ("f", "b") for k in ("n_on", "mean_on", "diff")}
    out[name] = {"recomputed": r, "max_abs_diff": float(max(dif.values())), "detail": dif}


# 1) 일봉 조합
ev = pd.read_parquet(RES / "events_N20.parquet")
ev["t"] = pd.to_datetime(ev["t"])
b4 = pd.read_parquet(RES / "b4_flow_events.parquet")
b4 = b4[b4["N"] == 20][["sym", "t", "pc_equity5_p", "nvdl_flow20_p"]]
b4["t"] = pd.to_datetime(b4["t"])
ev = ev.merge(b4, on=["sym", "t"], how="left")
k = ev["t"].map(pos).to_numpy()
C = W["Close"]
ev["ret5_chk"] = [C[s].iloc[i - 1] / C[s].iloc[i - 6] - 1 for s, i in zip(ev["sym"], k)]
ev["sq_chk"] = sqb[k]
st = ev[ev["kind"] == "stock"].copy()
fr = (st["t"] >= FRONT[0]) & (st["t"] <= FRONT[1])
q_r5 = np.quantile(st.loc[fr & st["ret5_chk"].notna(), "ret5_chk"], 2 / 3)
fr_all = (ev["t"] >= FRONT[0]) & (ev["t"] <= FRONT[1])
q_pc = np.quantile(ev.loc[fr_all & ev["pc_equity5_p"].notna(), "pc_equity5_p"], 2 / 3)
q_nv = np.quantile(ev.loc[fr_all & ev["nvdl_flow20_p"].notna(), "nvdl_flow20_p"], 2 / 3)
x = st[["ret5_chk", "pc_equity5_p", "sq_chk"]]
on = pd.Series(np.where(x.isna().any(axis=1), np.nan,
                        ((st["ret5_chk"] >= q_r5) & (st["pc_equity5_p"] >= q_pc) & (st["sq_chk"] == 0)).astype(float)),
               index=st.index)
compare("D|N20|stock|ret5_pre≥-0.02387 & pc_equity5_p≥0.8447 & sq_brk=0", stats(st, on, "short_ret_h10"))
out["thresholds"] = {"ret5_q2": float(q_r5), "pc_q2": float(q_pc), "nvdl_q2": float(q_nv)}

# 2) 확인 대기 bounce × NVDL (갈래 2 정의를 일봉에서 직접)
rows = []
O, Hh = W["Open"], W["High"]
for r in st.itertuples():
    i, Lv = pos[r.t], r.L
    o, h, c = O[r.sym].to_numpy(), Hh[r.sym].to_numpy(), C[r.sym].to_numpy()
    if i + 11 > len(c):          # t+10 자료가 없는 끝 사건(판정 구간 밖)
        rows.append(np.nan)
        continue
    started, hb, dd = False, np.nan, None
    for j in range(i + 1, i + 6):
        if j >= len(c) or np.isnan(c[j]):
            break
        if c[j] > Lv:
            break
        if not started:
            if c[j] > c[j - 1]:
                started, hb = True, h[j]
        elif c[j] < c[j - 1] and h[j] <= hb and c[j] < Lv:
            dd = j
            break
        else:
            hb = max(hb, h[j])
    y = np.nan
    if dd is not None and dd + 1 < len(o):
        e = dd + 1
        ent = o[e]
        for kk in range(e, i + 10):
            if c[kk] > Lv:
                y = (ent - o[kk + 1]) / ent - COST
                break
        else:
            if i + 10 < len(c):
                y = (ent - c[i + 10]) / ent - COST
    rows.append(y)
st["bounce_y"] = rows
res = {}
for nm, per in (("f", FRONT), ("b", BACK)):
    m = (st["t"] >= per[0]) & (st["t"] <= per[1])
    imm = st.loc[m & st["short_ret_h10"].notna(), "short_ret_h10"]
    a = st.loc[m & st["short_ret_h10"].notna() & st["bounce_y"].notna() & (st["nvdl_flow20_p"] >= q_nv), "bounce_y"]
    res[nm] = {"n_on": int(len(a)), "mean_on": float(a.mean()), "diff": float(a.mean() - imm.mean())}
compare("b2bounce|N20|stock|nvdl_flow20_p≥0.6426", res)

# 3) 60분봉 × 전 거래일 박스 상태
e8 = pd.read_parquet(RES / "b8_events_N14.parquet")
e8["t"] = pd.to_datetime(e8["t"])
j = e8["t"].map(pos).to_numpy()
e8["sq_prev"] = sqb[j - 1]
f8 = (e8["t"] >= FRONT[0]) & (e8["t"] <= FRONT[1]) & e8["dbox_pos"].notna()
q_db = np.quantile(e8.loc[f8, "dbox_pos"], 2 / 3)
on8 = pd.Series(np.where(e8[["dbox_pos", "sq_prev"]].isna().any(axis=1), np.nan,
                         ((e8["dbox_pos"] >= q_db) & (e8["sq_prev"] == 0)).astype(float)), index=e8.index)
compare("b8|N14|dbox_pos≥0.1937 & sq_brk20_prev=0", stats(e8, on8, "short_ret_h10"))
out["thresholds"]["b8_dbox_q2"] = float(q_db)
(RES / "b6_combo_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: (v["max_abs_diff"] if isinstance(v, dict) and "max_abs_diff" in v else v) for k, v in out.items()},
                 ensure_ascii=False, indent=1))
