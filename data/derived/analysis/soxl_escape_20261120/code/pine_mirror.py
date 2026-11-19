"""soxl_escape.pine 의 두 신호를 파인과 같은 봉 단위 계산으로 다시 짠 미러 + 최종 성적표 (2026-11-20).

탈출(1분봉, exit_fail 갈래 E_volfade_1m_EMA10_thr0.7):
  armed_t = close_t >= lowest(low,60)_t*1.015 and sma(vol,10)_t/sma(vol,40)_t < 0.7
  sig_t   = close_t < ema10_t and close_{t-1} >= ema10_{t-1} and armed_{t-1}, 체결 봉(time_close)이 정규장
바닥(3분봉, bot_dev 갈래 '3m 장중고점 −4% 구역 진입'):
  dayHigh = 04:00 ET부터 3분봉 고가 누적 최댓값, zone = close <= dayHigh*0.96, entry = zone and not zone[1]
  gate = close <= highest(high,40)*0.97, 체결 봉(time_close)이 정규장, 같은 날 30분 안 재신호 억제
확인: ① 미러 신호 = 갈래 정의(harness 경로) 신호 ② 최종 성적(수정된 harness, 11-19 정규장 완결 자료) TRAIN·TEST·PRE
결과: results/pine_mirror.json
"""
import json, os
import numpy as np, pandas as pd
import harness as H

OUT = os.path.join(H.ROOT, "results", "pine_mirror.json")
RNG = {"TRAIN": H.TRAIN, "TEST": H.TEST, "PRE": H.PRE}
d1 = H.load_1m()
b1, b3 = H.resample(1), H.resample(3)
t1 = d1.index.values


def reg_end(end_ts) -> bool:
    m = end_ts.hour * 60 + end_ts.minute
    return H.REG_OPEN <= m < H.REG_CLOSE


# ── 탈출 미러(봉 단위) ──
def exit_mirror():
    C, L, V = b1["Close"].to_numpy(float), b1["Low"].to_numpy(float), b1["Volume"].to_numpy(float)
    e10 = H.ema(b1["Close"], 10).to_numpy()
    lo60 = b1["Low"].rolling(60, min_periods=1).min().to_numpy()
    s10 = b1["Volume"].rolling(10).mean().to_numpy()
    s40 = b1["Volume"].rolling(40).mean().to_numpy()
    ends = b1["end"]
    out = []
    armed_prev = False
    for t in range(len(C)):
        with np.errstate(invalid="ignore", divide="ignore"):
            armed = bool(C[t] >= lo60[t] * 1.015 and (s10[t] / s40[t]) < 0.7)
        if t > 0 and armed_prev and C[t] < e10[t] and C[t - 1] >= e10[t - 1] and reg_end(ends.iloc[t]):
            out.append(ends.iloc[t])
        armed_prev = armed
    return out


# ── 바닥 미러(봉 단위, 같은 날 30분 억제 포함) ──
def bottom_mirror():
    C, Hh = b3["Close"].to_numpy(float), b3["High"].to_numpy(float)
    hi40 = b3["High"].rolling(40, min_periods=1).max().to_numpy()
    dates = b3["date"].values
    ends = b3["end"]
    out, day_hi, zone_prev, last = [], np.nan, False, None
    for t in range(len(C)):
        new_day = t == 0 or dates[t] != dates[t - 1]
        day_hi = Hh[t] if (new_day or np.isnan(day_hi)) else max(day_hi, Hh[t])
        zone = C[t] <= day_hi * 0.96
        entry = zone and not zone_prev
        gate = C[t] <= hi40[t] * 0.97
        e = ends.iloc[t]
        if entry and gate and reg_end(e):
            if last is None or last.date() != e.date() or (e - last) >= pd.Timedelta(minutes=30):
                out.append(e)
                last = e
        zone_prev = zone
    return out


# ── 갈래 정의(harness 경로) ──
def exit_branch():
    e10 = H.ema(b1["Close"], 10)
    rally = H.ctx_rally(b1, 60, 0.015)
    vr = (b1.Volume.rolling(10).mean() / b1.Volume.rolling(40).mean()).shift(1)
    s = (b1.Close < e10) & (b1.Close.shift(1) >= e10.shift(1)) & rally.shift(1).fillna(False).astype(bool) & (vr < 0.7)
    pos = H.exec_index(s, b1)
    pos = pos[pos >= 0]
    return pos[d1["reg"].values[pos]]


def bottom_branch():
    hi = b3.groupby("date")["High"].cummax()
    z = b3["Close"] <= hi * 0.96
    s = z & ~z.shift(1, fill_value=False)
    dec = H.to_1m_mask(H.ctx_decline(b3, 40, 0.03), b3)
    pos = H.exec_index(s, b3)
    pos = pos[pos >= 0]
    pos = pos[d1["reg"].values[pos] & dec[pos]]
    return H.dedup(pos, 30)


def to_pos(times):
    return np.searchsorted(t1, np.array([pd.Timestamp(x).to_datetime64() for x in times], dtype="datetime64[ns]"))


res = {}
em, eb = to_pos(exit_mirror()), exit_branch()
bm, bb = to_pos(bottom_mirror()), bottom_branch()
res["parity"] = {"exit_mirror_n": int(len(em)), "exit_branch_n": int(len(eb)), "exit_same": bool(np.array_equal(np.sort(em), np.sort(eb))),
                 "bottom_mirror_n": int(len(bm)), "bottom_branch_n": int(len(bb)), "bottom_same": bool(np.array_equal(np.sort(bm), np.sort(bb)))}
print(res["parity"], flush=True)

DECLINE = H.to_1m_mask(H.ctx_decline(b3, 40, 0.03), b3)
RALLY = H.to_1m_mask(H.ctx_rally(b3, 20, 0.015), b3)
res["exit"], res["bottom"] = {}, {}
for nm, rng in RNG.items():
    hs = H.hold_sim(eb, rng)
    hr = H.hold_sim(eb, rng, entry_mask=RALLY)
    sm = H.summarize(H.event_metrics(H.dedup(eb, 30)), rng, "exit", RALLY)
    res["exit"][nm] = {"hold_all": hs, "hold_rally": hr, "event": {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in sm.items()}}
    sb = H.summarize(H.event_metrics(bb), rng, "bottom", DECLINE)
    ev = H.event_metrics(bb)
    ev = ev[H.in_range(ev["date"].values, rng)]
    res["bottom"][nm] = {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in sb.items()}
    res["bottom"][nm]["r60_by_day%"] = {str(k)[:10]: round(100 * float(v), 2) for k, v in ev.groupby("date")["r60"].mean().items()}
    print(nm, "EXIT x_rand", hs["x_rand_mean%"], hs["x_rand_t_days"], "cvar", hs["rule_cvar5%"], hs["rand_cvar5%"],
          "| BOTTOM n", sb.get("n"), "r60_x_ctx", sb.get("r60_x_ctx%"), "mae60", sb.get("mae60_med%"), flush=True)

# 11-19(사용자 스크린샷 날) 신호
day = np.datetime64("2026-11-19")
res["nov19"] = {
    "exit": [str(d1.index[p])[11:16] + f" ET @{d1['Open'].values[p]:.2f}" for p in eb if d1["date"].values[p] == day],
    "bottom": [str(d1.index[p])[11:16] + f" ET @{d1['Open'].values[p]:.2f}" for p in bb if d1["date"].values[p] == day],
}
print("11-19", res["nov19"])
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print("saved", OUT)
