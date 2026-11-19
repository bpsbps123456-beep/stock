"""오케스트레이터 마무리 점검(2026-11-20, harness 버그 2개 수정 뒤).
1) 사용자 차트 선(EMA 10·25·60, 1·3분봉)의 종가 이탈 손절: 무작위 탈출 대비 + 같은 날 같은 개수 무작위 사건 귀무 보정, TRAIN·TEST·PRE.
2) '장 초반 칼날': 바닥 신호 5종을 개장 20분(09:30~09:50 ET) 안/밖으로 나눠 같은 시간대 하락 상황 대조와 비교, TRAIN·TEST·PRE.
3) 사후 관찰 후보(3분 VWAP−1σ 진입, 1분 강세 다이버전스 R5)의 세 구간 성적 — TEST를 본 뒤 꺼낸 것이라 선택 근거가 아니다.
결과: results/final_check.json
"""
import json, os
import numpy as np, pandas as pd
import harness as H

OUT = os.path.join(H.ROOT, "results", "final_check.json")
RNG = {"TRAIN": H.TRAIN, "TEST": H.TEST, "PRE": H.PRE}
d1 = H.load_1m()
b1, b3 = H.resample(1), H.resample(3)
DECLINE = H.to_1m_mask(H.ctx_decline(b3, 40, 0.03), b3)
f = H._fwd_tables()


def null_adj(exit_pos, rng, draws=30):
    """같은 날 같은 개수의 무작위 정규장 봉을 탈출 사건으로 넣은 hold_sim x_rand 평균(드문 규칙의 기저 편향)."""
    ev = np.asarray(exit_pos)
    ev = ev[ev >= 0]
    dates = d1["date"].values
    regpos = np.nonzero(d1["reg"].to_numpy() & H.in_range(dates, rng))[0]
    cnt = pd.Series(dates[ev]).value_counts()
    rs = np.random.default_rng(11)
    vals = []
    for _ in range(draws):
        pick = []
        for dt, c in cnt.items():
            pool = regpos[dates[regpos] == np.datetime64(dt)]
            if len(pool):
                pick.append(rs.choice(pool, size=min(c, len(pool)), replace=False))
        if pick:
            vals.append(H.hold_sim(np.concatenate(pick), rng, n_perm=5)["x_rand_mean%"])
    return float(np.mean(vals)) if vals else None


res = {"exit_user_lines": {}, "opening_knife": {}, "post_hoc": {}}

# 1) 사용자 차트 선 이탈(상태형: 종가가 선 아래인 동안)
for tf, bars in (("1m", b1), ("3m", b3)):
    for n in (10, 25, 60):
        e = H.ema(bars["Close"], n)
        state = bars["Close"] < e
        pos = H.exec_index(state, bars)
        cross = state & ~state.shift(1, fill_value=False)
        cpos = H.exec_index(cross, bars)
        row = {}
        for nm, rng in RNG.items():
            hs = H.hold_sim(pos, rng)
            m = H.in_range(d1["date"].values[cpos[cpos >= 0]], rng)
            na = null_adj(cpos[cpos >= 0][m], rng)
            hc = H.hold_sim(cpos, rng)
            row[nm] = {"state_x_rand": hs["x_rand_mean%"], "state_t": hs["x_rand_t_days"],
                       "state_cvar": [hs["rule_cvar5%"], hs["rand_cvar5%"], hs["hold_cvar5%"]],
                       "cross_x_rand": hc["x_rand_mean%"], "cross_null": None if na is None else round(na, 3),
                       "cross_adj": None if na is None else round(hc["x_rand_mean%"] - na, 3)}
        res["exit_user_lines"][f"{tf}_EMA{n}"] = row
        print(tf, n, {k: (v["state_x_rand"], v["cross_adj"]) for k, v in row.items()}, flush=True)

# 2) 바닥 신호 5종 × 개장 20분 안/밖
r1 = H.rsi(b1["Close"], 14)
r3 = H.rsi(b3["Close"], 14)
e60_1 = H.ema(b1["Close"], 60)
vw3 = H.session_vwap(b3)
tp3 = (b3["High"] + b3["Low"] + b3["Close"]) / 3
dev = (tp3 - vw3) ** 2
sd3 = np.sqrt((dev * b3["Volume"]).groupby(b3["date"]).cumsum() / b3["Volume"].groupby(b3["date"]).cumsum().replace(0, np.nan))
low60 = b1["Low"].rolling(60, min_periods=1).min()
dec1 = pd.Series(DECLINE, index=d1.index)
dec3 = H.ctx_decline(b3, 40, 0.03)
sigs = {
    "1m RSI 30 상향 교차(하락 상황)": ((r1 > 30) & (r1.shift(1) <= 30) & dec1, b1),
    "3m RSI 25 상향 교차": ((r3 > 25) & (r3.shift(1) <= 25), b3),
    "1m 60분 저점 +1% 반등(하락 상황)": ((b1["Close"] >= low60 * 1.01) & (b1["Close"].shift(1) < low60.shift(1) * 1.01) & dec1, b1),
    "1m EMA60 −3% 구역 진입": ((b1["Close"] <= e60_1 * 0.97) & (b1["Close"].shift(1) > e60_1.shift(1) * 0.97) & dec1, b1),
    "3m 장중고점 −4% 구역 진입": None,
}
hi3 = b3.groupby("date")["High"].cummax()
z = b3["Close"] <= hi3 * 0.96
sigs["3m 장중고점 −4% 구역 진입"] = (z & ~z.shift(1, fill_value=False) & dec3, b3)
vz = (b3["Close"] <= vw3 - sd3)
sigs["3m VWAP−1σ 진입(사후 관찰)"] = (vz & ~vz.shift(1, fill_value=False) & dec3, b3)

mod = d1["mod"].to_numpy()
early_mask = (mod >= H.REG_OPEN) & (mod < H.REG_OPEN + 20)
for name, (sig, bars) in sigs.items():
    pos = H.dedup(H.exec_index(sig, bars), 30)
    row = {}
    for nm, rng in RNG.items():
        p = pos[H.in_range(d1["date"].values[pos], rng) & d1["reg"].values[pos]]
        cell = {}
        for part, msk in (("early20", early_mask), ("later", ~early_mask)):
            pp = p[msk[p]]
            base = H.baseline_positions(rng, DECLINE & msk)
            v, b = f["r60"][pp], f["r60"][base]
            cell[part] = {"n": int(len(pp)), "r60%": round(100 * np.nanmean(v), 3) if len(pp) else None,
                          "x_ctx_same_time%": round(100 * (np.nanmean(v) - np.nanmean(b)), 3) if len(pp) else None,
                          "mae60_med%": round(100 * np.nanmedian(f["mae60"][pp]), 3) if len(pp) else None}
        allp = H.summarize(H.event_metrics(p), rng, "bottom", DECLINE) if len(p) else {"n": 0}
        cell["all_r60_x_ctx%"] = allp.get("r60_x_ctx%")
        row[nm] = cell
    res["opening_knife"][name] = row
    print(name, {k: (v["early20"]["n"], v["early20"]["x_ctx_same_time%"], v["later"]["n"], v["later"]["x_ctx_same_time%"]) for k, v in row.items()}, flush=True)

# 하락 상황 대조 자체의 시간대 차이(개장 20분이 원래 더 험한가)
res["decline_base_by_time"] = {}
for nm, rng in RNG.items():
    a = H.baseline_positions(rng, DECLINE & early_mask)
    b = H.baseline_positions(rng, DECLINE & ~early_mask)
    res["decline_base_by_time"][nm] = {"early20_n": int(len(a)), "early20_r60%": round(100 * np.nanmean(f["r60"][a]), 3),
                                       "later_n": int(len(b)), "later_r60%": round(100 * np.nanmean(f["r60"][b]), 3),
                                       "early20_mae60_med%": round(100 * np.nanmedian(f["mae60"][a]), 3),
                                       "later_mae60_med%": round(100 * np.nanmedian(f["mae60"][b]), 3)}
print(res["decline_base_by_time"])

json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved", OUT)
