"""설명용 감도: 같은 진입에 −1% 손절을 함께 건 경우(±1% 묶음 주문). impl_a를 쓰지 않고 원자료에서 다시 잰다.
반대 = events_a.csv 30건, 같은 시각 추종 = 같은 30건의 신호 종목을 산 경우, B0 = W60·E2 모든 분 × 두 종목.
손절: 저가 ≤ P×0.99 − 0.01이면 손절, 체결 = min(시가, P×0.99) − 0.01. 같은 봉에 목표·손절이 다 있으면 손절(보수).
둘 다 없으면 30번째 봉 종가 − 0.01. 결과: results/bracket.json"""
import json, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
bars = {s: pd.read_parquet(os.path.join(ROOT, f"cache/1m/{s}.parquet")) for s in ["SOXL", "SOXS"]}
for s, d in bars.items():
    d.index = d.index.tz_localize(None)


def one(sym, entry_ts, stop=True):
    d = bars[sym]
    w = d.loc[entry_ts: entry_ts + pd.Timedelta(minutes=29)]
    if len(w) < 30 or w.index[0] != entry_ts:
        return None
    P = w.Open.iloc[0] + 0.01
    tgt, stp = P * 1.01 + 0.01, P * 0.99 - 0.01
    for _, b in w.iterrows():
        lo_hit, hi_hit = b.Low <= stp, b.High >= tgt
        if stop and lo_hit:
            return (min(b.Open, P * 0.99) - 0.01) / P - 1
        if hi_hit:
            return 0.01
    return (w.Close.iloc[-1] - 0.01) / P - 1


ev = pd.read_csv(os.path.join(ROOT, "results/events_a.csv"))
fade, follow, nostop = [], [], []
for r in ev.itertuples():
    t = pd.Timestamp(f"{r.day} {r.entry_time}")
    fade.append(one(r.buy, t)); follow.append(one(r.signal_ticker, t)); nostop.append(one(r.buy, t, stop=False))
b0 = []
for day in sorted(set(bars["SOXL"].index.date)):
    for m in range(60):
        t = pd.Timestamp(f"{day} 09:30") + pd.Timedelta(minutes=m + 2)
        for s in ["SOXL", "SOXS"]:
            v = one(s, t)
            if v is not None:
                b0.append(v)
mean = lambda x: round(100 * sum(v for v in x if v is not None) / len([v for v in x if v is not None]), 4)
out = {"fade_bracket_ev_pct": mean(fade), "follow_same_time_bracket_ev_pct": mean(follow), "b0_bracket_ev_pct": mean(b0),
       "fade_nostop_ev_pct_check": mean(nostop), "n_fade": len(fade), "n_b0": len(b0)}
json.dump(out, open(os.path.join(ROOT, "results/bracket.json"), "w"), indent=1)
print(out)
