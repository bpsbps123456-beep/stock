"""감사 2: 미래 참조·손 계산 대조. impl_a.py를 import하지 않고 원자료(cache/1m)에서 주 칸(T1|R|E2|W60) 반대 매수를
시각 문자열 조회만으로 다시 만든다(RSI는 반복문, 매매는 봉 하나씩 훑기). events_a.csv 30건 전부와 대조하고,
무작위 10건(seed 7)의 계산 과정을 손 계산 형식으로 남긴다.
사용: py code/audit_recompute.py → results/audit_recompute.json
"""
import json
import os
import random

import numpy as np
import pandas as pd

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(ROOT, "results", "audit_recompute.json")
OTHER = {"SOXL": "SOXS", "SOXS": "SOXL"}
N = 14


def load(sym):
    d = pd.read_parquet(os.path.join(ROOT, "cache", "1m", f"{sym}.parquet"))
    d.index = d.index.tz_convert("America/New_York")
    d = d.sort_index()
    m = d.index.hour * 60 + d.index.minute
    return d[(m >= 570) & (m <= 959)]          # 정규장 09:30~15:59 봉 시작만


def rsi_loop(closes):
    out = [None] * len(closes)
    ag = al = None
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i - 1]
        g, l = max(ch, 0.0), max(-ch, 0.0)
        if i < N:
            continue
        if i == N:
            gs = [max(closes[j] - closes[j - 1], 0.0) for j in range(1, N + 1)]
            ls = [max(closes[j - 1] - closes[j], 0.0) for j in range(1, N + 1)]
            ag, al = sum(gs) / N, sum(ls) / N
        else:
            ag = (ag * (N - 1) + g) / N
            al = (al * (N - 1) + l) / N
        out[i] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return out


reg = {t: load(t) for t in ["SOXL", "SOXS"]}
rsi = {}
for t, d in reg.items():
    r = rsi_loop(d["Close"].tolist())
    rsi[t] = pd.Series(r, index=d.index, dtype=float)

# 신호: 창 W60 = 봉 시작 09:30~10:29, T1 = 앞 봉(같은 정규장 계열 바로 앞, 9:30이면 전날 15:59) ≤ 70 < 이 봉
sigs = []
for t in ["SOXL", "SOXS"]:
    s = rsi[t]
    prev = s.shift(1)
    idx = s.index
    m = idx.hour * 60 + idx.minute
    for i in np.flatnonzero((m >= 570) & (m <= 629)):
        if pd.notna(prev.iloc[i]) and pd.notna(s.iloc[i]) and prev.iloc[i] <= 70 < s.iloc[i]:
            sigs.append((idx[i], t, prev.iloc[i], s.iloc[i], idx[i - 1]))
sigs.sort(key=lambda x: (x[0], x[1]))

# 9:30 봉의 앞 봉이 전날 15:59인지 확인
prev_check = [(str(ts), str(pts)) for ts, t, a, b, pts in sigs if ts.hour == 9 and ts.minute == 30]
assert all(p.endswith("15:59:00-04:00") for _, p in prev_check), prev_check


def simulate(Y, entry_ts):
    d = reg[Y]
    if entry_ts not in d.index:
        return None
    P = d.loc[entry_ts, "Open"] + 0.01
    tgt = P * 1.01
    bars = d.loc[entry_ts: entry_ts + pd.Timedelta(minutes=29)]
    assert len(bars) == 30, (Y, entry_ts, len(bars))
    low = np.inf
    for ts, b in bars.iterrows():
        low = min(low, b["Low"])
        if b["High"] >= tgt + 0.01:
            return dict(P=P, hit=1, exit=ts, pnl=1.0, mae=(low / P - 1) * 100, tgt=tgt, hbar_high=b["High"])
    last = bars.iloc[-1]
    return dict(P=P, hit=0, exit=bars.index[-1], pnl=((last["Close"] - 0.01) / P - 1) * 100,
                mae=(low / P - 1) * 100, tgt=tgt, last_close=last["Close"])


trades, skipped = [], []
open_until = {}   # (날, Y) → 청산 봉 시작
for ts, X, rp, rc, pts in sigs:
    Y = OTHER[X]
    e = ts + pd.Timedelta(minutes=2)
    key = (ts.date(), Y)
    if key in open_until and e < open_until[key] + pd.Timedelta(minutes=1):
        skipped.append((str(ts), X, "overlap"))
        continue
    r = simulate(Y, e)
    if r is None:
        skipped.append((str(ts), X, "nobar"))
        continue
    open_until[key] = r["exit"]
    trades.append(dict(day=ts.date().isoformat(), signal_ticker=X, signal_time=ts.strftime("%H:%M"),
                       rsi_prev=round(rp, 4), rsi=round(rc, 4), prev_bar=str(pts), buy=Y,
                       entry_time=e.strftime("%H:%M"), entry_open=reg[Y].loc[e, "Open"], **{
                           k: (v.strftime("%H:%M") if isinstance(v, pd.Timestamp) else v) for k, v in r.items()}))

mine = pd.DataFrame(trades)
ev = pd.read_csv(os.path.join(ROOT, "results", "events_a.csv"))
cmp = mine.merge(ev, on=["day", "signal_ticker", "signal_time"], how="outer", suffixes=("_me", "_a"), indicator=True)
mism = []
for _, r in cmp.iterrows():
    if r["_merge"] != "both":
        mism.append({"row": f"{r['day']} {r['signal_ticker']} {r['signal_time']}", "only_in": r["_merge"]})
        continue
    bad = []
    if r["buy_me"] != r["buy_a"] or r["entry_time_me"] != r["entry_time_a"]:
        bad.append("buy/entry")
    if abs(r["P"] - r["entry_px"]) > 1e-3:
        bad.append(f"P {r['P']} vs {r['entry_px']}")
    if int(r["hit_me"]) != int(r["hit_a"]) or r["exit"] != r["exit_time"]:
        bad.append("hit/exit")
    if abs(r["pnl"] - r["pnl_pct"]) > 1e-4:
        bad.append(f"pnl {r['pnl']} vs {r['pnl_pct']}")
    if abs(r["mae"] - r["mae_pct"]) > 1e-4:
        bad.append(f"mae {r['mae']} vs {r['mae_pct']}")
    if bad:
        mism.append({"row": f"{r['day']} {r['signal_ticker']} {r['signal_time']}", "diff": bad})

random.seed(7)
pick = sorted(random.sample(range(len(mine)), 10))
hand = mine.iloc[pick].copy()
for c in ["entry_open", "P", "tgt", "mae", "pnl", "hbar_high", "last_close"]:
    if c in hand:
        hand[c] = hand[c].astype(float).round(4)

out = {"n_signals_T1_W60": len(sigs), "n_trades_mine": len(mine), "n_trades_impl_a": len(ev),
       "skipped": skipped, "n_skipped_overlap": sum(1 for s in skipped if s[2] == "overlap"),
       "mismatches_vs_events_a": mism, "all_30_match": len(mism) == 0,
       "prev_bar_of_0930_signals_is_prior_day_1559": True, "n_0930_signals": len(prev_check),
       "hand_check_10": hand.drop(columns=["prev_bar"]).to_dict("records"),
       "mean_pnl_mine": round(float(mine["pnl"].mean()), 6), "hit_rate_mine": round(float(mine["hit"].mean()), 6)}
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
print("signals", len(sigs), "trades", len(mine), "skipped", len(skipped), "mismatch", mism)
print("mean pnl", out["mean_pnl_mine"], "hit", out["hit_rate_mine"])
print(hand.drop(columns=["prev_bar"]).to_string())
