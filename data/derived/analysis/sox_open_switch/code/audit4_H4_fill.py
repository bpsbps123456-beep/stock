"""BRIEF4 C3 H4 감사 — 틱 단위 실제 체결 재현과 차트 출처 민감도.

A) 2026 사용자 방향 x = 3 사건 중 봉 윗꼬리 0.3~0.5(문턱 아래)인 사건의 신호 봉을 모든 체결(100주 미만 포함)로 다시 만들어,
   '모든 체결' 차트에서는 H4에 새로 들어오는 사건이 있는지 본다(들어오면 그 사건의 r_w90도 적는다).
B) H4 25개(2026 9 + 2023~2025 16)를 틱으로 다시 체결: 진입 = 체결 봉 끝(다음 봉 시작 직전) SOXS 매도 호가(ask),
   목표 = ask × 1.01을 센트 위로 올린 지정가(1달러 이상 주식은 센트 단위만 주문 가능),
   채워짐 = 진입 뒤 목표가보다 높은 체결(관통, 대기열 무관)이 10:59 봉 끝 전에 나옴 / 낙관판 = 목표가 이상 체결(닿음).
   못 채우면 10:59 봉 끝 직전 매수 호가(bid)에 판다.

사용: cd code && py audit4_H4_fill.py → results/audit4_H4_fill.json
"""
import json, math, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from audit4_H4_ticks import get, ts, wick_of, TR, QT, keys, Pace  # noqa: E402

RES = os.path.normpath(os.path.join(HERE, "..", "results"))
CUT = 420                                     # 11:00 ET 색인


def last_quote(sym, day, i_end, head, pace):
    """색인 i_end 분 시작 직전 5초의 마지막 NBBO."""
    q = get(QT.format(sym=sym), {"start": ts(day, i_end - 1, 55), "end": ts(day, i_end), "limit": 10000, "feed": "sip", "sort": "asc"}, head, pace)
    dq = pd.DataFrame(q)
    if dq.empty:
        return None
    dq = dq[(dq["ap"] > 0) & (dq["bp"] > 0) & (dq["ap"] >= dq["bp"])]
    r = dq.iloc[-1]
    return float(r["ap"]), float(r["bp"])


def tick_trade(e, head, pace):
    day = e["day"].strftime("%Y-%m-%d")
    fill = int(e["fill_i"])
    q0 = last_quote("SOXS", day, fill + 1, head, pace)
    if q0 is None:
        return None
    ask, bid = q0
    tgt = ask * 1.01
    lim = math.ceil(round(tgt * 100, 6)) / 100 if tgt >= 1 else math.ceil(round(tgt * 10000, 6)) / 10000
    t = get(TR.format(sym="SOXS"), {"start": ts(day, fill + 1), "end": ts(day, CUT), "limit": 10000, "feed": "sip", "sort": "asc"}, head, pace)
    df = pd.DataFrame(t)
    out = {"day": day, "ask": ask, "bid": bid, "half_spread_pct": (ask - bid) / ((ask + bid) / 2) / 2 * 100, "limit": lim,
           "limit_vs_ask_pct": (lim / ask - 1) * 100, "r_w90_engine": float(e["r_w90"]), "engine_hit": int(e["hit"])}
    if df.empty:
        return out
    df["c"] = df["c"].apply(lambda v: v or [])
    df = df.sort_values("t")
    rl = df[(df["s"] >= 100) & ~df["c"].apply(lambda cs: "I" in cs)]
    thr = rl[rl["p"] > lim + 1e-9]
    tch = df[df["p"] >= lim - 1e-9]
    out["max_px"] = float(df["p"].max())
    out["first_touch"] = tch["t"].iloc[0][11:19] if len(tch) else None
    out["first_through_roundlot"] = thr["t"].iloc[0][11:19] if len(thr) else None
    out["shares_at_limit"] = int(df.loc[(df["p"] - lim).abs() < 1e-9, "s"].sum())
    out["shares_through"] = int(thr["s"].sum())
    qe = last_quote("SOXS", day, CUT, head, pace)
    bid_end = qe[1] if qe else float(df["p"].iloc[-1])
    out["bid_10_59_end"] = bid_end
    r_cut = (bid_end / ask - 1) * 100
    out["r_through"] = (lim / ask - 1) * 100 if len(thr) else r_cut
    out["r_touch"] = (lim / ask - 1) * 100 if len(tch) else r_cut
    return out


def main():
    ev = pd.read_parquet(os.path.join(RES, "ev4_events.parquet"))
    u = ev[(ev["side"] == "user") & np.isclose(ev["x"], 0.03)]
    head, pace = keys(), Pace(3.0)
    res = {}
    # A) 문턱 아래 사건의 모든 체결 윗꼬리
    cand = u[(u["year"] == 2026) & (u["c10_wick"] >= 0.25) & (u["c10_wick"] < 0.5 - 1e-9)].sort_values("day")
    rows = []
    for _, e in cand.iterrows():
        day, dec = e["day"].strftime("%Y-%m-%d"), int(e["dec_i"])
        df = pd.DataFrame(get(TR.format(sym="SOXL"), {"start": ts(day, dec), "end": ts(day, dec + 1), "limit": 10000, "feed": "sip", "sort": "asc"}, head, pace))
        if df.empty:
            continue
        df["c"] = df["c"].apply(lambda v: v or [])
        df = df.sort_values("t")
        w_all = wick_of(df)["wick"]
        rows.append({"day": day, "dec_close": e["dec_close"], "bar_wick": round(float(e["c10_wick"]), 4), "all_prints_wick": round(float(w_all), 4),
                     "flips_in": bool(w_all >= 0.5 - 1e-9), "r_w90": round(float(e["r_w90"]), 3), "hit": int(e["hit"])})
        print("A", rows[-1], flush=True)
    res["A_flip_in_2026"] = rows
    # B) 틱 체결
    h = u[u["c10_wick"] >= 0.5 - 1e-9].sort_values("day")
    tb = []
    for _, e in h.iterrows():
        r = tick_trade(e, head, pace)
        if r:
            r["year"] = int(e["year"])
            tb.append(r)
            print("B", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    res["B_tick_trades"] = tb
    s = pd.DataFrame(tb)
    summ = {}
    for y, g in s.groupby("year"):
        summ[int(y)] = {"n": int(len(g)), "engine_mean": float(g["r_w90_engine"].mean()), "tick_through_mean": float(g["r_through"].mean()),
                        "tick_touch_mean": float(g["r_touch"].mean()), "through_hits": int((g["first_through_roundlot"].notna()).sum())}
    res["B_summary_by_year"] = summ
    with open(os.path.join(RES, "audit4_H4_fill.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v))
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
