"""경고 층의 근거와 빠른 선의 9종목 점검 (2026-11-22, grid40 결과를 본 뒤의 후속 점검).

1) 경고 = 느린 선(일봉 EMA10·20 종가)이 보유 중일 때, 정규장 가격이 '전날 마지막 정규장 40분봉의 EMA100'에 처음 닿은 순간.
   - 경고 뒤 앞으로 수익(닿은 가격 → 그날 종가, 1·5거래일 뒤 종가)을 같은 종목·같은 2026 보유일의 시가 기준 앞으로 수익과 비교한다.
   - 경고 뒤 5거래일 안에 느린 선 대피가 실제로 나온 비율(경고가 대피를 미리 알려 주는가)도 센다.
2) 빠른 선(연장 5분봉 EMA200 ±2% 종가, SOXL에서만 검증)의 9종목 성적.
결과: ../results/warn_check.json
"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../soxl_escape_20261120/lines/code"))
import engine as E  # noqa: E402

SYMS = ["SOXL", "MU", "AMD", "ARM", "DELL", "INTC", "SNDK", "STX", "WDC"]
COST = 0.001
Y26 = ("2026-01-02", "2026-11-20")
res = {"warn": {}, "fast": {}}
for sym in SYMS:
    E.use(sym)
    d1 = E.load(sym)
    reg = d1[d1["reg"]]
    # 느린 선 상태(각 날 종가 뒤 상태 → 다음 날 적용)
    bd = E.bars(sym, "D", "reg"); Cd = bd["Close"]
    e10, e20 = E.ema(Cd, 10), E.ema(Cd, 20)
    W, R = np.minimum(e10, e20), np.maximum(e10, e20)
    st, hold_after = True, []
    for j in range(len(bd)):
        if st and Cd.iloc[j] < W.iloc[j]:
            st = False
        elif (not st) and Cd.iloc[j] > R.iloc[j]:
            st = True
        hold_after.append(st)
    hold_after = pd.Series(hold_after, index=pd.to_datetime(bd["date"].values))
    holding_today = hold_after.shift(1)                     # 오늘 장중 보유 여부 = 어제 종가 뒤 상태
    exit_sig = hold_after.shift(1).fillna(True) & ~hold_after   # 오늘 종가로 대피 신호
    # 경고선: 전날 마지막 정규장 40분봉의 EMA100
    b40 = E.bars(sym, 40, "reg")
    e100 = E.ema(b40["Close"], 100)
    last = e100.groupby(b40["date"].values).last()
    lvl = last.shift(1)
    lvl.index = pd.to_datetime(lvl.index)
    dclose = reg.groupby("date")["Close"].last()
    dopen = reg.groupby("date")["Open"].first()
    days = [d for d in dclose.index if Y26[0] <= str(d.date()) <= Y26[1]]
    rows = []
    for k, d in enumerate(dclose.index):
        if not (Y26[0] <= str(d.date()) <= Y26[1]):
            continue
        if not bool(holding_today.get(d, False)) or pd.isna(lvl.get(d, np.nan)):
            continue
        day = reg[reg["date"] == d]
        hit = day[day["Low"] <= lvl[d]]
        pos_k = list(dclose.index).index(d)
        c1 = dclose.iloc[pos_k + 1] if pos_k + 1 < len(dclose) else np.nan
        c5 = dclose.iloc[pos_k + 5] if pos_k + 5 < len(dclose) else np.nan
        later = exit_sig.iloc[pos_k:pos_k + 5] if d in exit_sig.index else pd.Series(dtype=bool)
        if len(hit):
            px = min(float(hit["Open"].iloc[0]), float(lvl[d]))
            rows.append({"date": str(d.date()), "warn": True, "r0": dclose[d] / px - 1, "r1": c1 / px - 1, "r5": c5 / px - 1,
                         "exit_within5": bool(exit_sig.loc[exit_sig.index >= d].iloc[:5].any())})
        else:
            px = float(dopen[d])
            rows.append({"date": str(d.date()), "warn": False, "r0": dclose[d] / px - 1, "r1": c1 / px - 1, "r5": c5 / px - 1,
                         "exit_within5": bool(exit_sig.loc[exit_sig.index >= d].iloc[:5].any())})
    t = pd.DataFrame(rows)
    w, n = t[t["warn"]], t[~t["warn"]]
    res["warn"][sym] = {"hold_days": int(len(t)), "warn_days": int(len(w)), "warn_per_month": round(len(w) / 10.6, 2),
                        "warn_r5_mean%": round(100 * w["r5"].mean(), 2), "nowarn_r5_mean%": round(100 * n["r5"].mean(), 2),
                        "warn_r1_mean%": round(100 * w["r1"].mean(), 2), "nowarn_r1_mean%": round(100 * n["r1"].mean(), 2),
                        "warn_r0_mean%": round(100 * w["r0"].mean(), 2), "nowarn_r0_mean%": round(100 * n["r0"].mean(), 2),
                        "warn_exit_within5%": round(100 * w["exit_within5"].mean(), 1), "nowarn_exit_within5%": round(100 * n["exit_within5"].mean(), 1)}
    # 빠른 선
    b5 = E.bars(sym, 5, "ext"); C5 = b5["Close"]; e200 = E.ema(C5, 200)
    p = E.simulate(b5, exit_close=C5 < e200 * 0.98, entry_close=C5 > e200 * 1.02, exec_sess="ext")
    slow = E.simulate(bd, exit_close=Cd < W, entry_close=Cd > R, exec_sess="reg")
    for nm, pp in (("빠른 선", p), ("느린 선", slow)):
        ev = E.evaluate(*pp, Y26, COST)
        q = E.evaluate(*pp, ("2026-08-20", "2026-11-20"), COST)
        res["fast"].setdefault(nm, {})[sym] = {"2026 수익%": ev["ret%"], "2026 낙폭%": ev["mdd%"], "시점값": ev["tv%p"], "최근분기 수익%": q["ret%"], "보유 2026%": ev["bh%"]}
    print(sym, res["warn"][sym], flush=True)

W_ = pd.DataFrame(res["warn"]).T
res["warn_summary"] = {"median_warn_r5%": round(float(W_["warn_r5_mean%"].median()), 2), "median_nowarn_r5%": round(float(W_["nowarn_r5_mean%"].median()), 2),
                       "symbols_warn_r5_lower": int((W_["warn_r5_mean%"] < W_["nowarn_r5_mean%"]).sum()),
                       "median_warn_exit5%": round(float(W_["warn_exit_within5%"].median()), 1), "median_nowarn_exit5%": round(float(W_["nowarn_exit_within5%"].median()), 1),
                       "median_warn_per_month": round(float(W_["warn_per_month"].median()), 2)}
json.dump(res, open(os.path.join(HERE, "..", "results", "warn_check.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0, default=float)
pd.set_option("display.width", 250)
print(W_[["warn_per_month", "warn_r0_mean%", "nowarn_r0_mean%", "warn_r1_mean%", "nowarn_r1_mean%", "warn_r5_mean%", "nowarn_r5_mean%", "warn_exit_within5%", "nowarn_exit_within5%"]].to_string())
print(res["warn_summary"])
F = {nm: pd.DataFrame(v).T for nm, v in res["fast"].items()}
print(pd.concat({k: v[["2026 수익%", "2026 낙폭%", "최근분기 수익%", "보유 2026%"]] for k, v in F.items()}, axis=1).to_string())
print({k: {c: round(float(v[c].median()), 1) for c in ("2026 수익%", "2026 낙폭%", "시점값", "최근분기 수익%")} for k, v in F.items()})
