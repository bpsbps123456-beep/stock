"""경고·대피·재매수 규칙이 다른 종목에도 먹히나 — 넓은 종목군 검증 (2026-11-23).

사용자(2026-11-23): "MU에만 먹히는 게 아니라 다른 주식에도 먹혀야 하는데, SOXX 같은 거 말야."
사전 선언(결과 보기 전)
- 종목 129: watchlist.AI_SUBSECTOR 미국 상장 115 + ETF 12(SOXX SMH QQQ SPY IWM DIA XLK IGV XLF XLE XLV XLI) + 3배 레버리지 2(SOXL TQQQ).
  종목마다 따로 맞추지 않는다. 자료가 그 구간에 없는 종목은 그 구간에서 뺀다.
- 자료: Alpaca SIP 10분봉(분할 반영). 정규장(09:30~16:00) 봉으로 일봉 OHLC와 정규장 40분봉(09:30 기준)을 만든다.
- 체결: 종가 신호는 다음 정규장 시가. '닿으면'은 전날 확정된 선에 그날 정규장 저가가 닿으면 min(시가, 선). 한 날 한 번만 바뀐다.
  손익 = 보유한 날의 시가→다음 시가 로그수익(1분봉 엔진의 시가→다음 시가 합과 같다). 편도 0.1% 차감. 낙폭은 일별 시가 기준(장중 낙폭은 더 깊을 수 있다).
- 규칙 12개: ① 느린 선 일봉 EMA10·20 두 줄 띠 종가(지표 기본값) ② 띠 5·10 ③ 띠 20·50 ④ 띠 50·100 ⑤~⑨ 한 줄 EMA 10·20·50·100·200 종가
  ⑩ 보통 선(띠 3·15 닿으면) ⑪ 느린 선 닿으면 ⑫ 40분봉 EMA100(전날 마지막 40분봉 값에 닿으면 대피, 그날 종가 > 그날 마지막 40분봉 EMA100이면 재매수).
- 구간: 고르는 구간 2026-01-02~10-14, 확인 구간 10-15~11-21, 최근 분기 08-21~11-21, 2026 전체, 2025(03-03~12-31, 보조).
- 지표: 시점 값 = 규칙 로그수익 − 평균 비중×보유 로그수익(%p, 비용 차감). 급락·최근 분기는 '평소 비중'(고르는 구간 평균 비중) 기준도 함께.
- 경고 층(느린 선 보유 중, 정규장 저가가 전날 마지막 40분봉 EMA100에 닿은 날): 다음 날·5일 수익과 5거래일 안 실제 대피 비율을 경고 없는 날과 비교.
- 이동 전방 검증: 12개 중 직전 3·6개월 종목 중앙 시점 값 1위를 다음 달 모든 종목에.
결과: ../results/broad.json, broad_per_symbol.csv
"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fetch10 import AI, ETF, LEV, SYMS  # noqa: E402

CACHE = os.path.join(HERE, "..", "cache", "10m")
OUT = os.path.join(HERE, "..", "results")
COST = 0.001
WIN = {"SEL": ("2026-01-02", "2026-10-14"), "CONF": ("2026-10-15", "2026-11-21"), "Q": ("2026-08-21", "2026-11-21"),
       "Y26": ("2026-01-02", "2026-11-21"), "Y2025": ("2025-03-03", "2025-12-31")}
GROUP = {s: ("ETF" if s in ETF else "레버리지" if s in LEV else "AI 종목") for s in SYMS}


def load(sym):
    d = pd.read_parquet(os.path.join(CACHE, f"{sym}.parquet"))
    m = d.index.hour * 60 + d.index.minute
    d = d[(m >= 570) & (m < 960)].copy()
    d["date"] = pd.to_datetime(d.index.date)
    g = d.groupby("date")
    D = pd.DataFrame({"O": g["Open"].first(), "H": g["High"].max(), "L": g["Low"].min(), "C": g["Close"].last()})
    mm = d.index.hour * 60 + d.index.minute
    key = d.index.normalize() + pd.to_timedelta(570 + ((mm - 570) // 40) * 40, unit="m")
    c40 = d.groupby(key)["Close"].last()
    e100 = c40.ewm(span=100, adjust=False).mean()
    lvl = e100.groupby(pd.to_datetime(e100.index.date)).last()          # 그날 마지막 정규장 40분봉의 EMA100
    D["LV"] = lvl.reindex(D.index)
    return D


def ema(x, n):
    return x.ewm(span=n, adjust=False).mean()


def lines(D, rule):
    C = D["C"]
    kind, a, b, touch = rule
    if kind == "band":
        ea, eb = ema(C, a), ema(C, b)
        return np.minimum(ea, eb).to_numpy(), np.maximum(ea, eb).to_numpy(), touch
    if kind == "one":
        e = ema(C, a).to_numpy()
        return e, e, touch
    lv = D["LV"].to_numpy()
    return lv, lv, True


RULES = {
    "① 느린 선 띠 EMA10·20 종가(기본값)": ("band", 10, 20, False),
    "② 띠 EMA5·10 종가": ("band", 5, 10, False),
    "③ 띠 EMA20·50 종가": ("band", 20, 50, False),
    "④ 띠 EMA50·100 종가": ("band", 50, 100, False),
    "⑤ 한 줄 EMA10 종가": ("one", 10, 0, False),
    "⑥ 한 줄 EMA20 종가": ("one", 20, 0, False),
    "⑦ 한 줄 EMA50 종가": ("one", 50, 0, False),
    "⑧ 한 줄 EMA100 종가": ("one", 100, 0, False),
    "⑨ 한 줄 EMA200 종가": ("one", 200, 0, False),
    "⑩ 보통 선 띠 EMA3·15 닿으면": ("band", 3, 15, True),
    "⑪ 느린 선 띠 EMA10·20 닿으면": ("band", 10, 20, True),
    "⑫ 40분봉 EMA100 닿으면": ("lv", 0, 0, True),
}


def simulate(D, W, R, touch):
    O, Lo, C = D["O"].to_numpy(), D["L"].to_numpy(), D["C"].to_numpy()
    n = len(O)
    r = np.zeros(n)
    r[:-1] = np.log(O[1:] / O[:-1])
    pos, inc, sw = np.zeros(n), np.zeros(n), np.zeros(n)
    s = True                                             # 그날 시가의 보유 상태
    for t in range(n):
        if s:
            pos[t] = 1.0
            if touch and t > 0 and W[t - 1] == W[t - 1] and Lo[t] <= W[t - 1]:
                inc[t] = np.log(min(O[t], W[t - 1]) / O[t])
                sw[t] += 1
                nxt, end = False, False
            else:
                inc[t] = r[t]
                end = True
                nxt = not ((not touch) and W[t] == W[t] and C[t] < W[t])
        else:
            end = False
            nxt = bool(R[t] == R[t] and C[t] > R[t])
        if t + 1 < n and nxt != end:
            sw[t + 1] += 1
        s = nxt
    return pos, inc, sw, r


def evaluate(D, pos, inc, sw, r, rng):
    m = (D.index >= rng[0]) & (D.index <= rng[1])
    if m.sum() < 15:
        return None
    p, i, c, b = pos[m], inc[m], sw[m], r[m]
    net = i - COST * c
    f = float(p.mean())
    eq = np.exp(np.cumsum(net))
    bhe = np.exp(np.cumsum(b))
    return {"ret%": 100 * np.expm1(net.sum()), "bh%": 100 * np.expm1(b.sum()), "tv%p": 100 * (net.sum() - f * b.sum()),
            "mdd%": 100 * float((eq / np.maximum.accumulate(eq) - 1).min()), "bh_mdd%": 100 * float((bhe / np.maximum.accumulate(bhe) - 1).min()),
            "expo%": 100 * f, "sw": int(c.sum()), "lr": float(net.sum()), "blr": float(b.sum())}


def main():
    rows, warn_rows, states = [], [], {}
    missing = []
    for sym in SYMS:
        path = os.path.join(CACHE, f"{sym}.parquet")
        if not os.path.exists(path):
            missing.append(sym)
            continue
        D = load(sym)
        if len(D) < 120:
            missing.append(sym)
            continue
        res_sym = {}
        for nm, rule in RULES.items():
            W, R, touch = lines(D, rule)
            pos, inc, sw, r = simulate(D, W, R, touch)
            ev = {w: evaluate(D, pos, inc, sw, r, rng) for w, rng in WIN.items()}
            if ev["SEL"] is not None:
                f = ev["SEL"]["expo%"] / 100
                for w in ("CONF", "Q"):
                    if ev[w] is not None:
                        ev[w]["tvL%p"] = 100 * (ev[w]["lr"] - f * ev[w]["blr"])
            res_sym[nm] = (pos, inc, sw, r)
            row = {"sym": sym, "group": GROUP[sym], "rule": nm}
            for w, e in ev.items():
                if e:
                    for k, v in e.items():
                        if k not in ("lr", "blr"):
                            row[f"{w}|{k}"] = v
            rows.append(row)
        states[sym] = (D, res_sym)
        # 경고 층(느린 선 종가 보유 중)
        W, R, _ = lines(D, RULES["① 느린 선 띠 EMA10·20 종가(기본값)"])
        pos = res_sym["① 느린 선 띠 EMA10·20 종가(기본값)"][0]
        O, Lo, C, LV = D["O"].to_numpy(), D["L"].to_numpy(), D["C"].to_numpy(), D["LV"].to_numpy()
        exit_sig = (C < W)
        y26 = (D.index >= WIN["Y26"][0]) & (D.index <= WIN["Y26"][1])
        for t in range(1, len(D) - 5):
            if not y26[t] or pos[t] != 1 or LV[t - 1] != LV[t - 1]:
                continue
            warn = Lo[t] <= LV[t - 1]
            px = min(O[t], LV[t - 1]) if warn else O[t]
            warn_rows.append({"sym": sym, "group": GROUP[sym], "warn": bool(warn), "r1": C[t + 1] / px - 1, "r5": C[t + 5] / px - 1,
                              "exit5": bool(exit_sig[t:t + 5].any())})
        print(sym, len(D), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "broad_per_symbol.csv"), index=False, encoding="utf-8-sig")

    def summ(g):
        out = {"종목 수": int(g["SEL|tv%p"].notna().sum()),
               "전환/월": g["SEL|sw"].median() / 9.4,
               "2026 수익%": g["Y26|ret%"].median(), "2026 낙폭%": g["Y26|mdd%"].median(),
               "고른구간 시점값": g["SEL|tv%p"].median(), "고른구간 양수%": 100 * (g["SEL|tv%p"] > 0).mean(),
               "급락 tvL": g["CONF|tvL%p"].median(), "급락 양수%": 100 * (g["CONF|tvL%p"] > 0).mean(),
               "최근분기 수익%": g["Q|ret%"].median(), "최근분기 tvL": g["Q|tvL%p"].median(), "최근분기 양수%": 100 * (g["Q|tvL%p"] > 0).mean(),
               "2025 시점값": g["Y2025|tv%p"].median(), "2025 양수%": 100 * (g["Y2025|tv%p"].dropna() > 0).mean(),
               "보유 2026%": g["Y26|bh%"].median(), "보유 2026 낙폭%": g["Y26|bh_mdd%"].median(), "보유 최근분기%": g["Q|bh%"].median()}
        return pd.Series(out)
    allg = df.groupby("rule").apply(summ).round(1)
    byg = df.groupby(["group", "rule"]).apply(summ).round(1)
    pick = df[df["sym"].isin(["SOXX", "SMH", "QQQ", "SPY", "IWM", "XLF", "XLE", "XLV", "SOXL", "TQQQ", "NVDA", "MU"]) & (df["rule"].isin(["① 느린 선 띠 EMA10·20 종가(기본값)", "⑫ 40분봉 EMA100 닿으면", "③ 띠 EMA20·50 종가"]))]
    W_ = pd.DataFrame(warn_rows)
    wsum = {}
    for gname, g in [("전체", W_)] + list(W_.groupby("group")):
        a, b = g[g["warn"]], g[~g["warn"]]
        wsum[gname] = {"경고일": int(len(a)), "보유일": int(len(g)), "경고 뒤 다음 날%": round(100 * a["r1"].mean(), 2), "경고 없는 날 다음 날%": round(100 * b["r1"].mean(), 2),
                       "경고 뒤 5일%": round(100 * a["r5"].mean(), 2), "경고 없는 날 5일%": round(100 * b["r5"].mean(), 2),
                       "5거래일 안 대피(경고)%": round(100 * a["exit5"].mean(), 1), "5거래일 안 대피(없음)%": round(100 * b["exit5"].mean(), 1)}
    per_sym_w = W_.groupby("sym").apply(lambda g: pd.Series({"warn_exit5": g[g["warn"]]["exit5"].mean(), "nowarn_exit5": g[~g["warn"]]["exit5"].mean(),
                                                              "warn_r5": g[g["warn"]]["r5"].mean(), "nowarn_r5": g[~g["warn"]]["r5"].mean()}))
    wsum["종목별: 경고 뒤 5거래일 대피 비율이 더 높은 종목%"] = round(100 * float((per_sym_w["warn_exit5"] > per_sym_w["nowarn_exit5"]).mean()), 1)
    wsum["종목별: 경고 뒤 5일 수익이 더 낮은 종목%"] = round(100 * float((per_sym_w["warn_r5"] < per_sym_w["nowarn_r5"]).mean()), 1)

    # 이동 전방 검증(12개 중 직전 L개월 종목 중앙 시점 값 1위)
    months = pd.period_range("2025-01", "2026-11", freq="M")
    names = list(RULES)
    tvm = {nm: [] for nm in names}
    for sym, (D, rs) in states.items():
        mon = D.index.to_period("M")
        for nm in names:
            pos, inc, sw, r = rs[nm]
            net = inc - COST * sw
            v = []
            for p in months:
                m = mon == p
                v.append(100 * (net[m].sum() - pos[m].mean() * r[m].sum()) if m.sum() >= 5 else np.nan)
            tvm[nm].append(v)
    T = {nm: np.array(v) for nm, v in tvm.items()}                    # 종목 × 달
    wf = {}
    for LB in (3, 6):
        picks = {}
        for k in range(LB, len(months)):
            score = {nm: np.nanmedian(np.nansum(T[nm][:, k - LB:k], axis=1)) for nm in names}
            picks[k] = max(score, key=score.get)
        per = {w: [] for w in ("Y26", "Q")}
        for sym, (D, rs) in states.items():
            mon = D.index.to_period("M")
            n = len(D)
            pos = np.ones(n); inc = np.zeros(n); sw = np.zeros(n); r = rs[names[0]][3]
            prev = None
            for k in range(LB, len(months)):
                m = np.nonzero(mon == months[k])[0]
                if not len(m):
                    continue
                pp, ii, ss, _ = rs[picks[k]]
                pos[m], inc[m], sw[m] = pp[m], ii[m], ss[m]
                if prev is not None and pp[m[0]] != prev:
                    sw[m[0]] += 1
                prev = pp[m[-1]]
            for w in per:
                e = evaluate(D, pos, inc, sw, r, WIN[w])
                if e:
                    per[w].append(e["ret%"])
        wf[f"직전 {LB}개월 1위"] = {"2026 수익% 중앙": round(float(np.median(per["Y26"])), 1), "최근분기 수익% 중앙": round(float(np.median(per["Q"])), 1),
                                 "고른 규칙": sorted(set(picks.values()))}
    res = {"symbols": len(states), "missing": missing, "all": allg.reset_index().to_dict("records"),
           "by_group": byg.reset_index().to_dict("records"), "picked": pick.to_dict("records"), "warn": wsum, "walkforward": wf}
    json.dump(res, open(os.path.join(OUT, "broad.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0, default=float)
    pd.set_option("display.width", 330); pd.set_option("display.max_columns", 30)
    cols = ["종목 수", "전환/월", "2026 수익%", "2026 낙폭%", "고른구간 시점값", "고른구간 양수%", "급락 tvL", "최근분기 수익%", "최근분기 tvL", "최근분기 양수%", "2025 시점값", "2025 양수%"]
    print("빠진 종목", missing)
    print("=== 전체 (보유 2026 중앙", allg["보유 2026%"].iloc[0], "/ 낙폭", allg["보유 2026 낙폭%"].iloc[0], "/ 최근분기", allg["보유 최근분기%"].iloc[0], ")")
    print(allg[cols].sort_values("고른구간 시점값", ascending=False).to_string())
    for gname in ("ETF", "레버리지", "AI 종목"):
        g = byg.loc[gname]
        print(f"=== {gname} (보유 2026 중앙 {g['보유 2026%'].iloc[0]} / 최근분기 {g['보유 최근분기%'].iloc[0]})")
        print(g[cols].sort_values("고른구간 시점값", ascending=False).to_string())
    show = pick[["sym", "rule", "Y26|ret%", "Y26|mdd%", "Y26|bh%", "Y26|bh_mdd%", "Q|ret%", "Q|bh%", "Y2025|ret%", "Y2025|bh%", "SEL|sw"]].round(1)
    print(show.to_string(index=False))
    print(json.dumps(wsum, ensure_ascii=False, indent=0))
    print(json.dumps(wf, ensure_ascii=False))


if __name__ == "__main__":
    main()
