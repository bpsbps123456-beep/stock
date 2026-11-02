"""①★(H2: ① 꼬리 반전 + 종가가 당일 VWAP보다 3% 넘게 아래)를 7·8·9월(2026 주, 2025 비교)에 달별로 잰다.
이 달들은 2차에서 단서를 찾을 때 본 기간이라 확인이 아니라 기록이다(사용자 요청 2026-11-03).
체결: E2(다음 봉 종가 진입) + 비용 모형, E1(신호 마감 + 10초 매도호가 진입, 창 끝 매수호가 청산 — 실측 호가).
사용: py h3_months.py         → results/h3_months.json, results/h3_months_signals.csv
      py h3_months.py split   → results/h3_months_split.json (탐색: 나스닥 하락일·신호 몰린 날 나누기)"""
import os, json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import engine as E
import cond_eval as CE
from fetch_alpaca import keys, Pace
from fetch_quotes import last_quote

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../results")
SRC = {"2026": "alp1m", "2025": "alp1m_2025"}


def pick(R, thr=-0.03):
    X = R[(R.pat == "A") & (R.vwap_gap < thr)].sort_values(["sid", "ts"], kind="stable")
    return X[E.dedup(X.sid.to_numpy().astype(np.int64), X.ts.to_numpy().astype(np.int64), 30)].copy()


def e1(S, year):
    head, pace = keys(), Pace(3.0)

    def one(r):
        t = pd.Timestamp(int(r[1]) * 60, unit="s", tz="UTC")
        ai, bi = last_quote(r[0], t + pd.Timedelta(seconds=70), head, pace)
        ao, bo = last_quote(r[0], t + pd.Timedelta(minutes=31), head, pace)
        return ai, bi, ao, bo

    with ThreadPoolExecutor(6) as ex:
        q = list(ex.map(one, S[["sym", "ts"]].itertuples(index=False, name=None)))
    S[["ask_in", "bid_in", "ask_out", "bid_out"]] = np.array(q, dtype=float)
    bars, out = {}, []
    for x in S.itertuples():
        if x.sym not in bars:
            bars[x.sym] = E.prep(E.load(x.sym, SRC[year]))
        a = bars[x.sym]
        t = int(x.t)
        if not (x.ask_in > 0):
            out.append(np.nan)
            continue
        z = t + 1
        while z + 1 < len(a["ts"]) and a["ts"][z + 1] <= a["ts"][t] + 30 and a["day"][z + 1] == a["day"][t]:
            z += 1
        tg = x.ask_in * 1.01
        hit = a["c"][t + 1] >= tg or (z >= t + 2 and np.max(a["h"][t + 2:z + 1]) >= tg)
        out.append(0.01 if hit else (x.bid_out if x.bid_out > 0 else a["c"][z]) / x.ask_in - 1)
    S["r1_e1"] = out
    return S


def stats(X):
    if len(X) == 0:
        return {"n": 0}
    e = X.r1_e1.dropna()
    return {"n": int(len(X)), "days": int(X.day.nunique()), "per_day": float(len(X) / X.day.nunique()),
            "syms": int(X.sym.nunique()), "hit_e2": float(X.hit_e2.mean()), "r1_e2": float(X.r1_e2.mean()),
            "r1n_model": float(X.r1n.mean()), "r2_e2": float(X.r2_e2.mean()),
            "e1_n": int(len(e)), "e1_mean": float(e.mean()), "e1_median": float(e.median()), "e1_hit": float((e == 0.01).mean()),
            "e1_sum": float(e.sum()), "e1_worst": float(e.min()), "e1_loss_gt1": float((e < -0.01).mean()),
            "e1_pos_days": float((X.groupby("day").r1_e1.sum() > 0).mean())}


def main():
    cost = json.load(open(os.path.join(RES, "cost_model.json"), encoding="utf-8"))
    res, rows = {}, []
    for year in ("2026", "2025"):
        R = CE.load(year, cost)
        X = pick(R)
        X = X[(X.day // 100 % 100).isin([7, 8, 9, 10])].copy()
        X = e1(X, year)
        X["month"] = X.day // 100 % 100
        X["year"] = year
        rows.append(X)
        res[year] = {f"{m}월": stats(g) for m, g in X.groupby("month")}
        res[year]["7~9월"] = stats(X[X.month.isin([7, 8, 9])])
        # 문턱 −4%(설명용)
        X4 = X[X.vwap_gap < -0.04]
        res[year]["7~9월_문턱-4%"] = stats(X4[X4.month.isin([7, 8, 9])])
    keep = ["year", "month", "sym", "ts", "day", "mins", "vwap_gap", "hit_e2", "r1_e2", "r1n", "r2_e2", "r1_e1",
            "ask_in", "bid_in", "ask_out", "bid_out"]
    pd.concat(rows)[keep].to_csv(os.path.join(RES, "h3_months_signals.csv"), index=False)
    json.dump(res, open(os.path.join(RES, "h3_months.json"), "w"), ensure_ascii=False, indent=1)
    for y, v in res.items():
        for k, s in v.items():
            if s.get("n"):
                print(y, k, f"n{s['n']} 하루 {s['per_day']:.1f}건 도달(E2) {s['hit_e2'] * 100:.0f}% | 비용모형 {s['r1n_model'] * 100:+.2f}% "
                      f"| 실측호가 평균 {s['e1_mean'] * 100:+.3f}% 도달 {s['e1_hit'] * 100:.0f}% 합 {s['e1_sum'] * 100:+.1f}% "
                      f"최악 {s['e1_worst'] * 100:+.2f}% 1%넘게 손실 {s['e1_loss_gt1'] * 100:.0f}% 플러스 날 {s['e1_pos_days'] * 100:.0f}%")


def split():
    """탐색(사전 등록 아님, 2갈래 × 3기간): 나스닥(QQQ) 당일 −1% 넘게 하락한 날과 그 외, 신호가 하루 10건 넘게 몰린 날과 그 외."""
    M = pd.read_csv(os.path.join(RES, "h3_months_signals.csv"))
    H = pd.read_csv(os.path.join(RES, "h3_signals_e1.csv"))
    H = H[H.hyp == "H2"]
    sets = {}
    for lab, X, cf in (("2026 7~9월", M[(M.year == 2026) & M.month.isin([7, 8, 9])], "cond_2026"),
                       ("2025 7~9월", M[(M.year == 2025) & M.month.isin([7, 8, 9])], "cond_2025"),
                       ("2026 1~6월", H, "cond_2026h1")):
        c = pd.read_parquet(os.path.join(E.CACHE, f"{cf}.parquet"), columns=["sym", "ts", "arm", "pat", "qqq_day"])
        c = c[(c.arm == "sig") & (c.pat == "A")]
        X = X.merge(c[["sym", "ts", "qqq_day"]], on=["sym", "ts"], how="left").dropna(subset=["r1_e1"])
        X["dayn"] = X.day.map(X.groupby("day").size())
        f = lambda g: {"n": int(len(g)), "e1_mean": float(g.r1_e1.mean()) if len(g) else None,
                       "e1_hit": float((g.r1_e1 == 0.01).mean()) if len(g) else None}
        sets[lab] = {"전체": f(X), "QQQ<-1%": f(X[X.qqq_day < -0.01]), "그 외": f(X[X.qqq_day >= -0.01]),
                     "신호 10건+ 날": f(X[X.dayn >= 10]), "그 외(몰림 아님)": f(X[X.dayn < 10])}
    json.dump(sets, open(os.path.join(RES, "h3_months_split.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(sets, ensure_ascii=False))


if __name__ == "__main__":
    import sys
    split() if len(sys.argv) > 1 and sys.argv[1] == "split" else main()
