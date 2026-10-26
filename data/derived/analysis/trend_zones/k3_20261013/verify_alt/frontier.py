"""K3n 반박 시험: 이득이 '평균 기준점' 덕인가, '짧은 실효 기간' 덕인가.

사전 등록(바꾸지 않음):
  K0 계열 N in {7,10,12,15,20}: T = ln(C/C[N]) / (sdN*sqrt(N))
  K3 계열 N in {10,15,20,30,40}: T = (ln C - mean(ln C[0..N-1])) / (sdN*sqrt((N-1)(2N-1)/(6N)))
  sdN = 일 로그수익 N개 표본 표준편차(ddof=1). RSI14 >=70/<=30 먼저. 일봉 130개째부터.
  문턱 (a) +-0.5  (b) 2024-01-02~2025-12-31 150종목 비극단 애매 비율 = K0(N=20,+-0.5) 값이 되는 대칭 tau
  지표: 150종목, 2024-01-02~2026-10-12 (1년 바뀜은 2025-10-13~2026-10-12)
보조(사전 등록 밖, 표시함): 급등 지연, 10일 S/T1(zones.py 엔진), 경계선용 tau 촘촘 훑기.

실행: PYTHONIOENCODING=utf-8 py frontier.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "impl_A"))
import compare_mid as A  # noqa: E402  (impl_A의 측정 함수 재사용: rsi_pine, stage5, calibrate, m1~m4, ZM)

ZM = A.ZM
K0_N = [7, 10, 12, 15, 20]
K3_N = [10, 15, 20, 30, 40]
ALL_N = sorted(set(K0_N + K3_N))
SWEEP = [round(x, 2) for x in np.arange(0.20, 1.21, 0.05)]   # 보조 경계선용 문턱 21개


def mult3(N):
    return np.sqrt((N - 1) * (2 * N - 1) / (6 * N))


def per_symbol(g):
    g = g.sort_values("ts").reset_index(drop=True)
    C = g["Close"].to_numpy(float)
    s = pd.Series(np.log(C))
    lr = s.diff()
    o = pd.DataFrame({"sym": g["sym"], "date": g["ts"].dt.strftime("%Y-%m-%d"), "is_etf": g["is_etf"],
                      "C": C, "idx": np.arange(len(C))})
    o["lr"] = lr.to_numpy()
    o["sd20"] = lr.rolling(20, min_periods=20).std(ddof=1).to_numpy()
    o["rsi"] = A.rsi_pine(C)
    for N in ALL_N:
        sd = lr.rolling(N, min_periods=N).std(ddof=1)
        if N in K0_N:
            o[f"T0_{N}"] = ((s - s.shift(N)) / (sd * np.sqrt(N))).to_numpy()
        if N in K3_N:
            o[f"T3_{N}"] = ((s - s.rolling(N, min_periods=N).mean()) / (sd * mult3(N))).to_numpy()
    hi10 = pd.Series(C).rolling(10, min_periods=10).max()
    lo10 = pd.Series(C).rolling(10, min_periods=10).min()
    o["hi10"], o["lo10"] = hi10.to_numpy(), lo10.to_numpy()
    return o


def main():
    raw = pd.read_parquet(A.PANEL)
    raw["ts"] = pd.to_datetime(raw["ts"])
    df = pd.concat([per_symbol(g) for _, g in raw.groupby("sym", sort=True)], ignore_index=True)
    TCOLS = [f"T0_{N}" for N in K0_N] + [f"T3_{N}" for N in K3_N]

    stock = ~df["is_etf"]
    base_valid = (df["idx"] >= A.MIN_IDX) & np.isfinite(df["rsi"])
    nonext = (df["rsi"] > 30) & (df["rsi"] < 70)
    common = base_valid & np.all([np.isfinite(df[c]) for c in TCOLS], axis=0) & np.isfinite(df["sd20"]) \
        & np.isfinite(df["hi10"]) & np.isfinite(df["lo10"])
    common = pd.Series(np.asarray(common), index=df.index)
    print("공통 유효 행 차이(base_valid - common):", int((base_valid & ~common).sum()))

    cal = stock & A.inper(df["date"], A.P_CAL) & common & nonext
    target = float(np.mean(np.abs(df.loc[cal, "T0_20"]) < 0.5))
    print("K0(20,0.5) 비극단 애매 2024~25:", round(target, 4))

    R = df["rsi"].to_numpy()
    cfgs = []   # (name, fam, N, kind, tau, col)
    taus = {}
    for col in TCOLS:
        fam, N = col.split("_")
        N = int(N)
        famname = "K0" if fam == "T0" else "K3"
        tau_c, share_c = A.calibrate(df, col, target, cal & np.isfinite(df[col]))
        taus[col] = (tau_c, share_c)
        cfgs.append((f"{famname}-{N}-a", famname, N, "a", 0.5, col))
        cfgs.append((f"{famname}-{N}-b", famname, N, "b", tau_c, col))

    ev = {"E_dn_abs": df["C"] / df["hi10"] - 1 <= -0.10, "E_up_abs": df["C"] / df["lo10"] - 1 >= 0.10}
    S_all = stock & A.inper(df["date"], A.P_ALL) & common
    S_ly = stock & A.inper(df["date"], A.P_LASTY) & common
    dn_ev = [e for e in A.m4_events(df, common, stock, up=False) if not df.at[e[2], "is_etf"]]
    up_ev = [e for e in A.m4_events(df, common, stock, up=True) if not df.at[e[2], "is_etf"]]
    print("급락 사건", len(dn_ev), "급등 사건", len(up_ev))
    dn_ev26 = [e for e in dn_ev if df.at[e[2], "date"] >= A.P_2026[0]]

    # zones.py 엔진(보조 예측)
    st_raw = raw[~raw["is_etf"]].copy()
    st_raw["d"] = st_raw["ts"].dt.strftime("%Y-%m-%d")
    piv = {k: st_raw.pivot_table(index="d", columns="sym", values=k).sort_index() for k in ("Open", "High", "Low", "Close")}
    I = ZM.Ind(piv["Open"], piv["High"], piv["Low"], piv["Close"])
    F = ZM.Fwd(I, 10)
    prow = {p: ZM.period_rows(I, p, 10) for p in ("앞", "뒤", "2025")}

    ew = df["sym"] == "EWY"

    def measure(stage, full=True):
        df["_z"] = np.where(common, stage, -9).astype(int)
        r = {}
        a = A.m1(df, "_z", S_all)
        r["rev_share"] = a["rev_share"]
        b = A.m2(df, "_z", S_all, ev)
        r["dn_down"], r["dn_gray"], r["dn_n"] = b["E_dn_abs"]["down"], b["E_dn_abs"]["gray"], b["E_dn_abs"]["n"]
        r["up_up"], r["up_gray"], r["up_n"] = b["E_up_abs"]["up"], b["E_up_abs"]["gray"], b["E_up_abs"]["n"]
        c = A.m3(df, "_z", S_ly)
        r["flips"] = c["flips_per_sy"]
        d1 = A.m4(df, "_z", dn_ev, up=False)
        r["lag_dn"], r["early_dn"] = d1["median_lag"], d1["early"]
        if full:
            S26 = stock & A.inper(df["date"], A.P_2026) & common
            a26 = A.m1(df, "_z", S26)
            b26 = A.m2(df, "_z", S26, ev)
            r["rev_26"], r["dn_down_26"], r["up_up_26"] = a26["rev_share"], b26["E_dn_abs"]["down"], b26["E_up_abs"]["up"]
            r["lag_dn_26"] = A.m4(df, "_z", dn_ev26, up=False)["median_lag"]
            r["n_dn_26"] = len(dn_ev26)
            d2 = A.m4(df, "_z", up_ev, up=True)
            r["lag_up"] = d2["median_lag"]
            M = stock & A.inper(df["date"], A.P_CAL) & common
            r["gray_cal"] = float(np.mean(df.loc[M, "_z"] == 0))
            tmp = pd.DataFrame({"d": df["date"], "sym": df["sym"], "z": df["_z"]})[stock.to_numpy()]
            Z = tmp.pivot(index="d", columns="sym", values="z").reindex(index=I.dates, columns=I.syms).fillna(-9)
            Z = Z.to_numpy().astype(np.int8)
            for p, lab in (("앞", "front"), ("뒤", "back"), ("2025", "2025")):
                e5 = ZM.evaluate(Z, F, prow[p], full=False)
                r[f"S_{lab}"] = e5["S"] * 100
                r[f"T1_{lab}"] = e5["terms"]["T1 상방-하방"] * 100
            e = df.loc[ew].set_index("date")
            r["ewy_1012"] = int(e.at["2026-10-12", "_z"])
            aft = e[(e.index > "2026-10-02") & (e["_z"] < 0) & (e["_z"] != -9)]
            r["ewy_first_dn"] = aft.index[0] if len(aft) else "없음"
        return r

    res = {"meta": {"target_gray": target, "taus": {k: {"tau": v[0], "share": v[1]} for k, v in taus.items()},
                    "n_dn_events": len(dn_ev), "n_up_events": len(up_ev),
                    "rows_stock_all": int(S_all.sum())}, "main": {}, "sweep": {}}
    for name, fam, N, kind, tau, col in cfgs:
        r = measure(A.stage5(R, df[col].to_numpy(), tau), full=True)
        r.update({"fam": fam, "N": N, "kind": kind, "tau": tau})
        res["main"][name] = r
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    # 보조: 경계선용 문턱 훑기(10 설정 x 21 문턱 = 210)
    for col in TCOLS:
        for tau in SWEEP:
            r = measure(A.stage5(R, df[col].to_numpy(), tau), full=False)
            res["sweep"][f"{col}@{tau}"] = {"col": col, "tau": tau, **r}
        print("sweep", col, "done")

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        return o
    with open(os.path.join(HERE, "frontier.json"), "w", encoding="utf-8") as f:
        json.dump(clean(res), f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
