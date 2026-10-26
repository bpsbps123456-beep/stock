"""고친 trend_5zone.pine(K3n: 20일 평균 대비 위치) 계산을 파이썬으로 그대로 옮겨 impl_A·impl_B K3n 단계와 대조한다(2026-10-13).

파인 규칙: ta.rsi = 첫 14개 변화 평균으로 시작하는 RMA(down==0 → 100, up==0 → 0),
ta.stdev(lr, 20, false) = 표본(ddof=1), ta.sma(lc, 20), bar_index < 129 → 계산 안 함(종목 자기 봉 순번).
결과: 날짜×종목 일치율(2024-01-02 이후), 불일치 원인, 2026-10-12 트레이딩뷰 대조값 → port_k3_out.json
실행: PYTHONIOENCODING=utf-8 py port_k3.py
"""
import json
import os
import sys

sys.dont_write_bytecode = True  # impl 코드가 저장소 zones.py를 import한다 → 저장소에 .pyc를 만들지 않는다

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EWY = os.path.dirname(HERE)
PANEL = os.path.join(EWY, "panel_1d_20261012.parquet")
START = "2024-01-02"
DAY = "2026-10-12"
CHECK = ["EWY", "KORU", "SOXX", "QQQ", "MU", "SNDK", "NVDA", "AMD", "PLTR", "INTC", "TSLA", "STX", "DELL"]
NAMES = {2: "극상방", 1: "상방", 0: "애매", -1: "하방", -2: "극하방"}


# ------------------------------------------------------------------ 파인 함수
def pine_rma(x, n):
    """ta.rma: 첫 값은 처음 n개(na 없는 창) 단순 평균, 그 뒤 alpha=1/n."""
    out = np.full(len(x), np.nan)
    start = None
    for i in range(len(x)):
        if not np.isfinite(x[i]):
            continue
        if start is None:
            start = i
        k = i - start
        if k == n - 1:
            out[i] = np.mean(x[start:i + 1])
        elif k >= n:
            out[i] = out[i - 1] + (x[i] - out[i - 1]) / n
    return out


def pine_rsi(c, n=14):
    d = np.r_[np.nan, np.diff(c)]
    u = np.where(np.isfinite(d), np.maximum(d, 0.0), np.nan)
    v = np.where(np.isfinite(d), np.maximum(-d, 0.0), np.nan)
    up, dn = pine_rma(u, n), pine_rma(v, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(dn == 0, 100.0, np.where(up == 0, 0.0, 100 - 100 / (1 + up / dn)))
    return np.where(np.isfinite(up) & np.isfinite(dn), r, np.nan)


def pine_zone(c, L=20, th=0.5, rl=14, hi=70, lo=30):
    """zoneCalc() 그대로. 반환: 새 단계, 옛 단계, rsi, 새 z(T3), 옛 z(T0)."""
    lc = np.log(c)
    lr = np.r_[np.nan, np.diff(lc)]                                      # lc - lc[1]
    sd = pd.Series(lr).rolling(L, min_periods=L).std(ddof=1).to_numpy()  # ta.stdev(lr, L, false)
    sma = pd.Series(lc).rolling(L, min_periods=L).mean().to_numpy()      # ta.sma(lc, L)
    k = np.sqrt((L - 1.0) * (2.0 * L - 1.0) / (6.0 * L))
    z3 = (lc - sma) / (sd * k)
    z0 = (lc - np.r_[np.full(L, np.nan), lc[:-L]]) / (sd * np.sqrt(L))  # 옛 정의
    r = pine_rsi(c, rl)
    ok_bar = np.arange(len(c)) >= 129

    def stage(z):
        zn = np.select([r >= hi, r <= lo, z >= th, z <= -th], [2, -2, 1, -1], 0).astype(float)
        return np.where(np.isfinite(z) & np.isfinite(r) & ok_bar, zn, np.nan)
    return stage(z3), stage(z0), r, z3, z0


# ------------------------------------------------------------------ 계산
def port(raw, L=20):
    rows = []
    for s, g in raw.groupby("sym", sort=True):
        g = g.sort_values("ts")
        c = g["Close"].to_numpy(float)
        z_new, z_old, r, t3, t0 = pine_zone(c, L)
        rows.append(pd.DataFrame({"sym": s, "date": g["ts"].dt.strftime("%Y-%m-%d").to_numpy(), "C": c,
                                  "rsi": r, "T3": t3, "T0": t0, "new": z_new, "old": z_old}))
    return pd.concat(rows, ignore_index=True)


def impl_a(raw):
    sys.path.insert(0, os.path.join(EWY, "impl_A"))
    import compare_mid as A
    df = pd.concat([A.per_symbol(g) for _, g in raw.groupby("sym", sort=True)], ignore_index=True)
    ok = (df["idx"] >= A.MIN_IDX) & np.isfinite(df["rsi"]) & np.isfinite(df["T3"])
    st = A.stage5(df["rsi"].to_numpy(), df["T3"].to_numpy(), 0.5)
    return pd.DataFrame({"sym": df["sym"], "date": df["date"], "A": np.where(ok, st, np.nan),
                         "rsiA": df["rsi"], "T3A": df["T3"]})


def impl_b():
    sys.path.insert(0, os.path.join(EWY, "impl_B"))
    import compare as B
    f, etf = B.load()
    F = B.features(f["Close"], 1)
    Z = B.five(F, F["T3"], 0.5).astype(float)
    Z[Z == -9] = np.nan
    C = f["Close"]
    out = pd.DataFrame(Z, index=C.index, columns=C.columns).stack(future_stack=True).rename("B").reset_index()
    out.columns = ["date", "sym", "B"]
    rsi = F["rsi"].stack(future_stack=True).rename("rsiB").reset_index()
    t3 = F["T3"].stack(future_stack=True).rename("T3B").reset_index()
    rsi.columns = ["date", "sym", "rsiB"]
    t3.columns = ["date", "sym", "T3B"]
    return out.merge(rsi, on=["date", "sym"]).merge(t3, on=["date", "sym"]), F["V"]


def parity(P, other, col, tcol, rcol):
    m = P.merge(other, on=["sym", "date"], how="left")
    m = m[m["date"] >= START]
    both = m["new"].notna() & m[col].notna()
    only_p = m["new"].notna() & m[col].isna()
    only_o = m["new"].isna() & m[col].notna()
    agree = int((m.loc[both, "new"] == m.loc[both, col]).sum())
    bad = m[both & (m["new"] != m[col])]
    return {"cells": int(both.sum()), "agree": agree, "share": agree / both.sum() if both.sum() else float("nan"),
            "only_port_valid": int(only_p.sum()), "only_impl_valid": int(only_o.sum()),
            "max_abs_dT3": float(np.nanmax(np.abs(m.loc[both, "T3"] - m.loc[both, tcol]))),
            "max_abs_dRSI": float(np.nanmax(np.abs(m.loc[both, "rsi"] - m.loc[both, rcol]))),
            "mismatch_rows": bad[["sym", "date", "C", "rsi", rcol, "T3", tcol, "new", col]].to_dict("records")}


def main():
    raw = pd.read_parquet(PANEL)
    raw["ts"] = pd.to_datetime(raw["ts"])
    P = port(raw)
    print("이식 종목 수", P["sym"].nunique(), "행", len(P))

    # 종목 안 빈 날(달력 날짜 합집합 대비) — 파인은 있는 봉만 이어 붙이므로 판 자료에 빈 날이 있으면 impl_B(날짜 피벗)와 갈린다
    cal = sorted(raw["ts"].dt.strftime("%Y-%m-%d").unique())
    gaps = {}
    for s, g in P.groupby("sym"):
        d = g["date"].tolist()
        span = [x for x in cal if d[0] <= x <= d[-1]]
        if len(span) != len(d):
            gaps[s] = len(span) - len(d)
    print("종목 안 빈 날:", gaps if gaps else "없음")

    A = impl_a(raw)
    Bdf, _ = impl_b()
    ra = parity(P, A, "A", "T3A", "rsiA")
    rb = parity(P, Bdf, "B", "T3B", "rsiB")
    for lab, r in (("impl_A", ra), ("impl_B", rb)):
        print(f"{lab}: 비교 칸 {r['cells']} 일치 {r['agree']} ({r['share'] * 100:.4f}%), 이식만 유효 {r['only_port_valid']}, "
              f"impl만 유효 {r['only_impl_valid']}, 최대 |ΔT3| {r['max_abs_dT3']:.2e}, 최대 |ΔRSI| {r['max_abs_dRSI']:.2e}")
        for row in r["mismatch_rows"][:20]:
            print("   불일치", row)

    # 트레이딩뷰 대조값
    chk = []
    last = P[P["date"] == DAY].set_index("sym")
    print(f"\n트레이딩뷰 대조값 ({DAY} 정규장 일봉 종가, 배당 미조정·분할 조정)")
    print("종목    종가        RSI14   옛T0     새T3     옛단계  새단계")
    for s in CHECK:
        r = last.loc[s]
        zo = NAMES[int(r["old"])] if np.isfinite(r["old"]) else "—"
        zn = NAMES[int(r["new"])] if np.isfinite(r["new"]) else "—"
        chk.append({"sym": s, "close": round(float(r["C"]), 4), "rsi": round(float(r["rsi"]), 2),
                    "t_old": round(float(r["T0"]), 4), "t_new": round(float(r["T3"]), 4), "zone_old": zo, "zone_new": zn})
        print(f"{s:<6} {r['C']:>10.2f}  {r['rsi']:6.2f}  {r['T0']:+7.4f}  {r['T3']:+7.4f}  {zo:<5} {zn}")

    # 머리 주석 'N = 60 … MU' 줄 확인용: MU 6/26~7/28 새 정의 N=20·60
    mu_rows = {}
    for L in (20, 60):
        g = raw[raw.sym == "MU"].sort_values("ts")
        zn, zo, r, t3, t0 = pine_zone(g["Close"].to_numpy(float), L)
        d = g["ts"].dt.strftime("%Y-%m-%d").to_numpy()
        m = (d >= "2026-06-26") & (d <= "2026-07-28")
        seq = [(d[i], NAMES.get(int(zn[i]), "—"), round(float(t3[i]), 2)) for i in np.flatnonzero(m)]
        cnt = pd.Series([x[1] for x in seq]).value_counts().to_dict()
        first_dn = next((x[0] for x in seq if x[1] in ("하방", "극하방")), "없음")
        mu_rows[L] = {"count": cnt, "first_down": first_dn, "seq": seq}
        print(f"MU 6/26~7/28 새 정의 N={L}: {cnt}, 첫 하방 {first_dn}")
        print("   ", seq)

    # EWY 9/30~10/12 옛·새
    e = P[(P.sym == "EWY") & (P.date >= "2026-09-30")][["date", "C", "rsi", "T0", "T3", "old", "new"]]
    print("\nEWY", e.round(4).to_string(index=False))

    out = {"panel": PANEL, "start": START, "check_day": DAY, "n_syms": int(P["sym"].nunique()), "gaps": gaps,
           "parity": {"impl_A": ra, "impl_B": rb}, "check": chk, "mu_jul": mu_rows,
           "ewy_tail": e.to_dict("records")}
    with open(os.path.join(HERE, "port_k3_out.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    print("saved", os.path.join(HERE, "port_k3_out.json"))


if __name__ == "__main__":
    main()
