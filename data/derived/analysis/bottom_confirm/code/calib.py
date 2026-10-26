"""사후(결과 본 뒤) 설명표: 하락 저점 대비 이미 오른 폭(up)별 '저점 한 달 유지(hold20)' 확률.

갈래 54개 결과에서 70%를 넘긴 규칙이 거의 모두 반등 폭으로 설명됐다(critic·verify). 그래서 답을 규칙 대신 '반등 폭별 확률표'로 낸다.
등록 판정이 아니다. in_dd 날 전부를 쓰고(쿨다운 없음), 같은 하락 국면에 속한 날들은 독립이 아니므로 국면 수를 함께 적는다.
국면 = 같은 계열에서 Ldd 값이 같은 연속 구간(같은 하락 저점을 기준으로 한 날들).

py code/calib.py → results/calib_up.csv · calib_cushion.csv · calib_now.txt
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import bc_lib as L

BINS = [0.0, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, np.inf]
LABELS = ["0~3%", "3~5%", "5~8%", "8~10%", "10~15%", "15~20%", "20~30%", "30%+"]
CUSH = [0, 1, 2, 3, 4, 6, 8, np.inf]
CLAB = ["0~1σ", "1~2σ", "2~3σ", "3~4σ", "4~6σ", "6~8σ", "8σ+"]


def long_table(fr, closes, kind):
    up, st = fr["up"], fr["in_dd"]
    sig = closes.pct_change(fill_method=None).rolling(20).std()
    rows = []
    for col in up.columns:
        d = pd.DataFrame(dict(up=up[col], st=st[col], L=fr["L60"][col], hold20=fr["hold20"][col], near5=fr["near5"][col],
                              win20=fr["win20"][col], r20=fr["r20"][col], cush=up[col] / (sig[col] * np.sqrt(20))))
        d["series"] = col
        rows.append(d[d.st & (d.index >= L.EVAL_START[kind]) & d.up.notna()])
    t = pd.concat(rows)
    t["episode"] = t.series + "|" + t.L.round(6).astype(str)
    return t


def table(t, col, bins, labels, periods):
    out = []
    for pname, a, b in periods:
        g = t[(t.index >= a) & (t.index <= b)].copy()
        g["bin"] = pd.cut(g[col], bins, right=False, labels=labels)
        for lab, h in g.groupby("bin", observed=False):
            v = h.hold20.dropna()
            out.append(dict(period=pname, bin=lab, days=len(v), episodes=h.loc[v.index, "episode"].nunique(),
                            hold20=v.mean() if len(v) else np.nan, near5=h.near5.mean(), win20=h.win20.mean(), r20=h.r20.mean()))
    return pd.DataFrame(out)


def main():
    pd.set_option("display.width", 220)
    sx = L.load_soxx()
    fr = L.soxx_frames()
    tS = long_table(fr, sx[["C"]].rename(columns={"C": "SOXX"}), "SOXX")
    o, c = L.load_stocks()
    sf = L.stock_frames()
    tK = long_table(sf, c, "STK")
    a = table(tS, "up", BINS, LABELS, L.PERIODS["SOXX"]).assign(kind="SOXX")
    b = table(tK, "up", BINS, LABELS, L.PERIODS["STK"]).assign(kind="STK")
    up = pd.concat([a, b], ignore_index=True)
    up.to_csv(L.RES / "calib_up.csv", index=False, encoding="utf-8-sig")
    cu = pd.concat([table(tS, "cush", CUSH, CLAB, L.PERIODS["SOXX"]).assign(kind="SOXX"),
                    table(tK, "cush", CUSH, CLAB, L.PERIODS["STK"]).assign(kind="STK")], ignore_index=True)
    cu.to_csv(L.RES / "calib_cushion.csv", index=False, encoding="utf-8-sig")
    for k, g in up.groupby("kind"):
        print(f"== {k} up 구간별 ==")
        print(g.pivot_table(index="bin", columns="period", values=["hold20", "episodes"], observed=False).round(3).to_string())
    print(cu[cu.period.isin(["2024~2026", "2026"])].pivot_table(index="bin", columns=["kind", "period"], values="hold20", observed=False).round(3).to_string())
    # 지금(마지막 날) 상태
    last = fr["up"].index[-1]
    lines = [f"마지막 세션 {last.date()}", f"SOXX 종가 {sx.C.iloc[-1]:.2f}, 하락 저점 Ldd {fr['L60']['SOXX'].iloc[-1]:.2f}, "
             f"up {fr['up']['SOXX'].iloc[-1]:+.2%}, 60일 고점 대비 {fr['dd60']['SOXX'].iloc[-1]:+.2%}, in_dd {bool(fr['in_dd']['SOXX'].iloc[-1])}"]
    pk = sx.C.iloc[-60:]
    lines.append(f"60일 고점 {pk.max():.2f} ({pk.idxmax().date()}), 하락 저점 날짜 {sx.C.loc[pk.idxmax():].idxmin().date()}")
    st = sf["in_dd"].iloc[-1]
    ups = sf["up"].iloc[-1][st]
    lines.append(f"종목 in_dd {int(st.sum())}/{len(st)}개, 그중 up 중앙 {ups.median():+.1%}, up≥20% {int((ups >= 0.20).sum())}개")
    (L.RES / "calib_now.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
