"""종합 보고서 7절용: 2026-10-13 종가 기준 '지금 켜진 것' 관찰(판정 아님).

- ETF 7개의 N=20·40·60 박스 상태(종가 vs L_N, ATR 단위 거리)
- 10/1~10/13 일봉 이탈 사건의 10/13 종가 기준 상태(되돌림 여부, 확인 대기 hold1~3 상태)와
  근접 칸 표지(b3 혼자 이탈, b7 SOXX·QQQ 박스 안, b1/b6 이탈 전 5일 수익, b1 깊이)
- 10/1~10/13 60분봉 이탈(b8 사건표) 결과 요약
- 10/13 종가 기준 박스 아래 있는 종목 수
출력: results/now_1013.csv(일봉 사건 상태), 표준 출력 요약.
문턱은 각 갈래 보고서의 앞 구간 3분위 값을 그대로 옮겼다(새로 정하지 않음).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import harness as H  # noqa: E402

ASOF = pd.Timestamp("2026-10-13")
SINCE = pd.Timestamp("2026-10-01")
# 갈래 보고서의 앞 구간 문턱(그대로 옮김)
THR = {
    "cobreak_N20": 0.0354,   # b3 N20 cobreak_share 하위 3분위(0.0353982)
    "cobreak_N40": 0.02655,  # b3 N40·N60
    "cobreak_N60": 0.02655,
    "ret5_pre_N20": -0.02387,  # b1 N20 상위 3분위
    "depth_N20": 0.4892,       # b1 N20 depth_atr 상위 3분위
    "dbox_N28": -0.09375,      # b8 N28 dbox_pos 하위 3분위
}


def etf_state(P):
    rows = []
    i = len(P.cal) - 1
    assert P.cal[i] == ASOF, P.cal[i]
    for s in H.ETFS:
        o, h, l, c = (P.col(f, s) for f in "OHLC")
        for N in H.NS:
            Ls, Hs, atr = H.box_levels(o, h, l, c, N)
            rows.append(dict(sym=s, N=N, close=c[i], L=Ls[i], gap_atr=(c[i] - Ls[i]) / atr[i],
                             broken=bool(c[i] < Ls[i])))
    return pd.DataFrame(rows)


def daily_events(P):
    ev = H.load_events_all()
    ev = ev[(ev["t"] >= SINCE) & (ev["t"] <= ASOF)].copy()
    out = []
    last = len(P.cal) - 1
    for _, r in ev.iterrows():
        c = P.col("C", r["sym"])
        i = int(r["i"])
        after = c[i + 1:last + 1]
        above = np.flatnonzero(after > r["L"])
        reclaim_day = P.cal[i + 1 + above[0]].date() if len(above) else None
        nb = len(after)
        # 확인 대기(b2) 상태: t+1..t+k 종가 모두 < L 이면 k일 확인(10/13 종가까지 아는 범위)
        held = 0
        for k in range(min(nb, 3)):
            if after[k] < r["L"]:
                held += 1
            else:
                break
        N = int(r["N"])
        out.append(dict(
            sym=r["sym"], N=N, t=r["t"].date(), days_after=nb,
            close_1013=c[last], L=r["L"], vs_L_pct=(c[last] / r["L"] - 1) * 100,
            reclaim_day=reclaim_day, hold_closes_below=held,
            cls_so_far=r["cls"],
            alone_b3=bool(r["cobreak_share"] <= THR[f"cobreak_N{N}"]),
            ret5_pre=r["ret5_pre"],
            ret5_pre_on_b1=bool(N == 20 and r["ret5_pre"] >= THR["ret5_pre_N20"]),
            depth_atr=r["depth_atr"],
            short_h10=r["short_ret_h10"], exit_reason=r["exit_reason"],
        ))
    return pd.DataFrame(out)


def below_now(P):
    i = len(P.cal) - 1
    res = {}
    for N in H.NS:
        n_below = n_have = 0
        for s in H.STOCKS:
            if s not in P.j:
                continue
            o, h, l, c = (P.col(f, s) for f in "OHLC")
            Ls, _, _ = H.box_levels(o, h, l, c, N)
            if np.isnan(Ls[i]) or np.isnan(c[i]):
                continue
            n_have += 1
            n_below += int(c[i] < Ls[i])
        res[N] = (n_below, n_have)
    return res


def intraday():
    rows = []
    for N in (7, 14, 28):
        e = pd.read_parquet(H.RESULTS / f"b8_events_N{N}.parquet")
        e = e[(e["t"] >= SINCE) & (e["t"] <= ASOF)]
        done = e["exit_reason"].notna()
        rows.append(dict(N=N, events=len(e), etf=int((e["kind"] == "etf").sum()), decided=int(done.sum()),
                         reclaim=int((e["exit_reason"] == "reclaim").sum()),
                         mean_h10_decided=e.loc[done, "short_ret_h10"].mean() * 100,
                         dbox_on_N28=",".join(e.loc[e["dbox_pos"] <= THR["dbox_N28"], "sym"]) if N == 28 else ""))
    return pd.DataFrame(rows)


def main():
    P = H.load_panel()
    pd.set_option("display.width", 250)
    es = etf_state(P)
    print("== ETF 박스 상태(10/13 종가) ==")
    print(es.round(3).to_string(index=False))
    de = daily_events(P)
    de.to_csv(H.RESULTS / "now_1013.csv", index=False, encoding="utf-8-sig")
    print("\n== 10/1~10/13 일봉 이탈 사건(10/13 종가 기준) ==")
    print(de.round(3).to_string(index=False))
    print("\n== 10/13 종가가 박스 아래인 종목 수 ==")
    for N, (b, n) in below_now(P).items():
        print(f"N{N}: {b}/{n}")
    print("\n== 10/1~10/13 60분봉 이탈 ==")
    print(intraday().round(2).to_string(index=False))


if __name__ == "__main__":
    main()
