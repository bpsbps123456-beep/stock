"""반박 검증(견고성 렌즈): 갈래 4 통과 칸 `N20|gvip_rel20_p≥0.5382(상위3분위)`.

사용: py verify_b4_flow_robust.py   → 표준 출력 + results/verify_b4_flow_robust.json

보는 것(결과를 보기 전에 정함)
  1) 재현: 같은 사건표·같은 하네스(evaluate_cells)로 앞 차이·뒤 차이·뒤 t·순열 p·뒤 켜짐 평균을 다시 낸다.
  2) 문턱 이웃: 앞 구간 N20 사건 분포의 2분위(중앙값 q0.50)·4분위(q0.75) 경계. (보조: q0.60·q0.80)
  3) N 이웃: N40·N60(각자 앞 구간 3분위 문턱), 그리고 같은 절대 문턱 0.5382.
  4) 진입 하루 늦춤: (a) t+2 시가 진입, 청산 일정은 그대로(t+1 종가 > L이면 t+2 시가에 들어가자마자 나옴 = −비용,
     t+2..t+9 종가 > L → 다음 시가, 아니면 t+10 종가). (b) 통째로 하루 밀기: t+1 종가가 L 위면 진입 안 함,
     t+2 시가 진입 → t+2..t+10 종가 > L → 다음 시가, 아니면 t+11 종가. (c) 특징도 하루 늦춤(lag=1 값) + (a).
  5) 비용 0.3%(왕복): 공매도 손익에서 0.2%p 더 뺀다.
  6) 뒤 구간 국면(episode)별: 켜진 사건 손익, 한 국면 빼기, 켜진 날짜 하나씩 빼기(잭나이프), 국면 34 안에서만
     켜짐 대 꺼짐.
  7) 2025 방향(보조).
판정 규칙(지시 그대로): 한 국면을 빼서 사라지거나, 이웃 값 절반 이상에서 뒤 차이 부호가 뒤집히면 refuted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import harness as H  # noqa: E402
import b4_flow as B  # noqa: E402

OUT = H.RESULTS / "verify_b4_flow_robust.json"
FEAT = "gvip_rel20_p"
BASE_THR = 0.5382  # 칸 이름의 문턱(표시용 반올림)


def front_q(ev, N, q):
    x = ev.loc[(ev["N"] == N) & H.in_period(ev, H.FRONT) & ev[FEAT].notna(), FEAT].astype(float)
    return float(np.quantile(x, q))


def cell_series(ev, N, thr, feat=FEAT):
    s = pd.Series(np.nan, index=ev.index)
    m = ev["N"] == N
    x = ev.loc[m, feat].astype(float)
    s.loc[m] = np.where(x.isna(), np.nan, (x >= thr).astype(float))
    return s


def evaluate_one(ev, name, s, y="short_ret_h10"):
    """하네스 evaluate_cells를 칸 하나로 부른다(고르기 조건은 앞 차이 > 0만 남김). 순열 은행은 원 실행과 같은 시드."""
    tab, _ = H.evaluate_cells(ev, [(name, s)], top=1, min_on=1, min_dates=1, y=y, null=False, nnull=0)
    r = tab.iloc[0].to_dict()
    keep = ["f_diff", "f_n_on", "f_dates_on", "f_t", "b_diff", "b_t", "b_p", "b_p_shift", "b_mean_on", "b_mean_off",
            "b_n_on", "b_dates_on", "b_ex_diff", "a_diff", "a_t", "a_n_on", "a_dates_on", "pass",
            "regime_top_ep", "regime_top_share"]
    out = {k: (float(r[k]) if isinstance(r[k], (int, float, np.floating, np.integer, np.bool_)) else r[k]) for k in keep}
    out["cell"] = name
    out["pass"] = bool(r["pass"])
    return out


def delayed_short(P, ev, mode="a", h=10, cost=H.COST):
    """진입 하루 늦춤 공매도 손익(설명은 모듈 독스트링 4)."""
    out = np.full(len(ev), np.nan)
    T = len(P.cal)
    for k, (s, i, Lv) in enumerate(zip(ev["sym"], ev["i"].astype(int), ev["L"].astype(float))):
        o, c = P.col("O", s), P.col("C", s)
        e = i + 2
        if e >= T or np.isnan(o[e]) or np.isnan(c[i + 1]):
            continue
        entry = o[e]
        if mode == "a":
            if c[i + 1] > Lv:
                out[k] = -cost
                continue
            rng_ = range(i + 2, i + h)
            last = i + h
        else:  # 'b' 통째로 하루 밀기
            if c[i + 1] > Lv:
                continue          # 진입 안 함
            rng_ = range(i + 2, i + h + 1)
            last = i + h + 1
        done = False
        for kk in rng_:
            if kk >= T or np.isnan(c[kk]):
                done = True
                break
            if c[kk] > Lv:
                if kk + 1 < T and not np.isnan(o[kk + 1]):
                    out[k] = (entry - o[kk + 1]) / entry - cost
                done = True
                break
        if done:
            continue
        if last < T and not np.isnan(c[last]):
            out[k] = (entry - c[last]) / entry - cost
    return out


def main():
    P = H.load_panel()
    ev = pd.read_parquet(H.RESULTS / "b4_flow_events.parquet")
    res = {"variants": [], "episodes": {}, "jackknife": [], "within_ep34": {}, "notes": []}

    # 하루 늦춘 특징(lag=1) 값
    ev_lag = B.attach(H.load_events_all(), cal=P.cal, lag=1)
    assert (ev_lag["i"].to_numpy() == ev["i"].to_numpy()).all() and (ev_lag["sym"].to_numpy() == ev["sym"].to_numpy()).all()
    ev["gvip_lag1"] = ev_lag[FEAT].to_numpy()

    # 결과 변형 열
    ev["y_cost03"] = ev["short_ret_h10"] - 0.002
    ev["y_delay_a"] = delayed_short(P, ev, "a")
    ev["y_delay_b"] = delayed_short(P, ev, "b")

    q = {N: {qq: front_q(ev, N, qq) for qq in (0.5, 0.6, 2 / 3, 0.75, 0.8)} for N in H.NS}
    res["thresholds"] = {str(N): {f"{k:.3f}": v for k, v in d.items()} for N, d in q.items()}

    V = []
    t20 = q[20][2 / 3]
    V.append(("base N20 q2/3", 20, t20, FEAT, "short_ret_h10", "기준(재현)", False))
    V.append(("N20 q0.50(2분위 경계)", 20, q[20][0.5], FEAT, "short_ret_h10", "문턱 이웃", True))
    V.append(("N20 q0.75(4분위 경계)", 20, q[20][0.75], FEAT, "short_ret_h10", "문턱 이웃", True))
    V.append(("N40 q2/3(자기 문턱)", 40, q[40][2 / 3], FEAT, "short_ret_h10", "N 이웃", True))
    V.append(("N60 q2/3(자기 문턱)", 60, q[60][2 / 3], FEAT, "short_ret_h10", "N 이웃", True))
    V.append(("진입 t+2 시가(a: 청산 일정 그대로)", 20, t20, FEAT, "y_delay_a", "진입 늦춤", True))
    V.append(("비용 0.3%", 20, t20, FEAT, "y_cost03", "비용", True))
    # 보조(이웃 판정 수에 넣지 않음)
    V.append(("N20 q0.60", 20, q[20][0.6], FEAT, "short_ret_h10", "보조 문턱", False))
    V.append(("N20 q0.80", 20, q[20][0.8], FEAT, "short_ret_h10", "보조 문턱", False))
    V.append(("N40 같은 문턱 0.538", 40, t20, FEAT, "short_ret_h10", "보조 N", False))
    V.append(("N60 같은 문턱 0.538", 60, t20, FEAT, "short_ret_h10", "보조 N", False))
    V.append(("진입 t+2 시가(b: 통째로 하루 밀기)", 20, t20, FEAT, "y_delay_b", "보조 진입", False))
    V.append(("특징도 하루 늦춤 + 진입 t+2(a)", 20, t20, "gvip_lag1", "y_delay_a", "보조 진입", False))
    V.append(("N20 q2/3 h5", 20, t20, FEAT, "short_ret_h5", "보조 보유", False))
    V.append(("N20 q2/3 h20", 20, t20, FEAT, "short_ret_h20", "보조 보유", False))

    for name, N, thr, feat, y, grp, counts in V:
        s = cell_series(ev, N, thr, feat)
        r = evaluate_one(ev, f"{name}", s, y=y)
        r.update(N=N, thr=thr, feat=feat, y=y, group=grp, counts_as_neighbor=counts)
        res["variants"].append(r)
        print(f"{name:34s} thr={thr:.4f} 앞차 {r['f_diff']*100:+.2f} (n{int(r['f_n_on'])}/{int(r['f_dates_on'])}d) | "
              f"뒤차 {r['b_diff']*100:+.2f} t {r['b_t']:.2f} p {r['b_p'] if r['b_p']==r['b_p'] else float('nan'):.3f} "
              f"켜짐 {r['b_mean_on']*100:+.2f} (n{int(r['b_n_on'])}/{int(r['b_dates_on'])}d) | 2025 {r['a_diff']*100:+.2f} "
              f"| 통과 {r['pass']}")

    # ── 국면별(뒤 구간, 기준 칸) ──
    yv = "short_ret_h10"
    base = cell_series(ev, 20, t20)
    b = ev[(ev["N"] == 20) & H.in_period(ev, H.BACK) & ev[yv].notna()].copy()
    b["on"] = base.loc[b.index]
    on, off = b[b["on"] == 1], b[b["on"] == 0]
    ep_on = on.groupby("episode")[yv].agg(["size", "sum", "mean"])
    ep_off = off.groupby("episode")[yv].agg(["size", "sum", "mean"])
    res["episodes"] = {"on": {str(k): v for k, v in ep_on.to_dict("index").items()},
                       "off": {str(k): v for k, v in ep_off.to_dict("index").items()}}
    print("\n뒤 구간 켜짐 국면별:\n", ep_on, "\n꺼짐 국면별:\n", ep_off)
    for e in sorted(set(b["episode"])):
        rest = b[b["episode"] != e]
        o1, o0 = rest[rest["on"] == 1][yv], rest[rest["on"] == 0][yv]
        g = H._codes(rest["t"].to_numpy())
        if len(o1) and len(o0):
            n1, n0, m1, m0, d, t = H.cluster_diff(rest["on"].to_numpy(float), rest[yv].to_numpy(float), g)
            d, t, m1 = float(d[0]), float(t[0]), float(m1[0])
        else:
            d = t = m1 = float("nan")
        res["episodes"][f"drop_{e}"] = {"n_on": int(len(o1)), "dates_on": int(rest.loc[rest["on"] == 1, "t"].nunique()),
                                         "n_off": int(len(o0)), "diff": d, "t": t, "mean_on": m1}
        print(f"국면 {e} 빼면: 켜짐 {len(o1)}건, 차이 {d*100 if d==d else float('nan'):+.2f}, t {t:.2f}, 켜짐 평균 {m1*100 if m1==m1 else float('nan'):+.2f}")

    # 켜진 날짜 하나씩 빼기
    on_dates = sorted(on["t"].unique())
    for d0 in on_dates:
        rest = b[b["t"] != d0]
        g = H._codes(rest["t"].to_numpy())
        n1, n0, m1, m0, d, t = H.cluster_diff(rest["on"].to_numpy(float), rest[yv].to_numpy(float), g)
        res["jackknife"].append({"drop_date": str(pd.Timestamp(d0).date()), "n_drop": int((b["t"] == d0).sum()),
                                 "diff": float(d[0]), "t": float(t[0]), "mean_on": float(m1[0])})
    jk = pd.DataFrame(res["jackknife"])
    print("\n켜진 날짜 하나씩 빼기:\n", jk.to_string())
    # 상위 두 날 빼기(손익 합 기준)
    day_sum = on.groupby("t")[yv].sum().sort_values(ascending=False)
    top2 = list(day_sum.index[:2])
    rest = b[~b["t"].isin(top2)]
    g = H._codes(rest["t"].to_numpy())
    n1, n0, m1, m0, d, t = H.cluster_diff(rest["on"].to_numpy(float), rest[yv].to_numpy(float), g)
    res["drop_top2_days"] = {"days": [str(pd.Timestamp(x).date()) for x in top2], "diff": float(d[0]), "t": float(t[0]),
                             "mean_on": float(m1[0]), "n_on": int(n1[0])}
    print("손익 상위 2일 빼기:", res["drop_top2_days"])

    # 국면 34 안에서만 켜짐 대 꺼짐
    e34 = b[b["episode"] == 34]
    g = H._codes(e34["t"].to_numpy())
    n1, n0, m1, m0, d, t = H.cluster_diff(e34["on"].to_numpy(float), e34[yv].to_numpy(float), g)
    res["within_ep34"] = {"n_on": int(n1[0]), "n_off": int(n0[0]), "dates_on": int(e34.loc[e34["on"] == 1, "t"].nunique()),
                          "dates_off": int(e34.loc[e34["on"] == 0, "t"].nunique()),
                          "mean_on": float(m1[0]), "mean_off": float(m0[0]), "diff": float(d[0]), "t": float(t[0])}
    print("국면 34 안:", res["within_ep34"])
    # 날짜 순서: 켜진 날 / 꺼진 날
    days = b.groupby("t").agg(on=("on", "first"), n=(yv, "size"), mean=(yv, "mean"), ep=("episode", "first"))
    res["back_days"] = [{"t": str(pd.Timestamp(k).date()), **{kk: (float(vv) if kk != "ep" else int(vv)) for kk, vv in v.items()}}
                        for k, v in days.to_dict("index").items()]
    print("\n뒤 구간 날짜별:\n", days.to_string())

    # 앞 구간 국면별(켜진 사건 손익)
    f = ev[(ev["N"] == 20) & H.in_period(ev, H.FRONT) & ev[yv].notna()].copy()
    f["on"] = base.loc[f.index]
    fe = f.groupby(["episode", "on"])[yv].agg(["size", "mean"]).unstack()
    res["front_episodes"] = {f"{a}|on={b}": {str(k): (None if pd.isna(v) else float(v)) for k, v in col.items()}
                             for (a, b), col in fe.to_dict().items()}
    print("\n앞 구간 국면×켜짐:\n", fe)

    # 2025 국면별
    a = ev[(ev["N"] == 20) & H.in_period(ev, H.AUX) & ev[yv].notna()].copy()
    a["on"] = base.loc[a.index]
    ae = a.groupby(["episode", "on"])[yv].agg(["size", "mean"]).unstack()
    print("\n2025 국면×켜짐:\n", ae)
    res["aux_episodes"] = {f"{a_}|on={b_}": {str(k): (None if pd.isna(v) else float(v)) for k, v in col.items()}
                           for (a_, b_), col in ae.to_dict().items()}

    # 특징이 켜진 달력 구간(뒤 구간 모든 거래일). 이탈일 t의 특징 = 진입일 t+1에 놓인 값 → 한 칸 당겨 t에 맞춘다.
    dv, _ = B.date_features(B.load_sources(), P.cal)
    xt = pd.Series(dv[FEAT].to_numpy(), index=P.cal).shift(-1)
    xb = xt[(xt.index >= H.BACK[0]) & (xt.index <= H.BACK[1])]
    onb = (xb >= t20)
    blocks, cur = [], None
    for d, v in onb.items():
        if v and cur is None:
            cur = [d, d]
        elif v:
            cur[1] = d
        elif cur is not None:
            blocks.append(cur)
            cur = None
    if cur is not None:
        blocks.append(cur)
    res["back_on_blocks"] = [[str(x0.date()), str(x1.date())] for x0, x1 in blocks]
    res["back_on_days"], res["back_days_total"] = int(onb.sum()), int(len(onb))
    print("뒤 구간 특징 켜진 거래일", int(onb.sum()), "/", len(onb), "연속 묶음:", res["back_on_blocks"])
    # 같은 날 대조(초과 손익) 차이
    for r in res["variants"][:1]:
        print("기준 칸 뒤 초과 손익 차이:", r["b_ex_diff"])

    # 판정 규칙
    nb = [r for r in res["variants"] if r["counts_as_neighbor"]]
    flips = [r["cell"] for r in nb if not (r["b_diff"] > 0)]
    res["neighbor_count"] = len(nb)
    res["neighbor_flips"] = flips
    drop34 = res["episodes"].get("drop_34", {})
    res["vanishes_without_one_episode"] = bool(drop34.get("n_on", 0) == 0 or not (drop34.get("diff", np.nan) > 0))
    print("\n이웃", len(nb), "개 중 뒤 차이 ≤ 0:", flips)
    print("국면 34 빼면 사라짐:", res["vanishes_without_one_episode"])
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return res


if __name__ == "__main__":
    main()
