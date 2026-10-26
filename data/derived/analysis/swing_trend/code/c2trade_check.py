"""갈래 c2trade 점검(사후, 사전 등록 밖): and_b(AND 규칙 장세 전환)가 왜 이겼는지와 얼마나 단단한지.

1. 분해: 매수 쪽만 규칙(매도 확정은 모두 V자 → 보유), 매도 쪽만 규칙(매수 확정은 모두 V자 → 현금). 둘 다 V자면 flip.
2. 순위 2~10: P1 y1 평균 i위 매수 규칙 × i위 매도 규칙으로 같은 장세 전환(i = 1..10).
3. 누수 제거: P1 사건 중 청산 체결이 6/1 이후인 것(반대 확정 5/29 이후)의 y1을 5/29 종가로 잘라 다시 학습(5/29 확정 사건은 학습 제외).
   다시 학습한 lg·and 4전략과, 바뀐 규칙이면 날짜 블록 순열 귀무 200회.
→ results/c2trade_check.csv, c2trade_check_rules.csv
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

import c2trade_run as R
import swing
from events import ETF, FEATURES

OUT = swing.HERE / "results"
PERS = ("P1", "P2", "Y25", "Y24", "Y23")
P1_END = date(2026, 5, 29)


def top_and(X1, y1, d1, top=10, min_n=50, min_d=10):
    cuts = np.array([np.nanquantile(X1[:, f], [1 / 3, 2 / 3]) for f in range(X1.shape[1])])
    B = R.tert_bins(X1, cuts)
    E = np.stack([B[:, f] == b for f in range(X1.shape[1]) for b in range(3)], axis=1)
    _, inv = np.unique(d1, return_inverse=True)
    out = []
    for i in range(E.shape[1]):
        for j in range(i + 1, E.shape[1]):
            if i // 3 == j // 3:
                continue
            m = E[:, i] & E[:, j]
            if m.sum() < min_n or len(np.unique(inv[m])) < min_d:
                continue
            out.append((float(y1[m].mean()), i, j, int(m.sum()), len(np.unique(inv[m]))))
    out.sort(key=lambda r: -r[0])
    return [dict(cuts=cuts, f1=i // 3, b1=i % 3, f2=j // 3, b2=j % 3, p1_mean=mu, p1_n=n, p1_dates=nd) for mu, i, j, n, nd in out[:top]], len(out)


def setup():
    """사건표·ZZA 상태·구간 포함·ZZA 기준값과 평가 함수."""
    D = swing.load()
    stocks = sorted(set(D) - ETF)
    ev = pd.read_parquet(OUT / "events_ZZA3.parquet").reset_index(drop=True)
    X = ev[FEATURES].values.astype(float)
    key = {(s, int(c)): i for i, (s, c) in enumerate(zip(ev.sym.values, ev.conf_idx.values))}
    info = {}
    for s in stocks:
        g = D[s]
        O, H, L, C = (g[k].values.astype(np.float64) for k in ("O", "H", "L", "C"))
        st, pi, pc, pt = swing.zigzag(C, 0.0, 3.0 * swing.atr(H, L, C, 20))
        info[s] = dict(o=O, c=C, st=st, pc=pc.astype(np.int64), pt=pt.astype(np.int64), n=len(C), dates=g.date.values,
                       evrow=np.array([key.get((s, int(c)), -1) for c in pc], np.int64))
    ref = D["SPY"]
    Dper = {p: len(swing.period_index(ref, p)) for p in PERS}
    incl = {p: [s for s in stocks if (lambda idx: len(idx) == Dper[p] and idx[0] >= R.WARM and D[s].date.iloc[-1] >= swing.PERIODS[p][1])(swing.period_index(D[s], p))] for p in PERS}
    jj = {(s, p): (int(swing.period_index(D[s], p)[0]), int(swing.period_index(D[s], p)[-1])) for p in PERS for s in incl[p]}
    zz = {(p, s, cost): R.sim_stats(info[s]["o"], info[s]["c"], info[s]["st"], *jj[(s, p)], cost) for p in PERS for s in incl[p] for cost in R.COSTS}

    def flags(s, pr):
        er = info[s]["evrow"]
        return np.where(er >= 0, pr[np.maximum(er, 0)].astype(np.int64), -1)

    def evaluate(name, buy_flag, sell_flag, mode=1):
        """buy_flag·sell_flag: 사건별 bool(이어짐). 매수 사건엔 buy_flag, 매도 사건엔 sell_flag를 쓴다."""
        pr = np.where(ev.type.values == 1, buy_flag, sell_flag)
        rows = []
        for s in stocks:
            I = info[s]
            st = R.build_state(I["n"], I["pc"], I["pt"], flags(s, pr), mode)
            for p in PERS:
                if s not in incl[p]:
                    continue
                for cost in R.COSTS:
                    r = R.sim_stats(I["o"], I["c"], st, *jj[(s, p)], cost)
                    z = zz[(p, s, cost)]
                    rows.append((name, p, s, cost, r[0] - r[4], r[0] - z[0], r[2]))
        return rows

    return ev, X, info, evaluate


def leak_free_labels(ev, info):
    """P1 사건 중 청산 체결이 6/1 이후인 것의 y1을 5/29 종가로 자른다. 반환: y, 학습 제외 마스크(P1·결측), 자른 수."""
    y_tr = ev.y1.values.copy()
    exd = pd.to_datetime(ev.exit_date)
    late = (ev.period.values == "P1") & (exd.isna().values | (exd >= pd.Timestamp(P1_END)).values)
    for e in np.flatnonzero(late):
        I = info[ev.sym.values[e]]
        c = int(ev.conf_idx.values[e]); t = int(ev.type.values[e])
        k = int(np.searchsorted(I["dates"], P1_END, side="right") - 1)
        y_tr[e] = np.nan if c + 1 > k else t * np.log(I["c"][k] / I["o"][c + 1])
    keep = ~((ev.period.values == "P1") & np.isnan(y_tr))
    return y_tr, keep, int(late.sum())


def main():
    ev, X, info, evaluate = setup()
    T = len(ev)
    rows = []
    rows += evaluate("flip(모두 V자)", np.zeros(T, bool), np.zeros(T, bool))
    pred, models = R.fit_predict(ev, X)
    rows += evaluate("and_b", pred["and"], pred["and"])
    rows += evaluate("and_b 매수만 규칙", pred["and"], np.zeros(T, bool))
    rows += evaluate("and_b 매도만 규칙", np.zeros(T, bool), pred["and"])
    rows += evaluate("lg_b", pred["lg"], pred["lg"])
    rows += evaluate("lg_b 매수만", pred["lg"], np.zeros(T, bool))
    rows += evaluate("lg_b 매도만", np.zeros(T, bool), pred["lg"])

    # 순위 1~10
    rule_rows = []
    tops = {}
    for t in (1, -1):
        tr = (ev.type.values == t) & (ev.period.values == "P1")
        tops[t], nvalid = top_and(X[tr], ev.y1.values[tr], ev.date.values[tr])
        for rk, md in enumerate(tops[t], 1):
            sel_t = ev.type.values == t
            fire = R.pred_and(md, X[sel_t])
            sub = ev[sel_t]
            row = dict(src="원래", type=t, rank=rk, rule=f"{FEATURES[md['f1']]}={md['b1']} & {FEATURES[md['f2']]}={md['b2']}",
                       p1_mean=md["p1_mean"], p1_n=md["p1_n"], p1_dates=md["p1_dates"], n_valid=nvalid)
            for p in ("P2", "Y25"):
                m = (sub.period.values == p)
                row[f"{p}_n"] = int(fire[m].sum())
                row[f"{p}_diff"] = float(sub.y1.values[m][fire[m]].mean() - sub.y1.values[m][~fire[m]].mean()) if fire[m].any() else np.nan
            rule_rows.append(row)
    for rk in range(10):
        fb = np.zeros(T, bool); fs = np.zeros(T, bool)
        sb = ev.type.values == 1; ss = ~sb
        fb[sb] = R.pred_and(tops[1][rk], X[sb]); fs[ss] = R.pred_and(tops[-1][rk], X[ss])
        rows += evaluate(f"and_b 순위{rk + 1}", fb, fs)

    # 누수 제거 다시 학습
    y_tr, keep, nlate = leak_free_labels(ev, info)
    nd = int((~keep).sum())
    ev2 = ev.copy(); ev2["y1"] = y_tr
    ev3 = ev2[keep].reset_index(drop=True); X3 = X[keep]
    pr3, models3 = R.fit_predict(ev3, X3)
    full_pr = {k: np.zeros(T, bool) for k in pr3}
    for k in pr3:
        full_pr[k][np.flatnonzero(keep)] = pr3[k]
    rows += evaluate("누수 제거 lg_a", full_pr["lg"], full_pr["lg"], mode=0)
    rows += evaluate("누수 제거 lg_b", full_pr["lg"], full_pr["lg"])
    rows += evaluate("누수 제거 and_a", full_pr["and"], full_pr["and"], mode=0)
    rows += evaluate("누수 제거 and_b", full_pr["and"], full_pr["and"])
    for t, (lg, an) in models3.items():
        rule_rows.append(dict(src="누수 제거", type=t, rank=1, rule=None if an is None else f"{FEATURES[an['f1']]}={an['b1']} & {FEATURES[an['f2']]}={an['b2']}",
                              p1_mean=None if an is None else an["p1_mean"], p1_n=None if an is None else an["p1_n"],
                              p1_dates=None if an is None else an["p1_dates"], n_valid=None,
                              lg_corr_with_orig=float(np.corrcoef(lg["coef"], models[t][0]["coef"])[0, 1]),
                              lg_same_pred_share=float((full_pr["lg"][ev.type.values == t] == pred["lg"][ev.type.values == t]).mean())))
    pd.DataFrame(rule_rows).to_csv(OUT / "c2trade_check_rules.csv", index=False)

    df = pd.DataFrame(rows, columns=["variant", "per", "sym", "cost", "ex_sw", "ex_zz", "x"])
    sm = df.groupby(["variant", "per", "cost"]).agg(n=("sym", "size"), ex_sw=("ex_sw", "median"), ex_zz=("ex_zz", "median"),
                                                     win_sw=("ex_sw", lambda s: float((s > 0).mean())), x=("x", "median")).reset_index()
    sm.to_csv(OUT / "c2trade_check.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 400)
    print("누수: P1 사건 중 청산 체결 6/1 이후", nlate, "개, 학습 제외", nd)
    print(pd.DataFrame(rule_rows).to_string(float_format=lambda v: f"{v:.3f}"))
    w = sm[sm.cost == 0].pivot(index="variant", columns="per", values=["ex_sw", "ex_zz"])
    print((w * 100).round(1).to_string())
    w5 = sm[sm.cost == 0.0005].pivot(index="variant", columns="per", values=["ex_sw", "ex_zz"])
    print("비용 0.05%"); print((w5 * 100).round(1).to_string())


if __name__ == "__main__":
    main()
