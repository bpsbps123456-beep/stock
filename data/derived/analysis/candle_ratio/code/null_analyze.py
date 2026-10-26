"""MU 원형 이동 귀무 분석 (null_shift.py run 결과를 읽는다).

py code/null_analyze.py 5m   → results/null_MU_5m_report.json, _cells_both95_{all,leakfree,clean}.csv, _tod_misalign.csv
                               (_shifts.csv에 eff_days·fut_days 열을 덧붙인다)

귀무 집합 세 가지를 함께 보고한다.
- all: 과제대로 k=1..D-1 전부.
- leakfree: 미래 쪽 출처 거리(모든 tf·세션 최소)가 15거래일 이상인 k(미리보기 누출 제거, 가까운 과거 지연판은 남김).
- clean: 실효 이동 거리(모든 tf·세션에서 min(과거 s봉, 미래 n-s봉)÷하루 봉 수)가 15거래일 이상인 k만.
  15일 = 격자에서 가장 긴 신호 창(N 30 × 정규장 360분봉 하루 2봉). np.roll은 원형이라 k가 D에 가까우면
  p를 '미래' 며칠 뒤 값에서 가져온다(10/7~10/13 빠진 봉 때문에 1h 연장 60분 k=120은 단 1봉 미래 = 직접 미리보기,
  5m 일부 tf는 k=68이 이동 0 = 실제와 같음). 가까운 과거 이동은 실제 신호의 지연판이라 실제와 겹친다.
칸별 분위: leave-one-out(각 행을 나머지 행에 대해)과, 이웃 편향을 없앤 대칭 보정(per_cell_sym)을 함께 낸다.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from engine import WIN, TF_5M, TF_1H, load, tf_bars
from null_shift import SYM, OUT, ORIG, CELL_KEYS

GUARD = 15  # 거래일
SENS_GUARDS = [5, 10, 15, 20]


def pct(x):
    return round(float(np.expm1(x)) * 100, 1)


def bar_tables(base):
    d = load(SYM, base)
    win = WIN[base]
    tfs = TF_5M if base == "5m" else TF_1H
    for sess in ("eth", "rth"):
        for tf in tfs:
            tb = tf_bars(d, tf, sess)
            tb = tb[tb.day >= win["warm"]].reset_index(drop=True)
            if len(tb) < 40:
                continue
            yield sess, tf, tb, int(pd.Series(tb.day.values).value_counts().median())


def eff_dist_and_tod(base, D):
    """k별 실효 최소 이동 거리(일)와 (tf, 세션)별 출처·받는 봉 시각 불일치 비율."""
    mind = np.full(D, np.inf)
    mind[0] = 0.0
    fut = np.full(D, np.inf)  # 미래 쪽 출처 거리(일): 작으면 미리보기 누출
    tod_rows = []
    for sess, tf, tb, bpd in bar_tables(base):
        n = len(tb)
        st = pd.DatetimeIndex(tb.start)
        tod = np.asarray(st.hour * 60 + st.minute)
        idx = np.arange(n)
        mis = []
        for k in range(1, D):
            s = (k * bpd) % n
            mind[k] = min(mind[k], (min(s, n - s) / bpd) if s else 0.0)
            fut[k] = min(fut[k], ((n - s) / bpd) if s else 0.0)
            mis.append(float((tod[(idx - k * bpd) % n] != tod).mean()))
        tod_rows.append(dict(sess=sess, tf=tf, n=n, bpd=bpd, missing=bpd * D - n,
                             misalign_mean=float(np.mean(mis)), misalign_max=float(np.max(mis))))
    return mind, fut, pd.DataFrame(tod_rows)


def strict_below(A):
    """같은 열(칸)에서 A[i]보다 엄격히 작은 값의 개수. 자기 자신은 엄격 부등호라 빠진다."""
    out = np.empty(A.shape, np.int32)
    for i in range(A.shape[0]):
        out[i] = (A < A[i]).sum(0)
    return out


def proc_pvals(act, nul):
    K = len(nul)
    res = {}
    for c in ["top_conf", "top_sel", "top20_conf_med", "rho", "exf_sel", "exf_conf", "exd_sel", "exd_conf"]:
        n_ge = int((nul[c] >= act[c]).sum())
        res[c] = dict(actual=float(act[c]), null_med=float(nul[c].median()), null_q95=float(nul[c].quantile(0.95)),
                      n_ge=n_ge, K=K, p=n_ge / K, p_plus1=(n_ge + 1) / (K + 1))
    return res


def per_cell(sel, conf, cells, rows, tag, base):
    """rows[0]=실제(k=0). 각 행을 나머지 행에 대해 잰 분위로 '두 구간 모두 95분위 초과' 비율을 구하고
    귀무 행들 자신의 같은 비율과 비교(칸끼리 상관이 있어도 보정된다)."""
    A = sel[rows]
    B = conf[rows]
    K = len(rows) - 1
    bs = strict_below(A)
    bc = strict_below(B)
    thr = 0.95 * K
    ex_s = bs > thr
    ex_c = bc > thr
    both = ex_s & ex_c
    fb, fs, fc = both.mean(1), ex_s.mean(1), ex_c.mean(1)
    out = dict(K=K, rule=f"실제보다 엄격히 작은 귀무 > 0.95×{K}, 즉 귀무 {K}개 중 같거나 큰 값 {K - int(np.floor(thr)) - 1}개 이하",
               indep_expect_both=float(((K - np.floor(thr)) / (K + 1)) ** 2),
               both_actual=float(fb[0]), both_n=int(both[0].sum()),
               both_null_mean=float(fb[1:].mean()), both_null_q95=float(np.quantile(fb[1:], 0.95)),
               both_p=float((fb[1:] >= fb[0]).mean()),
               sel_actual=float(fs[0]), sel_null_mean=float(fs[1:].mean()), sel_null_q95=float(np.quantile(fs[1:], 0.95)),
               sel_p=float((fs[1:] >= fs[0]).mean()),
               conf_actual=float(fc[0]), conf_null_mean=float(fc[1:].mean()), conf_null_q95=float(np.quantile(fc[1:], 0.95)),
               conf_p=float((fc[1:] >= fc[0]).mean()))
    cl = cells[CELL_KEYS].copy()
    cl["q_sel"] = bs[0] / K
    cl["q_conf"] = bc[0] / K
    cl["ret_sel_pct"] = np.expm1(A[0].astype(np.float64)) * 100
    cl["ret_conf_pct"] = np.expm1(B[0].astype(np.float64)) * 100
    cl["bh_sel_pct"] = np.expm1(cells.bh_sel.values) * 100
    cl["bh_conf_pct"] = np.expm1(cells.bh_conf.values) * 100
    cl["null_conf_med_pct"] = np.expm1(np.median(B[1:], 0).astype(np.float64)) * 100
    hit = cl[both[0]].sort_values("ret_conf_pct", ascending=False)
    hit.to_csv(OUT / f"null_{SYM}_{base}_cells_both95_{tag}.csv", index=False)
    shape = {}
    for c in ["tf", "sess", "N", "dir", "ratio", "dayflat", "b", "s"]:
        gs = cl[c].value_counts(normalize=True)
        hs = hit[c].value_counts()
        shape[c] = {str(k): dict(n=int(v), share=round(v / max(len(hit), 1), 3), grid_share=round(float(gs[k]), 3))
                    for k, v in hs.sort_values(ascending=False).head(8).items()}
    out["both_shape"] = shape
    if len(hit):
        out["both_ret"] = dict(sel_med_pct=round(float(hit.ret_sel_pct.median()), 1),
                               conf_med_pct=round(float(hit.ret_conf_pct.median()), 1),
                               conf_beat_bh=round(float((hit.ret_conf_pct > hit.bh_conf_pct).mean()), 3),
                               conf_pos=round(float((hit.ret_conf_pct > 0).mean()), 3),
                               top10=hit.head(10).round(3).to_dict("records"))
    # 원안 칸
    m = np.ones(len(cl), bool)
    for c, v in ORIG.items():
        if c == "tf":
            continue
        m &= (cl[c] == v).values
    tf_o = 40 if base == "5m" else 60
    mm = np.flatnonzero(m & (cl.tf == tf_o).values)[0]
    o = cl.iloc[mm]
    out["orig" if base == "5m" else "orig_60m_ref"] = dict(
        tf=tf_o, q_sel=round(float(o.q_sel), 3), q_conf=round(float(o.q_conf), 3),
        ret_sel_pct=round(float(o.ret_sel_pct), 1), ret_conf_pct=round(float(o.ret_conf_pct), 1),
        null_sel_med_pct=pct(float(np.median(A[1:, mm]))), null_conf_med_pct=pct(float(np.median(B[1:, mm]))),
        null_conf_q95_pct=pct(float(np.quantile(B[1:, mm], 0.95))))
    # 고르기 1위 칸의 자기 귀무 분위
    top = int(np.argmax(A[0]))
    out["top_cell_q"] = dict(q_sel=round(float(bs[0, top] / K), 3), q_conf=round(float(bc[0, top] / K), 3))
    return out


def per_cell_sym(sel, conf, U, draws=10, seed=0):
    """leave-one-out 보정은 귀무 행에 하루 차이 이웃이 끼어 귀무 쪽 초과 비율이 낮게 나온다(실제에 유리).
    여기서는 모든 행을 같은 규칙(자기와 GUARD일 이상 떨어진 누출 없는 행 Kp개)으로 잰다."""
    elig = {r: [j for j in U if j != r and abs(r - j) >= GUARD] for r in U}
    Kp = min(len(v) for v in elig.values())
    thr = 0.95 * Kp
    rng = np.random.default_rng(seed)
    rec = []
    for r in U:
        fs = fc = fb = 0.0
        for _ in range(draws):
            js = np.sort(rng.choice(elig[r], Kp, replace=False))
            es = (sel[js] < sel[r]).sum(0) > thr
            ec = (conf[js] < conf[r]).sum(0) > thr
            fs += es.mean(); fc += ec.mean(); fb += (es & ec).mean()
        rec.append((r, fs / draws, fc / draws, fb / draws))
    df = pd.DataFrame(rec, columns=["k", "sel", "conf", "both"])
    a = df.iloc[0]
    nl = df.iloc[1:]
    far = nl[nl.k >= GUARD]
    out = dict(rows=len(U), Kp=Kp, draws=draws, expect_one=round(float((Kp - np.floor(thr)) / (Kp + 1)), 4))
    for c in ("sel", "conf", "both"):
        out[c] = dict(actual=round(float(a[c]), 4), null_mean=round(float(nl[c].mean()), 4), null_q95=round(float(nl[c].quantile(0.95)), 4),
                      p=round(float((nl[c] >= a[c]).mean()), 3), p_far_only=round(float((far[c] >= a[c]).mean()), 3))
    return out


def sub_procedure(sel, conf, mask, ks):
    """칸 일부(mask)에서 같은 고르기 절차: 고르기 1위의 확인, 상위 20칸 확인 중앙값. 실제 vs ks 귀무."""
    idx = np.flatnonzero(mask)
    rows = [0] + list(ks)
    top_c, t20 = [], []
    for r in rows:
        a = sel[r, idx]
        o = np.argsort(-a, kind="stable")
        top_c.append(float(conf[r, idx[o[0]]]))
        t20.append(float(np.median(conf[r, idx[o[:20]]])))
    top_c, t20 = np.array(top_c), np.array(t20)
    K = len(ks)
    return dict(cells=int(len(idx)), K=K,
                top_conf_pct=pct(top_c[0]), top_conf_null_med_pct=pct(np.median(top_c[1:])), top_conf_null_q95_pct=pct(np.quantile(top_c[1:], 0.95)),
                top_conf_p=round(float((top_c[1:] >= top_c[0]).mean()), 3),
                top20_pct=pct(t20[0]), top20_null_med_pct=pct(np.median(t20[1:])), top20_null_q95_pct=pct(np.quantile(t20[1:], 0.95)),
                top20_p=round(float((t20[1:] >= t20[0]).mean()), 3))


def analyze(base):
    sh = pd.read_csv(OUT / f"null_{SYM}_{base}_shifts.csv").sort_values("k").reset_index(drop=True)
    sel = np.load(OUT / f"null_{SYM}_{base}_sel.npy")
    conf = np.load(OUT / f"null_{SYM}_{base}_conf.npy")
    cells = pd.read_parquet(OUT / f"null_{SYM}_{base}_cells.parquet")
    D = sel.shape[0]
    assert list(sh.k) == list(range(D)) and not np.isnan(sel).any() and not np.isnan(conf).any()
    mind, fut, tod = eff_dist_and_tod(base, D)
    tod.to_csv(OUT / f"null_{SYM}_{base}_tod_misalign.csv", index=False)
    sh["eff_days"] = mind
    sh["fut_days"] = fut
    sh.to_csv(OUT / f"null_{SYM}_{base}_shifts.csv", index=False)
    act = sh.iloc[0]
    sets = {"all": list(range(1, D)),
            "leakfree": [k for k in range(1, D) if fut[k] >= GUARD],
            "clean": [k for k in range(1, D) if mind[k] >= GUARD]}
    rep = dict(base=base, cells=int(sel.shape[1]), D=D, guard_days=GUARD,
               shifts={k: len(v) for k, v in sets.items()},
               clean_k=[min(sets["clean"]), max(sets["clean"])],
               bh_pct=dict(eth_sel=pct(cells.bh_sel[cells.sess == "eth"].iloc[0]), eth_conf=pct(cells.bh_conf[cells.sess == "eth"].iloc[0]),
                           rth_sel=pct(cells.bh_sel[cells.sess == "rth"].iloc[0]), rth_conf=pct(cells.bh_conf[cells.sess == "rth"].iloc[0])),
               top_cell_actual={c: str(act["top_" + c]) for c in CELL_KEYS},
               tod_misalign=dict(eth_mean=round(float(tod[tod.sess == "eth"].misalign_mean.mean()), 3),
                                 eth_max=round(float(tod[tod.sess == "eth"].misalign_max.max()), 3),
                                 rth_mean=round(float(tod[tod.sess == "rth"].misalign_mean.mean()), 3),
                                 eth_tf_with_missing=int((tod[tod.sess == "eth"].missing > 0).sum()),
                                 eth_tf_total=int((tod.sess == "eth").sum())))
    for tag, ks in sets.items():
        nul = sh[sh.k.isin(ks)]
        pv = proc_pvals(act, nul)
        rep[f"procedure_{tag}"] = pv
        rep[f"procedure_{tag}_pct"] = {c: dict(actual=pct(pv[c]["actual"]), null_med=pct(pv[c]["null_med"]),
                                               null_q95=pct(pv[c]["null_q95"]), p=round(pv[c]["p"], 3))
                                       for c in ("top_conf", "top_sel", "top20_conf_med")}
        rep[f"per_cell_{tag}"] = per_cell(sel, conf, cells, [0] + ks, tag, base)
        # 세션별 절차 (정규장은 빠진 봉이 없어 시각 정렬이 완전히 유지된다)
        rep[f"procedure_by_sess_{tag}"] = {s: sub_procedure(sel, conf, (cells.sess == s).values, ks) for s in ("rth", "eth")}
    # 보호 구간 민감도 (절차 p값만)
    sens = {}
    for g in SENS_GUARDS:
        ks = [k for k in range(1, D) if mind[k] >= g]
        nul = sh[sh.k.isin(ks)]
        sens[g] = {c: round((nul[c] >= act[c]).mean(), 3) for c in ("top_conf", "top_sel", "top20_conf_med")}
        sens[g]["K"] = len(ks)
    rep["guard_sensitivity_p"] = sens
    # 칸별 대칭 보정: 누출 없는 행 U(실제 포함) 각각을 '실제'로 놓고, 자기와 15일 이상 떨어진 U 행 Kp개(무작위 10회)에 대해 잰다.
    U = [0] + sets["leakfree"]
    rep["per_cell_symmetric"] = per_cell_sym(sel, conf, U)
    # 미래 가까운 이동(누출) 진단: 실효 거리 < 5일 중 미래 쪽 k
    near_fut = sh[(sh.k > D // 2) & (sh.eff_days < 5)]
    rep["leak_check_near_future"] = near_fut[["k", "eff_days", "top_sel", "top_conf", "rho"]].round(3).to_dict("records")
    with open(OUT / f"null_{SYM}_{base}_report.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    analyze(sys.argv[1])
