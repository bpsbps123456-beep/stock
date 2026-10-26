"""capture 1: 실시간 지그재그 확정 뒤 '남은 움직임'(이어진 몫·되돌림·먹은 몫) — 실제 대 무작위 가격(봉 섞기 200개).

출력: results/capture_confirm_events.csv(실제 사건 전부), results/capture_confirm.csv(설정 × 구간 × 방향 요약과 무작위 비교),
      results/capture_confirm_stock.csv(종목별 실제·무작위 평균 먹은 몫).
- 무작위는 rep마다 모든 종목에 같은 날 순서를 쓴다(같은 날 종목들이 함께 움직이는 묶음 유지 → 풀링 평균의 분산이 실제와 맞음).
- 사건은 확정일이 구간 안인 것. 구간 끝까지 반대 확정이 없으면 구간 끝 종가로 자른다(실제·무작위 같게). 미절단판(구간 뒤 자료로 끝까지 따라감)도 실제에 한해 낸다.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

import swing
from capture_lib import ZZ_CFG, ZZ_K, ZZ_TH, run_block

NREP = 200
WARM = 60
PERS = ("P1", "P2", "Y25")
COLS = ["cfg", "rep", "typ", "cf", "capt", "cont", "retr", "captx", "hold", "cens", "gone"]
OUT = swing.HERE / "results"


def eligible(D):
    spy = D["SPY"]
    cal = {p: list(spy.date.values[swing.period_index(spy, p)]) for p in PERS}
    el = {}
    for p in PERS:
        for s, g in D.items():
            idx = swing.period_index(g, p)
            if len(idx) and idx[0] >= WARM and list(g.date.values[idx]) == cal[p]:
                el.setdefault(p, []).append(s)
    return el


def summarize(df, keys):
    g = df.groupby(keys)
    out = g.agg(n=("capt", "size"), capt_mean=("capt", "mean"), capt_med=("capt", "median"),
                win=("capt", lambda x: float((x > 0).mean())), cont_mean=("cont", "mean"), cont_med=("cont", "median"),
                retr_mean=("retr", "mean"), captx_mean=("captx", "mean"), winx=("captx", lambda x: float((x > 0).mean())),
                hold_med=("hold", "median"), cens=("cens", "mean"), gone_mean=("gone", "mean"), lc_mean=("lc", "mean"),
                cont_gt_retr=("cont", "size"))
    out["cont_gt_retr"] = g.apply(lambda x: float((x.cont > x.retr).mean()), include_groups=False)
    return out.reset_index()


def main():
    t0 = time.time()
    D = swing.load()
    el = eligible(D)
    real_rows, null_rows = [], []
    for pi_, p in enumerate(PERS):
        for si, s in enumerate(el[p]):
            g = D[s]
            O, H, L, C = (g[k].values.astype(np.float64) for k in ("O", "H", "L", "C"))
            idx = swing.period_index(g, p)
            j0, j1 = int(idx[0]), int(idx[-1])
            w0 = j0 - WARM
            wO, wH, wL, wC = O[w0:j1 + 1].copy(), H[w0:j1 + 1].copy(), L[w0:j1 + 1].copy(), C[w0:j1 + 1].copy()
            n = len(wC)
            # 실제(절단)
            b = run_block(wO, wH, wL, wC, WARM, WARM, n - 1, n - 1, 1, 0, False, ZZ_TH, ZZ_K)
            f = pd.DataFrame(b, columns=COLS); f["sym"] = s; f["per"] = p; f["ver"] = "cut"
            f["date"] = g.date.values[(f.cf.values + w0).astype(int)]
            real_rows.append(f)
            # 실제(미절단: 구간 뒤 자료까지 따라감)
            fO, fH, fL, fC = O[w0:].copy(), H[w0:].copy(), L[w0:].copy(), C[w0:].copy()
            b = run_block(fO, fH, fL, fC, WARM, WARM, j1 - w0, len(fC) - 1, 1, 0, False, ZZ_TH, ZZ_K)
            f = pd.DataFrame(b, columns=COLS); f["sym"] = s; f["per"] = p; f["ver"] = "full"
            f["date"] = g.date.values[(f.cf.values + w0).astype(int)]
            real_rows.append(f)
            # 무작위
            b = run_block(wO, wH, wL, wC, WARM, WARM, n - 1, n - 1, NREP, 1000 * pi_ + 7, True, ZZ_TH, ZZ_K)  # 같은 seed → 모든 종목 같은 날 순서
            f = pd.DataFrame(b, columns=COLS); f["sym"] = s; f["per"] = p
            null_rows.append(f)
        print(p, len(el[p]), "종목", round(time.time() - t0, 1), "s", flush=True)
    R = pd.concat(real_rows, ignore_index=True)
    N = pd.concat(null_rows, ignore_index=True)
    name = {i: c[0] for i, c in enumerate(ZZ_CFG)}
    R["cfg"] = R.cfg.astype(int).map(name); N["cfg"] = N.cfg.astype(int).map(name)
    R["typ"] = R.typ.astype(int); N["typ"] = N.typ.astype(int)
    # 로그 먹은 몫: 매수 확정은 ln(반대 확정 종가/확정 종가), 매도 확정은 그 부호 반대(피한 로그 손실)
    for X in (R, N):
        X["lc"] = np.where(X.typ == 1, np.log1p(X.capt.clip(lower=-0.999999)), -np.log1p((-X.capt).clip(lower=-0.999999)))
    N["w"] = (N.capt > 0).astype(float)
    R.to_csv(OUT / "capture_confirm_events.csv", index=False)

    # 요약: 실제(절단·미절단), 무작위 rep별 → 분포
    keys = ["per", "cfg", "typ"]
    rs = summarize(R[R.ver == "cut"], keys)
    rf = summarize(R[R.ver == "full"], keys)[keys + ["n", "capt_mean", "win", "cont_mean", "cens"]]
    rf.columns = keys + [c + "_full" for c in ("n", "capt_mean", "win", "cont_mean", "cens")]
    ns = N.groupby(keys + ["rep"]).agg(n=("capt", "size"), capt_mean=("capt", "mean"), capt_med=("capt", "median"),
                                       win=("w", "mean"), cont_mean=("cont", "mean"), lc_mean=("lc", "mean"),
                                       retr_mean=("retr", "mean"), captx_mean=("captx", "mean"), gone_mean=("gone", "mean")).reset_index()
    agg = {}
    for m in ("n", "capt_mean", "capt_med", "win", "cont_mean", "retr_mean", "captx_mean", "gone_mean", "lc_mean"):
        agg[m + "_null"] = (m, "mean")
    for m in ("capt_mean", "win", "cont_mean", "captx_mean", "lc_mean", "capt_med"):
        agg[m + "_null05"] = (m, lambda x: float(np.quantile(x, 0.05)))
        agg[m + "_null95"] = (m, lambda x: float(np.quantile(x, 0.95)))
    nsum = ns.groupby(keys).agg(**agg).reset_index()
    out = rs.merge(rf, on=keys, how="left").merge(nsum, on=keys, how="left")
    # 단측 p: 무작위 rep 값이 실제 이상인 비율(위쪽), 이하인 비율(아래쪽)
    for m in ("capt_mean", "win", "cont_mean", "lc_mean", "capt_med"):
        pu, pdn = [], []
        for _, r in out.iterrows():
            x = ns[(ns.per == r.per) & (ns.cfg == r.cfg) & (ns.typ == r.typ)][m].values
            pu.append(float((x >= r[m]).mean())); pdn.append(float((x <= r[m]).mean()))
        out["p_hi_" + m] = pu; out["p_lo_" + m] = pdn
    out.to_csv(OUT / "capture_confirm.csv", index=False)
    ns.to_csv(OUT / "capture_confirm_null_reps.csv", index=False)

    # 종목별: 실제 평균 먹은 몫 − 그 종목 무작위 평균
    sr = R[R.ver == "cut"].groupby(["per", "cfg", "typ", "sym"]).capt.agg(["size", "mean"]).rename(columns={"size": "n", "mean": "capt"})
    sn = N.groupby(["per", "cfg", "typ", "sym"]).capt.mean().rename("capt_null")
    st = sr.join(sn, how="left").reset_index()
    st.to_csv(OUT / "capture_confirm_stock.csv", index=False)
    print("끝", round(time.time() - t0, 1), "s", len(R), len(N))


if __name__ == "__main__":
    sys.exit(main())
