"""adaptive 갈래 점검: 미래 정보 교란 시험, 같은 바구니 고정판과의 짝 비교, 계열별 분포, 노출 전환 비용.
results.json 에 diagnostics 키로 덧붙인다(adaptive.py 다음에 실행)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adaptive as A  # noqa: E402

E = A.E


def lookahead_test(fam, p, cut="2026-04-01"):
    """cut 이후 가격을 무작위 배율로 바꿔도 cut 전날 종가까지 경로가 같아야 한다."""
    e0 = E.Engine()
    r0 = A.Builder(e0).build(fam, p)
    e1 = E.Engine()
    kc = e1.pos[cut]
    rng = np.random.default_rng(0)
    f = np.exp(rng.normal(0, 0.05, e1.AC[kc:].shape).cumsum(axis=0))
    e1.AC = e1.AC.copy()
    e1.AO = e1.AO.copy()
    e1.AC[kc:] *= f
    e1.AO[kc:] *= f
    e1.bench_ac = e1.bench_ac.copy()
    for s in ["QQQ", "SOXX"]:
        g = np.exp(rng.normal(0, 0.05, len(e1.days) - kc).cumsum())
        e1.bench_ac.loc[e1.bench_ac.index[kc:], s] *= g
    r1 = A.Builder(e1).build(fam, p)
    j = r0.days.index(cut)
    same_before = bool(np.allclose(r0.v_close[:j], r1.v_close[:j]) and np.allclose(r0.v_open[:j], r1.v_open[:j]))  # 컷 개장가는 r_oo[j-1] 가 컷 시가를 써서 바뀌는 게 정상
    changed_after = bool(not np.allclose(r0.v_close[j:], r1.v_close[j:]))
    return {"same_before_cut": same_before, "changed_after_cut": changed_after}


def main():
    path = HERE / "results.json"
    d = json.load(open(path, encoding="utf-8"))
    rows = d["rows"]
    by = {r["id"]: r for r in rows}
    e = E.Engine()
    B = A.Builder(e)
    diag = {}

    # 1) 미래 정보 교란 시험(후보 3개 + 각 계열 1개)
    tests = [(f["family"], f["params"]) for f in d["finalists"]]
    tests.append(("F4", dict(u="TIME", n=7, tgt=0.25)))
    tests.append(("F1", dict(u="ALL50", n=10, lb=120, h=10, K=20, alt="mom20", du="ALL50")))
    diag["lookahead"] = {A.cid(f, p): lookahead_test(f, p) for f, p in tests}

    # 2) 짝 비교: F1 vs 같은 u,n,lb 고정 모멘텀 / F2 vs lb120 고정 / F3·F4 vs 같은 평균 노출 대조
    def st(u, n, lb):
        return by[A.cid("F0", dict(u=u, n=n, lb=lb))]
    pair = {}
    for fam in ["F1", "F2"]:
        dh1, dh2, drc, win2 = [], [], [], 0
        for r in rows:
            if r["family"] != fam:
                continue
            p = r["params"]
            s = st(p["u"], p["n"], p.get("lb", 120))
            dh1.append(r["H1"] - s["H1"])
            dh2.append(r["H2"] - s["H2"])
            drc.append(r["RECENT"] - s["RECENT"])
            win2 += r["H2"] > s["H2"]
        pair[fam] = {"n": len(dh1), "median_dH1": float(np.median(dh1)), "median_dH2": float(np.median(dh2)),
                     "median_dRECENT": float(np.median(drc)), "share_H2_better": win2 / len(dh1)}
    for fam in ["F3", "F4"]:
        dh2, win2, dm = [], 0, []
        for r in rows:
            if r["family"] != fam:
                continue
            p = r["params"]
            b0 = B.static(p["u"], p["n"], 120)
            c = e.metrics(e.exposure_control(b0, r["avg_expo"]))
            dh2.append(r["H2"] - c["H2"])
            dm.append(r["MDD_H2"] - c["MDD_H2"])
            win2 += r["H2"] > c["H2"]
        pair[fam + "_vs_expo_ctrl"] = {"n": len(dh2), "median_dH2": float(np.median(dh2)),
                                       "median_dMDD_H2": float(np.median(dm)), "share_H2_better": win2 / len(dh2)}
    diag["paired"] = pair

    # 3) F1 갈아타기 비중(상·하반기) — 후보
    f1 = [f for f in d["finalists"] if f["family"] == "F1"]
    if f1:
        p = f1[0]["params"]
        D = B.dialer.dial(p["du"], p["lb"], p["h"], p["K"])
        k0, km, k1 = e.pos[E.Y26], e.pos["2026-07-01"], e.pos[E.LAST]
        rebs = e.rebalance_days(k0, k1, "W")
        alt = {k: bool(np.isfinite(D[k]) and D[k] < 0) for k in rebs}
        diag["F1_finalist_alt_weeks"] = {"H1": f"{sum(v for k, v in alt.items() if k < km)}/{sum(1 for k in alt if k < km)}",
                                         "H2": f"{sum(v for k, v in alt.items() if k >= km)}/{sum(1 for k in alt if k >= km)}"}
        s25 = B.static(p["u"], p["n"], p["lb"], start=A.W25[0], end=A.W25[1])
        diag["F1_finalist_static_same_basket_2025H2"] = e.metrics_2025(s25)
        sd = e.metrics(B.static(p["u"], p["n"], p["lb"], delay=1))
        diag["F1_finalist_static_same_basket_delay1_H2"] = sd["H2"]
    f2 = [f for f in d["finalists"] if f["family"] == "F2"]
    if f2:
        p = f2[0]["params"]
        diag["F2_finalist_static_lb120_2025H2"] = e.metrics_2025(B.static(p["u"], p["n"], 120, start=A.W25[0], end=A.W25[1]))
        log = []
        B.lbrot(p["u"], p["n"], p["L"], log=log)
        diag["F2_finalist_lb_share_2026"] = {str(lb): float(np.mean([x == lb for x in log])) for lb in A.LBS}

    # 4) 노출 전환 비용(엔진은 노출 변화에 비용을 매기지 않음): 편도 10bp × |Δ노출| 를 덧붙인 H2
    f3 = [f for f in d["finalists"] if f["family"] in ("F3", "F4")]
    for f in f3:
        p = f["params"]
        r = B.build(f["family"], p, cost_bps=10)
        ex = r.expo
        dex = np.abs(np.diff(np.r_[ex[0], ex]))
        vo, vc = np.ones(len(ex)), np.ones(len(ex))
        for j in range(len(ex)):
            if j:
                vo[j] = vo[j - 1] * (1 + ex[j - 1] * r.r_oo[j - 1])
            vo[j] *= (r.v_open[j] / (r.v_open[j - 1] * (1 + ex[j - 1] * r.r_oo[j - 1])) if j else 1.0)  # 엔진 바구니 비용
            vo[j] *= 1 - 10 / 1e4 * dex[j]
            vc[j] = vo[j] * (1 + ex[j] * r.r_oc[j])
        rr = E.Result(r.days, vo, vc, ex, r.r_oo, r.r_oc, r.turnover, {})
        diag[f["id"] + "_cost10_with_expo_switch_H2"] = e.span(rr, "2026-07-01", E.LAST)
        diag[f["id"] + "_n_expo_changes_2026"] = int((dex > 0).sum())

    # 5) 계열별 분포
    fam = {}
    for f in ["F0", "F1", "F2", "F3", "F4"]:
        rs = [r for r in rows if r["family"] == f]
        fam[f] = {k: float(np.median([r[k] for r in rs])) for k in ["H1", "H1_MDD", "H2", "RECENT", "MDD_H2"]}
        fam[f]["share_H2_gt_base"] = float(np.mean([r["H2"] > d["baseline"]["H2"] for r in rs]))
    diag["family_median"] = fam

    d["diagnostics"] = diag
    json.dump(d, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(diag, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
