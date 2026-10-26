"""검증자 #1 보충 — K3(경고 주에만 '극상방=매도')의 표본 힘(2026-10-11, 연구용).
1) 날짜별 T3 계열을 경고 더미에 회귀(NW, 겹침 10일) → 경고-평상 차이의 t. 구간별 + 4구간 합침.
2) 경고 연속 묶음(에피소드)마다 T3 평균 → 양수 묶음 수 / 전체 묶음 수.
3) 자격 107개를 과열 축(xrsi70·xret1.5·xdev1.5·이격)으로 나눠 경고/평상 T3 양수 비율.
4) DRUG 제외·날짜별 순위로 같은 표.
실행: $env:PYTHONIOENCODING='utf-8'; py data/derived/analysis/trend_zones/code/verify_indep1_k3pool.py
결과: results/verify_indep1_k3pool.json, .txt
"""
import json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import verify_indep1_core as V  # noqa: E402

H = 10
PS = ("앞", "뒤", "2025", "2024")


def nw_ols(y, x, lag):
    """y = a + b x, NW(Bartlett) 표준오차. b와 t를 돌려준다."""
    m = np.isfinite(y)
    y, x = y[m], x[m].astype(float)
    X = np.c_[np.ones_like(x), x]
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    n = len(y)
    S = (X * u[:, None]).T @ (X * u[:, None])
    for k in range(1, min(lag, n - 1) + 1):
        w = 1 - k / (lag + 1)
        G = (X[k:] * u[k:, None]).T @ (X[:-k] * u[:-k, None])
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    return float(b[1]), float(b[1] / np.sqrt(cov[1, 1])), float(b[0]), int(n)


def blocks(r):
    out, cur = [], [r[0]]
    for a, b in zip(r[:-1], r[1:]):
        if b == a + 1:
            cur.append(b)
        else:
            out.append(cur)
            cur = [b]
    out.append(cur)
    return out


def analyse(Pn, EX, dates, warn, label, L, cfgs_fam):
    res = {}
    reps = {"사전1위 mz60·xrsi70": V.WIN, "옛칼만": V.BASE,
            "mz60·0.5+xret1.5": ("tx", "mz", 60, 0.5, "xret", 1.5),
            "cross(20,60)1.0+xdev1.5": ("tx", "cross", (20, 60), 1.0, "xdev", 1.5)}
    L.append(f"\n==== {label}")
    for nm, c in reps.items():
        Z = V.zones(Pn, c)
        ys, xs, eps = [], [], []
        parts = []
        for p in PS:
            r = V.rows_of(dates, p, H)
            sc = V.score(Z, EX, r)
            y = sc["ser"]["T3"]
            x = warn[r]
            b, t, a, n = nw_ols(y, x, H)
            parts.append(f"{p} 차 {b * 100:+.2f}(t {t:+.2f})")
            ys.append(y)
            xs.append(x)
            rw = r[warn[r]]
            if len(rw):
                for bl in blocks(list(rw)):
                    sb = V.score(Z, EX, np.array(bl))
                    v = sb["T"]["T3"]
                    if np.isfinite(v):
                        eps.append((p, dates[bl[0]], len(bl), v))
        # 4구간 합침(구간 사이 경계 겹침은 무시, 날짜는 서로 다름)
        b, t, a, n = nw_ols(np.concatenate(ys), np.concatenate(xs), H)
        pos = sum(1 for e in eps if e[3] > 0)
        L.append(f"[{nm}] 경고-평상 T3 차: " + " | ".join(parts) + f" || 4구간 합침 {b * 100:+.2f}%p (t {t:+.2f}, n {n}), 평상 수준 {a * 100:+.2f}")
        L.append(f"    경고 묶음 {len(eps)}개 중 T3 양수 {pos}개: " + ", ".join(f"{e[0]}:{e[1]}({e[2]}일) {e[3] * 100:+.1f}" for e in eps))
        res[nm] = {"pooled_diff": b, "pooled_t": t, "episodes": len(eps), "episodes_pos": pos}
    # 과열 축별
    L.append("과열 축별 T3 양수 비율(평상 → 경고):")
    fam_res = {}
    for fam, ks in cfgs_fam.items():
        parts = []
        for p in PS:
            r = V.rows_of(dates, p, H)
            vals = {"평상": [], "경고": []}
            for c in ks:
                Z = V.zones(Pn, c)
                for s, m in (("평상", ~warn), ("경고", warn)):
                    vals[s].append(V.score(Z, EX, r[m[r]])["T"]["T3"])
            a_, b_ = np.array(vals["평상"]), np.array(vals["경고"])
            parts.append(f"{p} {np.mean(a_ > 0) * 100:3.0f}%→{np.mean(b_ > 0) * 100:3.0f}%")
            fam_res[f"{fam}|{p}"] = (float(np.mean(a_ > 0)), float(np.mean(b_ > 0)))
        L.append(f"  {fam}({len(ks)}개): " + "  ".join(parts))
    res["fam"] = fam_res
    return res


def main():
    dates, syms, P = V.load()
    Pn = V.Panel(dates, syms, P)
    F = Pn.fwd(H)
    EX = V.excess(F)
    warn, _ = V.soxx_warn(dates)
    G = V.grid()
    rows = V.rows_of(dates, "앞", H)
    el = []
    for c in G:
        sc = V.score(V.zones(Pn, c), EX, rows)
        if V.eligible(sc["share"]):
            el.append(c)
    fam = {}
    for c in el:
        f = "이격" if c[0] == "dev" else f"{c[4]}{c[5]}"
        fam.setdefault(f, []).append(c)
    L = [f"자격 {len(el)}개, 과열 축: " + ", ".join(f"{k} {len(v)}" for k, v in fam.items())]
    out = {"base": analyse(Pn, EX, dates, warn, "기준(원본 정의)", L, fam)}
    d2, s2, P2 = V.load(drop=("DRUG",))
    P_ = V.Panel(d2, s2, P2)
    w2, _ = V.soxx_warn(d2)
    fam2 = fam   # 같은 설정 묶음
    out["noDRUG"] = analyse(P_, V.excess(P_.fwd(H)), d2, w2, "DRUG 제외", L, fam2)
    rk = pd.DataFrame(F).rank(axis=1, pct=True).to_numpy()
    out["rank"] = analyse(Pn, np.where(np.isfinite(F), rk - 0.5, np.nan), dates, warn, "날짜별 순위(분위 단위, %p 아님)", L, fam)
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(V.RES, "verify_indep1_k3pool.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(V.RES, "verify_indep1_k3pool.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
