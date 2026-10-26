"""z3 comp 갈래 2단계: 3단계(양봉 비율 + 추세 판정)와 기존 추세 5단계(일봉)의 보완성.

입력 results/z3comp_grid.npz(z3comp_build.py). 출력 results/z3comp_*.csv·json.
- 일치율: 40분봉마다 5단계(마지막 완성 일봉) × 3단계 교차표(144종목 합), 카파·같은 방향 비율, 귀무 = 3단계 표지 공통 k 이동.
- 뒤 수익(1·5거래일, 다음 봉 시가 → 같은 시각): 날것(%) / 같은 시각 종목 평균을 뺀 값(%) / 같은 시각 종목 순위(%p, −50~+50).
  T0 3단계 혼자, T1 5단계(5칸) 안에서 3단계, T2m 5단계 혼자(상방·극상방 − 하방·극하방), T2 3단계 안에서 5단계.
  표준오차 = 날짜 묶음 + 겹침 보정(Driscoll–Kraay, Bartlett 지연 h+1). 귀무 = 표지만 공통 k일 원형 이동(15 ≤ k ≤ D−15, 같은 시각 칸).
  3단계 귀무 두 가지: 표지 전체 이동(L), 추세 판정 flag만 이동하고 s는 그대로(F).
"""
from __future__ import annotations

import json
import pathlib
import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from engine import HERE, WIN  # noqa: E402

RES = HERE / "results"
REPS = ["ER_K20_th0.3", "FLIP_K40_m3", "EXT_n20_d0.1", "HTF_tfD"]
MIN_XS = 30  # 같은 시각 비교에 필요한 종목 수

g = np.load(RES / "z3comp_grid.npz")
SYMS, FLAGS = list(g["syms"]), list(g["flags"])
DAYS = [date.fromisoformat(x) for x in g["days"]]
S, D, K = g["O"].shape
PRES = g["present"]
SIG = g["s"].astype(np.int64)
FL = g["fl"].astype(bool)
Z5 = g["z5"]
DIDX = np.broadcast_to(np.arange(D)[None, :, None], (S, D, K))
w = WIN["5m"]
SEGD = np.array([0 if w["sel"][0] <= d <= w["sel"][1] else 1 if w["conf"][0] <= d <= w["conf"][1] else -1 for d in DAYS])
SEGS = {"sel": np.flatnonzero(SEGD == 0), "conf": np.flatnonzero(SEGD == 1)}
SHIFTS = list(range(15, D - 15 + 1))
SUBSETS = {"all": PRES, "aw_dst": g["aw_dst"] & PRES, "aw_std": g["aw_std"] & PRES}


def xs_measures(f):
    """같은 시각(날짜·칸) 종목 단면: 평균 뺀 값과 순위(−0.5~0.5)."""
    v = f.reshape(S, D * K)
    n = (~np.isnan(v)).sum(0)
    mu = np.nanmean(v, 0)
    dm = np.where(n >= MIN_XS, v - mu, np.nan)
    rk = pd.DataFrame(v).rank(axis=0).values
    rk = np.where(n >= MIN_XS, (rk - 0.5) / np.maximum(n, 1) - 0.5, np.nan)
    return dm.reshape(S, D, K), rk.reshape(S, D, K)


import warnings  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
YS = {}
for h in (1, 5):
    f = g[f"fwd{h}"]
    dm, rk = xs_measures(f)
    YS[(h, "raw")] = f * 100
    YS[(h, "xs")] = dm * 100
    YS[(h, "rank")] = rk * 100


def z3_code(flag, s, pres):
    """0: −1(하락 추세), 1: 0(애매), 2: +1(상승 추세), −1: 없음."""
    c = np.where(flag, np.where(s == 1, 2, 0), 1)
    return np.where(pres & (s >= 0), c, -1)


Z5CODE = np.where(PRES & ~np.isnan(Z5), np.nan_to_num(Z5).astype(np.int64) + 2, -1)  # 0..4 = −2..2


def roll(a, k):
    return np.roll(a, k, axis=-2)  # 거래일 축, 같은 시각 칸


# ---------------------------------------------------------------------------
# 칸 집계와 회귀
# ---------------------------------------------------------------------------

def tab(la, A, lb, B, y, m):
    v = m & (la >= 0) & (lb >= 0) & ~np.isnan(y)
    idx = DIDX[v] * (A * B) + la[v] * B + lb[v]
    N = np.bincount(idx, minlength=D * A * B).reshape(D, A * B).astype(float)
    Y = np.bincount(idx, weights=y[v], minlength=D * A * B).reshape(D, A * B)
    return N, Y


# 5칸(z5 −2..2) × 3칸(z3 −1,0,+1) = 15칸. 설계 행렬
CELL_A = np.repeat(np.arange(5), 3)
CELL_B = np.tile(np.arange(3), 5)
Z5C = np.array([0, 0, 1, 2, 2])[CELL_A]  # 하방·애매·상방


def design(kind):
    one = np.ones(15)
    plus, zero = (CELL_B == 2).astype(float), (CELL_B == 1).astype(float)
    up, mid = (Z5C == 2).astype(float), (Z5C == 1).astype(float)
    if kind == "T0":
        return np.c_[one, plus, zero], {"p_m": [0, 1, 0], "z_m": [0, 0, 1], "p_z": [0, 1, -1]}
    if kind == "T1":
        fe = np.stack([(CELL_A == a).astype(float) for a in range(5)], 1)
        return np.c_[fe, plus, zero], {"p_m": [0] * 5 + [1, 0], "z_m": [0] * 5 + [0, 1], "p_z": [0] * 5 + [1, -1]}
    if kind == "T2m":
        return np.c_[one, up, mid], {"u_d": [0, 1, 0], "m_d": [0, 0, 1]}
    if kind == "T2":
        fe = np.stack([(CELL_B == b).astype(float) for b in range(3)], 1)
        return np.c_[fe, up, mid], {"u_d": [0, 0, 0, 1, 0], "m_d": [0, 0, 0, 0, 1]}
    if kind == "COMBO":  # 9칸(5단계 3묶음 × 3단계) 평균
        cid = Z5C * 3 + CELL_B
        X = np.stack([(cid == c).astype(float) for c in range(9)], 1)
        e = np.eye(9)
        con = {"bothup_vs_up_m": e[8] - e[6], "bothdn_vs_dn_p": e[0] - e[2], "bothup_vs_bothdn": e[8] - e[0]}
        return X, con
    raise ValueError(kind)


DESIGNS = {k: design(k) for k in ("T0", "T1", "T2m", "T2", "COMBO")}


def ols(N, Y, X, lag=None):
    n = N.sum(0)
    XtX = X.T @ (n[:, None] * X)
    Bi = np.linalg.pinv(XtX)
    beta = Bi @ (X.T @ Y.sum(0))
    if lag is None:
        return beta, None
    fit = X @ beta
    Sd = (Y - N * fit[None, :]) @ X  # 날짜별 점수
    G = int((N.sum(1) > 0).sum())  # 관측이 있는 날짜 수(끝의 뒤 수익 없는 날은 점수 0이라 빼고 센다)
    om = Sd.T @ Sd
    for l in range(1, lag + 1):
        gl = Sd[l:].T @ Sd[:-l]
        om += (1 - l / (lag + 1)) * (gl + gl.T)
    V = Bi @ om @ Bi * G / (G - 1)
    return beta, V


def contrasts(N, Y, kind, lag=None):
    X, con = DESIGNS[kind]
    b, V = ols(N, Y, X, lag)
    out = {}
    colN = X.T @ N.sum(0)  # 열마다 관측 수
    for nm, c in con.items():
        c = np.asarray(c, float)
        if (colN[c != 0] == 0).any():  # 비교할 칸이 비면 값 없음
            out[nm] = (np.nan, np.nan)
            continue
        est = float(c @ b)
        se = float(np.sqrt(c @ V @ c)) if V is not None else np.nan
        out[nm] = (est, se)
    if kind == "COMBO":  # 5단계 묶음 평균 대비(같은 구간 칸 수 가중)
        n9 = np.zeros(9)
        np.add.at(n9, Z5C * 3 + CELL_B, N.sum(0))
        for nm, row, cell in (("bothup_vs_up", 2, 8), ("bothdn_vs_dn", 0, 0)):
            c = np.zeros(9)
            c[row * 3:row * 3 + 3] = -n9[row * 3:row * 3 + 3] / max(n9[row * 3:row * 3 + 3].sum(), 1)
            c[cell] += 1
            out[nm] = (float(c @ b), float(np.sqrt(c @ V @ c)) if V is not None else np.nan)
    return out


def all_stats(z3c, z5c, kinds, with_se, subsets=SUBSETS, measures=("raw", "xs", "rank")):
    """(subset, h, measure, seg, kind, contrast) → (est, se)."""
    res = {}
    for sn, m in subsets.items():
        for h in (1, 5):
            for ms in measures:
                N, Y = tab(z5c, 5, z3c, 3, YS[(h, ms)], m)
                for seg, di in SEGS.items():
                    for kd in kinds:
                        for nm, v in contrasts(N[di], Y[di], kd, (h + 1) if with_se else None).items():
                            res[(sn, h, ms, seg, kd, nm)] = v
    return res


# ---------------------------------------------------------------------------
# 일치율
# ---------------------------------------------------------------------------

def agreement(z3c, z5c, m):
    """z5 3묶음(하·애·상) × z3(−1,0,+1) 교차표와 지표."""
    v = m & (z3c >= 0) & (z5c >= 0)
    a = np.array([0, 0, 1, 2, 2])[z5c[v]]
    b = z3c[v]
    T = np.bincount(a * 3 + b, minlength=9).reshape(3, 3).astype(float)
    n = T.sum()
    po = np.trace(T) / n
    pe = (T.sum(1) @ T.sum(0)) / n ** 2
    kappa = (po - pe) / (1 - pe)
    dd = T[[0, 2]][:, [0, 2]]  # 둘 다 방향 있음
    same = (dd[0, 0] + dd[1, 1]) / dd.sum()
    exp_same = (dd.sum(1)[0] * dd.sum(0)[0] + dd.sum(1)[1] * dd.sum(0)[1]) / dd.sum() ** 2
    return dict(n=n, po=po, pe=pe, kappa=kappa, same_dir=same, same_dir_exp=exp_same,
                p_up_given_p=T[2, 2] / T[:, 2].sum(), p_dn_given_m=T[0, 0] / T[:, 0].sum(), p_mid_given_z=T[1, 1] / T[:, 1].sum(),
                base_up=T[2].sum() / n, base_dn=T[0].sum() / n, base_mid=T[1].sum() / n,
                p_p_given_up=T[2, 2] / T[2].sum(), p_m_given_dn=T[0, 0] / T[0].sum(), p_z_given_mid=T[1, 1] / T[1].sum(),
                base_p=T[:, 2].sum() / n, base_m=T[:, 0].sum() / n, base_z=T[:, 1].sum() / n, share_dir_both=dd.sum() / n), T


def run_lengths(code, pres):
    """종목별 시간 순서(있는 봉만)로 같은 상태가 이어진 길이(40분봉 수) 평균."""
    out = []
    for i in range(S):
        c = code[i][pres[i]]
        c = c[c >= 0]
        if len(c) < 2:
            continue
        ch = np.r_[True, c[1:] != c[:-1]]
        out.append(len(c) / ch.sum())
    return float(np.median(out))


def main():
    classifiers = {nm: z3_code(FL[:, FLAGS.index(nm)], SIG, PRES) for nm in REPS}
    classifiers["S_only"] = z3_code(np.ones_like(PRES), SIG, PRES)  # 추세 판정 없이 원안 상태만(+1 보유·−1 현금)
    seg_mask = {sg: np.isin(DIDX, di) for sg, di in SEGS.items()}
    seg_mask["both"] = seg_mask["sel"] | seg_mask["conf"]

    # 1. 일치율 ------------------------------------------------------------
    agree_rows, cross_rows = [], []
    for nm, z3c in classifiers.items():
        for sg, sm in seg_mask.items():
            a, T = agreement(z3c, Z5CODE, PRES & sm)
            nul = [agreement(roll(z3c, k), Z5CODE, PRES & sm & roll(PRES, k))[0] for k in SHIFTS]
            for key in ("kappa", "same_dir"):
                arr = np.array([x[key] for x in nul])
                a[f"{key}_null_med"] = float(np.median(arr))
                a[f"{key}_null_p05"], a[f"{key}_null_p95"] = np.quantile(arr, [0.05, 0.95])
                a[f"{key}_pct"] = float((arr < a[key]).mean())
            agree_rows.append(dict(clf=nm, seg=sg, **a))
        # 5칸 × 3칸 교차표(전체 구간, 144종목 합)
        v = PRES & seg_mask["both"] & (z3c >= 0)
        z5v = np.where(Z5CODE >= 0, Z5CODE - 2, 9)[v]
        ct = pd.crosstab(pd.Series(z5v, name="z5"), pd.Series(z3c[v] - 1, name="z3"))
        for z5v_, row in ct.iterrows():
            cross_rows.append(dict(clf=nm, z5=("없음" if z5v_ == 9 else int(z5v_)), **{f"z3_{int(c):+d}": int(row[c]) for c in ct.columns}))
    agree = pd.DataFrame(agree_rows)
    agree.to_csv(RES / "z3comp_agree.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cross_rows).to_csv(RES / "z3comp_cross.csv", index=False, encoding="utf-8-sig")
    rl = {nm: run_lengths(c, PRES) for nm, c in classifiers.items()}
    rl["Z5"] = run_lengths(np.where(Z5CODE >= 0, Z5CODE, -1), PRES)
    rl["Z5_3"] = run_lengths(np.where(Z5CODE >= 0, np.array([0, 0, 1, 2, 2])[np.clip(Z5CODE, 0, 4)], -1), PRES)
    print("일치율 완료", flush=True)

    # 2. 뒤 수익 검정 ----------------------------------------------------------
    rows = []
    kinds_z3, kinds_z5 = ("T0", "T1", "COMBO"), ("T2m", "T2")
    for nm, z3c in classifiers.items():
        act = all_stats(z3c, Z5CODE, kinds_z3 + kinds_z5, True)
        nul_L = [all_stats(roll(z3c, k), Z5CODE, kinds_z3, False, {s: m & roll(PRES, k) for s, m in SUBSETS.items()}) for k in SHIFTS]
        nul_Z = [all_stats(z3c, roll(Z5CODE, k), kinds_z5, False, {s: m & roll(PRES, k) for s, m in SUBSETS.items()}) for k in SHIFTS]
        if nm != "S_only":
            fi = FLAGS.index(nm)
            nul_F = [all_stats(z3_code(roll(FL[:, fi], k), SIG, PRES & roll(PRES, k)), Z5CODE, ("T1", "COMBO"), False) for k in SHIFTS]
        else:
            nul_F = None
        for key, (est, se) in act.items():
            sn, h, ms, seg, kd, cn = key
            r = dict(clf=nm, subset=sn, h=h, measure=ms, seg=seg, test=kd, contrast=cn, est=est, se=se, t=est / se if se > 0 else np.nan)
            for tag, nul in (("L", nul_L if kd in kinds_z3 else None), ("Z", nul_Z if kd in kinds_z5 else None),
                             ("F", nul_F if (nul_F is not None and kd in ("T1", "COMBO")) else None)):
                if nul is None:
                    continue
                arr = np.array([x[key][0] for x in nul])
                r[f"{tag}_med"] = float(np.median(arr))
                r[f"{tag}_p05"], r[f"{tag}_p95"] = np.quantile(arr, [0.05, 0.95])
                r[f"{tag}_pct"] = float((arr < est).mean())
            rows.append(r)
        print(nm, "검정 완료", flush=True)
    tests = pd.DataFrame(rows)
    tests.to_csv(RES / "z3comp_tests.csv", index=False, encoding="utf-8-sig")

    # 3. 칸별 평균·빈도(5칸 × 3칸, 그리고 9칸) -----------------------------------
    cell_rows = []
    for nm, z3c in classifiers.items():
        for sn, m in SUBSETS.items():
            for h in (1, 5):
                for ms in ("raw", "xs", "rank"):
                    N, Y = tab(Z5CODE, 5, z3c, 3, YS[(h, ms)], m)
                    for seg, di in SEGS.items():
                        n, y = N[di].sum(0), Y[di].sum(0)
                        for c in range(15):
                            cell_rows.append(dict(clf=nm, subset=sn, h=h, measure=ms, seg=seg, z5=int(CELL_A[c]) - 2,
                                                  z3=int(CELL_B[c]) - 1, n=int(n[c]), share=n[c] / n.sum(), mean=y[c] / n[c] if n[c] else np.nan))
    cells = pd.DataFrame(cell_rows)
    cells.to_csv(RES / "z3comp_cells.csv", index=False, encoding="utf-8-sig")

    # 4. 27개 판정 전체(시험 범위 공개용): 5단계 안 3단계 +1 − −1 ------------------
    rob = []
    sub2 = {k: SUBSETS[k] for k in ("all", "aw_dst")}
    for nm in [f for f in FLAGS if f != "Z5"]:
        z3c = z3_code(FL[:, FLAGS.index(nm)], SIG, PRES)
        act = all_stats(z3c, Z5CODE, ("T0", "T1"), False, sub2, ("xs", "rank"))
        nul = [all_stats(roll(z3c, k), Z5CODE, ("T0", "T1"), False, {s: m & roll(PRES, k) for s, m in sub2.items()}, ("xs", "rank")) for k in SHIFTS]
        for key, (est, _) in act.items():
            if key[5] != "p_m":
                continue
            arr = np.array([x[key][0] for x in nul])
            rob.append(dict(clf=nm, subset=key[0], h=key[1], measure=key[2], seg=key[3], test=key[4], est=est,
                            L_med=float(np.median(arr)), L_p95=float(np.quantile(arr, 0.95)), L_pct=float((arr < est).mean())))
    rob = pd.DataFrame(rob)
    rob.to_csv(RES / "z3comp_robust27.csv", index=False, encoding="utf-8-sig")
    print("27개 완료", flush=True)

    meta = dict(S=S, D=D, shifts=[SHIFTS[0], SHIFTS[-1], len(SHIFTS)], days_sel=len(SEGS["sel"]), days_conf=len(SEGS["conf"]),
                bars=int(PRES.sum()), z5_missing_syms=[SYMS[i] for i in range(S) if (PRES[i] & np.isnan(Z5[i])).all() or np.isnan(Z5[i][PRES[i]]).mean() > 0.9],
                run_length_bars_median=rl, n_obs={f"h{h}": int((~np.isnan(g[f'fwd{h}']) & PRES).sum()) for h in (1, 5)},
                n_xs={f"h{h}": int((~np.isnan(YS[(h, 'rank')])).sum()) for h in (1, 5)})
    (RES / "z3comp_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
