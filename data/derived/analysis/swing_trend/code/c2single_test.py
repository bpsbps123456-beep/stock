"""BRIEF2 시험 1(단독): 단서 18 × type 2 × 결과 3(y1·y10·y20) = 108칸.

py code/c2single_test.py → results/c2single_cells.csv · c2single_family.json · c2single_terciles.csv
                           · c2single_vsplit.csv · c2single_within.csv · c2single_tables.txt

- 3분위 경계: P1 사건(같은 det·같은 type)으로만 정한다. 아래 = f ≤ q1/3, 위 = f > q2/3. 0/1 단서(M3a·S6)는 1 − 0.
- 차이 = 위 − 아래 평균 결과(로그). 단순 % 판은 ys = t × (exp(t × y) − 1)(매수 실제 손익, 매도 피한 손실).
- 날짜 묶음 귀무(주): 같은 구간·같은 type의 사건 날짜 U(D개)를 원형으로 k칸 민다. 날짜 U[i]의 사건은 날짜 U[(i+k) mod D]의
  사건 단서 벡터(18개 통째)를 가져온다. 같은 종목이 그 날짜에 있으면 그 사건, 없으면 그 날짜 사건에서 무작위(섞어 돌려 쓰기).
  k = 15 … min_type(D) − 15 전부(두 type·108칸 공통 k), k마다 무작위 추출 R_DRAW회.
  결과·날짜 군집·단서의 날짜 간 자기상관은 그대로 두고 단서와 결과의 연결만 끊는다.
- 보조 귀무: 날짜 블록 순열 N_PERM회(같은 종목 맞추기·무작위 추출 규칙 동일). 날짜 간 자기상관을 끊으므로 덜 보수적이다.
- 통과: s = sign(P1 차이), T = s × P2 차이. T > 0 이고 T > 귀무 s × P2 차이의 95분위.
- 계열 귀무: 귀무 반복마다 108칸 통과 수를 같은 문턱으로 센다(공통 k라 칸 사이 의존을 보존).
- 사전 등록 밖(표시): 3분위 평균 결과 표, P2 'V자'(y1 < 0) 가르기 AUC, 종목 단서의 같은 날 안 비교(날짜 평균 뺀 y1).
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = pathlib.Path(__file__).resolve().parent.parent
RES = HERE / "results"
FEATURES = ["M1", "M2", "M3a", "M3b", "M4a", "M4b", "M4c", "M5",
            "S1", "S2", "S3a", "S3b", "S4", "S5", "S6", "S7", "S8", "S9"]
STOCK = [f for f in FEATURES if f.startswith("S")]
OUTS = ["y1", "y10", "y20"]
PERS = ["P1", "P2", "Y25", "Y24", "Y23"]
TYPES = (1, -1)
KMIN, R_DRAW, N_PERM, N_WITHIN = 15, 20, 200, 500
SEED = 20261025


def cuts_of(x):
    u = np.unique(x[~np.isnan(x)])
    if set(u.tolist()) <= {0.0, 1.0}:
        return (0.5, 0.5, True)
    a, b = np.quantile(x[~np.isnan(x)], [1 / 3, 2 / 3])
    return (float(a), float(b), False)


def groups(x, cut):
    a, b, binary = cut
    if binary:
        return x == 1, x == 0
    return x > b, x <= a


def tercile_label(x, cut):
    a, b, binary = cut
    if binary:
        return np.where(x == 1, "1", "0")
    return np.where(x <= a, "아래", np.where(x > b, "위", "가운데"))


def mdiff(y, top, bot):
    ok = ~np.isnan(y)
    t, b = top & ok, bot & ok
    if not t.any() or not b.any():
        return np.nan
    return y[t].mean() - y[b].mean()


def auc(score, lab):
    """P(score_V > score_nonV), 동점 0.5. lab True = V."""
    ok = ~np.isnan(score)
    s, l = score[ok], lab[ok]
    n1, n0 = l.sum(), (~l).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[l].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


class Blocks:
    """한 구간·한 type 사건의 날짜 묶음."""

    def __init__(self, sub: pd.DataFrame):
        self.dates = sub.date.values
        self.U = np.array(sorted(set(self.dates)))
        self.D = len(self.U)
        pos = {d: i for i, d in enumerate(self.U)}
        self.di = np.array([pos[d] for d in self.dates])
        self.members = [np.flatnonzero(self.di == i) for i in range(self.D)]
        self.sym = sub.sym.values
        self.symidx = [{s: j for s, j in zip(self.sym[m], m)} for m in self.members]

    def mapping(self, target, rng):
        m = np.empty(len(self.dates), np.int64)
        for i in range(self.D):
            src, tgt = self.members[i], self.members[target[i]]
            ts = self.symidx[target[i]]
            pool = rng.permutation(tgt)
            q = 0
            for e in src:
                j = ts.get(self.sym[e])
                if j is None:
                    j = pool[q % len(pool)]
                    q += 1
                m[e] = j
        return m


def cell_stats(F, Y, cuts, m=None):
    """F: dict 단서 → 배열, Y: dict 결과 → 배열. m이 있으면 단서를 m으로 재배열. 반환 (18, 3)."""
    out = np.full((len(FEATURES), len(OUTS)), np.nan)
    for i, f in enumerate(FEATURES):
        x = F[f] if m is None else F[f][m]
        top, bot = groups(x, cuts[f])
        for j, o in enumerate(OUTS):
            out[i, j] = mdiff(Y[o], top, bot)
    return out


def auc_stats(F, vlab, m=None):
    return np.array([auc(F[f] if m is None else F[f][m], vlab) for f in FEATURES])


def run_det(det: str, rng) -> dict:
    ev = pd.read_parquet(RES / f"events_{det}.parquet")
    ev["date"] = pd.to_datetime(ev.date).dt.date
    subs, cuts = {}, {}
    for t in TYPES:
        p1 = ev[(ev.period == "P1") & (ev.type == t)]
        cuts[t] = {f: cuts_of(p1[f].values.astype(float)) for f in FEATURES}
        for per in PERS:
            subs[(per, t)] = ev[(ev.period == per) & (ev.type == t)].reset_index(drop=True)

    def FY(sub, simple=False):
        F = {f: sub[f].values.astype(float) for f in FEATURES}
        Y = {o: sub[o].values.astype(float) for o in OUTS}
        if simple:
            t = sub.type.values
            Y = {o: t * (np.exp(t * y) - 1) for o, y in Y.items()}
        return F, Y

    # 실제 차이
    actual = {}      # (per, t) → (18, 3) 로그
    actual_s = {}    # 단순 %
    for (per, t), sub in subs.items():
        F, Y = FY(sub)
        actual[(per, t)] = cell_stats(F, Y, cuts[t])
        F, Ys = FY(sub, simple=True)
        actual_s[(per, t)] = cell_stats(F, Ys, cuts[t])

    # 귀무(P2 주, Y25 표시용)
    nulls, null_auc, null_desc = {}, {}, {}
    for per in ("P2", "Y25"):
        B = {t: Blocks(subs[(per, t)]) for t in TYPES}
        Dmin = min(B[t].D for t in TYPES)
        ks = list(range(KMIN, Dmin - KMIN + 1))
        for kind in ("shift", "perm"):
            reps = [(k, r) for k in ks for r in range(R_DRAW)] if kind == "shift" else [(None, r) for r in range(N_PERM)]
            arr = np.full((len(reps), 2, len(FEATURES), len(OUTS)), np.nan)
            aarr = np.full((len(reps), 2, len(FEATURES)), np.nan)
            for ri, (k, r) in enumerate(reps):
                for ti, t in enumerate(TYPES):
                    b = B[t]
                    target = (np.arange(b.D) + k) % b.D if kind == "shift" else rng.permutation(b.D)
                    m = b.mapping(target, rng)
                    F, Y = FY(subs[(per, t)])
                    arr[ri, ti] = cell_stats(F, Y, cuts[t], m)
                    if per == "P2":
                        aarr[ri, ti] = auc_stats(F, Y["y1"] < 0, m)
            nulls[(per, kind)] = arr
            null_auc[(per, kind)] = aarr
            null_desc[(per, kind)] = dict(reps=len(reps), k=[ks[0], ks[-1]] if kind == "shift" else None,
                                          D={str(t): B[t].D for t in TYPES})

    # 칸별 판정
    rows = []
    sgn = np.zeros((2, len(FEATURES), len(OUTS)))
    for ti, t in enumerate(TYPES):
        sgn[ti] = np.sign(actual[("P1", t)])
    # P2 위·아래 3분위 날짜 수(작은 쪽)와 Y25 부호 일치
    mind = np.zeros((2, len(FEATURES), len(OUTS)), int)
    for ti, t in enumerate(TYPES):
        sub2 = subs[("P2", t)]
        for i, f in enumerate(FEATURES):
            top, bot = groups(sub2[f].values.astype(float), cuts[t][f])
            for j, o in enumerate(OUTS):
                ok_ = ~np.isnan(sub2[o].values.astype(float))
                mind[ti, i, j] = min(sub2.date[top & ok_].nunique(), sub2.date[bot & ok_].nunique())
    y25same = np.stack([(np.sign(actual[("Y25", t)]) == sgn[ti]) & (sgn[ti] != 0) for ti, t in enumerate(TYPES)])
    is_mkt = np.zeros((2, len(FEATURES), len(OUTS)), bool)
    is_mkt[:, [i for i, f in enumerate(FEATURES) if f.startswith("M")], :] = True
    is_y1 = np.zeros_like(is_mkt)
    is_y1[:, :, 0] = True
    subsets = {"전체": np.ones_like(is_mkt), "y1": is_y1, "시장 M": is_mkt, "종목 S": ~is_mkt, "P2 날짜 양쪽 10일 이상": mind >= 10}

    thr, passes, fam = {}, {}, {}
    for per in ("P2", "Y25"):
        act = np.stack([actual[(per, t)] for t in TYPES])
        for kind in ("shift", "perm"):
            T = sgn * act
            Tn = sgn[None] * nulls[(per, kind)]
            q95 = np.nanquantile(Tn, 0.95, axis=0)
            p = np.nanmean(Tn >= T[None], axis=0)
            ok = (sgn != 0) & (T > 0) & (T > q95)
            npass = (Tn > 0) & (Tn > q95[None]) & (sgn[None] != 0)
            thr[(per, kind)] = (q95, p)
            passes[(per, kind)] = ok
            nd = null_desc[(per, kind)]
            fam[(per, kind)] = dict(reps=nd["reps"], k=nd["k"], D=nd["D"],
                                    distinct=(nd["k"][1] - nd["k"][0] + 1) if kind == "shift" else nd["reps"])
            for nm, msk in subsets.items():
                a = int((ok & msk).sum())
                nl = (npass & msk[None]).reshape(len(Tn), -1).sum(axis=1)
                fam[(per, kind)][nm] = dict(actual=a, cells=int(msk.sum()), null_mean=float(nl.mean()),
                                            null_q95=float(np.quantile(nl, 0.95)), null_max=int(nl.max()), p=float(np.mean(nl >= a)))
            if per == "P2":
                fam[(per, kind)]["Y25 같은 부호"] = dict(pass_same=int((ok & y25same).sum()), passes=int(ok.sum()),
                                                      base_rate=float(y25same.mean()))
    np.savez_compressed(RES / f"c2single_null_{det}.npz", **{f"{per}_{kind}": nulls[(per, kind)] for (per, kind) in nulls},
                        sign_P1=sgn, mind=mind)

    for ti, t in enumerate(TYPES):
        sub2 = subs[("P2", t)]
        for i, f in enumerate(FEATURES):
            top, bot = groups(sub2[f].values.astype(float), cuts[t][f])
            for j, o in enumerate(OUTS):
                y = sub2[o].values.astype(float)
                ok = ~np.isnan(y)
                s = sgn[ti, i, j]
                row = dict(det=det, type=t, feature=f, outcome=o, cut_lo=cuts[t][f][0], cut_hi=cuts[t][f][1], binary=cuts[t][f][2],
                           **{f"{per}_diff": actual[(per, t)][i, j] for per in PERS},
                           P2_diff_simple=actual_s[("P2", t)][i, j], Y25_diff_simple=actual_s[("Y25", t)][i, j],
                           P2_n_top=int((top & ok).sum()), P2_n_bot=int((bot & ok).sum()),
                           P2_dates_top=int(sub2.date[top & ok].nunique()), P2_dates_bot=int(sub2.date[bot & ok].nunique()),
                           sign_P1=int(s),
                           P2_q95_shift=thr[("P2", "shift")][0][ti, i, j], P2_p_shift=thr[("P2", "shift")][1][ti, i, j],
                           pass_shift=bool(passes[("P2", "shift")][ti, i, j]),
                           P2_q95_perm=thr[("P2", "perm")][0][ti, i, j], P2_p_perm=thr[("P2", "perm")][1][ti, i, j],
                           pass_perm=bool(passes[("P2", "perm")][ti, i, j]),
                           Y25_same_sign=bool(np.sign(actual[("Y25", t)][i, j]) == s and s != 0),
                           Y25_p_shift=thr[("Y25", "shift")][1][ti, i, j],
                           Y24_same_sign=bool(np.sign(actual[("Y24", t)][i, j]) == s and s != 0),
                           Y23_same_sign=bool(np.sign(actual[("Y23", t)][i, j]) == s and s != 0))
                rows.append(row)
    cells = pd.DataFrame(rows)

    # 3분위 평균 결과(표시)
    trows = []
    for (per, t), sub in subs.items():
        if per not in ("P1", "P2", "Y25"):
            continue
        for f in FEATURES:
            lab = tercile_label(sub[f].values.astype(float), cuts[t][f])
            for g in (["0", "1"] if cuts[t][f][2] else ["아래", "가운데", "위"]):
                s_ = sub[lab == g]
                trows.append(dict(det=det, period=per, type=t, feature=f, tercile=g, n=len(s_), dates=s_.date.nunique(),
                                  y1=s_.y1.mean(), y10=s_.y10.mean(), y20=s_.y20.mean(), v_rate=(s_.y1 < 0).mean(),
                                  y1_simple=(t * (np.exp(t * s_.y1) - 1)).mean()))
    terc = pd.DataFrame(trows)

    # P2 V자(y1 < 0) 가르기(표시)
    vrows = []
    for ti, t in enumerate(TYPES):
        aucs = {per: auc_stats({f: subs[(per, t)][f].values.astype(float) for f in FEATURES}, subs[(per, t)].y1.values < 0)
                for per in ("P1", "P2", "Y25")}
        an = null_auc[("P2", "shift")][:, ti]
        ap = null_auc[("P2", "perm")][:, ti]
        sub2 = subs[("P2", t)]
        for i, f in enumerate(FEATURES):
            a = aucs["P2"][i]
            top, bot = groups(sub2[f].values.astype(float), cuts[t][f])
            v = sub2.y1.values < 0
            vrows.append(dict(det=det, type=t, feature=f, P1_auc=aucs["P1"][i], P2_auc=a, Y25_auc=aucs["Y25"][i],
                              P2_p_shift=float(np.mean(np.abs(an[:, i] - 0.5) >= abs(a - 0.5))),
                              P2_p_perm=float(np.mean(np.abs(ap[:, i] - 0.5) >= abs(a - 0.5))),
                              P2_v_top=v[top].mean() if top.any() else np.nan, P2_v_bot=v[bot].mean() if bot.any() else np.nan,
                              P2_v_all=v.mean(), P2_n=len(v)))
    vsplit = pd.DataFrame(vrows)

    # 종목 단서의 같은 날 안 비교(표시): y1 − 그날(같은 type) 평균, 귀무는 같은 날 안에서 단서 섞기
    wrows = []
    for ti, t in enumerate(TYPES):
        for per in ("P1", "P2", "Y25"):
            sub = subs[(per, t)]
            yd = (sub.y1 - sub.groupby("date").y1.transform("mean")).values
            multi = sub.groupby("date").y1.transform("size").values > 1
            b = Blocks(sub)
            for f in STOCK:
                x = sub[f].values.astype(float)
                top, bot = groups(x, cuts[t][f])
                act = mdiff(np.where(multi, yd, np.nan), top, bot)
                nl = np.empty(N_WITHIN)
                for r in range(N_WITHIN):
                    xp = x.copy()
                    for mem in b.members:
                        if len(mem) > 1:
                            xp[mem] = x[rng.permutation(mem)]
                    tp, bt = groups(xp, cuts[t][f])
                    nl[r] = mdiff(np.where(multi, yd, np.nan), tp, bt)
                wrows.append(dict(det=det, type=t, period=per, feature=f, diff=act,
                                  null_q025=np.quantile(nl, 0.025), null_q975=np.quantile(nl, 0.975),
                                  p_two=float(np.mean(np.abs(nl) >= abs(act)))))
    within = pd.DataFrame(wrows)

    meta = {kind: fam[("P2", kind)] for kind in ("shift", "perm")}
    meta["Y25_shift"] = fam[("Y25", "shift")]
    meta["events"] = {f"{per}_{t}": dict(n=len(subs[(per, t)]), dates=int(subs[(per, t)].date.nunique())) for (per, t) in subs}
    meta["symbols"] = int(ev.sym.nunique())
    return dict(cells=cells, terc=terc, vsplit=vsplit, within=within, meta=meta)


def pct(x, d=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:+.{d}f}"


def tables(res: dict) -> str:
    L = []
    for det, r in res.items():
        c, m = r["cells"], r["meta"]
        L.append(f"=== {det} (종목 {m['symbols']}) ===")
        L.append("사건(개/날짜): " + ", ".join(f"{k} {v['n']}/{v['dates']}" for k, v in m["events"].items()))
        for key, lab in (("shift", "P2 원형 이동"), ("perm", "P2 날짜 블록 순열"), ("Y25_shift", "Y25 원형 이동(표시)")):
            f = m[key]
            L.append(f"[{lab}] 반복 {f['reps']}(서로 다른 날짜 배치 {f['distinct']}) k {f['k']} 사건 날짜 D {f['D']} · 계열 통과 수(실제 / 귀무 평균 · 95분위 · 최대 · p, p 해상도 1/{f['distinct']})")
            for nm in ("전체", "y1", "시장 M", "종목 S", "P2 날짜 양쪽 10일 이상"):
                g = f[nm]
                L.append(f"   {nm:14} {g['actual']:3}/{g['cells']:3} | {g['null_mean']:.2f} · {g['null_q95']:.0f} · {g['null_max']} · p {g['p']:.3f}")
            if "Y25 같은 부호" in f:
                g = f["Y25 같은 부호"]
                L.append(f"   통과 중 Y25 같은 부호 {g['pass_same']}/{g['passes']} (108칸 전체 비율 {g['base_rate']:.0%})")
        L.append("")
        for t in TYPES:
            L.append(f"-- type {'+1 매수 확정' if t == 1 else '−1 매도 확정'} · 위 − 아래 평균 결과, 로그 %p · P2 단순 %p · 귀무 95분위(P1 방향) · 통과 ✓(shift)/·(perm) --")
            L.append(f"{'단서':5} | {'결과':4} | {'P1':>6} {'P2':>6} {'P2단순':>6} {'q95':>5} {'p':>5} {'Y25':>6} | 통과 | Y25같음 | P2 날짜(위/아래)")
            cc = c[c.type == t]
            for _, x in cc.iterrows():
                mark = ("✓" if x.pass_shift else " ") + ("·" if x.pass_perm else " ")
                L.append(f"{x.feature:5} | {x.outcome:4} | {pct(x.P1_diff):>6} {pct(x.P2_diff):>6} {pct(x.P2_diff_simple):>6} {pct(x.P2_q95_shift):>5} {x.P2_p_shift:5.2f} {pct(x.Y25_diff):>6} | {mark:4} | {'예' if x.Y25_same_sign else '아니오':4} | {x.P2_dates_top}/{x.P2_dates_bot}")
            L.append("")
        # 3분위 평균 결과(y1)
        tr = r["terc"]
        L.append("-- (표시) 3분위 평균 y1(로그 %) [사건 수/날짜 수], 경계는 P1 --")
        for t in TYPES:
            L.append(f"type {t}: {'단서':4} | P2 아래 · 가운데 · 위 | Y25 아래 · 가운데 · 위")
            for f in FEATURES:
                parts = []
                for per in ("P2", "Y25"):
                    q = tr[(tr.type == t) & (tr.feature == f) & (tr.period == per)]
                    parts.append(" · ".join(f"{pct(x.y1)} [{x.n}/{x.dates}]" for _, x in q.iterrows()))
                L.append(f"   {f:4} | {parts[0]} | {parts[1]}")
        L.append("")
        # V자 가르기
        v = r["vsplit"].copy()
        v["gap"] = (v.P2_auc - 0.5).abs()
        L.append("-- (표시) P2 V자(y1<0) 가르기: AUC = P(V 사건 단서 > 이어진 사건 단서), 날짜 묶음 귀무 양쪽 p --")
        for t in TYPES:
            vv = v[v.type == t].sort_values("gap", ascending=False)
            L.append(f"type {t}: V 비율 {vv.P2_v_all.iloc[0]:.0%} (사건 {vv.P2_n.iloc[0]})")
            for _, x in vv.head(6).iterrows():
                L.append(f"  {x.feature:4} AUC P1 {x.P1_auc:.3f} P2 {x.P2_auc:.3f} Y25 {x.Y25_auc:.3f} | p shift {x.P2_p_shift:.2f} perm {x.P2_p_perm:.3f} | V 비율 위 {x.P2_v_top:.0%} 아래 {x.P2_v_bot:.0%}")
        L.append("")
        w = r["within"]
        L.append("-- (표시) 종목 단서 같은 날 안 비교: 위 − 아래 (y1 − 그날 평균), 로그 %p, 같은 날 섞기 귀무 양쪽 p --")
        for t in TYPES:
            ww = w[w.type == t].pivot(index="feature", columns="period", values=["diff", "p_two"])
            L.append(f"type {t}")
            for f in STOCK:
                L.append(f"  {f:4} P1 {pct(ww.loc[f, ('diff', 'P1')]):>6} (p {ww.loc[f, ('p_two', 'P1')]:.2f}) P2 {pct(ww.loc[f, ('diff', 'P2')]):>6} (p {ww.loc[f, ('p_two', 'P2')]:.2f}) Y25 {pct(ww.loc[f, ('diff', 'Y25')]):>6} (p {ww.loc[f, ('p_two', 'Y25')]:.2f})")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    res = {det: run_det(det, rng) for det in ("ZZA3", "ZZP10")}
    pd.concat([r["cells"] for r in res.values()]).to_csv(RES / "c2single_cells.csv", index=False, encoding="utf-8-sig")
    pd.concat([r["terc"] for r in res.values()]).to_csv(RES / "c2single_terciles.csv", index=False, encoding="utf-8-sig")
    pd.concat([r["vsplit"] for r in res.values()]).to_csv(RES / "c2single_vsplit.csv", index=False, encoding="utf-8-sig")
    pd.concat([r["within"] for r in res.values()]).to_csv(RES / "c2single_within.csv", index=False, encoding="utf-8-sig")
    (RES / "c2single_family.json").write_text(json.dumps({d: r["meta"] for d, r in res.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
    txt = tables(res)
    (RES / "c2single_tables.txt").write_text(txt, encoding="utf-8")
    print(txt)
