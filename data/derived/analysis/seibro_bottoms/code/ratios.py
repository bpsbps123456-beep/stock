"""2차 비율 조합 탐색(BRIEF2.md). 매수·매도·순매수 비율과 종목 간 상대 비율 580개 × 창 4개를
P(250세션 백분위)·Eq(제곱 메아리 잔차)로 바꿔, 2024 이후 바닥의 공통점과 우연 기대를 잰다.

실행(연구 폴더에서): py -B code/ratios.py [--reps 1000]
출력(results/ratios/): features.csv(문법), cells.csv(칸별 통계), common_null.csv, select_2425.csv,
  check_2026.csv, combos.csv, intuitive.csv, summary.md
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

STUDY = Path(__file__).resolve().parents[1]
DER = STUDY / "derived"
OUT = STUDY / "results" / "ratios"
START = pd.Timestamp("2024-01-01")
NAMED = ["SOXL", "SOXS", "TQQQ", "SQQQ", "NVDA", "TSLA"]
GROUPS = NAMED + ["LEVSGL", "LEV3", "LEV2IDX", "INV", "INC", "BOND", "SEMI1", "IDX1", "STK1", "OTH"]
LEV_LONG = ["SOXL", "TQQQ", "LEVSGL", "LEV3", "LEV2IDX"]
INVS = ["SOXS", "SQQQ", "INV"]
WINS = {"W1": (15, -6), "W2": (5, -1), "W3": (1, 0), "W4": (5, 5)}
RNG = np.random.default_rng(20261104)
GNAME = {"SOXL": "SOXL", "SOXS": "SOXS", "TQQQ": "TQQQ", "SQQQ": "SQQQ", "NVDA": "NVDA", "TSLA": "TSLA",
         "LEVSGL": "단일종목 레버", "LEV3": "그 밖 3배 롱", "LEV2IDX": "그 밖 지수 레버", "INV": "그 밖 인버스",
         "INC": "인컴", "BOND": "채권·현금", "SEMI1": "그 밖 1배 반도체", "IDX1": "1배 지수 ETF",
         "STK1": "그 밖 1배 개별주", "OTH": "나머지"}


# ---------- 1. 묶음과 창 합 ----------
def group_of(r) -> str:
    if r["product"] in NAMED:
        return r["product"]
    if r["lev_long"] and r["single"]:
        return "LEVSGL"
    if r["EQ3L"]:
        return "LEV3"
    if r["lev_long"] and r["IDX"]:
        return "LEV2IDX"
    if r["EQ_INV"]:
        return "INV"
    for flag, g in (("R5", "INC"), ("R3", "BOND"), ("R1", "SEMI1"), ("R2", "IDX1"), ("R4", "STK1")):
        if r[flag]:
            return g
    return "OTH"


def winsum(X: np.ndarray, L: int) -> np.ndarray:
    cs = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, axis=0)])
    out = np.full(X.shape, np.nan)
    out[L - 1:] = cs[L:] - cs[:-L]
    return out


def load():
    f = pd.read_parquet(DER / "flows_trade_product.parquet")
    f = f[~f["dummy"]].copy()
    f["date"] = pd.to_datetime(f["date"])
    dates = pd.DatetimeIndex(sorted(f["date"].unique()))
    flags = f.groupby("product")[["lev_long", "single", "EQ3L", "IDX", "EQ_INV", "R1", "R2", "R3", "R4", "R5"]].first()
    pg = flags.reset_index().apply(group_of, axis=1)
    pg.index = flags.index
    prods = list(pg.index)
    pidx = {p: i for i, p in enumerate(prods)}
    di, pi = dates.get_indexer(f["date"]), f["product"].map(pidx).values
    Bm, Sm = np.zeros((len(dates), len(prods))), np.zeros((len(dates), len(prods)))
    np.add.at(Bm, (di, pi), f["buy"].values)
    np.add.at(Sm, (di, pi), f["sell"].values)
    G = np.zeros((len(prods), len(GROUPS)))
    for p, g in pg.items():
        G[pidx[p], GROUPS.index(g)] = 1.0
    return dates, prods, pg, Bm, Sm, G


# ---------- 2. 비율 문법 ----------
def features(Bm, Sm, G, prods, L: int) -> pd.DataFrame:
    Bp, Sp = winsum(Bm, L), winsum(Sm, L)
    Np = Bp - Sp
    Bg, Sg = Bp @ G, Sp @ G
    Ng, Tg = Bg - Sg, Bg + Sg
    Ball, Sall = Bp.sum(1), Sp.sum(1)
    Tall = Ball + Sall
    pos, neg = np.clip(Np, 0, None), np.clip(-Np, 0, None)
    Pall, Mall = pos.sum(1), neg.sum(1)
    gross = Pall + Mall
    cols = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        imb = np.where(Tg > 0, Ng / Tg, np.nan)
        for j, g in enumerate(GROUPS):
            live = Tg[:, j] > 0  # T = 0(목록 밖 절단)이면 결측(BRIEF2 §3)
            cols[f"A:imb:{g}"] = imb[:, j]
            cols[f"B:tshare:{g}"] = np.where(live, Tg[:, j] / Tall, np.nan)
            cols[f"B:bshare:{g}"] = np.where(live, Bg[:, j] / Ball, np.nan)
            cols[f"B:sshare:{g}"] = np.where(live, Sg[:, j] / Sall, np.nan)
            cols[f"B:nshare:{g}"] = np.where(live, Ng[:, j] / gross, np.nan)
        for (a, g), (b, h) in itertools.combinations(enumerate(GROUPS), 2):
            both = (Tg[:, a] > 0) & (Tg[:, b] > 0)
            cols[f"C:dimb:{g}|{h}"] = imb[:, a] - imb[:, b]
            cols[f"C:ltr:{g}|{h}"] = np.where(both, np.log(Tg[:, a] / Tg[:, b]), np.nan)
            cols[f"C:lbr:{g}|{h}"] = np.where((Bg[:, a] > 0) & (Bg[:, b] > 0), np.log(Bg[:, a] / Bg[:, b]), np.nan)
            cols[f"C:nrel:{g}|{h}"] = np.where(both, (Ng[:, a] - Ng[:, b]) / (Tg[:, a] + Tg[:, b]), np.nan)
        npos, nneg = (Np > 0).sum(1), (Np < 0).sum(1)
        cols["D:fund"] = np.where(Pall > 0, Mall / Pall, np.nan)
        cols["D:breadth"] = np.where(npos + nneg > 0, npos / (npos + nneg), np.nan)
        cols["D:top1pos"] = np.where(Pall > 0, pos.max(1) / Pall, np.nan)
        cols["D:hhipos"] = np.where(Pall > 0, ((pos / Pall[:, None]) ** 2).sum(1), np.nan)
        cols["D:top1neg"] = np.where(Mall > 0, neg.max(1) / Mall, np.nan)
        cols["D:imball"] = (Ball - Sall) / Tall
        lj, ij = [GROUPS.index(g) for g in LEV_LONG], [GROUPS.index(g) for g in INVS]
        cols["D:levnet"] = Ng[:, lj].sum(1) / gross
        bl, bi = Bg[:, lj].sum(1), Bg[:, ij].sum(1)
        cols["D:bull"] = np.where(bl + bi > 0, bl / (bl + bi), np.nan)
        nz, sz = Np != 0, Sp > 0
        nN, nS = nz.sum(1), sz.sum(1)
        for name in NAMED:
            k = prods.index(name)
            x = Np[:, k]
            rn = (np.where(nz, Np, -np.inf) > x[:, None]).sum(1) + 1
            cols[f"E:rN:{name}"] = np.where(nz[:, k], rn / nN, np.nan)
            rs = (np.where(sz, Sp, -np.inf) > Sp[:, k][:, None]).sum(1) + 1
            cols[f"E:rS:{name}"] = np.where(sz[:, k], rs / nS, np.nan)
    out = pd.DataFrame(cols)
    out[np.isinf(out)] = np.nan
    return out


# ---------- 3. 정규화 ----------
def rolling_pct(X: np.ndarray, w: int = 250, mn: int = 120) -> tuple[np.ndarray, np.ndarray]:
    n, m = X.shape
    P, C = np.full((n, m), np.nan), np.zeros((n, m))
    for t in range(n):
        W = X[max(0, t - w + 1):t + 1]
        x = X[t]
        cnt = (~np.isnan(W)).sum(0)
        p = 100.0 * ((W < x).sum(0) + 0.5 * (W == x).sum(0)) / np.maximum(cnt, 1)
        ok = (cnt >= mn) & ~np.isnan(x)
        P[t, ok] = p[ok]
        C[t] = cnt
    return P, C


def echo_resid(P: np.ndarray, C: np.ndarray, X: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    q = np.clip(P / 100.0, 0.5 / np.maximum(C, 1), 1 - 0.5 / np.maximum(C, 1))
    Y = stats.norm.ppf(q)
    Z = np.full(P.shape, np.nan)
    xok = np.isfinite(X).all(1)
    for j in range(P.shape[1]):
        m = fit_mask & xok & np.isfinite(Y[:, j])
        if m.sum() < 120:
            continue
        beta, *_ = np.linalg.lstsq(X[m], Y[m, j], rcond=None)
        r = Y[:, j] - X @ beta
        sd = np.std(r[m], ddof=X.shape[1])
        use = xok & np.isfinite(Y[:, j]) & (np.arange(len(Y)) >= np.flatnonzero(fit_mask)[0])
        Z[use, j] = r[use] / sd
    return Z


def design(pf: pd.DataFrame, L: int) -> np.ndarray:
    ret, pre = pf[f"ret_{L}"].values, pf[f"pre20_{L}"].values
    dd, rv = pf["dd60"].values, pf["rv20"].values
    reg = pd.get_dummies(pf["regime"].astype(str), drop_first=True).values.astype(float)
    return np.column_stack([np.ones(len(pf)), ret, ret ** 2, pre, dd, dd ** 2, rv, reg])


# ---------- 4. 사건 ----------
def event_sets(ev: dict, dates: pd.DatetimeIndex, pf: pd.DataFrame) -> dict:
    pos = {d: i for i, d in enumerate(dates)}
    b10 = [pd.Timestamp(x["bottom"]) for x in ev["B10_events"] if pd.Timestamp(x["bottom"]) >= START]
    b15 = [pd.Timestamp(x["bottom"]) for x in ev["B10_events"] if x["B15"] and pd.Timestamp(x["bottom"]) >= START]
    fd = [pd.Timestamp(x["date"]) for x in ev["F"] if pd.Timestamp(x["date"]) >= START and x.get("has_flows", True)]
    fd = [d for d in fd if d in pos]
    lr = pf.set_index("date")["lr1"].reindex(dates)
    sub = lr[(lr.index >= START)].dropna()
    cut = sub.quantile(0.10)
    near = set()
    for b in b10:
        i = pos[b]
        near.update(range(i - 5, i + 6))
    dd = [d for d, v in sub.items() if v <= cut and pos[d] not in near and d not in set(fd)]
    idx = lambda xs: np.array([pos[d] for d in xs])
    return {"B10": idx(b10), "B15": idx(b15), "F": idx(fd), "D": idx(dd),
            "B10_2425": idx([d for d in b10 if d.year < 2026]), "B10_26": idx([d for d in b10 if d.year == 2026]),
            "F_2425": idx([d for d in fd if d.year < 2026]), "F_26": idx([d for d in fd if d.year == 2026]),
            "D_2425": idx([d for d in dd if d.year < 2026]), "D_26": idx([d for d in dd if d.year == 2026]),
            "names": {"B10": [d.date().isoformat() for d in b10], "F": [d.date().isoformat() for d in fd],
                      "D": [d.date().isoformat() for d in dd]}}


def placed(V: np.ndarray, idx: np.ndarray, off: int) -> np.ndarray:
    """사건 날 idx(k = 0)에 창 끝 오프셋을 더한 행. 범위 밖은 NaN."""
    r = idx + off
    out = np.full((len(idx), V.shape[1]), np.nan)
    ok = (r >= 0) & (r < V.shape[0])
    out[ok] = V[r[ok]]
    return out


# ---------- 5. 칸 통계 ----------
def sides(v: np.ndarray, norm: str):
    """사건 × 칸 값 → (높은 쪽 표시, 낮은 쪽 표시, 결측 아님)."""
    ok = np.isfinite(v)
    if norm == "P":
        return ok & (v >= 80), ok & (v <= 20), ok
    return ok & (v > 0), ok & (v < 0), ok


def common_pass(vB: np.ndarray, norm: str, k: int, zmin: float | None = 0.5):
    hi, lo, ok = sides(vB, norm)
    nh, nl = hi.sum(0), lo.sum(0)
    side = np.where(nh >= nl, 1, -1)
    cnt = np.where(side == 1, nh, nl)
    passed = cnt >= k
    if norm == "E" and zmin is not None:  # BRIEF2 §4: 같은 부호 + 중앙 |z| ≥ 0.5
        with np.errstate(all="ignore"):
            passed &= np.nanmedian(np.abs(vB), axis=0) >= zmin
    return passed, side, cnt, ok.sum(0)


def rate_on(vX: np.ndarray, side: np.ndarray, norm: str) -> np.ndarray:
    hi, lo, ok = sides(vX, norm)
    on = np.where(side == 1, hi.sum(0), lo.sum(0))
    return on / np.maximum(ok.sum(0), 1)


def spaced_draw(pool: np.ndarray, k: int, gap: int = 10, tries: int = 200) -> np.ndarray | None:
    for _ in range(tries):
        pick = []
        for i in RNG.permutation(pool):
            if all(abs(i - j) > gap for j in pick):
                pick.append(i)
                if len(pick) == k:
                    return np.array(pick)
    return None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    dates, prods, pg, Bm, Sm, G = load()
    pf = pd.read_parquet(DER / "price_feats.parquet")
    pf["date"] = pd.to_datetime(pf["date"])
    pf = pf.set_index("date").reindex(dates).reset_index().rename(columns={"index": "date"})
    ev = json.loads((DER / "events.json").read_text(encoding="utf-8"))
    S = event_sets(ev, dates, pf)
    fit_mask = (dates >= START)

    # 창 길이별 특징·P·Eq
    feat, Pm, Em = {}, {}, {}
    for L in (1, 5, 15):
        F = features(Bm, Sm, G, prods, L)
        feat[L] = F
        P, C = rolling_pct(F.values)
        Pm[L] = P
        Em[L] = echo_resid(P, C, design(pf, L), fit_mask)
        print(f"[L={L}] 특징 {F.shape[1]}개", flush=True)
    names = list(feat[1].columns)
    pd.DataFrame({"feature": names, "family": [n.split(":")[0] for n in names]}).to_csv(OUT / "features.csv", index=False)

    # 칸 = (특징, 창, 정규화). 사건 값 행렬을 창마다 모은다
    def vals(norm: str, w: str, idx: np.ndarray) -> np.ndarray:
        L, off = WINS[w]
        return placed((Pm if norm == "P" else Em)[L], idx, off)

    rows = []
    for norm in ("P", "E"):
        for w in WINS:
            vB10, vB15 = vals(norm, w, S["B10"]), vals(norm, w, S["B15"])
            vF, vD = vals(norm, w, S["F"]), vals(norm, w, S["D"])
            p1, side, cnt, nok = common_pass(vB10, norm, 8)
            fr, dr = rate_on(vF, side, norm), rate_on(vD, side, norm)
            b15c = np.where(side == 1, sides(vB15, norm)[0].sum(0), sides(vB15, norm)[1].sum(0))
            v24, v26 = vals(norm, w, S["B10_2425"]), vals(norm, w, S["B10_26"])
            p3, side3, cnt3, _ = common_pass(v24, norm, 5, zmin=None)  # 고르기는 z 문턱 없음(BRIEF2 §4)
            f24, d24 = rate_on(vals(norm, w, S["F_2425"]), side3, norm), rate_on(vals(norm, w, S["D_2425"]), side3, norm)
            sel = p3 & (f24 <= 1 / 3) & (d24 <= 1 / 3)
            c26 = np.where(side3 == 1, sides(v26, norm)[0].sum(0), sides(v26, norm)[1].sum(0))
            f26 = rate_on(vals(norm, w, S["F_26"]), side3, norm)
            med = np.nanmedian(vB10, axis=0)
            for j, n in enumerate(names):
                rows.append({"feature": n, "window": w, "norm": norm, "side": int(side[j]), "B10_on": int(cnt[j]),
                             "B10_n": int(nok[j]), "B10_med": med[j], "B15_on": int(b15c[j]), "F_rate": fr[j],
                             "D_rate": dr[j], "common": bool(p1[j]), "sel_side": int(side3[j]),
                             "B2425_on": int(cnt3[j]), "F2425_rate": f24[j], "D2425_rate": d24[j], "selected": bool(sel[j]),
                             "B26_on": int(c26[j]), "F26_rate": f26[j]})
    cells = pd.DataFrame(rows)
    cells["checked"] = cells["selected"] & (cells["B26_on"] >= 3) & (cells["F26_rate"] <= 1 / 3)
    print("칸", len(cells), "공통", int(cells["common"].sum()), "고름", int(cells["selected"].sum()),
          "확인 통과", int(cells["checked"].sum()), flush=True)

    # 우연 기대(공통): 바닥·F·D 모은 날에서 10세션 넘게 떨어진 가짜 사건 10개
    pool = np.unique(np.concatenate([S["B10"], S["F"], S["D"]]))
    null_rows = []
    for rep in range(a.reps):
        pick = spaced_draw(pool, len(S["B10"]))
        if pick is None:
            continue
        near = np.unique((pick[:, None] + np.arange(-5, 6)[None, :]).ravel())  # 실제처럼 사건 ±5 대조 제외
        rest_F = np.setdiff1d(S["F"], near)
        rest_D = np.setdiff1d(S["D"], near)
        rec = {"rep": rep}
        for norm in ("P", "E"):
            tot = dist = 0
            for w in WINS:
                p1, side, _, _ = common_pass(vals(norm, w, pick), norm, 8)
                fr, dr = rate_on(vals(norm, w, rest_F), side, norm), rate_on(vals(norm, w, rest_D), side, norm)
                tot += int(p1.sum())
                dist += int((p1 & (fr <= 1 / 3) & (dr <= 1 / 3)).sum())
            rec[f"{norm}_common"], rec[f"{norm}_distinct"] = tot, dist
        null_rows.append(rec)
    nulls = pd.DataFrame(null_rows)
    nulls.to_csv(OUT / "common_null.csv", index=False)

    # 우연 기대(확인): 고른 칸마다 2026 바닥·F·D 모은 날에서 가짜 바닥 4개
    pool26 = np.unique(np.concatenate([S["B10_26"], S["F_26"], S["D_26"]]))
    selc = cells[cells["selected"]].copy()
    hits = np.zeros(len(selc))
    nrep = 0
    for rep in range(a.reps):
        pick = spaced_draw(pool26, len(S["B10_26"]))
        if pick is None:
            continue
        nrep += 1
        restF = np.setdiff1d(S["F_26"], np.unique((pick[:, None] + np.arange(-5, 6)[None, :]).ravel()))
        for (norm, w), grp in selc.groupby(["norm", "window"]):
            j = [names.index(f) for f in grp["feature"]]
            sd = grp["sel_side"].values
            v = vals(norm, w, pick)[:, j]
            hi, lo, _ = sides(v, norm)
            on = np.where(sd == 1, hi.sum(0), lo.sum(0))
            fr = rate_on(vals(norm, w, restF)[:, j], sd, norm)
            hits[selc.index.get_indexer(grp.index)] += ((on >= 3) & (fr <= 1 / 3)).astype(float)
    selc["p_null_check"] = hits / max(nrep, 1)
    selc.to_csv(OUT / "select_2425.csv", index=False)
    exp_pass = float(selc["p_null_check"].sum())
    act_pass = int(selc["checked"].sum())
    # 포아송 이항 근사: 정규 근사 대신 몬테카를로
    sims = (RNG.random((20000, len(selc))) < selc["p_null_check"].values).sum(1) if len(selc) else np.zeros(1)
    p_ge = float((sims >= act_pass).mean())
    selc[selc["checked"]].to_csv(OUT / "check_2026.csv", index=False)

    # 조합: 고른 칸 상위 20개의 두 칸 동시 켜짐
    combos = []
    top = selc.assign(score=selc["B2425_on"] / 6 - selc["F2425_rate"]).sort_values("score", ascending=False).head(20)

    def onmask(r, idx):
        L, off = WINS[r["window"]]
        v = placed((Pm if r["norm"] == "P" else Em)[L], idx, off)[:, names.index(r["feature"])]
        hi, lo, ok = sides(v[:, None], r["norm"])
        return np.where(r["sel_side"] == 1, hi[:, 0], lo[:, 0]), ok[:, 0]

    for (_, r1), (_, r2) in itertools.combinations(top.iterrows(), 2):
        rec = {"c1": f"{r1['feature']}·{r1['window']}·{r1['norm']}", "c2": f"{r2['feature']}·{r2['window']}·{r2['norm']}"}
        for key in ("B10_2425", "F_2425", "D_2425", "B10_26", "F_26"):
            m1, o1 = onmask(r1, S[key])
            m2, o2 = onmask(r2, S[key])
            both = m1 & m2
            rec[key] = float(both.sum()) if key.startswith("B10") else float(both.sum() / max((o1 & o2).sum(), 1))
        rec["selected"] = rec["B10_2425"] >= 5 and rec["F_2425"] <= 1 / 3 and rec["D_2425"] <= 1 / 3
        rec["checked"] = rec["selected"] and rec["B10_26"] >= 3 and rec["F_26"] <= 1 / 3
        combos.append(rec)
    combos = pd.DataFrame(combos)
    combos.to_csv(OUT / "combos.csv", index=False)
    cells.to_csv(OUT / "cells.csv", index=False)

    # 직관 표: 원값(비율 그대로) 중앙값
    def raw(name, w, idx):
        L, off = WINS[w]
        return placed(feat[L][[name]].values, idx, off)[:, 0]

    allidx = np.flatnonzero(fit_mask & (np.arange(len(dates)) < len(dates)))
    pick_rows = [("A:imb:SOXL", "SOXL 매수 ÷ 매도(배)", "bs"), ("D:fund", "시장 순매도 합 ÷ 순매수 합", "x"),
                 ("D:breadth", "순매수 종목 비율", "x"), ("D:bull", "레버리지 롱 매수 몫(롱 ÷ (롱 + 인버스))", "x"),
                 ("B:nshare:SOXL", "SOXL 순매수 ÷ 시장 순매매 총량", "x"), ("C:lbr:SOXL|NVDA", "SOXL 매수 ÷ NVDA 매수(배)", "exp"),
                 ("C:lbr:SOXL|TQQQ", "SOXL 매수 ÷ TQQQ 매수(배)", "exp"), ("C:lbr:SOXL|SEMI1", "SOXL 매수 ÷ 그 밖 1배 반도체 매수(배)", "exp"),
                 ("A:imb:SEMI1", "그 밖 1배 반도체 매수 ÷ 매도(배)", "bs"), ("A:imb:STK1", "그 밖 1배 개별주 매수 ÷ 매도(배)", "bs"),
                 ("A:imb:IDX1", "1배 지수 ETF 매수 ÷ 매도(배)", "bs"), ("E:rN:SOXL", "SOXL 순매수 순위(÷ 종목 수)", "x")]
    intu = []
    for name, label, kind in pick_rows:
        for w in ("W2", "W3", "W4"):
            rec = {"특징": name, "뜻": label, "창": w}
            for key, idx in (("B15", S["B15"]), ("B10", S["B10"]), ("F", S["F"]), ("D", S["D"]), ("평소", allidx)):
                v = raw(name, w, idx)
                with np.errstate(divide="ignore", invalid="ignore"):
                    if kind == "bs":
                        v = (1 + v) / (1 - v)  # 매수 ÷ 매도
                    elif kind == "exp":
                        v = np.exp(v)
                rec[key] = float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
            intu.append(rec)
    intu = pd.DataFrame(intu)
    intu.to_csv(OUT / "intuitive.csv", index=False, encoding="utf-8-sig")

    meta = {"cells": int(len(cells)), "features": len(names), "events": S["names"],
            "common": {n: int(cells[(cells.norm == n) & cells.common].shape[0]) for n in ("P", "E")},
            "common_distinct": {n: int(cells[(cells.norm == n) & cells.common & (cells.F_rate <= 1 / 3) & (cells.D_rate <= 1 / 3)].shape[0])
                                for n in ("P", "E")},
            "null_reps": int(len(nulls)),
            "null_common_mean": {n: float(nulls[f"{n}_common"].mean()) for n in ("P", "E")},
            "null_common_p95": {n: float(nulls[f"{n}_common"].quantile(0.95)) for n in ("P", "E")},
            "null_distinct_mean": {n: float(nulls[f"{n}_distinct"].mean()) for n in ("P", "E")},
            "selected": int(len(selc)), "checked": act_pass, "checked_expected_null": exp_pass, "checked_p_ge": p_ge,
            "check_null_reps": nrep, "combos": int(len(combos)), "combos_selected": int(combos["selected"].sum()) if len(combos) else 0,
            "combos_checked": int(combos["checked"].sum()) if len(combos) else 0}
    for n in ("P", "E"):
        meta[f"null_common_p_ge_{n}"] = float((nulls[f"{n}_common"] >= meta["common"][n]).mean())
        meta[f"null_distinct_p_ge_{n}"] = float((nulls[f"{n}_distinct"] >= meta["common_distinct"][n]).mean())
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "events"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
