"""갈래 contrast(BRIEF §6): 진짜 바닥(B10) 대 가짜 바닥(F) — 가격 메아리 너머 차이. 설명용(바닥·F 모두 사후 표지).

실행(연구 폴더에서): py code/contrast.py
읽는 것: derived/measures_win.parquet · price_feats.parquet · events.json (고치지 않음)
출력(results/contrast/): episodes.csv · paired.csv · perm.csv · holdout_2026.csv · robustness.csv · summary.md
연결 뒤 추가 점검(BRIEF 정의는 그대로): ① 메아리에 제곱항을 더한 잔차 E_quad ② 선형 메아리 예측이 바닥만큼
극단인 사건 밖 날의 잔차 ③ 버팀 기준의 귀무 보정(국면 부호 함께 뒤집기) ④ 합친 묶음 낀 국면 빼기.
"""
from __future__ import annotations

import datetime as dt
import json
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

STUDY = Path(__file__).resolve().parents[1]
DER = STUDY / "derived"
OUT = STUDY / "results" / "contrast"

WINS = {"W1": (-20, -6), "W2": (-5, -1), "W3": (0, 0), "W4": (1, 5)}
WLAB = {"W1": "[−20,−6]", "W2": "[−5,−1]", "W3": "[0]", "W4": "[+1,+5]"}
B2_WIN = (pd.Timestamp("2026-07-16"), pd.Timestamp("2026-08-07"))  # B2 막차 창(미국 거래일, 지시)
SEED = 20261103
R = 10_000
ALPHA = 0.05
NAMES = {
    "A1": "전체 거래 규모", "A2": "전체 매수 규모", "A3": "전체 매도 규모", "A4": "합집합 순매수 합",
    "X1": "SOXL 순매수", "X2": "SOXL 매수", "X3": "SOXL 매도", "X4": "SOXL 순매수 비중", "X5": "SOXL 거래 몫",
    "X6": "SOXL 매도÷매수", "H1": "SOXS 순매수", "H2": "SOXL 강세 몫", "H3": "주식 인버스 매수 몫",
    "H4": "주식 인버스 순매수", "L1": "3배 롱 순매수", "L2": "3배 롱 순매수 비중", "L3": "단일종목 레버리지 롱 순매수",
    "L4": "지수형 레버리지÷현물 매수", "R1": "1배 반도체 순매수", "R2": "1배 지수 ETF 순매수", "R3": "채권·현금 순매수",
    "R4": "그 밖 1배 개별주 순매수", "R5": "인컴 순매수", "C1": "SOXL 보유 좌수 변화", "C2": "SOXL 보관 비중",
    "C3": "보관 Top50 합 변화",
}
MCODES = list(NAMES)
NORM = {"P": "P", "E": "E_full"}
# 182칸: 26 × {W1,W2,W3} × {P,E} + 26 × W4 × E (W4 원값 차이는 정의상 반등 메아리)
CELLS = [(m, w, n) for m in MCODES for w in ("W1", "W2", "W3") for n in ("P", "E")] + [(m, "W4", "E") for m in MCODES]


def cid(m, w, n):
    return f"{m}·{w}·{n}"


# ---------- 검정 ----------
_SG: dict = {}


def signs(n):
    """부호 2^n 모두(행 = 한 가지, 1 = 양)."""
    if n not in _SG:
        _SG[n] = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(float)
    return _SG[n]


def wil_exact(d):
    """정확 양측 윌콕슨 부호 순위 p. 0은 빼고, 같은 크기는 중간 순위, 부호 2^n 모두 센다."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d != 0)]
    n = len(d)
    if n == 0:
        return np.nan
    r = stats.rankdata(np.abs(d))
    mu = r.sum() / 2
    T = signs(n) @ r
    return float((np.abs(T - mu) >= abs(r[d > 0].sum() - mu) - 1e-9).mean())


def sign_p(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    k, m = int((d > 0).sum()), int((d != 0).sum())
    return float(stats.binomtest(k, m, 0.5).pvalue) if m else np.nan


def null_frac(d):
    """귀무(부호 대칭)에서 이 칸이 p < 0.05를 낼 정확 확률: 윌콕슨(같은 순위), 부호."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d != 0)]
    n = len(d)
    if n == 0:
        return np.nan, np.nan
    r = stats.rankdata(np.abs(d))
    dev = np.abs(signs(n) @ r - r.sum() / 2)
    s = np.sort(dev)
    pw = (len(s) - np.searchsorted(s, dev - 1e-9, side="left")) / len(s)
    ps = np.array([stats.binomtest(k, n, 0.5).pvalue for k in range(n + 1)])
    fs = sum(comb(n, k) for k in range(n + 1) if ps[k] < ALPHA) / 2 ** n
    return float((pw < ALPHA).mean()), float(fs)


def tstat(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return np.nan
    sd = d.std(ddof=1)
    return float(d.mean() / (sd / np.sqrt(len(d)))) if sd > 0 else float(np.inf * np.sign(d.mean()))


def pstats(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    return {"n_ep": n, "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()),
            "median_d": float(np.median(d)) if n else np.nan, "mean_d": float(d.mean()) if n else np.nan,
            "t_stat": tstat(d), "wil_p": wil_exact(d), "sign_p": sign_p(d)}


def bh(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    m = len(pv)
    if m:
        o = np.argsort(pv)
        rk = pv[o] * m / np.arange(1, m + 1)
        qq = np.minimum(np.minimum.accumulate(rk[::-1])[::-1], 1.0)
        tmp = np.empty(m)
        tmp[o] = qq
        q[ok] = tmp
    return q


def rank_order(wp, sp, t):
    """짝 p 순위: 윌콕슨 p → 부호 p → |t| 큰 순."""
    return np.lexsort((-np.nan_to_num(np.abs(t), nan=0.0), np.nan_to_num(sp, nan=9.0), np.nan_to_num(wp, nan=9.0)))


def label_matrix(N, nB, rng):
    """날짜 섞기 R회: 행마다 바닥 표지 nB개."""
    idx = np.argsort(rng.random((R, N)), axis=1)[:, :nB]
    lab = np.zeros((R, N))
    lab[np.arange(R)[:, None], idx] = 1.0
    return lab


def perm_test(x, isB, lab):
    """날짜 섞기: 관측 차(바닥 평균 − F 평균), 양측 p, 섞은 차 R개. 값 없는 날은 그 칸에서만 빠진다."""
    ok = np.isfinite(x)
    xb = np.where(ok, x, 0.0)
    obs = x[isB & ok].mean() - x[~isB & ok].mean()
    sB, nB = lab @ xb, lab @ ok
    diffs = sB / nB - (xb.sum() - sB) / (ok.sum() - nB)
    return obs, (1 + np.sum(np.abs(diffs) >= abs(obs) - 1e-12)) / (R + 1), diffs


def exact_perm(x, isB):
    """작은 표본: 바닥 표지 모든 조합으로 정확 양측 p."""
    ok = np.isfinite(x)
    x, isB = x[ok], isB[ok]
    N, nB = len(x), int(isB.sum())
    if nB == 0 or nB == N:
        return np.nan
    obs = x[isB].mean() - x[~isB].mean()
    tot, cnt, k = x.sum(), 0, 0
    for c in combinations(range(N), nB):
        s = x[list(c)].sum()
        cnt += abs(s / nB - (tot - s) / (N - nB)) >= abs(obs) - 1e-12
        k += 1
    return cnt / k


# ---------- 자료 ----------
def load():
    mw = pd.read_parquet(DER / "measures_win.parquet")
    pf = pd.read_parquet(DER / "price_feats.parquet").set_index("date").sort_index()
    ev = json.loads((DER / "events.json").read_text(encoding="utf-8"))
    return mw, pf, ev


def echo_fits(mw, pf):
    """메아리 적합 다시 하기: 선형(build.py와 같음, 대조용)과 제곱항 추가(연결 뒤 점검). 전체 표본 적합."""
    out = []
    for L in sorted(mw["L"].unique()):
        sub = mw[mw["L"] == L]
        idx = pd.DatetimeIndex(sorted(sub["date"].unique()))
        X = pd.DataFrame(index=idx)
        X["const"] = 1.0
        X["ret"] = pf[f"ret_{L}"].reindex(idx)
        X["pre20"] = pf[f"pre20_{L}"].reindex(idx)
        X["dd60"] = pf["dd60"].reindex(idx)
        X["rv20"] = pf["rv20"].reindex(idx)
        reg = pf["regime"].reindex(idx)
        for k in range(1, 5):
            X[f"reg{k}"] = (reg == k).astype(float)
        Xq = X.copy()
        for c in ("ret", "pre20", "dd60", "rv20"):
            Xq[c + "_sq"] = X[c] ** 2
        Xq["ret_x_dd60"] = X["ret"] * X["dd60"]
        for m in MCODES:
            y = sub[sub["measure"] == m].set_index("date")["y"].reindex(idx).values
            res = {"date": idx, "L": L, "measure": m}
            for tag, XX in (("lin", X.values), ("quad", Xq.values)):
                e, f = np.full(len(y), np.nan), np.full(len(y), np.nan)
                ok = np.isfinite(y) & np.isfinite(XX).all(1)
                b, _, rank, _ = np.linalg.lstsq(XX[ok], y[ok], rcond=None)
                r_ = y[ok] - XX[ok] @ b
                e[ok] = r_ / np.sqrt(r_ @ r_ / (ok.sum() - rank))
                f[ok] = XX[ok] @ b
                res[f"E_{tag}"], res[f"fit_{tag}"] = e, f
            out.append(pd.DataFrame(res))
    return pd.concat(out, ignore_index=True)


def event_days(ev, pf):
    b15 = {b["bottom"]: bool(b["B15"]) for b in ev["B10_events"]}
    rows = [{"date": pd.Timestamp(b["bottom"]), "role": "B", "episode": b["bottom"], "has_flows": True}
            for b in ev["B10_events"]]
    rows += [{"date": pd.Timestamp(f["date"]), "role": "F", "episode": f["episode"], "has_flows": bool(f["has_flows"])}
             for f in ev["F"]]
    days = pd.DataFrame(rows)
    days["ep_B15"] = [b15.get(e, False) for e in days["episode"]]
    days["pos"] = days["date"].map({d: i for i, d in enumerate(pf.index)}).astype(int)
    days["close"] = pf["SOXX_Close"].reindex(days["date"]).values
    days["dd60"] = pf["dd60"].reindex(days["date"]).values
    return days


def window_table(days, mw, fits, pf):
    """사건 날마다 W1~W4(그날 = k 0) 창 끝을 잡아 창 값·P·E·절단·합침을 붙인다."""
    sess = pf.index
    recs = []
    for d in days.itertuples():
        for w, (a, z) in WINS.items():
            i0, i1 = d.pos + a, d.pos + z
            if i0 < 0 or i1 >= len(sess):
                continue
            s0, s1 = sess[i0], sess[i1]
            recs.append({"date": d.date, "W": w, "L": z - a + 1, "start": s0, "end": s1,
                         "touch_B2": bool(s0 <= B2_WIN[1] and s1 >= B2_WIN[0])})
    v = pd.DataFrame(recs).merge(mw.rename(columns={"date": "end"}), on=["end", "L"], how="inner")  # 흐름 끝(10/29) 뒤 창은 빠짐
    v = v.merge(fits.rename(columns={"date": "end"}), on=["end", "L", "measure"], how="left")
    return v.merge(days[["date", "role", "episode", "ep_B15"]], on="date", how="left")


def paired_eps(ev, days):
    """짝 국면: B10 바닥이 있고 흐름 있는 F 날이 1개 이상."""
    return [e["episode"] for e in ev["episodes"] if e["episode"] != "current"
            and len(days[(days["role"] == "F") & (days["episode"] == e["episode"]) & days["has_flows"]])]


def get(lk, d, w, m, col):
    """창이 흐름 끝 뒤면 NaN."""
    k = (pd.Timestamp(d), w, m)
    return float(lk.loc[k, col]) if k in lk.index else np.nan


def nmean(xs):
    xs = np.asarray(xs, float)
    return round(float(np.nanmean(xs)), 4) if np.isfinite(xs).any() else np.nan


def ep_d(lk, ep, f_dates, m, w, col):
    """국면 하나의 d = 바닥 값 − 그 국면 F 날 평균(값 있는 F만)."""
    k = (pd.Timestamp(ep), w, m)
    if k not in lk.index or not np.isfinite(lk.loc[k, col]):
        return None
    rb = lk.loc[k]
    fr = [lk.loc[(f, w, m)] for f in f_dates if (f, w, m) in lk.index]
    fr = [r for r in fr if np.isfinite(r[col])]
    if not fr:
        return None
    fm = float(np.mean([r[col] for r in fr]))
    return {"d": float(rb[col]) - fm, "b": float(rb[col]), "f": fm, "nF": len(fr),
            "B_cens": bool(rb["n_cens"] > 0), "F_cens": sum(r["n_cens"] > 0 for r in fr),
            "B_merged": bool(rb["n_merged"] > 0), "F_merged": sum(r["n_merged"] > 0 for r in fr),
            "touch_B2": bool(rb["touch_B2"] or any(r["touch_B2"] for r in fr)),
            "merged_any": bool(rb["n_merged"] > 0 or any(r["n_merged"] > 0 for r in fr))}


def survive_flags(d, keep_a, keep_b, dr):
    """버팀: (a) B2 막차 국면 뺀 뒤 (b) B15만 중앙 d 부호 유지, (c) 하나 빼기 부호 뒤집힘 0."""
    ok = np.isfinite(d)
    sa = np.sign(np.median(d[ok & keep_a])) == dr if (ok & keep_a).any() else False
    sb = np.sign(np.median(d[ok & keep_b])) == dr if (ok & keep_b).any() else False
    idx = np.flatnonzero(ok)
    flips = sum(np.sign(np.median(d[np.setdiff1d(idx, [j])])) != dr for j in idx)
    return bool(sa), bool(sb), int(flips)


# ---------- 본 계산 ----------
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    mw, pf, ev = load()
    fits = echo_fits(mw, pf)
    chk = fits.merge(mw[["date", "L", "measure", "E_full"]], on=["date", "L", "measure"])
    lin_gap = float(np.nanmax(np.abs(chk["E_lin"] - chk["E_full"])))
    assert lin_gap < 1e-6, f"선형 메아리 재적합이 E_full과 다름: {lin_gap}"
    days = event_days(ev, pf)
    V = window_table(days, mw, fits, pf)
    lk = V.set_index(["date", "W", "measure"]).sort_index()
    eps_all = paired_eps(ev, days)
    fmap = {e: list(days.loc[(days["role"] == "F") & (days["episode"] == e) & days["has_flows"], "date"]) for e in eps_all}
    b15 = {b["bottom"]: bool(b["B15"]) for b in ev["B10_events"]}
    sess = pf.index
    run_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    fend = pd.Timestamp(mw["date"].max())  # 결제 흐름 마지막 미국 거래일

    # 1. episodes.csv
    def k0(d):
        return float(pf.loc[d, "lr1"])

    def w4(d):  # log C(k+5)/C(k): W4 가격 경로(정의상 바닥 반등·F 추가 하락)
        i = sess.get_loc(d)
        return float(np.log(pf["SOXX_Close"].iloc[i + 5] / pf["SOXX_Close"].iloc[i])) if i + 5 < len(sess) else np.nan

    erows = []
    for e in ev["episodes"]:
        ep = e["episode"]
        fd = days[(days["role"] == "F") & (days["episode"] == ep)]
        bb = next((b for b in ev["B10_events"] if b["bottom"] == ep), None)
        spans = []
        for d in ([pd.Timestamp(ep)] if bb else []) + list(fd["date"]):
            i = sess.get_loc(d)
            spans.append((sess[max(i - 20, 0)], sess[min(i + 5, len(sess) - 1)]))
        row = {"episode": ep, "top10": e["start"], "top10_close": round(float(pf.loc[pd.Timestamp(e["start"]), "SOXX_Close"]), 2),
               "bottom": ep if bb else "", "bottom_close": round(bb["close"], 2) if bb else np.nan,
               "bottom_dd60": round(float(pf.loc[pd.Timestamp(ep), "dd60"]), 4) if bb else np.nan,
               "bottom_dd_top": bb["dd10"] if bb else np.nan, "B15": b15.get(ep, False),
               "n_F": len(fd), "n_F_flows": int(fd["has_flows"].sum()),
               "F_dates": ";".join(d.date().isoformat() for d in fd["date"]),
               "F_close": ";".join(f"{c:.2f}" for c in fd["close"]),
               "F_dd60": ";".join(f"{x:.4f}" for x in fd["dd60"]),
               "F_dd60_mean": nmean(fd["dd60"]), "paired": ep in eps_all,
               "ret_k0_B": round(k0(pd.Timestamp(ep)), 4) if bb else np.nan,
               "ret_k0_F_mean": nmean([k0(d) for d in fd["date"]]),
               "ret_W4_B": round(w4(pd.Timestamp(ep)), 4) if bb else np.nan,
               "ret_W4_F_mean": nmean([w4(d) for d in fd["date"]]),
               "touch_B2_span": any(s0 <= B2_WIN[1] and s1 >= B2_WIN[0] for s0, s1 in spans), "note": ""}
        if ep == "current":
            cur, u = ev["current"], ev["F_undetermined"]
            # 흐름 끝 뒤에 창 끝이 놓여 빠지는 창(F 날별)
            def late(d, z):  # 창 끝이 흐름 끝 뒤(또는 가격 끝 뒤)
                j = sess.get_loc(d) + z
                return j >= len(sess) or sess[j] > fend
            miss = [f"{d.date().isoformat()} {'·'.join(w for w, (_, z) in WINS.items() if late(d, z))}"
                    for d in fd["date"] if any(late(d, z) for _, (_, z) in WINS.items())]
            row["note"] = (f"미확정 하락: 고점 {cur['10']['last_top']} {cur['10']['last_top_close']:.2f} → "
                           f"{cur['latest']['date']} {cur['latest']['close']:.2f}({cur['10']['dd_latest_from_last_top']:+.4f}). "
                           f"F 미정 {', '.join(x['date'] for x in u)}. 흐름 끝(미국 거래 {fend.date()}) 뒤라 빠진 창: "
                           f"{'; '.join(miss) if miss else '없음'}")
        elif not len(fd):
            row["note"] = "F 날 없음 → 짝 비교 밖(날짜 섞기에는 바닥으로 듦)"
        erows.append(row)
    E = pd.DataFrame(erows)
    E.to_csv(OUT / "episodes.csv", index=False, encoding="utf-8-sig")

    # 2. paired.csv (182칸) + E 칸은 제곱항 잔차로도
    pairs, pairs_q, prow = {}, {}, []
    for m, w, n in CELLS:
        pr = {ep: r for ep in eps_all if (r := ep_d(lk, ep, fmap[ep], m, w, NORM[n])) is not None}
        pairs[(m, w, n)] = pr
        d = np.array([x["d"] for x in pr.values()])
        st = pstats(d)
        nw, ns = null_frac(d)
        row = {"cell": cid(m, w, n), "measure": m, "name": NAMES[m], "W": w, "win": WLAB[w], "norm": n, **st,
               "dir": int(np.sign(st["median_d"])) if np.isfinite(st["median_d"]) else 0,
               "null_p05_wil": nw, "null_p05_sign": ns}
        if n == "E":
            prq = {ep: r for ep in eps_all if (r := ep_d(lk, ep, fmap[ep], m, w, "E_quad")) is not None}
            pairs_q[(m, w, n)] = prq
            sq = pstats(np.array([x["d"] for x in prq.values()]))
            row.update({"quad_n_pos": sq["n_pos"], "quad_median_d": sq["median_d"], "quad_wil_p": sq["wil_p"],
                        "quad_sign_p": sq["sign_p"]})
        for ep in eps_all:
            row[f"d_{ep}"] = pr[ep]["d"] if ep in pr else np.nan
        row.update({"n_win_B_cens": sum(x["B_cens"] for x in pr.values()), "n_win_F_cens": sum(x["F_cens"] for x in pr.values()),
                    "n_win_F": sum(x["nF"] for x in pr.values()), "n_win_B_merged": sum(x["B_merged"] for x in pr.values()),
                    "n_win_F_merged": sum(x["F_merged"] for x in pr.values())})
        prow.append(row)
    PD = pd.DataFrame(prow)
    PD["q_wil"], PD["q_sign"] = bh(PD["wil_p"]), bh(PD["sign_p"])
    PD = PD.iloc[rank_order(PD["wil_p"].values, PD["sign_p"].values, PD["t_stat"].values)].reset_index(drop=True)
    PD.insert(0, "rank", np.arange(1, len(PD) + 1))
    PD.to_csv(OUT / "paired.csv", index=False, encoding="utf-8-sig", float_format="%.6g")

    # 2b. 국면 부호 함께 뒤집기(2^7): 칸 상관을 둔 채 p < 0.05 칸 수·버팀 수의 귀무 분포
    neps = len(eps_all)
    ecells = [c for c in CELLS if c[2] == "E"]
    D = np.array([[pairs[c][ep]["d"] if ep in pairs[c] else np.nan for ep in eps_all] for c in CELLS])
    DQ = np.array([[pairs_q[c][ep]["d"] if ep in pairs_q[c] else np.nan for ep in eps_all] for c in ecells])
    KA = np.array([[not (ep in pairs[c] and pairs[c][ep]["touch_B2"]) for ep in eps_all] for c in CELLS])
    KB = np.array([[b15.get(ep, False) for ep in eps_all] for _ in CELLS])
    fl = {"wil": [], "sign": [], "quad": [], "surv": [], "lin_E": []}
    isE = np.array([c[2] == "E" for c in CELLS])
    for s in signs(neps):
        sg = np.where(s > 0, 1.0, -1.0)
        Ds = D * sg
        wp = np.array([wil_exact(x) for x in Ds])
        sp = np.array([sign_p(x) for x in Ds])
        tt = np.array([tstat(x) for x in Ds])
        fl["wil"].append(int((wp < ALPHA).sum()))
        fl["sign"].append(int((sp < ALPHA).sum()))
        fl["lin_E"].append(int((wp[isE] < ALPHA).sum()))
        fl["quad"].append(int(sum(wil_exact(x) < ALPHA for x in DQ * sg)))
        sv = 0
        for i in rank_order(wp, sp, tt)[:10]:
            dr = np.sign(np.nanmedian(Ds[i]))
            a_, b_, f_ = survive_flags(Ds[i], KA[i], KB[i], dr)
            sv += int(wp[i] < ALPHA and a_ and b_ and f_ == 0)
        fl["surv"].append(sv)
    fl = {k: np.array(v) for k, v in fl.items()}
    obs = {"wil": int((PD["wil_p"] < ALPHA).sum()), "sign": int((PD["sign_p"] < ALPHA).sum()),
           "lin_E": int((PD.loc[PD["norm"] == "E", "wil_p"] < ALPHA).sum()),
           "quad": int((PD["quad_wil_p"] < ALPHA).sum())}

    # 3. perm.csv: 날짜 단위 표지 섞기(B10 11개 대 F 28개 전체, 창이 흐름 끝 뒤면 그 칸에서만 빠짐)
    pool = days.reset_index(drop=True)
    isB = (pool["role"] == "B").values
    lab = label_matrix(len(pool), int(isB.sum()), np.random.default_rng(SEED))
    qrow, perm_abs, perm_abs_q = [], [], []
    for m, w, n in CELLS:
        x = np.array([get(lk, d, w, m, NORM[n]) for d in pool["date"]], float)
        cens = np.array([get(lk, d, w, m, "n_cens") > 0 for d in pool["date"]])
        mrg = np.array([get(lk, d, w, m, "n_merged") > 0 for d in pool["date"]])
        ok = np.isfinite(x)
        o, p, diffs = perm_test(x, isB, lab)
        perm_abs.append(np.abs(diffs))
        row = {"cell": cid(m, w, n), "measure": m, "name": NAMES[m], "W": w, "win": WLAB[w], "norm": n,
               "n_B": int((isB & ok).sum()), "n_F": int((~isB & ok).sum()), "mean_B": x[isB & ok].mean(),
               "mean_F": x[~isB & ok].mean(), "diff": o, "p_perm": p}
        if n == "E":
            xq = np.array([get(lk, d, w, m, "E_quad") for d in pool["date"]], float)
            oq, pq, dq = perm_test(xq, isB, lab)
            perm_abs_q.append(np.abs(dq))
            row.update({"quad_diff": oq, "quad_p_perm": pq})
        row.update({"n_win_B_cens": int((isB & cens).sum()), "n_win_F_cens": int((~isB & cens).sum()),
                    "n_win_B_merged": int((isB & mrg).sum()), "n_win_F_merged": int((~isB & mrg).sum())})
        qrow.append(row)
    PM = pd.DataFrame(qrow)
    PM["q_perm"] = bh(PM["p_perm"])

    def perm_count_null(arrs):  # 같은 섞기를 관측처럼 놓고 p < 0.05 칸 수를 센다(칸 상관 유지)
        cnt = np.zeros(R, int)
        for a in arrs:
            s_ = np.sort(a)
            cnt += (R - np.searchsorted(s_, a - 1e-12, side="left")) / R < ALPHA
        return cnt
    pnull, pnull_q = perm_count_null(perm_abs), perm_count_null(perm_abs_q)
    obs["perm"], obs["perm_q"] = int((PM["p_perm"] < ALPHA).sum()), int((PM["quad_p_perm"] < ALPHA).sum())
    obs["perm_E"] = int((PM.loc[PM["norm"] == "E", "p_perm"] < ALPHA).sum())
    PM = PM.sort_values("p_perm", kind="stable").reset_index(drop=True)
    PM.to_csv(OUT / "perm.csv", index=False, encoding="utf-8-sig", float_format="%.6g")

    # 4. holdout_2026.csv: 2023~2025 짝 국면으로 W1~W3 상위 5칸 → 2026
    eps_old = [e for e in eps_all if e < "2026"]
    eps_new = [e for e in eps_all if e >= "2026"]
    hsel = pd.DataFrame([{"cell": cid(m, w, n), "measure": m, "W": w, "norm": n,
                          **pstats(np.array([pairs[(m, w, n)][e]["d"] for e in eps_old if e in pairs[(m, w, n)]]))}
                         for m, w, n in CELLS if w != "W4"])
    hsel = hsel.iloc[rank_order(hsel["wil_p"].values, hsel["sign_p"].values, hsel["t_stat"].values)].reset_index(drop=True)
    HS = hsel.head(5)
    n_tied = int(((hsel["wil_p"] == HS["wil_p"].iloc[0]) & (hsel["sign_p"] == HS["sign_p"].iloc[0])).sum())
    b26 = [b for b in ev["B10"] if b >= "2026"]
    f26 = days[(days["role"] == "F") & (days["date"] >= "2026-01-01")]
    hrows = []
    for i, h in HS.iterrows():
        m, w, n = h["measure"], h["W"], h["norm"]
        dr = int(np.sign(h["median_d"]))
        base = {"sel_rank": i + 1, "cell": h["cell"], "measure": m, "name": NAMES[m], "W": w, "win": WLAB[w], "norm": n,
                "dir_2325": dr, "n_ep_2325": h["n_ep"], "n_pos_2325": h["n_pos"], "median_d_2325": h["median_d"],
                "wil_p_2325": h["wil_p"], "sign_p_2325": h["sign_p"], "t_2325": h["t_stat"]}
        for ep in eps_new:
            r = pairs[(m, w, n)].get(ep)
            rx = ep_d(lk, ep, fmap[ep], m, w, "E_exp") if n == "E" else None
            hrows.append({**base, "unit": ep, "n_B": 1, "n_F": r["nF"] if r else 0,
                          "val_B": r["b"] if r else np.nan, "val_F": r["f"] if r else np.nan,
                          "d": r["d"] if r else np.nan, "same_dir": bool(r and np.sign(r["d"]) == dr),
                          "p_exact": np.nan, "n_B_beyond_Fmed": np.nan, "d_E_exp": rx["d"] if rx else np.nan,
                          "same_dir_E_exp": bool(rx and np.sign(rx["d"]) == dr) if n == "E" else np.nan,
                          "note": "B2 막차 창 겹침" if r and r["touch_B2"] else ""})
        xb = np.array([get(lk, b, w, m, NORM[n]) for b in b26], float)
        xf = np.array([get(lk, d, w, m, NORM[n]) for d in f26["date"]], float)
        diff = np.nanmean(xb) - np.nanmean(xf)
        xe = None
        if n == "E":
            xe = np.nanmean([get(lk, b, w, m, "E_exp") for b in b26]) - np.nanmean([get(lk, d, w, m, "E_exp") for d in f26["date"]])
        nfb, nff = int(np.isfinite(xb).sum()), int(np.isfinite(xf).sum())
        n_cur = int(((f26["episode"] == "current").values & np.isfinite(xf)).sum())
        hrows.append({**base, "unit": "2026 날 단위", "n_B": nfb, "n_F": nff, "val_B": np.nanmean(xb), "val_F": np.nanmean(xf),
                      "d": diff, "same_dir": bool(np.sign(diff) == dr),
                      "p_exact": exact_perm(np.r_[xb, xf], np.r_[np.ones(len(xb), bool), np.zeros(len(xf), bool)]),
                      "n_B_beyond_Fmed": int(np.sum(np.sign(xb - np.nanmedian(xf)) == dr)),
                      "d_E_exp": xe if xe is not None else np.nan,
                      "same_dir_E_exp": bool(np.sign(xe) == dr) if xe is not None else np.nan,
                      "note": f"B {nfb} 대 F {nff}(지금 국면 F {n_cur} 포함), 정확 섞기 C({nfb + nff},{nfb}) = "
                              f"{comb(nfb + nff, nfb)}가지. 2026-06-05는 F 없어 날 단위에만"})
    HO = pd.DataFrame(hrows)
    HO.to_csv(OUT / "holdout_2026.csv", index=False, encoding="utf-8-sig", float_format="%.6g")

    # 5. robustness.csv: 짝 p 상위 10칸
    evpos = set()
    for p_ in days["pos"]:
        evpos.update(range(p_ - 5, p_ + 6))
    rrows = []
    for h in PD.head(10).itertuples():
        key = (h.measure, h.W, h.norm)
        pr, dr = pairs[key], h.dir
        dv = np.array([pr[e]["d"] if e in pr else np.nan for e in eps_all])
        row = {"rank": h.rank, "cell": h.cell, "name": h.name, "win": h.win, "norm": h.norm, "n_ep": h.n_ep,
               "n_pos": h.n_pos, "median_d": h.median_d, "wil_p": h.wil_p, "sign_p": h.sign_p, "dir": dr}
        # (a) B2 막차 창(미국 거래 2026-07-16~08-07)에 이 칸 창이 닿는 국면 뺌
        drop_a = [e for e, x in pr.items() if x["touch_B2"]]
        sa = pstats(np.array([x["d"] for e, x in pr.items() if e not in drop_a]))
        row.update({"a_dropped": ";".join(drop_a), "a_n": sa["n_ep"], "a_median_d": sa["median_d"], "a_wil_p": sa["wil_p"],
                    "a_sign_p": sa["sign_p"], "a_same_dir": bool(np.sign(sa["median_d"]) == dr)})
        # (b) B15 국면만
        sb = pstats(np.array([x["d"] for e, x in pr.items() if b15.get(e)]))
        row.update({"b_n": sb["n_ep"], "b_median_d": sb["median_d"], "b_wil_p": sb["wil_p"], "b_sign_p": sb["sign_p"],
                    "b_same_dir": bool(np.sign(sb["median_d"]) == dr)})
        # (c) 국면 하나씩 빼기
        loo = [pstats(np.array([x["d"] for e2, x in pr.items() if e2 != e])) for e in pr]
        meds = np.array([s["median_d"] for s in loo])
        row.update({"c_med_min": meds.min(), "c_med_max": meds.max(),
                    "c_wil_p_min": min(s["wil_p"] for s in loo), "c_wil_p_max": max(s["wil_p"] for s in loo),
                    "c_sign_p_min": min(s["sign_p"] for s in loo), "c_sign_p_max": max(s["sign_p"] for s in loo),
                    "c_flips": int(np.sum(np.sign(meds) != dr))})
        a_, b_, f_ = survive_flags(dv, KA[CELLS.index(key)], KB[0], dr)
        row["survive"] = bool(h.wil_p < ALPHA and a_ and b_ and f_ == 0)
        # (d, 추가) 이 칸 창에 합친 묶음 거래일이 낀 국면 뺌(PANEL §0 주의)
        drop_d = [e for e, x in pr.items() if x["merged_any"]]
        sd_ = pstats(np.array([x["d"] for e, x in pr.items() if e not in drop_d]))
        row.update({"d_dropped": ";".join(drop_d), "d_n": sd_["n_ep"], "d_median_d": sd_["median_d"], "d_wil_p": sd_["wil_p"],
                    "d_same_dir": bool(np.sign(sd_["median_d"]) == dr) if sd_["n_ep"] else np.nan})
        # (e, 추가) 제곱항 메아리 잔차 / (f, 추가) 선형 예측이 바닥만큼 극단인 사건 밖 날의 선형 잔차
        if h.norm == "E":
            row.update({"e_quad_n_pos": h.quad_n_pos, "e_quad_median_d": h.quad_median_d, "e_quad_wil_p": h.quad_wil_p,
                        "e_quad_same_dir": bool(np.sign(h.quad_median_d) == dr)})
            a0, z0 = WINS[h.W]
            s = fits[(fits["L"] == z0 - a0 + 1) & (fits["measure"] == h.measure)].set_index("date")
            bends = [sess[sess.get_loc(pd.Timestamp(e)) + z0] for e in pr]
            fb = float(np.median(s["fit_lin"].reindex(bends)))
            side = np.sign(fb - np.nanmedian(s["fit_lin"]))
            spos = pd.Series(np.arange(len(sess)), index=sess)
            non = s[[(spos[d] - z0) not in evpos for d in s.index]]
            ext = non[((non["fit_lin"] - fb) * side >= 0) & non["E_lin"].notna()]
            row.update({"f_B_fit_med": fb, "f_B_E_mean": float(np.mean([pr[e]["b"] for e in pr])),
                        "f_F_E_mean": float(np.mean([pr[e]["f"] for e in pr])),
                        "f_nonevent_n": len(ext), "f_nonevent_E_mean": float(ext["E_lin"].mean()) if len(ext) else np.nan})
        rrows.append(row)
    RB = pd.DataFrame(rrows)
    RB.to_csv(OUT / "robustness.csv", index=False, encoding="utf-8-sig", float_format="%.6g")

    write_summary(E, PD, PM, HO, HS, RB, eps_all, eps_old, eps_new, fl, obs, pnull, pnull_q, pool, n_tied, run_at, lin_gap, fend)
    print("끝:", OUT)
    return 0


def fmt_p(p):
    return "—" if not np.isfinite(p) else f"{p:.3f}"


def fmt_d(d, n):
    return "—" if not np.isfinite(d) else (f"{d:+.1f}" if n == "P" else f"{d:+.2f}")


def nullrow(name, ncell, o, exact, nul):
    return (f"| {name} | {ncell} | **{o}** | {exact:.1f} | {nul.mean():.1f} · {np.percentile(nul, 95):.0f} | "
            f"{np.mean(nul >= o):.3f} |")


def updown(x):
    return "높음" if x > 0 else "낮음"


def ox(b):
    return "○" if b else "×"


def write_summary(E, PD, PM, HO, HS, RB, eps_all, eps_old, eps_new, fl, obs, pnull, pnull_q, pool, n_tied, run_at, lin_gap, fend):
    n = len(eps_all)
    nofp = [r.episode for r in E.itertuples() if r.episode != "current" and not r.paired]
    isE = PD["norm"] == "E"
    exp_w, exp_s = PD["null_p05_wil"].sum(), PD["null_p05_sign"].sum()
    exp_wE, exp_wP = PD.loc[isE, "null_p05_wil"].sum(), PD.loc[~isE, "null_p05_wil"].sum()
    nPp = int((PD.loc[~isE, "wil_p"] < ALPHA).sum())
    q_any = int((PD["q_wil"] < 0.10).sum() + (PD["q_sign"] < 0.10).sum() + (PM["q_perm"] < 0.10).sum())
    qmin = min(PD["q_wil"].min(), PD["q_sign"].min(), PM["q_perm"].min())
    pm = PM.set_index("cell")
    ep = E[E["paired"]]
    deeper = int((ep["bottom_dd60"] < ep["F_dd60_mean"]).sum())
    surv = int(RB["survive"].sum())
    top_e = RB[RB["norm"] == "E"]
    q_gone = int((top_e["e_quad_wil_p"] >= ALPHA).sum())
    keep = PD[(PD["wil_p"] < ALPHA) & ((PD["norm"] == "P") | (PD["quad_wil_p"] < ALPHA))]
    win34 = PD[isE & PD["W"].isin(["W3", "W4"]) & (PD["wil_p"] < ALPHA)]
    # 사건 밖 극단 예측 날 잔차가 바닥과 같은 부호인 W3·W4 E 칸
    t34 = top_e[top_e["win"].isin([WLAB["W3"], WLAB["W4"]])]
    same_ne = int((np.sign(t34["f_nonevent_E_mean"]) == t34["dir"]).sum())
    ep_rows = HO[HO["unit"].isin(eps_new)]
    day_rows = HO[~HO["unit"].isin(eps_new)]
    allok = [bool(HO[HO["cell"] == c]["same_dir"].all()) for c in HS["cell"]]
    xP = PD.set_index("cell")
    L = []
    a = L.append
    a("# 갈래 contrast — 진짜 바닥(B10) 대 가짜 바닥(F): 가격 메아리 너머 차이 (설명용)")
    a("")
    head = ("**이 자료로는 진짜 바닥과 가짜 바닥을 가르는 한국인 결제 흐름 차이를 가격 메아리 너머에서 찾지 못했다.**"
            if q_any == 0 else f"**BH q < 0.10 칸 {q_any}개가 있다(아래). 그래도 설명용이다.**")
    a(f"{head} q < 0.10 {q_any}칸(최소 q {qmin:.2f}). 메아리 잔차 E 칸이 우연보다 많이 걸렸지만 메아리에 제곱항을 더하면 "
      f"{obs['lin_E']}칸 → {obs['quad']}칸으로 사라지고, 2023~2025로 고른 5칸 중 2026 국면·날 단위 모두 같은 쪽인 칸은 {sum(allok)}개다.")
    a("")
    a(f"- 자료 `derived/`(결제 미국 거래 ~{fend.date()}), 코드 `code/contrast.py`, 실행 {run_at} KST. 바닥·F 모두 뒤 가격으로 붙인 사후 표지라 전부 설명용이고, 시험에 가까운 것은 §5뿐이다.")
    a(f"- 짝 국면 {n}개({', '.join(e[2:] for e in eps_all)}). F 없는 B10 {len(nofp)}개({', '.join(e[2:] for e in nofp)})는 짝 밖. "
      f"n = {n}이면 양측 최소 p는 부호·윌콕슨 모두 {2 / 2 ** n:.4f}(부호는 {n}/{n} 같은 부호일 때만), BH q < 0.10은 그 최소 p 칸이 {int(np.ceil(2 / 2 ** n * 182 / 0.10))}개 이상이어야 나온다.")
    a(f"- 날짜 섞기(보조): B10 {int((pool['role'] == 'B').sum())}개 대 F {int((pool['role'] == 'F').sum())}개(지금 국면 5개 포함, 창이 흐름 끝 뒤면 그 칸에서만 빠짐), {R:,}회, 씨앗 {SEED}. 국면 안 F 날끼리 창이 겹쳐 날이 독립이 아니다.")
    a(f"- 절단(목록 밖 = 0, 주로 SOXS·인컴): 짝 칸 창 중 바닥 {int(PD['n_win_B_cens'].sum())}/{len(PD) * n}·F {int(PD['n_win_F_cens'].sum())}/{int(PD['n_win_F'].sum())}창. "
      f"합친 묶음 낀 창: 바닥 {int(PD['n_win_B_merged'].sum())}·F {int(PD['n_win_F_merged'].sum())}. 칸별 수는 CSV.")
    a("")
    a("## 1. p < 0.05 칸 수 (명목 우연 기대 182 × 0.05 = 9.1)")
    a("")
    a("| 시험 | 칸 | p < 0.05 | 정확 기대 | 칸 상관 둔 귀무 평균 · 95% | 이만큼 이상 나올 확률 |")
    a("|---|---|---|---|---|---|")
    a(nullrow("짝 윌콕슨(주)", 182, obs["wil"], exp_w, fl["wil"]))
    a(nullrow("짝 부호", 182, obs["sign"], exp_s, fl["sign"]))
    a(nullrow("짝 윌콕슨, E 칸만", 104, obs["lin_E"], exp_wE, fl["lin_E"]))
    a(nullrow("짝 윌콕슨, E 제곱항(연결 뒤 추가)", 104, obs["quad"], exp_wE, fl["quad"]))
    a(nullrow("날짜 섞기(보조)", 182, obs["perm"], 9.1, pnull))
    a(nullrow("날짜 섞기, E 제곱항(추가)", 104, obs["perm_q"], 5.2, pnull_q))
    a("")
    a(f"- 정확 기대 = 칸마다 n·순위로 센 귀무 확률의 합(이산이라 9.1보다 작다). 칸 상관 둔 귀무: 짝은 국면 {n}개 부호를 모든 칸에서 함께 뒤집은 {2 ** n}가지, 날짜 섞기는 같은 섞기 {R:,}회."
      f" 원래 흐름 백분위(P 78칸)는 {nPp}칸(기대 {exp_wP:.1f}). 넘친 칸은 E이고 그중 W3·W4가 {len(win34)}칸이다. "
      f"SOXL 순매수 그날 P(X1·W3·P)는 중앙 d {xP.loc['X1·W3·P', 'median_d']:+.1f}점, p {xP.loc['X1·W3·P', 'wil_p']:.2f} — 바닥과 F에서 비슷하게 산다.")
    a("")
    a("## 2. 짝 p 상위 10칸 (윌콕슨 p → 부호 p → |t|, `paired.csv`; P 백분위 점·E 메아리 잔차 z; W1 [−20,−6]·W2 [−5,−1]·W3 그날·W4 뒤 5세션)")
    a("")
    a("| 순 | 칸 | 측정 | 바닥 − F | 양/n | 중앙 d | 윌콕슨 p | 부호 p | E 제곱항 p | 날짜 섞기 p |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for h in PD.head(10).itertuples():
        qp = fmt_p(h.quad_wil_p) if h.norm == "E" else "—"
        a(f"| {h.rank} | {h.cell} | {h.name} {h.win} | 바닥이 {updown(h.dir)} | {h.n_pos}/{h.n_ep} | {fmt_d(h.median_d, h.norm)} | "
          f"{fmt_p(h.wil_p)} | {fmt_p(h.sign_p)} | {qp} | {fmt_p(pm.loc[h.cell, 'p_perm'])} |")
    a("")
    a("## 3. 메아리 점검 (연결 뒤 추가, BRIEF 정의는 그대로)")
    a("")
    a(f"- 바닥은 F보다 깊다(dd60 {deeper}/{n} 국면, 중앙 {ep['bottom_dd60'].median():+.3f} 대 {ep['F_dd60_mean'].median():+.3f}). "
      f"그날 수익 중앙 {ep['ret_k0_B'].median():+.3f} 대 {ep['ret_k0_F_mean'].median():+.3f}, 뒤 5세션 {ep['ret_W4_B'].median():+.3f} 대 {ep['ret_W4_F_mean'].median():+.3f}(정의상 반등 대 추가 하락)."
      " 선형 메아리는 더 깊은 낙폭·큰 반등에 비례해 흐름이 커지길 예측하지만 실제 백분위는 그만큼 움직이지 않는다(포화). 바닥이 가장 극단이라 잔차가 '예측보다 덜'로 나온다.")
    a(f"- 상위 10칸 E {len(top_e)}개 중 {q_gone}개가 제곱항 잔차에서 p ≥ 0.05. 사건 밖 날 중 선형 예측이 바닥만큼 극단인 날"
      f"({int(t34['f_nonevent_n'].min())}~{int(t34['f_nonevent_n'].max())}일, 적음)의 잔차도 W3·W4 E {len(t34)}칸 중 {same_ne}칸이 같은 부호(`robustness.csv` f 열).")
    a("")
    a("## 4. 강건성 (상위 10칸, `robustness.csv`)")
    a("")
    a(f"- (a) B2 막차 창(미국 거래 07-16~08-07)에 그 칸 창이 닿는 국면 뺌(07-29 전부, W1은 09-14도): n {RB['a_n'].min()}~{RB['a_n'].max()}, "
      f"p < 0.05 {int((RB['a_wil_p'] < ALPHA).sum())}/10, 방향 유지 {int(RB['a_same_dir'].sum())}/10."
      f" (b) B15 국면만 n {RB['b_n'].max()}(최소 p {2 / 2 ** RB['b_n'].max():.4f}, p < 0.05 불가): 방향 유지 {int(RB['b_same_dir'].sum())}/10. "
      f"(c) 하나 빼기: 중앙 d 부호 뒤집힘 {int(RB['c_flips'].sum())}건, 최대 p {RB['c_wil_p_max'].min():.3f}~{RB['c_wil_p_max'].max():.3f}.")
    a(f"- 세 조건(윌콕슨 p < 0.05·(a)(b) 방향 유지·(c) 뒤집힘 0)을 모두 지킨 칸 **{surv}/10**. 그러나 국면 부호를 함께 뒤집은 귀무에서도 평균 {fl['surv'].mean():.1f}/10이 지킨다"
      f"(이만큼 이상 {np.mean(fl['surv'] >= surv):.2f}) → n = {n}에서 이 기준은 거르지 못한다.")
    if len(keep):
        items = ", ".join(f"{r.cell}({r.name} {r.win}, 바닥이 {updown(r.dir)} {r.n_pos}/{r.n_ep})" for r in keep.itertuples())
        a(f"- 제곱항 뒤에도 p < 0.05인 E 칸과 P 칸: {items}. 이 수는 우연 기대 안이다. 다음 바닥에서 다시 볼 후보로만 적는다.")
    a("")
    a(f"## 5. 2026 확인: 2023~2025 짝 {len(eps_old)}국면으로 W1~W3 156칸에서 고른 5칸")
    a("")
    a(f"- n = {len(eps_old)}이라 최소 p {2 / 2 ** len(eps_old):.3f}인 칸이 {n_tied}개 → 사실상 |t|로 골랐다(약한 고르기). "
      "2026 날 단위는 2026 B10 4개 대 2026 F 날(지금 국면 포함), 정확 섞기.")
    a("")
    a(f"| 칸 | 측정 | 23~25 방향(양/n) | 2026 국면 d 같은 쪽({', '.join(e[5:] for e in eps_new)}) | 2026 날 단위 차 · 같은 쪽 · 정확 p |")
    a("|---|---|---|---|---|")
    for h in HS.itertuples():
        sub = HO[HO["cell"] == h.cell]
        er = sub[sub["unit"].isin(eps_new)]
        day = sub[~sub["unit"].isin(eps_new)].iloc[0]
        marks = "".join(ox(r.same_dir) for r in er.itertuples())
        a(f"| {h.cell} | {NAMES[h.measure]} {WLAB[h.W]} | 바닥이 {updown(h.median_d)}({h.n_pos}/{h.n_ep}) | "
          f"{marks} ({int(er['same_dir'].sum())}/{len(er)}) | {fmt_d(day['d'], h.norm)} · {ox(day['same_dir'])} · {fmt_p(day['p_exact'])} |")
    a("")
    a(f"- 국면 단위 같은 쪽 {int(ep_rows['same_dir'].sum())}/{len(ep_rows)}(우연 기대 절반), 날 단위 같은 쪽 {int(day_rows['same_dir'].sum())}/{len(day_rows)}이나 "
      f"정확 p < 0.05 {int((day_rows['p_exact'] < ALPHA).sum())}칸. 07-29 국면은 B2 막차 창과 겹친다.")
    a("")
    a("## 6. 결론")
    a("")
    a("- 바닥과 가짜 바닥에서 한국인 SOXL 매수 크기(원래 백분위)는 다르지 않다. 선형 메아리를 빼면 생기는 차이는 더 깊은 낙폭·큰 반등을 흐름이 비례해 따라가지 않는 포화로 설명되고, 2026에서 다시 서지 않는다.")
    a(f"- 이 자료로는 '진짜 바닥에서만 다른 흐름'을 말할 수 없다. 짝 국면 {n}개라 효과가 있어도 이 표본으로는 잡기 어렵다는 점도 함께 적는다."
      f" 점검: 선형 메아리 재적합과 자료판 E_full 최대 차 {lin_gap:.1e}. 짝 d 3칸(X1·W4·E, C1·W2·E, H3·W1·P)은 자료판에서 따로 다시 계산해 같았다.")
    L = [x for i, x in enumerate(L) if not (x == "" and i and L[i - 1].startswith("#"))]  # 제목 뒤 빈 줄 뺌
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"summary {len(L)}줄")


if __name__ == "__main__":
    raise SystemExit(main())
