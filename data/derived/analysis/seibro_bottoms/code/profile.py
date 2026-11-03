"""바닥 공통점(설명용): BRIEF §6 공통(P)·공통(E)와 §6 '2026 확인'의 공통 부분.

실행(연구 폴더에서): py code/profile.py
읽는 것(고치지 않음): derived/measures_win·measures_daily·price_feats(.parquet), derived/events.json
출력(results/profile/): event_table.csv, daily_paths.csv, commonality.csv, holdout_2026.csv, summary.md
BRIEF 밖 진단(규칙·표지는 바꾸지 않음): 가격 이웃 보정 E_nn, 합친 묶음 뺀 민감도.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

STUDY = Path(__file__).resolve().parents[1]
DER = STUDY / "derived"
OUT = STUDY / "results" / "profile"

# 창: (끝 k, 길이 L) — BRIEF §5. measures_win의 L 계열에서 끝 날짜 값을 읽는다
WINS = {"W1": (-6, 15), "W2": (-1, 5), "W3": (0, 1), "W4": (5, 5), "W5": (20, 15)}
NAME = {"A1": "전체 거래", "A2": "전체 매수", "A3": "전체 매도", "A4": "합집합 순매수",
        "X1": "SOXL 순매수", "X2": "SOXL 매수", "X3": "SOXL 매도", "X4": "SOXL 순매수 비중",
        "X5": "SOXL 거래 몫", "X6": "SOXL 매도÷매수", "H1": "SOXS 순매수", "H2": "SOXL 강세 몫",
        "H3": "인버스 매수 몫", "H4": "인버스 순매수", "L1": "3배 롱 순매수", "L2": "3배 롱 비중",
        "L3": "단일종목 레버 순매수", "L4": "지수 레버÷현물 매수", "R1": "1배 반도체 순매수",
        "R2": "1배 지수 ETF 순매수", "R3": "채권·현금 순매수", "R4": "1배 개별주 순매수",
        "R5": "인컴 순매수", "C1": "SOXL 좌수 변화", "C2": "SOXL 보관 비중", "C3": "보관 합 변화"}
MCODES = list(NAME)
DECL_M = ["A1", "X1", "X2", "X3", "H1", "L1", "R1", "R3"]
# 하락 누적의 절단 날 수를 셀 열(PANEL §6 기준, A1은 절단 없음)
DECL_CENS = {"A1": None, "X1": "cens_SOXL", "X2": "cens_SOXL", "X3": "cens_SOXL", "H1": "cens_SOXS",
             "L1": "zero_EQ3L", "R1": "zero_SEMI1", "R3": "zero_BC"}
PATH_M = ["A1", "X1", "X2", "X3", "X4", "X6", "H1", "H2", "H3", "L1", "L3", "R1", "R2", "R3", "R4", "C1", "C2"]
HI, LO, ZMIN = 80.0, 20.0, 0.5
THR = {"B15": 5, "B10": 9, "T7": 6}  # 같은 쪽 최소 개수. B10의 E 규칙·T7(2023~25 7개)은 BRIEF 비율에 맞춘 해석
NN_K, NN_EXCL = 30, 20               # 진단: 이웃 수, 후보에서 뺄 B10 바닥 앞뒤 세션
NSIM = 200_000


def iso(d) -> str:
    return pd.Timestamp(d).date().isoformat() if pd.notna(d) else ""


def feat_cols(L: int) -> list[str]:
    return [f"ret_{L}", f"pre20_{L}", "dd60", "rv20"]  # E 회귀와 같은 가격 특성 4개


def load():
    mw = pd.read_parquet(DER / "measures_win.parquet")
    md = pd.read_parquet(DER / "measures_daily.parquet").set_index("date")
    pf = pd.read_parquet(DER / "price_feats.parquet").set_index("date")
    ev = json.loads((DER / "events.json").read_text(encoding="utf-8"))
    return mw, md, pf, ev


def lookup(anchors: pd.DataFrame, S: pd.DatetimeIndex, mw: pd.DataFrame, pf: pd.DataFrame) -> pd.DataFrame:
    """기준일(k = 0)마다 W1~W5 창 끝 날짜를 NYSE 세션으로 세고 measures_win 값을 붙인다."""
    pos = pd.Series(np.arange(len(S)), index=S)
    rows = []
    for a in anchors["anchor"]:
        i = int(pos[a])
        for w, (ke, L) in WINS.items():
            j = i + ke
            rows.append((a, w, ke, L, S[j] if 0 <= j < len(S) else pd.NaT))
    keys = pd.DataFrame(rows, columns=["anchor", "window", "k_end", "L", "end_date"])
    keys = keys.merge(pd.DataFrame({"measure": MCODES}), how="cross")
    out = keys.merge(mw.rename(columns={"date": "end_date"}), on=["end_date", "L", "measure"], how="left")
    out = out.merge(anchors, on="anchor", how="left")
    # 창 끝 가격 맥락(설명): 창 로그 수익, 60세션 낙폭
    out["win_ret"] = [pf.at[d, f"ret_{L}"] if pd.notna(d) else np.nan for d, L in zip(out["end_date"], out["L"])]
    out["win_dd60"] = [pf.at[d, "dd60"] if pd.notna(d) else np.nan for d in out["end_date"]]
    return out.reset_index(drop=True)


def nn_adjust(q: pd.DataFrame, mw: pd.DataFrame, pf: pd.DataFrame, S, bottoms: list) -> pd.DataFrame:
    """BRIEF 밖 진단: E_nn = E_full − (같은 L·측정, 가격 특성 4개가 가장 가까운 NN_K일의 E_full 평균).
    후보 날 = B10 바닥 ±NN_EXCL 세션 밖이고 질의 창과 겹치지 않는 날(|Δ세션| ≥ L). 특성은 후보 안 표준화."""
    pos = pd.Series(np.arange(len(S)), index=S)
    bpos = np.array([pos[pd.Timestamp(b)] for b in bottoms])
    base, dmean, nret, ndd = (np.full(len(q), np.nan) for _ in range(4))
    for (L, m), idx in q.groupby(["L", "measure"]).groups.items():
        pool = mw[(mw["L"] == L) & (mw["measure"] == m) & mw["E_full"].notna()]
        Fp = pf.loc[pool["date"], feat_cols(L)].values
        pp = pos[pool["date"]].values
        keep = np.isfinite(Fp).all(1) & (np.abs(pp[:, None] - bpos[None, :]).min(1) > NN_EXCL)
        Fp, Ep, pp = Fp[keep], pool["E_full"].values[keep], pp[keep]
        mu, sd = Fp.mean(0), Fp.std(0)
        Zp = (Fp - mu) / sd
        for i in idx:
            d, e = q.at[i, "end_date"], q.at[i, "E_full"]
            if pd.isna(d) or not np.isfinite(e):
                continue
            zq = (pf.loc[d, feat_cols(L)].values.astype(float) - mu) / sd
            dist = ((Zp - zq) ** 2).sum(1)
            dist[np.abs(pp - pos[d]) < L] = np.inf
            nn = np.argpartition(dist, NN_K)[:NN_K]
            base[i], dmean[i] = Ep[nn].mean(), np.sqrt(dist[nn]).mean()
            nret[i], ndd[i] = Fp[nn, 0].mean(), Fp[nn, 2].mean()  # 이웃 평균 창 수익·낙폭(맞춤 품질)
    q = q.copy()
    q["E_nn_base"], q["E_nn"] = base, q["E_full"] - base
    q["nn_dist"], q["nn_ret"], q["nn_dd60"] = dmean, nret, ndd
    return q


def extreme_check(mw: pd.DataFrame, pf: pd.DataFrame, S, bottoms: list) -> dict:
    """E 직선 맞춤의 극단 수익 치우침(진단): 가장 큰 하락일 10%(L 1)와 낙폭 10% 이상 중 5일 수익 상위 10%(L 5)."""
    pos = pd.Series(np.arange(len(S)), index=S)
    bpos = np.array([pos[pd.Timestamp(b)] for b in bottoms])
    out = {}
    for key, L, m, dec, dd in (("x3_down", 1, "X3", 0, False), ("x1_rebound", 5, "X1", 9, True)):
        s = mw[(mw["L"] == L) & (mw["measure"] == m) & mw["E_full"].notna()].copy()
        s["ret"] = pf[f"ret_{L}"].reindex(s["date"]).values
        s["dd"] = pf["dd60"].reindex(s["date"]).values
        s["dec"] = pd.qcut(s["ret"], 10, labels=False)
        s["near"] = np.abs(pos[s["date"]].values[:, None] - bpos[None, :]).min(1) <= NN_EXCL
        x = s[(s["dec"] == dec) & ((s["dd"] <= -0.10) if dd else True)]
        out[key] = {"n": int(len(x)), "n_near": int(x["near"].sum()), "E_all": float(x["E_full"].mean()),
                    "E_near": float(x.loc[x["near"], "E_full"].mean()), "E_far": float(x.loc[~x["near"], "E_full"].mean())}
    return out


# ---------- 칸 통계 ----------
def p_cell(P: np.ndarray, thr: int) -> dict:
    P = P[np.isfinite(P)]
    if not len(P):
        return {"n": 0, "side": "hi", "k_side": 0, "flag": ""}
    n_hi, n_lo, med = int((P >= HI).sum()), int((P <= LO).sum()), float(np.median(P))
    side = "hi" if (n_hi > n_lo or (n_hi == n_lo and med >= 50)) else "lo"
    flag = "공통 높음" if n_hi >= thr else ("공통 낮음" if n_lo >= thr else "")
    if flag:
        side = "hi" if flag == "공통 높음" else "lo"
    return {"n": len(P), "n_hi": n_hi, "n_lo": n_lo, "med": med, "side": side,
            "k_side": n_hi if side == "hi" else n_lo, "flag": flag}


def e_cell(E: np.ndarray, thr: int, side: str | None = None) -> dict:
    """E 규칙: 같은 부호 thr개 이상 + 중앙 |z| ≥ 0.5. side를 주면 그 쪽으로 센다(진단용)."""
    E = E[np.isfinite(E)]
    if not len(E):
        return {"n": 0, "side": side or "+", "k_side": 0, "flag": ""}
    n_pos, n_neg = int((E > 0).sum()), int((E < 0).sum())
    med, mabs = float(np.median(E)), float(np.median(np.abs(E)))
    if side is None:
        side = "+" if (n_pos > n_neg or (n_pos == n_neg and med >= 0)) else "-"
    k = n_pos if side == "+" else n_neg
    same = (E > 0) if side == "+" else (E < 0)
    flag = f"공통 {side}" if (k >= thr and mabs >= ZMIN) else ""
    return {"n": len(E), "n_pos": n_pos, "n_neg": n_neg, "med": med, "med_absz": mabs, "side": side,
            "k_side": k, "k_side_z05": int((same & (np.abs(E) >= ZMIN)).sum()), "flag": flag}


def hit(v: np.ndarray, norm: str, side: str) -> np.ndarray:
    if norm == "P":
        return v >= HI if side == "hi" else v <= LO
    return v > 0 if side == "+" else v < 0


def f_share(v: np.ndarray, ep: np.ndarray, norm: str, side: str) -> dict:
    """F 동반율: F 날을 k = 0으로 둔 같은 창 값이 같은 쪽인 비율(날 단위, 국면 평균)."""
    ok = np.isfinite(v)
    v, ep = v[ok], ep[ok]
    if not len(v):
        return {"F_n": 0, "F_k": 0, "F_share": np.nan}
    h = hit(v, norm, side)
    out = {"F_n": int(len(v)), "F_ep": int(len(set(ep))), "F_k": int(h.sum()), "F_share": float(h.mean()),
           "F_share_ep": float(pd.Series(h).groupby(ep).mean().mean()), "F_med": float(np.median(v))}
    if norm == "E":
        out["F_share_z05"] = float((h & (np.abs(v) >= ZMIN)).mean())
    return out


def vals_str(df: pd.DataFrame, col: str) -> str:
    return " ".join(f"{x:.0f}" if col == "P" else f"{x:+.2f}" for x in df[col].values) if len(df) else ""


def chance(n: int, thr: int, rng) -> dict:
    """사건이 독립이고 흐름과 무관할 때 한 칸이 '공통'으로 걸릴 확률(P: 한쪽 0.2 이항 × 2쪽, E: 표준정규 모의)."""
    p_one = stats.binom.sf(thr - 1, n, 0.2)
    z = rng.standard_normal((NSIM, n))
    k = np.maximum((z > 0).sum(1), (z < 0).sum(1))
    e = float(((k >= thr) & (np.median(np.abs(z), 1) >= ZMIN)).mean())
    return {"P_cell": float(2 * p_one), "E_cell": e}


# ---------- 표 1: 사건 × 측정 × 창 + 하락 누적 ----------
def event_table(L_ev: pd.DataFrame, evs: pd.DataFrame, md: pd.DataFrame, pf: pd.DataFrame, S) -> pd.DataFrame:
    t = L_ev.copy()
    t["k_start"] = t["k_end"] - t["L"] + 1
    t["start_date"] = [S[S.get_loc(a) + k] for a, k in zip(t["anchor"], t["k_start"])]
    t = t.rename(columns={"anchor": "event"})
    t["kind"] = "window"
    keep = ["event", "B15", "year", "kind", "window", "k_start", "k_end", "L", "start_date", "end_date", "measure",
            "value", "P", "nP", "E_full", "E_exp", "E_nn", "E_nn_base", "n_merged", "n_cens", "win_ret", "win_dd60"]
    t = t[keep]
    pos = pd.Series(np.arange(len(S)), index=S)
    rows = []
    for e in evs.itertuples():
        tops = [("D10", e.top10)] + ([("D15", e.top15)] if e.B15 else [])
        for lab, top in tops:
            top, bot = pd.Timestamp(top), pd.Timestamp(e.event)
            seg = md.loc[(md.index >= top) & (md.index <= bot)]
            i0 = int(pos[top])
            pre = md["A1"].reindex(S[i0 - 60:i0])  # 고점 앞 60세션(고점 제외) 하루 A1
            med_a1 = float(pre.median())
            n_ses = int(pos[bot] - pos[top] + 1)
            for m in DECL_M:
                cum = float(seg[m].sum())
                rows.append({"event": e.event, "B15": e.B15, "year": e.year, "kind": "decline", "window": lab,
                             "k_start": int(pos[top] - pos[bot]), "k_end": 0, "L": n_ses, "start_date": top,
                             "end_date": bot, "measure": m, "value": cum,
                             "n_merged": float(seg["merged"].sum()),
                             "n_cens": float(seg[DECL_CENS[m]].sum()) if DECL_CENS[m] else 0.0,
                             "win_ret": float(np.log(pf.at[bot, "SOXX_Close"] / pf.at[top, "SOXX_Close"])),
                             "soxx_ret": float(pf.at[bot, "SOXX_Close"] / pf.at[top, "SOXX_Close"] - 1),
                             "pre60_med_A1": med_a1, "pre60_n": int(pre.notna().sum()),
                             "mult_medA1": cum / med_a1, "mult_medA1_per_session": cum / med_a1 / n_ses})
    out = pd.concat([t, pd.DataFrame(rows)], ignore_index=True)
    for c in ("start_date", "end_date"):
        out[c] = pd.to_datetime(out[c]).dt.date
    return out


# ---------- 표 2: 날마다 경로 ----------
def daily_paths(evs, mw, md, pf, S) -> pd.DataFrame:
    w1 = mw[mw["L"] == 1].pivot(index="date", columns="measure", values=["value", "P"])
    pos = pd.Series(np.arange(len(S)), index=S)
    rows = []
    for e in evs.itertuples():
        b = pd.Timestamp(e.event)
        cb = float(pf.at[b, "SOXX_Close"])
        for k in range(-20, 21):
            d = S[int(pos[b]) + k]
            has = d in md.index
            r = {"event": e.event, "B15": e.B15, "k": k, "date": d.date(),
                 "soxx_close": float(pf.at[d, "SOXX_Close"]), "soxx_rel_bottom": float(pf.at[d, "SOXX_Close"] / cb - 1),
                 "merged": bool(md.at[d, "merged"]) if has else None,
                 "unit_id": int(md.at[d, "unit_id"]) if has else None,
                 "unit_n_trades": int(md.at[d, "unit_n_trades"]) if has else None,
                 "cens_SOXL": bool(md.at[d, "cens_SOXL"]) if has else None,
                 "cens_SOXS": bool(md.at[d, "cens_SOXS"]) if has else None,
                 "c_stale": bool(md.at[d, "c_stale"]) if has else None}
            for m in PATH_M:
                r[f"{m}_v"] = float(w1.at[d, ("value", m)]) if d in w1.index else np.nan
                r[f"{m}_P"] = float(w1.at[d, ("P", m)]) if d in w1.index else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


# ---------- 표 3: 공통 칸 ----------
def commonality(L_ev: pd.DataFrame, L_f: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sname, sub in (("B15", L_ev[L_ev["B15"]]), ("B10", L_ev)):
        thr = THR[sname]
        for m in MCODES:
            for w in WINS:
                c = sub[(sub["measure"] == m) & (sub["window"] == w)].sort_values("anchor")
                f = L_f[(L_f["measure"] == m) & (L_f["window"] == w)]
                um = c[c["n_merged"] == 0]  # 민감도: 창 안 합친 묶음 없는 사건만
                for norm, col in (("P", "P"), ("E", "E_full")):
                    if norm == "P":
                        st = p_cell(c[col].values, thr)
                        su = p_cell(um[col].values, 0)
                        k_um = su.get("n_hi", 0) if st["side"] == "hi" else su.get("n_lo", 0)
                        nn = {}
                    else:
                        st = e_cell(c[col].values, thr)
                        k_um = e_cell(um[col].values, 0, st["side"])["k_side"]
                        en = e_cell(c["E_nn"].values, thr, st["side"])
                        fn = f_share(f["E_nn"].values.astype(float), f["episode"].values, "E", st["side"])
                        nn = {"nn_k_side": en["k_side"], "nn_med": en.get("med", np.nan),
                              "nn_med_absz": en.get("med_absz", np.nan), "nn_pass": bool(en["flag"]),
                              "F_nn_share": fn["F_share"], "F_nn_med": fn.get("F_med", np.nan)}
                    fs = f_share(f[col].values.astype(float), f["episode"].values, norm, st["side"])
                    rows.append({"set": sname, "measure": m, "name": NAME[m], "window": w, "norm": norm,
                                 "thr": thr, **st, **fs, **nn, "um_n": len(um), "um_k_side": k_um,
                                 "n_merged_W": int((c["n_merged"] > 0).sum()),
                                 "n_cens_W": int((c["n_cens"] > 0).sum()), "event_vals": vals_str(c, col)})
    t = pd.DataFrame(rows)
    t["bottom_share"] = t["k_side"] / t["n"]
    # 메아리 가능성: P 공통인데 같은 방향 E 공통이 없으면 '메아리 가능'
    e = t[t["norm"] == "E"].set_index(["set", "measure", "window"])
    p = t[t["norm"] == "P"].set_index(["set", "measure", "window"])
    lab, ez, ek = [], [], []
    for r in t.itertuples():
        if r.norm == "P":
            er = e.loc[(r.set, r.measure, r.window)]
            want = "+" if r.side == "hi" else "-"
            ez.append(er["med"])
            ek.append(int(er["n_pos"] if want == "+" else er["n_neg"]))
            if not r.flag:
                lab.append("")
            elif er["flag"] == f"공통 {want}":
                lab.append("E도 공통(메아리 넘음)")
            elif er["flag"]:
                lab.append("E 반대 공통")
            else:
                lab.append("메아리 가능(E 공통 아님)")
        else:
            ez.append(np.nan)
            ek.append(np.nan)
            want = "공통 높음" if r.side == "+" else "공통 낮음"
            pf_ = p.loc[(r.set, r.measure, r.window), "flag"]
            lab.append("" if not r.flag else ("P도 공통" if pf_ == want else "가격 보정 뒤에만(P 공통 아님)"))
    t["E_med_same_cell"], t["E_k_dir_same_cell"], t["echo"] = ez, ek, lab
    t["brief_rule"] = np.where(t["norm"] == "P", "BRIEF §6 공통(P)",
                               np.where(t["set"] == "B15", "BRIEF §6 공통(E)", "해석: B10에 E 규칙을 9/11로 맞춤"))
    cols = ["set", "measure", "name", "window", "norm", "brief_rule", "thr", "n", "flag", "side", "k_side",
            "bottom_share", "n_hi", "n_lo", "n_pos", "n_neg", "med", "med_absz", "k_side_z05",
            "F_n", "F_ep", "F_k", "F_share", "F_share_ep", "F_share_z05", "F_med", "echo",
            "E_med_same_cell", "E_k_dir_same_cell", "nn_k_side", "nn_med", "nn_med_absz", "nn_pass",
            "F_nn_share", "F_nn_med", "um_n", "um_k_side", "n_merged_W", "n_cens_W", "event_vals"]
    return t[cols]


# ---------- 표 4: 2023~2025로 고르고 2026에서 보기 ----------
def holdout(L_ev: pd.DataFrame, L_f: pd.DataFrame) -> pd.DataFrame:
    tr, te = L_ev[L_ev["year"] <= 2025], L_ev[L_ev["year"] == 2026]
    ftr, fte = L_f[L_f["year"] <= 2025], L_f[L_f["year"] == 2026]
    ev26 = sorted(te["anchor"].unique())
    rows = []
    for m in MCODES:
        for w in WINS:
            c = tr[(tr["measure"] == m) & (tr["window"] == w)].sort_values("anchor")
            for norm, col in (("P", "P"), ("E", "E_full")):
                st = p_cell(c[col].values, THR["T7"]) if norm == "P" else e_cell(c[col].values, THR["T7"])
                r = {"measure": m, "name": NAME[m], "window": w, "norm": norm,
                     "use": "W1~W3(바닥 때까지 아는 창)" if w in ("W1", "W2", "W3") else "W4~W5(설명용)",
                     "train_n": st["n"], "train_side": st["side"], "train_k": st["k_side"], "train_med": st["med"],
                     "train_med_absz": st.get("med_absz", np.nan), "selected": bool(st["flag"]),
                     "train_vals": vals_str(c, col),
                     "strength": abs(stats.norm.ppf(np.clip(st["med"] / 100, 1e-6, 1 - 1e-6))) if norm == "P"
                     else st["med_absz"]}
                if r["selected"]:
                    side = st["side"]
                    x = te[(te["measure"] == m) & (te["window"] == w)].set_index("anchor")
                    same = []
                    for d in ev26:
                        v = float(x.at[d, col])
                        r[f"v_{iso(d)}"] = v
                        r[f"same_{iso(d)}"] = bool(hit(np.array([v]), norm, side)[0])
                        same.append(r[f"same_{iso(d)}"])
                        if norm == "E":
                            r[f"Eexp_{iso(d)}"] = float(x.at[d, "E_exp"])
                            r[f"Enn_{iso(d)}"] = float(x.at[d, "E_nn"])
                    r["n26"], r["k26_same"] = len(ev26), int(sum(same))
                    if norm == "E":
                        r["k26_same_z05"] = int(sum(1 for d in ev26 if r[f"same_{iso(d)}"] and abs(r[f"v_{iso(d)}"]) >= ZMIN))
                        r["k26_same_Eexp"] = int(sum(bool(hit(np.array([r[f"Eexp_{iso(d)}"]]), "E", side)[0]) for d in ev26))
                        r["k26_same_Enn"] = int(sum(bool(hit(np.array([r[f"Enn_{iso(d)}"]]), "E", side)[0]) for d in ev26))
                    for lab, fdf in (("F26", fte), ("F2325", ftr)):
                        f = fdf[(fdf["measure"] == m) & (fdf["window"] == w)]
                        fs = f_share(f[col].values.astype(float), f["episode"].values, norm, side)
                        r[f"{lab}_n"], r[f"{lab}_k"], r[f"{lab}_share"] = fs["F_n"], fs["F_k"], fs["F_share"]
                rows.append(r)
    t = pd.DataFrame(rows)
    # 주 확인(최대 5칸, BRIEF §8): W1~W3에서 고른 칸을 2023~25 같은 쪽 수 → 세기(z 척도) 순으로. 2026 값은 보지 않고 정함
    cand = t[t["selected"] & t["window"].isin(["W1", "W2", "W3"])].sort_values(["train_k", "strength"],
                                                                                ascending=[False, False])
    t["primary_rank"] = pd.Series(np.arange(1, len(cand) + 1), index=cand.index).reindex(t.index)
    t["primary"] = t["primary_rank"] <= 5
    t["note"] = np.where(t["selected"], "2026-07-29는 B2 막차 창(7/16 발표·7/31 시행)과 겹침", "")
    return t


# ---------- 요약 ----------
def pc(x) -> str:
    return "" if pd.isna(x) else f"{100 * x:.0f}%"


SYM = {"hi": "P↑", "lo": "P↓", "+": "E+", "-": "E−"}


def kk(r, side=None) -> int:
    """칸 행에서 주어진 쪽(없으면 그 행의 쪽) 개수."""
    side = side or r["side"]
    col = {"hi": "n_hi", "lo": "n_lo", "+": "n_pos", "-": "n_neg"}[side]
    return int(r[col]) if pd.notna(r[col]) else 0


def matrix_cell(cell: pd.DataFrame) -> str:
    """한 (측정, 창)의 B15·B10 공통 표지를 한 칸 글로. 앞세우는 묶음 = B10(문턱 9/11)이 걸리면 B10, 아니면 B15."""
    out = []
    for norm in ("P", "E"):
        r15, r10 = cell.loc[("B15", norm)], cell.loc[("B10", norm)]
        if not (r15["flag"] or r10["flag"]):
            continue
        lead = r10 if r10["flag"] else r15
        side = lead["side"]
        val = f"{lead['med']:.0f}" if norm == "P" else f"{lead['med']:+.2f}"
        if norm == "P":
            tag = {"E도 공통(메아리 넘음)": "넘", "E 반대 공통": "반"}.get(lead["echo"], "메")
        else:
            tag = "남" if lead["nn_pass"] else "약"
        s15 = f"{kk(r15, side)}/6{'*' if r15['flag'] else ''}"
        s10 = f"{kk(r10, side)}/11{'*' if r10['flag'] else ''}"
        out.append(f"{SYM[side]} {s15}·{s10} {val} F{pc(lead['F_share'])} {tag}")
    return "<br>".join(out)


def kdist(s: pd.Series) -> str:
    return ", ".join(f"{k}/4 {v}칸" for k, v in s.astype(int).value_counts().sort_index(ascending=False).items())


def write_summary(com, ho, et, info) -> list[str]:
    c = com.set_index(["set", "norm", "measure", "window"])

    def g(s, n, m, w):
        return c.loc[(s, n, m, w)]

    nfl = {(s, n): int((com[(com["set"] == s) & (com["norm"] == n)]["flag"] != "").sum())
           for s in ("B15", "B10") for n in ("P", "E")}
    ch, xc = info["chance"], info["extreme"]
    d10 = et[(et["kind"] == "decline") & (et["window"] == "D10")]
    dec = d10.pivot(index="event", columns="measure", values="value")
    decm = d10.pivot(index="event", columns="measure", values="mult_medA1")
    # 자주 쓰는 칸(B10)
    x1, x2, l1, h2, c1 = (g("B10", "P", m, "W3") for m in ("X1", "X2", "L1", "H2", "C1"))
    x6 = g("B10", "P", "X6", "W3")
    x3e, x3e2 = g("B10", "E", "X3", "W3"), g("B10", "E", "X3", "W2")
    x5p, x5e = g("B10", "P", "X5", "W3"), g("B10", "E", "X5", "W3")
    r5e, x1e = g("B10", "E", "R5", "W3"), g("B10", "E", "X1", "W3")
    x3p, h1p = g("B10", "P", "X3", "W3"), g("B10", "P", "H1", "W3")
    a1p, a1e = g("B10", "P", "A1", "W3"), g("B10", "E", "A1", "W3")
    echoF = [r["F_share"] for r in (x1, x2, l1, h2, c1)]
    echoK = [kk(r) for r in (x1, x2, l1, h2, c1)]
    x1e26 = [float(v) for v in x1e["event_vals"].split()[-4:]]

    L = ["# SEIBRO 바닥 공통점 — profile 갈래 (설명용, 자료판 2026-11-04 07:58 `--merge-rule cal`)"]
    L.append(f"**한 줄**: 바닥 당일 SOXL 순매수·매수·3배 롱·강세 몫·좌수 증가 P ≥ 80은 B10 {min(echoK)}~{max(echoK)}/11로 늘 나오지만 "
             f"가짜 바닥(F) 날도 {pc(min(echoF))}~{pc(max(echoF))}라 가격 메아리다. 메아리(E)를 넘는 B10 공통은 "
             f"당일 SOXL 매도 E+ {kk(x3e)}/11(F {pc(x3e['F_share'])})·직전 5일 SOXL 매도 E+ {kk(x3e2)}/11(F {pc(x3e2['F_share'])})·"
             f"당일 인컴 순매수 E− {kk(r5e)}/11(F {pc(r5e['F_share'])}) 정도이고 모두 설명용이다. 바닥 날 SOXL 순매도(투매)는 "
             f"{int(x1['n_lo'])}/11이다.")
    L.append("")
    L.append(f"- 사건 B15 {info['nB15']}개·B10 {info['nB10']}개, F {info['nF']}날({info['nFep']}국면; W4 {info['nF_W4']}날·"
             f"W5 {info['nF_W5']}날, 끝 쪽은 자료 밖). 창 W1 [−20,−6]·W2 [−5,−1]·W3 [0]·W4 [+1,+5]·W5 [+6,+20]. "
             "P = 250세션 백분위, E = 메아리 잔차 z(전체 적합). F 동반율 = F 날을 k = 0으로 둔 같은 창이 같은 쪽(P ≥ 80/≤ 20, E 같은 부호)인 비율. 바닥 쪽 비율(k/n)과 견준다. "
             "바닥 날(W3)이 합친 묶음 안인 사건 0개.")
    L.append("")
    L.append("## 1. 공통 칸 (B15 5/6 또는 B10 9/11 문턱, * = 통과)")
    L.append("| 측정 | W1 [−20,−6] | W2 [−5,−1] | W3 바닥 | W4 [+1,+5] | W5 [+6,+20] |")
    L.append("|---|---|---|---|---|---|")
    for m in MCODES:
        cells = []
        for w in WINS:
            cell = com[(com["measure"] == m) & (com["window"] == w)].set_index(["set", "norm"])
            cells.append(matrix_cell(cell))
        if any(cells):
            L.append(f"| {m} {NAME[m]} | " + " | ".join(cells) + " |")
    L.append("- 읽기: `P↑ 6/6*·11/11* 95 F86% 메` = B15 6개 중 6·B10 11개 중 11이 P ≥ 80, 중앙 P 95(B10 통과면 B10 값), "
             "F 동반율 86%. 메 = 메아리 가능(같은 방향 E 공통 없음), 넘 = E도 공통, 반 = E는 반대 공통. "
             "E 칸 남/약 = 같은 가격 상태 30일 평균 E를 뺀 진단(BRIEF 밖)에서도 문턱 유지/미달.")
    L.append("")
    L.append("## 2. 전·당일·후 (B10 기준, 괄호 F = F 동반율)")

    def pw(w):
        f = com[(com["window"] == w) & (com["norm"] == "P") & (com["flag"] != "")]
        return f"{', '.join(sorted(set(f['measure'])))} (F {pc(f['F_share'].min())}~{pc(f['F_share'].max())})" if len(f) else "없음"
    L.append(f"- **전**: P 공통 W1 {pw('W1')}, W2 {pw('W2')} — 대부분 B15만 통과하고 메아리. "
             f"E로는 W2 SOXL 매도 E+ {kk(x3e2)}/11(중앙 {x3e2['med']:+.2f}, F {pc(x3e2['F_share'])}, 이웃 진단 뒤 "
             f"{int(x3e2['nn_k_side'])}/11·F {pc(x3e2['F_nn_share'])})가 F와 가장 크게 갈린다.")
    L.append(f"- **당일**: SOXL 순매수 P↑ {kk(x1)}/11(F {pc(x1['F_share'])}), 매도÷매수 P↓ {kk(x6)}/11(F {pc(x6['F_share'])}) — 메아리. "
             f"SOXL 매도 E+ {kk(x3e)}/11(중앙 {x3e['med']:+.2f}, F {pc(x3e['F_share'])}; 이웃 진단 뒤 {int(x3e['nn_k_side'])}/11·F "
             f"{pc(x3e['F_nn_share'])}). SOXL 거래 몫 P↑ {kk(x5p)}/11·E+ {kk(x5e)}/11(F {pc(x5p['F_share'])}·{pc(x5e['F_share'])}). "
             f"인컴 순매수 E− {kk(r5e)}/11(중앙 {r5e['med']:+.2f}, F {pc(r5e['F_share'])}).")
    L.append(f"- **투매 없음**: SOXL 순매수 P ≤ 20 {int(x1['n_lo'])}/11, SOXL 매도 P ≥ 80 {int(x3p['n_hi'])}/11, "
             f"SOXS 순매수 P ≥ 80 {int(h1p['n_hi'])}/11·P ≤ 20 {int(h1p['n_lo'])}/11, 전체 거래 P ≥ 80 {int(a1p['n_hi'])}/11"
             f"(E+ {kk(a1e, '+')}/11이나 중앙 {a1e['med']:+.2f}로 문턱 미달). SOXL 순매수 E는 − 쪽 {kk(x1e, '-')}/11"
             f"(중앙 {x1e['med']:+.2f}, 2026 {' '.join(f'{v:+.2f}' for v in x1e26)}): 가격이 말하는 것보다 덜 산다(공통 아님).")
    w4 = [m for m in ("X1", "X2", "X4", "L1", "L2", "L3", "L4", "C1") if g("B10", "E", m, "W4")["flag"]]
    L.append(f"- **후**: W4 SOXL 매도 P↑ {kk(g('B10', 'P', 'X3', 'W4'))}/11는 반등 메아리. W4 E+ 공통 {', '.join(w4)}·X6 E−는 "
             f"'반등에도 덜 판다'로 보이나, 낙폭 10% 이상 날 중 5일 수익 상위 10%({xc['x1_rebound']['n']}날)의 X1 E 평균도 "
             f"{xc['x1_rebound']['E_all']:+.2f}(바닥 ±{NN_EXCL} 밖 {xc['x1_rebound']['E_far']:+.2f})라 직선 보정 치우침이 섞인다. "
             "W5 E 공통은 B15 우연 기대 수준. "
             f"**하락 누적**(10% 고점→바닥): SOXL 순매수 > 0 {int((dec['X1'] > 0).sum())}/11(고점 앞 60세션 하루 A1 중앙의 "
             f"{decm['X1'].min():.2f}~{decm['X1'].max():.2f}배), SOXS 순매도 {int((dec['H1'] < 0).sum())}/11, "
             f"3배 롱 > 0 {int((dec['L1'] > 0).sum())}/11. X1 ÷ A1은 2023-10 "
             f"{dec.at['2023-10-30', 'X1'] / dec.at['2023-10-30', 'A1']:.1%} → 2026-07 "
             f"{dec.at['2026-07-29', 'X1'] / dec.at['2026-07-29', 'A1']:.1%}, 2026-09 "
             f"{dec.at['2026-09-14', 'X1'] / dec.at['2026-09-14', 'A1']:.1%}.")
    L.append("")
    sel = ho[ho["selected"]]
    L.append("## 3. 2026 확인 (2023~25 B10 7개로 고름: P 6/7 같은 쪽, E 6/7 같은 부호 + 중앙 |z| ≥ 0.5)")
    L.append("| 칸 | 2023~25 | 2026 바닥 3/30·6/5·7/29·9/14 | 2026 F | 2023~25 F |")
    L.append("|---|---|---|---|---|")
    days = ["2026-03-30", "2026-06-05", "2026-07-29", "2026-09-14"]
    b10e = {(r.measure, r.window) for r in com[(com["set"] == "B10") & (com["norm"] == "E") & (com["flag"] != "")].itertuples()}
    is_b10e = pd.Series([(m, w) in b10e for m, w in zip(sel["measure"], sel["window"])], index=sel.index)
    show = sel[sel["primary"] | ((sel["norm"] == "E") & sel["window"].isin(["W1", "W2", "W3"]) & is_b10e)]
    show = show.sort_values(["primary_rank", "window", "measure"], na_position="last")
    for r in show.itertuples():
        vals = [ho.at[r.Index, f"v_{d}"] for d in days]
        vs = " ".join(f"{v:.0f}" if r.norm == "P" else f"{v:+.2f}" for v in vals)
        tv = f"{int(r.train_k)}/7 ({r.train_med:.0f})" if r.norm == "P" else f"{int(r.train_k)}/7 ({r.train_med:+.2f})"
        lab = f"주{int(r.primary_rank)} " if r.primary else ""
        L.append(f"| {lab}{r.measure} {r.name} {r.window} {SYM[r.train_side]} | {tv} | {int(r.k26_same)}/4 ({vs}) | "
                 f"{int(r.F26_k)}/{int(r.F26_n)} {pc(r.F26_share)} | {pc(r.F2325_share)} |")
    early = sel[sel["window"].isin(["W1", "W2", "W3"])]
    late = sel[sel["window"].isin(["W4", "W5"])]
    ek = early[early["norm"] == "E"]
    L.append(f"- 고른 칸 {len(sel)}개(W1~W3 {len(early)}: P {int((early['norm'] == 'P').sum())}·E {len(ek)}; W4~W5 설명용 "
             f"{len(late)}). 주 5칸(BRIEF §8 상한, 2023~25 값만으로 순위)은 모두 P 메아리 칸이다. W1~W3 E {len(ek)}칸의 "
             f"2026 같은 쪽: {kdist(ek['k26_same'])}. W4~W5: {kdist(late['k26_same'])}. 위 표 밖 행은 holdout_2026.csv. "
             "7/29는 B2 막차 창(7/16 발표·7/31 시행)과 겹친다.")
    L.append("## 4. 본 칸 수 · 우연 기대 · 주의")
    L.append(f"- 공통 판정 520칸 = 측정 26 × 창 5 × (P·E) × (B15·B10), 520칸 모두 F 동반율. 걸린 칸: B15 P {nfl[('B15', 'P')]}·"
             f"E {nfl[('B15', 'E')]}, B10 P {nfl[('B10', 'P')]}·E {nfl[('B10', 'E')]}. 사건 독립·흐름 무관일 때 우연 기대: "
             f"B15 P {130 * ch['B15']['P_cell']:.1f}·E {130 * ch['B15']['E_cell']:.1f}, B10 P {130 * ch['B10']['P_cell']:.3f}·"
             f"E {130 * ch['B10']['E_cell']:.1f}칸 — B15 E 규칙은 우연으로도 약 {130 * ch['B15']['E_cell']:.0f}칸이 걸린다. "
             "E의 '중앙 |z|'는 |z|의 중앙값으로 읽었다. P 칸 수는 우연이 아니라 메아리로 읽는다. 측정끼리 상관(X1·X2·X4·L1·L2·C1)이 커서 칸 수 ≠ 독립 발견 수.")
    L.append(f"- 2026 고르기 260칸(26 × 5 × 2) → {len(sel)}칸(우연 기대 P {130 * ch['T7']['P_cell']:.1f}·E "
             f"{130 * ch['T7']['E_cell']:.1f}칸), {len(sel)}칸 모두 2026 바닥 4개·2026 F {int(ho['F26_n'].max())}날에서 확인. "
             f"BRIEF 밖 진단: E 260칸에 가격 이웃 보정(이웃 {NN_K}일, ret·pre20·dd60·rv20, 바닥 ±{NN_EXCL} 제외)·합친 묶음 뺀 민감도. "
             f"설명용: 하락 누적 {len(DECL_M)}측정 × (B10 11 + B15 고점 6), 날마다 경로 {len(PATH_M)}측정 × 41일 × 11.")
    L.append(f"- E는 직선 잔차라 극단 수익일에 치우친다: 가장 큰 하락일 10%({xc['x3_down']['n']}날)의 X3 E 평균 "
             f"{xc['x3_down']['E_all']:+.2f}(바닥 ±{NN_EXCL} 밖 {xc['x3_down']['E_far']:+.2f}) — 당일 X3 E+는 '급락일 일반'과 섞인다. "
             "바닥·F는 사후 표지이고 바닥 대 F 시험은 contrast 갈래 몫. F는 2025-04 국면 8날에 몰림(국면 평균은 `F_share_ep`). "
             f"절단: 사건 창 안 절단 날 있는 칸 {info['n_cens_cells']}개(측정 {info['cens_measures']}), SOXL은 늘 목록 안.")
    return L


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    mw, md, pf, ev = load()
    S = pf.index
    evs = pd.DataFrame([{"event": e["bottom"], "B15": bool(e["B15"]), "top10": e["top10"], "top15": e.get("top15")}
                        for e in ev["B10_events"]])
    evs["year"] = evs["event"].str[:4].astype(int)
    Fd = pd.DataFrame([f for f in ev["F"] if f["has_flows"]])
    Fd["year"] = Fd["date"].str[:4].astype(int)

    a_ev = pd.DataFrame({"anchor": pd.to_datetime(evs["event"]), "B15": evs["B15"], "year": evs["year"]})
    a_f = pd.DataFrame({"anchor": pd.to_datetime(Fd["date"]), "episode": Fd["episode"], "year": Fd["year"]})
    L_ev = nn_adjust(lookup(a_ev, S, mw, pf), mw, pf, S, ev["B10"])
    L_f = nn_adjust(lookup(a_f, S, mw, pf), mw, pf, S, ev["B10"])
    assert L_ev["P"].notna().all() and L_ev["E_full"].notna().all() and L_ev["E_nn"].notna().all(), "사건 창 값 빠짐"

    et = event_table(L_ev, evs, md, pf, S)
    et.to_csv(OUT / "event_table.csv", index=False, encoding="utf-8-sig")
    daily_paths(evs, mw, md, pf, S).to_csv(OUT / "daily_paths.csv", index=False, encoding="utf-8-sig")
    com = commonality(L_ev, L_f)
    com.to_csv(OUT / "commonality.csv", index=False, encoding="utf-8-sig")
    ho = holdout(L_ev, L_f)
    ho.to_csv(OUT / "holdout_2026.csv", index=False, encoding="utf-8-sig")

    rng = np.random.default_rng(20261103)
    cens = L_ev[L_ev["n_cens"] > 0]
    info = {"nB15": int(evs["B15"].sum()), "nB10": len(evs), "nF": len(Fd), "nFep": int(Fd["episode"].nunique()),
            "chance": {k: chance(n, THR[k], rng) for k, n in (("B15", 6), ("B10", 11), ("T7", 7))},
            "extreme": extreme_check(mw, pf, S, ev["B10"]),
            "nF_W4": int(L_f[(L_f["window"] == "W4") & (L_f["measure"] == "X1")]["P"].notna().sum()),
            "nF_W5": int(L_f[(L_f["window"] == "W5") & (L_f["measure"] == "X1")]["P"].notna().sum()),
            "n_cens_cells": int(len(cens)), "cens_measures": ",".join(sorted(cens["measure"].unique())) or "없음"}
    L = write_summary(com, ho, et, info)
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=1), "\nsummary 줄 수:", len(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
