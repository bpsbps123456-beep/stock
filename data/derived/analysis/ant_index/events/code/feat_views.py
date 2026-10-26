"""부록 13 조회·추천 특징과 누출 진단(가격 미사용).

입력(읽기만):
  (부록 14 갱신: 아래 snapshot_1023 = snap_paths.SNAP. 10/22 진단판(final 스냅샷)은 events/results/views_*와 views_leak.md)
  snapshot_1023/results/views_rows.parquet  views_fetch.py 산출(줄마다 수치 필드·수집 시각 추정)
  snapshot_1023/results/units.parquet   글·댓글 단위(창 id, 마감 전 표지, 댓글 시각)
  snapshot_1023/results/windows_kr|us.parquet   창 달력(본판·완전 창)
  snapshot_1023/results/feat_activity.parquet   창 표지(judge_ok, realtime)만
  snapshot_1023/label_key.parquet + snapshot_1023/labels/, preview/label_key.parquet + preview/labels/  예측 층 AI 라벨
산출:
  snapshot_1023/results/views_posts.parquet    글 하나에 한 줄(첫·마지막 줄 수치, 수집 나이, 마감 전 댓글)
  snapshot_1023/results/feat_views.parquet     창(W_KR·W_U) 하나에 한 줄: 부록 13 (a)(b)(c) + 추천·다른의견 + 조회 상위 10% 글 P̄
  snapshot_1023/results/views_growth.json      성장 곡선·누출 진단 수치(views_leak.md의 근거)
가격·앞날 값은 읽지 않는다. 포럼 특징과 가격을 한 표에 놓지 않는다.
실행: py feat_views.py   (views_fetch.py 먼저)
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import snap_paths as SP

HERE = Path(__file__).resolve().parent
EV = HERE.parent
ANT = EV.parent
FINAL = ANT / "final"            # 동결 코드 원천(읽기만)
PREVIEW = ANT / "preview"
RES = SP.SNAP_RES                # 부록 14: 사건 연구 스냅샷(2026-10-23 07:59)
OUT = SP.SNAP_RES

KNOW_END = pd.Timestamp("2026-07-01")
KNOTS_H = np.array([0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 20, 24, 26, 28, 30, 32])
MIN_SUPPORT = 30                 # 마디를 쓰려면 그 마디 기저가 0이 아닌 쌍이 이만큼 있어야 한다
REF_H = 24.0                     # 나이 보정 기준: '24시간 환산 조회'
LONG_BINS_D = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]   # 소급분 나이 칸(일)
MIN_POSTS = 10                   # (a) 분포 특징 최소 글 수


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n < 2 or x.sum() <= 0:
        return float("nan")
    i = np.arange(1, n + 1)
    return float(2 * (i * x).sum() / (n * x.sum()) - (n + 1) / n)


def top_share(x: np.ndarray, k: int = 10) -> float:
    x = np.sort(np.asarray(x, dtype=float))[::-1]
    return float(x[:k].sum() / x.sum()) if x.sum() > 0 else float("nan")


def pav(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """가중 단조 증가 회귀(풀 인접 위반자)."""
    y, w = list(map(float, y)), list(map(float, w))
    blocks = [[y[i], w[i], 1] for i in range(len(y))]
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            v2, w2, n2 = out.pop()
            v1, w1, n1 = out.pop()
            out.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    res = []
    for v, _, n in out:
        res += [v] * n
    return np.array(res)


def hat_basis(a_h: np.ndarray, knots_h: np.ndarray) -> np.ndarray:
    """log 나이에서 조각 직선(삼각) 기저. 범위 밖은 끝 마디로 자른다."""
    x = np.log(np.clip(a_h, knots_h[0], knots_h[-1]))
    k = np.log(knots_h)
    B = np.zeros((len(x), len(k)))
    j = np.clip(np.searchsorted(k, x, side="right") - 1, 0, len(k) - 2)
    t = (x - k[j]) / (k[j + 1] - k[j])
    B[np.arange(len(x)), j] = 1 - t
    B[np.arange(len(x)), j + 1] = t
    return B


def fit_short_curve(a1: np.ndarray, a2: np.ndarray, y: np.ndarray) -> dict:
    """같은 글 두 줄: log(v2/v1) = h(a2) − h(a1). h는 log 나이 조각 직선, h(24h) = 0. Huber IRLS → 단조 증가로 고침."""
    knots = KNOTS_H.copy()
    while True:   # 받침이 얇은 마디(기준 24h 제외)를 하나씩 뺀다
        B = hat_basis(a2, knots) - hat_basis(a1, knots)
        sup0 = (np.abs(B) > 1e-12).sum(0)
        weak = [i for i in range(len(knots)) if sup0[i] < MIN_SUPPORT and knots[i] != REF_H]
        if not weak:
            break
        knots = np.delete(knots, min(weak, key=lambda i: sup0[i]))
    ref = int(np.flatnonzero(knots == REF_H)[0])
    keep = [i for i in range(len(knots)) if i != ref]
    X = B[:, keep]
    support = (np.abs(X) > 0).sum(0)
    w = np.ones(len(y))
    for _ in range(30):
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        r = y - X @ beta
        s = 1.4826 * np.median(np.abs(r - np.median(r))) or 1.0
        c = 1.345 * s
        w_new = np.where(np.abs(r) <= c, 1.0, c / np.maximum(np.abs(r), 1e-12))
        if np.allclose(w_new, w):
            break
        w = w_new
    h = np.zeros(len(knots))
    h[keep] = beta
    sup = np.zeros(len(knots))
    sup[keep] = support
    sup[ref] = len(y)
    h_mono = pav(h, np.maximum(sup, 1))
    h_mono = h_mono - h_mono[ref]
    resid = y - (hat_basis(a2, knots) - hat_basis(a1, knots)) @ h_mono
    return {"knots_h": knots.tolist(), "knots_dropped": sorted(set(KNOTS_H.tolist()) - set(knots.tolist())), "h_raw": h.tolist(), "h": h_mono.tolist(), "support": sup.astype(int).tolist(),
            "resid_mad": float(1.4826 * np.median(np.abs(resid - np.median(resid)))), "n": int(len(y))}


def eval_curve(a_h: np.ndarray, knots_h: list, h: list) -> np.ndarray:
    return hat_basis(np.asarray(a_h, float), np.asarray(knots_h)) @ np.asarray(h)


def read_label_units(units_key: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """예측 층 라벨(최종 + 미리보기)을 pred_windows_full.read_labels 그대로 읽는다. 7/1 이후 창의 지식 기간 단위는 뺀다."""
    sys.dont_write_bytecode = True   # final/·interim/·preview/ 코드 폴더에 캐시를 쓰지 않는다(읽기만)
    spec = importlib.util.spec_from_file_location("pwf", FINAL / "code" / "pred_windows_full.py")
    pwf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pwf)
    SP.patch_final(pwf)              # 라벨 폴더·결과 폴더만 스냅샷으로(로직 그대로)
    cols = ["id", "win_id", "a", "dt_kst", "utype", "shard", "D", "judge_ok", "in_knowledge_period", "no", "cid"]
    kf = pd.read_parquet(SP.SNAP / "label_key.parquet", columns=cols).assign(src="final")
    kp = pd.read_parquet(PREVIEW / "label_key.parquet", columns=cols).assign(src="preview")
    key = pd.concat([kf, kp], ignore_index=True)
    key["dt_kst"] = pd.to_datetime(key["dt_kst"])
    m, info = pwf.read_labels(key)
    m = m[m["pred"].notna()].copy()
    m["pred"] = m["pred"].astype(int)
    late = pd.to_datetime(m["D"]) >= KNOW_END
    drop = late & (m["dt_kst"] < KNOW_END)
    m = m[~drop].copy()
    # W_U 창: 단위(글·댓글)의 자기 시각 창을 units에서 찾는다
    m = m.merge(units_key, on=["no", "cid"], how="left", validate="m:1")
    return m, {"usable_labels": int(info["usable_labels"]), "missing_labels": int(info["missing_labels"]),
               "dropped_knowledge_units_in_late_windows": int(drop.sum()), "units_used": int(len(m)),
               "post_units": int((m["utype"] == "post").sum()), "comment_units": int((m["utype"] == "comment").sum()),
               "unmatched_to_units": int(m["u_found"].isna().sum())}


def main() -> int:
    rows = pd.read_parquet(OUT / "views_rows.parquet")
    fmeta = json.loads((OUT / "views_fetch_meta.json").read_text("utf-8"))
    S = pd.Timestamp(fmeta["snapshot"]["S"])
    U = pd.read_parquet(RES / "units.parquet", columns=["no", "utype", "cid", "dt_kst", "dt_conf", "win_kr", "win_u",
                                                        "pwin_kr", "pwin_u", "pre_close_kr", "pre_close_u", "before_post",
                                                        "n_rows", "n_union"])
    wk = pd.read_parquet(RES / "windows_kr.parquet")
    wu = pd.read_parquet(RES / "windows_us.parquet")
    wins = pd.concat([wk[(wk["variant"] == "main") & wk["complete"]], wu[(wu["variant"] == "main") & wu["complete"]]],
                     ignore_index=True)
    fa = pd.read_parquet(RES / "feat_activity.parquet", columns=["win_id", "judge_ok", "realtime", "outage", "over96"])

    # ------------------------------------------------------------ 1. 글 표
    r = rows.sort_values(["no", "k"])
    first = r[r["k"] == 0].set_index("no")
    last = r.groupby("no").tail(1).set_index("no")
    P = pd.DataFrame(index=first.index)
    P["dt"] = first["dt"]
    P["n_rows"] = first["n_rows"]
    P["cls_first"] = first["cls"]
    for c in ("views", "up", "down", "ncomment", "nsaved", "nroot", "T", "T_lo", "T_hi", "age_h", "method"):
        P[f"{c}_first"] = first[c]
        P[f"{c}_last"] = last[c]
    posts_u = U[U["utype"] == "post"].set_index("no")
    P["win_kr"] = posts_u["win_kr"].reindex(P.index)
    P["win_u"] = posts_u["win_u"].reindex(P.index)
    cm = U[U["utype"] == "comment"].copy()
    P["c_pre_kr"] = cm[cm["pre_close_kr"]].groupby("no").size().reindex(P.index).fillna(0).astype(int)
    P["c_pre_u"] = cm[cm["pre_close_u"]].groupby("no").size().reindex(P.index).fillna(0).astype(int)
    P["c_union"] = cm.groupby("no").size().reindex(P.index).fillna(0).astype(int)
    cpd = cm["no"].map(P["dt"])
    age_c = (cm["dt_kst"] - cpd).dt.total_seconds() / 3600
    for lab, hh in (("c24", 24), ("c1", 1), ("c3", 3)):
        P[lab] = cm[(age_c >= 0) & (age_c < hh)].groupby("no").size().reindex(P.index).fillna(0).astype(int)
    P["c_late7d"] = cm[age_c >= 24 * 7].groupby("no").size().reindex(P.index).fillna(0).astype(int)
    P["c_late1d"] = cm[age_c >= 24].groupby("no").size().reindex(P.index).fillna(0).astype(int)
    # 수집 때까지 단 댓글(시각 기준, 합집합): 첫 줄·마지막 줄 수집 시각까지
    cmT1 = cm["no"].map(P["T_first"])
    cmT2 = cm["no"].map(P["T_last"])
    P["c_at_T1"] = cm[cm["dt_kst"] <= cmT1].groupby("no").size().reindex(P.index).fillna(0).astype(int)
    P["c_at_T2"] = cm[cm["dt_kst"] <= cmT2].groupby("no").size().reindex(P.index).fillna(0).astype(int)
    P["realtime_post"] = P["age_h_first"] < 24
    P = P.reset_index()

    # ------------------------------------------------------------ 2. 성장 곡선
    G = {}
    pair = P[(P["n_rows"] == 2) & (P["views_first"] > 0)].copy()
    G["pairs"] = {"n": int(len(pair)), "views_decrease": int((pair["views_last"] < pair["views_first"]).sum()),
                  "up_decrease": int((pair["up_last"] < pair["up_first"]).sum()),
                  "down_decrease": int((pair["down_last"] < pair["down_first"]).sum()),
                  "cls_first": pair["cls_first"].value_counts().to_dict()}
    pair = pair[pair["views_last"] >= pair["views_first"]]
    y = np.log(pair["views_last"].to_numpy(float) / pair["views_first"].to_numpy(float))
    fit = fit_short_curve(pair["age_h_first"].to_numpy(float), pair["age_h_last"].to_numpy(float), y)
    G["short_curve"] = fit
    # 나이 칸별 증가율(같은 글 두 줄)
    bins_h = [0, 1, 2, 3, 4, 6, 12, 24]
    pair["a1_bin"] = pd.cut(pair["age_h_first"], bins_h, right=False)
    tab = []
    for b, g in pair.groupby("a1_bin", observed=True):
        def q(s):
            s = s.replace([np.inf, -np.inf], np.nan).dropna()
            return {"median": round(float(s.median()), 3), "p25": round(float(s.quantile(.25)), 3),
                    "p75": round(float(s.quantile(.75)), 3)} if len(s) else {}
        tab.append({"age1_h": str(b), "n": int(len(g)), "age1_median_h": round(float(g["age_h_first"].median()), 2),
                    "age2_median_h": round(float(g["age_h_last"].median()), 2),
                    "views_ratio": q(g["views_last"] / g["views_first"]),
                    "views_sum_ratio": round(float(g["views_last"].sum() / g["views_first"].sum()), 3),
                    "views_ratio_top10pct_posts": q((g["views_last"] / g["views_first"])[g["views_last"] >= g["views_last"].quantile(.9)]),
                    "up_ratio(up1>0)": q((g["up_last"] / g["up_first"])[g["up_first"] > 0]),
                    "up_added_median": float((g["up_last"] - g["up_first"]).median()),
                    "down_added_mean": round(float((g["down_last"] - g["down_first"]).mean()), 3),
                    "ncomment_ratio(nc1>0)": q((g["ncomment_last"] / g["ncomment_first"])[g["ncomment_first"] > 0]),
                    "comments_ts_ratio(c1>0)": q((g["c_at_T2"] / g["c_at_T1"])[g["c_at_T1"] > 0]),
                    "share_up_first_zero": round(float((g["up_first"] == 0).mean()), 3),
                    "share_up_last_zero": round(float((g["up_last"] == 0).mean()), 3)})
    G["pair_by_age1"] = tab
    G["age_by_class_h"] = {}
    for lab_, ser in (("forward_first", P.loc[P["cls_first"] == "forward", "age_h_first"]),
                      ("backward_first_realtime(<24h)", P.loc[(P["cls_first"] == "backward") & (P["age_h_first"] < 24), "age_h_first"]),
                      ("recheck", P.loc[P["n_rows"] == 2, "age_h_last"]),
                      ("backward_first_all", P.loc[P["cls_first"] == "backward", "age_h_first"])):
        G["age_by_class_h"][lab_] = {"n": int(len(ser)), **{str(qq): round(float(ser.quantile(qq)), 2) for qq in (0, .1, .25, .5, .75, .9, 1)}}
    # 합계 증가(10/14의 116글 기록과 같은 꼴)
    G["pair_totals"] = {"views": [int(pair["views_first"].sum()), int(pair["views_last"].sum())],
                        "up": [int(pair["up_first"].sum()), int(pair["up_last"].sum())],
                        "down": [int(pair["down_first"].sum()), int(pair["down_last"].sum())],
                        "ncomment": [int(pair["ncomment_first"].sum()), int(pair["ncomment_last"].sum())]}
    # 소급분(거꾸로 받은 한 줄 글): 나이 칸별 log((조회+1)/(24시간 댓글+1)) 중앙값 — 기준은 실시간 24h 줄
    one = P[(P["n_rows"] == 1) & (P["cls_first"] == "backward")].copy()
    one["age_d"] = one["age_h_first"] / 24
    one["lq"] = np.log((one["views_first"] + 1) / (one["c24"] + 1))
    one["lu"] = np.log((one["up_first"] + 1) / (one["c24"] + 1))
    pr = P[P["n_rows"] == 2].copy()
    pr["lq"] = np.log((pr["views_last"] + 1) / (pr["c24"] + 1))
    pr["lu"] = np.log((pr["up_last"] + 1) / (pr["c24"] + 1))
    base_q, base_u = float(pr["lq"].median()), float(pr["lu"].median())
    base_age = float(pr["age_h_last"].median())
    one["age_bin"] = pd.cut(one["age_d"], LONG_BINS_D, right=False)
    lb = []
    for b, g in one.groupby("age_bin", observed=True):
        lb.append({"age_d": str(b), "n": int(len(g)), "age_median_d": round(float(g["age_d"].median()), 2),
                   "post_dates": f"{g['dt'].min().date()}~{g['dt'].max().date()}",
                   "views_median": float(g["views_first"].median()), "views_p90": float(g["views_first"].quantile(.9)),
                   "c24_median": float(g["c24"].median()), "posts_per_day": round(float(len(g) / max(g["dt"].dt.normalize().nunique(), 1)), 1),
                   "log_views_per_c24_minus_base": round(float(g["lq"].median()) - base_q, 4),
                   "log_up_per_c24_minus_base": round(float(g["lu"].median()) - base_u, 4),
                   "late_comment_share_after1d": round(float(g["c_late1d"].sum() / max(g["c_union"].sum(), 1)), 4),
                   "late_comment_share_after7d": round(float(g["c_late7d"].sum() / max(g["c_union"].sum(), 1)), 4)})
    G["long_cross_section"] = {"base": {"n_pairs": int(len(pr)), "age_median_h": round(base_age, 2),
                                        "log_views_per_c24": round(base_q, 4), "log_up_per_c24": round(base_u, 4)},
                               "bins": lb}
    # 긴 나이 보정 곡선: 칸 중앙값을 단조 증가로 고친 값(기준 0). 24h 근처는 짧은 곡선과 잇는다
    lx = np.array([math.log(base_age)] + [math.log(x["age_median_d"] * 24) for x in lb])
    ly = np.array([0.0] + [x["log_views_per_c24_minus_base"] for x in lb])
    lw = np.array([len(pr)] + [x["n"] for x in lb], dtype=float)
    o = np.argsort(lx)
    ly_m = pav(ly[o], lw[o])
    G["long_curve"] = {"log_age_h": lx[o].tolist(), "delta_raw": ly[o].tolist(), "delta_monotone": ly_m.tolist(),
                       "note": "소급분 나이와 등록 시기가 겹친다(나이 ≈ 수집일 − 등록일). 긴 나이 쪽 값에는 시기 차이가 섞인다"}

    def g_full(a_h: np.ndarray) -> np.ndarray:
        """log g(a), g(24h) = 0. 30시간까지 짧은 곡선, 그 뒤는 긴 곡선(기준 나이에서 짧은 곡선 값에 이음)."""
        a_h = np.asarray(a_h, float)
        hs = eval_curve(a_h, fit["knots_h"], fit["h"])
        base_h = float(eval_curve(np.array([base_age]), fit["knots_h"], fit["h"])[0])
        long_ = base_h + np.interp(np.log(np.maximum(a_h, 1e-3)), lx[o], ly_m, left=0.0, right=ly_m[-1])
        return np.where(a_h <= 30, hs, np.maximum(long_, float(eval_curve(np.array([30.0]), fit["knots_h"], fit["h"])[0])))

    def g_short(a_h: np.ndarray) -> np.ndarray:
        """log g(a), 30시간 뒤로는 더 늘지 않는다고 본 판."""
        return eval_curve(np.minimum(np.asarray(a_h, float), 30.0), fit["knots_h"], fit["h"])

    P["lg_full_first"] = g_full(P["age_h_first"].to_numpy())
    P["lg_short_first"] = g_short(P["age_h_first"].to_numpy())
    P["views24_full"] = P["views_first"] / np.exp(P["lg_full_first"])
    P["views24_short"] = P["views_first"] / np.exp(P["lg_short_first"])
    G["curve_table"] = {f"{a}h": {"log_g_full": round(float(g_full(np.array([a]))[0]), 4),
                                  "g_full": round(float(np.exp(g_full(np.array([a]))[0])), 4),
                                  "g_short": round(float(np.exp(g_short(np.array([a]))[0])), 4)}
                        for a in (0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 12, 24, 26, 48, 96, 24 * 7, 24 * 30, 24 * 90, 24 * 180, 24 * 365)}

    # ------------------------------------------------------------ 3. 라벨(예측 층)
    ukey = U[["no", "cid", "win_u", "win_kr"]].rename(columns={"win_u": "win_u_unit", "win_kr": "win_kr_unit"}).copy()
    ukey["cid"] = ukey["cid"].astype("Int64").fillna(-1).astype(int)
    ukey["u_found"] = True
    lab, linfo = read_label_units(ukey.drop_duplicates(["no", "cid"]))

    # ------------------------------------------------------------ 4. 창 특징
    feats = []
    close_of = dict(zip(wins["win_id"], wins["close_kst"]))
    for market, wcol, ccol, ucol in (("kr", "win_kr", "c_pre_kr", "win_kr_unit"), ("us", "win_u", "c_pre_u", "win_u_unit")):
        grp = P[P[wcol].notna()].groupby(wcol)
        for wid, g in grp:
            if wid not in close_of:
                continue
            n = len(g)
            v1, v2 = g["views_first"].to_numpy(float), g["views_last"].to_numpy(float)
            va = g["views24_short"].to_numpy(float)          # 나이 보정 조회(30시간까지 곡선). 소급 창에서는 원값 × 상수
            up1, dn1 = g["up_first"].to_numpy(float), g["down_first"].to_numpy(float)
            up2, dn2 = g["up_last"].to_numpy(float), g["down_last"].to_numpy(float)
            c = g[ccol].to_numpy(float)
            vpc = v1 / (c + 1)
            vpc_f = g["views24_full"].to_numpy(float) / (c + 1)
            vpc_s = va / (c + 1)
            ok = n >= MIN_POSTS
            k10 = max(1, math.ceil(0.1 * n))
            close = close_of[wid]
            ud1, ud2 = up1 + dn1, up2 + dn2
            rec = {"market": market, "win_id": wid, "n_posts": n,
                   "age_first_median_h": float(np.median(g["age_h_first"])),
                   "age_first_p10_h": float(np.quantile(g["age_h_first"], .1)),
                   "age_first_p90_h": float(np.quantile(g["age_h_first"], .9)),
                   "share_realtime_posts": float(g["realtime_post"].mean()),
                   "share_fetched_before_close": float((g["T_first"] < close).mean()),
                   "share_two_rows": float((g["n_rows"] == 2).mean()),
                   # (a) 같은 창 안 분포. 주판 = 나이 보정 조회(실시간 첫 줄의 나이 차이를 걷어 냄), 보조 = 첫 줄 원값·마지막 줄
                   "a_top10_view_share": top_share(va) if ok else np.nan,
                   "a_top10pct_view_share": top_share(va, k10) if ok else np.nan,
                   "a_view_gini": gini(va) if ok else np.nan,
                   "a_up_gini": gini(up1) if ok else np.nan,
                   "a_top10_view_share_raw": top_share(v1) if ok else np.nan,
                   "a_view_gini_raw": gini(v1) if ok else np.nan,
                   "a_top10_view_share_last": top_share(v2) if ok else np.nan,
                   "a_view_gini_last": gini(v2) if ok else np.nan,
                   "a_up_gini_last": gini(up2) if ok else np.nan,
                   # (b) 조회 ÷ (마감 전 댓글 + 1)
                   "b_vpc_median": float(np.median(vpc)), "b_vpc_p90": float(np.quantile(vpc, .9)),
                   "b_vpc_adj_short_median": float(np.median(vpc_s)), "b_vpc_adj_short_p90": float(np.quantile(vpc_s, .9)),
                   "b_vpc_adj_full_median": float(np.median(vpc_f)), "b_vpc_adj_full_p90": float(np.quantile(vpc_f, .9)),
                   "b_share_c_pre_zero": float((c == 0).mean()),
                   "c_pre_sum": float(c.sum()), "c_pre_median": float(np.median(c)),
                   # 추천·다른의견: 다른의견 ÷ (추천 + 다른의견)
                   "up_sum_first": float(up1.sum()), "down_sum_first": float(dn1.sum()),
                   "down_ratio_first": float(dn1.sum() / ud1.sum()) if ud1.sum() > 0 else np.nan,
                   "down_ratio_post_median": float(np.median(dn1[ud1 > 0] / ud1[ud1 > 0])) if (ud1 > 0).any() else np.nan,
                   "share_posts_up_or_down": float((ud1 > 0).mean()),
                   "down_ratio_last": float(dn2.sum() / ud2.sum()) if ud2.sum() > 0 else np.nan,
                   "views_sum_first": float(v1.sum()), "views_median_first": float(np.median(v1)),
                   "views_sum_last": float(v2.sum()),
                   "views24_short_median": float(np.median(va)), "views24_full_median": float(np.median(g["views24_full"]))}
            # 조회 상위 10% 글(나이 보정 조회 순위; 소급 창에서는 원값 순위와 같다)
            top_nos = set(g.nlargest(k10, "views24_short")["no"])
            rec["top10pct_k"] = k10
            rec["top10pct_c_pre_share"] = float(g[g["no"].isin(top_nos)][ccol].sum() / max(c.sum(), 1))
            lw_ = lab[(lab[ucol] == wid)]
            lp = lw_[(lw_["utype"] == "post")]
            lt = lp[lp["no"].isin(top_nos)]
            lc = lw_[(lw_["utype"] == "comment") & lw_["no"].isin(top_nos)]
            for tag, d in (("top10", lt), ("top10_withc", pd.concat([lt, lc])), ("posts_all", lp)):
                pairs = d.drop_duplicates(["a", "pred"])     # 작성자당 1회(창 안, 같은 pred)
                pp = pairs[pairs["pred"] > 0]
                rec[f"P_bar_{tag}"] = float(pp["pred"].mean() - 3) if len(pp) else np.nan
                rec[f"P_n_units_{tag}"] = int(len(d))
                rec[f"P_n_pred_{tag}"] = int(len(pp))
            feats.append(rec)
    F = pd.DataFrame(feats)
    F = F.merge(wins[["win_id", "D", "kind", "close_kst", "length_h", "missing_rate"]], on="win_id", how="left")
    F = F.merge(fa, on="win_id", how="left")
    # (c) 실시간 원값: 실시간 글 몫 95% 이상 창만
    rt = F["share_realtime_posts"] >= 0.95
    F["realtime_window"] = rt
    for c_src, c_dst in (("views_sum_first", "c_views_sum_first"), ("views_median_first", "c_views_median_first"),
                         ("views_sum_last", "c_views_sum_last"), ("up_sum_first", "c_up_sum_first"),
                         ("down_sum_first", "c_down_sum_first")):
        F[c_dst] = np.where(rt, F[c_src], np.nan)
    F = F.sort_values(["market", "D"]).reset_index(drop=True)

    # ------------------------------------------------------------ 5. 누출 진단
    L = {}
    # 5-1 창별 (b)와 수집 나이
    from scipy.stats import spearmanr  # noqa: E402
    for market in ("kr", "us"):
        f = F[(F["market"] == market)]
        fb = f[~f["realtime_window"] & (f["share_realtime_posts"] < 0.05)]
        out = {}
        for col in ("b_vpc_median", "b_vpc_adj_short_median", "b_vpc_adj_full_median", "b_vpc_p90", "b_share_c_pre_zero",
                    "views_median_first", "c_pre_median", "n_posts",
                    "a_top10_view_share", "a_top10pct_view_share", "a_view_gini", "a_up_gini",
                    "down_ratio_first", "share_posts_up_or_down"):
            rr = {}
            for lab_, d in (("all", f), ("backward_only", fb), ("backward_2026", fb[pd.to_datetime(fb["D"]) >= "2026-01-01"])):
                d = d[[col, "age_first_median_h"]].dropna()
                if len(d) >= 10:
                    s = spearmanr(np.log(d["age_first_median_h"]), d[col])
                    rr[lab_] = {"rho": round(float(s.statistic), 3), "n": int(len(d))}
            out[col] = rr
        L[f"corr_with_age_{market}"] = out
        # 실시간 경계 앞뒤
        bnd = {}
        rtw = f[f["realtime_window"]]
        prev = f[(~f["realtime_window"]) & (pd.to_datetime(f["D"]) >= pd.Timestamp("2026-09-21"))]
        for col in ("b_vpc_median", "b_vpc_adj_short_median", "b_vpc_adj_full_median", "views_median_first",
                    "views24_short_median", "c_pre_median", "n_posts", "a_top10_view_share", "a_top10_view_share_raw",
                    "a_top10_view_share_last", "a_top10pct_view_share", "a_view_gini", "a_view_gini_raw", "a_view_gini_last",
                    "a_up_gini", "down_ratio_first", "down_ratio_last"):
            bnd[col] = {"before_3w_median": round(float(prev[col].median()), 4), "n_before": int(prev[col].notna().sum()),
                        "realtime_median": round(float(rtw[col].median()), 4), "n_realtime": int(rtw[col].notna().sum())}
        L[f"boundary_{market}"] = bnd
    # 5-2 실시간 창: 첫 줄 vs 24시간 줄 순위 안정성
    stab = []
    for market, wcol in (("kr", "win_kr"), ("us", "win_u")):
        for wid in F.loc[(F["market"] == market) & F["realtime_window"], "win_id"]:
            g = P[(P[wcol] == wid) & (P["n_rows"] == 2)]
            if len(g) < 20:
                continue
            s = spearmanr(g["views_first"], g["views_last"]).statistic
            sa = spearmanr(g["views24_short"], g["views_last"]).statistic
            su = spearmanr(g["up_first"], g["up_last"]).statistic
            k = max(1, math.ceil(0.1 * len(g)))
            t1 = set(g.nlargest(k, "views_first")["no"])
            ta = set(g.nlargest(k, "views24_short")["no"])
            t2 = set(g.nlargest(k, "views_last")["no"])
            stab.append({"market": market, "win_id": wid, "n_two_row_posts": int(len(g)),
                         "age_first_median_h": round(float(g["age_h_first"].median()), 2),
                         "spearman_views_first_last": round(float(s), 3), "spearman_views_adj_last": round(float(sa), 3),
                         "spearman_up_first_last": round(float(su), 3),
                         "top10pct_overlap": round(len(t1 & t2) / k, 3), "top10pct_overlap_adj": round(len(ta & t2) / k, 3),
                         "top10_share_adj": round(top_share(g["views24_short"].to_numpy()), 4),
                         "top10_share_first": round(top_share(g["views_first"].to_numpy()), 4),
                         "top10_share_last": round(top_share(g["views_last"].to_numpy()), 4),
                         "gini_first": round(gini(g["views_first"].to_numpy()), 4),
                         "gini_last": round(gini(g["views_last"].to_numpy()), 4)})
    L["realtime_rank_stability"] = stab
    # 5-3 늦은 댓글(시각 기준 재조명): 등록 뒤 1일·7일 넘어 달린 댓글 몫, 등록 시기별
    P["ym"] = P["dt"].dt.strftime("%Y-%m")
    late = P.groupby("ym").agg(posts=("no", "size"), c_union=("c_union", "sum"), late1=("c_late1d", "sum"),
                               late7=("c_late7d", "sum"), age_med_d=("age_h_first", lambda s: float(s.median() / 24)))
    late["late1_share"] = (late["late1"] / late["c_union"]).round(4)
    late["late7_share"] = (late["late7"] / late["c_union"]).round(4)
    L["late_comments_by_month"] = late.reset_index().to_dict("records")
    # 5-4 같은 날(창) 안에서 조회 ÷ 마감 전 댓글 이 큰 글 = 늦게 쌓인 주목 후보: 그런 글의 늦은 댓글 몫
    P["vpc_kr"] = P["views_first"] / (P["c_pre_kr"] + 1)
    P["vpc_rank_in_win"] = P.groupby("win_kr")["vpc_kr"].rank(pct=True)
    hi_ = P[(P["vpc_rank_in_win"] >= 0.9) & (P["cls_first"] == "backward")]
    lo_ = P[(P["vpc_rank_in_win"] < 0.9) & (P["cls_first"] == "backward")]
    L["vpc_top10_vs_rest_backward"] = {
        "late1_share_top": round(float(hi_["c_late1d"].sum() / max(hi_["c_union"].sum(), 1)), 4),
        "late1_share_rest": round(float(lo_["c_late1d"].sum() / max(lo_["c_union"].sum(), 1)), 4),
        "late7_share_top": round(float(hi_["c_late7d"].sum() / max(hi_["c_union"].sum(), 1)), 4),
        "late7_share_rest": round(float(lo_["c_late7d"].sum() / max(lo_["c_union"].sum(), 1)), 4),
        "c_pre_zero_share_top": round(float((hi_["c_pre_kr"] == 0).mean()), 4),
        "c_pre_zero_share_rest": round(float((lo_["c_pre_kr"] == 0).mean()), 4)}

    # ------------------------------------------------------------ 6. 저장
    OUT.mkdir(parents=True, exist_ok=True)
    keepP = ["no", "dt", "n_rows", "cls_first", "realtime_post", "win_kr", "win_u", "views_first", "views_last",
             "up_first", "up_last", "down_first", "down_last", "ncomment_first", "ncomment_last", "nsaved_first",
             "nsaved_last", "T_first", "T_lo_first", "T_hi_first", "age_h_first", "method_first", "T_last", "age_h_last",
             "method_last", "c_pre_kr", "c_pre_u", "c_union", "c1", "c3", "c24", "c_late1d", "c_late7d", "c_at_T1",
             "c_at_T2", "lg_full_first", "lg_short_first", "views24_full", "views24_short"]
    P[keepP].to_parquet(OUT / "views_posts.parquet", index=False)
    F.to_parquet(OUT / "feat_views.parquet", index=False)
    meta = {"made": pd.Timestamp.now().isoformat(timespec="seconds"), "S": str(S),
            "labels": linfo, "growth": G, "leak": L,
            "posts": {"n": int(len(P)), "two_rows": int((P["n_rows"] == 2).sum()),
                      "realtime_posts": int(P["realtime_post"].sum()),
                      "cls_first": P["cls_first"].value_counts().to_dict()},
            "windows": {"kr": int((F["market"] == "kr").sum()), "us": int((F["market"] == "us").sum()),
                        "realtime_windows": F.loc[F["realtime_window"], "win_id"].tolist(),
                        "min_posts_for_a": MIN_POSTS},
            "label_windows_kr": {"windows_with_top10_label": int(((F["market"] == "kr") & (F["P_n_units_top10"] > 0)).sum()),
                                 "windows_with_top10_pred": int(((F["market"] == "kr") & (F["P_n_pred_top10"] > 0)).sum()),
                                 "top10_units_total": int(F.loc[F["market"] == "kr", "P_n_units_top10"].sum()),
                                 "top10_pred_total": int(F.loc[F["market"] == "kr", "P_n_pred_top10"].sum()),
                                 "top10_withc_pred_total": int(F.loc[F["market"] == "kr", "P_n_pred_top10_withc"].sum())},
            "label_windows_us": {"windows_with_top10_pred": int(((F["market"] == "us") & (F["P_n_pred_top10"] > 0)).sum()),
                                 "top10_pred_total": int(F.loc[F["market"] == "us", "P_n_pred_top10"].sum())},
            "code_sha256": {"events/code/feat_views.py": sha(Path(__file__)),
                            "events/code/views_fetch.py": sha(HERE / "views_fetch.py")},
            "outputs_sha256": {"feat_views.parquet": sha(OUT / "feat_views.parquet"),
                               "views_posts.parquet": sha(OUT / "views_posts.parquet")}}
    (OUT / "views_growth.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), "utf-8")
    print(json.dumps({k: meta[k] for k in ("labels", "posts", "windows", "label_windows_kr", "label_windows_us")},
                     ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
