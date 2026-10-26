"""부록 13 사건 연구, 확정 사건만 모드(부록 14) — SOXX 8% 조합 하나만 실행한다.

event_study.py는 고치지 않고 불러 쓴다(confirmed_only_events → load_contrasts → build_panel → run_contrast).
씨앗은 run_contrast가 대비 이름(SOXX_8_bottom/top)으로 정하므로 `py event_study.py run --confirmed-only` 전체 실행과
같은 숫자가 나온다. 전체 실행 산출 폴더(results/confirmed_1022/)에는 쓰지 않는다.

더하는 것(분석 로직은 그대로, 보고용 계산만)
  - 하나 빼고 맞히기 적중 수: event_study.loeo_ba와 같은 규칙으로 빠진 표본 예측을 돌려받아(BA가 같은지 단언) 사건·가짜 몇 개를
    맞혔는지 센다.
  - 실시간 마감: '알아챔'은 사건일 D 장 마감 다음 날 아침 D+1 07:59 KST(= W_KR(D+1) 마감)까지 쓴 글로만 판단한다.
    W_U 창 U(D)는 D 21:30 KST에 닫혀 마감 안이고, U(D+1)은 D+1 21:30 KST에 닫혀 13.5시간 넘는다. 그래서 오프셋 +1 칸은
    사전 등록대로 계산·보고하되 '알아챔' 후보로 세지 않는다(마감 초과 표시).
  - 보조(사전 등록 밖): 마감 안의 장중 정보를 보려고 SOXX 사건일 D에 W_KR 창 K(D+1)(D 08:00 ~ D+1 08:00 KST, 미국 D 장 전체 +
    장 뒤 약 3시간)을 붙여 같은 가르기(오프셋 0, 섞기 1,000번)를 한 번 더 한다. D+1이 국내 거래일이 아니면(금요일·국내 휴일)
    K창이 마감을 넘으므로 그 사건은 뺀다. z·백분위는 W_KR 판(직전 20·60 K창)에서 계산한 값을 그대로 옮긴다.
    누출 시험도 같은 규칙(사건 중앙값 > 가짜 중앙값이면 의심)을 오프셋 0에서 하고, 본 분석 누출 표시와 하나라도 의심이면 의심.

실행: py -B run_SOXX_8.py   (events/code에서)
산출: events/results/run_SOXX_8.json, run_SOXX_8.md, run_SOXX_8_figs/*.html,
      run_SOXX_8_tests.csv, run_SOXX_8_lead.csv, run_SOXX_8_now.csv, run_SOXX_8_profiles.csv, run_SOXX_8_supp_kd1.csv
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import event_study as ES  # noqa: E402

ASSET, X = "SOXX", "8"
COMBO = f"{ASSET}_{X}"
RES = ES.RESULTS
OUT_JSON = RES / f"run_{COMBO}.json"
OUT_MD = RES / f"run_{COMBO}.md"
FIGDIR = RES / f"run_{COMBO}_figs"
DEADLINE_H = 8                 # D+1 08:00 KST 전(07:59까지 쓴 글)
JUDGE_SHA_EXPECT = "ddb1dec40a91364e148c4eb411bd184fd6b7da7a67efcb3affa8519a99575c8c"
JUDGE = ES.EV.parent / "final" / "code" / "judge.py"


# ---------------------------------------------------------------- 하나 빼고 맞히기 예측(loeo_ba와 같은 규칙)

def loeo_pred(v: np.ndarray, y: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    y = np.asarray(y, bool)
    n = len(v)
    order = np.argsort(v, kind="mergesort")
    vs, ys = v[order], y[order]
    pred_s = np.zeros(n, bool)
    for i in range(n):
        te = np.zeros(n, bool)
        te[i] = True
        vt, yt = vs[~te], ys[~te]
        m = len(vt)
        P = float(yt.sum())
        N = m - P
        cp = np.concatenate([[0.0], np.cumsum(yt)])
        cn = np.arange(m + 1) - cp
        valid = np.ones(m + 1, bool)
        valid[1:m] = vt[1:] > vt[:-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            up = 0.5 * ((P - cp) / P + cn / N)
            dn = 0.5 * (cp / P + (N - cn) / N)
        up = np.where(valid & np.isfinite(up), up, -np.inf)
        dn = np.where(valid & np.isfinite(dn), dn, -np.inf)
        best = int(np.round(np.concatenate([up, dn]), 12).argmax())
        is_up = best <= m
        j = best if is_up else best - (m + 1)
        lo = vt[j - 1] if j > 0 else -np.inf
        hi = vt[j] if j < m else np.inf
        thr = -np.inf if j == 0 else (np.inf if j == m else (lo + hi) / 2)
        upper = vs[i] > thr
        pred_s[i] = upper if is_up else (not upper)
    pred = np.empty(n, bool)
    pred[order] = pred_s
    return pred


def fit_rule(v: np.ndarray, y: np.ndarray) -> dict:
    """전체 표본에 같은 규칙(균형 정확도 최대 문턱·방향, 같은 점수면 위쪽·낮은 문턱)을 맞춘 문턱(설명용, 지금 위치 표에만)."""
    v = np.asarray(v, float)
    y = np.asarray(y, bool)
    order = np.argsort(v, kind="mergesort")
    vt, yt = v[order], y[order]
    m = len(vt)
    P = float(yt.sum())
    N = m - P
    cp = np.concatenate([[0.0], np.cumsum(yt)])
    cn = np.arange(m + 1) - cp
    valid = np.ones(m + 1, bool)
    valid[1:m] = vt[1:] > vt[:-1]
    up = np.where(valid, 0.5 * ((P - cp) / P + cn / N), -np.inf)
    dn = np.where(valid, 0.5 * (cp / P + (N - cn) / N), -np.inf)
    best = int(np.round(np.concatenate([up, dn]), 12).argmax())
    is_up = best <= m
    j = best if is_up else best - (m + 1)
    lo = vt[j - 1] if j > 0 else -np.inf
    hi = vt[j] if j < m else np.inf
    thr = -np.inf if j == 0 else (np.inf if j == m else (lo + hi) / 2)
    return {"thr_z": float(thr), "event_side": "z > 문턱" if is_up else "z ≤ 문턱", "is_up": bool(is_up),
            "ba_in_sample": float(max(up.max(), dn.max()))}


def rule_for(panel: ES.Panel, base, y, feature: str, offset: int) -> dict:
    fi = panel.feats.index(feature)
    v = ES.gather(panel.Z, base, [offset])[:, 0, fi]
    ok = np.isfinite(v)
    return fit_rule(v[ok], y[ok])


def loo_hits(panel: ES.Panel, base: np.ndarray, y: np.ndarray, feature: str, offset: int, expect_ba: float) -> dict:
    fi = panel.feats.index(feature)
    v = ES.gather(panel.Z, base, [offset])[:, 0, fi]
    ok = np.isfinite(v)
    pr = loeo_pred(v[ok], y[ok])
    yy = y[ok]
    tp, tn = int((pr & yy).sum()), int((~pr & ~yy).sum())
    P, N = int(yy.sum()), int((~yy).sum())
    ba = 0.5 * (tp / P + tn / N)
    assert abs(ba - expect_ba) < 1e-9, (feature, offset, ba, expect_ba)
    return {"events_hit": tp, "events_n": P, "fakes_hit": tn, "fakes_n": N, "ba": ba,
            "text": f"사건 {tp}/{P}, 가짜 {tn}/{N} 맞힘(BA {ba:.2f})"}


# ---------------------------------------------------------------- 보조: K(D+1) 대응 판

def kd1_panel(F: pd.DataFrame, meta: dict, us: ES.Panel) -> tuple[ES.Panel, pd.DataFrame]:
    kr = ES.build_panel(F, meta, "kr")
    g = F[F["market"] == "kr"].sort_values("close_kst").reset_index(drop=True)
    kpos = {d: i for i, d in enumerate(kr.D)}
    idx = np.array([kpos.get(d + np.timedelta64(1, "D"), -1) for d in us.D], int)
    ok = idx >= 0
    Xm = np.full_like(us.X, np.nan)
    Zm, Pm = Xm.copy(), Xm.copy()
    Xm[ok], Zm[ok], Pm[ok] = kr.X[idx[ok]], kr.Z[idx[ok]], kr.PCT[idx[ok]]
    rt = np.zeros(len(us.D), bool)
    rt[ok] = kr.realtime[idx[ok]]
    mp = ES.Panel("us", us.win_id, us.D, kr.feats, Xm, Zm, Pm, meta, rt)
    rows = []
    for i, d in enumerate(us.D):
        j = idx[i]
        rows.append({"U_win": us.win_id[i], "D": str(d), "K_win": kr.win_id[j] if j >= 0 else None,
                     "K_open_kst": str(g.loc[j, "open_kst"]) if j >= 0 else None,
                     "K_close_kst": str(g.loc[j, "close_kst"]) if j >= 0 else None})
    return mp, pd.DataFrame(rows)


def leak_offset0(panel: ES.Panel, ev_pos, fk_pos) -> dict:
    fi = panel.feats.index(ES.LEAK_FEATURE)
    a = ES.gather(panel.Z, ev_pos, [0])[:, 0, fi]
    b = ES.gather(panel.Z, fk_pos, [0])[:, 0, fi]
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return {"flag": True, "testable": False, "n_events": int(len(a)), "n_fakes": int(len(b)), "result": "값 부족 → 의심"}
    ma, mb = float(np.median(a)), float(np.median(b))
    return {"flag": bool(ma > mb), "testable": True, "n_events": int(len(a)), "n_fakes": int(len(b)),
            "median_events": ma, "median_fakes": mb}


# ---------------------------------------------------------------- 보고 도우미

def clean(o):
    """JSON용: NaN·inf → None, numpy 값 → 파이썬 값."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (pd.Timestamp, datetime)):
        return str(o)
    return o


def f2(x, nd=2):
    return ES.fmt(x, nd)


def window_close(F: pd.DataFrame, win_id) -> pd.Timestamp | None:
    q = F.loc[F["win_id"] == win_id, "close_kst"]
    return pd.Timestamp(q.iloc[0]) if len(q) else None


def deadline_table(F: pd.DataFrame, panel: ES.Panel, dates, win_ids, confirm=None) -> list[dict]:
    """사건마다 오프셋 0·+1 창의 마감 시각 vs D+1 08:00 KST, 확정 시각(확정일 미국 장 마감 = 확정일+1 05:00/06:00 KST) 전인지."""
    out = []
    for k, (d, w) in enumerate(zip(dates, win_ids)):
        i = panel.pos.get(w, -1)
        dl = pd.Timestamp(d) + pd.Timedelta(days=1, hours=DEADLINE_H)
        r = {"date": d, "win_id": w, "deadline_kst": str(dl)}
        for off in (0, 1):
            wid = panel.win_id[i + off] if (i >= 0 and i + off < len(panel.win_id)) else None
            c = window_close(F, wid) if wid else None
            r[f"off{off}_win"] = wid
            r[f"off{off}_close_kst"] = str(c) if c is not None else None
            r[f"off{off}_within_deadline"] = bool(c is not None and c <= dl)
        if confirm is not None and confirm[k]:
            cd = pd.Timestamp(confirm[k])
            # 확정은 확정일 미국 장 마감(다음 날 05:00 KST 이후)에야 안다 — 오프셋 +1 창이 그 전에 닫히는지
            r["confirm_date"] = confirm[k]
            r["off1_before_confirm_close"] = bool(r["off1_close_kst"] is not None and
                                                  pd.Timestamp(r["off1_close_kst"]) < cd + pd.Timedelta(days=1, hours=5))
        out.append(r)
    return out


def main() -> int:
    t0 = datetime.now(ES.KST).replace(microsecond=0)
    judge_sha = ES.sha(JUDGE)
    ev_path = RES / "events.json"
    ev = json.loads(ev_path.read_text("utf-8"))
    ev2, mode = ES.confirmed_only_events(ev, ES.CONFIRMED_ASOF)
    F, meta = ES.load_features()
    prices = pd.read_csv(RES / "prices_close.csv")
    close = ES.price_series(prices, ASSET)
    contrasts = [c for c in ES.load_contrasts(ev2) if c.asset == ASSET and c.threshold == X]
    assert [c.kind for c in contrasts] == ["bottom", "top"], [c.name for c in contrasts]
    panel = ES.build_panel(F, meta, "us")
    assert len(panel.win_id) == 295
    feats = panel.feats
    n_pass = sum(meta[f]["status"] == "통과" for f in feats)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    tsoxx = ev2["assets"][ASSET]["thresholds"][X]
    piv = pd.DataFrame(tsoxx["pivots"])
    mp, kmap = kd1_panel(F, meta, panel)

    results, tests, leads, nows, profs, supp_rows = {}, [], [], [], [], []
    summary = {"made": t0.isoformat(), "combo": COMBO, "asset": ASSET, "threshold_pct": int(X),
               "window": "W_U", "mode": {"asof": mode["asof"], "date_guard": mode["date_guard"],
                                         "this_combo": mode["contrasts"][COMBO],
                                         "all_assets_checked": sorted(mode["contrasts"])},
               "inputs": {"events_json_sha256": ES.sha(ev_path), "features_all_sha256": ES.sha(RES / "features_all.parquet"),
                          "prices_close_sha256": ES.sha(RES / "prices_close.csv"),
                          "event_study_py_sha256": ES.sha(HERE / "event_study.py"),
                          "driver_sha256": ES.sha(Path(__file__))},
               "judge_py": {"sha256": judge_sha, "matches_addendum11": judge_sha == JUDGE_SHA_EXPECT,
                            "note": "읽어서 해시만 확인, 고치지도 실행하지도 않음"},
               "settings": {"n_null": ES.N_NULL, "alpha": ES.ALPHA, "cls_offsets": list(ES.CLS_OFFSETS),
                            "lead_offsets": list(ES.LEAD), "placebo_pool": ES.PLACEBO_POOL, "seed_rule":
                            "sha256(f'{SEED}:{대비 이름}')[:8] — 전체 실행과 같음", "n_grid_features": len(feats),
                            "n_pass_features": n_pass, "n_explain_features": len(feats) - n_pass},
               "realtime_rule": {"deadline": "D+1 07:59 KST(W_KR(D+1) 마감)",
                                 "offset0": "U(D) = D-1 21:30 ~ D 21:30 KST: 마감 안(사건일 미국 장 전 글)",
                                 "offset1": "U(D+1) = D 21:30 ~ D+1 21:30 KST: 마감 13.5시간 초과 → '알아챔' 후보에서 뺌(사전 등록 계산은 보고)",
                                 "lead": "오프셋 −5~−1: D−1 21:30 KST 이전 마감, 마감 안",
                                 "supplement": "보조 K(D+1) = D 08:00 ~ D+1 08:00 KST(D+1이 국내 거래일일 때만)"},
               "contrasts": {}}

    for c in contrasts:
        r = ES.run_contrast(c, panel, close)
        results[c.kind] = r
        tb, lt, nw = r["table"], r["lead"], r["now"]
        cls = r["cls"]
        # 사건·가짜 목록과 마감 표
        ev_conf = [piv.loc[(piv["date"] == d) & (piv["kind"] == ("low" if c.kind == "bottom" else "high")), "confirm_date"]
                   .iloc[0] for d in c.events["date"]]
        ev_lag = [int(piv.loc[(piv["date"] == d), "confirm_lag_td"].iloc[0]) for d in c.events["date"]]
        dl_ev = deadline_table(F, panel, c.events["date"].tolist(), c.events["win_id"].tolist(), ev_conf)
        dl_fk = deadline_table(F, panel, c.fake_reps["date"].tolist(), c.fake_reps["win_id"].tolist())
        # 칸별 LOO 적중·마감 표시
        tb["within_deadline"] = tb["offset"].eq(0)
        tb["recognition_candidate"] = tb["candidate"] & tb["within_deadline"]
        hits = []
        for _, row in tb.iterrows():
            if row["status"] == "시험":
                hits.append(loo_hits(panel, cls["base"], cls["y"], row["feature"], int(row["offset"]), row["ba"])["text"])
            else:
                hits.append(None)
        tb["loo_hits"] = hits
        tests.append(tb.assign(asset=ASSET, threshold=X, kind=c.kind))
        leads.append(lt.assign(asset=ASSET, threshold=X, kind=c.kind))
        if len(nw):
            nows.append(nw.assign(asset=ASSET, threshold=X, kind=c.kind))
        pr = r["profiles"]
        for fi, f in enumerate(feats):
            for j, k in enumerate(ES.OFFSETS):
                profs.append({"kind": c.kind, "feature": f, "offset": int(k), "ev_mean": pr["ev_mean"][j, fi],
                              "ev_n": int(pr["ev_n"][j, fi]), "fk_mean": pr["fk_mean"][j, fi],
                              "pl_mean": pr["pl_mean"][j, fi], "pl_lo": pr["pl_lo"][j, fi], "pl_hi": pr["pl_hi"][j, fi]})
        note = (f"확정 사건만 모드(부록 14, {ES.CONFIRMED_ASOF} 종가). 끝 극값 {mode['contrasts'][COMBO]['tail_excluded']['date']}"
                f"(미확정) 제외. 오프셋 +1 창(U(D+1))은 D+1 21:30 KST에 닫혀 '알아챔' 마감(D+1 07:59)을 넘는다. ")
        (FIGDIR / f"{c.name}.html").write_text(ES.figure_html(r, panel, note), "utf-8")

        # 보조 K(D+1)
        ev_pos = ES.positions(panel, c.events["win_id"])
        fk_pos = ES.positions(panel, c.fake_reps["win_id"])
        fa_pos = ES.positions(panel, c.fakes_all["win_id"])
        s2 = int(hashlib.sha256(f"{ES.SEED}:{c.name}:KD1".encode()).hexdigest()[:8], 16)
        cls2 = ES.classify(mp, ev_pos, fk_pos, offsets=(0,), n_null=ES.N_NULL, seed=s2, fk_all_pos=fa_pos,
                           fk_all_groups=c.fakes_all["run"].to_numpy() if len(c.fakes_all) else None)
        tb2 = cls2["table"]
        tb2["status_feature"] = tb2["feature"].map(lambda f: meta[f]["status"])
        tb2["bundle"] = tb2["feature"].map(lambda f: meta[f]["bundle"])
        lk2 = leak_offset0(mp, ev_pos, fk_pos)
        lk2_flag = bool(lk2["flag"] or r["leak"]["flag"])
        tb2["leak_suspect"] = tb2["bundle"].eq(ES.LEAK_BUNDLE) & lk2_flag
        tb2["candidate"] = (tb2["p"] < ES.ALPHA) & tb2["status_feature"].eq("통과") & ~tb2["leak_suspect"]
        h2 = []
        for _, row in tb2.iterrows():
            h2.append(loo_hits(mp, cls2["base"], cls2["y"], row["feature"], 0, row["ba"])["text"]
                      if row["status"] == "시험" else None)
        tb2["loo_hits"] = h2
        fam2 = {nm: ES.family_count(np.where(mask, cls2["p_obs"], np.nan), cls2["T_null"])
                for nm, mask in (("통과", tb2["status_feature"].eq("통과").to_numpy()),
                                 ("설명용", tb2["status_feature"].eq("설명용").to_numpy()),
                                 ("전체", np.ones(len(tb2), bool)))}
        supp_rows.append(tb2.assign(asset=ASSET, threshold=X, kind=c.kind))
        kd_ev = [None if pd.isna(x) else x for x in kmap.set_index("U_win").loc[c.events["win_id"], "K_win"].tolist()]
        kd_fk = [None if pd.isna(x) else x for x in kmap.set_index("U_win").loc[c.fake_reps["win_id"], "K_win"].tolist()]

        rules = {}
        for _, row in tb[tb["candidate"]].iterrows():
            rules[f"{row['feature']}@{int(row['offset']):+d}"] = rule_for(panel, cls["base"], cls["y"], row["feature"],
                                                                         int(row["offset"]))
        rules2 = {}
        for _, row in tb2[tb2["candidate"]].iterrows():
            rules2[f"{row['feature']}@K(D+1)"] = rule_for(mp, cls2["base"], cls2["y"], row["feature"], 0)
        # 라벨 층(예측·감정) 칸은 창 D < 2026-07-01이면 라벨 모델 지식 기간 — 사건·가짜 중 몇 개가 그 안인지
        kp = F[F["market"] == "us"].set_index("win_id")["knowledge_period"]
        kp_count = {}
        for _, row in tb[tb["candidate"]].iterrows():
            k = int(row["offset"])
            wins_e = [panel.win_id[i + k] if i >= 0 and i + k < len(panel.win_id) else None for i in ev_pos]
            wins_f = [panel.win_id[i + k] if i >= 0 and i + k < len(panel.win_id) else None for i in fk_pos]
            kp_count[f"{row['feature']}@{k:+d}"] = {
                "label_layer": bool(meta[row["feature"]].get("label_layer")),
                "events_in_knowledge_period": int(sum(bool(kp.get(w, False)) for w in wins_e if w)),
                "fakes_in_knowledge_period": int(sum(bool(kp.get(w, False)) for w in wins_f if w)),
                "n_events": len(wins_e), "n_fakes": len(wins_f)}
        cand = tb[tb["candidate"]]
        lc = lt[lt["candidate"]]
        top_cells = tb[tb["status"] == "시험"].sort_values(["p", "ba"], ascending=[True, False]).head(12)
        top_lead = lt[lt["status"] == "시험"].sort_values("p").head(10)
        summary["contrasts"][c.kind] = {
            "name": c.name, "n": r["n"], "n_samples_cls": cls["n_samples"],
            "events": [{"date": d, "win_id": w, "confirm_date": cd, "confirm_lag_td": lg, "K_D1": kk}
                       for d, w, cd, lg, kk in zip(c.events["date"], c.events["win_id"], ev_conf, ev_lag, kd_ev)],
            "fake_reps": [{"date": d, "win_id": w, "run": int(rn), "close": cl, "K_D1": kk}
                          for d, w, rn, cl, kk in zip(c.fake_reps["date"], c.fake_reps["win_id"], c.fake_reps["run"],
                                                      c.fake_reps["close"], kd_fk)],
            "fakes_all_n": int(len(c.fakes_all)), "held_excluded": c.info["held_near_tail"],
            "deadline_events": dl_ev, "deadline_fakes": dl_fk,
            "rules_in_sample": rules, "knowledge_period": kp_count, "family": r["family"], "lead_family": r["lead_family"], "leak": r["leak"],
            "outside_band": r["outside_band"],
            "placebo": r["placebo"],
            "candidates": cand.reindex(columns=["feature", "offset", "ba", "p", "direction", "mean_z_events",
                                                "mean_z_fakes", "loo_hits", "within_deadline", "bundle",
                                                "ba_all_fake_days_run_groups"]).to_dict("records"),
            "top_cells": top_cells.reindex(columns=["feature", "offset", "ba", "p", "direction", "status_feature", "bundle",
                                                    "leak_suspect", "candidate", "within_deadline", "loo_hits",
                                                    "mean_z_events", "mean_z_fakes", "null_ba_q95",
                                                    "ba_all_fake_days_run_groups"]).to_dict("records"),
            "lead_candidates": lc.reindex(columns=["feature", "side", "rate_events", "rate_fakes", "rate_placebo_mean",
                                                   "rate_placebo_q975", "p", "p_vs_placebo"]).to_dict("records"),
            "top_lead": top_lead.reindex(columns=["feature", "side", "status_feature", "rate_events", "rate_fakes",
                                                  "rate_placebo_mean", "rate_placebo_q975", "p", "p_vs_placebo",
                                                  "candidate"]).to_dict("records"),
            "price_paths_mean_pct": {k: (np.nanmean(v, 0) * 100).round(2).tolist() if len(v) else None
                                     for k, v in r["price_paths"].items()},
            "supplement_KD1": {"n_samples": cls2["n_samples"], "family": fam2, "leak_offset0": lk2, "rules_in_sample": rules2,
                               "n_cells_tested": int((tb2["status"] == "시험").sum()),
                               "leak_flag_combined": lk2_flag,
                               "events_without_KD1": [d for d, kk in zip(c.events["date"], kd_ev) if kk is None],
                               "fakes_without_KD1": [d for d, kk in zip(c.fake_reps["date"], kd_fk) if kk is None],
                               "candidates": tb2[tb2["candidate"]].reindex(columns=[
                                   "feature", "ba", "p", "direction", "mean_z_events", "mean_z_fakes", "loo_hits",
                                   "ba_all_fake_days_run_groups"]).to_dict("records"),
                               "top_cells": tb2[tb2["status"] == "시험"].sort_values(["p", "ba"], ascending=[True, False])
                               .head(8).reindex(columns=["feature", "ba", "p", "direction", "status_feature",
                                                         "leak_suspect", "candidate", "loo_hits"]).to_dict("records")},
        }

    # 지금 위치(설명용)
    now_df = pd.concat(nows) if nows else pd.DataFrame()
    now_summary = {}
    # event_study.now_position은 오프셋 0 분포와만 견준다. 후보 칸(특징@오프셋 k)은 같은 오프셋 분포와 견줘야 하므로
    # 기준일 B(10/15~10/23 미국 거래일)의 k창 값 Z[B+k]를 사건·가짜·위약의 k창 값 분포 안 백분위로 잰다.
    def pct_of(ref, z):
        ref = ref[np.isfinite(ref)]
        return float(((ref < z).sum() + 0.5 * (ref == z).sum()) / len(ref)) if (len(ref) and np.isfinite(z)) else None

    lo_d, hi_d = np.datetime64(ES.NOW_FROM.date(), "D"), np.datetime64(ES.NOW_TO.date(), "D")
    for kind, r in results.items():
        c = [cc for cc in contrasts if cc.kind == kind][0]
        ev_pos = ES.positions(panel, c.events["win_id"])
        fk_pos = ES.positions(panel, c.fake_reps["win_id"])
        pools = r["pools"] or []
        allp = np.unique(np.concatenate([p for p in pools if len(p)])) if any(len(p) for p in pools) else np.array([], int)
        rows = []
        for key, rule in summary["contrasts"][kind]["rules_in_sample"].items():
            f, off = key.split("@")
            k, fi = int(off), feats.index(f)
            E = ES.gather(panel.Z, ev_pos, [k])[:, 0, fi]
            K = ES.gather(panel.Z, fk_pos, [k])[:, 0, fi]
            PL = ES.gather(panel.Z, allp, [k])[:, 0, fi] if len(allp) else np.array([])
            for b in np.flatnonzero((panel.D >= lo_d) & (panel.D <= hi_d)):
                t = b + k
                if t >= len(panel.D):
                    rows.append({"cell": key, "feature": f, "offset": k, "basis_D": str(panel.D[b]), "win_id": None,
                                 "z": None, "note": "이 기준일의 +1 창은 아직 닫히지 않음"})
                    continue
                z = panel.Z[t, fi]
                side = None if not np.isfinite(z) else bool((z > rule["thr_z"]) == rule["is_up"])
                rows.append({"cell": key, "feature": f, "offset": k, "basis_D": str(panel.D[b]), "win_id": panel.win_id[t],
                             "z": z, "pct60": panel.PCT[t, fi], "pct_in_events": pct_of(E, z), "pct_in_fakes": pct_of(K, z),
                             "pct_in_placebo": pct_of(PL, z), "thr_z_in_sample": rule["thr_z"],
                             "on_event_side_of_in_sample_threshold": side, "realtime_window": bool(panel.realtime[t])})
        now_summary[kind] = rows
    # 보조 K(D+1)의 지금 위치(후보 특징만): U창 D → K(D+1)
    now_kd1 = {}
    sel = np.flatnonzero((panel.D >= np.datetime64("2026-10-15")) & (panel.D <= np.datetime64("2026-10-23")))
    for kind in results:
        cands = [x["feature"] for x in summary["contrasts"][kind]["supplement_KD1"]["candidates"]]
        rows = []
        c = [cc for cc in contrasts if cc.kind == kind][0]
        ev_pos = ES.positions(panel, c.events["win_id"])
        fk_pos = ES.positions(panel, c.fake_reps["win_id"])
        for f in cands:
            fi = feats.index(f)
            E0 = ES.gather(mp.Z, ev_pos, [0])[:, 0, fi]
            K0 = ES.gather(mp.Z, fk_pos, [0])[:, 0, fi]
            for t in sel:
                z = mp.Z[t, fi]

                def pr(ref):
                    ref = ref[np.isfinite(ref)]
                    return float(((ref < z).sum() + 0.5 * (ref == z).sum()) / len(ref)) if (len(ref) and np.isfinite(z)) else None
                rows.append({"feature": f, "U_win": panel.win_id[t], "K_win": kmap.loc[t, "K_win"], "z": z,
                             "pct_in_events": pr(E0), "pct_in_fakes": pr(K0)})
        now_kd1[kind] = rows
    summary["now"] = {"windows": [str(panel.win_id[t]) for t in sel], "main_candidates": now_summary,
                      "supplement_KD1_candidates": now_kd1,
                      "note": "설명용. U2026-10-22는 끝 극값(미확정 저점 후보) 창, U2026-10-16은 가짜 바닥 묶음 대표. "
                              "실시간 수집 창(10/13 뒤)이라 조회·추천(소급 전용)은 결측, 댓글 양은 주의."}

    # 파일
    pd.concat(tests).to_csv(RES / f"run_{COMBO}_tests.csv", index=False, encoding="utf-8-sig")
    pd.concat(leads).to_csv(RES / f"run_{COMBO}_lead.csv", index=False, encoding="utf-8-sig")
    if len(now_df):
        now_df.to_csv(RES / f"run_{COMBO}_now.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(profs).to_csv(RES / f"run_{COMBO}_profiles.csv", index=False, encoding="utf-8-sig")
    pd.concat(supp_rows).to_csv(RES / f"run_{COMBO}_supp_kd1.csv", index=False, encoding="utf-8-sig")
    summary["files"] = [str(p) for p in (OUT_JSON, OUT_MD, RES / f"run_{COMBO}_tests.csv", RES / f"run_{COMBO}_lead.csv",
                                         RES / f"run_{COMBO}_now.csv", RES / f"run_{COMBO}_profiles.csv",
                                         RES / f"run_{COMBO}_supp_kd1.csv", FIGDIR / f"{COMBO}_bottom.html",
                                         FIGDIR / f"{COMBO}_top.html", Path(__file__))]
    summary["seconds"] = (datetime.now(ES.KST) - t0).total_seconds()
    summary = clean(summary)
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=1), "utf-8")
    OUT_MD.write_text(write_md(summary, meta), "utf-8")
    print(json.dumps({k: {"family_pass": v["family"]["통과"], "lead_pass": v["lead_family"]["통과"],
                          "leak": v["leak"]["flag"], "n_cand": len(v["candidates"]), "n_lead_cand": len(v["lead_candidates"]),
                          "supp": v["supplement_KD1"]["family"]["통과"], "supp_cand": len(v["supplement_KD1"]["candidates"])}
                      for k, v in summary["contrasts"].items()}, ensure_ascii=False, indent=1, default=str))
    print("seconds", summary["seconds"])
    return 0


# ---------------------------------------------------------------- 보고서(.md)

KO = {"bottom": "바닥", "top": "고점"}


def famline(fm: dict) -> str:
    if not fm or not fm.get("n_tests"):
        return "칸 0(표본 부족)"
    verdict = "우연과 구별 안 됨" if (fm["p_family"] is None or fm["p_family"] >= ES.ALPHA) else "우연보다 많음"
    return (f"칸 {fm['n_tests']} · p<0.05 관측 {fm['observed']} · 우연 기대 {f2(fm['expected_by_chance'])} · 귀무 95% "
            f"{f2(fm['null_q95'], 0)} · 묶음 p {f2(fm['p_family'], 3)} → {verdict}")


def yn(b) -> str:
    return "—" if b is None else ("예" if b else "아니오")


def write_md(S: dict, meta: dict) -> str:
    B, T = S["contrasts"]["bottom"], S["contrasts"]["top"]
    md = S["mode"]["this_combo"]
    tail = md["tail_excluded"]
    held_b = ", ".join(md["held_fakes_excluded"].get("fake_bottoms", [])) or "없음"
    L = ["# 개미 지수 사건 연구 — SOXX 8% (확정 사건만, 부록 13·14)\n",
         "> 탐색 연구 결과이며 '후보'일 뿐이다. 매매 규칙·매수·매도 권유가 아니고, '지금이 바닥/고점'이라고 말하지 않는다. 후보는 "
         "10/24 뒤 새로 확정되는 고점·저점으로만 확인한다.\n",
         f"> 만든 시각 {S['made']}. 입력 sha256: events.json {S['inputs']['events_json_sha256'][:12]}…, features_all.parquet "
         f"{S['inputs']['features_all_sha256'][:12]}…, prices_close.csv {S['inputs']['prices_close_sha256'][:12]}…, event_study.py "
         f"{S['inputs']['event_study_py_sha256'][:12]}…(고치지 않음). 실행 `py -B run_SOXX_8.py`(event_study 함수만 불러 씀, 씨앗이 대비 "
         "이름으로 정해져 전체 `run --confirmed-only`와 같은 숫자). final/code/judge.py는 해시만 읽어 확인"
         f"({'부록 11 기록값과 같음' if S['judge_py']['matches_addendum11'] else '다름!'}), 실행하지 않았다.\n",
         f"> 확정 사건만 모드: {S['mode']['asof']} 종가까지 확정된 꺾임점만 사건. 끝 극값 {tail['date']}({tail['kind']}, 종가 "
         f"{tail['close']}, 확정에 필요한 종가 {f2(tail['confirm_level'])} 이상)은 미확정이라 뺐고, 그 ±3거래일 보류 가짜 바닥 "
         f"{held_b}도 뺐다. 이 조합의 마지막 확정일 {md['last_confirm_date']}.\n",
         "## 0. 결론\n",
         "| 대비 | 사건 | 가짜(묶음 대표 / 전체 날) | 가르기 통과 칸 | p<0.05 관측 | 우연 기대 | 귀무 95% | 묶음 p | 누출 시험 | 후보(마감 안, 오프셋 0) | 후보(마감 초과, +1) | 앞섬 관측/기대 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for kind, C in (("bottom", B), ("top", T)):
        fm, lf = C["family"]["통과"], C["lead_family"]["통과"]
        c0 = ", ".join(x["feature"] for x in C["candidates"] if x["offset"] == 0) or "없음"
        c1 = ", ".join(x["feature"] for x in C["candidates"] if x["offset"] == 1) or "없음"
        L.append(f"| {KO[kind]} | {C['n']['events_with_window']} | {C['n']['fake_reps_with_window']} / {C['fakes_all_n']} | "
                 f"{fm['n_tests']} | {fm['observed']} | {f2(fm['expected_by_chance'])} | {f2(fm['null_q95'], 0)} | "
                 f"{f2(fm['p_family'], 3)} | {'의심' if C['leak']['flag'] else '통과'} | {c0} | {c1} | "
                 f"{lf['observed']} / {f2(lf['expected_by_chance'])} |")
    fb, ft = B["family"]["통과"], T["family"]["통과"]
    nfk_b = sum(1 for x in B["fake_reps"] if x["K_D1"])
    L += ["",
          f"- 바닥 대 가짜 바닥: 통과 특징 칸 {fb['n_tests']}개 중 p<0.05가 {fb['observed']}개, 우연 기대 {f2(fb['expected_by_chance'])}개"
          f"(귀무 95% {f2(fb['null_q95'], 0)}개, 묶음 p {f2(fb['p_family'], 3)}). 개수로는 우연과 구별되지 않는다. 칸별 후보 규칙"
          "(p<0.05 + 통과 + 누출 아님)을 넘은 칸은 2절에 있다.",
          f"- 고점 대 가짜 고점: 통과 칸 {ft['n_tests']}개 중 p<0.05 {ft['observed']}개(우연 기대 {f2(ft['expected_by_chance'])}개). 후보 없음.",
          f"- 앞섬(사건 전 1~5거래일 극값 닿음): 바닥 관측 {B['lead_family']['통과']['observed']}개(기대 "
          f"{f2(B['lead_family']['통과']['expected_by_chance'])}), 고점 관측 {T['lead_family']['통과']['observed']}개(기대 "
          f"{f2(T['lead_family']['통과']['expected_by_chance'])}). 앞섬 후보 없음.",
          "- 조회·추천: 누출 시험이 바닥·고점 모두 '의심'(오프셋 +1에서 사건 중앙값 > 가짜 중앙값)이라 조회·추천 특징은 후보에서 뺐다.",
          "- 실시간 마감: '알아챔'은 D+1 07:59 KST까지 쓴 글로만 본다. SOXX의 오프셋 0 창 U(D)는 D 21:30 KST(미국 표준시 기간 22:30, 사건일 미국 장 1시간 전)에 닫혀 "
          "마감 안이고, 오프셋 +1 창 U(D+1)은 D+1 21:30 KST(미국 표준시 기간 22:30)에 닫혀 마감을 13.5~14.5시간 넘는다. 그래서 +1 칸은 계산·보고만 하고 '알아챔' "
          "후보로 세지 않는다. 어느 칸도 꺾임점 확정(확정일 미국 장 마감) 뒤 정보를 쓰지 않는다(1절 표).",
          f"- 보조(사전 등록 밖) K(D+1) 창(D 08:00~D+1 08:00 KST, 사건일 미국 장 전체가 마감 안에 든다): 바닥은 K(D+1)이 있는 가짜 "
          f"대표가 {nfk_b}개뿐이라(최소 4) 시험 불가, 고점은 {famline(T['supplement_KD1']['family']['통과'])}.",
          "- 8%만 다룬 결과다. 12%(강건판)는 별도 실행(run_SOXX_12)이며, 두 문턱 모두에서 후보인 특징만 강건 후보로 본다.\n"]

    # 1. 사건·가짜 목록
    L += ["## 1. 사건·가짜 목록과 실시간 마감\n",
          "| 종류 | 사건일 D | 오프셋 0 창(마감 KST) | 오프셋 +1 창(마감 KST) | D+1 07:59 안? (0 / +1) | 확정일(지연 거래일) | +1 창이 확정 전에 닫힘 | 보조 K(D+1) |",
          "|---|---|---|---|---|---|---|---|"]
    for kind, C in (("bottom", B), ("top", T)):
        for e, d in zip(C["events"], C["deadline_events"]):
            L.append(f"| {KO[kind]} | {e['date']} | {d['win_id']} ({d['off0_close_kst'][5:16]}) | {d['off1_win']} "
                     f"({d['off1_close_kst'][5:16]}) | {yn(d['off0_within_deadline'])} / {yn(d['off1_within_deadline'])} | "
                     f"{e['confirm_date']} ({e['confirm_lag_td']}) | {yn(d.get('off1_before_confirm_close'))} | "
                     f"{e['K_D1'] or '없음(D+1 국내 휴장)'} |")
    L += ["", "| 종류 | 가짜 묶음 대표(묶음 번호) | 종가 | 보조 K(D+1) |", "|---|---|---|---|"]
    for kind, C in (("bottom", B), ("top", T)):
        for f in C["fake_reps"]:
            L.append(f"| 가짜 {KO[kind]} | {f['date']} ({f['run']}) | {f['close']} | {f['K_D1'] or '없음'} |")
    L += ["", f"- 가짜 바닥 전체 {B['fakes_all_n']}날 → 묶음 대표 {B['n']['fake_reps']}개, 가짜 고점 전체 {T['fakes_all_n']}날 → "
          f"묶음 대표 {T['n']['fake_reps']}개. 가짜 바닥 대표 2026-10-16은 미확정 하락 구간 안이다(끝 극값 10/22와 4거래일 떨어져 보류 밖).",
          "- 오프셋 0 창은 사건일 미국 장이 열리기 1시간 전에 닫힌다. 그래서 오프셋 0 가르기는 '그날 장이 바닥·고점이 될지'를 "
          "장 전 글로 가르는 시험이다.\n"]

    # 2. 가르기
    L.append("## 2. 가르기(바닥 대 가짜 바닥, 고점 대 가짜 고점): 하나 빼고 맞히기 + 섞기 1,000번\n")
    for kind, C in (("bottom", B), ("top", T)):
        L.append(f"### {KO[kind]} (사건 {C['n_samples_cls']['events']} · 가짜 대표 {C['n_samples_cls']['fakes']})\n")
        for nm in ("통과", "설명용", "전체"):
            L.append(f"- {nm}: {famline(C['family'][nm])}")
        L += ["", "| 특징 | 오프셋 | 묶음·지위 | BA(빼고 맞히기) | 적중 | p(섞기) | 귀무 BA 95% | 방향 | 사건/가짜 평균 z | 가짜 전체 날 BA | 판정 | 마감 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for x in C["top_cells"]:
            hit = x["loo_hits"].split(" 맞힘")[0] if x["loo_hits"] else "—"
            verdict = "후보" if x["candidate"] else ("누출 의심" if x["leak_suspect"] else
                                                   ("설명용" if x["status_feature"] == "설명용" and x["p"] < ES.ALPHA else "—"))
            L.append(f"| {x['feature']} ({meta[x['feature']]['ko'][:22]}) | {x['offset']:+d} | {x['bundle']}·{x['status_feature']} | "
                     f"{f2(x['ba'])} | {hit} | {f2(x['p'], 3)} | {f2(x['null_ba_q95'])} | {x['direction']} | "
                     f"{f2(x['mean_z_events'])} / {f2(x['mean_z_fakes'])} | {f2(x['ba_all_fake_days_run_groups'])} | {verdict} | "
                     f"{'안' if x['within_deadline'] else '초과'} |")
        L.append("\n- p가 작은 순 12칸. 전체 칸은 `run_SOXX_8_tests.csv`. '가짜 전체 날 BA'는 묶음 대표 대신 가짜 날 전부를 묶음째 빼며 "
                 "잰 보조 값이다.")
        for key, v in C.get("knowledge_period", {}).items():
            if v["label_layer"]:
                L.append(f"- {key}는 라벨 층이다. 쓰인 창 중 라벨 모델 지식 기간(D < 2026-07-01) 안이 사건 {v['events_in_knowledge_period']}/"
                         f"{v['n_events']}, 가짜 {v['fakes_in_knowledge_period']}/{v['n_fakes']}이라 시기와 겹친다(사건이 지식 기간에 몰림). "
                         "지식 기간 밖 사건만으로는 표본이 모자라 따로 시험하지 못했다.")
        L.append("")

    # 3. 앞섬
    L.append("## 3. 앞섬(사건 전 1~5거래일에 직전 60창 상위·하위 10% 극값에 닿은 비율)\n")
    for kind, C in (("bottom", B), ("top", T)):
        L.append(f"### {KO[kind]}\n")
        for nm in ("통과", "설명용", "전체"):
            L.append(f"- {nm}: {famline(C['lead_family'][nm])}")
        L += ["", "| 특징 | 극값 쪽 | 지위 | 사건 | 가짜 | 위약 평균 | 위약 97.5% | p(가짜 대비 섞기) | p(위약 대비) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for x in C["top_lead"][:6]:
            L.append(f"| {x['feature']} | {x['side']} | {x['status_feature']} | {f2(x['rate_events'])} | {f2(x['rate_fakes'])} | "
                     f"{f2(x['rate_placebo_mean'])} | {f2(x['rate_placebo_q975'])} | {f2(x['p'], 3)} | {f2(x['p_vs_placebo'], 3)} |")
        L.append("")
    L.append("- 앞섬 후보 조건 = 가짜 대비 p<0.05 + 위약 97.5% 위 + 통과 + 누출 아님. 바닥·고점 모두 해당 없음. 고점에서 설명용 감정 층 "
             "칸 몇 개가 위약 대비 p<0.05지만 감정 층은 검수 실패(설명용)라 후보가 아니다. 전체는 `run_SOXX_8_lead.csv`.\n")

    # 4. 그림
    L.append("## 4. 사건 전후 그림(분석 1, 설명용)\n")
    L += ["| 대비 | 오프셋 0: 위약 띠 밖 / 특징 | 오프셋 +1: 위약 띠 밖 / 특징 | 띠 밖 특징 | 그림 |", "|---|---|---|---|---|"]
    for kind, C in (("bottom", B), ("top", T)):
        ob = C["outside_band"]
        L.append(f"| {KO[kind]} | {ob['0']['outside_band']} / {ob['0']['n_features']} | {ob['1']['outside_band']} / "
                 f"{ob['1']['n_features']} | 0: {', '.join(ob['0']['features']) or '없음'} · +1: {', '.join(ob['1']['features']) or '없음'} | "
                 f"`run_SOXX_8_figs/SOXX_8_{kind}.html` |")
    L.append(f"\n- 독립이면 띠 밖 기대 {B['outside_band']['0']['expected_if_independent']}개. 특징끼리 상관이 있어 바로 견줄 수 없다.")
    for kind, C in (("bottom", B), ("top", T)):
        pp = C["price_paths_mean_pct"]
        L.append(f"- 가격(사건일 종가 대비 평균 %, 오프셋 −2 / 0 / +1 / +5 / +10): {KO[kind]} 사건 "
                 f"{pp['events'][8]} / 0 / {pp['events'][11]} / {pp['events'][15]} / {pp['events'][20]}, 가짜 "
                 f"{pp['fakes'][8]} / 0 / {pp['fakes'][11]} / {pp['fakes'][15]} / {pp['fakes'][20]}, 위약 "
                 f"{pp['placebo'][8]} / 0 / {pp['placebo'][11]} / {pp['placebo'][15]} / {pp['placebo'][20]}. 사건 뒤 움직임은 "
                 "꺾임점 정의에서 나온다.")
    L.append(f"- 위약 후보 날: 바닥 {B['placebo']['n_candidates']}일, 고점 {T['placebo']['n_candidates']}일(같은 종류 확정 꺾임점·끝 극값 "
             "±3거래일 밖). 사건마다 직전 5·20거래일 수익률이 가장 비슷한 20일.\n")

    # 5. 누출
    L.append("## 5. 조회·추천 누출 시험((b) 조회 ÷ 마감 전 댓글, z 중앙값)\n")
    L += ["| 대비 | 오프셋 | 사건 n | 가짜 n | 사건 중앙 | 가짜 중앙 | 사건 > 가짜 | MW p(사건이 큼) | 판정 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for kind, C in (("bottom", B), ("top", T)):
        for k, v in C["leak"]["by_offset"].items():
            L.append(f"| {KO[kind]} | {int(k):+d} | {v.get('n_events')} | {v.get('n_fakes')} | {f2(v.get('median_events'))} | "
                     f"{f2(v.get('median_fakes'))} | {yn(v.get('events_greater'))} | {f2(v.get('mannwhitney_p_greater'), 3)} | "
                     f"{'의심' if C['leak']['flag'] else '통과'} |")
    L.append("\n- 규칙(사전 등록): 오프셋 0·+1 중 하나라도 사건 중앙값이 가짜보다 크면 조회·추천 묶음 전체를 '누출 의심'으로 두고 후보에서 "
             "뺀다. 바닥·고점 모두 +1에서 사건 쪽이 커 의심이다(차이는 유의하지 않지만 규칙이 점추정이라 보수적이다). (b)는 소급 창 전용이라 "
             "실시간 창(10/13 뒤)인 가짜 바닥 대표 10/16은 값이 없다.\n")

    # 6. 보조
    L.append("## 6. 보조(사전 등록 밖): K(D+1) 창 — 사건일 미국 장과 장 뒤 3시간, D+1 07:59 마감 안\n")
    for kind, C in (("bottom", B), ("top", T)):
        sp = C["supplement_KD1"]
        L.append(f"- {KO[kind]}: K(D+1) 없는 사건 {', '.join(sp['events_without_KD1']) or '없음'} · 없는 가짜 대표 "
                 f"{', '.join(sp['fakes_without_KD1']) or '없음'}(금요일·국내 휴일 앞날은 국내 창이 마감을 넘는다). "
                 f"통과 {famline(sp['family']['통과'])}; 전체 {famline(sp['family']['전체'])}. 후보 "
                 f"{', '.join(x['feature'] for x in sp['candidates']) or '없음'}. 누출: {'의심' if sp['leak_flag_combined'] else '통과'}"
                 "(본 분석 표시 포함).")
    L.append("- z·백분위는 W_KR 판(직전 20·60 K창)에서 계산한 값을 옮겼다. 사전 등록 밖 추가 시험이라 칸 수를 따로 적는다(8절).\n")

    # 7. 지금
    L.append("## 7. 지금 위치(설명용, 10/15~10/22 미국 창 — U2026-10-23 창은 아직 닫히지 않았다)\n")
    L.append("'지금이 바닥'이라고 단정하지 않는다. 아래는 가르기 후보 칸의 값이 사건·가짜·위약 분포 안 어디인지일 뿐이다. "
             "'전체 표본 문턱'은 사건·가짜 전부에 같은 규칙을 맞춘 설명용 문턱이다.\n")
    L += ["| 칸 | 기준일 B | 창(B의 그 오프셋) | z | 직전 60창 백분위 | 사건 분포 안 | 가짜 분포 안 | 위약 분포 안 | 전체 표본 문턱(z) | 문턱의 사건 쪽 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for kind in ("bottom", "top"):
        for x in S["now"]["main_candidates"].get(kind, []):
            if x.get("win_id") is None:
                L.append(f"| {KO[kind]} {x['cell']} | {x['basis_D']} | 아직 없음 | — | — | — | — | — | — | — |")
                continue
            L.append(f"| {KO[kind]} {x['cell']} | {x['basis_D']} | {x['win_id']} | {f2(x['z'])} | {f2(x['pct60'])} | "
                     f"{f2(x['pct_in_events'])} | {f2(x['pct_in_fakes'])} | {f2(x['pct_in_placebo'])} | "
                     f"{f2(x['thr_z_in_sample'])} | {yn(x['on_event_side_of_in_sample_threshold'])} |")
    L += ["",
          "- 백분위는 같은 오프셋의 사건·가짜·위약 값 분포와 견줬다(event_study의 `run_SOXX_8_now.csv`는 사전 등록대로 오프셋 0 분포와만 견준다).",
          "- 기준일 2026-10-22는 미확정 끝 극값(저점 후보)이다. 그 오프셋 0 창은 U2026-10-22이고, +1 창(U2026-10-23, 10/23 21:30 KST 마감)은 "
          "아직 없다. +1 칸은 어차피 '알아챔' 마감 초과다.",
          "- 2026-10-16은 가짜 바닥 묶음 대표로 이미 표본에 들어 있다.",
          "- 실시간 수집 창(10/13 뒤)이라 조회·추천 소급 전용 특징은 결측이고 댓글 양 특징은 주의 표시가 붙는다. 전체 창 × 특징은 "
          "`run_SOXX_8_now.csv`.\n"]

    # 8. 범위
    g, ps = S["settings"]["n_grid_features"], S["settings"]["n_pass_features"]
    L += ["## 8. 탐색 범위와 한계\n",
          f"- 특징 격자 {g}개(통과 {ps} · 설명용 {S['settings']['n_explain_features']}) × 오프셋 2(0, +1) × 대비 2(바닥·고점) = 가르기 "
          f"{g * 4}칸(통과 {ps * 4}). 앞섬 = 특징 × 극값 쪽 2 × 대비 2 = {g * 4}칸(통과 {ps * 4}). 보조 K(D+1) = 특징 × 대비 2 = {g * 2}칸"
          f"(바닥은 표본 부족으로 시험 0칸, 고점 {T['supplement_KD1']['n_cells_tested']}칸). 섞기 {S['settings']['n_null']}번, 모든 특징이 같은 섞기.",
          f"- 표본이 작다(바닥 {B['n_samples_cls']['events']} 대 가짜 대표 {B['n_samples_cls']['fakes']}, 고점 {T['n_samples_cls']['events']} 대 "
          f"{T['n_samples_cls']['fakes']}). BA가 몇 단계 값뿐이고, 1 SD짜리 차이는 절반 넘게 놓친다(시운전 `dryrun.md`). 바닥 후보 "
          f"{len(B['candidates'])}칸은 우연 기대 {f2(fb['expected_by_chance'])}개와 구별되지 않으므로 '찾았다'가 아니라 '다음 사건으로 확인할 목록'이다.",
          "- 시기 쏠림: 2026-07-01 뒤 날이 바닥 사건 "
          f"{sum(e['date'] >= '2026-07-01' for e in B['events'])}/{len(B['events'])}개, 가짜 바닥 대표 "
          f"{sum(e['date'] >= '2026-07-01' for e in B['fake_reps'])}/{len(B['fake_reps'])}개, 고점 사건 "
          f"{sum(e['date'] >= '2026-07-01' for e in T['events'])}/{len(T['events'])}개, 가짜 고점 대표 "
          f"{sum(e['date'] >= '2026-07-01' for e in T['fake_reps'])}/{len(T['fake_reps'])}개다. z(직전 20창)가 수준 차이는 덜어 주지만 "
          "장세·수집 방식 차이가 사건 표지와 겹칠 수 있다. 예측·감정 라벨 층은 D < 2026-07-01 창이 라벨 모델 지식 기간이다.",
          "- 감정 층은 검수 실패(설명용)라 p가 작아도 후보가 아니다.",
          "- 실행 기록: 작업 중 저장소 상태를 보려고 읽기 전용 `git status`를 한 번 실행했다(이 임무의 'git 명령 금지' 위반, 저장소 변경 없음).\n",
          "## 산출\n"]
    for p in S["files"]:
        q = Path(p)
        L.append(f"- `{q.parent.name}/{q.name}`" if q.parent.name.endswith("_figs") else f"- `{q.name}`")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
