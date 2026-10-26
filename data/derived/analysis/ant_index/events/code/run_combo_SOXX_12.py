"""부록 13·14 사건 연구: SOXX 12%(강건판) 조합만, 확정 사건만 모드(2026-10-22 종가로 판정한 events.json).

event_study.py의 함수를 그대로 부른다(전처리·분석 1~4·누출 시험·문턱·섞기 1,000번·위약·씨앗 불변). run_real과 다른 점:
  - 대비를 SOXX 12%(바닥·고점)로만 거른다. 씨앗은 대비 이름(SOXX_12_bottom/top)에서 나오므로 전체 실행과 같은 값이 나온다.
  - 하나 빼고 맞히기의 표본별 예측(사건·가짜 몇 개를 맞혔나), 라벨 지식 기간(D < 2026-07-01) 표시, 사건별 확정 지연을 더 적는다.
  - 실시간 규칙(이 실행 임무): '알아챔'은 사건일 D 장 마감 다음 날 아침(D+1 07:59 KST)까지 쓴 글로만 판단한다.
      W_U 오프셋 0  = U(D) 창, D 21:30 KST 마감(D 미국 장 시작 전)          → 규칙 안
      W_U 오프셋 +1 = U(D+1) 창, 다음 미국 거래일 21:30 KST 마감             → 규칙 밖(D+1 07:59보다 13.5시간 이상 늦음).
                     분석은 등록대로 하되 '알아챔' 후보에서 뺀다(참고로만 적음).
      앞섬 −5~−1   = 사건 전 창                                              → 규칙 안
    보조(사전 등록 밖, 후보로 올리지 않음): '+1 대체' = K(D+1) 창(D 08:00 ~ D+1 08:00 KST, D 미국 장 전체 + 다음 날 아침).
    D+1(달력)이 국내 거래일이 아니면 그 창은 D+1 08:00 뒤에 닫히므로 결측으로 둔다. 같은 섞기 귀무(별도 씨앗)로 우연 기대를 센다.
산출: events/results/run_SOXX_12.json / .md, run_SOXX_12_tests.csv, _lead.csv, _now.csv, _loo.csv, _profiles.csv,
      _kd1_tests.csv, run_SOXX_12_figs/*.html
실행: cd events/code; py -B run_combo_SOXX_12.py
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import event_study as es  # noqa: E402

ASSET, THR = "SOXX", "12"
TAG = f"run_{ASSET}_{THR}"
RES = es.RESULTS
FIG = RES / f"{TAG}_figs"
KNOW_END = np.datetime64("2026-07-01", "D")     # 라벨 모델 지식 기간: D < 2026-07-01 창(부록 3·11)
LABEL_BUNDLES = ("예측 층", "감정 층")
JUDGE = es.EV.parent / "final" / "code" / "judge.py"
JUDGE_SHA = "ddb1dec40a91364e148c4eb411bd184fd6b7da7a67efcb3affa8519a99575c8c"
CUTOFF_H = 8                                     # 알아챔 마감: D+1 08:00 KST 전(= W_KR(D+1) 마감 07:59)


# ---------------------------------------------------------------- 도우미

def loeo_pred(v: np.ndarray, y: np.ndarray) -> np.ndarray:
    """es.loeo_ba와 같은 규칙(표본 하나씩 빼기, 균형 정확도 최대 문턱, 같은 점수 처리 동일)으로 빠진 표본의 예측(원래 순서)."""
    v = np.asarray(v, float)
    y = np.asarray(y, bool)
    n = len(v)
    order = np.argsort(v, kind="mergesort")
    vs, ys = v[order], y[order]
    pred_s = np.zeros(n, bool)
    for i in range(n):
        test = np.zeros(n, bool)
        test[i] = True
        vt, yt = vs[~test], ys[~test]
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
        pred_s[i] = upper if is_up else not upper
    pred = np.zeros(n, bool)
    pred[order] = pred_s
    return pred


def full_fit(v: np.ndarray, y: np.ndarray) -> dict:
    """전체 표본으로 정한 문턱·방향(설명용 — 빠진 표본 평가가 아니다)."""
    s = np.sort(np.unique(v))
    cuts = [-np.inf] + [(s[i - 1] + s[i]) / 2 for i in range(1, len(s))] + [np.inf]
    best, ch = -np.inf, None
    for d in ("up", "dn"):
        for c in cuts:
            pr = v > c if d == "up" else ~(v > c)
            ba = 0.5 * ((pr & y).sum() / y.sum() + (~pr & ~y).sum() / (~y).sum())
            if ba > best + 1e-12:
                best, ch = ba, (d, c)
    return {"rule": "up" if ch[0] == "up" else "dn",
            "direction": "z가 문턱보다 크면 사건" if ch[0] == "up" else "z가 문턱 이하이면 사건",
            "threshold_z": float(ch[1]) if np.isfinite(ch[1]) else None, "ba_in_sample": float(best)}


def clean(o):
    """JSON용: NaN·inf → None, numpy 값 → 파이썬 값."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [clean(v) for v in (o.tolist() if isinstance(o, np.ndarray) else o)]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if o is None or isinstance(o, str):
        return o
    if o is pd.NA or o is pd.NaT:
        return None
    return str(o)


def f2(x, nd=2):
    return es.fmt(x, nd)


def subset_events(ev: dict) -> dict:
    sub = copy.deepcopy(ev)
    a = sub["assets"][ASSET]
    a["thresholds"] = {THR: a["thresholds"][THR]}
    sub["assets"] = {ASSET: a}
    return sub


def realtime_ok_by_offset(panel: es.Panel, F: pd.DataFrame, base_pos: np.ndarray, dates, k: int) -> np.ndarray:
    """표본마다 오프셋 k 창 마감이 D+1 08:00 KST 전인지(창 없음 = 결측 → False)."""
    g = F[F["market"] == panel.market].sort_values("close_kst").reset_index(drop=True)
    close = g["close_kst"].to_numpy("datetime64[m]")
    out = np.zeros(len(base_pos), bool)
    for i, (b, d) in enumerate(zip(base_pos, dates)):
        j = b + k
        if b < 0 or j < 0 or j >= len(close):
            continue
        cut = np.datetime64(pd.Timestamp(d) + pd.Timedelta(days=1, hours=CUTOFF_H), "m")
        out[i] = close[j] < cut
    return out


def kd1_window(Fk: pd.DataFrame, date: str) -> str | None:
    """보조: 미국 날짜 D의 장 전체와 다음 날 아침을 담는 K 창 = D+1 08:00 KST에 닫히는 K 창(없으면 None)."""
    target = pd.Timestamp(date) + pd.Timedelta(days=1, hours=8)
    hit = Fk.loc[Fk["close_kst"] == target, "win_id"]
    return str(hit.iloc[0]) if len(hit) else None


# ---------------------------------------------------------------- 실행

def main() -> int:
    t0 = datetime.now(es.KST).replace(microsecond=0)
    ev_path = RES / "events.json"
    feat_path = RES / "features_all.parquet"
    meta_path = RES / "features_all_meta.json"
    price_path = RES / "prices_close.csv"
    judge_sha = es.sha(JUDGE)
    ev_raw = json.loads(ev_path.read_text("utf-8"))
    ev, mode = es.confirmed_only_events(ev_raw, es.CONFIRMED_ASOF)     # 모든 자산 점검(멈춤 조건 그대로)
    sub = subset_events(ev)
    tinfo = sub["assets"][ASSET]["thresholds"][THR]
    contrasts = es.load_contrasts(sub)
    assert [c.name for c in contrasts] == [f"{ASSET}_{THR}_bottom", f"{ASSET}_{THR}_top"], [c.name for c in contrasts]
    F, meta = es.load_features(feat_path, meta_path)
    panel = es.build_panel(F, meta, "us")
    kpanel = es.build_panel(F, meta, "kr")
    Fk = F[F["market"] == "kr"]
    prices = pd.read_csv(price_path)
    close = es.price_series(prices, ASSET)
    FIG.mkdir(parents=True, exist_ok=True)
    piv = {p["date"]: p for p in tinfo["pivots"]}

    tests, leads, nows, loos, prof_rows, kd1_rows = [], [], [], [], [], []
    summ = {"made": t0.isoformat(), "combo": f"{ASSET} {THR}%", "mode": mode["contrasts"].get(f"{ASSET}_{THR}"),
            "mode_all_assets_checked": True, "asof": es.CONFIRMED_ASOF,
            "inputs_sha256": {"events.json": es.sha(ev_path), "features_all.parquet": es.sha(feat_path),
                              "features_all_meta.json": es.sha(meta_path), "prices_close.csv": es.sha(price_path)},
            "judge_py_sha256": judge_sha, "judge_py_sha_matches_addendum11": judge_sha == JUDGE_SHA,
            "judge_py_executed": False,
            "settings": {"n_null": es.N_NULL, "alpha": es.ALPHA, "cls_offsets": list(es.CLS_OFFSETS), "lead": list(es.LEAD),
                         "placebo_pool": es.PLACEBO_POOL, "min_true": es.MIN_TRUE, "min_fake": es.MIN_FAKE,
                         "z": [es.Z_N, es.Z_MIN], "pct": [es.P_N, es.P_MIN, es.P_HI, es.P_LO]},
            "realtime_rule": {"cutoff": "사건일 D의 다음 날 08:00 KST 전(= W_KR(D+1) 마감 07:59)",
                              "W_U_offset0": "U(D) 창 D 21:30 KST 마감 → 규칙 안",
                              "W_U_offset1": "U(D+1) 창 다음 미국 거래일 21:30 KST 마감 → 규칙 밖(알아챔 후보에서 뺌, 참고)",
                              "lead": "오프셋 −5~−1 → 규칙 안",
                              "supplement_KD1": "K(D+1) 창(D 08:00 ~ D+1 08:00 KST) = 보조, 사전 등록 밖, 후보로 올리지 않음"},
            "n_grid_features": len(panel.feats),
            "grid_by_status": {s: sum(meta[f]["status"] == s for f in panel.feats) for s in ("통과", "설명용")},
            "contrasts": []}
    tail = tinfo.get("tail_unconfirmed") or {}
    summ["tail_unconfirmed"] = {k: tail.get(k) for k in ("kind", "date", "win_id", "close", "status", "confirm_level", "amp_in",
                                                         "prev_pivot_date")}
    summ["tail_unconfirmed"]["prev_pivot_close"] = piv.get(tail.get("prev_pivot_date"), {}).get("close")
    dr = json.loads((RES / "dryrun.json").read_text("utf-8"))
    summ["dryrun_power_us_tp_LS_A_8ev"] = {str(r["delta_sd"]): r["detect_offset0"] for r in dr["injection"]
                                           if r["market"] == "us" and r["target"] == "tp_LS_A" and r["scenario"].startswith("중간")}

    for c in contrasts:
        r = es.run_contrast(c, panel, close, n_null=es.N_NULL)
        (FIG / f"{c.name}.html").write_text(es.figure_html(
            r, panel, f"확정 사건만 모드(2026-10-22 종가). 오프셋 +1(U(D+1) 창)은 D+1 21:30 KST 마감이라 '알아챔' 실시간 규칙 밖이다. "
                      "오프셋이 '확정 지연 + 1' 이상인 창은 확정 뒤 정보다(사후 설명용). "), "utf-8")
        cls = r["cls"]
        base, y = cls["base"], cls["y"]
        all_dates = np.r_[c.events["date"].to_numpy(), c.fake_reps["date"].to_numpy()]
        ev_pos = es.positions(panel, c.events["win_id"])
        fk_pos = es.positions(panel, c.fake_reps["win_id"])
        have = np.r_[ev_pos, fk_pos] >= 0
        dates = all_dates[have]
        rt_ok = {k: realtime_ok_by_offset(panel, F, base, dates, k) for k in es.CLS_OFFSETS}
        know = np.array([np.datetime64(d, "D") < KNOW_END for d in dates])

        tb = r["table"].copy()
        tb["realtime_ok"] = tb["offset"].map(lambda k: bool(rt_ok[k].all()))
        tb["alarm_candidate"] = tb["candidate"] & tb["realtime_ok"]
        hits_ev, hits_fk, fits, n_know = [], [], [], []
        V = es.gather(panel.Z, base, es.CLS_OFFSETS)
        for _, row in tb.iterrows():
            if row["status"] != "시험":
                hits_ev.append(np.nan)
                hits_fk.append(np.nan)
                fits.append({})
                n_know.append(np.nan)
                continue
            oi = list(es.CLS_OFFSETS).index(row["offset"])
            fi = panel.feats.index(row["feature"])
            v = V[:, oi, fi]
            ok = np.isfinite(v)
            pr = loeo_pred(v[ok], y[ok])
            yy = y[ok]
            ba_chk = 0.5 * ((pr & yy).sum() / yy.sum() + (~pr & ~yy).sum() / (~yy).sum())
            assert abs(ba_chk - row["ba"]) < 1e-9, (row["feature"], row["offset"], ba_chk, row["ba"])
            hits_ev.append(int((pr & yy).sum()))
            hits_fk.append(int((~pr & ~yy).sum()))
            fits.append(full_fit(v[ok], yy))
            n_know.append(int((know[ok] & yy).sum()))
            dd = dates[ok]
            for d_, yv, zv, pv in zip(dd, yy, v[ok], pr):
                loos.append({"kind": c.kind, "feature": row["feature"], "offset": int(row["offset"]), "date": d_,
                             "label": "사건" if yv else "가짜", "z": zv, "pred_event": bool(pv), "correct": bool(pv == yv),
                             "knowledge_period": bool(np.datetime64(d_, "D") < KNOW_END)})
        tb["loo_hit_events"] = hits_ev
        tb["loo_hit_fakes"] = hits_fk
        tb["fit_direction"] = [f.get("direction") for f in fits]
        tb["fit_threshold_z"] = [f.get("threshold_z") for f in fits]
        tb["fit_ba_in_sample"] = [f.get("ba_in_sample") for f in fits]
        tb["events_in_knowledge_period"] = n_know
        tb["label_knowledge_caution"] = tb["bundle"].isin(LABEL_BUNDLES) & (tb["events_in_knowledge_period"] > 0)
        tests.append(tb.assign(asset=ASSET, threshold=THR, kind=c.kind))

        fam = r["family"]
        pass_mask = tb["status_feature"].eq("통과").to_numpy()
        off0 = tb["offset"].eq(0).to_numpy()
        fam_rt = es.family_count(np.where(pass_mask & off0, cls["p_obs"], np.nan), cls["T_null"])
        fam_off1 = es.family_count(np.where(pass_mask & ~off0, cls["p_obs"], np.nan), cls["T_null"])

        lt = r["lead"].copy()
        lt["events_in_knowledge_period"] = int((know & y).sum())
        leads.append(lt.assign(asset=ASSET, threshold=THR, kind=c.kind))
        nw = r["now"]
        if len(nw):
            nows.append(nw.assign(asset=ASSET, threshold=THR, kind=c.kind))
        pr_ = r["profiles"]
        for fi, f in enumerate(panel.feats):
            for j, k in enumerate(es.OFFSETS):
                prof_rows.append({"kind": c.kind, "feature": f, "offset": int(k), "ev_mean": pr_["ev_mean"][j, fi],
                                  "ev_n": int(pr_["ev_n"][j, fi]),
                                  "fk_mean": pr_["fk_mean"][j, fi] if pr_["fk_mean"] is not None else np.nan,
                                  "pl_mean": pr_["pl_mean"][j, fi] if "pl_mean" in pr_ else np.nan,
                                  "pl_lo": pr_["pl_lo"][j, fi] if "pl_lo" in pr_ else np.nan,
                                  "pl_hi": pr_["pl_hi"][j, fi] if "pl_hi" in pr_ else np.nan})
        # 가격 경로(사건일 종가 대비 %)
        pp = r["price_paths"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            price_mean = {k: (np.nanmean(v, 0) * 100).round(2).tolist() if len(v) else None for k, v in pp.items()}

        # 보조: K(D+1)
        kev = [kd1_window(Fk, d) for d in c.events["date"]]
        kfk = [kd1_window(Fk, d) for d in c.fake_reps["date"]]
        kev_pos, kfk_pos = es.positions(kpanel, kev), es.positions(kpanel, kfk)
        ks = int(hashlib.sha256(f"{es.SEED}:{c.name}:KD1".encode()).hexdigest()[:8], 16)
        kcls = es.classify(kpanel, kev_pos, kfk_pos, offsets=(0,), n_null=es.N_NULL, seed=ks)
        ktb = kcls["table"].copy()
        ktb["status_feature"] = ktb["feature"].map(lambda f: meta[f]["status"])
        ktb["bundle"] = ktb["feature"].map(lambda f: meta[f]["bundle"])
        kleak = es.leak_test(kpanel, kev_pos, kfk_pos)
        ktb["leak_suspect"] = ktb["bundle"].eq(es.LEAK_BUNDLE) & kleak["flag"]
        ktb["signal_supplement"] = (ktb["p"] < es.ALPHA) & ktb["status_feature"].eq("통과") & ~ktb["leak_suspect"]
        kfam = es.family_count(np.where(ktb["status_feature"].eq("통과").to_numpy(), kcls["p_obs"], np.nan), kcls["T_null"])
        kd1_rows.append(ktb.assign(asset=ASSET, threshold=THR, kind=c.kind))
        # 보조 지금 위치: K2026-10-23(10/22 미국 장 + 10/23 아침)
        know_rows = []
        kE = es.gather(kpanel.Z, kev_pos, [0])[:, 0, :]
        kFk = es.gather(kpanel.Z, kfk_pos, [0])[:, 0, :]
        for wid in ("K2026-10-16", "K2026-10-17", "K2026-10-20", "K2026-10-21", "K2026-10-22", "K2026-10-23"):
            t = kpanel.pos.get(wid)
            if t is None:
                continue
            for f in ktb.loc[ktb["signal_supplement"], "feature"]:
                fi = kpanel.feats.index(f)
                z = kpanel.Z[t, fi]
                e_ = kE[:, fi][np.isfinite(kE[:, fi])]
                k_ = kFk[:, fi][np.isfinite(kFk[:, fi])]
                know_rows.append({"win_id": wid, "feature": f, "z": z,
                                  "pct_in_events": float(((e_ < z).sum() + .5 * (e_ == z).sum()) / len(e_)) if len(e_) and np.isfinite(z) else None,
                                  "pct_in_fakes": float(((k_ < z).sum() + .5 * (k_ == z).sum()) / len(k_)) if len(k_) and np.isfinite(z) else None})

        # 사건·가짜 목록
        evs = []
        for d, w in zip(c.events["date"], c.events["win_id"]):
            p = piv[d]
            evs.append({"date": d, "win_id": w, "close": p["close"], "confirm_date": p["confirm_date"],
                        "confirm_lag_td": p["confirm_lag_td"], "first_post_confirm_offset": int(p["confirm_lag_td"]) + 1,
                        "knowledge_period": bool(np.datetime64(d, "D") < KNOW_END), "kd1_window": kd1_window(Fk, d),
                        "amp_in": p.get("amp_in"), "amp_out": p.get("amp_out")})
        fks = [{"date": d, "win_id": w, "run": int(rn), "close": cl, "kd1_window": kd1_window(Fk, d)}
               for d, w, rn, cl in zip(c.fake_reps["date"], c.fake_reps["win_id"], c.fake_reps["run"], c.fake_reps["close"])]

        cand = tb[tb["candidate"]]
        alarm = tb[tb["alarm_candidate"]]
        lc = lt[lt["candidate"]]
        summ["contrasts"].append({
            "name": c.name, "kind": c.kind, "n": r["n"], "events": evs, "fake_reps": fks,
            "n_fakes_all": int(len(c.fakes_all)), "held_near_tail_excluded": c.info.get("held_near_tail"),
            "tail_unconfirmed": c.info.get("tail_unconfirmed"),
            "family": fam, "family_pass_offset0_realtime": fam_rt, "family_pass_offset1_outside_rule": fam_off1,
            "lead_family": r["lead_family"], "leak": r["leak"], "outside_band": r["outside_band"], "placebo": r["placebo"],
            "candidates_preregistered": cand.reindex(columns=["feature", "offset", "ba", "p", "direction", "loo_hit_events",
                                                              "loo_hit_fakes", "realtime_ok", "label_knowledge_caution"]).to_dict("records"),
            "alarm_candidates_realtime": alarm.reindex(columns=["feature", "offset", "ba", "p", "direction", "loo_hit_events",
                                                                "loo_hit_fakes", "label_knowledge_caution"]).to_dict("records"),
            "lead_candidates": lc.reindex(columns=["feature", "side", "rate_events", "rate_fakes", "rate_placebo_mean",
                                                   "rate_placebo_q975", "p", "p_vs_placebo"]).to_dict("records"),
            "top_tests_pass": tb[(tb["status"] == "시험") & tb["status_feature"].eq("통과")].sort_values(["p", "ba"], ascending=[True, False])
            .head(10).reindex(columns=["feature", "offset", "ba", "p", "direction", "loo_hit_events", "loo_hit_fakes", "n_events",
                                       "n_fakes", "realtime_ok", "leak_suspect", "label_knowledge_caution", "fit_direction",
                                       "fit_threshold_z"]).to_dict("records"),
            "top_lead_pass": lt[(lt["status"] == "시험") & lt["status_feature"].eq("통과")].sort_values("p").head(10)
            .reindex(columns=["feature", "side", "rate_events", "rate_fakes", "rate_placebo_mean", "rate_placebo_q975", "p",
                              "p_vs_placebo", "leak_suspect", "candidate"]).to_dict("records"),
            "price_path_mean_pct": price_mean,
            "supplement_KD1": {"n_events_with_window": int((kev_pos >= 0).sum()), "n_fakes_with_window": int((kfk_pos >= 0).sum()),
                               "missing_events": [d for d, w in zip(c.events["date"], kev) if w is None],
                               "missing_fakes": [d for d, w in zip(c.fake_reps["date"], kfk) if w is None],
                               "family_pass": kfam, "leak": kleak,
                               "signals": ktb[ktb["signal_supplement"]].reindex(columns=["feature", "ba", "p", "mean_z_events",
                                                                                         "mean_z_fakes", "direction"]).to_dict("records"),
                               "now": know_rows},
        })

    T = pd.concat(tests)
    T.to_csv(RES / f"{TAG}_tests.csv", index=False, encoding="utf-8-sig")
    pd.concat(leads).to_csv(RES / f"{TAG}_lead.csv", index=False, encoding="utf-8-sig")
    NW = pd.concat(nows) if nows else pd.DataFrame()
    NW.to_csv(RES / f"{TAG}_now.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(loos).to_csv(RES / f"{TAG}_loo.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prof_rows).to_csv(RES / f"{TAG}_profiles.csv", index=False, encoding="utf-8-sig")
    pd.concat(kd1_rows).to_csv(RES / f"{TAG}_kd1_tests.csv", index=False, encoding="utf-8-sig")

    # 지금 위치 요약(설명용): 통과 특징 중 오프셋 0 p 낮은 3개 + 후보 + 앞섬 상위
    now_summary = {}
    if len(NW):
        for cc in summ["contrasts"]:
            feats = []
            for rr in cc["candidates_preregistered"] + cc["top_tests_pass"][:3]:
                if rr["feature"] not in feats:
                    feats.append(rr["feature"])
            for rr in cc["lead_candidates"]:
                if rr["feature"] not in feats:
                    feats.append(rr["feature"])
            q = NW[(NW["kind"] == cc["kind"]) & NW["feature"].isin(feats)]
            now_summary[cc["kind"]] = q.reindex(columns=["win_id", "feature", "z", "pct60", "pct_in_events", "pct_in_fakes",
                                                         "pct_in_placebo", "realtime_caution"]).to_dict("records")
    summ["now_summary"] = now_summary
    summ["outputs"] = [str(RES / f"{TAG}.json"), str(RES / f"{TAG}.md")] + \
        [str(RES / f"{TAG}_{s}") for s in ("tests.csv", "lead.csv", "now.csv", "loo.csv", "profiles.csv", "kd1_tests.csv")] + \
        [str(FIG / f"{c.name}.html") for c in contrasts]
    summ["seconds"] = (datetime.now(es.KST) - t0).total_seconds()
    (RES / f"{TAG}.json").write_text(json.dumps(clean(summ), ensure_ascii=False, indent=1), "utf-8")
    (RES / f"{TAG}.md").write_text(report_md(summ, T, NW), "utf-8")
    print(json.dumps(clean({c["name"]: {"family": c["family"]["통과"], "rt": c["family_pass_offset0_realtime"],
                                        "lead": c["lead_family"]["통과"], "leak": c["leak"]["flag"],
                                        "cand": c["candidates_preregistered"], "lead_cand": c["lead_candidates"]}
                            for c in summ["contrasts"]}), ensure_ascii=False, indent=1))
    print("seconds", summ["seconds"])
    return 0


NOW_WINS = ["U2026-10-15", "U2026-10-16", "U2026-10-19", "U2026-10-20", "U2026-10-21", "U2026-10-22"]


def _pct(x):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.0f}%"


def report_md(s: dict, T: pd.DataFrame, NW: pd.DataFrame) -> str:
    L = []
    C = {c["kind"]: c for c in s["contrasts"]}
    m = s["mode"]
    ins = s["inputs_sha256"]
    L.append(f"# 개미 지수 사건 연구: SOXX 12%(강건판), 확정 사건만 (부록 13·14)\n")
    L.append("> 탐색 연구다. 결과는 '후보'이며 매매 규칙이 아니다. 후보는 2026-10-24 뒤 새로 확정되는 고점·저점으로만 확인한다. "
             "'지금이 바닥·고점'이라고 단정하지 않는다.\n")
    L.append(f"> {s['made']} 실행. 코드 `events/code/run_combo_SOXX_12.py`(event_study.py 함수 그대로, 대비 씨앗 = 대비 이름 → "
             f"전체 실행과 같은 값). 입력 sha256: events.json {ins['events.json'][:12]}…, features_all.parquet "
             f"{ins['features_all.parquet'][:12]}…, prices_close.csv {ins['prices_close.csv'][:12]}…. 확정 사건만 모드: "
             f"{s['asof']} 종가까지 확정된 꺾임점만(마지막 확정일 {m['last_confirm_date']}), 끝 극값 {m['tail_excluded']['date']}"
             f"({'저점' if m['tail_excluded']['kind'] == 'low' else '고점'}, 미확정)과 보류 가짜 바닥 "
             f"{', '.join(m['held_fakes_excluded']['fake_bottoms']) or '없음'}은 뺐다. `final/code/judge.py`는 실행하지 않았고 "
             f"SHA-256이 부록 11 기록과 {'같다' if s['judge_py_sha_matches_addendum11'] else '다르다(!)'}.\n")
    L.append(f"> 탐색 범위: W_U 특징 {s['n_grid_features']}개(통과 {s['grid_by_status']['통과']} · 설명용 "
             f"{s['grid_by_status']['설명용']}). 가르기 = 특징 × 오프셋 2개(0, +1) × 종류 2개 = {s['n_grid_features'] * 4}칸, "
             f"앞섬 = 특징 × 쪽 2개(상위·하위 10%) × 종류 2개 = {s['n_grid_features'] * 4}칸, 보조 K(D+1) = 특징 × 종류 2개 = "
             f"{s['n_grid_features'] * 2}칸. 섞기 {s['settings']['n_null']}번, 위약 = 사건마다 가격이 비슷한 날 "
             f"{s['settings']['placebo_pool']}일.\n")

    # 0. 결론
    L.append("## 0. 결론\n")
    b, t = C["bottom"], C["top"]
    L.append(f"- 바닥 {b['n']['events_with_window']}개·고점 {t['n']['events_with_window']}개로는 우연과 구별되는 특징이 없다. "
             f"통과 특징 가르기에서 p < 0.05 칸은 바닥 {b['family']['통과']['observed']}개(우연 기대 "
             f"{f2(b['family']['통과']['expected_by_chance'])}), 고점 {t['family']['통과']['observed']}개(우연 기대 "
             f"{f2(t['family']['통과']['expected_by_chance'])})다.")
    al = [r for k in ("bottom", "top") for r in C[k]["alarm_candidates_realtime"]]
    pre = [(k, r) for k in ("bottom", "top") for r in C[k]["candidates_preregistered"]]
    if pre:
        L.append("- 사전 등록 조건(p < 0.05·검수 통과·누출 의심 아님)을 채운 칸: " + "; ".join(
            f"{TYPE_KO[k]} `{r['feature']}` @{r['offset']:+d}(BA {f2(r['ba'])}, p {f2(r['p'], 3)}, 하나 빼고 맞히기 사건 "
            f"{int(r['loo_hit_events'])}/{C[k]['n']['events_with_window']}·가짜 {int(r['loo_hit_fakes'])}/"
            f"{C[k]['n']['fake_reps_with_window']}, {'실시간 규칙 안' if r['realtime_ok'] else '실시간 규칙 밖'})" for k, r in pre)
                 + ". " + ("실시간 규칙 안 '알아챔' 후보는 없다." if not al else "") +
                 " 관측 개수가 우연 기대보다 적어 후보로 보기 어렵다.")
    L.append(f"- 실시간 규칙 안(오프셋 0, D 21:30 KST 마감)만 세면 바닥 {b['family_pass_offset0_realtime']['observed']}/"
             f"{b['family_pass_offset0_realtime']['n_tests']}칸(우연 기대 {f2(b['family_pass_offset0_realtime']['expected_by_chance'])}), "
             f"고점 {t['family_pass_offset0_realtime']['observed']}/{t['family_pass_offset0_realtime']['n_tests']}칸(우연 기대 "
             f"{f2(t['family_pass_offset0_realtime']['expected_by_chance'])}).")
    L.append(f"- 앞섬(사건 전 1~5창 극값 닿음): 바닥 {b['lead_family']['통과']['observed']}/{b['lead_family']['통과']['n_tests']}칸"
             f"(우연 기대 {f2(b['lead_family']['통과']['expected_by_chance'])}), 고점 {t['lead_family']['통과']['observed']}/"
             f"{t['lead_family']['통과']['n_tests']}칸(우연 기대 {f2(t['lead_family']['통과']['expected_by_chance'])}). 앞섬 후보 없음.")
    L.append(f"- 조회·추천 누출 시험: 바닥 {'의심' if b['leak']['flag'] else '통과'}, 고점 {'의심' if t['leak']['flag'] else '통과'}"
             "(사건일 (b) z 중앙값이 가짜보다 낮음). 다만 조회·추천 칸도 우연 수준이라 후보가 없다.")
    pw = s["dryrun_power_us_tp_LS_A_8ev"]
    L.append(f"- 사건이 4~5개라 검정력이 낮다: 시운전(W_U `tp_LS_A`, 사건 8·가짜 6 — 이번보다 큰 판)에서 오프셋 0 검출률이 1 SD 차이면 "
             f"{f2(pw.get('1.0'))}, 2 SD면 {f2(pw.get('2.0'))}였다. '신호 없음'이 아니라 '이 사건 수로는 가를 수 없음'이다.\n")
    L.append("| 종류 | 사건(창 있음) | 가짜 묶음 대표 | 칸(통과) | p<0.05 관측 | 우연 기대 | 귀무 95% | 묶음 p | 규칙 안(오프셋 0) 관측/기대 | "
             "앞섬 관측/기대 | 누출 시험 | 후보 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in ("bottom", "top"):
        c = C[k]
        fa, f0, lf = c["family"]["통과"], c["family_pass_offset0_realtime"], c["lead_family"]["통과"]
        cand = ", ".join(f"{r['feature']}@{r['offset']:+d}({'규칙 밖' if not r['realtime_ok'] else '규칙 안'})"
                         for r in c["candidates_preregistered"]) or "없음"
        L.append(f"| {TYPE_KO[k]} | {c['n']['events_with_window']}/{c['n']['events_total']} | {c['n']['fake_reps_with_window']} | "
                 f"{fa['n_tests']} | {fa['observed']} | {f2(fa['expected_by_chance'])} | {f2(fa['null_q95'], 0)} | "
                 f"{f2(fa['p_family'], 3)} | {f0['observed']}/{f2(f0['expected_by_chance'])} | {lf['observed']}/"
                 f"{f2(lf['expected_by_chance'])} | {'의심' if c['leak']['flag'] else '통과'} | {cand} |")
    L.append("\n- '우연 기대' = 사건·가짜 표지를 섞은 1,000번에서 p < 0.05가 나온 칸 수의 평균. 칸 수가 표에 있는 수와 다르면 사건이 "
             "4개 미만으로 남은 칸(표본 부족)을 뺀 것이다.\n")

    # 실시간 규칙
    L.append("## 실시간 규칙(알아챔 = 사건일 D 다음 날 08:00 KST 전에 쓴 글)\n")
    L.append("| 쓰임 | 창 | 마감 | 규칙 |")
    L.append("|---|---|---|---|")
    L.append("| 가르기 오프셋 0 | U(D) | D 21:30 KST(D 미국 장 시작 전) | 안 |")
    L.append("| 가르기 오프셋 +1 | U(D+1) | 다음 미국 거래일 21:30 KST | **밖**(13.5시간 이상 늦음, 국내 장중·미국 프리장 정보 포함) → 참고만 |")
    L.append("| 앞섬 −5~−1 | U(D−5)~U(D−1) | 사건 전 | 안 |")
    L.append("| 보조 '+1 대체'(사전 등록 밖) | K(D+1) | D+1 08:00 KST(D 미국 장 전체 포함) | 안. D+1이 국내 휴장이면 결측 |")
    L.append("\n- 극값 확정은 사건 뒤 2~16거래일이다(아래 표). 오프셋 '확정 지연 + 1' 이상 창은 확정 뒤 정보라 그림에서도 사후 설명용이다.\n")

    # 사건 목록
    L.append("## 사건과 가짜\n")
    L.append("| 종류 | 사건일 | 창 | 종가 | 확정일 | 확정 지연(거래일) | 확정 뒤 첫 오프셋 | 라벨 지식 기간(D < 7/1) | K(D+1) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k in ("bottom", "top"):
        for e in C[k]["events"]:
            L.append(f"| {TYPE_KO[k]} | {e['date']} | {e['win_id']} | {e['close']:,.2f} | {e['confirm_date']} | {e['confirm_lag_td']} | "
                     f"+{e['first_post_confirm_offset']} | {'예' if e['knowledge_period'] else ''} | {e['kd1_window'] or '없음(다음 날 국내 비거래일)'} |")
    for k, nm in (("bottom", "가짜 바닥"), ("top", "가짜 고점")):
        c = C[k]
        L.append(f"\n- {nm}: 날 {c['n_fakes_all']}개, 묶음 대표 {c['n']['fake_reps']}개 — " +
                 ", ".join(f"{f['date']}" for f in c["fake_reps"]) + ".")
    L.append(f"- 뺀 것: 끝 극값 {m['tail_excluded']['date']}(저점 후보, 종가 {m['tail_excluded']['close']:,.2f}, 확정에 필요한 종가 "
             f"{m['tail_excluded']['confirm_level']:,.2f} 이상)과 그 ±3거래일 보류 가짜 바닥 "
             f"{', '.join(m['held_fakes_excluded']['fake_bottoms'])}. 위약 후보에서도 끝 극값 ±3거래일은 뺐다(기존 규칙).\n")

    # 1. 그림
    L.append("## 1. 사건 전후 그림(설명용)\n")
    L.append("| 종류 | 오프셋 0: 위약 띠 밖 / 특징 | 오프셋 +1: 띠 밖 / 특징(띠 밖 특징) | 독립이면 기대 | 가격(사건일 종가 대비 평균 %, −10 / −5 / +5 / +10) 사건 · 가짜 · 위약 | 그림 |")
    L.append("|---|---|---|---|---|---|")
    for k in ("bottom", "top"):
        c, ob = C[k], C[k]["outside_band"]
        pm = c["price_path_mean_pct"]

        def pth(a):
            return "—" if not a else f"{a[0]:+.1f} / {a[5]:+.1f} / {a[15]:+.1f} / {a[20]:+.1f}"
        L.append(f"| {TYPE_KO[k]} | {ob['0']['outside_band']} / {ob['0']['n_features']} | {ob['1']['outside_band']} / "
                 f"{ob['1']['n_features']}({', '.join(ob['1']['features']) or '없음'}) | {ob['0']['expected_if_independent']} | "
                 f"{pth(pm['events'])} · {pth(pm['fakes'])} · {pth(pm['placebo'])} | `{TAG}_figs/{ASSET}_{THR}_{k}.html` |")
    L.append("\n- 오프셋 0에서는 바닥·고점 모두 위약 띠 밖 특징이 0개다(독립이면 기대 2.3). +1의 띠 밖 특징은 감정 층(설명용)·다른의견 "
             "비율(설명용) 위주이고 실시간 규칙 밖이다.\n")

    # 2. 가르기
    L.append("## 2. 가르기(하나 빼고 맞히기, 통과 특징 p 낮은 순 6칸)\n")
    L.append("| 종류 | 특징@오프셋 | BA | 섞기 p | 맞힌 사건 | 맞힌 가짜 | 방향 | 가짜 날 전부 BA | 실시간 | 라벨 지식 기간 주의 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for k in ("bottom", "top"):
        c = C[k]
        tb = T[(T["kind"] == k) & (T["status"] == "시험") & (T["status_feature"] == "통과")].sort_values(["p", "ba"],
                                                                                                    ascending=[True, False])
        for _, r in tb.head(6).iterrows():
            L.append(f"| {TYPE_KO[k]} | `{r['feature']}`@{int(r['offset']):+d} | {f2(r['ba'])} | {f2(r['p'], 3)} | "
                     f"{int(r['loo_hit_events'])}/{int(r['n_events'])} | {int(r['loo_hit_fakes'])}/{int(r['n_fakes'])} | "
                     f"{r['direction']} | {f2(r['ba_all_fake_days_run_groups'])} | {'안' if r['realtime_ok'] else '밖'} | "
                     f"{'예' if r['label_knowledge_caution'] else ''} |")
    ex = T[(T["status"] == "시험") & (T["status_feature"] == "설명용") & (T["p"] < es.ALPHA)]
    if len(ex):
        L.append("\n- 설명용(감정 층 검수 실패 등) p < 0.05: " + ", ".join(
            f"{TYPE_KO[r['kind']]} `{r['feature']}`@{int(r['offset']):+d}(p {f2(r['p'], 3)})" for _, r in ex.iterrows()) +
            f". 고점 설명용 묶음 관측 {t['family']['설명용']['observed']} vs 우연 기대 {f2(t['family']['설명용']['expected_by_chance'])}"
            f"(묶음 p {f2(t['family']['설명용']['p_family'], 3)}). 모두 실시간 규칙 밖(+1)이고 후보가 아니다.")
    sb = T[T["status"] == "표본 부족"]
    L.append(f"- 표본 부족 칸 {len(sb)}개: 라벨 층 최소 표본 규칙으로 U2026-06-03(고점)·U2026-06-05(바닥) 창의 감정·예측 층이 결측이라 "
             "오프셋 0 라벨 층 칸 다수가 사건 3개로 남았다.")
    kp_b = sum(e["knowledge_period"] for e in b["events"])
    kp_t = sum(e["knowledge_period"] for e in t["events"])
    L.append(f"- 라벨 지식 기간: 바닥 {kp_b}/{len(b['events'])}, 고점 {kp_t}/{len(t['events'])} 사건이 2026-07-01 전이다. 예측·감정 층 "
             "특징은 라벨 모델이 결과를 알았을 수 있는 기간이라, 이 층에서 신호가 나와도 7/1 뒤 사건(바닥 1·고점 1)만으로는 확인할 수 없다.\n")

    # 3. 앞섬
    L.append("## 3. 앞섬(사건 전 1~5창에 직전 60창 상위·하위 10% 닿음, 통과 특징 p 낮은 순 4칸)\n")
    L.append("| 종류 | 특징(쪽) | 사건 닿음 | 가짜 닿음 | 위약 닿음(평균) | 위약 97.5% 분위 | 섞기 p | 위약 대비 p |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k in ("bottom", "top"):
        for r in C[k]["top_lead_pass"][:4]:
            L.append(f"| {TYPE_KO[k]} | `{r['feature']}`({r['side']}) | {_pct(r['rate_events'])} | {_pct(r['rate_fakes'])} | "
                     f"{_pct(r['rate_placebo_mean'])} | {_pct(r['rate_placebo_q975'])} | {f2(r['p'], 3)} | {f2(r['p_vs_placebo'], 3)} |")
    LD = pd.read_csv(RES / f"{TAG}_lead.csv")
    q1 = LD[(LD["status"] == "시험") & (LD["rate_placebo_q975"] >= 1 - 1e-9)]
    L.append(f"\n- 사건이 4~5개면 위약 닿음 비율이 40% 안팎만 돼도 97.5% 분위가 100%가 된다. 그런 칸이 {len(q1)}/"
             f"{int((LD['status'] == '시험').sum())}칸이라, 그 칸들은 사건이 모두 닿아도 앞섬 후보가 될 수 없다(구조적 한계).\n")

    # 누출
    L.append("## 누출 시험(조회·추천 (b) = 조회 ÷ 마감 전 댓글, 사건일 vs 가짜 사건일 z 중앙값)\n")
    L.append("| 종류 | 오프셋 0: 사건 / 가짜 (n) | 오프셋 +1: 사건 / 가짜 (n) | 판정 |")
    L.append("|---|---|---|---|")
    for k in ("bottom", "top"):
        lk = C[k]["leak"]["by_offset"]

        def cell(v):
            return (f"{f2(v.get('median_events'))} / {f2(v.get('median_fakes'))} ({v['n_events']}·{v['n_fakes']})"
                    if "median_events" in v else v.get("result", "—"))
        L.append(f"| {TYPE_KO[k]} | {cell(lk.get(0, lk.get('0', {})))} | {cell(lk.get(1, lk.get('1', {})))} | "
                 f"{'누출 의심' if C[k]['leak']['flag'] else '통과(의심 아님)'} |")
    L.append("\n- 규칙: 오프셋 0·+1 중 하나라도 사건 중앙값 > 가짜 중앙값(또는 값 부족)이면 조회·추천 묶음 전체를 누출 의심으로 뺀다. "
             "두 대비 모두 사건 쪽이 낮아 의심이 켜지지 않았다. 조회·추천 원값 특징(실시간 비교 불가)은 소급 창(10/12까지)만 쓴다.\n")

    # 4. 지금 위치
    L.append("## 4. 지금 위치(설명용 — '지금이 바닥·고점'이라는 뜻이 아니다)\n")
    L.append("U(D) 창 = D 장 시작 전 글. U2026-10-22는 미확정 끝 극값(10/22 저점 후보)의 오프셋 0 창이다. 백분위 = 그 창 z가 사건(오프셋 0) "
             "z 분포·가짜 분포에서 차지하는 자리. 10/13 뒤 창은 실시간 수집 창이다.\n")
    L.append("| 종류 | 특징 | " + " | ".join(w[1:] for w in NOW_WINS) + " | 10/22 창: 사건 분포 / 가짜 분포 백분위 |")
    L.append("|---|---|" + "---|" * len(NOW_WINS) + "---|")
    show = {"bottom": ["tp_MC_A", "tp_LL_US_A", "pr_P_bar", "act_share_day"], "top": ["tp_LS_A", "tp_LL_US_A", "pr_P_bar"]}
    for k, feats in show.items():
        for f in feats:
            q = NW[(NW["kind"] == k) & (NW["feature"] == f)].set_index("win_id")
            zs = [f2(q.loc[w, "z"]) if w in q.index else "—" for w in NOW_WINS]
            last = q.loc["U2026-10-22"] if "U2026-10-22" in q.index else None
            tail = (f"{_pct(last['pct_in_events'])} (n {int(last['n_events'])}) / {_pct(last['pct_in_fakes'])} "
                    f"(n {int(last['n_fakes'])})") if last is not None else "—"
            L.append(f"| {TYPE_KO[k]} | `{f}` | " + " | ".join(zs) + f" | {tail} |")
    def nz(k, f, w, col="z"):
        q = NW[(NW["kind"] == k) & (NW["feature"] == f) & (NW["win_id"] == w)]
        return float(q[col].iloc[0]) if len(q) else np.nan
    tmc = T[(T["kind"] == "bottom") & (T["feature"] == "tp_MC_A") & (T["offset"] == 0)].iloc[0]
    L.append(f"\n- 바닥 대비에서 실시간 규칙 안(오프셋 0) p가 가장 낮았던 `tp_MC_A`(반대매매·마진콜 작성자 몫, p {f2(tmc['p'], 3)} — "
             f"후보 아님)는 U10/15·10/16에 z {f2(nz('bottom', 'tp_MC_A', 'U2026-10-15'))}·{f2(nz('bottom', 'tp_MC_A', 'U2026-10-16'))}"
             f"(직전 60창 백분위 {_pct(nz('bottom', 'tp_MC_A', 'U2026-10-15', 'pct60'))}·"
             f"{_pct(nz('bottom', 'tp_MC_A', 'U2026-10-16', 'pct60'))})였다. 그런데 10/16은 가짜 바닥 묶음 대표(그 뒤 더 내려감)이고, "
             f"끝 극값 창 U10/22에서는 z {f2(nz('bottom', 'tp_MC_A', 'U2026-10-22'))}로 바닥 사건 분포 백분위 "
             f"{_pct(nz('bottom', 'tp_MC_A', 'U2026-10-22', 'pct_in_events'))} 자리다.")
    lb = LD[(LD["kind"] == "bottom") & (LD["feature"] == "pr_P_bar") & (LD["side"] == "하위 10%")].iloc[0]
    L.append(f"- 예측 층 `pr_P_bar`(군중 예측 P̄)는 U10/15·10/22에 z {f2(nz('bottom', 'pr_P_bar', 'U2026-10-15'))}·"
             f"{f2(nz('bottom', 'pr_P_bar', 'U2026-10-22'))}로 낮다. 앞섬 표에서 바닥 사건 {_pct(lb['rate_events'])}가 사건 전 1~5창에 "
             f"하위 10%에 닿았지만 가짜({_pct(lb['rate_fakes'])})·위약({_pct(lb['rate_placebo_mean'])})도 자주 닿아 후보가 아니다"
             f"(섞기 p {f2(lb['p'], 3)}, 라벨 지식 기간 사건 {int(lb['events_in_knowledge_period'])}/{int(lb['n_events'])}개).\n")

    # 보조
    L.append("## 보조: '+1 대체' K(D+1) 창(사전 등록 밖, 후보로 올리지 않음)\n")
    for k in ("bottom", "top"):
        sp = C[k]["supplement_KD1"]
        fam = sp["family_pass"]
        L.append(f"- {TYPE_KO[k]}: 창 있는 사건 {sp['n_events_with_window']}·가짜 {sp['n_fakes_with_window']}"
                 f"(없음: 사건 {', '.join(sp['missing_events']) or '—'}, 가짜 {', '.join(sp['missing_fakes']) or '—'}). " +
                 (f"통과 칸 {fam['n_tests']}, p < 0.05 관측 {fam['observed']}, 우연 기대 {f2(fam['expected_by_chance'])}."
                  if fam["n_tests"] else "사건 3개라 모든 칸 표본 부족(시험 못 함)."))
    L.append("")

    # 미확정
    L.append("## 미확정 끝 극값(사건에서 뺌)\n")
    tl = s["tail_unconfirmed"]
    L.append(f"- SOXX 12% 끝 극값 = 저점 후보 {tl['date']}(창 {tl['win_id']}, 종가 {tl['close']:,.2f}, {tl['status']}). "
             f"{tl['prev_pivot_date']} 고점({tl['prev_pivot_close']:,.2f}) 뒤 {100 * tl['amp_in']:.1f}% 하락 구간이고, 확정에는 "
             f"{tl['confirm_level']:,.2f} 이상 종가가 필요하다(10/23 뒤 반등은 이번 사건에 쓰지 않음). 10/23 종가 뒤 `py events.py all`로 "
             "다시 판정한다.")
    L.append("- 10/15 종가(514.71)는 10/16·19·20·21·22에 더 낮은 종가로 깨져 저점 후보가 아니다.\n")

    # 산출
    L.append("## 산출\n")
    L.append("- " + ", ".join(f"`{Path(p).name}`" for p in s["outputs"]))
    L.append(f"- 걸린 시간 {s['seconds']:.0f}초.")
    return "\n".join(L) + "\n"


TYPE_KO = es.TYPE_KO


if __name__ == "__main__":
    raise SystemExit(main())
