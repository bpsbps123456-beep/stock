"""부록 13 사건 연구 · 확정 사건만 모드(부록 14) · SK하이닉스 12% 한 조합만 실행.

event_study.py의 함수를 그대로 불러 쓴다(분석 로직·문턱·귀무·위약·누출 시험 불변). 이 파일이 더하는 것은
  - 조합 하나(SKHYNIX × 12%)만 고르기, 결과를 results/run_SKHYNIX_12.* 로 쓰기(다른 조합 실행과 산출이 섞이지 않게)
  - 하나 빼고 맞히기(LOEO)의 사건·가짜별 맞힘 수(loeo_ba와 같은 규칙을 다시 돌려 BA가 같은지 단언)
  - 지식 기간(라벨 모델이 알 수 있던 2026-07-01 전) 표시, 실시간 창 주의 표시
  - 지금 위치: 후보·근접 특징의 10/15~10/23 창 z가 그 오프셋의 사건·가짜 분포에서 차지하는 백분위(설명용)
실행: cd events/code && py -B run_combo_SKHYNIX_12.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import event_study as es  # noqa: E402

ASSET, THR = "SKHYNIX", "12"
COMBO = f"{ASSET}_{THR}"
RES = es.RESULTS
OUT_MD = RES / f"run_{COMBO}.md"
OUT_JSON = RES / f"run_{COMBO}.json"
FIGDIR = RES / f"run_{COMBO}_figs"
KNOW_END = np.datetime64("2026-07-01", "D")
NEAR_P = 0.10                         # '근접'(설명용으로만 적는 p 0.05~0.10)


def loeo_detail(v: np.ndarray, y: np.ndarray) -> dict:
    """loeo_ba와 같은 규칙(균형 정확도 최대, 같은 점수면 '위쪽이 사건'·낮은 문턱)으로 표본마다 빠진 예측과 접힌 방향."""
    v = np.asarray(v, float)
    y = np.asarray(y, bool)
    n = len(v)
    pred = np.zeros(n, bool)
    dirs = []
    for i in range(n):
        te = np.zeros(n, bool)
        te[i] = True
        vt, yt = v[~te], y[~te]
        s = np.sort(vt)
        cuts = [-np.inf] + [(s[k - 1] + s[k]) / 2 for k in range(1, len(s)) if s[k] > s[k - 1]] + [np.inf]
        best, choice = -np.inf, None
        P, N = yt.sum(), (~yt).sum()
        for d in ("up", "dn"):
            for c in cuts:
                pr = vt > c if d == "up" else ~(vt > c)
                ba = 0.5 * ((pr & yt).sum() / P + (~pr & ~yt).sum() / N)
                if ba > best + 1e-12:
                    best, choice = ba, (d, c)
        d, c = choice
        pred[i] = (v[i] > c) if d == "up" else not (v[i] > c)
        dirs.append(d)
    P, N = y.sum(), (~y).sum()
    tp, tn = int((pred & y).sum()), int((~pred & ~y).sum())
    # 전체 표본 문턱(설명용)
    s = np.sort(v)
    cuts = [-np.inf] + [(s[k - 1] + s[k]) / 2 for k in range(1, len(s)) if s[k] > s[k - 1]] + [np.inf]
    best, choice = -np.inf, None
    for d in ("up", "dn"):
        for c in cuts:
            pr = v > c if d == "up" else ~(v > c)
            ba = 0.5 * ((pr & y).sum() / P + (~pr & ~y).sum() / N)
            if ba > best + 1e-12:
                best, choice = ba, (d, c)
    return {"pred": pred, "ba": 0.5 * (tp / P + tn / N), "hit_events": tp, "n_events": int(P), "hit_fakes": tn,
            "n_fakes": int(N), "share_folds_up": float(np.mean([d == "up" for d in dirs])),
            "full_rule": {"direction": "z가 문턱보다 크면 사건" if choice[0] == "up" else "z가 문턱 이하이면 사건",
                          "threshold_z": float(choice[1]), "in_sample_ba": float(best)}}


def f2(x, nd=2):
    return es.fmt(x, nd)


def clean(o):
    """JSON용: NaN·inf → None, numpy 값 → 파이썬 값."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [clean(v) for v in o.tolist()]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (int, str)) or o is None:
        return o
    if isinstance(o, pd.Timestamp):
        return str(o)
    return str(o)


def main() -> int:
    t0 = datetime.now(es.KST).replace(microsecond=0)
    ev_path, feat_path = RES / "events.json", RES / "features_all.parquet"
    ev = json.loads(ev_path.read_text("utf-8"))
    ev2, mode = es.confirmed_only_events(ev, es.CONFIRMED_ASOF)       # 모든 자산 자료 끝·확정일·끝 극값 확인
    F, meta = es.load_features()
    prices = pd.read_csv(RES / "prices_close.csv")
    close = es.price_series(prices, ASSET)
    contrasts = [c for c in es.load_contrasts(ev2) if c.asset == ASSET and c.threshold == THR]
    assert [c.kind for c in contrasts] == ["bottom", "top"], [c.kind for c in contrasts]
    panel = es.build_panel(F, meta, "kr")
    assert panel.X.shape[0] == 285, panel.X.shape
    FIGDIR.mkdir(parents=True, exist_ok=True)
    tsev = ev["assets"][ASSET]["thresholds"][THR]
    out = {"made": t0.isoformat(), "combo": COMBO, "asset_ko": es.ASSET_KO[ASSET], "threshold_pct": int(THR),
           "mode": {k: v for k, v in mode.items() if k != "contrasts"} | {"this_combo": mode["contrasts"][COMBO]},
           "inputs": {"events_json_sha256": es.sha(ev_path), "features_all_sha256": es.sha(feat_path),
                      "prices_close_sha256": es.sha(RES / "prices_close.csv"),
                      "event_study_py_sha256": es.sha(HERE / "event_study.py")},
           "settings": {"n_null": es.N_NULL, "offsets_cls": list(es.CLS_OFFSETS), "lead": list(es.LEAD),
                        "z": f"직전 {es.Z_N}창(값 {es.Z_MIN}+)", "pct": f"직전 {es.P_N}창(값 {es.P_MIN}+)",
                        "placebo_pool": es.PLACEBO_POOL, "alpha": es.ALPHA, "min_true_fake": [es.MIN_TRUE, es.MIN_FAKE],
                        "grid_features": len(panel.feats),
                        "by_status": {s: sum(meta[f]["status"] == s for f in panel.feats) for s in ("통과", "설명용")}},
           "realtime_rule": "알아챔 = 오프셋 0(사건일 08:00 KST 마감 창)·+1(사건일 다음 날 07:59 마감 창)까지. 오프셋 +2 이후는 사후 설명용",
           "confirm_lag_td": {p["date"]: p["confirm_lag_td"] for p in tsev["pivots"]},
           "contrasts": {}}
    all_tests, all_lead, all_now, prof_rows = [], [], [], []
    for c in contrasts:
        r = es.run_contrast(c, panel, close, n_null=es.N_NULL)
        (FIGDIR / f"{c.name}.html").write_text(es.figure_html(r, panel, "확정 사건만 모드(10/22 종가). "), "utf-8")
        tb, lt, nw, cls = r["table"].copy(), r["lead"].copy(), r["now"].copy(), r["cls"]
        y, base = cls["y"], cls["base"]
        ev_pos = es.positions(panel, c.events["win_id"])
        fk_pos = es.positions(panel, c.fake_reps["win_id"])
        # 표본(창 있음) 목록: base = [사건 창 있음] + [가짜 대표 창 있음]
        ev_rows = [(d, w, "사건") for d, w, p in zip(c.events["date"], c.events["win_id"], ev_pos) if p >= 0]
        fk_rows = [(d, w, "가짜") for d, w, p in zip(c.fake_reps["date"], c.fake_reps["win_id"], fk_pos) if p >= 0]
        samples = ev_rows + fk_rows
        assert len(samples) == len(base)
        # LOEO 맞힘 수(시험 칸 전부)
        V = es.gather(panel.Z, base, es.CLS_OFFSETS)
        det_rows, per_sample = [], {}
        for oi, k in enumerate(es.CLS_OFFSETS):
            for fi, f in enumerate(panel.feats):
                v = V[:, oi, fi]
                ok = np.isfinite(v)
                row = tb[(tb["feature"] == f) & (tb["offset"] == k)].iloc[0]
                if row["status"] != "시험":
                    continue
                d = loeo_detail(v[ok], y[ok])
                assert abs(d["ba"] - row["ba"]) < 1e-9, (f, k, d["ba"], row["ba"])
                kno = np.array([np.datetime64(samples[i][0], "D") < KNOW_END for i in np.flatnonzero(ok)])
                det_rows.append({"feature": f, "offset": int(k), "hit_events": d["hit_events"], "hit_fakes": d["hit_fakes"],
                                 "share_folds_up": d["share_folds_up"], "full_rule": d["full_rule"]["direction"],
                                 "full_threshold_z": d["full_rule"]["threshold_z"],
                                 "n_events_knowledge": int((kno & y[ok]).sum()), "n_fakes_knowledge": int((kno & ~y[ok]).sum())})
                per_sample[(f, int(k))] = [{"date": samples[i][0], "win_id": samples[i][1], "label": samples[i][2],
                                            "z": float(v[i]), "loeo_correct": bool(pp == (samples[i][2] == "사건")),
                                            "knowledge_period": bool(np.datetime64(samples[i][0], "D") < KNOW_END)}
                                           for i, pp in zip(np.flatnonzero(ok), d["pred"])]
        D = pd.DataFrame(det_rows)
        tb = tb.merge(D, on=["feature", "offset"], how="left")
        tb["ko"] = tb["feature"].map(lambda f: meta[f]["ko"])
        tb["label_layer"] = tb["bundle"].isin(["예측 층", "감정 층"])
        # 칸 표(정렬: p, BA)
        tb_sorted = tb.sort_values(["p", "ba"], ascending=[True, False], na_position="last")
        cand = tb_sorted[tb_sorted["candidate"]]
        near = tb_sorted[(tb_sorted["p"] < NEAR_P) & ~tb_sorted["candidate"]]
        # 앞섬
        lt["ko"] = lt["feature"].map(lambda f: meta[f]["ko"])
        lt_sorted = lt.sort_values("p", na_position="last")
        lead_cand = lt_sorted[lt_sorted["candidate"]]
        lead_p05 = lt_sorted[lt_sorted["p"] < es.ALPHA]
        # 지금 위치: 후보·근접(p<0.10) 칸의 그 오프셋 사건·가짜 분포 안 백분위
        now_rows = []
        sel = np.flatnonzero((panel.D >= np.datetime64("2026-10-15", "D")) & (panel.D <= np.datetime64("2026-10-23", "D")))
        focus = pd.concat([cand, near]).drop_duplicates(["feature", "offset"])
        for _, q in focus.iterrows():
            fi = panel.feats.index(q["feature"])
            k = int(q["offset"])
            ref_e = es.gather(panel.Z, ev_pos, [k])[:, 0, fi]
            ref_f = es.gather(panel.Z, fk_pos, [k])[:, 0, fi]
            ref_e, ref_f = ref_e[np.isfinite(ref_e)], ref_f[np.isfinite(ref_f)]
            for t in sel:
                z = panel.Z[t, fi]

                def pr(ref):
                    return float(((ref < z).sum() + 0.5 * (ref == z).sum()) / len(ref)) if (len(ref) and np.isfinite(z)) else np.nan
                side = None
                if np.isfinite(z):
                    up = q["full_rule"] == "z가 문턱보다 크면 사건"
                    side = "사건 쪽" if ((z > q["full_threshold_z"]) == up) else "가짜 쪽"
                now_rows.append({"feature": q["feature"], "offset": k, "candidate": bool(q["candidate"]),
                                 "win_id": panel.win_id[t], "D": str(panel.D[t]),
                                 "as_offset_of": f"{'그 날' if k == 0 else '전 거래일'} 사건 가정", "z": z,
                                 "pct60": panel.PCT[t, fi], "pct_in_events_same_offset": pr(ref_e),
                                 "pct_in_fakes_same_offset": pr(ref_f), "full_rule_side": side,
                                 "realtime_window": bool(panel.realtime[t]),
                                 "realtime_caution": (not meta[q["feature"]]["realtime_comparable"]) or
                                                     ("실시간" in meta[q["feature"]].get("note", ""))})
        NW = pd.DataFrame(now_rows)
        # 사건·가짜 목록
        lists = {"events": [{"date": d, "win_id": w, "has_window": bool(p >= 0),
                             "confirm_lag_td": next((x["confirm_lag_td"] for x in tsev["pivots"] if x["date"] == d), None),
                             "z_rows_before": int(panel.pos.get(w, -1))}
                            for d, w, p in zip(c.events["date"], c.events["win_id"], ev_pos)],
                 "fake_reps": [{"date": d, "win_id": w, "run": int(rn), "has_window": bool(p >= 0)}
                               for d, w, rn, p in zip(c.fake_reps["date"], c.fake_reps["win_id"], c.fake_reps["run"], fk_pos)],
                 "n_fakes_all": int(len(c.fakes_all))}
        # 가격 경로(사건 평균)
        pp = r["price_paths"]
        with np.errstate(invalid="ignore"):
            price_mean = {nm: [None if not np.isfinite(x) else round(float(x) * 100, 2) for x in np.nanmean(a, 0)]
                          for nm, a in pp.items() if len(a)}
        prof = r["profiles"]
        for fi, f in enumerate(panel.feats):
            for j, k in enumerate(es.OFFSETS):
                prof_rows.append({"kind": c.kind, "feature": f, "offset": int(k), "ev_mean": prof["ev_mean"][j, fi],
                                  "ev_n": int(prof["ev_n"][j, fi]), "fk_mean": prof["fk_mean"][j, fi],
                                  "pl_mean": prof["pl_mean"][j, fi], "pl_lo": prof["pl_lo"][j, fi],
                                  "pl_hi": prof["pl_hi"][j, fi]})
        keep = ["feature", "ko", "bundle", "status_feature", "offset", "n_events", "n_fakes", "ba", "p", "hit_events",
                "hit_fakes", "direction", "mean_z_events", "mean_z_fakes", "auc_in_sample", "null_ba_q95",
                "ba_all_fake_days_run_groups", "share_folds_up", "full_rule", "full_threshold_z", "n_events_knowledge",
                "n_fakes_knowledge", "label_layer", "leak_suspect", "candidate"]
        res = {"n": r["n"], "samples_with_window": {"events": int(y.sum()), "fakes": int((~y).sum())}, "lists": lists,
               "family_cls": r["family"], "family_lead": r["lead_family"], "leak": r["leak"],
               "outside_band": r["outside_band"], "placebo": r["placebo"], "price_mean_path_pct": price_mean,
               "candidates_cls": cand.reindex(columns=keep).to_dict("records"),
               "near_cls_p_lt_0_10": near.reindex(columns=keep).to_dict("records"),
               "per_sample_candidates": {f"{f}@{k:+d}": per_sample[(f, k)] for f, k in
                                         zip(focus["feature"], focus["offset"].astype(int))},
               "lead_candidates": lead_cand.to_dict("records"), "lead_p_lt_0_05": lead_p05.to_dict("records"),
               "now_focus": NW.to_dict("records"),
               "figure": (FIGDIR / f"{c.name}.html").relative_to(RES).as_posix()}
        out["contrasts"][c.kind] = res
        all_tests.append(tb.reindex(columns=keep).assign(kind=c.kind))
        all_lead.append(lt.assign(kind=c.kind))
        if len(nw):
            all_now.append(nw.assign(kind=c.kind))
        out.setdefault("_tb", {})[c.kind] = tb
        out.setdefault("_lt", {})[c.kind] = lt
        out.setdefault("_nw", {})[c.kind] = NW
    tbs, lts, nws = out.pop("_tb"), out.pop("_lt"), out.pop("_nw")
    # ---- 설명용 덧붙임(사전 등록 판정에는 넣지 않음)
    extra = {}
    # (1) 전방 확인: 2026-07-01 전 표본으로 문턱·방향을 정하고 7/1 뒤 표본에 적용(연구 보고 규칙의 연도 내 전방 검증, 표본 아주 작음)
    fwd = []
    for kind, res in out["contrasts"].items():
        tb = tbs[kind]
        for _, q in tb[tb["candidate"] | (tb["p"] < NEAR_P)].iterrows():
            rows = res["per_sample_candidates"].get(f"{q['feature']}@{int(q['offset']):+d}")
            if not rows:
                continue
            S = pd.DataFrame(rows)
            S["y"] = S["label"].eq("사건")
            tr, te = S[S["date"] < "2026-07-01"], S[S["date"] >= "2026-07-01"]
            if tr["y"].sum() < 2 or (~tr["y"]).sum() < 2 or te.empty:
                continue
            d = loeo_detail(tr["z"].to_numpy(), tr["y"].to_numpy())
            up = d["full_rule"]["direction"] == "z가 문턱보다 크면 사건"
            pr = (te["z"] > d["full_rule"]["threshold_z"]) == up
            fwd.append({"kind": kind, "feature": q["feature"], "offset": int(q["offset"]), "candidate": bool(q["candidate"]),
                        "train_n": [int(tr["y"].sum()), int((~tr["y"]).sum())], "train_rule": d["full_rule"],
                        "test_events_hit": f"{int((pr & te['y']).sum())}/{int(te['y'].sum())}",
                        "test_fakes_hit": f"{int((~pr & ~te['y']).sum())}/{int((~te['y']).sum())}",
                        "test_dates_events": te.loc[te["y"], "date"].tolist()})
    extra["forward_2026_07"] = fwd
    # (2) 지금 위치의 앞섬: 10/15~10/23 각 창을 '그 날 사건'으로 둘 때 −5~−1창이 직전 60창 상위·하위 10%에 닿았나
    nl = []
    sel = np.flatnonzero((panel.D >= np.datetime64("2026-10-15", "D")) & (panel.D <= np.datetime64("2026-10-23", "D")))
    for kind in out["contrasts"]:
        lt = lts[kind]
        for _, q in lt[lt["candidate"] | (lt["p"] < es.ALPHA)].iterrows():
            fi = panel.feats.index(q["feature"])
            hi = q["side"] == "상위 10%"
            for t in sel:
                P = panel.PCT[max(0, t + es.LEAD[0]):t + es.LEAD[1] + 1, fi]
                ok = np.isfinite(P)
                hit = (bool((P[ok] >= es.P_HI).any()) if hi else bool((P[ok] <= es.P_LO).any())) if ok.sum() >= es.LEAD_MIN else None
                nl.append({"kind": kind, "feature": q["feature"], "side": q["side"], "candidate": bool(q["candidate"]),
                           "as_event_day": str(panel.D[t]), "lead_windows": list(panel.win_id[max(0, t - 5):t]),
                           "pct60_lead_windows": [None if not np.isfinite(x) else round(float(x), 3) for x in P],
                           "hit": hit, "rate_events": q["rate_events"], "rate_fakes": q["rate_fakes"],
                           "rate_placebo_mean": q.get("rate_placebo_mean")})
    extra["now_lead"] = nl
    # (3) 같은 칸의 8% 결과(강건 후보 판단 참고; 8% 정식 산출은 그 조합 실행이 낸다 — 같은 코드·같은 시드라 값이 같다)
    c8 = [c for c in es.load_contrasts(ev2) if c.asset == ASSET and c.threshold == "8"]
    x8 = []
    for c in c8:
        r8 = es.run_contrast(c, panel, close, n_null=es.N_NULL, with_now=False)
        t8, l8 = r8["table"], r8["lead"]
        tb = tbs[c.kind]
        for _, q in tb[tb["candidate"]].iterrows():
            m = t8[(t8["feature"] == q["feature"]) & (t8["offset"] == q["offset"])].iloc[0]
            x8.append({"kind": c.kind, "type": "가르기", "feature": q["feature"], "offset": int(q["offset"]),
                       "ba_8": m.get("ba"), "p_8": m.get("p"), "direction_8": m.get("direction"),
                       "direction_12": q["direction"], "candidate_8": bool(m["candidate"]),
                       "n_8": [int(m["n_events"]), int(m["n_fakes"])], "leak_8": bool(r8["leak"]["flag"])})
        lt = lts[c.kind]
        for _, q in lt[lt["candidate"]].iterrows():
            m = l8[(l8["feature"] == q["feature"]) & (l8["side"] == q["side"])].iloc[0]
            x8.append({"kind": c.kind, "type": "앞섬", "feature": q["feature"], "side": q["side"],
                       "rate_events_8": m.get("rate_events"), "rate_fakes_8": m.get("rate_fakes"), "p_8": m.get("p"),
                       "rate_placebo_q975_8": m.get("rate_placebo_q975"), "candidate_8": bool(m["candidate"]),
                       "n_8": [int(m["n_events"]), int(m["n_fakes"])]})
        extra.setdefault("family_8", {})[c.kind] = {"cls": r8["family"]["통과"], "lead": r8["lead_family"]["통과"],
                                                    "n": r8["n"]}
    extra["same_cells_at_8pct"] = x8
    out["extra_descriptive"] = extra
    pd.concat(all_tests).to_csv(RES / f"run_{COMBO}_tests.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_lead).to_csv(RES / f"run_{COMBO}_lead.csv", index=False, encoding="utf-8-sig")
    if all_now:
        pd.concat(all_now).to_csv(RES / f"run_{COMBO}_now.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prof_rows).to_csv(RES / f"run_{COMBO}_profiles.csv", index=False, encoding="utf-8-sig")
    out["seconds"] = (datetime.now(es.KST) - t0).total_seconds()
    OUT_JSON.write_text(json.dumps(clean(out), ensure_ascii=False, indent=1, allow_nan=False), "utf-8")
    OUT_MD.write_text(write_md(clean(out), meta), "utf-8")
    print(json.dumps({k: {"n": v["n"], "fam": v["family_cls"]["통과"], "lead": v["family_lead"]["통과"],
                          "leak": v["leak"]["flag"], "cand": [(x["feature"], x["offset"], round(x["ba"], 3), x["p"])
                                                             for x in v["candidates_cls"]]}
                      for k, v in out["contrasts"].items()}, ensure_ascii=False, indent=1, default=str))
    print("seconds", out["seconds"])
    return 0


KIND_KO = {"bottom": "바닥", "top": "고점"}


def _p(x):
    return "—" if x is None else f"{x:.3f}"


def _n(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def write_md(o: dict, meta: dict) -> str:
    C, X = o["contrasts"], o["extra_descriptive"]
    tc = o["mode"]["this_combo"]
    L = [f"# SK하이닉스 12% 사건 연구 (부록 13 · 확정 사건만 모드, 부록 14)\n",
         f"> {o['made']}. 탐색 연구라 결과는 '후보'다. 매매 규칙이 아니고, '지금이 바닥/고점'이라고 말하지 않는다. 후보는 2026-10-24 뒤 새로 "
         "확정되는 고점·저점으로만 확인한다.",
         f"> 입력: `events.json` sha256 {o['inputs']['events_json_sha256'][:12]}…(2026-10-22 종가 판정), `features_all.parquet` "
         f"{o['inputs']['features_all_sha256'][:12]}…(스냅샷 2026-10-23 07:59, W_KR 285창), `prices_close.csv` "
         f"{o['inputs']['prices_close_sha256'][:12]}…, 분석 코드 `event_study.py` {o['inputs']['event_study_py_sha256'][:12]}…(고치지 않음, "
         "이 조합 실행기 `run_combo_SKHYNIX_12.py`가 그 함수를 그대로 부른다).",
         f"> 설정: 특징 {o['settings']['grid_features']}개(검수 통과 {o['settings']['by_status']['통과']} · 설명용 "
         f"{o['settings']['by_status']['설명용']}) × 오프셋 0·+1, 표지 섞기 {o['settings']['n_null']}번, 위약 = 사건마다 직전 5·20거래일 "
         f"수익률이 가장 비슷한 날 {o['settings']['placebo_pool']}일, 앞섬 = 오프셋 −5~−1에서 직전 60창 상위·하위 10%.",
         "> 알아챔의 시간 한계: 오프셋 0 창은 사건일 08:00 KST에, +1 창은 사건일 다음 날 07:59에 닫힌다. 이 두 창과 그 앞 창만 '알아챔' "
         "근거로 쓴다. 극값 확정은 이 조합에서 1~14거래일 늦다(아래 1절). 확정 뒤 정보로 극값을 알아챘다고 하지 않는다.\n",
         "## 0. 결론\n",
         "| 묶음 | 사건(값 있음) | 가짜 묶음 대표(창 있음) | 칸 수(검수 통과) | p<0.05 관측 | 우연 기대 | 귀무 95% | 묶음 p | 후보 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for kind in ("bottom", "top"):
        r = C[kind]
        fc, fl = r["family_cls"]["통과"], r["family_lead"]["통과"]
        cs = ", ".join(f"{x['feature']}@{x['offset']:+d}" for x in r["candidates_cls"]) or "없음"
        ls = ", ".join(f"{x['feature']}({x['side']})" for x in r["lead_candidates"]) or "없음"
        ne = r["candidates_cls"][0]["n_events"] if r["candidates_cls"] else r["samples_with_window"]["events"]
        L.append(f"| {KIND_KO[kind]} 가르기 | {ne} | {r['samples_with_window']['fakes']} | {fc['n_tests']} | {fc['observed']} | "
                 f"{_n(fc['expected_by_chance'])} | {_n(fc['null_q95'], 0)} | {_p(fc['p_family'])} | {cs} |")
        L.append(f"| {KIND_KO[kind]} 앞섬 | | | {fl['n_tests']} | {fl['observed']} | {_n(fl['expected_by_chance'])} | "
                 f"{_n(fl['null_q95'], 0)} | {_p(fl['p_family'])} | {ls} |")
    rob = [x for x in X["same_cells_at_8pct"] if x["candidate_8"]]
    gone = [x for x in X["same_cells_at_8pct"] if not x["candidate_8"]]
    L += ["",
          "- 네 묶음 모두 p<0.05 칸 수가 우연 기대 개수와 구별되지 않는다(묶음 p 0.34~0.85, 관측이 귀무 95%를 넘은 묶음 없음). "
          "그래서 아래 후보 하나하나는 '여럿 시험하면 이 정도는 우연히 나온다' 수준을 벗어났다는 근거가 없다.",
          "- 후보 조건(부록 13): 섞기 p<0.05 + 검수 통과 특징 + 조회·추천이면 누출 시험 통과. 앞섬은 여기에 위약 97.5% 분위 위.",
          "- 8%(주판)의 같은 칸: " + ("; ".join(
              f"{KIND_KO[x['kind']]} {x['type']} {x['feature']}" + (f"@{x['offset']:+d}" if 'offset' in x else f"({x['side']})")
              + f" 8%에서도 후보(p {_p(x['p_8'])})" for x in rob) or "없음")
          + ". 8%에서 사라진 것: " + ("; ".join(
              f"{KIND_KO[x['kind']]} {x['feature']}" + (f"@{x['offset']:+d}" if 'offset' in x else f"({x['side']})")
              + f"(8% p {_p(x['p_8'])})" for x in gone) or "없음")
          + ". 12% 꺾임점 18개는 모두 8% 꺾임점(36개)에 들어 있어 8%는 독립 확인이 아니다.",
          "- 두 문턱 모두 후보인 칸은 고점 가르기 vw_ln_b_median@+1과 바닥 앞섬 두 개다. vw_ln_b_median은 위약 띠 안이고 조회 누적값이라 "
          "그 날 아침에 같은 값을 잴 수 없다. 바닥 앞섬 두 개는 가짜·위약보다 높지만 12% 묶음 전체가 우연 수준이다.",
          "- 감정 층(설명용, 검수 실패)과 다른의견 비율(설명용, 누출 있음)은 p가 작아도 후보로 올리지 않았다.\n"]
    # 1. 사건
    L += ["## 1. 사건과 가짜(가격만으로 정함, 10/22 종가)\n",
          f"- 확정 꺾임점만 썼다: 저점 {tc['events_used']['low']}개, 고점 {tc['events_used']['high']}개, 마지막 확정일 {tc['last_confirm_date']}.",
          f"- 뺀 끝 극값(미확정): 저점 {tc['tail_excluded']['date']}(종가 {tc['tail_excluded']['close']:,.0f}원, 창 "
          f"{tc['tail_excluded']['win_id']}). 12% 확정에 필요한 종가 {tc['tail_excluded']['confirm_level']:,.0f}원 이상, 10/22 종가 "
          f"{tc['tail_excluded']['last_close']:,.0f}원(같은 종가면 처음 날 규칙으로 10/20). 10/23 종가 뒤 다시 판정한다.",
          f"- 뺀 보류 가짜 바닥(끝 극값 ±3거래일): {', '.join(tc['held_fakes_excluded']['fake_bottoms'])}. 보류 가짜 고점 없음.",
          "- 2025-08-21 저점은 자료 첫 창이라 직전 20창 z가 없다 → 바닥 가르기·앞섬의 실제 사건 수는 8개다.\n",
          "| 종류 | 사건일(창) | 확정까지 거래일 |", "|---|---|---|"]
    for kind in ("bottom", "top"):
        for e in C[kind]["lists"]["events"]:
            L.append(f"| {KIND_KO[kind]} | {e['date']} ({e['win_id']}) | {e['confirm_lag_td']} |")
    for kind in ("bottom", "top"):
        fr = C[kind]["lists"]["fake_reps"]
        L.append(f"\n- 가짜 {KIND_KO[kind]} 묶음 대표 {len(fr)}개(가짜 날 전체 {C[kind]['lists']['n_fakes_all']}개): "
                 + ", ".join(f["date"] for f in fr))
    L.append("- 가짜 바닥 2026-10-14(묶음 13 대표)는 실시간 수집 창이라 조회·추천(추천 지니 빼고)이 결측이다.\n")
    # 2. 가르기
    L += ["## 2. 가르기(분석 2): 하나 빼고 맞히기 + 섞기 1,000번\n",
          "| 종류 | 특징@오프셋 | 묶음 | 지위 | 사건/가짜 | BA(하나 빼고) | 맞힘 사건 | 맞힘 가짜 | 섞기 p | 방향(z 평균 사건 vs 가짜) | "
          "가짜 날 전체 BA | 위약 띠 | 후보 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for kind in ("bottom", "top"):
        r = C[kind]
        rows = r["candidates_cls"] + r["near_cls_p_lt_0_10"]
        prof_out = {k: set(v["features"]) for k, v in r["outside_band"].items()}
        for x in sorted(rows, key=lambda x: x["p"]):
            band = "밖" if x["feature"] in prof_out.get(str(x["offset"]), set()) else "안"
            L.append(f"| {KIND_KO[kind]} | {x['feature']}@{x['offset']:+d} ({x['ko'][:18]}) | {x['bundle']} | {x['status_feature']} | "
                     f"{x['n_events']}/{x['n_fakes']} | {_n(x['ba'])} | {x['hit_events']}/{x['n_events']} | {x['hit_fakes']}/{x['n_fakes']} | "
                     f"{_p(x['p'])} | {x['direction']} ({_n(x['mean_z_events'])} vs {_n(x['mean_z_fakes'])}) | "
                     f"{_n(x.get('ba_all_fake_days_run_groups'))} | {band} | {'후보' if x['candidate'] else ''} |")
    L += ["\n- 표에는 후보와 p<0.10 칸만 적었다. 전 칸은 `run_SKHYNIX_12_tests.csv`(바닥·고점 각 92칸).",
          "- '위약 띠'는 분석 1의 오프셋 같은 칸에서 사건 평균 z가 위약 날(가격 움직임이 비슷한 날) 95% 범위 밖인지다. 가짜 사건과 "
          "갈려도 위약 띠 안이면 '사건 그 자체'보다 '그 무렵 가격 움직임'과 겹치는 신호일 수 있다.",
          "- 후보별 사건 값(z, 하나 빼고 맞힘)은 `run_SKHYNIX_12.json`의 `per_sample_candidates`에 있다.\n"]
    notes = {
        ("bottom", "tp_LS_A", 0): "평균 차이는 작다(0.17 vs 0.51, 표본 안 AUC 0.39). 문턱 z ≤ −1.06에 바닥 8개 중 4개(11/25·3/31·6/23·8/10)가 "
                                  "들고 가짜 13개는 하나도 안 든다(가장 가까운 가짜 11/14가 −1.02). 나머지 바닥 4개는 z +1.1~+2.4로 오히려 높다. "
                                  "문턱 바로 옆에 가짜가 있어 깨지기 쉬운 갈림이다.",
        ("top", "tp_ANG_A", 0): "고점 9개 중 7개가 z > 0.11, 가짜 12개 중 10개가 그 아래. 위약 띠(−1.14~0.50)도 사건 평균 0.54가 살짝 넘는다. "
                                "다만 z가 가장 높은 두 고점(7/15 +2.10, 6/25 +1.57)은 확정 저점 2거래일 뒤의 반등 고점이라, 직전 급락의 분노가 "
                                "남은 것일 수 있다.",
        ("top", "vw_ln_b_median", 1): "고점 9개 중 8개가 z ≤ −0.22(같은 창 마감 전 댓글에 비해 조회가 적다). 누출이면 사건 쪽이 커져야 하는데 "
                                      "반대 방향이라 누출 시험은 통과다. 그러나 사건 평균 −0.71은 위약 띠(−1.07~0.71) 안이고, 조회는 수집 때 쌓인 "
                                      "누적값이라 그 날 아침에 같은 값을 볼 수 없다(실시간 수집 창은 10/14 뒤 8창뿐이고 그 안에 확정 사건이 없어 검증할 수 없다).",
        ("top", "act_new30_share", 1): "고점 9개 중 7개가 z ≤ 0.02(새 작성자 몫이 평소보다 낮다). 가짜는 10개만 값이 있다(30일 이력이 필요). "
                                       "사건 평균 −0.30은 위약 띠(−0.74~1.31) 안이다. 9/21 고점은 z +2.28로 반대.",
    }
    L.append("### 후보 해설\n")
    for kind in ("bottom", "top"):
        for x in C[kind]["candidates_cls"]:
            L.append(f"- {KIND_KO[kind]} {x['feature']}@{x['offset']:+d}: " + notes.get((kind, x["feature"], x["offset"]), ""))
    L.append("")
    # 3. 사건 전후
    L += ["## 3. 사건 전후(분석 1, 설명용)\n",
          "| 종류 | 오프셋 0: 위약 띠 밖 / 특징 | 오프셋 +1: 위약 띠 밖 / 특징 | 서로 독립이면 기대 | 그림 |", "|---|---|---|---|---|"]
    for kind in ("bottom", "top"):
        ob = C[kind]["outside_band"]
        L.append(f"| {KIND_KO[kind]} | {ob['0']['outside_band']} / {ob['0']['n_features']} ({', '.join(ob['0']['features']) or '—'}) | "
                 f"{ob['1']['outside_band']} / {ob['1']['n_features']} ({', '.join(ob['1']['features']) or '—'}) | "
                 f"{ob['0']['expected_if_independent']} | `{C[kind]['figure']}` |")
    L.append("\n- 가격 평균 경로(사건일 종가 대비 %, 오프셋 −2·−1·0·+1·+2):")
    for kind in ("bottom", "top"):
        pp = C[kind]["price_mean_path_pct"]
        L.append(f"  - {KIND_KO[kind]}: 사건 " + " / ".join(_n(v, 1) for v in pp["events"][8:13]) + "; 가짜 "
                 + " / ".join(_n(v, 1) for v in pp["fakes"][8:13]) + "; 위약 " + " / ".join(_n(v, 1) for v in pp["placebo"][8:13]))
    L.append("- 그림 자료(오프셋 −10~+10, 특징마다 사건 평균·가짜 평균·위약 띠): `run_SKHYNIX_12_profiles.csv`. 오프셋 +2 이후는 사후 설명용이다.\n")
    # 4. 앞섬
    L += ["## 4. 앞섬(분석 3): 사건 전 1~5창에 직전 60창 상위·하위 10% 닿은 비율\n",
          "| 종류 | 특징(쪽) | 지위 | 사건 | 가짜 | 위약 평균 | 위약 97.5% 분위 | 섞기 p | 위약 대비 p | 후보 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for kind in ("bottom", "top"):
        for x in C[kind]["lead_p_lt_0_05"]:
            L.append(f"| {KIND_KO[kind]} | {x['feature']}({x['side']}) | {x['status_feature']} | {_n(x['rate_events'])} ({x['n_events']}) | "
                     f"{_n(x['rate_fakes'])} ({x['n_fakes']}) | {_n(x.get('rate_placebo_mean'))} | {_n(x.get('rate_placebo_q975'))} | "
                     f"{_p(x['p'])} | {_p(x.get('p_vs_placebo'))} | {'후보' if x['candidate'] else ''} |")
    L += ["\n- 고점 act_new30_share(하위 10%)는 사건 비율 0.625가 위약 97.5% 분위 0.625와 같아(넘지 않음) 후보가 아니다.",
          "- 바닥 앞섬 두 후보는 바닥 전 며칠 동안 반대매매·마진콜 언급 작성자와 저장 글이 크게 는다는 뜻이다. 가짜 바닥(하루 −5% 이하 또는 "
          "5일 −8% 이하)도 급락 뒤라 같은 쪽으로 움직이고(가짜 0.25·0.42), 위약 날도 0.29·0.34다.\n"]
    # 5. 누출
    L += ["## 5. 조회·추천 누출 시험((b) 조회 ÷ 마감 전 댓글, z 중앙값)\n",
          "| 종류 | 오프셋 | 사건 중앙(개수) | 가짜 중앙(개수) | 사건이 큰가 | 단측 Mann–Whitney p | 판정 |", "|---|---|---|---|---|---|---|"]
    for kind in ("bottom", "top"):
        lk = C[kind]["leak"]
        for k, v in lk["by_offset"].items():
            L.append(f"| {KIND_KO[kind]} | {int(k):+d} | {_n(v.get('median_events'))} ({v['n_events']}) | {_n(v.get('median_fakes'))} "
                     f"({v['n_fakes']}) | {'예' if v.get('events_greater') else '아니오'} | {_p(v.get('mannwhitney_p_greater'))} | "
                     f"{'의심' if lk['flag'] else '통과'} |")
    L += ["\n- 두 종류 모두 사건 쪽 (b)가 가짜보다 작아 '사건 뒤 재조명된 주목' 신호가 없다 → 조회·추천 묶음은 누출 의심이 아니다.",
          "- 그래도 (b)·조회 지니·상위 몫은 소급 창(실시간 글 5% 미만)끼리만 비교한다. 10/13 뒤 창은 결측이다.\n"]
    # 6. 지금 위치
    L += ["## 6. 지금 위치(설명용, 10/15~10/23 창)\n",
          "- 끝 극값 10/20 저점은 미확정이다. 아래는 최근 창 값이 확정 사건·가짜의 같은 오프셋 분포에서 어디쯤인지일 뿐, 바닥·고점 판단이 아니다.",
          "- 10/14 뒤 창은 실시간 수집 창이다. 저장 글 수는 소급 창이 지운 글(결번율 약 8%)만큼 작으므로 실시간 창 값이 부풀 수 있고, 댓글 양은 "
          "덜 붙은 채 받힌다. 조회 (b)는 실시간 창에서 결측이다.\n",
          "| 종류 | 후보 특징@오프셋 | 창 | z | 직전 60창 백분위 | 사건 분포 안 | 가짜 분포 안 |", "|---|---|---|---|---|---|---|"]
    for kind in ("bottom", "top"):
        for x in C[kind]["now_focus"]:
            if not x["candidate"] or x["z"] is None:
                continue
            L.append(f"| {KIND_KO[kind]} | {x['feature']}@{x['offset']:+d} | {x['win_id']} | {_n(x['z'])} | {_n(x['pct60'])} | "
                     f"{_n(x['pct_in_events_same_offset'])} | {_n(x['pct_in_fakes_same_offset'])} |")
    L += ["\n- 고점 vw_ln_b_median@+1은 10/15~10/23 창이 모두 실시간 수집 창이라 결측이다(소급 창끼리만 비교).",
          "\n앞섬 후보의 최근 값(각 날을 사건일로 둘 때 앞 5창의 직전 60창 백분위):\n",
          "| 특징(쪽) | 사건일로 둔 날 | 앞 5창 백분위 | 닿음 | 확정 사건 / 가짜 / 위약 닿은 비율 |", "|---|---|---|---|---|"]
    for x in X["now_lead"]:
        if not x["candidate"]:
            continue
        L.append(f"| {x['feature']}({x['side']}) | {x['as_event_day']} | {', '.join(_n(v) for v in x['pct60_lead_windows'])} | "
                 f"{'예' if x['hit'] else '아니오'} | {_n(x['rate_events'])} / {_n(x['rate_fakes'])} / {_n(x.get('rate_placebo_mean'))} |")
    L += ["\n- 반대매매·마진콜 작성자 몫은 K10/14~K10/19 네 창이 직전 60창 상위 10%였다(K10/16 0.97). 같은 모양은 확정 바닥 8개 중 6개 "
          "앞에 있었지만 가짜 바닥 12개 중 3개, 가격이 비슷한 날의 29%에도 있었고, 바닥 앞섬 묶음 전체가 우연과 구별되지 않는다. "
          "그래서 이 값은 '지금이 바닥'의 근거가 아니다.\n"]
    # 7. 덧붙임
    L += ["## 7. 설명용 덧붙임(사전 등록 판정 밖)\n",
          "### 7-1. 연도 내 전방 확인(2026-07-01 전 표본으로 문턱 → 7/1 뒤 표본에 적용)\n",
          "| 종류 | 특징@오프셋 | 후보 | 앞 구간 사건/가짜 | 앞 구간 규칙 | 뒤 구간 맞힘 사건 | 뒤 구간 맞힘 가짜 |", "|---|---|---|---|---|---|---|"]
    for x in X["forward_2026_07"]:
        L.append(f"| {KIND_KO[x['kind']]} | {x['feature']}@{x['offset']:+d} | {'후보' if x['candidate'] else ''} | "
                 f"{x['train_n'][0]}/{x['train_n'][1]} | {x['train_rule']['direction']} {x['train_rule']['threshold_z']:+.2f} | "
                 f"{x['test_events_hit']} | {x['test_fakes_hit']} |")
    L += ["\n- 뒤 구간 사건이 3개, 가짜가 2~7개라 우연과 가를 힘이 없다. 고점 tp_ANG_A는 뒤 구간 고점 3/3을 맞혔지만 가짜 2개를 모두 "
          "고점으로 불렀다(앞 구간 문턱 −0.75가 낮음). 바닥 tp_LS_A는 뒤 구간 바닥 1/3.",
          "### 7-2. 같은 칸의 8% 결과(강건 후보 판단)\n",
          "| 종류 | 분석 | 특징 | 12% | 8% | 8% 후보 |", "|---|---|---|---|---|---|"]
    for x in X["same_cells_at_8pct"]:
        if x["type"] == "가르기":
            c12 = next(y for y in C[x["kind"]]["candidates_cls"] if y["feature"] == x["feature"] and y["offset"] == x["offset"])
            L.append(f"| {KIND_KO[x['kind']]} | 가르기 | {x['feature']}@{x['offset']:+d} | BA {_n(c12['ba'])}, p {_p(c12['p'])} | "
                     f"BA {_n(x['ba_8'])}, p {_p(x['p_8'])}, 사건/가짜 {x['n_8'][0]}/{x['n_8'][1]}"
                     f"{', 누출 의심' if x['leak_8'] else ''} | {'예' if x['candidate_8'] else '아니오'} |")
        else:
            c12 = next(y for y in C[x["kind"]]["lead_candidates"] if y["feature"] == x["feature"] and y["side"] == x["side"])
            L.append(f"| {KIND_KO[x['kind']]} | 앞섬 | {x['feature']}({x['side']}) | 사건 {_n(c12['rate_events'])} vs 가짜 "
                     f"{_n(c12['rate_fakes'])}, p {_p(c12['p'])} | 사건 {_n(x['rate_events_8'])} vs 가짜 {_n(x['rate_fakes_8'])}, "
                     f"p {_p(x['p_8'])}, 사건/가짜 {x['n_8'][0]}/{x['n_8'][1]} | {'예' if x['candidate_8'] else '아니오'} |")
    f8 = X["family_8"]
    L += ["\n- 8% 묶음(검수 통과): 바닥 가르기 관측 {}/우연 {}, 바닥 앞섬 {}/{}(귀무 95% {}, 묶음 p {}), 고점 가르기 {}/{}, 고점 앞섬 {}/{}. "
          "8% 바닥은 가짜 대표가 {}개(앞섬 값 4개)뿐이다. 8% 정식 산출은 그 조합 실행이 낸다(같은 코드·같은 시드라 값이 같다).\n".format(
              f8["bottom"]["cls"]["observed"], _n(f8["bottom"]["cls"]["expected_by_chance"]),
              f8["bottom"]["lead"]["observed"], _n(f8["bottom"]["lead"]["expected_by_chance"]),
              _n(f8["bottom"]["lead"]["null_q95"], 0), _p(f8["bottom"]["lead"]["p_family"]),
              f8["top"]["cls"]["observed"], _n(f8["top"]["cls"]["expected_by_chance"]),
              f8["top"]["lead"]["observed"], _n(f8["top"]["lead"]["expected_by_chance"]), f8["bottom"]["n"]["fake_reps"])]
    # 8. 한계
    L += ["## 8. 탐색 범위와 한계\n",
          "- 이 조합에서 시험한 칸: 가르기 46특징 × 2오프셋 × 2종류 = 184칸(검수 통과 96), 앞섬 46 × 2쪽 × 2종류 = 184칸(통과 96). "
          "부록 13 전체는 3자산 × 2문턱이라 이 6배다. 많이 시험한 만큼 p<0.05가 몇 개 나오는 것은 정상이다(위 우연 기대).",
          "- 표본이 작다: 바닥 8·고점 9 대 가짜 대표 13~14. 시운전(무작위 날짜 + 주입 신호)에서 사건 8개 안팎이면 1 SD 차이도 절반 넘게 놓쳤다. "
          "'후보 없음'도 '신호 없음'의 증거가 아니다.",
          "- 예측 층·감정 층 라벨은 2026-07-01 전 창이 라벨 모델의 지식 기간이다(바닥 8개 중 5개, 고점 9개 중 6개가 그 안). 이번 후보에는 "
          "라벨 층이 없다.",
          "- 2026년 장세 기준: 사건 대부분이 2026년(바닥 8개 중 7개, 고점 9개 중 8개)이다. 2025년 사건은 바닥 11/25, 고점 11/03뿐이다.",
          "- 다음 확인: 2026-10-24 뒤 새로 확정되는 하이닉스 고점·저점(10/20 끝 극값 포함, 확정되면)에서 위 후보 칸이 같은 방향인지 본다.\n",
          "## 산출\n",
          "- `run_SKHYNIX_12.json`(전 수치), `run_SKHYNIX_12_tests.csv`(가르기 전 칸), `run_SKHYNIX_12_lead.csv`(앞섬 전 칸), "
          "`run_SKHYNIX_12_now.csv`(지금 위치 전 특징), `run_SKHYNIX_12_profiles.csv`(사건 전후 −10~+10), "
          "`run_SKHYNIX_12_figs/SKHYNIX_12_bottom.html`·`SKHYNIX_12_top.html`(그림)",
          f"- 실행: `cd events/code && py -B run_combo_SKHYNIX_12.py`({o['seconds']:.0f}초)"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
