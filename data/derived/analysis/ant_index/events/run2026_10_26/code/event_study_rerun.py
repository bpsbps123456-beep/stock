"""부록 15 사건 연구 2단계: event_study.py를 로직 변경 없이 6조합(KOSPI·SK하이닉스·SOXX × 8%·12%)으로 실행한다.

event_study.py의 함수(build_panel·load_contrasts·run_contrast·figure_html·guard_real_run)를 그대로 부른다. 분석·문턱·귀무
(섞기 1,000번)·위약(20일)·씨앗(대비 이름에서 나옴 → 앞선 실행과 같은 씨앗)은 그대로다. 이 파일이 하는 일은
  1) 입력 위치: events/run2026_10_26/results/{events.json, features_all.parquet(+meta), prices_close.csv}
  2) 사건 기간 거르기(부록 15: 2025-01-21 이후, 앞 20거래일은 z 예열): 극값 날짜가 2025-01-21 전인 확정 꺾임점은 사건에서 빼고
     warmup_pivots로 옮긴다(앞선 실행이 2025-08-21 전 꺾임점을 다룬 방식과 같다 — 위약 후보에서 ±3거래일을 빼는 데만 쓴다).
     2025-01-21 전 가짜 후보도 뺀다. 끝 극값(미확정)은 load_contrasts가 원래 사건에 넣지 않고, 그 ±3거래일 보류 가짜도
     가짜 목록에 없다. guard_real_run(10/24 06:00 뒤·10/23 종가 재판정 표지)을 그대로 통과해야 한다.
  3) 조합별 산출: results/run_<조합>.json/.md, run_<조합>_tests.csv·_lead.csv·_now.csv, run_<조합>_figs/*.html
  4) 덧붙임(설명용): 후보 칸의 하나 빼고 맞히기 표본별 적중(사건 새·옛 기간 나눔), 전체 표본 문턱, 실시간성,
     앞선 실행(events/results/run_<조합>.json)과 나란히.
실행: py -B event_study_rerun.py [조합 …]   (조합 생략 = 6개 전부)
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import snap_paths as SP  # noqa: E402

assert Path(SP.__file__).resolve().parent == HERE, "이 실행의 snap_paths가 아님"

OUT = SP.RUN_EV / "results"
OLD = SP.EV / "results"
EVJ, FEAT, META, PRICES = (OUT / "events.json", OUT / "features_all.parquet", OUT / "features_all_meta.json",
                           OUT / "prices_close.csv")
STUDY_START = "2025-01-21"
NEW_END = "2025-08-20"                 # 앞선 실행 기간(2025-08-21~) 앞 = 이번에 새로 들어온 사건
KNOW_END = "2026-07-01"                # 라벨 모델 지식 기간: D < 2026-07-01
JUDGE_SHA_EXPECT = "ddb1dec40a91364e148c4eb411bd184fd6b7da7a67efcb3affa8519a99575c8c"
JUDGE = SP.FINAL / "code" / "judge.py"
COMBOS = [("KOSPI", "8"), ("KOSPI", "12"), ("SKHYNIX", "8"), ("SKHYNIX", "12"), ("SOXX", "8"), ("SOXX", "12")]
KO = {"bottom": "바닥", "top": "고점"}
ASSET_KO = {"KOSPI": "KOSPI", "SKHYNIX": "SK하이닉스", "SOXX": "SOXX"}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_es():
    spec = importlib.util.spec_from_file_location("events_code_event_study", SP.EV / "code" / "event_study.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ES = load_es()


# ---------------------------------------------------------------- 사건 기간 거르기

def study_period_events(ev: dict, start: str = STUDY_START) -> tuple[dict, dict]:
    out = copy.deepcopy(ev)
    rec = {}
    for asset, a in out["assets"].items():
        for X, t in a["thresholds"].items():
            pre = [p for p in t["pivots"] if p["date"] < start]
            t["warmup_pivots"] = list(t.get("warmup_pivots", [])) + [
                {"kind": p["kind"], "date": p["date"], "close": p["close"], "confirm_date": p["confirm_date"]} for p in pre]
            t["pivots"] = [p for p in t["pivots"] if p["date"] >= start]
            fb0 = [f["date"] for f in t["fake_bottoms"] if f["date"] < start]
            ft0 = [f["date"] for f in t["fake_tops"] if f["date"] < start]
            t["fake_bottoms"] = [f for f in t["fake_bottoms"] if f["date"] >= start]
            t["fake_tops"] = [f for f in t["fake_tops"] if f["date"] >= start]
            rec[f"{asset}_{X}"] = {"pivots_moved_to_warmup": [(p["kind"], p["date"]) for p in pre],
                                   "fake_bottoms_dropped": fb0, "fake_tops_dropped": ft0,
                                   "tail_unconfirmed": {k: (t.get("tail_unconfirmed") or {}).get(k) for k in
                                                        ("kind", "date", "close", "status", "confirm_level", "last_close")},
                                   "held_near_tail": {k: [h["date"] for h in v] for k, v in t["held_near_tail"].items()}}
    return out, rec


# ---------------------------------------------------------------- 칸별 덧붙임(설명용)

def loeo_pred(v: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ES.loeo_ba와 같은 규칙(표본 하나 빼기, 균형 정확도 최대 문턱, 같은 점수 처리 동일)으로 빠진 표본의 예측(원래 순서)."""
    v = np.asarray(v, float)
    y = np.asarray(y, bool)
    n = len(v)
    order = np.argsort(v, kind="mergesort")
    vs, ys = v[order], y[order]
    pred_s = np.zeros(n, bool)
    for i in range(n):
        keep = np.ones(n, bool)
        keep[i] = False
        vt, yt = vs[keep], ys[keep]
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


def fit_rule(v: np.ndarray, y: np.ndarray) -> dict:
    """전체 표본에 같은 규칙(균형 정확도 최대 문턱·방향, 같은 점수면 위쪽·낮은 문턱)을 맞춘 문턱(설명용)."""
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
    return {"thr_z": float(thr), "is_up": bool(is_up), "event_side": "z > 문턱" if is_up else "z ≤ 문턱",
            "ba_in_sample": float(max(up.max(), dn.max()))}


def realtime_tag(market: str, offset: int, meta_f: dict) -> str:
    if meta_f["bundle"] == ES.LEAK_BUNDLE or not meta_f["realtime_comparable"]:
        return "측정 불가(조회·추천 누적값, 사건 뒤에 처음 받음)"
    if market == "kr":
        return "마감 안(K창, D+1 07:59까지)"
    return "마감 안(U창 D 21:30)" if offset == 0 else "마감 초과(U창 D+1 21:30)"


def cell_detail(panel, r: dict, feature: str, offset: int, ev_dates: list[str], fk_dates: list[str]) -> dict:
    cls = r["cls"]
    base, y = cls["base"], cls["y"]
    fi = panel.feats.index(feature)
    v = ES.gather(panel.Z, base, [offset])[:, 0, fi]
    ok = np.isfinite(v)
    dates = np.array(ev_dates + fk_dates, dtype=object)
    have = np.r_[ES.positions(panel, r["contrast"].events["win_id"]), ES.positions(panel, r["contrast"].fake_reps["win_id"])] >= 0
    dates = dates[have]
    pr = loeo_pred(v[ok], y[ok])
    yy, dd, vv = y[ok], dates[ok], v[ok]
    P, N = int(yy.sum()), int((~yy).sum())
    tp, tn = int((pr & yy).sum()), int((~pr & ~yy).sum())
    ba = 0.5 * (tp / P + tn / N)
    newp = np.array([d < "2025-08-21" for d in dd])
    rule = fit_rule(vv, yy)
    per = [{"date": str(d), "is_event": bool(a), "z": float(z), "loeo_correct": bool(p == a),
            "period": "새(2025-01-21~08-20)" if n_ else "옛(2025-08-21~)"} for d, a, z, p, n_ in zip(dd, yy, vv, pr, newp)]
    know = np.array([d < KNOW_END for d in dd])
    return {"loeo_events_hit": tp, "loeo_events_n": P, "loeo_fakes_hit": tn, "loeo_fakes_n": N, "loeo_ba": ba,
            "knowledge_period": {"events": int((yy & know).sum()), "fakes": int((~yy & know).sum())},
            "loeo_by_period": {nm: {"events_hit": int((pr & yy & m).sum()), "events_n": int((yy & m).sum()),
                                    "fakes_hit": int((~pr & ~yy & m).sum()), "fakes_n": int((~yy & m).sum())}
                               for nm, m in (("새", newp), ("옛", ~newp))},
            "full_rule": rule, "per_sample": per}


# ---------------------------------------------------------------- 앞선 실행 요약 읽기(형식이 조합마다 다르다)

def prior_summary(asset: str, X: str) -> dict:
    p = OLD / f"run_{asset}_{X}.json"
    d = json.loads(p.read_text("utf-8"))
    out = {"file": str(p.relative_to(SP.ANT)).replace("\\", "/"), "sha256": sha(p), "contrasts": {}}
    c = d.get("contrasts")
    blocks = {}
    if isinstance(c, list):
        for b in c:
            k = b.get("kind") or b["name"].split("_")[-1]
            blocks[k] = b
    elif isinstance(c, dict):
        blocks = c
    else:                                      # KOSPI_12: 위쪽에 종류별 사전
        for k in ("bottom", "top"):
            blocks[k] = {"family": d["family"][k], "lead_family": d["lead_family"][k], "candidates": d["candidates"][k],
                         "lead_candidates": d["lead_candidates"][k], "leak": d["leak"][k], "n": d["n"][k]}
    for k, b in blocks.items():
        fam = b.get("family") or b.get("family_cls")
        lfam = b.get("lead_family") or b.get("family_lead")
        cand = b.get("candidates")
        if cand is None:
            cand = b.get("candidates_cls")
        if cand is None:
            cand = b.get("candidates_preregistered")
        lc = b.get("lead_candidates") or []
        n = b.get("n") or {}
        out["contrasts"][k] = {
            "n_events": n.get("events_with_window", n.get("events")), "n_fake_reps": n.get("fake_reps_with_window", n.get("fakes")),
            "family_pass": {kk: fam["통과"][kk] for kk in ("n_tests", "observed", "expected_by_chance", "null_q95", "p_family")},
            "lead_family_pass": {kk: lfam["통과"][kk] for kk in ("n_tests", "observed", "expected_by_chance", "null_q95", "p_family")},
            "leak_flag": bool(b["leak"]["flag"]),
            "candidates": [{"feature": x["feature"], "offset": int(x["offset"]), "ba": x.get("ba"), "p": x.get("p")} for x in cand],
            "lead_candidates": [{"feature": x["feature"], "side": x["side"], "rate_events": x.get("rate_events"),
                                 "rate_fakes": x.get("rate_fakes"), "rate_placebo_mean": x.get("rate_placebo_mean"),
                                 "p": x.get("p")} for x in lc]}
    return out


# ---------------------------------------------------------------- 조합 실행

def clean(o):
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
    try:
        if pd.isna(o):
            return None
    except (TypeError, ValueError):
        pass
    return str(o)


def f2(x, nd=2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.{nd}f}"


def run_combo(asset: str, X: str, ev_f: dict, rec: dict, panels: dict, prices: pd.DataFrame, inputs: dict) -> dict:
    t0 = time.time()
    tag = f"run_{asset}_{X}"
    figdir = OUT / f"{tag}_figs"
    figdir.mkdir(parents=True, exist_ok=True)
    contrasts = [c for c in ES.load_contrasts(ev_f) if c.asset == asset and c.threshold == X]
    assert [c.kind for c in contrasts] == ["bottom", "top"], [c.name for c in contrasts]
    tconf = ev_f["assets"][asset]["thresholds"][X]
    close = ES.price_series(prices, asset)
    prior = prior_summary(asset, X)
    blocks, tests, leads, nows, results = [], [], [], [], []
    for c in contrasts:
        panel = panels[c.market]
        r = ES.run_contrast(c, panel, close, n_null=ES.N_NULL)
        results.append(r)
        (figdir / f"{c.name}.html").write_text(ES.figure_html(r, panel, title_note="부록 15 재실행(2025-01-21~2026-10-23). "), "utf-8")
        tb, lt, nw = r["table"].copy(), r["lead"].copy(), r["now"].copy()
        for df, acc in ((tb, tests), (lt, leads), (nw, nows)):
            if len(df):
                acc.append(df.assign(asset=asset, threshold=X, kind=c.kind))
        ev_dates = c.events["date"].tolist()
        fk_dates = c.fake_reps["date"].tolist()
        ev_pos = ES.positions(panel, c.events["win_id"])
        fk_pos = ES.positions(panel, c.fake_reps["win_id"])
        z0 = ES.gather(panel.Z, ev_pos, [0])[:, 0, :]
        piv = {p["date"]: p for p in tconf["pivots"]}
        ev_list = [{"date": d, "win_id": w, "period": "새" if d <= NEW_END else "옛",
                    "knowledge_period": d < KNOW_END, "confirm_date": piv[d]["confirm_date"],
                    "confirm_lag_td": piv[d]["confirm_lag_td"], "n_features_with_z0": int(np.isfinite(z0[i]).sum())}
                   for i, (d, w) in enumerate(zip(ev_dates, c.events["win_id"]))]
        fk_list = [{"date": d, "win_id": w, "run": int(rn), "period": "새" if d <= NEW_END else "옛"}
                   for d, w, rn in zip(fk_dates, c.fake_reps["win_id"], c.fake_reps["run"])]
        # 후보·p<0.05 칸
        cand = tb[tb["candidate"]]
        cand_rows = []
        for _, q in cand.iterrows():
            det = cell_detail(panel, r, q["feature"], int(q["offset"]), ev_dates, fk_dates)
            assert abs(det["loeo_ba"] - q["ba"]) < 1e-9, (c.name, q["feature"], det["loeo_ba"], q["ba"])
            cand_rows.append({"feature": q["feature"], "ko": panel.meta[q["feature"]]["ko"], "bundle": q["bundle"],
                              "offset": int(q["offset"]), "ba": q["ba"], "p": q["p"], "direction": q["direction"],
                              "mean_z_events": q["mean_z_events"], "mean_z_fakes": q["mean_z_fakes"],
                              "n_events": int(q["n_events"]), "n_fakes": int(q["n_fakes"]),
                              "null_ba_q95": q["null_ba_q95"], "ba_all_fake_days_run_groups": q.get("ba_all_fake_days_run_groups"),
                              "realtime": realtime_tag(c.market, int(q["offset"]), panel.meta[q["feature"]]),
                              "label_layer": bool(panel.meta[q["feature"]]["label_layer"]),
                              **{k: v for k, v in det.items() if k != "per_sample"}, "per_sample": det["per_sample"]})
        p05 = tb[(tb["p"] < ES.ALPHA)].sort_values("p")
        lc = lt[lt["candidate"]]
        lp05 = lt[(lt["p"] < ES.ALPHA)].sort_values("p")
        # 앞선 실행 후보 칸의 이번 값
        pc = prior["contrasts"].get(c.kind, {})
        prior_cells = []
        for x in pc.get("candidates", []):
            q = tb[(tb["feature"] == x["feature"]) & (tb["offset"] == x["offset"])]
            q = q.iloc[0] if len(q) else None
            prior_cells.append({"feature": x["feature"], "offset": x["offset"], "prior_ba": x["ba"], "prior_p": x["p"],
                                "now_ba": None if q is None else q["ba"], "now_p": None if q is None else q["p"],
                                "now_direction": None if q is None else q.get("direction"),
                                "now_candidate": None if q is None else bool(q["candidate"])})
        prior_lead = []
        for x in pc.get("lead_candidates", []):
            q = lt[(lt["feature"] == x["feature"]) & (lt["side"] == x["side"])]
            q = q.iloc[0] if len(q) else None
            prior_lead.append({"feature": x["feature"], "side": x["side"], "prior_rate_events": x["rate_events"],
                               "prior_rate_fakes": x["rate_fakes"], "prior_rate_placebo": x["rate_placebo_mean"],
                               "prior_p": x["p"],
                               **({} if q is None else {"now_rate_events": q.get("rate_events"), "now_rate_fakes": q.get("rate_fakes"),
                                                        "now_rate_placebo": q.get("rate_placebo_mean"),
                                                        "now_rate_placebo_q975": q.get("rate_placebo_q975"),
                                                        "now_p": q.get("p"), "now_candidate": bool(q["candidate"])})})
        # 지금 위치(설명용): 후보 칸만, 같은 오프셋 분포
        now_rows = []
        sel = np.flatnonzero((panel.D >= np.datetime64(ES.NOW_FROM.date(), "D")) & (panel.D <= np.datetime64(ES.NOW_TO.date(), "D")))
        for cr in cand_rows:
            fi = panel.feats.index(cr["feature"])
            k = cr["offset"]
            Ek = ES.gather(panel.Z, ev_pos, [k])[:, 0, fi]
            Fk = ES.gather(panel.Z, fk_pos, [k])[:, 0, fi]
            Ek, Fk = Ek[np.isfinite(Ek)], Fk[np.isfinite(Fk)]
            rule = cr["full_rule"]
            for tpos in sel:
                z = panel.Z[tpos, fi]
                side = None
                if np.isfinite(z):
                    side = "사건 쪽" if ((z > rule["thr_z"]) == rule["is_up"]) else "가짜 쪽"
                now_rows.append({"feature": cr["feature"], "offset": k, "win_id": str(panel.win_id[tpos]), "D": str(panel.D[tpos]),
                                 "z": z, "pct_in_events_same_offset": float(((Ek < z).sum() + 0.5 * (Ek == z).sum()) / len(Ek))
                                 if (len(Ek) and np.isfinite(z)) else None,
                                 "pct_in_fakes_same_offset": float(((Fk < z).sum() + 0.5 * (Fk == z).sum()) / len(Fk))
                                 if (len(Fk) and np.isfinite(z)) else None,
                                 "full_rule_side": side, "realtime_window": bool(panel.realtime[tpos])})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pp = r["price_paths"]
            ppm = {k2: (np.round(np.nanmean(v, 0) * 100, 2).tolist() if len(v) else None) for k2, v in pp.items()}
        blocks.append({
            "name": c.name, "kind": c.kind, "n": r["n"],
            "n_by_period": {"events_new": sum(e["period"] == "새" for e in ev_list), "events_old": sum(e["period"] == "옛" for e in ev_list),
                            "fake_reps_new": sum(f["period"] == "새" for f in fk_list),
                            "fake_reps_old": sum(f["period"] == "옛" for f in fk_list),
                            "events_knowledge_period": sum(e["knowledge_period"] for e in ev_list)},
            "samples_used_cls": r["cls"]["n_samples"],
            "events": ev_list, "fake_reps": fk_list, "n_fakes_all_days": int(len(c.fakes_all)),
            "pivot_dates_excluded_from_placebo": c.pivot_dates,
            "family": r["family"], "lead_family": r["lead_family"], "leak": r["leak"], "outside_band": r["outside_band"],
            "placebo": r["placebo"],
            "candidates": cand_rows,
            "cells_p_lt_005": p05.reindex(columns=["feature", "offset", "bundle", "status_feature", "ba", "p", "direction",
                                                   "n_events", "n_fakes", "leak_suspect", "candidate"]).to_dict("records"),
            "lead_candidates": lc.reindex(columns=["feature", "side", "bundle", "n_events", "n_fakes", "rate_events", "rate_fakes",
                                                   "rate_placebo_mean", "rate_placebo_q975", "p", "p_vs_placebo"]).to_dict("records"),
            "lead_p_lt_005": lp05.reindex(columns=["feature", "side", "bundle", "status_feature", "rate_events", "rate_fakes",
                                                   "rate_placebo_mean", "rate_placebo_q975", "p", "p_vs_placebo", "leak_suspect",
                                                   "candidate"]).to_dict("records"),
            "prior": pc, "prior_candidate_cells_now": prior_cells, "prior_lead_candidates_now": prior_lead,
            "now_candidates": now_rows, "price_path_mean_pct": ppm, "figure": f"{tag}_figs/{c.name}.html"})
    pd.concat(tests).to_csv(OUT / f"{tag}_tests.csv", index=False, encoding="utf-8-sig")
    pd.concat(leads).to_csv(OUT / f"{tag}_lead.csv", index=False, encoding="utf-8-sig")
    if nows:
        pd.concat(nows).to_csv(OUT / f"{tag}_now.csv", index=False, encoding="utf-8-sig")
    # 두 문턱 공통(이미 끝난 다른 문턱 결과가 있으면)
    panel0 = panels[contrasts[0].market]
    out = {"made": datetime.now().replace(microsecond=0).isoformat(), "combo": f"{asset}_{X}", "asset": asset,
           "threshold_pct": X, "window": ev_f["assets"][asset]["window"],
           "mode": {"rule": "부록 15 재실행: guard_real_run 통과(10/24 06:00 뒤·10/23 종가 재판정), 사건 기간 2025-01-21~2026-10-23. "
                            "끝 극값(미확정)은 사건에서, 그 ±3거래일 보류 가짜는 가짜에서 빠진다(load_contrasts 원래 동작)",
                    "this_combo": rec[f"{asset}_{X}"]},
           "inputs": inputs,
           "settings": {"n_null": ES.N_NULL, "alpha": ES.ALPHA, "cls_offsets": list(ES.CLS_OFFSETS), "lead": list(ES.LEAD),
                        "placebo_pool": ES.PLACEBO_POOL, "z": [ES.Z_N, ES.Z_MIN], "pct": [ES.P_N, ES.P_MIN, ES.P_HI, ES.P_LO],
                        "min_true_fake": [ES.MIN_TRUE, ES.MIN_FAKE], "seed_rule": "sha256(f'{SEED}:{대비 이름}') 앞 8자리 — 앞선 실행과 같음",
                        "grid_features": len(panel0.feats),
                        "grid_by_status": {s: sum(panel0.meta[f]["status"] == s for f in panel0.feats) for s in ("통과", "설명용")},
                        "panel_windows": int(len(panel0.win_id)), "panel_first_last": [str(panel0.win_id[0]), str(panel0.win_id[-1])]},
           "contrasts": blocks, "prior_file": {"file": prior["file"], "sha256": prior["sha256"]},
           "seconds": round(time.time() - t0, 1)}
    out = clean(out)
    (OUT / f"{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    (OUT / f"{tag}.md").write_text(write_md(out), "utf-8")
    return out


# ---------------------------------------------------------------- 보고서

def famline(f: dict) -> str:
    return f"{f['observed']} / {f2(f['expected_by_chance'])} (95% {f2(f['null_q95'], 0)}, p {f2(f['p_family'], 3)})"


def write_md(o: dict) -> str:
    a, X = o["asset"], o["threshold_pct"]
    st, inp = o["settings"], o["inputs"]
    L = [f"# 개미 지수 부록 13 사건 연구 재실행 — {ASSET_KO[a]} {X}% (부록 15, 2025-01-21 ~ 2026-10-23)\n",
         "> 탐색 연구다. 결과는 '후보'이고 매매 규칙이 아니다. '지금이 바닥/고점'이라는 판단은 하지 않는다. 사건 대부분은 앞선 실행에서 "
         "이미 본 사건이다(2025-08-21 뒤). 새로 들어온 사건은 2025-01-21 ~ 2025-08-20 확정 꺾임점뿐이다.\n",
         f"> 코드 `events/code/event_study.py`(변경 없음, sha256 {inp['event_study_py_sha256'][:12]}…)를 "
         f"`events/run2026_10_26/code/event_study_rerun.py`(sha256 {inp['wrapper_sha256'][:12]}…)가 부른다. 입력 `events.json` "
         f"{inp['events_json_sha256'][:12]}…, `features_all.parquet` {inp['features_all_sha256'][:12]}…, `prices_close.csv` "
         f"{inp['prices_close_sha256'][:12]}…. judge.py 지문 {'일치' if inp['judge_py_sha256_matches'] else '불일치'}.\n",
         f"> 격자: 특징 {st['grid_features']}개(통과 {st['grid_by_status']['통과']} · 설명용 {st['grid_by_status']['설명용']}) × 오프셋 2개 "
         f"= {2 * st['grid_features']}칸. 섞기 {st['n_null']}번, 위약 {st['placebo_pool']}일, 앞섬 = 사건 전 1~5창, z = 직전 20창(값 15개 이상). "
         f"창 {st['panel_windows']}개({st['panel_first_last'][0]} ~ {st['panel_first_last'][1]}).\n"]
    md = o["mode"]["this_combo"]
    tl = md["tail_unconfirmed"]
    L.append(f"> 끝 극값: {tl.get('kind')} {tl.get('date')}(종가 {f2(tl.get('close'))}, 확정에 {f2(tl.get('confirm_level'))} 이상 필요, "
             f"10/23 종가 {f2(tl.get('last_close'))}) — {tl.get('status')}이라 사건에서 뺐다. 보류 가짜 바닥 "
             f"{', '.join(md['held_near_tail'].get('fake_bottoms', [])) or '없음'}. 2025-01-21 전이라 사건에서 뺀 꺾임점 "
             f"{', '.join(f'{k} {d}' for k, d in md['pivots_moved_to_warmup']) or '없음'}.\n")
    L += ["## 0. 결론표\n",
          "| 종류 | 사건(새/옛) | 가짜 묶음 대표(새/옛) | 가르기 통과 칸: 관측 / 우연 기대 | 앞섬 통과 칸: 관측 / 우연 기대 | 누출 시험 | 가르기 후보 | 앞섬 후보 |",
          "|---|---|---|---|---|---|---|---|"]
    for b in o["contrasts"]:
        nb = b["n_by_period"]
        cand = ", ".join(f"{x['feature']}@{x['offset']:+d}" for x in b["candidates"]) or "없음"
        lc = ", ".join(f"{x['feature']}({x['side']})" for x in b["lead_candidates"]) or "없음"
        L.append(f"| {KO[b['kind']]} | {b['n']['events_with_window']} ({nb['events_new']}/{nb['events_old']}) | "
                 f"{b['n']['fake_reps_with_window']} ({nb['fake_reps_new']}/{nb['fake_reps_old']}) | {famline(b['family']['통과'])} | "
                 f"{famline(b['lead_family']['통과'])} | {'의심' if b['leak']['flag'] else '통과'} | {cand} | {lc} |")
    L.append("\n- '우연 기대'는 사건·가짜 표지를 섞은 1,000번에서 p<0.05가 나온 칸 수의 평균, 괄호는 귀무 95% 분위와 묶음 p다.")
    L.append("- 사건 수는 창이 있는 사건이다. 2025-01~02 사건은 z 예열(직전 20창 중 15개)이 모자라 많은 칸이 결측이다(아래 1절).\n")
    L += ["## 1. 사건\n", "| 종류 | 날짜 | 기간 | 확정 지연(거래일) | 오프셋 0 z 있는 특징 수 |", "|---|---|---|---|---|"]
    for b in o["contrasts"]:
        for e in b["events"]:
            L.append(f"| {KO[b['kind']]} | {e['date']} | {e['period']} | {e['confirm_lag_td']} | {e['n_features_with_z0']} / {st['grid_features']} |")
    L.append("")
    L += ["## 2. 가르기 후보(하나 빼고 맞히기)\n",
          "| 종류 | 칸 | 방향 | 맞힘(사건·가짜, BA) | 새 기간 맞힘(사건·가짜) | p | 전체 표본 문턱 | 실시간 |", "|---|---|---|---|---|---|---|---|"]
    anyc = False
    for b in o["contrasts"]:
        for x in b["candidates"]:
            anyc = True
            nw = x["loeo_by_period"]["새"]
            fr = x["full_rule"]
            L.append(f"| {KO[b['kind']]} | {x['feature']}@{x['offset']:+d} ({x['ko'][:20]}) | {x['direction']} | "
                     f"{x['loeo_events_hit']}/{x['loeo_events_n']}·{x['loeo_fakes_hit']}/{x['loeo_fakes_n']} ({f2(x['ba'])}) | "
                     f"{nw['events_hit']}/{nw['events_n']}·{nw['fakes_hit']}/{nw['fakes_n']} | {f2(x['p'], 3)} | "
                     f"{fr['event_side']} {f2(fr['thr_z'])} | {x['realtime']}"
                     + (f"; 라벨 층 — 지식 기간 사건 {x['knowledge_period']['events']}/{x['loeo_events_n']}·가짜 "
                        f"{x['knowledge_period']['fakes']}/{x['loeo_fakes_n']}" if x['label_layer'] else "") + " |")
    if not anyc:
        L.append("| — | 없음 | | | | | | |")
    L.append("")
    L += ["### 통과·설명용 칸 중 p<0.05 (후보 아닌 이유 포함)\n", "| 종류 | 칸 | 지위 | BA | p | 방향 | 누출 의심 | 후보 |", "|---|---|---|---|---|---|---|---|"]
    for b in o["contrasts"]:
        for x in b["cells_p_lt_005"]:
            L.append(f"| {KO[b['kind']]} | {x['feature']}@{int(x['offset']):+d} | {x['status_feature']} | {f2(x['ba'])} | {f2(x['p'], 3)} | "
                     f"{x['direction']} | {'예' if x['leak_suspect'] else ''} | {'예' if x['candidate'] else ''} |")
    L.append("")
    L += ["## 3. 앞섬(사건 전 1~5창 극값 닿음)\n", "| 종류 | 칸 | 지위 | 사건 / 가짜 / 위약(97.5%) | p(대 가짜) | p(대 위약) | 후보 |",
          "|---|---|---|---|---|---|---|"]
    for b in o["contrasts"]:
        for x in b["lead_p_lt_005"]:
            L.append(f"| {KO[b['kind']]} | {x['feature']} {x['side']} | {x['status_feature']} | {f2(x['rate_events'])} / {f2(x['rate_fakes'])} / "
                     f"{f2(x['rate_placebo_mean'])}({f2(x['rate_placebo_q975'])}) | {f2(x['p'], 3)} | {f2(x['p_vs_placebo'], 3)} | "
                     f"{'예' if x['candidate'] else ''} |")
    L.append("")
    L += ["## 4. 누출 시험(조회·추천 (b) vw_ln_b_median z 중앙값, 사건 / 가짜)\n"]
    for b in o["contrasts"]:
        lk = b["leak"]
        L.append(f"- {KO[b['kind']]}: {'의심' if lk['flag'] else '통과'} — " + "; ".join(
            f"오프셋 {int(k):+d}: " + (f"{f2(v.get('median_events'))} / {f2(v.get('median_fakes'))}" if "median_events" in v
                                      else v.get("result", "")) for k, v in lk["by_offset"].items()))
    L.append("")
    L += ["## 5. 앞선 실행(2025-08-21 뒤 사건, 10/22 종가)과 나란히\n",
          "| 종류 | 사건 앞선→이번 | 가르기 통과 칸 관측/기대 앞선 → 이번 | 앞섬 통과 칸 관측/기대 앞선 → 이번 | 누출 앞선 → 이번 |", "|---|---|---|---|---|"]
    for b in o["contrasts"]:
        p = b["prior"]
        L.append(f"| {KO[b['kind']]} | {p.get('n_events')} → {b['n']['events_with_window']} | "
                 f"{p['family_pass']['observed']}/{f2(p['family_pass']['expected_by_chance'])} → "
                 f"{b['family']['통과']['observed']}/{f2(b['family']['통과']['expected_by_chance'])} | "
                 f"{p['lead_family_pass']['observed']}/{f2(p['lead_family_pass']['expected_by_chance'])} → "
                 f"{b['lead_family']['통과']['observed']}/{f2(b['lead_family']['통과']['expected_by_chance'])} | "
                 f"{'의심' if p['leak_flag'] else '통과'} → {'의심' if b['leak']['flag'] else '통과'} |")
    L += ["", "| 종류 | 앞선 후보 칸 | 앞선 값(가르기 BA·p, 앞섬 사건/가짜/위약·p) | 이번 값 | 이번 방향 | 이번 후보 |", "|---|---|---|---|---|---|"]
    anyp = False
    for b in o["contrasts"]:
        for x in b["prior_candidate_cells_now"]:
            anyp = True
            L.append(f"| {KO[b['kind']]} | {x['feature']}@{x['offset']:+d} | {f2(x['prior_ba'])}·{f2(x['prior_p'], 3)} | "
                     f"{f2(x['now_ba'])}·{f2(x['now_p'], 3)} | {x['now_direction'] or '—'} | {'예' if x['now_candidate'] else '아니오'} |")
        for x in b["prior_lead_candidates_now"]:
            anyp = True
            L.append(f"| {KO[b['kind']]} | 앞섬 {x['feature']} {x['side']} | {f2(x['prior_rate_events'])}/{f2(x['prior_rate_fakes'])}/"
                     f"{f2(x['prior_rate_placebo'])}·{f2(x['prior_p'], 3)} | {f2(x.get('now_rate_events'))}/{f2(x.get('now_rate_fakes'))}/"
                     f"{f2(x.get('now_rate_placebo'))}·{f2(x.get('now_p'), 3)} | — | {'예' if x.get('now_candidate') else '아니오'} |")
    if not anyp:
        L.append("| — | 앞선 실행 후보 없음 | | | | |")
    L.append("\n- 이번 실행은 앞선 사건을 대부분 다시 쓴다. 두 결과가 같아도 독립 확인이 아니다. 새 사건만의 확인은 "
             "`new_event_check.md`에 있다.")
    L.append("- 앞선 사건의 z도 바뀔 수 있다. 2025-08-21 앞 창이 생겨 2025-08~09 창의 직전 20창·60창이 채워졌기 때문이다.\n")
    L += ["## 6. 지금 위치(설명용, 10/15~10/23 창, 가르기 후보 칸만)\n",
          "| 종류 | 칸 | 창 | z | 사건 분포 안(같은 오프셋) | 가짜 분포 안 | 전체 표본 문턱 기준 | 실시간 창 |", "|---|---|---|---|---|---|---|---|"]
    anyn = False
    for b in o["contrasts"]:
        for x in b["now_candidates"]:
            anyn = True
            L.append(f"| {KO[b['kind']]} | {x['feature']}@{x['offset']:+d} | {x['win_id']} | {f2(x['z'])} | {f2(x['pct_in_events_same_offset'])} | "
                     f"{f2(x['pct_in_fakes_same_offset'])} | {x['full_rule_side'] or '—'} | {'예' if x['realtime_window'] else ''} |")
    if not anyn:
        L.append("| — | 가르기 후보 없음 | | | | | | |")
    L.append("\n- 위 표는 후보 칸 값이 어디에 있는지 보여 줄 뿐이다. 후보가 우연과 구별되지 않으면 지금 값은 바닥·고점의 근거가 아니다. "
             "10/13 뒤 창은 실시간 수집 창이라 댓글 양·저장 글은 소급 창과 바로 비교하면 안 된다.\n")
    L += ["## 산출\n", f"- `run_{a}_{X}.json`, `run_{a}_{X}_tests.csv`, `run_{a}_{X}_lead.csv`, `run_{a}_{X}_now.csv`, "
          f"`run_{a}_{X}_figs/*.html` (걸린 시간 {o['seconds']:.0f}초)"]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("combos", nargs="*", help="예: KOSPI_8 SOXX_12 (생략 = 6개 전부)")
    a = ap.parse_args(argv)
    todo = [c for c in COMBOS if not a.combos or f"{c[0]}_{c[1]}" in a.combos]
    jsha = sha(JUDGE)
    ev = json.loads(EVJ.read_text("utf-8"))
    ES.guard_real_run(ev)
    ev_f, rec = study_period_events(ev)
    F, meta = ES.load_features(FEAT, META)
    prices = pd.read_csv(PRICES)
    panels = {m: ES.build_panel(F, meta, m) for m in ("kr", "us")}
    inputs = {"events_json_sha256": sha(EVJ), "features_all_sha256": sha(FEAT), "features_all_meta_sha256": sha(META),
              "prices_close_sha256": sha(PRICES), "event_study_py_sha256": sha(SP.EV / "code" / "event_study.py"),
              "wrapper_sha256": sha(Path(__file__)), "judge_py_sha256": jsha,
              "judge_py_sha256_matches": jsha == JUDGE_SHA_EXPECT}
    for asset, X in todo:
        o = run_combo(asset, X, ev_f, rec, panels, prices, inputs)
        for b in o["contrasts"]:
            print(o["combo"], b["kind"], "n", b["n"]["events_with_window"], b["n_by_period"], "fam",
                  b["family"]["통과"]["observed"], b["family"]["통과"]["expected_by_chance"], "lead",
                  b["lead_family"]["통과"]["observed"], b["lead_family"]["통과"]["expected_by_chance"],
                  "leak", b["leak"]["flag"], "cand", [(x["feature"], x["offset"]) for x in b["candidates"]],
                  "lead_cand", [(x["feature"], x["side"]) for x in b["lead_candidates"]], f"{o['seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
