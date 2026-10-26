"""최종 판정 준비: 예측 층(부록 4) AI 라벨 집계 → W_KR 창별 특징. 가격을 읽지 않는다.

입력(읽기만):
  final/label_key.parquet + final/labels/shard_*.json(새 표본 211창 12,660단위)
  preview/label_key.parquet + preview/labels/pred_*.json(재사용 72창 4,320단위)
  final/results/windows_kr.parquet(이 스냅샷 로더의 W_KR 창 달력. 가격 열 없음)
정의(부록 4 '창 특징', 미리보기·prechecks_full과 같은 함수):
  preview/code/pred_windows.group_stats·psd_norm, interim/code/feat_text.rz8를 고치지 않고 불러 쓴다.
  - 작성자당 1회 = (창, 작성자 해시, pred) 쌍 1회.
  - P_bar = 예측 있는 쌍(1~5) 평균 − 3. P_share = 예측 있는 쌍 ÷ 전체 쌍.
  - P_strong1/5 = pred 1/5 쌍 ÷ 예측 있는 쌍(주판), _all = ÷ 전체 쌍(보조).
  - P_sd = 예측 있는 쌍 pred SD(ddof 1) ÷ 같은 n 재표집 기대값(직전 60창 예측 쌍 풀에서 n개를 1,000번 뽑은 SD 평균).
    등록판: 풀 창 40개·표 20개 이상(동결 D_SD 상수). 설명판 P_sd_lite: 풀 10창·n 5 이상(비등록, 미리보기와 같음).
AI 지식 기간(§2.2, 라벨 모델 지식 마감 2026-06): 단위 시각 < 2026-07-01.
  - knowledge_period = 창 D < 2026-07-01. 이 창들의 값은 설명·예열용이고 판정에 쓰지 않는다.
  - 7/1 이후 창(K2026-07-01)에 든 6/30 단위는 미리보기처럼 뺀다(n_units_dropped_knowledge).
  - 주판(열 이름 그대로): 풀·rz8 기준에 지식 기간 창을 예열로 쓴다(prechecks_full (a)판, judge.py 메모리판과 같음).
  - _late 열: 7/1 이후 창·단위만으로 풀·rz8 기준을 만든 판(prechecks_full (b)판, 미리보기와 같음).
산출: final/results/pred_windows_full.parquet, final/results/pred_windows_full_meta.json
실행: py pred_windows_full.py
"""
from __future__ import annotations

import glob
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
ANT = FINAL.parent
RES = FINAL / "results"
PREVIEW = ANT / "preview"
INTERIM = ANT / "interim"
sys.path.insert(0, str(INTERIM / "code"))
import feat_text as FX  # noqa: E402  (rz8·상수: 동결 코드)

_sp = importlib.util.spec_from_file_location("preview_pred_windows", PREVIEW / "code" / "pred_windows.py")
PW = importlib.util.module_from_spec(_sp)   # group_stats·psd_norm: 미리보기 코드 그대로
_sp.loader.exec_module(PW)

KNOW_END = pd.Timestamp("2026-07-01")       # §2.2: 이날부터만 판정에 쓴다
SEGMENTS = [("pre2026", "2025-08-21", "2025-12-31"), ("front", "2026-01-02", "2026-05-29"),
            ("back_june", "2026-06-01", "2026-06-30"), ("back_llm", "2026-07-01", "2026-10-23")]
PRED_OK = {0, 1, 2, 3, 4, 5}
TARGET_OK = {"market", "stock", "unknown"}
CONF_OK = {1, 2, 3}
FIELDS = {"id", "pred", "pred_target", "conf"}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_int(x) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


# ---------------------------------------------------------------- 라벨 읽기·형식 점검

def read_labels(key: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """두 출처의 라벨 파일을 읽어 단위별 pred를 붙인다. 쓸 수 있는 라벨 = id가 대응표에 있고 pred가 0~5 정수."""
    srcs = [("final", sorted(glob.glob(str(FINAL / "labels" / "shard_*.json")))),
            ("preview", sorted(glob.glob(str(PREVIEW / "labels" / "pred_*.json"))))]
    key_src = dict(zip(key["id"], key["src"]))
    key_shard = dict(zip(key["id"], key["shard"]))
    err = Counter()          # 단위를 빼는 오류
    warn = Counter()         # 단위는 쓰고 세기만 하는 형식 문제
    per_file, recs, seen = {}, [], Counter()
    for src, files in srcs:
        for f in files:
            p = Path(f)
            name = f"{src}/labels/{p.name}"
            fi = {"sha256": sha(p), "records": 0, "usable": 0}
            try:
                data = json.loads(p.read_text("utf-8"))
            except (OSError, ValueError) as e:
                err["file_unreadable"] += 1
                fi["error"] = type(e).__name__
                per_file[name] = fi
                continue
            if not isinstance(data, list):
                err["file_not_list"] += 1
                per_file[name] = fi
                continue
            fi["records"] = len(data)
            shard_no = int(p.stem.split("_")[1]) if src == "final" else None
            for r in data:
                if not isinstance(r, dict):
                    err["record_not_object"] += 1
                    continue
                rid = r.get("id")
                if not isinstance(rid, str):
                    err["id_missing_or_not_str"] += 1
                    continue
                seen[rid] += 1
                if rid not in key_src:
                    err["id_not_in_key"] += 1
                    continue
                if key_src[rid] != src:
                    err["id_in_other_source_key"] += 1
                    continue
                if src == "final" and key_shard[rid] != shard_no:
                    warn["id_in_wrong_shard_file"] += 1
                miss = FIELDS - set(r)
                extra = set(r) - FIELDS
                if miss - {"pred"}:
                    warn["field_missing:" + ",".join(sorted(miss - {"pred"}))] += 1
                if extra:
                    warn["field_extra:" + ",".join(sorted(extra))] += 1
                pv = r.get("pred")
                if "pred" not in r:
                    err["pred_missing"] += 1
                    continue
                if not (is_int(pv) and int(pv) in PRED_OK):
                    err["pred_not_int_0_5"] += 1
                    continue
                pv = int(pv)
                tg = r.get("pred_target")
                if tg is not None and tg not in TARGET_OK:
                    warn["pred_target_bad_value"] += 1
                if pv == 0 and tg is not None:
                    warn["pred0_with_target"] += 1
                if pv > 0 and tg is None:
                    warn["pred_pos_without_target"] += 1
                cf = r.get("conf")
                if not (is_int(cf) and int(cf) in CONF_OK):
                    warn["conf_not_1_3"] += 1
                recs.append({"id": rid, "pred": pv, "pred_target": tg, "conf": int(cf) if is_int(cf) else None,
                             "label_file": name})
                fi["usable"] += 1
            per_file[name] = fi
    lab = pd.DataFrame(recs, columns=["id", "pred", "pred_target", "conf", "label_file"])
    dup_ids = sorted(k for k, v in seen.items() if v > 1)
    if dup_ids:                      # 같은 id가 두 번 → 값이 같으면 하나만, 다르면 모두 뺀다
        err["duplicate_id_records"] += int(sum(seen[k] - 1 for k in dup_ids))
        g = lab[lab["id"].isin(dup_ids)].groupby("id")["pred"].nunique()
        conflict = set(g[g > 1].index)
        err["duplicate_id_conflicting_pred_ids"] += len(conflict)
        lab = lab[~lab["id"].isin(conflict)].drop_duplicates("id")
    m = key.merge(lab, on="id", how="left", validate="1:1")
    missing = m["pred"].isna()
    info = {"files": len(per_file), "records": int(sum(v["records"] for v in per_file.values())),
            "usable_labels": int((~missing).sum()), "key_units": int(len(key)),
            "missing_labels": int(missing.sum()),
            "missing_by_source": {s: int(missing[m["src"] == s].sum()) for s in ("final", "preview")},
            "missing_windows": sorted(m.loc[missing, "win_id"].unique().tolist()),
            "format_errors_excluded": dict(err), "format_errors_excluded_total": int(sum(err.values())),
            "format_warnings_kept": dict(warn), "format_warnings_kept_total": int(sum(warn.values())),
            "duplicate_ids": len(dup_ids), "per_file": per_file}
    return m, info


# ---------------------------------------------------------------- 창 특징

def build(m: pd.DataFrame, w: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """m: 쓸 단위(win_id, a, pred, dt_kst). w: 창 달력(close 순). 반환: 특징 표, 쌍."""
    idx = pd.Index(w["win_id"], name="win_id")
    F, pairs = PW.group_stats(m, "win_id", idx)
    valid = ~w["outage"].fillna(False).astype(bool).to_numpy()
    holiday = (w["main_kind"] == "holiday").to_numpy()
    wd = w["weekday"].to_numpy()
    PW.psd_norm(F, pairs, valid, PW.POOL_MIN_REG, PW.N_MIN_REG, "P_sd")
    PW.psd_norm(F, pairs, valid, PW.POOL_MIN_LITE, PW.N_MIN_LITE, "P_sd_lite")
    for c in ("P_sd", "P_share"):
        F[f"{c}_rz8"] = FX.rz8(F[c].to_numpy(float), valid, wd, holiday)
    return F, pairs


def q(s: pd.Series) -> dict:
    s = pd.Series(s).dropna().astype(float)
    if s.empty:
        return {"n": 0}
    return {"n": int(len(s)), "min": float(s.min()), "p10": float(s.quantile(.1)), "p25": float(s.quantile(.25)),
            "median": float(s.median()), "p75": float(s.quantile(.75)), "p90": float(s.quantile(.9)),
            "max": float(s.max()), "mean": round(float(s.mean()), 4)}


def dist_block(F: pd.DataFrame, pairs: pd.DataFrame, mu: pd.DataFrame) -> dict:
    hist = F["n_pred"].value_counts().sort_index()
    return {"windows": int(len(F)),
            "n_units": q(F["n_units"]), "n_authors": q(F["n_authors"]), "n_pairs": q(F["n_pairs"]),
            "n_pred_pairs": q(F["n_pred"]),
            "n_pred_hist": {int(k): int(v) for k, v in hist.items()},
            "windows_n_pred_ge": {str(k): int((F["n_pred"] >= k).sum()) for k in (1, 5, 10, 15, 20)},
            "windows_n_pred1_gt0": int((F["n_pred1"] > 0).sum()), "n_pred1_pairs_total": int(F["n_pred1"].sum()),
            "windows_n_pred5_gt0": int((F["n_pred5"] > 0).sum()), "n_pred5_pairs_total": int(F["n_pred5"].sum()),
            "pred_counts_units": {int(k): int(v) for k, v in mu["pred"].value_counts().sort_index().items()},
            "pred_counts_pairs": {int(k): int(v) for k, v in pairs["pred"].value_counts().sort_index().items()},
            "P_share": q(F["P_share"]), "P_bar": q(F["P_bar"]), "P_strong1": q(F["P_strong1"]),
            "P_sd_raw": q(F["P_sd_raw"]), "P_sd_lite": q(F["P_sd_lite"])}


def main() -> int:
    t0 = datetime.now().replace(microsecond=0)
    # 창 달력(가격 없음)
    w = pd.read_parquet(RES / "windows_kr.parquet")
    w = w[(w["variant"] == "main") & w["complete"]].sort_values("close_kst").reset_index(drop=True)
    w["Dt"] = pd.to_datetime(w["D"])
    w["judge_ok"] = ~(w["over96"].fillna(False).astype(bool) | w["outage"].fillna(False).astype(bool))

    # 대응표 두 개
    cols = ["id", "win_id", "a", "dt_kst", "utype", "shard", "D", "judge_ok", "in_knowledge_period"]
    kf = pd.read_parquet(FINAL / "label_key.parquet", columns=cols).assign(src="final")
    kp = pd.read_parquet(PREVIEW / "label_key.parquet", columns=cols).assign(src="preview")
    key = pd.concat([kf, kp], ignore_index=True)
    assert key["id"].is_unique, "두 대응표 id 겹침"
    key["dt_kst"] = pd.to_datetime(key["dt_kst"])
    m_all, linfo = read_labels(key)

    checks = {}
    wins = set(w["win_id"])
    checks["key_windows_in_calendar"] = bool(set(key["win_id"]) <= wins)
    checks["calendar_windows_without_sample"] = sorted(wins - set(key["win_id"]))
    checks["windows_both_sources"] = sorted(set(kf["win_id"]) & set(kp["win_id"]))
    checks["key_knowledge_flag_matches_dt"] = bool((key["in_knowledge_period"] == (key["dt_kst"] < KNOW_END)).all())
    checks["key_D_matches_calendar"] = bool((pd.to_datetime(key["D"]) == key["win_id"].map(dict(zip(w["win_id"], w["Dt"])))).all())
    wb = m_all.merge(w[["win_id", "open_kst", "close_kst"]], on="win_id")
    checks["units_inside_window_bounds"] = bool(((wb["dt_kst"] >= wb["open_kst"]) & (wb["dt_kst"] < wb["close_kst"])).all())
    checks["key_judge_ok_matches_calendar"] = bool((key["judge_ok"] == key["win_id"].map(dict(zip(w["win_id"], w["judge_ok"])))).all())
    summ = json.loads((FINAL / "label_sample_summary.json").read_text("utf-8"))
    pv_sha = summ["reuse"]["preview_labels"]["sha256"]
    checks["preview_label_files_sha_as_recorded"] = all(
        linfo["per_file"].get(f"preview/labels/{k}", {}).get("sha256") == v for k, v in pv_sha.items())
    checks["final_label_key_sha_as_recorded"] = sha(FINAL / "label_key.parquet") == summ["label_key"]["sha256"]
    checks["preview_label_key_sha_as_recorded"] = sha(PREVIEW / "label_key.parquet") == summ["reuse"]["preview_label_key"]["sha256"]

    # 쓸 단위: 라벨 있음, 7/1 이후 창에서는 지식 기간(6/30 이전) 단위를 뺀다
    m = m_all[m_all["pred"].notna()].copy()
    m["pred"] = m["pred"].astype(int)
    late_win = pd.to_datetime(m["D"]) >= KNOW_END
    drop_k = late_win & (m["dt_kst"] < KNOW_END)
    m_a = m[~drop_k].copy()                                 # 주판: 모든 창(지식 기간 창 포함, 예열)
    m_b = m[late_win & ~drop_k].copy()                      # _late판: 7/1 이후 창·단위만

    F, pairs = build(m_a, w)
    w_late = w[w["Dt"] >= KNOW_END].reset_index(drop=True)
    Fb, _ = build(m_b, w_late)

    # 창 정보·표본 수·표시 열
    F = F.reset_index()
    cal = w.set_index("win_id")
    for c in ["D", "weekday", "kind", "main_kind", "length_h", "open_kst", "close_kst", "over96", "outage", "judge_ok"]:
        F[c] = F["win_id"].map(cal[c])
    F["D"] = pd.to_datetime(F["D"])
    F["valid"] = ~F["outage"].fillna(False).astype(bool)
    F["knowledge_period"] = F["D"] < KNOW_END
    F["segment"] = None
    for name, lo, hi in SEGMENTS:
        F.loc[(F["D"] >= lo) & (F["D"] <= hi), "segment"] = name
    g_all = m_all.groupby("win_id")
    F["label_src"] = F["win_id"].map(g_all["src"].agg(lambda s: "+".join(sorted(set(s)))))
    F["n_sampled"] = F["win_id"].map(g_all.size()).fillna(0).astype(int)
    F["n_sampled_post"] = F["win_id"].map(m_all[m_all["utype"] == "post"].groupby("win_id").size()).fillna(0).astype(int)
    F["n_sampled_comment"] = F["win_id"].map(m_all[m_all["utype"] == "comment"].groupby("win_id").size()).fillna(0).astype(int)
    F["n_missing_label"] = F["win_id"].map(m_all[m_all["pred"].isna()].groupby("win_id").size()).fillna(0).astype(int)
    F["n_units_knowledge"] = F["win_id"].map(m_all[m_all["dt_kst"] < KNOW_END].groupby("win_id").size()).fillna(0).astype(int)
    F["n_units_dropped_knowledge"] = F["win_id"].map(m[drop_k].groupby("win_id").size()).fillna(0).astype(int)
    for ut in ("post", "comment"):
        mu = m_a[m_a["utype"] == ut]
        F[f"n_pred_unit_{ut}"] = F["win_id"].map(mu[mu["pred"] > 0].groupby("win_id").size()).fillna(0).astype(int)
    # _late판(풀·rz8 기준이 7/1 이후만): 원값은 주판과 같아야 한다
    Fb = Fb.reset_index().set_index("win_id")
    same_raw = {}
    for c in ("n_units", "n_pairs", "n_pred", "P_bar", "P_share", "P_strong1", "P_sd_raw"):
        x = F.set_index("win_id").loc[Fb.index, c].astype(float).to_numpy()
        same_raw[c] = bool(np.allclose(x, Fb[c].astype(float).to_numpy(), equal_nan=True, atol=1e-12))
    checks["late_variant_raw_equal_main"] = same_raw
    for c in ("P_sd_exp", "P_sd", "P_sd_pool_windows", "P_sd_lite_exp", "P_sd_lite", "P_sd_lite_pool_windows",
              "P_sd_rz8", "P_share_rz8"):
        F[f"{c}_late"] = F["win_id"].map(Fb[c])

    lead = ["win_id", "D", "weekday", "kind", "main_kind", "segment", "knowledge_period", "judge_ok", "valid",
            "over96", "outage", "length_h", "open_kst", "close_kst", "label_src", "n_sampled", "n_sampled_post",
            "n_sampled_comment", "n_missing_label", "n_units_knowledge", "n_units_dropped_knowledge",
            "n_units", "n_authors", "n_pairs", "n_pred", "P_bar", "P_share", "P_strong1", "P_strong5",
            "P_sd", "P_sd_exp", "P_sd_pool_windows", "P_sd_lite", "P_sd_lite_exp", "P_sd_lite_pool_windows",
            "P_sd_raw", "P_share_rz8", "P_sd_rz8"]
    F = F[lead + [c for c in F.columns if c not in lead]]

    # 재현 점검: 미리보기 72창(_late판 = 미리보기 규칙), prechecks_full (a)판(주판)
    pv = pd.read_parquet(PREVIEW / "results" / "pred_windows.parquet").set_index("win_id")
    rep = {}
    Fi = F.set_index("win_id")
    for c in ("n_pred", "P_bar", "P_share", "P_strong1", "P_sd_raw", "P_sd_lite", "P_share_rz8"):
        src = Fi.loc[pv.index, c if c in ("n_pred", "P_bar", "P_share", "P_strong1", "P_sd_raw") else f"{c}_late"]
        rep[c] = int(np.isclose(src.astype(float).to_numpy(), pv[c].astype(float).to_numpy(), equal_nan=True, atol=1e-12).sum())
    checks["preview_reproduction"] = {"windows": int(len(pv)), "equal": rep,
                                      "ok": all(v == len(pv) for v in rep.values())}
    pf = RES / "prechecks_full_pred_windows.parquet"
    if pf.exists():
        pa = pd.read_parquet(pf).set_index("win_id").reindex(Fi.index)
        eq = {}
        for c in ("n_units", "n_pairs", "n_pred", "P_bar", "P_share", "P_strong1", "P_strong5", "P_sd_raw", "P_sd",
                  "P_sd_lite", "P_share_rz8", "P_sd_rz8"):
            eq[c] = int(np.isclose(Fi[c].astype(float).to_numpy(), pa[c].astype(float).to_numpy(), equal_nan=True,
                                   atol=1e-12).sum())
        checks["prechecks_full_a_reproduction"] = {"windows": int(len(Fi)), "equal": eq,
                                                   "ok": all(v == len(Fi) for v in eq.values())}

    # 분포
    dist = {"all": dist_block(F, pairs, m_a)}
    for name, _, _ in SEGMENTS:
        ids = set(F.loc[F["segment"] == name, "win_id"])
        dist[name] = dist_block(F[F["win_id"].isin(ids)], pairs[pairs["win_id"].isin(ids)], m_a[m_a["win_id"].isin(ids)])
    kn = set(F.loc[F["knowledge_period"], "win_id"])
    dist["knowledge_period"] = dist_block(F[F["win_id"].isin(kn)], pairs[pairs["win_id"].isin(kn)], m_a[m_a["win_id"].isin(kn)])
    jd = set(F.loc[~F["knowledge_period"], "win_id"])
    dist["judge_period"] = dist_block(F[F["win_id"].isin(jd)], pairs[pairs["win_id"].isin(jd)], m_a[m_a["win_id"].isin(jd)])
    F["ym"] = F["D"].dt.strftime("%Y-%m")
    monthly = F.groupby("ym").agg(windows=("win_id", "size"), n_pred_median=("n_pred", "median"),
                                  n_pred_min=("n_pred", "min"), P_share_mean=("P_share", "mean"),
                                  P_bar_mean=("P_bar", "mean"), n_pred1_sum=("n_pred1", "sum"))
    F = F.drop(columns="ym")
    jok = F[~F["knowledge_period"] & F["judge_ok"]]
    finite_cols = ("P_bar", "P_share", "P_strong1", "P_sd", "P_sd_lite", "P_sd_raw", "P_share_rz8", "P_sd_rz8",
                   "P_sd_lite_late", "P_share_rz8_late", "P_sd_rz8_late")
    finite = {"all_windows": {c: int(F[c].notna().sum()) for c in finite_cols},
              "judge_period_judge_ok": {"windows": int(len(jok)), **{c: int(jok[c].notna().sum()) for c in finite_cols}},
              "P_share_rz8_first": {"main": str(F.loc[F["P_share_rz8"].notna(), "D"].min().date()),
                                    "late": (str(F.loc[F["P_share_rz8_late"].notna(), "D"].min().date())
                                             if F["P_share_rz8_late"].notna().any() else None)}}

    out_p = RES / "pred_windows_full.parquet"
    F.to_parquet(out_p, index=False)
    meta = {
        "made": t0.isoformat(), "price_data_read": False,
        "inputs": {"final_label_key": {"rows": int(len(kf)), "windows": int(kf["win_id"].nunique()),
                                       "sha256": sha(FINAL / "label_key.parquet")},
                   "preview_label_key": {"rows": int(len(kp)), "windows": int(kp["win_id"].nunique()),
                                         "sha256": sha(PREVIEW / "label_key.parquet")},
                   "windows_kr": {"main_complete": int(len(w)), "first": w["win_id"].iloc[0], "last": w["win_id"].iloc[-1],
                                  "sha256": sha(RES / "windows_kr.parquet")}},
        "labels": linfo,
        "knowledge_period": {
            "rule": "§2.2: 라벨 모델 지식 마감(2026-06) 다음 날 2026-07-01부터만 판정. 단위 시각 < 2026-07-01이면 지식 기간",
            "windows_knowledge": int(F["knowledge_period"].sum()), "windows_judge": int((~F["knowledge_period"]).sum()),
            "units_knowledge_in_key": int((key["dt_kst"] < KNOW_END).sum()),
            "units_dropped_from_judge_windows": int(drop_k.sum()),
            "windows_with_dropped_units": sorted(m.loc[drop_k, "win_id"].unique().tolist()),
            "units_used_main": int(len(m_a)), "units_used_late": int(len(m_b))},
        "author_once_removed_units": int(F["n_units"].sum() - F["n_pairs"].sum()),
        "distribution": dist,
        "monthly": {k: {kk: (round(float(vv), 4) if isinstance(vv, (float, np.floating)) else int(vv))
                        for kk, vv in r.items()} for k, r in monthly.iterrows()},
        "finite": finite,
        "checks": checks,
        "rules": {"author_once": "(창, 작성자 해시, pred) 쌍 1회 — preview/code/pred_windows.group_stats 그대로",
                  "P_strong_denominator": "주판 ÷ 예측 있는 쌍(1~5), _all ÷ 전체 쌍",
                  "P_sd": f"등록판: 직전 {PW.POOL_N}창 풀 중 {PW.POOL_MIN_REG}창 이상, 표 {PW.N_MIN_REG}개 이상, 재표집 {FX.DSD_DRAWS}회",
                  "P_sd_lite": f"비등록 설명판: 풀 {PW.POOL_MIN_LITE}창 이상, n {PW.N_MIN_LITE} 이상",
                  "rz8": "interim/code/feat_text.rz8 그대로(같은 요일 직전 8창, 연휴 뒤·무효 창 제외, ±6)",
                  "valid": "outage가 아닌 창(W_KR 장애 창 0). 96시간 초과 창은 특징은 만들고 judge_ok=False",
                  "main_vs_late": "주판 = 지식 기간 창을 풀·rz8 예열에 씀(prechecks_full (a), judge.py 메모리판). "
                                  "_late = 7/1 이후만(미리보기·prechecks_full (b))",
                  "label_usable": "id가 같은 출처 대응표에 있고 pred가 0~5 정수(bool 제외). conf·pred_target 문제는 세기만 함"},
        "outputs_sha256": {"pred_windows_full.parquet": sha(out_p)},
        "code_sha256": {"final/code/pred_windows_full.py": sha(Path(__file__)),
                        "preview/code/pred_windows.py": sha(PREVIEW / "code" / "pred_windows.py"),
                        "interim/code/feat_text.py": sha(INTERIM / "code" / "feat_text.py")},
    }
    (RES / "pred_windows_full_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), "utf-8")
    brief = {"labels": {k: linfo[k] for k in ("records", "usable_labels", "missing_labels", "format_errors_excluded_total",
                                              "format_warnings_kept_total", "duplicate_ids")},
             "knowledge": meta["knowledge_period"], "finite": finite, "checks": checks}
    print(json.dumps(brief, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
