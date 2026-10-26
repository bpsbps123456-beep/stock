"""미리보기(부록 9) 예측 층 1단계: AI 예측 라벨(pred 0~5) → W_KR 창 특징. 가격을 읽지 않는다.

정의(부록 4 '창 특징', 작성자당 1회):
- 작성자당 1회 = (창, 작성자 해시, pred 값) 쌍을 한 번만 센다. 감정 층 window_table의 (창, 작성자, 범주) 쌍과 같은 방식
  (emotion/code/emotion_batchA.py, audit.md ⑤에서 독립 재계산으로 확인한 방식).
- P̄ = 예측 있는 쌍(pred 1~5) 평균 − 3.  P_share = 예측 있는 쌍 ÷ 전체 쌍.
- P_strong1 / P_strong5 = pred 1 / 5인 쌍 ÷ 예측 있는 쌍(주판). 전체 쌍 분모판(_all)도 함께 남긴다(분모 선택 근거는 md).
- P_sd = 예측 있는 쌍 pred의 SD(ddof 1) ÷ 같은 n 재표집 기대값. 풀 = 직전 60창의 예측 쌍 pred 값.
  등록판: 동결 D_SD 구현 그대로(풀 창 40개 이상, 표 20개 이상, feat_text.Z_MIN·DSD_MIN_AUTH).
  설명판 P_sd_lite(비등록, 판정 안 함): 풀 창 10개 이상, n 5 이상. 가격을 보기 전에 정했다.
- rz8 = interim/code/feat_text.rz8 그대로(같은 요일 직전 8창, 연휴 뒤 창·장애 창은 기준에서 뺌, 8창 다 차야 계산, ±6).
- 표본: label_key의 D ≥ 2026-07-01 창. 단위 시각이 2026-07-01 전(지식 마감 달, K2026-07-01 창의 06-30 단위 50개)은 뺀다(§2.2).
- 같은 함수로 주 창(월 08:00 마감) 특징도 만든다(설명용).

입력(읽기만): preview/labels/pred_*.json, preview/label_key.parquet, preview/results/windows_kr.parquet
산출: preview/results/pred_windows.parquet, preview/results/pred_weeks.parquet, preview/results/pred_windows_meta.json
실행: py pred_windows.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PREVIEW = HERE.parent
ANT = PREVIEW.parent
RES = PREVIEW / "results"
sys.path.insert(0, str(ANT / "interim" / "code"))
import feat_text as FX  # noqa: E402  (rz8·dsd_expected·상수: 동결 코드 그대로)

D_FIRST = pd.Timestamp("2026-07-01")
SEED = 20261018
POOL_N = FX.Z_N                 # 직전 60창
POOL_MIN_REG = FX.Z_MIN         # 등록판 풀 창 40개
N_MIN_REG = FX.DSD_MIN_AUTH     # 등록판 표 20개
POOL_MIN_LITE = 10              # 설명판(비등록)
N_MIN_LITE = 5


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_units() -> tuple[pd.DataFrame, dict]:
    key = pd.read_parquet(PREVIEW / "label_key.parquet")
    files = sorted((PREVIEW / "labels").glob("pred_*.json"))
    lab = []
    for f in files:
        lab += json.loads(f.read_text("utf-8"))
    lab = pd.DataFrame(lab)
    assert lab["id"].is_unique
    m = key.merge(lab, on="id", how="left", validate="1:1")
    info = {"label_files": len(files), "labels": int(len(lab)), "key_rows": int(len(key)),
            "unlabeled": int(m["pred"].isna().sum()), "labels_not_in_key": int(len(set(lab["id"]) - set(key["id"]))),
            "author_hash_present": bool(m["a"].notna().all()),
            "dropped_knowledge_period_units": int((m["dt_kst"] < D_FIRST).sum()),
            "dropped_knowledge_period_windows": sorted(m.loc[m["dt_kst"] < D_FIRST, "win_id"].unique().tolist())}
    m = m[m["dt_kst"] >= D_FIRST].copy()
    m["pred"] = m["pred"].astype(int)
    info["units_used"] = int(len(m))
    info["pred_counts_units"] = {int(k): int(v) for k, v in m["pred"].value_counts().sort_index().items()}
    info["sha256"] = {"label_key.parquet": sha(PREVIEW / "label_key.parquet"),
                      **{f"labels/{f.name}": sha(f) for f in files}}
    return m, info


def group_stats(d: pd.DataFrame, gkey: str, index: pd.Index) -> pd.DataFrame:
    """d: 단위(gkey, a, pred). 작성자당 1회 쌍과 단위 그대로(보조) 두 판."""
    out = pd.DataFrame(index=index)
    out["n_units"] = d.groupby(gkey).size().reindex(index).fillna(0).astype(int)
    out["n_authors"] = d.groupby(gkey)["a"].nunique().reindex(index).fillna(0).astype(int)
    pairs = d.drop_duplicates([gkey, "a", "pred"])
    for tag, src in (("", pairs), ("_unit", d)):
        g = src.groupby(gkey)
        n_all = g.size().reindex(index).fillna(0)
        pr = src[src["pred"] > 0]
        gp = pr.groupby(gkey)["pred"]
        n_pred = gp.size().reindex(index).fillna(0)
        out[f"n_pairs{tag}"] = n_all.astype(int)
        out[f"n_pred{tag}"] = n_pred.astype(int)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"P_bar{tag}"] = gp.mean().reindex(index) - 3.0
            out[f"P_share{tag}"] = np.where(n_all > 0, n_pred / n_all, np.nan)
            for v in (1, 5):
                c = pr[pr["pred"] == v].groupby(gkey).size().reindex(index).fillna(0)
                out[f"P_strong{v}{tag}"] = np.where(n_pred > 0, c / n_pred, np.nan)
                out[f"P_strong{v}_all{tag}"] = np.where(n_all > 0, c / n_all, np.nan)
        out[f"P_sd_raw{tag}"] = gp.std(ddof=1).reindex(index)
        for v in (1, 2, 3, 4, 5):
            out[f"n_pred{v}{tag}"] = pr[pr["pred"] == v].groupby(gkey).size().reindex(index).fillna(0).astype(int)
    return out, pairs


def psd_norm(F: pd.DataFrame, pairs: pd.DataFrame, valid: np.ndarray, pool_min: int, n_min: int, prefix: str):
    """P_sd = SD ÷ E[SD(n개 재표집)], 풀 = 직전 POOL_N창 예측 쌍 값(자기 제외, 앞 자료만). feat_text.score_block 구조."""
    idx = list(F.index)
    vals = {w: g["pred"].to_numpy(float) for w, g in pairs[pairs["pred"] > 0].groupby("win_id")}
    pools, sd, ex, npool = [], [], [], []
    for i, wid in enumerate(idx):
        v = vals.get(wid, np.array([]))
        prev = [pools[j] for j in range(max(0, i - POOL_N), i) if pools[j] is not None]
        npool.append(len(prev))
        e = np.nan
        s = np.nan
        if valid[i] and len(v) >= max(n_min, 2) and len(prev) >= pool_min:
            seed = int(hashlib.sha256(f"{SEED}{prefix}{wid}".encode()).hexdigest()[:8], 16)
            e = FX.dsd_expected(np.concatenate(prev), len(v), seed)
            s = float(np.std(v, ddof=1)) / e if e > 0 else np.nan
        ex.append(e)
        sd.append(s)
        pools.append(v if (valid[i] and len(v) > 0) else None)
    F[f"{prefix}_exp"] = ex
    F[prefix] = sd
    F[f"{prefix}_pool_windows"] = npool


def build_windows(m: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    w = pd.read_parquet(RES / "windows_kr.parquet")
    w = w[(w["variant"] == "main") & w["complete"]].sort_values("close_kst").reset_index(drop=True)
    w = w[w["win_id"].isin(set(m["win_id"]))].reset_index(drop=True)
    idx = pd.Index(w["win_id"], name="win_id")
    F, pairs = group_stats(m, "win_id", idx)
    for c in ["D", "weekday", "kind", "main_kind", "length_h", "over96", "open_kst", "close_kst"]:
        F[c] = w[c].to_numpy()
    valid = ~w["outage"].fillna(False).astype(bool).to_numpy()
    holiday = (w["main_kind"] == "holiday").to_numpy()
    wd = w["weekday"].to_numpy()
    F["valid"] = valid
    psd_norm(F, pairs, valid, POOL_MIN_REG, N_MIN_REG, "P_sd")
    psd_norm(F, pairs, valid, POOL_MIN_LITE, N_MIN_LITE, "P_sd_lite")
    for c in ["P_sd", "P_share"]:              # 등록 칸(H67·H70)에 쓰는 rz8만 만든다
        F[f"{c}_rz8"] = FX.rz8(F[c].to_numpy(float), valid, wd, holiday)
    # 진단: 창 안 단위가 실제로 그 창 경계 안인가(시각 필터는 label_sample이 이미 걸었다)
    mm = m.merge(w[["win_id", "open_kst", "close_kst"]], on="win_id")
    inside = bool(((mm["dt_kst"] >= mm["open_kst"]) & (mm["dt_kst"] < mm["close_kst"])).all())
    meta = {"windows": int(len(F)), "first": str(idx[0]), "last": str(idx[-1]), "units_inside_window_bounds": inside,
            "n_pred_pairs": {"min": int(F["n_pred"].min()), "median": float(F["n_pred"].median()),
                             "max": int(F["n_pred"].max()), "windows_ge_20": int((F["n_pred"] >= 20).sum()),
                             "windows_ge_5": int((F["n_pred"] >= 5).sum())},
            "finite": {c: int(F[c].notna().sum()) for c in ["P_bar", "P_share", "P_strong1", "P_sd_raw", "P_sd",
                                                            "P_sd_lite", "P_sd_rz8", "P_share_rz8"]},
            "author_dedup_removed_units": int(F["n_units"].sum() - F["n_pairs"].sum()),
            "P_sd_reg_reason": (f"등록판: 풀 창 {POOL_MIN_REG}개·표 {N_MIN_REG}개 이상. 창당 예측 쌍 최대 "
                                f"{int(F['n_pred'].max())}개라 표 조건을 채우는 창이 0개이고, 풀 조건도 41번째 창부터라 "
                                f"rz8(같은 요일 8창 더)까지는 값이 0창이다")}
    return F.reset_index(), meta


def build_weeks(m: pd.DataFrame) -> pd.DataFrame:
    """주 창: 월 08:00(07:59 포함) 마감. 단위 시각으로 배정한다. 같은 작성자당 1회 규칙(주, 작성자, pred)."""
    t = m["dt_kst"]
    days_to_mon = (7 - t.dt.weekday) % 7
    cand = (t.dt.normalize() + pd.to_timedelta(days_to_mon, unit="D")) + pd.Timedelta(hours=8)
    cand = cand.where(cand > t, cand + pd.Timedelta(days=7))
    d = m.assign(week_close=cand)
    idx = pd.Index(sorted(d["week_close"].unique()), name="week_close")
    W, _ = group_stats(d, "week_close", idx)
    W["first_unit"] = d.groupby("week_close")["dt_kst"].min().reindex(idx)
    W["last_unit"] = d.groupby("week_close")["dt_kst"].max().reindex(idx)
    W["n_daily_windows"] = d.groupby("week_close")["win_id"].nunique().reindex(idx)
    W["week_open"] = idx - pd.Timedelta(days=7)
    last_close = pd.Timestamp("2026-10-16 08:00")          # 마지막 라벨 창 K2026-10-16 마감
    W["complete"] = (W["week_open"] >= D_FIRST) & (idx <= last_close)
    W["note"] = np.where(W["week_open"] < D_FIRST, "앞 일부(07-01 전 단위 제외)",
                         np.where(idx > last_close, "뒤 일부(라벨 10-16 07:59까지)", ""))
    return W.reset_index()


def main() -> int:
    t0 = datetime.now().replace(microsecond=0)
    m, info = load_units()
    F, meta = build_windows(m)
    W = build_weeks(m)
    F.to_parquet(RES / "pred_windows.parquet", index=False)
    W.to_parquet(RES / "pred_weeks.parquet", index=False)
    out = {"made": t0.isoformat(), "price_data_read": False, "inputs": info, "windows": meta,
           "weeks": {"n": int(len(W)), "complete": int(W["complete"].sum())},
           "rules": {"author_once": "(창, 작성자 해시, pred) 쌍 1회 — emotion window_table (창, 작성자, 범주) 쌍과 같은 방식",
                     "P_strong_denominator": "주판 = 예측 있는 쌍(1~5). _all = 전체 쌍(보조)",
                     "P_sd_registered": f"풀 직전 {POOL_N}창 중 {POOL_MIN_REG}창 이상, 표 {N_MIN_REG}개 이상(feat_text 동결 상수)",
                     "P_sd_lite": f"비등록 설명판: 풀 {POOL_MIN_LITE}창 이상, n {N_MIN_LITE} 이상",
                     "rz8": "interim/code/feat_text.rz8 그대로",
                     "knowledge_cutoff": "단위 시각 < 2026-07-01 제외(§2.2 LLM 라벨)"},
           "outputs_sha256": {p: sha(RES / p) for p in ("pred_windows.parquet", "pred_weeks.parquet")},
           "code_sha256": sha(Path(__file__))}
    (RES / "pred_windows_meta.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), "utf-8")
    print(json.dumps({k: out[k] for k in ("windows", "weeks")}, ensure_ascii=False, indent=1, default=str))
    print(info["pred_counts_units"], info["dropped_knowledge_period_units"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
