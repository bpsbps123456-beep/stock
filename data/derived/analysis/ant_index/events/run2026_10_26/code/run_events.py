"""부록 15 집계 2·3단계: 감정 층 라벨 모음 → 조회·추천(views_fetch·feat_views) → 통합 특징 표(features_all). 가격 미사용.

events/code 모듈은 한 줄도 고치지 않고 불러 쓴다. 이 폴더의 snap_paths.py(같은 이름)를 sys.path 맨 앞에 두어 스냅샷
위치만 이 실행(final/run2026_10_26)으로 바꾼다. features_all의 감정 층 폴더(EV)와 산출 폴더(OUT)만 이 실행 폴더로 바꾼다.

emo_prep  감정 층 대응표·라벨·층 표를 events/run2026_10_26/에 모은다(events/code/sample_new_windows.py가 한 것과 같은 꼴).
          label_key.parquet = events/label_key 행(새 V·W, 재사용 emotion_labels4·bottom_case; unit_idx를 이 스냅샷 단위
          번호로 다시 맞춤 — final/run2026_10_26/prior_keys_rematch.parquet 기준, 못 찾은 행은 뺀다)
          + 이번 표본 행(final/run2026_10_26/label_key, Y 9,180 = 예측 층과 같은 표본에 감정을 함께 매김, src 'new',
          w_day = emotion_sample.day_weights 그대로, shard = 100 + 원래 번호).
          labels/ = events/labels/shard_001~032(바이트 그대로) + run2026_10_26/labels/shard_0kk → shard_1kk(바이트 그대로).
          재사용 원본(emotion/labels4, bottom_case/labels)은 features_all이 원래 자리에서 읽는다.
          results/label_strata.csv = events/results/label_strata.csv(N_frame = 이 스냅샷 틀 크기, N_frame_draw = 뽑을 때)
          + 이번 표본 층(final/run2026_10_26/label_strata.csv, mode 'srs_pred60').
views     events/code/views_fetch.main → feat_views.main(산출은 final/run2026_10_26/results, snapshot_1023/results와 같은 꼴)
features  events/code/features_all.main → events/run2026_10_26/results/features_all.parquet·_meta.json
summary   창 수·특징 결측 요약 → events/run2026_10_26/results/agg_summary.json
실행: py -B run_events.py emo_prep|views|features|summary
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import snap_paths as SP  # noqa: E402

assert Path(SP.__file__).resolve().parent == HERE, "이 실행의 snap_paths가 아님"
Y_SHARD_OFFSET = 100
OUT = SP.RUN_EV / "results"
UNIT_COLS = ["no", "cid", "a", "utype", "dt_kst", "post_dt_kst", "dt_conf", "src_line", "title", "text", "win_kr",
             "pwin_kr"]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def meta_update(section: str, value) -> None:
    p = OUT / "run_events_meta.json"
    m = json.loads(p.read_text("utf-8")) if p.exists() else {}
    m[section] = value
    p.write_text(json.dumps(m, ensure_ascii=False, indent=1, default=str), "utf-8")


# ---------------------------------------------------------------- 감정 층 모음

def cmd_emo_prep() -> dict:
    FLS = load(SP.FINAL / "code" / "label_sample.py", "final_label_sample")
    SP.patch_final(FLS)
    u = pd.read_parquet(SP.SNAP_RES / "units.parquet", columns=UNIT_COLS)
    u = u.reset_index().rename(columns={"index": "unit_idx"})
    w_all = FLS.load_windows()
    assert len(w_all) == SP.EXPECT_WINDOWS["kr"], len(w_all)
    f, _ = FLS.frame(u, w_all)
    in_fr = set(f["unit_idx"])

    ek = pd.read_parquet(SP.EV / "label_key.parquet")
    rm = pd.read_parquet(SP.RUN_FINAL / "prior_keys_rematch.parquet")
    r = rm[rm["src"] == "events/label_key(감정)"].set_index("id")
    assert set(r.index) == set(ek["id"])
    r = r.loc[ek["id"]]
    ok = (r["status"] == "ok").to_numpy()
    dropped = ek.loc[~ok, ["id", "win_id", "utype", "dt_kst"]].assign(status=r.loc[~ok, "status"].to_numpy())
    e2 = ek[ok].copy()
    ni = r.loc[ok, "unit_idx_new"].astype(int).to_numpy()
    uu = u.set_index("unit_idx")
    assert (uu.loc[ni, "no"].to_numpy() == e2["no"].to_numpy()).all()
    assert (uu.loc[ni, "a"].to_numpy() == e2["a"].to_numpy()).all()
    assert (uu.loc[ni, "utype"].to_numpy() == e2["utype"].to_numpy()).all()
    assert (uu.loc[ni, "dt_kst"].astype("datetime64[ns]").to_numpy() == e2["dt_kst"].astype("datetime64[ns]").to_numpy()).all()
    assert (uu.loc[ni, "win_kr"].to_numpy() == e2["win_id"].to_numpy()).all()
    assert (uu.loc[ni, "cid"].astype("Int64").fillna(-9).to_numpy() == e2["cid"].astype("Int64").fillna(-9).to_numpy()).all()
    assert r.loc[ok, "same_masked_text"].astype(bool).all()
    e2["unit_idx"] = ni

    yk = pd.read_parquet(SP.RUN_FINAL / "label_key.parquet")
    y = yk.copy()
    y["src"] = "new"
    for c in ("src_id", "alt_src_id", "guideline", "label_file", "src_text_sha256"):
        y[c] = None
    y["same_masked_text_as_src"] = pd.array([None] * len(y), dtype="boolean")
    y["parent_same_window"] = pd.array(np.where(y["utype"] == "comment", y["pwin_kr"] == y["win_id"], None), dtype="boolean")
    y_shard_old = y["shard"].copy()
    y["shard"] = y["shard"] + Y_SHARD_OFFSET
    y = y[ek.columns.tolist()].copy()
    for c in ek.columns:
        if str(y[c].dtype) != str(ek[c].dtype):
            y[c] = y[c].astype(ek[c].dtype)
    allk = pd.concat([e2, y], ignore_index=True)
    allk = allk.sort_values(["win_id", "utype", "dt_kst", "id"], kind="stable").reset_index(drop=True)
    assert allk["id"].is_unique and allk["unit_idx"].is_unique
    assert np.allclose(allk.groupby(["win_id", "utype"])["w_day"].sum(), 1.0)
    SP.RUN_EV.mkdir(parents=True, exist_ok=True)
    kp = SP.RUN_EV / "label_key.parquet"
    allk.to_parquet(kp, index=False)

    # 라벨 파일(바이트 그대로)
    ld = SP.RUN_EV / "labels"
    ld.mkdir(parents=True, exist_ok=True)
    plan = [(p, ld / p.name) for p in sorted((SP.EV / "labels").glob("shard_*.json"))]
    for p in sorted((SP.RUN_FINAL / "labels").glob("shard_*.json")):
        plan.append((p, ld / f"shard_{int(p.stem.split('_')[1]) + Y_SHARD_OFFSET:03d}.json"))
    names = [q.name for _, q in plan]
    assert len(names) == len(set(names))
    for old in ld.glob("*.json"):
        if old.name not in names:
            old.unlink()
    copied = {}
    for src, dst in plan:
        if not dst.exists() or sha(dst) != sha(src):
            shutil.copyfile(src, dst)
        assert sha(dst) == sha(src)
        copied[dst.name] = {"from": str(src.relative_to(SP.ANT)), "sha256": sha(dst)}

    # 층 표: N_frame = 이 스냅샷 틀 크기
    nf = f.groupby(["win_kr", "utype"]).size()
    fd = f.assign(day=(f["dt_kst"] - pd.Timedelta(hours=8)).dt.normalize()).groupby(["win_kr", "utype"])["day"].nunique()
    so = pd.read_csv(SP.EV / "results" / "label_strata.csv")
    ut = so["kind"].map({"글": "post", "댓글": "comment"})
    mi = pd.MultiIndex.from_arrays([so["win_id"], ut])
    n_prev = so["N_frame"].copy()
    so["N_frame"] = nf.reindex(mi).to_numpy()
    changed = so.loc[so["N_frame"] != n_prev, ["win_id", "kind", "N_frame_draw"]].assign(
        N_frame_snapshot_1023=n_prev[so["N_frame"] != n_prev].to_numpy(),
        N_frame_now=so.loc[so["N_frame"] != n_prev, "N_frame"].to_numpy())
    ys = pd.read_csv(SP.RUN_FINAL / "label_strata.csv")
    yut = ys["kind"].map({"글": "post", "댓글": "comment"})
    ymi = pd.MultiIndex.from_arrays([ys["win_id"], yut])
    assert (nf.reindex(ymi).to_numpy() == ys["N_frame"].to_numpy()).all(), "이번 표본 층 틀 크기 ≠ 이 스냅샷 틀"
    ys2 = pd.DataFrame({"win_id": ys["win_id"], "kind": ys["kind"], "N_frame": ys["N_frame"],
                        "N_frame_draw": ys["N_frame"], "n_days": fd.reindex(ymi).to_numpy(), "n_reused": 0,
                        "n_new": ys["n_sample"], "n_new_literal_rule": ys["n_sample"], "n_sample": ys["n_sample"],
                        "mode": "srs_pred60", "short": ys["n_sample"] < 30})
    st = pd.concat([so, ys2[so.columns]], ignore_index=True)
    cnt = allk.groupby(["win_id", "kind"]).size()
    st_n = st.set_index(["win_id", "kind"])["n_sample"]
    OUT.mkdir(parents=True, exist_ok=True)
    st.to_csv(OUT / "label_strata.csv", index=False, encoding="utf-8-sig")

    info = {
        "made": datetime.now().replace(microsecond=0).isoformat(),
        "label_key": {"path": str(kp), "sha256": sha(kp), "rows": int(len(allk)),
                      "rows_by_src": {k: int(v) for k, v in allk["src"].value_counts().items()},
                      "rows_by_prefix": {k: int(v) for k, v in allk["id"].str[0].value_counts().items()},
                      "windows": int(allk["win_id"].nunique()), "first": str(allk["win_id"].min()),
                      "last": str(allk["win_id"].max())},
        "old_key_rows_dropped_not_in_this_snapshot": dropped.to_dict("records"),
        "y_shard_renumber": {f"{int(a):03d}": f"{int(a) + Y_SHARD_OFFSET:03d}" for a in sorted(y_shard_old.unique())},
        "label_files": copied,
        "strata": {"rows": int(len(st)), "strata_count_equals_key": bool((cnt.reindex(st_n.index).to_numpy()
                                                                            == st_n.to_numpy()).all()),
                   "old_strata_frame_changed_vs_snapshot_1023": changed.to_dict("records"),
                   "rule": "N_frame = 이 스냅샷 틀 크기(features_all 점검·가중이 쓰는 값), N_frame_draw = 표본을 뽑을 때 틀 크기"},
        "key_units_not_in_this_frame": int((~allk["unit_idx"].isin(in_fr)).sum()),
        "key_units_not_in_this_frame_windows": sorted(allk.loc[~allk["unit_idx"].isin(in_fr), "win_id"].unique().tolist()),
        "units_per_window": {"min": int(allk.groupby("win_id").size().min()),
                             "max": int(allk.groupby("win_id").size().max())},
        "note": "2025-01-03 ~ 2025-08-20 창의 감정 층은 예측 층 60단위 표본에 함께 매긴 라벨(창마다 글 30·댓글 30). "
                "그 뒤 창은 events 감정 표본(창마다 글 20·댓글 20, 재사용 포함)",
        "code_sha256": {"events/run2026_10_26/code/run_events.py": sha(Path(__file__)),
                        "events/run2026_10_26/code/snap_paths.py": sha(HERE / "snap_paths.py")},
    }
    meta_update("emo_prep", info)
    print(json.dumps({k: v for k, v in info.items() if k != "label_files"}, ensure_ascii=False, indent=1, default=str))
    return info


# ---------------------------------------------------------------- 조회·추천

def cmd_views() -> int:
    VF = load(SP.EV / "code" / "views_fetch.py", "events_views_fetch")
    assert VF.SNAP == SP.SNAP and VF.OUT == SP.SNAP_RES
    assert inspect.signature(VF.read_rows).parameters["units_path"].default == SP.SNAP_RES / "units.parquet"
    rc = VF.main()
    FV = load(SP.EV / "code" / "feat_views.py", "events_feat_views")
    assert FV.RES == SP.SNAP_RES and FV.OUT == SP.SNAP_RES
    rc2 = FV.main()
    vm = json.loads((SP.SNAP_RES / "views_fetch_meta.json").read_text("utf-8"))
    gm = json.loads((SP.SNAP_RES / "views_growth.json").read_text("utf-8"))
    meta_update("views", {"views_fetch": {"snapshot": vm["snapshot"], "class_counts": vm["class_counts"],
                                          "method_counts": vm["method_counts"], "log_runs": len(vm["log"]["runs"])},
                          "feat_views": {k: gm[k] for k in ("labels", "posts", "windows", "label_windows_kr",
                                                            "label_windows_us", "outputs_sha256")},
                          "out_dir": str(SP.SNAP_RES)})
    return rc or rc2


# ---------------------------------------------------------------- 통합 특징 표

def cmd_features() -> int:
    FA = load(SP.EV / "code" / "features_all.py", "events_features_all")
    FA.EV = SP.RUN_EV                     # 감정 층 대응표·라벨·층 표 = 이 실행 모음
    FA.OUT = OUT
    assert FA.RES == SP.SNAP_RES and FA.PWF.FINAL == SP.SNAP and FA.PWF.RES == SP.SNAP_RES
    assert FA.FLS.RES == SP.SNAP_RES and FA.FLS.FINAL == SP.SNAP
    rc = FA.main()
    return rc


# ---------------------------------------------------------------- 요약

def cmd_summary() -> dict:
    F = pd.read_parquet(OUT / "features_all.parquet")
    fm = json.loads((OUT / "features_all_meta.json").read_text("utf-8"))
    P = pd.read_parquet(SP.SNAP_RES / "pred_windows_full.parquet")
    pm = json.loads((SP.SNAP_RES / "pred_windows_full_meta.json").read_text("utf-8"))
    feats = list(fm["features"])
    out = {"windows": {"pred_windows_full_kr": int(len(P)), "features_all": fm["rows"],
                       "pred_first_last": [str(P["win_id"].iloc[0]), str(P["win_id"].iloc[-1])],
                       "knowledge_period_kr": int(P["knowledge_period"].sum()),
                       "judge_period_kr": int((~P["knowledge_period"]).sum())}}
    miss = {}
    for m in ("kr", "us"):
        g = F[F["market"] == m]
        miss[m] = {c: int(g[c].isna().sum()) for c in feats}
    out["features_all_missing"] = miss
    out["features_all_missing_by_period"] = {}
    for m in ("kr", "us"):
        g = F[F["market"] == m]
        out["features_all_missing_by_period"][m] = {
            lab: {c: int(gg[c].isna().sum()) for c in feats if gg[c].isna().any()} | {"_windows": int(len(gg))}
            for lab, gg in (("2025-01~08-20", g[g["D"] < "2025-08-21"]), ("2025-08-21~2026-06", g[(g["D"] >= "2025-08-21")
                                                                                                 & g["knowledge_period"]]),
                            ("2026-07~10-23", g[~g["knowledge_period"]]))}
    pc = ["P_bar", "P_share", "P_strong1", "P_strong5", "P_sd", "P_sd_lite", "P_sd_raw", "P_share_rz8", "P_sd_rz8",
          "P_sd_lite_late", "P_share_rz8_late"]
    out["pred_windows_full_missing"] = {c: int(P[c].isna().sum()) for c in pc}
    out["pred_windows_full_missing_judge_period"] = {c: int(P.loc[~P["knowledge_period"], c].isna().sum()) for c in pc}
    out["pred_labels"] = {k: pm["labels"][k] for k in ("records", "usable_labels", "missing_labels",
                                                       "format_errors_excluded", "format_warnings_kept")}
    out["features_checks"] = fm["checks"]
    (OUT / "agg_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), "utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["emo_prep", "views", "features", "summary"])
    a = ap.parse_args(argv)
    if a.cmd == "emo_prep":
        cmd_emo_prep()
        return 0
    if a.cmd == "views":
        return cmd_views()
    if a.cmd == "features":
        return cmd_features()
    cmd_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
