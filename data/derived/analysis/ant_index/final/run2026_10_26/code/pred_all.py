"""부록 15 집계 1단계: 예측 층 창 특징(2025-01-03 ~ 2026-10-23 W_KR 창 전체). 가격은 읽지 않는다.

final/code/pred_windows_full.py는 한 줄도 고치지 않고 불러 쓴다. 이 모듈의 라벨 읽기는 'final' 출처 폴더 하나(대응표 +
labels/shard_*.json)와 preview 폴더를 읽으므로, 이 스냅샷의 'final' 출처 폴더를 pred_all/로 모은다
(events/code/sample_new_windows.py가 snapshot_1023/에 한 것과 같은 꼴).
  pred_all/label_key.parquet   = final/label_key 행(12,660, Q) + events/snapshot_1023 새 창 행(T, K2026-10-22·23)
                                 + 이번 표본 행(run2026_10_26/label_key, Y 9,180). unit_idx는 이 스냅샷 단위 번호로
                                 다시 맞춘다(run2026_10_26/prior_keys_rematch.parquet, sample.py 기록). 이 스냅샷 단위에서
                                 못 찾은 행(틀 밖)은 넣지 않는다. Y 행의 shard는 100 + 원래 번호(파일 이름이 겹치지 않게).
  pred_all/labels/             = final/labels/shard_001~043(바이트 그대로) + snapshot_1023/labels/shard_044(T)
                                 + run2026_10_26/labels/shard_0kk → shard_1kk(바이트 그대로)
  pred_all/label_sample_summary.json = final 요약(재사용 preview 지문 포함) + 이 대응표 지문 + 모은 기록
  pred_all/snapshot_manifest.json    = run2026_10_26/snapshot_manifest.json 바이트 사본(events 코드가 SP.SNAP에서 읽음)
preview/label_key·labels는 그대로 쓴다(pred_windows_full의 PREVIEW).
실행: PWF.FINAL = pred_all, PWF.RES = run2026_10_26/results → results/pred_windows_full.parquet·_meta.json.
  지식 기간 표시(knowledge_period = D < 2026-07-01)는 pred_windows_full 규칙 그대로 붙는다.
실행: py -B pred_all.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import shutil  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import run_paths as RP  # noqa: E402

PA = RP.RUN / "pred_all"
Y_SHARD_OFFSET = 100
SRC_TAGS = {"Q": "final/label_key", "T": "events/snapshot_1023/label_key(T)"}


def remap(k: pd.DataFrame, tag: str, rm: pd.DataFrame, u: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """앞선 대응표 행의 unit_idx를 이 스냅샷 단위 번호로 바꾼다. 못 찾은 행은 뺀다(기록)."""
    r = rm[rm["src"] == tag].set_index("id")
    assert set(r.index) == set(k["id"]), tag
    r = r.loc[k["id"]]
    ok = (r["status"] == "ok").to_numpy()
    dropped = [{"id": i, "win_id": w, "status": s, "utype": t, "dt_kst_key": str(d)}
               for i, w, s, t, d in zip(k.loc[~ok, "id"], k.loc[~ok, "win_id"], r.loc[~ok, "status"],
                                        k.loc[~ok, "utype"], k.loc[~ok, "dt_kst"])]
    k2 = k[ok].copy()
    ni = r.loc[ok, "unit_idx_new"].astype(int).to_numpy()
    # 같은 단위인지(번호·작성자·종류·시각·창·cid·가린 글 지문)
    assert (u["no"].to_numpy()[ni] == k2["no"].to_numpy()).all()
    assert (u["a"].to_numpy()[ni] == k2["a"].to_numpy()).all()
    assert (u["utype"].to_numpy()[ni] == k2["utype"].to_numpy()).all()
    assert (u["dt_kst"].to_numpy()[ni] == k2["dt_kst"].astype("datetime64[ns]").to_numpy()).all()
    assert (u["win_kr"].to_numpy()[ni] == k2["win_id"].to_numpy()).all()
    assert (u["cid"].astype("Int64").fillna(-9).to_numpy()[ni] == k2["cid"].astype("Int64").fillna(-9).to_numpy()).all()
    assert r.loc[ok, "same_masked_text"].astype(bool).all()
    k2["unit_idx"] = ni
    return k2, dropped


def build_folder() -> dict:
    rm = pd.read_parquet(RP.RUN / "prior_keys_rematch.parquet")
    u = pd.read_parquet(RP.RES / "units.parquet", columns=["no", "cid", "a", "utype", "dt_kst", "win_kr"])
    u["dt_kst"] = u["dt_kst"].astype("datetime64[ns]")
    fk = pd.read_parquet(RP.FINAL / "label_key.parquet")
    cols = fk.columns.tolist()
    tk = pd.read_parquet(RP.SNAP1023 / "label_key.parquet")
    tk = tk[tk["id"].str.startswith("T")].reset_index(drop=True)
    yk = pd.read_parquet(RP.RUN / "label_key.parquet")
    q2, dq = remap(fk, SRC_TAGS["Q"], rm, u)
    t2, dt_ = remap(tk[cols], SRC_TAGS["T"], rm, u)
    y2 = yk[cols].copy()
    y_shard_old = y2["shard"].copy()
    y2["shard"] = y2["shard"] + Y_SHARD_OFFSET
    parts = [q2, t2, y2]
    for p in parts:
        for c in cols:
            if str(p[c].dtype) != str(fk[c].dtype):
                p[c] = p[c].astype(fk[c].dtype)
    comb = pd.concat(parts, ignore_index=True)
    assert comb["id"].is_unique and comb["unit_idx"].is_unique
    pv = pd.read_parquet(RP.ANT / "preview" / "label_key.parquet")
    assert not (set(comb["id"]) & set(pv["id"])) and not (set(comb["win_id"]) & set(pv["win_id"]))
    PA.mkdir(parents=True, exist_ok=True)
    kp = PA / "label_key.parquet"
    comb.to_parquet(kp, index=False)

    # 라벨 파일: 바이트 그대로 복사
    ld = PA / "labels"
    ld.mkdir(parents=True, exist_ok=True)
    plan = [(p, ld / p.name) for p in sorted((RP.FINAL / "labels").glob("shard_*.json"))]
    plan += [(RP.SNAP1023 / "labels" / "shard_044.json", ld / "shard_044.json")]
    for p in sorted((RP.RUN / "labels").glob("shard_*.json")):
        k = int(p.stem.split("_")[1])
        plan.append((p, ld / f"shard_{k + Y_SHARD_OFFSET:03d}.json"))
    names = [q.name for _, q in plan]
    assert len(names) == len(set(names))
    for old in ld.glob("*.json"):
        if old.name not in names:
            old.unlink()
    copied = {}
    for src, dst in plan:
        if not dst.exists() or RP.sha(dst) != RP.sha(src):
            shutil.copyfile(src, dst)
        assert RP.sha(dst) == RP.sha(src)
        copied[dst.name] = {"from": str(src.relative_to(RP.ANT)), "sha256": RP.sha(dst)}
    # shard 열 = 라벨 파일 번호인지(pred_windows_full의 id_in_wrong_shard_file 점검과 같은 기준)
    file_of = {}
    for _, dst in plan:
        for r in json.loads(dst.read_text("utf-8")):
            file_of[r["id"]] = int(dst.stem.split("_")[1])
    sh_ok = comb["id"].map(file_of)
    shard_match = int((sh_ok == comb["shard"]).sum())

    # 스냅샷 목록 사본(events 코드가 SP.SNAP/snapshot_manifest.json을 읽는다)
    man_src, man_dst = RP.RUN / "snapshot_manifest.json", PA / "snapshot_manifest.json"
    if not man_dst.exists() or RP.sha(man_dst) != RP.sha(man_src):
        shutil.copyfile(man_src, man_dst)
    assert RP.sha(man_dst) == RP.sha(man_src)

    fs = json.loads((RP.FINAL / "label_sample_summary.json").read_text("utf-8"))
    fs["label_key_final_original"] = fs["label_key"]
    fs["label_key"] = {"path": str(kp), "sha256": RP.sha(kp), "rows": int(len(comb))}
    by = comb["id"].str[0]
    upd = {
        "made": datetime.now().replace(microsecond=0).isoformat(),
        "rule": "final/code/pred_windows_full.py가 읽는 'final' 출처 폴더를 이 스냅샷용으로 모음(로직 그대로). "
                "앞선 대응표 행은 unit_idx만 이 스냅샷 단위 번호로 바꾸고, 이 스냅샷에서 못 찾은 행은 뺀다",
        "rows_by_prefix": {k: int(v) for k, v in by.value_counts().items()},
        "windows_by_prefix": {k: int(comb.loc[by == k, "win_id"].nunique()) for k in sorted(by.unique())},
        "windows_by_prefix_first_last": {k: [str(comb.loc[by == k, "win_id"].min()), str(comb.loc[by == k, "win_id"].max())]
                                         for k in sorted(by.unique())},
        "preview_rows": int(len(pv)), "preview_windows": int(pv["win_id"].nunique()),
        "dropped_not_in_this_snapshot": dq + dt_,
        "dropped_reason": "이 스냅샷 로더가 그 댓글 날짜를 'latest'(미확정)로 바꿔 창 밖·틀 밖이 됨(label_sample.frame 규칙: "
                          "날짜 미확정 댓글은 뺀다). 라벨은 파일에 남아 pred_windows_full이 id_not_in_key로 센다",
        "y_shard_renumber": {f"{int(a):03d}": f"{int(a) + Y_SHARD_OFFSET:03d}" for a in sorted(y_shard_old.unique())},
        "label_files": copied,
        "shard_column_equals_label_file": {"rows": int(len(comb)), "equal": shard_match},
        "snapshot_manifest_copy_sha256": RP.sha(man_dst),
        "units_sha256": RP.sha(RP.RES / "units.parquet"),
        "rematch_source": {"path": str(RP.RUN / "prior_keys_rematch.parquet"),
                           "sha256": RP.sha(RP.RUN / "prior_keys_rematch.parquet")},
        "code_sha256": {"final/run2026_10_26/code/pred_all.py": RP.sha(Path(__file__)),
                        "final/run2026_10_26/code/run_paths.py": RP.sha(RP.HERE / "run_paths.py")},
    }
    fs["update_run2026_10_26"] = upd
    (PA / "label_sample_summary.json").write_text(json.dumps(fs, ensure_ascii=False, indent=1, default=str), "utf-8")
    return upd


def run_pred() -> dict:
    j = RP.sha(RP.FINAL / "code" / "judge.py")
    PWF = RP.load(RP.FINAL / "code" / "pred_windows_full.py", "final_pred_windows_full")
    PWF.FINAL = PA
    PWF.RES = RP.RES
    rc = PWF.main()
    assert rc == 0
    F = pd.read_parquet(RP.RES / "pred_windows_full.parquet")
    meta = json.loads((RP.RES / "pred_windows_full_meta.json").read_text("utf-8"))
    out = {"rows": int(len(F)), "first": str(F["win_id"].iloc[0]), "last": str(F["win_id"].iloc[-1]),
           "knowledge_period": int(F["knowledge_period"].sum()), "judge_period": int((~F["knowledge_period"]).sum()),
           "segment_null": int(F["segment"].isna().sum()),
           "na": {c: int(F[c].isna().sum()) for c in ("P_bar", "P_share", "P_strong1", "P_strong5", "P_sd", "P_sd_lite",
                                                      "P_sd_raw", "P_share_rz8", "P_sd_rz8")},
           "labels": {k: meta["labels"][k] for k in ("records", "usable_labels", "missing_labels",
                                                     "format_errors_excluded", "format_errors_excluded_total",
                                                     "format_warnings_kept", "format_warnings_kept_total",
                                                     "duplicate_ids")},
           "checks": meta["checks"], "judge_py_sha256_as_registered": j == RP.JUDGE_SHA256}
    # 다른 단계와 함께 쓰는 update_meta.json은 건드리지 않고 이 단계 기록만 따로 둔다
    (RP.RES / "pred_all_run.json").write_text(json.dumps(
        {"made": meta["made"], "folder": str(PA), "outputs_sha256": meta["outputs_sha256"], **out},
        ensure_ascii=False, indent=1, default=str), "utf-8")
    return out


def main() -> int:
    upd = build_folder()
    print(json.dumps({k: upd[k] for k in ("rows_by_prefix", "windows_by_prefix", "windows_by_prefix_first_last",
                                          "dropped_not_in_this_snapshot", "shard_column_equals_label_file")},
                     ensure_ascii=False, indent=1, default=str))
    out = run_pred()
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
