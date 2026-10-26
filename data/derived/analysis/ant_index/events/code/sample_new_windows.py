"""부록 14 갱신: 새 W_KR 창(K2026-10-22·23)의 라벨 표본 — 예측 층 60단위·감정 층 40단위, 기존 표본과 같은 규칙.

가격은 읽지 않는다. 기존 창의 표본은 다시 뽑지 않는다(이미 매긴 라벨을 그대로 쓴다).
틀: final/code/label_sample.frame 그대로(snapshot_1023 units·windows_kr). 층 = 창 × 글/댓글.
예측 층: final/code/label_sample.draw 그대로(층마다 30, 시드 20261022, 층 시드 = sha256('20261022:<창>:<글|댓글>')).
  id 'T' + 8자리 16진수(시드 'IDS:2026-10-23'), 섞기 시드 'SHUFFLE:2026-10-23', 조각 snapshot_1023/label_shards/shard_044.json.
  대응표 snapshot_1023/label_key.parquet = final/label_key.parquet 행(이 스냅샷 단위에 다시 맞춤) + 새 행(shard 44).
  final/labels/shard_001~043.json을 snapshot_1023/labels/로 바이트 그대로 복사한다(pred_windows_full이 읽는 폴더).
감정 층: events/code/emotion_sample.draw 그대로(층마다 20, 재사용 없음 → 단순 무작위, 시드 20261023,
  층 시드 = sha256('20261023:<창>:<글|댓글>')), 하루 가중 w_day = emotion_sample.day_weights 그대로.
  id 'W' + 8자리 16진수(시드 'IDS:2026-10-23'), 조각 events/label_shards/shard_032.json.
  대응표 events/label_key.parquet = 기존 행(unit_idx를 이 스냅샷 단위에 다시 맞춤) + 새 행(shard 32).
  층 표 events/results/label_strata.csv: 새 층을 더하고, N_frame은 이 스냅샷 틀 크기로 두며 뽑을 때 틀 크기는 N_frame_draw에 남긴다.
단위 다시 맞추기(emotion_sample.match_to_final과 같은 기준): 글 = 번호, 댓글 = 번호·작성자·시각, 원문(제목·본문/내용) 일치.
실행: py -B sample_new_windows.py
"""
from __future__ import annotations

import glob
import hashlib
import importlib.util
import json
import shutil
from datetime import datetime
from pathlib import Path

import snap_paths as SP

import numpy as np
import pandas as pd

UNIT_COLS = ["no", "cid", "a", "utype", "dt_kst", "post_dt_kst", "dt_conf", "src_line", "title", "text", "win_kr",
             "pwin_kr"]
PRED_SHARD, EMO_SHARD = 44, 32
PRED_PREFIX, EMO_PREFIX = "T", "W"
TAG = "2026-10-23"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FLS = load(SP.FINAL / "code" / "label_sample.py", "final_label_sample")
SP.patch_final(FLS)
ES = load(SP.HERE / "emotion_sample.py", "events_emotion_sample")   # draw·day_weights·_seed(시드 20261023)만 쓴다


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def tsha(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def nz(s) -> str:
    return s if isinstance(s, str) else ""


def taken_ids() -> set[str]:
    paths = [SP.FINAL / "label_key.parquet", SP.ANT / "preview" / "label_key.parquet", SP.EV / "label_key.parquet",
             SP.ANT / "emotion" / "sample_key.parquet", SP.ANT / "emotion" / "review_key.parquet",
             SP.ANT / "emotion" / "review_label_key.parquet", SP.ANT / "bottom_case" / "label_key.parquet",
             SP.ANT / "interim" / "results" / "blind_key.parquet"]
    out: set[str] = set()
    for p in paths:
        if p.exists():
            k = pd.read_parquet(p)
            for c in ("id", "review_id", "sample_id", "src_id", "alt_src_id"):
                if c in k:
                    out |= set(k[c].dropna().astype(str))
    for pat in ("final/labels/*.json", "preview/labels/*.json", "events/labels/*.json", "emotion/labels4/*.json",
                "bottom_case/labels/*.json"):
        for f in glob.glob(str(SP.ANT / pat)):
            data = json.loads(Path(f).read_text("utf-8"))
            if isinstance(data, list):
                out |= {r["id"] for r in data if isinstance(r, dict) and isinstance(r.get("id"), str)}
    return out


def new_ids(n: int, prefix: str, seed: int, taken: set[str]) -> list[str]:
    assert not any(x.startswith(prefix) for x in taken), f"접두어 {prefix}가 이미 쓰임"
    rng = np.random.default_rng(seed)
    ids, seen = [], set(taken)
    while len(ids) < n:
        x = f"{prefix}{int(rng.integers(0, 2**32)):08x}"
        if x not in seen:
            seen.add(x)
            ids.append(x)
    return ids


def rematch(key: pd.DataFrame, u: pd.DataFrame, tag: str, strict: bool = True) -> tuple[pd.DataFrame, dict]:
    """대응표 행 → 이 스냅샷 unit_idx(글: 번호, 댓글: 번호·작성자·시각, 후보가 여럿이면 가린 글 지문 일치).
    cid·unit_idx만 바꾼다. strict면 못 찾음·창 바뀜·가린 글 바뀜이 0이어야 한다."""
    uu = u[u["no"].isin(set(key["no"]))]
    posts = uu[uu["utype"] == "post"].groupby("no")["unit_idx"].apply(list).to_dict()
    com = uu[uu["utype"] == "comment"].assign(t_ns=lambda d: d["dt_kst"].astype("datetime64[ns]").astype("int64"))
    cidx = com.groupby(["no", "a", "t_ns"])["unit_idx"].apply(list).to_dict()
    ui = u.set_index("unit_idx")
    k_ns = pd.to_datetime(key["dt_kst"]).astype("datetime64[ns]").astype("int64").to_numpy()
    new_idx, stat = [], {"ok": 0, "not_found": 0, "ambiguous": 0}
    for j, r in enumerate(key.itertuples(index=False)):
        cand = posts.get(r.no, []) if r.utype == "post" else cidx.get((r.no, r.a, int(k_ns[j])), [])
        if not cand:
            stat["not_found"] += 1
            new_idx.append(None)
            continue
        if len(cand) > 1:
            same = [i for i in cand if tsha(FLS.unit_text(r.utype, ui.at[i, "title"], ui.at[i, "text"])[0]) == r.text_sha256]
            if len(same) != 1:
                stat["ambiguous"] += 1
                new_idx.append(None)
                continue
            cand = same
        stat["ok"] += 1
        new_idx.append(int(cand[0]))
    k = key.copy()
    k["unit_idx_new"] = pd.array(new_idx, dtype="Int64")
    found = k["unit_idx_new"].notna()
    ni = k.loc[found, "unit_idx_new"].astype(int).to_numpy()
    txt = [tsha(FLS.unit_text(ut, ui.at[i, "title"], ui.at[i, "text"])[0]) for ut, i in zip(k.loc[found, "utype"], ni)]
    same_text = np.array(txt) == k.loc[found, "text_sha256"].to_numpy()
    stat.update({"tag": tag, "rows": int(len(k)), "unit_idx_changed": int((k.loc[found, "unit_idx"].to_numpy() != ni).sum()),
                 "cid_changed": int((k.loc[found, "cid"].astype("Int64").fillna(-1).astype(int).to_numpy()
                                     != ui.loc[ni, "cid"].astype("Int64").fillna(-1).astype(int).to_numpy()).sum()),
                 "win_changed": int((k.loc[found, "win_id"].to_numpy() != ui.loc[ni, "win_kr"].to_numpy()).sum()),
                 "masked_text_changed": int((~same_text).sum()),
                 "changed_rows_windows": sorted(k.loc[found].loc[k.loc[found, "unit_idx"].to_numpy() != ni, "win_id"].unique().tolist())})
    if not strict:
        return key, stat
    assert stat["not_found"] == 0 and stat["ambiguous"] == 0, stat
    assert stat["win_changed"] == 0 and stat["masked_text_changed"] == 0, stat
    k["unit_idx"] = k["unit_idx_new"].astype("int64")
    k["cid"] = pd.array(ui.loc[k["unit_idx"].to_numpy(), "cid"].to_numpy(), dtype="Int64")
    return k.drop(columns="unit_idx_new"), stat


def masked(d: pd.DataFrame) -> pd.DataFrame:
    tt = [FLS.unit_text(ut, t, b) for ut, t, b in zip(d["utype"], d["title"], d["text"])]
    d = d.copy()
    d["text_masked"] = [t for t, _ in tt]
    d["text_cut"] = [c for _, c in tt]
    return d


def write_shard(path: Path, d: pd.DataFrame) -> dict:
    part = d.sort_values("shard_pos")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"id": i, "text": t} for i, t in zip(part["id"], part["text_masked"])],
                               ensure_ascii=False, indent=1), "utf-8")
    return {"path": str(path), "n": int(len(part)), "posts": int((part["utype"] == "post").sum()),
            "comments": int((part["utype"] == "comment").sum()), "windows": int(part["win_kr"].nunique()),
            "sha256": sha(path)}


def main() -> int:
    t0 = datetime.now().replace(microsecond=0)
    u = pd.read_parquet(SP.SNAP_RES / "units.parquet", columns=UNIT_COLS)
    u = u.reset_index().rename(columns={"index": "unit_idx"})
    w_all = FLS.load_windows()
    assert len(w_all) == SP.EXPECT_WINDOWS["kr"], len(w_all)
    w_new = w_all[w_all["win_id"].isin(SP.NEW_KR_WINDOWS)].reset_index(drop=True)
    assert w_new["win_id"].tolist() == SP.NEW_KR_WINDOWS
    f, st = FLS.frame(u, w_new)
    taken = taken_ids()
    wmeta = w_new.set_index("win_id")
    summary = {"made": t0.isoformat(), "new_windows": SP.NEW_KR_WINDOWS, "frame": st, "price_data_read": False,
               "windows": w_new[["win_id", "open_kst", "close_kst", "length_h", "kind", "judge_ok"]].to_dict("records")}

    # ------------------------------------------------------------ 예측 층(60단위)
    smp, strata = FLS.draw(f, w_new)
    smp["id"] = new_ids(len(smp), PRED_PREFIX, FLS._seed(f"IDS:{TAG}"), taken)
    smp = masked(smp)
    smp = smp.iloc[np.random.default_rng(FLS._seed(f"SHUFFLE:{TAG}")).permutation(len(smp))].reset_index(drop=True)
    smp["shard"] = PRED_SHARD + smp.index // FLS.SHARD_SIZE
    smp["shard_pos"] = smp.index % FLS.SHARD_SIZE
    assert smp["shard"].nunique() == 1
    pred_file = write_shard(SP.SNAP / "label_shards" / f"shard_{PRED_SHARD:03d}.json", smp)
    key = smp[["id", "win_kr", "utype", "unit_idx", "no", "cid", "a", "dt_kst", "post_dt_kst", "dt_conf", "src_line",
               "pwin_kr", "shard", "shard_pos", "text_cut"]].rename(columns={"win_kr": "win_id"}).copy()
    key["kind"] = np.where(key["utype"] == "post", "글", "댓글")
    key["D"] = key["win_id"].map(wmeta["D"])
    key["judge_ok"] = key["win_id"].map(wmeta["judge_ok"]).astype(bool)
    key["in_knowledge_period"] = key["dt_kst"] < FLS.KNOWLEDGE_END
    key["text_sha256"] = [tsha(t) for t in smp["text_masked"]]
    fk = pd.read_parquet(SP.FINAL / "label_key.parquet")
    fk2, st_pred = rematch(fk, u, "final/label_key")
    pk = pd.read_parquet(SP.ANT / "preview" / "label_key.parquet")
    _, st_prev = rematch(pk, u, "preview/label_key(점검만, 파일은 그대로)", strict=False)
    key = key.astype({c: fk2[c].dtype for c in fk2.columns if c in key.columns and c not in ("cid",)})
    key["cid"] = key["cid"].astype("Int64")
    comb = pd.concat([fk2, key[fk2.columns]], ignore_index=True)
    assert comb["id"].is_unique
    kp = SP.SNAP / "label_key.parquet"
    comb.to_parquet(kp, index=False)
    # 라벨 폴더: final 라벨을 바이트 그대로 복사(새 창 라벨은 shard_044.json으로 따로 쓴다)
    ld = SP.SNAP / "labels"
    ld.mkdir(parents=True, exist_ok=True)
    copied = {}
    for p in sorted((SP.FINAL / "labels").glob("shard_*.json")):
        q = ld / p.name
        if not q.exists() or sha(q) != sha(p):
            shutil.copyfile(p, q)
        assert sha(q) == sha(p)
        copied[p.name] = sha(q)
    fs = json.loads((SP.FINAL / "label_sample_summary.json").read_text("utf-8"))
    fs["label_key_final_original"] = fs["label_key"]
    fs["label_key"] = {"path": str(kp), "sha256": sha(kp), "rows": int(len(comb))}
    fs["update_1023"] = {
        "rule": "final/code/label_sample.frame·draw 그대로(층마다 글 30·댓글 30, 시드 20261022). 새 창만 뽑고 기존 창은 그대로",
        "windows": SP.NEW_KR_WINDOWS, "frame": st, "sampled": int(len(smp)),
        "strata": strata.to_dict("records"), "id_prefix": PRED_PREFIX, "shard": pred_file,
        "final_key_rematch": st_pred, "preview_key_rematch_check": st_prev,
        "final_labels_copied": copied,
        "text_cut": {"글": int(((smp["utype"] == "post") & smp["text_cut"]).sum()),
                     "댓글": int(((smp["utype"] == "comment") & smp["text_cut"]).sum())},
        "leak_scan_units": {k: int(sum(bool(rx.search(t)) for t in smp["text_masked"])) for k, rx in FLS.LEAK_CHECKS.items()},
        "in_knowledge_period_units": int(key["in_knowledge_period"].sum()),
        "snapshot_manifest_sha256": sha(SP.SNAP / "snapshot_manifest.json"),
        "units_sha256": sha(SP.SNAP_RES / "units.parquet"), "windows_kr_sha256": sha(SP.SNAP_RES / "windows_kr.parquet"),
        "code_sha256": {"events/code/sample_new_windows.py": sha(Path(__file__)),
                        "final/code/label_sample.py": sha(SP.FINAL / "code" / "label_sample.py")}}
    (SP.SNAP / "label_sample_summary.json").write_text(json.dumps(fs, ensure_ascii=False, indent=1, default=str), "utf-8")
    fst = pd.read_csv(SP.FINAL / "label_strata.csv")
    pd.concat([fst, strata], ignore_index=True).to_csv(SP.SNAP / "label_strata.csv", index=False, encoding="utf-8-sig")
    summary["pred"] = fs["update_1023"]

    # ------------------------------------------------------------ 감정 층(40단위)
    picks, est = ES.draw(f, w_new, set())
    new = masked(f.loc[picks].copy())
    new["id"] = new_ids(len(new), EMO_PREFIX, ES._seed(f"IDS:{TAG}"), taken | set(smp["id"]))
    new = new.iloc[np.random.default_rng(ES._seed(f"SHUFFLE:{TAG}")).permutation(len(new))].reset_index(drop=True)
    new["shard"] = EMO_SHARD + new.index // ES.SHARD_SIZE
    new["shard_pos"] = new.index % ES.SHARD_SIZE
    assert new["shard"].nunique() == 1
    emo_file = write_shard(SP.EV / "label_shards" / f"shard_{EMO_SHARD:03d}.json", new)
    for c in ("src_id", "guideline", "label_file", "alt_src_id", "src_text_sha256"):
        new[c] = None
    new["src"] = "new"
    ek = new[["id", "src", "src_id", "alt_src_id", "guideline", "label_file", "win_kr", "utype", "unit_idx", "no", "cid",
              "a", "dt_kst", "post_dt_kst", "dt_conf", "src_line", "pwin_kr", "shard", "shard_pos", "text_cut"]] \
        .rename(columns={"win_kr": "win_id"}).copy()
    ek["kind"] = ek["utype"].map(ES.KIND)
    ek["D"] = ek["win_id"].map(wmeta["D"])
    ek["judge_ok"] = ek["win_id"].map(wmeta["judge_ok"]).astype(bool)
    ek["in_knowledge_period"] = ek["dt_kst"] < ES.KNOWLEDGE_END
    ek["parent_same_window"] = pd.array(np.where(ek["utype"] == "comment", ek["pwin_kr"] == ek["win_id"], None),
                                        dtype="boolean")
    ek["text_sha256"] = [tsha(t) for t in new["text_masked"]]
    ek["src_text_sha256"] = None
    ek["same_masked_text_as_src"] = pd.array([None] * len(ek), dtype="boolean")
    ek["w_day"], wd_info = ES.day_weights(ek, f)
    old = pd.read_parquet(SP.EV / "label_key.parquet")
    assert not (set(old["win_id"]) & set(SP.NEW_KR_WINDOWS)), "새 창 표본이 이미 있다"
    old2, st_emo = rematch(old, u, "events/label_key")
    for c in old2.columns:
        if c in ek.columns and str(ek[c].dtype) != str(old2[c].dtype):
            ek[c] = ek[c].astype(old2[c].dtype)
    allk = pd.concat([old2, ek[old2.columns]], ignore_index=True)
    allk = allk.sort_values(["win_id", "utype", "dt_kst", "id"], kind="stable").reset_index(drop=True)
    assert allk["id"].is_unique and allk["unit_idx"].is_unique
    assert np.allclose(allk.groupby(["win_id", "utype"])["w_day"].sum(), 1.0)
    allk.to_parquet(SP.EV / "label_key.parquet", index=False)
    # 층 표: N_frame = 이 스냅샷 틀 크기(features_all 점검과 가중이 쓰는 값), N_frame_draw = 뽑을 때 틀 크기
    stp = SP.EV / "results" / "label_strata.csv"
    so = pd.read_csv(stp)
    if "N_frame_draw" not in so:
        so.insert(so.columns.get_loc("N_frame") + 1, "N_frame_draw", so["N_frame"])
    w_old = w_all[w_all["win_id"].isin(set(so["win_id"]))]
    f_old, _ = FLS.frame(u, w_old)
    nf = f_old.groupby(["win_kr", "utype"]).size()
    ut = so["kind"].map({"글": "post", "댓글": "comment"})
    so["N_frame"] = nf.reindex(pd.MultiIndex.from_arrays([so["win_id"], ut])).to_numpy()
    changed = so[so["N_frame"] != so["N_frame_draw"]][["win_id", "kind", "N_frame_draw", "N_frame"]]
    est.insert(est.columns.get_loc("N_frame") + 1, "N_frame_draw", est["N_frame"])
    pd.concat([so, est[so.columns]], ignore_index=True).to_csv(stp, index=False, encoding="utf-8-sig")
    # 뽑은 단위가 이 스냅샷 틀에 그대로 있는지(K2026-10-21처럼 틀이 바뀐 층)
    f_all, _ = FLS.frame(u, w_all)
    in_fr = set(f_all["unit_idx"])
    summ_p = SP.EV / "results" / "label_sample_summary.json"
    es = json.loads(summ_p.read_text("utf-8"))
    es["update_1023"] = {
        "rule": "events/code/emotion_sample.draw·day_weights 그대로(층마다 글 20·마감 전 댓글 20, 재사용 없음 → 단순 무작위, "
                "시드 20261023). 새 창만 뽑고 기존 창 표본은 그대로",
        "snapshot": str(SP.SNAP / "snapshot_manifest.json"),
        "snapshot_manifest_sha256": sha(SP.SNAP / "snapshot_manifest.json"),
        "units_sha256": sha(SP.SNAP_RES / "units.parquet"),
        "windows": SP.NEW_KR_WINDOWS, "frame": st, "sampled_new": int(len(new)), "strata": est.to_dict("records"),
        "id_prefix": EMO_PREFIX, "shard": emo_file, "w_day": wd_info, "old_key_rematch": st_emo,
        "strata_frame_changed": changed.to_dict("records"),
        "key_units_not_in_this_frame": int((~allk["unit_idx"].isin(in_fr)).sum()),
        "key_units_not_in_this_frame_windows": sorted(allk.loc[~allk["unit_idx"].isin(in_fr), "win_id"].unique().tolist()),
        "strata_csv_rule": "N_frame = 이 스냅샷(2026-10-23 07:59) 틀 크기, N_frame_draw = 표본을 뽑을 때 틀 크기",
        "label_key": {"path": str(SP.EV / "label_key.parquet"), "sha256": sha(SP.EV / "label_key.parquet"),
                      "rows": int(len(allk))},
        "text_cut_new": {"글": int(((new["utype"] == "post") & new["text_cut"]).sum()),
                         "댓글": int(((new["utype"] == "comment") & new["text_cut"]).sum())},
        "leak_scan_new_units": {k: int(sum(bool(rx.search(t)) for t in new["text_masked"])) for k, rx in FLS.LEAK_CHECKS.items()},
        "code_sha256": {"events/code/sample_new_windows.py": sha(Path(__file__)),
                        "events/code/emotion_sample.py": sha(SP.HERE / "emotion_sample.py")}}
    summ_p.write_text(json.dumps(es, ensure_ascii=False, indent=1, default=str), "utf-8")
    summary["emotion"] = es["update_1023"]
    m = json.loads((SP.SNAP / "update_meta.json").read_text("utf-8"))
    m["label_samples"] = summary
    (SP.SNAP / "update_meta.json").write_text(json.dumps(m, ensure_ascii=False, indent=1, default=str), "utf-8")
    print(json.dumps({"pred": {k: summary["pred"][k] for k in ("sampled", "strata", "final_key_rematch",
                                                                 "preview_key_rematch_check", "leak_scan_units")},
                      "emotion": {k: summary["emotion"][k] for k in ("sampled_new", "strata", "old_key_rematch",
                                                                    "strata_frame_changed", "key_units_not_in_this_frame",
                                                                    "key_units_not_in_this_frame_windows", "w_day")}},
                     ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
