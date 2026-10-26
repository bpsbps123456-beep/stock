"""부록 15 준비 3단계: 라벨 표본(두 층 — 예측 층 지침 v2 §4와 감정 층 9감정을 같은 표본에 함께 매김). 가격은 읽지 않는다.

대상 창: 이 스냅샷(run2026_10_26/results) W_KR 본판 완전 창 중
  ① D 2025-01-02 ~ 2025-08-20(앞선 실행에 없던 창),
  ② K2026-10-22·K2026-10-23 중 아직 예측 라벨이 없는 창. events/snapshot_1023(예측 층 'T', 같은 규칙: 창마다 글 30·댓글 30)과
     events/labels(감정 층 'W', 창마다 20·20)에 라벨이 있으면 다시 뽑지 않고 재사용으로 기록한다.
틀: final/code/label_sample.frame 그대로(날짜 미확정 댓글, 삭제 댓글, 사이트 알림, 동영상 자리 문구만 남은 빈 단위,
    §2.4 정치 기간(2025-01~06, 2026-06)의 정치 전용 단위를 뺀다).
층: final/code/label_sample.draw 그대로(창 × 글/댓글, 층마다 30, 모자라면 전부). 시드만 20261026,
    층 시드 = sha256('20261026:<창 id>:<글|댓글>') 앞 8자리.
가림: final/code/label_sample.unit_text 그대로(interim/code/blind_sample.mask_text, 글 '제목: …\\n본문: …' 600자, 댓글 400자).
id: 'Y' + 8자리 16진수(시드 'IDS:2026-10-26'), 기존 대응표·라벨·조각의 id와 겹치지 않는다(접두어 Y 미사용 확인).
조각: 새 표본 전체를 섞어(시드 'SHUFFLE:2026-10-26') 300단위씩 run2026_10_26/label_shards/shard_XXX.json([{id, text}]).
    창·날짜·시각·번호·작성자는 싣지 않는다.
대응표: run2026_10_26/label_key.parquet(final/label_key.parquet와 같은 열 + 감정 층용 하루 가중 w_day
    = events/code/emotion_sample.day_weights 그대로). 라벨러에게 주지 않는다.
점검: 앞선 대응표(final·preview·snapshot_1023 예측, events 감정)를 이 스냅샷 단위에 다시 맞춘 결과
    (sample_new_windows.rematch와 같은 기준: 글 = 번호, 댓글 = 번호·작성자·시각, 후보가 여럿이면 가린 글 지문)를 기록만 한다.
실행: py -B sample.py
"""
from __future__ import annotations

import glob
import hashlib
import json
from datetime import datetime
from pathlib import Path

import run_paths as RP

import numpy as np
import pandas as pd

UNIT_COLS = ["no", "cid", "a", "utype", "dt_kst", "post_dt_kst", "dt_conf", "src_line", "title", "text", "win_kr",
             "pwin_kr"]
FLS = RP.load(RP.FINAL / "code" / "label_sample.py", "final_label_sample")
RP.patch_final(FLS)
FLS.SEED = RP.SEED                      # 층 시드 = sha256('20261026:<창>:<글|댓글>')
ES = RP.load(RP.EV / "code" / "emotion_sample.py", "events_emotion_sample")   # day_weights만 쓴다
E9 = ES.E9


def tsha(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def taken_ids() -> tuple[set[str], dict]:
    """ant_index 아래 모든 대응표(*.parquet의 id류 열)와 라벨·조각 json의 id."""
    out: set[str] = set()
    src: dict[str, int] = {}
    for p in RP.ANT.rglob("*.parquet"):
        if RP.RUN in p.parents or p.stat().st_size > 50_000_000:
            continue
        if not any(s in p.name for s in ("key", "train_set", "sample")):
            continue
        k = pd.read_parquet(p)
        n0 = len(out)
        for c in ("id", "review_id", "sample_id", "src_id", "alt_src_id"):
            if c in k:
                out |= set(k[c].dropna().astype(str))
        src[str(p.relative_to(RP.ANT))] = len(out) - n0
    for p in RP.ANT.rglob("*.json"):
        if RP.RUN in p.parents or not any(s.startswith(("labels", "label_shards")) for s in p.parts):
            continue
        data = json.loads(p.read_text("utf-8"))
        if isinstance(data, list):
            out |= {r["id"] for r in data if isinstance(r, dict) and isinstance(r.get("id"), str)}
    return out, src


def new_ids(n: int, taken: set[str]) -> list[str]:
    assert not any(x.startswith(RP.ID_PREFIX) for x in taken), f"접두어 {RP.ID_PREFIX}가 이미 쓰임"
    rng = np.random.default_rng(FLS._seed(f"IDS:{RP.TAG}"))
    ids, seen = [], set(taken)
    while len(ids) < n:
        x = f"{RP.ID_PREFIX}{int(rng.integers(0, 2**32)):08x}"
        if x not in seen:
            seen.add(x)
            ids.append(x)
    return ids


def rematch_table(key: pd.DataFrame, u: pd.DataFrame, tag: str) -> tuple[pd.DataFrame, dict]:
    """대응표 행 → 이 스냅샷 unit_idx. events/code/sample_new_windows.rematch와 같은 기준(기록만, 대응표는 고치지 않음)."""
    uu = u[u["no"].isin(set(key["no"]))]
    posts = uu[uu["utype"] == "post"].groupby("no")["unit_idx"].apply(list).to_dict()
    com = uu[uu["utype"] == "comment"].assign(t_ns=lambda d: d["dt_kst"].astype("datetime64[ns]").astype("int64"))
    cidx = com.groupby(["no", "a", "t_ns"])["unit_idx"].apply(list).to_dict()
    ui = u.set_index("unit_idx")
    k_ns = pd.to_datetime(key["dt_kst"]).astype("datetime64[ns]").astype("int64").to_numpy()
    rows = []
    for j, r in enumerate(key.itertuples(index=False)):
        cand = posts.get(r.no, []) if r.utype == "post" else cidx.get((r.no, r.a, int(k_ns[j])), [])
        status, idx = "ok", None
        if not cand:
            status = "not_found"
        elif len(cand) > 1:
            same = [i for i in cand if tsha(FLS.unit_text(r.utype, ui.at[i, "title"], ui.at[i, "text"])[0]) == r.text_sha256]
            if len(same) == 1:
                idx = int(same[0])
            else:
                status = "ambiguous"
        else:
            idx = int(cand[0])
        if idx is None:
            rows.append({"id": r.id, "src": tag, "win_id": r.win_id, "status": status, "unit_idx_new": None,
                         "win_id_new": None, "same_masked_text": None, "in_frame_new": None})
            continue
        t = tsha(FLS.unit_text(r.utype, ui.at[idx, "title"], ui.at[idx, "text"])[0])
        rows.append({"id": r.id, "src": tag, "win_id": r.win_id, "status": status, "unit_idx_new": idx,
                     "win_id_new": ui.at[idx, "win_kr"], "same_masked_text": t == r.text_sha256, "in_frame_new": None})
    t = pd.DataFrame(rows)
    t["unit_idx_new"] = t["unit_idx_new"].astype("Int64")
    found = t["status"] == "ok"
    stat = {"tag": tag, "rows": int(len(t)), "ok": int(found.sum()), "not_found": int((t["status"] == "not_found").sum()),
            "ambiguous": int((t["status"] == "ambiguous").sum()),
            "win_changed": int((found & (t["win_id_new"] != t["win_id"])).sum()),
            "masked_text_changed": int((found & (t["same_masked_text"] == False)).sum()),  # noqa: E712
            "windows_with_problem": sorted(t.loc[~found | (t["win_id_new"] != t["win_id"])
                                                 | (t["same_masked_text"] == False), "win_id"].dropna().unique().tolist())}  # noqa: E712
    return t, stat


def labels_for(ids: set[str], pattern: str, need: set[str]) -> dict:
    """라벨 파일들에서 ids의 라벨을 찾는다. need = 있어야 하는 필드."""
    found, bad, files = {}, [], set()
    for p in sorted(glob.glob(str(RP.ANT / pattern))):
        for r in json.loads(Path(p).read_text("utf-8")):
            if isinstance(r, dict) and r.get("id") in ids:
                if need <= set(r):
                    found[r["id"]] = r
                    files.add(str(Path(p).relative_to(RP.ANT)))
                else:
                    bad.append(r["id"])
    return {"found": found, "bad": bad, "files": sorted(files)}


def check_windows(u: pd.DataFrame, w_all: pd.DataFrame) -> tuple[list[str], dict]:
    """K2026-10-22·23: 같은 규칙 예측 라벨(events/snapshot_1023 'T')·감정 라벨(events 'W')이 있는지."""
    pk = pd.read_parquet(RP.SNAP1023 / "label_key.parquet")
    ek = pd.read_parquet(RP.EV / "label_key.parquet")
    st_old = pd.read_csv(RP.SNAP1023 / "label_strata.csv")
    est_old = pd.read_csv(RP.EV / "results" / "label_strata.csv")
    f_chk, _ = FLS.frame(u, w_all[w_all["win_id"].isin(RP.CHECK_WINDOWS)])
    nf = f_chk.groupby(["win_kr", "utype"]).size()
    need_sample, info = [], {}
    for wid in RP.CHECK_WINDOWS:
        kp = pk[pk["win_id"] == wid]
        ke = ek[ek["win_id"] == wid]
        lp = labels_for(set(kp["id"]), "events/snapshot_1023/labels/shard_*.json", {"id", "pred", "pred_target", "conf"})
        le = labels_for(set(ke["id"]), "events/labels/shard_*.json", {"id", "emo", "pred", "pred_target", "conf"})
        pred_ok = len(kp) > 0 and len(lp["found"]) == len(kp) and not lp["bad"]
        emo_ok = len(ke) > 0 and len(le["found"]) == len(ke) and not le["bad"]
        rp, sp = rematch_table(kp, u, f"snapshot_1023/label_key:{wid}")
        re_, se = rematch_table(ke, u, f"events/label_key:{wid}")
        in_fr = f_chk["unit_idx"].to_numpy()
        info[wid] = {
            "in_this_snapshot": bool(wid in set(w_all["win_id"])),
            "pred_layer": {"key": "events/snapshot_1023/label_key.parquet", "rows": int(len(kp)),
                           "by_kind": {k: int(v) for k, v in kp["utype"].value_counts().items()},
                           "id_prefix": sorted(kp["id"].str[0].unique().tolist()), "labels_found": len(lp["found"]),
                           "labels_bad_format": lp["bad"], "label_files": lp["files"],
                           "rule": "final/code/label_sample.frame·draw(층마다 글 30·댓글 30, 시드 20261022), 지침 v2 §4",
                           "rematch": sp,
                           "units_still_in_frame": int(rp["unit_idx_new"].dropna().astype(int).isin(in_fr).sum())},
            "emotion_layer": {"key": "events/label_key.parquet", "rows": int(len(ke)),
                              "by_kind": {k: int(v) for k, v in ke["utype"].value_counts().items()},
                              "id_prefix": sorted(ke["id"].str[0].unique().tolist()), "labels_found": len(le["found"]),
                              "labels_bad_format": le["bad"], "label_files": le["files"],
                              "rule": "events/code/emotion_sample.draw(층마다 글 20·댓글 20, 시드 20261023), 9감정 + 예측",
                              "rematch": se,
                              "units_still_in_frame": int(re_["unit_idx_new"].dropna().astype(int).isin(in_fr).sum())},
            "frame_now": {"글": int(nf.get((wid, "post"), 0)), "댓글": int(nf.get((wid, "comment"), 0))},
            "frame_at_draw_1023": {r.kind: int(r.N_frame) for r in st_old[st_old["win_id"] == wid].itertuples()},
            "emotion_frame_at_draw_1023": {r.kind: int(r.N_frame_draw) for r in est_old[est_old["win_id"] == wid].itertuples()},
            "reused": bool(pred_ok),
        }
        if not pred_ok:
            need_sample.append(wid)
    return need_sample, info


def write_shard(path: Path, part: pd.DataFrame) -> dict:
    part = part.sort_values("shard_pos")
    path.write_text(json.dumps([{"id": i, "text": t} for i, t in zip(part["id"], part["text_masked"])],
                               ensure_ascii=False, indent=1), "utf-8")
    return {"n": int(len(part)), "posts": int((part["utype"] == "post").sum()),
            "comments": int((part["utype"] == "comment").sum()), "windows": int(part["win_kr"].nunique()),
            "sha256": RP.sha(path)}


def main() -> int:
    t0 = datetime.now().replace(microsecond=0)
    u = pd.read_parquet(RP.RES / "units.parquet", columns=UNIT_COLS)
    u = u.reset_index().rename(columns={"index": "unit_idx"})
    w_all = FLS.load_windows()
    a, b = pd.Timestamp(RP.NEW_RANGE[0]), pd.Timestamp(RP.NEW_RANGE[1])
    need, chk = check_windows(u, w_all)
    sel = ((w_all["D"] >= a) & (w_all["D"] <= b)) | w_all["win_id"].isin(need)
    w_new = w_all[sel].reset_index(drop=True)
    # 범위 안인데 본판 완전 창이 아닌 날(첫 창 등)
    wk_raw = pd.read_parquet(RP.RES / "windows_kr.parquet")
    wk_main = wk_raw[(wk_raw["variant"] == "main") & (wk_raw["D"] >= a) & (wk_raw["D"] <= b)]
    not_complete = sorted(wk_main.loc[~wk_main["complete"].astype(bool), "win_id"].tolist())

    f, st = FLS.frame(u, w_new)
    smp, strata = FLS.draw(f, w_new)
    taken, taken_src = taken_ids()
    smp["id"] = new_ids(len(smp), taken)
    tt = [FLS.unit_text(ut, t, bb) for ut, t, bb in zip(smp["utype"], smp["title"], smp["text"])]
    smp["text_masked"] = [t for t, _ in tt]
    smp["text_cut"] = [c for _, c in tt]
    smp = smp.iloc[np.random.default_rng(FLS._seed(f"SHUFFLE:{RP.TAG}")).permutation(len(smp))].reset_index(drop=True)
    smp["shard"] = smp.index // FLS.SHARD_SIZE + 1
    smp["shard_pos"] = smp.index % FLS.SHARD_SIZE
    RP.SHARDS.mkdir(parents=True, exist_ok=True)
    for old in RP.SHARDS.glob("shard_*.json"):
        old.unlink()
    files = {}
    n_sh = int(smp["shard"].max())
    for k in range(1, n_sh + 1):
        p = RP.SHARDS / f"shard_{k:03d}.json"
        files[p.name] = write_shard(p, smp[smp["shard"] == k])

    wmeta = w_new.set_index("win_id")
    key = smp[["id", "win_kr", "utype", "unit_idx", "no", "cid", "a", "dt_kst", "post_dt_kst", "dt_conf", "src_line",
               "pwin_kr", "shard", "shard_pos", "text_cut"]].rename(columns={"win_kr": "win_id"}).copy()
    key["kind"] = np.where(key["utype"] == "post", "글", "댓글")
    key["D"] = key["win_id"].map(wmeta["D"])
    key["judge_ok"] = key["win_id"].map(wmeta["judge_ok"]).astype(bool)
    key["in_knowledge_period"] = key["dt_kst"] < FLS.KNOWLEDGE_END
    key["text_sha256"] = [tsha(t) for t in smp["text_masked"]]
    w_day, wd_info = ES.day_weights(key, f)        # 자료형을 맞추기 전에(틀과 같은 시각 해상도로) 계산
    fk = pd.read_parquet(RP.FINAL / "label_key.parquet")
    for c in fk.columns:
        if c in key.columns and c != "cid" and str(key[c].dtype) != str(fk[c].dtype):
            key[c] = key[c].astype(fk[c].dtype)
    key["cid"] = key["cid"].astype("Int64")
    key = key[fk.columns.tolist()].copy()
    key["w_day"] = w_day.to_numpy()
    key["layers"] = "pred+emo"
    kp = RP.RUN / "label_key.parquet"
    key.to_parquet(kp, index=False)

    # 검사
    assert key["id"].is_unique and not (set(key["id"]) & taken)
    assert key["unit_idx"].is_unique
    assert len(key) == int(strata["n_sample"].sum())
    assert key["win_id"].isin(set(w_new["win_id"])).all()
    assert np.allclose(key.groupby(["win_id", "utype"])["w_day"].sum(), 1.0)
    back = {r["id"] for p in RP.SHARDS.glob("shard_*.json") for r in json.loads(p.read_text("utf-8"))}
    assert back == set(key["id"])
    short = strata[strata["n_sample"] < 30]

    # 앞선 대응표를 이 스냅샷 단위에 다시 맞춘 기록(대응표는 고치지 않는다)
    prior = [("final/label_key", RP.FINAL / "label_key.parquet"), ("preview/label_key", RP.ANT / "preview" / "label_key.parquet"),
             ("events/snapshot_1023/label_key(T)", RP.SNAP1023 / "label_key.parquet"),
             ("events/label_key(감정)", RP.EV / "label_key.parquet")]
    tabs, rstats = [], []
    f_all, _ = FLS.frame(u, w_all)
    in_fr = f_all["unit_idx"].to_numpy()
    for tag, p in prior:
        k = pd.read_parquet(p)
        if tag.endswith("(T)"):
            k = k[k["id"].str.startswith("T")]
        t, s = rematch_table(k, u, tag)
        t["in_frame_new"] = t["unit_idx_new"].fillna(-1).astype("int64").isin(in_fr)
        s["not_in_frame_now"] = int((t["status"] == "ok").sum() - (t["in_frame_new"] & (t["status"] == "ok")).sum())
        tabs.append(t)
        rstats.append(s)
    rem = pd.concat(tabs, ignore_index=True)
    rem.to_parquet(RP.RUN / "prior_keys_rematch.parquet", index=False)

    summary = {
        "made": t0.isoformat(), "seed": RP.SEED, "id_prefix": RP.ID_PREFIX,
        "spec": "부록 9·11·15와 같은 규칙: W_KR 본판 완전 창마다 글 30·마감 전 댓글 30(모자라면 전부). "
                "같은 표본에 예측 층(지침 v2 §4)과 감정 층(9감정 세기 0~3, 조롱 대상)을 함께 매긴다",
        "snapshot_manifest_sha256": RP.sha(RP.RUN / "snapshot_manifest.json"),
        "units_sha256": RP.sha(RP.RES / "units.parquet"), "windows_kr_sha256": RP.sha(RP.RES / "windows_kr.parquet"),
        "windows_kr_main_complete": {"n": int(len(w_all)), "first": str(w_all["win_id"].iloc[0]),
                                     "last": str(w_all["win_id"].iloc[-1])},
        "target_range": list(RP.NEW_RANGE),
        "range_main_windows_not_complete": not_complete,
        "new_windows": {"n": int(len(w_new)), "first": str(w_new["win_id"].iloc[0]), "last": str(w_new["win_id"].iloc[-1]),
                        "list": w_new["win_id"].tolist(),
                        "kinds": {k: int(v) for k, v in w_new["kind"].value_counts().items()},
                        "judge_ok_false": sorted(w_new.loc[~w_new["judge_ok"], "win_id"].tolist()),
                        "over96": sorted(w_new.loc[w_new["over96"].astype(bool), "win_id"].tolist()),
                        "outage": sorted(w_new.loc[w_new["outage_b"], "win_id"].tolist())},
        "check_windows": chk, "check_windows_sampled_now": need,
        "frame": st, "sampled": int(len(smp)), "sampled_posts": int((smp["utype"] == "post").sum()),
        "sampled_comments": int((smp["utype"] == "comment").sum()),
        "strata_short": short.to_dict(orient="records"),
        "authors_repeated_within_window": int(key.groupby("win_id")["a"].apply(lambda s: s.duplicated().sum()).sum()),
        "in_knowledge_period_units": int(key["in_knowledge_period"].sum()),
        "knowledge_rule": "§2.2·부록 11 #1·부록 15: 7월 전 라벨(이번 표본 전부)은 예열·적합에만 쓴다. 판정 신호에는 쓰지 않는다",
        "emotion_layer_note": "부록 15는 감정 층을 창마다 40단위로 적었다. 이번에는 예측 층 60단위 표본에 감정도 함께 매긴다"
                              "(실행 지시). 감정 창 특징은 w_day(events/code/emotion_sample.day_weights 그대로)로 가중할 수 있다",
        "w_day": wd_info,
        "n_shards": n_sh, "shard_size": FLS.SHARD_SIZE, "files": files,
        "label_key": {"path": str(kp), "sha256": RP.sha(kp), "rows": int(len(key)), "columns": key.columns.tolist()},
        "taken_ids": {"n": len(taken), "sources": taken_src},
        "mask": "interim/code/blind_sample.mask_text 그대로([링크]·[날짜]·[시각]·[사건]·[숫자])",
        "text_cut": {"글": int(((smp["utype"] == "post") & smp["text_cut"]).sum()),
                     "댓글": int(((smp["utype"] == "comment") & smp["text_cut"]).sum())},
        "leak_scan_units": {k: int(sum(bool(rx.search(t)) for t in smp["text_masked"])) for k, rx in FLS.LEAK_CHECKS.items()},
        "prior_keys_rematch": {"path": str(RP.RUN / "prior_keys_rematch.parquet"), "stats": rstats,
                               "rule": "sample_new_windows.rematch와 같은 기준. 기록만 하고 앞선 대응표는 고치지 않는다"},
        "code_sha256": {"final/run2026_10_26/code/sample.py": RP.sha(Path(__file__)),
                        "final/run2026_10_26/code/run_paths.py": RP.sha(RP.HERE / "run_paths.py"),
                        "final/code/label_sample.py": RP.sha(RP.FINAL / "code" / "label_sample.py"),
                        "events/code/emotion_sample.py": RP.sha(RP.EV / "code" / "emotion_sample.py"),
                        "interim/code/blind_sample.py": RP.sha(RP.ANT / "interim" / "code" / "blind_sample.py")},
        "price_data_read": False,
    }
    (RP.RUN / "label_sample_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), "utf-8")
    strata.to_csv(RP.RUN / "label_strata.csv", index=False, encoding="utf-8-sig")
    m = json.loads(RP.META.read_text("utf-8")) if RP.META.exists() else {}
    m["label_sample"] = {k: summary[k] for k in ("made", "sampled", "n_shards", "label_key", "check_windows_sampled_now")}
    RP.META.write_text(json.dumps(m, ensure_ascii=False, indent=1, default=str), "utf-8")
    print(json.dumps({k: summary[k] for k in ("windows_kr_main_complete", "range_main_windows_not_complete", "frame",
                                              "sampled", "sampled_posts", "sampled_comments", "strata_short", "n_shards",
                                              "text_cut", "leak_scan_units", "check_windows_sampled_now", "w_day")},
                     ensure_ascii=False, indent=1, default=str))
    print(json.dumps({"new_windows": {k: v for k, v in summary["new_windows"].items() if k != "list"},
                      "check": {w: {"reused": v["reused"], "pred_rows": v["pred_layer"]["rows"],
                                    "pred_labels": v["pred_layer"]["labels_found"], "pred_rematch": v["pred_layer"]["rematch"],
                                    "emo_rows": v["emotion_layer"]["rows"], "emo_labels": v["emotion_layer"]["labels_found"],
                                    "emo_rematch": v["emotion_layer"]["rematch"], "frame_now": v["frame_now"],
                                    "frame_at_draw": v["frame_at_draw_1023"],
                                    "pred_in_frame": v["pred_layer"]["units_still_in_frame"],
                                    "emo_in_frame": v["emotion_layer"]["units_still_in_frame"]} for w, v in chk.items()},
                      "prior_rematch": rstats}, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
