"""최종 판정 준비 3단계: 예측 층(부록 4, 지침 v2 §4) AI 라벨 표본 — 부록 9와 같은 규칙.

규칙(preview/code/label_sample.py 그대로, 바뀐 것은 창 범위·시드·조각 이름뿐):
  틀: final/results/units.parquet(이 스냅샷 로더 산출)의 W_KR 본판 완전 창마다, 자기 시각이 그 창 안인 단위
      = 글 첫 줄(제목·본문) + 마감 전 댓글. 날짜 미확정(latest) 댓글은 뺀다. 내용이 없는 단위만 뺀다:
        ① 삭제 댓글(lexicon_v02.DELETED) ② 사이트 알림 '인기글로 선정되어 뽐쿠폰이 발급되었습니다.'
        ③ 동영상 자리 문구를 지운 뒤 빈 단위 ④ §2.4 정치 기간(2025-01~06, 2026-06) 날짜의 정치 전용 단위
        (feat_activity.politics_only). 미리보기는 표본 창이 07-01 이후라 ④가 06-30 단위에만 걸렸다. 같은 규칙을
        이번 범위(2026-06 전체)에 그대로 적용한다. 가격·사건일 변수는 쓰지 않는다.
  층: 창 × 글/댓글. 층마다 30개(모자라면 전부) 단순 무작위 비복원. 시드 20261022,
      층 시드 = sha256('20261022:<창 id>:<글|댓글>') 앞 8자리.
  가림: interim/code/blind_sample.mask_text 그대로. 글 = '제목: …\\n본문: …'(본문 앞 600자), 댓글 = 400자.
  조각: 새 창 전체를 섞어(시드 'SHUFFLE') 300단위씩 final/label_shards/shard_XXX.json([{id, text}]).
      창·날짜·시각·번호·작성자는 싣지 않는다. id는 'Q' + 8자리 16진수이고 미리보기 id와 겹치지 않게 뽑는다.
  대응표: final/label_key.parquet(라벨러에게 주지 않음).

창 범위:
  새로 뽑음: D 2025-08-21~2026-06-30, 2026-10-19~10-21.
  재사용: D 2026-07-01~10-16(미리보기 72창). preview/labels/pred_*.json·preview/label_key.parquet를 그대로 쓴다.
      이 스크립트는 재사용 창을 다시 뽑지 않고, 미리보기 표본 단위가 이 스냅샷에도 같은 창·같은 가린 글로 있는지와
      두 스냅샷의 틀 크기 차이만 기록한다.
가격은 읽지 않는다.

실행: py label_sample.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
ANT = FINAL.parent
PREVIEW = ANT / "preview"
RES = FINAL / "results"
SHARDS = FINAL / "label_shards"
sys.path.insert(0, str(ANT / "interim" / "code"))
import blind_sample as B  # noqa: E402  (H02 가림 규칙 그대로)
import lexicon_v02 as L  # noqa: E402
from feat_activity import politics_only  # noqa: E402

SEED = 20261022
NEW_RANGES = [(pd.Timestamp("2025-08-21"), pd.Timestamp("2026-06-30")),
              (pd.Timestamp("2026-10-19"), pd.Timestamp("2026-10-21"))]
REUSE_RANGE = (pd.Timestamp("2026-07-01"), pd.Timestamp("2026-10-16"))
POLITICS_PERIODS = [(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-07-01")),     # §2.4 탄핵·대선
                    (pd.Timestamp("2026-06-01"), pd.Timestamp("2026-07-01"))]     # §2.4 지방선거
KNOWLEDGE_END = pd.Timestamp("2026-07-01")   # §2.2 LLM 라벨: 이날부터만 판정에 쓴다
N_PER = {"post": 30, "comment": 30}
SHARD_SIZE = 300
POST_BODY_MAX, COMMENT_MAX = 600, 400
SYSTEM_NOTICE = "인기글로 선정되어 뽐쿠폰이 발급되었습니다."
VIDEO_STUB = "Your browser does not support the video tag."
PLACEHOLDER_RX = re.compile(r"\[(?:날짜|사건|시각|숫자|링크)\]")
LEAK_CHECKS = {
    "3자리 이상 숫자": re.compile(r"(?<![\d.,])\d{3,}"),
    "% 기호": re.compile(r"[%％]"),
    "N월": re.compile(r"\d\s?월"),
    "M/D": re.compile(r"(?<![\d.])\d{1,2}/\d{1,2}(?![\d/])"),
    "시각 hh:mm": re.compile(r"\d:\d\d"),
    "연도 20xx": re.compile(r"20[12]\d"),
    "명절·선거": re.compile(r"추석|설날|선거|한글날|광복절|개천절"),
    "쇼크·사태": re.compile(r"쇼크|사태|파동|게이트"),
    "숫자+만·억·조": re.compile(r"\d\s?(?:만|억|조)"),
}


def _seed(name: str) -> int:
    return int(hashlib.sha256(f"{SEED}:{name}".encode()).hexdigest()[:8], 16)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def clean(s) -> str:
    return s.replace(VIDEO_STUB, "").strip() if isinstance(s, str) else ""


def cut(s: str, limit: int) -> tuple[str, bool]:
    if len(s) <= limit:
        return s, False
    k = limit
    for m in PLACEHOLDER_RX.finditer(s):
        if m.start() < k < m.end():
            k = m.start()
            break
    return s[:k].rstrip() + " …(뒤 생략)", True


def unit_text(utype: str, title, text) -> tuple[str, bool]:
    if utype == "post":
        t = B.mask_text(clean(title)).strip()
        b, tr = cut(B.mask_text(clean(text)).strip(), POST_BODY_MAX)
        return f"제목: {t or '(없음)'}\n본문: {b or '(없음)'}", tr
    b, tr = cut(B.mask_text(clean(text)).strip(), COMMENT_MAX)
    return b, tr


def in_ranges(d: pd.Series, ranges) -> pd.Series:
    m = pd.Series(False, index=d.index)
    for a, b in ranges:
        m |= (d >= a) & (d <= b)
    return m


def load_windows() -> pd.DataFrame:
    w = pd.read_parquet(RES / "windows_kr.parquet")
    w = w[(w["variant"] == "main") & w["complete"]].sort_values("close_kst").reset_index(drop=True)
    w["outage_b"] = w["outage"].fillna(False).astype(bool)
    w["judge_ok"] = ~w["over96"].astype(bool) & ~w["outage_b"]
    return w


def frame(u: pd.DataFrame, w: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    x = u[u["win_kr"].isin(set(w["win_id"]))].copy()
    st = {"windows": int(len(w)), "first": str(w["win_id"].iloc[0]), "last": str(w["win_id"].iloc[-1]),
          "units_in_windows": int(len(x)), "posts_in_windows": int((x["utype"] == "post").sum()),
          "comments_in_windows": int((x["utype"] == "comment").sum())}
    is_c = x["utype"] == "comment"
    drop = {}
    m_lat = is_c & (x["dt_conf"] == "latest")
    drop["latest_date_comment"] = int(m_lat.sum())
    m_del = is_c & x["text"].fillna("").map(lambda s: any(d in s for d in L.DELETED))
    drop["deleted_comment"] = int((m_del & ~m_lat).sum())
    m_sys = x["text"].fillna("").str.strip().eq(SYSTEM_NOTICE)
    drop["system_notice"] = int((m_sys & ~m_lat & ~m_del).sum())
    empty = np.where(x["utype"] == "post",
                     (x["title"].map(clean) == "") & (x["text"].map(clean) == ""),
                     x["text"].map(clean) == "")
    m_emp = pd.Series(empty, index=x.index)
    drop["empty_after_video_stub"] = int((m_emp & ~m_lat & ~m_del & ~m_sys).sum())
    pol_period = pd.Series(False, index=x.index)
    for a, b in POLITICS_PERIODS:
        pol_period |= (x["dt_kst"] >= a) & (x["dt_kst"] < b)
    m_pol = pol_period & pd.Series([politics_only(t, b, ut) for t, b, ut in zip(x["title"], x["text"], x["utype"])],
                                   index=x.index)
    drop["politics_only_in_politics_period"] = int((m_pol & ~m_lat & ~m_del & ~m_sys & ~m_emp).sum())
    keep = ~(m_lat | m_del | m_sys | m_emp | m_pol)
    f = x[keep].copy()
    st["dropped"] = drop
    st["frame_units"] = int(len(f))
    st["frame_posts"] = int((f["utype"] == "post").sum())
    st["frame_comments"] = int((f["utype"] == "comment").sum())
    return f, st


def draw(f: pd.DataFrame, w: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = f.sort_values(["dt_kst", "utype", "no", "cid"], kind="stable")
    picks, rows = [], []
    for wid in w["win_id"]:
        for ut, kname in (("post", "글"), ("comment", "댓글")):
            pos = f.index[(f["win_kr"] == wid) & (f["utype"] == ut)].to_numpy()
            k = min(N_PER[ut], len(pos))
            sel = np.sort(np.random.default_rng(_seed(f"{wid}:{kname}")).choice(pos, size=k, replace=False)) \
                if k else np.array([], dtype=int)
            picks.extend(sel.tolist())
            rows.append({"win_id": wid, "kind": kname, "N_frame": int(len(pos)), "n_sample": int(k)})
    return f.loc[picks].copy(), pd.DataFrame(rows)


def reuse_check(u: pd.DataFrame, w_all: pd.DataFrame, f_reuse: pd.DataFrame) -> dict:
    """미리보기 표본(72창)이 이 스냅샷에도 같은 창·같은 가린 글로 있는지, 두 스냅샷의 층별 틀 크기 차이."""
    pk = pd.read_parquet(PREVIEW / "label_key.parquet")
    lab_files = sorted(glob.glob(str(PREVIEW / "labels" / "pred_*.json")))
    lab_ids: set[str] = set()
    for p in lab_files:
        lab_ids |= {r["id"] for r in json.loads(Path(p).read_text("utf-8"))}
    reuse_w = w_all[in_ranges(w_all["D"], [REUSE_RANGE])]
    pw = sorted(pk["win_id"].unique())
    nw = sorted(reuse_w["win_id"])
    # 같은 단위 찾기: 글은 (no), 댓글은 (no, 작성자 해시, 시각). 이 스냅샷 단위의 가린 글 지문과 맞춘다
    uu = u[u["win_kr"].isin(set(pw))]
    posts = uu[uu["utype"] == "post"].set_index("no")
    com = uu[uu["utype"] == "comment"].copy()
    com_idx = {(n, a, pd.Timestamp(t)): i for i, (n, a, t) in enumerate(zip(com["no"], com["a"], com["dt_kst"]))}
    found = same_win = same_text = 0
    missing = []
    for r in pk.itertuples():
        if r.utype == "post":
            if r.no not in posts.index:
                missing.append(r.id)
                continue
            row = posts.loc[r.no]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            title, text, wid = row["title"], row["text"], row["win_kr"]
        else:
            j = com_idx.get((r.no, r.a, pd.Timestamp(r.dt_kst)))
            if j is None:
                missing.append(r.id)
                continue
            row = com.iloc[j]
            title, text, wid = None, row["text"], row["win_kr"]
        found += 1
        same_win += int(wid == r.win_id)
        t, _ = unit_text(r.utype, title, text)
        same_text += int(hashlib.sha256(t.encode("utf-8")).hexdigest() == r.text_sha256)
    # 층별 틀 크기: 미리보기 요약에는 창 합계만 있어 이 스냅샷 값만 적고, 합계를 미리보기 요약과 비교한다
    ps = json.loads((PREVIEW / "label_sample_summary.json").read_text("utf-8"))
    fr = f_reuse
    return {
        "reused": True,
        "windows": nw, "n_windows": len(nw),
        "same_window_set_as_preview": pw == nw,
        "preview_only_windows": sorted(set(pw) - set(nw)), "final_only_windows": sorted(set(nw) - set(pw)),
        "preview_label_key": {"path": str(PREVIEW / "label_key.parquet"), "sha256": sha(PREVIEW / "label_key.parquet"),
                              "rows": int(len(pk)), "posts": int((pk["utype"] == "post").sum()),
                              "comments": int((pk["utype"] == "comment").sum())},
        "preview_labels": {"files": [Path(p).name for p in lab_files],
                           "sha256": {Path(p).name: sha(Path(p)) for p in lab_files},
                           "ids": len(lab_ids), "ids_in_key": len(lab_ids & set(pk["id"]))},
        "preview_seed": ps.get("seed"),
        "units_found_in_this_snapshot": found, "same_window": same_win, "same_masked_text": same_text,
        "not_found_ids": missing[:50], "not_found": len(missing),
        "frame_this_snapshot": {"units": int(len(fr)), "posts": int((fr["utype"] == "post").sum()),
                                "comments": int((fr["utype"] == "comment").sum())},
        "frame_preview_snapshot": {"units": ps["frame"]["frame_units"], "posts": ps["frame"]["frame_posts"],
                                   "comments": ps["frame"]["frame_comments"]},
        "note": "재사용 창은 미리보기 스냅샷(2026-10-18 21:04) 틀에서 같은 규칙으로 뽑은 표본이다. 이 스냅샷에서 틀이 "
                "커진 몫(그 뒤 다시 받은 줄에서 확인된 마감 전 댓글 등)은 표본 기회가 없었다",
    }


def main() -> int:
    cols = ["no", "cid", "a", "utype", "dt_kst", "post_dt_kst", "dt_conf", "src_line", "title", "text", "win_kr",
            "pwin_kr"]
    u = pd.read_parquet(RES / "units.parquet", columns=cols)
    u = u.reset_index().rename(columns={"index": "unit_idx"})
    w_all = load_windows()
    w_new = w_all[in_ranges(w_all["D"], NEW_RANGES)].reset_index(drop=True)
    w_reuse = w_all[in_ranges(w_all["D"], [REUSE_RANGE])].reset_index(drop=True)
    w_other = w_all[~w_all["win_id"].isin(set(w_new["win_id"]) | set(w_reuse["win_id"]))]

    f, st = frame(u, w_new)
    f_reuse, _ = frame(u, w_reuse)
    smp, strata = draw(f, w_new)

    prev_ids = set(pd.read_parquet(PREVIEW / "label_key.parquet", columns=["id"])["id"])
    id_rng = np.random.default_rng(_seed("IDS"))
    ids: list[str] = []
    seen: set[str] = set(prev_ids)
    while len(ids) < len(smp):
        x = f"Q{int(id_rng.integers(0, 2**32)):08x}"
        if x not in seen:
            seen.add(x)
            ids.append(x)
    smp["id"] = ids
    tt = [unit_text(ut, t, b) for ut, t, b in zip(smp["utype"], smp["title"], smp["text"])]
    smp["text_masked"] = [t for t, _ in tt]
    smp["text_cut"] = [c for _, c in tt]
    order = np.random.default_rng(_seed("SHUFFLE")).permutation(len(smp))
    smp = smp.iloc[order].reset_index(drop=True)
    smp["shard"] = smp.index // SHARD_SIZE + 1
    smp["shard_pos"] = smp.index % SHARD_SIZE
    SHARDS.mkdir(parents=True, exist_ok=True)
    for old in SHARDS.glob("shard_*.json"):
        old.unlink()
    files = {}
    n_sh = int(smp["shard"].max())
    for k in range(1, n_sh + 1):
        part = smp[smp["shard"] == k].sort_values("shard_pos")
        p = SHARDS / f"shard_{k:03d}.json"
        p.write_text(json.dumps([{"id": i, "text": t} for i, t in zip(part["id"], part["text_masked"])],
                                ensure_ascii=False, indent=1), "utf-8")
        files[p.name] = {"n": int(len(part)), "posts": int((part["utype"] == "post").sum()),
                         "comments": int((part["utype"] == "comment").sum()),
                         "windows": int(part["win_kr"].nunique()), "sha256": sha(p)}
    wmeta = w_new.set_index("win_id")
    key = smp[["id", "win_kr", "utype", "unit_idx", "no", "cid", "a", "dt_kst", "post_dt_kst", "dt_conf", "src_line",
               "pwin_kr", "shard", "shard_pos", "text_cut"]].rename(columns={"win_kr": "win_id"}).copy()
    key["kind"] = np.where(key["utype"] == "post", "글", "댓글")
    key["D"] = key["win_id"].map(wmeta["D"])
    key["judge_ok"] = key["win_id"].map(wmeta["judge_ok"]).astype(bool)
    key["in_knowledge_period"] = key["dt_kst"] < KNOWLEDGE_END    # §2.2: 라벨 모델 지식 마감 달 이전 단위
    key["cid"] = key["cid"].astype("Int64")
    key["text_sha256"] = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in smp["text_masked"]]
    kp = FINAL / "label_key.parquet"
    key.to_parquet(kp, index=False)

    # 검사: id 유일·미리보기와 안 겹침, 조각 하나에 여러 창이 섞였는지, 조각 글에 창·날짜 정보 없음
    assert key["id"].is_unique and not (set(key["id"]) & prev_ids)
    assert len(key) == int(strata["n_sample"].sum())
    short = strata[strata["n_sample"] < strata["kind"].map({"글": 30, "댓글": 30})]
    reuse = reuse_check(u, w_all, f_reuse)
    summary = {
        "made": datetime.now().replace(microsecond=0).isoformat(), "seed": SEED,
        "spec": "부록 9와 같은 규칙: W_KR 본판 완전 창마다 글 30·마감 전 댓글 30(모자라면 전부), 지침 v2 §4로 매길 것",
        "snapshot_manifest_sha256": sha(FINAL / "snapshot_manifest.json"),
        "units_sha256": sha(RES / "units.parquet"), "windows_kr_sha256": sha(RES / "windows_kr.parquet"),
        "windows_kr_main_complete": {"n": int(len(w_all)), "first": str(w_all["win_id"].iloc[0]),
                                     "last": str(w_all["win_id"].iloc[-1])},
        "new_windows": {"n": int(len(w_new)), "ranges": [[str(a.date()), str(b.date())] for a, b in NEW_RANGES],
                        "first": str(w_new["win_id"].iloc[0]), "last": str(w_new["win_id"].iloc[-1]),
                        "list": w_new["win_id"].tolist(),
                        "judge_ok_false": sorted(w_new.loc[~w_new["judge_ok"], "win_id"].tolist()),
                        "over96": sorted(w_new.loc[w_new["over96"].astype(bool), "win_id"].tolist()),
                        "outage": sorted(w_new.loc[w_new["outage_b"], "win_id"].tolist())},
        "windows_not_sampled": w_other["win_id"].tolist(),
        "reuse": reuse,
        "frame": st, "sampled": int(len(smp)), "sampled_posts": int((smp["utype"] == "post").sum()),
        "sampled_comments": int((smp["utype"] == "comment").sum()),
        "strata_short": short.to_dict(orient="records"),
        "authors_repeated_within_window": int(key.groupby("win_id")["a"].apply(lambda s: s.duplicated().sum()).sum()),
        "in_knowledge_period_units": int(key["in_knowledge_period"].sum()),
        "knowledge_rule": "§2.2: 라벨 모델 지식 마감(2026-06) 다음 달 2026-07-01부터만 판정에 쓴다. "
                          "in_knowledge_period=True 단위는 판정에 쓰지 않는다(설명·예열용)",
        "n_shards": n_sh, "shard_size": SHARD_SIZE, "files": files,
        "label_key": {"path": str(kp), "sha256": sha(kp)},
        "mask": "interim/code/blind_sample.mask_text 그대로([링크]·[날짜]·[시각]·[사건]·[숫자])",
        "text_cut": {"글": int(((smp["utype"] == "post") & smp["text_cut"]).sum()),
                     "댓글": int(((smp["utype"] == "comment") & smp["text_cut"]).sum())},
        "leak_scan_units": {k: int(sum(bool(rx.search(t)) for t in smp["text_masked"])) for k, rx in LEAK_CHECKS.items()},
        "code_sha256": sha(Path(__file__)), "price_data_read": False,
    }
    (FINAL / "label_sample_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str),
                                                     "utf-8")
    strata.to_csv(FINAL / "label_strata.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({k: summary[k] for k in ("windows_kr_main_complete", "frame", "sampled", "sampled_posts",
                                              "sampled_comments", "strata_short", "n_shards", "text_cut",
                                              "leak_scan_units", "in_knowledge_period_units",
                                              "windows_not_sampled")},
                     ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: v for k, v in reuse.items() if k not in ("windows", "preview_labels")},
                     ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
