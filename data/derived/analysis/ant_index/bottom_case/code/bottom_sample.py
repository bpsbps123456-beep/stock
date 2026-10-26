"""바닥 사례 비교(설명용) 2단계: 두 층 라벨(emotion/guideline_v2.md)용 가림 표본.

판정이 아니다. 동결 가설과 무관하고, 가격은 읽지 않는다(units.parquet에는 가격 열이 없다).

틀: preview/results/units.parquet에서 자기 시각이 아래 날짜의 '하루 창'(KST 그날 08:00 ~ 다음 날 08:00 직전)인 단위
  = 글 첫 줄(제목·본문) + 댓글. 날짜 51개(가격으로 고정한 사건 둘레, 결과를 보고 바꾸지 않는다):
    6/22~6/24, 7/1~7/3, 7/7~7/9, 7/12~7/14, 7/20~8/7, 9/28~10/17.
  빼는 단위는 preview/code/label_sample.py와 같다: ① 날짜 미확정(latest) 댓글 ② 삭제 댓글(lexicon_v02.DELETED)
    ③ 사이트 알림 '인기글로 선정되어 뽐쿠폰이 발급되었습니다.' ④ 동영상 자리 문구를 지운 뒤 빈 단위
    ⑤ 2026-06 날짜의 정치 전용 단위(feat_activity.politics_only).
층: 날짜 × 글/댓글. 층마다 40개(모자라면 전부)를 단순 무작위 비복원으로 뽑는다. 시드 20261019,
  층 시드 = sha256('20261019:<YYYY-MM-DD>:<글|댓글>') 앞 8자리.
가림·글자 수: preview/code/label_sample.unit_text를 그대로 불러 쓴다(interim/code/blind_sample.mask_text, 글 = '제목: …\\n본문: …'
  본문 600자, 댓글 400자). 날짜·시각·번호·작성자·창은 싣지 않는다.
조각: 전체를 섞어(시드 'SHUFFLE') 300단위씩 shards/shard_NN.json([{id, text}]). 라벨러가 날짜를 모르게 날짜가 섞인다.
대응표: bottom_case/label_key.parquet(id → 날짜·묶음·단위 키·작성자 해시·조각 위치, 라벨러에게 주지 않음).
요약: results/label_sample_summary.json.

실행: py bottom_sample.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BC = HERE.parent
ANT = BC.parent
PRE = ANT / "preview"
RES = PRE / "results"
SHARDS = BC / "shards"
OUT = BC / "results"

_spec = importlib.util.spec_from_file_location("preview_label_sample", PRE / "code" / "label_sample.py")
PLS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PLS)          # 가림·자르기·점검 규칙(그 모듈 main은 돌리지 않는다)
B, L, politics_only = PLS.B, PLS.L, PLS.politics_only

SEED = 20261019
N_PER = {"post": 40, "comment": 40}
SHARD_SIZE = 300
OPEN_H = 8
D_KNOW = pd.Timestamp("2026-07-01")     # 라벨 AI 지식 마감 달(preview 규칙) 표시용
RT_START = pd.Timestamp("2026-10-13")
T_END = pd.Timestamp("2026-10-18 00:00:00")
BLOCKS = [("J1_0622", "2026-06-22", "2026-06-24"), ("J2_0701", "2026-07-01", "2026-07-03"),
          ("J3_0707", "2026-07-07", "2026-07-09"), ("J4_0712", "2026-07-12", "2026-07-14"),
          ("JB_0720", "2026-07-20", "2026-08-07"), ("OB_0928", "2026-09-28", "2026-10-17")]
EVENTS = {"2026-07-30": "7월 저점", "2026-10-15": "10월 저점(지금까지)", "2026-06-23": "바닥 아닌 급락",
          "2026-07-02": "바닥 아닌 급락", "2026-07-08": "바닥 아닌 급락", "2026-07-13": "바닥 아닌 급락"}


def _seed(name: str) -> int:
    return int(hashlib.sha256(f"{SEED}:{name}".encode()).hexdigest()[:8], 16)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def target_days() -> dict[pd.Timestamp, str]:
    out = {}
    for name, a, b in BLOCKS:
        for d in pd.date_range(a, b, freq="D"):
            out[d] = name
    return out


def frame(days: dict) -> tuple[pd.DataFrame, dict]:
    cols = ["no", "cid", "a", "utype", "dt_kst", "post_dt_kst", "dt_conf", "src_line", "title", "text"]
    u = pd.read_parquet(RES / "units.parquet", columns=cols)
    u = u.reset_index().rename(columns={"index": "unit_idx"})
    for c in ("dt_kst", "post_dt_kst"):
        u[c] = u[c].astype("datetime64[ns]")
    u["date"] = (u["dt_kst"] - pd.Timedelta(hours=OPEN_H)).dt.normalize()
    x = u[u["date"].isin(set(days))].copy()
    st = {"days": len(days), "units_in_days": int(len(x)), "posts_in_days": int((x["utype"] == "post").sum()),
          "comments_in_days": int((x["utype"] == "comment").sum())}
    is_c = x["utype"] == "comment"
    drop = {}
    m_lat = is_c & (x["dt_conf"] == "latest")
    drop["latest_date_comment"] = int(m_lat.sum())
    m_del = is_c & x["text"].fillna("").map(lambda s: any(d in s for d in L.DELETED))
    drop["deleted_comment"] = int((m_del & ~m_lat).sum())
    m_sys = x["text"].fillna("").str.strip().eq(PLS.SYSTEM_NOTICE)
    drop["system_notice"] = int((m_sys & ~m_lat & ~m_del).sum())
    empty = np.where(x["utype"] == "post",
                     (x["title"].map(PLS.clean) == "") & (x["text"].map(PLS.clean) == ""),
                     x["text"].map(PLS.clean) == "")
    m_emp = pd.Series(empty, index=x.index)
    drop["empty_after_video_stub"] = int((m_emp & ~m_lat & ~m_del & ~m_sys).sum())
    june = x["dt_kst"] < D_KNOW
    m_pol = june & pd.Series([politics_only(t, b, ut) for t, b, ut in zip(x["title"], x["text"], x["utype"])],
                             index=x.index)
    drop["politics_only_june"] = int((m_pol & ~m_lat & ~m_del & ~m_sys & ~m_emp).sum())
    f = x[~(m_lat | m_del | m_sys | m_emp | m_pol)].copy()
    st["dropped"] = drop
    st["frame_units"] = int(len(f))
    st["frame_posts"] = int((f["utype"] == "post").sum())
    st["frame_comments"] = int((f["utype"] == "comment").sum())
    return f, st


def draw(f: pd.DataFrame, days: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = f.sort_values(["dt_kst", "utype", "no", "cid"], kind="stable")
    picks, rows = [], []
    for d in sorted(days):
        ds = d.strftime("%Y-%m-%d")
        for ut, kname in (("post", "글"), ("comment", "댓글")):
            pos = f.index[(f["date"] == d) & (f["utype"] == ut)].to_numpy()
            k = min(N_PER[ut], len(pos))
            sel = np.sort(np.random.default_rng(_seed(f"{ds}:{kname}")).choice(pos, size=k, replace=False)) \
                if k else np.array([], dtype=int)
            picks.extend(sel.tolist())
            rows.append({"date": ds, "block": days[d], "kind": kname, "N_frame": int(len(pos)), "n_sample": int(k)})
    return f.loc[picks].copy(), pd.DataFrame(rows)


def main() -> int:
    days = target_days()
    f, st = frame(days)
    smp, strata = draw(f, days)
    id_rng = np.random.default_rng(_seed("IDS"))
    ids: list[str] = []
    seen: set[str] = set()
    while len(ids) < len(smp):
        x = f"B{int(id_rng.integers(0, 2**32)):08x}"
        if x not in seen:
            seen.add(x)
            ids.append(x)
    smp["id"] = ids
    tt = [PLS.unit_text(ut, t, b) for ut, t, b in zip(smp["utype"], smp["title"], smp["text"])]
    smp["text_masked"] = [t for t, _ in tt]
    smp["text_cut"] = [c for _, c in tt]
    order = np.random.default_rng(_seed("SHUFFLE")).permutation(len(smp))
    smp = smp.iloc[order].reset_index(drop=True)
    smp["shard"] = smp.index // SHARD_SIZE + 1
    smp["shard_pos"] = smp.index % SHARD_SIZE
    SHARDS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for old in SHARDS.glob("shard_*.json"):
        old.unlink()
    files = {}
    n_sh = int(smp["shard"].max())
    for k in range(1, n_sh + 1):
        part = smp[smp["shard"] == k].sort_values("shard_pos")
        p = SHARDS / f"shard_{k:02d}.json"
        p.write_text(json.dumps([{"id": i, "text": t} for i, t in zip(part["id"], part["text_masked"])],
                                ensure_ascii=False, indent=1), "utf-8")
        files[p.name] = {"n": int(len(part)), "posts": int((part["utype"] == "post").sum()),
                         "comments": int((part["utype"] == "comment").sum()), "days": int(part["date"].nunique()),
                         "sha256": sha(p)}
    prev_key = pd.read_parquet(PRE / "label_key.parquet", columns=["unit_idx"])
    key = smp[["id", "date", "utype", "unit_idx", "no", "cid", "a", "dt_kst", "post_dt_kst", "dt_conf", "src_line",
               "shard", "shard_pos", "text_cut"]].copy()
    key["kind"] = np.where(key["utype"] == "post", "글", "댓글")
    key["block"] = key["date"].map(days)
    ds = key["date"].dt.strftime("%Y-%m-%d")
    key["event_ko"] = ds.map(EVENTS).fillna("")
    key["collection"] = np.where(key["date"] + pd.Timedelta(hours=OPEN_H) >= RT_START, "실시간", "소급")
    key["window_complete"] = key["date"] + pd.Timedelta(hours=OPEN_H + 24) <= T_END
    key["in_knowledge_period"] = key["dt_kst"] < D_KNOW
    key["in_preview_sample"] = key["unit_idx"].isin(set(prev_key["unit_idx"]))
    key["cid"] = key["cid"].astype("Int64")
    key["text_sha256"] = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in smp["text_masked"]]
    kp = BC / "label_key.parquet"
    key.to_parquet(kp, index=False)
    short = strata[strata["n_sample"] < 40]
    pos_in_shard = smp.groupby("shard")["date"].nunique()
    summary = {
        "made": datetime.now().replace(microsecond=0).isoformat(), "seed": SEED,
        "spec": "바닥 사례 비교(설명용, 판정 아님): 하루 창(그날 08:00~다음 날 07:59) 51일 × 글 40·댓글 40, "
                "지침 v2 두 층(1층 분위기 감정, 2층 예측)으로 매길 것",
        "frame": st, "sampled": int(len(smp)), "sampled_posts": int((smp["utype"] == "post").sum()),
        "sampled_comments": int((smp["utype"] == "comment").sum()),
        "strata_short": short.to_dict(orient="records"),
        "per_block": key.groupby("block").agg(days=("date", "nunique"), units=("id", "size")).reset_index()
        .to_dict(orient="records"),
        "authors_repeated_within_day": int(key.groupby("date")["a"].apply(lambda s: s.duplicated().sum()).sum()),
        "in_knowledge_period_units": int(key["in_knowledge_period"].sum()),
        "in_preview_sample_units": int(key["in_preview_sample"].sum()),
        "realtime_units": int((key["collection"] == "실시간").sum()),
        "incomplete_window_units": int((~key["window_complete"]).sum()),
        "days_per_shard": {"min": int(pos_in_shard.min()), "max": int(pos_in_shard.max())},
        "n_shards": n_sh, "files": files, "label_key": {"path": str(kp), "sha256": sha(kp)},
        "mask": "preview/code/label_sample.unit_text 그대로(interim/code/blind_sample.mask_text: [링크]·[날짜]·[시각]·[사건]·[숫자])",
        "text_cut": {"글": int(((smp["utype"] == "post") & smp["text_cut"]).sum()),
                     "댓글": int(((smp["utype"] == "comment") & smp["text_cut"]).sum())},
        "leak_scan_units": {k: int(sum(bool(rx.search(t)) for t in smp["text_masked"])) for k, rx in
                            PLS.LEAK_CHECKS.items()},
        "duplicate_masked_texts": int(smp["text_masked"].duplicated().sum()),
        "code_sha256": sha(Path(__file__)), "preview_label_sample_sha256": sha(PRE / "code" / "label_sample.py"),
        "inputs_sha256": {"units.parquet": sha(RES / "units.parquet")}, "price_data_read": False,
    }
    (OUT / "label_sample_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str),
                                                   "utf-8")
    print(json.dumps({k: summary[k] for k in ("frame", "sampled", "sampled_posts", "sampled_comments",
                                              "strata_short", "per_block", "n_shards", "text_cut", "leak_scan_units",
                                              "in_knowledge_period_units", "in_preview_sample_units",
                                              "realtime_units", "incomplete_window_units", "days_per_shard",
                                              "duplicate_masked_texts")},
                     ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
