"""미리보기(부록 9) 4단계: 예측 층(부록 4, 지침 v2 §4) AI 라벨 표본.

틀: preview/results/units.parquet(이 스냅샷 로더 산출)에서 D ≥ 2026-07-01인 W_KR 본판 완전 창(동결 H02·부록 3
  지식 마감 규칙: 판정 구간 2026-07-01~)마다, 자기 시각이 그 창 안인 단위 = 글 첫 줄(제목·본문) + 마감 전 댓글.
  날짜 미확정(latest) 댓글은 뺀다(sample_a와 같음). 내용이 없는 단위만 뺀다(가격·사건일 변수 미사용):
    ① 삭제 댓글(lexicon_v02.DELETED, 동결 사전 규칙) ② 사이트 알림 '인기글로 선정되어 뽐쿠폰이 발급되었습니다.'
    ③ 동영상 자리 문구 'Your browser does not support the video tag.'를 지운 뒤 빈 단위
    ④ 2026-06 날짜의 정치 전용 단위(§2.4 정치 기간, feat_activity.politics_only 규칙)
층: 창 × 글/댓글. 층마다 30개(모자라면 전부)를 단순 무작위 비복원으로 뽑는다. 시드 20261018,
  층 시드 = sha256('20261018:<창 id>:<글|댓글>') 앞 8자리.
가림: interim/code/blind_sample.mask_text(주소 → [링크], 날짜 → [날짜], 시각 → [시각], 사건 이름 → [사건],
  가격·지수·수익률·금액 숫자 → [숫자]) 그대로. 날짜·시각·번호·작성자·창은 싣지 않는다.
  글 = '제목: …\n본문: …'(본문 가린 뒤 앞 600자), 댓글 = 가린 내용 400자(emotion/code/sample_a.py와 같은 길이).
조각: 전체를 섞어(시드 'SHUFFLE') 300단위씩 label_shards/shard_NN.json([{id, text}]).
대응표: preview/label_key.parquet(id → 창·단위 키·작성자 해시·조각 위치, 라벨러에게 주지 않음).
가격은 읽지 않는다.

실행: py label_sample.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PREVIEW = HERE.parent
ANT = PREVIEW.parent
RES = PREVIEW / "results"
SHARDS = PREVIEW / "label_shards"
sys.path.insert(0, str(ANT / "interim" / "code"))
import blind_sample as B  # noqa: E402  (H02 가림 규칙 그대로)
import lexicon_v02 as L  # noqa: E402
from feat_activity import politics_only  # noqa: E402

SEED = 20261018
D_FIRST = pd.Timestamp("2026-07-01")
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


def frame() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cols = ["no", "cid", "a", "utype", "dt_kst", "post_dt_kst", "dt_conf", "src_line", "title", "text", "win_kr",
            "pwin_kr"]
    u = pd.read_parquet(RES / "units.parquet", columns=cols)
    u = u.reset_index().rename(columns={"index": "unit_idx"})
    w = pd.read_parquet(RES / "windows_kr.parquet")
    w = w[(w["variant"] == "main") & w["complete"] & (w["D"] >= D_FIRST)].sort_values("close_kst")
    w = w.reset_index(drop=True)
    w["outage_b"] = w["outage"].fillna(False).astype(bool)
    w["judge_ok"] = ~w["over96"].astype(bool) & ~w["outage_b"]
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
    june = x["dt_kst"] < D_FIRST
    m_pol = june & pd.Series([politics_only(t, b, ut) for t, b, ut in zip(x["title"], x["text"], x["utype"])],
                             index=x.index)
    drop["politics_only_june"] = int((m_pol & ~m_lat & ~m_del & ~m_sys & ~m_emp).sum())
    keep = ~(m_lat | m_del | m_sys | m_emp | m_pol)
    f = x[keep].copy()
    st["dropped"] = drop
    st["frame_units"] = int(len(f))
    st["frame_posts"] = int((f["utype"] == "post").sum())
    st["frame_comments"] = int((f["utype"] == "comment").sum())
    return f, w, st


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


def main() -> int:
    f, w, st = frame()
    smp, strata = draw(f, w)
    id_rng = np.random.default_rng(_seed("IDS"))
    ids: list[str] = []
    seen: set[str] = set()
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
        p = SHARDS / f"shard_{k:02d}.json"
        p.write_text(json.dumps([{"id": i, "text": t} for i, t in zip(part["id"], part["text_masked"])],
                                ensure_ascii=False, indent=1), "utf-8")
        files[p.name] = {"n": int(len(part)), "posts": int((part["utype"] == "post").sum()),
                         "comments": int((part["utype"] == "comment").sum()), "sha256": sha(p)}
    wmeta = w.set_index("win_id")
    key = smp[["id", "win_kr", "utype", "unit_idx", "no", "cid", "a", "dt_kst", "post_dt_kst", "dt_conf", "src_line",
               "pwin_kr", "shard", "shard_pos", "text_cut"]].rename(columns={"win_kr": "win_id"}).copy()
    key["kind"] = np.where(key["utype"] == "post", "글", "댓글")
    key["D"] = key["win_id"].map(wmeta["D"])
    key["judge_ok"] = key["win_id"].map(wmeta["judge_ok"]).astype(bool)
    key["in_knowledge_period"] = key["dt_kst"] < D_FIRST      # K2026-07-01 창의 06-30 단위(지식 마감 달)
    key["cid"] = key["cid"].astype("Int64")
    key["text_sha256"] = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in smp["text_masked"]]
    kp = PREVIEW / "label_key.parquet"
    key.to_parquet(kp, index=False)
    short = strata[strata["n_sample"] < strata["kind"].map({"글": 30, "댓글": 30})]
    summary = {
        "made": datetime.now().replace(microsecond=0).isoformat(), "seed": SEED,
        "spec": "부록 9 예측 층 표본: D ≥ 2026-07-01 W_KR 창마다 글 30·댓글 30(모자라면 전부), 지침 v2 §4로 매길 것",
        "frame": st, "sampled": int(len(smp)), "sampled_posts": int((smp["utype"] == "post").sum()),
        "sampled_comments": int((smp["utype"] == "comment").sum()),
        "strata_short": short.to_dict(orient="records"),
        "authors_repeated_within_window": int(key.groupby("win_id")["a"].apply(lambda s: s.duplicated().sum()).sum()),
        "in_knowledge_period_units": int(key["in_knowledge_period"].sum()),
        "judge_ok_false_windows": sorted(w.loc[~w["judge_ok"], "win_id"].tolist()),
        "n_shards": n_sh, "files": files, "label_key": {"path": str(kp), "sha256": sha(kp)},
        "mask": "interim/code/blind_sample.mask_text 그대로([링크]·[날짜]·[시각]·[사건]·[숫자])",
        "text_cut": {"글": int(((smp["utype"] == "post") & smp["text_cut"]).sum()),
                     "댓글": int(((smp["utype"] == "comment") & smp["text_cut"]).sum())},
        "leak_scan_units": {k: int(sum(bool(rx.search(t)) for t in smp["text_masked"])) for k, rx in LEAK_CHECKS.items()},
        "code_sha256": sha(Path(__file__)), "price_data_read": False,
    }
    (PREVIEW / "label_sample_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str),
                                                       "utf-8")
    print(json.dumps({k: summary[k] for k in ("frame", "sampled", "sampled_posts", "sampled_comments",
                                              "strata_short", "n_shards", "text_cut", "leak_scan_units",
                                              "in_knowledge_period_units", "judge_ok_false_windows")},
                     ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
