"""부록 13 감정 층(검수 실패, 설명용) AI 라벨 표본: W_KR 283창마다 글 20 + 마감 전 댓글 20.

가격은 읽지 않는다(units.parquet·windows_kr.parquet에는 가격 열이 없다). 포럼 특징만 만든다.

틀: final/results/units.parquet(스냅샷 2025-08-20~2026-10-21 로더 산출)의 W_KR 본판 완전 창(283)마다, 자기 시각이
    그 창 안인 단위 = 글 첫 줄(제목·본문) + 마감 전 댓글. 빼는 단위는 final/code/label_sample.frame 그대로
    (날짜 미확정 댓글, 삭제 댓글, 사이트 알림, 동영상 자리 문구만 남은 빈 단위, 정치 기간의 정치 전용 단위).
층: 창 × 글/댓글. 층마다 20개.
재사용: 이미 감정 9가지 라벨이 있는 단위
    ① emotion/labels4/primary_01~10.json(지침 v1, 대응표 emotion/sample_key.parquet → interim/results/units.parquet)
    ② bottom_case/labels/shard_*.json(지침 v2, 대응표 bottom_case/label_key.parquet → preview/results/units.parquet)
    를 이 스냅샷 단위와 맞춘다(글 = 글 번호, 댓글 = 글 번호·작성자 해시·시각, 후보가 여럿이면 원문 일치로 고름).
    원문(글은 제목·본문, 댓글은 내용)이 바뀐 단위와 이 틀 밖 단위는 재사용하지 않는다. 두 곳에 모두 라벨이 있으면
    v2(②)를 쓰고 다른 id는 alt_src_id에 적는다. second_*·review200_ai는 같은 단위의 재라벨이라 쓰지 않는다.
    재사용 단위는 모두 표본에 넣는다.
새로 뽑기(시드 20261023):
    재사용이 없는 층: 층 틀에서 단순 무작위 비복원 20(모자라면 전부). 층 시드 = sha256('20261023:<창 id>:<글|댓글>') 앞 8자리.
    재사용이 있는 층: 20을 하루(08:00 경계) 틀 크기로 비례 배분(최대 나머지법)하고, 하루마다
      max(0, 배분 − 그 하루 재사용 수)를 재사용 단위를 뺀 그 하루 틀에서 뽑는다. 시드 = sha256('20261023:<창 id>:<글|댓글>:<YYYY-MM-DD>').
      재사용 수가 고르게 퍼져 있으면 '새로 뽑는 수 = 20 − 재사용 수'와 같다. 재사용이 일부 하루에 몰린 창(바닥 사례 표본은
      하루 40개씩, 주말·연휴를 낀 창의 일부 날만 덮음)에서 글자 그대로의 규칙을 쓰면 나머지 하루가 통째로 빠져(최악 층 틀의
      89%) 창 몫이 틀어지므로 이렇게 바꿨다. 층마다 글자 그대로의 규칙 수(n_new_literal_rule)도 층 표에 남긴다.
가중: 재사용 표본은 하루(08:00 경계)마다 같은 수(②)거나 부모 글이 같은 창인 댓글만(①)이라 창 안 구성이 틀과 다를 수
    있다. 그래서 창 × 글/댓글 × 하루 사후 층 가중 w_day = 틀 수 / 표본 수(층 안 합 1로 맞춤)를 대응표에 싣는다.
가림: final/code/label_sample.unit_text 그대로(interim/code/blind_sample.mask_text, 글 = '제목: …\\n본문: …' 본문
    600자, 댓글 400자). 날짜·시각·번호·작성자·창은 싣지 않는다.
조각: 새로 뽑은 단위만 섞어(시드 'SHUFFLE') 300단위씩 events/label_shards/shard_XXX.json([{id, text}]).
    id는 'V' + 8자리 16진수이고 기존 대응표(final·preview·emotion·bottom_case·interim)의 id와 겹치지 않는다.
대응표: events/label_key.parquet(새로 뽑은 단위 + 재사용 단위, 라벨러에게 주지 않음).
요약: events/results/label_sample_summary.json, 층 표 events/results/label_strata.csv.

실행: py emotion_sample.py
"""
from __future__ import annotations

import glob
import hashlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EV = HERE.parent
ANT = EV.parent
FINAL = ANT / "final"
RES = FINAL / "results"
EMO = ANT / "emotion"
BC = ANT / "bottom_case"
INTERIM = ANT / "interim"
PRE = ANT / "preview"
SHARDS = EV / "label_shards"
OUT = EV / "results"

_spec = importlib.util.spec_from_file_location("final_label_sample", FINAL / "code" / "label_sample.py")
FLS = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FLS)          # 틀·가림·자르기·점검 규칙(그 모듈 main은 돌리지 않는다)

SEED = 20261023
N_PER = {"post": 20, "comment": 20}
SHARD_SIZE = 300
OPEN_H = 8
KNOWLEDGE_END = pd.Timestamp("2026-07-01")
E9 = ["HYPE", "JOY", "MOCK", "DESP", "ANGER", "FEAR", "HOPE", "REGRET", "ENVY"]
KIND = {"post": "글", "comment": "댓글"}
ID_PREFIX = "V"
OTHER_KEYS = [FINAL / "label_key.parquet", PRE / "label_key.parquet", EMO / "sample_key.parquet",
              EMO / "review_key.parquet", EMO / "review_label_key.parquet", BC / "label_key.parquet",
              INTERIM / "results" / "blind_key.parquet"]


def _seed(name: str) -> int:
    return int(hashlib.sha256(f"{SEED}:{name}".encode()).hexdigest()[:8], 16)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tsha(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def nz(s) -> str:
    return s if isinstance(s, str) else ""


# ---------------------------------------------------------------- 기존 라벨

def load_labels(files: list[str], tag: str) -> dict[str, dict]:
    out = {}
    for p in files:
        for r in json.loads(Path(p).read_text("utf-8")):
            assert set(r["emo"]) == set(E9), f"{tag} {r['id']} 감정 키"
            assert all(isinstance(r["emo"][e], int) and 0 <= r["emo"][e] <= 3 for e in E9), f"{tag} {r['id']} 값"
            assert r["id"] not in out, f"{tag} id 중복 {r['id']}"
            out[r["id"]] = {"file": Path(p).name}
    return out


def source_units(key: pd.DataFrame, units_path: Path, tag: str) -> pd.DataFrame:
    """대응표 unit_idx(그 스냅샷 units.parquet 행 위치) → 원문·작성자·시각. 대응표 값과 대조한다."""
    cols = ["no", "cid", "a", "utype", "dt_kst", "title", "text"]
    src = pd.read_parquet(units_path, columns=cols)
    s = src.iloc[key["unit_idx"].to_numpy()].reset_index(drop=True)
    k = key.reset_index(drop=True)
    assert (s["no"].to_numpy() == k["no"].to_numpy()).all(), f"{tag} 글 번호 불일치"
    assert (s["utype"].to_numpy() == k["utype"].to_numpy()).all(), f"{tag} 단위 종류 불일치"
    assert (pd.to_datetime(s["dt_kst"]).to_numpy() == pd.to_datetime(k["dt_kst"]).to_numpy()).all(), f"{tag} 시각 불일치"
    if "a" in k:
        assert (s["a"].to_numpy() == k["a"].to_numpy()).all(), f"{tag} 작성자 불일치"
    s.insert(0, "src_id", k["id"].to_numpy())
    s["src_text_sha256"] = k["text_sha256"].to_numpy()
    return s


def match_to_final(s: pd.DataFrame, u: pd.DataFrame) -> pd.DataFrame:
    """원천 단위 → 이 스냅샷 unit_idx. 글: 번호. 댓글: 번호·작성자·시각(여럿이면 내용 일치로)."""
    uu = u[u["no"].isin(set(s["no"]))]
    posts = uu[uu["utype"] == "post"]
    p_idx = posts.groupby("no")["unit_idx"].apply(list).to_dict()
    com = uu[uu["utype"] == "comment"].assign(t_ns=lambda d: d["dt_kst"].astype("datetime64[ns]").astype("int64"))
    c_idx = com.groupby(["no", "a", "t_ns"])["unit_idx"].apply(list).to_dict()
    ttl = uu.set_index("unit_idx")["title"]
    txt = uu.set_index("unit_idx")["text"]
    s_ns = pd.to_datetime(s["dt_kst"]).astype("datetime64[ns]").astype("int64").to_numpy()
    res = []
    for j, r in enumerate(s.itertuples(index=False)):
        if r.utype == "post":
            cand = p_idx.get(r.no, [])
        else:
            cand = c_idx.get((r.no, r.a, int(s_ns[j])), [])
        same = [i for i in cand if nz(txt[i]) == nz(r.text) and (r.utype != "post" or nz(ttl[i]) == nz(r.title))]
        if not cand:
            res.append((None, "not_found", 0))
        elif same:
            res.append((same[0], "ok" if len(same) == 1 else "ok_multi_same_text", len(cand)))
        else:
            res.append((cand[0] if len(cand) == 1 else None, "text_changed", len(cand)))
    s = s.copy()
    s["unit_idx_final"] = pd.array([x[0] for x in res], dtype="Int64")
    s["match"] = [x[1] for x in res]
    s["n_cand"] = [x[2] for x in res]
    return s


# ---------------------------------------------------------------- 표본

def allocate(sizes: pd.Series, n: int) -> pd.Series:
    """비례 배분(최대 나머지법). 나머지가 같으면 앞 날짜부터."""
    q = sizes / sizes.sum() * n
    a = np.floor(q).astype(int)
    rest = n - int(a.sum())
    if rest > 0:
        frac = (q - a).sort_values(ascending=False, kind="stable")
        a.loc[frac.index[:rest]] += 1
    return a


def draw(f: pd.DataFrame, w: pd.DataFrame, reused: set[int]) -> tuple[list[int], pd.DataFrame]:
    """재사용이 없는 층: 단순 무작위 20. 재사용이 있는 층: 20을 하루(08:00 경계) 틀 크기로 비례 배분하고 하루마다
    max(0, 배분 − 그 하루 재사용 수)를 새로 뽑는다(재사용이 일부 하루에 몰려 나머지 하루가 빠지는 것을 막는다)."""
    f = f.sort_values(["dt_kst", "utype", "no", "cid"], kind="stable")
    f = f.assign(day=(f["dt_kst"] - pd.Timedelta(hours=OPEN_H)).dt.normalize())
    picks, rows = [], []
    for wid in w["win_id"]:
        for ut, kname in (("post", "글"), ("comment", "댓글")):
            st = f[(f["win_kr"] == wid) & (f["utype"] == ut)]
            ri = st["unit_idx"].isin(reused)
            r = int(ri.sum())
            literal = min(max(0, N_PER[ut] - r), int((~ri).sum()))
            if r == 0:
                pool = st.index.to_numpy()
                k = min(N_PER[ut], len(pool))
                sel = np.random.default_rng(_seed(f"{wid}:{kname}")).choice(pool, size=k, replace=False) \
                    if k else np.array([], dtype=int)
                sel_all, mode = sel.tolist(), "srs"
            else:
                alloc = allocate(st.groupby("day").size(), N_PER[ut])
                rc = st[ri].groupby("day").size().reindex(alloc.index, fill_value=0)
                sel_all = []
                for d in alloc.index:
                    pool = st.index[(st["day"] == d).to_numpy() & ~ri.to_numpy()].to_numpy()
                    k = min(max(0, int(alloc[d]) - int(rc[d])), len(pool))
                    if k:
                        rng = np.random.default_rng(_seed(f"{wid}:{kname}:{d.strftime('%Y-%m-%d')}"))
                        sel_all.extend(rng.choice(pool, size=k, replace=False).tolist())
                mode = "day_alloc"
            k = len(sel_all)
            picks.extend(sorted(sel_all))
            rows.append({"win_id": wid, "kind": kname, "N_frame": int(len(st)), "n_days": int(st["day"].nunique()),
                         "n_reused": r, "n_new": int(k), "n_new_literal_rule": int(literal),
                         "n_sample": r + int(k), "mode": mode, "short": bool(r + k < N_PER[ut])})
    return picks, pd.DataFrame(rows)


def day_weights(key: pd.DataFrame, f: pd.DataFrame) -> tuple[pd.Series, dict]:
    """창 × 글/댓글 × 하루(08:00 경계) 사후 층 가중. 층(창 × 글/댓글) 안 합 1."""
    fd = f.assign(day=(f["dt_kst"] - pd.Timedelta(hours=OPEN_H)).dt.normalize())
    N = fd.groupby(["win_kr", "utype", "day"]).size().rename("N")
    k = key.assign(day=(key["dt_kst"] - pd.Timedelta(hours=OPEN_H)).dt.normalize())
    n = k.groupby(["win_id", "utype", "day"]).size().rename("n")
    Nk = N.reindex(pd.MultiIndex.from_frame(k[["win_id", "utype", "day"]])).to_numpy()
    nk = n.reindex(pd.MultiIndex.from_frame(k[["win_id", "utype", "day"]])).to_numpy()
    raw = pd.Series(Nk / nk, index=key.index)
    w = raw / raw.groupby([key["win_id"], key["utype"]]).transform("sum")
    # 표본이 0인 하루(그 하루 몫은 가중으로 되살릴 수 없다)
    N_df = N.reset_index().rename(columns={"win_kr": "win_id"})
    have = set(map(tuple, k[["win_id", "utype", "day"]].drop_duplicates().to_numpy().tolist()))
    N_df["sampled"] = [(a, b, c) in have for a, b, c in zip(N_df["win_id"], N_df["utype"], N_df["day"])]
    tot = N_df.groupby(["win_id", "utype"])["N"].transform("sum")
    miss = N_df[~N_df["sampled"]]
    info = {"strata_with_unsampled_day": int(miss.groupby(["win_id", "utype"]).ngroups),
            "unsampled_day_cells": int(len(miss)),
            "max_frame_share_in_unsampled_days": float((miss["N"] / tot[miss.index]).groupby(
                [miss["win_id"], miss["utype"]]).sum().max()) if len(miss) else 0.0,
            "mean_frame_share_in_unsampled_days": float(
                (miss["N"] / tot[miss.index]).groupby([miss["win_id"], miss["utype"]]).sum().reindex(
                    pd.MultiIndex.from_frame(N_df[["win_id", "utype"]].drop_duplicates()), fill_value=0).mean())}
    return w, info


def main() -> int:
    t0 = datetime.now()
    cols = ["no", "cid", "a", "utype", "dt_kst", "post_dt_kst", "dt_conf", "src_line", "title", "text", "win_kr",
            "pwin_kr"]
    u = pd.read_parquet(RES / "units.parquet", columns=cols)
    u = u.reset_index().rename(columns={"index": "unit_idx"})
    w = FLS.load_windows()
    assert len(w) == 283, len(w)
    f, st = FLS.frame(u, w)
    in_frame = set(f["unit_idx"])
    unit_win = u.set_index("unit_idx")["win_kr"]

    # ---- 기존 9감정 라벨 → 이 스냅샷 단위
    e_lab = load_labels(sorted(glob.glob(str(EMO / "labels4" / "primary_*.json"))), "labels4")
    b_lab = load_labels(sorted(glob.glob(str(BC / "labels" / "shard_*.json"))), "bottom_case")
    ek = pd.read_parquet(EMO / "sample_key.parquet")
    bk = pd.read_parquet(BC / "label_key.parquet")
    assert set(e_lab) == set(ek["id"]), "labels4 primary ↔ sample_key"
    assert set(b_lab) == set(bk["id"]), "bottom_case labels ↔ label_key"
    es = source_units(ek, INTERIM / "results" / "units.parquet", "emotion")
    bs = source_units(bk, PRE / "results" / "units.parquet", "bottom_case")
    es["src"], es["guideline"] = "emotion_labels4", "v1"
    bs["src"], bs["guideline"] = "bottom_case", "v2"
    es["label_file"] = es["src_id"].map(lambda i: "emotion/labels4/" + e_lab[i]["file"])
    bs["label_file"] = bs["src_id"].map(lambda i: "bottom_case/labels/" + b_lab[i]["file"])
    m = pd.concat([match_to_final(es, u), match_to_final(bs, u)], ignore_index=True)
    ok = m["match"].str.startswith("ok")
    wset = set(w["win_id"])
    m["in_window"] = m["unit_idx_final"].map(lambda i: pd.notna(i) and unit_win.get(int(i)) in wset)
    m["in_frame"] = m["unit_idx_final"].map(lambda i: pd.notna(i) and int(i) in in_frame)
    m["reusable"] = ok & m["in_frame"]
    match_stats = {}
    for src, g in m.groupby("src"):
        gok = g["match"].str.startswith("ok")
        match_stats[src] = {"labeled_units": int(len(g)), "match": g["match"].value_counts().to_dict(),
                            "matched_but_outside_283_windows": int((gok & ~g["in_window"]).sum()),
                            "matched_in_window_but_dropped_by_frame_rules": int((gok & g["in_window"] & ~g["in_frame"]).sum()),
                            "reusable": int(g["reusable"].sum()),
                            "src_snapshot_units_sha256": sha((INTERIM if src == "emotion_labels4" else PRE)
                                                             / "results" / "units.parquet")}
    r = m[m["reusable"]].copy()
    r["unit_idx_final"] = r["unit_idx_final"].astype(int)
    dup_same_src = int(r.duplicated(["unit_idx_final", "src"]).sum())
    assert dup_same_src == 0, "한 원천 안에서 같은 단위가 두 번"
    r["pri"] = r["guideline"].map({"v2": 0, "v1": 1})
    r = r.sort_values(["unit_idx_final", "pri"])
    alt = r.groupby("unit_idx_final")["src_id"].apply(lambda s: s.iloc[1] if len(s) > 1 else None)
    r = r.drop_duplicates("unit_idx_final", keep="first").copy()
    r["alt_src_id"] = r["unit_idx_final"].map(alt)
    reused = set(r["unit_idx_final"])

    # ---- 새로 뽑기
    picks, strata = draw(f, w, reused)
    fi = f.set_index("unit_idx", drop=False)
    new = f.loc[picks].copy()
    new["src"], new["src_id"], new["guideline"], new["label_file"], new["alt_src_id"] = "new", None, None, None, None
    new["src_text_sha256"] = None
    old = fi.loc[r["unit_idx_final"].to_numpy()].copy().reset_index(drop=True)
    rr = r.reset_index(drop=True)
    for c in ("src", "src_id", "guideline", "label_file", "alt_src_id", "src_text_sha256"):
        old[c] = rr[c].to_numpy()

    # ---- id
    taken: set[str] = set()
    for p in OTHER_KEYS:
        if p.exists():
            k = pd.read_parquet(p)
            for c in ("id", "review_id", "sample_id"):
                if c in k:
                    taken |= set(k[c].dropna().astype(str))
    taken |= set(e_lab) | set(b_lab)
    assert not any(x.startswith(ID_PREFIX) for x in taken), "접두어 V가 이미 쓰임"
    id_rng = np.random.default_rng(_seed("IDS"))
    ids, seen = [], set(taken)
    while len(ids) < len(new):
        x = f"{ID_PREFIX}{int(id_rng.integers(0, 2**32)):08x}"
        if x not in seen:
            seen.add(x)
            ids.append(x)
    new["id"] = ids
    old["id"] = old["src_id"]

    # ---- 가린 글(새 단위는 조각에 싣고, 재사용 단위는 지문만 남긴다)
    for d in (new, old):
        tt = [FLS.unit_text(ut, t, b) for ut, t, b in zip(d["utype"], d["title"], d["text"])]
        d["text_masked"] = [t for t, _ in tt]
        d["text_cut"] = [c for _, c in tt]
    order = np.random.default_rng(_seed("SHUFFLE")).permutation(len(new))
    new = new.iloc[order].reset_index(drop=True)
    new["shard"] = new.index // SHARD_SIZE + 1
    new["shard_pos"] = new.index % SHARD_SIZE
    old["shard"], old["shard_pos"] = pd.NA, pd.NA
    SHARDS.mkdir(parents=True, exist_ok=True)
    for p in SHARDS.glob("shard_*.json"):
        p.unlink()
    files = {}
    n_sh = int(new["shard"].max())
    for k in range(1, n_sh + 1):
        part = new[new["shard"] == k].sort_values("shard_pos")
        p = SHARDS / f"shard_{k:03d}.json"
        p.write_text(json.dumps([{"id": i, "text": t} for i, t in zip(part["id"], part["text_masked"])],
                                ensure_ascii=False, indent=1), "utf-8")
        files[p.name] = {"n": int(len(part)), "posts": int((part["utype"] == "post").sum()),
                         "comments": int((part["utype"] == "comment").sum()),
                         "windows": int(part["win_kr"].nunique()), "sha256": sha(p)}

    # ---- 대응표
    allk = pd.concat([new, old], ignore_index=True)
    wmeta = w.set_index("win_id")
    key = allk[["id", "src", "src_id", "alt_src_id", "guideline", "label_file", "win_kr", "utype", "unit_idx", "no",
                "cid", "a", "dt_kst", "post_dt_kst", "dt_conf", "src_line", "pwin_kr", "shard", "shard_pos",
                "text_cut"]].rename(columns={"win_kr": "win_id"}).copy()
    key["kind"] = key["utype"].map(KIND)
    key["D"] = key["win_id"].map(wmeta["D"])
    key["judge_ok"] = key["win_id"].map(wmeta["judge_ok"]).astype(bool)
    key["in_knowledge_period"] = key["dt_kst"] < KNOWLEDGE_END
    key["parent_same_window"] = pd.array(np.where(key["utype"] == "comment", key["pwin_kr"] == key["win_id"], None),
                                         dtype="boolean")
    key["cid"] = key["cid"].astype("Int64")
    key["shard"] = key["shard"].astype("Int64")
    key["shard_pos"] = key["shard_pos"].astype("Int64")
    key["text_sha256"] = [tsha(t) for t in allk["text_masked"]]
    key["src_text_sha256"] = allk["src_text_sha256"].to_numpy()
    key["same_masked_text_as_src"] = pd.array(np.where(key["src"] == "new", None,
                                                       key["text_sha256"] == key["src_text_sha256"]), dtype="boolean")
    key["w_day"], wd_info = day_weights(key, f)
    key = key.sort_values(["win_id", "utype", "dt_kst", "id"], kind="stable").reset_index(drop=True)
    kp = EV / "label_key.parquet"
    key.to_parquet(kp, index=False)

    # ---- 점검
    assert key["id"].is_unique
    assert not (set(new["id"]) & taken)
    assert key["unit_idx"].is_unique
    assert key["unit_idx"].isin(in_frame).all()
    assert len(new) == int(strata["n_new"].sum()) and len(old) == int(strata["n_reused"].sum())
    cnt = key.groupby(["win_id", "kind"]).size()
    sidx = strata.set_index(["win_id", "kind"])["n_sample"]
    assert (cnt.reindex(sidx.index, fill_value=0) == sidx).all()
    assert np.allclose(key.groupby(["win_id", "utype"])["w_day"].sum(), 1.0)
    for p in SHARDS.glob("shard_*.json"):
        for it in json.loads(p.read_text("utf-8")):
            assert set(it) == {"id", "text"} and it["id"].startswith(ID_PREFIX)

    # 댓글 구성: 부모 글이 같은 창인 몫(틀 대 표본). ①은 그런 댓글만 뽑았다
    fc = f[f["utype"] == "comment"]
    kc = key[key["utype"] == "comment"]
    comp = {"frame_share_parent_same_window": float((fc["pwin_kr"] == fc["win_kr"]).mean()),
            "sample_share_parent_same_window": float(kc["parent_same_window"].astype(bool).mean()),
            "sample_share_parent_same_window_weighted": float(
                (kc["parent_same_window"].astype(bool) * kc["w_day"]).sum() / kc["w_day"].sum()),
            "by_src": {s: float(g["parent_same_window"].astype(bool).mean()) for s, g in kc.groupby("src")}}
    ra = strata[strata["n_reused"] > 0]
    summary = {
        "made": datetime.now().replace(microsecond=0).isoformat(), "seed": SEED,
        "spec": "부록 13 감정 층(검수 실패, 설명용): W_KR 본판 완전 283창마다 글 20·마감 전 댓글 20. 이미 9감정 라벨이 있는 "
                "단위(emotion/labels4 primary, bottom_case/labels)는 재사용하고 층마다 새로 뽑는 수 = max(0, 20 − 재사용 수)",
        "snapshot_manifest_sha256": sha(FINAL / "snapshot_manifest.json"),
        "units_sha256": sha(RES / "units.parquet"), "windows_kr_sha256": sha(RES / "windows_kr.parquet"),
        "windows": {"n": int(len(w)), "first": str(w["win_id"].iloc[0]), "last": str(w["win_id"].iloc[-1]),
                    "judge_ok_false": sorted(w.loc[~w["judge_ok"], "win_id"].tolist())},
        "frame": st,
        "reuse": {"by_source": match_stats,
                  "units_with_both_sources": int(r["alt_src_id"].notna().sum()),
                  "reused_units": int(len(r)), "reused_by_src": r["src"].value_counts().to_dict(),
                  "reused_by_kind": r["utype"].map(KIND).value_counts().to_dict(),
                  "reused_same_masked_text_as_src": {s: int(g.sum()) for s, g in
                                                     key.loc[key["src"] != "new"].groupby("src")[
                                                         "same_masked_text_as_src"]},
                  "masked_text_note": "emotion_labels4는 그 표본 자체 가림(sample_a.mask_text)이라 지문이 달라도 정상. "
                                      "재사용 판정은 원문(글 제목·본문, 댓글 내용) 일치로 했다",
                  "windows_with_reuse": int(ra["win_id"].nunique()),
                  "strata_reuse_over_20": int((strata["n_reused"] > 20).sum()),
                  "max_reused_in_stratum": int(strata["n_reused"].max()),
                  "not_used": ["emotion/labels4/second_1~3.json(같은 단위 재라벨)",
                               "emotion/labels4/review200_ai.json(같은 단위 검수용 재라벨)"]},
        "sampled_total": int(len(key)), "sampled_new": int(len(new)), "sampled_reused": int(len(old)),
        "new_vs_literal_rule": {"literal_rule_new_total": int(strata["n_new_literal_rule"].sum()),
                                "actual_new_total": int(strata["n_new"].sum()),
                                "strata_day_alloc": int((strata["mode"] == "day_alloc").sum()),
                                "strata_differing": int((strata["n_new"] != strata["n_new_literal_rule"]).sum()),
                                "extra_new_units": int((strata["n_new"] - strata["n_new_literal_rule"]).clip(lower=0).sum()),
                                "fewer_new_units": int((strata["n_new_literal_rule"] - strata["n_new"]).clip(lower=0).sum())},
        "sampled_posts": int((key["utype"] == "post").sum()), "sampled_comments": int((key["utype"] == "comment").sum()),
        "strata_short": strata[strata["short"]].to_dict(orient="records"),
        "comment_composition": comp,
        "w_day": {"rule": "창 × 글/댓글 × 하루(08:00 경계) 사후 층 가중 = 틀 수 / 표본 수, 층 안 합 1. 창 몫을 낼 때 쓴다",
                  **wd_info},
        "in_knowledge_period_units": int(key["in_knowledge_period"].sum()),
        "knowledge_rule": "§2.2: 라벨 모델 지식 마감(2026-06) 이전 단위(in_knowledge_period=True)는 설명·예열용",
        "authors_repeated_within_window": int(key.groupby("win_id")["a"].apply(lambda s: s.duplicated().sum()).sum()),
        "n_shards": n_sh, "shard_size": SHARD_SIZE, "files": files,
        "label_key": {"path": str(kp), "sha256": sha(kp), "rows": int(len(key))},
        "mask": "final/code/label_sample.unit_text 그대로(interim/code/blind_sample.mask_text: "
                "[링크]·[날짜]·[시각]·[사건]·[숫자])",
        "text_cut_new": {"글": int(((new["utype"] == "post") & new["text_cut"]).sum()),
                         "댓글": int(((new["utype"] == "comment") & new["text_cut"]).sum())},
        "leak_scan_new_units": {k: int(sum(bool(rx.search(t)) for t in new["text_masked"]))
                                for k, rx in FLS.LEAK_CHECKS.items()},
        "duplicate_masked_texts_new": int(new["text_masked"].duplicated().sum()),
        "code_sha256": sha(Path(__file__)), "final_label_sample_sha256": sha(FINAL / "code" / "label_sample.py"),
        "price_data_read": False,
        "runtime_sec": round((datetime.now() - t0).total_seconds(), 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "label_sample_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str),
                                                   "utf-8")
    strata.to_csv(OUT / "label_strata.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({k: summary[k] for k in ("windows", "frame", "reuse", "sampled_total", "sampled_new",
                                              "new_vs_literal_rule",
                                              "sampled_reused", "sampled_posts", "sampled_comments", "n_shards",
                                              "comment_composition", "w_day", "text_cut_new", "leak_scan_new_units",
                                              "duplicate_masked_texts_new", "in_knowledge_period_units",
                                              "runtime_sec")},
                     ensure_ascii=False, indent=1, default=str))
    print("strata_short:", len(summary["strata_short"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
