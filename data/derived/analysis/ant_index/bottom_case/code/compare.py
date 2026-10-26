"""바닥 사례 비교(설명용) 3단계: 라벨 날짜 표 → 묶음 표 → '바닥에서만 달랐던 지표' 후보 → 10월 대조 → 대표 글 → 그림 자료.

판정이 아니다. 동결 가설(HYPOTHESES*.md)과 무관한 사례 비교이고, 매수·매도 근거가 아니다. 비교할 바닥은 7월 하나(7/29·7/30)뿐이다.

입력(읽기만)
- bottom_case/results/daily.parquet(1단계), unit_hits.parquet(1단계), label_key.parquet·shards/·labels/(2단계 라벨)
- preview/results/units.parquet(대표 글·주제 묶음 작성자), preview/results/feat_topic_unit_hits.parquet(SEMI·MEM·TILT 적중)
라벨 날짜 표(results/daily_labels.parquet), 작성자당 1회
- 감정: (날짜, 작성자)마다 그 작성자 단위들의 감정별 최대 세기. share_E = 세기 ≥ 1 작성자 몫, mean_E = 최대 세기 평균(0 포함),
  int_E = 세기 ≥ 1 작성자의 평균 세기. neutral = 그날 쓴 단위가 모두 감정 0인 작성자 몫, mixed = 감정 2개 이상인 단위가 있는 작성자 몫.
- 예측: preview pred_windows와 같은 (날짜, 작성자, pred) 쌍 1회. P_bar = 예측 쌍(1~5) 평균 − 3, P_share = 예측 쌍 ÷ 전체 쌍,
  P_strong1 = pred 1 쌍 ÷ 예측 쌍. _u 접미사 = 단위 그대로(보조).
주제 묶음(results/daily_topic.parquet): 유효 단위(1단계 excl == '')에서 SEMI·MEM 작성자 몫(분모 = 1단계 n_auth_valid와 같음),
  TILT = ln(미장 단어 글 ÷ 국장 단어 글)(feat_topic H46과 같은 글 기준, 하루 창).
'바닥에서만 달랐던 지표' 후보: 바닥 값 = 7/29·7/30 두 날 값의 평균. 바닥 아닌 급락 4일(고정 목록) 값의 [최솟값, 최댓값] 밖이면 후보.
  보조 점검(후보 정의는 바꾸지 않음): 두 날 모두 같은 쪽 밖인가, 규칙 급락일 확장판 범위 밖인가, 6/23(라벨 지식 마감 달) 빼고도
  밖인가, 라벨 지표는 작성자 재표본(2,000회)에서 밖으로 남는 비율.
우연 수준 보조(점검 2026-10-19): 바닥 2일 + 급락 4일을 서로 바꾼 15가지 배정에서 같은 규칙의 범위 밖 개수(지표 사이 상관 유지),
  라벨 51일 하루씩 후보 가운데 7월 바닥과 같은 쪽인 개수와 그 뒤 더 낮은 종가 여부. 가림 점검: 조각 칸, 조각당 날짜 수,
  조각 순서↔날짜 순위 상관, 계절·요일 단서 단위 수, 미리보기 표본과 겹친 단위의 예측 값 재현(preview/label_key·labels, 읽기만).
실행: py compare.py
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BC = HERE.parent
ANT = BC.parent
PRE = ANT / "preview" / "results"
OUT = BC / "results"

EMO = ["HYPE", "JOY", "MOCK", "DESP", "ANGER", "FEAR", "HOPE", "REGRET", "ENVY"]
EMO_KO = {"HYPE": "들뜸", "JOY": "기쁨", "MOCK": "조롱", "DESP": "우울·절망", "ANGER": "분노", "FEAR": "불안·공포",
          "HOPE": "바람", "REGRET": "후회", "ENVY": "부러움"}
OPEN_H = 8
SEED = 20261019
N_BOOT = 2000
T = pd.Timestamp

BOTTOM = [T("2026-07-29"), T("2026-07-30")]
DROP = [T("2026-06-23"), T("2026-07-02"), T("2026-07-08"), T("2026-07-13")]
OCT_DAYS = [T("2026-10-15"), T("2026-10-16"), T("2026-10-17")]
REP_DAYS = [T("2026-07-29"), T("2026-07-30"), T("2026-10-15"), T("2026-10-16")]


def drange(a: str, b: str) -> list[pd.Timestamp]:
    return list(pd.date_range(a, b, freq="D"))


GROUPS = [
    ("bottom", "바닥(7/29·7/30)", BOTTOM),
    ("pre", "바닥 직전(7/20~7/28)", drange("2026-07-20", "2026-07-28")),
    ("post", "바닥 직후(7/31~8/7)", drange("2026-07-31", "2026-08-07")),
    ("drop", "바닥 아닌 급락(6/23·7/2·7/8·7/13)", DROP),
    ("sep", "9월 말 평상(9/28~10/2)", drange("2026-09-28", "2026-10-02")),
    ("oct_fall", "10월 하락 중(10/6~10/14)", drange("2026-10-06", "2026-10-14")),
    ("oct_low", "10월 저점(10/15)", [T("2026-10-15")]),
    ("oct_after", "10월 저점 뒤(10/16·10/17)", [T("2026-10-16"), T("2026-10-17")]),
]

# (열, 이름, 갈래, 분모 열, 분모 이름, 수집 방식 영향)
IND = [
    ("d_no_z20", "글 수 z(직전 20일)", "활동", "d_no", "글", "바탕 수준"),
    ("comments_per_post", "글당 마감 전 댓글", "활동", "n_posts", "글", "수집 방식"),
    ("missing_rate", "결번율(지워진 글)", "활동", "d_no", "글 번호", "수집 방식"),
    ("new30_share", "새 작성자 몫(30일)", "활동", "n_auth_nonpol", "작성자", "수집 방식"),
    ("MC_A", "반대매매·마진콜 언급 작성자 몫", "사전", "n_auth_valid", "작성자", ""),
    ("LL_US_A", "미국 레버리지 언급 작성자 몫", "사전", "n_auth_valid", "작성자", ""),
    ("LS_A", "인버스·숏 언급 작성자 몫", "사전", "n_auth_valid", "작성자", ""),
    ("KX_A", "국장 탈출 언급 작성자 몫", "사전", "n_auth_valid", "작성자", ""),
    ("ANG_A", "세력·외인 분노 언급 작성자 몫", "사전", "n_auth_valid", "작성자", ""),
    ("SEMI_A", "반도체 언급 작성자 몫", "사전", "n_auth_valid", "작성자", ""),
    ("MEM_A", "메모리 언급 작성자 몫", "사전", "n_auth_valid", "작성자", ""),
    ("TILT", "미장÷국장 글 ln", "사전", "n_tilt_posts", "글", ""),
] + [(f"share_{e}", f"{EMO_KO[e]} 몫", "라벨 감정", "n_units", "라벨 단위", "") for e in EMO] + [
    ("neutral_share", "감정 없음(중립) 몫", "라벨 감정", "n_units", "라벨 단위", ""),
    ("mixed_share", "섞인 감정 몫", "라벨 감정", "n_units", "라벨 단위", ""),
] + [(f"mean_{e}", f"{EMO_KO[e]} 평균 세기", "라벨 세기", "n_units", "라벨 단위", "") for e in EMO] + [
    ("P_bar", "예측 평균 P̄(−2~+2)", "라벨 예측", "n_pred", "예측 쌍", ""),
    ("P_share", "예측 몫", "라벨 예측", "n_pairs", "쌍", ""),
    ("P_strong1", "강한 하락(1) 몫", "라벨 예측", "n_pred", "예측 쌍", ""),
]
IND_NAME = {k: n for k, n, *_ in IND}
LABEL_FAM = ("라벨 감정", "라벨 세기", "라벨 예측")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def day_of(ts: pd.Series) -> pd.Series:
    return (ts - pd.Timedelta(hours=OPEN_H)).dt.normalize()


def ds(d: pd.Timestamp) -> str:
    return f"{d.month}/{d.day:02d}"


# ------------------------------------------------------------------ 1. 라벨
def load_labels() -> tuple[pd.DataFrame, dict]:
    rows, files = [], {}
    for f in sorted((BC / "labels").glob("shard_*.json")):
        lab = json.loads(f.read_text("utf-8"))
        sh = json.loads((BC / "shards" / f.name).read_text("utf-8"))
        assert [x["id"] for x in lab] == [x["id"] for x in sh], f.name
        files[f"labels/{f.name}"] = sha(f)
        for x in lab:
            assert set(x["emo"]) == set(EMO)
            r = {"id": x["id"], "pred": int(x["pred"]), "conf": int(x["conf"]),
                 "mock_target": x.get("mock_target"), "pred_target": x.get("pred_target")}
            for e in EMO:
                v = int(x["emo"][e])
                assert 0 <= v <= 3
                r[e] = v
            assert 0 <= r["pred"] <= 5
            rows.append(r)
    lab = pd.DataFrame(rows)
    key = pd.read_parquet(BC / "label_key.parquet")
    key["date"] = pd.to_datetime(key["date"]).dt.normalize()
    m = key.merge(lab, on="id", how="inner", validate="1:1")
    info = {"labels": len(lab), "key_rows": len(key), "merged": len(m), "unlabeled": int(len(key) - len(m)),
            "author_hash_present": bool(m["a"].notna().all()), "sha256": files,
            "pred_counts_units": {int(k): int(v) for k, v in m["pred"].value_counts().sort_index().items()}}
    assert len(m) == len(key) == len(lab)
    m["n_emo"] = (m[EMO] >= 1).sum(axis=1)
    m["neut"] = m["n_emo"] == 0
    m["mixed"] = m["n_emo"] >= 2
    return m, info


def author_frame(m: pd.DataFrame) -> pd.DataFrame:
    """(날짜, 작성자) 한 줄: 감정 최대 세기, 중립(모든 단위 0), 섞임(섞인 단위 있음), 예측 쌍 요약."""
    g = m.groupby(["date", "a"])
    A = g[EMO].max()
    A["neut"] = g["n_emo"].max() == 0
    A["mixed"] = g["mixed"].max()
    pr = m.drop_duplicates(["date", "a", "pred"])
    gp = pr.groupby(["date", "a"])
    A["k_pairs"] = gp.size()
    pp = pr[pr["pred"] > 0].groupby(["date", "a"])["pred"]
    A["k_pred"] = pp.size().reindex(A.index).fillna(0)
    A["s_pred"] = pp.sum().reindex(A.index).fillna(0)
    A["k_p1"] = pr[pr["pred"] == 1].groupby(["date", "a"]).size().reindex(A.index).fillna(0)
    return A


def metrics_from_authors(A: pd.DataFrame) -> dict:
    o = {}
    for e in EMO:
        v = A[e].to_numpy(float)
        o[f"share_{e}"] = float((v >= 1).mean())
        o[f"mean_{e}"] = float(v.mean())
        o[f"int_{e}"] = float(v[v >= 1].mean()) if (v >= 1).any() else np.nan
    o["neutral_share"] = float(A["neut"].mean())
    o["mixed_share"] = float(A["mixed"].mean())
    kp, sp, k1, kall = (A[c].sum() for c in ("k_pred", "s_pred", "k_p1", "k_pairs"))
    o["P_bar"] = float(sp / kp - 3) if kp > 0 else np.nan
    o["P_share"] = float(kp / kall) if kall > 0 else np.nan
    o["P_strong1"] = float(k1 / kp) if kp > 0 else np.nan
    o["P_strong1_all"] = float(k1 / kall) if kall > 0 else np.nan
    o["n_pairs"], o["n_pred"], o["n_pred1"] = int(kall), int(kp), int(k1)
    return o


def combo(r: pd.Series) -> str:
    on = [e for e in EMO if r[e] >= 1]
    return "+".join(EMO_KO[e] for e in on) if on else "중립"


def daily_labels(m: pd.DataFrame, A: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d, sub in m.groupby("date"):
        o = {"date": d, "n_units": len(sub), "n_posts_s": int((sub["kind"] == "글").sum()),
             "n_comments_s": int((sub["kind"] == "댓글").sum()), "n_authors": int(sub["a"].nunique()),
             "collection": sub["collection"].iloc[0], "window_complete": bool(sub["window_complete"].all()),
             "in_knowledge_period": bool(sub["in_knowledge_period"].any())}
        o.update(metrics_from_authors(A.loc[d]))
        for e in EMO:                                   # 단위 그대로(보조)
            o[f"share_{e}_u"] = float((sub[e] >= 1).mean())
        o["neutral_share_u"], o["mixed_share_u"] = float(sub["neut"].mean()), float(sub["mixed"].mean())
        pr = sub[sub["pred"] > 0]["pred"]
        o["P_bar_u"] = float(pr.mean() - 3) if len(pr) else np.nan
        o["P_share_u"] = float(len(pr) / len(sub))
        o["P_strong1_u"] = float((pr == 1).mean()) if len(pr) else np.nan
        cb = sub.apply(combo, axis=1).value_counts()
        o["top_combo"], o["top_combo_n"] = cb.index[0], int(cb.iloc[0])
        o["top_combo_emo"] = next((c for c in cb.index if c != "중립"), None)
        o["top_combo_emo_n"] = int(cb.get(o["top_combo_emo"], 0)) if o["top_combo_emo"] else 0
        rows.append(o)
    return pd.DataFrame(rows)


def boot_days(A: pd.DataFrame, days: list[pd.Timestamp], keys: list[str]) -> dict[pd.Timestamp, np.ndarray]:
    """작성자 재표본(날짜 안). 반환: 날짜 → (N_BOOT, len(keys))."""
    out = {}
    for d in days:
        a = A.loc[d]
        n = len(a)
        rng = np.random.default_rng(int(hashlib.sha256(f"{SEED}:BOOT:{d.date()}".encode()).hexdigest()[:8], 16))
        idx = rng.integers(0, n, size=(N_BOOT, n))
        cols = {}
        for e in EMO:
            v = a[e].to_numpy(float)[idx]
            cols[f"share_{e}"] = (v >= 1).mean(1)
            cols[f"mean_{e}"] = v.mean(1)
        cols["neutral_share"] = a["neut"].to_numpy(float)[idx].mean(1)
        cols["mixed_share"] = a["mixed"].to_numpy(float)[idx].mean(1)
        kp, sp = a["k_pred"].to_numpy(float)[idx].sum(1), a["s_pred"].to_numpy(float)[idx].sum(1)
        k1, kall = a["k_p1"].to_numpy(float)[idx].sum(1), a["k_pairs"].to_numpy(float)[idx].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cols["P_bar"] = np.where(kp > 0, sp / kp - 3, np.nan)
            cols["P_share"] = kp / kall
            cols["P_strong1"] = np.where(kp > 0, k1 / kp, np.nan)
        out[d] = np.column_stack([cols[k] for k in keys])
    return out


# ------------------------------------------------------------------ 2. 주제 묶음(SEMI·MEM·TILT)
def daily_topic(dates: pd.DatetimeIndex) -> pd.DataFrame:
    u = pd.read_parquet(PRE / "units.parquet", columns=["no", "cid", "utype", "a"])
    t = pd.read_parquet(PRE / "feat_topic_unit_hits.parquet", columns=["no", "cid", "utype", "mem", "semi", "tus", "tkr"])
    assert (u["no"].to_numpy() == t["no"].to_numpy()).all()
    assert (u["cid"].fillna(-1).astype("int64").to_numpy() == t["cid"].to_numpy()).all()
    h = pd.read_parquet(OUT / "unit_hits.parquet", columns=["uidx", "excl", "wdate"]).set_index("uidx")
    V = h[h["excl"] == ""].copy()
    for c in ("a", "utype"):
        V[c] = u[c].to_numpy()[V.index.to_numpy()]
    for c in ("mem", "semi", "tus", "tkr"):
        V[c] = t[c].to_numpy()[V.index.to_numpy()] > 0
    g = V.groupby(["wdate", "a"])[["semi", "mem"]].max()
    D = pd.DataFrame(index=dates)
    D["n_auth_valid_topic"] = g.groupby(level=0).size().reindex(dates)
    D["SEMI_A"] = g["semi"].groupby(level=0).mean().reindex(dates)
    D["MEM_A"] = g["mem"].groupby(level=0).mean().reindex(dates)
    P = V[V["utype"] == "post"]
    nus = P.groupby("wdate")["tus"].sum().reindex(dates).fillna(0)
    nkr = P.groupby("wdate")["tkr"].sum().reindex(dates).fillna(0)
    D["n_us_posts"], D["n_kr_posts"] = nus.astype(int), nkr.astype(int)
    D["n_tilt_posts"] = (nus + nkr).astype(int)
    with np.errstate(invalid="ignore", divide="ignore"):
        D["TILT"] = np.where((nus > 0) & (nkr > 0), np.log(nus / nkr), np.nan)
    D.index.name = "date"
    return D.reset_index()


# ------------------------------------------------------------------ 3. 대표 글
def rep_posts(m: pd.DataFrame, dl: pd.DataFrame) -> list[dict]:
    u = pd.read_parquet(PRE / "units.parquet", columns=["no", "utype", "title", "dt_kst", "post_dt_kst"])
    for c in ("dt_kst", "post_dt_kst"):
        u[c] = u[c].astype("datetime64[ns]")
    h = pd.read_parquet(OUT / "unit_hits.parquet", columns=["uidx", "excl"]).set_index("uidx")["excl"]
    u["excl"] = h.reindex(u.index).fillna("").to_numpy()
    u["wdate"] = day_of(u["dt_kst"])
    u["pwdate"] = day_of(u["post_dt_kst"])
    out = []
    for d in REP_DAYS:
        close = d + pd.Timedelta(days=1, hours=OPEN_H)
        posts = u[(u["utype"] == "post") & (u["wdate"] == d)]
        cm = u[(u["utype"] == "comment") & (u["pwdate"] == d) & (u["dt_kst"] < close)]
        cnt = cm.groupby("no").size()
        p = posts.assign(n_c=posts["no"].map(cnt).fillna(0).astype(int)).sort_values(["n_c", "no"], ascending=[False, True])
        items = []
        for _, r in p.head(5).iterrows():
            ti = re.sub(r"@\S+", "", str(r["title"] or "")).strip()
            ti = ti if len(ti) <= 40 else ti[:39] + "…"
            items.append({"title": ti, "n_comments_preclose": int(r["n_c"]), "hour_kst": int(pd.Timestamp(r["dt_kst"]).hour),
                          "excl": r["excl"] or ""})
        row = dl[dl["date"] == d].iloc[0]
        sub = m[m["date"] == d]
        cb = sub.apply(combo, axis=1).value_counts()
        out.append({"date": str(d.date()), "n_posts_day": int(len(posts)), "top5": items,
                    "label_units": int(len(sub)),
                    "combos_top3": [{"combo": k, "n": int(v), "share": round(v / len(sub), 3)} for k, v in cb.head(3).items()],
                    "top_emotion_combo": {"combo": row["top_combo_emo"], "n": int(row["top_combo_emo_n"])}})
    return out


# ------------------------------------------------------------------ 3b. 보조 점검: 장중만(08:00~15:30), 직전 20일 대비 배수
LEX = ["MC", "LL_US", "LS", "KX", "ANG"]
CUT_H = 15.5                      # KRX 정규장 마감 15:30


def intraday_values(keys: list[str], m: pd.DataFrame, days: list[pd.Timestamp]) -> dict[str, dict[str, float]]:
    """하루 창 가운데 그날 08:00~15:30 단위만(장 마감 때 알 수 있던 몫). 사전 묶음과 라벨 지표만."""
    out: dict[str, dict[str, float]] = {k: {} for k in keys}
    lex_keys = [k for k in keys if k.endswith("_A") and k[:-2] in LEX]
    if lex_keys:
        u = pd.read_parquet(PRE / "units.parquet", columns=["a", "dt_kst"])
        u["dt_kst"] = u["dt_kst"].astype("datetime64[ns]")
        h = pd.read_parquet(OUT / "unit_hits.parquet").set_index("uidx")
        V = h[(h["excl"] == "") & h["wdate"].isin(days)].copy()
        ix = V.index.to_numpy()
        V["a"], V["dt"] = u["a"].to_numpy()[ix], u["dt_kst"].to_numpy()[ix]
        V = V[V["dt"] < V["wdate"] + pd.Timedelta(hours=CUT_H)]
        g = V.groupby(["wdate", "a"])[[f"h_{k[:-2]}" for k in lex_keys]].max()
        r = g.groupby(level=0).mean()
        for k in lex_keys:
            for d in days:
                out[k][str(d.date())] = float(r.loc[d, f"h_{k[:-2]}"]) if d in r.index else np.nan
    lab_keys = [k for k in keys if k.startswith(("share_", "mean_", "neutral", "mixed", "P_"))]
    if lab_keys:
        mm = m[m["dt_kst"] < m["date"] + pd.Timedelta(hours=CUT_H)]
        A = author_frame(mm)
        for d in days:
            if d not in A.index.get_level_values(0):
                continue
            o = metrics_from_authors(A.loc[d])
            for k in lab_keys:
                out[k][str(d.date())] = o[k]
            out.setdefault("_n_units", {})[str(d.date())] = int((mm["date"] == d).sum())
    return out


def base20_ratio(s: pd.Series, complete: pd.Series) -> pd.Series:
    """값 ÷ 직전 20개 하루 창 평균(자기 제외, 20개가 다 완전할 때만)."""
    x, ok = s.to_numpy(float), complete.to_numpy(bool) & np.isfinite(s.to_numpy(float))
    r = np.full(len(x), np.nan)
    for i in range(20, len(x)):
        if ok[i - 20:i].all():
            b = x[i - 20:i].mean()
            r[i] = x[i] / b if b > 0 else np.nan
    return pd.Series(r, index=s.index)


# ------------------------------------------------------------------ 3c. 가림 점검(점검 2026-10-19): 라벨러가 날짜를 알 수 있었나
CLUES = {"여름(장마감·중복 빼고)": r"폭염|더위|더운|무더위|여름|휴가|열대야", "가을·연휴": r"가을|추워지|쌀쌀|단풍|연휴|국경절",
         "요일": r"[월화수목금토일]요일|[월화수목금]욜"}


def blind_check() -> dict:
    key = pd.read_parquet(BC / "label_key.parquet", columns=["id", "date", "block", "shard", "shard_pos", "unit_idx"])
    texts, fields = {}, set()
    for f in sorted((BC / "shards").glob("shard_*.json")):
        for x in json.loads(f.read_text("utf-8")):
            fields |= set(x)
            texts[x["id"]] = x["text"]
    order = (key["shard"] - 1) * 300 + key["shard_pos"]
    kd = key.set_index("id")
    clues = {}
    for nm, rx in CLUES.items():
        r_ = re.compile(rx)
        hit = [i for i, t in texts.items() if r_.search(t)]
        clues[nm] = {"units": len(hit), "by_block": {k: int(v) for k, v in kd.loc[hit, "block"].value_counts().items()} if hit else {}}
    # 미리보기 표본과 겹친 단위: 예측 값 재현(다른 실행, 같은 글)
    pk = pd.read_parquet(ANT / "preview" / "label_key.parquet", columns=["id", "unit_idx"])
    pl = [x for f in sorted((ANT / "preview" / "labels").glob("pred_*.json")) for x in json.loads(f.read_text("utf-8"))]
    pl = pd.DataFrame(pl)[["id", "pred"]].merge(pk, on="id")
    bl = []
    for f in sorted((BC / "labels").glob("shard_*.json")):
        bl += [{"id": x["id"], "pred": int(x["pred"])} for x in json.loads(f.read_text("utf-8"))]
    ov = pd.DataFrame(bl).merge(key[["id", "unit_idx"]], on="id").merge(pl[["unit_idx", "pred"]], on="unit_idx", suffixes=("", "_preview"))
    return {"shard_fields": sorted(fields), "n_units": len(texts),
            "days_per_shard": [int(key.groupby("shard")["date"].nunique().min()), int(key.groupby("shard")["date"].nunique().max())],
            "order_date_rank_corr": round(float(np.corrcoef(key["date"].rank(), order.rank())[0, 1]), 4),
            "clues": clues, "preview_overlap_units": int(len(ov)), "preview_overlap_pred_same": int((ov["pred"] == ov["pred_preview"]).sum())}


# ------------------------------------------------------------------ 4. 비교
def chance_rates(n_ref: int, rng: np.random.Generator, n_sim: int = 200_000) -> dict:
    """서로 바꿔도 같은 분포(iid)일 때: ① 두 날 평균이 n_ref일 범위 밖 ② 두 날 모두 같은 쪽 밖 ③ 한 날이 범위 밖(2/(n+1))."""
    x = rng.standard_normal((n_sim, n_ref + 2))
    ref, b = x[:, :n_ref], x[:, n_ref:]
    lo, hi = ref.min(1), ref.max(1)
    mb = b.mean(1)
    p_mean = float(((mb > hi) | (mb < lo)).mean())
    p_both = 2 / math.comb(n_ref + 2, 2)
    return {"n_ref": n_ref, "p_mean_outside_sim": round(p_mean, 4), "p_both_same_side_exact": round(p_both, 4),
            "p_single_outside_exact": round(2 / (n_ref + 1), 4)}


def main() -> int:
    t0 = datetime.now()
    D = pd.read_parquet(OUT / "daily.parquet")
    D["date"] = pd.to_datetime(D["date"]).dt.normalize()
    dates = pd.DatetimeIndex(D["date"])

    m, linfo = load_labels()
    A = author_frame(m)
    DL = daily_labels(m, A)
    DL.to_parquet(OUT / "daily_labels.parquet", index=False)

    TP = daily_topic(dates)
    TP.to_parquet(OUT / "daily_topic.parquet", index=False)
    chk_auth = (TP.set_index("date")["n_auth_valid_topic"] - D.set_index("date")["n_auth_valid"]).abs().max()

    X = D.merge(TP, on="date", how="left").merge(DL.drop(columns=["collection"]), on="date", how="left")
    X = X.set_index("date")

    # ---- 묶음 표
    groups = []
    for gid, gname, gd in GROUPS:
        gd = [d for d in gd if d in X.index]
        rec = {"id": gid, "name": gname, "days": [str(d.date()) for d in gd],
               "non_trading_days": [str(d.date()) for d in gd if not bool(X.loc[d, "krx_session"])],
               "collection": sorted(set(X.loc[gd, "collection"])), "ind": {}}
        for k, nm, fam, den, den_nm, _ in IND:
            v = X.loc[gd, k]
            ok = v.notna()
            rec["ind"][k] = {"value": float(v[ok].mean()) if ok.any() else None, "n_days": int(ok.sum()),
                             "n_units": int(X.loc[gd, den][ok].fillna(0).sum()) if ok.any() else 0}
        rec["kospi_ret_mean"] = float(X.loc[gd, "kospi_ret"].dropna().mean()) if X.loc[gd, "kospi_ret"].notna().any() else None
        rec["kospi_dd_ref_min"] = float(X.loc[gd, "kospi_dd_ref"].min()) if X.loc[gd, "kospi_dd_ref"].notna().any() else None
        groups.append(rec)

    # ---- 후보 찾기
    rule_days = [T(s) for s in json.loads((OUT / "daily_meta.json").read_text("utf-8"))["checks"]["rule_drop5_later_lower_all_days"]]
    ext = [d for d in rule_days if d not in BOTTOM and d < T("2026-10-01")]       # 7/29(바닥 묶음)·10/13(10월 대상) 제외
    lab_days = set(DL["date"])
    label_keys = [k for k, _, fam, *_ in IND if fam in LABEL_FAM]
    B = boot_days(A, BOTTOM + DROP + OCT_DAYS, label_keys)
    rng = np.random.default_rng(SEED)
    chance = {n: chance_rates(n, rng) for n in (3, 4)}

    seps = []
    for k, nm, fam, den, den_nm, coll in IND:
        s = X[k]
        dv = {str(d.date()): (None if pd.isna(s.get(d)) else float(s[d])) for d in DROP}
        dvals = np.array([v for v in dv.values() if v is not None], float)
        b29, b30 = s.get(BOTTOM[0]), s.get(BOTTOM[1])
        bvals = [float(x) for x in (b29, b30) if pd.notna(x)]
        rec = {"key": k, "name": nm, "family": fam, "drop_values": dv, "bottom_values": {"2026-07-29": None if pd.isna(b29) else float(b29),
               "2026-07-30": None if pd.isna(b30) else float(b30)}, "n_drop": int(len(dvals))}
        if len(dvals) < 3 or not bvals:
            rec.update({"status": "값 부족"})
            seps.append(rec)
            continue
        lo, hi = float(dvals.min()), float(dvals.max())
        bm = float(np.mean(bvals))
        side = "위" if bm > hi else ("아래" if bm < lo else None)
        rec.update({"bottom_mean": bm, "drop_min": lo, "drop_max": hi, "drop_mean": float(dvals.mean()),
                    "outside": side is not None, "side": side,
                    "excess_over_range": (bm - hi) / (hi - lo) if side == "위" and hi > lo else ((lo - bm) / (hi - lo) if side == "아래" and hi > lo else 0.0)})
        both = len(bvals) == 2 and ((min(bvals) > hi) or (max(bvals) < lo))
        rec["both_days_outside_same_side"] = bool(both)
        # 확장 급락일(규칙 기계 적용, 라벨 지표는 라벨 있는 날만)
        ev = [s.get(d) for d in ext if (fam not in LABEL_FAM or d in lab_days)]
        ev = np.array([x for x in ev if pd.notna(x)], float)
        ext_days = [str(d.date()) for d in ext if (fam not in LABEL_FAM or d in lab_days) and pd.notna(s.get(d))]
        rec["ext_days"] = ext_days
        rec["ext_outside_same_side"] = bool(side == "위" and bm > ev.max() or side == "아래" and bm < ev.min()) if len(ev) else None
        rec["ext_min"], rec["ext_max"] = (float(ev.min()), float(ev.max())) if len(ev) else (None, None)
        # 6/23 빼고
        d3 = np.array([v for kk, v in dv.items() if v is not None and kk != "2026-06-23"], float)
        rec["outside_without_0623"] = bool(side == "위" and bm > d3.max() or side == "아래" and bm < d3.min()) if (side and len(d3) >= 3) else (None if side else False)
        # 전체 날에서 바닥 평균보다 더 극단인 날
        allv = s.dropna()
        allv = allv[~allv.index.isin(BOTTOM)]
        if side == "위":
            more = allv[allv >= bm].sort_values(ascending=False)
        elif side == "아래":
            more = allv[allv <= bm].sort_values()
        else:
            more = allv.iloc[0:0]
        rec["n_days_all"] = int(len(allv) + len(bvals))
        rec["n_days_more_extreme"] = int(len(more))
        rec["more_extreme_days"] = [str(d.date()) for d in more.index[:6]]
        # 재표본
        if fam in LABEL_FAM and side:
            j = label_keys.index(k)
            bb = (B[BOTTOM[0]][:, j] + B[BOTTOM[1]][:, j]) / 2
            dd = np.column_stack([B[d][:, j] for d in DROP])
            with np.errstate(invalid="ignore"):
                keep = bb > np.nanmax(dd, 1) if side == "위" else bb < np.nanmin(dd, 1)
            rec["boot_keep_outside"] = float(np.nanmean(keep))
            rec["bottom_ci95"] = [float(np.nanpercentile(bb, 2.5)), float(np.nanpercentile(bb, 97.5))]
        rec["collection_caveat"] = coll
        rec["status"] = "후보" if side else "범위 안"
        seps.append(rec)

    checked = [r for r in seps if r["status"] != "값 부족"]
    cands = [r for r in checked if r["outside"]]
    exp_mean = sum(chance[r["n_drop"]]["p_mean_outside_sim"] for r in checked)
    exp_both = sum(chance[r["n_drop"]]["p_both_same_side_exact"] for r in checked)

    # ---- 우연 수준 보조(점검 2026-10-19, 후보 정의는 그대로)
    # (a) 바닥 2일 + 급락 4일 = 6일 가운데 2일을 '바닥'으로 고르는 15가지 배정. 지표 사이 상관을 그대로 둔 채 같은 규칙으로 센다.
    def n_outside(bdays: list, ddays: list) -> tuple[int, int]:
        n_o = n_b = 0
        for k_, *_ in IND:
            s_ = X[k_]
            b_ = [float(s_[d]) for d in bdays if pd.notna(s_.get(d))]
            d_ = [float(s_[d]) for d in ddays if pd.notna(s_.get(d))]
            if len(b_) < 2 or len(d_) < 3:
                continue
            m_, lo_, hi_ = float(np.mean(b_)), min(d_), max(d_)
            n_o += int(m_ > hi_ or m_ < lo_)
            n_b += int(min(b_) > hi_ or max(b_) < lo_)
        return n_o, n_b
    six = BOTTOM + DROP
    exch = []
    for pair in [(a_, b_) for i_, a_ in enumerate(six) for b_ in six[i_ + 1:]]:
        o_, b2_ = n_outside(list(pair), [d for d in six if d not in pair])
        exch.append({"pair": [str(d.date()) for d in pair], "n_out": o_, "n_both": b2_})
    obs_o, obs_b = n_outside(BOTTOM, DROP)
    assert (obs_o, obs_b) == (len(cands), sum(r.get("both_days_outside_same_side", False) for r in checked))
    exchange = {"n_assign": len(exch), "observed": {"n_out": obs_o, "n_both": obs_b},
                "n_assign_out_ge_obs": sum(e["n_out"] >= obs_o for e in exch),
                "n_assign_both_ge_obs": sum(e["n_both"] >= obs_b for e in exch),
                "max_non_bottom": max((e for e in exch if not set(e["pair"]) & {"2026-07-29", "2026-07-30"}),
                                      key=lambda e: e["n_out"]),
                "assignments": exch}
    # (b) 라벨 51일 하루씩: 후보 11개 가운데 7월 바닥과 같은 쪽(급락 4일 범위 밖, 같은 방향)인 개수. 후보를 바닥에서 골랐으니
    #     바닥 이틀은 정의상 크다. 다른 날이 얼마나 자주 '바닥처럼' 보이는지와 그 뒤 종가(10/16까지: 첫 더 낮은 종가 날짜,
    #     다음 15거래일 최저 종가 대비)를 본다. 뒤 가격을 쓴 묘사 열이다.
    kc = X["kospi_last_close"]
    lab_sorted = sorted(lab_days)
    same = []
    for d in lab_sorted:
        n_ = sum(1 for r in cands if pd.notna(X.loc[d, r["key"]]) and
                 ((r["side"] == "위" and X.loc[d, r["key"]] > r["drop_max"]) or (r["side"] == "아래" and X.loc[d, r["key"]] < r["drop_min"])))
        later = X.loc[(X.index > d) & X["krx_session"], "kospi_close"]
        low = later[later < kc[d]]
        w15 = later.iloc[:15]                                  # 다음 15거래일(10/16까지 있는 만큼)
        same.append({"date": str(d.date()), "n_same_side": int(n_), "kospi_dd_ref": float(X.loc[d, "kospi_dd_ref"]),
                     "first_lower_close_date": str(low.index[0].date()) if len(low) else None,
                     "min_ret_15td": float(w15.min() / kc[d] - 1) if len(w15) else None,
                     "n_later_trading_days": int(len(later))})
    oth = [s_ for s_ in same if T(s_["date"]) not in BOTTOM]
    placebo_days = {"n_cands": len(cands), "n_other_days": len(oth), "by_day": same,
                    "other_ge7": [s_ for s_ in oth if s_["n_same_side"] >= 7],
                    "other_ge5_n": sum(s_["n_same_side"] >= 5 for s_ in oth),
                    "other_median": float(np.median([s_["n_same_side"] for s_ in oth]))}

    # ---- 10월 대조(후보만, 지표마다 따로)
    octo = []
    fall = drange("2026-10-06", "2026-10-14")
    for r in cands:
        k = r["key"]
        s = X[k]
        lo, hi, bm, dmean, side = r["drop_min"], r["drop_max"], r["bottom_mean"], r["drop_mean"], r["side"]

        def where(v):
            if v is None or pd.isna(v):
                return "값 없음"
            if lo <= v <= hi:
                return "급락일 범위 안"
            out_up = v > hi
            return "바닥 쪽" if (out_up and side == "위") or (not out_up and side == "아래") else "반대쪽"

        def nearer(v):
            if v is None or pd.isna(v):
                return None
            return "바닥" if abs(v - bm) < abs(v - dmean) else "급락일 평균"
        vf = s.reindex(fall).dropna()
        cells = {"oct_fall_mean": {"value": float(vf.mean()) if len(vf) else None, "n_days": int(len(vf))}}
        for d in OCT_DAYS:
            v = s.get(d)
            cells[str(d.date())] = {"value": None if pd.isna(v) else float(v)}
        for c in cells.values():
            c["where"], c["nearer"] = where(c["value"]), nearer(c["value"])
        if r["family"] in LABEL_FAM:
            j = label_keys.index(k)
            for d in OCT_DAYS:
                bv = B[d][:, j]
                with np.errstate(invalid="ignore"):
                    c = cells[str(d.date())]
                    c["boot_share_bottom_side"] = float(np.nanmean(bv > hi if side == "위" else bv < lo))
        octo.append({"key": k, "name": r["name"], "family": r["family"], "side": side, "bottom_mean": bm,
                     "drop_min": lo, "drop_max": hi, "drop_mean": dmean, "cells": cells, "caveat": r["collection_caveat"]})

    # ---- 보조 점검(후보 정의는 그대로): 장중만, 직전 20일 대비 배수
    idays = BOTTOM + DROP + [T("2026-10-15"), T("2026-10-16")]
    IV = intraday_values([r["key"] for r in cands], m, idays)

    def range_test(vals: dict[str, float], side: str) -> dict:
        b = [vals.get(str(d.date())) for d in BOTTOM]
        dd = [vals.get(str(d.date())) for d in DROP]
        b = [x for x in b if x is not None and np.isfinite(x)]
        dd = [x for x in dd if x is not None and np.isfinite(x)]
        if len(b) < 2 or len(dd) < 3:
            return {"outside_same_side": None}
        bm, lo, hi = float(np.mean(b)), min(dd), max(dd)
        res_ = bm > hi if side == "위" else bm < lo
        return {"bottom_mean": bm, "drop_min": lo, "drop_max": hi, "outside_same_side": bool(res_)}
    for r in cands:
        k = r["key"]
        vals = IV.get(k, {})
        if vals:
            rt = range_test(vals, r["side"])
            r["intraday"] = {"values": vals, **rt}
            for o in octo:
                if o["key"] == k:
                    for dd_ in ("2026-10-15", "2026-10-16"):
                        v = vals.get(dd_)
                        if v is None or not np.isfinite(v):
                            pos_ = "값 없음"
                        elif not rt.get("outside_same_side"):
                            pos_ = "기준 없음"
                        else:
                            lo, hi = rt["drop_min"], rt["drop_max"]
                            pos_ = "급락일 범위 안" if lo <= v <= hi else (
                                "바닥 쪽" if (v > hi) == (r["side"] == "위") else "반대쪽")
                        o["cells"][dd_]["intraday_value"], o["cells"][dd_]["intraday_where"] = v, pos_
        if r["family"] == "사전":
            br = base20_ratio(X[k], X["complete"])
            vals = {str(d.date()): (None if pd.isna(br.get(d)) else float(br[d])) for d in BOTTOM + DROP + OCT_DAYS}
            r["base20_ratio"] = {"values": vals, **range_test(vals, r["side"])}
            for o in octo:
                if o["key"] == k:
                    rt = r["base20_ratio"]
                    for d in OCT_DAYS:
                        v = vals.get(str(d.date()))
                        c = o["cells"][str(d.date())]
                        c["base20_ratio"] = v
                        if v is None:
                            c["base20_where"] = "값 없음"
                        elif not rt.get("outside_same_side"):
                            c["base20_where"] = "기준 없음"
                        else:
                            lo, hi = rt["drop_min"], rt["drop_max"]
                            c["base20_where"] = "급락일 범위 안" if lo <= v <= hi else (
                                "바닥 쪽" if (v > hi) == (r["side"] == "위") else "반대쪽")
    intraday_units = IV.get("_n_units", {})

    # ---- 깊이 맥락(설명용): 7월 하락 구간 거래일에서 사전 후보와 고점 대비 낙폭의 상관, 배수판 6~9월 최댓값
    depth = {}
    for lo_, hi_ in (("2026-06-22", "2026-07-30"), ("2026-06-22", "2026-09-22")):
        w = X[(X["krx_session"]) & (X.index >= T(lo_)) & (X.index <= T(hi_))]
        depth[f"{lo_}~{hi_}"] = {"n_trading_days": int(len(w)), **{k: round(float(np.corrcoef(w[k], w["kospi_dd_ref"])[0, 1]), 3)
                                                                  for k in ("MC_A", "LS_A", "ANG_A")}}
    b20 = {}
    for k in ("MC_A", "LS_A", "ANG_A"):
        br = base20_ratio(X[k], X["complete"])
        js = br[(br.index >= T("2026-06-01")) & (br.index <= T("2026-09-30"))].dropna()
        b20[k] = {"max_jun_sep": round(float(js.max()), 3), "max_jun_sep_date": str(js.idxmax().date()),
                  "oct_13_16": {str(d.date()): round(float(br[d]), 3) for d in drange("2026-10-13", "2026-10-16")}}

    reps = rep_posts(m, DL)

    # ---- 맥락(가격)
    ctx = []
    for d in BOTTOM + DROP + [T("2026-10-13"), T("2026-10-14")] + OCT_DAYS:
        ctx.append({"date": str(d.date()), "kospi_close": None if pd.isna(X.loc[d, "kospi_close"]) else float(X.loc[d, "kospi_close"]),
                    "kospi_ret": None if pd.isna(X.loc[d, "kospi_ret"]) else float(X.loc[d, "kospi_ret"]),
                    "kospi_dd_ref": None if pd.isna(X.loc[d, "kospi_dd_ref"]) else float(X.loc[d, "kospi_dd_ref"]),
                    "ref_peak": str(pd.Timestamp(X.loc[d, "kospi_ref_peak_date"]).date()) if pd.notna(X.loc[d, "kospi_ref_peak_date"]) else None,
                    "weekday": int(X.loc[d, "weekday"]), "collection": X.loc[d, "collection"], "complete": bool(X.loc[d, "complete"]),
                    "d_no": float(X.loc[d, "d_no"]), "d_no_base20_mean": None if pd.isna(X.loc[d, "d_no_base20_mean"]) else float(X.loc[d, "d_no_base20_mean"])})

    # ---- 그림 자료
    CH = [("d_no_z20", "글 수 z(직전 20일)"), ("new30_share", "새 작성자 몫(30일)"), ("MC_A", "반대매매·마진콜 언급 몫"),
          ("LS_A", "인버스·숏 언급 몫"), ("ANG_A", "세력·외인 분노 언급 몫"), ("share_FEAR", "불안·공포 몫(라벨)"),
          ("share_DESP", "우울·절망 몫(라벨)"), ("P_bar", "예측 평균 P̄(라벨)")]

    def nn(v):
        return None if pd.isna(v) else round(float(v), 5)
    chart = {"title": "개미 게시판 바닥 사례 비교(설명용)", "made": datetime.now().isoformat(timespec="seconds"),
             "note": "판정 아님. 하루 창 = 그날 08:00~다음 날 07:59. 라벨 지표는 라벨 51일만 값이 있다. 10/13부터 실시간 수집, 그 전은 소급. 10/17은 16시간 창.",
             "dates": [str(d.date()) for d in X.index],
             "kospi_close": [nn(v) for v in X["kospi_close"]],
             "kospi_last_close": [nn(v) for v in X["kospi_last_close"]],
             "series": [{"key": k, "label": lb, "values": [nn(v) for v in X[k]]} for k, lb in CH],
             "events": [{"date": "2026-07-29", "label": "바닥 묶음"}, {"date": "2026-07-30", "label": "7월 저점"},
                        *[{"date": str(d.date()), "label": "바닥 아닌 급락"} for d in DROP],
                        {"date": "2026-10-15", "label": "10월 저점(지금까지)"}],
             "collection": list(X["collection"]), "complete": [bool(v) for v in X["complete"]]}
    (OUT / "chart.json").write_text(json.dumps(chart, ensure_ascii=False, indent=1), "utf-8")

    res = {"made": datetime.now().isoformat(timespec="seconds"),
           "purpose": "바닥 사례 비교(설명용). 판정 아님, 매수·매도 근거 아님. 바닥 사례 1개(7/29·7/30)",
           "label_info": linfo, "check_n_auth_valid_topic_vs_daily_max_abs_diff": float(chk_auth),
           "n_indicators": len(IND), "n_checked": len(checked), "n_candidates": len(cands),
           "chance": chance, "expected_outside_by_chance_mean_rule": round(exp_mean, 2),
           "expected_both_days_by_chance": round(exp_both, 2),
           "n_both_days_outside": int(sum(r.get("both_days_outside_same_side", False) for r in checked)),
           "chance_exchange_6days": exchange, "placebo_days_same_side": placebo_days, "blind_check": blind_check(),
           "ext_rule_days_used": [str(d.date()) for d in ext],
           "intraday_cut": "그날 08:00 ~ 15:30 직전(KRX 정규장 마감)", "intraday_label_units": intraday_units,
           "depth_corr_dd_ref": depth, "base20_ratio_summary": b20,
           "groups": groups, "separators": seps, "october": octo, "rep_posts": reps, "context": ctx}
    (OUT / "compare_tables.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), "utf-8")
    meta = {"made": res["made"], "code_sha256": sha(Path(__file__)),
            "inputs_sha256": {p.name: sha(p) for p in (OUT / "daily.parquet", OUT / "unit_hits.parquet", BC / "label_key.parquet",
                                                       PRE / "units.parquet", PRE / "feat_topic_unit_hits.parquet")},
            "preview_labels_sha256": {f"preview/{p.relative_to(ANT / 'preview').as_posix()}": sha(p) for p in
                                      [ANT / "preview" / "label_key.parquet", *sorted((ANT / "preview" / "labels").glob("pred_*.json"))]},
            "labels_sha256": linfo["sha256"],
            "outputs_sha256": {p: sha(OUT / p) for p in ("daily_labels.parquet", "daily_topic.parquet", "chart.json", "compare_tables.json")},
            "n_boot": N_BOOT, "seed": SEED, "runtime_sec": round((datetime.now() - t0).total_seconds(), 1)}
    (OUT / "compare_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps({k: res[k] for k in ("n_indicators", "n_checked", "n_candidates", "expected_outside_by_chance_mean_rule",
                                          "expected_both_days_by_chance", "n_both_days_outside", "check_n_auth_valid_topic_vs_daily_max_abs_diff")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
