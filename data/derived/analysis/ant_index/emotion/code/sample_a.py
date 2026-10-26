"""감정 범주 라벨용 묶음 A 표본 (HYPOTHESES_addendum3.md 측정 절차 1·2단계).

- 틀: interim/results/units.parquet(로더, 10/15 스냅샷). 자기 날짜 2026-07-13~10-15.
    글 = 첫 줄 제목·본문(전부). 댓글 = 부모 글 W_KR 창 마감 전 댓글(pre_close_kr). 날짜 미확정(latest) 댓글은 뺀다.
    가격·사건일 변수는 쓰지 않는다(급락일도 빼지 않는다). 20자 미만 단위도 그대로 둔다.
- 층: 월 × 시간대(장중 09:00~15:30 / 저녁 15:30~24:00 / 심야 00:00~09:00) × 글/댓글 × 길이 삼분위.
    길이 = 공백을 뺀 글자 수(글은 제목 + 본문). 삼분위 경계는 종류별 틀 전체의 1/3·2/3 분위수.
    글 1,000·댓글 1,000을 층마다 비례 배분(최대 나머지법), 층 안은 단순 무작위 비복원.
- 가림: 날짜·시각·번호·작성자를 싣지 않는다. 본문 안 날짜 표현과 사건 이름 → [사건], 시각 → [시각],
    가격·지수·%·포인트·금액·3자리 이상 숫자·부호 붙은 수 → [숫자], 주소 → [링크].
    규칙은 interim/code/blind_sample.py(H02 가림)의 URL·시각·사건·숫자 정규식을 그대로 가져오고, 날짜 표현의 표지를
    [사건]으로 바꾸며, 서킷브레이커·사이드카·블랙먼데이를 사건에 더한다. 글은 제목 + 가린 본문 첫 600자, 댓글은 가린 내용 400자.
- 산출(emotion/): shards/shard_01~10.json(각 200: id, kind, text), shards/overlap_1~3.json(조각 01·05·09의 600을 다시 섞음),
    sample_key.parquet(정답 대응표, 라벨러에게 주지 않음), sample_a_summary.json(층 분포·가림 점검·지문).
가격·지수·수익 자료는 읽지 않는다(units.parquet에는 가격 열이 없다).

실행: py sample_a.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EMO = HERE.parent
ANT = EMO.parent
INTERIM = ANT / "interim"
sys.path.insert(0, str(INTERIM / "code"))
import blind_sample as B  # noqa: E402  (H02 가림 규칙: URL·시각·사건·숫자)

SEED = 20261017
N_POST, N_COMMENT = 1000, 1000
D0, D1 = date(2026, 7, 13), date(2026, 10, 15)
POST_BODY_MAX, COMMENT_MAX = 600, 400
N_SHARDS, SHARD_SIZE = 10, 200
OVERLAP_SHARDS = (1, 5, 9)

# ---------------------------------------------------------------- 가림

# 날짜 표현(blind_sample.DATE_RXS와 같은 꼴). 다른 점: M/D를 'N일'보다 먼저 건다('7/29일'이 '7/[사건]'으로 남지 않게),
# 'N일'은 기간·이평 뜻을 남긴다(20일선, 3일간, 3일 연속, 2일 뒤, 5일 추세 …)
DATE_RXS = [
    re.compile(r"(?<!\d)(?:19|20)\d{2}\s?[.\-/]\s?\d{1,2}(?:\s?[.\-/]\s?\d{1,2})?(?!\d)"),
    re.compile(r"(?<!\d)(?:(?:19|20)\d{2}|'?2\d)\s?년"),
    re.compile(r"(?<!\d)\d{1,2}\s?월\s?\d{1,2}\s?일"),
    re.compile(r"(?<!\d)\d{1,2}\s?월"),
    re.compile(r"(?<![\d.])\d{1,2}/\d{1,2}(?:\s?일)?(?![\d/])"),
    re.compile(r"(?<!\d)\d{1,2}\s?일(?!간|째|치|선|봉|이평|평균"
               r"|\s?(?:동안|연속|만|내|뒤|후|전|이내|넘게|정도|추세|이평|이동|평균|선|봉))"),
    re.compile(r"(?<!\d)\d{1,2}\s?분기"),
]
EXTRA_EVENT_RXS = [
    re.compile(r"(?:서킷|써킷)\s?(?:브레이커|브레이크)?"),
    re.compile(r"circuit\s?breaker", re.I),
    re.compile(r"(?:매수|매도)?\s?사이드\s?카"),
    re.compile(r"블랙\s?(?:먼데이|튜즈데이|웬즈데이|써스데이|프라이데이|[월화수목금]요일)|검은\s?[월화수목금]요일"),
    re.compile(r"black\s?(?:monday|tuesday|wednesday|thursday|friday)", re.I),
]
PLACEHOLDER_RX = re.compile(r"\[(?:사건|시각|숫자|링크)\]")
# 가림 뒤 다듬기: 범위의 왼쪽 수('80-[숫자]', '17~[사건]'), 부호 붙은 1~2자리 수('-4', '+3'), 붙은 자리표시('[숫자][숫자]')
RANGE_LEFT_RX = re.compile(r"(?<![\d.])[+\-−]?\d{1,2}(?:\.\d+)?\s?[-~～]\s?(?=\[(?:숫자|사건)\])")
SIGNED_RX = re.compile(r"(?<![\w\])])[+\-−]\d{1,2}(?:\.\d+)?(?![\d.,])")
# '지지선'·'저지선' 안의 '지선'(지방선거 줄임말)이 사건으로 바뀌지 않게, 사건 단계 동안만 낱말 잇개(U+2060)로 끊는다
JISEON_RX = re.compile(r"([지저])지선")
WJ = "\u2060"


def _keep_space(m: re.Match) -> str:
    """숫자 정규식이 앞 공백까지 삼켰으면 공백을 남긴다('호가 10% 급등' → '호가 [숫자] 급등')."""
    return (" " if m.group(0)[:1].isspace() else "") + "[숫자]"


def mask_text(s: str) -> str:
    """순서: 주소 → 날짜(→[사건]) → 시각 → 사건 → 숫자 → 다듬기.
    'N월 쇼크'는 날짜가 먼저 [사건]이 된 뒤 '[사건] 쇼크'가 한 [사건]으로 묶인다."""
    if not s:
        return ""
    s = B.URL_RX.sub("[링크]", s)
    for rx in DATE_RXS:
        s = rx.sub("[사건]", s)
    for rx in B.TIME_RXS:
        s = rx.sub("[시각]", s)
    s = JISEON_RX.sub(lambda m: m.group(1) + "지" + WJ + "선", s)
    for rx in list(B.EVENT_RXS) + EXTRA_EVENT_RXS:
        s = rx.sub("[사건]", s)
    s = s.replace(WJ, "")
    for rx in B.NUM_RXS:
        s = rx.sub(_keep_space, s)
    s = RANGE_LEFT_RX.sub("", s)
    s = SIGNED_RX.sub("[숫자]", s)
    s = re.sub(r"(?:\[숫자\]){2,}", "[숫자]", s)
    s = re.sub(r"(?:\[사건\]\s?){2,}", "[사건] ", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def cut(s: str, limit: int) -> tuple[str, bool]:
    """가린 글 앞 limit자. 자리표시([숫자] 등) 한가운데에서 자르지 않는다."""
    if len(s) <= limit:
        return s, False
    k = limit
    for m in PLACEHOLDER_RX.finditer(s):
        if m.start() < k < m.end():
            k = m.start()
            break
    return s[:k].rstrip() + " …(뒤 생략)", True


def unit_text(row) -> tuple[str, bool]:
    if row["utype"] == "post":
        title = mask_text(row["title"] or "")
        body, tr = cut(mask_text(row["text"] or ""), POST_BODY_MAX)
        return f"제목: {title or '(없음)'}\n본문: {body or '(없음)'}", tr
    body, tr = cut(mask_text(row["text"] or ""), COMMENT_MAX)
    return (body or "(내용 없음)"), tr


LEAK_CHECKS = {
    "3자리 이상 숫자": re.compile(r"(?<![\d.,])\d{3,}"),
    "% 기호": re.compile(r"[%％]"),
    "N월": re.compile(r"\d\s?월"),
    "M/D": re.compile(r"(?<![\d.])\d{1,2}/\d{1,2}(?![\d/])"),
    "시각 hh:mm": re.compile(r"\d:\d\d"),
    "연도 20xx": re.compile(r"20[12]\d"),
    "명절·선거": re.compile(r"추석|설날|선거|한글날|광복절|개천절"),
    "ADR": re.compile(r"ADR", re.I),
    "서킷·사이드카": re.compile(r"서킷|써킷|사이드\s?카"),
    "블랙먼데이": re.compile(r"블랙\s?먼데이|black\s?monday", re.I),
    "쇼크·사태": re.compile(r"쇼크|사태|파동|게이트"),
    "숫자+만·억·조": re.compile(r"\d\s?(?:만|억|조)"),
    "숫자 바로 뒤 자리표시": re.compile(r"\d\s?[-~～/]\s?\[(?:숫자|사건)\]"),
}


def leak_scan(texts: list[str]) -> dict:
    out = {k: 0 for k in LEAK_CHECKS}
    for t in texts:
        for k, rx in LEAK_CHECKS.items():
            if rx.search(t):
                out[k] += 1
    return out


# ---------------------------------------------------------------- 틀·층

def _seed(name: str) -> int:
    return int(hashlib.sha256(f"{SEED}:{name}".encode()).hexdigest()[:8], 16)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def tod_of(t: pd.Series) -> pd.Series:
    m = t.dt.hour * 60 + t.dt.minute
    return pd.Series(np.select([(m >= 540) & (m < 930), m >= 930], ["장중", "저녁"], "심야"), index=t.index)


def nchar(s: pd.Series) -> pd.Series:
    return s.fillna("").str.replace(r"\s", "", regex=True).str.len()


def load_frame() -> tuple[pd.DataFrame, dict]:
    cols = ["no", "cid", "utype", "dt_kst", "post_dt_kst", "dt_conf", "src_line", "title", "text",
            "win_kr", "pwin_kr", "win_u", "pwin_u", "pre_close_kr"]
    u = pd.read_parquet(INTERIM / "results" / "units.parquet", columns=cols)
    u = u.reset_index().rename(columns={"index": "unit_idx"})
    d = u["dt_kst"].dt.date
    inr = (d >= D0) & (d <= D1)
    is_p, is_c = u["utype"].eq("post"), u["utype"].eq("comment")
    lat = u["dt_conf"].eq("latest")
    st = {"units_total": int(len(u)), "posts_total": int(is_p.sum()), "comments_total": int(is_c.sum()),
          "posts_out_of_range": int((is_p & ~inr).sum()), "comments_out_of_range": int((is_c & ~inr).sum()),
          "comments_in_range_not_pre_close_kr": int((is_c & inr & ~u["pre_close_kr"]).sum()),
          "comments_in_range_pre_close_latest": int((is_c & inr & u["pre_close_kr"] & lat).sum())}
    keep = inr & (is_p | (is_c & u["pre_close_kr"] & ~lat))
    f = u.loc[keep].copy()
    f["kind"] = np.where(f["utype"].eq("post"), "글", "댓글")
    f["month"] = f["dt_kst"].dt.month.astype(int)
    f["tod"] = tod_of(f["dt_kst"])
    f["nchar"] = np.where(f["utype"].eq("post"), nchar(f["title"]) + nchar(f["text"]), nchar(f["text"]))
    thr = {}
    f["len3"] = 0
    for k in ("글", "댓글"):
        m = f["kind"].eq(k)
        q1, q2 = np.quantile(f.loc[m, "nchar"], [1 / 3, 2 / 3])
        thr[k] = {"q1": float(q1), "q2": float(q2), "rule": "T1 ≤ q1 < T2 ≤ q2 < T3 (공백 뺀 글자 수)"}
        f.loc[m, "len3"] = np.where(f.loc[m, "nchar"] <= q1, 1, np.where(f.loc[m, "nchar"] <= q2, 2, 3))
    f["len3"] = f["len3"].astype(int)
    f["stratum"] = (f["kind"] + "|" + f["month"].map("{:02d}월".format) + "|" + f["tod"] + "|T"
                    + f["len3"].astype(str))
    st.update({"frame_units": int(len(f)), "frame_posts": int(f["kind"].eq("글").sum()),
               "frame_comments": int(f["kind"].eq("댓글").sum()),
               "frame_posts_without_win_kr": int((f["kind"].eq("글") & f["win_kr"].isna()).sum()),
               "frame_first_date": str(f["dt_kst"].min().date()), "frame_last_date": str(f["dt_kst"].max().date()),
               "length_tercile_thresholds": thr})
    return f.reset_index(drop=True), st


def allocate(sizes: pd.Series, n: int) -> pd.Series:
    """비례 배분(최대 나머지법). 나머지가 같으면 층 이름순."""
    q = sizes / sizes.sum() * n
    base = np.floor(q).astype(int)
    rem = q - base
    for k in sorted(rem.index, key=lambda k: (-rem[k], k))[: n - int(base.sum())]:
        base[k] += 1
    base = base.clip(upper=sizes)
    assert int(base.sum()) == n, "층 크기 상한 때문에 배분 합이 모자란다"
    return base


def draw(f: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, picks = [], []
    for kind, n in (("글", N_POST), ("댓글", N_COMMENT)):
        fk = f[f["kind"].eq(kind)]
        sizes = fk.groupby("stratum").size().sort_index()
        alloc = allocate(sizes, n)
        for s in sizes.index:
            pos = fk.index[fk["stratum"].eq(s)].to_numpy()  # 틀 순서(dt, 종류, 번호, 댓글 번호)
            k = int(alloc[s])
            sel = np.sort(np.random.default_rng(_seed(s)).choice(pos, size=k, replace=False)) if k else []
            picks.extend(sel)
            rows.append({"stratum": s, "N_frame": int(sizes[s]), "share_frame": float(sizes[s] / sizes.sum()),
                         "expected": float(sizes[s] / sizes.sum() * n), "n_sample": k})
    return f.loc[picks].copy(), pd.DataFrame(rows)


# ---------------------------------------------------------------- 산출

def main() -> int:
    f, st = load_frame()
    smp, strata = draw(f)
    assert len(smp) == N_POST + N_COMMENT and smp["unit_idx"].is_unique

    id_rng = np.random.default_rng(_seed("IDS"))
    ids: list[str] = []
    while len(ids) < len(smp):
        x = f"E{int(id_rng.integers(0, 2**32)):08x}"
        if x not in ids:
            ids.append(x)
    smp["id"] = ids
    texts, trunc = zip(*(unit_text(r) for _, r in smp.iterrows()))
    smp["text_masked"], smp["text_cut"] = list(texts), list(trunc)

    order = np.random.default_rng(_seed("SHUFFLE")).permutation(len(smp))
    smp = smp.iloc[order].reset_index(drop=True)
    smp["shard"] = smp.index // SHARD_SIZE + 1
    smp["shard_pos"] = smp.index % SHARD_SIZE
    smp["overlap"] = pd.array([pd.NA] * len(smp), dtype="Int64")
    smp["overlap_pos"] = pd.array([pd.NA] * len(smp), dtype="Int64")
    ov_idx = smp.index[smp["shard"].isin(OVERLAP_SHARDS)].to_numpy()
    ov_idx = ov_idx[np.random.default_rng(_seed("OVERLAP")).permutation(len(ov_idx))]
    smp.loc[ov_idx, "overlap"] = np.arange(len(ov_idx)) // SHARD_SIZE + 1
    smp.loc[ov_idx, "overlap_pos"] = np.arange(len(ov_idx)) % SHARD_SIZE

    sd = EMO / "shards"
    sd.mkdir(parents=True, exist_ok=True)
    files = {}

    def dump(path: Path, part: pd.DataFrame):
        items = [{"id": r["id"], "kind": r["kind"], "text": r["text_masked"]} for _, r in part.iterrows()]
        path.write_text(json.dumps(items, ensure_ascii=False, indent=1), "utf-8")
        files[str(path.relative_to(EMO)).replace("\\", "/")] = {"n": len(items), "sha256": _sha(path),
                                                                "posts": int((part["kind"] == "글").sum()),
                                                                "comments": int((part["kind"] == "댓글").sum())}

    for k in range(1, N_SHARDS + 1):
        dump(sd / f"shard_{k:02d}.json", smp[smp["shard"] == k].sort_values("shard_pos"))
    for k in range(1, len(OVERLAP_SHARDS) + 1):
        part = smp[smp["overlap"] == k].sort_values("overlap_pos")
        dump(sd / f"overlap_{k}.json", part)
        files[f"shards/overlap_{k}.json"]["from_shard"] = {f"{s:02d}": int((part["shard"] == s).sum())
                                                           for s in OVERLAP_SHARDS}

    key = smp[["id", "kind", "utype", "unit_idx", "no", "cid", "src_line", "dt_kst", "post_dt_kst", "dt_conf",
               "win_kr", "pwin_kr", "win_u", "pwin_u", "month", "tod", "len3", "nchar", "stratum",
               "shard", "shard_pos", "overlap", "overlap_pos", "text_cut"]].copy()
    key["date_kst"] = key["dt_kst"].dt.date.astype(str)
    key["text_sha256"] = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in smp["text_masked"]]
    key["cid"] = key["cid"].astype("Int64")
    kp = EMO / "sample_key.parquet"
    key.to_parquet(kp, index=False)
    files["sample_key.parquet"] = {"n": int(len(key)), "sha256": _sha(kp)}

    def tab(df: pd.DataFrame, by: str) -> dict:
        return {k: {str(i): int(v) for i, v in df[df["kind"] == k][by].value_counts().sort_index().items()}
                for k in ("글", "댓글")}

    summary = {
        "made": datetime.now().replace(microsecond=0).isoformat(),
        "spec": "HYPOTHESES_addendum3.md 측정 절차 1·2 (묶음 A)", "seed": SEED,
        "seed_rule": "층마다 sha256('20261017:<층>') 앞 8자리, 섞기 'SHUFFLE', 이중 라벨 'OVERLAP', id 'IDS'",
        "frame": st,
        "strata_def": {"month": "자기 시각(KST) 달", "tod": "장중 09:00~15:30 / 저녁 15:30~24:00 / 심야 00:00~09:00",
                       "kind": "글 / 댓글", "len3": "종류별 공백 뺀 글자 수 삼분위"},
        "n_strata": {"글": int(strata["stratum"].str.startswith("글").sum()),
                     "댓글": int(strata["stratum"].str.startswith("댓글").sum()),
                     "zero_alloc": int((strata["n_sample"] == 0).sum())},
        "strata": strata.to_dict(orient="records"),
        "frame_vs_sample": {
            "month": {"frame": tab(f, "month"), "sample": tab(smp, "month")},
            "tod": {"frame": tab(f, "tod"), "sample": tab(smp, "tod")},
            "len3": {"frame": tab(f, "len3"), "sample": tab(smp, "len3")},
        },
        "sample_short_lt20": {k: int(((smp["kind"] == k) & (smp["nchar"] < 20)).sum()) for k in ("글", "댓글")},
        "sample_cut": {k: int(((smp["kind"] == k) & smp["text_cut"]).sum()) for k in ("글", "댓글")},
        "mask": {"post_body_max": POST_BODY_MAX, "comment_max": COMMENT_MAX,
                 "placeholders": ["[사건]", "[시각]", "[숫자]", "[링크]"],
                 "source": "interim/code/blind_sample.py URL_RX·TIME_RXS·EVENT_RXS·NUM_RXS + 이 파일 DATE_RXS(→[사건])·"
                           "EXTRA_EVENT_RXS(서킷브레이커·사이드카·블랙먼데이)·범위 왼쪽 수·부호 붙은 1~2자리 수",
                 "kept": "부호 없는 1~2자리 정수(2배, 3번 등), 제품·기업·인물 이름, 요일·오늘·어제, FOMC 등 정례 일정 이름",
                 "placeholder_units": {p: int(sum(p in t for t in smp["text_masked"])) for p in
                                       ("[사건]", "[시각]", "[숫자]", "[링크]")},
                 "leak_scan_units": leak_scan(list(smp["text_masked"]))},
        "files": files,
        "code_sha256": _sha(Path(__file__)),
        "price_data_read": False,
    }
    (EMO / "sample_a_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str),
                                               "utf-8")
    print(json.dumps({k: summary[k] for k in ("frame", "n_strata", "frame_vs_sample", "sample_short_lt20",
                                              "sample_cut", "mask", "files")}, ensure_ascii=False, indent=1,
                     default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
