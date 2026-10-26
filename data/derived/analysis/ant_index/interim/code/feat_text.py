"""개미 지수 텍스트 특징(감정·어휘) — HYPOTHESES.md §2·§3과 부록 1 정의를 그대로 구현한다.

입력(읽기만):
  results/units.parquet, results/windows_kr.parquet, results/windows_us.parquet  (로더 산출, 10/15 스냅샷)
  사전 v0.2: lexicon_v02_final.load_v02()  (results/lexicon_v02.json 단어 목록과 같은지 검사한다)
  가격: H24 사건 표지용 KOSPI·SK하이닉스 '지난' 종가만(yfinance, 마지막 창의 P일까지). 사건 뒤 수익은 계산하지 않는다.
산출:
  results/feat_text.parquet        행 = 본판 창(W_KR·W_U, 완전한 창), 열 = 특징
  results/feat_text_units.parquet  단위별 사전 적중 요약(다음 단계 재사용용)
  results/feat_text_meta.json      지문·결측률·사건 목록
해석 기록: interim/notes_feat_text.md,  열 사전: results/feat_text.md

규칙(공통): 창 안 단위만(댓글은 자기 시각이 든 창 = 마감 전 댓글만), 조회·추천·다른의견·ncomment는 쓰지 않는다.
rz8·z60·pct250은 직전 창으로만(자기 제외) 계산하고 최소 창 수가 모자라면 결측으로 둔다(채우지 않는다).
실행: py feat_text.py   (이 폴더에서)
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lexicon_v02 as L  # noqa: E402
from lexicon_v02_final import load_v02  # noqa: E402

INTERIM = Path(__file__).resolve().parents[1]
RESULTS = INTERIM / "results"

# ---------------------------------------------------------------- 고정값(문서 값 그대로, 문서에 없는 값은 notes에 근거)
Z_N, Z_MIN = 60, 40            # z60: 직전 60창, 40창 미만이면 결측(§2.0)
P_N, P_MIN = 250, 120          # pct250: 직전 250거래일, 최소 120(§2.0)
RZ_N = 8                       # rz8: 같은 요일 직전 8창(연휴 뒤 창 제외). 8창이 다 차야 계산(notes)
RZ_CLIP = 6.0
MAD_K = 1.4826                 # MAD → 정규 SD 눈금(notes)
DSD_MIN_AUTH = 20              # D_SD: 표(점수 ≠ 0 작성자) 20개 미만이면 결측(H06·H10)
DSD_DRAWS = 1000
BOT_DAYS, BOT_LOOKBACK = 15, 30  # 봇·연재: 직전 30일 안 같은 제목 패턴 15일 이상(§2.2)
ATT_DAYS = 20                  # H12 관심 기준: 같은 등록 시간대의 직전 20일 중앙값
TOP_SHARE = 0.10               # 관심 상위 10%
SH_BODY, SH_TITLE = 20, 25     # H54 짧은 글
EV_Z, EV_MERGE = -2.0, 5       # H24 사건: r/σ60 ≤ −2, 5거래일 안 사건은 묶는다
EV_BASE = 20                   # R = F_A(사건 창) − 직전 20창 평균
WEEK_N = 52                    # 부록 1 주 단계: 직전 52주
SEED = 20261016

STRATA = ["E", "F", "PX", "CAP", "HES", "FO", "RES", "ANG", "LL_US", "LL_KR", "LS", "MC", "SEMI"]
POST_ONLY = {"SEMI"}           # lexicon_v02.STRATUM_SCOPE: SEMI는 글 단위
NEWS_PRICE_KW = ["시황", "마감", "브리핑"]       # H16 '가격 보도'(지수·종목명 + 수치%·포인트, 시황·마감·브리핑)
NEWS_EVENT_KW = ["정책", "실적", "수주", "공시"]  # H16 '사건 뉴스'
URL_RX = re.compile(r"https?://\S+|www\.\S+|\b[\w.-]+\.(?:com|co\.kr|kr|net|org|io)(?:/\S*)?", re.I)
MARK_RXS = [re.compile(r"ㅋ{4,}"), re.compile(r"[ㅠㅜ]{2,}"), re.compile(r"ㄷ{2,}|ㄷ\d|덜덜|후덜")]  # M1b·M3b·M4
TITLE_PAT_RX = re.compile(r"[^가-힣A-Za-z]")
KST_SLOTS = (("day", 9 * 60, 15 * 60 + 30), ("eve", 15 * 60 + 30, 22 * 60))   # 나머지 = night(22:00~09:00)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ================================================================ 1. 단위별 사전 적중

def check_lexicon(an_words_path: Path = RESULTS / "lexicon_v02.json") -> dict:
    """load_v02()가 쓰는 단어 = lexicon_v02.json 단어 목록인지 확인한다(층별 집합 비교)."""
    v02 = json.loads(an_words_path.read_text("utf-8"))
    full = json.loads((RESULTS / "lexicon_v02_full.json").read_text("utf-8"))
    rule_names = {w for ws in full["rule_items_candidate"].values() for w in ws}
    removed = {st: {L._compact(w) for w in ws} for st, ws in full["removed"].items()}   # 매처는 공백 지운 키로 막는다
    bad = {}
    cand: dict[str, list[str]] = {}
    for g, b, st, t, words in L.SPEC:
        cand.setdefault(st, []).extend(words)
    for st, words in cand.items():
        keep = {L._compact(w) for w in words if L._compact(w) not in removed.get(st, set())}
        want = {L._compact(w) for w in v02.get(st, []) if w not in rule_names}
        if keep != want:
            bad[st] = {"matcher_only": sorted(keep - want), "json_only": sorted(want - keep)}
    return {"lexicon_v02_json_sha256": sha256_file(an_words_path),
            "lexicon_v02_full_sha256": sha256_file(RESULTS / "lexicon_v02_full.json"),
            "candidate_sha256": L.lexicon_sha256(), "word_set_mismatch": bad,
            "status": full["status"]}


def classify_news(title: str, body: str) -> str:
    """H16: 뉴스형 글을 첫 저장 줄의 제목 + 본문 첫 줄로 나눈다. 가격 보도가 먼저 잡는다."""
    first = next((x for x in (body or "").split("\n") if x.strip()), "")
    c = L._compact((title or "") + " " + first)
    if any(k in c for k in NEWS_PRICE_KW):
        return "price"
    if L.PRICE_NUM_RX.search(c) and (any(n in c for n in L.PRICE_NAMES) or any(k in c for k in L.ALIAS_KEYS)):
        return "price"
    if any(k in c for k in NEWS_EVENT_KW):
        return "event"
    return "other"


def match_units(u: pd.DataFrame) -> pd.DataFrame:
    """창(국내·미국 본판)에 든 단위마다 v0.2 적중 요약. 뉴스형 글은 H16용으로 뉴스 거르기를 끈 판도 센다."""
    an = load_v02()
    sel = u[u["win_kr"].notna() | u["win_u"].notna()]
    recs = []
    orig_is_news = L.is_news
    market_words = [m.lower() for m in L.MARKET]
    for ridx, ut, ti, tx in zip(sel.index, sel["utype"], sel["title"], sel["text"]):
        title = (ti if isinstance(ti, str) else "") if ut == "post" else None
        body = tx if isinstance(tx, str) else ""
        r = an(title, body, ut)
        full = (title + "\n" + body) if ut == "post" else body
        rec = {"uidx": ridx, "excl": r.excluded}
        strata = {h.stratum for h in r.hits}
        for st in STRATA:
            rec[f"h_{st}"] = (st in strata) and not (st in POST_ONLY and ut != "post")
        cnt = Counter((h.stratum, h.word) for h in r.hits if h.stratum in ("E", "F"))
        rec["E_cnt3"] = sum(min(c, 3) for (st, _), c in cnt.items() if st == "E")
        rec["F_cnt3"] = sum(min(c, 3) for (st, _), c in cnt.items() if st == "F")
        rec["h_LL_US_noNVDL"] = any(h.stratum == "LL_US" and h.word != "NVDL" for h in r.hits)
        comp = L._compact(full)
        rec["h_FO_mkt"] = rec["h_FO"] and any(m in comp for m in market_words)
        rec["px_unit"] = rec["h_PX"] or r.price_dropped > 0
        rec["score"] = float(r.score) if (ut == "post" and r.score is not None and not r.excluded) else np.nan
        rec["mark"] = any(rx.search(full) for rx in MARK_RXS) or full.count("!") >= 2
        rec["eoj"] = len(full.split())
        rec["news_cat"] = ""
        rec["nE"] = rec["nF"] = False
        if r.excluded == "news":
            rec["news_cat"] = classify_news(title, body)
            L.is_news = lambda t, b: False      # H16: 뉴스형 글의 E·F를 세려고 거르기만 잠시 끈다
            try:
                r2 = an(title, body, "post")
            finally:
                L.is_news = orig_is_news
            s2 = {h.stratum for h in r2.hits}
            rec["nE"], rec["nF"] = "E" in s2, "F" in s2
        recs.append(rec)
    out = pd.DataFrame(recs).set_index("uidx")
    return out


# ================================================================ 2. 정규화(직전 창만, 자기 제외)

def z_prev(x: np.ndarray, valid: np.ndarray, n: int = Z_N, min_n: int = Z_MIN) -> np.ndarray:
    """직전 n개 창(행 위치) 가운데 값이 있는 창이 min_n개 이상이면 (x − 평균) ÷ SD. 자기 제외."""
    x = np.asarray(x, float)
    vals = np.where(valid, x, np.nan)
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        if not (valid[i] and np.isfinite(x[i])):
            continue
        h = vals[max(0, i - n):i]
        h = h[np.isfinite(h)]
        if len(h) >= min_n:
            sd = np.std(h, ddof=1)
            if sd > 0:
                out[i] = (x[i] - h.mean()) / sd
    return out


def pct_prev(x: np.ndarray, valid: np.ndarray, n: int = P_N, min_n: int = P_MIN) -> np.ndarray:
    """직전 n개 창 백분위(자기 제외): (작은 수 + ½ 같은 수) ÷ 기준 수. 기준이 min_n 미만이면 결측."""
    x = np.asarray(x, float)
    vals = np.where(valid, x, np.nan)
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        if not (valid[i] and np.isfinite(x[i])):
            continue
        h = vals[max(0, i - n):i]
        h = h[np.isfinite(h)]
        if len(h) >= min_n:
            out[i] = (np.sum(h < x[i]) + 0.5 * np.sum(h == x[i])) / len(h)
    return out


def rz8(x: np.ndarray, valid: np.ndarray, weekday: np.ndarray, holiday: np.ndarray) -> np.ndarray:
    """같은 요일 직전 8창(연휴 뒤 창·무효 창 제외)의 중앙값·MAD로 표준화, ±6에서 자른다. 8창이 다 차야 계산."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    hist: dict[int, list[float]] = {}
    for i in range(len(x)):
        h = hist.setdefault(int(weekday[i]), [])
        if valid[i] and np.isfinite(x[i]) and len(h) >= RZ_N:
            base = np.array(h[-RZ_N:])
            if np.all(np.isfinite(base)):
                med = np.median(base)
                mad = np.median(np.abs(base - med)) * MAD_K
                if mad > 0:
                    out[i] = float(np.clip((x[i] - med) / mad, -RZ_CLIP, RZ_CLIP))
        if valid[i] and not holiday[i]:
            h.append(float(x[i]) if np.isfinite(x[i]) else np.nan)
    return out


def stage_of(p: np.ndarray, k: int) -> np.ndarray:
    """부록 1: ⌈k × pct250⌉, 1~k로 자른다(백분위 0이면 1단계)."""
    s = np.ceil(k * np.asarray(p, float))
    return np.where(np.isfinite(s), np.clip(s, 1, k), np.nan)


def roll_mean_full(x: np.ndarray, w: int) -> np.ndarray:
    """끝이 i인 w개(행 위치)가 모두 값이 있을 때만 평균."""
    s = pd.Series(np.asarray(x, float))
    return s.rolling(w, min_periods=w).mean().to_numpy()


# ================================================================ 3. 창 집계

def author_ratio(df: pd.DataFrame, key: str, flags: list[str], index: pd.Index) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """작성자 비율 X_A = 창 안에서 X를 한 번이라도 쓴 고유 작성자 ÷ 창 안 활동 고유 작성자(같은 단위 틀)."""
    if df.empty:
        z = pd.DataFrame(np.nan, index=index, columns=flags)
        return z, pd.Series(0, index=index), z.fillna(0)
    g = df.groupby([key, "a"], sort=False)[flags].max()
    ratio = g.groupby(level=0).mean().reindex(index)
    n = g.groupby(level=0).size().reindex(index).fillna(0).astype(int)
    cnt = g.groupby(level=0).sum().reindex(index).fillna(0)
    return ratio, n, cnt


def slot_of(ts: pd.Series) -> np.ndarray:
    m = ts.dt.hour * 60 + ts.dt.minute
    out = np.full(len(ts), "night", dtype=object)
    for name, lo, hi in KST_SLOTS:
        out[((m >= lo) & (m < hi)).to_numpy()] = name
    return out


def bot_sets(posts: pd.DataFrame, closes: pd.Series) -> dict[str, set]:
    """창마다 봇·연재 작성자: 마감 전 직전 30일 안 같은 제목 패턴(한글·영문만 남김)을 15일 이상 쓴 작성자."""
    p = posts[posts["pat"] != ""]
    out = {}
    for wid, c in closes.items():
        sub = p[(p["dt_kst"] >= c - pd.Timedelta(days=BOT_LOOKBACK)) & (p["dt_kst"] < c)]
        nd = sub.groupby(["a", "pat"])["day"].nunique()
        out[wid] = set(nd[nd >= BOT_DAYS].index.get_level_values(0))
    return out


def dsd_expected(pool: np.ndarray, n: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    draws = pool[rng.integers(0, len(pool), size=(DSD_DRAWS, n))]
    return float(np.std(draws, axis=1, ddof=1).mean())


def score_block(posts_w: pd.DataFrame, key: str, wins: pd.DataFrame, bots: dict[str, set], prefix: str) -> pd.DataFrame:
    """글 점수 → 작성자 점수(창 안 그 작성자 글 점수 평균, 0점 작성자 제외) → B, D_SD, D_AF."""
    idx = wins["win_id"]
    recs, pools = [], []
    valid = wins["valid"].to_numpy()
    for i, wid in enumerate(idx):
        sub = posts_w[posts_w[key] == wid]
        nb = sub["a"].isin(bots.get(wid, set()))
        sub2 = sub[~nb]
        a_score = sub2.groupby("a")["score"].mean()
        nz = a_score[a_score != 0].to_numpy()
        bull, bear = int((nz > 0).sum()), int((nz < 0).sum())
        rec = {f"{prefix}n_scored_posts": int(len(sub)), f"{prefix}n_bot_posts_excl": int(nb.sum()),
               f"{prefix}n_votes": int(len(nz)), f"{prefix}B": (bull - bear) / (bull + bear) if bull + bear else np.nan,
               f"{prefix}D_SD_raw": float(np.std(nz, ddof=1)) if len(nz) >= 2 else np.nan,
               f"{prefix}D_SD_exp": np.nan, f"{prefix}D_SD": np.nan, f"{prefix}D_SD_pool_windows": 0}
        prev = [pools[j] for j in range(max(0, i - Z_N), i) if pools[j] is not None]
        rec[f"{prefix}D_SD_pool_windows"] = len(prev)
        if valid[i] and len(nz) >= DSD_MIN_AUTH and len(prev) >= Z_MIN:
            pool = np.concatenate(prev)
            seed = int(hashlib.sha256(f"{SEED}{prefix}{wid}".encode()).hexdigest()[:8], 16)
            e = dsd_expected(pool, len(nz), seed)
            rec[f"{prefix}D_SD_exp"] = e
            rec[f"{prefix}D_SD"] = rec[f"{prefix}D_SD_raw"] / e if e > 0 else np.nan
        pools.append(nz if (valid[i] and len(nz) > 0) else None)
        recs.append(rec)
    df = pd.DataFrame(recs, index=idx)
    b = df[f"{prefix}B"].to_numpy()
    df[f"{prefix}D_AF"] = np.sqrt(np.clip(1 - b ** 2, 0, None))
    return df


def attention(posts: pd.DataFrame, c120: str, rank_ok: str, hist_start: date) -> pd.Series:
    """H12 관심 = 마감 전·등록 뒤 120분 댓글 수 ÷ 같은 등록 시간(1시간 칸)의 직전 20일 중앙값.
    기준은 순위 자격 글(c120이 120분을 다 셈)만, 직전 20일이 모두 자료 안(hist_start 이후)일 때만. 중앙값 0이면 결측."""
    p = posts[posts[rank_ok] & posts[c120].notna()]
    vals: dict[tuple[date, int], np.ndarray] = {
        k: g[c120].to_numpy(float) for k, g in p.groupby([p["day"], p["hour"]])}
    days = sorted({d for d, _ in vals})
    med: dict[tuple[date, int], float] = {}
    for d in days:
        if d - timedelta(days=ATT_DAYS) < hist_start:
            continue
        for h in range(24):
            arr = [vals[(d - timedelta(days=k), h)] for k in range(1, ATT_DAYS + 1)
                   if (d - timedelta(days=k), h) in vals]
            if arr:
                med[(d, h)] = float(np.median(np.concatenate(arr)))
    m = pd.Series([med.get((d, h), np.nan) for d, h in zip(p["day"], p["hour"])], index=p.index)
    att = p[c120] / m.where(m > 0)
    return att.reindex(posts.index)


def top_mask(att: pd.Series, c120: pd.Series, dt: pd.Series) -> pd.Series:
    """관심 상위 10%(⌈10%⌉개). 같은 관심은 c120 많은 글, 그다음 먼저 쓴 글."""
    order = pd.DataFrame({"att": att, "c": c120, "t": dt}).sort_values(["att", "c", "t"],
                                                                        ascending=[False, False, True])
    k = math.ceil(TOP_SHARE * len(order))
    m = pd.Series(False, index=att.index)
    m.loc[order.index[:k]] = True
    return m


def past_closes(ticker: str, start: date, last_day: date) -> pd.Series:
    """'지난' 종가만: last_day(마지막 창의 직전 거래일) 이하. 시가·고가·저가·거래대금은 받지 않는다."""
    import yfinance as yf
    h = yf.Ticker(ticker).history(start=start.isoformat(), end=(last_day + timedelta(days=1)).isoformat(),
                                  auto_adjust=False)
    s = pd.Series(h["Close"].to_numpy(float), index=[pd.Timestamp(t).date() for t in h.index])
    s = s[~s.index.duplicated(keep="last")]
    s = s[s.index <= last_day]
    if len(s) and max(s.index) > last_day:
        raise RuntimeError("방화벽: 마지막 창 뒤 가격이 들어왔다")
    return s


def shock_z(close: pd.Series) -> pd.Series:
    """r(d) ÷ σ60(d): σ60은 d 직전 60개 일수익의 SD(d 제외, 60개 미만이면 결측)."""
    r = close.pct_change()
    sig = r.shift(1).rolling(60, min_periods=60).std(ddof=1)
    return r / sig


def assign_ranges(t: pd.Series, opens: np.ndarray, closes: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """[open, close) 범위(겹치지 않음)에 시각 배정."""
    tt = t.to_numpy(dtype="datetime64[s]")
    o = opens.astype("datetime64[s]")
    c = closes.astype("datetime64[s]")
    j = np.searchsorted(c, tt, side="right")
    out = np.full(len(tt), None, dtype=object)
    ok = (j < len(c)) & ~np.isnat(tt)
    jj = j[ok]
    inside = tt[ok] >= o[jj]
    sel = np.flatnonzero(ok)[inside]
    out[sel] = ids[jj[inside]]
    return out


def build_market(market: str, u: pd.DataFrame, m: pd.DataFrame, wins_all: pd.DataFrame,
                 prices: dict[str, pd.Series] | None) -> pd.DataFrame:
    key = "win_kr" if market == "kr" else "win_u"
    c120, rank_ok = ("c120_kr", "rank_ok_kr") if market == "kr" else ("c120_u", "rank_ok_u")
    pwin, precl = ("pwin_kr", "pre_close_kr") if market == "kr" else ("pwin_u", "pre_close_u")
    w = wins_all[(wins_all["variant"] == "main") & wins_all["complete"]].sort_values("close_kst").reset_index(drop=True)
    w["outage_flag"] = w["outage"].fillna(False).astype(bool)
    w["outage_unknown"] = w["outage"].isna()
    w["valid"] = ~w["outage_flag"]                          # §2.4: 장애 창은 모든 특징에서 결측
    idx = pd.Index(w["win_id"])
    F = pd.DataFrame(index=idx)
    meta_cols = ["market", "D", "P", "weekday", "open_kst", "close_kst", "length_h", "kind", "main_kind", "over96",
                 "xholiday_mismatch", "dst_week", "early_close_D", "complete", "outage_flag", "outage_unknown",
                 "spam_wave", "missing_rate", "trunc_share", "c_conf_share", "c_latest", "n_posts", "n_comments",
                 "n_authors", "d_no", "valid"]
    for c in meta_cols:
        F[c] = w[c].to_numpy()
    valid = w["valid"].to_numpy()
    holiday = (w["main_kind"] == "holiday").to_numpy()
    wd = w["weekday"].to_numpy()

    uu = u[u[key].isin(idx)].join(m, how="inner")
    ok = uu["excl"] == ""
    P_ = uu[ok & (uu["utype"] == "post")]
    C_ = uu[ok & (uu["utype"] == "comment")]
    A_ = uu[ok]
    flags = [f"h_{s}" for s in STRATA] + ["h_FO_mkt", "h_LL_US_noNVDL"]
    names = [f[2:] for f in flags]
    for tag, df in (("P", P_), ("C", C_), ("all", A_)):
        r, n, cnt = author_ratio(df, key, flags, idx)
        for f, nm in zip(flags, names):
            F[f"{nm}_A_{tag}"] = r[f].to_numpy()
        F[f"n_auth_{tag}"] = n.to_numpy()
        if tag == "all":
            for nm in ("CAP", "HES", "MC", "LL_US", "LL_US_noNVDL", "LS", "RES", "ANG", "F", "E"):
                F[f"{nm}_n_all"] = cnt[f"h_{nm}"].to_numpy()
    F["n_valid_posts"] = P_.groupby(key).size().reindex(idx).fillna(0).astype(int).to_numpy()
    F["n_valid_comments"] = C_.groupby(key).size().reindex(idx).fillna(0).astype(int).to_numpy()
    F["n_news_posts"] = uu[(uu["excl"] == "news")].groupby(key).size().reindex(idx).fillna(0).astype(int).to_numpy()
    F["PX_unit_share"] = A_.groupby(key)["px_unit"].mean().reindex(idx).to_numpy()   # §2.3 메아리 계기

    # S(순심리)와 Pess(H15)
    F["S_P"] = F["E_A_P"] - F["F_A_P"]
    F["S_C"] = F["E_A_C"] - F["F_A_C"]
    F["S"] = 0.5 * F["S_P"] + 0.5 * F["S_C"]
    F["S_z60"] = z_prev(F["S"].to_numpy(), valid)
    F["Pess"] = -F["S_z60"]
    for tag, df in (("P", P_), ("C", C_)):
        g = df.groupby(key)[["E_cnt3", "F_cnt3", "eoj"]].sum().reindex(idx)
        F[f"S_tok_{tag}"] = ((g["E_cnt3"] - g["F_cnt3"]) / g["eoj"].where(g["eoj"] > 0)).to_numpy()
    F["S_tok"] = 0.5 * F["S_tok_P"] + 0.5 * F["S_tok_C"]
    F["Pess_tok"] = -z_prev(F["S_tok"].to_numpy(), valid)

    # H19 환호, H21 소외(시장어 동반 판이 주), 그 밖의 묶음 z60
    def zpc(nm):
        zp = z_prev(F[f"{nm}_A_P"].to_numpy(), valid)
        zc = z_prev(F[f"{nm}_A_C"].to_numpy(), valid)
        return zp, zc, 0.5 * zp + 0.5 * zc
    F["E_A_P_z60"], F["E_A_C_z60"], F["H19_E"] = zpc("E")
    F["FO_mkt_A_P_z60"], F["FO_mkt_A_C_z60"], F["H21_FO"] = zpc("FO_mkt")
    _, _, F["H21_FO_plain"] = zpc("FO")
    for nm in ("F", "CAP", "HES", "RES", "ANG"):
        F[f"{nm}_A_all_z60"] = z_prev(F[f"{nm}_A_all"].to_numpy(), valid)

    # H05 감정 밀도: 뉴스형·PX 단위를 뺀 단위 중 E·F·FO·ANG 단어 또는 강한 표지. 시간대(장중·저녁·심야)별 z60을 단위 수로 가중
    ed = uu[ok & ~uu["px_unit"]].copy()
    ed["slot"] = slot_of(ed["dt_kst"])
    ed["w_any"] = ed[["h_E", "h_F", "h_FO", "h_ANG"]].any(axis=1)
    ed["ed"] = ed["w_any"] | ed["mark"]
    for tag, ut in (("P", "post"), ("C", "comment")):
        e1 = ed[ed["utype"] == ut]
        F[f"ED_{tag}_raw"] = e1.groupby(key)["ed"].mean().reindex(idx).to_numpy()
        F[f"ED_{tag}_word"] = e1.groupby(key)["w_any"].mean().reindex(idx).to_numpy()
        F[f"ED_{tag}_mark"] = e1.groupby(key)["mark"].mean().reindex(idx).to_numpy()
        num = np.zeros(len(idx))
        den = np.zeros(len(idx))
        for s in ("day", "eve", "night"):
            es = e1[e1["slot"] == s]
            share = es.groupby(key)["ed"].mean().reindex(idx).to_numpy()
            n_s = es.groupby(key).size().reindex(idx).fillna(0).to_numpy(float)
            zs = z_prev(share, valid)
            okz = np.isfinite(zs)
            num += np.where(okz, zs * n_s, 0)
            den += np.where(okz, n_s, 0)
        F[f"ED_{tag}"] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    F["H05_ED"] = 0.5 * F["ED_P"] + 0.5 * F["ED_C"]

    # H16 뉴스형 분리(첫 줄로 분류) — 의견형(뉴스 아닌 글)과 사건 뉴스형의 Pess
    news = uu[uu["excl"] == "news"].copy()
    for cat in ("price", "event", "other"):
        nc = news[news["news_cat"] == cat]
        F[f"n_news_{cat}"] = nc.groupby(key).size().reindex(idx).fillna(0).astype(int).to_numpy()
        if nc.empty:
            F[f"S_news_{cat}"] = np.nan
        else:
            g = nc.groupby([key, "a"])[["nE", "nF"]].max().groupby(level=0).mean().reindex(idx)
            F[f"S_news_{cat}"] = (g["nE"] - g["nF"]).to_numpy()
    F["S_opinion"] = F["S_P"]
    F["Pess_opinion"] = -z_prev(F["S_opinion"].to_numpy(), valid)
    F["Pess_news_event"] = -z_prev(F["S_news_event"].to_numpy(), valid)
    F["Pess_news_price"] = -z_prev(F["S_news_price"].to_numpy(), valid)

    # 글 점수·작성자 점수·B·D_SD·D_AF (봇·연재 제외), 반도체 태그판(H10)
    allposts = u[u["utype"] == "post"].copy()
    allposts["pat"] = allposts["title"].fillna("").map(lambda s: TITLE_PAT_RX.sub("", s))
    allposts["day"] = allposts["dt_kst"].dt.date
    bots = bot_sets(allposts, pd.Series(w["close_kst"].to_numpy(), index=idx))
    F["n_bot_authors"] = [len(bots[i]) for i in idx]
    sp = P_[P_["score"].notna()]
    F = F.join(score_block(sp, key, w, bots, ""))
    F = F.join(score_block(sp[sp["h_SEMI"]], key, w, bots, "semi_"))
    F["mean_post_score"] = sp.groupby(key)["score"].mean().reindex(idx).to_numpy()
    F["B_z60"] = z_prev(F["B"].to_numpy(), valid)
    F["D_SD_z60"] = z_prev(F["D_SD"].to_numpy(), valid)
    F["semi_D_SD_z60"] = z_prev(F["semi_D_SD"].to_numpy(), valid)
    dsd = F["D_SD"].to_numpy()
    lag = np.full(len(dsd), np.nan)
    for i in range(5, len(dsd)):
        b3 = dsd[i - 5:i - 2]
        if np.all(np.isfinite(b3)):
            lag[i] = b3.mean()
    F["dD3"] = dsd - lag                                      # H11: D_SD(D) − 평균(D_SD(D−5..D−3))
    F["dD3_z60"] = z_prev(F["dD3"].to_numpy(), valid)
    F["H11_X"] = -F["dD3_z60"] * F["B_z60"]

    # H12 Gap, H55 K (관심 상위 10%), H54 SH
    pw = uu[uu["utype"] == "post"].copy()
    pw["day"] = pw["dt_kst"].dt.date
    pw["hour"] = pw["dt_kst"].dt.hour
    hist_start = (pd.Timestamp(wins_all.loc[wins_all["complete"], "open_kst"].min()).normalize()
                  + pd.Timedelta(days=1)).date()
    allp = u[u["utype"] == "post"].copy()
    allp["day"] = allp["dt_kst"].dt.date
    allp["hour"] = allp["dt_kst"].dt.hour
    allp["att"] = attention(allp, c120, rank_ok, hist_start)
    pw["att"] = allp["att"].reindex(pw.index)
    cm = u[(u["utype"] == "comment") & u[precl]]
    precl_n = cm.groupby(["no"]).size()
    pw["n_precl"] = pw["no"].map(precl_n).fillna(0)
    ranked = pw[pw[rank_ok] & pw["att"].notna()]
    gap_raw, k_share, n_rank, n_rank_g = [], [], [], []
    # 창의 첫 글(열림 시각)까지 직전 20일 기준이 다 자료 안이어야 한다(시작 쪽 창이 일부 글로만 순위를 매기지 않게)
    att_full = [(pd.Timestamp(o).date() - timedelta(days=ATT_DAYS)) >= hist_start for o in w["open_kst"]]
    for wid, af in zip(idx, att_full):
        rk = ranked[ranked[key] == wid] if af else ranked.iloc[0:0]
        n_rank.append(len(rk))
        if len(rk) and rk["n_precl"].sum() > 0:
            tm = top_mask(rk["att"], rk[c120], rk["dt_kst"])
            k_share.append(rk.loc[tm, "n_precl"].sum() / rk["n_precl"].sum())
        else:
            k_share.append(np.nan)
        rg = rk[(rk["excl"] == "") & ~rk["px_unit"] & rk["score"].notna()]
        n_rank_g.append(len(rg))
        allw = pw[(pw[key] == wid) & (pw["excl"] == "") & ~pw["px_unit"] & pw["score"].notna()]
        if len(rg) and len(allw):
            tg = top_mask(rg["att"], rg[c120], rg["dt_kst"])
            top_sign = np.sign(rg.loc[tg, "score"]).mean()
            auth_sign = np.sign(allw.groupby("a")["score"].mean()).mean()
            gap_raw.append(top_sign - auth_sign)
        else:
            gap_raw.append(np.nan)
    F["n_ranked_posts"] = n_rank
    F["n_ranked_gap"] = n_rank_g
    F["Gap_raw"] = gap_raw
    F["H12_Gap"] = z_prev(F["Gap_raw"].to_numpy(), valid)
    F["K"] = k_share
    F["A7_K"] = rz8(F["K"].to_numpy(), valid, wd, holiday)
    bodyc = pw["text"].fillna("").map(lambda s: L._compact(URL_RX.sub("", s)))
    tlen = pw["title"].fillna("").str.strip().str.len()
    pw["empty_body"] = bodyc.str.len() == 0
    pw["short"] = (bodyc.str.len() <= SH_BODY) & (tlen <= SH_TITLE)
    shd = pw[~pw["empty_body"]]
    F["n_empty_body"] = pw.groupby(key)["empty_body"].sum().reindex(idx).fillna(0).astype(int).to_numpy()
    F["SH"] = shd.groupby(key)["short"].mean().reindex(idx).to_numpy()
    F["A6_SH"] = rz8(F["SH"].to_numpy(), valid, wd, holiday)

    # H26 PH = RES_z − ANG_z (5창 합 비율의 z60)
    N5 = roll_mean_full(np.where(valid, F["n_auth_all"].to_numpy(float), np.nan), 5) * 5
    for nm in ("RES", "ANG"):
        S5 = roll_mean_full(np.where(valid, F[f"{nm}_n_all"].to_numpy(float), np.nan), 5) * 5
        F[f"{nm}5"] = S5 / np.where(N5 > 0, N5, np.nan)
        F[f"{nm}5_z60"] = z_prev(F[f"{nm}5"].to_numpy(), valid)
    F["H26_PH"] = F["RES5_z60"] - F["ANG5_z60"]

    # H31 레버리지 롱(미국 하위 묶음 작성자 비율)과 20창 변화, 보조 (L−I)/(L+I)
    for nm in ("LL_US", "LL_US_noNVDL"):
        x = np.where(valid, F[f"{nm}_A_all"].to_numpy(float), np.nan)
        d20 = np.full(len(x), np.nan)
        d20[20:] = x[20:] - x[:-20]
        F[f"{nm}_d20"] = d20
    L_, I_ = F["LL_US_n_all"].to_numpy(float), F["LS_n_all"].to_numpy(float)
    F["LLLS_ratio"] = np.where(L_ + I_ > 0, (L_ - I_) / np.where(L_ + I_ > 0, L_ + I_, 1), np.nan)

    # H33 반대매매
    F["MC_A_all_z60"] = z_prev(F["MC_A_all"].to_numpy(), valid)
    F["MC_A_all_pct250"] = pct_prev(F["MC_A_all"].to_numpy(), valid)
    sig = ((F["MC_A_all_z60"] >= 2) | (F["MC_A_all_pct250"] >= 0.90)) & (F["MC_n_all"] >= 3)
    known = F["MC_A_all_z60"].notna() | F["MC_A_all_pct250"].notna()
    F["H33_sig"] = pd.array(np.where(known, sig, pd.NA), dtype="boolean")

    # 부록 1: 단계·몇 주 분위기·엇갈림(가격 조건 제외)
    F["S_pct250"] = pct_prev(F["S"].to_numpy(), valid)
    F["stage10"] = stage_of(F["S_pct250"].to_numpy(), 10)
    F["stage20"] = stage_of(F["S_pct250"].to_numpy(), 20)
    F["SG"] = roll_mean_full(F["stage10"].to_numpy(), 20)
    sg = F["SG"].to_numpy()
    sgd = np.full(len(sg), np.nan)
    sgd[20:] = sg[20:] - sg[:-20]
    F["SG_d20"] = sgd
    F["H59_sent_bear"] = pd.array(np.where(np.isfinite(sgd), sgd <= -2, pd.NA), dtype="boolean")
    F["H59_sent_bull"] = pd.array(np.where(np.isfinite(sgd), sgd >= 2, pd.NA), dtype="boolean")
    # 주 창: 월요일 같은 시각 마감(국내 월 07:59, 미국 월 08:29 ET). 창 D의 주 = D 이후 첫 월요일(D가 월이면 D)
    Dd = pd.to_datetime(w["D"])
    F["week_id"] = (Dd + pd.to_timedelta((7 - Dd.dt.weekday) % 7, unit="D")).dt.date.astype(str).to_numpy()
    sv = np.where(valid, F["S"].to_numpy(float), np.nan)
    wk = pd.DataFrame({"wk": F["week_id"].to_numpy(), "S": sv})
    wmean = wk.groupby("wk")["S"].agg(lambda x: x.mean() if x.notna().all() else np.nan)
    wmean.loc[wmean.index.min()] = np.nan          # 자료 첫 주는 앞부분이 비어 있다(창이 이어지는 둘째 주부터 완전)
    last_D = str(pd.Timestamp(w["D"].max()).date())
    wmean.loc[wmean.index > last_D] = np.nan       # 끝 주: 월요일 마감이 자료 뒤라 아직 덜 찼다
    last_in_week = ~wk["wk"].duplicated(keep="last")
    F["week_S_mean"] = np.where(last_in_week, wk["wk"].map(wmean), np.nan)
    wser = wmean.sort_index()
    wpct = pct_prev(wser.to_numpy(), np.isfinite(wser.to_numpy()), n=WEEK_N, min_n=WEEK_N)
    wstage = pd.Series(stage_of(wpct, 10), index=wser.index)
    F["week_stage10"] = np.where(last_in_week, wk["wk"].map(wstage), np.nan)

    # H24~H26 사건(국내만): 사건 표지는 직전 거래일 P의 지난 종가만. F_A(P 09:00 ~ D 07:59) − 직전 20창 평균
    if market == "kr":
        opens = np.array([pd.Timestamp(p) + pd.Timedelta(hours=9) for p in w["P"]], dtype="datetime64[s]")
        closes = w["close_kst"].to_numpy(dtype="datetime64[s]")
        A9 = uu[ok].copy()
        A9["w9"] = assign_ranges(A9["dt_kst"], opens, closes, idx.to_numpy())
        A9 = A9[A9["w9"].notna()]
        r9, n9, c9 = author_ratio(A9, "w9", ["h_F", "h_CAP", "h_HES"], idx)
        F["F_A_d9"] = r9["h_F"].to_numpy()
        F["n_auth_d9"] = n9.to_numpy()
        F["CAP_n_d9"] = c9["h_CAP"].to_numpy()
        F["HES_n_d9"] = c9["h_HES"].to_numpy()
        F["H25_CR_d9"] = (F["CAP_n_d9"] + 1) / (F["HES_n_d9"] + 1)
        F.loc[~valid, ["F_A_d9", "H25_CR_d9"]] = np.nan
        fa = np.where(valid, F["F_A_all"].to_numpy(float), np.nan)
        base = np.full(len(fa), np.nan)
        for i in range(len(fa)):
            h = fa[max(0, i - EV_BASE):i]
            if i >= EV_BASE and np.all(np.isfinite(h)):
                base[i] = h.mean()
        F["F_A_base20"] = base
        F["H24_R"] = F["F_A_d9"] - F["F_A_base20"]
        if prices:
            for nm, ser in prices.items():
                z = shock_z(ser)
                F[f"px_{nm}_rz_P"] = [z.get(pd.Timestamp(p).date(), np.nan) for p in w["P"]]
            ek = (F["px_kospi_rz_P"] <= EV_Z).to_numpy()
            eh = (F["px_hynix_rz_P"] <= EV_Z).to_numpy() if "px_hynix_rz_P" in F else np.zeros(len(F), bool)
            F["ev_kospi"], F["ev_hynix"] = ek, eh
            F["ev_any"] = ek | eh
            for col, ev in (("ev_first", ek | eh), ("ev_first_kospi", ek)):
                first = np.zeros(len(ev), bool)
                last = -10 ** 9
                for i in range(len(ev)):
                    if ev[i]:
                        if i - last > EV_MERGE:
                            first[i] = True
                        last = i
                F[col] = first
    F["market"] = market
    feat_cols = [c for c in F.columns if c not in meta_cols + ["market", "week_id"]]
    for c in feat_cols:
        if F[c].dtype == bool:
            F[c] = pd.array(np.where(valid, F[c].to_numpy(), pd.NA), dtype="boolean")
        else:
            F.loc[~valid, c] = np.nan if F[c].dtype.kind in "fiu" else pd.NA
    F = F.reset_index().rename(columns={"index": "win_id"})
    first = ["win_id", "market"]
    return F[first + [c for c in F.columns if c not in first]].copy()


# ================================================================ 4. 실행

def run() -> dict:
    lex = check_lexicon()
    if lex["word_set_mismatch"]:
        raise RuntimeError(f"v0.2 매처와 lexicon_v02.json 단어가 다르다: {lex['word_set_mismatch']}")
    cols = ["no", "cid", "a", "utype", "dt_kst", "title", "text", "dt_conf", "win_kr", "win_u", "pwin_kr", "pwin_u",
            "pre_close_kr", "pre_close_u", "rank_ok_kr", "rank_ok_u", "c120_kr", "c120_u"]
    u = pd.read_parquet(RESULTS / "units.parquet", columns=cols)
    m = match_units(u)
    m.reset_index().to_parquet(RESULTS / "feat_text_units.parquet", index=False)
    wk = pd.read_parquet(RESULTS / "windows_kr.parquet")
    wu = pd.read_parquet(RESULTS / "windows_us.parquet")
    wk_main = wk[(wk["variant"] == "main") & wk["complete"]]
    last_P = pd.Timestamp(wk_main["P"].max()).date()
    first_P = pd.Timestamp(wk_main["P"].min()).date()
    prices = {"kospi": past_closes("^KS11", first_P - timedelta(days=150), last_P),
              "hynix": past_closes("000660.KS", first_P - timedelta(days=150), last_P)}
    price_info = {k: {"first": str(min(v.index)), "last": str(max(v.index)), "n": int(len(v))} for k, v in prices.items()}
    for k, v in prices.items():
        if max(v.index) > last_P:
            raise RuntimeError("방화벽 위반")
    fk = build_market("kr", u, m, wk, prices)
    fu = build_market("us", u, m, wu, None)
    feat = pd.concat([fk, fu], ignore_index=True, sort=False)
    feat.to_parquet(RESULTS / "feat_text.parquet", index=False)
    meta = {"created": datetime.now().isoformat(timespec="seconds"), "lexicon": {k: v for k, v in lex.items()
                                                                                    if k != "status"},
            "lexicon_status": lex["status"], "units_sha256": sha256_file(RESULTS / "units.parquet"),
            "windows_kr_sha256": sha256_file(RESULTS / "windows_kr.parquet"),
            "windows_us_sha256": sha256_file(RESULTS / "windows_us.parquet"),
            "prices": price_info, "rows": {"kr": int(len(fk)), "us": int(len(fu))},
            "units_matched": int(len(m)), "excl": m["excl"].value_counts().to_dict(),
            "news_cat": m.loc[m["excl"] == "news", "news_cat"].value_counts().to_dict()}
    ev = fk[fk.get("ev_any", pd.Series(False, index=fk.index)).astype(bool)]
    meta["events"] = [{"win_id": r.win_id, "P": str(pd.Timestamp(r.P).date()), "kospi_rz": r.px_kospi_rz_P,
                       "hynix_rz": r.px_hynix_rz_P, "ev_kospi": bool(r.ev_kospi), "ev_hynix": bool(r.ev_hynix),
                       "ev_first": bool(r.ev_first), "H24_R": r.H24_R, "H25_CR_d9": r.H25_CR_d9,
                       "H26_PH": r.H26_PH} for r in ev.itertuples()]
    (RESULTS / "feat_text_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), "utf-8")
    return meta


if __name__ == "__main__":
    out = run()
    print(json.dumps({k: out[k] for k in ("rows", "units_matched", "excl", "news_cat", "prices")}, ensure_ascii=False,
                     default=str))
