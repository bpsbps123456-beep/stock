"""개미 지수 특징 3단계 — 종목·국가·시간 창 (HYPOTHESES.md §2·§3, 부록 1 정의 그대로. 동결 문서는 고치지 않는다).

대상
  H39+H48 AS(종목 관심 충격, 별칭 lexicon_draft §6) · H41 준비(같은 AS 판) · H43 Herd(추격 고백 쏠림)
  H46 Tilt·ΔTilt · H50 M(낮 메모리 글 비율 z20) · H51 S^NM(밤·새벽 순심리) · H49 S^E(저녁 순심리)
  H53 ΔS(주말 감정 변화) · H10 반도체 글 D_SD · H06/H09 포럼 전체 D_SD
읽기(읽기만)
  interim/results/units.parquet, windows_kr.parquet·windows_us.parquet(본판), loader_calendar.parquet(날짜만),
  사전 v0.2(lexicon_v02_final.load_v02, 뺀 단어는 자리를 차지한 채 버린다)
앞날 주가 방화벽
  가격은 읽지도 만들지도 않는다. M0 잔차화(H49·H50·H51·H53), R20(H41), 시장 대비 수익은 가격 단계에서 한다.
산출(interim/results/)
  feat_topic.parquet          행 = 창(table: KR_D, US_D, KR_W, US_W, WKND), 열 = 특징·표지
  feat_topic_stock.parquet    창 × 종목 긴 표(AS·Herd 재료)
  feat_topic_unit_hits.parquet 단위별 v0.2 적중 캐시(가격 없음)
  feat_topic_summary.json     특징별 결측률과 결측 사유
정의가 모호한 곳의 해석은 results/feat_topic.md와 interim/notes_feat_topic.md에 적었다(모두 누출 없는 쪽).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import lexicon_v02 as L  # noqa: E402
from lexicon_v02_final import load_v02  # noqa: E402
from loader import _ns, et_to_kst, number_boundary  # noqa: E402

INTERIM = CODE.parent
RESULTS = INTERIM / "results"

LEX_V02_SHA = "915a2d24b82eebe6ae3ca12c697be4710675ebdfe82477fbb4fa2e0605cf32d3"
LEX_V02_FULL_SHA = "b067c3d023b0bd123f04dd932926f59acf34f763084ee983029996bac0ea1762"

SEED = 20261016
DRAWS = 1000                      # D_SD 재표집 횟수(§2.0)
DSD_MIN_VOTES = 20                # H06 '표 20개 미만이면 결측', H10 '작성자 20명 미만이면 결측'
DSD_POOL = 60                     # '직전 60창 작성자 점수 풀'
DSD_POOL_PROV_MIN = 40            # 잠정판(중간 검증용, 판정 아님): 풀 40창 이상(z60 최소와 같은 비율)
Z60_N, Z60_MIN = 60, 40           # §2.0 z60
Z20_N, Z20_MIN = 20, 20           # H50 '20창 z'. 최소 창 수가 문서에 없어 20 전부를 요구한다(보수)
NM_MIN_UNITS = 30                 # H51 '단위가 30개 미만이면 결측'
SERIAL_DAYS, SERIAL_MIN_DAYS = 30, 15   # §2.2·H06 봇·연재: 직전 30일 안 같은 제목 패턴 15일 이상
AS_BASE_WEEKS, AS_MIN_MENTION, AS_MIN_STOCKS, AS_CAP, AS_EPS = 8, 5, 15, 3, 1e-4   # H39+H48
HERD_BASE, HERD_Q, HERD_DEDUP = 120, 0.95, 10           # H43
TILT_MEAN, TILT_LAG = 5, 20                              # H46
OUTAGE_RATIO, BASE_N, BASE_MIN = 0.20, 8, 3              # §2.4 장애 창(주 창에도 같은 규칙)

# ---------------------------------------------------------------- 종목 별칭(lexicon_draft §6 = v0.2 ALIAS 층)
KR_STOCKS = ["SEC", "SKH"]
US_ALIAS = {
    "NVDA": ["엔비디아", "엔비", "엔비디", "NVDA", "젠슨"], "MU": ["마이크론", "마크", "MU"],
    "SNDK": ["샌디스크", "샌디", "SNDK"], "AVGO": ["브로드컴", "브컴", "AVGO"], "TSM": ["TSMC", "티에스엠"],
    "AMD": ["AMD"], "WDC": ["웨스턴디지털", "WDC"], "DELL": ["DELL"], "PLTR": ["팔란티어", "팔란", "PLTR"],
    "TSLA": ["테슬라", "테슬", "TSLA"],
}
US_STOCKS = list(US_ALIAS)


def word_stocks() -> dict[str, frozenset]:
    m: dict[str, frozenset] = {}
    for w in L.ALIAS_SEC_WORDS:
        m[w] = frozenset({"SEC"})
    for w in L.ALIAS_SKH_WORDS:
        m[w] = frozenset({"SKH"})
    for w in L.ALIAS_BOTH_WORDS:
        m[w] = frozenset({"SEC", "SKH"})
    for t, ws in US_ALIAS.items():
        for w in ws:
            m[w] = frozenset({t})
    if set(sum(US_ALIAS.values(), [])) != set(L.ALIAS_US_WORDS):
        raise RuntimeError("미국 별칭 대응표가 사전의 ALIAS_US와 다르다")
    return m


WORD_STOCKS = word_stocks()
COIN_RX = re.compile(r"(?<![가-힣])(?:코인|비트|이더)|(?<![a-z])btc(?![a-z])")   # H53 강건판 '코인 글'
STRAT_COL = {"E": "nE", "F": "nF", "CHASE_BUY": "cb", "CHASE_SELL": "cs", "MEM": "mem", "SEMI": "semi",
             "TILT_US": "tus", "TILT_KR": "tkr"}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- 단위별 v0.2 적중

def unit_hits(units: pd.DataFrame, analyze) -> pd.DataFrame:
    """단위마다 v0.2 적중을 센다. 글 = 첫 줄 제목+본문, 댓글 = 본문. 조회·추천은 보지 않는다."""
    n = len(units)
    excl = np.empty(n, dtype=object)
    score = np.full(n, np.nan)
    cols = {c: np.zeros(n, dtype=np.int32) for c in STRAT_COL.values()}
    alias = np.empty(n, dtype=object)
    coin = np.zeros(n, dtype=bool)
    ut_a, ti_a, tx_a = units["utype"].to_numpy(), units["title"].to_numpy(), units["text"].to_numpy()
    for i in range(n):
        ut = ut_a[i]
        ti = ti_a[i] if isinstance(ti_a[i], str) else ""
        tx = tx_a[i] if isinstance(tx_a[i], str) else ""
        r = analyze(ti if ut == "post" else None, tx, ut)
        excl[i] = r.excluded
        if ut == "post" and not r.excluded and r.score is not None:
            score[i] = r.score
        st: set = set()
        for h in r.hits:
            c = STRAT_COL.get(h.stratum)
            if c is not None:
                cols[c][i] += 1
            if h.bundle == "ALIAS":
                st |= WORD_STOCKS[h.word]
        alias[i] = ",".join(sorted(st))
        coin_text = (ti + "\n" + tx) if ut == "post" else tx
        coin[i] = bool(COIN_RX.search(coin_text.lower()))
    out = pd.DataFrame({"no": units["no"].to_numpy(), "cid": units["cid"].to_numpy(), "utype": ut_a,
                        "excl": excl, "score": score, "alias": alias, "coin": coin, **cols})
    return out


# ---------------------------------------------------------------- 창 안 단위 꺼내기

class Frame:
    """usable(잡음 거르기 통과) 단위를 시각순으로 두고 [open, close) 조각을 꺼낸다."""

    def __init__(self, units: pd.DataFrame, hits: pd.DataFrame):
        if not (np.array_equal(units["no"].to_numpy(), hits["no"].to_numpy())
                and np.array_equal(units["cid"].fillna(-9).to_numpy(), hits["cid"].fillna(-9).to_numpy())):
            raise RuntimeError("적중 캐시와 units의 행 순서가 다르다")
        df = pd.DataFrame({
            "dt": _ns(units["dt_kst"]), "utype": units["utype"].to_numpy(), "a": units["a"].to_numpy(),
            "no": units["no"].to_numpy(), "title": units["title"].to_numpy(),
        })
        for c in ["excl", "score", "alias", "coin", *STRAT_COL.values()]:
            df[c] = hits[c].to_numpy()
        post_coin = df.loc[df["utype"] == "post"].set_index("no")["coin"]
        df["pcoin"] = df["no"].map(post_coin).fillna(False).astype(bool)   # 부모 글이 코인 글
        if not (np.diff(df["dt"].to_numpy().astype("int64")) >= 0).all():
            raise RuntimeError("units가 시각순이 아니다")
        self.all = df
        posts = df[df["utype"] == "post"]
        self.first_post = posts["dt"].min()
        self.u = df[df["excl"] == ""].reset_index(drop=True)
        self.udt = self.u["dt"].to_numpy()
        # 번호 증가(지운 글 포함, 주 창 장애 판정용): 저장 글 전부
        self.pdt = posts["dt"].to_numpy()
        self.pno = posts["no"].to_numpy()

    def get(self, o, c) -> pd.DataFrame:
        o, c = np.datetime64(o, "s"), np.datetime64(c, "s")
        lo, hi = np.searchsorted(self.udt, o, "left"), np.searchsorted(self.udt, c, "left")
        return self.u.iloc[lo:hi]

    def complete(self, o, c, t_end) -> bool:
        """열림 전에 저장 글이 있고(거꾸로 받기가 그 앞까지 내려옴) 마감이 컷오프 안(로더 규칙과 같다)."""
        return bool(self.first_post < np.datetime64(o, "s")) and bool(np.datetime64(c, "s")
                                                                     <= np.datetime64(t_end, "s"))


# ---------------------------------------------------------------- 공통 계산

def s_stats(sub: pd.DataFrame) -> dict:
    """S = ½(E_A − F_A)글 + ½(E_A − F_A)댓글. X_A = 그 단위 종류에서 X 단어를 한 번이라도 쓴 고유 작성자 ÷ 활동 고유 작성자."""
    out = {"n_units": int(len(sub))}
    for ut, tg in (("post", "p"), ("comment", "c")):
        x = sub[sub["utype"] == ut]
        na = x["a"].nunique()
        out[f"n_auth_{tg}"] = int(na)
        if na == 0:
            out[f"E_A_{tg}"] = out[f"F_A_{tg}"] = np.nan
        else:
            out[f"E_A_{tg}"] = x.loc[x["nE"] > 0, "a"].nunique() / na
            out[f"F_A_{tg}"] = x.loc[x["nF"] > 0, "a"].nunique() / na
    out["S"] = 0.5 * (out["E_A_p"] - out["F_A_p"]) + 0.5 * (out["E_A_c"] - out["F_A_c"])
    return out


def author_scores(posts: pd.DataFrame, exclude: set | frozenset = frozenset()) -> np.ndarray:
    """작성자 점수 = 창 안 그 작성자 글 점수의 평균, 0점인 작성자는 뺀다(§2.0). 뉴스형 글은 점수가 없다."""
    p = posts[posts["utype"] == "post"]
    p = p[p["score"].notna()]
    if exclude:
        p = p[~p["a"].isin(exclude)]
    m = p.groupby("a")["score"].mean()
    return m[m != 0].to_numpy(dtype=float)


def expected_sd(pool: np.ndarray, n: int, rng: np.random.Generator, draws: int = DRAWS) -> float:
    """풀에서 n개를 비복원으로 draws번 뽑은 표본 SD(ddof=1)의 평균."""
    sds = np.empty(draws)
    for k in range(draws):
        sds[k] = pool[rng.choice(len(pool), n, replace=False)].std(ddof=1)
    return float(sds.mean())


def rng_for(tag: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256(f"{SEED}:{tag}".encode()).hexdigest()[:8], 16))


def dsd_series(scores: list[np.ndarray | None], ids: list[str], tag: str) -> pd.DataFrame:
    """창 순서대로 D_SD. scores[i] = 그 창의 0이 아닌 작성자 점수(창이 무효면 None).

    풀 = 자기 앞 유효 창 최대 60개의 작성자 점수(자기 제외). 본판은 풀 60창을 요구한다(문서 '직전 60창').
    잠정판(prov40)은 풀 40창 이상이면 계산한다(중간 검증용, 판정 아님). 표가 20개 미만이면 결측.
    """
    hist: list[np.ndarray] = []
    recs = []
    for sc, wid in zip(scores, ids):
        r = {"sd_raw": np.nan, "n_votes": np.nan, "pool_windows": len(hist[-DSD_POOL:]), "e_sd": np.nan,
             "dsd": np.nan, "dsd_prov40": np.nan}
        if sc is not None:
            n = len(sc)
            r["n_votes"] = n
            if n >= 2:
                r["sd_raw"] = float(np.std(sc, ddof=1))
            k = len(hist[-DSD_POOL:])
            if n >= DSD_MIN_VOTES and k >= DSD_POOL_PROV_MIN:
                pool = np.concatenate(hist[-DSD_POOL:])
                if len(pool) >= n:
                    e = expected_sd(pool, n, rng_for(f"{tag}:{wid}"))
                    r["e_sd"] = e
                    if e > 0:
                        r["dsd_prov40"] = r["sd_raw"] / e
                        if k >= DSD_POOL:
                            r["dsd"] = r["dsd_prov40"]
            hist.append(sc)
        recs.append(r)
    return pd.DataFrame(recs)


def z_prior(x: np.ndarray, valid: np.ndarray, n: int, nmin: int) -> np.ndarray:
    """직전 유효 값 최대 n개(자기 제외)로 표준화. 유효 값이 nmin개 미만이면 결측. SD는 ddof=1."""
    out = np.full(len(x), np.nan)
    hist: list[float] = []
    for i in range(len(x)):
        ok = bool(valid[i]) and np.isfinite(x[i])
        if ok and len(hist) >= nmin:
            h = np.asarray(hist[-n:])
            sd = h.std(ddof=1)
            if sd > 0:
                out[i] = (x[i] - h.mean()) / sd
        if ok:
            hist.append(float(x[i]))
    return out


def same_weekday_base(x: np.ndarray, wd: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """직전 같은 요일 유효 창 최대 8개 중앙값(최소 3, 자기 제외). 로더 _same_weekday_median과 같은 규칙."""
    out = np.full(len(x), np.nan)
    hist: dict[int, list[float]] = {}
    for i in range(len(x)):
        h = hist.setdefault(int(wd[i]), [])
        if len(h) >= BASE_MIN:
            out[i] = float(np.median(h[-BASE_N:]))
        if ok[i] and np.isfinite(x[i]):
            h.append(float(x[i]))
    return out


def serial_authors(posts_hist: pd.DataFrame, close) -> set:
    """봇·연재: 마감 전 30일 안에 같은 제목 패턴(숫자·기호·공백을 뺀 한글·영문)을 15일 이상 쓴 작성자(그날까지 이력)."""
    c = np.datetime64(close, "s")
    m = (posts_hist["dt"] >= c - np.timedelta64(SERIAL_DAYS, "D")) & (posts_hist["dt"] < c) & \
        (posts_hist["pat"] != "")
    g = posts_hist[m].groupby(["a", "pat"])["day"].nunique()
    return set(g[g >= SERIAL_MIN_DAYS].index.get_level_values(0))


def title_pattern(t) -> str:
    return re.sub(r"[^가-힣a-z]", "", (t if isinstance(t, str) else "").lower())


def herd_events(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """H43 사건: 직전 유효 120창(120창 전부 요구, 자기 제외) 95% 분위수를 넘는 창(> q95, 그래서 Herd > 0).
    값이 성기어 분위수에 같은 값이 몰리면 '이상(≥)'은 상위 5%보다 훨씬 많이 켜지므로 '초과'로 읽었다(덜 켜는 쪽).
    10창 안 재점화는 같은 사건(첫날만 1, 나머지 0)."""
    out = np.full(len(x), np.nan)
    hist: list[float] = []
    last_on = -10 ** 9
    for i in range(len(x)):
        ok = bool(valid[i]) and np.isfinite(x[i])
        if ok and len(hist) >= HERD_BASE:
            q = float(np.quantile(hist[-HERD_BASE:], HERD_Q))
            on = x[i] > max(q, 0.0)
            if on:
                out[i] = 1.0 if i - last_on > HERD_DEDUP else 0.0
                last_on = i
            else:
                out[i] = 0.0
        if ok:
            hist.append(float(x[i]))
    return out


def tilt_delta(x: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """H46: 유효 창(한쪽 휴장·장애 아님, 값 있음)만으로 Tilt5 = 직전 5개 유효 창 평균(자기 포함),
    ΔTilt = Tilt5 − 20 유효 창 전의 Tilt5(주), 보조 = Tilt5 − 20 유효 창 전의 Tilt."""
    ok = valid & np.isfinite(x)
    t5 = np.full(len(x), np.nan)
    d_main = np.full(len(x), np.nan)
    d_alt = np.full(len(x), np.nan)
    vals: list[float] = []
    for i in range(len(x)):
        if ok[i]:
            vals.append(float(x[i]))
        k = len(vals)
        if k >= TILT_MEAN:
            t5[i] = float(np.mean(vals[k - TILT_MEAN:k]))
        if k - TILT_LAG >= TILT_MEAN:
            d_main[i] = t5[i] - float(np.mean(vals[k - TILT_LAG - TILT_MEAN:k - TILT_LAG]))
        if k - TILT_LAG >= 1 and k >= TILT_MEAN:
            d_alt[i] = t5[i] - vals[k - TILT_LAG - 1]
    return t5, d_main, d_alt


# ---------------------------------------------------------------- 표 만들기

def window_flags(w: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame({
        "win_id": w["win_id"].to_numpy(), "D": pd.to_datetime(w["D"]).dt.date.to_numpy(),
        "P": pd.to_datetime(w["P"]).dt.date.to_numpy(),
        "open_kst": pd.to_datetime(w["open_kst"]).to_numpy(), "close_kst": pd.to_datetime(w["close_kst"]).to_numpy(),
        "length_h": w["length_h"].to_numpy(), "kind": w["kind"].to_numpy(), "over96": w["over96"].to_numpy(),
        "xholiday_mismatch": w["xholiday_mismatch"].to_numpy(),
        "outage": w["outage"].fillna(False).astype(bool).to_numpy(),
        "outage_unknown": w["outage"].isna().to_numpy(),
        "spam_wave": w["spam_wave"].fillna(False).astype(bool).to_numpy(),
        "missing_rate": w["missing_rate"].to_numpy(), "trunc_share": w["trunc_share"].astype(float).to_numpy(),
    })
    return f


def build_kr_daily(F: Frame, wk: pd.DataFrame, cal: dict) -> tuple[pd.DataFrame, list[dict]]:
    """W_KR 본판 완전 창: 일 S, H06/H09 D_SD(포럼 전체 글, 봇·연재 제외), H51 S^NM, H43 Herd(국내 종목)."""
    flags = window_flags(wk)
    nyse = sorted(cal["nyse"])
    ph = F.all[F.all["utype"] == "post"][["dt", "a", "title"]].copy()
    ph["pat"] = [title_pattern(t) for t in ph["title"]]
    ph["day"] = ph["dt"].dt.normalize()
    rows, stock_rows, scores = [], [], []
    for _, r in flags.iterrows():
        o, c = r["open_kst"], r["close_kst"]
        sub = F.get(o, c)
        ok = not r["outage"]
        rec = {}
        posts = sub[sub["utype"] == "post"]
        rec["n_posts_usable"] = int(len(posts))
        st = s_stats(sub)
        rec.update({f"s_day_{k}": v for k, v in st.items()})
        # H06/H09 D_SD
        ser = serial_authors(ph, c)
        rec["h06_n_serial"] = len(ser)
        rec["h06_n_serial_in_win"] = int(posts.loc[posts["a"].isin(ser), "a"].nunique())
        rec["h06_serial_hist_full"] = bool(np.datetime64(c, "s") - np.timedelta64(SERIAL_DAYS, "D") >= F.first_post)
        sc = author_scores(posts, ser)
        scores.append(sc if ok else None)
        nz = posts["score"].notna() & (posts["score"] != 0) & ~posts["a"].isin(ser)
        rec["h06_n_nonzero_posts"] = int(nz.sum())          # 참고: '표'를 글 수로 읽으면 이 값(본판은 작성자 수)
        nb, nr = int((sc > 0).sum()), int((sc < 0).sum())
        rec["h06_B"] = (nb - nr) / (nb + nr) if nb + nr else np.nan
        rec["h06_daf"] = float(np.sqrt(1 - rec["h06_B"] ** 2)) if np.isfinite(rec["h06_B"]) else np.nan
        # H51 S^NM: (직전 미국 정규장 개장, D 07:59]. 월요일은 금요일 개장부터
        D = r["D"]
        nm_o, nm_c, nm_kind = nm_window(D, nyse)
        rec.update({"h51_nm_open": nm_o, "h51_nm_close": nm_c, "h51_nm_kind": nm_kind,
                    "h51_monday": D.weekday() == 0, "h51_nm_len_h": (nm_c - nm_o).total_seconds() / 3600})
        nm = F.get(nm_o, nm_c)
        snm = s_stats(nm)
        rec["h51_n_units"] = snm["n_units"]
        rec["h51_complete"] = F.complete(nm_o, nm_c, F_T_END)
        rec["h51_s_nm"] = snm["S"] if (rec["h51_complete"] and snm["n_units"] >= NM_MIN_UNITS) else np.nan
        # H43 Herd(국내 종목, W_KR)
        hrec = herd_window(posts, KR_STOCKS)
        for k in KR_STOCKS:
            rec[f"h43_buy_{k}"] = hrec[k]["buy"] if ok else np.nan
            rec[f"h43_sell_{k}"] = hrec[k]["sell"] if ok else np.nan
            stock_rows.append({"table": "KR_D", "win_id": r["win_id"], "D": D, "stock": k,
                               "n_posts_usable": len(posts), "h43_buy_auth": hrec[k]["buy_auth"],
                               "h43_sell_auth": hrec[k]["sell_auth"],
                               "h43_buy": rec[f"h43_buy_{k}"], "h43_sell": rec[f"h43_sell_{k}"]})
        rows.append(rec)
    out = pd.concat([flags, pd.DataFrame(rows)], axis=1)
    mask_outage(out, ["s_day_S", "s_day_E_A_p", "s_day_F_A_p", "s_day_E_A_c", "s_day_F_A_c", "h06_B", "h06_daf",
                      "h51_s_nm"])
    d = dsd_series(scores, list(out["win_id"]), "KR_D:h06")
    for c in d.columns:
        out[f"h06_{c}"] = d[c].to_numpy()
    out["h06_valid_win"] = ~out["outage"]
    add_herd_events(out, KR_STOCKS)
    return out, stock_rows


def nm_window(D: date, nyse: list[date]) -> tuple[datetime, datetime, str]:
    """H51 NM 창 = (직전 미국 정규장 개장, D 07:59] → [개장 다음 분, D 08:00).
    직전 미국 세션 = D보다 앞선 마지막 NYSE 거래일(월요일이면 금요일). 미국이 쉰 다음 날은 더 앞 세션부터('extended')."""
    sp = [d for d in nyse if d < D][-1]
    o = et_to_kst(sp, 9, 30) + timedelta(minutes=1)
    c = datetime.combine(D, time(8))
    kind = "night" if sp == D - timedelta(days=1) else (
        "weekend" if D.weekday() == 0 and sp == D - timedelta(days=3) else "extended")
    return o, c, kind


def weekend_windows(sat: date) -> tuple[tuple[datetime, datetime], tuple[datetime, datetime]]:
    """H53: 금 15:31 ~ 토 05:59 → [금 15:31, 토 06:00), 토 06:00 ~ 월 06:59/07:59 → [토 06:00, 일 18:00 ET)."""
    fri, sun = sat - timedelta(days=1), sat + timedelta(days=1)
    a = (datetime.combine(fri, time(15, 31)), datetime.combine(sat, time(6)))
    return a, (a[1], et_to_kst(sun, 18, 0))


def herd_window(posts: pd.DataFrame, stocks: list[str]) -> dict:
    """H43: 추격 어구(CHASE_BUY)·매도 고백(CHASE_SELL)과 종목 별칭이 같이 나온 첫 줄 글, 작성자당 1건 ÷ 창 글."""
    n = len(posts)
    out = {}
    al = posts["alias"].fillna("")
    for k in stocks:
        has = al.str.split(",").apply(lambda s: k in s) if n else pd.Series(dtype=bool)
        b = posts.loc[has & (posts["cb"] > 0), "a"].nunique() if n else 0
        s = posts.loc[has & (posts["cs"] > 0), "a"].nunique() if n else 0
        out[k] = {"buy_auth": int(b), "sell_auth": int(s), "buy": b / n if n else np.nan,
                  "sell": s / n if n else np.nan}
    return out


def mask_outage(out: pd.DataFrame, cols: list[str]) -> None:
    """§2.4 장애 창은 모든 특징에서 결측."""
    out.loc[out["outage"].to_numpy(bool), [c for c in cols if c in out]] = np.nan


def add_herd_events(out: pd.DataFrame, stocks: list[str]) -> None:
    valid = ~out["outage"].to_numpy()
    for k in stocks:
        for side in ("buy", "sell"):
            out[f"h43_event_{side}_{k}"] = herd_events(out[f"h43_{side}_{k}"].to_numpy(float), valid)


def build_us_daily(F: Frame, wu: pd.DataFrame, cal: dict) -> tuple[pd.DataFrame, list[dict]]:
    """W_U 본판 완전 창: 일 S, H46 Tilt, H50 M, H49 S^E, H10 반도체 D_SD, H43 Herd(미국 종목)."""
    flags = window_flags(wu)
    krx = set(cal["krx"])
    rows, stock_rows, semi_scores = [], [], []
    for _, r in flags.iterrows():
        o, c = r["open_kst"], r["close_kst"]
        D = r["D"]
        ok = not r["outage"]
        sub = F.get(o, c)
        posts = sub[sub["utype"] == "post"]
        rec = {"n_posts_usable": int(len(posts)), "krx_closed_D": D not in krx}
        st = s_stats(sub)
        rec.update({f"s_day_{k}": v for k, v in st.items()})
        # H46 Tilt(W_U, 두 쪽 단어가 다 나온 글은 사전 매처가 이미 뺐다)
        n_us = int((posts["tus"] > 0).sum())
        n_kr = int((posts["tkr"] > 0).sum())
        a_us = posts.loc[posts["tus"] > 0, "a"].nunique()
        a_kr = posts.loc[posts["tkr"] > 0, "a"].nunique()
        rec.update({"h46_n_us": n_us, "h46_n_kr": n_kr, "h46_auth_us": int(a_us), "h46_auth_kr": int(a_kr)})
        rec["h46_tilt"] = float(np.log(n_us / n_kr)) if (ok and n_us > 0 and n_kr > 0) else np.nan
        rec["h46_tilt_auth"] = float(np.log(a_us / a_kr)) if (ok and a_us > 0 and a_kr > 0) else np.nan
        # H50 M: (D 08:00, W_U 마감] 메모리 글 비율. 여는 괄호는 W_KR '(07:59' 읽기와 같게 그 분을 뺀다 → 08:01부터
        m_o = datetime.combine(D, time(8, 1))
        mwin = F.get(m_o, c)
        mp = mwin[mwin["utype"] == "post"]
        m_complete = F.complete(m_o, c, F_T_END)
        rec.update({"h50_m_open": m_o, "h50_m_n_posts": int(len(mp)), "h50_m_n_mem": int((mp["mem"] > 0).sum()),
                    "h50_m_complete": m_complete})
        rec["h50_m_ratio"] = (rec["h50_m_n_mem"] / len(mp)) if (ok and m_complete and len(mp)) else np.nan
        # H49 S^E: d 15:31 ~ W_U 마감, 강건판은 개장 1분 전(09:29 ET까지)
        e_o = datetime.combine(D, time(15, 31))
        e_c_rob = et_to_kst(D, 9, 30)
        se = s_stats(F.get(e_o, c))
        se_r = s_stats(F.get(e_o, e_c_rob))
        e_complete = F.complete(e_o, c, F_T_END)
        rec.update({"h49_e_open": e_o, "h49_n_units": se["n_units"], "h49_n_units_r0929": se_r["n_units"],
                    "h49_s_e": se["S"] if (ok and e_complete) else np.nan,
                    "h49_s_e_r0929": se_r["S"] if (ok and F.complete(e_o, e_c_rob, F_T_END)) else np.nan})
        # H10 반도체 태그 글 D_SD
        sp = posts[posts["semi"] > 0]
        rec["h10_n_semi_posts"] = int(len(sp))
        rec["h10_n_semi_authors"] = int(sp["a"].nunique())  # 참고: '작성자 20명'을 태그 글 작성자로 읽으면 이 값
        rec["h10_n_nonzero_posts"] = int((sp["score"].notna() & (sp["score"] != 0)).sum())
        sc = author_scores(sp)
        semi_scores.append(sc if ok else None)
        # H43 Herd(미국 종목, W_U)
        hrec = herd_window(posts, US_STOCKS)
        for k in US_STOCKS:
            rec[f"h43_buy_{k}"] = hrec[k]["buy"] if ok else np.nan
            rec[f"h43_sell_{k}"] = hrec[k]["sell"] if ok else np.nan
            stock_rows.append({"table": "US_D", "win_id": r["win_id"], "D": D, "stock": k,
                               "n_posts_usable": len(posts), "h43_buy_auth": hrec[k]["buy_auth"],
                               "h43_sell_auth": hrec[k]["sell_auth"],
                               "h43_buy": rec[f"h43_buy_{k}"], "h43_sell": rec[f"h43_sell_{k}"]})
        rows.append(rec)
    out = pd.concat([flags, pd.DataFrame(rows)], axis=1)
    mask_outage(out, ["s_day_S", "s_day_E_A_p", "s_day_F_A_p", "s_day_E_A_c", "s_day_F_A_c"])
    valid46 = (~out["xholiday_mismatch"].to_numpy()) & (~out["outage"].to_numpy())
    out["h46_valid_win"] = valid46
    for src, dst in (("h46_tilt", ""), ("h46_tilt_auth", "_auth")):
        t5, dm, da = tilt_delta(out[src].to_numpy(float), valid46)
        out[f"h46_tilt5{dst}"], out[f"h46_dtilt{dst}"], out[f"h46_dtilt_alt{dst}"] = t5, dm, da
    for col in [c for c in out.columns if c.startswith("h46_tilt5") or c.startswith("h46_dtilt")]:
        out.loc[out["outage"], col] = np.nan
    mvalid = (~out["outage"].to_numpy()) & out["h50_m_complete"].to_numpy(bool)
    out["h50_m_z20"] = z_prior(out["h50_m_ratio"].to_numpy(float), mvalid, Z20_N, Z20_MIN)
    d = dsd_series(semi_scores, list(out["win_id"]), "US_D:h10")
    for c in d.columns:
        out[f"h10_{c}"] = d[c].to_numpy()
    v = ~out["outage"].to_numpy()
    out["h10_dsd_z60"] = z_prior(out["h10_dsd"].to_numpy(float), v, Z60_N, Z60_MIN)
    out["h10_dsd_prov40_z60"] = z_prior(out["h10_dsd_prov40"].to_numpy(float), v, Z60_N, Z60_MIN)
    add_herd_events(out, US_STOCKS)
    return out, stock_rows


F_T_END = datetime(2026, 10, 16)   # run()에서 로더 요약 t_end로 바꾼다


def first_trading_days_of_weeks(days: list[date]) -> list[date]:
    seen, out = set(), []
    for d in sorted(days):
        k = d.isocalendar()[:2]
        if k not in seen:
            seen.add(k)
            out.append(d)
    return out


def build_weekly(F: Frame, wdaily: pd.DataFrame, market: str, cal: dict) -> tuple[pd.DataFrame, list[dict]]:
    """주 창(국내 = 그 주 첫 KRX 거래일 07:59 마감, 미국 = 그 주 첫 NYSE 거래일 08:29 ET 마감; 보통 월요일).
    H39+H48 AS: s = 언급 글(작성자당 종목 3글 상한) ÷ 전체 글, AS = ln(s + 1e−4) − 직전 8주 평균.
    유니버스는 lexicon §6 별칭(현재 목록 = 강건판)뿐이다. KIND·예탁원 유니버스는 자료가 없어 못 만든다."""
    days = sorted(cal["krx" if market == "kr" else "nyse"])
    ftd = first_trading_days_of_weeks(days)
    close_of = {d: c for d, c in zip(pd.to_datetime(wdaily["D"]).dt.date, pd.to_datetime(wdaily["close_kst"]))}
    stocks = KR_STOCKS if market == "kr" else US_STOCKS
    tab = "KR_W" if market == "kr" else "US_W"
    recs, stock_rows = [], []
    for prev, cur in zip(ftd, ftd[1:]):
        if prev not in close_of or cur not in close_of:
            continue
        o, c = close_of[prev].to_pydatetime(), close_of[cur].to_pydatetime()
        if not F.complete(o, c, F_T_END):
            continue
        sub = F.get(o, c)
        posts = sub[sub["utype"] == "post"]
        d_open, d_close = number_boundary(F.pdt, F.pno, np.array([np.datetime64(o, "s"), np.datetime64(c, "s")]))
        length = (c - o).total_seconds() / 3600
        rec = {"win_id": f"{'KW' if market == 'kr' else 'UW'}{cur.isoformat()}", "D": cur, "P": prev,
               "open_kst": o, "close_kst": c, "length_h": length, "kind": "week",
               "n_posts_usable": int(len(posts)), "d_no_per_h": (d_close - d_open) / length,
               "first_day_not_monday": cur.weekday() != 0}
        al = posts["alias"].fillna("").str.split(",")
        for k in stocks:
            has = al.apply(lambda s: k in s) if len(posts) else pd.Series(dtype=bool)
            per_a = posts.loc[has, "a"].value_counts() if len(posts) else pd.Series(dtype=int)
            m = int(np.minimum(per_a.to_numpy(), AS_CAP).sum()) if len(per_a) else 0
            rec[f"h39_m_{k}"] = m
            rec[f"h39_m_uncapped_{k}"] = int(per_a.sum()) if len(per_a) else 0
        recs.append(rec)
    w = pd.DataFrame(recs)
    if w.empty:
        return w, stock_rows
    w["weekday"] = 0                                   # 주 창은 모두 '그 주 첫 거래일' 마감이라 한 기준선
    base = same_weekday_base(w["d_no_per_h"].to_numpy(float), w["weekday"].to_numpy(), np.ones(len(w), bool))
    w["base_no_per_h"] = base
    w["outage_unknown"] = ~np.isfinite(base)
    w["outage"] = np.isfinite(base) & (w["d_no_per_h"].to_numpy(float) < OUTAGE_RATIO * np.nan_to_num(base))
    w = w.drop(columns="weekday")
    valid = ~w["outage"].to_numpy()
    N = w["n_posts_usable"].to_numpy(float)
    elig_count = np.zeros(len(w), dtype=int)
    for k in stocks:
        m = w[f"h39_m_{k}"].to_numpy(float)
        s = np.where(valid & (N > 0), m / np.where(N > 0, N, 1), np.nan)
        lns = np.log(s + AS_EPS)
        w[f"h39_s_{k}"], w[f"h39_lns_{k}"] = s, lns
        as_ = np.full(len(w), np.nan)
        base_m = np.full(len(w), np.nan)
        hist_l: list[float] = []
        hist_m: list[float] = []
        for i in range(len(w)):
            if len(hist_l) >= AS_BASE_WEEKS and np.isfinite(lns[i]):
                as_[i] = lns[i] - float(np.mean(hist_l[-AS_BASE_WEEKS:]))
                base_m[i] = float(np.mean(hist_m[-AS_BASE_WEEKS:]))
            if np.isfinite(lns[i]):
                hist_l.append(float(lns[i]))
                hist_m.append(float(m[i]))
        elig = np.isfinite(base_m) & (base_m >= AS_MIN_MENTION)
        w[f"h39_base8_m_{k}"] = base_m
        w[f"h39_elig_{k}"] = elig
        w[f"h39_as_lex6_{k}"] = np.where(elig, as_, np.nan)     # 강건판(현재 목록) 값. 주 규칙 전
        elig_count += elig.astype(int)
    w["h39_n_elig"] = elig_count
    w["h39_week_ok15"] = elig_count >= AS_MIN_STOCKS
    for k in stocks:
        w[f"h39_as_{k}"] = np.where(w["h39_week_ok15"], w[f"h39_as_lex6_{k}"], np.nan)   # 문서 규칙 그대로
        for i, r in w.iterrows():
            stock_rows.append({"table": tab, "win_id": r["win_id"], "D": r["D"], "stock": k,
                               "n_posts_usable": r["n_posts_usable"], "h39_m": r[f"h39_m_{k}"],
                               "h39_m_uncapped": r[f"h39_m_uncapped_{k}"], "h39_s": r[f"h39_s_{k}"],
                               "h39_lns": r[f"h39_lns_{k}"], "h39_base8_m": r[f"h39_base8_m_{k}"],
                               "h39_elig": r[f"h39_elig_{k}"], "h39_as_lex6": r[f"h39_as_lex6_{k}"],
                               "h39_as": r[f"h39_as_{k}"], "h39_week_ok15": r["h39_week_ok15"]})
    w["table"] = tab
    return w, stock_rows


def build_weekend(F: Frame, cal: dict, wk: pd.DataFrame) -> pd.DataFrame:
    """H53 ΔS = S(토 06:00 ~ 월 06:59/07:59) − S(금 15:31 ~ 토 05:59). 월 마감 = CME 재개(일 18:00 ET) 직전."""
    krx = set(cal["krx"])
    lo = pd.Timestamp(F.first_post).to_pydatetime().date()
    hi = F_T_END.date()
    recs = []
    d = lo
    while d <= hi:
        if d.weekday() == 5:
            fri, sat, mon = d - timedelta(days=1), d, d + timedelta(days=2)
            (a_o, a_c), (b_o, b_c) = weekend_windows(sat)
            if F.complete(a_o, a_c, F_T_END) and F.complete(b_o, b_c, F_T_END):
                A, B = F.get(a_o, a_c), F.get(b_o, b_c)
                sa, sb = s_stats(A), s_stats(B)
                An, Bn = A[~(A["coin"] | A["pcoin"])], B[~(B["coin"] | B["pcoin"])]
                san, sbn = s_stats(An), s_stats(Bn)
                nxt = min(x for x in krx if x > fri) if any(x > fri for x in krx) else None
                # 그 주 창의 장애 표지: 주말을 품는 국내 본판 창
                wrow = wk[(pd.to_datetime(wk["open_kst"]) <= pd.Timestamp(b_c))
                          & (pd.to_datetime(wk["close_kst"]) > pd.Timestamp(a_o))]
                outage = bool(wrow["outage"].fillna(False).any()) if len(wrow) else False
                recs.append({
                    "table": "WKND", "win_id": f"WE{mon.isoformat()}", "D": mon, "P": fri,
                    "open_kst": a_o, "close_kst": b_c, "length_h": (b_c - a_o).total_seconds() / 3600,
                    "kind": "weekend_pair", "outage": outage,
                    "h53_fri_open": a_o, "h53_sat_split": a_c, "h53_mon_close": b_c,
                    "h53_fri_krx_closed": fri not in krx, "h53_mon_krx_closed": mon not in krx,
                    "h53_next_krx_day": nxt,
                    "h53_s_fri": sa["S"], "h53_s_wknd": sb["S"], "h53_n_units_fri": sa["n_units"],
                    "h53_n_units_wknd": sb["n_units"],
                    "h53_ds": (sb["S"] - sa["S"]) if not outage else np.nan,
                    "h53_s_fri_nocoin": san["S"], "h53_s_wknd_nocoin": sbn["S"],
                    "h53_ds_nocoin": (sbn["S"] - san["S"]) if not outage else np.nan,
                    "h53_n_units_fri_nocoin": san["n_units"], "h53_n_units_wknd_nocoin": sbn["n_units"],
                })
        d += timedelta(days=1)
    return pd.DataFrame(recs)


# ---------------------------------------------------------------- 결측 요약

FEATURES = {
    # 열 → (표, 가설, 사전 묶음 상태 키)
    "h06_dsd": ("KR_D", "H06·H09", "BULL/BEAR"), "h06_dsd_prov40": ("KR_D", "H06·H09 잠정", "BULL/BEAR"),
    "h06_sd_raw": ("KR_D", "H06·H09 재료", "BULL/BEAR"), "h06_B": ("KR_D", "H09 통제 B", "BULL/BEAR"),
    "h06_daf": ("KR_D", "H06 보조 D_AF", "BULL/BEAR"),
    "h51_s_nm": ("KR_D", "H51", "E·F"), "s_day_S": ("KR_D|US_D", "일 S(H49·H51 통제)", "E·F"),
    "h46_tilt": ("US_D", "H46 재료", "TILT"), "h46_tilt5": ("US_D", "H46 재료", "TILT"),
    "h46_dtilt": ("US_D", "H46 주", "TILT"), "h46_dtilt_alt": ("US_D", "H46 보조", "TILT"),
    "h46_tilt_auth": ("US_D", "H46 보조(작성자)", "TILT"), "h46_dtilt_auth": ("US_D", "H46 보조(작성자)", "TILT"),
    "h50_m_ratio": ("US_D", "H50 재료", "MEM"), "h50_m_z20": ("US_D", "H50 주(잔차화 전)", "MEM"),
    "h49_s_e": ("US_D", "H49(잔차화 전)", "E·F"), "h49_s_e_r0929": ("US_D", "H49 강건(09:29 ET)", "E·F"),
    "h10_dsd": ("US_D", "H10", "BULL/BEAR·SEMI"), "h10_dsd_prov40": ("US_D", "H10 잠정", "BULL/BEAR·SEMI"),
    "h10_dsd_z60": ("US_D", "H10 주(z60)", "BULL/BEAR·SEMI"),
    "h10_dsd_prov40_z60": ("US_D", "H10 잠정 z60", "BULL/BEAR·SEMI"),
    "h10_sd_raw": ("US_D", "H10 재료", "BULL/BEAR·SEMI"),
    "h53_ds": ("WKND", "H53(잔차화 전)", "E·F"), "h53_ds_nocoin": ("WKND", "H53 강건(코인 글 뺌)", "E·F"),
}


def missing_summary(feat: pd.DataFrame) -> dict:
    out = {}
    for col, (tabs, hyp, lex) in FEATURES.items():
        for tab in tabs.split("|"):
            x = feat.loc[feat["table"] == tab, col] if col in feat else pd.Series(dtype=float)
            if x.empty:
                continue
            out[f"{tab}:{col}"] = {"hyp": hyp, "lex": lex, "rows": int(len(x)), "n_ok": int(x.notna().sum()),
                                   "missing_rate": round(float(x.isna().mean()), 4)}
    # 종목 열은 묶어서
    for tab, stocks, pref in (("KR_D", KR_STOCKS, "h43_buy_"), ("US_D", US_STOCKS, "h43_buy_"),
                              ("KR_D", KR_STOCKS, "h43_sell_"), ("US_D", US_STOCKS, "h43_sell_"),
                              ("KR_D", KR_STOCKS, "h43_event_buy_"), ("US_D", US_STOCKS, "h43_event_buy_"),
                              ("KR_W", KR_STOCKS, "h39_as_lex6_"), ("US_W", US_STOCKS, "h39_as_lex6_"),
                              ("KR_W", KR_STOCKS, "h39_as_"), ("US_W", US_STOCKS, "h39_as_"),
                              ("KR_W", KR_STOCKS, "h39_s_"), ("US_W", US_STOCKS, "h39_s_")):
        sub = feat[feat["table"] == tab]
        cols = [pref + k for k in stocks if pref + k in sub]
        if not cols:
            continue
        v = sub[cols]
        out[f"{tab}:{pref}*"] = {"hyp": {"h43": "H43", "h39": "H39+H48·H41"}[pref[:3]], "rows": int(len(sub)),
                                 "cells": int(v.size), "n_ok": int(v.notna().sum().sum()),
                                 "missing_rate": round(float(v.isna().mean().mean()), 4),
                                 "per_stock_missing": {k: round(float(sub[pref + k].isna().mean()), 4)
                                                       for k in stocks if pref + k in sub}}
    return out


def missing_reasons(feat: pd.DataFrame) -> dict:
    """결측 사유를 겹치지 않게 앞 사유부터 센다(장애 → 예열 → 표·단위 부족 → 그 밖)."""
    kr, us = feat[feat["table"] == "KR_D"], feat[feat["table"] == "US_D"]
    kw, uw = feat[feat["table"] == "KR_W"], feat[feat["table"] == "US_W"]

    def cnt(mask):
        return int(np.asarray(mask, bool).sum())

    out = {}
    for tab, d, p in (("KR_D", kr, "h06"), ("US_D", us, "h10")):
        out_ = d["outage"].to_numpy(bool)
        pool = d[f"{p}_pool_windows"].to_numpy(float)
        votes = d[f"{p}_n_votes"].to_numpy(float)
        low = ~(votes >= DSD_MIN_VOTES)
        for col, need in ((f"{p}_dsd", DSD_POOL), (f"{p}_dsd_prov40", DSD_POOL_PROV_MIN)):
            na = d[col].isna().to_numpy()
            r1 = out_
            r2 = ~r1 & (pool < need)
            r3 = ~r1 & ~r2 & low
            out[f"{tab}:{col}"] = {"rows": len(d), "ok": cnt(~na), "outage": cnt(na & r1), "warmup_pool": cnt(na & r2),
                                   "votes_lt20": cnt(na & r3), "other": cnt(na & ~r1 & ~r2 & ~r3)}
        out[f"{tab}:{p}_votes_lt20_all_valid_windows"] = {"valid_windows": cnt(~out_), "lt20": cnt(~out_ & low),
                                                          "share": round(cnt(~out_ & low) / max(cnt(~out_), 1), 4)}
    for col in ("h10_dsd_z60", "h10_dsd_prov40_z60"):
        na = us[col].isna().to_numpy()
        src = us[col.replace("_z60", "")].notna().to_numpy()
        out[f"US_D:{col}"] = {"rows": len(us), "ok": cnt(~na), "source_values_available": cnt(src),
                              "need_prior_values": Z60_MIN}
    o = us["outage"].to_numpy(bool)
    for col in ("h46_dtilt", "h46_dtilt_alt", "h50_m_z20", "h46_tilt5"):
        na = us[col].isna().to_numpy()
        out[f"US_D:{col}"] = {"rows": len(us), "ok": cnt(~na), "outage": cnt(na & o), "warmup": cnt(na & ~o)}
    out["US_D:h46_mismatch_rows_carried"] = cnt(us["xholiday_mismatch"].to_numpy(bool) & ~o
                                                & us["h46_dtilt"].notna().to_numpy())
    for tab, w, stocks in (("KR_W", kw, KR_STOCKS), ("US_W", uw, US_STOCKS)):
        per = {}
        for k in stocks:
            lns_ok = w[f"h39_lns_{k}"].notna().to_numpy()
            base = w[f"h39_base8_m_{k}"].to_numpy(float)
            warm = ~np.isfinite(base)
            inel = np.isfinite(base) & (base < AS_MIN_MENTION)
            per[k] = {"ok_lex6": cnt(w[f"h39_as_lex6_{k}"].notna()), "warmup_8w": cnt(warm),
                      "ineligible_mean_lt5": cnt(inel), "no_value": cnt(~lns_ok)}
        out[f"{tab}:h39_as_lex6"] = {"weeks": len(w), "per_stock": per,
                                     "n_elig_max": int(w["h39_n_elig"].max()) if len(w) else 0,
                                     "weeks_ok15": cnt(w["h39_week_ok15"].fillna(False))}
    return out


def describe(feat: pd.DataFrame) -> dict:
    """값 분포(중앙값·사분위·최소·최대). 가격과 잇지 않은 포럼 값만."""
    out = {}
    for col, (tabs, _, _) in FEATURES.items():
        for tab in tabs.split("|"):
            x = feat.loc[feat["table"] == tab, col].dropna().astype(float) if col in feat else pd.Series(dtype=float)
            if len(x):
                out[f"{tab}:{col}"] = {"n": int(len(x)), "min": round(float(x.min()), 4),
                                       "q25": round(float(x.quantile(.25)), 4), "median": round(float(x.median()), 4),
                                       "q75": round(float(x.quantile(.75)), 4), "max": round(float(x.max()), 4)}
    return out


# ---------------------------------------------------------------- 실행

def run(results: Path = RESULTS, rebuild_hits: bool = False) -> dict:
    global F_T_END
    if sha256(results / "lexicon_v02.json") != LEX_V02_SHA or sha256(results / "lexicon_v02_full.json") != \
            LEX_V02_FULL_SHA:
        raise RuntimeError("사전 v0.2 지문이 고정값과 다르다")
    summary_l = json.loads((results / "loader_summary.json").read_text("utf-8"))
    F_T_END = datetime.fromisoformat(summary_l["snapshot"]["t_end"])
    units = pd.read_parquet(results / "units.parquet")
    hits_path = results / "feat_topic_unit_hits.parquet"
    if hits_path.exists() and not rebuild_hits:
        hits = pd.read_parquet(hits_path)
    else:
        hits = unit_hits(units, load_v02())
        hits.to_parquet(hits_path, index=False)
    F = Frame(units, hits)
    calp = pd.read_parquet(results / "loader_calendar.parquet")
    cal = {"krx": [d.date() for d in calp.loc[calp["krx"], "date"]],
           "nyse": [d.date() for d in calp.loc[calp["nyse"], "date"]]}
    wk_all = pd.read_parquet(results / "windows_kr.parquet")
    wk_all = wk_all[wk_all["variant"] == "main"].sort_values("close_kst").reset_index(drop=True)
    wk = wk_all[wk_all["complete"]].reset_index(drop=True)
    wu_all = pd.read_parquet(results / "windows_us.parquet")
    wu_all = wu_all[wu_all["variant"] == "main"].sort_values("close_kst").reset_index(drop=True)
    wu = wu_all[wu_all["complete"]].reset_index(drop=True)

    kr, kr_st = build_kr_daily(F, wk, cal)
    kr["table"] = "KR_D"
    us, us_st = build_us_daily(F, wu, cal)
    us["table"] = "US_D"
    kw, kw_st = build_weekly(F, wk_all, "kr", cal)    # 주 창 경계만 일 창 마감 시각에서 빌린다(완전성은 따로 본다)
    uw, uw_st = build_weekly(F, wu_all, "us", cal)
    we = build_weekend(F, cal, wk)
    feat = pd.concat([kr, us, kw, uw, we], ignore_index=True, sort=False)
    front = ["table", "win_id", "D", "P", "open_kst", "close_kst", "length_h", "kind"]
    feat = feat[front + [c for c in feat.columns if c not in front]]
    for c in ("D", "P", "h53_next_krx_day", "open_kst", "close_kst", "h51_nm_open", "h51_nm_close", "h50_m_open",
              "h49_e_open", "h53_fri_open", "h53_sat_split", "h53_mon_close"):
        if c in feat:
            feat[c] = pd.to_datetime(feat[c])
    for c in feat.columns:   # 불리언 표지: 표에 없는 행은 NA
        if feat[c].dtype == object and c not in ("table", "win_id", "kind", "h51_nm_kind"):
            vals = set(feat[c].dropna().map(type))
            if vals <= {bool, np.bool_}:
                feat[c] = feat[c].astype("boolean")
    feat.to_parquet(results / "feat_topic.parquet", index=False)
    st = pd.DataFrame(kr_st + us_st + kw_st + uw_st)
    st["D"] = pd.to_datetime(st["D"])
    st.to_parquet(results / "feat_topic_stock.parquet", index=False)

    ms = missing_summary(feat)
    lex_full = json.loads((results / "lexicon_v02_full.json").read_text("utf-8"))
    summary = {
        "inputs": {"units_rows": int(len(units)), "t_end": str(F_T_END), "lexicon_v02_sha256": LEX_V02_SHA,
                   "lexicon_v02_full_sha256": LEX_V02_FULL_SHA,
                   "unit_hits_excluded": {k: int(v) for k, v in hits["excl"].value_counts().items()}},
        "rows": {t: int((feat["table"] == t).sum()) for t in ("KR_D", "US_D", "KR_W", "US_W", "WKND")},
        "lexicon_status": {k: lex_full["status"][k] for k in ("E", "F", "BULL", "BEAR", "CHASE_BUY", "CHASE_SELL",
                                                              "MEM", "SEMI", "TILT_US", "TILT_KR", "ALIAS_SEC",
                                                              "ALIAS_SKH", "ALIAS_BOTH", "ALIAS_US")},
        "missing": ms,
        "dsd": {
            "kr_pool_windows_max": int(kr["h06_pool_windows"].max()),
            "kr_votes": {"median": float(kr["h06_n_votes"].median()), "min": float(kr["h06_n_votes"].min()),
                         "lt20": int((kr["h06_n_votes"] < DSD_MIN_VOTES).sum())},
            "us_semi_votes": {"median": float(us["h10_n_votes"].median()), "min": float(us["h10_n_votes"].min()),
                              "lt20": int((us["h10_n_votes"] < DSD_MIN_VOTES).sum())},
            "kr_serial": {"max_authors": int(kr["h06_n_serial"].max()),
                          "median_in_window": float(kr["h06_n_serial_in_win"].median())},
        },
        "hit_units": {c: int((hits[c] > 0).sum()) for c in STRAT_COL.values()},
        "missing_reasons": missing_reasons(feat),
        "describe": describe(feat),
    }
    (results / "feat_topic_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1,
                                                                default=str), "utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="py feat_topic.py")
    ap.add_argument("--rebuild-hits", action="store_true", help="단위별 v0.2 적중 캐시를 다시 만든다")
    a = ap.parse_args(argv)
    s = run(rebuild_hits=a.rebuild_hits)
    print(json.dumps(s, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
