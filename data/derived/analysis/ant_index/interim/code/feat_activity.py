"""개미 지수 특징 계산: 활동량·작성자·댓글 구조 (HYPOTHESES.md §2·§3, 부록 1 동결판 그대로).

대상: H03 A, H04 U·B⊥, H08 Fight, H52 WK, H35 NW, H36 NEW, H37 단골 전향(v0.2 글 점수), H40·H34 준비값,
결번율·장애·스팸 표지.

읽기만 한다: interim/results/windows_kr.parquet·windows_us.parquet·units.parquet·loader_calendar.parquet(날짜만),
사전 v0.2(lexicon_v02_full.json). 가격 값은 읽지도 만들지도 않는다(앞날 주가 방화벽). 조회·추천·다른의견은 쓰지 않는다
(units에 없다). ncomment는 로더가 만든 첫 줄 잘림 표지(trunc_first)로만 쓴다(audit_leak.md: 마지막 줄 표지는
마감 뒤 붙은 댓글 수에 기대어 그 글이 마감 뒤 받은 주목으로 표본을 고르게 된다).

기준선(rz8·z60)·확장 회귀·작성자 이력은 모두 그 창 앞의 자료로만 계산한다(자기 제외).

산출(interim/results/):
  feat_activity.parquet            행 = 완전한 본판 창(국내 W_KR, 미국 W_U), 열 = 특징·표지
  feat_activity_weekend.parquet    H52 주말 창(토 06:00 ~ CME 재개 전) 한 줄씩
  feat_activity_h40_weekly.parquet H40 준비값(주 창 × 종목)
  feat_activity_summary.json       결측률·자기상관·가격 전 점검 수치
  feat_activity.md                 정의·결측 메모(사람용)

실행: py feat_activity.py   (이 폴더에서)
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lexicon_v02 as L  # noqa: E402
import loader as LD  # noqa: E402
from lexicon_v02_final import load_v02  # noqa: E402

INTERIM = Path(__file__).resolve().parents[1]
RESULTS = INTERIM / "results"
ET = ZoneInfo("America/New_York")

# ---------------------------------------------------------------- 고정값(근거는 notes_feat_activity.md)
RZ_N = 8                 # §2.0 rz8: 같은 요일 직전 8창
RZ_MIN_REG = 8           # 등록판: 직전 기준 창이 8개 다 있어야 한다(모자라면 결측)
RZ_MIN_LITE = 4          # 비등록 중간 점검판(_m4): 4개 이상이면 계산. 판정에 쓰지 않는다
MAD_K = 1.4826           # MAD → 표준편차 눈금
CLIP = 6.0               # ±6에서 자른다
Z60_N, Z60_MIN = 60, 40  # §2.0 z60
BPERP_MIN = 20           # B⊥ 확장 회귀 최소 창 수(문서에 없음, notes)
POST_CAP = 5             # H03 저장 글 수판: 작성자당 창 5건 상한
PAGE_ROOTS = 10          # 댓글 첫 쪽 = 가장 이른 뿌리 댓글 10개(+대댓글)(prechecks §3: 잘린 글 98.4%가 뿌리 10개). _pc 강건판용
HIST_REG, HIST_LITE = 90, 30   # H35·H36 이력 일수(등록 90일, 비등록 30일판)
REG_DAYS, REG_MIN, REG_THR = 30, 3, 0.5   # H37 단골: 직전 30일 점수 ≠ 0 글 3개 이상, 평균 ±0.5
BLOCK, DEN_MIN = 5, 5    # H37: 5창 누적, 분모 5 미만이면 결측
RT_START = datetime(2026, 10, 13)  # 실시간 수집 시작일(정확한 시각 미기록 → 이날 0시 이후 열린 창을 실시간으로 표시)

US_WORD_TO_TICKER = {   # lexicon_draft §6 미국 AI 별칭 표(현재 목록, 동결) → 종목
    "엔비디아": "NVDA", "엔비": "NVDA", "엔비디": "NVDA", "NVDA": "NVDA", "젠슨": "NVDA",
    "마이크론": "MU", "마크": "MU", "MU": "MU",
    "샌디스크": "SNDK", "샌디": "SNDK", "SNDK": "SNDK",
    "브로드컴": "AVGO", "브컴": "AVGO", "AVGO": "AVGO",
    "TSMC": "TSM", "티에스엠": "TSM",
    "AMD": "AMD",
    "웨스턴디지털": "WDC", "WDC": "WDC",
    "DELL": "DELL",
    "팔란티어": "PLTR", "팔란": "PLTR", "PLTR": "PLTR",
    "테슬라": "TSLA", "테슬": "TSLA", "TSLA": "TSLA",
}
KR_STOCKS = ["SEC", "SKH"]
US_STOCKS = ["NVDA", "MU", "SNDK", "AVGO", "TSM", "AMD", "WDC", "DELL", "PLTR", "TSLA"]

UNIT_COLS = ["no", "dt_kst", "a", "title", "text", "utype", "cid", "dt_conf", "depth", "root_cid", "post_dt_kst",
             "trunc_first", "win_kr", "pwin_kr", "pre_close_kr", "win_u", "pwin_u", "pre_close_u"]


# ================================================================ 공통 표준화

def rz8(values, weekday, base_ok, min_base=RZ_MIN_REG, n_base=RZ_N, k=MAD_K, clip=CLIP) -> np.ndarray:
    """같은 요일 직전 n_base개 기준 창(base_ok)의 중앙값·MAD로 표준화, ±clip. 자기 제외, 앞 자료만.
    기준 창이 min_base개 미만이거나 MAD = 0이면 결측. 값이 결측인 창은 기준에도 넣지 않는다."""
    x = np.asarray(values, dtype=float)
    out = np.full(len(x), np.nan)
    hist: dict[int, list[float]] = defaultdict(list)
    for i, (wd, ok, v) in enumerate(zip(weekday, base_ok, x)):
        h = hist[wd]
        if np.isfinite(v) and len(h) >= min_base:
            b = np.asarray(h[-n_base:])
            med = np.median(b)
            mad = np.median(np.abs(b - med))
            if mad > 0:
                out[i] = float(np.clip((v - med) / (k * mad), -clip, clip))
        if ok and np.isfinite(v):
            h.append(float(v))
    return out


def rz_seq(values, base_ok, min_base=RZ_MIN_REG, n_base=RZ_N, k=MAD_K, clip=CLIP) -> np.ndarray:
    """요일 구분 없이 직전 n_base개(주말 창 rz8용: '직전 8주말')."""
    return rz8(values, np.zeros(len(values), dtype=int), base_ok, min_base, n_base, k, clip)


def z_trailing(values, base_ok, n=Z60_N, min_n=Z60_MIN) -> np.ndarray:
    """§2.0 z60: 직전 n창(자기 제외)의 평균·SD로 z, min_n창 미만이면 결측."""
    x = np.asarray(values, dtype=float)
    out = np.full(len(x), np.nan)
    h: list[float] = []
    for i, (ok, v) in enumerate(zip(base_ok, x)):
        if np.isfinite(v) and len(h) >= min_n:
            b = np.asarray(h[-n:])
            sd = b.std(ddof=1)
            if sd > 0:
                out[i] = (v - b.mean()) / sd
        if ok and np.isfinite(v):
            h.append(float(v))
    return out


def expanding_resid(y, x, fit_ok, min_n=BPERP_MIN) -> tuple[np.ndarray, np.ndarray]:
    """y를 x에 확장 OLS(상수 포함)로 회귀한 잔차. i번째 창의 계수는 i 앞 창(fit_ok, 둘 다 유한)만으로 적합한다.
    반환: (잔차, 잔차 ÷ 적합 표본 잔차 SD)."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    res, resz = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for i in range(len(y)):
        if not (np.isfinite(y[i]) and np.isfinite(x[i])):
            continue
        m = np.zeros(len(y), bool)
        m[:i] = True
        m &= np.isfinite(y) & np.isfinite(x) & np.asarray(fit_ok, bool)
        if m.sum() < min_n:
            continue
        X = np.column_stack([np.ones(m.sum()), x[m]])
        beta, *_ = np.linalg.lstsq(X, y[m], rcond=None)
        e = y[m] - X @ beta
        sd = e.std(ddof=2) if m.sum() > 2 else np.nan
        res[i] = y[i] - (beta[0] + beta[1] * x[i])
        resz[i] = res[i] / sd if sd and np.isfinite(sd) and sd > 0 else np.nan
    return res, resz


# ================================================================ 댓글 싸움(H08)

def alt_len_pair(seq: list[str]) -> int:
    """줄기 안 두 해시 쌍의 '번갈아' 길이 최대값: 두 해시의 댓글만 남기고 연속 중복을 합친 길이(X→Y→X = 3)."""
    cnt = Counter(seq)
    if len(cnt) < 2:
        return 1 if seq else 0
    best = 0
    for x in (a for a, c in cnt.items() if c >= 2):
        for y in cnt:
            if y == x:
                continue
            n, last = 0, None
            for t in seq:
                if (t == x or t == y) and t != last:
                    n += 1
                    last = t
            best = max(best, n)
    return max(best, 2)


def alt_adjacent(seq: list[str]) -> bool:
    """강건판: 끼어든 사람 없이 바로 이어 X, Y, X(seq[i] == seq[i+2] != seq[i+1])."""
    return any(seq[i] == seq[i + 2] != seq[i + 1] for i in range(len(seq) - 2))


def fight_by_post(cm: pd.DataFrame) -> pd.DataFrame:
    """cm: 글별 마감 전 댓글(no, root_cid, dt_kst, cid, a). 글마다 싸움 줄기 유무(쌍판·인접판)."""
    cm = cm.sort_values(["no", "root_cid", "dt_kst", "cid"], kind="stable")
    rec = {}
    for (no, _root), seq in cm.groupby(["no", "root_cid"], sort=False)["a"]:
        s = seq.tolist()
        if len(s) < 3:
            continue
        r = rec.setdefault(no, [False, False])
        if not r[0] and alt_len_pair(s) >= 3:
            r[0] = True
        if not r[1] and alt_adjacent(s):
            r[1] = True
    return pd.DataFrame([(k, v[0], v[1]) for k, v in rec.items()], columns=["no", "fight", "fight_adj"])


# ================================================================ 재표집 기대값(H40)

def expected_unique(counts: np.ndarray, n: int) -> float:
    """작성자별 단위 수 counts인 풀에서 n개를 비복원으로 뽑을 때 고유 작성자 수의 기대값(정확식 = 무한 재표집)."""
    m = np.asarray(counts, dtype=float)
    N = m.sum()
    if n <= 0:
        return 0.0
    if n >= N:
        return float(len(m))
    a = N - m
    lp = np.full(len(m), -np.inf)
    ok = a >= n
    lp[ok] = gammaln(a[ok] + 1) - gammaln(a[ok] - n + 1) - gammaln(N + 1) + gammaln(N - n + 1)
    return float(np.sum(1.0 - np.exp(lp)))


# ================================================================ 자료 읽기

def load_inputs(results: Path = RESULTS) -> dict:
    wk = pd.read_parquet(results / "windows_kr.parquet")
    wu = pd.read_parquet(results / "windows_us.parquet")
    u = pd.read_parquet(results / "units.parquet", columns=UNIT_COLS)
    cal = pd.read_parquet(results / "loader_calendar.parquet")
    summ = json.loads((results / "loader_summary.json").read_text("utf-8"))
    return {"wk": wk, "wu": wu, "u": u, "cal": cal, "t_end": pd.Timestamp(summ["snapshot"]["t_end"])}


def politics_only(title, text, utype) -> bool:
    """lexicon §2 정치 잡담(정치어 2개 이상 + 시장어 0개). lexicon_v02.analyze와 같은 규칙."""
    t = ((title or "") + "\n" + (text or "")) if utype == "post" else (text or "")
    c = L._compact(t)
    return sum(c.count(p) for p in L.POLITICS) >= 2 and not any(m in c for m in L.MARKET)


def run_lexicon(u: pd.DataFrame) -> pd.DataFrame:
    """v0.2 매처를 모든 단위에 한 번 돌린다: 글 점수(BULL·BEAR), 종목 언급, 정치 전용 여부."""
    analyze = load_v02()
    score, stocks, pol, excl = [], [], [], []
    for t, b, ut in zip(u["title"].tolist(), u["text"].tolist(), u["utype"].tolist()):
        r = analyze(t, b, ut)
        score.append(r.score if ut == "post" else None)
        ss = set()
        for h in r.hits:
            if h.stratum == "ALIAS_SEC":
                ss.add("SEC")
            elif h.stratum == "ALIAS_SKH":
                ss.add("SKH")
            elif h.stratum == "ALIAS_BOTH":
                ss.update(("SEC", "SKH"))
            elif h.stratum == "ALIAS_US":
                tk = US_WORD_TO_TICKER.get(h.word)
                if tk is None:
                    raise KeyError(f"ALIAS_US 단어 '{h.word}'의 종목 대응이 없다")
                ss.add(tk)
        stocks.append(tuple(sorted(ss)))
        pol.append(politics_only(t, b, ut))
        excl.append(r.excluded)
    return pd.DataFrame({"score": pd.array(score, dtype="Int64"), "stocks": stocks, "politics_only": pol,
                         "lex_excluded": excl}, index=u.index)


# ================================================================ 창별 특징

def window_frame(w: pd.DataFrame) -> pd.DataFrame:
    """완전한 본판 창만, 마감 순."""
    f = w[(w["variant"] == "main") & w["complete"]].sort_values("close_kst").reset_index(drop=True).copy()
    f["outage_b"] = f["outage"].fillna(False).astype(bool)
    f["outage_unknown"] = f["outage"].isna()
    f["spam_b"] = f["spam_wave"].fillna(False).astype(bool)
    # 기준선에 넣을 수 있는 창: 연휴 뒤·96시간 초과·장애 창 제외(§2.0·§2.4)
    f["base_ok"] = (f["main_kind"] != "holiday") & ~f["over96"] & ~f["outage_b"]
    # 판정 자격 표지: 96시간 초과·장애 창 제외(값은 남기고 표지만)
    f["judge_ok"] = ~f["over96"] & ~f["outage_b"]
    f["realtime"] = f["open_kst"] >= RT_START
    return f


def activity_features(f: pd.DataFrame, u: pd.DataFrame, idc: str, pidc: str, prec: str) -> pd.DataFrame:
    """H03 A·저장판, H04 U·N·B⊥, H08 Fight, 댓글 수·잘림. f: window_frame, idc: 단위 자기 창 열."""
    f = f.copy()
    ui = u[u[idc].notna()]
    g = ui.groupby(idc)
    posts = ui[ui["utype"] == "post"]
    # H03 저장 글 수판: 작성자당 창 5건 상한
    capped = posts.groupby([idc, "a"]).size().clip(upper=POST_CAP).groupby(level=0).sum()
    f["saved_cap5"] = f["win_id"].map(capped).fillna(0).astype(float)
    f["U"] = f["win_id"].map(g["a"].nunique()).fillna(0).astype(float)
    f["N"] = f["win_id"].map(g.size()).fillna(0).astype(float)
    f["n_comments_win"] = f["win_id"].map(ui[ui["utype"] == "comment"].groupby(idc).size()).fillna(0)
    # 시간당(§2.4 '양 지표는 시간당'), 장애 창은 모든 특징에서 결측(§2.4)
    out = f["outage_b"].to_numpy()
    for col, src in (("A_raw", "d_no_per_h"),):
        f[col] = np.where(out, np.nan, f[src].astype(float))
    for col, src in (("saved_cap5_per_h", "saved_cap5"), ("U_per_h", "U"), ("N_per_h", "N")):
        f[col] = np.where(out, np.nan, f[src] / f["length_h"])
    f["ln_n_comments_win"] = np.where(out, np.nan, np.log(f["n_comments_win"].clip(lower=1)))
    wd, bo = f["weekday"].to_numpy(), f["base_ok"].to_numpy()
    for col, name in (("A_raw", "A"), ("saved_cap5_per_h", "A_saved"), ("U_per_h", "U"), ("N_per_h", "N")):
        f[f"{name}_rz8"] = rz8(f[col], wd, bo, RZ_MIN_REG)
        f[f"{name}_rz8m4"] = rz8(f[col], wd, bo, RZ_MIN_LITE)
    fit_ok = f["judge_ok"].to_numpy()
    for sfx in ("rz8", "rz8m4"):
        r, rz = expanding_resid(f[f"U_{sfx}"], f[f"N_{sfx}"], fit_ok, BPERP_MIN)
        f[f"Bperp_{sfx}"], f[f"Bperp_{sfx}_sd"] = r, rz

    # H08 Fight: 창에 등록된 글, 그 글의 부모 창 마감 전 댓글만
    wp = posts[["no", idc, "trunc_first"]].rename(columns={idc: "wid"})
    cm = u[(u["utype"] == "comment") & u[prec].astype(bool) & u[pidc].notna()]
    fb = fight_by_post(cm[["no", "root_cid", "dt_kst", "cid", "a"]])
    wp = wp.merge(fb, on="no", how="left")
    wp[["fight", "fight_adj"]] = wp[["fight", "fight_adj"]].fillna(False).astype(bool)
    wp["trunc"] = wp["trunc_first"].fillna(False).astype(bool)   # 첫 줄 표지(마감에 가장 가까운 줄, audit_leak.md)
    nt = wp[~wp["trunc"]]
    gp, gn = wp.groupby("wid"), nt.groupby("wid")
    f["n_posts_nontrunc"] = f["win_id"].map(gn.size()).fillna(0)
    f["n_fight_posts_nontrunc"] = f["win_id"].map(gn["fight"].sum()).fillna(0)
    f["fight"] = np.where(out | (f["n_posts_nontrunc"] == 0), np.nan,
                          f["n_fight_posts_nontrunc"] / f["n_posts_nontrunc"].replace(0, np.nan))
    f["fight_all"] = np.where(out, np.nan, f["win_id"].map(gp["fight"].mean()))
    f["fight_adj"] = np.where(out, np.nan, f["win_id"].map(gn["fight_adj"].mean()))
    f["fight_z60"] = z_trailing(f["fight"], f["base_ok"].to_numpy())
    # 감사 강건판 _pc(등록 정의 아님, 설명용. audit_leak.md): 잘림을 마감 전 정보로만 정한다. 첫 쪽이 가장 이른 뿌리 10개라
    # 마감 전 뿌리 댓글이 10개 이상 저장된 글만 '그 창 마감 전 댓글이 잘렸을 수 있는 글'로 본다. ncomment(수집 때까지의
    # 댓글 수, 소급분은 마감 뒤 몇 주 치)를 쓰지 않아 마감 뒤 주목이 표본 고르기에 들어오지 않는다
    nroot = cm[cm["depth"].fillna(0) == 0].groupby("no").size()
    wp["trunc_pc"] = wp["no"].map(nroot).fillna(0).to_numpy() >= PAGE_ROOTS
    npc = wp[~wp["trunc_pc"]]
    gpc = npc.groupby("wid")
    f["trunc_share_pc"] = np.where(out, np.nan, f["win_id"].map(wp.groupby("wid")["trunc_pc"].mean()))
    f["n_posts_nontrunc_pc"] = f["win_id"].map(gpc.size()).fillna(0)
    f["fight_pc"] = np.where(out | (f["n_posts_nontrunc_pc"] == 0), np.nan,
                             f["win_id"].map(gpc["fight"].sum()).fillna(0) / f["n_posts_nontrunc_pc"].replace(0, np.nan))
    f["fight_adj_pc"] = np.where(out, np.nan, f["win_id"].map(gpc["fight_adj"].mean()))
    cw = cm.groupby(pidc)
    f["n_comments_onposts"] = f["win_id"].map(cw.size()).fillna(0)
    f["ln_n_comments_onposts"] = np.where(out, np.nan, np.log(f["n_comments_onposts"].clip(lower=1)))
    f["fight_c_conf_share"] = f["win_id"].map(cw["dt_conf"].apply(lambda s: float((s != "latest").mean())))
    return f


def author_history(u: pd.DataFrame) -> dict[str, np.ndarray]:
    """작성자 해시 → 단위 시각(정렬, datetime64[s]). 이력 판정용(그날까지의 자료만 꺼내 쓴다)."""
    t = u["dt_kst"].to_numpy(dtype="datetime64[s]")
    order = np.argsort(t, kind="stable")
    a = u["a"].to_numpy()[order]
    t = t[order]
    out: dict[str, list] = defaultdict(list)
    for ai, ti in zip(a, t):
        out[ai].append(ti)
    return {k: np.asarray(v, dtype="datetime64[s]") for k, v in out.items()}


def newcomer_features(f: pd.DataFrame, u: pd.DataFrame, lex: pd.DataFrame, idc: str, c0: pd.Timestamp) -> pd.DataFrame:
    """H35 NW(신참 비율), H36 NEW(신참 수, 시간당). 이력 = [창 열림 − L일, 창 열림). 이력 구간이 자료 시작(c0) 앞이면 결측."""
    f = f.copy()
    hist = author_history(u)
    ui = u[u[idc].notna()].assign(pol=lex.loc[u[idc].notna(), "politics_only"].to_numpy())
    # 정치 전용 단위만 쓴 작성자는 뺀다(H35)
    keep_auth = ui.groupby([idc, "a"])["pol"].all()
    keep_auth = keep_auth[~keep_auth]
    auth_by_win: dict[str, list[str]] = defaultdict(list)
    for (wid, a) in keep_auth.index:
        auth_by_win[wid].append(a)
    for L_days, sfx in ((HIST_REG, "90"), (HIST_LITE, "30")):
        nw, new, nauth = [], [], []
        for wid, o in zip(f["win_id"], f["open_kst"]):
            o = pd.Timestamp(o)
            lo = o - pd.Timedelta(days=L_days)
            if lo < c0:
                nw.append(np.nan), new.append(np.nan), nauth.append(np.nan)
                continue
            lo64, o64 = np.datetime64(lo.to_pydatetime(), "s"), np.datetime64(o.to_pydatetime(), "s")
            auths = auth_by_win.get(wid, [])
            n_new = 0
            for a in auths:
                ts = hist[a]
                j = np.searchsorted(ts, lo64, "left")
                if not (j < len(ts) and ts[j] < o64):
                    n_new += 1
            nauth.append(len(auths))
            nw.append(n_new / len(auths) if auths else np.nan)
            new.append(n_new)
        f[f"n_authors_nonpol"] = f["win_id"].map({k: len(v) for k, v in auth_by_win.items()}).fillna(0)
        f[f"NW{sfx}"] = np.where(f["outage_b"], np.nan, nw)
        f[f"NEW{sfx}_per_h"] = np.where(f["outage_b"], np.nan, np.asarray(new, float) / f["length_h"])
    wd, bo = f["weekday"].to_numpy(), f["base_ok"].to_numpy()
    for sfx in ("90", "30"):
        f[f"NW{sfx}_rz8"] = rz8(f[f"NW{sfx}"], wd, bo, RZ_MIN_REG)
        f[f"NW{sfx}_rz8m4"] = rz8(f[f"NW{sfx}"], wd, bo, RZ_MIN_LITE)
        f[f"NEW{sfx}_rz8"] = rz8(f[f"NEW{sfx}_per_h"], wd, bo, RZ_MIN_REG)
        f[f"NEW{sfx}_rz8m4"] = rz8(f[f"NEW{sfx}_per_h"], wd, bo, RZ_MIN_LITE)
    return f


def regular_conversion(f: pd.DataFrame, u: pd.DataFrame, lex: pd.DataFrame, c0: pd.Timestamp) -> pd.DataFrame:
    """H37: 5창 묶음(D−4..D)의 시작 시각 기준으로 직전 30일 글 점수로 단골을 정하고, 묶음 안 글로 전향률을 잰다."""
    f = f.copy()
    p = u[u["utype"] == "post"][["a", "dt_kst"]].assign(score=lex.loc[u["utype"] == "post", "score"].to_numpy())
    p = p.sort_values("dt_kst", kind="stable")
    pt = p["dt_kst"].to_numpy(dtype="datetime64[s]")
    cols = {k: [] for k in ("h37_n_bear_reg", "h37_n_bull_reg", "h37_den_bear", "h37_num_bull_conv",
                            "h37_den_bull", "h37_num_bear_conv")}
    opens = f["open_kst"].to_numpy(dtype="datetime64[s]")
    closes = f["close_kst"].to_numpy(dtype="datetime64[s]")
    for i in range(len(f)):
        if i < BLOCK - 1:
            for k in cols:
                cols[k].append(np.nan)
            continue
        start, end = opens[i - BLOCK + 1], closes[i]
        lo = start - np.timedelta64(REG_DAYS, "D")
        if pd.Timestamp(lo) < c0:
            for k in cols:
                cols[k].append(np.nan)
            continue
        h = p.iloc[np.searchsorted(pt, lo, "left"):np.searchsorted(pt, start, "left")]
        h = h[h["score"].notna() & (h["score"] != 0)]
        st = h.groupby("a")["score"].agg(["size", "mean"])
        st = st[st["size"] >= REG_MIN]
        bear = set(st.index[st["mean"] <= -REG_THR])
        bull = set(st.index[st["mean"] >= REG_THR])
        b = p.iloc[np.searchsorted(pt, start, "left"):np.searchsorted(pt, end, "left")]
        wrote = set(b["a"])
        pos = set(b.loc[b["score"].fillna(0) > 0, "a"])
        neg = set(b.loc[b["score"].fillna(0) < 0, "a"])
        cols["h37_n_bear_reg"].append(len(bear))
        cols["h37_n_bull_reg"].append(len(bull))
        cols["h37_den_bear"].append(len(bear & wrote))
        cols["h37_num_bull_conv"].append(len(bear & pos))
        cols["h37_den_bull"].append(len(bull & wrote))
        cols["h37_num_bear_conv"].append(len(bull & neg))
    for k, v in cols.items():
        f[k] = v
    den_b, den_u = f["h37_den_bear"].astype(float), f["h37_den_bull"].astype(float)
    f["h37_bull_conv_rate"] = np.where(den_b >= DEN_MIN, f["h37_num_bull_conv"] / den_b.replace(0, np.nan), np.nan)
    f["h37_bear_conv_rate"] = np.where(den_u >= DEN_MIN, f["h37_num_bear_conv"] / den_u.replace(0, np.nan), np.nan)
    return f


def _sec(t) -> np.ndarray:
    return np.asarray(pd.to_datetime(pd.Series(t)).to_numpy(dtype="datetime64[s]").astype(np.int64), float)


def number_interp(posts: pd.DataFrame):
    """지운 번호의 시각을 앞뒤 저장 글 시각으로 선형 보간한 누적 번호 함수 N(t; close)(H34).
    창 마감 뒤 글은 쓰지 않는다: 보간 점은 dt < close인 저장 글(분별 최대 번호의 누적 최대)뿐이고,
    마지막 점 뒤는 그 값(= 경계 번호 B(close))에 고정된다. 그래서 마감 직전의 지운 번호를 마감 뒤 글로 끌어오지 않는다."""
    s = posts.groupby("dt_kst")["no"].max().sort_index()
    xp = _sec(s.index)
    fp = np.maximum.accumulate(s.to_numpy().astype(float))

    def N(t, close) -> np.ndarray:
        k = int(np.searchsorted(xp, _sec([close])[0], "left"))
        if k == 0:
            return np.full(len(t), np.nan)
        return np.interp(_sec(t), xp[:k], fp[:k])
    return N


def night_blocks(o: datetime, c: datetime, nyse_days: set) -> list[tuple[datetime, datetime, bool]]:
    """창 [o, c) 안 KST 01:00~05:00(미국 서머타임) / 02:00~06:00(해제) 칸. 셋째 값 = 그 밤에 미국 정규장이 있었는지."""
    out = []
    d = o.date() - timedelta(days=1)
    while d <= c.date():
        probe = datetime(d.year, d.month, d.day, 3, 0, tzinfo=LD.KST).astimezone(ET)
        h0 = 1 if probe.dst() else 2
        b0, b1 = datetime.combine(d, time(h0)), datetime.combine(d, time(h0 + 4))
        s, e = max(b0, o), min(b1, c)
        if s < e:
            out.append((s, e, (d - timedelta(days=1)) in nyse_days))
        d += timedelta(days=1)
    return out


def night_share(f: pd.DataFrame, u: pd.DataFrame, nyse_days: set) -> pd.DataFrame:
    """H34 준비값: W_U 안 심야(01~05 KST, 서머타임) 몫. 번호는 보간, 댓글은 날짜 확정분만. 잔차화·20일 평균은 가격 단계."""
    f = f.copy()
    posts = u[u["utype"] == "post"][["no", "dt_kst"]]
    N = number_interp(posts)
    cm = u[(u["utype"] == "comment") & (u["dt_conf"] != "latest")]
    ct = np.sort(cm["dt_kst"].to_numpy(dtype="datetime64[s]"))

    def ccount(s, e):
        return int(np.searchsorted(ct, np.datetime64(e, "s"), "left") - np.searchsorted(ct, np.datetime64(s, "s"), "left"))

    rec = []
    for o, c, outage in zip(f["open_kst"], f["close_kst"], f["outage_b"]):
        o, c = pd.Timestamp(o).to_pydatetime(), pd.Timestamp(c).to_pydatetime()
        nb = night_blocks(o, c, nyse_days)

        def dn(s, e):
            v = N([s, e], c)
            return float(v[1] - v[0])
        tot_no = dn(o, c)
        tot_c = ccount(o, c)
        n_no = sum(dn(s, e) for s, e, _ in nb)
        n_c = sum(ccount(s, e) for s, e, _ in nb)
        s_no = sum(dn(s, e) for s, e, ss in nb if ss)
        s_c = sum(ccount(s, e) for s, e, ss in nb if ss)
        den = tot_no + tot_c
        rec.append({"ns_tot_no_interp": tot_no, "ns_tot_comments_conf": tot_c, "ns_night_no_interp": n_no,
                    "ns_night_comments_conf": n_c, "ns_night_hours": sum((e - s).total_seconds() / 3600 for s, e, _ in nb),
                    "NS_share": np.nan if outage or den <= 0 else (n_no + n_c) / den,
                    "NS_share_sess": np.nan if outage or den <= 0 else (s_no + s_c) / den})
    return pd.concat([f, pd.DataFrame(rec, index=f.index)], axis=1)


# ================================================================ 주말 창(H52)

def weekend_table(u: pd.DataFrame, wk_main: pd.DataFrame, cal: pd.DataFrame, c0: pd.Timestamp,
                  t_end: pd.Timestamp) -> pd.DataFrame:
    """WK = (토 06:00 ~ CME 재개 전(일 18:00 ET = 월 07:00/08:00 KST)) 시간당 (번호 증가 + 마감 전 댓글 수), 직전 8주말 rz8."""
    krx = set(cal.loc[cal["krx"], "date"].dt.date)
    posts = u[u["utype"] == "post"]
    pdt, pno = LD._ns(posts["dt_kst"]), posts["no"].to_numpy()
    ct = np.sort(u.loc[u["utype"] == "comment", "dt_kst"].to_numpy(dtype="datetime64[s]"))
    pt = np.sort(pdt)
    kr = wk_main[wk_main["variant"] == "main"].sort_values("close_kst")
    d0 = (c0 + pd.Timedelta(days=1)).date()
    sat = d0 + timedelta(days=(5 - d0.weekday()) % 7)
    rec = []
    while True:
        o = datetime.combine(sat, time(6))
        c = LD.et_to_kst(sat + timedelta(days=1), 18, 0)   # CME 재개 = 일 18:00 ET
        if pd.Timestamp(c) > t_end:
            break
        if pd.Timestamp(o) >= c0:
            mon, fri = sat + timedelta(days=2), sat - timedelta(days=1)
            b = LD.number_boundary(pdt, pno, LD._ns([o, c]))
            o64, c64 = np.datetime64(o, "s"), np.datetime64(c, "s")
            n_c = int(np.searchsorted(ct, c64, "left") - np.searchsorted(ct, o64, "left"))
            n_p = int(np.searchsorted(pt, c64, "left") - np.searchsorted(pt, o64, "left"))
            hours = (c - o).total_seconds() / 3600
            cont = kr[(kr["open_kst"] <= o) & (kr["close_kst"] > o)]
            chuseok = bool(cont["over96"].any()) if len(cont) else False
            mon_win = kr[kr["win_id"] == f"K{mon.isoformat()}"]
            rec.append({"sat": pd.Timestamp(sat), "mon": pd.Timestamp(mon), "open_kst": o, "close_kst": c,
                        "hours": hours, "d_no": float(b[1] - b[0]), "n_comments": n_c, "n_posts": n_p,
                        "mon_open": mon in krx, "fri_closed": fri not in krx, "seol_chuseok": chuseok,
                        "kr_win_id": mon_win["win_id"].iloc[0] if len(mon_win) else None,
                        "mon_win_kind": mon_win["main_kind"].iloc[0] if len(mon_win) else None})
        sat += timedelta(days=7)
    w = pd.DataFrame(rec)
    w["WK_raw"] = (w["d_no"] + w["n_comments"]) / w["hours"]
    w["WK_saved_raw"] = (w["n_posts"] + w["n_comments"]) / w["hours"]
    w["no_per_h"] = w["d_no"] / w["hours"]
    # 장애: 시간당 번호 증가가 직전 기준 주말(최대 8, 최소 3) 중앙값의 20% 미만
    w["long_weekend"] = w["fri_closed"] | ~w["mon_open"]
    base_pre = ~w["long_weekend"] & ~w["seol_chuseok"]
    med = np.full(len(w), np.nan)
    h: list[float] = []
    for i, (ok, v) in enumerate(zip(base_pre, w["no_per_h"])):
        if len(h) >= LD.BASE_MIN:
            med[i] = np.median(h[-LD.BASE_N:])
        if ok:
            h.append(v)
    w["base_no_per_h"] = med
    w["outage"] = pd.array(np.where(np.isfinite(med), w["no_per_h"] < LD.OUTAGE_RATIO * med, pd.NA), dtype="boolean")
    out_b = w["outage"].fillna(False).to_numpy()
    w["base_ok"] = base_pre & ~out_b
    # 본 값: 설·추석, 월요일 휴장(월요일 갭이 없다), 장애 주말은 결측
    drop = w["seol_chuseok"].to_numpy() | ~w["mon_open"].to_numpy() | out_b
    for col, name in (("WK_raw", "WK"), ("WK_saved_raw", "WK_saved")):
        v = np.where(drop, np.nan, w[col])
        w[f"{name}_rz8"] = rz_seq(v, w["base_ok"].to_numpy(), RZ_MIN_REG)
        w[f"{name}_rz8m4"] = rz_seq(v, w["base_ok"].to_numpy(), RZ_MIN_LITE)
    w["WK_val"] = np.where(drop, np.nan, w["WK_raw"])
    return w


# ================================================================ 주 창(H40 준비값)

def week_windows(days: list[date], market: str) -> list[tuple[date, datetime, datetime]]:
    """주 창 = (직전 주 첫 거래일 마감 시각, 이번 주 첫 거래일 마감 시각]. 국내 07:59, 미국 08:29 ET."""
    first = {}
    for d in sorted(days):
        k = d.isocalendar()[:2]
        first.setdefault(k, d)
    fs = sorted(first.values())
    out = []
    for p, d in zip(fs, fs[1:]):
        if market == "kr":
            out.append((d, datetime.combine(p, time(8)), datetime.combine(d, time(8))))
        else:
            out.append((d, LD.et_to_kst(p, 8, 30), LD.et_to_kst(d, 8, 30)))
    return out


def h40_weekly(u: pd.DataFrame, lex: pd.DataFrame, cal: pd.DataFrame, c0: pd.Timestamp, t_end: pd.Timestamp) -> pd.DataFrame:
    t = u["dt_kst"].to_numpy(dtype="datetime64[s]")
    order = np.argsort(t, kind="stable")
    ts = t[order]
    a_all = u["a"].to_numpy()[order]
    stocks_all = lex["stocks"].to_numpy()[order]
    is_post = (u["utype"].to_numpy() == "post")[order]
    rec = []
    for market, col, names in (("kr", "krx", KR_STOCKS), ("us", "nyse", US_STOCKS)):
        days = list(cal.loc[cal[col], "date"].dt.date)
        for d, o, c in week_windows(days, market):
            if pd.Timestamp(o) < c0 or pd.Timestamp(c) > t_end:
                continue
            i0 = np.searchsorted(ts, np.datetime64(o, "s"), "left")
            i1 = np.searchsorted(ts, np.datetime64(c, "s"), "left")
            a, st, ip = a_all[i0:i1], stocks_all[i0:i1], is_post[i0:i1]
            pool = np.asarray(list(Counter(a).values()), float)
            for s in names:
                m = np.fromiter((s in x for x in st), bool, count=len(st))
                n = int(m.sum())
                uu = len(set(a[m]))
                eu = expected_unique(pool, n) if n else np.nan
                pm = m & ip
                cap3 = int(pd.Series(a[pm]).value_counts().clip(upper=3).sum()) if pm.any() else 0
                rec.append({"market": market, "week_close_day": pd.Timestamp(d), "open_kst": o, "close_kst": c,
                            "stock": s, "n_units_pool": int(len(a)), "n_authors_pool": int(len(pool)),
                            "n_posts_pool": int(ip.sum()), "mention_units": n, "mention_authors": uu,
                            "mention_posts": int(pm.sum()), "mention_posts_cap3": cap3,
                            "E_unique_resample": eu, "Btilde": (uu / eu) if n and eu > 0 else np.nan})
    w = pd.DataFrame(rec).sort_values(["market", "stock", "week_close_day"]).reset_index(drop=True)
    w["mention_posts_cap3_avg8_prior"] = w.groupby(["market", "stock"])["mention_posts_cap3"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=8).mean())
    return w


# ================================================================ 요약

FEATURES_KR = ["A_raw", "A_rz8", "A_rz8m4", "A_saved_rz8", "A_saved_rz8m4", "U_per_h", "N_per_h", "U_rz8", "U_rz8m4",
               "N_rz8", "N_rz8m4", "Bperp_rz8", "Bperp_rz8m4", "fight", "fight_all", "fight_adj", "fight_z60",
               "ln_n_comments_win", "ln_n_comments_onposts", "trunc_share", "fight_c_conf_share",
               "fight_pc", "fight_adj_pc", "trunc_share_pc",
               "NW90", "NW90_rz8", "NEW90_per_h", "NEW90_rz8", "NW30", "NW30_rz8", "NW30_rz8m4",
               "NEW30_per_h", "NEW30_rz8", "NEW30_rz8m4", "h37_bull_conv_rate", "h37_bear_conv_rate",
               "WK_val", "WK_rz8", "WK_rz8m4", "WK_saved_rz8", "WK_saved_rz8m4", "missing_rate"]
FEATURES_US = ["A_raw", "A_rz8", "A_rz8m4", "A_saved_rz8", "A_saved_rz8m4", "U_per_h", "N_per_h", "U_rz8", "U_rz8m4",
               "N_rz8", "N_rz8m4", "Bperp_rz8", "Bperp_rz8m4", "fight", "fight_all", "fight_adj", "fight_z60",
               "fight_pc", "fight_adj_pc", "trunc_share_pc", "NS_share", "NS_share_sess", "missing_rate"]


def lag1(x: pd.Series) -> float | None:
    x = x.astype(float)
    a, b = x.iloc[1:].to_numpy(), x.iloc[:-1].to_numpy()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8 or a[m].std() == 0 or b[m].std() == 0:
        return None
    return float(np.corrcoef(a[m], b[m])[0, 1])


def corr_pair(a: pd.Series, b: pd.Series) -> dict:
    m = a.notna() & b.notna()
    if m.sum() < 5:
        return {"n": int(m.sum()), "pearson": None, "spearman": None}
    return {"n": int(m.sum()), "pearson": float(np.corrcoef(a[m], b[m])[0, 1]),
            "spearman": float(a[m].rank().corr(b[m].rank()))}


def summarize(fk: pd.DataFrame, fu: pd.DataFrame, wkend: pd.DataFrame, h40: pd.DataFrame) -> dict:
    out = {"n_windows": {"kr": int(len(fk)), "us": int(len(fu))},
           "n_judge_ok": {"kr": int(fk["judge_ok"].sum()), "us": int(fu["judge_ok"].sum())}, "features": {}}
    for mk, f, cols in (("kr", fk, FEATURES_KR), ("us", fu, FEATURES_US)):
        for c in cols:
            s = f[c].astype(float)
            j = s[f["judge_ok"]]
            first = f.loc[s.notna(), "D"].min()
            out["features"][f"{mk}:{c}"] = {
                "missing_all": float(s.isna().mean()), "missing_judge_ok": float(j.isna().mean()),
                "n_valid": int(s.notna().sum()), "first_valid_D": None if pd.isna(first) else str(first.date()),
                "lag1_ac": lag1(s)}
    checks = {}
    for sfx in ("rz8", "rz8m4"):
        checks[f"corr_U_N_{sfx}_kr"] = corr_pair(fk[f"U_{sfx}"], fk[f"N_{sfx}"])
        checks[f"corr_NEW30_A_{sfx}_kr"] = corr_pair(fk[f"NEW30_{sfx}"], fk[f"A_{sfx}"])
        checks[f"corr_A_Asaved_{sfx}_kr"] = corr_pair(fk[f"A_{sfx}"], fk[f"A_saved_{sfx}"])
    checks["corr_fight_lncomments_kr"] = corr_pair(fk["fight"], fk["ln_n_comments_onposts"])
    d = fk["h37_den_bear"].astype(float)
    comp = d.notna()
    checks["h37_den_ge5_share_of_computed"] = float((d[comp] >= DEN_MIN).mean()) if comp.any() else None
    checks["h37_den_ge5_share_of_all_kr"] = float((d >= DEN_MIN).mean())
    checks["h37_den_bear_median"] = float(d[comp].median()) if comp.any() else None
    checks["h37_den_bull_median"] = float(fk.loc[comp, "h37_den_bull"].astype(float).median()) if comp.any() else None
    pre, post = fk[~fk["realtime"]], fk[fk["realtime"]]
    checks["nw30_mean_before_rt"] = float(pre["NW30"].mean())
    checks["nw30_mean_rt"] = float(post["NW30"].mean())
    checks["nw30_last10_before_rt"] = float(pre["NW30"].dropna().tail(10).mean())
    checks["nw30_trend_slope_per_window"] = float(np.polyfit(np.arange(pre["NW30"].notna().sum()),
                                                             pre["NW30"].dropna().to_numpy(), 1)[0])
    checks["fight_conf_share_min_kr"] = float(fk["fight_c_conf_share"].min())
    checks["fight_conf_share_by_rt"] = {"before": float(pre["fight_c_conf_share"].mean()),
                                        "rt": float(post["fight_c_conf_share"].mean())}
    out["checks"] = checks
    out["weekend"] = {"n": int(len(wkend)), "valid_WK": int(wkend["WK_val"].notna().sum()),
                      "WK_rz8_valid": int(wkend["WK_rz8"].notna().sum()),
                      "WK_rz8m4_valid": int(wkend["WK_rz8m4"].notna().sum()),
                      "base_ok": int(wkend["base_ok"].sum()), "outage": int(wkend["outage"].fillna(False).sum())}
    out["h40"] = {"rows": int(len(h40)), "weeks_kr": int(h40.loc[h40["market"] == "kr", "week_close_day"].nunique()),
                  "weeks_us": int(h40.loc[h40["market"] == "us", "week_close_day"].nunique()),
                  "btilde_missing": float(h40["Btilde"].isna().mean()),
                  "avg8_prior_valid": int(h40["mention_posts_cap3_avg8_prior"].notna().sum())}
    return out


def build(results: Path = RESULTS) -> dict:
    inp = load_inputs(results)
    u, cal, t_end = inp["u"], inp["cal"], inp["t_end"]
    c0 = pd.Timestamp(u.loc[u["utype"] == "post", "dt_kst"].min())   # 자료(번호 연속) 시작 = 이력 시작
    lex = run_lexicon(u)
    fk = activity_features(window_frame(inp["wk"]), u, "win_kr", "pwin_kr", "pre_close_kr")
    fu = activity_features(window_frame(inp["wu"]), u, "win_u", "pwin_u", "pre_close_u")
    fk = newcomer_features(fk, u, lex, "win_kr", c0)
    fk = regular_conversion(fk, u, lex, c0)
    nyse = set(cal.loc[cal["nyse"], "date"].dt.date)
    fu = night_share(fu, u, nyse)
    wkend = weekend_table(u, inp["wk"], cal, c0, t_end)
    att = wkend.dropna(subset=["kr_win_id"]).set_index("kr_win_id")
    for col in ("WK_val", "WK_rz8", "WK_rz8m4", "WK_saved_rz8", "WK_saved_rz8m4"):
        fk[col] = fk["win_id"].map(att[col])
    fk["WK_flag_long_weekend"] = fk["win_id"].map(att["long_weekend"])
    h40 = h40_weekly(u, lex, cal, c0, t_end)
    return {"fk": fk, "fu": fu, "wkend": wkend, "h40": h40, "lex": lex, "c0": c0, "t_end": t_end}


KEEP_ID = ["win_id", "market", "D", "P", "weekday", "open_kst", "close_kst", "length_h", "kind", "over96",
           "xholiday_mismatch", "dst_week", "early_close_D", "realtime", "judge_ok", "base_ok", "outage_b",
           "outage_unknown", "spam_b", "missing_rate", "ratio_no", "ratio_saved", "d_no", "d_no_per_h",
           "n_posts", "n_comments", "n_authors", "c_conf_share", "trunc_share"]


def write(b: dict, results: Path = RESULTS) -> dict:
    fk, fu = b["fk"], b["fu"]
    allcols = list(dict.fromkeys(KEEP_ID + [c for c in fk.columns if c not in KEEP_ID]
                                 + [c for c in fu.columns if c not in KEEP_ID]))
    drop = {"variant", "main_kind", "closed_weekdays", "early_close_P", "b_open", "b_close", "n_saved_in_range",
            "complete", "saved_per_h", "base_no_per_h", "base_saved_per_h", "outage", "spam_wave", "u_lo", "u_hi"}
    allcols = [c for c in allcols if c not in drop]
    out = pd.concat([fk.reindex(columns=allcols), fu.reindex(columns=allcols)], ignore_index=True)
    out = out.rename(columns={"outage_b": "outage", "spam_b": "spam_wave"})
    out["trunc_share"] = out["trunc_share"].astype(float)
    out.to_parquet(results / "feat_activity.parquet", index=False)
    b["wkend"].to_parquet(results / "feat_activity_weekend.parquet", index=False)
    b["h40"].to_parquet(results / "feat_activity_h40_weekly.parquet", index=False)
    summ = summarize(fk, fu, b["wkend"], b["h40"])
    summ["c0"] = str(b["c0"])
    summ["t_end"] = str(b["t_end"])
    lex = b["lex"]
    summ["lexicon"] = {"posts_scored_nonzero": int((lex["score"].fillna(0) != 0).sum()),
                       "posts_news_excluded": int((lex["lex_excluded"] == "news").sum()),
                       "units_politics_only": int(lex["politics_only"].sum()),
                       "units_with_stock_mention": int(lex["stocks"].map(bool).sum())}
    (results / "feat_activity_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), "utf-8")
    return summ


def main() -> int:
    b = build()
    summ = write(b)
    print(json.dumps({k: summ[k] for k in ("n_windows", "n_judge_ok", "checks", "weekend", "h40", "lexicon")},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
