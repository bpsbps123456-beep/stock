"""가격 전 점검표(HYPOTHESES.md §5.1, 동결판) — 중간 검증판(10/15 스냅샷, 7/13~10/15).

항목 1~8·10은 포럼 자료만 쓴다. 9는 prechecks_price.py(지난 종가만), 11은 prechecks_external.py(받을 수 있는지만)가 만든
JSON을 읽어 표에 싣는다. 앞날 주가(R_h·MDD_h·HL·갭 등 진입 이후 값)는 어디서도 계산하지 않는다.

읽기만 한다:
  interim/results/{units,windows_kr,windows_us,feat_activity,feat_text,feat_topic,feat_topic_unit_hits,
                   feat_activity_weekend,feat_activity_h40_weekly,feat_topic_stock}.parquet, loader_*.{json,csv},
                   lexicon_v02*.json, prechecks_price.json, prechecks_external.json
  data/derived/desk/ant/posts/*.jsonl(loader_manifest.json과 같은 바이트만), state.json(소금 지문만, 값은 출력하지 않음)
산출(interim/results/): prechecks.md, prechecks_features.csv, prechecks_summary.json, prechecks_by_postdate.csv
실행: py prechecks.py   (먼저 py prechecks_price.py, py prechecks_external.py)
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CODE = Path(__file__).resolve().parent
INTERIM = CODE.parent
RESULTS = INTERIM / "results"
ANT = INTERIM.parent
ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(CODE))
import loader as LD  # noqa: E402

FROZEN = {  # docs/desk_system.md §12.5
    "HYPOTHESES.md": "b23d1eacc95d0ae4ce1361c2449d9efd0f8feecc9792c4515ee2aff80ac308e5",
    "HYPOTHESES_addendum1.md": "430dd4ffe856420fa9779dff91dc6e3cb3d3b7548551e02ed4dfab2596044eb4",
    "lexicon_draft.md": "a4e28b08ddc3e6cef8b4f49c746c98845cc0f97bb8f03f534ba181a42795f8e1",
}
LEX_V02 = {  # 사전 v0.2 고정 단계가 남긴 지문
    "lexicon_v02.json": "915a2d24b82eebe6ae3ca12c697be4710675ebdfe82477fbb4fa2e0605cf32d3",
    "lexicon_v02_full.json": "b067c3d023b0bd123f04dd932926f59acf34f763084ee983029996bac0ea1762",
}
MISS_MAX = 0.30          # §2.7 자동 설명용 결측률
MIN_BLOCKS = 8           # §2.7 최소 표본(구간마다 독립 묶음)
FRONT = (date(2026, 1, 2), date(2026, 5, 29))
BACK = (date(2026, 6, 1), date(2026, 10, 23))
MDE_K = 2.49             # §2.7 MDE = 2.49/√n_eff (단측 5%, 검정력 80%)
REALTIME_FROM = date(2026, 10, 13)


# ---------------------------------------------------------------- 공통 셈

def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ac1(x: np.ndarray) -> tuple[float, int]:
    """이웃 창 1차 자기상관(두 창 모두 값이 있는 쌍만)."""
    x = np.asarray(x, float)
    a, b = x[:-1], x[1:]
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() < 5 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan"), int(m.sum())
    return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())


def n_eff(n: int, h_days: int, rho: float, step: int = 1) -> float:
    """§2.7 MDE 표와 같은 식: (n − h_rows) ÷ (1 + 2 Σ ρ^k·max(0, 1 − k·step/h)).
    특징은 AR(1)(ρ = 1차 자기상관, 0~0.99로 자름), 목표는 h일 겹침(창 간격 step일). n = 96, h = 1·5·10·21, ρ = 0.5면
    문서 값 95·41·33·27, ρ = 0.95·h = 10이면 약 10이 나온다(test_prechecks)."""
    h_rows = math.ceil(h_days / step)
    if n is None or n <= h_rows:
        return 0.0
    r = 0.0 if (rho is None or rho != rho) else min(max(rho, 0.0), 0.99)
    s = sum((r ** k) * max(0.0, 1 - k * step / h_days) for k in range(1, h_rows + 1))
    return (n - h_rows) / (1 + 2 * s)


def greedy_blocks(pos, gap: int) -> int:
    """정렬된 위치에서 간격 ≥ gap으로 고른 묶음 수(앞에서부터)."""
    c, last = 0, -10**9
    for p in pos:
        if p - last >= gap:
            c, last = c + 1, p
    return c


def corr_pair(x: pd.Series, y: pd.Series) -> dict:
    m = x.notna() & y.notna()
    n = int(m.sum())
    if n < 5:
        return {"n": n, "pearson": None, "spearman": None}
    xx, yy = x[m].astype(float), y[m].astype(float)
    if xx.std() == 0 or yy.std() == 0:
        return {"n": n, "pearson": None, "spearman": None}
    return {"n": n, "pearson": round(float(np.corrcoef(xx, yy)[0, 1]), 3),
            "spearman": round(float(stats.spearmanr(xx, yy).statistic), 3)}


def pct(x, d=1) -> str:
    return "—" if x is None or x != x else f"{100 * x:.{d}f}%"


def fmt(x, d=2) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{d}f}"


# ---------------------------------------------------------------- 입력

def load_tables() -> dict:
    fa = pd.read_parquet(RESULTS / "feat_activity.parquet")
    ft = pd.read_parquet(RESULTS / "feat_text.parquet")
    fto = pd.read_parquet(RESULTS / "feat_topic.parquet")
    daily = {}
    for mk, tb in (("kr", "KR_D"), ("us", "US_D")):
        a = fa[fa.market == mk].set_index("win_id")
        b = ft[ft.market == mk].set_index("win_id")
        c = fto[fto.table == tb].set_index("win_id")
        df = a.join(b[[x for x in b.columns if x not in a.columns]])
        df = df.join(c[[x for x in c.columns if x not in df.columns]])
        df["D"] = pd.to_datetime(df["D"]).dt.date
        daily[mk] = df.sort_values("D")
    weekly = {}
    for mk, tb in (("kr", "KR_W"), ("us", "US_W")):
        w = fto[fto.table == tb].copy()
        w["D"] = pd.to_datetime(w["D"]).dt.date
        weekly[mk] = w.sort_values("D").set_index("win_id")
    wknd = fto[fto.table == "WKND"].copy()
    wknd["D"] = pd.to_datetime(wknd["D"]).dt.date
    wk = pd.read_parquet(RESULTS / "feat_activity_weekend.parquet")
    wk["D"] = pd.to_datetime(wk["mon"]).dt.date
    h40 = pd.read_parquet(RESULTS / "feat_activity_h40_weekly.parquet")
    stock = pd.read_parquet(RESULTS / "feat_topic_stock.parquet")
    return {"daily": daily, "weekly": weekly, "wknd": wknd.sort_values("D").set_index("win_id"),
            "wk": wk.sort_values("D"), "h40": h40, "stock": stock}


def trading_days(cal: pd.DataFrame, lo: date, hi: date, col: str) -> int:
    """loader 달력(10/15까지) + 10/16~10/23은 평일(이 구간 두 거래소 휴장 없음, exchange_calendars로 확인)."""
    c = cal[(cal.date >= lo) & (cal.date <= hi)]
    n = int(c[col].sum())
    last = max(cal.date)
    if hi > last:
        try:
            import exchange_calendars as xc
            name = "XKRX" if col == "krx" else "XNYS"
            sess = xc.get_calendar(name).sessions_in_range(pd.Timestamp(last + timedelta(days=1)), pd.Timestamp(hi))
            n += len(sess)
        except Exception:  # noqa: BLE001
            n += sum(1 for k in range((hi - last).days) if (last + timedelta(days=k + 1)).weekday() < 5)
    return n


# ---------------------------------------------------------------- 6. 특징 표(사양)

def raw_missing(*cols):
    def f(df):
        m = pd.Series(False, index=df.index)
        for c in cols:
            m |= df[c].isna()
        return m
    return f


def after_first_valid(col):
    """예열(값이 처음 생기기 전) 창을 뺀 뒤에도 비는 창 = 구조 결측. 예열 창은 False(구조 아님)로 둔다."""
    def f(df):
        v = df[col].notna().to_numpy()
        if not v.any():
            return pd.Series(np.nan, index=df.index)
        first = int(np.argmax(v))
        m = pd.Series(False, index=df.index)
        m.iloc[first:] = ~v[first:]
        return m
    return f


def votes_lt20(vcol, rawcol):
    def f(df):
        return (df[vcol].fillna(0) < 20) | df[rawcol].isna()
    return f


def h11_struct(df):
    ok = ~((df["n_votes"].fillna(0) < 20) | df["D_SD_raw"].isna())
    o = ok.to_numpy()
    out = np.zeros(len(o), bool)
    for i in range(len(o)):
        out[i] = not (o[i] and i >= 5 and o[i - 5] and o[i - 4] and o[i - 3])
    return pd.Series(out, index=df.index)


def h37_struct(df):
    den = df["h37_den_bear"]
    m = pd.Series(np.nan, index=df.index)
    comp = den.notna()
    m[comp] = (den[comp] < 5).astype(float)
    return m


SPECS = [
    # id, 칸, 표, 열, h(거래일), 정규화/예열, 구조 결측 규칙, 사전 묶음, 메모
    ("H03", "T1 대표", "kr", "A_rz8", 1, "rz8(같은 요일 8창)", raw_missing("A_raw"), "없음", ""),
    ("H03", "T1 대표(저장판)", "kr", "A_saved_rz8", 1, "rz8", raw_missing("A_raw"), "없음", "번호판과 부호가 같아야 인정"),
    ("H06", "T1 대표", "kr", "h06_dsd", 1, "풀 60창 + 표 20", votes_lt20("h06_n_votes", "h06_sd_raw"), "BULL·BEAR", ""),
    ("H10", "T1 대표", "us", "h10_dsd_z60", 1, "풀 60창 + 표 20 + z60", votes_lt20("h10_n_votes", "h10_sd_raw"), "BULL·BEAR·SEMI", ""),
    ("H09", "T2 대표", "kr", "h06_dsd", 5, "풀 60창 + 표 20", votes_lt20("h06_n_votes", "h06_sd_raw"), "BULL·BEAR", "H06과 같은 D_SD"),
    ("H14", "T2 대표", "kr", "S", 5, "M0 확장 회귀 120창(z_e = S⊥)", raw_missing("S"), "E·F", "z_e는 가격 단계. 재료 S로 셈"),
    ("H19", "T2 대표", "kr", "H19_E", 5, "z60(40창)", raw_missing("E_A_P", "E_A_C"), "E", ""),
    ("H24", "T2 대표(사건)", "kr", "H24_R", 10, "직전 20창 평균(+2025 계수)", raw_missing("F_A_d9"), "F", "값은 모든 창, 판정은 사건 창만"),
    ("H25", "F5-1(사건)", "kr", "H25_CR_d9", 10, "없음(사건 창)", raw_missing("H25_CR_d9"), "CAP·HES", "판정은 사건 창만"),
    ("H26", "F5-2(사건)", "kr", "H26_PH", 10, "5창 합 + z60", after_first_valid("RES5"), "RES·ANG", "판정은 사건 창만"),
    ("H31", "T2 대표", "us", "LL_US_d20", 10, "20창 변화(+SOX 잔차)", raw_missing("LL_US_A_all"), "LL_US", "SOX 잔차화는 가격 단계"),
    ("H35", "T2 대표", "kr", "NW90_rz8", 10, "90일 이력 + rz8", after_first_valid("NW30"), "없음(작성자 해시)", "구조 결측은 30일판으로 셈"),
    ("H46", "T2 대표", "us", "h46_dtilt", 10, "유효 창 5+20", raw_missing("h46_tilt"), "TILT_US·TILT_KR", ""),
    ("H49", "T2 대표", "us", "h49_s_e", 1, "확장 회귀(가격)", raw_missing("h49_s_e"), "E·F", "잔차화는 가격 단계"),
    ("H51", "T2 대표", "kr", "h51_s_nm", 1, "확장 회귀(가격)", raw_missing("h51_s_nm"), "E·F", "잔차화는 가격 단계"),
    ("H57a", "F11 대표", "kr", "stage10", 5, "pct250(120창)", raw_missing("S"), "E·F", ""),
    ("H57b", "F11 대표", "kr", "stage10", 1, "pct250(120창)", raw_missing("S"), "E·F", "끝 단계 더미"),
    ("H05", "F1-1", "kr", "H05_ED", 1, "z60", raw_missing("ED_P_raw", "ED_C_raw"), "E·F·FO·ANG", ""),
    ("H04", "F1-2", "kr", "Bperp_rz8", 1, "rz8 + 확장 회귀 20창", raw_missing("U_per_h", "N_per_h"), "없음(작성자 해시)", ""),
    ("H08", "F2-1", "kr", "fight", 1, "없음(원값)", raw_missing("fight"), "없음(댓글 구조)", ""),
    ("H11", "F2b-1", "kr", "H11_X", 5, "D_SD + ΔD3 + z60", h11_struct, "BULL·BEAR", ""),
    ("H15압력", "F3-1", "kr", "Pess", 1, "z60", raw_missing("S"), "E·F", ""),
    ("H12", "F3-2", "kr", "H12_Gap", 5, "관심 기준 20일 + z60", after_first_valid("Gap_raw"), "BULL·BEAR", ""),
    ("H15반전", "F3-3", "kr", "Pess", 5, "z60", raw_missing("S"), "E·F", "D+1 시가→D+5 종가"),
    ("H16", "F3-4", "kr", "Pess_news_event", 5, "z60", lambda df: df["n_news_event"].fillna(0) == 0, "E·F", "사건 뉴스형 0개 창 = 결측"),
    ("H16", "F3-4(의견형)", "kr", "Pess_opinion", 5, "z60", raw_missing("S_opinion"), "E·F", ""),
    ("H54", "F3-5(SH)", "kr", "A6_SH", 5, "rz8", raw_missing("SH"), "없음", ""),
    ("H55", "F3-5(K)", "kr", "A7_K", 5, "관심 기준 20일 + rz8", after_first_valid("K"), "없음", ""),
    ("H21", "F4-1", "kr", "H21_FO", 5, "z60", raw_missing("FO_mkt_A_P", "FO_mkt_A_C"), "FO", ""),
    ("H36", "F7-1", "kr", "NEW90_rz8", 10, "90일 이력 + rz8", after_first_valid("NEW30_per_h"), "없음(작성자 해시)", "상승 조건 거르기는 가격 단계"),
    ("H37", "F7-2", "kr", "h37_bull_conv_rate", 10, "30일 이력 + 5창", h37_struct, "BULL·BEAR", "구조 결측 = 계산된 창 중 분모 < 5"),
    ("H50", "F9-1", "us", "h50_m_z20", 5, "z20(20창)", raw_missing("h50_m_ratio"), "MEM", "잔차화는 가격 단계"),
    ("H33", "실시간", "kr", "H33_sig", 10, "z60 또는 pct250", raw_missing("MC_A_all"), "MC", "실시간 수집분만 판정"),
    ("H34", "실시간", "us", "NS_share", 10, "20일 평균(+가격 잔차)", raw_missing("NS_share"), "없음", "실시간 수집분만 판정"),
    ("H01", "관문", "kr", "S", 0, "없음(상관), M0 R²는 120창", raw_missing("S"), "E·F", "가격과의 상관은 5단계에서"),
]


def feature_rows(T: dict, n_back: dict, n_front: dict) -> list[dict]:
    rows = []
    for hid, cell, tb, col, h, fam, struct, lex, memo in SPECS:
        df = T["daily"][tb]
        x = df[col].astype(float)
        valid = x.notna().to_numpy()
        judge = df["judge_ok"].astype(bool).to_numpy()
        sm = struct(df)
        if sm.isna().all():
            s_rate = None
        else:
            sj = sm[judge].dropna()
            s_rate = float(sj.astype(float).mean()) if len(sj) else None
        r1, npair = ac1(x.to_numpy())
        nv = int(valid.sum())
        hh = max(h, 1)
        pos = np.flatnonzero(valid)
        nb = n_back["krx" if tb == "kr" else "nyse"]
        nf = n_front["krx" if tb == "kr" else "nyse"]
        ne_now = n_eff(nv, hh, r1)
        ne_back = n_eff(nb, hh, r1)
        rows.append({
            "id": hid, "cell": cell, "market": tb, "col": col, "h": h, "family": fam, "lex": lex, "memo": memo,
            "n_rows": len(df), "n_valid": nv, "miss_all": 1 - nv / len(df), "miss_judge": 1 - valid[judge].mean(),
            "miss_struct": s_rate, "first": str(df["D"][valid].min()) if nv else None,
            "ac1": r1, "ac1_pairs": npair,
            "neff_now": ne_now, "blocks_now": greedy_blocks(pos, hh), "blocks_now_strict": greedy_blocks(pos, hh + 1),
            "neff_back": ne_back, "mde_back": MDE_K / math.sqrt(ne_back) if ne_back > 0 else None,
            "blocks_back": greedy_blocks(range(nb - hh), hh), "blocks_back_strict": greedy_blocks(range(nb - hh), hh + 1),
            "blocks_front_strict": greedy_blocks(range(nf - hh), hh + 1),
        })
    return rows


def weekly_rows(T: dict, n_back_weeks: int) -> list[dict]:
    """주 단위 칸: H39+H48(AS), H40(B̃), H52(WK), H53(ΔS)."""
    rows = []
    st = T["stock"]
    for mk, tb in (("kr", "KR_W"), ("us", "US_W")):
        s = st[st.table == tb]
        nw = s.win_id.nunique()
        doc = s["h39_as"].notna()
        lex6 = s["h39_as_lex6"].notna()
        weeks = sorted(s.win_id.unique())
        widx = {w: i for i, w in enumerate(weeks)}
        warm = s["win_id"].map(widx) >= 8
        elig_fail = (~s["h39_elig"].fillna(False).astype(bool))[warm]
        pos_doc = sorted({widx[w] for w in s.loc[doc, "win_id"]})
        pos_lex = sorted({widx[w] for w in s.loc[lex6, "win_id"]})
        rows.append({"id": "H39+H48", "cell": "T2 대표(문서 규칙)", "market": mk, "col": "h39_as", "h": 21,
                     "family": "8주 + 유니버스 15종목", "lex": "ALIAS", "n_rows": int(len(s)), "n_valid": int(doc.sum()),
                     "weeks": nw, "miss_all": 1 - doc.mean(), "miss_struct": 1.0 - float(T["weekly"][mk]["h39_week_ok15"].mean()),
                     "blocks_now": greedy_blocks(pos_doc, 5), "blocks_back": greedy_blocks(range(n_back_weeks - 5), 5),
                     "memo": "15종목 미만 주는 버림"})
        rows.append({"id": "H39+H48", "cell": "강건판(현재 12종목)", "market": mk, "col": "h39_as_lex6", "h": 21,
                     "family": "8주", "lex": "ALIAS", "n_rows": int(len(s)), "n_valid": int(lex6.sum()), "weeks": nw,
                     "miss_all": 1 - lex6.mean(), "miss_struct": float(elig_fail.mean()) if len(elig_fail) else None,
                     "blocks_now": greedy_blocks(pos_lex, 5), "blocks_back": greedy_blocks(range(n_back_weeks - 5), 5),
                     "memo": "구조 결측 = 8주 예열 뒤 언급 자격(평균 5건) 미달 종목·주"})
    h40 = T["h40"]
    for mk in ("kr", "us"):
        s = h40[h40.market == mk]
        rows.append({"id": "H40", "cell": "F8-2", "market": mk, "col": "Btilde", "h": 21, "family": "주 창(예열 없음)",
                     "lex": "ALIAS", "n_rows": int(len(s)), "n_valid": int(s.Btilde.notna().sum()),
                     "weeks": int(s.week_close_day.nunique()), "miss_all": float(s.Btilde.isna().mean()),
                     "miss_struct": float(s.Btilde.isna().mean()), "blocks_back": greedy_blocks(range(n_back_weeks - 5), 5),
                     "memo": "AS×B̃ 항이라 AS 결측도 물려받음"})
    wk = T["wk"]
    v = wk["WK_rz8"].notna().to_numpy()
    rows.append({"id": "H52", "cell": "F1-3", "market": "wknd", "col": "WK_rz8", "h": 1, "family": "rz8(직전 8주말)",
                 "lex": "없음", "n_rows": len(wk), "n_valid": int(v.sum()), "weeks": len(wk), "miss_all": 1 - v.mean(),
                 "miss_struct": float(wk["WK_val"].isna().mean()), "ac1": ac1(wk["WK_rz8m4"].to_numpy())[0],
                 "blocks_back": greedy_blocks(range(n_back_weeks), 1),
                 "memo": "구조 결측 = 월요일 휴장·추석 주말(값 없음)"})
    wd = T["wknd"]
    v = wd["h53_ds"].notna().to_numpy()
    rows.append({"id": "H53", "cell": "F10-1", "market": "wknd", "col": "h53_ds", "h": 5, "family": "없음(잔차화는 가격)",
                 "lex": "E·F", "n_rows": len(wd), "n_valid": int(v.sum()), "weeks": len(wd), "miss_all": 1 - v.mean(),
                 "miss_struct": float(wd["h53_ds"].isna().mean()), "ac1": ac1(wd["h53_ds"].to_numpy())[0],
                 "blocks_now": greedy_blocks(np.flatnonzero(v), 2), "blocks_back": greedy_blocks(range(n_back_weeks - 1), 2),
                 "memo": "주 목표(월 시가→그 주 끝 종가), 엄한 간격 2주"})
    return rows


# ---------------------------------------------------------------- 2·3·4·5 원자료 점검

def raw_checks(units: pd.DataFrame, hits: pd.DataFrame) -> dict:
    man = json.loads((RESULTS / "loader_manifest.json").read_text("utf-8"))
    rows, _ = LD.read_snapshot(manifest=man)
    by_no: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_no[r["rec"]["no"]].append(r)
    out: dict = {"snapshot_lines": len(rows), "posts": len(by_no)}

    # 2. 소금: 같은 글 두 줄의 작성자 해시 일치, 같은 댓글(본문·시각) 두 줄의 해시 일치
    multi = {no: sorted(rs, key=lambda r: (r["file"], r["line"])) for no, rs in by_no.items() if len(rs) > 1}
    same_a = sum(1 for rs in multi.values() if len({r["rec"]["a"] for r in rs}) == 1)
    c_pairs = c_same = 0
    for rs in multi.values():
        maps = []
        for r in rs[:2]:
            d = defaultdict(set)
            for c in r["rec"]["comments"]:
                d[(c["text"], c["dt"][-8:])].add(c["a"])
            maps.append(d)
        for k, a1 in maps[0].items():
            a2 = maps[1].get(k)
            if a2 and len(a1) == 1 and len(a2) == 1 and k[0]:
                c_pairs += 1
                c_same += int(a1 == a2)
    out["salt_multirow"] = {"multi_row_posts": len(multi), "post_author_same": same_a,
                            "comment_pairs": c_pairs, "comment_author_same": c_same}

    # 2. 번호 묶음(500번) 사이 작성자 겹침: 수집 실행 경계(10/16 02:58 이후 멈춤 → 09:41 재개, 405,523~406,523)
    posts = units[units.utype == "post"][["no", "a"]]
    coms = units[units.utype == "comment"][["no", "a"]]
    allu = pd.concat([posts, coms])
    allu = allu[allu.a.astype(str) != ""]
    allu["blk"] = allu["no"] // 500
    sets = allu.groupby("blk")["a"].apply(set)
    blks = sorted(sets.index)
    ov = []
    for b0, b1 in zip(blks[:-1], blks[1:]):
        if b1 != b0 + 1:
            continue
        s0, s1 = sets[b0], sets[b1]
        ov.append({"blk": int(b1) * 500, "overlap": len(s0 & s1) / max(1, min(len(s0), len(s1)))})
    ovd = pd.DataFrame(ov)
    near = ovd[(ovd.blk >= 405000) & (ovd.blk <= 407500)]
    out["salt_blocks"] = {"pairs": len(ovd), "median": float(ovd.overlap.median()), "min": float(ovd.overlap.min()),
                          "min_blk": int(ovd.loc[ovd.overlap.idxmin(), "blk"]),
                          "lt_half_median": int((ovd.overlap < 0.5 * ovd.overlap.median()).sum()),
                          "near_run_boundary": [{"blk": int(r.blk), "overlap": round(float(r.overlap), 3)}
                                                for r in near.itertuples()]}

    # 3. 첫 쪽 = 가장 이른 댓글인가: 두 줄 글에서 첫 줄 댓글이 둘째 줄의 앞자리에 그대로 있는가
    t_rows = t_kept = t_prefix = 0
    a_rows = a_kept_num = a_kept_den = 0
    new_root_after = new_root_total = 0
    for rs in multi.values():
        r1, r2 = rs[0]["rec"], rs[-1]["rec"]
        k1 = [(c["a"], c["text"], c["dt"][-8:]) for c in r1["comments"]]
        k2 = [(c["a"], c["text"], c["dt"][-8:]) for c in r2["comments"]]
        if not k1:
            continue
        s2 = set(k2)
        kept = sum(1 for k in k1 if k in s2)
        a_rows += 1
        a_kept_num += kept
        a_kept_den += len(k1)
        roots1 = [(c["a"], c["text"], c["dt"][-8:]) for c in r1["comments"] if c["depth"] == 0]
        roots2 = [(c["a"], c["text"], c["dt"][-8:]) for c in r2["comments"] if c["depth"] == 0]
        trunc2 = int(r2["ncomment"]) > len(r2["comments"])
        if trunc2 and len(roots2) >= 10 and roots1:
            t_rows += 1
            sr2 = set(roots2)
            t_kept += int(all(k in sr2 for k in roots1))
            pos_ = [roots2.index(k) for k in roots1 if k in sr2]
            t_prefix += int(len(pos_) == len(roots1) and max(pos_) < len(roots1))
        sr1 = set(roots1)
        newr = [i for i, k in enumerate(roots2) if k not in sr1]
        new_root_total += len(newr)
        idx1 = [roots2.index(k) for k in roots1 if k in set(roots2)]
        if idx1:
            new_root_after += sum(1 for i in newr if i > max(idx1))
    out["first_page"] = {"rows_with_c1": a_rows, "c1_kept_share": a_kept_num / max(1, a_kept_den),
                         "trunc_rows_10roots": t_rows, "trunc_c1_roots_all_kept": t_kept, "trunc_c1_roots_prefix": t_prefix,
                         "new_roots_total": new_root_total, "new_roots_after_old": new_root_after}

    # 3. 쪽 크기와 첫 댓글 지연(잘린 글 vs 안 잘린 글): 잘린 글의 첫 쪽이 최신 쪽이면 첫 댓글이 늦게 나온다
    p = units[units.utype == "post"].set_index("no")
    c = units[(units.utype == "comment") & units.dt_conf.isin(["full", "matched", "row"]) & ~units.before_post.fillna(False)]
    c = c.assign(delay=(c.dt_kst - c.post_dt_kst).dt.total_seconds() / 60)
    g = c.groupby("no")["delay"]
    first_delay, last_delay = g.min(), g.max()
    roots = units[(units.utype == "comment") & (units.depth == 0)].groupby("no").size()
    p = p.assign(roots=roots.reindex(p.index).fillna(0), fd=first_delay.reindex(p.index), ld=last_delay.reindex(p.index))
    tr = p[p.trunc_last.fillna(False).astype(bool)]
    nt10 = p[~p.trunc_last.fillna(False).astype(bool) & (p.roots >= 8)]
    diff = (tr.ncomment_last - tr.nsaved_last).astype(float)
    out["page"] = {"trunc_posts": int(len(tr)), "roots_max": int(p.roots.max()),
                   "posts_10roots": int((p.roots == 10).sum()), "trunc_roots_ge10": int((tr.roots >= 10).sum()),
                   "trunc_diff_le2": int((diff <= 2).sum()), "trunc_diff_median": float(diff.median()),
                   "first_delay_min_trunc_median": float(tr.fd.median()), "first_delay_min_nontrunc8_median": float(nt10.fd.median()),
                   "last_delay_min_trunc_median": float(tr.ld.median()), "last_delay_min_nontrunc8_median": float(nt10.ld.median())}

    # 4. 본문 수정(두 줄 글) + 수정 표지 몫
    pp = units[units.utype == "post"].copy()
    pp["pdate"] = pd.to_datetime(pp.dt_kst).dt.date
    pp["rt"] = pp.pdate >= REALTIME_FROM
    two = pp[pp.n_rows.fillna(1) >= 2]
    mark = re.compile(r"추가|수정|업데이트|(?<![A-Za-z])p\.?s(?![A-Za-z])", re.I)
    pp["edit_mark"] = (pp.title.fillna("") + "\n" + pp.text.fillna("")).str.contains(mark)
    ed = {}
    for lab, sub in (("실시간(10/13~)", two[two.rt]), ("그 전", two[~two.rt]), ("전체", two)):
        ed[lab] = {"two_row_posts": int(len(sub)), "body_edited": int(sub.body_edited.fillna(False).sum()),
                   "title_edited": int(sub.title_edited.fillna(False).sum())}
    ed["edit_marker_share"] = {"실시간(10/13~)": float(pp[pp.rt].edit_mark.mean()), "그 전": float(pp[~pp.rt].edit_mark.mean())}
    out["edits"] = ed

    # 5. 날짜별 구조(글 등록일 기준): 글당 저장 댓글, 지운 댓글 몫, 잘림, 결번
    cm = units[units.utype == "comment"][["no"]].join(hits.loc[units.utype == "comment", ["excl"]])
    cm["deleted"] = cm.excl == "deleted"
    dele = cm.groupby("no")["deleted"].agg(["sum", "count"])
    pp = pp.set_index("no")
    pp = pp.join(dele)
    by = pp.groupby("pdate").agg(posts=("a", "size"), stored=("count", "sum"), deleted=("sum", "sum"),
                                 trunc=("trunc_last", "sum"), two_rows=("n_rows", lambda s: int((s >= 2).sum())))
    by["stored_per_post"] = by.stored / by.posts
    by["deleted_share"] = by.deleted / by.stored
    by["trunc_share"] = by.trunc / by.posts
    mb = pd.read_csv(RESULTS / "loader_missing_by_day.csv")
    mb["date"] = pd.to_datetime(mb["date"]).dt.date
    by = by.join(mb.set_index("date")[["missing_rate"]])
    cd = pd.read_csv(RESULTS / "loader_comment_dates.csv")
    cd = cd[cd["by"] == "post_date"].copy()
    cd["date"] = pd.to_datetime(cd["date"]).dt.date
    by = by.join(cd.set_index("date")[["time_only", "conf_rate_time_only"]])
    by.to_csv(RESULTS / "prechecks_by_postdate.csv", encoding="utf-8-sig")

    def grp(lo, hi):
        s = by[(by.index >= lo) & (by.index <= hi)]
        return {"days": int(len(s)), "posts": int(s.posts.sum()), "stored_per_post": float(s.stored.sum() / s.posts.sum()),
                "deleted_share": float(s.deleted.sum() / s.stored.sum()), "trunc_share": float(s.trunc.sum() / s.posts.sum()),
                "missing_rate_median": float(s.missing_rate.median()) if s.missing_rate.notna().any() else None,
                "two_row_share": float(s.two_rows.sum() / s.posts.sum())}
    wkd = by.index.map(lambda d: d.weekday() < 5)

    def grp_wd(lo, hi):
        s = by[(by.index >= lo) & (by.index <= hi) & wkd]
        return {"days": int(len(s)), "stored_per_post": float(s.stored.sum() / s.posts.sum()),
                "trunc_share": float(s.trunc.sum() / s.posts.sum()), "deleted_share": float(s.deleted.sum() / s.stored.sum())}
    out["by_period_weekday"] = {"9/14~10/11 평일": grp_wd(date(2026, 9, 14), date(2026, 10, 11)),
                                "10/13~10/15 평일(실시간)": grp_wd(date(2026, 10, 13), date(2026, 10, 15))}
    out["by_period"] = {"7/13~8/31(약 1.5~3달 뒤 수집)": grp(date(2026, 7, 13), date(2026, 8, 31)),
                        "9/1~10/11(4일~6주 뒤 수집)": grp(date(2026, 9, 1), date(2026, 10, 11)),
                        "10/12(경계)": grp(date(2026, 10, 12), date(2026, 10, 12)),
                        "10/13~10/15(실시간 + 24시간 다시 받기)": grp(date(2026, 10, 13), date(2026, 10, 15))}
    trs = by.trunc_share
    out["trunc_by_day"] = {"median": float(trs.median()), "min": float(trs.min()), "max": float(trs.max()),
                           "max_day": str(trs.idxmax())}
    return out


# ---------------------------------------------------------------- 7. 분모 분포

M9_PCT = re.compile(r"([+\-]|플러스|마이너스)?\s?(\d{1,4}(?:\.\d+)?)\s?(%|퍼|프로)")
M9_MULT = re.compile(r"(\d+(?:\.\d+)?)\s?배")
M9_CTX = ["수익", "손실", "익절", "손절", "계좌", "평단", "원금", "물려", "마이너스", "플러스", "올랐", "빠졌", "녹"]
M9_EXCL = ["금리", "지분", "확률", "세율", "관세", "성장률", "점유율", "할인", "이자", "배당률", "경쟁률", "이익률", "yoy", "전년"]
M9_NEG = ["마이너스", "손실", "손절", "빠졌", "녹", "물려"]
M9_POS = ["플러스", "수익", "익절", "올랐"]
M10 = re.compile(r"저는|제가|나는|내 계좌|제 계좌|제 평단|저도|(?:^|[\s,.!?])전(?=[\s,.!?]|$)")


def m9_claims(text: str) -> list[tuple[int, float, str]]:
    """lexicon §5 M9: (부호 +1/−1, 값, 종류 '%'/'배'). 맥락 없음·제외어·부호 불명은 버린다."""
    out = []
    t = text or ""
    tl = t.lower()
    for rx, kind in ((M9_PCT, "%"), (M9_MULT, "배")):
        for m in rx.finditer(t):
            lo, hi = max(0, m.start() - 10), min(len(t), m.end() + 10)
            ctx, ctxl = t[lo:hi], tl[lo:hi]
            if not any(w in ctx for w in M9_CTX) or any(w in ctxl for w in M9_EXCL):
                continue
            sign = 0
            g1 = m.group(1) if kind == "%" else None
            if g1 in ("+", "플러스"):
                sign = 1
            elif g1 in ("-", "마이너스"):
                sign = -1
            else:
                c2 = ctx.replace("수익률", "")           # '수익률'은 부호가 아니다(notes)
                neg = any(w in c2 for w in M9_NEG)
                pos = any(w in c2 for w in M9_POS)
                sign = 1 if (pos and not neg) else (-1 if (neg and not pos) else 0)
            if sign == 0:
                continue
            val = float(m.group(2) if kind == "%" else m.group(1))
            out.append((sign, val, kind))
    return out


def denominators(T: dict, units: pd.DataFrame, hits: pd.DataFrame) -> dict:
    kr = T["daily"]["kr"]
    out: dict = {}
    # H09 보조 코스닥 태그(하한: '코스닥' 낱말만, 이름 30개 없음). 표 = 점수 ≠ 0 작성자(봇 미제외 = 상한 쪽)
    ok = (hits.excl == "") & (units.utype == "post")
    p = units[ok][["no", "a", "title", "text", "win_kr"]].join(hits[ok][["score"]])
    tag = p[(p.title.fillna("") + "\n" + p.text.fillna("")).str.contains("코스닥")]
    a_mean = tag.groupby(["win_kr", "a"])["score"].mean()
    votes = (a_mean != 0).groupby(level=0).sum().reindex(kr.index).fillna(0)
    j = kr.judge_ok.astype(bool)
    out["H09_kosdaq_tag_lower"] = {"tag_posts_median": float(tag.groupby("win_kr").size().reindex(kr.index).fillna(0).median()),
                                   "votes_median": float(votes[j].median()), "votes_max": float(votes[j].max()),
                                   "share_votes_ge20": float((votes[j] >= 20).mean())}
    # H23 수익률 주장(M9, 그리고 M9+M10 1인칭)
    u = units[(hits.excl == "")][["utype", "title", "text", "win_kr"]]
    txt = np.where(u.utype == "post", u.title.fillna("") + "\n" + u.text.fillna(""), u.text.fillna(""))
    rec = []
    for wid, t in zip(u.win_kr.to_numpy(), txt):
        if wid is None or wid != wid:
            continue
        cl = m9_claims(t)
        if not cl:
            continue
        first = bool(M10.search(t))
        bw = any(s > 0 and ((k == "%" and v >= 100) or (k == "배" and v >= 2)) for s, v, k in cl)
        bl = any(s < 0 and k == "%" and v >= 30 for s, v, k in cl)
        rec.append((wid, first, bw, bl))
    cdf = pd.DataFrame(rec, columns=["win", "m10", "bw", "bl"])
    res = {}
    for lab, sub in (("M9", cdf), ("M9+M10", cdf[cdf.m10])):
        n = sub.groupby("win").size().reindex(kr.index).fillna(0)
        s5 = n.rolling(5, min_periods=5).sum()
        ev = s5.notna() & j
        res[lab] = {"claims_per_window_median": float(n[j].median()), "claims_total": int(n.sum()),
                    "bw_total": int(sub.bw.sum()), "bl_total": int(sub.bl.sum()),
                    "miss_5win_lt5": float((s5[ev] < 5).mean())}
    out["H23_claims"] = res
    # H33 MC 작성자
    mc = kr["MC_n_all"]
    out["H33_mc_authors"] = {"median": float(mc[j].median()), "share_ge3": float((mc[j] >= 3).mean()),
                             "sig_windows": int(kr["H33_sig"].fillna(False).astype(bool).sum()),
                             "sig_computable": int(kr["H33_sig"].notna().sum())}
    us = T["daily"]["us"]
    ju = us.judge_ok.astype(bool)
    out["H33_mc_authors_us"] = {"median": float(us["MC_n_all"][ju].median()), "share_ge3": float((us["MC_n_all"][ju] >= 3).mean())}
    # H37 단골 분모
    den = kr["h37_den_bear"]
    comp = den.notna() & j
    mon = pd.Series([d.month for d in kr["D"]], index=kr.index)
    out["H37_den"] = {"computed": int(comp.sum()), "ge5_of_computed": float((den[comp] >= 5).mean()),
                      "ge5_of_all": float((den[j] >= 5).sum() / j.sum()), "median": float(den[comp].median()),
                      "by_month_median": {int(m): float(den[comp & (mon == m)].median()) for m in sorted(mon[comp].unique())}}
    # H16 사건 뉴스
    ne = kr["n_news_event"]
    out["H16_news"] = {"event_zero_share": float((ne[j] == 0).mean()), "event_lt5_share": float((ne[j] < 5).mean()),
                       "event_median": float(ne[j].median()), "price_zero_share": float((kr["n_news_price"][j] == 0).mean()),
                       "us_event_zero_share": float((us["n_news_event"][ju] == 0).mean())}
    return out


# ---------------------------------------------------------------- 8. 대표 특징끼리의 상관

def correlations(T: dict) -> dict:
    kr, us = T["daily"]["kr"], T["daily"]["us"]
    jk, ju = kr.judge_ok.astype(bool), us.judge_ok.astype(bool)
    k, u = kr[jk], us[ju]
    out = {
        "H04_corr_U_N_rz8": corr_pair(k.U_rz8, k.N_rz8),
        "H04_corr_U_N_rz8m4": corr_pair(k.U_rz8m4, k.N_rz8m4),
        "H04_corr_U_N_raw_per_h": corr_pair(k.U_per_h, k.N_per_h),
        "H04_corr_U_N_rz8_us": corr_pair(u.U_rz8, u.N_rz8),
        "H54_corr_SH_raw_A_rz8(문자 그대로)": corr_pair(k.SH, k.A_rz8),
        "H55_corr_K_raw_A_rz8(문자 그대로)": corr_pair(k.K, k.A_rz8),
        "H54_corr_SH_A_rz8": corr_pair(k.A6_SH, k.A_rz8),
        "H54_corr_SH_A_raw": corr_pair(k.SH, k.A_raw),
        "H54_corr_SH_raw_A_rz8m4": corr_pair(k.SH, k.A_rz8m4),
        "H55_corr_K_A_rz8": corr_pair(k.A7_K, k.A_rz8),
        "H55_corr_K_A_raw": corr_pair(k.K, k.A_raw),
        "H55_corr_K_raw_A_rz8m4": corr_pair(k.K, k.A_rz8m4),
        "H36_corr_A4_A_rz8(NEW90)": corr_pair(k.NEW90_rz8, k.A_rz8),
        "H36_corr_NEW30_A_rz8m4": corr_pair(k.NEW30_rz8m4, k.A_rz8m4),
        "H36_corr_NEW30_A_raw": corr_pair(k.NEW30_per_h, k.A_raw),
        "H11_corr_DSDraw_B(표20+)": corr_pair(k.D_SD_raw.where(k.n_votes >= 20), k.B),
        "H11_corr_DSDraw_absB(표20+)": corr_pair(k.D_SD_raw.where(k.n_votes >= 20), k.B.abs()),
        "H11_corr_DSDprov40_B": corr_pair(k.h06_dsd_prov40, k.h06_B),
        "H03_corr_A_Asaved_rz8": corr_pair(k.A_rz8, k.A_saved_rz8),
        "H08_corr_fight_lncomments": corr_pair(k.fight, k.ln_n_comments_onposts),
    }
    # H34: NS 20창 평균 vs LL 신호(같은 W_U). 잔차화(가격)는 빼고 원값으로. 상위 15%(문턱 85) 겹침은 전 표본 순위(설명용 점검값)
    ns20 = u.NS_share.rolling(20, min_periods=20).mean()
    ll20 = u.LL_US_A_all.rolling(20, min_periods=20).mean()
    d = pd.DataFrame({"ns20": ns20, "ll_d20": u.LL_US_d20, "ll20": ll20}).dropna()
    ov = {"n": int(len(d)), "corr_ns20_lld20": corr_pair(d.ns20, d.ll_d20), "corr_ns20_ll20": corr_pair(d.ns20, d.ll20)}
    if len(d) >= 10:
        for nm in ("ll_d20", "ll20"):
            on_ns = d.ns20 >= d.ns20.quantile(0.85)
            on_ll = d[nm] >= d[nm].quantile(0.85)
            both = int((on_ns & on_ll).sum())
            ov[f"top15_overlap_{nm}"] = {"ns_on": int(on_ns.sum()), "ll_on": int(on_ll.sum()), "both": both,
                                          "share_of_ns_on": both / max(1, int(on_ns.sum()))}
    out["H34_NS20_vs_LL"] = ov
    # H35 가격 전 점검(§3): 이력 누적 추세, 10/13 앞뒤 수준 점프(NW30 비등록판, NW90은 3창뿐)
    nw = kr[["D", "NW30", "realtime"]].dropna(subset=["NW30"])
    pre = nw[~nw.realtime.astype(bool)]
    rt = nw[nw.realtime.astype(bool)]
    if len(pre) >= 10:
        xs = np.arange(len(pre))
        slope = float(np.polyfit(xs, pre.NW30.to_numpy(float), 1)[0])
        last10 = pre.NW30.iloc[-10:]
        out["H35_NW30_boundary"] = {"pre_n": int(len(pre)), "pre_mean": float(pre.NW30.mean()), "pre_last10_mean": float(last10.mean()),
                                    "pre_last10_sd": float(last10.std(ddof=1)), "rt_n": int(len(rt)), "rt_mean": float(rt.NW30.mean()),
                                    "jump_z_vs_last10": float((rt.NW30.mean() - last10.mean()) / last10.std(ddof=1)) if len(rt) else None,
                                    "pre_slope_per_window": slope,
                                    "NW90_values": {str(d): round(float(v), 4) for d, v in kr[["D", "NW90"]].dropna().itertuples(index=False)}}
    return out


# ---------------------------------------------------------------- 최소 창 수(예열) 표

def warmup_table(T: dict) -> list[dict]:
    kr, us = T["daily"]["kr"], T["daily"]["us"]
    fam = [
        ("rz8(같은 요일 8창)", [("kr", "A_rz8"), ("kr", "A_saved_rz8"), ("kr", "U_rz8"), ("kr", "N_rz8"), ("kr", "A6_SH"),
                             ("kr", "A7_K"), ("kr", "NW30_rz8"), ("kr", "NEW30_rz8"), ("us", "A_rz8"), ("us", "U_rz8"),
                             ("kr", "NW90_rz8"), ("kr", "NEW90_rz8"), ("kr", "Bperp_rz8"), ("us", "Bperp_rz8")]),
        ("z60(40창)", [("kr", "S_z60"), ("kr", "Pess"), ("kr", "Pess_tok"), ("kr", "H19_E"), ("kr", "H21_FO"), ("kr", "H05_ED"),
                      ("kr", "B_z60"), ("kr", "MC_A_all_z60"), ("kr", "fight_z60"), ("kr", "F_A_all_z60"), ("kr", "H26_PH"),
                      ("kr", "Pess_opinion"), ("kr", "Pess_news_price"), ("kr", "H12_Gap"), ("us", "S_z60"),
                      ("kr", "Pess_news_event"), ("us", "Pess_news_event"), ("kr", "D_SD_z60"), ("kr", "dD3_z60"),
                      ("kr", "H11_X"), ("us", "h10_dsd_z60")]),
        ("pct250(120창)", [("kr", "S_pct250"), ("kr", "stage10"), ("kr", "stage20"), ("kr", "SG"), ("kr", "H59_sent_bear"),
                          ("kr", "MC_A_all_pct250"), ("us", "stage10")]),
        ("기타 예열", [("kr", "h06_dsd"), ("kr", "h06_dsd_prov40"), ("us", "h10_dsd"), ("us", "LL_US_d20"), ("us", "h46_dtilt"),
                     ("us", "h50_m_z20"), ("kr", "H24_R"), ("kr", "h37_bull_conv_rate"), ("kr", "NW90"),
                     ("kr", "week_stage10"), ("kr", "h43_event_buy_SKH")]),
    ]
    rows = []
    for fname, cols in fam:
        for mk, col in cols:
            df = kr if mk == "kr" else us
            if col not in df.columns:
                continue
            v = df[col].notna()
            rows.append({"family": fname, "market": mk, "col": col, "n_valid": int(v.sum()),
                         "first": str(df.D[v].min()) if v.any() else None})
    return rows


# ---------------------------------------------------------------- 실행

def main() -> int:
    t0 = datetime.now()
    fp = {n: sha256(ANT / n) for n in FROZEN}
    frozen_ok = {n: fp[n] == FROZEN[n] for n in FROZEN}
    lex_fp = {n: sha256(RESULTS / n) for n in LEX_V02}
    lex_ok = {n: lex_fp[n] == LEX_V02[n] for n in LEX_V02}

    T = load_tables()
    cal = pd.read_parquet(RESULTS / "loader_calendar.parquet")
    cal["date"] = pd.to_datetime(cal["date"]).dt.date
    n_back = {c: trading_days(cal, *BACK, c) for c in ("krx", "nyse")}
    n_front = {c: trading_days(cal, *FRONT, c) for c in ("krx", "nyse")}
    n_back_weeks = sum(1 for k in range((BACK[1] - BACK[0]).days + 1) if (BACK[0] + timedelta(days=k)).weekday() == 0)

    units = pd.read_parquet(RESULTS / "units.parquet")
    hits = pd.read_parquet(RESULTS / "feat_topic_unit_hits.parquet", columns=["no", "cid", "utype", "excl", "score"])
    if not ((hits.no.to_numpy() == units.no.to_numpy()).all() and (hits.utype.to_numpy() == units.utype.to_numpy()).all()):
        raise RuntimeError("feat_topic_unit_hits와 units의 줄 순서가 다르다")
    hits.index = units.index

    summ = json.loads((RESULTS / "loader_summary.json").read_text("utf-8"))
    man = json.loads((RESULTS / "loader_manifest.json").read_text("utf-8"))
    salt_now = LD.salt_check()
    salt_same = salt_now.get("fingerprint") == man.get("salt_fingerprint")

    feats = feature_rows(T, n_back, n_front)
    wrows = weekly_rows(T, n_back_weeks)
    raw = raw_checks(units, hits)
    den = denominators(T, units, hits)
    cor = correlations(T)
    warm = warmup_table(T)
    price = json.loads((RESULTS / "prechecks_price.json").read_text("utf-8"))
    ext = json.loads((RESULTS / "prechecks_external.json").read_text("utf-8"))
    lexp = json.loads((RESULTS / "lexicon_v02_precision.json").read_text("utf-8"))

    pd.DataFrame(feats + wrows).to_csv(RESULTS / "prechecks_features.csv", index=False, encoding="utf-8-sig")
    out = {"generated": t0.isoformat(timespec="seconds"), "frozen_ok": frozen_ok, "lex_v02_ok": lex_ok,
           "salt": {"present": salt_now.get("present"), "same_as_loader_snapshot": salt_same},
           "n_back": n_back, "n_front": n_front, "n_back_weeks": n_back_weeks,
           "features": feats, "weekly": wrows, "raw": raw, "denominators": den, "correlations": cor, "warmup": warm}
    (RESULTS / "prechecks_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), "utf-8")

    try:
        import prechecks_report as R  # noqa: E402
    except ModuleNotFoundError:
        print("prechecks_report.py 없음: 요약 JSON만 썼다")
        return 0
    md = R.render(out, summ, price, ext, lexp, T)
    (RESULTS / "prechecks.md").write_text(md, "utf-8")
    # 실행 뒤에도 동결 문서가 그대로인지 다시 본다
    if any(sha256(ANT / n) != FROZEN[n] for n in FROZEN):
        raise RuntimeError("동결 문서 지문이 바뀌었다")
    print("ok", (datetime.now() - t0).total_seconds(), "s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
