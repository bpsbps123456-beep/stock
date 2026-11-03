"""SEIBRO 바닥 연구 자료판(BRIEF §2~§5, 흐름–사건 연결 전). 결제·보관을 미국 거래일 표로 옮기고, 사건은 가격만으로 정한다.

실행(연구 폴더에서): py code/build.py [--out derived] [--until-settle 2026-11-03] [--until-custody 2026-11-02]
                                      [--price-end 2026-11-03] [--prices-csv <받아 둔 prices.csv>] [--merge-rule cal|pred|brief]
읽는 것: ../seibro_flows/raw(결제 d1~d4·보관), classes.csv, derived/ipo.json, 저장소 watchlist.py, yfinance 일봉.
대조로만 읽는 것: ../seibro_flows/derived/seibro_calendar.csv, sig_daily.parquet.
출력(<out>/): units·flows_trade_product·measures_daily·measures_win·price_feats·custody_daily(.parquet),
prices.csv, events.json, semis_1x.json, qa.json, PANEL.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

STUDY = Path(__file__).resolve().parents[1]
PRIOR = STUDY.parent / "seibro_flows"
REPO = STUDY.parents[3]
sys.dont_write_bytecode = True  # 앞선 연구 폴더(읽기 전용)에 __pycache__를 쓰지 않는다
sys.path.insert(0, str(PRIOR / "code"))
import calendar_map as cm  # noqa: E402  앞선 연구 날짜 규약(calendars·seibro_of)
from events import zigzag_tops  # noqa: E402  앞선 연구 지그재그(종가)

NONEQ = {"BOND", "CASH", "CMDTY", "VIX", "CRYPTO", "FX"}
DUMMY = "US1234567824"
T1 = pd.Timestamp("2024-05-28")
BREAKS = [pd.Timestamp(d) for d in ("2024-05-28", "2025-12-15", "2026-05-22", "2026-07-31")]
SCOPE = pd.Timestamp("2023-01-02")
B15_EXPECT = ["2023-10-30", "2024-04-19", "2024-08-07", "2025-04-08", "2026-03-30", "2026-07-29"]
B10_EXPECT = sorted(B15_EXPECT + ["2024-09-06", "2024-11-15", "2025-11-20", "2026-06-05", "2026-09-14"])
SEMI_GROUPS = ["칩 설계", "파운드리·후공정", "메모리·스토리지", "장비·소재·테스트"]
SEMI_EXTRA = ["SKHY", "SKHYV"]
TICKERS = ["SOXX", "SOXL", "SOXS", "^SOX", "QQQ", "SPY", "EWY", "NVDA"]
LS = (1, 5, 15)
WINS = {"W1": (-20, -6), "W2": (-5, -1), "W3": (0, 0), "W4": (1, 5), "W5": (6, 20)}
GIVEN_SOXX = {"2026-07-29": 465.00, "2026-10-22": 475.69, "2026-10-28": 521.94, "2026-10-29": 513.86,
              "2026-10-30": 471.51, "2026-11-02": 453.65}
CHAIN_RE = re.compile(r"\s(SPLR|MRGR|EXOF|CHAN|SOFF)\s\d+\s([A-Z]{2}[A-Z0-9]{9}\d)\s*$")
LEV_WORDS = re.compile(  # 앞선 연구 classify_prep.py와 같은 단어
    r"\d(?:\.\d+)?\s?X\b|\bX\d|ULTRA|\bBULL|\bBEAR|\bBU\b|\bBE\b|INVERSE|\bSHORT|\bSHT\b|\bSRT\b|\bLONG\b|\bLNG\b|LEVERAG|\bLEV\b|"
    r"DIREX|DIRXN|\bDXN\b|PROSHARES|PROETF|GRANITE|GRNTSHR|GNTSH|T-REX|\bTREX\b|\bREX\b|DEFIANCE|TRADR|LEVERAGE SHARES|"
    r"ROUNDHILL|RUNDH|VOLATILITY SHARES|MICROSECTORS|\bAXS\b|TUTTLE|VS TRUST|\bETF\b|\bETN\b|\bETP\b|FUND|\bTRUST\b|\bSHS\b|"
    r"YIELDMAX|\bYL\b|ISHARES|SPDR|VANGUARD|INVESCO|SCHWAB|GLOBAL X|ARK |KRANE|WISDOMTREE|FIRST TRUST|JP MORGAN|JPMORGAN|PIMCO",
    re.I)
# 2026-11-03 새 ISIN(10/26~11/02 결제) 가운데 규칙 기본값(보통주 1배)과 다르게 본 것. 이름으로 판단했다.
MANUAL = {
    "CA85207H1047": {"utype": "CMDTY", "issuer": "Sprott", "underlying": "실물 금", "single": False,
                     "note": "SPROTT PHYSICAL GOLD TRUST: 금 실물 신탁 → 비주식(CMDTY)"},
    "US0321085732": {"utype": "EQ_INDEX", "issuer": "Amplify", "underlying": "글로벌 클라우드 기술주", "single": False,
                     "note": "AMPLIFY GLOBAL CLOUD TECH: 1배 주식 ETF. MRGR 상대 US26924G8050은 자료에 없어 잇지 않음"},
    "US9396531017": {"note": "WASHINGTON REAL ESTATE INVESTMENT TRUST: 리츠 보통주 → 기본값(EQ_SINGLE 1배)"},
}

# 측정(BRIEF §4): 코드, 종류(sum = 금액 합, ratio = 창 합의 비, cust = 보관), 분자, 분모, 설명
MEAS = [
    ("A1", "sum", "A1", None, "전체 거래 규모: d3 Top50 tot 합"),
    ("A2", "sum", "A2", None, "전체 매수 규모: d1 Top50 buy 합"),
    ("A3", "sum", "A3", None, "전체 매도 규모: d2 Top50 sell 합"),
    ("A4", "sum", "A4", None, "합집합 순매수 합: 네 목록 합집합 전체 net 합(근사)"),
    ("X1", "sum", "SOXL_net", None, "SOXL 순매수"),
    ("X2", "sum", "SOXL_buy", None, "SOXL 매수"),
    ("X3", "sum", "SOXL_sell", None, "SOXL 매도"),
    ("X4", "ratio", "SOXL_net", "N_plus", "SOXL 순매수 비중 n(SOXL) ÷ N⁺"),
    ("X5", "ratio", "SOXL_tot", "A1", "SOXL 거래 몫 t(SOXL) ÷ A1"),
    ("X6", "ratio", "SOXL_sell", "SOXL_buy", "SOXL 매도 ÷ 매수"),
    ("H1", "sum", "SOXS_net", None, "SOXS 순매수"),
    ("H2", "ratio", "SOXL_buy", "SOXLS_buy", "SOXL 강세 몫 b(SOXL) ÷ (b(SOXL) + b(SOXS))"),
    ("H3", "ratio", "INV_buy", "B_eq", "주식 인버스 매수 몫 Σb(EQ_INV) ÷ ΣB_eq"),
    ("H4", "sum", "INV_net", None, "주식 인버스 순매수 Σn(EQ_INV)"),
    ("L1", "sum", "EQ3L_net", None, "3배 롱 순매수 Σn(EQ3L)"),
    ("L2", "ratio", "EQ3L_net", "N_plus", "3배 롱 순매수 비중 Σn(EQ3L) ÷ N⁺"),
    ("L3", "sum", "LSGL_net", None, "단일종목 레버리지 롱 순매수 Σn(lev_long & single)"),
    ("L4", "ratio", "LIDX_buy", "SPOT_buy", "지수형 레버리지 롱 매수 ÷ 현물 주식 매수"),
    ("R1", "sum", "SEMI1_net", None, "1배 반도체 순매수(semis_1x.json)"),
    ("R2", "sum", "IDX1_net", None, "1배 지수 ETF 순매수(반도체 제외)"),
    ("R3", "sum", "BC_net", None, "채권·현금 순매수(BOND·CASH, sign ≥ 0)"),
    ("R4", "sum", "SGL1_net", None, "그 밖 1배 개별주 순매수(EQ_SINGLE, R1 제외, 분류 밖 포함)"),
    ("R5", "sum", "INC_net", None, "인컴(커버드콜) 순매수(utype INCOME)"),
    ("C1", "cust", None, None, "SOXL 보유 좌수 변화 Δlog(보관 ÷ SOXL 종가), 창 시작 전 → 창 끝"),
    ("C2", "cust", None, None, "SOXL 보관 비중 = 창 끝 SOXL 보관 ÷ 보관 Top50 합"),
    ("C3", "cust", None, None, "보관 Top50 합 변화 Δlog, 창 시작 전 → 창 끝"),
]
MCODES = [m[0] for m in MEAS]
# 절단 판정에 쓰는 열(측정마다): SOXL·SOXS는 목록 부재, 묶음 측정은 '목록에 든 구성원 0개'
CENS_OF = {**{c: ["cens_SOXL"] for c in ("X1", "X2", "X3", "X4", "X5", "X6")}, "H1": ["cens_SOXS"],
           "H2": ["cens_SOXL", "cens_SOXS"], "H3": ["zero_INV"], "H4": ["zero_INV"], "L1": ["zero_EQ3L"],
           "L2": ["zero_EQ3L"], "L3": ["zero_LSGL"], "L4": ["zero_LIDX"], "R1": ["zero_SEMI1"], "R2": ["zero_IDX1"],
           "R3": ["zero_BC"], "R4": ["zero_SGL1"], "R5": ["zero_INC"],
           "C1": ["cens_cust_SOXL"], "C2": ["cens_cust_SOXL"], "C3": []}
SUM_COLS = ["A1", "A2", "A3", "A4", "SOXL_net", "SOXL_buy", "SOXL_sell", "SOXL_tot", "SOXS_net", "SOXS_buy",
            "SOXS_sell", "SOXLS_buy", "N_plus", "B_eq", "SPOT_buy", "INV_buy", "INV_net", "EQ3L_net", "LSGL_net",
            "LIDX_buy", "SEMI1_net", "IDX1_net", "BC_net", "SGL1_net", "INC_net", "N_plus_ipo", "B_eq_ipo"]
CATS = {"INV": "EQ_INV", "EQ3L": "EQ3L", "LSGL": "LSGL", "LIDX": "LIDX", "SEMI1": "R1", "IDX1": "R2", "BC": "R3",
        "SGL1": "R4", "INC": "R5"}


# ---------- 공통 ----------
def ksd_sessions() -> pd.DatetimeIndex:
    """KSD 영업일 = XKRX 세션 − 2026-06-03(앞선 연구 signals.ksd와 같음)."""
    import exchange_calendars as xc

    s = xc.get_calendar("XKRX", start="2022-12-01", end="2027-03-01").sessions
    return pd.DatetimeIndex(s[s != pd.Timestamp("2026-06-03")])


def nyse_sessions() -> pd.DatetimeIndex:
    import exchange_calendars as xc

    return pd.DatetimeIndex(xc.get_calendar("XNYS", start="2022-06-01", end="2027-03-01").sessions)


def next_after(idx: pd.DatetimeIndex, d) -> pd.Timestamp:
    i = idx.searchsorted(pd.Timestamp(d), side="right")
    return idx[i] if i < len(idx) else pd.NaT


def on_or_after(idx: pd.DatetimeIndex, d) -> pd.Timestamp:
    i = idx.searchsorted(pd.Timestamp(d), side="left")
    return idx[i] if i < len(idx) else pd.NaT


def iso(d) -> str | None:
    return None if d is None or pd.isna(d) else pd.Timestamp(d).date().isoformat()


def r(x, n=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


def md_table(df: pd.DataFrame, fmt="{:.4g}") -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("" if not np.isfinite(v) else fmt.format(v))
            elif v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append("")
            else:
                cells.append(str(v).replace("|", "/"))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# ---------- 1. 결제 원자료 ----------
def load_settle(until: pd.Timestamp, qa: dict):
    recs, files = [], []
    for d in "1234":
        for p in sorted((PRIOR / "raw" / "daily" / f"d{d}").glob("*.json")):
            rec = json.loads(p.read_text(encoding="utf-8"))
            s = pd.Timestamp(rec["start"])
            if s > until:
                continue
            if rec["start"] != rec["end"]:
                raise ValueError(f"하루 창이 아님: {p}")
            files.append((s, d, len(rec["rows"])))
            for x in rec["rows"]:
                recs.append((s, d, x["rank"], x["isin"], x["name"], x["buy"], x["sell"], x["tot"], x["net"]))
    raw = pd.DataFrame(recs, columns=["sdate", "dtype", "rank", "isin", "name", "buy", "sell", "tot", "net"])
    fl = pd.DataFrame(files, columns=["sdate", "dtype", "rows"])
    per = fl.groupby("sdate")["dtype"].nunique()
    amt = ["buy", "sell", "tot", "net"]
    g = raw.groupby(["sdate", "isin"])
    spread = g[amt].max() - g[amt].min()
    first = raw.sort_values(["sdate", "isin", "dtype"]).groupby(["sdate", "isin"])[["name"] + amt].first()
    ranks = raw.pivot_table(index=["sdate", "isin"], columns="dtype", values="rank", aggfunc="min")
    ranks.columns = [f"rank_d{c}" for c in ranks.columns]
    sd = first.join(ranks).reset_index()
    for c in "1234":
        sd[f"in_d{c}"] = sd[f"rank_d{c}"].notna()
    dates = pd.DatetimeIndex(sorted(fl["sdate"].unique()))
    tot_d3 = raw[raw["dtype"] == "3"].groupby("sdate")["tot"].sum().reindex(dates).fillna(0.0)
    names = raw.sort_values("sdate").groupby("isin")["name"].last()
    qa["settle"] = {
        "files": int(len(fl)), "dates": int(len(dates)), "first": iso(dates[0]), "last": iso(dates[-1]),
        "dates_missing_lists": [iso(d) for d in per[per < 4].index],
        "rows": int(len(raw)), "isin_rows": int(len(sd)), "unique_isins": int(sd["isin"].nunique()),
        "inconsistent_rows": int((spread > 0.5).any(axis=1).sum()),
        "identity_bad_rows": int(((raw["tot"] - raw["buy"] - raw["sell"]).abs() > 1).sum()
                                 + ((raw["net"] - raw["buy"] + raw["sell"]).abs() > 1).sum()),
        "empty_list_dates": [iso(d) for d in fl[(fl["dtype"] == "3") & (fl["rows"] == 0)]["sdate"]],
        "dummy_rows": int((raw["isin"] == DUMMY).sum()),
        "dummy_dates": int(sd.loc[sd["isin"] == DUMMY, "sdate"].nunique()),
        "dummy_tot_sum": float(sd.loc[sd["isin"] == DUMMY, "tot"].sum()),
    }
    return sd, dates, tot_d3, names


# ---------- 2. 날짜표·묶음 ----------
def build_units(dates: pd.DatetimeIndex, tot_d3: pd.Series, ksd, nyse, qa: dict, merge_rule: str = "brief"):
    nyse_l, settle = cm.calendars()
    first, last = dates[0].date(), dates[-1].date()
    m: dict = {}
    for t in nyse_l:
        if dt.date(2022, 12, 26) <= t <= last:
            s = cm.seibro_of(t, settle)
            if first <= s <= last:
                m.setdefault(s, []).append(t)
    cal = pd.DataFrame({"seibro_date": dates})
    cal["pred_trades"] = [";".join(t.isoformat() for t in m.get(d.date(), [])) for d in dates]
    cal["n_pred"] = [len(m.get(d.date(), [])) for d in dates]
    cal["total_d3"] = tot_d3.values
    # 관측 분류는 calendar_map.main()과 같다: 한 거래일 몫 날짜의 41날짜 가운데 이동 중앙값 대비 비율
    single = cal["total_d3"].where(cal["n_pred"] == 1)
    cal["med41"] = single.rolling(41, center=True, min_periods=10).median()
    cal["ratio"] = cal["total_d3"] / cal["med41"]
    cal["obs"] = (pd.cut(cal["ratio"], [-1, 0.3, 1.6, 99], labels=["empty", "normal", "double"])
                  .astype(object).fillna("na").astype(str))
    cal["pred"] = np.where(cal["n_pred"] == 0, "empty", np.where(cal["n_pred"] >= 2, "double", "normal"))
    cal["match"] = cal["obs"] == cal["pred"]
    # 날짜 밀림 흔적 = 빈 날이 낀 어긋남. 관측 double·예측 normal(거래 급증일)은 밀림이 아니다
    shift = (~cal["match"] & ((cal["obs"] == "empty") | (cal["pred"] == "empty"))).values
    if merge_rule == "brief":  # BRIEF 초판 §2: 예측 이틀치 또는 관측 ≠ 예측(거래 급증일까지 합침)
        cal["flagged"] = (cal["n_pred"] >= 2) | ~cal["match"]
    else:  # cal(기본, 2026-11-04 연결 전 수정)·pred: 예측 이틀치와 빈 날이 낀 어긋남만
        cal["flagged"] = (cal["n_pred"] >= 2) | shift
    cal["reason"] = [";".join(([f"pred{n}"] if n >= 2 else []) + ([f"obs_{o}/pred_{p}"] if not mt else []))
                     for n, o, p, mt in zip(cal["n_pred"], cal["obs"], cal["pred"], cal["match"])]
    cal["ratio_window_partial"] = np.arange(len(cal)) >= len(cal) - 20  # 끝 20날짜는 가운데 창이 잘림(뒤 자료로 바뀔 수 있음)
    # 묶음: 표시된 날짜 + 앞뒤 '빈 날 아닌'(예측 거래일 있음 또는 관측 자료 있음) 날짜. 겹치는 묶음은 하나로 잇고 사이 날짜도 넣는다
    ne = np.flatnonzero(((cal["n_pred"] > 0) | (cal["obs"] != "empty")).values)
    spans = []
    for i in np.flatnonzero(cal["flagged"].values):
        if merge_rule == "cal" and not shift[i]:  # 예측 이틀치: 그 날짜 하나가 묶음(이웃은 합치지 않음)
            spans.append([int(i), int(i), False])
            continue
        lo_c, hi_c = ne[ne < i], ne[ne > i]
        spans.append([int(lo_c[-1]) if len(lo_c) else int(i), int(hi_c[0]) if len(hi_c) else int(i),
                      bool(len(lo_c) == 0 or len(hi_c) == 0)])
    spans.sort()
    joined: list = []
    for s in spans:
        if joined and s[0] <= joined[-1][1]:
            joined[-1][1] = max(joined[-1][1], s[1])
            joined[-1][2] = joined[-1][2] or s[2]
        else:
            joined.append(list(s))
    unit = np.full(len(cal), -1)
    span_of = {}
    for k, (lo, hi, edge) in enumerate(joined):
        unit[lo:hi + 1] = -(k + 2)  # 임시 표지
        span_of[-(k + 2)] = (lo, hi, edge)
    uid, cur, out_units = np.zeros(len(cal), int), None, []
    nid = -1
    for i in range(len(cal)):
        if unit[i] < -1:
            if unit[i] != cur:
                nid += 1
                cur = unit[i]
                lo, hi, edge = span_of[cur]
                out_units.append({"unit_id": nid, "lo": lo, "hi": hi, "merged": True, "edge": edge})
        else:
            nid += 1
            cur = None
            out_units.append({"unit_id": nid, "lo": i, "hi": i, "merged": False, "edge": False})
        uid[i] = nid
    cal["unit_id"] = uid
    rows = []
    for u in out_units:
        sub = cal.iloc[u["lo"]:u["hi"] + 1]
        tr = sorted({t for s in sub["pred_trades"] if s for t in s.split(";")})
        rows.append({"unit_id": u["unit_id"], "unit_first": sub["seibro_date"].iloc[0],
                     "unit_last": sub["seibro_date"].iloc[-1], "unit_seibro_dates": ";".join(iso(d) for d in sub["seibro_date"]),
                     "unit_trades": ";".join(tr), "unit_n_trades": len(tr), "merged": u["merged"], "edge": u["edge"],
                     "unit_reason": " | ".join(f"{iso(d)}:{x}" for d, x in zip(sub["seibro_date"], sub["reason"]) if x)})
    units = pd.DataFrame(rows)
    cal = cal.merge(units, on="unit_id", how="left")
    # 이웃 비율(진단): 앞뒤 '빈 날 아닌' 날짜의 비율
    ratio = cal["ratio"].values
    prv, nxt = np.full(len(cal), np.nan), np.full(len(cal), np.nan)
    for i in range(len(cal)):
        lo_c, hi_c = ne[ne < i], ne[ne > i]
        prv[i] = ratio[lo_c[-1]] if len(lo_c) else np.nan
        nxt[i] = ratio[hi_c[0]] if len(hi_c) else np.nan
    cal["nbr_prev_ratio"], cal["nbr_next_ratio"] = prv, nxt
    cal["known_kst_date"] = [next_after(ksd, d) for d in cal["seibro_date"]]
    # 거래일 표
    tm = []
    for _, u in units.iterrows():
        for t in (u["unit_trades"].split(";") if u["unit_trades"] else []):
            tm.append({"date": pd.Timestamp(t), "unit_id": u["unit_id"], "unit_n_trades": u["unit_n_trades"],
                       "merged": u["merged"], "edge": u["edge"], "unit_first": u["unit_first"],
                       "unit_last": u["unit_last"]})
    tmap = pd.DataFrame(tm).set_index("date").sort_index()
    pred_of = {pd.Timestamp(t): d for d, s in zip(cal["seibro_date"], cal["pred_trades"]) if s for t in s.split(";")}
    tmap["seibro_date"] = [pred_of[t] for t in tmap.index]
    tmap["known_raw"] = [next_after(ksd, d) for d in tmap["seibro_date"]]
    tmap["known_settle"] = [next_after(ksd, d) for d in tmap["unit_last"]]
    tmap["entry_settle"] = [on_or_after(nyse, k) for k in tmap["known_settle"]]
    # 점검: 거래일이 빠짐없이 한 번씩, 최신 날짜 대응
    sess = nyse[(nyse >= tmap.index.min()) & (nyse <= tmap.index.max())]
    newest = {iso(d): s for d, s in zip(cal["seibro_date"].tail(6), cal["pred_trades"].tail(6))}
    qa["calendar"] = {
        "seibro_dates": int(len(cal)), "trade_dates": int(len(tmap)), "trade_first": iso(tmap.index.min()),
        "trade_last": iso(tmap.index.max()), "trade_dates_missing": [iso(d) for d in sess.difference(tmap.index)],
        "trade_dates_not_session": [iso(d) for d in tmap.index.difference(sess)],
        "match_rate": r(cal["match"].mean()), "mismatch": int((~cal["match"]).sum()),
        "pred_multi": int((cal["n_pred"] >= 2).sum()), "flagged": int(cal["flagged"].sum()),
        "units": int(len(units)), "merged_units": int(units["merged"].sum()),
        "merged_trade_dates": int(tmap["merged"].sum()), "edge_units": int(units["edge"].sum()),
        "newest": newest, "newest_ok": newest.get("2026-11-02") == "2026-10-29" if "2026-11-02" in newest else None,
        "orphan_dates": [iso(d) for d in cal.loc[(cal["unit_n_trades"] == 0) & (cal["total_d3"] > 0), "seibro_date"]],
        "max_unit_trades": int(units["unit_n_trades"].max()),
    }
    if "2026-11-02" in newest and newest["2026-11-02"] != "2026-10-29":
        raise SystemExit(f"날짜 대응 오류: SEIBRO 2026-11-02 → {newest['2026-11-02']}")
    # 앞선 연구 날짜표(≤ 2026-10-23)와 대조
    prev = pd.read_csv(PRIOR / "derived" / "seibro_calendar.csv", parse_dates=["seibro_date"])
    j = prev.merge(cal, on="seibro_date", suffixes=("_prior", ""), how="left")
    j["trades_prior"] = j["trades"].fillna("")
    same = {k: int((j[f"{a}_prior" if a != "trades" else "trades_prior"].astype(str)
                    == j[b].astype(str)).sum())
            for k, a, b in [("trades", "trades", "pred_trades"), ("n_pred", "n_pred", "n_pred"),
                            ("obs", "obs", "obs"), ("pred", "pred", "pred"), ("match", "match", "match")]}
    diff = j[(j["obs_prior"] != j["obs"]) | (j["trades_prior"] != j["pred_trades"]) | (j["match_prior"] != j["match"])]
    qa["calendar_vs_prior"] = {
        "dates_prior": int(len(prev)), "identical": same,
        "ratio_absdiff_max": r((j["ratio_prior"] - j["ratio"]).abs().max(), 6),
        "diff_rows": diff[["seibro_date", "trades_prior", "pred_trades", "ratio_prior", "ratio", "obs_prior", "obs"]]
        .assign(seibro_date=lambda d: d["seibro_date"].dt.date.astype(str)).astype(str).to_dict("records"),
    }
    return cal, units, tmap


# ---------- 3. 분류 ----------
def load_classes(names: pd.Series, seen: pd.DataFrame, qa: dict) -> pd.DataFrame:
    c = pd.read_csv(PRIOR / "classes.csv")
    for k in ("semi", "single"):
        c[k] = c[k].astype(str).str.lower().eq("true")
    c["product"] = c["product"].fillna(c["isin"])  # 티커 없는 규칙 분류 행 → ISIN을 상품 키로
    c["src"] = "classes"
    c["lev_words"] = False
    base = c.set_index("isin")
    new = []
    for isin in sorted(set(names.index) - set(base.index)):
        name = str(names[isin])
        rec = {"isin": isin, "product": isin, "issuer": None, "L": 1.0, "sign": 0, "utype": "EQ_SINGLE",
               "underlying": name, "semi": False, "single": True, "confidence": "rule",
               "note": "새 ISIN: 분류 밖 → 보통주 1배", "src": "new_rule", "lev_words": bool(LEV_WORDS.search(name))}
        mt = CHAIN_RE.search(name)
        if mt and mt.group(1) in ("SPLR", "CHAN") and mt.group(2) in base.index:
            t = base.loc[mt.group(2)]
            rec.update({k: t[k] for k in ("product", "issuer", "L", "sign", "utype", "semi", "single")})
            rec.update(src="new_chain", note=f"새 ISIN: {mt.group(1)} 사슬 → {mt.group(2)}(상품 {t['product']})")
        if isin in MANUAL:
            rec.update(MANUAL[isin])
            rec["src"] = "new_manual"
        new.append(rec)
    cls = pd.concat([c, pd.DataFrame(new)], ignore_index=True) if new else c
    nd = pd.DataFrame(new)
    if len(nd):
        nd = nd.merge(seen, on="isin", how="left")
        nd["name"] = nd["isin"].map(names)
    qa["new_isins"] = nd[["isin", "name", "product", "utype", "L", "sign", "src", "lev_words", "first", "last",
                          "n_dates", "lists", "tot_sum"]].astype(str).to_dict("records") if len(nd) else []
    qa["new_isins_unreviewed_levwords"] = [x["isin"] for x in new if x["lev_words"] and x["src"] != "new_manual"]
    return cls


def product_table(cls: pd.DataFrame, qa: dict) -> pd.DataFrame:
    cols = ["utype", "L", "sign", "semi", "single"]
    g = cls.groupby("product")
    incons = [p for p, d in g if len(d[cols].drop_duplicates()) > 1]
    qa["product_class_inconsistent"] = incons
    # 측정에 쓰는 표지가 사슬 안 ISIN끼리 같은지 따로 본다(다르면 첫 행 분류가 일부 기간을 틀리게 셈)
    fl = cls[["product", "utype", "L", "sign", "semi", "single"]].copy()
    fl["equity"] = ~fl["utype"].isin(NONEQ)
    fl["lev_long"] = fl["equity"] & (fl["sign"] == 1) & (fl["L"] >= 1.25)
    fl["EQ3L"] = fl["lev_long"] & (fl["L"] >= 3)
    fl["EQ_INV"] = fl["equity"] & (fl["sign"] == -1)
    fl["spot"] = fl["equity"] & (fl["sign"] == 0)
    fl["LSGL"] = fl["lev_long"] & fl["single"]
    fl["LIDX"] = fl["lev_long"] & fl["utype"].eq("EQ_INDEX")
    fl["R2"] = fl["utype"].eq("EQ_INDEX") & (fl["L"] == 1) & (fl["sign"] == 0) & ~fl["semi"]
    fl["R3"] = fl["utype"].isin(["BOND", "CASH"]) & (fl["sign"] >= 0)
    fl["R4b"] = fl["utype"].eq("EQ_SINGLE") & (fl["L"] == 1) & (fl["sign"] == 0)
    fl["R5"] = fl["utype"].eq("INCOME")
    fl["R1c"] = fl["semi"] & (fl["L"] == 1) & (fl["sign"] == 0)
    fcols = ["equity", "lev_long", "EQ3L", "EQ_INV", "spot", "LSGL", "LIDX", "R2", "R3", "R4b", "R5", "R1c"]
    bad = fl.groupby("product")[fcols].nunique().gt(1).any(axis=1)
    qa["product_flag_inconsistent"] = sorted(bad[bad].index)
    pc = g[cols].first()  # 사슬 안 분류가 다르면 classes.csv 첫 행을 쓴다(측정 표지가 갈리는 상품은 위 목록)
    pc["isins_all"] = g["isin"].agg(lambda s: ";".join(sorted(s)))
    pc["equity"] = ~pc["utype"].isin(NONEQ)
    pc["lev_long"] = pc["equity"] & (pc["sign"] == 1) & (pc["L"] >= 1.25)
    pc["EQ3L"] = pc["lev_long"] & (pc["L"] >= 3)
    pc["IDX"] = pc["utype"].eq("EQ_INDEX")
    pc["EQ_INV"] = pc["equity"] & (pc["sign"] == -1)
    pc["spot"] = pc["equity"] & (pc["sign"] == 0)
    pc["LSGL"] = pc["lev_long"] & pc["single"]
    pc["LIDX"] = pc["lev_long"] & pc["IDX"]
    pc["dummy"] = pc.index == DUMMY
    return pc


def semis_list(pc: pd.DataFrame) -> tuple[list, dict]:
    sys.path.insert(0, str(REPO))
    import watchlist

    tick = []
    for grp in SEMI_GROUPS:
        if grp not in watchlist.AI_UNIVERSE:
            raise SystemExit(f"watchlist.AI_UNIVERSE에 '{grp}' 없음")
        for syms in watchlist.AI_UNIVERSE[grp].values():
            tick += [s for s in syms if "." not in s]
    tick = sorted(set(tick) | set(SEMI_EXTRA))
    matched = {t: (t if t in pc.index else None) for t in tick}
    cls_semi = sorted(pc.index[pc["semi"] & (pc["L"] == 1) & (pc["sign"] == 0)])
    prods = sorted({p for p in matched.values() if p} | set(cls_semi))
    info = {"groups": SEMI_GROUPS, "extra": SEMI_EXTRA, "tickers": matched,
            "unmatched": [t for t, p in matched.items() if p is None], "classes_semi_1x": cls_semi,
            "classes_semi_1x_utype": {p: pc.loc[p, "utype"] for p in cls_semi}, "products": prods}
    return prods, info


def load_ipo(prod_of: dict, qa: dict) -> dict:
    ipo = json.loads((PRIOR / "derived" / "ipo.json").read_text(encoding="utf-8"))
    out = {}
    for isin, v in ipo.items():
        out[prod_of.get(isin, isin)] = pd.Timestamp(v["first_trade"])
    qa["ipo"] = {"n": len(out), "latest": sorted(((iso(d), p) for p, d in out.items()))[-5:]}
    return out


def ipo_check(sd: pd.DataFrame, prod_of: dict, qa: dict) -> None:
    """앞선 ipo.py 기준으로 새 신규 상장 후보를 본다(보고만). 첫 등장 ≥ 2026-09-24(앞선 판정 때 첫 20날짜가 덜 찬 것),
    ipo.json 밖, 첫 20 SEIBRO 날짜 tot ≥ 2천만 달러 → yfinance 첫 일봉이 첫 등장 45일 전 이후면 '규칙상 신규 상장'."""
    known = set(json.loads((PRIOR / "derived" / "ipo.json").read_text(encoding="utf-8")))
    first = sd.groupby("isin")["sdate"].min()
    rows = []
    for isin in first.index[(first >= pd.Timestamp("2026-09-24"))]:
        if isin in known:
            continue
        sub = sd[sd["isin"] == isin]
        ds = sorted(sub["sdate"].unique())[:20]
        amt = float(sub.loc[sub["sdate"].isin(ds), "tot"].sum())
        if amt < 2e7:
            continue
        prod = prod_of.get(isin, isin)
        yf_first = None
        if prod != isin:
            try:
                import yfinance as yf

                h = yf.Ticker(prod).history(period="max", auto_adjust=False)
                yf_first = iso(h.index[0].tz_localize(None)) if len(h) else None
            except Exception as e:  # 받지 못하면 보고만
                yf_first = f"error:{type(e).__name__}"
        rule = bool(yf_first and not yf_first.startswith("error")
                    and pd.Timestamp(yf_first) >= first[isin] - pd.Timedelta(days=45))
        rows.append({"isin": isin, "product": prod, "name": str(sub["name"].iloc[-1])[:60], "first_seen": iso(first[isin]),
                     "dates_seen": int(sub["sdate"].nunique()), "tot_first20": round(amt), "yf_first": yf_first,
                     "rule_ipo": rule, "first_seen_after_prior": bool(first[isin] > pd.Timestamp("2026-10-23"))})
    qa["ipo_candidates"] = rows


# ---------- 4. 거래일 × 상품 ----------
def build_flows(sd, cal, units, tmap, pc, prod_of, ipo, qa) -> pd.DataFrame:
    s = sd.merge(cal[["seibro_date", "unit_id", "unit_n_trades"]], left_on="sdate", right_on="seibro_date", how="left")
    orphan = s[s["unit_n_trades"].fillna(0) == 0]
    qa["orphan"] = {"rows": int(len(orphan)), "dates": sorted({iso(d) for d in orphan["sdate"]}),
                    "tot_sum": r(orphan["tot"].sum(), 0), "soxl_rows": int((orphan["isin"].map(prod_of) == "SOXL").sum()),
                    "soxl_net_sum": r(orphan.loc[orphan["isin"].map(prod_of) == "SOXL", "net"].sum(), 0),
                    "by_date": {iso(d): float(v) for d, v in orphan.groupby("sdate")["tot"].sum().items()}}
    s = s[s["unit_n_trades"] > 0].copy()
    s["product"] = s["isin"].map(prod_of)
    for col, b, flag in (("buy_d1", "buy", "in_d1"), ("sell_d2", "sell", "in_d2"), ("tot_d3", "tot", "in_d3"),
                         ("net_d4", "net", "in_d4")):
        s[col] = s[b].where(s[flag], 0.0)
    amt = ["buy", "sell", "net", "tot", "buy_d1", "sell_d2", "tot_d3", "net_d4"]
    g = s.groupby(["unit_id", "product"])
    agg = g[amt].sum().join(g[["in_d1", "in_d2", "in_d3", "in_d4"]].any())
    # 부분 절단 판정은 '빈 날 아닌' 날짜(예측 거래일 있음 또는 관측 empty 아님)만 센다(묶음 사이에 낀 거의 빈 날짜는 뺌)
    nonempty = cal[(cal["n_pred"] > 0) | (cal["obs"] != "empty")]
    s["ne"] = s["sdate"].isin(set(nonempty["seibro_date"]))
    agg["n_in"] = s[s["ne"]].groupby(["unit_id", "product"])["sdate"].nunique().reindex(agg.index).fillna(0).astype(int)
    agg["isins"] = g["isin"].agg(lambda x: ";".join(sorted(set(x))))
    agg = agg.reset_index()
    agg["n_data"] = agg["unit_id"].map(nonempty.groupby("unit_id").size()).fillna(0).astype(int)
    ut = tmap.reset_index()[["date", "unit_id", "unit_n_trades", "merged", "unit_last"]]
    f = agg.merge(ut, on="unit_id", how="inner")
    f[amt] = f[amt].div(f["unit_n_trades"], axis=0)  # 묶음 금액을 담긴 미국 거래일 수로 똑같이 나눈다
    f["in_any_list"] = True
    f["full_obs"] = f["n_in"] == f["n_data"]
    # SOXL·SOXS는 날마다 행을 둔다(목록 밖이면 금액 0, in_any_list False = 절단)
    pad = []
    have = set(zip(f["date"], f["product"]))
    for t, u in tmap.iterrows():
        for p in ("SOXL", "SOXS"):
            if (t, p) not in have:
                pad.append({"date": t, "product": p, "isins": "", **{c: 0.0 for c in amt},
                            "in_d1": False, "in_d2": False, "in_d3": False, "in_d4": False, "n_in": 0,
                            "n_data": np.nan, "unit_id": u["unit_id"], "unit_n_trades": u["unit_n_trades"],
                            "merged": u["merged"], "unit_last": u["unit_last"], "in_any_list": False,
                            "full_obs": False})
    if pad:
        f = pd.concat([f, pd.DataFrame(pad)], ignore_index=True)
    # 분류 열과 신규 상장 제외 표시(묶음의 마지막 거래일 기준, 첫 거래 ~ +28일)
    keep = ["utype", "L", "sign", "semi", "single", "equity", "lev_long", "EQ3L", "IDX", "EQ_INV", "spot", "LSGL",
            "LIDX", "R1", "R2", "R3", "R4", "R5", "dummy"]
    f = f.join(pc[keep], on="product")
    last_tr = f["unit_id"].map(tmap.reset_index().groupby("unit_id")["date"].max())
    first = f["product"].map(ipo)
    f["ipo_excl"] = first.notna() & (first <= last_tr) & (last_tr <= first + pd.Timedelta(days=28))
    f = f.drop(columns=["unit_last"]).sort_values(["date", "product"]).reset_index(drop=True)
    cols = ["date", "product", "isins", "buy", "sell", "net", "tot", "buy_d1", "sell_d2", "tot_d3", "net_d4",
            "in_d1", "in_d2", "in_d3", "in_d4", "in_any_list", "full_obs", "n_in", "n_data", "merged", "unit_id",
            "unit_n_trades", "ipo_excl"] + keep
    return f[cols]


# ---------- 5. 가격 ----------
def get_prices(price_end: pd.Timestamp, csv: str | None, qa: dict) -> pd.DataFrame:
    if csv:
        px = pd.read_csv(csv, parse_dates=["date"])
        src = f"file:{csv}"
    else:
        import yfinance as yf

        frames = []
        for t in TICKERS:
            df = yf.download(t, start="2022-06-01", end=(price_end + pd.Timedelta(days=1)).date().isoformat(),
                             auto_adjust=False, progress=False, actions=False, threads=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index().rename(columns={"Date": "date", "Adj Close": "AdjClose"})
            df["ticker"] = t
            frames.append(df[["date", "ticker", "Open", "High", "Low", "Close", "AdjClose", "Volume"]])
        px = pd.concat(frames, ignore_index=True)
        px["date"] = pd.to_datetime(px["date"]).dt.tz_localize(None).dt.normalize()
        src = "yfinance auto_adjust=False"
    px = px[(px["date"] >= "2022-06-01") & (px["date"] <= price_end)].sort_values(["ticker", "date"])
    qa["prices"] = {"source": src, "rows": {t: int((px["ticker"] == t).sum()) for t in TICKERS},
                    "last": {t: iso(px.loc[px["ticker"] == t, "date"].max()) for t in TICKERS}}
    return px.reset_index(drop=True)


def price_feats(px: pd.DataFrame, nyse: pd.DatetimeIndex, price_end, qa: dict) -> pd.DataFrame:
    sx = px[px["ticker"] == "SOXX"].set_index("date").sort_index()
    sl = px[px["ticker"] == "SOXL"].set_index("date")["Close"]
    sess = nyse[(nyse >= sx.index.min()) & (nyse <= price_end)]
    qa["prices"]["soxx_missing_sessions"] = [iso(d) for d in sess.difference(sx.index)]
    qa["prices"]["soxx_extra_dates"] = [iso(d) for d in sx.index.difference(sess)]
    qa["prices"]["soxx_given_check"] = {d: {"given": v, "yf": r(sx["Close"].get(pd.Timestamp(d), np.nan), 2)}
                                        for d, v in GIVEN_SOXX.items()}
    c, o = sx["Close"], sx["Open"]
    lr = np.log(c).diff()
    pf = pd.DataFrame(index=sx.index)
    pf["SOXX_Open"], pf["SOXX_Close"], pf["lr1"] = o, c, lr
    for L in LS:
        pf[f"ret_{L}"] = np.log(c / c.shift(L))
        pf[f"pre20_{L}"] = np.log(c.shift(L) / c.shift(L + 20))
    pf["ret_20"] = np.log(c / c.shift(20))
    pf["max60"] = c.rolling(60).max()
    pf["dd60"] = c / pf["max60"] - 1
    pf["rv20"] = lr.rolling(20).std()
    pf["min10"] = c.rolling(10).min()
    pf["is_10d_low"] = (c <= pf["min10"]) & pf["min10"].notna()
    pf["SOXL_Close"] = sl.reindex(pf.index)
    pf["regime"] = np.searchsorted(np.array(BREAKS, dtype="datetime64[ns]"), pf.index.values.astype("datetime64[ns]"),
                                   side="right")
    pf.index.name = "date"
    return pf


# ---------- 6. 거래일 측정(L = 1 재료) ----------
def daily_helpers(f: pd.DataFrame, tmap: pd.DataFrame) -> pd.DataFrame:
    dates = tmap.index
    f = f[~f["dummy"].fillna(False).astype(bool)]

    def S(mask, col):
        return f.loc[mask].groupby("date")[col].sum().reindex(dates, fill_value=0.0)

    def N(mask):
        return f.loc[mask & f["in_any_list"]].groupby("date")["product"].nunique().reindex(dates, fill_value=0)

    allr = pd.Series(True, index=f.index)
    md = pd.DataFrame(index=dates)
    md["A1"], md["A2"], md["A3"], md["A4"] = S(allr, "tot_d3"), S(allr, "buy_d1"), S(allr, "sell_d2"), S(allr, "net")
    for p in ("SOXL", "SOXS"):
        sub = f[f["product"] == p].set_index("date")
        for col in ("net", "buy", "sell", "tot"):
            md[f"{p}_{col}"] = sub[col].reindex(dates).fillna(0.0)
        md[f"cens_{p}"] = ~sub["in_any_list"].reindex(dates).fillna(False).astype(bool)
        md[f"part_{p}"] = (sub["in_any_list"] & ~sub["full_obs"]).reindex(dates).fillna(False).astype(bool)
    md["SOXLS_buy"] = md["SOXL_buy"] + md["SOXS_buy"]
    eq = f["equity"].astype(bool)
    den = eq & ~f["ipo_excl"]
    md["N_plus"] = f.loc[den].assign(npos=lambda d: d["net"].clip(lower=0)).groupby("date")["npos"].sum() \
        .reindex(dates, fill_value=0.0)
    md["N_plus_ipo"] = f.loc[eq & f["ipo_excl"]].assign(npos=lambda d: d["net"].clip(lower=0)).groupby("date")["npos"] \
        .sum().reindex(dates, fill_value=0.0)
    md["B_eq"], md["B_eq_ipo"] = S(den, "buy_d1"), S(eq & f["ipo_excl"], "buy_d1")
    md["SPOT_buy"] = S(den & f["spot"].astype(bool), "buy")
    md["INV_buy"], md["INV_net"] = S(f["EQ_INV"].astype(bool), "buy"), S(f["EQ_INV"].astype(bool), "net")
    md["EQ3L_net"] = S(f["EQ3L"].astype(bool), "net")
    md["LSGL_net"] = S(f["LSGL"].astype(bool), "net")
    md["LIDX_buy"] = S(f["LIDX"].astype(bool), "buy")
    for name, col in (("SEMI1", "R1"), ("IDX1", "R2"), ("BC", "R3"), ("SGL1", "R4"), ("INC", "R5")):
        md[f"{name}_net"] = S(f[col].astype(bool), "net")
    for name, col in CATS.items():
        md[f"n_{name}"] = N(f[col].astype(bool))
        md[f"zero_{name}"] = md[f"n_{name}"] == 0
    md["n_products"] = N(allr)
    for c in ("seibro_date", "unit_id", "unit_n_trades", "merged", "edge", "unit_first", "unit_last", "known_raw",
              "known_settle", "entry_settle"):
        md[c] = tmap[c]
    md.index.name = "date"
    return md


# ---------- 7. 보관 ----------
def load_custody(until: pd.Timestamp, qa: dict) -> pd.DataFrame:
    recs, empty = [], []
    for p in sorted((PRIOR / "raw" / "custody" / "amt").glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        d = pd.Timestamp(rec["start"])
        if d > until:
            continue
        if not rec["rows"]:
            empty.append(iso(d))
        for x in rec["rows"]:
            recs.append((d, x["rank"], x["isin"], x["name"], x["amt"]))
    cu = pd.DataFrame(recs, columns=["cdate", "rank", "isin", "name", "amt"])
    qa["custody"] = {"dates": int(cu["cdate"].nunique()), "first": iso(cu["cdate"].min()), "last": iso(cu["cdate"].max()),
                     "empty_dates": empty, "dummy_rows": int((cu["isin"] == DUMMY).sum())}
    return cu


def build_custody(cu, prod_of, pf, nyse, ksd, md, qa):
    cu = cu[cu["isin"] != DUMMY].copy()
    cu["product"] = cu["isin"].map(prod_of)
    long = cu.groupby(["cdate", "product"]).agg(amt=("amt", "sum"), rank=("rank", "min"),
                                                 isins=("isin", lambda x: ";".join(sorted(set(x))))).reset_index()
    per = cu.groupby("cdate").agg(top50_total=("amt", "sum"), n_list=("isin", "size"))
    per["SOXL_amt"] = long[long["product"] == "SOXL"].set_index("cdate")["amt"].reindex(per.index)
    # 평가 종가: 보관 날짜 D와 같은 미국 날짜의 SOXL 종가(D가 휴장이면 그 전 마지막 세션)
    sess = nyse
    s0 = sess[sess.searchsorted(per.index, side="right") - 1]
    per["val_date"] = s0
    per["SOXL_close"] = pf["SOXL_Close"].reindex(s0).values
    per["SOXL_shares"] = per["SOXL_amt"] / per["SOXL_close"]
    # 지연 실측: Δ좌수 × 종가 ~ 거래일 순매수(오프셋 j = D 이하 마지막 세션에서 j세션 앞)
    pos = pd.Series(np.arange(len(sess)), index=sess)
    i0 = pos.reindex(s0).values
    net = md["SOXL_net"]
    cum = net.cumsum()
    cmg = md["merged"].astype(int).cumsum()
    y = (per["SOXL_shares"].diff() * per["SOXL_close"]).values
    taus = {j: sess[i0 - j] for j in range(5)}
    xs = {}
    for j, tau in taus.items():
        cnow = cum.reindex(tau).values
        xs[j] = cnow - np.r_[np.nan, cnow[:-1]]
    clean = (cmg.reindex(taus[0]).values - np.r_[np.nan, cmg.reindex(taus[4]).values[:-1]]) == 0
    base = np.isfinite(y) & clean & np.all([np.isfinite(xs[j]) for j in range(5)], axis=0)
    dts = per.index

    def fit(mask):
        res = {}
        for j in range(5):
            m = mask & base
            if m.sum() < 30:
                res[j] = {"r2": None, "slope": None, "n": int(m.sum())}
                continue
            x_, y_ = xs[j][m], y[m]
            sl, ic, rr, *_ = stats.linregress(x_, y_)
            res[j] = {"r2": r(rr ** 2), "slope": r(sl, 3), "n": int(m.sum())}
        return res

    groups = {"all": np.ones(len(dts), bool), "T+2(보관 ≤ 2024-05-24)": dts <= pd.Timestamp("2024-05-24"),
              "T+1(보관 ≥ 2024-06-03)": dts >= pd.Timestamp("2024-06-03")}
    for yr in sorted({d.year for d in dts}):
        groups[str(yr)] = dts.year == yr
    lagfit = {k: fit(np.asarray(v)) for k, v in groups.items()}

    def best(k):
        d = lagfit[k]
        return max((j for j in d if d[j]["r2"] is not None), key=lambda j: d[j]["r2"])

    j2, j1 = best("T+2(보관 ≤ 2024-05-24)"), best("T+1(보관 ≥ 2024-06-03)")
    tau1 = sess[i0 - j1]
    per["lag_used"] = np.where(tau1 >= T1, j1, j2)  # 세션 지연(진단용)
    # 대응(2026-11-04 검증 뒤 수정): 보관 D는 SEIBRO 날짜 s(t) ≤ D인 거래일 t까지 담는다.
    # 세션 지연(D 이하 마지막 세션 − j)은 NYSE 휴장·연준 휴일 언저리 거래일에서 하루 어긋났다.
    sd_of = md["seibro_date"].dropna().sort_index()
    sdv = sd_of.values.astype("datetime64[ns]")
    if (np.diff(sdv) < np.timedelta64(0)).any():
        raise SystemExit("거래일 → SEIBRO 날짜가 단조가 아님")
    kk = np.searchsorted(sdv, per.index.values.astype("datetime64[ns]"), side="right") - 1
    if (kk < 0).any():
        raise SystemExit("첫 거래일보다 앞선 보관 날짜")
    per["trade_date"] = sd_of.index.values[kk]
    cnew = cum.reindex(per["trade_date"]).values
    xnew = cnew - np.r_[np.nan, cnew[:-1]]
    mk = np.isfinite(y) & np.isfinite(xnew)
    rr_new = stats.linregress(xnew[mk], y[mk]).rvalue ** 2
    qa["custody"]["seibro_rule_r2"] = r(rr_new)
    per["known_kst_date"] = [next_after(ksd, next_after(ksd, d)) for d in per.index]  # 둘째 KSD 영업일
    if not per["trade_date"].is_monotonic_increasing:
        raise SystemExit("보관 → 거래일 대응이 단조가 아님")
    qa["custody"].update({
        "lag_r2": {k: {j: v[j] for j in v} for k, v in lagfit.items()}, "lag_T2": int(j2), "lag_T1": int(j1),
        "rule": "보관 D → 거래일 = SEIBRO 날짜 s(t) ≤ D인 마지막 거래일 t(2026-11-04 수정). 세션 지연 j는 진단용",
        "soxl_absent_dates": [iso(d) for d in per.index[per["SOXL_amt"].isna()]],
        "spot_2026_07_16_SOXL": r(per["SOXL_amt"].get(pd.Timestamp("2026-07-16"), np.nan), 0),
        "spot_expected": 4.448e9,
        "same_trade_date_dups": int(per["trade_date"].duplicated().sum()),
    })
    # 거래일마다 쓸 보관 관측: 대응 거래일 ≤ t인 마지막 보관 날짜(같은 거래일이 둘이면 늦은 날짜)
    tds = md.index
    tdv = per["trade_date"].values.astype("datetime64[ns]")
    k = np.searchsorted(tdv, tds.values.astype("datetime64[ns]"), side="right") - 1
    cmap = pd.DataFrame(index=tds)
    cmap["c_row"] = k
    ok = k >= 0
    cmap["c_date"] = pd.NaT
    cmap.loc[ok, "c_date"] = per.index[k[ok]]
    cmap["c_trade"] = pd.NaT
    cmap.loc[ok, "c_trade"] = per["trade_date"].values[k[ok]]
    cmap["c_stale"] = ok & (cmap["c_trade"] != cmap.index.to_series())
    cmap["c_known"] = pd.NaT
    cmap.loc[ok, "c_known"] = per["known_kst_date"].values[k[ok]]
    cmap.loc[~ok, "c_row"] = -1
    long = long.merge(per[["top50_total", "n_list", "trade_date", "lag_used", "val_date", "SOXL_close",
                           "known_kst_date"]], left_on="cdate", right_index=True, how="left")
    long["SOXL_shares"] = np.where(long["product"] == "SOXL", long["amt"] / long["SOXL_close"], np.nan)
    long = long.rename(columns={"cdate": "custody_date"}).sort_values(["custody_date", "rank"]).reset_index(drop=True)
    return long, per, cmap


# ---------- 8. 창 값·백분위·메아리 잔차 ----------
def pct_rank(v: np.ndarray, win=250, minn=120):
    P, n = np.full(len(v), np.nan), np.zeros(len(v), int)
    for i in range(len(v)):
        x = v[i]
        if not np.isfinite(x):
            continue
        w = v[max(0, i - win + 1): i + 1]
        w = w[np.isfinite(w)]
        if len(w) < minn:
            continue
        P[i] = 100.0 * ((w < x).sum() + 0.5 * (w == x).sum()) / len(w)
        n[i] = len(w)
    return P, n


def resid_full(y, X):
    out = np.full(len(y), np.nan)
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    if ok.sum() < X.shape[1] + 5:
        return out
    b, _, rank, _ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    e = y[ok] - X[ok] @ b
    sd = np.sqrt(e @ e / (ok.sum() - rank))
    if sd > 1e-12:
        out[ok] = e / sd
    return out


def resid_exp(y, X, minn=250):
    out = np.full(len(y), np.nan)
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    idx = np.flatnonzero(ok)
    for k in range(minn, len(idx)):
        tr, i = idx[:k], idx[k]
        b, _, rank, _ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        e = y[tr] - X[tr] @ b
        sd = np.sqrt(e @ e / (len(tr) - rank))
        if sd > 1e-12:
            out[i] = (y[i] - X[i] @ b) / sd
    return out


def window_values(md: pd.DataFrame, per: pd.DataFrame, cmap: pd.DataFrame, L: int) -> pd.DataFrame:
    h = md[SUM_COLS].rolling(L, min_periods=L).sum()
    v = pd.DataFrame(index=md.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        for code, kind, num, den, _ in MEAS:
            if kind == "sum":
                v[code] = h[num]
            elif kind == "ratio":
                v[code] = (h[num] / h[den]).where(h[den] != 0)
        rows = cmap["c_row"].values
        end = rows
        start = np.r_[np.full(L, -1), rows[:-L]]  # 창 시작 전 거래일(t − L)의 보관 관측
        sh = per["SOXL_shares"].values
        tot = per["top50_total"].values
        amt = per["SOXL_amt"].values

        def at(arr, ix):
            out = np.full(len(ix), np.nan)
            ok = ix >= 0
            out[ok] = arr[ix[ok]]
            return out

        v["C1"] = np.log(at(sh, end)) - np.log(at(sh, start))
        v["C2"] = at(amt, end) / at(tot, end)
        v["C3"] = np.log(at(tot, end)) - np.log(at(tot, start))
    return v


def cens_flags(md: pd.DataFrame) -> pd.DataFrame:
    """측정마다 그 거래일 값이 절단(0으로 채움)인지."""
    cf = pd.DataFrame(index=md.index)
    for code in MCODES:
        cols = CENS_OF.get(code, [])
        cf[code] = md[cols].any(axis=1) if cols else False
    return cf


def build_windows(md, pf, per, cmap, qa):
    rows, daily = [], None
    cf = cens_flags(md).astype(float)
    mg = md["merged"].astype(float)
    for L in LS:
        v = window_values(md, per, cmap, L)
        if L == 1:
            daily = v.copy()
        n_merged = mg.rolling(L, min_periods=L).sum().values
        n_cens = cf.rolling(L, min_periods=L).sum()
        X = pd.DataFrame(index=v.index)
        X["const"] = 1.0
        X["ret"] = pf[f"ret_{L}"].reindex(v.index)
        X["pre20"] = pf[f"pre20_{L}"].reindex(v.index)
        X["dd60"] = pf["dd60"].reindex(v.index)
        X["rv20"] = pf["rv20"].reindex(v.index)
        reg = pf["regime"].reindex(v.index)
        for k in range(1, len(BREAKS) + 1):
            X[f"reg{k}"] = (reg == k).astype(float)
        Xa = X.values
        for code in MCODES:
            x = v[code].values.astype(float)
            P, n = pct_rank(x)
            with np.errstate(divide="ignore", invalid="ignore"):
                q = np.clip(P / 100.0, 0.5 / np.maximum(n, 1), 1 - 0.5 / np.maximum(n, 1))
            yv = np.where(np.isfinite(P), stats.norm.ppf(q), np.nan)
            ef, ee = resid_full(yv, Xa), resid_exp(yv, Xa)
            rows.append(pd.DataFrame({"date": v.index, "L": L, "measure": code, "value": x, "P": P,
                                      "nP": n, "y": yv, "E_full": ef, "E_exp": ee, "n_merged": n_merged,
                                      "n_cens": n_cens[code].values}))
        print(f"  창 L={L} 끝", flush=True)
    mw = pd.concat(rows, ignore_index=True)
    qa["windows"] = {"rows": int(len(mw)), "by_L": {int(L): int((mw["L"] == L).sum()) for L in LS},
                     "first_P": {f"{L}": iso(mw.loc[(mw["L"] == L) & mw["P"].notna(), "date"].min()) for L in LS},
                     "first_E_exp": {f"{L}": iso(mw.loc[(mw["L"] == L) & mw["E_exp"].notna(), "date"].min()) for L in LS}}
    return mw, daily


# ---------- 9. 사건 ----------
def build_events(px, pf, md, units, nyse, ksd, price_end, out: Path, qa: dict) -> dict:
    sx = px[px["ticker"] == "SOXX"].set_index("date").sort_index()
    c_all = sx["Close"]
    c = c_all[(c_all.index >= "2022-10-01") & (c_all.index <= price_end)]
    zz = {p: zigzag_tops(c, None, p) for p in (0.10, 0.15)}
    b15 = [b["B"] for b in zz[0.15][1] if b["B"] >= iso(SCOPE)]
    raw10 = [b for b in zz[0.10][1] if b["B"] >= iso(SCOPE)]
    pos = {d: i for i, d in enumerate(c.index)}
    clusters: list = []
    for b in raw10:
        i = pos[pd.Timestamp(b["B"])]
        if clusters and i - clusters[-1][-1][0] <= 15:  # 15세션 안에 이어진 바닥 → 한 묶음
            clusters[-1].append((i, b))
        else:
            clusters.append([(i, b)])
    b10 = [min(cl, key=lambda x: c.iloc[x[0]])[1]["B"] for cl in clusters]
    check = {"B15": b15, "B15_ok": b15 == B15_EXPECT, "B10": b10, "B10_ok": b10 == B10_EXPECT,
             "B10_raw": [b["B"] for b in raw10]}
    qa["events_check"] = check
    if not (check["B15_ok"] and check["B10_ok"]):
        (out / "events_FAILED.json").write_text(json.dumps(check, ensure_ascii=False, indent=1), encoding="utf-8")
        raise SystemExit(f"사건 불일치 — 멈춤. B15={b15} B10={b10} (기대 B15={B15_EXPECT}, B10={B10_EXPECT})")

    spos = pd.Series(np.arange(len(nyse)), index=nyse)
    tops10, tops15 = zz[0.10][0], zz[0.15][0]

    def top_before(tops, d):
        prev = [t for t in tops if t["T"] < d]
        return prev[-1] if prev else None

    def close_on(d):
        return float(c_all.get(pd.Timestamp(d), np.nan))

    bots15 = {b["B"]: b for b in zz[0.15][1]}
    ev, segs = [], []
    for cl in clusters:
        keep = min(cl, key=lambda x: c.iloc[x[0]])[1]
        b = pd.Timestamp(keep["B"])
        t10 = top_before(tops10, cl[0][1]["B"])
        segs.append({"episode": keep["B"], "start": t10["T"] if t10 else None, "end": cl[-1][1]["B"]})
        rec = {"bottom": keep["B"], "close": close_on(b), "B15": keep["B"] in b15,
               "cluster": [x[1]["B"] for x in cl], "confirmed10": keep["confirmed"],
               "top10": t10["T"] if t10 else None, "top10_close": close_on(t10["T"]) if t10 else None,
               "dd10": r(close_on(b) / close_on(t10["T"]) - 1) if t10 else None}
        if rec["B15"]:
            t15 = top_before(tops15, keep["B"])
            rec.update({"top15": t15["T"], "top15_close": close_on(t15["T"]),
                        "dd15": r(close_on(b) / close_on(t15["T"]) - 1), "confirmed15": bots15[keep["B"]]["confirmed"]})
        if b in md.index:
            u = md.loc[b]
            uu = units.set_index("unit_id").loc[u["unit_id"]]
            kb = spos[b]
            entry_raw = on_or_after(nyse, u["known_raw"])
            entry_u = u["entry_settle"]
            ks = {k: nyse[kb + k] for k in range(-20, 21) if 0 <= kb + k < len(nyse)}
            mk = [k for k, d in ks.items() if d in md.index and bool(md.loc[d, "merged"])]
            nomd = [k for k, d in ks.items() if d not in md.index]
            wm = {w: sum(1 for k in range(a, z + 1) if k in mk) for w, (a, z) in WINS.items()}

            def gain(e):
                return r(sx["Open"].get(e, np.nan) / close_on(b) - 1) if pd.notna(e) and e in sx.index else None
            rec.update({
                "windows": {w: [iso(ks.get(a)), iso(ks.get(z))] for w, (a, z) in WINS.items()},
                "seibro_date": iso(u["seibro_date"]), "unit_id": int(u["unit_id"]), "merged": bool(u["merged"]),
                "unit_seibro_dates": uu["unit_seibro_dates"], "unit_trades": uu["unit_trades"],
                "known_kst_raw": f"{iso(u['known_raw'])} 09:00 KST", "entry_raw": iso(entry_raw),
                "entry_raw_k": int(spos[entry_raw] - kb) if pd.notna(entry_raw) else None,
                "known_kst_unit": f"{iso(u['known_settle'])} 09:00 KST", "entry_unit": iso(entry_u),
                "entry_unit_k": int(spos[entry_u] - kb) if pd.notna(entry_u) else None,
                "gain_to_entry_open_raw": gain(entry_raw), "gain_to_entry_open_unit": gain(entry_u),
                "merged_k": mk, "merged_by_window": wm, "k_without_flows": nomd})
        ev.append(rec)
    # 지금 진행 중인 하락
    cur = {}
    for p in (0.10, 0.15):
        tops, bots, pend = zz[p]
        lt, lb = tops[-1], bots[-1]
        ext = pd.Timestamp(pend["extreme"])
        cur[f"{int(p * 100)}"] = {
            "mode": pend["mode"], "extreme": pend["extreme"], "extreme_close": close_on(ext),
            "last_top": lt["T"], "last_top_close": close_on(lt["T"]), "last_top_confirmed": lt["confirmed"],
            "last_bottom": lb["B"], "last_bottom_close": close_on(lb["B"]), "last_bottom_confirmed": lb["confirmed"],
            "dd_latest_from_last_top": r(c.iloc[-1] / close_on(lt["T"]) - 1)}
    cur["latest"] = {"date": iso(c.index[-1]), "close": float(c.iloc[-1]),
                     "below_2026_07_29": bool(c.iloc[-1] < close_on("2026-07-29")),
                     "close_2026_07_29": close_on("2026-07-29")}
    if cur["10"]["mode"] == "down":
        segs.append({"episode": "current", "start": cur["10"]["last_top"], "end": iso(c.index[-1])})
    # F 가짜 바닥: ① 10세션 최저 ② 60세션 최고 대비 −10% 이하 ③ 뒤 20세션 안 종가 < 0.97 × close ④ B10 ±5세션 밖
    cc = c_all[c_all.index <= price_end]
    v = cc.values
    n = len(v)
    fut_min, n_after = np.full(n, np.nan), np.zeros(n, int)
    for i in range(n):
        w = v[i + 1: i + 21]
        n_after[i] = len(w)
        fut_min[i] = w.min() if len(w) else np.nan
    c1 = (cc <= cc.rolling(10).min()).values
    dd = (cc / cc.rolling(60).max() - 1).values
    c2 = dd <= -0.10
    c3 = fut_min < 0.97 * v
    bidx = [cc.index.get_loc(pd.Timestamp(b)) for b in b10]
    far = np.array([all(abs(i - j) > 5 for j in bidx) for i in range(n)])
    inscope = np.asarray(cc.index >= SCOPE)
    base = c1 & c2 & far & inscope
    isF = base & c3
    undet = base & ~c3 & (n_after < 20)
    # 강건 점검: ②를 '그날 뺀 직전 60세션 최고'로 바꿔도 같은지
    c2_alt = (cc / cc.shift(1).rolling(60).max() - 1).values <= -0.10
    alt = c1 & c2_alt & far & inscope & c3
    qa["F_alt60_diff"] = [iso(d) for i, d in enumerate(cc.index) if isF[i] != alt[i]]
    last_flow = md.index.max()

    def episode(d):
        ds = iso(d)
        for s in segs:
            if s["start"] and s["start"] <= ds <= s["end"]:
                return s["episode"]
        return "none"
    F = [{"date": iso(d), "close": float(v[i]), "dd60": r(dd[i]), "fut20_min_ret": r(fut_min[i] / v[i] - 1),
          "episode": episode(d), "has_flows": bool(d <= last_flow)} for i, d in enumerate(cc.index) if isF[i]]
    U = [{"date": iso(d), "close": float(v[i]), "dd60": r(dd[i]), "sessions_after": int(n_after[i]),
          "episode": episode(d)} for i, d in enumerate(cc.index) if undet[i]]
    eps = []
    for s in segs:
        fs = [x["date"] for x in F if x["episode"] == s["episode"]]
        eps.append({**s, "n_F": len(fs), "F": fs})
    res = {
        "params": {"zigzag": "../seibro_flows/code/events.py zigzag_tops(종가)", "close_from": "2022-10-01",
                   "price_end": iso(price_end), "scope_from": iso(SCOPE), "b10_merge_sessions": 15,
                   "F_rule": "① close = 10세션(그날 포함) 최저 ② close ÷ 60세션(그날 포함) 최고 − 1 ≤ −10% "
                             "③ 뒤 20세션 안 종가 < 0.97 × close ④ B10 바닥 ±5세션 밖. ③ 미충족 + 뒤 세션 < 20 = 미정",
                   "known": "결제 날짜 다음 KSD 영업일 09:00 KST(묶음이면 묶음 마지막 결제 날짜 기준). 진입 = 그 KST 날짜 이후 첫 NYSE 세션 시가"},
        "B15": b15, "B10": b10, "B10_events": ev, "episodes": eps,
        "F": F, "F_undetermined": U, "F_count": len(F), "F_episodes": len({x["episode"] for x in F}),
        "current": cur,
        "zigzag": {f"{int(p * 100)}": {"tops": zz[p][0], "bottoms": zz[p][1], "pending": zz[p][2]} for p in (0.10, 0.15)},
    }
    return res


# ---------- 10. 점검·문서 ----------
def prior_s1_check(md: pd.DataFrame, mw: pd.DataFrame, f: pd.DataFrame, qa: dict) -> dict:
    pr = pd.read_parquet(PRIOR / "derived" / "sig_daily.parquet").drop_duplicates("L")
    pr["t_end"], pr["t_start"] = pd.to_datetime(pr["t_end"]), pd.to_datetime(pr["t_start"])
    x4 = mw[(mw["L"] == 5) & (mw["measure"] == "X4")].set_index("date")["value"]
    j = pr[["L", "t_start", "t_end", "S1", "n_dates"]].join(x4.rename("X4_5"), on="t_end").dropna(subset=["S1", "X4_5"])
    pos = pd.Series(np.arange(len(md)), index=md.index)
    j["prior_trades"] = [pos.get(e, np.nan) - pos.get(s, np.nan) + 1 for s, e in zip(j["t_start"], j["t_end"])]
    mg = md["merged"].astype(int).rolling(5, min_periods=5).sum()
    j["my_merged"] = mg.reindex(j["t_end"]).values
    # 앞선 연구 방식 분모(5일 ISIN 합의 양수 부분 → 여기서는 상품 단위)로 다시 잰 S1
    eqf = f[f["equity"].astype(bool) & ~f["ipo_excl"] & ~f["dummy"].fillna(False).astype(bool)]
    piv = eqf.pivot_table(index="date", columns="product", values="net", aggfunc="sum").reindex(md.index).fillna(0.0)
    n5 = piv.rolling(5, min_periods=5).sum().clip(lower=0).sum(axis=1)
    s1_alt = md["SOXL_net"].rolling(5, min_periods=5).sum() / n5
    j["S1_alt"] = s1_alt.reindex(j["t_end"]).values

    def cors(d, a, b):
        return {"n": int(len(d)), "pearson": r(d[a].corr(d[b])), "spearman": r(d[a].corr(d[b], method="spearman")),
                "mean_ratio": r((d[a] / d[b]).median())}
    clean = j[(j["prior_trades"] == 5) & (j["my_merged"] == 0)]
    out = {"all": cors(j, "S1", "X4_5"), "clean(5거래일·묶음 없음)": cors(clean, "S1", "X4_5"),
           "S1_vs_S1alt": cors(j, "S1", "S1_alt"), "S1_vs_S1alt_clean": cors(clean, "S1", "S1_alt"),
           "S1alt_vs_X4_clean": cors(clean, "S1_alt", "X4_5"),
           "prior_6trade_windows": int((j["prior_trades"] > 5).sum())}
    qa["prior_S1_corr"] = out
    return out


def censor_rates(md: pd.DataFrame, mw: pd.DataFrame, ev: dict, nyse) -> pd.DataFrame:
    spos = pd.Series(np.arange(len(nyse)), index=nyse)
    near = {"B10±20": set(), "B15±20": set(), "F±20": set(), "F당일": set()}
    for e in ev["B10_events"]:
        kb = spos[pd.Timestamp(e["bottom"])]
        ds = {nyse[kb + k] for k in range(-20, 21) if 0 <= kb + k < len(nyse)}
        near["B10±20"] |= ds
        if e["B15"]:
            near["B15±20"] |= ds
    for x in ev["F"]:
        kb = spos[pd.Timestamp(x["date"])]
        near["F±20"] |= {nyse[kb + k] for k in range(-20, 21) if 0 <= kb + k < len(nyse)}
        near["F당일"].add(pd.Timestamp(x["date"]))
    rows = []
    cf = cens_flags(md)
    for code, kind, *_ in MEAS:
        cols = CENS_OF.get(code, [])
        flag = cf[code].astype(bool)
        part = md[[c.replace("cens_", "part_") for c in cols if c in ("cens_SOXL", "cens_SOXS")]].any(axis=1) \
            if any(c in ("cens_SOXL", "cens_SOXS") for c in cols) else pd.Series(False, index=md.index)
        rec = {"measure": code, "기준": "+".join(cols) if cols else "목록 합(절단 없음)", "전체": flag.mean(),
               "부분(묶음 안)": part.mean()}
        for k, ds in near.items():
            idx = md.index.isin(list(ds))
            rec[k] = flag[idx].mean() if idx.any() else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


DESC = {
    "units": {
        "seibro_date": "SEIBRO 결제 날짜(KST 기준 표기 날짜)", "pred_trades": "규약으로 예측한 담긴 미국 거래일(;로 이음)",
        "n_pred": "예측 거래일 수", "total_d3": "그 날짜 d3(합계 순위) Top50 tot 합(USD, DUMMY 포함 원값)",
        "med41": "한 거래일 몫 날짜(n_pred = 1) total_d3의 41날짜 가운데 이동 중앙값(최소 10)",
        "ratio": "total_d3 ÷ med41", "obs": "관측 분류: ratio < 0.3 empty, > 1.6 double, 그 밖 normal",
        "pred": "예측 분류: n_pred 0 empty, 1 normal, ≥2 double", "match": "obs = pred",
        "flagged": "묶음 대상(n_pred ≥ 2 또는 obs ≠ pred)", "reason": "표시 이유(predN, obs_x/pred_y)",
        "ratio_window_partial": "끝 20날짜: 가운데 창이 잘려 다음 자료로 분류가 바뀔 수 있음",
        "unit_id": "묶음 번호(날짜 순)", "unit_first": "묶음 첫 SEIBRO 날짜", "unit_last": "묶음 마지막 SEIBRO 날짜",
        "unit_seibro_dates": "묶음 SEIBRO 날짜들", "unit_trades": "묶음에 담긴 미국 거래일들",
        "unit_n_trades": "묶음 미국 거래일 수(금액을 이 수로 똑같이 나눔)", "merged": "합친 묶음이면 True",
        "edge": "자료 끝이라 앞 또는 뒤 이웃이 없는 묶음", "unit_reason": "묶음 안 표시 날짜와 이유",
        "nbr_prev_ratio": "앞 '빈 날 아닌' 날짜의 ratio(진단)", "nbr_next_ratio": "뒤 '빈 날 아닌' 날짜의 ratio(진단)",
        "known_kst_date": "그 결제 날짜를 아는 날(다음 KSD 영업일) 09:00 KST",
    },
    "flows_trade_product": {
        "date": "미국 거래일", "product": "상품 키(classes.csv product, 사슬 ISIN 묶음. 티커 없는 규칙 행·새 ISIN은 ISIN)",
        "isins": "그 묶음에서 나타난 ISIN(;로 이음, 빈칸 = 목록 밖 채움 행)",
        "buy": "매수(USD, 묶음 합 ÷ 거래일 수)", "sell": "매도", "net": "순매수", "tot": "매수+매도",
        "buy_d1": "d1 목록에 든 날의 매수만(ΣB_eq·A2 재료)", "sell_d2": "d2 목록에 든 날의 매도만(A3 재료)",
        "tot_d3": "d3 목록에 든 날의 tot만(A1 재료)", "net_d4": "d4 목록에 든 날의 net만(참고)",
        "in_d1": "묶음 안 어느 날이든 d1 목록에 들었음", "in_d2": "d2 목록", "in_d3": "d3 목록", "in_d4": "d4 목록",
        "in_any_list": "네 목록 중 하나라도 들었음. False 행은 SOXL·SOXS 채움 행(목록 밖 = 50위보다 작음, 금액 0으로 둠)",
        "full_obs": "묶음 안 '빈 날 아닌' 모든 SEIBRO 날짜에 나타났음(False면 일부 날짜 절단)",
        "n_in": "묶음 안 '빈 날 아닌' SEIBRO 날짜 중 나타난 날짜 수",
        "n_data": "묶음 안 '빈 날 아닌' SEIBRO 날짜 수(예측 거래일 있음 또는 관측 empty 아님)",
        "merged": "합친 묶음의 거래일", "unit_id": "묶음 번호(units와 같음)", "unit_n_trades": "묶음 거래일 수",
        "ipo_excl": "신규 상장 28일 안(묶음 마지막 거래일 기준) → 분모(N⁺·ΣB_eq·현물 매수)에서 뺌",
        "utype": "분류 유형", "L": "배수", "sign": "+1 롱, −1 숏, 0", "semi": "반도체(classes)",
        "single": "단일종목형(classes)", "equity": "주식(utype이 NONEQ가 아님)",
        "lev_long": "주식 & sign = 1 & L ≥ 1.25", "EQ3L": "lev_long & L ≥ 3", "IDX": "utype EQ_INDEX",
        "EQ_INV": "주식 & sign = −1", "spot": "주식 & sign = 0", "LSGL": "lev_long & single(L3)",
        "LIDX": "lev_long & IDX(L4)", "R1": "1배 반도체(semis_1x.json)",
        "R2": "EQ_INDEX & L = 1 & sign = 0 & semi 아님", "R3": "BOND·CASH & sign ≥ 0",
        "R4": "EQ_SINGLE & L = 1 & sign = 0 & R1 아님", "R5": "utype INCOME", "dummy": "DUMMY ISIN(모든 측정에서 뺌)",
    },
    "custody_daily": {
        "custody_date": "보관 날짜 D", "product": "상품 키", "amt": "보관 금액(USD, Top50, 사슬 합)",
        "rank": "보관 순위(사슬이면 최소)", "isins": "ISIN", "top50_total": "그날 보관 Top50 합(DUMMY 제외)",
        "n_list": "그날 목록 줄 수", "trade_date": "이 보관이 담는 마지막 미국 거래일(지연 실측)",
        "lag_used": "쓴 지연(D 이하 마지막 NYSE 세션에서 몇 세션 앞)", "val_date": "평가 종가 날짜(D 이하 마지막 NYSE 세션)",
        "SOXL_close": "val_date SOXL 종가(분할 조정, 배당 미조정)", "known_kst_date": "아는 날(둘째 KSD 영업일) 09:00 KST",
        "SOXL_shares": "SOXL 행만: amt ÷ SOXL_close(좌수)",
    },
    "price_feats": {
        "date": "NYSE 세션", "SOXX_Open": "SOXX 시가", "SOXX_Close": "SOXX 종가", "lr1": "SOXX 하루 로그 수익",
        "ret_1": "log(C_t ÷ C_t−1)", "ret_5": "log(C_t ÷ C_t−5)", "ret_15": "log(C_t ÷ C_t−15)",
        "pre20_1": "창(L=1) 시작 전 20세션 로그 수익 log(C_t−1 ÷ C_t−21)", "pre20_5": "log(C_t−5 ÷ C_t−25)",
        "pre20_15": "log(C_t−15 ÷ C_t−35)", "ret_20": "log(C_t ÷ C_t−20)", "max60": "60세션(그날 포함) 최고 종가",
        "dd60": "C_t ÷ max60 − 1", "rv20": "20세션 로그 수익 표준편차(ddof 1, 연율화 안 함)",
        "min10": "10세션(그날 포함) 최저 종가", "is_10d_low": "C_t = min10", "SOXL_Close": "SOXL 종가",
        "regime": "체제 번호(0: < 2024-05-28, 1: T+1, 2: ≥ 2025-12-15 B0, 3: ≥ 2026-05-22 B1, 4: ≥ 2026-07-31 B2)",
    },
    "measures_win": {
        "date": "창 끝 미국 거래일 t", "L": "창 길이(세션)", "measure": "측정 코드(MEAS)",
        "value": "창 값: 금액 측정은 L일 합, 비율은 창 합의 비, C1·C3은 t−L → t 보관 Δlog, C2는 t 값",
        "P": "백분위: 100 × (작은 수 + 0.5 × 같은 수) ÷ n, 같은 L 계열 지난 250세션(t 포함), n ≥ 120",
        "nP": "P에 쓴 n(NaN 아닌 값 수)", "y": "정규 점수 Φ⁻¹(clip(P/100, 0.5/n, 1 − 0.5/n))",
        "E_full": "메아리 잔차(전체 표본 OLS, 잔차 ÷ √(SSR/(n − rank)))",
        "E_exp": "메아리 잔차(창 끝 < t인 행만 적합, 최소 250행, 그 적합의 잔차 표준편차로 나눔)",
        "n_merged": "창 안 합친 묶음 거래일 수(0 ~ L)",
        "n_cens": "창 안 절단 거래일 수(측정별 기준은 PANEL §6, C 측정은 보관 SOXL 부재 날 수)",
    },
}


def daily_desc() -> dict:
    d = {"date": "미국 거래일(창 L = 1의 끝)"}
    for code, kind, num, den, txt in MEAS:
        d[code] = f"{txt} (L = 1 값, measures_win L = 1과 같음)"
    d.update({
        "A1": "d3 Top50 tot 합(DUMMY 제외, 모든 상품)", "SOXL_net": "SOXL 순매수(목록 밖 = 0)", "SOXL_buy": "SOXL 매수",
        "SOXL_sell": "SOXL 매도", "SOXL_tot": "SOXL 매수+매도", "SOXS_net": "SOXS 순매수(사슬 ISIN 합)", "SOXS_buy": "SOXS 매수",
        "SOXS_sell": "SOXS 매도", "SOXS_tot": "SOXS 매수+매도", "SOXLS_buy": "SOXL_buy + SOXS_buy(H2 분모)",
        "N_plus": "N⁺: 주식(NONEQ·DUMMY·신규 상장 28일 제외) 상품별 순매수 양수 부분 합",
        "N_plus_ipo": "신규 상장 제외분의 순매수 양수 합(분모 대안 재료)",
        "B_eq": "ΣB_eq: d1 목록 주식 매수 합(신규 상장 제외)", "B_eq_ipo": "신규 상장 제외분의 d1 매수",
        "SPOT_buy": "현물 주식(sign 0, 신규 상장 제외) 매수 합(L4 분모, 목록 무관 합집합)",
        "INV_buy": "EQ_INV 매수", "INV_net": "EQ_INV 순매수", "EQ3L_net": "EQ3L 순매수", "LSGL_net": "lev_long & single 순매수",
        "LIDX_buy": "lev_long & IDX 매수", "SEMI1_net": "R1 순매수", "IDX1_net": "R2 순매수", "BC_net": "R3 순매수",
        "SGL1_net": "R4 순매수", "INC_net": "R5 순매수",
        "cens_SOXL": "SOXL이 묶음 안 네 목록 어디에도 없음(절단, 금액 0으로 둠)", "part_SOXL": "묶음 일부 날짜만 나타남",
        "cens_SOXS": "SOXS 절단", "part_SOXS": "SOXS 일부 날짜 절단", "n_products": "목록에 든 상품 수",
        "seibro_date": "예측 SEIBRO 결제 날짜 s(t)", "unit_id": "묶음 번호", "unit_n_trades": "묶음 거래일 수",
        "merged": "합친 묶음의 거래일", "edge": "자료 끝 묶음", "unit_first": "묶음 첫 SEIBRO 날짜",
        "unit_last": "묶음 마지막 SEIBRO 날짜", "known_raw": "s(t) 다음 KSD 영업일(09:00 KST에 앎)",
        "known_settle": "묶음 마지막 SEIBRO 날짜 다음 KSD 영업일(묶음 값을 아는 날, t로 끝나는 모든 창에 적용)",
        "entry_settle": "known_settle 날짜 이후 첫 NYSE 세션(시가 진입)",
        "c_date": "t에 붙인 보관 날짜(대응 거래일 ≤ t인 마지막 보관)", "c_trade": "그 보관의 대응 거래일",
        "c_stale": "c_trade ≠ t(대응 보관 없음, 앞 값 사용)", "c_known": "그 보관을 아는 날(둘째 KSD 영업일)",
        "cust_SOXL_amt": "c_date SOXL 보관 금액", "cust_top50": "c_date 보관 Top50 합", "cust_SOXL_shares": "c_date SOXL 좌수",
        "cens_cust_SOXL": "c_date 보관 Top50에 SOXL 없음",
    })
    for name in CATS:
        d[f"n_{name}"] = f"목록에 든 {name} 상품 수"
        d[f"zero_{name}"] = f"{name} 상품이 하나도 목록에 없음(전부 절단 → 값 0)"
    return d


def write_panel(out: Path, qa: dict, frames: dict, cal, units, md, ev, semi_info, cens, s1, cls_new) -> None:
    L = []
    w = L.append
    q = qa
    w("# SEIBRO 바닥 연구 자료판 (`code/build.py`, 흐름–사건 연결 전)\n")
    w(f"- 만든 시각: {q['built_at']} KST. 인자: 결제 ≤ {q['args']['until_settle']}, 보관 ≤ {q['args']['until_custody']}, "
      f"가격 ≤ {q['args']['price_end']}.")
    w(f"- 미국 거래일 {q['calendar']['trade_first']} ~ {q['calendar']['trade_last']} ({q['calendar']['trade_dates']}일), "
      f"SEIBRO 결제 날짜 {q['settle']['first']} ~ {q['settle']['last']} ({q['settle']['dates']}개), "
      f"보관 {q['custody']['first']} ~ {q['custody']['last']} ({q['custody']['dates']}개).")
    w(f"- 사건 점검: B15 {'일치' if q['events_check']['B15_ok'] else '불일치'}, B10 "
      f"{'일치' if q['events_check']['B10_ok'] else '불일치'}. F 날 {ev['F_count']}개(국면 {ev['F_episodes']}개), 미정 {len(ev['F_undetermined'])}개.")
    w(f"- 묶음 규칙: `--merge-rule {q['args']['merge_rule']}`" + (" (주 분석, BRIEF §2 연결 전 수정 2026-11-04)." if q["args"]["merge_rule"] == "cal"
                                                              else " — 민감도판. 주 분석에 쓰지 않는다.") + "\n")
    w("## 0. 분석 때 주의\n")
    w3 = [x["bottom"] for x in ev["B10_events"] if 0 in x.get("merged_k", [])]
    big_orphan = [(d, a) for d, a in q["orphan"]["by_date"].items() if a >= 1e6]
    cur = ev["current"]
    caut = [
        ("바닥 날(W3)이 합친 묶음 안", ", ".join(w3) or "없음",
         "바닥 날이 묶음 안이라 W3 값이 묶음 평균(예측 이틀치면 두 날 나눔)이다. B15 6개 중 "
         f"{sum(1 for x in ev['B10_events'] if x['B15'] and x['bottom'] in w3)}개"),
        ("F 날이 합친 묶음 안", ", ".join(x for r_ in q["merged_units_near_events"] for x in r_["묶음 안 F 날"].split(", ") if x) or "없음",
         "그 F 날의 W3 값도 묶음 평균"),
        ("보관 대응", f"s(t) ≤ D 규칙(세션 지연으로는 T+2 {q['custody']['lag_T2']}·T+1 {q['custody']['lag_T1']})",
         f"보관 D는 SEIBRO 날짜가 D 이하인 거래일까지 담는다. 대응 보관이 없는 거래일 C 값은 앞 보관(c_stale {q['custody_stale']}일)"),
        ("N⁺ 창 정의", "하루 N⁺의 합", "앞선 S1(5일 ISIN 합의 양수)보다 분모가 커서 X4·L2가 작다(앞선 S1 ÷ X4 중앙 "
         f"{s1['all']['mean_ratio']})"),
        ("버린 결제 날짜", ", ".join(f"{d} {a / 1e6:,.0f}M" for d, a in big_orphan) or "없음",
         "예측 거래일 없는 날짜(관측 empty)의 금액. 묶음 규칙 밖이라 버렸다(전체 tot "
         f"{q['orphan']['tot_sum'] / 1e6:,.0f}M)"),
        ("BRIEF 초판 '지금' 행(고침)", f"마지막 고점 {cur['15']['last_top']} {cur['15']['last_top_close']:.2f}",
         f"8/17(559.12)이 아니다. 10/05가 더 높아 10%·15% 모두 10/05가 고점, 가격 끝 낙폭 {cur['15']['dd_latest_from_last_top']:+.3f}"),
        ("사슬 안 측정 표지 차이", ", ".join(q["product_flag_inconsistent"]) or "없음",
         f"분류 열이 다른 사슬 {', '.join(q['product_class_inconsistent'])}은 L(인버스)·single(INCOME)만 달라 측정에 영향 없음"),
    ]
    w(md_table(pd.DataFrame(caut, columns=["항목", "값", "뜻"])) + "\n")

    w("## 1. 핵심 점검\n")
    cons = q["conservation"]
    rows = [
        ("최신 날짜 대응", "SEIBRO 2026-11-02 ← 미국 2026-10-29",
         {True: "맞음", False: "틀림", None: "해당 없음(자료 끝이 다름)"}[q["calendar"]["newest_ok"]]),
        ("날짜표 대 앞선 연구(≤ 10/23)", f"{q['calendar_vs_prior']['dates_prior']}날짜",
         f"거래일 {q['calendar_vs_prior']['identical']['trades']} · 관측 {q['calendar_vs_prior']['identical']['obs']} · "
         f"일치 판정 {q['calendar_vs_prior']['identical']['match']} 같음"),
        ("거래일 빠짐", "NYSE 세션 대비", f"{len(q['calendar']['trade_dates_missing'])}일"),
        ("목록 간 금액 불일치", "같은 날·ISIN", f"{q['settle']['inconsistent_rows']}줄"),
        ("SOXL 순매수 보존", "거래일 합 = SEIBRO 날짜 합", f"{cons['soxl_trade_sum']:,.0f} = {cons['soxl_seibro_sum']:,.0f} (차 {cons['soxl_diff']:.3g})"),
        ("전체 순매수 보존", "묶음 밖 날짜 뺀 뒤", f"차 {cons['all_diff']:.3g}, 버린 날짜 {len(q['orphan']['dates'])}개 tot {q['orphan']['tot_sum']:,.0f}"),
        ("보관 SOXL 2026-07-16", "≈ 4.448e9", f"{q['custody']['spot_2026_07_16_SOXL']:,.0f}"),
        ("5일 X4 대 앞선 S1", "같은 끝 거래일", f"상관 {s1['all']['pearson']} (n {s1['all']['n']}), 묶음 없는 창 {s1['clean(5거래일·묶음 없음)']['pearson']}"),
    ]
    w(md_table(pd.DataFrame(rows, columns=["점검", "기준", "결과"])) + "\n")
    w("- 5일 X4와 앞선 S1이 다른 이유: ① 분모 N⁺를 앞선 연구는 'ISIN별 5일 순매수 합의 양수 부분'으로, 여기서는 '하루 N⁺의 5일 합'으로 잰다"
      "(하루 단위가 크다 → X4가 작다). ② 앞선 연구 창은 '비지 않은 SEIBRO 날짜 5개'라 이틀치 날이 끼면 6거래일을 담는다"
      f"({s1['prior_6trade_windows']}창). ③ 합친 묶음은 금액을 거래일에 똑같이 나눈다. 앞선 방식 분모로 다시 잰 S1_alt와 앞선 S1의 상관은 "
      f"전체 {s1['S1_vs_S1alt']['pearson']}, 묶음 없는 5거래일 창 {s1['S1_vs_S1alt_clean']['pearson']}(크기 비 중앙 "
      f"{s1['S1_vs_S1alt_clean']['mean_ratio']})다. 곧 이 자료판은 묶음 없는 창에서 앞선 S1을 그대로 재현하고, 전체 상관이 낮아지는 것은 "
      "②·③ 때문이다.\n")

    w("## 2. 파일\n")
    rows = []
    for name, (df, kind) in frames.items():
        if name.startswith("_"):
            continue
        dcol = "date" if "date" in df.columns else ("seibro_date" if "seibro_date" in df.columns else
                                                     ("custody_date" if "custody_date" in df.columns else None))
        rng = f"{iso(df[dcol].min())} ~ {iso(df[dcol].max())}" if dcol else ""
        rows.append((name, kind, len(df), len(df.columns), rng))
    rows += [("events.json", "사건(B15·B10·F·국면·지금)", len(ev["B10_events"]), "", ""),
             ("semis_1x.json", "1배 반도체 명단", len(semi_info["products"]), "", ""),
             ("qa.json", "점검 수치 전부", "", "", "")]
    w(md_table(pd.DataFrame(rows, columns=["파일", "내용", "행", "열", "날짜 범위"])) + "\n")

    w("## 3. 날짜표와 묶음\n")
    c = q["calendar"]
    w(f"- 규약: SEIBRO 날짜 = 미국 결제일 + 1일(주말 → 월). 결제 달력 = NYSE 세션 − 연준 휴일 + 2025-01-09. 2024-05-28 전 거래는 T+2.")
    w(f"- 관측 분류 일치율 {c['match_rate']} (불일치 {c['mismatch']}날짜), 예측 이틀치 {c['pred_multi']}날짜. 표시 {c['flagged']}날짜 → "
      f"합친 묶음 {c['merged_units']}개(담긴 거래일 {c['merged_trade_dates']}일, 묶음 최대 {c['max_unit_trades']}거래일). 끝 묶음 {c['edge_units']}개.")
    w("- 묶음 규칙(cal, 기본): 예측 2거래일 이상 날짜는 그 날짜 하나를 묶음으로 두고 담긴 거래일 수로 나눈다. 빈 날이 낀 어긋남(관측 또는 예측 empty)은 "
      "앞뒤 '빈 날 아닌' 날짜와 한 묶음으로 합치고, 겹치는 묶음은 하나로 잇는다(사이의 빈 날짜 포함). 금액은 묶음 거래일 수로 똑같이 나눈다.")
    w("- 관측 'double'인데 예측 'normal'인 날은 합치지 않는다. 앞뒤 이웃 ratio가 정상이라 날짜 밀림이 아니라 실제 거래 급증일이다"
      "(앞선 연구 README도 어긋난 25날짜를 거래 폭증일로 봤다). BRIEF 초판 규칙은 이 날까지 합쳐 바닥 날 값을 이웃과 평균했다.")
    w("- 민감도판: `--merge-rule pred`(예측 이틀치도 이웃과 합침), `--merge-rule brief`(초판 규칙). 다른 `--out`에 쓴다.\n")
    mm = cal[cal["flagged"]][["seibro_date", "pred_trades", "ratio", "obs", "pred", "nbr_prev_ratio", "nbr_next_ratio", "unit_id"]].copy()
    mm["seibro_date"] = mm["seibro_date"].dt.date.astype(str)
    w("### 3.1 표시 날짜\n")
    w(md_table(mm.rename(columns={"seibro_date": "SEIBRO", "pred_trades": "예측 거래일", "nbr_prev_ratio": "앞 이웃",
                                  "nbr_next_ratio": "뒤 이웃", "unit_id": "묶음"}), fmt="{:.2f}") + "\n")
    w("### 3.2 합친 묶음과 사건 근처(±20세션)\n")
    w(md_table(frames["_merged_near"][0]) + "\n")
    w(f"- 바닥 W3(k = 0)가 합친 묶음 안인 B10: {', '.join(x['bottom'] for x in ev['B10_events'] if 0 in x.get('merged_k', [])) or '없음'}.")
    w(f"- 앞선 연구 날짜표와 다른 날짜(≤ 10/23): {len(q['calendar_vs_prior']['diff_rows'])}개. 차이는 끝 쪽 가운데 중앙값 창이 새 자료로 바뀐 것이다(ratio 최대 차 {q['calendar_vs_prior']['ratio_absdiff_max']}).\n")
    if q["calendar_vs_prior"]["diff_rows"]:
        w(md_table(pd.DataFrame(q["calendar_vs_prior"]["diff_rows"])) + "\n")

    w("## 4. 분류\n")
    w(f"- classes.csv {q['classes']['rows']}행(상품 {q['classes']['products']}개). 티커 없는 규칙 행 {q['classes']['product_nan']}개는 ISIN을 상품 키로 썼다. "
      f"사슬 안 분류가 다른 상품: {q['product_class_inconsistent'] or '없음'}.")
    w(f"- 새 ISIN(10/23 뒤 결제·보관에서 처음 보임) {len(q['new_isins'])}개. 레버리지·발행사 단어가 든 것은 이름으로 따로 판단했다(검토 안 된 것 {len(q['new_isins_unreviewed_levwords'])}개). "
      "합계 금액이 작아 측정 영향은 미미하다.\n")
    if len(cls_new):
        w(md_table(cls_new.rename(columns={"name": "SEIBRO 이름", "product": "상품 키", "src": "출처", "first": "첫 등장",
                                           "n_dates": "날짜 수", "lists": "든 목록", "tot_sum": "tot 합"})) + "\n")
    ic = pd.DataFrame(q["ipo_candidates"])
    w(f"- 신규 상장(분모 28일 제외): 앞선 ipo.json {q['ipo']['n']}건을 그대로 썼다. 새 후보 점검(앞선 ipo.py 기준: 첫 등장 ≥ 2026-09-24, "
      f"ipo.json 밖, 첫 20 SEIBRO 날짜 tot ≥ 2천만 달러, yfinance 첫 일봉 ≥ 첫 등장 − 45일): 후보 {len(ic)}개, 규칙상 신규 상장 "
      f"{int(ic['rule_ipo'].sum()) if len(ic) else 0}개, 10/23 뒤 처음 보인 것 {int(ic['first_seen_after_prior'].sum()) if len(ic) else 0}개. "
      "더한 것은 없다(아래 표).")
    if len(ic):
        w("")
        w(md_table(ic[["isin", "product", "name", "first_seen", "dates_seen", "tot_first20", "yf_first", "rule_ipo"]]) + "\n")
    w(f"- ipo.json의 GETYQ(첫 거래 2026-10-13, SEIBRO 첫 등장 2023-04-27)는 티커 변경으로 보이는 오탐이지만 지시대로 그대로 썼다"
      f"(10/13~11/10 N⁺에서 빠진 양수 순매수 합 {q['ipo_getyq_nplus']:,.0f} USD).\n")
    w("### 4.1 1배 반도체(R1) 명단\n")
    sm = pd.DataFrame([(t, p or "(목록에 없음)") for t, p in semi_info["tickers"].items()], columns=["티커", "상품 키"])
    w(f"- watchlist 4묶음({', '.join(SEMI_GROUPS)})의 점 없는 티커 + SKHY·SKHYV: {len(sm)}개, 자료에 있는 것 {int((sm['상품 키'] != '(목록에 없음)').sum())}개. "
      f"없는 것: {', '.join(semi_info['unmatched']) or '없음'}.")
    w(f"- classes semi = True & L = 1 & sign = 0: {', '.join(f'{p}({u})' for p, u in semi_info['classes_semi_1x_utype'].items())}. "
      "이 가운데 INCOME(커버드콜) 상품은 R5에도 들어간다(정의 그대로, 겹침 금액은 qa.json `r1_r5_overlap`).\n")
    w("- 명단: " + ", ".join(t if p != "(목록에 없음)" else f"~~{t}~~" for t, p in zip(sm["티커"], sm["상품 키"]))
      + " (취소선 = 목록에 한 번도 없음)\n")

    w("## 5. 보관\n")
    cu = q["custody"]
    lr_rows = []
    for g, d in cu["lag_r2"].items():
        lr_rows.append([g] + [f"{d[j]['r2']} ({d[j]['slope']})" if d[j]["r2"] is not None else "" for j in range(5)]
                       + [d[0]["n"]])
    w("- Δ(SOXL 좌수) × 종가를 보관 날짜 D 이하 마지막 NYSE 세션에서 j세션 앞 거래일의 SOXL 순매수에 회귀했다(R², 괄호 안 기울기). "
      "합친 묶음 거래일이 든 행은 뺐다.\n")
    w(md_table(pd.DataFrame(lr_rows, columns=["구간", "j=0", "j=1", "j=2", "j=3", "j=4", "n"])) + "\n")
    w(f"- 대응 규칙(2026-11-04 수정): 보관 D는 SEIBRO 날짜 s(t) ≤ D인 거래일 t까지의 결제를 담는다(Δ좌수 × 종가 ~ 순매수 R² {cu.get('seibro_rule_r2')}). "
      f"세션 지연으로 재면 T+2 j = {cu['lag_T2']}, T+1 j = {cu['lag_T1']}이 가장 맞지만, NYSE 휴장·연준 휴일 언저리에서 하루 어긋나 대응에는 쓰지 않는다. "
      f"평가 종가는 D와 같은 미국 날짜(휴장이면 그 전 세션) SOXL 종가다.")
    w(f"- 거래일 중 대응 보관이 없어 앞 보관을 쓴 날(c_stale): {q['custody_stale']}일. 보관 Top50에 SOXL이 없는 날: {len(cu['soxl_absent_dates'])}일.\n")

    w("## 6. 측정\n")
    w("- 26개(BRIEF §4) 정의와 재료 열은 아래 사전에 있다. 창 값 = 금액은 L일 합, 비율은 L일 재료 합의 비. 목록 밖 상품은 0으로 더하고 절단 표시를 남긴다.")
    w("- 절단 비율(L = 1 거래일 기준, 값이 0으로 채워진 날의 비율). '부분'은 합친 묶음 안 일부 SEIBRO 날짜에만 나타난 날이다.\n")
    w(md_table(cens, fmt="{:.3f}") + "\n")
    w(f"- SOXL이 목록에 없는 거래일: {', '.join(iso(d) for d in md.index[md['cens_SOXL']]) or '없음'}. "
      f"SOXS가 없는 거래일: {', '.join(iso(d) for d in md.index[md['cens_SOXS']]) or '없음'}(그날 H1 = 0, H2 = 1). "
      "묶음 측정(H3~R5)의 절단은 '구성원 하나도 없음'만 셀 수 있다. 50위 밖 구성원 금액은 늘 빠져 있다.")
    w(f"- R1 대 R5 겹침(반도체 INCOME 상품): 거래일 순매수 합 {q['r1_r5_overlap']['net_sum']:,.0f} USD, 그런 상품이 목록에 든 날 {q['r1_r5_overlap']['days']}일.\n")

    w("## 7. 사건\n")
    rows = []
    for e in ev["B10_events"]:
        rows.append((e["bottom"], "B15" if e["B15"] else "B10", e.get("top10"), e.get("dd10"), e.get("top15") or "",
                     e.get("dd15") if e["B15"] else np.nan, e.get("seibro_date"), "예" if e.get("merged") else "",
                     e.get("known_kst_unit", "")[:10], e.get("entry_unit"), e.get("entry_unit_k"),
                     e.get("gain_to_entry_open_unit"), len(e.get("merged_k", []))))
    w(md_table(pd.DataFrame(rows, columns=["바닥", "묶음", "10% 고점", "낙폭10", "15% 고점", "낙폭15", "SEIBRO", "합침",
                                         "아는 날", "진입", "k", "진입 시가 상승", "±20 합친 날"]), fmt="{:.3f}") + "\n")
    w("- 아는 날 = 바닥 거래일이 든 묶음의 마지막 SEIBRO 날짜 다음 KSD 영업일 09:00 KST. 묶음이 아니면 s(t) 다음 영업일과 같다(events.json에 둘 다).")
    ep = pd.DataFrame([(x["episode"], x["start"], x["end"], x["n_F"]) for x in ev["episodes"]],
                      columns=["국면(바닥)", "10% 고점", "끝", "F 날"])
    w(f"- F 날 {ev['F_count']}개. 국면 밖(none) {sum(1 for x in ev['F'] if x['episode'] == 'none')}개, 흐름 없음(10/29 뒤) "
      f"{sum(1 for x in ev['F'] if not x['has_flows'])}개. 미정(뒤 20세션 안 됨, 3% 조건 아직 아님) {len(ev['F_undetermined'])}개: "
      f"{', '.join(x['date'] for x in ev['F_undetermined']) or '없음'}. ②를 '그날 뺀 직전 60세션 최고'로 바꿔도 달라지는 F 날: "
      f"{', '.join(q['F_alt60_diff']) or '없음'}.\n")
    w(md_table(ep) + "\n")
    cur = ev["current"]
    w(f"- 지금({cur['latest']['date']} 종가 {cur['latest']['close']:.2f}, 7/29 {cur['latest']['close_2026_07_29']:.2f} 아래: "
      f"{'예' if cur['latest']['below_2026_07_29'] else '아니오'}): 10% 지그재그 {cur['10']['mode']}, 마지막 고점 {cur['10']['last_top']} "
      f"{cur['10']['last_top_close']:.2f} 대비 {cur['10']['dd_latest_from_last_top']:+.3f}. 15% 지그재그 {cur['15']['mode']}, 마지막 고점 "
      f"{cur['15']['last_top']} {cur['15']['last_top_close']:.2f} 대비 {cur['15']['dd_latest_from_last_top']:+.3f}(미확정, 판정하지 않음).\n")

    w("## 8. BRIEF와 다르거나 해석한 점\n")
    for i, x in enumerate(q["deviations"], 1):
        w(f"{i}. {x}")
    w("")
    w("## 9. 열 사전\n")
    for name, (df, kind) in frames.items():
        if name.startswith("_"):
            continue
        key = name.replace(".parquet", "")
        desc = daily_desc() if key == "measures_daily" else DESC.get(key, {})
        w(f"### {name}\n")
        w(md_table(pd.DataFrame([(col, str(df[col].dtype), desc.get(col, "")) for col in df.columns],
                                columns=["열", "형", "뜻"])) + "\n")
    w("### prices.csv\n")
    w("long 형: date, ticker(SOXX·SOXL·SOXS·^SOX·QQQ·SPY·EWY·NVDA), Open, High, Low, Close(분할 조정), AdjClose(분할+배당), Volume. "
      "yfinance auto_adjust=False, 2022-06-01 ~ 가격 끝.\n")
    w("### events.json\n")
    w("`B15`, `B10`(날짜), `B10_events`(바닥마다: close, B15, cluster, confirmed10, top10·top10_close·dd10, B15면 top15·dd15·confirmed15, "
      "seibro_date, unit_id, merged, unit_seibro_dates, unit_trades, known_kst_raw·entry_raw·entry_raw_k, known_kst_unit·entry_unit·entry_unit_k, "
      "gain_to_entry_open_*, merged_k(k ∈ [−20, 20] 중 합친 날), merged_by_window, k_without_flows), `episodes`(국면: 10% 고점 → 묶음 마지막 바닥, F 목록), "
      "`F`(date, close, dd60, fut20_min_ret, episode, has_flows), `F_undetermined`, `current`(10%·15% 지그재그 상태와 최신 종가), `zigzag`(원 지그재그).\n")
    (out / "PANEL.md").write_text("\n".join(L), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SEIBRO 바닥 연구 자료판")
    ap.add_argument("--out", default="derived")
    ap.add_argument("--until-settle", default="2026-11-03")
    ap.add_argument("--until-custody", default="2026-11-02")
    ap.add_argument("--price-end", default="2026-11-03")
    ap.add_argument("--prices-csv", default=None, help="받아 둔 prices.csv를 다시 쓸 때(기본: yfinance에서 새로 받음)")
    ap.add_argument("--merge-rule", choices=["cal", "pred", "brief"], default="cal",
                    help="cal = 기본(예측 이틀치는 그 날짜만, 빈 날이 낀 어긋남은 이웃과 합침). "
                         "pred = 예측 이틀치도 이웃과 합침. brief = BRIEF 초판(거래 급증일까지 합침). 민감도판은 다른 --out에 쓸 것")
    a = ap.parse_args(argv)
    out = Path(a.out) if Path(a.out).is_absolute() else STUDY / a.out
    out.mkdir(parents=True, exist_ok=True)
    qa: dict = {"args": vars(a), "built_at": dt.datetime.now().isoformat(timespec="seconds")}
    us, uc, pe = pd.Timestamp(a.until_settle), pd.Timestamp(a.until_custody), pd.Timestamp(a.price_end)
    global KSD
    KSD = ksd_sessions()
    nyse = nyse_sessions()

    print("[1] 결제 원자료", flush=True)
    sd, sdates, tot_d3, names_s = load_settle(us, qa)
    print("[2] 날짜표·묶음", flush=True)
    cal, units, tmap = build_units(sdates, tot_d3, KSD, nyse, qa, a.merge_rule)
    print("[3] 분류", flush=True)
    cu = load_custody(uc, qa)
    names = pd.concat([names_s, cu.sort_values("cdate").groupby("isin")["name"].last()])
    names = names[~names.index.duplicated(keep="first")]
    seen_s = sd.groupby("isin").agg(first=("sdate", "min"), last=("sdate", "max"), n_dates=("sdate", "nunique"),
                                    tot_sum=("tot", "sum"))
    lab = (sd["in_d1"].map({True: "1", False: ""}) + sd["in_d2"].map({True: "2", False: ""})
           + sd["in_d3"].map({True: "3", False: ""}) + sd["in_d4"].map({True: "4", False: ""}))
    lists = lab.groupby(sd["isin"]).agg(lambda s: ",".join(sorted(set(s))))
    seen = seen_s.join(lists.rename("lists")).reset_index()
    seen[["first", "last"]] = seen[["first", "last"]].apply(lambda s: s.dt.date.astype(str))
    cls = load_classes(names, seen, qa)
    qa["classes"] = {"rows": int((cls["src"] == "classes").sum()), "products": int(cls["product"].nunique()),
                     "product_nan": int(pd.read_csv(PRIOR / "classes.csv")["product"].isna().sum())}
    prod_of = dict(zip(cls["isin"], cls["product"]))
    pc = product_table(cls, qa)
    semi_prods, semi_info = semis_list(pc)
    pc["R1"] = pc.index.isin(semi_prods)
    pc["R2"] = pc["utype"].eq("EQ_INDEX") & (pc["L"] == 1) & (pc["sign"] == 0) & ~pc["semi"]
    pc["R3"] = pc["utype"].isin(["BOND", "CASH"]) & (pc["sign"] >= 0)
    pc["R4"] = pc["utype"].eq("EQ_SINGLE") & (pc["L"] == 1) & (pc["sign"] == 0) & ~pc["R1"]
    pc["R5"] = pc["utype"].eq("INCOME")
    ipo = load_ipo(prod_of, qa)
    ipo_check(sd, prod_of, qa)
    (out / "semis_1x.json").write_text(json.dumps(semi_info, ensure_ascii=False, indent=1), encoding="utf-8")

    print("[4] 거래일 × 상품", flush=True)
    flows = build_flows(sd, cal, units, tmap, pc, prod_of, ipo, qa)
    # 보존: SOXL 순매수와 전체 순매수(묶음 밖 날짜는 따로)
    soxl_isins = [i for i, p in prod_of.items() if p == "SOXL"]
    sd_in = sd[~sd["sdate"].isin(pd.to_datetime(qa["orphan"]["dates"]))]
    qa["conservation"] = {
        "soxl_trade_sum": float(flows.loc[flows["product"] == "SOXL", "net"].sum()),
        "soxl_seibro_sum": float(sd_in.loc[sd_in["isin"].isin(soxl_isins), "net"].sum()),
        "all_trade_sum": float(flows["net"].sum()), "all_seibro_sum": float(sd_in["net"].sum())}
    cons = qa["conservation"]
    cons["soxl_diff"] = cons["soxl_trade_sum"] - cons["soxl_seibro_sum"]
    cons["all_diff"] = cons["all_trade_sum"] - cons["all_seibro_sum"]
    if abs(cons["soxl_diff"]) > 1.0 or abs(cons["all_diff"]) > 10.0:
        raise SystemExit(f"보존 실패: {cons}")

    print("[5] 가격", flush=True)
    px = get_prices(pe, a.prices_csv, qa)
    px.assign(date=px["date"].dt.date).to_csv(out / "prices.csv", index=False)
    pf = price_feats(px, nyse, pe, qa)
    print("[6] 거래일 측정 재료", flush=True)
    md = daily_helpers(flows, tmap)
    print("[7] 보관", flush=True)
    cdaily, per, cmap = build_custody(cu, prod_of, pf, nyse, KSD, md, qa)
    md = md.join(cmap[["c_date", "c_trade", "c_stale", "c_known"]])
    ok = cmap["c_row"].values >= 0
    for col, src in (("cust_SOXL_amt", "SOXL_amt"), ("cust_top50", "top50_total"), ("cust_SOXL_shares", "SOXL_shares")):
        vals = np.full(len(md), np.nan)
        vals[ok] = per[src].values[cmap["c_row"].values[ok]]
        md[col] = vals
    md["cens_cust_SOXL"] = ok & ~np.isfinite(md["cust_SOXL_amt"].values)
    qa["custody_stale"] = int(md["c_stale"].sum())
    print("[8] 창 값·백분위·잔차", flush=True)
    mw, v1 = build_windows(md, pf, per, cmap, qa)
    for code in MCODES:
        md[code] = v1[code]
    print("[9] 사건", flush=True)
    ev = build_events(px, pf, md, units, nyse, KSD, pe, out, qa)

    print("[10] 점검·문서", flush=True)
    s1 = prior_s1_check(md, mw, flows, qa)
    cens = censor_rates(md, mw, ev, nyse)
    qa["censor_rates"] = cens.round(4).astype(object).where(cens.notna(), None).to_dict("records")
    ov = flows[flows["R1"].astype(bool) & flows["R5"].astype(bool) & flows["in_any_list"]]
    qa["r1_r5_overlap"] = {"products": sorted(ov["product"].unique().tolist()), "net_sum": float(ov["net"].sum()),
                           "days": int(ov["date"].nunique())}
    gq = [p for p, d in ipo.items() if d == pd.Timestamp("2026-10-13")]
    qa["ipo_getyq_nplus"] = float(flows[flows["product"].isin(gq) & flows["ipo_excl"]]["net"].clip(lower=0).sum())
    # 합친 묶음 근처 사건
    spos = pd.Series(np.arange(len(nyse)), index=nyse)
    fdates = [pd.Timestamp(x["date"]) for x in ev["F"]]
    rows = []
    for _, u in units[units["merged"]].iterrows():
        tr = [pd.Timestamp(t) for t in u["unit_trades"].split(";") if t]
        if not tr:
            continue
        near_b = []
        for e in ev["B10_events"]:
            kb = spos[pd.Timestamp(e["bottom"])]
            ks = [int(spos[t] - kb) for t in tr if abs(spos[t] - kb) <= 20]
            if ks:
                near_b.append(f"{e['bottom']}{'*' if e['B15'] else ''}[k {min(ks)}..{max(ks)}]")
        nf = [d for d in fdates if any(abs(spos[t] - spos[d]) <= 20 for t in tr)]
        fin = [iso(d) for d in fdates if d in tr]
        rows.append((int(u["unit_id"]), u["unit_seibro_dates"], u["unit_trades"], int(u["unit_n_trades"]),
                     u["unit_reason"], "; ".join(near_b), len(nf), ", ".join(fin)))
    near = pd.DataFrame(rows, columns=["묶음", "SEIBRO 날짜", "미국 거래일", "거래일 수", "이유", "B10 ±20(* B15)",
                                       "F ±20 수", "묶음 안 F 날"])
    qa["merged_units_near_events"] = near.to_dict("records")
    qa["deviations"] = [
        "겹치는 묶음은 하나로 이었다(연결 성분, 사이 빈 날짜 포함). BRIEF '그 날짜와 앞뒤 한 날짜'를 겹칠 때 이어 붙인 해석이다. 그래서 묶음이 4거래일을 넘을 수 있다.",
        "앞뒤 이웃 '빈 날 아닌 날짜' = 예측 거래일이 있거나 관측이 empty가 아닌 날짜.",
        "N⁺는 상품(사슬) 단위로 하루마다 잰다. 창 값은 하루 N⁺의 합이다(앞선 연구 S1은 ISIN별 5일 합의 양수 부분). 그래서 5일 X4가 앞선 S1보다 작다.",
        "신규 상장 28일 제외를 N⁺뿐 아니라 ΣB_eq·현물 매수(L4 분모)에도 적용했다(앞선 연구 분모 규칙). 판정 날짜는 묶음의 마지막 거래일. 제외분은 N_plus_ipo·B_eq_ipo 열로 남겼다.",
        f"DUMMY ISIN(US1234567824)은 A1~A4를 포함한 모든 측정에서 뺐다(결제 {qa['settle']['dummy_dates']}날짜, "
        f"tot 합 {qa['settle']['dummy_tot_sum'] / 1e6:,.1f}M USD).",
        "R1에 classes semi = True & L = 1 & sign = 0 행을 그대로 넣어 반도체 INCOME 상품이 R5와 겹친다(이 기간 목록에 든 것: "
        + "·".join(qa["r1_r5_overlap"]["products"]) + ", 순매수 합 " + f"{qa['r1_r5_overlap']['net_sum'] / 1e6:,.0f}M USD).",
        "목록 밖(절단) 상품은 합에서 0으로 더했다. 비율은 0을 넣은 창 합으로 계산한다(H2는 SOXS 절단 날 1에 가까워짐). 절단 비율은 §6.",
        "보관 평가 종가는 D와 같은 미국 날짜, D가 휴장이면 그 전 세션 종가다. 보관 → 거래일 지연은 T+2·T+1 구간을 따로 실측해 구간별로 썼다.",
        "C1·C2·C3은 거래일마다 '대응 거래일 ≤ t인 마지막 보관'을 쓴다. 대응 보관이 없으면 앞 보관(c_stale)이다.",
        "F ②의 60세션 최고와 dd60은 그날을 포함한 60세션이다. F는 2023-01-02 ~ 가격 끝 날짜에서 찾았다. 마지막 흐름 거래일 뒤 F 날은 흐름이 없다(has_flows False).",
        "F 국면 = 10% 지그재그 고점(B10 묶음 첫 바닥 앞) ~ 묶음 마지막 바닥. 어느 하락에도 속하지 않는 F 날은 'none'으로 둔다.",
        "B10 15세션 묶음 = 이어진 두 바닥 사이 세션 수 ≤ 15이면 한 묶음, 더 낮은 종가 하나만 남김.",
        "예측 거래일이 없고 묶음에도 안 든 SEIBRO 날짜(관측 empty)의 금액은 버렸다(BRIEF 규칙 밖). 100만 달러 넘는 날짜는 §0 표에 있다"
        "(2025-01-13은 2025-01-09 NYSE 휴장 뒤 날짜다).",
        "아는 시각: 묶음 거래일의 값은 묶음 마지막 SEIBRO 날짜 다음 KSD 영업일에 안다(known_settle). s(t) 기준(known_raw)도 함께 남겼다.",
        "ipo.json의 GETYQ는 오탐으로 보이지만 지시대로 그대로 썼다(N⁺ 영향 §4).",
        "BRIEF 초판 §3 '지금' 행(8/17 고점 559.12 대비 −18.9%)은 가격 자료와 달랐다(2026-11-04 BRIEF 고침). 2026-10-05 종가 589.51이 더 높아 10%·15% 지그재그 모두 마지막 "
        "고점이 10/05다(15% 고점 확정 10/16). 사건 집합(B15·B10)은 BRIEF와 같고, 지금 상태는 events.json `current`에 계산값으로 둔다.",
        "BRIEF가 정하지 않은 열을 더했다: measures_win n_merged·n_cens·nP·y, measures_daily 보관·시각 열, events.json windows·gain_to_entry_open(가격만).",
        "자료판 검증 중 수치 대조용으로 사건 근처 값 두 개를 봤다(2026-07-29 X4 L5·L15의 P·E, 2026-10-22 측정 표본 행). "
        "정의 선택에는 쓰지 않았다(그 뒤 바꾼 것은 점검·보고 열, 부분 절단 표지의 날짜 세기, 날짜 형 맞춤, 기본이 아닌 민감도 선택지뿐이다).",
    ]
    if a.merge_rule == "cal":
        qa["deviations"].insert(0, "묶음 규칙 cal(BRIEF §2 연결 전 수정, 2026-11-04): 예측 이틀치 날짜는 그 날짜 하나만 묶음으로 두고 "
                                   "두 거래일에 나눈다. 빈 날이 낀 어긋남만 앞뒤 이웃과 합친다. 관측 double·예측 normal 날은 거래 "
                                   "급증일이라 합치지 않는다(앞선 연구 README: 어긋난 25날짜는 거래 폭증일).")
    elif a.merge_rule != "brief":
        qa["deviations"].insert(0, f"이 판은 --merge-rule {a.merge_rule} 민감도판이다(주 분석은 cal).")
    else:
        qa["deviations"].insert(0, "이 판은 --merge-rule brief(BRIEF 초판 §2) 민감도판이다: 거래 급증일까지 이웃과 합쳤다(주 분석은 cal).")
    # 저장(날짜 열은 모두 datetime64[ns]로 맞춘다)
    def ns(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype("datetime64[ns]")
        return df
    flows, cdaily, mw = ns(flows), ns(cdaily), ns(mw)
    units_out = ns(cal.copy())
    flows.to_parquet(out / "flows_trade_product.parquet", index=False)
    units_out.to_parquet(out / "units.parquet", index=False)
    md_out = md.reset_index()
    first_cols = ["date"] + MCODES
    md_out = ns(md_out[first_cols + [c for c in md_out.columns if c not in first_cols]])
    md_out.to_parquet(out / "measures_daily.parquet", index=False)
    mw.to_parquet(out / "measures_win.parquet", index=False)
    pf_out = ns(pf.reset_index())
    pf_out.to_parquet(out / "price_feats.parquet", index=False)
    cdaily.to_parquet(out / "custody_daily.parquet", index=False)
    (out / "events.json").write_text(json.dumps(ev, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    # 사전 누락 점검(모든 열에 설명이 있어야 한다)
    frames = {"units.parquet": (units_out, "SEIBRO 날짜 → 미국 거래일, 묶음"),
              "flows_trade_product.parquet": (flows, "미국 거래일 × 상품 금액"),
              "measures_daily.parquet": (md_out, "거래일 측정 26개 + 재료·절단·시각"),
              "measures_win.parquet": (mw, "창(L 1·5·15) 값·P·E, long"),
              "price_feats.parquet": (pf_out, "SOXX 가격 특성"),
              "custody_daily.parquet": (cdaily, "보관 날짜 × 상품"),
              "_merged_near": (near, "")}
    missing = {}
    for name, (df, _) in frames.items():
        if name.startswith("_"):
            continue
        key = name.replace(".parquet", "")
        desc = daily_desc() if key == "measures_daily" else DESC.get(key, {})
        miss = [c_ for c_ in df.columns if c_ not in desc]
        if miss:
            missing[name] = miss
    qa["dictionary_missing"] = missing
    cls_new = pd.DataFrame(qa["new_isins"])
    if len(cls_new):
        cls_new = cls_new[["isin", "name", "product", "utype", "src", "first", "n_dates", "lists", "tot_sum"]]
    write_panel(out, qa, frames, cal, units, md, ev, semi_info, cens, s1, cls_new)
    (out / "qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: qa[k] for k in ("calendar", "conservation", "prior_S1_corr", "events_check")},
                     ensure_ascii=False, default=str)[:3000])
    if missing:
        print("[경고] 열 사전 누락:", missing)
    return 0


KSD = None

if __name__ == "__main__":
    sys.exit(main())
