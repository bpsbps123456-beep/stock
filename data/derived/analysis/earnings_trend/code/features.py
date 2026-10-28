"""실적 추세 연구 — 이벤트별 실적 점수 요소(발표 시점에 알 수 있는 값만).

실행: py data/derived/analysis/earnings_trend/code/features.py  → results/features.csv

요소(괄호는 사용자 프롬프트의 출처):
  성장 수준   yoy            보도자료 표의 매출 YoY(%) (퀀트 평가표·S&N ①)
  성장 가속   accel          yoy − 직전 분기 yoy (%p) (황금률 3, 2차 도함수·S&N ②)
  전망 궤도   guide_accel    다음 분기 가이던스 YoY − 이번 yoy (%p) (2-A·S&N ③)
              guide_yoy      다음 분기 가이던스 중간값의 YoY(%)
  가이던스 비트 beat_guide   이번 실적 / 직전 분기에 준 가이던스 중간값 − 1 (%) (컨센서스 대용, 시점 안전)
  연간 전망 변화 fy_change   상향 +1 / 유지 0 / 하향 −1 / 철회 −2 (S&N ③ D·F)
  EPS 서프라이즈 eps_surprise 발표 당시 컨센서스 대비(%) (yfinance)
  수익성 질   rule40, fcf_margin, sbc_gp, op_minus_sbc_pos, dilution_yoy, op_leverage (황금률 5~7·S&N ④⑤)
  정성(LLM)   narrative, tone, catalyst, showstopper (내러티브 프롬프트·렌즈 1~3, 인용 확인된 것만)
  기대치      pre60_x(발표 전 60일 QQQ 초과), ps_ttm(시총/TTM 매출) — 사용자 '좋지만 비싼 주식' 축
  규모        mcap_pre_b, size_tier
기계식 S&N(사용자 원래 33점 표를 숫자 문턱으로 옮긴 것): sn1..sn6, sn_total — 문턱은 아래 함수에 적었다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_sec import RESULTS  # noqa: E402
from peer_table import top_pct  # noqa: E402

CHANGE = {"raised": 1, "maintained": 0, "lowered": -1, "withdrawn": -2}
TONE = {"strong": 1, "neutral": 0, "cautious": -1}
STOP = {"none": 0, "hurdle": 0, "structural_potential": -1, "structural_quantified": -2}


def size_tier(m: float) -> str:
    if pd.isna(m):
        return "NA"
    return "mega" if m >= 300 else "large" if m >= 50 else "mid" if m >= 10 else "small" if m >= 2 else "micro"


def mid(g: dict | None) -> float | None:
    if not g:
        return None
    if g.get("point_musd") is not None:
        return g["point_musd"]
    lo, hi = g.get("low_musd"), g.get("high_musd")
    if lo is not None and hi is not None:
        return (lo + hi) / 2
    return lo if lo is not None else hi


def quarter_rev(q: pd.DataFrame, t: str, pe: pd.Timestamp, shift_days: int) -> float | None:
    """분기말 pe에서 shift_days 떨어진 분기의 XBRL 매출($M)."""
    qt = q[q["ticker"] == t]
    d = (pd.to_datetime(qt["period_end"]) - (pe + pd.Timedelta(days=shift_days))).abs().dt.days
    m = qt[d <= 15]
    return float(m["revenue"].iloc[0]) / 1e6 if len(m) and pd.notna(m["revenue"].iloc[0]) else None


def ttm(q: pd.DataFrame, t: str, pe: pd.Timestamp, col: str) -> float | None:
    qt = q[(q["ticker"] == t) & (pd.to_datetime(q["period_end"]) <= pe + pd.Timedelta(days=7))].tail(4)
    if len(qt) < 4 or qt[col].isna().any():
        return None
    return float(qt[col].sum())


def add_xbrl_eps(df: pd.DataFrame, q: pd.DataFrame, eps: pd.DataFrame) -> pd.DataFrame:
    """XBRL 수익성(이 발표가 다루는 분기, TTM)과 발표 당시 EPS 서프라이즈를 붙인다. df에는 yoy가 있어야 한다."""
    # XBRL 수익성(이 발표가 다루는 분기 기준, TTM)
    add = []
    for _, r in df.iterrows():
        pe = pd.Timestamp(r["period_end"]) if pd.notna(r["period_end"]) else None
        x = {}
        if pe is not None:
            qt = q[(q["ticker"] == r["ticker"])].copy()
            qt["pe"] = pd.to_datetime(qt["period_end"])
            cur = qt[(qt["pe"] - pe).abs().dt.days <= 7]
            py = qt[(qt["pe"] - (pe - pd.Timedelta(days=364))).abs().dt.days <= 10]
            c = cur.iloc[0] if len(cur) else None
            p = py.iloc[0] if len(py) else None
            rev_ttm = ttm(q, r["ticker"], pe, "revenue")
            ocf_ttm, cap_ttm = ttm(q, r["ticker"], pe, "ocf"), ttm(q, r["ticker"], pe, "capex")
            x["fcf_margin"] = (ocf_ttm - cap_ttm) / rev_ttm * 100 if rev_ttm and ocf_ttm is not None and cap_ttm is not None else np.nan
            x["ps_ttm"] = r["mcap_pre_b"] * 1e9 / rev_ttm if rev_ttm and pd.notna(r["mcap_pre_b"]) else np.nan
            if c is not None:
                gp = c["gross_profit"]
                x["sbc_gp"] = c["sbc"] / gp * 100 if pd.notna(c["sbc"]) and pd.notna(gp) and gp > 0 else np.nan
                x["op_minus_sbc_pos"] = int(c["op_income"] - c["sbc"] > 0) if pd.notna(c["op_income"]) and pd.notna(c["sbc"]) else np.nan
                x["ocf_q"] = c["ocf"]
                x["op_margin"] = c["op_income"] / c["revenue"] * 100 if pd.notna(c["op_income"]) and c["revenue"] else np.nan
                cash = (c["cash"] if pd.notna(c["cash"]) else 0) + (c["st_invest"] if pd.notna(c["st_invest"]) else 0)
                fcf_q = (c["ocf"] - c["capex"]) if pd.notna(c["ocf"]) and pd.notna(c["capex"]) else np.nan
                x["cash_runway_q"] = cash / -fcf_q if pd.notna(fcf_q) and fcf_q < 0 else 99
                if p is not None:
                    x["dilution_yoy"] = (c["dil_shares"] / p["dil_shares"] - 1) * 100 if pd.notna(c["dil_shares"]) and pd.notna(p["dil_shares"]) and p["dil_shares"] > 0 else np.nan
                    opx, opx_py = c["gross_profit"] - c["op_income"], p["gross_profit"] - p["op_income"]
                    x["opex_yoy"] = (opx / opx_py - 1) * 100 if pd.notna(opx) and pd.notna(opx_py) and opx_py > 0 else np.nan
                    x["op_margin_chg"] = x["op_margin"] - p["op_income"] / p["revenue"] * 100 if pd.notna(x.get("op_margin")) and pd.notna(p["op_income"]) and p["revenue"] else np.nan
        add.append(x)
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(add)], axis=1)
    df["rule40"] = df["yoy"] + df["fcf_margin"]
    df["op_leverage"] = df["yoy"] - df["opex_yoy"]
    # EPS 서프라이즈: 반응일 ±4일 안의 yfinance 발표
    surp = []
    for _, r in df.iterrows():
        e = eps[(eps["ticker"] == r["ticker"]) & ((eps["d"] - pd.Timestamp(r["filing_date"])).abs().dt.days <= 4)]
        surp.append(e["eps_surprise_pct"].iloc[0] if len(e) else np.nan)
    df["eps_surprise"] = pd.to_numeric(surp, errors="coerce").clip(-100, 100)
    return df


# ---------------------------------------------------------------- 기계식 S&N (사용자 33점 표)
def sn1(r) -> float:
    """① 실적의 강도(5): 역성장 1(F) / 가이던스 하향 2(D) / 압도 5 / 확실 4 / 소폭 3 / 인라인 2 / 미스 1.
    비트 정도 = 가이던스 비트(없으면 EPS 서프라이즈의 1/3 — EPS 서프라이즈는 매출 비트의 약 3배 크기)."""
    if pd.notna(r["yoy"]) and r["yoy"] < 0:
        return 1
    if r["fy_change"] in (-1, -2) or r["profit_change"] in (-1, -2):
        return 2
    b = r["beat_guide"] if pd.notna(r["beat_guide"]) else (r["eps_surprise"] / 3 if pd.notna(r["eps_surprise"]) else np.nan)
    if pd.isna(b):
        return np.nan
    return 5 if b >= 5 else 4 if b >= 2 else 3 if b >= 0.5 else 2 if b >= -0.5 else 1


def sn2(r) -> float:
    """② 비즈니스 모멘텀(6): 가속 +10%p↑ 6 / +5 5 / +1 4 / ±1 유지 3 / 둔화 2."""
    a = r["accel"]
    if pd.isna(a):
        return np.nan
    return 6 if a >= 10 else 5 if a >= 5 else 4 if a >= 1 else 3 if a > -1 else 2


def sn3(r) -> float:
    """③ 미래 전망(7): 철회 1 / 재가속 7 / 유지 6 / 소폭 둔화(−5%p 이내) 5 / 둔화 4 / 하향 3."""
    if r["fy_change"] == -2:
        return 1
    if r["fy_change"] == -1:
        return 3
    g = r["guide_accel"]
    if pd.isna(g):
        return np.nan
    return 7 if g >= 1 else 6 if g > -1 else 5 if g > -5 else 4


def sn4(r) -> float:
    """④ SBC 건전성(4): [영업익−SBC]>0 & SBC/GP<10% 4 / <15% 3 / ≤0이나 <25% 2 / OCF>0 1 / 0."""
    s, pos, ocf = r["sbc_gp"], r["op_minus_sbc_pos"], r["ocf_q"]
    if pd.isna(s):
        return np.nan
    if pos == 1 and s < 10:
        return 4
    if pos == 1 and s < 15:
        return 3
    if s < 25:
        return 2
    return 1 if pd.notna(ocf) and ocf > 0 else 0


def sn5(r) -> float:
    """⑤ 재무 자생력(6): FCF마진 >20 6 / >10 5 / >0 4 / ≤0 3(현금 여유 있으면) 또는 1."""
    f = r["fcf_margin"]
    if pd.isna(f):
        return np.nan
    if f > 20:
        return 6
    if f > 10:
        return 5
    if f > 0:
        return 4
    return 3 if pd.notna(r["cash_runway_q"]) and r["cash_runway_q"] >= 8 else 1


def sn6(r) -> float:
    """⑥ 내러티브와 리스크(5): LLM 내러티브 −2..+2 → 1..5."""
    return r["narrative"] + 3 if pd.notna(r["narrative"]) else np.nan


def main() -> None:
    ev = pd.read_csv(RESULTS / "events.csv")
    q = pd.read_csv(RESULTS / "quarters.csv")
    eps = pd.read_csv(RESULTS / "eps_consensus.csv")
    eps["d"] = pd.to_datetime(eps["date"])
    rows = []
    for path in sorted((RESULTS / "extract").glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        t = rec["ticker"]
        bad = {(b["accession"], b["field"]) for b in rec["qc"]["bad_quotes"]}
        prev = None
        hist: list[dict] = []
        for e in rec["events"]:
            r = {"ticker": t, "accession": e["accession"], "is_q": bool(e.get("is_quarterly_results")),
                 "fiscal_period": e.get("fiscal_period"), "period_end_pr": e.get("period_end")}
            rev, rpy = e.get("revenue_musd"), e.get("revenue_py_musd")
            r["yoy"] = (rev / rpy - 1) * 100 if rev and rpy else e.get("revenue_yoy_pct_stated")
            qs = [h for h in hist if h["is_q"]]
            r["yoy_prev"] = qs[-1]["yoy"] if qs else np.nan
            r["yoy_prev2"] = qs[-2]["yoy"] if len(qs) > 1 else np.nan
            r["accel"] = r["yoy"] - r["yoy_prev"] if r["yoy"] is not None and pd.notna(r["yoy_prev"]) else np.nan
            # 다음 분기 가이던스 YoY: 총매출이면 3분기 전 분기(= 다음 분기의 전년 동기) 매출로 나눈다
            g = e.get("next_q_guidance")
            pe = pd.Timestamp(e["period_end"]) if e.get("period_end") else None
            r["guide_yoy"] = np.nan
            if g:
                gm = mid(g)
                if g.get("is_total_revenue") and gm and pe is not None:
                    base = quarter_rev(q, t, pe, 91 - 364)
                    if base:
                        r["guide_yoy"] = (gm / base - 1) * 100
                if pd.isna(r["guide_yoy"]) and g.get("guided_yoy_pct_stated") is not None:
                    r["guide_yoy"] = g["guided_yoy_pct_stated"]
            r["guide_accel"] = r["guide_yoy"] - r["yoy"] if pd.notna(r["guide_yoy"]) and r["yoy"] is not None else np.nan
            r["has_q_guide"] = int(g is not None)
            # 직전 분기 가이던스 대비 이번 실적
            r["beat_guide"] = np.nan
            pg = prev.get("next_q_guidance") if prev else None
            if pg and mid(pg):
                act = None
                pa = e.get("prior_guided_actual")
                if pa and pa.get("actual_musd"):
                    act = pa["actual_musd"]
                elif pg.get("is_total_revenue") and rev:
                    act = rev
                if act:
                    r["beat_guide"] = (act / mid(pg) - 1) * 100
            fy = e.get("fy_guidance")
            r["fy_change"] = CHANGE.get(fy.get("change_vs_prior"), np.nan) if fy else np.nan
            r["profit_change"] = CHANGE.get((e.get("profit_guidance") or {}).get("change_vs_prior"), np.nan)
            seg = e.get("key_segment") or {}
            r["seg_yoy"] = seg.get("yoy_pct")
            bl = e.get("backlog") or {}
            r["backlog_yoy"] = bl.get("yoy_pct")
            r["buyback"] = int(bool(e.get("buyback_new_musd")))
            # 정성: 인용이 원문에 없으면 그 판단을 쓰지 않는다
            r["tone"] = TONE.get(e.get("tone")) if (e["accession"], "tone_quote") not in bad else np.nan
            cat = e.get("catalyst") or {}
            r["catalyst"] = int(bool(cat.get("present"))) if not (cat.get("present") and (e["accession"], "catalyst.quote") in bad) else np.nan
            ss = e.get("showstopper") or {}
            r["showstopper"] = STOP.get(ss.get("type"), np.nan) if not (ss.get("type") != "none" and (e["accession"], "showstopper.quote") in bad) else np.nan
            r["narrative"] = e.get("narrative_shift") if (e["accession"], "narrative_quote") not in bad else np.nan
            r["one_offs"] = e.get("one_offs") or ""
            hist.append({"is_q": r["is_q"], "yoy": r["yoy"] if r["yoy"] is not None else np.nan})
            if r["is_q"]:
                prev = e
            rows.append(r)
    f = pd.DataFrame(rows)
    df = ev.merge(f, on=["ticker", "accession"], how="inner")
    df = add_xbrl_eps(df, q, eps)
    df["size_tier"] = df["mcap_pre_b"].apply(size_tier)
    # 사용자 평가표 백분위('상위 X%') → 100−X(클수록 좋음)
    df["yoy_top_pct"] = [top_pct(t, s, y) for t, s, y in zip(df["ticker"], df["size_tier"], pd.to_numeric(df["yoy"], errors="coerce"))]
    df["yoy_peer_pct"] = 100 - df["yoy_top_pct"]
    for i, fn in enumerate([sn1, sn2, sn3, sn4, sn5, sn6], 1):
        df[f"sn{i}"] = df.apply(fn, axis=1)
    df["sn_total"] = df[[f"sn{i}" for i in range(1, 7)]].sum(axis=1, min_count=6)
    rd = pd.to_datetime(df["reaction_date"])
    df["season"] = rd.dt.year.astype(str) + "Q" + rd.dt.quarter.astype(str)
    df.to_csv(RESULTS / "features.csv", index=False)
    print(f"특징 {len(df)}건 × {df.shape[1]}열. 분기 실적 {int(df['is_q'].sum())}건")
    print(df.groupby("season").size().to_string())


if __name__ == "__main__":
    main()
