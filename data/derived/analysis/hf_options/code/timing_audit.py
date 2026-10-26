"""갈래 timing_audit — 패널 미래 참조 반박 감사(2026-10-14, 3단계 hf_options).

주장 "패널은 공개 전 정보를 쓰지 않는다"를 깨뜨리려는 점검. harness.py·grid.py·cache는 고치지 않고 읽기만 한다
(build_panel은 cache/panel.parquet을 쓰므로 이 파일은 그 쓰기를 막은 채 다시 만든다).

점검
  A. 다시 만든 패널 = cache/panel.parquet
  B. 거래일 달력(NYSE 휴장일 표와 대조, yahoo.parquet 휴장일 행), 마지막 완료 세션
  C. 결과 R_h·M_h 독립 재계산(진입일 시가 기준)
  D. 24개 신호 독립 재구성: 원값 → 독립 백분위(numpy) → 독립 배치(이 파일의 '정당한 첫 진입일' 규칙) → 패널과 칸별 대조
  E. 끝에서 끝 교란 시험: 기준일 C에 아직 쓸 수 없는 원자료만 무작위로 바꾸고 패널을 다시 만들어 C 이전 행이 그대로인지
  F. 방향(ai_cpcr5 = 1 − AI_5, ai_breadth5), 엄격 백분위 비대칭
  G. 묵은 값(ffill 한도·측정일 나이), CFTC 셧다운
  H. 갈래별 추적 예시(2026)
  I. 격자 민감도: 개장 전 공개 근거가 약한 갈래를 하루 늦춘 패널, front/back 경계 겹침 제거 → grid.json 결론 비교
실행: py timing_audit.py  → results/timing_audit.json
"""
from __future__ import annotations

import json
import sys
import time
from bisect import bisect_right
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as H  # noqa: E402
import grid as G  # noqa: E402
import fetch as osf  # noqa: E402  (options_sentiment, harness가 경로를 넣는다)
import lag as osl  # noqa: E402
import build as osb  # noqa: E402

C = H.CACHE
OUT = H.RES / "timing_audit.json"

# ---------- cache 쓰기 막기 ----------
_orig_to_parquet = pd.DataFrame.to_parquet


def _guard(self, path, *a, **k):
    p = Path(str(path)).resolve()
    if C.resolve() in p.parents or p.parent == C.resolve():
        return None  # cache/ 아래는 쓰지 않는다(panel.parquet 포함)
    return _orig_to_parquet(self, path, *a, **k)


pd.DataFrame.to_parquet = _guard

NYSE_HOL = {
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02",
    "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-01-09", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26", "2025-06-19", "2025-07-04",
    "2025-09-01", "2025-11-27", "2025-12-25",
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
    "2026-11-26", "2026-12-25"}
AGE = {s: H.DAILY_AGE for s in H.SIGNALS}
AGE.update({s: H.WEEKLY_AGE for s in ("nq_lev_net_oi", "nq_lev_chg4", "nq_am_net_oi", "nq_am_chg4", "nq_dealer_net_oi",
                                       "es_lev_net_oi", "vx_lev_net_oi", "nq_nonrept_net_oi", "aaii_spread", "naaim")})
AGE["si_ai_chg"] = H.BIMONTH_AGE
WIN = {s: 250 for s in H.SIGNALS}
WIN.update({s: 52 for s in ("nq_lev_net_oi", "nq_lev_chg4", "nq_am_net_oi", "nq_am_chg4", "nq_dealer_net_oi",
                             "es_lev_net_oi", "vx_lev_net_oi", "nq_nonrept_net_oi", "aaii_spread", "naaim")})
WIN["si_ai_chg"] = 24
# 공개 시각 근거(QA 메모 요약) — 진입 = 아래 '공개 기준일' 다음 거래일 시가
EVIDENCE = {
    "eq_pcr5": ("CBOE 일별 통계, 측정일 d", "d 저녁(가정, 이 연구 QA에 시각 근거 없음)", "d 다음 거래일"),
    "ai_cpcr5": ("OCC 고객 거래량 AI 115종목, 측정일 d", "options_sentiment BRIEF '다음 날 개장 전 공개로 본다'(가정, 근거 없음)", "d 다음 거래일"),
    "ai_breadth5": ("OCC 종목별 고객 풋/콜, 측정일 d", "위와 같음(가정)", "d 다음 거래일"),
    "skew": ("^SKEW 종가 d", "d 장 마감", "d 다음 거래일"),
    "vix_ts": ("^VIX/^VIX3M 종가 d", "d 장 마감", "d 다음 거래일"),
    "gex": ("SqueezeMetrics asof d", "d 17:56~19:55 ET(사본 24개: 2025-04-29, 2026-09-09~10-13). 2024~2025 대부분 직접 근거 없음", "asof 다음 거래일(= QA known)"),
    "dix": ("SqueezeMetrics asof d", "위와 같음", "asof 다음 거래일(= QA known)"),
    "lev_qqq_bull5": ("Yahoo TQQQ·SQQQ 종가×거래량 d(5일 합)", "d 장 마감", "d 다음 거래일"),
    "lev_soxx_bull5": ("Yahoo SOXL·SOXS d", "d 장 마감", "d 다음 거래일"),
    "gvip_rel20": ("Yahoo 수정종가 d", "d 장 마감", "d 다음 거래일"),
    "qai_beta60": ("Yahoo 수정종가 d", "d 장 마감", "d 다음 거래일"),
    "lev_qqq_flow20p": ("ProShares NAV·발행주식수 asof T", "T 20시 이후(Last-Modified 1일 20:18 ET, Wayback 1건 T+1 06:10 ET). QA known = T+1", "T 다음 거래일(= QA known 당일 시가; QA 검증자는 T+2를 가정)"),
    "shvr_ai5": ("FINRA CNMSshvol asof D(5일 평균)", "파일별 Last-Modified, 2025~ 445일 모두 D+1 9:30 전(open_ok=1, 최대 D 18:47 ET)", "open_ok=1이면 D 다음 거래일, 아니면 known 다음 거래일"),
    "si_ai_chg": ("FINRA 공매도 잔고 기준일", "기준일+7영업일 저녁, known = 그 다음 거래일", "known 다음 거래일(공개 뒤 둘째 거래일, 하루 보수)"),
    "aaii_spread": ("AAII 수요일 마감 투표", "목요일 아침(공식 문구), known = 목", "known 다음 거래일(= 금, 하루 보수일 수 있음)"),
    "naaim": ("NAAIM 수요일 노출", "엑셀 업로드 시각(16시 전이면 그날)", "known 다음 거래일"),
}
for s in ("nq_lev_net_oi", "nq_lev_chg4", "nq_am_net_oi", "nq_am_chg4", "nq_dealer_net_oi", "es_lev_net_oi", "vx_lev_net_oi",
          "nq_nonrept_net_oi"):
    EVIDENCE[s] = ("CFTC TFF 화요일 기준", "금 15:30 ET(PRE :created_at 213주 확인, 16시 뒤 공개 2주는 known 월요일로 고침)", "known 다음 거래일")


# ---------- 공용 ----------
def td_next(dd: list[str], day: str) -> int:
    """day 다음(엄격히 뒤) 첫 거래일의 위치."""
    return bisect_right(dd, day)


def pct_np(x: np.ndarray, win: int, minp: int, strict: bool = False) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        w = x[max(0, i - win + 1):i + 1]
        if len(w) >= minp:
            out[i] = np.mean(w < x[i]) if strict else np.mean(w <= x[i])
    return out


def place_indep(obs: pd.DataFrame, dd: list[str], age: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """obs: 열 value·entry_pos·meas(측정일)·known. 각 거래일에 '그날 시가 전에 쓸 수 있는 가장 최근 관측'을 age일까지.
    돌려줌: 값, 그 값의 측정일, 그 값의 진입 위치(경과 일수 계산용)."""
    n = len(dd)
    val, meas, epos = np.full(n, np.nan), np.full(n, None, dtype=object), np.full(n, -1)
    o = obs.dropna(subset=["value"]).sort_values(["entry_pos", "meas"])
    o = o[o["entry_pos"] < n]
    best = {}
    for r in o.itertuples():
        best[int(r.entry_pos)] = (r.value, r.meas)  # 같은 진입일이면 나중 측정
    cur = None
    for p in range(n):
        if p in best:
            cur = (best[p][0], best[p][1], p)
        if cur is not None and p - cur[2] <= age:
            val[p], meas[p], epos[p] = cur
    return val, meas, epos


def build_panel_nowrite() -> pd.DataFrame:
    return H.build_panel()


# ---------- A ----------
def check_rebuild() -> dict:
    P0 = pd.read_parquet(C / "panel.parquet")
    P1 = build_panel_nowrite()
    same_idx = bool((P0.index == P1.index).all())
    diff = float(np.nanmax(np.abs(P0.to_numpy() - P1.to_numpy())))
    nan_same = bool((P0.isna() == P1.isna()).all().all())
    return {"pass": same_idx and diff == 0 and nan_same, "rows": len(P0), "max_abs_diff": diff, "nan_pattern_same": nan_same}


# ---------- B ----------
def check_calendar() -> dict:
    dd = list(H.days())
    exp = [d for d in pd.bdate_range(dd[0], dd[-1]).strftime("%Y-%m-%d") if d not in NYSE_HOL]
    missing, extra = sorted(set(exp) - set(dd)), sorted(set(dd) - set(exp))
    y = pd.read_parquet(H.OS / "cache/yahoo.parquet")
    o = pd.read_parquet(H.OS / "cache/yahoo_ohlc.parquet")
    hol_rows = y[y["QQQ"].isna() & (y.index >= dd[0])].index.tolist()
    hp = pd.read_parquet(C / "hf_proxies.parquet")
    sq = pd.read_parquet(C / "squeeze.parquet")
    cb = osf.load_cboe()
    others = {}
    for nm, ix in (("hf_proxies(yfinance QQQ)", hp.index), ("squeeze asof", pd.Index(sq["asof"])), ("cboe", cb.index)):
        s = set(x for x in ix if dd[0] <= x <= dd[-1])
        others[nm] = {"missing_vs_dd": sorted(set(dd) - s)[:5], "extra_vs_dd": sorted(s - set(dd))[:5]}
    last_ohlc = o.dropna().index.max()
    skew_last = y.dropna(subset=["^SKEW"]).index.max()
    return {"pass": not missing and not extra and not (set(hol_rows) & set(dd)),
            "dd_first": dd[0], "dd_last": dd[-1], "n": len(dd), "missing_trading_days": missing, "extra_non_trading": extra,
            "yahoo_holiday_rows_QQQ_nan": hol_rows, "holiday_rows_in_dd": sorted(set(hol_rows) & set(dd)), "other_calendars": others,
            "last_session": {"yahoo_ohlc_last_complete": last_ohlc, "yahoo_parquet_last_SKEW": skew_last, "dd_last": dd[-1],
                             "yahoo_parquet_last_row_QQQ": float(y["QQQ"].iloc[-1]), "yahoo_ohlc_last_close": float(o["QQQ_c"].iloc[-1]),
                             "note": "yahoo.parquet은 10-13 12:04 ET(장중)에 받아 10-13 QQQ가 장중 값이고 SKEW가 비어 있다 → dd는 10-12에서 끝난다. "
                                     "10-13은 완료 세션이지만 빠진다(보수적, 미래 참조 아님). 10-13 장중 vix_ts 관측은 진입 10-14로 가 달력 밖에서 버려진다."}}


# ---------- C ----------
def check_outcomes() -> dict:
    dd = list(H.days())
    px = pd.read_parquet(H.OS / "cache/yahoo_ohlc.parquet").loc[dd]
    res = {"pass": True, "max_abs_diff": {}, "trace": {}}
    for a in ("QQQ", "SOXX"):
        Y = H.outcomes(a)
        o, c = px[f"{a}_o"].to_numpy(), px[f"{a}_c"].to_numpy()
        for h in H.HORIZONS:
            r, m = np.full(len(dd), np.nan), np.full(len(dd), np.nan)
            for i in range(len(dd) - h + 1):
                r[i] = (c[i + h - 1] / o[i] - 1) * 100
                path = [o[i]] + list(c[i:i + h])
                peak, mdd = -np.inf, 0.0
                for v in path:
                    peak = max(peak, v)
                    mdd = min(mdd, v / peak - 1)
                m[i] = mdd * 100
            dr = np.nanmax(np.abs(r - Y[f"R{h}"].to_numpy()))
            dm = np.nanmax(np.abs(m - Y[f"M{h}"].to_numpy()))
            nan_ok = bool((np.isnan(r) == np.isnan(Y[f"R{h}"].to_numpy())).all())
            res["max_abs_diff"][f"{a}_R{h}"], res["max_abs_diff"][f"{a}_M{h}"] = float(dr), float(dm)
            res["pass"] &= dr < 1e-9 and dm < 1e-9 and nan_ok
            res.setdefault("last_valid", {})[f"{a}_R{h}"] = Y[f"R{h}"].dropna().index.max()
    i = dd.index("2026-03-16")
    res["trace"] = {"entry": "2026-03-16", "QQQ_open_entry": float(px["QQQ_o"].iloc[i]), "QQQ_close_d+4": float(px["QQQ_c"].iloc[i + 4]),
                    "close_date_d+4": dd[i + 4], "R5_manual": float((px["QQQ_c"].iloc[i + 4] / px["QQQ_o"].iloc[i] - 1) * 100),
                    "R5_harness": float(H.outcomes("QQQ")["R5"].iloc[i]),
                    "prev_close_2026-03-13": float(px["QQQ_c"].iloc[i - 1])}
    return res


# ---------- D: 독립 재구성 ----------
def raw_observations() -> dict[str, pd.DataFrame]:
    """신호마다 관측 표: meas(측정일), known(QA), value_raw, legit(정당한 첫 진입 거래일 = 이 '공개 기준일'의 다음 거래일)."""
    dd = list(H.days())
    obs = {}

    def mk(meas, legit_after, raw, known=None):
        d = pd.DataFrame({"meas": list(meas), "legit_after": list(legit_after), "raw": np.asarray(raw, float)})
        d["known"] = list(known) if known is not None else d["legit_after"]
        d["entry_pos"] = [td_next(dd, x) for x in d["legit_after"]]
        return d

    # CBOE
    cb = osf.load_cboe()
    pv, cv = cb["equity_pv"].to_numpy(float), cb["equity_cv"].to_numpy(float)
    eq = np.array([pv[i - 4:i + 1].sum() / cv[i - 4:i + 1].sum() if i >= 4 else np.nan for i in range(len(cb))])
    obs["eq_pcr5"] = mk(cb.index, cb.index, eq)
    # OCC AI 고객 P/C(독립: 파일을 직접 읽는다)
    syms = set(osb.ai_us_symbols())
    cp, cc = {}, {}
    for f in sorted((H.OS / "cache/occ").glob("*.parquet")):
        d = pd.read_parquet(f)
        d = d[d["underlying"].isin(syms)]
        cp[f.stem], cc[f.stem] = float(d["C_P"].sum()), float(d["C_C"].sum())
    days = list(cb.index)
    P = np.array([cp.get(x, np.nan) for x in days])
    Cc = np.array([cc.get(x, np.nan) for x in days])
    ai = np.array([P[i - 4:i + 1].sum() / Cc[i - 4:i + 1].sum() if i >= 4 and not np.isnan(P[i - 4:i + 1]).any() else np.nan
                   for i in range(len(days))])
    obs["ai_cpcr5"] = mk(days, days, ai)
    # breadth 원값은 lag.indicators 내부 계산(종목별 백분위)을 그대로 쓴다 — 방향·배치만 독립 확인
    ind = osl.indicators(cb.index)
    obs["ai_breadth5"] = mk(days, days, ind["AIB_5"].to_numpy())  # 이미 백분위 → 아래에서 백분위 재계산 생략
    # Yahoo 지수
    y = pd.read_parquet(H.OS / "cache/yahoo.parquet").dropna(subset=["QQQ"])
    obs["skew"] = mk(y.index, y.index, y["^SKEW"])
    obs["vix_ts"] = mk(y.index, y.index, y["^VIX"] / y["^VIX3M"])
    # Squeeze: 측정일 asof, QA known
    sq = pd.read_parquet(C / "squeeze.parquet")
    obs["gex"] = mk(sq["asof"], sq["asof"], sq["gex_bn"], known=sq.index)
    obs["dix"] = mk(sq["asof"], sq["asof"], sq["dix"], known=sq.index)
    # Yahoo 가격 지표(hp index = 가격 날짜)
    hp = pd.read_parquet(C / "hf_proxies.parquet")
    for s in ("lev_qqq_bull5", "lev_soxx_bull5", "gvip_rel20", "qai_beta60"):
        obs[s] = mk(hp.index, hp.index, hp[s])
    fl = hp.dropna(subset=["lev_qqq_flow20"])
    obs["lev_qqq_flow20p"] = mk(fl["asof"], fl["asof"], fl["lev_qqq_flow20"] / (fl["aum_tqqq"] + fl["aum_sqqq"]), known=fl.index)
    # FINRA 공매도 거래량: 5일 평균, 창 안 모든 관측이 공개된 뒤에만
    sv = pd.read_parquet(C / "finra_shvol.parquet")
    v = sv["shvr_ai_eq"].to_numpy(float)
    own = [a if ok == 1 else k for a, k, ok in zip(sv["asof"], sv.index, sv["open_ok"])]
    v5 = np.array([v[i - 4:i + 1].mean() if i >= 4 else np.nan for i in range(len(v))])
    leg = [max(own[max(0, i - 4):i + 1]) for i in range(len(v))]
    obs["shvr_ai5"] = mk(sv["asof"], leg, v5, known=sv.index)
    # CFTC
    cf = pd.read_parquet(C / "cftc_tff.parquet")
    nq_lev = (cf["nq_lev_net"] / cf["nq_oi"]).to_numpy()
    nq_am = (cf["nq_am_net"] / cf["nq_oi"]).to_numpy()
    d4 = lambda x: np.array([x[i] - x[i - 4] if i >= 4 else np.nan for i in range(len(x))])  # noqa: E731
    ser = {"nq_lev_net_oi": nq_lev, "nq_lev_chg4": d4(nq_lev), "nq_am_net_oi": nq_am, "nq_am_chg4": d4(nq_am),
           "nq_dealer_net_oi": (cf["nq_dealer_net"] / cf["nq_oi"]).to_numpy(), "es_lev_net_oi": (cf["es_lev_net"] / cf["es_oi"]).to_numpy(),
           "vx_lev_net_oi": (cf["vx_lev_net"] / cf["vx_oi"]).to_numpy(), "nq_nonrept_net_oi": (cf["nq_nonrept_net"] / cf["nq_oi"]).to_numpy()}
    for s, x in ser.items():
        obs[s] = mk(cf["asof"], cf.index, x, known=cf.index)
    si = pd.read_parquet(C / "finra_si.parquet")
    obs["si_ai_chg"] = mk(si["asof"], si.index, si["si_ai_chg"], known=si.index)
    su = pd.read_parquet(C / "surveys.parquet")
    a = su.dropna(subset=["aaii_spread"])
    obs["aaii_spread"] = mk(a["aaii_asof"], a.index, a["aaii_spread"], known=a.index)
    n = su.dropna(subset=["naaim"])
    obs["naaim"] = mk(n["naaim_asof"], n.index, n["naaim"], known=n.index)
    return obs


def check_reconstruction(P: pd.DataFrame, obs: dict) -> dict:
    dd = list(P.index)
    out = {}
    for s in H.SIGNALS:
        o = obs[s].copy()
        o = o[~np.isnan(o["raw"])].reset_index(drop=True)
        win = WIN[s]
        if s == "ai_breadth5":
            o["value"] = o["raw"]  # 이미 250일 백분위(AIB_5)
        elif s == "ai_cpcr5":
            # 하네스: 1 − pct(−P/C) = P/C의 '엄격' 백분위, 창 250·최소 120(build.MINP)
            o["value"] = pct_np(o["raw"].to_numpy(), win, 120, strict=True)
            o["value_incl"] = pct_np(o["raw"].to_numpy(), win, 120, strict=False)
        else:
            o["value"] = pct_np(o["raw"].to_numpy(), win, max(win // 2, 3))
        val, meas, epos = place_indep(o, dd, AGE[s])
        hv = P[s].to_numpy()
        both = ~np.isnan(val) & ~np.isnan(hv)
        mism = int((np.abs(val[both] - hv[both]) > 1e-9).sum()) + int((np.isnan(val) != np.isnan(hv)).sum())
        y2526 = np.array([d >= "2025-01-01" for d in dd])
        # 측정일 나이(달력일) — 진입일과 측정일 차이
        age_cal = np.array([(pd.Timestamp(d) - pd.Timestamp(m)).days if m is not None else np.nan for d, m in zip(dd, meas)], float)
        age_td = np.array([p - e if e >= 0 else np.nan for p, e in zip(range(len(dd)), epos)], float)
        rec = {"mismatch_cells_all": mism, "mismatch_cells_2025_26": int(((np.abs(np.nan_to_num(val) - np.nan_to_num(hv)) > 1e-9) & y2526).sum()
                                                                         + ((np.isnan(val) != np.isnan(hv)) & y2526).sum()),
               "nonnull_2025": int((~np.isnan(hv) & np.array([d[:4] == "2025" for d in dd])).sum()),
               "nonnull_front": int((~np.isnan(hv) & H.period_mask(P.index, "front")).sum()),
               "nonnull_back": int((~np.isnan(hv) & H.period_mask(P.index, "back")).sum()),
               "max_age_tradingdays_since_entry": float(np.nanmax(age_td)) if np.isfinite(age_td).any() else None,
               "age_limit": AGE[s],
               "measure_age_calendar_days_2025_26": {"median": float(np.nanmedian(age_cal[y2526])) if np.isfinite(age_cal[y2526]).any() else None,
                                                     "max": float(np.nanmax(age_cal[y2526])) if np.isfinite(age_cal[y2526]).any() else None},
               "min_measure_age_calendar_days": float(np.nanmin(age_cal)) if np.isfinite(age_cal).any() else None,
               "measure_age_max_by_period": {per: (float(np.nanmax(age_cal[H.period_mask(P.index, per)]))
                                                   if np.isfinite(age_cal[H.period_mask(P.index, per)]).any() else None)
                                             for per in ("y2025", "front", "back")},
               "rows_measure_age_gt14d_by_period": {per: int((age_cal[H.period_mask(P.index, per)] > 14).sum())
                                                    for per in ("y2025", "front", "back")}}
        if s == "ai_cpcr5":
            vi, _, _ = place_indep(o.assign(value=o["value_incl"]), dd, AGE[s])
            m26 = y2526 & ~np.isnan(hv)
            flips = {}
            for th in (0.8, 0.7, 0.65):
                hi = int(((hv >= th) != (vi >= th))[m26].sum())
                lo = int(((hv <= 1 - th) != (vi <= 1 - th))[m26].sum())
                flips[f"{th}"] = {"high_side_rows": hi, "low_side_rows": lo}
            rec["strict_vs_inclusive_pct"] = {"max_diff": float(np.nanmax(np.abs(vi - hv))), "rows_2025_26": int(m26.sum()), "threshold_flips": flips}
            INCLUSIVE["ai_cpcr5"] = vi
        out[s] = rec
    return out


INCLUSIVE: dict[str, np.ndarray] = {}


# ---------- 날짜 표시 맞음(원자료 날짜가 하루 밀리면 같은 날 상관 정점이 ±1로 옮겨 간다) ----------
def check_date_alignment() -> dict:
    px = pd.read_parquet(H.OS / "cache/yahoo_ohlc.parquet")
    q, sx = px["QQQ_c"].pct_change() * 100, px["SOXX_c"].pct_change() * 100
    cb = osf.load_cboe()
    y = pd.read_parquet(H.OS / "cache/yahoo.parquet").dropna(subset=["QQQ"])
    sq = pd.read_parquet(C / "squeeze.parquet").set_index("asof")
    hp = pd.read_parquet(C / "hf_proxies.parquet")
    fl = hp.dropna(subset=["asof"]).set_index("asof")
    sv = pd.read_parquet(C / "finra_shvol.parquet").set_index("asof")
    by = pd.read_parquet(C / "hf_proxies_by_symbol.parquet")
    syms = set(osb.ai_us_symbols())
    cp, cc = {}, {}
    for f in sorted((H.OS / "cache/occ").glob("*.parquet")):
        d = pd.read_parquet(f)
        d = d[d["underlying"].isin(syms)]
        cp[f.stem], cc[f.stem] = d["C_P"].sum(), d["C_C"].sum()

    def lagcorr(x: pd.Series, r: pd.Series) -> dict:
        x = x.replace([np.inf, -np.inf], np.nan).dropna()
        x = x[(x.index >= "2025-01-01") & (x.index <= "2026-10-12")]
        o = {}
        for k, nm in ((-1, "prev"), (0, "same"), (1, "next")):
            rr = r.shift(-k).reindex(x.index)
            m = rr.notna()
            o[nm] = round(float(np.corrcoef(x[m], rr[m])[0, 1]), 3)
        o["peak"] = max(("prev", "same", "next"), key=lambda z: abs(o[z]))
        return o

    def vol(t):
        """log 거래량 − 직전 20일 평균(차분은 전날 |수익률|과 기계적으로 음의 상관이라 쓰지 않는다)."""
        d = by[by["symbol"] == t].set_index("known")
        lv = np.log(d["volume"])
        return lv - lv.shift(1).rolling(20).mean()

    tests = {
        "CBOE 주식 풋/콜(1일) vs QQQ 같은 날 수익률": (cb["equity_pv"] / cb["equity_cv"], q),
        "OCC AI 고객 풋/콜(1일) vs QQQ": (pd.Series(cp) / pd.Series(cc), q),
        "ΔVIX vs QQQ": (y["^VIX"].diff(), q),
        "Δ(VIX/VIX3M) vs QQQ": ((y["^VIX"] / y["^VIX3M"]).diff(), q),
        "ΔGEX vs QQQ": (sq["gex_bn"].diff(), q),
        "Squeeze spx 수익률 vs QQQ": (sq["spx"].pct_change() * 100, q),
        "ΔDIX vs ΔFINRA 공매도 비율(같은 재료)": (sq["dix"].diff(), sv["shvr_all"].diff()),
        "TQQQ−SQQQ 순흐름(asof) vs QQQ": (fl["lev_qqq_flow"], q),
        "TQQQ 거래량(20일 대비) vs |QQQ|": (vol("TQQQ"), q.abs()),
        "SQQQ 거래량(20일 대비) vs |QQQ|": (vol("SQQQ"), q.abs()),
        "SOXL 거래량(20일 대비) vs |SOXX|": (vol("SOXL"), sx.abs()),
        "SOXS 거래량(20일 대비) vs |SOXX|": (vol("SOXS"), sx.abs()),
        "GVIP 수익률 vs QQQ": (by[by["symbol"] == "GVIP"].set_index("known")["close"].pct_change() * 100, q),
        "QAI 수익률 vs QQQ": (by[by["symbol"] == "QAI"].set_index("known")["close"].pct_change() * 100, q),
        "참고: DIX 수준 vs QQQ": (sq["dix"], q),
        "참고: FINRA AI 공매도 비율 vs QQQ": (sv["shvr_ai_eq"], q),
        "참고: TQQQ 거래대금 몫(1일) vs QQQ": (hp["lev_qqq_bull"], q),
        "참고: SOXL 거래대금 몫(1일) vs SOXX": (hp["lev_soxx_bull"], sx),
    }
    out = {k: lagcorr(x, r) for k, (x, r) in tests.items()}
    core = [k for k in out if not k.startswith("참고")]
    return {"pass": all(out[k]["peak"] == "same" for k in core), "tests": out,
            "note": "'참고' 줄은 날짜 표시가 아니라 시장 반응(전날 수익률에 반응하는 공매도·다크풀, 다음 날과의 약한 상관)이다. "
                    "같은 원자료의 가격·거래량·FINRA 대조 줄이 모두 같은 날 정점이라 표시 밀림은 없다."}


def cell_status(P: pd.DataFrame, cells: list[tuple[str, int, str, str]]) -> list[dict]:
    """고르기와 상관없이 칸 하나의 front·back 차이와 back 순열 p(가설 방향)."""
    idx = P.index
    Y = H.outcomes("QQQ").reindex(idx)
    S, names = G.cells(P)
    out = []
    for cell, h, y, side in cells:
        sig = S[:, names.index(cell)]
        yy = Y[f"{y}{h}"].to_numpy()
        rec = {"cell": f"{cell} h{h} {y}"}
        for per in ("front", "back", "y2025"):
            m = H.period_mask(idx, per)
            d, t, n = H.welch(np.where(m, sig, np.nan), np.where(m, yy, np.nan))
            rec[per] = {"d": None if np.isnan(d) else round(d, 2), "t": None if np.isnan(t) else round(t, 1), "n": n}
        m = H.period_mask(idx, "back")
        rec["back_p"] = H.shift_p(np.where(m, sig, np.nan), np.where(m, yy, np.nan), "less" if side == "top" else "greater")
        out.append(rec)
    return out


# ---------- E: 끝에서 끝 교란 ----------
PROTECT = {"open_ok", "post_lag_h", "n_ai", "n_sym", "naaim_known_src", "si_ai_n"}


def _perturb_frame(df: pd.DataFrame, rowmask: np.ndarray, rng, cols=None) -> pd.DataFrame:
    df = df.copy()
    cols = [c for c in (cols if cols is not None else df.columns)
            if c not in PROTECT and pd.api.types.is_numeric_dtype(df[c])]
    if not rowmask.any() or not cols:
        return df
    for c in cols:
        x = df[c].to_numpy(float).copy()
        x[rowmask] = x[rowmask] * rng.lognormal(0.0, 1.0, rowmask.sum()) * rng.choice([-1, 1], rowmask.sum())
        df[c] = x
    return df


class Perturb:
    """기준일 cut 시가에 아직 쓸 수 없는(정당한 첫 진입 > cut) 원자료만 바꾼다. 규칙은 이 파일 EVIDENCE·raw_observations와 같다."""

    def __init__(self, cut: str, seed: int):
        self.cut, self.rng = cut, np.random.default_rng(seed)
        self.dd = list(H.days())
        self._rp = pd.read_parquet
        self._lc = osf.load_cboe

    def unusable(self, after_days) -> np.ndarray:
        """'공개 기준일'(이 날 다음 거래일부터 사용) 배열 → 그 다음 거래일 > cut 인가."""
        cutpos = self.dd.index(self.cut)
        return np.array([td_next(self.dd, x) > cutpos if isinstance(x, str) else False for x in after_days])

    def read_parquet(self, path, *a, **k):
        df = self._rp(path, *a, **k)
        p = Path(str(path))
        name = p.name
        if p.parent.name == "occ":
            mask = np.full(len(df), bool(self.unusable([p.stem])[0]))
            return _perturb_frame(df, mask, self.rng, ["C_C", "C_P", "F_C", "F_P", "M_C", "M_P"])
        if name == "yahoo.parquet":
            return _perturb_frame(df, self.unusable(list(df.index)), self.rng)
        if name == "squeeze.parquet":
            return _perturb_frame(df, self.unusable(list(df["asof"])), self.rng)
        if name in ("cftc_tff.parquet", "finra_si.parquet", "surveys.parquet"):
            return _perturb_frame(df, self.unusable(list(df.index)), self.rng)
        if name == "finra_shvol.parquet":
            when = [a_ if ok == 1 else k_ for a_, k_, ok in zip(df["asof"], df.index, df["open_ok"])]
            return _perturb_frame(df, self.unusable(when), self.rng)
        if name == "hf_proxies.parquet":
            flow = [c for c in df.columns if c.startswith(("so_", "flow_", "aum_")) or "flow" in c or c == "lev_qqq_aum_bull"]
            price = [c for c in df.columns if c not in flow and c != "asof"]
            df = _perturb_frame(df, self.unusable(list(df.index)), self.rng, price)
            return _perturb_frame(df, self.unusable(list(df["asof"])), self.rng, flow)
        return df

    def load_cboe(self):
        df = self._lc()
        return _perturb_frame(df, self.unusable(list(df.index)), self.rng)

    def __enter__(self):
        pd.read_parquet = self.read_parquet
        osf.load_cboe = self.load_cboe
        return self

    def __exit__(self, *e):
        pd.read_parquet = self._rp
        osf.load_cboe = self._lc


def check_perturbation(P0: pd.DataFrame, cuts: list[str]) -> dict:
    rows, fails = [], 0
    for j, cut in enumerate(cuts):
        with Perturb(cut, 1000 + j):
            Pc = H.build_panel()
        before = P0.index <= cut
        a, b = P0.to_numpy(), Pc.to_numpy()
        diff = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        n_before = int(diff[before].sum())
        cols_before = sorted(P0.columns[diff[before].any(axis=0)])
        nxt = (P0.index > cut) & (P0.index <= P0.index[min(len(P0) - 1, P0.index.get_loc(cut) + 10)])
        n_after = int(diff[nxt].sum())
        cols_after = int(diff[nxt].any(axis=0).sum())
        fails += n_before > 0
        rows.append({"cut": cut, "changed_cells_rows_le_cut": n_before, "cols_changed_le_cut": cols_before,
                     "changed_cells_next10": n_after, "signals_changed_next10": cols_after})
    return {"pass": fails == 0, "n_cuts": len(cuts), "cuts_with_leak": fails, "rows": rows}


# ---------- F: 방향 ----------
def check_orientation(P: pd.DataFrame, obs: dict) -> dict:
    dd = list(P.index)
    res = {}
    for s in H.SIGNALS:
        o = obs[s].dropna(subset=["raw"])
        o = o[o["entry_pos"] < len(dd)]
        # 진입일의 원값과 패널 값의 순위 상관(250일 창 안 순위라 1은 아니지만 양수여야 한다)
        ent = o.drop_duplicates("entry_pos", keep="last")
        pv = P[s].to_numpy()[ent["entry_pos"].to_numpy()]
        ok = ~np.isnan(pv)
        rho = pd.Series(ent["raw"].to_numpy()[ok]).rank().corr(pd.Series(pv[ok]).rank()) if ok.sum() > 10 else np.nan
        res[s] = {"spearman_raw_vs_panel_at_entry": None if np.isnan(rho) else round(float(rho), 3), "n": int(ok.sum())}
    m = P[["ai_cpcr5", "ai_breadth5", "eq_pcr5"]].dropna()
    res["_cross"] = {"corr_ai_cpcr5_vs_ai_breadth5": round(float(m["ai_cpcr5"].corr(m["ai_breadth5"], method="spearman")), 3),
                     "corr_ai_cpcr5_vs_eq_pcr5": round(float(m["ai_cpcr5"].corr(m["eq_pcr5"], method="spearman")), 3), "n": len(m)}
    # ai_breadth5 원값을 독립 계산: 종목별 5일 고객 풋/콜이 자기 250 관측 중 하위 20%(= −P/C 백분위 ≥ 0.8)인 종목 비율
    br = raw_breadth()
    b = obs["ai_breadth5"].set_index("meas")["raw"]
    j = pd.concat([br.rename("mine"), b.rename("AIB_5")], axis=1).dropna()
    res["ai_breadth5_raw"] = {"spearman_my_raw_breadth_vs_AIB_5_same_day": round(float(j["mine"].corr(j["AIB_5"], method="spearman")), 3),
                              "n": len(j), "meaning": "높음 = 콜 쏠림(풋/콜 낮음) 종목이 많음 = 환호"}
    res["pass"] = all((v["spearman_raw_vs_panel_at_entry"] or 0) > 0 for k, v in res.items() if k in H.SIGNALS and k != "ai_breadth5") \
        and res["_cross"]["corr_ai_cpcr5_vs_ai_breadth5"] < 0 and res["ai_breadth5_raw"]["spearman_my_raw_breadth_vs_AIB_5_same_day"] > 0.5
    return res


def raw_breadth(W: int = 5) -> pd.Series:
    syms = sorted(set(osb.ai_us_symbols()))
    days = list(osf.load_cboe().index)
    files = {f.stem: f for f in (H.OS / "cache/occ").glob("*.parquet")}
    have = np.array([d in files for d in days])
    cc = np.zeros((len(days), len(syms)))
    cp = np.zeros((len(days), len(syms)))
    col = {s: k for k, s in enumerate(syms)}
    for i, d in enumerate(days):
        if d in files:
            x = pd.read_parquet(files[d])
            x = x[x["underlying"].isin(col)]
            for u, c_, p_ in zip(x["underlying"], x["C_C"], x["C_P"]):
                cc[i, col[u]] += c_
                cp[i, col[u]] += p_
    cc[~have], cp[~have] = np.nan, np.nan
    n = len(days)
    sc = np.full((n, len(syms)), np.nan)
    sp = np.full((n, len(syms)), np.nan)
    for i in range(W - 1, n):
        sc[i], sp[i] = cc[i - W + 1:i + 1].sum(0), cp[i - W + 1:i + 1].sum(0)
    ok = (sc + sp) / W >= 1000
    with np.errstate(invalid="ignore", divide="ignore"):
        pcr = np.where(ok & (sc > 0), sp / sc, np.nan)
    fr = np.zeros_like(ok)
    for k in range(len(syms)):
        v = pcr[:, k]
        pos = np.where(~np.isnan(v))[0]
        p = pct_np(-v[pos], 250, 120)
        fr[pos, k] = p >= 0.8
    nok = ok.sum(1)
    br = np.where(nok >= 20, (fr & ok).sum(1) / np.maximum(nok, 1), np.nan)
    return pd.Series(br, index=days)


# ---------- H: 추적 예시 ----------
def traces(P: pd.DataFrame, obs: dict) -> list[dict]:
    dd = list(P.index)
    out = []

    def show(sig, meas, label, extra=None):
        o = obs[sig]
        r = o[o["meas"] == meas]
        if r.empty:
            out.append({"signal": sig, "meas": meas, "note": "관측 없음"})
            return
        r = r.iloc[-1]
        e = int(r["entry_pos"])
        rec = {"family": label, "signal": sig, "measured": meas, "known_QA": r["known"], "public_basis": r["legit_after"],
               "entry_day": dd[e] if e < len(dd) else None,
               "panel_prev_day": (dd[e - 1], None if np.isnan(P[sig].iloc[e - 1]) else round(float(P[sig].iloc[e - 1]), 4)),
               "panel_entry_day": (dd[e], None if np.isnan(P[sig].iloc[e]) else round(float(P[sig].iloc[e]), 4))}
        if extra:
            rec.update(extra)
        out.append(rec)

    sv = pd.read_parquet(C / "finra_shvol.parquet").set_index("asof")
    show("nq_lev_net_oi", "2026-03-10", "CFTC 평주(화→금 15:30→월)")
    show("nq_lev_net_oi", "2026-07-14", "CFTC 16시 뒤 공개 주(known 금→월로 고침)")
    show("nq_lev_net_oi", "2025-12-30", "CFTC 셧다운 마지막 밀린 주")
    show("lev_qqq_flow20p", "2026-03-16", "ProShares 흐름")
    show("shvr_ai5", "2026-03-16", "FINRA 공매도 거래량", {"open_ok": float(sv.loc["2026-03-16", "open_ok"]),
                                                       "post_lag_h(16시 뒤 시간)": round(float(sv.loc["2026-03-16", "post_lag_h"]), 2)})
    show("dix", "2026-03-16", "SqueezeMetrics DIX")
    show("gvip_rel20", "2026-03-16", "Yahoo 가격 지표")
    show("lev_qqq_bull5", "2026-03-16", "Yahoo 레버리지 거래대금")
    show("eq_pcr5", "2026-03-16", "CBOE")
    show("ai_cpcr5", "2026-03-16", "OCC")
    show("aaii_spread", "2026-03-11", "AAII(수 마감→목 아침 공개)")
    show("naaim", "2026-04-29", "NAAIM(금 17:07 업로드 → 월)")
    show("naaim", "2026-07-29", "NAAIM 마지막 무료 주")
    show("si_ai_chg", "2026-03-31", "FINRA 공매도 잔고")
    show("skew", "2026-03-16", "CBOE SKEW 지수")
    # NAAIM 끝: 마지막 값이 쓰인 날
    nn = P["naaim"].dropna()
    out.append({"family": "NAAIM 끝", "last_nonnull_panel_day": nn.index.max(), "rows_back_nonnull": int(nn[nn.index >= "2026-06-01"].size)})
    return out


# ---------- I: 격자 민감도 ----------
def run_grid(P: pd.DataFrame, purge_front: bool = False, n_null: int = G.N_NULL) -> dict:
    idx = P.index
    Y = H.outcomes("QQQ").reindex(idx)
    S, names = G.cells(P)
    re, sd = G.composite(P, G.RETAIL_EUPH), G.composite(P, G.SMART_DEF)
    t1 = ((re >= G.TH2) & (sd >= G.TH2)).astype(float).where(re.notna() & sd.notna())
    t2 = ((re <= 1 - G.TH2) & (sd <= 1 - G.TH2)).astype(float).where(re.notna() & sd.notna())
    S = np.column_stack([S, t1.to_numpy(), t2.to_numpy()])
    names = names + ["T1_개미환호&큰손방어", "T2_개미공포&큰손매수"]
    y26 = np.asarray(idx >= "2026-01-02")
    S26, Y26, i26 = S[y26], Y[y26].reset_index(drop=True), idx[y26]
    fr, bk = H.period_mask(i26, "front"), H.period_mask(i26, "back")
    if purge_front:
        sel = select_confirm_purged
    else:
        sel = G.select_confirm
    obs = sel(S26, Y26, fr, bk, names, detail=True)
    rng = np.random.default_rng(G.SEED)
    null = {"top": [], "bottom": []}
    for _ in range(n_null):
        k = int(rng.integers(H.MIN_SHIFT, len(i26) - H.MIN_SHIFT))
        r = sel(np.roll(S26, k, axis=0), Y26, fr, bk, names)
        null["top"].append(r["top"])
        null["bottom"].append(r["bottom"])
    res = {}
    for side in ("top", "bottom"):
        c, rows = obs[side]
        nn = np.array(null[side])
        res[side] = {"confirmed": int(c), "null_mean": float(nn.mean()), "null_p95": float(np.percentile(nn, 95)),
                     "p": float(((nn >= c).sum() + 1) / (len(nn) + 1)),
                     "confirmed_cells": [f"{r['cell']} h{r['h']} {r['y']}" for r in rows if r["confirmed"]],
                     "picks_top5": [f"{r['cell']} h{r['h']} {r['y']} t{r['front_t']:+.1f}" for r in rows[:5]]}
    res["T1_T2_days_2026"] = {"T1": int(np.nansum(t1.to_numpy()[y26])), "T2": int(np.nansum(t2.to_numpy()[y26]))}
    return res


def select_confirm_purged(S26, Y26, fr, bk, names, detail=False):
    """grid.select_confirm과 같되, front 고르기에서 결과 창(d … d+h−1)이 back으로 넘어가는 front 날을 뺀다(경계 겹침 제거)."""
    last_fr = np.where(fr)[0].max()
    Yp = Y26.copy()
    for h in H.HORIZONS:
        cut = np.arange(len(Yp)) + h - 1 > last_fr
        for y in ("R", "M"):
            col = Yp[f"{y}{h}"].to_numpy().copy()
            col[cut & fr] = np.nan
            Yp[f"{y}{h}"] = col
    recs = []
    for h in H.HORIZONS:
        for y in ("R", "M"):
            d, t, n = H.grid_diffs(S26, Yp[f"{y}{h}"].to_numpy(), fr)
            recs.append(pd.DataFrame({"k": np.arange(len(names)), "h": h, "y": y, "t": t, "n": n}))
    g = pd.concat(recs, ignore_index=True)
    g = g[(g["n"] >= G.MIN_FRONT) & g["t"].notna()]
    out = {}
    for side in ("top", "bottom"):
        sub = g if side == "top" else g[g["y"] == "R"]
        pick = sub.nsmallest(G.TOPK, "t") if side == "top" else sub.nlargest(G.TOPK, "t")
        conf, rows = 0, []
        for _, r in pick.iterrows():
            yy = Y26[f"{r['y']}{int(r['h'])}"].to_numpy()
            sig = S26[:, int(r["k"])]
            bs, by = np.where(bk, sig, np.nan), np.where(bk, yy, np.nan)
            d, _, nb = H.welch(bs, by)
            want = (d < 0) if side == "top" else (d > 0)
            p = H.shift_p(bs, by, "less" if side == "top" else "greater") if (nb >= 3 and want) else np.nan
            ok = bool(want and p < 0.05)
            conf += ok
            if detail:
                fd, ft, fn = H.welch(np.where(fr, sig, np.nan), np.where(fr, Yp[f"{r['y']}{int(r['h'])}"].to_numpy(), np.nan))
                rows.append({"cell": names[int(r["k"])], "h": int(r["h"]), "y": r["y"], "front_d": fd, "front_t": ft, "front_n": fn,
                             "back_d": d, "back_n": nb, "back_p": p, "confirmed": ok})
        out[side] = (conf, rows) if detail else conf
    return out


def shifted(P: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """진입을 하루 늦춘 패널(공개 기준일을 QA known으로 = 측정일 다음 거래일 장 마감 뒤에야 안다고 본다). _place가 위치 단위라 열 shift(1)과 같다."""
    Q = P.copy()
    for c in cols:
        Q[c] = P[c].shift(1)
    return Q


def main() -> None:
    t0 = time.time()
    res = {"branch": "timing_audit", "date": "2026-10-14"}
    res["A_rebuild"] = check_rebuild()
    P = pd.read_parquet(C / "panel.parquet")
    res["B_calendar"] = check_calendar()
    res["C_outcomes"] = check_outcomes()
    obs = raw_observations()
    rec = check_reconstruction(P, obs)
    res["D_reconstruction"] = {"pass": all(v["mismatch_cells_all"] == 0 for v in rec.values()), "signals": rec}
    res["evidence"] = {s: {"measured": EVIDENCE[s][0], "public": EVIDENCE[s][1], "entry_rule": EVIDENCE[s][2]} for s in H.SIGNALS}
    # 교란 시험 기준일: 2025~2026 무작위 24일 + 경계(CFTC 금요일·16시 뒤 공개 금요일, AAII 목요일, NAAIM 금 17시 업로드, 셧다운 공개일)
    dd = list(P.index)
    rng = np.random.default_rng(5)
    pool = [d for d in dd if d >= "2025-01-02"]
    cuts = sorted(set(list(rng.choice(pool, 24, replace=False)) +
                      ["2025-07-18", "2026-07-17", "2026-03-13", "2026-03-12", "2026-05-01", "2025-12-31", "2026-01-05",
                       "2026-04-30", "2026-07-30", "2026-03-16", "2026-10-09", "2026-10-12"]))
    res["E_perturbation"] = check_perturbation(P, cuts)
    res["F_orientation"] = check_orientation(P, obs)
    res["F2_date_alignment"] = check_date_alignment()
    res["H_traces"] = traces(P, obs)
    res["vintage_risk"] = {
        "CBOE·OCC": "몇 달 뒤 한꺼번에 받은 값. 당시 첫 게시값과 같은지 확인 불가",
        "SqueezeMetrics": "웨이백 사본 24개 = 지금 파일 앞부분(2025-04-29, 2026-09-09~10-13). 2025-04-30~2026-09-08 사이 소급 수정 여부는 모름",
        "ProShares": "매일 전체 이력을 다시 쓰는 CSV. 소급 수정 확인 불가(분할 소급 외 흔적 없음)",
        "Yahoo": "수정종가는 배당 배율 — 250일 순위·20일 변화엔 앞당김 없음(QA). 거래량은 사후 확정값",
        "FINRA 공매도 거래량": "파일별 Last-Modified로 수정본 7건 식별, 늦은 3건 제외",
        "CFTC": "API 최신 수정본, 2024년 이후 수정 행 없음(QA)",
        "FINRA 공매도 잔고": "current는 첫 공개값, 정정은 다음 기준일 prev로만(QA 검증)",
        "AAII·NAAIM": "AAII 현재 파일 하나(수정 확인 불가), NAAIM 판 비교 수정 0건",
        "AI 종목 묶음": "2026 워치리스트를 과거에 적용(ai_cpcr5·ai_breadth5·shvr_ai5·si_ai_chg) — 시점이 아닌 사후 선택 편향, 크기 미측정"}
    # 격자 민감도
    weak = ["eq_pcr5", "ai_cpcr5", "ai_breadth5", "gex", "dix", "lev_qqq_flow20p"]
    alld = weak + ["skew", "vix_ts", "lev_qqq_bull5", "lev_soxx_bull5", "gvip_rel20", "qai_beta60", "shvr_ai5"]
    P_incl = P.copy()
    P_incl["ai_cpcr5"] = INCLUSIVE["ai_cpcr5"]
    variants = {"baseline_recomputed": (P, False, "없음"),
                "shift1_weak_evidence": (shifted(P, weak), False, weak),
                "shift1_proshares_only": (shifted(P, ["lev_qqq_flow20p"]), False, ["lev_qqq_flow20p"]),
                "shift1_all_daily": (shifted(P, alld), False, alld),
                "ai_cpcr5_inclusive_pct": (P_incl, False, "ai_cpcr5를 포함 백분위(w ≤ x)로"),
                "purge_front_boundary": (P, True, "front 고르기에서 결과 창이 6월로 넘어가는 날 제외")}
    conf_cells = [("eq_pcr5+&shvr_ai5-", 21, "M", "top"), ("skew+&shvr_ai5-", 21, "M", "top"),
                  ("skew+&nq_am_chg4-", 21, "M", "top"), ("es_lev_net_oi+", 21, "R", "bottom")]
    res["I_grid"] = {}
    for k, (Pv, purge, what) in variants.items():
        res["I_grid"][k] = {"change": what, **run_grid(Pv, purge_front=purge), "grid_json_confirmed_cells_status": cell_status(Pv, conf_cells)}
    g0 = json.loads((H.RES / "grid.json").read_text(encoding="utf-8"))
    res["I_grid"]["grid_json"] = {s: g0["null"][s] for s in ("top", "bottom")}
    res["secs"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    # 요약 출력
    print("A", res["A_rebuild"])
    print("B pass", res["B_calendar"]["pass"], res["B_calendar"]["missing_trading_days"], res["B_calendar"]["extra_non_trading"],
          res["B_calendar"]["last_session"]["dd_last"])
    print("C pass", res["C_outcomes"]["pass"], res["C_outcomes"]["trace"])
    print("D pass", res["D_reconstruction"]["pass"])
    for s, v in rec.items():
        print(f"  {s:18s} mism {v['mismatch_cells_all']:3d}  max_td_age {v['max_age_tradingdays_since_entry']} (lim {v['age_limit']})  "
              f"meas_age med/max {v['measure_age_calendar_days_2025_26']}  min_meas_age {v['min_measure_age_calendar_days']}  "
              f"nonnull 25/fr/bk {v['nonnull_2025']}/{v['nonnull_front']}/{v['nonnull_back']}")
    print("  ai_cpcr5 strict vs incl", rec["ai_cpcr5"].get("strict_vs_inclusive_pct"))
    print("E pass", res["E_perturbation"]["pass"], "cuts", res["E_perturbation"]["n_cuts"], "leaks", res["E_perturbation"]["cuts_with_leak"])
    for r in res["E_perturbation"]["rows"]:
        print("  ", r)
    print("F", res["F_orientation"]["pass"], res["F_orientation"]["_cross"], res["F_orientation"]["ai_breadth5_raw"])
    print("F2 pass", res["F2_date_alignment"]["pass"])
    for k, v in res["F2_date_alignment"]["tests"].items():
        print("  ", k, v)
    for s in ("nq_lev_net_oi", "naaim", "si_ai_chg", "aaii_spread"):
        print("  age", s, rec[s]["measure_age_max_by_period"], rec[s]["rows_measure_age_gt14d_by_period"])
    print({k: v["spearman_raw_vs_panel_at_entry"] for k, v in res["F_orientation"].items() if k in H.SIGNALS})
    for t in res["H_traces"]:
        print("H", t)
    for k, v in res["I_grid"].items():
        print("I", k, json.dumps(v, ensure_ascii=False, default=str)[:900])
    print("secs", res["secs"])


if __name__ == "__main__":
    main()
