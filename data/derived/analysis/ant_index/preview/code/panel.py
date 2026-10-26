"""미리보기(부록 9) 3단계 ②: 창별 M0(지난 값)·결과(앞날) 표.

panel_kr.parquet: W_KR(D) 본판 완전 창 한 줄. 진입 = D KRX 시가(09:00, 10:00 개장일도 D 시가).
panel_us.parquet: W_U(D) 본판 완전 창 한 줄. 진입 = D 미국 정규장 시가.
열 이름: m0_* = 창 마감 전에 알 수 있는 값(§2.1 가격 마감 규칙)과 진입 순간 값 g(D). fut_* = 진입 시가 뒤 값(결과).
포럼 특징은 이 표에 넣지 않는다(win_id로 잇는다). 특징 계산에는 fut_*를 쓰지 않는다.

정의(동결 §2.0·§2.3, 해석은 interim/notes_h01_preview.md를 따른다)
  r(P) 종가 대 종가, r5·r20·r60 = P 종가 ÷ 5·20·60세션 앞 종가 − 1, DD20 = P 종가 ÷ P까지 20세션 최고 종가 − 1,
  lnRV n = P까지 n개 일 로그수익 SD(ddof 1)의 로그, 거래량 이상치 = ln(P 거래량 ÷ 앞 20세션 평균, 15세션 이상)
  (yfinance에 원화 거래대금이 없어 거래량으로 대신), g(D) = D 시가 ÷ 직전 종가 − 1.
  국내 창의 미국 세션 u = D−1일 이하 마지막 NYSE 세션. 미국 창의 국내 세션 k = D일(KST) 이하 마지막 KRX 세션.
  미국 창 P = 직전 NYSE 세션. NQ = 바 끝 ≤ D 08:30 ET 마지막 60분봉 종가 ÷ 바 끝 ≤ P 16:00 ET 마지막 60분봉 종가 − 1
  (창 마감 08:29 ET보다 이르거나 같은 값만 쓴다. 08:00~09:00 봉은 마감을 넘으므로 쓰지 않는다).
  Rh = 진입 시가 → 진입일+h−1 종가. MDDh = 경로 [진입 시가, D 종가, …, D+h−1 종가]의 최대 낙폭(≤ 0).
  MDDloh(보조) = min(MDDh, 각 날 저가 ÷ 그날 앞까지의 최고(진입 시가·앞 종가) − 1). HL = ln(D 고가 ÷ D 저가).
  경로 중간의 불량 봉은 건너뛴다(진입 시가와 끝 종가가 없으면 결측).
  불량 봉: 지수의 시가=고가=저가=종가이면서 거래량 0인 봉(^KS11·^KQ11 2026-06-29)은 네 값을 결측으로 둔다.
  이동 창 통계는 결측 봉 하나를 견디게 최소 개수를 n−2로 둔다(DD20·RV20은 18, RV22는 20, RV5는 4).
  VKOSPI는 받을 수 없어 넣지 않는다. CPI·고용·MSCI 일정 더미는 원천이 없어 넣지 않는다(h01과 같음).
일정 더미: KOSPI 옵션 만기(둘째 목요일), 미국 월물 만기(셋째 금요일), FOMC(h01_preview.FOMC_2026), 실적일
  (yfinance get_earnings_dates 날짜, NVDA·MU·AVGO·하닉·삼전). 일정일 e ∈ [P, D−1]이면 1(h01 규칙).

실행: py panel.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PREVIEW = HERE.parent
ANT = PREVIEW.parent
RES = PREVIEW / "results"
sys.path.insert(0, str(ANT / "interim" / "code"))
import h01_preview as H  # noqa: E402  (고치지 않고 일정 함수·FOMC 목록만 빌린다)
import loader as LD  # noqa: E402

KR_ASSETS = {"kospi": "^KS11", "skh": "000660.KS", "sec": "005930.KS", "kosdaq": "^KQ11"}
US_ASSETS = {"soxx": "SOXX", "sox": "^SOX", "qqq": "QQQ", "ewy": "EWY"}
HORIZONS = (1, 5, 10, 21)
MDD_H = (5, 10, 21)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- 가격
def load_prices() -> tuple[dict[str, pd.DataFrame], list[dict]]:
    p = pd.read_parquet(RES / "prices.parquet")
    p["date"] = pd.to_datetime(p["date"]).dt.date
    bad = p["flag_flat_bar"] & p["flag_zero_volume"]
    bad_rows = [{"ticker": t, "date": str(d)} for t, d in zip(p.loc[bad, "ticker"], p.loc[bad, "date"])]
    p.loc[bad, ["open", "high", "low", "close"]] = np.nan
    out = {t: g.set_index("date").sort_index() for t, g in p.groupby("ticker")}
    return out, bad_rows


def sess_feat(df: pd.DataFrame) -> pd.DataFrame:
    """세션 d의 값: d 종가까지(뒤로만 보는 창). g·HL은 그 세션의 시가·고가·저가를 쓴다(진입 순간·결과용)."""
    c, o, h, l, v = (df[k].astype(float) for k in ("close", "open", "high", "low", "volume"))
    r = c / c.shift(1) - 1
    lr = np.log(c / c.shift(1))
    out = pd.DataFrame(index=df.index)
    out["r"] = r
    out["rpos"] = r.clip(lower=0)
    out["rneg"] = r.clip(upper=0)
    out["absr"] = r.abs()
    out["ln_absr"] = np.log(r.abs().where(r.abs() > 0))
    for k in (5, 20, 60):
        out[f"r{k}"] = c / c.shift(k) - 1
    out["DD20"] = c / c.rolling(20, min_periods=18).max() - 1
    out["lnRV5"] = np.log(lr.rolling(5, min_periods=4).std(ddof=1))
    out["lnRV20"] = np.log(lr.rolling(20, min_periods=18).std(ddof=1))
    out["lnRV22"] = np.log(lr.rolling(22, min_periods=20).std(ddof=1))
    vv = v.where(v > 0)
    out["ln_vol_anom"] = np.log(vv / vv.shift(1).rolling(20, min_periods=15).mean())
    out["g"] = o / c.shift(1) - 1
    out["oc"] = c / o - 1
    out["close"] = c
    return out


def fut_block(df: pd.DataFrame, D: date, prefix: str) -> dict:
    """진입 = D 시가. Rh, MDDh(종가 경로), MDDloh(저가 보조), HL(D), 끝 세션 날짜."""
    idx = list(df.index)
    out = {}
    if D not in df.index:
        for hh in HORIZONS:
            out[f"fut_{prefix}_R{hh}"] = np.nan
        for hh in MDD_H:
            out[f"fut_{prefix}_MDD{hh}"] = np.nan
            out[f"fut_{prefix}_MDDlo{hh}"] = np.nan
        out[f"fut_{prefix}_HL"] = np.nan
        out[f"fut_{prefix}_lnHL"] = np.nan
        return out
    i = idx.index(D)
    o = df["open"].to_numpy(float)
    c = df["close"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    for hh in HORIZONS:
        e = i + hh - 1
        out[f"fut_{prefix}_R{hh}"] = (c[e] / o[i] - 1) if e < len(idx) and np.isfinite(c[e]) and np.isfinite(o[i]) \
            and o[i] > 0 else np.nan
    for hh in MDD_H:
        e = i + hh - 1
        if e >= len(idx):
            out[f"fut_{prefix}_MDD{hh}"] = out[f"fut_{prefix}_MDDlo{hh}"] = np.nan
            continue
        path = np.concatenate([[o[i]], c[i:e + 1]])
        if not (np.isfinite(path[0]) and np.isfinite(path[-1])):   # 진입 시가·끝 종가가 없으면 결측
            out[f"fut_{prefix}_MDD{hh}"] = out[f"fut_{prefix}_MDDlo{hh}"] = np.nan
            continue
        # 경로 중간의 불량 봉(결측)은 건너뛴다: 앞 최고는 유한한 점까지로 잇는다
        peak = np.fmax.accumulate(np.where(np.isfinite(path), path, -np.inf))
        ok = np.isfinite(path)
        mdd = float(np.min(path[ok] / peak[ok] - 1))
        lows = l[i:e + 1]
        prior_peak = peak[:-1]                # 그날 앞까지의 최고(진입 시가와 앞 종가)
        lok = np.isfinite(lows)
        mlo = min(mdd, float(np.min(lows[lok] / prior_peak[lok] - 1))) if lok.any() else mdd
        out[f"fut_{prefix}_MDD{hh}"] = min(mdd, 0.0)
        out[f"fut_{prefix}_MDDlo{hh}"] = min(mlo, 0.0) if np.isfinite(mlo) else np.nan
    hl = np.log(h[i] / l[i]) if np.isfinite(h[i]) and np.isfinite(l[i]) and l[i] > 0 else np.nan
    out[f"fut_{prefix}_HL"] = hl
    out[f"fut_{prefix}_lnHL"] = np.log(hl) if np.isfinite(hl) and hl > 0 else np.nan
    return out


def end_dates(sessions: list[date], D: date, tag: str) -> dict:
    out = {}
    if D not in sessions:
        return {f"fut_end_{tag}{hh}": pd.NaT for hh in HORIZONS}
    i = sessions.index(D)
    for hh in HORIZONS:
        e = i + hh - 1
        out[f"fut_end_{tag}{hh}"] = pd.Timestamp(sessions[e]) if e < len(sessions) else pd.NaT
    return out


def last_le(sessions: list[date], d: date) -> date | None:
    c = [s for s in sessions if s <= d]
    return max(c) if c else None


def nq_at(nq: pd.DataFrame, t_et: datetime) -> tuple[float, str | None]:
    m = nq[nq["bar_end_et"] <= pd.Timestamp(t_et)]
    if not len(m):
        return np.nan, None
    r = m.iloc[-1]
    return float(r["close"]), str(r["bar_end_et"])


# ---------------------------------------------------------------- 일정
def schedule(cal: pd.DataFrame, D_all: list[date], P_all: list[date]) -> dict:
    kr_cal = {d.date() for d in cal.loc[cal["krx"], "date"]}
    us_cal = {d.date() for d in cal.loc[cal["nyse"], "date"]}
    cal_last = pd.Timestamp(cal["date"].max()).date()
    months = sorted({(d.year, d.month) for d in P_all + D_all})
    kr_opex = [H.to_prev_session(H.nth_weekday(y, mo, 3, 2), kr_cal, cal_last) for y, mo in months]
    us_opex = [H.to_prev_session(H.nth_weekday(y, mo, 4, 3), us_cal, cal_last) for y, mo in months]
    earn = H.fetch_earnings()
    earn_days = sorted({d for v in earn.values() if isinstance(v, list) for d in v})
    return {"kr_opex": kr_opex, "us_opex": us_opex, "fomc": list(H.FOMC_2026), "earn_days": earn_days,
            "earn_raw": {k: ([str(x) for x in v] if isinstance(v, list) else v) for k, v in earn.items()}}


def cal_cols(P: date, D: date, kind: str, xh: bool, sch: dict, kr: bool) -> dict:
    out = {"m0_cal_weekend": float(kind == "weekend"), "m0_cal_holiday": float(kind == "holiday"),
           "m0_cal_xholiday": float(bool(xh)),
           "m0_cal_us_opex": float(any(H.in_window(e, P, D) for e in sch["us_opex"])),
           "m0_cal_fomc": float(any(H.in_window(e, P, D) for e in sch["fomc"])),
           "m0_cal_earn": float(any(H.in_window(e, P, D) for e in sch["earn_days"]))}
    if kr:
        out["m0_cal_vacation"] = float(D.month in (1, 2, 7, 8))
        out["m0_cal_kr_opex"] = float(any(H.in_window(e, P, D) for e in sch["kr_opex"]))
    return out


# ---------------------------------------------------------------- 표
WIN_COLS = ["win_id", "D", "P", "weekday", "open_kst", "close_kst", "length_h", "kind", "main_kind", "over96",
            "xholiday_mismatch", "dst_week", "early_close_D", "missing_rate", "d_no", "d_no_per_h", "n_posts",
            "n_comments", "n_authors", "outage", "spam_wave"]


def windows(market: str) -> pd.DataFrame:
    w = pd.read_parquet(RES / f"windows_{market}.parquet")
    w = w[(w["variant"] == "main") & w["complete"]].sort_values("close_kst").reset_index(drop=True)
    w = w[[c for c in WIN_COLS if c in w.columns]].copy()
    w["outage_b"] = w["outage"].fillna(False).astype(bool)
    w["judge_ok"] = ~w["over96"].astype(bool) & ~w["outage_b"]
    return w


def build_kr(px: dict, sch: dict) -> pd.DataFrame:
    w = windows("kr")
    F = {k: sess_feat(px[t]) for k, t in KR_ASSETS.items()}
    U = {k: sess_feat(px[t]) for k, t in (("sox", "^SOX"), ("ewy", "EWY"), ("soxx", "SOXX"), ("qqq", "QQQ"),
                                          ("vix", "^VIX"))}
    us_sessions = sorted(px["^SOX"].index)
    kr_sessions = sorted(px["^KS11"].index)
    rows = []
    for r in w.itertuples(index=False):
        D, P = pd.Timestamp(r.D).date(), pd.Timestamp(r.P).date()
        row = {}
        for k, f in F.items():
            s = f.loc[P] if P in f.index else None
            cols = ["r", "rpos", "rneg", "r5", "r20", "r60", "DD20", "lnRV20"]
            if k == "kospi":
                cols += ["absr", "ln_absr", "lnRV5", "lnRV22", "ln_vol_anom"]
            for c in cols:
                row[f"m0_{k}_{c}"] = float(s[c]) if s is not None else np.nan
            g = float(f.at[D, "g"]) if D in f.index else np.nan
            row[f"m0_{k}_g"] = g
            if k == "kospi":
                row["m0_kospi_absg"] = abs(g) if np.isfinite(g) else np.nan
        u = last_le(us_sessions, D - timedelta(days=1))
        row["m0_us_session"] = pd.Timestamp(u) if u else pd.NaT
        for k in ("sox", "ewy", "soxx", "qqq"):
            row[f"m0_{k}_r_u"] = float(U[k].at[u, "r"]) if u in U[k].index else np.nan
        for k in ("sox", "ewy"):
            row[f"m0_{k}_ln_absr_u"] = float(U[k].at[u, "ln_absr"]) if u in U[k].index else np.nan
        vix = U["vix"]
        row["m0_vix_u"] = float(vix.at[u, "close"]) if u in vix.index else np.nan
        iu = list(vix.index).index(u) if u in vix.index else None
        row["m0_vix_chg_u"] = float(vix["close"].iloc[iu] - vix["close"].iloc[iu - 1]) if iu else np.nan
        row.update(cal_cols(P, D, r.kind, r.xholiday_mismatch, sch, kr=True))
        row["m0_missing_rate"] = float(r.missing_rate) if pd.notna(r.missing_rate) else np.nan
        row["m0_weekday"] = int(D.weekday())
        for k, t in KR_ASSETS.items():
            row.update(fut_block(px[t], D, k))
        row.update(end_dates(kr_sessions, D, "krx"))
        rows.append(row)
    return pd.concat([w.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def build_us(px: dict, sch: dict, nq: pd.DataFrame) -> pd.DataFrame:
    w = windows("us")
    F = {k: sess_feat(px[t]) for k, t in US_ASSETS.items()}
    vix = sess_feat(px["^VIX"])
    K = {k: sess_feat(px[t]) for k, t in (("kospi", "^KS11"), ("skh", "000660.KS"))}
    kr_sessions = sorted(px["^KS11"].index)
    us_sessions = sorted(px["^SOX"].index)
    rows = []
    for r in w.itertuples(index=False):
        D, P = pd.Timestamp(r.D).date(), pd.Timestamp(r.P).date()
        row = {}
        for k, f in F.items():
            s = f.loc[P] if P in f.index else None
            for c in ("r", "absr", "rpos", "rneg", "r5", "r20", "DD20", "lnRV20"):
                row[f"m0_{k}_{c}"] = float(s[c]) if s is not None else np.nan
            row[f"m0_{k}_g"] = float(f.at[D, "g"]) if D in f.index else np.nan
        row["m0_soxx_minus_ewy_r20"] = row["m0_soxx_r20"] - row["m0_ewy_r20"]
        row["m0_vix"] = float(vix.at[P, "close"]) if P in vix.index else np.nan
        iv = list(vix.index).index(P) if P in vix.index else None
        row["m0_vix_chg"] = float(vix["close"].iloc[iv] - vix["close"].iloc[iv - 1]) if iv else np.nan
        k_ = last_le(kr_sessions, D)
        row["m0_kr_session"] = pd.Timestamp(k_) if k_ else pd.NaT
        for k, f in K.items():
            ok = k_ in f.index
            row[f"m0_{k}_r_k"] = float(f.at[k_, "r"]) if ok else np.nan
            row[f"m0_{k}_oc_k"] = float(f.at[k_, "oc"]) if ok else np.nan
        row["m0_kospi_r20_k"] = float(K["kospi"].at[k_, "r20"]) if k_ in K["kospi"].index else np.nan
        a, a_t = nq_at(nq, datetime(P.year, P.month, P.day, 16, 0))
        b, b_t = nq_at(nq, datetime(D.year, D.month, D.day, 8, 30))
        row["m0_nq_close_to_winclose"] = (b / a - 1) if np.isfinite(a) and np.isfinite(b) else np.nan
        row["m0_nq_from_bar_end_et"], row["m0_nq_to_bar_end_et"] = a_t, b_t
        row.update(cal_cols(P, D, r.kind, r.xholiday_mismatch, sch, kr=False))
        row["m0_missing_rate"] = float(r.missing_rate) if pd.notna(r.missing_rate) else np.nan
        row["m0_weekday"] = int(D.weekday())
        for k, t in US_ASSETS.items():
            row.update(fut_block(px[t], D, k))
        row.update(end_dates(us_sessions, D, "nyse"))
        rows.append(row)
    return pd.concat([w.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def missing_report(df: pd.DataFrame) -> dict:
    out = {}
    for c in df.columns:
        if c.startswith(("m0_", "fut_")) and df[c].dtype.kind in "fi":
            n = int(df[c].isna().sum())
            if n:
                out[c] = {"n_missing": n, "missing_D": [str(pd.Timestamp(d).date()) for d in df.loc[df[c].isna(), "D"]][:12]}
    return out


def main() -> int:
    px, bad = load_prices()
    nq = pd.read_parquet(RES / "prices_nq_1h.parquet")
    cal = pd.read_parquet(RES / "loader_calendar.parquet")
    wk, wu = windows("kr"), windows("us")
    D_all = [pd.Timestamp(d).date() for d in pd.concat([wk["D"], wu["D"]])]
    P_all = [pd.Timestamp(d).date() for d in pd.concat([wk["P"], wu["P"]])]
    sch = schedule(cal, D_all, P_all)
    kr = build_kr(px, sch)
    us = build_us(px, sch, nq)
    kr.to_parquet(RES / "panel_kr.parquet", index=False)
    us.to_parquet(RES / "panel_us.parquet", index=False)
    last_px = {t: str(max(df.index)) for t, df in px.items()}
    meta = {"made": datetime.now().replace(microsecond=0).isoformat(),
            "rows": {"kr": int(len(kr)), "us": int(len(us))},
            "judge_ok": {"kr": int(kr["judge_ok"].sum()), "us": int(us["judge_ok"].sum())},
            "first_D": {"kr": str(kr["D"].min().date()), "us": str(us["D"].min().date())},
            "last_D": {"kr": str(kr["D"].max().date()), "us": str(us["D"].max().date())},
            "last_price_date": last_px, "bad_bars_set_missing": bad,
            "not_included": {"VKOSPI 1일 변화": "yfinance ^VKOSPI 0행(prechecks_external)",
                             "CPI·고용·MSCI 더미": "원천 없음(웹 검색 금지, h01과 같음)",
                             "거래대금": "yfinance에 원화 거래대금 없음 → KOSPI 거래량(주식 수) 이상치로 대신"},
            "m0_entry_instant": ["m0_*_g", "m0_kospi_absg"],
            "m0_note": "m0_*_g는 진입 순간(D 시가) 값이다. 창 마감 뒤 값이므로 특징에는 쓰지 않고 판정 회귀 통제로만 쓴다",
            "schedule": {"kr_opex": [str(d) for d in sch["kr_opex"]], "us_opex": [str(d) for d in sch["us_opex"]],
                         "fomc": [str(d) for d in sch["fomc"]], "earnings": sch["earn_raw"]},
            "missing_kr": missing_report(kr), "missing_us": missing_report(us),
            "files_sha256": {"panel_kr.parquet": sha(RES / "panel_kr.parquet"),
                             "panel_us.parquet": sha(RES / "panel_us.parquet"),
                             "prices.parquet": sha(RES / "prices.parquet")}}
    (RES / "panel_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str), "utf-8")
    print(json.dumps({k: meta[k] for k in ("rows", "judge_ok", "first_D", "last_D", "bad_bars_set_missing")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
