"""앞날 주가 방화벽 감사 — interim/ 아래 코드·결과에서 창 마감 뒤 가격을 계산·저장한 곳이 있는지 찾는다.

점검 4가지(HYPOTHESES.md §2.0·§2.1 가격 마감, §2.2 누출 규칙, §2.3 M0):
  A. 코드 정적 검사: interim/code/*.py 모든 줄에서 음의 이동(shift(-k)·pct_change(-k)·diff(-k)), 원형 이동(np.roll),
     가운데 창(center=True), 뒤 채우기(bfill), 시가·고가·저가 접근, 앞날 값 이름(R_h·MDD·HL·갭·fwd 등), 가격 받기를 찾는다.
  B. 결과 열 검사: interim/results/*.parquet 모든 열 이름·자료형. 숫자 열 이름에 앞날 값 표지가 있으면 위반 후보.
     가격에서 나온 열(허용 목록)은 C·D로 시점을 확인한다.
  C. 시점 검사: 가격 열이 붙은 창마다 P = D 직전 KRX 거래일(로더 달력), 미국 세션 u = D−1 이하 마지막 NYSE 거래일,
     u 정규장 마감(16:00 ET)이 창 마감(D 08:00 KST, 07:59 포함) 전인지.
  D. 잘라 다시 셈: 창마다 P(또는 u) 이하 종가만 남겨 같은 함수로 다시 셈 → 전체 종가로 셈한 값·저장 값과 비교.
     같으면 저장 값이 P 뒤 가격에 기대지 않는다. 가격은 2026-10-14 이하 종가(KOSPI는 거래량도)만 받고
     시가·고가·저가는 받자마자 버린다. 받은 가격은 저장하지 않는다.
산출: results/audit_firewall.json (보고서 audit_firewall.md는 이 JSON의 숫자로 쓴다)
실행: py audit_firewall.py   (이 폴더에서)
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CODE = Path(__file__).resolve().parent
INTERIM = CODE.parent
RESULTS = INTERIM / "results"
sys.path.insert(0, str(CODE))

LAST_PRICE_DAY = date(2026, 10, 14)
KST, ET = ZoneInfo("Asia/Seoul"), ZoneInfo("America/New_York")
SELF = Path(__file__).name

# ---------------------------------------------------------------- A. 코드 정적 검사
PATTERNS = {
    "neg_shift": r"\.shift\(\s*-|shift\(\s*periods\s*=\s*-|pct_change\(\s*(periods\s*=\s*)?-|\.diff\(\s*(periods\s*=\s*)?-",
    "circular_roll": r"np\.roll\(",
    "centered_or_bfill": r"center\s*=\s*True|bfill|backfill|method\s*=\s*['\"]b",
    "ohlc_access": r"\[\s*['\"](Open|High|Low)['\"]\s*\]|\.(Open|High|Low)\b",
    "forward_name": r"\bR_h\b|\bRh\b|\bR(1|2|3|5|10|21)\b|MDD|\bHL\b|시가\s*→\s*종가|다음 날 갭|\bfwd|forward|future",
    "price_fetch": r"yfinance|\.history\(|get_earnings_dates|fchart\.stock|requests\.(get|post|request)\(",
    "forward_index": r"\[\s*[a-z_]\w*\s*\+\s*\d+\s*\]|iloc\[\s*[a-z_]\w*\s*\+|\[\s*[a-z_]\w*\s*\+\s*\d+\s*:",
}


def scan_code() -> dict:
    hits = []
    files = sorted(p for p in CODE.glob("*.py") if p.name != SELF)
    n_lines = 0
    for f in files:
        for i, line in enumerate(f.read_text("utf-8").splitlines(), 1):
            n_lines += 1
            for cat, rx in PATTERNS.items():
                if re.search(rx, line):
                    hits.append({"file": f.name, "line": i, "cat": cat, "text": line.strip()[:200]})
    by_cat: dict = {}
    for h in hits:
        by_cat.setdefault(h["cat"], 0)
        by_cat[h["cat"]] += 1
    return {"files": [f.name for f in files], "n_files": len(files), "n_lines": n_lines,
            "n_hits": len(hits), "by_cat": by_cat, "hits": hits}


# ---------------------------------------------------------------- B. 결과 열 검사
FWD_COL_I = re.compile(r"(^|_)(r_?h|r(1|2|3|5|10|21)_?fwd|mdd\w*|hl|fwd|forward|future|gap|g_?d|oc|o2c|open_close|"
                       r"ret_next|next_ret|lead\d*|open|high|low)($|_)", re.I)
# 문서 기호 R_h·R5·R21·MDD_h·HL(대문자)은 대소문자를 가려 잡는다. 소문자 r5·r20·r60은 '지난' 값 이름(§2.0)이라 허용
FWD_COL_CS = re.compile(r"(^|_)(R_?h|Rh|R(1|2|3|5|10|21)|MDD\w*|HL)($|_)")


class _FwdCol:
    @staticmethod
    def search(name: str):
        return FWD_COL_I.search(name) or FWD_COL_CS.search(name)


FWD_COL = _FwdCol()
PRICE_COLS = {   # 가격에서 나온 열(허용 목록). 모두 C·D로 시점 확인
    "feat_text.parquet": ["px_kospi_rz_P", "px_hynix_rz_P", "ev_kospi", "ev_hynix", "ev_any", "ev_first", "ev_first_kospi"],
    "h01_m0_matrix.parquet": ["kospi_r", "hynix_r", "hynix_rpos", "hynix_rneg", "hynix_r5", "hynix_r20", "kospi_rpos",
                              "kospi_rneg", "kospi_r5", "kospi_r20", "kospi_r60", "kospi_DD20", "sox_r_u", "ewy_r_u",
                              "kospi_lnRV20", "kospi_ln_vol_anom"],
}
PRICE_WORD = re.compile(r"kospi|hynix|sox|ewy|px_|close|price|ret|vol_anom|rv\d|dd\d", re.I)


def scan_parquet() -> dict:
    out = {}
    for f in sorted(RESULTS.glob("*.parquet")):
        sch = pq.read_schema(f)
        cols = []
        for name, typ in zip(sch.names, sch.types):
            t = str(typ)
            numeric = any(k in t for k in ("double", "float", "int")) and "timestamp" not in t
            fwd = bool(FWD_COL.search(name))
            cols.append({"col": name, "type": t, "numeric": numeric, "fwd_name": fwd,
                         "price_word": bool(PRICE_WORD.search(name))})
        allowed = set(PRICE_COLS.get(f.name, []))
        out[f.name] = {
            "rows": pq.read_metadata(f).num_rows, "n_cols": len(cols),
            "fwd_name_cols": [c["col"] + f" ({c['type']})" for c in cols if c["fwd_name"]],
            "fwd_name_numeric": [c["col"] for c in cols if c["fwd_name"] and c["numeric"]],
            "price_word_cols": [c["col"] for c in cols if c["price_word"]],
            "price_cols_listed": sorted(allowed),
            "price_cols_missing_from_file": sorted(allowed - set(sch.names)),
        }
    return out


# ---------------------------------------------------------------- C. 시점 검사
def calendars():
    cal = pd.read_parquet(RESULTS / "loader_calendar.parquet")
    krx = sorted(pd.Timestamp(d).date() for d in cal.loc[cal["krx"], "date"])
    nyse = sorted(pd.Timestamp(d).date() for d in cal.loc[cal["nyse"], "date"])
    return krx, nyse


def prev_session(D: date, sess: list[date]) -> date | None:
    c = [s for s in sess if s < D]
    return max(c) if c else None


def us_close_kst(u: date) -> datetime:
    return datetime.combine(u, time(16), ET).astimezone(KST).replace(tzinfo=None)


def timing_checks() -> dict:
    krx, nyse = calendars()
    wk = pd.read_parquet(RESULTS / "windows_kr.parquet")
    wk = wk[wk["variant"] == "main"].set_index("win_id")
    res = {}

    ft = pd.read_parquet(RESULTS / "feat_text.parquet")
    pxc = PRICE_COLS["feat_text.parquet"]
    kr = ft[ft["market"] == "kr"]
    us = ft[ft["market"] == "us"]
    bad = []
    for r in kr.itertuples():
        D, P = pd.Timestamp(r.D).date(), pd.Timestamp(r.P).date()
        w = wk.loc[r.win_id]
        okP = (P == prev_session(D, krx)) and (pd.Timestamp(w["P"]).date() == P)
        okW = pd.Timestamp(r.close_kst) == pd.Timestamp(datetime.combine(D, time(8)))
        okS = datetime.combine(P, time(15, 30)) < pd.Timestamp(r.close_kst).to_pydatetime()
        if not (okP and okW and okS and P <= LAST_PRICE_DAY):
            bad.append(r.win_id)
    res["feat_text"] = {"kr_rows": int(len(kr)), "kr_rows_with_px": int(kr["px_kospi_rz_P"].notna().sum()),
                        "us_rows": int(len(us)),
                        "us_rows_with_any_px": int(us[pxc].apply(lambda s: s.notna() & (s.astype(object) != False)).any(axis=1).sum())
                        if set(pxc) <= set(us.columns) else 0,
                        "kr_rows_bad_timing": bad}

    mm = pd.read_parquet(RESULTS / "h01_m0_matrix.parquet")
    badP, badU, badUclose = [], [], []
    for r in mm.itertuples():
        D, P = pd.Timestamp(r.D).date(), pd.Timestamp(r.P).date()
        u = date.fromisoformat(r.us_session)
        if not (P == prev_session(D, krx) and pd.Timestamp(wk.loc[r.win_id, "P"]).date() == P and P <= LAST_PRICE_DAY):
            badP.append(r.win_id)
        if not (u == max(s for s in nyse if s <= D - timedelta(days=1)) and u <= LAST_PRICE_DAY):
            badU.append(r.win_id)
        if not (us_close_kst(u) < datetime.combine(D, time(8))):
            badUclose.append(r.win_id)
    lead_min = min((datetime.combine(pd.Timestamp(r.D).date(), time(8)) - us_close_kst(date.fromisoformat(r.us_session)))
                   .total_seconds() / 3600 for r in mm.itertuples())
    res["h01_m0_matrix"] = {"rows": int(len(mm)), "P_not_prev_krx_session": badP, "us_session_not_last_le_Dm1": badU,
                            "us_close_not_before_window_close": badUclose,
                            "min_hours_us_close_to_window_close": round(lead_min, 2),
                            "max_P": str(mm["P"].max().date()), "max_us_session": max(mm["us_session"])}
    return res


# ---------------------------------------------------------------- D. 잘라 다시 셈
def fetch_closes(ticker: str, start: date, volume: bool = False) -> pd.DataFrame:
    import yfinance as yf
    h = yf.Ticker(ticker).history(start=start.isoformat(), end=(LAST_PRICE_DAY + timedelta(days=1)).isoformat(),
                                  auto_adjust=False)
    h = h[["Close"] + (["Volume"] if volume else [])].copy()      # 시가·고가·저가는 여기서 버린다
    h.index = [pd.Timestamp(t).date() for t in h.index]
    h = h[~h.index.duplicated(keep="last")].dropna(subset=["Close"])
    if len(h) and max(h.index) > LAST_PRICE_DAY:
        raise RuntimeError("방화벽: 10/14 뒤 가격")
    return h


def cmp(a, b, tol=1e-9) -> bool:
    if (a is None or (isinstance(a, float) and np.isnan(a))) and (b is None or (isinstance(b, float) and np.isnan(b))):
        return True
    try:
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(a)), abs(float(b)))
    except (TypeError, ValueError):
        return a == b


def truncation_h01() -> dict:
    import h01_preview as H
    px = {"kospi": fetch_closes("^KS11", H.PRICE_START, True), "hynix": fetch_closes("000660.KS", H.PRICE_START),
          "sox": fetch_closes("^SOX", H.PRICE_START), "ewy": fetch_closes("EWY", H.PRICE_START)}
    full = {"kospi": H.session_features(px["kospi"]["Close"], px["kospi"]["Volume"]),
            "hynix": H.session_features(px["hynix"]["Close"]),
            "sox": H.session_features(px["sox"]["Close"], lags=0), "ewy": H.session_features(px["ewy"]["Close"], lags=0)}
    mm = pd.read_parquet(RESULTS / "h01_m0_matrix.parquet")
    kcols = ["r", "rpos", "rneg", "r5", "r20", "r60", "DD20", "lnRV20", "ln_vol_anom"]
    hcols = ["r", "rpos", "rneg", "r5", "r20"]
    n_cmp = n_trunc_ne_full = n_full_ne_stored = n_trunc_ne_stored = 0
    ex_tf, ex_fs = [], []
    direct_r_bad = 0
    for r in mm.itertuples():
        P = pd.Timestamp(r.P).date()
        u = date.fromisoformat(r.us_session)
        k = px["kospi"][px["kospi"].index <= P]
        tk = H.session_features(k["Close"], k["Volume"]).iloc[-1]
        hx = px["hynix"][px["hynix"].index <= P]
        th = H.session_features(hx["Close"]).iloc[-1]
        ts = H.session_features(px["sox"]["Close"][px["sox"].index <= u], lags=0).iloc[-1]
        te = H.session_features(px["ewy"]["Close"][px["ewy"].index <= u], lags=0).iloc[-1]
        # 직접 공식(함수와 따로): r(P) = c(P)/c(직전 세션) − 1
        c = k["Close"]
        if not cmp(c.iloc[-1] / c.iloc[-2] - 1, getattr(r, "kospi_r")):
            direct_r_bad += 1
        trip = ([(f"kospi_{c_}", tk[c_], full["kospi"].at[P, c_]) for c_ in kcols]
                + [(f"hynix_{c_}", th[c_], full["hynix"].at[P, c_]) for c_ in hcols]
                + [("sox_r_u", ts["r"], full["sox"].at[u, "r"]), ("ewy_r_u", te["r"], full["ewy"].at[u, "r"])])
        for col, tv, fv in trip:
            sv = getattr(r, col)
            n_cmp += 1
            if not cmp(tv, fv):
                n_trunc_ne_full += 1
                ex_tf.append([r.win_id, col])
            if not cmp(fv, sv, 1e-6):
                n_full_ne_stored += 1
                ex_fs.append([r.win_id, col, None if pd.isna(fv) else float(fv), None if pd.isna(sv) else float(sv)])
            if not cmp(tv, sv, 1e-6):
                n_trunc_ne_stored += 1
    return {"rows": int(len(mm)), "values_compared": n_cmp, "trunc_ne_full": n_trunc_ne_full,
            "full_ne_stored": n_full_ne_stored, "trunc_ne_stored": n_trunc_ne_stored,
            "direct_kospi_r_mismatch": direct_r_bad,
            "examples_trunc_ne_full": ex_tf[:10], "examples_full_ne_stored": ex_fs[:10],
            "fetched_last": {k_: str(max(v.index)) for k_, v in px.items()}}


def truncation_feat_text() -> dict:
    import feat_text as T
    ft = pd.read_parquet(RESULTS / "feat_text.parquet")
    wk = pd.read_parquet(RESULTS / "windows_kr.parquet")
    wk_main = wk[(wk["variant"] == "main") & wk["complete"]]
    first_P = pd.Timestamp(wk_main["P"].min()).date()
    start = first_P - timedelta(days=150)
    ser = {"kospi": fetch_closes("^KS11", start)["Close"], "hynix": fetch_closes("000660.KS", start)["Close"]}
    full = {k: T.shock_z(v) for k, v in ser.items()}
    kr = ft[(ft["market"] == "kr")]
    n_cmp = tf = fs = ts_ = 0
    ex = []
    ev_bad = 0
    for r in kr.itertuples():
        P = pd.Timestamp(r.P).date()
        for nm in ("kospi", "hynix"):
            s = ser[nm][ser[nm].index <= P]
            tv = T.shock_z(s).iloc[-1] if len(s) else np.nan
            fv = full[nm].get(P, np.nan)
            sv = getattr(r, f"px_{nm}_rz_P")
            if not r.valid:
                continue
            n_cmp += 1
            tf += not cmp(tv, fv)
            fs += not cmp(fv, sv, 1e-6)
            ts_ += not cmp(tv, sv, 1e-6)
            if not cmp(fv, sv, 1e-6):
                ex.append([r.win_id, nm, None if pd.isna(fv) else float(fv), None if pd.isna(sv) else float(sv)])
        # 사건 표지는 저장된 rz ≤ −2로만 정해지는지(다른 가격을 안 쓰는지)
        if r.valid and pd.notna(r.px_kospi_rz_P):
            ev_bad += bool(r.ev_kospi) != bool(r.px_kospi_rz_P <= T.EV_Z)
    return {"kr_rows": int(len(kr)), "values_compared": n_cmp, "trunc_ne_full": tf, "full_ne_stored": fs,
            "trunc_ne_stored": ts_, "ev_kospi_not_from_stored_rz": ev_bad, "examples_full_ne_stored": ex[:10],
            "fetched_last": {k: str(max(v.index)) for k, v in ser.items()}}


def main() -> int:
    out = {"created": datetime.now().isoformat(timespec="seconds"), "last_price_day": LAST_PRICE_DAY.isoformat()}
    out["A_code"] = scan_code()
    out["B_parquet"] = scan_parquet()
    out["C_timing"] = timing_checks()
    for nm, fn in (("D_trunc_h01", truncation_h01), ("D_trunc_feat_text", truncation_feat_text)):
        try:
            out[nm] = fn()
        except Exception as ex:  # noqa: BLE001
            out[nm] = {"error": f"못 함: {type(ex).__name__}: {str(ex)[:200]}"}
    (RESULTS / "audit_firewall.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), "utf-8")
    brief = {k: v for k, v in out.items() if k not in ("A_code", "B_parquet")}
    brief["A_code"] = {k: out["A_code"][k] for k in ("n_files", "n_lines", "n_hits", "by_cat")}
    brief["B_parquet"] = {f: {"fwd_name_numeric": v["fwd_name_numeric"], "fwd_name_cols": v["fwd_name_cols"]}
                          for f, v in out["B_parquet"].items()}
    print(json.dumps(brief, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
