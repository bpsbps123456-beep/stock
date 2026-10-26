"""레버리지 홍수 경보 2단계 판정 (BRIEF.md 사전 등록 규칙).

지표마다 같은 꼴의 규칙(250일 백분위 수준·20일 변화, 위·아래)을 공개 시차를 반영해 켜고,
2026 앞 구간(1~5월)에서 고른 칸을 뒤 구간(6~10월)에서 확인한다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "stage2"
OUT.mkdir(exist_ok=True)

CRASH = -0.08
HORIZON_EVT = 5      # 급락 판정 창(거래일)
GAP = 10             # 사건 시작 사이 최소 간격
LEAD = 5             # 경고 창 E-5..E-1
FWD = 10             # lift: 켜진 날 뒤 10거래일 안 사건 시작
WIN, MINP = 250, 60
HI, LO = 90.0, 10.0
FRONT = ("2026-01-02", "2026-05-29")
BACK = ("2026-06-01", "2026-10-13")
AUX_KR = ("2025-06-02", "2025-12-31")
AUX_US = ("2023-06-01", "2025-12-31")
NSHIFT = 200
rng = np.random.default_rng(20261014)


def events(close: pd.Series) -> list[pd.Timestamp]:
    """급락 확인일 R = 종가가 직전 5거래일 최고 종가보다 8% 이상 낮아진 날.
    사건 시작 A = 그 5일 고점 다음 날(하락 첫날). 직전 사건 시작에서 GAP 거래일 이내면 같은 사건."""
    c = close.dropna()
    v = c.values
    starts, last = [], -10**9
    for r in range(HORIZON_EVT, len(v)):
        w = v[r - HORIZON_EVT:r]
        if v[r] / w.max() - 1 <= CRASH:
            a = r - HORIZON_EVT + int(w.argmax()) + 1
            if a - last > GAP:
                starts.append(c.index[a])
                last = a
    return starts


def pct_rank(s: pd.Series) -> pd.Series:
    def f(w):
        w = w[~np.isnan(w)]
        if len(w) < MINP:
            return np.nan
        return (w[:-1] < w[-1]).mean() * 100 + (w[:-1] == w[-1]).mean() * 50
    return s.rolling(WIN, min_periods=MINP).apply(f, raw=True)


def build_signals(df: pd.DataFrame, cols: dict, cal: pd.DatetimeIndex) -> dict:
    """cols: {열: (공개 시차, 변화 시험 여부)} → {칸 이름: 공개일 기준 켜짐 bool 시리즈(cal 색인)}"""
    sig = {}
    for col, (lag, do_chg) in cols.items():
        # 휴장일 빈 줄(예: 미국 2026-05-25)을 먼저 빼야 shift(20)이 20거래일이 된다
        s = df[col].astype(float).dropna()
        if len(s) < MINP + 10:
            continue
        variants = {"lvl": s}
        if do_chg:
            variants["chg20"] = s / s.shift(20) - 1
        for vname, v in variants.items():
            p = pct_rank(v.dropna()).reindex(s.index)
            for side, on in (("hi", p >= HI), ("lo", p <= LO)):
                on = on & p.notna()
                # 측정일 → 공개일(시장 달력 기준 lag 거래일 뒤)
                pos = cal.searchsorted(on.index, side="left")
                known = pd.Series(False, index=cal)
                for d_on, i in zip(on.values, pos):
                    j = i + lag
                    if d_on and j < len(cal):
                        known.iloc[j] = True
                sig[f"{col}|{vname}|{side}"] = known
    return sig


_YCACHE = {}


def _ytarget(starts, cal):
    key = (id(cal), tuple(starts))
    if key not in _YCACHE:
        pos = {d: i for i, d in enumerate(cal)}
        sidx = np.array([pos[s] for s in starts if s in pos])
        i = np.arange(len(cal))[:, None]
        y = ((sidx[None, :] > i) & (sidx[None, :] <= i + FWD)).any(axis=1) if len(sidx) else np.zeros(len(cal), bool)
        _YCACHE[key] = (y, pos)
    return _YCACHE[key]


def metrics(on: pd.Series, starts: list, cal: pd.DatetimeIndex, win: tuple) -> dict:
    lo, hi = pd.Timestamp(win[0]), pd.Timestamp(win[1])
    y, pos = _ytarget(starts, cal)
    m = (cal >= lo) & (cal <= hi)
    onv = on.values & m
    base = y[m].mean() if m.sum() else np.nan
    p_on = y[onv].mean() if onv.sum() else np.nan
    ev = [s for s in starts if lo <= s <= hi]
    hits = [bool(on.values[max(0, pos[s] - LEAD):pos[s]].any()) for s in ev]
    return {"on_days": int(onv.sum()), "days": int(m.sum()), "p_on": p_on, "base": base,
            "lift": (p_on / base) if base and not np.isnan(p_on) else np.nan,
            "events": len(ev), "hits": int(sum(hits)), "hit_list": hits}


def shift_lift(on: pd.Series, starts, cal, win, k):
    lo, hi = pd.Timestamp(win[0]), pd.Timestamp(win[1])
    m = (cal >= lo) & (cal <= hi)
    seg = on.values[m]
    rolled = on.copy()
    vals = rolled.values.copy()
    vals[m] = np.roll(seg, k)
    rolled = pd.Series(vals, index=cal)
    return metrics(rolled, starts, cal, win)


def run(market: str, close: pd.Series, sigs: dict, aux: tuple) -> pd.DataFrame:
    cal = close.index
    starts = events(close)
    rows = []
    for name, on in sigs.items():
        f, b, a = (metrics(on, starts, cal, w) for w in (FRONT, BACK, aux))
        rows.append({"market": market, "cell": name,
                     **{f"f_{k}": v for k, v in f.items() if k != "hit_list"},
                     **{f"b_{k}": v for k, v in b.items() if k != "hit_list"},
                     "b_hit_list": "".join("O" if h else "X" for h in b["hit_list"]),
                     **{f"a_{k}": v for k, v in a.items() if k != "hit_list"}})
    return pd.DataFrame(rows), starts


def select_confirm(df: pd.DataFrame, sigs, close, top=10, shifts=None):
    cal = close.index
    starts = events(close)
    cand = df[(df.f_on_days >= 5) & df.f_lift.notna()].sort_values(["f_lift", "f_hits"], ascending=False).head(top)
    out = []
    for _, r in cand.iterrows():
        need = max(1, int(np.ceil(r.b_events / 2)))
        ok = (r.b_lift >= 1.5) and (r.b_hits >= need)
        p = np.nan
        if ok:
            on = sigs[r.cell]
            nb = int(((cal >= pd.Timestamp(BACK[0])) & (cal <= pd.Timestamp(BACK[1]))).sum())
            null = [shift_lift(on, starts, cal, BACK, int(rng.integers(10, nb - 10)))["lift"] for _ in range(NSHIFT)]
            null = np.nan_to_num(np.array(null), nan=0.0)
            p = float((null >= r.b_lift).mean())
        out.append({**r.to_dict(), "need_hits": need, "pass_pre": ok, "perm_p": p,
                    "pass": bool(ok and p < 0.05)})
    return pd.DataFrame(out)


def main():
    kr = pd.read_csv(RES / "kr_retail_indicators.csv", parse_dates=["date"]).set_index("date")
    ks = pd.read_csv(RES / "kr_short_daily_merged.csv")
    ks["date"] = pd.to_datetime(ks["date"].astype(str))
    ks = ks.set_index("date")
    us = pd.read_csv(RES / "us_daily_panel.csv", parse_dates=["date"]).set_index("date")

    kospi = ks["kospi_close"].dropna()
    sox = us["SOX"].dropna()

    kr_cols = {c: (1, True) for c in [
        "credit_loan_total", "credit_loan_kospi", "credit_loan_kosdaq", "securities_collateral_loan",
        "investor_deposit", "misu_balance", "cfd_gross_total", "credit_loan_shares_total",
        "credit_loan_total_to_mcap", "securities_collateral_loan_to_mcap", "investor_deposit_to_mcap",
        "misu_balance_to_mcap", "cfd_gross_total_to_mcap", "credit_to_deposit", "misu_to_credit_pct"]}
    kr_cols.update({c: (1, False) for c in ["forced_sale_amount", "forced_sale_ratio_pct"]})
    ks_cols = {c: (1, True) for c in ["lend_bal_shares", "lend_bal_tril", "sec_lend_bal_sh", "hynix_lend_bal_sh"]}
    ks_cols.update({c: (1, False) for c in ["lend_fill_shares"]})
    ks_cols.update({c: (0, False) for c in [
        "fut_foreign_net", "fut_foreign_net_5d", "fut_foreign_net_20d", "prog_nonarb_net", "prog_nonarb_net_5d",
        "kospi_foreign_net", "kospi_foreign_net_5d", "kospi_indiv_net", "samsung_gdr_premium_pct",
        "hynix_gdr_premium_pct"]})
    ks_cols.update({c: (0, True) for c in ["sec_frgn_ratio", "hynix_frgn_ratio"]})

    bal = [c for c in us.columns if c.endswith(("_aum", "_shares", "_dollar_vol_20d")) or c == "equity_call_vol_20d"]
    flow = [c for c in us.columns if c.endswith(("_flow_20d", "_flow_5d", "_ratio_5d", "_short_ratio_5d", "_pcr5"))
            or c.startswith("pc_") and c.endswith("_5d")]
    misc = ["GVIP_SPY", "GVIP_SPY_chg20", "lev_bull_dollar_vol_share", "VIX", "VXN", "SKEW", "VVIX",
            "VIX_VIX3M", "VXN_VIX", "TQQQ_SQQQ_aum_ratio", "UPRO_SPXU_aum_ratio"]
    us_cols = {c: (0, True) for c in bal}
    us_cols.update({c: (0, False) for c in flow + misc if c in us.columns})

    # 한국 지표 → KOSPI 사건
    kr_sig = build_signals(kr, kr_cols, kospi.index)
    kr_sig.update(build_signals(ks, ks_cols, kospi.index))
    # 미국 지표 → SOX 사건, 그리고 KOSPI 사건(미국 d 종가는 다음 한국 개장 전에 안다 → 같은 날짜 이하 마지막 한국 거래일)
    us_sig_sox = build_signals(us, us_cols, sox.index)
    us_on_kospi = {}
    for k, on in us_sig_sox.items():
        s = on[on].index
        known = pd.Series(False, index=kospi.index)
        for d in s:
            j = kospi.index.searchsorted(d, side="right")  # d 다음 한국 거래일 개장 전에 앎 → 그날을 공개일로
            if j < len(kospi):
                known.iloc[j] = True
        # 공개일 j는 경고 창 E-5..E-1 안이어야 하므로 E 당일 개장 전 정보는 j=E로 들어간다 → 한 칸 당겨 E-1로 맞춘다
        us_on_kospi[k] = known.shift(-1, fill_value=False)

    res = {}
    for market, close, sigs, aux in (("KR→KOSPI", kospi, kr_sig, AUX_KR), ("US→SOX", sox, us_sig_sox, AUX_US),
                                     ("US→KOSPI", kospi, us_on_kospi, AUX_KR)):
        df, starts = run(market, close, sigs, aux)
        sc = select_confirm(df, sigs, close)
        df.to_csv(OUT / f"cells_{market.replace('→', '_')}.csv", index=False, encoding="utf-8-sig")
        sc.to_csv(OUT / f"select_{market.replace('→', '_')}.csv", index=False, encoding="utf-8-sig")
        res[market] = {"n_cells": len(df), "events": [d.strftime("%Y-%m-%d") for d in starts],
                       "n_pass_pre": int(sc.pass_pre.sum()) if len(sc) else 0,
                       "pass": sc.loc[sc["pass"], "cell"].tolist() if len(sc) else []}

        # 선택편향 귀무: 모든 신호를 2026 안에서 같은 거리만큼 원형 이동 → 고르기·확인 반복
        cal = close.index
        m26 = (cal >= pd.Timestamp(FRONT[0])) & (cal <= pd.Timestamp(BACK[1]))
        n26 = int(m26.sum())
        cnt = []
        for _ in range(NSHIFT):
            k = int(rng.integers(20, n26 - 20))
            sh = {}
            for name, on in sigs.items():
                v = on.values.copy()
                v[m26] = np.roll(v[m26], k)
                sh[name] = pd.Series(v, index=cal)
            d2, _ = run(market, close, sh, aux)
            cand = d2[(d2.f_on_days >= 5) & d2.f_lift.notna()].sort_values(["f_lift", "f_hits"], ascending=False).head(10)
            need = cand.b_events.apply(lambda e: max(1, int(np.ceil(e / 2))))
            cnt.append(int(((cand.b_lift >= 1.5) & (cand.b_hits >= need)).sum()))
        cnt = np.array(cnt)
        obs = res[market]["n_pass_pre"]
        res[market]["null_mean_pass_pre"] = float(cnt.mean())
        res[market]["null_p_pass_pre"] = float((cnt >= obs).mean())
        print(market, json.dumps(res[market], ensure_ascii=False), flush=True)

    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
