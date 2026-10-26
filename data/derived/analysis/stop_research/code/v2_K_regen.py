"""갈래 K(레드팀) 보조: 1차 8갈래의 전 변형 노출(헤지)을 각 갈래 코드 함수로 다시 만든다(2026-10-10).

공유 파일·fam_*.py는 고치지 않고 import만 한다. 반환 행:
  {"fam", "name", "params", "e": {bn: 노출(at_open 뒤, index=b.days)}, "h": {bn: {종목: 비중}} 또는 None,
   "r": {bn: 일수익} (B만, StockEngine 결과)}
각 갈래 main()과 같은 순서·같은 이름을 쓰고, check_reproduction()으로 results/*.json과 대조한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harness as H  # noqa: E402

BNS = ("2026", "hold")
FAM_FILES = {"A": "A_portfolio_dd", "B": "B_stock_stops", "C": "C_vol_target", "D": "D_trend_breadth",
             "E": "E_macro", "F": "F_hedge", "G": "G_hold_reentry", "H": "H_events_funds"}


def fam_A():
    import fam_A_portfolio_dd as A
    A._FEAT.clear()
    out = []
    for p in A.grid():
        out.append({"fam": "A", "name": A.label(p), "params": p, "e": {bn: A.exposure(bn, p) for bn in BNS}, "h": None})
    return out


def fam_B():
    import fam_B_stock_stops as B
    out = []
    for sp in B.grid():
        res, rr, ee, _ = B.run_variant(sp)
        out.append({"fam": "B", "name": res["name"], "params": sp, "e": ee, "h": None, "r": rr})
    return out


def fam_C():
    import fam_C_vol_target as C
    bb = {bn: H.base(bn) for bn in BNS}
    est = {bn: C.estimators(bn) for bn in BNS}
    out = []
    for en in est["2026"]:
        for tgt in C.TARGETS + ["상대"]:
            for mode in C.MODES:
                out.append({"fam": "C", "name": f"{en}|{tgt}|{mode}", "params": (en, tgt, mode),
                            "e": {bn: C.exposure(est[bn][en], bb[bn], tgt, mode) for bn in BNS}, "h": None})
    return out


def fam_D():
    import fam_D_trend_breadth as D
    D._BR.clear()
    out = []
    for name, params, fn in D.build_grid():
        s = fn()
        out.append({"fam": "D", "name": name, "params": params, "e": {bn: H.at_open(s, H.base(bn)) for bn in BNS},
                    "h": None})
    return out


def fam_E(fred_extra_bday: bool = False):
    import fam_E_macro as E
    if fred_extra_bday:
        orig = H.fred_known
        H.fred_known = fred_known_bday
    try:
        R = E.raw_indicators()
    finally:
        if fred_extra_bday:
            H.fred_known = orig
    S = E.signals(R)
    bb = {bn: H.base(bn) for bn in BNS}
    ON = {bn: E.on_at_open(S, b) for bn, b in bb.items()}
    out = []
    for sn in ON["2026"]:
        for low in E.LOW:
            for mode in E.REENTRY:
                out.append({"fam": "E", "name": f"{sn} | 노출{low:g} | {mode}", "params": (sn, low, mode),
                            "e": {bn: E.exposure(ON[bn][sn], low, mode, b.days) for bn, b in bb.items()}, "h": None})
    return out


def fam_F():
    import fam_F_hedge as F
    sig = {bn: F.raw_signals(bn) for bn in BNS}
    bet = F.betas()
    on = {bn: {t: H.at_open(s, H.base(bn), fill=0.0) for t, s in sig[bn].items()} for bn in BNS}
    beta_open = {bn: {u: H.at_open(s, H.base(bn), fill=np.nan).fillna(F.BETA_FALLBACK).clip(lower=F.BETA_FLOOR)
                      for u, s in bet.items()} for bn in BNS}
    out = []
    for mode, trig, prod, q in F.GRID:
        L, u = F.PRODUCTS[prod]
        ex, hd = {}, {}
        for bn in BNS:
            o = on[bn][trig]
            if mode == "헤지":
                ex[bn] = 1.0 - q * o
                hd[bn] = {prod: q * o}
            else:
                e_eq = ((1 - q) - q * L / beta_open[bn][u]).clip(0.0, 1.0)
                ex[bn] = (1.0 - o) + o * e_eq
        out.append({"fam": "F", "name": f"{mode}|{trig}|{prod}|q={q}", "params": (mode, trig, prod, q), "e": ex,
                    "h": hd if mode == "헤지" else None})
    return out


def fam_G():
    import fam_G_hold_reentry as G
    G._SIG.clear()
    out = []
    for spec in G.variants():
        name, ek, X, re_, Di, Dii = spec
        ex = {}
        for bn in BNS:
            e_close, _ = G.simulate(G.signals(bn), ek, X, re_, Di, Dii)
            ex[bn] = H.at_open(e_close, H.base(bn), fill=1.0)
        out.append({"fam": "G", "name": name, "params": spec, "e": ex, "h": None})
    return out


def fam_H():
    import fam_H_events_funds as HH
    ev = HH.fetch_events()
    b26, bh = H.base("2026"), H.base("hold")
    out = []
    for g, syms in HH.EVENT_GROUPS.items():
        for lvl in HH.EVENT_EXPO:
            for win in HH.EVENT_WINDOW:
                out.append({"fam": "H", "name": f"사건[{g}] {win} 노출{lvl}", "params": ("event", g, lvl, win),
                            "e": {"2026": HH.event_exposure(b26, ev, syms, lvl, win),
                                  "hold": HH.event_exposure(bh, ev, syms, lvl, win)}, "h": None})
    fs = HH.load_fund_series()
    for kind, thr in HH.FUND_SIGNALS:
        t26 = HH.fund_signal(fs["TIME26"], kind, thr, 5)
        k26 = HH.fund_signal(fs["KoAct26"], kind, thr, 5)
        th = HH.fund_signal(fs["TIMEhold"], kind, thr, 1)
        both26 = (t26 & k26.reindex(t26.index).fillna(False))
        for fund, s26 in (("TIME", t26), ("KoAct", k26), ("둘다", both26)):
            for lvl in HH.FUND_EXPO:
                for n in HH.FUND_HOLD:
                    out.append({"fam": "H", "name": f"펀드[{fund} {kind}{thr}] 노출{lvl} 유지{n}",
                                "params": ("fund", fund, kind, thr, lvl, n),
                                "e": {"2026": HH.fund_exposure(b26, s26, lvl, n), "hold": HH.fund_exposure(bh, th, lvl, n)},
                                "h": None})
    return out


GEN = {"A": fam_A, "B": fam_B, "C": fam_C, "D": fam_D, "E": fam_E, "F": fam_F, "G": fam_G, "H": fam_H}


def fred_known_bday() -> pd.DataFrame:
    """민감도: 일간 FRED 관측 t를 '다음 영업일(t+1B)에 공개'로 본다(금요일 관측은 월요일 공개 → 화요일 시가 사용).
    주간(STLFSI4·NFCI)은 harness와 같게 7일."""
    f = pd.read_parquet(H.CACHE / "fred.parquet")
    f.index = pd.to_datetime(f.index)
    out = {}
    for c in f.columns:
        s = f[c].dropna()
        if c in ("STLFSI4", "NFCI"):
            s.index = (s.index + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        else:
            s.index = (s.index + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
        out[c] = s[~s.index.duplicated(keep="last")]
    return pd.DataFrame(out).sort_index()


def returns(row: dict, bn: str, rate: float = 0.0) -> pd.Series:
    """변형 일수익(USD). rate = 현금 연이자(하루 rate/252, 주식·헤지에 안 쓰인 몫 전부: 기본 경로 상한 현금 포함)."""
    b = H.base(bn)
    if "r" in row:
        r = row["r"][bn]
        e = row["e"][bn].reindex(b.ret.index).astype(float)
    else:
        e = row["e"][bn].reindex(b.ret.index).astype(float)
        r = H.overlay(b, row["e"][bn], (row["h"] or {}).get(bn))
    if rate:
        r = r + (1.0 - invested(row, bn)) * rate / 252.0
    return r


def invested(row: dict, bn: str) -> pd.Series:
    """그날 시가 거래 뒤 주식+헤지 비중(나머지 = 현금). 기본 경로 상한 현금을 반영한다."""
    b = H.base(bn)
    inv = pd.Series([sum(w.values()) for w in b.weights], index=b.days).reindex(b.ret.index)
    e = row["e"][bn].reindex(b.ret.index).astype(float)
    tot = (e * inv).clip(upper=1.0)
    for sym, h in ((row.get("h") or {}).get(bn) or {}).items():
        tot = tot + h.reindex(b.ret.index).astype(float).fillna(0.0)
    return tot.clip(0.0, 1.0)


def base_row() -> dict:
    return {"fam": "base", "name": "기본", "params": None,
            "e": {bn: pd.Series(1.0, index=H.base(bn).days) for bn in BNS}, "h": None}


def check_reproduction(rows: list, fam: str) -> dict:
    """다시 만든 변형의 2026 전체·hold 2025 수익이 results/<fam>.json과 같은지(소수 4자리)."""
    js = json.loads((H.RESULTS / f"{FAM_FILES[fam]}.json").read_text(encoding="utf-8"))
    ref = {r["name"]: r for r in js["rows"]}
    bad, n = [], 0
    for row in rows:
        j = ref.get(row["name"])
        if j is None:
            bad.append((row["name"], "이름 없음"))
            continue
        for bn, k, key in (("2026", "전체", "2026_전체_수익"), ("hold", "2025", "hold_2025_수익"),
                           ("2026", "앞", "2026_앞_낙폭")):
            b = H.base(bn)
            st = H.stats(returns(row, bn), *b.periods[k])
            v = st["낙폭"] if key.endswith("낙폭") else st["수익"]
            n += 1
            if abs(round(v, 4) - j[key]) > 1.5e-4:
                bad.append((row["name"], key, round(v, 4), j[key]))
    return {"fam": fam, "rows": len(rows), "json_rows": len(js["rows"]), "checks": n, "mismatch": bad[:10],
            "n_mismatch": len(bad)}
