"""판정 실행기. `py code/run.py dry`(대리 신호 시운전) / `py code/run.py real`(본판정, BRIEF 고정 뒤 한 번).

본판정은 BRIEF.md SHA-256이 freeze.json과 같을 때만 돈다.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import harness as hz
from steps import regime_adjust

ROOT = Path(__file__).resolve().parents[1]
DER = ROOT / "derived"
RES = ROOT / "results"
PCT_SIGS = ["S1", "S2", "S3", "S4", "S5n", "S5b", "S7", "S8", "S9n", "S9b", "S10n", "S10b", "L1", "L2",
            "C1", "C2", "C3", "C4"]


def build(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    pct = pd.DataFrame({s: hz.rolling_pct(raw[s]) for s in PCT_SIGS})
    for s, c in hz.CENSORED.items():
        pct[s] = pct[s].where(hz.censor_ok(raw[c]))
    return pct, hz.on_vectors(pct, raw)


def setup():
    raw = pd.read_parquet(DER / "sig_daily.parquet").set_index("entry")
    raw.index = pd.DatetimeIndex(raw.index)
    px = hz.load_prices()
    entries = raw.index
    front = hz.make_period("front", hz.FRONT, px, entries)
    back = hz.make_period("back", hz.BACK, px, entries)
    sd_front = {k: float(np.nanstd(v)) for k, v in front.y.items()}
    steps = json.loads((DER / "steps.json").read_text(encoding="utf-8"))
    stepped = tuple(s for s, v in steps.items() if any(r and r["step"] for r in v.values()))
    return raw, px, entries, front, back, sd_front, steps, stepped


def planted(raw, px, entries, rho, rng, phi):
    """S1에 SOXX 앞 5일 수익과 상관 rho(높음 → 나쁨)를 심은 대리 신호."""
    p = px[px["ticker"] == "SOXX"].set_index("date").reindex(entries)
    o, c = p["Open"].values, p["Close"].values
    r5 = np.full(len(entries), np.nan)
    r5[:-4] = c[4:] / o[:-4] - 1
    z = (r5 - np.nanmean(r5)) / np.nanstd(r5)
    e = np.zeros(len(entries))
    eps = rng.standard_normal(len(entries))
    for i in range(1, len(e)):
        e[i] = phi * e[i - 1] + np.sqrt(1 - phi ** 2) * eps[i]
    sig = -rho * np.nan_to_num(z) + np.sqrt(1 - rho ** 2) * e
    return pd.Series(sig, index=entries)


def dry(B=200, step=8):
    raw, px, entries, front, back, sd_front, steps, stepped = setup()
    cells = hz.all_cells()
    X = raw[PCT_SIGS + ["K1"]].copy()
    rng = np.random.default_rng(20261026)
    null_rows = []
    for b in range(B):
        S = hz.surrogate(X, rng)
        S["K1"] = (S["K1"] >= S["K1"].quantile(1 - raw["K1"].mean())).astype(float).where(S["K1"].notna())
        S = S.join(raw[[c for c in raw.columns if c.startswith("cens_")]])
        pct, ons = build(S)
        r = hz.judge(cells, ons, front, back, sd_front)
        fam = hz.family_null(cells, ons, px, entries, hz.FRONT, hz.BACK, sd_front, step=step, stepped=stepped) \
            if r["n_pass_explore"] > 0 else np.array([])
        p_fam = (1 + (fam >= r["n_pass_explore"]).sum()) / (1 + len(fam)) if len(fam) else np.nan
        null_rows.append({"b": b, "primary_pass": r["n_pass_primary"], "explore_pass": r["n_pass_explore"],
                          "p_family": p_fam,
                          "concl_distinct": bool(r["n_pass_primary"] > 0 or (r["n_pass_explore"] > 0 and p_fam < 0.05)),
                          "primary_p_min": float(np.nanmin(r["primary"]["p"])) if r["primary"]["p"].notna().any() else np.nan})
        if b % 20 == 0:
            print(b, null_rows[-1], flush=True)
    nul = pd.DataFrame(null_rows)
    phi = float(raw["S1"].autocorr(1))
    plant = []
    for rho in [0.1, 0.2, 0.35, 0.5]:
        for b in range(100):
            S = hz.surrogate(X, rng)
            S["K1"] = (S["K1"] >= S["K1"].quantile(1 - raw["K1"].mean())).astype(float).where(S["K1"].notna())
            S["S1"] = planted(raw, px, entries, rho, rng, phi).where(raw["S1"].notna())
            S = S.join(raw[[c for c in raw.columns if c.startswith("cens_")]])
            pct, ons = build(S)
            r = hz.judge(cells, ons, front, back, sd_front)
            pr = r["primary"]
            row = pr[(pr["signal"] == "S1") & (pr["outcome"] == "R5")].iloc[0]
            ex = r["explore"]
            plant.append({"rho": rho, "b": b, "primary_S1R5_pass": bool(row["pass"]), "p_raw": row["p"],
                          "p_holm": row["p_holm"],
                          "explore_has_S1": bool((ex["signal"] == "S1").any()),
                          "explore_S1_pass": bool(((ex["signal"] == "S1") & ex["pass"]).any())})
        print("rho", rho, pd.DataFrame([x for x in plant if x["rho"] == rho])[
            ["primary_S1R5_pass", "explore_has_S1", "explore_S1_pass"]].mean().to_dict(), flush=True)
    pl = pd.DataFrame(plant)
    RES.mkdir(exist_ok=True)
    nul.to_csv(RES / "dryrun_null.csv", index=False)
    pl.to_csv(RES / "dryrun_planted.csv", index=False)
    summ = {
        "B_null": B, "family_step": step, "phi_S1": phi,
        "null_rate_primary_any_pass": float((nul["primary_pass"] > 0).mean()),
        "null_rate_explore_any_pass": float((nul["explore_pass"] > 0).mean()),
        "null_rate_conclusion_distinct": float(nul["concl_distinct"].mean()),
        "planted": pl.groupby("rho")[["primary_S1R5_pass", "explore_has_S1", "explore_S1_pass"]].mean().to_dict(),
    }
    (RES / "dryrun_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))


SHARE_NUM = {"S1": "SOXL_net", "S2": "SOXL_buy", "S3": "EQ3L_net", "S4": "EQ3L_buy", "S5n": "LIDX_net",
             "S5b": "LIDX_buy", "S9n": "KORU_net", "S9b": "KORU_buy", "S10n": "TQQQ_net", "S10b": "TQQQ_buy"}
COMBO = {"P1": ("S1", "C1"), "P2": ("S3", "S7"), "P4": ("L2", "C3")}


def on_for(raw_mod: pd.DataFrame, cell: hz.Cell) -> pd.Series:
    sigs = COMBO.get(cell.signal, (cell.signal,))
    if cell.signal == "K1":
        return raw_mod["K1"].astype(float)
    pct = {s: hz.rolling_pct(raw_mod[s]) for s in sigs}
    for s in sigs:
        if s in hz.CENSORED:
            pct[s] = pct[s].where(hz.censor_ok(raw_mod[hz.CENSORED[s]]))
    if cell.signal in COMBO:
        a, b = (pct[s] for s in sigs)
        return ((a >= cell.thr) & (b >= cell.thr)).astype(float).where(a.notna() & b.notna())
    p = pct[cell.signal]
    on = (p >= cell.thr) if cell.direction == "high" else (p <= cell.thr)
    return on.astype(float).where(p.notna())


def make_rebuild(raw: pd.DataFrame, allow_real: bool = True):
    def rebuild(tag, raw_mod=None, cell=None):
        if tag == "on":
            return on_for(raw_mod, cell)
        if not allow_real:  # 대리 신호 시험: 실제 SEIBRO 분자·보관을 읽는 판은 끈다
            return None
        if tag == "C2star":
            cu = pd.read_parquet(DER / "sig_custody.parquet").set_index("custody_date")
            px = hz.load_prices()
            close = px[px["ticker"] == "SOXL"].set_index("date")["Close"]
            pnow = close.reindex(pd.to_datetime(cu["val_date"])).values
            sh20 = cu["SOXL_shares"].shift(20).values
            c2s = sh20 * pnow / (cu["eq_sum"].values - cu["SOXL_amt"].values + sh20 * pnow)
            s = pd.Series(c2s, index=pd.to_datetime(cu["entry"]))
            s = s.groupby(level=0).last()
            return s.reindex(raw.index).ffill()
        sigs = COMBO.get(cell.signal, (cell.signal,))
        r2 = raw.copy()
        changed = False
        if tag == "denom":
            b65 = raw["B"].rolling(65, min_periods=20).median()
            for s in sigs:
                if s in SHARE_NUM:
                    r2[s] = raw[SHARE_NUM[s]].fillna(0) / b65
                    changed = True
        if tag == "fixed":
            fx = {"S3": raw["EQ3L_net_fx"] / raw["N"], "S4": raw["EQ3L_buy_fx"] / raw["B"],
                  "S5n": raw["LIDX_net_fx"] / raw["N"], "S5b": raw["LIDX_buy_fx"] / raw["B"],
                  "S8": raw["LIDX_buy_fx"] / (raw["LIDX_buy_fx"] + raw["INVIDX_buy_fx"]),
                  "L2": raw["LIDX_buy_fx"] / raw["SPOT_buy"]}
            for s in sigs:
                if s in fx:
                    r2[s] = fx[s]
                    changed = True
        return on_for(r2, cell) if changed else None
    return rebuild


def gate_all(res, cells, ons, raw, px, front, back, sd_front, ons_adj, steps, stepped, allow_real=True):
    """주 가설 8칸 + 탐색 상위 10칸의 관문·강건."""
    import gates

    rebuild = make_rebuild(raw, allow_real)
    rows = []
    adj_back = hz.evaluate_cells(cells, ons_adj, back, sd_ref=sd_front)
    for kind, df in [("primary", res["primary"]), ("explore", res["explore"])]:
        for i, r in df.iterrows():
            c = cells[i]
            key = (c.signal, c.direction, c.thr)
            g = gates.run_gates(c, ons[key], raw, None, px, front, back, sd_front, rebuild)
            comp = COMBO.get(c.signal, (c.signal,))
            if any(s in stepped for s in comp):
                ra = adj_back.loc[i]
                g["regime"] = {"e_h": float(ra["e_h"]), "p": float(ra["p"]), "n_on": int(ra["n_on"])}
                g["regime_ok"] = bool(ra["e_h"] > 0 and ra["e_h"] >= 0.5 * r["e_h"])
            else:
                g["regime"], g["regime_ok"] = None, True
            rows.append({"kind": kind, "i": int(i), "signal": c.signal, "target": c.target, "direction": c.direction,
                         "thr": c.thr, "outcome": c.outcome, "stat_pass": bool(r["pass"]), **{"gates": g}})
    return rows


def replication(raw, px, entries, rng) -> dict:
    """선행 후보 재현 2칸(BRIEF §6): R-a Ra ≥ 80 → SOXX R21 Δ, R-b Rb ≥ 90 → ^SOX A10 위험 차·lift."""
    from events import drawdown_starts

    lo, hi = pd.Timestamp(rng[0]), pd.Timestamp(rng[1])
    sel = entries[(entries >= lo) & (entries <= hi)]
    end_i = int(np.searchsorted(entries, hi, side="right")) - 1
    ii = np.searchsorted(entries, sel)
    out = {}
    p = px[px["ticker"] == "SOXX"].set_index("date")
    p = p[~p.index.duplicated()].reindex(entries)
    o, c = p["Open"].values, p["Close"].values
    r21 = np.where(ii + 20 <= end_i, c[np.minimum(ii + 20, len(c) - 1)] / o[ii] - 1, np.nan)
    sox = px[px["ticker"] == "^SOX"].set_index("date")["Close"]
    sox = sox[~sox.index.duplicated()]
    a = [pd.Timestamp(x["A"]) for x in drawdown_starts(sox, rng[1])]
    pos = {d: k for k, d in enumerate(entries)}
    a10 = np.zeros(len(ii))
    for d in a:
        if d in pos:
            a10[(ii <= pos[d]) & (pos[d] <= ii + 9)] = 1
    a10[~(ii + 13 <= end_i)] = np.nan
    for name, sig, thr, y, ev in [("R-a", "Ra", 80, r21, False), ("R-b", "Rb", 90, a10, True)]:
        pct = hz.rolling_pct(raw[sig])
        on = (pct >= thr).astype(float).where(pct.notna()).reindex(sel).values
        eff = hz.shift_effects(on[None, :], y, ev)[0]
        e = eff * (1.0 if ev else -1.0)
        null = e[1:][~np.isnan(e[1:])]
        pv = (1 + (null >= e[0]).sum()) / (1 + len(null)) if len(null) else np.nan
        on_m = (on == 1) & ~np.isnan(y)
        base = float(np.nanmean(y))
        out[name] = {"effect_raw": float(eff[0]), "p_one_sided": float(pv), "n_on": int(on_m.sum()),
                     "mean_on": float(np.nanmean(y[on_m])) if on_m.any() else None, "base_rate_or_mean": base,
                     "lift": float(np.nanmean(y[on_m]) / base) if ev and on_m.any() and base > 0 else None}
    return out


def real(step=1, check_freeze=True, raw_override=None, tag="real"):
    if check_freeze:
        frz = json.loads((ROOT / "freeze.json").read_text(encoding="utf-8"))
        for f, h in frz["files"].items():
            assert hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == h, f"{f}가 고정본과 다르다"
    raw, px, entries, front, back, sd_front, steps, stepped = setup()
    if raw_override is not None:
        raw = raw_override
    cells = hz.all_cells()
    pct, ons = build(raw)
    r = hz.judge(cells, ons, front, back, sd_front)
    adj_raw = raw.copy()
    for s in stepped:
        adj_raw[s] = regime_adjust(raw.reset_index(), s, steps).values
    _, ons_adj = build(adj_raw)
    gated = gate_all(r, cells, ons, raw, px, front, back, sd_front, ons_adj, steps, stepped,
                     allow_real=raw_override is None)
    out = RES / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "gates.json").write_text(json.dumps(gated, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    fam = hz.family_null(cells, ons, px, entries, hz.FRONT, hz.BACK, sd_front, step=step, stepped=stepped)
    p_fam = float((1 + (fam >= r["n_pass_explore"]).sum()) / (1 + len(fam)))
    r["front"].to_csv(out / "cells_front.csv")
    r["back"].to_csv(out / "cells_back.csv")
    r["primary"].to_csv(out / "primary.csv")
    r["explore"].to_csv(out / "explore_top10.csv")
    pct.to_parquet(out / "pct_daily.parquet")
    aux = hz.make_period("aux2025", hz.AUX25, px, entries)
    hz.evaluate_cells(cells, ons, aux).to_csv(out / "cells_aux2025.csv")
    if raw_override is None:
        rep = {"back": replication(raw, px, entries, hz.BACK), "front": replication(raw, px, entries, hz.FRONT),
               "aux2025": replication(raw, px, entries, hz.AUX25)}
        (out / "replication.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    # 결론 규칙(BRIEF §8)
    ok = {(g["kind"], g["i"]): g for g in gated}
    prim_ok = [g for g in gated if g["kind"] == "primary" and g["stat_pass"] and g["gates"]["echo_ok"] is not False
               and g["gates"]["regime_ok"] and g["gates"]["robust_12_ok"]]
    expl_ok = [g for g in gated if g["kind"] == "explore" and g["stat_pass"] and g["gates"]["echo_ok"] is not False
               and g["gates"]["regime_ok"] and g["gates"]["robust_12_ok"]]
    if prim_ok or (expl_ok and p_fam < 0.05):
        concl = "2026 안에서 우연과 구별됨(가설 구간과 일치)"
    elif r["n_pass_primary"] or r["n_pass_explore"]:
        concl = "후보(우연과 구별 안 됨)"
    else:
        concl = "이 자료로는 판정 불가"
    summ = {"cells": len(cells), "n_pass_primary": r["n_pass_primary"], "n_pass_explore": r["n_pass_explore"],
            "primary_all_gates": [f"{g['signal']}->{g['target']} {g['outcome']}" for g in prim_ok],
            "explore_all_gates": [f"{g['signal']}->{g['target']} {g['direction']}{g['thr']} {g['outcome']}" for g in expl_ok],
            "family_counts_mean": float(fam.mean()), "family_p": p_fam, "family_K": int(len(fam)),
            "stepped": stepped, "conclusion": concl}
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    print(r["primary"][["signal", "target", "outcome", "e_h", "p", "p_holm", "n_on", "clusters", "front_e_h",
                        "judgeable", "pass"]].to_string())
    print(r["explore"][["signal", "target", "direction", "thr", "outcome", "front_z", "e_h", "p", "n_on",
                        "clusters", "judgeable", "pass"]].to_string())


def gatetest():
    """관문 코드가 도는지 대리 신호 한 벌로 확인(실제 SEIBRO–가격 연결 없음)."""
    raw, *_ = setup()
    X = raw[PCT_SIGS + ["K1"]]
    S = hz.surrogate(X, np.random.default_rng(7))
    S["K1"] = (S["K1"] >= S["K1"].quantile(1 - raw["K1"].mean())).astype(float).where(S["K1"].notna())
    keep = ["cens_KORU", "cens_SOXS", "cens_NVDL", "t_start", "t_end", "cdate", "L"]  # 날짜·절단 표지만(흐름 값 없음)
    real(step=25, check_freeze=False, raw_override=S.join(raw[keep]), tag="gatetest_surrogate")


if __name__ == "__main__":
    {"dry": dry, "real": real, "gatetest": gatetest}[sys.argv[1]]()
