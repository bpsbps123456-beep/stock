"""timing 갈래 — 언제 알 수 있었나, 쓸모가 있나, 지금은 어디인가 (BRIEF §7, 설명용).

입력: derived/(고치지 않음), results/profile/commonality.csv, results/rules.md 사전 등록 절.
출력: results/timing/{timing.csv, rules.csv, rules_daily.csv, rules.md(결과 절 덧붙임), now.csv, now.md, summary.md}
실행: py -B code/timing.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
DER = STUDY / "derived"
OUT = STUDY / "results" / "timing"
PROF = STUDY / "results" / "profile"

# 창: (끝 k, 길이 L) — profile.py와 같음
WINS = {"W1": (-6, 15), "W2": (-1, 5), "W3": (0, 1), "W4": (5, 5), "W5": (20, 15)}
HI, LO, ZMIN = 80.0, 20.0, 0.5
STATE_DD = -0.10
HS = (5, 10, 20)
NEAR = 5  # B10 바닥 ±5세션
GAP = 5   # 상태 날이 이 세션 수보다 길게 끊기면 새 하락 구간
# 사전 등록 지문(rules.md 위 절과 같아야 함)
FPS = [
    {"id": "FP1", "measure": "X3", "window": "W2", "L": 5, "side": "+", "src": "profile E(2023~25 7/7, F 0.41)",
     "name": "SOXL 매도 [t−4,t] E_exp ≥ +0.5"},
    {"id": "FP2", "measure": "L1", "window": "W3", "L": 1, "side": "-", "src": "contrast sel_rank 1(2023~25 0/4 양)",
     "name": "3배 롱 순매수 [t] E_exp ≤ −0.5"},
]
MARK = "<!-- 결과 -->"


def iso(d) -> str:
    return pd.Timestamp(d).date().isoformat() if pd.notna(d) else ""


def load():
    mw = pd.read_parquet(DER / "measures_win.parquet")
    md = pd.read_parquet(DER / "measures_daily.parquet").set_index("date")
    pf = pd.read_parquet(DER / "price_feats.parquet").set_index("date")
    cu = pd.read_parquet(DER / "custody_daily.parquet")
    ev = json.loads((DER / "events.json").read_text(encoding="utf-8"))
    return mw, md, pf, cu, ev


def session_on_or_after(S: pd.DatetimeIndex, d) -> pd.Timestamp | None:
    i = S.searchsorted(pd.Timestamp(d))
    return S[i] if i < len(S) else None


# ---------- 1. 아는 시점 ----------
def timing_table(ev, pf, cu, S) -> pd.DataFrame:
    pos = pd.Series(np.arange(len(S)), index=S)
    soxl = cu[cu["product"] == "SOXL"].sort_values("custody_date")
    rows = []
    for b in ev["B10_events"]:
        bd = pd.Timestamp(b["bottom"])
        cb = float(pf.at[bd, "SOXX_Close"])
        r = {"bottom": iso(bd), "B15": b["B15"], "close": round(cb, 2), "seibro_dates": b["unit_seibro_dates"],
             "merged": b["merged"]}
        # 결제: 묶음 마지막 SEIBRO 날짜 다음 KSD 영업일 09:00 KST
        kd = pd.Timestamp(b["known_kst_unit"].split()[0])
        e = session_on_or_after(S, kd)
        r.update({"settle_known_kst": b["known_kst_unit"], "settle_entry": iso(e), "settle_k": int(pos[e] - pos[bd]),
                  "settle_gain_open": float(pf.at[e, "SOXX_Open"] / cb - 1),
                  "settle_gain_close": float(pf.at[e, "SOXX_Close"] / cb - 1),
                  "settle_min_close_to_entry": float(pf.loc[bd:e, "SOXX_Close"].iloc[1:].min() / cb - 1)
                  if pos[e] > pos[bd] else np.nan})
        assert abs(r["settle_gain_open"] - b["gain_to_entry_open_unit"]) < 6e-4, "events.json 진입 상승과 다름"
        # 보관: 바닥 거래일을 처음 담는 보관 날짜 D, 둘째 KSD 영업일
        c = soxl[soxl["trade_date"] >= bd].iloc[0]
        ce = session_on_or_after(S, c["known_kst_date"])
        r.update({"cust_date": iso(c["custody_date"]), "cust_trade": iso(c["trade_date"]), "cust_lag": int(c["lag_used"]),
                  "cust_known_kst": iso(c["known_kst_date"]) + " 09:00 KST", "cust_entry": iso(ce),
                  "cust_k": int(pos[ce] - pos[bd]),
                  "cust_gain_open": float(pf.at[ce, "SOXX_Open"] / cb - 1),
                  "cust_gain_close": float(pf.at[ce, "SOXX_Close"] / cb - 1)})
        rows.append(r)
    return pd.DataFrame(rows)


# ---------- 2. 지문 규칙 ----------
def daily_rules(mw, md, pf, ev, S) -> pd.DataFrame:
    pos = pd.Series(np.arange(len(S)), index=S)
    d = pd.DataFrame(index=md.index)
    d["dd60"] = pf["dd60"].reindex(d.index)
    d["state"] = d["dd60"] <= STATE_DD
    d["known_settle"] = md["known_settle"]
    mine = pd.to_datetime(pd.Series([session_on_or_after(S, k) for k in md["known_settle"]], index=md.index))
    ok = mine.notna()
    assert (mine[ok] == md.loc[ok, "entry_settle"]).all(), "entry_settle와 다름"
    d["entry"] = md["entry_settle"]                         # 가격 끝 뒤 진입(미국 10/30 → 11/04)은 수익 없음
    for fp in FPS:
        s = mw[(mw["measure"] == fp["measure"]) & (mw["L"] == fp["L"])].set_index("date")
        d[f"{fp['id']}_Eexp"] = s["E_exp"].reindex(d.index)
        d[f"{fp['id']}_P"] = s["P"].reindex(d.index)
        e = d[f"{fp['id']}_Eexp"]
        sig = e >= ZMIN if fp["side"] == "+" else e <= -ZMIN
        d[f"{fp['id']}_on"] = np.where(e.notna(), sig & d["state"], np.nan)
    op, cl = pf["SOXX_Open"], pf["SOXX_Close"]
    for h in HS:
        out = []
        for e in d["entry"]:
            j = pos[e] + h - 1 if e in pos.index else None
            out.append(cl.iloc[j] / op.loc[e] - 1 if j is not None and j < len(S) else np.nan)
        d[f"R{h}"] = out
    # B10 바닥까지 세션 거리
    bpos = np.array([pos[pd.Timestamp(b)] for b in ev["B10"]])
    d["dist_B10"] = [int(np.min(np.abs(bpos - pos[t]))) for t in d.index]
    # 하락 구간 번호: 상태 날이 GAP 세션 넘게 끊기면 새 구간
    ep, cur, last = [], 0, None
    for t, st in zip(d.index, d["state"]):
        if st:
            if last is None or pos[t] - last > GAP:
                cur += 1
            last = pos[t]
            ep.append(cur)
        else:
            ep.append(0)
    d["state_ep"] = ep
    # 하락 국면(10% 고점 → B10 바닥, events.json episodes). 국면 밖(반등 쪽)은 ""
    lab = pd.Series("", index=d.index)
    for e in ev["episodes"]:
        lab[(d.index >= pd.Timestamp(e["start"])) & (d.index <= pd.Timestamp(e["end"]))] = e["episode"]
    d["dd_ep"] = lab
    d["year"] = d.index.year
    d["period"] = np.where(d["year"] <= 2025, "2023~25", "2026")
    return d


def shift_null(on: np.ndarray, r: np.ndarray, base_mean: float) -> tuple[float, float, int]:
    """2026 안 원형 이동: 켜짐 표지를 1..n−1 칸 밀어 (켜진 날 평균 − 기준 평균)의 분포."""
    n = len(on)
    obs = r[on].mean() - base_mean
    ds = np.array([r[np.roll(on, s)].mean() - base_mean for s in range(1, n)])
    return float(np.mean(ds >= obs - 1e-12)), float(np.mean(np.abs(ds - ds.mean()) >= abs(obs - ds.mean()) - 1e-12)), n - 1


def rule_stats(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fp in FPS:
        on_col = f"{fp['id']}_on"
        for per in ("2023~25", "2026"):
            x = d[(d["period"] == per) & d[on_col].notna()]
            on = x[on_col] == 1
            for h in HS:
                y = x[x[f"R{h}"].notna()]
                o = y[y[on_col] == 1]
                st = y[y["state"]]
                r = {"fp": fp["id"], "rule": fp["name"], "period": per, "h": h,
                     "days_eval": len(x), "first_eval": iso(x.index.min()), "last_eval": iso(x.index.max()),
                     "state_days": int(x["state"].sum()), "firings": int(on.sum()),
                     "episodes": int(x.loc[on, "state_ep"].nunique()),
                     "episodes_state": int(x.loc[x["state"], "state_ep"].nunique()),
                     "dd_episodes": int(x.loc[on & (x["dd_ep"] != ""), "dd_ep"].nunique()),
                     "on_outside_dd": int((on & (x["dd_ep"] == "")).sum()),
                     "near_B10_share": float((x.loc[on, "dist_B10"] <= NEAR).mean()) if on.any() else np.nan,
                     "near_B10_state_share": float((x.loc[x["state"], "dist_B10"] <= NEAR).mean()) if x["state"].any() else np.nan,
                     "n_on": len(o), "mean_on": o[f"R{h}"].mean(), "med_on": o[f"R{h}"].median(),
                     "win_on": (o[f"R{h}"] > 0).mean() if len(o) else np.nan,
                     "n_state": len(st), "mean_state": st[f"R{h}"].mean(), "win_state": (st[f"R{h}"] > 0).mean(),
                     "n_all": len(y), "mean_all": y[f"R{h}"].mean(), "win_all": (y[f"R{h}"] > 0).mean()}
                r["diff_state"] = r["mean_on"] - r["mean_state"]
                r["diff_all"] = r["mean_on"] - r["mean_all"]
                if per == "2026" and len(o) > 0:
                    # 상태 날 순서 안에서 이동(켜짐 vs 상태), 모든 날 순서 안에서 이동(켜짐 vs 모든 날)
                    sv = st[on_col].values == 1
                    r["null_state_p_up"], r["null_state_p_2s"], r["null_state_shifts"] = shift_null(
                        sv, st[f"R{h}"].values, st[f"R{h}"].mean())
                    av = y[on_col].values == 1
                    r["null_all_p_up"], r["null_all_p_2s"], r["null_all_shifts"] = shift_null(
                        av, y[f"R{h}"].values, y[f"R{h}"].mean())
                rows.append(r)
    return pd.DataFrame(rows)


# ---------- 3. 지금 ----------
def anchor_vals(mw, S, anchors, cells) -> pd.DataFrame:
    """기준일(k = 0)마다 칸의 창 끝 값(P·E_full·E_exp)."""
    pos = pd.Series(np.arange(len(S)), index=S)
    key = mw.set_index(["date", "L", "measure"])[["P", "E_full", "E_exp"]]
    rows = []
    for a in anchors:
        i = int(pos[pd.Timestamp(a)])
        for m, w in cells:
            ke, L = WINS[w]
            j = i + ke
            end = S[j] if 0 <= j < len(S) else None
            v = key.loc[(end, L, m)] if end is not None and (end, L, m) in key.index else None
            rows.append({"anchor": pd.Timestamp(a), "measure": m, "window": w, "end": end,
                         "P": np.nan if v is None else v["P"], "E_full": np.nan if v is None else v["E_full"],
                         "E_exp": np.nan if v is None else v["E_exp"]})
    return pd.DataFrame(rows)


def side_hit(v: float, norm: str, side: str) -> bool:
    if pd.isna(v):
        return False
    if norm == "P":
        return v >= HI if side == "hi" else v <= LO
    return (v >= ZMIN) if side == "+" else (v <= -ZMIN)


def now_table(mw, md, ev, S, com: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    last = md.index.max()                                   # 마지막 흐름 거래일(미국 10/30)
    last_c = md.index[~md["c_stale"]].max()                 # 앞 보관이 아닌 마지막 거래일(10/29)
    f = com[com["flag"].notna()].copy()
    cells = (f.groupby(["measure", "window", "norm", "side"])
             .agg(sets=("set", lambda s: "+".join(sorted(s, reverse=True))), name=("name", "first"),
                  echo=("echo", "first"), F_share=("F_share", "max")).reset_index())
    cells["kind"] = "공통"
    fpx = pd.DataFrame([{"measure": fp["measure"], "window": fp["window"], "norm": "E", "side": fp["side"],
                         "sets": fp["id"], "name": fp["name"], "echo": fp["src"], "F_share": np.nan, "kind": "지문"}
                        for fp in FPS])
    cells = pd.concat([fpx, cells], ignore_index=True)
    mw_cells = sorted(set(zip(cells["measure"], cells["window"])))
    B15 = [pd.Timestamp(b) for b in ev["B15"]]
    B10 = [pd.Timestamp(b) for b in ev["B10"]]
    F = [pd.Timestamp(x["date"]) for x in ev["F"]]
    av = anchor_vals(mw, S, B15 + [b for b in B10 if b not in B15] + F, mw_cells)
    key = mw.set_index(["date", "L", "measure"])[["P", "E_full", "E_exp", "n_cens", "n_merged"]]
    import exchange_calendars as xc
    FS = pd.DatetimeIndex(xc.get_calendar("XNYS").sessions_in_range(S[0], S[-1] + pd.Timedelta(days=30))).tz_localize(None)
    rows = []
    for _, c in cells.iterrows():
        m, w, norm, side = c["measure"], c["window"], c["norm"], c["side"]
        ke, L = WINS[w]
        end = last_c if m.startswith("C") else last          # 보관 측정: 10/30은 앞 보관(c_stale) → 10/29에 끝냄
        cur = key.loc[(end, L, m)]
        implied = FS[FS.get_loc(end) - ke]                  # 이 창이 '바닥 k = 0'이라면 바닥 날(가격 끝 뒤는 NYSE 달력)
        a = av[(av["measure"] == m) & (av["window"] == w)].set_index("anchor")
        r = {"kind": c["kind"], "cell": f"{m}·{w}·{norm}{'↑' if side == 'hi' else '↓' if side == 'lo' else side}",
             "measure": m, "name": c["name"], "window": w, "norm": norm, "side": side, "sets": c["sets"],
             "echo": c["echo"], "F_share_profile": c["F_share"], "win_end": iso(end), "implied_bottom": iso(implied),
             "usable_now": w in ("W1", "W2", "W3"), "n_cens": cur["n_cens"], "n_merged": cur["n_merged"]}
        for col in ("P", "E_full", "E_exp"):
            r[f"cur_{col}"] = cur[col]
            r[f"B15_med_{col}"] = a.loc[a.index.isin(B15), col].median()
            r[f"B10_med_{col}"] = a.loc[a.index.isin(B10), col].median()
            fv = a.loc[a.index.isin(F), col].dropna()
            r[f"F_med_{col}"] = fv.median()
            r[f"F_n_{col}"] = len(fv)
        main = "P" if norm == "P" else "E_full"
        v, b, fm = r[f"cur_{main}"], r[f"B15_med_{main}"], r[f"F_med_{main}"]
        r["cur_on_side"] = side_hit(v, norm, side)
        r["cur_on_side_Eexp"] = side_hit(r["cur_E_exp"], "E", side) if norm == "E" else np.nan
        r["closer_to"] = ("B15" if abs(v - b) < abs(v - fm) else "F") if pd.notna(v) else ""
        # 기울기: 바닥 쪽이고 F 중앙보다 B15 중앙에 가까우면 바닥 쪽
        r["lean"] = ("바닥 쪽" if r["cur_on_side"] and r["closer_to"] == "B15" else
                     "바닥 쪽(F와 비슷)" if r["cur_on_side"] else
                     "F 쪽" if r["closer_to"] == "F" else "어느 쪽도 아님")
        rows.append(r)
    nt = pd.DataFrame(rows)
    # QA: profile commonality의 B15 중앙(P 또는 E_full)과 같아야 함
    qa = []
    for _, c in f[f["set"] == "B15"].iterrows():
        x = nt[(nt["kind"] == "공통") & (nt["measure"] == c["measure"]) & (nt["window"] == c["window"])
               & (nt["norm"] == c["norm"])]
        col = "B15_med_P" if c["norm"] == "P" else "B15_med_E_full"
        qa.append(abs(float(x[col].iloc[0]) - float(c["med"])))
    info = {"last_flow": iso(last), "last_cust_trade": iso(last_c), "qa_B15_med_maxdiff": float(np.max(qa)) if qa else np.nan, "n_cells": len(nt)}
    return nt, info


# ---------- 쓰기 ----------
def pct(x, d=1) -> str:
    return "" if pd.isna(x) else f"{100 * x:+.{d}f}%"


def f2(x, d=2) -> str:
    return "" if pd.isna(x) else f"{x:+.{d}f}"


def write_rules_md(rs: pd.DataFrame, d: pd.DataFrame, ev: dict) -> list[str]:
    BOT = [pd.Timestamp(b) for b in ev["B10"]]
    FDAYS = [pd.Timestamp(x["date"]) for x in ev["F"]]
    path = OUT / "rules.md"
    pre = path.read_text(encoding="utf-8")
    assert MARK in pre, "사전 등록 절이 없음"
    pre = pre.split(MARK)[0] + MARK + "\n"
    L = ["# 결과 (설명용, `code/timing.py`)", ""]
    for fp in FPS:
        L.append(f"## {fp['id']} {fp['name']} — {fp['src']}")
        x = rs[rs["fp"] == fp["id"]]
        r0 = x[x["h"] == 5].set_index("period")
        L.append("| 구간 | 판정 날(첫~끝) | 상태 날 | 켜짐 | 상태 구간(켜짐/상태) | 하락 국면 수 · 국면 밖 켜짐 | 바닥 ±5 비율(켜짐/상태) |")
        L.append("|---|---|---|---|---|---|---|")
        for per in ("2023~25", "2026"):
            r = r0.loc[per]
            L.append(f"| {per} | {r['days_eval']}({r['first_eval']}~{r['last_eval']}) | {r['state_days']} | {r['firings']} | "
                     f"{r['episodes']}/{r['episodes_state']} | {r['dd_episodes']} · {r['on_outside_dd']} | {pct(r['near_B10_share'], 0)[1:]}/{pct(r['near_B10_state_share'], 0)[1:]} |")
        L.append("")
        L.append("| 구간 | h | 켜짐 n · 평균 · 오른 비율 | 상태 n · 평균 · 오른 비율 | 모든 날 n · 평균 | 켜짐−상태 | 켜짐−모든 날 | 2026 이동 귀무 p(상태·위쪽 / 모든 날·위쪽) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for _, r in x.iterrows():
            nl = (f"{r['null_state_p_up']:.2f} / {r['null_all_p_up']:.2f} ({int(r['null_state_shifts'])}·{int(r['null_all_shifts'])}이동)"
                  if pd.notna(r.get("null_state_p_up", np.nan)) else "")
            L.append(f"| {r['period']} | {r['h']} | {r['n_on']} · {pct(r['mean_on'])} · {pct(r['win_on'], 0)[1:]} | "
                     f"{r['n_state']} · {pct(r['mean_state'])} · {pct(r['win_state'], 0)[1:]} | {r['n_all']} · {pct(r['mean_all'])} | "
                     f"{pct(r['diff_state'])} | {pct(r['diff_all'])} | {nl} |")
        on = d[d[f"{fp['id']}_on"] == 1]
        L.append("")
        L.append("| 켜짐 묶음(상태 구간) | 첫~끝 | 켜짐 | R10 평균 | 같은 구간 상태 날 R10 평균 |")
        L.append("|---|---|---|---|---|")
        for ep_id, g in on.groupby("state_ep"):
            stg = d[(d["state_ep"] == ep_id) & d[f"{fp['id']}_on"].notna()]
            L.append(f"| {ep_id} | {iso(g.index.min())}~{iso(g.index.max())} | {len(g)} | {pct(g['R10'].mean())} | {pct(stg['R10'].mean())} |")
        L.append("")
        for lab, days in (("B10 바닥 날", BOT), ("F 날", FDAYS)):
            z = d.loc[d.index.isin(days), f"{fp['id']}_on"].dropna()
            L.append(f"- {lab} 켜짐(E_exp 있는 날만): {int((z == 1).sum())}/{len(z)}"
                     + (f" — {', '.join(iso(t)[2:] for t in z[z == 1].index)}" if (z == 1).any() else ""))
        L.append(f"- 켜진 날(미국 t): {', '.join(iso(t)[2:] for t in on.index)}")
        L.append("")
    both = d[(d["FP1_on"] == 1) & (d["FP2_on"] == 1)]
    L.append(f"- 두 지문이 같은 날 켜진 날: {len(both)}개 {', '.join(iso(t) for t in both.index)}. 둘을 합친 규칙은 등록하지 않아 평가하지 않는다.")
    L.append("- 읽기: 수익은 SOXX 시가 진입 기준. 켜진 날끼리 수익 창이 겹쳐 날 수가 독립 표본 수가 아니다. 이동 귀무 p = 켜짐 표지를 2026 안에서 원형으로 민 모든 경우 중 차이가 관측 이상인 비율.")
    path.write_text(pre + "\n".join(L) + "\n", encoding="utf-8")
    return L


def write_now_md(nt: pd.DataFrame, d: pd.DataFrame, info: dict, tt: pd.DataFrame) -> list[str]:
    def v(r, col, norm):
        x = r[col]
        return "" if pd.isna(x) else (f"{x:.0f}" if norm == "P" else f"{x:+.2f}")
    L = ["# 지금 자리 (설명용, 판정하지 않음)", "",
         f"- 흐름 끝 미국 {info['last_flow']}(SEIBRO 11/03), 보관 끝 11/02(미국 10/29까지 담음 → 10/30 C 값은 앞 보관), 가격 끝 11/03.",
         "- 결제 창은 10/30, 보관 창(C1~C3)은 10/29에 끝난다(10/30 보관은 아직 없음). '가정 바닥' = 그 창이 바닥 창이라면 바닥 날(W2면 다음 세션 11/02, W3면 10/30). 10/30은 이미 F 날(11/02 종가가 3% 넘게 낮음)이다.",
         "- 기울기: 현재 값이 칸의 바닥 쪽(P ≥ 80/≤ 20, E 같은 부호 |z| ≥ 0.5)이고 B15 중앙에 F 중앙보다 가까우면 '바닥 쪽'. W4·W5는 바닥이 지났다는 가정이라 참고만.", "",
         "## 지문과 바닥 때까지 아는 칸(W1~W3)",
         "| 칸 | 측정 | 묶음 | 창 끝 | 지금 P · E_full · E_exp | B15 중앙 P · E | F 중앙 P · E | 기울기 | 메아리 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for _, r in nt[nt["usable_now"]].iterrows():
        L.append(f"| {r['cell']} | {r['name']} | {r['sets']} | {r['win_end'][5:]} | {v(r, 'cur_P', 'P')} · {v(r, 'cur_E_full', 'E')} · {v(r, 'cur_E_exp', 'E')} | "
                 f"{v(r, 'B15_med_P', 'P')} · {v(r, 'B15_med_E_full', 'E')} | {v(r, 'F_med_P', 'P')} · {v(r, 'F_med_E_full', 'E')} | "
                 f"{r['lean']} | {r['echo'] if isinstance(r['echo'], str) else ''} |")
    u = nt[nt["usable_now"] & (nt["kind"] == "공통")]
    cnt = u["lean"].value_counts()
    L += ["", f"- W1~W3 공통 칸 {len(u)}개: " + ", ".join(f"{k} {n}" for k, n in cnt.items()) + "."]
    w45 = nt[~nt["usable_now"]]
    L += [f"- W4·W5 칸 {len(w45)}개(가정 바닥 W4 {w45.loc[w45['window'] == 'W4', 'implied_bottom'].iloc[0] if (w45['window'] == 'W4').any() else '-'}, "
          f"W5 {w45.loc[w45['window'] == 'W5', 'implied_bottom'].iloc[0] if (w45['window'] == 'W5').any() else '-'}): "
          + ", ".join(f"{k} {n}" for k, n in w45["lean"].value_counts().items()) + ". W5 가정 바닥은 10/05 고점보다 앞이라 뜻이 없고, W4도 바닥이 지났다는 근거가 아니다.", ""]
    L += ["## 지문 최근 10거래일", "| 미국 t | dd60 | FP1 E_exp | FP1 | FP2 E_exp | FP2 | 아는 날(KST 09:00) | 진입 |", "|---|---|---|---|---|---|---|---|"]
    for t, r in d.tail(10).iterrows():
        L.append(f"| {iso(t)} | {r['dd60']:+.3f} | {f2(r['FP1_Eexp'])} | {'켜짐' if r['FP1_on'] == 1 else '-'} | "
                 f"{f2(r['FP2_Eexp'])} | {'켜짐' if r['FP2_on'] == 1 else '-'} | {iso(r['known_settle'])} | {iso(r['entry'])} |")
    L += ["", "## 아직 없는 것",
          "| 미국 거래일 | SOXX 종가 | SEIBRO 날짜 | 규약상 아는 때 | 진입 시가 |", "|---|---|---|---|---|",
          "| 2026-11-02 | 453.65(7/29 465.00 아래) | 11/04 | 11/05 09:00 KST(공개는 11/05 00:10 KST 무렵) | 미국 11/05 |",
          "| 2026-11-03 | 465.00 | 11/05 | 11/06 09:00 KST | 미국 11/06 |",
          "- 11/02가 바닥이라면 그날 흐름은 바닥 뒤 3세션(11/05 시가)에 안다. 지난 B10 11개도 3~4세션 뒤였다(`timing.csv`).",
          "- 11/02의 W3 값, 10/27~11/02 창(W2를 11/03 바닥에 맞춘 것), 보관 10/30·11/02분은 아직 없다. 지금 표는 '11/02가 바닥이라면 W2'와 '10/30이 바닥이라면 W3'만 보여 준다."]
    (OUT / "now.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return L


def write_summary(tt, rs, nt, d, info, ev, pf) -> None:
    g = tt
    f1 = rs[(rs["fp"] == "FP1") & (rs["period"] == "2026") & (rs["h"] == 10)].iloc[0]
    f2_ = rs[(rs["fp"] == "FP2") & (rs["period"] == "2026") & (rs["h"] == 10)].iloc[0]
    n_sep = int(((d["FP1_on"] == 1) & (d["dd_ep"] == "2026-09-14")).sum())
    BOT = [pd.Timestamp(b) for b in ev["B10"]]
    FD = [pd.Timestamp(x["date"]) for x in ev["F"]]
    bf = {}
    for fp in ("FP1", "FP2"):
        zb = d.loc[d.index.isin(BOT), f"{fp}_on"].dropna()
        zf = d.loc[d.index.isin(FD), f"{fp}_on"].dropna()
        bf[fp] = (f"{int((zb == 1).sum())}/{len(zb)}", f"{int((zf == 1).sum())}/{len(zf)}")
    curF = [pd.Timestamp(x["date"]) for x in ev["F"] if x["episode"] == "current"]
    curF_on = [t for t in curF if t in d.index and d.at[t, "FP2_on"] == 1]
    L = ["# timing 갈래 요약 — 언제 알았나 · 쓸모 · 지금 (설명용)", "",
         "**한 줄**: 바닥 날 흐름은 바닥 뒤 3~4세션 아침에야 알고, 그때 SOXX는 바닥 종가보다 이미 "
         f"{100 * g['settle_gain_open'].min():.1f}~{100 * g['settle_gain_open'].max():.1f}%(중앙 {100 * g['settle_gain_open'].median():.1f}%) 올라 있다. "
         f"미리 고른 지문 2개 중 FP1(직전 5일 SOXL 매도가 메아리보다 많음)은 2026에도 같은 상태 날보다 높았지만(R10 {pct(f1['diff_state'])}) "
         f"2026 이동 귀무 p {f1['null_state_p_up']:.2f}이고 켜짐 {int(f1['n_on'])}날 중 {n_sep}날이 9/14 국면 한 곳이다. FP2는 2026에 반대로 갔다(R10 {pct(f2_['diff_state'])}). "
         "쓸 수 있는 신호라는 근거는 없다. 지금 자리는 판정하지 않는다.", "",
         "## 1. 아는 시점 (B10 11개, `timing.csv`)",
         "| 바닥 | 결제 아는 날 | 진입 k | 진입 시가 상승 | 진입 종가 상승 | 보관 아는 날 | 보관 k | 보관 진입 시가 상승 |",
         "|---|---|---|---|---|---|---|---|"]
    for _, r in g.iterrows():
        L.append(f"| {r['bottom']}{'*' if r['B15'] else ''} | {r['settle_known_kst'][:10]} | {r['settle_k']} | {pct(r['settle_gain_open'])} | "
                 f"{pct(r['settle_gain_close'])} | {r['cust_known_kst'][:10]} | {r['cust_k']} | {pct(r['cust_gain_open'])} |")
    L += [f"- * = B15. 결제 k {g['settle_k'].min()}~{g['settle_k'].max()}(T+2 시절 4, T+1 3), 보관 k {g['cust_k'].min()}~{g['cust_k'].max()}. "
          f"진입 시가가 바닥 종가보다 낮았던 사건 {int((g['settle_gain_open'] < 0).sum())}/11.", ""]
    L += ["## 2. 지문 2개 (사전 등록 `rules.md`, 상태 dd60 ≤ −10% 안에서 켜짐, E_exp는 2024-06 뒤만 있음)",
          "| 지문 | 구간 | 켜짐/상태 날 | 상태 구간·하락 국면 | 바닥 ±5 비율(켜짐/상태) | R5 켜짐−상태 | R10 켜짐−상태 | R20 켜짐−상태 | 2026 이동 p(위쪽, R5·R10·R20) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for fp in FPS:
        for per in ("2023~25", "2026"):
            x = rs[(rs["fp"] == fp["id"]) & (rs["period"] == per)].set_index("h")
            r = x.loc[5]
            nl = " · ".join(f"{x.loc[h, 'null_state_p_up']:.2f}" for h in HS) if per == "2026" and "null_state_p_up" in x and x["null_state_p_up"].notna().all() else ""
            L.append(f"| {fp['id']} {fp['measure']}·{fp['window']}·E{fp['side']} | {per} | {r['firings']}/{r['state_days']} | {r['episodes']}·{r['dd_episodes']} | "
                     f"{pct(r['near_B10_share'], 0)[1:]}/{pct(r['near_B10_state_share'], 0)[1:]} | "
                     + " | ".join(f"{pct(x.loc[h, 'diff_state'])}(n {x.loc[h, 'n_on']})" for h in HS) + f" | {nl} |")
    L += ["- 켜짐−상태 = 켜진 날 평균 SOXX 수익 − 같은 상태 모든 날 평균. 모든 날 대비·오른 비율은 `rules.md`·`rules.csv`. 켜진 날끼리 수익 창이 겹친다.",
          f"- 바닥 대 가짜 바닥 켜짐(E_exp 있는 날): FP1 B10 {bf['FP1'][0]} · F {bf['FP1'][1]}, FP2 B10 {bf['FP2'][0]} · F {bf['FP2'][1]}. "
          f"FP2는 지금 국면 F 날 {len(curF)}개 중 {len(curF_on)}개({'·'.join(iso(t)[5:] for t in curF_on)})에서 켜졌다.", ""]
    u = nt[nt["usable_now"] & (nt["kind"] == "공통")]
    fpn = nt[nt["kind"] == "지문"]
    gv = lambda c, col: u.loc[u["cell"] == c, col].iloc[0]
    last = pd.Timestamp(info["last_flow"])
    i = pf.index.get_loc(last)
    r1030 = pf["SOXX_Close"].iloc[i] / pf["SOXX_Close"].iloc[i - 1] - 1
    L += ["## 3. 지금 (`now.md`, 흐름 끝 미국 10/30 = 이미 F 날)",
          f"- W1~W3 공통 칸 {len(u)}개: " + ", ".join(f"{k} {n}" for k, n in u["lean"].value_counts().items()) + ".",
          "- 지문: " + "; ".join(f"{r['sets']} E_full {f2(r['cur_E_full'])}·E_exp {f2(r['cur_E_exp'])}(B15 중앙 {f2(r['B15_med_E_full'])}, F 중앙 {f2(r['F_med_E_full'])}) → {r['lean']}"
                                 for _, r in fpn.iterrows()) + ".",
          "- profile이 메아리를 넘는다고 본 B10 E 공통 3칸(E_full): " + "; ".join(
              f"{nm} {f2(gv(c, 'cur_E_full'))}(B15 {f2(gv(c, 'B15_med_E_full'))}, F {f2(gv(c, 'F_med_E_full'))}) {gv(c, 'lean')}"
              for c, nm in (("X3·W3·E+", "당일 SOXL 매도"), ("X3·W2·E+", "직전 5일 SOXL 매도"), ("R5·W3·E-", "당일 인컴 순매수")))
          + f". 10/30은 SOXX {pct(r1030)} 급락일이라 당일 E+는 급락일 치우침과 섞인다(profile: 큰 하락일 X3 E 평균 +0.76).",
          f"- FP2가 '바닥 쪽'이어도 2026에는 반대로 갔고 지금 국면 F 날 {len(curF)}개 중 {len(curF_on)}개에서 켜졌다. 기울기는 판정이 아니다.",
          "- 빠진 것: 미국 11/02(SOXX 453.65, 7/29 아래) 흐름은 SEIBRO 11/04 → 11/05 09:00 KST, 11/03은 11/06 09:00 KST에 안다. 보관 10/30·11/02분도 아직 없다.", ""]
    L += ["## 4. 본 범위 · 주의",
          f"- 시점 표 11사건. 지문 후보 15칸(profile E 10 + contrast 5)에서 2개를 식으로 골랐다(고르기 전 profile·contrast 요약을 읽음, 식은 2023~25 열만 씀). "
          f"날마다 판정 {len(d)}날 중 E_exp 있는 날만. 지금 표 {info['n_cells']}칸(지문 2 + 공통 칸).",
          "- 지문 고르기의 2023~25 근거(E_full)는 2026까지 넣은 전체 적합이다. 날마다 판정은 확장 적합 E_exp라 2023~25 성과도 겉보기 표본 밖이지만, 칸 고르기는 그 해 사건을 보고 했다.",
          "- 2023~25 판정은 E_exp가 생긴 2024-06 뒤만이다(B10 5개: 2024-08-07·09-06·11-15, 2025-04-08·11-20).",
          "- 바닥·F는 사후 표지다. 시험이 아니라 설명용이고, 매수 추천·주문 계산이 아니다."]
    assert len(L) <= 60, f"요약 {len(L)}줄"
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    mw, md, pf, cu, ev = load()
    S = pf.index
    tt = timing_table(ev, pf, cu, S)
    tt.to_csv(OUT / "timing.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    d = daily_rules(mw, md, pf, ev, S)
    rs = rule_stats(d)
    rs.to_csv(OUT / "rules.csv", index=False, encoding="utf-8-sig", float_format="%.5f")
    dd = d.copy()
    dd.index.name = "date"
    dd.to_csv(OUT / "rules_daily.csv", encoding="utf-8-sig", float_format="%.5f")
    write_rules_md(rs, d, ev)
    com = pd.read_csv(PROF / "commonality.csv")
    nt, info = now_table(mw, md, ev, S, com)
    assert info["qa_B15_med_maxdiff"] < 1e-6, f"B15 중앙 불일치 {info['qa_B15_med_maxdiff']}"
    nt.to_csv(OUT / "now.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    write_now_md(nt, d, info, tt)
    write_summary(tt, rs, nt, d, info, ev, pf)
    print(json.dumps(info, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
