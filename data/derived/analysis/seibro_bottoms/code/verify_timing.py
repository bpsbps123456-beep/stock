"""timing 갈래 반박 검증 — derived/만 읽어 다시 계산한다(timing.py 코드는 쓰지 않음).

실행: py -B code/verify_timing.py   출력: 화면 요약(짧게). 해석은 results/verify_timing.md
"""
from __future__ import annotations

import json
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
DER = STUDY / "derived"
WINS = {"W1": (-6, 15), "W2": (-1, 5), "W3": (0, 1)}  # (창 끝 k, 길이)
OUT: dict = {}


def load():
    md = pd.read_parquet(DER / "measures_daily.parquet").set_index("date")
    mw = pd.read_parquet(DER / "measures_win.parquet")
    pf = pd.read_parquet(DER / "price_feats.parquet").set_index("date")
    ev = json.load(open(DER / "events.json", encoding="utf-8"))
    return md, mw, pf, ev


# ---------- 1. 아는 시점 ----------
def timing(md, pf, ev):
    S = pf.index
    pos = pd.Series(np.arange(len(S)), index=S)
    nyse = xc.get_calendar("XNYS")
    rows = []
    for b in ev["B10_events"]:
        t = pd.Timestamp(b["bottom"])
        sd = md.at[t, "seibro_date"]
        kn = md.at[t, "known_settle"]
        e = S[S >= kn][0]
        cb = pf.at[t, "SOXX_Close"]
        # 공개 약 00:10 KST(다음 평일) = 미국 날짜로 SEIBRO 날짜 D의 장중(월~목 D). D가 NYSE 세션이고 다음 날이 평일이면 D 종가 진입 가능
        nxt_wd = sd + pd.Timedelta(days=1)
        early = sd if (nyse.is_session(sd) and nxt_wd.weekday() < 5) else None
        r = {"bottom": b["bottom"], "B15": b["B15"], "seibro": str(sd.date()), "known": str(kn.date()),
             "entry": str(e.date()), "k": int(pos[e] - pos[t]),
             "gain_open": round(pf.at[e, "SOXX_Open"] / cb - 1, 4),
             "gain_close": round(pf.at[e, "SOXX_Close"] / cb - 1, 4)}
        if early is not None and early in pos.index:
            r.update({"early_session": str(early.date()), "early_k": int(pos[early] - pos[t]),
                      "early_gain_close": round(pf.at[early, "SOXX_Close"] / cb - 1, 4)})
        rows.append(r)
    tt = pd.DataFrame(rows)
    OUT["timing"] = tt.to_dict("records")
    print("[1] k:", tt["k"].tolist(), "gain_open min/med/max",
          tt["gain_open"].min(), tt["gain_open"].median(), tt["gain_open"].max(), "neg", int((tt["gain_open"] <= 0).sum()))
    if "early_k" in tt:
        x = tt.dropna(subset=["early_k"])
        print("    early(00:10 KST 공개 가정) n", len(x), "k", x["early_k"].astype(int).tolist(),
              "gain_close min/med/max", x["early_gain_close"].min(), x["early_gain_close"].median(), x["early_gain_close"].max())
    return tt


# ---------- 2. 지문 ----------
def resid_exp(y, X, minn=250, reg_min=0):
    """확장 적합 잔차. reg_min>0이면 훈련 안 행 수가 reg_min 미만인 체제 더미는 0으로 둔다(그 체제를 앞 체제에 합침)."""
    out = np.full(len(y), np.nan)
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    idx = np.flatnonzero(ok)
    regc = [j for j in range(X.shape[1]) if j >= 5]
    for k in range(minn, len(idx)):
        tr, i = idx[:k], idx[k]
        Xt, xi = X[tr].copy(), X[i].copy()
        if reg_min:
            for j in regc:
                if Xt[:, j].sum() < reg_min:
                    Xt[:, j] = 0.0
                    xi[j] = 0.0
        b, _, rank, _ = np.linalg.lstsq(Xt, y[tr], rcond=None)
        e = y[tr] - Xt @ b
        sd = np.sqrt(e @ e / (len(tr) - rank))
        out[i] = (y[i] - xi @ b) / sd
    return out


def design(pf, idx, L):
    X = pd.DataFrame(index=idx)
    X["c"] = 1.0
    X["ret"] = pf[f"ret_{L}"].reindex(idx)
    X["pre"] = pf[f"pre20_{L}"].reindex(idx)
    X["dd"] = pf["dd60"].reindex(idx)
    X["rv"] = pf["rv20"].reindex(idx)
    reg = pf["regime"].reindex(idx)
    for k in range(1, 5):
        X[f"r{k}"] = (reg == k).astype(float)
    return X


def shift_p(on, r):
    base = r.mean()
    obs = r[on].mean() - base
    ds = np.array([r[np.roll(on, s)].mean() - base for s in range(1, len(on))])
    return float(np.mean(ds >= obs - 1e-12))


def rules(md, mw, pf, ev):
    S = pf.index
    pos = pd.Series(np.arange(len(S)), index=S)
    d = pd.DataFrame(index=md.index)
    d["dd60"] = pf["dd60"].reindex(d.index)
    d["state"] = d["dd60"] <= -0.10
    ent = [S[S >= k][0] if (S >= k).any() else pd.NaT for k in md["known_settle"]]
    d["entry"] = ent
    op, cl = pf["SOXX_Open"], pf["SOXX_Close"]
    for h in (5, 10, 20):
        d[f"R{h}"] = [cl.iloc[pos[e] + h - 1] / op[e] - 1 if pd.notna(e) and pos[e] + h - 1 < len(S) else np.nan
                      for e in d["entry"]]
    bpos = np.array([pos[pd.Timestamp(b)] for b in ev["B10"]])
    d["distB"] = [int(np.min(np.abs(bpos - pos[t]))) for t in d.index]
    lab = pd.Series("", index=d.index)
    for e in ev["episodes"]:
        lab[(d.index >= pd.Timestamp(e["start"])) & (d.index <= pd.Timestamp(e["end"]))] = e["episode"]
    d["ep"] = lab
    Fd = set(pd.Timestamp(x["date"]) for x in ev["F"])
    res = {}
    specs = {"FP1": ("X3", 5, +1), "FP2": ("L1", 1, -1)}
    for fid, (m, L, sg) in specs.items():
        s = mw[(mw["measure"] == m) & (mw["L"] == L)].set_index("date").reindex(d.index)
        X = design(pf, d.index, L)
        mine = resid_exp(s["y"].values, X.values)
        dif = np.nanmax(np.abs(mine - s["E_exp"].values))
        alt = {"reg60": resid_exp(s["y"].values, X.values, reg_min=60),
               "noreg": resid_exp(s["y"].values, X.values[:, :5])}
        for tag, ee in [("panel", s["E_exp"].values)] + list(alt.items()):
            on = np.where(np.isfinite(ee), ((sg * ee) >= 0.5) & d["state"].values, np.nan)
            col = f"{fid}_{tag}"
            d[col] = on
            for per, msk in (("2023~25", d.index.year <= 2025), ("2026", d.index.year == 2026)):
                x = d[msk & d[col].notna()]
                rr = {"days": len(x), "state": int(x["state"].sum()), "on": int((x[col] == 1).sum())}
                for h in (5, 10, 20):
                    y = x[x[f"R{h}"].notna()]
                    o, st = y[y[col] == 1], y[y["state"]]
                    rr[f"d{h}"] = round(o[f"R{h}"].mean() - st[f"R{h}"].mean(), 4) if len(o) else None
                    rr[f"on_mean{h}"] = round(o[f"R{h}"].mean(), 4) if len(o) else None
                    if per == "2026" and len(o):
                        rr[f"p{h}"] = round(shift_p(st[col].values == 1, st[f"R{h}"].values), 3)
                if tag == "panel":
                    onx = x[x[col] == 1]
                    rr["near5"] = round(float((onx["distB"] <= 5).mean()), 3) if len(onx) else None
                    rr["near5_state"] = round(float((x.loc[x["state"], "distB"] <= 5).mean()), 3)
                    rr["by_ep"] = onx.groupby("ep").size().to_dict()
                    bt = [t for t in ev["B10"] if pd.Timestamp(t) in x.index]
                    rr["B10_on"] = f"{int(sum(x.at[pd.Timestamp(t), col] == 1 for t in bt))}/{len(bt)}"
                    fd = [t for t in Fd if t in x.index]
                    rr["F_on"] = f"{int(sum(x.at[t, col] == 1 for t in fd))}/{len(fd)}"
                res[f"{fid}|{tag}|{per}"] = rr
        res[f"{fid}|maxdiff_Eexp"] = float(dif)
        d[f"{fid}_E"] = s["E_exp"].values
        d[f"{fid}_Ereg60"] = alt["reg60"]
    OUT["rules"] = res
    for k, v in res.items():
        if isinstance(v, dict):
            print("[2]", k, {a: v[a] for a in v if a in ("state", "on", "d5", "d10", "d20", "p5", "p10", "p20", "near5", "near5_state", "B10_on", "F_on")})
        else:
            print("[2]", k, v)
    return d


def rules_extra(d, pf, ev):
    """FP1: 고른 사건(2023~25 B10) ±5 뺀 성과, 2026 국면별, 가격만 회귀."""
    sel = [pd.Timestamp(b) for b in ev["B10"] if pd.Timestamp(b).year <= 2025]
    S = pf.index
    pos = pd.Series(np.arange(len(S)), index=S)
    sp = np.array([pos[b] for b in sel])
    d["distSel"] = [int(np.min(np.abs(sp - pos[t]))) for t in d.index]
    out = {}
    x = d[(d.index.year <= 2025) & d["FP1_panel"].notna()]
    o = x[x["FP1_panel"] == 1]
    out["FP1_2325_on_near_sel"] = f"{int((o['distSel'] <= 5).sum())}/{len(o)}"
    xo = x[x["distSel"] > 5]
    for h in (5, 10, 20):
        y = xo[xo[f"R{h}"].notna()]
        oo, st = y[y["FP1_panel"] == 1], y[y["state"]]
        out[f"FP1_2325_exSel_d{h}"] = (len(oo), round(oo[f"R{h}"].mean() - st[f"R{h}"].mean(), 4) if len(oo) else None)
    # 2026 국면별(켜짐 평균 − 같은 국면 상태 날 평균)
    x6 = d[(d.index.year == 2026) & d["FP1_panel"].notna()]
    for ep, g in x6[x6["state"]].groupby("ep"):
        on = g[g["FP1_panel"] == 1]
        if len(on):
            out[f"FP1_2026_ep_{ep or 'none'}"] = (len(on), len(g), round(on["R10"].mean() - g["R10"].mean(), 4))
    # 2026 상태 날: R10 ~ 켜짐 + dd60 + ret_5 + rv20 (가격 보정 뒤에도 남나, 설명용)
    y = x6[x6["state"] & x6["R10"].notna()].copy()
    Z = np.column_stack([np.ones(len(y)), (y["FP1_panel"] == 1).astype(float), y["dd60"],
                         pf["ret_5"].reindex(y.index), pf["rv20"].reindex(y.index)])
    b = np.linalg.lstsq(Z, y["R10"].values, rcond=None)[0]
    out["FP1_2026_R10_ols_on_coef"] = round(float(b[1]), 4)
    # 첫 켜짐만(같은 상태 구간 안 연속 켜짐 묶음의 첫날): 독립 표본에 가깝게
    on6 = x6[x6["FP1_panel"] == 1]
    gaps = np.diff([pos[t] for t in on6.index])
    first = on6.index[np.r_[True, gaps > 5]]
    out["FP1_2026_firsts"] = [str(t.date()) for t in first]
    out["FP1_2026_firsts_R10"] = [round(float(v), 4) for v in d.loc[first, "R10"]]
    # 체제 더미: 2026 켜짐 날마다 그 체제가 시작된 뒤 세션 수
    br = {3: pd.Timestamp("2026-05-22"), 4: pd.Timestamp("2026-07-31"), 2: pd.Timestamp("2025-12-15")}
    reg = pf["regime"]
    out["FP1_2026_on_regime_age"] = {str(t.date()): int(pos[t] - pos[S[S >= br[int(reg[t])]][0]]) for t in on6.index
                                     if int(reg[t]) in br}
    # 체제 더미 60행 규칙으로 바꾼 E 값 비교(2026 켜짐 날)
    out["FP1_2026_on_E_vs_reg60"] = {str(t.date()): (round(float(d.at[t, 'FP1_E']), 2), round(float(d.at[t, 'FP1_Ereg60']), 2))
                                     for t in on6.index}
    # FP2: 지금 국면 F 날
    cur = [pd.Timestamp(x["date"]) for x in ev["F"] if x["episode"] == "current"]
    out["FP2_current_F"] = {str(t.date()): (round(float(d.at[t, 'FP2_E']), 2), d.at[t, 'FP2_panel']) for t in cur if t in d.index}
    OUT["rules_extra"] = out
    for k, v in out.items():
        print("[2x]", k, v)


# ---------- 3. 지금 ----------
def now(mw, pf, ev):
    nt = pd.read_csv(STUDY / "results" / "timing" / "now.csv")
    nt = nt[nt["window"].isin(WINS)]
    S = pf.index
    pos = pd.Series(np.arange(len(S)), index=S)
    key = mw.set_index(["date", "L", "measure"])[["P", "E_full"]]
    B15 = [pd.Timestamp(b) for b in ev["B15"]]
    B10 = [pd.Timestamp(b) for b in ev["B10"]]
    F = [(pd.Timestamp(x["date"]), x["episode"]) for x in ev["F"]]
    last = pd.Timestamp("2026-10-30")

    def val(end, L, m, col):
        try:
            return float(key.loc[(end, L, m), col])
        except KeyError:
            return np.nan

    def anchor(a, w, L, m, col):
        ke = WINS[w][0]
        j = pos[a] + ke
        return val(S[j], L, m, col)

    def onside(v, norm, side):
        if not np.isfinite(v):
            return False
        if norm == "P":
            return v >= 80 if side == "hi" else v <= 20
        return v >= 0.5 if side == "+" else v <= -0.5

    cells = nt[nt["kind"] == "공통"]  # 공통 36칸(지문 2칸 제외)

    def score(cur_end, excl_b=None, excl_ep=None):
        """cur_end에서 끝나는 창의 값으로 기울기 칸 수(C는 하루 앞 보관이면 그대로 둠)."""
        c = {"바닥 쪽": 0, "바닥 쪽(F와 비슷)": 0, "F 쪽": 0, "어느 쪽도 아님": 0}
        for _, r in cells.iterrows():
            m, w, norm, side = r["measure"], r["window"], r["norm"], r["side"]
            L = WINS[w][1]
            col = "P" if norm == "P" else "E_full"
            end = cur_end
            if m.startswith("C") and cur_end == last:
                end = pd.Timestamp("2026-10-29")
            v = val(end, L, m, col)
            bm = np.nanmedian([anchor(b, w, L, m, col) for b in B15 if b != excl_b])
            fm = np.nanmedian([anchor(f, w, L, m, col) for f, ep in F if f != cur_end and ep != excl_ep])
            cl = "B15" if abs(v - bm) < abs(v - fm) else "F"
            on = onside(v, norm, side)
            lab = "바닥 쪽" if on and cl == "B15" else "바닥 쪽(F와 비슷)" if on else "F 쪽" if cl == "F" else "어느 쪽도 아님"
            c[lab] += 1
        return c

    # (a) 그대로 다시(F 중앙에 10/30 자신 포함) — 브랜치 재현
    def score_incl(cur_end):
        c = {"바닥 쪽": 0, "바닥 쪽(F와 비슷)": 0, "F 쪽": 0, "어느 쪽도 아님": 0}
        for _, r in cells.iterrows():
            m, w, norm, side = r["measure"], r["window"], r["norm"], r["side"]
            L = WINS[w][1]
            col = "P" if norm == "P" else "E_full"
            end = pd.Timestamp("2026-10-29") if m.startswith("C") else cur_end
            v = val(end, L, m, col)
            bm = np.nanmedian([anchor(b, w, L, m, col) for b in B15])
            fm = np.nanmedian([anchor(f, w, L, m, col) for f, _ in F])
            cl = "B15" if abs(v - bm) < abs(v - fm) else "F"
            on = onside(v, norm, side)
            c["바닥 쪽" if on and cl == "B15" else "바닥 쪽(F와 비슷)" if on else "F 쪽" if cl == "F" else "어느 쪽도 아님"] += 1
        return c
    res = {"n_cells": len(cells), "now_incl": score_incl(last), "now_excl_current_ep": score(last, excl_ep="current")}
    # (b) 같은 점수를 지난 바닥 날(B10)과 F 날에 매긴다(그날 자신·같은 국면 F 제외)
    epmap = {f: ep for f, ep in F}
    cb, cf = [], []
    for b in B10:
        if pos[b] - 35 < 0 or b < pd.Timestamp("2023-06-01"):
            continue
        sc = score(b, excl_b=b, excl_ep=str(b.date()))
        cb.append((str(b.date()), sc["바닥 쪽"] + sc["바닥 쪽(F와 비슷)"], sc["바닥 쪽"]))
    for f, ep in F:
        if f == last or ep == "current" or f > last:
            continue
        sc = score(f, excl_ep=ep)
        cf.append((str(f.date()), sc["바닥 쪽"] + sc["바닥 쪽(F와 비슷)"], sc["바닥 쪽"]))
    res["B10_scores(on_side,bottom_side)"] = cb
    res["F_scores(on_side,bottom_side)"] = cf
    # 큰 하락일(SOXX 하루 −5% 이하, 바닥·F 아님)도
    big = [t for t in pf.index if pf.at[t, "ret_1"] <= np.log(0.95) and pd.Timestamp("2023-06-01") <= t <= last
           and t not in B10 and t not in epmap]
    res["bigdown_scores"] = [(str(t.date()), *(lambda s: (s["바닥 쪽"] + s["바닥 쪽(F와 비슷)"], s["바닥 쪽"]))(score(t))) for t in big]
    OUT["now"] = res
    print("[3] cells", len(cells), "incl", res["now_incl"], "excl", res["now_excl_current_ep"])
    for k in ("B10_scores(on_side,bottom_side)", "F_scores(on_side,bottom_side)", "bigdown_scores"):
        a = np.array([x[1] for x in res[k]]); bb = np.array([x[2] for x in res[k]])
        print("[3]", k[:6], "n", len(a), "on_side med/min/max", np.median(a), a.min(), a.max(), "bottom_side med", np.median(bb))
    # F 중앙에 10/30 자신이 들어가는가
    print("[3] 10/30 in F:", last in epmap, "current F:", [str(f.date()) for f, ep in F if ep == "current"])


def main():
    import sys
    md, mw, pf, ev = load()
    if "--now" in sys.argv:  # 지금 표만 빨리
        now(mw, pf, ev)
        return
    timing(md, pf, ev)
    d = rules(md, mw, pf, ev)
    rules_extra(d, pf, ev)
    now(mw, pf, ev)


if __name__ == "__main__":
    main()
