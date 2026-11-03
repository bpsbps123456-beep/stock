"""보관 갈래(보유자가 한 일): SOXX 바닥(B10)·가짜 바닥(F)·지금 하락 전후 SOXL 보유 좌수, 평단(추정), 결제–보관 대조.

실행(연구 폴더에서): py -B code/custody.py
읽기만: derived/*.parquet·events.json, ../seibro_flows/raw/daily/d3(버린 날짜의 SOXL 금액 확인용)
출력(results/custody/): paths.csv, event_summary.csv, group_summary.csv, breakeven.csv, reconcile.csv,
                        reconcile_stats.csv, qa.json  (summary.md는 사람이 숫자를 보고 쓴다)

보관 → 미국 거래일 대응: 보관 D(KST)는 미국 달력 날짜 D−1까지 결제된 거래를 담는다(결제 달력은 앞선 연구 calendar_map).
자료판(PANEL §5) 지연 규칙(D 이하 마지막 NYSE 세션 − 2·3)과는 D가 미국 휴장 평일이거나 연준 휴일 언저리인 47날짜만 다르다.
대조는 reconcile_stats(휴장 날짜에서 자료판 대응은 하루 늦게 붙는다).
설명용이다. 매매 근거가 아니다.
"""
from __future__ import annotations

import bisect
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.dont_write_bytecode = True
STUDY = Path(__file__).resolve().parents[1]
DER = STUDY / "derived"
PRIOR = STUDY.parent / "seibro_flows"
OUT = STUDY / "results" / "custody"
SOXL_ISIN = "US25459W4583"
KS = range(-20, 21)
# 구간 = (시작 k, 끝 k): 창 시작 전 보관 → 창 끝 보관(BRIEF §4 C1)
SPANS = {"W1": (-21, -6), "W2": (-6, -1), "W3": (-1, 0), "W4": (0, 5), "W5": (5, 20), "post20": (0, 20)}
GAP_BINS = [-np.inf, -0.5, -0.3, -0.1, 0.0, 0.1, np.inf]
GAP_LABELS = ["< -50%", "-50~-30%", "-30~-10%", "-10~0%", "0~+10%", ">= +10%"]
sys.path.insert(0, str(PRIOR / "code"))
import calendar_map as cm  # noqa: E402  앞선 연구 결제 달력(읽기만)

HOLDER_START = pd.Timestamp("2023-01-03")  # 보유자 전체 평단 추정 시작일(첫 보유분은 그날 종가로 둔다 = 가정)


# ---------- 0. 보관 → 거래일 ----------
def settle_td():
    """td(D): 미국 날짜 D−1까지 결제된 마지막 거래일. 결제 = 다음 결제일(T+1, 2024-05-28 전 거래는 T+2)."""
    nyse, settle = cm.calendars()
    trades = [t for t in nyse if dt.date(2022, 11, 1) <= t <= dt.date(2026, 12, 31)]
    sl = [settle[bisect.bisect_right(settle, t) + (0 if t >= cm.T1 else 1)] for t in trades]

    def td(D):
        return pd.Timestamp(trades[bisect.bisect_right(sl, (D - pd.Timedelta(days=1)).date()) - 1])
    return td


# ---------- 1. 자료 ----------
def load():
    pf = pd.read_parquet(DER / "price_feats.parquet").set_index("date")
    md = pd.read_parquet(DER / "measures_daily.parquet").set_index("date")
    cd = pd.read_parquet(DER / "custody_daily.parquet")
    fl = pd.read_parquet(DER / "flows_trade_product.parquet", columns=["product", "lev_long"]).drop_duplicates("product")
    ev = json.loads((DER / "events.json").read_text(encoding="utf-8"))
    return pf, md, cd, fl.set_index("product")["lev_long"].astype(bool), ev


def custody_by_date(cd, lev, pf, qa):
    """보관 날짜별 SOXL 금액·좌수·레버리지 롱 금액. 평가 종가 대안(앞 세션)도 함께."""
    cd = cd.copy()
    cd["lev_long"] = cd["product"].map(lev).fillna(False).astype(bool)
    qa["custody_products_without_flag"] = sorted(set(cd["product"]) - set(lev.index))  # 레버리지 아님으로 둠
    per = cd[cd["product"] == "SOXL"].set_index("custody_date").sort_index()
    per = per[["amt", "top50_total", "trade_date", "lag_used", "val_date", "SOXL_close", "SOXL_shares", "known_kst_date"]]
    g = cd[cd["lev_long"]].groupby("custody_date")
    per["levlong_amt"] = g["amt"].sum().reindex(per.index).fillna(0.0)
    per["n_levlong"] = g["product"].nunique().reindex(per.index).fillna(0).astype(int)
    T = pf.index
    pos = pd.Series(np.arange(len(T)), index=T)
    prev_val = T[pos[per["val_date"]].values - 1]
    per["val_prev"] = prev_val
    per["shares_vprev"] = per["amt"].values / pf["SOXL_Close"].reindex(prev_val).values
    td = settle_td()
    per["td"] = [td(D) for D in per.index]
    if not per["td"].is_monotonic_increasing:
        raise SystemExit("보관 → 거래일 대응이 단조가 아님")
    diff = per.index[per["td"] != per["trade_date"]]
    qa["mapping_diff_vs_panel"] = {"n": len(diff), "dates": [d.date().isoformat() for d in diff],
                                   "all_us_holiday": bool(all(d not in pf.index for d in diff))}
    qa["custody_dates"] = int(len(per))
    qa["soxl_missing_custody_dates"] = int(cd["custody_date"].nunique() - len(per))
    return per


def trade_table(per, pf):
    """거래일 표: 거래일 t까지 담은 첫 보관(가장 일찍 아는 값). 대응 보관이 없는 거래일은 앞 값(c_stale)."""
    first = per.reset_index().groupby("td").first()
    T = pf.index
    Tc = T[(T >= first.index.min()) & (T <= first.index.max())]
    ct = first.reindex(Tc)
    ct["c_stale"] = ct["custody_date"].isna()
    ct = ct.ffill()
    ct = ct.rename(columns={"custody_date": "c_date", "known_kst_date": "c_known", "amt": "soxl_amt", "SOXL_shares": "shares"})
    ct["soxl_share"] = ct["soxl_amt"] / ct["top50_total"]
    ct["levlong_share"] = ct["levlong_amt"] / ct["top50_total"]
    return ct


def orphan_soxl(qa):
    """자료판이 버린 SEIBRO 날짜(예측 거래일 없음)의 SOXL 순매수(원 d3 목록)."""
    u = pd.read_parquet(DER / "units.parquet")
    out = {}
    for d in u.loc[u["n_pred"] == 0, "seibro_date"]:
        p = PRIOR / "raw" / "daily" / "d3" / f"{d:%Y%m%d}_{d:%Y%m%d}.json"
        if p.exists():
            for x in json.loads(p.read_text(encoding="utf-8"))["rows"]:
                if x["isin"] == SOXL_ISIN and abs(x["net"]) > 0:
                    out[d.date().isoformat()] = float(x["net"])
    qa["orphan_soxl_net"] = out
    return out


# ---------- 2. 메아리 모형(간이) ----------
def echo_fit(ct, pf, qa):
    """일별 Δlog 좌수 ~ 체제 상수 + 같은 날 SOXX 로그 수익. 앞뒤 날 모두 자기 보관이 있는 날만 적합."""
    y = np.log(ct["shares"]).diff()
    ok = ~ct["c_stale"] & ~ct["c_stale"].shift(1, fill_value=True)
    r = pf["lr1"].reindex(ct.index)
    X = pd.get_dummies(pf["regime"].reindex(ct.index).astype(int), prefix="g").astype(float)
    X["r"] = r
    m = ok & y.notna() & r.notna()
    b, *_ = np.linalg.lstsq(X[m].values, y[m].values, rcond=None)
    fit = pd.Series(X.values @ b, index=ct.index)
    res = (y - fit).where(m)
    qa["echo"] = {"n": int(m.sum()), "beta_soxx_ret": round(float(b[-1]), 4),
                  "r2": round(float(1 - np.nansum(res ** 2) / np.nansum((y[m] - y[m].mean()) ** 2)), 4),
                  "const_by_regime_bp": [round(float(v) * 1e4, 2) for v in b[:-1]]}
    return fit, res


# ---------- 3. 평단(추정) ----------
def cohort_cost(md, P, start, end):
    """(start, end] 거래일 순매수 합 ÷ 순매수 좌수 합(종가 환산). 둘 다 양수일 때만 정의."""
    s = md.loc[(md.index > start) & (md.index <= end), "SOXL_net"]
    usd = float(s.sum())
    sh = float((s / P.reindex(s.index)).sum())
    return usd, sh, (usd / sh if usd > 0 and sh > 0 else np.nan)


def holder_cost(ct, md, P, qa):
    """보유자 전체 평단(추정, 두 방식). 시작 보유분은 HOLDER_START 종가로 둔다(가정).
    net: (시작 가치 + 누적 순매수 $) ÷ (시작 좌수 + 누적 순매수 좌수).
    mavg: 이동평균법 — 매수는 그날 종가로 더하고 매도는 평단을 바꾸지 않음, 보유 좌수는 날마다 보관 좌수로 맞춤."""
    idx = ct.index[ct.index >= HOLDER_START]
    t0 = idx[0]
    s0, p0 = ct.loc[t0, "shares"], P[t0]
    f = md["SOXL_net"].reindex(idx).fillna(0.0)
    f.iloc[0] = 0.0
    net = (s0 * p0 + f.cumsum()) / (s0 + (f / P.reindex(idx)).cumsum())
    avg, out = p0, {t0: p0}
    for i in range(1, len(idx)):
        t = idx[i]
        h = ct["shares"].iloc[ct.index.get_loc(t) - 1]
        b = md["SOXL_buy"].get(t, 0.0)
        b = 0.0 if not np.isfinite(b) else b
        avg = (avg * h + b) / (h + b / P[t])
        out[t] = avg
    qa["holder_cost_start"] = {"date": t0.date().isoformat(), "shares_M": round(s0 / 1e6, 3), "close": round(float(p0), 3)}
    return net, pd.Series(out)


# ---------- 4. 사건 ----------
def events(ev, last_cust):
    top = {e["episode"]: pd.Timestamp(e["start"]) for e in ev["episodes"]}
    rows = [{"event": pd.Timestamp(e["bottom"]), "group": "B15" if e["B15"] else "B10", "episode": e["bottom"],
             "top": pd.Timestamp(e["top10"])} for e in ev["B10_events"]]
    rows += [{"event": pd.Timestamp(f["date"]), "group": "F", "episode": f["episode"], "top": top[f["episode"]]}
             for f in ev["F"]]
    # 지금: k=0 = 마지막 보관 대응 거래일(바닥 판정 아님)
    rows.append({"event": last_cust, "group": "current", "episode": "current",
                 "top": pd.Timestamp(ev["current"]["10"]["last_top"])})
    return pd.DataFrame(rows)


def build(evs, pf, md, ct, fit, res, hnet, hmavg):
    T = pf.index
    pos = pd.Series(np.arange(len(T)), index=T)
    P, X = pf["SOXL_Close"], pf["SOXX_Close"]
    lsh = np.log(ct["shares"])
    cfit = fit.cumsum()
    last_c, last_s = ct.index.max(), md.index.max()
    paths, summ = [], []
    for _, e in evs.iterrows():
        i0, t0, top = pos[e["event"]], e["event"], e["top"]

        def at(k):
            j = i0 + k
            return T[j] if 0 <= j < len(T) else None

        def sh(t):
            return ct["shares"].get(t, np.nan) if t is not None and t <= last_c else np.nan

        sh0, shm20 = sh(t0), sh(at(-20))
        cu, cs, cost = cohort_cost(md, P, top, t0)
        # 경로
        for k in KS:
            t = at(k)
            if t is None:
                continue
            row = {"event": t0.date(), "group": e["group"], "episode": e["episode"], "k": k, "trade_date": t.date(),
                   "soxx_close": X[t], "soxx_rel_bottom": X[t] / X[t0] - 1, "soxl_close": P[t]}
            if t <= last_c:
                c = ct.loc[t]
                s = c["shares"]
                row.update({"custody_date": c["c_date"].date(), "custody_known": c["c_known"].date(), "c_stale": c["c_stale"],
                            "soxl_custody_usd": c["soxl_amt"], "soxl_shares": s,
                            "d_sh_vs_m20": s - shm20, "dlog_sh_vs_m20": np.log(s / shm20),
                            "d_sh_vs_0": s - sh0, "dlog_sh_vs_0": np.log(s / sh0),
                            "soxl_share_top50": c["soxl_share"], "levlong_share_top50": c["levlong_share"],
                            "top50_total": c["top50_total"], "echo_resid_day": res.get(t, np.nan)})
            if t <= last_s:
                row.update({"settle_net_usd": md.at[t, "SOXL_net"], "settle_merged": bool(md.at[t, "merged"]),
                            "settle_impl_sh": md.at[t, "SOXL_net"] / P[t]})
            if k >= 1 and np.isfinite(cost):
                row["gap_cohort"] = np.log(P[t] / cost)
            tp = at(k - 1)
            if tp in hnet.index:
                row["gap_holder_mavg"] = np.log(P[t] / hmavg[tp])
            paths.append(row)
        # 요약
        rec = {"event": t0.date(), "group": e["group"], "episode": e["episode"], "top_date": top.date(),
               "n_sess_decline": int(pos[t0] - pos[top]), "soxx_decline": np.log(X[t0] / X[top]),
               "soxl_decline": np.log(P[t0] / P[top]), "custody_through": min(last_c, at(20) or last_c).date()}
        spans = {"decline": (top, t0), **{w: (at(a), at(b)) for w, (a, b) in SPANS.items()}}
        for w, (a, b) in spans.items():
            ok = a is not None and b is not None and b <= last_c and a in ct.index
            rec[f"dlog_sh_{w}"] = lsh[b] - lsh[a] if ok else np.nan
            rec[f"resid_{w}"] = (lsh[b] - lsh[a]) - (cfit[b] - cfit[a]) if ok else np.nan
            rec[f"soxx_{w}"] = np.log(X[b] / X[a]) if a is not None and b is not None else np.nan
            # 같은 구간 결제 순매수로 본 좌수 변화 ÷ 시작 보유(대조)
            if ok and b <= last_s:
                seg = md.loc[(md.index > a) & (md.index <= b), "SOXL_net"]
                rec[f"impl_{w}"] = float((seg / P.reindex(seg.index)).sum()) / ct["shares"][a]
            else:
                rec[f"impl_{w}"] = np.nan
        rec["sh_top_M"] = sh(top) / 1e6
        rec["sh_0_M"] = sh0 / 1e6
        for nm, col in (("soxl_share", "soxl_share"), ("lev_share", "levlong_share")):
            for tag, t in (("top", top), ("0", t0), ("p20", at(20))):
                rec[f"{nm}_{tag}"] = ct[col].get(t, np.nan) if t is not None and t <= last_c else np.nan
        # 하락 매수자 평단(추정): (고점, k=0] 순매수
        rec["cohort_net_usd_M"], rec["cohort_net_sh_M"], rec["cohort_cost_est"] = cu / 1e6, cs / 1e6, cost
        rec["cohort_net_over_sh_top"] = cs / sh(top) if np.isfinite(sh(top)) else np.nan
        rec["soxl_0"] = P[t0]
        rec["gap_cohort_0"] = np.log(P[t0] / cost) if np.isfinite(cost) else np.nan
        tm1 = at(-1)
        rec["holder_cost_mavg_0"] = hmavg.get(tm1, np.nan)
        rec["gap_holder_mavg_0"] = np.log(P[t0] / hmavg.get(tm1, np.nan))
        k_be = np.nan
        if np.isfinite(cost):
            for k in range(1, 21):
                t = at(k)
                if t is None:
                    break
                if P[t] >= cost:
                    k_be = k
                    break
        rec["k_breakeven_cohort"] = k_be
        for k in (5, 20):
            t = at(k)
            rec[f"gap_cohort_{k:+d}"] = np.log(P[t] / cost) if t is not None and np.isfinite(cost) else np.nan
        # 본전 근처 매도: 본전 닿은 날부터 3세션 메아리 잔차 평균 대 나머지 k=1~20
        near, other = [], []
        for k in range(1, 21):
            t = at(k)
            if t is None or t > last_c or not np.isfinite(res.get(t, np.nan)):
                continue
            (near if np.isfinite(k_be) and k_be <= k <= k_be + 2 else other).append(res[t])
        rec["resid_be3_mean"] = np.mean(near) if near else np.nan
        rec["resid_other_mean"] = np.mean(other) if other else np.nan
        d = rec["dlog_sh_decline"]
        rec["label_decline"] = "" if not np.isfinite(d) else ("좌수 증가(물타기)" if d > 0 else "좌수 감소(투매)")
        summ.append(rec)
    return pd.DataFrame(paths), pd.DataFrame(summ)


# ---------- 5. 묶음 비교 ----------
METRICS = [f"{p}_{w}" for p in ("dlog_sh", "resid") for w in ("decline", "W1", "W2", "W3", "W4", "W5", "post20")]


def group_summary(es):
    b10 = es[es["group"].isin(["B15", "B10"])]
    b15 = es[es["group"] == "B15"]
    f = es[es["group"] == "F"]
    fe = f.groupby("episode")[METRICS].mean()
    rows = []
    for m in METRICS:
        rec = {"metric": m}
        for nm, d in (("B15", b15[m]), ("B10", b10[m]), ("F", f[m]), ("F_ep", fe[m])):
            v = d.dropna()
            rec[f"{nm}_n"], rec[f"{nm}_pos"] = len(v), int((v > 0).sum())
            rec[f"{nm}_median"] = v.median() if len(v) else np.nan
        dd = np.array([b[m] - fe.at[b["episode"], m] for _, b in b10.iterrows()
                       if b["episode"] in fe.index and np.isfinite(b[m]) and np.isfinite(fe.at[b["episode"], m])])
        nz = int((dd != 0).sum())
        rec.update({"pair_n": len(dd), "pair_pos": int((dd > 0).sum()), "pair_median": np.median(dd) if len(dd) else np.nan,
                    "pair_sign_p": stats.binomtest(int((dd > 0).sum()), nz).pvalue if nz else np.nan,
                    "pair_wilcoxon_p": stats.wilcoxon(dd).pvalue if nz >= 5 else np.nan})
        rows.append(rec)
    return pd.DataFrame(rows)


def breakeven(paths, res, hgap):
    """본전 근처 매도(설명용): 일별 메아리 잔차를 평단 대비 가격 칸으로 나눔."""
    rows = []

    def add(panel, ref, x, col):
        x = x.dropna(subset=[col, "r"])
        x = x.assign(bin=pd.cut(x[col], GAP_BINS, labels=GAP_LABELS, right=False))
        for b, g in x.groupby("bin", observed=False):
            rows.append({"panel": panel, "ref_cost": ref, "gap_bin": b, "n_days": len(g),
                         "n_events": g["event"].nunique() if "event" in g else np.nan,
                         "mean_resid": g["r"].mean() if len(g) else np.nan,
                         "se_resid": g["r"].std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else np.nan,
                         "share_resid_neg": (g["r"] < 0).mean() if len(g) else np.nan})

    a = pd.DataFrame({"r": res, "gap": hgap}).dropna()
    add("모든 거래일", "보유자 전체 평단 mavg(추정)", a, "gap")
    p = paths[(paths["k"] >= 1)].rename(columns={"echo_resid_day": "r"})
    for grp, gl in (("B10 바닥 뒤 k=1~20", ["B15", "B10"]), ("F 날 뒤 k=1~20", ["F"])):
        q = p[p["group"].isin(gl)]
        add(grp, "하락 매수자 평단(추정)", q, "gap_cohort")
        add(grp, "보유자 전체 평단 mavg(추정)", q, "gap_holder_mavg")
    return pd.DataFrame(rows)


# ---------- 6. 결제–보관 대조 ----------
def reconcile(per, md, pf, orphan, qa):
    P = pf["SOXL_Close"]
    T = pf.index
    imp = (md["SOXL_net"] / P.reindex(md.index)).cumsum()
    big = (np.log(P).diff().abs() > 0.15).astype(int).cumsum()
    mg = md["merged"].astype(int).cumsum()
    rec = per[["td", "trade_date", "lag_used", "val_date", "SOXL_close", "amt"]].rename(columns={"trade_date": "td_panel"})
    rec = rec[rec["td"] <= md.index.max()]
    a = rec["td"].shift(1)
    rec["prev_td"] = a
    rec["n_trade_days"] = [len(T[(T > x) & (T <= b)]) if pd.notna(x) else np.nan for x, b in zip(a, rec["td"])]
    rec["mapping_differs"] = rec["td"] != rec["td_panel"]
    rec["shares_M"] = per["SOXL_shares"] / 1e6
    rec["dsh_custody_M"] = rec["shares_M"].diff()
    rec["dsh_custody_vprev_M"] = (per["shares_vprev"] / 1e6).diff()
    v = imp.reindex(rec["td"]).values
    rec["dsh_settle_M"] = (v - np.r_[np.nan, v[:-1]]) / 1e6
    v = imp.reindex(rec["td_panel"]).values
    rec["dsh_settle_panelmap_M"] = (v - np.r_[np.nan, v[:-1]]) / 1e6
    for nm, s in (("merged_in_interval", mg), ("soxl_15pct_day_in_interval", big)):
        w = s.reindex(rec["td"]).values
        rec[nm] = (w - np.r_[np.nan, w[:-1]]) > 0
    od = set(pd.to_datetime(list(orphan)))
    rec["orphan_seibro_in_interval"] = [any(x < d <= y for d in od) for x, y in zip(rec.index.to_series().shift(1), rec.index)]
    rec["resid_M"] = rec["dsh_custody_M"] - rec["dsh_settle_M"]
    rec["resid_pct_hold"] = rec["resid_M"] / rec["shares_M"].shift(1)
    rec["cum_resid_M"] = rec["resid_M"].fillna(0).cumsum()
    rec = rec.reset_index()

    def st(name, sub, y, x, m):
        m = m & np.isfinite(y) & np.isfinite(x)
        if m.sum() < 3:
            return {"comparison": name, "subset": sub, "n": int(m.sum())}
        yy, xx = y[m], x[m]
        if np.ptp(xx) == 0:
            return {"comparison": name, "subset": sub, "n": int(m.sum()), "median_abs_err_M": float(np.median(np.abs(yy - xx)))}
        lr = stats.linregress(xx, yy)
        return {"comparison": name, "subset": sub, "n": int(m.sum()), "r2": lr.rvalue ** 2, "slope": lr.slope,
                "median_abs_err_M": float(np.median(np.abs(yy - xx))), "sum_custody_M": float(yy.sum()),
                "sum_settle_M": float(xx.sum()), "sum_resid_M": float((yy - xx).sum())}

    y, yp, x, xpm = (rec[c].values for c in ("dsh_custody_M", "dsh_custody_vprev_M", "dsh_settle_M", "dsh_settle_panelmap_M"))
    near = rec["mapping_differs"].values | np.r_[rec["mapping_differs"].values[1:], False] | np.r_[False, rec["mapping_differs"].values[:-1]]
    allm = np.ones(len(rec), bool)
    mgm = rec["merged_in_interval"].fillna(False).values.astype(bool)
    exm = rec["soxl_15pct_day_in_interval"].fillna(False).values.astype(bool)
    orm = np.array(rec["orphan_seibro_in_interval"].values, bool)
    dup = (rec["n_trade_days"] == 0).values
    t2 = (rec["td"] < pd.Timestamp("2024-05-28")).values
    name = "보관 Δ좌수 대 결제 순매수 ÷ 종가"
    rows = [st(name, "전체", y, x, allm), st(name, "합친 묶음 든 구간", y, x, mgm), st(name, "합친 묶음 밖", y, x, ~mgm),
            st(name, "SOXL 하루 ±15% 넘는 날 든 구간", y, x, exm), st(name, "그 밖", y, x, ~exm & ~mgm & ~orm & ~dup),
            st(name, "버린 SEIBRO 날짜 든 구간", y, x, orm),
            st(name, "T+2 구간", y, x, t2), st(name, "T+1 구간", y, x, ~t2),
            st("평가 종가 = 앞 세션(대안)", "전체", yp, x, allm),
            st(name, "대응이 자료판과 다른 날짜와 앞뒤", y, x, near),
            st("자료판 지연 대응(PANEL §5)", "전체", y, xpm, allm),
            st("자료판 지연 대응(PANEL §5)", "대응이 다른 날짜와 앞뒤", y, xpm, near)]
    for yr in sorted(rec["td"].dt.year.unique()):
        rows.append(st(name, f"{yr}년", y, x, (rec["td"].dt.year == yr).values))
    # 새 거래일이 없는 보관 날짜(같은 보유, 평가만 다름): Δ좌수는 0이어야 한다
    for nm, yy in (("D 이하 마지막 세션 종가(자료판)", y), ("앞 세션 종가(대안)", yp)):
        d = yy[dup & np.isfinite(yy)]
        rows.append({"comparison": f"새 거래일 없는 보관 날짜 Δ좌수, 평가 {nm}", "subset": "n_trade_days=0", "n": len(d),
                     "median_abs_err_M": float(np.median(np.abs(d))) if len(d) else np.nan})
    for h in (5, 20, 60):
        g = pd.DataFrame({"y": y, "x": x, "g": np.arange(len(rec)) // h}).dropna().groupby("g").sum()
        rows.append(st(f"{name}, 겹치지 않는 {h}보관일 합", "전체", g["y"].values, g["x"].values, np.ones(len(g), bool)))
    qa["reconcile"] = {"rows": len(rec), "custody_first_last_shares_M": [round(float(rec["shares_M"].iloc[0]), 3),
                                                                          round(float(rec["shares_M"].iloc[-1]), 3)],
                       "cum_custody_M": float(np.nansum(y)), "cum_settle_M": float(np.nansum(x))}
    return rec, pd.DataFrame(rows)


# ---------- 7. 실행 ----------
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    qa: dict = {}
    pf, md, cd, lev, ev = load()
    P = pf["SOXL_Close"]
    per = custody_by_date(cd, lev, pf, qa)
    ct = trade_table(per, pf)
    # 거래일 표는 결제 끝(10/30)까지 있으나 마지막 날은 앞 보관(c_stale) → 보관 끝 = 마지막 비-stale 거래일
    last_c = ct.index[~ct["c_stale"]].max()
    ct = ct[ct.index <= last_c]
    qa["custody_last_trade"] = last_c.date().isoformat()
    qa["stale_trade_dates"] = [d.date().isoformat() for d in ct.index[ct["c_stale"]]]
    orphan = orphan_soxl(qa)
    fit, res = echo_fit(ct, pf, qa)
    hnet, hmavg = holder_cost(ct, md, P, qa)
    hgap = np.log(P.reindex(ct.index) / hmavg.shift(1).reindex(ct.index))
    evs = events(ev, last_c)
    paths, es = build(evs, pf, md, ct, fit, res, hnet, hmavg)
    gs = group_summary(es)
    be = breakeven(paths, res, hgap)
    rec, rst = reconcile(per, md, pf, orphan, qa)
    # 지금: 가격 끝 날짜로 본 평단 대비
    pe = P.index.max()
    top = pd.Timestamp(ev["current"]["10"]["last_top"])
    cu, cs, cost = cohort_cost(md, P, top, md.index.max())
    qa["current"] = {"top": top.date().isoformat(), "settle_through": md.index.max().date().isoformat(),
                     "custody_through_trade": last_c.date().isoformat(), "price_end": pe.date().isoformat(),
                     "cohort_net_usd_M": round(cu / 1e6, 1), "cohort_net_sh_M": round(cs / 1e6, 3),
                     "cohort_cost_est": round(cost, 3), "soxl_price_end": round(float(P[pe]), 3),
                     "gap_cohort_price_end": round(float(np.log(P[pe] / cost)), 4),
                     "holder_cost_mavg_last": round(float(hmavg.iloc[-1]), 3), "holder_cost_net_last": round(float(hnet.iloc[-1]), 3),
                     "gap_holder_mavg_price_end": round(float(np.log(P[pe] / hmavg.iloc[-1])), 4)}
    qa["max_lev_products_in_custody"] = int(per["n_levlong"].max())
    fmt = dict(index=False, encoding="utf-8-sig", float_format="%.6g")
    paths.to_csv(OUT / "paths.csv", **fmt)
    es.to_csv(OUT / "event_summary.csv", **fmt)
    gs.to_csv(OUT / "group_summary.csv", **fmt)
    be.to_csv(OUT / "breakeven.csv", **fmt)
    rec.to_csv(OUT / "reconcile.csv", **fmt)
    rst.to_csv(OUT / "reconcile_stats.csv", **fmt)
    (OUT / "qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, default=str)[:2500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
