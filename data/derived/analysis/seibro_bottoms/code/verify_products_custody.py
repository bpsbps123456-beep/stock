"""상품·보관 갈래 반박 검증(독립 재계산). derived/만 읽는다. 실행: py -B code/verify_products_custody.py
출력: 화면에 JSON(숫자 전부). 보고서는 results/verify_products_custody.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.dont_write_bytecode = True
ST = Path(__file__).resolve().parents[1]
DER = ST / "derived"
WINS = {"W1": (-20, -6), "W2": (-5, -1), "W3": (0, 0), "W4": (1, 5), "W5": (6, 20)}
R = {}  # 결과 모음


def load():
    f = pd.read_parquet(DER / "flows_trade_product.parquet")
    f = f[~f["dummy"].astype(bool)]
    md = pd.read_parquet(DER / "measures_daily.parquet").set_index("date").sort_index()
    pf = pd.read_parquet(DER / "price_feats.parquet").set_index("date").sort_index()
    px = pd.read_csv(DER / "prices.csv", parse_dates=["date"])
    ev = json.loads((DER / "events.json").read_text(encoding="utf-8"))
    cd = pd.read_parquet(DER / "custody_daily.parquet")
    return f, md, pf, px, ev, cd


# ---------- 상품 ----------
def mats(f, dates):
    net = f.pivot_table(index="date", columns="product", values="net", aggfunc="sum").reindex(dates).fillna(0.0)
    attr = f.drop_duplicates("product", keep="last").set_index("product").reindex(net.columns)
    return net, attr


def evlist(ev, T):
    pos = {d: i for i, d in enumerate(T)}
    out = []
    for e in ev["B10_events"]:
        d = pd.Timestamp(e["bottom"])
        out.append(dict(kind="B10", B15=e["B15"], date=d, k0=pos[d], top=pos[pd.Timestamp(e["top10"])], ep=e["bottom"]))
    st = {x["episode"]: pd.Timestamp(x["start"]) for x in ev["episodes"]}
    for x in ev["F"]:
        d = pd.Timestamp(x["date"])
        out.append(dict(kind="F", B15=False, date=d, k0=pos[d], top=pos[st[x["episode"]]], ep=x["episode"]))
    return out


def span(e, w, n):
    if w == "decline":
        a, b = e["top"] + 1, e["k0"]
    else:
        a, b = e["k0"] + WINS[w][0], e["k0"] + WINS[w][1]
    return (a, b) if a >= 0 and b < n else None


def products(f, md, pf, px):
    T = md.index
    n = len(T)
    net, attr = mats(f, T)
    P = list(net.columns)
    X = net.to_numpy()
    cs = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, 0)])
    soxx = pf["SOXX_Close"]
    i0 = soxx.index.get_loc(T[0])
    cl = {"SOXX": np.r_[soxx.iloc[i0 - 1], soxx.reindex(T).to_numpy()]}
    q = px[px.ticker == "QQQ"].set_index("date")["Close"]
    cl["QQQ"] = np.r_[q.iloc[q.index.get_loc(T[0]) - 1], q.reindex(T).to_numpy()]
    return T, n, P, X, cs, cl, attr


def wsum(cs, a, b):
    return cs[b + 1] - cs[a]


def wret(cl, a, b):
    return cl[b + 1] / cl[a] - 1


def rank_of(s, j, side):
    if side == "buy":
        return int((s > s[j]).sum() + 1) if s[j] > 0 else None
    return int((s < s[j]).sum() + 1) if s[j] < 0 else None


def roll(cs, L, n):
    ends = np.arange(L - 1, n)
    return ends, cs[ends + 1] - cs[ends + 1 - L]


def check_products(f, md, pf, px, ev):
    T, n, P, X, cs, cl, attr = products(f, md, pf, px)
    J = {p: i for i, p in enumerate(P)}
    E = evlist(ev, T)
    B = [e for e in E if e["kind"] == "B10"]
    B15 = [e for e in B if e["B15"]]
    F = [e for e in E if e["kind"] == "F"]
    yrs = T.year.to_numpy()
    out = {}

    # P1: SOXL 순매수 1위(사건)와 1위 기준률(같은 길이 창, SOXX 수익이 사건 창 이하)
    def rank_table(prod, side, wins, grp):
        j = J[prod]
        res = {}
        for w in wins:
            rk = []
            for e in grp:
                ab = span(e, w, n)
                if ab is None:
                    rk.append(None)
                    continue
                rk.append(rank_of(wsum(cs, *ab), j, side))
            res[w] = rk
        return res

    out["P1_SOXL_buy_rank_B15"] = rank_table("SOXL", "buy", ["decline", "W2", "W3", "W4"], B15)
    out["P1_SOXL_buy_rank_B10"] = rank_table("SOXL", "buy", ["decline", "W2", "W3"], B)
    out["P1_SOXL_buy_rank_F"] = rank_table("SOXL", "buy", ["decline", "W2", "W3"], F)

    # 1위 기준률: 같은 길이 굴린 창 전체, SOXX 창 수익 ≤ 사건 창 수익(사건과 겹치는 창 포함)
    def base_rank1(prod, L, rmax=None, rmin=None, top=1, side="buy"):
        j = J[prod]
        ends, S = roll(cs, L, n)
        r = cl["SOXX"][ends + 1] / cl["SOXX"][ends + 1 - L] - 1
        m = np.ones(len(ends), bool)
        if rmax is not None:
            m &= r <= rmax
        if rmin is not None:
            m &= r >= rmin
        Sm = S[m]
        if side == "buy":
            rk = (Sm > Sm[:, [j]]).sum(1) + 1
            ok = Sm[:, j] > 0
        else:
            rk = (Sm < Sm[:, [j]]).sum(1) + 1
            ok = Sm[:, j] < 0
        return round(float(((rk <= top) & ok).mean()), 3), int(m.sum())

    b1 = {}
    for L, nm in ((5, "W2"), (1, "W3")):
        b1[nm] = {"all": base_rank1("SOXL", L), "soxx<=-5%": base_rank1("SOXL", L, rmax=-0.05),
                  "soxx<=-2%": base_rank1("SOXL", L, rmax=-0.02), "soxx<0": base_rank1("SOXL", L, rmax=0.0)}
    out["P1_SOXL_rank1_base"] = b1

    # 공통 함수: 상위 k 맞음
    def hit(prod, side, w, e, k=5):
        ab = span(e, w, n)
        if ab is None:
            return None
        r = rank_of(wsum(cs, *ab), J[prod], side)
        return r is not None and r <= k

    def matched_rate(prod, side, w, e, tick, k=5, same_year=False):
        """사건 창과 같은 길이, 같은 방향으로 가격이 사건 창만큼 이상 움직인 창(사건 창과 겹침 제외)에서 상위 k 비율."""
        ab = span(e, w, n)
        L = ab[1] - ab[0] + 1
        ends, S = roll(cs, L, n)
        r = cl[tick][ends + 1] / cl[tick][ends + 1 - L] - 1
        re = wret(cl[tick], *ab)
        m = (r <= re) if re < 0 else (r >= re)
        m &= np.abs(ends - ab[1]) >= L
        if same_year:
            m &= yrs[ends] == yrs[ab[1]]
        j = J[prod]
        Sm = S[m]
        if side == "buy":
            rk = (Sm > Sm[:, [j]]).sum(1) + 1
            ok = Sm[:, j] > 0
        else:
            rk = (Sm < Sm[:, [j]]).sum(1) + 1
            ok = Sm[:, j] < 0
        return (float(((rk <= k) & ok).mean()) if m.sum() else np.nan), int(m.sum()), round(float(re), 4)

    def pb_tail(ps, k):
        dp = np.zeros(len(ps) + 1)
        dp[0] = 1.0
        for p in ps:
            dp[1:] = dp[1:] * (1 - p) + dp[:-1] * p
            dp[0] *= 1 - p
        return float(dp[k:].sum())

    def item(prod, side, w, ticks=("SOXX", "QQQ")):
        h15 = [hit(prod, side, w, e) for e in B15]
        h10 = [hit(prod, side, w, e) for e in B]
        hf = [hit(prod, side, w, e) for e in F]
        hf = [x for x in hf if x is not None]
        res = {"B15": f"{sum(h15)}/{len(h15)}", "B15_hits": h15, "B10": f"{sum(h10)}/{len(h10)}",
               "F": f"{sum(hf)}/{len(hf)}"}
        for t in ticks:
            for sy in (False, True):
                rr = [matched_rate(prod, side, w, e, t, same_year=sy) for e in B15]
                ps = np.array([x[0] for x in rr])
                ok = np.isfinite(ps)
                k = int(np.array(h15)[ok].sum())
                tag = f"{t}{'_sameyr' if sy else '_allyr'}"
                res[tag] = {"exp": round(float(ps[ok].sum()), 2), "n_ev": int(ok.sum()), "hits": k,
                            "p": round(pb_tail(ps[ok], k), 4), "n_cmp": [x[1] for x in rr], "ev_ret": [x[2] for x in rr]}
        return res

    out["P1_SOXL_decline"] = item("SOXL", "buy", "decline")
    out["P1_SOXL_W2"] = item("SOXL", "buy", "W2")
    out["P1_SOXL_W3"] = item("SOXL", "buy", "W3")
    out["P2_TQQQ_W2"] = item("TQQQ", "buy", "W2")
    out["P3_TSLA_W4"] = item("TSLA", "buy", "W4")
    out["P4_SOXL_W4_sell"] = item("SOXL", "sell", "W4")
    out["P4_NVDA_W5_sell"] = item("NVDA", "sell", "W5")
    out["P4_TQQQ_W5_sell"] = item("TQQQ", "sell", "W5")
    # TSLA: B10 비바닥 5개
    out["P3_TSLA_W4_B10_nonB15"] = [hit("TSLA", "buy", "W4", e) for e in B if not e["B15"]]
    # TQQQ W2 상위 5 사건별 QQQ·SOXX 수익
    out["P2_TQQQ_W2_rets"] = [(e["date"].date().isoformat(), round(wret(cl["QQQ"], *span(e, "W2", n)), 4),
                               round(wret(cl["SOXX"], *span(e, "W2", n)), 4)) for e in B15]

    # 범주 순매수(W4 인버스·L3)와 FUND
    cats = {"INV": attr["EQ_INV"].fillna(False).astype(bool).to_numpy(),
            "LSGL": attr["LSGL"].fillna(False).astype(bool).to_numpy()}
    rowsL3 = {}
    for e in B15:
        d = {}
        for w in ("W2", "W3", "W4", "W5"):
            ab = span(e, w, n)
            s = wsum(cs, *ab)
            d[w] = {"L3_net_M": round(float(s[cats["LSGL"]].sum()) / 1e6, 1),
                    "L3_share": round(float(s[cats["LSGL"]].sum() / np.abs(s).sum()), 3),
                    "INV_net_M": round(float(s[cats["INV"]].sum()) / 1e6, 1),
                    "SOXL_net_M": round(float(s[J["SOXL"]]) / 1e6, 1)}
        rowsL3[e["date"].date().isoformat()] = d
    out["P4_P6_cat_by_event"] = rowsL3
    # B10 비바닥과 F의 W4·W5 L3 부호
    out["P6_L3_W4W5_B10_all"] = {e["date"].date().isoformat(): [round(float(wsum(cs, *span(e, w, n))[cats["LSGL"]].sum()) / 1e6, 1)
                                                                 for w in ("W4", "W5")] for e in B}
    # L3 W4·W5 순매도의 평소 비율(같은 길이 창, 해별)
    for L, w in ((5, "W4"), (15, "W5")):
        ends, S = roll(cs, L, n)
        v = S[:, cats["LSGL"]].sum(1)
        y = yrs[ends]
        out[f"P6_L3_{w}_neg_rate_by_year"] = {int(yy): round(float((v[y == yy] < 0).mean()), 3) for yy in np.unique(y)}
        r = cl["SOXX"][ends + 1] / cl["SOXX"][ends + 1 - L] - 1
        out[f"P6_L3_{w}_neg_rate_up8pct"] = {int(yy): [round(float((v[(y == yy) & (r >= 0.08)] < 0).mean()), 3) if ((y == yy) & (r >= 0.08)).any() else None,
                                                    int(((y == yy) & (r >= 0.08)).sum())] for yy in np.unique(y)}
    # 인버스 W4 순매수 양수: 평소·반등 맞춤 비율
    ends, S = roll(cs, 5, n)
    v = S[:, cats["INV"]].sum(1)
    r = cl["SOXX"][ends + 1] / cl["SOXX"][ends + 1 - 5] - 1
    out["P4_INV_W4_pos_rate"] = {"all": round(float((v > 0).mean()), 3), "soxx5>=+8%": [round(float((v[r >= 0.08] > 0).mean()), 3), int((r >= 0.08).sum())],
                                 "soxx5>=+5%": [round(float((v[r >= 0.05] > 0).mean()), 3), int((r >= 0.05).sum())]}

    # FUND = |Σ음| ÷ Σ양 (상품 창 합). 하락 구간·W5
    def fund(s):
        return float(-s[s < 0].sum() / s[s > 0].sum())

    fr = {}
    for w in ("decline", "W5"):
        vals, base_med, px_med, Ppx = [], [], [], []
        for e in B15:
            ab = span(e, w, n)
            L = ab[1] - ab[0] + 1
            v = fund(wsum(cs, *ab))
            ends, S = roll(cs, L, n)
            fu = (-np.clip(S, None, 0).sum(1)) / np.clip(S, 0, None).sum(1)
            r = cl["SOXX"][ends + 1] / cl["SOXX"][ends + 1 - L] - 1
            re = wret(cl["SOXX"], *ab)
            sy = yrs[ends] == yrs[ab[1]]
            mm = sy & ((r <= re) if re < 0 else (r >= re))
            vals.append(round(v, 3))
            base_med.append(round(float(np.median(fu[sy])), 3))
            px_med.append([round(float(np.median(fu[mm])), 3) if mm.any() else None, int(mm.sum())])
            Ppx.append(round(float((fu[mm] < v).mean() * 100), 0) if mm.any() else None)
        fr[w] = {"B15": vals, "median": float(np.median(vals)), "base_sameyr_med": base_med,
                 "matched_med_n": px_med, "P_in_matched": Ppx}
    # F 하락 구간 FUND
    fv = [fund(wsum(cs, *span(e, "decline", n))) for e in F if span(e, "decline", n)]
    fr["decline_F_median"] = round(float(np.median(fv)), 3)
    out["P5_FUND"] = fr
    return out


# ---------- 보관 ----------
def check_custody(md, pf, ev, cd):
    out = {}
    T = pf.index
    Pc = pf["SOXL_Close"]
    s = cd[cd["product"] == "SOXL"].set_index("custody_date").sort_index()
    sh = s["SOXL_shares"]
    # 독립 대응: 보관 D는 SEIBRO 결제 날짜 s(t) ≤ D 인 거래를 모두 담는다(자료판 seibro_date 열 사용)
    sd = md["seibro_date"]
    alt = pd.Series([sd.index[sd <= D].max() if (sd <= D).any() else pd.NaT for D in s.index], index=s.index)
    panel = s["trade_date"]
    diff = alt != panel
    out["map_n_dates"] = int(len(s))
    out["map_n_diff_alt_vs_panel"] = int(diff.sum())
    out["map_diff_on_us_holiday_or_not"] = {"D_not_NYSE": int(sum(d not in T for d in s.index[diff])),
                                            "D_is_NYSE": int(sum(d in T for d in s.index[diff]))}
    out["map_diff_dates_sample"] = [d.date().isoformat() for d in s.index[diff]][:60]
    impl = (md["SOXL_net"] / Pc.reindex(md.index)).cumsum()
    dsh = sh.diff() / 1e6

    def r2(mapping, mask=None):
        v = impl.reindex(mapping).to_numpy()
        x = pd.Series(np.r_[np.nan, np.diff(v)], index=mapping.index) / 1e6
        m = x.notna() & dsh.notna() & (mapping <= md.index.max())
        if mask is not None:
            m &= mask
        lr = stats.linregress(x[m], dsh[m])
        return {"n": int(m.sum()), "r2": round(lr.rvalue ** 2, 4), "slope": round(lr.slope, 3),
                "mae_M": round(float(np.median(np.abs(dsh[m] - x[m]))), 3)}

    near = diff | diff.shift(1, fill_value=False) | diff.shift(-1, fill_value=False)
    out["map_r2"] = {"alt_all": r2(alt), "panel_all": r2(panel), "alt_near": r2(alt, near), "panel_near": r2(panel, near),
                     "alt_far": r2(alt, ~near), "panel_far": r2(panel, ~near)}
    # 예: 2025-04-18
    for D in ("2025-04-18", "2026-04-03"):
        D = pd.Timestamp(D)
        if D in s.index:
            out[f"ex_{D.date()}"] = {"dsh_M": round(float(dsh[D]), 2), "panel_td": str(panel[D].date()), "alt_td": str(alt[D].date()),
                                     "settle_sh_alt_M": round(float((impl[alt[D]] - impl[alt.shift(1)[D]]) / 1e6), 2),
                                     "settle_sh_panel_M": round(float((impl[panel[D]] - impl[panel.shift(1)[D]]) / 1e6), 2)}
    # 사건 창 영향: 자료판 c_trade 대응 차이가 B15 W3~W5 보관 끝점에 걸리는가
    # 거래일 표(독립): t까지 담은 첫 보관
    first = pd.Series(s.index, index=alt.values).groupby(level=0).first()
    td_tab = first.reindex(md.index)
    stale = td_tab.isna()
    td_tab = td_tab.ffill()
    shares_t = sh.reindex(td_tab.values).to_numpy()
    shT = pd.Series(shares_t, index=md.index)
    paneltd = md["c_date"]
    out["trade_days_custody_differs_panel"] = int((td_tab != paneltd).sum())
    lastc = shT.index[~stale].max()
    shT = shT[shT.index <= lastc]
    out["last_custody_trade"] = str(lastc.date())
    # 메아리 모형
    y = np.log(shT).diff()
    ok = ~stale.reindex(shT.index) & ~stale.reindex(shT.index).shift(1, fill_value=True)
    reg = pf["regime"].reindex(shT.index)
    Xd = pd.get_dummies(reg.astype(int), prefix="g").astype(float)
    Xd["r"] = pf["lr1"].reindex(shT.index)
    m = ok & y.notna() & Xd["r"].notna()
    b, *_ = np.linalg.lstsq(Xd[m].values, y[m].values, rcond=None)
    fit = pd.Series(Xd.values @ b, index=shT.index)
    res = (y - fit).where(m)
    out["echo"] = {"n": int(m.sum()), "beta": round(float(b[-1]), 3),
                   "r2": round(float(1 - (res[m] ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()), 3),
                   "const_bp": [round(float(v) * 1e4, 1) for v in b[:-1]]}
    # 메아리 + 앞 5일 SOXX 수익(지연 메아리) 추가 모형
    Xd2 = Xd.copy()
    Xd2["r_l1"] = pf["lr1"].shift(1).reindex(shT.index)
    Xd2["r_pre5"] = (np.log(pf["SOXX_Close"]).diff(5).shift(1)).reindex(shT.index)
    m2 = m & Xd2.notna().all(1)
    b2, *_ = np.linalg.lstsq(Xd2[m2].values, y[m2].values, rcond=None)
    out["echo_lag"] = {"coef_r_l1": round(float(b2[-2]), 3), "coef_pre5": round(float(b2[-1]), 3)}
    res2 = (y - pd.Series(Xd2.values @ b2, index=shT.index)).where(m2)

    E = []
    st = {x["episode"]: pd.Timestamp(x["start"]) for x in ev["episodes"]}
    for e in ev["B10_events"]:
        E.append(("B15" if e["B15"] else "B10", pd.Timestamp(e["bottom"]), pd.Timestamp(e["top10"]), e["bottom"]))
    for x in ev["F"]:
        E.append(("F", pd.Timestamp(x["date"]), st[x["episode"]], x["episode"]))
    pos = {d: i for i, d in enumerate(T)}
    cfit = fit.cumsum()
    rows = []
    for g, d, top, ep in E:
        i = pos[d]

        def at(k):
            return T[i + k] if i + k < len(T) else None

        rr = {"g": g, "date": d, "ep": ep}
        for w, (a, bb) in {"decline": (top, d), "W4": (d, at(5))}.items():
            if bb is None or bb > lastc or a not in shT.index:
                rr[w] = rr[f"res_{w}"] = rr[f"soxx_{w}"] = np.nan
                continue
            rr[w] = float(np.log(shT[bb] / shT[a]))
            rr[f"res_{w}"] = rr[w] - float(cfit[bb] - cfit[a])
            rr[f"soxx_{w}"] = float(np.log(pf["SOXX_Close"][bb] / pf["SOXX_Close"][a]))
        rows.append(rr)
    es = pd.DataFrame(rows)
    b10 = es[es.g.isin(["B15", "B10"])]
    fe = es[es.g == "F"]
    out["C1_decline_pos"] = {"B10": f"{int((b10.decline > 0).sum())}/{int(b10.decline.notna().sum())}",
                             "B15_med": round(float(b10[b10.g == 'B15'].decline.median()), 3),
                             "B10_med": round(float(b10.decline.median()), 3),
                             "F": f"{int((fe.decline > 0).sum())}/{int(fe.decline.notna().sum())}",
                             "F_med": round(float(fe.decline.median()), 3),
                             "res_B15_pos": f"{int((b10[b10.g == 'B15'].res_decline > 0).sum())}/6",
                             "res_F_pos": f"{int((fe.res_decline > 0).sum())}/{int(fe.res_decline.notna().sum())}"}
    # 하락 좌수 vs 하락 SOXX 수익(B10+F): 단순 회귀
    a = es.dropna(subset=["decline", "soxx_decline"])
    lr = stats.linregress(a.soxx_decline, a.decline)
    out["C1_decline_vs_soxx"] = {"n": len(a), "slope": round(lr.slope, 2), "r2": round(lr.rvalue ** 2, 3)}
    # 국면 짝
    fm = fe[fe.ep != "current"].groupby("ep")[["decline", "res_decline", "W4", "res_W4"]].mean()
    pr = {}
    for c in ("decline", "res_decline", "W4", "res_W4"):
        dd = np.array([r[c] - fm.at[r.ep, c] for _, r in b10.iterrows() if r.ep in fm.index and np.isfinite(r[c]) and np.isfinite(fm.at[r.ep, c])])
        pr[c] = {"n": len(dd), "pos": int((dd > 0).sum()), "med": round(float(np.median(dd)), 3),
                 "sign_p": round(float(stats.binomtest(int((dd > 0).sum()), int((dd != 0).sum())).pvalue), 3),
                 "wilcoxon_p": round(float(stats.wilcoxon(dd).pvalue), 3) if len(dd) >= 5 else None}
    out["C1C2_pairs"] = pr
    out["C2_W4"] = {"B10_pos": f"{int((b10.W4 > 0).sum())}/{int(b10.W4.notna().sum())}", "B10_med": round(float(b10.W4.median()), 3),
                    "res_pos": f"{int((b10.res_W4 > 0).sum())}/{int(b10.res_W4.notna().sum())}",
                    "F_pos": f"{int((fe.W4 > 0).sum())}/{int(fe.W4.notna().sum())}"}

    # C3 본전: 하락 매수자 평단(추정) = (고점, 바닥] 순매수 $ ÷ 순매수 좌수(종가)
    br = []
    for g, d, top, ep in E:
        if g == "F":
            continue
        seg = md.loc[(md.index > top) & (md.index <= d), "SOXL_net"]
        cost = seg.sum() / (seg / Pc.reindex(seg.index)).sum()
        i = pos[d]
        for k in range(1, 21):
            if i + k >= len(T):
                break
            t = T[i + k]
            if t > lastc or not np.isfinite(res.get(t, np.nan)):
                continue
            br.append({"ev": d, "k": k, "gap": float(np.log(Pc[t] / cost)), "gap_simple": float(Pc[t] / cost - 1),
                       "res": float(res[t]), "res2": float(res2.get(t, np.nan)),
                       "cum_soxx": float(np.log(pf["SOXX_Close"][t] / pf["SOXX_Close"][d]))})
    bd = pd.DataFrame(br)
    bins = [-np.inf, -0.5, -0.3, -0.1, 0.0, 0.1, np.inf]
    bd["bin"] = pd.cut(bd.gap, bins, right=False)
    g = bd.groupby("bin", observed=False)
    out["C3_bins"] = {str(k): {"n": int(len(v)), "n_ev": int(v.ev.nunique()), "mean_res": round(float(v.res.mean()) * 100, 2) if len(v) else None,
                               "mean_res_laggedmodel": round(float(v.res2.mean()) * 100, 2) if len(v) else None,
                               "ev_mean_of_means": round(float(v.groupby('ev').res.mean().mean()) * 100, 2) if len(v) else None,
                               "n_ev_neg": int((v.groupby('ev').res.mean() < 0).sum()) if len(v) else None}
                      for k, v in g}
    # 같은 날들을 바닥 뒤 SOXX 누적 수익 칸으로 나눠도 같은가(본전 대 단순 반등)
    bd["cbin"] = pd.qcut(bd.cum_soxx, 4)
    out["C3_by_cum_soxx_quartile"] = {str(k): [int(len(v)), round(float(v.res.mean()) * 100, 2)] for k, v in bd.groupby("cbin", observed=False)}
    # 0~+10% 칸 대 −10~0% 칸 차의 사건 군집 이표본(사건별 평균)
    a1 = bd[(bd.gap >= 0) & (bd.gap < 0.1)].groupby("ev").res.mean()
    a0 = bd[(bd.gap >= -0.1) & (bd.gap < 0)].groupby("ev").res.mean()
    out["C3_cluster"] = {"ev_in_0_10": len(a1), "ev_in_m10_0": len(a0),
                         "welch_p_event_means": round(float(stats.ttest_ind(a1, a0, equal_var=False).pvalue), 3) if len(a1) > 1 and len(a0) > 1 else None}
    # 잔차 자기상관(1일)
    out["resid_ac1"] = round(float(res.autocorr(1)), 3)

    # C4 지금: (10/05, 결제 끝] 순매수, 평단, 11/03 종가 대비 단순·로그
    top = pd.Timestamp(ev["current"]["10"]["last_top"])
    seg = md.loc[md.index > top, "SOXL_net"]
    usd, shs = seg.sum(), (seg / Pc.reindex(seg.index)).sum()
    cost = usd / shs
    pe = Pc.index.max()
    out["C4_current"] = {"net_usd_M": round(usd / 1e6, 1), "net_sh_M": round(shs / 1e6, 2), "cost": round(cost, 2),
                         "price_end": str(pe.date()), "soxl_end": round(float(Pc[pe]), 2),
                         "gap_simple": round(float(Pc[pe] / cost - 1), 4), "gap_log": round(float(np.log(Pc[pe] / cost)), 4),
                         "sh_top_M": round(float(shT[top]) / 1e6, 2), "sh_last_M": round(float(shT.iloc[-1]) / 1e6, 2),
                         "ratio_net_sh_over_top": round(float(shs / shT[top]), 3),
                         "dlog": round(float(np.log(shT.iloc[-1] / shT[top])), 3),
                         "soxx_top_to_lastc": round(float(np.log(pf['SOXX_Close'][lastc] / pf['SOXX_Close'][top])), 3)}
    # 같은 순매수÷고점 좌수 비율을 B15 하락 구간에서(단순·로그 간극)
    cmp = {}
    for g, d, top_, ep in E:
        if g == "F" and ep != "2025-04-08":
            continue
        seg = md.loc[(md.index > top_) & (md.index <= d), "SOXL_net"]
        shs_ = (seg / Pc.reindex(seg.index)).sum()
        c_ = seg.sum() / shs_
        cmp[f"{g}:{d.date()}"] = [round(float(shs_ / shT[top_]), 2), round(float(Pc[d] / c_ - 1), 3), round(float(np.log(Pc[d] / c_)), 3)]
    out["C4_ratio_gap_simple_log"] = cmp

    # C6 결제–보관 누적 차
    x = pd.Series(np.r_[np.nan, np.diff(impl.reindex(alt).to_numpy())], index=s.index) / 1e6
    m = x.notna() & dsh.notna() & (alt <= md.index.max())
    big = (np.log(Pc).diff().abs() > 0.15)
    bigc = big.astype(int).cumsum()
    bv = bigc.reindex(alt.values).to_numpy()
    bigint = pd.Series(np.r_[False, np.diff(bv) > 0], index=s.index)
    resid = (dsh - x)[m]
    out["C6"] = {"sum_custody_M": round(float(dsh[m].sum()), 2), "sum_settle_M": round(float(x[m].sum()), 2),
                 "cum_resid_M": round(float(resid.sum()), 2), "big_intervals": int(bigint[m].sum()),
                 "resid_big_M": round(float(resid[bigint[m]].sum()), 2), "resid_other_M": round(float(resid[~bigint[m]].sum()), 2),
                 "big_days": [str(d.date()) for d in Pc.index[big]][:60]}
    # 큰 날 부호별
    sgn = np.sign(np.log(Pc).diff())
    bigdn = big & (sgn < 0)
    dn = bigdn.astype(int).cumsum().reindex(alt.values).to_numpy()
    dnint = pd.Series(np.r_[False, np.diff(dn) > 0], index=s.index)
    out["C6"]["resid_bigdown_M"] = round(float(resid[dnint[m]].sum()), 2)
    out["C6"]["n_bigdown_intervals"] = int(dnint[m].sum())
    return out


def main():
    f, md, pf, px, ev, cd = load()
    R["products"] = check_products(f, md, pf, px, ev)
    R["custody"] = check_custody(md, pf, ev, cd)
    print(json.dumps(R, ensure_ascii=False, default=str))


if __name__ == "__main__" and "--extra" not in sys.argv:
    main()


# ---------- 추가 점검(2차) ----------
def extra():
    f, md, pf, px, ev, cd = load()
    T, n, P, X, cs, cl, attr = products(f, md, pf, px)
    J = {p: i for i, p in enumerate(P)}
    E = evlist(ev, T)
    B15 = [e for e in E if e["kind"] == "B10" and e["B15"]]
    out = {}
    # SOXL이 1위가 아닌 B15 칸의 1위 상품
    for e in B15:
        for w in ("decline", "W2", "W3"):
            s = wsum(cs, *span(e, w, n))
            o = np.argsort(-s)[:3]
            if P[o[0]] != "SOXL":
                out[f"top3_{e['date'].date()}_{w}"] = [(P[i], round(float(s[i]) / 1e6)) for i in o]
    # FUND: 사건 하락 길이와 같은 창, SOXX ≤ −10%, 사건과 안 겹침, 모든 해
    fu_all = {}
    for e in B15:
        ab = span(e, "decline", n)
        L = ab[1] - ab[0] + 1
        ends, S = roll(cs, L, n)
        fu = (-np.clip(S, None, 0).sum(1)) / np.clip(S, 0, None).sum(1)
        r = cl["SOXX"][ends + 1] / cl["SOXX"][ends + 1 - L] - 1
        m = (np.abs(ends - ab[1]) >= L)
        fu_all[str(e["date"].date())] = {"L": L, "dn10_med": [round(float(np.median(fu[m & (r <= -0.10)])), 3) if (m & (r <= -0.10)).any() else None, int((m & (r <= -0.10)).sum())],
                                         "up_med": [round(float(np.median(fu[m & (r >= 0)])), 3), int((m & (r >= 0)).sum())],
                                         "corr_fund_ret": round(float(np.corrcoef(fu, r)[0, 1]), 3)}
    out["FUND_vs_ret"] = fu_all
    # 판(B2) 뒤 L3 W4 창 순매도 비율
    lsgl = attr["LSGL"].fillna(False).astype(bool).to_numpy()
    for L in (5, 15):
        ends, S = roll(cs, L, n)
        v = S[:, lsgl].sum(1)
        st = T[ends - L + 1] >= pd.Timestamp("2026-07-31")
        out[f"L3_L{L}_neg_rate_postB2"] = [round(float((v[st] < 0).mean()), 3), int(st.sum())]
    # 보관 대응 차이가 사건 창 끝점에 걸리는가(자료판 c_date 대 독립 대응)
    s = cd[cd["product"] == "SOXL"].set_index("custody_date").sort_index()
    sd = md["seibro_date"]
    alt = pd.Series([sd.index[sd <= D].max() for D in s.index], index=s.index)
    first = pd.Series(s.index, index=alt.values).groupby(level=0).first().reindex(md.index).ffill()
    bad = set(md.index[first != md["c_date"]])
    pos = {d: i for i, d in enumerate(T)}
    hits = {}
    for e in E:
        ks = {"top": e["top"], **{f"k{k}": e["k0"] + k for k in (-21, -6, -1, 0, 5, 20)}}
        hh = [nm for nm, i in ks.items() if 0 <= i < n and T[i] in bad]
        if hh:
            hits[f"{e['kind']}{'*' if e['B15'] else ''}:{e['date'].date()}"] = hh
    out["panel_C_endpoints_affected"] = hits
    # 지금: 10월 보관 좌수 경로
    sh = s["SOXL_shares"] / 1e6
    out["oct_shares_M"] = {str(d.date()): round(float(v), 1) for d, v in sh[sh.index >= pd.Timestamp("2026-10-05")].items()}
    out["settle_1030"] = [round(float(md.at[pd.Timestamp("2026-10-30"), "SOXL_net"]) / 1e6, 1),
                          round(float(md.at[pd.Timestamp("2026-10-30"), "SOXL_net"] / pf.at[pd.Timestamp("2026-10-30"), "SOXL_Close"]) / 1e6, 2)]
    print(json.dumps(out, ensure_ascii=False, default=str))


if __name__ == "__main__" and "--extra" in sys.argv:
    extra()
