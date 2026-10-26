"""점검(critic.md) 반영 계산 — 2026-10-14 23시. 정의는 BRIEF·하네스 그대로, 새로 붙인 변형은 모두 '사후'.

사용
    py review_fixes.py inverse   # 3번: 단일종목 인버스 매핑(yfinance 확인) → 실행 가능 비율·실제 인버스 h10
    py review_fixes.py lag       # 2번: b4 '하루 묵힌 값'(원자료 when ≤ t−1)으로 깃발·138칸 다시 판정
    py review_fixes.py b8win     # 4번: 60분봉 진입·청산 한국시간 08~24시 비율, 창 안 사건만 판정
    py review_fixes.py x6        # 5번: ETF 이탈 → 인버스, 기초 고가가 스탑(L, L+0.5·ATR)에 닿으면 장중 청산
    py review_fixes.py regime    # 6번: 통과·준통과 칸의 하위 국면(ep34 6월/7월, ep35)·주 묶음 t
    py review_fixes.py t11       # 15번(하): 시간 청산을 t+11 시가로 바꾼 민감도(종목 기준선·b7 3배 짝)
    py review_fixes.py b5ep      # 19번(하): b5 동반 이탈 칸 손실의 국면 비중 csv
    py review_fixes.py multi     # 13번: 갈래를 합친 다중 비교 요약(칸 수·귀무 평균 합)
    py review_fixes.py all

정의(결과 보기 전에 정함)
- 인버스 후보: 기억해 둔 미국 상장 단일종목 인버스 티커 후보를 넣고, yfinance 일봉(분할 반영)으로
  '그 종목의 인버스'인지 확인한다: 겹치는 날 일간 수익률(종가→종가) 회귀 기울기 ≤ −0.5, 상관 ≤ −0.8,
  겹치는 날 ≥ 60. 배수 = −기울기를 0.25 단위로 반올림. 통과 못 하면 매핑하지 않는다(웹 검색 없음).
- 실행 가능: 사건 t+1에 그 인버스 일봉 시가가 있음(상장 뒤). 한 종목에 인버스가 여럿이면 1배에 가장 가까운 것
  (같으면 먼저 상장한 것) 하나로 잰다.
- 실제 인버스 h10: t+1 인버스 시가 매수 → 하네스 사건표의 청산일·청산 종류(되돌림이면 그날 시가, 시간이면 종가)에
  인버스 가격으로 판다. 비용 왕복 0.10%. 비교: 같은 사건의 종목 공매도 대리(short_ret_h10)와 배수 × 대리.
- b4 lag: attach(lag=-1) = 진입일 t+1 대신 이탈일 t에 놓인 값(원자료 when ≤ t−1). 기존 `*_lag1`(lag=+1)은
  t+2에 놓인 값(when ≤ t+1, 진입 뒤 공개)이라 '하루 미래' 값이었다.
- 60분봉 창: 체결 시각(진입 = entry_ts 봉 시가, 청산 = 시가 청산이면 그 봉 시가, 종가 청산이면 봉 끝
  min(봉 시작+1시간, 16:00)) 을 미 동부 → 한국시간으로 바꿔 08:00 ≤ 시각 < 24:00이면 창 안.
  판정: 원래 칸(문턱 = 전체 앞 구간 3분위 그대로)을 창 안 사건에만 evaluate_cells.
- X6: b7 사건(ETF 6짝, E1 = t+1 정규장 시가 인버스 매수). 스탑 S = L(X6a) 또는 L + 0.5·ATR20(t−1)(X6b).
  t+1..t+10 정규장 1시간봉(기초)을 차례로 보고 봉 고가 ≥ S인 첫 봉에서 기초 체결가 = max(S, 봉 시가).
  인버스 체결가 = 인버스 전일 종가 × (1 − 배수 × (기초 체결가 / 기초 전일 종가 − 1))(일일 재설정 식).
  안 걸리면 t+10 종가. 비용 0.10%. 슬리피지 없음(낙관). 정규장에서만 스탑이 산다고 본다.
- 하위 국면: ep34-6월 = 이탈일 2026-05-29~06-30, ep34-7월 = 07-01~07-31, ep35 = 08-11~09-16, 그 밖 = 나머지.
  주 묶음 = 이탈일의 ISO 주.
- t+11: 시간 청산(t+h 종가)을 t+h+1 시가로 바꾼다(되돌림 청산은 원래 다음 시가라 그대로).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import harness as H  # noqa: E402

RES = H.RESULTS
PER = {"앞": H.FRONT, "뒤": H.BACK, "2025": H.AUX}

# ───────────────────────────── 3번: 단일종목 인버스 ─────────────────────────────
CAND = {
    "NVDA": ["NVDD", "NVDQ", "NVDS", "NVD"],
    "AMD": ["AMDD", "AMDS"],
    "AVGO": ["AVS", "AVGD"],
    "MU": ["MUD"],
    "TSM": ["TSMZ"],
    "MSFT": ["MSFD"],
    "GOOGL": ["GGLS"],
    "AMZN": ["AMZD"],
    "META": ["METD"],
    "AAPL": ["AAPD"],
    "TSLA": ["TSLS", "TSLQ", "TSDD", "TSLZ"],
    "PLTR": ["PLTD", "PLTZ"],
    "QCOM": ["QCMD"],
    "PANW": ["PALD"],
    "CSCO": ["CSCS"],
    "SMCI": ["SMCZ"],
    "MRVL": ["MRVD"],
    "ORCL": ["ORCS"],
    "ARM": ["ARMD"],
    "CRWD": ["CRWS"],
    "INTC": ["INTD"],
    "CRWV": ["CORD"],
}
INV_DAILY = RES / "review_inverse_daily.parquet"


def fetch_inverse(force=False):
    if INV_DAILY.exists() and not force:
        return pd.read_parquet(INV_DAILY)
    tick = sorted({t for v in CAND.values() for t in v})
    d = H._yf(tick, "2022-06-01", "2026-10-14")
    d.to_parquet(INV_DAILY)
    return d


# 자료 수정(사후, 결과 보기 전 손익 계산 전에 발견): yfinance MUD 일봉이 2026-03-05에 −89.9%(MU −0.9%)로 끊긴다.
# 분할 미반영으로 보고 그 앞 OHLC를 10으로 나눈다(그날 기대 수익 +0.93% / 관측 비율 0.101 = 9.99).
SPLIT_FIX = {"MUD": ("2026-03-05", 10.0)}


def fix_splits(d):
    d = d.copy()
    for t, (day, f) in SPLIT_FIX.items():
        m = (d["sym"] == t) & (d["date"] < day)
        for c in ("Open", "High", "Low", "Close"):
            d.loc[m, c] = d.loc[m, c] / f
    return d


def verify_inverse(P, d):
    rows = []
    for u, cands in CAND.items():
        if u not in P.j:
            continue
        uc = pd.Series(P.col("C", u), index=P.cal)
        for t in cands:
            g = d[d["sym"] == t].set_index("date")["Close"].reindex(P.cal)
            n_px = int(g.notna().sum())
            first = g.first_valid_index()
            ru, ri = uc.pct_change(fill_method=None), g.pct_change(fill_method=None)
            ok = ru.notna() & ri.notna() & (ru.abs() < 0.5) & (ri.abs() < 0.9)
            n = int(ok.sum())
            if n >= 2:
                x, y = ru[ok].to_numpy(), ri[ok].to_numpy()
                beta = float(np.cov(x, y)[0, 1] / np.var(x, ddof=1))
                corr = float(np.corrcoef(x, y)[0, 1])
            else:
                beta = corr = np.nan
            acc = bool(n >= 60 and beta <= -0.5 and corr <= -0.8)
            rows.append({"underlying": u, "inverse": t, "days_with_price": n_px,
                         "first_date": None if first is None else str(first.date()),
                         "overlap_days": n, "beta": beta, "corr": corr,
                         "lev_est": round(-beta * 4) / 4 if acc else np.nan, "accepted": acc})
    return pd.DataFrame(rows)


def pick_map(ver):
    """종목 → (인버스, 배수, 첫날). 여럿이면 배수 1에 가까운 것, 같으면 먼저 상장."""
    a = ver[ver["accepted"]].copy()
    a["k"] = (a["lev_est"] - 1).abs()
    a = a.sort_values(["underlying", "k", "first_date"])
    return {r.underlying: (r.inverse, float(r.lev_est), r.first_date) for r in a.groupby("underlying").head(1).itertuples()}


def inverse_panel(P, d, syms):
    out = {}
    for t in syms:
        g = d[d["sym"] == t].set_index("date").reindex(P.cal)
        out[t] = (g["Open"].to_numpy(float), g["Close"].to_numpy(float))
    return out


def cmd_inverse():
    P = H.load_panel()
    d = fix_splits(fetch_inverse())
    ver = verify_inverse(P, d)
    ver.to_csv(RES / "review_inverse_map.csv", index=False, encoding="utf-8-sig")
    mp = pick_map(ver)
    ip = inverse_panel(P, d, sorted({v[0] for v in mp.values()}))
    ev = H.load_events_all()
    ev = ev[(ev["kind"] == "stock") & ev["short_ret_h10"].notna()].copy()
    ev["inv"] = ev["sym"].map(lambda s: mp.get(s, (None,))[0])
    ev["lev"] = ev["sym"].map(lambda s: mp.get(s, (None, np.nan))[1])
    inv_ret = np.full(len(ev), np.nan)
    for k, r in enumerate(ev.itertuples()):
        if not isinstance(r.inv, str):
            continue
        o, c = ip[r.inv]
        e, x = int(r.i) + 1, int(r.exit_i)
        if e >= len(o) or x < 0 or np.isnan(o[e]):
            continue
        px = o[x] if r.exit_type == "open" else c[x]
        if np.isnan(px):
            continue
        inv_ret[k] = px / o[e] - 1 - H.COST
    ev["inv_ret_h10"] = inv_ret
    ev["exec"] = ev["inv_ret_h10"].notna()
    rows = []
    for pn, p in PER.items():
        for N in list(H.NS) + ["합"]:
            g = ev[H.in_period(ev, p)]
            if N != "합":
                g = g[g["N"] == N]
            x = g[g["exec"]]
            m_inv, t_inv, n_inv, d_inv = H.cluster_mean(x["inv_ret_h10"], x["t"]) if len(x) >= 3 else (np.nan,) * 4
            m_px, t_px, _, _ = H.cluster_mean(x["short_ret_h10"], x["t"]) if len(x) >= 3 else (np.nan,) * 4
            rows.append({"period": pn, "N": N, "events": len(g), "dates": g["t"].nunique(),
                         "exec_events": len(x), "exec_dates": x["t"].nunique(),
                         "exec_share": len(x) / len(g) if len(g) else np.nan,
                         "exec_syms": ",".join(sorted(x["sym"].unique())),
                         "inv_mean": m_inv, "inv_t": t_inv, "proxy_mean_same": m_px, "proxy_t_same": t_px,
                         "lev_proxy_mean": float((x["lev"] * (x["short_ret_h10"] + H.COST) - H.COST).mean()) if len(x) else np.nan,
                         "inv_win": float((x["inv_ret_h10"] > 0).mean()) if len(x) else np.nan,
                         "all_proxy_mean": float(g["short_ret_h10"].mean()) if len(g) else np.nan})
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / "review_inverse_exec.csv", index=False, encoding="utf-8-sig")
    keep = ["sym", "N", "t", "inv", "lev", "cls", "short_ret_h10", "inv_ret_h10", "exit_reason", "episode"]
    ev.loc[ev["exec"] & (ev["t"] >= "2025-01-01"), keep].to_csv(RES / "review_inverse_events.csv", index=False,
                                                                   encoding="utf-8-sig")
    # b4 GVIP 통과 칸의 뒤 구간 켜짐 중 실행 가능한 것
    b4 = pd.read_parquet(RES / "b4_flow_events.parquet")
    thr = 0.5381526104417667
    on = b4[(b4["N"] == 20) & H.in_period(b4, H.BACK) & b4["short_ret_h10"].notna() & (b4["gvip_rel20_p"] >= thr)]
    on = on.merge(ev[["sym", "N", "t", "inv", "lev", "inv_ret_h10"]], on=["sym", "N", "t"], how="left")
    gv = {"on_events": len(on), "on_stock": int((on["kind"] == "stock").sum()),
          "exec_events": int(on["inv_ret_h10"].notna().sum()),
          "exec_rows": on.loc[on["inv_ret_h10"].notna(), ["sym", "t", "inv", "lev", "short_ret_h10", "inv_ret_h10"]]
          .assign(t=lambda x: x["t"].dt.date.astype(str)).to_dict("records"),
          "exec_inv_mean": float(on["inv_ret_h10"].mean()) if on["inv_ret_h10"].notna().any() else None}
    out = {"map": {k: list(v) for k, v in mp.items()}, "n_candidates": int(len(ver)),
           "n_accepted": int(ver["accepted"].sum()), "gvip_back_on": gv}
    (RES / "review_inverse_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str),
                                                     encoding="utf-8")
    pd.set_option("display.width", 250)
    print(ver.round(3).to_string(index=False))
    print(tab.round(4).drop(columns=["exec_syms"]).to_string(index=False))
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:3000])


# ───────────────────────────── 2번: b4 하루 묵힌 값 ─────────────────────────────
def cmd_lag():
    import b4_flow as B4
    cal = H.load_panel().cal
    ev0 = H.load_events_all()
    src = B4.load_sources()
    ev0m = B4.attach(ev0, src=src, cal=cal, lag=-1)
    # 시점 점검: 원자료 when ≤ t−1 (진입일 t+1보다 이틀 앞 거래일 이하가 아니라, 이탈일 t보다 앞)
    viol = {}
    for f in B4.FEAT_NAMES:
        w = ev0m[f"w_{f}"]
        m = ev0m[f].notna()
        viol[f] = int((w[m] >= ev0m.loc[m, "t"]).sum())
    base = pd.read_parquet(RES / "b4_flow_events.parquet")
    ev = ev0m[[c for c in ev0m.columns if not c.startswith("w_")]].copy()
    # 가설 깃발(설명용) — 기존 hyp_table 그대로, 값만 하루 묵힌 것
    hyp = B4.hyp_table(ev.assign(nvdl_top10_lag1=ev["nvdl_top10"], upro_top10_lag1=ev["upro_top10"]))
    hyp = hyp[hyp["flag"].isin(["nvdl_top10", "upro_top10", "gex_neg", "shvr_spike"])]
    hyp.to_csv(RES / "review_b4_stale_flags.csv", index=False, encoding="utf-8-sig")
    # 138칸 다시 판정(사후 민감도)
    cells = H.tercile_cells(ev, B4.FEAT_NAMES, by=("N",))
    tab, summ = H.evaluate_cells(ev, cells)
    tab.to_csv(RES / "review_b4_stale_cells.csv", index=False, encoding="utf-8-sig")
    s = {k: v for k, v in summ.items() if k not in ("null", "null_shift")}
    for k in ("null", "null_shift"):
        if k in summ:
            s[k] = {kk: vv for kk, vv in summ[k].items() if kk != "dist"}
    s["timing_violations_w_ge_t"] = viol
    # 기존(lag=0)과 값이 같은 비율
    same = {}
    for f in B4.FEAT_NAMES:
        a = base.set_index(["sym", "N", "t"])[f]
        b = ev.set_index(["sym", "N", "t"])[f].reindex(a.index)
        ok = a.notna() & b.notna()
        same[f] = float((np.isclose(a[ok], b[ok])).mean()) if ok.any() else None
    s["share_equal_to_lag0"] = same
    (RES / "review_b4_stale_summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=1, default=str),
                                                      encoding="utf-8")
    pd.set_option("display.width", 250)
    print(hyp.round(4).to_string(index=False))
    cols = ["cell", "rank", "f_diff", "f_n_on", "b_diff", "b_t", "b_p", "b_mean_on", "b_n_on", "b_dates_on", "a_diff", "pass"]
    print(tab.loc[tab["selected"], cols].round(4).to_string(index=False))
    print(json.dumps({k: v for k, v in s.items() if k not in ("share_equal_to_lag0",)}, ensure_ascii=False, default=str)[:2500])
    print("viol", viol)


# ───────────────────────────── 4번: 60분봉 주문 가능 창 ─────────────────────────────
def kst(ts):
    x = pd.to_datetime(ts)
    return x.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("Asia/Seoul")


def in_window(ts):
    k = kst(ts)
    return (k.dt.hour >= 8) & k.notna()


def b8_load():
    ev = pd.concat([pd.read_parquet(RES / f"b8_events_N{N}.parquet") for N in (7, 14, 28)], ignore_index=True)
    ev = ev[ev["short_ret_h10"].notna()].copy()
    ent = pd.to_datetime(ev["entry_ts"])
    xd = pd.to_datetime(ev["exit_date"])
    close_at = (xd + pd.Timedelta(hours=1)).where(xd.dt.strftime("%H:%M") != "15:30",
                                                  xd.dt.normalize() + pd.Timedelta(hours=16))
    xt = xd.where(ev["exit_type"] == "open", close_at)
    ev["entry_in"] = in_window(ent).to_numpy()
    ev["exit_in"] = in_window(xt).to_numpy()
    ev["both_in"] = ev["entry_in"] & ev["exit_in"]
    ev["entry_kst"] = kst(ent).dt.strftime("%m-%d %H:%M").to_numpy()
    return ev


def cmd_b8win():
    import b8_intraday as B8
    ev = b8_load()
    rows = []
    for pn, p in PER.items():
        for N in (7, 14, 28):
            g = ev[H.in_period(ev, p) & (ev["N"] == N)]
            for sub, m in (("전체", np.ones(len(g), bool)), ("진입 창 안", g["entry_in"].to_numpy()),
                           ("진입·청산 창 안", g["both_in"].to_numpy())):
                x = g[m]
                mm, tt, nn, dd = H.cluster_mean(x["short_ret_h10"], x["t"]) if len(x) >= 3 else (np.nan,) * 4
                rows.append({"period": pn, "N": N, "subset": sub, "events": len(x), "dates": x["t"].nunique(),
                             "share": len(x) / len(g) if len(g) else np.nan, "h10_mean": mm, "h10_t": tt,
                             "stock_share": float((x["kind"] == "stock").mean()) if len(x) else np.nan})
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / "review_b8_window.csv", index=False, encoding="utf-8-sig")
    # 판정: 원래 칸(전체 사건으로 문턱) → 창 안 부분집합
    feats = [f for f in ("depth_atr", "vol_ratio", "tod_open", "tod_close", "tod_mid", "dbox_pos", "box_w_atr",
                         "gap_break", "ret_day") if f in ev]
    full = pd.concat([pd.read_parquet(RES / f"b8_events_N{N}.parquet") for N in (7, 14, 28)], ignore_index=True)
    full = full.merge(ev[["sym", "N", "ts", "entry_in", "both_in"]], on=["sym", "N", "ts"], how="left")
    cells = H.tercile_cells(full, feats, by=("N",))
    out = {}
    for sub in ("entry_in", "both_in"):
        df = full[full[sub].fillna(False).astype(bool)]
        tab2, summ = H.evaluate_cells(df, cells)
        tab2.to_csv(RES / f"review_b8_{sub}_cells.csv", index=False, encoding="utf-8-sig")
        out[sub] = {k: v for k, v in summ.items() if k not in ("null", "null_shift")}
        for k in ("null", "null_shift"):
            if k in summ:
                out[sub][k] = {kk: vv for kk, vv in summ[k].items() if kk != "dist"}
        sel = tab2[tab2["selected"]]
        out[sub]["selected_rows"] = sel[["cell", "f_diff", "f_n_on", "b_diff", "b_t", "b_p", "b_mean_on", "b_n_on",
                                         "pass"]].round(5).to_dict("records")
    out["n_cells"] = len(cells)
    (RES / "review_b8_window_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str),
                                                       encoding="utf-8")
    pd.set_option("display.width", 250)
    print(tab.round(4).to_string(index=False))
    print(json.dumps(out, ensure_ascii=False, default=str)[:3000])


# ───────────────────────────── 5번: X6 장중 스탑 청산 ─────────────────────────────
LEV = {"SOXS": 3, "SQQQ": 3, "QID": 2, "PSQ": 1, "SH": 1}
PAIRS = [("SOXX", "SOXS"), ("SMH", "SOXS"), ("QQQ", "SQQQ"), ("QQQ", "QID"), ("QQQ", "PSQ"), ("SPY", "SH")]


def load_h1(sym):
    q = pd.read_parquet(H.ROOT / "cache/h1" / f"{sym}.parquet")
    q.index = pd.to_datetime(q.index)
    return q


def cmd_x6():
    P = H.load_panel()
    ev = H.load_events_all()
    e = ev[(ev["kind"] == "etf") & ev["sym"].isin({u for u, _ in PAIRS}) & (ev["t"] >= "2025-01-01")
           & (ev["t"] <= H.BACK[1])]
    h1 = {u: load_h1(u) for u in {u for u, _ in PAIRS}}
    day = {u: h1[u].index.normalize() for u in h1}
    rows = []
    chk = []
    for r in e.itertuples():
        i, Lv, atr = int(r.i), float(r.L), float(r.atr20)
        uc = P.col("C", r.sym)
        for u, inv in PAIRS:
            if u != r.sym:
                continue
            io, ic = P.col("O", inv), P.col("C", inv)
            k_lev = LEV[inv]
            if i + 10 >= len(P.cal) or np.isnan(io[i + 1]):
                continue
            px_in = io[i + 1]
            for nm, S in (("X6a_스탑L", Lv), ("X6b_스탑L+0.5ATR", Lv + 0.5 * atr)):
                px_out, why, xday, miss = np.nan, None, None, False
                for k in range(i + 1, i + 11):
                    d = P.cal[k]
                    bars = h1[u][day[u] == d]
                    if bars.empty:
                        miss = True
                        break
                    hit = bars[bars["High"] >= S]
                    if len(hit):
                        b = hit.iloc[0]
                        u_fill = max(S, float(b["Open"]))
                        if k == i + 1 and hit.index[0] == bars.index[0] and float(b["Open"]) >= S:
                            u_fill = float(b["Open"])
                        px_out = ic[k - 1] * (1 - k_lev * (u_fill / uc[k - 1] - 1))
                        why, xday = "스탑", d
                        break
                if miss:
                    continue
                if why is None:
                    px_out, why, xday = ic[i + 10], "시간t+10", P.cal[i + 10]
                rows.append({"sym": u, "inv": inv, "lev": k_lev, "N": r.N, "t": r.t, "cls": r.cls, "exit": nm,
                             "inv_ret": px_out / px_in - 1 - H.COST, "exit_why": why, "exit_day": xday,
                             "episode": r.episode})
        # 일일 재설정 식 점검: 모형 종가 vs 실제 인버스 종가(t+1..t+10)
    for u, inv in PAIRS:
        uc, ic = P.col("C", u), P.col("C", inv)
        m = (P.cal >= "2025-01-01") & (P.cal <= "2026-09-30")
        idx = np.flatnonzero(m)
        model = ic[idx - 1] * (1 - LEV[inv] * (uc[idx] / uc[idx - 1] - 1))
        err = np.abs(model / ic[idx] - 1)
        chk.append({"pair": f"{u}->{inv}", "median_abs_err": float(np.nanmedian(err)),
                    "p99_abs_err": float(np.nanquantile(err, 0.99))})
    df = pd.DataFrame(rows)
    df["period"] = None
    for pn, (a, b) in PER.items():
        df.loc[(df["t"] >= a) & (df["t"] <= b), "period"] = pn
    df.to_csv(RES / "review_x6_events.csv", index=False, encoding="utf-8-sig")
    # 비교용: b7 E1 X1·X3(같은 사건)
    b7 = pd.read_csv(RES / "b7_hedge_exec_events.csv", parse_dates=["t"])
    b7 = b7[b7["ok"] & (b7["entry"] == "E1_t+1시가") & b7["exit"].isin(["X1_되돌림", "X3_시간t+10"])]
    b7 = b7.rename(columns={"exit": "exit"})[["sym", "inv", "lev", "N", "t", "cls", "exit", "inv_ret", "episode", "period"]]
    allx = pd.concat([df[b7.columns], b7], ignore_index=True)
    out = []
    for grp, pairs in (("3배 짝", {("SOXX", "SOXS"), ("SMH", "SOXS"), ("QQQ", "SQQQ")}), ("6짝", set(PAIRS))):
        g0 = allx[[(a, b) in pairs for a, b in zip(allx["sym"], allx["inv"])]]
        for (ex, pn), g in g0.groupby(["exit", "period"]):
            m, t, n, G = H.cluster_mean(g["inv_ret"], g["t"])
            out.append({"group": grp, "exit": ex, "period": pn, "events": len(g), "dates": g["t"].nunique(),
                        "mean": m, "t": t, "win": float((g["inv_ret"] > 0).mean()),
                        "stop_share": float((df.loc[g.index.intersection(df.index), "exit_why"] == "스탑").mean())
                        if ex.startswith("X6") else np.nan})
    tab = pd.DataFrame(out)
    tab.to_csv(RES / "review_x6_summary.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(tab.round(4).to_string(index=False))
    print(pd.DataFrame(chk).to_string(index=False))
    (RES / "review_x6_check.json").write_text(json.dumps(chk, ensure_ascii=False, indent=1), encoding="utf-8")


# ───────────────────────────── 6번: 하위 국면·주 묶음 ─────────────────────────────
SUBREG = [("ep34-6월", "2026-05-29", "2026-06-30"), ("ep34-7월", "2026-07-01", "2026-07-31"),
          ("ep35", "2026-08-11", "2026-09-16")]


def subreg(t):
    t = pd.to_datetime(pd.Series(t)).to_numpy()
    lab = np.full(len(t), "그 밖", dtype=object)
    for nm, a, b in SUBREG:
        a, b = np.datetime64(a), np.datetime64(b)
        lab[(t >= a) & (t <= b)] = nm
    return lab


def week_codes(t):
    iso = pd.to_datetime(pd.Series(t)).dt.isocalendar()
    return H._codes((iso["year"].astype(str) + "-" + iso["week"].astype(str)).to_numpy())


def regime_rows(name, df, on):
    """df: 뒤 구간 사건(모집단), on: 1/0 배열."""
    out = []
    y = df["short_ret_h10"].to_numpy(float)
    M = np.asarray(on, float)
    g = H._codes(df["t"].to_numpy())
    n1, n0, m1, m0, d, t = H.cluster_diff(M, y, g)
    wk = week_codes(df["t"])
    _, _, _, _, dw, tw = H.cluster_diff(M, y, wk)
    mo, to, _, _ = H.cluster_mean(y[M == 1], df["t"].to_numpy()[M == 1])
    iso = pd.to_datetime(df["t"]).dt.isocalendar()
    wk_on = (iso["year"].astype(str) + "-" + iso["week"].astype(str)).to_numpy()[M == 1]
    mw, tw_on, _, _ = H.cluster_mean(y[M == 1], wk_on)
    head = {"cell": name, "part": "뒤 전체", "n_on": int(n1[0]), "dates_on": int(df.loc[M == 1, "t"].nunique()),
            "mean_on": m1[0], "n_off": int(n0[0]), "mean_off": m0[0], "diff": d[0], "t_date": t[0],
            "t_week": tw[0], "weeks_on": len(set(wk_on)), "t_on_mean_date": to, "t_on_mean_week": tw_on,
            "on_sum": float(y[M == 1].sum()), "share_of_on_sum": 1.0}
    out.append(head)
    lab = subreg(df["t"].to_numpy())
    tot = y[M == 1].sum()
    for nm in [s[0] for s in SUBREG] + ["그 밖"]:
        m = lab == nm
        if not m.any():
            continue
        yy, MM = y[m], M[m]
        r = {"cell": name, "part": nm, "n_on": int((MM == 1).sum()), "dates_on": int(df.loc[m & (M == 1), "t"].nunique()),
             "mean_on": float(yy[MM == 1].mean()) if (MM == 1).any() else np.nan,
             "n_off": int((MM == 0).sum()), "mean_off": float(yy[MM == 0].mean()) if (MM == 0).any() else np.nan,
             "on_sum": float(yy[MM == 1].sum())}
        r["diff"] = r["mean_on"] - r["mean_off"]
        if (MM == 1).sum() >= 2 and (MM == 0).sum() >= 2:
            gg = H._codes(df.loc[m, "t"].to_numpy())
            r["t_date"] = float(H.cluster_diff(MM, yy, gg)[5][0])
        r["share_of_on_sum"] = r["on_sum"] / tot if tot != 0 else np.nan
        out.append(r)
    return out


def cmd_regime():
    P = H.load_panel()
    rows = []
    # (1) b4 GVIP 통과 칸
    b4 = pd.read_parquet(RES / "b4_flow_events.parquet")
    x = b4[(b4["N"] == 20) & H.in_period(b4, H.BACK) & b4["short_ret_h10"].notna() & b4["gvip_rel20_p"].notna()]
    rows += regime_rows("b4 N20|gvip_rel20_p≥0.5382", x.reset_index(drop=True),
                        (x["gvip_rel20_p"] >= 0.5381526104417667).astype(float).to_numpy())
    # (2) b7 준통과 두 칸(t ≥ 2, p < 0.05, 켜짐 평균 < 0)
    import b7_hedge as B7
    ev = B7.add_features(H.load_events_all(), P)
    cells = dict(H.tercile_cells(ev, B7.FEATS, by=("N",)))
    for nm in ("N20|sq_brk=0(lo)", "N40|n_etf_brk≤0(하위3분위)"):
        s = cells[nm]
        m = H.in_period(ev, H.BACK) & ev["short_ret_h10"].notna() & s.notna()
        rows += regime_rows(f"b7 {nm}", ev[m].reset_index(drop=True), s[m].to_numpy())
    # (3) b6 1위(같은 사건 집합 2칸)
    f = b4[["sym", "N", "t", "pc_equity5_p"]]
    e6 = ev.merge(f, on=["sym", "N", "t"], how="left")
    pop = (e6["N"] == 20) & (e6["kind"] == "stock")
    on6 = (e6["ret5_pre"] >= -0.0238729) & (e6["pc_equity5_p"] >= 0.844712) & (e6["sq_brk"] == 0)
    nn = e6["ret5_pre"].notna() & e6["pc_equity5_p"].notna() & e6["sq_brk"].notna()
    m = pop & nn & H.in_period(e6, H.BACK) & e6["short_ret_h10"].notna()
    rows += regime_rows("b6 1·2위(ret5_pre & pc_equity5_p & sq_brk=0)", e6[m].reset_index(drop=True),
                        on6[m].astype(float).to_numpy())
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / "review_regime.csv", index=False, encoding="utf-8-sig")
    # 뒤 구간 사건 전체의 하위 국면 분포
    evb = H.load_events_all()
    evb = evb[H.in_period(evb, H.BACK) & evb["short_ret_h10"].notna()]
    dist = pd.Series(subreg(evb["t"].to_numpy())).value_counts().to_dict()
    (RES / "review_regime_dist.json").write_text(json.dumps({"back_events_by_subregime": dist,
                                                             "back_events": len(evb)}, ensure_ascii=False),
                                                 encoding="utf-8")
    pd.set_option("display.width", 250)
    print(tab.round(4).to_string(index=False))
    print(dist)


# ───────────────────────────── 15번(하): t+11 시가 청산 ─────────────────────────────
def short_outcome_open(o, c, i, Lv, h, cost=H.COST):
    r, why, xi, xt = H.short_outcome(o, c, i, Lv, h, cost)
    if why != "time":
        return r
    k = i + h + 1
    if k >= len(o) or np.isnan(o[k]):
        return np.nan
    return (o[i + 1] - o[k]) / o[i + 1] - cost


def cmd_t11():
    P = H.load_panel()
    ev = H.load_events_all()
    ev = ev[ev["short_ret_h10"].notna()].copy()
    ev["h10_t11open"] = [short_outcome_open(P.col("O", r.sym), P.col("C", r.sym), int(r.i), float(r.L), 10)
                         for r in ev.itertuples()]
    rows = []
    for pn, p in PER.items():
        for k in ("stock", "etf"):
            for N in H.NS:
                g = ev[H.in_period(ev, p) & (ev["kind"] == k) & (ev["N"] == N) & ev["h10_t11open"].notna()]
                if len(g) < 3:
                    continue
                a = H.cluster_mean(g["short_ret_h10"], g["t"])
                b = H.cluster_mean(g["h10_t11open"], g["t"])
                rows.append({"period": pn, "kind": k, "N": N, "events": len(g), "time_exit_share":
                             float((g["exit_reason"] == "time").mean()), "h10_close": a[0], "t_close": a[1],
                             "h10_t11open": b[0], "t_t11open": b[1]})
    tab = pd.DataFrame(rows)
    tab.to_csv(RES / "review_t11.csv", index=False, encoding="utf-8-sig")
    print(tab.round(4).to_string(index=False))
    # b7 인버스 E1: 시간 청산(X1의 시간 청산분·X2~X4) 종가 → 다음 날 시가
    b7 = pd.read_csv(RES / "b7_hedge_exec_events.csv", parse_dates=["t", "entry_day", "exit_day"])
    b7 = b7[b7["ok"] & b7["traded"] & (b7["entry"] == "E1_t+1시가")].copy()
    pos = {d: k for k, d in enumerate(P.cal)}
    alt = []
    for r in b7.itertuples():
        if not str(r.exit_why).startswith("시간"):
            alt.append(r.inv_ret)
            continue
        k = pos[r.exit_day] + 1
        o = P.col("O", r.inv)
        alt.append(o[k] / r.entry_px - 1 - H.COST if k < len(o) and not np.isnan(o[k]) else np.nan)
    b7["inv_ret_t11open"] = alt
    r3 = []
    pairs3 = {("SOXX", "SOXS"), ("SMH", "SOXS"), ("QQQ", "SQQQ")}
    b7 = b7[[(a, b) in pairs3 for a, b in zip(b7["sym"], b7["inv"])]]
    for (ex, pn), g in b7.groupby(["exit", "period"]):
        a = H.cluster_mean(g["inv_ret"], g["t"])
        b = H.cluster_mean(g["inv_ret_t11open"], g["t"])
        r3.append({"exit": ex, "period": pn, "events": len(g), "time_exit_share": float(g["exit_why"].str.startswith("시간").mean()),
                   "mean_close": a[0], "t_close": a[1], "mean_next_open": b[0], "t_next_open": b[1]})
    t3 = pd.DataFrame(r3)
    t3.to_csv(RES / "review_t11_b7.csv", index=False, encoding="utf-8-sig")
    print(t3.round(4).to_string(index=False))


# ───────────────────────────── 19번(하): b5 동반 이탈 칸 국면 비중 ─────────────────────────────
def cmd_b5ep():
    import b5_catalyst as B5
    ev, P = B5.load_with_features()
    fr = H.in_period(ev, H.FRONT) & (ev["N"] == 20) & (ev["kind"] == "stock") & ev["sub_cobreak_share"].notna()
    q2 = float(np.quantile(ev.loc[fr, "sub_cobreak_share"], 2 / 3))
    m = H.in_period(ev, H.BACK) & (ev["N"] == 20) & ev["short_ret_h10"].notna() & (ev["sub_cobreak_share"] >= q2)
    g = ev[m]
    s = g.groupby("episode")["short_ret_h10"].agg(["size", "sum", "mean"])
    s["share_of_net"] = s["sum"] / g["short_ret_h10"].sum()
    loss = g.loc[g["short_ret_h10"] < 0]
    s["share_of_losses"] = loss.groupby("episode")["short_ret_h10"].sum() / loss["short_ret_h10"].sum()
    s = s.reset_index()
    s.insert(0, "threshold_q2", q2)
    s.to_csv(RES / "review_b5_cobreak_episode.csv", index=False, encoding="utf-8-sig")
    print(q2, len(g), g["t"].nunique())
    print(s.round(4).to_string(index=False))


# ───────────────────────────── 13번: 다중 비교 요약 ─────────────────────────────
def cmd_multi():
    import re
    rows = []

    def add(name, cells_csv, null_mean, src, posthoc=False):
        t = pd.read_csv(RES / cells_csv)
        rows.append({"branch": name, "cells": len(t), "eligible": int(t["eligible"].sum()),
                     "selected": int(t["selected"].sum()), "pass": int(t["pass"].sum()),
                     "null_mean": null_mean, "null_source": src, "posthoc": posthoc})

    js = lambda f: json.loads((RES / f).read_text(encoding="utf-8"))
    add("b1_shape", "b1_shape_cells.csv", js("b1_shape_summary.json")["summ"]["null"]["mean"], "b1_shape_summary.json summ.null.mean")
    add("b2_confirm", "b2_confirm_cells.csv", js("b2_confirm_summary.json")["null"]["mean"], "b2_confirm_summary.json null.mean")
    add("b3_market", "b3_market_cells.csv", js("b3_market_summary.json")["null"]["mean"], "b3_market_summary.json null.mean")
    add("b4_flow", "b4_flow_cells.csv", js("b4_flow_summary.json")["null"]["mean"], "b4_flow_summary.json null.mean")
    b5 = js("b5_catalyst_summary.json")
    add("b5_catalyst", "b5_catalyst_cells.csv", b5["main"]["null"]["mean"] + b5["variant"]["null"]["mean"],
        "b5_catalyst_summary.json main.null.mean + variant.null.mean")
    add("b6_combo", "b6_combo_cells.csv", js("b6_combo_summary.json")["null"]["mean"], "b6_combo_summary.json null.mean")
    m = re.search(r"통과 칸 수 평균 ([0-9.]+)", (RES / "b7_hedge.md").read_text(encoding="utf-8"))
    add("b7_hedge", "b7_hedge_cells.csv", float(m.group(1)), "b7_hedge.md (c) 선택편향 귀무 문장(요약 json 없음)")
    add("b8_intraday", "b8_intraday_cells.csv", js("b8_intraday_summary.json")["null"]["mean"], "b8_intraday_summary.json null.mean")
    add("사후 b4 하루 묵힌 값", "review_b4_stale_cells.csv", js("review_b4_stale_summary.json")["null"]["mean"],
        "review_b4_stale_summary.json null.mean", True)
    w = js("review_b8_window_summary.json")
    add("사후 b8 진입 창 안", "review_b8_entry_in_cells.csv", w["entry_in"]["null"]["mean"],
        "review_b8_window_summary.json entry_in.null.mean", True)
    add("사후 b8 진입·청산 창 안", "review_b8_both_in_cells.csv", w["both_in"]["null"]["mean"],
        "review_b8_window_summary.json both_in.null.mean", True)
    t = pd.DataFrame(rows)
    pre = t[~t["posthoc"]]
    out = {"rows": t.to_dict("records"),
           "registered": {"cells": int(pre["cells"].sum()), "eligible": int(pre["eligible"].sum()),
                          "pass": int(pre["pass"].sum()), "null_mean_sum": float(pre["null_mean"].sum())},
           "with_posthoc": {"cells": int(t["cells"].sum()), "pass": int(t["pass"].sum()),
                            "null_mean_sum": float(t["null_mean"].sum())}}
    (RES / "review_multiple_comparisons.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(t.to_string(index=False))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    fns = {"inverse": cmd_inverse, "lag": cmd_lag, "b8win": cmd_b8win, "x6": cmd_x6, "regime": cmd_regime,
           "t11": cmd_t11, "b5ep": cmd_b5ep, "multi": cmd_multi}
    for k in (fns if cmd == "all" else [cmd]):
        print(f"\n######## {k}")
        fns[k]()
