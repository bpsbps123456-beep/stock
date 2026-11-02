"""BRIEF4 C3:H4 독립 재구현(눈가림) — lib.py·ev4_build.py·c1/c2/c3·ev4_events.parquet을 쓰지 않는다.

입력: cache/alp1m_long/SOXL.parquet, SOXS.parquet, cost_table.json 만.
규칙(BRIEF4 사건·결과·조건 절 + lib.py 머리말 체결 문장만 보고 다시 짬):
- 사용자 방향, 문턱 3%: SOXL 1분 종가 >= L*1.03, L = 09:30 봉(또는 앞 사건 청산 다음 봉)부터 지금 봉까지 SOXL 저가 최저.
- 결정 봉 마감 <= 10:45 (= 10:44 봉까지). 한 번에 하나, 청산 다음 봉부터 L 다시 잼.
- 진입 = 결정 봉 다음 봉 종가 * (1 + SOXS 반 호가). 목표 = 체결가 * 1.01, 체결 다음 봉부터 고가 >= 목표*1.0002면 목표가.
  못 닿으면 10:59 봉 종가 * (1 - 반 호가).
- H4: 신호 봉 SOXL (고가 - max(시가, 종가)) / (고가 - 저가) >= 0.5 - 1e-9.
- 빈 분은 그날 앞 종가로 채움(O=H=L=C), 거래량 0.
- 반 호가 시간대: 프리장(<09:30) / 09:30~09:34 / 09:35~09:59 / 10:00~ — 체결 봉의 시작 시각 기준.
"""
import json, os, sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
CACHE = os.path.join(ROOT, "cache", "alp1m_long")
OUT = os.path.join(ROOT, "results")
ET, KST = ZoneInfo("America/New_York"), ZoneInfo("Asia/Seoul")
NMIN = 960


def ix(hhmm):
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m - 240


def tstr(i):
    m = i + 240
    return f"{m // 60:02d}:{m % 60:02d}"


I0930, I1044, I1059 = ix("09:30"), ix("10:44"), ix("10:59")

cost = json.load(open(os.path.join(ROOT, "cost_table.json"), encoding="utf-8"))


def half_spread(sym, i, mult=1.0):
    c = cost[sym]
    if i < ix("09:30"):
        v = c["pre"]
    elif i <= ix("09:34"):
        v = c["open5"]
    elif i <= ix("09:59"):
        v = c["early"]
    else:
        v = c["late"]
    return v * mult


def load(sym):
    f = pd.read_parquet(os.path.join(CACHE, f"{sym}.parquet"), columns=["Open", "High", "Low", "Close"])
    return f


soxl, soxs = load("SOXL"), load("SOXS")
mm = soxl.index.hour * 60 + soxl.index.minute
days = sorted({d for d, m in zip(soxl.index.date, mm) if 570 <= m < 960})
pos = {d: k for k, d in enumerate(days)}


def grid(f):
    mins = (f.index.hour * 60 + f.index.minute - 240).values
    di = np.array([pos.get(d, -1) for d in f.index.date])
    ok = (di >= 0) & (mins >= 0) & (mins < NMIN)
    G = {}
    for c in ["Open", "High", "Low", "Close"]:
        a = np.full((len(days), NMIN), np.nan)
        a[di[ok], mins[ok]] = f[c].values[ok]
        G[c] = a
    # 빈 분: 그날 앞 종가로 O=H=L=C
    miss = np.isnan(G["Close"])
    cl = pd.DataFrame(G["Close"]).ffill(axis=1).values
    for c in ["Open", "High", "Low", "Close"]:
        G[c] = np.where(miss, cl, G[c])
    G["filled"] = miss
    return G


L3 = grid(soxl)
S3 = grid(soxs)
del soxl, soxs


def run_day(d, x=0.03, delay=1, cost_mult=1.0, stop=None):
    """그 날 사용자 방향 사건 목록. delay=결정 봉 뒤 몇 번째 봉 종가로 사나. stop=손절 % (예: 2.0)."""
    k = pos[d]
    lo, cl, op, hi = L3["Low"][k], L3["Close"][k], L3["Open"][k], L3["High"][k]
    so, sh, sl, sc = S3["Open"][k], S3["High"][k], S3["Low"][k], S3["Close"][k]
    evs = []
    i = I0930
    Lmin = np.inf
    Lbar = None
    while i <= I1044:
        if lo[i] < Lmin:
            Lmin, Lbar = lo[i], i
        if cl[i] >= Lmin * (1 + x):
            f = i + delay
            hs_in = half_spread("SOXS", f, cost_mult)
            fill = sc[f] * (1 + hs_in)
            T = fill * 1.01
            SL = fill * (1 - stop / 100) if stop else None
            exit_bar, exit_px, how = None, None, None
            for j in range(f + 1, I1059 + 1):
                if SL is not None and sl[j] <= SL:
                    exit_bar, exit_px, how = j, min(so[j], SL) * (1 - half_spread("SOXS", j, cost_mult)), "stop"
                    break
                if sh[j] >= T * 1.0002:
                    exit_bar, exit_px, how = j, T, "target"
                    break
            if exit_bar is None:
                exit_bar = I1059
                exit_px = sc[I1059] * (1 - half_spread("SOXS", I1059, cost_mult))
                how = "time"
            r = (exit_px / fill - 1) * 100
            seg_lo = sl[f + 1:exit_bar + 1]
            mae = (np.min(seg_lo) / fill - 1) * 100 if len(seg_lo) else 0.0
            rng = hi[i] - lo[i]
            wick = (hi[i] - max(op[i], cl[i])) / rng if rng > 0 else np.nan
            evs.append(dict(day=d, dec_bar=i, dec_close_t=tstr(i + 1), L_bar=Lbar, L=Lmin, soxl_close=cl[i],
                            rise_pct=(cl[i] / Lmin - 1) * 100, fill_bar=f, fill=fill, exit_bar=exit_bar,
                            exit_px=exit_px, how=how, hit=how == "target", r=r, mae=mae, mae0=min(mae, 0.0),
                            mins_to_exit=exit_bar - f, wick=wick,
                            soxl_bar_filled=bool(L3["filled"][k][i]),
                            soxs_any_filled=bool(S3["filled"][k][f:exit_bar + 1].any())))
            # 청산 다음 봉부터 다시
            i = exit_bar + 1
            Lmin, Lbar = np.inf, None
            continue
        i += 1
    return evs


def all_events(**kw):
    rows = []
    for d in days:
        rows += run_day(d, **kw)
    ev = pd.DataFrame(rows)
    ev["day"] = pd.to_datetime(ev["day"])
    ev["H4"] = ev["wick"] >= 0.5 - 1e-9
    return ev


def daily(ev_sel, dlist):
    g = ev_sel.groupby("day")["r"].apply(lambda s: (np.prod(1 + s.values / 100) - 1) * 100)
    return pd.Series([g.get(pd.Timestamp(d), 0.0) for d in dlist], index=pd.to_datetime(dlist))


def boot_mean(vals, n=10000, seed=20261102):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals)
    m = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return m


def stats(ev, year_lo, year_hi, label):
    dl = [d for d in days if year_lo <= d.isoformat() <= year_hi]
    e = ev[(ev["day"] >= pd.Timestamp(dl[0])) & (ev["day"] <= pd.Timestamp(dl[-1]))]
    s = e[e["H4"]]
    rest = e[~e["H4"]]
    dd = daily(s, dl)
    out = dict(label=label, n_days=len(dl), events_all=int(len(e)), trades=int(len(s)),
               active_days=int(s["day"].nunique()))
    if len(s) == 0:
        return out, s, dd
    out.update(hit_rate=round(float(s["hit"].mean()), 3), trade_mean_pct=round(float(s["r"].mean()), 3),
               sum_pct_points=round(float(s["r"].sum()), 2), daily_mean_0fill_pct=round(float(dd.mean()), 4))
    # 매매당 95%: 날 묶음(활동일 복원 추출)
    sd = s.groupby("day")["r"].apply(list)
    keys = list(sd.index)
    rng = np.random.default_rng(20261102)
    tb = []
    for _ in range(10000):
        pick = rng.integers(0, len(keys), len(keys))
        vals = [v for p in pick for v in sd.iloc[p]]
        tb.append(np.mean(vals))
    out["trade_ci95_activedays"] = [round(float(np.percentile(tb, 2.5)), 3), round(float(np.percentile(tb, 97.5)), 3)]
    # 매매당 95%: 그 해 모든 날을 복원 추출(사건 없는 날 포함, 매매 수가 변함) — 원 결과와 같은 관례
    byd = {pd.Timestamp(d): s[s["day"] == pd.Timestamp(d)]["r"].values for d in dl}
    dk = list(byd)
    rng = np.random.default_rng(20261102)
    tb2 = []
    for _ in range(10000):
        v = np.concatenate([byd[dk[p]] for p in rng.integers(0, len(dk), len(dk))])
        if len(v):
            tb2.append(v.mean())
    out["trade_ci95"] = [round(float(np.percentile(tb2, 2.5)), 3), round(float(np.percentile(tb2, 97.5)), 3)]
    bm = boot_mean(dd.values)
    out["daily_ci95"] = [round(float(np.percentile(bm, 2.5)), 3), round(float(np.percentile(bm, 97.5)), 3)]
    out["p_boot_le0"] = round(float(np.mean(bm <= 0)), 5)
    out["p_boot_le0_plus1"] = round(float((np.sum(bm <= 0) + 1) / (len(bm) + 1)), 5)
    out["holm_p_m10"] = round(min(1.0, out["p_boot_le0"] * 10), 4)
    bc = boot_mean(dd.values - dd.mean())
    out["p_boot_null_shift"] = round(float((np.sum(bc >= dd.mean()) + 1) / (len(bc) + 1)), 5)
    from scipy import stats as st
    t = dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))
    out["p_t_onesided"] = round(float(st.t.sf(t, len(dd) - 1)), 5)
    out["by_month_trades"] = {k: int(v) for k, v in s.groupby(s["day"].dt.strftime("%Y-%m")).size().items()}
    out["by_month_sum"] = {k: round(float(v), 3) for k, v in s.groupby(s["day"].dt.strftime("%Y-%m"))["r"].sum().items()}
    out["mae_median"] = round(float(s["mae"].median()), 3)
    out["mae0_median"] = round(float(s["mae0"].median()), 3)
    out["worst_mae"] = round(float(s["mae"].min()), 3)
    h = s[s["hit"]]
    out["min_to_target_median"] = float(h["mins_to_exit"].median()) if len(h) else None
    out["misses"] = [dict(day=str(r.day.date()), r=round(r.r, 3), wick=round(r.wick, 4)) for r in s[~s["hit"]].itertuples()]
    # vs 나머지 사건(같은 면·문턱, H4 아님) — 날 묶음
    if len(rest):
        out["rest_trades"] = int(len(rest))
        out["rest_mean"] = round(float(rest["r"].mean()), 3)
        out["vs_rest_trade_diff"] = round(float(s["r"].mean() - rest["r"].mean()), 3)
        byd = {d: (g[g["H4"]]["r"].values, g[~g["H4"]]["r"].values) for d, g in e.groupby("day")}
        dk = list(byd)
        rng = np.random.default_rng(20261102)
        diffs = []
        for _ in range(10000):
            pick = rng.integers(0, len(dk), len(dk))
            a = np.concatenate([byd[dk[p]][0] for p in pick])
            b = np.concatenate([byd[dk[p]][1] for p in pick])
            if len(a) and len(b):
                diffs.append(a.mean() - b.mean())
        out["vs_rest_ci95"] = [round(float(np.percentile(diffs, 2.5)), 3), round(float(np.percentile(diffs, 97.5)), 3)]
    return out, s, dd


def kst(d, i):
    t = datetime(d.year, d.month, d.day, tzinfo=ET) + timedelta(minutes=i + 240)
    return t.astimezone(KST).strftime("%H:%M")


if __name__ == "__main__":
    ev = all_events()
    print("days", len(days), days[0], days[-1])
    res = {}
    main, s26, dd26 = stats(ev, "2026-01-01", "2026-12-31", "2026")
    res["2026"] = main
    print(json.dumps(main, ensure_ascii=False, indent=1))
    lst = s26.copy()
    lst["entry_kst"] = [kst(r.day.date(), r.fill_bar) for r in lst.itertuples()]
    lst["exit_kst"] = [kst(r.day.date(), r.exit_bar) for r in lst.itertuples()]
    lst["fill_t"] = lst["fill_bar"].map(tstr)
    lst["exit_t"] = lst["exit_bar"].map(tstr)
    lst["L_t"] = lst["L_bar"].map(tstr)
    cols = ["day", "L_t", "dec_close_t", "fill_t", "exit_t", "entry_kst", "exit_kst", "rise_pct", "wick", "fill",
            "exit_px", "how", "r", "mae", "mins_to_exit", "soxl_bar_filled", "soxs_any_filled"]
    pd.set_option("display.width", 250)
    print(lst[cols].to_string())
    # 문턱 근처(0.45~0.55) 사건
    e26 = ev[ev["day"].dt.year == 2026]
    near = e26[(e26["wick"] > 0.45) & (e26["wick"] < 0.55)]
    print("near-threshold 2026:\n", near[["day", "dec_close_t", "wick", "r", "how"]].to_string())
    print("2026 user-dir x=3 events all:", len(e26), "wick NaN:", int(e26["wick"].isna().sum()))
    # 다른 해 재현
    for y in ["2023", "2024", "2025"]:
        o, _, _ = stats(ev, f"{y}-01-01", f"{y}-12-31", y)
        res[y] = {k: o.get(k) for k in ["n_days", "events_all", "trades", "hit_rate", "trade_mean_pct", "daily_mean_0fill_pct",
                                        "daily_ci95", "rest_mean", "vs_rest_trade_diff", "vs_rest_ci95"]}
        print(y, json.dumps(res[y], ensure_ascii=False))
    # 감사: 비용 2배·지연 2·3분·손절 2% (사건 목록은 각 설정으로 다시 생성)
    aud = {}
    for name, kw in [("cost2x", dict(cost_mult=2.0)), ("delay2", dict(delay=2)), ("delay3", dict(delay=3)),
                     ("stop2", dict(stop=2.0))]:
        e2 = all_events(**kw)
        o, s2, d2 = stats(e2, "2026-01-01", "2026-12-31", name)
        aud[name] = {k: o.get(k) for k in ["trades", "hit_rate", "trade_mean_pct", "daily_mean_0fill_pct", "daily_ci95"]}
        print(name, json.dumps(aud[name], ensure_ascii=False))
    # 가장 좋은 달 빼기
    bm = s26.groupby(s26["day"].dt.strftime("%Y-%m"))["r"].sum()
    best = bm.idxmax()
    keep = dd26[dd26.index.strftime("%Y-%m") != best]
    aud["ex_best_month"] = dict(month=best, daily_mean=round(float(keep.mean()), 4),
                                trade_mean=round(float(s26[s26["day"].dt.strftime("%Y-%m") != best]["r"].mean()), 3))
    halves = dict(h1=round(float(dd26[dd26.index.month <= 6].mean()), 4), h2=round(float(dd26[dd26.index.month >= 7].mean()), 4))
    aud["halves"] = halves
    print("ex_best_month", aud["ex_best_month"], "halves", halves)

    # 같은 9개 결정 봉을 고정하고 체결만 바꾼 판(사건 재생성 없이)
    def outcome(d, dec, delay=1, cost_mult=1.0, stop=None, fill_open=False):
        k = pos[d]
        so, sh, sl, sc = (S3[c][k] for c in ["Open", "High", "Low", "Close"])
        f = dec + delay
        fill = (so[f] if fill_open else sc[f]) * (1 + half_spread("SOXS", f, cost_mult))
        T = fill * 1.01
        SL = fill * (1 - stop / 100) if stop else None
        for j in range(f if fill_open else f + 1, I1059 + 1):
            if SL is not None and sl[j] <= SL:
                return (min(so[j], SL) * (1 - half_spread("SOXS", j, cost_mult)) / fill - 1) * 100
            if sh[j] >= T * 1.0002:
                return 1.0
        return (sc[I1059] * (1 - half_spread("SOXS", I1059, cost_mult)) / fill - 1) * 100

    fx = []
    for r in s26.itertuples():
        d = r.day.date()
        fx.append(dict(day=str(d), seq_in_day=int(((ev["day"] == r.day) & (ev["dec_bar"] <= r.dec_bar)).sum()),
                       base=r.r, open_next=outcome(d, r.dec_bar, 1, fill_open=True), delay2=outcome(d, r.dec_bar, 2),
                       delay3=outcome(d, r.dec_bar, 3), stop2=outcome(d, r.dec_bar, 1, stop=2.0),
                       cost2x=outcome(d, r.dec_bar, 1, cost_mult=2.0)))
    FX = pd.DataFrame(fx)
    print(FX.round(3).to_string())
    aud["fixed_events"] = dict(rows=FX.round(4).to_dict("records"),
                               daily_mean_207=(FX.drop(columns=["day", "seq_in_day"]).sum() / len(dd26)).round(4).to_dict())
    print("fixed-event daily mean:", aud["fixed_events"]["daily_mean_207"])
    res["audit"] = aud
    # 결정 봉 마감 해석 감도: 10:45 봉까지 허용
    I1044_save = I1044
    I1044 = ix("10:45")
    e3 = all_events()
    o3, s3, _ = stats(e3, "2026-01-01", "2026-12-31", "dec<=10:45bar")
    I1044 = I1044_save
    res["sens_dec_1045bar"] = {k: o3.get(k) for k in ["trades", "trade_mean_pct", "daily_mean_0fill_pct"]}
    print("sens dec<=10:45 bar", res["sens_dec_1045bar"])
    lst_out = lst[cols].copy()
    lst_out["day"] = lst_out["day"].dt.strftime("%Y-%m-%d")
    lst_out.to_csv(os.path.join(OUT, "verify4_C3_H4_trades_2026.csv"), index=False)
    pd.DataFrame({"day": dd26.index.strftime("%Y-%m-%d"), "r": dd26.values}).to_csv(
        os.path.join(OUT, "verify4_C3_H4_daily_2026.csv"), index=False)
    json.dump(res, open(os.path.join(OUT, "verify4_C3_H4.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
