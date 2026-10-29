"""F1-A 미래 참조·현실성 감사(얼린 뒤, 확인 단계 결과에 대한 감사). 규칙·매개변수는 바꾸지 않고 '얼마나 깎이나'만 잰다.

F1-A = fam_F1.make({"N": 1, "entry": "confirm", "tgt": 2, "stop": None, "rev": False, "filt": None})
  09:30 봉이 양봉이면 그 봉 고가 위 1분 종가 마감, 음봉이면 저가 아래 마감을 기다려 SOXL/SOXS(1분 지연 다음 봉 종가),
  목표 = 2 × (결정가 → 09:30 봉 반대 끝, SOXL 기준 %) 지정가, 손절 없음, 마감 청산.

재는 것
 A 재현(lib.Engine)과 lib를 쓰지 않는 독립 재구현(parquet 원자료) 날별 일치
 B 매매 시각·보유·청산 구성, 마감 직전 진입(노출 0 왕복)
 C 한 날 몰림: 상위 k일 빼기, 대칭 잘라내기, 중앙값
 D 사람 지연: 결정 뒤 2·3·5분 체결
 E 목표 % 근사: SOXS 매매에 SOXL 거리 대신 SOXS 자기 봉으로 잰 거리
 F 목표 관통 조건 1.0002 → 1.001·1.002
 G 체결·청산 봉이 거래 없는(앞 종가로 채운) 봉이었나
 H 09:30 봉을 야후 1분봉(9/30~10/28)으로 바꿔도 같은 신호인가
 I 방향 성분 = (전략 − 거울)/2, 중립 성분 = (전략 + 거울)/2, 같은 진입 시각 무작위 방향 대조
사용: cd code && py audit_F1-A.py → results/audit_F1-A.json
"""
import json
import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

os.environ["SOS_UNLOCK"] = "frozen"           # 감사: 얼린 후보의 conf·aux 를 다시 잰다(규칙은 안 바꾼다)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                             # noqa: E402
import pandas as pd                            # noqa: E402

import lib                                     # noqa: E402
from lib import Data, Engine, Order, RTH0, summarize, cutoff_idx, idx, hm   # noqa: E402
import fam_F1                                  # noqa: E402

ROOT = os.path.normpath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
P = {"N": 1, "entry": "confirm", "tgt": 2, "stop": None, "rev": False, "filt": None}
SCHEMES = ["actual", "std"]
PERIODS = ["sel", "conf", "aux"]


def m(dl):
    return float(dl["ret"].mean() * 100)


def mt(dl):
    s = summarize(dl, 2000, 20261029)
    return {"mean": round(s["mean_pct"], 4), "t": round(s["t"], 3), "active": s["active_days"]}


# ---------------- D 지연 ----------------
class Delay:
    def __init__(self, inner, extra):
        self.inner, self.extra, self.q = inner, extra, []
        self.start = inner.start

    def on_day(self, ctx):
        if hasattr(self.inner, "on_day"):
            self.inner.on_day(ctx)

    def on_bar(self, ctx, i):
        o = self.inner.on_bar(ctx, i)
        if o is not None:
            self.q.append((i + self.extra, o))
        if self.q and self.q[0][0] <= i:
            return self.q.pop(0)[1]
        return None


class TgtLag:
    """진입 주문은 목표 없이 내고, 체결 뒤 lag분 늦게 같은 목표 %를 keep 주문으로 건다(체결가 기준 그대로)."""
    def __init__(self, inner, lag):
        self.inner, self.lag, self.pending = inner, lag, None
        self.start = inner.start

    def on_day(self, ctx):
        if hasattr(self.inner, "on_day"):
            self.inner.on_day(ctx)

    def on_bar(self, ctx, i):
        o = self.inner.on_bar(ctx, i)
        if o is not None and o.pos != 0 and o.target and not o.keep:
            self.pending = (o.pos, o.stop, o.target, i + self.lag)
            return Order(o.pos, stop=o.stop, target=None)
        if o is None and self.pending and i >= self.pending[3] and ctx.pos == self.pending[0]:
            pos, st, tg, _ = self.pending
            self.pending = None
            return Order(pos, stop=st, target=tg, keep=True)
        return o


# ---------------- E SOXS 자기 거리 ----------------
class ORBNative(fam_F1.ORB):
    """SOXS 매매만 목표 거리를 SOXS 자기 봉으로 잰다: R_s = (P_s − 09:30 봉 SOXS 저가) / P_s."""
    log = []

    def on_bar(self, ctx, i):
        self._ctx, self._i = ctx, i
        return super().on_bar(ctx, i)

    def _order(self, d, P_):
        o = super()._order(d, P_)
        if d < 0:
            ctx, i = self._ctx, self._i
            ps = float(ctx.c("SOXS")[i])
            los = float(ctx.bars("SOXS", "Low")[RTH0:RTH0 + self.N].min())
            rs = (ps - los) / ps
            ORBNative.log.append((ctx.day, (self.hi - P_) / P_, rs))
            self.tfrac = self.tgt * rs
            o = Order(o.pos, stop=o.stop, target=self.tfrac)
        return o


# ---------------- A 독립 재구현(lib 미사용) ----------------
ET, KST = ZoneInfo("America/New_York"), ZoneInfo("Asia/Seoul")


def indep(days, scheme, costs):
    raw = {s: pd.read_parquet(os.path.join(ROOT, "cache", "alp1m", f"{s}.parquet")) for s in ["SOXL", "SOXS"]}
    byday = {s: {d: g for d, g in raw[s].groupby(raw[s].index.date)} for s in raw}

    def hs(sym, t):                      # t = 체결 봉 시작 분(ET, 자정부터)
        b = "pre" if t < 570 else "open5" if t < 575 else "early" if t < 600 else "late"
        return costs[sym][b]

    out = {}
    for day in days:
        if scheme == "std":
            cut = 600
        else:
            k = datetime(day.year, day.month, day.day, tzinfo=KST) + timedelta(days=1)
            e = k.astimezone(ET)
            cut = e.hour * 60 + e.minute
        bars = {}
        for s in ["SOXL", "SOXS"]:
            g = byday[s].get(day)
            mins = g.index.hour * 60 + g.index.minute
            g = g[(mins >= 240) & (mins < 1200)]
            mins = g.index.hour * 60 + g.index.minute
            full = pd.DataFrame(index=range(240, 1200), columns=["o", "h", "l", "c"], dtype=float)
            full.loc[mins, ["o", "h", "l", "c"]] = g[["Open", "High", "Low", "Close"]].values
            c = full["c"].ffill()
            for f in ["o", "h", "l"]:
                full[f] = full[f].fillna(c)
            full["c"] = c
            bars[s] = full
        L = bars["SOXL"]
        o0, h0, l0, c0 = L.loc[570, ["o", "h", "l", "c"]]
        dr = np.sign(c0 - o0)
        r = 0.0
        if dr != 0:
            dec = None
            for t in range(571, cut - 1):            # 결정 봉 571(09:31)~cut−2
                c = L.at[t, "c"]
                if (dr > 0 and c > h0) or (dr < 0 and c < l0):
                    dec = (t, c)
                    break
            if dec is not None:
                t, Pd = dec
                R = (Pd - l0) / Pd if dr > 0 else (h0 - Pd) / Pd
                sym = "SOXL" if dr > 0 else "SOXS"
                B = bars[sym]
                px = B.at[t + 1, "c"] * (1 + hs(sym, t + 1))
                tgt = px * (1 + 2 * R)
                ex = None
                for u in range(t + 2, cut):
                    if B.at[u, "h"] >= tgt * 1.0002:
                        ex = tgt
                        break
                if ex is None:
                    ex = B.at[cut - 1, "c"] * (1 - hs(sym, cut - 1))
                r = ex / px - 1
        out[day] = r
    return out


def main():
    D = Data()
    costs = lib.load_costs()
    res = {"params": P, "factory": fam_F1.make.__module__}
    mk = fam_F1.make(P)
    dl, tr, dly = {}, {}, {}
    for s in SCHEMES:
        E = Engine(D, scheme=s)
        for p in PERIODS:
            dl[(s, p)], tr[(s, p)] = E.run(mk, period=p)

    # A 재현 + 독립
    A = {}
    for s in SCHEMES:
        csv = pd.read_csv(os.path.join(RES, f"confirm_daily_F1-A_{s}.csv"))
        c = csv[csv.period == "conf"]
        mine = dl[(s, "conf")]
        A[f"engine_vs_confirm_{s}_maxdiff_pct"] = float(np.max(np.abs(mine["ret"].values * 100 - c["ret_pct"].values)))
        ind = indep(list(mine["day"]), s, costs)
        v = np.array([ind[d] for d in mine["day"]])
        A[f"indep_vs_engine_{s}_maxdiff_pct"] = float(np.max(np.abs(v - mine["ret"].values)) * 100)
        A[f"indep_{s}_conf_mean_pct"] = float(v.mean() * 100)
        dly[s] = c
    res["A_repro"] = A

    # B 매매 구성
    B = {}
    for s in SCHEMES:
        t = tr[(s, "conf")].copy()
        ent = t["entry"].map(idx)
        ext = t["exit"].map(idx)
        cuts = t["day"].map(lambda d: cutoff_idx(d, s))
        B[s] = {"trades": int(len(t)), "entry_hist": {
                    "09:32-09:34": int(((ent >= idx("09:32")) & (ent < idx("09:35"))).sum()),
                    "09:35-09:59": int(((ent >= idx("09:35")) & (ent < idx("10:00"))).sum()),
                    "10:00-10:29": int(((ent >= idx("10:00")) & (ent < idx("10:30"))).sum()),
                    "10:30+": int((ent >= idx("10:30")).sum())},
                "entry_median": hm(int(np.median(ent))),
                "zero_exposure_roundtrips(entry=cut-1)": int((ent == cuts - 1).sum()),
                "entry_in_last_10min": int((ent >= cuts - 10).sum()),
                "entry_last10_mean_pct": float(t.loc[ent >= cuts - 10, "ret"].mean() * 100) if (ent >= cuts - 10).any() else None,
                "hold_min_median": float(np.median(ext - ent)),
                "why": t["why"].value_counts().to_dict(),
                "mean_by_sym_pct": {k: float(g["ret"].mean() * 100) for k, g in t.groupby("sym")},
                "mean_by_why_pct": {k: float(g["ret"].mean() * 100) for k, g in t.groupby("why")},
                "target_frac_median_pct": None}
        # 목표 크기(진입가 대비): 목표 청산 매매의 수익 ≈ 2R
        tg = t[t.why == "target"]["ret"] * 100
        B[s]["target_ret_median_pct"] = float(tg.median()) if len(tg) else None
    res["B_trades"] = B

    # C 몰림
    C = {}
    for s in SCHEMES:
        for p in ("conf", "aux"):
            r = dl[(s, p)]["ret"].values * 100
            o = np.sort(r)[::-1]
            C[f"{s}_{p}"] = {"n": int(len(r)), "mean": float(r.mean()), "median": float(np.median(r)),
                             "drop_top1": float(o[1:].mean()), "drop_top3": float(o[3:].mean()),
                             "drop_top5": float(o[5:].mean()),
                             "trim5_both": float(o[5:-5].mean()),
                             "top5_days_sum_share_of_total": float(o[:5].sum() / r.sum()) if r.sum() != 0 else None,
                             "top5": [round(float(x), 2) for x in o[:5]], "bottom5": [round(float(x), 2) for x in o[-5:]],
                             "sd": float(r.std(ddof=1))}
    # 상위 5일 날짜
    for s in SCHEMES:
        d = dl[(s, "conf")].copy()
        d["ret"] *= 100
        C[f"{s}_conf_top5_days"] = d.sort_values("ret", ascending=False).head(5)[["day", "ret"]].astype(str).values.tolist()
    res["C_concentration"] = C

    # D 지연
    Dd = {}
    for s in SCHEMES:
        E = Engine(D, scheme=s)
        for ex in (0, 1, 2, 4):
            row = {}
            for p in ("sel", "conf", "aux"):
                d_, _ = E.run(lambda ex=ex: Delay(mk(), ex), period=p)
                row[p] = mt(d_)
            Dd[f"{s}_delay{1 + ex}min"] = row
    res["D_delay"] = Dd

    # E SOXS 자기 거리
    Ed = {}
    for s in SCHEMES:
        E = Engine(D, scheme=s)
        row = {}
        for p in ("sel", "conf", "aux"):
            ORBNative.log = []
            d_, t_ = E.run(lambda: ORBNative(**P), period=p)
            lg = np.array([(a, b) for _, a, b in ORBNative.log])
            row[p] = {**mt(d_), "soxs_trades": int(len(lg)),
                      "Rs_over_R_median": float(np.median(lg[:, 1] / lg[:, 0])) if len(lg) else None,
                      "Rs_over_R_p10_p90": [float(np.percentile(lg[:, 1] / lg[:, 0], 10)),
                                            float(np.percentile(lg[:, 1] / lg[:, 0], 90))] if len(lg) else None,
                      "soxs_trade_mean_pct": float(t_[t_.sym == "SOXS"]["ret"].mean() * 100),
                      "base_mean": round(m(dl[(s, p)]), 4)}
        Ed[s] = row
    res["E_soxs_native_target"] = Ed

    # F 관통 조건
    Fd = {}
    old = lib.TOUCH
    for tv in (1.0002, 1.001, 1.002):
        lib.TOUCH = tv
        for s in SCHEMES:
            E = Engine(D, scheme=s)
            Fd[f"{s}_touch{tv}"] = {p: mt(E.run(mk, period=p)[0]) for p in ("sel", "conf", "aux")}
    lib.TOUCH = old
    res["F_touch"] = Fd

    # G 채운 봉
    G = {}
    for s in SCHEMES:
        t = tr[(s, "conf")]
        stale_in, stale_out = 0, 0
        for _, r in t.iterrows():
            d = D.pos[r["day"]]
            if not D.has[r["sym"]][d, idx(r["entry"])]:
                stale_in += 1
            if not D.has[r["sym"]][d, idx(r["exit"])]:
                stale_out += 1
        G[s] = {"entry_bar_no_trade": stale_in, "exit_bar_no_trade": stale_out, "trades": int(len(t))}
    cd = D.days_in("conf")
    G["conf_missing_bars_0930_1059"] = {sym: int((~D.has[sym][cd, idx("09:30"):idx("11:00")]).sum()) for sym in ["SOXL", "SOXS"]}
    res["G_stale"] = G

    # H 야후 09:30 봉 대조
    yroot = os.path.normpath(os.path.join(ROOT, "..", "soxl_rsi_fade", "cache", "1m"))
    H = {}
    try:
        y = pd.read_parquet(os.path.join(yroot, "SOXL.parquet"))
        y.index = y.index.tz_convert("America/New_York")
        ydays = sorted({d for d in y.index.date})
        rows = []
        for day in ydays:
            if day not in D.pos:
                continue
            d = D.pos[day]
            g = y[y.index.date == day]
            mins = g.index.hour * 60 + g.index.minute
            yb = g[mins == 570]
            if not len(yb):
                continue
            yo, yh, yl, yc = (float(yb[k].iloc[0]) for k in ["Open", "High", "Low", "Close"])
            A_ = D.A["SOXL"]
            ao, ah, al, ac = (float(A_[k][d, RTH0]) for k in ["Open", "High", "Low", "Close"])
            # 야후 09:30 봉으로 같은 confirm 규칙의 결정 분(Alpaca 종가로 판정)
            def dec(o, h, l, c):
                dr = np.sign(c - o)
                if dr == 0:
                    return 0, None
                cl = A_["Close"][d]
                for t_ in range(RTH0 + 1, cutoff_idx(day) - 1):
                    if (dr > 0 and cl[t_] > h) or (dr < 0 and cl[t_] < l):
                        return int(dr), hm(t_)
                return int(dr), None
            rows.append({"day": str(day), "alp": [ao, ah, al, ac], "yah": [yo, yh, yl, yc],
                         "dir_same": bool(np.sign(ac - ao) == np.sign(yc - yo)),
                         "hi_diff_pct": (ah / yh - 1) * 100, "lo_diff_pct": (al / yl - 1) * 100,
                         "dec_alp": dec(ao, ah, al, ac), "dec_yah": dec(yo, yh, yl, yc)})
        H = {"days": len(rows), "dir_disagree": int(sum(not r["dir_same"] for r in rows)),
             "decision_disagree": int(sum(r["dec_alp"] != r["dec_yah"] for r in rows)),
             "hi_absdiff_max_pct": float(max(abs(r["hi_diff_pct"]) for r in rows)),
             "lo_absdiff_max_pct": float(max(abs(r["lo_diff_pct"]) for r in rows)),
             "disagree_rows": [r for r in rows if (r["dec_alp"] != r["dec_yah"]) or not r["dir_same"]]}
    except Exception as ex_:
        H = {"error": repr(ex_)}
    res["H_yahoo_0930"] = H

    # I 방향·중립 성분, 같은 진입 시각 무작위 방향
    I = {}
    for s in SCHEMES:
        c = dly[s]
        st, mi = c["ret_pct"].values, c["mirror_ret_pct"].values
        I[s] = {"strategy": float(st.mean()), "mirror": float(mi.mean()),
                "direction_component": float(((st - mi) / 2).mean()),
                "neutral_component": float(((st + mi) / 2).mean()),
                "always_soxl": float(c["always_soxl_pct"].mean()), "always_soxs": float(c["always_soxs_pct"].mean()),
                "h1": {"dir": float(((st - mi) / 2)[c.half == "h1"].mean()), "neu": float(((st + mi) / 2)[c.half == "h1"].mean())},
                "h2": {"dir": float(((st - mi) / 2)[c.half == "h2"].mean()), "neu": float(((st + mi) / 2)[c.half == "h2"].mean())}}
        # SOXS 쪽 수익이 장 흐름(늘 SOXS) 때문인가: SOXS 매매일의 늘 SOXS 평균
        t = tr[(s, "conf")]
        sd = set(t[t.sym == "SOXS"]["day"])
        ld = set(t[t.sym == "SOXL"]["day"])
        cc = c.set_index(pd.to_datetime(c["day"]).dt.date)
        I[s]["soxs_days_always_soxs_mean"] = float(cc.loc[sorted(sd), "always_soxs_pct"].mean())
        I[s]["soxl_days_always_soxl_mean"] = float(cc.loc[sorted(ld), "always_soxl_pct"].mean())
    res["I_decomp"] = I

    # J 목표 지정가를 체결 뒤 1·2분 늦게 건다(사람이 2R 계산·입력하는 시간)
    J = {}
    for s in SCHEMES:
        E = Engine(D, scheme=s)
        for lag in (1, 2):
            J[f"{s}_target_lag{lag}min"] = {p: mt(E.run(lambda lag=lag: TgtLag(mk(), lag), period=p)[0])
                                            for p in ("sel", "conf", "aux")}
    res["J_target_lag"] = J

    # K 마감 청산을 1·2분 앞당긴다(24:00 KST 직전 여유)
    K = {}
    orig = lib.cutoff_idx
    for k in (1, 2):
        lib.cutoff_idx = lambda day, scheme="actual", k=k: orig(day, scheme) - k
        for s in SCHEMES:
            K[f"{s}_exit_{k}min_early"] = {p: mt(Engine(D, scheme=s).run(mk, period=p)[0]) for p in ("sel", "conf", "aux")}
    lib.cutoff_idx = orig
    res["K_early_exit"] = K

    # H2 10/15 봉 자세히 + 21일 고가·저가 차이 분포
    try:
        y = pd.read_parquet(os.path.join(yroot, "SOXL.parquet"))
        y.index = y.index.tz_convert("America/New_York")
        diffs = []
        for day in sorted({d for d in y.index.date}):
            if day not in D.pos:
                continue
            g = y[y.index.date == day]
            mm = g.index.hour * 60 + g.index.minute
            yb = g[mm == 570]
            d = D.pos[day]
            diffs.append((str(day), round((D.A["SOXL"]["High"][d, RTH0] / float(yb["High"].iloc[0]) - 1) * 100, 3),
                          round((D.A["SOXL"]["Low"][d, RTH0] / float(yb["Low"].iloc[0]) - 1) * 100, 3)))
        day = pd.Timestamp("2026-10-15").date()
        d = D.pos[day]
        g = y[y.index.date == day]
        mm = g.index.hour * 60 + g.index.minute
        det = {"alpaca": {hm(i): [round(float(D.A["SOXL"][k][d, i]), 4) for k in ["Open", "High", "Low", "Close", "Volume"]]
                          for i in range(RTH0 - 2, RTH0 + 4)},
               "yahoo": {f"{x // 60:02d}:{x % 60:02d}": [round(float(v), 4) for v in g[mm == x][["Open", "High", "Low", "Close", "Volume"]].iloc[0]]
                         for x in range(568, 574) if (mm == x).any()}}
        cdl = dl[("actual", "conf")]
        det["F1A_actual_ret_pct_1015"] = float(cdl[cdl.day == day]["ret"].iloc[0] * 100)
        cdl = dl[("std", "conf")]
        det["F1A_std_ret_pct_1015"] = float(cdl[cdl.day == day]["ret"].iloc[0] * 100)
        res["H2_yahoo_detail"] = {"hi_lo_diff_pct_by_day": diffs, "1015": det}
    except Exception as ex_:
        res["H2_yahoo_detail"] = {"error": repr(ex_)}

    out = os.path.join(RES, "audit_F1-A.json")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
