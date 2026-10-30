"""BRIEF3 'H0' 검정 — SOXL이 3% 더 오르면 SOXS를 사서 +1%(거울: SOXS 3% → SOXL). 고르는 값 없음, 한 번에 잰다.

결과 보기 전에 정한 해석(바꾸지 않는다)
1. V1(H0): 기준 L = 09:30 봉(또는 앞 매매 청산 다음 봉)부터 지금 봉까지 그 종목 저가의 최저. 봉 i 종가 ≥ L × 1.03이면
   봉 i 마감에 반대 종목 시장가 주문(엔진: 봉 i+1 종가 + 반 호가). 목표 +1% 지정가(고가 ≥ 목표 × 1.0002), 손절 없음,
   못 닿으면 창 끝 봉(w90 10:59, std 09:59) 종가 − 반 호가. 진입 마감 = 결정 봉 마감 ≤ 창 끝 15분 전(i ≤ cut − 16).
   들고 있는 동안(주문 낸 봉부터 청산 봉까지) 신호는 무시하고, 청산 봉 다음 봉부터 두 종목의 L을 다시 잰다.
   청산 봉 자체에서는 새로 들어가지 않는다.
2. 같은 봉에 두 종목이 함께 문턱을 넘으면 (종가/기준) 비율이 큰 쪽을 따른다(드묾, 건수를 센다).
3. 다른 읽기(범위만 보고, 판정 없음): V2 = 오늘 09:30 봉 시가, V3 = 전날 정규장 종가(갭 포함),
   V4 = 최근 15봉(i−14..i) 저가 최저, V5 = 최근 30봉 저가 최저. V4·V5 창은 09:30·청산 다음 봉 앞으로 넘어가지 않는다.
   V2·V3는 종목마다 하루 한 번만 신호를 낸다: 그 종목이 처음 문턱을 넘은 봉에서 신호가 쓰이고, 그때 들고 있거나
   진입 마감이 지났으면 그 신호는 버린다(뒤에 다시 나지 않는다).
4. 매매당 통계: +1% 도달률, 평균, 못 닿은 매매(창 끝 청산) 분포, MAE = min(체결 다음 봉부터 청산 봉까지 저가 최저 / 체결가 − 1, 0)
   (청산 봉 저가 포함 = 보수적; 체결가 아래로 한 번도 안 갔으면 0 — '불리한' 폭이라 양수가 없다. 2026-10-30 눈가림 대조에서
   통일, 883건 중 4건만 바뀜), 도달까지 분 = 목표 봉 − 체결 봉.
5. 대조(매매당, 같은 결정 봉·같은 청산 규칙으로 따로 잰 매매): 추종 = 오른 종목을 산다, 같은 분 무작위 = (H0 + 추종)/2.
   그래서 'H0 > 같은 분 무작위' ⇔ 'H0 > 추종'이고 방향 성분 = (H0 − 추종)/2. 창 무작위 = 결정 가능한 모든 봉
   (09:30 ~ cut−16) × 두 종목. 추종은 따로 '독립 실행 전략'(같은 신호 규칙, 산 종목만 반대)으로도 잰다.
6. 지연 2·3분 = 결정을 1·2봉 늦게 내는 래퍼(Delay). 시가 체결 = Engine(fill='open'). 손절 2·3% = Order.stop.
   감도는 판정에 쓰지 않는다.
7. H0 통과(방식마다): 2026 하루 평균 > 0, 95% 하한 > 0, 1~6월·7~10월 모두 > 0, 비용 2배 2026 평균 > 0,
   2026 매매당 평균 > 같은 분 무작위 매매당 평균.
8. 엔진 결과를 이 파일의 독립 매매 계산(sim_trade)과 독립 하루 반복(indep_day)으로 다시 계산해 일치를 확인한다.

사용: cd code && py h0.py → results/h0.json, results/h0_trades_<변형>_<방식>.csv, results/h0_daily_<변형>_<방식>.csv
"""
import json
import os
import sys
import time

os.environ["SOS_UNLOCK"] = "frozen"          # BRIEF3 H0는 고르는 값이 없어 2026 전체·2025를 한 번에 잰다(오케스트레이터 지시)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402
from lib import (Data, Engine, Order, INST, TOUCH, RTH0, ROOT, PERIODS, bucket, cutoff_idx, hm, idx,  # noqa: E402
                 load_costs, summarize)

THR, TGT = 0.03, 0.01
RES = os.path.join(ROOT, "results")
COSTS = load_costs()
SCHEMES = ["w90", "std"]
VARIANTS = {"V1": "low", "V2": "open", "V3": "prev", "V4": "w15", "V5": "w30"}
VNAME = {"V1": "H0 장중 최저 대비(청산 뒤 다시 잼)", "V2": "오늘 09:30 시가 대비(하루 1번)",
         "V3": "전날 정규장 종가 대비, 갭 포함(하루 1번)", "V4": "최근 15분 저가 대비", "V5": "최근 30분 저가 대비"}
SIDE = {-1: "SOXL_up→buy_SOXS", 1: "SOXS_up→buy_SOXL"}
PLAB = {"sel": "2026H1", "conf": "2026H2", "aux": "2025"}


class H0:
    """한 번에 한 종목. buy=-1(SOXL 오름 → SOXS 매수), +1(SOXS 오름 → SOXL 매수). follow=True면 오른 종목을 산다."""
    start = "09:30"

    def __init__(self, ref="low", thr=THR, target=TGT, stop=None, follow=False, sides=(-1, 1), log=None):
        self.ref, self.thr, self.target, self.stop = ref, thr, target, stop
        self.follow, self.sides, self.log = follow, set(sides), log
        self.once = ref in ("open", "prev")
        self.lag = 1                 # 결정 봉 → 체결 봉 거리(Delay가 바꾼다)
        self.state, self.fired = "flat", set()

    def on_day(self, ctx):
        self.s0 = ctx.i + 1          # 09:30
        if self.ref == "prev":
            self.pc = {s: (ctx.prev_day(s) or {}).get("close", np.nan) for s in ("SOXL", "SOXS")}

    def level(self, ctx, sym, i):
        if self.ref == "low":
            return ctx.bars(sym, "Low")[self.s0:i + 1].min()
        if self.ref == "open":
            return ctx.bars(sym, "Open")[RTH0]
        if self.ref == "prev":
            return self.pc[sym]
        w = 15 if self.ref == "w15" else 30
        return ctx.bars(sym, "Low")[max(self.s0, i - w + 1):i + 1].min()

    def on_bar(self, ctx, i):
        just_exited = False
        if self.state == "ordered":
            if i >= self.fill_bar and ctx.pos == 0:      # 청산이 이 봉에서 일어났다
                self.state, self.s0, just_exited = "flat", i + 1, True
            elif not self.once:
                return None
        if not self.once and (just_exited or i < self.s0):
            return None
        cands = []
        for sym, buy in (("SOXL", -1), ("SOXS", 1)):
            if self.once and sym in self.fired:
                continue
            c, ref = ctx.c(sym)[i], self.level(ctx, sym, i)
            if np.isfinite(c) and np.isfinite(ref) and ref > 0 and c >= ref * (1 + self.thr):
                cands.append((c / ref, sym, buy, ref, c))
        if self.once:
            self.fired |= {x[1] for x in cands}          # 처음 넘은 봉에서 신호가 쓰인다
        if self.state != "flat" or just_exited or i > ctx.cut - 16:
            return None
        cands = [x for x in cands if x[2] in self.sides]
        if not cands:
            return None
        cands.sort(reverse=True)
        ratio, sym, buy, ref, c = cands[0]
        pos = -buy if self.follow else buy
        self.state, self.fill_bar = "ordered", i + self.lag
        if self.log is not None:
            self.log.append({"d": ctx.d, "dec": i, "fill": i + self.lag, "trig": sym, "buy": buy, "pos": pos,
                             "ref": ref, "trig_close": c, "ratio": ratio, "tie": len(cands) > 1})
        return Order(pos, stop=self.stop, target=self.target)


class Delay:
    """결정을 k봉 늦게 낸다(사람 지연 1+k분)."""
    def __init__(self, inner, k):
        self.inner, self.k, self.q = inner, k, {}
        self.start = inner.start
        inner.lag = 1 + k

    def on_day(self, ctx):
        self.inner.on_day(ctx)

    def on_bar(self, ctx, i):
        o = self.inner.on_bar(ctx, i)
        if o is not None:
            self.q[i + self.k] = o
        return self.q.pop(i, None)


def sim_trade(D, d, dec, pos, cut, target=TGT, stop=None, fill="close", lag=1, cm=1.0):
    """엔진과 같은 체결 규칙으로 한 매매만 잰다. 반환: (수익, 체결 봉, 청산 봉, 사유, MAE)."""
    sym = INST[pos]
    A = D.A[sym]
    hs = lambda j: COSTS[sym][bucket(j)] * cm                 # noqa: E731
    f = dec + lag
    px = A["Close" if fill == "close" else "Open"][d, f] * (1 + hs(f))
    tgt = px * (1 + target) if target else None
    stp = px * (1 - stop) if stop else None
    j0 = f + 1 if fill == "close" else f
    out = why = None
    for j in range(j0, cut):
        if stp is not None and A["Low"][d, j] <= stp:
            out, why = min(A["Open"][d, j], stp) * (1 - hs(j)), "stop"
            break
        if tgt is not None and A["High"][d, j] >= tgt * TOUCH:
            out, why = tgt, "target"
            break
    if why is None:
        j = cut - 1
        out, why = A["Close"][d, j] * (1 - hs(j)), "cutoff"
    mae = min(A["Low"][d, j0:j + 1].min() / px - 1, 0.0)      # 불리한 폭: 체결가 아래로 안 갔으면 0
    return out / px - 1, f, j, why, mae


def indep_day(D, d, ref, cut, sides=(-1, 1)):
    """Ctx·Engine 없이 다시 쓴 V1~V5 하루 반복(점검용, fill='close', 지연 1분, 손절 없음)."""
    s0, i, fired, out = RTH0, RTH0, set(), []
    pc = {s: D.daily.loc[(s, D.days[d - 1])]["close"] for s in ("SOXL", "SOXS")} if ref == "prev" else None

    def lvl(sym, i, s0):
        A = D.A[sym]
        if ref == "low":
            return A["Low"][d, s0:i + 1].min()
        if ref == "open":
            return A["Open"][d, RTH0]
        if ref == "prev":
            return pc[sym]
        w = 15 if ref == "w15" else 30
        return A["Low"][d, max(s0, i - w + 1):i + 1].min()

    busy_until = -1
    while i <= cut - 2:
        cands = []
        if ref in ("open", "prev") or (i > busy_until and i >= s0):
            for sym, buy in (("SOXL", -1), ("SOXS", 1)):
                if ref in ("open", "prev") and sym in fired:
                    continue
                c, L = D.A[sym]["Close"][d, i], lvl(sym, i, s0)
                if c >= L * (1 + THR):
                    cands.append((c / L, sym, buy))
            if ref in ("open", "prev"):
                fired |= {x[1] for x in cands}
        cands = [x for x in cands if x[2] in sides]
        if i > busy_until and i >= s0 and i <= cut - 16 and cands:
            cands.sort(reverse=True)
            r, f, j, why, mae = sim_trade(D, d, i, cands[0][2], cut)
            out.append((i, cands[0][2], r))
            busy_until, s0 = j, j + 1
        i += 1
    return out


def trade_stats(t):
    """매매당 통계(수익 %)."""
    if len(t) == 0:
        return {"n": 0}
    r = t["ret_pct"].values
    cut = t[t["why"] == "cutoff"]["ret_pct"].values
    tg = t[t["why"] == "target"]
    o = {"n": int(len(r)), "hit_rate": float((t["why"] == "target").mean()),
         "mean_pct": float(r.mean()), "median_pct": float(np.median(r)),
         "sd_pct": float(r.std(ddof=1)) if len(r) > 1 else None,
         "win_rate": float((r > 0).mean()),
         "stop_n": int((t["why"] == "stop").sum()),
         "cutoff_n": int(len(cut))}
    if len(cut):
        o["cutoff_pct"] = {"mean": float(cut.mean()), "min": float(cut.min()),
                           "p10": float(np.percentile(cut, 10)), "p25": float(np.percentile(cut, 25)),
                           "median": float(np.median(cut)), "p75": float(np.percentile(cut, 75)),
                           "p90": float(np.percentile(cut, 90)), "max": float(cut.max()),
                           "share_below_-3": float((cut < -3).mean())}
    if "mae_pct" in t:
        m = t["mae_pct"].values
        o["mae_pct"] = {"mean": float(m.mean()), "median": float(np.median(m)), "p10": float(np.percentile(m, 10)),
                        "min": float(m.min())}
    if len(tg) and "min_to_target" in t:
        mt = tg["min_to_target"].values
        o["min_to_target"] = {"mean": float(mt.mean()), "median": float(np.median(mt)),
                              "p75": float(np.percentile(mt, 75)), "p90": float(np.percentile(mt, 90))}
    if "follow_pct" in t:
        fo = t["follow_pct"].values
        o["follow_mean_pct"] = float(fo.mean())
        o["follow_hit_rate"] = float((t["follow_why"] == "target").mean())
        o["rand_same_mean_pct"] = float(t["rand_same_pct"].mean())
        o["beats_rand_same"] = bool(r.mean() > t["rand_same_pct"].mean())
        o["direction_component_pct"] = float(((r - fo) / 2).mean())
        o["direction_ci95"] = cluster_ci(t["day"].values, (r - fo) / 2)
        o["ties"] = int(t["tie"].sum())
    o["per_trade_ci95"] = cluster_ci(t["day"].values, r)
    return o


def cluster_ci(days, x, nboot=10000, seed=20261029):
    """날 묶음 부트스트랩 매매당 평균 95% 구간."""
    df = pd.DataFrame({"d": days, "x": x}).groupby("d")["x"].agg(["sum", "count"])
    S, N = df["sum"].values, df["count"].values
    if len(S) < 2:
        return None
    rng = np.random.default_rng(seed)
    k = rng.integers(0, len(S), (nboot, len(S)))
    bs = S[k].sum(1) / N[k].sum(1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def window_random(D, days, scheme, target=TGT):
    """창 무작위: 결정 가능한 모든 봉(09:30 ~ cut−16) × 두 종목, 같은 청산 규칙(fill='close', 지연 1분)."""
    rows = []
    for d in days:
        cut = cutoff_idx(D.days[d], scheme)
        I = np.arange(RTH0, cut - 15)
        F = I + 1
        J = np.arange(cut)
        for pos in (1, -1):
            sym = INST[pos]
            A = D.A[sym]
            hsF = np.array([COSTS[sym][bucket(f)] for f in F])
            px = A["Close"][d, F] * (1 + hsF)
            tgt = px * (1 + target)
            M = (A["High"][d, :cut][None, :] >= tgt[:, None] * TOUCH) & (J[None, :] > F[:, None])
            hit = M.any(1)
            out = np.where(hit, tgt, A["Close"][d, cut - 1] * (1 - COSTS[sym][bucket(cut - 1)]))
            r = (out / px - 1) * 100
            for k in range(len(I)):
                rows.append((D.days[d], sym, r[k], hit[k]))
    w = pd.DataFrame(rows, columns=["day", "sym", "ret_pct", "hit"])
    per_day = w.groupby("day")["ret_pct"].mean()
    return {"n": int(len(w)), "mean_pct": float(w["ret_pct"].mean()), "hit_rate": float(w["hit"].mean()),
            "by_sym_mean_pct": {s: float(g["ret_pct"].mean()) for s, g in w.groupby("sym")},
            "per_day_mean_pct": float(per_day.mean()),
            "per_trade_ci95": cluster_ci(w["day"].values, w["ret_pct"].values)}


def period_of(D, d):
    day = D.days[d]
    for p, (a, b) in PERIODS.items():
        if pd.Timestamp(a).date() <= day <= pd.Timestamp(b).date():
            return PLAB[p]
    return None


def run(D, scheme, ref="low", fill="close", cm=1.0, stop=None, delay=0, follow=False, sides=(-1, 1), days=None,
        enrich=True):
    """엔진 실행 → (날별, 매매) DataFrame. enrich면 매매마다 대조·MAE·도달 분을 붙이고 sim_trade와 일치를 확인한다."""
    log = []

    def make():
        s = H0(ref=ref, stop=stop, follow=follow, sides=sides, log=log)
        return Delay(s, delay) if delay else s
    E = Engine(D, fill=fill, scheme=scheme, cost_mult=cm)
    daily, tr = E.run(make, days=days)
    assert len(log) == len(tr), (len(log), len(tr))
    lg = pd.DataFrame(log)
    tr = tr.reset_index(drop=True)
    daily["period"] = [period_of(D, D.pos[x]) for x in daily["day"]]
    daily["ret_pct"] = daily["ret"] * 100
    if len(tr) == 0:
        return daily, tr, {"max_abs_diff_vs_sim": 0.0}
    tr["d"] = lg["d"].values
    tr["dec_i"] = lg["dec"].values
    tr["fill_i"] = lg["fill"].values
    assert (tr["entry"].values == np.array([hm(i) for i in lg["fill"]])).all()
    tr["period"] = [period_of(D, d) for d in tr["d"]]
    tr["side"] = [SIDE[b] for b in lg["buy"]]
    tr["trig"] = lg["trig"].values
    tr["ref"] = lg["ref"].values
    tr["trig_close"] = lg["trig_close"].values
    tr["ratio"] = lg["ratio"].values
    tr["tie"] = lg["tie"].values
    tr["dec_close"] = [hm(i + 1) for i in lg["dec"]]
    tr["ret_pct"] = tr["ret"] * 100
    chk = {}
    if enrich:
        lag = 1 + delay
        sims, fol, mae, mtt, fwhy = [], [], [], [], []
        for _, row in tr.iterrows():
            d, dec, pos = int(row["d"]), int(row["dec_i"]), int(lg.loc[_, "pos"])
            cut = cutoff_idx(D.days[d], scheme)
            r, f, j, why, m = sim_trade(D, d, dec, pos, cut, stop=stop, fill=fill, lag=lag, cm=cm)
            assert why == row["why"] and hm(j) == row["exit"], (row.to_dict(), why, hm(j))
            sims.append(r)
            mae.append(m * 100)
            mtt.append(j - f if why == "target" else np.nan)
            r2, _f, _j, why2, _m = sim_trade(D, d, dec, -pos, cut, stop=stop, fill=fill, lag=lag, cm=cm)
            fol.append(r2 * 100)
            fwhy.append(why2)
        chk["max_abs_diff_vs_sim"] = float(np.max(np.abs(np.array(sims) - tr["ret"].values)))
        tr["mae_pct"] = mae
        tr["min_to_target"] = mtt
        tr["follow_pct"] = fol
        tr["follow_why"] = fwhy
        tr["rand_same_pct"] = (tr["ret_pct"] + tr["follow_pct"]) / 2
    return daily, tr, chk





def by_period(daily, tr):
    out = {}
    for lab, ps in (("2026", ["2026H1", "2026H2"]), ("2026H1", ["2026H1"]), ("2026H2", ["2026H2"]), ("2025", ["2025"])):
        dd = daily[daily["period"].isin(ps)]
        s = summarize(dd)
        s["mean_active_day_pct"] = float(dd.loc[dd["n"] > 0, "ret_pct"].mean()) if (dd["n"] > 0).any() else None
        t = tr[tr["period"].isin(ps)] if len(tr) else tr
        out[lab] = {"daily": s, "trades": trade_stats(t) if len(t) else {"n": 0}}
    return out


def short(bp):
    """보고용 한 줄 요약."""
    o = {}
    for lab, v in bp.items():
        s, t = v["daily"], v["trades"]
        o[lab] = {"mean_daily": s["mean_pct"], "ci95": s.get("ci95"), "t": s.get("t"), "active_days": s["active_days"],
                  "days": s["days"], "trades": t.get("n", 0), "per_trade_mean": t.get("mean_pct"),
                  "hit_rate": t.get("hit_rate"), "max_dd": s.get("max_dd_pct")}
        if "rand_same_mean_pct" in t:
            o[lab]["rand_same_mean"] = t["rand_same_mean_pct"]
            o[lab]["follow_mean"] = t["follow_mean_pct"]
    return o


def save(daily, tr, name):
    daily[["day", "period", "ret_pct", "n"]].to_csv(os.path.join(RES, f"h0_daily_{name}.csv"), index=False)
    cols = ["day", "period", "side", "trig", "sym", "ref", "trig_close", "ratio", "dec_close", "entry", "exit",
            "entry_px", "exit_px", "ret_pct", "why", "min_to_target", "mae_pct", "follow_pct", "follow_why",
            "rand_same_pct", "tie"]
    tr[[c for c in cols if c in tr]].to_csv(os.path.join(RES, f"h0_trades_{name}.csv"), index=False)


def main():
    t0 = time.time()
    D = Data()
    days = sorted(D.days_in("aux") + D.days_in("sel") + D.days_in("conf"))
    d26 = sorted(D.days_in("sel") + D.days_in("conf"))
    out = {"meta": {"brief": "BRIEF3.md H0", "thr": THR, "target": TGT, "stop": None, "entry_deadline": "결정 봉 마감 ≤ 창 끝 15분 전",
                    "fill": "다음 봉 종가 + 반 호가(cost_table.json), 목표 지정가 고가 ≥ 목표×1.0002, 창 끝 봉 종가 − 반 호가",
                    "periods": {"2026H1": PERIODS["sel"], "2026H2": PERIODS["conf"], "2025": PERIODS["aux"]},
                    "n_days": {"2026": len(d26), "2026H1": len(D.days_in("sel")), "2026H2": len(D.days_in("conf")),
                               "2025": len(D.days_in("aux"))},
                    "bootstrap": "날 묶음 10,000회 seed 20261029", "unlock": "SOS_UNLOCK=frozen(스크립트 안)",
                    "units": "%(1.0 = 1%)"},
           "primary": {}, "variants": {}, "sensitivity": {}, "by_side": {}, "controls": {}, "checks": {}, "pass": {}}
    # 1) 다섯 읽기 × 두 방식
    base = {}
    for v, ref in VARIANTS.items():
        out["variants"][v] = {"name": VNAME[v]}
        for sc in SCHEMES:
            daily, tr, chk = run(D, sc, ref=ref, days=days)
            save(daily, tr, f"{v}_{sc}")
            bp = by_period(daily, tr)
            out["variants"][v][sc] = {"by_period": bp, "short": short(bp)}
            out["checks"][f"{v}_{sc}_engine_vs_sim"] = chk["max_abs_diff_vs_sim"]
            # 독립 하루 반복과 일치
            mism = 0
            g = tr.groupby("d") if len(tr) else {}
            for d in days:
                cut = cutoff_idx(D.days[d], sc)
                ind = indep_day(D, d, ref, cut)
                eng = [] if d not in g.groups else [(int(a), int(b), float(c)) for a, b, c in
                                                     zip(g.get_group(d)["dec_i"], [(-1 if s == "SOXS" else 1) for s in g.get_group(d)["sym"]],
                                                         g.get_group(d)["ret"])]
                if len(ind) != len(eng) or any(a[0] != b[0] or a[1] != b[1] or abs(a[2] - b[2]) > 1e-12 for a, b in zip(ind, eng)):
                    mism += 1
            out["checks"][f"{v}_{sc}_indep_day_mismatch_days"] = mism
            if v == "V1":
                base[sc] = (daily, tr)
            print(f"{v} {sc}: {short(bp)['2026']}  ({time.time() - t0:.0f}s)", flush=True)
    # 2) 대조
    for sc in SCHEMES:
        daily, tr = base[sc]
        c = {"same_minute": {}, "window_random": {}, "follow_strategy": {}}
        for lab, ps in (("2026", ["2026H1", "2026H2"]), ("2026H1", ["2026H1"]), ("2026H2", ["2026H2"]), ("2025", ["2025"])):
            t = tr[tr["period"].isin(ps)]
            c["same_minute"][lab] = {"h0_mean_pct": float(t["ret_pct"].mean()), "follow_mean_pct": float(t["follow_pct"].mean()),
                                     "rand_same_mean_pct": float(t["rand_same_pct"].mean()),
                                     "h0_hit": float((t["why"] == "target").mean()),
                                     "follow_hit": float((t["follow_why"] == "target").mean()),
                                     "direction_component_pct": float(((t["ret_pct"] - t["follow_pct"]) / 2).mean()),
                                     "direction_ci95": cluster_ci(t["day"].values, ((t["ret_pct"] - t["follow_pct"]) / 2).values)}
        for lab, pk in (("2026", ["sel", "conf"]), ("2026H1", ["sel"]), ("2026H2", ["conf"]), ("2025", ["aux"])):
            dd = sorted(sum((D.days_in(p) for p in pk), []))
            c["window_random"][lab] = window_random(D, dd, sc)
        fd, ft, _ = run(D, sc, ref="low", follow=True, days=days, enrich=False)
        c["follow_strategy"] = short(by_period(fd, ft))
        out["controls"][sc] = c
        print(f"controls {sc} done ({time.time() - t0:.0f}s)", flush=True)
    # 3) 감도(V1)
    sens = {"cost2x": dict(cm=2.0), "fill_open": dict(fill="open"), "delay2": dict(delay=1), "delay3": dict(delay=2),
            "stop2": dict(stop=0.02), "stop3": dict(stop=0.03)}
    for sc in SCHEMES:
        out["sensitivity"][sc] = {}
        for k, kw in sens.items():
            daily, tr, chk = run(D, sc, ref="low", days=days, **kw)
            save(daily, tr, f"V1-{k}_{sc}")
            bp = by_period(daily, tr)
            out["sensitivity"][sc][k] = short(bp)
            out["checks"][f"V1-{k}_{sc}_engine_vs_sim"] = chk.get("max_abs_diff_vs_sim")
        print(f"sens {sc} done ({time.time() - t0:.0f}s)", flush=True)
    # 4) 방향별(V1): 같은 실행의 매매를 나눔 + 한쪽만 하는 독립 실행
    for sc in SCHEMES:
        daily, tr = base[sc]
        bs = {}
        for buy, lab in SIDE.items():
            t = tr[tr["side"] == lab]
            o = {}
            for pl, ps in (("2026", ["2026H1", "2026H2"]), ("2026H1", ["2026H1"]), ("2026H2", ["2026H2"]), ("2025", ["2025"])):
                o[pl] = trade_stats(t[t["period"].isin(ps)]) if len(t[t["period"].isin(ps)]) else {"n": 0}
            sd, st, _ = run(D, sc, ref="low", sides=(buy,), days=days, enrich=False)
            bs[lab] = {"trades_in_main_run": o, "side_only_run": short(by_period(sd, st))}
        out["by_side"][sc] = bs
    # 5) 판정
    for sc in SCHEMES:
        daily, tr = base[sc]
        bp = out["variants"]["V1"][sc]["by_period"]
        s26 = bp["2026"]["daily"]
        t26 = bp["2026"]["trades"]
        c2 = out["sensitivity"][sc]["cost2x"]["2026"]["mean_daily"]
        crit = {"mean_2026_gt0": s26["mean_pct"] > 0, "ci_low_gt0": s26["ci95"][0] > 0,
                "h1_gt0": bp["2026H1"]["daily"]["mean_pct"] > 0, "h2_gt0": bp["2026H2"]["daily"]["mean_pct"] > 0,
                "cost2x_gt0": c2 > 0, "per_trade_gt_rand_same": t26["mean_pct"] > t26["rand_same_mean_pct"]}
        out["pass"][sc] = {"criteria": crit, "pass": all(crit.values()),
                           "values": {"mean_2026": s26["mean_pct"], "ci95": s26["ci95"], "h1": bp["2026H1"]["daily"]["mean_pct"],
                                      "h2": bp["2026H2"]["daily"]["mean_pct"], "cost2x_2026": c2,
                                      "per_trade_2026": t26["mean_pct"], "rand_same_2026": t26["rand_same_mean_pct"]}}
        out["primary"][sc] = {"short": out["variants"]["V1"][sc]["short"], "by_period": bp}
    out["meta"]["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(RES, "h0.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print(json.dumps(out["pass"], ensure_ascii=False, indent=1, default=str))
    print(json.dumps(out["checks"], ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
