"""지지·저항 반영 초단타: 추매(평균 단가 낮추기) x 매도 목표(고정/새 저항선) x 지지선 필터. 프리장(깨어 있을 때)·본장.
기본 매수 = 추세 따라(칼만 240분 구간 매수·과매수면 전일 종가 이하, 아니면 -1%). 목표 미달이면 다음 날 프리장 정리."""
import itertools
import numpy as np, pandas as pd
from numba import njit
from core import *
from evalcfg import data, indicator
from grid_short import portfolio, window_metrics
from run_on import ctx, day_arrays
from run_adapt import day_info
from run_sr import pivots
from run_v5 import eth_native

D, H = data()
KMAX = 12


def sr_table(S, s, win):
    """세션별(날짜) 개장 전(프리장 시작 전)까지 확정된 스윙으로 만든 지지·저항 군집 가격(오름차순, 최대 KMAX)."""
    A = S[s]
    if win == "short":
        tb = tf_bars(D[(s, "5m")], 40, True, H); per_day = 24
    else:
        tb = eth_native(s); per_day = 17
    pli, plp, phi, php = pivots(tb.L.values, tb.H.values, 5)
    idx = np.r_[pli, phi]; px = np.r_[plp, php]
    end = tb.end_ns.values
    n_days = len(A["days"])
    out = np.full((n_days, KMAX), np.nan)
    t = pd.to_datetime(A["start_ns"], utc=True).tz_convert("America/New_York")
    first = pd.Series(A["start_ns"]).groupby(A["day_idx"]).min()
    for d, st in first.items():
        last = np.searchsorted(end, st, side="right") - 1
        sel = (idx <= last) & (idx >= last - 10 * per_day)
        p = np.sort(px[sel])
        if len(p) == 0:
            continue
        cl, cur = [], [p[0]]
        for v in p[1:]:
            if (v - cur[0]) / cur[0] <= 0.004:
                cur.append(v)
            else:
                cl.append(np.mean(cur)); cur = [v]
        cl.append(np.mean(cur))
        cl = np.array(cl)
        out[d, :min(KMAX, len(cl))] = cl[:KMAX] if len(cl) <= KMAX else cl[np.argsort(np.abs(cl - np.median(cl)))[:KMAX]]
        out[d] = np.sort(out[d])
    return out


@njit(cache=True)
def sim_sr(o, h, l, c, is_reg, is_pre, is_awake, is_end, first_reg, day, j0, lvl, srt, add_pct, tp_mode, tp_fix, close_fill):
    """tp_mode 0 고정(평균 단가 + tp_fix), 1 새 저항선(평균 단가 위 가장 가까운 군집, +0.5~3%)."""
    n = len(o)
    cash, sh = 1.0, 0.0
    holding, added, tp_on = False, False, False
    first_px, avg, eday, tp = 0.0, 0.0, -1, 0.0
    half = add_pct > 0
    n_end = 0
    for j in range(j0, n):
        if is_end[j]:
            n_end += 1
    eq = np.empty(n_end)
    k = 0
    ntr, nadd = 0, 0
    for j in range(j0, n):
        d = day[j]
        if holding and (not tp_on) and is_awake[j]:
            tp_on = True
        pre_ok = is_pre[j] and is_awake[j]
        can = is_reg[j] or pre_ok
        if holding and can:
            hit = (c[j] >= tp) if close_fill else (h[j] >= tp)
            if d >= eday + 1 and (pre_ok or is_reg[j]):
                cash += sh * o[j]; sh = 0.0; holding = False; ntr += 1
            elif tp_on and hit:
                cash += sh * (c[j] if close_fill else max(o[j], tp)); sh = 0.0; holding = False; ntr += 1
            elif half and (not added):
                lim = first_px * (1 - add_pct)
                hit_add = (c[j] <= lim) if close_fill else (l[j] <= lim)
                if hit_add:
                    px = c[j] if close_fill else min(o[j], lim)
                    sh += cash / px; cash = 0.0; added = True; nadd += 1
                    avg = (first_px + px) / 2.0
                    tp = avg * (1 + tp_fix)
                    if tp_mode == 1:
                        tp = avg * 1.03
                        for q in range(srt.shape[1]):
                            v = srt[d, q]
                            if not np.isnan(v) and v * 0.998 >= avg * 1.005:
                                tp = min(v * 0.998, avg * 1.03)
                                break
        if (not holding) and can and d >= 0 and not np.isnan(lvl[d]):
            hit_in = (c[j] <= lvl[d]) if close_fill else (l[j] <= lvl[d])
            if hit_in:
                px = c[j] if close_fill else min(o[j], lvl[d])
                spend = cash * 0.5 if half else cash
                sh += spend / px; cash -= spend
                holding, added, first_px, avg, eday = True, False, px, px, d
                tp = avg * (1 + tp_fix)
                if tp_mode == 1:
                    tp = avg * 1.03
                    for q in range(srt.shape[1]):
                        v = srt[d, q]
                        if not np.isnan(v) and v * 0.998 >= avg * 1.005:
                            tp = min(v * 0.998, avg * 1.03)
                            break
                tp_on = is_awake[j]
        if is_end[j]:
            eq[k] = cash + sh * c[j]
            k += 1
    return eq, ntr, nadd


def run(win, close_fill):
    S, common, sel = ctx(win)
    rows = []
    SRT, LV, LVF = {}, {}, {}
    for s in SYMS:
        day_arrays(S, s)
        A = S[s]
        t = pd.to_datetime(A["start_ns"], utc=True).tz_convert("America/New_York")
        mins = np.asarray(t.hour * 60 + t.minute)
        A["is_pre4"] = (mins >= 240) & (mins < 570)
        idx, pc, trd, nd = day_info(S, s)
        lv = np.full(nd, np.nan)
        lv[idx] = np.where(trd["구간 매수·과매수"], pc, pc * 0.99)
        SRT[s] = sr_table(S, s, win)
        LV[s] = lv
        # 지지선 필터: 지정가 아래 2% 안에 지지 군집이 있어야 매수
        lvf = lv.copy()
        for d in range(nd):
            if np.isnan(lv[d]):
                continue
            row = SRT[s][d]
            near = row[(row <= lv[d]) & (row >= lv[d] * 0.98)]
            if len(near) == 0:
                lvf[d] = np.nan
        LVF[s] = lvf
    for add, (tpn, tpm, tpf), flt in itertools.product([0.0, 0.02, 0.03], [("+0.5%", 0, 0.005), ("+1%", 0, 0.01), ("새 저항선", 1, 0.005)],
                                                      ["지금 규칙", "지지선 2% 안"]):
        eqs, nt, na_ = [], 0, 0
        for s in SYMS:
            A = S[s]
            lv = LV[s] if flt == "지금 규칙" else LVF[s]
            e, t_, a_ = sim_sr(A["o"], A["h"], A["l"], A["c"], A["is_reg"], A["is_pre4"], A["is_awake"], A["is_end"],
                               A["first_reg"], A["day_idx"], A["j0"], lv, SRT[s], add, tpm, tpf, close_fill)
            eqs.append(e); nt += t_; na_ += a_
        m = window_metrics(portfolio(eqs, S), sel)
        rows.append({"win": win, "cf": close_fill, "추매": "없음" if add == 0 else f"-{add*100:g}%", "매도": tpn, "매수": flt,
                     "ret": m["all_ret"], "mdd": m["all_mdd"], "sh": m["all_sh"], "trades/day": nt / len(common), "adds": na_})
    return rows


if __name__ == "__main__":
    rows = []
    for w in ("long", "y2025", "short"):
        for cf in (False, True):
            rows += run(w, cf)
    R = pd.DataFrame(rows)
    R.to_parquet("run_sr2.parquet")
    F = lambda x: f"{x:+.0%}"
    P = R.pivot_table(index=["추매", "매도", "매수"], columns=["win", "cf"], values="ret")
    M = R.pivot_table(index=["추매", "매도", "매수"], columns=["win"], values="mdd", aggfunc="first")
    out = pd.DataFrame(index=P.index)
    for w, lab in (("long", "2026"), ("y2025", "2025"), ("short", "최근")):
        out[lab] = P[(w, False)].map(F); out[lab + " 종가"] = P[(w, True)].map(F)
    out["낙폭26"] = M["long"].map(F)
    rk = pd.concat([P[c].rank(ascending=False) for c in P.columns], axis=1).mean(axis=1)
    out["평균순위"] = rk.round(1)
    pd.set_option("display.width", 220)
    print(out.sort_values("평균순위").to_string())
