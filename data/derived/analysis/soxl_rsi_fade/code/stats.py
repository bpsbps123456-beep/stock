"""SOXL·SOXS 장초 1분봉 RSI 70 반대 매수 — 판정·부트스트랩·분할·날 성격·32칸 표·손익 분포 (BRIEF.md 그대로).

매매 계산은 code/impl_a.py 함수를 그대로 다시 쓰고(정본 results/impl_a.json·events_a.csv와 일치 확인),
여기서는 BRIEF '판정'·'부트스트랩·분할'·'날 성격' 절의 통계만 잰다.
결과: results/stats.json. 사용: py code/stats.py

해석 선택(BRIEF 문구에 가장 가까운 쪽):
- 날 묶음 부트스트랩: 대상 날들을 복원 추출(크기 = 그 날 수), 뽑힌 날의 반대·추종·B0 매매를 통째로 모아
  평균(매매 단위 합 ÷ 건수)을 낸다. 반대·추종·B0는 같은 추출 날을 쓴다(짝 비교). 매 호출 새 rng(seed 20261029).
  21일 전체는 impl_a와 같은 rng.integers(0, 21, (10000, 21))라 같은 구간이 나온다.
- 앞·뒤 반 구간은 그 반의 날만(10일·11일) 복원 추출한다. 판정 3은 BRIEF대로 점추정 > 0만 본다.
- 판정에서 1·2 통과·3 실패는 BRIEF에 이름이 없어 '미정의(후보 아님)'로, 1·3 통과·2 실패는 '1만 넘음'과 같은
  뜻(신호 덕 아님)으로 보아 '변동성 장세 덕'으로 둔다.
- 분위는 numpy 선형 보간(np.percentile 기본).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import impl_a as A  # noqa: E402

OUT_JSON = os.path.join(A.OUT_DIR, "stats.json")
SPLIT = A.HALF_SPLIT


def _f(x):
    return A._f(x)


# ---------------------------------------------------------------- 매매 다시 만들기(impl_a 함수 그대로)

def build():
    data = {t: A.load(t) for t in A.TICKERS}
    dsets = [sorted(set(data[t].index[(A.minute_of_day(data[t].index) >= A.OPEN_MIN)
                                      & (A.minute_of_day(data[t].index) <= 959)].date)) for t in A.TICKERS]
    assert dsets[0] == dsets[1], "SOXL·SOXS 거래일 불일치"
    days = dsets[0]
    ndays = len(days)
    grids = {t: A.regular_grid(data[t], days) for t in A.TICKERS}
    for t in A.TICKERS:
        assert not np.isnan(grids[t]["C"][:, :91]).any(), f"{t} 09:30~11:00 봉 누락"
    tables = {t: A.trade_table(grids[t]) for t in A.TICKERS}
    rsi = {(b, t): A.rsi_series(data[t], b) for b in A.BASES for t in A.TICKERS}
    sigs = {(b, w): {t: A.signals_for(rsi[(b, t)], days, nw) for t in A.TICKERS}
            for b in A.BASES for w, nw in A.WINDOWS.items()}
    b0s = {(e, w): A.b0_trades(tables, k, nw, ndays) for e, k in A.FILLS.items() for w, nw in A.WINDOWS.items()}
    cells = {}
    for T in A.SIGNALS:
        for b in A.BASES:
            for e, k in A.FILLS.items():
                for w in A.WINDOWS:
                    key = f"{T}|{b}|{e}|{w}"
                    sbx = {t: sigs[(b, w)][t][T] for t in A.TICKERS}
                    fade, fo, fn = A.run_trades(sbx, tables, grids, k, "fade")
                    foll, lo, ln = A.run_trades(sbx, tables, grids, k, "follow")
                    cells[key] = {"fade": fade, "follow": foll, "b0": b0s[(e, w)],
                                  "skip": (fo, fn, lo, ln)}
    return days, grids, cells


def check_against_impl_a(cells, days):
    """정본 impl_a.json 칸 수치·events_a.csv 와 같은지 확인."""
    with open(A.OUT_JSON, encoding="utf-8") as fh:
        ref = json.load(fh)["cells"]
    bad = []
    for key, c in cells.items():
        fo, fn, lo, ln = c["skip"]
        mine = {"fade": A.trade_stats(c["fade"], fo, fn), "follow": A.trade_stats(c["follow"], lo, ln),
                "b0": A.b0_stats(c["b0"])}
        for side in mine:
            for fld, v in mine[side].items():
                r = ref[key][side].get(fld)
                if (v is None) != (r is None) or (v is not None and abs(float(v) - float(r)) > 1e-6):
                    bad.append((key, side, fld, v, r))
    ev = pd.read_csv(os.path.join(A.OUT_DIR, "events_a.csv"))
    fade = cells[A.MAIN]["fade"]
    mine = sorted(zip([days[d].isoformat() for d in fade["d"]], [A.hhmm(s) for s in fade["s"]],
                      fade["X"], np.round(fade["pnl"].to_numpy(), 6)))
    theirs = sorted(zip(ev["day"], ev["signal_time"], ev["signal_ticker"], ev["pnl_pct"].round(6)))
    csv_ok = len(mine) == len(theirs) and all(a[:3] == b[:3] and abs(a[3] - b[3]) < 1e-6
                                              for a, b in zip(mine, theirs))
    return {"cell_fields_mismatch": len(bad), "examples": [list(map(str, x)) for x in bad[:5]],
            "events_csv_match": bool(csv_ok), "events_rows": len(theirs)}


# ---------------------------------------------------------------- 날 묶음 부트스트랩

def boot(fade, follow, b0, day_idx, ndays):
    """day_idx(대상 날 번호들)를 복원 추출 NBOOT회. 점추정 + 2.5·97.5% 분위."""
    day_idx = np.asarray(day_idx, dtype=int)
    m = len(day_idx)
    rng = np.random.default_rng(A.SEED)
    draw = day_idx[rng.integers(0, m, size=(A.NBOOT, m))]
    agg, point = {}, {}
    for name, df in [("fade", fade), ("follow", follow), ("b0", b0)]:
        n, s, h = A.per_day(df, ndays)
        N = n[draw].sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            agg[name] = {"ev": s[draw].sum(1) / N, "hit": h[draw].sum(1) / N}
        n0 = n[day_idx].sum()
        point[name] = {"n": int(n0),
                       "ev": s[day_idx].sum() / n0 if n0 else np.nan,
                       "hit": h[day_idx].sum() / n0 if n0 else np.nan}
    defs = {
        "fade_ev_mean_pct": (agg["fade"]["ev"], point["fade"]["ev"]),
        "fade_hit_rate": (agg["fade"]["hit"], point["fade"]["hit"]),
        "b0_ev_mean_pct": (agg["b0"]["ev"], point["b0"]["ev"]),
        "b0_hit_rate": (agg["b0"]["hit"], point["b0"]["hit"]),
        "follow_ev_mean_pct": (agg["follow"]["ev"], point["follow"]["ev"]),
        "follow_hit_rate": (agg["follow"]["hit"], point["follow"]["hit"]),
        "fade_minus_b0_hit": (agg["fade"]["hit"] - agg["b0"]["hit"], point["fade"]["hit"] - point["b0"]["hit"]),
        "fade_minus_b0_ev_pct": (agg["fade"]["ev"] - agg["b0"]["ev"], point["fade"]["ev"] - point["b0"]["ev"]),
        "fade_minus_follow_ev_pct": (agg["fade"]["ev"] - agg["follow"]["ev"],
                                     point["fade"]["ev"] - point["follow"]["ev"]),
    }
    out = {"n_days": m, "fade_n": point["fade"]["n"], "follow_n": point["follow"]["n"], "b0_n": point["b0"]["n"]}
    for k, (v, p) in defs.items():
        ok = ~np.isnan(v)
        lo, hi = (np.percentile(v[ok], [2.5, 97.5]) if ok.any() else (np.nan, np.nan))
        out[k] = {"point": _f(p), "lo": _f(lo), "hi": _f(hi), "nan_draws": int((~ok).sum())}
    return out


# ---------------------------------------------------------------- 본 실행

def main():
    days, grids, cells = build()
    ndays = len(days)
    chk = check_against_impl_a(cells, days)
    assert chk["cell_fields_mismatch"] == 0 and chk["events_csv_match"], chk
    all_days = np.arange(ndays)
    first = np.array([d <= SPLIT for d in days])
    half_days = {"first_9/30-10/13": np.flatnonzero(first), "second_10/14-10/28": np.flatnonzero(~first)}

    # ---- 주 칸
    M = cells[A.MAIN]
    fade, foll, b0 = M["fade"], M["follow"], M["b0"]
    primary = boot(fade, foll, b0, all_days, ndays)
    halves = {name: boot(fade, foll, b0, idx, ndays) for name, idx in half_days.items()}
    for name, idx in half_days.items():
        halves[name]["days"] = [days[i].isoformat() for i in idx]

    c1 = primary["fade_ev_mean_pct"]["point"] > 0 and primary["fade_ev_mean_pct"]["lo"] > 0
    c2 = primary["fade_minus_b0_hit"]["point"] > 0 and primary["fade_minus_b0_hit"]["lo"] > 0
    c3 = all((h["fade_ev_mean_pct"]["point"] or -1) > 0 for h in halves.values())
    if not c1:
        verdict = "불가"
    elif c2 and c3:
        verdict = "후보"
    elif not c2:
        verdict = "변동성 장세 덕"
    else:
        verdict = "1·2 통과, 3 실패 — BRIEF 미정의(후보 아님)"

    # ---- 날 성격(설명용)
    gS = grids["SOXL"]
    rng_ = np.nanmax(gS["H"][:, :90], 1) - np.nanmin(gS["L"][:, :90], 1)
    eff = (gS["O"][:, 90] - gS["O"][:, 0]) / rng_
    label = np.where(np.abs(eff) >= 0.6, "한 방향", "오락가락")
    by_type = {}
    for lab in ["한 방향", "오락가락"]:
        dset = set(np.flatnonzero(label == lab))
        f = fade[fade["d"].isin(dset)]
        fl = foll[foll["d"].isin(dset)]
        bb = b0[b0["d"].isin(dset)]
        by_type[lab] = {"n_days": len(dset), "days_with_fade": int(f["d"].nunique()),
                        "fade_n": len(f), "fade_hits": int(f["hit"].sum()),
                        "fade_hit_rate": _f(f["hit"].mean()) if len(f) else None,
                        "fade_ev_mean_pct": _f(f["pnl"].mean()) if len(f) else None,
                        "fade_pnl_sum_pct": _f(f["pnl"].sum()),
                        "follow_n": len(fl), "follow_ev_mean_pct": _f(fl["pnl"].mean()) if len(fl) else None,
                        "b0_hit_rate": _f(bb["hit"].mean()), "b0_ev_mean_pct": _f(bb["pnl"].mean())}
    day_rows = []
    for i in range(ndays):
        f = fade[fade["d"] == i]
        fl = foll[foll["d"] == i]
        bb = b0[b0["d"] == i]
        day_rows.append({
            "day": days[i].isoformat(), "half": "앞" if first[i] else "뒤",
            "eff": _f(eff[i]), "type": str(label[i]),
            "soxl_0930_open": _f(gS["O"][i, 0]), "soxl_1100_open": _f(gS["O"][i, 90]),
            "fade_n": len(f), "fade_hits": int(f["hit"].sum()),
            "fade_pnl_sum_pct": _f(f["pnl"].sum()),
            "fade_pnl_mean_pct": _f(f["pnl"].mean()) if len(f) else None,
            "fade_trades": [f"{A.hhmm(s)} {x}→{y} {p:+.3f}" for s, x, y, p in zip(f["s"], f["X"], f["Y"], f["pnl"])],
            "follow_n": len(fl), "follow_pnl_mean_pct": _f(fl["pnl"].mean()) if len(fl) else None,
            "b0_hit_rate": _f(bb["hit"].mean()), "b0_ev_mean_pct": _f(bb["pnl"].mean()),
        })

    # ---- 32칸 표
    grid_rows = {}
    for key, c in cells.items():
        fa, fl, bb = c["fade"], c["follow"], c["b0"]
        bt = boot(fa, fl, bb, all_days, ndays)
        g_c1 = (bt["fade_ev_mean_pct"]["point"] or -1) > 0 and (bt["fade_ev_mean_pct"]["lo"] or -1) > 0
        g_c2 = (bt["fade_minus_b0_hit"]["point"] or -1) > 0 and (bt["fade_minus_b0_hit"]["lo"] or -1) > 0
        hv = {name: _f(fa.loc[fa["d"].isin(set(idx)), "pnl"].mean()) if fa["d"].isin(set(idx)).any() else None
              for name, idx in half_days.items()}
        g_c3 = all((v or -1) > 0 for v in hv.values())
        grid_rows[key] = {
            "fade_n": len(fa), "fade_hit_rate": _f(fa["hit"].mean()) if len(fa) else None,
            "fade_ev_mean_pct": _f(fa["pnl"].mean()) if len(fa) else None,
            "fade_ev_ci95": [bt["fade_ev_mean_pct"]["lo"], bt["fade_ev_mean_pct"]["hi"]],
            "b0_hit_rate": _f(bb["hit"].mean()), "b0_ev_mean_pct": _f(bb["pnl"].mean()),
            "follow_n": len(fl), "follow_ev_mean_pct": _f(fl["pnl"].mean()) if len(fl) else None,
            "fade_minus_b0_hit": bt["fade_minus_b0_hit"]["point"],
            "fade_minus_b0_hit_ci95": [bt["fade_minus_b0_hit"]["lo"], bt["fade_minus_b0_hit"]["hi"]],
            "half_fade_ev_mean_pct": hv,
            "c1": bool(g_c1), "c2": bool(g_c2), "c3": bool(g_c3),
        }
    evs = {k: v["fade_ev_mean_pct"] for k, v in grid_rows.items() if v["fade_ev_mean_pct"] is not None}
    kmin, kmax = min(evs, key=evs.get), max(evs, key=evs.get)
    grid_summary = {
        "n_cells": len(grid_rows),
        "fade_ev_min": {"cell": kmin, "ev_mean_pct": evs[kmin], "n": grid_rows[kmin]["fade_n"]},
        "fade_ev_max": {"cell": kmax, "ev_mean_pct": evs[kmax], "n": grid_rows[kmax]["fade_n"]},
        "cells_fade_ev_point_pos": sum(1 for v in evs.values() if v > 0),
        "cells_c1_pass": sum(r["c1"] for r in grid_rows.values()),
        "cells_c1_pass_list": [k for k, r in grid_rows.items() if r["c1"]],
        "cells_c2_pass": sum(r["c2"] for r in grid_rows.values()),
        "cells_c3_pass": sum(r["c3"] for r in grid_rows.values()),
        "cells_all3_pass": [k for k, r in grid_rows.items() if r["c1"] and r["c2"] and r["c3"]],
        "fade_n_range": [min(r["fade_n"] for r in grid_rows.values()), max(r["fade_n"] for r in grid_rows.values())],
        "note": "주 칸 1개로 판정, 나머지 31칸은 범위만. 32칸 안에서 가장 좋은 칸을 고르면 선택편향(같은 21일·겹치는 신호).",
        "table": grid_rows,
    }

    # ---- 반대 매수 손익 분포(주 칸)
    def profile(df, with_rows=True):
        hit = df["hit"].astype(bool).to_numpy()
        pnl = df["pnl"].to_numpy(dtype=float)
        miss = pnl[~hit]
        n, nh = len(pnl), int(hit.sum())
        out = {"n": n, "hits": nh, "misses": n - nh, "hit_rate": _f(nh / n),
               "hit_pnl_all_1pct": bool(np.allclose(pnl[hit], 1.0)),
               "miss_mean_pct": _f(miss.mean()) if len(miss) else None,
               "miss_q10_q50_q90_pct": [_f(x) for x in np.percentile(miss, [10, 50, 90])] if len(miss) else None,
               "miss_min_pct": _f(miss.min()) if len(miss) else None,
               "miss_max_pct": _f(miss.max()) if len(miss) else None,
               "miss_le_minus1_n": int((miss <= -1.0).sum()),
               "miss_positive_n": int((miss > 0).sum()),
               "gain_from_hits_sum_pct": _f(pnl[hit].sum()), "loss_from_misses_sum_pct": _f(miss.sum()),
               "net_sum_pct": _f(pnl.sum()), "ev_mean_pct": _f(pnl.mean())}
        if len(miss):
            mm = miss.mean()
            out["breakeven_hit_rate"] = _f(-mm / (1.0 - mm))        # p·1 + (1−p)·miss_mean = 0
            out["avg_win_over_avg_miss_abs"] = _f(1.0 / abs(mm)) if mm < 0 else None
        return out

    fp = profile(fade)
    order = np.argsort(fade["pnl"].to_numpy())
    worst = []
    for j in order[:3]:
        r = fade.iloc[j]
        worst.append({"day": days[int(r["d"])].isoformat(), "signal": f"{A.hhmm(r['s'])} {r['X']} RSI>70",
                      "buy": r["Y"], "entry_time": A.hhmm(r["e"]),
                      "exit_time": A.hhmm(r["e"] + r["exit_off"]), "pnl_pct": _f(r["pnl"]), "mae_pct": _f(r["mae"])})
    fp["worst3"] = worst
    w3 = sum(w["pnl_pct"] for w in worst)
    fp["worst3_sum_pct"] = _f(w3)
    fp["worst3_sum_over_hit_gain"] = _f(abs(w3) / fp["gain_from_hits_sum_pct"])
    fp["mae_q10_q50_q90_pct"] = [_f(x) for x in np.percentile(fade["mae"], [10, 50, 90])]
    fp["hits_mae_le_minus1_n"] = int(((fade["mae"] <= -1.0) & fade["hit"].astype(bool)).sum())
    # 설명용(사후 분할, 판정에 안 씀): 09:30 첫 봉 신호(기준 R은 전날 15:59 봉과 이어 붙여 갭이 RSI를 밀어 올림) vs 그 뒤
    is_open = fade["s"] == 0
    fp["open_bar_split_descriptive"] = {
        part: {"n": int(m.sum()), "hits": int(fade.loc[m, "hit"].sum()),
               "ev_mean_pct": _f(fade.loc[m, "pnl"].mean()) if m.any() else None,
               "pnl_sum_pct": _f(fade.loc[m, "pnl"].sum())}
        for part, m in [("signal_0930", is_open), ("signal_0931_1029", ~is_open)]}
    loss_profile = {"fade": fp, "follow": profile(foll), "b0": profile(b0)}
    fade_pnl = fade["pnl"].to_numpy(dtype=float)
    loss_profile["reading"] = (
        f"반대 매수 {fp['n']}건 중 {fp['hits']}건 +1%(합 {fp['gain_from_hits_sum_pct']:.2f}%), "
        f"미도달 {fp['misses']}건 평균 {fp['miss_mean_pct']:.3f}%(합 {fp['loss_from_misses_sum_pct']:.2f}%), "
        f"미도달 중 −1% 이하 {fp['miss_le_minus1_n']}건, 최악 3건 합 {w3:.2f}% = 도달 이익의 "
        f"{fp['worst3_sum_over_hit_gain']*100:.0f}%. 손익분기 도달률 {fp['breakeven_hit_rate']:.3f} vs 실제 {fp['hit_rate']:.3f}. "
        f"최악 3건을 빼면 평균 {np.sort(fade_pnl)[3:].mean():.3f}%."
    )

    out = {
        "main_cell": A.MAIN,
        "source": {"trades": "code/impl_a.py 함수로 다시 계산", "check_vs_impl_a": chk,
                   "days": [d.isoformat() for d in days], "n_boot": A.NBOOT, "seed": A.SEED,
                   "bootstrap": "대상 날 복원 추출(크기 = 날 수), 반대·추종·B0 같은 추출, 매매 단위 합 ÷ 건수, 2.5·97.5% 분위(np.percentile 선형)"},
        "criteria": {
            "c1_fade_ev_pos_and_ci_lo_pos": bool(c1),
            "c1_detail": primary["fade_ev_mean_pct"],
            "c2_fade_minus_b0_hit_pos_and_ci_lo_pos": bool(c2),
            "c2_detail": primary["fade_minus_b0_hit"],
            "c3_both_halves_ev_pos": bool(c3),
            "c3_detail": {k: v["fade_ev_mean_pct"]["point"] for k, v in halves.items()},
        },
        "verdict": verdict,
        "primary_ci": primary,
        "halves": halves,
        "day_types": {"rule": "SOXL (11:00 시가 − 09:30 시가) ÷ (09:30~10:59 고저 폭), |효율| ≥ 0.6 → 한 방향, 아니면 오락가락. 같은 날 정보라 설명용",
                      "by_type": by_type, "days": day_rows},
        "grid_summary": grid_summary,
        "loss_profile": loss_profile,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)

    # ---- 화면 요약
    print("정본 대조", chk)
    print("판정", verdict, "c1", c1, "c2", c2, "c3", c3)
    for k, v in primary.items():
        print("  주", k, v)
    for name, h in halves.items():
        print(name, {k: h[k] for k in ["n_days", "fade_n"]})
        for k in ["fade_ev_mean_pct", "fade_hit_rate", "fade_minus_b0_hit", "fade_minus_b0_ev_pct", "fade_minus_follow_ev_pct"]:
            print("   ", k, h[k])
    print("날 성격", json.dumps(by_type, ensure_ascii=False))
    for r in day_rows:
        print(f"  {r['day']} {r['eff']:+.3f} {r['type']:4s} n={r['fade_n']} hit={r['fade_hits']} sum={r['fade_pnl_sum_pct']:+.3f}")
    print(f"{'cell':16s} {'n':>3s} {'hit':>6s} {'ev':>7s} {'ci_lo':>7s} {'ci_hi':>7s} {'b0hit':>6s} {'follEV':>7s} c1 c2 c3")
    fm = lambda x: "    n/a" if x is None else f"{x:7.3f}"
    for k, r in grid_rows.items():
        print(f"{k:16s} {r['fade_n']:3d} {fm(r['fade_hit_rate'])} {fm(r['fade_ev_mean_pct'])} {fm(r['fade_ev_ci95'][0])} "
              f"{fm(r['fade_ev_ci95'][1])} {fm(r['b0_hit_rate'])} {fm(r['follow_ev_mean_pct'])} "
              f"{int(r['c1'])}  {int(r['c2'])}  {int(r['c3'])}")
    print("32칸", {k: v for k, v in grid_summary.items() if k != "table"})
    print("손익 분포", json.dumps({k: v for k, v in loss_profile.items()}, ensure_ascii=False, indent=0))


if __name__ == "__main__":
    main()
