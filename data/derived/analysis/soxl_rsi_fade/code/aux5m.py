"""SOXL·SOXS 장초 RSI 70 반대 매수 — 보조 5분봉 확인(BRIEF.md '보조 5분봉' 절 그대로, 봉 단위 반복 구현).

규칙: 5분 RSI(14, 기준 R = 정규장 5분봉만 날을 넘어 이어 붙임, 8/31 첫 14봉 예열) T1 돌파(RSI[t−1] ≤ 70 < RSI[t]),
신호 봉 시작 09:30~10:25(12봉), 반대 종목을 다음 5분봉 시가 + $0.01에 매수, 진입 봉 포함 6봉(30분) 보유,
목표 P×1.01(고가 ≥ 목표 + $0.01이면 +1.00%), 미도달은 6번째 봉 종가 − $0.01 청산, 같은 날 같은 매수 종목 겹침 금지.
대조: 추종(같은 시각 X 자신 매수), B0(창 안 모든 5분봉 12개 × SOXL·SOXS, 겹침 허용).
기간: 2026-08-31~09-29, 2026-09-30~10-28. 날 묶음 부트스트랩 5,000회(seed 20261029, 기간마다 새로 시드).
결과: results/aux5m.json, results/aux5m_events.csv. 사용: py code/aux5m.py
"""
import json
import math
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
OUT_JSON = os.path.join(ROOT, "results", "aux5m.json")
OUT_CSV = os.path.join(ROOT, "results", "aux5m_events.csv")

TICKERS = ("SOXL", "SOXS")
OTHER = {"SOXL": "SOXS", "SOXS": "SOXL"}
RSI_N = 14
LEVEL = 70.0
COST = 0.01
TGT = 0.01
STEP = 5                 # 분
NSLOT = 78               # 정규장 5분봉 09:30~15:55
WIN_SLOTS = range(0, 12)  # 09:30~10:25 봉 시작
HOLD = 6                 # 진입 봉 포함 6봉
OPEN_MIN = 9 * 60 + 30
PERIODS = [("2026-08-31~09-29", "2026-08-31", "2026-09-29"),
           ("2026-09-30~10-28", "2026-09-30", "2026-10-28")]
SEED = 20261029
NBOOT = 5000


# ------------------------------------------------------------ 자료

def load_regular(sym):
    d = pd.read_parquet(os.path.join(ROOT, "cache", "5m", f"{sym}.parquet"))
    d = d[~d.index.duplicated(keep="first")].sort_index()
    d.index = d.index.tz_convert("America/New_York")
    mod = d.index.hour * 60 + d.index.minute
    assert ((mod % STEP) == 0).all(), f"{sym} 5분 격자 밖 봉"
    reg = d[(mod >= OPEN_MIN) & (mod <= 15 * 60 + 55)][["Open", "High", "Low", "Close"]].astype(float)
    assert not reg.isna().any().any(), f"{sym} 정규장 값 비어 있음"
    return reg


def wilder_rsi(closes, n=RSI_N):
    """처음 n개 변화의 단순평균으로 시작, 이후 (평균×(n−1) + 새 값)/n. 손실 평균 0이면 100. 앞 n봉은 NaN."""
    out = [math.nan] * len(closes)
    if len(closes) <= n:
        return out
    ag = al = 0.0
    for i in range(1, n + 1):
        ch = closes[i] - closes[i - 1]
        ag += max(ch, 0.0)
        al += max(-ch, 0.0)
    ag /= n
    al /= n
    out[n] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    for i in range(n + 1, len(closes)):
        ch = closes[i] - closes[i - 1]
        ag = (ag * (n - 1) + max(ch, 0.0)) / n
        al = (al * (n - 1) + max(-ch, 0.0)) / n
        out[i] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out


def build(sym):
    """반환: days(list[date]), bars{day: dict(O,H,L,C 각 길이 78 list)}, rsi{day: list 78}, prev{day: list 78}."""
    reg = load_regular(sym)
    rsi = wilder_rsi(reg["Close"].tolist())
    prev = [math.nan] + rsi[:-1]          # 같은 계열 바로 앞 봉(09:30 봉의 앞 봉 = 전 거래일 15:55 봉)
    days = sorted(set(reg.index.date))
    bars, rs, pv = {}, {}, {}
    slot = ((reg.index.hour * 60 + reg.index.minute - OPEN_MIN) // STEP).to_numpy()
    dates = np.array(reg.index.date)
    O, H, L, C = (reg[c].to_numpy() for c in ("Open", "High", "Low", "Close"))
    for day in days:
        ii = np.flatnonzero(dates == day)
        assert len(ii) == NSLOT and (slot[ii] == np.arange(NSLOT)).all(), f"{sym} {day} 정규장 5분봉 {len(ii)}개"
        bars[day] = {"O": O[ii].tolist(), "H": H[ii].tolist(), "L": L[ii].tolist(), "C": C[ii].tolist()}
        rs[day] = [rsi[i] for i in ii]
        pv[day] = [prev[i] for i in ii]
    return days, bars, rs, pv


# ------------------------------------------------------------ 한 매매

def simulate(b, e):
    """b = 그날 Y 봉, e = 진입 봉 번호. 진입 봉 포함 HOLD봉 보유. 봉이 모자라면 None."""
    if e + HOLD - 1 >= NSLOT:
        return None
    P = b["O"][e] + COST
    up_thr = P * (1 + TGT) + COST
    dn_thr = P * (1 - TGT) - COST
    hit_j = None
    race = "none"
    for j in range(HOLD):
        up = b["H"][e + j] >= up_thr
        dn = b["L"][e + j] <= dn_thr
        if race == "none":
            if up and dn:
                race = "ambig"
            elif up:
                race = "win"
            elif dn:
                race = "loss"
        if up and hit_j is None:
            hit_j = j
    exit_j = hit_j if hit_j is not None else HOLD - 1
    low = min(b["L"][e + j] for j in range(exit_j + 1))
    pnl = TGT * 100.0 if hit_j is not None else ((b["C"][e + HOLD - 1] - COST) / P - 1.0) * 100.0
    return {"P": P, "hit": hit_j is not None, "exit_slot": e + exit_j, "pnl": pnl,
            "mae": (low / P - 1.0) * 100.0, "race": race}


def hhmm(slot):
    m = OPEN_MIN + STEP * int(slot)
    return f"{m // 60:02d}:{m % 60:02d}"


# ------------------------------------------------------------ 실행

def main():
    data = {t: build(t) for t in TICKERS}
    assert data["SOXL"][0] == data["SOXS"][0], "SOXL·SOXS 거래일 불일치"
    days = data["SOXL"][0]
    bars = {t: data[t][1] for t in TICKERS}

    # 신호(T1) — 창 안 봉, X 자신의 RSI
    signals = []   # (day, slot, X)
    warm_first_rsi = {}
    for X in TICKERS:
        _, _, rs, pv = data[X]
        first = next((d, s) for d in days for s in range(NSLOT) if not math.isnan(rs[d][s]))
        warm_first_rsi[X] = f"{first[0].isoformat()} {hhmm(first[1])}"
        for d in days:
            for s in WIN_SLOTS:
                p, r = pv[d][s], rs[d][s]
                if not (math.isnan(p) or math.isnan(r)) and p <= LEVEL < r:
                    signals.append((d, s, X))

    def run(mode):
        orders = []
        for d, s, X in signals:
            Y = OTHER[X] if mode == "fade" else X
            orders.append((d, Y, s + 1, X, s))
        orders.sort()
        trades, skip_ov, skip_nb = [], 0, 0
        last_exit = {}
        for d, Y, e, X, s in orders:
            res = simulate(bars[Y][d], e)
            if res is None:
                skip_nb += 1
                continue
            ex = last_exit.get((d, Y))
            # 새 진입 시각 < 앞 청산 봉 시작 + 1분 → 건너뜀(5분 격자에서는 진입 봉 ≤ 청산 봉)
            if ex is not None and OPEN_MIN + STEP * e < OPEN_MIN + STEP * ex + 1:
                skip_ov += 1
                continue
            last_exit[(d, Y)] = res["exit_slot"]
            trades.append(dict(day=d, X=X, s=s, Y=Y, e=e, **res))
        return trades, skip_ov, skip_nb

    fade, f_ov, f_nb = run("fade")
    foll, l_ov, l_nb = run("follow")
    b0 = []
    for d in days:
        for Y in TICKERS:
            for s in WIN_SLOTS:
                res = simulate(bars[Y][d], s + 1)
                assert res is not None
                b0.append(dict(day=d, Y=Y, s=s, e=s + 1, **res))

    def stats(tr, ov=None, nb=None):
        n = len(tr)
        out = {"n": n, "n_days": len({t["day"] for t in tr})}
        if ov is not None:
            out.update(skipped_overlap=ov, skipped_nobar=nb)
        if n == 0:
            out.update(hit_rate=None, ev_mean_pct=None, ev_median_pct=None, miss_mean_pct=None,
                       mae_median_pct=None, race_win=None, race_loss=None, race_ambig_n=0, race_none=None)
            return out
        pn = np.array([t["pnl"] for t in tr])
        hit = np.array([t["hit"] for t in tr], dtype=bool)
        rc = pd.Series([t["race"] for t in tr]).value_counts()
        r6 = lambda x: None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 6)
        out.update(
            hit_rate=r6(hit.mean()), ev_mean_pct=r6(pn.mean()), ev_median_pct=r6(np.median(pn)),
            miss_mean_pct=r6(pn[~hit].mean()) if (~hit).any() else None,
            mae_median_pct=r6(np.median([t["mae"] for t in tr])),
            race_win=r6(rc.get("win", 0) / n), race_loss=r6((rc.get("loss", 0) + rc.get("ambig", 0)) / n),
            race_ambig_n=int(rc.get("ambig", 0)), race_none=r6(rc.get("none", 0) / n))
        return out

    def per_day(tr, pdays):
        ix = {d: i for i, d in enumerate(pdays)}
        cnt, s, h = np.zeros(len(pdays)), np.zeros(len(pdays)), np.zeros(len(pdays))
        for t in tr:
            i = ix[t["day"]]
            cnt[i] += 1
            s[i] += t["pnl"]
            h[i] += t["hit"]
        return cnt, s, h

    result = {"rule": "5분 RSI(14, Wilder, 기준 R: 정규장 5분봉 09:30~15:55만 8/31부터 날을 넘어 이음, 8/31 첫 14봉 예열) "
                      "T1(RSI[t−1] ≤ 70 < RSI[t]), 신호 봉 시작 09:30~10:25(12봉), 반대 종목 다음 5분봉 시가 + $0.01 진입, "
                      "진입 봉 포함 6봉 보유, 목표 P×1.01(고가 ≥ 목표 + $0.01 → +1.00%), 미도달 6번째 봉 종가 − $0.01, "
                      "같은 날 같은 매수 종목 겹침 금지(진입 시각 < 앞 청산 봉 시작 + 1분이면 건너뜀). "
                      "추종 = 같은 시각 X 자신 매수. B0 = 창 안 12봉 × SOXL·SOXS, 같은 체결·보유, 겹침 허용.",
              "rsi_first_value_bar": warm_first_rsi,
              "bootstrap": {"n_boot": NBOOT, "seed": SEED,
                            "method": "기간마다 np.random.default_rng(seed).integers(0, 날 수, (5000, 날 수))로 날 복원 추출, "
                                      "뽑힌 날의 매매를 통째로 합쳐 평균(합÷건수). 반대 매매가 0건인 추출은 NaN으로 빼고 개수 보고."},
              "periods": {}}

    events = []
    for name, a, z in PERIODS:
        a, z = pd.Timestamp(a).date(), pd.Timestamp(z).date()
        pdays = [d for d in days if a <= d <= z]
        inp = lambda tr: [t for t in tr if a <= t["day"] <= z]
        fp, lp, bp = inp(fade), inp(foll), inp(b0)
        # 겹침·봉 없음 건너뜀도 기간별로 다시 셈
        sig_p = [x for x in signals if a <= x[0] <= z]
        n_sig = len(sig_p)
        f_ov_p = n_sig - len(fp)        # 봉 없음 건너뜀은 아래 단언으로 0 확인
        l_ov_p = n_sig - len(lp)
        assert f_nb == 0 and l_nb == 0, "봉 없음 건너뜀 발생"
        fs, ls, bs = stats(fp, f_ov_p, 0), stats(lp, l_ov_p, 0), stats(bp)
        bs = {k: bs[k] for k in ("n", "n_days", "hit_rate", "ev_mean_pct", "ev_median_pct", "race_win", "race_loss")}

        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(pdays), size=(NBOOT, len(pdays)))
        agg = {}
        for nm, tr in (("fade", fp), ("follow", lp), ("b0", bp)):
            c, s, h = per_day(tr, pdays)
            N = c[idx].sum(1)
            with np.errstate(divide="ignore", invalid="ignore"):
                agg[nm] = {"ev": s[idx].sum(1) / N, "hit": h[idx].sum(1) / N}
        draws = {
            "fade_ev_mean_pct": agg["fade"]["ev"],
            "fade_minus_b0_ev_pct": agg["fade"]["ev"] - agg["b0"]["ev"],
            "fade_hit_rate": agg["fade"]["hit"],
            "fade_minus_b0_hit": agg["fade"]["hit"] - agg["b0"]["hit"],
            "follow_ev_mean_pct": agg["follow"]["ev"],
            "fade_minus_follow_ev_pct": agg["fade"]["ev"] - agg["follow"]["ev"],
            "b0_ev_mean_pct": agg["b0"]["ev"],
        }
        ci = {}
        for k, v in draws.items():
            lo, hi = np.nanpercentile(v, [2.5, 97.5])
            ci[k] = {"lo": round(float(lo), 6), "hi": round(float(hi), 6), "nan_draws": int(np.isnan(v).sum())}

        diff = lambda x, y: None if x is None or y is None else round(x - y, 6)
        result["periods"][name] = {
            "n_days": len(pdays), "first_day": pdays[0].isoformat(), "last_day": pdays[-1].isoformat(),
            "n_signals": n_sig,
            "n_signals_by_ticker": {t: sum(1 for x in sig_p if x[2] == t) for t in TICKERS},
            "fade": fs, "follow": ls, "b0": bs,
            "point_diff": {"fade_minus_b0_ev_pct": diff(fs["ev_mean_pct"], bs["ev_mean_pct"]),
                           "fade_minus_b0_hit": diff(fs["hit_rate"], bs["hit_rate"]),
                           "fade_minus_follow_ev_pct": diff(fs["ev_mean_pct"], ls["ev_mean_pct"]),
                           "fade_minus_follow_hit": diff(fs["hit_rate"], ls["hit_rate"])},
            "ci95": ci,
        }
        for mode, tr in (("fade", fp), ("follow", lp)):
            for t in tr:
                events.append({"period": name, "mode": mode, "day": t["day"].isoformat(), "signal_ticker": t["X"],
                               "signal_time": hhmm(t["s"]), "buy": t["Y"], "entry_time": hhmm(t["e"]),
                               "entry_px": round(t["P"], 4), "hit": int(t["hit"]),
                               "exit_time": hhmm(t["exit_slot"]), "pnl_pct": round(t["pnl"], 6),
                               "mae_pct": round(t["mae"], 6), "race": t["race"]})

    assert sum(p["n_signals"] for p in result["periods"].values()) == len(signals)
    result["events"] = "results/aux5m_events.csv"
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    pd.DataFrame(events).to_csv(OUT_CSV, index=False, encoding="utf-8")
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)

    print("RSI 첫 값 봉", warm_first_rsi)
    for name, p in result["periods"].items():
        print(f"== {name} ({p['n_days']}일, 신호 {p['n_signals']} {p['n_signals_by_ticker']})")
        for k in ("fade", "follow", "b0"):
            s = p[k]
            print(f"  {k:6s} n={s['n']:4d} 도달률={s['hit_rate']} 평균손익={s['ev_mean_pct']} 중앙값={s['ev_median_pct']}"
                  + (f" 겹침건너뜀={s['skipped_overlap']}" if "skipped_overlap" in s else ""))
        print("  차이", p["point_diff"])
        print("  CI", json.dumps(p["ci95"], ensure_ascii=False))


if __name__ == "__main__":
    main()
