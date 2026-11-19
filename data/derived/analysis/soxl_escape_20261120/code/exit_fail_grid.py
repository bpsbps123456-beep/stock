"""갈래 exit_fail — '올라가는 척하다 실패하는' 랠리를 잡는 탈출 (2026-11-20).

후보군(시간봉 1·3분 각각):
  A2 reject : A와 같은 이탈·되돌림에서 '선에 닿은 음봉(종가<선, 종가<시가)' 완성 즉시 매도
  A retest  : 선(EMA10/25/60·VWAP) 하향 이탈 뒤 W봉 안에 고가가 선에 닿고(종가는 선 아래) 그 '되돌림 봉' 저가를 종가가 깨면 매도
  B dtop    : 확정 피벗 고점 두 개(간격 ≤40봉, 높이 차 ≤tol) → 그 사이 최저가(넥라인)를 종가가 20봉 안에 깨면 매도
  C nhfail  : N봉 신고가 갱신 → 갱신 시점의 직전 확정 피벗 저점(R=3)을 W봉 안에 종가가 깨면 매도
  D rsidiv  : 확정 피벗 고점 p1<p2, 고가 p2>p1·RSI14 p2<p1(하락 다이버전스) → 확정 뒤 20봉 안에 종가<EMA면 매도
  E volfade : 랠리 상황(최근 60분 저점 +1.5%) + 거래량 비(SMA10/SMA40, 직전 봉) < thr 에서 종가가 EMA 하향 교차
  F stochx  : Stoch RSI K가 D를 하향 교차(직전 K>80) → W봉 안(0=같은 봉) 첫 종가<EMA 봉에서 매도
  REF cross : 선 단순 하향 교차(참고용, 선택 제외)
신호는 봉 완성 뒤 다음 1분봉 시가 체결(H.exec_index). 피벗은 확정 시점(j+R) 이후만 쓴다.
hold_sim에는 겹침 제거 전 사건 위치 전부를, summarize에는 dedup(30분)을 넘긴다.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

import harness as H

OUT = os.path.join(H.ROOT, "results", "exit_fail.json")


# ───────────── 공통 ─────────────
def lines(b: pd.DataFrame) -> dict:
    return {"EMA10": H.ema(b.Close, 10), "EMA25": H.ema(b.Close, 25), "EMA60": H.ema(b.Close, 60),
            "VWAP": H.session_vwap(b)}


def pivots(x: np.ndarray, R: int, kind: str) -> np.ndarray:
    """확정 피벗 표시(피벗 봉 j에 True). ta.pivothigh(R,R)와 같은 뜻: 왼쪽 R봉보다 크고 오른쪽 R봉 이상.
    사용은 반드시 j+R봉 완성 뒤."""
    n = len(x)
    out = np.zeros(n, dtype=bool)
    s = x if kind == "high" else -x
    for j in range(R, n - R):
        v = s[j]
        if np.isnan(v):
            continue
        left = s[j - R:j]
        right = s[j + 1:j + R + 1]
        if v > np.nanmax(left) and v >= np.nanmax(right):
            out[j] = True
    return out


def day_change(b: pd.DataFrame) -> np.ndarray:
    d = b["date"].values
    ch = np.ones(len(d), dtype=bool)
    ch[1:] = d[1:] != d[:-1]
    return ch


# ───────────── 후보 신호 ─────────────
def sig_cross(b, L):
    c = b.Close
    return (c < L) & (c.shift(1) >= L.shift(1))


def sig_retest(b, L, W):
    C, Hh, Lo = b.Close.to_numpy(), b.High.to_numpy(), b.Low.to_numpy()
    l = L.to_numpy()
    n = len(C)
    out = np.zeros(n, dtype=bool)
    brk = -10**9
    touched, rt_low, live = False, np.nan, False
    for i in range(1, n):
        if np.isnan(l[i]) or np.isnan(l[i - 1]):
            continue
        if C[i] < l[i] and C[i - 1] >= l[i - 1]:
            brk, touched, rt_low, live = i, False, np.nan, True
            continue
        if not live:
            continue
        if i - brk > W or C[i] >= l[i]:
            live = False
            continue
        if Hh[i] >= l[i]:                 # 선까지 되올라옴(종가는 선 아래)
            touched, rt_low = True, Lo[i]
        elif touched and C[i] < rt_low:   # 되돌림 봉 저가 종가 이탈 = 실패 확정
            out[i] = True
            live = False
    return pd.Series(out, index=b.index)


def sig_reject(b, L, W):
    """A2: 이탈 뒤 W봉 안, 고가가 선에 닿고 종가<선·종가<시가(음봉 거부)인 첫 봉에서 매도(되돌림 저가 이탈을 기다리지 않음)."""
    C, O, Hh = b.Close.to_numpy(), b.Open.to_numpy(), b.High.to_numpy()
    l = L.to_numpy()
    n = len(C)
    out = np.zeros(n, dtype=bool)
    brk, live = -10**9, False
    for i in range(1, n):
        if np.isnan(l[i]) or np.isnan(l[i - 1]):
            continue
        if C[i] < l[i] and C[i - 1] >= l[i - 1]:
            brk, live = i, True
            continue
        if not live:
            continue
        if i - brk > W or C[i] >= l[i]:
            live = False
            continue
        if Hh[i] >= l[i] and C[i] < O[i]:
            out[i] = True
            live = False
    return pd.Series(out, index=b.index)


def sig_dtop(b, R, tol, M=40, W=20):
    Hh, Lo, C = b.High.to_numpy(), b.Low.to_numpy(), b.Close.to_numpy()
    n = len(C)
    ph = pivots(Hh, R, "high")
    out = np.zeros(n, dtype=bool)
    piv = []            # 확정된 피벗 고점 위치
    neck, top, arm_until = np.nan, np.nan, -1
    for i in range(n):
        j = i - R
        if j >= 0 and ph[j]:              # 이번 봉 완성으로 j가 확정
            for p1 in reversed(piv):
                if j - p1 > M:
                    break
                if j - p1 > R and abs(Hh[j] / Hh[p1] - 1) <= tol:
                    neck = np.nanmin(Lo[p1 + 1:j])
                    top = max(Hh[p1], Hh[j])
                    arm_until = i + W
                    break
            piv.append(j)
        if i <= arm_until:
            if C[i] > top:
                arm_until = -1
            elif C[i] < neck and (i == 0 or C[i - 1] >= neck):
                out[i] = True
                arm_until = -1
    return pd.Series(out, index=b.index)


def sig_nhfail(b, N, W, R=3):
    Hh, Lo, C = b.High.to_numpy(), b.Low.to_numpy(), b.Close.to_numpy()
    n = len(C)
    pl = pivots(Lo, R, "low")
    prevmax = pd.Series(Hh).shift(1).rolling(N, min_periods=N).max().to_numpy()
    out = np.zeros(n, dtype=bool)
    last_pl = np.nan
    ref, until = np.nan, -1
    for i in range(n):
        j = i - R
        if j >= 0 and pl[j]:
            last_pl = Lo[j]
        if not np.isnan(prevmax[i]) and Hh[i] > prevmax[i]:   # 신고가 갱신 → 기준 저점 갱신
            ref, until = last_pl, i + W
            continue
        if i <= until and not np.isnan(ref) and C[i] < ref:
            out[i] = True
            until = -1
    return pd.Series(out, index=b.index)


def sig_rsidiv(b, R, L, M=40, W=20):
    Hh, C = b.High.to_numpy(), b.Close.to_numpy()
    r = H.rsi(b.Close, 14).to_numpy()
    l = L.to_numpy()
    n = len(C)
    ph = pivots(Hh, R, "high")
    out = np.zeros(n, dtype=bool)
    piv, until = [], -1
    for i in range(n):
        j = i - R
        if j >= 0 and ph[j]:
            if piv and j - piv[-1] <= M:
                p1 = piv[-1]
                if Hh[j] > Hh[p1] and r[j] < r[p1]:
                    until = i + W
            piv.append(j)
        if i <= until and C[i] < l[i]:
            out[i] = True
            until = -1
    return pd.Series(out, index=b.index)


def sig_volfade(b, L, thr, tf):
    rally = H.ctx_rally(b, max(1, 60 // tf), 0.015)
    vr = (b.Volume.rolling(10).mean() / b.Volume.rolling(40).mean()).shift(1)
    s = sig_cross(b, L) & rally.shift(1).fillna(False).astype(bool)
    if np.isfinite(thr):
        s &= vr < thr
    return s


def sig_stochx(b, L, W):
    K, D = H.stoch_rsi(b.Close)
    K, D = K.to_numpy(), D.to_numpy()
    C, l = b.Close.to_numpy(), L.to_numpy()
    n = len(C)
    out = np.zeros(n, dtype=bool)
    until = -1
    for i in range(1, n):
        if K[i] < D[i] and K[i - 1] >= D[i - 1] and K[i - 1] > 80:
            until = i + W
        if i <= until and C[i] < l[i]:
            out[i] = True
            until = -1
    return pd.Series(out, index=b.index)


def configs():
    cfg = []
    for tf in (1, 3):
        for ln in ("EMA10", "EMA25", "EMA60", "VWAP"):
            cfg.append(dict(fam="REF", tf=tf, line=ln))
            for W in (10, 30):
                cfg.append(dict(fam="A_retest", tf=tf, line=ln, W=W))
                cfg.append(dict(fam="A2_reject", tf=tf, line=ln, W=W))
        for R in (3, 5):
            for tol in (0.003, 0.008):
                cfg.append(dict(fam="B_dtop", tf=tf, R=R, tol=tol))
        for N in (20, 60):
            for W in (15, 40):
                cfg.append(dict(fam="C_nhfail", tf=tf, N=N, W=W))
        for R in (3, 5):
            for ln in ("EMA10", "EMA25"):
                cfg.append(dict(fam="D_rsidiv", tf=tf, R=R, line=ln))
        for ln in ("EMA10", "EMA25"):
            for thr in (0.7, 1.0, float("inf")):
                cfg.append(dict(fam="E_volfade", tf=tf, line=ln, thr=thr))
        for ln in ("EMA10", "EMA25"):
            for W in (0, 10):
                cfg.append(dict(fam="F_stochx", tf=tf, line=ln, W=W))
    return cfg


def make_signal(c, bars, lns):
    b = bars[c["tf"]]
    L = lns[c["tf"]].get(c.get("line", "EMA10"))
    f = c["fam"]
    if f == "REF":
        return sig_cross(b, L)
    if f == "A_retest":
        return sig_retest(b, L, c["W"])
    if f == "A2_reject":
        return sig_reject(b, L, c["W"])
    if f == "B_dtop":
        return sig_dtop(b, c["R"], c["tol"])
    if f == "C_nhfail":
        return sig_nhfail(b, c["N"], c["W"])
    if f == "D_rsidiv":
        return sig_rsidiv(b, c["R"], L)
    if f == "E_volfade":
        return sig_volfade(b, L, c["thr"], c["tf"])
    if f == "F_stochx":
        return sig_stochx(b, L, c["W"])
    raise ValueError(f)


def cname(c):
    return "_".join(f"{k}{v}" for k, v in c.items())


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def evaluate(c, bars, lns, RALLY, ranges):
    s = make_signal(c, bars, lns)
    raw = H.exec_index(s, bars[c["tf"]])
    raw = np.sort(raw[raw >= 0])
    dd = H.dedup(raw, 30)
    ev = H.event_metrics(dd)
    res = {}
    for nm, rng in ranges:
        hs = H.hold_sim(raw, rng)
        hr = H.hold_sim(raw, rng, entry_mask=RALLY)
        sm = H.summarize(ev, rng, "exit", RALLY)
        res[nm] = {"hold": hs, "hold_rally": hr, "sum": sm}
    return res, raw


def main():
    t0 = time.time()
    d1 = H.load_1m()
    b3 = H.resample(3)
    RALLY = H.to_1m_mask(H.ctx_rally(b3, 20, 0.015), b3)
    bars = {1: H.resample(1), 3: b3}
    lns = {tf: lines(bars[tf]) for tf in bars}
    ranges = (("TRAIN", H.TRAIN), ("TEST", H.TEST))
    rows = []
    case = {}
    for c in configs():
        res, raw = evaluate(c, bars, lns, RALLY, ranges)
        nm = cname(c)
        rows.append({"name": nm, "cfg": c, **res})
        t = d1.index[raw]
        sel = (t >= pd.Timestamp("2026-11-19 09:30", tz=H.ET)) & (t <= pd.Timestamp("2026-11-19 10:36", tz=H.ET))
        case[nm] = [f"{x:%H:%M} @{p:.2f}" for x, p in zip(t[sel], d1["Open"].values[raw][sel])]
        tr = res["TRAIN"]
        print(f"{nm:45s} TRAIN x={tr['hold']['x_rand_mean%']:+.3f} t={tr['hold']['x_rand_t_days']} "
              f"pd={tr['sum'].get('per_day')} | TEST x={res['TEST']['hold']['x_rand_mean%']:+.3f}", flush=True)
    # 선택: REF 제외, TRAIN 신호일 ≥6·하루 0.3~6회, TRAIN x_rand_mean% 순
    elig = [r for r in rows if r["cfg"]["fam"] != "REF" and r["TRAIN"]["sum"].get("days", 0) >= 6
            and 0.3 <= r["TRAIN"]["sum"].get("per_day", 0) <= 6]
    elig.sort(key=lambda r: r["TRAIN"]["hold"]["x_rand_mean%"], reverse=True)
    top = [r["name"] for r in elig[:3]]
    # 장세 위험 한 줄(PRE): 최종 후보만
    pre = {}
    for r in rows:
        if r["name"] in top:
            res, _ = evaluate(r["cfg"], bars, lns, RALLY, (("PRE", H.PRE),))
            pre[r["name"]] = res["PRE"]
    out = {
        "branch": "exit_fail", "generated": time.strftime("%Y-%m-%d %H:%M"),
        "n_configs": len(rows), "n_ref": sum(r["cfg"]["fam"] == "REF" for r in rows),
        "n_candidate_configs": sum(r["cfg"]["fam"] != "REF" for r in rows),
        "n_eligible": len(elig), "selection": "TRAIN only: days>=6, per_day 0.3~6, max hold_sim x_rand_mean%",
        "top3": top, "pre_check": pre, "case_2026_11_19": case, "rows": rows,
        "ranking_train": [(r["name"], r["TRAIN"]["hold"]["x_rand_mean%"], r["TRAIN"]["hold"]["x_rand_t_days"],
                           r["TEST"]["hold"]["x_rand_mean%"]) for r in elig],
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(clean(out), fh, ensure_ascii=False, indent=1)
    print("TOP3", top)
    print(round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
