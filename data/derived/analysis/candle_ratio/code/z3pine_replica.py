"""candle_ratio_3zone.pine 동치 확인: Pine 논리를 줄 단위로 옮긴 파이썬 재현 vs zones3.prepare(flags + s).

py code/z3pine_replica.py            → MU·BE·NBIS × 추세 판정 25칸(Pine에서 고를 수 있는 zones3 분류 전부)
py code/z3pine_replica.py --all      → 위 + 144종목 × 기본 5칸(각 방식의 Pine 기본값)

결과
- results/z3pine_equiv.csv         종목 × 칸별 구간 불일치 수, 재료(p·s·ER·뒤집힘·p2·ADX·일봉 상태) 최대 차이
- results/z3pine_htf_variants.csv  큰 봉 일치의 경계 가정 3개(아래)별 불일치 수
- results/z3pine_equiv_144.csv     (--all) 144종목 × 기본 5칸
- results/z3pine_zones_MU_40m.csv  트레이딩뷰 대조 목록(ET·KST 시각, p, s, ER, 구간, 다른 방식 구간)

재현 원칙
- engine·zones3의 계산 함수를 쓰지 않는다. 봉(zones3.prepare의 40분 연장 봉)과 일봉 원자료만 받아 Pine 줄 순서대로 계산한다.
- 과거 봉에서 barstate.isconfirmed = 참, timenow = 지금(모든 봉 마감보다 뒤).
- math.sum(x, n): 앞 n−1봉은 na, 그 뒤는 최근 n개 합.
- request.security("D", lookahead_off)의 과거 봉 값(트레이딩뷰 내부 규칙이라 가정으로 둔다)
  A: 일봉 마감 시각 ≤ 차트 봉 마감 시각인 마지막 일봉(그 일봉의 마지막 차트 봉부터 보인다) — 주 가정
  B: 일봉 마감 시각 ≤ 차트 봉 시작 시각인 마지막 일봉(마감 뒤 시작한 봉부터 보인다)
  마감 비교: '>=' (이 지표) / '>' (trend_5zone.pine 원문)
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np
import pandas as pd

import zones3 as z3
from engine import CACHE, ET, HERE, LAST_DAY, ratio

OUT = HERE / "results"
NAN = math.nan
DEF = dict(n=10, b=0.5, s=0.5, mode="효율 비율", erLen=20, erTh=0.3, flLen=40, flMax=3, exN=20, exD=0.1,
           adxLen=14, adxTh=20.0)
ZLAB = {1: "상승 추세", -1: "하락 추세", 0: "애매"}

CONFIGS: list[tuple[str, dict]] = (
    [(f"ER_K{k}_th{t}", dict(mode="효율 비율", erLen=k, erTh=t)) for k in (10, 20, 40) for t in (0.2, 0.3, 0.4)]
    + [(f"FLIP_K{k}_m{m}", dict(mode="뒤집힘 수", flLen=k, flMax=m)) for k in (20, 40, 80) for m in (2, 3, 4)]
    + [(f"EXT_n{n}_d{d}", dict(mode="비율 뚜렷함", exN=n, exD=d)) for n, d in ((10, 0.2), (10, 0.3), (20, 0.1), (20, 0.2))]
    + [(f"ADX_th{t}", dict(mode="ADX", adxTh=float(t))) for t in (20, 25)]
    + [("HTF_tfD", dict(mode="큰 봉 일치"))]
)
DEFAULT5 = {"효율 비율": "ER_K20_th0.3", "뒤집힘 수": "FLIP_K40_m3", "비율 뚜렷함": "EXT_n20_d0.1", "ADX": "ADX_th20",
            "큰 봉 일치": "HTF_tfD"}


def to_ms(ts) -> np.ndarray:
    """시각(시간대 있음) → 1970 UTC부터 밀리초(Pine time 단위). 판다스 내부 단위(s·ms·us·ns)와 무관하게."""
    idx = pd.DatetimeIndex(ts).tz_convert("UTC")
    return np.asarray((idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1), dtype=np.int64)


def msum(arr: np.ndarray, i: int, n: int) -> float:
    """Pine math.sum(x, n)을 봉 i에서."""
    if i < n - 1:
        return NAN
    return float(arr[i - n + 1: i + 1].sum())


# ---------------------------------------------------------------------------
# Pine 재현 (candle_ratio_3zone.pine 줄 순서 그대로)
# ---------------------------------------------------------------------------

def pine_daily(O, C, n, b, s):
    """dailyState(len, bTh, sTh) — 정규장 일봉 위에서 돈다. 돌려주는 값 [ds, ds[1]]."""
    m = len(O)
    us = np.zeros(m)
    qs = np.zeros(m)
    dsC = np.full(m, NAN)
    dsP = np.full(m, NAN)
    ds = 0                                                      # var int ds = 0
    for bar_index in range(m):
        close, open_ = C[bar_index], O[bar_index]
        u = 1.0 if close > open_ else 0.0                        # float u = close > open ? 1.0 : 0.0
        d = 1.0 if close < open_ else 0.0                        # float d = close < open ? 1.0 : 0.0
        us[bar_index] = u
        qs[bar_index] = u + d
        a = msum(us, bar_index, n)                               # math.sum(u, len)
        q = msum(qs, bar_index, n)                               # math.sum(u + d, len)
        pd_ = NAN if bar_index < n - 1 or math.isnan(a) or math.isnan(q) else (a / q if q > 1e-12 else 0.5)
        if not math.isnan(pd_):
            if ds == 0:
                if pd_ > b:
                    ds = 1
            else:
                if pd_ < s:
                    ds = 0
        dsC[bar_index] = ds
        dsP[bar_index] = dsC[bar_index - 1] if bar_index > 0 else NAN   # ds[1]
    return dsC, dsP


def security_index(t_open, t_close, d_close, rule: str) -> np.ndarray:
    key = t_close if rule == "A" else t_open
    return np.searchsorted(d_close, key, side="right") - 1


def pine_run(bars: dict, daily: dict, inp: dict, rule: str = "A", cmp: str = ">="):
    """한 칸(입력 묶음)을 차트 봉 전체에 돌린다. 봉별 값 사전을 돌려준다."""
    O, H, L, C = bars["O"], bars["H"], bars["L"], bars["C"]
    t_open, t_close = bars["t_open"], bars["t_close"]
    n, b, s, mode = inp["n"], inp["b"], inp["s"], inp["mode"]
    erLen, erTh, flLen, flMax = inp["erLen"], inp["erTh"], inp["flLen"], inp["flMax"]
    exN, exD, adxLen, adxTh = inp["exN"], inp["exD"], inp["adxLen"], inp["adxTh"]
    m = len(O)
    timenow = np.iinfo(np.int64).max
    dsC, dsP = pine_daily(daily["O"], daily["C"], n, b, s)
    d_close = daily["t_close"]
    didx = security_index(t_open, t_close, d_close, rule)

    ups = np.zeros(m); uds = np.zeros(m); chgs = np.zeros(m)
    res = {k: np.full(m, NAN) for k in ("p", "p2", "er", "flN", "adx", "dState")}
    res["st"] = np.zeros(m, np.int64)
    res["trend"] = np.zeros(m, np.bool_)
    res["zone"] = np.zeros(m, np.int64)
    st = 0                                                                   # var int st = 0
    aTr = aP = aN = aDx = 0.0                                                # var float ...
    aCnt = 0                                                                 # var int aCnt = 0
    zone = 0                                                                 # var int zone = 0
    for bar_index in range(m):
        close, open_, high, low = C[bar_index], O[bar_index], H[bar_index], L[bar_index]
        confirmed = True
        # ── 비율 p, p2
        up = 1.0 if close > open_ else 0.0
        dn = 1.0 if close < open_ else 0.0
        ups[bar_index] = up
        uds[bar_index] = up + dn
        sU, sD = msum(ups, bar_index, n), msum(uds, bar_index, n)
        p = NAN if bar_index < n - 1 or math.isnan(sU) or math.isnan(sD) else (sU / sD if sD > 1e-12 else 0.5)
        sU2, sD2 = msum(ups, bar_index, exN), msum(uds, bar_index, exN)
        p2 = NAN if bar_index < exN - 1 or math.isnan(sU2) or math.isnan(sD2) else (sU2 / sD2 if sD2 > 1e-12 else 0.5)
        # ── 원안 상태 기계
        if confirmed and not math.isnan(p):
            if st == 0:
                if p > b:
                    st = 1
            else:
                if p < s:
                    st = 0
        res["st"][bar_index] = st
        # ── 효율 비율
        erPath = 0.0
        if bar_index >= erLen:
            for i in range(erLen):
                back = erLen - 1 - i
                erPath += abs(C[bar_index - back] - C[bar_index - back - 1])
        er = NAN if bar_index < erLen else (abs(close - C[bar_index - erLen]) / erPath if erPath > 0 else 0.0)
        # ── 뒤집힘 수
        st_prev = res["st"][bar_index - 1] if bar_index > 0 else st             # nz(st[1], st)
        chg = 1 if st != st_prev else 0
        chgs[bar_index] = chg
        flSum = msum(chgs, bar_index, flLen)                                    # 삼항식 밖에서 모든 봉에
        flN = NAN if bar_index < flLen else flSum
        # ── ADX
        adxV = NAN
        if bar_index >= 1:
            hUp = high - H[bar_index - 1]
            lDn = L[bar_index - 1] - low
            pdm = hUp if (hUp > lDn and hUp > 0) else 0.0
            ndm = lDn if (lDn > hUp and lDn > 0) else 0.0
            tr1 = max(high - low, abs(high - C[bar_index - 1]), abs(low - C[bar_index - 1]))
            if bar_index <= adxLen:
                aTr += tr1
                aP += pdm
                aN += ndm
            else:
                aTr = aTr - aTr / adxLen + tr1
                aP = aP - aP / adxLen + pdm
                aN = aN - aN / adxLen + ndm
            if bar_index >= adxLen:
                pdi = 100 * aP / aTr if aTr > 0 else 0.0
                ndi = 100 * aN / aTr if aTr > 0 else 0.0
                dx = 100 * abs(pdi - ndi) / (pdi + ndi) if pdi + ndi > 0 else 0.0
                aCnt += 1
                if aCnt <= adxLen:
                    aDx += dx
                    if aCnt == adxLen:
                        aDx = aDx / adxLen
                        adxV = aDx
                else:
                    aDx = (aDx * (adxLen - 1) + dx) / adxLen
                    adxV = aDx
        # ── 큰 봉 일치: [dC, dP, dClose] = request.security(...)
        k = didx[bar_index]
        dC, dP, dClose = (dsC[k], dsP[k], d_close[k]) if k >= 0 else (NAN, NAN, None)
        tc = t_close[bar_index]
        if dClose is None:
            dDone = True
        else:
            dDone = timenow >= dClose and (tc >= dClose if cmp == ">=" else tc > dClose)
        dState = dC if dDone else dP
        # ── 구간
        if mode == "효율 비율":
            trend = not math.isnan(er) and er >= erTh
        elif mode == "뒤집힘 수":
            trend = not math.isnan(flN) and flN < flMax
        elif mode == "비율 뚜렷함":
            trend = not math.isnan(p2) and abs(p2 - 0.5) >= exD - 1e-12
        elif mode == "ADX":
            trend = not math.isnan(adxV) and adxV >= adxTh
        else:
            trend = not math.isnan(dState) and dState == st
        if confirmed:
            zone = (1 if st == 1 else -1) if trend else 0
        for key, v in (("p", p), ("p2", p2), ("er", er), ("flN", flN), ("adx", adxV), ("dState", dState)):
            res[key][bar_index] = v
        res["trend"][bar_index] = trend
        res["zone"][bar_index] = zone
    return res


# ---------------------------------------------------------------------------
# 입력 자료와 zones3 쪽 값
# ---------------------------------------------------------------------------

def load_inputs(sym: str):
    P = z3.prepare(sym)
    tb = P["tb"]
    t_open = to_ms(tb.start)
    sess_end = to_ms(pd.DatetimeIndex(pd.to_datetime(tb.day.values)).tz_localize(ET) + pd.Timedelta(minutes=1200))
    t_close = np.minimum(t_open + z3.TF * 60_000, sess_end)                  # Pine time_close (연장 세션 끝 20:00 ET에서 자름)
    bars = dict(O=P["O"], H=P["H"], L=P["L"], C=P["C"], t_open=t_open, t_close=t_close)
    d1 = pd.read_parquet(CACHE / f"{sym}_1d.parquet")
    d1 = d1[pd.DatetimeIndex(d1.index).date <= LAST_DAY]
    d_days = pd.DatetimeIndex(d1.index).date
    d_close = to_ms(pd.DatetimeIndex(pd.to_datetime(d_days)).tz_localize(ET) + pd.Timedelta(minutes=960))  # 정규장 일봉 time_close
    daily = dict(O=d1.Open.values.astype(float), C=d1.Close.values.astype(float), t_close=d_close)
    return P, bars, daily, d1


def z3_zone(P, name):
    f = P["flags"][name]
    return np.where(f, np.where(P["s"] == 1, 1, -1), 0)


def z3_parts(P, d1):
    """zones3 계산 함수로 만든 재료(비교용)."""
    O, H, L, C, V = P["O"], P["H"], P["L"], P["C"], P["V"]
    return dict(
        p10=P["p10"], s=P["s"],
        er={k: z3.efficiency(C, k) for k in (10, 20, 40)},
        fl={k: z3.flips(P["s"], k) for k in (20, 40, 80)},
        p20=ratio(O, C, V, 20, "count"),
        adx=z3.adx(H, L, C, 14),
        hs=z3.last_daily(z3.daily_state(d1), P["tb"], P["end"]),
    )


def maxdiff(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    nan_mm = int((np.isnan(a) != np.isnan(b)).sum())
    both = ~np.isnan(a) & ~np.isnan(b)
    return (float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0), nan_mm


def check_symbol(sym: str, configs, variants: bool):
    P, bars, daily, d1 = load_inputs(sym)
    Z = z3_parts(P, d1)
    rows, vrows, runs = [], [], {}
    for name, over in configs:
        inp = {**DEF, **over}
        r = pine_run(bars, daily, inp)
        runs[name] = r
        zz = z3_zone(P, name)
        rec = dict(sym=sym, config=name, mode=inp["mode"], bars=len(zz), zone_mm=int((r["zone"] != zz).sum()),
                   trend_mm=int((r["trend"] != P["flags"][name]).sum()), s_mm=int((r["st"] != Z["s"]).sum()),
                   n_up=int((zz == 1).sum()), n_dn=int((zz == -1).sum()), n_mid=int((zz == 0).sum()))
        rec["p_maxdiff"], rec["p_na_mm"] = maxdiff(r["p"], Z["p10"])
        if inp["mode"] == "효율 비율":
            rec["val_maxdiff"], rec["val_na_mm"] = maxdiff(r["er"], Z["er"][inp["erLen"]])
            v = r["er"][~np.isnan(r["er"])]
            rec["min_margin"] = float(np.abs(v - inp["erTh"]).min())
        elif inp["mode"] == "뒤집힘 수":
            zf = Z["fl"][inp["flLen"]].astype(float); zf[zf < 0] = np.nan
            rec["val_maxdiff"], rec["val_na_mm"] = maxdiff(r["flN"], zf)
            rec["min_margin"] = float("nan")
        elif inp["mode"] == "비율 뚜렷함":
            ref = Z["p10"] if inp["exN"] == 10 else Z["p20"]
            rec["val_maxdiff"], rec["val_na_mm"] = maxdiff(r["p2"], ref)
            v = r["p2"][~np.isnan(r["p2"])]
            rec["min_margin"] = float(np.abs(np.abs(v - 0.5) - inp["exD"]).min())
        elif inp["mode"] == "ADX":
            rec["val_maxdiff"], rec["val_na_mm"] = maxdiff(r["adx"], Z["adx"])
            v = r["adx"][~np.isnan(r["adx"])]
            rec["min_margin"] = float(np.abs(v - inp["adxTh"]).min())
        else:
            rec["val_maxdiff"], rec["val_na_mm"] = maxdiff(r["dState"], Z["hs"])
            rec["min_margin"] = float("nan")
        rows.append(rec)
        if variants and name == "HTF_tfD":
            isb = np.asarray(pd.DatetimeIndex(P["tb"].start).hour * 60 + pd.DatetimeIndex(P["tb"].start).minute) == 920  # 15:20 ET 봉(16:00 마감)
            for rule, cmp in (("A", ">="), ("A", ">"), ("B", ">=")):
                rv = pine_run(bars, daily, inp, rule, cmp)
                mm = rv["zone"] != zz
                vrows.append(dict(sym=sym, rule=rule, cmp=cmp, zone_mm=int(mm.sum()), mm_at_1520=int((mm & isb).sum()),
                                  dstate_mm=int(((rv["dState"] != Z["hs"]) & ~(np.isnan(rv["dState"]) & np.isnan(Z["hs"]))).sum()),
                                  bars_1520=int(isb.sum())))
    return P, rows, vrows, runs


def zones_csv(sym, P, runs):
    tb = P["tb"]
    st = pd.DatetimeIndex(tb.start)
    r0 = runs[DEFAULT5["효율 비율"]]
    df = pd.DataFrame({
        "bar_start_et": st.strftime("%Y-%m-%d %H:%M"),
        "bar_start_kst": st.tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M"),
        "open": P["O"].round(4), "high": P["H"].round(4), "low": P["L"].round(4), "close": P["C"].round(4),
        "p": np.round(r0["p"], 4), "s": r0["st"], "ER": np.round(r0["er"], 4),
        "구간": [ZLAB[int(v)] for v in r0["zone"]],
    })
    extra = (("뒤집힘 수", "flips_K40", "flN", 0), ("비율 뚜렷함", "p20", "p2", 4), ("ADX", "ADX14", "adx", 2),
             ("큰 봉 일치", "일봉상태", "dState", 0))
    for mode, col, key, nd in extra:
        r = runs[DEFAULT5[mode]]
        df[col] = np.round(r[key], nd) if nd else pd.Series(r[key]).astype("Int64")
        df[f"구간_{mode}"] = [ZLAB[int(v)] for v in r["zone"]]
    df.to_csv(OUT / f"z3pine_zones_{sym}_40m.csv", index=False, encoding="utf-8-sig")
    return df


def main():
    t0 = time.time()
    do_all = "--all" in sys.argv
    rows, vrows = [], []
    for sym in ("MU", "BE", "NBIS"):
        P, r, v, runs = check_symbol(sym, CONFIGS, variants=True)
        rows += r
        vrows += v
        if sym == "MU":
            df = zones_csv(sym, P, runs)
            print("MU 대조 목록", len(df), "봉,", df["구간"].value_counts().to_dict())
        print(sym, "끝", round(time.time() - t0, 1), "초")
    eq = pd.DataFrame(rows)
    eq.to_csv(OUT / "z3pine_equiv.csv", index=False, encoding="utf-8-sig")
    vv = pd.DataFrame(vrows)
    vv.to_csv(OUT / "z3pine_htf_variants.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print("\n[MU·BE·NBIS × 25칸] 구간 불일치 합", int(eq.zone_mm.sum()), "/ 봉×칸", int(eq.bars.sum()),
          "| 추세 판정 불일치", int(eq.trend_mm.sum()), "| s 불일치", int(eq.s_mm.sum()),
          "| p 최대 차이", eq.p_maxdiff.max(), "na 위치", int(eq.p_na_mm.sum()),
          "| 재료 최대 차이", eq.val_maxdiff.max(), "na 위치", int(eq.val_na_mm.sum()))
    print(eq[eq.config.isin(DEFAULT5.values())][["sym", "config", "bars", "zone_mm", "trend_mm", "val_maxdiff", "val_na_mm",
                                                   "min_margin", "n_up", "n_dn", "n_mid"]].to_string(index=False))
    print("\n[큰 봉 일치 경계 가정]")
    print(vv.to_string(index=False))
    if do_all:
        syms = sorted({p.name.split("_")[0] for p in CACHE.glob("*_5m.parquet")})
        cfg5 = [(nm, ov) for nm, ov in CONFIGS if nm in DEFAULT5.values()]
        rows144 = []
        for i, sym in enumerate(syms):
            try:
                _, r, _, _ = check_symbol(sym, cfg5, variants=False)
                rows144 += r
            except Exception as e:  # 자료 없는 종목은 적고 넘어간다
                rows144.append(dict(sym=sym, config="ERROR", mode=str(e)[:80]))
            if (i + 1) % 24 == 0:
                print(i + 1, "종목", round(time.time() - t0, 1), "초")
        e144 = pd.DataFrame(rows144)
        e144.to_csv(OUT / "z3pine_equiv_144.csv", index=False, encoding="utf-8-sig")
        ok = e144[e144.config != "ERROR"]
        print("\n[144종목 × 기본 5칸] 종목", ok.sym.nunique(), "오류", int((e144.config == "ERROR").sum()),
              "구간 불일치 합", int(ok.zone_mm.sum()), "/ 봉×칸", int(ok.bars.sum()))
        print(ok.groupby("config")[["zone_mm", "trend_mm", "s_mm", "val_maxdiff", "val_na_mm"]].agg(["sum", "max"]).to_string())
        print("문턱까지 최소 거리(종목 최솟값):", ok.groupby("config").min_margin.min().to_dict())
    print(round(time.time() - t0, 1), "초")


if __name__ == "__main__":
    main()
