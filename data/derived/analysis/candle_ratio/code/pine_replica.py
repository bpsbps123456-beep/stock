"""candle_ratio.pine 동치 확인: Pine 논리를 줄 단위로 옮긴 파이썬 재현 vs engine.positions.

py code/pine_replica.py  → results/pine_equiv.csv, results/pine_equiv_alltf.csv, results/pine_returns.csv

- 재현은 engine.ratio를 쓰지 않고 Pine 식(math.sum 창 합)으로 p를 따로 계산한다.
- 엔진 신호 봉: 시가 체결 봉(ev = ±1)의 바로 앞 봉. 장 마감 청산은 보유(pos = 1)이고 lod인 봉 자체(종가 체결).
- 재현의 session.islastbar 두 가지: 'time' = 봉 시작 + 시간봉 ≥ 세션 끝(트레이딩뷰 방식, 시각으로 정함),
  'lod' = 엔진처럼 그날 마지막으로 있는 봉. 둘은 자료에 빈 봉(MU 10/7~10/13 애프터)이 있을 때만 갈린다.
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np
import pandas as pd

from engine import BS, HERE, NS, RATIOS, SESS, SS, TF_5M, WIN, load, positions, ratio, segments, tf_bars

KIND_KO = {"count": "개수", "body": "몸통", "vol": "거래량"}
OUT = HERE / "results"


# ---------------------------------------------------------------------------
# Pine 재현 (candle_ratio.pine 줄 순서 그대로)
# ---------------------------------------------------------------------------

def pine_ratio(O, C, V, n: int, kind: str) -> np.ndarray:
    """// ── 비율 p (봉 마감 때 안다) ── 블록. 봉 하나 = 스크립트 한 번 실행."""
    m = len(O)
    nums = np.zeros(m)
    dens = np.zeros(m)
    p_out = np.full(m, np.nan)
    for bar_index in range(m):
        close, open_, volume = C[bar_index], O[bar_index], V[bar_index]
        up = 1.0 if close > open_ else 0.0                                   # float up = close > open ? 1.0 : 0.0
        dn = 1.0 if close < open_ else 0.0                                   # float dn = close < open ? 1.0 : 0.0
        num = close - open_ if kind == "몸통" else (volume * up if kind == "거래량" else up)
        den = abs(close - open_) if kind == "몸통" else (volume * (up + dn) if kind == "거래량" else up + dn)
        nums[bar_index] = num
        dens[bar_index] = den
        # math.sum(x, n): 최근 n개 합, 앞 n-1봉은 na
        sNum = math.nan if bar_index < n - 1 else float(nums[bar_index - n + 1: bar_index + 1].sum())
        sDen = math.nan if bar_index < n - 1 else float(dens[bar_index - n + 1: bar_index + 1].sum())
        if bar_index < n - 1 or math.isnan(sNum) or math.isnan(sDen):
            r = math.nan
        else:
            r = sNum / sDen if sDen > 1e-12 else 0.5
        p = (1.0 + r) / 2.0 if kind == "몸통" else r
        p_out[bar_index] = p
    return p_out


def pine_state(p, islastbar, dir_: str, b: float, s: float, dayflat: bool):
    """// ── 상태 기계 ── 블록과 신호 세 줄. 과거 봉은 barstate.isconfirmed = 참."""
    m = len(p)
    buy = np.zeros(m, np.bool_)
    sell = np.zeros(m, np.bool_)
    flat_a = np.zeros(m, np.bool_)
    fol = dir_ == "따라가기"
    st = 0                                                                   # var int st = 0
    for i in range(m):
        confirmed = True
        was = st                                                             # int was = st
        pi = p[i]
        if confirmed and not math.isnan(pi):
            if st == 0:
                if (pi > b) if fol else (pi < 1.0 - b):
                    st = 1
            else:
                if (pi < s) if fol else (pi > 1.0 - s):
                    st = 0
        flat = dayflat and bool(islastbar[i]) and was == 1
        buy[i] = confirmed and st == 1 and (was == 0 or flat)
        sell[i] = confirmed and st == 0 and was == 1 and not flat
        flat_a[i] = flat
    return buy, sell, flat_a


# ---------------------------------------------------------------------------
# 엔진 쪽 신호 봉
# ---------------------------------------------------------------------------

def engine_signals(O, C, p, lod, b, s, follow, dayflat):
    awake = np.ones(len(O), np.bool_)
    pos, ev = positions(O, C, p, lod, awake, b, s, follow, dayflat)
    m = len(O)
    buy = np.zeros(m, np.bool_)
    sell = np.zeros(m, np.bool_)
    buy[:-1] = ev[1:] == 1
    sell[:-1] = ev[1:] == -1
    flat = (pos == 1) & lod.astype(np.bool_) & bool(dayflat)
    return buy, sell, flat, pos, ev


def compare(rep, eng):
    """마지막 봉(엔진에 다음 봉 체결이 없는 봉)을 뺀 봉별 불일치 수."""
    rb, rs, rf = rep
    eb, es, ef = eng
    mm_b = int((rb[:-1] != eb[:-1]).sum())
    mm_s = int((rs[:-1] != es[:-1]).sum())
    mm_f = int((rf != ef).sum())
    tail = int(rb[-1]) + int(rs[-1])
    return mm_b, mm_s, mm_f, tail


def bars(sym, sess, tf):
    d = load(sym, "5m")
    tb = tf_bars(d, tf, sess)
    tb = tb[tb.day >= WIN["5m"]["warm"]].reset_index(drop=True)
    start_min = np.asarray(tb.start.dt.hour * 60 + tb.start.dt.minute)
    tb["islast_time"] = (start_min + tf) >= SESS[sess][1]
    return tb


def rep_returns(O, C, rb, rs, rf, seg):
    """재현 신호만으로 포지션을 만들어 엔진 sim과 같은 회계(시가→다음 시가, 청산 봉은 시가→종가)로 구간 수익."""
    m = len(O)
    h = 0
    ret = [0.0, 0.0]
    ent = [0, 0]
    pos = np.zeros(m, np.int64)
    for j in range(m):
        if j > 0:
            if rb[j - 1]:
                h = 1
                if seg[j] >= 0:
                    ent[seg[j]] += 1
            elif rs[j - 1] or rf[j - 1]:
                h = 0
        pos[j] = h
        if h == 1 and seg[j] >= 0:
            if rf[j]:
                r = math.log(C[j] / O[j])
            elif j + 1 < m:
                r = math.log(O[j + 1] / O[j])
            else:
                r = math.log(C[j] / O[j])
            ret[seg[j]] += r
    return ret, ent, pos


def main(sym="MU"):
    t0 = time.time()
    rows = []
    ret_rows = []
    for sess in ("eth", "rth"):
        tb = bars(sym, sess, 40)
        O, C, V = (tb[c].values.astype(np.float64) for c in ("O", "C", "V"))
        lod = tb.lod.values.astype(np.bool_)
        isl_t = tb.islast_time.values.astype(np.bool_)
        seg = segments(tb.day.values, WIN["5m"])
        print(sess, "40분봉", len(tb), "봉, lod", int(lod.sum()), "시각 기준 마지막 봉", int(isl_t.sum()),
              "둘이 다른 봉", int((lod != isl_t).sum()))
        for n in NS:
            for kind in RATIOS:
                pe = ratio(O, C, V, n, kind)
                pr = pine_ratio(O, C, V, n, KIND_KO[kind])
                both = ~np.isnan(pe) & ~np.isnan(pr)
                na_mm = int((np.isnan(pe) != np.isnan(pr)).sum())
                pdiff = float(np.abs(pe[both] - pr[both]).max())
                for follow in (True, False):
                    for dayflat in (False, True):
                        for b in BS:
                            for s in SS:
                                eb, es, ef, pos, ev = engine_signals(O, C, pe, lod, b, s, follow, dayflat)
                                rec = dict(sess=sess, tf=40, N=n, ratio=kind, dir="follow" if follow else "fade",
                                           dayflat=dayflat, b=b, s=s, p_maxdiff=pdiff, p_na_mm=na_mm,
                                           eng_buy=int(eb.sum()), eng_sell=int(es.sum()), eng_flat=int(ef.sum()))
                                for tag, isl in (("lod", lod), ("time", isl_t)):
                                    rb, rs, rf = pine_state(pr, isl, "따라가기" if follow else "거꾸로", b, s, dayflat)
                                    mb, ms, mf, tail = compare((rb, rs, rf), (eb, es, ef))
                                    rec[f"mm_buy_{tag}"] = mb
                                    rec[f"mm_sell_{tag}"] = ms
                                    rec[f"mm_flat_{tag}"] = mf
                                    rec[f"mm_{tag}"] = mb + ms + mf
                                    rec["last_bar_sig"] = tail
                                    if tag == "lod" and n == 10 and kind == "count" and follow and b == 0.5 and s == 0.5:
                                        ret, ent, rpos = rep_returns(O, C, rb, rs, rf, seg)
                                        ret_rows.append(dict(sess=sess, dayflat=dayflat, rep_ret_sel=ret[0], rep_ret_conf=ret[1],
                                                             rep_ent_sel=ent[0], rep_ent_conf=ent[1],
                                                             pos_mm=int((rpos != pos).sum())))
                                rows.append(rec)
    eq = pd.DataFrame(rows)
    eq.to_csv(OUT / "pine_equiv.csv", index=False)

    # 트레이딩뷰 대조용: 원안 칸(장 마감 청산 끔·켬)의 재현 신호 봉 목록
    sig_rows = []
    for sess in ("eth", "rth"):
        tb = bars(sym, sess, 40)
        O, C, V = (tb[c].values.astype(np.float64) for c in ("O", "C", "V"))
        pr = pine_ratio(O, C, V, 10, "개수")
        for dayflat in (False, True):
            rb, rs, rf = pine_state(pr, tb.islast_time.values, "따라가기", 0.5, 0.5, dayflat)
            for i in np.flatnonzero(rb | rs | rf):
                sig_rows.append(dict(sess=sess, dayflat=dayflat, bar_start_et=tb.start[i].strftime("%Y-%m-%d %H:%M"),
                                     bar_start_kst=tb.start[i].tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M"),
                                     signal="매수" if rb[i] else "매도", flat_close=bool(rf[i]), p=round(float(pr[i]), 3),
                                     open=round(O[i], 2), close=round(C[i], 2)))
    pd.DataFrame(sig_rows).to_csv(OUT / f"pine_signals_{sym}_40m.csv", index=False, encoding="utf-8-sig")

    # 원안 칸(N10·count·follow·0.5/0.5)을 5분봉 창 시간봉 72개 전부에서
    rows2 = []
    for sess in ("eth", "rth"):
        for tf in TF_5M:
            tb = bars(sym, sess, tf)
            if len(tb) < 40:
                continue
            O, C, V = (tb[c].values.astype(np.float64) for c in ("O", "C", "V"))
            lod = tb.lod.values.astype(np.bool_)
            isl_t = tb.islast_time.values.astype(np.bool_)
            pe = ratio(O, C, V, 10, "count")
            pr = pine_ratio(O, C, V, 10, "개수")
            for dayflat in (False, True):
                eb, es, ef, pos, ev = engine_signals(O, C, pe, lod, 0.5, 0.5, True, dayflat)
                rec = dict(sess=sess, tf=tf, dayflat=dayflat, bars=len(tb), eng_buy=int(eb.sum()), eng_sell=int(es.sum()),
                           eng_flat=int(ef.sum()), lod_vs_time=int((lod != isl_t).sum()))
                for tag, isl in (("lod", lod), ("time", isl_t)):
                    rb, rs, rf = pine_state(pr, isl, "따라가기", 0.5, 0.5, dayflat)
                    mb, ms, mf, tail = compare((rb, rs, rf), (eb, es, ef))
                    rec[f"mm_{tag}"] = mb + ms + mf
                rows2.append(rec)
    eq2 = pd.DataFrame(rows2)
    eq2.to_csv(OUT / "pine_equiv_alltf.csv", index=False)

    # 재현 신호로 다시 잰 원안 수익 vs 격자 결과
    g = pd.read_parquet(OUT / f"grid_{sym}_5m.parquet",
                        filters=[("tf", "==", 40), ("N", "==", 10), ("ratio", "==", "count"), ("dir", "==", "follow"),
                                 ("b", "==", 0.5), ("s", "==", 0.5), ("cost", "==", 0.0)])
    rr = pd.DataFrame(ret_rows)
    rr = rr.merge(g[["sess", "dayflat", "ret_sel", "ret_conf", "ent_sel", "ent_conf"]], on=["sess", "dayflat"])
    for c in ("rep_ret_sel", "rep_ret_conf", "ret_sel", "ret_conf"):
        rr[c + "_pct"] = (np.exp(rr[c]) - 1) * 100
    rr.to_csv(OUT / "pine_returns.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n[40분봉 전체 칸]", len(eq), "칸 (세션 2 × N 11 × 비율 3 × 방향 2 × 청산 2 × 문턱 16)")
    print("p 최대 차이", eq.p_maxdiff.max(), " p na 위치 불일치", int(eq.p_na_mm.sum()))
    for tag in ("lod", "time"):
        bad = eq[eq[f"mm_{tag}"] > 0]
        print(f"islastbar={tag}: 불일치 봉 합 {int(eq[f'mm_{tag}'].sum())}, 불일치 있는 칸 {len(bad)}"
              f" (청산 끔 {int((bad.dayflat == False).sum())}, 켬 {int((bad.dayflat == True).sum())})")
    orig = eq[(eq.N == 10) & (eq.ratio == "count") & (eq.dir == "follow") & (eq.b == 0.5) & (eq.s == 0.5)]
    print("\n[원안 칸]")
    print(orig[["sess", "dayflat", "eng_buy", "eng_sell", "eng_flat", "mm_lod", "mm_time", "mm_flat_time", "last_bar_sig"]].to_string(index=False))
    print("\n[원안 칸, 시간봉 72개]", len(eq2), "칸")
    for tag in ("lod", "time"):
        print(f"islastbar={tag}: 청산 끔 불일치 {int(eq2[eq2.dayflat == False][f'mm_{tag}'].sum())}, "
              f"켬 불일치 {int(eq2[eq2.dayflat == True][f'mm_{tag}'].sum())}")
    print("\n[재현 신호 수익 vs 격자]")
    print(rr[["sess", "dayflat", "rep_ret_sel_pct", "ret_sel_pct", "rep_ret_conf_pct", "ret_conf_pct",
              "rep_ent_sel", "ent_sel", "rep_ent_conf", "ent_conf", "pos_mm"]].round(3).to_string(index=False))
    print(round(time.time() - t0, 1), "초")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["MU"]))
