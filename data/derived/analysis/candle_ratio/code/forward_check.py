"""FORWARD.md 사전 등록 칸(40분 연장 N10 개수 따라가기 0.5/0.5 밤 보유)을 10/23 이후 자료로만 잰다.

py code/forward_check.py [SYM,SYM,...]
- yfinance 5분봉(프리·애프터, 최근 60일)을 받아 cache/forward/<SYM>_5m.parquet에 쌓는다(기존 cache와 이어 붙임).
- 결과: results/forward_<마지막 거래일>.csv, 화면에 판정 3개.
"""
import pathlib
import sys
import time
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from engine import CACHE, ET, HERE, awake_mask, positions, ratio, sim, tf_bars  # noqa: E402

FWD = CACHE / "forward"
START, WARM = date(2026, 10, 23), date(2026, 10, 1)
TF, SESS, N, B, S, COST = 40, "eth", 10, 0.5, 0.5, 0.001
MIN_DAYS = 30


def fetch(sym):
    parts = []
    for p in (CACHE / f"{sym}_5m.parquet", FWD / f"{sym}_5m.parquet"):
        if p.exists():
            parts.append(pd.read_parquet(p))
    for k in range(3):
        try:
            d = yf.download(sym, interval="5m", period="60d", prepost=True, progress=False, auto_adjust=False, threads=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            parts.append(d[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"]))
            break
        except Exception as e:  # 잠깐 막히면 쉬었다 다시
            print(sym, "다시 시도", type(e).__name__, flush=True)
            time.sleep(5 * (k + 1))
    for x in parts:
        x.index = x.index.tz_convert(ET)
    d = pd.concat(parts)
    d = d[~d.index.duplicated(keep="last")].sort_index()
    FWD.mkdir(exist_ok=True)
    d.to_parquet(FWD / f"{sym}_5m.parquet")
    return d


def mdd(eq):
    return float((eq / np.maximum.accumulate(eq) - 1).min())


def one(sym, today):
    d = fetch(sym)
    d = d[(d.index.date >= WARM) & (d.index.date < today)]  # 오늘(장중)은 뺀다
    tb = tf_bars(d, TF, SESS)
    O, C, V = (tb[k].values.astype(np.float64) for k in ("O", "C", "V"))
    seg = np.where(tb.day.values >= START, 0, -1).astype(np.int64)
    if (seg == 0).sum() < 2:
        return None
    p = ratio(O, C, V, N, "count")
    j = np.flatnonzero(seg == 0)
    px = np.r_[O[j], C[j[-1]]]  # 시가→다음 시가 경로, 마지막은 종가
    r = {"sym": sym, "days": int(pd.Series(tb.day.values[j]).nunique()), "bh": px[-1] / px[0] - 1, "bh_mdd": mdd(px / px[0])}
    for mode, aw, cost in (("ideal", np.ones(len(tb), np.bool_), 0.0), ("real", awake_mask(tb.start), COST)):
        out = np.zeros(11)
        sim(O, C, p, tb.lod.values, seg, aw, B, S, True, False, cost, out)
        x = out[4] / out[6]
        st = 1 + x * (px / px[0] - 1)
        r.update({f"{mode}_ret": float(np.expm1(out[0])), f"{mode}_mdd": float(np.expm1(out[8])), f"{mode}_x": x,
                  f"{mode}_static": st[-1] - 1, f"{mode}_static_mdd": mdd(st), f"{mode}_ent": int(out[2])})
        if mode == "real":
            pos, _ = positions(O, C, p, tb.lod.values, aw, B, S, True, False)
            # 큰 추세일: 정규장 종가(16:00 ET 전 마지막 봉 종가) 기준 날 수익 상위 |3|일의 평균 보유 비율
            t = pd.DatetimeIndex(tb.start)
            rth = (t.hour * 60 + t.minute) < 960
            day_close = pd.Series(C[rth], index=tb.day.values[rth]).groupby(level=0).last()
            dret = day_close.pct_change().loc[lambda s: s.index >= START].dropna()
            big = dret.abs().sort_values(ascending=False).index[:3]
            hold = pd.Series(pos[j], index=tb.day.values[j]).groupby(level=0).mean()
            dn = [hold.get(k, np.nan) for k in big if dret[k] < 0]
            up = [hold.get(k, np.nan) for k in big if dret[k] > 0]
            r["big_dn_hold"] = float(np.nanmean(dn)) if dn else np.nan
            r["big_up_hold"] = float(np.nanmean(up)) if up else np.nan
    return r


if __name__ == "__main__":
    today = pd.Timestamp.now(tz=ET).date()
    syms = sys.argv[1].split(",") if len(sys.argv) > 1 else sorted(p.stem[:-3] for p in CACHE.glob("*_5m.parquet"))
    rows = [x for x in (one(s, today) for s in syms) if x]
    df = pd.DataFrame(rows)
    days = int(df.days.median()) if len(df) else 0
    df.to_csv(HERE / "results" / f"forward_{today}.csv", index=False)
    pc = lambda v: f"{v * 100:+.1f}%"
    print(f"종목 {len(df)}, 앞으로 구간 거래일 {days} (10/23~, 판정은 {MIN_DAYS}일 이상)")
    m = df.median(numeric_only=True)
    print("실제 판 중앙: 원안", pc(m.real_ret), "낙폭", pc(m.real_mdd), "| 같은 비중", pc(m.real_static), "낙폭", pc(m.real_static_mdd),
          "| 보유", pc(m.bh), "낙폭", pc(m.bh_mdd), "| 비중", round(m.real_x, 2))
    print("이상적 판 중앙: 원안", pc(m.ideal_ret), "낙폭", pc(m.ideal_mdd))
    c1 = m.real_mdd > m.real_static_mdd
    c2 = m.real_ret >= m.real_static
    dn, up = df.big_dn_hold.mean(), df.big_up_hold.mean()
    c3 = dn <= 0.35 and up >= 0.65
    print(f"① 낙폭 < 같은 비중: {'통과' if c1 else '실패'}  ② 수익 ≥ 같은 비중: {'통과' if c2 else '실패'}  "
          f"③ 큰 추세일 하락 {dn:.2f}·상승 {up:.2f}: {'통과' if c3 else '실패'}")
    if days < MIN_DAYS:
        print("판정 보류: 거래일이 아직 부족하다.")
    for s in ("MU", "BE", "NBIS"):
        if s in set(df.sym):
            x = df[df.sym == s].iloc[0]
            print(s, "실제", pc(x.real_ret), "낙폭", pc(x.real_mdd), "| 같은 비중", pc(x.real_static), "낙폭", pc(x.real_static_mdd), "| 보유", pc(x.bh))
