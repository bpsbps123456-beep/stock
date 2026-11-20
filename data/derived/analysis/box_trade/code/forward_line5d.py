"""사전 등록한 5일저가선 전방 검증(report.md 결론 6·'관찰용 지표' 절, 2026-10-09 등록) — 2026-11-21 실행.

등록 내용 그대로
  - 기간: 2026-10-09 ~ 11-06 진입. 깊이 3%(결론 6의 숫자는 2% 기준이라 3%는 이번이 첫 확인).
  - 매수: L = 그날 이전 5거래일 정규장 저가 최솟값(5분봉 정규장 저가로 만든 일봉, verify_intra_B (a) 방식),
          정규장(9:30~16:00 ET) 봉 저가 < L x 0.97이면 min(봉 시가, 지정가) 체결, 종목당 하루 1회.
  - 매도 TP2W: 체결 봉 시작 < 11:00 ET면 +2% 지정가(체결 다음 정규장 봉 고가 >= 목표면 max(시가, 목표)),
          못 닿으면(또는 11:00 뒤 체결) 체결 뒤 첫 한국 08:00 이후 첫 봉 시가.
  - 초과 = 무작위1(같은 종목·같은 30분 칸, 기간 안 모든 정규장 봉 시가 매수 → 같은 매도) 대비. te = 날짜 군집 t(날짜별 합).
  - 승격 기준 셋: 전체 137에서 te >= 2, 사용자78 밖 종목도 플러스, 상위 1일(거래 수익 합 1위)을 빼도 플러스.
종목: verify_intra_B_trades.csv에 실린 137종목(10/9 고정 목록), 사용자78 = dip_reason fetch.user_universe()['합집합'].
봉: yfinance 5분봉(프리·애프터 포함, 60일 한도). 등록 당시 확인 구간은 3분봉이었다(5분봉은 앞 구간과 같은 단위).
사용: py forward_line5d.py [--refresh] → ../results/forward_line5d.txt, ../results/forward_line5d_trades.csv
캐시: ../cache/forward_5m.parquet (Git 제외, 60일이 지나면 다시 받을 수 없다)
"""
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BOX = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BOX, "..", "dip_reason", "code"))
from fetch import user_universe  # noqa: E402

LO, HI = date(2026, 10, 9), date(2026, 11, 6)
DEPTH = 0.03
CACHE = os.path.join(BOX, "cache", "forward_5m.parquet")
OUT_TXT = os.path.join(BOX, "results", "forward_line5d.txt")
OUT_CSV = os.path.join(BOX, "results", "forward_line5d_trades.csv")


def symbols():
    t = pd.read_csv(os.path.join(BOX, "results", "verify_intra_B_trades.csv"), usecols=["sym"])
    return sorted(t.sym.unique())


def bars(syms, refresh=False):
    if os.path.exists(CACHE) and not refresh:
        return pd.read_parquet(CACHE)
    import yfinance as yf

    d = yf.download(syms, interval="5m", period="60d", prepost=True, progress=False, auto_adjust=False,
                    group_by="ticker", threads=True)
    rows = []
    for s in syms:
        try:
            x = d[s][["Open", "High", "Low", "Close"]].dropna(subset=["Open", "High", "Low", "Close"]).copy()
        except KeyError:
            continue
        x = x[x.High > 0]
        if not len(x):
            continue
        x.index = x.index.tz_convert("America/New_York")
        x["sym"] = s
        rows.append(x)
    h = pd.concat(rows)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    h.to_parquet(CACHE)
    return h


def l5_map(dly):
    """날짜 → 그날 이전 5거래일 정규장 저가 최솟값. 하루 −40%/+80% 넘는 종가 변화(분할·자료 오류)가 창에 있으면 버림."""
    days, lo, cl = list(dly.index), dly.Low.values, dly.Close.values
    r = np.r_[np.nan, cl[1:] / cl[:-1] - 1]
    bad = (r <= -0.40) | (r >= 0.80)
    return {days[i]: (np.nan if bad[i - 5:i + 1].any() else lo[i - 5:i].min()) for i in range(5, len(days))}


def tp2w(o, h, reg, x8, e, fill, tp_on):
    d = x8[e]
    if d < 0:
        return np.nan
    px = o[d]
    if tp_on:
        tg = fill * 1.02
        for k in range(e + 1, d):
            if reg[k] and h[k] >= tg:
                return max(o[k], tg)
    return px


def one(x):
    x = x.sort_index()
    idx = x.index
    o, h, lo, c = (x[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    mod = (idx.hour * 60 + idx.minute).to_numpy()
    reg = (mod >= 570) & (mod < 960)
    day = np.array(idx.date)
    kst = idx.tz_convert("Asia/Seoul")
    t8 = kst.normalize() + pd.Timedelta(hours=8)
    nxt = t8.where(kst < t8, t8 + pd.Timedelta(days=1))
    x8 = np.searchsorted(idx.as_unit("ns").asi8, nxt.as_unit("ns").asi8, side="left")
    x8 = np.where(x8 < len(o), x8, -1)
    rb = x[reg]
    dly = rb.groupby(rb.index.date).agg(Low=("Low", "min"), Close=("Close", "last"))
    L = l5_map(dly)
    inper = (day >= LO) & (day <= HI)
    lim = np.array([L.get(d, np.nan) for d in day]) * (1 - DEPTH)
    # 무작위1: 기간 안 모든 정규장 봉 시가 매수 → 같은 매도, 30분 칸 평균
    ri = np.nonzero(reg & inper)[0]
    rr = np.array([tp2w(o, h, reg, x8, i, o[i], mod[i] < 660) / o[i] - 1 for i in ri])
    r1 = pd.Series(rr).groupby(mod[ri] // 30).mean()
    trades, seen = [], set()
    for i in np.nonzero(reg & inper & np.isfinite(lim) & (lo < lim))[0]:
        if day[i] in seen:
            continue
        seen.add(day[i])
        fill = min(o[i], lim[i])
        px = tp2w(o, h, reg, x8, i, fill, mod[i] < 660)
        if not np.isfinite(px):
            continue
        trades.append(dict(day=day[i], time=idx[i].strftime("%H:%M"), fill=fill, ret=px / fill - 1,
                           r1=r1.get(mod[i] // 30, np.nan), tp_on=bool(mod[i] < 660), gap=bool(o[i] < lim[i])))
    return trades


def dct(day, v):
    s = pd.Series(v).groupby(np.asarray(day)).sum()
    if len(s) < 3 or s.std(ddof=1) == 0:
        return np.nan
    return s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))


def summary(t):
    t = t[np.isfinite(t.exc)]
    if not len(t):
        return "n0"
    return (f"n {len(t)} · 평균 {t.ret.mean():+.2%} · 승률 {(t.ret > 0).mean():.0%} · 최악 {t.ret.min():+.1%} · 합 {t.ret.sum():+.2f}"
            f" · 초과 {t.exc.mean():+.2%} · te {dct(t.day.values, t.exc.values):+.2f} ({t.day.nunique()}일 {t.sym.nunique()}종목)")


def main(refresh=False):
    syms = symbols()
    u78 = set(user_universe()["합집합"])
    h = bars(syms, refresh)
    rows = []
    for s, x in h.groupby("sym"):
        for tr in one(x.drop(columns="sym")):
            rows.append({"sym": s, **tr})
    t = pd.DataFrame(rows).sort_values(["day", "time", "sym"]).reset_index(drop=True)
    t["exc"] = t.ret - t.r1
    t.to_csv(OUT_CSV, index=False, encoding="utf-8", float_format="%.6g")
    have = sorted(h.sym.unique())
    first = h.index.min().tz_convert("America/New_York").date()
    out = [f"5일저가선 전방 검증 (깊이 {DEPTH:.0%}, 진입 {LO}~{HI}, TP2W, 5분봉) — 실행 {date.today()}",
           f"종목 137 중 5분봉 받음 {len(have)}, 첫 봉 {first}, 사용자78 {len(u78)}", ""]
    t78, tout = t[t.sym.isin(u78)], t[~t.sym.isin(u78)]
    out += ["전체 137: " + summary(t), "사용자78: " + summary(t78), "78 밖: " + summary(tout)]
    bd = t.groupby("day").ret.sum().sort_values(ascending=False)
    top = bd.index[0]
    nt = t[t.day != top]
    out += [f"상위 1일 {top}: 거래 {int((t.day == top).sum())}건, 수익 합 {bd.iloc[0]:+.2f}(전체 합 {t.ret.sum():+.2f}의 {bd.iloc[0] / t.ret.sum():.0%})",
            "상위 1일 뺌: " + summary(nt), ""]
    te137 = dct(t.day.values, t.exc.values)
    c1 = te137 >= 2
    c2 = len(tout) > 0 and tout.ret.mean() > 0 and tout.exc.mean() > 0
    c3 = nt.ret.mean() > 0 and nt.exc.mean() > 0
    out += ["승격 기준",
            f"  1) 전체 137 te >= 2: {'통과' if c1 else '불합격'} (te {te137:+.2f})",
            f"  2) 사용자78 밖 종목도 플러스(평균·초과): {'통과' if c2 else '불합격'} (평균 {tout.ret.mean():+.2%}, 초과 {tout.exc.mean():+.2%})",
            f"  3) 상위 1일 빼도 플러스(평균·초과): {'통과' if c3 else '불합격'} (평균 {nt.ret.mean():+.2%}, 초과 {nt.exc.mean():+.2%})",
            f"  판정: {'승격 후보(사용자에게 관찰 화면 추가를 묻는다)' if c1 and c2 and c3 else '닫힘(승격 안 함)'}", ""]
    out.append("날짜별(전체 137): 날짜 건수 평균 초과")
    for d, g in t.groupby("day"):
        out.append(f"  {d} {len(g):>3} {g.ret.mean():+.2%} {g.exc.mean():+.2%}")
    text = "\n".join(out) + "\n"
    with open(OUT_TXT, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(text)
    return dict(te=te137, c1=c1, c2=c2, c3=c3)


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)
