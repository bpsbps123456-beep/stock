"""프리장 급락 매수 사례 연구.

질문: 프리장에서 전일 종가 대비 크게 빠졌을 때 지정가로 사서 언제 팔면 되나(BULL 2026-10-07, SMCI 2026-03-20).
사건: 프리장(04:00~09:30 ET)에 종가 -10% 이하 1시간봉이 연속 두 개 이상이고, 그 봉들 저가가 전일 정규장 종가 대비 -15% 이하인 날.
     yfinance 프리장 봉에는 오류 체결이 많다(CTAS 2026-09-22 08시 봉: -17% 한 번 찍고 다음 봉 -0.1%). 한 봉짜리는 버린다.
     거래 없는 시간의 이어 찍힌 봉(시가=고가=저가=종가)은 확인 봉으로 세지 않는다.
     1시간 뒤 봉으로 확인하므로 1시간 안에 완전히 회복한 진짜 급락도 빠진다(전략에 불리한 쪽).
매수: 전일 종가 대비 -X% 지정가. 그 값 이하 저가가 처음 찍힌 인정 봉에서 체결, 봉 시가가 이미 그 아래면 시가 체결.
     기본은 닿으면 체결, 엄격은 0.1% 관통(사용자 매매 조건 메모).
매도: 당일 정규장 시가 / 첫 1시간 종가(10:30 ET) / 당일 종가(참고, 직접 체결 불가) / 다음 날 시가 / 5·20거래일 뒤 종가.
매수 순간 알 수 있는 구분: 뉴스가 전날 애프터에 나왔나, 실적 발표인가, 시장(SPY)도 빠졌나, 사건 시점 시총.
사용: py analyze.py   (bars/는 fetch.py가 채운다)"""
import glob, json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BARS = os.path.join(HERE, "../bars")
RES = os.path.join(HERE, "../results")
EVENT = 0.15                        # 사건 문턱: 프리장 저가 -15%
CONFIRM = 0.10                      # 오류 체결 거르기: 그 봉 종가도 -10% 이하여야 인정(1시간 안 완전 회복은 버려져 보수적)
LIMITS = [0.15, 0.20, 0.25, 0.30, 0.35]
MIN_CAP = 1e9                       # 사건 시점 추정 시총(현재 주식 수 × 전일 종가)
TPS = [0.05, 0.10, 0.15]            # 익절 지정가(체결가 대비)


def sessions(x):
    m = x.index.hour * 60 + x.index.minute
    x = x.assign(day=x.index.normalize().tz_localize(None),
                 ses=np.select([m < 570, m < 960], ["pm", "reg"], "ah"))
    return x


def daily(x):
    """정규장 일봉(시가·종가·저가) — 1시간봉에서 만든다."""
    r = x[x.ses == "reg"]
    return r.groupby("day").agg(o=("Open", "first"), h=("High", "max"), l=("Low", "min"), c=("Close", "last"))


def market_moves():
    """SPY 프리장 시간별 종가의 전일 종가 대비 변화 {(day, hour): ret}."""
    s = sessions(pd.read_parquet(f"{BARS}/1h_SPY.parquet"))
    d = daily(s)
    prev = d.c.shift(1)
    out = {}
    for (day, h), c in s[s.ses == "pm"].groupby(["day", s[s.ses == "pm"].index.hour]).Close.last().items():
        if day in prev.index and pd.notna(prev[day]):
            out[(day, h)] = c / prev[day] - 1
    return out


def events_for(t, x, shares, spy):
    x = sessions(x)
    d = daily(x)
    days = list(d.index)
    rows = []
    pmx = x[x.ses == "pm"]
    c0b = pmx.day.map(d.c.shift(1))
    # 거래 없는 시간은 직전 가격이 시가=고가=저가=종가로 이어 찍혀서(SNDA 2024-11-18) 폭이 있는 봉만 센다
    okb = (c0b > 0) & (pmx.Close <= c0b * (1 - CONFIRM)) & (pmx.High > pmx.Low * 1.0005)
    # 오류 체결은 한 봉에서 끝난다 → -10% 이하 종가가 연속 두 봉 이상인 구간만 인정
    run = okb & (okb.groupby(pmx.day).shift(1, fill_value=False) | okb.groupby(pmx.day).shift(-1, fill_value=False))
    pmx = pmx.assign(conf=run)
    hitd = set(pmx.day[run & (pmx.Low <= c0b * (1 - EVENT))])
    cand = [i for i, day in enumerate(days) if i > 0 and day in hitd]
    if not cand:
        return rows
    byday = {k: g for k, g in x.groupby(["day", "ses"])}
    ext = x.ses != "reg"
    for i in cand:
        day, prev = days[i], days[i - 1]
        c0 = d.c.iloc[i - 1]
        pm0 = pmx[pmx.day == day]
        pm = pm0[pm0.conf]                                          # 인정 봉만(저가·체결 판단용)
        ah = byday.get((prev, "ah"), x.iloc[:0])
        ahc = ah.Close.iloc[-1] if len(ah) else c0
        reg = byday.get((day, "reg"), x.iloc[:0])
        # 첫 -15% 봉 시점의 시장(SPY) 변화
        first = pm[pm.Low <= c0 * (1 - EVENT)].index[0]
        spy_ret = spy.get((day, first.hour), np.nan)
        c20 = d.c.iloc[i - 21] if i >= 21 else np.nan
        # 실시간 확인 매수(미래 정보 없음): 인정 봉이 두 번째로 마감한 순간 그 종가에 산다
        second = pm.iloc[1] if len(pm) >= 2 else None
        ev = dict(ticker=t, day=day.date(), c0=c0, cap=shares * c0 if shares else np.nan,
                  ah_ret=ahc / c0 - 1, ah_low=(ah.Low.min() / c0 - 1) if len(ah) else 0.0,
                  pm_open=pm0.Open.iloc[0] / c0 - 1, pm_low=pm.Low.min() / c0 - 1,
                  pm_low_raw=pm0.Low.min() / c0 - 1,
                  pm_low_hour=pm.Low.idxmin().hour, pm_last=pm0.Close.iloc[-1] / c0 - 1,
                  first_hour=first.hour, spy=spy_ret, prior20=c0 / c20 - 1 if c20 == c20 else np.nan,
                  open=d.o.iloc[i] / c0 - 1, reg_low=d.l.iloc[i] / c0 - 1, close=d.c.iloc[i] / c0 - 1,
                  h1=reg.Close.iloc[0] / c0 - 1 if len(reg) else np.nan,
                  conf2=second.Close / c0 - 1 if second is not None else np.nan,
                  conf2_hour=pm.index[1].hour if second is not None else np.nan)
        for k, col in [(1, "d1o"), (1, "d1c"), (5, "d5c"), (20, "d20c")]:
            j = i + k
            v = (d.o if col.endswith("o") else d.c).iloc[j] if j < len(days) else np.nan
            ev[col] = v / c0 - 1
        # 5거래일 최저(당일 포함, 정규장) — 더 빠졌는지
        ev["low5"] = d.l.iloc[i:i + 6].min() / c0 - 1 if i + 5 < len(days) else np.nan
        # 지정가별 체결 — 프리장 인정 봉 안에서만
        for lim in LIMITS:
            L = c0 * (1 - lim)
            for tag, pen in (("", 1.0), ("s", 0.999)):
                hit = pm[pm.Low <= L * pen]
                if hit.empty:
                    ev[f"fill{tag}_{int(lim*100)}"] = np.nan
                    continue
                b = hit.iloc[0]
                fill = min(L, b.Open)
                ev[f"fill{tag}_{int(lim*100)}"] = fill / c0 - 1
                if tag == "":
                    ev[f"fillhour_{int(lim*100)}"] = hit.index[0].hour
                    after = pm[pm.index >= hit.index[0]]
                    ev[f"mae_{int(lim*100)}"] = min(after.Low.min(), d.l.iloc[i]) / fill - 1
                    # 익절 지정가: 체결 봉 다음 봉부터 5거래일 뒤 장 마감까지. 장외 봉은 오류 체결이 많아 종가로만 판정
                    if i + 5 < len(days):
                        end = days[i + 5] + pd.Timedelta(hours=16)
                        m = (x.index > hit.index[0]) & (x.index.tz_localize(None) < end)
                        path_hi = np.where(ext[m], x.Close[m], x.High[m])
                        for tp in TPS:
                            ev[f"tp{int(tp*100)}_{int(lim*100)}"] = tp if (path_hi >= fill * (1 + tp)).any()                                 else d.c.iloc[i + 5] / fill - 1
        rows.append(ev)
    return rows


def load_universe():
    p = os.path.join(BARS, "universe.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


_SPY, _UNI = None, None


def _init():
    global _SPY, _UNI
    _SPY, _UNI = market_moves(), load_universe()


def _one(f):
    t = os.path.basename(f)[3:-8]
    u = _UNI.get(t)
    shares = u["cap"] / u["price"] if u else None
    try:
        return events_for(t, pd.read_parquet(f), shares, _SPY)
    except Exception as e:                         # 한 종목 자료 문제로 전체를 멈추지 않는다
        print("건너뜀", t, e)
        return []


def build():
    from concurrent.futures import ProcessPoolExecutor
    files = [f for f in sorted(glob.glob(f"{BARS}/1h_*.parquet"))
             if os.path.basename(f)[3:-8] not in ("SPY", "QQQ")]
    rows = []
    with ProcessPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) - 1), initializer=_init) as pool:
        for r in pool.map(_one, files, chunksize=20):
            rows += r
    ev = pd.DataFrame(rows)
    ev.to_csv(os.path.join(RES, "events_raw.csv"), index=False)
    return ev


if __name__ == "__main__":
    ev = build()
    print(len(ev), "건")
