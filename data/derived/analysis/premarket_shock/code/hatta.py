"""사용자 '하따'(급락 후 따라잡기) 규칙 검증 — 최근 30일 미국 급락일, 1분봉·3분봉.

사용자 규칙(2026-10-08 대화, 지표 코드는 사용자가 붙여 준 트레이딩뷰 스크립트):
  지표  CapVol(투매): 봉 종가가 직전 봉 종가 대비 -5.3% 미만, 거래량 > 그 봉 포함 20봉 단순평균 × 2
        Ultimate Buy and Sell(UBS, 짧은 이름 'Ultimate'): RSI 32(RMA), RSI 볼린저(SMA 32, 표준편차 2, 모집단),
          주의 신호 = RSI가 아래 밴드를 상향 돌파 또는 30 하향 돌파(매도는 위 밴드 하향 돌파·70 상향 돌파), 35봉 안 유지.
          매수 = RSI가 기준선(SMA 32) 상향 돌파 또는 25 상향 돌파, 매도 = 기준선 하향 돌파 또는 75 하향 돌파.
          신호가 나면 두 주의 목록을 비운다. 봉 마감 기준.
        VR(사용자 코드 미수령 — 한국 통상 정의): 20봉 (상승 거래량 + 보합/2) / (하락 거래량 + 보합/2) × 100,
          70 아래에서 위로 돌파하면 신호(가정).
  매수  그날 정규장부터 3분봉 투매를 센다(RSI 14 > 55인 투매는 세지 않는다). 둘째 투매는 고가가 첫째 투매 저가보다
        낮아야 센다. 둘째 투매 종가에 1.0(1,000만원) 매수, 그 뒤 -5%마다 0.5(500만원) 지정가 추가.
        1분봉 UBS 매수 신호에 0.5(500만원), 그 뒤 VR 신호에 0.3(300만원) 추가.
  매도  1분봉 UBS 매도 신호와 3분봉 RSI 14 ≥ 70 중 먼저 나온 봉 종가에 70% 매도. 나머지 30%는 규칙에 없어 다음 날 시가.
        그날 매도 신호가 없으면 전부 다음 날 시가.
자료 한계: 트레이딩뷰 정규장 차트를 재현한다(전날 정규장 마지막 봉이 첫 봉의 직전 봉). Yahoo 프리장 1분봉은 거래량이 0이라
  프리장 투매는 셀 수 없다. 1분봉은 최근 30일·요청당 8일.
사용: py hatta.py events → fetch → run → results/hatta.txt"""
import io, json, os, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../results")
CACHE = os.path.join(RES, "hatta")
CAP_DROP, CAP_VOL, CAP_LEN = 5.3, 2.0, 20
RSI_MAX_COUNT = 55
UNIT0, UNIT_DCA, UNIT_UBS, UNIT_VR, DCA_STEP = 1.0, 0.5, 0.5, 0.3, 0.05
RSI_HOT = 70
SELL_FRACTION = 0.70
AWAKE_END = 11 * 60


# ---------------------------------------------------------------- 지표
def rma(x: pd.Series, n: int) -> pd.Series:
    """Pine ta.rma: 처음 n개 단순평균으로 시작하는 지수평균(alpha=1/n)."""
    a = x.to_numpy(float)
    out = np.full(len(a), np.nan)
    if len(a) < n:
        return pd.Series(out, index=x.index)
    out[n - 1] = np.nanmean(a[:n])
    for i in range(n, len(a)):
        out[i] = out[i - 1] + (a[i] - out[i - 1]) / n
    return pd.Series(out, index=x.index)


def rsi(close: pd.Series, n: int) -> pd.Series:
    ch = close.diff()
    up, dn = rma(ch.clip(lower=0).fillna(0), n), rma((-ch.clip(upper=0)).fillna(0), n)
    r = 100 - 100 / (1 + up / dn)
    return r.where(dn != 0, 100).where(up != 0, 0)


def capvol(b: pd.DataFrame, drop=CAP_DROP, mult=CAP_VOL, n=CAP_LEN) -> pd.Series:
    avg = b.Volume.rolling(n, min_periods=n).mean()          # 그 봉 포함(Pine ta.sma)
    chg = (b.Close / b.Close.shift(1) - 1) * 100
    return (b.Volume > avg * mult) & (chg < -drop)


def ubs(close: pd.Series, n=32, ma_n=32, mult=2.0, lookback=35):
    """Ultimate Buy and Sell 기본 설정의 매수·매도 신호(봉 마감)."""
    r = rsi(close, n)
    basis = r.rolling(ma_n, min_periods=ma_n).mean()
    dev = r.rolling(ma_n, min_periods=ma_n).std(ddof=0)
    upper, lower = basis + mult * dev, basis - mult * dev
    cross_up = lambda a, b: (a > b) & (a.shift(1) <= b.shift(1))
    cross_dn = lambda a, b: (a < b) & (a.shift(1) >= b.shift(1))
    const = lambda v: pd.Series(v, index=r.index)
    buy_watch = (cross_up(r, lower) | cross_dn(r, const(30))).to_numpy()
    sell_watch = (cross_dn(r, upper) | cross_up(r, const(70))).to_numpy()
    buy_sig = (cross_up(r, basis) | cross_up(r, const(25))).to_numpy()
    sell_sig = (cross_dn(r, basis) | cross_dn(r, const(75))).to_numpy()
    bw, sw = [], []
    buys, sells = np.zeros(len(r), bool), np.zeros(len(r), bool)
    for i in range(len(r)):
        bw.append(bool(buy_watch[i])); sw.append(bool(sell_watch[i]))
        bw, sw = bw[-lookback:], sw[-lookback:]
        if buy_sig[i] and any(bw):
            buys[i] = True; bw, sw = [], []
        elif sell_sig[i] and any(sw):
            sells[i] = True; bw, sw = [], []
    return pd.Series(buys, index=r.index), pd.Series(sells, index=r.index)


def vr(b: pd.DataFrame, n=20) -> pd.Series:
    ch = b.Close.diff()
    upv = b.Volume.where(ch > 0, 0.0)
    dnv = b.Volume.where(ch < 0, 0.0)
    flat = b.Volume.where(ch == 0, 0.0)
    num = upv.rolling(n).sum() + flat.rolling(n).sum() / 2
    den = dnv.rolling(n).sum() + flat.rolling(n).sum() / 2
    return num / den.replace(0, np.nan) * 100


# ---------------------------------------------------------------- 자료
def events():
    """최근 30일(9/8~) 정규장 시가 -7% 이하 또는 장중 저가 -10% 이하, 시총 10억 달러 이상."""
    import glob
    from analyze import sessions, daily, load_universe
    uni = load_universe()
    rows = []
    for f in glob.glob(os.path.join(HERE, "../bars/1h_*.parquet")):
        t = os.path.basename(f)[3:-8]
        if t in ("SPY", "QQQ") or t not in uni:
            continue
        x = pd.read_parquet(f)
        x = x[x.index >= pd.Timestamp("2026-08-28", tz="America/New_York")]
        if x.empty:
            continue
        d = daily(sessions(x))
        c0 = d.c.shift(1)
        shares = uni[t]["cap"] / uni[t]["price"]
        for day in d.index:
            if day < pd.Timestamp("2026-09-08") or day >= pd.Timestamp("2026-10-07") or not c0[day] > 0:
                continue
            gap, low = d.o[day] / c0[day] - 1, d.l[day] / c0[day] - 1
            if (gap <= -0.07 or low <= -0.10) and shares * c0[day] >= 1e9:
                rows.append((t, day.date(), gap, low, shares * c0[day]))
    ev = pd.DataFrame(rows, columns=["ticker", "day", "gap", "low", "cap"])
    ev.to_csv(os.path.join(RES, "hatta_events.csv"), index=False)
    print(len(ev), "건")


def fetch():
    import yfinance as yf
    os.makedirs(CACHE, exist_ok=True)
    ev = pd.read_csv(os.path.join(RES, "hatta_events.csv"))
    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    jobs = [(t, d) for t, d in zip(ev.ticker, ev.day) if not os.path.exists(os.path.join(CACHE, f"1m_{t}_{d}.parquet"))]

    def one(job):
        t, d = job
        day = pd.Timestamp(d)
        start = max(day - pd.Timedelta(days=4), today - pd.Timedelta(days=29))
        end = min(start + pd.Timedelta(days=8), today + pd.Timedelta(days=1))
        try:
            x = yf.download(t, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval="1m",
                            prepost=False, progress=False, auto_adjust=False)
            if isinstance(x.columns, pd.MultiIndex):
                x.columns = x.columns.get_level_values(0)
            if len(x):
                x.index = x.index.tz_convert("America/New_York")
                x[["Open", "High", "Low", "Close", "Volume"]].to_parquet(os.path.join(CACHE, f"1m_{t}_{d}.parquet"))
            return 1
        except Exception as e:
            print("실패", t, d, e)
            return 0
    print("받을 사건", len(jobs))
    with ThreadPoolExecutor(max_workers=6) as pool:
        print("받음", sum(pool.map(one, jobs)))


def bars3(m1: pd.DataFrame) -> pd.DataFrame:
    """정규장 1분봉 → 날짜마다 09:30에 맞춘 3분봉(정규장 차트처럼 날짜를 이어 붙인다)."""
    parts = []
    for _, g in m1.groupby(m1.index.date):
        parts.append(g.resample("3min", origin="start_day").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(subset=["Open"]))
    return pd.concat(parts) if parts else m1.iloc[:0]


# ---------------------------------------------------------------- 시뮬레이션
def simulate(m1: pd.DataFrame, day, drop=CAP_DROP, awake_only=False) -> dict:
    day = pd.Timestamp(day).date()
    m = m1.index.hour * 60 + m1.index.minute
    m1 = m1[(m >= 570) & (m < 960)]
    b3 = bars3(m1)
    if b3.empty or day not in set(b3.index.date):
        return {"status": "자료 없음"}
    rsi3 = rsi(b3.Close, 14)
    cap = capvol(b3, drop=drop)
    ub_buy, ub_sell = ubs(m1.Close)
    vr1 = vr(m1)
    vr_sig = (vr1 > 70) & (vr1.shift(1) <= 70)
    today3 = b3[b3.index.date == day]
    today1 = m1[m1.index.date == day]
    later = m1[m1.index.date > day]
    next_open = later.Open.iloc[0] if len(later) else np.nan
    end_of = lambda ts, mins: ts + pd.Timedelta(minutes=mins)
    awake = lambda ts: (ts.hour * 60 + ts.minute) <= AWAKE_END
    # 투매 세기(3분봉 마감 시각에 안다)
    counts = [ts for ts in today3.index if cap[ts] and rsi3[ts] <= RSI_MAX_COUNT]
    out = {"caps": len(counts), "status": "투매 0개"}
    if not counts:
        return out
    c1 = counts[0]
    l1 = b3.Low[c1]
    c2 = next((ts for ts in counts[1:] if b3.High[ts] < l1), None)
    out.update(c1=end_of(c1, 3).strftime("%H:%M"), status="둘째 투매 없음")
    if c2 is None:
        return out
    entry_t = end_of(c2, 3)
    if awake_only and not awake(c2):
        out.update(status="둘째 투매가 11시 뒤")
        return out
    px0 = b3.Close[c2]
    lots = [(entry_t, px0, UNIT0, "둘째 투매")]
    path = today1[today1.index >= entry_t]
    # 매도 신호: 1분봉 UBS 매도(봉 마감) 또는 3분봉 RSI 14 ≥ 70(봉 마감)
    sell_1m = [end_of(ts, 1) for ts in path.index if ub_sell[ts]]
    hot_3m = [end_of(ts, 3) for ts in today3.index if end_of(ts, 3) > entry_t and rsi3[ts] >= RSI_HOT]
    exits = sorted(([(t_, "UBS 매도") for t_ in sell_1m] + [(t_, "RSI 70") for t_ in hot_3m]))
    if awake_only:
        exits = [e for e in exits if (e[0].hour * 60 + e[0].minute) <= AWAKE_END]
    exit_t, exit_why = exits[0] if exits else (None, None)
    horizon = path[path.index < exit_t] if exit_t is not None else path
    # -5%마다 0.5 지정가(자는 동안에도 걸어 둘 수 있다)
    k = 1
    for ts, row in horizon.iterrows():
        while row.Low <= px0 * (1 - DCA_STEP * k) and k <= 6:
            lots.append((ts, px0 * (1 - DCA_STEP * k), UNIT_DCA, f"-{int(DCA_STEP * 100 * k)}%"))
            k += 1
    # 1분봉 UBS 매수 → 그 뒤 VR 신호
    ub_t = next((ts for ts in horizon.index if ub_buy[ts] and (not awake_only or awake(end_of(ts, 1)))), None)
    if ub_t is not None:
        lots.append((end_of(ub_t, 1), m1.Close[ub_t], UNIT_UBS, "UBS 매수"))
        vr_t = next((ts for ts in horizon.index if ts > ub_t and vr_sig[ts] and (not awake_only or awake(end_of(ts, 1)))), None)
        if vr_t is not None:
            lots.append((end_of(vr_t, 1), m1.Close[vr_t], UNIT_VR, "VR"))
    units = sum(u for _, _, u, _ in lots)
    cost = sum(p * u for _, p, u, _ in lots)
    if exit_t is not None:
        bar = m1[m1.index < exit_t].iloc[-1]
        px_exit = bar.Close
        rest = next_open if next_open == next_open else today1.Close.iloc[-1]
        proceeds = units * (SELL_FRACTION * px_exit + (1 - SELL_FRACTION) * rest)
    else:
        rest = next_open if next_open == next_open else today1.Close.iloc[-1]
        px_exit, proceeds = None, units * rest
    out.update(status="매수", c2=entry_t.strftime("%H:%M"), entry=px0, units=units, avg=cost / units,
               exit=exit_t.strftime("%H:%M") if exit_t is not None else None, why=exit_why,
               ret=proceeds / cost - 1, ret_close=today1.Close.iloc[-1] / (cost / units) - 1,
               mae=today1[today1.index >= entry_t].Low.min() / (cost / units) - 1,
               lots=[(t_.strftime("%H:%M"), round(p, 2), u, w) for t_, p, u, w in lots])
    return out


def run():
    ev = pd.read_csv(os.path.join(RES, "hatta_events.csv"))
    ev = ev[ev.gap > -0.6]                       # CTVA 10/1 -81%는 회사 분할(가격 조정)
    rows = []
    for t, d, gap, low in zip(ev.ticker, ev.day, ev.gap, ev.low):
        p = os.path.join(CACHE, f"1m_{t}_{d}.parquet")
        if not os.path.exists(p):
            continue
        m1 = pd.read_parquet(p)
        for drop in (CAP_DROP, 3.0, 2.0):
            for awake_only in (False, True):
                r = simulate(m1, d, drop=drop, awake_only=awake_only)
                rows.append(dict(ticker=t, day=d, gap=gap, low=low, drop=drop, awake=awake_only, **{k: v for k, v in r.items() if k != "lots"},
                                 lots=json.dumps(r.get("lots"), ensure_ascii=False)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "hatta_runs.csv"), index=False)
    o = io.StringIO()
    p_ = lambda *a: print(*a, file=o)
    p_(f"사건 {df.groupby(['ticker', 'day']).ngroups}개(9/8~10/6, 시가 -7% 이하 또는 장중 저가 -10% 이하, 시총 10억 달러 이상)")
    for drop in (CAP_DROP, 3.0, 2.0):
        for awake_only in (False, True):
            x = df[(df["drop"] == drop) & (df.awake == awake_only)]
            st = x.status.value_counts().to_dict()
            buy = x[x.status == "매수"]
            line = f"투매 문턱 {drop}%{' · 11시 전만' if awake_only else ''}: {st}"
            if len(buy):
                line += (f" | 매수 {len(buy)}건 평균 {buy.ret.mean():+.1%} 중앙 {buy.ret.median():+.1%} 승률 {(buy.ret > 0).mean():.0%} "
                         f"최악 {buy.ret.min():+.1%} · 평균 매수 단위 {buy.units.mean():.1f} · 매수 뒤 최대 손실 중앙 {buy.mae.median():+.1%}")
            p_(line)
    p_("\n[매수가 난 사건]")
    for _, r in df[(df.status == "매수") & (~df.awake)].sort_values(["drop", "day"]).iterrows():
        p_(f"  문턱 {r['drop']}% {r.ticker} {r.day} 갭 {r.gap:+.1%} 저가 {r.low:+.1%} | 첫 투매 {r.c1} 둘째 {r.c2} 매도 {r.exit}({r.why}) "
           f"단위 {r.units:.1f} 수익 {r.ret:+.1%} 최대 손실 {r.mae:+.1%} | {r.lots}")
    txt = o.getvalue()
    open(os.path.join(RES, "hatta.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__" and sys.argv[1] in ("events", "fetch", "run"):
    {"events": events, "fetch": fetch, "run": run}[sys.argv[1]]()


def component():
    """규칙 조각 시험: 첫 투매(5.3%·2배·RSI≤55) 뒤 1분봉 UBS 매수 신호에 1.0만 사고(물타기 없음),
    매도는 사용자 규칙(UBS 매도·3분봉 RSI 70 중 먼저 70%, 나머지 다음 날 시가). 비교: 첫 투매 종가 매수, 시가 매수."""
    ev = pd.read_csv(os.path.join(RES, "hatta_events.csv"))
    ev = ev[ev.gap > -0.6]
    rows = []
    for t, d in zip(ev.ticker, ev.day):
        p = os.path.join(CACHE, f"1m_{t}_{d}.parquet")
        if not os.path.exists(p):
            continue
        m1 = pd.read_parquet(p)
        mm = m1.index.hour * 60 + m1.index.minute
        m1 = m1[(mm >= 570) & (mm < 960)]
        day = pd.Timestamp(d).date()
        b3 = bars3(m1)
        if day not in set(b3.index.date):
            continue
        rsi3, cap = rsi(b3.Close, 14), capvol(b3)
        ub_buy, ub_sell = ubs(m1.Close)
        t3, t1 = b3[b3.index.date == day], m1[m1.index.date == day]
        later = m1[m1.index.date > day]
        nxt = later.Open.iloc[0] if len(later) else t1.Close.iloc[-1]
        c1 = next((ts for ts in t3.index if cap[ts] and rsi3[ts] <= RSI_MAX_COUNT), None)
        if c1 is None:
            continue
        c1_end = c1 + pd.Timedelta(minutes=3)

        def trade(entry_t, px, awake_only):
            sells = [ts + pd.Timedelta(minutes=1) for ts in t1.index if ts + pd.Timedelta(minutes=1) > entry_t and ub_sell[ts]]
            hots = [ts + pd.Timedelta(minutes=3) for ts in t3.index if ts + pd.Timedelta(minutes=3) > entry_t and rsi3[ts] >= RSI_HOT]
            ex = sorted(sells + hots)
            if awake_only:
                ex = [e for e in ex if e.hour * 60 + e.minute <= AWAKE_END]
            if ex:
                px_x = m1[m1.index < ex[0]].Close.iloc[-1]
                return SELL_FRACTION * px_x / px + (1 - SELL_FRACTION) * nxt / px - 1
            return nxt / px - 1
        for awake_only in (False, True):
            ub_t = next((ts for ts in t1.index if ts >= c1_end and ub_buy[ts]
                         and (not awake_only or (ts.hour * 60 + ts.minute + 1) <= AWAKE_END)), None)
            r = dict(ticker=t, day=d, awake=awake_only)
            r["open"] = trade(t1.index[0], t1.Open.iloc[0], awake_only)
            if not awake_only or c1_end.hour * 60 + c1_end.minute <= AWAKE_END:
                r["c1"] = trade(c1_end, b3.Close[c1], awake_only)
            if ub_t is not None:
                r["ubs"] = trade(ub_t + pd.Timedelta(minutes=1), m1.Close[ub_t], awake_only)
                r["ubs_at"] = (ub_t + pd.Timedelta(minutes=1)).strftime("%H:%M")
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "hatta_component.csv"), index=False)
    o = io.StringIO()
    p_ = lambda *a: print(*a, file=o)
    for awake_only in (False, True):
        x = df[df.awake == awake_only]
        p_(f"\n[첫 투매가 나온 사건 {len(x)}건{' · 매수·매도 판단은 11시 전만' if awake_only else ''}] 매도는 사용자 규칙")
        for k, lab in (("open", "시가 매수"), ("c1", "첫 투매 종가 매수"), ("ubs", "첫 투매 뒤 UBS 1분봉 매수")):
            v = x[k].dropna() if k in x else pd.Series(dtype=float)
            if len(v):
                p_(f"  {lab:<16} n={len(v):>3} 평균 {v.mean():+.1%} 중앙 {v.median():+.1%} 승률 {(v > 0).mean():.0%} 하위10% {v.quantile(.1):+.1%} 최악 {v.min():+.1%}")
    txt = o.getvalue()
    open(os.path.join(RES, "hatta_component.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__" and sys.argv[1:] == ["component"]:
    component()
