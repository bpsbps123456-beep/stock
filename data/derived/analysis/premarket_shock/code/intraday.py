"""갭 하락 출발 종목의 정규장 분봉 연구 (STX 2026-10-02 같은 '프리장 급락 → 본장 반등').

사건: 정규장 시가가 전일 정규장 종가 대비 -7% 이하, 사건 시점 추정 시총 10억 달러 이상(results/gap_events.csv).
자료: yfinance 분봉. 최근 30일은 1분봉 → 3분봉(사용자가 보는 봉)과 5분봉, 그 앞 기간은 5분봉만 있다(1분봉은 30일).
      규칙은 앞 기간(5분봉)으로 고르고 뒤 기간(5분봉·3분봉)으로 확인한다(AGENTS.md 연구 보고 규칙).
매수(정규장 09:30~11:00 ET = 서머타임 22:30~24:00 KST, 봉 종가로 판단·체결, 사건마다 규칙별 한 번):
  open   시가
  t1000  10:00에 끝나는 봉 종가
  rev2   그날 저가가 시가 아래로 내려간 뒤 봉 종가가 그때까지 저가의 +2% 이상
  vwap   VWAP 아래에서 마감한 봉 다음에 봉 종가가 VWAP 위로 올라섬
  orb15  09:45 이후 봉 종가가 첫 15분 고가를 넘어섬
매도: 매수 다음 봉부터 +TP 지정가(봉 고가로 판정, 당일 마감까지). 못 닿으면 다음 날 정규장 시가(실행 가능). 당일 종가는 참고.
손절 변형: 매수 뒤 봉 종가가 매수 전 그날 저가(시가 매수는 -5%) 아래면 그 종가에 판다. 11:00 ET까지만(그 뒤엔 잠든다).
사용: py intraday.py events → results/gap_events.csv(1시간봉에서 고름), py intraday.py fetch → results/intraday/*.parquet,
      py intraday.py build, py intraday.py report → results/intraday.txt"""
import io, json, os, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../results")
CACHE = os.path.join(RES, "intraday")
RULES = ["open", "t1000", "rev2", "vwap", "orb15"]
RULE_KO = {"open": "시가", "t1000": "10:00", "rev2": "저점+2% 반등", "vwap": "VWAP 회복", "orb15": "15분 고가 돌파"}
TPS = [0.03, 0.05]
WINDOW_END = 11 * 60          # 11:00 ET까지 매수·손절 판단(서머타임 24:00 KST)
OPEN_STOP = 0.05


def events(since="2026-08-03"):
    """1시간봉(../bars)에서 정규장 시가가 전일 종가 -7% 이하이고 사건 시점 시총 10억 달러 이상인 날을 고른다."""
    import glob
    from analyze import sessions, daily, load_universe
    uni = load_universe()
    rows = []
    for f in glob.glob(os.path.join(HERE, "../bars/1h_*.parquet")):
        t = os.path.basename(f)[3:-8]
        if t in ("SPY", "QQQ") or t not in uni:
            continue
        x = pd.read_parquet(f)
        x = x[x.index >= pd.Timestamp(since, tz="America/New_York") - pd.Timedelta(days=7)]
        if x.empty:
            continue
        d = daily(sessions(x))
        if len(d) < 2:
            continue
        gap = d.o / d.c.shift(1) - 1
        shares = uni[t]["cap"] / uni[t]["price"]
        for day, g in gap.items():
            if g <= -0.07 and day >= pd.Timestamp(since) and shares * d.c.shift(1)[day] >= 1e9:
                rows.append((t, day.date(), g, shares * d.c.shift(1)[day]))
    ev = pd.DataFrame(rows, columns=["ticker", "day", "gap", "cap"])
    ev.to_csv(os.path.join(RES, "gap_events.csv"), index=False)
    print(len(ev), "건")


def _fetch(t, day, interval):
    import yfinance as yf
    start = pd.Timestamp(day)
    d = yf.download(t, start=start.strftime("%Y-%m-%d"), end=(start + pd.Timedelta(days=6)).strftime("%Y-%m-%d"),
                    interval=interval, prepost=True, progress=False, auto_adjust=False)
    if d is None or d.empty:
        return None
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d.index = d.index.tz_convert("America/New_York")
    return d[["Open", "High", "Low", "Close", "Volume"]]


def fetch():
    os.makedirs(CACHE, exist_ok=True)
    ev = pd.read_csv(os.path.join(RES, "gap_events.csv"))
    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    jobs = []
    for t, day in zip(ev.ticker, ev.day):
        recent = (today - pd.Timestamp(day)).days <= 28
        for iv in (["1m"] if recent else []) + ["5m"]:
            p = os.path.join(CACHE, f"{iv}_{t}_{day}.parquet")
            if not os.path.exists(p):
                jobs.append((t, day, iv, p))
    print("받을 사건", len(jobs), flush=True)

    def one(job):
        t, day, iv, p = job
        try:
            d = _fetch(t, day, iv)
            if d is not None:
                d.to_parquet(p)
            return 1
        except Exception as e:
            print("실패", t, day, iv, e, flush=True)
            return 0
    with ThreadPoolExecutor(max_workers=6) as pool:
        print("받음", sum(pool.map(one, jobs)), flush=True)


def regular(d, day):
    """그날 정규장 봉과 다음 거래일 정규장 시가."""
    day = pd.Timestamp(day).date()
    m = d.index.hour * 60 + d.index.minute
    reg = d[(m >= 570) & (m < 960)]
    today = reg[reg.index.date == day]
    later = reg[reg.index.date > day]
    nxt = later.Open.iloc[0] if len(later) else np.nan
    return today, nxt


def resample(today, minutes):
    if minutes == 1:
        return today
    return today.resample(f"{minutes}min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last",
                                                "Volume": "sum"}).dropna(subset=["Open"])


def entries(b, minutes):
    """규칙별 (매수 봉 위치, 매수가, 매수 전 저가). 매수 봉 위치 -1 = 시가(첫 봉 전)."""
    o = b.Open.iloc[0]
    mins = b.index.hour * 60 + b.index.minute
    end_min = mins + minutes                       # 봉이 끝나는 시각(분)
    tp = (b.High + b.Low + b.Close) / 3
    vw = (tp * b.Volume).cumsum() / b.Volume.cumsum().replace(0, np.nan)
    run_low = b.Low.cummin()
    orb = b[mins < 585].High.max()                 # 09:30~09:45
    out = {"open": (-1, o, np.nan)}
    for i in range(len(b)):
        if end_min[i] > WINDOW_END:
            break
        c = b.Close.iloc[i]
        if "t1000" not in out and end_min[i] == 600:
            out["t1000"] = (i, c, run_low.iloc[i])
        if "rev2" not in out and run_low.iloc[i] < o and c >= run_low.iloc[i] * 1.02:
            out["rev2"] = (i, c, run_low.iloc[i])
        if "vwap" not in out and i > 0 and b.Close.iloc[i - 1] < vw.iloc[i - 1] and c > vw.iloc[i]:
            out["vwap"] = (i, c, run_low.iloc[i])
        if "orb15" not in out and mins[i] >= 585 and c > orb:
            out["orb15"] = (i, c, run_low.iloc[i])
    return out


def outcome(b, minutes, i0, px, low_before, nxt):
    """TP별 (닿음, 수익) — 손절 없음/손절 있음, 그리고 당일 종가 수익."""
    end_min = b.index.hour * 60 + b.index.minute + minutes
    after = b.iloc[i0 + 1:] if i0 >= 0 else b
    stop = px * (1 - OPEN_STOP) if np.isnan(low_before) else low_before
    close_r = b.Close.iloc[-1] / px - 1
    fallback = nxt / px - 1 if nxt == nxt else close_r
    res = {"close": close_r, "d1o": fallback}
    for tp in TPS:
        target = px * (1 + tp)
        hit = after.High.ge(target)
        res[f"hit{int(tp*100)}"] = bool(hit.any())
        res[f"r{int(tp*100)}"] = tp if hit.any() else fallback
        # 손절: 11:00까지 봉 종가가 stop 아래면 그 종가. 같은 봉에서 지정가가 먼저 닿으면 익절.
        r_stop = None
        for j in range(len(after)):
            if after.High.iloc[j] >= target:
                r_stop = tp
                break
            if end_min[i0 + 1 + j] <= WINDOW_END and after.Close.iloc[j] < stop:
                r_stop = after.Close.iloc[j] / px - 1
                break
        res[f"s{int(tp*100)}"] = fallback if r_stop is None else r_stop
    return res


def build():
    ev = pd.read_csv(os.path.join(RES, "gap_events.csv"))
    ev = ev[ev.gap > -0.6]                                 # CTVA 10/1 -81%는 회사 분할(가격 조정), 급락이 아니다
    rows, shape = [], []
    for t, day, gap in zip(ev.ticker, ev.day, ev.gap):
        p1, p5 = (os.path.join(CACHE, f"{iv}_{t}_{day}.parquet") for iv in ("1m", "5m"))
        if os.path.exists(p1):                             # 최근 30일: 1분봉에서 3분·5분봉을 만든다
            src, path, sizes, step = "1m", p1, (3, 5), 1
        elif os.path.exists(p5):
            src, path, sizes, step = "5m", p5, (5,), 5
        else:
            continue
        today, nxt = regular(pd.read_parquet(path), day)
        if len(today) * step < 60:                         # 정규장 1시간도 안 되는 자료는 버린다
            continue
        for mnt in sizes:
            b = resample(today, mnt if src == "1m" else 1)
            o, lo = b.Open.iloc[0], b.Low.min()
            lo_t = b.Low.idxmin()
            shape.append(dict(ticker=t, day=day, bar=mnt, src=src, gap=gap, low_vs_open=lo / o - 1,
                              low_min=(lo_t.hour * 60 + lo_t.minute) - 570, close_vs_open=b.Close.iloc[-1] / o - 1,
                              close_vs_low=b.Close.iloc[-1] / lo - 1,
                              high_after_low=b[b.index >= lo_t].High.max() / lo - 1))
            for rule, (i0, px, lowb) in entries(b, mnt).items():
                r = outcome(b, mnt, i0, px, lowb, nxt)
                when = "09:30" if i0 < 0 else (b.index[i0] + pd.Timedelta(minutes=mnt)).strftime("%H:%M")
                rows.append(dict(ticker=t, day=day, bar=mnt, src=src, gap=gap, rule=rule, at=when,
                                 entry_vs_open=px / o - 1, **r))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "intraday_trades.csv"), index=False)
    pd.DataFrame(shape).to_csv(os.path.join(RES, "intraday_shape.csv"), index=False)
    print(len(rows), "매수,", len(shape), "사건-봉")


if __name__ == "__main__":
    if sys.argv[1:] == ["events"]:
        events()
    elif sys.argv[1:] == ["fetch"]:
        fetch()
    elif sys.argv[1:] == ["build"]:
        build()


def stats(r, days):
    r = r.dropna()
    if len(r) < 3:
        return None
    g = r.groupby(days.loc[r.index]).mean()
    t = g.mean() / (g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 2 and g.std() > 0 else np.nan
    return dict(n=len(r), 일=len(g), 평균=r.mean(), 중앙=r.median(), 승률=(r > 0).mean(), 하위10=r.quantile(0.1), t=t)


def fmt(c, hit=None):
    if c is None:
        return "(3건 미만)"
    h = f"닿음 {hit:.0%}  " if hit is not None else ""
    return (f"n={c['n']:>3}({c['일']:>2}일) {h}평균 {c['평균']:+5.1%} 중앙 {c['중앙']:+5.1%} 승률 {c['승률']:.0%} "
            f"하위10% {c['하위10']:+6.1%} t(일) {c['t']:+.1f}")


def report():
    tr = pd.read_csv(os.path.join(RES, "intraday_trades.csv"))
    sh = pd.read_csv(os.path.join(RES, "intraday_shape.csv"))
    earn_p = os.path.join(RES, "earnings_dates.json")
    ed = json.load(open(earn_p, encoding="utf-8")) if os.path.exists(earn_p) else {}
    sys.path.insert(0, HERE)
    from summarize import flag_earnings
    for df in (tr, sh):
        df["dayts"] = pd.to_datetime(df.day)
        df["earn"] = flag_earnings(df.assign(day=df.dayts), ed)
        df["win"] = np.where(df.src == "5m", "앞(8/10~9/5)", "뒤(9/8~10/6)")
    o = io.StringIO()
    p = lambda *a: print(*a, file=o)
    p("갭 하락(시가 -7% 이하, 시총 10억 달러 이상) 사건의 정규장 분봉. 앞 기간은 5분봉만, 뒤 기간은 1분봉에서 3·5분봉을 만든다.")
    p("\n[그날 저점은 언제 나오나]")
    for (win, bar), g in sh.groupby(["win", "bar"]):
        lm = g.low_min
        p(f"  {win} {bar}분봉 n={len(g)}: 첫 15분 {(lm < 15).mean():.0%} · 첫 30분 {(lm < 30).mean():.0%} · 11시 전 {(lm < 90).mean():.0%} · "
          f"13시 이후 {(lm >= 210).mean():.0%} | 시가→저점 중앙 {g.low_vs_open.median():+.1%} · 저점→그 뒤 고점 중앙 "
          f"{g.high_after_low.median():+.1%} · 시가→종가 중앙 {g.close_vs_open.median():+.1%} (실적 {g.earn.mean():.0%})")
    groups = {"전체": tr.earn | ~tr.earn, "실적 외": ~tr.earn, "실적 발표": tr.earn}
    for gname, gm in groups.items():
        p(f"\n==== {gname}: 매수 규칙 × 익절 (못 닿으면 다음 날 시가) ====")
        for rule in RULES:
            for tp in ("3", "5"):
                for win, bar in [("앞(8/10~9/5)", 5), ("뒤(9/8~10/6)", 5), ("뒤(9/8~10/6)", 3)]:
                    x = tr[gm & (tr.rule == rule) & (tr.win == win) & (tr.bar == bar)]
                    if not len(x):
                        continue
                    a = stats(x[f"r{tp}"], x.dayts)
                    s = stats(x[f"s{tp}"], x.dayts)
                    p(f"  {RULE_KO[rule]:<9} +{tp}% {win} {bar}분 | 손절 없음 {fmt(a, x[f'hit{tp}'].mean())}")
                    p(f"  {'':<9}     {'':<13}    | 11시까지 손절 {fmt(s)}")
    # 앞 기간으로 고르고 뒤 기간으로 확인(5분봉끼리, 전체 사건)
    cand = []
    for rule in RULES:
        for tp in ("3", "5"):
            for kind in ("r", "s"):
                x = tr[(tr.rule == rule) & (tr.bar == 5)]
                a = stats(x[x.win == "앞(8/10~9/5)"][f"{kind}{tp}"], x.dayts)
                b = stats(x[x.win == "뒤(9/8~10/6)"][f"{kind}{tp}"], x.dayts)
                y = tr[(tr.rule == rule) & (tr.bar == 3)]
                c = stats(y[f"{kind}{tp}"], y.dayts)
                if a:
                    cand.append((a["평균"], rule, tp, kind, a, b, c))
    p(f"\n[앞 기간 평균 상위 5개(후보 {len(cand)}개: 규칙 5 × 익절 2 × 손절 유무 2) → 뒤 기간 확인]")
    for m, rule, tp, kind, a, b, c in sorted(cand, key=lambda z: -z[0])[:5]:
        p(f"  {RULE_KO[rule]} +{tp}% {'손절' if kind == 's' else '손절 없음'}\n    앞 5분 {fmt(a)}\n    뒤 5분 {fmt(b)}\n    뒤 3분 {fmt(c)}")
    p("\n시험 범위: 매수 규칙 5개 × 익절 2개(+3·5%) × 손절 유무 2개 × 사건 묶음 3개 × 기간·봉 3개. 매수가는 봉 종가, 스프레드 미반영.")
    txt = o.getvalue()
    open(os.path.join(RES, "intraday.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__" and sys.argv[1:] == ["report"]:
    report()
