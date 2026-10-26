"""급락일 '마지막 하락 지점' 찾기 — 기울기가 평평해지면 사는 규칙을 앞 기간에서 고르고 뒤 기간에서 확인.

사건: 정규장에서 한 번이라도 전일 종가 대비 -7% 이하, 시총 10억 달러 이상. 앞 2026-08-10~09-05, 뒤 09-08~10-06.
      (처음에 '장중 저가 -10% 이하'로 골랐더니 끝까지 빠진 날만 남아 규칙에 불리하게 치우쳤다 — 매수 조건과 같은 문턱으로 고친다.)
자료: yfinance 5분봉(60일, 정규장, 전후 며칠 포함), 뒤 기간은 1분봉에서 만든 3분봉도 함께(사용자가 보는 봉).
규칙은 모두 봉 마감 때 아는 값만 쓴다. 창 길이는 분 단위로 정해 5분봉·3분봉에 같게 적용한다.
  NL(N, d)   그날 저가가 N분 동안 새로 나오지 않았고, 종가가 그 저가의 +d% 안(아직 바닥 근처)일 때 산다.
  SLOPE(f)   최근 30분 종가 회귀 기울기가 30분당 f% 이상으로 평평해졌고, 앞 60분 안에 30분당 -3% 이하로 가팔랐으며,
             15분 동안 새 저가가 없고 종가가 저가 +3% 안일 때 산다.
  HL(r)      저가에서 +r% 반등했다가 그 저가보다 높은 곳에서 다시 받친 뒤(더 높은 저점), 반등 고점을 넘어서 마감하면 산다.
  공통 조건  매수 순간 전일 종가 대비 -7% 이하, 09:45~15:30 사이, 사건마다 규칙별 첫 신호 한 번.
매도: +TP 지정가(다음 봉부터, 고가로 판정). 손절은 매수 때까지의 그날 저가 -1%(봉 저가가 닿으면, 같은 봉이면 손절 먼저).
  손절 변형: 하루 종일 / 11:00 ET까지만(그 뒤엔 잠든다) / 없음. 못 닿으면 다음 날 시가.
사용: py lowfind.py events → fetch → run → results/lowfind.txt"""
import io, json, os, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "../results")
CACHE5 = os.path.join(RES, "lowfind")
CACHE1 = os.path.join(RES, "hatta")
EARLY = ("2026-08-10", "2026-09-05")
LATE = ("2026-09-08", "2026-10-06")
CRASH = -0.07                 # 매수 순간 전일 종가 대비
STOP_BELOW = 0.01
AWAKE_END = 11 * 60
NL_GRID = [(n, d) for n in (15, 30, 45) for d in (0.02, 0.04)]
SLOPE_GRID = [-0.005, -0.010]
HL_GRID = [0.02, 0.03]
TPS = [0.02, 0.03]


# ---------------------------------------------------------------- 자료
def events():
    import glob
    from analyze import sessions, daily, load_universe
    uni = load_universe()
    rows = []
    for f in glob.glob(os.path.join(HERE, "../bars/1h_*.parquet")):
        t = os.path.basename(f)[3:-8]
        if t in ("SPY", "QQQ") or t not in uni:
            continue
        x = pd.read_parquet(f)
        x = x[x.index >= pd.Timestamp("2026-07-30", tz="America/New_York")]
        if x.empty:
            continue
        d = daily(sessions(x))
        c0 = d.c.shift(1)
        shares = uni[t]["cap"] / uni[t]["price"]
        for day in d.index:
            if not (pd.Timestamp(EARLY[0]) <= day <= pd.Timestamp(LATE[1])) or not c0[day] > 0:
                continue
            gap, low = d.o[day] / c0[day] - 1, d.l[day] / c0[day] - 1
            if low <= -0.07 and gap > -0.6 and shares * c0[day] >= 1e9:   # 매수 조건과 같은 문턱(결과로 고르지 않는다)
                win = "앞" if day <= pd.Timestamp(EARLY[1]) else "뒤"
                rows.append((t, day.date(), win, gap, low, shares * c0[day]))
    ev = pd.DataFrame(rows, columns=["ticker", "day", "win", "gap", "low", "cap"])
    ev.to_csv(os.path.join(RES, "lowfind_events.csv"), index=False)
    print(ev.groupby("win").size().to_dict())


def fetch():
    import yfinance as yf
    os.makedirs(CACHE5, exist_ok=True)
    ev = pd.read_csv(os.path.join(RES, "lowfind_events.csv"))
    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    jobs = [(t, d) for t, d in zip(ev.ticker, ev.day) if not os.path.exists(os.path.join(CACHE5, f"5m_{t}_{d}.parquet"))]

    def one(job):
        t, d = job
        day = pd.Timestamp(d)
        start = max(day - pd.Timedelta(days=5), today - pd.Timedelta(days=58))
        end = min(day + pd.Timedelta(days=5), today + pd.Timedelta(days=1))
        try:
            x = yf.download(t, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval="5m",
                            prepost=False, progress=False, auto_adjust=False)
            if isinstance(x.columns, pd.MultiIndex):
                x.columns = x.columns.get_level_values(0)
            if len(x):
                x.index = x.index.tz_convert("America/New_York")
                x[["Open", "High", "Low", "Close", "Volume"]].to_parquet(os.path.join(CACHE5, f"5m_{t}_{d}.parquet"))
            return 1
        except Exception as e:
            print("실패", t, d, e)
            return 0
    print("받을 사건", len(jobs))
    with ThreadPoolExecutor(max_workers=6) as pool:
        print("받음", sum(pool.map(one, jobs)))


def fetch1m():
    """뒤 기간 사건의 1분봉(3분봉용): 사건일 앞뒤 며칠, 최근 30일 한도 안."""
    import yfinance as yf
    os.makedirs(CACHE1, exist_ok=True)
    ev = pd.read_csv(os.path.join(RES, "lowfind_events.csv"))
    ev = ev[ev.win == "뒤"]
    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    jobs = [(t, d) for t, d in zip(ev.ticker, ev.day) if not os.path.exists(os.path.join(CACHE1, f"1m_{t}_{d}.parquet"))]

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
                x[["Open", "High", "Low", "Close", "Volume"]].to_parquet(os.path.join(CACHE1, f"1m_{t}_{d}.parquet"))
            return 1
        except Exception as e:
            print("실패", t, d, e)
            return 0
    print("받을 사건", len(jobs))
    with ThreadPoolExecutor(max_workers=6) as pool:
        print("받음", sum(pool.map(one, jobs)))


def _resample(x, minutes):
    return x.resample(f"{minutes}min", origin="start_day").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(subset=["Open"])


def regular_bars(t, d, minutes, with_prev=False, with_next=False):
    """그날 정규장 봉(minutes분), 전일 종가, 다음 날 시가. with_prev면 전날 정규장 봉(supports.py),
    with_next면 다음 날 정규장 봉(nightbid.py)을 뒤에 붙인다."""
    day = pd.Timestamp(d).date()
    if minutes == 5:
        p = os.path.join(CACHE5, f"5m_{t}_{d}.parquet")
        if not os.path.exists(p):
            return None
        x = pd.read_parquet(p)
    else:
        p = os.path.join(CACHE1, f"1m_{t}_{d}.parquet")
        if not os.path.exists(p):
            return None
        x = pd.read_parquet(p)
    mm = x.index.hour * 60 + x.index.minute
    x = x[(mm >= 570) & (mm < 960)]
    dates = np.asarray(x.index.date)
    prev = x[dates < day]
    today = x[dates == day]
    nxt = x[dates > day]
    if prev.empty or len(today) < 6:
        return None
    if minutes != 5:
        today = _resample(today, minutes)
    got = today, float(prev.Close.iloc[-1]), (float(nxt.Open.iloc[0]) if len(nxt) else np.nan)
    rs = (lambda z: _resample(z, minutes)) if minutes != 5 else (lambda z: z)
    if with_prev:
        got += (rs(prev[np.asarray(prev.index.date) == prev.index[-1].date()]),)
    if with_next:
        got += (rs(nxt[np.asarray(nxt.index.date) == nxt.index[0].date()]) if len(nxt) else nxt,)
    return got


# ---------------------------------------------------------------- 마지막 하락 지점(묘사)
def shape(b, c0):
    mins = (b.index.hour * 60 + b.index.minute - 570).to_numpy()
    lo_i = int(np.argmin(b.Low.to_numpy()))
    low = b.Low.iloc[lo_i]
    out = {"low_min": int(mins[lo_i]), "low_vs_c0": low / c0 - 1, "low_vs_open": low / b.Open.iloc[0] - 1,
           "close_vs_low": b.Close.iloc[-1] / low - 1, "high_after_low": b.High.iloc[lo_i:].max() / low - 1}
    for w in (15, 30, 60):
        first = b[mins < w]
        if len(first):
            fl = first.Low.min()
            out[f"held{w}"] = low >= fl * 0.9999            # 첫 w분 저가가 그날 저가로 남았나
            out[f"under{w}"] = low / fl - 1                 # 깨졌으면 그 아래로 얼마나 더
    return out


# ---------------------------------------------------------------- 규칙
def signals(b, c0, minutes):
    """규칙별 첫 신호: {규칙: (봉 위치, 매수가, 그때까지 저가)}."""
    mins = (b.index.hour * 60 + b.index.minute).to_numpy()
    end = mins + minutes
    lo, hi, cl = b.Low.to_numpy(), b.High.to_numpy(), b.Close.to_numpy()
    run_low = np.minimum.accumulate(lo)
    low_i = np.zeros(len(b), int)                           # 그때까지 저가가 나온 봉 위치
    for i in range(1, len(b)):
        low_i[i] = i if lo[i] <= run_low[i - 1] else low_i[i - 1]
    w30 = max(2, 30 // minutes)
    slope = np.full(len(b), np.nan)                         # 최근 30분 회귀 기울기, 30분당 비율
    xs = np.arange(w30)
    for i in range(w30 - 1, len(b)):
        y = cl[i - w30 + 1:i + 1]
        k = np.polyfit(xs, y, 1)[0]
        slope[i] = k * (w30 - 1) / y.mean()
    out = {}
    hl_state = {r: {"peak": None, "dip": None} for r in HL_GRID}
    for i in range(len(b)):
        if end[i] < 585 or end[i] > 930:                    # 09:45~15:30에 마감한 봉만
            continue
        crash = cl[i] / c0 - 1 <= CRASH
        since_low = end[i] - (mins[low_i[i]] + minutes)
        near = cl[i] / run_low[i] - 1
        for n, d in NL_GRID:
            k = f"NL{n}_{int(d * 100)}"
            if k not in out and crash and since_low >= n and near <= d:
                out[k] = (i, cl[i], run_low[i])
        steep_before = np.nanmin(slope[max(0, i - 60 // minutes):i + 1]) if i >= w30 else np.nan
        for f in SLOPE_GRID:
            k = f"SLOPE{abs(f) * 100:.1f}"
            if (k not in out and crash and slope[i] >= f and steep_before <= -0.03 and since_low >= 15
                    and near <= 0.03):
                out[k] = (i, cl[i], run_low[i])
    # 더 높은 저점 뒤 반등 고점 돌파
    for r in HL_GRID:
        k = f"HL{int(r * 100)}"
        peak = dip = None
        base_i = 0
        for i in range(len(b)):
            if lo[i] <= run_low[i] and (i == 0 or lo[i] < run_low[i - 1]):
                peak = dip = None                           # 새 저가: 처음부터
                base_i = i
                continue
            if peak is None:
                if hi[i] >= run_low[i] * (1 + r):
                    peak = hi[i]
                continue
            peak = max(peak, hi[i]) if dip is None else peak
            if dip is None and cl[i] < peak * (1 - r / 2):
                dip = lo[i]                                 # 반등 뒤 되밀림(저가 위에서)
            elif dip is not None:
                dip = min(dip, lo[i])
                if (cl[i] > peak and dip > run_low[i] and 585 <= end[i] <= 930 and cl[i] / c0 - 1 <= CRASH
                        and k not in out):
                    out[k] = (i, cl[i], run_low[i])
                    break
    return out


def outcome(b, minutes, i0, px, low_before, nxt, tp, stop_mode):
    mins = (b.index.hour * 60 + b.index.minute).to_numpy()
    end = mins + minutes
    stop = low_before * (1 - STOP_BELOW)
    target = px * (1 + tp)
    for j in range(i0 + 1, len(b)):
        can_stop = stop_mode == "all" or (stop_mode == "awake" and end[j] <= AWAKE_END)
        if can_stop and b.Low.iloc[j] <= stop:
            return (min(stop, b.Open.iloc[j]) / px - 1), "손절"
        if b.High.iloc[j] >= target:
            return tp, "익절"
    last = nxt if nxt == nxt else b.Close.iloc[-1]
    return last / px - 1, "다음 날 시가"


def run():
    ev = pd.read_csv(os.path.join(RES, "lowfind_events.csv"))
    shapes, trades = [], []
    for t, d, win, gap in zip(ev.ticker, ev.day, ev.win, ev.gap):
        for minutes in ((5, 3) if win == "뒤" else (5,)):
            got = regular_bars(t, d, minutes)
            if got is None:
                continue
            b, c0, nxt = got
            shapes.append(dict(ticker=t, day=d, win=win, bar=minutes, gap=gap, **shape(b, c0)))
            for rule, (i0, px, lowb) in signals(b, c0, minutes).items():
                at = (b.index[i0] + pd.Timedelta(minutes=minutes))
                for tp in TPS:
                    for sm in ("all", "awake", "none"):
                        r, why = outcome(b, minutes, i0, px, lowb, nxt, tp, sm)
                        trades.append(dict(ticker=t, day=d, win=win, bar=minutes, rule=rule, tp=tp, stop=sm,
                                           at=at.strftime("%H:%M"), awake_entry=at.hour * 60 + at.minute <= AWAKE_END,
                                           dd=px / c0 - 1, ret=r, why=why))
    sh, tr = pd.DataFrame(shapes), pd.DataFrame(trades)
    sh.to_csv(os.path.join(RES, "lowfind_shape.csv"), index=False)
    tr.to_csv(os.path.join(RES, "lowfind_trades.csv"), index=False)
    report(sh, tr)


def stat(x):
    if len(x) < 5:
        return None
    g = x.groupby("day").ret.mean()
    t = g.mean() / (g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 2 and g.std() > 0 else np.nan
    return dict(n=len(x), days=x.day.nunique(), avg=x.ret.mean(), med=x.ret.median(), win=(x.ret > 0).mean(),
                tp=(x.why == "익절").mean(), stop=(x.why == "손절").mean(), p10=x.ret.quantile(0.1), worst=x.ret.min(), t=t)


def fmt(s):
    if s is None:
        return "(5건 미만)"
    return (f"n={s['n']:>3}({s['days']:>2}일) 평균 {s['avg']:+.2%} 중앙 {s['med']:+.1%} 승률 {s['win']:.0%} "
            f"익절 {s['tp']:.0%} 손절 {s['stop']:.0%} 하위10% {s['p10']:+.1%} 최악 {s['worst']:+.1%} t(일) {s['t']:+.1f}")


def report(sh, tr):
    o = io.StringIO()
    p = lambda *a: print(*a, file=o)
    p("[마지막 하락 지점 — 급락일 정규장]")
    for (win, bar), g in sh.groupby(["win", "bar"]):
        p(f"  {win} {bar}분봉 n={len(g)}: 저점 시각 첫 15분 {(g.low_min < 15).mean():.0%} · 30분 {(g.low_min < 30).mean():.0%} · "
          f"60분 {(g.low_min < 60).mean():.0%} · 11시 전 {(g.low_min < 90).mean():.0%} · 13시 이후 {(g.low_min >= 210).mean():.0%}")
        for w in (15, 30, 60):
            held = g[f"held{w}"].astype("boolean").fillna(True).astype(bool)
            broke = g[~held]
            p(f"     첫 {w}분 저가가 그날 저가로 남음 {held.mean():.0%} · 깨진 날 더 빠진 폭 중앙 {broke[f'under{w}'].median():+.1%} "
              f"(하위 25% {broke[f'under{w}'].quantile(.25):+.1%}, 하위 10% {broke[f'under{w}'].quantile(.1):+.1%})")
        p(f"     저점→그 뒤 고점 중앙 {g.high_after_low.median():+.1%} · 저점→종가 중앙 {g.close_vs_low.median():+.1%} · "
          f"시가→저점 중앙 {g.low_vs_open.median():+.1%}")
    p("\n[규칙 × 익절 × 손절 — 앞(5분) / 뒤(5분) / 뒤(3분)]")
    rows = []
    for (rule, tp, sm), g in tr.groupby(["rule", "tp", "stop"]):
        a = stat(g[(g.win == "앞") & (g.bar == 5)])
        b5 = stat(g[(g.win == "뒤") & (g.bar == 5)])
        b3 = stat(g[(g.win == "뒤") & (g.bar == 3)])
        rows.append((rule, tp, sm, a, b5, b3))
        p(f"  {rule:<8} +{tp:.0%} 손절:{sm:<5} 앞5 {fmt(a)}")
        p(f"  {'':<8}       {'':<9} 뒤5 {fmt(b5)}")
        p(f"  {'':<8}       {'':<9} 뒤3 {fmt(b3)}")
    cand = [r for r in rows if r[3] and r[3]["n"] >= 30]
    p(f"\n[앞 기간 평균 상위 8개(후보 {len(cand)}개, 앞 n ≥ 30) → 뒤 기간]")
    for rule, tp, sm, a, b5, b3 in sorted(cand, key=lambda z: -z[3]["avg"])[:8]:
        p(f"  {rule} +{tp:.0%} 손절:{sm}\n    앞5 {fmt(a)}\n    뒤5 {fmt(b5)}\n    뒤3 {fmt(b3)}")
    p(f"\n시험 범위: 매수 규칙 {len(NL_GRID) + len(SLOPE_GRID) + len(HL_GRID)}개(NL 6·SLOPE 2·HL 2) × 익절 {len(TPS)}개 × 손절 3개 = "
      f"{(len(NL_GRID) + len(SLOPE_GRID) + len(HL_GRID)) * len(TPS) * 3}개. 앞 기간으로 고르고 뒤 기간 두 봉으로 확인.")
    txt = o.getvalue()
    open(os.path.join(RES, "lowfind.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


def report_only():
    report(pd.read_csv(os.path.join(RES, "lowfind_shape.csv")), pd.read_csv(os.path.join(RES, "lowfind_trades.csv")))
    cpp, ntp = os.path.join(RES, "lowfind_checkpoints.csv"), os.path.join(RES, "lowfind_night.csv")
    if os.path.exists(cpp) and os.path.exists(ntp):
        extra = report2(pd.read_csv(cpp), pd.read_csv(ntp))
        with open(os.path.join(RES, "lowfind.txt"), "a", encoding="utf-8") as fh:
            fh.write(extra)
        print(extra)




# ---------------------------------------------------------------- 바닥 확률표·자기 전 지정가
CHECK_TIMES = set(range(600, 931, 15))        # 10:00~15:30, 15분마다(봉이 끝나는 시각)
SINCE_BUCKETS = [(0, 15), (15, 30), (30, 60), (60, 120), (120, 999)]
NIGHT_K = [0.0, 0.03, 0.05, 0.08]             # 11:00 ET에 그때까지 저가의 -k%에 매수 지정가


def _lows(b):
    lo = b.Low.to_numpy()
    run_low = np.minimum.accumulate(lo)
    low_i = np.zeros(len(b), int)
    for i in range(1, len(b)):
        low_i[i] = i if lo[i] <= run_low[i - 1] else low_i[i - 1]
    return run_low, low_i


def checkpoints(b, c0, minutes):
    """15분마다(그 순간 -7% 이하일 때) 지금까지 저가가 그날 끝까지 버티나, 깨지면 얼마나 더 빠지나."""
    mins = (b.index.hour * 60 + b.index.minute).to_numpy()
    end = mins + minutes
    lo, hi, cl = b.Low.to_numpy(), b.High.to_numpy(), b.Close.to_numpy()
    run_low, low_i = _lows(b)
    out = []
    for i in range(len(b) - 1):
        if int(end[i]) not in CHECK_TIMES or cl[i] / c0 - 1 > CRASH:
            continue
        rl = run_low[i]
        later = lo[i + 1:].min()
        out.append(dict(t=int(end[i]), since=int(end[i] - (mins[low_i[i]] + minutes)), near=cl[i] / rl - 1,
                        dd=cl[i] / c0 - 1, held=bool(later >= rl), further=min(later, rl) / rl - 1,
                        up_after=hi[i + 1:].max() / cl[i] - 1, to_close=cl[-1] / cl[i] - 1))
    return out


def night(b, c0, nxt, minutes):
    """11:00 ET(서머타임 24:00 KST)에 -7% 이하면, 그때까지 저가의 -k%에 매수 지정가를 걸어 두고 잔다.
    체결은 그 값보다 0.1% 더 내려간 봉(엄격), 봉 시가가 이미 아래면 시가. 다음 날 아침(한국 08시 = 뉴욕 19시)에
    판다고 보고 당일 종가(장후 가격 대용)와 다음 날 시가를 함께 잰다."""
    mins = (b.index.hour * 60 + b.index.minute).to_numpy()
    end = mins + minutes
    idx = np.nonzero(end == 660)[0]
    if not len(idx):
        return []
    i11 = int(idx[0])
    cl = b.Close.to_numpy()
    if cl[i11] / c0 - 1 > CRASH:
        return []
    run_low, _ = _lows(b)
    rl = run_low[i11]
    later = b.iloc[i11 + 1:]
    out = []
    for k in NIGHT_K:
        level = rl * (1 - k)
        hit = later[later.Low <= level * 0.999]
        if hit.empty:
            out.append(dict(k=k, filled=False))
            continue
        px = min(level, hit.Open.iloc[0])
        out.append(dict(k=k, filled=True, fill_at=(hit.index[0] + pd.Timedelta(minutes=minutes)).strftime("%H:%M"),
                        ret_close=cl[-1] / px - 1, ret_next=(nxt / px - 1) if nxt == nxt else cl[-1] / px - 1,
                        dd11=cl[i11] / c0 - 1))
    return out


def run2():
    ev = pd.read_csv(os.path.join(RES, "lowfind_events.csv"))
    shapes, trades, cps, nights = [], [], [], []
    for t, d, win, gap in zip(ev.ticker, ev.day, ev.win, ev.gap):
        for minutes in ((5, 3) if win == "뒤" else (5,)):
            got = regular_bars(t, d, minutes)
            if got is None:
                continue
            b, c0, nxt = got
            key = dict(ticker=t, day=d, win=win, bar=minutes)
            shapes.append(dict(**key, gap=gap, **shape(b, c0)))
            cps += [dict(**key, **c) for c in checkpoints(b, c0, minutes)]
            nights += [dict(**key, **n) for n in night(b, c0, nxt, minutes)]
            for rule, (i0, px, lowb) in signals(b, c0, minutes).items():
                at = b.index[i0] + pd.Timedelta(minutes=minutes)
                for tp in TPS:
                    for sm in ("all", "awake", "none"):
                        r, why = outcome(b, minutes, i0, px, lowb, nxt, tp, sm)
                        trades.append(dict(**key, rule=rule, tp=tp, stop=sm, at=at.strftime("%H:%M"),
                                           awake_entry=at.hour * 60 + at.minute <= AWAKE_END, dd=px / c0 - 1, ret=r, why=why))
    for name, rows in (("shape", shapes), ("trades", trades), ("checkpoints", cps), ("night", nights)):
        pd.DataFrame(rows).to_csv(os.path.join(RES, f"lowfind_{name}.csv"), index=False)
    report_only()


def report2(cp, nt):
    o = io.StringIO()
    p = lambda *a: print(*a, file=o)
    p("\n[바닥 확률표 — 15분마다 그 순간 -7% 이하인 종목: 지금까지 저가가 그날 끝까지 버틴 비율]")
    p("  저가 뒤 지난 시간 | 기간·봉 | 버팀 | 깨진 날 더 빠진 폭 중앙(하위 10%) | 그 뒤 최고 중앙 | 종가까지 중앙 | 종목 수")
    for lo_, hi_ in SINCE_BUCKETS:
        for (win, bar), g in cp.groupby(["win", "bar"]):
            x = g[(g.since >= lo_) & (g.since < hi_)].drop_duplicates(["ticker", "day"])
            if len(x) < 10:
                continue
            br = x[~x.held]
            lab = f"{lo_}~{hi_ - 1}분" if hi_ < 999 else f"{lo_}분 이상"
            p(f"  {lab:<10} {win}{bar}분 | {x.held.mean():.0%} | {br.further.median():+.1%}({br.further.quantile(.1):+.1%}) | "
              f"{x.up_after.median():+.1%} | {x.to_close.median():+.1%} | {len(x)}")
    p("\n[같은 표를 시각으로 나눔 — 저가 뒤 30분 이상 지났을 때]")
    for (win, bar), g in cp.groupby(["win", "bar"]):
        g = g[g.since >= 30]
        for lab, a, b_ in (("10:00~11:00", 600, 660), ("11:15~13:00", 675, 780), ("13:15~15:30", 795, 930)):
            x = g[(g.t >= a) & (g.t <= b_)].drop_duplicates(["ticker", "day"])
            if len(x) >= 10:
                p(f"  {win}{bar}분 {lab}: 버팀 {x.held.mean():.0%} · 깨진 날 더 빠짐 중앙 {x[~x.held].further.median():+.1%} · 종가까지 중앙 {x.to_close.median():+.1%} · {len(x)}종목")
    p("\n[자기 전 지정가 — 11:00 ET에 -7% 이하면 그때까지 저가의 -k%에 매수, 다음 날 아침 매도]")
    for (win, bar, k), g in nt.groupby(["win", "bar", "k"]):
        f = g[g.filled == True]
        if len(g) < 10:
            continue
        line = f"  {win}{bar}분 k={k:.0%}: 대상 {len(g)} · 체결 {len(f) / len(g):.0%}"
        if len(f) >= 5:
            line += (f" · 당일 종가 평균 {f.ret_close.mean():+.2%}(승률 {(f.ret_close > 0).mean():.0%}, 하위10% {f.ret_close.quantile(.1):+.1%})"
                     f" · 다음 날 시가 평균 {f.ret_next.mean():+.2%}(승률 {(f.ret_next > 0).mean():.0%}, 하위10% {f.ret_next.quantile(.1):+.1%})")
        p(line)
    return o.getvalue()


if __name__ == "__main__":
    if sys.argv[1] not in ("filters", "awake", "waiting", "bounce"):
        {"events": events, "fetch": fetch, "fetch1m": fetch1m, "run": run2, "report": report_only}[sys.argv[1]]()


def filters():
    """장세 필터(최근 N거래일 같은 규칙 성과가 플러스일 때만)와 시장 필터(신호 순간 SPY 등락)를 대표 규칙에 씌운다.
    대표 규칙은 앞 기간 상위였던 NL15_2·NL30_2(+2%, 손절 없음), 5분봉. 장세 필터는 그날 전까지 끝난 거래만 쓴다."""
    tr = pd.read_csv(os.path.join(RES, "lowfind_trades.csv"), parse_dates=["day"])
    spy = pd.read_parquet(os.path.join(RES, "lowfind", "spy_5m.parquet"))
    sd = spy.groupby(spy.index.date).Close.last()
    prev_close = sd.shift(1)
    o = io.StringIO()
    p = lambda *a: print(*a, file=o)

    def spy_chg(day, at):
        hh, mm = map(int, at.split(":"))
        ts = pd.Timestamp(day).tz_localize("America/New_York") + pd.Timedelta(hours=hh, minutes=mm)
        bars = spy[(spy.index.date == day.date()) & (spy.index + pd.Timedelta(minutes=5) <= ts)]
        pc = prev_close.get(day.date())
        return np.nan if bars.empty or not pc == pc else bars.Close.iloc[-1] / pc - 1

    for rule in ("NL15_2", "NL30_2"):
        x = tr[(tr.rule == rule) & (tr.tp == 0.02) & (tr.stop == "none") & (tr.bar == 5)].copy()
        x["spy"] = [spy_chg(d, a) for d, a in zip(x.day, x["at"])]
        daily_mean = x.groupby("day").ret.mean().sort_index()
        days = list(daily_mean.index)
        p(f"\n==== {rule} +2% 손절 없음 5분봉: 전체 n={len(x)} 평균 {x.ret.mean():+.2%} ====")
        for n in (5, 10, 20):
            keep = {}
            for i, d in enumerate(days):
                past = daily_mean.iloc[max(0, i - n):i]
                keep[d] = len(past) >= min(n, 3) and past.mean() > 0
            on = x[x.day.map(keep)]
            off = x[~x.day.map(keep)]
            for win in ("앞", "뒤"):
                a, b_ = on[on.win == win], off[off.win == win]
                p(f"  장세 필터 {n}거래일 {win}: 할 때 n={len(a)} 평균 {a.ret.mean():+.2%} 승률 {(a.ret > 0).mean():.0%} "
                  f"({a.day.nunique()}일) | 쉴 때 n={len(b_)} 평균 {b_.ret.mean():+.2%} ({b_.day.nunique()}일)")
        for lab, m in (("SPY ≥ 0", x.spy >= 0), ("SPY -0.5~0", (x.spy < 0) & (x.spy >= -0.005)), ("SPY < -0.5%", x.spy < -0.005)):
            for win in ("앞", "뒤"):
                a = x[m & (x.win == win)]
                if len(a):
                    p(f"  시장 {lab:<10} {win}: n={len(a)} 평균 {a.ret.mean():+.2%} 승률 {(a.ret > 0).mean():.0%} ({a.day.nunique()}일)")
        for win in ("앞", "뒤"):
            a = x[x.awake_entry & (x.win == win)]
            p(f"  11시 전 신호만 {win}: n={len(a)} 평균 {a.ret.mean():+.2%} 승률 {(a.ret > 0).mean():.0%}")
        # 날짜별 성과 흐름(장세가 언제 바뀌었나)
        wk = x.groupby(x.day.dt.to_period("W")).ret.agg(["mean", "size"])
        p("  주별 평균: " + " · ".join(f"{str(k.start_time.date())[5:]} {v['mean']:+.1%}({int(v['size'])})" for k, v in wk.iterrows()))
    txt = o.getvalue()
    open(os.path.join(RES, "lowfind_filters.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__" and sys.argv[1:] == ["filters"]:
    filters()


def awake():
    """사용자가 깨어 있는 11:00 ET 전 신호만(서머타임 24:00 KST) — 모든 규칙·익절·손절과 두 기간·두 봉."""
    tr = pd.read_csv(os.path.join(RES, "lowfind_trades.csv"))
    tr = tr[tr.awake_entry]
    o = io.StringIO()
    p = lambda *a: print(*a, file=o)
    p("[11:00 ET 전 신호만 — 규칙 × 익절 × 손절: 앞5 / 뒤5 / 뒤3]")
    both = 0
    total = 0
    for (rule, tp, sm), g in tr.groupby(["rule", "tp", "stop"]):
        a = stat(g[(g.win == "앞") & (g.bar == 5)])
        b5 = stat(g[(g.win == "뒤") & (g.bar == 5)])
        b3 = stat(g[(g.win == "뒤") & (g.bar == 3)])
        total += 1
        ok = a and b5 and b3 and a["avg"] > 0 and b5["avg"] > 0 and b3["avg"] > 0
        both += bool(ok)
        p(f"  {'★' if ok else ' '} {rule:<8} +{tp:.0%} 손절:{sm:<5} 앞5 {fmt(a)}")
        p(f"    {'':<8}       {'':<9} 뒤5 {fmt(b5)}")
        p(f"    {'':<8}       {'':<9} 뒤3 {fmt(b3)}")
    p(f"\n세 칸(앞5·뒤5·뒤3) 모두 평균 플러스: {both}/{total}개 조합(★)")
    txt = o.getvalue()
    open(os.path.join(RES, "lowfind_awake.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__" and sys.argv[1:] == ["awake"]:
    awake()


# ---------------------------------------------------------------- 기다린 비용: 버틴 시간 × 저가에서 떨어진 정도
NEAR_BUCKETS = [(-1, 0.01), (0.01, 0.02), (0.02, 0.04), (0.04, 0.07), (0.07, 9)]
SINCE_BUCKETS2 = [(15, 30), (30, 60), (60, 120), (120, 999)]


def wait_points(b, c0, minutes):
    """15분마다(그 순간 -7% 이하) 사면: 지금 저가 대비 위치, 저가 뒤 지금까지의 최고 반등, 그리고
    +2% 지정가와 '그때까지 저가 -1%' 손절 중 먼저 닿은 쪽(같은 봉이면 손절), 둘 다 없으면 당일 종가."""
    mins = (b.index.hour * 60 + b.index.minute).to_numpy()
    end = mins + minutes
    lo, hi, cl = b.Low.to_numpy(), b.High.to_numpy(), b.Close.to_numpy()
    run_low, low_i = _lows(b)
    out = []
    for i in range(len(b) - 1):
        if int(end[i]) not in CHECK_TIMES or cl[i] / c0 - 1 > CRASH:
            continue
        rl = run_low[i]
        peak = hi[low_i[i]:i + 1].max() / rl - 1            # 저가 뒤 지금까지 가장 많이 튄 폭
        stop, target = rl * (1 - STOP_BELOW), cl[i] * 1.02
        res, why = cl[-1] / cl[i] - 1, "종가"
        for j in range(i + 1, len(b)):
            if lo[j] <= stop:
                res, why = min(stop, b.Open.iloc[j]) / cl[i] - 1, "손절"
                break
            if hi[j] >= target:
                res, why = 0.02, "익절"
                break
        out.append(dict(t=int(end[i]), since=int(end[i] - (mins[low_i[i]] + minutes)), near=cl[i] / rl - 1,
                        peak=peak, held=bool(lo[i + 1:].min() >= rl), up_after=hi[i + 1:].max() / cl[i] - 1,
                        ret=res, why=why))
    return out


def waiting():
    ev = pd.read_csv(os.path.join(RES, "lowfind_events.csv"))
    rows = []
    for t, d, win in zip(ev.ticker, ev.day, ev.win):
        for minutes in ((5, 3) if win == "뒤" else (5,)):
            got = regular_bars(t, d, minutes)
            if got is None:
                continue
            b, c0, _ = got
            rows += [dict(ticker=t, day=d, win=win, bar=minutes, **w) for w in wait_points(b, c0, minutes)]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "lowfind_wait.csv"), index=False)
    o = io.StringIO()
    p = lambda *a: print(*a, file=o)
    lab_s = lambda a, b_: f"{a}~{b_ - 1}분" if b_ < 999 else f"{a}분 이상"
    lab_n = lambda a, b_: "저가 +1% 안" if a < 0 else (f"+{a:.0%}~{b_:.0%}" if b_ < 9 else f"+{a:.0%} 이상")
    for (win, bar), g in df.groupby(["win", "bar"]):
        p(f"\n==== {win} {bar}분봉 ====")
        p("[기다린 비용: 저가가 버틴 시간별, 지금 가격이 저가보다 얼마나 위인가(중앙)와 그 분포]")
        for a, b_ in SINCE_BUCKETS2:
            x = g[(g.since >= a) & (g.since < b_)].drop_duplicates(["ticker", "day"])
            if len(x) >= 10:
                p(f"  {lab_s(a, b_):<9} 저가 대비 중앙 {x.near.median():+.1%} · 저가 2% 안 {(x.near <= 0.02).mean():.0%} · "
                  f"이미 +4% 이상 {(x.near > 0.04).mean():.0%} · 저가 뒤 최고 반등 중앙 {x.peak.median():+.1%} · {len(x)}종목")
        p("[버틴 시간 × 저가에서 떨어진 정도 → 지금 사면: +2% 익절 / 저가 -1% 손절 / 그날 종가]")
        for a, b_ in SINCE_BUCKETS2:
            for na, nb in NEAR_BUCKETS:
                x = g[(g.since >= a) & (g.since < b_) & (g.near > na) & (g.near <= nb)].drop_duplicates(["ticker", "day"])
                if len(x) < 15:
                    continue
                p(f"  {lab_s(a, b_):<9} {lab_n(na, nb):<10} n={len(x):>3} 평균 {x.ret.mean():+.2%} 익절 {(x.why == '익절').mean():.0%} "
                  f"손절 {(x.why == '손절').mean():.0%} 저가 버팀 {x.held.mean():.0%} 그 뒤 최고 중앙 {x.up_after.median():+.1%}")
        p("[30분 이상 버틴 뒤 저가 2% 안: 한 번 +3% 넘게 튄 뒤 되돌아온 '재확인' vs 계속 평평]")
        x = g[(g.since >= 30) & (g.near <= 0.02)].drop_duplicates(["ticker", "day"])
        for lab, m in (("재확인(튄 뒤 복귀)", x.peak >= 0.03), ("계속 평평", x.peak < 0.03)):
            y = x[m]
            if len(y) >= 10:
                p(f"  {lab:<12} n={len(y):>3} 평균 {y.ret.mean():+.2%} 익절 {(y.why == '익절').mean():.0%} 손절 {(y.why == '손절').mean():.0%} "
                  f"저가 버팀 {y.held.mean():.0%}")
        p("[같은 표, 11:00 ET 전 시점만]")
        x = g[(g.t <= 660) & (g.since >= 15)].drop_duplicates(["ticker", "day"])
        for na, nb in NEAR_BUCKETS:
            y = x[(x.near > na) & (x.near <= nb)]
            if len(y) >= 15:
                p(f"  15분 이상·{lab_n(na, nb):<10} n={len(y):>3} 평균 {y.ret.mean():+.2%} 익절 {(y.why == '익절').mean():.0%} 손절 {(y.why == '손절').mean():.0%}")
    txt = o.getvalue()
    open(os.path.join(RES, "lowfind_wait.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__" and sys.argv[1:] == ["waiting"]:
    waiting()


# ---------------------------------------------------------------- 반등 확인 매수: 저가 15분 유지 + 이미 a~b% 반등
BOUNCE_GRID = [(0.03, 0.06), (0.04, 0.07), (0.05, 0.08)]


def bounce_signal(b, c0, minutes, a, b_hi, near_rule=False):
    """첫 신호(09:45~15:30에 마감한 봉): 저가가 15분 이상 안 깨졌고, 종가가 저가 +a~+b%, 전일 대비 -7% 이하.
    near_rule=True면 비교용 NL15_2(저가 +2% 안)."""
    mins = (b.index.hour * 60 + b.index.minute).to_numpy()
    end = mins + minutes
    cl = b.Close.to_numpy()
    run_low, low_i = _lows(b)
    for i in range(len(b)):
        if end[i] < 585 or end[i] > 930:
            continue
        since = end[i] - (mins[low_i[i]] + minutes)
        near = cl[i] / run_low[i] - 1
        ok = near <= 0.02 if near_rule else a <= near <= b_hi
        if since >= 15 and ok and cl[i] / c0 - 1 <= CRASH:
            return i, cl[i], run_low[i]
    return None


def bounce_trade(b, minutes, i0, px, low_before, nxt, tp, stop_mode, fallback):
    mins = (b.index.hour * 60 + b.index.minute).to_numpy()
    end = mins + minutes
    stop, target = low_before * (1 - STOP_BELOW), px * (1 + tp)
    for j in range(i0 + 1, len(b)):
        if (stop_mode == "all" or (stop_mode == "awake" and end[j] <= AWAKE_END)) and b.Low.iloc[j] <= stop:
            return min(stop, b.Open.iloc[j]) / px - 1, "손절"
        if b.High.iloc[j] >= target:
            return tp, "익절"
    last = b.Close.iloc[-1] if fallback == "close" or nxt != nxt else nxt
    return last / px - 1, "종가" if fallback == "close" else "다음 날 시가"


def bounce():
    ev = pd.read_csv(os.path.join(RES, "lowfind_events.csv"))
    rows = []
    variants = [(f"반등{int(a * 100)}~{int(b_ * 100)}%", a, b_, False) for a, b_ in BOUNCE_GRID] + [("NL15_2(저가 2% 안)", 0, 0, True)]
    for t, d, win in zip(ev.ticker, ev.day, ev.win):
        for minutes in ((5, 3) if win == "뒤" else (5,)):
            got = regular_bars(t, d, minutes)
            if got is None:
                continue
            b, c0, nxt = got
            for name, a, b_, nr in variants:
                sig = bounce_signal(b, c0, minutes, a, b_, nr)
                if sig is None:
                    continue
                i0, px, lowb = sig
                at = b.index[i0] + pd.Timedelta(minutes=minutes)
                for tp in (0.02, 0.03):
                    for sm in ("all", "awake", "none"):
                        for fb in ("close", "next"):
                            r, why = bounce_trade(b, minutes, i0, px, lowb, nxt, tp, sm, fb)
                            rows.append(dict(ticker=t, day=d, win=win, bar=minutes, rule=name, tp=tp, stop=sm, fb=fb,
                                             at=at.strftime("%H:%M"), awake=at.hour * 60 + at.minute <= AWAKE_END,
                                             near=px / lowb - 1, ret=r, why=why))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "lowfind_bounce.csv"), index=False)
    o = io.StringIO()
    p = lambda *a_: print(*a_, file=o)
    p("반등 확인 매수 — 첫 신호 한 번, 앞5 / 뒤5 / 뒤3 (★ = 세 칸 모두 평균 플러스)")
    for only_awake in (False, True):
        p(f"\n==== {'11:00 ET 전 신호만' if only_awake else '모든 신호(09:45~15:30)'} ====")
        x = df[df.awake] if only_awake else df
        for (rule, tp, sm, fb), g in x.groupby(["rule", "tp", "stop", "fb"]):
            cells = [stat(g[(g.win == w) & (g.bar == bb)]) for w, bb in (("앞", 5), ("뒤", 5), ("뒤", 3))]
            star = all(c and c["avg"] > 0 for c in cells)
            p(f"  {'★' if star else ' '} {rule:<16} +{tp:.0%} 손절:{sm:<5} 못 닿으면:{'종가' if fb == 'close' else '다음날시가'}")
            for lab, c in zip(("앞5", "뒤5", "뒤3"), cells):
                p(f"       {lab} {fmt(c)}")
    txt = o.getvalue()
    open(os.path.join(RES, "lowfind_bounce.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__" and sys.argv[1:] == ["bounce"]:
    bounce()
