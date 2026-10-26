"""프리장 몇 시에 사면 +5% 익절 확률이 높은가 (2025·2026).

매수 조건(그 시각에 아는 값만): 프리장 1시간봉 종가가 전일 정규장 종가 대비 -15% 이하이고, 직전 봉(첫 프리장 봉이면
전날 애프터 마지막 봉)도 -10% 이하로 마감, 두 봉 모두 폭이 있음(오류 체결·이어 찍힌 봉 거르기, report.md 참조).
매수가 = 그 봉 종가, 매수 시각 = 봉이 끝나는 시각(04시 봉 → 05:00, …, 09시 봉 → 09:29). 비교용으로 09:29에 같은 조건이면
정규장 시가에 사는 경우도 센다. 조건이 맞는 시각마다 한 건(같은 날 여러 시각이 들어갈 수 있다).
익절: 매수 다음 봉부터 지정가 +TP. 장외 봉은 오류 체결이 많아 종가로, 정규장 봉은 고가로 판정.
  당일: 그날 정규장 마감까지, 못 닿으면 다음 날 시가 매도(사용자는 자는 동안 지정가 매도만 걸어 둔다).
  5일: 5거래일 뒤 정규장 마감까지, 못 닿으면 5일 뒤 종가(참고).
사용: py hours.py build → results/hours_raw.csv,  py earn.py hours_raw.csv,  py hours.py report → results/hours.txt"""
import io, json, os, sys
import numpy as np, pandas as pd
from analyze import BARS, RES, MIN_CAP, sessions, daily, market_moves, load_universe

ENTRY = 0.15            # 매수 시각 종가 -15% 이하
CONFIRM = 0.10          # 직전 봉 종가 -10% 이하
TPS = [0.03, 0.05, 0.10]
LABEL = {4: "05:00", 5: "06:00", 6: "07:00", 7: "08:00", 8: "09:00", 9: "09:29", "open": "시가"}


def _ok(b, c0):
    return b.Close <= c0 * (1 - CONFIRM) and b.High > b.Low * 1.0005


def entries_for(t, x, shares, spy):
    x = sessions(x)
    d = daily(x)
    days = list(d.index)
    pm = x[x.ses == "pm"]
    prevc = pm.day.map(d.c.shift(1))
    hitd = set(pm.day[pm.Close <= prevc * (1 - ENTRY)])
    rows = []
    if not hitd:
        return rows
    byday = {k: g for k, g in x.groupby(["day", "ses"])}
    ext = (x.ses != "reg").values
    hi_px = np.where(ext, x.Close.values, x.High.values)          # 익절 판정용 가격
    lo_px = np.where(ext, x.Close.values, x.Low.values)           # 손실 폭 판정용 가격
    naive = x.index.tz_localize(None)
    for i, day in enumerate(days):
        if i == 0 or day not in hitd or i + 1 >= len(days):
            continue
        c0 = d.c.iloc[i - 1]
        p = byday.get((day, "pm"))
        ah = byday.get((days[i - 1], "ah"))
        prev = ah.iloc[-1] if ah is not None and len(ah) else None
        lows = []
        end_day = day + pd.Timedelta(hours=16)
        end_5 = days[i + 5] + pd.Timedelta(hours=16) if i + 5 < len(days) else None
        for ts, b in p.iterrows():
            if _ok(b, c0):
                lows.append(b.Low)
            if b.Close <= c0 * (1 - ENTRY) and _ok(b, c0) and prev is not None and _ok(prev, c0):
                buys = [(ts.hour, b.Close, ts)]
                if ts.hour == 9:                                   # 09:29에 같은 조건 → 시가 매수도 비교
                    o = d.o.iloc[i]
                    buys.append(("open", o, ts))
                for key, px, ts0 in buys:
                    after = naive > ts0.tz_localize(None)
                    m1 = after & (naive < end_day)
                    r = dict(ticker=t, day=day.date(), hour=key, buy=LABEL.get(key, str(key)), c0=c0,
                             cap=shares * c0 if shares else np.nan, depth=px / c0 - 1,
                             low_so_far=min(lows) / c0 - 1 if lows else np.nan,
                             ah_ret=(ah.Close.iloc[-1] / c0 - 1) if ah is not None and len(ah) else 0.0,
                             spy=spy.get((day, ts.hour), np.nan),
                             ret_close=d.c.iloc[i] / px - 1, ret_d1o=d.o.iloc[i + 1] / px - 1,
                             mae_day=(lo_px[m1].min() / px - 1) if m1.any() else np.nan)
                    for tp in TPS:
                        hit = bool((hi_px[m1] >= px * (1 + tp)).any())
                        r[f"hit{int(tp*100)}"] = hit
                        r[f"r{int(tp*100)}"] = tp if hit else r["ret_d1o"]
                    if end_5 is not None:
                        m5 = after & (naive < end_5)
                        hit5 = bool((hi_px[m5] >= px * 1.05).any())
                        r["hit5_5d"] = hit5
                        r["r5_5d"] = 0.05 if hit5 else d.c.iloc[i + 5] / px - 1
                    rows.append(r)
            prev = b
    return rows


_SPY, _UNI = None, None


def _init():
    global _SPY, _UNI
    _SPY, _UNI = market_moves(), load_universe()


def _one(f):
    t = os.path.basename(f)[3:-8]
    u = _UNI.get(t)
    try:
        return entries_for(t, pd.read_parquet(f), u["cap"] / u["price"] if u else None, _SPY)
    except Exception as e:
        print("건너뜀", t, e)
        return []


def build():
    import glob
    from concurrent.futures import ProcessPoolExecutor
    files = [f for f in sorted(glob.glob(f"{BARS}/1h_*.parquet")) if os.path.basename(f)[3:-8] not in ("SPY", "QQQ")]
    rows = []
    with ProcessPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) - 1), initializer=_init) as pool:
        for r in pool.map(_one, files, chunksize=20):
            rows += r
    ev = pd.DataFrame(rows)
    ev.to_csv(os.path.join(RES, "hours_raw.csv"), index=False)
    print(len(ev), "건")


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build()


def load():
    from summarize import flag_earnings
    ev = pd.read_csv(os.path.join(RES, "hours_raw.csv"), parse_dates=["day"])
    ev = ev[(ev.cap >= MIN_CAP) & (ev.day >= "2025-01-01")].copy()
    uni = load_universe()
    names = ev.ticker.map(lambda t: uni.get(t, {}).get("name", ""))
    ev = ev[~names.str.contains(r"Warrant|\bUnits?\b|\bRights?\b|Preferred|Notes|Debentures", case=False)].copy()
    ed = json.load(open(os.path.join(RES, "earnings_dates.json"), encoding="utf-8"))
    ev["earn"] = flag_earnings(ev, ed)
    ev["mkt"] = ev.spy <= -0.015
    ev["news_ah"] = ev.ah_ret <= -0.05
    ev["reb"] = (1 + ev.depth) / (1 + ev.low_so_far) - 1          # 그 시각까지 프리장 저점 대비 반등
    ev["period"] = np.where(ev.day.dt.year == 2025, "2025", np.where(ev.day.dt.month <= 6, "2026상", "2026하"))
    ev["key"] = ev.ticker + ev.day.dt.strftime("%Y%m%d")
    return ev


ORDER = ["05:00", "06:00", "07:00", "08:00", "09:00", "09:29", "시가"]
PERIODS = ["2025", "2026상", "2026하"]


def cell(x, tp="5"):
    if len(x) < 3:
        return None
    r = x[f"r{tp}"]
    miss = r[~x[f"hit{tp}"]]
    g = r.groupby(x.day).mean()
    t = g.mean() / (g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 2 and g.std() > 0 else np.nan
    return dict(n=len(x), 일=x.day.nunique(), 닿음=x[f"hit{tp}"].mean(), 평균=r.mean(), 중앙=r.median(),
                못닿음=miss.mean() if len(miss) else np.nan, 하위10=r.quantile(0.1), t=t)


def line(c):
    if c is None:
        return "  (3건 미만)"
    return (f"n={c['n']:>3}({c['일']:>3}일) +익절 닿음 {c['닿음']:>4.0%}  평균 {c['평균']:+6.1%}  중앙 {c['중앙']:+6.1%}  "
            f"못 닿은 건 평균 {c['못닿음']:+6.1%}  하위10% {c['하위10']:+6.1%}  t(일) {c['t']:+.1f}")


def report():
    ev = load()
    o = io.StringIO()
    p = lambda *a: print(*a, file=o)
    p(f"매수 기회 {len(ev)}건(사건 {ev.key.nunique()}개, {ev.ticker.nunique()}종목), {ev.day.min().date()} ~ {ev.day.max().date()}. "
      "같은 사건이 여러 시각에 들어간다.")
    p("매수 시각은 미국 동부시각. 한국시각은 서머타임 +13시간(예: 08:00 → 21:00), 표준시 +14시간.")
    p("평균 = +5% 익절 지정가(당일 마감까지), 못 닿으면 다음 날 시가 매도. 수수료 0, 스프레드 미반영(왕복 0.6%를 빼서 읽는다).")
    groups = {"실적 외·개별 악재": ev[~ev.earn & ~ev.mkt],
              "  ├ 프리장 뉴스(BULL형)": ev[~ev.earn & ~ev.mkt & ~ev.news_ah],
              "  └ 전날 애프터 뉴스(SMCI형)": ev[~ev.earn & ~ev.mkt & ev.news_ah],
              "실적 발표·개별": ev[ev.earn & ~ev.mkt],
              "시장 동반 급락(SPY -1.5% 이하)": ev[ev.mkt]}
    for name, g in groups.items():
        p(f"\n==== {name}: 매수 시각별 +5% 당일 익절 ====")
        for b in ORDER:
            for per in PERIODS:
                x = g[(g.buy == b) & (g.period == per)]
                if len(x):
                    p(f"  {b:<6}{per:<6}{line(cell(x))}")
    g = groups["실적 외·개별 악재"]
    p("\n==== 실적 외·개별: 익절 폭 바꾸기(2025 / 2026 전체) ====")
    for tp in ["3", "10"]:
        for b in ORDER:
            for per, m in [("2025", g.period == "2025"), ("2026", g.period != "2025")]:
                x = g[(g.buy == b) & m]
                if len(x):
                    p(f"  +{tp}%  {b:<6}{per:<6}{line(cell(x, tp))}")
    p("\n==== 실적 외·개별: 같은 사건끼리 비교(09:00·09:29·시가 세 시각 모두 매수 조건이 맞은 사건만) ====")
    for per in PERIODS:
        x = g[g.period == per]
        keys = set.intersection(*[set(x[x.buy == b].key) for b in ["09:00", "09:29", "시가"]])
        for b in ["09:00", "09:29", "시가"]:
            p(f"  {b:<6}{per:<6}{line(cell(x[(x.buy == b) & x.key.isin(keys)]))}")
    p("\n==== 실적 외·개별: 같은 사건끼리 비교(06:00~시가 모두 조건이 맞은 사건, 대부분 전날 애프터 뉴스) ====")
    for per in PERIODS:
        x = g[g.period == per]
        keys = set.intersection(*[set(x[x.buy == b].key) for b in ORDER[1:]])
        for b in ORDER[1:]:
            p(f"  {b:<6}{per:<6}{line(cell(x[(x.buy == b) & x.key.isin(keys)]))}")
    p("\n==== 실적 외·개별: 그 시각에 프리장 저점에서 이미 5% 이상 반등 중이었나 ====")
    for lab, m in [("반등 중(5%+)", g.reb >= 0.05), ("저점 근처(<5%)", g.reb < 0.05)]:
        for b in ORDER:
            for per in PERIODS:
                x = g[m & (g.buy == b) & (g.period == per)]
                if len(x):
                    p(f"  {lab:<12}{b:<6}{per:<6}{line(cell(x))}")
    p("\n==== 실적 외·개별: 5거래일 안 +5% 익절(못 닿으면 5일 뒤 종가) ====")
    for b in ORDER:
        for per in PERIODS:
            x = g[(g.buy == b) & (g.period == per) & g.hit5_5d.notna()].assign(hit5=lambda d: d.hit5_5d.astype(bool), r5=lambda d: d.r5_5d)
            if len(x):
                p(f"  {b:<6}{per:<6}{line(cell(x))}")
    # 연도 내 전방 검증: 2026 상반기에서 고르고 하반기·2025로 확인
    cand = []
    for name, gg in groups.items():
        for b in ORDER:
            for lab, m in [("전체", gg.reb > -9), ("반등 중", gg.reb >= 0.05), ("저점 근처", gg.reb < 0.05)]:
                x = gg[m & (gg.buy == b)]
                a = cell(x[x.period == "2026상"])
                if a and a["n"] >= 8:
                    cand.append((name.strip(), b, lab, a, cell(x[x.period == "2026하"]), cell(x[x.period == "2025"])))
    for key, title in [("평균", "평균 수익"), ("닿음", "+5% 닿은 비율")]:
        p(f"\n[2026 전방 검증 — 상반기 {title} 상위 5개(후보 {len(cand)}개 중)를 하반기·2025로 확인]")
        for name, b, lab, a, bb, c in sorted(cand, key=lambda z: -z[3][key])[:5]:
            p(f"  {name} / {b} / {lab}\n    상 {line(a)}\n    하 {line(bb)}\n    2025 {line(c)}")
    p(f"\n시험 범위: 매수 시각 7개 × 사건 묶음 5개 × 기간 3개, 익절 3개(+3·5·10%) × 시각 7개(실적 외만), 반등 여부 2개, "
      f"5일 익절 1개. 전방 검증 후보 {len(cand)}개(묶음 5 × 시각 7 × 반등 3 중 상반기 8건 이상).")
    txt = o.getvalue()
    open(os.path.join(RES, "hours.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__" and sys.argv[1:] == ["report"]:
    report()
