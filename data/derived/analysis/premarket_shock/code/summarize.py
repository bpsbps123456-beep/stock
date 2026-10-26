"""events_raw.csv(+earnings_dates.json) → results/summary.txt.
수익률은 체결가 대비. 같은 날 사건이 몰리면(관세 급락일 등) 사건 수만큼 독립이 아니므로 날짜별 평균으로 낸 t도 같이 적는다."""
import io, json, os
import numpy as np, pandas as pd
from analyze import LIMITS, MIN_CAP, RES, TPS

EXITS = [("open", "시가"), ("h1", "첫1시간"), ("close", "종가*"), ("d1o", "익일시가"),
         ("d1c", "익일종가"), ("d5c", "5일뒤"), ("d20c", "20일뒤")]
COST = 0.003                                 # 편도 스프레드 민감도


def flag_earnings(ev, ed):
    """전날 12시 이후~당일 09:30 사이 실적 발표(시각 없는 00:00 표기는 전날·당일 모두 인정)."""
    out = []
    for t, day in zip(ev.ticker, ev.day):
        hit = False
        for s in ed.get(t, []):
            ts = pd.Timestamp(s).tz_localize(None)
            d = ts.normalize()
            prev_ok = d < day and (day - d).days <= 4 and (ts.hour >= 12 or ts.hour == 0)   # 주말 낀 전 거래일
            same_ok = d == day and ts.hour < 12
            if prev_ok or same_ok:
                hit = True
                break
        out.append(hit)
    return out


def load():
    ev = pd.read_csv(os.path.join(RES, "events_raw.csv"), parse_dates=["day"])
    p = os.path.join(RES, "earnings_dates.json")
    ed = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    ev = ev[(ev.cap >= MIN_CAP) & (ev.day < pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize())].copy()   # 진행 중인 오늘 사건 제외
    uni = json.load(open(os.path.join(RES, "../bars/universe.json"), encoding="utf-8"))
    names = ev.ticker.map(lambda t: uni.get(t, {}).get("name", ""))
    ev = ev[~names.str.contains(r"Warrant|\bUnits?\b|\bRights?\b|Preferred|Notes|Debentures", case=False)]   # 보통주만
    ev["earn"] = flag_earnings(ev, ed)
    ev["ed_known"] = ev.ticker.isin([t for t, v in ed.items() if v])
    ev["news_ah"] = ev.ah_ret <= -0.05
    ev["mkt"] = ev.spy <= -0.015
    ev["period"] = np.where(ev.day.dt.year < 2025, "2024",
                            np.where(ev.day.dt.year == 2025, "2025",
                                     np.where(ev.day.dt.month <= 6, "2026상", "2026하")))
    return ev


def ret(ev, lim, ex, strict=False):
    f = ev[f"fill{'s' if strict else ''}_{int(lim*100)}"]
    r = (1 + ev[ex]) / (1 + f) - 1
    return r[f.notna() & ev[ex].notna()]


def stats(r, days):
    r = r.dropna()
    if len(r) < 3:
        return dict(n=len(r))
    g = r.groupby(days.loc[r.index]).mean()
    tday = g.mean() / (g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 2 and g.std() > 0 else np.nan
    return dict(n=len(r), 일=len(g), 평균=r.mean(), 중앙=r.median(), 승률=(r > 0).mean(),
                하위10=r.quantile(0.1), 최악=r.min(), t일=tday)


def fmt(d):
    if d.get("n", 0) < 3:
        return f"n={d.get('n', 0)}"
    return (f"n={d['n']:>3}({d['일']:>3}일) 평균 {d['평균']:+.1%} 중앙 {d['중앙']:+.1%} 승률 {d['승률']:.0%} "
            f"하위10% {d['하위10']:+.1%} 최악 {d['최악']:+.1%} t(일) {d['t일']:+.1f}")


def main():
    ev = load()
    o = io.StringIO()
    p = lambda *a: print(*a, file=o)
    p(f"사건: 프리장 저가 ≤ 전일 종가 -15%, 사건 시점 추정 시총 ≥ ${MIN_CAP/1e9:.0f}B. 총 {len(ev)}건, {ev.ticker.nunique()}종목, "
      f"{ev.day.min().date()} ~ {ev.day.max().date()}")
    p(f"실적 일정 확인 종목 비율 {ev.ed_known.mean():.0%} (모르면 실적 아님으로 분류됨)")
    p("\n[기간 × 구분별 사건 수]")
    p(pd.crosstab(ev.period, [ev.earn.map({True: "실적", False: "비실적"}), ev.mkt.map({True: "시장급락", False: "개별"})]).to_string())

    groups = {
        "비실적·개별(BULL형+SMCI형)": ev[~ev.earn & ~ev.mkt],
        "  ├ 프리장 뉴스(전날 애프터 -5% 미만, BULL형)": ev[~ev.earn & ~ev.mkt & ~ev.news_ah],
        "  └ 전날 애프터 뉴스(SMCI형)": ev[~ev.earn & ~ev.mkt & ev.news_ah],
        "실적·개별": ev[ev.earn & ~ev.mkt],
        "시장 동반 급락(SPY ≤ -1.5%)": ev[ev.mkt],
    }
    p("\n[프리장 저가가 그날 저점이었나 — 정규장이 프리장 저가를 깨지 않은 비율, 5일 안에 더 낮은 저가가 나온 비율]")
    for name, g in groups.items():
        for per in ["2026상", "2026하", "2025", "2024"]:
            x = g[g.period == per]
            if len(x) >= 3:
                held = (x.reg_low > x.pm_low).mean()
                lower5 = (x.low5.dropna() < x.pm_low[x.low5.notna()]).mean()
                p(f"{name:<38} {per:<6} n={len(x):>3} 정규장이 안 깸 {held:.0%}  5일 안 신저가 {lower5:.0%}  "
                  f"프리장 저가 중앙 {x.pm_low.median():+.1%} → 시가 중앙 {x.open.median():+.1%} → 종가 중앙 {x.close.median():+.1%}")

    for name, g in groups.items():
        p(f"\n==== {name} ====")
        for lim in LIMITS:
            p(f"-- 지정가 -{int(lim*100)}% --")
            for ex, lab in EXITS:
                for per in ["2026상", "2026하", "2025", "2024"]:
                    x = g[g.period == per]
                    s = stats(ret(x, lim, ex), x.day)
                    if s.get("n", 0) >= 3:
                        p(f"  {lab:<5} {per:<6} {fmt(s)}")
        p("-- 익절 지정가(체결 다음 봉~5거래일, 못 닿으면 5일 뒤 종가) --")
        for lim in LIMITS:
            for tp in TPS:
                col = f"tp{int(tp*100)}_{int(lim*100)}"
                if col not in g:
                    continue
                for per in ["2026상", "2026하", "2025"]:
                    x = g[(g.period == per) & g[col].notna()]
                    s = stats(x[col], x.day)
                    if s.get("n", 0) >= 3:
                        p(f"  -{int(lim*100)}% 매수 +{int(tp*100)}% 익절 {per:<6} {fmt(s)}  닿은 비율 {(x[col] == tp).mean():.0%}")

    # 연도 내 전방 검증: 2026 상반기에서 고르고 하반기로 확인 (비실적·개별)
    g = groups["비실적·개별(BULL형+SMCI형)"]
    p("\n[2026 전방 검증 — 상반기 평균 최고 조합을 하반기에 확인, 상반기 n ≥ 8]")
    cand = []
    for lim in LIMITS:
        for ex, lab in EXITS:
            if ex == "close":
                continue                                       # 종가는 직접 체결 불가
            a = stats(ret(g[g.period == "2026상"], lim, ex), g.day)
            if a.get("n", 0) >= 8:
                cand.append((a["평균"], lim, ex, lab, a))
    for m, lim, ex, lab, a in sorted(cand, reverse=True)[:5]:
        b = stats(ret(g[g.period == "2026하"], lim, ex), g.day)
        c = stats(ret(g[g.period == "2025"], lim, ex), g.day)
        p(f"  -{int(lim*100)}% → {lab}: 상 {fmt(a)}\n      하 {fmt(b)}\n      2025 {fmt(c)}")
    p(f"  시험한 조합: 지정가 {len(LIMITS)}개 × 매도 {len(EXITS) - 1}개 = {len(LIMITS) * (len(EXITS) - 1)}개")
    p(f"\n스프레드 민감도: 위 평균에서 왕복 {2*COST:.1%}를 빼서 읽는다(급락 프리장은 호가가 더 벌어질 수 있다).")

    p("\n[엄격 체결(0.1% 관통) — 비실적·개별, 2026 전체]")
    x = g[g.period.str.startswith("2026")]
    for lim in LIMITS:
        for ex, lab in [("open", "시가"), ("h1", "첫1시간"), ("d5c", "5일뒤")]:
            p(f"  -{int(lim*100)}% {lab:<5} 닿음 {fmt(stats(ret(x, lim, ex), x.day))}")
            p(f"  -{int(lim*100)}% {lab:<5} 엄격 {fmt(stats(ret(x, lim, ex, strict=True), x.day))}")

    cols = ["ticker", "day", "earn", "news_ah", "mkt", "cap", "ah_ret", "pm_low", "pm_low_hour", "open", "h1", "close", "d1c", "d5c", "low5"]
    p("\n[2026 비실적·개별 사건 목록]")
    p(x.sort_values("day")[cols].assign(cap=lambda d: (d.cap / 1e9).round(1)).round(3).to_string(index=False))
    txt = o.getvalue()
    open(os.path.join(RES, "summary.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
