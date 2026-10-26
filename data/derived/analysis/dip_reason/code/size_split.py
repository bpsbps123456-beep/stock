"""'전일 급등 다음 날 정규장 이격 매수가 나쁘다'가 작고 위험한 종목(TEM 같은)만의 이야기인지 나눠 본다.
나누는 기준(모두 매수 전에 아는 값, 시가총액만 현재값):
  시가총액 3분위(yfinance fast_info, 현재값 - 순위만 쓴다)
  평소 변동성 3분위(종목별 직전 60거래일 일간 수익률 표준편차의 기간 중앙값)
  급등 정의 두 가지: 전일 +5% 이상(고정) / 전일 수익률이 평소 하루 변동의 2배 이상(종목별 정규화)
실적 반응일 급등을 뺀 결과와, 규칙을 실제로 적용했을 때 그룹별 총수익 변화도 본다.
사용: py size_split.py <raw 폴더> <ev 폴더>"""
import sys
from datetime import date
import numpy as np, pandas as pd
from analyze import load, cl_t
from blowoff import add
from fetch import universe
from earn_test import earn_dates, window_end
from rules import run, blocked, hot_days
from events import bars, prep


def daily(raw, s):
    d = pd.read_parquet(f"{raw}/1d_{s}.parquet")
    r = d.Close.pct_change()
    f = pd.DataFrame({"vol60": r.rolling(60, min_periods=40).std().shift(1)})   # 전일까지
    f.index = pd.to_datetime(f.index).date
    return f


def mcaps(raw):
    """현재 시가총액(yfinance fast_info). 처음 한 번 받아 raw에 저장한다."""
    p = f"{raw}/mcap.parquet"
    try:
        return pd.read_parquet(p).set_index("sym").mcap
    except FileNotFoundError:
        import yfinance as yf
        rows = []
        for s in universe():
            try:
                rows.append((s, yf.Ticker(s).fast_info["market_cap"]))
            except Exception:
                rows.append((s, np.nan))
        d = pd.DataFrame(rows, columns=["sym", "mcap"])
        d.to_parquet(p)
        return d.set_index("sym").mcap


def buckets(raw):
    mc = mcaps(raw)
    v = {s: daily(raw, s).vol60.loc[date(2026, 7, 14):date(2026, 10, 6)].median() for s in universe()}
    v = pd.Series(v)
    size = pd.qcut(mc.rank(method="first"), 3, labels=["소형", "중형", "대형"])
    risk = pd.qcut(v.rank(method="first"), 3, labels=["저변동", "중변동", "고변동"])
    cut_m = [mc[size == k].agg(["min", "max"]) for k in ("소형", "중형", "대형")]
    cut_v = [v[risk == k].agg(["min", "max"]) for k in ("저변동", "중변동", "고변동")]
    return size, risk, cut_m, cut_v, v


def hot_z(raw, s, d, k=2.0):
    """전일 수익률이 평소 하루 변동(직전 60일 표준편차)의 k배 이상인 날의 정규장."""
    c = pd.read_parquet(f"{raw}/1d_{s}.parquet").Close
    c.index = pd.to_datetime(c.index).date
    p1 = (c / c.shift(1) - 1).shift(1)
    v = daily(raw, s).vol60
    hot = set(p1[(p1 / v.reindex(p1.index)) >= k].index)
    return np.asarray([x in hot for x in d.date]) & (d.ses.values == "정규")


def earn_prev(ev, E):
    """전일이 실적 반응일이었나(전일 급등이 실적 때문인가)."""
    out = np.zeros(len(ev), bool)
    for s, g in E.groupby("sym"):
        days = set()
        for t in g.t:
            e = window_end(t, 0)
            days.add(e.date())                                   # 반응 정규장 날짜
        m = (ev.sym == s).values
        if m.any():
            prevd = pd.to_datetime(ev.date[m]) - pd.offsets.BDay(1)
            out[m] = [x.date() in days for x in prevd]
    return out


def line(v, m):
    if m.sum() < 5:
        return f"{m.sum():3d}건 (표본 부족)"
    _, t, _ = cl_t(m.astype(float).values, v.ret.values, v.date.astype(str).values)
    return (f"{m.sum():3d}건 평균 {v[m].ret.mean():+.2%} 승률 {(v[m].ret > 0).mean():.0%} 최악 {v[m].ret.min():+.1%}"
            f" | 나머지 {v[~m].ret.mean():+.2%} | t {t:+.2f}")


if __name__ == "__main__":
    raw, R = sys.argv[1], sys.argv[2]
    size, risk, cut_m, cut_v, vmed = buckets(raw)
    print("시가총액 구간:", ", ".join(f"{k} ${c['min'] / 1e9:,.1f}B~${c['max'] / 1e9:,.0f}B" for k, c in zip(("소형", "중형", "대형"), cut_m)))
    print("변동성 구간(하루 표준편차):", ", ".join(f"{k} {c['min']:.1%}~{c['max']:.1%}" for k, c in zip(("저변동", "중변동", "고변동"), cut_v)))
    print(f"TEM: {size['TEM']}·{risk['TEM']} (하루 변동 {vmed['TEM']:.1%})")
    E = earn_dates(raw)
    for name, e in (("3분봉 9/9~10/6", load(f"{R}/ev3.parquet")), ("5분봉 7/14~9/8", load(f"{R}/ev5.parquet", hi="2026-09-08"))):
        e = add(e, raw)
        vol = {s: daily(raw, s) for s in e.sym.unique()}
        e["vol60"] = [vol[s].vol60.get(d, np.nan) for s, d in zip(e.sym, e.date)]
        e["z1"] = e.prev1 / e.vol60
        e["size"], e["risk"] = e.sym.map(size), e.sym.map(risk)
        e["earn_prev"] = earn_prev(e, E)
        v = e[(e.ses == "정규")].dropna(subset=["prev1", "z1"])
        print(f"\n===== {name} · 정규장 이격 매수 {len(v)}건")
        for lab, hot in (("전일 +5% 이상", v.prev1 >= 0.05), ("전일 평소 변동 2배 이상", v.z1 >= 2)):
            print(f" [{lab}] 전체: {line(v, hot)}")
            print(f"   실적 반응 다음 날 제외: {line(v[~v.earn_prev], hot[~v.earn_prev])}")
            for col, ks in (("size", ("소형", "중형", "대형")), ("risk", ("저변동", "중변동", "고변동"))):
                for k in ks:
                    w = v[col] == k
                    print(f"   {k}: {line(v[w], hot[w])}")
        hot = v.prev1 >= 0.05
        top = v[hot].groupby("sym").ret.agg(["count", "mean"]).sort_values("count", ascending=False)
        print(" 전일 +5% 해당 거래 많은 종목:", ", ".join(f"{s} {r['count']:.0f}건 {r['mean']:+.1%}" for s, r in top.head(12).iterrows()))
        print(f" 해당 종목 수 {v[hot].sym.nunique()}개 / 거래 상위 10종목 비중 {top['count'].head(10).sum() / top['count'].sum():.0%}")

    # 규칙을 실제로 적용했을 때 그룹별 총수익 변화(전일 +5% 다음 날 정규장 매수 금지)
    for mins, lo, hi in ((3, date(2026, 9, 9), date(2026, 10, 6)), (5, date(2026, 7, 14), date(2026, 9, 8))):
        el, rl = (60, 14) if mins == 3 else (36, 9)
        D = {s: prep(bars(raw, s, mins), el, rl) for s in universe()}
        H = {s: hot_days(raw, s, d) for s, d in D.items()}
        HZ = {s: hot_z(raw, s, d) for s, d in D.items()}
        for reg in (False, True):
            base = run(D, None, lo, hi, H, reg_only=reg)
            for lab, HH in (("전일 +5%", H), ("평소 변동 2배", HZ)):
                rule = run(D, None, lo, hi, HH, reg_only=reg, hot=True)
                parts = []
                for col, ks in (("size", ("소형", "중형", "대형")), ("risk", ("저변동", "중변동", "고변동"))):
                    for k in ks:
                        a = base[base.sym.map(size if col == "size" else risk) == k].ret.sum()
                        b = rule[rule.sym.map(size if col == "size" else risk) == k].ret.sum()
                        parts.append(f"{k} {a:+.1f}→{b:+.1f}")
                bio = rule[(rule.t.dt.date == date(2026, 10, 6)) & rule.sym.isin(["TEM", "TWST", "TXG", "GRAL", "SDGR"])].ret.sum()
                print(f"\n{mins}분봉 {lo}~{hi} {'정규장만' if reg else '24시간'} [{lab} 다음 날 정규장 금지] 총수익 합(지금→규칙): "
                      f"전체 {base.ret.sum():+.1f}→{rule.ret.sum():+.1f} | " + " · ".join(parts) + (f" | 10/6 유전체5 {bio:+.1%}" if mins == 3 else ""))
