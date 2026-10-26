"""가장 확실한 '이유 있는 하락'인 실적 발표 직후에 산 이격 매수의 성적.
실적 반응 구간 = 발표 시각 ~ 발표 뒤 첫 정규장 종가(+1이면 그다음 정규장 종가까지). 발표일은 yfinance(지난 일정만 사용)."""
import os, sys
import numpy as np, pandas as pd, yfinance as yf
from fetch import universe
from analyze import load, cl_t


def earn_dates(raw):
    p = f"{raw}/earnings.parquet"
    if os.path.exists(p):
        return pd.read_parquet(p)
    rows = []
    for s in universe():
        try:
            e = yf.Ticker(s).get_earnings_dates(limit=8)
        except Exception:
            continue
        if e is not None:
            rows += [(s, t.tz_convert("America/New_York")) for t in e.index]
    df = pd.DataFrame(rows, columns=["sym", "t"])
    df.to_parquet(p)
    return df


def window_end(t, extra):
    """발표 뒤 첫 정규장(발표가 9:30 이후면 다음 거래일) 16:00 ET, extra일 더."""
    d = t.normalize()
    if t.hour * 60 + t.minute >= 570:
        d += pd.offsets.BDay(1)
    return d + pd.offsets.BDay(extra) + pd.Timedelta(hours=16)


def flag(ev, E, extra):
    f = np.zeros(len(ev), bool)
    for s, g in E.groupby("sym"):
        m = (ev.sym == s).values
        if not m.any():
            continue
        tt = ev.t[m].values
        hit = np.zeros(m.sum(), bool)
        for t in g.t:
            a, b = t.to_datetime64(), window_end(t, extra).to_datetime64()
            hit |= (tt >= a) & (tt < b)
        f[m] = hit
    return f


if __name__ == "__main__":
    raw, R = sys.argv[1], sys.argv[2]
    E = earn_dates(raw)
    print("실적 일정", len(E), "건,", E.sym.nunique(), "종목")
    for name, ev in (("3분봉 9/9~10/6", load(f"{R}/ev3.parquet")), ("5분봉 7/14~9/8", load(f"{R}/ev5.parquet", hi="2026-09-08"))):
        print(f"\n===== {name}")
        for extra in (0, 1):
            fl = flag(ev, E, extra)
            b, t, G = cl_t(fl.astype(float), ev.ret.values, ev.date.astype(str).values)
            on, off = ev[fl], ev[~fl]
            print(f"  반응 구간 +{extra}일: 해당 {len(on)}건 평균 {on.ret.mean():+.2%} 승률 {(on.ret > 0).mean():.0%} 최악 {on.ret.min():+.1%} 평균MAE {on.mae.mean():+.1%}"
                  f" | 나머지 평균 {off.ret.mean():+.2%} 승률 {(off.ret > 0).mean():.0%} | 차이 t {t:+.2f}")
            for s in ("정규", "프리", "애프터"):
                o2, f2 = ev[(ev.ses == s) & fl], ev[(ev.ses == s) & ~fl]
                if len(o2):
                    print(f"     {s}: 실적 {len(o2)}건 {o2.ret.mean():+.2%}(승률 {(o2.ret > 0).mean():.0%}, 최악 {o2.ret.min():+.1%}) vs 나머지 {f2.ret.mean():+.2%}")
        worst = ev.assign(e=flag(ev, E, 1)).nsmallest(12, "ret")[["sym", "t", "ses", "ret", "e", "grp_dev", "rvol"]]
        print("  최악 12건(e = 실적 반응 구간):\n" + worst.to_string(index=False))
