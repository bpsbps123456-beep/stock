"""이격매수매도를 3분봉으로 계산할까, 5분봉으로 할까. 같은 1분봉(9/9~10/6, 프리·애프터 포함)에서 봉을 만들어 같은 규칙으로 돌린다.
변형(이름 = 봉 · EMA · RSI, 괄호는 시간 길이):
  3분 60·14   지금 지표(EMA 180분, RSI 42분)
  5분 60·14   설정만 5분으로 바꾼 경우(EMA 300분, RSI 70분)
  5분 36·8    시간을 맞춘 5분(EMA 180분, RSI 40분)
  3분 100·23  5분 60·14와 시간이 같은 3분 - 차이가 봉 크기 때문인지 시간 길이 때문인지 가른다
  참고: 1·2·4·6·10분, 시간 맞춤(EMA 180분, RSI 약 42분)
규칙: 지금 규칙 / 전일 +5% 다음 날 정규장 금지(rules.hot_days).
대상: fetch.user_universe()(트레이딩뷰 AI 38, TIME·KoAct 61, 합집합 78)와 참고용 127종목.
첫날(9/9)은 EMA 준비 구간이라 9/10~10/6 매수만 센다. 사용: py tf_compare.py <raw 폴더>"""
import sys
from datetime import date
import numpy as np, pandas as pd
from events import prep
from fetch import universe, user_universe
from rules import sim, hot_days

LO, HI = date(2026, 9, 10), date(2026, 10, 6)
VARIANTS = {"3분 60·14 (지금)": (3, 60, 14), "5분 60·14 (설정만 변경)": (5, 60, 14),
            "5분 36·8 (시간 맞춤)": (5, 36, 8), "3분 100·23 (5분 60·14와 같은 시간)": (3, 100, 23)}
EXTRA = {f"{m}분 시간 맞춤": (m, round(180 / m), max(2, round(42 / m))) for m in (1, 2, 4, 6, 10)}


def bars1m(raw, s, mins):
    m = pd.read_parquet(f"{raw}/1m_{s}.parquet")
    if mins == 1:
        return m[m.High > 0]
    b = m.resample(f"{mins}min", origin="start_day", label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(subset=["Close"])
    return b[b.High > 0]


def trades(raw, syms, mins, el, rl):
    rows = []
    for s in syms:
        try:
            d = prep(bars1m(raw, s, mins), el, rl)
        except FileNotFoundError:
            continue
        hot = hot_days(raw, s, d)
        for rule, h in (("지금 규칙", None), ("전일 +5% 금지", hot)):
            for reg in (False, True):
                for ei, xi, ent, px, why in sim(d, reg_only=reg, hot=h):
                    t = d.index[ei]
                    if LO <= t.date() <= HI:
                        rows.append((s, rule, "정규장만" if reg else "24시간", t, t.date(), d.ses.iloc[ei], px / ent - 1,
                                     (xi - ei) * mins))
    return pd.DataFrame(rows, columns=["sym", "rule", "mode", "t", "day", "ses", "ret", "hold_min"])


def day_t(a, b):
    """날짜별 합의 차이(b - a) 평균과 t."""
    x = pd.concat([a.groupby("day").ret.sum(), b.groupby("day").ret.sum()], axis=1).fillna(0)
    d = x.iloc[:, 1] - x.iloc[:, 0]
    return d.mean(), d.mean() / (d.std(ddof=1) / np.sqrt(len(d))), (d > 0).sum(), len(d)


if __name__ == "__main__":
    raw = sys.argv[1]
    U = user_universe()
    U["참고 127종목"] = list(universe())
    allsyms = sorted(set().union(*U.values()))
    T = {}
    for name, (mins, el, rl) in {**VARIANTS, **EXTRA}.items():
        T[name] = trades(raw, allsyms, mins, el, rl)
        print(f"계산: {name} {len(T[name])}행", file=sys.stderr)
    base = "3분 60·14 (지금)"
    for uname, syms in U.items():
        print(f"\n===== {uname} ({len(set(syms))}종목, {LO:%m/%d}~{HI:%m/%d} 매수)")
        for rule in ("지금 규칙", "전일 +5% 금지"):
            for mode in ("24시간", "정규장만"):
                print(f" [{rule} · {mode}]")
                b = T[base][(T[base].sym.isin(syms)) & (T[base].rule == rule) & (T[base]["mode"] == mode)]
                for name in list(VARIANTS) + (list(EXTRA) if uname == "합집합" else []):
                    t = T[name][(T[name].sym.isin(syms)) & (T[name].rule == rule) & (T[name]["mode"] == mode)]
                    mean_d, tt, up, nd = day_t(b, t)
                    cmp = "" if name == base else f" | 날짜별 차이 {mean_d:+.2f}/일 (t {tt:+.2f}, 나은 날 {up}/{nd})"
                    print(f"   {name:24s} {len(t):5d}건 평균 {t.ret.mean():+.2%} 승률 {(t.ret > 0).mean():.0%} "
                          f"최악 {t.ret.min():+.1%} 합 {t.ret.sum():+6.1f} 보유중앙 {t.hold_min.median():.0f}분{cmp}")
