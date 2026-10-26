"""매수 뒤 피해를 줄이는 규칙 비교. dip_scalper.pine 상태 기계에 한 가지씩 더한다.
  earn       실적 반응 구간(발표 ~ 발표 뒤 첫 정규장 종가) 매수 금지
  skip_open  정규장 개장 후 N분 매수 금지
  stop       종가가 매수가 -k% 이하면 다음 봉 시가에 판다(Pine devStop과 같음)
  cool       손절한 뒤 원래 정리 시각(다음 아침 8시 한국)까지 그 종목 재매수 안 함
  reg_only   정규장에서만 매수(yfinance 애프터·프리 꼬리 체결 배제용)
  hot        전일 +5% 이상 오른 종목은 다음 날 정규장 매수 금지(장 시작 전에 아는 값, 2026-10-07 TEM 질문 뒤 추가)
창: 3분봉 9/9~10/6, 5분봉 7/14~9/8. 사용: py rules.py <raw 폴더>"""
import sys
from datetime import date
import numpy as np, pandas as pd
from events import bars, prep, GAP, X, HR
from fetch import universe
from earn_test import earn_dates, window_end

BIO = ["TEM", "TWST", "TXG", "GRAL", "SDGR"]          # 10/6 같이 무너진 유전체 종목


def blocked(d, E, s):
    t = d.index.values
    f = np.zeros(len(t), bool)
    for x in E.t[E.sym == s]:
        f |= (t >= x.to_datetime64()) & (t < window_end(x, 0).to_datetime64())
    return f


def hot_days(raw, s, d, th=0.05):
    """봉마다 '전일 수익률 >= th인 날의 정규장'인지."""
    c = pd.read_parquet(f"{raw}/1d_{s}.parquet").Close
    c.index = pd.to_datetime(c.index).date
    p1 = (c / c.shift(1) - 1).shift(1)
    hot = set(p1[p1 >= th].index)
    return np.asarray([x in hot for x in d.date]) & (d.ses.values == "정규")


def sim(d, earn=None, skip_open=0, stop=0.0, cool=False, reg_only=False, hot=None):
    o, l, c, r, ema = d.o.values, d.l.values, d.c.values, d.r.values, d.ema.values
    mod, ses = d["mod"].values, d.ses.values
    kst = d.index.tz_convert("Asia/Seoul")
    tk = kst.as_unit("ns").asi8
    out, hold, ent, ei, dead, until = [], False, 0.0, 0, 0, 0
    for i in range(61, len(d)):
        if hold:
            why = None
            if i - 1 >= ei and r[i - 1] > X:
                why = "RSI"
            elif stop > 0 and i - 1 >= ei and c[i - 1] <= ent * (1 - stop):
                why = "손절"
            elif tk[i] >= dead:
                why = "8시"
            if why:
                out.append((ei, i, ent, o[i], why)); hold = False
                if why == "손절" and cool:
                    until = dead
        else:
            if tk[i] < until or (earn is not None and earn[i]) or (reg_only and ses[i] != "정규") or (hot is not None and hot[i]):
                continue
            if skip_open and ses[i] == "정규" and mod[i] < 570 + skip_open:
                continue
            lvl = ema[i - 1] * (1 - GAP)
            if l[i] <= lvl:
                ent, ei, hold = min(o[i], lvl), i, True
                t8 = kst[i].normalize() + pd.Timedelta(hours=HR)
                dead = (t8 if kst[i] < t8 else t8 + pd.Timedelta(days=1)).as_unit("ns").value
    return out


def run(D, B, lo, hi, H=None, **kw):
    rows = []
    for s, d in D.items():
        k = dict(kw)
        if k.pop("earn", False):
            k["earn"] = B[s]
        if k.pop("hot", False):
            k["hot"] = H[s]
        for ei, xi, ent, px, why in sim(d, **k):
            t = d.index[ei]
            if lo <= t.date() <= hi:
                rows.append((s, t, d.ses.iloc[ei], d["mod"].iloc[ei], px / ent - 1, why))
    return pd.DataFrame(rows, columns=["sym", "t", "ses", "mod", "ret", "why"])


def line(t):
    return (f"{len(t):5d}건 평균 {t.ret.mean():+.2%} 승률 {(t.ret > 0).mean():.0%} "
            f"최악 {t.ret.min():+.1%} 하위5% {t.ret.quantile(.05):+.1%} 합 {t.ret.sum():+.1f}")


VARIANTS = {
    "기준(지금 지표)": {},
    "실적 직후 금지": dict(earn=True),
    "개장 30분 금지": dict(skip_open=30),
    "개장 60분 금지": dict(skip_open=60),
    "손절 5% 곧바로 재매수": dict(stop=0.05),
    "손절 5% + 그날 재매수 안 함": dict(stop=0.05, cool=True),
    "손절 7% 곧바로 재매수": dict(stop=0.07),
    "손절 7% + 그날 재매수 안 함": dict(stop=0.07, cool=True),
    "손절 10% + 그날 재매수 안 함": dict(stop=0.10, cool=True),
    "전일 +5% 다음 날 정규장 금지": dict(hot=True),
    "전일 +5% 금지 + 손절 7%·재매수 안 함": dict(hot=True, stop=0.07, cool=True),
}

if __name__ == "__main__":
    raw = sys.argv[1]
    E = earn_dates(raw)
    for mins, lo, hi in ((3, date(2026, 9, 9), date(2026, 10, 6)), (5, date(2026, 7, 14), date(2026, 9, 8))):
        el, rl = (60, 14) if mins == 3 else (36, 9)
        D = {s: prep(bars(raw, s, mins), el, rl) for s in universe()}
        B = {s: blocked(d, E, s) for s, d in D.items()}
        H = {s: hot_days(raw, s, d) for s, d in D.items()}
        for reg in (False, True):
            print(f"\n===== {mins}분봉 {lo}~{hi} ({len(D)}종목) · {'정규장에서만 매수' if reg else '24시간'}")
            for name, kw in VARIANTS.items():
                t = run(D, B, lo, hi, H, reg_only=reg, **kw)
                bio = t[(t.t.dt.date == date(2026, 10, 6)) & t.sym.isin(BIO)]
                tail = f" | 10/6 유전체5 {bio.ret.sum():+.1%}({len(bio)}건) TEM {'/'.join(f'{x:+.1%}' for x in bio[bio.sym == 'TEM'].ret)}" if mins == 3 else ""
                print(f"  {name:20s}", line(t), tail)
            if reg:
                t = run(D, B, lo, hi, reg_only=True)
                t["open"], t["day"] = t["mod"] < 600, t.t.dt.date
                g = t.groupby(["day", "open"]).ret.mean().unstack().dropna()
                print(f"  날짜별: 개장 30분 매수가 그 뒤 매수보다 나쁜 날 {(g[True] < g[False]).sum()}/{len(g)}일, 차이 평균 {(g[True] - g[False]).mean():+.2%}")
