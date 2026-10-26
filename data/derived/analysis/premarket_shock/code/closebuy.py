"""사용자 아이디어(2026-10-10): 급락주를 종가에 사면 — 1시간봉으로 만든 정규장 일봉으로 2025~2026을 잰다.

사건: 정규장 종가가 전일 종가 대비 -7% 이하, 사건 시점 추정 시총 10억 달러 이상(현재 주식 수 × 전일 종가),
      시가 갭 -60% 초과(액면분할·오류 거르기). 그날 종가에 알 수 있는 값만 쓴다.
매수: 종가     — 장 마감 주문(LOC·MOC)이 있어야 한다(서머타임 한국 05:00).
      장후     — 20:00 ET 전 마지막 장후 1시간봉 종가(한국 09:00 무렵). 주간거래 시세가 없어 대신 쓴다. 종가와 20% 넘게 다르면 버린다.
매도: 다음날시가 / 다음날+2%(정규장 지정가, 시가가 이미 위면 시가, 못 닿으면 다음 날 종가)
      / 다음날+2%·손절(급락일 저가 -1%, 시가가 이미 아래면 시가, 같은 날 둘 다 닿으면 손절 — 일봉이라 순서를 모른다)
      / 다음날종가 / 3일 뒤 종가 / 5일 뒤 종가.
비교: 같은 날 SPY를 종가에 사서 같은 때 판 수익(차이 = 급락주 - SPY).
기간: 2026 1~6월(고르기) → 2026 7월~10/5(확인), 2025(보조). 10/7은 자료를 받은 날이라 10/6까지만 쓴다.
사용: py closebuy.py run → results/closebuy.txt, closebuy.csv"""
import glob, io, os, sys
import numpy as np, pandas as pd
from analyze import sessions, daily, load_universe, BARS, RES

DROP = -0.07
LAST_DAY = pd.Timestamp("2026-10-06")
TP, STOP_BELOW, AH_MAX = 0.02, 0.01, 0.20
EXITS = ("다음날시가", "다음날+2%", "다음날+2%·손절", "다음날종가", "3일뒤종가", "5일뒤종가")
PLAIN = {"다음날시가": "o1", "다음날종가": "c1", "3일뒤종가": "c3", "5일뒤종가": "c5"}
PERIODS = (("2025", "2025-01-01", "2025-12-31"), ("2026 1~6월", "2026-01-01", "2026-06-30"),
           ("2026 7월~10/5", "2026-07-01", "2026-10-05"), ("그중 9/8~10/5", "2026-09-08", "2026-10-05"))
DEPTH = (("-7~-10%", -0.10, -0.07), ("-10~-15%", -0.15, -0.10), ("-15% 이하", -0.60, -0.15))


def table(x):
    """정규장 일봉 + 그날 정규장 봉 수 + 20:00 ET 전 마지막 장후 가격."""
    s = sessions(x)
    d = daily(s)
    d["n"] = s[s.ses == "reg"].groupby("day").size()
    d["h1c"] = s[s.ses == "reg"].groupby("day").Close.first()       # 첫 1시간봉 종가(10:30 ET, 한국 23:30)
    ah = s[(s.ses == "ah") & (s.index.hour < 20)]
    d["ah"] = ah.groupby("day").Close.last()
    return d[(d.index <= LAST_DAY) & (d.n >= 4)]


def legs(d, i, px):
    """i일 px에 사서 각 방법으로 판 수익."""
    o, h, l, c = d.o.to_numpy(), d.h.to_numpy(), d.l.to_numpy(), d.c.to_numpy()
    o1, h1, l1, c1 = o[i + 1], h[i + 1], l[i + 1], c[i + 1]
    target, stop = px * (1 + TP), l[i] * (1 - STOP_BELOW)
    plain_tp = o1 / px - 1 if o1 >= target else (TP if h1 >= target else c1 / px - 1)
    if o1 >= target:
        guarded = o1 / px - 1
    elif o1 <= stop:
        guarded = o1 / px - 1
    elif l1 <= stop:
        guarded = stop / px - 1
    elif h1 >= target:
        guarded = TP
    else:
        guarded = c1 / px - 1
    return {"다음날시가": o1 / px - 1, "다음날+2%": plain_tp, "다음날+2%·손절": guarded, "다음날종가": c1 / px - 1,
            "3일뒤종가": c[i + 3] / px - 1 if i + 3 < len(d) else np.nan,
            "5일뒤종가": c[i + 5] / px - 1 if i + 5 < len(d) else np.nan}


def run():
    uni = load_universe()
    spy = table(pd.read_parquet(os.path.join(BARS, "1h_SPY.parquet")))
    sc = spy.c
    spy_day = (sc / sc.shift(1) - 1).to_dict()                # 그날 SPY 등락(시장 동반 급락일 구분)
    spy_ret = {}
    for i, day in enumerate(spy.index[:-1]):
        spy_ret[day] = {k: (spy[col].iloc[i + j] / sc.iloc[i] - 1) if i + j < len(spy) else np.nan
                        for k, (col, j) in {"o1": ("o", 1), "c1": ("c", 1), "c3": ("c", 3), "c5": ("c", 5)}.items()}
    rows, dropped = [], 0
    for f in glob.glob(os.path.join(BARS, "1h_*.parquet")):
        t = os.path.basename(f)[3:-8]
        if t in ("SPY", "QQQ") or t not in uni:
            continue
        d = table(pd.read_parquet(f))
        if len(d) < 3:
            continue
        shares = uni[t]["cap"] / uni[t]["price"]
        c0 = d.c.shift(1)
        dd, gap = d.c / c0 - 1, d.o / c0 - 1
        for i in np.flatnonzero(((dd <= DROP) & (gap > -0.6) & (shares * c0 >= 1e9)).to_numpy()):
            if i + 1 >= len(d):
                continue
            day = d.index[i]
            if d.o.iloc[i + 1] / d.c.iloc[i] - 1 > 0.6:          # 다음 날 +60% 넘는 갭은 자료 오류·병합으로 보고 버린다
                dropped += 1
                continue
            base = dict(ticker=t, day=day, dd=dd.iloc[i], near=d.c.iloc[i] / d.l.iloc[i] - 1, cap=shares * c0.iloc[i],
                        spy_day=spy_day.get(day, np.nan), h1dd=d.h1c.iloc[i] / c0.iloc[i] - 1)
            r = legs(d, i, d.c.iloc[i])
            sp = spy_ret.get(day, {})
            rows.append(dict(base, entry="종가", **r, **{f"spy_{k}": sp.get(v, np.nan) for k, v in PLAIN.items()}))
            ah = d.ah.iloc[i]
            if ah == ah and abs(ah / d.c.iloc[i] - 1) <= AH_MAX:
                rows.append(dict(base, entry="장후", ah_vs_close=ah / d.c.iloc[i] - 1, **legs(d, i, ah)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "closebuy.csv"), index=False)
    print("버린 사건(다음 날 +60% 갭)", dropped)
    report(df)


def tday(x, col):
    g = x.dropna(subset=[col]).groupby("day")[col].mean()
    return g.mean() / (g.std(ddof=1) / np.sqrt(len(g))) if len(g) > 2 and g.std() > 0 else np.nan


def line(x, col):
    """건 평균(급락주가 몰린 날 — 시장 폭락일 — 이 크게 실린다)과 날 평균(하루 한 표)을 함께."""
    r = x[col].dropna()
    if len(r) < 20:
        return f"({len(r)}건)"
    dm = x.dropna(subset=[col]).groupby("day")[col].mean().mean()
    return (f"{r.mean():+.2%}·날 평균 {dm:+.2%}(t {tday(x, col):+.1f}, 승률 {(r > 0).mean():.0%}, "
            f"하위10% {r.quantile(.1):+.1%}, n {len(r)})")


def report(df):
    df["day"] = pd.to_datetime(df["day"])
    o = io.StringIO()
    p = lambda *a: print(*a, file=o)
    p("[급락주 종가 매수 — 정규장 종가가 전일 대비 -7% 이하인 날, 평균(t(일), 승률, 하위10%, 건수)]")
    for name, a, b in PERIODS:
        x0 = df[(df.day >= a) & (df.day <= b)]
        x = x0[x0.entry == "종가"]
        p(f"\n==== {name} · 종가 매수 {len(x)}건({x.day.nunique()}일) ====")
        for col in EXITS:
            p(f"  {col:<12} {line(x, col)}")
            if col in PLAIN:
                ex = x[col] - x[f"spy_{col}"]
                p(f"  {'':<12}   같은 때 SPY {x[f'spy_{col}'].mean():+.2%} → 차이 {ex.mean():+.2%}"
                  f"(t {tday(x.assign(ex=ex), 'ex'):+.1f})")
        for lab, lo_, hi_ in DEPTH:
            y = x[(x.dd > lo_) & (x.dd <= hi_)]
            p(f"  {lab:<9} 다음날시가 {line(y, '다음날시가')} · 다음날종가 {y['다음날종가'].mean():+.2%} · 5일 {y['5일뒤종가'].mean():+.2%}")
        for lab, m in (("종가가 저가 +1% 안", x.near <= 0.01), ("종가가 저가 +1% 위", x.near > 0.01),
                       ("개별 급락(SPY > -1.5%)", x.spy_day > -0.015), ("시장 동반(SPY ≤ -1.5%)", x.spy_day <= -0.015)):
            y = x[m]
            p(f"  {lab:<14} 다음날시가 {line(y, '다음날시가')} · 다음날종가 {y['다음날종가'].mean():+.2%}")
        y = x[x.h1dd <= DROP]
        p(f"  실행판(10:30 ET에 이미 -7% 이하 → 자기 전 LOC 매수, 한도 전일 종가 -7%) 다음날시가 {line(y, '다음날시가')} · "
          f"다음날종가 {y['다음날종가'].mean():+.2%}")
        z = x0[x0.entry == "장후"]
        if len(z):
            p(f"  장후 가격 매수(한국 09시 무렵, 종가 대비 중앙 {z.ah_vs_close.median():+.1%}): "
              f"다음날시가 {line(z, '다음날시가')} · 다음날+2% {z['다음날+2%'].mean():+.2%} · 다음날종가 {z['다음날종가'].mean():+.2%}")
    p(f"\n시험 범위: 매수 2개(종가·장후) × 매도 {len(EXITS)}개 × 기간 {len(PERIODS)}개, 깊이 3구간·종가 위치 2구간은 결과 보기 전에 정한 묘사다. "
      f"고르기는 2026 1~6월, 확인은 2026 7월~10/5, 2025는 보조.")
    txt = o.getvalue()
    open(os.path.join(RES, "closebuy.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


def report_only():
    report(pd.read_csv(os.path.join(RES, "closebuy.csv")))


if __name__ == "__main__":
    {"run": run, "report": report_only}[sys.argv[1]]()
