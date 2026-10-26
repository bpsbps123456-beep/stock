"""TEM 10/6 모양(전날 급등 → 당일 고점 갱신 → 반락)에서 산 이격 매수 성적. 모두 매수 순간에 알 수 있는 값이다.
  prev1  전일 수익률(장 시작 전에 앎)
  spike  오늘 4시(ET) 이후 매수 직전까지 최고가 / 전일 종가 - 1
TEM 10/6은 prev1 +9.0%, spike +5.3%다. 패턴은 10/6(3분봉 기간)에서 만들었으므로 5분봉 기간(7/14~9/8)이 독립 확인이다.
사용: py blowoff.py <raw 폴더> <ev 폴더>"""
import sys
import numpy as np, pandas as pd
from analyze import load, cl_t


def add(ev, raw):
    p1 = {}
    for s in ev.sym.unique():
        c = pd.read_parquet(f"{raw}/1d_{s}.parquet").Close
        c.index = pd.to_datetime(c.index).date
        p1[s] = (c / c.shift(1) - 1).shift(1)            # 그날 기준 '전일 수익률'
    ev["prev1"] = [p1[s].get(d, np.nan) for s, d in zip(ev.sym, ev.date)]
    ev["spike"] = (1 + ev.d_prev) / (1 + ev.d_high) - 1   # 오늘 최고가 / 전일 종가 - 1
    return ev


def show(lab, on, off, e):
    _, t, _ = cl_t(e.index.isin(on.index).astype(float), e.ret.values, e.date.astype(str).values)
    print(f"  {lab}: {len(on)}건 평균 {on.ret.mean():+.2%} 승률 {(on.ret > 0).mean():.0%} 최악 {on.ret.min():+.1%} "
          f"| 나머지 {off.ret.mean():+.2%} {(off.ret > 0).mean():.0%} | t {t:+.2f}")


if __name__ == "__main__":
    raw, R = sys.argv[1], sys.argv[2]
    for name, e in (("3분봉 9/9~10/6", load(f"{R}/ev3.parquet")), ("5분봉 7/14~9/8", load(f"{R}/ev5.parquet", hi="2026-09-08"))):
        e = add(e, raw)
        print(f"\n===== {name}")
        for ses in ("정규", None):
            v = e if ses is None else e[e.ses == ses]
            v = v.dropna(subset=["prev1", "spike"])
            print(f" [{ses or '전체 세션'}] {len(v)}건")
            for lab, m in (("전일 +5% 이상", v.prev1 >= 0.05),
                           ("전일 +5% & 오늘 +3% 넘게 갔다가 반락", (v.prev1 >= 0.05) & (v.spike >= 0.03)),
                           ("오늘 +3% 넘게 갔다가 반락", v.spike >= 0.03)):
                show(lab, v[m], v[~m], v)
        tem = e[(e.sym == "TEM") & (e.date.astype(str) == "2026-10-06")]
        if len(tem):
            print("  TEM 10/6:", tem[["prev1", "spike", "ret"]].round(3).to_dict("records"))
        m = (e.ses == "정규") & (e.prev1 >= 0.05) & (e.spike >= 0.03)
        print("  패턴 해당 거래:", ", ".join(f"{r.sym} {r.t:%m/%d} {r.ret:+.1%}" for r in e[m].sort_values("t").itertuples()))
