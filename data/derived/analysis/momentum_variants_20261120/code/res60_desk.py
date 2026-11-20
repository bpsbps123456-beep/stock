"""잔차 모멘텀 60일(composite 갈래 c RES60, 상반기 선택 계열)을 데스크 구조(TIME, n7, buffer5, TIME 비중, 상한)에 점수만 바꿔 넣은 확인.
결과: results/res60_desk.json
"""
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "code"))
sys.path.insert(0, str(BASE / "branches/composite"))
import engine as E  # noqa: E402
import composite as C  # noqa: E402


def months(e, res):
    out = {}
    for a, b in (("2026-07-01", "2026-07-31"), ("2026-08-01", "2026-08-31"), ("2026-09-01", "2026-09-30"),
                 ("2026-10-01", "2026-10-31"), ("2026-11-01", E.LAST)):
        out[a[:7]] = round(e.span(res, a, b), 4)
    return out


def main():
    e = E.Engine()
    res60 = C.residual(e, 60)
    cfgs = {
        "현행(120일 수익)": dict(score=e.ret(120), weight="time"),
        "잔차 60일 · TIME 비중": dict(score=res60, weight="time"),
        "잔차 60일 · 같은 비중": dict(score=res60, weight="equal"),
        "120일 수익 · 같은 비중(대조)": dict(score=e.ret(120), weight="equal"),
        "60일 수익 · TIME 비중(대조)": dict(score=e.ret(60), weight="time"),
    }
    out = {}
    for nm, c in cfgs.items():
        r = e.run(c["score"], universe="TIME", n=7, buffer=5, weight=c["weight"], caps=True)
        m = e.metrics(r)
        d1 = e.metrics(e.run(c["score"], universe="TIME", n=7, buffer=5, weight=c["weight"], caps=True, delay=1))
        c10 = e.metrics(e.run(c["score"], universe="TIME", n=7, buffer=5, weight=c["weight"], caps=True, cost_bps=10))
        out[nm] = {**{k: round(v, 4) for k, v in m.items()}, "H1_MDD": round(e.mdd(r, "2026-01-02", "2026-06-30"), 4),
                   "delay1_H2": round(d1["H2"], 4), "delay1_RECENT": round(d1["RECENT"], 4), "cost10_H2": round(c10["H2"], 4),
                   "months_H2": months(e, r),
                   "last_holdings": dict(sorted(list(r.holdings.items()))[-1][1]) if r.holdings else {}}
        out[nm]["last_holdings"] = {k: round(v, 3) for k, v in out[nm]["last_holdings"].items()}
    q = e.bench("QQQ")
    out["QQQ"] = {**{k: round(v, 4) for k, v in e.metrics(q).items()}, "months_H2": months(e, q)}
    (BASE / "results/res60_desk.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for nm, v in out.items():
        print(nm, {k: v.get(k) for k in ("Y2026", "H1", "H1_MDD", "H2", "RECENT", "MDD2026", "MDD_H2", "delay1_H2", "cost10_H2")})
        print("   months", v["months_H2"])
    print("RES60 last", out["잔차 60일 · TIME 비중"]["last_holdings"])
    print("MOM last", out["현행(120일 수익)"]["last_holdings"])


if __name__ == "__main__":
    main()
