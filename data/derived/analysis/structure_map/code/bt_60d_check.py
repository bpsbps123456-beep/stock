"""최근 60일 미국 AI 115 — 40분봉 플러스가 우연인지 점검 (2026-10-23).

같은 60일 5분봉으로 만든 20·30·40·45·50·60분봉(이웃 시간봉)을 같은 규칙으로 견준다. 40분만 튀면 우연, 이웃도 비슷하면 구간 효과다.
그 밖에 특이값(갭으로 손절선을 건너뛴 거래 등)·종목 쏠림·셋업별·매수/매도별·한국 주문 시간(08~24시) 안에서 걸 수 있는 거래만의 성적을 본다.
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402
import bt_2026 as y  # noqa: E402
import bt_60d as d6  # noqa: E402

OUT = d6.OUT
NEI = ["20m", "30m", "40m", "45m", "50m", "1hc"]
for k in ("20m", "30m", "45m", "50m"):
    bt.MS[k] = int(k[:-1]) * 60_000


def robust(xs):
    R = sorted(x["R0"] for x in xs)
    if not R:
        return "—"
    k = max(1, len(R) // 100)
    trim = R[k:-k] if len(R) > 2 * k else R
    core = [v for v in R if abs(v) <= 10]
    return (f"중앙값 {statistics.median(R):+.2f} · 위아래 1% 뺀 평균 {sum(trim)/len(trim):+.2f} · |R|>10 {len(R)-len(core)}건 뺀 평균 "
            f"{(sum(core)/len(core) if core else float('nan')):+.2f}")


def main():
    w0, mid, days = d6.window()
    cache, bads = {}, {}
    for sym in d6.AI:
        bset = set()
        for fn in (f"{sym}_bad.json", f"{sym}_bad5m.json"):
            p = bt.HIST / fn
            if p.exists():
                bset |= {dt.date.fromisoformat(x) for x in json.loads(p.read_text())}
        bads[sym] = bset
        sub = bt.Sub(sym, "5m")
        for tf in NEI:
            cache[(sym, tf)] = d6.build(sym, tf, True, 0.0003, sub, w0)
    print("계산 끝", flush=True)

    def trades(tf, syms=None):
        out = []
        for sym in (syms or d6.AI):
            v = cache.get((sym, tf))
            if v:
                out += [x for x in y.follow(v[0], set(y.KINDS), 1.0) if x["t"] >= w0 and not y.crosses_bad(x, bads[sym])]
        return out

    def rnd(tf):
        out = []
        for sym in d6.AI:
            v = cache.get((sym, tf))
            if v:
                out += [x for x in v[1] if x["t"] >= w0 and not y.crosses_bad(x, bads[sym])]
        return out

    L = ["# 최근 60일 미국 AI 115 — 40분봉 점검", "",
         f"창 {days[0]} ~ {days[-1]} ({len(days)}거래일), 앞뒤 경계 {days[len(days) // 2]}. 모든 시간봉을 같은 60일 5분봉으로 만들었다(09:30 기준). 비용 0.", "",
         "## 이웃 시간봉", "",
         "| 시간봉 | 60일 전체 | 앞 절반 → 뒤 절반 | 무작위 진입 | 한국 08~24시에 걸 수 있는 거래만 | 특이값 점검 |",
         "|---|---|---|---|---|---|"]
    for tf in NEI:
        xs = trades(tf)
        s = y.st(xs)
        a, b = y.st([x for x in xs if x["t"] < mid]), y.st([x for x in xs if x["t"] >= mid])
        r = y.st(rnd(tf))
        ok = y.st([x for x in xs if not x["miss"]])
        L.append(f"| {tf} | {y.cell(s)} | {a['avg']:+.2f} ({a['n']}) → {b['avg']:+.2f} ({b['n']}) | {r['avg']:+.2f} ({r['n']}) "
                 f"| {ok['avg']:+.2f} ({ok['n']}건, t{ok['t']:+.1f}) | {robust(xs)} |")
    xs = trades("40m")
    L += ["", "## 40분봉 나눠 보기", "", "| 나눔 | 성적 |", "|---|---|"]
    for k in y.KINDS:
        L.append(f"| 셋업 {k} | {y.cell(y.st([x for x in xs if x['kind'] == k]))} |")
    for d, nm in ((1, "매수"), (-1, "매도")):
        L.append(f"| {nm} | {y.cell(y.st([x for x in xs if x['d'] == d]))} |")
    by = defaultdict(list)
    for x in xs:
        by[x["sym"]].append(x["R0"])
    top = sorted(by, key=lambda s: -sum(by[s]))[:5]
    rest = [x for x in xs if x["sym"] not in top]
    L.append(f"| 합계 R 상위 5종목 뺀 나머지 ({', '.join(f'{s} {sum(by[s]):+.1f}R' for s in top)}) | {y.cell(y.st(rest))} |")
    pos = sum(1 for s in by if sum(by[s]) > 0)
    L.append(f"| 종목 수 (합계 R 플러스 / 거래 있는 종목) | {pos} / {len(by)} |")
    semi = [x for x in xs if x["sym"] in d6.SEMI]
    L.append(f"| 반도체 10 | {y.cell(y.st(semi))} |")
    L.append(f"| 반도체 10 뺀 105 | {y.cell(y.st([x for x in xs if x['sym'] not in d6.SEMI]))} |")
    txt = "\n".join(L)
    (OUT / "d60_check.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
