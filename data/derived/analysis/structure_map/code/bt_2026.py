"""2026년만으로 매수·매도 규칙과 시간봉 고르기 (2026-10-22 사용자 요청 "2026년 기준으로만, 이것도 트렌드인 것 같다", "SOXX·EWY도", "MU 같은 개별 종목도").

AGENTS.md 연구 보고 규칙: 2026 상반기(1~6월)로 고르고 하반기(7월~10/21)로 확인한다.
묶음: 코인(BTC·ETH·SOL — 40분·1·2시간·160분·4시간, 봉 안 순서는 5분봉)
      미국 ETF(SOXL·SOXX·EWY), 미국 개별(watchlist.py AI 반도체 6 + 메모리 4) — 1·2·4시간·일봉 정규장(봉 안 순서는 1시간봉, 1시간봉은 트레이딩뷰 가정).
      미국 40분·160분봉은 못 잰다(yfinance 5분봉은 최근 60일뿐).
고르는 것: 시간봉 × 켤 셋업 조합(A-TL·A-CH·E-CH4·D-RT 중 15가지) × 손익비 하한(0·1·1.5·2) — 코인 300가지, 미국 묶음마다 240가지.
익절은 가장 가까운 반대 구조물에서 전량, 종목마다 한 번에 한 포지션(지표와 같음). 신호별 결과는 한 번만 계산하고 조합마다 거르기만 다시 한다.
지표 설정은 봉 단위 그대로다(피벗 5봉 등). 질문이 '같은 지표를 어느 차트에 걸까'라서 시간 단위로 맞추지 않는다.
대조군: 같은 청산 규칙(전량·손익비 1 이상)·같은 손절 폭 분포(ATR 배수)로 2026의 아무 봉 다음 시가에 진입.
미국 실전 점검: 사용자 주문 가능 시간은 한국시간 08~24시(docs/desk_system.md §1). 리테스트 지정가 신호가 00~08시에 확정되고
  아침 8시 전에 체결됐으면 '못 거는 거래'로 센다(채널 4P는 미리 걸어 두는 지정가라 뺀다).
"""
from __future__ import annotations

import calendar
import datetime as dt
import itertools
import json
import math
import pathlib
import random
import statistics
import sys
import zlib
from collections import Counter
from zoneinfo import ZoneInfo

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent.parent / "results"
KINDS = ["A-TL", "A-CH", "E-CH4", "D-RT"]
ALL = "+".join(KINDS)
RRS = [0.0, 1.0, 1.5, 2.0]
GROUPS = {
    "코인": dict(syms=["BTCUSDT", "ETHUSDT", "SOLUSDT"], tfs=["40m", "1h", "2h", "160m", "4h"], sub="5m", path="stop", fee=0.0005, us=False, nmin=(30, 15)),
    "미국ETF": dict(syms=["SOXL", "SOXX", "EWY"], tfs=["1h", "2h", "4h", "1d"], sub="1h", path="tv", fee=0.0003, us=True, nmin=(15, 8)),
    "미국개별": dict(syms=["MU", "SNDK", "WDC", "STX", "NVDA", "AVGO", "AMD", "MRVL", "CRDO", "ALAB"], tfs=["1h", "2h", "4h", "1d"],
                 sub="1h", path="tv", fee=0.0003, us=True, nmin=(30, 15)),
}
VAR0 = bt.Var(buf=0.1, exit="full", minrr=0.0, ch_close=False)
VARR = bt.Var(buf=0.0, exit="full", minrr=1.0, ch_close=False)
P1 = calendar.timegm((2026, 1, 1, 0, 0, 0)) * 1000
P2 = calendar.timegm((2026, 7, 1, 0, 0, 0)) * 1000
ET = ZoneInfo("America/New_York")
KST = ZoneInfo("Asia/Seoul")


def half(t):
    return None if t < P1 else 1 if t < P2 else 2


def ldt(ms, tz):
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).astimezone(tz)


def bar_close(t, tf, us):
    if not us:
        return t + bt.MS[tf]
    c = ldt(t, ET).replace(hour=16, minute=0, second=0, microsecond=0)
    return min(t + bt.MS[tf], int(c.timestamp() * 1000))


def asleep_miss(sig_close, t_fill):
    k = ldt(sig_close, KST)
    if k.hour >= 8:
        return False
    return t_fill < int(k.replace(hour=8, minute=0, second=0, microsecond=0).timestamp() * 1000)


def crosses_bad(x, bad):
    if not bad:
        return False
    a, b = ldt(x["t"], ET).date(), ldt(x["te"], ET).date()
    return any(a <= d <= b for d in bad)


def risk_atr(eng, s, f):
    fj, px, _ = f
    a = eng.atr[fj - 1]
    ru = abs(px - s.line.sub(fj)) + 1.1 * a if s.kind == "E-CH4" else abs(px - (s.stop - s.d * 0.1 * a))
    return (s.d, ru / a) if a > 0 and ru > 0 else None


def build(sym, tf, cfg, sub):
    eng = bt.Engine(sym, tf)
    cands, pool = [], []
    for s in eng.setups:
        base, _, mk = s.kind.partition(":")
        if mk or base not in KINDS:
            continue
        f = bt.fill(eng, s)
        if f is None:
            continue
        r = bt.simulate(eng, s, VAR0, cfg["fee"], f, sub, cfg["path"])
        if r is None:
            continue
        fj = f[0]
        if half(r["t_fill"]):
            ra = risk_atr(eng, s, f)
            if ra:
                pool.append(ra)
        sig = bar_close(eng.b[s.i]["t"], tf, cfg["us"])
        cands.append(dict(sym=sym, kind=base, t_req=fj if base == "E-CH4" else s.i, fj=fj, xj=fj + r["hold"], rr1=r["rr1"],
                          R0=r["Rg"], R5=r["R"], t=r["t_fill"], te=r["t_exit"],
                          miss=bool(cfg["us"] and base != "E-CH4" and asleep_miss(sig, r["t_fill"]))))
    cands.sort(key=lambda c: (c["t_req"], c["fj"]))
    # 무작위 진입 대조군
    rnd = []
    idx = [i for i in range(eng.n - 2) if half(eng.b[i + 1]["t"])]
    if pool and idx:
        rng = random.Random(zlib.crc32(f"{sym}|{tf}|2026".encode()))
        for _ in range(3 * len(pool)):
            i = rng.choice(idx)
            d, k = rng.choice(pool)
            px = eng.b[i + 1]["o"]
            so = bt.Setup("rnd", i, d, None, px - d * k * eng.atr[i], i + 1, i + 1, -1, px)
            rr = bt.simulate(eng, so, VARR, 0.0, (i + 1, px, True), sub, cfg["path"])
            if rr is not None:
                rnd.append(dict(t=rr["t_fill"], te=rr["t_exit"], R0=rr["Rg"]))
    return cands, rnd


def follow(cands, kinds, minrr):
    """지표처럼 한 번에 한 포지션: 보유 중 새 신호 무시, 대기 지정가는 새 신호로 교체."""
    trades, pending, active_exit = [], None, -1
    for c in cands:
        if c["kind"] not in kinds or c["rr1"] < minrr:
            continue
        if pending is not None and pending["fj"] <= c["t_req"]:
            trades.append(pending)
            active_exit = pending["xj"]
            pending = None
        if c["t_req"] < active_exit:
            continue
        pending = c
        if c["fj"] == c["t_req"]:
            trades.append(c)
            active_exit = c["xj"]
            pending = None
    if pending is not None:
        trades.append(pending)
    return trades


def st(xs, key="R0"):
    if not xs:
        return dict(n=0, avg=float("nan"), sum=0.0, win=float("nan"), t=0.0, dd=0.0)
    xs = sorted(xs, key=lambda x: x["t"])
    R = [x[key] for x in xs]
    m = sum(R) / len(R)
    sd = statistics.pstdev(R) if len(R) > 1 else 0.0
    eq = pk = dd = 0.0
    for v in R:
        eq += v
        pk = max(pk, eq)
        dd = min(dd, eq - pk)
    return dict(n=len(R), avg=m, sum=sum(R), win=sum(1 for v in R if v > 0) / len(R), t=(m / (sd / math.sqrt(len(R))) if sd else 0.0), dd=dd)


def cell(s):
    return "—" if not s["n"] else f"{s['win']*100:.0f}% · {s['avg']:+.2f} (t{s['t']:+.1f}) · {s['n']}건"


def main():
    cache, rnds, bads = {}, {}, {}
    for g, cfg in GROUPS.items():
        for sym in cfg["syms"]:
            bads[sym] = {dt.date.fromisoformat(x) for x in json.loads((bt.HIST / f"{sym}_bad.json").read_text())} if cfg["us"] else set()
            sub_all = bt.Sub(sym, cfg["sub"])
            for tf in cfg["tfs"]:
                sub = None if tf == cfg["sub"] else sub_all
                cache[(sym, tf)], rnds[(sym, tf)] = build(sym, tf, cfg, sub)
            print(g, sym, flush=True)

    def run(g, tf, kinds, minrr):
        h = {1: [], 2: []}
        for sym in GROUPS[g]["syms"]:
            for x in follow(cache[(sym, tf)], kinds, minrr):
                p = half(x["t"])
                if p and not crosses_bad(x, bads[sym]):
                    h[p].append(x)
        return h

    grid = []
    for g, cfg in GROUPS.items():
        for tf in cfg["tfs"]:
            for k in range(1, len(KINDS) + 1):
                for combo in itertools.combinations(KINDS, k):
                    for rr in RRS:
                        h = run(g, tf, set(combo), rr)
                        grid.append(dict(g=g, tf=tf, combo="+".join(combo), rr=rr, h1=st(h[1]), h2=st(h[2]), h1f=st(h[1], "R5"), h2f=st(h[2], "R5")))
    OUT.mkdir(exist_ok=True)
    (OUT / "y2026_grid.json").write_text(json.dumps(grid))

    L = ["# 2026년만으로 본 시간봉 (구조물 지도 매수·매도 규칙)", "",
         "상반기(1~6월)로 고르고 하반기(7월~10/21)로 확인. 칸 = 승률 · 평균 R (t) · 거래 수. R = 손절 폭 1개. 비용 0이 기본(미국은 사용자 수수료 0원).", ""]
    for g, cfg in GROUPS.items():
        L += [f"## {g} ({'·'.join(cfg['syms'])})", "",
              "### 지표 기본값(4셋업 전부, 손익비 1 이상)", "",
              "| 시간봉 | 상반기 | 하반기 | 2026 합계 · 최대 낙폭 | 비용 넣으면 상반기 → 하반기 | 무작위 진입 상반기 → 하반기 |" + (" 새벽이라 못 거는 거래 · 뺀 평균 R |" if cfg["us"] else ""),
              "|---|---|---|---|---|---|" + ("---|" if cfg["us"] else "")]
        for tf in cfg["tfs"]:
            h = run(g, tf, set(KINDS), 1.0)
            s1, s2, sa = st(h[1]), st(h[2]), st(h[1] + h[2])
            f1, f2 = st(h[1], "R5"), st(h[2], "R5")
            rz = {1: [], 2: []}
            for sym in cfg["syms"]:
                for x in rnds[(sym, tf)]:
                    p = half(x["t"])
                    if p and not crosses_bad(x, bads[sym]):
                        rz[p].append(x)
            r1, r2 = st(rz[1]), st(rz[2])
            row = (f"| {tf} | {cell(s1)} | {cell(s2)} | {sa['sum']:+.1f}R · {sa['dd']:.1f}R | {f1['avg']:+.2f} → {f2['avg']:+.2f} (편도 {cfg['fee']*100:.2f}%) "
                   f"| {r1['avg']:+.2f} → {r2['avg']:+.2f} ({r1['n']}·{r2['n']}건) |")
            if cfg["us"]:
                allx = h[1] + h[2]
                m = sum(1 for x in allx if x["miss"])
                keep = st([x for x in allx if not x["miss"]])
                row += f" {m}/{len(allx)} · {keep['avg']:+.2f} |"
            L.append(row)
        L.append("")
        n1, n2 = cfg["nmin"]
        ok = sorted([r for r in grid if r["g"] == g and r["h1"]["n"] >= n1], key=lambda r: -r["h1"]["avg"])
        L += [f"### 상반기 평균 R 상위 5 (조합 {sum(1 for r in grid if r['g'] == g)}가지 중 상반기 {n1}건 이상) → 하반기", "",
              "| 시간봉 | 셋업 | 손익비 하한 | 상반기 | 하반기 |", "|---|---|---|---|---|"]
        for r in ok[:5]:
            L.append(f"| {r['tf']} | {r['combo']} | {r['rr']:.1f} | {cell(r['h1'])} | {cell(r['h2'])} |")
        both = [r for r in grid if r["g"] == g and r["h1"]["n"] >= n1 and r["h2"]["n"] >= n2 and r["h1"]["avg"] > 0 and r["h2"]["avg"] > 0]
        L += ["", f"상·하반기 모두 플러스(상반기 {n1}·하반기 {n2}건 이상): {len(both)}가지 — 시간봉별 {dict(Counter(r['tf'] for r in both))}", ""]
        if cfg["us"]:
            L += ["### 종목별 2026(기본값) — 거래 수 · 평균 R", "", "| 종목 | " + " | ".join(cfg["tfs"]) + " |", "|---|" + "---|" * len(cfg["tfs"])]
            for sym in cfg["syms"]:
                cs = []
                for tf in cfg["tfs"]:
                    xs = [x for x in follow(cache[(sym, tf)], set(KINDS), 1.0) if half(x["t"]) and not crosses_bad(x, bads[sym])]
                    s = st(xs)
                    cs.append("—" if not s["n"] else f"{s['n']} · {s['avg']:+.2f}")
                L.append(f"| {sym} | " + " | ".join(cs) + " |")
            L.append("")
    txt = "\n".join(L)
    (OUT / "y2026_tables.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
