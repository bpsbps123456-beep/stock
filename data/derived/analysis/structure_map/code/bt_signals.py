"""지표 '매수·매도·익절·손절' 표시를 그대로 따랐을 때의 성적 — 시간봉 비교 (2026-10-22 사용자 요청).

지표 기본 규칙(구조물 지도 매수·매도 표시):
  셋업: 추세선·채널 거짓 돌파 뒤 리테스트(A-TL·A-CH), 채널 4포인트(E-CH4, 닿은 봉에 진입), 추세선 돌파 뒤 리테스트(D-RT).
  손절: 거짓 돌파 극점·돌파 파동 극점 0.1 ATR 밖, 채널은 반대선 1.1 ATR 밖.
  익절: 가장 가까운 반대 구조물(채널은 반대편)에서 전량. 손익비(목표 거리 ÷ 손절 폭) 1 미만이면 거른다.
  한 번에 한 포지션: 보유 중에는 새 신호를 무시하고, 대기 중인 지정가는 새 신호로 바꾼다.
시간봉: 코인 40분·1·2시간·160분·4시간(봉 안 순서는 5분봉), 미국 ETF 1·2·4시간·일봉(1시간봉·트레이딩뷰 가정).
판정: 2026 상반기로 고르고 하반기로 확인, 2025(2~12월)는 보조. 비용: 코인 편도 0·0.05%, 미국 0·0.03%.
"""
from __future__ import annotations

import calendar
import datetime as dt
import json
import math
import pathlib
import statistics
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent.parent / "results"
KINDS = {"A-TL", "A-CH", "E-CH4", "D-RT"}
VAR = bt.Var(buf=0.1, exit="full", minrr=1.0, ch_close=False)
GROUPS = {
    "코인": (["BTCUSDT", "ETHUSDT", "SOLUSDT"], ["40m", "1h", "2h", "160m", "4h"], 0.0005),
    "미국ETF": (["SOXL", "SOXX", "EWY"], ["1h", "2h", "4h", "1d"], 0.0003),
}
ET = ZoneInfo("America/New_York")
P0 = calendar.timegm((2025, 2, 1, 0, 0, 0)) * 1000      # 40분·160분봉은 5분봉(2024-12-25~)으로 만들어 한 달 준비 기간을 둔다
P1 = calendar.timegm((2026, 1, 1, 0, 0, 0)) * 1000
P2 = calendar.timegm((2026, 7, 1, 0, 0, 0)) * 1000
PER = ("2025", "26H1", "26H2")


def period(t):
    return None if t < P0 else "2025" if t < P1 else "26H1" if t < P2 else "26H2"


def follow(eng, sub, path, fee, bad):
    """지표처럼 한 번에 한 포지션으로 신호를 따라간다."""
    cands = []
    for s in eng.setups:
        base, _, mk = s.kind.partition(":")
        if mk or base not in KINDS:
            continue
        f = bt.fill(eng, s)
        if f is None:
            continue
        r = bt.simulate(eng, s, VAR, fee, f, sub, path)
        if r is None:
            continue
        fj = f[0]
        t_req = fj if base == "E-CH4" else s.i          # 채널 4포인트는 닿은 봉에서 신호와 진입이 같이 난다
        cands.append((t_req, fj, fj + r["hold"], r))
    cands.sort(key=lambda x: (x[0], x[1]))
    trades, pending, active_exit = [], None, -1

    def settle(upto):
        nonlocal pending, active_exit
        if pending is not None and pending[1] <= upto:
            trades.append(pending[3])
            active_exit = pending[2]
            pending = None

    for c in cands:
        settle(c[0])
        if c[0] < active_exit:            # 보유 중이면 새 신호 무시
            continue
        pending = c                       # 대기 지정가는 새 신호로 교체
        if c[1] == c[0]:
            settle(c[0])
    settle(10 ** 12)
    if bad:
        def ok(r):
            a = dt.datetime.fromtimestamp(r["t_fill"] / 1000, dt.timezone.utc).astimezone(ET).date()
            b = dt.datetime.fromtimestamp(r["t_exit"] / 1000, dt.timezone.utc).astimezone(ET).date()
            return not any(a <= x <= b for x in bad)
        trades = [r for r in trades if ok(r)]
    return trades


def stats(x):
    if not x:
        return dict(n=0)
    R = [r["R"] for r in x]
    m = sum(R) / len(R)
    sd = statistics.pstdev(R) if len(R) > 1 else 0.0
    eq = pk = dd = 0.0
    for v in R:
        eq += v
        pk = max(pk, eq)
        dd = min(dd, eq - pk)
    return dict(n=len(R), win=sum(1 for v in R if v > 0) / len(R), avg=m, t=(m / (sd / math.sqrt(len(R))) if sd else 0.0), sum=sum(R), dd=dd)


def main():
    rows = []
    for g, (syms, tfs, fee_hi) in GROUPS.items():
        us = g == "미국ETF"
        for sym in syms:
            bad = {dt.date.fromisoformat(x) for x in json.loads((bt.HIST / f"{sym}_bad.json").read_text())} if us else set()
            sub_all = bt.Sub(sym, "1h" if us else "5m")
            for tf in tfs:
                eng = bt.Engine(sym, tf)
                sub = None if (us and tf == "1h") else sub_all
                path = "tv" if us else "stop"
                for fee in (0.0, fee_hi):
                    for r in follow(eng, sub, path, fee, bad):
                        per = period(r["t_fill"])
                        if per:
                            rows.append(dict(g=g, sym=sym, tf=tf, fee=fee, per=per, R=r["R"], t=r["t_fill"], kind=r["kind"], why=r["why"]))
                print(g, sym, tf, flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "signals_rows.json").write_text(json.dumps(rows))
    L = []
    for g, (syms, tfs, fee_hi) in GROUPS.items():
        for fee in (0.0, fee_hi):
            L.append(f"### {g} — 비용 편도 {fee*100:.2f}% — 승률 · 평균 R (t) · 거래 수 · 합계 R")
            L.append("| 시간봉 | 2025(2~12월) | 2026 상반기 | 2026 하반기 | 2026 합계·최대 낙폭 |")
            L.append("|---|---|---|---|---|")
            for tf in tfs:
                xs = [r for r in rows if r["g"] == g and r["tf"] == tf and r["fee"] == fee]
                cells = []
                for p in PER:
                    s = stats([r for r in xs if r["per"] == p])
                    cells.append("—" if not s.get("n") else f"{s['win']*100:.0f}% · {s['avg']:+.2f} (t{s['t']:+.1f}) · {s['n']}건")
                s26 = stats(sorted([r for r in xs if r["per"] != "2025"], key=lambda r: r["t"]))
                tail = "—" if not s26.get("n") else f"{s26['sum']:+.1f}R · 낙폭 {s26['dd']:.1f}R"
                L.append(f"| {tf} | " + " | ".join(cells) + f" | {tail} |")
            L.append("")
    txt = "\n".join(L)
    (OUT / "signals_tables.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
