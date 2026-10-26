"""미국 ETF 대응 백테스트: SOXL·SOXX·EWY × 1·2·4시간봉·일봉, 정규장, 강의 기본 규칙과 변형, 무작위 진입 대조.

- 봉 안 순서: 2·4시간봉·일봉은 1시간봉으로, 1시간봉 차트는 트레이딩뷰 전략 테스터 가정(시가에 가까운 극값 먼저)으로 정한다.
  하위 봉 안에서도 같은 가정으로 세 구간으로 나눈다(코인에서 5분봉 결과와 0.02R 안으로 같았다).
- 나쁜 날(`cache/hist/<종목>_bad.json`)을 신호~청산 사이에 낀 거래는 뺀다.
- 비용: 수수료 0(사용자 조건). 호가 스프레드는 편도 0.03%·0.1%로 따로 본다.
KORU는 1시간봉·일봉 분할 보정이 어긋나 EWY로 대신했다.
"""
from __future__ import annotations

import calendar
import datetime as dt
import json
import math
import pathlib
import random
import statistics
import sys
import zlib
from collections import defaultdict
from zoneinfo import ZoneInfo

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
SYMS = ["SOXL", "SOXX", "EWY"]
TFS = ["1h", "2h", "4h", "1d"]
KINDS = ["A-LQ", "A-TL", "A-CH", "A-MA", "B-OB", "C-FVG", "D-RT", "D1-BN", "E-CH4"]
DEFAULT = {"A-TL", "A-CH", "E-CH4", "D-RT"}       # 지표 대응 계획 기본 켜짐
ET = ZoneInfo("America/New_York")
P0 = calendar.timegm((2025, 1, 1, 0, 0, 0)) * 1000
P1 = calendar.timegm((2026, 1, 1, 0, 0, 0)) * 1000
P2 = calendar.timegm((2026, 7, 1, 0, 0, 0)) * 1000
PER = ("2025", "26H1", "26H2")
VARS = {"기본": bt.Var(), "1차전량": bt.Var(exit="full"), "2R고정": bt.Var(exit="r2"), "손익비≥1": bt.Var(minrr=1.0)}
FEES = (0.0, 0.0003, 0.001)


def period(t):
    return None if t < P0 else "2025" if t < P1 else "26H1" if t < P2 else "26H2"


def etd(ms):
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).astimezone(ET).date()


def touches_bad(r, bad):
    a, b = etd(r["t"]), etd(r["t_exit"])
    return any(a <= x <= b for x in bad)


def stats(x):
    if not x:
        return dict(n=0)
    R = [v["R"] for v in x]
    m = sum(R) / len(R)
    sd = statistics.pstdev(R) if len(R) > 1 else 0.0
    eq = pk = dd = 0.0
    for v in sorted(x, key=lambda v: v["t"]):
        eq += v["R"]
        pk = max(pk, eq)
        dd = min(dd, eq - pk)
    return dict(n=len(R), win=sum(1 for v in R if v > 0) / len(R), avg=m, t=(m / (sd / math.sqrt(len(R))) if sd else 0.0), sum=sum(R), dd=dd)


def run():
    rows, rnd = [], []
    excluded = defaultdict(int)
    for sym in SYMS:
        bad = {dt.date.fromisoformat(x) for x in json.loads((bt.HIST / f"{sym}_bad.json").read_text())}
        sub1h = bt.Sub(sym, "1h")
        for tf in TFS:
            eng = bt.Engine(sym, tf)
            sub = None if tf == "1h" else sub1h
            seen = set()
            for s in eng.setups:
                per = period(eng.b[s.i]["t"])
                if per is None:
                    continue
                base, _, mk = s.kind.partition(":")
                key = (base, s.i, s.d, mk)
                if key in seen:
                    continue
                seen.add(key)
                f = bt.fill(eng, s)
                if f is None:
                    continue
                for fee in FEES:
                    for vn, v in VARS.items():
                        r = bt.simulate(eng, s, v, fee, f, sub, "tv")
                        if r is None:
                            continue
                        if touches_bad(r, bad):
                            excluded[(sym, tf)] += (fee == 0 and vn == "기본")
                            continue
                        r.update(var=vn, per=per, base=base, entry="mkt" if mk else "lim", fee=fee)
                        rows.append(r)
            # 무작위 진입(같은 청산 규칙, 손절 폭은 그 셋업들의 ATR 배수 분포, 2025~2026)
            idx = [i for i in range(eng.n - 2) if period(eng.b[i]["t"])]
            risk_atr = defaultdict(list)
            for s in eng.setups:
                if ":" in s.kind or not period(eng.b[s.i]["t"]):
                    continue
                f = bt.fill(eng, s)
                if f is None:
                    continue
                fj, px, _ = f
                a = eng.atr[fj - 1]
                if s.kind == "D1-BN":
                    ru = 0.6 * a                                   # 선 0.5 ATR + 버퍼 0.1 ATR 밖
                elif s.kind == "E-CH4":
                    ru = abs(px - s.line.sub(fj)) + 1.1 * a        # 반대선 1 ATR + 버퍼 0.1 ATR 밖
                else:
                    ru = abs(px - (s.stop - s.d * 0.1 * a))
                if a > 0 and ru > 0:
                    risk_atr[s.kind].append((s.d, ru / a))
            for k, lst in risk_atr.items():
                rng = random.Random(zlib.crc32(f"{sym}|{tf}|{k}".encode()))
                for _ in range(3 * len(lst)):
                    i = rng.choice(idx)
                    d, katr = rng.choice(lst)
                    px = eng.b[i + 1]["o"]
                    so = bt.Setup("rnd", i, d, None, px - d * katr * eng.atr[i], i + 1, i + 1, -1, px)
                    rr = bt.simulate(eng, so, bt.Var(buf=0.0), 0.0, (i + 1, px, True), sub, "tv")
                    if rr is not None and not touches_bad(rr, bad):
                        rr.update(base=k, per=period(eng.b[i]["t"]))
                        rnd.append(rr)
            print(sym, tf, "셋업", len(eng.setups), "봉", eng.n, flush=True)
    return rows, rnd, excluded


def fmt(st):
    if st.get("n", 0) == 0:
        return "—"
    return f"{st['avg']:+.2f} (t{st['t']:+.1f}, {st['n']})"


def tables(rows, rnd, excluded):
    L = []
    base = [r for r in rows if r["var"] == "기본" and r["entry"] == "lim" and r["fee"] == 0]
    L.append("### 1. 강의 기본 규칙, 수수료 0 (3종목·4시간봉 합침) — 평균 R (t값, 거래 수)")
    L.append("| 셋업 | 2025 | 2026 상반기 | 2026 하반기 | 무작위 진입 2026 |")
    L.append("|---|---|---|---|---|")
    for k in KINDS:
        cells = [fmt(stats([r for r in base if r["base"] == k and r["per"] == p])) for p in PER]
        rz = fmt(stats([r for r in rnd if r["base"] == k and r["per"] in ("26H1", "26H2")]))
        L.append(f"| {k} | " + " | ".join(cells) + f" | {rz} |")
    L.append("")
    L.append("### 2. 종목·시간봉별 (기본 규칙, 수수료 0, 9개 셋업 합침) — 2026 평균 R (거래 수) / 합계 R")
    L.append("| 종목 | " + " | ".join(TFS) + " |")
    L.append("|---|" + "---|" * len(TFS))
    for sym in SYMS:
        cells = []
        for tf in TFS:
            st = stats([r for r in base if r["sym"] == sym and r["tf"] == tf and r["per"] in ("26H1", "26H2")])
            cells.append("—" if not st.get("n") else f"{st['avg']:+.2f} ({st['n']}) / {st['sum']:+.0f}R")
        L.append(f"| {sym} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("### 3. 누적 (기본 규칙) — 합계 R · 평균 R · 최대 낙폭 R")
    L.append("| 묶음 | 비용 | 2025 | 2026 |")
    L.append("|---|---|---|---|")
    groups = {
        "9개 셋업 전부": lambda r: True,
        "지표 기본 4종": lambda r: r["base"] in DEFAULT,
        "채널 4포인트": lambda r: r["base"] == "E-CH4",
        "채널 4포인트·일봉": lambda r: r["base"] == "E-CH4" and r["tf"] == "1d",
        "일봉 9개 셋업": lambda r: r["tf"] == "1d",
    }
    for gn, fl in groups.items():
        for fee in FEES:
            x = [r for r in rows if r["var"] == "기본" and r["entry"] == "lim" and r["fee"] == fee and fl(r)]
            a, b2 = stats([r for r in x if r["per"] == "2025"]), stats([r for r in x if r["per"] in ("26H1", "26H2")])
            def c(st):
                return "—" if not st.get("n") else f"{st['sum']:+.1f}R · {st['avg']:+.2f} · 낙폭 {st['dd']:.1f}R ({st['n']}건)"
            L.append(f"| {gn} | 편도 {fee*100:.2f}% | {c(a)} | {c(b2)} |")
    L.append("")
    L.append("### 4. 변형 (수수료 0, 9개 셋업 각각) — 2026 상반기 → 하반기 평균 R")
    L.append("| 셋업 | " + " | ".join(VARS) + " |")
    L.append("|---|" + "---|" * len(VARS))
    for k in KINDS:
        cells = []
        for vn in VARS:
            x = [r for r in rows if r["var"] == vn and r["entry"] == "lim" and r["fee"] == 0 and r["base"] == k]
            a, b2 = stats([r for r in x if r["per"] == "26H1"]), stats([r for r in x if r["per"] == "26H2"])
            cells.append("—" if not a.get("n") else f"{a['avg']:+.2f}→{b2.get('avg', float('nan')):+.2f}")
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("나쁜 날 때문에 뺀 거래(기본 규칙, 수수료 0): " + ", ".join(f"{s} {t} {n}건" for (s, t), n in excluded.items() if n))
    return "\n".join(L)


if __name__ == "__main__":
    rows, rnd, excluded = run()
    OUT.mkdir(exist_ok=True)
    (OUT / "bt_us_rows.json").write_text(json.dumps(rows))
    (OUT / "bt_us_rnd.json").write_text(json.dumps(rnd))
    txt = tables(rows, rnd, excluded)
    (OUT / "bt_us_tables.md").write_text(txt, encoding="utf-8")
    print(txt)
