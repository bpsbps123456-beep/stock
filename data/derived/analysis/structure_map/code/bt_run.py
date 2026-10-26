"""대응 백테스트 실행: BTC·ETH·SOL × 1·2·4시간봉, 강의 기본 규칙 + 한 가지씩 바꾼 변형, 기간 2025 / 2026 상반기 / 2026 하반기."""
from __future__ import annotations

import calendar
import json
import math
import pathlib
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent.parent / "results"
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TFS = ["1h", "2h", "4h"]
FEE = 0.0005        # 편도 0.05% (바이낸스 선물 시장가 수준)
P0 = calendar.timegm((2025, 1, 1, 0, 0, 0)) * 1000
P1 = calendar.timegm((2026, 1, 1, 0, 0, 0)) * 1000
P2 = calendar.timegm((2026, 7, 1, 0, 0, 0)) * 1000
PER = ("2025", "26H1", "26H2")
KINDS = ["A-LQ", "A-TL", "A-CH", "A-MA", "B-OB", "C-FVG", "D-RT", "D1-BN", "E-CH4"]
VARS = {
    "기본": bt.Var(),
    "버퍼0": bt.Var(buf=0.0),
    "버퍼0.3": bt.Var(buf=0.3),
    "1차전량": bt.Var(exit="full"),
    "2R고정": bt.Var(exit="r2"),
    "손익비≥1": bt.Var(minrr=1.0),
    "손익비≥2": bt.Var(minrr=2.0),
    "봉마감손절": bt.Var(stopmode="close"),
}


def period(t):
    if t < P0:
        return None
    return "2025" if t < P1 else "26H1" if t < P2 else "26H2"


def stats(rs):
    n = len(rs)
    if n == 0:
        return dict(n=0)
    R = [x["R"] for x in rs]
    m = sum(R) / n
    sd = statistics.pstdev(R) if n > 1 else 0.0
    pos, neg = sum(x for x in R if x > 0), -sum(x for x in R if x < 0)
    return dict(n=n, win=sum(1 for x in R if x > 0) / n, avg=m, t=(m / (sd / math.sqrt(n)) if sd > 0 else 0.0),
                sum=sum(R), pf=(pos / neg if neg > 0 else float("inf")), cost=sum(x["cost"] for x in rs) / n,
                hold=sum(x["hold"] for x in rs) / n)


def run(fee=FEE, use_sub=True):
    rows = []                       # 모든 결과 (변형 이름 포함)
    fills = defaultdict(lambda: [0, 0])
    for sym in SYMS:
        sub = bt.Sub(sym) if use_sub else None
        for tf in TFS:
            eng = bt.Engine(sym, tf)
            seen = set()
            for s in eng.setups:
                per = period(eng.b[s.i]["t"])
                if per is None:
                    continue
                base, _, mk = s.kind.partition(":")
                key = (base, s.i, s.d, mk)
                if key in seen:     # 같은 봉에서 유동성 여러 개를 함께 쓸면 하나만
                    continue
                seen.add(key)
                f = bt.fill(eng, s)
                fills[(base + (":mkt" if mk else ""), per)][0] += 1
                if f is None:
                    continue
                fills[(base + (":mkt" if mk else ""), per)][1] += 1
                for vn, v in VARS.items():
                    if vn == "봉마감손절" and base not in ("B-OB", "C-FVG"):
                        continue
                    r = bt.simulate(eng, s, v, fee, f, sub)
                    if r is None:
                        continue
                    r.update(var=vn, per=per, base=base, entry="mkt" if mk else "lim")
                    rows.append(r)
            print(sym, tf, "셋업", len(eng.setups), flush=True)
    return rows, fills


def table(rows, fills):
    lines = []
    def f(st):
        if st.get("n", 0) == 0:
            return "—"
        return f"{st['n']} · {st['win']*100:.0f}% · {st['avg']:+.2f}R · t{st['t']:+.1f}"
    lines.append("### 1. 강의 기본 규칙 (3코인·3시간봉 합침) — 거래 수 · 승률 · 평균 R · t값")
    lines.append("| 셋업 | 2025 | 2026 상반기 | 2026 하반기 | 체결률(26) |")
    lines.append("|---|---|---|---|---|")
    for k in KINDS:
        for ent in (("lim",) if not k.startswith("A-") else ("lim", "mkt")):
            sel = [r for r in rows if r["base"] == k and r["var"] == "기본" and r["entry"] == ent]
            cells = [f(stats([r for r in sel if r["per"] == p])) for p in PER]
            fk = k + (":mkt" if ent == "mkt" else "")
            tot = sum(fills[(fk, p)][0] for p in ("26H1", "26H2"))
            fil = sum(fills[(fk, p)][1] for p in ("26H1", "26H2"))
            name = k + ("" if not k.startswith("A-") else (" 리테스트" if ent == "lim" else " 다음봉시가"))
            lines.append(f"| {name} | " + " | ".join(cells) + f" | {fil}/{tot} |")
    lines.append("")
    lines.append("### 2. 시간봉별 (기본 규칙, A는 리테스트 진입) — 2026 상반기 / 하반기 평균 R (거래 수)")
    lines.append("| 셋업 | " + " | ".join(f"{tf} 상반기 | {tf} 하반기" for tf in TFS) + " |")
    lines.append("|---|" + "---|" * (2 * len(TFS)))
    for k in KINDS:
        cells = []
        for tf in TFS:
            for p in ("26H1", "26H2"):
                st = stats([r for r in rows if r["base"] == k and r["var"] == "기본" and r["entry"] == "lim" and r["tf"] == tf and r["per"] == p])
                cells.append("—" if st.get("n", 0) == 0 else f"{st['avg']:+.2f} ({st['n']})")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("### 3. 한 가지씩 바꾼 변형 (A는 리테스트 진입) — 2026 상반기 → 하반기 평균 R")
    lines.append("| 셋업 | " + " | ".join(VARS) + " |")
    lines.append("|---|" + "---|" * len(VARS))
    for k in KINDS:
        cells = []
        for vn in VARS:
            sel = [r for r in rows if r["base"] == k and r["var"] == vn and r["entry"] == "lim"]
            a, b2 = stats([r for r in sel if r["per"] == "26H1"]), stats([r for r in sel if r["per"] == "26H2"])
            cells.append("—" if a.get("n", 0) == 0 else f"{a['avg']:+.2f}→{b2.get('avg', float('nan')):+.2f}")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("### 4. 거르기 (기본 규칙, A는 리테스트) — 2026 상반기 → 하반기 평균 R (거래 수)")
    flt = {
        "전체": lambda r: True,
        "겹침≥1": lambda r: r["conf"] >= 1,
        "겹침≥2": lambda r: r["conf"] >= 2,
        "★만": lambda r: r["star"],
        "12H 추세 방향": lambda r: r["d"] == r["dir12"],
        "12H 추세 반대": lambda r: r["d"] == -r["dir12"],
    }
    lines.append("| 셋업 | " + " | ".join(flt) + " |")
    lines.append("|---|" + "---|" * len(flt))
    for k in KINDS:
        cells = []
        for fn, fl in flt.items():
            sel = [r for r in rows if r["base"] == k and r["var"] == "기본" and r["entry"] == "lim" and fl(r)]
            a, b2 = stats([r for r in sel if r["per"] == "26H1"]), stats([r for r in sel if r["per"] == "26H2"])
            cells.append("—" if a.get("n", 0) < 5 else f"{a['avg']:+.2f}→{b2.get('avg', float('nan')):+.2f} ({a['n']}/{b2.get('n', 0)})")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("### 5. 코인별 (기본 규칙, A는 리테스트, 3시간봉 합침) — 2026 전체 평균 R (거래 수)")
    lines.append("| 셋업 | " + " | ".join(SYMS) + " |")
    lines.append("|---|" + "---|" * len(SYMS))
    for k in KINDS:
        cells = []
        for sym in SYMS:
            st = stats([r for r in rows if r["base"] == k and r["var"] == "기본" and r["entry"] == "lim" and r["sym"] == sym and r["per"] != "2025"])
            cells.append("—" if st.get("n", 0) == 0 else f"{st['avg']:+.2f} ({st['n']})")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    fee = float(sys.argv[1]) if len(sys.argv) > 1 else FEE
    use_sub = "--nosub" not in sys.argv
    rows, fills = run(fee, use_sub)
    OUT.mkdir(exist_ok=True)
    tag = f"fee{fee*100:.2f}" + ("" if use_sub else "_nosub")
    (OUT / f"bt_rows_{tag}.json").write_text(json.dumps(rows))
    txt = table(rows, fills)
    (OUT / f"bt_tables_{tag}.md").write_text(txt, encoding="utf-8")
    print(txt)
