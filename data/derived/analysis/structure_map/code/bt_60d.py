"""최근 60일 검증 — 미국 40분·160분봉 포함 (2026-10-23 사용자 "최근 60일로 검증해 보면 되지").

창: 미국 5분봉이 있는 첫날(yfinance 한도 60일)부터 끝까지. 같은 날짜로 코인도 잰다. 체결이 창 안인 거래만 센다.
규칙: 구조물 지도 매수·매도 기본값(4셋업, 손익비 1 이상, 전량 익절, 종목마다 한 포지션) — bt_2026.py와 같다.
봉 안 순서: 모든 시간봉을 5분봉으로 가린다(5분봉 안에서 손절·익절이 함께 닿으면 손절).
미국 40분·160분봉은 60일 5분봉으로만 만들어 창 앞부분에 신호가 적다. 같은 방식으로 만든 1hc·4hc와 이력이 긴 1h·4h를 같은 창에서 견줘 그 영향을 잰다.
묶음: 코인 3 · 미국 ETF 3(SOXL·SOXX·EWY) · 미국 반도체 10(§13과 같음) · 미국 AI 115(watchlist.AI_UNIVERSE의 미국 상장, 반도체 10 포함).
대조군: 같은 청산 규칙·같은 손절 폭 분포(ATR 배수)로 창 안 아무 봉 다음 시가에 진입.
앞뒤 확인: 창을 미국 거래일 수로 반 나눠 앞 절반 → 뒤 절반.
미국 실전 점검: 한국 08~24시 주문(docs/desk_system.md §1) — bt_2026.py와 같은 기준으로 '못 거는 거래'를 센다.
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import random
import sys
import zlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402
import bt_2026 as y  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
bt.MS["1hc"] = bt.MS["1h"]
bt.MS["4hc"] = bt.MS["4h"]
ETF = ["SOXL", "SOXX", "EWY"]
SEMI = ["MU", "SNDK", "WDC", "STX", "NVDA", "AVGO", "AMD", "MRVL", "CRDO", "ALAB"]
AI = [s for s in (ROOT / "cache" / "us_ai_syms.txt").read_text().strip().split(",") if s not in ETF]
TF_US = ["40m", "1h", "2h", "160m", "4h", "1d"]
TF_CAL = ["1hc", "4hc"]
GROUPS = {
    "코인": dict(syms=["BTCUSDT", "ETHUSDT", "SOLUSDT"], tfs=["40m", "1h", "2h", "160m", "4h"], us=False, fee=0.0005),
    "미국ETF": dict(syms=ETF, tfs=TF_US + TF_CAL, us=True, fee=0.0003),
    "미국반도체10": dict(syms=SEMI, tfs=TF_US + TF_CAL, us=True, fee=0.0003),
    "미국AI115": dict(syms=AI, tfs=TF_US + TF_CAL, us=True, fee=0.0003),
}
COLD = {"40m", "160m", "1hc", "4hc"}
VAR0 = bt.Var(buf=0.1, exit="full", minrr=0.0, ch_close=False)
VARR = bt.Var(buf=0.0, exit="full", minrr=1.0, ch_close=False)


def window():
    rows = json.loads((bt.HIST / "SOXX_5m.json").read_text())
    days = sorted({y.ldt(r[0], y.ET).date() for r in rows})
    w0 = dt.datetime.combine(days[0], dt.time(0), tzinfo=dt.timezone.utc)
    mid = dt.datetime.combine(days[len(days) // 2], dt.time(0), tzinfo=dt.timezone.utc)
    return int(w0.timestamp() * 1000), int(mid.timestamp() * 1000), days


def build(sym, tf, us, fee, sub, w0):
    try:
        eng = bt.Engine(sym, tf)
    except FileNotFoundError:
        return None
    cands, pool = [], []
    for s in eng.setups:
        base, _, mk = s.kind.partition(":")
        if mk or base not in y.KINDS:
            continue
        f = bt.fill(eng, s)
        if f is None:
            continue
        r = bt.simulate(eng, s, VAR0, fee, f, sub, "stop")
        if r is None:
            continue
        fj = f[0]
        if r["t_fill"] >= w0:
            ra = y.risk_atr(eng, s, f)
            if ra:
                pool.append(ra)
        sig = y.bar_close(eng.b[s.i]["t"], tf.rstrip("c"), us)
        cands.append(dict(sym=sym, kind=base, d=s.d, t_req=fj if base == "E-CH4" else s.i, fj=fj, xj=fj + r["hold"], rr1=r["rr1"],
                          R0=r["Rg"], R5=r["R"], t=r["t_fill"], te=r["t_exit"],
                          miss=bool(us and base != "E-CH4" and y.asleep_miss(sig, r["t_fill"]))))
    cands.sort(key=lambda c: (c["t_req"], c["fj"]))
    rnd = []
    idx = [i for i in range(eng.n - 2) if eng.b[i + 1]["t"] >= w0]
    if pool and idx:
        rng = random.Random(zlib.crc32(f"{sym}|{tf}|60d".encode()))
        for _ in range(3 * len(pool)):
            i = rng.choice(idx)
            d, k = rng.choice(pool)
            px = eng.b[i + 1]["o"]
            so = bt.Setup("rnd", i, d, None, px - d * k * eng.atr[i], i + 1, i + 1, -1, px)
            rr = bt.simulate(eng, so, VARR, 0.0, (i + 1, px, True), sub, "stop")
            if rr is not None:
                rnd.append(dict(t=rr["t_fill"], te=rr["t_exit"], R0=rr["Rg"]))
    return cands, rnd


def main():
    w0, mid, days = window()
    print("창", days[0], "~", days[-1], f"({len(days)}거래일), 앞뒤 경계", days[len(days) // 2], flush=True)
    cache, bads = {}, {}
    for g, cfg in GROUPS.items():
        for sym in cfg["syms"]:
            if (sym, cfg["tfs"][0]) in cache:
                continue
            if cfg["us"]:
                bset = set()
                for fn in (f"{sym}_bad.json", f"{sym}_bad5m.json"):
                    p = bt.HIST / fn
                    if p.exists():
                        bset |= {dt.date.fromisoformat(x) for x in json.loads(p.read_text())}
                bads[sym] = bset
            else:
                bads[sym] = set()
            p5 = bt.HIST / f"{sym}_5m.json"
            sub = bt.Sub(sym, "5m") if p5.exists() else None
            for tf in cfg["tfs"]:
                cache[(sym, tf)] = build(sym, tf, cfg["us"], cfg["fee"], sub, w0) if sub is not None else None
        print(g, "계산 끝", flush=True)

    def collect(g, tf, key="trades"):
        out = []
        used = 0
        for sym in GROUPS[g]["syms"]:
            v = cache.get((sym, tf))
            if v is None:
                continue
            used += 1
            xs = y.follow(v[0], set(y.KINDS), 1.0) if key == "trades" else v[1]
            out += [x for x in xs if x["t"] >= w0 and not y.crosses_bad(x, bads[sym])]
        return out, used

    L = ["# 최근 60일 검증 (구조물 지도 매수·매도 규칙)", "",
         f"창: 미국 {days[0]} ~ {days[-1]} ({len(days)}거래일), 앞 절반 {days[0]} ~ {days[len(days) // 2 - 1]}, 뒤 절반 {days[len(days) // 2]} ~. 코인도 같은 날짜부터.",
         "칸 = 승률 · 평균 R (t) · 거래 수. R = 손절 폭 1개. 비용 0. 봉 안 순서는 모두 5분봉.",
         "40분·160분봉(미국)·1hc·4hc는 60일 5분봉으로만 만든 봉이다(이력 짧음). 1h·2h·4h·1d는 이력이 긴 봉이다.", ""]
    rows = []
    for g, cfg in GROUPS.items():
        L += [f"## {g} ({len(cfg['syms'])}종목)", "",
              "| 시간봉 | 60일 전체 | 앞 절반 → 뒤 절반 | 무작위 진입 | 비용 넣으면 |" + (" 새벽이라 못 거는 거래 |" if cfg["us"] else ""),
              "|---|---|---|---|---|" + ("---|" if cfg["us"] else "")]
        for tf in cfg["tfs"]:
            xs, used = collect(g, tf)
            rz, _ = collect(g, tf, "rnd")
            s, f = y.st(xs), y.st(xs, "R5")
            a, b = y.st([x for x in xs if x["t"] < mid]), y.st([x for x in xs if x["t"] >= mid])
            r = y.st(rz)
            name = tf + (" (60일 봉)" if cfg["us"] and tf in COLD else "")
            line = (f"| {name} | {y.cell(s)} | {a['avg']:+.2f} ({a['n']}) → {b['avg']:+.2f} ({b['n']}) | {r['avg']:+.2f} ({r['n']}) "
                    f"| {f['avg']:+.2f} (편도 {cfg['fee']*100:.2f}%) |")
            if cfg["us"]:
                m = sum(1 for x in xs if x["miss"])
                line += f" {m}/{len(xs)} |"
            L.append(line)
            rows.append(dict(g=g, tf=tf, used=used, all=s, h1=a, h2=b, rnd=r, fee=f, miss=sum(1 for x in xs if x["miss"])))
        L.append("")
    L += ["## 종목별 60일 (ETF 3 + 반도체 10) — 거래 수 · 평균 R", "", "| 종목 | " + " | ".join(TF_US) + " |", "|---|" + "---|" * len(TF_US)]
    for sym in ETF + SEMI:
        cs = []
        for tf in TF_US:
            v = cache.get((sym, tf))
            if v is None:
                cs.append("—")
                continue
            xs = [x for x in y.follow(v[0], set(y.KINDS), 1.0) if x["t"] >= w0 and not y.crosses_bad(x, bads[sym])]
            s = y.st(xs)
            cs.append("—" if not s["n"] else f"{s['n']} · {s['avg']:+.2f}")
        L.append(f"| {sym} | " + " | ".join(cs) + " |")
    missing = sorted({sym for (sym, tf), v in cache.items() if v is None})
    L += ["", f"5분봉·봉 파일이 없어 뺀 종목: {', '.join(missing) if missing else '없음'}"]
    txt = "\n".join(L)
    OUT.mkdir(exist_ok=True)
    (OUT / "d60_tables.md").write_text(txt, encoding="utf-8")
    (OUT / "d60_rows.json").write_text(json.dumps(rows))
    print(txt)


if __name__ == "__main__":
    main()
