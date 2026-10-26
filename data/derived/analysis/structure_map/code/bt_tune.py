"""60분봉 수치 조정 — 트랩·이탈에 덜 낚이는 조건 찾기 (2026-10-23 사용자 "60분봉 기준으로 수익률을 더 개선할 디테일, 특히 트랩이나 이탈에 낚이면 안 될 것 같다").

AGENTS.md 연구 보고 규칙: 2026 상반기(1~6월)로 고르고 하반기(7월~)로 확인한다. 시험할 조건은 결과를 보기 전에 아래로 정했다.
기준(B0): 지표 기본값 — 4셋업(A-TL·A-CH·E-CH4·D-RT), 손익비 1 이상, 손절 여유 0.1 ATR, 전량 익절, 종목마다 한 포지션.
조건 17개(하나씩 기준에 더한다):
  트랩 피하기  T1~T3 거짓 돌파 되돌림(A-TL·A-CH)은 첫 마감 돌파가 약했을 때만(선 밖 마감 거리 < 0.3·0.2·0.5 ATR, 꼬리 Fake는 유지)
               T4 Fake(3봉 안 복귀)·꼬리 Fake만, T5 Trap(4~12봉 복귀)만
  이탈 피하기  E1·E2 돌파 뒤 리테스트(D-RT)는 첫 마감 돌파가 강했을 때만(≥ 0.3·0.5 ATR)
               E3 채널 4P는 채널 밖 봉마감이면 손절(비상 손절 1.1 ATR은 그대로), E4 채널 4P 끄기
  추세·겹침    H1 12시간 구조 방향과 같은 거래만, H2 12시간 구조와 반대인 거래만 빼기(방향 없음은 유지)
               C1·C2 진입 기준 가격 ±0.6 ATR 안에 다른 구조물이 1·2종류 이상(강의 '근거가 겹친 자리')
  손절 여유    S1·S2 0.25·0.5 ATR
  시간         M1 뉴욕 개장 첫 1시간(09:30~10:30)에 걸친 봉의 신호 빼기(5장 '개장 직후 거짓 돌파')
마감 거리는 §11과 같다: 처음 선 밖에서 마감한 봉의 종가와 선의 거리(그 봉 ATR 배수). 돌파 판정 기록(Engine.breaks)에서 신호마다 찾는다.
채택 기준(미리 정함): 상반기 평균 R이 기준보다 0.05R 넘게 낫고 거래가 기준의 절반 이상이며, 하반기에도 기준보다 나을 것.
묶음: 코인 3(BTC·ETH·SOL, 봉 안 순서 5분봉), 미국 AI 115(watchlist.AI_UNIVERSE 미국 상장, 1시간봉, 트레이딩뷰 가정, 나쁜 날 제외).
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402
import bt_2026 as y  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
AI = [s for s in (ROOT / "cache" / "us_ai_syms.txt").read_text().strip().split(",") if s not in ("SOXL", "SOXX", "EWY")]
GROUPS = {
    "코인": dict(syms=["BTCUSDT", "ETHUSDT", "SOLUSDT"], sub="5m", path="stop", fee=0.0005, us=False),
    "미국AI115": dict(syms=AI, sub=None, path="tv", fee=0.0003, us=True),
}
VARS = {
    "base": bt.Var(buf=0.1, exit="full", minrr=0.0, ch_close=False),
    "buf25": bt.Var(buf=0.25, exit="full", minrr=0.0, ch_close=False),
    "buf50": bt.Var(buf=0.5, exit="full", minrr=0.0, ch_close=False),
    "chclose": bt.Var(buf=0.1, exit="full", minrr=0.0, ch_close=True),
}
A = ("A-TL", "A-CH")


def weak(c, x):
    return c["kind"] not in A or c["typ"] == "wick" or c["dist"] < x


def strong(c, x):
    return c["kind"] != "D-RT" or (c["dist"] is not None and c["dist"] >= x)


VARIANTS = {
    "B0 기준": ("base", lambda c: True),
    "T1 약한 돌파만 되돌림 <0.3": ("base", lambda c: weak(c, 0.3)),
    "T2 약한 돌파만 되돌림 <0.2": ("base", lambda c: weak(c, 0.2)),
    "T3 약한 돌파만 되돌림 <0.5": ("base", lambda c: weak(c, 0.5)),
    "T4 Fake·꼬리만 (Trap 빼기)": ("base", lambda c: c["kind"] not in A or c["typ"] != "trap"),
    "T5 Trap만": ("base", lambda c: c["kind"] not in A or c["typ"] == "trap"),
    "E1 강한 돌파만 리테스트 ≥0.3": ("base", lambda c: strong(c, 0.3)),
    "E2 강한 돌파만 리테스트 ≥0.5": ("base", lambda c: strong(c, 0.5)),
    "E3 채널 4P 마감 손절": ("chclose", lambda c: True),
    "E4 채널 4P 끄기": ("base", lambda c: c["kind"] != "E-CH4"),
    "H1 12H 방향만": ("base", lambda c: c["dir12"] == c["d"]),
    "H2 12H 반대 빼기": ("base", lambda c: c["dir12"] != -c["d"]),
    "H3 12H 방향만 (4P는 닿은 봉 기준, 사후 추가)": ("base", lambda c: (c["dir12f"] if c["kind"] == "E-CH4" else c["dir12"]) == c["d"]),
    "C1 겹침 1+": ("base", lambda c: c["conf"] >= 1),
    "C2 겹침 2+": ("base", lambda c: c["conf"] >= 2),
    "S1 손절 여유 0.25": ("buf25", lambda c: True),
    "S2 손절 여유 0.5": ("buf50", lambda c: True),
    "M1 개장 첫 1시간 빼기": ("base", lambda c: not c["open1"]),
}


def annotate(eng):
    """셋업마다 (종류, 첫 마감 거리) — A는 Fake·Trap으로 끝난 돌파, D-RT는 진짜로 끝난 돌파에서 찾는다."""
    by = {}
    for r in eng.breaks:
        if r.get("res_i") is not None:
            by.setdefault((r["kind"], r["res_i"]), []).append(r)
    out = {}
    for s in eng.setups:
        base = s.kind.split(":")[0]
        typ, dist = None, None
        if base in A:
            rs = [r for r in by.get(("TL" if base == "A-TL" else "CH", s.i), []) if r["out"] in ("fake", "trap") and r["s"] == -s.d]
            if rs:
                r = rs[-1]
                typ, dist = r["out"], r["s"] * (eng.b[r["i"]]["c"] - r["lv"]) / eng.atr[r["i"]]
            else:
                typ, dist = "wick", 0.0
        elif base == "D-RT":
            rs = [r for r in by.get(("TL", s.i), []) if r["out"] == "real" and r["s"] == s.d]
            if rs:
                r = rs[-1]
                typ, dist = ("rebreak" if r["rebreak"] else "real"), r["s"] * (eng.b[r["i"]]["c"] - r["lv"]) / eng.atr[r["i"]]
        out[id(s)] = (typ, dist)
    return out


def open_hour(t, tf_ms):
    """신호 봉이 뉴욕 평일 09:30~10:30과 겹치는가."""
    a, b = y.ldt(t, y.ET), y.ldt(t + tf_ms, y.ET)
    if a.weekday() >= 5:
        return False
    o = a.replace(hour=9, minute=30, second=0, microsecond=0)
    return a < o + dt.timedelta(hours=1) and b > o


def build(sym, cfg):
    eng = bt.Engine(sym, "1h")
    sub = bt.Sub(sym, cfg["sub"]) if cfg["sub"] else None
    ann = annotate(eng)
    res = {k: [] for k in VARS}
    for s in eng.setups:
        base, _, mk = s.kind.partition(":")
        if mk or base not in y.KINDS:
            continue
        f = bt.fill(eng, s)
        if f is None:
            continue
        typ, dist = ann[id(s)]
        meta = dict(sym=sym, kind=base, d=s.d, typ=typ, dist=dist, dir12=s.dir12, conf=s.conf, dir12f=(eng.htf["dir12"][f[0]] or 0),
                    open1=open_hour(eng.b[s.i]["t"], bt.MS["1h"]), t_req=f[0] if base == "E-CH4" else s.i, fj=f[0])
        for k, v in VARS.items():
            r = bt.simulate(eng, s, v, cfg["fee"], f, sub, cfg["path"])
            if r is None:
                continue
            res[k].append(dict(meta, xj=f[0] + r["hold"], rr1=r["rr1"], R0=r["Rg"], R5=r["R"], t=r["t_fill"], te=r["t_exit"]))
    for k in res:
        res[k].sort(key=lambda c: (c["t_req"], c["fj"]))
    return res


def main():
    L = ["# 60분봉 수치 조정 — 트랩·이탈 피하기 (2026 상반기로 고르고 하반기로 확인)", "",
         "칸 = 평균 R (거래 수) · 승률. R = 손절 폭 1개, 비용 0. 차이 = 기준(B0) 대비. 채택 기준: 상반기 +0.05R 넘게 낫고 거래 절반 이상, 하반기에도 나음.", ""]
    rows = []
    for g, cfg in GROUPS.items():
        cache, bads = {}, {}
        for sym in cfg["syms"]:
            try:
                cache[sym] = build(sym, cfg)
            except FileNotFoundError:
                continue
            bads[sym] = {dt.date.fromisoformat(x) for x in json.loads((bt.HIST / f"{sym}_bad.json").read_text())} if cfg["us"] else set()
        print(g, len(cache), "종목", flush=True)

        def run(vk, keep):
            h = {1: [], 2: []}
            for sym, res in cache.items():
                cs = [c for c in res[vk] if keep(c)]
                for x in y.follow(cs, set(y.KINDS), 1.0):
                    p = y.half(x["t"])
                    if p and not y.crosses_bad(x, bads[sym]):
                        h[p].append(x)
            return h

        b0 = run(*VARIANTS["B0 기준"])
        s1b, s2b = y.st(b0[1]), y.st(b0[2])
        L += [f"## {g} ({len(cache)}종목, 1시간봉)", "",
              "| 조건 | 상반기 | 하반기 | 상반기 차이 | 하반기 차이 | 채택 |", "|---|---|---|---|---|---|"]
        for name, (vk, keep) in VARIANTS.items():
            h = run(vk, keep)
            s1, s2 = y.st(h[1]), y.st(h[2])
            d1 = s1["avg"] - s1b["avg"] if s1["n"] else float("nan")
            d2 = s2["avg"] - s2b["avg"] if s2["n"] else float("nan")
            ok = name != "B0 기준" and s1["n"] >= 0.5 * s1b["n"] and d1 > 0.05 and d2 > 0
            L.append(f"| {name} | {s1['avg']:+.2f} ({s1['n']}) · {s1['win']*100:.0f}% | {s2['avg']:+.2f} ({s2['n']}) · {s2['win']*100:.0f}% "
                     f"| {d1:+.2f} | {d2:+.2f} | {'✅' if ok else ''} |")
            rows.append(dict(g=g, name=name, h1=s1, h2=s2, d1=d1, d2=d2, ok=ok))
        # 손실 거래가 무엇에 낚였나(기준, 2026 전체): 셋업별 손절 비율과 평균 R
        allx = b0[1] + b0[2]
        L += ["", "기준 거래의 셋업별 2026 성적(낚인 자리 찾기):", "", "| 셋업 | 거래 | 승률 | 평균 R | 첫 마감 거리 중앙값(ATR) |", "|---|---|---|---|---|"]
        for k in y.KINDS:
            xs = [x for x in allx if x["kind"] == k]
            if not xs:
                continue
            s = y.st(xs)
            ds = sorted(x["dist"] for x in xs if x["dist"] is not None)
            L.append(f"| {k} | {s['n']} | {s['win']*100:.0f}% | {s['avg']:+.2f} | {(ds[len(ds)//2] if ds else float('nan')):.2f} |")
        L.append("")
    txt = "\n".join(L)
    OUT.mkdir(exist_ok=True)
    (OUT / "tune_tables.md").write_text(txt, encoding="utf-8")
    (OUT / "tune_rows.json").write_text(json.dumps(rows))
    print(txt)


if __name__ == "__main__":
    main()
