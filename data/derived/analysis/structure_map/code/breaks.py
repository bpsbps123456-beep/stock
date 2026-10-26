"""진짜 돌파 vs 가짜 돌파 — 돌파 순간에 알 수 있는 근거로 가를 수 있나 (2026-10-22, 사용자 질문).

대상: 추세선(TL)·채널 반대선(CH)·고점저점 유동성(LQ)·상위 이동평균(MA)을 처음 종가로 넘은 봉(판정기 사건 1).
결말: 같은 판정기가 돌파 확정(6·7)이면 진짜, Fake out(3)·Trap(4)이면 가짜. 선이 먼저 지워지면 뺀다.
근거(돌파 봉 마감까지, '다음 봉 시가'만 그다음 봉 시가에 안다) — 기준값은 결과를 보기 전에 고정:
  안착: 다음 봉 시가도 선 밖(4장 Q&A) / 마감 거리: (종가-선)÷ATR ≥ 0.5 / 거래량: 직전 20봉 평균의 1.5배 이상 /
  반대쪽 꼬리: 봉 길이의 20% 이하(긴 꼬리 = 거짓 돌파 흔적, 5장) / 상위 추세 같은 방향(12H, 미국은 일봉 구조) /
  같은 선의 직전 거짓 돌파 있음 / 개장 직후 봉(5장: 개장·발표 직후는 Fake out이 많다).
수익: 다음 봉 시가에 돌파 방향 진입, 손절 = 선 안쪽 0.3 ATR, 익절 2R(최대 120봉), 수수료 0. 봉 안 순서는 코인 5분봉, 미국 1시간봉·트레이딩뷰 가정.
판정: 2026 상반기 → 하반기, 2025는 보조.
"""
from __future__ import annotations

import calendar
import datetime as dt
import json
import math
import pathlib
import statistics
import sys
from collections import defaultdict
from zoneinfo import ZoneInfo

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent.parent / "results"
GROUPS = {
    "코인": (["BTCUSDT", "ETHUSDT", "SOLUSDT"], ["1h", "2h", "4h"]),
    "미국ETF": (["SOXL", "SOXX", "EWY"], ["1h", "2h", "4h", "1d"]),
}
ET = ZoneInfo("America/New_York")
P0 = calendar.timegm((2025, 1, 1, 0, 0, 0)) * 1000
P1 = calendar.timegm((2026, 1, 1, 0, 0, 0)) * 1000
P2 = calendar.timegm((2026, 7, 1, 0, 0, 0)) * 1000
VAR = bt.Var(buf=0.3, exit="r2")


def period(t):
    return None if t < P0 else "2025" if t < P1 else "26H1" if t < P2 else "26H2"


def collect():
    rows = []
    for g, (syms, tfs) in GROUPS.items():
        us = g == "미국ETF"
        for sym in syms:
            bad = set(json.loads((bt.HIST / f"{sym}_bad.json").read_text())) if us else set()
            sub_all = bt.Sub(sym, "1h" if us else "5m")
            for tf in tfs:
                eng = bt.Engine(sym, tf)
                b, a_, n = eng.b, eng.atr, eng.n
                sub = None if (us and tf == "1h") else sub_all
                path = "tv" if us else "stop"
                for rec in eng.breaks:
                    i = rec["i"]
                    per = period(b[i]["t"])
                    if per is None or i + 1 >= n or rec["rebreak"]:
                        continue
                    r, s, lv, a = b[i], rec["s"], rec["lv"], a_[i]
                    if a <= 0 or r["h"] <= r["l"]:
                        continue
                    if us:
                        d0 = dt.datetime.fromtimestamp(r["t"] / 1000, dt.timezone.utc).astimezone(ET)
                        if str(d0.date()) in bad:
                            continue
                        openbar = tf != "1d" and d0.hour == 9 and d0.minute == 30
                    else:
                        hr = dt.datetime.fromtimestamp(r["t"] / 1000, dt.timezone.utc).hour
                        openbar = (hr * 60 + int(dt.timedelta(milliseconds=bt.MS[tf]).total_seconds() / 60)) > 13 * 60 + 30 and hr * 60 < 15 * 60
                    vols = [x["v"] for x in b[max(0, i - 20):i]]
                    vavg = sum(vols) / len(vols) if vols else 0.0
                    rng = r["h"] - r["l"]
                    far = (r["h"] - max(r["o"], r["c"])) if s > 0 else (min(r["o"], r["c"]) - r["l"])
                    d12 = eng.htf["dir12"][i] or 0
                    f = dict(
                        g=g, sym=sym, tf=tf, typ=rec["kind"], per=per, out=rec["out"],
                        settled=(b[i + 1]["o"] - lv) * s > 0,
                        dist=(r["c"] - lv) * s / a,
                        vol=(r["v"] / vavg) if vavg > 0 else float("nan"),
                        wick=far / rng,
                        htf=s * d12,
                        prior=rec["prior"],
                        openbar=openbar,
                    )
                    st = bt.Setup("BRK", i, s, None, lv, i + 1, i + 1, -1, lv)
                    res = bt.simulate(eng, st, VAR, 0.0, None, sub, path)
                    f["R"] = res["R"] if res else None
                    f["cost5"] = bt.simulate(eng, st, VAR, 0.0005, None, sub, path)["cost"] if res else None   # 편도 0.05%일 때 비용 R (비용은 수수료에 비례)
                    rows.append(f)
                print(g, sym, tf, "돌파", len(eng.breaks), flush=True)
    return rows


FEATS = [
    ("전체", lambda x: True),
    ("안착(다음 봉 시가도 밖)", lambda x: x["settled"]),
    ("안착 안 됨", lambda x: not x["settled"]),
    ("마감 거리 ≥ 0.5 ATR", lambda x: x["dist"] >= 0.5),
    ("마감 거리 < 0.2 ATR", lambda x: x["dist"] < 0.2),
    ("거래량 ≥ 1.5배", lambda x: x["vol"] == x["vol"] and x["vol"] >= 1.5),
    ("거래량 < 1배", lambda x: x["vol"] == x["vol"] and x["vol"] < 1.0),
    ("반대쪽 꼬리 ≤ 20%", lambda x: x["wick"] <= 0.2),
    ("반대쪽 꼬리 ≥ 50%", lambda x: x["wick"] >= 0.5),
    ("상위 추세 같은 방향", lambda x: x["htf"] > 0),
    ("상위 추세 반대 방향", lambda x: x["htf"] < 0),
    ("직전 거짓 돌파 있음", lambda x: x["prior"]),
    ("개장 직후 봉", lambda x: x["openbar"]),
]


def score(x):
    return int(x["settled"]) + int(x["dist"] >= 0.5) + int(x["vol"] == x["vol"] and x["vol"] >= 1.5) + int(x["wick"] <= 0.2) + int(x["htf"] > 0)


def cell(xs):
    res = [x for x in xs if x["out"] is not None]
    rr = [x["R"] for x in xs if x["R"] is not None]
    if len(res) < 20:
        return "—"
    p = sum(1 for x in res if x["out"] == "real") / len(res)
    m = sum(rr) / len(rr) if rr else float("nan")
    sd = statistics.pstdev(rr) if len(rr) > 1 else 0.0
    t = m / (sd / math.sqrt(len(rr))) if sd else 0.0
    return f"{p*100:.0f}% · {m:+.2f}R (t{t:+.1f}) · {len(res)}"


def tables(rows):
    L = []
    for g in GROUPS:
        xs = [x for x in rows if x["g"] == g]
        L.append(f"### {g} — 진짜 비율 · 돌파 방향 평균 R (t) · 건수")
        L.append("| 돌파 순간 근거 | 2025 | 2026 상반기 | 2026 하반기 |")
        L.append("|---|---|---|---|")
        for name, fl in FEATS:
            L.append(f"| {name} | " + " | ".join(cell([x for x in xs if x["per"] == p and fl(x)]) for p in ("2025", "26H1", "26H2")) + " |")
        L.append("")
        L.append("| 근거 개수(안착·거리·거래량·짧은 꼬리·상위 추세) | 2025 | 2026 상반기 | 2026 하반기 |")
        L.append("|---|---|---|---|")
        for k in range(6):
            L.append(f"| {k}개 | " + " | ".join(cell([x for x in xs if x["per"] == p and score(x) == k]) for p in ("2025", "26H1", "26H2")) + " |")
        L.append("")
        L.append("| 선 종류 | 2025 | 2026 상반기 | 2026 하반기 |")
        L.append("|---|---|---|---|")
        for typ in ("TL", "CH", "LQ", "MA"):
            L.append(f"| {typ} | " + " | ".join(cell([x for x in xs if x["per"] == p and x["typ"] == typ]) for p in ("2025", "26H1", "26H2")) + " |")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    rows = collect()
    OUT.mkdir(exist_ok=True)
    (OUT / "breaks_rows.json").write_text(json.dumps(rows))
    txt = tables(rows)
    (OUT / "breaks_tables.md").write_text(txt, encoding="utf-8")
    print(txt)
