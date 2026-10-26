"""프리장 저점에서 이미 반등한 사건만 시가에 사면? (BULL 2026-10-07: 저가 -32.7% → 09:29 -19.8%, 저점 대비 +19%)
매수: 정규장 시가(09:30). 조건은 개장 전에 아는 값만: 프리장 인정 봉 저가(pm_low), 프리장 마지막 봉 종가(pm_last, 09:29경).
반등률 = (1+pm_last)/(1+pm_low) - 1. 매도: 첫 1시간 종가 / 당일 종가(참고) / 다음 날 시가·종가 / 5일 뒤 종가.
사용: py rebound.py  → results/rebound.txt"""
import io, os
import numpy as np, pandas as pd
from summarize import load, stats, fmt, RES

EXITS = [("h1", "첫1시간"), ("close", "종가*"), ("d1o", "익일시가"), ("d1c", "익일종가"), ("d5c", "5일뒤")]
BANDS = [(-1, 0.05, "반등 5% 미만"), (0.05, 0.10, "반등 5~10%"), (0.10, 0.20, "반등 10~20%"), (0.20, 9, "반등 20% 이상")]

if __name__ == "__main__":
    ev = load()
    ev["reb"] = (1 + ev.pm_last) / (1 + ev.pm_low) - 1
    o = io.StringIO()
    p = lambda *a: print(*a, file=o)
    groups = {"비실적·개별": ev[~ev.earn & ~ev.mkt], "  프리장 뉴스(BULL형)": ev[~ev.earn & ~ev.mkt & ~ev.news_ah],
              "  전날 애프터 뉴스(SMCI형)": ev[~ev.earn & ~ev.mkt & ev.news_ah], "실적·개별": ev[ev.earn & ~ev.mkt]}
    for name, g in groups.items():
        p(f"\n==== {name}: 시가 매수, 프리장 저점 대비 반등률 구간별 ====")
        for lo, hi, lab in BANDS:
            x = g[(g.reb >= lo) & (g.reb < hi)]
            for ex, el in EXITS:
                for per in ["2026상", "2026하", "2025", "2024"]:
                    y = x[x.period == per]
                    r = ((1 + y[ex]) / (1 + y.open) - 1).dropna()
                    s = stats(r, y.day)
                    if s.get("n", 0) >= 3:
                        p(f"  {lab:<10} {el:<5} {per:<6} {fmt(s)}")
    p("\n시험한 조합: 반등 구간 4개 × 매도 5개 × 사건 구분 4개 = 80개(기간 4개는 확인용).")
    # 연도 내 전방 검증: 2026 상반기 평균으로 고르고(상반기 n ≥ 8, 종가 매도 제외) 하반기·2025로 확인
    cand = []
    for name, g in groups.items():
        for lo, hi, lab in BANDS:
            x = g[(g.reb >= lo) & (g.reb < hi)]
            for ex, el in EXITS:
                if ex == "close":
                    continue
                f = lambda per: stats((1 + x[x.period == per][ex]) / (1 + x[x.period == per].open) - 1, x.day)
                a = f("2026상")
                if a.get("n", 0) >= 8:
                    cand.append((a["평균"], name.strip(), lab, el, a, f("2026하"), f("2025")))
    p(f"\n[2026 전방 검증 — 상반기 평균 상위 5개(후보 {len(cand)}개 중)를 하반기·2025로 확인]")
    for m, name, lab, el, a, b, c in sorted(cand, key=lambda z: -z[0])[:5]:
        p(f"  {name} / {lab} / {el}\n    상 {fmt(a)}\n    하 {fmt(b)}\n    2025 {fmt(c)}")
    txt = o.getvalue()
    open(os.path.join(RES, "rebound.txt"), "w", encoding="utf-8").write(txt)
    print(txt)
