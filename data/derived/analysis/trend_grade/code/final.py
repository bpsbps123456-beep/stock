"""추세 등급 최종 사양 측정 (2026-10-12, 7계열 결과를 본 뒤 정한 설계 — 선택편향 있음, report.md에 공개).

결론부터: 7계열 사전 등록 약 230개 변형이 모두 1~5월 +, 6~10월 −였고(잔차 모멘텀만 둘 다 0 근처),
장세 지표 12개도 언제 통하는지 가르지 못했다. 그래서 '앞 구간 1위'를 고르지 않고, 앞 구간 성적이
상위권과 통계적으로 같은 변형 가운데 사용 목적(이평선 중심·여러 이평선 조합·신고가·등급이 매일 흔들리지
않음)에 맞는 사전 등록 변형을 골랐다.

최종(FINAL): 점수 = composite E14_s10 = 단면 백분위 평균{C/SMA200−1, C/SMA50−1, C/252일 최고 고가−1,
클레노우 126일(기울기 연율×R²)}의 최근 10일 평균. 등급 = 그 점수의 그날 백분위로 상위 10% S / 20% A /
40% B / 20% C / 10% D, 관문: S·A는 C>SMA50>SMA200일 때만(아니면 B), C<SMA200 그리고 SMA50<SMA200이면 D.

비교(같은 표에 같이 적는 이웃): E14 원점수, E14_s10 관문 없음, E6(L·H·Q)_s10, 클레노우 90일(앞 구간 1위), C/SMA200.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import fam_composite as FC  # noqa: E402
import fam_ladder as FL  # noqa: E402
import harness as H  # noqa: E402


def gated(score: pd.DataFrame, P: dict, mask: pd.DataFrame) -> pd.DataFrame:
    """피라미드 + 관문(fam_ladder.hyb_grades와 같다). mask 안에서 백분위."""
    return FL.hyb_grades(score, P, mask)


def monthly_ic(score: pd.DataFrame, h: int, start: str) -> dict:
    e = H.eligible()
    pr = H.pct_rank(score.reindex_like(e), e & score.reindex_like(e).notna())
    fr = H.forward(h).where(e)
    ic = H._row_spearman(pr, fr)
    ex = H.exit_date(h).reindex(ic.index)
    ic = ic[(ic.index >= pd.Timestamp(start)) & ex.notna()]
    m = ic.groupby(ic.index.to_period("M")).mean()
    return {str(k): round(float(v), 3) for k, v in m.items() if np.isfinite(v)}


def _clenow90(P):
    return FC.clenow(P["C"], 90)


SPECS = {
    "FINAL_E14s10_HYB": (FC.make(FC.GRID["E14"], 10), "HYB"),
    "E14s10_REL_P": (FC.make(FC.GRID["E14"], 10), "REL"),
    "E14_HYB": (FC.make(FC.GRID["E14"], 0), "HYB"),
    "E6s10_HYB": (FC.make(FC.GRID["E6"], 10), "HYB"),
    "clenow90_HYB": (_clenow90, "HYB"),
    "L_HYB": (lambda P: P["C"] / FC.sma(P["C"], 200) - 1.0, "HYB"),
}


def run() -> dict:
    P = H.load_panel()
    E = H.eligible()
    out = {}
    for name, (fn, kind) in SPECS.items():
        sc = fn(P)
        ev = H.evaluate(sc)
        G = gated(sc, P, E) if kind == "HYB" else FL.rel_grades(sc, H.PYRAMID, E)
        lad = FL.ladder_stats(G, windows=("front", "back", "y2025", "late"))
        ev_tab = {k: FL.event_table(G, *v) for k, v in FL.EVENTS.items()}
        out[name] = {
            "no_lookahead": H.no_lookahead(fn),
            "eval": ev, "ladder": lad, "events": ev_tab,
            "monthly_ic20_2026": monthly_ic(sc, 20, "2026-01-01"),
            "monthly_ic10_2026": monthly_ic(sc, 10, "2026-01-01"),
            "monthly_ic20_2025": monthly_ic(sc, 20, "2025-01-01"),
        }
        f, b = ev["front"]["h20"], ev["back"]["h20"]
        print(f"{name:20s} front IC20 {f['ic']:+.3f} (t {f['ic_t']}) | back {b['ic']:+.3f} (t {b['ic_t']}) | "
              f"late IC10 {ev['late']['h10']['ic']:+.3f} | 2025 {ev['y2025']['h20']['ic']:+.3f} | turn {lad['front'].get('turnover')}")
    path = os.path.join(H.RESULTS, "final.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"benchmarks": H.benchmarks(), "specs": out}, fh, ensure_ascii=False, indent=1)
    return out


if __name__ == "__main__":
    run()
