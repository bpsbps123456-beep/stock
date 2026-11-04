"""2차 비율 탐색 요약(results/ratios/summary.md). ratios.py 결과 파일을 읽고, 직관 표 특징의 백분위 중앙값을 더한다.

실행(연구 폴더에서): py -B code/ratios_summary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ratios as R  # noqa: E402

OUT = R.OUT
LABEL = {
    "A:imb": "{g} 매수 ÷ 매도", "B:tshare": "{g} 거래 몫", "B:bshare": "{g} 매수 몫", "B:sshare": "{g} 매도 몫",
    "B:nshare": "{g} 순매수 ÷ 시장 순매매 총량", "C:dimb": "{g} 대 {h} 매수 우위 차", "C:ltr": "{g} 거래 ÷ {h} 거래",
    "C:lbr": "{g} 매수 ÷ {h} 매수", "C:nrel": "{g} 대 {h} 순매수 차(거래 대비)", "D:fund": "시장 순매도 합 ÷ 순매수 합",
    "D:breadth": "순매수 종목 비율", "D:top1pos": "순매수 1위 몫", "D:hhipos": "순매수 집중도", "D:top1neg": "순매도 1위 몫",
    "D:imball": "시장 전체 매수 우위", "D:levnet": "레버리지 롱 순매수 몫", "D:bull": "레버리지 롱 ÷ (롱 + 인버스) 매수",
    "E:rN": "{g} 순매수 순위", "E:rS": "{g} 매도 순위",
}
WTXT = {"W1": "바닥 20~6일 전", "W2": "바닥 직전 5일", "W3": "바닥 날", "W4": "바닥 뒤 5일"}


def label(feature: str) -> str:
    fam, kind, *rest = feature.split(":")
    key = f"{fam}:{kind}"
    g, h = (rest[0].split("|") + [""])[:2] if rest else ("", "")
    return LABEL.get(key, feature).format(g=R.GNAME.get(g, g), h=R.GNAME.get(h, h))


def side_txt(norm: str, side: int) -> str:
    if norm == "P":
        return "높음(상위 20%)" if side == 1 else "낮음(하위 20%)"
    return "메아리보다 큼" if side == 1 else "메아리보다 작음"


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{100 * x:.0f}%"


def intuitive_p() -> pd.DataFrame:
    """직관 표 특징의 백분위 중앙값(사건 묶음별)."""
    dates, prods, pg, Bm, Sm, G = R.load()
    pf = pd.read_parquet(R.DER / "price_feats.parquet")
    pf["date"] = pd.to_datetime(pf["date"])
    pf = pf.set_index("date").reindex(dates).reset_index().rename(columns={"index": "date"})
    ev = json.loads((R.DER / "events.json").read_text(encoding="utf-8"))
    S = R.event_sets(ev, dates, pf)
    it = pd.read_csv(OUT / "intuitive.csv")
    it = it[[c for c in it.columns if not c.endswith("_P")]]
    names = list(dict.fromkeys(it["특징"]))
    P = {}
    for L in (1, 5):
        F = R.features(Bm, Sm, G, prods, L)[names]
        P[L], _ = R.rolling_pct(F.values)
    allidx = np.flatnonzero(dates >= R.START)
    rows = []
    for _, r in it.iterrows():
        L, off = R.WINS[r["창"]]
        j = names.index(r["특징"])
        rec = {}
        for key, idx in (("B15", S["B15"]), ("B10", S["B10"]), ("F", S["F"]), ("D", S["D"]), ("평소", allidx)):
            v = R.placed(P[L], idx, off)[:, j]
            rec[f"{key}_P"] = float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
        rows.append(rec)
    return pd.concat([it, pd.DataFrame(rows)], axis=1)


def main() -> None:
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8"))
    cells = pd.read_csv(OUT / "cells.csv")
    sel = pd.read_csv(OUT / "select_2425.csv")
    combos = pd.read_csv(OUT / "combos.csv")
    it = intuitive_p()
    it.to_csv(OUT / "intuitive.csv", index=False, encoding="utf-8-sig")
    com = cells[cells["common"]].copy()
    com["바닥"] = com["B10_on"] / com["B10_n"]
    com["gap"] = com["바닥"] - com[["F_rate", "D_rate"]].max(axis=1)
    top = com.sort_values(["B10_on", "gap"], ascending=False).head(12)

    L = ["# 2차 — 매수·매도·순매수 비율과 종목 간 상대 비율 조합 (설명용, `code/ratios.py`)", ""]
    L.append(f"**결론: 비율 {meta['features']}개 × 창 4개 × 정규화 2개 = {meta['cells']:,}칸을 훑었다. 2024 이후 바닥 10개 중 8개 이상에서 "
             f"같은 쪽인 칸은 백분위 {meta['common']['P']}칸, 메아리 잔차 {meta['common']['E']}칸이다. 바닥 대신 아무 급락 날 10개를 뽑아도 "
             f"평균 {meta['null_common_mean']['P']:.0f}칸·{meta['null_common_mean']['E']:.0f}칸이 걸린다(우연으로 이만큼 이상 나올 확률 "
             f"{meta['null_common_p_ge_P']:.2f}·{meta['null_common_p_ge_E']:.2f}). 가짜 바닥·큰 하락일에는 드문 '바닥만의 공통'은 "
             f"{meta['common_distinct']['P'] + meta['common_distinct']['E']}칸이다(우연 기대 {meta['null_distinct_mean']['P'] + meta['null_distinct_mean']['E']:.2f}). "
             f"2024~2025로 고른 {meta['selected']}칸은 2026 확인을 {meta['checked']}칸 통과했다.**")
    L.append("")
    L.append("- 사건(2024 이후, 2024년 이전 자료는 판정에 쓰지 않음): 바닥 B10 10개, 가짜 바닥 F "
             f"{len(meta['events']['F'])}일, 큰 하락일 D {len(meta['events']['D'])}일(SOXX 하루 수익 하위 10%, 바닥 ±5·F 제외).")
    L.append("- 동반율 = 가짜 바닥 날(또는 큰 하락일)을 바닥 자리에 놓았을 때 같은 쪽인 비율이다. 높으면 바닥이 아니라 급락 날의 공통이다.")
    L.append("")
    L.append("## 1. 가장 일관된 비율 (바닥 10개 기준, 상위 12칸)")
    L.append("")
    L.append("| 비율 | 시점 | 쪽 | 바닥 10개 | 큰 바닥 5개 | 가짜 바닥 동반율 | 큰 하락일 동반율 |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in top.iterrows():
        L.append(f"| {label(r['feature'])} | {WTXT[r['window']]} | {side_txt(r['norm'], r['side'])} | {r['B10_on']}/{r['B10_n']} | "
                 f"{r['B15_on']}/5 | {pct(r['F_rate'])} | {pct(r['D_rate'])} |")
    L.append("")
    L.append("## 2. 직관 비율 원값 (중앙값, 괄호는 250세션 백분위 중앙)")
    L.append("")
    L.append("| 비율 | 시점 | 큰 바닥 5 | 바닥 10 | 가짜 바닥 | 큰 하락일 | 평소(2024~) |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in it.iterrows():
        cells_txt = []
        for key in ("B15", "B10", "F", "D", "평소"):
            v, p = r[key], r[f"{key}_P"]
            cells_txt.append("" if pd.isna(v) else f"{v:.2f} ({p:.0f})")
        L.append(f"| {r['뜻']} | {WTXT[r['창']]} | " + " | ".join(cells_txt) + " |")
    L.append("- 원값 중앙값은 해마다 크기가 달라 섞인다(가짜 바닥은 2025-04와 2026에 몰림). 비교는 괄호 안 백분위로 한다.")
    L.append("")
    L.append("## 3. 2024~2025로 고르고 2026으로 확인")
    L.append("")
    L.append("| 비율 | 시점 | 정규화 | 쪽 | 2024~25 바닥 6개 | 가짜·큰 하락 | 2026 바닥 4개 | 2026 가짜 바닥 | 우연 통과 확률 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in sel.iterrows():
        L.append(f"| {label(r['feature'])} | {WTXT[r['window']]} | {r['norm']} | {side_txt(r['norm'], r['sel_side'])} | {r['B2425_on']}/6 | "
                 f"{pct(r['F2425_rate'])}·{pct(r['D2425_rate'])} | {r['B26_on']}/4 | {pct(r['F26_rate'])} | {r['p_null_check']:.2f} |")
    L.append(f"- 통과 기준: 2026 바닥 4개 중 3개 이상 같은 쪽, 2026 가짜 바닥 동반율 1/3 이하. 통과 {meta['checked']}칸, 우연 기대 "
             f"{meta['checked_expected_null']:.2f}칸.")
    L.append(f"- 두 칸 조합: 고른 칸들로 {meta['combos']}쌍을 만들었다. 2024~25 기준 통과 {meta['combos_selected']}쌍, 2026 확인 통과 {meta['combos_checked']}쌍.")
    L.append("")
    L.append("## 4. 범위와 주의")
    L.append("")
    L.append("- 문법(BRIEF2 §3): 자기 균형 16, 시장 대비 몫 64, 종목 대비 짝 120 × 4 = 480, 시장 전체 8, 순위 12 = 580개. "
             "창 4개(바닥 20~6일 전·직전 5일·당일·뒤 5일), 정규화 2개(250세션 백분위 P, 제곱 메아리 잔차 Eq).")
    L.append(f"- 우연 기대: 바닥·가짜 바닥·큰 하락일을 모은 날에서 10세션 넘게 떨어진 가짜 사건 10개를 {meta['null_reps']}번 뽑아 같은 규칙을 돌렸다. "
             f"백분위 공통 칸의 우연 95% 상한은 {meta['null_common_p95']['P']:.0f}칸, 메아리 잔차는 {meta['null_common_p95']['E']:.0f}칸이다.")
    L.append("- 시점: 바닥 날·바닥 뒤 칸은 SEIBRO에 2~4세션 뒤에야 보인다. 바닥 전 칸도 바닥 자체가 사후 표지라 실시간으로 '바닥 직전'을 알 수 없다.")
    L.append("- Top50 절단: 목록 밖 상품은 0으로 더했다. 거래가 0인 묶음의 비율은 결측이라 작은 묶음(SQQQ·인컴·나머지)은 결측이 잦다.")
    L.append("")
    L.append("## 5. 검증 (`verify_recompute.md`, `verify_audit.md`)")
    L.append("")
    L.append("| 갈래 | 결과 |")
    L.append("|---|---|")
    L.append("| 독립 재계산 | `ratios.py`를 보지 않고 다시 구현했다. 묶음 배정, F 24일·D 34일 목록, 헤드라인 8칸이 같다. 구별 칸은 양쪽 모두 0이다 |")
    L.append("| 코드 감사 | 결론을 뒤집는 버그는 없다. 창 위치는 손계산과 차이 0이고, Eq 잔차와 사건 거르기도 정상이다 |")
    L.append("| 감사 뒤 고친 것(BRIEF2 문구에 맞춤) | ① 거래 0인 묶음의 몫·짝 순매수 차를 결측으로 했다(P 공통 37 → 35). ② 메아리 잔차 공통을 '같은 부호 + \\|z\\| 중앙값 ≥ 0.5'로, 고르기는 z 문턱 없이 했다. ③ 우연 계산에서도 가짜 사건 ±5세션 대조를 뺐다(실제 계산과 같게). ④ 반복을 1,000회로 늘렸다. ⑤ 직관 표는 사건마다 비율로 바꾼 뒤 중앙을 냈다 |")
    L.append("| 감사가 사후로 더한 3족(1,164칸) | 매도 ÷ 매도 상대 비율과 순매수 ÷ 시장 거래는 우연 수준이었다(P 3 대 4.7, E 19 대 17.2). 창 사이 변화족에서 바닥 뒤 5일 레버리지 매수 몫 변화 2칸이 F·D와 갈렸다. 그러나 반등 메아리였다(바닥 W4 SOXX +10.3% 대 F −1.8%). 잔차에서는 차이가 사라졌고 2026 확인에서도 탈락했다 |")
    L.append("| 민감도 | 체제 더미를 빼도 잔차 공통 156(우연 115, p 0.14), 구별 0, 확인 0이다. 조합을 2024~25 5/6 이상 칸 상위 20·40개로 넓히면 확인 통과 1·7개로, 우연 기대 3.0·10.6보다 적다 |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:4]))


if __name__ == "__main__":
    main()
