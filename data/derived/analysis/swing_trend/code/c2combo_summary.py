"""c2combo 결과 요약표(한국어). py code/c2combo_summary.py → results/c2combo_summary.txt"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from c2combo_lib import RES

TY = {1: "매수", -1: "매도"}
TR = {"P1": "P1 학습", "Y25P1": "Y25+P1 학습"}


def pp(x, nd=1):
    return "—" if not np.isfinite(x) else f"{100 * x:+.{nd}f}%p"


def pv(x):
    return "—" if not np.isfinite(x) else f"{x:.3f}"


def short(rule):
    return re.sub(r"\([^)]*\)", "", rule)


L = []
L.append("[판정] 채택 안 함(탐색 결과). P2 명목 통과는 1위 규칙 8개 중 2개(둘 다 매도), 모형 40개 중 4개(ZZA3 매도 3개, ZZA3 매수 종목 단서 모형 1개 — ZZP10에서 AUC 0.41로 재현 안 됨).")
L.append("  매도 통과는 Y25에서 반대(채택 기준 3 실패)이고, 10/22에 잘린 10월 매도 사건을 빼면 사라진다(한 국면). 매수는 P1 학습 모형 P2 AUC 0.33(ZZA3)으로 뒤집혔다.")
L.append("  P1 1위 규칙도 P1 안에서 날짜 이동 귀무의 1위보다 낫지 않았다(고르기 귀무 p 0.07~0.99).")
L.append("시험 범위: 규칙 1,377 × type 2 × 학습 2 × 사건표 2 = 11,016칸(자격 55~1,202개), 모형 5종(로지스틱 18개·윈저·시장 8개·종목 10개, 깊이 2 나무) × 2 × 2 × 2 = 40개, C 후보 9개 × 시간순 5덩어리 교차검증.")
L.append("실효 표본(사건 날짜 수): ZZA3 P1 매수 71·매도 65일, P2 69·67일, Y25 145·134일 / ZZP10 P1 84·82일, P2 76·84일, Y25 177·175일.")
L.append("")
L.append("BRIEF2 시험 2(조합) 요약 — 단위: y1 평균 차이(고른 − 안 고른), 로그 수익 %p. p는 한쪽(귀무 ≥ 실제).")
L.append("귀무: 이동 = 날짜 원형 이동 k 15..D−15(주), 전범위 = k 1..D−1, 순열 = 날짜 블록 순열, 전과정 = 학습·확인 모두 이동 후 고르기 반복. 각 200회.")
L.append("")
L.append("[2a·2c] 두 단서 3분위 AND 규칙 1,377개 × type 2 (학습 2가지 × 사건표 2가지 = 규칙 칸 11,016개)")
for det in ("ZZA3", "ZZP10"):
    s = pd.read_csv(RES / f"c2combo_rules_{det}_summary.csv")
    for _, r in s.iterrows():
        L.append(f"- {det} {TY[r.type]} {TR[r.train]}: 자격 규칙 {r.n_elig}개, 1위 '{short(r.top1)}' 학습 평균 {pp(r.top1_train_mean)} (고르기 귀무 p {pv(r.selnull_p_top1)})")
        L.append(f"    P2 1위: 고른 {int(r.P2_top1_n)}개/{int(r.P2_top1_dates)}일 (전체 {int(r.P2_N)}개/{int(r.P2_dates)}일), 차이 {pp(r.P2_top1_diff)} "
                 f"[날짜 부트스트랩 {pp(r.P2_top1_ci_lo)}~{pp(r.P2_top1_ci_hi)}], p 이동 {pv(r.P2_top1_fix_p)}(귀무 평균 {pp(r.P2_top1_fix_nullmean)}) "
                 f"전범위 {pv(r.P2_top1_all_p)} 순열 {pv(r.P2_top1_perm_p)} 전과정 {pv(r.P2_top1_pipe_p)}; y10 {pp(r.P2_top1_diff_y10)} y20 {pp(r.P2_top1_diff_y20)}")
        L.append(f"    P2 상위 10: 양수 {int(r.P2_top10_npos)}/10 (귀무 평균 {r.P2_top10_npos_null_mean:.1f}), 평균 차이 {pp(r.P2_top10_mean_diff)} p 이동 {pv(r.P2_top10_fix_p)} 전과정 {pv(r.P2_top10_pipe_p)}; "
                 f"P2 표본 10개·5일 이상 규칙 {int(r.P2_top10_n_ok)}개; 하나라도 해당(OR) {int(r.P2_union_n)}개 차이 {pp(r.P2_union_diff)} p 이동 {pv(r.P2_union_fix_p)} 순열 {pv(r.P2_union_perm_p)}")
        if r.train == "P1":
            L.append(f"    Y25 1위: 고른 {int(r.Y25_top1_n)}개/{int(r.Y25_top1_dates)}일, 차이 {pp(r.Y25_top1_diff)} [{pp(r.Y25_top1_ci_lo)}~{pp(r.Y25_top1_ci_hi)}] p {pv(r.Y25_top1_fix_p)}; "
                     f"상위 10 양수 {int(r.Y25_top10_npos)}/10 평균 {pp(r.Y25_top10_mean_diff)}; 참고 Y24 {pp(r.Y24_top1_diff)} Y23 {pp(r.Y23_top1_diff)}")
d = pd.read_csv(RES / "c2combo_diag.csv")
L.append("")
L.append("[진단] P2 1위 규칙의 10/22 잘림·10월 쏠림")
for _, r in d[(d.period == "P2") & (d.n_sel > 0)].iterrows():
    L.append(f"- {r.det} {TY[r.type]} {TR[r.train]}: 고른 {r.n_sel}개 중 잘림 {r.n_sel_censored}개·10월 {int(r.n_sel_oct)}개; 차이 {pp(r['diff'])} → 잘림 뺀 {pp(r.diff_uncensored)} "
             f"[{pp(r.unc_ci_lo)}~{pp(r.unc_ci_hi)}], 10월 뺀 {pp(r.diff_ex_oct)}")
mp = RES / "c2combo_models_ZZA3.csv"
if mp.exists():
    L.append("")
    L.append("[2b·2c·2d] 모형: y1 > 0 맞히기 (AUC 0.5 = 무작위). 상위 1/3 = P2 점수 순위 상위 1/3 사건 평균 y1 − 나머지")
    for det in ("ZZA3", "ZZP10"):
        f = RES / f"c2combo_models_{det}.csv"
        if not f.exists():
            continue
        m = pd.read_csv(f)
        for _, r in m.iterrows():
            extra = f" C {r.C:.3g}(교차검증 AUC {r.cv_auc:.3f})" if np.isfinite(r.C) else ""
            y25 = f"; Y25 AUC {r.Y25_auc:.3f} p {pv(r.Y25_auc_fix_p)} 상위1/3 {pp(r.Y25_top3_diff)}" if "Y25_auc" in r and np.isfinite(r.get("Y25_auc", np.nan)) else ""
            pipe = f" 전과정 {pv(r.P2_auc_pipe_p)}" if np.isfinite(r.get("P2_auc_pipe_p", np.nan)) else ""
            L.append(f"- {det} {TY[r.type]} {TR[r.train]} {r.model}{extra}: P2 AUC {r.P2_auc:.3f} [{r.P2_auc_ci_lo:.3f}~{r.P2_auc_ci_hi:.3f}] "
                     f"(이동 95분위 {r.P2_auc_fix_q95:.3f}, p {pv(r.P2_auc_fix_p)} 전범위 {pv(r.P2_auc_all_p)} 순열 {pv(r.P2_auc_perm_p)}{pipe}); "
                     f"상위1/3 {int(r.P2_top3_n)}개/{int(r.P2_top3_dates)}일 {pp(r.P2_top3_diff)} [{pp(r.P2_top3_ci_lo)}~{pp(r.P2_top3_ci_hi)}] p {pv(r.P2_top3_fix_p)}; "
                     f"학습 문턱 고른 비율 {r.P2_thr_share:.0%} 차이 {pp(r.P2_thr_diff)}{y25}")
        c = pd.read_csv(RES / f"c2combo_models_{det}_coef.csv")
        for (t, tr), g in c.groupby(["type", "train"], sort=False):
            g = g.reindex(g.coef.abs().sort_values(ascending=False).index).head(5)
            L.append(f"    계수 {det} {TY[t]} {TR[tr]} (표준화 1단위당 로그오즈, 큰 순 5개): " + ", ".join(f"{a} {b:+.2f}" for a, b in zip(g.name, g.coef)))
        L.append("    나무:")
        L.append("    " + (RES / f"c2combo_models_{det}_tree.txt").read_text(encoding="utf-8").replace("\n", "\n    "))
(RES / "c2combo_summary.txt").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
