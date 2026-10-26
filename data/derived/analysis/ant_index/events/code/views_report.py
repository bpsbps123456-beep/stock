"""views_leak.md 작성: views_fetch.py·feat_views.py 산출 수치만 옮긴다(가격 없음).

실행: py views_report.py   (views_fetch.py → feat_views.py 다음)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results"


def f(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    return f"{x:,.{nd}f}"


def rho_cell(d: dict, key: str) -> str:
    v = d.get(key)
    return "—" if not v else f"{v['rho']:+.2f} ({v['n']})"


def main() -> int:
    fm = json.loads((OUT / "views_fetch_meta.json").read_text("utf-8"))
    gm = json.loads((OUT / "views_growth.json").read_text("utf-8"))
    F = pd.read_parquet(OUT / "feat_views.parquet")
    grp = pd.read_parquet(OUT / "views_fetch_groups.parquet")
    ev = pd.read_parquet(OUT / "views_fetch_events.parquet")
    G, L, rt, ck = gm["growth"], gm["leak"], fm["realtime"], fm["checks"]
    runs = fm["log"]["runs"]
    lines: list[str] = []
    w = lines.append

    # n_posts와의 상관(2026 소급 창)
    npc = {}
    for mk in ("kr", "us"):
        fb = F[(F["market"] == mk) & ~F["realtime_window"] & (F["share_realtime_posts"] < 0.05) & (F["D"] >= "2026-01-01")]
        npc[mk] = {c: float(spearmanr(fb[c], fb["n_posts"], nan_policy="omit").statistic)
                   for c in ("a_top10_view_share", "a_top10pct_view_share", "a_view_gini", "a_up_gini", "b_vpc_median",
                             "down_ratio_first")}
    ck_kr, ck_us = L["corr_with_age_kr"], L["corr_with_age_us"]
    bk, bu = L["boundary_kr"], L["boundary_us"]
    stab = pd.DataFrame(L["realtime_rank_stability"])
    pairs_tab = G["pair_by_age1"]
    sc = G["short_curve"]
    ct = G["curve_table"]
    lab_kr, lab_us = gm["label_windows_kr"], gm["label_windows_us"]
    pt = G["pair_totals"]
    vt = L["vpc_top10_vs_rest_backward"]

    w("# 조회·추천 특징과 누출 진단 (부록 13 사전 등록, 가격 미사용)")
    w("")
    w(f"> 2026-10-22. 스냅샷 `final/snapshot_manifest.json`(2025-08-20~2026-10-21, 파일 {fm['snapshot']['files']}개, 줄 {f(fm['snapshot']['rows'])}개, 글 {f(fm['snapshot']['posts'])}개, 기준 시각 {fm['snapshot']['S']}).")
    w("> 가격은 읽지 않았다. 포럼 특징만 만들었고 가격과 한 표에 놓지 않았다. 동결 문서는 고치지 않았다.")
    w("> 코드 `events/code/views_fetch.py`(수집 시각) → `feat_views.py`(특징·진단) → `views_report.py`(이 문서). 테스트 `test_views.py` 9개 통과.")
    w("> 산출 `events/results/`: `feat_views.parquet`(창 577줄 = W_KR 283 + W_U 294), `views_posts.parquet`(글 129,638줄), `views_rows.parquet`(줄 133,579줄), `views_growth.json`, `views_fetch_meta.json`, `views_fetch_events.parquet`(따라잡기 사건), `views_fetch_groups.parquet`(증거 묶음), `views_fetch_log_intervals.csv`.")
    w("")
    w("## 0. 결론")
    w("")
    w("| 특징 | 수집 나이에 흔들리나 | 누출 위험 | 사건 연구에서 쓰는 법 |")
    w("|---|---|---|---|")
    w(f"| (a) 조회 상위 10글 몫 | 첫날에 크게 흔들림(실시간 첫 줄 {f(bk['a_top10_view_share_raw']['realtime_median'])} → 24시간 줄 {f(bk['a_top10_view_share_last']['realtime_median'])} → 직전 3주 소급 {f(bk['a_top10_view_share']['before_3w_median'])}, W_KR 중앙값). 2026 소급 창끼리는 나이와 무관(ρ {rho_cell(ck_kr['a_top10_view_share'], 'backward_2026')}) | 낮음(순위만 씀). 다만 **창 글 수와 ρ {npc['kr']['a_top10_view_share']:+.2f}**라 사실상 글 수 지표 | 소급 창끼리만. 글 수와 겹치므로 상위 10% 글 몫(`a_top10pct_view_share`, 글 수와 ρ {npc['kr']['a_top10pct_view_share']:+.2f})을 함께 본다 |")
    w(f"| (a) 조회 지니 | 첫날에 흔들림(첫 줄 {f(bk['a_view_gini_raw']['realtime_median'])} → 24시간 {f(bk['a_view_gini_last']['realtime_median'])} → 소급 {f(bk['a_view_gini']['before_3w_median'])}). 2026 소급 창끼리는 ρ {rho_cell(ck_kr['a_view_gini'], 'backward_2026')} | 낮음 | 소급 창끼리만. 실시간 창(10/14~)은 따로 |")
    w(f"| (a) 추천 지니 | 가장 덜 흔들림(실시간 {f(bk['a_up_gini']['realtime_median'])} vs 직전 3주 {f(bk['a_up_gini']['before_3w_median'])}; 2026 소급 ρ {rho_cell(ck_kr['a_up_gini'], 'backward_2026')}) | 낮음 | 조회·추천 특징 가운데 나이·실시간 경계에 가장 덜 흔들린다 |")
    w(f"| (b) 조회 ÷ (마감 전 댓글+1) | 첫날 조회가 중앙값 ×{pairs_tab[0]['views_ratio']['median']:.2f}(첫 수집 1시간 안)~×{pairs_tab[2]['views_ratio']['median']:.2f}(2~3시간) 더 늘고, 인기 글은 ×{pairs_tab[0]['views_ratio_top10pct_posts']['median']:.1f}까지 는다. 소급 창 나이와 ρ {rho_cell(ck_kr['b_vpc_median'], 'backward_only')} | **있음**: 창 안 (b) 상위 10% 글은 하루 뒤 댓글 몫이 {vt['late1_share_top']*100:.1f}%로 나머지({vt['late1_share_rest']*100:.1f}%)의 2배, 7일 뒤 댓글 몫 {vt['late7_share_top']*100:.1f}% vs {vt['late7_share_rest']*100:.1f}%. 그중 {vt['c_pre_zero_share_top']*100:.0f}%는 마감 전 댓글 0개 | 부록 13 누출 시험(사건일 vs 가짜 사건일)을 반드시 거친다. 나이 보정 짧은판으로도 실시간 경계 차이가 남는다(아래 3-4) |")
    w(f"| 다른의견 비율 | 흔들림: 같은 글이 첫 줄→24시간에 W_KR 실시간 창 중앙값 {f(bk['down_ratio_first']['realtime_median'], 3)}→{f(bk['down_ratio_last']['realtime_median'], 3)}(추천이 다른의견보다 늦게 쌓임). 2026 소급 창 나이와 ρ {rho_cell(ck_kr['down_ratio_first'], 'backward_2026')} | **있음**(나이와 함께 내려감) | 수집 나이·시기를 통제하고만 본다 |")
    rtw0 = F[F["realtime_window"]]
    fb0 = {mk: (rtw0.loc[rtw0["market"] == mk, "share_fetched_before_close"].min(), rtw0.loc[rtw0["market"] == mk, "share_fetched_before_close"].max())
           for mk in ("kr", "us")}
    w(f"| (c) 실시간 원값 | 실시간 창만(W_KR {int((rtw0['market'] == 'kr').sum())}창, W_U {int((rtw0['market'] == 'us').sum())}창). 첫 줄 나이 중앙값 1.5~1.9시간(K10/14·U10/13 창은 11~12시간) | 낮음. 다만 창 마감 전에 받은 글 몫은 W_KR {fb0['kr'][0]*100:.0f}~{fb0['kr'][1]*100:.0f}%, W_U {fb0['us'][0]*100:.0f}~{fb0['us'][1]*100:.0f}%(U10/13은 0%: 실행 1이 그 창 마감 뒤 시작) | 사건 수가 모자라 설명용 |")
    w(f"| 조회 상위 10% 글의 예측 층 P̄ | — | — | **창 단위로 못 씀**: 라벨 단위 중 그런 글이 W_KR 창당 예측 평균 {lab_kr['top10_pred_total']/283:.2f}개(합 {lab_kr['top10_pred_total']}, 값 있는 창 {lab_kr['windows_with_top10_pred']}/283). 댓글까지 넣어도 합 {lab_kr['top10_withc_pred_total']} |")
    w("")
    w("- 조회는 첫 몇 시간에 대부분 쌓인다: 24시간 조회를 1로 두면 등록 15분 뒤 " +
      f"{ct['0.25h']['g_short']:.2f}, 30분 {ct['0.5h']['g_short']:.2f}, 1시간 {ct['1h']['g_short']:.2f}, 3시간 {ct['3h']['g_short']:.2f}, 12시간 {ct['12h']['g_short']:.2f}(같은 글 두 줄 {sc['n']:,}쌍).")
    w("- 그러나 많이 본 글일수록 늦게까지 는다. 그래서 같은 창 안 조회 몫·지니는 나이가 들수록 커지고(같은 글 집합에서 지니 첫 줄 0.36~0.50 → 24시간 0.49~0.59), 한 곡선으로 나눠도 고쳐지지 않는다(보정판 상위 10글 몫이 첫 줄 원값과 거의 같음).")
    w("- 24시간 뒤 몇 달 동안 더 쌓이는 양은 지금 자료로 가를 수 없다. 소급분은 나이와 등록 시기가 한 줄로 묶여 있다(나이 ≈ 수집일 − 등록일). 소급 창의 나이 칸별 '조회 ÷ 24시간 댓글'은 기준(실시간 24시간 줄)보다 log로 +0.06~+0.46 높지만 단조롭지 않다(6~7월 급락기 칸이 가장 낮음 = 시기 효과).")
    w("- 그래서 '바닥·고점 예측'에 조회·추천을 쓸 때 가장 안전한 것은 추천 지니와, 소급 창끼리 비교하는 조회 지니·상위 10% 몫이다. (b)와 다른의견 비율은 누출 시험 없이 후보로 올리지 않는다.")
    w("")
    w("## 1. 글마다 모은 값")
    w("")
    w("| 열(`views_posts.parquet`) | 뜻 |")
    w("|---|---|")
    w("| `views/up/down/ncomment_first`, `_last` | 첫 줄·마지막 줄의 조회·추천·다른의견·댓글 수(두 줄 글 3,941개, 나머지는 같은 값) |")
    w("| `c_pre_kr`, `c_pre_u` | 마감 전 댓글 수: 로더 규칙(`units.pre_close_kr/u` = 부모 글 창 마감 전, 모든 줄 합집합의 시각 기준). 창별 합이 `feat_activity.n_comments_onposts`와 577창 모두 같음 |")
    w("| `c1`, `c3`, `c24`, `c_late1d`, `c_late7d`, `c_union` | 등록 뒤 1·3·24시간 안 댓글, 1일·7일 뒤 댓글, 전체(시각 기준) |")
    w("| `T_first`, `age_h_first`, `T_last`, `age_h_last`, `method_*` | 수집 시각 추정과 그때 나이(시간), 추정 방법 |")
    w("| `views24_short`, `views24_full` | 나이 보정 조회(24시간 환산). short = 30시간까지 곡선만(그 뒤 더 늘지 않는다고 봄), full = 소급 단면 곡선까지(위 시기 효과가 섞인 상한) |")
    w("")
    w("## 2. 수집 시각 추정")
    w("")
    w("원자료 줄에는 수집 시각이 없다. 수집기(`desk/ant.py`) 규칙과 세 가지 증거로 줄마다 수집 시각 T̂를 놓았다.")
    w("")
    w("| 줄 종류 | 줄 수 | 방법 | 오차 |")
    w("|---|---|---|---|")
    rb = ck["bound_width_min"]["backward"]
    ivs = pd.read_csv(OUT / "views_fetch_log_intervals.csv")
    ex_ev = ivs.loc[(ivs["n_events"] > 0) & (pd.to_datetime(ivs["t1"]) <= pd.Timestamp(fm["snapshot"]["S"])), "excess_min"]

    w(f"| 거꾸로 받은 첫 줄(소급분) | {f(fm['class_counts']['backward'])} | `collect.log`의 '시각 · low 번호' 점({fm['log']['lines']}줄)을 실행별로 잇고 번호로 직선 보간. 실행 경계 = 저장 수가 다시 작아진 곳({len(runs)}실행). 첫 로그 전 1,000번호는 순수 속도 {fm['log']['rate_s_per_no']:.2f}초/번호(로그 간격 하위 25%)로 거꾸로 늘림 | 감싸는 로그 두 점 사이(폭 중앙값 {rb['0.5']:.0f}분, 90% {rb['0.9']:.0f}분) 안. 보간 오차는 그 간격 안 따라잡기에 쓴 시간(사건 든 간격의 남는 시간 중앙값 {ex_ev.median():.0f}분, 최대 {ex_ev.max():.0f}분) 이하. 나이(중앙값 {G['age_by_class_h']['backward_first_all']['0.5']/24:.0f}일)에 비하면 무시할 크기 |")
    w(f"| 앞으로 받은 첫 줄(실시간) | {f(fm['class_counts']['forward'])} | 하한 = 글 시각·그 줄 댓글 시각 최대·날짜 붙은 댓글의 다음 날 0시, 상한 = 다음 줄에만 있는 댓글 중 가장 이른 시각·시각만 댓글의 날짜. 같은 파일 줄 순서·번호 순서로 누적(하한 앞으로, 상한 뒤로). 번호 순 교집합으로 '따라잡기 사건'을 묶고(증거 사건 {rt['anchor_groups']}개, 폭 중앙값 {grp.loc[grp['anchor'], 'w_min'].median():.1f}분), 빈 곳은 관측 간격 {rt['period_h']:.2f}시간으로 채움(가상 {rt['virtual_events']}개) | 증거 범위 폭 중앙값 {ck['bound_width_min']['forward']['0.5']:.0f}분. 한 사건 안 받기 시간 중앙값 {ev['pred_minutes'].median():.0f}분·90% {ev['pred_minutes'].quantile(.9):.0f}분(번호 순으로 5초씩 늦어짐)을 따로 나누지 않았다. 가상 사건에 놓인 줄 {rt['rows_on_virtual']['forward']}개는 ±1시간 |")
    w(f"| 24시간 다시 받은 줄 | {f(fm['class_counts']['recheck'])} | max(등록 + 24시간, 댓글 하한) 뒤 첫 따라잡기 사건 | 증거 사건이면 위와 같음, 가상 사건 {rt['rows_on_virtual']['recheck']}줄은 ±1시간 |")
    w("")
    le = ck["log_excess"]
    w("검증(모두 가격 없이):")
    w("")
    w("| 점검 | 결과 |")
    w("|---|---|")
    w(f"| 추정 시각이 그 줄 댓글보다 이른 줄 | 0 / {f(fm['snapshot']['rows'])} |")
    w(f"| 시각만 댓글로 정해진 수집 날짜와 추정 날짜 | 소급 {ck['coll_date_check_backward']['same_date']}/{ck['coll_date_check_backward']['rows_with_date']}, 실시간 첫 줄 {ck['coll_date_check_forward']['same_date']}/{ck['coll_date_check_forward']['rows_with_date']}, 다시 받은 줄 {ck['coll_date_check_recheck']['same_date']}/{ck['coll_date_check_recheck']['rows_with_date']} 같음 |")
    w(f"| 로그 간격의 남는 시간(간격 − 1,000번호 × 순수 속도) | 따라잡기 사건이 든 간격에만 남는 시간이 있다: 5분 이상 남은 간격 {le['excess_ge5_with_event'] + le['excess_ge5_without_event']}개 중 {le['excess_ge5_with_event']}개에 사건(나머지 2개는 스냅샷 뒤 10/22 16시 이후). 증거 사건만 든 간격 {le['anchor_intervals']}개에서 남는 시간과 '그 사건이 연 번호 수 × 5초'의 순위상관 {le['anchor_excess_vs_pred_spearman']:.3f}, 차이 중앙값 {le['anchor_excess_minus_pred_median_min']:.1f}분 |")
    w(f"| 상한으로 쓰지 않은 '새 댓글' | {rt['ub_dropped_conflict']}개(첫 줄 수집 뒤라 할 수 없는 시각 = 숨김·차단 해제·시각 고친 댓글). 쓴 상한 {rt['ub_used']}개, 모순 줄 {rt['infeasible_rows']} |")
    w(f"| 실행 | " + "; ".join(f"실행 {r['run']+1}: 시작 추정 {r['start_t_est'][:16]}, 로그 {r['first_t'][5:16]}~{r['last_t'][5:16]}, 번호 {r['L0']-1:,}→{max(r['end_low'], r['last_low']):,}" for r in runs) + " |")
    w("")
    ab = G["age_by_class_h"]
    w("수집 나이 분포(시간):")
    w("")
    w("| 줄 | n | 10% | 50% | 90% |")
    w("|---|---|---|---|---|")
    for k, lab_ in (("forward_first", "실시간 첫 줄(앞으로 받기)"), ("backward_first_realtime(<24h)", "실행 1 시작에 거꾸로 받은 10/13 글(24시간 안)"),
                    ("recheck", "24시간 다시 받은 줄"), ("backward_first_all", "소급 첫 줄 전체")):
        a = ab[k]
        w(f"| {lab_} | {f(a['n'])} | {a['0.1']:.2f} | {a['0.5']:.2f} | {a['0.9']:.2f} |")
    w("")
    w("## 3. 누출 진단")
    w("")
    w("### 3-1. 나이에 따른 조회 성장(같은 글 두 줄)")
    w("")
    w(f"같은 글을 첫 줄(나이 a₁)과 24시간 뒤 줄(a₂ ≈ 24~30시간)로 두 번 받은 {G['pairs']['n']:,}쌍(조회가 줄어든 쌍 {G['pairs']['views_decrease']}개). "
      "log(조회₂/조회₁) = h(a₂) − h(a₁)로 h를 log 나이 조각 직선(h(24h)=0)으로 맞추고 Huber 가중 → 단조 증가로 고쳤다. "
      f"잔차 MAD {sc['resid_mad']:.3f}. 받침이 얇은 마디({', '.join(str(x) for x in sc['knots_dropped'])}시간)는 뺐다.")
    w("")
    w("| 첫 수집 나이 | 쌍 | 나이₁ 중앙 | 나이₂ 중앙 | 조회 증가 중앙값 (25~75%) | 합계 증가 | 24시간 조회 상위 10% 글의 증가 중앙값 | 추천 증가(추천₁>0) 중앙 | 시각 기준 댓글 증가 중앙 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for r in pairs_tab:
        vr = r["views_ratio"]
        w(f"| {r['age1_h']}시간 | {r['n']:,} | {r['age1_median_h']:.2f} | {r['age2_median_h']:.2f} | ×{vr['median']:.3f} ({vr['p25']:.2f}~{vr['p75']:.2f}) | ×{r['views_sum_ratio']:.2f} | ×{r['views_ratio_top10pct_posts']['median']:.2f} | ×{r['up_ratio(up1>0)']['median']:.2f} (75% ×{r['up_ratio(up1>0)']['p75']:.2f}) | ×{r['comments_ts_ratio(c1>0)']['median']:.2f} (75% ×{r['comments_ts_ratio(c1>0)']['p75']:.2f}) |")
    w("")
    w("보정 곡선 g(a) = 24시간 조회 대비 그 나이 조회(짧은판은 30시간 뒤 1.005로 고정, 넓은판은 소급 단면 곡선):")
    w("")
    w("| 나이 | 15분 | 30분 | 1시간 | 2시간 | 3시간 | 6시간 | 12시간 | 24시간 | 2일 | 7일 | 30일 | 90일 | 180일 | 365일 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    keys = ["0.25h", "0.5h", "1h", "2h", "3h", "6h", "12h", "24h", "48h", "168h", "720h", "2160h", "4320h", "8760h"]
    w("| g 짧은판 | " + " | ".join(f"{ct[k]['g_short']:.3f}" for k in keys) + " |")
    w("| g 넓은판 | " + " | ".join(f"{ct[k]['g_full']:.3f}" for k in keys) + " |")
    w("")
    w("### 3-2. 같은 글 두 줄의 조회·추천 증가")
    w("")
    w("| 값 | 첫 줄 합 | 마지막 줄 합 | 증가 |")
    w("|---|---|---|---|")
    for k, lab_ in (("views", "조회"), ("up", "추천"), ("down", "다른의견"), ("ncomment", "댓글 수(원천 표시)")):
        a, b = pt[k]
        w(f"| {lab_} | {a:,} | {b:,} | {100*(b/a-1):+.0f}% |")
    w("")
    w("- 추천은 대부분 그대로다(추천₁>0 글의 증가 중앙값 ×1.00). 늘어난 추천은 일부 인기 글에 몰린다. 다른의견은 추천보다 덜 늘어 다른의견 비율이 나이와 함께 내려간다.")
    w(f"- 실시간 창에서 같은 글 집합의 첫 줄 vs 24시간 줄 순위상관: 조회 {stab['spearman_views_first_last'].min():.2f}~{stab['spearman_views_first_last'].max():.2f}(나이 보정 {stab['spearman_views_adj_last'].min():.2f}~{stab['spearman_views_adj_last'].max():.2f}), 추천 {stab['spearman_up_first_last'].min():.2f}~{stab['spearman_up_first_last'].max():.2f}. 조회 상위 10% 겹침 {stab['top10pct_overlap'].min():.2f}~{stab['top10pct_overlap'].max():.2f}(보정 {stab['top10pct_overlap_adj'].min():.2f}~{stab['top10pct_overlap_adj'].max():.2f}).")
    w("")
    w("| 실시간 창 | 두 줄 글 | 첫 줄 나이 중앙(시간) | 조회 순위상관 첫→24h | 보정→24h | 상위 10% 겹침 | 상위 10글 몫 첫/보정/24h | 지니 첫/24h |")
    w("|---|---|---|---|---|---|---|---|")
    for r in stab.itertuples():
        w(f"| {r.win_id} | {r.n_two_row_posts} | {r.age_first_median_h:.1f} | {r.spearman_views_first_last:.3f} | {r.spearman_views_adj_last:.3f} | {r.top10pct_overlap:.2f} | {r.top10_share_first:.3f} / {r.top10_share_adj:.3f} / {r.top10_share_last:.3f} | {r.gini_first:.3f} / {r.gini_last:.3f} |")
    w("")
    w("### 3-3. 소급분 나이별 분포")
    w("")
    lc = G["long_cross_section"]
    w(f"거꾸로 받은 한 줄 글. 기준 = 실시간 24시간 줄({lc['base']['n_pairs']:,}글, 나이 중앙 {lc['base']['age_median_h']:.1f}시간)의 log((조회+1)/(24시간 댓글+1)) 중앙값.")
    w("")
    w("| 나이(일) | 글 | 나이 중앙 | 등록일 | 조회 중앙값 | 조회 90% | 24시간 댓글 중앙 | log(조회/24h 댓글) − 기준 | 하루 뒤 댓글 몫 | 7일 뒤 댓글 몫 |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in lc["bins"]:
        w(f"| {r['age_d']} | {r['n']:,} | {r['age_median_d']:.1f} | {r['post_dates']} | {r['views_median']:,.0f} | {r['views_p90']:,.0f} | {r['c24_median']:.0f} | {r['log_views_per_c24_minus_base']:+.3f} | {100*r['late_comment_share_after1d']:.2f}% | {100*r['late_comment_share_after7d']:.2f}% |")
    w("")
    w("- 64~128일 칸(2026-06-12~08-11, 7월 급락기)은 하루 글이 가장 많고 글당 조회가 가장 낮다. 나이가 아니라 시기(글이 많으면 한 글이 받는 조회가 준다) 효과다. 그래서 넓은판 곡선은 '나이 효과 상한'일 뿐이고, 넓은판 (b)가 나이와 상관 0이 된 것은 같은 단면으로 맞췄기 때문(순환)이라 근거가 되지 못한다.")
    w("- 시각으로 잰 늦은 댓글(등록 하루 뒤)은 전체 댓글의 1~3.5%다. 댓글로 보면 '몇 달 뒤 재조명'은 작다. 조회도 그런지는 같은 글을 다시 받아야 알 수 있다.")
    w("")
    w("### 3-4. 날짜(창)별 '조회 ÷ 마감 전 댓글'과 수집 나이")
    w("")
    w("창별 값과 창의 첫 줄 수집 나이 중앙값(log)의 Spearman ρ (창 수). '소급만' = 실시간 글 5% 미만 창.")
    w("")
    w("| 열 | W_KR 전체 | W_KR 소급만 | W_KR 2026 소급 | W_U 전체 | W_U 2026 소급 | 2026 소급 창 글 수와 ρ (KR) |")
    w("|---|---|---|---|---|---|---|")
    for c, lab_ in (("b_vpc_median", "(b) 중앙값"), ("b_vpc_p90", "(b) 상위 10%"), ("b_vpc_adj_short_median", "(b) 나이 보정 짧은판"),
                    ("b_vpc_adj_full_median", "(b) 나이 보정 넓은판(순환)"), ("views_median_first", "조회 중앙값(분자)"),
                    ("c_pre_median", "마감 전 댓글 중앙값(분모)"), ("n_posts", "창 글 수"), ("b_share_c_pre_zero", "마감 전 댓글 0개 글 몫"),
                    ("a_top10_view_share", "(a) 상위 10글 몫"), ("a_top10pct_view_share", "(a) 상위 10% 글 몫"),
                    ("a_view_gini", "(a) 조회 지니"), ("a_up_gini", "(a) 추천 지니"), ("down_ratio_first", "다른의견 비율")):
        npv = npc["kr"].get(c)
        w(f"| {lab_} | {rho_cell(ck_kr[c], 'all')} | {rho_cell(ck_kr[c], 'backward_only')} | {rho_cell(ck_kr[c], 'backward_2026')} | {rho_cell(ck_us[c], 'all')} | {rho_cell(ck_us[c], 'backward_2026')} | {'—' if npv is None else f'{npv:+.2f}'} |")
    w("")
    w("실시간 경계 앞뒤(창 중앙값):")
    w("")
    w("| 열 | W_KR 직전 3주 소급(9/21~) | W_KR 실시간 | W_U 직전 3주 | W_U 실시간 |")
    w("|---|---|---|---|---|")
    for c, lab_ in (("b_vpc_median", "(b) 원값"), ("b_vpc_adj_short_median", "(b) 보정 짧은판"), ("b_vpc_adj_full_median", "(b) 보정 넓은판"),
                    ("views_median_first", "조회 중앙값 첫 줄"), ("views24_short_median", "조회 중앙값 보정"), ("c_pre_median", "마감 전 댓글 중앙값"),
                    ("n_posts", "창 글 수"), ("a_top10_view_share_raw", "상위 10글 몫 첫 줄"), ("a_top10_view_share", "상위 10글 몫 보정"),
                    ("a_top10_view_share_last", "상위 10글 몫 24시간"), ("a_top10pct_view_share", "상위 10% 몫 보정"),
                    ("a_view_gini_raw", "조회 지니 첫 줄"), ("a_view_gini_last", "조회 지니 24시간"), ("a_up_gini", "추천 지니"),
                    ("down_ratio_first", "다른의견 비율 첫 줄"), ("down_ratio_last", "다른의견 비율 24시간")):
        w(f"| {lab_} | {f(bk[c]['before_3w_median'], 3)} ({bk[c]['n_before']}) | {f(bk[c]['realtime_median'], 3)} ({bk[c]['n_realtime']}) | {f(bu[c]['before_3w_median'], 3)} ({bu[c]['n_before']}) | {f(bu[c]['realtime_median'], 3)} ({bu[c]['n_realtime']}) |")
    w("")
    w("- (b)는 소급 창 안에서 나이와 ρ 약 +0.4로 같이 움직인다. 분자(조회)도 분모(마감 전 댓글)도 나이와 같은 방향(2026 소급 ρ +0.51, +0.35)이고, 글 수가 많은 창일수록 (b)가 낮다(ρ −0.41). 나이 효과와 시기 효과를 이 자료로는 가를 수 없다.")
    w("- 실시간 경계에서 (b)는 30분~3시간 나이 보정 뒤에도 소급보다 약 30% 낮다. 남은 차이는 (1) 24시간 뒤에도 조회가 더 쌓이는 몫이거나 (2) 10/13~21 급락·반등기에 댓글이 조회보다 많이 붙은 시기 효과다. 실시간 창을 소급 창과 한 줄로 이어 쓰지 않는다.")
    w("")
    w("### 3-5. 재조명 신호")
    w("")
    w(f"- 창 안에서 (b)가 상위 10%인 소급 글: 하루 뒤 댓글 몫 {vt['late1_share_top']*100:.2f}%(나머지 {vt['late1_share_rest']*100:.2f}%), 7일 뒤 {vt['late7_share_top']*100:.2f}%(나머지 {vt['late7_share_rest']*100:.2f}%), 마감 전 댓글 0개 {vt['c_pre_zero_share_top']*100:.0f}%(나머지 {vt['c_pre_zero_share_rest']*100:.0f}%). (b)가 큰 글에는 늦게 주목받은 글이 더 섞여 있다.")
    w("- 등록 달별 늦은 댓글 몫(하루 뒤 / 7일 뒤): " + ", ".join(
        f"{r['ym']} {100*r['late1_share']:.1f}%/{100*r['late7_share']:.1f}%" for r in L["late_comments_by_month"]) + ". 오래된 글일수록 조금 높다(지운 댓글이 늘어나는 몫도 섞임).")
    w("")
    w("## 4. 창별 특징(`feat_views.parquet`)")
    w("")
    w("| 묶음 | 열 | 정의 |")
    w("|---|---|---|")
    w("| 표지 | `n_posts`, `age_first_median_h`(p10·p90), `share_realtime_posts`, `share_fetched_before_close`, `share_two_rows`, `realtime_window`(실시간 글 95% 이상), `judge_ok`, `realtime`(feat_activity) | 창 = 글 자기 시각이 든 W_KR·W_U 본판 완전 창(로더 산출) |")
    w(f"| (a) | `a_top10_view_share`, `a_top10pct_view_share`, `a_view_gini`, `a_up_gini` (+ `_raw` 첫 줄 원값, `_last` 마지막 줄) | 주판은 나이 보정 조회(views24_short; 소급 창에서는 원값과 순위·몫이 같다). 글 {gm['windows']['min_posts_for_a']}개 미만 창은 결측 |")
    w("| (b) | `b_vpc_median`, `b_vpc_p90`, `b_vpc_adj_short_*`, `b_vpc_adj_full_*`, `b_share_c_pre_zero`, `c_pre_*` | 글마다 조회 ÷ (마감 전 댓글 + 1)의 창 중앙값·90% 분위 |")
    w("| (c) | `c_views_sum_first`, `c_views_median_first`, `c_views_sum_last`, `c_up_sum_first`, `c_down_sum_first` | 실시간 창만 값(나머지 결측) |")
    w("| 추천·다른의견 | `down_ratio_first`(창 합 다른의견 ÷ (추천+다른의견)), `down_ratio_post_median`, `down_ratio_last`, `share_posts_up_or_down` | |")
    w(f"| 예측 층 | `P_bar_top10`(조회 상위 10% 글 단위), `P_bar_top10_withc`(그 글의 댓글 단위 포함), `P_bar_posts_all`(라벨 붙은 글 단위 전체) + 단위 수 `P_n_units_*`, 예측 쌍 수 `P_n_pred_*` | 라벨 = final + preview(16,930단위, 7/1 이후 창의 지식 기간 단위 50개 뺌). 작성자당 1회(창·작성자·pred). P̄ = 예측 쌍(1~5) 평균 − 3. W_U는 단위 시각의 W_U 창으로 옮김(예측 쌍 합 {lab_us['top10_pred_total']}, 값 있는 창 {lab_us['windows_with_top10_pred']}/294) |")
    w("")
    w("## 5. 한계")
    w("")
    w("- 24시간 뒤 조회 증가는 같은 글을 다시 받기 전에는 모른다. 동결 H07 재등록 조건(수집기가 수집 시각을 저장하고 등록 뒤 24±1시간 줄을 120거래일 쌓기)과 같은 결론이다. 지금 자료로 정한 나이 보정은 30시간까지만 근거가 있다.")
    w("- 실시간 첫 줄 시각은 따라잡기 사건 시각으로 놓았다. 한 사건 안에서 번호 순으로 5초씩 늦어지는 몫(중앙값 6분, 최대 38분)은 나누지 않았다. 첫 나이 15분 안의 줄은 이 오차가 g를 크게 바꾼다(g 15분 0.56 vs 6분 0.27).")
    w("- 지운 글·지운 댓글은 없다. 오래된 글은 지운 댓글 몫이 커서(2025 1.5%) 분모(마감 전 댓글)가 조금 작아진다.")
    w("- 조회 상위 10% 글의 P̄는 라벨 표본(창당 60단위)이 그런 글을 거의 뽑지 않아 창 단위 값이 대부분 0~1개 예측이다. 쓰려면 그 글들을 따로 라벨해야 한다.")
    (OUT / "views_leak.md").write_text("\n".join(lines) + "\n", "utf-8")
    print(f"wrote {OUT / 'views_leak.md'} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
