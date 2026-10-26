"""개미 지수 최종 판정 하네스 — 동결 HYPOTHESES.md §2·§5, 부록 1~10을 한 번에 돌린다.

지위: 판정 코드다. 동결 정의·문턱·단어·대상·기간을 바꾸지 않는다. 결과를 보기 전에 정한 해석은 아래 '해석' 상수와
      results/dryrun.md '해석' 절에 적는다.

입력(읽기만)
  특징 표(창별): final/results/windows_{kr,us}, feat_activity, feat_topic, feat_text (run_features.py 산출)
  예측 라벨 창 특징: final/results/pred_windows.parquet가 있으면 그것을 쓴다. 없으면 final·preview 라벨과 대응표로
      메모리에서 만든다(preview/code/pred_windows.py의 group_stats·psd_norm, interim feat_text.rz8를 그대로 불러 씀).
  감정 층(설명용 H60~H65): emotion/window_emotion.parquet
  가격 패널: load_prices(mode) 하나로만 받는다. 진입·결과(fut_*)·M0(m0_*)·Hodrick 1기간 수익(eps_*) 열.
    mode='fake'  실제 가격을 읽지 않는다. 거래일 날짜(loader_calendar·exchange_calendars)만 쓰고 가격은 난수로 만든다.
                 scenario='walk'  무작위 걸음(GARCH 변동) 합성 가격
                 scenario='plant' 같은 합성에 '심어 둔 효과'(특징 → 진입일 KOSPI HL, 스피어만 IC 0.30)를 더한다
    mode='real'  confirm_real=True이고 KST 2026-10-24 06:00(10/23 미국 종가 뒤) 이후에만 연다. 그 전에는 파일을 하나도
                 읽지 않고 멈춘다. 읽은 뒤에도 ^KS11·SOXX에 2026-10-23 종가가 없으면 멈춘다.

판정 규칙(동결 문서 그대로)
  M0: §2.3 확장 OLS(D−1까지, 적용 달 첫 거래일 전 자료, 최소 120창, 월 1회 재적합). X̂ = 가격 열만의 예측값,
      X⊥ = (X − 적합값) ÷ 적합 잔차 SD. 가설별로 더하는 통제가 3개를 넘으면 그 열만 ridge(λ는 GCV, 특징만 사용).
  T1(HAR-K, HAC 5): H03, H08(부록 2: F2 대표), H67, H70. 설명용 계산: H64·H65(감정 층).
  T2(IC, Hodrick 1B): H35, H46, H31(규제 시행일이 있을 때만, §5.1 #11), H66, H68(부록 4). 설명용 계산: H60~H63.
  M0: 한국 §2.3 한국 행, 미국 §2.3 미국 행 + 결번율(§2.2) + 달력 더미(§2.4). CPI·고용·MSCI 더미는 schedule_dates.json이
      있을 때만(없으면 결과 meta에 '빠짐'으로 적는다).
  p: §2.7 ① 2026 특징열 원형 이동 전수(21~N−21거래일)와 ② 해석 p 중 큰 값. 대표는 족 안 특징을 같은 이동 몫으로
     함께 돌린 최대 통계(Westfall-Young). 보정: T1 Holm(α 0.05), T2 BH(q 0.10), 실제 시험 칸 수로.
  통과: 보정 통과 + 앞 구간 같은 부호(라벨 층은 보조) + 강건 3개(급락 4일·기여 상위 2일·열어 본 8일 빼기)에서
        점추정 50% 이상·같은 부호. 기각: 뒤 부호 반대, 90% 상한 < 기대 효과 절반, 메아리 위약. 그 밖은 판정 불가.
  족 안 순차: 대표가 통과한 족만, 대표가 통과한 기준값(국소 α)으로, 고정 순서. 하나가 실패하면 뒤는 설명용.
  전체 원형 이동 200회: 모든 주 칸을 같은 이동 몫으로 돌려 '우연 통과 기대 칸 수'를 잰다.
  실시간 전용(H33·H34·H76~H78)과 전방 트랙 칸은 계산하지 않고 '전방'으로 표시한다.

실행
  py judge.py dryrun [--seeds 20] [--n-global 200]   가짜 주가 시운전 두 가지 → results/dryrun.md, dryrun.json
  py judge.py fake --scenario walk|plant --seed 0     한 번
  py judge.py real --confirm-real [--h31-reg-date YYYY-MM-DD]   10/23 종가 뒤에만(가드)
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
FINAL = HERE.parent
ANT = FINAL.parent
RES = FINAL / "results"
INTERIM = ANT / "interim"
PREVIEW = ANT / "preview"
EMOTION = ANT / "emotion"

KST = ZoneInfo("Asia/Seoul")
ET = ZoneInfo("America/New_York")

# ---------------------------------------------------------------- 동결 숫자(§2.5·§2.7·부록 3·4)
FRONT = (date(2026, 1, 2), date(2026, 5, 29))
BACK = (date(2026, 6, 1), date(2026, 10, 23))
BACK_LABEL = (date(2026, 7, 1), date(2026, 10, 23))     # §2.2 LLM 라벨: 지식 마감 다음 달부터(부록 3·4)
SPAN = (date(2026, 1, 2), date(2026, 10, 23))           # §2.7 ①: 2026 전체 특징열
JUDGE_END = date(2026, 10, 23)                          # §2.5: 판정 구간 끝 = 10/23 종가
REAL_EARLIEST = datetime(2026, 10, 24, 6, 0, tzinfo=KST)  # SOXX 10/23 종가 = KST 10/24 05:00(서머타임) 뒤
CRASH_DAYS = [date(2026, 7, 29), date(2026, 7, 30), date(2026, 7, 31), date(2026, 10, 13)]
JUL3_DAYS = [date(2026, 7, 29), date(2026, 7, 30), date(2026, 7, 31)]   # §3 H03 기각 칸: 7/29~31을 뺀 판
OPENED_DAYS = [date(2026, 8, 20), date(2026, 9, 2), date(2026, 9, 3), date(2026, 9, 8), date(2026, 9, 16),
               date(2026, 10, 12), date(2026, 10, 14), date(2026, 10, 15)]
SPAN_2025 = (date(2025, 1, 1), date(2025, 12, 31))       # §2.5: 2025는 같은 부호 확인 보조(판정에 안 씀)
ALPHA_T1 = 0.05
Q_T2 = 0.10
EXPECTED = {"T1": 0.15, "IC": 0.10, "SEQ": 0.10}
HAC_L = 5
SHIFT_DAYS = 21
MIN_M0 = 120
MIN_BLOCKS = 8
MISSING_MAX = 0.30
RIDGE_MAX_EXTRA = 3
DUMMY_MIN = 2
PLACEBO_RATIO = 0.70
ROBUST_RATIO = 0.50
ECHO_R2 = 0.50
N_GLOBAL = 200
GLOBAL_SEED = 20261023
H04_CORR_MAX = 0.9
H52_FIT_MIN = 20                   # H52 예측 |갭| 확장 회귀 최소 월요일 수(M0 밖 확장 회귀 = H04 B⊥와 같은 20, interim notes)
DR2_SMALL = {"H03": (0.01, "부분 R² 증가 < 0.01 — 실용 효과 작음(§3 H03)"),
             "H08": (0.005, "부분 R² 증가 < 0.005 — 댓글 수만으로 같은 설명력(§3 H08)")}
SCHEDULE_EXTRA = ("cpi", "jobs", "msci")   # §2.4 예정 일정 중 원천 파일(schedule_dates.json)이 있어야 하는 것
EARN_MEM = ("MU", "005930.KS")             # §3 H50 메모리 일정 더미(MU 실적, 삼전 실적일 — 잠정 대신 확정일, interim notes)
PLANT_IC = 0.30
MDE_K = 2.49                       # §2.7 MDE ≈ 2.49/√n_eff (단측 5%, 검정력 80%)

# 일정(§2.4): FOMC 성명일. 2026은 interim/code/h01_preview.FOMC_2026과 같다(단위 테스트로 대조). 2025 하반기는 연준 공개 일정.
FOMC = [date(2025, 9, 17), date(2025, 10, 29), date(2025, 12, 10),
        date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29), date(2026, 6, 17), date(2026, 7, 29),
        date(2026, 9, 16), date(2026, 10, 28), date(2026, 12, 9)]

KR_T = {"kospi": "^KS11", "skh": "000660.KS", "sec": "005930.KS", "kosdaq": "^KQ11"}
US_T = {"soxx": "SOXX", "sox": "^SOX", "qqq": "QQQ", "ewy": "EWY", "mu": "MU", "sndk": "SNDK", "wdc": "WDC"}
VIX_T, NQ_T = "^VIX", "NQ=F"
HORIZONS = (1, 5, 10, 21)
MDD_H = (5, 10, 21)

# ---------------------------------------------------------------- 회귀 열(preview/code/test_free.py와 같은 이름·해석)
# HAR-K(§3 F1). 전일 VKOSPI는 원천 없음(yfinance 0행). 거래대금 이상치는 거래량 이상치로 대신.
HAR_CONT = ["m0_kospi_ln_absr", "m0_kospi_lnRV5", "m0_kospi_lnRV22", "m0_kospi_rneg", "m0_sox_ln_absr_u",
            "m0_ewy_ln_absr_u", "m0_kospi_ln_vol_anom", "m0_kospi_absg", "m0_missing_rate"]
HAR_DUM = ["wd_1", "wd_2", "wd_3", "wd_4", "m0_cal_fomc", "m0_cal_earn", "m0_cal_kr_opex", "m0_cal_us_opex",
           "m0_cal_cpi", "m0_cal_jobs", "m0_cal_msci"]
# M0(§2.3). 한국 |r|은 max·min과 완전 공선이라 뺀다. VKOSPI 없음.
M0_KR_CONT = ["m0_kospi_rpos", "m0_kospi_rneg", "m0_kospi_r5", "m0_kospi_r20", "m0_kospi_r60", "m0_kospi_DD20",
              "m0_kospi_lnRV20", "m0_kospi_ln_vol_anom", "m0_sox_r_u", "m0_ewy_r_u", "m0_missing_rate"]
M0_KR_DUM = ["m0_cal_weekend", "m0_cal_holiday", "m0_cal_xholiday", "m0_cal_vacation", "m0_cal_kr_opex",
             "m0_cal_us_opex", "m0_cal_fomc", "m0_cal_earn", "m0_cal_cpi", "m0_cal_jobs", "m0_cal_msci"]
# 미국 M0: §2.3 미국 행 + 결번율(§2.2 '모든 텍스트 가설의 통제 변수') + 달력 더미(§2.4 '그날을 빼지 않고 더미로 통제')
M0_US_CONT = ["m0_kospi_r_k", "m0_kospi_oc_k", "m0_skh_r_k", "m0_skh_oc_k", "m0_sox_r", "m0_sox_r5", "m0_sox_r20",
              "m0_sox_DD20", "m0_sox_lnRV20", "m0_vix_chg", "m0_nq_close_to_winclose", "m0_missing_rate"]
M0_US_DUM = list(M0_KR_DUM)
KR_IC_CTRL = ["m0_kospi_r", "m0_kospi_r5", "m0_kospi_r20", "m0_kospi_lnRV20", "m0_kospi_g", "m0_sox_r_u"]
H46_CTRL = ["m0_soxx_r", "m0_soxx_absr", "m0_soxx_r5", "m0_soxx_r20", "m0_soxx_lnRV20", "m0_soxx_minus_ewy_r20",
            "m0_kospi_r20_k", "m0_nq_close_to_winclose", "m0_soxx_g"]
H31_CTRL = ["m0_soxx_r", "m0_soxx_r5", "m0_soxx_r20", "m0_soxx_lnRV20", "m0_soxx_g"]
# H36: §3 칸 통제(r20, |r20|, r20·RV20, r5, RV20) + §2.3 방향 판정 회귀 공통(r(D−1), g(D), 전야 SOX)
H36_CTRL = KR_IC_CTRL + ["m0_kospi_absr20", "m0_kospi_r20xRV20"]
H50_CTRL = ["m0_mem3x_r", "m0_mem3x_r5", "m0_mem3x_r20", "m0_mem3x_lnRV20", "m0_mem3x_g"]
H52_CTRL = ["m0_sox_absr_u", "m0_qqq_absr_u", "m0_nq_kr_abs", "m0_kospi_absr"]

# ---------------------------------------------------------------- 칸 등록
# role: main(주 판정) / seq(족 안 순차) / expl(설명용 계산) / stopped(멈춤, 순차에서 건너뜀)
CELLS: list[dict] = [
    dict(id="H03", role="main", fam="T1", family="F1", kind="har", market="kr", x="A_rz8", y="fut_kospi_lnHL",
         h=1, sign=+1, extra=[], companion="A_saved_rz8", rob_extra=[("jul3", JUL3_DAYS)], layer="forum",
         label="글량 이상치 → KOSPI 다음 장 ln HL"),
    dict(id="H08", role="main", fam="T1", family="F2", kind="har", market="kr", x="fight", y="fut_kospi_lnHL",
         h=1, sign=+1, extra=["A_rz8", "ln_n_comments_onposts", "trunc_share"], layer="forum",
         label="댓글 싸움 → KOSPI 다음 장 ln HL (부록 2: F2 대표)"),
    dict(id="H67", role="main", fam="T1", family="H67", kind="har", market="kr", x="P_sd_rz8", y="fut_kospi_lnHL",
         h=1, sign=+1, extra=[], layer="label", label="예측 분산 rz8 → KOSPI 다음 장 ln HL"),
    dict(id="H70", role="main", fam="T1", family="H70", kind="har", market="kr", x="P_share_rz8",
         y="fut_kospi_lnHL", h=1, sign=+1, extra=[], layer="label", label="예측 몫 rz8 → KOSPI 다음 장 ln HL"),
    dict(id="H35", role="main", fam="T2", family="F7", kind="ic", market="kr", x="NW90_rz8", y="fut_kospi_R10",
         ytrans="log1p", h=10, sign=-1, ctrl=KR_IC_CTRL, se="hodrick",
         m0_extra=[("A_rz8", "forum"), ("m0_kospi_absr", "price")], layer="forum", label="신참 비율 → KOSPI R10"),
    dict(id="H46", role="main", fam="T2", family="F9", kind="ic", market="us", x="h46_dtilt", y="fut_soxx_R10",
         ytrans="log1p", h=10, sign=-1, ctrl=H46_CTRL, se="hodrick", mask="h46_valid_win", layer="forum",
         label="관심의 미장 이동 ΔTilt → SOXX R10"),
    dict(id="H31", role="main", fam="T2", family="F6", kind="ic", market="us", x="LL_US_d20", y="fut_soxx_MDD10",
         ytrans=None, h=10, sign=-1, ctrl=H31_CTRL, se="hac", needs_reg_date=True, layer="forum",
         aux=["LL_US_noNVDL_d20"], label="레버리지 롱 언급 20창 변화 → SOXX MDD10 (NVDL은 사후 단어, 뺀 판은 기록)"),
    dict(id="H66", role="main", fam="T2", family="H66", kind="ic", market="kr", x="P_bar", y="fut_kospi_R5",
         ytrans="log1p", h=5, sign=-1, ctrl=KR_IC_CTRL, se="hodrick", layer="label", label="군중 예측 P̄ → KOSPI R5"),
    # 족 안 순차(§3 등급 칸). 사전 측정 실패 칸은 멈춤(§2.6: 건너뛰고 실패로 세지 않음)
    dict(id="H05", role="stopped", family="F1", order=1, why="감정 사전(E·F·FO·ANG 표지) 측정 실패(부록 2 #4)"),
    dict(id="H04", role="seq", fam="T1", family="F1", order=2, kind="har", market="kr", x="Bperp_rz8",
         y="fut_kospi_lnHL", h=1, sign=+1, extra=["A_rz8"], layer="forum", label="참여 폭 B⊥ → KOSPI ln HL"),
    dict(id="H52", role="seq", fam="T1", family="F1", order=3, kind="har_mon", market="kr", x="WK_rz8",
         y="fut_kospi_absgap", h=1, sign=+1, ctrl=[], gap_pred=H52_CTRL, companion="WK_saved_rz8", layer="forum",
         label="주말 곱씹기 → 월요일 KOSPI |갭| 잔차(예측 |갭| 확장 회귀)"),
    dict(id="H36", role="seq", fam="T2", family="F7", order=1, kind="ic", market="kr", x="NEW90_rz8",
         y="fut_kospi_R10", ytrans="log1p", h=10, sign=-1, ctrl=H36_CTRL, se="hodrick", cond="m0_kospi_r20",
         m0_extra=[("m0_kospi_absr20", "price"), ("m0_kospi_absr5", "price"), ("A_rz8", "forum")],
         layer="forum", label="상승 중 새 사람 A4⊥ → KOSPI R10(상승 조건일)"),
    dict(id="H37", role="stopped", family="F7", order=2, why="강세·약세 점수 사전(BULL·BEAR) 측정 실패"),
    dict(id="H50", role="seq", fam="T2", family="F9", order=1, kind="ic", market="us", x="h50_m_z20",
         y="fut_mem3x_R5", ytrans=None, h=5, sign=-1, ctrl=H50_CTRL, se="hodrick",
         m0_extra=[("m0_sec_r_k", "price"), ("m0_sec_oc_k", "price"), ("m0_mu_r", "price"), ("m0_nq_1530", "price"),
                   ("m0_cal_earn_mem", "price")],
         layer="forum", label="낮 메모리 수다 → (MU·SNDK·WDC − SOXX) R5"),
    # 설명용 계산(α 없음)
    dict(id="H64", role="expl", fam="T1", family="-", kind="har", market="kr", x="emo_FEAR_rz8", y="fut_kospi_lnHL",
         h=1, sign=+1, extra=[], layer="label", why="감정 층(부록 3~8: 주된 감정 검수 51% 실패)",
         label="불안·공포 비율 rz8 → KOSPI ln HL"),
    dict(id="H65", role="expl", fam="T1", family="-", kind="har", market="kr", x="emo_negent_rz8",
         y="fut_kospi_lnHL", h=1, sign=+1, extra=[], layer="label", why="감정 층(부록 3~8)",
         label="감정 쏠림(−엔트로피) rz8 → KOSPI ln HL"),
    # H68: 부록 4가 T2 판정 칸으로 등록. 부록 9의 '설명용'은 미리보기 처리라 최종 판정에는 점검표·최소 표본 규칙을 그대로 적용
    dict(id="H68", role="main", fam="T2", family="H68", kind="ic", market="kr", x="P_strong1", y="fut_kospi_R10",
         ytrans="log1p", h=10, sign=+1, ctrl=KR_IC_CTRL, se="hodrick", layer="label",
         label="강한 하락 예측 몫 → KOSPI R10"),
    dict(id="H60", role="expl", fam="T2", family="-", kind="ic", market="kr", x="emo_HYPE", y="fut_kospi_R5",
         ytrans="log1p", h=5, sign=-1, ctrl=KR_IC_CTRL, se="hodrick", layer="label", why="감정 층(부록 8)",
         label="들뜸·자랑 비율 → KOSPI R5"),
    dict(id="H61", role="expl", fam="T2", family="-", kind="ic", market="kr", x="emo_DESP", y="fut_kospi_R10",
         ytrans="log1p", h=10, sign=+1, ctrl=KR_IC_CTRL, se="hodrick", layer="label", why="감정 층(부록 8)",
         label="우울·절망 비율 → KOSPI R10"),
    dict(id="H62", role="expl", fam="T2", family="-", kind="ic", market="kr", x="emo_ANGER", y="fut_kospi_R5",
         ytrans="log1p", h=5, sign=-1, ctrl=KR_IC_CTRL, se="hodrick", layer="label", why="감정 층(부록 8)",
         label="분노·욕 비율 → KOSPI R5"),
    dict(id="H63", role="expl", fam="T2", family="-", kind="ic", market="kr", x="emo_mockbb", y="fut_kospi_R10",
         ytrans="log1p", h=10, sign=-1, ctrl=KR_IC_CTRL, se="hodrick", layer="label", why="감정 층(부록 8)",
         label="약세론 조롱 − 강세론 조롱 → KOSPI R10"),
]
CELL = {c["id"]: c for c in CELLS}

# 계산하지 않는 칸의 상태(동결 문서·부록·점검표 근거)
STATUS_ONLY = {
    "전방(실시간 전용)": {"H33": "§3 실시간 수집분만", "H34": "§3 실시간 수집분만(2027-04 말)",
                    "H76": "부록 10", "H77": "부록 10", "H78": "부록 10"},
    "전방 트랙(설명용 + 전방)": {"H57c": "부록 1", "H59": "부록 1", "H29": "§3 설명용, ±3% 날 20개 뒤",
                          "H31 사건 비": "§3 F6 보조, §5.5"},
    "설명용(감정 층, 부록 3~8: 검수 시험 실패)": {
        "H57a": "부록 3: S 대신 들뜸 지수(감정 층) → 부록 8 실패. 부록 1의 활동량 단계 대체는 부록 2(보류)·부록 3(들뜸 지수)이 대신함",
        "H57b": "같음(부록 1 활동량 단계 대체판은 쓰지 않음)",
        "H69": "감정 × 예측 칸(감정 층)", "H71": "다감정(감정 층)", "H72": "다감정(감정 층)",
        "H73": "다감정(감정 층)", "H74": "다감정(감정 층)", "H75": "다감정(감정 층)"},
    "멈춤(사전 측정 실패, §2.6·부록 2 #4)": {
        "H14": "S⊥(E·F 정밀도 미달), TMP(H02) 미구성", "H19": "환호 어휘 E", "H49": "저녁 S", "H51": "새벽 S",
        "H11": "D_SD·B", "H12": "점수", "H15": "S", "H16": "S", "H21": "FO",
        "H25": "CAP·HES", "H26": "RES·ANG(z)", "H53": "S", "H54·H55": "순차, 대표 H14 멈춤"},
    "설명용(순차 닫힘)": {"H43": "F8 순차 1, 대표 H39+H48 설명용", "H40": "F8 순차 2, 같음", "H41": "F8 순차 3, 같음"},
    "관문·측정(매매 칸 아님)": {
        "H01": "관문. 사전 E·F 정밀도 미달(부록 2 #4)로 S가 측정 실패 → 감정 사전 칸 멈춤(§2.6). 관문 상관은 이 하네스 밖",
        "H02": "측정 선택. TMP 미구성 → 부록 3~8 감정 층으로 대체했으나 검수 실패, 예측 층(부록 4)만 판정에 씀"},
    "설명용(§3 설명용 표, α 없음)": {k: "§3" for k in ("H13", "H20", "H22", "H23", "H27", "H28", "H30", "H38",
                                                    "H45", "H47")},
    "기각(§3 반박 검토)": {k: "§3 기각 목록" for k in ("H07", "H17", "H18", "H32", "H42", "H44", "H56")},
}
# 점검표 이유로 dryrun 표에 따로 쓰는 칸(checklist 'why'): H09, H24, H39+H48, H06·H10
CHECKLIST_ROWS = ("H09", "H24", "H39+H48", "H06", "H10")


# ================================================================ 공용
def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _bool(s: pd.Series) -> np.ndarray:
    return s.astype("boolean").fillna(False).to_numpy(bool)


def _dates(s: pd.Series) -> np.ndarray:
    return pd.to_datetime(s).dt.date.to_numpy()


def in_range(d: np.ndarray, rng: tuple[date, date]) -> np.ndarray:
    return np.array([rng[0] <= x <= rng[1] for x in d], bool)


# ================================================================ 1. 특징 입력
WIN_COLS = ["win_id", "D", "P", "weekday", "open_kst", "close_kst", "length_h", "kind", "main_kind", "over96",
            "xholiday_mismatch", "dst_week", "missing_rate", "outage"]


def load_windows(res: Path = RES) -> dict[str, pd.DataFrame]:
    out = {}
    for mk in ("kr", "us"):
        w = pd.read_parquet(res / f"windows_{mk}.parquet")
        w = w[(w["variant"] == "main") & w["complete"]].sort_values("close_kst").reset_index(drop=True)
        w = w[WIN_COLS].copy()
        w["pos"] = np.arange(len(w))
        w["Dd"] = _dates(w["D"])
        w["Pd"] = _dates(w["P"])
        w["judge_ok"] = ~w["over96"].astype(bool).to_numpy() & ~_bool(w["outage"])
        out[mk] = w
    return out


FEAT_ACT = ["A_rz8", "A_saved_rz8", "fight", "ln_n_comments_onposts", "trunc_share", "NW90_rz8", "NEW90_rz8",
            "Bperp_rz8", "U_rz8", "N_rz8", "WK_rz8", "WK_saved_rz8"]
FEAT_TOPIC_US = ["h46_dtilt", "h46_valid_win", "h50_m_z20"]
FEAT_TEXT_US = ["LL_US_d20", "LL_US_noNVDL_d20"]
FEAT_TEXT_KR = ["D_SD"]


def load_features(windows: dict[str, pd.DataFrame], res: Path = RES) -> tuple[dict[str, pd.DataFrame], dict]:
    fa = pd.read_parquet(res / "feat_activity.parquet")
    ft = pd.read_parquet(res / "feat_topic.parquet")
    fx = pd.read_parquet(res / "feat_text.parquet")
    out = {}
    for mk in ("kr", "us"):
        df = windows[mk].merge(fa[fa["market"] == mk][["win_id"] + FEAT_ACT], on="win_id", how="left", validate="1:1")
        if mk == "us":
            t = ft[ft["table"] == "US_D"][["win_id"] + FEAT_TOPIC_US]
            df = df.merge(t, on="win_id", how="left", validate="1:1")
            df["h46_valid_win"] = _bool(df["h46_valid_win"])
            df = df.merge(fx[fx["market"] == "us"][["win_id"] + FEAT_TEXT_US], on="win_id", how="left",
                          validate="1:1")
        else:
            df = df.merge(fx[fx["market"] == "kr"][["win_id"] + FEAT_TEXT_KR], on="win_id", how="left",
                          validate="1:1")
        assert (df["win_id"].to_numpy() == windows[mk]["win_id"].to_numpy()).all()
        out[mk] = df
    info = {"sha256": {f: sha(res / f) for f in ("feat_activity.parquet", "feat_topic.parquet", "feat_text.parquet",
                                                  "windows_kr.parquet", "windows_us.parquet")}}
    return out, info


def load_pred_features(windows_kr: pd.DataFrame, res: Path = RES) -> tuple[pd.DataFrame, dict]:
    """예측 층 창 특징(부록 4). 정식 표(res/pred_windows.parquet)가 있으면 그대로, 없으면 메모리에서 만든다."""
    p = res / "pred_windows.parquet"
    if p.exists():
        F = pd.read_parquet(p)
        return F, {"source": str(p), "sha256": sha(p), "built_in_memory": False}
    import importlib.util
    if str(INTERIM / "code") not in sys.path:
        sys.path.insert(0, str(INTERIM / "code"))
    import feat_text as FX  # noqa: E402  (rz8·상수: 동결 코드)
    sp = importlib.util.spec_from_file_location("preview_pred_windows", PREVIEW / "code" / "pred_windows.py")
    PW = importlib.util.module_from_spec(sp)        # group_stats·psd_norm: 미리보기 코드 그대로(경로 오염 없이)
    sp.loader.exec_module(PW)
    keys, labs, files = [], [], []
    for kp, pat in ((FINAL / "label_key.parquet", str(FINAL / "labels" / "shard_*.json")),
                    (PREVIEW / "label_key.parquet", str(PREVIEW / "labels" / "pred_*.json"))):
        keys.append(pd.read_parquet(kp))
        for f in sorted(glob.glob(pat)):
            labs += json.loads(Path(f).read_text("utf-8"))
            files.append(f)
    key = pd.concat(keys, ignore_index=True)
    lab = pd.DataFrame(labs)[["id", "pred"]]
    assert key["id"].is_unique and lab["id"].is_unique
    m = key.merge(lab, on="id", how="inner", validate="1:1")
    m = m[m["win_id"].isin(set(windows_kr["win_id"]))].copy()
    # §2.2: 판정 구간(7/1 이후 창)에는 지식 마감 달(6월) 단위를 넣지 않는다(미리보기와 같음). 그 앞 창은 예열·보조용.
    D = pd.to_datetime(m["D"])
    drop = (D >= pd.Timestamp(BACK_LABEL[0])) & (pd.to_datetime(m["dt_kst"]) < pd.Timestamp(BACK_LABEL[0]))
    m = m[~drop].copy()
    m["pred"] = m["pred"].astype(int)
    w = windows_kr.sort_values("close_kst").reset_index(drop=True)
    idx = pd.Index(w["win_id"], name="win_id")
    F, pairs = PW.group_stats(m, "win_id", idx)
    valid = ~_bool(w["outage"])
    holiday = (w["main_kind"] == "holiday").to_numpy()
    wd = w["weekday"].to_numpy()
    PW.psd_norm(F, pairs, valid, PW.POOL_MIN_REG, PW.N_MIN_REG, "P_sd")
    for c in ("P_sd", "P_share"):
        F[f"{c}_rz8"] = FX.rz8(F[c].to_numpy(float), valid, wd, holiday)
    F = F.reset_index()
    info = {"source": "메모리(final+preview 라벨, preview/code/pred_windows.group_stats·psd_norm, feat_text.rz8)",
            "built_in_memory": True, "units_labeled": int(len(m)), "label_files": len(files),
            "units_dropped_june_in_label_window": int(drop.sum()),
            "windows_with_units": int((F["n_units"] > 0).sum()),
            "finite": {c: int(F[c].notna().sum()) for c in ("P_bar", "P_share", "P_share_rz8", "P_sd", "P_sd_rz8",
                                                            "P_strong1")}}
    return F, info


def load_emotion_features(windows_kr: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """감정 층(부록 3 7범주, 묶음 A 분류기) 창 특징 — 설명용 H60~H65만."""
    p = EMOTION / "window_emotion.parquet"
    if not p.exists():
        return pd.DataFrame({"win_id": []}), {"source": None}
    if str(INTERIM / "code") not in sys.path:
        sys.path.insert(0, str(INTERIM / "code"))
    import feat_text as FX  # noqa: E402
    e = pd.read_parquet(p)
    e = e[e["market"] == "kr"]
    w = windows_kr[["win_id", "weekday", "main_kind", "outage"]].merge(e, on="win_id", how="left",
                                                                       suffixes=("", "_e"), validate="1:1")
    valid = ~_bool(w["outage"]) & w["n_units"].notna().to_numpy()
    holiday = (w["main_kind"] == "holiday").to_numpy()
    out = pd.DataFrame({"win_id": w["win_id"],
                        "emo_HYPE": w["r_emo_HYPE"], "emo_DESP": w["r_emo_DESP"], "emo_ANGER": w["r_emo_ANGER"],
                        "emo_mockbb": w["mock_bear_minus_bull"]})
    out["emo_FEAR_rz8"] = FX.rz8(w["r_emo_FEAR"].to_numpy(float), valid, w["weekday"].to_numpy(), holiday)
    out["emo_negent_rz8"] = FX.rz8(-w["entropy_emo"].to_numpy(float), valid, w["weekday"].to_numpy(), holiday)
    return out, {"source": str(p), "sha256": sha(p), "kr_windows": int(e["win_id"].nunique())}


# ================================================================ 2. 가격 패널 — load_prices 하나로만
class RealPriceGuard(RuntimeError):
    pass


def real_allowed(now: datetime | None = None) -> tuple[bool, str]:
    """실제 가격을 열어도 되는가: 10/23 종가(미국은 KST 10/24 05:00)가 들어온 뒤만."""
    now = now or datetime.now(KST)
    if now.tzinfo is None:
        now = now.replace(tzinfo=KST)
    if now < REAL_EARLIEST:
        return False, f"지금 {now.astimezone(KST):%Y-%m-%d %H:%M} KST < {REAL_EARLIEST:%Y-%m-%d %H:%M} KST(10/23 종가 전)"
    return True, "시각 조건 통과"


def _assert_real_allowed(now: datetime | None, confirm: bool) -> None:
    if not confirm:
        raise RealPriceGuard("mode='real'은 confirm_real=True로만 연다(이번 시운전은 가짜 주가만)")
    ok, why = real_allowed(now)
    if not ok:
        raise RealPriceGuard(why)


def _assert_real_data(px: dict[str, pd.DataFrame]) -> None:
    for t in ("^KS11", "SOXX"):
        if t not in px or JUDGE_END not in px[t].index or not np.isfinite(px[t].at[JUDGE_END, "close"]):
            raise RealPriceGuard(f"{t} 2026-10-23 종가 없음 → 판정 실행 불가")


def load_prices(mode: str = "fake", *, windows: dict[str, pd.DataFrame] | None = None, seed: int = 0,
                scenario: str = "walk", plant: dict | None = None, confirm_real: bool = False,
                now: datetime | None = None) -> dict:
    """가격 패널(창 × m0_*·fut_*·eps_*)을 돌려준다. 판정 코드는 가격을 이 함수로만 받는다."""
    if mode == "real":
        _assert_real_allowed(now, confirm_real)          # 아무것도 읽기 전에 가드
        px, nq, cal, src = _read_real_daily()
        _assert_real_data(px)
    elif mode == "fake":
        px, nq, cal, src = _fake_daily(seed, scenario, plant, windows)
    else:
        raise ValueError(mode)
    windows = windows if windows is not None else load_windows()
    panel = build_panel(px, nq, windows, cal)
    src.update({"mode": mode, "scenario": scenario if mode == "fake" else None, "seed": seed,
                "schedule": panel["schedule_meta"]})
    return {"kr": panel["kr"], "us": panel["us"], "meta": src}


# ---------------------------------------------------------------- 2a. 실제(가드 뒤에서만)
def _read_real_daily():
    """판정 실행 때 쓰는 실제 가격: final/results/prices.parquet(ticker,date,OHLCV, 미리보기 prices.py 형식),
    prices_nq_1h.parquet(bar_end_et, close), earnings_dates.json(없으면 yfinance 실적일 날짜만 받는다)."""
    p = RES / "prices.parquet"
    if not p.exists():
        raise RealPriceGuard(f"{p} 없음 — 가격 단계(preview/code/prices.py 방식)를 먼저 돌린다")
    raw = pd.read_parquet(p)
    raw["date"] = pd.to_datetime(raw["date"]).dt.date
    if {"flag_flat_bar", "flag_zero_volume"} <= set(raw.columns):
        bad = raw["flag_flat_bar"].astype(bool) & raw["flag_zero_volume"].astype(bool)
        raw.loc[bad, ["open", "high", "low", "close"]] = np.nan
    px = {t: g.set_index("date").sort_index()[["open", "high", "low", "close", "volume"]].astype(float)
          for t, g in raw.groupby("ticker")}
    nq = pd.read_parquet(RES / "prices_nq_1h.parquet")[["bar_end_et", "close"]]
    ep = RES / "earnings_dates.json"
    if ep.exists():
        earn = {k: [date.fromisoformat(x) for x in v] for k, v in json.loads(ep.read_text("utf-8")).items()
                if isinstance(v, list)}
    else:
        sys.path.insert(0, str(INTERIM / "code"))
        import h01_preview as H  # noqa: E402
        e = H.fetch_earnings()
        ep.write_text(json.dumps({k: ([str(x) for x in v] if isinstance(v, list) else v) for k, v in e.items()},
                                 ensure_ascii=False, indent=1), "utf-8")
        earn = {k: list(v) for k, v in e.items() if isinstance(v, list)}
    cal = {"earn": earn}
    sp = RES / "schedule_dates.json"          # §2.4 CPI·고용·MSCI(사전 공개 일정, 미국 날짜). 없으면 그 더미는 빠진다
    if sp.exists():
        raw_s = json.loads(sp.read_text("utf-8"))
        cal.update({k: [date.fromisoformat(x) for x in raw_s.get(k, [])] for k in SCHEDULE_EXTRA})
    src = {"prices_sha256": sha(p), "nq_sha256": sha(RES / "prices_nq_1h.parquet"),
           "schedule_sha256": sha(sp) if sp.exists() else None}
    return px, nq, cal, src


# ---------------------------------------------------------------- 2b. 가짜(실제 가격을 읽지 않음)
def fake_sessions() -> tuple[list[date], list[date]]:
    """거래일 '날짜'만: loader_calendar(로더가 쓴 거래일, 2024-11-01~2026-10-21) + 그 뒤는 exchange_calendars."""
    cal = pd.read_parquet(RES / "loader_calendar.parquet")
    d = _dates(cal["date"])
    kr = set(d[cal["krx"].to_numpy(bool)])
    us = set(d[cal["nyse"].to_numpy(bool)])
    import exchange_calendars as xc
    for code, s in (("XKRX", kr), ("XNYS", us)):
        extra = xc.get_calendar(code).sessions_in_range(pd.Timestamp(max(d) + timedelta(days=1)),
                                                        pd.Timestamp("2026-12-31"))
        s.update(x.date() for x in extra)
    return sorted(kr), sorted(us)


def _garch(rng: np.random.Generator, n: int, vol: float, a: float = 0.08, b: float = 0.90):
    w = vol ** 2 * (1 - a - b)
    s2 = np.empty(n)
    r = np.empty(n)
    z = rng.standard_normal(n)
    s2[0] = vol ** 2
    for t in range(n):
        if t:
            s2[t] = w + a * r[t - 1] ** 2 + b * s2[t - 1]
        r[t] = math.sqrt(s2[t]) * z[t]
    return r, np.sqrt(s2)


def _draws(rng: np.random.Generator, n: int) -> dict:
    return {"zg": rng.standard_normal(n), "ze": rng.standard_normal(n), "u": rng.random(n),
            "zv": rng.standard_normal(n)}


def _compose(lr: np.ndarray, sig: np.ndarray, dr: dict, p0: float, lnex_add: np.ndarray | None = None,
             vol0: float = 1e7) -> pd.DataFrame:
    """로그수익 → OHLCV. ln(H/L) = |시가→종가| + ex, ex = 0.8σ·exp(0.35·z + 심은 항)."""
    g = 0.35 * lr + 0.25 * sig * dr["zg"]
    oc = lr - g
    c = p0 * np.exp(np.cumsum(lr))
    cprev = c / np.exp(lr)
    o = cprev * np.exp(g)
    ex = 0.8 * sig * np.exp(0.35 * dr["ze"] + (0 if lnex_add is None else lnex_add))
    hi = np.maximum(o, c) * np.exp(dr["u"] * ex)
    lo = np.minimum(o, c) * np.exp(-(1 - dr["u"]) * ex)
    lv = np.zeros(len(lr))
    shock = 1.5 * np.abs(lr) / np.maximum(sig, 1e-6) + 0.3 * dr["zv"]
    for t in range(len(lr)):
        lv[t] = (0.7 * lv[t - 1] if t else 0.0) + shock[t]
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": vol0 * np.exp(lv - lv.mean())})


def _plant_rank_scores(x: np.ndarray) -> np.ndarray:
    r = stats.rankdata(x)
    return stats.norm.ppf((r - 0.5) / len(x))


PLANT_CAL_DRAWS = 40
_PLANT_BETA_CACHE: dict = {}


def _plant_beta(target: float, n_sess: int, si: np.ndarray, q_: np.ndarray, ci: np.ndarray, xok: np.ndarray,
                p0: float) -> float:
    """모집단 IC 맞추기: 씨앗과 무관한 KOSPI 잡음 PLANT_CAL_DRAWS벌(같은 GARCH·봉 규칙)에서 평균 스피어만 IC가
    target이 되는 β(이분법). 특징·달력만 쓰고 결정적이다."""
    key = (round(target, 6), n_sess, len(si), float(np.round(xok.sum(), 6)))
    if key in _PLANT_BETA_CACHE:
        return _PLANT_BETA_CACHE[key]
    sims = []
    for j in range(PLANT_CAL_DRAWS):
        rng = np.random.default_rng(np.random.SeedSequence([777, j]))
        lr, sig = _garch(rng, n_sess, 0.012)
        sims.append((lr, sig, _draws(rng, n_sess)))
    xr = stats.rankdata(xok[ci])

    def mean_ic(beta: float) -> float:
        vals = []
        for lr, sig, dr in sims:
            add = np.zeros(n_sess)
            add[si] = beta * q_
            d = _compose(lr, sig, dr, p0, add)
            hl = np.log(d["high"].to_numpy() / d["low"].to_numpy())[si][ci]
            vals.append(np.corrcoef(xr, stats.rankdata(hl))[0, 1])
        return float(np.mean(vals))

    lo_b, hi_b = 0.0, 6.0
    for _ in range(30):
        mid = 0.5 * (lo_b + hi_b)
        if mean_ic(mid) < target:
            lo_b = mid
        else:
            hi_b = mid
    beta = 0.5 * (lo_b + hi_b)
    _PLANT_BETA_CACHE[key] = beta
    return beta


def _fake_daily(seed: int, scenario: str, plant: dict | None, windows: dict | None):
    kr_s, us_s = fake_sessions()
    rng = np.random.default_rng(np.random.SeedSequence([20261022, seed]))
    nk, nu = len(kr_s), len(us_s)
    fk, sk = _garch(rng, nk, 0.012)
    fu, su = _garch(rng, nu, 0.018)
    kr_pos = {d: i for i, d in enumerate(kr_s)}
    # EWY가 KOSPI와 묶이도록: 같은 날(없으면 0) KR 요인
    fk_on_us = np.array([fk[kr_pos[d]] if d in kr_pos else 0.0 for d in us_s])
    sk_on_us = np.array([sk[kr_pos[d]] if d in kr_pos else 0.012 for d in us_s])
    specs = {  # 이름: (시장, 요인 적재, 고유 변동, 시작가, 상장일)
        "^KS11": ("kr", 1.0, 0.000, 2600.0, None), "000660.KS": ("kr", 1.3, 0.015, 200000.0, None),
        "005930.KS": ("kr", 1.1, 0.012, 60000.0, None), "^KQ11": ("kr", 0.9, 0.008, 750.0, None),
        "^SOX": ("us", 1.0, 0.000, 5000.0, None), "SOXX": ("us", 1.0, 0.002, 230.0, None),
        "QQQ": ("us", 0.55, 0.007, 500.0, None), "EWY": ("us", 0.45, 0.009, 60.0, None),
        "MU": ("us", 1.4, 0.020, 100.0, None), "SNDK": ("us", 1.5, 0.025, 40.0, date(2025, 2, 24)),
        "WDC": ("us", 1.2, 0.020, 60.0, None)}
    px, parts = {}, {}
    for t, (mk, b, s_id, p0, start) in specs.items():
        sess = kr_s if mk == "kr" else us_s
        f, sf = (fk, sk) if mk == "kr" else (fu, su)
        n = len(sess)
        lr = b * f + s_id * rng.standard_normal(n)
        sig = np.sqrt((b * sf) ** 2 + s_id ** 2)
        if t == "EWY":
            lr = lr + 0.5 * fk_on_us
            sig = np.sqrt(sig ** 2 + (0.5 * sk_on_us) ** 2)
        dr = _draws(rng, n)
        parts[t] = (lr, sig, dr, p0, sess)
        df = _compose(lr, sig, dr, p0)
        df.index = sess
        if start is not None:
            df.loc[[d for d in sess if d < start]] = np.nan
        px[t] = df
    # VIX: 로그 OU, QQQ 수익과 반대로 움직인다
    q = parts["QQQ"][0]
    lv = np.empty(nu)
    zv = rng.standard_normal(nu)
    lv[0] = math.log(18)
    for t_ in range(1, nu):
        lv[t_] = 0.95 * lv[t_ - 1] + 0.05 * math.log(18) - 4.0 * q[t_] + 0.04 * zv[t_]
    v = np.exp(lv)
    px[VIX_T] = pd.DataFrame({"open": v, "high": v * 1.02, "low": v * 0.98, "close": v, "volume": 0.0}, index=us_s)
    # NQ=F 60분봉(ET, 봉 끝 시각)
    hrs = pd.date_range("2024-11-01 01:00", "2026-12-31 23:00", freq="h")
    keep = ((hrs.weekday < 4) | ((hrs.weekday == 4) & (hrs.hour <= 17)) | ((hrs.weekday == 6) & (hrs.hour >= 19)))
    hrs = hrs[keep]
    nq = pd.DataFrame({"bar_end_et": hrs, "close": 20000.0 * np.exp(np.cumsum(0.003 * rng.standard_normal(len(hrs))))})
    meta = {"source": "가짜 주가(GARCH 무작위 걸음). 실제 가격 파일을 읽지 않음", "sessions_kr": [str(kr_s[0]), str(kr_s[-1])],
            "sessions_us": [str(us_s[0]), str(us_s[-1])]}
    if scenario == "plant":
        if plant is None:
            raise ValueError("plant 시나리오에는 plant={'x': 창별 특징 Series(win_id), 'windows': 창 표}가 필요하다")
        lr, sig, dr, p0, sess = parts["^KS11"]
        w = plant["windows"]
        x = w["win_id"].map(plant["x"]).to_numpy(float)
        ok = np.isfinite(x) & np.array([d in kr_pos for d in w["Dd"]])
        si = np.array([kr_pos[d] for d in w["Dd"][ok]])
        q_ = _plant_rank_scores(x[ok])
        calib = ok.copy()
        calib[ok] = in_range(w["Dd"].to_numpy()[ok], SPAN) & w["judge_ok"].to_numpy()[ok]
        ci = calib[ok]

        def ic_of(beta: float, lr_, sig_, dr_) -> float:
            add = np.zeros(len(sess))
            add[si] = beta * q_
            d = _compose(lr_, sig_, dr_, p0, add)
            hl = np.log(d["high"].to_numpy() / d["low"].to_numpy())[si]
            return float(stats.spearmanr(x[ok][ci], hl[ci])[0])

        target = plant.get("ic", PLANT_IC)
        beta = _plant_beta(target, len(sess), si, q_, ci, x[ok], p0)
        add = np.zeros(len(sess))
        add[si] = beta * q_
        df = _compose(lr, sig, dr, p0, add)
        df.index = sess
        px["^KS11"] = df
        meta.update({"plant": {"feature": plant.get("name"), "target": "KOSPI 진입일 HL(ln 고가÷저가)",
                               "ic_target_population": target, "beta": beta,
                               "ic_realized_2026": ic_of(beta, lr, sig, dr), "ic_beta0": ic_of(0.0, lr, sig, dr),
                               "n_planted": int(ok.sum()), "n_calib": int(ci.sum()),
                               "calibration": f"β는 씨앗과 무관한 잡음 {PLANT_CAL_DRAWS}벌의 평균 IC가 목표가 되게 정한다"
                                              "(모집단 IC). 씨앗별 실현 IC는 그 둘레에서 흔들린다"}})
    return px, nq, {}, meta


# ---------------------------------------------------------------- 2c. 패널(진짜·가짜 공용)
def _sess_feat(df: pd.DataFrame) -> pd.DataFrame:
    """세션 d의 값: d 종가까지(뒤로만 보는 창). g = d 시가 ÷ 직전 종가 − 1(진입 순간 값)."""
    c, o, v = (df[k].astype(float) for k in ("close", "open", "volume"))
    r = c / c.shift(1) - 1
    lr = np.log(c / c.shift(1))
    out = pd.DataFrame(index=df.index)
    out["r"] = r
    out["rpos"] = r.clip(lower=0)
    out["rneg"] = r.clip(upper=0)
    out["absr"] = r.abs()
    out["ln_absr"] = np.log(r.abs().where(r.abs() > 0))
    for k in (5, 20, 60):
        out[f"r{k}"] = c / c.shift(k) - 1
    out["DD20"] = c / c.rolling(20, min_periods=18).max() - 1
    out["lnRV5"] = np.log(lr.rolling(5, min_periods=4).std(ddof=1))
    out["lnRV20"] = np.log(lr.rolling(20, min_periods=18).std(ddof=1))
    out["lnRV22"] = np.log(lr.rolling(22, min_periods=20).std(ddof=1))
    vv = v.where(v > 0)
    out["ln_vol_anom"] = np.log(vv / vv.shift(1).rolling(20, min_periods=15).mean())
    out["g"] = o / c.shift(1) - 1
    out["oc"] = c / o - 1
    out["lr"] = lr
    out["close"] = c
    return out


def _fut(df: pd.DataFrame, Ds: np.ndarray, prefix: str, end_limit: date) -> dict[str, np.ndarray]:
    """진입 = D 시가. Rh·MDDh(경로 [시가, 종가…])·HL·eps(D 종가 대 직전 종가 로그, Hodrick 분산용).
    끝 세션이 end_limit(10/23)을 넘는 목표는 결측(§2.5)."""
    idx = {d: i for i, d in enumerate(df.index)}
    o, c, h, l = (df[k].to_numpy(float) for k in ("open", "close", "high", "low"))
    sess = list(df.index)
    n = len(Ds)
    out = {f"fut_{prefix}_R{hh}": np.full(n, np.nan) for hh in HORIZONS}
    out.update({f"fut_{prefix}_MDD{hh}": np.full(n, np.nan) for hh in MDD_H})
    for k in ("HL", "lnHL", "absgap"):
        out[f"fut_{prefix}_{k}"] = np.full(n, np.nan)
    out[f"eps_{prefix}"] = np.full(n, np.nan)
    for j, D in enumerate(Ds):
        i = idx.get(D)
        if i is None or not np.isfinite(o[i]) or o[i] <= 0:
            continue
        for hh in HORIZONS:
            e = i + hh - 1
            if e < len(sess) and sess[e] <= end_limit and np.isfinite(c[e]):
                out[f"fut_{prefix}_R{hh}"][j] = c[e] / o[i] - 1
        for hh in MDD_H:
            e = i + hh - 1
            if e < len(sess) and sess[e] <= end_limit and np.isfinite(c[e]):
                path = np.concatenate([[o[i]], c[i:e + 1]])
                okp = np.isfinite(path)
                peak = np.fmax.accumulate(np.where(okp, path, -np.inf))
                out[f"fut_{prefix}_MDD{hh}"][j] = min(float(np.min(path[okp] / peak[okp] - 1)), 0.0)
        if sess[i] <= end_limit and np.isfinite(h[i]) and np.isfinite(l[i]) and l[i] > 0:
            hl = math.log(h[i] / l[i])
            out[f"fut_{prefix}_HL"][j] = hl
            out[f"fut_{prefix}_lnHL"][j] = math.log(hl) if hl > 0 else np.nan
        if i > 0 and np.isfinite(c[i]) and np.isfinite(c[i - 1]) and sess[i] <= end_limit:
            out[f"eps_{prefix}"][j] = math.log(c[i] / c[i - 1])
            out[f"fut_{prefix}_absgap"][j] = abs(o[i] / c[i - 1] - 1)
    return out


def _last_le(sessions: list[date], ds: np.ndarray) -> list[date | None]:
    arr = np.array(sessions, dtype="datetime64[D]")
    q = np.array(list(ds), dtype="datetime64[D]")
    k = np.searchsorted(arr, q, side="right") - 1
    return [sessions[i] if i >= 0 else None for i in k]


def _nq_at(nq: pd.DataFrame, ts: list[pd.Timestamp]) -> np.ndarray:
    t = pd.to_datetime(nq["bar_end_et"]).to_numpy("datetime64[m]")
    c = nq["close"].to_numpy(float)
    q = np.array([np.datetime64(x, "m") if x is not None else np.datetime64("NaT") for x in ts])
    k = np.searchsorted(t, q, side="right") - 1
    return np.where(k >= 0, c[np.clip(k, 0, None)], np.nan)


def _kst_to_et(d: date, hh: int, mm: int) -> pd.Timestamp:
    return pd.Timestamp(datetime(d.year, d.month, d.day, hh, mm, tzinfo=KST).astimezone(ET).replace(tzinfo=None))


def _nth_weekday(y: int, m: int, wd: int, n: int) -> date:
    d = date(y, m, 1)
    d += timedelta(days=(wd - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _prev_session(d: date, s: set) -> date:
    while d not in s:
        d -= timedelta(days=1)
    return d


def schedule(kr_s: list[date], us_s: list[date], cal: dict | list | None) -> dict:
    """§2.4 예정 일정. cal = {'earn': {티커: [날짜]} 또는 [날짜], 'cpi'/'jobs'/'msci': [날짜]}(없는 항목은 빈 목록)."""
    ks, us = set(kr_s), set(us_s)
    cal = {"earn": cal} if isinstance(cal, list) else dict(cal or {})
    earn = cal.get("earn") or {}
    if isinstance(earn, dict):
        earn_all = sorted({d for v in earn.values() for d in v})
        earn_mem = sorted({d for t in EARN_MEM for d in earn.get(t, [])})
    else:
        earn_all, earn_mem = sorted(set(earn)), []
    months = sorted({(d.year, d.month) for d in kr_s if date(2025, 1, 1) <= d <= date(2026, 12, 31)})
    out = {"kr_opex": [_prev_session(_nth_weekday(y, m, 3, 2), ks) for y, m in months],
           "us_opex": [_prev_session(_nth_weekday(y, m, 4, 3), us) for y, m in months],
           "fomc": FOMC, "earn": earn_all, "earn_mem": earn_mem}
    for k in SCHEDULE_EXTRA:
        out[k] = sorted(cal.get(k) or [])
    return out


def _in_window(es: list[date], P: date, D: date) -> float:
    return float(any(P <= e <= D - timedelta(days=1) for e in es))


def _cal_cols(sch: dict, P: np.ndarray, D: np.ndarray, kind: np.ndarray, xhol: np.ndarray) -> dict[str, np.ndarray]:
    """창 종류·달력 더미(§2.4). 국내·미국 창 공용(같은 M0 더미 목록)."""
    cols = {"m0_cal_weekend": (kind == "weekend").astype(float), "m0_cal_holiday": (kind == "holiday").astype(float),
            "m0_cal_xholiday": xhol.astype(bool).astype(float),
            "m0_cal_vacation": np.array([float(d.month in (1, 2, 7, 8)) for d in D])}
    for nm in ("kr_opex", "us_opex", "fomc", "earn", "earn_mem") + SCHEDULE_EXTRA:
        cols[f"m0_cal_{nm}"] = np.array([_in_window(sch[nm], p_, d_) for p_, d_ in zip(P, D)])
    return cols


def build_panel(px: dict[str, pd.DataFrame], nq: pd.DataFrame, windows: dict[str, pd.DataFrame],
                cal: dict | list | None) -> dict[str, pd.DataFrame]:
    kr_s = sorted(d for d in px["^KS11"].index)
    us_s = sorted(d for d in px["^SOX"].index)
    sch = schedule(kr_s, us_s, cal)
    SF = {t: _sess_feat(df) for t, df in px.items()}
    # 메모리 3종 − SOXX 지수(H50 통제용): 하루 로그수익 = 세 종목 평균 − SOXX
    m3 = pd.concat([SF[t]["lr"] for t in ("MU", "SNDK", "WDC")], axis=1).mean(axis=1) - SF["SOXX"]["lr"]
    g3 = pd.concat([SF[t]["g"] for t in ("MU", "SNDK", "WDC")], axis=1).mean(axis=1) - SF["SOXX"]["g"]
    idx3 = np.exp(m3.fillna(0).cumsum())
    mem3 = pd.DataFrame({"close": idx3, "open": idx3, "volume": 1.0})
    SF["MEM3X"] = _sess_feat(mem3)
    SF["MEM3X"]["g"] = g3
    out = {}
    # ---------------- 국내 창
    w = windows["kr"].copy()
    D, P = w["Dd"].to_numpy(), w["Pd"].to_numpy()
    cols: dict[str, np.ndarray] = {}
    for k, t in KR_T.items():
        f = SF[t].reindex(P)
        for c in ("r", "rpos", "rneg", "r5", "r20", "r60", "DD20", "lnRV20", "absr"):
            cols[f"m0_{k}_{c}"] = f[c].to_numpy(float)
        if k == "kospi":
            for c in ("ln_absr", "lnRV5", "lnRV22", "ln_vol_anom"):
                cols[f"m0_kospi_{c}"] = f[c].to_numpy(float)
        cols[f"m0_{k}_g"] = SF[t]["g"].reindex(D).to_numpy(float)
    cols["m0_kospi_absg"] = np.abs(cols["m0_kospi_g"])
    cols["m0_kospi_absr20"] = np.abs(cols["m0_kospi_r20"])
    cols["m0_kospi_absr5"] = np.abs(cols["m0_kospi_r5"])
    cols["m0_kospi_r20xRV20"] = cols["m0_kospi_r20"] * np.exp(cols["m0_kospi_lnRV20"])
    u = _last_le(us_s, np.array([d - timedelta(days=1) for d in D]))
    for k, t in (("sox", "^SOX"), ("ewy", "EWY"), ("soxx", "SOXX"), ("qqq", "QQQ")):
        f = SF[t].reindex(u)
        cols[f"m0_{k}_r_u"] = f["r"].to_numpy(float)
        cols[f"m0_{k}_ln_absr_u"] = f["ln_absr"].to_numpy(float)
        cols[f"m0_{k}_absr_u"] = f["absr"].to_numpy(float)
    vx = px[VIX_T]["close"].astype(float)
    cols["m0_vix_u"] = vx.reindex(u).to_numpy(float)
    cols["m0_vix_chg_u"] = (vx - vx.shift(1)).reindex(u).to_numpy(float)
    a = _nq_at(nq, [pd.Timestamp(datetime(x.year, x.month, x.day, 16, 0)) if x else None for x in u])
    b = _nq_at(nq, [_kst_to_et(d, 7, 59) for d in D])
    cols["m0_nq_kr"] = b / a - 1
    cols["m0_nq_kr_abs"] = np.abs(cols["m0_nq_kr"])
    cols.update(_cal_cols(sch, P, D, w["kind"].to_numpy(), w["xholiday_mismatch"].to_numpy()))
    cols["m0_missing_rate"] = w["missing_rate"].to_numpy(float)
    wd = np.array([d.weekday() for d in D])
    for k in (1, 2, 3, 4):
        cols[f"wd_{k}"] = (wd == k).astype(float)
    for k, t in KR_T.items():
        cols.update(_fut(px[t], D, k, JUDGE_END))
    out["kr"] = pd.concat([w.reset_index(drop=True), pd.DataFrame(cols)], axis=1)
    # ---------------- 미국 창
    w = windows["us"].copy()
    D, P = w["Dd"].to_numpy(), w["Pd"].to_numpy()
    cols = {}
    for k, t in (("soxx", "SOXX"), ("sox", "^SOX"), ("qqq", "QQQ"), ("ewy", "EWY"), ("mu", "MU"),
                 ("mem3x", "MEM3X")):
        f = SF[t].reindex(P)
        for c in ("r", "absr", "rpos", "rneg", "r5", "r20", "DD20", "lnRV20"):
            cols[f"m0_{k}_{c}"] = f[c].to_numpy(float)
        cols[f"m0_{k}_g"] = SF[t]["g"].reindex(D).to_numpy(float)
    cols["m0_soxx_minus_ewy_r20"] = cols["m0_soxx_r20"] - cols["m0_ewy_r20"]
    cols["m0_vix"] = vx.reindex(P).to_numpy(float)
    cols["m0_vix_chg"] = (vx - vx.shift(1)).reindex(P).to_numpy(float)
    k_ = _last_le(kr_s, D)
    for k, t in (("kospi", "^KS11"), ("skh", "000660.KS"), ("sec", "005930.KS")):
        f = SF[t].reindex(k_)
        cols[f"m0_{k}_r_k"] = f["r"].to_numpy(float)
        cols[f"m0_{k}_oc_k"] = f["oc"].to_numpy(float)
    cols["m0_kospi_r20_k"] = SF["^KS11"]["r20"].reindex(k_).to_numpy(float)
    a = _nq_at(nq, [pd.Timestamp(datetime(x.year, x.month, x.day, 16, 0)) for x in P])
    b = _nq_at(nq, [pd.Timestamp(datetime(x.year, x.month, x.day, 8, 30)) for x in D])
    c15 = _nq_at(nq, [_kst_to_et(x, 15, 30) for x in D])
    cols["m0_nq_close_to_winclose"] = b / a - 1
    cols["m0_nq_1530"] = b / c15 - 1
    cols.update(_cal_cols(sch, P, D, w["kind"].to_numpy(), w["xholiday_mismatch"].to_numpy()))
    cols["m0_missing_rate"] = w["missing_rate"].to_numpy(float)
    for k, t in (("soxx", "SOXX"), ("sox", "^SOX"), ("qqq", "QQQ"), ("ewy", "EWY"), ("mu", "MU"), ("sndk", "SNDK"),
                 ("wdc", "WDC")):
        cols.update(_fut(px[t], D, k, JUDGE_END))
    cols["fut_mem3x_R5"] = (np.mean([cols[f"fut_{k}_R5"] for k in ("mu", "sndk", "wdc")], axis=0)
                            - cols["fut_soxx_R5"])
    cols["eps_mem3x"] = (np.mean([cols[f"eps_{k}"] for k in ("mu", "sndk", "wdc")], axis=0) - cols["eps_soxx"])
    out["us"] = pd.concat([w.reset_index(drop=True), pd.DataFrame(cols)], axis=1)
    out["schedule_meta"] = {"n_dates": {k: len(v) for k, v in sch.items()},
                            "missing": [k for k in ("earn", "earn_mem") + SCHEDULE_EXTRA if not sch[k]],
                            "not_available": ["VKOSPI(전일·1일 변화)", "거래대금(거래량 이상치로 대신)", "ES=F·BTC(H52)",
                                              "KOSPI200 야간선물(H51)"]}
    return out


# ================================================================ 3. 통계 도구
def control_design(d: pd.DataFrame, cont: list[str], dums: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    """상수 뒤에 연속 통제·더미를 하나씩 더한다. 랭크를 늘리지 않는 열, 1(또는 0)이 2개 미만인 더미는 뺀다.
    빼는 결정은 통제 열과 표본에만 기대고 결과(y)·특징(x)과 무관하다."""
    n = len(d)
    cols = [np.ones(n)]
    names, dropped = [], []
    rank = 1
    for c in cont + dums:
        v = d[c].to_numpy(float)
        if c in dums:
            ones = int(np.nansum(v))
            if ones < DUMMY_MIN or n - ones < DUMMY_MIN:
                dropped.append(c)
                continue
        trial = np.column_stack(cols + [v])
        rk = np.linalg.matrix_rank(trial)
        if rk > rank:
            cols.append(v)
            names.append(c)
            rank = rk
        else:
            dropped.append(c)
    return np.column_stack(cols), names, dropped


def _resid(Q: np.ndarray, V: np.ndarray) -> np.ndarray:
    return V - Q @ (Q.T @ V)


def _qr(C: np.ndarray) -> np.ndarray:
    return np.linalg.qr(C)[0]


def _lag_pairs(pos: np.ndarray, L: int) -> list[tuple[np.ndarray, np.ndarray, float]]:
    where = {int(p): i for i, p in enumerate(pos)}
    out = []
    for l in range(1, L + 1):
        ii, jj = [], []
        for i, p in enumerate(pos):
            j = where.get(int(p) - l)
            if j is not None:
                ii.append(i)
                jj.append(j)
        out.append((np.array(ii, int), np.array(jj, int), 1.0 - l / (L + 1.0)))
    return out


def fwl_hac(EX: np.ndarray, ey: np.ndarray, pos: np.ndarray, L: int, kfull: int) -> dict:
    """특징 열마다(FWL 잔차) 계수·Newey–West(Bartlett) t. 지연 짝은 창 위치 차가 정확히 l인 행끼리.
    전체 회귀의 HAC 분산과 같다(FWL). 작은 표본 보정 n/(n−k)."""
    n = len(ey)
    sxx = (EX ** 2).sum(0)
    sxy = EX.T @ ey
    beta = sxy / sxx
    E = ey[:, None] - EX * beta
    U = EX * E
    S = (U ** 2).sum(0)
    for ii, jj, w in _lag_pairs(pos, L):
        if len(ii):
            S = S + 2 * w * (U[ii] * U[jj]).sum(0)
    var = S / sxx ** 2 * n / (n - kfull)
    se = np.sqrt(np.maximum(var, 0))
    pr = sxy / np.sqrt(sxx * (ey @ ey))
    return {"beta": beta, "se": se, "t": beta / se, "pr": pr, "df": n - kfull}


def fwl_hodrick(EZ: np.ndarray, ey: np.ndarray, pos: np.ndarray, eps_all: np.ndarray, h: int, kfull: int) -> dict:
    """Hodrick(1992) 1B: h기간 수익 = 1기간 수익 h개의 합이라는 귀무 아래 점수 Σ_t X_t e_t = Σ_s ε_s w_s,
    w_s = Σ_{j<h} X_{s−j}(표본 행만). ε_s = 위치 s의 1기간 로그수익 − 평균. preview/test_free.hodrick_cov의 FWL판."""
    n = len(ey)
    sxx = (EZ ** 2).sum(0)
    beta = (EZ.T @ ey) / sxx
    pmin, pmax = int(pos.min()), int(pos.max())
    hi = min(pmax + h - 1, len(eps_all) - 1)
    span = hi - pmin + 1
    Zf = np.zeros((span + h, EZ.shape[1]))
    Zf[pos - pmin + h] = EZ                      # h칸 앞 여백
    cs = np.cumsum(Zf, axis=0)
    W = cs[h:h + span] - cs[0:span]              # W[s] = Σ_{j<h} Z[s−j]
    eps = eps_all[pmin:hi + 1]
    ok = np.isfinite(eps)
    mu = eps[ok].mean()
    S = (((eps[ok] - mu) ** 2)[:, None] * W[ok] ** 2).sum(0)
    var = S / sxx ** 2 * n / (n - kfull)
    se = np.sqrt(np.maximum(var, 0))
    return {"beta": beta, "se": se, "t": beta / se, "df": n - kfull}


def rank_cols(M: np.ndarray) -> np.ndarray:
    return stats.rankdata(M, axis=0)


def col_corr(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (A.T @ b) / np.sqrt((A ** 2).sum(0) * (b @ b))


def n_blocks(pos: np.ndarray, h: int) -> int:
    """h일 넘게 떨어진 독립 묶음 수(앞에서부터 욕심껏)."""
    cnt, last = 0, -10 ** 9
    for p in np.sort(pos):
        if p - last > h:
            cnt += 1
            last = p
    return cnt


def n_eff_bartlett(n: int, rho: np.ndarray, h: int) -> np.ndarray:
    r = np.where(np.isfinite(rho), np.maximum(rho, 0), 0.0)
    den = 1 + 2 * sum((r ** k) * (1 - k / h) for k in range(1, h))
    return np.minimum(n, n / den)


def ac1_cols(M: np.ndarray, pos: np.ndarray) -> np.ndarray:
    where = {int(p): i for i, p in enumerate(pos)}
    ii, jj = [], []
    for i, p in enumerate(pos):
        j = where.get(int(p) - 1)
        if j is not None:
            ii.append(i)
            jj.append(j)
    if len(ii) < 4:
        return np.full(M.shape[1], np.nan)
    a, b = M[jj], M[ii]
    a = a - a.mean(0)
    b = b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))


def overlap_mask(df: pd.DataFrame, days: list[date]) -> np.ndarray:
    """창 (open, close)이 달력일 [d 00:00, d+1 00:00)과 겹치면 True(h01·미리보기와 같은 '가장 많이 빼는' 읽기)."""
    o = pd.to_datetime(df["open_kst"])
    c = pd.to_datetime(df["close_kst"])
    m = np.zeros(len(df), bool)
    for d in days:
        a = pd.Timestamp(d)
        m |= ((o < a + pd.Timedelta(days=1)) & (c >= a)).to_numpy()
    return m


def holm(ps: dict[str, float], alpha: float) -> tuple[dict, dict]:
    """조정 p와 국소 문턱(통과한 칸만: α/(m−i+1))."""
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, thr, run, going = {}, {}, 0.0, True
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        adj[k] = run
        if going and p <= alpha / (m - i):
            thr[k] = alpha / (m - i)
        else:
            going = False
    return adj, thr


def bh(ps: dict[str, float], q: float) -> tuple[dict, dict]:
    """q값과 국소 문턱(통과한 칸: k*·q/m)."""
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m = len(items)
    qs = [min(1.0, p * m / (i + 1)) for i, (_, p) in enumerate(items)]
    for i in range(m - 2, -1, -1):
        qs[i] = min(qs[i], qs[i + 1])
    kstar = max([i + 1 for i, (_, p) in enumerate(items) if p <= (i + 1) * q / m], default=0)
    adj = {k: qs[i] for i, (k, _) in enumerate(items)}
    thr = {k: kstar * q / m for i, (k, _) in enumerate(items) if i < kstar}
    return adj, thr


# ================================================================ 4. M0 확장 OLS(§2.3)
def expanding_m0(df: pd.DataFrame, X: np.ndarray, cont: list[str], dums: list[str],
                 extras: list[tuple[str, str]], fit_ok: np.ndarray, min_n: int = MIN_M0) -> dict:
    """X(n×R, 원형 이동한 특징 열들)를 M0에 확장 OLS. 달 m의 창은 m 첫날 전 창(D < m-01)으로 적합한 계수를 쓴다.
    적합 표본: fit_ok(판정 자격·표지·특징 유한) & M0 열 유한, 최소 min_n. 더하는 통제가 RIDGE_MAX_EXTRA를 넘으면
    그 열만 표준화해 ridge(λ = GCV, 첫 열(관측 특징)로 고르고 모든 열에 같은 λ). X̂ = 상수·M0·가격 통제 몫,
    X⊥ = (X − 전체 적합) ÷ 적합 잔차 SD(자유도 n − k, ridge는 n − tr H)."""
    n, R = X.shape
    ex_names = [e for e, _ in extras]
    price_ex = {e for e, k in extras if k == "price"}
    allc = cont + dums + ex_names
    Zall = df[allc].to_numpy(float)
    Dt = pd.to_datetime(df["D"]).to_numpy("datetime64[D]")
    mon = Dt.astype("datetime64[M]")
    xhat = np.full((n, R), np.nan)
    xperp = np.full((n, R), np.nan)
    info = []
    zfin = np.isfinite(Zall).all(1)
    for m in np.unique(mon):
        start = m.astype("datetime64[D]")
        fit = fit_ok & zfin & (Dt < start)
        rows = mon == m
        rec = {"month": str(m), "n_fit": int(fit.sum())}
        if fit.sum() < min_n:
            rec["status"] = f"적합 창 {int(fit.sum())} < {min_n}"
            info.append(rec)
            continue
        d = df[fit]
        Dm, names, dropped = control_design(d, cont + ex_names, dums)
        kept = names
        ek = [c for c in kept if c in ex_names]
        P = rows & np.isfinite(df[kept].to_numpy(float)).all(1) if kept else rows.copy()
        Dp = np.column_stack([np.ones(int(P.sum()))] + [df.loc[P, c].to_numpy(float) for c in kept])
        Xf = X[fit]
        if len(ek) > RIDGE_MAX_EXTRA:
            j_ex = [1 + kept.index(c) for c in ek]
            mu = Dm[:, j_ex].mean(0)
            sd = Dm[:, j_ex].std(0, ddof=1)
            Dm = Dm.copy()
            Dp = Dp.copy()
            Dm[:, j_ex] = (Dm[:, j_ex] - mu) / sd
            Dp[:, j_ex] = (Dp[:, j_ex] - mu) / sd
            pen = np.zeros(Dm.shape[1])
            pen[j_ex] = 1.0
            best = None
            for lam in 10.0 ** np.arange(-2, 3.01, 0.5):
                A = Dm.T @ Dm + lam * np.diag(pen)
                Hm = Dm @ np.linalg.solve(A, Dm.T)
                trh = float(np.trace(Hm))
                res0 = Xf[:, 0] - Hm @ Xf[:, 0]
                gcv = len(Xf) * float(res0 @ res0) / (len(Xf) - trh) ** 2
                if best is None or gcv < best[0]:
                    best = (gcv, lam, trh)
            _, lam, trh = best
            beta = np.linalg.solve(Dm.T @ Dm + lam * np.diag(pen), Dm.T @ Xf)
            dof = len(Xf) - trh
            rec.update({"ridge": True, "lambda": float(lam), "n_extra": len(ek)})
        else:
            beta = np.linalg.lstsq(Dm, Xf, rcond=None)[0]
            dof = len(Xf) - Dm.shape[1]
        res = Xf - Dm @ beta
        sd_r = np.sqrt((res ** 2).sum(0) / dof)
        price_j = [0] + [1 + i for i, c in enumerate(kept) if c not in ex_names or c in price_ex]
        full = Dp @ beta
        xhat[P] = Dp[:, price_j] @ beta[price_j]
        xperp[P] = (X[P] - full) / sd_r
        x0 = Xf[:, 0]
        rec.update({"k": int(Dm.shape[1]), "kept": kept, "dropped": dropped,
                    "R2_obs": float(1 - (res[:, 0] @ res[:, 0]) / ((x0 - x0.mean()) @ (x0 - x0.mean())))})
        info.append(rec)
    return {"xhat": xhat, "xperp": xperp, "months": info}


# ================================================================ 5. 칸 계산(모든 원형 이동을 한꺼번에)
def _rot_index(N: int) -> np.ndarray:
    """열 r = r칸 원형 이동(np.roll(a, r)): new[i] = old[(i − r) mod N]."""
    return (np.arange(N)[:, None] - np.arange(N)[None, :]) % N


def _rotate(x0: np.ndarray, F: np.ndarray, R: int) -> np.ndarray:
    """x0을 F(시간순 위치) 안에서 0..R−1칸 원형 이동한 열들. F 밖 값은 그대로."""
    X = np.repeat(x0[:, None], R, axis=1)
    if len(F):
        N = len(F)
        cols = np.round(np.arange(R) * N / R).astype(int) % N if R != N else np.arange(N)
        idx = (np.arange(N)[:, None] - cols[None, :]) % N
        X[F, :] = x0[F][idx]
    return X


class CellRun:
    """한 칸의 관측(r=0)과 모든 원형 이동(r=1..N−1) 통계."""

    def __init__(self, spec: dict, df: pd.DataFrame, opts: dict):
        self.s = spec
        self.df = df
        self.opts = opts
        self.status = "시험"
        self.reasons: list[str] = []
        self.info: dict = {}
        self.arr: dict[str, np.ndarray] = {}
        self.N = 0

    # -------------------------------------------------- 표본
    def _masks(self):
        s, df = self.s, self.df
        d = df["Dd"].to_numpy()
        base = df["judge_ok"].to_numpy(bool).copy()
        if s.get("mask"):
            base &= df[s["mask"]].astype(bool).to_numpy()
        if s.get("needs_reg_date") and self.opts.get("h31_reg_date"):
            rd = self.opts["h31_reg_date"]
            p0 = int(np.searchsorted(np.array(d, dtype="datetime64[D]"), np.datetime64(rd)))
            base &= np.abs(df["pos"].to_numpy() - p0) > 10
        if s["kind"] == "har_mon":
            base &= (df["weekday"].to_numpy() == 0)
        self.base = base
        self.inspan = in_range(d, SPAN)
        back = BACK_LABEL if s["layer"] == "label" else BACK
        self.back = in_range(d, back)
        self.front = in_range(d, FRONT)
        self.back_rng = back

    def run(self) -> "CellRun":
        s, df = self.s, self.df
        if s["x"] not in df.columns:
            self.status, self.reasons = "설명용", [f"특징 열 {s['x']} 없음"]
            return self
        self._masks()
        x0 = df[s["x"]].to_numpy(float)
        F = np.where(self.base & self.inspan & np.isfinite(x0))[0]
        self.N = N = len(F)
        span_days = int((df["judge_ok"].to_numpy(bool) & self.inspan).sum())
        self.m_min = max(1, math.ceil(SHIFT_DAYS * N / max(span_days, 1)))
        self.info.update({"N_rot": N, "span_windows_2026": span_days, "shift_min": self.m_min,
                          "shift_K": max(0, N - 2 * self.m_min + 1)})
        if N < 2 * self.m_min + 1:
            self.status, self.reasons = "설명용", [f"2026 특징 창 {N}개 — 원형 이동 불가"]
            return self
        X = _rotate(x0, F, N)
        comp = None
        if s.get("companion") and s["companion"] in df.columns:
            c0 = df[s["companion"]].to_numpy(float)
            Fc = np.where(self.base & self.inspan & np.isfinite(c0))[0]
            comp = _rotate(c0, Fc, N)
        # M0(주 특징 X⊥ 또는 위약용)
        mk = s["market"]
        cont, dums = (M0_KR_CONT, M0_KR_DUM) if mk == "kr" else (M0_US_CONT, M0_US_DUM)
        self.m0_cols = (cont, dums)
        fit_ok = self.base & np.isfinite(x0)
        if s["kind"] == "har_mon":           # 월요일 칸: M0 적합은 모든 판정 자격 창의 같은 특징이 없어 위약을 생략
            m0 = None
        else:
            m0 = expanding_m0(df, X, cont, dums, s.get("m0_extra", []), fit_ok)
        self.m0 = m0
        if m0 is not None:
            last = [r for r in m0["months"] if "R2_obs" in r]
            self.info["M0"] = {"first_month_fitted": last[0]["month"] if last else None,
                               "R2_last": last[-1]["R2_obs"] if last else None,
                               "R2_mean": float(np.mean([r["R2_obs"] for r in last])) if last else None,
                               "echo_dominant": bool(last and last[-1]["R2_obs"] >= ECHO_R2),
                               "ridge_months": sum(1 for r in last if r.get("ridge")),
                               "months": m0["months"]}
        main = m0["xperp"] if s["kind"] == "ic" else X
        self.X, self.main, self.comp = X, main, comp
        y = df[s["y"]].to_numpy(float)
        if s.get("ytrans") == "log1p":
            y = np.log1p(y)
        if s.get("gap_pred"):
            y = self._gap_resid(y)
        self.y = y
        need = list(s.get("ctrl", [])) if s["kind"] in ("ic", "har_mon") else HAR_CONT + list(s.get("extra", []))
        cond = np.ones(len(df), bool)
        if s.get("cond"):
            cond = df[s["cond"]].to_numpy(float) > 0
        common = self.base & cond & np.isfinite(y) & np.isfinite(df[need].to_numpy(float)).all(1) \
            & np.isfinite(main[:, 0])
        self.common = common
        self.rob_keys = ["crash4", "top2", "opened8"] + [k for k, _ in s.get("rob_extra", [])]
        self.S_back = common & self.back
        self.S_front = common & self.front
        h = s["h"]
        pos = df["pos"].to_numpy()
        nb_b, nb_f = n_blocks(pos[self.S_back], h), n_blocks(pos[self.S_front], h)
        self.info.update({"n_back": int(self.S_back.sum()), "n_front": int(self.S_front.sum()),
                          "blocks_back": nb_b, "blocks_front": nb_f,
                          "back": [str(self.back_rng[0]), str(self.back_rng[1])],
                          "first_D_back": str(df.loc[self.S_back, "Dd"].min()) if self.S_back.any() else None,
                          "last_D_back": str(df.loc[self.S_back, "Dd"].max()) if self.S_back.any() else None})
        front_req = s["layer"] != "label"
        self.front_required = front_req
        # §2.7 최소 표본: 구간마다 h일 넘게 떨어진 독립 묶음 8개
        if nb_b < MIN_BLOCKS:
            self.status = "설명용"
            self.reasons.append(f"뒤 구간 독립 묶음 {nb_b} < {MIN_BLOCKS}(§2.7 최소 표본)")
        if front_req and nb_f < MIN_BLOCKS:
            self.status = "설명용"
            self.reasons.append(f"앞 구간 독립 묶음 {nb_f} < {MIN_BLOCKS}(§2.7 최소 표본)")
        if self.S_back.sum() < 12:
            self.status = "설명용"
            m0i = self.info.get("M0") or {}
            if m0 is not None and not m0i.get("first_month_fitted"):
                self.reasons.append(f"M0 확장 OLS 적합 창이 끝까지 {MIN_M0}개 미만(특징 유효 창 "
                                    f"{int(fit_ok.sum())}개) → X⊥ 없음, 계산 생략")
            else:
                self.reasons.append("뒤 구간 표본 12개 미만 — 계산 생략")
            return self
        self._compute_all()
        return self

    def _gap_resid(self, y: np.ndarray) -> np.ndarray:
        """§3 H52 목표 '월요일 |갭| 잔차': 예측 |갭|을 앞 월요일(D−1까지)만으로 확장 OLS(매 월요일 재적합,
        최소 H52_FIT_MIN개)해 만들고 뺀다. 가격 열만 쓰고 특징(x)과 무관하다."""
        Z = self.df[self.s["gap_pred"]].to_numpy(float)
        ok = self.base & np.isfinite(y) & np.isfinite(Z).all(1)
        idx = np.where(ok)[0]                     # 창은 마감 시각순이라 앞 행 = 앞 월요일
        out = np.full(len(y), np.nan)
        for k, i in enumerate(idx):
            if k < H52_FIT_MIN:
                continue
            prev = idx[:k]
            A = np.column_stack([np.ones(k), Z[prev]])
            b = np.linalg.lstsq(A, y[prev], rcond=None)[0]
            out[i] = y[i] - float(np.r_[1.0, Z[i]] @ b)
        fin = np.isfinite(out)
        self.info["gap_model"] = {"predictors": list(self.s["gap_pred"]), "fit_min": H52_FIT_MIN,
                                  "first_D": str(self.df.loc[fin, "Dd"].min()) if fin.any() else None,
                                  "n_resid": int(fin.sum())}
        return out

    # -------------------------------------------------- 통계(모든 이동 열)
    def _seg(self, S: np.ndarray, M: np.ndarray, want_resid: bool = False) -> dict:
        s, df = self.s, self.df
        idx = np.where(S)[0]
        d = df.iloc[idx]
        y = self.y[idx]
        if s["kind"] in ("har", "har_mon"):
            if s["kind"] == "har":
                C, names, dropped = control_design(d, HAR_CONT + list(s.get("extra", [])), HAR_DUM)
                pos = d["pos"].to_numpy()
            else:
                C, names, dropped = control_design(d, list(s["ctrl"]), [])
                pos = np.arange(len(d))          # 월요일 순번(연속 월요일 사이 HAC 5)
            Q = _qr(C)
            EX = _resid(Q, M[idx])
            ey = _resid(Q, y)
            r = fwl_hac(EX, ey, pos, HAC_L, C.shape[1] + 1)
            out = {"est": r["pr"], "t": r["t"], "beta": r["beta"], "df": r["df"], "n": len(idx),
                   "stat": r["t"], "names": names, "dropped": dropped}
            if want_resid:
                out["ex"], out["ey"] = EX, ey
            out["se_ci"] = np.full(M.shape[1], 1 / math.sqrt(max(r["df"], 1)))
            return out
        # IC: 순위 부분 상관(통제 순위로 잔차화한 Pearson) + 회귀 t(Hodrick 1B 또는 HAC(h))
        ctrl = list(s["ctrl"])
        Cr, names, dropped = control_design(d, ctrl, [])
        RC = np.column_stack([np.ones(len(d))] + [stats.rankdata(d[c].to_numpy(float)) for c in names])
        Q1 = _qr(RC)
        RX = _resid(Q1, rank_cols(M[idx]))
        ry = _resid(Q1, stats.rankdata(y))
        ic = col_corr(RX, ry)
        Z = stats.zscore(rank_cols(M[idx]), axis=0, ddof=1)
        Q2 = _qr(Cr)
        EZ = _resid(Q2, Z)
        ey2 = _resid(Q2, y)
        pos = d["pos"].to_numpy()
        if s.get("se", "hodrick") == "hodrick":
            eps_col = {"fut_kospi": "eps_kospi", "fut_soxx": "eps_soxx", "fut_mem3x": "eps_mem3x"}
            ek = next(v for k, v in eps_col.items() if s["y"].startswith(k))
            eps_all = df[ek].to_numpy(float)
            r = fwl_hodrick(EZ, ey2, pos, eps_all, s["h"], Cr.shape[1] + 1)
        else:
            r = fwl_hac(EZ, ey2, pos, s["h"], Cr.shape[1] + 1)
        rho = ac1_cols(M[idx], pos)
        ne = n_eff_bartlett(len(idx), rho, s["h"])
        out = {"est": ic, "t": r["t"], "beta": r["beta"], "df": r["df"], "n": len(idx), "stat": ic,
               "names": names, "dropped": dropped, "n_eff": ne, "rho1": rho, "se_ci": 1 / np.sqrt(ne)}
        if want_resid:
            out["ex"], out["ey"] = RX, ry
        return out

    def _est_one(self, S: np.ndarray, col: np.ndarray) -> float:
        """col: (n, 1) 한 열."""
        return float(self._seg(S, col)["est"][0])

    def _robust(self, S: np.ndarray, M: np.ndarray, b: dict) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """§2.7 강건 3개(급락 4일·기여 상위 2창·열어 본 8일 빼기) + 칸별 추가(rob_extra, H03 7/29~31).
        b = 같은 표본·열의 _seg(want_resid=True)."""
        df, N, sg = self.df, M.shape[1], self.s["sign"]
        out = {}
        for key, days in [("crash4", CRASH_DAYS), ("opened8", OPENED_DAYS)] + list(self.s.get("rob_extra", [])):
            Sx = S & ~overlap_mask(df, days)
            if M is self.main:
                self.info.setdefault("n_rob", {})[key] = int(Sx.sum())
            out[key] = self._seg(Sx, M)["est"] if Sx.sum() >= 12 else np.full(N, np.nan)
        contrib = sg * b["ex"] * b["ey"][:, None]
        top = np.argsort(contrib, axis=0)[-2:, :]
        idx = np.where(S)[0]
        r2 = np.empty(N)
        for r in range(N):
            m = S.copy()
            m[idx[top[:, r]]] = False
            r2[r] = self._est_one(m, M[:, [r]])
        out["top2"] = r2
        return out, idx[top[:, 0]]

    def _compute_all(self):
        s, df = self.s, self.df
        sg = s["sign"]
        N = self.N
        b = self._seg(self.S_back, self.main, want_resid=True)
        self.info["controls_used"] = b["names"]
        self.info["controls_dropped"] = b["dropped"]
        a = self.arr
        a["est"], a["t"], a["stat"], a["se_ci"] = b["est"], b["t"], b["stat"], b["se_ci"]
        a["p_an"] = stats.t.sf(sg * b["t"], b["df"])
        self.info["df_back"] = int(b["df"])
        if "n_eff" in b:
            a["n_eff"] = b["n_eff"]
        else:
            a["n_eff"] = np.full(N, float(b["df"]))
        # 앞 구간
        if self.S_front.sum() >= 12:
            f = self._seg(self.S_front, self.main)
            a["est_front"], a["t_front"] = f["est"], f["t"]
            a["p_an_front"] = stats.t.sf(sg * f["t"], f["df"])
        else:
            a["est_front"] = a["t_front"] = a["p_an_front"] = np.full(N, np.nan)
        # §2.5 2025 보조(같은 부호 확인만, 판정에 안 씀)
        S25 = self.common & in_range(df["Dd"].to_numpy(), SPAN_2025)
        self.info["n_2025"] = int(S25.sum())
        a["est_2025"] = self._seg(S25, self.main)["est"] if S25.sum() >= 12 else np.full(N, np.nan)
        # 강건 3개(+ 칸별 추가)
        rob, dropped = self._robust(self.S_back, self.main, b)
        for k, v in rob.items():
            a[f"rob_{k}"] = v
        self.info["top2_dropped_obs"] = [str(df.loc[i, "win_id"]) for i in dropped]
        # 짝판(번호판 ↔ 저장판) 부호
        if self.comp is not None:
            Sc = self.S_back & np.isfinite(self.comp[:, 0])
            a["comp_est"] = self._seg(Sc, self.comp)["est"] if Sc.sum() >= 12 else np.full(N, np.nan)
        # 메아리 위약: X̂로 같은 시험. 같은 부분 표본에서 X(원값) 효과와 비교
        if self.m0 is not None:
            xh = self.m0["xhat"]
            Sp = self.S_back & np.isfinite(xh[:, 0]) & np.isfinite(self.X[:, 0])
            self.info["n_placebo"] = int(Sp.sum())
            if Sp.sum() >= 12:
                a["ph_est"] = self._seg(Sp, xh)["est"]
                a["ph_x_est"] = self._seg(Sp, self.X)["est"]
            else:
                a["ph_est"] = a["ph_x_est"] = np.full(N, np.nan)
            if s["kind"] == "har":       # T1: X⊥판(원값이 주 특징이라 따로 잰다). '통과'는 §2.7 그대로(p·앞 부호·강건)
                xp = self.m0["xperp"]
                Sxp = self.S_back & np.isfinite(xp[:, 0])
                self.info["n_xperp"] = int(Sxp.sum())
                nan_n = np.full(N, np.nan)
                if Sxp.sum() >= 12:
                    q = self._seg(Sxp, xp, want_resid=True)
                    a["xp_est"], a["xp_t"], a["xp_stat"] = q["est"], q["t"], q["stat"]
                    a["xp_p_an"] = stats.t.sf(sg * q["t"], q["df"])
                    Sxf = self.S_front & np.isfinite(xp[:, 0])
                    a["xp_est_front"] = self._seg(Sxf, xp)["est"] if Sxf.sum() >= 12 else nan_n
                    xr, _ = self._robust(Sxp, xp, q)
                    for k, v in xr.items():
                        a[f"xp_rob_{k}"] = v
                else:
                    a["xp_est"] = a["xp_t"] = a["xp_stat"] = a["xp_p_an"] = a["xp_est_front"] = nan_n
                    for k in self.rob_keys:
                        a[f"xp_rob_{k}"] = nan_n
        # 보고용 보조 특징(판정에 안 씀): H31 NVDL 뺀 판
        for col in s.get("aux", []):
            if col not in df.columns:
                continue
            xa = df[col].to_numpy(float)[:, None]
            if s["kind"] == "ic":
                cont, dums = self.m0_cols
                xa = expanding_m0(df, xa, cont, dums, s.get("m0_extra", []), self.base & np.isfinite(xa[:, 0]))["xperp"]
            Sa = self.S_back & np.isfinite(xa[:, 0])
            self.info.setdefault("aux", {})[col] = {"n_back": int(Sa.sum()),
                                                    "est": float(self._seg(Sa, xa)["est"][0]) if Sa.sum() >= 12 else None}
        # 원형 이동 p(①)
        a["p_shift"] = shift_p_all(a["stat"], sg, self.m_min)
        if "xp_stat" in a:
            a["xp_p_shift"] = shift_p_all(a["xp_stat"], sg, self.m_min)
        if s["kind"] != "ic":       # H03: 부분 R² 증가 < 0.01이면 '실용 효과 작음', H08: < 0.005 기록(§3)
            self.info["dR2_back"] = self._delta_r2()

    def _delta_r2(self):
        """관측 열의 부분 R² 증가(설명 기록용)."""
        idx = np.where(self.S_back)[0]
        d = self.df.iloc[idx]
        s = self.s
        if s["kind"] == "har":
            C, _, _ = control_design(d, HAR_CONT + list(s.get("extra", [])), HAR_DUM)
        else:
            C, _, _ = control_design(d, list(s["ctrl"]), [])
        y = self.y[idx]
        X1 = np.column_stack([C, self.main[idx, 0]])

        def r2(A):
            e = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
            dd = y - y.mean()
            return 1 - (e @ e) / (dd @ dd)
        return float(r2(X1) - r2(C))


def shift_p_all(stat: np.ndarray, sign: int, m_min: int) -> np.ndarray:
    """모든 r에 대해: 이동열 r의 통계를 그 열을 다시 m_min..N−m_min칸 돌린 통계들과 비교한 단측 p."""
    N = len(stat)
    J = np.arange(m_min, N - m_min + 1)
    if len(J) == 0:
        return np.full(N, np.nan)
    idx = (np.arange(N)[:, None] + J[None, :]) % N
    null = sign * stat[idx]
    obs = sign * stat[:, None]
    return (1 + (null >= obs).sum(1)) / (1 + len(J))


def wy_p_all(rep: CellRun, members: list[CellRun]) -> np.ndarray:
    """Westfall-Young: 족 안 특징을 같은 이동 몫(u = 이동칸/N)으로 함께 돌린 최대 통계(각 칸 이동 분포로 표준화,
    예상 부호 방향)와 대표의 통계를 비교."""
    Nr = rep.N
    J = np.arange(rep.m_min, Nr - rep.m_min + 1)
    if len(J) == 0:
        return np.full(Nr, np.nan)

    def z(c: CellRun) -> np.ndarray:
        st = c.s["sign"] * c.arr["stat"]
        return (st - st.mean()) / st.std(ddof=1)
    zr = z(rep)
    u = ((np.arange(Nr)[:, None] + J[None, :]) % Nr) / Nr
    mx = np.full(u.shape, -np.inf)
    for c in members:
        zc = z(c)
        ic = np.round(u * c.N).astype(int) % c.N
        mx = np.maximum(mx, zc[ic])
    return (1 + (mx >= zr[:, None]).sum(1)) / (1 + len(J))


# ================================================================ 6. 판정(관측 r=0과 전체 이동 공용)
def _at(a: np.ndarray, r: int) -> float:
    return float(a[r]) if a is not None else float("nan")


def cell_values(c: CellRun, r: int, p_alt: np.ndarray | None) -> dict:
    a, sg = c.arr, c.s["sign"]
    p_sh = _at(p_alt if p_alt is not None else a["p_shift"], r)
    p_an = _at(a["p_an"], r)
    est = _at(a["est"], r)
    v = {"est": est, "t": _at(a["t"], r), "p_an": p_an, "p_shift": p_sh,
         "p_final": max(p_an, p_sh) if np.isfinite(p_sh) else p_an,
         "se_ci": _at(a["se_ci"], r), "est_front": _at(a["est_front"], r), "p_an_front": _at(a["p_an_front"], r)}
    keys = getattr(c, "rob_keys", ["crash4", "top2", "opened8"])
    rob = {k: _at(a[f"rob_{k}"], r) for k in keys}
    v["robust"] = {k: (x / est if est else float("nan")) for k, x in rob.items()}
    v["robust_ok"] = bool(all(np.isfinite(x) and x >= ROBUST_RATIO for x in v["robust"].values()))
    e25 = _at(a.get("est_2025"), r)
    v["est_2025"] = e25
    v["opposite_2025"] = bool(np.isfinite(e25) and sg * e25 < 0)
    if "ph_est" in a:
        ph, phx = _at(a["ph_est"], r), _at(a["ph_x_est"], r)
        v["placebo_est"], v["placebo_x_est"] = ph, phx
        v["placebo_flag"] = bool(np.isfinite(ph) and np.isfinite(phx) and phx != 0 and np.sign(ph) == np.sign(phx)
                                 and abs(ph) >= PLACEBO_RATIO * abs(phx))
    else:
        v["placebo_flag"] = False
    if "xp_est" in a:
        xe = _at(a["xp_est"], r)
        v["xp_est"], v["xp_p_final"] = xe, max(_at(a["xp_p_an"], r), _at(a["xp_p_shift"], r))
        xf = _at(a.get("xp_est_front"), r)
        v["xp_front_ok"] = bool(np.isfinite(xf) and sg * xf > 0)
        xr = {k: (_at(a.get(f"xp_rob_{k}"), r) / xe if xe else float("nan")) for k in keys}
        v["xp_robust"] = xr
        v["xp_robust_ok"] = bool(all(np.isfinite(x) and x >= ROBUST_RATIO for x in xr.values()))
    if "comp_est" in a:
        v["comp_est"] = _at(a["comp_est"], r)
        v["comp_ok"] = bool(np.isfinite(v["comp_est"]) and np.sign(v["comp_est"]) == sg)
    else:
        v["comp_ok"] = True
    v["front_ok"] = bool(np.isfinite(v["est_front"]) and sg * v["est_front"] > 0)
    ne = _at(a["n_eff"], r)
    v["n_eff"] = ne
    if np.isfinite(v["p_an_front"]) and np.isfinite(p_an):
        zf = stats.norm.isf(min(max(v["p_an_front"], 1e-12), 1 - 1e-12))
        zb = stats.norm.isf(min(max(p_an, 1e-12), 1 - 1e-12))
        v["stouffer_z"] = (zf + zb) / math.sqrt(2)
    return v


def verdict(c: CellRun, v: dict, pass_p: bool, thr_local: float, expected: float) -> tuple[str, str]:
    sg = c.s["sign"]
    signed = sg * v["est"]
    if not np.isfinite(signed):
        return "판정 불가", "점추정 없음"
    if signed < 0:
        return "기각", "뒤 구간 점추정 부호 반대"
    if signed + 1.645 * v["se_ci"] < expected / 2:
        return "기각", f"90% 상한 {signed + 1.645 * v['se_ci']:.3f} < 기대 효과 절반 {expected / 2:.3f}"
    if v["placebo_flag"]:
        if c.s["kind"] == "ic":
            xp_pass = pass_p and (v["front_ok"] or not c.front_required) and v["robust_ok"]
        else:                    # T1: X⊥판이 §2.7 '통과'(국소 문턱 p·예상 부호·앞 구간 같은 부호·강건)를 채우는가
            xp_pass = bool(np.isfinite(v.get("xp_p_final", np.nan)) and v["xp_p_final"] <= thr_local
                           and sg * v.get("xp_est", np.nan) > 0
                           and (v.get("xp_front_ok", False) or not c.front_required)
                           and v.get("xp_robust_ok", False))
        if not xp_pass:
            return "기각", "메아리 위약(X̂가 같은 부호로 X 효과의 70% 이상, X⊥ 미통과)"
    front_ok = v["front_ok"] or not c.front_required
    if pass_p and front_ok and v["comp_ok"]:
        return ("통과", "") if v["robust_ok"] else ("취약", "강건 조건 중 50% 미만")
    why = []
    if not pass_p:
        why.append("보정 p 미달")
    if not front_ok:
        why.append("앞 구간 부호 다름/없음")
    if not v["comp_ok"]:
        why.append("번호판·저장판 부호 다름")
    return "판정 불가", ", ".join(why) + " → 전방 트랙(§2.7·§5.5, '효과 없음' 아님)"


def decide(runs: dict[str, CellRun], rmap: dict[str, int], p_wy: dict[str, np.ndarray],
           h31_active: bool) -> dict[str, dict]:
    out: dict[str, dict] = {}
    tested = {cid: c for cid, c in runs.items() if c.s["role"] == "main" and c.status == "시험"
              and not (c.s.get("needs_reg_date") and not h31_active)}
    vals = {cid: cell_values(c, rmap.get(cid, 0), p_wy.get(cid)) for cid, c in runs.items()
            if c.status == "시험" or c.arr}
    fam_ps = {"T1": {k: vals[k]["p_final"] for k, c in tested.items() if c.s["fam"] == "T1"},
              "T2": {k: vals[k]["p_final"] for k, c in tested.items() if c.s["fam"] == "T2"}}
    adj1, thr1 = holm(fam_ps["T1"], ALPHA_T1) if fam_ps["T1"] else ({}, {})
    adj2, thr2 = bh(fam_ps["T2"], Q_T2) if fam_ps["T2"] else ({}, {})
    m1, m2 = len(fam_ps["T1"]), len(fam_ps["T2"])
    for cid, c in runs.items():
        v = vals.get(cid)
        rec = {"role": c.s["role"], "status": c.status, "reasons": list(c.reasons), "values": v}
        if cid in tested:
            fam = c.s["fam"]
            adj = adj1 if fam == "T1" else adj2
            thr = thr1 if fam == "T1" else thr2
            alpha = ALPHA_T1 if fam == "T1" else Q_T2
            p_adj = adj[cid]
            pass_p = p_adj <= alpha
            # 국소 문턱(위약의 X⊥판 비교용): 통과한 칸은 그 문턱, 아니면 Holm/BH 순위 문턱
            ps = fam_ps[fam]
            rank = sorted(ps, key=ps.get).index(cid)
            m = m1 if fam == "T1" else m2
            loc = thr.get(cid, (alpha / (m - rank)) if fam == "T1" else ((rank + 1) * alpha / m))
            exp_ = EXPECTED["T1"] if fam == "T1" else EXPECTED["IC"]
            vd, why = verdict(c, v, pass_p, loc, exp_)
            rec.update({"p_adj": p_adj, "alpha": alpha, "m_family": m, "thr_local": loc, "verdict": vd, "why": why})
        elif c.s["role"] == "main" and c.s.get("needs_reg_date") and not h31_active:
            rec.update({"verdict": "설명용", "why": "8월 단일종목 레버리지 규제 시행일 미확보(§5.1 #11)"})
        elif c.s["role"] in ("main", "expl"):
            rec.update({"verdict": "설명용", "why": "; ".join(c.reasons) or c.s.get("why", "")})
        out[cid] = rec
    # 족 안 순차(§2.6): 대표가 통과한 족만, 대표의 국소 문턱으로, 고정 순서
    reps = {c.s["family"]: cid for cid, c in runs.items() if c.s["role"] == "main"}
    seqs: dict[str, list] = {}
    for cid, c in runs.items():
        if c.s["role"] in ("seq", "stopped"):
            seqs.setdefault(c.s["family"], []).append(c)
    for fam, lst in seqs.items():
        lst.sort(key=lambda c: c.s["order"])
        rep = reps.get(fam)
        rv = out.get(rep, {})
        opened = rv.get("verdict") == "통과"
        thr = rv.get("thr_local") if opened else None
        closed_by = None if opened else f"대표 {rep} 미통과"
        for c in lst:
            cid = c.s["id"]
            rec = out.setdefault(cid, {"role": c.s["role"], "status": c.status, "reasons": list(c.reasons),
                                       "values": vals.get(cid)})
            if c.s["role"] == "stopped":
                rec.update({"verdict": "멈춤", "why": c.s["why"] + " — 순차에서 건너뜀(실패로 세지 않음)"})
                continue
            if c.status != "시험":
                # 해석: 가격 전에 정해진 설명용(점검표·최소 표본)은 멈춤 칸처럼 순차에서 건너뛴다(실패로 세지 않음).
                # 결과값을 보지 않고 정해진 제외라 고정 순서 검정의 오류율을 바꾸지 않는다.
                rec.update({"verdict": "설명용", "why": "; ".join(c.reasons) + " — 순차에서 건너뜀"
                            + (f" (순차 닫힘: {closed_by})" if closed_by else "")})
                continue
            if closed_by:
                rec.update({"verdict": "설명용", "why": f"순차 닫힘: {closed_by}"})
                continue
            v = vals[cid]
            pass_p = v["p_final"] <= thr
            vd, why = verdict(c, v, pass_p, thr, EXPECTED["SEQ"])
            rec.update({"verdict": vd, "why": why, "thr_local": thr})
            if vd != "통과":
                closed_by = f"{cid} {vd}"
    return out


# ================================================================ 7. 실행
def checklist(feat: dict[str, pd.DataFrame], pred_info: dict, opts: dict) -> dict:
    """§5.1 가격 전 점검(포럼 특징만): 자동 설명용 근거."""
    kr = feat["kr"]
    d = kr["Dd"].to_numpy()
    j = kr["judge_ok"].to_numpy(bool)
    span = in_range(d, SPAN)
    out = {}
    for c in CELLS:
        if c["role"] not in ("main", "seq", "expl") or c["x"] not in feat[c["market"]].columns:
            continue
        df = feat[c["market"]]
        dd = df["Dd"].to_numpy()
        rng = (FRONT[0], BACK_LABEL[1]) if c["layer"] == "forum" else BACK_LABEL
        m = df["judge_ok"].to_numpy(bool) & in_range(dd, rng)
        if c.get("mask"):
            m &= df[c["mask"]].astype(bool).to_numpy()
        if c["kind"] == "har_mon":
            m &= df["weekday"].to_numpy() == 0
        miss = float(1 - np.isfinite(df.loc[m, c["x"]].to_numpy(float)).mean()) if m.any() else 1.0
        out[c["id"]] = {"missing_rate": miss, "range": [str(rng[0]), str(rng[1])], "n_windows": int(m.sum()),
                        "auto_explanatory": miss > MISSING_MAX}
    uu = kr.loc[j & span, ["U_rz8", "N_rz8"]].dropna()
    cu = float(np.corrcoef(uu["U_rz8"], uu["N_rz8"])[0, 1]) if len(uu) > 3 else float("nan")
    out["H04"]["corr_U_N_rz8_2026"] = cu
    out["H04"]["auto_explanatory"] = out["H04"]["auto_explanatory"] or (cu > H04_CORR_MAX)
    dsd = kr.loc[j & span, "D_SD"]
    out["H09"] = {"auto_explanatory": True, "D_SD_missing_2026": float(dsd.isna().mean()),
                  "why": f"D_SD 2026 결측 {dsd.isna().mean():.1%} > 30%, 그리고 강세·약세 점수 사전(BULL·BEAR) 측정 실패"}
    out["H24"] = {"auto_explanatory": True, "why": "공포 사전 F 정밀도 35% < 60%(부록 2 #4) → 감정 사전 칸 멈춤"}
    nb = [len(pd.bdate_range(*r)) // 22 for r in (FRONT, BACK)]
    out["H39+H48"] = {"auto_explanatory": True, "blocks_21d": nb,
                      "why": f"21일 목표의 겹치지 않는 묶음 {nb} < 8(부록 2 #2), KIND 유니버스 미구성"}
    out["H31"]["reg_date"] = str(opts.get("h31_reg_date")) if opts.get("h31_reg_date") else None
    out["H31"]["auto_explanatory"] = out["H31"]["auto_explanatory"] or not opts.get("h31_reg_date")
    out["_pred"] = pred_info
    return out


def merge_features(feat: dict, pred: pd.DataFrame, emo: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """포럼 특징 + 예측 층 + 감정 층(가격 없음)."""
    out = {}
    for mk in ("kr", "us"):
        df = feat[mk]
        if mk == "kr":
            pc = [c for c in ("P_bar", "P_share", "P_share_rz8", "P_sd", "P_sd_rz8", "P_strong1", "n_pred")
                  if c in pred.columns]
            df = df.merge(pred[["win_id"] + pc], on="win_id", how="left", validate="1:1")
            if len(emo):
                df = df.merge(emo, on="win_id", how="left", validate="1:1")
        assert (df["win_id"].to_numpy() == feat[mk]["win_id"].to_numpy()).all()
        out[mk] = df
    return out


def build_frames(feat: dict, panel: dict) -> dict[str, pd.DataFrame]:
    """특징 표(merge_features) + 가격 패널(load_prices). win_id로만 잇는다."""
    out = {}
    for mk in ("kr", "us"):
        df = feat[mk]
        pn = panel[mk]
        newc = [c for c in pn.columns if c.startswith(("m0_", "fut_", "eps_", "wd_"))]
        df = df.merge(pn[["win_id"] + newc], on="win_id", how="left", validate="1:1")
        assert (df["win_id"].to_numpy() == feat[mk]["win_id"].to_numpy()).all()
        out[mk] = df
    return out


def cell_notes(c: CellRun, v: dict | None) -> list[str]:
    """판정에 쓰지 않는 '기록' 항목(동결 문서가 적으라고 한 것)."""
    out = []
    if not c.arr:
        return out
    cid = c.s["id"]
    if cid in DR2_SMALL and c.info.get("dR2_back") is not None and c.info["dR2_back"] < DR2_SMALL[cid][0]:
        out.append(f"{DR2_SMALL[cid][1]}: {c.info['dR2_back']:.4f}")
    m0 = c.info.get("M0") or {}
    if m0.get("echo_dominant"):
        out.append(f"메아리 우세(M0 R² {m0['R2_last']:.2f} ≥ {ECHO_R2}, §2.3)")
    if v and v.get("opposite_2025"):
        out.append(f"2025 보조 부호 반대(추정 {v['est_2025']:+.3f}) — 장세 전환 위험(§2.5)")
    if c.s.get("layer") == "label":
        out.append("라벨 층: M0 계수·rz8 기준에 7월 전(AI 지식 기간) 라벨을 예열로 씀(prechecks_full 조건)")
    for col, a in (c.info.get("aux") or {}).items():
        out.append(f"보조 {col}: 추정 {a['est']:+.3f} (n {a['n_back']})" if a.get("est") is not None
                   else f"보조 {col}: 표본 부족")
    return out


def run_all(frames: dict[str, pd.DataFrame], opts: dict, chk: dict, n_global: int = N_GLOBAL) -> dict:
    t0 = time.time()
    runs: dict[str, CellRun] = {}
    for spec in CELLS:
        if spec["role"] == "stopped":
            cr = CellRun(spec, frames["kr"], opts)
            cr.status = "멈춤"
            runs[spec["id"]] = cr
            continue
        cr = CellRun(spec, frames[spec["market"]], opts).run()
        ck = chk.get(spec["id"], {})
        if ck.get("auto_explanatory") and cr.status in ("시험", "설명용"):
            cr.status = "설명용"
            cr.reasons.insert(0,f"가격 전 점검 자동 설명용(결측 {ck.get('missing_rate', float('nan')):.1%}"
                              + (f", corr(U,N) {ck['corr_U_N_rz8_2026']:.2f}" if "corr_U_N_rz8_2026" in ck else "")
                              + (", 규제 시행일 없음" if spec["id"] == "H31" and not opts.get("h31_reg_date") else "")
                              + ")")
        runs[spec["id"]] = cr
    h31_active = bool(opts.get("h31_reg_date"))
    # 대표의 WY p: 족 안 '시험' 상태 특징(대표 + 순차)
    p_wy: dict[str, np.ndarray] = {}
    fam_members: dict[str, list[str]] = {}
    for cid, c in runs.items():
        if c.s["role"] == "main" and c.arr:
            mem = [runs[k] for k, cc in runs.items() if cc.s.get("family") == c.s["family"] and cc.arr
                   and cc.s["role"] in ("main", "seq") and (cc.status == "시험" or k == cid)]
            fam_members[cid] = [m.s["id"] for m in mem]
            p_wy[cid] = wy_p_all(c, mem)
    obs = decide(runs, {k: 0 for k in runs}, p_wy, h31_active)
    # 전체 원형 이동 200회(§2.6 전체 보고): 모든 칸을 같은 이동 몫 u로
    span_ref = int((frames["kr"]["judge_ok"].to_numpy(bool) & in_range(frames["kr"]["Dd"].to_numpy(), SPAN)).sum())
    rng = np.random.default_rng(GLOBAL_SEED)
    us_ = rng.uniform(SHIFT_DAYS / span_ref, 1 - SHIFT_DAYS / span_ref, n_global)
    main_ids = [k for k, c in runs.items() if c.s["role"] == "main"]
    counts, per_cell, seq_pass = [], {k: 0 for k in runs}, 0
    for u in us_:
        rmap = {k: (int(round(u * c.N)) % c.N if c.N else 0) for k, c in runs.items()}
        dd = decide(runs, rmap, p_wy, h31_active)
        n_pass = 0
        for k, rec in dd.items():
            if rec.get("verdict") == "통과":
                per_cell[k] += 1
                if k in main_ids:
                    n_pass += 1
                elif runs[k].s["role"] == "seq":
                    seq_pass += 1
        counts.append(n_pass)
    counts = np.array(counts, int)
    glob_ = {"n": int(n_global), "seed": GLOBAL_SEED, "u_range": [SHIFT_DAYS / span_ref, 1 - SHIFT_DAYS / span_ref],
             "n_main_tested": None, "seq_passes_total": int(seq_pass)}
    if n_global:
        glob_.update({"mean_main_passes": float(counts.mean()), "p_ge1": float((counts >= 1).mean()),
                      "max": int(counts.max()), "hist": {int(k): int((counts == k).sum()) for k in np.unique(counts)},
                      "per_cell_pass_rate": {k: v / n_global for k, v in per_cell.items() if v}})
    tested = [k for k, rec in obs.items() if "p_adj" in rec]
    cells_out = {}
    for cid, c in runs.items():
        rec = obs.get(cid, {})
        cells_out[cid] = {"label": c.s.get("label"), "role": c.s["role"], "fam": c.s.get("fam"),
                          "family": c.s.get("family"), "layer": c.s.get("layer"), "x": c.s.get("x"),
                          "y": c.s.get("y"), "h": c.s.get("h"), "sign": c.s.get("sign"),
                          "status": c.status, "info": c.info, "wy_members": fam_members.get(cid),
                          "notes": cell_notes(c, rec.get("values")),
                          **{k: v for k, v in rec.items() if k not in ("role", "status")}}
    mde = []
    for cid, c in runs.items():
        if not c.arr:
            continue
        ne = float(c.arr["n_eff"][0])
        exp_ = EXPECTED["T1"] if c.s.get("fam") == "T1" else EXPECTED["IC"]
        if c.s["role"] == "seq":
            exp_ = EXPECTED["SEQ"]
        mde.append({"id": cid, "n_back": c.info.get("n_back"), "n_eff": ne, "mde": MDE_K / math.sqrt(ne),
                    "expected": exp_, "detectable": MDE_K / math.sqrt(ne) <= exp_})
    return {"cells": cells_out, "tested": {"T1": [k for k in tested if runs[k].s["fam"] == "T1"],
                                           "T2": [k for k in tested if runs[k].s["fam"] == "T2"]},
            "global_rotation": glob_, "mde": mde, "seconds": round(time.time() - t0, 1)}


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (date, datetime)):
        return str(o)
    return o


def prepare_inputs(opts: dict) -> dict:
    windows = load_windows()
    feat, finfo = load_features(windows)
    pred, pinfo = load_pred_features(windows["kr"])
    emo, einfo = load_emotion_features(windows["kr"])
    ff = merge_features(feat, pred, emo)
    chk = checklist(ff, pinfo, opts)
    return {"windows": windows, "feat": ff, "chk": chk,
            "info": {"features": finfo, "pred": pinfo, "emotion": einfo}}


def run_once(inp: dict, mode: str, scenario: str, seed: int, opts: dict, n_global: int,
             plant_ic: float = PLANT_IC, **kw) -> dict:
    plant = None
    if mode == "fake" and scenario == "plant":
        kr = inp["feat"]["kr"]
        plant = {"name": "A_rz8(H03)", "x": kr.set_index("win_id")["A_rz8"], "windows": inp["windows"]["kr"],
                 "ic": plant_ic}
    pr = load_prices(mode, windows=inp["windows"], seed=seed, scenario=scenario, plant=plant, **kw)
    frames = build_frames(inp["feat"], pr)
    res = run_all(frames, opts, inp["chk"], n_global=n_global)
    res["prices_meta"] = pr["meta"]
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dryrun", "fake", "real"])
    ap.add_argument("--scenario", default="walk", choices=["walk", "plant"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--n-global", type=int, default=N_GLOBAL)
    ap.add_argument("--h31-reg-date", default=None)
    ap.add_argument("--confirm-real", action="store_true")
    a = ap.parse_args(argv)
    opts = {"h31_reg_date": date.fromisoformat(a.h31_reg_date) if a.h31_reg_date else None}
    if a.cmd == "real":
        _assert_real_allowed(None, a.confirm_real)       # 입력을 읽기 전에도 한 번 더
        inp = prepare_inputs(opts)
        res = run_once(inp, "real", "", 0, opts, a.n_global, confirm_real=True)
        res["inputs"] = inp["info"]
        res["checklist"] = inp["chk"]
        (RES / "judge_real.json").write_text(json.dumps(clean(res), ensure_ascii=False, indent=1), "utf-8")
        return 0
    inp = prepare_inputs(opts)
    if a.cmd == "fake":
        res = run_once(inp, "fake", a.scenario, a.seed, opts, a.n_global)
        res["inputs"] = inp["info"]
        res["checklist"] = inp["chk"]
        p = RES / f"judge_fake_{a.scenario}_s{a.seed}.json"
        p.write_text(json.dumps(clean(res), ensure_ascii=False, indent=1), "utf-8")
        for k, c in res["cells"].items():
            print(k, c.get("status"), c.get("verdict"), c.get("why"))
        print(res["global_rotation"])
        return 0
    return dryrun(inp, opts, a.seeds, a.n_global)


# ================================================================ 8. 시운전 보고(가짜 주가만)
def _f(x, nd=3, pct=False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, (bool, np.bool_)):
        return "예" if x else "아니오"
    if pct:
        return f"{100 * x:.1f}%"
    return f"{x:+.{nd}f}" if isinstance(x, float) and nd and abs(x) < 10 else (f"{x:.{nd}f}" if isinstance(x, float)
                                                                             else str(x))


def _p(x) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.4f}"


def _seed_summary(runs: list[dict]) -> dict:
    ids = sorted({k for r in runs for k, c in r["cells"].items() if c.get("p_adj") is not None})
    out = {"n_seeds": len(runs), "tested_sets": sorted({",".join(r["tested"]["T1"] + r["tested"]["T2"]) for r in runs}),
           "main_passes": [sum(1 for k in r["tested"]["T1"] + r["tested"]["T2"]
                               if r["cells"][k].get("verdict") == "통과") for r in runs], "cells": {}}
    for k in ids:
        vs = [r["cells"][k] for r in runs if r["cells"][k].get("p_adj") is not None]
        alpha = vs[0]["alpha"]
        out["cells"][k] = {"n": len(vs),
                           "pass": sum(v.get("verdict") == "통과" for v in vs),
                           "fragile": sum(v.get("verdict") == "취약" for v in vs),
                           "reject": sum(v.get("verdict") == "기각" for v in vs),
                           "undecided": sum(v.get("verdict") == "판정 불가" for v in vs),
                           "raw_p_le_alpha": sum((v["values"]["p_final"] or 1) <= alpha for v in vs),
                           "adj_p_le_alpha": sum(v["p_adj"] <= alpha for v in vs),
                           "alpha": alpha}
    mp = np.array(out["main_passes"])
    out["mean_main_passes"] = float(mp.mean())
    out["share_ge1"] = float((mp >= 1).mean())
    return out


def dryrun(inp: dict, opts: dict, n_seeds: int, n_global: int) -> int:
    t0 = time.time()
    res = {}
    res["walk"] = run_once(inp, "fake", "walk", 0, opts, n_global)
    res["plant"] = run_once(inp, "fake", "plant", 0, opts, n_global)
    seeds = {"walk": [res["walk"]], "plant": [res["plant"]], "plant15": [], "plant50": []}
    for s in range(1, n_seeds):
        seeds["walk"].append(run_once(inp, "fake", "walk", s, opts, 0))
        seeds["plant"].append(run_once(inp, "fake", "plant", s, opts, 0))
    for s in range(n_seeds):
        seeds["plant15"].append(run_once(inp, "fake", "plant", s, opts, 0, plant_ic=0.15))
        seeds["plant50"].append(run_once(inp, "fake", "plant", s, opts, 0, plant_ic=0.50))
    summ = {k: _seed_summary(v) for k, v in seeds.items()}
    summ["plant_meta"] = {k: [r["prices_meta"].get("plant") for r in v] for k, v in seeds.items() if k != "walk"}
    summ["h03_by_seed"] = {k: [{"seed": i, "verdict": r["cells"]["H03"].get("verdict"), "why": r["cells"]["H03"].get("why"),
                                "est": r["cells"]["H03"]["values"]["est"], "p_final": r["cells"]["H03"]["values"]["p_final"],
                                "p_adj": r["cells"]["H03"].get("p_adj"),
                                "ic_realized": (r["prices_meta"].get("plant") or {}).get("ic_realized_2026")}
                               for i, r in enumerate(v)] for k, v in seeds.items()}
    out = {"made": datetime.now().replace(microsecond=0).isoformat(), "seconds": round(time.time() - t0, 1),
           "real_price_calls": 0, "runs": res, "seeds": summ, "inputs": inp["info"], "checklist": inp["chk"],
           "code_sha256": sha(Path(__file__)),
           "frozen_sha256": {p.name: sha(p) for p in sorted(ANT.glob("HYPOTHESES*.md"))}}
    (RES / "dryrun.json").write_text(json.dumps(clean(out), ensure_ascii=False, indent=1), "utf-8")
    (RES / "dryrun.md").write_text(dryrun_md(clean(out)), "utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("mean_main_passes", "share_ge1")}
                      for k, v in summ.items() if k != "plant_meta"}, ensure_ascii=False))
    return 0


def _cell_row(cid: str, c: dict) -> str:
    v = c.get("values") or {}
    rob = v.get("robust") or {}
    robs = "/".join(_f(rob.get(k), 2) for k in ("crash4", "top2", "opened8")) if rob else "—"
    if rob.get("jul3") is not None:
        robs += f" (7/29~31 {_f(rob.get('jul3'), 2)})"
    ph = (f"{_f(v.get('placebo_est'), 2)} vs {_f(v.get('placebo_x_est'), 2)}"
          + (" 표지" if v.get("placebo_flag") else "")) if v.get("placebo_est") is not None else "—"
    stat = "부분 r" if c.get("fam") == "T1" else "IC"
    return (f"| {cid} | {c['info'].get('n_back', '—')} | {stat} {_f(v.get('est'))} | {_f(v.get('t'), 2)} | "
            f"{_p(v.get('p_an'))} | {_p(v.get('p_shift'))} ({c['info'].get('shift_K', '—')}) | {_p(v.get('p_final'))} | "
            f"{_p(c.get('p_adj'))} | {_f(v.get('est_front'), 2)} | {_f(v.get('est_2025'), 2)} | {robs} | {ph} | "
            f"**{c.get('verdict')}** {('— ' + c['why']) if c.get('why') else ''} |")


def dryrun_md(o: dict) -> str:
    W, P = o["runs"]["walk"], o["runs"]["plant"]
    sw, sp, s15, s50 = (o["seeds"][k] for k in ("walk", "plant", "plant15", "plant50"))
    gw, gp = W["global_rotation"], P["global_rotation"]
    pm = P["prices_meta"]["plant"]
    t1, t2 = W["tested"]["T1"], W["tested"]["T2"]
    h03p = sp["cells"].get("H03", {})
    h03q = s15["cells"].get("H03", {})
    h03r = s50["cells"].get("H03", {})
    ics = [m["ic_realized_2026"] for m in o["seeds"]["plant_meta"]["plant"] if m]
    p0c = P["cells"]["H03"]
    L = []
    L.append("# 개미 지수 판정 하네스 시운전 (가짜 주가)\n")
    L.append("> 지위: 시운전이다. 판정이 아니다. 실제 가격(앞날·지난 값 모두)은 한 번도 읽지 않았다 — `load_prices(mode='real')`은 "
             "호출하지 않았고, 코드에 가드가 있다(KST 2026-10-24 06:00 전이거나 `confirm_real`이 없으면 파일을 읽기 전에 멈춤). "
             "포럼 특징은 실제 스냅샷(2025-08-20~2026-10-21 파일)이고, 가격만 난수로 만들었다.\n")
    L.append("## 1. 결론\n")
    L.append(f"1. 무작위 걸음 가짜 주가에서 통과는 **0칸**이다(시험 {len(t1) + len(t2)}칸). 원형 이동 200회로 잰 우연 기대는 "
             f"{gw['mean_main_passes']:.3f}칸, 1칸 이상 나올 확률은 {100 * gw['p_ge1']:.1f}%다. 가짜 가격 씨앗 {sw['n_seeds']}개에서 "
             f"통과 칸 평균은 {sw['mean_main_passes']:.2f}칸, 1칸 이상인 씨앗은 {100 * sw['share_ge1']:.0f}%다. 우연 수준이다.")
    L.append(f"2. 효과를 심은 가짜 주가(H03 특징 A_rz8 → 진입일 KOSPI HL, 모집단 스피어만 IC 0.30)의 씨앗 0에서 H03은 "
             f"**{p0c.get('verdict')}**이다(부분 r {p0c['values']['est']:.2f}, 최종 p {p0c['values']['p_final']:.4f}, "
             f"Holm {p0c['p_adj']:.3f}, 실현 IC {pm['ic_realized_2026']:.2f}). 씨앗 {sp['n_seeds']}개로 보면 심은 칸 통과는 "
             f"IC 0.15에서 {h03q.get('pass', 0)}/{s15['n_seeds']}, 0.30에서 {h03p.get('pass', 0)}/{sp['n_seeds']}, "
             f"0.50에서 **{h03r.get('pass', 0)}/{s50['n_seeds']}**다. 효과가 커질수록 통과가 늘고, 심지 않은 칸의 통과는 "
             "(a)와 같은 수준이다. 하네스는 효과를 잡아내고, 효과가 없는 곳에서는 우연 이상으로 통과시키지 않는다.")
    L.append("   - IC 0.30에서 검정력이 절반 안팎인 까닭은 동결 설계 때문이다.")
    L.append("     - 뒤 구간 약 95창이면 T1 검출 한계(MDE)가 부분 r 약 0.28이다(7절).")
    L.append("     - Holm 3칸이라 국소 문턱이 0.0167이다. 원형 이동 152개 중 2위 안에 들어야 한다.")
    L.append("     - A_rz8은 지속성이 있다. 21거래일 넘게 떨어진 시차에서도 2026 자기상관이 최대 0.32다. 그래서 효과가 있으면 돌린 "
             "특징도 효과 일부를 잡고, 이동 귀무가 넓어진다. H03 이동 t의 SD는 씨앗 6개 평균으로 무작위 걸음 1.08, IC 0.30에서 "
             "1.25, IC 0.50에서 1.42다. 이 p는 효과가 있을 때 보수적이고, 효과가 없을 때는 거의 표준(1.08)이다.")
    L.append("3. 동결 규칙을 그대로 적용하면 실제 판정에서 시험되는 주 칸은 "
             f"**T1 {len(t1)}칸({'·'.join(t1)}), T2 {len(t2)}칸({'·'.join(t2)})**뿐이다. 나머지는 가격 전 점검·최소 표본 규칙으로 "
             "설명용이다(2절). 특히 H46·H31·H35는 포럼 자료가 2025-08-20부터라 M0 최소 120창을 늦게 채워, 앞 구간(1~5월)에서 "
             "독립 묶음 8개를 못 만든다.")
    L.append("4. 실제 판정은 10/23 종가가 들어온 뒤(KST 10/24 06:00 이후) `py judge.py real --confirm-real`로 한 번 돌린다. "
             "그 전에 10/22·10/23 창이 든 스냅샷과 가격 파일(9절)이 필요하다.\n")
    # 2. 상태
    L.append("## 2. 칸 상태(가격과 무관하게 정해짐)\n")
    L.append("| id | 칸 | 족 | 층 | 상태 | 근거 |")
    L.append("|---|---|---|---|---|---|")
    for cid, c in W["cells"].items():
        st = c.get("verdict") if c.get("p_adj") is None else "시험"
        why = c.get("why") if c.get("p_adj") is None else (f"{c['family']} 대표" if c["role"] == "main" else "")
        if c["role"] == "seq" and c.get("p_adj") is None and c["status"] == "시험":
            st, why = "시험(순차)", "대표가 통과할 때만 열림"
        lab = (c.get("label") or "—").replace("|", "\\|")
        L.append(f"| {cid} | {lab} | {c.get('fam') or '—'}·{c.get('role')} | {c.get('layer') or '—'} | "
                 f"{st} | {(why or '—').replace('|', chr(92) + '|')} |")
    for grp, d in STATUS_ONLY.items():
        L.append(f"| {', '.join(d.keys())} | — | — | — | {grp} | " + "; ".join(f"{k}: {v}" for k, v in d.items()) + " |")
    L.append("| H09 | Miller 분산 → KOSDAQ | T2 | — | 설명용 | " + o["checklist"]["H09"]["why"] + " |")
    L.append("| H24 | 공포 과부족 → 반등 | T2 | — | 설명용(멈춤) | " + o["checklist"]["H24"]["why"] + " |")
    L.append("| H39+H48 | 종목 관심 횡단면 | T2 | — | 설명용 | " + o["checklist"]["H39+H48"]["why"] + " |")
    L.append("| H06·H10 | 엇갈림 → 거래대금 / SOXX HL | T1 | — | 설명용 | 부록 2 #1(구조 결측) |\n")
    # M0 시작 달
    L.append("M0 확장 OLS가 처음 적합된 달(X⊥·X̂가 생기는 달)과 구간별 독립 묶음 수:\n")
    L.append("| id | 특징 첫 값 → M0 첫 적합 달 | M0 R²(마지막 달) | 뒤 n | 뒤 묶음 | 앞 n | 앞 묶음 | 2025 n | 이동 K |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for cid, c in W["cells"].items():
        inf = c.get("info") or {}
        if "n_back" not in inf:
            continue
        m0i = inf.get("M0") or {}
        m0 = m0i.get("first_month_fitted") or ("예측 갭 확장 회귀 첫 잔차 " + str((inf.get("gap_model") or {}).get("first_D"))
                                               if inf.get("gap_model") else "—")
        L.append(f"| {cid} | {m0} | {_f(m0i.get('R2_last'), 2)} | {inf['n_back']} | {inf['blocks_back']} | "
                 f"{inf['n_front']} | {inf['blocks_front']} | {inf.get('n_2025', '—')} | {inf.get('shift_K')} |")
    L.append("")
    sm = W["prices_meta"].get("schedule") or {}
    L.append(f"- 일정 더미 원천이 없어 빠진 것: {', '.join(sm.get('missing', [])) or '없음'} "
             f"(가짜 주가 시운전은 실적일·CPI·고용·MSCI 날짜를 넣지 않는다). 원천 자체가 없는 통제: "
             f"{', '.join(sm.get('not_available', []))}.\n")
    notes = [(cid, c["notes"]) for cid, c in W["cells"].items() if c.get("notes")]
    if notes:
        L.append("기록(판정에 쓰지 않음, (a) 씨앗 0):\n")
        for cid, ns in notes:
            L.append(f"- {cid}: " + "; ".join(ns))
        L.append("")
    hdr = ("| id | 뒤 n | 추정 | t | 해석 p | 이동 p (K) | 최종 p | 보정 p | 앞 추정 | 2025 보조 | 강건 비(급락4/상위2/열람8) | "
           "위약 X̂ vs X | 판정 |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for title, R in (("3. (a) 무작위 걸음 가짜 주가 — 씨앗 0", W), ("4. (b) 심은 효과 가짜 주가 — 씨앗 0", P)):
        L.append(f"## {title}\n")
        if R is P:
            L.append(f"- 심은 것: A_rz8(W_KR 글량 rz8) 순위 정규 점수 × β를 진입일 KOSPI ln(고가÷저가)의 '범위 초과분' 로그에 더했다. "
                     f"β {pm['beta']:.3f}는 씨앗과 무관한 잡음 {PLANT_CAL_DRAWS}벌에서 2026 판정 창 {pm['n_calib']}개의 평균 "
                     f"스피어만 IC가 0.30이 되게 정했다(모집단 IC). 씨앗 0의 실현 IC는 {pm['ic_realized_2026']:.3f}"
                     f"(β=0이면 {pm['ic_beta0']:+.3f})이고, 씨앗 {len(ics)}개의 실현 IC는 {min(ics):.2f}~{max(ics):.2f}"
                     f"(평균 {np.mean(ics):.2f})다. 종가·시가·M0 열은 (a)와 같다(같은 씨앗이면 KOSPI 고가·저가만 다름).\n")
        L.append(hdr)
        for cid, c in R["cells"].items():
            if c.get("values") and c["values"].get("est") is not None:
                L.append(_cell_row(cid, c))
        L.append("")
        L.append(f"- 시험 칸: T1 {', '.join(R['tested']['T1'])} (Holm α 0.05, {len(R['tested']['T1'])}칸) / "
                 f"T2 {', '.join(R['tested']['T2'])} (BH q 0.10, {len(R['tested']['T2'])}칸). 순차 열림: "
                 + (", ".join(k for k, c in R["cells"].items() if c["role"] == "seq" and c.get("thr_local")) or "없음")
                 + ".\n")
    # 5. 씨앗
    L.append(f"## 5. 가짜 가격 씨앗 {sw['n_seeds']}개 반복(포럼 특징은 같고 가격 난수만 바꿈)\n")
    L.append("| 시나리오 | 통과 칸 평균 | 1칸 이상 씨앗 | " + " | ".join(f"{k} 통과 / 보정 p≤α / 원 p≤α" for k in t1 + t2) + " |")
    L.append("|---|---|---|" + "---|" * len(t1 + t2))
    for nm, s in (("(a) 무작위 걸음", sw), ("(b′) 심은 효과 IC 0.15", s15), ("(b) 심은 효과 IC 0.30", sp),
                  ("(b″) 심은 효과 IC 0.50", s50)):
        cells = " | ".join(f"{s['cells'].get(k, {}).get('pass', 0)} / {s['cells'].get(k, {}).get('adj_p_le_alpha', 0)}"
                           f" / {s['cells'].get(k, {}).get('raw_p_le_alpha', 0)}" for k in t1 + t2)
        L.append(f"| {nm} | {s['mean_main_passes']:.2f} | {100 * s['share_ge1']:.0f}% | {cells} |")
    L.append(f"\n- 칸마다 '통과 수 / 보정 p ≤ α 수 / 원 p ≤ α 수'(씨앗 {sw['n_seeds']}개 중). α는 T1 0.05, T2 0.10. "
             "IC는 모집단 값(β 고정)이다.")
    L.append("- (a)의 원 p ≤ α 비율로 각 칸의 1종 오류를 점검한다(기대 약 5%·10%). 통과는 앞 부호·강건·위약 조건이 더 붙어 더 드물다. "
             "심은 효과는 KOSPI 고가·저가만 바꾸므로, H08·H70·H66의 수는 (a)와 거의 같아야 정상이다"
             "(H70은 A와 상관이 있으면 효과 일부가 번질 수 있다).\n")
    L.append("H03 씨앗별(IC 0.30):\n")
    L.append("| 씨앗 | 실현 IC | 부분 r | 최종 p | Holm p | 판정 | 이유 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in o["seeds"]["h03_by_seed"]["plant"]:
        L.append(f"| {r['seed']} | {_f(r['ic_realized'], 2)} | {_f(r['est'], 2)} | {_p(r['p_final'])} | {_p(r['p_adj'])} | "
                 f"{r['verdict']} | {r['why'] or '—'} |")
    L.append("")
    # 6. 전체 이동
    L.append("## 6. 전체 원형 이동 200회(§2.6 '우연 통과 기대 칸 수')\n")
    L.append("| 시나리오 | 이동 몫 범위 | 우연 기대(주 칸) | 1칸 이상 | 최대 | 칸별 통과율 |")
    L.append("|---|---|---|---|---|---|")
    for nm, g in (("(a) 무작위 걸음", gw), ("(b) 심은 효과", gp)):
        per = ", ".join(f"{k} {100 * v:.1f}%" for k, v in (g.get("per_cell_pass_rate") or {}).items()) or "없음"
        L.append(f"| {nm} | {g['u_range'][0]:.3f}~{g['u_range'][1]:.3f} | {g['mean_main_passes']:.3f} | "
                 f"{100 * g['p_ge1']:.1f}% | {g['max']} | {per} |")
    L.append("\n- 모든 칸의 특징열을 같은 이동 몫 u(2026 창 수 기준 21~N−21거래일)로 함께 돌리고, 보정·앞 부호·강건·위약·순차까지 "
             "판정 전체를 다시 돌려 센 값이다. (b)에서도 (a)와 비슷하면 심은 효과가 이동으로 깨진다는 뜻이다.")
    if len(t2) == 1:
        L.append(f"- T2가 {t2[0]} 한 칸뿐이라 BH q 0.10이 곧 그 칸의 단측 10% 문턱이고, 라벨 층은 앞 부호 조건이 보조라 우연 "
                 f"통과가 대부분 {t2[0]}에서 나온다. 동결 문서가 적은 '13칸 중 약 0.1칸'과 같은 크기다.\n")
    else:
        L.append(f"- T2 시험 칸은 {len(t2)}칸({'·'.join(t2)})이고 BH q 0.10으로 보정한다. 동결 문서의 기대는 '13칸 중 약 0.1칸'이다.\n")
    # 7. MDE
    L.append("## 7. 검출 한계(MDE = 2.49/√n_eff, 뒤 구간)\n")
    L.append("| id | 뒤 n | n_eff | MDE | 기대 효과 | 검출 가능 |")
    L.append("|---|---|---|---|---|---|")
    for m in W["mde"]:
        L.append(f"| {m['id']} | {m['n_back']} | {m['n_eff']:.1f} | {m['mde']:.2f} | {m['expected']:.2f} | "
                 f"{'예' if m['detectable'] else '아니오'} |")
    L.append("\n- n_eff: T1(1일 목표)은 자유도 n − k, T2는 X⊥ AR(1)과 목표 기간 h의 Bartlett 근사(미리보기와 같음).\n")
    # 8. 해석
    L.append("## 8. 해석(가격을 보기 전에 정해 코드에 고정한 것)\n")
    for s in INTERPRETATION:
        L.append(f"- {s}")
    L.append("")
    # 9. 실제
    L.append("## 9. 실제 판정 실행(이번에는 하지 않음)\n")
    L.append("- 명령: `py judge.py real --confirm-real` (`--h31-reg-date YYYY-MM-DD`는 규제 시행일을 확인했을 때만). "
             "KST 2026-10-24 06:00 전이면 아무 파일도 읽지 않고 멈추고, 가격 파일에 ^KS11·SOXX의 2026-10-23 종가가 없어도 멈춘다.")
    L.append("- 필요한 입력: ① 10/22·10/23 창이 든 포럼 스냅샷으로 다시 만든 특징 표(run_features.py), ② 같은 창의 예측 라벨 "
             "(없으면 판정 코드가 final·preview 라벨로 메모리에서 만든다), ③ `final/results/prices.parquet`(ticker·date·OHLCV, "
             f"{', '.join(list(KR_T.values()) + list(US_T.values()) + [VIX_T])}, 2024-11 이후), `prices_nq_1h.parquet`(NQ=F 60분봉), "
             "`earnings_dates.json`(티커별 실적일. 없으면 yfinance 실적일 날짜만 받는다), `schedule_dates.json`"
             "({\"cpi\": [...], \"jobs\": [...], \"msci\": [...]}, 미국 발표일·MSCI 시행일. 없으면 그 더미가 빠지고 결과에 적힌다).")
    L.append("- 결과: `final/results/judge_real.json`. 목표 기간이 10/23 종가를 넘는 신호는 결측으로 둔다(§2.5).\n")
    # 10. 입력
    L.append("## 10. 입력과 지문\n")
    pi = o["inputs"]["pred"]
    L.append(f"- 예측 라벨 창 특징: {pi.get('source')}. 라벨 단위 {pi.get('units_labeled')}개(파일 {pi.get('label_files')}개), "
             f"유한 값 {pi.get('finite')}.")
    L.append(f"- 감정 층: {o['inputs']['emotion'].get('source')} (국내 창 {o['inputs']['emotion'].get('kr_windows')}개).")
    for k, v in o["inputs"]["features"]["sha256"].items():
        L.append(f"- {k}: `{v}`")
    for k, v in o["frozen_sha256"].items():
        L.append(f"- {k}: `{v}`")
    L.append(f"- judge.py: `{o['code_sha256']}`")
    L.append(f"- 실행 {o['made']}, {o['seconds']}초. 실제 가격 호출 {o['real_price_calls']}회.")
    return "\n".join(L) + "\n"


INTERPRETATION = [
    "원형 이동 ①은 원 특징열(rz8 등 등록 특징)을 2026 유효 창 안에서 돌리고, M0·X⊥·X̂·통계를 매번 다시 계산한다(미리보기와 같음). "
    "이동 범위는 21거래일에 해당하는 칸 수(⌈21 × N ÷ 2026 판정 창 수⌉)부터 N에서 그만큼 뺀 칸까지다. 라벨 층도 2026 전체 특징열을 돌린다"
    "(지식 기간 값은 귀무 분포에만 들어간다).",
    "M0 확장 OLS: 달 m의 창은 m-01 전 창으로 적합한다. 적합 표본은 판정 자격 창(96시간 이하·장애 아님)·칸 표지(H46 휴장 엇갈림 등)·"
    "특징과 M0 열이 모두 있는 창이고, 120개 미만이면 그 달 X⊥·X̂는 결측이다. 더미는 적합 표본에서 1(또는 0)이 2개 미만이면, "
    "열은 랭크를 늘리지 않으면 뺀다. X̂는 상수·M0·가격 통제 몫(포럼 통제 몫 제외), X⊥는 잔차 ÷ 적합 잔차 SD다.",
    "미국 M0는 §2.3 미국 행(KOSPI·하닉 D 종가 대 종가·시가→종가, SOX 직전·r5·r20·DD20·RV20, VIX 1일 변화, NQ=F 직전 종가→창 마감)에 "
    "결번율(§2.2 '모든 텍스트 가설의 통제 변수')과 한국 M0와 같은 달력 더미(§2.4 '그날을 빼지 않고 더미로 통제')를 더한다. "
    "일정 더미는 FOMC·실적일·KOSPI/미국 옵션 만기에 CPI·고용·MSCI(schedule_dates.json이 있을 때만)를 더하고, 없는 원천은 결과 meta에 적는다.",
    "특징 정의에 붙은 잔차화(H35의 A·|r|·r20·r60·결번율, H31의 SOX r5·r20·RV20·DD20, H36의 |r20|·|r5|·A, H50의 하닉·삼전·MU·NQ와 "
    "메모리 일정 더미(MU·삼전 실적일))는 M0와 한 회귀로 함께 뺀다(FWL로 같은 표본이면 두 번 빼는 것과 같다). H50은 M0 밖 통제가 "
    "4개를 넘어 그 열만 ridge(λ는 GCV, 특징만 사용).",
    "방향(T2) 판정 회귀의 통제는 §2.3 공통(대상 r(D−1)·r5·r20·RV20·g(D), 한국 대상이면 전야 SOX)에 칸별 통제를 더한 것이다"
    "(H36 = 공통 + |r20|·r20×RV20, H46·H31·H50은 대상 자산의 같은 값).",
    "T1 점추정은 HAR-K 안 부분 상관(FWL), t는 HAC(5)(지연 짝은 창 위치 차가 정확히 l인 행끼리, n/(n−k) 보정). H52는 월요일 순번으로 "
    "HAC(5)를 잰다. T2 점추정은 순위 부분 상관(통제 순위로 잔차화), t는 순위 z 회귀의 Hodrick 1B(1기간 수익 = 진입일 종가 대 직전 종가 "
    "로그수익). H31(MDD 목표)은 1기간 합으로 나뉘지 않아 HAC(h)다. 해석 p는 단측 t 분포(자유도 n − k).",
    "90% 구간: T1 SE = 1/√(n − k), T2 SE = 1/√n_eff(동결 MDE 표와 같은 근사). 기각: 뒤 부호 반대 또는 상한 < 기대 효과 절반"
    "(T1 0.075, T2 0.05, 순차 0.05).",
    "메아리 위약: X̂와 X(원값)를 같은 부분 표본(X̂가 있는 뒤 구간 창)에서 같은 회귀에 넣어 비교한다. T1은 주 특징이 원값이라 X⊥판을 "
    "따로 재고(같은 원형 이동), 'X⊥ 통과' = §2.7 통과 그대로(그 칸의 국소 문턱 이하 + 예상 부호 + 앞 구간 같은 부호 + 강건 조건). "
    "T2는 주 특징이 X⊥라 칸 자체의 통과 조건이다.",
    "강건 3개: 급락 4일·열어 본 8일은 창 (open, close)이 그 달력일과 겹치면 뺀다(h01·미리보기의 '가장 많이 빼는' 읽기). "
    "기여 상위 2일 = 예상 방향 × FWL(또는 순위) 잔차 곱이 가장 큰 두 창. 세 판 모두 점추정 ≥ 원래의 50%(같은 부호 포함)여야 한다. "
    "H03은 §3 기각 칸대로 7/29~31만 뺀 판도 50% 이상이어야 한다(못 채우면 취약).",
    "H03·H52는 §2.4대로 번호판과 저장판(A_saved_rz8·WK_saved_rz8)의 뒤 구간 부호가 같아야 통과로 센다.",
    "라벨 층(H66·H67·H68·H70, 설명용 H64·H65·H60~H63)의 뒤 구간은 2026-07-01~10-23이고, 앞 구간 부호는 'AI 지식 기간' 보조로만 낸다"
    "(부록 3·4: 앞 구간 확인 없음). 그 밖 칸의 뒤 구간은 2026-06-01~10-23이다. 라벨 층의 M0 계수와 rz8 기준은 7월 전(지식 기간) "
    "라벨을 예열로 쓴다(prechecks_full 조건, 칸 '기록'에 적는다).",
    "2025 보조(§2.5): 2025 창의 같은 통계를 '같은 부호인지' 확인용으로만 내고, 반대 부호면 장세 전환 위험으로 한 줄 적는다(판정에 안 씀).",
    "H68(강한 하락 예측)은 부록 4가 T2 판정 칸으로 등록했다. 부록 9의 '설명용'은 미리보기 처리라, 최종 판정에서는 점검표·최소 표본 규칙을 "
    "그대로 적용한다(이번 자료: 7/1~10/23 독립 묶음 7 < 8이라 설명용).",
    "최소 표본: 구간마다 h거래일 넘게 떨어진 창을 앞에서부터 고른 수가 8개 이상(라벨 층은 뒤 구간만). 못 채우면 설명용이고 보정 칸 수에서 뺀다.",
    "대표의 p는 족 안 '시험' 상태 특징(대표 + 열 수 있는 순차 칸)을 같은 이동 몫으로 함께 돌린 최대 통계(각 칸 이동 분포로 표준화)로 낸다"
    "(Westfall-Young). 칸이 하나면 보통의 이동 p와 같다.",
    "족 안 순차: '대표가 통과한 기준값' = 대표가 통과한 국소 문턱(Holm α/(m−i+1), BH k*·q/m). 멈춤 칸과 가격 전에 정해진 설명용 칸은 "
    "건너뛰고(실패로 세지 않음), 시험한 순차 칸이 통과하지 못하면 뒤는 설명용이다.",
    "H31은 8월 단일종목 레버리지 규제 시행일이 없으면 설명용(§5.1 #11). 날짜를 주면 ±10거래일을 빼고 T2에 넣는다. "
    "이번 자료로는 날짜가 있어도 앞 구간 독립 묶음이 8개 미만이라 설명용이다. NVDL(사후 단어)을 뺀 판(LL_US_noNVDL_d20)은 '기록'으로만 낸다.",
    "H52 목표 '월요일 |갭| 잔차' = |갭| − 예측 |갭|. 예측 |갭|은 금 SOX·QQQ |r|, NQ=F 금 종가→창 마감 |변화|, 금 KOSPI |r|를 "
    "앞 월요일(D−1까지)만으로 확장 OLS(매 월요일 재적합, 최소 20개 = H04 B⊥ 확장 회귀와 같은 최소)해 만든다(VKOSPI·ES·BTC는 원천이 "
    "없어 뺌). 판정 회귀는 상수 + WK, 월요일 순번 HAC(5). 월요일 칸은 M0가 없어 위약을 생략한다. §3의 '2025-04 이후 월요일은 보조'는 "
    "§2.5(2025 = 예열·같은 부호 보조)에 맞춰 2025 안에서 보조 확인에 쓰는 월요일의 시작으로 읽는다(2026 판정 표본을 줄이지 않음).",
    "H57a·H57b: 부록 1의 'H01 실패 시 H57b를 활동량 단계로 바꿔 시험'은, 사전 실패 뒤 부록 2가 보류로 두고 부록 3이 들뜸 지수로 바꿨으므로 "
    "쓰지 않는다. 들뜸 지수(감정 층)가 부록 8 검수에서 실패해 둘 다 설명용이다.",
    "H14·H19·H49·H51(T2 대표)과 그 족 순차는 S·환호 사전이 측정 실패(부록 2 #4)이고 TMP(H02)를 만들지 않았으므로 §2.6대로 '멈춤'이다"
    "(부록 3은 H57·H59만 들뜸 지수로 옮겼다).",
    "H67 등록 특징 P_sd(창당 예측 쌍 20개 이상)는 값이 0창이라 자동 설명용, H04는 corr(rz8 U, rz8 N) > 0.9라 자동 설명용이다(§2.7).",
    "가짜 주가: KRX·NYSE 거래일 날짜(loader_calendar + exchange_calendars)에 GARCH(1,1) 요인 + 고유 잡음으로 로그수익을 만들고, "
    "시가(갭 몫 35%)·고가·저가(ln 고가÷저가 = |시가→종가| + 범위 초과분)·거래량을 만든다. VIX는 QQQ와 반대로 움직이는 로그 OU, "
    "NQ=F는 60분봉 무작위 걸음. 실제 가격 파일은 열지 않는다.",
    "심은 효과: 'IC 0.3'은 모집단 IC로 읽는다. β 하나를 씨앗과 무관한 잡음 40벌의 평균 IC로 정해 모든 씨앗에 쓴다. 표본 IC를 0.30에 "
    "억지로 맞추면 우연 상관까지 맞춰져 씨앗마다 실제 효과가 달라진다(첫 구현에서 확인하고 바꿈). 심는 곳은 범위 초과분뿐이라 "
    "종가·시가·M0·HAR 통제는 무작위 걸음과 같다.",
]


if __name__ == "__main__":
    raise SystemExit(main())
