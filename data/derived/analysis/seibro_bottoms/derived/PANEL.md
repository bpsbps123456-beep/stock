# SEIBRO 바닥 연구 자료판 (`code/build.py`, 흐름–사건 연결 전)

- 만든 시각: 2026-11-04T08:32:51 KST. 인자: 결제 ≤ 2026-11-03, 보관 ≤ 2026-11-02, 가격 ≤ 2026-11-03.
- 미국 거래일 2022-12-28 ~ 2026-10-30 (964일), SEIBRO 결제 날짜 2023-01-02 ~ 2026-11-03 (1002개), 보관 2023-01-02 ~ 2026-11-02 (1001개).
- 사건 점검: B15 일치, B10 일치. F 날 28개(국면 8개), 미정 1개.
- 묶음 규칙: `--merge-rule cal` (주 분석, BRIEF §2 연결 전 수정 2026-11-04).

## 0. 분석 때 주의

| 항목 | 값 | 뜻 |
|---|---|---|
| 바닥 날(W3)이 합친 묶음 안 | 없음 | 바닥 날이 묶음 안이라 W3 값이 묶음 평균(예측 이틀치면 두 날 나눔)이다. B15 6개 중 0개 |
| F 날이 합친 묶음 안 | 없음 | 그 F 날의 W3 값도 묶음 평균 |
| 보관 대응 | s(t) ≤ D 규칙(세션 지연으로는 T+2 3·T+1 2) | 보관 D는 SEIBRO 날짜가 D 이하인 거래일까지 담는다. 대응 보관이 없는 거래일 C 값은 앞 보관(c_stale 8일) |
| N⁺ 창 정의 | 하루 N⁺의 합 | 앞선 S1(5일 ISIN 합의 양수)보다 분모가 커서 X4·L2가 작다(앞선 S1 ÷ X4 중앙 1.3679) |
| 버린 결제 날짜 | 2025-01-02 22M, 2025-01-13 315M | 예측 거래일 없는 날짜(관측 empty)의 금액. 묶음 규칙 밖이라 버렸다(전체 tot 337M) |
| BRIEF 초판 '지금' 행(고침) | 마지막 고점 2026-10-05 589.51 | 8/17(559.12)이 아니다. 10/05가 더 높아 10%·15% 모두 10/05가 고점, 가격 끝 낙폭 -0.211 |
| 사슬 안 측정 표지 차이 | 없음 | 분류 열이 다른 사슬 CONI, CONY, MSTY, TSLQ은 L(인버스)·single(INCOME)만 달라 측정에 영향 없음 |

## 1. 핵심 점검

| 점검 | 기준 | 결과 |
|---|---|---|
| 최신 날짜 대응 | SEIBRO 2026-11-02 ← 미국 2026-10-29 | 맞음 |
| 날짜표 대 앞선 연구(≤ 10/23) | 995날짜 | 거래일 995 · 관측 995 · 일치 판정 995 같음 |
| 거래일 빠짐 | NYSE 세션 대비 | 0일 |
| 목록 간 금액 불일치 | 같은 날·ISIN | 0줄 |
| SOXL 순매수 보존 | 거래일 합 = SEIBRO 날짜 합 | -665,942,570 = -665,942,570 (차 0) |
| 전체 순매수 보존 | 묶음 밖 날짜 뺀 뒤 | 차 0, 버린 날짜 23개 tot 337,451,402 |
| 보관 SOXL 2026-07-16 | ≈ 4.448e9 | 4,447,914,156 |
| 5일 X4 대 앞선 S1 | 같은 끝 거래일 | 상관 0.9877 (n 851), 묶음 없는 창 0.9894 |

- 5일 X4와 앞선 S1이 다른 이유: ① 분모 N⁺를 앞선 연구는 'ISIN별 5일 순매수 합의 양수 부분'으로, 여기서는 '하루 N⁺의 5일 합'으로 잰다(하루 단위가 크다 → X4가 작다). ② 앞선 연구 창은 '비지 않은 SEIBRO 날짜 5개'라 이틀치 날이 끼면 6거래일을 담는다(28창). ③ 합친 묶음은 금액을 거래일에 똑같이 나눈다. 앞선 방식 분모로 다시 잰 S1_alt와 앞선 S1의 상관은 전체 0.9988, 묶음 없는 5거래일 창 1.0(크기 비 중앙 1.0)다. 곧 이 자료판은 묶음 없는 창에서 앞선 S1을 그대로 재현하고, 전체 상관이 낮아지는 것은 ②·③ 때문이다.

## 2. 파일

| 파일 | 내용 | 행 | 열 | 날짜 범위 |
|---|---|---|---|---|
| units.parquet | SEIBRO 날짜 → 미국 거래일, 묶음 | 1002 | 24 | 2023-01-02 ~ 2026-11-03 |
| flows_trade_product.parquet | 미국 거래일 × 상품 금액 | 82120 | 42 | 2022-12-28 ~ 2026-10-30 |
| measures_daily.parquet | 거래일 측정 26개 + 재료·절단·시각 | 964 | 92 | 2022-12-28 ~ 2026-10-30 |
| measures_win.parquet | 창(L 1·5·15) 값·P·E, long | 75192 | 11 | 2022-12-28 ~ 2026-10-30 |
| price_feats.parquet | SOXX 가격 특성 | 1111 | 18 | 2022-06-01 ~ 2026-11-03 |
| custody_daily.parquet | 보관 날짜 × 상품 | 50050 | 13 | 2023-01-02 ~ 2026-11-02 |
| events.json | 사건(B15·B10·F·국면·지금) | 11 |  |  |
| semis_1x.json | 1배 반도체 명단 | 53 |  |  |
| qa.json | 점검 수치 전부 |  |  |  |

## 3. 날짜표와 묶음

- 규약: SEIBRO 날짜 = 미국 결제일 + 1일(주말 → 월). 결제 달력 = NYSE 세션 − 연준 휴일 + 2025-01-09. 2024-05-28 전 거래는 T+2.
- 관측 분류 일치율 0.9741 (불일치 26날짜), 예측 이틀치 7날짜. 표시 7날짜 → 합친 묶음 7개(담긴 거래일 14일, 묶음 최대 2거래일). 끝 묶음 0개.
- 묶음 규칙(cal, 기본): 예측 2거래일 이상 날짜는 그 날짜 하나를 묶음으로 두고 담긴 거래일 수로 나눈다. 빈 날이 낀 어긋남(관측 또는 예측 empty)은 앞뒤 '빈 날 아닌' 날짜와 한 묶음으로 합치고, 겹치는 묶음은 하나로 잇는다(사이의 빈 날짜 포함). 금액은 묶음 거래일 수로 똑같이 나눈다.
- 관측 'double'인데 예측 'normal'인 날은 합치지 않는다. 앞뒤 이웃 ratio가 정상이라 날짜 밀림이 아니라 실제 거래 급증일이다(앞선 연구 README도 어긋난 25날짜를 거래 폭증일로 봤다). BRIEF 초판 규칙은 이 날까지 합쳐 바닥 날 값을 이웃과 평균했다.
- 민감도판: `--merge-rule pred`(예측 이틀치도 이웃과 합침), `--merge-rule brief`(초판 규칙). 다른 `--out`에 쓴다.

### 3.1 표시 날짜

| SEIBRO | 예측 거래일 | ratio | obs | pred | 앞 이웃 | 뒤 이웃 | 묶음 |
|---|---|---|---|---|---|---|---|
| 2023-10-12 | 2023-10-06;2023-10-09 | 1.77 | double | double | 0.78 | 1.11 | 203 |
| 2024-05-30 | 2024-05-24;2024-05-28 | 2.13 | double | double | 1.53 | 1.02 | 368 |
| 2024-10-16 | 2024-10-11;2024-10-14 | 2.39 | double | double | 0.96 | 1.38 | 467 |
| 2024-11-13 | 2024-11-08;2024-11-11 | 3.48 | double | double | 1.26 | 1.61 | 487 |
| 2025-10-15 | 2025-10-10;2025-10-13 | 2.63 | double | double | 0.90 | 1.28 | 727 |
| 2025-11-13 | 2025-11-10;2025-11-11 | 1.54 | normal | double | 1.01 | 0.85 | 748 |
| 2026-10-14 | 2026-10-09;2026-10-12 | 2.06 | double | double | 1.36 | 1.60 | 987 |

### 3.2 합친 묶음과 사건 근처(±20세션)

| 묶음 | SEIBRO 날짜 | 미국 거래일 | 거래일 수 | 이유 | B10 ±20(* B15) | F ±20 수 | 묶음 안 F 날 |
|---|---|---|---|---|---|---|---|
| 203 | 2023-10-12 | 2023-10-06;2023-10-09 | 2 | 2023-10-12:pred2 | 2023-10-30*[k -16..-15] | 4 |  |
| 368 | 2024-05-30 | 2024-05-24;2024-05-28 | 2 | 2024-05-30:pred2 |  | 0 |  |
| 467 | 2024-10-16 | 2024-10-11;2024-10-14 | 2 | 2024-10-16:pred2 |  | 0 |  |
| 487 | 2024-11-13 | 2024-11-08;2024-11-11 | 2 | 2024-11-13:pred2 | 2024-11-15[k -5..-4] | 0 |  |
| 727 | 2025-10-15 | 2025-10-10;2025-10-13 | 2 | 2025-10-15:pred2 |  | 0 |  |
| 748 | 2025-11-13 | 2025-11-10;2025-11-11 | 2 | 2025-11-13:pred2;obs_normal/pred_double | 2025-11-20[k -8..-7] | 0 |  |
| 987 | 2026-10-14 | 2026-10-09;2026-10-12 | 2 | 2026-10-14:pred2 | 2026-09-14[k 19..20] | 5 |  |

- 바닥 W3(k = 0)가 합친 묶음 안인 B10: 없음.
- 앞선 연구 날짜표와 다른 날짜(≤ 10/23): 0개. 차이는 끝 쪽 가운데 중앙값 창이 새 자료로 바뀐 것이다(ratio 최대 차 0.023912).

## 4. 분류

- classes.csv 3271행(상품 3178개). 티커 없는 규칙 행 622개는 ISIN을 상품 키로 썼다. 사슬 안 분류가 다른 상품: ['CONI', 'CONY', 'MSTY', 'TSLQ'].
- 새 ISIN(10/23 뒤 결제·보관에서 처음 보임) 16개. 레버리지·발행사 단어가 든 것은 이름으로 따로 판단했다(검토 안 된 것 0개). 합계 금액이 작아 측정 영향은 미미하다.

| isin | SEIBRO 이름 | 상품 키 | utype | 출처 | 첫 등장 | 날짜 수 | 든 목록 | tot 합 |
|---|---|---|---|---|---|---|---|---|
| AU0000424897 | BRAIIN LIMITED | AU0000424897 | EQ_SINGLE | new_rule | 2026-10-30 | 1 | 14 | 3925324.0 |
| CA85207H1047 | SPROTT PHYSICAL GOLD TRUST | CA85207H1047 | CMDTY | new_manual | 2026-10-27 | 1 | 2 | 2059091.0 |
| IE000GVCI8S8 | DAVION HEALTHCARE PLC | IE000GVCI8S8 | EQ_SINGLE | new_rule | 2026-10-26 | 1 | 4 | 911724.0 |
| KYG7396L1370 | JIADE LIMITED SPLR 2641921616002 KYG7396L1297 | KYG7396L1370 | EQ_SINGLE | new_rule | 2026-11-03 | 1 | 14 | 2279483.0 |
| MHY2258E1077 | AI OKTO CORPORATION | MHY2258E1077 | EQ_SINGLE | new_rule | 2026-11-02 | 1 | 4 | 2406910.0 |
| US0029421007 | AARDVK THPTC ORD | US0029421007 | EQ_SINGLE | new_rule | 2026-10-27 | 1 | 4 | 1338013.0 |
| US0209522061 | ALPHA MODUS HLDGS SPLR 2641923096002 US0209521071 | US0209521071 | EQ_SINGLE | new_chain | 2026-10-26 | 1 | 4 | 1177469.0 |
| US0321085732 | AMPLIFY GLOBAL CLOUD TECHNO AMPLIFY GLOBAL CLOUD TECH MRGR 955998811 US26924G8050 | US0321085732 | EQ_INDEX | new_manual | 2026-10-30 | 1 | 4 | 1851452.0 |
| US1311001093 | CALLAN JMB INC | US1311001093 | EQ_SINGLE | new_rule | 2026-10-26 | 1 | 4 | 1336505.0 |
| US45773H4092 | INOVIO PHARMACEUTICALS INC COM USD0.001(POST REV SPLIT) SPLR 955968915 US45773H2013 | US45773H4092 | EQ_SINGLE | new_rule | 2026-11-03 | 1 | 4 | 1572987.0 |
| US5000811043 | KODIAK ROBOTICS INC | US5000811043 | EQ_SINGLE | new_rule | 2026-11-02 | 1 | 4 | 1349411.0 |
| US51255A2015 | LAKEWOOD-AMEDEX SPLR 2642122802002 US51255A1025 | US51255A1025 | EQ_SINGLE | new_chain | 2026-10-28 | 1 | 4 | 1638408.0 |
| US67021W5094 | NUBURU INC SPLR 2642600635002 US67021W4006 | US67021W4006 | EQ_SINGLE | new_chain | 2026-11-03 | 1 | 14 | 3132007.0 |
| US90137F7078 | 22ND CENTURY GROUP, INC. SPLR 2641012450002 US90137F6088 | US90137F6088 | EQ_SINGLE | new_chain | 2026-10-27 | 1 | 134 | 3488759.0 |
| US9396531017 | WASHINGTON REAL ESTATE INVESTMENT TRUST | US9396531017 | EQ_SINGLE | new_manual | 2026-10-27 | 1 | 4 | 1711742.0 |
| VGG1110E1079 | Biohaven | VGG1110E1079 | EQ_SINGLE | new_rule | 2026-10-26 | 1 | 4 | 1891130.0 |

- 신규 상장(분모 28일 제외): 앞선 ipo.json 31건을 그대로 썼다. 새 후보 점검(앞선 ipo.py 기준: 첫 등장 ≥ 2026-09-24, ipo.json 밖, 첫 20 SEIBRO 날짜 tot ≥ 2천만 달러, yfinance 첫 일봉 ≥ 첫 등장 − 45일): 후보 3개, 규칙상 신규 상장 0개, 10/23 뒤 처음 보인 것 0개. 더한 것은 없다(아래 표).

| isin | product | name | first_seen | dates_seen | tot_first20 | yf_first | rule_ipo |
|---|---|---|---|---|---|---|---|
| US30260D1037 | FIGS | FIGS INC CL A | 2026-09-29 | 6 | 34634136 | 2021-05-27 | False |
| US49876K2024 | GRML | KLOTHO NEUROSCIENCES, INC. SPLR 2641921890002 US49876K1034 | 2026-09-25 | 5 | 24497703 | 2022-04-29 | False |
| US66987P5089 | SDEV | NOVABAY PHARMACEUTICALS INC | 2026-09-30 | 9 | 48555086 | 2007-10-26 | False |

- ipo.json의 GETYQ(첫 거래 2026-10-13, SEIBRO 첫 등장 2023-04-27)는 티커 변경으로 보이는 오탐이지만 지시대로 그대로 썼다(10/13~11/10 N⁺에서 빠진 양수 순매수 합 0 USD).

### 4.1 1배 반도체(R1) 명단

- watchlist 4묶음(칩 설계, 파운드리·후공정, 메모리·스토리지, 장비·소재·테스트)의 점 없는 티커 + SKHY·SKHYV: 45개, 자료에 있는 것 42개. 없는 것: AOSL, MTSI, SKHYV.
- classes semi = True & L = 1 & sign = 0: AMDY(INCOME), ASML(EQ_SINGLE), CHPY(INCOME), DISK(EQ_INDEX), DRAM(EQ_INDEX), NVDY(INCOME), NVYY(INCOME), PSI(EQ_INDEX), SMH(EQ_INDEX), SOXQ(EQ_INDEX), SOXX(EQ_INDEX), STM(EQ_SINGLE), XSD(EQ_INDEX). 이 가운데 INCOME(커버드콜) 상품은 R5에도 들어간다(정의 그대로, 겹침 금액은 qa.json `r1_r5_overlap`).

- 명단: ADI, AEHR, ALAB, AMAT, AMD, AMKR, ~~AOSL~~, ARM, ASML, ASX, AVGO, AXTI, CBRS, CRDO, ENTG, FORM, GFS, INTC, KEYS, KLAC, LRCX, MCHP, MPWR, MRVL, ~~MTSI~~, MU, NVDA, NVTS, NXPI, ON, QCOM, RMBS, SIMO, SKHY, ~~SKHYV~~, SNDK, STM, STX, SWKS, TER, TSEM, TSM, TXN, UMC, WDC (취소선 = 목록에 한 번도 없음)

## 5. 보관

- Δ(SOXL 좌수) × 종가를 보관 날짜 D 이하 마지막 NYSE 세션에서 j세션 앞 거래일의 SOXL 순매수에 회귀했다(R², 괄호 안 기울기). 합친 묶음 거래일이 든 행은 뺐다.

| 구간 | j=0 | j=1 | j=2 | j=3 | j=4 | n |
|---|---|---|---|---|---|---|
| all | 0.0139 (-0.116) | 0.0 (0.003) | 0.9189 (0.966) | 0.0013 (0.036) | 0.0074 (-0.086) | 954 |
| T+2(보관 ≤ 2024-05-24) | 0.0091 (0.097) | 0.0398 (0.203) | 0.078 (0.287) | 0.9578 (1.005) | 0.0854 (0.302) | 355 |
| T+1(보관 ≥ 2024-06-03) | 0.0145 (-0.119) | 0.0 (0.001) | 0.9346 (0.974) | 0.0006 (0.024) | 0.0082 (-0.091) | 599 |
| 2023 | 0.005 (0.073) | 0.0321 (0.184) | 0.0669 (0.265) | 0.9461 (0.997) | 0.0604 (0.253) | 251 |
| 2024 | 0.0258 (0.156) | 0.0293 (0.167) | 0.705 (0.828) | 0.0926 (0.3) | 0.0543 (0.228) | 243 |
| 2025 | 0.0061 (0.079) | 0.0071 (0.086) | 0.9319 (0.988) | 0.0104 (0.104) | 0.0019 (0.045) | 249 |
| 2026 | 0.0238 (-0.152) | 0.0002 (-0.015) | 0.9349 (0.973) | 0.0001 (0.01) | 0.0137 (-0.117) | 211 |

- 대응 규칙(2026-11-04 수정): 보관 D는 SEIBRO 날짜 s(t) ≤ D인 거래일 t까지의 결제를 담는다(Δ좌수 × 종가 ~ 순매수 R² 0.9827). 세션 지연으로 재면 T+2 j = 3, T+1 j = 2이 가장 맞지만, NYSE 휴장·연준 휴일 언저리에서 하루 어긋나 대응에는 쓰지 않는다. 평가 종가는 D와 같은 미국 날짜(휴장이면 그 전 세션) SOXL 종가다.
- 거래일 중 대응 보관이 없어 앞 보관을 쓴 날(c_stale): 8일. 보관 Top50에 SOXL이 없는 날: 0일.

## 6. 측정

- 26개(BRIEF §4) 정의와 재료 열은 아래 사전에 있다. 창 값 = 금액은 L일 합, 비율은 L일 재료 합의 비. 목록 밖 상품은 0으로 더하고 절단 표시를 남긴다.
- 절단 비율(L = 1 거래일 기준, 값이 0으로 채워진 날의 비율). '부분'은 합친 묶음 안 일부 SEIBRO 날짜에만 나타난 날이다.

| measure | 기준 | 전체 | 부분(묶음 안) | B10±20 | B15±20 | F±20 | F당일 |
|---|---|---|---|---|---|---|---|
| A1 | 목록 합(절단 없음) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| A2 | 목록 합(절단 없음) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| A3 | 목록 합(절단 없음) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| A4 | 목록 합(절단 없음) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| X1 | cens_SOXL | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| X2 | cens_SOXL | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| X3 | cens_SOXL | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| X4 | cens_SOXL | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| X5 | cens_SOXL | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| X6 | cens_SOXL | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| H1 | cens_SOXS | 0.004 | 0.000 | 0.007 | 0.008 | 0.006 | 0.000 |
| H2 | cens_SOXL+cens_SOXS | 0.004 | 0.000 | 0.007 | 0.008 | 0.006 | 0.000 |
| H3 | zero_INV | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| H4 | zero_INV | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| L1 | zero_EQ3L | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| L2 | zero_EQ3L | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| L3 | zero_LSGL | 0.001 | 0.000 | 0.000 | 0.000 | 0.003 | 0.000 |
| L4 | zero_LIDX | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| R1 | zero_SEMI1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| R2 | zero_IDX1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| R3 | zero_BC | 0.002 | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 |
| R4 | zero_SGL1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| R5 | zero_INC | 0.038 | 0.000 | 0.046 | 0.033 | 0.035 | 0.000 |
| C1 | cens_cust_SOXL | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| C2 | cens_cust_SOXL | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| C3 | 목록 합(절단 없음) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

- SOXL이 목록에 없는 거래일: 없음. SOXS가 없는 거래일: 2025-11-03, 2026-01-26, 2026-03-04, 2026-07-14(그날 H1 = 0, H2 = 1). 묶음 측정(H3~R5)의 절단은 '구성원 하나도 없음'만 셀 수 있다. 50위 밖 구성원 금액은 늘 빠져 있다.
- R1 대 R5 겹침(반도체 INCOME 상품): 거래일 순매수 합 292,984,314 USD, 그런 상품이 목록에 든 날 157일.

## 7. 사건

| 바닥 | 묶음 | 10% 고점 | 낙폭10 | 15% 고점 | 낙폭15 | SEIBRO | 합침 | 아는 날 | 진입 | k | 진입 시가 상승 | ±20 합친 날 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2023-10-30 | B15 | 2023-08-01 | -0.182 | 2023-08-01 | -0.182 | 2023-11-02 |  | 2023-11-03 | 2023-11-03 | 4 | 0.068 | 2 |
| 2024-04-19 | B15 | 2024-03-07 | -0.166 | 2024-03-07 | -0.166 | 2024-04-24 |  | 2024-04-25 | 2024-04-25 | 4 | 0.049 | 0 |
| 2024-08-07 | B15 | 2024-07-10 | -0.249 | 2024-07-10 | -0.249 | 2024-08-09 |  | 2024-08-12 | 2024-08-12 | 3 | 0.064 | 0 |
| 2024-09-06 | B10 | 2024-08-21 | -0.134 |  |  | 2024-09-10 |  | 2024-09-11 | 2024-09-11 | 3 | 0.038 | 0 |
| 2024-11-15 | B10 | 2024-10-14 | -0.119 |  |  | 2024-11-19 |  | 2024-11-20 | 2024-11-20 | 3 | 0.009 | 2 |
| 2025-04-08 | B15 | 2025-01-22 | -0.343 | 2024-10-14 | -0.354 | 2025-04-10 |  | 2025-04-11 | 2025-04-11 | 3 | 0.078 | 0 |
| 2025-11-20 | B10 | 2025-10-29 | -0.134 |  |  | 2025-11-24 |  | 2025-11-25 | 2025-11-25 | 3 | 0.041 | 2 |
| 2026-03-30 | B15 | 2026-02-25 | -0.158 | 2026-02-25 | -0.158 | 2026-04-01 |  | 2026-04-02 | 2026-04-02 | 3 | 0.050 | 0 |
| 2026-06-05 | B10 | 2026-06-03 | -0.123 |  |  | 2026-06-09 |  | 2026-06-10 | 2026-06-10 | 3 | 0.022 | 0 |
| 2026-07-29 | B15 | 2026-06-22 | -0.290 | 2026-06-22 | -0.290 | 2026-07-31 |  | 2026-08-03 | 2026-08-03 | 3 | 0.058 | 0 |
| 2026-09-14 | B10 | 2026-08-17 | -0.110 |  |  | 2026-09-16 |  | 2026-09-17 | 2026-09-17 | 3 | 0.040 | 2 |

- 아는 날 = 바닥 거래일이 든 묶음의 마지막 SEIBRO 날짜 다음 KSD 영업일 09:00 KST. 묶음이 아니면 s(t) 다음 영업일과 같다(events.json에 둘 다).
- F 날 28개. 국면 밖(none) 0개, 흐름 없음(10/29 뒤) 0개. 미정(뒤 20세션 안 됨, 3% 조건 아직 아님) 1개: 2026-11-02. ②를 '그날 뺀 직전 60세션 최고'로 바꿔도 달라지는 F 날: 없음.

| 국면(바닥) | 10% 고점 | 끝 | F 날 |
|---|---|---|---|
| 2023-10-30 | 2023-08-01 | 2023-10-30 | 4 |
| 2024-04-19 | 2024-03-07 | 2024-04-19 | 0 |
| 2024-08-07 | 2024-07-10 | 2024-08-07 | 4 |
| 2024-09-06 | 2024-08-21 | 2024-09-06 | 1 |
| 2024-11-15 | 2024-10-14 | 2024-11-15 | 0 |
| 2025-04-08 | 2025-01-22 | 2025-04-21 | 8 |
| 2025-11-20 | 2025-10-29 | 2025-11-20 | 0 |
| 2026-03-30 | 2026-02-25 | 2026-03-30 | 1 |
| 2026-06-05 | 2026-06-03 | 2026-06-05 | 0 |
| 2026-07-29 | 2026-06-22 | 2026-07-29 | 4 |
| 2026-09-14 | 2026-08-17 | 2026-09-14 | 1 |
| current | 2026-10-05 | 2026-11-03 | 5 |

- 지금(2026-11-03 종가 465.00, 7/29 465.00 아래: 아니오): 10% 지그재그 down, 마지막 고점 2026-10-05 589.51 대비 -0.211. 15% 지그재그 down, 마지막 고점 2026-10-05 589.51 대비 -0.211(미확정, 판정하지 않음).

## 8. BRIEF와 다르거나 해석한 점

1. 묶음 규칙 cal(BRIEF §2 연결 전 수정, 2026-11-04): 예측 이틀치 날짜는 그 날짜 하나만 묶음으로 두고 두 거래일에 나눈다. 빈 날이 낀 어긋남만 앞뒤 이웃과 합친다. 관측 double·예측 normal 날은 거래 급증일이라 합치지 않는다(앞선 연구 README: 어긋난 25날짜는 거래 폭증일).
2. 겹치는 묶음은 하나로 이었다(연결 성분, 사이 빈 날짜 포함). BRIEF '그 날짜와 앞뒤 한 날짜'를 겹칠 때 이어 붙인 해석이다. 그래서 묶음이 4거래일을 넘을 수 있다.
3. 앞뒤 이웃 '빈 날 아닌 날짜' = 예측 거래일이 있거나 관측이 empty가 아닌 날짜.
4. N⁺는 상품(사슬) 단위로 하루마다 잰다. 창 값은 하루 N⁺의 합이다(앞선 연구 S1은 ISIN별 5일 합의 양수 부분). 그래서 5일 X4가 앞선 S1보다 작다.
5. 신규 상장 28일 제외를 N⁺뿐 아니라 ΣB_eq·현물 매수(L4 분모)에도 적용했다(앞선 연구 분모 규칙). 판정 날짜는 묶음의 마지막 거래일. 제외분은 N_plus_ipo·B_eq_ipo 열로 남겼다.
6. DUMMY ISIN(US1234567824)은 A1~A4를 포함한 모든 측정에서 뺐다(결제 17날짜, tot 합 18.0M USD).
7. R1에 classes semi = True & L = 1 & sign = 0 행을 그대로 넣어 반도체 INCOME 상품이 R5와 겹친다(이 기간 목록에 든 것: CHPY·NVDY·NVYY, 순매수 합 293M USD).
8. 목록 밖(절단) 상품은 합에서 0으로 더했다. 비율은 0을 넣은 창 합으로 계산한다(H2는 SOXS 절단 날 1에 가까워짐). 절단 비율은 §6.
9. 보관 평가 종가는 D와 같은 미국 날짜, D가 휴장이면 그 전 세션 종가다. 보관 → 거래일 지연은 T+2·T+1 구간을 따로 실측해 구간별로 썼다.
10. C1·C2·C3은 거래일마다 '대응 거래일 ≤ t인 마지막 보관'을 쓴다. 대응 보관이 없으면 앞 보관(c_stale)이다.
11. F ②의 60세션 최고와 dd60은 그날을 포함한 60세션이다. F는 2023-01-02 ~ 가격 끝 날짜에서 찾았다. 마지막 흐름 거래일 뒤 F 날은 흐름이 없다(has_flows False).
12. F 국면 = 10% 지그재그 고점(B10 묶음 첫 바닥 앞) ~ 묶음 마지막 바닥. 어느 하락에도 속하지 않는 F 날은 'none'으로 둔다.
13. B10 15세션 묶음 = 이어진 두 바닥 사이 세션 수 ≤ 15이면 한 묶음, 더 낮은 종가 하나만 남김.
14. 예측 거래일이 없고 묶음에도 안 든 SEIBRO 날짜(관측 empty)의 금액은 버렸다(BRIEF 규칙 밖). 100만 달러 넘는 날짜는 §0 표에 있다(2025-01-13은 2025-01-09 NYSE 휴장 뒤 날짜다).
15. 아는 시각: 묶음 거래일의 값은 묶음 마지막 SEIBRO 날짜 다음 KSD 영업일에 안다(known_settle). s(t) 기준(known_raw)도 함께 남겼다.
16. ipo.json의 GETYQ는 오탐으로 보이지만 지시대로 그대로 썼다(N⁺ 영향 §4).
17. BRIEF 초판 §3 '지금' 행(8/17 고점 559.12 대비 −18.9%)은 가격 자료와 달랐다(2026-11-04 BRIEF 고침). 2026-10-05 종가 589.51이 더 높아 10%·15% 지그재그 모두 마지막 고점이 10/05다(15% 고점 확정 10/16). 사건 집합(B15·B10)은 BRIEF와 같고, 지금 상태는 events.json `current`에 계산값으로 둔다.
18. BRIEF가 정하지 않은 열을 더했다: measures_win n_merged·n_cens·nP·y, measures_daily 보관·시각 열, events.json windows·gain_to_entry_open(가격만).
19. 자료판 검증 중 수치 대조용으로 사건 근처 값 두 개를 봤다(2026-07-29 X4 L5·L15의 P·E, 2026-10-22 측정 표본 행). 정의 선택에는 쓰지 않았다(그 뒤 바꾼 것은 점검·보고 열, 부분 절단 표지의 날짜 세기, 날짜 형 맞춤, 기본이 아닌 민감도 선택지뿐이다).

## 9. 열 사전

### units.parquet

| 열 | 형 | 뜻 |
|---|---|---|
| seibro_date | datetime64[ns] | SEIBRO 결제 날짜(KST 기준 표기 날짜) |
| pred_trades | str | 규약으로 예측한 담긴 미국 거래일(;로 이음) |
| n_pred | int64 | 예측 거래일 수 |
| total_d3 | float64 | 그 날짜 d3(합계 순위) Top50 tot 합(USD, DUMMY 포함 원값) |
| med41 | float64 | 한 거래일 몫 날짜(n_pred = 1) total_d3의 41날짜 가운데 이동 중앙값(최소 10) |
| ratio | float64 | total_d3 ÷ med41 |
| obs | str | 관측 분류: ratio < 0.3 empty, > 1.6 double, 그 밖 normal |
| pred | str | 예측 분류: n_pred 0 empty, 1 normal, ≥2 double |
| match | bool | obs = pred |
| flagged | bool | 묶음 대상(n_pred ≥ 2 또는 obs ≠ pred) |
| reason | str | 표시 이유(predN, obs_x/pred_y) |
| ratio_window_partial | bool | 끝 20날짜: 가운데 창이 잘려 다음 자료로 분류가 바뀔 수 있음 |
| unit_id | int64 | 묶음 번호(날짜 순) |
| unit_first | datetime64[ns] | 묶음 첫 SEIBRO 날짜 |
| unit_last | datetime64[ns] | 묶음 마지막 SEIBRO 날짜 |
| unit_seibro_dates | str | 묶음 SEIBRO 날짜들 |
| unit_trades | str | 묶음에 담긴 미국 거래일들 |
| unit_n_trades | int64 | 묶음 미국 거래일 수(금액을 이 수로 똑같이 나눔) |
| merged | bool | 합친 묶음이면 True |
| edge | bool | 자료 끝이라 앞 또는 뒤 이웃이 없는 묶음 |
| unit_reason | str | 묶음 안 표시 날짜와 이유 |
| nbr_prev_ratio | float64 | 앞 '빈 날 아닌' 날짜의 ratio(진단) |
| nbr_next_ratio | float64 | 뒤 '빈 날 아닌' 날짜의 ratio(진단) |
| known_kst_date | datetime64[ns] | 그 결제 날짜를 아는 날(다음 KSD 영업일) 09:00 KST |

### flows_trade_product.parquet

| 열 | 형 | 뜻 |
|---|---|---|
| date | datetime64[ns] | 미국 거래일 |
| product | str | 상품 키(classes.csv product, 사슬 ISIN 묶음. 티커 없는 규칙 행·새 ISIN은 ISIN) |
| isins | str | 그 묶음에서 나타난 ISIN(;로 이음, 빈칸 = 목록 밖 채움 행) |
| buy | float64 | 매수(USD, 묶음 합 ÷ 거래일 수) |
| sell | float64 | 매도 |
| net | float64 | 순매수 |
| tot | float64 | 매수+매도 |
| buy_d1 | float64 | d1 목록에 든 날의 매수만(ΣB_eq·A2 재료) |
| sell_d2 | float64 | d2 목록에 든 날의 매도만(A3 재료) |
| tot_d3 | float64 | d3 목록에 든 날의 tot만(A1 재료) |
| net_d4 | float64 | d4 목록에 든 날의 net만(참고) |
| in_d1 | bool | 묶음 안 어느 날이든 d1 목록에 들었음 |
| in_d2 | bool | d2 목록 |
| in_d3 | bool | d3 목록 |
| in_d4 | bool | d4 목록 |
| in_any_list | bool | 네 목록 중 하나라도 들었음. False 행은 SOXL·SOXS 채움 행(목록 밖 = 50위보다 작음, 금액 0으로 둠) |
| full_obs | bool | 묶음 안 '빈 날 아닌' 모든 SEIBRO 날짜에 나타났음(False면 일부 날짜 절단) |
| n_in | int64 | 묶음 안 '빈 날 아닌' SEIBRO 날짜 중 나타난 날짜 수 |
| n_data | float64 | 묶음 안 '빈 날 아닌' SEIBRO 날짜 수(예측 거래일 있음 또는 관측 empty 아님) |
| merged | bool | 합친 묶음의 거래일 |
| unit_id | int64 | 묶음 번호(units와 같음) |
| unit_n_trades | int64 | 묶음 거래일 수 |
| ipo_excl | bool | 신규 상장 28일 안(묶음 마지막 거래일 기준) → 분모(N⁺·ΣB_eq·현물 매수)에서 뺌 |
| utype | str | 분류 유형 |
| L | float64 | 배수 |
| sign | int64 | +1 롱, −1 숏, 0 |
| semi | bool | 반도체(classes) |
| single | bool | 단일종목형(classes) |
| equity | bool | 주식(utype이 NONEQ가 아님) |
| lev_long | bool | 주식 & sign = 1 & L ≥ 1.25 |
| EQ3L | bool | lev_long & L ≥ 3 |
| IDX | bool | utype EQ_INDEX |
| EQ_INV | bool | 주식 & sign = −1 |
| spot | bool | 주식 & sign = 0 |
| LSGL | bool | lev_long & single(L3) |
| LIDX | bool | lev_long & IDX(L4) |
| R1 | bool | 1배 반도체(semis_1x.json) |
| R2 | bool | EQ_INDEX & L = 1 & sign = 0 & semi 아님 |
| R3 | bool | BOND·CASH & sign ≥ 0 |
| R4 | bool | EQ_SINGLE & L = 1 & sign = 0 & R1 아님 |
| R5 | bool | utype INCOME |
| dummy | bool | DUMMY ISIN(모든 측정에서 뺌) |

### measures_daily.parquet

| 열 | 형 | 뜻 |
|---|---|---|
| date | datetime64[ns] | 미국 거래일(창 L = 1의 끝) |
| A1 | float64 | d3 Top50 tot 합(DUMMY 제외, 모든 상품) |
| A2 | float64 | 전체 매수 규모: d1 Top50 buy 합 (L = 1 값, measures_win L = 1과 같음) |
| A3 | float64 | 전체 매도 규모: d2 Top50 sell 합 (L = 1 값, measures_win L = 1과 같음) |
| A4 | float64 | 합집합 순매수 합: 네 목록 합집합 전체 net 합(근사) (L = 1 값, measures_win L = 1과 같음) |
| X1 | float64 | SOXL 순매수 (L = 1 값, measures_win L = 1과 같음) |
| X2 | float64 | SOXL 매수 (L = 1 값, measures_win L = 1과 같음) |
| X3 | float64 | SOXL 매도 (L = 1 값, measures_win L = 1과 같음) |
| X4 | float64 | SOXL 순매수 비중 n(SOXL) ÷ N⁺ (L = 1 값, measures_win L = 1과 같음) |
| X5 | float64 | SOXL 거래 몫 t(SOXL) ÷ A1 (L = 1 값, measures_win L = 1과 같음) |
| X6 | float64 | SOXL 매도 ÷ 매수 (L = 1 값, measures_win L = 1과 같음) |
| H1 | float64 | SOXS 순매수 (L = 1 값, measures_win L = 1과 같음) |
| H2 | float64 | SOXL 강세 몫 b(SOXL) ÷ (b(SOXL) + b(SOXS)) (L = 1 값, measures_win L = 1과 같음) |
| H3 | float64 | 주식 인버스 매수 몫 Σb(EQ_INV) ÷ ΣB_eq (L = 1 값, measures_win L = 1과 같음) |
| H4 | float64 | 주식 인버스 순매수 Σn(EQ_INV) (L = 1 값, measures_win L = 1과 같음) |
| L1 | float64 | 3배 롱 순매수 Σn(EQ3L) (L = 1 값, measures_win L = 1과 같음) |
| L2 | float64 | 3배 롱 순매수 비중 Σn(EQ3L) ÷ N⁺ (L = 1 값, measures_win L = 1과 같음) |
| L3 | float64 | 단일종목 레버리지 롱 순매수 Σn(lev_long & single) (L = 1 값, measures_win L = 1과 같음) |
| L4 | float64 | 지수형 레버리지 롱 매수 ÷ 현물 주식 매수 (L = 1 값, measures_win L = 1과 같음) |
| R1 | float64 | 1배 반도체 순매수(semis_1x.json) (L = 1 값, measures_win L = 1과 같음) |
| R2 | float64 | 1배 지수 ETF 순매수(반도체 제외) (L = 1 값, measures_win L = 1과 같음) |
| R3 | float64 | 채권·현금 순매수(BOND·CASH, sign ≥ 0) (L = 1 값, measures_win L = 1과 같음) |
| R4 | float64 | 그 밖 1배 개별주 순매수(EQ_SINGLE, R1 제외, 분류 밖 포함) (L = 1 값, measures_win L = 1과 같음) |
| R5 | float64 | 인컴(커버드콜) 순매수(utype INCOME) (L = 1 값, measures_win L = 1과 같음) |
| C1 | float64 | SOXL 보유 좌수 변화 Δlog(보관 ÷ SOXL 종가), 창 시작 전 → 창 끝 (L = 1 값, measures_win L = 1과 같음) |
| C2 | float64 | SOXL 보관 비중 = 창 끝 SOXL 보관 ÷ 보관 Top50 합 (L = 1 값, measures_win L = 1과 같음) |
| C3 | float64 | 보관 Top50 합 변화 Δlog, 창 시작 전 → 창 끝 (L = 1 값, measures_win L = 1과 같음) |
| SOXL_net | float64 | SOXL 순매수(목록 밖 = 0) |
| SOXL_buy | float64 | SOXL 매수 |
| SOXL_sell | float64 | SOXL 매도 |
| SOXL_tot | float64 | SOXL 매수+매도 |
| cens_SOXL | bool | SOXL이 묶음 안 네 목록 어디에도 없음(절단, 금액 0으로 둠) |
| part_SOXL | bool | 묶음 일부 날짜만 나타남 |
| SOXS_net | float64 | SOXS 순매수(사슬 ISIN 합) |
| SOXS_buy | float64 | SOXS 매수 |
| SOXS_sell | float64 | SOXS 매도 |
| SOXS_tot | float64 | SOXS 매수+매도 |
| cens_SOXS | bool | SOXS 절단 |
| part_SOXS | bool | SOXS 일부 날짜 절단 |
| SOXLS_buy | float64 | SOXL_buy + SOXS_buy(H2 분모) |
| N_plus | float64 | N⁺: 주식(NONEQ·DUMMY·신규 상장 28일 제외) 상품별 순매수 양수 부분 합 |
| N_plus_ipo | float64 | 신규 상장 제외분의 순매수 양수 합(분모 대안 재료) |
| B_eq | float64 | ΣB_eq: d1 목록 주식 매수 합(신규 상장 제외) |
| B_eq_ipo | float64 | 신규 상장 제외분의 d1 매수 |
| SPOT_buy | float64 | 현물 주식(sign 0, 신규 상장 제외) 매수 합(L4 분모, 목록 무관 합집합) |
| INV_buy | float64 | EQ_INV 매수 |
| INV_net | float64 | EQ_INV 순매수 |
| EQ3L_net | float64 | EQ3L 순매수 |
| LSGL_net | float64 | lev_long & single 순매수 |
| LIDX_buy | float64 | lev_long & IDX 매수 |
| SEMI1_net | float64 | R1 순매수 |
| IDX1_net | float64 | R2 순매수 |
| BC_net | float64 | R3 순매수 |
| SGL1_net | float64 | R4 순매수 |
| INC_net | float64 | R5 순매수 |
| n_INV | int64 | 목록에 든 INV 상품 수 |
| zero_INV | bool | INV 상품이 하나도 목록에 없음(전부 절단 → 값 0) |
| n_EQ3L | int64 | 목록에 든 EQ3L 상품 수 |
| zero_EQ3L | bool | EQ3L 상품이 하나도 목록에 없음(전부 절단 → 값 0) |
| n_LSGL | int64 | 목록에 든 LSGL 상품 수 |
| zero_LSGL | bool | LSGL 상품이 하나도 목록에 없음(전부 절단 → 값 0) |
| n_LIDX | int64 | 목록에 든 LIDX 상품 수 |
| zero_LIDX | bool | LIDX 상품이 하나도 목록에 없음(전부 절단 → 값 0) |
| n_SEMI1 | int64 | 목록에 든 SEMI1 상품 수 |
| zero_SEMI1 | bool | SEMI1 상품이 하나도 목록에 없음(전부 절단 → 값 0) |
| n_IDX1 | int64 | 목록에 든 IDX1 상품 수 |
| zero_IDX1 | bool | IDX1 상품이 하나도 목록에 없음(전부 절단 → 값 0) |
| n_BC | int64 | 목록에 든 BC 상품 수 |
| zero_BC | bool | BC 상품이 하나도 목록에 없음(전부 절단 → 값 0) |
| n_SGL1 | int64 | 목록에 든 SGL1 상품 수 |
| zero_SGL1 | bool | SGL1 상품이 하나도 목록에 없음(전부 절단 → 값 0) |
| n_INC | int64 | 목록에 든 INC 상품 수 |
| zero_INC | bool | INC 상품이 하나도 목록에 없음(전부 절단 → 값 0) |
| n_products | int64 | 목록에 든 상품 수 |
| seibro_date | datetime64[ns] | 예측 SEIBRO 결제 날짜 s(t) |
| unit_id | int64 | 묶음 번호 |
| unit_n_trades | int64 | 묶음 거래일 수 |
| merged | bool | 합친 묶음의 거래일 |
| edge | bool | 자료 끝 묶음 |
| unit_first | datetime64[ns] | 묶음 첫 SEIBRO 날짜 |
| unit_last | datetime64[ns] | 묶음 마지막 SEIBRO 날짜 |
| known_raw | datetime64[ns] | s(t) 다음 KSD 영업일(09:00 KST에 앎) |
| known_settle | datetime64[ns] | 묶음 마지막 SEIBRO 날짜 다음 KSD 영업일(묶음 값을 아는 날, t로 끝나는 모든 창에 적용) |
| entry_settle | datetime64[ns] | known_settle 날짜 이후 첫 NYSE 세션(시가 진입) |
| c_date | datetime64[ns] | t에 붙인 보관 날짜(대응 거래일 ≤ t인 마지막 보관) |
| c_trade | datetime64[ns] | 그 보관의 대응 거래일 |
| c_stale | bool | c_trade ≠ t(대응 보관 없음, 앞 값 사용) |
| c_known | datetime64[ns] | 그 보관을 아는 날(둘째 KSD 영업일) |
| cust_SOXL_amt | float64 | c_date SOXL 보관 금액 |
| cust_top50 | float64 | c_date 보관 Top50 합 |
| cust_SOXL_shares | float64 | c_date SOXL 좌수 |
| cens_cust_SOXL | bool | c_date 보관 Top50에 SOXL 없음 |

### measures_win.parquet

| 열 | 형 | 뜻 |
|---|---|---|
| date | datetime64[ns] | 창 끝 미국 거래일 t |
| L | int64 | 창 길이(세션) |
| measure | str | 측정 코드(MEAS) |
| value | float64 | 창 값: 금액 측정은 L일 합, 비율은 창 합의 비, C1·C3은 t−L → t 보관 Δlog, C2는 t 값 |
| P | float64 | 백분위: 100 × (작은 수 + 0.5 × 같은 수) ÷ n, 같은 L 계열 지난 250세션(t 포함), n ≥ 120 |
| nP | int64 | P에 쓴 n(NaN 아닌 값 수) |
| y | float64 | 정규 점수 Φ⁻¹(clip(P/100, 0.5/n, 1 − 0.5/n)) |
| E_full | float64 | 메아리 잔차(전체 표본 OLS, 잔차 ÷ √(SSR/(n − rank))) |
| E_exp | float64 | 메아리 잔차(창 끝 < t인 행만 적합, 최소 250행, 그 적합의 잔차 표준편차로 나눔) |
| n_merged | float64 | 창 안 합친 묶음 거래일 수(0 ~ L) |
| n_cens | float64 | 창 안 절단 거래일 수(측정별 기준은 PANEL §6, C 측정은 보관 SOXL 부재 날 수) |

### price_feats.parquet

| 열 | 형 | 뜻 |
|---|---|---|
| date | datetime64[ns] | NYSE 세션 |
| SOXX_Open | float64 | SOXX 시가 |
| SOXX_Close | float64 | SOXX 종가 |
| lr1 | float64 | SOXX 하루 로그 수익 |
| ret_1 | float64 | log(C_t ÷ C_t−1) |
| pre20_1 | float64 | 창(L=1) 시작 전 20세션 로그 수익 log(C_t−1 ÷ C_t−21) |
| ret_5 | float64 | log(C_t ÷ C_t−5) |
| pre20_5 | float64 | log(C_t−5 ÷ C_t−25) |
| ret_15 | float64 | log(C_t ÷ C_t−15) |
| pre20_15 | float64 | log(C_t−15 ÷ C_t−35) |
| ret_20 | float64 | log(C_t ÷ C_t−20) |
| max60 | float64 | 60세션(그날 포함) 최고 종가 |
| dd60 | float64 | C_t ÷ max60 − 1 |
| rv20 | float64 | 20세션 로그 수익 표준편차(ddof 1, 연율화 안 함) |
| min10 | float64 | 10세션(그날 포함) 최저 종가 |
| is_10d_low | bool | C_t = min10 |
| SOXL_Close | float64 | SOXL 종가 |
| regime | int64 | 체제 번호(0: < 2024-05-28, 1: T+1, 2: ≥ 2025-12-15 B0, 3: ≥ 2026-05-22 B1, 4: ≥ 2026-07-31 B2) |

### custody_daily.parquet

| 열 | 형 | 뜻 |
|---|---|---|
| custody_date | datetime64[ns] | 보관 날짜 D |
| product | str | 상품 키 |
| amt | float64 | 보관 금액(USD, Top50, 사슬 합) |
| rank | int64 | 보관 순위(사슬이면 최소) |
| isins | str | ISIN |
| top50_total | float64 | 그날 보관 Top50 합(DUMMY 제외) |
| n_list | int64 | 그날 목록 줄 수 |
| trade_date | datetime64[ns] | 이 보관이 담는 마지막 미국 거래일(지연 실측) |
| lag_used | int64 | 쓴 지연(D 이하 마지막 NYSE 세션에서 몇 세션 앞) |
| val_date | datetime64[ns] | 평가 종가 날짜(D 이하 마지막 NYSE 세션) |
| SOXL_close | float64 | val_date SOXL 종가(분할 조정, 배당 미조정) |
| known_kst_date | datetime64[ns] | 아는 날(둘째 KSD 영업일) 09:00 KST |
| SOXL_shares | float64 | SOXL 행만: amt ÷ SOXL_close(좌수) |

### prices.csv

long 형: date, ticker(SOXX·SOXL·SOXS·^SOX·QQQ·SPY·EWY·NVDA), Open, High, Low, Close(분할 조정), AdjClose(분할+배당), Volume. yfinance auto_adjust=False, 2022-06-01 ~ 가격 끝.

### events.json

`B15`, `B10`(날짜), `B10_events`(바닥마다: close, B15, cluster, confirmed10, top10·top10_close·dd10, B15면 top15·dd15·confirmed15, seibro_date, unit_id, merged, unit_seibro_dates, unit_trades, known_kst_raw·entry_raw·entry_raw_k, known_kst_unit·entry_unit·entry_unit_k, gain_to_entry_open_*, merged_k(k ∈ [−20, 20] 중 합친 날), merged_by_window, k_without_flows), `episodes`(국면: 10% 고점 → 묶음 마지막 바닥, F 목록), `F`(date, close, dd60, fut20_min_ret, episode, has_flows), `F_undetermined`, `current`(10%·15% 지그재그 상태와 최신 종가), `zigzag`(원 지그재그).
