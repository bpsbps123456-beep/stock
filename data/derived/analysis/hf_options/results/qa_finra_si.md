# QA — finra_si (FINRA 보름 단위 공매도 잔고)

수집 2026-10-14 · 코드 `code/fetch_finra_si.py` · 결과 `cache/finra_si.parquet`(묶음), `cache/finra_si_by_symbol.parquet`(종목 긴 표)

## 범위

| 항목 | 값 |
|---|---|
| 출처 | FINRA Query API `otcMarket/consolidatedShortInterest`(인증 없음, POST 필터) |
| 종목 | AI 미국 115 + ETF 7(QQQ SOXX SMH TQQQ SQQQ SOXL SOXS) = 122, 전부 받음 |
| 기준일(asof) | 2023-01-13 ~ 2026-09-30, 90개(매달 15일·월말) |
| known | 2023-01-26 ~ 2026-10-12, 90행. 2025년 24행, 2026년 19행 |
| 행 수 | 종목 표 10,364행 |
| 아직 없음 | 2026-10-15 기준일(공개 예정 2026-10-26) |

2023년은 2024년 백분위 창(250거래일 ≈ 기준일 24개)을 채우는 용도다. 보름 단위라 창 하나에 관측이 24개뿐이어서 백분위가 거칠다.

## 공개 시점(known) 규칙

| 단계 | 규칙 | 근거 |
|---|---|---|
| 공개일 | 기준일 뒤 7번째 영업일 | finra.org/finra-data/browse-catalog/equity-short-interest: "provided for publication on the 7th business day after the reporting settlement date" |
| 공식 표 | 2025-11-14 ~ 2026-12-31 기준일은 FINRA 공식 공개일 표를 코드에 넣어 씀 | finra.org/filing-reporting/regulatory-filing-systems/short-interest (2026-10-14 열람) |
| 검증 | 7영업일 규칙으로 계산한 날짜가 받은 기간 공식 표 22개 기준일과 전부 같음 | 실행 출력 "불일치: 없음" |
| 2025-11 전 | 공식 표를 구하지 못해 규칙으로 계산. 영업일 = NYSE 거래일에서 콜럼버스·재향군인의 날까지 뺀 날(보수적) | 2024-10-31·2025-10-31 기준일은 11-11을 빼서 공개일이 11-12. 실제 11-11 공개였다면 하루 늦게 잡은 것 |
| 특수 휴장 | 2025-01-09(카터 애도 휴장)은 QQQ 달력에 없어서 자동으로 빠짐 → 2024-12-31 기준일 공개일 2025-01-13 | |
| known | 공개일 **다음 거래일**. 공개 시각이 적혀 있지 않아 장 마감 뒤로 본다 | 과제의 대안 규칙 "기준일+8거래일"과 같은 날이 나온다 |

기준일에서 known까지 달력일 10~14일. 진입은 known 다음 거래일 시가.

## 칸

`finra_si.parquet`(index `known`, `asof` = 기준일 문자열, 나머지 float)

| 칸 | 뜻 |
|---|---|
| si_ai_n | 묶음에 들어간 AI 종목 수(102 → 115) |
| si_ai_shares | 그 종목들의 공매도 주식 수 합(FINRA 원래 단위) |
| si_ai_chg | 같은 종목 합 기준 직전 기준일 대비 변화율(분할 맞춤) |
| si_ai_chg_median | 종목별 변화율 중앙값 |
| si_ai_up_share | 공매도가 늘어난 종목 비율 |
| si_ai_dtc_median / si_ai_dtc_agg | 커버 일수 중앙값 / 합계 공매도÷합계 거래량 |
| si_ai_idx | si_ai_chg를 이어 붙인 지수(첫 행 100). 수준 백분위는 이 칸을 쓴다 |
| si_<etf> / si_<etf>_chg / si_<etf>_dtc | ETF 공매도 주식 수 / 분할 맞춘 변화율 / 커버 일수 |

종목 표 칸: known, asof, published, symbol, issue_name, market, short_shares, prev_short_shares, avg_daily_volume, days_to_cover(FINRA 원값), split_flag, revision_flag, split_factor, prev_short_adj, chg, dtc_raw.

## 손본 것과 남은 한계

| 항목 | 처리 |
|---|---|
| 분할 | FINRA 직전 수량은 분할 전 단위 그대로다(NVDA 2024-06-14 직전 2,920만 → 현재 3억 1,500만). yfinance 분할 배수로 분모를 맞춤. 분사(FLEX 1.327, WDC 1.323)는 깔끔한 비율이 아니라 뺌. ANET 2024-12 4:1은 FINRA 표시가 없지만 yfinance로 잡음 |
| 수준 계단 | si_ai_shares와 si_<etf> 수준은 분할 날 계단이 남는다(NVDA·AVGO·SMCI·LRCX·KLAC ×10, SOXS·SQQQ 병합 등). 수준 백분위에는 si_ai_idx나 _chg를 쓴다 |
| 신규 상장 | 상장 뒤 첫 두 기준일은 묶음에서 뺌(ARM 2023-09-29 직전 400주 → 1,383만 주가 합계 변화율을 0.9%p 부풀림) |
| 늦게 들어온 종목 | ARM 2023-09, ALAB 2024-03, GEV 2024-04, OKLO 2024-05, TEM 2024-06, TLN 2024-07, NBIS 2024-08, SNDK 2025-02, CRWV 2025-03, GLXY 2025-05, P(Everpure)·XE(X-Energy) 2026-04, CBRS 2026-05 |
| 선택 편향 | 종목 목록은 2026년 워치리스트(사후에 고른 AI 종목)다. 과거 묶음은 "나중에 AI 대장이 된 종목"이라 그때 시장이 본 묶음과 다르다 |
| 이름 변경 | CIFR(→Cipher Digital), IREN, COHR, FSLY, ROP는 이름만 바뀜. HUT는 2023-12 합병으로 새 증권(직전 0주라 그 기준일은 빠짐) |
| 정정 | revisionFlag R 158행(1.5%, 2023-05-31·11-15·2024-04-30에 몰림). API는 정정 반영 최신 값이라 당시 처음 공개된 값과 조금 다를 수 있다 |
| 큰 변화 | CMI 2024-03 +457%(Atmus 교환 공개매수 차익), CRM 2026-03-31 +240%는 진짜 값으로 둠. 합계 칸에는 영향, 중앙값 칸에는 영향이 작다 |
| 커버 일수 | FINRA는 1 미만을 1.00으로, 거래량 0을 999.99로 적는다(1.00인 행 10.9%). 묶음 칸은 직접 나눈 dtc_raw를 씀 |
| 해석 | 공매도 잔고에는 헤지펀드 말고도 시장조성자 헤지와 ETF 차익거래 몫이 섞여 있다. ETF 공매도 주식 수는 설정·환매로 발행 주식이 바뀌어 수준 비교가 약하다 |

## 수집 기록

- FINRA API 47회(한 달에 1회, 1.5초 간격), yfinance 분할 122회 + QQQ 달력(1.5초 간격). 429 없음.
- archive.org(Wayback)에서 2024~2025 공식 표를 찾으려다 **첫 응답부터 HTTP 429**. 반복문이 상태 확인 없이 1.5초 간격으로 3번 더 보낸 뒤 그 호스트는 더 쓰지 않았다(같은 IP를 쓰는 다른 수집 작업 때문으로 보임).
- nasdaqtrader.com·finra.org 페이지는 curl로는 봇 차단(Incapsula·Cloudflare). WebFetch로 FINRA 일정 표와 설명을 읽음. WebSearch 1회.
- 이어 받기: `cache/raw_finra_si/YYYY-MM.json`(기준일 2개가 다 찬 달은 건너뜀), `_splits.json`(받은 뒤 FINRA 분할 표시가 새로 생긴 종목만 다시 받음).

## 검증(verification, 2026-10-14, 독립 검증자)

판정: **통과**. 코드·parquet 수정 없음. 아래 '정정' 문장만 바로잡는다(원 QA보다 시점 문제가 작다).

| 항목 | 방법 | 결과 |
|---|---|---|
| 원값 대조 | 수집 코드를 쓰지 않고 FINRA API에 직접 POST(compareFilters 방식, 2초 간격 5회). 22개 관측: NVDA 2025-06-13·06-30, AMD·SOXX·SQQQ 2025-11-14, CRM·MU·SOXS 2026-03-31, AAPL·PLTR·QQQ·SOXL 2026-09-30, NVDA·SMH·TQQQ 2024-06-14 | 현재·직전 수량, 평균 거래량, 커버 일수 4칸 모두 **완전 일치**. 묶음 표 ETF 칸 7개도 일치 |
| 공식 공개일 | finra.org short-interest 일정 표(WebFetch) | 2025-11-14~2026-12-31 기준일 28개 모두 코드 OFFICIAL_PUB와 일치. 성금요일(2026-04-03)·준틴스(06-19)를 영업일에서 뺀다 → NYSE 휴장일 제외 규칙과 맞음 |
| 규칙 구간(2025-11 전) | cdn.finra.org `equity/otcmarket/biweekly/shrtYYYYMMDD.csv` 90개 HEAD(1.5초 간격)로 Last-Modified 확인 | 파일은 공개일 뒤 첫 월요일 06:00 ET에 묶어서 올라온다. 계산 공개일이 파일 게시일보다 늦은 경우 0건. 계산 공개일이 월요일인 규칙 구간 6건(2023-11-27, 12-11, 2024-03-11, 08-26, 2025-01-27, 08-11)은 모두 그날 아침 묶음에 없고 다음 월요일에 들어감 → 실제 공개가 계산일보다 이르지 않고 그날 저녁 공개와 맞음. 금요일 계산일은 바로 다음 월요일 묶음에 있음 → 늦지도 않음 |
| 예외 1건 | 2024-12-31 기준일 | 파일이 계산 공개일(2025-01-13 월) 06:00 ET에 이미 있음 → 실제 공개는 01-10(금)로 보이고 코드는 하루 늦게(보수적) 잡는다. 미리보기 오류 아님 |
| known | QQQ 비결측 거래일 기준 | known = 기준일+8거래일, 예외는 재향군인의 날이 낀 2024-10-31·2025-10-31(하루 늦음) 2건뿐. 기준일→known 달력일 10~14일. known 90개 모두 거래일 |
| 정정(revisionFlag) | R 158행에서 이번 prev와 직전 기준일 current 비교 | 157행이 다르다 → API의 current는 **처음 공개된 값 그대로**이고 정정분은 다음 기준일의 prev로만 나온다(메타데이터: R = "prior cycle was revised"). 따라서 short_shares 수준에는 사후 정정이 섞이지 않고, chg는 같은 날 공개된 정정 prev를 쓰므로 시점 문제 없음. 위 '정정' 행의 "정정 반영 최신 값이라 처음 공개값과 다를 수 있다"는 해당 없음 |
| 분할 | FINRA S 표시 18행 vs yfinance 배수 | 18행 모두 맞음. API 메타데이터가 직전 수량을 "Previous Unadjusted"로 명시 → 분모 맞춤이 옳다. 불일치는 ANET 2024-12-13(표시 없음, 실제 4:1, 맞춤 후 +22%)·HUT(합병, 직전 0이라 제외) 2건뿐 |
| TQQQ 2025-11-28 | 2:1 분할 기준일 −49.7% | FINRA 원값(직전 1,757만 → 현재 1,768만, S 표시). 분할 뒤 수준(1,288만~2,274만주)이 분할 전과 비슷해 주식 수 환산으로 절반이 된 상태가 이어진다. 오류 증거 없음, 해석 주의 |
| 형식 | 색인·칸·값 | index `known` 문자열·정렬·고유 90행, 칸 30개(asof + float 29개). 음수·NaN 없음(첫 행 묶음 칸만 NaN). 122종목 모두, 중간에 빠진 기준일 없음 |
| 묶음 계산 | 4개 기준일(2025-03-14, 2025-11-28, 2026-04-15, 2026-09-30) 독립 재계산 | si_ai_n·si_ai_chg·si_ai_chg_median·si_ai_dtc_median 모두 일치 |
| 재실행 | `py fetch_finra_si.py` | 이어 받기로 4.8초, 두 parquet이 실행 전과 같음(DataFrame.equals). 2026-10은 'partial'(10-15 기준일 공개 10-26 예정)이라 실행마다 1회 다시 요청 |
| 요청 예절 | FINRA API 5회, cdn.finra.org 94회 | 429 없음. archive.org는 쓰지 않음 |

남은 주의
- `_splits.json`은 FINRA S 표시가 새로 생긴 종목만 다시 받는다. ANET처럼 표시 없는 미래 분할은 다음 실행에서 놓칠 수 있다(지금 자료는 2026-10-14에 122종목을 모두 받아 영향 없음).
- `options_sentiment/cache/yahoo.parquet` 색인에는 휴장일 행(QQQ NaN, 예: 2026-05-25·09-07)이 있다. 거래일 달력으로 쓸 때는 QQQ 비결측 행만 쓴다.
- CDN 내려받기 파일만 보는 사람에게는 공개일 뒤 첫 월요일에야 보이지만, 공식 공개(FINRA 조회 화면·API·거래소)는 공개일 당일이라 known = 공개일 다음 거래일이 맞다.
