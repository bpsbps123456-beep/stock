# sec13f QA: 헤지펀드 13F 분기 보유 (2026-10-14)

## 결론: 자료를 받지 못했다
- SEC가 정해진 User-Agent `GBRC-research personal-noncommercial`(이메일 없음)를 **HTTP 403**으로 거절했다. 차단 페이지 제목은 "Request Rate Threshold Exceeded"이고, 본문은 "Automated access ... must comply with SEC.gov's Privacy and Security Policy"다.
- 시도는 2회다. 08:56에 curl로 1회, 09:07에 Python urllib로 1회 보냈다. 두 번째에는 SEC가 권장하는 `Accept-Encoding: gzip, deflate`를 넣었다. 둘 다 랜딩 페이지 요청이었고 11분 간격이었다. 규칙대로 거기서 멈췄다. UA를 바꾸거나 제3자 사이트로 우회하지 않았다.
- 그래서 `cache/sec13f.parquet`과 `cache/sec13f_by_symbol.parquet`은 **없다**. 만든 숫자는 0개다. 실제 펀드 이름·CIK 확인과 실제 제출일 중앙값 계산도 하지 못했다.
- 원인 추정: SEC 공정 접근 지침은 UA에 "이름 + 연락 이메일"을 요구한다. 이메일이 없는 UA는 자동화 도구로 막는 것으로 보인다. 요청이 2회뿐이었으므로 속도 초과는 아니다. 같은 IP의 다른 작업이 sec.gov를 많이 불렀을 가능성은 완전히 배제하지 못한다. 다만 SEC 속도 차단은 보통 10분이고, 11분 뒤에도 같은 결과가 나왔다.
- 증거: `cache/raw_sec13f/fetch.log`, `cache/raw_sec13f/landing_403.html`.
- 다시 받으려면 이메일을 넣은 UA로 SEC에 접근할지 사용자가 정해야 한다. 이 에이전트는 개인 이메일을 UA에 넣지 않는다. 바꿀 곳은 코드의 `UA` 상수 한 줄이다.

## 코드 상태 (`code/fetch_sec13f.py`)
- 한 명령으로 전부 수행한다: 랜딩 페이지에서 zip 링크 수집, 제출 구간 2023-07~최신의 zip 약 13개(각 50~150MB) 내려받기, 필요한 열만 parquet으로 저장한 뒤 zip 삭제, 계산. 이어 받기가 되고, `--offline`은 받아 둔 것만으로 다시 계산한다.
- 오프라인 점검을 통과했다. 실제 tsv 열 구성으로 만든 가짜 zip과 가짜 가격으로 전 과정을 돌렸다. 아래 경우를 넣었고 결과가 손계산과 맞았다.

  | 점검한 경우 | 기대 동작 |
  |---|---|
  | NVDA 10:1 분할 | 직전 분기 주식 수를 분할 기준으로 맞춘다 |
  | GOOG(클래스 C) | GOOGL에 합친다 |
  | 마감 뒤 원본 제출 | 뺀다 |
  | 마감 뒤 수정본 | 무시한다 |
  | RESTATEMENT 수정본 | 원본을 대체한다 |
  | Citadel Securities·Vanguard | 뺀다 |
  | Two Sigma의 CIK 2개 | 한 펀드로 합친다 |

- 실제 자료로는 검증하지 못했다. 열 이름(ACCESSION_NUMBER·FILINGMANAGER_NAME·PUTCALL 등)과 날짜 형식(`30-JUN-2024`)은 SEC 13F 자료 설명서를 기억해서 옮긴 것이다. 형식이 다르면 열 검사에서 명확한 오류를 내고 멈추게 했다.

## 시점 규칙 (known)
- 근거: Rule 13f-1. 13F는 분기 말 뒤 45일 안에 내야 한다. 마감일이 주말이나 연방 휴일이면 다음 영업일이 마감이다.
- 마감일 이하에 제출된 원본과 수정본만 쓴다.
- EDGAR는 오후 5시 30분(ET)까지 접수한 것을 그날 제출로 친다(Regulation S-T 13(a)). 그래서 마감일 날짜로 찍힌 제출은 4시 종가 뒤에 공개될 수 있다.
- **known은 마감일 다음 뉴욕 거래일이다**(검증에서 한 칸 늦춤). 그날 4시 종가 전에는 쓰는 제출이 모두 공개돼 있다.

| 분기 말 | 마감일 | known | 비고 |
|---|---|---|---|
| 2023-06-30 | 2023-08-14 | 2023-08-15 | 순매수 기준점 |
| 2023-09-30 | 2023-11-14 | 2023-11-15 | |
| 2023-12-31 | 2024-02-14 | 2024-02-15 | |
| 2024-03-31 | 2024-05-15 | 2024-05-16 | |
| 2024-06-30 | 2024-08-14 | 2024-08-15 | |
| 2024-09-30 | 2024-11-14 | 2024-11-15 | |
| 2024-12-31 | 2025-02-14 | 2025-02-18 | 2/17 Presidents' Day 휴장 |
| 2025-03-31 | 2025-05-15 | 2025-05-16 | |
| 2025-06-30 | 2025-08-14 | 2025-08-15 | |
| 2025-09-30 | 2025-11-14 | 2025-11-17 | 셧다운(10/1~11/12)은 마감 전에 끝남 |
| 2025-12-31 | 2026-02-17 | 2026-02-18 | 2/14 토요일, 2/16 Presidents' Day |
| 2026-03-31 | 2026-05-15 | 2026-05-18 | |
| 2026-06-30 | 2026-08-14 | 2026-08-17 | 최신 |
| 2026-09-30 | 2026-11-16 | 2026-11-17 | 아직 마감 전(11/14 토요일) |

- 실제 제출일 중앙값은 자료가 없어 재지 못했다. 코드는 이 값을 `cache/sec13f_filing_dates.csv`에 기록한다. 기록 항목은 중앙값, 마감일 당일 제출 비율, 늦은 원본·수정본 수다.

## 설계 (자료가 열리면 나올 값)
- **헤지펀드**: 76개 운용사 이름 정규식(`HF`)이다. 분류는 4가지다.
  - multi 12개: Citadel Advisors, Millennium, Point72, Balyasny 등
  - quant 12개: Two Sigma, Renaissance, D. E. Shaw, AQR, Bridgewater 등
  - growth 31개: Tiger Global, Coatue, Viking, Lone Pine, Altimeter, D1 등
  - macro_value 21개: Pershing, Elliott, Soros, Duquesne, Tudor 등

  이름에 `SECURITIES`·`BANK`·`BROKER`가 들어간 법인은 뺀다. 해석된 CIK와 정확한 이름은 `cache/sec13f_funds.csv`에 남는다.
- **AI 종목**: 115개 티커의 발행사 이름 정규식(`ISSUER`)으로 CUSIP을 찾는다. 13F 평가액 ÷ 주식 수를 Yahoo 분기 말 원래 종가(분할 전)와 비교해 중앙값이 0.8~1.25인 CUSIP만 쓴다. 이 검사로 워런트와 오매칭이 걸러진다.
  - 행 하나의 가격 비율이 0.2~5 밖이면 큰 기재 오류로 보고 뺀다.
  - 분기 말에 미국에서 거래되지 않던 종목(상장 전·거래 정지, 예: 2024년 10월 이전 Yandex/NBIS)도 뺀다.
  - 결과는 `cache/sec13f_cusip_map.csv`에 남는다.
- **주식 행과 옵션 행**: PUTCALL이 비어 있고 SSHPRNAMTTYPE이 SH면 보유 주식이다. Put·Call 행은 기초자산 명목 가치다.
- **분기 열**:

  | 묶음 | 열 |
  |---|---|
  | 사양 열 | `hf_ai_value`, `hf_ai_n_pos`, `hf_ai_shares_chg`, `hf_ai_call_value`, `hf_ai_put_value`, `hf_ai_putcall`, `hf_ai_n_funds_increasing`, `hf_ai_n_funds_decreasing` |
  | 추가 열 | `hf_ai_weight`(AI 비중 = 13F 주식 전체 대비), `hf_ai_shares_chg_pct`(순매수 ÷ 직전 보유 가치), `hf_n_funds_filed`, `hf_n_funds_both`, `hf_total_long_value`, `hf_ai_n_funds` |
  | 분류별 열 | `hf_ai_{value,shares_chg,call_value,put_value}_{multi,quant,growth,macro_value}` |

  순매수(`shares_chg`)는 두 분기 모두 제출한 펀드만 계산한다. 식은 (이번 주식 수 − 지난 주식 수 × 분할 비율) × 이번 분기 말 종가다.
- **종목별 표**: `known, asof, symbol, hf_value, hf_shares, n_funds, hf_shares_chg, hf_call_value, hf_put_value, n_funds_increasing, n_funds_decreasing, price`.
- 제출 펀드 수가 분기 중앙값의 80%에 못 미치는 분기는 뺀다. 자료 집합이 그 분기 제출을 아직 다 담지 않은 경우를 거르기 위해서다.

## 해석 주의 (자료를 받은 뒤에도 남는 한계)
- 13F에는 롱 주식과 상장 옵션만 나온다. 공매도, 스와프, 해외 주식은 보이지 않는다. 그래서 multi·quant 펀드의 숫자는 순노출이 아니다.
- 옵션 명목은 헤지와 방향성 베팅을 구분하지 못한다. Citadel Advisors의 옵션 수치는 특히 크다.
- 정보는 분기 말 시점이고 45일 늦게 나온다. 데스크 정본(`docs/desk_system.md`)이 13F를 저점 이후 신호로 쓰지 않은 이유와 같다.
- 마감 뒤에 공개되는 기밀 처리분(NEW HOLDINGS 수정본)은 버린다.

## 검증 (2026-10-14, 독립 검증자)

**판정: 실패(자료 없음). 수집자 보고는 정직하다.** 코드의 시점 규칙 1건과 빈 분기 처리 2건을 고쳤다.

| 점검 | 결과 |
|---|---|
| 원자료 대조 | 할 수 없음. 검증자도 같은 UA로 랜딩 페이지를 1회(09:09:54 KST) 요청했고 HTTP 403 "Request Rate Threshold Exceeded"를 받았다. sec.gov를 부른 다른 묶음은 없다(다른 묶음 코드 확인). 14분 동안 요청이 3회뿐이니 속도 초과가 아니라 UA 차단이다. SEC 공정 접근 지침은 UA에 "회사 이름 + 연락 이메일"을 요구한다. UA 변경·제3자 출처 우회는 하지 않았다. |
| 캐시 parquet | `sec13f.parquet`, `sec13f_by_symbol.parquet` 없음 확인. 지어낸 숫자 0개. |
| 시점 규칙 | 마감일 계산(45일, 주말·연방 휴일 이월)은 14개 분기 모두 맞다. 다만 known = 마감일은 연구 정의("그날 4시 종가까지 공개")를 어긴다. 마감일 4시~5시 30분 접수분이 섞이기 때문이다. 큰 운용사는 마감일 장 마감 뒤에 내는 일이 흔하다. **known을 마감일 다음 거래일로 고쳤다**(`to_trading_day`, 위 표 갱신). 쓰는 제출은 모두 FILING_DATE ≤ 마감일 < known이라 미래 정보가 들어갈 수 없다. EDGAR가 갑자기 닫혀 마감이 미뤄지면, 그날 낸 제출은 늦은 것으로 버려진다(보수 쪽). |
| 2025 셧다운 | 13F는 EDGAR 제출 즉시 공개되고 데이터셋 zip 게시일과 무관하다. 2025Q3 마감(11/14)은 셧다운 종료(11/12) 뒤다. 규칙에 바꿀 것 없음. |
| 빈 분기 | 고침: 제출 0건 분기(최신 zip 미게시 등)가 0으로 저장될 수 있었다. 이제 빼고, 80% 기준은 제출이 있는 분기의 중앙값으로 잰다. 직전 분기 자료가 없을 때 순매수가 0이 되던 것도 빈칸(NaN)으로 바꿨다. |
| 가격 캐시 | 고침: `yahoo_close.parquet`이 한 번 생기면 다시 받지 않아, 몇 달 뒤 재실행하면 새 분기의 known·분기 말 가격이 빠졌다. 이제 마지막 날짜가 4일보다 오래됐으면 전부 다시 받는다. |
| 응답 인코딩 | 고침: `Content-Encoding: deflate` 응답도 푼다(전에는 gzip만). |
| 끝까지 실행 | 실제 tsv 열 구성의 가짜 zip 3개(9~11개 제출)로 `process_zip`→`build`를 다시 돌렸다. 손계산과 모두 맞았다. 맞춘 값: known 2024-05-16·08-15·11-15, AI 보유 1,025,000·1,235,000·29,000, 콜 500,000, 풋 100,000, 순매수 210,000(NVDA 10:1 분할 반영)·−6,000(마감 전 RESTATEMENT로 대체, 마감 뒤 원본·수정본 제외). Citadel Securities·Vanguard 제외, GOOG→GOOGL, Two Sigma CIK 2개 병합도 맞았다. 점검은 scratchpad에서 했고 연구 캐시에는 쓰지 않았다. `--offline`은 받은 zip이 없다는 메시지를 내고 멈춘다(exit 1). |
| 남은 위험(실제 자료 전엔 확인 불가) | tsv 열 이름·날짜 형식은 기억에 기댄 것이다(다르면 열 검사에서 멈춤). 운용사 정규식이 같은 운용사의 다른 법인(예: Marshall Wace 두 법인, ValueAct)을 함께 잡을 수 있다. 첫 실행 뒤 `sec13f_funds.csv`를 눈으로 확인해야 한다. |

다시 받으려면 사용자가 SEC용 연락처 UA(이메일 포함)를 쓸지 정해야 한다. 바꿀 곳은 코드의 `UA` 상수 한 줄이다. 그 뒤 `py data/derived/analysis/hf_options/code/fetch_sec13f.py`를 실행한다.
