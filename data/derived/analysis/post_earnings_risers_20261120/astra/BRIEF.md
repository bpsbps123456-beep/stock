# 실적 뒤 오른 종목의 특징 — Astra 1단계 과제서 (2026-11-20, Claude + Codex gpt-6.2-astra)

## 사용자 요청 (원문 뜻)
"실적 발표일 이후에 올랐던 기업들의 특징을 찾아보자. 내 생각에는 **개미들이 전혀 예측 못한 기업** 위주로 올랐을 것 같다. 종목은 TIME(456600)·KoAct(2ETFQ1)에 있는 것. 뻔한 이유(실적 숫자 등)로는 설명이 안 될 것이다."

앞선 저장소 연구(`data/derived/analysis/earnings_trend/report.md`)는 이미 이렇게 결론 냈다: 실적 내용(EPS 서프라이즈·가이던스)은 **반응일 하루**는 맞히지만 **그 뒤 흐름**은 못 맞힌다. 그래서 이번에는 실적 숫자 밖의 특징, 특히 '개미가 미리 몰려 있었나 / 몰라봤나'를 잰다.

## 역할 나눔
- Claude: 실적 사건표(반응일·결과 수익률), 가격 요인, 펀드매니저 수량 변화, OCC 고객 옵션 거래량, SEIBRO 한국인 결제 요인, 최종 분석·보고서·커밋.
- **Astra(너): 이 과제서의 두 가지.** ① 사전 등록 가설서 ② 저장소에 없는 '관심·포지션' 자료 수집. 2단계(사례 조사·가설 검정)는 끝난 뒤 Claude가 `send`로 따로 준다.

## 종목
`../results/universe.csv`의 미국 상장 161종목(2026-01-02~11-18 두 펀드 중 하나라도 보유). 기준 지수 QQQ·SPY·SOXX도 같이 받는다.

## 1번(필수, 가장 먼저): 사전 등록 가설서 `astra/HYPOTHESES.md`
**결과(사건 뒤 수익률)를 보기 전에** 쓴다. Claude가 만드는 `../results/events*.csv`의 수익률 칸은 이 파일을 다 쓰기 전에는 열지 않는다.
- 실적 숫자(EPS·매출·가이던스) **밖의** 측정 가능한 가설 15개 이상. 적어도 6개는 사용자 가설('개미가 예측 못함')을 서로 다른 방식으로 옮긴 것(관심 부족, 기대 낮음, 이미 반영 안 됨, 개미 쏠림 반대 등).
- 가설마다 한 줄 표: 정의(계산식), **언제 알 수 있나**(발표 전/반응일 종가/사후 설명용), 무료 자료원, 예상 방향, 틀릴 수 있는 이유.
- 사후에만 아는 사건(지수 편입, 실적 뒤 애널리스트 상향 등)은 '설명용'으로 따로 표시한다. 매매 근거로 쓰지 않는다(AGENTS.md 연구 보고 규칙 '변수 시점').
- 다 쓰면 맨 끝에 `작성 완료 시각:` 한 줄.

## 2번: 자료 수집 (기간 2025-09-01~2026-11-19, 결과는 파이썬으로 저장)
| 파일 | 내용 | 메모 |
|---|---|---|
| `astra/data/wiki_map.csv` | ticker, 영어 위키 문서명, 찾은 방법(Wikidata P249·수동), 확신(높음/낮음) | 같은 회사 두 종류 주식(GOOG·GOOGL)은 같은 문서 |
| `astra/data/wiki_views.parquet` | date, ticker, views(일별, user 접속) | Wikimedia REST pageviews. 사람용 User-Agent는 `GBRC-research personal-noncommercial research@example.com`처럼 개인 정보 없이 |
| `astra/data/short_interest.parquet` | settlement_date, publish_date(또는 추정 규칙), ticker, short_qty, avg_daily_vol, days_to_cover | FINRA 공매도 잔고(월 2회). **발표 지연**을 반드시 적는다 — 사건 날 이미 공개된 값만 쓰기 위해서다 |
| `astra/data/insider.parquet` | filing_date, trade_date, ticker, code(P/S), shares, value | SEC Form 4 장내 매수·매도. User-Agent는 `data/derived/analysis/earnings_trend/code/fetch_sec.py`의 `UA` 그대로, 요청 간격 0.2초 이상 |
| (선택) `astra/data/trends.parquet` | 구글 트렌드 주간 관심도 | 막히면 시도 결과만 PROGRESS에 |
| (선택) 그 밖에 가설서에 쓴 무료 자료 | 레딧·스톡트윗 언급 이력 등 과거 이력이 실제로 있는 것만 | 지금 값만 주는 자료는 받지 않는다 |

- 코드는 `astra/code/*.py`. 캐시·원본 큰 파일은 `astra/cache/`(Git 제외 경로 `data/**/cache/`). 결과 parquet 하나가 10MB를 넘으면 나눠서 저장하고 알린다.
- 각 파일의 커버리지(종목 수·날짜 범위·빈 종목)를 `astra/PROGRESS.md`에 적는다.

## 금지
- `docs/`, `AGENTS.md`, `../code/`, `../results/`, 다른 연구 폴더 수정 금지. 쓰는 곳은 `astra/` 아래만.
- git 커밋·푸시 금지(Claude가 한꺼번에 한다).
- 가격·수익률로 종목을 미리 고르거나, 결과를 보고 가설을 고치지 않는다(고친 것은 `HYPOTHESES.md` 끝에 '추가(결과 보기 전/후)'로 시각과 함께 덧붙인다).
- 멈춤: 요청이 실제로 거부되는 **사용량 한도 오류가 날 때만** PROGRESS에 상태를 적고 멈춘다. 그 밖에는 끝까지 한다.

## 완료 보고
`worker_done`에 ① 가설 수 ② 자료 파일별 커버리지 ③ 실패한 자료원과 이유 ④ 가설 중 가장 유망하다고 보는 3개와 이유.
