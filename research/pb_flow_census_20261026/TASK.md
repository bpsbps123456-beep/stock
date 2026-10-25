# Codex 조사 과제: 헤지펀드 프라임 브로커리지 자금 흐름 보도 전수 표본 (2026-10-26, Claude 요청)

## 배경

- `research/news_turning_points_20261025/`(Codex, 10/25)는 기사 31건으로 D·S·R·T 4유형을 검증했다.
- 이 표본은 검색으로 고른 것이라 빠진 주가 많을 수 있다. 저점 15개 중 6개, 고점 16개 중 3개에만 기사가 있었다.
- 사용자(2026-10-26)는 "Codex의 뉴스 분석을 이어서 분석하고, 더 필요한 데이터는 Codex에게 조사시키라"고 했다.
- Claude는 이 자료로 '뉴스가 중단기(1주~1달) 추세 대 V자 되돌림을 가르는가'를 시험한다. Claude의 앞선 결론은 `data/derived/analysis/swing_trend/report.md`에 있다. 가격·거래량·VIX·시장 폭 단서로는 가르지 못했다.
- 그래서 **기사 선택 편향이 없는 주간 시계열**이 필요하다.

## 해 줄 일 (자료 수집과 부호화만)

1. 대상과 기간은 2023-01-01 ~ 2026-10-23에 공개된 보도다.
   1. Goldman Sachs Prime Brokerage(PB) 주간·월간 헤지펀드 흐름 노트 보도(최우선)
   2. Morgan Stanley PB
   3. JPMorgan PB
   4. BofA 주간 고객 자금 흐름(Client Flow Trends: 헤지펀드·기관·개인 순매수/순매도)
2. 수집
   - 연·분기·월 단위로 검색어 틀 여러 개를 돌린다(Google News RSS, 언론사 사이트 검색, WebSearch 등).
   - **모든 검색어와 모든 결과(제목·날짜·URL·채택 여부·제외 이유)**를 `search_log.csv`에 남긴다.
   - 같은 원 노트의 재보도는 `source_group`으로 묶는다.
3. 부호화는 가격을 보기 전에 정한다. 규칙은 `protocol.json`에 먼저 고정하고, 지문(SHA-256)을 `freeze.json`에 남긴 뒤 부호화를 시작한다.
   - 한 행 = 한 원 노트(재보도는 대표 1행 + 그룹).
   - 열
     - `id`, `bank`(GS/MS/JPM/BofA), `note_date`(노트가 다루는 주 마지막 날 또는 발행일), `published_at`(뉴욕 시각, 모르면 날짜만), `time_precision`
     - `scope`(US 전체/세계/정보기술/반도체/TMT/AI 바구니 등)
     - `flow_direction`(net_buy / net_sell / mixed)
     - `long_flow`(buy/sell/na), `short_flow`(add/cover/na)
     - `magnitude_text`(원문 표현, 예: "largest since 2016", "fastest pace in 5 months")
     - `lookback_months`(원문 '~이후 최대'를 개월로, 없으면 빈칸)
     - `gross_leverage`·`net_leverage`(값과 백분위·"record" 여부, 있으면), `short_crowding`(원문 표현)
     - `family`(D·S·R·T, 10/25 연구 정의 그대로, 해당 없으면 N)
     - `quote`(핵심 1~2문장, 짧게), `url`, `source`, `verification`(원문 열람/재보도/차트만), `source_group`
   - 10/25 연구의 32건은 같은 열로 다시 부호화해 포함한다(그때 분류와 다르면 둘 다 남긴다).
4. 전수성 점검
   - `coverage.csv`: 주(월요일 시작)별로 GS 노트를 찾았는지, 몇 개 찾았는지.
   - `README.md`: GS 노트를 찾은 주 비율, 못 찾은 이유(유료 벽, 보도 없음 등)를 요약한다.
   - 블룸버그 유료 기사는 제목·발행일·요약만 있어도 된다. 다만 `verification`에 그렇게 적는다.

## 하지 말 것

- 가격과 결합해 결과를 보지 않는다(Claude가 사전 등록 뒤 분석한다). 수집 중 가격 차트를 보고 포함 여부를 바꾸지 않는다.
- `docs/desk_system.md`, `docs/operations_changelog.md`, `AGENTS.md`, `data/derived/analysis/`는 고치지 않는다. Claude가 결과를 받아 기록한다.
- 쓰는 곳은 이 폴더(`research/pb_flow_census_20261026/`)뿐이다. 원문 저장본이 크면 `data/derived/analysis/pb_flow_census_20261026/`(git 제외)에 둔다.

## 끝낼 때

- 이 폴더만 경로를 지정해 커밋한다: `git add research/pb_flow_census_20261026` 다음 `git commit -m "research: PB 자금 흐름 보도 전수 표본 2023~2026 (codex)" -- research/pb_flow_census_20261026`.
- 마지막 메시지에는 다음을 적는다: 행 수(은행별), GS 노트를 찾은 주 비율, 주요 한계, 커밋 해시.
- 시간이나 사용량이 모자라면 GS만이라도 끝까지 하고, 어디까지 했는지 `README.md`에 적은 뒤 커밋한다(나머지는 Claude가 이어서 한다).
