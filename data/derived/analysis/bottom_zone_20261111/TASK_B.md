# Codex 과제 B: 어떤 뉴스가 떠야 진짜 바닥인가 — 전략가 의견 전수 표본과 바닥 창 뉴스 대조 (2026-11-11, Claude 요청)

## 배경

사용자 요청(2026-11-11, 원문 요지):

- 뉴스 매매: 어떤 뉴스가 떠야 정말 바닥이 나오는지 다양하게 자료를 수집해서 본다.
- 예: "씨티그룹의 주식 비중이 축소에서 중립으로 상향됐다" 같은 뉴스.
- 사용자 추측: "정말 완전 바닥에서는 오히려 계속 부정적인 뉴스가 쏟아질 것 같다."
- 완전 바닥이 아니라 바닥 근처를 맞히는 것이 목표다.

먼저 읽을 것:

| 자료 | 경로 |
|---|---|
| 운영 규칙·연구 보고 규칙 | `AGENTS.md` |
| 앞선 뉴스 연구 결론 | `docs/desk_system.md` §13, §13.2 ~ §13.5 |
| 앞선 전수 표본의 방식(검색 기록·부호화·동결) | `research/pb_flow_census_20261026/`(`TASK.md`, `protocol.json`, `README.md`) |
| 바닥·가짜 바닥 날짜 | `data/derived/analysis/seibro_bottoms/derived/events.json`(B10, `episodes[].F`, `current`) |

앞선 연구는 **헤지펀드 PB 자금 흐름 노트**(D·R·S·T)만 다뤘다. 이번에는 그 밖의 뉴스를 본다. PB 노트를 다시 모으지 않는다.

## 평가 구간

공개일 2024-01-01 ~ 2026-11-10. 2024·2025·2026을 나란히 내고 주 판정은 2026이다(§13.2 뉴스 연구와 같은 다년 범위). 2023 이전 자료는 쓰지 않는다.

## B1. 전략가·자산배분 의견 변경 전수 표본

**대상 기관**: Citi, Goldman Sachs, Morgan Stanley, JPMorgan, BofA, UBS, HSBC, Barclays, Deutsche Bank, Wells Fargo, BlackRock Investment Institute, Evercore ISI, Jefferies, Oppenheimer. 검색 중 반복해 나오는 다른 기관은 추가하고 `protocol_B.json`에 적는다.

**넣는 것**

1. 자산배분·전략 의견 변경: 미국 주식 / 세계 주식 / 미국 기술·성장·AI / 반도체 업종에 대한 상향·하향(예: underweight → neutral, neutral → overweight).
2. S&P 500·나스닥 지수 목표치 상향·하향.
3. 반도체 **업종 단위** 의견 변경(예: 업종 등급 하향, "반도체 비중 축소").

**빼는 것**: 개별 종목 투자의견·목표주가 변경, 의견 변경 없는 전망 반복, PB 자금 흐름 노트. 뺀 후보도 검색 기록에는 남긴다.

**열**(`calls_census.csv`, 한 행 = 한 의견 변경, 재보도는 `source_group`으로 묶음)

`id`, `house`, `strategist`, `published_date`, `published_at`(뉴욕 시각, 모르면 빈칸), `time_precision`(exact/date), `asset_scope`(us_equity / global_equity / us_tech / semis / index_target), `action`(upgrade / downgrade / target_raise / target_cut), `from_rating`, `to_rating`, `direction`(+1 / −1), `after_selloff`(원문이 급락·밸류에이션 조정을 근거로 들었으면 Y), `rationale_tag`(valuation / earnings / macro / positioning / policy / other), `quote`(짧게), `url`, `source`, `verification`(원문 열람 / 재보도 / 제목·요약만), `source_group`.

**수집**: 기관 × 분기 × 검색어 틀 여러 개(영문 + 한글). **모든 검색어와 모든 결과**(제목·날짜·URL·채택 여부·제외 이유)를 `search_log_B1.csv`에 남긴다. `coverage_B1.csv`에 기관 × 분기별 찾은 건수를 낸다.

## B2. 바닥 창 뉴스 목록 (진짜 바닥 대 가짜 바닥)

**날짜 목록**

- 진짜 바닥: `events.json` B10 가운데 2024~2026의 10개.
- 가짜 바닥: `episodes[].F`가 있는 2024 이후 국면마다 **첫 F 날** 하나. 국면 수는 `events.json`에서 센다.
- 진행 중: 2026-10-05 고점 이후 하락의 현재 최저 종가일. 판정에 넣지 않고 '지금' 표에만 쓴다.

**창**: 각 날짜의 −5 ~ +5 미국 세션.

**고정 검색어 틀**: 날짜마다 똑같이 쓴다. `protocol_B.json`에 먼저 적는다. 최소 포함: 반도체주·칩주 등락 이유, 나스닥·미국 증시 마감 기사, 엔비디아, 연준·금리·관세·수출 규제, 전략가·투자의견, 약세장·조정·패닉·투매, 헤지펀드·개인 매수, 한글 "반도체주 급락", "반도체주 반등", "뉴욕증시 마감".

**열**(`bottom_news.csv`, 한 행 = 한 기사)

`id`, `anchor_date`, `anchor_type`(real / fake / current), `published_date`, `published_at`, `time_precision`, `k`(기준일 대비 세션), `headline`, `type`, `tone`, `is_turn_catalyst`(기준일 또는 +1세션의 방향 전환과 함께 보도된 재료면 Y), `url`, `source`, `verification`, `source_group`.

- `type`: macro_policy / earnings_fundamental / flows_positioning / strategist_call / regulation_geopolitics / capitulation_language / technical / other.
- `tone`: 반도체·주식에 대해 neg / pos / neutral. 판정 규칙을 `protocol_B.json`에 낱말·예문으로 먼저 고정한다.
- 창마다 세션별로 최소 한 건의 마감 기사(통신사)를 넣어, 뉴스가 없어서 비는 날과 못 찾아서 비는 날을 구분한다.
- 모든 검색어·결과는 `search_log_B2.csv`.

**알려진 편향**: 날짜가 과거라 진짜·가짜를 알고 부호화하게 된다. 그래서 검색어와 부호화 규칙을 먼저 고정하고, 날짜 순서를 섞어 부호화하며, 이 한계를 보고서에 적는다.

## B3. 동결 (가격 연결 전)

`protocol_B.json`, `calls_census.csv`, `bottom_news.csv`, 검색 기록의 SHA-256을 `freeze_B.json`에 남기고 커밋한다. **이 커밋 전에는 가격을 연결한 결과를 계산하지 않는다.** 수집 중 가격 차트를 보고 포함 여부를 바꾸지 않는다.

## B4. 분석 (동결 뒤)

가격은 SOXX(보조 QQQ) 전체 이력을 새로 받는다. SOXX 3:1 분할(2026-11-05)이 섞이지 않았는지 검사한다. 공개일만 알면 다음 정규장 시가 진입이다.

**전략가 의견(B1)** — 4종류(상향·하향·목표 상향·목표 하향) × 상태 2개(하락 상태 안 / 밖) = 8칸.

- 하락 상태 = 직전 SOXX 종가가 60세션 최고 종가보다 10% 이상 낮음.
- 결과: 진입 대비 20세션 안 최저 종가 ≥ −5%(바닥 근처 진입, 주 결과), 5·10·20세션 수익.
- 대조: 같은 해·같은 상태의 아무 날, 그리고 (DD60, 직전 5세션 수익)이 가장 가까운 날 1:1 가격 짝.
- 바닥 대비 위치: 의견마다 가장 가까운 B10 바닥에서 몇 세션 전·후인지. 바닥 ±10세션 안의 상향·하향 개수. **바닥 전에 나온 상향은 몇 건인가, 상향은 바닥 뒤 며칠째에 나오나.**
- 사용자 가설 두 개에 직접 답한다.
  - H-U: 하락 상태에서 나온 상향(축소 → 중립 포함) 뒤가 대조보다 바닥 근처인가.
  - H-D: 하락 상태에서 하향·목표 하향이 몰린 뒤가 오히려 바닥 근처인가(역발상).
- 주 검정은 H-U·H-D의 2026 두 칸(연도 안 원형 이동 10,000회, 시드 20261111, Holm 2). p < 0.05를 일괄 합격선으로 쓰지 않는다.

**바닥 창 뉴스(B2)**

- 구간 [−5, −1], 0, [+1, +5]별 부정 비율과 유형 분포를 진짜·가짜로 나란히.
- **사용자 추측 검증: 진짜 바닥의 [+1, +5] 부정 비율이 가짜 바닥보다 높은가(바닥 뒤에도 악재가 이어졌나).**
- 전환 재료 유형 빈도(정책 / 실적 / 수급 / 전략가 / 없음). 바닥마다 한 줄 요약(무슨 뉴스와 함께 돌았나).
- 사례가 10개 안팎이므로 확률·적중률로 바꾸지 않고 건수로 쓴다.

**지금**: 2026-10-05 이후의 전략가 의견 목록과 최근 10세션 뉴스 프로필. 확률·매수 추천으로 바꾸지 않는다.

**과제 A에 넘길 계열**: `news_series_B.csv`(열 `date`, `family` ∈ U 상향 / DN 하향 / TU 목표 상향 / TC 목표 하향, `asset_scope`). `date`는 공개일이다.

## 산출물 (쓰는 곳은 이 폴더의 `news/` 아래 + `tests/`의 새 테스트 파일뿐)

`news/protocol_B.json`, `news/freeze_B.json`, `news/calls_census.csv`, `news/bottom_news.csv`, `news/search_log_B1.csv`, `news/search_log_B2.csv`, `news/coverage_B1.csv`, `news/news_series_B.csv`, `news/code/`, `news/results/`, `report_B.md`.

`report_B.md`: 결론 첫 줄 → 표 → 근거 3~5줄. 검색 횟수·후보 수·채택 수·원문 열람 비율, 8칸이라는 시험 범위, 편향·한계 포함. 문장은 짧은 한국어.

## 하지 말 것

- `docs/`, `AGENTS.md`, `desk/`, `gbrc/`, `data/news_evidence/ledger.json`, 다른 연구 폴더를 고치지 않는다. 정본 기록은 Claude가 한다.
- 다른 작업자의 미커밋 파일(`data/derived/desk/*`, `data/derived/analysis/ma_entry_filter_20261108/*`, `ant_index/*`)과 과제 A의 파일(`bottom_zone_20261111/`의 `news/` 밖)을 건드리지 않는다.
- 사이트 알림 권한 창·로그인 벽을 우회하지 않는다. 못 읽은 기사는 `verification`에 그렇게 적는다.
- 사용자 이름·이메일을 검색어나 요청에 넣지 않는다. TradingView MCP를 쓰지 않는다.

## 끝낼 때

- 경로 지정 커밋: `git add data/derived/analysis/bottom_zone_20261111/news data/derived/analysis/bottom_zone_20261111/report_B.md` → `git commit -m "research: 바닥 뉴스 전략가 의견 전수 표본과 바닥 창 대조 (codex)" -- <경로…>`. 원문 저장본이 10MB를 넘으면 올리지 않는다.
- `git push origin master`. 거부되면 rebase하지 말고 커밋을 로컬에 둔 채 마지막 메시지에 적는다.
- 마지막 메시지: B1 행 수(기관별·종류별), 검색 횟수, 원문 열람 비율, H-U·H-D 2026 결과, 진짜 대 가짜 바닥의 부정 비율, 전환 재료 유형, 한계, 커밋 해시.
- 사용량이 모자라면 ① B1 수집·동결 ② B1 분석 ③ B2 순으로 끝낸 데까지 `news/PROGRESS_B.md`에 적고 커밋한다. Claude가 이어받는다.
