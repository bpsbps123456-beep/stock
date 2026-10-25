# 헤지펀드 PB 자금 흐름 보도 전수 표본 (2023-01-01 ~ 2026-10-23 공개)

Goldman Sachs·Morgan Stanley·JPMorgan 프라임 브로커리지(PB)와 BofA Client Flow Trends를 다룬 공개 보도를 모아, 원 노트 하나를 한 행으로 부호화했다. 가격·수익률은 보지 않았다. 가격 결합 분석은 Claude가 사전 등록(BRIEF N4)을 마친 뒤 따로 한다.

## 결론 요약

- 확정 그룹(원 노트) **252개**: GS 174 · BofA 38 · MS 22 · JPM 18. 출처 연결 457건(`group_sources.csv`).
- GS 노트를 찾은 주: 200주 중 **125주(62.5%)**. 중복이 불확실한 그룹을 빼면 113주(56.5%)이고, 노트 날짜가 명시된 주만 세면 96주(48.0%)다. 공개일 기준으로는 124주(62.0%)다.
- 은행 귀속을 확인하지 못한 후보 228건은 `census_unverified.csv`에 따로 두었고 주 통계에서 뺐다. 이 중 143건은 은행명이 없고, 158건은 2026년 보도다.
- 10/25 연구 32건: 29건은 그룹에 이었고 3건은 사유를 적어 제외했다. 분류가 바뀐 것은 N27 1건(T→R)이다.

### 은행별·연도별 그룹 수(공개일 기준)

| 연도 | GS | MS | JPM | BofA | 합계 |
|---|---:|---:|---:|---:|---:|
| 2023 | 27 | 8 | 10 | 6 | 51 |
| 2024 | 44 | 4 | 3 | 9 | 60 |
| 2025 | 53 | 4 | 2 | 6 | 65 |
| 2026(10/23까지) | 48 | 4 | 3 | 14 | 69 |
| 공개일 미상 | 2 | 2 | 0 | 3 | 7 |
| 합계 | 174 | 22 | 18 | 38 | 252 |

### 주별 포착(`coverage.csv`, 2022-12-26 ~ 2026-10-19, 월요일 시작 200주)

| 은행 | found | 불확실 중복 제외 | 노트 날짜 명시 | search_not_found | unresolved_candidate |
|---|---:|---:|---:|---:|---:|
| GS | 125 (62.5%) | 113 | 96 | 51 | 24 |
| MS | 21 (10.5%) | 21 | 9 | 160 | 19 |
| JPM | 18 (9.0%) | 18 | 7 | 169 | 13 |
| BofA | 37 (18.5%) | 37 | 8 | 158 | 5 |

GS를 연도별로 보면 2023년 24/52주, 2024년 35/53주, 2025년 36/52주, 2026년 30/42주다(2022-12-26 주는 1월 1일 하루뿐이고 0건). 한 주에 GS 그룹이 2개 이상인 주는 38주다. 일간 노트, 월간 노트, 주제별 노트가 같은 주에 겹치기 때문이다.

`search_not_found`는 "보도가 없었다"는 뜻이 아니다. 검색 색인에서 찾지 못했다는 뜻이다. `unresolved_candidate`는 그 주에 은행을 확인하지 못한 후보만 있다는 뜻이다.

### family 분포(10/25 연구 정의)

| 은행 | D | S | R | T | N |
|---|---:|---:|---:|---:|---:|
| GS | 57 | 10 | 54 | 11 | 42 |
| MS | 2 | 1 | 6 | 5 | 8 |
| JPM | 1 | 0 | 5 | 4 | 8 |
| BofA | 6 | 0 | 21 | 0 | 11 |
| 합계 | 66 | 11 | 86 | 20 | 69 |

- 흐름 방향: net_sell 106, net_buy 80, mixed 10, 빈칸 56. 빈칸은 보유·레버리지 수준만 있거나 방향을 모르는 경우다.
- 노트 날짜: 명시 137, 공개일 대용 115.
- 검증 수준: headline_summary_only 140(55.6%), syndicated_article_read 82, original_article_read 24, prior_verified_fact 6.
- 부호자: claude 207, codex 27, codex(+claude) 15, claude(+codex) 3. 두 부호자의 family가 다른 그룹은 1개(G0001)이고, 양쪽 부호를 모두 `coding_notes`에 남겼다(검증 반박에서 G0007의 Codex N을 원문 근거로 D로 고쳐 2개에서 1개가 됨).

## 파일

| 파일 | 내용 |
|---|---|
| `census.csv` | 한 행 = 한 원 노트 그룹. protocol 열에 `group_id`, `note_date_basis`, `n_sources`, `coder`, `uncertain_dup`, `legacy_ids`, `legacy_family`, `coding_notes`를 더했다. |
| `group_sources.csv` | 그룹별 출처(재보도 포함). `is_representative=yes`는 가장 이른 공개 출처다. |
| `census_unverified.csv` | 은행 귀속이나 관측 사실을 확인하지 못한 행 228건. 주 통계에서 뺐다. |
| `search_log.csv` | 질의 657개 × 결과 10,099건, 결과 0건 질의 75행을 더해 10,174행. 행마다 결정·제외 코드·이유가 있다. |
| `legacy_crosswalk.csv` | 10/25 연구 32건 → 그룹 id 또는 제외 사유. 원 분류와 현재 분류를 함께 적었다. |
| `coverage.csv` | 주 × 은행: 노트 날짜 기준(명시/대용), 공개일 기준, 불확실 중복, 미확인 후보 수, 상태. |
| `dedup_overrides.json` | 조립 단계의 수동 결정: 행 수정 4건(날짜 2·family 1·lookback 1), 병합 6건, 10/25 대조 20건, 다시 부호화한 10/25 행 6건. |
| `assemble.py` / `validate.py` | 조립과 검증. 결과는 `validation.json`(66개 점검 전부 통과)에 남는다. |
| `summary.json` | 조립 단계별 개수. |
| `protocol.json` / `freeze.json` | 부호화 규칙과 SHA-256 지문(c6047664…). 고치지 않았다. |

## 방법

1. **Codex 수집(10/25~26)**: 은행별로 월·분기·연 검색어 틀을 정해 Google News RSS 질의 586개를 돌렸고, 보충 WebSearch 71개를 더했다. 그 결과 RSS 후보 4,426건, 출처 5,887건, 등장 10,099건을 모았다. Codex는 키워드로 4,165건을 자동 제외하고, 검토 큐를 만들고, 기사 본문을 열람하고(76건 시도, 67건 본문 확보), 45개 노트를 부호화했다(`reviewed_notes.json`). 그러다 사용량 한도로 멈췄다.
2. **Claude 선별**: 후보 5,707건 전부를 LLM으로 제목·요약 판정했다.
   - PB 용어 포함 후보와 Codex 검토 큐 1,542건: exclude 1,135 · include 232 · needs_read 175.
   - Codex 자동 제외 4,165건은 먼저 150건 표본을 점검했다(exclude 144 · needs_read 5 · include 1). 놓친 후보가 있어서 나머지 4,015건도 전부 다시 판정했다(exclude 3,904 · needs_read 94 · include 17).
   - 합계: exclude 5,183 · include 250 · needs_read 274.
3. **Claude 부호화(706개 출처, 714행)**: 선별 include·needs_read 524건과 Codex 열람 큐 182건을 부호화했다. 열람 큐 182건 중 150건(작업 지시는 149건으로 셌다)은 브라우저 알림 권한 창 때문에 본문을 열지 못했다. 이 경우 Codex가 수집한 검색 요약문을 근거로 썼고 `verification=headline_summary_only`로 표시했다. 결과는 eligible 431행, unverified 230행, no 53행이다.
4. **10/25 대조**: 원 32건 중 29건을 URL 일치(12건), 제목 일치(9건), 같은 은행·날짜·사실(6건), 다시 부호화(2건)로 이었다. 지금 출처가 은행명 없는 제목뿐이라 unverified였던 N03·N20은 10/25 연구가 브라우저로 확인한 은행별 사실로 GS·JPM·MS 행을 다시 부호화했다(`prior_verified_fact`, 6행). 이 6행이 해당 출처의 unverified 행 2건을 대신한다.
5. **묶기(protocol dedup)**: 다음 경우에만 같은 그룹으로 묶었다.
   - 부호자가 같은 노트 키를 준 경우
   - 같은 출처·은행·노트 날짜
   - 정규화한 제목이 26자 이상 일치하고 은행이 같은 경우
   - 같은 은행·명시 노트 날짜
   - Codex `also_sources`
   - 수동 병합 6건(이유는 `dedup_overrides.json`에 적었다)

   같은 주의 비슷한 제목만으로는 묶지 않았다. 같은 은행 그룹이 4일 안에 있고 다음 중 하나에 해당하면 나중 그룹에 `uncertain_dup=possible_dup_of:<그룹>`을 붙였다: 부호자가 다른 그룹의 출처를 지목함, 제목 토큰 Jaccard 0.3 이상, 한쪽 날짜가 공개일 대용. 이렇게 표시한 그룹은 52개다(GS 50). 이 그룹은 `coverage.csv`의 `found_confirmed`와 `groups_note_confirmed_excl_uncertain`에서 뺐다.
6. **대표 행**: 대표 출처(`id`, `url`, `published_at`)는 그룹에서 가장 이른 공개 출처다. 부호 내용은 검증 수준이 가장 높은 행을 따랐고, 수준이 같으면 Codex 행을 썼다. 같은 노트를 다르게 부호화한 경우 `[codex …]`/`[claude …]` 형태로 차이와 원 메모를 `coding_notes`에 남겼다. 원문 인용(quote+magnitude_text+short_crowding의 원어 단어)이 25단어를 넘는 행은 quote를 잘랐다. `[요약]`으로 시작하거나 한글로 쓴 부호자 요약은 인용이 아니라서 세지 않는다.
7. **제외 코드 정규화**: 자유 서술로 된 사유를 protocol 코드 9개로 바꿨다. 결과는 unrelated 7,207, 13f_or_manager_holdings 623, forecast_not_observed 267, out_of_window 123, fund_flows_not_bank_client 72, other_bank 46, junk_page 37, insufficient_attribution 31, bofa_fms_without_cft 7이다(`search_log.csv`의 행 기준이라 같은 출처가 여러 번 등장하면 중복해서 센다).

## 한계

- **공개 언론 표본이다.** PB 고객 노트 원문 보관소가 아니다. 노트 원문은 거의 볼 수 없고, `article_read`는 기사를 읽었다는 뜻이지 노트를 읽었다는 뜻이 아니다. 언론이 고른 노트만 남으므로 극단적인 흐름이 과대 표집되었을 수 있다.
- **유료 벽**: Bloomberg 기사 상당수는 제목·요약만 썼다. 제목만으로 부호화한 그룹(headline_summary_only)이 55.6%다. 이런 그룹은 은행 귀속, 범위, 기간 환산(`lookback_months`)이 거칠다.
- **색인 누락**: Google News RSS는 한 질의에 최대 100건이고 색인 시각이 공개 시각이 아니다(`time_precision=date_feed` 58그룹). Claude 단계에서는 WebSearch를 쓰지 않았다. MS·JPM·BofA는 GS보다 검색 격자가 성기다. 결과 0건 질의도 75개다. 못 찾은 주가 노트 부재의 증거가 되지 않는다.
- **미확인 후보 228건**: 은행명 없는 제목(Reuters HEDGE FLOW, Bloomberg 유료 등)이 많다. 그중 158건이 2026년 보도라 2026년 GS 포착률은 실제보다 낮게 나왔을 수 있다. 10/25 원 기사 중 N14·N25도 이 이유로 제외 상태다.
- **날짜**: 노트 날짜의 46%(116그룹)가 공개일 대용이다. 공개일 미상 7그룹(Scribd 게재 자료 등)은 노트 날짜만 있다. 매매 시점 판단에는 `published_at`만 쓰고, 날짜만 있는 행은 다음 거래일로 보수적으로 다뤄야 한다(protocol timing).
- **판정 오류 가능성**: 선별과 부호화는 LLM 판정이고 독립 이중 부호화가 아니다. Codex 45건과 겹치는 18그룹에서 family가 갈린 것은 2건이었고, 그중 G0007은 검증 반박에서 원문을 확인해 D로 맞췄다. 중복 판정이 불확실한 그룹은 52개다.
- 가격·결과 근거로 포함 여부를 바꾸지 않았다. 다만 회고적 수집이라 선택 편향이 완전히 없어졌다고 증명하지는 못한다(protocol known_bias).

## 검증 반박(2026-10-26, Claude)

| 점검 | 결과 |
|---|---|
| `validate.py --unittest` | 고치기 전 65개, 고친 뒤 66개 점검 모두 통과. 저장소 unittest 302개 통과 |
| 무작위 20그룹(`random.Random(20261027)`) 원문 대조 | 은행 0/20, 흐름 방향 0/20(MS G0011 순매도는 해석 여지), family 1/20(G0007), note_date 1/20(G0021, 같은 주 안 1일). WebFetch 2회(1회 403) |
| 중복 묶기 | 같은 노트가 두 그룹에 나뉜 확정 사례 0. 같은 은행 3일 안 미표시 쌍 10개는 모두 사실·범위가 다른 노트. 한 그룹에 다른 노트가 섞인 사례 0. 부호화가 빠진 노트 1건: G0161 기사 속 GS PB 별도 노트(세계 산업재 6주 최대 매수). 그룹 번호가 밀리지 않도록 행을 더하지 않았다 |
| GS 주 포착률 | 125/200(62.5%)·113·96·124와 연도별 24/35/36/30을 census에서 다시 계산해 일치 |
| protocol | 가격 접근 0. 인용 25단어 초과 0. 단, 조립 단계가 한글 요약을 인용 단어로 세어 quote 12개를 지나치게 잘랐다(예: G0142 "hedge …") |

고친 것(`dedup_overrides.json` fix 3건 추가, `assemble.py`·`validate.py` 인용 계산):
- G0007: Codex N → D(롱 sell·숏 add·12개월). 원문 "largest notional net selling in more than a year"
- G0021: JPM note_date 7/14(공개일 대용) → 7/13(명시, "note dated July 13"). JPM 명시 주가 6에서 7로 늘었다
- G0161: "first time in seven weeks"는 연속 기록이라 lookback 1.62를 비웠다(선례와 통일)
- 인용 예산은 원어 단어만 센다. 잘렸던 quote 12개가 복원되었고, 잘린 quote가 예산을 다 쓰는지 점검을 더했다

## 재현

```
py assemble.py              # census·group_sources·census_unverified·search_log·legacy_crosswalk·coverage·summary 재생성
py validate.py --unittest   # protocol 점검 + 저장소 unittest → validation.json
```

- 입력은 이 폴더의 Codex 수집 파일(`sources_work.json`, `appearances_work.json`, `queries_work.json`, `candidates.json`, `reviewed_notes.json`, `article_reads/`)과 Claude 작업 파일 `data/derived/analysis/pb_flow_census_20261026/claude_work/`(`coded_*.json`, `screen_decisions_all.json`, `auto_excluded_ids.json`)이다. 둘 다 git에서 제외되어 있어 로컬 사본이 있어야 재생성된다. 10/25 원 32건은 `research/news_turning_points_20261025/events.json`에서 읽기만 했다(가격·결과 파일은 읽지 않음).
- 수집 자체의 재실행은 Codex의 `census.py {freeze,collect,supplement,inventory,prepare} --bank <은행>`이다. 검색 엔진 결과는 시점에 따라 달라진다.
