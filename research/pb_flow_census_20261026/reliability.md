# PB 자금 흐름 표본 부호화 신뢰도 점검

- 작성: 2026-10-26, Claude 독립 재부호화(일치율 담당)
- 대상: `data/derived/analysis/pb_flow_census_20261026/claude_work/coded_*.json`의 eligible=yes 행과 `reviewed_notes.json`(Codex) 행
- 행별 결과: `reliability.csv`(40행). 재현 자료: `data/derived/analysis/pb_flow_census_20261026/claude_work/reliability_recode/`(git 제외)
- `protocol.json` SHA-256 `c6047664…501cea3`는 바뀌지 않았다. 가격 자료와 WebSearch는 쓰지 않았다.

## 결론

40행 7개 항목에서 은행·숏 흐름 98%, 흐름 방향·롱 흐름·lookback 90%, 계열 85%(κ 0.77), note_date 82%가 일치했다. note_date는 ±3일 기준으로 95%다. 불일치 27건 중 16건은 규칙 해석 차이, 11건은 정보 부족이다. 규칙 해석 차이는 note_date 환산(6건)과 한 노트 안에서 어떤 사실을 지배 사실로 볼지(계열 4건)에 몰렸다.

| 항목 | 전체 40 | Claude 28 | Codex 12 | κ(전체) | 전체 95% CI |
|---|---|---|---|---|---|
| bank | 39 (98%) | 27 | 12 | 0.94 | 0.87~1.00 |
| flow_direction | 36 (90%) | 26 | 10 | 0.85 | 0.77~0.96 |
| long_flow | 36 (90%) | 25 | 11 | 0.80 | 0.77~0.96 |
| short_flow | 39 (98%) | 27 | 12 | 0.96 | 0.87~1.00 |
| family | 34 (85%) | 24 | 10 | 0.77 | 0.71~0.93 |
| lookback_months (±1개월) | 36 (90%) | 26 | 10 | — | 0.77~0.96 |
| note_date (정확히 같은 날) | 33 (82%) | 26 | 7 | — | 0.68~0.91 |
| note_date (±3일) | 38 (95%) | 26 | 12 | — | — |
| 7개 항목 모두 일치한 행 | 24 | 21 | 3 | — | — |

- lookback: 값이 있는지 없는지는 37/40행이 같았다. 둘 다 숫자인 16행 중 15행은 값까지 같았다.
- bank: 노트가 여럿인 기사 8행은 기존 행의 은행으로 짝을 지었으므로 이 8행은 은행 검정이 아니다. 이 8행을 빼면 31/32행이 일치했다.
- Codex 12행에서 note_date 정확 일치가 7행뿐이다. Codex가 note_date를 비워 두면 `assemble.py` 규칙대로 공개일이 들어가는데, 원문에 날짜가 있는 경우에도 빈칸이었다. 12행 모두 ±3일 안이다.

## 방법

| 단계 | 내용 |
|---|---|
| 모집단 | Claude eligible=yes 431행, Codex 45행. 행 키는 `id#같은 id 안의 순번`(파일 이름 순서, 행 순서) |
| 추출 | `random.Random(20261026)`으로 정렬한 Codex 키에서 12행, 이어서 정렬한 Claude 키에서 28행을 뽑았다. 모두 40행이고 기사로는 39건이다 |
| 입력 | 제목·출처·URL·공개일과 evidence·snippet·article_flow_paragraphs, `article_reads/` 본문(13건). 앞 단계 판단(screen_bank·screen_note·screen_decision)은 넣지 않았다 |
| 입력 수준 | evidence 본문 21행, 짧은 snippet 13행, 제목만 6행 |
| WebFetch | 14회 호출(한도 15회). 9회는 재부호화에 쓸 본문을 받았다. 4회는 내용을 받지 못했다(403, Google News 껍데기, reddit 차단, 리다이렉트 1회). 나머지 1회는 잠금 뒤 원인 확인용이라 부호에는 반영하지 않았다 |
| 맹검 | 재부호화를 `recode.json`에 쓰고 SHA-256 `e77ceefc…e81c76`으로 잠근 뒤에 기존 부호를 열었다. 잠그기 전에는 기존 파일의 구조·행 수·eligible 값만 확인했고 부호 값은 보지 않았다 |
| 비교 | 공란·na·None은 같은 값으로 본다. lookback은 ±1개월 안이면 일치로 보고 정확히 같은 경우를 따로 센다. note_date는 정확히 같은 날을 일치로 보고 ±3일과 같은 ISO 주를 따로 센다 |

재부호화에 적용한 해석은 `recode.json`의 `rules_applied`에 적었다. 요약하면 아래와 같다.

- note_date: 원문에 적힌 대상 주의 마지막 날을 먼저 쓴다. 없으면 기사 날짜를 기준으로 환산한 발행일("note dated Friday" 같은 표현)을 쓰고, 그것도 없으면 공개일 대용으로 둔다.
- lookback: 지배 사실에 붙은 극단 표현만 개월로 바꾼다. "first time in N weeks" 같은 연속 기록은 lookback으로 보지 않는다.

## 불일치 27건과 원인

| # | 행 | 항목 | 기존 → 재부호화 | 원인 | 내용 |
|---|---|---|---|---|---|
| 1 | Codex 2843b81e72e4#0 (GS) | flow_direction | 공란 → net_buy | 규칙 | 원문 "Net buying ... largest level since October last year"는 명시적 순매수다. Codex는 숏 커버가 주도한 매수라 흐름 칸을 비웠다 |
| 1 | 〃 | lookback | 공란 → 9 | 규칙 | 순매수 lookback(2022-10 이후)을 Codex가 비웠다. 매크로 숏 커버의 "since November 2020"(32개월)을 고를 여지도 있다 |
| 3 | Codex 2843b81e72e4#1 (JPM) | note_date | 07-14 → 07-13 | 규칙 | 원문이 "note dated July 13"인데 빈칸이라 공개일이 들어갔다 |
| 4 | Codex ad98d9050c54 | family | N → S | 규칙 | 에너지 비중이 2020-05 저점 근처다. Codex는 "낮은 보유 비중은 극단성 근거가 아니다"라고 판단했다 |
| 4 | 〃 | lookback | 공란 → 40 | 규칙 | "approaching a May 2020 low"를 lookback으로 볼지에 따라 갈렸다 |
| 5 | Codex 89c0590362e8#1 (JPM) | family | N → T | 규칙 | 지배 사실을 매도 흐름으로 볼지, 2018년 이후 기록 근처 노출로 볼지 |
| 6 | Codex cd7a6d5b86e5 | note_date | 08-17 → 08-16 | 규칙 | "report released on Wednesday"를 날짜로 환산할지 |
| 7 | Codex 1c0b45b1f49a | note_date | 10-31 → 11-01 | 규칙 | 월간 노트에서 대상 월말을 쓸지 공개일을 쓸지 |
| 8 | Codex 0200d6b3c0ac | long_flow | buy → na | 규칙 | "long positions in IT reached the largest in eight months"(보유 수준)를 롱 매수로 볼지 |
| 10 | Codex 7eee58bd5f5f | flow_direction | 공란 → net_sell | 정보 | Codex는 발췌만 봤고, 재부호화는 본문의 "generally net sellers of global stocks"를 봤다. 범위도 IT와 세계로 다르다 |
| 10 | 〃 | note_date | 11-17 → 11-16 | 규칙 | 프로토콜이 대상 주 종료일과 발행일을 둘 다 허용한다 |
| 12 | Codex e3dc68ad1bf8 | note_date | 07-28 → 07-31 | 정보 | 기사에 GS 보고서가 둘(금요일, 월요일) 나오고, 숏 청산 사실이 어느 쪽 보고서인지 원문이 모호하다 |
| 18 | Claude 6ebf9522baab | family | D → R | 규칙 | 지역이 여럿인 노트에서 유럽의 5년 만의 최대 매도(D)와 미국 재편입(R) 중 무엇을 지배 사실로 볼지 |
| 19 | Claude cd3b6e80c7a4#1 | flow·long·family | net_sell·na·N → net_buy·buy·R | 정보(단위) | 기사 속 GS 미국 "separate note"를 재부호화가 놓치고 세계 노트 하나만 만들었다. 잠금 뒤 확인하니 별도 노트가 맞았다. 반영하면 3건 모두 일치하지만 통계에는 넣지 않았다 |
| 27 | Claude 44c26ac783a2 | family·lookback | T·60 → R·12 | 정보 | 기존 부호자는 같은 그룹 기사 본문의 "leverage now stands at a five-year high"를 봤다. 재부호화 입력에는 5월 매수 속도만 있었다 |
| 27 | 〃 | note_date | 05-31 → 06-05 | 규칙 | 월간 노트에서 대상 월말을 쓸지 공개일을 쓸지 |
| 28 | Claude 2da96dc2e812 | family | D → S | 규칙 | 둘 다 전사문을 읽었다. 3·5년 기준 2~3백분위인 순노출을 기존은 상당한 디레버리징(D)으로, 재부호화는 극단적 저노출(S)로 봤다 |
| 29 | Claude e7bac6103b58 | bank·flow·long·lookback | GS·net_sell·buy·60 → 공란·공란·na·공란 | 정보 | 재부호화 입력은 제목뿐이었다(은행명 없음, Google News 열람 실패). 기존 부호자는 247wallst 원 주소를 열람했다 |
| 37 | Claude 881ee06554a6 | long·short | na·na → buy·cover | 규칙 | 개별주 사실(롱 매수·숏 커버)을 북미 행에 넣을지. 기존 부호자는 "지역 구분이 없는 사실이므로 na"라는 선례를 따랐다 |
| 39 | Claude 5d371ca00cc3 | note_date | 01-15 → 01-19 | 규칙 | "through Thursday"를 날짜로 환산할지. 재부호화는 6ebf9522baab에서는 환산해서 자기 안에서도 일관되지 않았다 |

항목별 원인 집계는 아래와 같다.

| 항목 | 규칙 해석 | 정보 부족 |
|---|---|---|
| note_date | 6 | 1 |
| family | 4 | 2 |
| flow_direction | 1 | 3 |
| long_flow | 2 | 2 |
| lookback_months | 2 | 2 |
| short_flow | 1 | 0 |
| bank | 0 | 1 |
| 합계 | 16 | 11 |

## 해석

- **note_date는 규칙의 빈틈 때문에 갈린다.** 프로토콜은 "covered week-ending or note issue date"를 둘 다 허용한다. 월간 노트의 월말과 "on Wednesday"·"through Thursday" 같은 상대 요일을 어떻게 환산할지는 정하지 않았다. 그래서 세 부호자(기존 Claude, Codex, 재부호화)가 서로 다르게, 한 부호자 안에서도 다르게 처리했다. 차이는 최대 5일이고 37/40행은 같은 주에 들어간다.
- **계열 차이 6건 중 4건은 한 노트에 성격이 다른 사실이 함께 있는 경우다.** 매도와 기록적 노출, 유럽 매도와 미국 재매수, 저노출과 디레버리징이 그렇다. 기존 coding_notes는 대부분 대안 계열을 적어 두었다(6ebf9522baab, 2da96dc2e812, 89c0590362e8).
- **정보 수준이 일치율을 크게 바꾼다.** 재부호화가 WebFetch 없이 입력만 썼다면 롱 흐름은 33/40, 숏 흐름은 34/40, 흐름 방향은 35/40에 그쳤다. 기존 Claude 부호자도 WebFetch나 같은 그룹 판본으로 본문을 읽었기 때문이다. 제목만 있는 행(e7bac6103b58)에서는 은행 귀속까지 갈렸다.
- **Codex는 빈칸을 남기는 쪽으로 기울었다.** note_date 빈칸, 숏 커버가 주도한 순매수의 흐름 빈칸, 저노출을 S로 보지 않은 경우가 그렇다. 7개 항목이 모두 일치한 Codex 행은 3/12행이다. Codex 행의 불일치 12건 중 5건은 note_date의 1~3일 차이다.

## 하류 분석(사전 등록)에 주는 함의

1. **시점:** note_date를 일 단위로 정밀하게 쓰지 않는다. 주 단위로 묶고, 거래 가용 시점은 프로토콜대로 published_at을 쓴다.
2. **주 변수:** flow_direction(κ 0.85)과 short_flow(κ 0.96)는 주간 순매수·순매도 시계열의 주 변수로 쓸 만하다.
3. **계열:** family(κ 0.77)로 검정할 때는 coding_notes의 대안 계열로 바꿔 넣는 민감도 분석을 함께 둔다. 특히 D/R, T/N, D/S가 함께 있는 행이다.
4. **lookback 문턱:** lookback으로 문턱(예: 12개월 이상)을 걸 때는 문턱 근처 행과 매크로·섹터 범위가 따로 있는 행을 확인한다.
5. **Codex 행:** Codex 45행 중 note_date가 빈 27행은 원문에 날짜가 있는지 다시 확인하면 정확도가 오른다. 이 과제에서는 `assemble.py`나 기존 부호를 고치지 않았다.
6. **제목뿐인 행:** 제목만 있는 행의 비중을 확인한다. 이 표본에서는 6/40행이었다. 이런 행은 은행·방향 귀속에 정보 위험이 있다.

## 한계

- n=40이라 신뢰구간이 넓다. 예를 들어 family는 0.71~0.93이고, Codex 12행만 보면 더 넓다.
- 재부호화자와 기존 Claude 부호자는 같은 모델 계열이다. 프로토콜을 같은 방식으로 이해할 수 있어 Claude 28행의 일치율은 과대 추정일 수 있다. Codex 12행 비교가 더 독립적이다.
- 노트가 여럿인 기사 8행은 은행으로 짝을 지었다. 재부호화와 기존의 노트 수가 다른 기사도 있었다(cd3b6e80c7a4, 0200d6b3c0ac, 6ebf9522baab).
- 원인 분류(규칙 해석/정보 부족)는 재부호화자가 기존 coding_notes를 읽고 내린 판단이다.

## 부록: 기존 Claude 대 기존 Codex(같은 출처 id, 맹검 아님)

Codex 행 12개가 Claude도 부호화한 출처 id 8개에 걸쳐 있다. 은행으로 짝지은 8행의 일치는 flow_direction 7, short_flow 7, family 6, long_flow 5, lookback 5행이었다. 같은 은행의 첫 Claude 행과 단순 대조한 값이라 참고용이다.
