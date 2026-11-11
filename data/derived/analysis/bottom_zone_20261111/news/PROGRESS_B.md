# 과제 B 완료 — 2026-11-11

B1 전수 판정·v2 동결·가격 분석, B2 수집·부호화·동결·비교, report_B.md 작성과 전체 테스트를 완료했다. 이번 재개 작업에서는 실제 사용량 한도 거부 오류가 없었다. 종전 미완료 인계는 현재 상태에 적용하지 않는다.

| 단계 | 완료 증거 |
|---|---|
| 최초 규칙·부분 동결 | protocol_B.json / freeze_B.json; 4452ced / dd65093c |
| B1 전수 판정 | review_B1_v2.csv 4,200행; v2 추가 제목 판정 1,713행 |
| B1 완성 표본 | v2/calls_census.csv 181행, 24기관; 178행 공개일 확인 |
| B1 가격 전 동결 | freeze_B_v2.json; c5edc8affe00b4126b152f7aed9fd0ec9b75e95d |
| B1 가격 분석 | v2/results/analysis_B1.json, calls_with_prices.csv, split_check.json |
| B2 부호화 | bottom_news.csv 1,138행; 제목 수동 판정 1,458개 |
| B2 비교 전 동결 | freeze_B2_v2.json; 7c380ad6b842e95f901c823cc6bec271a85c6109 |
| B2 분석 | v2/results/analysis_B2.json, B2_period_counts.csv, B2_episode_counts.csv, B2_types.csv |
| 과제 A 전달 | v2/news_series_B.csv 178행; 공개일 기준 U/DN/TU/TC·asset_scope |
| 전체 검증 | v2/results/unittest_full_B.txt; py -m unittest, 520 tests, OK, exit 0 |
| 최종 보고서 | ../report_B.md; 최종 커밋 해시는 전달 메시지에 기록 |

최종 B1 정본은 v2/ 아래다. 루트 calls_census.csv·search_log_B1.csv·coverage_B1.csv·news_series_B.csv와 freeze_B.json은 첫 55행 부분 동결 당시의 불변 감사 자료로 보존했다. pending_review_B1.csv는 최초 큐의 역사적 파일이며 현재 미검토 상태를 뜻하지 않는다.

2026 SOXX H-U는 완전 관측 2건 중 바닥 근처 0건, 대조 27/66건, Holm p=1이다. H-D는 3건 중 2건, 가격 짝 1/3건, Holm p=0.442다. 진짜 바닥 +1~+5 부정 기사는 전체 72/274건(26.3%), 가짜는 72/184건(39.1%); 2026은 51/160건(31.9%) 대 47/133건(35.3%)이다.

남은 연구 실행 단계는 없다. 원문 접근 제한·날짜 대용·비공개 표본 누락·소표본·업데이트 의존성·라벨 비맹검은 보고서에 공개한 자료 한계다. 확률이나 매수 추천으로 바꾸지 않았고, 허용 경로 밖 다른 작업자의 변경은 그대로 두었다.
