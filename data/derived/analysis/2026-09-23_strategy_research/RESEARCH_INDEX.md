# TIME·KoAct 전략 연구 문서 안내

현재 사용자 조건은 **미국 상장 주식만, 최대 10종목, 명시적 매매비용 0원**이다. 주문 가능 여부는 사용자 확인을 따른다. 아래 문서들은 과거 데이터 연구이며 실제 주문이나 자동 운용을 시작하는 문서가 아니다.

| 읽는 목적 | 문서 |
|---|---|
| 비중·섹터 제한을 추가한 균형 연구의 결론과 실행 규칙 | [집중도 연구 보고서](concentration_research/report.md) |
| 추가 실험의 사전 고정 조건·가정·입력 해시 | [집중도 연구 프로토콜](concentration_research/protocol.json) |
| 기존 후보가 어느 업종에 몰렸는지 | [섹터 감사](concentration_research/sector_audit.md) |
| 기존 후보와 TIME·KoAct의 동일 날짜 비교 수치 | [기준 비교표](concentration_research/baseline_table.md) |
| Claude H6/H6e와 Codex 기존 후보를 같은 시가 회계로 비교 | [공통 비교 보고서](strategy_comparison/comparison_report.md) |
| Claude 16장 의견, 종목 기여도, 초기 50:50 혼합 재검증 | [16장 후속 검토](strategy_comparison/section16_review/review.md) |
| 미국만·비용 0원으로 다시 계산한 이전 종가 연구 | [이전 종가 실험](us_only_experiment/zero_cost/report.md) |
| 맨 처음 수행한 전체 탐색의 기록 | [초기 report.md](report.md) |

초기 `report.md`에는 이후 사용자 조건과 달라진 시장 범위·비용 가정이 남아 있다. 원래 실험 기록으로 보존했으며, 그 수치를 현재 시가 체결 실험과 그대로 비교하면 안 된다. 각 문서의 기간, 공시 지연, 체결 시점, 현금·환율 가정을 함께 읽는다.

추가 연구의 코드 정본은 `gbrc/etf/research_concentration.py`, 종목 분류 정본은 기존 `watchlist.py`다. 연구 결과를 보고 운영 규칙이나 정본 분류를 자동 변경하지 않는다.
