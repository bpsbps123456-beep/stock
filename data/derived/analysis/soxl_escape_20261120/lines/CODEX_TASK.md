# Codex 과제 1 — 엔진 독립 검산 + 채널·변동성·VWAP 갈래 (2026-11-20)

작업 폴더: `data/derived/analysis/soxl_escape_20261120/lines/` (저장소 루트 기준). 먼저 `BRIEF.md`와 `code/engine.py`를 끝까지 읽는다. 구간·채점·미래 정보 금지·보고 규칙은 BRIEF.md 그대로다.
Claude는 같은 엔진으로 이평선 갈래(단일 이평선+띠, 두 이평선 비대칭, 이평 교차, SOXX 이평)를 따로 돌린다. Claude 파일(`code/ma_*.py`, `results/ma_*.json`)은 읽지 않는다.

## 1단계 (필수, 맨 먼저): 엔진 독립 검산
엔진 함수(`simulate`·`evaluate`·`bars`)를 **쓰지 않고**, `cache/SOXL_1m.parquet` 원자료에서 네 코드로 아래 두 규칙의 SEL·CONF 성적(ret%, bh%, tv%p, mdd%, expo%, switches)을 다시 계산해 엔진 값과 비교한다.
- R1: 정규장 일봉(09:30~16:00) 종가 < EMA20(재귀식 alpha=2/21, 첫 값으로 시작)이면 대피, 종가 > EMA20이면 복귀. 봉 완성 뒤 첫 정규장 1분봉 시가 체결(다음 날 09:30).
- R2: 연장 포함 60분봉(04:00 기준 칸) 같은 규칙, 봉 완성 뒤 첫 1분봉 시가 체결(04:00~20:00 아무 봉).
손익은 SOXL 1분봉 시가→다음 시가 로그수익을 보유 중인 봉만 더한다(밤사이 갭 포함, 시작 상태는 보유, 2025-01-02부터 상태를 이어 옴).
엔진 값: R1 SEL ret 180.5 / tv 53.26 / mdd −52.18 / expo 58.5 / switches 23, CONF expo 0. R2 SEL ret 180.53 / tv 54.09 / mdd −57.8 / expo 57.5 / switches 302, CONF ret −16.29 / tv 16.79 / expo 39.4 / switches 46.
0.05%p 넘게 다르면 원인을 찾아 `results/codex_engine_check.json`에 적는다(엔진은 고치지 않는다).

## 2단계 (필수): 갈래 탐색 — 엔진 사용
- E 채널: 종가 < 직전 N봉 최저가면 대피(또는 그 선에 닿으면 스탑 체결), 종가 > 직전 M봉 최고가면 복귀. tf·N·M·세션.
- F N일 저가선: 직전 N거래일 정규장 저가(사용자 차트의 '5일저가선'과 같은 선)에 닿으면 대피, 복귀는 깨진 선 회복 또는 직전 M일 고가 돌파.
- G 변동성 추적: 샹들리에(최근 N봉 최고가 − k·ATR), 수퍼트렌드, 켈트너/볼린저 하단 이탈 대피·중심선 회복 복귀.
- H 앵커 VWAP: 주간·월간 앵커 VWAP 아래 대피/위 복귀.
- (선택) 위 계열의 SOXX 신호판(종가 판정만), '닿으면'과 '종가 확인' 비교.
격자는 성기게 시작해 설정 수를 정확히 센다. SEL로만 고르고 상위 3개(서로 다른 계열 우선)의 CONF·Y2025·B1·B2, placebo 백분위(SEL·CONF), 비용 0.1% 성적을 적는다.

## 결과물
- `code/codex_engine_check.py`, `code/codex_lines.py`(필요하면 나눠도 됨), `results/codex_engine_check.json`, `results/codex_lines.json`
- 사람이 읽을 요약 `results/codex_lines_summary.txt`를 파이썬으로 쓴다(한국어, 결론 먼저, 표 하나, 근거 3~5줄, 상위 3개 규칙의 Pine으로 옮길 정확한 정의).

## 금지·종료
- BRIEF.md의 금지 사항 그대로. 웹 검색 불필요. 자료 재수집 금지.
- 중간에 스스로 멈추지 않는다. 요청이 실제로 거부되는 한도 오류가 날 때만 `results/codex_progress.txt`에 진행 상황을 적고 멈춘다.
- 끝나면 worker_done으로 1단계 일치 여부와 상위 3개 규칙의 SEL/CONF 핵심 숫자를 보고한다.
