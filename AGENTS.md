# 성장기반 리서치 센터(GBRC) 운영 라우터

> 활성 기준일: **2026-09-27**  
> 이 파일은 트리거와 문서 라우팅만 관리한다. 계산·출력·검증 세부 규칙은 각 도메인 문서를 정본(SSoT)으로 삼는다.

## 현재 운영 모드

- 매일 기본 업무는 TIME 글로벌AI인공지능액티브 `456600`과 KoAct 미국나스닥성장기업액티브 `2ETFQ1`의 펀드 추종뿐이다.
- Path-B와 Path-F는 ETF-only다. 옵션, 옵션 순위, 옵션×ETF 등급, 개별 종목 밸류에이션/PER를 섞지 않는다.
- Path-B와 Path-F에서는 `daily_analysis.py`, `options_scanner.py`, `options_tracker.py`, valuation cache 갱신, PER 검색·계산을 실행하지 않는다.
- 옵션은 Path-O 또는 사용자의 명시적 옵션 요청, 실적은 Path-E/V/S/I, 뉴스는 Path-N, 애널리스트 추적은 Path-A에서만 수행한다.
- 섹터·종목 분류의 코드 정본은 `watchlist.py`다. 문서에 워치리스트를 중복 기재하지 않는다.
- TradingView MCP는 대화 중 조회(Path-E/V/Q)에만 쓴다. 스크립트, `automation/` 프롬프트, 옵션 스코어 입력에 연결하지 않는다(약관 3조 display-only·25조 비자동화 개인 용도). 사용 규칙은 `docs/earnings_framework.md`를 따른다.

## 정본 문서

| 범위 | 정본 |
|---|---|
| 트리거·라우팅·자기검증 | `AGENTS.md` |
| 애널리스트 신호 수집·기록·채점·행동 규칙 | `docs/analyst_system.md` |
| 두 펀드 수집·비교·성과·출력·저장 | `docs/etf_system.md` |
| 옵션 스코어·복기·돌발 재료 검증 | `docs/options_system.md` |
| 실적 분석·검증·성적표·개선 루프 | `docs/earnings_framework.md` |
| 섹터·종목 매핑 | `watchlist.py` |
| 운영 규칙 변경 이력 | `docs/operations_changelog.md` |
| 투자 브리핑 전달·활용 | `docs/investment_briefing_playbook.md` |
| 야간 데스크 전략·주문 계산·매매 기록·멈춤 규칙, AI 탐방 탭(지도 `desk/tour_map.json`), 거인의 어깨 탭(`desk/giant.py`) | `docs/desk_system.md` |

`CLAUDE.md`는 호환용 진입점이며 독립 규칙을 두지 않는다. `docs/v40_handoff.md`와 `docs/archive/v40_handoff.md`는 과거 참고자료로서 활성 운영 규칙이 아니다.

## Gate A. 트리거 식별

모든 요청은 응답을 시작하기 전에 아래 Path로 분류한다. 명시적 트리거가 둘 이상이면 각 Path를 독립 실행하되 Path-B/F 결과에는 옵션·PER를 합치지 않는다.

| 입력 키워드 | Path |
|---|---|
| 브리핑 / 오늘 브리핑 / 장 브리핑 / 누적 브리핑 | **Path-B** |
| 실적 분석 / 어닝 분석 / `<티커> 실적` / 실적 자료 첨부 | **Path-E** |
| 실적 복기 / `<티커> 복기` / 실적 검증 | **Path-V** |
| 옵션 복기 / 어제 복기 | **Path-O** |
| 실적 성적표 | **Path-S** |
| 실적 프롬프트 개선 | **Path-I** |
| 뉴스 검색해줘 / 뉴스 찾아줘 | **Path-N** |
| 3명 업데이트 / 애널리스트 업데이트 / 애널리스트 성적표 | **Path-A** |
| 펀드매니저의 포트폴리오 / 펀드매니저 포트폴리오 / ETF 점검 / 구성종목 변화 / 자금흐름 / 실매매 / 매매 분석 / 매매 타이밍 / 언제 수익 / 야간 데스크 | **Path-F** |
| 기타 | **Path-Q** |

## Gate B. Path 라우팅

| Path | 선행 문서 | 필수 작업 |
|---|---|---|
| **Path-B / Path-F** | `docs/etf_system.md` | `py gbrc.py briefing` 단일 명령으로 건너뛴 평일 채우기→TIME·KoAct 수집→검증→비교→내 7종목 점검·실매매 합의·관찰군 요약→가격 성과 비교를 수행하고 ETF-only 보고서를 출력한다. 누적 브리핑 요청이면 이어서 `py cumulative_briefing.py`로 마지막 전달 브리핑 이후 구간을 한 보고서로 묶는다. 실매매·매매 타이밍·수익 구간 분석 요청이면 `py gbrc.py etf trades --fund <time\|koact>`를 실행한다. 매매는 비중이 아니라 수량 기준 실매매로만 말한다. 세부 순서와 전 항목은 ETF 정본을 따른다. 야간 데스크 요청이면 `docs/desk_system.md`도 읽고, 데스크 코드만 고칠 때는 briefing 대신 그 §6 검증을 통과시킨다. |
| **Path-E** | `docs/earnings_framework.md` | `py earnings_prefill.py <TICKER>` → 누락값을 정본의 데이터 소스 순서대로 보완 → 6단계 분석 → `data/earnings_log/YYYY-MM-DD_<TICKER>.json` 저장. 시총·컨센서스·전년값을 가정으로 채운 채 분석하지 않는다. |
| **Path-V** | `docs/earnings_framework.md` | 해당 로그 로드 → yfinance D+1/D+5/D+30 수익률 → 오차가 큰 종목의 돌발 재료 WebSearch → `verification` 갱신. 검색 없이 분석 실패로 단정하지 않는다. |
| **Path-O** | `docs/options_system.md` | 전일 로그와 현재 가격 매칭 → Top/Bottom 평균 수익률 → 역전 Top 3 각각 WebSearch → Tier-1/2/3 분류와 결론. 돌발 재료 확인 없이 가중치 변경을 제안하지 않는다. |
| **Path-S** | `docs/earnings_framework.md` | 전체 실적 로그의 적중률과 평균 수익률 집계. |
| **Path-I** | `docs/earnings_framework.md` | 검증 5건 이상일 때만 개선 후보 제안 → 사용자 확정 후 관련 정본과 변경 이력 갱신. `CLAUDE.md`에 독립 규칙을 추가하지 않는다. |
| **Path-N** | 없음 | WebSearch 4회: 빅테크 → AI반도체 → 메모리 → AI인프라. 오늘 → 어제 → 이번 주 순으로 확인하고 한글로 출력. |
| **Path-A** | `docs/analyst_system.md` | 업데이트: WebSearch로 3인(가브리엘·윌슨·카토네) 신규 발언 수집 → 신규 콜 `data/analyst_log/` 기록 → `py analyst_tracker.py verify` → 이전 전망 대비 변화·행동 조건 표 출력. 신규 없으면 "신규 없음" 종결. 성적표: `py analyst_tracker.py scorecard`. 3인 의견만으로 매도 제안을 하지 않는다. |
| **Path-Q** | 없음 | 요청 자체를 수행한다. AI 종목의 전고점 거리 분류는 `py ai_ath_gap.py`로 계산한다(유니버스 정본 `watchlist.AI_UNIVERSE`, 결과 `data/derived/analysis/ai_ath_gap/`). AI 탐방 갱신은 `docs/desk_system.md` §8.1 체크리스트대로 `desk/tour_map.json`을 고치고 `py -m unittest tests.test_tour tests.test_scenes`를 통과시킨다(유니버스 종목은 모두 탐방 지도에 배치, 부서마다 출처 있는 대장, 단면도 이름표는 `py -m desk.scenes.layout`으로 최신화). 섹터 상대강도·주도주·고래 신호(거인의 어깨 탭)는 `docs/desk_system.md` §9를 따르고 `py -m unittest tests.test_giant`를 통과시킨다. 매수 추천·주문 계산은 하지 않는다. |

## Gate C. 응답 종결 자기검증

모든 응답 마지막에 아래 블록을 출력한다. 하나라도 충족하지 못하면 재수행한 뒤 종결한다.

```markdown
### ✅ 자기검증 (Self-Check)
- 트리거 매칭: [Path-? ✅]
- Gate B 통과: [선행 문서 Read + 선행 작업 Y/N — N이면 이유]
- 가정으로 채운 숫자: [없음 ✅ / 있음 — 항목]
- 돌발 호재 확인 (Path-O/V 한정): [확인함 ✅ / 불필요 ✅ / 생략 — 이유]
- 규칙 위반: [없음 ✅ / 있음 — 규칙]
```

세션 중 같은 규칙을 두 번 위반하면 라우터 또는 관련 정본의 구조화 개선안을 제안한다.

## 기본 위치

- 작업 디렉터리: `C:\Users\geunt\OneDrive\Desktop\주식`
- ETF 원본 v1 보존: `data/etf_holdings/`
- ETF 검증 스냅샷 정본: `data/normalized/etf_holdings/`
- 옵션 로그: `data/options_log/`
- 실적 로그: `data/earnings_log/`, `data/earnings_prefill/`
- 애널리스트 콜 로그: `data/analyst_log/`
- 야간 데스크 매매 기록: `data/journal/` (Git 제외 개인 데이터)

## 규칙 변경 원칙

새 요구사항은 계산·검증이면 코드, 라우팅이면 이 파일, 도메인 규칙이면 해당 정본 문서, 변경 이유와 시행일이면 `docs/operations_changelog.md`에 한 번만 기록한다.
