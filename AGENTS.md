# 성장기반 리서치 센터(GBRC) 운영 라우터

> 활성 기준일: **2026-10-08**  
> Codex와 Claude가 함께 따르는 유일한 운영 규칙이다. Codex는 이 파일을 직접 읽고, Claude는 `CLAUDE.md`가 이 파일을 불러와 읽는다.  
> 이 파일은 트리거·라우팅·공동 작업 규칙만 관리한다. 계산·출력·검증 세부 규칙은 각 도메인 문서를 정본(SSoT)으로 삼는다.

## 현재 운영 모드

- 매일 기본 업무는 TIME 글로벌AI인공지능액티브 `456600`과 KoAct 미국나스닥성장기업액티브 `2ETFQ1`의 펀드 추종뿐이다.
- Path-B와 Path-F는 ETF-only다. 옵션, 옵션 순위, 옵션×ETF 등급, 개별 종목 밸류에이션/PER를 섞지 않는다.
- Path-B와 Path-F에서는 `options_scanner.py`, `options_tracker.py`, PER 검색·계산을 실행하지 않는다.
- 옵션은 Path-O 또는 사용자의 명시적 옵션 요청, 실적은 Path-E/V/S/I, 뉴스는 Path-N, 애널리스트 추적은 Path-A에서만 수행한다.
- 섹터·종목 분류의 코드 정본은 `watchlist.py`다. 문서에 워치리스트를 중복 기재하지 않는다.
- TradingView MCP는 대화 중 조회(Path-E/V/Q)에만 쓴다. 스크립트, `automation/` 프롬프트, 옵션 스코어 입력에 연결하지 않는다(약관 3조 display-only·25조 비자동화 개인 용도). 사용 규칙은 `docs/earnings_framework.md`를 따른다.

## 정본 문서

| 범위 | 정본 |
|---|---|
| 트리거·라우팅·공동 작업 규칙·자기검증 | `AGENTS.md` |
| 애널리스트 신호 수집·기록·채점·행동 규칙 | `docs/analyst_system.md` |
| 두 펀드 수집·비교·성과·출력·저장 | `docs/etf_system.md` |
| 옵션 스코어·복기·돌발 재료 검증 | `docs/options_system.md` |
| 실적 분석·검증·성적표·개선 루프 | `docs/earnings_framework.md` |
| 섹터·종목 매핑 | `watchlist.py` |
| 운영 규칙 변경 이력 | `docs/operations_changelog.md` |
| 투자 브리핑 전달·활용 | `docs/investment_briefing_playbook.md` |
| 사용자 매매 조건, 야간 데스크(7종목 규칙·주문 계산·매매 기록·멈춤 규칙), AI 탐방·거인의 어깨·급락 후 담기 탭, AI 산업 지도 휴대폰 설명 사이트(Vercel) | `docs/desk_system.md` |

규칙이 부딪히면 ① 사용자의 현재 요청 ② 이 파일 ③ 해당 Path 정본 ④ `docs/operations_changelog.md`의 과거 이력 순으로 따른다. `docs/archive/`는 명시적으로 요청받지 않으면 활성 규칙으로 쓰지 않는다.

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
| **Path-B / Path-F** | `docs/etf_system.md` | `py gbrc.py briefing` 단일 명령으로 건너뛴 평일 채우기→TIME·KoAct 수집→검증→비교→내 7종목 점검·실매매 합의·관찰군 요약→가격 성과 비교를 수행하고 ETF-only 보고서를 출력한다. 누적 브리핑 요청이면 이어서 `py gbrc.py etf cumulative`로 마지막 전달 브리핑 이후 구간을 한 보고서로 묶는다. 실매매·매매 타이밍·수익 구간 분석 요청이면 `py gbrc.py etf trades --fund <time\|koact>`를 실행한다. 매매는 비중이 아니라 수량 기준 실매매로만 말한다. 세부 순서와 전 항목은 ETF 정본을 따른다. 야간 데스크 요청이면 `docs/desk_system.md`도 읽고, 데스크 코드만 고칠 때는 briefing 대신 그 §6 검증을 통과시킨다. |
| **Path-E** | `docs/earnings_framework.md` | `py earnings_prefill.py <TICKER>` → 누락값을 정본의 데이터 소스 순서대로 보완 → 6단계 분석 → `data/earnings_log/YYYY-MM-DD_<TICKER>.json` 저장. 시총·컨센서스·전년값을 가정으로 채운 채 분석하지 않는다. |
| **Path-V** | `docs/earnings_framework.md` | 해당 로그 로드 → yfinance D+1/D+5/D+30 수익률 → 오차가 큰 종목의 돌발 재료 WebSearch → `verification` 갱신. 검색 없이 분석 실패로 단정하지 않는다. |
| **Path-O** | `docs/options_system.md` | 전일 로그와 현재 가격 매칭 → Top/Bottom 평균 수익률 → 역전 Top 3 각각 WebSearch → Tier-1/2/3 분류와 결론. 돌발 재료 확인 없이 가중치 변경을 제안하지 않는다. |
| **Path-S** | `docs/earnings_framework.md` | 전체 실적 로그의 적중률과 평균 수익률 집계. |
| **Path-I** | `docs/earnings_framework.md` | 검증 5건 이상일 때만 개선 후보 제안 → 사용자 확정 후 관련 정본과 변경 이력 갱신. `CLAUDE.md`에 독립 규칙을 추가하지 않는다. |
| **Path-N** | 없음 | WebSearch 4회: 빅테크 → AI반도체 → 메모리 → AI인프라. 오늘 → 어제 → 이번 주 순으로 확인하고 한글로 출력. |
| **Path-A** | `docs/analyst_system.md` | 업데이트: WebSearch로 3인(가브리엘·윌슨·카토네) 신규 발언 수집 → 신규 콜 `data/analyst_log/` 기록 → `py analyst_tracker.py verify` → 이전 전망 대비 변화·행동 조건 표 출력. 신규 없으면 "신규 없음" 종결. 성적표: `py analyst_tracker.py scorecard`. 3인 의견만으로 매도 제안을 하지 않는다. |
| **Path-Q** | 해당 영역 정본 | 요청 자체를 수행한다. 다루는 영역에 정본이 있으면 먼저 읽고 그 검증을 통과시킨다: AI 종목 전고점 거리 `py ai_ath_gap.py`(유니버스 `watchlist.AI_UNIVERSE`, 결과 `data/derived/analysis/ai_ath_gap/`), AI 탐방 갱신 `docs/desk_system.md` §8.1, 설명 사이트 `py -m aimap deploy` §8.3, 거인의 어깨 §9, 급락 후 담기 §10. 연구·백테스트는 아래 연구 보고 규칙을 따른다. 매수 추천·주문 계산은 하지 않는다. |

## Gate C. 응답 종결 자기검증

모든 응답 마지막에 아래 블록을 출력한다. 하나라도 충족하지 못하면 재수행한 뒤 종결한다.

```markdown
### ✅ 자기검증 (Self-Check)
- 트리거 매칭: [Path-? ✅]
- Gate B 통과: [선행 문서 Read + 선행 작업 Y/N — N이면 이유]
- 가정으로 채운 숫자: [없음 ✅ / 있음 — 항목]
- 돌발 호재 확인 (Path-O/V 한정): [확인함 ✅ / 불필요 ✅ / 생략 — 이유]
- 규칙 위반: [없음 ✅ / 있음 — 규칙]
- 커밋: [커밋 해시 ✅ / 추적 파일 변경 없음 ✅ / 미커밋 — 이유]
```

세션 중 같은 규칙을 두 번 위반하면 라우터 또는 관련 정본의 구조화 개선안을 제안한다.

## 공동 작업 규칙 (Codex·Claude)

두 에이전트가 같은 폴더를 승인 없이, 때로는 같은 요청으로 동시에 고친다. 파일을 바꾸는 작업은 아래 순서를 지킨다.

| 시점 | 필수 |
|---|---|
| 시작 | `git status --short`로 커밋되지 않은 다른 작업을 확인한다. 내가 고칠 파일에 남의 변경이 있으면 덮어쓰지 않고 그 위에 고친다. |
| 공유 파일을 고치기 직전 | 정본 문서, `docs/operations_changelog.md`, `watchlist.py`, `gbrc/`, `desk/`는 고치기 직전에 다시 읽는다(세션 중에 다른 에이전트가 바꿀 수 있다). 같은 절이 동시에 바뀌고 있으면 새 절로 나눈다. 같은 계산의 두 번째 정본 모듈을 만들지 않는다. |
| 사실·결정 기록 | 사용자가 알려 준 사실(매매 조건 등)과 결정은 해당 정본 한 곳에 쓴다. 에이전트 개인 메모리(Claude 자동 메모리, Codex 메모리)에만 두지 않는다. 다른 에이전트는 그 메모리를 보지 못한다. |
| 코드 | 실행 경로에 미구현 자리표시(`NotImplementedError`, 사용자 작성 대기 TODO)를 남기지 않는다. 설계 선택은 근거로 직접 정해 구현하고 기준값과 근거를 정본에 남긴다. 손실 한도처럼 사용자 개인 사실이 필요한 값만 묻는다. |
| 정리 | 참조 0건·중복·빈 템플릿인 문서와 코드는 발견한 작업에서 `git rm`으로 지운다. 되살릴 때는 git 이력을 쓰고, 보관용 사본을 만들지 않는다. |
| 검증 | 코드를 바꿨으면 `py -m unittest`가 전부 통과해야 한다. `py -m pytest`와 같은 테스트를 모으며, 모듈 함수 방식 테스트 파일에 `load_tests` 훅이 빠지면 `tests/test_suite_contract.py`가 실패한다. 영역별 추가 검증은 해당 정본을 따른다(데스크 §6, 탐방 §8.1, 거인의 어깨 §9, 급락 후 담기 §10). |
| 종결 | 바꾼 파일만 경로를 지정해 커밋한다: 새 파일은 `git add <경로>`, 그다음 `git commit -m "<영역>: <요약> (codex\|claude)" -- <바꾼 경로…>`. 루틴 실행이 만든 추적 데이터(`data/normalized/` 스냅샷 등)는 `data: <날짜> <내용>` 커밋으로 따로 넣는다. `git add -A`, `git add .`, 푸시, `reset --hard`·`checkout -- .`·`clean`은 쓰지 않는다. |

## 연구 보고 규칙

전략·백테스트·요인 분석 결과를 보고할 때 적용한다.

| 항목 | 규칙 |
|---|---|
| 평가 구간 | 합격 기준은 현재 장세(2026년)에서의 성과다. 앞 구간으로 고르고 뒤 구간으로 확인하는 연도 내 전방 검증을 주 근거로 쓰고, 2025년은 보조 확인으로 쓸 수 있다. 모든 해에 통하는 규칙을 찾지 않는다(과적합). 다른 해의 실패는 장세 전환 위험으로 한두 줄만 적는다(2026-09-27·29 사용자 결정). |
| 변수 시점 | 근거 변수마다 언제 알 수 있었는지 적는다. 사건 뒤에야 알 수 있는 값(사후 공매도 잔고, 지금 PER)은 설명용으로 표시하고 매매 근거로 쓰지 않는다. 사전 값으로 다시 잰다(예: `eps_trend`의 90·60일 전 컨센서스, 급락 전 가격으로 계산한 PER, 앞 구간 값으로 뒤 구간 수익 맞히기)(2026-10-06 사용자 원칙). |
| 탐색 범위 | 결론과 함께 시험 범위(개수·간격·조합 수)를 숫자로 밝힌다. 시간봉끼리는 평활·창 길이를 거래일처럼 같은 단위로 맞춘다. 대량 탐색의 선택편향을 공개한다. |
| 거래 조건 | 미국 주식·수수료·주문 가능 시간은 `docs/desk_system.md` §1의 사용자 매매 조건을 따른다. |

## 기본 위치

- 작업 디렉터리: `C:\Users\geunt\OneDrive\Desktop\주식`
- ETF 원본 v1 보존: `data/etf_holdings/`
- ETF 검증 스냅샷 정본: `data/normalized/etf_holdings/`
- 옵션 로그: `data/options_log/`
- 실적 로그: `data/earnings_log/`, `data/earnings_prefill/`
- 애널리스트 콜 로그: `data/analyst_log/`
- 야간 데스크 매매 기록: `data/journal/` (Git 제외 개인 데이터)

## 규칙 변경 원칙

새 요구사항은 계산·검증이면 코드와 테스트, 라우팅·공동 작업 규칙이면 이 파일, 도메인 규칙이면 해당 정본 문서, 변경 이유와 시행일이면 `docs/operations_changelog.md`에 한 번만 기록한다.

- 규칙은 산문 대신 표·체크리스트·스크립트·테스트로 쓴다. 기억에 기대는 문장은 가장 약한 안전장치다.
- 사용자 요청이 정본 원칙과 부딪히면 한 번 확인한 뒤, 바뀐 범위만 좁혀 정본과 변경 이력을 고친다. 기존 원칙이 그대로 적용되는 범위는 문장에 남긴다.
- `CLAUDE.md`, `.claude/`, `.codex/`, 에이전트 메모리에는 프로젝트 규칙을 두지 않는다.
