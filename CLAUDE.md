# 성장기반 리서치 센터 (GBRC)
> FY27·FY28 Forward PER 기반 성장주 분석 + 옵션 포지셔닝 + 돌발 호재 감지 통합 시스템

---

## 🚦 Gate A. 트리거 식별 (모든 응답 시작 시 — 이탈 불가)

| 입력 키워드 | Path |
|---|---|
| 브리핑 / 오늘 브리핑 / 장 브리핑 | **[Path-B]** |
| 실적 분석 / 어닝 분석 / `<티커> 실적` / 실적 자료 첨부 | **[Path-E]** |
| 실적 복기 / `<티커> 복기` / 실적 검증 | **[Path-V]** |
| 옵션 복기 / 어제 복기 | **[Path-O]** |
| 기타 | **[Path-Q]** |

---

## 🗂 Gate B. 라우팅 — Path별 선행 문서 + 필수 순서

| Path | 먼저 Read할 문서 | 필수 실행 순서 | 금지 |
|---|---|---|---|
| **Path-B** | `docs/options_system.md` | ① `py daily_analysis.py` → ② `py options_scanner.py` → ③ `py options_tracker.py --fill` → ④ `py options_tracker.py --catalyst` → ⑤ 장전이면 수동 복기 (Top/Bottom 평균 + 역전 Top 3 WebSearch) → ⑥ 통합 출력 | 트래커 출력만으로 복기 종결 |
| **Path-E** | `docs/earnings_framework.md` | ① `py earnings_prefill.py <TICKER>` → ② WebSearch 누락 보완 → ③ 6단계 분석 출력 → ④ `data/earnings_log/YYYY-MM-DD_<TICKER>.json` 저장 | 시총·컨센서스·전년 데이터 가정으로 채운 채 분석 진입 |
| **Path-V** | `docs/earnings_framework.md` | ① 해당 JSON 로드 → ② yfinance D+1/D+5/D+30 수익률 계산 → ③ 오차 큰 종목 WebSearch 돌발 호재 확인 → ④ verification 필드 업데이트 | WebSearch 없이 "분석 실패" 단정 |
| **Path-O** | `docs/options_system.md` | ① 어제 로그 vs 오늘 가격 매칭 → ② Top/Bottom 평균 수익률 계산 → ③ 역전 Top 3 각각 WebSearch 돌발 호재 확인 → ④ Tier-1/2/3 분류 후 결론 | 돌발 호재 확인 없이 가중치 변경 제안 |
| **Path-Q** | 없음 | — | — |

---

## ✅ Gate C. 응답 종결 자기검증

모든 응답 마지막에 아래 블록 필수 출력. 누락 자체가 위반.

```
### ✅ 자기검증 (Self-Check)
- 트리거 매칭: [Path-?]
- Gate B 통과: [선행 문서 Read + 선행 작업 Y/N — N이면 이유]
- 가정으로 채운 숫자: [없음 / 있음 — 항목]
- 돌발 호재 확인 (Path-O/V 한정): [확인함 / 불필요 / 생략 — 이유]
- 규칙 위반: [없음 / 있음 — 규칙]
```

- 모든 항목 ✅여야 종료 허용. ❌면 재수행 후 재검증.
- 세션 중 동일 규칙 2회 위반 → CLAUDE.md 구조화 반영 제안 필수.

---

## ⚙️ 기본 설정

- **작업 디렉토리**: `C:\Users\geunt\OneDrive\Desktop\주식`
- **데이터 소스**: yfinance (정량 + 기본 뉴스) + WebSearch (정밀 뉴스)
- **스크립트**: `daily_analysis.py` / `options_scanner.py` / `options_tracker.py` / `earnings_prefill.py`
- **로그**: `data/options_log/YYYY-MM-DD.json` / `data/earnings_log/YYYY-MM-DD_<TICKER>.json`
- **뉴스 검색 순서**: 오늘 → 어제 → 이번 주

## 📋 워치리스트

### 흑자기업
| 섹터 | 종목 |
|---|---|
| 빅테크 | AAPL, MSFT, AMZN, META, GOOGL |
| AI 반도체 | NVDA, AVGO, AMD, MRVL, CRDO, ALAB |
| 메모리 | MU, SNDK, WDC, STX |
| 네트워크/광통신 | ANET, CIEN, LITE, GLW |
| 반도체 장비 | LRCX, TER |
| 엔터프라이즈 SW | ORCL |

### 적자기업
| 섹터 | 종목 |
|---|---|
| AI 인프라 | CRWV, NBIS, IREN, CIFR, APLD |
| 양자컴퓨팅 | IONQ, RGTI |
| eVTOL | JOBY |
| 헬스케어 AI | TEM |
| 원자력/에너지 | UUUU, SMR, OKLO |
| 반도체 장비(적자) | AEHR |
| 반도체(적자) | INTC, NVTS |

## 💡 투자 철학 & 목표가

- 정량(FY27·FY28 Forward PER)으로 후보 선별 → 뉴스로 타이밍·확신 확보
- **흑자기업 목표가**: FY29 EPS × 20 (PER 20배)
- **적자기업 목표가**: yfinance `targetMeanPrice` (애널리스트 컨센서스)

## 🔧 하네스 엔지니어링 원칙

새 규칙 추가 시 반드시 이 순서로 고려:
1. 스크립트화 가능 → `.py` 이관
2. 트리거 매트릭스 행으로 추가 가능 → Gate B 편입
3. Gate 체크리스트 항목으로 추가 가능 → Gate A/B/C 편입
4. 위 3개 모두 불가 시에만 산문 서술

**금지**: "~해야 한다" 산문 지침. 세션 2회 이상 어긴 규칙은 산문으로 실패 판정.
