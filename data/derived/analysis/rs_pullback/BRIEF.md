# 상대강도 + 눌림 연구 브리프 (2026-10-13, 사전 등록)

## 질문(사용자)
트레이딩뷰 Pine 스크리너 지표 '20260615'(열: Score, State 2 Strong 1 Watch, RS Short, RS Main, Latent, Fade Gap)의 컨셉
"QQQ 대비 상대적으로 강한 종목을 찾으면서, 그중 눌려 있는 것을 조합해 매수"를 우리 매수·매도(데스크 7종목, 추세 등급 계좌)에
반영할 만한 게 있는지 연구한다. 코드를 똑같이 재현할 필요는 없다(2026-10-13 사용자).

## 지표에서 확인된 사실 (스크린샷 역산, `results/indicator_shots_2026-07.csv` 118행)
- RS Main = 100×(C/C[4] − QQQ/QQQ[4]) — 4거래일 QQQ 대비 초과수익률(종가 78행 평균 오차 0.065).
- Fade Gap = max(0, Latent − RS Main) (118행 전부).
- Latent: 며칠 유지되다 바뀌는 '최근 상대강도 고점'류 값. 가장 가까운 근사는 고가 기준 4일 초과수익 100×(H/C[4] − QQQ_H/QQQ[4])의 최근 4~5봉 최고(평균 오차 1.1, 정확 일치 19%).
- RS Short: 일봉 단순식으로는 재현 안 됨. 당일 봉 안 종가 위치(고가·저가 대비)가 섞인 짧은 상대강도로 보인다.
- State(2 Strong/1 Watch/0)·Score(0~78, 0에서 잘림)는 위 열만으로 설명 안 되는 숨은 항목이 있다(선형 R² 0.72).

## 공통 자료·채점 (반드시 이것을 쓴다)
- 패널: `data/derived/analysis/trend_grade/code/harness.py`의 `load_panel()`(두 펀드 미국 보유 이력 150종목 + QQQ, 분할 반영 일봉, 2023-01-03~2026-10-09).
  `sys.path.insert(0, <trend_grade/code>)` 후 `import harness`. 단, `harness.save_results`는 trend_grade 폴더에 쓰므로 쓰지 않는다. 결과는 이 폴더 `results/`에 쓴다.
- 시점: t일 종가까지로 신호 → t+1 시가 매수 → t+1+h 시가 매도(h = 5·10·20). `harness.forward(h)`, `harness.eligible()`, `harness.universe_mask(P)`(점수 안 단면 순위용), `harness.no_lookahead(fn)`(미래 정보 시험)을 쓴다.
- 구간: front 2026-01-02~05-29(고르는 구간), back 2026-06-01~10-09(확인 구간), y2025·y2024(보조, 2026 보유 이력 전체라 생존 편향). `harness.window_dates(name, h)`.
- 초과수익은 그날 eligible 종목 평균 대비(%p, 날짜 평균), 날짜별 값의 Newey-West t(`harness.newey_west_t`, lag=h).
- 데스크 7종목 규칙 재현: `data/derived/analysis/rsi_breakout/code/deskx.py`의 `Engine`/`Params`(기본값 = desk.strategies.run_model과 같은 경로, check()로 확인). desk/ 코드는 고치지 않는다.
- 거래 조건(docs/desk_system.md §1): 수수료 0, 호가 차이 별도(민감도로 편도 0.1% 보고), 주문 가능 한국 08:00~24:00.

## 공통 정의 (사전 등록)
- ex(n) = 100×(C/C[n] − QQQ/QQQ[n]). RS Main = ex(4).
- 강도 S_L = ex(L), L ∈ {20, 60, 120}. '강함' = 그날 universe_mask 안 백분위 ≥ 0.70.
- 눌림 P (클수록 더 눌림), 12개:
  - P1(n) = −ex(n), n ∈ {3, 4, 5}
  - P2(K) = Fade = max(ex(4), 최근 K봉) − ex(4), K ∈ {5, 10, 20}   (지표의 Latent/Fade Gap 대응, 종가 기준)
  - P3(K) = 상대강도선 낙폭 = 1 − R/max(R, K봉), R = C/QQQ, K ∈ {5, 10, 20}
  - P4(K) = 가격 낙폭 = 1 − C/max(C, K봉), K ∈ {5, 10, 20}

## 보고 규칙 (AGENTS.md 연구 보고 규칙)
- 합격 기준은 2026 장세: front로 고르고 back으로 확인. 2025는 보조. 다른 해 실패는 한두 줄.
- 모든 변수는 t 종가까지 아는 값. 시험한 개수·격자를 숫자로 밝힌다. 대량 탐색의 선택편향을 공개한다.
- 한 종목·하루 쏠림(SNDK, 7월 급락·반등일)을 반드시 점검한다.
- 결과 파일: `results/<family>.json`(+ 필요하면 .md), 코드: `code/<family>*.py`. 다른 갈래 파일은 건드리지 않는다. git 커밋은 하지 않는다(메인 세션이 한다).
