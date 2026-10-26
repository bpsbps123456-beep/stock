# W1. 실제 기관의 리밸런싱·매매 주기 (2026-10-12)

결론: 큰 기관은 대개 **분기 단위로 정기 점검**하지만 그 사이를 **순위 완충, 절반만 이동, 회전 상한, 매월 조건부 점검**으로 메운다. 반대로 우리가 따라가는 두 펀드는 **매일 매매**하며 연간 회전율이 미국 액티브 ETF 평균의 10배를 넘는다. 따라서 등급은 매일 계산하고, 등급 변경은 완충을 둬서 확정하는 방식이 맞다.

검증 표시: **원문** = 1차 자료를 직접 열어 확인함 / **보도** = 기사 원문으로 확인함 / **요약** = 검색 요약에만 있고 원문은 확인 못 함. 웹 검색은 25회를 썼다.

## 1. 지수·패시브

| 대상 | 주기 | 장치 | 숫자 | 출처 | 검증 |
|---|---|---|---|---|---|
| S&P 500 | 구성 종목 변경은 수시(정기 재구성 없음), 주식 수·유동비율은 분기마다 갱신(3·6·9·12월 셋째 금요일 마감 후) | 편입 기준은 새로 넣을 때만 쓰고 유지 조건으로는 쓰지 않음. 잠시 어겨도 바로 빼지 않아 회전을 줄임 | 분기 기준일은 둘째 금요일 | [S&P U.S. Indices Methodology 2024-05 사본](https://www.betashares.com.au/wp-content/uploads/2024/07/sp-500-equal-weight-currency-hedged-etf-index-methodology-1.pdf) (spglobal 원본은 403) | 원문 |
| Nasdaq-100 | 12월 연 1회 재구성 + 3·6·9월 분기 리밸런싱 | 75위 안은 자동 편입, 기존 종목은 100위 안이면 유지하고 101~125위도 조건부 유지. 분기에는 125위 밖만 뺌. 40위 안 신규는 즉시 편입. 단일 종목 24% 초과, 4.5% 초과 종목 합 48% 초과 시 특별 리밸런싱 | 재구성 기준일 11월 말, 분기 기준일 2·5·8월 말 | [Nasdaq-100 방법론 2026](https://indexes.nasdaqomx.com/docs/Methodology_NDX.pdf) | 원문 |
| MSCI 글로벌 지수 | 2·5·8·11월 분기 검토(2023-11부터 '분기 종합 검토'로 일원화) | | | [MSCI Momentum 방법론 2025-07 변경 이력](https://www.msci.com/indexes/documents/methodology/2_MSCI_Momentum_Indexes_Methodology_20250725.pdf) | 원문 |
| Russell US | **2026년부터 반기(6월·12월)**, 그전에는 연 1회 6월 | 바꾼 이유는 최근 5년 변동성 확대로 연 1회 재구성이 시장 변화를 못 따라가서 | 2025-01-16 발표. 2026년 6월분은 6/26 마감 후 적용, 12월분은 순위 기준일 10/30·적용일 12/11. 2026년 6월 재구성일 마감 무렵 NYSE에서 $219.9B, Nasdaq에서 $334.0B 거래 | [LSEG 보도자료 2025-01-16](https://lseg.com/en/media-centre/press-releases/ftse-russell/2025/russell-us-indexes-move-to-semi-annual-reconstitution), [LSEG 재구성 안내](https://www.lseg.com/en/ftse-russell/russell-reconstitution) | 원문 |
| 거래 쏠림(종가) | 지수 변경분은 적용일 종가에 체결 | 마감 동시호가 | 2026년 1분기 NYSE 마감 경매 일평균 6.055억 주($43B). 2026-03-20(S&P 분기 리밸런싱과 쿼드러플 위칭이 겹친 날)에 35.7억 주·$230.5B로 신기록 | [NYSE 2026-04-08](https://www.nyse.com/research/insights/behind-the-record-volumes-a-hidden-opportunity) | 원문 |
| 마감 경매 비중 | | | 평일 약 10%, 리밸런싱일 26%라는 수치는 위 NYSE 페이지에 없음 | — | 요약, 확인 못 함 |

## 2. 팩터·모멘텀 ETF와 관련 연구

| 대상 | 주기 | 장치 | 숫자 | 출처 | 검증 |
|---|---|---|---|---|---|
| **MSCI Momentum 지수** | **2025-08부터 분기(2·5·8·11월)**, 그전에는 반기(5·11월). 조건부 리밸런싱은 매월 점검 | ① 점수: 최근 1개월을 뺀 6·12개월 초과수익을 3년 주간 변동성으로 나눔 ② 순위 완충: 목표 종목 수의 50%(500종목이면 250위 안 우선, 기존 종목은 251~750위면 유지) ③ **회전 완충 50%**: 새 비중 쪽으로 절반만 이동, x+(y−x)/2, 편출에는 적용 안 함 ④ 조건부: 매월 T−9일에 모지수 3개월 변동성의 월 변화가 과거 95백분위를 넘으면 6개월 모멘텀만으로 임시 재구성 | | [MSCI 2025-07](https://www.msci.com/indexes/documents/methodology/2_MSCI_Momentum_Indexes_Methodology_20250725.pdf) §2.2, §3.1, 부록 III | 원문 |
| iShares MTUM | 분기 재구성 | 재구성 1회당 단방향 회전 30% 상한, 125종목, 종목당 5% 상한 | 회계연도(2025-07-31 종료) 회전율 116% | [요약 투자설명서 2025-11-28](https://www.ishares.com/us/literature/summary-prospectus/sp-ishares-edge-msci-usa-momentum-factor-etf-7-31.pdf) | 원문 |
| AQR Large Cap Momentum | '월 1회 리밸런싱 예상'은 원문 확인 못 함 | 기계적으로 리밸런싱하지 않고 기회가 될 때 매매해 비용을 줄임 | 회전율 88%(FY2017), TM판 60%(FY2020) | [497K 2018-01-29](https://www.sec.gov/Archives/edgar/data/0001444822/000119312518023466/d505877d497k.htm), [497K 2021-01-29](https://www.sec.gov/Archives/edgar/data/1444822/000119312521022726/d33810d497k.htm) | 원문(월 1회만 요약) |
| Invesco DWA Momentum(PDP) | 분기 재구성 | 벤치마크 대비 상대강도를 매일 계산하고 포인트앤피겨로 작은 등락은 무시. 약 100종목 | | [Nasdaq 2018-07-18](https://www.nasdaq.com/articles/dorsey-wright-technical-leaders-index-2018-07-18) | 요약(원문 시간 초과) |
| Novy-Marx·Velikov 연구 | 월 리밸런싱 / 매달 1/3씩 나눠 리밸런싱 / 보유·매수 문턱 차등 | 보유·매수 문턱 차등이 가장 효과적인 단순 비용 절감법 | 모멘텀: 월 리밸런싱은 단방향 회전 34.52%/월, 순수익 0.68%/월. 1/3 분할은 16.66%, 0.91%. 문턱 차등은 18.82%, 0.85%. 둘을 함께 쓰면 15.48%, 1.01%. 산업 모멘텀은 회전 90.13%/월에 순수익 −0.29% | [NBER WP 20721 (2014-12)](https://www.nber.org/system/files/working_papers/w20721/w20721.pdf) 표 3·7·8·10 | 원문 |
| 리밸런싱 날짜 운 (Newfound) | | 같은 전략도 리밸런싱 날짜만 달라 성과가 갈림. 날짜를 나눈 N개 하위 포트폴리오로 운을 1/N로 줄임. 주기는 신호가 소멸하는 속도와 비용으로 정해야 함 | 팩터 지수에서 연 100bp를 자주 넘음 | [thinknewfound](https://www.thinknewfound.com/rebalance-timing-luck) | 원문 |
| 서서히 쌓이는 정보 (Da·Gurun·Warachka) | | 작은 변화가 연속으로 쌓이는 정보에는 투자자가 덜 주목해서 모멘텀이 더 강하고 오래 감 | | [RFS 27(7) 2014](https://doi.org/10.1093/rfs/hhu003) | 요약(초록) |

## 3. 헤지펀드·CTA·퀀트

| 대상 | 주기 | 내용 | 출처 | 검증 |
|---|---|---|---|---|
| 추세추종 (AQR Hurst·Ooi·Pedersen) | 1·3·12개월 추세 신호를 매월 합산 | 연 변동성 목표 10% | [JPM 2017](https://www.trendfollowing.com/whitepaper/Century_Evidence_Trend_Following.pdf) | 원문 |
| Man AHL 'Need for Speed' | 여러 속도의 이동평균 교차를 동시에 운용 | 느린 모델은 장기 샤프가 높고, 빠른 모델은 추세가 꺾일 때 손실을 빨리 끊어 주식 급락기에 유리. 대신 회전과 비용이 큼 | [Man 2023-01-04](https://www.man.com/insights/need-for-speed-trend-following) | 원문 |
| Man AHL 보유 기간 | 빠름 2~3개월, 느림 6개월 | | [Man deep dive](https://www.man.com/insights/deep-dive-trend-following) | 요약, 확인 못 함 |
| 통계적 차익 (Avellaneda·Lee) | 매일 종가에 신호 계산·매매 | 60거래일 창으로 추정하고 평균회귀 시간이 30일 미만인 종목만 거래. 평균 기대 회귀 시간은 약 7일 | [Quant. Finance 2010](https://traders.studentorg.berkeley.edu/papers/Statistical%20arbitrage%20in%20the%20US%20equities%20market.pdf) | 원문 |
| 멀티매니저 (Millennium) | 손실 한도를 상시 감시 | 고점 대비 −5%면 위험 한도 절반, −7.5%면 팀 청산. PM 교체율 연 15~20% | [Net Interest 2023-09-15](https://www.netinterest.co/p/peak-pod) | 보도, 1차 확인 못 함 |

## 4. 연기금·자산배분

| 대상 | 방식 | 숫자 | 출처 | 검증 |
|---|---|---|---|---|
| Vanguard | 일~연 단위 주기와 1~15% 범위를 조합해 시험 | 월·분기처럼 너무 잦지도, 2년처럼 너무 드물지도 않은 주기가 최적. 세금 손실 수확을 하지 않는 투자자에게는 연 1회가 최적(목표는 위험 통제와 비용, 신호 포착이 아님) | [Vanguard 2022-10](https://www.vanguardmexico.com/content/dam/intl/americas/documents/latam/en/2022/10/mx-sa-2558523-rational-rebalancing-an-analytical-approach.pdf) | 원문 |
| 노르웨이 NBIM | 월말 점검 + 범위식 | 2012년부터 주식 비중이 목표에서 4%p 넘게 벗어나면 리밸런싱. 2018-08-28 서한에서 ±2%p로 좁히고 더 점진적으로 복귀하자고 권고. 현행 2%p는 확인 못 함 | [NBIM 2018-08-28](https://www.nbim.no/en/news-and-insights/submissions-to-ministry/2018/the-rule-for-rebalancing-the-equity-share-in-the-government-pension-fund-global/) | 원문 |
| 일본 GPIF | 범위식 | 4개 자산 각 25%. 2025-04-01부터 허용 폭은 국내채 ±6, 해외채 ±5, 국내주 ±6, 해외주 ±6, 글로벌 채권·주식 ±9(%) | [GPIF 2025](https://www.gpif.go.jp/en/topics/Adoption_of_New_Policy_Portfolio_details_2_en.pdf) | 원문 |
| 분기말 자금 (JPMorgan) | 분기말 | 2026-06-30 앞 최대 $165B 주식 매도 추정(GPIF 약 60, 미국 DB 연금 약 55, NBIM 약 40, SNB 약 25, 단위 $B). 월 단위로 움직이는 균형형 펀드는 약 $15B 매수 | [Investing.com 2026-06-18](https://www.investing.com/news/stock-market-news/investors-could-sell-165bn-worth-of-stocks-amid-june-quarterend-rebalancing-jpm-4749772) | 보도 |
| 분기말 미국 연금 (Goldman) | 분기말 | 약 $30B, 2000년 이후 95백분위 | — | 요약, 확인 못 함 |

## 5. 액티브 펀드

| 항목 | 숫자 | 출처 | 검증 |
|---|---|---|---|
| 미국 액티브 ETF 회전율 | 평균 연 36%(패시브 31%). 중위값은 2020년 49.5%에서 2024년 35%로 하락 | [SEC DERA 2026-02](https://www.sec.gov/files/dera-fast-growing-mrkt-2602.pdf) 13~14쪽 | 원문 |
| 13F | 분기 종료 후 45일 안에 제출, 운용 자산 $100M 이상 | [SEC FAQ (2026-09-28 갱신)](https://www.sec.gov/rules-regulations/staff-guidance/division-investment-management-frequently-asked-questions/frequently-asked-questions-about-form-13f) | 원문 |
| ICI 주식형 펀드 회전율 | 2024·2025·2026 Fact Book에 수치가 실려 있지 않음 | [ICI 2026](https://www.icifactbook.org/pdf/2026-factbook.pdf) | 확인 못 함 |

## 6. 한국 액티브 ETF: 운용사 설명과 실측

- **TIME**: 운용사는 매일 시장 데이터를 분석해 주도주를 교체한다고 설명한다. 반도체 집중에서 AI 소프트웨어·전력 인프라를 거쳐 스토리지·로봇으로 무게중심을 옮겼다. [뉴스핌 2026-01-23](https://www.newspim.com/news/view/20260123000806)(보도)
- **KoAct**: 혁신 기술 트렌드와 주도기업 변화에 맞춰 액티브하게 운용하고, 차세대 나스닥100 후보를 발굴한다고 설명한다. 매매 주기는 밝히지 않았다. [삼성액티브 2026-03-05](https://www.samsungactive.co.kr/insight/koactinsight/research-view.do?seqn=222)(원문)
- **실측**: `code_fund_trades.py` → `results_fund_trades.json`. 2026-01-02~10-08 CU당 수량 PDF를 썼다. 매매 판정은 `gbrc.etf.flows`의 ±5% 선별 수량 변화를 따랐다(설정·환매 공통 배율과 분할은 제거). KoAct는 품질 미통과 스냅샷 5일을 뺐다.

| 항목 | TIME | KoAct |
|---|---|---|
| 비교한 날 수 | 186 | 181 |
| 매매(±5% 이상·신규·편출)가 있던 날 | 61.8% | 56.9% |
| 신규·편출이 한 건이라도 있던 주 | 80% | 78% |
| 하루 단방향 회전(평균 / 중위) | 1.55%p / 0.69%p | 2.66%p / 0.72%p |
| 연환산 단방향 회전 | 약 389% | 약 671% |
| 매월 마지막 2일의 회전 비중(전체 날 중 비중) | 9.1% (10.8%) | 12.0% (11.0%) |
| 3·6·9월 말 2일의 회전 비중(날 비중) | 0.8% (3.2%) | 3.7% (3.3%) |
| 표본 안에서 시작과 끝이 모두 관측된 보유 구간의 중위(사분위) | 24거래일 (12.5~56) | 19.5거래일 (10~36) |
| 9개월 동안 거친 종목 수 / 끝까지 보유한 종목 수 | 103 / 10 | 132 / 9 |
| 편출 종목(직전 20일 최대 비중 1~3%): 편출 전에 미리 판 비율의 중위, 매도일 중위 | 15%, 1일 (n=47) | 0%, 1일 (n=70) |
| 편출 종목(최대 비중 3% 이상): 미리 판 비율 중위, 매도일 중위 | **61%, 4일** (n=9) | 1%, 2일 (n=15) |
| 신규 종목: 첫날에 산 양 / 20일 뒤 보유량(중위) | 52% | 66% |

정리하면 두 펀드는 **매일 조금씩, 그러나 몇 주 단위로 종목을 통째로 바꾼다.** 월말·분기말에 매매가 몰리지 않는다. 1~3% 비중 종목은 대개 하루에 정리한다. 비중이 줄어 보이는 '서서히 빠짐'은 주로 가격 하락 때문이다. 3% 이상 큰 종목은 TIME이 4일 정도에 걸쳐 먼저 줄인다.

## 등급 갱신 주기 설계에 주는 함의

1. 기관의 정기 점검 단위는 분기·반기지만 사이를 완충과 조건부 점검으로 메운다. 그러니 등급은 매일 계산하고, 등급 변경은 진입 문턱과 이탈 문턱을 다르게 둬서 확정한다.
2. 두 펀드는 미국 액티브 ETF 평균(연 36%)보다 10배 넘게 빨리 매매하고(389%·671%), 보유 구간 중위가 20~24거래일이다. 월·분기 등급은 너무 느리다. 최소 실용 단위로는 '주'가 후보이고, 확인은 매일 한다.
3. '서서히 빠짐'은 가격 하락과 큰 종목의 분할 축소로 나타난다. 등급 옆에 10일 평활 점수의 기울기(하락 속도)를 함께 보여 줄 가치가 있다.
4. 리밸런싱 날짜 운이 연 100bp를 넘을 수 있다. 특정 요일에 한 번 갱신하기보다 매일 계산하고 5~10일 평활이나 날짜 분할로 운을 줄인다.
5. 문턱 차등은 모멘텀 회전을 월 34.5%에서 18.8%로 줄이고 순수익을 월 0.68%에서 0.85%로 올렸다. 'S 진입은 상위 10%, 유지는 상위 20%'를 백테스트 후보로 둔다.
