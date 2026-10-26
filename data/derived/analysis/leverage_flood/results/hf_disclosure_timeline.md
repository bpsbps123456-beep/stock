# 헤지펀드 레버리지 공개 자료 — 외부인이 언제·어떻게 알 수 있나 (2026-10-14 조사)

조사 범위: 골드만삭스·모건스탠리·JP모건 프라임브로커 주간 자료(언론 인용), OFR 헤지펀드 모니터, SEC Form PF 통계, 연준 금융안정보고서, 13F·GVIP, 하이닉스 GDR 보유 공시, 한국 공매도·대량보유 보고. WebSearch 12회 사용(한도 12회), 나머지는 WebFetch·Google News RSS·OFR 공개 API로 보완했다.
급락 기준일: ① 6/23(6/22 고점, 삼성·하이닉스 −12%, 서킷브레이커) ② 7/29(코스피 −10.84%) ③ 10/6~13(5일 −13.8%). 참고 3/4(−12.06%).

모든 숫자는 실제로 받은 값만 적었다. 받지 못한 것은 "못 받음"으로 쓰고 이유를 적었다. 기사 숫자는 기사 그대로 인용했다(정확성은 기사 책임).

## 0. 결론 (10줄)

1. 헤지펀드 레버리지 "비율" 자체를 외부인이 **매일** 보는 길은 없다. 가장 빠른 것은 골드만 프라임서비스 주간 노트가 언론(Bloomberg·Reuters·Hedgeweek·Investing.com)에 실리는 월~화요일 보도(주말 기준 T+3~5일)다.
2. 공식 통계(OFR Form PF)는 분기·약 3개월 시차다. 2026-06-30 분기 값이 9/30 보도에 이미 인용됐고, 이 세션(10/14) API에서 받았다: 총명목익스포저 45.193조$, 프라임브로커 차입 3.688조$(전분기 +23.3%), GNE/NAV 8.2배.
3. 6/23 앞: "레버리지 사상 최고" 경고는 5/18(GS 총레버리지 318.0%, 5년 최고)·5/25(13F: 총레버리지 5년 중 94백분위, 반도체 비중 사상 최고 10%)·6/2("record gross leverage")·6/9(순레버리지 1년 중 89백분위)·6/22(순레버리지 4년 최고)로 **4~5주 전부터 반복** 나왔다. 단, 6/22 같은 날 "헤지펀드가 6/12~18주 순매도 전환, 총레버리지 1년 최저권"도 나왔다 — 급락은 디그로싱이 시작된 뒤 2주 반 만에 왔다.
4. 7/29 앞: "마진콜" 보도(FT, 7/29 런던 오전 = 한국 장 마감 후)는 사건 당일이다. 대신 **매도 압력**은 미리 보였다: 6/25주 美 IT 사상 최대 주간 순매도(z −4.0), 7/9 "총레버리지 1년 최하위 10%로 하락", 7/20 Bloomberg "테크 사상 최대 속도 매도(8주 중 6주)", 7/27 기준 아시아 8일 연속 디그로싱(보도는 7/30).
5. 10/6 앞: 9/14 "TMT 2주 매수 속도 5년 중 97백분위"(쏠림 재형성), 9/30 "은행 차입 3.7조$ 10년 최대"(OFR Q2), 10/5 "헤지펀드 확신 상실". IMF(10/6~7)·ECB(10/7)·FT(10/8)·GS Mag7 22%(10/5 기준, 10/8 보도)는 **동시·사후**다. 한국 종목을 특정한 헤지펀드 매도 자료는 10/6 전에 못 찾았다.
6. 따라서 "레버리지 최고" 보도는 **경고등으로는 쓸 수 있으나 타이밍 신호로는 못 쓴다**: 5/18 경고 → 6/23(5주), 9/30 경고 → 10/6(6일), 그러나 2025-12-03 Reuters "사상 최고 근접(JPM 297.9%)" → 다음 급락 3/4(3개월).
7. 13F는 45일 시차지만 GS Hedge Fund Trend Monitor(13F 마감 +1주)의 "쏠림" 지표(반도체 비중·총레버리지 백분위)가 5/25에 나와 6/23에 4주 앞섰다. GVIP ETF는 분기 재구성이라 일별 신호가 아니다.
8. 하이닉스 GDR(HXSCL, 144A·OTC)의 13F 보유자는 1곳(Pacer Advisors, 2026-02-04, 2,979,040주)뿐 — 헤지펀드 노출 추적에 쓸모없다. 시추에이셔널 어웨어니스는 하이닉스를 "플렉스 옵션"으로 들고 있어(Yahoo 9/12) 공시에 안 잡힌다.
9. 한국: KRX 공매도 잔고는 T+2 보고·공표지만 KRX 정보데이터시스템이 2026-10 현재 로그인 필수(이 세션 확인: getJsonData.cmd → "LOGOUT")라 스크립트로 못 받았다. 대량보유(5%)는 DART 5영업일. 한국은행 통화신용정책보고서(9/10)가 SA 펀드 레버리지 4배·홍콩 레버리지 ETF 시총 20배·TRS 헤지 흐름을 사후 설명했다.
10. 실용 결론: ① 월·화 아침 Hedgeweek/Investing.com의 GS 프라임 주간 기사(총·순레버리지 백분위, 매도 z-score) ② 분기 13F 직후 GS Trend Monitor의 반도체 비중·총레버리지 백분위 ③ OFR Form PF 분기(설명용)를 3단 경고등으로 쓰되, 실제 매도 타이밍은 다른 갈래(국내 신용·반대매매·외국인 수급)와 합쳐야 한다.

## 1. 자료원별 — 빈도·시차·받는 법

| # | 자료 | 무엇을 주나 | 빈도 | 발표 시차 | 받는 법 | 이 세션 결과 |
|---|---|---|---|---|---|---|
| ① | 골드만삭스 Prime Services 주간 노트 | 총/순레버리지(%, 1년·5년 백분위), 섹터별 순매수 z-score, 롱/숏 비율, 전략별 수익률 | 주간(목요일 마감 기준) + 월간 | 언론 보도 주말 기준 T+3~5일(월·화). 월간은 다음 달 1~5일 | 비공개 노트. Hedgeweek·Investing.com·Bloomberg·Reuters 인용을 Google News RSS로 모음 (`code/gnews_headlines.py`) | `results/gnews_hf_leverage_headlines.csv` (672행, 2026-05 이후 약 200건) |
| ① | 모건스탠리·JP모건 프라임 | 총레버리지(JPM 297.9%, 2025-11말), 전략별 레버리지(JPM 퀀트 645.3%·멀티 444.3%), MS "15년 중 상위 1%" | 월간 보도 | 월말 기준 다음 달 초(Reuters 2025-12-03·2026-08-04·Hedgeweek 2026-10-05) | 비공개. 언론 인용만 | 2026년 5~10월 MS·JPM 레버리지 "수치"는 못 받음(보도에 숫자 없음; JPM 8/4는 7월 수익률만) |
| ② | OFR Hedge Fund Monitor | Form PF 집계: GNE·GAV·NAV, 규모별·전략별 레버리지, 프라임브로커/레포 차입; CFTC TFF 주간 포지션; FICC 스폰서드 레포 일별; 연준 SCOOS 분기 | Form PF 분기, TFF 주간, FICC 일별(월 1회 갱신), SCOOS 분기 | Form PF 약 3개월(2026-06-30 값이 9/30 보도에 인용, 10/14 API에 존재). TFF는 10/14 현재 9/15까지(약 4주). FICC 9/18까지(약 4주) | 공개 API `https://data.financialresearch.gov/hf/v1/series/full?mnemonic=…` (`code/ofr_hfm_fetch.py`) | `results/ofr_hfm_series.csv`(88시리즈 33,827행), `results/ofr_formpf_leverage_quarterly.csv`, `results/cftc_tff_leveraged_funds_equity_2026.csv` |
| ③ | SEC Private Fund Statistics | Form PF·ADV 집계(펀드 수, GAV·NAV, 전략별 GNE/NAV) | 분기 | 2025 Q3 보고서가 "2026-03-13 이후 제출분 미반영" → 분기말 +약 5.5~6개월 | PDF `https://www.sec.gov/files/investment/private-funds-statistics-2025-q3.pdf` (requests로 받음, 목록 페이지는 403) | 1쪽만 확인. 표 값은 못 받음(PDF 표 파싱 안 함) |
| ④ | 연준 Financial Stability Report | "총명목레버리지 사상 최고 부근", 상위 15개 펀드 대차대조표 레버리지 | 반기(2026-05-08, 2025-11-07, 2025-04-25) | 본문이 쓰는 Form PF는 2025 Q3 → 약 7개월 시차 | WebFetch `federalreserve.gov/publications/2026-may-financial-stability-report-leverage.htm` | 인용 확보 |
| ⑤ | 13F → GS Hedge Fund Trend Monitor / GVIP ETF | 13F: 분기말 +45일. GS 보고서: 13F 마감 +약 1주(2026-05-25~27, 13F 기준일 5/18). GVIP: 50종목 동일가중 근사, 분기 재구성 | 분기(보유는 일별 공개) | 13F 45일 + 1주 | stockanalysis.com/etf/gvip/holdings (10/9 기준 50종목) | 인용 확보 |
| ⑥ | SK하이닉스 GDR(HXSCL) 13F·13G | 144A GDR, OTC | — | — | fintel.io/so/us/hxscl (검색 요약만, 페이지는 403) | 13F 보유자 1곳(Pacer Advisors, 2026-02-04 제출, 2,979,040주). TipRanks "헤지펀드 보유 없음". SEC 미등록이라 SC 13G 해당 없음(규정 기준, 이 세션 추가 검증 없음) |
| ⑦ | 한국 공매도 잔고·대량보유 | KRX 공매도 잔고(종목별, 0.01%·10억 이상 보고), DART 대량보유(5%) | 일별 / 수시 | 잔고 T+2 보고·공표, 대량보유 5영업일 | KRX `getJsonData.cmd` bld `dbms/MDC/STAT/srt/MDCSTAT30502` — 2026-10-14 응답 "LOGOUT"(로그인 필수) | 못 받음(로그인 필수). 규정 시차는 자본시장법 기준 지식(이 세션 검증 안 함) |

### 1.1 OFR Form PF 분기 값 (API, 2026-10-14 수신, 단위 조$)

| 분기말 | GNE | NAV | GNE/NAV | GAV/NAV | 프라임브로커 차입 | 레포 차입 | 주식전략 레버리지(GAV가중) | 상위10 레버리지(단순평균) | SCOOS 헤지펀드 레버리지 사용(순%) |
|---|---|---|---|---|---|---|---|---|---|
| 2025-03-31 | 32.926 | 4.319 | 7.62 | 2.53 | 2.610 | 2.855 | 2.861 | 18.3 | +4.5 |
| 2025-06-30 | 35.795 | 4.546 | 7.87 | 2.60 | 2.992 | 3.122 | 3.299 | 18.0 | −22.7 |
| 2025-09-30 | 37.898 | 4.743 | 7.99 | 2.57 | 3.240 | 3.165 | 2.627 | 18.2 | 0 |
| 2025-12-31 | 39.148 | 4.849 | 8.07 | 2.60 | 3.239 | 3.380 | 2.681 | 23.9 | +5.3 |
| 2026-03-31 | 42.291 | 4.994 | 8.47 | 2.56 | 3.241 | 3.249 | 2.759 | 23.7 | +10.0 |
| 2026-06-30 | 45.193 | 5.511 | 8.20 | 2.52 | 3.688 (+23.3% QoQ) | 3.382 (+8.3%) | 2.732 | 19.2 | 0 |

언제 알 수 있었나: 2026-03-31 값은 빨라야 6월 말~9월(정확한 OFR 게시일은 페이지에 없음 — 제3자 사이트는 "9/1 공개"라 적었으나 검증 못 함), 2026-06-30 값은 9/30 이전(Yahoo 9/30 "By midyear, hedge fund borrowing from banks had reached nearly $3.7 trillion — a figure the Treasury Department says is the largest in over a decade"). 세 급락 어느 것에도 **사전** 자료가 아니다. 설명용.

### 1.2 FICC 스폰서드 레포(일별, OFR 월 1회 갱신)

6/22 1.107조$, 6/23 1.122조$, 7/28 1.021조$, 7/29 1.025조$, 9/15 1.053조$, 9/18 1.085조$(마지막). 급락일 전후 특이 변동 없음. DTCC 자체 사이트의 일별 공개(T+1)는 이 세션에서 확인 안 함.

### 1.3 CFTC TFF 레버리지 펀드 순포지션(화요일 기준, CFTC는 금요일 공표, OFR 반영 2~4주)

S&P(`TFF-LF_SP_NET_POSITION`, OFR 메타에 단위 없음 — 값 그대로): 6/16 −25.08M → 6/23 −17.78M → 6/30 −17.42M; 7/21 −15.80M → 7/28 −14.36M; 9/15 −14.42M(마지막). 순숏이 급락 직전 주에 오히려 줄었다(숏 커버). 경고 신호 아님. 파일 `results/cftc_tff_leveraged_funds_equity_2026.csv`.

## 2. 2026년 보도 타임라인 (레버리지 최고 / 순매도 / 마진콜)

날짜는 기사 자체 날짜(현지). 인용은 기사 그대로. "GS PB" = 골드만 프라임브로커 데이터.

| 날짜 | 출처 | 내용(인용) | 가장 가까운 급락과의 관계 |
|---|---|---|---|
| 2025-12-03 | Reuters(Investing.com) | GS "gross leverage … 285.2%, up about 12.4 percentage points this year"; JPM "297.9%, the highest point in the last five years and close to an all-time high"; MS "levels were higher only 1% of the time … last 15 years" | 3/4 급락 3개월 전 |
| 2026-03-23 | Reuters·Hedgeweek | "selling of emerging Asia market stocks last week was the most since April 2025"(3/19 주), "concentrated in Taiwan, Korea, and India" | 3/4 이후(사후) |
| 2026-04-05 | Investing.com | "largest reduction in information technology stock positions in a decade over the past two weeks"(GS PB) | — |
| 2026-05-18 | GS PB(finvaulta 요약) | "GS Prime Brokerage overall book Gross leverage has reached a 5-year high of 318.0%"(1년 중 100백분위) | **6/23 전 36일** |
| 2026-05-21 | Yahoo(Bloomberg) | "Semiconductors … the most heavily net-sold US subsector over the past month"; 지수 숏 "decade-long peaks"; "Kospi … briefly surpassed 8,000" | 6/23 전 33일 |
| 2026-05-25~27 | GS Hedge Fund Trend Monitor(13F 5/18 기준) | "the most elevated long portfolio weight in Semiconductors on record, at 10 per cent"; "Gross leverage ranks in the 94th percentile vs. the last 5 years"; IT net tilt +853bp "largest quarterly increase … on record" | **6/23 전 4주** |
| 2026-06-02 | AlternativeSoft(GS PB 5/30 주) | "largest weekly net long position accumulation [in tech] by hedge funds in over five years"; "Record gross leverage is a market structure risk factor" | **6/23 전 3주** |
| 2026-06-05 | (나스닥 −4.18%) Crypto Briefing 6/9 | "bought global equities at the fastest net pace in four months during the week ending June 4" | 6/5 美 1차 충격 직전 매수 |
| 2026-06-07 | Investing.com | "third consecutive week of net stock sales globally, driven by short selling"(6/5 주); 전주 "reduced tech exposure at the fastest pace in a decade" | 6/23 전 16일 |
| 2026-06-09 | FXEmpire | "net leverage at the 89th percentile of its one-year range and the fundamental long/short ratio at the 99th" | 6/23 전 2주 |
| 2026-06-22 | Hedgeweek(GS PB 6/12~18주) | "shifted back into net selling of US equities … ending a four-week streak of net buying"; "Overall leverage in US long/short strategies declined, falling to one of the lowest readings of the past year"; "net leverage remained elevated at around the 74th percentile" | **6/23 전날** — 디그로싱 진행 중 |
| 2026-06-22 | Yahoo "$165 Billion Stock Selloff Looms as Goldman Flags Rising Leverage" | GS Coppersmith "net leverage has since pushed to four-year highs"; JPM "quarter-end rebalancing could drive up to $165 billion in equity sales" | **6/23 전날** |
| 6/25 주 | GS PB(cardano 7/20 인용) | "sold US Information Technology at the fastest single-week pace in the history of the dataset, a z-score of -4.0" | 6/23 직후(사후) |
| 2026-07-09 | Yahoo(GS) | 반전 시작 "June 22"; 시스템 매니저 "fallen 3.6%"; 펀더멘털 매니저 "aggressively cutting the AI longs", 총레버리지 "into the bottom decile over the past year" | **7/29 전 20일**(이미 레버리지 축소) |
| 2026-07-19 | Fortune(Tom Lee) | "1.2 million brokerage accounts had a margin call, representing as much as 10% of Korea's accounts"; "Kospi index has plunged 27%" | 7/29 전 10일(한국 개인) |
| 2026-07-20 | Bloomberg(뉴스핌 7/21 01:01) | "최근 8주 가운데 6주 동안 테크 업종의 순매도", 누적 "약 10%… 10여년 전 이래 최대 규모" | **7/29 전 9일** |
| 7/16 주 | GS PB(cardano 7/20) | IT 노출 "19.4% gross and 14.7% net"(6월 초 5년 최고 23.4%/26.3%에서 하락) | 7/29 전 |
| 2026-07-29 09:33(런던) | Hedgeweek(FT) / 서울경제 12:23 | 마진콜(GS·JPM); "gross leverage among hedge funds increased at the fastest cumulative pace … first five months of a year since … 2016"; "16% of its prime brokerage financing book was directly linked to AI memory stocks at the end of June"; 7/28 "long-short … down approximately 1.3% … multi-strategy … 1.7%" | **7/29 당일**(한국 장 마감 후) |
| 2026-07-30 | Reuters | "Asia-focused fundamental long-short funds have fallen 18.6% … through July 28"; "reduced exposure for eight straight trading days as of July 27"; "five-day cumulative de-grossing … the largest on record"; "Kospi tumbled nearly 11% on Tuesday" | 7/29 다음날(사후; 데이터는 7/27 기준) |
| 2026-07-31 | CNBC | 시추에이셔널 어웨어니스 붕괴($45B 정점 → 약 $35B 손실, 시타델에 염가 매각) | 사후 |
| 2026-08-04~07 | Reuters(JPM) / 연합뉴스 8/7 | JPM "2026 gains dented by tech trades in July"; 다이먼 "레버리지 사상 최고…누군가 시장 흔들 위험"(제목, 본문 미확인) | 사후 |
| 2026-08-21 | CNBC / MarketWatch | GS "worst underperformance vs S&P 500 in July in more than 20 years of data"; "doubling down on megacap tech stocks after summer volatility"(Q2 13F) | 10/6 전 6주(쏠림 재형성) |
| 2026-08-25 | Hedgeweek(GS PB) | "first weekly decline in global equity exposure … in a month … 2.3 standard deviations below"; "aggregate net leverage … to 76.8%, … 27th percentile" | 10/6 전 6주 |
| 2026-09-10 | 한국은행 통화신용정책보고서(BigGo 요약) | SA 펀드 "leverage of up to 4x"; 홍콩 삼성·하이닉스 레버리지 ETF "market capitalization surge more than 20-fold in the first half"; "trading Korean stock spot, futures, and options to hedge high-leverage total return swaps"; "350,000 to 460,000 leveraged accounts were forcibly liquidated"; 개인 레버리지 ETF 손실 "₩56.3 trillion" | 7/29 사후 설명, 10/6 전 26일 |
| 2026-09-12 | Yahoo(FT) | JPM, SA 프라임 관계 종료; SA는 Clear Street 통해 "flex options on … AMD, Intel, SK Hynix, and SanDisk" | 10/6 전 24일 |
| 2026-09-14 | The Dark Side of the Boom(GS PB) | "bought US technology, media and telecommunications stocks in 10 of the past 11 sessions"; 2주 롱 매수 "97th percentile of the past five years" | **10/6 전 22일**(쏠림 재형성) |
| 2026-09-30 | Yahoo / NYT | "hedge fund borrowing from banks had reached nearly $3.7 trillion … the largest in over a decade and roughly three times … 2020"(OFR Q2); SA "leverage ratio all the way to ten times … up from four times" | **10/6 전 6일** |
| 2026-09-30 | Quartz(CNBC) | 헤지펀드 미 국채 보유 "record 7%"(제목; 본문 403) | 10/6 전 6일 |
| 2026-10-05 | Hedgeweek(GS·MS 9월 성과) | L/S "losing 0.55%", 퀀트 "+3.46%"; MS "Asian hedge funds … fell 0.6% through 25 September" | 10/6 전날 |
| 2026-10-05 | The Dark Side of the Boom | "Goldman's Cullen Morgan: Hedge Funds Have Lost Conviction While the VIX Sleeps"(유료, 본문 미확인) | 10/6 전날 |
| 2026-10-06~07 | IMF GFSR(뉴스1 10/7 14:23) | "레버리지로 13년새 몸집 3배 이상…美국채시장 비중 2022년 4%→2025년 9%"; 세계 헤지펀드 13조달러 | **10/6 당일~익일** |
| 2026-10-07 | ECB 설문(글로벌이코노믹 10/8 14:26) | 주식담보 자금수요 "순응답 비율 27%포인트"; "헤지펀드의 레버리지 활용도 역시 소폭 상승" | 사후 |
| 2026-10-08 | FT / Hedgeweek | "Hedge fund borrowing from banks has tripled since 2020"; BoE "prime brokerage balances had risen by about 40% over the previous year"(7월); Fed "50 largest hedge funds borrow about $3 for every $1 … roughly $11 … among the 15 largest" | 사후(급락 3일차) |
| 2026-10-08~09 | Seeking Alpha / GuruFocus(GS PB, 10/5 기준) | Mag7 순노출 "nearly 22% of total U.S. holdings … highest level since the start of 2022", 7월 저점 "roughly 15%"; 총노출 "above 10%" | 데이터 10/5(전날), 보도 10/8(사후) |
| 2026-10-12 | Futu(GS PB) | "Hedge funds have reduced their holdings of tech stocks, yet Mag 7's positions have reached an all-time high"(제목; 본문 비어 있음) | 사후 |

주의: Google News RSS에 2026-09-29로 뜬 Hedgeweek "Hedge funds swap tech stocks for financials"는 받아 보니 2024-06-27 기사였다(재색인). Hedgeweek "dump global equities in June at fastest pace in two years"도 2024-07-03 기사. 둘 다 표에서 뺐다.

## 3. 세 급락과 보도의 선후

| 급락 | 사전 경고(보도일 → 급락까지) | 당일·사후 | 1차 판단 |
|---|---|---|---|
| ① 6/23 | 5/18 GS 총레버리지 318% 5년 최고(36일) · 5/25 13F 총레버리지 94백분위·반도체 비중 사상 최고(29일) · 6/2 "record gross leverage"(21일) · 6/9 순레버리지 89백분위(14일) · 6/22 순레버리지 4년 최고 + $165B 분기말 매도 예고(1일) · 6/22 "순매도 전환·총레버리지 1년 최저권"(1일) | 6/25주 IT 사상 최대 매도(z −4.0), 7/9 "총레버리지 최하위 10%" | **경고 있었음.** 레버리지 최고 + 반도체 쏠림이 4~5주 전부터 반복. 다만 급락은 디그로싱 시작(6/5~6/18) 뒤에 왔으므로 "레버리지 최고" 단독이 아니라 "최고 → 축소 시작"이 더 가까운 신호 |
| ② 7/29 | 7/9 총레버리지 최하위 10%(20일) · 7/19 한국 120만 계좌 마진콜(10일) · 7/20 테크 사상 최대 매도(9일) · (7/27 기준 아시아 8일 연속 디그로싱 — 보도는 7/30) | 7/29 FT 마진콜(한국 장 마감 후), 7/30 아시아 L/S −18.6%, 7/31 SA 붕괴 | **매도 압력은 사전, 마진콜은 당일.** "레버리지 사상 최고" 보도는 이 구간에 없었다(이미 축소 중). 경고로 쓸 수 있었던 것은 "매도 속도 사상 최대"(7/20) |
| ③ 10/6~13 | 8/21 메가캡 재집중(46일) · 9/14 TMT 매수 97백분위(22일) · 9/30 은행 차입 3.7조$ 10년 최대·SA 10배 레버리지(6일) · 10/5 "확신 상실"(1일) | 10/6~7 IMF, 10/7 ECB, 10/8 FT·GS Mag7 22%(10/5 기준), 10/12 Mag7 사상 최고 | **부분 경고.** "레버리지 사상 최고" 류 보도가 9/30~10/8에 몰려 있으나 10/6 전은 9/30 하나. 한국 종목 특정 헤지펀드 자료는 10/6 전 없음(검색 12회 한도 내) |

## 4. 받은 파일

| 파일 | 내용 |
|---|---|
| `results/hf_disclosure_timeline.md` | 이 문서 |
| `results/ofr_hfm_series.csv` | OFR HFM 88시리즈 전체 시계열(mnemonic, date, value) 33,827행 |
| `results/ofr_hfm_meta.json` | 시리즈 설명(출처·정의) |
| `results/ofr_formpf_leverage_quarterly.csv` | Form PF 분기 요약 2024Q1~2026Q2(GNE·GAV·NAV·차입·전략별 레버리지·SCOOS) |
| `results/cftc_tff_leveraged_funds_equity_2026.csv` | CFTC TFF 레버리지 펀드 주가지수 순포지션 2026-05-05~09-15 |
| `results/gnews_hf_leverage_headlines.csv` | Google News RSS 제목·날짜·출처·원문 URL(영문 6개·한글 4개 질의, 672행; 2026-05 이후 493건 중 492건 `resolved_url` 복원, `code/gnews_decode.py`) |
| `code/ofr_hfm_fetch.py`, `code/gnews_headlines.py`, `code/gnews_decode.py` | 수집 코드 |

## 5. 못 받은 것과 이유

- 모건스탠리·JP모건 2026년 5~10월 레버리지 수치: 보도에 숫자 없음(2025-12-03 Reuters가 마지막 수치).
- OFR HFM의 정확한 "게시일": 페이지·API 메타에 없음("Current vintage, as of last update"만).
- SEC Private Fund Statistics 표 값: PDF 표 파싱 안 함(1쪽 확인으로 시차만 확정).
- KRX 공매도 잔고: 로그인 필수("LOGOUT").
- Reuters·연합뉴스·it.chosun·Seeking Alpha·NYT 본문: 차단(403/도메인 차단). 제목·날짜만 썼고 "본문 미확인"으로 표시.
- 10/6 이전 한국 종목 특정 헤지펀드 매도 자료: 검색 한도(12회) 안에서 못 찾음.
