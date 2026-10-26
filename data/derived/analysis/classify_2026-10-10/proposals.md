# TIME 미국 보유 미분류 종목 분류 제안 (2026-10-10)

입력: 2023-05-22~2025-12-29 주간 PDF 137개(ISIN 3종 SMCI·ANET·BZFD 복원) + 2026 일간 스냅샷 187개. 현금·선물·ETF/펀드 제외 후 미국 상장 137개사 중 `desk.strategies.sector_of`가 `미분류`인 57개.

- 분류 제안 56개, 모두 기존 SECTOR_MAP 섹터(새 섹터 0개). 반도체 예산 대상 3개: MBLY, AMBA, AMKR.
- 제외 1개: DXYZ — Destiny Tech100: 비상장 기술주(SpaceX·OpenAI 등)를 담는 NYSE 상장 폐쇄형 펀드. exclusion_reason의 이름 규칙(ETF|FUND…)에 안 걸려 통과했다. 펀드라 분류 대상 아님.
- 2026 일간 스냅샷에 들어 있어 야간 데스크 판단(2026-01-02~)에 영향 주는 이름 11개: CEG, RDDT, APP, CCJ, BIDU, LEU, PL, FLNC, EOSE, AMKR, SERV. 나머지는 2023~2025 주간 PDF에만 있음.
- 웹 검색 0회(모두 잘 알려진 회사라 자체 지식으로 분류).

## 섹터별 묶음

| 섹터 | 반도체 예산 | 종목 |
|---|---|---|
| 원자력/에너지 | 아니오 | CEG, VST, CCJ, ETN, LEU, FLNC, POWL, KMI, EOSE, GNRC, LNG, EQT, NXT, BWXT, BKR, RUN |
| 엔터프라이즈 SW | 아니오 | AI, ADBE, PATH, WDAY, SOUN, TWLO, BBAI, RELX, CXM, INOD |
| 이커머스/플랫폼 | 아니오 | RDDT, PERI, APP, SE, CPNG, PINS, NRDY |
| 산업재 | 아니오 | MP, CRML, UAMY, USAR |
| 미디어/게임 | 아니오 | RBLX, BZFD, GETY, SSTK |
| AI 인프라 | 아니오 | SMCI, P, CORZ |
| AI 반도체 | 예 | MBLY, AMBA, AMKR |
| 헬스케어 AI | 아니오 | SDGR, RXRX |
| 양자컴퓨팅 | 아니오 | QBTS, QUBT |
| 중국 AI/플랫폼 | 아니오 | BIDU |
| 헬스케어 | 아니오 | ISRG |
| 우주/방산 | 아니오 | PL |
| 전기차/자율주행 | 아니오 | SERV |
| 핀테크/디지털자산 | 아니오 | LMND |

## 종목별 (최대 NAV 비중 순)

| 티커 | 회사 | 제안 섹터 | 반도체 예산 | 확신 | AI_UNIVERSE 소분류 | 보유 기간 | 최대 비중 | PDF 수(주간+일간) | 근거 |
|---|---|---|---|---|---|---|---:|---:|---|
| SMCI | Super Micro Computer Inc | AI 인프라 | 아니오 | high | 서버·스토리지 | 2023-05-22~2024-09-23 | 8.30% | 71+0 | AI·GPU 서버 랙 제조가 매출 대부분. DELL과 같은 서버·스토리지 층 |
| MP | MP Materials Corp | 산업재 | 아니오 | medium | - | 2025-07-28~2025-11-10 | 5.86% | 16+0 | 희토류 채굴·정제·영구자석(미 국방부 지분). 핵심광물 섹터가 없어 특수합금 CRS가 있는 산업재로 |
| CEG ★ | Constellation Energy Corp | 원자력/에너지 | 아니오 | high | 발전·연료전지 | 2024-03-11~2026-01-06 | 5.20% | 93+3 | 미국 최대 원전 운영 발전사. 빅테크 원전 전력 계약 |
| VST | Vistra Corp | 원자력/에너지 | 아니오 | high | 발전·연료전지 | 2024-05-07~2025-10-27 | 4.68% | 76+0 | 가스·원전 발전사(Energy Harbor 원전 인수) |
| RDDT ★ | Reddit Inc | 이커머스/플랫폼 | 아니오 | medium | - | 2024-12-02~2026-01-29 | 4.53% | 57+20 | 커뮤니티 플랫폼 광고 + AI 데이터 라이선스. 소셜 섹터가 없어 인터넷 플랫폼으로 |
| RBLX | ROBLOX Corp | 미디어/게임 | 아니오 | high | - | 2025-03-24~2025-11-10 | 4.40% | 34+0 | 게임 플랫폼, 게임 내 화폐(Robux) 판매 |
| PERI | Perion Network Ltd | 이커머스/플랫폼 | 아니오 | low | - | 2023-05-22~2023-08-28 | 4.18% | 15+0 | 검색·CTV·디스플레이 광고 기술(애드테크). APP와 같은 광고 플랫폼으로 묶음 |
| AI | C3.ai Inc | 엔터프라이즈 SW | 아니오 | high | - | 2023-05-22~2025-01-06 | 4.04% | 38+0 | 기업용 AI 애플리케이션 SW 구독 |
| ADBE | Adobe Inc | 엔터프라이즈 SW | 아니오 | high | - | 2023-05-22~2024-02-19 | 4.01% | 40+0 | 크리에이티브·문서 SW 구독(Creative Cloud·Acrobat) |
| APP ★ | AppLovin Corp | 이커머스/플랫폼 | 아니오 | medium | - | 2024-11-18~2026-02-02 | 4.01% | 59+21 | 모바일 게임·이커머스 광고 엔진(AXON). 2025년 앱 사업 매각 후 광고 플랫폼 100%. B2B SW로 보면 엔터프라이즈 SW도 가능 |
| CCJ ★ | Cameco Corp | 원자력/에너지 | 아니오 | high | - | 2025-06-16~2026-03-26 | 3.76% | 26+43 | 우라늄 채굴·핵연료 서비스, Westinghouse 지분. UEC·UUUU 관례 |
| ETN | Eaton Corp PLC | 원자력/에너지 | 아니오 | medium | 전력·냉각 장비 | 2024-05-07~2024-12-16 | 3.68% | 25+0 | 배전·개폐기·UPS 등 전력관리 장비. AI_UNIVERSE에서 같은 층인 LS일렉트릭·HD현대일렉트릭·효성중공업이 이 섹터(VRT는 냉각 중심이라 AI 인프라) |
| BIDU ★ | Baidu Inc | 중국 AI/플랫폼 | 아니오 | high | - | 2023-05-22~2026-02-04 | 3.19% | 60+23 | 중국 검색 광고·AI 클라우드(ERNIE)·자율주행 |
| LEU ★ | Centrus Energy Corp | 원자력/에너지 | 아니오 | high | - | 2025-06-16~2026-01-23 | 3.07% | 29+16 | 농축 우라늄(LEU·HALEU) 공급 |
| ISRG | Intuitive Surgical Inc | 헬스케어 | 아니오 | high | - | 2025-01-06~2025-10-13 | 2.89% | 41+0 | 수술 로봇(da Vinci)·소모품 |
| BZFD | BuzzFeed Inc | 미디어/게임 | 아니오 | high | - | 2023-05-22~2023-06-12 | 2.85% | 4+0 | 디지털 뉴스·엔터테인먼트 미디어 광고 |
| PATH | UiPath Inc | 엔터프라이즈 SW | 아니오 | high | - | 2023-12-04~2024-03-04 | 2.78% | 14+0 | 업무 자동화(RPA)·에이전트 SW |
| WDAY | Workday Inc | 엔터프라이즈 SW | 아니오 | high | - | 2023-12-04~2024-06-17 | 2.56% | 24+0 | 인사·재무 SaaS |
| P | Pure Storage Inc | AI 인프라 | 아니오 | high | 서버·스토리지 | 2024-04-01~2025-04-07 | 2.53% | 54+0 | 구 Pure Storage(PSTG, 현 Everpure). 올플래시 스토리지 어레이·구독, NTAP 관례 |
| PL ★ | Planet Labs PBC | 우주/방산 | 아니오 | high | - | 2025-12-29~2026-03-03 | 2.19% | 1+39 | 지구관측 위성 영상 데이터 구독(정부·국방 고객 중심) |
| FLNC ★ | Fluence Energy Inc | 원자력/에너지 | 아니오 | high | - | 2025-11-03~2026-03-03 | 2.19% | 9+39 | 전력망용 배터리 에너지저장장치(ESS) |
| SE | Sea Ltd | 이커머스/플랫폼 | 아니오 | high | - | 2025-03-24~2025-09-15 | 2.07% | 7+0 | 동남아 이커머스 Shopee 중심 + 게임·핀테크. MELI 관례 |
| CPNG | Coupang Inc | 이커머스/플랫폼 | 아니오 | high | - | 2024-04-22~2024-04-22 | 2.05% | 1+0 | 한국 이커머스(로켓배송) |
| SDGR | Schrodinger Inc/United States | 헬스케어 AI | 아니오 | high | - | 2023-07-03~2023-10-10 | 2.03% | 15+0 | 계산화학 신약개발 SW + 자체 파이프라인, 적자. TEM 관례 |
| SOUN | SoundHound AI Inc | 엔터프라이즈 SW | 아니오 | medium | - | 2023-05-22~2025-10-27 | 2.01% | 69+0 | 음성 AI SW 라이선스·구독(차량·외식 주문). 적자 소형주 |
| TWLO | Twilio Inc | 엔터프라이즈 SW | 아니오 | high | - | 2024-12-02~2024-12-16 | 1.87% | 3+0 | 메시지·음성 통신 API 클라우드 |
| POWL | Powell Industries Inc | 원자력/에너지 | 아니오 | medium | - | 2024-03-11~2024-03-25 | 1.83% | 3+0 | 맞춤형 개폐기·전력제어실(석유화학·전력·데이터센터). ETN과 같은 전력기기 |
| BBAI | BigBear.ai Holdings Inc | 엔터프라이즈 SW | 아니오 | medium | - | 2023-05-22~2025-10-13 | 1.81% | 32+0 | 국방·정부용 AI 의사결정 분석 SW. 고객이 국방이라 우주/방산도 가능하나 PLTR 관례 |
| MBLY | Mobileye Global Inc | AI 반도체 | 예 | medium | - | 2023-05-22~2024-01-02 | 1.80% | 33+0 | ADAS용 EyeQ SoC 칩 판매가 매출 대부분. 시장 분류(GICS)는 자동차부품이라 전기차/자율주행도 가능하지만, 차량 반도체 ON·NXPI·STM을 AI 반도체에 둔 관례를 따름 |
| KMI | Kinder Morgan Inc | 원자력/에너지 | 아니오 | medium | - | 2025-01-31~2025-05-19 | 1.71% | 8+0 | 천연가스 파이프라인(미드스트림). 데이터센터 가스 발전 수요 테마 |
| QBTS | D-Wave Quantum Inc | 양자컴퓨팅 | 아니오 | high | - | 2024-12-23~2025-03-04 | 1.55% | 11+0 | 양자 어닐링 컴퓨터·클라우드 접근 |
| EOSE ★ | Eos Energy Enterprises Inc | 원자력/에너지 | 아니오 | high | - | 2025-11-03~2026-03-03 | 1.54% | 9+39 | 아연 기반 장주기 에너지저장 배터리 |
| RELX | RELX PLC | 엔터프라이즈 SW | 아니오 | medium | - | 2023-05-22~2023-05-30 | 1.39% | 2+0 | 법률·과학·리스크 정보 분석 구독(LexisNexis·Elsevier). 출판이라 미디어도 가능 |
| GETY | Getty Images Holdings Inc | 미디어/게임 | 아니오 | high | - | 2023-05-22~2023-05-30 | 1.38% | 2+0 | 이미지·영상 콘텐츠 라이선스 |
| PINS | Pinterest Inc | 이커머스/플랫폼 | 아니오 | medium | - | 2023-05-22~2023-05-30 | 1.33% | 2+0 | 이미지 발견·쇼핑 플랫폼 광고 |
| AMBA | Ambarella Inc | AI 반도체 | 예 | high | - | 2023-11-13~2024-01-08 | 1.24% | 9+0 | 엣지 AI 영상처리 SoC 팹리스(차량·보안카메라) |
| GNRC | Generac Holdings Inc | 원자력/에너지 | 아니오 | medium | - | 2025-08-11~2025-09-15 | 1.19% | 6+0 | 가정·상업용 비상 발전기(데이터센터 백업 발전 포함). CAT·CMI 발전 설비 층과 유사 |
| LNG | Cheniere Energy Inc | 원자력/에너지 | 아니오 | medium | - | 2025-04-28~2025-09-15 | 1.00% | 10+0 | LNG 수출 터미널 운영 |
| RXRX | Recursion Pharmaceuticals Inc | 헬스케어 AI | 아니오 | high | - | 2023-07-24~2023-08-28 | 0.95% | 6+0 | AI 신약개발 바이오, 적자 |
| NRDY | Nerdy Inc | 이커머스/플랫폼 | 아니오 | low | - | 2023-05-22~2023-05-30 | 0.91% | 2+0 | 온라인 과외 플랫폼(Varsity Tutors), 소비자·학교 구독 |
| CORZ | Core Scientific Inc | AI 인프라 | 아니오 | high | - | 2025-09-29~2025-11-10 | 0.90% | 7+0 | 비트코인 채굴에서 CoreWeave용 HPC 데이터센터 임대로 전환. IREN·CIFR·APLD 관례 |
| AMKR ★ | Amkor Technology Inc | AI 반도체 | 예 | high | OSAT(후공정 외주) | 2025-12-22~2026-01-02 | 0.90% | 2+1 | 반도체 후공정(패키징·테스트) 외주. 파운드리 TSM·GFS·TSEM과 같은 섹터 |
| CXM | Sprinklr Inc | 엔터프라이즈 SW | 아니오 | high | - | 2023-05-22~2023-05-30 | 0.86% | 2+0 | 고객경험 관리 SaaS |
| SSTK | Shutterstock Inc | 미디어/게임 | 아니오 | high | - | 2023-05-22~2023-05-30 | 0.79% | 2+0 | 스톡 이미지·영상 라이선스(Getty와 합병 진행) |
| EQT | EQT Corp | 원자력/에너지 | 아니오 | medium | - | 2025-04-28~2025-05-19 | 0.72% | 4+0 | 미국 최대 천연가스 생산사 |
| NXT | Nextpower Inc | 원자력/에너지 | 아니오 | high | - | 2025-11-10~2025-12-01 | 0.63% | 4+0 | 태양광 추적장치(트래커). 2025년 Nextpower로 사명 변경. FSLR 관례 |
| SERV ★ | Serve Robotics Inc | 전기차/자율주행 | 아니오 | medium | - | 2025-12-08~2026-01-21 | 0.61% | 4+14 | 보도 자율주행 배달 로봇. 로봇 섹터가 없어 자율주행으로 |
| QUBT | Quantum Computing Inc | 양자컴퓨팅 | 아니오 | high | - | 2024-12-23~2025-03-04 | 0.54% | 10+0 | 광자 양자컴퓨팅·박막 리튬니오베이트 칩 |
| BWXT | BWX Technologies Inc | 원자력/에너지 | 아니오 | high | - | 2025-11-03~2025-11-10 | 0.52% | 2+0 | 미 해군 원자로·핵연료·원전 부품. 매출 대부분 정부 원자력 사업 |
| LMND | Lemonade Inc | 핀테크/디지털자산 | 아니오 | high | - | 2024-12-02~2024-12-02 | 0.44% | 1+0 | AI 기반 디지털 보험(인슈어테크) |
| BKR | Baker Hughes Co | 원자력/에너지 | 아니오 | medium | - | 2025-07-28~2025-10-13 | 0.40% | 12+0 | 유전 서비스 + 가스터빈·LNG 설비(IET). 데이터센터 가스터빈 수주 |
| INOD | Innodata Inc | 엔터프라이즈 SW | 아니오 | medium | - | 2024-12-02~2024-12-02 | 0.38% | 1+0 | 빅테크용 AI 학습 데이터·데이터 엔지니어링 서비스(IT 서비스, ACN 관례) |
| CRML | Critical Metals Corp | 산업재 | 아니오 | medium | - | 2025-10-20~2025-10-27 | 0.38% | 2+0 | 그린란드 Tanbreez 희토류 광산 개발, 적자. 핵심광물 테마 |
| UAMY | United States Antimony Corp | 산업재 | 아니오 | medium | - | 2025-10-20~2025-10-27 | 0.38% | 2+0 | 안티몬·제올라이트 채굴·제련. 핵심광물 테마 |
| USAR | USA Rare Earth Inc | 산업재 | 아니오 | medium | - | 2025-10-20~2025-10-27 | 0.32% | 2+0 | 희토류 자석 공장·Round Top 광산 개발, 적자. MP와 같은 핵심광물 테마 |
| RUN | Sunrun Inc | 원자력/에너지 | 아니오 | high | - | 2025-11-10~2025-12-01 | 0.26% | 4+0 | 주택용 태양광·저장 설치·리스 |

★ = 2026 일간 스냅샷에도 있음(데스크 엔진이 읽는 구간).

## 판단이 갈릴 수 있는 곳

- **MBLY**: 매출은 EyeQ 칩이지만 시장 분류는 자동차부품. 차량 반도체 ON·NXPI·STM이 이미 `AI 반도체`라 같은 관례로 반도체 예산에 넣었다. 2026 일간에는 없어 데스크 영향 없음.
- **ETN·POWL·GNRC**: 전력기기. AI_UNIVERSE에서 같은 층인 국내 변압기·개폐기 3사가 `원자력/에너지`라 그쪽으로. VRT(냉각 중심)는 `AI 인프라` 그대로.
- **KMI·LNG·EQT·BKR**: 석유·가스 섹터가 없어 `원자력/에너지`. 원자력·전력 기술주와 원자재 가격 민감도가 다르다는 점은 남는다(KoAct의 XOM도 같은 문제).
- **RDDT·PINS·APP·PERI·NRDY**: 소셜·광고 섹터가 없어 `이커머스/플랫폼`을 '인터넷 플랫폼(상거래+광고)'으로 넓혀 썼다. APP는 `엔터프라이즈 SW`도 가능.
- **MP·USAR·CRML·UAMY**: 핵심광물 섹터가 없어 특수합금 CRS가 있는 `산업재`. 네 종목끼리 상관이 높아, 묶음 상한을 따로 원하면 새 섹터 `핵심광물`(반도체 예산 아님)을 만들 수 있다.
- **DXYZ**: 폐쇄형 펀드인데 `exclusion_reason` 이름 규칙에 안 걸린다. 2024-04-08 주간 PDF 1회뿐이라 데스크 영향 없음.

## KoAct 미국 보유 중 이 제안 후에도 미분류: 26개

AEP, ALNY, AOSL, CBRS, CLS, COST, DG, DRUG, DUK, DYN, FIX, FN, FSLY, GH, KEYS, LUNR, ONDS, PEPG, POET, RIOT, RMBS, TTMI, VZ, WMT, XNDU, XOM

(KoAct 미분류 35개 중 이 제안으로 함께 해소: APP, BKR, CCJ, FLNC, LEU, NXT, PL, QBTS, RDDT)
