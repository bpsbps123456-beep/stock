# QA — hf_proxies(헤지펀드 쏠림·개인 레버리지 대리 지표)

수집 2026-10-14 00:00~00:04 UTC(= 10-13 20:00 ET). 코드 `code/fetch_hf_proxies.py`, 실행 `py fetch_hf_proxies.py` (이어 받기: 원본이 12시간보다 새로우면 건너뜀).

## 산출물
| 파일 | 내용 |
|---|---|
| `cache/hf_proxies.parquet` | index `known`(거래일 1199줄, 2022-01-03~2026-10-13), 숫자 61칸 + `asof` |
| `cache/hf_proxies_by_symbol.parquet` | 긴 표 known·symbol·close·adj_close·volume·dollar_volume(Yahoo 14종목) |
| `cache/raw_hf_proxies/` | yahoo_<T>.parquet 14개, proshares_<T>.csv 9개(전체 이력 원본) |

## 시점 규칙(known)
| 묶음 | 측정 | known | 근거 |
|---|---|---|---|
| Yahoo 가격 지표(gvip_ lev_*_bull lev_*_share qai_ hdg_) | 그날 종가·거래량 | 그날 | 종가 16:00 ET 확정. 분석은 다음 거래일 시가 진입 |
| ProShares 발행주식수·자금 흐름(so_ flow_ aum_ lev_*_flow* lev_qqq_aum_bull) | NAV 날짜 T(`asof`) | **T 다음 거래일** | 10-13 20:00 ET에 받은 CSV 마지막 줄이 10-12(→ T 줄은 T 저녁 20시 이후 게시). Wayback 사본(2024-03-26 06:10 ET)에는 03-25 줄이 있었다(→ T+1 개장 전 게시). 즉 T+1 종가 전 공개가 확실하다 |

- `asof` 칸 = 자금 흐름 칸의 측정일(직전 거래일). 가격 지표 칸은 known 그날 값이다.
- 같은 날 반응 확인(2024~, Spearman): TQQQ 흐름 vs 같은 날 TQQQ 수익률 −0.43(전날 −0.18), SQQQ −0.49(전날 −0.09) → T 줄의 주식 수 변화는 T 주문(당일 역행 매매)이다. 한 날 밀려 기록된 흔적 없음.
- 덜 보수적인 대안: 게시가 T+1 개장 전이므로 T+1 시가 진입도 가능하나, 여기서는 규칙대로 T+1을 known으로 둔다.

## 칸 정의(단위: 비율은 소수, 0.05 = 5%; 흐름은 달러; so_는 주식 수)
| 칸 | 정의 |
|---|---|
| gvip_rel, gvip_rel20, gvip_ret20 | GVIP/QQQ 수정종가 비율, 그 20일 변화율, GVIP 20일 수익률 |
| gvip_rel_spy, gvip_rel_spy20 | GVIP/SPY 비율과 20일 변화율(추가) |
| qai_beta60, qai_beta60_spy, hdg_beta60, hdg_beta60_qqq | 일간 수익률 60일 이동 OLS 기울기(cov/var, 60일 다 차야 값) |
| qai_ret20, hdg_ret20 | 20일 수익률(추가) |
| lev_qqq_bull(5) | TQQQ 거래대금 /(TQQQ+SQQQ), (5)는 5일 합끼리 |
| lev_soxx_bull(5) | SOXL /(SOXL+SOXS) |
| lev_nvda_bull(5) | NVDL /(NVDL+NVDD) — 2023-09-13부터 |
| lev_nvda_all_bull(5) | (NVDL+NVDU)/(NVDL+NVDU+NVDD+NVDS) — 2023-09-13부터(추가) |
| lev_qqq_share(5), lev_soxx_share(5) | (상승+하락 레버리지 거래대금)/기초 ETF 거래대금, (5)는 5일 평균(추가) |
| so_, aum_, flow_<t> | ProShares 9종(TQQQ SQQQ QLD QID PSQ USD SSG UPRO SPXU) 발행주식수·순자산·흐름(Δ주식 수 × 그날 NAV) |
| lev_qqq_flow(5,20), lev_qqq_flow5_pct | TQQQ 흐름 − SQQQ 흐름, 5·20일 합, 5일 합 ÷ (TQQQ+SQQQ 순자산) |
| lev_qqq_aum_bull | TQQQ 순자산 /(TQQQ+SQQQ) |
| lev_qqq_flow_all(5) | TQQQ+QLD − SQQQ−QID−PSQ 흐름 |
| lev_semi_flow(5), lev_spx_flow(5) | USD − SSG(반도체 2배), UPRO − SPXU |

## 확인한 것
- index 유일·정렬, 2024-01-01 이후 결측: 가격 지표 각 1칸(10-13), 흐름 지표 0칸. rolling 첫 값 2022-03-30(베타)이라 2023부터 250일 백분위 창이 찬다.
- ProShares 이력은 분할을 과거 줄까지 소급 반영한다. Prior NAV ≠ 전날 NAV인 날 0건. 주식 수 하루 30% 넘게 바뀐 날(QID 3·PSQ 4·SSG 11·SPXU 2일)은 순자산도 같이 뛰어 실제 설정·환매다(작은 펀드).
- 2026 요약: lev_qqq_bull5 중앙 0.65(0.47~0.73), lev_soxx_bull5 0.77, TQQQ−SQQQ 순흐름 합 −68.6억 달러(오르는 장에서 TQQQ 환매가 지속), qai_beta60 0.25~0.40.

## 빈 곳·못 구한 것
- **10-13 가격 지표 없음**: 20:03 ET에 Yahoo 10-13 일봉이 거래량만 있고 종가 NaN이었다. 행은 두고 값은 비웠다. Yahoo가 채운 뒤 `py fetch_hf_proxies.py --max-age-hours 0`으로 다시 받으면 채워진다(ProShares도 다시 받음).
- **SOXL·SOXS 발행주식수 이력 없음**: Direxion 상품 페이지(Next.js·AppSync API)에 이력 링크가 없고, `direxion.com/holdings/SOXL.csv`는 오늘 값 하나뿐(10-13 162,000,060주). sharesoutstandinghistory.com은 2018년 이후 약 243점(드문 간격)이라 일별 흐름에 못 쓴다. 반도체 흐름은 ProShares USD(2배)·SSG로 대신했다 — USD 거래 규모가 SOXL보다 훨씬 작다.
- **QQQ 발행주식수 이력 없음**: Invesco dng-api가 null·500·400을 돌려줬다. 무료 일별 이력 미확보.
- yfinance `get_shares_full`은 ETF에 None(과제 설명대로).
- Wayback 사본 확인은 2번째 요청부터 봇 차단 화면(429 아님)이 와서 바로 멈췄다. 게시 시각 근거는 사본 1건 + 이번 수집 관찰 1건이다.

## 해석 주의
- lev_*_bull은 거래대금 몫이다. 거래량에는 사고판 양쪽이 다 들어 있어 순매수 방향이 아니라 "어느 쪽이 더 붐비나"다. 순방향은 flow 칸(설정·환매)이 낫다.
- lev_nvda_bull은 NVDD가 작아(2024~ 중앙 거래대금 1,400만 달러 vs NVDL 8.9억) 0.83~1.00에 몰려 변별력이 약하다. nvda_all을 함께 본다.
- GVIP 구성은 13F(분기 말 45일 뒤 공개) 기준으로 분기마다 바뀐다. 가격은 그날 알 수 있지만 담긴 '헤지펀드 선호'는 최대 4~5개월 묵은 정보다.
- QAI·HDG는 공개 지수·요인으로 헤지펀드 수익을 복제한 ETF라 실제 포지션이 아니다. HDG는 하루 중앙 800주(약 4만 달러)로 얇아 종가 무변동일 2.7%(QAI 3.7%) — 베타에 잡음.

## 검증(verification, 독립 검증자 2026-10-14 00:06~00:47 UTC = 10-13 20:06~20:47 ET)
판정: **통과**. 값·시점·단위에서 고칠 버그 없음. QA 문구 1곳만 고침(NVDD 거래대금 중앙값 기간 '2025~' → '2024~', 실제 2024~ 1,420만 vs 8.88억 달러).

| 항목 | 방법 | 결과 |
|---|---|---|
| ProShares 원값 | 수집 코드를 거치지 않고 curl로 TQQQ·SQQQ·USD CSV를 다시 받아 6개 날짜(2024-03-25, 2024-11-15, 2025-04-07, 2025-08-21, 2026-02-17, 2026-10-12) × so·flow·aum 비교 | 54칸 모두 일치(차이 0) |
| Yahoo 원값 | yfinance 대신 Yahoo chart API(v8)를 직접 받아 TQQQ·GVIP·SOXS 5개 날짜(2024-06-14, 2025-04-08, 2025-11-20, 2026-03-13, 2026-10-12)의 Close·Volume·Adj Close 비교 | 일치(SOXS 2025-04-08 Adj Close 소수 넷째 자리 반올림 차이 1건) |
| 파생 지표 재계산 | 위 chart API 값(QQQ·SQQQ·SOXL·QAI 추가)으로 gvip_rel·gvip_rel20·lev_qqq_bull·lev_qqq_bull5·lev_soxx_bull·qai_beta60(np.polyfit 60일)을 따로 계산 | 5개 날짜 모두 소수 6자리까지 일치 |
| 흐름 내부 정합 | 2021-12~ 9종 전부: AUM = 주식 수 × NAV(비율 1.0000, SSG만 최대 1.0037), Prior NAV = 전날 NAV 불일치 0, Δ주식 수×NAV와 AUM으로 역산한 흐름 차이 > AUM 2%인 날 0 | 분할 소급 반영 확인(분할 날 가짜 흐름 없음) |
| **게시 시각(known)** | 서버 HTTP `Last-Modified` 헤더: 9개 CSV 모두 **2026-10-12 20:18:38~44 ET**(10-12 줄 게시 일괄 작업). 이어서 10-13 20:09~20:47 ET에 2.5분 간격 16회 조회 — 10-13 줄은 20:47 ET까지 아직 안 올라옴 | T 줄은 **T 장 마감 뒤 저녁(20시 이후, 날마다 다름)** 게시 → T 종가 때는 미공개, T+1 개장 전 공개(수집자의 Wayback 06:10 ET 사본과 일치). known = T+1 규칙이 맞고 앞당김(look-ahead) 없음 |
| T 줄 = T 주문인지 | 2024~ 주식 수 변화율 vs NAV 수익률 Spearman: TQQQ 전날 −0.18·당일 −0.42·다음날 +0.04, SQQQ −0.09·−0.49·+0.02 | 당일만 강함 → T 줄은 T 주문. 다음날 수익률과 무관 → T 줄에 T+1 주문이 섞이지 않음 |
| 휴장일 | asof가 known의 직전 거래일과 다른 줄 0(2025-01-09 카터 추모 휴장, 성금요일, 추수감사절 포함). ProShares에만 있는 날 0, QQQ에만 있는 날은 10-13 하나(게시 전) | 정상 |
| 형식 | index `known` 문자열·유일·정렬 1199줄, 숫자 61칸 + `asof`, 보고한 칸 목록과 같음. by_symbol 15,552줄, (known, symbol) 중복 0, dollar_volume = close×volume | 정상 |
| 재실행 | `py fetch_hf_proxies.py`(원본 12시간 이내라 네트워크 없이 재계산, 1.7초) | 두 parquet 모두 기존 파일과 완전히 같음 |
| 요약 수치 | 2026 lev_qqq_bull5 중앙 0.650(0.469~0.725), TQQQ−SQQQ 순흐름 known 2026 합 −68.6억 달러, qai_beta60 0.25~0.40 | 보고와 일치 |

남은 주의(버그 아님):
- 10-13 가격 지표는 여전히 비어 있다. 20:12 ET에도 Yahoo 일봉 종가가 null(meta의 정규장 가격 743.57은 있음). Yahoo가 채운 뒤 `--max-age-hours 0`으로 다시 받는다.
- 장 마감 직후(16:15~ ET) 실행하면 Yahoo 마지막 봉이 잠정 종가·부분 거래량일 수 있고 12시간 캐시에 남는다. 마지막 줄 lev_*_bull은 다음 실행 때 다시 받은 값으로 확인한다.
- ProShares CSV는 매일 전체 이력을 다시 쓰는 파일이다. 과거 줄이 게시 뒤에 고쳐졌는지(빈티지)는 무료로 확인할 길이 없다(Wayback CDX는 이번에 503 '일시 중단'으로 1회 시도 후 멈춤). 분할 소급 외의 수정 흔적은 위 정합 점검에서 보이지 않았다.
- 게시 시각 근거는 Last-Modified 1일(10-12 20:18 ET) + 미게시 관찰 1일(10-13 20:47 ET까지) + Wayback 1건이다. 게시가 T+1 16:00 ET를 넘긴 날이 있었는지는 알 수 없으나, 분석은 known 다음 날 시가(T+2)에 들어가므로 여유가 하루 더 있다.
- gvip_rel 등 수정종가 비율의 '수준'은 내려받은 날까지의 배당으로 일정 배율이 곱해진 값이다. 250일 창 안의 순위·변화율은 창 안 배당만 쓰므로 앞당김이 없지만, 고정 문턱값(절대 수준)으로 쓰면 안 된다.
- `asof` 칸 이름이 pandas `DataFrame.asof` 메서드와 겹친다. `df.asof`가 아니라 `df["asof"]`로 읽는다.
