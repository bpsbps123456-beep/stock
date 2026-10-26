# 갈래 6 조합 — 사전 계획 (2026-10-14, 계산 전 작성)

이 파일은 `code/b6_combo.py plan`이 특징·문턱만 계산해서 썼다. 결과(손익·분류)는 아직 보지 않았다. `run`은 이 목록과 칸 이름·순서가 같은지(sha256) 확인한 뒤에만 판정한다.

## 계획 전에 이미 본 것(공개)

- 과제가 준 다른 갈래 앞 구간 상위 칸의 정의와 앞 구간 차이.
- `code/b2_confirm.py`를 읽다가 보고서 문구 틀(결과 뒤 작성)에서 '확인 대기 변형이 뒤 구간에서 거래당 손실'이라는 결론 문장을 보았다.
- `results/verify_b4_flow_timing.json` 구조를 확인하다가 갈래 4의 `N20|gvip_rel20_p≥0.5382` 칸이 뒤 구간에서 통과했다는 값을 보았다. 이 특징은 과제가 준 상위 칸 목록에 없고, 이 갈래에서도 쓰지 않는다.
- 다른 갈래의 결과 보고서(.md)와 칸 표(.csv)는 열지 않았다.

## 원자(다른 갈래 앞 구간 상위 칸)

문턱은 같은 규칙(앞 구간 같은 N 사건의 1/3·2/3 분위, 갈래 1은 같은 kind도)으로 다시 계산했다. 과제가 준 문턱 문자열과 다른 원자: 없음. 하네스 `tercile_cells`와 켜짐 표지가 다른 일봉 원자: 없음. b4 날짜 특징 재계산 대조(배치 규칙): {'nvdl_flow20_p_when_not_before_day': 0, 'pc_equity5_p_when_not_before_day': 0, 'nvdl_flow20_p': {'max_abs': 0.0, 'nan_mismatch': 0}, 'pc_equity5_p': {'max_abs': 0.0, 'nan_mismatch': 0}}.

| 가족 | 갈래 | N | 특징 | 방향 | 문턱(정밀) | 모집단 | 언제 알 수 있나 |
|---|---|---:|---|---|---:|---|---|
| D | b1 | 20 | ret5_pre | hi | -0.0238729 | stock | t−1 종가 |
| D | b1 | 20 | depth_atr | hi | 0.489209 | stock | t 종가 |
| D | b1 | 40 | gap_break | =0 | 0 | stock | t 시가 |
| D | b1 | 60 | gap_break | =0 | 0 | stock | t 시가 |
| D | b3 | 20 | cobreak_share | lo | 0.0353982 | 종목+ETF | t 종가 |
| D | b3 | 40 | cobreak_share | lo | 0.0265487 | 종목+ETF | t 종가 |
| D | b3 | 20 | vix | lo | 18.38 | 종목+ETF | t ^VIX 종가(16:15 ET) |
| D | b3 | 60 | rs20_qqq | hi | -0.0817494 | 종목+ETF | t 종가 |
| D | b4 | 20 | nvdl_flow20_p | hi | 0.64257 | 종목+ETF | t 저녁 공개(GraniteShares, 가정) |
| D | b4 | 20 | pc_equity5_p | hi | 0.844712 | 종목+ETF | t 저녁 공개(CBOE, 가정) |
| D | b4 | 20 | si_dtc | hi | 2.83 | 종목+ETF | 가장 최근 공개일 다음 거래일 |
| D | b4 | 60 | shvr_z20 | hi | -0.119153 | 종목+ETF | t 저녁 공개(FINRA) |
| D | b5 | 20 | gap_size | lo | 0.0108686 | 종목+ETF | t 시가 |
| D | b5 | 40 | sub_cobreak_n | lo | 0 | 종목+ETF | t 종가 |
| D | b5 | 60 | earn_pm3 | =0 | 0 | 종목+ETF | t 종가(공시 일정) |
| D | b7 | 20 | soxx_brk | =0 | 0 | 종목+ETF | t 종가 |
| D | b7 | 20 | sq_brk | =0 | 0 | 종목+ETF | t 종가 |
| D | b7 | 20 | qqq_brk | =0 | 0 | 종목+ETF | t 종가 |
| D | b7 | 40 | soxx_brk | =0 | 0 | 종목+ETF | t 종가 |
| b8 | b8 | 28 | depth_atr | lo | 0.345383 | 종목+ETF | 이탈 봉 종가(봉 i−1까지 ATR, 일봉 박스는 전 거래일까지) |
| b8 | b8 | 14 | dbox_pos | hi | 0.193672 | 종목+ETF | 이탈 봉 종가(봉 i−1까지 ATR, 일봉 박스는 전 거래일까지) |
| b8 | b8 | 7 | dbox_pos | hi | 0.470884 | 종목+ETF | 이탈 봉 종가(봉 i−1까지 ATR, 일봉 박스는 전 거래일까지) |
| b8 | b3 | 28 | vix_prev | lo | 20.4433 | 종목+ETF | 이탈 거래일 시가 전(전 거래일 종가 또는 그 전 저녁 공개분) |
| b8 | b4 | 28 | nvdl_D | hi | 0.842035 | 종목+ETF | 이탈 거래일 시가 전(전 거래일 종가 또는 그 전 저녁 공개분) |
| b8 | b4 | 28 | pc_D | hi | 0.773092 | 종목+ETF | 이탈 거래일 시가 전(전 거래일 종가 또는 그 전 저녁 공개분) |
| b8 | b7 | 28 | sq_brk20_prev | =0 | 0 | 종목+ETF | 이탈 거래일 시가 전(전 거래일 종가 또는 그 전 저녁 공개분) |
| b8 | b7 | 28 | soxx_brk20_prev | =0 | 0 | 종목+ETF | 이탈 거래일 시가 전(전 거래일 종가 또는 그 전 저녁 공개분) |
| b8 | b7 | 28 | qqq_brk20_prev | =0 | 0 | 종목+ETF | 이탈 거래일 시가 전(전 거래일 종가 또는 그 전 저녁 공개분) |
| b8 | b3 | 14 | vix_prev | lo | 18.4133 | 종목+ETF | 이탈 거래일 시가 전(전 거래일 종가 또는 그 전 저녁 공개분) |
| b8 | b4 | 14 | nvdl_D | hi | 0.820616 | 종목+ETF | 이탈 거래일 시가 전(전 거래일 종가 또는 그 전 저녁 공개분) |
| b8 | b4 | 14 | pc_D | hi | 0.771754 | 종목+ETF | 이탈 거래일 시가 전(전 거래일 종가 또는 그 전 저녁 공개분) |
| b8 | b7 | 14 | sq_brk20_prev | =0 | 0 | 종목+ETF | 이탈 거래일 시가 전(전 거래일 종가 또는 그 전 저녁 공개분) |
| b8 | b7 | 14 | soxx_brk20_prev | =0 | 0 | 종목+ETF | 이탈 거래일 시가 전(전 거래일 종가 또는 그 전 저녁 공개분) |
| b8 | b7 | 14 | qqq_brk20_prev | =0 | 0 | 종목+ETF | 이탈 거래일 시가 전(전 거래일 종가 또는 그 전 저녁 공개분) |
| b8 | b3 | 7 | vix_prev | lo | 18 | 종목+ETF | 이탈 거래일 시가 전(전 거래일 종가 또는 그 전 저녁 공개분) |
| b8 | b4 | 7 | nvdl_D | hi | 0.682731 | 종목+ETF | 이탈 거래일 시가 전(전 거래일 종가 또는 그 전 저녁 공개분) |
| b8 | b4 | 7 | pc_D | hi | 0.771084 | 종목+ETF | 이탈 거래일 시가 전(전 거래일 종가 또는 그 전 저녁 공개분) |
| b8 | b7 | 7 | sq_brk20_prev | =0 | 0 | 종목+ETF | 이탈 거래일 시가 전(전 거래일 종가 또는 그 전 저녁 공개분) |
| b8 | b7 | 7 | soxx_brk20_prev | =0 | 0 | 종목+ETF | 이탈 거래일 시가 전(전 거래일 종가 또는 그 전 저녁 공개분) |
| b8 | b7 | 7 | qqq_brk20_prev | =0 | 0 | 종목+ETF | 이탈 거래일 시가 전(전 거래일 종가 또는 그 전 저녁 공개분) |

## 조합 규칙

- 같은 N 안에서만, 서로 다른 갈래 원자 2개 또는 3개의 AND(같은 갈래 원자끼리는 묶지 않는다). 모집단 = 모든 원자가 정의된 사건(갈래 1 원자가 끼면 종목만). 켜짐 = 모든 원자 켜짐, 꺼짐 = 모집단 나머지.
- 가족 D(일봉 사건, 결과 = 하네스 short_ret_h10, 초과 = excess_h10): N20 원자 11개(b1 2, b3 2, b4 3, b5 1, b7 3), N40 4개(b1·b3·b5·b7), N60 4개(b1·b3·b4·b5).
- 가족 b2(확인 대기 진입, 종목): bounce N60·N40·N20, hold2 N20 × 같은 N 일봉 원자 1개. 갈래 2와 같은 쌓기 — 켜짐 행 = 변형이 진입했고 원자가 켜진 사건의 변형 손익(거래당), 꺼짐 행 = 같은 N 종목 모든 사건의 t+1 시가 즉시 진입 손익. 확인 대기 변형은 진입 규칙이라 2개 조합(변형 + 원자 1개)만 만든다(3개면 106칸으로 상한 300을 넘김).
- 가족 b8(60분봉 사건, 갈래 8 저장 사건표, 결과 = 70봉 short_ret_h10): b8 원자 3개 × 시장 원자 6개(갈래 3 VIX, 갈래 4 NVDL·풋/콜, 갈래 7 SOXX·QQQ·둘 중 하나 일봉 20일 박스 이탈 상태). 시장 원자는 모두 이탈 거래일 시가 전에 알려진 값(VIX·박스 상태는 전 거래일 종가, b4 값은 그날 시가 전 공개분)이고 문턱은 b8 앞 구간 같은 N 사건 3분위다. 갈래 7 원자는 일봉 사건과 같은 '사건과 같은 N' 대신 갈래 7 상위 칸의 N20 일봉 박스를 쓴다(60분봉 N과 일봉 N이 다르다).
- 쓰지 않는 것: b8 N14 tod_close=0(앞 구간 꺼짐 1건뿐인 퇴화 칸), b5 sub_cobreak_share ≤ 0(sub_cobreak_n ≤ 0과 같은 사건 집합), 과제 목록 밖 특징.

## 로지스틱(1개)

- 학습: 가족 D 앞 구간(2026-01-02~05-29) 사건 전체(N20·40·60, 종목+ETF; 특징 8개가 모두 있는 사건). 목표 = 하네스 분류 '진짜'(t+1..t+10 안에 L 위 종가 전에 저가 ≤ L − 2·ATR20)면 1, 가짜·중간이면 0. 앞 구간 끝 사건의 분류는 6월 중순까지 정해진다(하네스 앞 구간 고르기와 같은 처리).
- 특징 8개: ret5_pre, depth_atr, gap_size, cobreak_share, vix, nvdl_flow20_p, pc_equity5_p, soxx_brk(갈래 1 두 개, 갈래 3 두 개, 갈래 4 두 개, 갈래 5 하나, 갈래 7 하나). 모두 t 종가(또는 t 저녁 공개) 값. 학습 사건 평균·표준편차로 표준화, sklearn LogisticRegression(L2, C = 0.1 고정, lbfgs).
- 칸: N마다 그 N 앞 구간 사건(종목+ETF)의 예측확률 2/3 분위 이상 = 켜짐, 나머지 = 꺼짐(특징 결측은 모집단 밖). 3칸.

## 판정(하네스 그대로)

- 칸 248개(조합 245 + 로지스틱 3)를 쌓은 표 하나에 넣고 `evaluate_cells` 한 번: 앞 구간 차이(켜짐 − 꺼짐 short_ret_h10) 상위 10칸(켜짐 ≥ 15건, 서로 다른 날짜 ≥ 8일), 뒤 구간 같은 방향·날짜 묶음 t ≥ 2·날짜 묶음 순열 200번 p < 0.05·켜짐 평균 > 0이면 통과. 선택편향 귀무 200번, 층 = 가족·N·kind, 한 국면 비중 > 50% 표시.
- 가족별 칸 수: {'D': 164, 'b8': 51, 'b2bounce': 19, 'b2hold2': 11}, 가족:원자 수별: {'D:3': 105, 'D:2': 59, 'b8:3': 33, 'b2:1': 30, 'b8:2': 18}.
- 칸 목록 sha256: `5a789d107e7c00a960257cf1f61762641d688beeb6282abaa848a4533aae8127`

## 칸 목록(순서 = 판정 입력 순서)

1. `D|N20|stock|ret5_pre≥-0.02387 & cobreak_share≤0.0354`
2. `D|N20|stock|ret5_pre≥-0.02387 & vix≤18.38`
3. `D|N20|stock|ret5_pre≥-0.02387 & nvdl_flow20_p≥0.6426`
4. `D|N20|stock|ret5_pre≥-0.02387 & pc_equity5_p≥0.8447`
5. `D|N20|stock|ret5_pre≥-0.02387 & si_dtc≥2.83`
6. `D|N20|stock|ret5_pre≥-0.02387 & gap_size≤0.01087`
7. `D|N20|stock|ret5_pre≥-0.02387 & soxx_brk=0`
8. `D|N20|stock|ret5_pre≥-0.02387 & sq_brk=0`
9. `D|N20|stock|ret5_pre≥-0.02387 & qqq_brk=0`
10. `D|N20|stock|depth_atr≥0.4892 & cobreak_share≤0.0354`
11. `D|N20|stock|depth_atr≥0.4892 & vix≤18.38`
12. `D|N20|stock|depth_atr≥0.4892 & nvdl_flow20_p≥0.6426`
13. `D|N20|stock|depth_atr≥0.4892 & pc_equity5_p≥0.8447`
14. `D|N20|stock|depth_atr≥0.4892 & si_dtc≥2.83`
15. `D|N20|stock|depth_atr≥0.4892 & gap_size≤0.01087`
16. `D|N20|stock|depth_atr≥0.4892 & soxx_brk=0`
17. `D|N20|stock|depth_atr≥0.4892 & sq_brk=0`
18. `D|N20|stock|depth_atr≥0.4892 & qqq_brk=0`
19. `D|N20|cobreak_share≤0.0354 & nvdl_flow20_p≥0.6426`
20. `D|N20|cobreak_share≤0.0354 & pc_equity5_p≥0.8447`
21. `D|N20|cobreak_share≤0.0354 & si_dtc≥2.83`
22. `D|N20|cobreak_share≤0.0354 & gap_size≤0.01087`
23. `D|N20|cobreak_share≤0.0354 & soxx_brk=0`
24. `D|N20|cobreak_share≤0.0354 & sq_brk=0`
25. `D|N20|cobreak_share≤0.0354 & qqq_brk=0`
26. `D|N20|vix≤18.38 & nvdl_flow20_p≥0.6426`
27. `D|N20|vix≤18.38 & pc_equity5_p≥0.8447`
28. `D|N20|vix≤18.38 & si_dtc≥2.83`
29. `D|N20|vix≤18.38 & gap_size≤0.01087`
30. `D|N20|vix≤18.38 & soxx_brk=0`
31. `D|N20|vix≤18.38 & sq_brk=0`
32. `D|N20|vix≤18.38 & qqq_brk=0`
33. `D|N20|nvdl_flow20_p≥0.6426 & gap_size≤0.01087`
34. `D|N20|nvdl_flow20_p≥0.6426 & soxx_brk=0`
35. `D|N20|nvdl_flow20_p≥0.6426 & sq_brk=0`
36. `D|N20|nvdl_flow20_p≥0.6426 & qqq_brk=0`
37. `D|N20|pc_equity5_p≥0.8447 & gap_size≤0.01087`
38. `D|N20|pc_equity5_p≥0.8447 & soxx_brk=0`
39. `D|N20|pc_equity5_p≥0.8447 & sq_brk=0`
40. `D|N20|pc_equity5_p≥0.8447 & qqq_brk=0`
41. `D|N20|si_dtc≥2.83 & gap_size≤0.01087`
42. `D|N20|si_dtc≥2.83 & soxx_brk=0`
43. `D|N20|si_dtc≥2.83 & sq_brk=0`
44. `D|N20|si_dtc≥2.83 & qqq_brk=0`
45. `D|N20|gap_size≤0.01087 & soxx_brk=0`
46. `D|N20|gap_size≤0.01087 & sq_brk=0`
47. `D|N20|gap_size≤0.01087 & qqq_brk=0`
48. `D|N20|stock|ret5_pre≥-0.02387 & cobreak_share≤0.0354 & nvdl_flow20_p≥0.6426`
49. `D|N20|stock|ret5_pre≥-0.02387 & cobreak_share≤0.0354 & pc_equity5_p≥0.8447`
50. `D|N20|stock|ret5_pre≥-0.02387 & cobreak_share≤0.0354 & si_dtc≥2.83`
51. `D|N20|stock|ret5_pre≥-0.02387 & cobreak_share≤0.0354 & gap_size≤0.01087`
52. `D|N20|stock|ret5_pre≥-0.02387 & cobreak_share≤0.0354 & soxx_brk=0`
53. `D|N20|stock|ret5_pre≥-0.02387 & cobreak_share≤0.0354 & sq_brk=0`
54. `D|N20|stock|ret5_pre≥-0.02387 & cobreak_share≤0.0354 & qqq_brk=0`
55. `D|N20|stock|ret5_pre≥-0.02387 & vix≤18.38 & nvdl_flow20_p≥0.6426`
56. `D|N20|stock|ret5_pre≥-0.02387 & vix≤18.38 & pc_equity5_p≥0.8447`
57. `D|N20|stock|ret5_pre≥-0.02387 & vix≤18.38 & si_dtc≥2.83`
58. `D|N20|stock|ret5_pre≥-0.02387 & vix≤18.38 & gap_size≤0.01087`
59. `D|N20|stock|ret5_pre≥-0.02387 & vix≤18.38 & soxx_brk=0`
60. `D|N20|stock|ret5_pre≥-0.02387 & vix≤18.38 & sq_brk=0`
61. `D|N20|stock|ret5_pre≥-0.02387 & vix≤18.38 & qqq_brk=0`
62. `D|N20|stock|ret5_pre≥-0.02387 & nvdl_flow20_p≥0.6426 & gap_size≤0.01087`
63. `D|N20|stock|ret5_pre≥-0.02387 & nvdl_flow20_p≥0.6426 & soxx_brk=0`
64. `D|N20|stock|ret5_pre≥-0.02387 & nvdl_flow20_p≥0.6426 & sq_brk=0`
65. `D|N20|stock|ret5_pre≥-0.02387 & nvdl_flow20_p≥0.6426 & qqq_brk=0`
66. `D|N20|stock|ret5_pre≥-0.02387 & pc_equity5_p≥0.8447 & gap_size≤0.01087`
67. `D|N20|stock|ret5_pre≥-0.02387 & pc_equity5_p≥0.8447 & soxx_brk=0`
68. `D|N20|stock|ret5_pre≥-0.02387 & pc_equity5_p≥0.8447 & sq_brk=0`
69. `D|N20|stock|ret5_pre≥-0.02387 & pc_equity5_p≥0.8447 & qqq_brk=0`
70. `D|N20|stock|ret5_pre≥-0.02387 & si_dtc≥2.83 & gap_size≤0.01087`
71. `D|N20|stock|ret5_pre≥-0.02387 & si_dtc≥2.83 & soxx_brk=0`
72. `D|N20|stock|ret5_pre≥-0.02387 & si_dtc≥2.83 & sq_brk=0`
73. `D|N20|stock|ret5_pre≥-0.02387 & si_dtc≥2.83 & qqq_brk=0`
74. `D|N20|stock|ret5_pre≥-0.02387 & gap_size≤0.01087 & soxx_brk=0`
75. `D|N20|stock|ret5_pre≥-0.02387 & gap_size≤0.01087 & sq_brk=0`
76. `D|N20|stock|ret5_pre≥-0.02387 & gap_size≤0.01087 & qqq_brk=0`
77. `D|N20|stock|depth_atr≥0.4892 & cobreak_share≤0.0354 & nvdl_flow20_p≥0.6426`
78. `D|N20|stock|depth_atr≥0.4892 & cobreak_share≤0.0354 & pc_equity5_p≥0.8447`
79. `D|N20|stock|depth_atr≥0.4892 & cobreak_share≤0.0354 & si_dtc≥2.83`
80. `D|N20|stock|depth_atr≥0.4892 & cobreak_share≤0.0354 & gap_size≤0.01087`
81. `D|N20|stock|depth_atr≥0.4892 & cobreak_share≤0.0354 & soxx_brk=0`
82. `D|N20|stock|depth_atr≥0.4892 & cobreak_share≤0.0354 & sq_brk=0`
83. `D|N20|stock|depth_atr≥0.4892 & cobreak_share≤0.0354 & qqq_brk=0`
84. `D|N20|stock|depth_atr≥0.4892 & vix≤18.38 & nvdl_flow20_p≥0.6426`
85. `D|N20|stock|depth_atr≥0.4892 & vix≤18.38 & pc_equity5_p≥0.8447`
86. `D|N20|stock|depth_atr≥0.4892 & vix≤18.38 & si_dtc≥2.83`
87. `D|N20|stock|depth_atr≥0.4892 & vix≤18.38 & gap_size≤0.01087`
88. `D|N20|stock|depth_atr≥0.4892 & vix≤18.38 & soxx_brk=0`
89. `D|N20|stock|depth_atr≥0.4892 & vix≤18.38 & sq_brk=0`
90. `D|N20|stock|depth_atr≥0.4892 & vix≤18.38 & qqq_brk=0`
91. `D|N20|stock|depth_atr≥0.4892 & nvdl_flow20_p≥0.6426 & gap_size≤0.01087`
92. `D|N20|stock|depth_atr≥0.4892 & nvdl_flow20_p≥0.6426 & soxx_brk=0`
93. `D|N20|stock|depth_atr≥0.4892 & nvdl_flow20_p≥0.6426 & sq_brk=0`
94. `D|N20|stock|depth_atr≥0.4892 & nvdl_flow20_p≥0.6426 & qqq_brk=0`
95. `D|N20|stock|depth_atr≥0.4892 & pc_equity5_p≥0.8447 & gap_size≤0.01087`
96. `D|N20|stock|depth_atr≥0.4892 & pc_equity5_p≥0.8447 & soxx_brk=0`
97. `D|N20|stock|depth_atr≥0.4892 & pc_equity5_p≥0.8447 & sq_brk=0`
98. `D|N20|stock|depth_atr≥0.4892 & pc_equity5_p≥0.8447 & qqq_brk=0`
99. `D|N20|stock|depth_atr≥0.4892 & si_dtc≥2.83 & gap_size≤0.01087`
100. `D|N20|stock|depth_atr≥0.4892 & si_dtc≥2.83 & soxx_brk=0`
101. `D|N20|stock|depth_atr≥0.4892 & si_dtc≥2.83 & sq_brk=0`
102. `D|N20|stock|depth_atr≥0.4892 & si_dtc≥2.83 & qqq_brk=0`
103. `D|N20|stock|depth_atr≥0.4892 & gap_size≤0.01087 & soxx_brk=0`
104. `D|N20|stock|depth_atr≥0.4892 & gap_size≤0.01087 & sq_brk=0`
105. `D|N20|stock|depth_atr≥0.4892 & gap_size≤0.01087 & qqq_brk=0`
106. `D|N20|cobreak_share≤0.0354 & nvdl_flow20_p≥0.6426 & gap_size≤0.01087`
107. `D|N20|cobreak_share≤0.0354 & nvdl_flow20_p≥0.6426 & soxx_brk=0`
108. `D|N20|cobreak_share≤0.0354 & nvdl_flow20_p≥0.6426 & sq_brk=0`
109. `D|N20|cobreak_share≤0.0354 & nvdl_flow20_p≥0.6426 & qqq_brk=0`
110. `D|N20|cobreak_share≤0.0354 & pc_equity5_p≥0.8447 & gap_size≤0.01087`
111. `D|N20|cobreak_share≤0.0354 & pc_equity5_p≥0.8447 & soxx_brk=0`
112. `D|N20|cobreak_share≤0.0354 & pc_equity5_p≥0.8447 & sq_brk=0`
113. `D|N20|cobreak_share≤0.0354 & pc_equity5_p≥0.8447 & qqq_brk=0`
114. `D|N20|cobreak_share≤0.0354 & si_dtc≥2.83 & gap_size≤0.01087`
115. `D|N20|cobreak_share≤0.0354 & si_dtc≥2.83 & soxx_brk=0`
116. `D|N20|cobreak_share≤0.0354 & si_dtc≥2.83 & sq_brk=0`
117. `D|N20|cobreak_share≤0.0354 & si_dtc≥2.83 & qqq_brk=0`
118. `D|N20|cobreak_share≤0.0354 & gap_size≤0.01087 & soxx_brk=0`
119. `D|N20|cobreak_share≤0.0354 & gap_size≤0.01087 & sq_brk=0`
120. `D|N20|cobreak_share≤0.0354 & gap_size≤0.01087 & qqq_brk=0`
121. `D|N20|vix≤18.38 & nvdl_flow20_p≥0.6426 & gap_size≤0.01087`
122. `D|N20|vix≤18.38 & nvdl_flow20_p≥0.6426 & soxx_brk=0`
123. `D|N20|vix≤18.38 & nvdl_flow20_p≥0.6426 & sq_brk=0`
124. `D|N20|vix≤18.38 & nvdl_flow20_p≥0.6426 & qqq_brk=0`
125. `D|N20|vix≤18.38 & pc_equity5_p≥0.8447 & gap_size≤0.01087`
126. `D|N20|vix≤18.38 & pc_equity5_p≥0.8447 & soxx_brk=0`
127. `D|N20|vix≤18.38 & pc_equity5_p≥0.8447 & sq_brk=0`
128. `D|N20|vix≤18.38 & pc_equity5_p≥0.8447 & qqq_brk=0`
129. `D|N20|vix≤18.38 & si_dtc≥2.83 & gap_size≤0.01087`
130. `D|N20|vix≤18.38 & si_dtc≥2.83 & soxx_brk=0`
131. `D|N20|vix≤18.38 & si_dtc≥2.83 & sq_brk=0`
132. `D|N20|vix≤18.38 & si_dtc≥2.83 & qqq_brk=0`
133. `D|N20|vix≤18.38 & gap_size≤0.01087 & soxx_brk=0`
134. `D|N20|vix≤18.38 & gap_size≤0.01087 & sq_brk=0`
135. `D|N20|vix≤18.38 & gap_size≤0.01087 & qqq_brk=0`
136. `D|N20|nvdl_flow20_p≥0.6426 & gap_size≤0.01087 & soxx_brk=0`
137. `D|N20|nvdl_flow20_p≥0.6426 & gap_size≤0.01087 & sq_brk=0`
138. `D|N20|nvdl_flow20_p≥0.6426 & gap_size≤0.01087 & qqq_brk=0`
139. `D|N20|pc_equity5_p≥0.8447 & gap_size≤0.01087 & soxx_brk=0`
140. `D|N20|pc_equity5_p≥0.8447 & gap_size≤0.01087 & sq_brk=0`
141. `D|N20|pc_equity5_p≥0.8447 & gap_size≤0.01087 & qqq_brk=0`
142. `D|N20|si_dtc≥2.83 & gap_size≤0.01087 & soxx_brk=0`
143. `D|N20|si_dtc≥2.83 & gap_size≤0.01087 & sq_brk=0`
144. `D|N20|si_dtc≥2.83 & gap_size≤0.01087 & qqq_brk=0`
145. `D|N40|stock|gap_break=0 & cobreak_share≤0.02655`
146. `D|N40|stock|gap_break=0 & sub_cobreak_n≤0`
147. `D|N40|stock|gap_break=0 & soxx_brk=0`
148. `D|N40|cobreak_share≤0.02655 & sub_cobreak_n≤0`
149. `D|N40|cobreak_share≤0.02655 & soxx_brk=0`
150. `D|N40|sub_cobreak_n≤0 & soxx_brk=0`
151. `D|N40|stock|gap_break=0 & cobreak_share≤0.02655 & sub_cobreak_n≤0`
152. `D|N40|stock|gap_break=0 & cobreak_share≤0.02655 & soxx_brk=0`
153. `D|N40|stock|gap_break=0 & sub_cobreak_n≤0 & soxx_brk=0`
154. `D|N40|cobreak_share≤0.02655 & sub_cobreak_n≤0 & soxx_brk=0`
155. `D|N60|stock|gap_break=0 & rs20_qqq≥-0.08175`
156. `D|N60|stock|gap_break=0 & shvr_z20≥-0.1192`
157. `D|N60|stock|gap_break=0 & earn_pm3=0`
158. `D|N60|rs20_qqq≥-0.08175 & shvr_z20≥-0.1192`
159. `D|N60|rs20_qqq≥-0.08175 & earn_pm3=0`
160. `D|N60|shvr_z20≥-0.1192 & earn_pm3=0`
161. `D|N60|stock|gap_break=0 & rs20_qqq≥-0.08175 & shvr_z20≥-0.1192`
162. `D|N60|stock|gap_break=0 & rs20_qqq≥-0.08175 & earn_pm3=0`
163. `D|N60|stock|gap_break=0 & shvr_z20≥-0.1192 & earn_pm3=0`
164. `D|N60|rs20_qqq≥-0.08175 & shvr_z20≥-0.1192 & earn_pm3=0`
165. `b2bounce|N60|stock|gap_break=0`
166. `b2bounce|N60|stock|rs20_qqq≥-0.08175`
167. `b2bounce|N60|stock|shvr_z20≥-0.1192`
168. `b2bounce|N60|stock|earn_pm3=0`
169. `b2bounce|N40|stock|gap_break=0`
170. `b2bounce|N40|stock|cobreak_share≤0.02655`
171. `b2bounce|N40|stock|sub_cobreak_n≤0`
172. `b2bounce|N40|stock|soxx_brk=0`
173. `b2bounce|N20|stock|ret5_pre≥-0.02387`
174. `b2bounce|N20|stock|depth_atr≥0.4892`
175. `b2bounce|N20|stock|cobreak_share≤0.0354`
176. `b2bounce|N20|stock|vix≤18.38`
177. `b2bounce|N20|stock|nvdl_flow20_p≥0.6426`
178. `b2bounce|N20|stock|pc_equity5_p≥0.8447`
179. `b2bounce|N20|stock|si_dtc≥2.83`
180. `b2bounce|N20|stock|gap_size≤0.01087`
181. `b2bounce|N20|stock|soxx_brk=0`
182. `b2bounce|N20|stock|sq_brk=0`
183. `b2bounce|N20|stock|qqq_brk=0`
184. `b2hold2|N20|stock|ret5_pre≥-0.02387`
185. `b2hold2|N20|stock|depth_atr≥0.4892`
186. `b2hold2|N20|stock|cobreak_share≤0.0354`
187. `b2hold2|N20|stock|vix≤18.38`
188. `b2hold2|N20|stock|nvdl_flow20_p≥0.6426`
189. `b2hold2|N20|stock|pc_equity5_p≥0.8447`
190. `b2hold2|N20|stock|si_dtc≥2.83`
191. `b2hold2|N20|stock|gap_size≤0.01087`
192. `b2hold2|N20|stock|soxx_brk=0`
193. `b2hold2|N20|stock|sq_brk=0`
194. `b2hold2|N20|stock|qqq_brk=0`
195. `b8|N28|depth_atr≤0.3454 & vix_prev≤20.44`
196. `b8|N28|depth_atr≤0.3454 & nvdl_D≥0.842`
197. `b8|N28|depth_atr≤0.3454 & pc_D≥0.7731`
198. `b8|N28|depth_atr≤0.3454 & sq_brk20_prev=0`
199. `b8|N28|depth_atr≤0.3454 & soxx_brk20_prev=0`
200. `b8|N28|depth_atr≤0.3454 & qqq_brk20_prev=0`
201. `b8|N28|depth_atr≤0.3454 & vix_prev≤20.44 & nvdl_D≥0.842`
202. `b8|N28|depth_atr≤0.3454 & vix_prev≤20.44 & pc_D≥0.7731`
203. `b8|N28|depth_atr≤0.3454 & vix_prev≤20.44 & sq_brk20_prev=0`
204. `b8|N28|depth_atr≤0.3454 & vix_prev≤20.44 & soxx_brk20_prev=0`
205. `b8|N28|depth_atr≤0.3454 & vix_prev≤20.44 & qqq_brk20_prev=0`
206. `b8|N28|depth_atr≤0.3454 & nvdl_D≥0.842 & sq_brk20_prev=0`
207. `b8|N28|depth_atr≤0.3454 & nvdl_D≥0.842 & soxx_brk20_prev=0`
208. `b8|N28|depth_atr≤0.3454 & nvdl_D≥0.842 & qqq_brk20_prev=0`
209. `b8|N28|depth_atr≤0.3454 & pc_D≥0.7731 & sq_brk20_prev=0`
210. `b8|N28|depth_atr≤0.3454 & pc_D≥0.7731 & soxx_brk20_prev=0`
211. `b8|N28|depth_atr≤0.3454 & pc_D≥0.7731 & qqq_brk20_prev=0`
212. `b8|N14|dbox_pos≥0.1937 & vix_prev≤18.41`
213. `b8|N14|dbox_pos≥0.1937 & nvdl_D≥0.8206`
214. `b8|N14|dbox_pos≥0.1937 & pc_D≥0.7718`
215. `b8|N14|dbox_pos≥0.1937 & sq_brk20_prev=0`
216. `b8|N14|dbox_pos≥0.1937 & soxx_brk20_prev=0`
217. `b8|N14|dbox_pos≥0.1937 & qqq_brk20_prev=0`
218. `b8|N14|dbox_pos≥0.1937 & vix_prev≤18.41 & nvdl_D≥0.8206`
219. `b8|N14|dbox_pos≥0.1937 & vix_prev≤18.41 & pc_D≥0.7718`
220. `b8|N14|dbox_pos≥0.1937 & vix_prev≤18.41 & sq_brk20_prev=0`
221. `b8|N14|dbox_pos≥0.1937 & vix_prev≤18.41 & soxx_brk20_prev=0`
222. `b8|N14|dbox_pos≥0.1937 & vix_prev≤18.41 & qqq_brk20_prev=0`
223. `b8|N14|dbox_pos≥0.1937 & nvdl_D≥0.8206 & sq_brk20_prev=0`
224. `b8|N14|dbox_pos≥0.1937 & nvdl_D≥0.8206 & soxx_brk20_prev=0`
225. `b8|N14|dbox_pos≥0.1937 & nvdl_D≥0.8206 & qqq_brk20_prev=0`
226. `b8|N14|dbox_pos≥0.1937 & pc_D≥0.7718 & sq_brk20_prev=0`
227. `b8|N14|dbox_pos≥0.1937 & pc_D≥0.7718 & soxx_brk20_prev=0`
228. `b8|N14|dbox_pos≥0.1937 & pc_D≥0.7718 & qqq_brk20_prev=0`
229. `b8|N7|dbox_pos≥0.4709 & vix_prev≤18`
230. `b8|N7|dbox_pos≥0.4709 & nvdl_D≥0.6827`
231. `b8|N7|dbox_pos≥0.4709 & pc_D≥0.7711`
232. `b8|N7|dbox_pos≥0.4709 & sq_brk20_prev=0`
233. `b8|N7|dbox_pos≥0.4709 & soxx_brk20_prev=0`
234. `b8|N7|dbox_pos≥0.4709 & qqq_brk20_prev=0`
235. `b8|N7|dbox_pos≥0.4709 & vix_prev≤18 & nvdl_D≥0.6827`
236. `b8|N7|dbox_pos≥0.4709 & vix_prev≤18 & pc_D≥0.7711`
237. `b8|N7|dbox_pos≥0.4709 & vix_prev≤18 & sq_brk20_prev=0`
238. `b8|N7|dbox_pos≥0.4709 & vix_prev≤18 & soxx_brk20_prev=0`
239. `b8|N7|dbox_pos≥0.4709 & vix_prev≤18 & qqq_brk20_prev=0`
240. `b8|N7|dbox_pos≥0.4709 & nvdl_D≥0.6827 & sq_brk20_prev=0`
241. `b8|N7|dbox_pos≥0.4709 & nvdl_D≥0.6827 & soxx_brk20_prev=0`
242. `b8|N7|dbox_pos≥0.4709 & nvdl_D≥0.6827 & qqq_brk20_prev=0`
243. `b8|N7|dbox_pos≥0.4709 & pc_D≥0.7711 & sq_brk20_prev=0`
244. `b8|N7|dbox_pos≥0.4709 & pc_D≥0.7711 & soxx_brk20_prev=0`
245. `b8|N7|dbox_pos≥0.4709 & pc_D≥0.7711 & qqq_brk20_prev=0`
246. `D|N20|logit_top3분위`
247. `D|N40|logit_top3분위`
248. `D|N60|logit_top3분위`
