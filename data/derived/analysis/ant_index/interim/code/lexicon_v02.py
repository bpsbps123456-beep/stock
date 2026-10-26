"""개미 지수 사전 v0.2 '후보' 매처 — 가격 전 정밀도 점검(lexicon_draft §8·§9, HYPOTHESES §4·§2.7)용.

v0.2 후보 = lexicon_draft.md v0.1 §4 + HYPOTHESES.md §4 수정
  - E: v0.1 40개에서 PX 7개(불기둥·상한가·점상·신고가·사상최고·빨간불·시뻘)와 '행복회로'(→ MOCK)를 옮겨 32개
  - F: v0.1 40개에서 PX 6개(하한가·점하·서킷브레이커·사이드카·시퍼렇·파란불)를 옮겨 34개
  - KR: '밸류업'(정책어)을 빼 19개
  - 가설별 묶음: BULL/BEAR, MOCK, REBUT, CHASE, CONV, DIST, IPO, MEM, SEMI, TILT, 종목 별칭(lexicon §6)
이 판은 점검 대상(후보)이다. 점검 뒤에는 단어를 '빼기만' 한다(HYPOTHESES §4). 확정 v0.2가 아니다.

셈 규칙: lexicon §1(단위)·§2(잡음 거르기)·§3(부정·질문형)·§4(제외 문맥), HYPOTHESES §2.3(가격 서술 문장)·§4(한 셈 안
가장 긴 표현 먼저, 셈끼리는 따로). 문서가 모호한 곳의 해석은 interim/notes_blind.md에 적었다. 가격은 쓰지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

# ================================================================ 어휘 (표시형 그대로; 매칭은 공백을 지운 소문자)

E_WORDS = ["떡상", "가즈아", "가즈아아", "풀매수", "풀매", "몰빵", "축포", "졸업", "익절", "수익인증", "수익 인증",
           "계좌인증", "수익률 미쳤", "개꿀", "개이득", "대박", "텐배거", "따상", "투더문", "화성 갈", "우주 간다",
           "치킨 쏩", "한턱", "경제적 자유", "하방 없", "무조건 오른", "끝없는 상승", "갓삼전", "갓닉스", "꽃길",
           "영차", "🚀"]
F_WORDS = ["한강", "한강물", "한강 수온", "폭락", "대폭락", "떡락", "나락", "지하실", "지옥", "패닉", "패닉셀", "투매",
           "곡소리", "피눈물", "멘붕", "무섭", "공포", "두렵", "살려", "망했", "끝났", "블랙먼데이", "대공황", "붕괴",
           "반토막", "반의반", "깡통", "계좌 녹", "녹아내", "물렸", "고점에 물", "상투", "떨어지는 칼날", "상폐"]
PX_WORDS = ["상한가", "하한가", "점상", "점하", "신고가", "사상최고", "빨간불", "파란불", "시뻘", "시퍼렇", "불기둥",
            "서킷브레이커", "사이드카"]
CAP_WORDS = ["다 팔았", "전부 팔았", "전량 매도했", "전량 매도함", "손절했", "손절함", "손절 쳤", "손절침", "손절 완료",
             "던졌", "다 던짐", "던지고 나왔", "투매했", "털렸", "털고 나왔", "탈출했", "탈출 완료", "탈출 성공",
             "계좌 정리했", "현금화했", "풀현금", "현금 100", "앱 지웠", "어플 지웠", "증권앱 삭제", "주식 접습니다",
             "주식 끊", "다시는 주식", "손실 확정", "수업료 냈", "매도버튼 눌렀", "풀매도했"]
HES_WORDS = ["손절할까", "손절해야 할까", "손절해야 하나", "손절 타이밍", "팔까요", "팔아야 할까", "팔아야 하나",
             "매도할까", "정리할까", "정리해야 할까", "탈출해야", "버틸까", "버텨야 할까", "들고 가야 할까", "홀딩할까",
             "어떻게 해야", "어쩌죠", "어떡하죠", "어떡해요", "물타기 할까", "물탈까", "지금이라도 팔", "손절이 답일까",
             "존버해야 할까"]
LL_KR_WORDS = ["신용", "신용매수", "신용풀", "미수", "미수거래", "빚투", "영끌", "마통", "마이너스통장", "주담대",
               "담보대출", "대출받아", "대출 받아서", "카드론", "레버리지", "레버", "풀레버", "2배", "3배",
               "KODEX 레버리지", "코덱스 레버리지", "코레버", "TIGER 레버리지", "하닉 2배", "삼전 2배",
               "하이닉스 레버리지", "삼성전자 레버리지"]
LL_US_WORDS = ["SOXL", "속슬", "TQQQ", "티큐", "NVDL", "엔비 2배", "TSLL", "UPRO", "KORU", "콜옵션", "콜 샀"]
LS_WORDS = ["곱버스", "곱버", "인버스", "인버스2X", "인버스 2배", "200선물인버스", "코스닥 인버스", "KODEX 인버스",
            "SQQQ", "스큐큐", "SOXS", "숙스", "속스", "NVDD", "SPXU", "풋옵션", "풋 샀", "풋 매수", "선물 숏", "숏 쳤",
            "숏 잡", "숏 포지션", "하락 베팅", "하락에 베팅", "곱버스 탔", "인버스 탔"]
MC_WORDS = ["반대매매", "반대 매매", "반대매매 물량", "반대매매 나왔", "반대매매 당", "반대매매 맞", "마진콜", "마진 콜",
            "담보부족", "담보 부족", "담보비율", "담보유지비율", "추가증거금", "추가 증거금", "추증", "증거금 부족",
            "강제청산", "강제 청산", "로스컷", "청산당", "청산 당했", "미수 정리", "깡통계좌", "깡통 찼", "신용 정리"]
NOV_WORDS = ["주린이", "주식 초보", "초보입니다", "초보라", "초보 질문", "입문", "처음 주식", "주식 처음", "처음 사봤",
             "첫 매수", "첫 주식", "계좌 개설", "계좌 처음", "증권사 어디", "어느 증권사", "어떤 증권사", "지금 사도",
             "지금 들어가도", "지금 들어가면", "지금 진입", "추천 좀", "추천해주", "추천 부탁", "종목 추천", "뭐 사요",
             "뭐 살까", "뭘 사야", "어디에 넣", "적금 깨", "예금 깨", "퇴직금 넣", "비상금 넣", "엄마가 주식",
             "와이프가 주식", "다들 주식 얘기", "시드 얼마"]
NOV_QONLY = {"지금사도", "지금들어가도", "지금들어가면", "지금진입"}   # lexicon §3 '지금 사도·지금 들어가도류'
FO_WORDS = ["벼락거지", "나만 못", "나만 없", "나만 안", "나 빼고", "못 탔", "못 샀", "놓쳤", "기차 떠났", "버스 떠났",
            "열차 떠났", "부럽", "배아프", "배 아프", "박탈감", "소외", "현타", "다들 벌었", "남들 다 벌", "지금이라도 탈까",
            "지금 타도", "늦었나요", "늦은 건가", "올라탈까", "추격매수", "추격 매수", "팔고 나니", "팔자마자 오르",
            "일찍 팔았", "더 들고 있을걸", "살걸", "샀어야", "그때 살걸", "안 사고 뭐"]
RES_WORDS = ["존버", "존버중", "강제 존버", "강제존버", "장투", "강제 장투", "강제장투", "자식에게", "자식한테", "물려줄",
             "물려준다", "손주", "묻어두", "묻어놓", "잊고 살", "잊고 지내", "앱 안 봄", "계좌 안 봄", "안 쳐다", "10년 뒤",
             "10년 묵", "언젠가 오르", "본전만", "본전 오면", "원금만", "원금 회복", "물탈 돈도 없", "물타기 포기", "해탈",
             "득도", "무념무상", "될 대로 되라", "못 팔고", "손절 못"]
ANG_WORDS = ["공매도 세력", "공매 세력", "공매충", "공매도 놈", "공매 밥", "세력", "작전주", "작전 세력", "주가조작",
             "개미털기", "개미 털기", "물량 털기", "설거지", "외인놈", "외인 놈", "외국인 놈", "외궈", "검머외", "기관놈",
             "기관 놈", "연기금 놈", "연기금 매도", "국민연금 매도", "선행매매", "짜고 치", "주가누르기", "주가 누르기",
             "쓰레기 국장", "국장 개판", "사기꾼", "개미 학살", "개미만 죽", "개미 등쳐", "개미 피", "금감원 뭐하",
             "자전거래"]
KX_WORDS = ["국장 탈출", "국장탈출", "탈출은 지능순", "지능순", "국장 접", "국장 손절", "국장 떠나", "국장 안 해",
            "국장은 답 없", "국장 노답", "국장 하는 사람", "미장으로", "미장 이민", "미장 갑니다", "미장만", "미장이 답",
            "미국 주식으로 갈아", "서학개미 됐", "코리아 디스카운트", "코스피는 안 돼", "미국 가야", "국장 쳐다도"]
KR_WORDS = ["국장 복귀", "국장 돌아", "국장으로 돌아", "국장이 답", "국장 최고", "국장 만세", "국장 효자", "국장이 미장보다",
            "미장보다 국장", "국장이 낫", "미장 탈출", "미장 끝났", "미국도 끝", "미국 끝났", "미장 손절", "국뽕",
            "코리아 프리미엄", "동학개미 승리", "국장 수익률 최고"]
BUY_WORDS = ["줍줍", "줍는다", "주워", "줍줍각", "저점 매수", "저가 매수", "저점 잡", "바닥 매수", "바닥 잡", "분할 매수",
             "분할매수", "추매", "추가 매수", "물타기", "물탔", "담았", "담는다", "담을 기회", "바겐세일", "세일 중",
             "매수 기회", "싸게 살", "줍줍 타임", "기회다"]
BULL_WORDS = ["가즈아", "떡상", "풀매수", "줍줍", "반등", "저점매수", "우상향", "바닥", "매수 기회", "간다"]
BEAR_WORDS = ["떡락", "폭락", "고점", "거품", "버블", "탈출해라", "우하향", "설거지", "인버스", "떨어진다"]
MOCK_BEAR_WORDS = ["공매충", "인버스충", "곱버스충", "쫄보", "겁쟁이", "배 아프", "못 탄", "국장 탈출은 지능순"]
#   + 특수 규칙 2개: '주식전문가'(단독), 'ㅋ4개 이상 + 2인칭'
MOCK_BULL_WORDS = ["행복회로", "희망회로", "설거지", "상투", "꼭지", "한강", "물 온도", "흑우", "호구", "응 아니야", "뇌피셜"]
REBUT_WORDS = ["아니요", "아닌데", "글쎄요", "오히려", "헛소리", "근거가", "착각", "뇌피셜", "행복회로", "상투", "꼭지"]
REBUT_FEAR_WORDS = ["줍줍 기회", "바닥", "공포에 사라", "오히려 기회"]
CHASE_BUY_WORDS = ["풀매수", "몰빵", "탑승했", "들어갔", "영끌", "빚투", "신용 썼", "샀습니다", "샀어요", "매수했"]
CHASE_WATER_WORDS = ["물탔", "추매", "줍줍", "저점 매수했"]
CHASE_SELL_WORDS = ["팔았", "매도했", "익절했", "손절했", "정리했", "털었"]
CONV_BULL_WORDS = ["결국 샀", "나도 탑승", "참다 참다", "못 참고", "포모"]
CONV_BEAR_WORDS = ["다 던졌", "존버 포기", "전량 매도", "주식 접", "멘탈 나갔"]
DIST_WORDS = ["상폐", "상장폐지", "거래정지", "관리종목", "의견거절", "정리매매", "횡령", "배임", "법정관리", "회생절차", "부도"]
IPO_WORDS = ["공모주", "청약", "공모가", "수요예측", "경쟁률", "균등배정", "따상", "따따블", "상장첫날", "시초가", "보호예수",
             "스팩"]
MEM_WORDS = ["하닉", "하이닉스", "삼전", "메모리", "HBM", "D램", "디램", "낸드", "마이크론", "샌디스크", "웨디", "슈퍼사이클"]
SEMI_WORDS = ["하이닉스", "하닉", "삼전", "HBM", "메모리", "디램", "낸드", "엔비디아", "마이크론", "샌디스크", "브로드컴",
              "TSMC", "필반", "SOXL", "반도체"]
TILT_US_WORDS = ["미장", "미국장", "나스닥", "서학", "엔비디아", "엔비", "마이크론", "브로드컴", "팔란티어", "테슬라", "필반",
                 "SOXL"]
TILT_KR_WORDS = ["코스피", "국장", "삼전", "하이닉스", "코스닥", "넥장", "상한가", "외인", "동시호가", "연기금", "SKHY"]
ALIAS_SEC_WORDS = ["삼성전자", "삼전", "갓삼전", "삼성 전자", "005930", "삼성"]
ALIAS_SKH_WORDS = ["SK하이닉스", "하이닉스", "하이닉", "하닉", "닉스", "SK하닉", "갓닉스", "000660", "SKHY"]
ALIAS_BOTH_WORDS = ["삼전닉스", "삼닉", "삼하"]
ALIAS_US_WORDS = ["엔비디아", "엔비", "엔비디", "NVDA", "젠슨", "마이크론", "마크", "MU", "샌디스크", "샌디", "SNDK", "브로드컴",
                  "브컴", "AVGO", "TSMC", "티에스엠", "AMD", "웨스턴디지털", "WDC", "DELL", "팔란티어", "팔란", "PLTR",
                  "테슬라", "테슬", "TSLA"]
CASE_SENSITIVE = {"MU", "DELL"}   # lexicon §6: 델은 대문자 DELL만. MU는 소문자 영단어와 겹쳐 대문자만(해석)

# (셈 그룹, 묶음, 층(점검 단위), 꼬리표, 단어들)
SPEC: list[tuple[str, str, str, str, list[str]]] = [
    ("emo", "E", "E", "환호", E_WORDS),
    ("emo", "F", "F", "공포", F_WORDS),
    ("emo", "PX", "PX", "가격 서술", PX_WORDS),
    ("emo", "CAP", "CAP", "항복 완료", CAP_WORDS),
    ("emo", "HES", "HES", "매도 고민", HES_WORDS),
    ("emo", "LL", "LL_KR", "레버리지 롱(국내)", LL_KR_WORDS),
    ("emo", "LL", "LL_US", "레버리지 롱(미국)", LL_US_WORDS),
    ("emo", "LS", "LS", "레버리지 숏", LS_WORDS),
    ("emo", "MC", "MC", "반대매매", MC_WORDS),
    ("emo", "NOV", "NOV", "초보 유입", NOV_WORDS),
    ("emo", "FO", "FO", "소외", FO_WORDS),
    ("emo", "RES", "RES", "체념·존버", RES_WORDS),
    ("emo", "ANG", "ANG", "분노·음모", ANG_WORDS),
    ("emo", "KX", "KX", "국장 이탈", KX_WORDS),
    ("emo", "KR", "KR", "국장 회귀", KR_WORDS),
    ("emo", "BUY", "BUY", "저가 매수", BUY_WORDS),
    ("score", "BULLBEAR", "BULL", "강세", BULL_WORDS),
    ("score", "BULLBEAR", "BEAR", "약세", BEAR_WORDS),
    ("mock", "MOCK", "MOCK_BEAR", "약세론 조롱", MOCK_BEAR_WORDS),
    ("mock", "MOCK", "MOCK_BULL", "강세론 조롱", MOCK_BULL_WORDS),
    ("rebut", "REBUT", "REBUT", "반박", REBUT_WORDS),
    ("rebut", "REBUT", "REBUT_FEAR", "반박(공포 쪽)", REBUT_FEAR_WORDS),
    ("chase", "CHASE", "CHASE_BUY", "추격 매수 고백", CHASE_BUY_WORDS),
    ("chase", "CHASE", "CHASE_BUY", "물타기 고백", CHASE_WATER_WORDS),
    ("chase", "CHASE", "CHASE_SELL", "매도 고백(대조)", CHASE_SELL_WORDS),
    ("conv", "CONV", "CONV_BULL", "강세 전향", CONV_BULL_WORDS),
    ("conv", "CONV", "CONV_BEAR", "약세 전향", CONV_BEAR_WORDS),
    ("dist", "DIST", "DIST", "부실", DIST_WORDS),
    ("ipo", "IPO", "IPO", "공모주", IPO_WORDS),
    ("mem", "MEM", "MEM", "메모리", MEM_WORDS),
    ("semi", "SEMI", "SEMI", "반도체 태그", SEMI_WORDS),
    ("tilt", "TILT", "TILT_US", "미국 쪽", TILT_US_WORDS),
    ("tilt", "TILT", "TILT_KR", "국내 쪽", TILT_KR_WORDS),
    ("alias", "ALIAS", "ALIAS_SEC", "삼성전자", ALIAS_SEC_WORDS),
    ("alias", "ALIAS", "ALIAS_SKH", "SK하이닉스", ALIAS_SKH_WORDS),
    ("alias", "ALIAS", "ALIAS_BOTH", "삼전·하닉 둘 다", ALIAS_BOTH_WORDS),
    ("alias", "ALIAS", "ALIAS_US", "미국 AI 종목", ALIAS_US_WORDS),
]

# 점검 층의 단위 범위(가설이 쓰는 곳에서 잰다): all = 글·댓글, post = 글(제목+본문), comment = 댓글
STRATUM_SCOPE = {s: "all" for (g, _, s, _, _) in SPEC if g in ("emo", "dist", "ipo", "alias")}
STRATUM_SCOPE.update({"BULL": "post", "BEAR": "post", "MOCK_BEAR": "comment", "MOCK_BULL": "comment",
                      "REBUT": "comment", "REBUT_FEAR": "comment", "CHASE_BUY": "post", "CHASE_SELL": "post",
                      "CONV_BULL": "post", "CONV_BEAR": "post", "MEM": "post", "SEMI": "post",
                      "TILT_US": "post", "TILT_KR": "post"})
STRATA = list(dict.fromkeys(s for (_, _, s, _, _) in SPEC))
STRATUM_BUNDLE = {s: b for (_, b, s, _, _) in SPEC}

# ================================================================ 제외 문맥·거르기 (lexicon §2·§3·§4)

# 같은 단위에 있으면 그 단어를 세지 않는다(lexicon §4 '제외'). 키·값 모두 공백을 지운 소문자
EXCL_UNIT = {
    "졸업": ["학교", "대학", "졸업식", "졸업생"],
    "한강": ["한강뷰", "한강변", "한강공원", "한강벨트", "이남"],
    "공포": ["공포탐욕", "공포지수"],
    "붕괴": ["건물", "다리", "교량"],
    "끝났": ["장끝났", "장끝나", "회의", "협상", "수업"],
    "던졌": ["외인", "외국인", "기관", "연기금"],
    "신용": ["신용카드", "신용등급", "신용점수", "국가신용", "신용평가", "신용도", "신용경색"],
    "미수": ["미수령", "미수습"],
    "레버리지": ["etf도입", "규제", "금융위", "정부"],
    "풋옵션": ["외인", "외국인", "기관"],
    "종목추천": ["방송", "유튜브", "리딩방", "리딩"],
    "추격매수": ["신중", "조심", "하지마", "금지"],
    "세력": ["정치", "진영", "친중", "친미", "반대세력"],
    "설거지": ["밥", "그릇", "식기"],
    "마크": ["마크다운", "체크마크"],
}
CAP_PROFIT = ["익절", "수익", "올라서", "벌고"]          # 같은 문장 → 환호(E)로 본다
IPO_EXCL = ["아파트", "분양", "청약통장"]
SAMSUNG_NOT = ["바이오", "sdi", "전기", "생명", "물산", "중공업", "화재", "증권", "e&a", "sds", "에스디에스"]
FIRST_PERSON = ["저는", "제가", "나는", "내계좌", "제계좌", "제평단", "저도"]   # M10(+ 단독 '전')
HOLDING = ["들고", "물려", "계좌"]

NEWS_TITLE = ["단독", "속보", "[기사]", "기사)", "정보]", "브리핑", "시황", "보고서", "리포트"]
NEWS_DOMAINS = ["mk.co.kr", "hankyung", "n.news.naver", "yna.co.kr", "edaily", "mt.co.kr", "chosun", "joongang",
                "donga", "bloomberg", "reuters", "cnbc", "finance.yahoo"]
DELETED = ["코멘트가 삭제되었습니다", "규칙위반", "삭제된 댓글"]
POLITICS = ["이재명", "윤석열", "민주당", "국힘", "국민의힘", "대통령", "정권", "탄핵", "좌파", "우파"]
MARKET = ["주가", "코스피", "코스닥", "매수", "매도", "종목", "반도체", "삼전", "하닉", "미장", "국장", "etf", "수익", "손절",
          "트럼프", "연준", "금리"]

# 가격 서술 문장(HYPOTHESES §2.3): 수치%·포인트 + 지수·종목명 + 움직임 동사가 한 문장에
PRICE_NUM_RX = re.compile(r"\d+(?:[.,]\d+)*(?:%|％|퍼센트|퍼|프로|포인트|pt|p(?![a-z]))")
PRICE_NAMES = ["코스피", "코스닥", "나스닥", "다우", "s&p", "필반", "필라델피아", "지수"]
PRICE_VERBS = ["올랐", "빠졌", "떨어", "상승", "하락", "급등", "급락", "폭락", "폭등"]
SENTIMENT_STRATA = {"E", "F", "CAP", "HES", "FO", "RES", "ANG", "BUY", "KX", "KR", "NOV", "BULL", "BEAR"}

# 부정어(lexicon §3): 단어 뒤 같은 문장 8자 안, 또는 앞 4자 안. '안'·'못'은 띄어 쓴 단독형이나 동사 바로 앞만
NEG_AFTER_RX = re.compile(r"(?<![가-힣])안 |안 함|안함|안해|(?<![가-힣])안(?=팔|사|던)|(?<![가-힣])못 |못함|않|없|아니|"
                          r"아님|아냐|아닌|아닙|말자|말아|마세요|금지|ㄴㄴ")   # 아님·아냐·아닌·아닙: §3 예시 '폭락 아님'(해석)
NEG_BEFORE_RX = re.compile(r"(?<![가-힣])(?:안|못|절대) ")
QUESTION_RX = re.compile(r"\?|까요|나요|할까|될까|되나|어쩌죠|어떡|맞나요|인가요")
INHERENT_NEG = re.compile(r"없|못|않|아니|안")   # 원래 부정 뜻을 담은 표현은 부정 판정에서 뺀다
SENT_DELIM = ".!?\n"

KK4_RX = re.compile(r"ㅋ{4,}")
SECOND_PERSON_RX = re.compile(r"(?<![가-힣])(너희들|너희|너네|너는|너가|너도|니가|니들|니네|님들|님아|님은|님이|님도|"
                              r"당신은|당신이|당신도|당신|댁은|댁|너|님)(?![가-힣])")
EXPERT_ONLY_RX = re.compile(r"[\sㅋㅎㅠㅜ.!?~^…,]+")
QUOTE_RXS = [re.compile(r"(?m)^[ \t]*>.*$"), re.compile(r"“[^”\n]{0,300}”"), re.compile(r"\"[^\"\n]{0,300}\"")]
LATIN_RX = re.compile(r"[a-z]")
SPACE_CH = set(" \t\r 　")


def _compact(s: str) -> str:
    return "".join(ch for ch in s if ch not in SPACE_CH).lower()


def _is_syl(ch: str) -> bool:
    return "가" <= ch <= "힣"


@dataclass(frozen=True)
class Entry:
    group: str
    bundle: str
    stratum: str
    tag: str
    word: str          # 표시형
    key: str           # 공백 지운 소문자
    two_syl: bool      # 2음절 한글 → 원문 왼쪽 낱말 경계
    inherent_neg: bool


def build_entries() -> list[Entry]:
    out: list[Entry] = []
    seen: set[tuple[str, str]] = set()
    for group, bundle, stratum, tag, words in SPEC:
        for w in words:
            k = _compact(w)
            if (group, k) in seen:      # 같은 셈 안에서 공백만 다른 같은 표현(수익인증/수익 인증 등)은 하나로
                continue
            seen.add((group, k))
            out.append(Entry(group, bundle, stratum, tag, w, k,
                             len(k) == 2 and all(_is_syl(c) for c in k), bool(INHERENT_NEG.search(k))))
    return out


ENTRIES = build_entries()


def _check_unique_in_group() -> None:
    by: dict[tuple[str, str], str] = {}
    for e in ENTRIES:
        prev = by.setdefault((e.group, e.key), e.stratum)
        if prev != e.stratum:
            raise ValueError(f"한 셈 안에 같은 표현이 두 묶음에: {e.group} {e.word}")


_check_unique_in_group()


_END = 0   # 끝 표지(문자열이 아닌 키라 본문 글자와 겹치지 않는다)


def _build_trie(entries: list[Entry]) -> dict:
    root: dict = {}
    for e in entries:
        node = root
        for ch in e.key:
            node = node.setdefault(ch, {})
        node.setdefault(_END, []).append(e)
    return root


TRIE = _build_trie(ENTRIES)
ALIAS_KEYS = sorted({e.key for e in ENTRIES if e.group == "alias"}, key=len, reverse=True)


def lexicon_json() -> str:
    """점검 대상 사전(후보)의 정본 직렬화. 이 문자열의 SHA-256이 '점검한 판'의 지문이다."""
    spec = [{"group": g, "bundle": b, "stratum": s, "tag": t, "words": w} for g, b, s, t, w in SPEC]
    rules = {"case_sensitive": sorted(CASE_SENSITIVE), "nov_question_only": sorted(NOV_QONLY), "excl_unit": EXCL_UNIT,
             "cap_profit_sentence": CAP_PROFIT, "ipo_excl": IPO_EXCL, "samsung_not": SAMSUNG_NOT,
             "first_person": FIRST_PERSON, "holding": HOLDING, "news_title": NEWS_TITLE, "news_domains": NEWS_DOMAINS,
             "deleted": DELETED, "politics": POLITICS, "market": MARKET, "price_names": PRICE_NAMES,
             "price_verbs": PRICE_VERBS, "price_num_rx": PRICE_NUM_RX.pattern,
             "sentiment_strata_price_rule": sorted(SENTIMENT_STRATA), "neg_after_rx": NEG_AFTER_RX.pattern,
             "neg_before_rx": NEG_BEFORE_RX.pattern, "question_rx": QUESTION_RX.pattern,
             "second_person_rx": SECOND_PERSON_RX.pattern, "scope": STRATUM_SCOPE}
    return json.dumps({"version": "v0.2-candidate (정밀도 점검 전)", "spec": spec, "rules": rules},
                      ensure_ascii=False, sort_keys=True, indent=1)


def lexicon_sha256() -> str:
    return hashlib.sha256(lexicon_json().encode("utf-8")).hexdigest()


# ================================================================ 단위 분석

@dataclass
class Hit:
    stratum: str
    bundle: str
    word: str
    start: int          # 원문(text) 위치
    end: int
    in_title: bool
    origin: str = ""    # 옮겨 온 경우(예: 'CAP 부정→RES')
    reading: str = ""   # BULL/BEAR: '강세' / '약세' (부정이면 뒤집음)
    sign: int = 0


@dataclass
class UnitResult:
    excluded: str = ""                      # 'news' / 'deleted' / 'politics' / ''
    hits: list[Hit] = field(default_factory=list)
    eq: int = 0                             # 환호 질문형(본 지표 밖)
    fq: int = 0                             # 공포 질문형
    neg_dropped: int = 0
    lev_refuse: int = 0                     # 레버리지 거부
    price_dropped: int = 0                  # 가격 서술 문장이라 뺀 수
    score: int | None = None                # 글 점수(글만)
    m11: bool = False                       # 반어 의심(lexicon §5 M11)


def is_news(title: str, body: str) -> bool:
    t = title or ""
    if t.lstrip().startswith("["):
        return True
    if any(k in t for k in NEWS_TITLE):
        return True
    b = body or ""
    return any(d in b for d in NEWS_DOMAINS)


def blank_quotes(text: str) -> str:
    out = text
    for rx in QUOTE_RXS:
        out = rx.sub(lambda m: " " * len(m.group(0)), out)
    return out


def _compact_map(text: str) -> tuple[str, list[int], str]:
    chars, idx, orig_case = [], [], []
    for i, ch in enumerate(text):
        if ch in SPACE_CH:
            continue
        chars.append(ch.lower())
        orig_case.append(ch)
        idx.append(i)
    return "".join(chars), idx, "".join(orig_case)


def _sentence(text: str, s: int, e: int) -> tuple[int, int]:
    a = s
    while a > 0 and text[a - 1] not in SENT_DELIM:
        a -= 1
    b = e
    while b < len(text) and text[b] not in SENT_DELIM:
        b += 1
    if b < len(text):
        b += 1          # 끝 기호(?)를 문장에 넣는다
    return a, b


def _negated(text: str, s: int, e: int, sent: tuple[int, int], after: int, before: int | None) -> bool:
    a, b = sent
    for m in NEG_AFTER_RX.finditer(text, e, b):
        if m.start() - e < after:
            return True
        break
    if before is not None:
        for m in NEG_BEFORE_RX.finditer(text, a, s):
            if s - m.end() <= before:
                return True
    return False


def _is_question(text: str, sent: tuple[int, int]) -> bool:
    return bool(QUESTION_RX.search(text, sent[0], sent[1]))


def _price_sentence(text: str, sent: tuple[int, int]) -> bool:
    c = _compact(text[sent[0]:sent[1]])
    if not PRICE_NUM_RX.search(c):
        return False
    if not any(v in c for v in PRICE_VERBS):
        return False
    return any(n in c for n in PRICE_NAMES) or any(k in c for k in ALIAS_KEYS)


def _boundary_ok(e: Entry, text: str, os_: int, oe: int) -> bool:
    if e.word in CASE_SENSITIVE and text[os_:oe] != e.word:
        return False
    prev = text[os_ - 1] if os_ > 0 else ""
    nxt = text[oe] if oe < len(text) else ""
    k = e.key
    if LATIN_RX.match(k[0]) and prev and LATIN_RX.match(prev.lower()):
        return False
    if LATIN_RX.match(k[-1]) and nxt and LATIN_RX.match(nxt.lower()):
        return False
    if k[0].isdigit() and prev and (prev.isdigit() or prev in ".,"):
        return False
    if k[-1].isdigit() and nxt and nxt.isdigit():
        return False
    if e.two_syl and prev and _is_syl(prev):
        return False
    return True


def _context_excluded(e: Entry, c: str, ccase: str, i: int, j: int, text: str, os_: int, oe: int,
                      ctx_compact: str) -> bool:
    k = e.key
    if k in EXCL_UNIT and any(x in ctx_compact for x in EXCL_UNIT[k]):
        return True
    if k == "대박" and ("쪽박" in c[max(0, i - 3):i] or "쪽박" in c[j:j + 3]):
        return True
    if e.group == "ipo" and any(x in ctx_compact for x in IPO_EXCL):
        return True
    if k in ("2배", "3배") and e.stratum == "LL_KR":
        win = c[max(0, i - 5):i] + "|" + c[j:j + 5]
        wcase = ccase[max(0, i - 5):i] + "|" + ccase[j:j + 5]
        if not ("etf" in win or "레버" in win or any(a in win for a in ALIAS_KEYS)
                or re.search(r"(?<![A-Za-z])[A-Z]{2,5}(?![A-Za-z])", wcase)):
            return True
    if k == "10년뒤":
        if not (any(x in ctx_compact for x in FIRST_PERSON + HOLDING) or re.search(r"(?<![가-힣])전\s", text)):
            return True
    if k == "삼성" and any(c[j:].startswith(x) for x in SAMSUNG_NOT):
        return True
    return False


def _scan(c: str) -> list[tuple[int, int, Entry]]:
    out = []
    n = len(c)
    for i in range(n):
        node = TRIE.get(c[i])
        if node is None:
            continue
        j = i + 1
        while True:
            if _END in node:
                for e in node[_END]:
                    out.append((i, j, e))
            if j >= n:
                break
            node = node.get(c[j])
            if node is None:
                break
            j += 1
    return out


def analyze(title: str | None, body: str | None, utype: str) -> UnitResult:
    """한 단위(글 = 제목+본문 첫 줄, 댓글 = 본문)의 사전 적중. 위치는 text = title + '\\n' + body(글) / body(댓글) 기준."""
    res = UnitResult()
    title = title or ""
    body = body or ""
    if utype == "post":
        if is_news(title, body):
            res.excluded = "news"
            return res
        text = title + "\n" + body
        tlen = len(title)
    else:
        if any(d in body for d in DELETED):
            res.excluded = "deleted"
            return res
        text = body
        tlen = 0
    full_c = _compact(text)
    if sum(full_c.count(p) for p in POLITICS) >= 2 and not any(m in full_c for m in MARKET):
        res.excluded = "politics"
        return res
    mt = blank_quotes(text)
    c, idx, ccase = _compact_map(mt)
    ctx = c
    cands = _scan(c)
    if not cands and "ㅋ" not in mt and "주식전문가" not in c:
        res.score = 0 if utype == "post" else None
        return res

    by_group: dict[str, list] = {}
    for i, j, e in cands:
        os_, oe = idx[i], idx[j - 1] + 1
        if not _boundary_ok(e, mt, os_, oe):
            continue
        by_group.setdefault(e.group, []).append((i, j, e, os_, oe))

    hits: list[Hit] = []
    sent_cache: dict[tuple[int, int], tuple[int, int]] = {}

    def sent(os_, oe):
        key = (os_, oe)
        if key not in sent_cache:
            sent_cache[key] = _sentence(mt, os_, oe)
        return sent_cache[key]

    score_val = 0
    for group, lst in by_group.items():
        lst.sort(key=lambda t: (-(t[1] - t[0]), t[0]))
        taken = [False] * len(c)
        winners = []
        for i, j, e, os_, oe in lst:
            if any(taken[i:j]):
                continue
            for p in range(i, j):
                taken[p] = True
            winners.append((i, j, e, os_, oe))
        winners.sort(key=lambda t: t[0])
        if group == "tilt":
            sides = {w[2].stratum for w in winners
                     if not _context_excluded(w[2], c, ccase, w[0], w[1], mt, w[3], w[4], ctx)}
        for i, j, e, os_, oe in winners:
            if _context_excluded(e, c, ccase, i, j, mt, os_, oe, ctx):
                continue
            st = e.stratum
            in_title = os_ < tlen
            sn = sent(os_, oe)
            origin = ""
            if group == "emo":
                if st == "CAP":
                    sc = _compact(mt[sn[0]:sn[1]])
                    if any(p in sc for p in CAP_PROFIT):
                        st, origin = "E", "CAP 이익 실현 문장→E"
                if st in ("E", "F"):
                    if not e.inherent_neg and _negated(mt, os_, oe, sn, 8, 4):
                        res.neg_dropped += 1
                        continue
                    if _is_question(mt, sn):
                        if st == "E":
                            res.eq += 1
                        else:
                            res.fq += 1
                        continue
                elif st == "CAP":
                    if not e.inherent_neg and _negated(mt, os_, oe, sn, 8, 4):
                        st, origin = "RES", "CAP 부정→RES"
                    elif _is_question(mt, sn):
                        st, origin = "HES", "CAP 질문형→HES"
                elif st in ("LL_KR", "LL_US", "LS"):
                    if not e.inherent_neg and _negated(mt, os_, oe, sn, 8, 4):
                        res.lev_refuse += 1
                        continue
                elif st == "NOV":
                    if not e.inherent_neg and _negated(mt, os_, oe, sn, 8, 4):
                        res.neg_dropped += 1
                        continue
                    if e.key in NOV_QONLY and not _is_question(mt, sn):
                        continue
                elif st == "BUY":
                    if not e.inherent_neg and _negated(mt, os_, oe, sn, 8, 4):
                        res.neg_dropped += 1
                        continue
                if st in SENTIMENT_STRATA and _price_sentence(mt, sn):
                    res.price_dropped += 1
                    continue
                hits.append(Hit(st, STRATUM_BUNDLE[st], e.word, os_, oe, in_title, origin))
            elif group == "score":
                if _price_sentence(mt, sn):
                    res.price_dropped += 1
                    continue
                sign = 1 if st == "BULL" else -1
                if not e.inherent_neg and _negated(mt, os_, oe, sn, 3, None):
                    sign = -sign
                    reading = ("약세" if st == "BULL" else "강세") + "(부정이라 뒤집음)"
                else:
                    reading = "강세" if st == "BULL" else "약세"
                hits.append(Hit(st, e.bundle, e.word, os_, oe, in_title, "", reading, sign))
                score_val += sign * (2 if in_title else 1)
            elif group == "tilt":
                if len(sides) > 1:
                    continue        # 두 쪽 단어가 다 나온 글은 뺀다(H46)
                hits.append(Hit(st, e.bundle, e.word, os_, oe, in_title))
            else:
                hits.append(Hit(st, e.bundle, e.word, os_, oe, in_title))

    # MOCK 특수 규칙(약세론 조롱): '주식전문가'(단독), 'ㅋ4개 이상 + 2인칭'
    if EXPERT_ONLY_RX.sub("", mt).lower() == "주식전문가":
        p = mt.find("주식전문가")
        hits.append(Hit("MOCK_BEAR", "MOCK", "주식전문가(단독)", p, p + 5, p < tlen))
    if KK4_RX.search(mt):
        m = SECOND_PERSON_RX.search(mt)
        if m:
            hits.append(Hit("MOCK_BEAR", "MOCK", "ㅋ4개 이상+2인칭", m.start(), m.end(), m.start() < tlen,
                            reading=f"2인칭 '{m.group(0)}'"))

    res.hits = hits
    if utype == "post":
        res.score = max(-3, min(3, score_val))
    # M11 반어 의심: 환호 단어와 공포 단어가 한 문장에, 또는 환호 + ㅋ3개 이상 + 하락 표현(빠졌·떨어)
    e_h = [h for h in hits if h.stratum == "E"]
    f_h = [h for h in hits if h.stratum == "F"]
    if e_h:
        es = {sent(h.start, h.end) for h in e_h}
        if any(sent(h.start, h.end) in es for h in f_h):
            res.m11 = True
        elif re.search(r"ㅋ{3,}", mt) and ("빠졌" in c or "떨어" in c):
            res.m11 = True
    return res
