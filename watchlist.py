import re


def ai_tour_sectors(tour_map, symbol_key):
    """AI 탐방의 사업 역할을 유니버스 심볼로 연결한다(관찰용, 중복 소속 허용).

    대상 종목은 AI_UNIVERSE, 사업 역할은 탐방 지도에서 읽어 별도 명단을 복제하지 않는다.
    symbol_key는 거래소 접미사를 지도 키로 바꾸는 함수다. 별칭은 지도 회사 정보로 해석한다.
    ETF 분류와 ◆ 선정에 사용하는 대표 소분류에는 이 중복 소속을 합치지 않는다.
    """
    aliases = {a: t for t, c in tour_map["companies"].items() for a in c.get("aliases", [])}
    lookup = {}
    for subs in AI_UNIVERSE.values():
        for symbols in subs.values():
            for sym in symbols:
                key = symbol_key(sym)
                key = aliases.get(key, key)
                if key in lookup and lookup[key] != sym:
                    raise ValueError(f"탐방 회사 키 중복: {key}")
                lookup[key] = sym
    rows = []
    for stop in tour_map.get("stops", []):
        for zone in stop.get("zones", []):
            if zone.get("special"):
                continue
            symbols = list(dict.fromkeys(lookup[m["t"]] for m in zone["members"] if m["t"] in lookup))
            if symbols:
                rows.append({"key": "tour:" + zone["id"], "group": stop["name"],
                             "sub": zone["name"], "symbols": symbols})
    return rows

PROFIT_SECTORS = {
    "빅테크":        ["AAPL", "MSFT", "AMZN", "META", "GOOGL"],
    "AI 반도체":     ["NVDA", "AVGO", "AMD", "MRVL", "CRDO", "ALAB"],
    "메모리":        ["MU", "SNDK", "WDC", "STX"],
    "네트워크/광통신": ["ANET", "CIEN", "LITE", "GLW"],
    "반도체 장비":   ["LRCX", "TER"],
    "엔터프라이즈 SW": ["ORCL"],
}

LOSS_SECTORS = {
    "AI 인프라":      ["CRWV", "NBIS", "IREN", "CIFR", "APLD"],
    "양자컴퓨팅":     ["IONQ", "RGTI"],
    "eVTOL":         ["JOBY"],
    "헬스케어 AI":    ["TEM"],
    "원자력/에너지":  ["UUUU", "SMR", "OKLO"],
    "반도체 장비(적자)": ["AEHR"],
    "반도체(적자)":   ["INTC", "NVTS"],
}

# 두 펀드에는 들어오지만 옵션·실적 워치리스트에는 포함되지 않는 종목의
# 분류도 이 파일에서만 관리한다. PROFIT_TICKERS/LOSS_TICKERS에는 섞지 않는다.
ETF_SECTORS = {
    "빅테크": ["GOOG"],
    "AI 반도체": [
        "ARM", "QCOM", "TSM", "TSEM", "GFS", "ON", "SIMO", "TXN",
        "ADI", "NXPI", "MCHP", "MPWR", "STM",
    ],
    "반도체 장비": ["ASML", "AMAT", "KLAC", "FORM"],
    "엔터프라이즈 SW": [
        "NOW", "CRM", "SNOW", "DDOG", "PANW", "PLTR", "CRWD", "IBM",
        "ACN", "ROP", "SAP", "FTNT",
        "MDB", "DT", "OKTA", "GTLB",
    ],
    "AI 인프라": ["NTAP", "DELL", "VRT"],
    "네트워크/광통신": [
        "COHR", "AAOI", "CSCO", "VIAV", "IPGP",
        "5803", "300502", "300308",
    ],
    # CDN/엣지는 광통신 하드웨어가 아니라 인터넷 인프라 SW다. 광모듈 수급
    # 신호를 흐리지 않도록 `네트워크/광통신`과 분리해 별도 섹터로 둔다.
    "인터넷 인프라/CDN": ["NET", "AKAM"],
    # 화합물 반도체 기판(InP/GaAs)은 설계도 장비도 아니라 소재다.
    "반도체 소재": ["AXTI"],
    "원자력/에너지": [
        "BE", "GEV", "034020", "UEC", "FSLR",
        "010120", "267260", "298040",
    ],
    "핀테크/디지털자산": ["COIN", "HOOD", "MSTR", "CRCL", "IBKR"],
    "이커머스/플랫폼": ["EBAY", "MELI"],
    "우주/방산": ["RKLB", "SPCX", "MOG/A", "ASTS", "SATS"],
    "전기차/자율주행": ["TSLA"],
    "헬스케어": [
        "LLY", "BIIB", "NTRA", "ILMN", "GILD", "JAZZ", "ONC", "LBRX", "ORKA",
        "VRTX",
    ],
    "메모리": ["000660", "005930", "285A", "440110", "SKHY", "SKHYV"],
    "전자부품/AI 하드웨어": ["009150", "006400", "6981", "FLEX", "VICR"],
    "중국 AI/플랫폼": ["9988", "700", "0043Y0", "2513", "PDD", "BABA", "100"],
    "산업재": ["TKR", "CRS"],
    "필수소비재": ["KO"],
    "미디어/게임": ["TTWO"],
}

# 만기마다 코드가 바뀌는 파생상품은 고정 티커로 관리할 수 없다. 정규식으로 잡지
# 않으면 롤오버 때마다 조용히 `미분류`로 떨어져 익스포저가 섹터 해석에서 사라진다.
# 예: NQM6(2026-06) → NQU6(2026-09) → NQZ6(2026-12) → NQH7(2027-03)
# 월물 코드: H=3월, M=6월, U=9월, Z=12월
SECTOR_PATTERNS = [
    (re.compile(r"^NQ[HMUZ]\d$"), "지수선물"),
]


def _merge_sector_maps(*sector_maps):
    merged = {}
    for sector_map in sector_maps:
        for sector, tickers in sector_map.items():
            merged.setdefault(sector, []).extend(tickers)
    return merged


SECTOR_MAP = _merge_sector_maps(PROFIT_SECTORS, LOSS_SECTORS, ETF_SECTORS)

TICKER_TO_SECTOR = {t: s for s, tickers in SECTOR_MAP.items() for t in tickers}


def resolve_sector(listing_key, default="미분류"):
    """상장 코드를 섹터로 해석한다. 고정 매핑 우선, 없으면 패턴 규칙."""
    if not listing_key:
        return default
    sector = TICKER_TO_SECTOR.get(listing_key)
    if sector:
        return sector
    for pattern, matched_sector in SECTOR_PATTERNS:
        if pattern.match(listing_key):
            return matched_sector
    return default

PROFIT_TICKERS = [t for tickers in PROFIT_SECTORS.values() for t in tickers]
LOSS_TICKERS   = [t for tickers in LOSS_SECTORS.values()  for t in tickers]

# AI 밸류체인 유니버스. 전고점 거리 분류(`ai_ath_gap.py`) 전용이라 옵션·실적 스캔과
# ETF 섹터 해석(SECTOR_MAP)에는 합치지 않는다. 종목은 yfinance 심볼로 적는다
# (KRX .KS/.KQ, 도쿄 .T, 대만 .TW/.TWO, 선전 .SZ, 홍콩 .HK, 빈 .VI, 독일 .DE, 파리 .PA).
# 여러 층에 걸친 회사는 대표 층 하나에만 둔다. 같은 회사의 다른 상장(GOOG,
# SKHY ADR, 9988)은 이력이 긴 쪽만 둔다.
# 국내 반도체 소부장은 네이버 업종 '반도체와반도체장비' 중 시총 5,000억 원 이상
# (2026-10-02 종가)을 모두 넣었다. 우선주·지주사·주력이 바이오인 곳은 뺐다. 다른
# 업종에 있지만 주력이 반도체 소재·계측인 한솔케미칼·레이크머티리얼즈·원익머트리얼즈·
# 파크시스템스도 넣었다.
# AI 탐방(desk/tour_map.json)의 물리 공급망 구역(칩·소재·부품·장비·조립·스토리지·
# 데이터센터·전력)에만 있던 상장사도 넣었다(2026-10-05, 27종목). 우라늄·규제 전력회사·
# 태양광·ESS·배터리는 AI 밖 수요가 주가를 정해서, 소프트뱅크그룹은 투자 지주사라서,
# 삼성물산·삼성중공업은 AI 사업이 아직 의향서 단계라서 뺐다. 탐방의 앱·서비스·연구소
# 구역은 넣지 않았다.
AI_UNIVERSE = {
    "칩 설계": {
        "GPU·AI 가속기": ["NVDA", "AMD", "CBRS"],
        "맞춤형 ASIC·디자인하우스": [
            "AVGO", "MRVL", "2454.TW", "3661.TW", "200710.KQ", "399720.KQ", "490470.KQ",
        ],
        "CPU·모바일·IP": ["INTC", "ARM", "QCOM", "SWKS"],
        "인터커넥트 칩": ["CRDO", "ALAB", "RMBS", "MTSI"],
        "아날로그·전력 반도체": [
            "TXN", "ADI", "NXPI", "MCHP", "ON", "MPWR", "STM", "NVTS", "AOSL", "094170.KQ",
        ],
    },
    "파운드리·후공정": {
        "파운드리": ["TSM", "UMC", "GFS", "TSEM", "000990.KS"],
        "OSAT(후공정 외주)": ["ASX", "AMKR", "036540.KQ", "067310.KQ", "131970.KQ", "033640.KQ"],
    },
    "메모리·스토리지": {
        # 삼성전자·SK하이닉스·MU는 NAND도 겸한다. 제주반도체는 팹리스 저전력 메모리,
        # 한양디지텍은 메모리 모듈이다.
        "DRAM·HBM": [
            "005930.KS", "000660.KS", "MU", "2408.TW", "2344.TW", "080220.KQ", "078350.KQ",
        ],
        "NAND·SSD": ["SNDK", "285A.T"],
        "SSD 컨트롤러": ["SIMO", "8299.TWO", "440110.KQ"],
        "HDD": ["STX", "WDC"],
        # TDK는 MLCC도 겸한다.
        "HDD 부품": ["6594.T", "6762.T", "7741.T", "4004.T"],
    },
    "장비·소재·테스트": {
        "전공정 장비(글로벌)": ["AMAT", "LRCX", "KLAC", "ASML", "8035.T"],
        "전공정 장비(국내)": [
            "084370.KQ", "240810.KQ", "036930.KQ", "095610.KQ",
            "403870.KQ", "319660.KQ", "281820.KS", "089970.KQ",
        ],
        # 식각 링·실리콘 부품·석영·세라믹 히터·세정 코팅·RF 전원처럼 장비 안에 들어가는 것.
        "장비 부품·소모품": [
            "064760.KQ", "166090.KQ", "183300.KQ", "388210.KQ",
            "074600.KQ", "059090.KQ", "101160.KQ", "144960.KQ",
        ],
        "팹 설비": ["083450.KQ", "045100.KQ"],
        "HBM·후공정 장비": [
            "042700.KS", "039030.KQ", "031980.KQ", "053610.KQ", "122640.KQ", "089890.KQ",
        ],
        "테스트 장비": [
            "TER", "6857.T", "KEYS", "AEHR", "089030.KQ", "232140.KQ", "086390.KQ", "003160.KS", "092870.KQ",
            "253590.KQ",
        ],
        "검사·계측 장비": ["420770.KQ", "064290.KQ", "140860.KQ"],
        "테스트 소켓·프로브": [
            "FORM", "095340.KQ", "058470.KQ", "252990.KQ",
            "131290.KQ", "425420.KQ", "080580.KQ", "098120.KQ", "219130.KQ",
        ],
        "소재": [
            "ENTG", "357780.KQ", "AXTI", "005290.KQ", "014680.KS", "101490.KQ",
            "102710.KQ", "077360.KQ", "036810.KQ", "281740.KQ", "104830.KQ",
        ],
    },
    "기판·부품": {
        "ABF 필름": ["2802.T"],
        "패키지 기판": [
            # 닛토보는 기판용 저열팽창 유리섬유 천(T-glass)을 만든다.
            "4062.T", "3037.TW", "8046.TW", "3189.TW", "ATS.VI", "3110.T",
            "353200.KS", "007810.KS", "011070.KS", "222800.KQ", "195870.KS",
        ],
        "AI 서버 PCB": ["TTMI", "007660.KS", "356860.KQ"],
        # 삼성전기는 FC-BGA 기판도 겸한다.
        "MLCC": [
            "009150.KS", "6981.T", "6976.T", "2327.TW", "2492.TW",
            "001820.KS", "052710.KQ", "149950.KQ",
        ],
        "MLCC 소재·장비": ["005070.KS", "078600.KQ", "320000.KQ", "306620.KQ"],
        "커넥터": ["APH"],
    },
    "네트워크·광통신": {
        "스위치·네트워크 장비": ["ANET", "CSCO"],
        "광 전송·광모듈": [
            "CIEN", "LITE", "COHR", "AAOI", "FN", "300308.SZ", "300502.SZ", "VIAV", "IPGP",
        ],
        "광섬유·케이블": ["GLW", "5803.T"],
    },
    "서버·시스템": {
        # P(에버퓨어)는 구 퓨어스토리지(PSTG)다.
        "서버·스토리지": ["DELL", "HPE", "SMCI", "NTAP", "P"],
        # 폭스콘·콴타는 엔비디아 랙을 조립하는 대만 ODM, 델타는 서버 전원·냉각이다.
        "EMS·전원 모듈": ["CLS", "FLEX", "2317.TW", "2382.TW", "VICR", "2308.TW"],
    },
    "전력·에너지": {
        # 히타치는 히타치에너지(변압기·송배전) 몫으로 넣었다.
        "전력·냉각 장비": [
            "VRT", "ETN", "SU.PA", "NVT", "MOD", "6501.T", "298040.KS", "010120.KS", "267260.KS",
        ],
        # 캐터필러·커민스는 데이터센터용 가스 발전기·터빈 몫이다.
        "발전 설비": ["GEV", "ENR.DE", "7011.T", "CAT", "CMI", "034020.KS"],
        "발전·연료전지": ["BE", "CEG", "VST", "TLN"],
        "SMR": ["OKLO", "SMR", "XE"],
    },
    "데이터센터": {
        "네오클라우드": ["CRWV", "NBIS"],
        "채굴→AI 전환": ["IREN", "CIFR", "APLD", "WULF", "HUT", "GLXY"],
        "설계·시공": ["FIX", "EME"],
    },
    "빅테크·AI 모델": {
        "미국 빅테크": ["MSFT", "GOOGL", "AMZN", "META", "ORCL", "AAPL"],
        "중국 AI": ["BABA", "2513.HK", "0100.HK"],
    },
    "소프트웨어·보안": {
        "데이터·인프라 SW": ["SNOW", "DDOG", "MDB", "DT", "GTLB"],
        "엣지·CDN": ["NET", "AKAM", "FSLY"],
        "엔터프라이즈 앱": ["PLTR", "NOW", "CRM", "SAP", "ROP"],
        "IT 서비스": ["IBM", "ACN"],
        "보안": ["PANW", "CRWD", "FTNT", "OKTA"],
    },
    "AI 응용": {
        "피지컬·헬스케어 AI": ["TSLA", "TEM"],
    },
}
