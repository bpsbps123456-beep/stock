"""Turn unclassified.json + hand-written classifications into proposals.json / proposals.md."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from desk.config import SEMIS_SECTORS  # noqa: E402
from watchlist import SECTOR_MAP, resolve_subsector  # noqa: E402

HERE = Path(__file__).resolve().parent

# Closed-end fund (private-tech holdings); not an operating company -> excluded, no sector.
EXCLUDE = {"DXYZ": "Destiny Tech100: 비상장 기술주(SpaceX·OpenAI 등)를 담는 NYSE 상장 폐쇄형 펀드. "
                   "exclusion_reason의 이름 규칙(ETF|FUND…)에 안 걸려 통과했다. 펀드라 분류 대상 아님."}

# ticker: (sector, confidence, rationale)
C = {
    # --- AI 인프라 (서버·스토리지·네오클라우드, DELL/NTAP/CRWV 관례)
    "SMCI": ("AI 인프라", "high", "AI·GPU 서버 랙 제조가 매출 대부분. DELL과 같은 서버·스토리지 층"),
    "P":    ("AI 인프라", "high", "구 Pure Storage(PSTG, 현 Everpure). 올플래시 스토리지 어레이·구독, NTAP 관례"),
    "CORZ": ("AI 인프라", "high", "비트코인 채굴에서 CoreWeave용 HPC 데이터센터 임대로 전환. IREN·CIFR·APLD 관례"),
    # --- AI 반도체 (반도체 예산)
    "AMKR": ("AI 반도체", "high", "반도체 후공정(패키징·테스트) 외주. 파운드리 TSM·GFS·TSEM과 같은 섹터"),
    "AMBA": ("AI 반도체", "high", "엣지 AI 영상처리 SoC 팹리스(차량·보안카메라)"),
    "MBLY": ("AI 반도체", "medium", "ADAS용 EyeQ SoC 칩 판매가 매출 대부분. 시장 분류(GICS)는 자동차부품이라 "
                                    "전기차/자율주행도 가능하지만, 차량 반도체 ON·NXPI·STM을 AI 반도체에 둔 관례를 따름"),
    # --- 원자력/에너지 (발전·원전연료·전력기기·저장·태양광·가스)
    "CEG":  ("원자력/에너지", "high", "미국 최대 원전 운영 발전사. 빅테크 원전 전력 계약"),
    "VST":  ("원자력/에너지", "high", "가스·원전 발전사(Energy Harbor 원전 인수)"),
    "CCJ":  ("원자력/에너지", "high", "우라늄 채굴·핵연료 서비스, Westinghouse 지분. UEC·UUUU 관례"),
    "LEU":  ("원자력/에너지", "high", "농축 우라늄(LEU·HALEU) 공급"),
    "BWXT": ("원자력/에너지", "high", "미 해군 원자로·핵연료·원전 부품. 매출 대부분 정부 원자력 사업"),
    "ETN":  ("원자력/에너지", "medium", "배전·개폐기·UPS 등 전력관리 장비. AI_UNIVERSE에서 같은 층인 LS일렉트릭·"
                                      "HD현대일렉트릭·효성중공업이 이 섹터(VRT는 냉각 중심이라 AI 인프라)"),
    "POWL": ("원자력/에너지", "medium", "맞춤형 개폐기·전력제어실(석유화학·전력·데이터센터). ETN과 같은 전력기기"),
    "GNRC": ("원자력/에너지", "medium", "가정·상업용 비상 발전기(데이터센터 백업 발전 포함). CAT·CMI 발전 설비 층과 유사"),
    "FLNC": ("원자력/에너지", "high", "전력망용 배터리 에너지저장장치(ESS)"),
    "EOSE": ("원자력/에너지", "high", "아연 기반 장주기 에너지저장 배터리"),
    "NXT":  ("원자력/에너지", "high", "태양광 추적장치(트래커). 2025년 Nextpower로 사명 변경. FSLR 관례"),
    "RUN":  ("원자력/에너지", "high", "주택용 태양광·저장 설치·리스"),
    "KMI":  ("원자력/에너지", "medium", "천연가스 파이프라인(미드스트림). 데이터센터 가스 발전 수요 테마"),
    "LNG":  ("원자력/에너지", "medium", "LNG 수출 터미널 운영"),
    "EQT":  ("원자력/에너지", "medium", "미국 최대 천연가스 생산사"),
    "BKR":  ("원자력/에너지", "medium", "유전 서비스 + 가스터빈·LNG 설비(IET). 데이터센터 가스터빈 수주"),
    # --- 엔터프라이즈 SW
    "AI":   ("엔터프라이즈 SW", "high", "기업용 AI 애플리케이션 SW 구독"),
    "ADBE": ("엔터프라이즈 SW", "high", "크리에이티브·문서 SW 구독(Creative Cloud·Acrobat)"),
    "PATH": ("엔터프라이즈 SW", "high", "업무 자동화(RPA)·에이전트 SW"),
    "WDAY": ("엔터프라이즈 SW", "high", "인사·재무 SaaS"),
    "TWLO": ("엔터프라이즈 SW", "high", "메시지·음성 통신 API 클라우드"),
    "CXM":  ("엔터프라이즈 SW", "high", "고객경험 관리 SaaS"),
    "SOUN": ("엔터프라이즈 SW", "medium", "음성 AI SW 라이선스·구독(차량·외식 주문). 적자 소형주"),
    "BBAI": ("엔터프라이즈 SW", "medium", "국방·정부용 AI 의사결정 분석 SW. 고객이 국방이라 우주/방산도 가능하나 PLTR 관례"),
    "INOD": ("엔터프라이즈 SW", "medium", "빅테크용 AI 학습 데이터·데이터 엔지니어링 서비스(IT 서비스, ACN 관례)"),
    "RELX": ("엔터프라이즈 SW", "medium", "법률·과학·리스크 정보 분석 구독(LexisNexis·Elsevier). 출판이라 미디어도 가능"),
    # --- 이커머스/플랫폼 (인터넷 플랫폼: 상거래 + 광고 기반)
    "SE":   ("이커머스/플랫폼", "high", "동남아 이커머스 Shopee 중심 + 게임·핀테크. MELI 관례"),
    "CPNG": ("이커머스/플랫폼", "high", "한국 이커머스(로켓배송)"),
    "RDDT": ("이커머스/플랫폼", "medium", "커뮤니티 플랫폼 광고 + AI 데이터 라이선스. 소셜 섹터가 없어 인터넷 플랫폼으로"),
    "PINS": ("이커머스/플랫폼", "medium", "이미지 발견·쇼핑 플랫폼 광고"),
    "APP":  ("이커머스/플랫폼", "medium", "모바일 게임·이커머스 광고 엔진(AXON). 2025년 앱 사업 매각 후 광고 플랫폼 100%. "
                                       "B2B SW로 보면 엔터프라이즈 SW도 가능"),
    "PERI": ("이커머스/플랫폼", "low", "검색·CTV·디스플레이 광고 기술(애드테크). APP와 같은 광고 플랫폼으로 묶음"),
    "NRDY": ("이커머스/플랫폼", "low", "온라인 과외 플랫폼(Varsity Tutors), 소비자·학교 구독"),
    # --- 미디어/게임
    "RBLX": ("미디어/게임", "high", "게임 플랫폼, 게임 내 화폐(Robux) 판매"),
    "BZFD": ("미디어/게임", "high", "디지털 뉴스·엔터테인먼트 미디어 광고"),
    "GETY": ("미디어/게임", "high", "이미지·영상 콘텐츠 라이선스"),
    "SSTK": ("미디어/게임", "high", "스톡 이미지·영상 라이선스(Getty와 합병 진행)"),
    # --- 헬스케어 / 헬스케어 AI
    "ISRG": ("헬스케어", "high", "수술 로봇(da Vinci)·소모품"),
    "SDGR": ("헬스케어 AI", "high", "계산화학 신약개발 SW + 자체 파이프라인, 적자. TEM 관례"),
    "RXRX": ("헬스케어 AI", "high", "AI 신약개발 바이오, 적자"),
    # --- 중국
    "BIDU": ("중국 AI/플랫폼", "high", "중국 검색 광고·AI 클라우드(ERNIE)·자율주행"),
    # --- 양자
    "QBTS": ("양자컴퓨팅", "high", "양자 어닐링 컴퓨터·클라우드 접근"),
    "QUBT": ("양자컴퓨팅", "high", "광자 양자컴퓨팅·박막 리튬니오베이트 칩"),
    # --- 우주/방산
    "PL":   ("우주/방산", "high", "지구관측 위성 영상 데이터 구독(정부·국방 고객 중심)"),
    # --- 전기차/자율주행
    "SERV": ("전기차/자율주행", "medium", "보도 자율주행 배달 로봇. 로봇 섹터가 없어 자율주행으로"),
    # --- 핀테크
    "LMND": ("핀테크/디지털자산", "high", "AI 기반 디지털 보험(인슈어테크)"),
    # --- 산업재 (핵심광물)
    "MP":   ("산업재", "medium", "희토류 채굴·정제·영구자석(미 국방부 지분). 핵심광물 섹터가 없어 특수합금 CRS가 있는 산업재로"),
    "USAR": ("산업재", "medium", "희토류 자석 공장·Round Top 광산 개발, 적자. MP와 같은 핵심광물 테마"),
    "CRML": ("산업재", "medium", "그린란드 Tanbreez 희토류 광산 개발, 적자. 핵심광물 테마"),
    "UAMY": ("산업재", "medium", "안티몬·제올라이트 채굴·제련. 핵심광물 테마"),
}

unc = json.loads((HERE / "unclassified.json").read_text(encoding="utf-8"))
known = set(SECTOR_MAP)
proposals, excluded = [], []
for r in unc:
    t = r["ticker"]
    held = f"{r['first']}~{r['last']} (주간 PDF {r['n_weekly']}, 2026 일간 {r['n_daily']})"
    if t in EXCLUDE:
        excluded.append({"ticker": t, "name": r["names"][-1], "held": held,
                         "max_weight": r["max_weight"], "reason": EXCLUDE[t]})
        continue
    sector, conf, why = C[t]
    assert sector in known, (t, sector)
    proposals.append({
        "ticker": t, "name": r["names"][-1], "sector": sector, "new_sector": False,
        "semis_budget": sector in SEMIS_SECTORS, "subsector": resolve_subsector(t),
        "confidence": conf, "rationale": why,
        "first": r["first"], "last": r["last"], "max_weight": r["max_weight"],
        "max_weight_date": r["max_weight_date"], "n_weekly_pdfs": r["n_weekly"],
        "n_daily_2026": r["n_daily"], "n_pdfs": r["n_weekly"] + r["n_daily"],
        "in_desk_2026": r["n_daily"] > 0, "held": held,
    })
missing = set(C) - {p["ticker"] for p in proposals}
assert not missing, missing
assert len(proposals) + len(excluded) == len(unc)

# KoAct US holdings still unclassified once these are added
ko = json.loads((HERE / "koact_unclassified_before.json").read_text(encoding="utf-8"))
covered = {p["ticker"] for p in proposals}
ko_after = sorted(v["ticker"] for v in ko if v["ticker"] not in covered)
ko_covered = sorted(v["ticker"] for v in ko if v["ticker"] in covered)

(HERE / "proposals.json").write_text(json.dumps(proposals, ensure_ascii=False, indent=1), encoding="utf-8")
(HERE / "excluded.json").write_text(json.dumps(excluded, ensure_ascii=False, indent=1), encoding="utf-8")
(HERE / "koact_unclassified_after.json").write_text(
    json.dumps({"count": len(ko_after), "tickers": ko_after, "covered_by_time_proposals": ko_covered},
               ensure_ascii=False, indent=1), encoding="utf-8")

# ---- markdown
by_sector = {}
for p in proposals:
    by_sector.setdefault(p["sector"], []).append(p["ticker"])
lines = [
    "# TIME 미국 보유 미분류 종목 분류 제안 (2026-10-10)",
    "",
    "입력: 2023-05-22~2025-12-29 주간 PDF 137개(ISIN 3종 SMCI·ANET·BZFD 복원) + 2026 일간 스냅샷 187개. "
    "현금·선물·ETF/펀드 제외 후 미국 상장 137개사 중 `desk.strategies.sector_of`가 `미분류`인 57개.",
    "",
    f"- 분류 제안 {len(proposals)}개, 모두 기존 SECTOR_MAP 섹터(새 섹터 0개). 반도체 예산 대상 "
    f"{sum(p['semis_budget'] for p in proposals)}개: "
    + ", ".join(p["ticker"] for p in proposals if p["semis_budget"]) + ".",
    f"- 제외 {len(excluded)}개: " + "; ".join(f"{e['ticker']} — {e['reason']}" for e in excluded),
    f"- 2026 일간 스냅샷에 들어 있어 야간 데스크 판단(2026-01-02~)에 영향 주는 이름 "
    f"{sum(p['in_desk_2026'] for p in proposals)}개: "
    + ", ".join(p["ticker"] for p in proposals if p["in_desk_2026"]) + ". 나머지는 2023~2025 주간 PDF에만 있음.",
    "- 웹 검색 0회(모두 잘 알려진 회사라 자체 지식으로 분류).",
    "",
    "## 섹터별 묶음",
    "",
    "| 섹터 | 반도체 예산 | 종목 |",
    "|---|---|---|",
]
for s, ts in sorted(by_sector.items(), key=lambda kv: -len(kv[1])):
    lines.append(f"| {s} | {'예' if s in SEMIS_SECTORS else '아니오'} | {', '.join(ts)} |")
lines += [
    "",
    "## 종목별 (최대 NAV 비중 순)",
    "",
    "| 티커 | 회사 | 제안 섹터 | 반도체 예산 | 확신 | AI_UNIVERSE 소분류 | 보유 기간 | 최대 비중 | PDF 수(주간+일간) | 근거 |",
    "|---|---|---|---|---|---|---|---:|---:|---|",
]
for p in proposals:
    lines.append(
        f"| {p['ticker']}{' ★' if p['in_desk_2026'] else ''} | {p['name']} | {p['sector']} | "
        f"{'예' if p['semis_budget'] else '아니오'} | {p['confidence']} | {p['subsector'] or '-'} | "
        f"{p['first']}~{p['last']} | {p['max_weight']:.2f}% | {p['n_weekly_pdfs']}+{p['n_daily_2026']} | {p['rationale']} |")
lines += [
    "",
    "★ = 2026 일간 스냅샷에도 있음(데스크 엔진이 읽는 구간).",
    "",
    "## 판단이 갈릴 수 있는 곳",
    "",
    "- **MBLY**: 매출은 EyeQ 칩이지만 시장 분류는 자동차부품. 차량 반도체 ON·NXPI·STM이 이미 `AI 반도체`라 같은 관례로 반도체 예산에 넣었다. 2026 일간에는 없어 데스크 영향 없음.",
    "- **ETN·POWL·GNRC**: 전력기기. AI_UNIVERSE에서 같은 층인 국내 변압기·개폐기 3사가 `원자력/에너지`라 그쪽으로. VRT(냉각 중심)는 `AI 인프라` 그대로.",
    "- **KMI·LNG·EQT·BKR**: 석유·가스 섹터가 없어 `원자력/에너지`. 원자력·전력 기술주와 원자재 가격 민감도가 다르다는 점은 남는다(KoAct의 XOM도 같은 문제).",
    "- **RDDT·PINS·APP·PERI·NRDY**: 소셜·광고 섹터가 없어 `이커머스/플랫폼`을 '인터넷 플랫폼(상거래+광고)'으로 넓혀 썼다. APP는 `엔터프라이즈 SW`도 가능.",
    "- **MP·USAR·CRML·UAMY**: 핵심광물 섹터가 없어 특수합금 CRS가 있는 `산업재`. 네 종목끼리 상관이 높아, 묶음 상한을 따로 원하면 새 섹터 `핵심광물`(반도체 예산 아님)을 만들 수 있다.",
    "- **DXYZ**: 폐쇄형 펀드인데 `exclusion_reason` 이름 규칙에 안 걸린다. 2024-04-08 주간 PDF 1회뿐이라 데스크 영향 없음.",
    "",
    f"## KoAct 미국 보유 중 이 제안 후에도 미분류: {len(ko_after)}개",
    "",
    ", ".join(ko_after),
    "",
    f"(KoAct 미분류 {len(ko)}개 중 이 제안으로 함께 해소: {', '.join(ko_covered)})",
    "",
]
(HERE / "proposals.md").write_text("\n".join(lines), encoding="utf-8")
print(len(proposals), "proposals;", len(excluded), "excluded; semis:",
      [p["ticker"] for p in proposals if p["semis_budget"]])
print("desk-2026:", [p["ticker"] for p in proposals if p["in_desk_2026"]])
print("KoAct after:", len(ko_after), ko_after)
print("KoAct covered:", ko_covered)
