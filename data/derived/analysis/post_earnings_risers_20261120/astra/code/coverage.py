"""Validate collected data, then build the handoff for stage two."""
from common import *
import json

def day(x):
    return str(x)[:10]

def missing(tickers):
    return ', '.join(sorted(set(TICKERS)-set(tickers))) or '없음'

def main():
    m=pd.read_csv(DATA/'wiki_map.csv')
    v=pd.read_parquet(DATA/'wiki_views.parquet')
    s=pd.read_parquet(DATA/'short_interest.parquet')
    ins=pd.read_parquet(DATA/'insider.parquet')
    assert len(TICKERS)==164 and set(m.ticker)==set(TICKERS) and m.ticker.is_unique
    assert set(v.ticker) <= set(m.dropna(subset=['article']).ticker)
    assert not v.duplicated(['date','ticker']).any() and v.views.ge(0).all()
    assert not s.duplicated(['settlement_date','ticker']).any() and (s.publish_date>s.settlement_date).all()
    assert s[['short_qty','avg_daily_vol','days_to_cover']].notna().all().all()
    assert set(ins.code)=={'P','S'} and not ins.duplicated(['ticker','accession','transaction_id']).any()
    assert ins.filing_date.between(START,END).all()
    assert (ins.trade_date.dropna()<=END).all()
    assert (ins.value.fillna(0)-ins.price.mul(ins.shares).fillna(0)).abs().max()==0
    assert all(p.stat().st_size<10_000_000 for p in DATA.iterdir())
    n=m.article.notna().sum()
    errors=json.loads((CACHE/'sec-errors.json').read_bytes())
    moves=json.loads((CACHE/'verified-moves.json').read_bytes())
    transitions=[f'{stamp[:10]} {old} → {new}' for items in moves.values() for stamp,old,new in items]
    views_missing=set(m.dropna(subset=['article']).ticker)-set(v.ticker)
    sec_empty=missing(ins.ticker)
    lines=[
        '# 1단계 자료 인계', '',
        '대상은 두 펀드의 미국 주식 161개와 기준 ETF QQQ·SPY·SOXX, 총 164개다. 가설 18개(예측 16, 사후 설명 2)를 먼저 고정했다. 이 단계에서 사건·결과 수익률 파일은 읽지 않았다. 날짜는 미국 거래·공시일 기준이고 위키는 UTC 일자다.', '',
        '|산출물|관측 종목 / 164|행|실제 날짜|빈 종목|','|---|---:|---:|---|---|',
        f'|wiki_map.csv|{n}개 영어 문서 매핑|{len(m):,}|수집일 2026-11-20|{missing(m.dropna(subset=["article"]).ticker)}|',
        f'|wiki_views.parquet|{v.ticker.nunique()}|{len(v):,}|{day(v.date.min())} ~ {day(v.date.max())}|{missing(v.ticker)}|',
        f'|short_interest.parquet|{s.ticker.nunique()}|{len(s):,}|결제 {s.settlement_date.min()} ~ {s.settlement_date.max()}, 공표 {s.publish_date.min()} ~ {s.publish_date.max()}|{missing(s.ticker)}|',
        f'|insider.parquet|{ins.ticker.nunique()}|{len(ins):,}|접수 {ins.filing_date.min()} ~ {ins.filing_date.max()}, 거래 {ins.trade_date.min()} ~ {ins.trade_date.max()}|{sec_empty}|','',
        '## 자료 해석과 미수집 범위','',
        f'- **위키.** 수동 후보를 영어 위키 API의 문서 존재 여부·넘겨주기·동음이의어·설명으로 확인했다. {n}개 심벌은 고유 문서 {m.article.nunique()}개를 공유한다. 미매핑은 후보 제목 및 검색으로 전용 문서를 확인하지 못한 경우다. 기업명이 아닌 학술지로 연결되는 UUUU를 교정했다. PDD(Pinduoduo)와 RDDT(Reddit)는 사업·서비스 관심이 섞여 확신을 낮게 뒀다. Wikidata SPARQL 엔드포인트 초기 연결은 읽기 시간 초과였다. [MediaWiki API](https://en.wikipedia.org/w/api.php)',
        f'- **조회수.** 영어 문서의 all-access/user 일별 요청이다. 없는 날짜에 0을 넣지 않았다. 제목은 수집 시점의 정규 문서명이므로 문서 생성·이전 전과 이전 직후는 H01/H07에서 제외하거나 제목 이력을 별도로 확인한다(예: P·NXT·CBRS·FORM). 브랜드·서비스명과 합산하지 않았다. 445일 전체를 관측한 심벌은 {v.groupby("ticker").size().eq(445).sum()}개다. 일부 요청은 HTTP 429로 거부되어 간격을 늘려 재시도했다. 매핑됐으나 관측값이 없는 종목은 {", ".join(sorted(views_missing)) or "없음"}. [Wikimedia Pageviews](https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/reference/page-views.html)',
        f'- **잔고.** FINRA 파일 {s.settlement_date.nunique()}회. 공표 시차는 결제 후 7영업일 안팎이다. 행의 publish_date가 사건일보다 빠를 때만 쓴다. 11월 13일 결제분은 11월 24일 예정으로 제외했다. 발표일 당일 장중 이용하지 않는다. days_to_cover는 FINRA 제공값으로 1일 하한과 거래량 0의 특수값에 유의한다. 원문 분할·정정 표시를 보존했다. 이력은 나중 정정될 수 있다. PSTG→P와 MOGA→MOG-A 표기를 연결했고, SATS의 Strive 우선주·SPCX의 옛 SPAC ETF는 발행사명으로 제거했다. ECHO/SATS는 각 시점 실제 심벌로 구분했다. [FINRA 공표 일정](https://www.finra.org/filing-reporting/regulatory-filing-systems/short-interest) · [원본 파일 예시](https://cdn.finra.org/equity/otcmarket/biweekly/shrt20261030.csv)',
        f'- **내부자.** 2025Q3~2026Q3 분기 압축 자료 뒤를 2026-09-30~11-19 발행사 submissions와 비파생 Form 4 원문 XML로 이었다. 분기 경계 중복은 원 접수번호로 제거했다. SEC 기준 UA와 요청 간 0.3초 이상 간격을 적용했다. 최근 기간 소스 오류 {len(errors)}건. 접수일 다음 날부터 사용하며 거래일로 소급하지 않는다. P/S는 공개시장 또는 사적 거래를 함께 포함해 장내 거래만 식별할 수 없다. 4/A 정정·파생·증여·보상은 제외했다. 주식 수 × 신고 주당가격이 value이며 원신고 단위다. 별도 통화 변환을 하지 않았다. 잔여 빈 종목은 관측된 P/S가 없음을 뜻한다. ETF나 해외 발행사의 신고 의무와 적용 기간이 다르다. [SEC 분기 자료](https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets) · [Form 4 코드 정의](https://www.sec.gov/about/forms/form4.pdf)',
        '- 내부자 자료를 회사 단위로 합칠 때는 accession과 transaction_id로 원거래를 하나만 센다. Alphabet 두 주식 종류는 같은 발행회사 내역을 공유한다. owner_ciks는 공동 신고자도 보존한다. 주식 종류는 security_title로 확인하고 동일 거래의 별도 신고 가능성을 점검한다. 90일 창은 창 전체에 자료가 있는 사건에만 쓴다.',
        '- **선택 자료.** 미국 Google Trends 기업 주제의 기간 지정 explore 요청이 HTTP 429를 반환해 시계열이 없다. 기타 선택 자료(게시판 등)는 사용하지 않았다.',
        '- **제목 이력.** 수동 후보와 현재 제목이 다른 문서 쌍 및 FormFactor의 위키 이동 로그를 확인했다. 확인된 변경은 전날까지 옛 제목, 다음 날부터 새 제목을 사용하고 이동 당일은 제외했다. 확인된 날짜와 제목은 '+(', '.join(transitions) or '없음')+'이다. 확인 범위 밖의 과거 명칭과 일반 넘겨주기 조회수는 포함하지 않았다.',
        '- **재현.** code 폴더에서 py -B fetch_archives.py sec, py -B fetch_archives.py short, py -B insider.py, py -B wiki.py, py -B wiki_moves.py logs, py -B wiki_moves.py apply, py -B coverage.py 순으로 실행한다. 자료는 10 MB 미만이다. 저장소의 data/**/*.parquet 제외 규칙이 세 parquet 결과에도 적용된다. 캐시에는 원본과 검증 로그가 있다.',
        '', '## 사전 선택한 세 가설', '',
        '- H01은 회사별 평상시 대비 조회 변화로 규모 차이를 줄인다.',
        '- H02는 한국 투자자의 실결제 상위표로 관심 쏠림을 측정한다. 투자자 유형은 구분되지 않는다. 담당자가 수집하는 SEIBRO를 사용한다.',
        '- H08은 발표 전 공개된 공매도 잔고의 청산 부담을 측정한다. 사전 약세 포지션을 직접 관찰할 수 있고 개인 투자자만의 지표로 해석하지 않는다.',
        '', '## 종목별 관측', '',
        '각 칸은 관측수와 첫날~끝날이다. 날짜는 각각 UTC 조회일, 결제일, 신고 접수일이다.', '',
        '|심벌|위키|잔고|Form 4 P/S|','|---|---|---|---|'
    ]
    def periods(frame,column):
        return {k:f'{len(g):,} ({day(g[column].min())}~{day(g[column].max())})' for k,g in frame.groupby('ticker')}
    counts=[periods(v,'date'),periods(s,'settlement_date'),periods(ins,'filing_date')]
    lines.extend('|'+k+'|'+'|'.join(x.get(k,'—') for x in counts)+'|' for k in TICKERS)
    if errors:
        lines.extend(['','## SEC 원문 오류','',*['- '+' · '.join(map(str,row)) for row in errors]])
    lock=CACHE/'blind-lock.json'
    if lock.exists() and (BASE/'IDEAS.md').exists():
        stamp=json.loads(lock.read_bytes())
        lines.extend(['','## 2단계 인계','',
            f"56개 사례 JSON과 요약표를 {stamp['completed_at']}에 먼저 잠근 뒤 역할표를 열었다. 요약표 SHA-256은 {stamp['summary_sha256']}이다. 사례 56건 중 54건은 SEC에서 해당 시기 원문을 연결했고, SAP 2건은 뉴스로 조사했다. 28쌍 × 13코드 표를 원자료로 대조했다. 결과와 후속 가설 3개는 IDEAS.md에 있다.",
            '검증용 저장소 사본에서 전체 unittest 557건이 통과했다. 열거된 산출물 외 원본·검증용 저장소 사본은 Git 제외 경로 cache/에 있다. 검증 사본의 재귀 삭제 명령은 실행 정책에서 거부되어 해당 사본은 cache/qa에 남겼다.'])
    (BASE/'PROGRESS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__': main()


