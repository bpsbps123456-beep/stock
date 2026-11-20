from common import *
import json
from urllib.parse import quote

# Candidate titles identified from the issuer names in the fund roster.
CANDIDATES = """AAOI Applied Optoelectronics
AAPL Apple Inc.
ACN Accenture
ADI Analog Devices
AEHR Aehr Test Systems
AEP American Electric Power
AKAM Akamai Technologies
ALAB Astera Labs
ALNY Alnylam Pharmaceuticals
AMAT Applied Materials
AMD AMD
AMKR Amkor Technology
AMZN Amazon (company)
ANET Arista Networks
AOSL Alpha and Omega Semiconductor
APP AppLovin
ARM Arm Holdings
ASML ASML Holding
ASTS AST SpaceMobile
AVGO Broadcom Inc.
AXTI AXT Inc.
BABA Alibaba Group
BE Bloom Energy
BIDU Baidu
BIIB Biogen
BKR Baker Hughes
CBRS Cerebras
CCJ Cameco
CEG Constellation Energy
CIEN Ciena
CLS Celestica
COHR Coherent Corp.
COIN Coinbase
COST Costco
CRCL Circle (company)
CRDO Credo Technology
CRM Salesforce
CRS Carpenter Technology Corporation
CRWD CrowdStrike
CRWV CoreWeave
CSCO Cisco
DDOG Datadog
DELL Dell Technologies
DG Dollar General
DRUG Bright Minds Biosciences
DT Dynatrace
DUK Duke Energy
DYN Dyne Therapeutics
EBAY EBay
ECHO EchoStar
EOSE Eos Energy Enterprises
FIX Comfort Systems USA
FLEX Flex Ltd.
FLNC Fluence Energy
FN Fabrinet
FORM FormFactor
FSLR First Solar
FSLY Fastly
FTNT Fortinet
GEV GE Vernova
GFS GlobalFoundries
GH Guardant Health
GILD Gilead Sciences
GLW Corning Inc.
GOOG Alphabet Inc.
GOOGL Alphabet Inc.
GTLB GitLab Inc.
HOOD Robinhood Markets
IBKR Interactive Brokers
IBM IBM
ILMN Illumina, Inc.
INTC Intel
IONQ IonQ
IPGP IPG Photonics
IREN IREN Limited
JAZZ Jazz Pharmaceuticals
KEYS Keysight
KLAC KLA Corporation
KO The Coca-Cola Company
LBRX LB Pharmaceuticals
LEU Centrus Energy
LITE Lumentum
LLY Eli Lilly and Company
LRCX Lam Research
LUNR Intuitive Machines
MCHP Microchip Technology
MDB MongoDB Inc.
MELI Mercado Libre
META Meta Platforms
MOG-A Moog Inc.
MPWR Monolithic Power Systems
MRVL Marvell Technology
MSFT Microsoft
MSTR Strategy (company)
MU Micron Technology
NBIS Nebius Group
NET Cloudflare
NOW ServiceNow
NTAP NetApp
NTRA Natera
NVDA Nvidia
NVTS Navitas Semiconductor
NXPI NXP Semiconductors
NXT Nextracker
OKLO Oklo Inc.
OKTA Okta, Inc.
ON Onsemi
ONC BeOne Medicines
ONDS Ondas Holdings
ORKA Oruka Therapeutics
P Pure Storage
PANW Palo Alto Networks
PDD PDD Holdings
PEPG PepGen
PL Planet Labs
PLTR Palantir Technologies
POET POET Technologies
QBTS D-Wave Systems
QCOM Qualcomm
QNT Quantinuum
RDDT Reddit, Inc.
RIOT Riot Platforms
RKLB Rocket Lab
RMBS Rambus
ROP Roper Technologies
SAP SAP SE
SATS EchoStar
SERV Serve Robotics
SIMO Silicon Motion
SKHY SK Hynix
SKHYV SK Hynix
SMR NuScale Power
SNDK SanDisk
SNOW Snowflake Inc.
SPCX SpaceX
STM STMicroelectronics
STX Seagate Technology
TEM Tempus AI
TER Teradyne
TKR Timken Company
TSEM Tower Semiconductor
TSLA Tesla, Inc.
TSM TSMC
TTMI TTM Technologies
TTWO Take-Two Interactive
TWLO Twilio
TXN Texas Instruments
U Unity Technologies
UEC Uranium Energy Corporation
UNH UnitedHealth Group
UUUU Energy Fuels
VIAV Viavi Solutions
VICR Vicor Corporation
VRT Vertiv
VRTX Vertex Pharmaceuticals
VZ Verizon Communications
WDAY Workday, Inc.
WDC Western Digital
WMT Walmart
XNDU Xanadu Quantum Technologies
XOM ExxonMobil
QQQ Invesco QQQ
SPY SPDR S&P 500 ETF Trust
SOXX iShares Semiconductor ETF"""
# Apply candidate corrections only after reviewing identity in English Wikipedia.
CORRECTIONS = {'IREN':'IREN','UUUU':'Energy Fuels Inc.'}
def run():
    candidates = dict(l.split(' ',1) for l in CANDIDATES.splitlines())
    candidates.update(CORRECTIONS)
    assert set(candidates)==set(TICKERS)
    titles = sorted(set(candidates.values()))
    mapping = {}
    for i in range(0,len(titles),25):
        group=titles[i:i+25]
        import hashlib
        key=hashlib.sha256('|'.join(group).encode()).hexdigest()[:12]
        path=fetch('https://en.wikipedia.org/w/api.php', 'titles/'+key+'.json',params={'action':'query','format':'json','formatversion':2,'titles':'|'.join(group),'redirects':1,'prop':'pageprops|description','ppprop':'disambiguation|wikibase_item'})
        query=json.loads(path.read_bytes())['query']
        norm={r['from']:r['to'] for k in ['normalized','redirects'] for r in query.get(k,[])}
        pages={p['title']:p for p in query['pages']}
        for t in group:
            canon=t
            while canon in norm: canon=norm[canon]
            page=pages[canon]
            ok = 'missing' not in page and 'disambiguation' not in page.get('pageprops',{})
            mapping[t]=(canon if ok else None,page.get('description',''),page.get('pageprops',{}).get('wikibase_item'))
    records=[]
    for ticker,title in candidates.items():
        canon,desc,qid=mapping[title]
        records.append((ticker,canon,'manual', 'high' if canon and ticker not in ('PDD','RDDT') else 'low',title,desc,qid))
    df=pd.DataFrame(records,columns=['ticker','article','method','confidence','candidate','description','wikidata_id'])
    df.to_csv(DATA/'wiki_map.csv', index=False)
    def views(title):
        url='https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/user/'+quote(title.replace(' ','_'),safe='')+'/daily/20250901/20261119'
        file='views/'+quote(title,safe='')+'.json'
        if not (CACHE/file).exists():
            time.sleep(6.2)
        try:
            path=fetch(url,file)
        except requests.HTTPError as exc:
            if exc.response.status_code != 429: raise
            time.sleep(max(15,int(exc.response.headers.get('Retry-After','15'))))
            path=fetch(url,file)
        rows=json.loads(path.read_bytes())['items']
        return [(r['timestamp'][:8], title,r['views']) for r in rows]
    entries=[]
    throttles=0
    for title in df.article.dropna().unique():
        if throttles >= 3 and not (CACHE/'views'/(quote(title,safe='')+'.json')).exists():
            continue
        try:
            entries.extend(views(title))
        except requests.HTTPError as exc:
            if exc.response.status_code==429:
                throttles+=1
            print(title,exc,flush=True)
    daily=pd.DataFrame(entries,columns=['date','article','views'])
    daily['date']=pd.to_datetime(daily.date,format='%Y%m%d')
    daily.merge(df[['article','ticker']],on='article')[['date','ticker','views']].to_parquet(DATA/'wiki_views.parquet',index=False)
if __name__=='__main__': run()

