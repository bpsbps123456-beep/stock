from common import *
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET
import json
import re
NAMES='Astera Labs|Arm Holdings|Aehr Test Systems|Oruka Therapeutics|Quantinuum|Intuitive Machines|SpaceX|IREN|Planet Labs|Accenture|First Solar|Bright Minds Biosciences|IPG Photonics|American Electric Power|Micron|NXP Semiconductors|Serve Robotics|Roper Technologies|Verizon|Riot Platforms|Strategy Inc|Circle Internet Group|Micron|Microchip Technology|Fortinet|Ciena|Tempus AI|Navitas Semiconductor|Monolithic Power Systems|PepGen|Bloom Energy|Aehr Test Systems|IonQ|SAP|Lumentum|Serve Robotics|AXT Inc|AST SpaceMobile|Texas Instruments|SAP|Vicor|Datadog|Intel|ServiceNow|IBM|Corning|UnitedHealth|Carpenter Technology|Marvell|PDD Holdings|Vicor|GE Vernova|Celestica|Ondas|Fastly|Silicon Motion'.split('|')
CASES=pd.read_csv(BASE/'cases_blind.csv')
def run(r,name):
    start=pd.Timestamp(r.ann_date)-pd.Timedelta(days=10)
    end=pd.Timestamp(r.reaction_date)+pd.Timedelta(days=30)
    key=f'"{name}"'
    if r.ticker=='MSTR': key='("Strategy" OR "MicroStrategy") Bitcoin'
    if r.ticker=='ARM': key='"Arm" semiconductor'
    if r.ticker=='PL': key='"Planet Labs"'
    pre=f' after:{(start-pd.Timedelta(days=1)).date()} before:{r.ann_date}'
    full=f' after:{(start-pd.Timedelta(days=1)).date()} before:{(end+pd.Timedelta(days=1)).date()}'
    queries=[
        key+' (launches OR announces OR partnership OR contract OR unveils OR offering OR acquisition OR selects OR approval)'+full,
        key+' (earnings preview OR expectations OR outlook OR ahead of earnings)'+pre,
        key+' (WallStreetBets OR "retail investors" OR "retail traders" OR "most bought" OR "meme stock")'+pre,
        key+' earnings results'+full,
        key+' (upgrade OR initiates coverage OR raises price target OR index inclusion OR buyback OR lawsuit)'+full,
    ]
    out=[]
    for j,q in enumerate(queries):
        path=fetch('https://news.google.com/rss/search',f'news/{r.case_id}-{j}.xml',params={'q':q,'hl':'en-US','gl':'US','ceid':'US:en'})
        items=[]
        for n in ET.fromstring(path.read_bytes()).findall('.//item'):
            date=parsedate_to_datetime(n.findtext('pubDate')).date().isoformat()
            if date<str(start.date()) or date>str(end.date()) or (j in (1,2) and date>=r.ann_date): continue
            title=n.findtext('title').rsplit(' - ',1)[0]
            # Blind the researcher to headlines describing realized share performance.
            if re.search(r'(?:shares?|stock|rally|gains?) (?:is |are |has |have |price |just )?(?:[+-]?\d+%|jump|soar|plung|pop|slid|surge|fell|fall|climb|ralli|tank|gain|drop|crash|sink|skyrocket|spik|t[ou]mbl|slump|rocke|outperform|rebound|rises|dips|slips)|% (?:gain|loss|higher|lower|rally)|why .* (?:up|down|higher|lower) |stock reaction',title,re.I): continue
            items.append({'date':date,'title':title,'publisher':n.findtext('source'),'url':n.findtext('link')})
        out.append(items)
    (CACHE/'news'/(r.case_id+'.json')).write_text(json.dumps(out,ensure_ascii=False),encoding='utf-8')
if __name__=='__main__':
    with ThreadPoolExecutor(3) as pool:
        futures=[pool.submit(run,r,name) for r,name in zip(CASES.itertuples(),NAMES,strict=True)]
        for f in futures: f.result()

