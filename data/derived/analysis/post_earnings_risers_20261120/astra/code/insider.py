from common import *
import json
import zipfile
from xml.etree import ElementTree
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

def issuers():
    path=fetch('https://www.sec.gov/files/company_tickers.json','company_tickers.json',True)
    t=pd.DataFrame(json.loads(path.read_bytes()).values()).set_index('ticker')
    ciks={s:int(t.loc[s,'cik_str']) for s in TICKERS if s in t.index}
    ciks.update(SATS=ciks['ECHO'], SKHYV=ciks['SKHY'], QQQ=1067839, SOXX=1100663)
    return ciks

def recent(issuer):
    path=fetch(f'https://data.sec.gov/submissions/CIK{issuer:010d}.json',f'sub/{issuer:010d}.json',True)
    allfilings=json.loads(path.read_bytes())['filings']
    tables=[pd.DataFrame(allfilings['recent'])]
    for old in allfilings['files']:
        if old['filingTo']>='2026-09-30' and old['filingFrom']<=END:
            path=fetch('https://data.sec.gov/submissions/'+old['name'],'sub/'+old['name'],True)
            tables.append(pd.DataFrame(json.loads(path.read_bytes())))
    tab=pd.concat(tables)
    return tab[tab.form.eq('4') & tab.filingDate.between('2026-09-30',END)]

def parse_xml(issuer,r):
    accession=r.accessionNumber
    doc=r.primaryDocument.split('/')[-1]
    cache='xml/'+accession+'.xml'
    path=fetch(f'https://www.sec.gov/Archives/edgar/data/{issuer}/{accession.replace("-","")}/{doc}',cache,True)
    tree=ElementTree.fromstring(path.read_bytes())
    if int(tree.findtext('issuer/issuerCik'))!=issuer:
        return []
    src=tree.findtext('issuer/issuerTradingSymbol')
    owners='|'.join(sorted({str(int(i.text)) for i in tree.findall('./reportingOwner/reportingOwnerId/rptOwnerCik')}))
    rows=[]
    for j,el in enumerate(tree.findall('nonDerivativeTable/nonDerivativeTransaction')):
        code=el.findtext('transactionCoding/transactionCode')
        if code not in ('P','S'): continue
        rows.append((r.filingDate,el.findtext('transactionDate/value'),issuer,src,code,el.findtext('transactionAmounts/transactionShares/value'),el.findtext('transactionAmounts/transactionPricePerShare/value'),accession,str(j),owners,el.findtext('securityTitle/value')))
    return rows

COLS=['filing_date','trade_date','issuer_cik','source_ticker','code','shares','price','accession','transaction_id','owner_ciks','security_title']

def quarterly(ciks):
    tables=[]
    for path in sorted(CACHE.glob('*q*.zip')):
        with zipfile.ZipFile(path) as z:
            def read(name):
                with z.open(name+'.tsv') as f: return pd.read_csv(f,sep='\t',low_memory=False,encoding_errors='replace')
            sub=read('SUBMISSION')
            sub=sub[sub.ISSUERCIK.isin([*ciks.values(),34088]) & sub.DOCUMENT_TYPE.eq('4')].copy()
            sub['filing_date']=pd.to_datetime(sub.FILING_DATE,format='%d-%b-%Y').dt.strftime('%Y-%m-%d')
            sub=sub[sub.filing_date.between(START,END)]
            trans=read('NONDERIV_TRANS')
            trans=trans[trans.TRANS_CODE.isin(['P','S'])].merge(sub,on='ACCESSION_NUMBER')
            owners=read('REPORTINGOWNER').groupby('ACCESSION_NUMBER').RPTOWNERCIK.agg(lambda s:'|'.join(sorted(set(s.dropna().astype(int).astype(str)))))
            trans['owner_ciks']=trans.ACCESSION_NUMBER.map(owners)
            trans['trade_date']=pd.to_datetime(trans.TRANS_DATE,format='%d-%b-%Y',errors='coerce').dt.strftime('%Y-%m-%d')
            trans=trans.rename(columns=dict(ISSUERCIK='issuer_cik',ISSUERTRADINGSYMBOL='source_ticker',TRANS_CODE='code',TRANS_SHARES='shares',TRANS_PRICEPERSHARE='price',ACCESSION_NUMBER='accession',NONDERIV_TRANS_SK='transaction_id',SECURITY_TITLE='security_title'))
            tables.append(trans[COLS])
    return pd.concat(tables,ignore_index=True)

def save(table,ciks):
    bycik=defaultdict(list)
    for s,c in ciks.items(): bycik[c].append(s)
    bycik[34088]=['XOM']
    table=table.copy()
    table['ticker']=table.issuer_cik.map(bycik)
    table=table.explode('ticker')
    # Funds sharing a trust CIK must not be matched by CIK alone.
    table=table[~table.ticker.isin(['SOXX','QQQ','SPY']) | table.ticker.eq(table.source_ticker)]
    # Preserve listing-specific dates for temporary and renamed tickers.
    table=table[~table.ticker.isin(['SATS','ECHO','SKHY','SKHYV']) | table.ticker.eq(table.source_ticker)]
    for col in ('price','shares'): table[col]=pd.to_numeric(table[col],errors='coerce')
    table['transaction_id']=table.transaction_id.astype(str)
    table['value']=table.price*table.shares
    table=table.sort_values(['filing_date','accession','ticker','transaction_id'])
    assert not table.duplicated(['accession','transaction_id','ticker']).any()
    table.to_parquet(DATA/'insider.parquet',index=False)

def main():
    ciks=issuers()
    bulk=quarterly(ciks)
    save(bulk,ciks)
    seen=set(bulk.accession)
    errors=[]
    rows=[]
    def collect(issuer):
        collected=[]
        failures=[]
        try:
            for r in recent(issuer).itertuples():
                if r.accessionNumber in seen: continue
                try:
                    collected.extend(parse_xml(issuer,r))
                except Exception as exc:
                    failures.append([issuer,r.accessionNumber,str(exc)])
        except Exception as exc: failures.append([issuer,'submissions',str(exc)])
        return collected,failures
    with ThreadPoolExecutor(5) as executor:
        futures=[executor.submit(collect,issuer) for issuer in sorted({*ciks.values(),34088})]
        for future in as_completed(futures):
            collected,failures=future.result()
            rows.extend(collected)
            errors.extend(failures)
            (CACHE/'sec-errors.json').write_text(json.dumps(errors,indent=1))
    save(pd.concat([bulk,pd.DataFrame(rows,columns=COLS)],ignore_index=True),ciks)
if __name__=='__main__': main()

