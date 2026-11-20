from common import *
import sys
import runpy

def sec_archives():
    for q in ['2025q3','2025q4','2026q1','2026q2','2026q3']:
        section = 'datastandardsinnovation' if q >= '2026q2' else 'structureddata'
        fetch(f'https://www.sec.gov/files/{section}/data/insider-transactions-data-sets/'+q+'_form345.zip',q+'.zip',True)

def short_interest():
    dates = runpy.run_path(str(BASE.parents[1]/'hf_options/code/fetch_finra_si.py'))['OFFICIAL_PUB'].copy()
    dates.update({'2025-09-15':'2025-09-24','2025-09-30':'2025-10-09','2025-10-15':'2025-10-24','2025-10-31':'2025-11-11'})
    rows = []
    for d,pub in sorted(dates.items()):
        if not START <= pub <= END:
            continue
        path = fetch('https://cdn.finra.org/equity/otcmarket/biweekly/shrt'+d.replace('-','')+'.csv', 'finra/'+d+'.csv')
        table = pd.read_csv(path, sep='|',low_memory=False)
        assert table.settlementDate.eq(d).all()
        sym = table.symbolCode.replace({'MOG.A':'MOG-A','MOG/A':'MOG-A','MOGA':'MOG-A','MOG A':'MOG-A','PSTG':'P'})
        take = table[sym.isin(TICKERS)].copy()
        take['ticker'] = sym[take.index]
        take = take[~((take.ticker=='P') & ~take.issueName.str.contains('pure storage|everpure',case=False,na=False))]
        take = take[~(take.ticker.isin(['SATS','ECHO']) & ~take.issueName.str.contains('echostar',case=False,na=False))]
        take = take[~((take.ticker=='SPCX') & ~take.issueName.str.contains('space exploration',case=False,na=False))]
        take['settlement_date'] = d
        take['publish_date'] = pub
        take.rename(columns={'currentShortPositionQuantity':'short_qty','averageDailyVolumeQuantity':'avg_daily_vol','daysToCoverQuantity':'days_to_cover','symbolCode':'source_ticker'},inplace=True)
        rows.append(take[['settlement_date','publish_date','ticker','short_qty','avg_daily_vol','days_to_cover','source_ticker','issueName','stockSplitFlag','revisionFlag']])
    out = pd.concat(rows)
    assert not out.duplicated(['ticker','settlement_date']).any()
    out.to_parquet(DATA/'short_interest.parquet',index=False)
if __name__=='__main__':
    {'sec':sec_archives,'short':short_interest}[sys.argv[1]]()


