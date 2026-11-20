from common import *
import json
import hashlib
from datetime import datetime

codes=[f'C{i}' for i in range(13)]
def main():
    cases=pd.read_csv(BASE/'cases_blind.csv').set_index('case_id')
    assert len(cases)==56
    rows=[]
    for path in sorted((BASE/'cases').glob('C*.json')):
        r=json.loads(path.read_bytes())
        k=r['case_id']
        assert path.stem==k and cases.loc[k,'ticker']==r['ticker']
        tags={x['code'] for x in r['catalysts']}
        assert tags and tags<=set(codes) and ('C0' not in tags or len(tags)==1)
        assert all(r['window'][0]<=x['date']<=r['observed_through'] and x['url'].startswith('https://') for x in r['catalysts'])
        assert r['pre_retail_hype']['value'] in ('있음','없음','모름') and r['narrative_new']['value'] in ('예','아니오','모름')
        rows.append({'case_id':k,'ticker':r['ticker'],**{c:int(c in tags) for c in codes},'pre_retail_hype':r['pre_retail_hype']['value'],'narrative_new':r['narrative_new']['value']})
    assert [r['case_id'] for r in rows]==sorted(cases.index)
    file=BASE/'cases_summary.csv'
    content=pd.DataFrame(rows).to_csv(index=False,lineterminator='\r\n').encode('utf-8')
    digest=hashlib.sha256(content).hexdigest()
    lock=CACHE/'blind-lock.json'
    if lock.exists():
        assert digest==json.loads(lock.read_bytes())['summary_sha256'], 'Blind records differ from locked summary'
        assert file.read_bytes()==content
        return
    file.write_bytes(content)
    lock.write_text(json.dumps({'completed_at':datetime.now().astimezone().isoformat(),'summary_sha256':digest}))
if __name__=='__main__':main()

