"""Write each hand-coded blind record from observations and its cited source."""
from case_sources import *
from urllib.parse import urlencode
import sys

def save(text):
    for block in text.strip().split('\n\n'):
        lines=block.splitlines()
        k=lines.pop(0)
        i=int(k)-1
        r=CASES.iloc[i]
        case=r.case_id
        news=json.loads((CACHE/'news'/(case+'.json')).read_bytes())
        prfile=CACHE/'casepr'/(case+'.json')
        pr=json.loads(prfile.read_bytes()) if prfile.exists() else None
        winstart=str((pd.Timestamp(r.ann_date)-pd.Timedelta(days=10)).date())
        winend=str((pd.Timestamp(r.reaction_date)+pd.Timedelta(days=30)).date())
        def searchlink(g):
            root=ET.fromstring((CACHE/'news'/f'{case}-{g}.xml').read_bytes())
            return root.findtext('./channel/link')
        def src(ref):
            if ref=='s':
                return pr['filing_date'],pr['url']
            a,b=map(int,ref.split('.'))
            n=news[a][b]
            return n['date'],n['url']
        row={'case_id':case,'ticker':r.ticker,'ann_date':r.ann_date,'reaction_date':r.reaction_date,
             'window':[winstart,winend],'observed_through':min(winend,END),'catalysts':[],
             'date_basis':'공식 발표일 또는 자료에서 사실을 확인할 수 있는 보도일. 이전 분기 거래는 창 안 실제 공시가 있는 경우 그 공시 내용을 명시했다.',
             'pre_retail_hype':{'value':'모름','one_line':'발표 전 10일간 개인 쏠림을 확인할 직접 자료 부족.','url':searchlink(2)},
             'expectation_before':{'value':'모름','one_line':'발표 전 시장 기대의 방향을 확인할 근거가 부족하다.','url':searchlink(1)},
             'confidence':'중간'}
        for line in lines:
            field,ref,note=line.split(' ',2)
            date,url=src(ref)
            assert winstart<=date<=min(winend,END),(case,date)
            if field.startswith('C'):
                row['catalysts'].append({'code':field,'date':date,'one_line':note,'url':url})
            elif field in ('예','아니오','모름'):
                row['narrative_new']={'value':field,'one_line':note,'url':url}
            elif field=='개인':
                assert date<r.ann_date
                row['pre_retail_hype']={'value':'있음','one_line':note,'date':date,'url':url}
            else:
                assert date<r.ann_date
                row['expectation_before']={'value':('모름' if field=='모름방향' else field),'one_line':note,'date':date,'url':url}
        if not row['catalysts']:
            cited=pr['url'] if pr else row['narrative_new']['url']
            row['catalysts']=[{'code':'C0','date':min(winend,END),'one_line':'검토한 공개 자료에서 이 기간의 추가 분류 사건을 확인하지 못했다.','url':cited}]
            row['confidence']='낮음'
        if row['narrative_new']['value']=='모름':row['confidence']='낮음'
        if row['narrative_new']['value']=='예' and row['catalysts'][0]['code']!='C0' and pr:row['confidence']='높음'
        row['earnings_source']=pr['url'] if pr else None
        path=BASE/'cases'/f'{case}.json'
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':
    save(Path(sys.argv[1]).read_text(encoding='utf-8-sig'))


