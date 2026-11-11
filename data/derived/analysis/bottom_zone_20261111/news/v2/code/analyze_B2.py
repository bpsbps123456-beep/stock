"""Compare already committed B2 codes; never alter frozen evidence."""
import csv,hashlib,json,re,subprocess
from collections import Counter
from pathlib import Path
from statistics import mean
from collect_v2 import NEWS,V2,save,csvwrite
OUT=V2/'results';B2=V2/'B2'

def verify_freeze():
 p=NEWS/'freeze_B2_v2.json';d=json.loads(p.read_text(encoding='utf-8'))
 for rel,expected in d['input_hashes'].items():
  actual=hashlib.sha256((NEWS/rel).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
  if actual!=expected:raise ValueError('Frozen B2 evidence changed: '+rel)
 commit=subprocess.check_output(['git','log','-1','--format=%H','--',str(p)],text=True).strip()
 if not commit:raise ValueError('B2 freeze must be committed before comparison')
 return commit

def period(k):return 'pre' if k<0 else 'zero' if k==0 else 'post'

def summarize(rows):
 tones=Counter(r['tone'] for r in rows)
 return dict(n=len(rows),neg=tones['neg'],pos=tones['pos'],neutral=tones['neutral'],negative_fraction=tones['neg']/len(rows) if rows else None,types=dict(Counter(r['type'] for r in rows)))

def analyze():
 commit=verify_freeze();OUT.mkdir(exist_ok=True)
 rows=list(csv.DictReader((NEWS/'bottom_news.csv').open(encoding='utf-8-sig')))
 for r in rows:r['k']=int(r['k']);r['period']=period(r['k']);r['year']=int(r['anchor_date'][:4])
 grid=[];episodes=[]
 anchors=sorted({(r['anchor_date'],r['anchor_type']) for r in rows})
 for day,typ in anchors:
  for p in ['pre','zero','post']:
   chosen=[r for r in rows if r['anchor_date']==day and r['period']==p]
   episodes.append(dict(anchor_date=day,anchor_type=typ,year=int(day[:4]),period=p,**summarize(chosen)))
 for year in ['all',2024,2025,2026]:
  for typ in ['real','fake']:
   for p in ['pre','zero','post']:
    chosen=[r for r in rows if r['anchor_type']==typ and r['period']==p and (year=='all' or r['year']==year)]
    equal=[r['negative_fraction'] for r in episodes if r['anchor_type']==typ and r['period']==p and (year=='all' or r['year']==year) and r['n']]
    grid.append(dict(year=year,anchor_type=typ,period=p,episode_count=len(equal),episode_equal_negative_fraction=mean(equal) if equal else None,**summarize(chosen)))
 # Homogeneous wire-close sensitivity: same AP/Reuters publisher frame.
 coverage=list(csv.DictReader((B2/'wire_coverage_B2.csv').open(encoding='utf-8-sig')))
 coded=json.loads((B2/'coded_candidates_B2.json').read_text(encoding='utf-8'))
 closeids={cid for w in coverage for cid in w['candidate_ids'].split(';') if cid}
 groups={r.get('source_group') for r in coded if r['candidate_id'] in closeids and re.search(r'reuters|apnews|ap news|associated press',r['source'],re.I)}
 wire=[r for r in rows if r['source_group'] in groups]
 wiregrid=[]
 for year in ['all',2024,2025,2026]:
  for typ in ['real','fake']:
   for p in ['pre','zero','post']:
    wiregrid.append(dict(year=year,anchor_type=typ,period=p,**summarize([r for r in wire if r['anchor_type']==typ and r['period']==p and (year=='all' or r['year']==year)])))
 turns=list(csv.DictReader((B2/'turn_catalysts_B2.csv').open(encoding='utf-8-sig')))
 turncounts={typ:dict(Counter(r['catalyst'] for r in turns if r['anchor_type']==typ)) for typ in ['real','fake','current']}
 recent=json.loads((B2/'protocol_B2_amendment.json').read_text(encoding='utf-8'))['recent10']
 unique={r['source_group']:r for r in coded if r['decision']=='include' and r.get('mapped_session') in recent}
 current=[]
 for day in recent:
  selected=[r for r in unique.values() if r['mapped_session']==day]
  current.append(dict(date=day,**summarize(selected)))
 included={r['candidate_id']:r for r in coded if r['decision']=='include'}
 verification=Counter(r.get('verification','제목·요약만') for r in included.values())
 read_original=sum(v for k,v in verification.items() if k=='원문 열람')
 read_reprint=sum(v for k,v in verification.items() if k=='재보도 본문 열람')
 out=dict(freeze_commit=commit,rows=len(rows),unique_articles=len({r['source_group'] for r in rows}),candidate_count=len(coded),adopted_candidates=len(included),grid=grid,episodes=episodes,wire_close_grid=wiregrid,turn_counts=turncounts,current10=current,verification=dict(verification),original_read=read_original,reprint_read=read_reprint,article_read_fraction=(read_original+read_reprint)/len(included),wire_coverage_status=dict(Counter(r['status'] for r in coverage)))
 save(OUT/'analysis_B2.json',out)
 csvwrite(OUT/'B2_period_counts.csv',[{k:v for k,v in r.items() if k!='types'} for r in grid])
 csvwrite(OUT/'B2_episode_counts.csv',[{k:v for k,v in r.items() if k!='types'} for r in episodes])
 csvwrite(OUT/'B2_types.csv',[dict(year=r['year'],anchor_type=r['anchor_type'],period=r['period'],type=t,n=n) for r in grid for t,n in r['types'].items()])
 csvwrite(OUT/'B2_current10.csv',[{k:v for k,v in r.items() if k!='types'} for r in current])
 print(json.dumps({k:out[k] for k in ['rows','unique_articles','adopted_candidates','original_read','reprint_read','article_read_fraction','wire_coverage_status','turn_counts']},ensure_ascii=False))
 for r in grid:
  if r['period']=='post':print(r['year'],r['anchor_type'],r['neg'],r['n'],r['negative_fraction'],r['episode_equal_negative_fraction'])
if __name__=='__main__':analyze()
