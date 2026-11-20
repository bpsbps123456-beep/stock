from case_sources import *
import sys
sys.stdout.reconfigure(encoding='utf-8')
for n in sys.argv[1:]:
    k=f'C{int(n):02d}'
    p=CACHE/'casepr'/(k+'.json')
    if not p.exists(): print(k,'—'); continue
    j=json.loads(p.read_bytes())
    print(k,j['filing_date'],j['url'].split('/')[-1])
    raw=' '.join(j['lines'])
    key=r'announced today|today (?:announced|unveiled|launched)|for the first time|first-ever|first time|new (?:customer|contract|business|platform|program|product|design)|newly|first (?:commercial|production|design|customer|shipment)|first-in-class|secured|initial (?:production|order)|repurchas|new high|reached an agreement|subsequent event|signed a|announced a|introduced|record backlog'
    hits=[s for s in re.split(r'(?<=[.!?])\s+| • ',raw) if re.search(key,s,re.I) and not re.search('forward-looking|prior authorization|income taxes|may differ|adjusted ebitda|accounting policies',s,re.I)]
    for h in hits[:5]:
        mat=re.search(key,h,re.I)
        print(h[max(0,mat.start()-65):mat.start()+215])

