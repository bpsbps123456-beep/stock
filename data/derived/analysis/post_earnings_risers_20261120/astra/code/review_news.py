from case_sources import *
import sys
sys.stdout.reconfigure(encoding='utf-8')
for i in map(int,sys.argv[1:]):
    r=CASES.iloc[i-1]
    groups=json.loads((CACHE/'news'/(r.case_id+'.json')).read_text(encoding='utf-8'))
    print('\n',r.case_id,r.ticker,r.ann_date)
    for g,items in enumerate(groups):
        scored=[]
        for j,item in enumerate(items):
            t=item['title']
            if re.search(r'(shares? (up|down|higher|lower)|stock (rout|selloff|rall|sell-off)|% (rally|surge)|\b(?:skyrockets|plummets|tumbles|rallies)\b)',t,re.I): continue
            key=[r.ticker, NAMES[i-1].split()[0]]
            company=any(re.search(r'\b'+re.escape(k)+r'\b',t,re.I) for k in key)
            verbs=r'announces|launches|unveils|secures|awards?|agreement|collaborat|partner|contract|selects|introduces|joins|approv|appoint|acqui|phase [12]|order|invests|invest(s|ment) in'
            if g==0 and not re.search(verbs,t,re.I):continue
            if g==0 and re.search(r'(quarter|financial|annual|interim|earnings) results|conference|earnings call|withhold|shareholder alert',t,re.I):continue
            if g==1 and not re.search(r'preview|ahead|expect|forecast|earnings|bullish|bearish|upgrad|downgrad',t,re.I): continue
            if g==2 and not re.search(r'retail (trader|investor|buy)|wallstreetbets|meme stock|most.bought|retail.favorite|reddit',t,re.I):continue
            if g==4 and not re.search(r'upgrad|initiates|initiated|coverage|index|joi(ns|ning) the s&p|buyback|launches.*lawsuit|raises .*target.* (from|to) ',t,re.I):continue
            if g==3 and not (company and abs((pd.Timestamp(item['date'])-pd.Timestamp(r.ann_date)).days)<=1):continue
            score=4*company+2*bool(re.search(verbs,t,re.I))+2*bool(re.search('business wire|globenewswire|pr newswire|reuters',item['publisher'],re.I))
            scored.append((score,j,item))
        seen=set()
        for _,j,a in sorted(scored,key=lambda x:-x[0]):
            sig=a['title'].lower()[:46]
            if sig in seen:continue
            seen.add(sig)
            print(f"{g}.{j} {a['date'][5:]} {a['title'][:140]}")
            if len(seen)>=[6,3,2,1,3][g]:break

