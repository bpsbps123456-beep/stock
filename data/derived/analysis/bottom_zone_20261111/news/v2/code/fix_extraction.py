"""Correct article selectors on public pages with related-story cards first."""
import json,requests
from bs4 import BeautifulSoup
from collect_v2 import V2,save
for path in (V2/'raw/access_B1').glob('*.json'):
 r=json.loads(path.read_text(encoding='utf-8'))
 if r.get('extraction_revision') or not any(h in r.get('resolved_url','') for h in ('medirect.com','investordaily.com','news1.kr')):continue
 try:
  response=requests.get(r['resolved_url'],timeout=25,headers={'User-Agent':'Mozilla/5.0 (compatible; ResearchEvidence/1.0)'})
  if response.status_code!=200:continue
  soup=BeautifulSoup(response.content,'html.parser')
  for tag in soup(['script','style','nav','header','footer','form','noscript']):tag.decompose()
  r['previous_extraction']=r['text'];r['text']=(soup.find('main') or soup.body or soup).get_text(' ',strip=True)[:100000]
  r['extraction_revision']='main/body selector; related article first selector corrected, public HTTP only'
  save(path,r)
 except requests.RequestException as e:print(path.name,type(e).__name__)
