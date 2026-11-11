"""Create versioned evidence manifest before any corresponding outcome join."""
from datetime import datetime,timezone
import argparse,hashlib,json
from collect_v2 import NEWS,V2,save

def digest(p):return hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()

def freeze(stage):
 name='freeze_B_v2.json' if stage=='B1' else 'freeze_B2_v2.json'
 target=NEWS/name
 if target.exists():raise SystemExit('Already frozen')
 if stage=='B1':
  files=[NEWS/'protocol_B_v2.json',NEWS/'review_B1_v2.csv']+[p for p in V2.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
 else:
  files=[NEWS/'protocol_B_v2.json',NEWS/'bottom_news.csv',NEWS/'search_log_B2.csv']+[p for p in (V2/'B2').rglob('*') if p.is_file() and '__pycache__' not in p.parts]+[p for p in (V2/'raw/B2').rglob('*') if p.is_file()]
 manifest=dict(stage=stage,status='evidence_frozen_before_'+stage+'_outcome_analysis',frozen_at=datetime.now(timezone.utc).isoformat(),price_access_before_B1_freeze=False,B2_outcome_join_before_B2_freeze=False,hash_algorithm='SHA-256 of UTF-8 files with CRLF normalized to LF; BOM retained',input_hashes={p.relative_to(NEWS).as_posix():digest(p) for p in sorted(set(files))},parent_freeze='freeze_B.json',counts=json.loads((V2/'census_summary_B1.json').read_text(encoding='utf-8')) if stage=='B1' else {})
 save(target,manifest);print(name,len(files),'files')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['B1','B2']);freeze(p.parse_args().stage)
