"""Explicit evidence adjudications made before the committed B2 freeze."""
import json,re
from collect_B2 import B2,save

def main():
 over=json.loads((B2/'manual_overrides_B2.json').read_text(encoding='utf-8'))
 def fix(cid,**fields):
  over.setdefault(cid,{}).update(fields)
 fix('430e8035f056393d',decision='include',tone='neg',type='regulation_geopolitics',possible_turn=False,note='final_raw_001 AP 3/30 close brief: S&P -.4 Nasdaq -.7; war and oil uncertainty. Search summary read.')
 for cid in ['9a54df866e88366e','dff13fa80ef41310','c2ae193af4bdd7c9','24f176678dc0bdab']:
  fix(cid,possible_turn=False)
 fix('668913d153a102d7',tone='pos',type='technical',possible_turn=False,note='Yields stabilized and tech rebound; no fresh positive policy/earnings release identified.')
 for cid in ['276add618101ab49','aa5cec8c2bf441bb']:
  fix(cid,tone='neutral',type='macro_policy',possible_turn=False,note='원문: 미 증시는 소폭 상승했으나 SOX 하락. 반도체 전환 재료로 판정하지 않음.')
 fix('a82d488eb2477f90',published_date='2025-11-21',date_basis='공개 리디렉션으로 확인한 Reuters 원문 URL 2025-11-21; RSS 11/20 대용 날짜 수정',tone='pos',type='macro_policy',possible_turn=True)
 fix('e09a5f4ce3e5b19d',tone='pos',type='flows_positioning',possible_turn=True,note='Reuters 재보도 검색 요약: Friday selloff 뒤 bargain hunting으로 칩주 반등.')
 fix('601d1a0159ec89b6',tone='pos',type='macro_policy',possible_turn=True,note='Reuters 마감 제목: 신규 실업수당 지표 뒤 S&P/Nasdaq 2% 상승. 거시 지표 안도.')
 fix('f8045ed1251f41df',tone='pos',type='flows_positioning',possible_turn=True,note='Reuters 재보도 요약: 투자자가 AI 기업 주식을 매수하며 Broadcom 등 반도체 반등.')
 for cid in ['9086ea95cb06e0a8','8cbc40669236562c','0fae89cafb639e08']:fix(cid,possible_turn=False,note='중동 상황 주목/단순 반등만으로 새 전환 재료를 확인할 수 없음.')
 fix('e747fb75f26ab3e3',wire_close_verified=True,note='7/31 Reuters 종합: scored biggest daily percentage gains, Dow rose. 마감 판정; 제목에 close가 없었음.')
 fix('38e364e1a98fcc3c',wire_close_verified=True,note='9/3 Reuters 16:02 ET 마감 보도: SP -2%, Nasdaq -3%. 단순 close 단어 부재로 누락된 마감 기사 복원.')
 # Only actually displayed and read body text is upgraded from summary-only.
 readids=['1c3aa','396','87f','b9ff','5f0a','1110','00486','fc0b','8a17','71e7','23696e679d904fdb','9a54df866e88366e','8b2ffe817284f013','276add618101ab49','aa5cec8c2bf441bb']
 logs=[]
 for p in (B2/'body').glob('*.json'):
  b=json.loads(p.read_text(encoding='utf-8'));cid=b['candidate_id'];isread=any(cid.startswith(i) for i in readids)
  if isread and b.get('access')=='retrieved_pending_read':
   src=b.get('resolved_url','');verification='재보도 본문 열람' if re.search('yahoo|kitco|citynews',src) else '원문 열람'
   fix(cid,verification=verification);logs.append(dict(candidate_id=cid,read=True,verification=verification,resolved_url=src))
  elif b.get('access','').startswith('unread'):
   fix(cid,verification='제목·요약만; 원문 못 읽음 ('+b.get('access','unread')+')')
  # Retrieved pending text remains unread unless explicitly adjudicated above.
  if 'seattlepi.com' in b.get('resolved_url','') and 'cannot load' in b.get('text','').lower():
   fix(cid,verification='제목·요약만; 원문 본문 로드 차단')
 save(B2/'body_read_verification_B2.json',logs)
 # A rule-suggested turn is not accepted when its final tone/type is ineligible.
 for d in over.values():
  if d.get('tone')!='pos' or d.get('type') in {'technical','other','capitulation_language'}:d['possible_turn']=False
 save(B2/'manual_overrides_B2.json',over)
 print('decisions',len(over),'actually read bodies',len(logs))
if __name__=='__main__':main()
