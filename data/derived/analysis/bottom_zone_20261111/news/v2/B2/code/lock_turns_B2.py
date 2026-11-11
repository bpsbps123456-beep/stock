"""Manual per-episode catalyst adjudication, locked before B2 comparisons."""
import json
from collect_B2 import B2,save,csvwrite

DECISIONS=[
 ('2024-04-19','real','none','','기술주 반등·금리 안정 보도. 새 정책·실적·수급 재료 확인 못함.'),
 ('2024-08-07','real','policy','601d1a0159ec89b6','+1 미국 신규 실업수당 지표 안도 뒤 Nasdaq/S&P 2% 반등.'),
 ('2024-09-06','real','none','','+1 직전 주 급락 뒤 기술적 반등. 물가 발표·금리 인하는 아직 예정.'),
 ('2024-11-15','real','none','','+1 Nvidia 실적을 기다리며 반등. 새 실적 발표는 아님.'),
 ('2025-04-08','real','policy','5bf765b928690431','+1 Trump 상호관세 90일 유예 발표와 폭등.'),
 ('2025-11-20','real','policy','82d77aadd8fbc885','+1 Fed 인사의 12월 금리 인하 가능성 발언과 반등.'),
 ('2026-03-30','real','policy','a07a124a120efcda','+1 중동 전쟁 종료 경로 기대와 급반등. 기대 보도이며 실제 종전 확인 아님.'),
 ('2026-06-05','real','flows','e09a5f4ce3e5b19d','+1 금요일 기술주 급락 뒤 저가 매수에 반도체 반등.'),
 ('2026-07-29','real','earnings','23696e679d904fdb','+1 Microsoft 실적 호조와 반도체 동반 강세.'),
 ('2026-09-14','real','none','','+1 AI 개발 속도 조절 요구·유가·국채금리 부담 지속. 새 긍정 재료 확인 못함.'),
 ('2024-07-19','fake','none','','+1 전주 손실을 일부 만회한 기술적 반등.'),
 ('2024-08-28','fake','none','','+1 Nvidia 실적 상회에도 기대 미달·주가 하락. 긍정 전환 재료 없음.'),
 ('2025-02-27','fake','policy','1110d703cd42058f','+1 물가 둔화와 국채금리 하락 안도로 반등.'),
 ('2026-03-06','fake','policy','462e31af286dd643','+1 전쟁 해결 기대 보도와 반등. 실제 종전 확인 아님.'),
 ('2026-07-02','fake','flows','f8045ed1251f41df','+1 AI 관련 기업 주식 매수·Broadcom 반등. 실적은 아직 예정.'),
 ('2026-08-19','fake','none','','+1에 읽은 미 국채 매입 관련 보도는 넓은 지수에 긍정이나 반도체 약세. 반도체 전환 재료 확인 못함.'),
 ('2026-11-05','current','flows','8b2ffe817284f013','+1 AI 쇼크 뒤 저가 매수와 반도체 반등. 진행 중이라 평가 제외.'),
]

def main():
 vals=json.loads((B2/'coded_candidates_B2.json').read_text(encoding='utf-8'));byid={r['candidate_id']:r for r in vals}
 rows=[]
 for anchor,typ,cat,cid,note in DECISIONS:
  r=byid.get(cid,{})
  rows.append(dict(anchor_date=anchor,anchor_type=typ,catalyst=cat,candidate_id=cid,published_date=r.get('published_date',''),headline=r.get('headline',''),url=r.get('url',''),verification=r.get('verification','제목·요약만') if cid else '창의 0/+1 기사 목록에서 확인한 긍정 재료 없음',note=note))
 csvwrite(B2/'turn_catalysts_B2.csv',rows)
 save(B2/'evidence_QA_B2.json',dict(status='locked before B2 real/fake tone comparison',manual_headline_decisions=1458,manual_queues=[1056,231,20,151],body_read_unique=14,source_date_repairs='Exact date in AP index-close title or displayed article timestamp supersedes missing web date; richest summary retained. Feed-only dates remain flagged. Reuters 2025-11-21 corrected from resolved public URL.',wire_coverage='181 unique requested days all have a wire close; 185 observable anchor-session rows, 2 future rows unobservable',turn_rule='Positive causal material on 0/+1, one main policy/earnings/flows/strategist/none per episode; technical bounce or earnings preview is none. Market rebound is not inferred from later prices.',source_frame='Uniform registered primary publishers and identifiable wire reprints; all eligible frame headlines manually reviewed; all other search results retained with rule-based exclusions',limitations=['Anchor real/fake labels known; dates shuffled, labels not blinded.','B2 source/date relevance parser amended during collection before outcomes.','Unarchived initial 4-query request repeated and registered; repeated results archived, initial response not locally preserved.','Most items headline/search-summary only. HTTP denial or site/login barrier not bypassed.','Feed date proxies and publication updates can move item across session; no exact intraday event chronology claimed.','Wire updates with different headlines can remain dependent.','B1 prices already available after separate B1 freeze; B2 inclusion did not use subsequent returns.']))
if __name__=='__main__':main()
