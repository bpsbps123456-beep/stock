"""Research safeguards: immutable evidence, information timing and censoring."""
import importlib.util,json,sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
NEWS=ROOT/'data/derived/analysis/bottom_zone_20261111/news'
CODE=NEWS/'v2/code'
sys.path.insert(0,str(CODE))
spec=importlib.util.spec_from_file_location('bottom_news_b_analysis',CODE/'analyze_B1.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class BottomNewsResearchTests(unittest.TestCase):
 def test_committed_evidence_still_matches_manifest(self):
  self.assertTrue(module.verify_freeze())

 def test_date_only_friday_waits_until_monday_open(self):
  dates=pd.bdate_range('2024-04-18','2024-04-24')
  i=module.entry_index(dates,'2024-04-19')
  self.assertEqual(str(dates[i].date()),'2024-04-22')
  self.assertEqual(module.entry_index(dates,'2024-04-22','2024-04-22T08:00:00-04:00'),2)

 def test_unobserved_horizon_is_not_a_success(self):
  dates=pd.bdate_range('2026-01-01',periods=30)
  d=pd.DataFrame({'Open':100.,'Close':100.},index=dates)
  z=module.features(d)
  self.assertEqual(int(z['near'].count()),11)
  self.assertTrue(np.isnan(z.iloc[-1]['near']))
  self.assertEqual(z.iloc[10]['near'],1.)

 def test_decline_state_uses_only_previous_closes(self):
  dates=pd.bdate_range('2026-01-01',periods=70)
  d=pd.DataFrame({'Open':100.,'Close':100.},index=dates)
  d.iloc[64,d.columns.get_loc('Close')]=80.
  z=module.features(d)
  self.assertFalse(z.iloc[64]['state'])
  self.assertTrue(z.iloc[65]['state'])

 def test_cluster_requires_two_different_houses(self):
  dates=pd.bdate_range('2026-01-01',periods=30)
  z=pd.DataFrame(index=dates)
  calls=[dict(house='A',action=a,published_date='2026-01-05',published_at='') for a in ['downgrade','target_cut']]
  self.assertEqual(module.cluster_indices(z,calls),[])
  calls.append(dict(house='B',action='downgrade',published_date='2026-01-06',published_at=''))
  self.assertEqual(module.cluster_indices(z,calls)[0],dates.get_loc('2026-01-07'))

 def test_actual_split_is_recorded_and_adjusted(self):
  check=json.loads((NEWS/'v2/results/split_check.json').read_text(encoding='utf-8'))
  self.assertEqual(check['reported_split_20261105'],3.)
  self.assertTrue(check['pass_continuity'])

if __name__=='__main__':unittest.main()
