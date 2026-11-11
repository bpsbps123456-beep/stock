"""B2 freeze, window completeness and denominator integrity."""
import csv,importlib.util,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NEWS=ROOT/'data/derived/analysis/bottom_zone_20261111/news'
CODE=NEWS/'v2/code';sys.path.insert(0,str(CODE))
spec=importlib.util.spec_from_file_location('bottom_b2',CODE/'analyze_B2.py')
b2=importlib.util.module_from_spec(spec);spec.loader.exec_module(b2)

def read_rows(path):
 with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

class BottomNewsWindowTests(unittest.TestCase):
 def test_b2_frozen_and_committed(self):self.assertTrue(b2.verify_freeze())

 def test_observable_window_has_wire_close(self):
  rows=read_rows(NEWS/'v2/B2/wire_coverage_B2.csv')
  self.assertEqual(len(rows),187)
  self.assertEqual(sum(r['status']=='future_unobservable' for r in rows),2)
  self.assertTrue(all(int(r['wire_close_count'])>0 for r in rows if r['status']!='future_unobservable'))

 def test_windows_are_not_confused_with_prediction_counts(self):
  rows=read_rows(NEWS/'bottom_news.csv')
  self.assertEqual(len({r['anchor_date'] for r in rows if r['anchor_type']=='real'}),10)
  self.assertEqual(len({r['anchor_date'] for r in rows if r['anchor_type']=='fake'}),6)
  self.assertTrue(all(-5<=int(r['k'])<=5 for r in rows))
  self.assertEqual(len(rows),len({(r['anchor_date'],r['source_group']) for r in rows}))

 def test_catalyst_requires_positive_material_on_zero_or_one(self):
  rows=read_rows(NEWS/'bottom_news.csv')
  for r in rows:
   if r['is_turn_catalyst']=='Y':
    self.assertIn(int(r['k']),[0,1]);self.assertEqual(r['tone'],'pos')
    self.assertNotIn(r['type'],['technical','other','capitulation_language'])

 def test_current_episode_excluded_from_comparison(self):
  d=json.loads((NEWS/'v2/results/analysis_B2.json').read_text(encoding='utf-8'))
  self.assertTrue(all(r['anchor_type'] in ['real','fake'] for r in d['grid']))
  for r in d['grid']:self.assertEqual(r['n'],r['neg']+r['pos']+r['neutral'])
  self.assertEqual(len(d['current10']),10)

if __name__=='__main__':unittest.main()
