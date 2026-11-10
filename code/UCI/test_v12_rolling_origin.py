from pathlib import Path
import unittest,json,numpy as np,pandas as pd
from v12_rolling_origin import masks,FOLDS,CANDIDATES
P=Path(__file__).parent/'outputs'
class V12Checks(unittest.TestCase):
 def test_rolling_boundaries(self):
  for tr,va,ev in FOLDS:
   self.assertLess(tr,va);self.assertLess(va,ev);self.assertLessEqual(ev,.8)
 def test_mask_matched_by_row_and_sensor(self):
  for n in (1973,1974):
   for seed in [101,102,103]:
    a,b=masks(n,seed)
    self.assertTrue(np.array_equal(a.any(axis=1),b.any(axis=1)))
    self.assertTrue(np.array_equal(a.sum(0),b.sum(0)))
    self.assertTrue(np.array_equal(a.sum(1),b.sum(1)))
    self.assertEqual(int(a.sum()),3240)
    self.assertEqual(int(a.any(1).sum()),540)
 def test_result_tables(self):
  scores=pd.read_csv(P/'V12_rolling_origin_clean_controls.csv')
  c=pd.read_csv(P/'V12_matched_topology_all_seeds.csv')
  ca=pd.read_csv(P/'V12_validation_candidate_audit.csv')
  m=pd.read_csv(P/'V12_mask_checks.csv')
  self.assertEqual((len(scores),len(c),len(ca),len(m)),(20,48,32,12))
  self.assertFalse(scores.isna().any().any());self.assertFalse(c.isna().any().any())
  self.assertTrue((m.affected_timestamps==540).all());self.assertTrue((m.missing_per_sensor==180).all())
  self.assertTrue(m.equal_sensor_margins.all());self.assertTrue(m.equal_timestamp_exposure.all())
  for fld in range(1,5):
   for name in ('Ridge','HistGradientBoosting'):
    q=scores[(scores.fold==fld)&(scores.model==name)]
    self.assertEqual(len(q),2)
    self.assertLess(float(q[q.features=='weather_calendar'].eval_MAE_Wh.iloc[0]),float(q[q.features.str.startswith('full')].eval_MAE_Wh.iloc[0]))
 def test_no_final_test_reuse(self):
  d=json.loads((P/'V12_protocol.json').read_text())
  self.assertEqual(d['n_topology_run_rows'],48)
  self.assertEqual(d['n_baseline_rows'],20)
  self.assertIn('not independent',d['status'])
if __name__=='__main__':unittest.main(verbosity=2)
