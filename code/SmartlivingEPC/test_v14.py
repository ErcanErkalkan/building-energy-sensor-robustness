import unittest,zipfile,hashlib,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import external_validation_v14 as e

class CheckV14(unittest.TestCase):
 def test_masks_identical_time_and_column_marginals(self):
  idx=pd.date_range('2025-01-01',periods=300,freq='h')
  data=pd.DataFrame(1.,index=idx,columns=list('abcd'))
  c,r,b,candidates=e.make_masks(data,list('abcd'),101)
  self.assertEqual(int(c.sum()),int(r.sum()))
  self.assertTrue(np.array_equal(c.sum(axis=0),r.sum(axis=0)))
  self.assertTrue(np.array_equal(c.any(axis=1),r.any(axis=1)))
  self.assertTrue((c.sum(axis=1)[c.any(axis=1)]==2).all())
  self.assertTrue(all(len(x)==6 for x in b))
  self.assertTrue(len(b)>2)
 def test_reject_nonconsecutive_or_negative_meter(self):
  # Formula applied to real data: only consecutive one-hour intervals with nonnegative differences.
  idx=pd.to_datetime(['2025-01-01 00:00','2025-01-01 01:00','2025-01-01 03:00','2025-01-01 04:00','2025-01-01 05:00'])
  v=pd.Series([10.,11.,12.,11.5,13],index=idx)
  d=v.diff();dt=v.index.to_series().diff().dt.total_seconds()/3600
  keep=(dt==1)&(d>=0)
  self.assertEqual(keep.tolist(),[False,True,False,False,True])
 def test_no_rows_without_all_four_readings_masked(self):
  idx=pd.date_range('2025-01-01',periods=300,freq='h')
  d=pd.DataFrame(1.,index=idx,columns=list('abcd'))
  d.iloc[:10,0]=np.nan
  c,r,_,_=e.make_masks(d,list('abcd'),101)
  self.assertFalse(c[:10].any() or r[:10].any())
 def test_all_archives_exact_zenodo_md5(self):
  expect={'City hall.zip':'c0d45f81dca10ce46bd56e4d685961b3','Mixed-use building.zip':'d2a295be815a43c909a48c4d1260e4a0','Private flat.zip':'fc8bb61652ce3bba817ffc816b05905e','Public lighting.zip':'bc5cd43c05a0293f5f31b7b2e022a751','School.zip':'a827e6674f6911ed44ca95448a9679d3','Single Family House.zip':'05a336f75d825889dded6a2d287a75b2','Sports Center.zip':'e3936c1094cffd1b21812940bff89858'}
  for fn,md5 in expect.items():
   path=ROOT/'data_external'/fn
   self.assertTrue(path.exists())
   self.assertEqual(hashlib.md5(path.read_bytes()).hexdigest(),md5)
 def test_outputs_experiment_counts(self):
  out=ROOT/'results'
  self.assertEqual(len(pd.read_csv(out/'V14_clean_models.csv')),40)
  self.assertEqual(len(pd.read_csv(out/'V14_all_fault_runs.csv')),96)
  self.assertEqual(len(pd.read_csv(out/'V14_paired_contrasts.csv')),48)
  self.assertEqual(len(pd.read_csv(out/'V14_mask_integrity.csv')),12)
 def test_all_site_split_chronology(self):
  d=pd.read_csv(ROOT/'results'/'V14_site_quality.csv')
  self.assertEqual(len(d),4)
  for _,r in d.iterrows():
   self.assertLess(r.train_end,r.validation_start)
   self.assertLess(r.validation_end,r.test_start)
   self.assertLess(r.test_complete_hours,r.test_hours_total)
 def test_source_dataset_mismatches_explicit(self):
  import json
  p=json.loads((ROOT/'results'/'V14_protocol.json').read_text())
  self.assertTrue('not model transfer' in p['research_status'])
  self.assertTrue('1-h' in p['target'])
if __name__=='__main__':unittest.main(verbosity=2)
