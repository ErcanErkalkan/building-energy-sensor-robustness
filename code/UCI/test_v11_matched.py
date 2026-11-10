import unittest
from pathlib import Path
import numpy as np, pandas as pd
from v11_matched_exposure import two_masks,N_BLOCKS,BLOCK_LEN,SENSORS,SEEDS
P=Path(__file__).resolve().parent/'outputs'

class TestV11(unittest.TestCase):
    def test_rows(self):self.assertEqual(len(pd.read_csv(P/'V11_matched_exposure_all_runs.csv')),60)
    def test_pairs(self):self.assertEqual(len(pd.read_csv(P/'V11_paired_topology_contrasts.csv')),6)
    def test_five_seeds(self):self.assertEqual(set(pd.read_csv(P/'V11_exact_mask_audit.csv').seed),set(SEEDS))
    def test_identical_exposure(self):
        for seed in SEEDS:
            a,b,_=two_masks(3947,seed);self.assertTrue(np.array_equal(a.any(axis=1),b.any(axis=1)))
    def test_row_margins(self):
        for seed in SEEDS:
            a,b,_=two_masks(3947,seed);self.assertTrue(np.array_equal(a.sum(axis=1),b.sum(axis=1)))
    def test_channel_margins(self):
        for seed in SEEDS:
            a,b,_=two_masks(3947,seed);self.assertTrue(np.array_equal(a.sum(axis=0),b.sum(axis=0)))
    def test_per_channel_loss(self):
        for seed in SEEDS:
            a,b,_=two_masks(3947,seed);self.assertTrue(np.all(a.sum(axis=0)==396))
    def test_budget(self):
        for seed in SEEDS:
            a,b,_=two_masks(3947,seed);self.assertEqual(int(a.sum()),7128);self.assertEqual(int(b.sum()),7128)
    def test_block_nonadjacency(self):
        for seed in SEEDS:
            a,b,slots=two_masks(3947,seed);self.assertEqual(len(slots),33);self.assertTrue(np.all(np.diff(slots)>1))
    def test_exact_rows(self):
        for seed in SEEDS:
            a,b,_=two_masks(3947,seed);self.assertEqual(int(a.any(axis=1).sum()),1188);self.assertEqual(int(b.any(axis=1).sum()),1188)
    def test_zero_excluded_never(self):
        c=pd.read_csv(P/'V11_paired_topology_contrasts.csv');self.assertTrue(((c.CI95_low_Wh<=0)&(c.CI95_high_Wh>=0)).all())
    def test_mean_difference(self):
        rows=pd.read_csv(P/'V11_matched_exposure_all_runs.csv');c=pd.read_csv(P/'V11_paired_topology_contrasts.csv')
        for row in c.itertuples():
            t=rows[(rows.model==row.model)&(rows.imputation==row.imputation)]
            means=t.groupby('design').MAE_all_Wh.mean()
            diff=means['coherent_six_channel_block']-means['rotating_six_channel_group']
            self.assertAlmostEqual(diff,row.coherent_minus_rotating_MAE_Wh,places=9)
    def test_unaffected_rows_same(self):
        rows=pd.read_csv(P/'V11_matched_exposure_all_runs.csv')
        for key,t in rows.groupby(['seed','model','imputation']):
            self.assertAlmostEqual(t.MAE_unaffected_Wh.iloc[0],t.MAE_unaffected_Wh.iloc[1],places=9)
    def test_caveat_in_manuscript(self):
        s=(Path(__file__).resolve().parent/'manuscript.tex').read_text()
        self.assertIn('repeatedly consulted',s);self.assertIn('not an all-18-channel common outage',s)

if __name__=='__main__':unittest.main()
