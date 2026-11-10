"""Checks for V13 post-hoc grouping sensitivity and V12 numerical parity."""
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import v13_grouping_sensitivity as v13
import v12_rolling_origin as v12
ROOT=Path(__file__).resolve().parent
class GroupAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=pd.read_csv(ROOT/'outputs/V13_all_groupings.csv')
        cls.mask=pd.read_csv(ROOT/'outputs/V13_mask_audit.csv')
    def test_result_count(self): self.assertEqual(len(self.data),480)
    def test_mask_count(self): self.assertEqual(len(self.mask),120)
    def test_expected_number_of_folds(self): self.assertEqual(set(self.data.fold),{1,2,3,4})
    def test_expected_seeds(self): self.assertEqual(set(self.data.seed),{101,102,103})
    def test_grouping_set(self): self.assertEqual(set(self.data.partition),set(range(10)))
    def test_two_models(self): self.assertEqual(set(self.data.model),{'Ridge','HistGradientBoosting'})
    def test_two_imputations(self): self.assertEqual(set(self.data.imputation),{'median','causal_ffill'})
    def test_exact_missing_counts(self): self.assertTrue((self.mask.missing_cells==3240).all())
    def test_exact_row_exposure(self): self.assertTrue((self.mask.affected_rows==540).all())
    def test_exact_sensor_margins(self): self.assertTrue((self.mask.per_channel_missing==180).all())
    def test_channel_orders_permutations(self):
        for x in self.mask.channel_order:
            p=[int(v) for v in x.split(';')]
            self.assertEqual(sorted(p),list(range(18)))
    def test_affected_metric_consistency(self):
        for x in self.data.itertuples():
            self.assertAlmostEqual(x.delta_affected_MAE_Wh*x.affected_rows/x.n_eval,x.delta_MAE_Wh,places=9)
    def test_prior_v12_identity_strict(self):
        prior=pd.read_csv(ROOT/'outputs/V12_matched_topology_all_seeds.csv')
        identity=self.data[self.data.partition==0]
        x=prior.merge(identity,on=['fold','seed','model','imputation'],validate='one_to_one')
        self.assertEqual(len(x),48)
        np.testing.assert_allclose(x['diff_MAE_Wh'],x['delta_MAE_Wh'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(x['coherent_MAE_Wh_x'],x['coherent_MAE_Wh_y'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(x['rotating_MAE_Wh_x'],x['rotating_MAE_Wh_y'],rtol=0,atol=1e-12)
    def test_mask_transform_pair(self):
        for seed in (101,102,103):
            c,r=v12.masks(1973,seed)
            for partition in range(10):
                cc,p=v13.randomize_grouping(c,seed,partition)
                rr,pr=v13.randomize_grouping(r,seed,partition)
                np.testing.assert_array_equal(p,pr)
                np.testing.assert_array_equal(cc.any(1),rr.any(1))
                np.testing.assert_array_equal(cc.sum(1),rr.sum(1))
                np.testing.assert_array_equal(cc.sum(0),rr.sum(0))
                self.assertEqual(int(cc.sum()),3240)
    def test_source_is_not_final_holdout(self):
        size=19735
        for tri in v12.FOLDS:
            self.assertLessEqual(int(size*tri[-1]),int(size*.8))
    def test_summary_mean(self):
        reported=pd.read_csv(ROOT/'outputs/V13_grouping_summary.csv')
        for row in reported.itertuples():
            x=self.data[(self.data.model==row.model)&(self.data.imputation==row.imputation)]
            self.assertAlmostEqual(x.delta_MAE_Wh.mean(),row.mean_delta,places=12)
if __name__=='__main__':unittest.main(verbosity=2)
