import importlib
from pathlib import Path
import sys
import tempfile
from helpers import temporary_directory
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from emsx_probabilistic import QUANTILES, QCOLS, score, wis_rows, normalize_saved_metrics
from migrate_legacy_wis_results import migrate_dict

class ScoringTests(unittest.TestCase):
    def test_analytical_interval_weights(self):
        # y=0, q=0..10: weighted interval terms are 1.8,3.2,4.2,4.8;
        # the weighted median term is 2.5, giving 16.5/4.5 = 11/3.
        self.assertAlmostEqual(float(wis_rows(np.array([0.]),np.arange(11.)[None,:])[0]),11/3)

    def test_independent_pinball_identity(self):
        rng=np.random.default_rng(20261101)
        q=np.sort(rng.normal(size=(127,11)),axis=1);y=rng.normal(size=127)
        e=y[:,None]-q
        expected=np.maximum(QUANTILES*e,(QUANTILES-1)*e)[:,1:10].sum(axis=1)/4.5
        np.testing.assert_allclose(wis_rows(y,q),expected,rtol=1e-13)

    def test_crossings_rejected(self):
        with self.assertRaises(ValueError):wis_rows(np.array([0.]),np.arange(11.)[::-1][None,:])

    def test_unrecoverable_summary_is_withdrawn_without_losing_provenance(self):
        data={'weighted_interval_score_kwh':{'mean':70.,'standard_deviation':1.}}
        changes=[];migrate_dict(data,[],changes)
        self.assertIsNone(data['weighted_interval_score_kwh'])
        self.assertEqual(data['legacy_invalid_weighted_interval_score_kwh']['mean'],70.)
        self.assertEqual(changes[0]['status'],'withdrawn_full_predictions_required')

    def test_migration_is_idempotent(self):
        data={'wis_kwh':48.910718623827385,'mean_pinball_kwh':19.57912600904663,'interval_score_90_kwh':389.8602738464132}
        changes=[];migrate_dict(data,[],changes);saved=dict(data)
        migrate_dict(data,[],changes)
        self.assertEqual(data,saved)
        self.assertAlmostEqual(data['wis_kwh'],43.52830497937603)

    def test_saved_metrics_recovery_does_not_modify_original(self):
        original={'mean_pinball_kwh':19.57912600904663,'central_90_interval_score_kwh':389.8602738464132,'weighted_interval_score_kwh':48.910718623827385}
        fixed=normalize_saved_metrics(original)
        self.assertAlmostEqual(fixed['weighted_interval_score_kwh'],43.52830497937603)
        self.assertEqual(original['weighted_interval_score_kwh'],48.910718623827385)
        with self.assertRaises(ValueError):normalize_saved_metrics({'weighted_interval_score_kwh':48.91})

    def test_every_legacy_scoring_entry_point_uses_correct_weights(self):
        f=pd.DataFrame({c:[float(i)] for i,c in enumerate(QCOLS)})
        f['actual_pv_kwh']=0.;f['scale_kwh']=1.;f['site_id']=1
        f['issue_time']=pd.Timestamp('2016-07-01',tz='UTC')
        f['valid_time']=f.issue_time+pd.Timedelta(minutes=15);f['delivery_step']=1
        for module,func,key in [('run_emsx_daily_quantile_lightgbm','metrics','wis_kwh'),('analyse_emsx_daily_manuscript','probabilistic_metrics','wis_kwh'),('emsx_daily_fewshot_common','score_predictions','weighted_interval_score_kwh')]:
            with self.subTest(module=module):
                value=getattr(importlib.import_module(module),func)(f)[key]
                self.assertAlmostEqual(value,11/3)
        with temporary_directory() as tmp:
            p=Path(tmp)/'pred.parquet';f.to_parquet(p,index=False)
            _,result=importlib.import_module('evaluate_emsx_daily_probabilistic').evaluate(p)
            self.assertAlmostEqual(result['weighted_interval_score_kwh'],11/3)

if __name__=='__main__':unittest.main()
