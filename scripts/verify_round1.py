"""Check bundled aggregates, optionally recomputing every new result from predictions.

Training is not required. With --prediction-root, provide the external directory
containing calendar_lightgbm_*.parquet, calendar_quantile_*.parquet, and all twelve
calendar neural .npz files. Inputs are never changed.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from emsx_paths import CODE_ROOT
from emsx_probabilistic import SCORING_VERSION, score

KEYS=['site_id','issue_time','valid_time','delivery_step']

def close(actual,expected,label,atol=.001):
    if not np.isclose(actual,expected,rtol=1e-8,atol=atol):
        raise AssertionError(f'{label}: calculated {actual}, expected {expected}')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction-root',type=Path)
    parser.add_argument('--expected-root',type=Path,default=CODE_ROOT/'results/round1')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    expected=args.expected_root
    tree=json.loads((expected/'calendar_lightgbm.json').read_text())
    neural=json.loads((expected/'calendar_neural.json').read_text())
    quant=json.loads((expected/'calendar_quantiles.json').read_text())
    protocol=json.loads((CODE_ROOT/'protocols/emsx_round1_calendar_20261101.json').read_text())
    expected_runs={(m,s) for m in ['bilstm','transformer'] for s in protocol['neural_seeds']}
    actual_runs={(r['model'],r['seed']) for r in neural['runs']}
    assert len(neural['runs'])==6 and actual_runs==expected_runs
    assert tree['sample']['profiles']=={'train':8047,'tuning':4053,'calibration':3947,'test':14509}
    assert len(tree['sample']['systems'])==47
    assert pd.Timestamp(tree['sample']['latest_fit_delivery'])<=pd.Timestamp('2016-04-01',tz='UTC')
    for key in ['calibration_raw','test_raw','test_rolling']:
        m=quant[key]
        close(m['wis_kwh'],(11*m['mean_pinball_kwh']-.05*m['interval_score_90_kwh'])/4.5,key,1e-8)
    checked=[]
    if args.prediction_root:
        from calendar_validation import metric
        for split in ['calibration','test']:
            p=args.prediction_root
            reference=pd.read_parquet(p/f'calendar_lightgbm_{split}.parquet').sort_values(KEYS).reset_index(drop=True)
            assert not reference.duplicated(KEYS).any()
            assert len(reference)==96*tree['sample']['profiles'][split]
            for name,column in [('lightgbm','prediction_kwh'),('vendor','vendor_prediction_kwh')]:
                got=metric(reference.actual_pv_kwh.to_numpy(),reference[column].to_numpy(),reference.site_id,reference.scale_kwh.to_numpy())
                for key in ['mae_kwh','system_averaged_mae_kwh','normalized_mae']:
                    close(got[key],tree[split][name][key],f'{split}/{name}/{key}')
            q=pd.read_parquet(p/f'calendar_quantile_{split}.parquet').sort_values(KEYS).reset_index(drop=True)
            assert reference[KEYS].equals(q[KEYS])
            assert np.array_equal(reference.actual_pv_kwh.to_numpy(),q.actual_pv_kwh.to_numpy())
            for prefix,key in [('', 'calibration_raw' if split=='calibration' else 'test_raw')]+([('cal_','test_rolling')] if split=='test' else []):
                got=score(q,prefix)
                for name,value in got.items():close(value,quant[key][name],f'{key}/{name}',1e-6)
            actual=reference.actual_pv_kwh.to_numpy().reshape(-1,96)
            site=reference.site_id.to_numpy().reshape(-1,96)[:,0]
            scale=reference.scale_kwh.to_numpy().reshape(-1,96)
            for run in neural['runs']:
                path=p/f"calendar_{run['model']}_{run['seed']}_{split}.npz"
                with np.load(path,allow_pickle=False) as data:
                    assert np.array_equal(data['site'],site)
                    prediction=data['prediction']
                    assert prediction.shape==actual.shape and np.isfinite(prediction).all()
                    got=metric(actual.ravel(),prediction.ravel(),np.repeat(site,96),scale.ravel())
                for name in ['mae_kwh','system_averaged_mae_kwh','normalized_mae']:
                    close(got[name],run[split][name],f"{run['model']}/{run['seed']}/{split}/{name}")
            checked.append(split)
    report={'passed':True,'scoring_version':SCORING_VERSION,'neural_runs':6,'systems':47,'test_deliveries':1392864,'prediction_partitions_checked':checked,'mode':'full_saved_prediction_check' if checked else 'aggregate_consistency_only','point_tolerance_kwh':.001}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
