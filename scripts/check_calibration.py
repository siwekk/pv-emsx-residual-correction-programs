"""Check stored calibration against the documented causal empirical offsets.

Checks a deterministic spread of 70 test profiles, including the first issue
of every system. With no matured history, the reference keeps raw quantiles.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from emsx_round1_paths import ROOT, PREDICTION_ROOT
from emsx_probabilistic import QUANTILES as Q, QCOLS

def offsets(history,issue,steps):
    available=history[(history.valid_time<=issue)&(history.valid_time>issue-pd.Timedelta(days=30))]
    if available.empty:
        available=history[history.valid_time<=issue]
    if available.empty:
        return np.zeros((len(steps),len(Q)))
    result=[]
    for tau,c in zip(Q,QCOLS):
        by_step=available.groupby('delivery_step',observed=True)['error_'+c].quantile(tau,interpolation='linear')
        values=pd.Series(steps).map(by_step).to_numpy(float)
        values[~np.isfinite(values)]=available['error_'+c].quantile(tau,interpolation='linear')
        result.append(values)
    return np.column_stack(result)

def main():
    (ROOT/'results').mkdir(parents=True,exist_ok=True)
    test=pd.read_parquet(PREDICTION_ROOT/'emsx_daily_quantile_lightgbm.parquet')
    cal=pd.read_parquet(PREDICTION_ROOT/'emsx_daily_quantile_lightgbm_calibration.parquet')
    history=pd.concat([cal,test],ignore_index=True)
    for c in QCOLS:
        history['error_'+c]=(history.actual_pv_kwh-history[c])/history.scale_kwh
    records=[]
    for site,block in test.groupby('site_id',observed=True):
        day=block[block.issue_time==block.issue_time.min()]
        adjusted=np.maximum(day[QCOLS].to_numpy(float)+offsets(history[history.site_id==site],day.issue_time.iloc[0],day.delivery_step.to_numpy())*day.scale_kwh.to_numpy(float)[:,None],0)
        adjusted=np.sort(adjusted,axis=1)
        stored=day[['cal_'+c for c in QCOLS]].to_numpy(float)
        difference=float(np.max(np.abs(adjusted-stored)))
        assert np.allclose(adjusted,stored,rtol=1e-10,atol=1e-8)
        records.append({'site_id':int(site),'issue_time':str(day.issue_time.iloc[0]),'max_absolute_difference_kwh':difference})
    (ROOT/'results/calibration_reference_check.json').write_text(json.dumps({'profiles_checked':len(records),'delivery_rows_checked':96*len(records),'maximum_difference_kwh':max(r['max_absolute_difference_kwh'] for r in records),'checks':records},indent=2)+'\n')
    print('Calibration reference checked',len(records),'profiles',flush=True)

if __name__=='__main__':main()
