"""Recompute review checks from preserved prediction files, never modify inputs."""
from pathlib import Path
import hashlib
import json
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from emsx_round1_paths import ROOT, CURVE_ROOT, PREDICTION_ROOT
from emsx_probabilistic import QUANTILES as Q, QCOLS, score, SCORING_VERSION
OUT = ROOT / 'results'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paths=sorted(CURVE_ROOT.glob('site=*/curves.parquet'))
    if not paths:
        raise FileNotFoundError(f'No prepared profiles under {CURVE_ROOT}')
    wide=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True)
    wide.issue_time=pd.to_datetime(wide.issue_time,utc=True)
    complete=wide[wide.complete_case].copy()
    dates=complete.groupby(['site_id','split']).issue_time.agg(['min','max','size']).reset_index()
    dates.to_csv(OUT/'partition_dates_by_system.csv',index=False)
    fit=complete[complete.split.isin(['train','tuning'])]
    fit_end=fit.issue_time.max()+pd.Timedelta(days=1)
    rows=[]
    for split,block in complete.groupby('split'):
        rows.append({'partition':split,'first_issue':str(block.issue_time.min()),'last_issue':str(block.issue_time.max()),'profiles':len(block),'delivery_rows':96*len(block),'profiles_before_global_fit_end':int((block.issue_time<fit_end).sum())})
    overlap={'latest_fit_target':str(fit_end),'partition_ranges':rows,
             'affected_test_systems':int(complete.loc[(complete.split=='test')&(complete.issue_time<fit_end),'site_id'].nunique()),
             'raw_eligible_profiles':len(wide),'complete_profiles':len(complete)}
    print(json.dumps(overlap,indent=2),flush=True)
    metrics={}
    for label,name,prefix in [('raw','emsx_daily_quantile_lightgbm',''),('calibrated','emsx_daily_quantile_lightgbm','cal_'),('calibration_quantile_median','emsx_daily_quantile_lightgbm_calibration',''),('chronos_vendor','emsx_daily_chronos2_vendor_covariate',''),('chronos_target','emsx_daily_chronos2_target_only',''),('moirai_target','emsx_daily_moirai2_target_only',''),('moirai_vendor','emsx_daily_moirai2_vendor_covariate','')]:
        frame=pd.read_parquet(PREDICTION_ROOT/f'{name}.parquet')
        frame[QCOLS]=np.sort(frame[QCOLS].to_numpy(float),axis=1)
        metrics[label]=score(frame,prefix)
        print(label,metrics[label],flush=True)
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'command':sys.argv,'scoring_version':SCORING_VERSION,'chronology':overlap,'scores':metrics,
            'wis_contract':'K=4; alpha=0.2,0.4,0.6,0.8; bounds q(alpha/2),q(1-alpha/2); weights alpha/2; median weight 0.5; divisor 4.5; pooled equally over delivery rows.',
            'source_prediction_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in PREDICTION_ROOT.glob('emsx_daily_*.parquet')}}
    (OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    main()
