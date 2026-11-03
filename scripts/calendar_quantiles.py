"""Recheck probabilistic correction after the common fitting boundary.

This follow-up was added after inspecting the point comparison. The quantile
grid and calibration window remain unchanged, and model structure/iterations
come from common-calendar point tuning, with no quantile test selection.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import lightgbm as lgb
from calendar_validation import ROOT, prepare, expand_split
from emsx_probabilistic import QUANTILES as Q, QCOLS, score, SCORING_VERSION

def calibrate(cal,test):
    history=pd.concat([cal,test],ignore_index=True).sort_values(['site_id','issue_time','delivery_step']).reset_index(drop=True)
    result=test.copy()
    for c in QCOLS:result['cal_'+c]=np.nan
    # At midnight, all deliveries of a previous daily issue are matured.
    # Therefore the available origins are [issue-30 days, issue), for every h.
    for site,block in history.groupby('site_id',observed=True,sort=False):
        dates=pd.DatetimeIndex(block.issue_time.drop_duplicates().sort_values())
        assert len(block)==len(dates)*96
        for tau,c in zip(Q,QCOLS):
            error=((block.actual_pv_kwh-block[c])/block.scale_kwh).to_numpy().reshape(-1,96)
            matrix=pd.DataFrame(error,index=dates)
            offsets=matrix.rolling('30D',closed='left',min_periods=1).quantile(tau,interpolation='linear')
            if offsets.isna().any().any():
                fallback=matrix.expanding(min_periods=1).quantile(tau,interpolation='linear').shift(1)
                offsets=offsets.fillna(fallback).fillna(0)
            target=result[result.site_id==site].sort_values(['issue_time','delivery_step'])
            values=offsets.reindex(pd.DatetimeIndex(target.issue_time.drop_duplicates().sort_values())).to_numpy().ravel()
            assert len(values)==len(target)
            result.loc[target.index,'cal_'+c]=np.maximum(target[c].to_numpy()+values*target.scale_kwh.to_numpy(),0)
    result[['cal_'+c for c in QCOLS]]=np.sort(result[['cal_'+c for c in QCOLS]].to_numpy(),axis=1)
    return result

def main():
    w,meta=prepare()
    train,features=expand_split(w,'train');tuning,_=expand_split(w,'tuning')
    fit=pd.concat([train,tuning],ignore_index=True);fit.site_id=fit.site_id.astype('category')
    selected=json.loads((ROOT/'results/calendar_lightgbm.json').read_text())['selected']
    def one(tau):
        model=lgb.LGBMRegressor(objective='quantile',alpha=float(tau),n_estimators=selected['iterations'],learning_rate=selected['learning_rate'],num_leaves=selected['num_leaves'],min_child_samples=selected['min_child_samples'],colsample_bytree=.9,subsample=.85,reg_lambda=2.,reg_alpha=.05,n_jobs=3,verbosity=-1,random_state=20261101)
        model.fit(fit[features],fit.residual_norm,categorical_feature=['site_id'])
        print('quantile fitted',tau,flush=True)
        return model
    with ThreadPoolExecutor(max_workers=3) as pool:models=list(pool.map(one,Q))
    def predict(frame):
        out=frame[['site_id','issue_time','valid_time','delivery_step','actual_pv_kwh','vendor_prediction_kwh','scale_kwh']].copy()
        for model,c in zip(models,QCOLS):
            out[c]=np.maximum(frame.vendor_prediction_kwh.to_numpy()+model.predict(frame[features])*frame.scale_kwh.to_numpy(),0)
        out[QCOLS]=np.sort(out[QCOLS].to_numpy(),axis=1)
        return out
    cal,_=expand_split(w,'calibration');test,_=expand_split(w,'test')
    cp=predict(cal);tp=predict(test)
    del train,tuning,fit,cal,test
    adjusted=calibrate(cp,tp)
    # Independently check the fast daily-window implementation at the first
    # test issue of each system against the general timestamp implementation.
    from check_calibration import offsets
    hist=pd.concat([cp,tp],ignore_index=True)
    for c in QCOLS:hist['error_'+c]=(hist.actual_pv_kwh-hist[c])/hist.scale_kwh
    checked=0
    for site,block in adjusted.groupby('site_id',observed=True):
        day=block[block.issue_time==block.issue_time.min()]
        reference=np.sort(np.maximum(day[QCOLS].to_numpy()+offsets(hist[hist.site_id==site],day.issue_time.iloc[0],day.delivery_step.to_numpy())*day.scale_kwh.to_numpy()[:,None],0),axis=1)
        assert np.allclose(reference,day[['cal_'+c for c in QCOLS]].to_numpy(),rtol=1e-10,atol=1e-8)
        checked+=1
    adjusted.to_parquet(ROOT/'results/calendar_quantile_test.parquet',index=False)
    cp.to_parquet(ROOT/'results/calendar_quantile_calibration.parquet',index=False)
    report={'sample':meta,'selected_from_point_tuning':selected,'window_days':30,'quantiles':Q.tolist(),'scoring_version':SCORING_VERSION,'created_utc':datetime.now(timezone.utc).isoformat(),'reference_profiles_checked':checked,'calibration_raw':score(cp),'test_raw':score(adjusted),'test_rolling':score(adjusted,'cal_')}
    (ROOT/'results/calendar_quantiles.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
