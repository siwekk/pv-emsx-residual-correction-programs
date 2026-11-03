"""Calendar validation with newly selected settings and matched neural baselines.

Output goes to the external EMSX_ROUND1_ROOT directory.
Run --model lightgbm and --model neural independently after the protocol is saved.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from emsx_round1_paths import ROOT, CURVE_ROOT
from emsx_daily_curve_common import load_wide, expand_split, TARGET_COLUMNS

def prepare():
    w=load_wide(CURVE_ROOT)
    a,b,c=[pd.Timestamp(x,tz='UTC') for x in ['2016-01-01','2016-04-01','2016-07-01']]
    w['split']=np.select([w.issue_time<a,w.issue_time<b,w.issue_time<c],['train','tuning','calibration'],default='test')
    counts=w[w.complete_case].groupby(['site_id','split']).size().unstack(fill_value=0).reindex(columns=['train','tuning','calibration','test'],fill_value=0)
    ids=counts.index[(counts.train>=30)&(counts.tuning>=7)&(counts.calibration>=7)&(counts.test>=7)]
    if not len(ids):
        raise ValueError('No system meets the common calendar sample requirements')
    w=w[w.site_id.isin(ids)].copy()
    for site,block in w.groupby('site_id'):
        scale=max(float(np.quantile(block.loc[block.split=='train',TARGET_COLUMNS].to_numpy(float),.995)),.001)
        w.loc[block.index,'scale_kwh']=scale
    metadata={'calendar_cutoffs_utc':[str(x) for x in [a,b,c]],'systems':[int(x) for x in ids],
              'profiles':w[w.complete_case].groupby('split').size().to_dict(),
              'selection':'At least 30 complete training profiles and 7 in each later partition; no outcome score used for eligibility.',
              'scale':'Per-system 99.5th target percentile in common-calendar training only.',
              'latest_fit_delivery':str(w.loc[w.complete_case&w.split.isin(['train','tuning']),'issue_time'].max()+pd.Timedelta(days=1))}
    (ROOT/'results').mkdir(parents=True,exist_ok=True)
    (ROOT/'results/calendar_sample.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return w,metadata

def metric(y,p,sites,scale):
    error=np.abs(y-p)
    return {'mae_kwh':float(error.mean()),'system_averaged_mae_kwh':float(pd.Series(error).groupby(np.asarray(sites)).mean().mean()),'normalized_mae':float((error/scale).mean()),'rows':len(error)}

def lightgbm(w,meta):
    import lightgbm as lgb
    train,features=expand_split(w,'train')
    tuning,_=expand_split(w,'tuning')
    trials=[]
    common=dict(objective='regression_l1',n_estimators=1000,colsample_bytree=.9,subsample=.85,reg_lambda=2.,reg_alpha=.05,n_jobs=6,verbosity=-1,random_state=20261101)
    for leaves,minimum,lr in [(63,300,.04),(127,200,.03),(255,100,.02)]:
        m=lgb.LGBMRegressor(**common,num_leaves=leaves,min_child_samples=minimum,learning_rate=lr)
        m.fit(train[features],train.residual_norm,categorical_feature=['site_id'],eval_set=[(tuning[features],tuning.residual_norm)],callbacks=[lgb.early_stopping(100,verbose=False)])
        pred=np.maximum(tuning.forecast_norm.to_numpy()+m.predict(tuning[features]),0)*tuning.scale_kwh.to_numpy()
        trials.append({'num_leaves':leaves,'min_child_samples':minimum,'learning_rate':lr,'iterations':int(m.best_iteration_),'tuning_mae_kwh':float(np.abs(tuning.actual_pv_kwh-pred).mean())})
        print(trials[-1],flush=True)
    best=min(trials,key=lambda x:x['tuning_mae_kwh'])
    fit=pd.concat([train,tuning],ignore_index=True)
    fit.site_id=fit.site_id.astype('category')
    m=lgb.LGBMRegressor(**{**common,'n_estimators':best['iterations']},**{k:best[k] for k in ['num_leaves','min_child_samples','learning_rate']})
    m.fit(fit[features],fit.residual_norm,categorical_feature=['site_id'])
    result={'sample':meta,'trials':trials,'selected':best,'python':sys.version,'lightgbm':lgb.__version__,'created_utc':datetime.now(timezone.utc).isoformat()}
    (ROOT/'models').mkdir(exist_ok=True)
    m.booster_.save_model(str(ROOT/'models/calendar_lightgbm.txt'))
    for split in ['calibration','test']:
        frame,_=expand_split(w,split)
        pred=np.maximum(frame.forecast_norm.to_numpy()+m.predict(frame[features]),0)*frame.scale_kwh.to_numpy()
        result[split]={'lightgbm':metric(frame.actual_pv_kwh.to_numpy(),pred,frame.site_id,frame.scale_kwh.to_numpy())}
        for name in ['vendor','seasonal_naive','persistence']:
            result[split][name]=metric(frame.actual_pv_kwh.to_numpy(),frame[name+'_prediction_kwh'].to_numpy(),frame.site_id,frame.scale_kwh.to_numpy())
        out=frame[['site_id','issue_time','valid_time','delivery_step','actual_pv_kwh','vendor_prediction_kwh','scale_kwh']].copy()
        out['prediction_kwh']=pred
        out.to_parquet(ROOT/'results'/f'calendar_lightgbm_{split}.parquet',index=False)
    (ROOT/'results/calendar_lightgbm.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

def neural(w,meta):
    import torch
    from torch import nn
    torch.set_num_threads(4)
    tensors={}
    for split in ['train','tuning','calibration','test']:
        f,features=expand_split(w,split)
        numeric=[x for x in features if x!='site_id']
        n=len(f)//96
        tensors[split]={'x':f[numeric].to_numpy(np.float32).reshape(n,96,-1),'y':f.residual_norm.to_numpy(np.float32).reshape(n,96),
                        'base':f.vendor_prediction_kwh.to_numpy(np.float32).reshape(n,96),'scale':f.scale_kwh.to_numpy(np.float32).reshape(n,96),
                        'actual':f.actual_pv_kwh.to_numpy(np.float32).reshape(n,96),'sites':f.site_id.to_numpy().reshape(n,96)[:,0].astype(np.int64)}
    mu=tensors['train']['x'].mean(axis=(0,1),keepdims=True)
    sd=tensors['train']['x'].std(axis=(0,1),keepdims=True).clip(.001)
    for d in tensors.values():
        d['x']=torch.from_numpy((d['x']-mu)/sd)
        d['site']=torch.from_numpy(d['sites'])
        d['target']=torch.from_numpy(d['y'])
    class Network(nn.Module):
        def __init__(self,kind):
            super().__init__()
            self.embedding=nn.Embedding(71,8)
            self.input=nn.Linear(len(numeric)+8,32)
            self.body=nn.LSTM(32,16,batch_first=True,bidirectional=True) if kind=='bilstm' else nn.TransformerEncoder(nn.TransformerEncoderLayer(32,4,64,.1,batch_first=True),1)
            self.output=nn.Linear(32,1)
            # The input includes the known delivery-step and sine/cosine position.
        def forward(self,x,site):
            z=torch.cat([x,self.embedding(site)[:,None,:].expand(-1,96,-1)],-1)
            z=self.body(torch.relu(self.input(z)))
            if isinstance(z,tuple): z=z[0]
            return self.output(z).squeeze(-1)
    def predict(model,d):
        model.eval()
        with torch.no_grad():
            return np.concatenate([model(d['x'][i:i+256],d['site'][i:i+256]).numpy() for i in range(0,len(d['x']),256)])
    result={'sample':meta,'seeds':[20261101,20261102,20261103],'architecture':'One encoder layer; width 32; Transformer 4 heads and feedforward width 64; BiLSTM 16 units per direction; site embedding 8.',
            'training':'Adam learning rate 0.001, batch size 256, maximum 25 epochs, patience 5, normalized residual absolute loss. Checkpoint by tuning pooled kWh MAE. Inputs standardized on training only. No refit after selection for neural models.',
            'torch':torch.__version__,'runs':[]}
    for kind in ['bilstm','transformer']:
        for seed in result['seeds']:
            torch.manual_seed(seed)
            rng=np.random.default_rng(seed)
            model=Network(kind)
            optim=torch.optim.Adam(model.parameters(),lr=.001)
            best=float('inf');state=None;best_epoch=0;log=[]
            train=tensors['train'];tune=tensors['tuning']
            for epoch in range(1,26):
                model.train()
                order=rng.permutation(len(train['x']))
                for start in range(0,len(order),256):
                    ix=order[start:start+256]
                    optim.zero_grad()
                    loss=(model(train['x'][ix],train['site'][ix])-train['target'][ix]).abs().mean()
                    loss.backward();optim.step()
                p=np.maximum(tune['base']+predict(model,tune)*tune['scale'],0)
                value=float(np.abs(tune['actual']-p).mean());log.append(value)
                if value<best:
                    best=value;best_epoch=epoch;state={k:v.detach().clone() for k,v in model.state_dict().items()}
                print(kind,seed,'epoch',epoch,'tuning',value,flush=True)
                if epoch-best_epoch>=5:break
            model.load_state_dict(state)
            run={'model':kind,'seed':seed,'best_epoch':best_epoch,'tuning_mae_kwh':best,'tuning_history':log}
            for split in ['calibration','test']:
                d=tensors[split]
                p=np.maximum(d['base']+predict(model,d)*d['scale'],0)
                run[split]=metric(d['actual'].ravel(),p.ravel(),np.repeat(d['sites'],96),d['scale'].ravel())
                np.savez_compressed(ROOT/'results'/f'calendar_{kind}_{seed}_{split}.npz',prediction=p,site=d['sites'])
            result['runs'].append(run)
            (ROOT/'results/calendar_neural.json').write_text(json.dumps(result,indent=2)+'\n')
            print(run,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['lightgbm','neural'],required=True)
    args=parser.parse_args()
    w,meta=prepare()
    print(json.dumps(meta,indent=2),flush=True)
    (lightgbm if args.model=='lightgbm' else neural)(w,meta)
