"""Summarize all calendar runs and draw their numerical comparison."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from emsx_round1_paths import ROOT

def main():
    out=ROOT/'results'
    tree=json.loads((out/'calendar_lightgbm.json').read_text())
    neural=json.loads((out/'calendar_neural.json').read_text())
    if len(neural['runs'])!=6:
        raise ValueError('All six planned neural fits must finish before summarizing')
    rows=[]
    for split in ['calibration','test']:
        for name in ['lightgbm','vendor','seasonal_naive','persistence']:
            rows.append({'method':name,'partition':split,**tree[split][name]})
        for name in ['bilstm','transformer']:
            subset=[r[split] for r in neural['runs'] if r['model']==name]
            row={'method':name,'partition':split,'seeds':3,'rows':subset[0]['rows']}
            for metric in ['mae_kwh','system_averaged_mae_kwh','normalized_mae']:
                row[metric]=float(np.mean([r[metric] for r in subset]))
                row[metric+'_sd']=float(np.std([r[metric] for r in subset],ddof=1))
            rows.append(row)
    frame=pd.DataFrame(rows)
    frame.to_csv(out/'calendar_comparison.csv',index=False)
    pred=pd.read_parquet(out/'calendar_lightgbm_test.parquet')
    pred['gain']=np.abs(pred.actual_pv_kwh-pred.vendor_prediction_kwh)-np.abs(pred.actual_pv_kwh-pred.prediction_kwh)
    daily=pred.groupby(['site_id','issue_time'],observed=True).gain.mean().reset_index()
    groups=[g.gain.to_numpy(float) for _,g in daily.groupby('site_id',observed=True)]
    rng=np.random.default_rng(20261101)
    estimates=[]
    for _ in range(2000):
        chosen=rng.integers(len(groups),size=len(groups))
        estimates.append(np.mean([rng.choice(groups[j],size=len(groups[j]),replace=True).mean() for j in chosen]))
    boot={'estimand':'equal-system mean vendor MAE minus LightGBM MAE; systems then daily profiles resampled','seed':20261101,'replicates':2000,'estimate_kwh':float(np.mean([x.mean() for x in groups])),'limits_95_kwh':np.quantile(estimates,[.025,.975]).tolist(),'systems_improved':int(sum(x.mean()>0 for x in groups)),'systems':len(groups)}
    (out/'calendar_bootstrap.json').write_text(json.dumps(boot,indent=2)+'\n')
    figures=ROOT/'figures';figures.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.7),sharey=True)
    order=['vendor','lightgbm','bilstm','transformer']
    for ax,split in zip(axes,['calibration','test']):
        sub=frame[frame.partition==split].set_index('method')
        values=[sub.loc[n,'mae_kwh'] for n in order]
        errors=[sub.loc[n,'mae_kwh_sd'] for n in ['bilstm','transformer']]
        ax.bar(range(4),values,color=['#7f7f7f','#0047bb','#4486de','#8bb3e9'])
        ax.errorbar([2,3],[values[2],values[3]],yerr=errors,fmt='none',ecolor='black',capsize=4)
        ax.set_xticks(range(4),['Vendor','LightGBM','BiLSTM','Transformer'],rotation=18,ha='right')
        ax.set_title(split.capitalize());ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Pooled MAE (kWh)')
    fig.tight_layout()
    for suffix in ['pdf','png']:fig.savefig(figures/f'calendar_validation.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    print(frame.to_string(index=False))
    print(json.dumps(boot,indent=2))

if __name__=='__main__':main()
