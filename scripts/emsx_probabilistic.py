"""Shared, corrected scoring for the eleven reported EMSx quantiles."""
import numpy as np

SCORING_VERSION = 'wis4-corrected-20261101'
QUANTILES = np.array([.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95])
QCOLS = [f'q{round(100*x):02d}_kwh' for x in QUANTILES]

def normalize_saved_metrics(metrics):
    """Recover standard WIS when a full run kept its unrounded component scores.

    This cannot recover seed variation from an aggregate WIS mean alone.
    No original record is overwritten by this function.
    """
    result=dict(metrics)
    if 'weighted_interval_score_kwh' in result:
        pb=result.get('mean_pinball_kwh')
        outer=result.get('central_90_interval_score_kwh')
        if not isinstance(pb,(int,float)) or not isinstance(outer,(int,float)):
            raise ValueError('Corrected WIS needs full quantiles or unrounded pinball and outer interval scores')
        result['weighted_interval_score_kwh']=(11*pb-.05*outer)/4.5
    return result

def trapezoid(values, grid):
    integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return integrate(values, grid)

def wis_rows(y, quantiles):
    """Standard WIS for the median and central 20, 40, 60, 80% intervals.

    The outer 90% interval is scored separately. Nonfinite rows propagate NaN
    so callers that audit missing values can retain their finite-row policy.
    """
    y=np.asarray(y,dtype=float)
    q=np.asarray(quantiles,dtype=float)
    if y.ndim != 1 or q.shape != (len(y),11):
        raise ValueError('Expected one target and eleven quantiles per row')
    if np.any(np.diff(q,axis=1)<0):
        raise ValueError('Rearrange crossing quantiles before scoring')
    numerator=.5*np.abs(y-q[:,5])
    for alpha,lo,hi in ((.2,1,9),(.4,2,8),(.6,3,7),(.8,4,6)):
        interval=q[:,hi]-q[:,lo]+2/alpha*np.maximum(q[:,lo]-y,0)+2/alpha*np.maximum(y-q[:,hi],0)
        numerator += alpha/2*interval
    return numerator/4.5

def score(frame,prefix=''):
    y=frame.actual_pv_kwh.to_numpy(float)
    q=frame[[prefix+c for c in QCOLS]].to_numpy(float)
    if not len(y) or not np.isfinite(y).all() or not np.isfinite(q).all():
        raise ValueError('Scoring requires nonempty, fully finite predictions')
    losses=np.maximum(QUANTILES*(y[:,None]-q),(QUANTILES-1)*(y[:,None]-q))
    wis=wis_rows(y,q)
    if not np.allclose(wis,losses[:,1:10].sum(axis=1)/4.5,rtol=1e-11,atol=1e-8):
        raise AssertionError('Interval and pinball definitions disagree')
    return {'rows':len(y),'median_mae_kwh':float(np.abs(y-q[:,5]).mean()),
            'mean_pinball_kwh':float(losses.mean()),'wis_kwh':float(wis.mean()),
            'grid_crps_kwh':float(2*trapezoid(losses.mean(axis=0),QUANTILES)),
            'coverage_90':float(((y>=q[:,0])&(y<=q[:,-1])).mean()),
            'width_90_kwh':float((q[:,-1]-q[:,0]).mean()),
            'interval_score_90_kwh':float((q[:,-1]-q[:,0]+20*np.maximum(q[:,0]-y,0)+20*np.maximum(y-q[:,-1],0)).mean())}
