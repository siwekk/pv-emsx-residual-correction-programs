from pathlib import Path
import sys
import tempfile
from helpers import temporary_directory
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import calendar_validation as cv
from calendar_quantiles import calibrate
from emsx_probabilistic import QCOLS

def profile(day,value):
    issue=pd.Timestamp(day,tz='UTC')
    f=pd.DataFrame({'site_id':1,'issue_time':issue,'delivery_step':np.arange(1,97),'scale_kwh':1.,'actual_pv_kwh':float(value)})
    f['valid_time']=issue+pd.to_timedelta(f.delivery_step*15,unit='m')
    for c in QCOLS:f[c]=0.
    return f

class CalendarTests(unittest.TestCase):
    def test_future_outcomes_do_not_change_current_calibration(self):
        cal=pd.concat([profile('2016-06-29',1),profile('2016-06-30',2)],ignore_index=True)
        test=pd.concat([profile('2016-07-01',3),profile('2016-07-02',4)],ignore_index=True)
        a=calibrate(cal,test)
        changed=test.copy();changed.actual_pv_kwh=1e9
        b=calibrate(cal,changed)
        columns=['cal_'+c for c in QCOLS]
        np.testing.assert_allclose(a.iloc[:96][columns],b.iloc[:96][columns])
        np.testing.assert_allclose(a.iloc[:96].cal_q50_kwh,1.5)

    def test_empty_history_uses_raw_quantiles(self):
        test=profile('2016-07-01',3)
        out=calibrate(test.iloc[:0],test)
        np.testing.assert_array_equal(out[['cal_'+c for c in QCOLS]].to_numpy(),test[QCOLS].to_numpy())

    def test_missing_month_falls_back_to_mature_history(self):
        out=calibrate(profile('2016-01-01',2),profile('2016-07-01',3))
        np.testing.assert_allclose(out.cal_q50_kwh,2)

    def test_scale_never_uses_holdout_targets(self):
        dates=[]
        for day,count in [('2015-06-01',30),('2016-02-01',7),('2016-05-01',7),('2016-08-01',7)]:
            dates.extend(pd.date_range(day,periods=count,freq='D',tz='UTC'))
        frame=pd.DataFrame({'site_id':1,'issue_time':dates,'complete_case':True,'scale_kwh':1.})
        for c in cv.TARGET_COLUMNS:frame[c]=10.
        with temporary_directory() as tmp:
            with patch.object(cv,'ROOT',Path(tmp)),patch.object(cv,'load_wide',return_value=frame.copy()):a,meta=cv.prepare()
            frame.loc[frame.issue_time>=pd.Timestamp('2016-07-01',tz='UTC'),cv.TARGET_COLUMNS]=1e8
            with patch.object(cv,'ROOT',Path(tmp)),patch.object(cv,'load_wide',return_value=frame.copy()):b,_=cv.prepare()
        np.testing.assert_array_equal(a.scale_kwh,b.scale_kwh)
        self.assertEqual(meta['profiles']['train'],30)
        self.assertLessEqual(pd.Timestamp(meta['latest_fit_delivery']),pd.Timestamp('2016-04-01',tz='UTC'))

if __name__=='__main__':unittest.main()
