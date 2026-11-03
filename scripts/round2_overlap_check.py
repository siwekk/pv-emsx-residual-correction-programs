"""Compares the original retrospective LightGBM and the common-calendar LightGBM on identical deliveries."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
root = Path(sys.argv[1]); calendar = Path(sys.argv[2]); out = Path(sys.argv[3])
o = pd.read_parquet(root / "predictions/emsx_daily_lightgbm.parquet")
c = pd.read_parquet(calendar)
for f in (o, c):
    f["issue_time"] = pd.to_datetime(f.issue_time, utc=True)
    f["site_id"] = f.site_id.astype(int)
c = c.rename(columns={"prediction_kwh": "calendar_kwh"})[["site_id", "issue_time", "delivery_step", "calendar_kwh"]]
m = o.merge(c, on=["site_id", "issue_time", "delivery_step"], how="inner")
fit_end = pd.Timestamp("2017-06-14", tz="UTC")
res = {}
for name, mask in [("before_fit_end", m.issue_time < fit_end), ("after_fit_end", m.issue_time >= fit_end), ("all", np.ones(len(m), bool))]:
    q = m[mask]; y = q.actual_pv_kwh.to_numpy(float)
    e = lambda col: float(np.abs(y - q[col].to_numpy(float)).mean())
    res[name] = {"rows": int(len(q)), "profiles": int(len(q) // 96), "systems": int(q.site_id.nunique()),
                 "vendor_mae_kwh": e("vendor_prediction_kwh"), "original_lightgbm_mae_kwh": e("lightgbm_prediction_kwh"),
                 "calendar_lightgbm_mae_kwh": e("calendar_kwh")}
    for k in ("original", "calendar"):
        res[name][k + "_reduction_percent"] = 100 * (1 - res[name][k + "_lightgbm_mae_kwh"] / res[name]["vendor_mae_kwh"])
print(json.dumps(res, indent=2))
out.write_text(json.dumps(res, indent=2) + "\n")
