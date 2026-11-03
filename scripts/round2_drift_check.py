"""Correction gain of the common-calendar LightGBM by calendar quarter of issue (test period only)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
c = pd.read_parquet(Path(sys.argv[1]))
c["issue_time"] = pd.to_datetime(c.issue_time, utc=True)
c["quarter"] = c.issue_time.dt.tz_convert(None).dt.to_period("Q").astype(str)
y = c.actual_pv_kwh.to_numpy(float)
c["ev"] = np.abs(y - c.vendor_prediction_kwh.to_numpy(float))
c["em"] = np.abs(y - c.prediction_kwh.to_numpy(float))
g = c.groupby("quarter").agg(rows=("ev", "size"), systems=("site_id", "nunique"), vendor_mae_kwh=("ev", "mean"), lightgbm_mae_kwh=("em", "mean")).reset_index()
g["profiles"] = g.rows // 96
g["reduction_percent"] = 100 * (1 - g.lightgbm_mae_kwh / g.vendor_mae_kwh)
print(g.to_string(index=False))
Path(sys.argv[2]).write_text(g.to_json(orient="records", indent=2) + "\n")
