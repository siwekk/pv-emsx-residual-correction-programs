"""Evaluate a causal seasonal ARIMA baseline at hourly EMSx forecast origins."""

from __future__ import annotations

import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from run_emsx_qgbm import load_panel


from emsx_paths import WORK_ROOT as ROOT
RESULTS = ROOT / "results"


def evaluate_site(item):
    site, frame = item
    frame = frame.set_index("issue_time").sort_index()
    hourly = frame["issue_actual_pv_kwh"].resample("1h").mean().asfreq("1h")
    split = int(len(hourly)*.8); train = hourly.iloc[:int(len(hourly)*.7)]; test = hourly.iloc[split:]
    model = SARIMAX(train, order=(2, 0, 1), seasonal_order=(1, 0, 0, 24), trend="c", enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False, maxiter=100)
    candidates = test.index[(test.notna()) & (test.shift(-24).notna())]
    origins = candidates[np.linspace(0, max(0, len(candidates)-1), min(100, len(candidates)), dtype=int)]
    rows = []
    for origin in origins:
        state = fitted.apply(hourly.loc[:origin], refit=False)
        forecast = float(np.asarray(state.forecast(steps=24))[-1]); target = float(hourly.loc[origin+pd.Timedelta(hours=24)])
        rows.append((int(site), origin, target, max(forecast, 0.0)))
    return int(site), rows


def main() -> None:
    data = load_panel(96); groups = [(site, frame) for site, frame in data.groupby("site_id", observed=True, sort=False)]
    rows = []
    with ProcessPoolExecutor(max_workers=12) as executor:
        for site, site_rows in executor.map(evaluate_site, groups):
            rows.extend(site_rows); print({"site": site, "origins": len(site_rows)}, flush=True)
    output = pd.DataFrame(rows, columns=["site_id", "issue_time", "actual_pv_kwh", "arima_prediction_kwh"])
    mae = float(np.abs(output.actual_pv_kwh-output.arima_prediction_kwh).mean())
    result = {"dataset": "EMSx", "lead_hours": 24., "method": "Per-site SARIMA(2,0,1)(1,0,0)24 baseline", "restriction": "Hourly origins only, 100 evenly spaced causal forecast origins per site. This is a classical reference, not a row-for-row 15-minute comparison.", "test_rows": int(len(output)), "mae_kwh": mae}
    RESULTS.mkdir(exist_ok=True); output.to_parquet(RESULTS/"emsx_arima_hourly_lead_96.parquet", index=False); (RESULTS/"emsx_arima_hourly_lead_96.json").write_text(json.dumps(result, indent=2)+"\n"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
