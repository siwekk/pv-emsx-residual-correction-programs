"""Evaluate deployment-faithful rolling conformal intervals for EMSx."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


from emsx_paths import WORK_ROOT as ROOT
RESULTS = ROOT / "results"
ALPHA = .10
WINDOW_DAYS = 30
LEAD = pd.Timedelta(hours=24)


def interval_score(target: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((upper - lower) + 2 / ALPHA * np.maximum(lower - target, 0) + 2 / ALPHA * np.maximum(target - upper, 0)))


def main() -> None:
    calibration = pd.read_parquet(RESULTS / "emsx_qgbm_conformal_calibration_lead_96.parquet")
    test = pd.read_parquet(RESULTS / "emsx_qgbm_conformal_lead_96.parquet")
    for frame in (calibration, test):
        frame["issue_time"] = pd.to_datetime(frame["issue_time"], utc=True)
        frame["residual_norm"] = np.abs(frame["actual_pv_kwh"] - frame["qgbm_prediction_kwh"]) / frame["scale_kwh"]
    outputs = []
    for site, future in test.groupby("site_id", observed=True, sort=False):
        future = future.sort_values("issue_time").copy()
        available = pd.concat([calibration.loc[calibration["site_id"] == site], future], ignore_index=True)
        future["issue_day"] = future["issue_time"].dt.floor("D")
        for day, block in future.groupby("issue_day", observed=True, sort=False):
            cutoff = day - LEAD
            window_start = cutoff - pd.Timedelta(days=WINDOW_DAYS)
            residuals = available.loc[(available["issue_time"] <= cutoff) & (available["issue_time"] >= window_start), "residual_norm"].to_numpy()
            if not len(residuals):
                residuals = available.loc[available["issue_time"] <= cutoff, "residual_norm"].to_numpy()
            if not len(residuals):
                raise RuntimeError(f"No causally available residuals for site {site} at {day}.")
            radius = float(np.quantile(residuals, 1 - ALPHA, method="higher"))
            half_width = radius * block["scale_kwh"].to_numpy()
            outputs.extend(zip(block["issue_time"], block["site_id"].astype(int), block["actual_pv_kwh"], block["qgbm_prediction_kwh"], np.maximum(block["qgbm_prediction_kwh"].to_numpy()-half_width, 0.), block["qgbm_prediction_kwh"].to_numpy()+half_width, np.full(len(block), radius), np.full(len(block), len(residuals))))
    result_frame = pd.DataFrame(outputs, columns=["issue_time", "site_id", "actual_pv_kwh", "prediction_kwh", "lower_kwh", "upper_kwh", "radius_norm", "calibration_count"])
    target = result_frame.actual_pv_kwh.to_numpy(); lower = result_frame.lower_kwh.to_numpy(); upper = result_frame.upper_kwh.to_numpy()
    result = {"dataset": "EMSx", "lead_hours": 24., "method": "Daily-updated rolling split conformal with a 30-day per-site residual window. For every forecast day, only residuals whose outcomes would be known before the beginning of that day after the 24-hour lead are used.", "interval_nominal_coverage": .90, "test": {"rows": int(len(result_frame)), "coverage": float(((target >= lower) & (target <= upper)).mean()), "mean_width_kwh": float((upper-lower).mean()), "mean_interval_score_kwh": interval_score(target, lower, upper), "median_calibration_count": int(result_frame.calibration_count.median())}}
    (RESULTS / "emsx_qgbm_adaptive_conformal_lead_96.json").write_text(json.dumps(result, indent=2) + "\n")
    result_frame.to_parquet(RESULTS / "emsx_qgbm_adaptive_conformal_lead_96.parquet", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
