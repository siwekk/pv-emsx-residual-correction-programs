"""Evaluate causal adaptive intervals for the final enhanced residual QGBM."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_emsx_tuned_qgbm import prepare_enhanced_with_splits


from emsx_paths import WORK_ROOT as ROOT
RESULTS = ROOT / "results"
ALPHA = .10


def score(y, lo, mid, hi):
    return {"mae_kwh": float(np.abs(y-mid).mean()), "coverage": float(((y >= lo) & (y <= hi)).mean()), "mean_width_kwh": float((hi-lo).mean()), "interval_score_kwh": float(np.mean((hi-lo)+20*np.maximum(lo-y, 0)+20*np.maximum(y-hi, 0)))}


def main() -> None:
    data, features = prepare_enhanced_with_splits(.65, .70, .80)
    sets = {name: data.loc[data["split"] == name].dropna(subset=features+["residual_norm"]).copy() for name in ("train", "tuning", "calibration", "test")}
    fit = pd.concat([sets["train"], sets["tuning"]], ignore_index=True)
    params = dict(n_estimators=5000, learning_rate=.02, num_leaves=255, min_child_samples=100, colsample_bytree=.9, subsample=.85, reg_lambda=2., reg_alpha=.05, n_jobs=32, verbosity=-1)
    prediction = {name: {} for name in ("calibration", "test")}
    for quantile in (.05, .50, .95):
        model = lgb.LGBMRegressor(objective="quantile", alpha=quantile, **params)
        model.fit(fit[features], fit.residual_norm, categorical_feature=["site_id"])
        for name in prediction:
            frame = sets[name]; prediction[name][quantile] = np.clip(frame.forecast_norm.to_numpy()+model.predict(frame[features]), 0, None)*frame.scale_kwh.to_numpy()
    calibration = sets["calibration"]; test = sets["test"]
    cal_y = calibration.actual_pv_kwh.to_numpy(); test_y = test.actual_pv_kwh.to_numpy()
    calibration_output = calibration[["issue_time", "site_id", "actual_pv_kwh", "scale_kwh"]].copy().reset_index(drop=True)
    calibration_output["low_raw_kwh"] = prediction["calibration"][.05]; calibration_output["median_kwh"] = prediction["calibration"][.50]; calibration_output["high_raw_kwh"] = prediction["calibration"][.95]
    nonconformity = np.maximum.reduce([prediction["calibration"][.05]-cal_y, cal_y-prediction["calibration"][.95], np.zeros(len(calibration))])/calibration.scale_kwh.to_numpy()
    cal_frame = calibration[["issue_time", "site_id"]].copy(); cal_frame["residual_norm"] = nonconformity
    output = test[["issue_time", "site_id", "actual_pv_kwh", "scale_kwh"]].copy().reset_index(drop=True)
    output["low_raw_kwh"] = prediction["test"][.05]; output["median_kwh"] = prediction["test"][.50]; output["high_raw_kwh"] = prediction["test"][.95]
    radii = np.zeros(len(output))
    for site, block in output.groupby("site_id", observed=True, sort=False):
        history = cal_frame.loc[cal_frame.site_id == site]
        all_residuals = pd.concat([history, block[["issue_time"]].assign(residual_norm=np.maximum.reduce([block.low_raw_kwh.to_numpy()-block.actual_pv_kwh.to_numpy(), block.actual_pv_kwh.to_numpy()-block.high_raw_kwh.to_numpy(), np.zeros(len(block))])/block.scale_kwh.to_numpy())], ignore_index=True)
        for day, day_block in block.groupby(block.issue_time.dt.floor("D"), observed=True):
            cutoff = day-pd.Timedelta(hours=24); start = cutoff-pd.Timedelta(days=30)
            values = all_residuals.loc[(all_residuals.issue_time <= cutoff) & (all_residuals.issue_time >= start), "residual_norm"].to_numpy()
            if not len(values): values = all_residuals.loc[all_residuals.issue_time <= cutoff, "residual_norm"].to_numpy()
            radii[day_block.index.to_numpy()] = np.quantile(values, .90, method="higher")
    output["radius_norm"] = radii; output["low_adaptive_kwh"] = np.maximum(output.low_raw_kwh-output.radius_norm*output.scale_kwh, 0); output["high_adaptive_kwh"] = output.high_raw_kwh+output.radius_norm*output.scale_kwh
    raw = score(test_y, output.low_raw_kwh.to_numpy(), output.median_kwh.to_numpy(), output.high_raw_kwh.to_numpy())
    adaptive = score(test_y, output.low_adaptive_kwh.to_numpy(), output.median_kwh.to_numpy(), output.high_adaptive_kwh.to_numpy())
    result = {"dataset": "EMSx", "lead_hours": 24., "method": "Enhanced residual quantile QGBM with daily-updated, 30-day causal adaptive conformal calibration", "feature_contract": "Final enhanced forecast-curve residual-QGBM feature set.", "split": "65 percent training, 5 percent tuning, 10 percent calibration, 20 percent test per site.", "rows": {name: int(len(frame)) for name, frame in sets.items()}, "raw_test": raw, "adaptive_test": adaptive}
    RESULTS.mkdir(exist_ok=True); (RESULTS/"emsx_enhanced_probabilistic_qgbm_lead_96.json").write_text(json.dumps(result, indent=2)+"\n"); output.to_parquet(RESULTS/"emsx_enhanced_probabilistic_qgbm_lead_96.parquet", index=False); calibration_output.to_parquet(RESULTS/"emsx_enhanced_probabilistic_qgbm_calibration_lead_96.parquet", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
