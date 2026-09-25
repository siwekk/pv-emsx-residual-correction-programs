"""Tune a residual QGBM using only information available at EMSx issue time."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_emsx_qgbm import load_panel, prepare, score


from emsx_paths import WORK_ROOT as ROOT
CURVES = ROOT / "data" / "processed" / "emsx" / "curve_features"
RESULTS = ROOT / "results"


def prepare_enhanced() -> tuple[pd.DataFrame, list[str]]:
    data, _, base = prepare(load_panel(96), history_lags=15)
    curves = pd.concat([pd.read_parquet(path) for path in sorted(CURVES.glob("site=*.parquet"))], ignore_index=True)
    curves["issue_time"] = pd.to_datetime(curves["issue_time"], utc=True)
    data["site_join"] = data["site_id"].astype(int)
    data = data.merge(curves, left_on=["site_join", "issue_time"], right_on=["site_id", "issue_time"], how="left", suffixes=("", "_curve"))
    curve_features = [column for column in curves.columns if column.startswith("curve_")]
    for column in curve_features: data[column] = data[column] / data["scale_kwh"]
    valid = data["issue_time"] + pd.Timedelta(hours=24)
    data["target_doy_sin"] = np.sin(2 * np.pi * valid.dt.dayofyear / 366)
    data["target_doy_cos"] = np.cos(2 * np.pi * valid.dt.dayofyear / 366)
    data["residual_norm"] = data["target_norm"] - data["forecast_norm"]
    return data, base + curve_features + ["target_doy_sin", "target_doy_cos"]


def prepare_enhanced_with_splits(train_end: float, tuning_end: float, calibration_end: float) -> tuple[pd.DataFrame, list[str]]:
    """Build enhanced features with a separate causal calibration partition."""
    data = load_panel(96).sort_values(["site_id", "issue_time"]).copy()
    rank = data.groupby("site_id", observed=True)["issue_time"].rank(pct=True, method="first")
    data["split"] = np.select([rank <= train_end, rank <= tuning_end, rank <= calibration_end], ["train", "tuning", "calibration"], default="test")
    scales = data.loc[data["split"] == "train"].groupby("site_id", observed=True)["actual_pv_kwh"].quantile(.995).clip(lower=1e-3)
    data["scale_kwh"] = data["site_id"].map(scales).astype(float)
    data["forecast_norm"] = data["forecast_pv_kwh"] / data["scale_kwh"]
    data["issue_actual_norm"] = data["issue_actual_pv_kwh"] / data["scale_kwh"]
    issue = data["issue_time"]
    data["hour_sin"] = np.sin(2*np.pi*issue.dt.hour/24); data["hour_cos"] = np.cos(2*np.pi*issue.dt.hour/24)
    data["doy_sin"] = np.sin(2*np.pi*issue.dt.dayofyear/366); data["doy_cos"] = np.cos(2*np.pi*issue.dt.dayofyear/366)
    data["target_norm"] = data["actual_pv_kwh"] / data["scale_kwh"]
    panels = []
    for _, site in data.groupby("site_id", observed=True, sort=False):
        observed = site.set_index("issue_time")["issue_actual_pv_kwh"]; site = site.copy()
        for lag in range(1, 16):
            site[f"history_lag_{lag}_norm"] = observed.reindex(site["issue_time"]-pd.Timedelta(minutes=15*lag)).to_numpy()/site["scale_kwh"].to_numpy()
        panels.append(site)
    data = pd.concat(panels, ignore_index=True)
    curves = pd.concat([pd.read_parquet(path) for path in sorted(CURVES.glob("site=*.parquet"))], ignore_index=True)
    curves["issue_time"] = pd.to_datetime(curves["issue_time"], utc=True); data["site_join"] = data["site_id"].astype(int)
    data = data.merge(curves, left_on=["site_join", "issue_time"], right_on=["site_id", "issue_time"], how="left", suffixes=("", "_curve"))
    curve_features = [column for column in curves if column.startswith("curve_")]
    for column in curve_features: data[column] = data[column] / data["scale_kwh"]
    valid = data["issue_time"] + pd.Timedelta(hours=24)
    data["target_doy_sin"] = np.sin(2*np.pi*valid.dt.dayofyear/366); data["target_doy_cos"] = np.cos(2*np.pi*valid.dt.dayofyear/366)
    data["residual_norm"] = data["target_norm"]-data["forecast_norm"]
    features = ["forecast_norm", "issue_actual_norm", "hour_sin", "hour_cos", "doy_sin", "doy_cos", "site_id", *[f"history_lag_{lag}_norm" for lag in range(1, 16)], *curve_features, "target_doy_sin", "target_doy_cos"]
    return data, features


def main() -> None:
    data, features = prepare_enhanced()
    sets = {name: data.loc[data["split"] == name].dropna(subset=features + ["residual_norm"]).copy() for name in ("train", "validation", "test")}
    candidates = [
        {"num_leaves": 63, "min_child_samples": 200, "learning_rate": .025},
        {"num_leaves": 127, "min_child_samples": 150, "learning_rate": .025},
        {"num_leaves": 255, "min_child_samples": 100, "learning_rate": .02},
        {"num_leaves": 127, "min_child_samples": 50, "learning_rate": .02},
    ]
    trial_results = []; best_model = None; best_params = None; best_validation = float("inf")
    for params in candidates:
        model = lgb.LGBMRegressor(objective="regression_l1", n_estimators=5000, colsample_bytree=.9, subsample=.85, reg_lambda=2., reg_alpha=.05, n_jobs=32, verbosity=-1, **params)
        model.fit(sets["train"][features], sets["train"].residual_norm, categorical_feature=["site_id"], eval_set=[(sets["validation"][features], sets["validation"].residual_norm)], callbacks=[lgb.early_stopping(150, verbose=False)])
        prediction = np.clip(sets["validation"].forecast_norm.to_numpy() + model.predict(sets["validation"][features]), 0, None) * sets["validation"].scale_kwh.to_numpy()
        mae = float(np.abs(sets["validation"].actual_pv_kwh.to_numpy()-prediction).mean())
        item = {**params, "best_iteration": int(model.best_iteration_ or model.n_estimators_), "validation_mae_kwh": mae}; trial_results.append(item)
        if mae < best_validation: best_validation, best_model, best_params = mae, model, item
    prediction = np.clip(sets["test"].forecast_norm.to_numpy() + best_model.predict(sets["test"][features]), 0, None) * sets["test"].scale_kwh.to_numpy()
    result = {"dataset": "EMSx", "lead_hours": 24., "method": "Tuned global LightGBM residual corrector", "feature_contract": "Vendor target forecast, 16 preceding PV values, issue-time forecast-curve summaries and anchor forecasts through 24 hours, target-time seasonal features, and categorical site identity.", "splitting": "Per-site chronological 70/10/20 split. Hyperparameters selected using validation MAE only.", "trials": trial_results, "selected": best_params, "test": score(sets["test"], prediction)}
    RESULTS.mkdir(exist_ok=True); (RESULTS / "emsx_tuned_residual_qgbm_lead_96.json").write_text(json.dumps(result, indent=2)+"\n")
    output = sets["test"][["issue_time", "site_id", "actual_pv_kwh", "forecast_pv_kwh", "scale_kwh"]].copy(); output["tuned_residual_qgbm_prediction_kwh"] = prediction; output.to_parquet(RESULTS / "emsx_tuned_residual_qgbm_lead_96.parquet", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
