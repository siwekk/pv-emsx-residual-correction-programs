"""Fit a history-matched QGBM and calibrate 90 percent intervals on EMSx."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from run_emsx_qgbm import BASE_FEATURES, PANEL_ROOT, RESULTS, load_panel


def prepare() -> tuple[pd.DataFrame, list[str]]:
    data = load_panel(96).drop_duplicates(["site_id", "issue_time"]).sort_values(["site_id", "issue_time"]).copy()
    rank = data.groupby("site_id", observed=True)["issue_time"].rank(pct=True, method="first")
    data["split"] = np.select([rank <= .65, rank <= .70, rank <= .80], ["train", "tuning", "calibration"], default="test")
    scales = data.loc[data["split"] == "train"].groupby("site_id", observed=True)["actual_pv_kwh"].quantile(.995).clip(lower=1e-3)
    data["scale_kwh"] = data["site_id"].map(scales).astype(float)
    data["forecast_norm"] = data["forecast_pv_kwh"] / data["scale_kwh"]
    data["issue_actual_norm"] = data["issue_actual_pv_kwh"] / data["scale_kwh"]
    issue = data["issue_time"]
    data["hour_sin"] = np.sin(2 * np.pi * issue.dt.hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * issue.dt.hour / 24)
    data["doy_sin"] = np.sin(2 * np.pi * issue.dt.dayofyear / 366)
    data["doy_cos"] = np.cos(2 * np.pi * issue.dt.dayofyear / 366)
    data["target_norm"] = data["actual_pv_kwh"] / data["scale_kwh"]
    history = [f"history_lag_{lag}_norm" for lag in range(1, 16)]
    panels = []
    for _, site in data.groupby("site_id", observed=True, sort=False):
        observed = site.set_index("issue_time")["issue_actual_pv_kwh"]
        site = site.copy()
        for lag, name in enumerate(history, 1):
            site[name] = observed.reindex(site["issue_time"] - pd.Timedelta(minutes=15 * lag)).to_numpy() / site["scale_kwh"].to_numpy()
        panels.append(site)
    return pd.concat(panels, ignore_index=True), BASE_FEATURES + history


def interval_score(target: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> float:
    return float(np.mean((upper - lower) + 2 / alpha * np.maximum(lower - target, 0) + 2 / alpha * np.maximum(target - upper, 0)))


def main() -> None:
    data, features = prepare()
    sets = {name: data.loc[data["split"] == name].dropna(subset=features + ["target_norm"]).copy() for name in ("train", "tuning", "calibration", "test")}
    params = dict(objective="regression_l1", n_estimators=1200, learning_rate=.04, num_leaves=63, min_child_samples=300, colsample_bytree=.9, subsample=.8, reg_lambda=1., n_jobs=32, verbosity=-1)
    selector = lgb.LGBMRegressor(**params)
    selector.fit(sets["train"][features], sets["train"]["target_norm"], categorical_feature=["site_id"], eval_set=[(sets["tuning"][features], sets["tuning"]["target_norm"])], callbacks=[lgb.early_stopping(80, verbose=False)])
    model = lgb.LGBMRegressor(**{**params, "n_estimators": int(selector.best_iteration_ or params["n_estimators"])})
    refit = pd.concat([sets["train"], sets["tuning"]], ignore_index=True)
    model.fit(refit[features], refit["target_norm"], categorical_feature=["site_id"])
    def predict(frame: pd.DataFrame) -> np.ndarray:
        return np.clip(model.predict(frame[features]), 0, None) * frame["scale_kwh"].to_numpy()
    calibration_prediction = predict(sets["calibration"])
    residual_norm = np.abs(sets["calibration"]["actual_pv_kwh"].to_numpy() - calibration_prediction) / sets["calibration"]["scale_kwh"].to_numpy()
    quantile = float(np.quantile(residual_norm, .90, method="higher"))
    prediction = predict(sets["test"])
    target = sets["test"]["actual_pv_kwh"].to_numpy(); scale = sets["test"]["scale_kwh"].to_numpy()
    half_width = quantile * scale; lower = np.maximum(prediction - half_width, 0); upper = prediction + half_width
    coverage = (target >= lower) & (target <= upper)
    result = {"dataset": "EMSx", "lead_hours": 24., "method": "QGBM point correction with held-out split conformal intervals", "feature_contract": "vendor forecast, 16 available 15-minute PV values, UTC calendar features, and a categorical site identifier", "split": "Per site: 65 percent training, 5 percent tuning, 10 percent calibration, and final 20 percent test.", "interval_nominal_coverage": .90, "conformal_normalized_half_width": quantile, "fit": {"best_iteration": int(selector.best_iteration_ or params["n_estimators"]), **{name: int(len(frame)) for name, frame in sets.items()}}, "test": {"mae_kwh": float(np.abs(target-prediction).mean()), "vendor_mae_kwh": float(np.abs(target-sets["test"]["forecast_pv_kwh"].to_numpy()).mean()), "coverage": float(coverage.mean()), "mean_width_kwh": float((upper-lower).mean()), "mean_interval_score_kwh": interval_score(target, lower, upper, .10)}}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "emsx_qgbm_conformal_lead_96.json").write_text(json.dumps(result, indent=2) + "\n")
    output = sets["test"][["issue_time", "site_id", "actual_pv_kwh", "forecast_pv_kwh", "scale_kwh"]].copy()
    output["qgbm_prediction_kwh"] = prediction; output["lower_kwh"] = lower; output["upper_kwh"] = upper
    output.to_parquet(RESULTS / "emsx_qgbm_conformal_lead_96.parquet", index=False)
    calibration_output = sets["calibration"][["issue_time", "site_id", "actual_pv_kwh", "scale_kwh"]].copy()
    calibration_output["qgbm_prediction_kwh"] = calibration_prediction
    calibration_output.to_parquet(RESULTS / "emsx_qgbm_conformal_calibration_lead_96.parquet", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
