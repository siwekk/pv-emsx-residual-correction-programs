"""Evaluate a leakage-safe global QGBM correction of EMSx vendor forecasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


from emsx_paths import WORK_ROOT as ROOT
PANEL_ROOT = ROOT / "data" / "processed" / "emsx" / "forecast_panels"
RESULTS = ROOT / "results"
BASE_FEATURES = ["forecast_norm", "issue_actual_norm", "hour_sin", "hour_cos", "doy_sin", "doy_cos", "site_id"]


def load_panel(lead_steps: int) -> pd.DataFrame:
    frames = []
    for path in sorted(PANEL_ROOT.glob("site=*/panel.parquet")):
        frame = pd.read_parquet(path, filters=[("lead_steps", "=", lead_steps)])
        # Match the TCN data contract exactly: one forecast issue per site and
        # timestamp before chronological splitting.
        frame = frame.drop_duplicates("issue_time").copy()
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["issue_time"] = pd.to_datetime(data["issue_time"], utc=True)
    data["site_id"] = data["site_id"].astype("category")
    return data


def prepare(data: pd.DataFrame, history_lags: int) -> tuple[pd.DataFrame, dict[int, float], list[str]]:
    data = data.sort_values(["site_id", "issue_time"]).copy()
    split_rank = data.groupby("site_id", observed=True)["issue_time"].rank(pct=True, method="first")
    data["split"] = np.where(split_rank <= 0.70, "train", np.where(split_rank <= 0.80, "validation", "test"))
    scales = (
        data.loc[data["split"] == "train"].groupby("site_id", observed=True)["actual_pv_kwh"].quantile(0.995).clip(lower=1e-3)
    )
    data["scale_kwh"] = data["site_id"].map(scales).astype(float)
    data["forecast_norm"] = data["forecast_pv_kwh"] / data["scale_kwh"]
    data["issue_actual_norm"] = data["issue_actual_pv_kwh"] / data["scale_kwh"]
    local = data["issue_time"]
    data["hour_sin"] = np.sin(2 * np.pi * local.dt.hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * local.dt.hour / 24)
    data["doy_sin"] = np.sin(2 * np.pi * local.dt.dayofyear / 366)
    data["doy_cos"] = np.cos(2 * np.pi * local.dt.dayofyear / 366)
    data["target_norm"] = data["actual_pv_kwh"] / data["scale_kwh"]
    history_features = []
    if history_lags:
        history_features = [f"history_lag_{lag}_norm" for lag in range(1, history_lags + 1)]
        frames = []
        for _, site in data.groupby("site_id", observed=True, sort=False):
            observed = site.set_index("issue_time")["issue_actual_pv_kwh"]
            site = site.copy()
            for lag in range(1, history_lags + 1):
                name = f"history_lag_{lag}_norm"
                site[name] = observed.reindex(site["issue_time"] - pd.Timedelta(minutes=15 * lag)).to_numpy() / site["scale_kwh"].to_numpy()
            frames.append(site)
        data = pd.concat(frames, ignore_index=True)
    return data, {int(key): float(value) for key, value in scales.items()}, BASE_FEATURES + history_features


def score(data: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    target = data["actual_pv_kwh"].to_numpy()
    baseline = data["forecast_pv_kwh"].to_numpy()
    scale = data["scale_kwh"].to_numpy()
    absolute_error = np.abs(target - prediction)
    return {
        "rows": int(len(data)),
        "mae_kwh": float(absolute_error.mean()),
        "nmae_q995": float((absolute_error / scale).mean()),
        "vendor_mae_kwh": float(np.abs(target - baseline).mean()),
        "vendor_nmae_q995": float((np.abs(target - baseline) / scale).mean()),
        "mae_improvement_pct": float(100 * (1 - absolute_error.mean() / np.abs(target - baseline).mean())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lead-steps", type=int, default=96)
    parser.add_argument("--history-lags", type=int, default=0)
    args = parser.parse_args()
    data, scales, features = prepare(load_panel(args.lead_steps), args.history_lags)
    train = data.loc[data["split"] == "train"].dropna(subset=features + ["target_norm"])
    valid = data.loc[data["split"] == "validation"].dropna(subset=features + ["target_norm"])
    test = data.loc[data["split"] == "test"].dropna(subset=features + ["target_norm"])
    model = lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=1200, learning_rate=0.04,
        num_leaves=63, min_child_samples=300, colsample_bytree=0.9,
        subsample=0.8, reg_lambda=1.0, n_jobs=32, verbosity=-1,
    )
    model.fit(train[features], train["target_norm"], categorical_feature=["site_id"], callbacks=[lgb.early_stopping(80, verbose=False)], eval_set=[(valid[features], valid["target_norm"])])
    prediction = np.clip(model.predict(test[features]), 0, None) * test["scale_kwh"].to_numpy()
    result = {
        "dataset": "EMSx",
        "lead_steps": args.lead_steps,
        "lead_hours": args.lead_steps / 4,
        "feature_contract": f"vendor forecast, {args.history_lags + 1} available 15-minute PV history values, UTC calendar features, and a categorical site identifier",
        "splitting": "Per-site chronological 70/10/20 train/validation/test split.",
        "test": score(test, prediction),
        "fit": {"best_iteration": int(model.best_iteration_ or model.n_estimators_), "train_rows": int(len(train)), "validation_rows": int(len(valid)), "test_rows": int(len(test))},
        "scales_q995_kwh": scales,
        "feature_importance_gain": dict(zip(features, model.booster_.feature_importance(importance_type="gain").tolist())),
    }
    RESULTS.mkdir(exist_ok=True)
    suffix = f"_history_{args.history_lags}" if args.history_lags else ""
    destination = RESULTS / f"emsx_qgbm_lead_{args.lead_steps}{suffix}.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    predictions = test[["issue_time", "site_id", "actual_pv_kwh", "forecast_pv_kwh", "scale_kwh"]].copy()
    predictions["qgbm_prediction_kwh"] = prediction
    predictions.to_parquet(destination.with_suffix(".parquet"), index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
