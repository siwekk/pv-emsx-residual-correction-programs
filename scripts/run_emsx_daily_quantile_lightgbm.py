"""Fit common-grid residual quantile LightGBM and causal daily calibration for EMSx curves."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from emsx_daily_curve_common import expand_split, load_wide


SEED = 20260906
QUANTILES = [.05, .10, .20, .30, .40, .50, .60, .70, .80, .90, .95]
QCOLS = [f"q{int(round(100 * q)):02d}_kwh" for q in QUANTILES]


def fit_quantile(quantile: float, fit: pd.DataFrame, features: list[str], iterations: int, threads: int) -> tuple[float, lgb.LGBMRegressor, float]:
    before = time.perf_counter()
    model = lgb.LGBMRegressor(
        objective="quantile", alpha=quantile, n_estimators=iterations, learning_rate=.02,
        num_leaves=255, min_child_samples=100, colsample_bytree=.9, subsample=.85,
        reg_lambda=2., reg_alpha=.05, n_jobs=threads, verbosity=-1, random_state=SEED,
    )
    model.fit(fit[features], fit.residual_norm, categorical_feature=["site_id"])
    return quantile, model, time.perf_counter() - before


def predict(models: dict[float, lgb.LGBMRegressor], frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    output = frame[["site_id", "issue_time", "valid_time", "delivery_step", "actual_pv_kwh", "vendor_prediction_kwh", "scale_kwh"]].copy().reset_index(drop=True)
    base = frame.forecast_norm.to_numpy(float)
    scale = frame.scale_kwh.to_numpy(float)
    for quantile, column in zip(QUANTILES, QCOLS):
        output[column] = np.maximum(base + models[quantile].predict(frame[features]), 0) * scale
    output[QCOLS] = np.sort(output[QCOLS].to_numpy(float), axis=1)
    return output


def metrics(frame: pd.DataFrame, prefix: str = "") -> dict:
    columns = [f"{prefix}{column}" for column in QCOLS]
    y = frame.actual_pv_kwh.to_numpy(float)
    q = frame[columns].to_numpy(float)
    losses = np.column_stack([np.maximum(tau * (y - q[:, index]), (tau - 1) * (y - q[:, index])) for index, tau in enumerate(QUANTILES)])
    interval_terms = []
    for alpha, lower, upper in ((.8, 1, 9), (.6, 2, 8), (.4, 3, 7), (.2, 4, 6)):
        score = q[:, upper] - q[:, lower] + 2 / alpha * (q[:, lower] - y) * (y < q[:, lower]) + 2 / alpha * (y - q[:, upper]) * (y > q[:, upper])
        interval_terms.append(alpha / 2 * score)
    wis = (0.5 * np.abs(y - q[:, 5]) + np.sum(interval_terms, axis=0)) / 4.5
    score90 = q[:, -1] - q[:, 0] + 20 * (q[:, 0] - y) * (y < q[:, 0]) + 20 * (y - q[:, -1]) * (y > q[:, -1])
    return {
        "median_mae_kwh": float(np.mean(np.abs(y - q[:, 5]))),
        "mean_pinball_kwh": float(losses.mean()),
        "grid_crps_kwh": float(2 * np.trapz(losses.mean(axis=0), QUANTILES)),
        "wis_kwh": float(wis.mean()),
        "coverage_90": float(np.mean((y >= q[:, 0]) & (y <= q[:, -1]))),
        "width_90_kwh": float(np.mean(q[:, -1] - q[:, 0])),
        "interval_score_90_kwh": float(score90.mean()),
        "observed_quantile_frequencies": {str(tau): float(np.mean(y <= q[:, index])) for index, tau in enumerate(QUANTILES)},
    }


def causal_calibrate(calibration: pd.DataFrame, test: pd.DataFrame, window_days: int) -> pd.DataFrame:
    output = test.copy()
    for column in QCOLS:
        output[f"cal_{column}"] = np.nan
    history_parts = []
    for source in (calibration, test):
        part = source[["site_id", "issue_time", "valid_time", "delivery_step", "actual_pv_kwh", "scale_kwh"]].copy()
        for quantile, column in zip(QUANTILES, QCOLS):
            part[f"error_{column}"] = (source.actual_pv_kwh.to_numpy(float) - source[column].to_numpy(float)) / source.scale_kwh.to_numpy(float)
        history_parts.append(part)
    history = pd.concat(history_parts, ignore_index=True)
    for site_id, block in output.groupby("site_id", observed=True, sort=False):
        site_history = history.loc[history.site_id == site_id]
        for issue_time, day in block.groupby("issue_time", observed=True, sort=False):
            start = issue_time - pd.Timedelta(days=window_days)
            available = site_history.loc[(site_history.valid_time <= issue_time) & (site_history.valid_time > start)]
            if available.empty:
                available = site_history.loc[site_history.valid_time <= issue_time]
            for quantile, column in zip(QUANTILES, QCOLS):
                by_step = available.groupby("delivery_step", observed=True)[f"error_{column}"].quantile(quantile)
                offsets = day.delivery_step.map(by_step).to_numpy(float)
                if not np.isfinite(offsets).all():
                    offsets[~np.isfinite(offsets)] = available[f"error_{column}"].quantile(quantile)
                output.loc[day.index, f"cal_{column}"] = np.maximum(day[column].to_numpy(float) + offsets * day.scale_kwh.to_numpy(float), 0)
    calibrated_columns = [f"cal_{column}" for column in QCOLS]
    output[calibrated_columns] = np.sort(output[calibrated_columns].to_numpy(float), axis=1)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve-root", type=Path, required=True)
    parser.add_argument("--prediction-output", type=Path, required=True)
    parser.add_argument("--calibration-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--parallel-models", type=int, default=3)
    parser.add_argument("--threads-per-model", type=int, default=10)
    parser.add_argument("--window-days", type=int, default=30)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)
    wide = load_wide(args.curve_root)
    train, features = expand_split(wide, "train")
    tuning, tuning_features = expand_split(wide, "tuning")
    calibration, calibration_features = expand_split(wide, "calibration")
    test, test_features = expand_split(wide, "test")
    if not (features == tuning_features == calibration_features == test_features):
        raise RuntimeError("Feature contracts differ across chronological partitions")
    fit = pd.concat([train, tuning], ignore_index=True)
    fit["site_id"] = fit.site_id.astype("category")
    fitted = Parallel(n_jobs=args.parallel_models)(delayed(fit_quantile)(q, fit, features, args.iterations, args.threads_per_model) for q in QUANTILES)
    models = {quantile: model for quantile, model, _ in fitted}
    runtimes = {str(quantile): runtime for quantile, _, runtime in fitted}
    calibration_prediction = predict(models, calibration, features)
    test_prediction = predict(models, test, features)
    calibrated = causal_calibrate(calibration_prediction, test_prediction, args.window_days)
    args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    calibrated.to_parquet(args.prediction_output, index=False)
    calibration_prediction.to_parquet(args.calibration_output, index=False)
    result = {
        "dataset": "EMSx daily complete 96-step curves", "started_utc": started.isoformat(), "completed_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv), "method": "Global residual quantile LightGBM with increasing rearrangement and causal rolling quantile calibration",
        "selection_contract": "Point-model structure selected on training and tuning only. The same structure and iteration count are frozen for all quantiles. Calibration is first accessed after fitting.",
        "calibration_contract": "For each system, delivery step, and quantile, use normalized errors from the preceding 30 days whose valid times are no later than the new issue time.",
        "quantiles": QUANTILES, "iterations": args.iterations, "window_days": args.window_days, "seed": SEED,
        "rows": {"training": len(train), "tuning": len(tuning), "calibration": len(calibration), "test": len(test)},
        "fit_runtime_seconds_by_quantile": runtimes, "raw_test": metrics(calibrated), "causally_calibrated_test": metrics(calibrated, "cal_"),
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "lightgbm": lgb.__version__},
    }
    args.summary_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
