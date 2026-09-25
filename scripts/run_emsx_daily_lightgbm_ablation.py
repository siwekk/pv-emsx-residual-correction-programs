"""Fit fixed-configuration ablations of the daily residual LightGBM model."""

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

from emsx_daily_curve_common import ANCHOR_STEPS, expand_split, load_wide, point_metrics


SEED = 20260906
BASE_FEATURES = [
    "forecast_norm", "delivery_step", "lead_sin", "lead_cos",
    "valid_doy_sin", "valid_doy_cos", "site_id",
]
CURVE_FEATURES = [
    *[f"curve_anchor_{step:02d}_norm" for step in ANCHOR_STEPS],
    "curve_mean_kwh_norm", "curve_std_kwh_norm", "curve_max_kwh_norm",
    "curve_energy_kwh_norm", "curve_max_abs_ramp_kwh_norm", "curve_peak_step_norm",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve-root", type=Path, required=True)
    parser.add_argument("--prediction-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=24)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)

    wide = load_wide(args.curve_root)
    train, all_features = expand_split(wide, "train")
    tuning, tuning_features = expand_split(wide, "tuning")
    test, test_features = expand_split(wide, "test")
    if all_features != tuning_features or all_features != test_features:
        raise RuntimeError("Feature contracts differ between chronological splits")
    fit = pd.concat([train, tuning], ignore_index=True)
    fit["site_id"] = fit.site_id.astype("category")

    feature_sets = {
        "target_step": BASE_FEATURES,
        "issued_curve": [*BASE_FEATURES, *CURVE_FEATURES],
    }
    parameters = dict(
        objective="regression_l1", n_estimators=4000, learning_rate=.02,
        num_leaves=255, min_child_samples=100, colsample_bytree=.9,
        subsample=.85, reg_lambda=2., reg_alpha=.05, n_jobs=args.threads,
        verbosity=-1, random_state=SEED,
    )
    output = test[[
        "site_id", "issue_time", "valid_time", "delivery_step", "actual_pv_kwh",
        "vendor_prediction_kwh", "scale_kwh",
    ]].copy()
    model_results = {}
    for name, features in feature_sets.items():
        before = time.perf_counter()
        model = lgb.LGBMRegressor(**parameters)
        model.fit(fit[features], fit.residual_norm, categorical_feature=["site_id"])
        fit_seconds = time.perf_counter() - before
        before = time.perf_counter()
        residual = model.predict(test[features])
        prediction = np.clip(test.forecast_norm.to_numpy() + residual, 0, None) * test.scale_kwh.to_numpy()
        inference_seconds = time.perf_counter() - before
        column = f"lightgbm_{name}_prediction_kwh"
        output[column] = prediction
        model_results[name] = {
            "features": features,
            "fit_seconds": float(fit_seconds),
            "inference_seconds": float(inference_seconds),
            "test": point_metrics(test, prediction),
            "feature_importance_gain": dict(zip(features, model.booster_.feature_importance(importance_type="gain").tolist())),
        }
        print(name, model_results[name]["test"], flush=True)

    result = {
        "dataset": "EMSx daily complete 96-step curves",
        "started_utc": started.isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "source": str(args.curve_root.resolve()),
        "method": "Fixed-configuration feature ablations of the selected residual LightGBM",
        "selection": "The hyperparameters and 4000-tree fit are inherited from the primary model selected before test evaluation. No ablation was tuned on test data.",
        "random_seed": SEED,
        "parameters": parameters,
        "models": model_results,
        "software": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "lightgbm": lgb.__version__,
        },
    }
    args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.prediction_output, index=False)
    args.summary_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
