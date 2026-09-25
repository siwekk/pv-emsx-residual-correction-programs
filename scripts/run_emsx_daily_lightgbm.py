"""Tune and evaluate the primary residual LightGBM on complete EMSx curves."""

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

from emsx_daily_curve_common import expand_split, load_wide, point_metrics


SEED = 20260906


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve-root", type=Path, required=True)
    parser.add_argument("--prediction-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)
    wide = load_wide(args.curve_root)
    train, features = expand_split(wide, "train")
    tuning, tuning_features = expand_split(wide, "tuning")
    if features != tuning_features:
        raise RuntimeError("Training and tuning feature contracts differ")
    candidates = [
        {"num_leaves": 63, "min_child_samples": 300, "learning_rate": .04},
        {"num_leaves": 127, "min_child_samples": 200, "learning_rate": .03},
        {"num_leaves": 255, "min_child_samples": 100, "learning_rate": .02},
    ]
    common = dict(
        objective="regression_l1", n_estimators=4000, colsample_bytree=.9, subsample=.85,
        reg_lambda=2., reg_alpha=.05, n_jobs=32, verbosity=-1, random_state=SEED,
    )
    trials = []
    best = None
    for candidate in candidates:
        before = time.perf_counter()
        model = lgb.LGBMRegressor(**common, **candidate)
        model.fit(
            train[features], train.residual_norm,
            categorical_feature=["site_id"],
            eval_set=[(tuning[features], tuning.residual_norm)],
            callbacks=[lgb.early_stopping(150, verbose=False)],
        )
        residual = model.predict(tuning[features])
        prediction = np.clip(tuning.forecast_norm.to_numpy() + residual, 0, None) * tuning.scale_kwh.to_numpy()
        mae = float(np.mean(np.abs(tuning.actual_pv_kwh.to_numpy() - prediction)))
        trial = {
            **candidate,
            "best_iteration": int(model.best_iteration_ or model.n_estimators),
            "tuning_mae_kwh": mae,
            "runtime_seconds": float(time.perf_counter() - before),
        }
        trials.append(trial)
        print(trial, flush=True)
        if best is None or mae < best[0]:
            best = (mae, trial)
    assert best is not None

    fit = pd.concat([train, tuning], ignore_index=True)
    fit["site_id"] = fit.site_id.astype("category")
    selected = best[1]
    final_parameters = {key: selected[key] for key in ("num_leaves", "min_child_samples", "learning_rate")}
    final = lgb.LGBMRegressor(**{**common, **final_parameters, "n_estimators": selected["best_iteration"]})
    before = time.perf_counter()
    final.fit(fit[features], fit.residual_norm, categorical_feature=["site_id"])
    fit_seconds = time.perf_counter() - before
    del train, tuning, fit

    test, test_features = expand_split(wide, "test")
    if features != test_features:
        raise RuntimeError("Test feature contract differs")
    before = time.perf_counter()
    residual = final.predict(test[features])
    inference_seconds = time.perf_counter() - before
    prediction = np.clip(test.forecast_norm.to_numpy() + residual, 0, None) * test.scale_kwh.to_numpy()
    result = {
        "dataset": "EMSx daily complete 96-step curves",
        "started_utc": started.isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "source": str(args.curve_root.resolve()),
        "method": "Global LightGBM residual correction of complete issued curves",
        "information_contract": "Issued target-step forecast, issue-time curve anchors and summaries, 16 causal target values, mature prior-day vendor error, delivery step, target calendar, and system identity.",
        "selection": "Candidates fitted on the 65 percent training block and selected only by MAE on the 5 percent tuning block; final model refitted on their union.",
        "random_seed": SEED,
        "features": features,
        "trials": trials,
        "selected": selected,
        "final_fit_seconds": float(fit_seconds),
        "test_inference_seconds": float(inference_seconds),
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "lightgbm": lgb.__version__},
        "test": point_metrics(test, prediction),
        "vendor_test": point_metrics(test, test.vendor_prediction_kwh.to_numpy(float)),
        "feature_importance_gain": dict(zip(features, final.booster_.feature_importance(importance_type="gain").tolist())),
    }
    output = test[["site_id", "issue_time", "valid_time", "delivery_step", "actual_pv_kwh", "vendor_prediction_kwh", "scale_kwh"]].copy()
    output["lightgbm_prediction_kwh"] = prediction
    args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.prediction_output, index=False)
    args.summary_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
