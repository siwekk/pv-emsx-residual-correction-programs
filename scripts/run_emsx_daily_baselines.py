"""Evaluate transparent baselines on the frozen EMSx daily-curve sample."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from emsx_daily_curve_common import expand_split, load_wide, point_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve-root", type=Path, required=True)
    parser.add_argument("--prediction-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)
    test, _ = expand_split(load_wide(args.curve_root), "test")
    models = {
        "vendor": test.vendor_prediction_kwh.to_numpy(float),
        "persistence": test.persistence_prediction_kwh.to_numpy(float),
        "seasonal_naive": test.seasonal_naive_prediction_kwh.to_numpy(float),
    }
    output = test[["site_id", "issue_time", "valid_time", "delivery_step", "actual_pv_kwh", "scale_kwh"]].copy()
    for name, prediction in models.items():
        output[f"{name}_prediction_kwh"] = prediction
    result = {
        "dataset": "EMSx daily complete 96-step curves",
        "started_utc": started.isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "source": str(args.curve_root.resolve()),
        "sample": "Common complete test curves issued at 00:00 UTC.",
        "random_seed": None,
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "metrics": {name: point_metrics(test, prediction) for name, prediction in models.items()},
    }
    args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.prediction_output, index=False)
    args.summary_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
