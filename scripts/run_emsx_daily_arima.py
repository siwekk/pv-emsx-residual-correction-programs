"""Evaluate causal per-system ARIMA with Fourier daily seasonality on EMSx curves."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from emsx_daily_curve_common import TARGET_COLUMNS, load_wide, point_metrics


FOURIER_ORDER = 3
FIT_WINDOW_DAYS = 180
MAX_ITERATIONS = 50


def fourier(index: pd.DatetimeIndex) -> np.ndarray:
    quarter = index.as_unit("ns").asi8 / (15 * 60 * 10**9)
    return np.column_stack([
        function(2 * np.pi * harmonic * quarter / 96)
        for harmonic in range(1, FOURIER_ORDER + 1)
        for function in (np.sin, np.cos)
    ])


def evaluate_site(arguments: tuple[int, Path, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    site_id, raw_root, curves = arguments
    before = time.perf_counter()
    raw = pd.read_csv(raw_root / f"{site_id}.csv.gz", sep=";", usecols=["timestamp", "actual_pv"])
    raw["timestamp"] = pd.to_datetime(raw.timestamp, utc=True)
    actual = raw.drop_duplicates("timestamp", keep="last").set_index("timestamp").actual_pv.sort_index()
    tuning = curves.loc[(curves.split == "tuning") & curves.complete_case]
    test = curves.loc[(curves.split == "test") & curves.complete_case].sort_values("issue_time")
    fit_end = tuning.issue_time.max() + pd.Timedelta(hours=24)
    fit_start = fit_end - pd.Timedelta(days=FIT_WINDOW_DAYS)
    fit_index = pd.date_range(fit_start, fit_end, freq="15min", tz="UTC")
    scale = float(curves.scale_kwh.iloc[0])
    target = actual.reindex(fit_index) / scale
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = SARIMAX(
            target, exog=fourier(fit_index), order=(1, 0, 0), trend="c",
            enforce_stationarity=False, enforce_invertibility=False,
        )
        state = model.fit(disp=False, maxiter=MAX_ITERATIONS)
    converged = bool(state.mle_retvals.get("converged", False))
    fit_iterations = int(state.mle_retvals.get("iterations", -1))
    convergence_warnings = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
    current_end = fit_end
    outputs = []
    for row in test.itertuples(index=False):
        issue_time = pd.Timestamp(row.issue_time)
        if issue_time > current_end:
            update_index = pd.date_range(current_end + pd.Timedelta(minutes=15), issue_time, freq="15min", tz="UTC")
            state = state.append(actual.reindex(update_index) / scale, exog=fourier(update_index), refit=False)
            current_end = issue_time
        future_index = pd.date_range(issue_time + pd.Timedelta(minutes=15), periods=96, freq="15min", tz="UTC")
        prediction = np.maximum(np.asarray(state.forecast(steps=96, exog=fourier(future_index))), 0) * scale
        outputs.append(pd.DataFrame({
            "site_id": site_id, "issue_time": issue_time, "valid_time": future_index,
            "delivery_step": np.arange(1, 97, dtype=np.int16),
            "actual_pv_kwh": np.asarray([getattr(row, name) for name in TARGET_COLUMNS], dtype=np.float32),
            "scale_kwh": scale, "arima_prediction_kwh": prediction,
        }))
    result = {
        "site_id": site_id, "fit_rows": int(len(target)), "fit_observed_rows": int(target.notna().sum()),
        "fit_start": fit_start.isoformat(), "fit_end": fit_end.isoformat(), "test_curves": int(len(test)),
        "converged": converged, "convergence_warnings": convergence_warnings,
        "iterations": fit_iterations, "runtime_seconds": float(time.perf_counter() - before),
    }
    return pd.concat(outputs, ignore_index=True), result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--prediction-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-sites", type=int)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)
    wide = load_wide(args.curve_root)
    jobs = [(int(site_id), args.raw_root, block.copy()) for site_id, block in wide.groupby("site_id", observed=True, sort=False)]
    if args.max_sites is not None:
        jobs = jobs[:args.max_sites]
    outputs = []
    site_results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for output, result in executor.map(evaluate_site, jobs):
            outputs.append(output)
            site_results.append(result)
            print(result, flush=True)
    predictions = pd.concat(outputs, ignore_index=True).sort_values(["site_id", "issue_time", "delivery_step"])
    result = {
        "dataset": "EMSx daily complete 96-step curves", "started_utc": started.isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(), "command": " ".join(sys.argv),
        "source_curves": str(args.curve_root.resolve()), "source_raw": str(args.raw_root.resolve()),
        "method": "Per-system ARIMA(1,0,0) errors with three Fourier pairs for 96-step daily seasonality",
        "fit_contract": "Parameters fitted on the latest 180 days ending after the last tuning target. States are updated causally through calibration and prior test observations without parameter refitting.",
        "fit_window_days": FIT_WINDOW_DAYS, "fourier_order": FOURIER_ORDER, "maximum_iterations": MAX_ITERATIONS,
        "random_seed": None, "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "statsmodels": statsmodels.__version__},
        "test": point_metrics(predictions, predictions.arima_prediction_kwh.to_numpy(float)),
        "sites_converged": int(sum(item["converged"] for item in site_results)), "site_results": site_results,
    }
    args.prediction_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(args.prediction_output, index=False)
    args.summary_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
