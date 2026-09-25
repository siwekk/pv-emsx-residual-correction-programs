"""Audit and score daily probabilistic forecasts on a common EMSx test sample."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


QUANTILES = np.asarray([.05, .10, .20, .30, .40, .50, .60, .70, .80, .90, .95])
QCOLS = [f"q{int(round(100 * q)):02d}_kwh" for q in QUANTILES]
KEYS = ["site_id", "issue_time", "valid_time", "delivery_step"]


def pinball(y: np.ndarray, prediction: np.ndarray, quantile: float) -> np.ndarray:
    error = y - prediction
    return np.maximum(quantile * error, (quantile - 1.0) * error)


def interval_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> np.ndarray:
    return upper - lower + 2.0 / alpha * (lower - y) * (y < lower) + 2.0 / alpha * (y - upper) * (y > upper)


def audit_history(frame: pd.DataFrame, raw_root: Path, context_length: int) -> dict:
    origins = frame[["site_id", "issue_time"]].drop_duplicates().sort_values(["site_id", "issue_time"])
    target_missing = []
    vendor_missing = []
    for site_id, block in origins.groupby("site_id", sort=False):
        raw = pd.read_csv(raw_root / f"{int(site_id)}.csv.gz", sep=";", usecols=["timestamp", "actual_pv", "pv_00"])
        raw["timestamp"] = pd.to_datetime(raw.timestamp, utc=True)
        raw = raw.drop_duplicates("timestamp", keep="last").set_index("timestamp")
        issue_times = pd.DatetimeIndex(pd.to_datetime(block.issue_time, utc=True))
        timeline = pd.date_range(
            start=issue_times.min() - pd.Timedelta(minutes=15 * context_length),
            end=issue_times.max(), freq="15min", tz="UTC",
        )
        target_mask = raw.actual_pv.reindex(timeline).isna().to_numpy(np.int64)
        vendor_mask = raw.pv_00.reindex(timeline - pd.Timedelta(minutes=15)).isna().to_numpy(np.int64)
        target_prefix = np.concatenate([[0], np.cumsum(target_mask)])
        vendor_prefix = np.concatenate([[0], np.cumsum(vendor_mask)])
        ends = timeline.get_indexer(issue_times) + 1
        starts = ends - context_length
        target_missing.extend((target_prefix[ends] - target_prefix[starts]).tolist())
        vendor_missing.extend((vendor_prefix[ends] - vendor_prefix[starts]).tolist())
    target_missing = np.asarray(target_missing)
    vendor_missing = np.asarray(vendor_missing)
    return {
        "context_length": context_length,
        "curves_with_missing_target_history": int(np.sum(target_missing > 0)),
        "curves_with_missing_vendor_history": int(np.sum(vendor_missing > 0)),
        "target_history_missing_values": int(target_missing.sum()),
        "vendor_history_missing_values": int(vendor_missing.sum()),
        "maximum_missing_target_values_per_curve": int(target_missing.max(initial=0)),
        "maximum_missing_vendor_values_per_curve": int(vendor_missing.max(initial=0)),
    }


def evaluate(path: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_parquet(path)
    missing_columns = sorted(set(KEYS + ["actual_pv_kwh", "scale_kwh"] + QCOLS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{path} lacks columns: {missing_columns}")
    frame["issue_time"] = pd.to_datetime(frame.issue_time, utc=True)
    frame["valid_time"] = pd.to_datetime(frame.valid_time, utc=True)
    raw_qvalues = frame[QCOLS].to_numpy(float)
    y = frame.actual_pv_kwh.to_numpy(float)
    scale = frame.scale_kwh.to_numpy(float)
    finite_by_column = {column: int(np.isfinite(frame[column].to_numpy(float)).sum()) for column in ["actual_pv_kwh", "scale_kwh"] + QCOLS}
    row_finite = np.isfinite(y) & np.isfinite(scale) & np.isfinite(raw_qvalues).all(axis=1)
    crossing_pairs = raw_qvalues[:, :-1] > raw_qvalues[:, 1:]
    qvalues = np.sort(raw_qvalues, axis=1)
    raw_losses = np.column_stack([pinball(y, raw_qvalues[:, index], quantile) for index, quantile in enumerate(QUANTILES)])
    losses = np.column_stack([pinball(y, qvalues[:, index], quantile) for index, quantile in enumerate(QUANTILES)])
    interval_terms = []
    for alpha, lower_col, upper_col in ((.8, 1, 9), (.6, 2, 8), (.4, 3, 7), (.2, 4, 6)):
        interval_terms.append((alpha / 2.0) * interval_score(y, qvalues[:, lower_col], qvalues[:, upper_col], alpha))
    wis = (0.5 * np.abs(y - qvalues[:, 5]) + np.sum(interval_terms, axis=0)) / 4.5
    is90 = interval_score(y, qvalues[:, 0], qvalues[:, 10], .1)
    result = {
        "file": str(path),
        "delivery_rows": int(len(frame)),
        "curves": int(frame[["site_id", "issue_time"]].drop_duplicates().shape[0]),
        "systems": int(frame.site_id.nunique()),
        "duplicate_key_rows": int(frame.duplicated(KEYS, keep=False).sum()),
        "fully_finite_rows": int(row_finite.sum()),
        "nonfinite_rows": int((~row_finite).sum()),
        "finite_values_by_column": finite_by_column,
        "rows_with_quantile_crossing": int(crossing_pairs.any(axis=1).sum()),
        "adjacent_quantile_crossings": int(crossing_pairs.sum()),
        "quantile_postprocessing": "Increasing rearrangement by sorting the eleven quantiles independently on each delivery row.",
        "median_mae_kwh": float(np.mean(np.abs(y[row_finite] - qvalues[row_finite, 5]))),
        "median_training_scale_normalized_mae": float(np.mean(np.abs(y[row_finite] - qvalues[row_finite, 5]) / scale[row_finite])),
        "raw_mean_pinball_kwh": float(raw_losses[row_finite].mean()),
        "mean_pinball_kwh": float(losses[row_finite].mean()),
        "pinball_kwh_by_quantile": {str(q): float(losses[row_finite, i].mean()) for i, q in enumerate(QUANTILES)},
        "central_grid_crps_approximation_kwh": float(2.0 * np.trapz(losses[row_finite].mean(axis=0), QUANTILES)),
        "weighted_interval_score_kwh": float(wis[row_finite].mean()),
        "central_90_interval_coverage": float(np.mean((y[row_finite] >= qvalues[row_finite, 0]) & (y[row_finite] <= qvalues[row_finite, 10]))),
        "central_90_interval_mean_width_kwh": float(np.mean(qvalues[row_finite, 10] - qvalues[row_finite, 0])),
        "central_90_interval_score_kwh": float(is90[row_finite].mean()),
    }
    return frame, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append")
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--context-length", type=int, default=672)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = args.label or [path.stem for path in args.prediction]
    if len(labels) != len(args.prediction):
        raise ValueError("Supply exactly one label for each prediction file")
    frames = {}
    results = {}
    for label, path in zip(labels, args.prediction):
        frames[label], results[label] = evaluate(path)
    reference_label = labels[0]
    reference_keys = frames[reference_label][KEYS].drop_duplicates()
    for label in labels:
        keys = frames[label][KEYS].drop_duplicates()
        comparison = reference_keys.merge(keys, on=KEYS, how="outer", indicator=True)
        results[label]["keys_missing_from_reference"] = int((comparison._merge == "right_only").sum())
        results[label]["reference_keys_missing_from_method"] = int((comparison._merge == "left_only").sum())
    history = audit_history(frames[reference_label], args.raw_root, args.context_length) if args.raw_root else None
    output = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference_method": reference_label,
        "quantile_grid": QUANTILES.tolist(),
        "metric_contract": "All scores use delivery rows with finite targets, scales, and all eleven reported quantiles. Quantiles are monotonically rearranged before scoring. Raw crossing counts and raw mean pinball loss are retained. The grid CRPS is twice the trapezoidal pinball integral from 0.05 to 0.95. WIS uses the median and central 20, 40, 60, and 80 percent intervals.",
        "history_audit": history,
        "methods": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
