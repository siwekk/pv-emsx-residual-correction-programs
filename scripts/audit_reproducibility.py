"""Audit the EMSx forecast contract, partitions, prediction keys, and frozen metrics.

This script performs no model fitting.  It reads the authoritative EMSx files and
the saved predictions, then writes a machine-readable audit report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260906
KEYS = ["issue_time", "site_id", "actual_pv_kwh", "forecast_pv_kwh", "scale_kwh"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def key_digest(frame: pd.DataFrame) -> str:
    keys = frame[["site_id", "issue_time"]].copy()
    keys["issue_time"] = pd.to_datetime(keys["issue_time"], utc=True).astype("int64")
    keys = keys.sort_values(["site_id", "issue_time"])
    return hashlib.sha256(pd.util.hash_pandas_object(keys, index=False).values.tobytes()).hexdigest()


def split_label(rank: pd.Series, boundaries: tuple[float, ...], labels: tuple[str, ...]) -> np.ndarray:
    conditions = [rank <= boundary for boundary in boundaries]
    return np.select(conditions, labels[:-1], default=labels[-1])


def split_trace(panel_root: Path) -> dict:
    rows = []
    for path in sorted(panel_root.glob("site=*/panel.parquet"), key=lambda p: int(p.parent.name.split("=")[1])):
        frame = pd.read_parquet(path, filters=[("lead_steps", "=", 96)])
        frame["issue_time"] = pd.to_datetime(frame["issue_time"], utc=True)
        frame = frame.drop_duplicates("issue_time").sort_values("issue_time").copy()
        rank = frame["issue_time"].rank(pct=True, method="first")
        frame["split_65_5_10_20"] = split_label(rank, (.65, .70, .80), ("train", "tuning", "calibration", "test"))
        frame["split_70_10_20"] = split_label(rank, (.70, .80), ("train", "validation", "test"))
        item = {"site_id": int(frame.site_id.iloc[0]), "eligible_rows_before_history": int(len(frame))}
        for column in ("split_65_5_10_20", "split_70_10_20"):
            item[column] = {}
            for name, block in frame.groupby(column, sort=False):
                item[column][str(name)] = {
                    "rows": int(len(block)),
                    "first_issue_time": block.issue_time.min().isoformat(),
                    "last_issue_time": block.issue_time.max().isoformat(),
                }
        rows.append(item)
    totals = {}
    for scheme in ("split_65_5_10_20", "split_70_10_20"):
        names = sorted({name for row in rows for name in row[scheme]})
        totals[scheme] = {name: sum(row[scheme].get(name, {}).get("rows", 0) for row in rows) for name in names}
    return {"definition": "Per-site chronological percentile rank with method='first'.", "sites": rows, "totals_before_history_filter": totals}


def history_and_complete_case_audit(panel_root: Path, curve_root: Path) -> dict:
    rows = []
    test_keys = []
    for path in sorted(panel_root.glob("site=*/panel.parquet"), key=lambda p: int(p.parent.name.split("=")[1])):
        frame = pd.read_parquet(path, filters=[("lead_steps", "=", 96)])
        frame["issue_time"] = pd.to_datetime(frame["issue_time"], utc=True)
        frame = frame.drop_duplicates("issue_time").sort_values("issue_time").copy()
        rank = frame.issue_time.rank(pct=True, method="first")
        frame["split"] = split_label(rank, (.65, .70, .80), ("train", "tuning", "calibration", "test"))
        observed = frame.set_index("issue_time")["issue_actual_pv_kwh"]
        history = []
        for lag in range(1, 16):
            name = f"history_lag_{lag}"
            frame[name] = observed.reindex(frame.issue_time - pd.Timedelta(minutes=15 * lag)).to_numpy()
            history.append(name)
        site_id = int(frame.site_id.iloc[0])
        curves = pd.read_parquet(curve_root / f"site={site_id}.parquet")
        curves["issue_time"] = pd.to_datetime(curves["issue_time"], utc=True)
        curve_columns = [column for column in curves if column.startswith("curve_")]
        frame = frame.merge(curves[["issue_time", *curve_columns]], on="issue_time", how="left", validate="one_to_one")
        required = ["actual_pv_kwh", "forecast_pv_kwh", "issue_actual_pv_kwh", *history, *curve_columns]
        frame["complete"] = frame[required].notna().all(axis=1)
        item = {"site_id": site_id, "splits": {}}
        for name, block in frame.groupby("split", sort=False):
            complete = block.loc[block.complete]
            item["splits"][str(name)] = {
                "rows_before_filter": int(len(block)),
                "complete_rows": int(len(complete)),
                "removed_rows": int(len(block) - len(complete)),
                "first_complete_issue_time": complete.issue_time.min().isoformat(),
                "last_complete_issue_time": complete.issue_time.max().isoformat(),
            }
            if name == "test":
                test_keys.append(complete[["site_id", "issue_time"]])
        rows.append(item)
    totals = {}
    for name in ("train", "tuning", "calibration", "test"):
        totals[name] = {
            "rows_before_filter": sum(row["splits"][name]["rows_before_filter"] for row in rows),
            "complete_rows": sum(row["splits"][name]["complete_rows"] for row in rows),
            "removed_rows": sum(row["splits"][name]["removed_rows"] for row in rows),
        }
    complete_test = pd.concat(test_keys, ignore_index=True)
    return {
        "history_definition": "issue_actual_pv_kwh is the interval ending at issue time; history_lag_1 through history_lag_15 end 15 through 225 minutes before issue time.",
        "all_15_lag_timestamps_strictly_precede_issue_time": True,
        "latest_history_lag_offset_minutes": -15,
        "earliest_history_lag_offset_minutes": -225,
        "complete_case_definition": "Nonmissing target, target forecast, issue-time actual, 15 strictly prior actuals, and all enhanced curve features.",
        "totals": totals,
        "complete_test_key_digest": key_digest(complete_test),
        "sites": rows,
    }


def raw_and_panel_contract(raw_root: Path, panel_root: Path) -> dict:
    record = json.loads((raw_root / "zenodo_record.json").read_text(encoding="utf-8"))
    description = record.get("metadata", {}).get("description", "")
    source_statements = {
        "row_energy_window": "last 15 minutes" in description,
        "pv_00_next_interval": "pv_00" in description and "next 15 minutes" in description,
        "pv_columns_kwh": "pv_XX</code>&nbsp;in kWh" in description,
    }
    checks = []
    for site in range(1, 71):
        raw_path = raw_root / f"{site}.csv.gz"
        raw = pd.read_csv(raw_path, sep=";", usecols=["timestamp", "site_id", "actual_pv", "pv_00", "pv_95"])
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
        raw = raw.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
        intervals = raw.timestamp.diff().dropna().dt.total_seconds().div(60)
        observed = raw.set_index("timestamp")["actual_pv"]
        panel_path = panel_root / f"site={site}" / "panel.parquet"
        panel = pd.read_parquet(panel_path, filters=[("lead_steps", "=", 96)])
        panel["issue_time"] = pd.to_datetime(panel["issue_time"], utc=True)
        panel["valid_time"] = pd.to_datetime(panel["valid_time"], utc=True)
        expected_actual = observed.reindex(panel.valid_time).to_numpy()
        raw_at_issue = raw.set_index("timestamp").reindex(panel.issue_time)
        checks.append({
            "site_id": site,
            "raw_rows_after_timestamp_deduplication": int(len(raw)),
            "panel_lead_96_rows": int(len(panel)),
            "raw_duplicate_timestamps_removed": int(pd.read_csv(raw_path, sep=";", usecols=["timestamp"]).duplicated("timestamp").sum()),
            "timestamps_are_strictly_increasing_after_sort": bool((intervals > 0).all()),
            "modal_interval_minutes": float(intervals.mode().iloc[0]),
            "fifteen_minute_intervals": int((intervals == 15).sum()),
            "non_fifteen_minute_intervals": int((intervals != 15).sum()),
            "raw_site_id_consistent": bool((raw.site_id == site).all()),
            "issue_time_matches_raw_row": bool(raw_at_issue.site_id.notna().all()),
            "issue_actual_matches_raw_actual": bool(np.allclose(panel.issue_actual_pv_kwh, raw_at_issue.actual_pv, equal_nan=True)),
            "forecast_matches_raw_pv_95": bool(np.allclose(panel.forecast_pv_kwh, raw_at_issue.pv_95, equal_nan=True)),
            "valid_time_is_issue_plus_24h": bool((panel.valid_time == panel.issue_time + pd.Timedelta(hours=24)).all()),
            "target_matches_raw_at_valid_time": bool(np.allclose(panel.actual_pv_kwh, expected_actual, equal_nan=True)),
            "first_issue_time": panel.issue_time.min().isoformat(),
            "last_issue_time": panel.issue_time.max().isoformat(),
        })
    return {
        "source": {
            "zenodo_record": record.get("id"),
            "title": record.get("metadata", {}).get("title"),
            "related_publication_doi": "https://doi.org/10.1007/s12667-020-00417-5",
            "description_checks": source_statements,
        },
        "interpretation": {
            "timestamp_encoding": "Parsed as offset-aware ISO 8601 and converted to UTC.",
            "timestamp": "End of the 15-minute interval represented by actual_pv.",
            "actual_pv": "PV energy produced over the 15 minutes ending at timestamp, in kWh.",
            "pv_00": "PV energy forecast for the next 15-minute interval, ending timestamp + 15 minutes, in kWh.",
            "pv_95": "PV energy forecast for the interval ending timestamp + 24 hours, in kWh.",
        },
        "site_checks": checks,
        "all_contract_checks_pass": bool(all(all(item[key] for key in (
            "timestamps_are_strictly_increasing_after_sort", "raw_site_id_consistent", "issue_time_matches_raw_row", "issue_actual_matches_raw_actual", "forecast_matches_raw_pv_95",
            "valid_time_is_issue_plus_24h", "target_matches_raw_at_valid_time")) for item in checks)),
    }


def prediction_audit(prediction_root: Path) -> dict:
    specifications = {
        "lightgbm": ("emsx_tuned_residual_qgbm_lead_96.parquet", "tuned_residual_qgbm_prediction_kwh"),
        "xgboost": ("emsx_xgboost_lead_96.parquet", "xgboost_prediction_kwh"),
        "random_forest": ("emsx_curve_random_forest_lead_96.parquet", "curve_random_forest_prediction_kwh"),
        "gru": ("emsx_gru_lead_96.parquet", "gru_prediction_kwh"),
        "tcn": ("emsx_tcn_lead_96.parquet", "tcn_prediction_kwh"),
    }
    report = {}
    reference_digest = None
    for name, (filename, prediction_column) in specifications.items():
        path = prediction_root / filename
        frame = pd.read_parquet(path)
        frame["issue_time"] = pd.to_datetime(frame["issue_time"], utc=True)
        digest = key_digest(frame)
        if reference_digest is None:
            reference_digest = digest
        target = frame.actual_pv_kwh.to_numpy(float)
        vendor = frame.forecast_pv_kwh.to_numpy(float)
        prediction = frame[prediction_column].to_numpy(float)
        scale = frame.scale_kwh.to_numpy(float)
        per_site = frame.assign(
            vendor_abs_error=np.abs(target - vendor),
            model_abs_error=np.abs(target - prediction),
        ).groupby("site_id", observed=True)[["vendor_abs_error", "model_abs_error"]].mean()
        report[name] = {
            "file": filename,
            "sha256": sha256(path),
            "rows": int(len(frame)),
            "sites": int(frame.site_id.nunique()),
            "duplicate_site_issue_keys": int(frame.duplicated(["site_id", "issue_time"]).sum()),
            "key_digest": digest,
            "same_keys_as_reference": digest == reference_digest,
            "first_issue_time": frame.issue_time.min().isoformat(),
            "last_issue_time": frame.issue_time.max().isoformat(),
            "pooled_mae_kwh": float(np.abs(target - prediction).mean()),
            "equal_system_mae_kwh": float(per_site.model_abs_error.mean()),
            "training_scale_normalized_mae": float(np.mean(np.abs(target - prediction) / scale)),
            "vendor_pooled_mae_kwh": float(np.abs(target - vendor).mean()),
            "vendor_equal_system_mae_kwh": float(per_site.vendor_abs_error.mean()),
            "vendor_training_scale_normalized_mae": float(np.mean(np.abs(target - vendor) / scale)),
        }
    return {"reference": "lightgbm", "all_key_sets_identical": all(item["same_keys_as_reference"] for item in report.values()), "models": report}


def probabilistic_audit(prediction_root: Path) -> dict:
    path = prediction_root / "emsx_enhanced_probabilistic_qgbm_lead_96.parquet"
    frame = pd.read_parquet(path)
    y = frame.actual_pv_kwh.to_numpy(float)

    def scores(lower_name: str, upper_name: str) -> dict:
        lower = frame[lower_name].to_numpy(float)
        upper = frame[upper_name].to_numpy(float)
        miss = 20 * np.maximum(lower - y, 0) + 20 * np.maximum(y - upper, 0)
        return {
            "coverage": float(np.mean((y >= lower) & (y <= upper))),
            "mean_width_kwh": float(np.mean(upper - lower)),
            "interval_score_kwh": float(np.mean(upper - lower + miss)),
        }

    return {
        "file": path.name,
        "sha256": sha256(path),
        "rows": int(len(frame)),
        "median_mae_kwh": float(np.mean(np.abs(y - frame.median_kwh.to_numpy(float)))),
        "raw_90_interval": scores("low_raw_kwh", "high_raw_kwh"),
        "adaptive_90_interval": scores("low_adaptive_kwh", "high_adaptive_kwh"),
        "adaptive_radius_nonnegative": bool((frame.radius_norm >= 0).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--curve-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)
    result = {
        "audit": "EMSx reproducibility audit",
        "started_utc": started.isoformat(),
        "completed_utc": None,
        "command": " ".join(__import__("sys").argv),
        "sources": {name: str(getattr(args, name).resolve()) for name in ("raw_root", "panel_root", "curve_root", "prediction_root")},
        "random_seed": SEED,
        "software": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
        "forecast_contract": raw_and_panel_contract(args.raw_root, args.panel_root),
        "split_trace": split_trace(args.panel_root),
        "history_and_complete_cases": history_and_complete_case_audit(args.panel_root, args.curve_root),
        "prediction_audit": prediction_audit(args.prediction_root),
        "probabilistic_audit": probabilistic_audit(args.prediction_root),
    }
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    result["history_and_complete_cases"]["complete_test_matches_prediction_keys"] = (
        result["history_and_complete_cases"]["complete_test_key_digest"]
        == result["prediction_audit"]["models"]["lightgbm"]["key_digest"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "contract_pass": result["forecast_contract"]["all_contract_checks_pass"],
        "common_prediction_keys": result["prediction_audit"]["all_key_sets_identical"],
        "test_rows": result["prediction_audit"]["models"]["lightgbm"]["rows"],
    }, indent=2))


if __name__ == "__main__":
    main()
