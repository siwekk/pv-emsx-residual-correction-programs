"""Audit EMSx point-error estimands and the legacy sMAPE calculation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260906
KEYS = ["site_id", "issue_time"]


def load_common(root: Path) -> pd.DataFrame:
    sources = {
        "lightgbm": ("emsx_tuned_residual_qgbm_lead_96.parquet", "tuned_residual_qgbm_prediction_kwh"),
        "xgboost": ("emsx_xgboost_lead_96.parquet", "xgboost_prediction_kwh"),
        "enhanced_random_forest": ("emsx_curve_random_forest_lead_96.parquet", "curve_random_forest_prediction_kwh"),
        "gru": ("emsx_gru_lead_96.parquet", "gru_prediction_kwh"),
        "tcn": ("emsx_tcn_lead_96.parquet", "tcn_prediction_kwh"),
    }
    base_file, base_column = sources.pop("lightgbm")
    base = pd.read_parquet(root / base_file)
    base["issue_time"] = pd.to_datetime(base.issue_time, utc=True)
    base = base.rename(columns={base_column: "lightgbm"})
    keep = [*KEYS, "actual_pv_kwh", "forecast_pv_kwh", "scale_kwh", "lightgbm"]
    base = base[keep]
    for name, (filename, column) in sources.items():
        other = pd.read_parquet(root / filename)
        other["issue_time"] = pd.to_datetime(other.issue_time, utc=True)
        check = base.merge(other[[*KEYS, "actual_pv_kwh", "forecast_pv_kwh", column]], on=KEYS, validate="one_to_one", suffixes=("", "_other"))
        if len(check) != len(base):
            raise RuntimeError(f"{name} does not have the common test sample")
        if not np.allclose(check.actual_pv_kwh, check.actual_pv_kwh_other, equal_nan=True):
            raise RuntimeError(f"{name} target values differ")
        if not np.allclose(check.forecast_pv_kwh, check.forecast_pv_kwh_other, equal_nan=True):
            raise RuntimeError(f"{name} vendor forecasts differ")
        base = check.drop(columns=["actual_pv_kwh_other", "forecast_pv_kwh_other"]).rename(columns={column: name})
    conventional = pd.read_parquet(root / "emsx_sklearn_baselines_lead_96.parquet")
    conventional["issue_time"] = pd.to_datetime(conventional.issue_time, utc=True)
    for name, column in {
        "persistence": "persistence_prediction_kwh",
        "ridge": "ridge_prediction_kwh",
        "random_forest": "random_forest_prediction_kwh",
    }.items():
        base = base.merge(conventional[[*KEYS, column]], on=KEYS, validate="one_to_one").rename(columns={column: name})
    neural = pd.read_parquet(root / "emsx_lstm_mlp_lead_96.parquet")
    neural["issue_time"] = pd.to_datetime(neural.issue_time, utc=True)
    base = base.merge(neural[[*KEYS, "mlp_prediction_kwh", "lstm_prediction_kwh"]], on=KEYS, validate="one_to_one")
    return base.rename(columns={"forecast_pv_kwh": "vendor", "mlp_prediction_kwh": "mlp", "lstm_prediction_kwh": "lstm"})


def metric_set(frame: pd.DataFrame, model: str) -> dict:
    y = frame.actual_pv_kwh.to_numpy(float)
    prediction = frame[model].to_numpy(float)
    error = np.abs(y - prediction)
    denominator = np.abs(y) + np.abs(prediction)
    terms = np.divide(2 * error, denominator, out=np.zeros_like(error), where=denominator > 0)
    site_mae = pd.Series(error, index=frame.index).groupby(frame.site_id).mean()
    by_site_total = pd.Series(error, index=frame.index).groupby(frame.site_id).sum()
    target_positive = y > 0
    return {
        "pooled_mae_kwh": float(error.mean()),
        "equal_system_mae_kwh": float(site_mae.mean()),
        "training_scale_normalized_mae": float(np.mean(error / frame.scale_kwh.to_numpy(float))),
        "largest_site_share_of_total_absolute_error": float(by_site_total.max() / by_site_total.sum()),
        "largest_error_contributor_site_id": int(by_site_total.idxmax()),
        "smape_all_rows_zero_over_zero_is_zero_pct": float(100 * terms.mean()),
        "smape_positive_denominator_only_pct": float(100 * terms[denominator > 0].mean()),
        "smape_positive_target_only_pct": float(100 * terms[target_positive].mean()),
        "zero_over_zero_rows": int((denominator == 0).sum()),
        "positive_target_rows": int(target_positive.sum()),
    }


def hierarchical_equal_system_bootstrap(frame: pd.DataFrame, comparisons: dict[str, tuple[str, str]], draws: int) -> dict:
    model_columns = {model for pair in comparisons.values() for model in pair}
    data = frame[["site_id", "issue_time", "actual_pv_kwh", *model_columns]].copy()
    data["day"] = data.issue_time.dt.floor("D")
    columns = []
    for name, (left, right) in comparisons.items():
        data[name] = np.abs(data.actual_pv_kwh - data[right]) - np.abs(data.actual_pv_kwh - data[left])
        columns.append(name)
    daily = data.groupby(["site_id", "day"], observed=True)[columns].mean().reset_index()
    sites = daily.site_id.unique()
    values = {site: daily.loc[daily.site_id == site, columns].to_numpy() for site in sites}
    rng = np.random.default_rng(SEED)
    samples = np.empty((draws, len(columns)))
    for draw in range(draws):
        selected_sites = rng.choice(sites, len(sites), replace=True)
        site_means = []
        for site in selected_sites:
            block = values[site]
            site_means.append(block[rng.integers(0, len(block), len(block))].mean(axis=0))
        samples[draw] = np.vstack(site_means).mean(axis=0)
    return {
        column: {
            "bootstrap_mean_kwh": float(samples[:, index].mean()),
            "ci95_low_kwh": float(np.quantile(samples[:, index], .025)),
            "ci95_high_kwh": float(np.quantile(samples[:, index], .975)),
        }
        for index, column in enumerate(columns)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)
    data = load_common(args.prediction_root)
    models = [column for column in data.columns if column not in {*KEYS, "actual_pv_kwh", "scale_kwh"}]
    report = {
        "audit": "EMSx estimand and sMAPE audit",
        "started_utc": started.isoformat(),
        "completed_utc": None,
        "command": " ".join(sys.argv),
        "source_files": sorted(path.name for path in args.prediction_root.glob("emsx*lead_96.parquet")),
        "common_sample_rows": int(len(data)),
        "systems": int(data.site_id.nunique()),
        "random_seed": SEED,
        "bootstrap_draws": args.bootstrap_draws,
        "software": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
        "estimands": {
            "pooled_mae_kwh": "Mean absolute error over all common test rows; systems contribute in proportion to their test-row counts and error magnitudes.",
            "equal_system_mae_kwh": "Unweighted mean of the 70 system-specific test MAEs; each observed system contributes equally.",
            "training_scale_normalized_mae": "Mean row error divided by the system 99.5th percentile calculated only from the deterministic model's training period.",
        },
        "metrics": {model: metric_set(data, model) for model in models},
        "hierarchical_equal_system_bootstrap": {
            "method": "Resample systems, then UTC issue days within each selected system; average days within system and systems equally.",
            "comparisons": hierarchical_equal_system_bootstrap(data, {
                "lightgbm_advantage_over_vendor_kwh": ("lightgbm", "vendor"),
                "lightgbm_advantage_over_xgboost_kwh": ("lightgbm", "xgboost"),
                "lightgbm_advantage_over_enhanced_random_forest_kwh": ("lightgbm", "enhanced_random_forest"),
            }, args.bootstrap_draws),
        },
        "smape_finding": "Legacy table values use all rows and define 0/0 as zero. Rankings change materially when zero-denominator rows are excluded or the sample is restricted to positive targets.",
    }
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(data), "models": len(models)}, indent=2))


if __name__ == "__main__":
    main()
