"""Create a matched, site-aware comparison of EMSx benchmark models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyse_emsx_matched_comparison import bootstrap


from emsx_paths import WORK_ROOT as ROOT
RESULTS = ROOT / "results"
KEYS = ["issue_time", "site_id", "actual_pv_kwh", "forecast_pv_kwh", "scale_kwh"]


def main() -> None:
    qgbm = pd.read_parquet(RESULTS / "emsx_qgbm_lead_96_history_15.parquet")
    tcn = pd.read_parquet(RESULTS / "emsx_tcn_lead_96.parquet")
    conventional = pd.read_parquet(RESULTS / "emsx_sklearn_baselines_lead_96.parquet")
    neural = pd.read_parquet(RESULTS / "emsx_lstm_mlp_lead_96.parquet")
    data = qgbm.merge(tcn[KEYS + ["tcn_prediction_kwh"]], on=KEYS, validate="one_to_one")
    data = data.merge(conventional[KEYS + ["persistence_prediction_kwh", "ridge_prediction_kwh", "random_forest_prediction_kwh"]], on=KEYS, validate="one_to_one")
    data = data.merge(neural[KEYS + ["mlp_prediction_kwh", "lstm_prediction_kwh"]], on=KEYS, validate="one_to_one")
    models = {"vendor": "forecast_pv_kwh", "persistence": "persistence_prediction_kwh", "ridge": "ridge_prediction_kwh", "random_forest": "random_forest_prediction_kwh", "mlp": "mlp_prediction_kwh", "lstm": "lstm_prediction_kwh", "tcn": "tcn_prediction_kwh", "qgbm": "qgbm_prediction_kwh"}
    errors = {}
    for name, column in models.items():
        error_name = f"{name}_abs_error"; data[error_name] = np.abs(data.actual_pv_kwh-data[column]); errors[name] = float(data[error_name].mean())
    differences = []
    for name in ("vendor", "ridge", "mlp", "lstm", "tcn", "qgbm"):
        field = f"random_forest_gain_vs_{name}_kwh"; data[field] = data[f"{name}_abs_error"] - data["random_forest_abs_error"]; differences.append(field)
    result = {"dataset": "EMSx", "lead_hours": 24., "matched_rows": int(len(data)), "sites": int(data.site_id.nunique()), "method": "Hierarchical bootstrap of site-day mean absolute-error differences, resampling sites and days within sites, 2,000 draws.", "mae_kwh": errors, "random_forest_advantage_kwh": bootstrap(data, differences)}
    (RESULTS / "emsx_standard_benchmark_lead_96.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
