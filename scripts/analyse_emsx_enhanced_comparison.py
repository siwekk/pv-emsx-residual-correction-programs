"""Quantify the matched enhanced QGBM versus random-forest difference."""

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
    qgbm = pd.read_parquet(RESULTS / "emsx_tuned_residual_qgbm_lead_96.parquet")
    forest = pd.read_parquet(RESULTS / "emsx_curve_random_forest_lead_96.parquet")
    data = qgbm.merge(forest[KEYS + ["curve_random_forest_prediction_kwh"]], on=KEYS, validate="one_to_one")
    data["qgbm_abs_error"] = np.abs(data.actual_pv_kwh-data.tuned_residual_qgbm_prediction_kwh)
    data["forest_abs_error"] = np.abs(data.actual_pv_kwh-data.curve_random_forest_prediction_kwh)
    data["qgbm_gain_kwh"] = data.forest_abs_error-data.qgbm_abs_error
    result = {"dataset": "EMSx", "lead_hours": 24., "matched_rows": int(len(data)), "sites": int(data.site_id.nunique()), "qgbm_mae_kwh": float(data.qgbm_abs_error.mean()), "random_forest_mae_kwh": float(data.forest_abs_error.mean()), "qgbm_advantage_kwh": bootstrap(data, ["qgbm_gain_kwh"])["qgbm_gain_kwh"]}
    (RESULTS / "emsx_enhanced_comparison_lead_96.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
