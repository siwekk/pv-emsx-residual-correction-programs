"""Fit a random-forest comparator with the enhanced QGBM feature contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from run_emsx_tuned_qgbm import prepare_enhanced
from run_emsx_qgbm import score


from emsx_paths import WORK_ROOT as ROOT
RESULTS = ROOT / "results"


def table(frame, features):
    output = frame[features].copy(); output["site_id"] = output["site_id"].astype(int); return output


def main() -> None:
    data, features = prepare_enhanced()
    train = data.loc[data["split"] == "train"].dropna(subset=features + ["residual_norm"])
    test = data.loc[data["split"] == "test"].dropna(subset=features + ["residual_norm"])
    rng = np.random.default_rng(20260827); index = rng.choice(len(train), min(400000, len(train)), replace=False)
    model = RandomForestRegressor(n_estimators=160, min_samples_leaf=20, max_features=.8, max_samples=.75, n_jobs=32, random_state=20260827, verbose=1)
    model.fit(table(train.iloc[index], features), train.iloc[index].residual_norm)
    residual = model.predict(table(test, features)); prediction = np.clip(test.forecast_norm.to_numpy() + residual, 0, None) * test.scale_kwh.to_numpy()
    result = {"dataset": "EMSx", "lead_hours": 24., "method": "Random-forest residual corrector with matched enhanced feature contract", "training_rows": int(len(index)), "test": score(test, prediction)}
    RESULTS.mkdir(exist_ok=True); (RESULTS / "emsx_curve_random_forest_lead_96.json").write_text(json.dumps(result, indent=2)+"\n")
    output = test[["issue_time", "site_id", "actual_pv_kwh", "forecast_pv_kwh", "scale_kwh"]].copy(); output["curve_random_forest_prediction_kwh"] = prediction
    output.to_parquet(RESULTS / "emsx_curve_random_forest_lead_96.parquet", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
