"""Fit conventional matched-input EMSx forecasting baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from run_emsx_qgbm import load_panel, prepare


from emsx_paths import WORK_ROOT as ROOT
RESULTS = ROOT / "results"


def matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    output = frame[features].copy()
    output["site_id"] = output["site_id"].astype(str)
    return output


def main() -> None:
    data, _, features = prepare(load_panel(96), history_lags=15)
    sets = {name: data.loc[data["split"] == name].dropna(subset=features + ["target_norm"]).copy() for name in ("train", "validation", "test")}
    train, test = sets["train"], sets["test"]
    numeric = [feature for feature in features if feature != "site_id"]
    transformer = ColumnTransformer([("numeric", StandardScaler(), numeric), ("site", OneHotEncoder(handle_unknown="ignore"), ["site_id"])])
    ridge = make_pipeline(transformer, Ridge(alpha=1.0))
    ridge.fit(matrix(train, features), train.target_norm)
    ridge_prediction = np.clip(ridge.predict(matrix(test, features)), 0, None) * test.scale_kwh.to_numpy()
    rng = np.random.default_rng(20260827)
    index = rng.choice(len(train), size=min(400000, len(train)), replace=False)
    rf_features = matrix(train.iloc[index], features)
    forest = RandomForestRegressor(n_estimators=160, min_samples_leaf=20, max_features=.8, max_samples=.75, n_jobs=32, random_state=20260827, verbose=1)
    forest.fit(rf_features, train.iloc[index].target_norm)
    forest_prediction = np.clip(forest.predict(matrix(test, features)), 0, None) * test.scale_kwh.to_numpy()
    target = test.actual_pv_kwh.to_numpy(); vendor = test.forecast_pv_kwh.to_numpy(); persistence = np.maximum(test.issue_actual_pv_kwh.to_numpy(), 0)
    predictions = test[["issue_time", "site_id", "actual_pv_kwh", "forecast_pv_kwh", "issue_actual_pv_kwh", "scale_kwh"]].copy()
    predictions["persistence_prediction_kwh"] = persistence
    predictions["ridge_prediction_kwh"] = ridge_prediction
    predictions["random_forest_prediction_kwh"] = forest_prediction
    scores = {"vendor": float(np.abs(target-vendor).mean()), "persistence": float(np.abs(target-persistence).mean()), "ridge": float(np.abs(target-ridge_prediction).mean()), "random_forest": float(np.abs(target-forest_prediction).mean())}
    result = {"dataset": "EMSx", "lead_hours": 24., "feature_contract": "Vendor forecast, 16 preceding 15-minute PV values, UTC calendar features, and site identity. All models use the same chronological 70/10/20 split and shared test observations.", "random_forest": {"training_rows": int(len(index)), "trees": 160, "minimum_leaf": 20}, "test_rows": int(len(test)), "mae_kwh": scores}
    RESULTS.mkdir(exist_ok=True); predictions.to_parquet(RESULTS / "emsx_sklearn_baselines_lead_96.parquet", index=False); (RESULTS / "emsx_sklearn_baselines_lead_96.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
