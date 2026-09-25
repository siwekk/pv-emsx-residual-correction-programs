"""Tune an XGBoost residual corrector with the enhanced EMSx feature set."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from run_emsx_tuned_qgbm import prepare_enhanced
from run_emsx_qgbm import score


from emsx_paths import WORK_ROOT as ROOT
RESULTS = ROOT / "results"


def matrix(frame, features):
    output = frame[features].copy(); output["site_id"] = output["site_id"].astype("category"); return output


def main() -> None:
    data, features = prepare_enhanced()
    sets = {name: data.loc[data["split"] == name].dropna(subset=features+["residual_norm"]).copy() for name in ("train", "validation", "test")}
    candidates = [{"max_depth": 8, "min_child_weight": 20}, {"max_depth": 10, "min_child_weight": 50}, {"max_depth": 12, "min_child_weight": 100}]
    best, selected, best_mae = None, None, float("inf"); trials = []
    for params in candidates:
        model = xgb.XGBRegressor(objective="reg:absoluteerror", n_estimators=4000, learning_rate=.03, subsample=.85, colsample_bytree=.9, reg_lambda=2., reg_alpha=.05, tree_method="hist", device="cuda", enable_categorical=True, early_stopping_rounds=150, n_jobs=32, random_state=20260828, **params)
        model.fit(matrix(sets["train"], features), sets["train"].residual_norm, eval_set=[(matrix(sets["validation"], features), sets["validation"].residual_norm)], verbose=False)
        prediction = np.clip(sets["validation"].forecast_norm.to_numpy()+model.predict(matrix(sets["validation"], features)), 0, None)*sets["validation"].scale_kwh.to_numpy()
        mae = float(np.abs(sets["validation"].actual_pv_kwh.to_numpy()-prediction).mean()); item = {**params, "best_iteration": int(model.best_iteration), "validation_mae_kwh": mae}; trials.append(item)
        if mae < best_mae: best, selected, best_mae = model, item, mae
    prediction = np.clip(sets["test"].forecast_norm.to_numpy()+best.predict(matrix(sets["test"], features)), 0, None)*sets["test"].scale_kwh.to_numpy()
    result = {"dataset": "EMSx", "lead_hours": 24., "method": "Tuned XGBoost residual corrector", "feature_contract": "Identical enhanced forecast-curve information set used by tuned QGBM.", "trials": trials, "selected": selected, "test": score(sets["test"], prediction)}
    output = sets["test"][["issue_time", "site_id", "actual_pv_kwh", "forecast_pv_kwh", "scale_kwh"]].reset_index(drop=True)
    output["xgboost_prediction_kwh"] = prediction
    RESULTS.mkdir(exist_ok=True); output.to_parquet(RESULTS/"emsx_xgboost_lead_96.parquet", index=False); (RESULTS/"emsx_xgboost_lead_96.json").write_text(json.dumps(result, indent=2)+"\n"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
