"""Shared loading and expansion utilities for the EMSx daily-curve benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FORECAST_COLUMNS = [f"forecast_{step:02d}_kwh" for step in range(96)]
TARGET_COLUMNS = [f"target_{step:02d}" for step in range(96)]
SEASONAL_NAIVE_COLUMNS = [f"seasonal_naive_{step:02d}_kwh" for step in range(96)]
HISTORY_COLUMNS = ["issue_actual_pv_kwh", *[f"history_lag_{lag:02d}_kwh" for lag in range(1, 16)]]
ANCHOR_STEPS = (0, 3, 7, 15, 31, 47, 63, 79, 95)


def load_wide(root: Path) -> pd.DataFrame:
    paths = sorted(root.glob("site=*/curves.parquet"), key=lambda path: int(path.parent.name.split("=")[1]))
    if not paths:
        raise FileNotFoundError(f"No daily curve files found below {root}")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    frame["issue_time"] = pd.to_datetime(frame.issue_time, utc=True)
    frame["site_id"] = frame.site_id.astype(int)
    if frame.duplicated(["site_id", "issue_time"]).any():
        raise RuntimeError("Duplicate site and issue-time keys in daily curve data")
    return frame.sort_values(["site_id", "issue_time"]).reset_index(drop=True)


def expand_split(wide: pd.DataFrame, split: str) -> tuple[pd.DataFrame, list[str]]:
    source = wide.loc[(wide.split == split) & wide.complete_case].copy().reset_index(drop=True)
    count = len(source)
    leads = np.tile(np.arange(1, 97, dtype=np.int16), count)
    scale = np.repeat(source.scale_kwh.to_numpy(np.float32), 96)
    issue = pd.Series(np.repeat(source.issue_time.to_numpy(), 96))
    output = pd.DataFrame({
        "site_id": np.repeat(source.site_id.to_numpy(np.int16), 96),
        "issue_time": pd.to_datetime(issue, utc=True),
        "valid_time": pd.to_datetime(issue, utc=True) + pd.to_timedelta(leads * 15, unit="m"),
        "delivery_step": leads,
        "scale_kwh": scale,
        "actual_pv_kwh": source[TARGET_COLUMNS].to_numpy(np.float32).reshape(-1),
        "vendor_prediction_kwh": source[FORECAST_COLUMNS].to_numpy(np.float32).reshape(-1),
        "seasonal_naive_prediction_kwh": source[SEASONAL_NAIVE_COLUMNS].to_numpy(np.float32).reshape(-1),
        "persistence_prediction_kwh": np.repeat(source.issue_actual_pv_kwh.to_numpy(np.float32), 96),
    })
    output["forecast_norm"] = output.vendor_prediction_kwh / scale
    for column in HISTORY_COLUMNS:
        output[f"{column}_norm"] = np.repeat(source[column].to_numpy(np.float32), 96) / scale
    for step in ANCHOR_STEPS:
        output[f"curve_anchor_{step:02d}_norm"] = np.repeat(source[f"forecast_{step:02d}_kwh"].to_numpy(np.float32), 96) / scale
    for column in ("curve_mean_kwh", "curve_std_kwh", "curve_max_kwh", "curve_energy_kwh", "curve_max_abs_ramp_kwh", "recent_24h_vendor_error_kwh"):
        output[f"{column}_norm"] = np.repeat(source[column].to_numpy(np.float32), 96) / scale
    output["curve_peak_step_norm"] = np.repeat(source.curve_peak_step.to_numpy(np.float32), 96) / 96
    output["lead_sin"] = np.sin(2 * np.pi * leads / 96).astype(np.float32)
    output["lead_cos"] = np.cos(2 * np.pi * leads / 96).astype(np.float32)
    output["valid_doy_sin"] = np.sin(2 * np.pi * output.valid_time.dt.dayofyear / 366).astype(np.float32)
    output["valid_doy_cos"] = np.cos(2 * np.pi * output.valid_time.dt.dayofyear / 366).astype(np.float32)
    output["target_norm"] = output.actual_pv_kwh / scale
    output["residual_norm"] = (output.actual_pv_kwh - output.vendor_prediction_kwh) / scale
    output["site_id"] = output.site_id.astype("category")
    features = [
        "forecast_norm", "delivery_step", "lead_sin", "lead_cos", "valid_doy_sin", "valid_doy_cos", "site_id",
        *[f"{column}_norm" for column in HISTORY_COLUMNS],
        *[f"curve_anchor_{step:02d}_norm" for step in ANCHOR_STEPS],
        "curve_mean_kwh_norm", "curve_std_kwh_norm", "curve_max_kwh_norm", "curve_energy_kwh_norm",
        "curve_max_abs_ramp_kwh_norm", "curve_peak_step_norm", "recent_24h_vendor_error_kwh_norm",
    ]
    if output[features + ["target_norm", "residual_norm"]].isna().any().any():
        raise RuntimeError(f"Nonfinite model data remain in complete {split} curves")
    return output, features


def point_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    target = frame.actual_pv_kwh.to_numpy(float)
    error = np.abs(target - prediction)
    per_curve = pd.DataFrame({"site_id": frame.site_id.astype(int), "issue_time": frame.issue_time, "error": error}).groupby(["site_id", "issue_time"], observed=True).error.mean()
    per_system = per_curve.groupby("site_id").mean()
    return {
        "delivery_rows": int(len(frame)),
        "curves": int(frame[["site_id", "issue_time"]].drop_duplicates().shape[0]),
        "systems": int(frame.site_id.nunique()),
        "pooled_mae_kwh": float(error.mean()),
        "equal_curve_mae_kwh": float(per_curve.mean()),
        "equal_system_mae_kwh": float(per_system.mean()),
        "training_scale_normalized_mae": float(np.mean(error / frame.scale_kwh.to_numpy(float))),
    }
