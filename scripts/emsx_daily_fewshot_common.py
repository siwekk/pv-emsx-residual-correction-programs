"""Shared data preparation and scoring for EMSx foundation model adaptation."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from emsx_daily_curve_common import FORECAST_COLUMNS, TARGET_COLUMNS


QUANTILES = np.asarray([.05, .10, .20, .30, .40, .50, .60, .70, .80, .90, .95])
QCOLS = [f"q{int(round(100 * q)):02d}_kwh" for q in QUANTILES]


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_origins(
    wide: pd.DataFrame,
    split: str,
    profiles_per_system: int | None,
) -> pd.DataFrame:
    selected = wide.loc[(wide.split == split) & wide.complete_case].copy()
    selected = selected.sort_values(["site_id", "issue_time"])
    if profiles_per_system is not None:
        selected = selected.groupby("site_id", observed=True, group_keys=False).tail(profiles_per_system)
    return selected.reset_index(drop=True)


def prepare_records(
    wide: pd.DataFrame,
    raw_root: Path,
    split: str,
    context_length: int,
    profiles_per_system: int | None = None,
    require_complete_context: bool = False,
    max_profiles: int | None = None,
) -> list[dict]:
    origins = select_origins(wide, split, profiles_per_system)
    if max_profiles is not None:
        origins = origins.head(max_profiles)
    records: list[dict] = []
    for site_id, site_origins in origins.groupby("site_id", observed=True, sort=False):
        raw = pd.read_csv(
            raw_root / f"{int(site_id)}.csv.gz",
            sep=";",
            usecols=["timestamp", "actual_pv", "pv_00"],
        )
        raw["timestamp"] = pd.to_datetime(raw.timestamp, utc=True)
        raw = raw.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
        actual = raw.set_index("timestamp").actual_pv
        one_step = raw.set_index("timestamp").pv_00
        for row in site_origins.itertuples(index=False):
            history_time = pd.date_range(
                end=row.issue_time,
                periods=context_length,
                freq="15min",
                tz="UTC",
            )
            future_time = pd.date_range(
                start=row.issue_time + pd.Timedelta(minutes=15),
                periods=96,
                freq="15min",
                tz="UTC",
            )
            history_target = actual.reindex(history_time).to_numpy(np.float32)
            history_vendor = one_step.reindex(
                history_time - pd.Timedelta(minutes=15)
            ).to_numpy(np.float32)
            if require_complete_context and (
                not np.isfinite(history_target).all()
                or not np.isfinite(history_vendor).all()
            ):
                continue
            records.append(
                {
                    "item_id": f"{int(site_id)}_{pd.Timestamp(row.issue_time).strftime('%Y%m%dT%H%M%SZ')}",
                    "site_id": int(site_id),
                    "issue_time": pd.Timestamp(row.issue_time),
                    "history_time": history_time,
                    "history_target": history_target,
                    "history_vendor": history_vendor,
                    "future_time": future_time,
                    "future_vendor": np.asarray(
                        [getattr(row, name) for name in FORECAST_COLUMNS],
                        dtype=np.float32,
                    ),
                    "actual": np.asarray(
                        [getattr(row, name) for name in TARGET_COLUMNS],
                        dtype=np.float32,
                    ),
                    "scale_kwh": float(row.scale_kwh),
                }
            )
    return records


def pinball(y: np.ndarray, prediction: np.ndarray, quantile: float) -> np.ndarray:
    error = y - prediction
    return np.maximum(quantile * error, (quantile - 1.0) * error)


def interval_score(
    y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float,
) -> np.ndarray:
    return (
        upper
        - lower
        + 2.0 / alpha * (lower - y) * (y < lower)
        + 2.0 / alpha * (y - upper) * (y > upper)
    )


def score_predictions(frame: pd.DataFrame) -> dict:
    raw_quantiles = frame[QCOLS].to_numpy(float)
    quantiles = np.sort(raw_quantiles, axis=1)
    target = frame.actual_pv_kwh.to_numpy(float)
    scale = frame.scale_kwh.to_numpy(float)
    finite = np.isfinite(target) & np.isfinite(scale) & np.isfinite(raw_quantiles).all(axis=1)
    target = target[finite]
    scale = scale[finite]
    raw_quantiles = raw_quantiles[finite]
    quantiles = quantiles[finite]
    metadata = frame.loc[finite, ["site_id", "issue_time"]].reset_index(drop=True)
    absolute_error = np.abs(target - quantiles[:, 5])
    grouped = metadata.assign(error=absolute_error).groupby(
        ["site_id", "issue_time"], observed=True
    ).error.mean()
    per_system = grouped.groupby("site_id").mean()
    losses = np.column_stack(
        [pinball(target, quantiles[:, index], q) for index, q in enumerate(QUANTILES)]
    )
    interval_terms = []
    for alpha, lower_col, upper_col in ((.8, 1, 9), (.6, 2, 8), (.4, 3, 7), (.2, 4, 6)):
        interval_terms.append(
            (alpha / 2.0)
            * interval_score(target, quantiles[:, lower_col], quantiles[:, upper_col], alpha)
        )
    wis = (0.5 * absolute_error + np.sum(interval_terms, axis=0)) / 4.5
    is90 = interval_score(target, quantiles[:, 0], quantiles[:, 10], .1)
    crossings = raw_quantiles[:, :-1] > raw_quantiles[:, 1:]
    grid_integral = (
        np.trapezoid(losses.mean(axis=0), QUANTILES)
        if hasattr(np, "trapezoid")
        else np.trapz(losses.mean(axis=0), QUANTILES)
    )
    return {
        "delivery_rows": int(len(frame)),
        "finite_delivery_rows": int(finite.sum()),
        "profiles": int(frame[["site_id", "issue_time"]].drop_duplicates().shape[0]),
        "systems": int(frame.site_id.nunique()),
        "rows_with_quantile_crossing": int(crossings.any(axis=1).sum()),
        "adjacent_quantile_crossings": int(crossings.sum()),
        "pooled_median_mae_kwh": float(absolute_error.mean()),
        "equal_profile_median_mae_kwh": float(grouped.mean()),
        "equal_system_median_mae_kwh": float(per_system.mean()),
        "training_scale_normalized_median_mae": float(np.mean(absolute_error / scale)),
        "mean_pinball_kwh": float(losses.mean()),
        "grid_crps_kwh": float(2.0 * grid_integral),
        "weighted_interval_score_kwh": float(wis.mean()),
        "central_90_interval_coverage": float(
            np.mean((target >= quantiles[:, 0]) & (target <= quantiles[:, 10]))
        ),
        "central_90_interval_width_kwh": float(
            np.mean(quantiles[:, 10] - quantiles[:, 0])
        ),
        "central_90_interval_score_kwh": float(is90.mean()),
    }
