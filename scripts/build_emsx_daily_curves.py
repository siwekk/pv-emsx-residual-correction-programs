"""Build one leakage-safe EMSx record per complete daily 96-step curve."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


FORECAST_COLUMNS = [f"pv_{step:02d}" for step in range(96)]
TARGET_COLUMNS = [f"target_{step:02d}" for step in range(96)]
SEASONAL_NAIVE_COLUMNS = [f"seasonal_naive_{step:02d}_kwh" for step in range(96)]
HISTORY_COLUMNS = ["issue_actual_pv_kwh", *[f"history_lag_{lag:02d}_kwh" for lag in range(1, 16)]]
SPLIT_NAMES = ("train", "tuning", "calibration", "test")


def assign_splits(frame: pd.DataFrame) -> pd.Series:
    rank = frame.issue_time.rank(pct=True, method="first")
    return pd.Series(
        np.select([rank <= .65, rank <= .70, rank <= .80], SPLIT_NAMES[:-1], default=SPLIT_NAMES[-1]),
        index=frame.index,
    )


def build_site(raw_path: Path, output_path: Path) -> dict:
    site_id = int(raw_path.name.split(".")[0])
    raw = pd.read_csv(raw_path, sep=";", usecols=["timestamp", "site_id", "actual_pv", *FORECAST_COLUMNS])
    raw["timestamp"] = pd.to_datetime(raw.timestamp, utc=True)
    raw = raw.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)
    observed = raw.set_index("timestamp").actual_pv
    daily = raw.loc[(raw.timestamp.dt.hour == 0) & (raw.timestamp.dt.minute == 0)].copy()
    daily = daily.rename(columns={"timestamp": "issue_time", **{name: f"forecast_{step:02d}_kwh" for step, name in enumerate(FORECAST_COLUMNS)}})

    issue_times = pd.DatetimeIndex(daily.issue_time)
    targets = np.column_stack([
        observed.reindex(issue_times + pd.Timedelta(minutes=15 * (step + 1))).to_numpy()
        for step in range(96)
    ])
    target_frame = pd.DataFrame(targets, columns=TARGET_COLUMNS, index=daily.index)
    seasonal_naive = np.column_stack([
        observed.reindex(issue_times - pd.Timedelta(hours=24) + pd.Timedelta(minutes=15 * (step + 1))).to_numpy()
        for step in range(96)
    ])
    seasonal_naive_frame = pd.DataFrame(seasonal_naive, columns=SEASONAL_NAIVE_COLUMNS, index=daily.index)
    history_frame = pd.DataFrame(
        {
            "issue_actual_pv_kwh": observed.reindex(issue_times).to_numpy(),
            **{
                f"history_lag_{lag:02d}_kwh": observed.reindex(issue_times - pd.Timedelta(minutes=15 * lag)).to_numpy()
                for lag in range(1, 16)
            },
        },
        index=daily.index,
    )
    daily = pd.concat([daily, target_frame, seasonal_naive_frame, history_frame], axis=1).copy()

    forecast_output = [f"forecast_{step:02d}_kwh" for step in range(96)]
    daily["forecast_complete"] = daily[forecast_output].notna().all(axis=1)
    daily["target_complete"] = daily[TARGET_COLUMNS].notna().all(axis=1)
    daily["history_complete"] = daily[HISTORY_COLUMNS].notna().all(axis=1)
    daily["seasonal_naive_complete"] = daily[SEASONAL_NAIVE_COLUMNS].notna().all(axis=1)
    eligible = daily.loc[daily.forecast_complete & daily.target_complete].copy()
    eligible["split"] = assign_splits(eligible)

    train_targets = eligible.loc[eligible.split == "train", TARGET_COLUMNS].to_numpy(float)
    scale = max(float(np.nanquantile(train_targets, .995)), 1e-3)
    eligible["scale_kwh"] = scale
    curve = eligible[forecast_output].to_numpy(float)
    eligible["curve_mean_kwh"] = curve.mean(axis=1)
    eligible["curve_std_kwh"] = curve.std(axis=1)
    eligible["curve_max_kwh"] = curve.max(axis=1)
    eligible["curve_energy_kwh"] = curve.sum(axis=1)
    eligible["curve_peak_step"] = curve.argmax(axis=1).astype(np.int16) + 1
    eligible["curve_max_abs_ramp_kwh"] = np.abs(np.diff(curve, axis=1)).max(axis=1)
    previous_issue = eligible.issue_time - pd.Timedelta(hours=24)
    prior_forecast_95 = raw.set_index("timestamp").pv_95.reindex(previous_issue).to_numpy()
    eligible["recent_24h_vendor_error_kwh"] = eligible.issue_actual_pv_kwh.to_numpy() - prior_forecast_95
    eligible["recent_error_available"] = np.isfinite(eligible.recent_24h_vendor_error_kwh)
    eligible["complete_case"] = eligible.history_complete & eligible.seasonal_naive_complete & eligible.recent_error_available

    ordered = [
        "site_id", "issue_time", "split", "scale_kwh", "complete_case",
        "forecast_complete", "target_complete", "history_complete", "seasonal_naive_complete", "recent_error_available",
        *HISTORY_COLUMNS, "recent_24h_vendor_error_kwh", "curve_mean_kwh", "curve_std_kwh",
        "curve_max_kwh", "curve_energy_kwh", "curve_peak_step", "curve_max_abs_ramp_kwh",
        *forecast_output, *TARGET_COLUMNS, *SEASONAL_NAIVE_COLUMNS,
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eligible[ordered].to_parquet(output_path, index=False)

    splits = {}
    for name in SPLIT_NAMES:
        block = eligible.loc[eligible.split == name]
        complete = block.loc[block.complete_case]
        splits[name] = {
            "eligible_curves": int(len(block)),
            "complete_curves": int(len(complete)),
            "removed_by_causal_input_availability": int(len(block) - len(complete)),
            "first_complete_issue_time": complete.issue_time.min().isoformat() if len(complete) else None,
            "last_complete_issue_time": complete.issue_time.max().isoformat() if len(complete) else None,
        }
    return {
        "site_id": site_id,
        "raw_file": raw_path.name,
        "raw_rows": int(len(raw)),
        "daily_midnight_rows": int(len(daily)),
        "eligible_complete_vendor_and_target_curves": int(len(eligible)),
        "training_q995_scale_kwh": scale,
        "splits": splits,
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sites", default="1-70", help="Inclusive range such as 1-70, or comma-separated IDs.")
    args = parser.parse_args()
    if "-" in args.sites:
        first, last = (int(value) for value in args.sites.split("-", 1))
        sites = list(range(first, last + 1))
    else:
        sites = [int(value) for value in args.sites.split(",")]
    started = datetime.now(timezone.utc)
    records = []
    for site in sites:
        record = build_site(args.raw_root / f"{site}.csv.gz", args.output_root / f"site={site}" / "curves.parquet")
        records.append(record)
        print({"site": site, "eligible_curves": record["eligible_complete_vendor_and_target_curves"]}, flush=True)
    totals = {
        name: {
            "eligible_curves": sum(record["splits"][name]["eligible_curves"] for record in records),
            "complete_curves": sum(record["splits"][name]["complete_curves"] for record in records),
            "delivery_rows": 96 * sum(record["splits"][name]["complete_curves"] for record in records),
        }
        for name in SPLIT_NAMES
    }
    result = {
        "dataset": "EMSx daily complete 96-step curves",
        "started_utc": started.isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "source_root": str(args.raw_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "issue_schedule": "One curve per system and eligible day at 00:00 UTC.",
        "target_contract": "Step k is energy in the 15-minute interval ending 15*k minutes after issue, for k=1,...,96.",
        "split": "Per-system chronological 65/5/10/20 percent split after requiring complete vendor and target curves.",
        "complete_case": "All 16 issue-time and prior target values, the complete causal previous-day curve, and the mature 24-hour vendor error are available.",
        "scale": "Per-system 99.5th percentile of all 96 target values from training curves only.",
        "random_seed": None,
        "software": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
        "totals": totals,
        "sites": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.manifest), "totals": totals}, indent=2))


if __name__ == "__main__":
    main()
