"""Compare saved foundation forecasts at each of the 96 delivery steps."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


QUANTILE_COLUMNS = [f"q{q:02d}_kwh" for q in (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95)]
PARTITIONS = {"calibration": "", "test": "_test"}
MODEL_VARIANTS = (
    ("chronos2", "target_only", "d7_lr3em05_r8_n300"),
    ("chronos2", "vendor_covariate", "d7_lr3em05_r8_n300"),
    ("moirai2", "target_only", "d7_lr1em05_full_n200"),
    ("moirai2", "vendor_covariate", "d7_lr1em05_full_n200"),
)


def step_error(frame: pd.DataFrame, column: np.ndarray) -> pd.DataFrame:
    data = pd.DataFrame({
        "delivery_step": frame.delivery_step.to_numpy(int),
        "error": np.abs(frame.actual_pv_kwh.to_numpy(float) - column),
        "active": ((frame.actual_pv_kwh.to_numpy(float) > 0)
                   | (frame.vendor_prediction_kwh.to_numpy(float) > 0)),
    })
    if data.delivery_step.min() != 1 or data.delivery_step.max() != 96:
        raise RuntimeError("Prediction file does not contain all 96 delivery steps")
    full = data.groupby("delivery_step", sort=True).agg(
        mae_kwh=("error", "mean"), rows=("error", "size"),
    )
    active = data.loc[data.active].groupby("delivery_step", sort=True).agg(
        active_mae_kwh=("error", "mean"), active_rows=("error", "size"),
    )
    return full.join(active).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    prediction_root = args.run_root / "frozen_repeats" / "predictions"
    rows = []
    vendor_rows = []
    for partition, suffix in PARTITIONS.items():
        for model, variant, setting in MODEL_VARIANTS:
            for seed in protocol["seeds"]:
                identifier = f"{model}_{variant}_{setting}_frozen_s{seed}"
                path = prediction_root / f"{identifier}{suffix}.parquet"
                frame = pd.read_parquet(path, columns=[
                    "delivery_step", "actual_pv_kwh", "vendor_prediction_kwh",
                    *QUANTILE_COLUMNS,
                ])
                raw = frame[QUANTILE_COLUMNS].to_numpy(float)
                if not np.isfinite(raw).all():
                    raise RuntimeError(f"Nonfinite quantile in {path}")
                median = np.sort(raw, axis=1)[:, 5]
                block = step_error(frame, median)
                block["partition"] = partition
                block["model"] = model
                block["variant"] = variant
                block["seed"] = seed
                rows.append(block)
                if model == "chronos2" and variant == "target_only" and seed == protocol["seeds"][0]:
                    vendor = step_error(frame, frame.vendor_prediction_kwh.to_numpy(float))
                    vendor["partition"] = partition
                    vendor_rows.append(vendor)
                print(json.dumps({"partition": partition, "identifier": identifier}), flush=True)

    seed_table = pd.concat(rows, ignore_index=True)
    aggregate = seed_table.groupby(
        ["partition", "model", "variant", "delivery_step"], sort=True,
    ).agg(
        mean_mae_kwh=("mae_kwh", "mean"),
        seed_sd_mae_kwh=("mae_kwh", "std"),
        mean_active_mae_kwh=("active_mae_kwh", "mean"),
        seed_sd_active_mae_kwh=("active_mae_kwh", "std"),
        rows_per_seed=("rows", "first"),
        active_rows_per_seed=("active_rows", "first"),
    ).reset_index()
    vendor = pd.concat(vendor_rows, ignore_index=True).rename(columns={
        "mae_kwh": "vendor_mae_kwh", "active_mae_kwh": "vendor_active_mae_kwh",
    })
    aggregate = aggregate.merge(
        vendor[["partition", "delivery_step", "vendor_mae_kwh", "vendor_active_mae_kwh"]],
        on=["partition", "delivery_step"], validate="many_to_one",
    )
    aggregate["gain_over_vendor_kwh"] = aggregate.vendor_mae_kwh - aggregate.mean_mae_kwh
    aggregate["active_gain_over_vendor_kwh"] = (
        aggregate.vendor_active_mae_kwh - aggregate.mean_active_mae_kwh
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    seed_table.to_csv(args.output_root / "foundation_step_error_by_seed.csv", index=False)
    aggregate.to_csv(args.output_root / "foundation_step_error_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True, layout="constrained")
    styles = {
        ("chronos2", "vendor_covariate"): ("Chronos 2 with vendor", "#0072B2"),
        ("moirai2", "target_only"): ("Moirai 2 target only", "#D55E00"),
        ("moirai2", "vendor_covariate"): ("Moirai 2 with vendor", "#CC79A7"),
    }
    for axis, partition in zip(axes, PARTITIONS):
        block = aggregate.loc[aggregate.partition == partition]
        reference = block.loc[(block.model == "chronos2") & (block.variant == "target_only")]
        axis.plot(reference.delivery_step, reference.vendor_mae_kwh, color="black", label="Vendor")
        for (model, variant), (label, color) in styles.items():
            series = block.loc[(block.model == model) & (block.variant == variant)]
            axis.plot(series.delivery_step, series.mean_mae_kwh, color=color, label=label)
        axis.axvline(64.5, color="0.55", linestyle=":", linewidth=1)
        axis.set(title=partition.capitalize(), xlabel="Delivery step", xlim=(1, 96))
        axis.grid(alpha=0.15)
    axes[0].set_ylabel("Mean absolute error (kWh)")
    axes[1].legend(loc="upper right", fontsize=7, frameon=False)
    fig.savefig(args.output_root / "foundation_step_error.pdf")
    plt.close(fig)

    bands = []
    for (partition, model, variant), block in aggregate.groupby(
        ["partition", "model", "variant"], sort=True,
    ):
        for start, end in ((1, 32), (33, 64), (65, 96)):
            section = block.loc[block.delivery_step.between(start, end)]
            bands.append({
                "partition": partition, "model": model, "variant": variant,
                "first_step": start, "last_step": end,
                "mean_mae_kwh": float(section.mean_mae_kwh.mean()),
                "vendor_mae_kwh": float(section.vendor_mae_kwh.mean()),
                "gain_over_vendor_kwh": float(section.gain_over_vendor_kwh.mean()),
                "steps_better_than_vendor": int((section.gain_over_vendor_kwh > 0).sum()),
            })
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(__import__("sys").argv),
        "protocol": str(args.protocol.resolve()),
        "prediction_root": str(prediction_root.resolve()),
        "seeds_per_setting": len(protocol["seeds"]),
        "active_definition": "Actual PV or issued vendor forecast is strictly positive.",
        "delivery_bands": bands,
    }
    (args.output_root / "foundation_step_error_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
