"""Audit and summarize matched Moirai horizon and vendor residual experiments."""

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

from analyse_emsx_foundation_step_errors import QUANTILE_COLUMNS, step_error
from emsx_daily_fewshot_common import score_predictions


KEYS = ["site_id", "issue_time", "valid_time", "delivery_step"]
METRICS = (
    "pooled_median_mae_kwh", "equal_system_median_mae_kwh",
    "weighted_interval_score_kwh", "central_90_interval_coverage",
)


def compare_saved(frame: pd.DataFrame, saved: dict, identifier: str) -> None:
    recalculated = score_predictions(frame)
    for key, value in saved.items():
        if not np.isclose(recalculated[key], value, rtol=1e-9, atol=1e-9):
            raise RuntimeError(f"{identifier}: saved {key} differs from predictions")
    if recalculated["finite_delivery_rows"] != recalculated["delivery_rows"]:
        raise RuntimeError(f"{identifier}: nonfinite deliveries")
    if frame.duplicated(KEYS).any():
        raise RuntimeError(f"{identifier}: duplicate delivery keys")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--new-run-root", type=Path, required=True)
    parser.add_argument("--frozen-run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    new_results = args.new_run_root / "results"
    new_predictions = args.new_run_root / "predictions"
    frozen_results = args.frozen_run_root / "frozen_repeats" / "results"
    frozen_predictions = args.frozen_run_root / "frozen_repeats" / "predictions"
    frozen_summary = json.loads((args.frozen_run_root / "frozen_repeats" / "summary.json").read_text())
    if len(protocol["arms"]) != 4 or len(protocol["seeds"]) != 8:
        raise RuntimeError("This summary expects the frozen four arm, eight seed protocol")

    rows, step_rows = [], []
    for seed in protocol["seeds"]:
        for arm in protocol["arms"]:
            target_mode, variant = arm["target_mode"], arm["variant"]
            identifier = f"moirai2_{target_mode}_{variant}_h96_s{seed}"
            result = json.loads((new_results / f"{identifier}.json").read_text(encoding="utf-8"))
            if result["identifier"] != identifier or result["seed"] != seed or result["arm"] != arm:
                raise RuntimeError(f"Result identity differs for {identifier}")
            baseline_id = f"moirai2_{variant}_d7_lr1em05_full_n200_frozen_s{seed}"
            baseline = json.loads((frozen_results / f"{baseline_id}.json").read_text(encoding="utf-8"))
            for partition in ("calibration", "test"):
                path = new_predictions / f"{identifier}_{partition}.parquet"
                frame = pd.read_parquet(path)
                saved = result["metrics"][partition]
                compare_saved(frame, saved, f"{identifier}/{partition}")
                baseline_path = frozen_predictions / (
                    f"{baseline_id}.parquet" if partition == "calibration"
                    else f"{baseline_id}_test.parquet"
                )
                baseline_frame = pd.read_parquet(baseline_path)
                if not frame[KEYS].equals(baseline_frame[KEYS]):
                    raise RuntimeError(f"Delivery keys differ for {identifier}/{partition}")
                if not np.allclose(
                    frame.actual_pv_kwh, baseline_frame.actual_pv_kwh, rtol=0, atol=1e-5,
                ) or not np.allclose(
                    frame.vendor_prediction_kwh, baseline_frame.vendor_prediction_kwh,
                    rtol=0, atol=1e-5,
                ):
                    raise RuntimeError(f"Actual or vendor values differ for {identifier}/{partition}")
                baseline_metrics = (
                    baseline["metrics"] if partition == "calibration"
                    else baseline["secondary_metrics"]
                )
                vendor_mae = frozen_summary["vendor_mae_kwh"][partition]
                row = {
                    "seed": seed, "target_mode": target_mode, "variant": variant,
                    "partition": partition, "identifier": identifier,
                    "best_step": result["best_step"],
                    "vendor_mae_kwh": vendor_mae,
                    "baseline_64_mae_kwh": baseline_metrics["pooled_median_mae_kwh"],
                }
                row.update({name: saved[name] for name in METRICS})
                row["improvement_over_64_kwh"] = (
                    row["baseline_64_mae_kwh"] - row["pooled_median_mae_kwh"]
                )
                row["improvement_over_vendor_kwh"] = (
                    vendor_mae - row["pooled_median_mae_kwh"]
                )
                rows.append(row)
                raw = frame[QUANTILE_COLUMNS].to_numpy(float)
                step = step_error(frame, np.sort(raw, axis=1)[:, 5])
                step["seed"] = seed
                step["target_mode"] = target_mode
                step["variant"] = variant
                step["partition"] = partition
                step_rows.append(step)
                print(json.dumps({"checked": identifier, "partition": partition}), flush=True)

    seed_table = pd.DataFrame(rows)
    direct_reference = seed_table.loc[
        seed_table.target_mode == "direct",
        ["seed", "variant", "partition", "pooled_median_mae_kwh"],
    ].rename(columns={"pooled_median_mae_kwh": "direct_96_mae_kwh"})
    seed_table = seed_table.merge(
        direct_reference, on=["seed", "variant", "partition"],
        validate="many_to_one",
    )
    seed_table["paired_improvement_over_direct_96_kwh"] = (
        seed_table.direct_96_mae_kwh - seed_table.pooled_median_mae_kwh
    )
    step_table = pd.concat(step_rows, ignore_index=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    seed_table.to_csv(args.output_root / "moirai_96_seed_metrics.csv", index=False)
    step_table.to_csv(args.output_root / "moirai_96_step_metrics_by_seed.csv", index=False)
    summary = seed_table.groupby(
        ["partition", "target_mode", "variant"], sort=True,
    ).agg(
        seeds=("seed", "nunique"),
        mean_mae_kwh=("pooled_median_mae_kwh", "mean"),
        seed_sd_mae_kwh=("pooled_median_mae_kwh", "std"),
        mean_baseline_64_mae_kwh=("baseline_64_mae_kwh", "mean"),
        mean_improvement_over_64_kwh=("improvement_over_64_kwh", "mean"),
        mean_paired_improvement_over_direct_96_kwh=(
            "paired_improvement_over_direct_96_kwh", "mean"
        ),
        sd_paired_improvement_over_direct_96_kwh=(
            "paired_improvement_over_direct_96_kwh", "std"
        ),
        mean_improvement_over_vendor_kwh=("improvement_over_vendor_kwh", "mean"),
        seeds_better_than_vendor=(
            "improvement_over_vendor_kwh", lambda values: int((values > 0).sum())
        ),
        seeds_better_than_direct_96=(
            "paired_improvement_over_direct_96_kwh",
            lambda values: int((values > 0).sum()),
        ),
        mean_wis_kwh=("weighted_interval_score_kwh", "mean"),
        mean_coverage=("central_90_interval_coverage", "mean"),
        median_best_step=("best_step", "median"),
    ).reset_index()
    summary.to_csv(args.output_root / "moirai_96_summary.csv", index=False)
    step_summary = step_table.groupby(
        ["partition", "target_mode", "variant", "delivery_step"], sort=True,
    ).agg(mean_mae_kwh=("mae_kwh", "mean"), seed_sd_mae_kwh=("mae_kwh", "std"),
          mean_active_mae_kwh=("active_mae_kwh", "mean"),
          rows_per_seed=("rows", "first"),
          active_rows_per_seed=("active_rows", "first")).reset_index()
    step_summary.to_csv(args.output_root / "moirai_96_step_summary.csv", index=False)

    baseline_steps = []
    vendor_steps = []
    for partition in ("calibration", "test"):
        for variant in ("target_only", "vendor_covariate"):
            for seed in protocol["seeds"]:
                identifier = f"moirai2_{variant}_d7_lr1em05_full_n200_frozen_s{seed}"
                suffix = "" if partition == "calibration" else "_test"
                frame = pd.read_parquet(
                    frozen_predictions / f"{identifier}{suffix}.parquet",
                    columns=["delivery_step", "actual_pv_kwh", "vendor_prediction_kwh", *QUANTILE_COLUMNS],
                )
                step = step_error(frame, np.sort(frame[QUANTILE_COLUMNS].to_numpy(float), axis=1)[:, 5])
                step["partition"] = partition
                step["variant"] = variant
                step["seed"] = seed
                baseline_steps.append(step)
                if variant == "target_only" and seed == protocol["seeds"][0]:
                    vendor = step_error(
                        frame, frame.vendor_prediction_kwh.to_numpy(float)
                    )
                    vendor["partition"] = partition
                    vendor_steps.append(vendor)
    baseline_steps = pd.concat(baseline_steps, ignore_index=True)
    baseline_summary = baseline_steps.groupby(
        ["partition", "variant", "delivery_step"], sort=True,
    ).mae_kwh.mean().rename("baseline_64_mae_kwh").reset_index()
    step_summary = step_summary.merge(
        baseline_summary, on=["partition", "variant", "delivery_step"],
        validate="many_to_one",
    )
    vendor_summary = pd.concat(vendor_steps, ignore_index=True).rename(
        columns={"mae_kwh": "vendor_mae_kwh"}
    )
    step_summary = step_summary.merge(
        vendor_summary[["partition", "delivery_step", "vendor_mae_kwh"]],
        on=["partition", "delivery_step"], validate="many_to_one",
    )
    step_summary["improvement_over_64_kwh"] = (
        step_summary.baseline_64_mae_kwh - step_summary.mean_mae_kwh
    )
    step_summary.to_csv(args.output_root / "moirai_96_step_summary.csv", index=False)
    bands = []
    for (partition, target_mode, variant), block in step_summary.groupby(
        ["partition", "target_mode", "variant"], sort=True,
    ):
        for first, last in ((1, 32), (33, 64), (65, 96)):
            section = block.loc[block.delivery_step.between(first, last)]
            bands.append({
                "partition": partition, "target_mode": target_mode,
                "variant": variant, "first_step": first, "last_step": last,
                "mean_mae_kwh": float(section.mean_mae_kwh.mean()),
                "baseline_64_mae_kwh": float(section.baseline_64_mae_kwh.mean()),
                "vendor_mae_kwh": float(section.vendor_mae_kwh.mean()),
            })
    band_table = pd.DataFrame(bands)
    band_table.to_csv(args.output_root / "moirai_96_step_bands.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True, layout="constrained")
    styles = {
        ("direct", "vendor_covariate"): ("Direct PV, 96 steps", "#0072B2"),
        ("vendor_residual", "target_only"): ("Vendor residual, 96 steps", "#D55E00"),
        ("vendor_residual", "vendor_covariate"): ("Residual plus covariate, 96 steps", "#CC79A7"),
    }
    for axis, partition in zip(axes, ("calibration", "test")):
        block = step_summary.loc[step_summary.partition == partition]
        reference = block.loc[(block.target_mode == "direct") & (block.variant == "vendor_covariate")]
        axis.plot(reference.delivery_step, reference.vendor_mae_kwh,
                  color="black", linestyle="--", label="Issued vendor")
        axis.plot(reference.delivery_step, reference.baseline_64_mae_kwh,
                  color="0.5", label="Direct PV, 64 steps")
        for (target_mode, variant), (label, color) in styles.items():
            series = block.loc[(block.target_mode == target_mode) & (block.variant == variant)]
            axis.plot(series.delivery_step, series.mean_mae_kwh, color=color, label=label)
        axis.axvline(64.5, color="0.55", linestyle=":", linewidth=1)
        axis.set(title=partition.capitalize(), xlabel="Delivery step", xlim=(1, 96))
        axis.grid(alpha=0.15)
    axes[0].set_ylabel("Mean absolute error (kWh)")
    axes[0].legend(loc="upper left", fontsize=7, frameon=False)
    fig.savefig(args.output_root / "moirai_96_comparison.pdf")
    plt.close(fig)

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(__import__("sys").argv),
        "protocol": str(args.protocol.resolve()),
        "new_run_root": str(args.new_run_root.resolve()),
        "frozen_run_root": str(args.frozen_run_root.resolve()),
        "expected_fits": len(protocol["seeds"]) * len(protocol["arms"]),
        "checked_fits": int(seed_table[["seed", "target_mode", "variant"]].drop_duplicates().shape[0]),
        "checked_prediction_files": int(len(rows)),
        "interpretation": "Seed variation describes initialization and training batches, not uncertainty across new systems or time periods. The original test informed selection of the old 64 step settings; calibration did not. The 96 step arms were frozen before their outcomes were examined.",
        "summary": summary.to_dict(orient="records"),
        "delivery_bands": bands,
    }
    (args.output_root / "moirai_96_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"checked_fits": report["checked_fits"],
                      "checked_prediction_files": report["checked_prediction_files"]}), flush=True)


if __name__ == "__main__":
    main()
