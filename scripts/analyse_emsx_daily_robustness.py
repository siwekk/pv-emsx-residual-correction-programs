"""Analyse feature ablations, system robustness, seasons, and active-period coverage."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KEYS = ["site_id", "issue_time", "valid_time", "delivery_step"]


def metrics(frame: pd.DataFrame, column: str) -> dict:
    error = np.abs(frame.actual_pv_kwh.to_numpy(float) - frame[column].to_numpy(float))
    curves = pd.DataFrame({"site_id": frame.site_id, "issue_time": frame.issue_time, "error": error}).groupby(["site_id", "issue_time"], observed=True).error.mean()
    systems = curves.groupby("site_id").mean()
    vendor_curves = frame.assign(error=np.abs(frame.actual_pv_kwh - frame.vendor_prediction_kwh)).groupby(["site_id", "issue_time"], observed=True).error.mean()
    vendor_systems = vendor_curves.groupby("site_id").mean()
    return {
        "pooled_mae_kwh": float(error.mean()),
        "equal_system_mae_kwh": float(systems.mean()),
        "normalized_mae": float(np.mean(error / frame.scale_kwh.to_numpy(float))),
        "systems_improved": int((systems < vendor_systems).sum()),
        "systems_tied": int(np.isclose(systems, vendor_systems).sum()),
        "curve_win_fraction": float((curves < vendor_curves).mean()),
    }


def write_table(path: Path, results: dict, seasons: dict) -> None:
    order = ["Vendor", "Target-step residual", "Issued-curve residual", "Full residual"]
    lines = [
        "\\begin{table*}[t]", "\\centering",
        "\\caption{Feature ablation and robustness of residual correction. All residual models use the frozen LightGBM configuration. The system count reports lower test MAE than the vendor.}",
        "\\label{tab:ablation-robustness}", "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrr}", "\\toprule",
        "Information set & Pooled MAE & Equal-system MAE & Normalised MAE & Systems improved & Curve win fraction \\\\",
        "\\midrule",
    ]
    for name in order:
        value = results[name]
        improved = "--" if name == "Vendor" else f"{value['systems_improved']}/70"
        wins = "--" if name == "Vendor" else f"{value['curve_win_fraction']:.3f}"
        lines.append(f"{name} & {value['pooled_mae_kwh']:.2f} & {value['equal_system_mae_kwh']:.2f} & {value['normalized_mae']:.4f} & {improved} & {wins} \\\\")
    lines.extend([
        "\\midrule", "Season & Vendor MAE & Full residual MAE & Reduction & Relative reduction & Curves \\\\", "\\midrule",
    ])
    for season in ("DJF", "MAM", "JJA", "SON"):
        value = seasons[season]
        relative = 100 * value["reduction_kwh"] / value["vendor_mae_kwh"]
        lines.append(f"{season} & {value['vendor_mae_kwh']:.2f} & {value['full_mae_kwh']:.2f} & {value['reduction_kwh']:.2f} & {relative:.1f}\\% & {value['curves']:,} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table-output", type=Path, required=True)
    parser.add_argument("--figure-root", type=Path, required=True)
    args = parser.parse_args()

    baseline = pd.read_parquet(args.prediction_root / "emsx_daily_baselines.parquet")
    full = pd.read_parquet(args.prediction_root / "emsx_daily_lightgbm.parquet")
    ablation = pd.read_parquet(args.prediction_root / "emsx_daily_lightgbm_ablation.parquet")
    frame = baseline.merge(full[KEYS + ["lightgbm_prediction_kwh"]], on=KEYS, validate="one_to_one")
    frame = frame.merge(ablation[KEYS + ["lightgbm_target_step_prediction_kwh", "lightgbm_issued_curve_prediction_kwh"]], on=KEYS, validate="one_to_one")
    columns = {
        "Vendor": "vendor_prediction_kwh",
        "Target-step residual": "lightgbm_target_step_prediction_kwh",
        "Issued-curve residual": "lightgbm_issued_curve_prediction_kwh",
        "Full residual": "lightgbm_prediction_kwh",
    }
    result_metrics = {name: metrics(frame, column) for name, column in columns.items()}

    month = frame.issue_time.dt.month
    season_value = np.select(
        [month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])],
        ["DJF", "MAM", "JJA"], default="SON",
    )
    frame["season"] = season_value
    seasons = {}
    for season, block in frame.groupby("season"):
        vendor = float(np.mean(np.abs(block.actual_pv_kwh - block.vendor_prediction_kwh)))
        corrected = float(np.mean(np.abs(block.actual_pv_kwh - block.lightgbm_prediction_kwh)))
        seasons[season] = {
            "vendor_mae_kwh": vendor, "full_mae_kwh": corrected,
            "reduction_kwh": vendor - corrected,
            "curves": int(block[["site_id", "issue_time"]].drop_duplicates().shape[0]),
        }

    system = frame.assign(
        vendor_error=np.abs(frame.actual_pv_kwh - frame.vendor_prediction_kwh),
        corrected_error=np.abs(frame.actual_pv_kwh - frame.lightgbm_prediction_kwh),
    ).groupby("site_id", observed=True)[["vendor_error", "corrected_error"]].mean()
    system["reduction"] = system.vendor_error - system.corrected_error
    system = system.sort_values("reduction")
    args.figure_root.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    colors = np.where(system.reduction >= 0, "#0072B2", "#D55E00")
    ax.bar(np.arange(len(system)), system.reduction, color=colors, width=.82)
    ax.set_yscale("log")
    ax.set_xlabel("Systems ordered by MAE reduction")
    ax.set_ylabel("Vendor minus residual MAE [kWh, log scale]")
    ax.set_title("System-level effect of residual correction")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(args.figure_root / f"system_gain.{suffix}", dpi=300)
    plt.close(fig)

    probabilistic = pd.read_parquet(args.prediction_root / "emsx_daily_quantile_lightgbm.parquet")
    coverage = {}
    for name in ("raw", "causal"):
        active = probabilistic.vendor_prediction_kwh > 0
        lower = "cal_q05_kwh" if name == "causal" else "q05_kwh"
        upper = "cal_q95_kwh" if name == "causal" else "q95_kwh"
        covered = (probabilistic.actual_pv_kwh >= probabilistic[lower]) & (probabilistic.actual_pv_kwh <= probabilistic[upper])
        by_step = probabilistic.loc[active].assign(covered=covered[active]).groupby("delivery_step").covered.mean()
        coverage[name] = {
            "all": float(covered.mean()), "issued_active": float(covered[active].mean()),
            "active_step_min": float(by_step.min()), "active_step_max": float(by_step.max()),
        }

    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "feature_ablation": result_metrics, "seasonal_robustness": seasons,
        "system_reduction_kwh": {str(int(index)): float(value) for index, value in system.reduction.items()},
        "coverage_90": coverage,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.table_output.parent.mkdir(parents=True, exist_ok=True)
    write_table(args.table_output, result_metrics, seasons)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
