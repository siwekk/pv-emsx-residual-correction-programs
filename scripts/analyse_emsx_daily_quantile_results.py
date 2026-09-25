"""Create manuscript outputs for raw and causally calibrated quantile LightGBM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


QUANTILES = np.asarray([.05, .10, .20, .30, .40, .50, .60, .70, .80, .90, .95])
QCOLS = [f"q{int(round(100 * q)):02d}_kwh" for q in QUANTILES]
CALCOLS = [f"cal_{column}" for column in QCOLS]
KEYS = ["site_id", "issue_time", "valid_time", "delivery_step"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--table-output", type=Path, required=True)
    parser.add_argument("--figure-root", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_parquet(args.prediction)
    reference = pd.read_parquet(args.reference, columns=KEYS)
    comparison = reference.merge(frame[KEYS], on=KEYS, how="outer", indicator=True)
    audit = {
        "delivery_rows": int(len(frame)), "curves": int(frame[["site_id", "issue_time"]].drop_duplicates().shape[0]),
        "systems": int(frame.site_id.nunique()), "duplicate_key_rows": int(frame.duplicated(KEYS, keep=False).sum()),
        "reference_keys_missing": int((comparison._merge == "left_only").sum()), "extra_keys": int((comparison._merge == "right_only").sum()),
        "nonfinite_raw_values": int((~np.isfinite(frame[QCOLS].to_numpy(float))).sum()),
        "nonfinite_calibrated_values": int((~np.isfinite(frame[CALCOLS].to_numpy(float))).sum()),
        "raw_crossing_rows": int((frame[QCOLS].to_numpy(float)[:, :-1] > frame[QCOLS].to_numpy(float)[:, 1:]).any(axis=1).sum()),
        "calibrated_crossing_rows": int((frame[CALCOLS].to_numpy(float)[:, :-1] > frame[CALCOLS].to_numpy(float)[:, 1:]).any(axis=1).sum()),
    }
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    raw = metrics["raw_test"]
    calibrated = metrics["causally_calibrated_test"]
    lines = [
        "\\begin{table*}[t]", "\\centering",
        "\\caption{Task-trained probabilistic residual correction before and after causal rolling calibration. Lower scores are better, while nominal central-interval coverage is 0.90.}",
        "\\label{tab:quantile-lightgbm}", "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{lrrrrrrr}", "\\toprule",
        "Variant & Median MAE & Pinball & Grid CRPS & WIS & Coverage & Width & Interval score \\\\", "\\midrule",
        f"Raw quantiles & {raw['median_mae_kwh']:.2f} & {raw['mean_pinball_kwh']:.2f} & {raw['grid_crps_kwh']:.2f} & {raw['wis_kwh']:.2f} & {raw['coverage_90']:.3f} & {raw['width_90_kwh']:.2f} & {raw['interval_score_90_kwh']:.2f} \\\\",
        f"Causal calibration & {calibrated['median_mae_kwh']:.2f} & {calibrated['mean_pinball_kwh']:.2f} & {calibrated['grid_crps_kwh']:.2f} & {calibrated['wis_kwh']:.2f} & {calibrated['coverage_90']:.3f} & {calibrated['width_90_kwh']:.2f} & {calibrated['interval_score_90_kwh']:.2f} \\\\",
        "\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}", "",
    ]
    args.table_output.parent.mkdir(parents=True, exist_ok=True)
    args.table_output.write_text("\n".join(lines), encoding="utf-8")
    args.figure_root.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), sharex=True, sharey=True)
    for ax, active, title in ((axes[0], False, "All deliveries"), (axes[1], True, "Issued-active deliveries")):
        subset = frame.loc[frame.vendor_prediction_kwh > 0] if active else frame
        y = subset.actual_pv_kwh.to_numpy(float)
        raw_observed = [float(np.mean(y <= subset[column].to_numpy(float))) for column in QCOLS]
        cal_observed = [float(np.mean(y <= subset[column].to_numpy(float))) for column in CALCOLS]
        ax.plot(QUANTILES, raw_observed, marker="o", label="Raw", color="#D55E00")
        ax.plot(QUANTILES, cal_observed, marker="o", label="Causal calibration", color="#0072B2")
        ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Nominal quantile level")
    axes[0].set_ylabel("Observed proportion below forecast")
    axes[1].legend(frameon=False)
    fig.suptitle("Residual LightGBM quantile reliability")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(args.figure_root / f"quantile_lightgbm_reliability.{suffix}", dpi=300)
    plt.close(fig)
    coverage = []
    for step, block in frame.groupby("delivery_step", observed=True):
        y = block.actual_pv_kwh.to_numpy(float)
        coverage.append({"step": int(step), "raw": float(np.mean((y >= block.q05_kwh) & (y <= block.q95_kwh))), "calibrated": float(np.mean((y >= block.cal_q05_kwh) & (y <= block.cal_q95_kwh)))})
    coverage = pd.DataFrame(coverage)
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    ax.plot(coverage.step / 4, coverage.raw, label="Raw", color="#D55E00")
    ax.plot(coverage.step / 4, coverage.calibrated, label="Causal calibration", color="#0072B2")
    ax.axhline(.9, color="black", linestyle="--", linewidth=1, label="Nominal 0.90")
    ax.set_xlabel("Hours after issue")
    ax.set_ylabel("Central 90% coverage")
    ax.set_ylim(0, 1.03)
    ax.set_title("Coverage across the delivery curve")
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(args.figure_root / f"quantile_lightgbm_coverage_by_step.{suffix}", dpi=300)
    plt.close(fig)
    output = {"audit": audit, "raw_test": raw, "causally_calibrated_test": calibrated, "coverage_by_step": coverage.to_dict("records")}
    args.summary_output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
