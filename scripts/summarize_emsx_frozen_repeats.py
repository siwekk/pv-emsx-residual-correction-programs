"""Summarize every frozen few shot repeat without selecting a favorable seed."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = (
    "pooled_median_mae_kwh",
    "equal_system_median_mae_kwh",
    "weighted_interval_score_kwh",
    "central_90_interval_coverage",
)


def summary(values: np.ndarray) -> dict:
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else None,
        "minimum": float(values.min()),
        "median": float(np.median(values)),
        "maximum": float(values.max()),
        "seed_percentile_2_5": float(np.percentile(values, 2.5)),
        "seed_percentile_97_5": float(np.percentile(values, 97.5)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    root = args.output_root / "frozen_repeats"
    result_root = root / "results"
    rows = []
    missing = []
    vendor = {}
    for model in ("chronos2", "moirai2"):
        spec = protocol[model]
        learning_rate = f"{spec['specification']['learning_rate']:.0e}".replace("-", "m")
        if model == "chronos2":
            setting = f"d7_lr{learning_rate}_r{spec['specification']['rank']}_n{spec['specification']['steps']}"
        else:
            setting = f"d7_lr{learning_rate}_full_n{spec['specification']['steps']}"
        for variant in spec["variants"]:
            for seed in protocol["seeds"]:
                identifier = f"{model}_{variant}_{setting}_frozen_s{seed}"
                path = result_root / f"{identifier}.json"
                if not path.exists():
                    missing.append(identifier)
                    continue
                result = json.loads(path.read_text(encoding="utf-8"))
                if result["random_seed"] != seed or result["variant"] != variant:
                    raise RuntimeError(f"Result identity mismatch: {path}")
                for partition, metrics, prediction_file in (
                    ("calibration", result["metrics"], result["prediction_file"]),
                    ("test", result["secondary_metrics"], result["secondary_prediction_file"]),
                ):
                    if metrics is None:
                        raise RuntimeError(f"Missing {partition} metrics in {path}")
                    row = {"model": model, "variant": variant, "seed": seed,
                           "partition": partition, "identifier": identifier}
                    row.update({name: metrics[name] for name in METRICS})
                    rows.append(row)
                    if partition not in vendor:
                        prediction = pd.read_parquet(prediction_file, columns=[
                            "actual_pv_kwh", "vendor_prediction_kwh",
                        ])
                        vendor[partition] = float(np.mean(np.abs(
                            prediction.actual_pv_kwh.to_numpy(float)
                            - prediction.vendor_prediction_kwh.to_numpy(float)
                        )))
    table = pd.DataFrame(rows)
    table.to_csv(root / "all_seed_metrics.csv", index=False)
    aggregates = {}
    for (model, variant, partition), block in table.groupby(
        ["model", "variant", "partition"], sort=True
    ):
        key = f"{model}/{variant}/{partition}"
        aggregates[key] = {name: summary(block[name].to_numpy(float)) for name in METRICS}
        aggregates[key]["vendor_mae_kwh"] = vendor[partition]
        aggregates[key]["mean_improvement_over_vendor_kwh"] = (
            vendor[partition] - aggregates[key]["pooled_median_mae_kwh"]["mean"]
        )
    output = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "planned_seeds_per_setting": len(protocol["seeds"]),
        "expected_fits": 2 * 2 * len(protocol["seeds"]),
        "completed_fits": int(len(table) // 2),
        "missing_identifiers": missing,
        "vendor_mae_kwh": vendor,
        "aggregate": aggregates,
        "interpretation": "Seed percentiles describe variation across random initializations and training batches. They are not confidence intervals for unseen sites or future periods. The original test period informed configuration selection; calibration did not.",
    }
    (root / "summary.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"completed_fits": output["completed_fits"],
                      "missing_fits": len(missing), "summary": str(root / "summary.json")}), flush=True)


if __name__ == "__main__":
    main()
