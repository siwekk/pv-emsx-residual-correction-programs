"""Combine EMSx few shot sweeps with frozen zero shot and trained references."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from emsx_daily_fewshot_common import QCOLS, score_predictions


KEYS = ["site_id", "issue_time", "valid_time", "delivery_step"]


def point_metrics(frame: pd.DataFrame, prediction_column: str) -> dict:
    target = frame.actual_pv_kwh.to_numpy(float)
    prediction = frame[prediction_column].to_numpy(float)
    scale = frame.scale_kwh.to_numpy(float)
    finite = np.isfinite(target) & np.isfinite(prediction) & np.isfinite(scale)
    error = np.abs(target[finite] - prediction[finite])
    metadata = frame.loc[finite, ["site_id", "issue_time"]].reset_index(drop=True)
    per_profile = metadata.assign(error=error).groupby(
        ["site_id", "issue_time"], observed=True
    ).error.mean()
    per_system = per_profile.groupby("site_id").mean()
    return {
        "delivery_rows": int(len(frame)),
        "finite_delivery_rows": int(finite.sum()),
        "profiles": int(frame[["site_id", "issue_time"]].drop_duplicates().shape[0]),
        "systems": int(frame.site_id.nunique()),
        "pooled_median_mae_kwh": float(error.mean()),
        "equal_profile_median_mae_kwh": float(per_profile.mean()),
        "equal_system_median_mae_kwh": float(per_system.mean()),
        "training_scale_normalized_median_mae": float(np.mean(error / scale[finite])),
    }


def flatten_result(result: dict) -> dict:
    spec = result["specification"]
    metrics = result["metrics"]
    return {
        "identifier": result["identifier"],
        "model": "Chronos 2" if result["identifier"].startswith("chronos2") else "Moirai 2.0",
        "adaptation": "few shot",
        "variant": result["variant"],
        "days_per_system": spec["days"],
        "learning_rate": spec["learning_rate"],
        "rank": spec.get("rank"),
        "trainable_scope": spec.get("trainable_scope", "LoRA"),
        "steps": spec["steps"],
        "best_step": result.get("best_step"),
        "training_examples": result["training_examples"],
        "fit_seconds": result["fit_seconds"],
        "inference_seconds": result["inference_seconds"],
        **metrics,
    }


def zero_shot_row(model: str, variant: str, path: Path) -> dict:
    frame = pd.read_parquet(path)
    return {
        "identifier": f"{model.lower().replace(' ', '').replace('.', '')}_{variant}_zero_shot",
        "model": model,
        "adaptation": "zero shot",
        "variant": variant,
        "days_per_system": 0,
        "learning_rate": None,
        "rank": None,
        "trainable_scope": None,
        "steps": 0,
        "best_step": None,
        "training_examples": 0,
        "fit_seconds": 0.0,
        "inference_seconds": None,
        **score_predictions(frame),
    }


def rearranged_median(frame: pd.DataFrame) -> np.ndarray:
    return np.sort(frame[QCOLS].to_numpy(float), axis=1)[:, 5]


def paired_summary(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    reference_prediction: np.ndarray,
) -> dict:
    left = candidate[KEYS + ["actual_pv_kwh"]].copy()
    left["candidate_prediction"] = rearranged_median(candidate)
    right = reference[KEYS].copy()
    right["reference_prediction"] = reference_prediction
    matched = left.merge(right, on=KEYS, how="inner", validate="one_to_one")
    candidate_error = np.abs(
        matched.actual_pv_kwh.to_numpy(float)
        - matched.candidate_prediction.to_numpy(float)
    )
    reference_error = np.abs(
        matched.actual_pv_kwh.to_numpy(float)
        - matched.reference_prediction.to_numpy(float)
    )
    differences = matched[["site_id", "issue_time"]].copy()
    differences["reduction"] = reference_error - candidate_error
    profile = differences.groupby(["site_id", "issue_time"], observed=True).reduction.mean()
    system = profile.groupby("site_id").mean()
    return {
        "matched_delivery_rows": int(len(matched)),
        "pooled_mae_reduction_kwh": float(np.mean(reference_error - candidate_error)),
        "equal_profile_mae_reduction_kwh": float(profile.mean()),
        "equal_system_mae_reduction_kwh": float(system.mean()),
        "profiles_improved_fraction": float((profile > 0).mean()),
        "systems_improved": int((system > 0).sum()),
        "systems_total": int(len(system)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--frozen-prediction-root", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    raw_results = {}
    for path in sorted(args.result_root.glob("chronos2_*.json")) + sorted(
        args.result_root.glob("moirai2_*.json")
    ):
        if "manifest" in path.name or "smoke" in path.name:
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if "metrics" not in result:
            continue
        raw_results[result["identifier"]] = result
        rows.append(flatten_result(result))
    zero_shot_files = {
        ("Chronos 2", "target_only"): "emsx_daily_chronos2_target_only.parquet",
        ("Chronos 2", "vendor_covariate"): "emsx_daily_chronos2_vendor_covariate.parquet",
        ("Moirai 2.0", "target_only"): "emsx_daily_moirai2_target_only.parquet",
        ("Moirai 2.0", "vendor_covariate"): "emsx_daily_moirai2_vendor_covariate.parquet",
    }
    for (model, variant), filename in zero_shot_files.items():
        rows.append(zero_shot_row(model, variant, args.frozen_prediction_root / filename))
    comparison = pd.DataFrame(rows)
    if comparison.empty:
        raise RuntimeError("No completed few shot results were found")
    comparison = comparison.sort_values(
        ["model", "variant", "pooled_median_mae_kwh", "identifier"]
    ).reset_index(drop=True)
    lightgbm = pd.read_parquet(
        args.frozen_prediction_root / "emsx_daily_lightgbm.parquet"
    )
    vendor_metrics = point_metrics(lightgbm, "vendor_prediction_kwh")
    lightgbm_metrics = point_metrics(lightgbm, "lightgbm_prediction_kwh")
    quantile_lightgbm = pd.read_parquet(
        args.frozen_prediction_root / "emsx_daily_quantile_lightgbm.parquet"
    )
    quantile_metrics = score_predictions(quantile_lightgbm)
    calibrated = quantile_lightgbm.copy()
    for column in QCOLS:
        calibrated[column] = calibrated[f"cal_{column}"]
    calibrated_metrics = score_predictions(calibrated)
    best_by_model_and_variant = (
        comparison.loc[comparison.adaptation == "few shot"]
        .sort_values("pooled_median_mae_kwh")
        .groupby(["model", "variant"], observed=True, as_index=False)
        .first()
    )
    zero_lookup = comparison.loc[comparison.adaptation == "zero shot"].set_index(
        ["model", "variant"]
    )
    improvements = []
    for row in best_by_model_and_variant.itertuples(index=False):
        zero = zero_lookup.loc[(row.model, row.variant)]
        candidate = pd.read_parquet(args.prediction_root / f"{row.identifier}.parquet")
        zero_filename = zero_shot_files[(row.model, row.variant)]
        zero_frame = pd.read_parquet(args.frozen_prediction_root / zero_filename)
        improvements.append(
            {
                "model": row.model,
                "variant": row.variant,
                "best_identifier": row.identifier,
                "zero_shot_mae_kwh": float(zero.pooled_median_mae_kwh),
                "best_few_shot_mae_kwh": float(row.pooled_median_mae_kwh),
                "absolute_mae_reduction_kwh": float(
                    zero.pooled_median_mae_kwh - row.pooled_median_mae_kwh
                ),
                "relative_mae_reduction": float(
                    (zero.pooled_median_mae_kwh - row.pooled_median_mae_kwh)
                    / zero.pooled_median_mae_kwh
                ),
                "difference_from_residual_lightgbm_kwh": float(
                    row.pooled_median_mae_kwh
                    - lightgbm_metrics["pooled_median_mae_kwh"]
                ),
                "paired_against_zero_shot": paired_summary(
                    candidate,
                    zero_frame,
                    rearranged_median(zero_frame),
                ),
                "paired_against_residual_lightgbm": paired_summary(
                    candidate,
                    lightgbm,
                    lightgbm.lightgbm_prediction_kwh.to_numpy(float),
                ),
            }
        )
    output = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "Best configuration within each model and input contract by pooled median MAE on the untouched test partition. This is descriptive selection after the requested sweep and must not be treated as a prespecified test estimate.",
        "adaptation_contract": "The most recent 1, 3, or 7 complete training profiles per system are used for adaptation. Tuning data select checkpoints. Calibration and test data are not used for fitting or checkpoint selection.",
        "reference_point_metrics": {
            "vendor": vendor_metrics,
            "residual_lightgbm": lightgbm_metrics,
        },
        "reference_probabilistic_metrics": {
            "quantile_lightgbm_raw": quantile_metrics,
            "quantile_lightgbm_causally_calibrated": calibrated_metrics,
        },
        "best_few_shot_improvements": improvements,
        "rows": comparison.astype(object).where(pd.notna(comparison), None).to_dict("records"),
        "raw_results": raw_results,
    }
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.csv_output, index=False)
    args.json_output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_few_shot_improvements": improvements}, indent=2))


if __name__ == "__main__":
    main()
