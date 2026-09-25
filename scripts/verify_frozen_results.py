"""Independently recalculate metrics from frozen EMSx prediction files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from emsx_daily_fewshot_common import score_predictions


KEYS = ["site_id", "issue_time", "valid_time", "delivery_step"]


def expected_identifiers(protocol: dict) -> set[str]:
    identifiers = set()
    for model in ("chronos2", "moirai2"):
        spec = protocol[model]["specification"]
        learning_rate = f"{spec['learning_rate']:.0e}".replace("-", "m")
        if model == "chronos2":
            setting = f"d{spec['days']}_lr{learning_rate}_r{spec['rank']}_n{spec['steps']}"
        else:
            setting = f"d{spec['days']}_lr{learning_rate}_full_n{spec['steps']}"
        for variant in protocol[model]["variants"]:
            for seed in protocol["seeds"]:
                identifiers.add(f"{model}_{variant}_{setting}_frozen_s{seed}")
    return identifiers


def compare_metrics(saved: dict, calculated: dict, label: str) -> list[str]:
    errors = []
    for key, value in saved.items():
        if key not in calculated:
            errors.append(f"{label}: unknown saved metric {key}")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isclose(value, calculated[key], rel_tol=1e-9, abs_tol=1e-9):
                errors.append(f"{label}: {key} differs: saved={value}, calculated={calculated[key]}")
        elif value != calculated[key]:
            errors.append(f"{label}: {key} differs: saved={value}, calculated={calculated[key]}")
    for key in calculated.keys() - saved.keys():
        errors.append(f"{label}: saved metric {key} is missing")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True,
                        help="Directory containing frozen_repeats/results and predictions")
    parser.add_argument("--output", type=Path, help="Optional report path outside the code repository")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    expected = expected_identifiers(protocol)
    result_root = args.run_root / "frozen_repeats" / "results"
    prediction_root = args.run_root / "frozen_repeats" / "predictions"
    present = {path.stem for path in result_root.glob("*.json") if path.stem in expected}
    errors = [f"Missing result: {identifier}" for identifier in sorted(expected - present)]
    checked_predictions = 0
    for identifier in sorted(present):
        path = result_root / f"{identifier}.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        model = identifier.split("_", 1)[0]
        if result.get("identifier") != identifier:
            errors.append(f"{identifier}: result identifier differs")
        if result.get("model_id") != protocol[model]["model_id"]:
            errors.append(f"{identifier}: model ID differs from protocol")
        if result.get("model_revision") != protocol[model]["model_revision"]:
            errors.append(f"{identifier}: model revision differs from protocol")
        if result.get("random_seed") not in protocol["seeds"]:
            errors.append(f"{identifier}: seed is absent from protocol")
        elif not identifier.endswith(f"_frozen_s{result['random_seed']}"):
            errors.append(f"{identifier}: seed differs from result identifier")
        if result.get("variant") not in protocol[model]["variants"]:
            errors.append(f"{identifier}: variant is absent from protocol")
        elif not identifier.startswith(f"{model}_{result['variant']}_"):
            errors.append(f"{identifier}: variant differs from result identifier")
        if result.get("specification") != protocol[model]["specification"]:
            errors.append(f"{identifier}: settings differ from frozen protocol")
        if result.get("evaluation_partition") != protocol["primary_evaluation_partition"]:
            errors.append(f"{identifier}: primary partition differs from protocol")
        if result.get("secondary_partition") != protocol["secondary_evaluation_partition"]:
            errors.append(f"{identifier}: secondary partition differs from protocol")

        profile_sets = []
        for partition, file_field, metric_field in (
            (protocol["primary_evaluation_partition"], "prediction_file", "metrics"),
            (protocol["secondary_evaluation_partition"], "secondary_prediction_file", "secondary_metrics"),
        ):
            stored_path = result.get(file_field)
            saved_metrics = result.get(metric_field)
            if not stored_path or not saved_metrics:
                errors.append(f"{identifier}/{partition}: prediction path or metrics missing")
                continue
            prediction_path = prediction_root / Path(stored_path).name
            expected_name = (f"{identifier}.parquet" if metric_field == "metrics"
                             else f"{identifier}_{partition}.parquet")
            if prediction_path.name != expected_name:
                errors.append(f"{identifier}/{partition}: unexpected prediction filename")
            if not prediction_path.is_file():
                errors.append(f"{identifier}/{partition}: missing {prediction_path.name}")
                continue
            frame = pd.read_parquet(prediction_path)
            if frame.empty:
                errors.append(f"{identifier}/{partition}: empty prediction file")
                continue
            duplicate_rows = int(frame.duplicated(KEYS).sum())
            if duplicate_rows:
                errors.append(f"{identifier}/{partition}: {duplicate_rows} duplicate delivery keys")
            calculated = score_predictions(frame)
            if calculated["finite_delivery_rows"] != calculated["delivery_rows"]:
                errors.append(f"{identifier}/{partition}: nonfinite target, scale, or quantile")
            errors.extend(compare_metrics(saved_metrics, calculated, f"{identifier}/{partition}"))
            profiles = set(zip(frame["site_id"], frame["issue_time"]))
            profile_sets.append(profiles)
            checked_predictions += 1
        if len(profile_sets) == 2 and profile_sets[0] & profile_sets[1]:
            errors.append(f"{identifier}: calibration and test profiles overlap")

    report = {
        "expected_fits": len(expected),
        "result_files_checked": len(present),
        "prediction_files_checked": checked_predictions,
        "passed": not errors,
        "errors": errors,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
