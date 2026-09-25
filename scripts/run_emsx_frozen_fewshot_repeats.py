"""Run frozen EMSx few shot settings with repeated seeds and two temporal evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

from emsx_daily_curve_common import load_wide
from emsx_daily_fewshot_common import prepare_records


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["chronos2", "moirai2"], required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--curve-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    if protocol["protocol_version"] != 1:
        raise ValueError("Unsupported frozen protocol version")
    model_spec = protocol[args.model]
    seeds = protocol["seeds"][:1] if args.smoke else protocol["seeds"]
    variants = model_spec["variants"][:1] if args.smoke else model_spec["variants"]
    output_root = args.output_root / ("smoke" if args.smoke else "frozen_repeats")
    result_root = output_root / "results"
    prediction_root = output_root / "predictions"
    checkpoint_root = output_root / "checkpoints"
    for path in (result_root, prediction_root, checkpoint_root):
        path.mkdir(parents=True, exist_ok=True)

    wide = load_wide(args.curve_root)
    context_length = protocol["context_length"]
    train = prepare_records(
        wide, args.raw_root, protocol["training_partition"], context_length,
        profiles_per_system=protocol["adaptation_profiles_per_site"],
        require_complete_context=True,
    )
    validation = prepare_records(
        wide, args.raw_root, protocol["checkpoint_selection_partition"], context_length,
        profiles_per_system=model_spec["validation_profiles_per_site"],
        require_complete_context=True,
    )
    calibration = prepare_records(
        wide, args.raw_root, protocol["primary_evaluation_partition"],
        context_length, max_profiles=16 if args.smoke else None,
    )
    test = prepare_records(
        wide, args.raw_root, protocol["secondary_evaluation_partition"],
        context_length, max_profiles=16 if args.smoke else None,
    )
    if not train or not validation or not calibration or not test:
        raise RuntimeError("A frozen partition contains no profiles")
    if {item["item_id"] for item in calibration} & {item["item_id"] for item in test}:
        raise RuntimeError("The evaluation partitions overlap")

    if args.model == "chronos2":
        from chronos import Chronos2Pipeline
        from run_emsx_daily_chronos2 import MODEL_ID, MODEL_REVISION
        from run_emsx_daily_chronos2_fewshot_sweep import run_one

        if (MODEL_ID, MODEL_REVISION) != (model_spec["model_id"], model_spec["model_revision"]):
            raise RuntimeError("Chronos checkpoint differs from frozen protocol")
        base = Chronos2Pipeline.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, device_map="cuda", local_files_only=True,
        )
    else:
        from run_emsx_daily_moirai2 import MODEL_ID, MODEL_REVISION
        from run_emsx_daily_moirai2_fewshot_sweep import run_one

        if (MODEL_ID, MODEL_REVISION) != (model_spec["model_id"], model_spec["model_revision"]):
            raise RuntimeError("Moirai checkpoint differs from frozen protocol")

    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    progress_path = result_root / f"{args.model}_progress.json"
    completed = []
    failures = []
    print(json.dumps({
        "event": "prepared", "utc": utc_now(), "model": args.model,
        "protocol_sha256": protocol_sha256, "train": len(train),
        "validation": len(validation), "calibration": len(calibration),
        "test": len(test), "planned_fits": len(seeds) * len(variants),
    }), flush=True)
    for seed in seeds:
        for variant in variants:
            run_tag = f"frozen_s{seed}"
            print(json.dumps({"event": "starting", "utc": utc_now(), "model": args.model,
                              "seed": seed, "variant": variant}), flush=True)
            try:
                common = dict(
                    spec=model_spec["specification"], variant=variant,
                    training_records=train, validation_records=validation,
                    test_records=calibration, context_length=context_length,
                    checkpoint_root=checkpoint_root, prediction_root=prediction_root,
                    result_root=result_root, smoke=args.smoke, seed=seed,
                    run_tag=run_tag, secondary_records=test,
                    secondary_name="test", evaluation_partition="calibration",
                )
                if args.model == "chronos2":
                    result = run_one(
                        base=base, training_batch_size=model_spec["training_batch_size"],
                        inference_batch_size=model_spec["inference_batch_size"],
                        chunk_size=model_spec["inference_chunk_size"], **common,
                    )
                else:
                    result = run_one(
                        training_batch_size=model_spec["training_batch_size"],
                        inference_batch_size=model_spec["inference_batch_size"],
                        validation_interval=model_spec["validation_interval"], **common,
                    )
                completed.append({"seed": seed, "variant": variant, "identifier": result["identifier"],
                                  "status": result.get("status", "completed"),
                                  "calibration_mae": result.get("metrics", {}).get("pooled_median_mae_kwh"),
                                  "test_mae": (result.get("secondary_metrics") or {}).get("pooled_median_mae_kwh")})
                print(json.dumps({"event": "completed", "utc": utc_now(), **completed[-1]}), flush=True)
            except Exception as error:
                failures.append({"seed": seed, "variant": variant, "error": repr(error),
                                 "traceback": traceback.format_exc()})
                print(json.dumps({"event": "failed", "utc": utc_now(),
                                  "seed": seed, "variant": variant, "error": repr(error)}), flush=True)
                if len(failures) >= 3 and not completed:
                    raise RuntimeError("Three initial fits failed; stopping to avoid repeated errors") from error
            progress_path.write_text(json.dumps({
                "updated_utc": utc_now(), "model": args.model,
                "protocol_sha256": protocol_sha256,
                "planned_fits": len(seeds) * len(variants),
                "completed": completed, "failures": failures,
            }, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"{len(failures)} frozen fits failed; see {progress_path}")


if __name__ == "__main__":
    main()
