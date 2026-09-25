"""Run a reproducible Chronos 2 LoRA adaptation sweep on EMSx."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from chronos import Chronos2Pipeline

from emsx_daily_curve_common import load_wide
from emsx_daily_fewshot_common import prepare_records, score_predictions, set_seed
from run_emsx_daily_chronos2 import (
    MODEL_ID,
    MODEL_REVISION,
    predict_variant,
)


SEED = 20260924


def specifications() -> list[dict]:
    specs = [
        {"days": days, "learning_rate": 1e-5, "rank": 8, "steps": 300}
        for days in (1, 3, 7)
    ]
    specs.extend(
        {"days": 7, "learning_rate": learning_rate, "rank": 8, "steps": 300}
        for learning_rate in (3e-6, 3e-5)
    )
    specs.extend(
        {"days": 7, "learning_rate": 1e-5, "rank": rank, "steps": 300}
        for rank in (4, 16)
    )
    return list({specification_id(spec): spec for spec in specs}.values())


def specification_id(spec: dict) -> str:
    learning_rate = f"{spec['learning_rate']:.0e}".replace("-", "m")
    return f"d{spec['days']}_lr{learning_rate}_r{spec['rank']}_n{spec['steps']}"


def adaptation_inputs(records: list[dict], with_vendor: bool) -> list[dict]:
    inputs = []
    for record in records:
        item = {
            "target": np.concatenate(
                [record["history_target"], record["actual"]]
            ).astype(np.float32)
        }
        if with_vendor:
            item["past_covariates"] = {
                "vendor_forecast": np.concatenate(
                    [record["history_vendor"], record["future_vendor"]]
                ).astype(np.float32)
            }
            item["future_covariates"] = {
                "vendor_forecast": record["future_vendor"].astype(np.float32)
            }
        inputs.append(item)
    return inputs


def run_one(
    base: Chronos2Pipeline,
    spec: dict,
    variant: str,
    training_records: list[dict],
    validation_records: list[dict],
    test_records: list[dict],
    context_length: int,
    training_batch_size: int,
    inference_batch_size: int,
    chunk_size: int,
    checkpoint_root: Path,
    prediction_root: Path,
    result_root: Path,
    smoke: bool,
    seed: int = SEED,
    run_tag: str = "",
    secondary_records: list[dict] | None = None,
    secondary_name: str = "calibration",
    evaluation_partition: str = "test",
) -> dict:
    with_vendor = variant == "vendor_covariate"
    identifier = f"chronos2_{variant}_{specification_id(spec)}"
    if run_tag:
        identifier += f"_{run_tag}"
    if smoke:
        identifier += "_smoke"
    result_path = result_root / f"{identifier}.json"
    prediction_path = prediction_root / f"{identifier}.parquet"
    secondary_path = prediction_root / f"{identifier}_{secondary_name}.parquet"
    if result_path.exists() and prediction_path.exists() and (secondary_records is None or secondary_path.exists()):
        return {"identifier": identifier, "status": "skipped"}
    run_spec = dict(spec)
    if smoke:
        run_spec["steps"] = 1
    set_seed(seed)
    checkpoint_dir = checkpoint_root / identifier
    lora_config = {
        "r": run_spec["rank"],
        "lora_alpha": 2 * run_spec["rank"],
        "target_modules": [
            "self_attention.q",
            "self_attention.v",
            "self_attention.k",
            "self_attention.o",
            "output_patch_embedding.output_layer",
        ],
    }
    fit_started = time.perf_counter()
    adapted = base.fit(
        inputs=adaptation_inputs(training_records, with_vendor),
        validation_inputs=adaptation_inputs(validation_records, with_vendor),
        prediction_length=96,
        finetune_mode="lora",
        lora_config=lora_config,
        context_length=context_length,
        learning_rate=run_spec["learning_rate"],
        num_steps=run_spec["steps"],
        batch_size=training_batch_size,
        min_past=context_length,
        output_dir=checkpoint_dir,
        logging_steps=max(1, min(25, run_spec["steps"])),
        save_steps=max(1, min(100, run_spec["steps"])),
        eval_steps=max(1, min(100, run_spec["steps"])),
        seed=seed,
        data_seed=seed,
        remove_printer_callback=True,
    )
    fit_seconds = time.perf_counter() - fit_started
    prediction, inference_seconds = predict_variant(
        adapted,
        test_records,
        with_vendor,
        batch_size=inference_batch_size,
        chunk_size=chunk_size,
    )
    prediction.to_parquet(prediction_path, index=False, compression="zstd")
    secondary_metrics = None
    secondary_seconds = None
    if secondary_records is not None:
        secondary_prediction, secondary_seconds = predict_variant(
            adapted,
            secondary_records,
            with_vendor,
            batch_size=inference_batch_size,
            chunk_size=chunk_size,
        )
        secondary_prediction.to_parquet(secondary_path, index=False, compression="zstd")
        secondary_metrics = score_predictions(secondary_prediction)
        del secondary_prediction
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "identifier": identifier,
        "method": "Chronos 2 LoRA adaptation on EMSx",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "variant": variant,
        "specification": run_spec,
        "random_seed": seed,
        "run_tag": run_tag,
        "context_length": context_length,
        "prediction_length": 96,
        "training_examples": len(training_records),
        "validation_examples": len(validation_records),
        "test_examples": len(test_records),
        "training_systems": len({record["site_id"] for record in training_records}),
        "training_origin_start": min(record["issue_time"] for record in training_records).isoformat(),
        "training_origin_end": max(record["issue_time"] for record in training_records).isoformat(),
        "checkpoint_selection": "Lowest validation loss among scheduled evaluations and the final step.",
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
        "checkpoint_bytes": sum(
            path.stat().st_size for path in checkpoint_dir.rglob("*") if path.is_file()
        ),
        "prediction_file": str(prediction_path),
        "evaluation_partition": evaluation_partition,
        "metrics": score_predictions(prediction),
        "secondary_partition": secondary_name if secondary_records is not None else None,
        "secondary_prediction_file": str(secondary_path) if secondary_records is not None else None,
        "secondary_metrics": secondary_metrics,
        "secondary_inference_seconds": secondary_seconds,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "chronos_forecasting": version("chronos-forecasting"),
            "transformers": version("transformers"),
            "peft": version("peft"),
        },
        "device": torch.cuda.get_device_name(0),
        "command": " ".join(sys.argv),
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    del adapted, prediction
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=672)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--inference-batch-size", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--validation-days", type=int, default=3)
    parser.add_argument("--only")
    parser.add_argument(
        "--variant",
        action="append",
        choices=["target_only", "vendor_covariate"],
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.prediction_root.mkdir(parents=True, exist_ok=True)
    args.result_root.mkdir(parents=True, exist_ok=True)
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    wide = load_wide(args.curve_root)
    validation_records = prepare_records(
        wide,
        args.raw_root,
        "tuning",
        args.context_length,
        profiles_per_system=1 if args.smoke else args.validation_days,
        require_complete_context=True,
    )
    test_records = prepare_records(
        wide,
        args.raw_root,
        "test",
        args.context_length,
        max_profiles=16 if args.smoke else None,
    )
    variants = args.variant or ["target_only", "vendor_covariate"]
    base = Chronos2Pipeline.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        device_map="cuda",
        local_files_only=True,
    )
    completed = []
    training_cache: dict[int, list[dict]] = {}
    for spec in specifications():
        spec_name = specification_id(spec)
        if args.only and spec_name != args.only:
            continue
        days = spec["days"]
        if days not in training_cache:
            training_cache[days] = prepare_records(
                wide,
                args.raw_root,
                "train",
                args.context_length,
                profiles_per_system=days,
                require_complete_context=True,
            )
        for variant in variants:
            print({"status": "starting", "specification": spec_name, "variant": variant}, flush=True)
            result = run_one(
                base,
                spec,
                variant,
                training_cache[days],
                validation_records,
                test_records,
                args.context_length,
                args.batch_size,
                args.inference_batch_size,
                args.chunk_size,
                args.checkpoint_root,
                args.prediction_root,
                args.result_root,
                args.smoke,
            )
            completed.append(result["identifier"])
            print(
                {
                    "status": result.get("status", "completed"),
                    "identifier": result["identifier"],
                    "metrics": result.get("metrics"),
                },
                flush=True,
            )
        if args.smoke:
            break
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "completed": completed,
        "specifications": specifications(),
        "variants": variants,
    }
    suffix = "_smoke" if args.smoke else ""
    (args.result_root / f"chronos2_manifest{suffix}.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
