"""Run a reproducible Moirai 2.0 adaptation sweep on EMSx."""

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
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

from emsx_daily_curve_common import load_wide
from emsx_daily_fewshot_common import prepare_records, score_predictions, set_seed
from run_emsx_daily_moirai2 import MODEL_ID, MODEL_REVISION, predict_variant


SEED = 20260924
ADAPTATION_PREDICTION_LENGTH = 64


def specifications() -> list[dict]:
    specs = [
        {
            "days": days,
            "learning_rate": 1e-4,
            "trainable_scope": "output_head",
            "steps": 200,
        }
        for days in (1, 3, 7)
    ]
    specs.extend(
        {
            "days": 7,
            "learning_rate": learning_rate,
            "trainable_scope": "output_head",
            "steps": 200,
        }
        for learning_rate in (3e-5, 3e-4)
    )
    specs.extend(
        [
            {
                "days": 7,
                "learning_rate": 3e-5,
                "trainable_scope": "last_block_and_head",
                "steps": 200,
            },
            {
                "days": 7,
                "learning_rate": 1e-5,
                "trainable_scope": "full",
                "steps": 200,
            },
        ]
    )
    return list({specification_id(spec): spec for spec in specs}.values())


def specification_id(spec: dict) -> str:
    learning_rate = f"{spec['learning_rate']:.0e}".replace("-", "m")
    scope = {
        "output_head": "head",
        "last_block_and_head": "last",
        "full": "full",
    }[spec["trainable_scope"]]
    return f"d{spec['days']}_lr{learning_rate}_{scope}_n{spec['steps']}"


def configure_trainable(module: Moirai2Module, scope: str) -> list[str]:
    for parameter in module.parameters():
        parameter.requires_grad = False
    selected = []
    for name, parameter in module.named_parameters():
        enabled = (
            scope == "full"
            or name.startswith("out_proj.")
            or (
                scope == "last_block_and_head"
                and name.startswith(f"encoder.layers.{module.num_layers - 1}.")
            )
        )
        parameter.requires_grad = enabled
        if enabled:
            selected.append(name)
    if not selected:
        raise RuntimeError(f"No parameters selected for scope {scope}")
    return selected


def adaptation_windows(
    records: list[dict],
    context_length: int,
    prediction_length: int = ADAPTATION_PREDICTION_LENGTH,
    offsets: tuple[int, ...] = (0, 32),
) -> list[dict]:
    windows = []
    for record in records:
        target = np.concatenate([record["history_target"], record["actual"]])
        vendor = np.concatenate([record["history_vendor"], record["future_vendor"]])
        for offset in offsets:
            future_start = offset + context_length
            future_end = future_start + prediction_length
            if future_end > len(target):
                raise ValueError("Adaptation window extends beyond the available target")
            windows.append(
                {
                    "history_target": target[offset:future_start].astype(np.float32),
                    "history_vendor": vendor[offset:future_start].astype(np.float32),
                    "future_vendor": vendor[future_start:future_end].astype(np.float32),
                    "actual": target[future_start:future_end].astype(np.float32),
                    "scale_kwh": record["scale_kwh"],
                }
            )
    return windows


def make_forecast(
    module: Moirai2Module,
    context_length: int,
    with_vendor: bool,
    prediction_length: int,
) -> Moirai2Forecast:
    return Moirai2Forecast(
        module=module,
        prediction_length=prediction_length,
        context_length=context_length,
        target_dim=1,
        feat_dynamic_real_dim=2 if with_vendor else 0,
        past_feat_dynamic_real_dim=0,
    ).to("cuda")


def make_batch(records: list[dict], indices: np.ndarray, with_vendor: bool) -> tuple[dict, torch.Tensor, torch.Tensor]:
    selected = [records[int(index)] for index in indices]
    history = np.stack([record["history_target"] for record in selected]).astype(np.float32)
    observed = np.isfinite(history)
    inputs = {
        "past_target": torch.from_numpy(np.nan_to_num(history, nan=0.0)[..., None]).to("cuda"),
        "past_observed_target": torch.from_numpy(observed[..., None]).to("cuda"),
        "past_is_pad": torch.zeros(history.shape, dtype=torch.bool, device="cuda"),
    }
    if with_vendor:
        past_vendor = np.stack([record["history_vendor"] for record in selected]).astype(np.float32)
        future_vendor = np.stack([record["future_vendor"] for record in selected]).astype(np.float32)
        missing = ~np.isfinite(past_vendor)
        vendor = np.concatenate([np.nan_to_num(past_vendor, nan=0.0), future_vendor], axis=1)
        missing_feature = np.concatenate(
            [missing.astype(np.float32), np.zeros_like(future_vendor)], axis=1
        )
        features = np.stack([vendor, missing_feature], axis=2)
        inputs["feat_dynamic_real"] = torch.from_numpy(features).to("cuda")
        inputs["observed_feat_dynamic_real"] = torch.ones(
            features.shape, dtype=torch.bool, device="cuda"
        )
    target = torch.from_numpy(
        np.stack([record["actual"] for record in selected]).astype(np.float32)
    ).to("cuda")
    scale = torch.tensor(
        [record["scale_kwh"] for record in selected],
        dtype=torch.float32,
        device="cuda",
    )
    return inputs, target, scale


def normalized_pinball_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    quantile_levels: list[float],
) -> torch.Tensor:
    quantiles = torch.tensor(
        quantile_levels,
        dtype=prediction.dtype,
        device=prediction.device,
    ).view(1, -1, 1)
    error = target.unsqueeze(1) - prediction
    loss = torch.maximum(quantiles * error, (quantiles - 1.0) * error)
    return (loss / scale.view(-1, 1, 1).clamp_min(1e-6)).mean()


@torch.no_grad()
def validation_loss(
    forecast: Moirai2Forecast,
    records: list[dict],
    with_vendor: bool,
    batch_size: int,
) -> float:
    forecast.eval()
    losses = []
    weights = []
    for start in range(0, len(records), batch_size):
        indices = np.arange(start, min(start + batch_size, len(records)))
        inputs, target, scale = make_batch(records, indices, with_vendor)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = forecast(**inputs)
            loss = normalized_pinball_loss(
                prediction,
                target,
                scale,
                forecast.module.quantile_levels,
            )
        losses.append(float(loss))
        weights.append(len(indices))
    return float(np.average(losses, weights=weights))


def fit_module(
    module: Moirai2Module,
    training_records: list[dict],
    validation_records: list[dict],
    with_vendor: bool,
    context_length: int,
    spec: dict,
    batch_size: int,
    validation_interval: int,
    seed: int = SEED,
    adaptation_prediction_length: int = ADAPTATION_PREDICTION_LENGTH,
    adaptation_window_offsets: tuple[int, ...] = (0, 32),
) -> tuple[Moirai2Module, dict]:
    selected_names = configure_trainable(module, spec["trainable_scope"])
    trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    total_count = sum(parameter.numel() for parameter in module.parameters())
    training_windows = adaptation_windows(
        training_records, context_length,
        adaptation_prediction_length, adaptation_window_offsets,
    )
    validation_windows = adaptation_windows(
        validation_records, context_length,
        adaptation_prediction_length, adaptation_window_offsets,
    )
    forecast = make_forecast(
        module,
        context_length,
        with_vendor,
        adaptation_prediction_length,
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=spec["learning_rate"],
        weight_decay=1e-2,
        betas=(0.9, 0.98),
        eps=1e-6,
    )
    rng = np.random.default_rng(seed)
    history = []
    initial_loss = validation_loss(forecast, validation_windows, with_vendor, batch_size)
    history.append({"step": 0, "validation_normalized_pinball": initial_loss})
    best_loss = initial_loss
    best_step = 0
    best_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }
    fit_started = time.perf_counter()
    for step in range(1, spec["steps"] + 1):
        forecast.train()
        indices = rng.integers(0, len(training_windows), size=batch_size)
        inputs, target, scale = make_batch(training_windows, indices, with_vendor)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = forecast(**inputs)
            loss = normalized_pinball_loss(
                prediction,
                target,
                scale,
                forecast.module.quantile_levels,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        if step % validation_interval == 0 or step == spec["steps"]:
            current_loss = validation_loss(
                forecast,
                validation_windows,
                with_vendor,
                batch_size,
            )
            history.append(
                {
                    "step": step,
                    "training_normalized_pinball": float(loss.detach()),
                    "validation_normalized_pinball": current_loss,
                }
            )
            print(
                {
                    "step": step,
                    "training_loss": float(loss.detach()),
                    "validation_loss": current_loss,
                },
                flush=True,
            )
            if current_loss < best_loss:
                best_loss = current_loss
                best_step = step
                best_state = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in module.named_parameters()
                    if parameter.requires_grad
                }
    fit_seconds = time.perf_counter() - fit_started
    module.load_state_dict(best_state, strict=False)
    return module, {
        "fit_seconds": fit_seconds,
        "best_step": best_step,
        "best_validation_normalized_pinball": best_loss,
        "initial_validation_normalized_pinball": initial_loss,
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
        "trainable_parameter_names": selected_names,
        "adaptation_prediction_length": adaptation_prediction_length,
        "adaptation_window_offsets": list(adaptation_window_offsets),
        "training_windows": len(training_windows),
        "validation_windows": len(validation_windows),
        "validation_history": history,
        "selected_state": best_state,
    }


def run_one(
    spec: dict,
    variant: str,
    training_records: list[dict],
    validation_records: list[dict],
    test_records: list[dict],
    context_length: int,
    training_batch_size: int,
    inference_batch_size: int,
    validation_interval: int,
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
    identifier = f"moirai2_{variant}_{specification_id(spec)}"
    if run_tag:
        identifier += f"_{run_tag}"
    if smoke:
        identifier += "_smoke"
    result_path = result_root / f"{identifier}.json"
    prediction_path = prediction_root / f"{identifier}.parquet"
    secondary_path = prediction_root / f"{identifier}_{secondary_name}.parquet"
    checkpoint_path = checkpoint_root / f"{identifier}.pt"
    if result_path.exists() and prediction_path.exists() and (secondary_records is None or secondary_path.exists()):
        return {"identifier": identifier, "status": "skipped"}
    run_spec = dict(spec)
    if smoke:
        run_spec["steps"] = 1
    set_seed(seed)
    module = Moirai2Module.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
    )
    module, training = fit_module(
        module,
        training_records,
        validation_records,
        with_vendor,
        context_length,
        run_spec,
        training_batch_size,
        1 if smoke else validation_interval,
        seed,
    )
    selected_state = training.pop("selected_state")
    torch.save(
        {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "specification": run_spec,
            "variant": variant,
            "random_seed": seed,
            "best_step": training["best_step"],
            "state_dict": selected_state,
        },
        checkpoint_path,
    )
    module.to("cpu")
    torch.cuda.empty_cache()
    prediction, inference_seconds = predict_variant(
        module,
        test_records,
        with_vendor,
        inference_batch_size,
    )
    prediction.to_parquet(prediction_path, index=False, compression="zstd")
    secondary_metrics = None
    secondary_seconds = None
    if secondary_records is not None:
        secondary_prediction, secondary_seconds = predict_variant(
            module,
            secondary_records,
            with_vendor,
            inference_batch_size,
        )
        secondary_prediction.to_parquet(secondary_path, index=False, compression="zstd")
        secondary_metrics = score_predictions(secondary_prediction)
        del secondary_prediction
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "identifier": identifier,
        "method": "Moirai 2.0 supervised adaptation on EMSx",
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
        "loss": "Mean pinball loss on the native nine quantiles, divided by the training scale of each system.",
        "checkpoint_selection": "Lowest tuning loss, including the unchanged step 0 model.",
        **training,
        "inference_seconds": inference_seconds,
        "checkpoint_file": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
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
            "uni2ts": version("uni2ts"),
            "gluonts": version("gluonts"),
            "lightning": version("lightning"),
        },
        "device": torch.cuda.get_device_name(0),
        "command": " ".join(sys.argv),
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    del module, prediction, selected_state
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
    parser.add_argument("--training-batch-size", type=int, default=16)
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--validation-days", type=int, default=1)
    parser.add_argument("--validation-interval", type=int, default=50)
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
                spec,
                variant,
                training_cache[days],
                validation_records,
                test_records,
                args.context_length,
                args.training_batch_size,
                args.inference_batch_size,
                args.validation_interval,
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
    (args.result_root / f"moirai2_manifest{suffix}.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
