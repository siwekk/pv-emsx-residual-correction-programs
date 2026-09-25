"""Test 96 step Moirai adaptation with direct PV and vendor residual targets."""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import inspect
import json
import platform
import sys
import time
import textwrap
import traceback
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

from emsx_daily_curve_common import load_wide
from emsx_daily_fewshot_common import prepare_records, score_predictions, set_seed
from run_emsx_daily_moirai2 import MODEL_ID, MODEL_REVISION, QUANTILES, make_dataset, predict_variant
import run_emsx_daily_moirai2_fewshot_sweep as moirai_sweep


def gradient_safe_forecast_class() -> type[Moirai2Forecast]:
    """Stop gradients through recursive feedback while retaining its values.

    The released packed scaler modifies its scale tensor in place. In a 96 step
    recursive pass this prevents gradients through the first predictions fed
    back as context. Detaching those two feedback assignments preserves every
    forward prediction and still trains both parts of the 96 step output.
    """
    source = textwrap.dedent(inspect.getsource(Moirai2Forecast.forward))
    tree = ast.parse(source)

    class DetachRecursiveFeedback(ast.NodeTransformer):
        count = 0

        def visit_Assign(self, node: ast.Assign):
            self.generic_visit(node)
            if len(node.targets) != 1:
                return node
            target = node.targets[0]
            if not (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "expand_target"
            ):
                return node
            self.count += 1
            node.value = ast.Call(
                func=ast.Attribute(value=node.value, attr="detach", ctx=ast.Load()),
                args=[], keywords=[],
            )
            return node

    transformer = DetachRecursiveFeedback()
    tree = ast.fix_missing_locations(transformer.visit(tree))
    if transformer.count != 2:
        raise RuntimeError(
            f"Expected two recursive target writes, found {transformer.count}; "
            "review this Uni2TS version before training"
        )
    namespace = dict(Moirai2Forecast.forward.__globals__)
    exec(compile(tree, inspect.getsourcefile(Moirai2Forecast.forward) or "<moirai>", "exec"), namespace)
    return type("TruncatedGradientMoirai2Forecast", (Moirai2Forecast,), {
        "forward": namespace["forward"],
    })


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def convert_records(records: list[dict], target_mode: str) -> list[dict]:
    if target_mode == "direct":
        return records
    converted = []
    for record in records:
        item = dict(record)
        item["actual_pv"] = record["actual"]
        item["history_target"] = (
            record["history_target"] - record["history_vendor"]
        ).astype(np.float32)
        item["actual"] = (
            record["actual"] - record["future_vendor"]
        ).astype(np.float32)
        converted.append(item)
    return converted


def predict_residual(
    module: Moirai2Module,
    records: list[dict],
    with_vendor: bool,
    batch_size: int,
) -> tuple[pd.DataFrame, float]:
    model = Moirai2Forecast(
        module=module, prediction_length=96,
        context_length=len(records[0]["history_target"]), target_dim=1,
        feat_dynamic_real_dim=2 if with_vendor else 0,
        past_feat_dynamic_real_dim=0,
    )
    predictor = model.create_predictor(batch_size=batch_size, device="cuda")
    before = time.perf_counter()
    outputs = []
    forecasts = predictor.predict(make_dataset(records, with_vendor))
    for record, forecast in zip(records, forecasts, strict=True):
        vendor = record["future_vendor"].astype(float)
        result = pd.DataFrame({
            "site_id": record["site_id"], "issue_time": record["issue_time"],
            "valid_time": pd.date_range(
                start=record["issue_time"] + pd.Timedelta(minutes=15),
                periods=96, freq="15min", tz="UTC",
            ),
            "delivery_step": np.arange(1, 97, dtype=np.int16),
            "actual_pv_kwh": record["actual_pv"],
            "vendor_prediction_kwh": vendor,
            "scale_kwh": record["scale_kwh"],
        })
        for quantile in QUANTILES:
            values = np.asarray(forecast.quantile(quantile)).squeeze()
            if values.shape != (96,):
                raise RuntimeError(f"Unexpected Moirai quantile shape {values.shape}")
            result[f"q{int(round(100 * quantile)):02d}_kwh"] = np.maximum(
                vendor + values, 0.0
            )
        outputs.append(result)
    if len(outputs) != len(records):
        raise RuntimeError("Moirai returned fewer forecasts than evaluation records")
    return pd.concat(outputs, ignore_index=True), time.perf_counter() - before


def run_one(
    protocol: dict,
    protocol_hash: str,
    arm: dict,
    seed: int,
    partitions: dict[str, list[dict]],
    output_root: Path,
    smoke: bool,
) -> dict:
    target_mode = arm["target_mode"]
    variant = arm["variant"]
    with_vendor = variant == "vendor_covariate"
    name = f"moirai2_{target_mode}_{variant}_h96_s{seed}"
    if smoke:
        name += "_smoke"
    result_root = output_root / "results"
    prediction_root = output_root / "predictions"
    checkpoint_root = output_root / "checkpoints"
    for directory in (result_root, prediction_root, checkpoint_root):
        directory.mkdir(parents=True, exist_ok=True)
    result_path = result_root / f"{name}.json"
    calibration_path = prediction_root / f"{name}_calibration.parquet"
    test_path = prediction_root / f"{name}_test.parquet"
    checkpoint_path = checkpoint_root / f"{name}.pt"
    if all(path.is_file() for path in (result_path, calibration_path, test_path, checkpoint_path)):
        return {"identifier": name, "status": "skipped"}

    set_seed(seed)
    records = {
        split: convert_records(block, target_mode)
        for split, block in partitions.items()
    }
    module = Moirai2Module.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
    )
    if smoke:
        inputs, _, _ = moirai_sweep.make_batch(
            records["train"], np.asarray([0]), with_vendor,
        )
        constructor = dict(
            module=module, prediction_length=96,
            context_length=protocol["context_length"], target_dim=1,
            feat_dynamic_real_dim=2 if with_vendor else 0,
            past_feat_dynamic_real_dim=0,
        )
        original = Moirai2Forecast(**constructor).to("cuda").eval()
        gradient_safe = moirai_sweep.Moirai2Forecast(**constructor).to("cuda").eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            original_values = original(**inputs)
            safe_values = gradient_safe(**inputs)
        maximum_difference = float((original_values - safe_values).abs().max())
        if maximum_difference > 1e-5:
            raise RuntimeError(
                f"Gradient safe forward changes Moirai predictions: {maximum_difference}"
            )
        del original, gradient_safe, original_values, safe_values, inputs
    spec = {
        "days": protocol["adaptation_profiles_per_site"],
        "learning_rate": protocol["learning_rate"],
        "trainable_scope": protocol["trainable_scope"],
        "steps": 1 if smoke else protocol["steps"],
    }
    module, training = moirai_sweep.fit_module(
        module, records["train"], records["tuning"], with_vendor,
        protocol["context_length"], spec,
        protocol["training_batch_size"],
        1 if smoke else protocol["validation_interval"],
        seed=seed,
        adaptation_prediction_length=protocol["adaptation_prediction_length"],
        adaptation_window_offsets=tuple(protocol["adaptation_window_offsets"]),
    )
    selected_state = training.pop("selected_state")
    torch.save({
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "protocol_sha256": protocol_hash, "arm": arm,
        "seed": seed, "specification": spec,
        "best_step": training["best_step"], "state_dict": selected_state,
    }, checkpoint_path)
    module.to("cpu")
    torch.cuda.empty_cache()

    predictions = {}
    metrics = {}
    inference_seconds = {}
    for split, path in (("calibration", calibration_path), ("test", test_path)):
        if target_mode == "direct":
            frame, elapsed = predict_variant(
                module, records[split], with_vendor, protocol["inference_batch_size"],
            )
        else:
            frame, elapsed = predict_residual(
                module, records[split], with_vendor, protocol["inference_batch_size"],
            )
        frame.to_parquet(path, index=False, compression="zstd")
        predictions[split] = str(path.resolve())
        metrics[split] = score_predictions(frame)
        inference_seconds[split] = elapsed
        del frame

    result = {
        "generated_utc": utc_now(), "identifier": name,
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "protocol_sha256": protocol_hash,
        "arm": arm, "seed": seed, "specification": spec,
        "context_length": protocol["context_length"],
        "adaptation_prediction_length": protocol["adaptation_prediction_length"],
        "forecast_steps": protocol["forecast_steps"],
        "training_examples": len(records["train"]),
        "validation_examples": len(records["tuning"]),
        "calibration_examples": len(records["calibration"]),
        "test_examples": len(records["test"]),
        "checkpoint_file": str(checkpoint_path.resolve()),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "prediction_files": predictions,
        "metrics": metrics,
        "inference_seconds": inference_seconds,
        **training,
        "software": {
            "python": platform.python_version(), "torch": torch.__version__,
            "uni2ts": version("uni2ts"), "gluonts": version("gluonts"),
        },
        "device": torch.cuda.get_device_name(0),
        "command": " ".join(sys.argv),
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    del module, records, selected_state
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--curve-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--arm", action="append", help="Restrict to target_mode:variant")
    parser.add_argument("--seed", action="append", type=int, help="Restrict to a protocol seed")
    args = parser.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    if protocol["protocol_version"] != 1:
        raise ValueError("Unsupported protocol version")
    if (protocol["model_id"], protocol["model_revision"]) != (MODEL_ID, MODEL_REVISION):
        raise RuntimeError("Moirai checkpoint differs from the protocol")
    if protocol["forecast_steps"] != 96 or protocol["adaptation_prediction_length"] != 96:
        raise RuntimeError("This experiment requires a 96 step horizon")
    moirai_sweep.Moirai2Forecast = gradient_safe_forecast_class()
    code_root = Path(__file__).resolve().parents[1]
    output = args.output_root.resolve()
    if output == code_root or code_root in output.parents:
        raise ValueError("Output must be outside the source repository")

    arms = protocol["arms"]
    seeds = protocol["seeds"]
    if args.arm:
        wanted = set(args.arm)
        arms = [arm for arm in arms if f"{arm['target_mode']}:{arm['variant']}" in wanted]
        if len(arms) != len(wanted):
            raise ValueError("Unknown arm requested")
    if args.seed:
        if not set(args.seed) <= set(seeds):
            raise ValueError("Requested seed is absent from the protocol")
        seeds = args.seed
    if args.smoke:
        torch.autograd.set_detect_anomaly(True)
        arms = arms[:1]
        seeds = seeds[:1]
        output = output / "smoke"

    wide = load_wide(args.curve_root)
    partitions = {
        "train": prepare_records(
            wide, args.raw_root, protocol["training_partition"],
            protocol["context_length"],
            profiles_per_system=protocol["adaptation_profiles_per_site"],
            require_complete_context=True,
            max_profiles=16 if args.smoke else None,
        ),
        "tuning": prepare_records(
            wide, args.raw_root, protocol["checkpoint_selection_partition"],
            protocol["context_length"],
            profiles_per_system=protocol["validation_profiles_per_site"],
            require_complete_context=True,
            max_profiles=8 if args.smoke else None,
        ),
        "calibration": prepare_records(
            wide, args.raw_root, protocol["primary_evaluation_partition"],
            protocol["context_length"], max_profiles=8 if args.smoke else None,
        ),
        "test": prepare_records(
            wide, args.raw_root, protocol["secondary_evaluation_partition"],
            protocol["context_length"], max_profiles=8 if args.smoke else None,
        ),
    }
    if any(not block for block in partitions.values()):
        raise RuntimeError("A partition contains no profiles")
    if {x["item_id"] for x in partitions["calibration"]} & {
        x["item_id"] for x in partitions["test"]
    }:
        raise RuntimeError("Calibration and test profiles overlap")
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.json"
    completed, failures = [], []
    print(json.dumps({
        "event": "prepared", "utc": utc_now(), "protocol_sha256": protocol_hash,
        "partitions": {name: len(block) for name, block in partitions.items()},
        "planned_fits": len(arms) * len(seeds),
    }), flush=True)
    for seed in seeds:
        for arm in arms:
            print(json.dumps({"event": "starting", "seed": seed, "arm": arm}), flush=True)
            try:
                result = run_one(
                    protocol, protocol_hash, arm, seed, partitions, output, args.smoke,
                )
                completed.append({
                    "seed": seed, "arm": arm, "identifier": result["identifier"],
                    "status": result.get("status", "completed"),
                    "calibration_mae": result.get("metrics", {}).get("calibration", {}).get("pooled_median_mae_kwh"),
                    "test_mae": result.get("metrics", {}).get("test", {}).get("pooled_median_mae_kwh"),
                })
                print(json.dumps({"event": "completed", **completed[-1]}), flush=True)
            except Exception as error:
                failures.append({
                    "seed": seed, "arm": arm, "error": repr(error),
                    "traceback": traceback.format_exc(),
                })
                print(json.dumps({"event": "failed", "seed": seed, "arm": arm,
                                  "error": repr(error)}), flush=True)
                if len(failures) >= 2 and not completed:
                    raise RuntimeError("Two initial fits failed; stopping") from error
            progress_path.write_text(json.dumps({
                "updated_utc": utc_now(), "protocol_sha256": protocol_hash,
                "planned_fits": len(arms) * len(seeds),
                "completed": completed, "failures": failures,
            }, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"{len(failures)} fits failed; see {progress_path}")


if __name__ == "__main__":
    main()
