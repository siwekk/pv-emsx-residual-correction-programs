"""Evaluate frozen Chronos-2 variants on the EMSx daily-curve test sample."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline

from emsx_daily_curve_common import FORECAST_COLUMNS, TARGET_COLUMNS, load_wide


MODEL_ID = "amazon/chronos-2"
MODEL_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
QUANTILES = [.05, .10, .20, .30, .40, .50, .60, .70, .80, .90, .95]


def prepare_records(wide: pd.DataFrame, raw_root: Path, context_length: int, max_curves: int | None) -> list[dict]:
    test = wide.loc[(wide.split == "test") & wide.complete_case].sort_values(["site_id", "issue_time"])
    if max_curves is not None:
        test = test.head(max_curves)
    records = []
    for site_id, origins in test.groupby("site_id", observed=True, sort=False):
        raw = pd.read_csv(raw_root / f"{int(site_id)}.csv.gz", sep=";", usecols=["timestamp", "actual_pv", "pv_00"])
        raw["timestamp"] = pd.to_datetime(raw.timestamp, utc=True)
        raw = raw.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
        actual = raw.set_index("timestamp").actual_pv
        one_step = raw.set_index("timestamp").pv_00
        for row in origins.itertuples(index=False):
            history_time = pd.date_range(end=row.issue_time, periods=context_length, freq="15min", tz="UTC")
            future_time = pd.date_range(start=row.issue_time + pd.Timedelta(minutes=15), periods=96, freq="15min", tz="UTC")
            records.append({
                "item_id": f"{int(site_id)}_{pd.Timestamp(row.issue_time).strftime('%Y%m%dT%H%M%SZ')}",
                "site_id": int(site_id),
                "issue_time": pd.Timestamp(row.issue_time),
                "history_time": history_time,
                "history_target": actual.reindex(history_time).to_numpy(np.float32),
                "history_vendor": one_step.reindex(history_time - pd.Timedelta(minutes=15)).to_numpy(np.float32),
                "future_time": future_time,
                "future_vendor": np.asarray([getattr(row, name) for name in FORECAST_COLUMNS], dtype=np.float32),
                "actual": np.asarray([getattr(row, name) for name in TARGET_COLUMNS], dtype=np.float32),
                "scale_kwh": float(row.scale_kwh),
            })
    return records


def predict_variant(pipeline: Chronos2Pipeline, records: list[dict], with_vendor: bool, batch_size: int, chunk_size: int) -> tuple[pd.DataFrame, float]:
    outputs = []
    before = time.perf_counter()
    for start in range(0, len(records), chunk_size):
        chunk = records[start:start + chunk_size]
        context_parts = []
        future_parts = []
        for record in chunk:
            context = pd.DataFrame({"item_id": record["item_id"], "timestamp": record["history_time"].tz_localize(None), "target": record["history_target"]})
            if with_vendor:
                context["vendor_forecast"] = record["history_vendor"]
                future_parts.append(pd.DataFrame({"item_id": record["item_id"], "timestamp": record["future_time"].tz_localize(None), "vendor_forecast": record["future_vendor"]}))
            context_parts.append(context)
        prediction = pipeline.predict_df(
            pd.concat(context_parts, ignore_index=True),
            future_df=pd.concat(future_parts, ignore_index=True) if with_vendor else None,
            id_column="item_id", timestamp_column="timestamp", target="target",
            prediction_length=96, quantile_levels=QUANTILES, batch_size=batch_size,
            context_length=len(chunk[0]["history_target"]), freq="15min", cross_learning=False,
        )
        prediction = prediction.sort_values(["item_id", "timestamp"])
        lookup = {record["item_id"]: record for record in chunk}
        for item_id, block in prediction.groupby("item_id", sort=False):
            record = lookup[item_id]
            block = block.sort_values("timestamp")
            if len(block) != 96:
                raise RuntimeError(f"Chronos-2 returned {len(block)} rows for {item_id}")
            result = pd.DataFrame({
                "site_id": record["site_id"], "issue_time": record["issue_time"],
                "valid_time": record["future_time"], "delivery_step": np.arange(1, 97, dtype=np.int16),
                "actual_pv_kwh": record["actual"], "vendor_prediction_kwh": record["future_vendor"],
                "scale_kwh": record["scale_kwh"],
            })
            for quantile in QUANTILES:
                result[f"q{int(round(100 * quantile)):02d}_kwh"] = np.maximum(block[str(quantile)].to_numpy(float), 0)
            outputs.append(result)
        print({"variant": "vendor_covariate" if with_vendor else "target_only", "completed_curves": min(start + len(chunk), len(records))}, flush=True)
    return pd.concat(outputs, ignore_index=True), time.perf_counter() - before


def summarize(frame: pd.DataFrame) -> dict:
    target = frame.actual_pv_kwh.to_numpy(float)
    median = frame.q50_kwh.to_numpy(float)
    return {
        "delivery_rows": int(len(frame)),
        "curves": int(frame[["site_id", "issue_time"]].drop_duplicates().shape[0]),
        "systems": int(frame.site_id.nunique()),
        "median_mae_kwh": float(np.mean(np.abs(target - median))),
        "training_scale_normalized_mae": float(np.mean(np.abs(target - median) / frame.scale_kwh.to_numpy(float))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=672)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-curves", type=int)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)
    records = prepare_records(load_wide(args.curve_root), args.raw_root, args.context_length, args.max_curves)
    pipeline = Chronos2Pipeline.from_pretrained(MODEL_ID, revision=MODEL_REVISION, device_map="cuda", local_files_only=True)
    variants = {}
    args.output_root.mkdir(parents=True, exist_ok=True)
    for name, with_vendor in (("target_only", False), ("vendor_covariate", True)):
        output, runtime = predict_variant(pipeline, records, with_vendor, args.batch_size, args.chunk_size)
        destination = args.output_root / f"emsx_daily_chronos2_{name}.parquet"
        output.to_parquet(destination, index=False)
        variants[name] = {"file": str(destination), "runtime_seconds": float(runtime), "metrics": summarize(output)}
    result = {
        "dataset": "EMSx daily complete 96-step curves",
        "started_utc": started.isoformat(), "completed_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv), "source_curves": str(args.curve_root.resolve()), "source_raw": str(args.raw_root.resolve()),
        "method": "Chronos-2 zero-shot probabilistic forecasting",
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "chronos_version": version("chronos-forecasting"),
        "torch_version": torch.__version__, "device": torch.cuda.get_device_name(0),
        "context_length": args.context_length, "prediction_length": 96, "quantiles": QUANTILES,
        "target_only_contract": "Causal PV target history only.",
        "vendor_covariate_contract": "Causal PV target history and historical one-step vendor forecasts, with the issued 96-step vendor curve as a known future covariate.",
        "task_specific_training": False, "random_seed": None, "variants": variants,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
