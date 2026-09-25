"""Evaluate frozen Moirai 2.0 variants on the EMSx daily-curve test sample."""

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
from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

from emsx_daily_curve_common import FORECAST_COLUMNS, TARGET_COLUMNS, load_wide


MODEL_ID = "Salesforce/moirai-2.0-R-small"
MODEL_REVISION = "30f43ff08c8494f4943ae1521e9d4e94a0fbb389"
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
            history_vendor = one_step.reindex(history_time - pd.Timedelta(minutes=15)).to_numpy(np.float32)
            records.append({
                "site_id": int(site_id), "issue_time": pd.Timestamp(row.issue_time), "history_time": history_time,
                "history_target": actual.reindex(history_time).to_numpy(np.float32),
                "history_vendor": history_vendor,
                "future_vendor": np.asarray([getattr(row, name) for name in FORECAST_COLUMNS], dtype=np.float32),
                "actual": np.asarray([getattr(row, name) for name in TARGET_COLUMNS], dtype=np.float32),
                "scale_kwh": float(row.scale_kwh),
            })
    return records


def make_dataset(records: list[dict], with_vendor: bool) -> ListDataset:
    entries = []
    for record in records:
        entry = {"start": pd.Period(record["history_time"][0].tz_localize(None), freq="15min"), "target": record["history_target"]}
        if with_vendor:
            past = record["history_vendor"]
            past_missing = ~np.isfinite(past)
            vendor = np.concatenate([np.nan_to_num(past, nan=0.0), record["future_vendor"]])
            missing = np.concatenate([past_missing.astype(np.float32), np.zeros(96, dtype=np.float32)])
            entry["feat_dynamic_real"] = np.stack([vendor, missing]).astype(np.float32)
        entries.append(entry)
    return ListDataset(entries, freq="15min")


def predict_variant(module: Moirai2Module, records: list[dict], with_vendor: bool, batch_size: int) -> tuple[pd.DataFrame, float]:
    model = Moirai2Forecast(
        module=module, prediction_length=96, context_length=len(records[0]["history_target"]),
        target_dim=1, feat_dynamic_real_dim=2 if with_vendor else 0, past_feat_dynamic_real_dim=0,
    )
    predictor = model.create_predictor(batch_size=batch_size, device="cuda")
    before = time.perf_counter()
    outputs = []
    forecasts = predictor.predict(make_dataset(records, with_vendor))
    for index, (record, forecast) in enumerate(zip(records, forecasts), 1):
        result = pd.DataFrame({
            "site_id": record["site_id"], "issue_time": record["issue_time"],
            "valid_time": pd.date_range(start=record["issue_time"] + pd.Timedelta(minutes=15), periods=96, freq="15min", tz="UTC"),
            "delivery_step": np.arange(1, 97, dtype=np.int16), "actual_pv_kwh": record["actual"],
            "vendor_prediction_kwh": record["future_vendor"], "scale_kwh": record["scale_kwh"],
        })
        for quantile in QUANTILES:
            values = np.asarray(forecast.quantile(quantile)).squeeze()
            if values.shape != (96,):
                raise RuntimeError(f"Unexpected Moirai quantile shape {values.shape}")
            result[f"q{int(round(100 * quantile)):02d}_kwh"] = np.maximum(values, 0)
        outputs.append(result)
        if index % 256 == 0 or index == len(records):
            print({"variant": "vendor_covariate" if with_vendor else "target_only", "completed_curves": index}, flush=True)
    return pd.concat(outputs, ignore_index=True), time.perf_counter() - before


def summarize(frame: pd.DataFrame) -> dict:
    target = frame.actual_pv_kwh.to_numpy(float)
    median = frame.q50_kwh.to_numpy(float)
    return {
        "delivery_rows": int(len(frame)), "curves": int(frame[["site_id", "issue_time"]].drop_duplicates().shape[0]),
        "systems": int(frame.site_id.nunique()), "median_mae_kwh": float(np.mean(np.abs(target - median))),
        "training_scale_normalized_mae": float(np.mean(np.abs(target - median) / frame.scale_kwh.to_numpy(float))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=672)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-curves", type=int)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)
    records = prepare_records(load_wide(args.curve_root), args.raw_root, args.context_length, args.max_curves)
    module = Moirai2Module.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    variants = {}
    for name, with_vendor in (("target_only", False), ("vendor_covariate", True)):
        output, runtime = predict_variant(module, records, with_vendor, args.batch_size)
        destination = args.output_root / f"emsx_daily_moirai2_{name}.parquet"
        output.to_parquet(destination, index=False)
        variants[name] = {"file": str(destination), "runtime_seconds": float(runtime), "metrics": summarize(output)}
    result = {
        "dataset": "EMSx daily complete 96-step curves", "started_utc": started.isoformat(),
        "completed_utc": datetime.now(timezone.utc).isoformat(), "command": " ".join(sys.argv),
        "source_curves": str(args.curve_root.resolve()), "source_raw": str(args.raw_root.resolve()),
        "method": "Moirai 2.0 zero-shot probabilistic forecasting", "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION, "uni2ts_version": version("uni2ts"), "gluonts_version": version("gluonts"),
        "torch_version": torch.__version__, "device": torch.cuda.get_device_name(0),
        "context_length": args.context_length, "prediction_length": 96, "quantiles": QUANTILES,
        "target_only_contract": "Causal PV target history only.",
        "vendor_covariate_contract": "Causal PV target history and historical one-step vendor forecasts, with the issued curve as a known future dynamic covariate; a binary missingness covariate accompanies historical vendor values.",
        "task_specific_training": False, "random_seed": None, "variants": variants,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
