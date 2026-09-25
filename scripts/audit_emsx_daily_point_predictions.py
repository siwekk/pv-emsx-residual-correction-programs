"""Check point-forecast files for finite values and identical delivery keys."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["site_id", "issue_time", "valid_time", "delivery_step"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.label) != len(args.prediction):
        raise ValueError("Supply one label per prediction file")
    reference = None
    results = {}
    for label, path in zip(args.label, args.prediction):
        frame = pd.read_parquet(path)
        prediction_columns = [column for column in frame if "prediction" in column and column != "vendor_prediction_kwh"]
        if not prediction_columns:
            prediction_columns = ["vendor_prediction_kwh"]
        keys = frame[KEYS].drop_duplicates()
        if reference is None:
            reference = keys
        comparison = reference.merge(keys, on=KEYS, how="outer", indicator=True)
        results[label] = {
            "file": str(path), "delivery_rows": int(len(frame)),
            "curves": int(frame[["site_id", "issue_time"]].drop_duplicates().shape[0]),
            "systems": int(frame.site_id.nunique()), "duplicate_key_rows": int(frame.duplicated(KEYS, keep=False).sum()),
            "prediction_columns": prediction_columns,
            "nonfinite_prediction_values": {column: int((~np.isfinite(frame[column].to_numpy(float))).sum()) for column in prediction_columns},
            "keys_missing_from_reference": int((comparison._merge == "right_only").sum()),
            "reference_keys_missing_from_method": int((comparison._merge == "left_only").sum()),
        }
    output = {"reference_method": args.label[0], "methods": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
