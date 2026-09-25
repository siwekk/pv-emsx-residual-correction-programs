"""Derive forecast-curve features available at EMSx issue time."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


from emsx_paths import WORK_ROOT as ROOT
RAW = ROOT / "data" / "raw" / "emsx"
OUT = ROOT / "data" / "processed" / "emsx" / "curve_features"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    forecasts = [f"pv_{step:02d}" for step in range(96)]
    anchors = [0, 3, 7, 15, 31, 47, 63, 79, 95]
    for site in range(1, 71):
        frame = pd.read_csv(RAW / f"{site}.csv.gz", sep=";", usecols=["timestamp", *forecasts])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        curve = frame[forecasts].to_numpy(float)
        output = pd.DataFrame({"issue_time": frame["timestamp"], "site_id": site})
        for step in anchors: output[f"curve_{step:02d}"] = curve[:, step]
        output["curve_mean"] = np.nanmean(curve, axis=1)
        output["curve_std"] = np.nanstd(curve, axis=1)
        output["curve_max"] = np.nanmax(curve, axis=1)
        output["curve_energy"] = np.nansum(curve, axis=1)
        output["curve_peak_step"] = np.nanargmax(np.nan_to_num(curve, nan=-np.inf), axis=1)
        output["curve_target_slope"] = curve[:, 95] - curve[:, 91]
        output.to_parquet(OUT / f"site={site}.parquet", index=False)
        print({"site": site, "rows": len(output)}, flush=True)


if __name__ == "__main__":
    main()
