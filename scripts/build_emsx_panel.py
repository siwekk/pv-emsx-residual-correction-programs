"""Build leakage-free EMSx forecast-correction panels.

Each EMSx row stores energy observed during the previous 15-minute interval.
The forecast ``pv_00`` is for the following interval, and ``pv_k`` is for
the interval ending 15 * (k + 1) minutes after the issue timestamp.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


from emsx_paths import WORK_ROOT as ROOT
RAW = ROOT / "data" / "raw" / "emsx"
OUTPUT = ROOT / "data" / "processed" / "emsx"


def parse_ints(value: str) -> list[int]:
    result = sorted({int(item) for item in value.split(",")})
    if not result or result[0] < 0 or result[-1] > 95:
        raise argparse.ArgumentTypeError("Leads must be integers from 0 through 95.")
    return result


def build_site(site: int, leads: list[int]) -> pd.DataFrame:
    fields = ["timestamp", "site_id", "actual_pv", *[f"pv_{lead:02d}" for lead in leads]]
    data = pd.read_csv(RAW / f"{site}.csv.gz", sep=";", usecols=fields)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    observed = data.set_index("timestamp")["actual_pv"]
    frames: list[pd.DataFrame] = []

    for lead in leads:
        valid_time = data["timestamp"] + pd.Timedelta(minutes=15 * (lead + 1))
        frame = pd.DataFrame(
            {
                "site_id": site,
                "issue_time": data["timestamp"].to_numpy(),
                "valid_time": valid_time.to_numpy(),
                "lead_steps": lead + 1,
                "forecast_pv_kwh": data[f"pv_{lead:02d}"].to_numpy(),
                "issue_actual_pv_kwh": data["actual_pv"].to_numpy(),
                "actual_pv_kwh": observed.reindex(valid_time).to_numpy(),
            }
        )
        frames.append(frame.dropna(subset=["forecast_pv_kwh", "actual_pv_kwh"]))

    panel = pd.concat(frames, ignore_index=True)
    panel["forecast_error_kwh"] = panel["actual_pv_kwh"] - panel["forecast_pv_kwh"]
    return panel.sort_values(["issue_time", "lead_steps"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", type=parse_ints, default=[1])
    parser.add_argument("--leads", type=parse_ints, default=[0, 3, 15, 95])
    args = parser.parse_args()

    for site in args.sites:
        panel = build_site(site, args.leads)
        destination = OUTPUT / "forecast_panels" / f"site={site}"
        destination.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(destination / "panel.parquet", index=False)
        print(
            {
                "site": site,
                "rows": len(panel),
                "leads": args.leads,
                "start": str(panel["issue_time"].min()),
                "end": str(panel["valid_time"].max()),
            }
        )


if __name__ == "__main__":
    main()
