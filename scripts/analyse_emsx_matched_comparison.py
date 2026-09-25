"""Report matched EMSx errors and a site-aware temporal bootstrap interval."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


from emsx_paths import WORK_ROOT as ROOT
RESULTS = ROOT / "results"


def bootstrap(frame: pd.DataFrame, columns: list[str], draws: int = 2000) -> dict[str, dict[str, float]]:
    frame = frame.copy(); frame["day"] = pd.to_datetime(frame["issue_time"], utc=True).dt.date
    daily = frame.groupby(["site_id", "day"], observed=True)[columns].mean().reset_index()
    sites = daily["site_id"].unique(); rng = np.random.default_rng(20260827); samples = np.empty((draws, len(columns)))
    by_site = {site: daily.loc[daily["site_id"] == site, columns].to_numpy() for site in sites}
    for draw in range(draws):
        chosen_sites = rng.choice(sites, len(sites), replace=True); values = []
        for site in chosen_sites:
            site_days = by_site[site]
            values.append(site_days[rng.integers(0, len(site_days), len(site_days))])
        samples[draw] = np.vstack(values).mean(axis=0)
    return {column: {"mean": float(samples[:, index].mean()), "ci95_low": float(np.quantile(samples[:, index], .025)), "ci95_high": float(np.quantile(samples[:, index], .975))} for index, column in enumerate(columns)}


def main() -> None:
    qgbm = pd.read_parquet(RESULTS / "emsx_qgbm_lead_96_history_15.parquet")
    tcn = pd.read_parquet(RESULTS / "emsx_tcn_lead_96.parquet")
    keys = ["issue_time", "site_id", "actual_pv_kwh", "forecast_pv_kwh", "scale_kwh"]
    data = qgbm.merge(tcn[keys + ["tcn_prediction_kwh"]], on=keys, how="inner", validate="one_to_one")
    data["vendor_abs_error"] = np.abs(data.actual_pv_kwh - data.forecast_pv_kwh)
    data["qgbm_abs_error"] = np.abs(data.actual_pv_kwh - data.qgbm_prediction_kwh)
    data["tcn_abs_error"] = np.abs(data.actual_pv_kwh - data.tcn_prediction_kwh)
    data["qgbm_gain_kwh"] = data.vendor_abs_error - data.qgbm_abs_error
    data["tcn_gain_vs_qgbm_kwh"] = data.qgbm_abs_error - data.tcn_abs_error
    data["tcn_gain_vs_vendor_kwh"] = data.vendor_abs_error - data.tcn_abs_error
    metrics = {name: float(data[name].mean()) for name in ("vendor_abs_error", "qgbm_abs_error", "tcn_abs_error", "qgbm_gain_kwh", "tcn_gain_vs_qgbm_kwh", "tcn_gain_vs_vendor_kwh")}
    result = {"dataset": "EMSx", "horizon_hours": 24., "matched_rows": int(len(data)), "sites": int(data.site_id.nunique()), "method": "Hierarchical bootstrap of site-day mean errors, resampling sites and then days within sites, 2,000 draws.", "mean_errors_kwh": metrics, "bootstrap_kwh": bootstrap(data, ["qgbm_gain_kwh", "tcn_gain_vs_qgbm_kwh", "tcn_gain_vs_vendor_kwh"])}
    (RESULTS / "emsx_matched_comparison_lead_96.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
