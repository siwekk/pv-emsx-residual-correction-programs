"""Create manuscript tables, figures, and conditional analyses for the EMSx daily benchmark."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEED = 20260906
KEYS = ["site_id", "issue_time", "valid_time", "delivery_step"]
QUANTILES = np.asarray([.05, .10, .20, .30, .40, .50, .60, .70, .80, .90, .95])
QCOLS = [f"q{int(round(100 * q)):02d}_kwh" for q in QUANTILES]
COLORS = {"Vendor": "#595959", "Residual LightGBM": "#0072B2", "Seasonal naive": "#009E73", "ARIMA": "#D55E00", "Persistence": "#CC79A7", "Chronos-2 + vendor": "#E69F00", "Chronos-2 target-only": "#56B4E9", "Moirai 2.0 target-only": "#8C564B", "Moirai 2.0 + vendor": "#9467BD"}


def point_metrics(frame: pd.DataFrame, column: str) -> dict:
    error = np.abs(frame.actual_pv_kwh.to_numpy(float) - frame[column].to_numpy(float))
    curve = pd.DataFrame({"site_id": frame.site_id, "issue_time": frame.issue_time, "error": error}).groupby(["site_id", "issue_time"], observed=True).error.mean()
    system = curve.groupby("site_id").mean()
    return {"pooled_mae_kwh": float(error.mean()), "equal_system_mae_kwh": float(system.mean()), "normalized_mae": float(np.mean(error / frame.scale_kwh.to_numpy(float)))}


def hierarchical_ci(frame: pd.DataFrame, a: str, b: str, draws: int) -> dict:
    curve = frame.assign(difference=np.abs(frame.actual_pv_kwh - frame[b]) - np.abs(frame.actual_pv_kwh - frame[a])).groupby(["site_id", "issue_time"], observed=True).difference.mean()
    site_arrays = [block.to_numpy(float) for _, block in curve.groupby(level=0)]
    observed = float(np.mean([values.mean() for values in site_arrays]))
    rng = np.random.default_rng(SEED)
    boot = np.empty(draws)
    for draw in range(draws):
        chosen = rng.integers(0, len(site_arrays), len(site_arrays))
        boot[draw] = np.mean([rng.choice(site_arrays[index], len(site_arrays[index]), replace=True).mean() for index in chosen])
    low, high = np.quantile(boot, [.025, .975])
    return {"equal_system_mae_reduction_kwh": observed, "ci95_low_kwh": float(low), "ci95_high_kwh": float(high), "draws": draws, "seed": SEED}


def foundation_frame(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    qvalues = np.sort(frame[QCOLS].to_numpy(float), axis=1)
    output = frame[KEYS + ["actual_pv_kwh", "vendor_prediction_kwh", "scale_kwh"]].copy()
    output[label] = qvalues[:, 5]
    for index, column in enumerate(QCOLS):
        output[column] = qvalues[:, index]
    return output


def missing_context_keys(reference: pd.DataFrame, raw_root: Path, context_length: int) -> set[tuple[int, pd.Timestamp]]:
    origins = reference[["site_id", "issue_time"]].drop_duplicates()
    missing = set()
    for site_id, block in origins.groupby("site_id", sort=False):
        raw = pd.read_csv(raw_root / f"{int(site_id)}.csv.gz", sep=";", usecols=["timestamp", "actual_pv"])
        raw["timestamp"] = pd.to_datetime(raw.timestamp, utc=True)
        actual = raw.drop_duplicates("timestamp", keep="last").set_index("timestamp").actual_pv
        for issue_time in pd.DatetimeIndex(block.issue_time):
            history = pd.date_range(end=issue_time, periods=context_length, freq="15min", tz="UTC")
            if actual.reindex(history).isna().any():
                missing.add((int(site_id), issue_time))
    return missing


def probabilistic_metrics(frame: pd.DataFrame) -> dict:
    y = frame.actual_pv_kwh.to_numpy(float)
    q = frame[QCOLS].to_numpy(float)
    losses = np.column_stack([np.maximum(tau * (y - q[:, i]), (tau - 1) * (y - q[:, i])) for i, tau in enumerate(QUANTILES)])
    interval_terms = []
    for alpha, lo, hi in ((.8, 1, 9), (.6, 2, 8), (.4, 3, 7), (.2, 4, 6)):
        score = q[:, hi] - q[:, lo] + 2 / alpha * (q[:, lo] - y) * (y < q[:, lo]) + 2 / alpha * (y - q[:, hi]) * (y > q[:, hi])
        interval_terms.append(alpha / 2 * score)
    wis = (0.5 * np.abs(y - q[:, 5]) + np.sum(interval_terms, axis=0)) / 4.5
    return {
        "median_mae_kwh": float(np.mean(np.abs(y - q[:, 5]))),
        "mean_pinball_kwh": float(losses.mean()),
        "grid_crps_kwh": float(2 * np.trapz(losses.mean(axis=0), QUANTILES)),
        "wis_kwh": float(wis.mean()),
        "coverage_90": float(np.mean((y >= q[:, 0]) & (y <= q[:, -1]))),
        "width_90_kwh": float(np.mean(q[:, -1] - q[:, 0])),
    }


def write_point_table(path: Path, metrics: dict, intervals: dict) -> None:
    order = ["Residual LightGBM", "Vendor", "Chronos-2 + vendor", "Chronos-2 target-only", "Seasonal naive", "Moirai 2.0 target-only", "ARIMA", "Moirai 2.0 + vendor", "Persistence"]
    lines = ["\\begin{table*}[t]", "\\centering", "\\caption{Point forecasting performance on 854,208 matched deliveries from 8,898 complete daily curves. Lower values are better. Normalisation uses a scale estimated from training observations only.}", "\\label{tab:point-performance}", "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{lrrr}", "\\toprule", "Method & Pooled MAE & Equal-system MAE & Normalised MAE \\\\", "\\midrule"]
    for name in order:
        value = metrics[name]
        lines.append(f"{name} & {value['pooled_mae_kwh']:.2f} & {value['equal_system_mae_kwh']:.2f} & {value['normalized_mae']:.4f} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}", "", "\\begin{table}[t]", "\\centering", "\\caption{Hierarchical bootstrap estimates of equal-system MAE reduction. Systems and then daily curves within systems were resampled. Positive values favour the first method.}", "\\label{tab:bootstrap}", "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{lrr}", "\\toprule", "Comparison & Reduction [kWh] & 95\\% interval \\\\", "\\midrule"])
    for name, value in intervals.items():
        lines.append(f"{name} & {value['equal_system_mae_reduction_kwh']:.2f} & [{value['ci95_low_kwh']:.2f}, {value['ci95_high_kwh']:.2f}] \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_prob_table(path: Path, metrics: dict, complete: dict) -> None:
    names = ["Chronos-2 + vendor", "Chronos-2 target-only", "Moirai 2.0 target-only", "Moirai 2.0 + vendor"]
    lines = ["\\begin{table*}[t]", "\\centering", "\\caption{Zero-shot probabilistic performance after increasing rearrangement. Grid CRPS integrates pinball loss from quantile 0.05 to 0.95.}", "\\label{tab:foundation-probabilistic}", "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{lrrrrrr}", "\\toprule", "Method & Median MAE & Pinball & Grid CRPS & WIS & 90\\% coverage & Width \\\\", "\\midrule"]
    for name in names:
        value = metrics[name]
        lines.append(f"{name} & {value['median_mae_kwh']:.2f} & {value['mean_pinball_kwh']:.2f} & {value['grid_crps_kwh']:.2f} & {value['wis_kwh']:.2f} & {value['coverage_90']:.3f} & {value['width_90_kwh']:.2f} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}", "", "\\begin{table}[t]", "\\centering", "\\caption{Sensitivity of foundation-model median MAE to incomplete seven-day contexts. The complete-context subset excludes 115 daily origins.}", "\\label{tab:context-sensitivity}", "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{lrr}", "\\toprule", "Method & Full test & Complete contexts \\\\", "\\midrule"])
    for name in names:
        lines.append(f"{name} & {metrics[name]['median_mae_kwh']:.2f} & {complete[name]['median_mae_kwh']:.2f} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--figure-root", type=Path, required=True)
    parser.add_argument("--table-root", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    started = datetime.now(timezone.utc)
    baseline = pd.read_parquet(args.prediction_root / "emsx_daily_baselines.parquet")
    lightgbm = pd.read_parquet(args.prediction_root / "emsx_daily_lightgbm.parquet")
    arima = pd.read_parquet(args.prediction_root / "emsx_daily_arima.parquet")
    point = baseline.merge(lightgbm[KEYS + ["lightgbm_prediction_kwh"]], on=KEYS, validate="one_to_one").merge(arima[KEYS + ["arima_prediction_kwh"]], on=KEYS, validate="one_to_one")
    foundation_specs = {
        "Chronos-2 target-only": "emsx_daily_chronos2_target_only.parquet",
        "Chronos-2 + vendor": "emsx_daily_chronos2_vendor_covariate.parquet",
        "Moirai 2.0 target-only": "emsx_daily_moirai2_target_only.parquet",
        "Moirai 2.0 + vendor": "emsx_daily_moirai2_vendor_covariate.parquet",
    }
    foundations = {name: foundation_frame(args.prediction_root / filename, name) for name, filename in foundation_specs.items()}
    for name, frame in foundations.items():
        point = point.merge(frame[KEYS + [name]], on=KEYS, validate="one_to_one")
    columns = {"Vendor": "vendor_prediction_kwh", "Residual LightGBM": "lightgbm_prediction_kwh", "Seasonal naive": "seasonal_naive_prediction_kwh", "Persistence": "persistence_prediction_kwh", "ARIMA": "arima_prediction_kwh", **{name: name for name in foundations}}
    metrics = {name: point_metrics(point, column) for name, column in columns.items()}
    intervals = {
        "LightGBM versus vendor": hierarchical_ci(point, "lightgbm_prediction_kwh", "vendor_prediction_kwh", args.bootstrap_draws),
        "Chronos-2 + vendor versus vendor": hierarchical_ci(point, "Chronos-2 + vendor", "vendor_prediction_kwh", args.bootstrap_draws),
        "LightGBM versus Chronos-2 + vendor": hierarchical_ci(point, "lightgbm_prediction_kwh", "Chronos-2 + vendor", args.bootstrap_draws),
    }
    prob = {name: probabilistic_metrics(frame) for name, frame in foundations.items()}
    missing = missing_context_keys(point, args.raw_root, 672)
    complete_prob = {}
    for name, frame in foundations.items():
        keep = np.asarray([(int(site), issue) not in missing for site, issue in zip(frame.site_id, frame.issue_time)])
        complete_prob[name] = probabilistic_metrics(frame.loc[keep])

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.figure_root.mkdir(parents=True, exist_ok=True)
    args.table_root.mkdir(parents=True, exist_ok=True)
    write_point_table(args.table_root / "point_performance.tex", metrics, intervals)
    write_prob_table(args.table_root / "foundation_probabilistic.tex", prob, complete_prob)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.0, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    boxes = [
        (0.3, 2.15, 2.1, .9, "Causal measurements\nending by issue time"),
        (0.3, .55, 2.1, .9, "Issued vendor curve\n96 future deliveries"),
        (3.25, 1.35, 2.15, .95, "Residual model\nshared across systems"),
        (6.2, 1.35, 1.45, .95, "Corrected\ncurve"),
        (8.25, 1.35, 1.45, .95, "Observed\ntarget curve"),
    ]
    for x, y, width, height, label in boxes:
        color = "#E6F2FA" if "Residual" in label or "Corrected" in label else "#F2F2F2"
        ax.add_patch(plt.Rectangle((x, y), width, height, facecolor=color, edgecolor="#333333", linewidth=1.2))
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=10)
    for start, end in [((2.4, 2.6), (3.25, 1.95)), ((2.4, 1.0), (3.25, 1.7)), ((5.4, 1.82), (6.2, 1.82)), ((7.65, 1.82), (8.25, 1.82))]:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.4})
    ax.text(7.93, 2.15, "evaluation", ha="center", va="bottom", fontsize=9)
    ax.text(5.8, .6, "No coordinates, PV capacity, orientation, or NWP", ha="center", fontsize=10, style="italic")
    ax.set_title("Operational forecast contract at 00:00 UTC", fontsize=14)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(args.figure_root / f"forecast_contract.{suffix}", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ranking = sorted(metrics, key=lambda name: metrics[name]["pooled_mae_kwh"])
    ax.barh(ranking[::-1], [metrics[name]["pooled_mae_kwh"] for name in ranking[::-1]], color=[COLORS[name] for name in ranking[::-1]])
    ax.set_xlabel("Pooled MAE [kWh]")
    ax.set_title("Point accuracy on the common daily-curve test set")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(args.figure_root / f"point_ranking.{suffix}", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for name in ("Vendor", "Residual LightGBM", "Chronos-2 + vendor", "Moirai 2.0 target-only", "ARIMA"):
        error = np.abs(point.actual_pv_kwh - point[columns[name]])
        values = pd.DataFrame({"step": point.delivery_step, "error": error}).groupby("step").error.mean()
        hours = np.arange(1, len(values) + 1, dtype=float) / 4
        ax.plot(hours, values.values, label=name, color=COLORS[name], linewidth=1.8)
    ax.set_xlabel("Hours after issue")
    ax.set_ylabel("MAE [kWh]")
    ax.set_title("Forecast error across the 24-hour delivery curve")
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(args.figure_root / f"delivery_step_mae.{suffix}", dpi=300)
    plt.close(fig)

    origin = point.groupby(["site_id", "issue_time"], observed=True).agg(vendor_energy=("vendor_prediction_kwh", "sum"), vendor_peak=("vendor_prediction_kwh", "max"))
    origin["vendor_ramp"] = point.groupby(["site_id", "issue_time"], observed=True).vendor_prediction_kwh.apply(lambda values: float(np.max(np.abs(np.diff(values)))))
    curve_errors = point.assign(vendor_error=np.abs(point.actual_pv_kwh - point.vendor_prediction_kwh), lightgbm_error=np.abs(point.actual_pv_kwh - point.lightgbm_prediction_kwh)).groupby(["site_id", "issue_time"], observed=True)[["vendor_error", "lightgbm_error"]].mean()
    regimes = origin.join(curve_errors)
    regime_output = {}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.1), sharey=True)
    for ax, variable, title in zip(axes, ("vendor_energy", "vendor_ramp"), ("Issued daily energy", "Issued maximum ramp")):
        regimes["bin"] = pd.qcut(regimes[variable], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        summary = regimes.groupby("bin", observed=True).agg(vendor_mae=("vendor_error", "mean"), lightgbm_mae=("lightgbm_error", "mean"))
        summary["reduction"] = summary.vendor_mae - summary.lightgbm_mae
        regime_output[variable] = summary.reset_index().to_dict("records")
        ax.bar(summary.index.astype(str), summary.reduction, color=COLORS["Residual LightGBM"])
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_title(title)
        ax.set_xlabel("Issue-time quartile")
    axes[0].set_ylabel("MAE reduction versus vendor [kWh]")
    fig.suptitle("Residual correction benefit under observable issue-time regimes")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(args.figure_root / f"conditional_gain.{suffix}", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5), sharex=True, sharey=True)
    for ax, active, title in ((axes[0], False, "All deliveries"), (axes[1], True, "Issued-active deliveries")):
        for name, frame in foundations.items():
            subset = frame.loc[frame.vendor_prediction_kwh > 0] if active else frame
            observed = [float(np.mean(subset.actual_pv_kwh.to_numpy(float) <= subset[column].to_numpy(float))) for column in QCOLS]
            ax.plot(QUANTILES, observed, marker="o", markersize=3, label=name, color=COLORS[name])
        ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Nominal quantile level")
        ax.set_title(title)
    axes[0].set_ylabel("Observed proportion below forecast")
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle("Marginal quantile reliability after rearrangement")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(args.figure_root / f"foundation_reliability.{suffix}", dpi=300)
    plt.close(fig)

    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "started_utc": started.isoformat(),
        "command": " ".join(sys.argv), "seed": SEED, "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "sample": {"delivery_rows": int(len(point)), "curves": int(point[["site_id", "issue_time"]].drop_duplicates().shape[0]), "systems": int(point.site_id.nunique())},
        "point_metrics": metrics, "hierarchical_bootstrap": intervals, "probabilistic_metrics": prob,
        "complete_context_probabilistic_metrics": complete_prob, "excluded_incomplete_context_curves": len(missing), "conditional_gain": regime_output,
    }
    (args.output_root / "emsx_daily_manuscript_analyses.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
