"""Second-round computations for the EMSx residual correction manuscript (Photovoltaics).

Run on the computing server, for example:
    python3 scripts/round2_compute.py --curves data/daily_curves \
        --original-predictions data/emsx_daily_lightgbm.parquet --out results

The script first checks the required libraries and installs missing ones with pip.
It never modifies its inputs. Outputs: round2_results.json, round2_macros.tex,
round2_predictions.parquet and a log printed to the console.

Analyses (all under the common calendar cutoffs of round 1 unless stated):
  A. Leakage size: original retrospective test split before and after the end of global fitting.
  B. Feature ablation (target step, issued profile, full) with the round-1 selected LightGBM settings.
  C. Linear references: global ridge and ridge per delivery step on the full feature set.
  D. Cross-system error feature: mean matured vendor error of the other systems at the same issue.
  E. Transfer to systems excluded from fitting (5 folds over systems, no system identity,
     scale from issued vendor forecasts only).
  F. Block bootstrap (systems, then calendar weeks) for the LightGBM improvement over the vendor.
"""
import importlib
import subprocess
import sys

REQUIRED = {"numpy": "numpy>=1.24", "pandas": "pandas>=2.0", "pyarrow": "pyarrow>=12", "lightgbm": "lightgbm>=4.0"}


def ensure_packages():
    missing = []
    for module, spec in REQUIRED.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(spec)
    if not missing:
        print("All required libraries are available.", flush=True)
        return
    print("Installing missing libraries:", ", ".join(missing), flush=True)
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    base = [sys.executable, "-m", "pip", "install", *([] if in_venv else ["--user"]), *missing]
    attempts = [base, base + ["--break-system-packages"]]
    for command in attempts:
        print("Running:", " ".join(command), flush=True)
        if subprocess.call(command) == 0:
            break
    else:
        sys.exit("Installation failed. Install manually: " + " ".join(missing))
    importlib.invalidate_caches()
    import site
    user_site = site.getusersitepackages()
    if user_site not in sys.path:
        sys.path.append(user_site)
    for module in REQUIRED:
        importlib.import_module(module)
    print("Libraries installed and importable.", flush=True)


ensure_packages()

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from emsx_daily_curve_common import ANCHOR_STEPS, TARGET_COLUMNS, FORECAST_COLUMNS, expand_split, load_wide

CUTOFFS = ["2016-01-01", "2016-04-01", "2016-07-01"]
SELECTED = dict(num_leaves=127, min_child_samples=200, learning_rate=0.03, n_estimators=952)
COMMON = dict(objective="regression_l1", colsample_bytree=.9, subsample=.85, reg_lambda=2., reg_alpha=.05,
              verbosity=-1, random_state=20261101)
BASE = ["forecast_norm", "delivery_step", "lead_sin", "lead_cos", "valid_doy_sin", "valid_doy_cos", "site_id"]
CURVE = [*[f"curve_anchor_{s:02d}_norm" for s in ANCHOR_STEPS], "curve_mean_kwh_norm", "curve_std_kwh_norm",
         "curve_max_kwh_norm", "curve_energy_kwh_norm", "curve_max_abs_ramp_kwh_norm", "curve_peak_step_norm"]
CROSS = ["cross_error_mean_norm", "cross_error_std_norm"]
BOOT_SEED, BOOT_REPS = 20261102, 2000


def log(*args):
    print(datetime.now().strftime("%H:%M:%S"), *args, flush=True)


def metrics(frame, prediction):
    y = frame.actual_pv_kwh.to_numpy(float)
    err = np.abs(y - prediction)
    sites = frame.site_id.astype(int).to_numpy()
    return {"rows": int(len(err)), "systems": int(len(np.unique(sites))),
            "mae_kwh": float(err.mean()),
            "system_averaged_mae_kwh": float(pd.Series(err).groupby(sites).mean().mean()),
            "normalized_mae": float((err / frame.scale_kwh.to_numpy(float)).mean())}


def improved_systems(frame, prediction):
    y = frame.actual_pv_kwh.to_numpy(float)
    sites = frame.site_id.astype(int).to_numpy()
    model = pd.Series(np.abs(y - prediction)).groupby(sites).mean()
    vendor = pd.Series(np.abs(y - frame.vendor_prediction_kwh.to_numpy(float))).groupby(sites).mean()
    return int((model < vendor).sum())


def calendar_frame(curves):
    w = load_wide(curves)
    a, b, c = [pd.Timestamp(x, tz="UTC") for x in CUTOFFS]
    w["split"] = np.select([w.issue_time < a, w.issue_time < b, w.issue_time < c],
                           ["train", "tuning", "calibration"], default="test")
    counts = w[w.complete_case].groupby(["site_id", "split"]).size().unstack(fill_value=0)
    ids = counts.index[(counts.train >= 30) & (counts.tuning >= 7) & (counts.calibration >= 7) & (counts.test >= 7)]
    w = w[w.site_id.isin(ids)].copy()
    for site, block in w.groupby("site_id"):
        values = block.loc[block.split == "train", TARGET_COLUMNS].to_numpy(float)
        w.loc[block.index, "scale_kwh"] = max(float(np.quantile(values, .995)), .001)
    return w


def add_cross_features(w):
    """Leave-one-out mean and spread of matured vendor error over the other systems at the same issue."""
    e = (w.recent_24h_vendor_error_kwh / w.scale_kwh).astype(float)
    ok = np.isfinite(e)
    g = pd.DataFrame({"t": w.issue_time, "e": e.where(ok, 0.), "e2": (e ** 2).where(ok, 0.), "n": ok.astype(float)})
    tot = g.groupby("t")[["e", "e2", "n"]].transform("sum")
    n = tot.n - g.n
    mean = (tot.e - g.e) / n.where(n > 0)
    var = (tot.e2 - g.e2) / n.where(n > 0) - mean ** 2
    w["cross_error_mean_norm"] = mean.fillna(0.).astype(np.float32)
    w["cross_error_std_norm"] = np.sqrt(var.clip(lower=0)).fillna(0.).astype(np.float32)
    return w


def expand(w, split, extra=()):
    out, features = expand_split(w, split)
    if extra:
        keys = w.loc[(w.split == split) & w.complete_case, ["site_id", "issue_time", *extra]]
        out = out.assign(site_int=out.site_id.astype(int)).merge(
            keys.rename(columns={"site_id": "site_int"}), on=["site_int", "issue_time"], how="left", validate="many_to_one")
        out = out.drop(columns="site_int")
        out["site_id"] = out.site_id.astype("category")
    return out, features


def fit_lgb(fit, features, threads):
    model = lgb.LGBMRegressor(**COMMON, **SELECTED, n_jobs=threads)
    cats = ["site_id"] if "site_id" in features else "auto"
    model.fit(fit[features], fit.residual_norm, categorical_feature=cats)
    return model


def predict(model, frame, features):
    return np.maximum(frame.forecast_norm.to_numpy() + model.predict(frame[features]), 0) * frame.scale_kwh.to_numpy()


def ridge(fit, test, numeric, by_step, lam=1.0):
    sites = np.union1d(fit.site_id.astype(int).unique(), test.site_id.astype(int).unique())
    values = fit[numeric].to_numpy(float)
    mu = values.mean(0)
    sd = values.std(0).clip(1e-6)
    del values

    def design(frame):
        x = (frame[numeric].to_numpy(float) - mu) / sd
        onehot = (frame.site_id.astype(int).to_numpy()[:, None] == sites[None, :]).astype(float)
        return np.hstack([np.ones((len(frame), 1)), x, onehot])

    fit_steps = fit.delivery_step.to_numpy()
    test_steps = test.delivery_step.to_numpy()
    pred = np.empty(len(test))
    for step in (range(1, 97) if by_step else [None]):
        idx = np.flatnonzero(fit_steps == step) if by_step else np.arange(len(fit))
        tidx = np.flatnonzero(test_steps == step) if by_step else np.arange(len(test))
        xtx, xty = 0., 0.
        for start in range(0, len(idx), 200000):
            part = fit.iloc[idx[start:start + 200000]]
            x = design(part)
            xtx = xtx + x.T @ x
            xty = xty + x.T @ part.residual_norm.to_numpy(float)
        reg = lam * np.eye(xtx.shape[0])
        reg[0, 0] = 0.
        beta = np.linalg.solve(xtx + reg, xty)
        for start in range(0, len(tidx), 200000):
            sel = tidx[start:start + 200000]
            pred[sel] = design(test.iloc[sel]) @ beta
    return np.maximum(test.forecast_norm.to_numpy() + pred, 0) * test.scale_kwh.to_numpy()


def block_bootstrap(frame, prediction):
    y = frame.actual_pv_kwh.to_numpy(float)
    d = pd.DataFrame({"site": frame.site_id.astype(int).to_numpy(),
                      "week": frame.issue_time.dt.tz_convert(None).dt.to_period("W").astype(str).to_numpy(),
                      "gain": np.abs(y - frame.vendor_prediction_kwh.to_numpy(float)) - np.abs(y - prediction)})
    agg = d.groupby(["site", "week"]).gain.agg(["sum", "count"]).reset_index()
    blocks = {s: (b["sum"].to_numpy(), b["count"].to_numpy()) for s, b in agg.groupby("site")}
    sites = np.array(sorted(blocks))
    rng = np.random.default_rng(BOOT_SEED)
    values = np.empty(BOOT_REPS)
    for r in range(BOOT_REPS):
        acc = []
        for s in rng.choice(sites, len(sites), replace=True):
            sm, ct = blocks[s]
            k = rng.integers(0, len(sm), len(sm))
            acc.append(sm[k].sum() / ct[k].sum())
        values[r] = np.mean(acc)
    point = float(np.mean([blocks[s][0].sum() / blocks[s][1].sum() for s in sites]))
    return {"system_averaged_gain_kwh": point, "ci95_kwh": [float(np.quantile(values, .025)), float(np.quantile(values, .975))],
            "replicates": BOOT_REPS, "seed": BOOT_SEED, "blocks": "systems resampled, then calendar weeks within each selected system"}


def leakage(curves, original_predictions):
    w = load_wide(curves)
    done = w[w.complete_case]
    fit_end = done.loc[done.split.isin(["train", "tuning"]), "issue_time"].max() + pd.Timedelta(days=1)
    p = pd.read_parquet(original_predictions)
    p["issue_time"] = pd.to_datetime(p.issue_time, utc=True)
    res = {"global_fit_end": str(fit_end)}
    for name, mask in [("before_fit_end", p.issue_time < fit_end), ("after_fit_end", p.issue_time >= fit_end),
                       ("all", pd.Series(True, index=p.index))]:
        q = p[mask]
        y = q.actual_pv_kwh.to_numpy(float)
        v = float(np.abs(y - q.vendor_prediction_kwh.to_numpy(float)).mean())
        m = float(np.abs(y - q.lightgbm_prediction_kwh.to_numpy(float)).mean())
        res[name] = {"rows": int(len(q)), "profiles": int(len(q) // 96), "systems": int(q.site_id.nunique()),
                     "vendor_mae_kwh": v, "lightgbm_mae_kwh": m, "reduction_percent": 100 * (1 - m / v)}
    return res


def macros(r):
    f2 = lambda x: f"{x:.2f}"
    f1 = lambda x: f"{x:.1f}"
    lk = r["leakage"]
    m = {
        "LeakFitEnd": lk["global_fit_end"][:10],
        "LeakPreProfiles": f"{lk['before_fit_end']['profiles']:,}",
        "LeakPreSystems": str(lk["before_fit_end"]["systems"]),
        "LeakPreVendor": f2(lk["before_fit_end"]["vendor_mae_kwh"]),
        "LeakPreLGB": f2(lk["before_fit_end"]["lightgbm_mae_kwh"]),
        "LeakPreRed": f1(lk["before_fit_end"]["reduction_percent"]),
        "LeakPostProfiles": f"{lk['after_fit_end']['profiles']:,}",
        "LeakPostSystems": str(lk["after_fit_end"]["systems"]),
        "LeakPostVendor": f2(lk["after_fit_end"]["vendor_mae_kwh"]),
        "LeakPostLGB": f2(lk["after_fit_end"]["lightgbm_mae_kwh"]),
        "LeakPostRed": f1(lk["after_fit_end"]["reduction_percent"]),
    }
    vendor = r["calendar"]["vendor"]["test"]["mae_kwh"]
    for key, name in [("target_step", "AblTarget"), ("issued_curve", "AblCurve"), ("full", "AblFull"),
                      ("full_cross", "Cross"), ("ridge_global", "RidgeGlobal"), ("ridge_step", "RidgeStep"),
                      ("transfer", "Transfer")]:
        t = r["calendar"][key]["test"]
        m[name + "MAE"] = f2(t["mae_kwh"])
        m[name + "SysMAE"] = f2(t["system_averaged_mae_kwh"])
        m[name + "NMAE"] = f"{t['normalized_mae']:.4f}"
        m[name + "Red"] = f1(100 * (1 - t["mae_kwh"] / vendor))
        m[name + "Improved"] = str(r["calendar"][key]["improved_systems"])
    m["TransferVendorSysMAE"] = f2(r["calendar"]["transfer"]["vendor_test_same_scale"]["system_averaged_mae_kwh"])
    m["CalSystems"] = str(r["calendar"]["systems"])
    m["VendorMAE"] = f2(vendor)
    m["VendorSysMAE"] = f2(r["calendar"]["vendor"]["test"]["system_averaged_mae_kwh"])
    m["BootGain"] = f2(r["bootstrap"]["system_averaged_gain_kwh"])
    m["BootLow"] = f2(r["bootstrap"]["ci95_kwh"][0])
    m["BootHigh"] = f2(r["bootstrap"]["ci95_kwh"][1])
    return "% Generated by round2_compute.py\n" + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", type=Path, required=True)
    ap.add_argument("--original-predictions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    r = {"created_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
         "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                      "lightgbm": lgb.__version__},
         "cutoffs": CUTOFFS, "lightgbm_settings": {**COMMON, **SELECTED}}

    log("A. leakage size in the original split")
    r["leakage"] = leakage(a.curves, a.original_predictions)
    log(json.dumps(r["leakage"]))

    w = add_cross_features(calendar_frame(a.curves))
    train, full = expand(w, "train", CROSS)
    tuning, _ = expand(w, "tuning", CROSS)
    test, _ = expand(w, "test", CROSS)
    fit = pd.concat([train, tuning], ignore_index=True)
    fit["site_id"] = fit.site_id.astype(int).astype("category")
    test["site_id"] = pd.Categorical(test.site_id.astype(int), categories=fit.site_id.cat.categories)
    cal = r["calendar"] = {"vendor": {"test": metrics(test, test.vendor_prediction_kwh.to_numpy(float))},
                           "systems": int(test.site_id.nunique())}
    sets = {"target_step": BASE, "issued_curve": BASE + CURVE, "full": full, "full_cross": full + CROSS}
    keep = test[["site_id", "issue_time", "delivery_step", "actual_pv_kwh", "vendor_prediction_kwh", "scale_kwh"]].copy()
    keep["site_id"] = keep.site_id.astype(int)
    for name, feats in sets.items():
        log("B/D. LightGBM", name)
        model = fit_lgb(fit, feats, a.threads)
        p = predict(model, test, feats)
        cal[name] = {"features": feats, "test": metrics(test, p), "improved_systems": improved_systems(test, p)}
        keep[f"{name}_kwh"] = p
        log(name, json.dumps(cal[name]["test"]))
        if name == "full":
            r["bootstrap"] = block_bootstrap(test, p)
            log("F. bootstrap", json.dumps(r["bootstrap"]))

    numeric = [x for x in full if x != "site_id"]
    for name, by_step in [("ridge_global", False), ("ridge_step", True)]:
        log("C. linear reference", name)
        p = ridge(fit, test, numeric, by_step)
        cal[name] = {"test": metrics(test, p), "improved_systems": improved_systems(test, p), "lambda": 1.0}
        keep[f"{name}_kwh"] = p
        log(name, json.dumps(cal[name]["test"]))

    log("E. transfer to systems excluded from fitting")
    wt = w.copy()
    for site, block in wt.groupby("site_id"):
        ref = block.loc[block.split == "calibration", FORECAST_COLUMNS].to_numpy(float)
        wt.loc[block.index, "scale_kwh"] = max(float(np.nanquantile(ref, .995)), .001)
    tr_t, _ = expand(wt, "train")
    tu_t, _ = expand(wt, "tuning")
    te_t, _ = expand(wt, "test")
    fit_t = pd.concat([tr_t, tu_t], ignore_index=True)
    feats_t = [x for x in full if x != "site_id"]
    sites = np.array(sorted(w.site_id.unique()))
    fold = {s: i % 5 for i, s in enumerate(sites)}
    pred_t = np.full(len(te_t), np.nan)
    fit_sites = fit_t.site_id.astype(int).to_numpy()
    test_sites = te_t.site_id.astype(int).to_numpy()
    for k in range(5):
        held = [int(s) for s in sites if fold[s] == k]
        m_fit = ~np.isin(fit_sites, held)
        m_te = np.isin(test_sites, held)
        model = fit_lgb(fit_t[m_fit], feats_t, a.threads)
        pred_t[m_te] = predict(model, te_t[m_te], feats_t)
        log("fold", k, "held systems", held)
    assert np.isfinite(pred_t).all()
    cal["transfer"] = {"test": metrics(te_t, pred_t), "improved_systems": improved_systems(te_t, pred_t),
                       "vendor_test_same_scale": metrics(te_t, te_t.vendor_prediction_kwh.to_numpy(float)),
                       "design": "5 folds over sorted system ids; held systems absent from fitting; no system identity; "
                                 "scale = 99.5th percentile of issued vendor forecasts in the calibration quarter, no measurements."}
    keep["transfer_kwh"] = pred_t
    log("transfer", json.dumps(cal["transfer"]["test"]))

    keep.to_parquet(a.out / "round2_predictions.parquet", index=False)
    r["runtime_minutes"] = (time.time() - started) / 60
    (a.out / "round2_results.json").write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
    (a.out / "round2_macros.tex").write_text(macros(r), encoding="utf-8")
    log("done; check: full model test MAE should be close to 83.26 kWh from round 1:",
        f"{cal['full']['test']['mae_kwh']:.2f}")


if __name__ == "__main__":
    main()
