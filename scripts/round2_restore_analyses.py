"""Analyses on the common calendar test that restore delivery step, system, season, regime,
reliability, and zero-shot foundation model results. Writes figures, LaTeX tables, macros, JSON."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R1 = Path(r"..\round1\results"); CODE = Path(r"..\..\pv-emsx-residual-correction-code\predictions")
OUT = Path("results"); FIG = Path(r"manuscript\figures"); TAB = Path(r"manuscript\tables")
Q = np.array([.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95]); QC = [f"q{round(100*x):02d}_kwh" for x in Q]
KEY = ["site_id", "issue_time", "delivery_step"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
COL = {"Vendor": "#7f7f7f", "Residual LightGBM": "#1f5fa8", "Ridge per step": "#e08a1e", "Residual BiLSTM": "#2a9d5b", "Residual Transformer": "#b0417a"}

def load(p):
    d = pd.read_parquet(p); d["issue_time"] = pd.to_datetime(d.issue_time, utc=True); d["site_id"] = d.site_id.astype(int); return d

def scores(y, q):
    q = np.sort(q, axis=1); e = y[:, None] - q; L = np.maximum(Q * e, (Q - 1) * e)
    terms = sum(a / 2 * ((q[:, hi] - q[:, lo]) + 2 / a * np.maximum(q[:, lo] - y, 0) + 2 / a * np.maximum(y - q[:, hi], 0))
                for a, lo, hi in ((.2, 1, 9), (.4, 2, 8), (.6, 3, 7), (.8, 4, 6)))
    wis = (.5 * np.abs(y - q[:, 5]) + terms) / 4.5
    return {"median_mae_kwh": float(np.abs(y - q[:, 5]).mean()), "mean_pinball_kwh": float(L.mean()), "wis_kwh": float(wis.mean()),
            "grid_crps_kwh": float(2 * np.trapezoid(L.mean(0), Q)) if hasattr(np, "trapezoid") else float(2 * np.trapz(L.mean(0), Q)),
            "coverage_90": float(((y >= q[:, 0]) & (y <= q[:, -1])).mean()), "width_90_kwh": float((q[:, -1] - q[:, 0]).mean())}

res = {}
qt = load(R1 / "calendar_quantile_test.parquet")
pt = load(R1 / "calendar_lightgbm_test.parquet")
r2 = load(OUT / "round2_predictions.parquet")
d = pt.merge(r2[KEY + ["ridge_step_kwh", "transfer_kwh"]], on=KEY, validate="one_to_one")
assert len(d) == len(pt)
nn = {}
sites_pt = pt.site_id.to_numpy().reshape(-1, 96)[:, 0]
for kind, label in (("bilstm", "Residual BiLSTM"), ("transformer", "Residual Transformer")):
    preds = []
    for s in (20261101, 20261102, 20261103):
        z = np.load(R1 / f"calendar_{kind}_{s}_test.npz"); assert (z["site"] == sites_pt).all(); preds.append(z["prediction"])
    nn[label] = np.stack(preds)  # seeds x profiles x 96
y = pt.actual_pv_kwh.to_numpy(float).reshape(-1, 96)
v = pt.vendor_prediction_kwh.to_numpy(float).reshape(-1, 96)
m = pt.prediction_kwh.to_numpy(float).reshape(-1, 96)
rs = d.ridge_step_kwh.to_numpy(float).reshape(-1, 96)

# 1. delivery step
step = {"Vendor": np.abs(y - v).mean(0), "Residual LightGBM": np.abs(y - m).mean(0), "Ridge per step": np.abs(y - rs).mean(0)}
for k, p in nn.items(): step[k] = np.abs(y[None] - p).mean(1).mean(0)
hours = np.arange(1, 97) / 4
fig, ax = plt.subplots(figsize=(6.5, 3.2))
for k, s in step.items(): ax.plot(hours, s, label=k, color=COL[k], lw=1.6 if k in ("Vendor", "Residual LightGBM") else 1.1)
ax.set_xlabel("Hours after issue (UTC)"); ax.set_ylabel("MAE (kWh)"); ax.set_xlim(0, 24); ax.legend(frameon=False, ncol=2)
fig.tight_layout(); fig.savefig(FIG / "calendar_step_mae.pdf"); plt.close(fig)
pk = int(np.argmax(step["Vendor"])); res["step"] = {"peak_hour": float(hours[pk]), "vendor_peak": float(step["Vendor"][pk]), "lgb_at_peak": float(step["Residual LightGBM"][pk]),
    "steps_lgb_below_vendor_active": int(((step["Residual LightGBM"] < step["Vendor"]) & (step["Vendor"] > 5)).sum()), "active_steps": int((step["Vendor"] > 5).sum())}

# 2. per system
sid = pt.site_id.to_numpy()
ev = pd.Series(np.abs(pt.actual_pv_kwh - pt.vendor_prediction_kwh).to_numpy()).groupby(sid).mean()
em = pd.Series(np.abs(pt.actual_pv_kwh - pt.prediction_kwh).to_numpy()).groupby(sid).mean()
et = pd.Series(np.abs(d.actual_pv_kwh - d.transfer_kwh).to_numpy()).groupby(sid).mean()
gain = (ev - em).sort_values(ascending=False); gt = (ev - et)[gain.index]
fig, ax = plt.subplots(figsize=(6.5, 3.0))
x = np.arange(len(gain)); ax.bar(x - .2, gain.values, .4, color=COL["Residual LightGBM"], label="Residual LightGBM")
ax.bar(x + .2, np.maximum(gt.values, 1e-3), .4, color="#9bbbe0", label="Transfer to excluded systems")
ax.set_yscale("log"); ax.set_xlabel("Systems ordered by LightGBM reduction"); ax.set_ylabel("MAE reduction (kWh)"); ax.set_xticks([]); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / "calendar_system_gain.pdf"); plt.close(fig)
rel = 100 * (ev - em) / ev
res["systems"] = {"min_gain": float(gain.min()), "max_gain": float(gain.max()), "median_gain": float(gain.median()),
                  "min_rel": float(rel.min()), "median_rel": float(rel.median()), "max_rel": float(rel.max()), "transfer_min_gain": float(gt.min())}

# 3. seasons
mon = pt.issue_time.dt.month.to_numpy().reshape(-1, 96)[:, 0]
season = np.select([np.isin(mon, [12, 1, 2]), np.isin(mon, [3, 4, 5]), np.isin(mon, [6, 7, 8])], ["DJF", "MAM", "JJA"], "SON")
rows = []
for s in ["DJF", "MAM", "JJA", "SON"]:
    k = season == s; a = np.abs(y[k] - v[k]).mean(); b = np.abs(y[k] - m[k]).mean()
    rows.append({"season": s, "vendor": a, "lgb": b, "red": a - b, "rel": 100 * (a - b) / a, "profiles": int(k.sum()), "systems": int(len(np.unique(sites_pt[k])))})
res["seasons"] = rows
with open(TAB / "calendar_seasons.tex", "w", encoding="utf8") as f:
    f.write("\\begin{table}[t]\n\\centering\n\\color{blue}\n\\caption{Correction by season of the UTC issue month on the common calendar test. MAE and absolute reductions are in kWh. Seasons combine different systems and years, so the summaries are descriptive.}\n\\label{tab:calendar-seasons}\n\\begin{tabular}{lrrrrr}\n\\toprule\nSeason & Vendor MAE & LightGBM MAE & Reduction & Relative reduction & Profiles \\\\\n\\midrule\n")
    for r in rows: f.write(f"{r['season']} & {r['vendor']:.2f} & {r['lgb']:.2f} & {r['red']:.2f} & {r['rel']:.1f}\\% & {r['profiles']:,} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# 4. regimes
energy = v.clip(min=0).sum(1); ramp = np.abs(np.diff(v, axis=1)).max(1)
gp = np.abs(y - v).mean(1) - np.abs(y - m).mean(1)
reg = {}
fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True)
for ax, (name, val) in zip(axs, (("Issued daily energy", energy), ("Maximum issued ramp", ramp))):
    qtl = pd.qcut(val, 4, labels=["Q1", "Q2", "Q3", "Q4"]); g = pd.Series(gp).groupby(qtl, observed=True).mean()
    rv = pd.Series(np.abs(y - v).mean(1)).groupby(qtl, observed=True).mean()
    reg[name] = {"gain": g.round(3).tolist(), "relative": (100 * g / rv).round(1).tolist()}
    ax.bar(g.index.astype(str), g.values, color=COL["Residual LightGBM"]); ax.set_title(name); ax.set_xlabel("Quartile observed at issue")
axs[0].set_ylabel("MAE reduction per profile (kWh)"); fig.tight_layout(); fig.savefig(FIG / "calendar_conditional_gain.pdf"); plt.close(fig)
res["regimes"] = reg

# 5. reliability and coverage by step
yq = qt.actual_pv_kwh.to_numpy(float); act = qt.vendor_prediction_kwh.to_numpy(float) > 0
raw = np.sort(qt[QC].to_numpy(float), 1); cal = np.sort(qt[["cal_" + c for c in QC]].to_numpy(float), 1)
fig, axs = plt.subplots(1, 2, figsize=(6.5, 3.0), sharey=True)
for ax, (title, mask) in zip(axs, (("All deliveries", np.ones(len(yq), bool)), ("Deliveries active in the issued forecast", act))):
    ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1)
    for lab, q, c in (("Raw", raw, "#9bbbe0"), ("Rolling calibration", cal, COL["Residual LightGBM"])):
        ax.plot(Q, (yq[mask, None] <= q[mask]).mean(0), "o-", ms=3, color=c, label=lab)
    ax.set_title(title, fontsize=9); ax.set_xlabel("Nominal quantile level")
axs[0].set_ylabel("Observed frequency"); axs[0].legend(frameon=False); fig.tight_layout(); fig.savefig(FIG / "calendar_reliability.pdf"); plt.close(fig)
cov = lambda q, mk: float(((yq[mk] >= q[mk, 0]) & (yq[mk] <= q[mk, -1])).mean())
res["coverage"] = {"raw_all": cov(raw, slice(None)), "cal_all": cov(cal, slice(None)), "raw_active": cov(raw, act), "cal_active": cov(cal, act)}
st = qt.delivery_step.to_numpy()
cs = pd.DataFrame({"s": st, "raw": (yq >= raw[:, 0]) & (yq <= raw[:, -1]), "cal": (yq >= cal[:, 0]) & (yq <= cal[:, -1]), "act": act})
ca = cs[cs.act].groupby("s")[["raw", "cal"]].mean()
fig, ax = plt.subplots(figsize=(6.5, 2.8))
ax.plot(ca.index / 4, ca.raw, color="#9bbbe0", label="Raw"); ax.plot(ca.index / 4, ca.cal, color=COL["Residual LightGBM"], label="Rolling calibration")
ax.axhline(.9, ls="--", color="#888888", lw=1); ax.set_xlabel("Hours after issue (UTC)"); ax.set_ylabel("Central 90% coverage, active deliveries"); ax.set_xlim(0, 24); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / "calendar_coverage_step.pdf"); plt.close(fig)
cnt = cs[cs.act].groupby("s").size(); big = ca[cnt >= 1000]
res["coverage"]["cal_active_step_min"] = float(big.cal.min()); res["coverage"]["cal_active_step_max"] = float(big.cal.max())
res["coverage"]["cal_active_steps_below_085"] = int((big.cal < .85).sum()); res["coverage"]["active_steps_counted"] = int(len(big))

# 6. zero-shot foundation models on shared deliveries
fm = {"Chronos-2 with vendor covariates": "emsx_daily_chronos2_vendor_covariate", "Chronos-2 using targets only": "emsx_daily_chronos2_target_only",
      "Moirai 2.0 using targets only": "emsx_daily_moirai2_target_only", "Moirai 2.0 with vendor covariates": "emsx_daily_moirai2_vendor_covariate"}
base = qt[KEY + ["actual_pv_kwh", "vendor_prediction_kwh"] + QC + ["cal_" + c for c in QC]].merge(pt[KEY + ["prediction_kwh"]], on=KEY)
keys = None
fdat = {}
for lab, f in fm.items():
    z = load(CODE / f"{f}.parquet")[KEY + QC].rename(columns={c: "fm_" + c for c in QC}); fdat[lab] = z
    k = z[KEY]; keys = k if keys is None else keys.merge(k, on=KEY)
sh = base.merge(keys, on=KEY)
ys = sh.actual_pv_kwh.to_numpy(float)
rows = [{"method": "Vendor", "median_mae_kwh": float(np.abs(ys - sh.vendor_prediction_kwh).mean())},
        {"method": "Residual LightGBM, point", "median_mae_kwh": float(np.abs(ys - sh.prediction_kwh).mean())},
        {"method": "Residual LightGBM quantiles, raw", **scores(ys, sh[QC].to_numpy(float))},
        {"method": "Residual LightGBM quantiles, calibrated", **scores(ys, sh[["cal_" + c for c in QC]].to_numpy(float))}]
for lab, z in fdat.items():
    w = sh[KEY].merge(z, on=KEY, how="left"); rows.append({"method": lab, **scores(ys, w[["fm_" + c for c in QC]].to_numpy(float))})
res["foundation_shared"] = {"rows": int(len(sh)), "profiles": int(len(sh) // 96), "systems": int(sh.site_id.nunique()),
                            "first_issue": str(sh.issue_time.min())[:10], "last_issue": str(sh.issue_time.max())[:10], "methods": rows}
f2 = lambda x: "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"
with open(TAB / "calendar_foundation.tex", "w", encoding="utf8") as f:
    f.write("\\begin{table}[t]\n\\centering\n\\color{blue}\n\\caption{Pretrained foundation models without task training and residual LightGBM fitted before the common cutoff, evaluated on the "
            f"{len(sh):,} deliveries shared by the common calendar test and the foundation model predictions ({len(sh)//96:,} profiles, {sh.site_id.nunique()} systems). "
            "MAE, WIS, grid CRPS, and width are in kWh; coverage is dimensionless. Quantiles are sorted before scoring.}\n\\label{tab:calendar-foundation}\n"
            "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{lrrrrr}\n\\toprule\nMethod & Median MAE & WIS & Grid CRPS & 90\\% coverage & Width \\\\\n\\midrule\n")
    for r in rows:
        g = lambda k: r.get(k)
        cv = "--" if g("coverage_90") is None else f"{g('coverage_90'):.3f}"
        f.write(f"{r['method']} & {f2(g('median_mae_kwh'))} & {f2(g('wis_kwh'))} & {f2(g('grid_crps_kwh'))} & {cv} & {f2(g('width_90_kwh'))} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n")

(OUT / "round2_restore.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
print(json.dumps(res, indent=2, default=float))
