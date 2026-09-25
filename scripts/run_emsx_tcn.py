"""Train a controlled TCN correction model for 24-hour EMSx PV forecasts."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


from emsx_paths import WORK_ROOT as ROOT
PANEL_ROOT = ROOT / "data" / "processed" / "emsx" / "forecast_panels"
RESULTS = ROOT / "results"


class TCN(nn.Module):
    def __init__(self, n_sites: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=2, dilation=1), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=4, dilation=2), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=8, dilation=4), nn.ReLU(),
        )
        self.embedding = nn.Embedding(n_sites, 8)
        self.head = nn.Sequential(nn.Linear(32 + 5 + 8, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, history: torch.Tensor, static: torch.Tensor, site: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(history)[:, :, -1]
        return self.head(torch.cat([encoded, static, self.embedding(site)], dim=1)).squeeze(1)


def assemble() -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], dict[int, float]]:
    frames = []
    scales: dict[int, float] = {}
    for path in sorted(PANEL_ROOT.glob("site=*/panel.parquet")):
        frame = pd.read_parquet(path, filters=[("lead_steps", "=", 96)])
        frame["issue_time"] = pd.to_datetime(frame["issue_time"], utc=True)
        frame = frame.drop_duplicates("issue_time").sort_values("issue_time").copy()
        rank = frame["issue_time"].rank(pct=True, method="first")
        frame["split"] = np.where(rank <= .70, "train", np.where(rank <= .80, "validation", "test"))
        scale = float(frame.loc[frame["split"] == "train", "actual_pv_kwh"].quantile(.995))
        scale = max(scale, 1e-3)
        site = int(frame["site_id"].iloc[0]); scales[site] = scale
        observed = frame.set_index("issue_time")["issue_actual_pv_kwh"]
        for lag in range(16):
            frame[f"history_{lag}"] = observed.reindex(frame["issue_time"] - pd.Timedelta(minutes=15 * lag)).to_numpy() / scale
        frame["forecast_norm"] = frame["forecast_pv_kwh"] / scale
        issue = frame["issue_time"]
        frame["hour_sin"] = np.sin(2 * np.pi * issue.dt.hour / 24)
        frame["hour_cos"] = np.cos(2 * np.pi * issue.dt.hour / 24)
        frame["doy_sin"] = np.sin(2 * np.pi * issue.dt.dayofyear / 366)
        frame["doy_cos"] = np.cos(2 * np.pi * issue.dt.dayofyear / 366)
        frame["target_norm"] = frame["actual_pv_kwh"] / scale
        frame["site_index"] = site - 1
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    history_cols = [f"history_{lag}" for lag in range(15, -1, -1)]
    static_cols = ["forecast_norm", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    data = data.dropna(subset=history_cols + static_cols + ["target_norm"])
    result = {}
    for split, frame in data.groupby("split", observed=True):
        result[split] = (
            frame[history_cols].to_numpy(np.float32)[:, None, :],
            frame[static_cols].to_numpy(np.float32),
            frame["site_index"].to_numpy(np.int64),
            frame["target_norm"].to_numpy(np.float32),
        )
    result["test_metadata"] = data.loc[data["split"] == "test", ["issue_time", "actual_pv_kwh", "forecast_pv_kwh", "site_id"]].to_numpy()
    return result, scales


def loader(values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(*(torch.from_numpy(item) for item in values)), batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    partitions, scales = assemble()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TCN(70).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    train_loader = loader(partitions["train"], args.batch_size, True)
    valid_loader = loader(partitions["validation"], args.batch_size, False)
    best_state = None; best_loss = float("inf"); stale = 0
    for epoch in range(args.epochs):
        model.train()
        for history, static, site, target in train_loader:
            history, static, site, target = (item.to(device, non_blocking=True) for item in (history, static, site, target))
            optimizer.zero_grad(); loss = torch.nn.functional.l1_loss(model(history, static, site), target); loss.backward(); optimizer.step()
        model.eval(); values = []
        with torch.no_grad():
            for history, static, site, target in valid_loader:
                values.append(torch.nn.functional.l1_loss(model(history.to(device), static.to(device), site.to(device)), target.to(device), reduction="sum").item())
        loss = sum(values) / len(partitions["validation"][3])
        print({"epoch": epoch + 1, "validation_mae_norm": loss}, flush=True)
        if loss < best_loss:
            best_loss = loss; best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}; stale = 0
        else:
            stale += 1
            if stale >= 3: break
    model.load_state_dict(best_state); model.eval()
    preds = []
    with torch.no_grad():
        for history, static, site, _ in loader(partitions["test"], args.batch_size, False):
            preds.append(model(history.to(device), static.to(device), site.to(device)).cpu().numpy())
    meta = partitions["test_metadata"]
    site_scale = np.array([scales[int(site)] for site in meta[:, 3]], dtype=float)
    prediction = np.maximum(np.concatenate(preds), 0) * site_scale
    target = meta[:, 1].astype(float); vendor = meta[:, 2].astype(float)
    result = {
        "dataset": "EMSx", "lead_hours": 24.0, "device": str(device),
        "feature_contract": "16 preceding 15-minute observed PV values, vendor forecast, UTC calendar features, and site embedding",
        "splitting": "Per-site chronological 70/10/20 train/validation/test split.",
        "fit": {"best_validation_mae_norm": best_loss, "train_rows": int(len(partitions["train"][3])), "validation_rows": int(len(partitions["validation"][3])), "test_rows": int(len(target))},
        "test": {"mae_kwh": float(np.abs(target-prediction).mean()), "vendor_mae_kwh": float(np.abs(target-vendor).mean()), "mae_improvement_pct": float(100*(1-np.abs(target-prediction).mean()/np.abs(target-vendor).mean()))},
    }
    RESULTS.mkdir(exist_ok=True); (RESULTS / "emsx_tcn_lead_96.json").write_text(json.dumps(result, indent=2)+"\n")
    predictions = pd.DataFrame({
        "issue_time": pd.to_datetime(meta[:, 0], utc=True),
        "site_id": meta[:, 3].astype(int),
        "actual_pv_kwh": target,
        "forecast_pv_kwh": vendor,
        "scale_kwh": site_scale,
        "tcn_prediction_kwh": prediction,
    })
    predictions.to_parquet(RESULTS / "emsx_tcn_lead_96.parquet", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
