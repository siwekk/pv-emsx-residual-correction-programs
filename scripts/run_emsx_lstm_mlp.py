"""Train matched MLP and LSTM EMSx correction baselines on the GPU."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from run_emsx_tcn import assemble


from emsx_paths import WORK_ROOT as ROOT
RESULTS = ROOT / "results"


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.embed = nn.Embedding(70, 8); self.net = nn.Sequential(nn.Linear(16 + 5 + 8, 96), nn.ReLU(), nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, 1))
    def forward(self, history: torch.Tensor, static: torch.Tensor, site: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([history.squeeze(1), static, self.embed(site)], dim=1)).squeeze(1)


class LSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.encoder = nn.LSTM(1, 32, batch_first=True); self.embed = nn.Embedding(70, 8); self.head = nn.Sequential(nn.Linear(32 + 5 + 8, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, history: torch.Tensor, static: torch.Tensor, site: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(history.transpose(1, 2)); return self.head(torch.cat([encoded[:, -1], static, self.embed(site)], dim=1)).squeeze(1)


def loader(values: tuple[np.ndarray, ...], shuffle: bool) -> DataLoader:
    tensors = tuple(torch.from_numpy(np.array(item, copy=True)) for item in values)
    return DataLoader(TensorDataset(*tensors), batch_size=4096, shuffle=shuffle, num_workers=2, pin_memory=True)


def train_and_predict(name: str, model: nn.Module, partitions: dict, device: torch.device) -> tuple[np.ndarray, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4); model.to(device)
    best, best_loss, stale = None, float("inf"), 0
    for _ in range(15):
        model.train()
        for history, static, site, target in loader(partitions["train"], True):
            history, static, site, target = (item.to(device, non_blocking=True) for item in (history, static, site, target)); optimizer.zero_grad(); loss = torch.nn.functional.l1_loss(model(history, static, site), target); loss.backward(); optimizer.step()
        model.eval(); total = 0.
        with torch.no_grad():
            for history, static, site, target in loader(partitions["validation"], False):
                total += torch.nn.functional.l1_loss(model(history.to(device), static.to(device), site.to(device)), target.to(device), reduction="sum").item()
        loss = total / len(partitions["validation"][3])
        if loss < best_loss: best, best_loss, stale = {key: value.cpu().clone() for key, value in model.state_dict().items()}, loss, 0
        else: stale += 1
        if stale >= 3: break
    model.load_state_dict(best); model.eval(); values = []
    with torch.no_grad():
        for history, static, site, _ in loader(partitions["test"], False): values.append(model(history.to(device), static.to(device), site.to(device)).cpu().numpy())
    return np.maximum(np.concatenate(values), 0), best_loss


def main() -> None:
    random.seed(20260827); np.random.seed(20260827); torch.manual_seed(20260827)
    partitions, scales = assemble(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp, mlp_valid = train_and_predict("mlp", MLP(), partitions, device)
    lstm, lstm_valid = train_and_predict("lstm", LSTM(), partitions, device)
    meta = partitions["test_metadata"]; scale = np.array([scales[int(site)] for site in meta[:, 3]], dtype=float)
    target = meta[:, 1].astype(float); vendor = meta[:, 2].astype(float)
    output = pd.DataFrame({"issue_time": pd.to_datetime(meta[:, 0], utc=True), "site_id": meta[:, 3].astype(int), "actual_pv_kwh": target, "forecast_pv_kwh": vendor, "scale_kwh": scale, "mlp_prediction_kwh": mlp*scale, "lstm_prediction_kwh": lstm*scale})
    result = {"dataset": "EMSx", "lead_hours": 24., "device": str(device), "feature_contract": "Vendor forecast, 16 preceding 15-minute PV values, UTC calendar features, and site embedding. Same chronological split and shared test observations as QGBM and TCN.", "test_rows": int(len(output)), "mae_kwh": {"vendor": float(np.abs(target-vendor).mean()), "mlp": float(np.abs(target-output.mlp_prediction_kwh).mean()), "lstm": float(np.abs(target-output.lstm_prediction_kwh).mean())}, "validation_mae_normalized": {"mlp": mlp_valid, "lstm": lstm_valid}}
    RESULTS.mkdir(exist_ok=True); output.to_parquet(RESULTS / "emsx_lstm_mlp_lead_96.parquet", index=False); (RESULTS / "emsx_lstm_mlp_lead_96.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
