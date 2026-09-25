"""Train an enhanced residual GRU baseline for EMSx."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from run_emsx_tuned_qgbm import prepare_enhanced
from run_emsx_qgbm import score


from emsx_paths import WORK_ROOT as ROOT
RESULTS = ROOT / "results"


class GRUCorrector(nn.Module):
    def __init__(self, static_size: int) -> None:
        super().__init__(); self.gru = nn.GRU(1, 64, batch_first=True); self.embedding = nn.Embedding(70, 8); self.head = nn.Sequential(nn.Linear(64+8+static_size, 96), nn.ReLU(), nn.Dropout(.1), nn.Linear(96, 1))
    def forward(self, history, static, site):
        sequence, _ = self.gru(history); return self.head(torch.cat([sequence[:, -1], static, self.embedding(site)], 1)).squeeze(1)


def make_loader(values, shuffle):
    tensors = tuple(torch.from_numpy(np.array(value, copy=True)) for value in values); return DataLoader(TensorDataset(*tensors), batch_size=4096, shuffle=shuffle, num_workers=2, pin_memory=True)


def main() -> None:
    random.seed(20260828); np.random.seed(20260828); torch.manual_seed(20260828)
    data, features = prepare_enhanced(); history_cols = [f"history_lag_{lag}_norm" for lag in range(15, 0, -1)]+["issue_actual_norm"]; static_cols = [col for col in features if col not in {"site_id", *history_cols}]
    sets = {}; metadata = {}
    for name in ("train", "validation", "test"):
        frame = data.loc[data["split"] == name].dropna(subset=features+["residual_norm"]).copy(); sets[name] = (frame[history_cols].to_numpy(np.float32)[:, :, None], frame[static_cols].to_numpy(np.float32), frame.site_id.cat.codes.to_numpy(np.int64), frame.residual_norm.to_numpy(np.float32)); metadata[name] = frame
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model = GRUCorrector(len(static_cols)).to(device); optimiser = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    best, best_loss, stale = None, float("inf"), 0
    for epoch in range(20):
        model.train()
        for history, static, site, target in make_loader(sets["train"], True):
            history, static, site, target = (value.to(device, non_blocking=True) for value in (history, static, site, target)); optimiser.zero_grad(); loss = torch.nn.functional.l1_loss(model(history, static, site), target); loss.backward(); optimiser.step()
        model.eval(); total = 0.
        with torch.no_grad():
            for history, static, site, target in make_loader(sets["validation"], False): total += torch.nn.functional.l1_loss(model(history.to(device), static.to(device), site.to(device)), target.to(device), reduction="sum").item()
        loss = total/len(sets["validation"][3]); print({"epoch": epoch+1, "validation_mae_norm": loss}, flush=True)
        if loss < best_loss: best, best_loss, stale = {key: value.cpu().clone() for key, value in model.state_dict().items()}, loss, 0
        else: stale += 1
        if stale == 4: break
    model.load_state_dict(best); model.eval(); values=[]
    with torch.no_grad():
        for history, static, site, _ in make_loader(sets["test"], False): values.append(model(history.to(device), static.to(device), site.to(device)).cpu().numpy())
    test = metadata["test"]; prediction = np.clip(test.forecast_norm.to_numpy()+np.concatenate(values), 0, None)*test.scale_kwh.to_numpy()
    result = {"dataset": "EMSx", "lead_hours": 24., "device": str(device), "method": "GRU residual corrector", "feature_contract": "Identical enhanced forecast-curve information set used by tuned QGBM.", "fit": {"best_validation_mae_normalized": best_loss, "epochs": epoch+1}, "test": score(test, prediction)}
    output = test[["issue_time", "site_id", "actual_pv_kwh", "forecast_pv_kwh", "scale_kwh"]].reset_index(drop=True)
    output["gru_prediction_kwh"] = prediction
    RESULTS.mkdir(exist_ok=True); output.to_parquet(RESULTS/"emsx_gru_lead_96.parquet", index=False); (RESULTS/"emsx_gru_lead_96.json").write_text(json.dumps(result, indent=2)+"\n"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
