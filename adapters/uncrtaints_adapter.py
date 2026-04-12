"""
UnCRtainTS adapter: AllClearDataset -> dict consumed by model.set_input().

UnCRtainTS's training loop (UnCRtainTS/model/train_reconstruct.py) builds
a per-batch dictionary of the form:

    inputs = {'A': x, 'B': y, 'dates': dates, 'masks': in_m}

with shapes (for the S2-only / no-SAR configuration used here):

    A       (B, T, 13, H, W)   cloudy Sentinel-2 input time series
    B       (B, 1, 13, H, W)   cloud-free Sentinel-2 target (s2p mode)
    masks   (B, T, H, W)       merged cloud+shadow mask per timestep
    dates   (B, T)             day offsets from first frame

This adapter returns per-sample dicts with batch dim removed; a custom
collate function stacks them along a new batch dim.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .common import build_allclear_dataset

NUM_S2_BANDS = 13


class UnCRtainTSAdapter(Dataset):
    def __init__(self, json_path: str | Path, *, tx: int = 3):
        self.base = build_allclear_dataset(
            json_path,
            main_sensor="s2_toa",
            aux_sensors=[],
            aux_data=["cld_shdw"],
            tx=tx,
            target_mode="s2p",
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]

        # input_images: (C, T, H, W) -> (T, 13, H, W)
        A = item["input_images"][:NUM_S2_BANDS].permute(1, 0, 2, 3).contiguous()

        # target: (C, 1, H, W) -> (1, 13, H, W)
        B = item["target"][:NUM_S2_BANDS].permute(1, 0, 2, 3).contiguous()

        # cloud+shadow masks: (2, T, H, W) -> (T, H, W), clipped to {0,1}
        cld_shdw = item.get("input_cld_shdw")
        if cld_shdw is None:
            raise RuntimeError("input_cld_shdw missing; UnCRtainTS needs cloud masks.")
        masks = torch.clip(cld_shdw.sum(dim=0), 0, 1)  # (T, H, W)

        dates = item["time_differences"].float()       # (T,)

        return {
            "A": A,
            "B": B,
            "masks": masks,
            "dates": dates,
            "data_id": item.get("data_id", f"idx_{idx}"),
        }


def uncrtaints_collate(batch):
    """Stack per-sample dicts into batched tensors with a leading batch dim."""
    out = {
        "A":     torch.stack([b["A"] for b in batch]),
        "B":     torch.stack([b["B"] for b in batch]),
        "masks": torch.stack([b["masks"] for b in batch]),
        "dates": torch.stack([b["dates"] for b in batch]),
        "data_ids": [b["data_id"] for b in batch],
    }
    return out
