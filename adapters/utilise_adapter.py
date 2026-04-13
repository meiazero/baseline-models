"""
U-TILISE adapter: AllClearDataset -> dict consumed by U-TILISE trainer/loss.

U-TILISE is a seq-to-seq model. Its SEN12MSCRTS dataset class
(U-TILISE/lib/datasets/SEN12MSCRTSDataset.py) produces batches of the form:

    {
        'x':               (T, C, H, W)   input sequence (cloud pixels masked)
        'y':               (T, C, H, W)   target sequence
        'masks':           (T, 1, H, W)   1 = masked input pixel, 0 = observed
        'masks_valid_obs': (T,)           1 = valid frame, 0 = zero-padded
        'position_days':   (T,)           day offset per frame
        'cloud_mask':      (T, 1, H, W)   1 = occluded target pixel, 0 = clear
    }

AllClear does not provide cloud-free time series, only a single cloud-free
target frame per sample. We therefore train U-TILISE following its
SEN12MSCRTS recipe: use the cloudy S2 sequence as both x and y, supply the
real cloud/shadow masks as `masks` and `cloud_mask`, and let the loss weight
only the clear pixels (cloud_mask == 0) — the model learns to reconstruct
clear pixels from the surrounding temporal context.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .common import build_allclear_dataset

NUM_S2_BANDS = 13


class UTILISEAdapter(Dataset):
    def __init__(self, json_path: str | Path, *, tx: int = 3, data_root: str | None = None):
        self.base = build_allclear_dataset(
            json_path,
            main_sensor="s2_toa",
            aux_sensors=[],
            aux_data=["cld_shdw"],
            tx=tx,
            target_mode="s2s",
            data_root=data_root,
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]

        # input_images: (C, T, H, W) -> (T, 13, H, W)
        seq = item["input_images"][:NUM_S2_BANDS].permute(1, 0, 2, 3).contiguous()
        T = seq.shape[0]

        # cloud+shadow: (2, T, H, W) -> (T, 1, H, W), clipped to {0,1}
        cld_shdw = item.get("input_cld_shdw")
        if cld_shdw is None:
            raise RuntimeError("input_cld_shdw missing; U-TILISE needs cloud masks.")
        masks = torch.clip(cld_shdw.sum(dim=0), 0, 1).unsqueeze(1)  # (T, 1, H, W)

        position_days = item["time_differences"].float()  # (T,)
        masks_valid = torch.ones(T, dtype=torch.float32)

        return {
            "x": seq,
            "y": seq,
            "masks": masks,
            "masks_valid_obs": masks_valid,
            "cloud_mask": masks,
            "position_days": position_days,
            "data_id": item.get("data_id", f"idx_{idx}"),
        }


def utilise_collate(batch):
    """Stack per-sample dicts into batched tensors with a leading batch dim."""
    out = {
        "x":               torch.stack([b["x"] for b in batch]),
        "y":               torch.stack([b["y"] for b in batch]),
        "masks":           torch.stack([b["masks"] for b in batch]),
        "masks_valid_obs": torch.stack([b["masks_valid_obs"] for b in batch]),
        "cloud_mask":      torch.stack([b["cloud_mask"] for b in batch]),
        "position_days":   torch.stack([b["position_days"] for b in batch]),
        "data_ids":        [b["data_id"] for b in batch],
    }
    return out
