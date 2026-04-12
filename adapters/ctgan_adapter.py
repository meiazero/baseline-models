"""
CTGAN adapter: AllClearDataset -> (list_of_three_cloudy, cloudless, data_id).

CTGAN's original training loop (CTGAN/train.py) expects per-batch items of
the form:

    real_A, real_B, name = next(iter(loader))
    # real_A is a list [img0, img1, img2], each shape (B, 4, H, W)
    # real_B is shape (B, 4, H, W)

The model consumes Sentinel-2 bands (R, G, B, NIR) only — tensor indices
(3, 2, 1, 7) out of the 13 S2 TOA bands — and pixel values in [-1, 1].

This adapter produces exactly that shape from the AllClear Brazil subset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

from .common import build_allclear_dataset

# R, G, B, NIR in AllClear's 0-indexed 13-band S2 TOA tensor
CTGAN_BANDS: tuple[int, int, int, int] = (3, 2, 1, 7)


class CTGANAdapter(Dataset):
    def __init__(
        self,
        json_path: str | Path,
        *,
        tx: int = 3,
        bands: Iterable[int] = CTGAN_BANDS,
    ):
        if tx != 3:
            raise ValueError("CTGAN is hard-coded to three input frames (tx=3).")
        self.bands = tuple(bands)
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

        # input_images: (C, T, H, W) -> (4, 3, H, W) after band selection
        inputs = item["input_images"][list(self.bands)]  # (4, T, H, W)
        inputs = inputs * 2.0 - 1.0                       # [0,1] -> [-1,1]
        real_A = [inputs[:, t].contiguous() for t in range(inputs.shape[1])]

        # target: (C, 1, H, W) -> (4, H, W)
        target = item["target"][list(self.bands), 0]      # (4, H, W)
        real_B = target * 2.0 - 1.0

        data_id = item.get("data_id", f"idx_{idx}")
        return real_A, real_B, data_id


def ctgan_collate(batch):
    """
    Stack adapter outputs into the shape CTGAN's training loop expects:
        real_A: list of 3 tensors (B, 4, H, W)
        real_B: tensor (B, 4, H, W)
        names:  list of data IDs
    """
    real_A_lists, real_Bs, names = zip(*batch)
    real_A = [torch.stack([a[t] for a in real_A_lists]) for t in range(3)]
    real_B = torch.stack(list(real_Bs))
    return real_A, real_B, list(names)
