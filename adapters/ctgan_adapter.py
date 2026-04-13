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
        data_root: str | None = None,
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
            data_root=data_root,
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

        # cloud+shadow masks per input frame: (T, H, W), clipped to {0,1}.
        # Replaces the FS2-pretrained cloud detector in CTGAN's original loop.
        cld_shdw = item.get("input_cld_shdw")
        if cld_shdw is None:
            raise RuntimeError("input_cld_shdw missing; CTGAN adapter needs cloud masks.")
        masks_t = torch.clip(cld_shdw.sum(dim=0), 0, 1)  # (T, H, W)
        masks = [masks_t[t].unsqueeze(0).contiguous() for t in range(masks_t.shape[0])]  # list of (1, H, W)

        data_id = item.get("data_id", f"idx_{idx}")
        return real_A, real_B, masks, data_id


def ctgan_collate(batch):
    """
    Stack adapter outputs into the shape CTGAN's training loop expects:
        real_A: list of 3 tensors (B, 4, H, W)
        real_B: tensor (B, 4, H, W)
        masks:  list of 3 tensors (B, 1, H, W)  — per-frame cloud masks
        names:  list of data IDs
    """
    real_A_lists, real_Bs, mask_lists, names = zip(*batch)
    real_A = [torch.stack([a[t] for a in real_A_lists]) for t in range(3)]
    real_B = torch.stack(list(real_Bs))
    masks  = [torch.stack([m[t] for m in mask_lists]) for t in range(3)]
    return real_A, real_B, masks, list(names)
