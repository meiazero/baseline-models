"""
Common helpers for building AllClearDataset from a filtered Brazil-subset JSON.

Uses the standalone adapters.dataset.AllClearDataset — no allclear package dep.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from torch.utils.data import Dataset

from .dataset import AllClearDataset


def load_dataset_json(json_path: str | Path) -> dict:
    with open(json_path) as f:
        return json.load(f)


def build_allclear_dataset(
    json_path: str | Path,
    *,
    main_sensor: str = "s2_toa",
    aux_sensors: Iterable[str] | None = None,
    aux_data: Iterable[str] | None = None,
    tx: int = 3,
    target_mode: str = "s2p",
    s2_toa_channels: list[int] | None = None,
    data_root: str | Path | None = None,
) -> Dataset:
    """
    Build an AllClearDataset from a filtered Brazil-subset JSON.

    Parameters
    ----------
    json_path : path to the metadata JSON file.
    data_root : base directory for resolving .tif paths in the JSON.
                Any path that does not exist is looked up as
                ``data_root / fpath.lstrip("/")``.
    """
    json_path = Path(json_path)
    if not json_path.is_absolute():
        json_path = Path.cwd() / json_path
    if not json_path.exists():
        raise FileNotFoundError(f"Dataset JSON not found: {json_path}")

    dataset = load_dataset_json(json_path)

    return AllClearDataset(
        dataset=dataset,
        selected_rois="all",
        main_sensor=main_sensor,
        aux_sensors=list(aux_sensors) if aux_sensors else [],
        aux_data=list(aux_data) if aux_data is not None else ["cld_shdw"],
        tx=tx,
        target_mode=target_mode,
        data_root=data_root,
        s2_toa_channels=s2_toa_channels,
    )
