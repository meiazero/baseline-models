"""
Common helpers for building AllClearDataset from a filtered Brazil-subset JSON.

All per-model adapters in this package wrap the dataset built here and
reshape/normalise the samples into the format their model expects.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable

from torch.utils.data import Dataset

# Allow importing allclear from the sibling repo without installing it.
_ALLCLEAR_ROOT = Path(__file__).resolve().parents[2] / "allclear"
if str(_ALLCLEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(_ALLCLEAR_ROOT))

from allclear import AllClearDataset  # noqa: E402


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
    Build an AllClearDataset pointing at a filtered Brazil-subset JSON.

    The JSON path is resolved relative to the caller's CWD if not absolute.
    Auxiliary sensors default to none (pure S2 reconstruction); auxiliary
    data defaults to cloud/shadow masks only — dynamic-world land cover is
    not required for training.
    """
    if data_root is not None:
        os.environ["ALLCLEAR_DATA_ROOT"] = str(data_root)

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
        s2_toa_channels=s2_toa_channels,
    )
