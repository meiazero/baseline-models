"""
Standalone AllClear dataset loader for baseline-models.

Self-contained copy of AllClearDataset logic, decoupled from the allclear
package. Adds native data_root support so relative or foreign-server paths
in the JSON metadata resolve correctly on any host.
"""

from __future__ import annotations

import glob
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio as rs
import torch
from torch.utils.data import Dataset


def _temporal_align(main_timestamps: list, aux_timestamp, max_diff: int = 2):
    diffs = [abs(dt - aux_timestamp) for dt in main_timestamps]
    if min(diffs).days > max_diff:
        return None
    return diffs.index(min(diffs))


def _resolve(fpath: str, data_root: str | None) -> str:
    """Return fpath, rebasing under data_root if the original doesn't exist."""
    if os.path.exists(fpath):
        return fpath
    if data_root:
        candidate = os.path.join(data_root, fpath.lstrip("/"))
        if os.path.exists(candidate):
            return candidate
    return fpath  # let caller raise a useful error


class AllClearDataset(Dataset):
    """
    Loads and preprocesses AllClear satellite imagery from a JSON metadata dict.

    Parameters
    ----------
    dataset : dict
        JSON metadata dict keyed by sample ID. Each entry has:
        ``roi``, ``s2_toa``, (optionally ``s1``, ``cld_shdw``, ``dw``), ``target``.
    selected_rois : list | "all"
        ROI names to keep, or "all" for the full dataset.
    main_sensor : str
        Primary sensor key, default "s2_toa".
    aux_sensors : list[str]
        Additional sensors to load alongside the main one.
    aux_data : list[str]
        Auxiliary mask types to load, e.g. ["cld_shdw"].
    tx : int
        Number of input time steps.
    target_mode : "s2p" | "s2s"
        seq2point (one cloud-free target) or seq2seq (cloudy sequence as target).
    data_root : str | Path | None
        Base directory for resolving relative or foreign-host .tif paths.
        When set, any path that does not exist on disk is looked up as
        ``data_root / fpath.lstrip("/")``.
    center_crop_size : tuple[int, int]
        Spatial crop applied to every loaded tile.
    max_diff : int
        Maximum day offset for aligning auxiliary sensors to the main sensor.
    """

    _S2_TOA_CHANNELS = list(range(1, 14))
    _CHANNELS: dict[str, list[int]] = {
        "s2_toa":   list(range(1, 14)),
        "s1":       [1, 2],
        "landsat8": list(range(1, 12)),
        "landsat9": list(range(1, 12)),
        "cld_shdw": [2, 5],
        "dw":       [1],
    }

    def __init__(
        self,
        dataset: dict,
        selected_rois: list | str = "all",
        main_sensor: str = "s2_toa",
        aux_sensors: list[str] | None = None,
        aux_data: list[str] | None = None,
        tx: int = 3,
        target_mode: str = "s2p",
        data_root: str | Path | None = None,
        center_crop_size: tuple[int, int] = (256, 256),
        max_diff: int = 2,
        s2_toa_channels: list[int] | None = None,
        s1_preprocess_mode: str = "default",
    ):
        if selected_rois == "all":
            self.dataset = dataset
        else:
            self.dataset = {k: v for k, v in dataset.items() if v["roi"][0] in selected_rois}

        self.main_sensor = main_sensor
        self.aux_sensors = aux_sensors or []
        self.sensors = [main_sensor] + self.aux_sensors
        self.aux_data = aux_data if aux_data is not None else ["cld_shdw", "dw"]
        self.tx = tx
        self.target_mode = target_mode
        self.data_root = str(data_root) if data_root is not None else None
        self.center_crop_size = center_crop_size
        self.max_diff = max_diff
        self.s1_preprocess_mode = s1_preprocess_mode
        self.dataset_ids = list(self.dataset.keys())

        self.channels = dict(self._CHANNELS)
        if s2_toa_channels is not None:
            self.channels["s2_toa"] = s2_toa_channels

    def __len__(self) -> int:
        return len(self.dataset)

    def _resolve(self, fpath: str) -> str:
        return _resolve(fpath, self.data_root)

    @staticmethod
    def load_and_center_crop(
        fpath: str,
        channels: list[int] | None = None,
        size: tuple[int, int] = (256, 256),
    ) -> torch.Tensor:
        with rs.open(fpath) as src:
            cx, cy = src.width // 2, src.height // 2
            window = rs.windows.Window(
                cx - size[0] // 2, cy - size[1] // 2, size[0], size[1]
            )
            data = src.read(channels, window=window) if channels else src.read(window=window)
        return torch.from_numpy(data).float()

    @staticmethod
    def extract_date(path: str) -> datetime:
        parts = path.split("_")
        return datetime.strptime(f"{parts[-4]}-{parts[-3]}-{parts[-2]}", "%Y-%m-%d")

    def preprocess(self, image: torch.Tensor, sensor: str) -> torch.Tensor:
        if sensor in ("s2_toa", "landsat8", "landsat9"):
            image = torch.clip(image, 0, 10000) / 10000
            image = torch.nan_to_num(image, nan=0.0)
        elif sensor == "s1":
            if self.s1_preprocess_mode == "default":
                image[image < -40] = -40
                image[0] = torch.clip(image[0] + 25, 0, 25) / 25
                image[1] = torch.clip(image[1] + 32.5, 0, 32.5) / 32.5
                image = torch.nan_to_num(image, nan=-1.0)
            else:  # uncrtaints
                dB_min, dB_max = -25.0, 0.0
                image = torch.clip(image, dB_min, dB_max)
                image = (image - dB_min) / (dB_max - dB_min)
        elif sensor == "cld_shdw":
            image = torch.nan_to_num(image, nan=1.0)
        return image

    def _load(self, fpath: str, sensor: str) -> torch.Tensor:
        fpath = self._resolve(fpath)
        image = self.load_and_center_crop(fpath, self.channels[sensor], self.center_crop_size)
        return self.preprocess(image, sensor)

    def __getitem__(self, idx: int) -> dict:
        data_id = self.dataset_ids[idx]
        sample = self.dataset[data_id]
        roi, latlong = sample["roi"][0], sample["roi"][1]

        inputs: dict[str, list] = {s: [] for s in self.sensors}
        inputs["input_cld_shdw"] = []
        inputs["input_dw"] = []

        timestamps: list[datetime] = []

        for sensor in self.sensors:
            for timestamp_str, fpath in sample[sensor]:
                ts = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                image = self._load(fpath, sensor)
                inputs[sensor].append((ts, image))

                if sensor == self.main_sensor:
                    timestamps.append(ts)
                    resolved = self._resolve(fpath)

                    if "cld_shdw" in self.aux_data:
                        cld_fpath = resolved.replace("s2_toa", "cld_shdw")
                        if not os.path.exists(cld_fpath):
                            raise ValueError(f"Cloud shadow file not found: {cld_fpath}")
                        cld = self.load_and_center_crop(cld_fpath, self.channels["cld_shdw"], self.center_crop_size)
                        cld = self.preprocess(cld, "cld_shdw")
                        inputs["input_cld_shdw"].append((ts, cld))

                    if "dw" in self.aux_data:
                        dw_fpath = resolved.replace("s2_toa", "dw")
                        if os.path.exists(dw_fpath):
                            dw = self.load_and_center_crop(dw_fpath, self.channels["dw"], self.center_crop_size)
                            dw = self.preprocess(dw, "dw")
                            inputs["input_dw"].append((ts, dw))
                        else:
                            inputs["input_dw"].append(
                                (ts, torch.ones(len(self.channels["dw"]), *self.center_crop_size))
                            )

            inputs[sensor].sort(key=lambda x: x[0])

        timestamps_main = [ts for ts, _ in inputs[self.main_sensor]]
        timestamps_sorted = sorted(set(timestamps))
        start = timestamps_sorted[0]
        time_diffs = [round((ts - start).total_seconds() / 86400) for ts in timestamps_sorted]
        timestamps_unix = [ts.timestamp() for ts in timestamps_sorted]

        # Build (T, C, H, W) input tensor, then permute to (C, T, H, W)
        stp = torch.stack([img for _, img in inputs[self.main_sensor]])  # (T, C, H, W)
        stp = stp.permute(1, 0, 2, 3)  # (C, T, H, W)

        inputs_cld_shdw = None
        if "cld_shdw" in self.aux_data and inputs["input_cld_shdw"]:
            inputs_cld_shdw = torch.stack([c for _, c in inputs["input_cld_shdw"]]).permute(1, 0, 2, 3)

        inputs_dw = None
        if "dw" in self.aux_data and inputs["input_dw"]:
            inputs_dw = torch.stack([d for _, d in inputs["input_dw"]]).permute(1, 0, 2, 3)

        # Target
        if self.target_mode == "s2p":
            if "target" not in sample:
                raise ValueError(f"Sample {data_id} has no 'target' key.")
            tgt_ts_str, tgt_fpath = sample["target"][0]
            tgt_image = self._load(tgt_fpath, self.main_sensor).unsqueeze(1)  # (C, 1, H, W)
            tgt_timestamps = datetime.strptime(tgt_ts_str, "%Y-%m-%d %H:%M:%S").timestamp()

            tgt_cld_shdw = None
            if "cld_shdw" in self.aux_data:
                resolved_tgt = self._resolve(tgt_fpath)
                cld_fpath = resolved_tgt.replace("s2_toa", "cld_shdw")
                if not os.path.exists(cld_fpath):
                    raise ValueError(f"Cloud shadow file not found: {cld_fpath}")
                cld = self.load_and_center_crop(cld_fpath, self.channels["cld_shdw"], self.center_crop_size)
                cld = self.preprocess(cld, "cld_shdw")
                tgt_cld_shdw = cld.unsqueeze(0).permute(1, 0, 2, 3)  # (2, 1, H, W)

        elif self.target_mode == "s2s":
            tgt_image = stp  # (C, T, H, W)
            tgt_timestamps = timestamps_unix
            tgt_cld_shdw = None
        else:
            raise ValueError(f"Unknown target_mode: {self.target_mode!r}")

        item: dict = {
            "data_id": data_id,
            "input_images": stp,
            "target": tgt_image,
            "timestamps": torch.tensor(timestamps_unix),
            "target_timestamps": torch.tensor(tgt_timestamps),
            "time_differences": torch.tensor(time_diffs, dtype=torch.float32),
            "roi": roi,
            "latlong": latlong,
        }
        if inputs_cld_shdw is not None:
            item["input_cld_shdw"] = inputs_cld_shdw
        if inputs_dw is not None:
            item["input_dw"] = inputs_dw
        if tgt_cld_shdw is not None:
            item["target_cld_shdw"] = tgt_cld_shdw

        return item
