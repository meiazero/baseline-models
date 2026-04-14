"""
Unified evaluation script for all three baseline models.

Computes MAE (↓), SSIM (↑), PSNR (↑) on the Brazil test subset and prints
a one-line result suitable for pasting into tab:baselines.

Usage:
    uv run python -m eval.evaluate --model ctgan       --config configs/ctgan_eval.yaml
    uv run python -m eval.evaluate --model uncrtaints  --config configs/uncrtaints_eval.yaml
    uv run python -m eval.evaluate --model utilise     --config configs/utilise_eval.yaml

Metrics match the AllClear paper protocol (Zhou et al., 2024):
  - All bands available for that model (S2 TOA [0,1] range after /10000 clip).
  - CTGAN uses 4 bands (R/G/B/NIR); UnCRtainTS and UTILISE use all 13 S2 bands.
  - Per-sample metrics averaged across the full test set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _to_numpy_01(t: torch.Tensor) -> np.ndarray:
    """Convert tensor to float32 numpy array clipped to [0, 1]."""
    return t.float().cpu().numpy().clip(0.0, 1.0)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """
    Compute MAE / SSIM / PSNR for a single sample.

    Both pred and target must be in [0, 1] with shape (C, H, W).
    Returns a dict with keys 'mae', 'ssim', 'psnr'.
    """
    p = _to_numpy_01(pred)     # (C, H, W)
    t = _to_numpy_01(target)   # (C, H, W)

    mae = float(np.abs(p - t).mean())

    # skimage expects (H, W, C) for multichannel
    p_hwc = p.transpose(1, 2, 0)
    t_hwc = t.transpose(1, 2, 0)

    ssim = float(structural_similarity(
        p_hwc, t_hwc,
        data_range=1.0,
        channel_axis=-1,
    ))
    psnr = float(peak_signal_noise_ratio(
        t_hwc, p_hwc,
        data_range=1.0,
    ))

    return {"mae": mae, "ssim": ssim, "psnr": psnr}


def aggregate(records: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean ± std over a list of per-sample metric dicts."""
    keys = list(records[0].keys())
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in records])
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return out


# ---------------------------------------------------------------------------
# Model-specific eval functions
# ---------------------------------------------------------------------------

def eval_ctgan(cfg: dict, device: torch.device) -> list[dict[str, float]]:
    """
    CTGAN inference: produce fake_B ∈ [-1, 1], rescale to [0, 1], compare to real_B.
    Only 4 bands (R/G/B/NIR at indices 3,2,1,7 of S2 TOA).
    """
    sys.path.insert(0, str(REPO_ROOT / "CTGAN"))
    from adapters.ctgan_adapter import CTGANAdapter, ctgan_collate
    from model.CTGAN import CTGAN_Generator  # type: ignore

    data_cfg = cfg["data"]
    ckpt_path = Path(cfg["checkpoint"])

    ds = CTGANAdapter(
        data_cfg["test_json"],
        tx=data_cfg["tx"],
        data_root=data_cfg.get("data_root"),
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.get("batch_size", 8),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=ctgan_collate,
        pin_memory=True,
    )

    model = CTGAN_Generator(image_size=int(data_cfg["image_size"])).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    records = []
    with torch.no_grad():
        for real_A, real_B, _masks, _names in loader:
            real_A = [a.to(device) for a in real_A]
            real_B = real_B.to(device)
            fake_B, _, _ = model(real_A)

            # Rescale [-1, 1] → [0, 1]
            pred_01 = (fake_B.clamp(-1, 1) + 1.0) / 2.0
            tgt_01  = (real_B.clamp(-1, 1) + 1.0) / 2.0

            for i in range(pred_01.shape[0]):
                records.append(compute_metrics(pred_01[i], tgt_01[i]))

    return records


def eval_uncrtaints(cfg: dict, device: torch.device) -> list[dict[str, float]]:
    """
    UnCRtainTS inference: set_input → forward → rescale → compare fake_B[:, :, :13]
    to real_B (both in [0, 1]).
    """
    import ast
    from types import SimpleNamespace

    sys.path.insert(0, str(REPO_ROOT / "UnCRtainTS" / "model"))
    from adapters.uncrtaints_adapter import UnCRtainTSAdapter, uncrtaints_collate

    data_cfg  = cfg["data"]
    model_cfg = cfg["model"]
    ckpt_path = Path(cfg["checkpoint"])

    def _parse_list(v):
        return ast.literal_eval(v) if isinstance(v, str) else v

    flat = {**model_cfg}
    for k in ("encoder_widths", "decoder_widths", "out_conv"):
        if k in flat:
            flat[k] = _parse_list(flat[k])
    flat.update({
        "lr": 1e-3,
        "batch_size": cfg.get("batch_size", 4),
        "experiment_name": "eval",
        "res_dir": "/tmp",
        "device": str(device),
        "profile": False,
    })
    ns = SimpleNamespace(**flat)

    from src.model_utils import get_base_model  # type: ignore
    model = get_base_model(ns).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict_G", ckpt.get("state_dict", ckpt))
    model.netG.load_state_dict(state)
    model.eval()

    ds = UnCRtainTSAdapter(
        data_cfg["test_json"],
        tx=data_cfg["tx"],
        data_root=data_cfg.get("data_root"),
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.get("batch_size", 4),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=uncrtaints_collate,
        pin_memory=True,
    )

    records = []
    with torch.no_grad():
        for batch in loader:
            inputs = {
                "A": batch["A"],
                "B": batch["B"],
                "dates": batch["dates"],
                "masks": batch["masks"],
            }
            model.set_input(inputs)
            model.forward()
            model.rescale()

            pred = model.fake_B[:, :, :13, ...].clamp(0, 1)  # (B, 1, 13, H, W)
            tgt  = model.real_B.clamp(0, 1)                  # (B, 1, 13, H, W)

            for i in range(pred.shape[0]):
                # squeeze temporal dim (1) → (13, H, W)
                records.append(compute_metrics(pred[i, 0], tgt[i, 0]))

    return records


def eval_utilise(cfg: dict, device: torch.device) -> list[dict[str, float]]:
    """
    U-TILISE inference: feed cloudy sequence, take mean prediction over T frames,
    compare to cloudless target (s2p mode).

    UTILISE is seq-to-seq; we compare the mean of its T output frames to the
    single cloudless AllClear reference. This matches the intent of the model
    (reconstruct all cloudy frames) while giving a single per-sample metric.
    """
    sys.path.insert(0, str(REPO_ROOT / "U-TILISE"))
    from lib.models.utilise import UTILISE  # type: ignore
    from adapters.common import build_allclear_dataset
    from torch.utils.data import Dataset

    data_cfg  = cfg["data"]
    model_cfg = cfg["model"]
    ckpt_path = Path(cfg["checkpoint"])

    # Load UTILISE model
    model = UTILISE(**model_cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = state.get("state_dict", state)
    model.load_state_dict(state)
    model.eval()

    # Eval adapter: cloudy sequence as input, cloudless target as reference.
    # Uses s2p target_mode so AllClearDataset returns a single cloud-free target.
    class _UTILISEEvalAdapter(Dataset):
        NUM_BANDS = 13

        def __init__(self, json_path, tx, data_root):
            # s2s for inputs (cloudy sequence)
            self.seq_ds = build_allclear_dataset(
                json_path,
                main_sensor="s2_toa",
                aux_sensors=[],
                aux_data=["cld_shdw"],
                tx=tx,
                target_mode="s2s",
                data_root=data_root,
            )
            # s2p for cloudless reference
            self.ref_ds = build_allclear_dataset(
                json_path,
                main_sensor="s2_toa",
                aux_sensors=[],
                aux_data=[],
                tx=tx,
                target_mode="s2p",
                data_root=data_root,
            )

        def __len__(self):
            return len(self.seq_ds)

        def __getitem__(self, idx):
            seq_item = self.seq_ds[idx]
            ref_item = self.ref_ds[idx]

            seq = seq_item["input_images"][:self.NUM_BANDS].permute(1, 0, 2, 3).contiguous()
            T   = seq.shape[0]

            cld_shdw = seq_item["input_cld_shdw"]
            masks    = torch.clip(cld_shdw.sum(0), 0, 1).unsqueeze(1)   # (T, 1, H, W)
            masks_v  = torch.ones(T, dtype=torch.float32)
            pos_days = seq_item["time_differences"].float()

            # Cloudless reference: (C, 1, H, W) → (13, H, W)
            target = ref_item["target"][:self.NUM_BANDS, 0].contiguous()

            return {
                "x":               seq,
                "masks":           masks,
                "masks_valid_obs": masks_v,
                "position_days":   pos_days,
                "target":          target,
            }

    def _collate(batch):
        return {
            "x":               torch.stack([b["x"] for b in batch]),
            "masks":           torch.stack([b["masks"] for b in batch]),
            "masks_valid_obs": torch.stack([b["masks_valid_obs"] for b in batch]),
            "position_days":   torch.stack([b["position_days"] for b in batch]),
            "target":          torch.stack([b["target"] for b in batch]),
        }

    ds = _UTILISEEvalAdapter(
        data_cfg["test_json"],
        tx=data_cfg["tx"],
        data_root=data_cfg.get("data_root"),
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.get("batch_size", 4),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=_collate,
        pin_memory=True,
    )

    records = []
    with torch.no_grad():
        for batch in loader:
            x       = batch["x"].to(device)          # (B, T, 13, H, W)
            pos     = batch["position_days"].to(device)
            target  = batch["target"].to(device)      # (B, 13, H, W)

            y_pred = model(x, batch_positions=pos)    # (B, T, 13, H, W)

            # Average over temporal dim → (B, 13, H, W)
            pred_mean = y_pred.mean(dim=1).clamp(0, 1)
            tgt_01    = target.clamp(0, 1)

            for i in range(pred_mean.shape[0]):
                records.append(compute_metrics(pred_mean[i], tgt_01[i]))

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_EVAL_FNS = {
    "ctgan":      eval_ctgan,
    "uncrtaints": eval_uncrtaints,
    "utilise":    eval_utilise,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True, choices=list(_EVAL_FNS))
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] model={args.model} device={device}")

    records = _EVAL_FNS[args.model](cfg, device)

    stats = aggregate(records)
    print(f"\n[eval] n_samples={len(records)}")
    print(f"  MAE  = {stats['mae']['mean']:.4f} ± {stats['mae']['std']:.4f}")
    print(f"  SSIM = {stats['ssim']['mean']:.4f} ± {stats['ssim']['std']:.4f}")
    print(f"  PSNR = {stats['psnr']['mean']:.4f} ± {stats['psnr']['std']:.4f}")

    # Machine-readable output for scripting
    out_path = Path(cfg.get("output_json", f"results/{args.model}_eval.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "n": len(records), "stats": stats}, f, indent=2)
    print(f"[eval] results saved to {out_path}")

    # LaTeX-friendly one-liner
    m, s, p = stats["mae"]["mean"], stats["ssim"]["mean"], stats["psnr"]["mean"]
    print(f"\n[latex] {m:.3f} & {s:.3f} & {p:.3f}")


if __name__ == "__main__":
    main()
