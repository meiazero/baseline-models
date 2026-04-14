"""
Lean UnCRtainTS training entrypoint.

Instead of driving UnCRtainTS/model/train_reconstruct.py (which hard-codes
SEN12MS-CR-TS directory layouts and embeds a 700-line training/validation
loop), we import the model class directly from UnCRtainTS.model.src and run
a minimal training loop on top of the AllClear Brazil subset via
UnCRtainTSAdapter.

The model's BaseModel exposes a clean interface:
    model.set_input({'A': x, 'B': y, 'dates': d, 'masks': m})
    model.optimize_parameters()
    model.fake_B  # current prediction
    model.loss_G  # current loss
which is all we need.

Usage:
    uv run python -m train.train_uncrtaints --config configs/uncrtaints.yaml
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "UnCRtainTS" / "model"))

from adapters.uncrtaints_adapter import UnCRtainTSAdapter, uncrtaints_collate  # noqa: E402


def _parse_list(v):
    if isinstance(v, str):
        return ast.literal_eval(v)
    return v


def _build_config(cfg_yaml: dict) -> SimpleNamespace:
    """Fold YAML into a flat namespace matching UnCRtainTS's argparse output."""
    from parse_args import create_parser  # type: ignore

    defaults = vars(create_parser(mode="train").parse_args([]))
    model = dict(cfg_yaml["model"])
    for k in ("encoder_widths", "decoder_widths", "out_conv"):
        if k in model:
            model[k] = _parse_list(model[k])
    training = cfg_yaml["training"]
    flat = {
        **defaults,
        **model,
        "lr": float(training["lr"]),
        "batch_size": training["batch_size"],
        "experiment_name": cfg_yaml["experiment_name"],
        "res_dir": cfg_yaml["res_dir"],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "profile": False,
    }
    return SimpleNamespace(**flat)


def _save_checkpoint(model, path: Path, epoch: int, best_val: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "state_dict_G": model.netG.state_dict(),
            "optimizer_G": model.optimizer_G.state_dict(),
            "scheduler_G": model.scheduler_G.state_dict(),
            "best_val": best_val,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = _build_config(cfg)

    # Import here so sys.path manipulations are already in effect.
    from src.model_utils import get_base_model  # type: ignore

    device = torch.device(model_cfg.device)
    model = get_base_model(model_cfg).to(device)
    model.len_epoch = 0  # required by BaseModel

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    data_root = data_cfg.get("data_root")
    train_ds = UnCRtainTSAdapter(data_cfg["train_json"], tx=data_cfg["tx"], data_root=data_root)
    val_ds = UnCRtainTSAdapter(data_cfg["val_json"], tx=data_cfg["tx"], data_root=data_root)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        collate_fn=uncrtaints_collate,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        collate_fn=uncrtaints_collate,
        pin_memory=True,
    )

    out_dir = Path(model_cfg.res_dir) / model_cfg.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bs = train_cfg["batch_size"]
    total_epochs = train_cfg["epochs"]
    log_every = int(train_cfg.get("log_every", 20))
    steps_per_epoch = len(train_loader)

    print(
        f"[uncrtaints] train={len(train_ds)} val={len(val_ds)} device={device} "
        f"batch_size={bs} epochs={total_epochs} steps/epoch={steps_per_epoch} "
        f"params={n_params/1e6:.2f}M trainable={n_trainable/1e6:.2f}M "
        f"out_dir={out_dir}"
    )

    global_step = 0
    train_start = time.time()

    for epoch in range(1, total_epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        window_loss = 0.0
        window_count = 0
        window_t0 = time.time()
        print(f"[uncrtaints] === epoch {epoch}/{total_epochs} start ===")

        for step, batch in enumerate(train_loader):
            inputs = {
                "A": batch["A"],
                "B": batch["B"],
                "dates": batch["dates"],
                "masks": batch["masks"],
            }
            model.set_input(inputs)
            model.optimize_parameters()
            step_loss = float(model.loss_G.detach().cpu())
            running += step_loss
            window_loss += step_loss
            window_count += 1
            global_step += 1

            if (step + 1) % log_every == 0 or (step + 1) == steps_per_epoch:
                dt = time.time() - window_t0
                imgs_per_s = (window_count * bs) / max(dt, 1e-6)
                avg_window = window_loss / max(window_count, 1)
                avg_epoch = running / (step + 1)
                lr = model.optimizer_G.param_groups[0]["lr"]
                mem = (
                    torch.cuda.max_memory_allocated() / 1024**3
                    if device.type == "cuda"
                    else 0.0
                )
                epoch_elapsed = time.time() - t0
                eta_epoch = epoch_elapsed / (step + 1) * (steps_per_epoch - step - 1)
                print(
                    f"[uncrtaints] ep {epoch}/{total_epochs} "
                    f"step {step + 1}/{steps_per_epoch} "
                    f"loss={step_loss:.4f} "
                    f"avg20={avg_window:.4f} "
                    f"avg_ep={avg_epoch:.4f} "
                    f"lr={lr:.2e} "
                    f"img/s={imgs_per_s:.1f} "
                    f"gpu={mem:.2f}G "
                    f"eta={eta_epoch:.0f}s"
                )
                window_loss = 0.0
                window_count = 0
                window_t0 = time.time()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()

        model.scheduler_G.step()
        train_time = time.time() - t0

        # --- validation: plain L1 on rescaled predictions
        model.eval()
        val_loss = 0.0
        val_nll = 0.0
        n = 0
        v0 = time.time()
        with torch.no_grad():
            for batch in val_loader:
                inputs = {
                    "A": batch["A"],
                    "B": batch["B"],
                    "dates": batch["dates"],
                    "masks": batch["masks"],
                }
                model.set_input(inputs)
                model.forward()
                model.get_loss_G()
                val_nll += float(model.loss_G.detach().cpu()) * batch["B"].size(0)
                model.rescale()
                pred = model.fake_B[:, :, :13, ...]
                target = model.real_B
                val_loss += torch.nn.functional.l1_loss(pred, target).item() * target.size(0)
                n += target.size(0)
        val_loss /= max(n, 1)
        val_nll /= max(n, 1)
        val_time = time.time() - v0
        total_elapsed = time.time() - train_start
        remaining_epochs = total_epochs - epoch
        eta_total = (total_elapsed / epoch) * remaining_epochs
        print(
            f"[uncrtaints] === epoch {epoch}/{total_epochs} done "
            f"train_loss={running / max(steps_per_epoch,1):.4f} "
            f"val_nll={val_nll:.4f} val_l1={val_loss:.4f} "
            f"best_val_l1={min(best_val, val_loss):.4f} "
            f"train_time={train_time:.1f}s val_time={val_time:.1f}s "
            f"total={total_elapsed/60:.1f}m eta={eta_total/60:.1f}m ==="
        )

        _save_checkpoint(model, out_dir / "last.pth.tar", epoch, best_val)
        if val_loss < best_val:
            best_val = val_loss
            _save_checkpoint(model, out_dir / "best.pth.tar", epoch, best_val)
            print(f"[uncrtaints] new best val_l1={best_val:.4f} saved -> {out_dir/'best.pth.tar'}")


if __name__ == "__main__":
    main()
