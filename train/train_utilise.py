"""
Lean U-TILISE training entrypoint.

U-TILISE's upstream run_train.py drives a config-heavy Trainer class tied
to SEN12MS-CR-TS/EarthNet2021 dataset wrappers. We skip that machinery
and instantiate the UTILISE model directly, then run a minimal loop that
reuses TrainLoss (from lib.loss) against batches produced by
UTILISEAdapter.

Usage:
    uv run python -m train.train_utilise --config configs/utilise.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from prodict import Prodict
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "U-TILISE"))

from adapters.utilise_adapter import UTILISEAdapter, utilise_collate  # noqa: E402


def _load_model_and_loss():
    from lib.models.utilise import UTILISE  # type: ignore
    from lib.loss import TrainLoss  # type: ignore
    return UTILISE, TrainLoss


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    loss_cfg = cfg["loss"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    UTILISE, TrainLoss = _load_model_and_loss()
    model = UTILISE(**model_cfg).to(device)

    loss_fn = TrainLoss(Prodict.from_dict(loss_cfg))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=float(train_cfg["scheduler_gamma"]))

    data_root = data_cfg.get("data_root")
    train_ds = UTILISEAdapter(data_cfg["train_json"], tx=data_cfg["tx"], data_root=data_root)
    val_ds = UTILISEAdapter(data_cfg["val_json"], tx=data_cfg["tx"], data_root=data_root)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        collate_fn=utilise_collate,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        collate_fn=utilise_collate,
        pin_memory=True,
    )

    out_dir = Path(cfg["res_dir"]) / cfg["experiment_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    print(f"[utilise] train={len(train_ds)} val={len(val_ds)} device={device}")

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        t0 = time.time()
        running = 0.0

        for step, batch in enumerate(train_loader):
            batch = _move_batch(batch, device)
            optimizer.zero_grad()
            y_pred = model(batch["x"], batch_positions=batch["position_days"])
            _, loss = loss_fn(batch, y_pred)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu())
            if (step + 1) % 20 == 0:
                print(
                    f"[utilise] epoch {epoch} step {step + 1}/{len(train_loader)} "
                    f"loss={running / (step + 1):.4f}"
                )

        scheduler.step()

        # --- validation: L1 on clear (non-occluded) target pixels
        model.eval()
        val_sum = 0.0
        val_px = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = _move_batch(batch, device)
                y_pred = model(batch["x"], batch_positions=batch["position_days"])
                clear = 1.0 - batch["cloud_mask"].expand_as(batch["y"])
                diff = (y_pred - batch["y"]).abs() * clear
                val_sum += diff.sum().item()
                val_px += clear.sum().item()
        val_loss = val_sum / max(val_px, 1.0)
        elapsed = time.time() - t0
        print(
            f"[utilise] epoch {epoch} done train_loss={running / max(len(train_loader),1):.4f} "
            f"val_l1_clear={val_loss:.4f} time={elapsed:.1f}s"
        )

        torch.save(
            {"epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()},
            out_dir / "last.pth.tar",
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict(), "best_val": best_val},
                out_dir / "best.pth.tar",
            )
            print(f"[utilise] new best val_l1_clear={best_val:.4f} saved")


if __name__ == "__main__":
    main()
