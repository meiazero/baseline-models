"""
Lean CTGAN training entrypoint.

The upstream training loop (CTGAN/train.py) expects:
  - Sen2_MTC dataset directory layout
  - a pretrained FS2 cloud-detection feature extractor (Pretrain/Feature_Extrator_FS2.pth)
    used only to supervise the generator's internal attention masks
  - an optional pretrained generator/discriminator to resume from

We drive the same CTGAN_Generator / CTGAN_Discriminator modules on the
AllClear Brazil subset via CTGANAdapter. The FS2 feature extractor is
replaced by the real AllClear cloud/shadow masks, which CTGANAdapter
already produces per input frame — an objectively better target than a
learned cloud probability map.

Usage:
    uv run python -m train.train_ctgan --config configs/ctgan.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "CTGAN"))

from adapters.ctgan_adapter import CTGANAdapter, ctgan_collate  # noqa: E402


def _load_model_classes():
    from model.CTGAN import CTGAN_Generator, CTGAN_Discriminator  # type: ignore
    from utils import GANLoss, set_requires_grad  # type: ignore
    return CTGAN_Generator, CTGAN_Discriminator, GANLoss, set_requires_grad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    image_size = int(data_cfg["image_size"])

    torch.manual_seed(train_cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    GEN_cls, DIS_cls, GANLoss, set_requires_grad = _load_model_classes()

    GEN = GEN_cls(image_size=image_size).to(device)
    DIS = DIS_cls().to(device)

    criterion_gan = GANLoss(train_cfg["gan_mode"]).to(device)
    criterion_l1 = nn.L1Loss().to(device)
    criterion_mse = nn.MSELoss().to(device)

    opt_G = torch.optim.AdamW(GEN.parameters(), lr=float(train_cfg["lr"]), betas=(0.5, 0.999), weight_decay=5e-4)
    opt_D = torch.optim.AdamW(DIS.parameters(), lr=float(train_cfg["lr"]), betas=(0.5, 0.999), weight_decay=5e-4)
    sched_G = CosineAnnealingLR(opt_G, T_max=train_cfg["epochs"], eta_min=1e-6)
    sched_D = CosineAnnealingLR(opt_D, T_max=train_cfg["epochs"], eta_min=1e-6)

    data_root = data_cfg.get("data_root")
    train_ds = CTGANAdapter(data_cfg["train_json"], tx=data_cfg["tx"], data_root=data_root)
    val_ds = CTGANAdapter(data_cfg["val_json"], tx=data_cfg["tx"], data_root=data_root)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        collate_fn=ctgan_collate,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        collate_fn=ctgan_collate,
        pin_memory=True,
    )

    out_dir = Path(cfg["res_dir"]) / cfg["experiment_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    noise = train_cfg["label_noise"]
    lam_L1 = float(train_cfg["lambda_L1"])
    lam_aux = float(train_cfg["lambda_aux"])
    use_aux = bool(train_cfg["aux_loss"])

    print(f"[ctgan] train={len(train_ds)} val={len(val_ds)} device={device}")

    for epoch in range(1, train_cfg["epochs"] + 1):
        GEN.train()
        DIS.train()
        t0 = time.time()
        running_G = 0.0

        for step, (real_A, real_B, masks, _names) in enumerate(train_loader):
            real_A = [a.to(device) for a in real_A]
            real_B = real_B.to(device)
            real_A_cat = torch.cat(real_A, dim=1)

            # Target cloud masks moved to device; spatial resize deferred to
            # loss computation so we match the generator's actual att_mask dims.
            M = [m.to(device) for m in masks]

            fake_B, att_masks, aux_pred = GEN(real_A)

            # --- Discriminator
            set_requires_grad(DIS, True)
            opt_D.zero_grad()
            fake_AB = torch.cat((real_A_cat, fake_B), dim=1)
            pred_fake = DIS(fake_AB.detach())
            loss_D_fake = criterion_gan(pred_fake, False, noise)
            real_AB = torch.cat((real_A_cat, real_B), dim=1)
            pred_real = DIS(real_AB)
            loss_D_real = criterion_gan(pred_real, True, noise)
            loss_D = 0.5 * (loss_D_fake + loss_D_real)
            loss_D.backward()
            opt_D.step()

            # --- Generator
            set_requires_grad(DIS, False)
            opt_G.zero_grad()
            fake_AB = torch.cat((real_A_cat, fake_B), dim=1)
            pred_fake = DIS(fake_AB)
            loss_G_gan = criterion_gan(pred_fake, True, noise)
            loss_G_l1 = criterion_l1(fake_B, real_B) * lam_L1
            loss_g_att = 0.0
            for i, att in enumerate(att_masks):
                att_h, att_w = att.shape[-2], att.shape[-1]
                m = F.interpolate(M[i], size=(att_h, att_w), mode="bilinear", align_corners=False)
                loss_g_att = loss_g_att + criterion_mse(att[:, 0], m[:, 0])

            if use_aux:
                loss_G_aux = (
                    criterion_l1(aux_pred[0], real_B)
                    + criterion_l1(aux_pred[1], real_B)
                    + criterion_l1(aux_pred[2], real_B)
                ) * lam_aux
                loss_G = loss_G_gan + loss_G_l1 + loss_g_att + loss_G_aux
            else:
                loss_G = loss_G_gan + loss_G_l1 + loss_g_att
            loss_G.backward()
            opt_G.step()
            running_G += float(loss_G.detach().cpu())

            if (step + 1) % 20 == 0:
                print(
                    f"[ctgan] epoch {epoch} step {step + 1}/{len(train_loader)} "
                    f"G={running_G / (step + 1):.4f} D={float(loss_D):.4f}"
                )

        sched_G.step()
        sched_D.step()

        # --- validation: plain L1 on (-1,1)-scale fake_B
        GEN.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for real_A, real_B, _masks, _names in val_loader:
                real_A = [a.to(device) for a in real_A]
                real_B = real_B.to(device)
                fake_B, _, _ = GEN(real_A)
                val_loss += F.l1_loss(fake_B, real_B).item() * real_B.size(0)
                n += real_B.size(0)
        val_loss /= max(n, 1)
        elapsed = time.time() - t0
        print(
            f"[ctgan] epoch {epoch} done train_G={running_G / max(len(train_loader),1):.4f} "
            f"val_l1={val_loss:.4f} time={elapsed:.1f}s"
        )

        torch.save(GEN.state_dict(), out_dir / "G_last.pth")
        torch.save(DIS.state_dict(), out_dir / "D_last.pth")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(GEN.state_dict(), out_dir / "G_best.pth")
            torch.save(DIS.state_dict(), out_dir / "D_best.pth")
            print(f"[ctgan] new best val_l1={best_val:.4f} saved")


if __name__ == "__main__":
    main()
