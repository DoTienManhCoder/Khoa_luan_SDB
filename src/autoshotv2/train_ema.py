"""Fine-tune the full AutoShot model on ClipShots train while maintaining a weight-EMA copy.

This is a *different* training regime from `autoshotv2.train_phase2`:

  - train_phase2: trains only a classification head on cached backbone features
    (backbone frozen), with optional head-only EMA.
  - train_ema (this module): fine-tunes the *entire* `TransNetV2Supernet` on raw
    ClipShots video windows, with optional full-model EMA (`autoshotv2.ema.ModelEMA`).

The final checkpoint is saved in the same ``{"net": state_dict, ...}`` format that
`autoshotv2.eval.load_model` expects, so it can be evaluated with the standard eval
pipeline. See `reports/ema_study/` for the EMA-vs-no-EMA results this produces.

Usage:
    python -m autoshotv2.train_ema --clipshots-root data/ClipShots \
        --base-ckpt artifacts/models/ckpt_0_200_0.pth
    python -m autoshotv2.train_ema --clipshots-root data/ClipShots --no-ema
"""

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from autoshotv2.clipshots_dataset import ClipShotsTrainDataset
from autoshotv2.ema import ModelEMA

# Force line-buffered stdout so progress shows up in tailed log files even while
# the process is still running (Python block-buffers stdout when redirected).
print = partial(print, flush=True)  # noqa: A001
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune full AutoShot model on ClipShots train with optional weight-EMA.",
    )
    parser.add_argument("--clipshots-root", default="data/ClipShots",
                        help="ClipShots dataset root (expects annotations/train.json and videos/train/).")
    parser.add_argument("--base-ckpt", default="artifacts/models/ckpt_0_200_0.pth",
                        help="Phase 1 (backbone) checkpoint to initialise from.")
    parser.add_argument("--out-dir", default="artifacts/models/training",
                        help="Directory for the final and per-epoch checkpoints.")
    parser.add_argument("--out-ckpt", default=None,
                        help="Override final checkpoint filename. Default: "
                             "ckpt_ema_alpha{tag}.pth (or ckpt_baseline_noema.pth with --no-ema).")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--many-hot-weight", type=float, default=0.1,
                        help="Loss weight for the many_hot head (TransNetV2 default).")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA decay alpha (theta_ema = decay*theta_ema + (1-decay)*theta).")
    parser.add_argument("--no-ema", action="store_true",
                        help="Disable the EMA shadow; save the actual trained weights (no-EMA control).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit training to the first N videos (smoke tests).")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def alpha_tag(decay: float) -> str:
    """0.999 -> '999', 0.9999 -> '9999', 0.99 -> '99'."""
    s = f"{decay:.6f}".rstrip("0")
    return s.split(".")[-1] if "." in s else s


def load_net(base_ckpt: Path, device: str):
    """Load TransNetV2Supernet, tolerantly copying only matching tensors from `base_ckpt`."""
    from autoshotv2.model.supernet import TransNetV2Supernet

    model = TransNetV2Supernet()
    sd = model.state_dict()
    ckpt = torch.load(base_ckpt, map_location=device, weights_only=False)
    net = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
    pretrained = {k: v for k, v in net.items() if k in sd and v.shape == sd[k].shape}
    sd.update(pretrained)
    model.load_state_dict(sd)
    print(f"  Loaded {len(pretrained)}/{len(sd)} tensors from {Path(base_ckpt).name}")
    return model.to(device)


def main():
    args = parse_args()
    clipshots_root = Path(args.clipshots_root)
    base_ckpt = Path(args.base_ckpt)
    out_dir = Path(args.out_dir)
    if not base_ckpt.is_file():
        raise FileNotFoundError(f"Base checkpoint not found: {base_ckpt}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    mode_label = "no-EMA baseline" if args.no_ema else f"EMA decay={args.ema_decay}"
    print(f"Device: {device}")
    print(f"Init from: {base_ckpt}")
    print(f"Mode: {mode_label}  epochs={args.epochs}  bs={args.batch_size}  lr={args.lr}")

    # 1) Model + (optional) EMA shadow
    model = load_net(base_ckpt, device)
    model.train()
    ema = None if args.no_ema else ModelEMA(model, decay=args.ema_decay, device=device)

    # 2) Optimizer + loss. AdamW matches the reported runs; if a constrained env
    #    fails to import torch.optim (cf. ManualAdam in train_phase2), swap it there.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()

    # 3) Dataset + loader
    dataset = ClipShotsTrainDataset(clipshots_root=clipshots_root, limit=args.limit)
    if len(dataset) == 0:
        raise RuntimeError("ClipShots train dataset is empty (no videos found).")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device == "cuda"),
    )

    # 4) Training loop
    step = 0
    t_total = time.time()
    tag = alpha_tag(args.ema_decay)
    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        running = 0.0
        n_seen = 0
        for frames, oh, mh in loader:
            frames = frames.to(device, non_blocking=True)
            oh = oh.to(device, non_blocking=True)
            mh = mh.to(device, non_blocking=True)

            logits1, logits2 = model(frames)             # (B, 100, 1) each
            logits1 = logits1[:, 25:75, 0]               # (B, 50)
            logits2 = logits2[:, 25:75, 0]               # (B, 50)

            loss1 = bce(logits1, oh)
            loss2 = bce(logits2, mh)
            loss = loss1 + args.many_hot_weight * loss2

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update(model)

            running += loss.item() * frames.size(0)
            n_seen += frames.size(0)
            step += 1
            if step % args.log_every == 0:
                avg = running / max(n_seen, 1)
                print(f"  epoch {epoch} step {step}  loss={loss.item():.4f}  "
                      f"avg={avg:.4f}  (l1={loss1.item():.4f} l2={loss2.item():.4f})")

        avg = running / max(n_seen, 1)
        print(f"[epoch {epoch}/{args.epochs}] avg_loss={avg:.4f}  "
              f"time={(time.time() - t_epoch) / 60:.1f}min")

        # Save per-epoch checkpoint so partial progress survives crashes / OOM.
        if args.no_ema:
            ep_name = f"ckpt_baseline_noema_epoch{epoch}.pth"
            ep_sd = model.state_dict()
            ep_cfg = {"mode": "no_ema_baseline", "epoch": epoch}
        else:
            ep_name = f"ckpt_ema_alpha{tag}_epoch{epoch}.pth"
            ep_sd = ema.state_dict()
            ep_cfg = {"mode": "ema", "alpha": args.ema_decay, "epoch": epoch}
        ep_path = out_dir / ep_name
        torch.save({"net": ep_sd, "ema_config": ep_cfg}, ep_path)
        print(f"  Saved per-epoch ckpt -> {ep_path}")

    print(f"Total training time: {(time.time() - t_total) / 60:.1f}min")

    # 5) Save final checkpoint (EMA shadow or trained weights based on --no-ema)
    default_name = "ckpt_baseline_noema.pth" if args.no_ema else f"ckpt_ema_alpha{tag}.pth"
    out_path = out_dir / (args.out_ckpt or default_name)
    final_sd = model.state_dict() if args.no_ema else ema.state_dict()
    payload = {
        "net": final_sd,
        "ema_config": {
            "mode": "no_ema_baseline" if args.no_ema else "ema",
            "alpha": None if args.no_ema else args.ema_decay,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "many_hot_weight": args.many_hot_weight,
            "init_from": str(base_ckpt),
            "n_videos": len(dataset),
        },
    }
    torch.save(payload, out_path)
    print(f"Saved final checkpoint -> {out_path}")


if __name__ == "__main__":
    main()
