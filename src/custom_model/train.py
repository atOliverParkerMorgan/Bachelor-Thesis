from __future__ import annotations

import argparse
import json
import os
import random
from importlib import import_module
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

from src.custom_model.dataset import WoodDefectDataset
from src.custom_model.losses import get_loss
from src.custom_model.model import get_swin_model
from src.custom_model.transforms import get_train_transforms, get_val_transforms


@dataclass
class TrainConfig:
    image_dir: str
    label_dir: str
    output_dir: str = "./output/custom_model"
    epochs: int = 1000
    batch_size: int = 2
    patch_size: tuple[int, int, int] = (128, 384, 128)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_classes: int = 7
    val_fraction: float = 0.25
    num_workers: int = 0
    cache_rate: float = 0.0
    seed: int = 42
    amp: bool = True
    sliding_window_overlap: float = 0.5
    # Rare-class (label 6 = Poškození hmyzem) handling
    rare_label_idx: int = 6
    rare_class_weight: float = 30.0
    oversample_factor: int = 8
    wandb: bool = False
    wandb_project: str = "bp-custom-model"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_cases(cases, val_fraction, seed):
    if len(cases) < 2:
        raise ValueError("Need at least two volumes to create a train/validation split.")

    rng = random.Random(seed)
    indices = list(range(len(cases)))
    rng.shuffle(indices)

    n_val = max(1, int(round(len(cases) * val_fraction)))
    n_val = min(n_val, len(cases) - 1)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    return [cases[i] for i in train_indices], [cases[i] for i in val_indices]


def _init_wandb(config: TrainConfig):
    try:
        wandb = import_module("wandb")
    except ImportError:
        print("WARNING: wandb is not installed. Continuing without W&B logging.")
        return None

    project = config.wandb_project or os.environ.get("WANDB_PROJECT") or "bp-custom-model"
    entity = config.wandb_entity or os.environ.get("WANDB_ENTITY") or None
    run_name = config.wandb_run_name or os.environ.get("WANDB_RUN_NAME") or None

    wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=asdict(config),
        resume="allow",
    )
    print(f"W&B run initialised: project={project}, entity={entity}, name={run_name}")
    return wandb


def train(config: TrainConfig) -> Path:
    _seed_everything(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    wandb = _init_wandb(config) if config.wandb else None

    # Build dataset with rare-class case-level oversampling
    dataset = WoodDefectDataset(
        config.image_dir,
        config.label_dir,
        rare_label_idx=config.rare_label_idx,
        oversample_factor=config.oversample_factor,
    )
    train_cases, val_cases = _split_cases(dataset.samples, config.val_fraction, config.seed)

    train_transform = get_train_transforms(
        config.patch_size,
        config.batch_size,
        num_classes=config.num_classes,
        rare_label_idx=config.rare_label_idx,
    )
    val_transform = get_val_transforms()

    if config.cache_rate > 0:
        print(
            f"Using CacheDataset with cache_rate={config.cache_rate} and "
            f"num_workers={config.num_workers}"
        )
        train_ds = CacheDataset(
            data=train_cases,
            transform=train_transform,
            cache_rate=config.cache_rate,
            num_workers=config.num_workers,
        )
        val_ds = CacheDataset(
            data=val_cases,
            transform=val_transform,
            cache_rate=config.cache_rate,
            num_workers=config.num_workers,
        )
    else:
        print("Using Dataset without cache (cache_rate=0.0) to reduce RAM usage.")
        train_ds = Dataset(data=train_cases, transform=train_transform)
        val_ds = Dataset(data=val_cases, transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_swin_model(
        num_classes=config.num_classes,
        img_size=config.patch_size,
    ).to(device)
    loss_fn = get_loss(
        num_classes=config.num_classes,
        rare_label_idx=config.rare_label_idx,
        rare_class_weight=config.rare_class_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    metric = DiceMetric(include_background=False, reduction="mean_batch")
    post_pred = AsDiscrete(argmax=True, to_onehot=config.num_classes)
    post_label = AsDiscrete(to_onehot=config.num_classes)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")

    best_metric = float("-inf")
    best_metric_epoch = -1
    best_path = output_dir / "best_model.pth"
    last_path = output_dir / "last_model.pth"

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0

        for batch_data in train_loader:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=config.amp and device.type == "cuda"):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        scheduler.step()

        model.eval()
        metric.reset()
        with torch.inference_mode():
            for batch_data in val_loader:
                inputs = batch_data["image"].to(device)
                labels = batch_data["label"].to(device)

                outputs = sliding_window_inference(
                    inputs,
                    roi_size=config.patch_size,
                    sw_batch_size=1,
                    predictor=model,
                    overlap=config.sliding_window_overlap,
                )

                outputs = [post_pred(item) for item in decollate_batch(outputs)]
                labels = [post_label(item) for item in decollate_batch(labels)]
                metric(y_pred=outputs, y=labels)

        dice_scores = metric.aggregate().detach().cpu()
        mean_dice = float(dice_scores.mean().item())
        print(
            f"Epoch {epoch + 1}/{config.epochs} - loss: {epoch_loss / max(1, len(train_loader)):.4f} - "
            f"val_dice: {mean_dice:.4f} - per_class: {dice_scores.tolist()}"
        )

        if wandb is not None and wandb.run is not None:
            metrics = {
                "epoch": epoch + 1,
                "train/loss": epoch_loss / max(1, len(train_loader)),
                "val/mean_dice": mean_dice,
            }
            for class_index, dice_value in enumerate(dice_scores.tolist()):
                metrics[f"val/dice_class_{class_index}"] = float(dice_value)
            wandb.log(metrics, step=epoch + 1)

        torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(), "config": asdict(config)}, last_path)
        if mean_dice > best_metric:
            best_metric = mean_dice
            best_metric_epoch = epoch + 1
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(), "config": asdict(config)}, best_path)

    print(f"Best validation dice: {best_metric:.4f} at epoch {best_metric_epoch}")
    if wandb is not None and wandb.run is not None:
        wandb.run.summary["best/val_dice"] = best_metric
        wandb.run.summary["best/epoch"] = best_metric_epoch
        wandb.finish()
    return best_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the custom 3D SwinUNETR wood-defect model.")
    parser.add_argument("--image-dir", required=True, help="Directory with input NIfTI volumes.")
    parser.add_argument("--label-dir", required=True, help="Directory with label NIfTI volumes.")
    parser.add_argument("--output-dir", default="./output/custom_model")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(128, 384, 128))
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader worker processes (default: 0 for cluster stability).")
    parser.add_argument("--cache-rate", type=float, default=0.0,
                        help="MONAI CacheDataset cache fraction in [0,1]. 0 disables caching (default: 0).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    parser.add_argument("--sliding-window-overlap", type=float, default=0.5)
    parser.add_argument("--rare-label-idx", type=int, default=6,
                        help="Label index to boost (default: 6 = Poškození hmyzem).")
    parser.add_argument("--rare-class-weight", type=float, default=30.0,
                        help="CE loss weight multiplier for the rare class (default: 30).")
    parser.add_argument("--oversample-factor", type=int, default=8,
                        help="Extra case copies per epoch for the rare class (default: 8).")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="bp-custom-model", metavar="PROJECT", help="W&B project name.")
    parser.add_argument("--wandb-entity", default=None, metavar="ENTITY", help="W&B entity / team name.")
    parser.add_argument("--wandb-run-name", default=None, metavar="NAME", help="Display name for this W&B run.")
    return parser


def parse_args(argv=None) -> TrainConfig:
    args = build_arg_parser().parse_args(argv)
    return TrainConfig(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patch_size=tuple(args.patch_size),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_classes=args.num_classes,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        cache_rate=args.cache_rate,
        seed=args.seed,
        amp=not args.no_amp,
        sliding_window_overlap=args.sliding_window_overlap,
        rare_label_idx=args.rare_label_idx,
        rare_class_weight=args.rare_class_weight,
        oversample_factor=args.oversample_factor,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
    )


def main(argv=None) -> int:
    config = parse_args(argv)
    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
