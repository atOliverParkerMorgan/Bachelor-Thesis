from __future__ import annotations

import argparse
import csv
import json
import os
import random
from importlib import import_module
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 1e-4
    early_stopping_min_epochs: int = 50
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


def _write_history_csv(history: list[dict[str, Any]], output_path: Path) -> None:
    if not history:
        return

    fieldnames: list[str] = []
    for row in history:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _save_history(history: list[dict[str, Any]], output_dir: Path) -> None:
    (output_dir / "metrics_history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    _write_history_csv(history, output_dir / "metrics_history.csv")


def _plot_history(history: list[dict[str, Any]], output_dir: Path) -> None:
    if not history:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib is not installed. Skipping training curve plots.")
        return

    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    val_loss = [float(row["val_loss"]) for row in history]
    train_dice = [float(row["train_mean_dice"]) for row in history]
    val_dice = [float(row["val_mean_dice"]) for row in history]
    learning_rate = [float(row["learning_rate"]) for row in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, train_loss, label="train_loss", color="tab:blue")
    axes[0].plot(epochs, val_loss, label="val_loss", color="tab:orange")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_dice, label="train_mean_dice", color="tab:green")
    axes[1].plot(epochs, val_dice, label="val_mean_dice", color="tab:red")
    axes[1].set_title("Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, learning_rate, label="learning_rate", color="tab:purple")
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle("Custom Model Training Curves")
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(fig)


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
    val_loss_fn = get_loss(
        num_classes=config.num_classes,
        rare_label_idx=config.rare_label_idx,
        rare_class_weight=config.rare_class_weight,
    ).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    metric = DiceMetric(include_background=False, reduction="mean_batch")
    post_pred = AsDiscrete(argmax=True, to_onehot=config.num_classes)
    post_label = AsDiscrete(to_onehot=config.num_classes)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")

    best_metric = float("-inf")
    best_metric_epoch = -1
    no_improve_epochs = 0
    early_stopped = False
    stopped_epoch = None
    best_path = output_dir / "best_model.pth"
    last_path = output_dir / "last_model.pth"
    history: list[dict[str, Any]] = []
    train_metric = DiceMetric(include_background=False, reduction="mean_batch")

    status = "finished"
    error_message = None

    try:
        for epoch in range(config.epochs):
            model.train()
            train_metric.reset()
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

                train_outputs = [post_pred(item) for item in decollate_batch(outputs.detach())]
                train_labels = [post_label(item) for item in decollate_batch(labels.detach())]
                train_metric(y_pred=train_outputs, y=train_labels)

            scheduler.step()

            model.eval()
            metric.reset()
            val_loss = 0.0
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

                    val_loss += float(val_loss_fn(outputs.detach().cpu(), labels.detach().cpu()).item())
                    outputs = [post_pred(item) for item in decollate_batch(outputs)]
                    labels = [post_label(item) for item in decollate_batch(labels)]
                    metric(y_pred=outputs, y=labels)

            train_dice_scores = train_metric.aggregate().detach().cpu()
            train_mean_dice = float(train_dice_scores.mean().item())
            dice_scores = metric.aggregate().detach().cpu()
            mean_dice = float(dice_scores.mean().item())
            avg_train_loss = epoch_loss / max(1, len(train_loader))
            avg_val_loss = val_loss / max(1, len(val_loader))
            learning_rate = float(optimizer.param_groups[0]["lr"])

            print(
                f"Epoch {epoch + 1}/{config.epochs} - train_loss: {avg_train_loss:.4f} - "
                f"val_loss: {avg_val_loss:.4f} - train_dice: {train_mean_dice:.4f} - "
                f"val_dice: {mean_dice:.4f} - val_per_class: {dice_scores.tolist()}"
            )

            metrics_row: dict[str, Any] = {
                "epoch": epoch + 1,
                "learning_rate": learning_rate,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_mean_dice": train_mean_dice,
                "val_mean_dice": mean_dice,
            }
            for class_index, train_dice_value in enumerate(train_dice_scores.tolist()):
                metrics_row[f"train_dice_class_{class_index}"] = float(train_dice_value)
            for class_index, val_dice_value in enumerate(dice_scores.tolist()):
                metrics_row[f"val_dice_class_{class_index}"] = float(val_dice_value)
            history.append(metrics_row)

            _save_history(history, output_dir)
            _plot_history(history, output_dir)

            if wandb is not None and wandb.run is not None:
                wandb_metrics = {
                    "epoch": epoch + 1,
                    "train/loss": avg_train_loss,
                    "val/loss": avg_val_loss,
                    "train/mean_dice": train_mean_dice,
                    "val/mean_dice": mean_dice,
                    "train/lr": learning_rate,
                }
                for class_index, train_dice_value in enumerate(train_dice_scores.tolist()):
                    wandb_metrics[f"train/dice_class_{class_index}"] = float(train_dice_value)
                for class_index, val_dice_value in enumerate(dice_scores.tolist()):
                    wandb_metrics[f"val/dice_class_{class_index}"] = float(val_dice_value)
                wandb.log(wandb_metrics, step=epoch + 1)

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "config": asdict(config),
                    "history": history,
                },
                last_path,
            )
            if mean_dice > (best_metric + config.early_stopping_min_delta):
                best_metric = mean_dice
                best_metric_epoch = epoch + 1
                no_improve_epochs = 0
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "config": asdict(config),
                        "history": history,
                    },
                    best_path,
                )
            else:
                no_improve_epochs += 1

            if (
                config.early_stopping_patience > 0
                and (epoch + 1) >= config.early_stopping_min_epochs
                and no_improve_epochs >= config.early_stopping_patience
            ):
                early_stopped = True
                stopped_epoch = epoch + 1
                print(
                    "Early stopping triggered at epoch "
                    f"{stopped_epoch}: no val_dice improvement greater than "
                    f"{config.early_stopping_min_delta} for {no_improve_epochs} epoch(s)."
                )
                break
    except Exception as exc:
        status = "failed"
        error_message = str(exc)
        raise
    finally:
        run_summary = {
            "status": status,
            "best_val_dice": best_metric,
            "best_epoch": best_metric_epoch,
            "completed_epochs": len(history),
            "early_stopped": early_stopped,
            "stopped_epoch": stopped_epoch,
            "no_improve_epochs": no_improve_epochs,
            "error": error_message,
            "artifacts": {
                "config": "config.json",
                "history_json": "metrics_history.json",
                "history_csv": "metrics_history.csv",
                "curves_png": "training_curves.png",
                "last_checkpoint": "last_model.pth",
                "best_checkpoint": "best_model.pth",
            },
        }
        (output_dir / "run_summary.json").write_text(
            json.dumps(run_summary, indent=2),
            encoding="utf-8",
        )

        if wandb is not None and wandb.run is not None:
            wandb.run.summary["status"] = status
            if best_metric_epoch > 0:
                wandb.run.summary["best/val_dice"] = best_metric
                wandb.run.summary["best/epoch"] = best_metric_epoch
            wandb.run.summary["early_stopped"] = early_stopped
            if stopped_epoch is not None:
                wandb.run.summary["stopped_epoch"] = stopped_epoch
            if error_message:
                wandb.run.summary["error"] = error_message
            wandb.finish()

    print(f"Best validation dice: {best_metric:.4f} at epoch {best_metric_epoch}")
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
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=50,
        help="Stop if val_dice does not improve for this many epochs (0 disables early stopping).",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Minimum val_dice increase to count as an improvement.",
    )
    parser.add_argument(
        "--early-stopping-min-epochs",
        type=int,
        default=50,
        help="Do not early-stop before this epoch count.",
    )
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
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_min_epochs=args.early_stopping_min_epochs,
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
