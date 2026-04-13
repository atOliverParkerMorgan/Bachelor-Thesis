from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete

from src.custom_model.dataset import WoodDefectDataset
from src.custom_model.losses import get_loss
from src.custom_model.model import get_model
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
    num_workers: int = 4
    seed: int = 42
    amp: bool = True
    sliding_window_overlap: float = 0.5


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


def train(config: TrainConfig) -> Path:
    _seed_everything(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    dataset = WoodDefectDataset(config.image_dir, config.label_dir)
    train_cases, val_cases = _split_cases(dataset.samples, config.val_fraction, config.seed)

    train_ds = CacheDataset(
        data=train_cases,
        transform=get_train_transforms(config.patch_size, config.batch_size),
        cache_rate=1.0,
        num_workers=config.num_workers,
    )
    val_ds = CacheDataset(
        data=val_cases,
        transform=get_val_transforms(),
        cache_rate=1.0,
        num_workers=config.num_workers,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=config.num_classes).to(device)
    loss_fn = get_loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    metric = DiceMetric(include_background=False, reduction="mean_batch")
    post_pred = AsDiscrete(argmax=True, to_onehot=config.num_classes)
    post_label = AsDiscrete(to_onehot=config.num_classes)
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp and device.type == "cuda")

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
            with torch.cuda.amp.autocast(enabled=config.amp and device.type == "cuda"):
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

        torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(), "config": asdict(config)}, last_path)
        if mean_dice > best_metric:
            best_metric = mean_dice
            best_metric_epoch = epoch + 1
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(), "config": asdict(config)}, best_path)

    print(f"Best validation dice: {best_metric:.4f} at epoch {best_metric_epoch}")
    return best_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the custom 3D wood-defect model.")
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
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    parser.add_argument("--sliding-window-overlap", type=float, default=0.5)
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
        seed=args.seed,
        amp=not args.no_amp,
        sliding_window_overlap=args.sliding_window_overlap,
    )


def main(argv=None) -> int:
    config = parse_args(argv)
    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
