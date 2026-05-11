from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict

import numpy as np

from .env import log
from .nnunet import build_holdout_split
from .evaluate import compute_nnunet_summary


def run_custom_train(args, env: Dict[str, str]) -> None:
    raw_root = args.nnunet_root / "nnUNet_raw" / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    image_dir = args.image_dir or (raw_root / "imagesTr")
    label_dir = args.label_dir or (raw_root / "labelsTr")

    if getattr(args, "test", False):
        _, _, split_path = build_holdout_split(
            nnunet_root=args.nnunet_root,
            dataset_id=args.dataset_id,
            dataset_name=args.dataset_name,
            test_tree=getattr(args, "test_tree", "dub_2"),
        )
        args.fold = 0
        args.splits_json = split_path
        log(f"Test mode: fold=0, splits_json={split_path}")

    cmd = [
        sys.executable, "-m", "src.implementation.custom_model.train",
        "--image-dir", str(image_dir),
        "--label-dir", str(label_dir),
        "--output-dir", str(args.output_dir),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--patch-size", str(args.patch_size[0]), str(args.patch_size[1]), str(args.patch_size[2]),
        "--learning-rate", str(args.learning_rate),
        "--weight-decay", str(args.weight_decay),
        "--num-classes", str(args.num_classes),
        "--val-fraction", str(args.val_fraction),
        "--num-workers", str(args.num_workers),
        "--seed", str(args.seed),
        "--sliding-window-overlap", str(args.sliding_window_overlap),
        "--rare-label-idx", str(args.rare_label_idx),
        "--rare-class-weight", str(args.rare_class_weight),
        "--oversample-factor", str(args.oversample_factor),
        "--early-stopping-patience", str(args.early_stopping_patience),
        "--early-stopping-min-delta", str(args.early_stopping_min_delta),
        "--early-stopping-min-epochs", str(args.early_stopping_min_epochs),
        "--grad-accumulation-steps", str(args.grad_accumulation_steps),
        "--warmup-epochs", str(args.warmup_epochs),
        "--max-grad-norm", str(args.max_grad_norm),
        "--dropout-path-rate", str(args.dropout_path_rate),
        "--model-name", str(args.model_name),
        "--model-feature-size", str(args.model_feature_size),
        "--unetr-hidden-size", str(args.unetr_hidden_size),
        "--unetr-mlp-dim", str(args.unetr_mlp_dim),
        "--unetr-num-heads", str(args.unetr_num_heads),
        "--basicunet-features", *[str(v) for v in args.basicunet_features],
        "--cache-rate", str(args.cache_rate),
        "--normalization", str(args.normalization),
        "--norm-clip-min", str(args.norm_clip_min),
        "--norm-clip-max", str(args.norm_clip_max),
    ]
    if getattr(args, "resume_checkpoint", None) is not None:
        cmd.extend(["--resume-checkpoint", str(Path(args.resume_checkpoint).resolve())])
    if getattr(args, "no_amp", False):
        cmd.append("--no-amp")
    if getattr(args, "pretrained_weights", None) is not None:
        cmd.extend(["--pretrained-weights", str(Path(args.pretrained_weights).resolve())])
    if getattr(args, "loss_type", None) is not None:
        cmd.extend(["--loss-type", str(args.loss_type)])
    if getattr(args, "wandb", False):
        cmd.append("--wandb")
        cmd.extend(["--wandb-project", str(args.wandb_project)])
        if getattr(args, "wandb_entity", None):
            cmd.extend(["--wandb-entity", str(args.wandb_entity)])
        if getattr(args, "wandb_run_name", None):
            cmd.extend(["--wandb-run-name", str(args.wandb_run_name)])
    if getattr(args, "debug_data", False):
        cmd.append("--debug-data")
    if getattr(args, "fold", None) is not None:
        cmd.extend(["--fold", str(args.fold), "--splits-json", str(args.splits_json)])
    if getattr(args, "test", False):
        cmd.extend(["--test-mode", "--test-tree", str(getattr(args, "test_tree", "dub_2"))])

    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError:
        log("Failed: custom-train")
        raise


def run_custom_predict(args, env: Dict[str, str]) -> None:
    import nibabel as nib
    from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation
    import torch
    from monai.inferers import sliding_window_inference
    from src.implementation.custom_model.model import get_model
    from src.implementation.custom_model.transforms import get_inference_transforms

    model_dir = Path(args.model_dir)
    with open(model_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_name = cfg.get("model_name", "swinunetr")
    num_classes = cfg.get("num_classes", 7)
    patch_size = tuple(cfg.get("patch_size", [128, 384, 128]))
    log(f"Model: {model_name}, classes={num_classes}, patch_size={patch_size}")

    model = get_model(
        model_name=model_name,
        num_classes=num_classes,
        img_size=patch_size,
        dropout_path_rate=cfg.get("dropout_path_rate", 0.1),
        feature_size=cfg.get("model_feature_size", 48),
    )
    checkpoint = torch.load(str(model_dir / "best_model.pth"), map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    log(f"Device: {device}")

    transforms = get_inference_transforms(
        normalization=cfg.get("normalization", "zscore"),
        clip_min=cfg.get("norm_clip_min", -1000.0),
        clip_max=cfg.get("norm_clip_max", 500.0),
        zscore_mean=cfg.get("normalization_mean"),
        zscore_std=cfg.get("normalization_std"),
    )

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*_0000.nii.gz")) or sorted(input_dir.glob("*.nii.gz"))
    if not input_files:
        raise RuntimeError(f"No NIfTI input files found in {input_dir}")
    log(f"Found {len(input_files)} input file(s)")

    sliding_window_overlap = cfg.get("sliding_window_overlap", 0.5)

    for nii_file in input_files:
        stem = nii_file.name[:-7] if nii_file.name.endswith(".nii.gz") else nii_file.stem
        case_name = re.sub(r"_0000(?:_\d+)?$", "", stem)
        log(f"Predicting: {nii_file.name} → {case_name}.nii.gz")

        inputs = transforms({"image": str(nii_file)})["image"].unsqueeze(0).to(device)
        with torch.inference_mode():
            outputs = sliding_window_inference(
                inputs, roi_size=patch_size, sw_batch_size=1,
                predictor=model, overlap=sliding_window_overlap,
            )

        pred_ras = outputs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Inference ran in RAS; reorient back to source orientation for voxel alignment.
        orig_nib = nib.load(str(nii_file))
        orig_ornt = io_orientation(orig_nib.affine)
        pred = apply_orientation(pred_ras, ornt_transform(axcodes2ornt(("R", "A", "S")), orig_ornt)).astype(np.uint8)

        nib.save(nib.Nifti1Image(pred, orig_nib.affine), str(output_dir / f"{case_name}.nii.gz"))
        log(f"Saved: {case_name}.nii.gz")
