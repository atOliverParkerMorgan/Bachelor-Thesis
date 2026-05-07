#!/usr/bin/env python3
"""Minimal command wrappers for nnU-Net v2 on Dataset001."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PLANNER = "nnUNetPlannerResEncL"
PLANNER_FOR_PRESET = {
    "M": "nnUNetPlannerResEncM",
    "L": "nnUNetPlannerResEncL",
    "XL": "nnUNetPlannerResEncXL",
}
DEFAULT_PLANS_FOR_PLANNER = {
    "ExperimentPlanner": "nnUNetPlans",
    "nnUNetPlannerResEncM": "nnUNetResEncUNetMPlans",
    "nnUNetPlannerResEncL": "nnUNetResEncUNetLPlans",
    "nnUNetPlannerResEncXL": "nnUNetResEncUNetXLPlans",
}
RESENC_PLAN_NAMES = [
    "nnUNetResEncUNetLPlans",
    "nnUNetResEncUNetMPlans",
    "nnUNetResEncUNetXLPlans",
]
CHECKPOINT_CANDIDATES = [
    "checkpoint_final.pth",
    "checkpoint_best.pth",
    "checkpoint_latest.pth",
]
DEFAULT_NNUNET_ROOT = Path("src/nn_UNet/nnunet_data")

def import_clusterfit_helpers():
    """Safely import cluster utilities, allowing local Windows runs without crashing."""
    try:
        from src.nn_UNet.clusterfit_utils import (
            SlurmJobSubmitter,
            add_clusterfit_arguments,
            build_slurm_config_from_args,
        )
        return SlurmJobSubmitter, add_clusterfit_arguments, build_slurm_config_from_args
    except ImportError:
        def dummy_add(*args, **kwargs): pass
        return None, dummy_add, None


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}")


def ensure_env(nnunet_root: Path) -> Dict[str, str]:
    env = os.environ.copy()
    raw = nnunet_root / "nnUNet_raw"
    preprocessed = nnunet_root / "nnUNet_preprocessed"
    results = nnunet_root / "nnUNet_results"
    project_root = Path(__file__).resolve().parents[2]

    raw.mkdir(parents=True, exist_ok=True)
    preprocessed.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    env["nnUNet_raw"] = str(raw.resolve())
    env["nnUNet_preprocessed"] = str(preprocessed.resolve())
    env["nnUNet_results"] = str(results.resolve())

    existing_pythonpath = env.get("PYTHONPATH", "")
    project_root_str = str(project_root)
    if existing_pythonpath:
        parts = existing_pythonpath.split(os.pathsep)
        if project_root_str not in parts:
            env["PYTHONPATH"] = os.pathsep.join([project_root_str, existing_pythonpath])
    else:
        env["PYTHONPATH"] = project_root_str

    return env


def apply_runtime_env_overrides(env: Dict[str, str], args: argparse.Namespace) -> None:
    env["PYTHONUNBUFFERED"] = "1"

    # Reduces CUDA allocator fragmentation for large 3D patch training
    if getattr(args, "command", None) == "custom-train":
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    save_every = getattr(args, "save_every", None)
    if save_every is not None:
        env["NNUNET_SAVE_EVERY"] = str(save_every)

    initial_lr = getattr(args, "initial_lr", None)
    if initial_lr is not None:
        env["NNUNET_INITIAL_LR"] = str(initial_lr)

    optimizer = getattr(args, "optimizer", None)
    if optimizer is not None:
        env["NNUNET_OPTIMIZER"] = optimizer

    weight_decay = getattr(args, "weight_decay", None)
    if weight_decay is not None:
        env["NNUNET_WEIGHT_DECAY"] = str(weight_decay)

    if getattr(args, "skip_arch_plot", False):
        env["NNUNET_SKIP_ARCH_PLOT"] = "1"

    compile_mode = getattr(args, "compile", None)
    if compile_mode == "off":
        env["nnUNet_compile"] = "0"
    elif compile_mode == "on":
        env["nnUNet_compile"] = "1"

    pretrained_weights = getattr(args, "pretrained_weights", None)
    if pretrained_weights is not None:
        env["NNUNET_PRETRAINED_WEIGHTS"] = str(Path(pretrained_weights).resolve())

    wandb_project = getattr(args, "wandb_project", None)
    if wandb_project:
        env["WANDB_PROJECT"] = wandb_project
    wandb_entity = getattr(args, "wandb_entity", None)
    if wandb_entity:
        env["WANDB_ENTITY"] = wandb_entity
    wandb_run_name = getattr(args, "wandb_run_name", None)
    if wandb_run_name:
        env["WANDB_RUN_NAME"] = wandb_run_name
    wandb_api_key = getattr(args, "wandb_api_key", None)
    if wandb_api_key:
        env["WANDB_API_KEY"] = wandb_api_key

    n_proc_da = getattr(args, "n_proc_da", None)
    if n_proc_da is not None:
        env["nnUNet_n_proc_DA"] = str(n_proc_da)

    cpu_threads = getattr(args, "cpu_threads", None)
    if cpu_threads is not None:
        thread_count = str(cpu_threads)
        env["OMP_NUM_THREADS"] = thread_count
        env["MKL_NUM_THREADS"] = thread_count
        env["OPENBLAS_NUM_THREADS"] = thread_count
        env["NUMEXPR_NUM_THREADS"] = thread_count


def run_cmd(command: List[str], env: Dict[str, str], label: str) -> None:
    if shutil.which(command[0]) is None:
        raise RuntimeError(
            f"Required executable not found: {command[0]}. Install nnunetv2 and run through poetry."
        )

    print("$", " ".join(command))
    try:
        subprocess.run(command, check=True, env=env)
    except subprocess.CalledProcessError:
        log(f"Failed {label}")
        raise


def planner_from_args(args: argparse.Namespace) -> str:
    if getattr(args, "planner", None):
        return args.planner
    preset = getattr(args, "resenc_preset", None)
    if preset:
        if preset not in PLANNER_FOR_PRESET:
            raise ValueError(f"Unsupported ResEnc preset: {preset}")
        return PLANNER_FOR_PRESET[preset]
    return DEFAULT_PLANNER


def default_plans_for_planner(planner: str) -> str:
    return DEFAULT_PLANS_FOR_PLANNER.get(planner, "nnUNetPlans")


def available_plans_identifiers(nnunet_root: Path, dataset_id: int, dataset_name: str) -> List[str]:
    dataset_dir = nnunet_root / "nnUNet_preprocessed" / f"Dataset{dataset_id:03d}_{dataset_name}"
    if not dataset_dir.exists():
        return []

    return sorted({path.stem for path in dataset_dir.glob("*Plans.json") if path.is_file()})


def resolve_plans_identifier(args: argparse.Namespace) -> str:
    explicit = getattr(args, "plans_identifier", None)
    if explicit:
        return explicit

    planner = planner_from_args(args)
    preferred = default_plans_for_planner(planner)
    available = available_plans_identifiers(args.nnunet_root, args.dataset_id, args.dataset_name)

    if preferred in available:
        return preferred

    for candidate in RESENC_PLAN_NAMES:
        if candidate in available:
            log(f"Using detected plans identifier '{candidate}' (preferred '{preferred}' not found).")
            return candidate

    if "nnUNetPlans" in available:
        if preferred != "nnUNetPlans":
            log(f"Using detected legacy plans 'nnUNetPlans' because preferred '{preferred}' is missing.")
        return "nnUNetPlans"

    if available:
        log(f"Using detected plans identifier '{available[0]}' (preferred '{preferred}' not found).")
        return available[0]

    return preferred


def prepared_dataset_root(nnunet_root: Path, dataset_id: int, dataset_name: str) -> Path:
    return nnunet_root / "nnUNet_raw" / f"Dataset{dataset_id:03d}_{dataset_name}"


def has_prepared_dataset(nnunet_root: Path, dataset_id: int, dataset_name: str) -> bool:
    dataset_root = prepared_dataset_root(nnunet_root, dataset_id, dataset_name)
    images_tr = dataset_root / "imagesTr"
    labels_tr = dataset_root / "labelsTr"
    dataset_json = dataset_root / "dataset.json"

    return (
        dataset_json.exists()
        and images_tr.exists()
        and labels_tr.exists()
        and any(images_tr.glob("*_0000.nii.gz"))
        and any(labels_tr.glob("*.nii.gz"))
    )


def ensure_crossval_splits(
    nnunet_root: Path,
    dataset_id: int,
    dataset_name: str,
    configuration: str,
    plans_identifier: str,
) -> None:
    dataset_dir = nnunet_root / "nnUNet_preprocessed" / f"Dataset{dataset_id:03d}_{dataset_name}"
    splits_file = dataset_dir / "splits_final.json"
    if splits_file.exists():
        return

    config_dir = dataset_dir / f"{plans_identifier}_{configuration}"
    if not config_dir.exists():
        fallback_dir = dataset_dir / f"nnUNetPlans_{configuration}"
        if not fallback_dir.exists():
            return
        config_dir = fallback_dir

    case_ids = sorted({path.stem for path in config_dir.glob("*.b2nd") if not path.stem.endswith("_seg")})
    if not case_ids:
        case_ids = sorted({path.stem for path in config_dir.glob("*.npz") if not path.stem.endswith("_seg")})

    num_cases = len(case_ids)
    if num_cases == 0 or num_cases >= 5:
        return
    if num_cases < 2:
        raise RuntimeError(f"Training requires at least 2 cases. Found only {num_cases} case in {config_dir}.")

    splits = []
    for idx in range(num_cases):
        val = [case_ids[idx]]
        train = [case_id for case_id in case_ids if case_id not in val]
        splits.append({"train": train, "val": val})

    with open(splits_file, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    log(f"Created splits_final.json with {num_cases} folds for tiny dataset.")


def build_plan_command(args: argparse.Namespace, planner: str, plans_identifier: str) -> List[str]:
    cmd = ["nnUNetv2_plan_and_preprocess", "-d", str(args.dataset_id), "-pl", planner]
    if getattr(args, "verify_dataset_integrity", False):
        cmd.append("--verify_dataset_integrity")
    if plans_identifier != "nnUNetPlans":
        cmd.extend(["-overwrite_plans_name", plans_identifier])

    configs = args.configurations if args.command == "plan" else args.plan_configurations
    num_processes = args.num_processes if args.command == "plan" else args.plan_num_processes
    if configs:
        cmd.extend(["-c", *configs])
    if num_processes is not None:
        cmd.extend(["-np", str(num_processes)])

    return cmd


def resolve_train_configuration(args: argparse.Namespace) -> str:
    if args.command != "all":
        return args.configuration
    if args.configuration:
        return args.configuration
    if args.plan_configurations and len(args.plan_configurations) == 1:
        return args.plan_configurations[0]
    return "3d_fullres"


def _fit_int_list(values: List[int], target_len: int) -> List[int]:
    if target_len <= 0:
        return []
    if not values:
        return [1] * target_len
    if len(values) == target_len:
        return values
    if len(values) > target_len:
        return values[:target_len]
    return values + [values[-1]] * (target_len - len(values))


def _decoder_len_for_stages(n_stages: int) -> int:
    return max(1, n_stages - 1)


def apply_plan_regularization_overrides(
    args: argparse.Namespace,
    plans_identifier: str,
    configuration: str,
) -> None:
    regularize = getattr(args, "regularize_arch", False)
    patch_size = getattr(args, "patch_size", None)
    model_batch_size = getattr(args, "model_batch_size", None)

    if not regularize and patch_size is None and model_batch_size is None:
        return

    plans_file = (
        args.nnunet_root
        / "nnUNet_preprocessed"
        / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
        / f"{plans_identifier}.json"
    )
    if not plans_file.exists():
        log(
            f"Plan override skipped: plans file not found at {plans_file}. "
            "Run plan/preprocess first."
        )
        return

    with open(plans_file, "r", encoding="utf-8") as f:
        plans = json.load(f)

    cfg = plans.get("configurations", {}).get(configuration)
    if not isinstance(cfg, dict):
        log(
            f"Plan override skipped: configuration '{configuration}' not found in {plans_file.name}."
        )
        return

    if regularize:
        arch = cfg.get("architecture", {})
        arch_kwargs = arch.get("arch_kwargs")
        if not isinstance(arch_kwargs, dict):
            log("Plan override skipped: architecture arch_kwargs missing in plans JSON.")
            return

        n_stages = int(getattr(args, "model_n_stages", 6) or 6)
        user_features = getattr(args, "model_features", None)
        if user_features:
            features = [int(v) for v in user_features]
        else:
            features = [32, 64, 128, 256, 256, 256] if n_stages == 6 else [32, 64, 128, 256, 256]
        features = _fit_int_list(features, n_stages)

        arch_kwargs["n_stages"] = n_stages
        arch_kwargs["features_per_stage"] = features
        arch_kwargs["dropout_op"] = "torch.nn.Dropout3d"
        arch_kwargs["dropout_op_kwargs"] = {
            "p": float(getattr(args, "model_dropout_p", 0.2)),
            "inplace": True,
        }

        for key in ("kernel_sizes", "strides", "n_blocks_per_stage"):
            if isinstance(arch_kwargs.get(key), list):
                arch_kwargs[key] = _fit_int_list(arch_kwargs[key], n_stages)

        decoder_key = "n_conv_per_stage_decoder"
        if isinstance(arch_kwargs.get(decoder_key), list):
            arch_kwargs[decoder_key] = _fit_int_list(
                arch_kwargs[decoder_key], _decoder_len_for_stages(n_stages)
            )

    if patch_size is not None:
        cfg["patch_size"] = [int(v) for v in patch_size]

    if model_batch_size is not None:
        cfg["batch_size"] = int(model_batch_size)

    with open(plans_file, "w", encoding="utf-8") as f:
        json.dump(plans, f, indent=2)

    override_parts = []
    if regularize:
        override_parts += [
            f"dropout={cfg.get('architecture', {}).get('arch_kwargs', {}).get('dropout_op_kwargs', {}).get('p')}",
            f"n_stages={getattr(args, 'model_n_stages', 6)}",
            f"features={cfg.get('architecture', {}).get('arch_kwargs', {}).get('features_per_stage')}",
        ]
    if patch_size is not None:
        override_parts.append(f"patch_size={cfg['patch_size']}")
    if model_batch_size is not None:
        override_parts.append(f"batch_size={cfg.get('batch_size')}")
    log("Applied plans override for training: " + ", ".join(override_parts))


def model_output_dir(
    nnunet_root: Path,
    dataset_id: int,
    dataset_name: str,
    configuration: str,
    plans_identifier: str,
    trainer: str = "nnUNetTrainer",
) -> Path:
    dataset_dirname = f"Dataset{dataset_id:03d}_{dataset_name}"
    model_dirname = f"{trainer}__{plans_identifier}__{configuration}"
    return nnunet_root / "nnUNet_results" / dataset_dirname / model_dirname


def available_prediction_trainers(
    nnunet_root: Path,
    dataset_id: int,
    dataset_name: str,
    configuration: str,
    plans_identifier: str,
) -> List[str]:
    dataset_dir = nnunet_root / "nnUNet_results" / f"Dataset{dataset_id:03d}_{dataset_name}"
    if not dataset_dir.exists():
        return []

    suffix = f"__{plans_identifier}__{configuration}"
    trainers: List[str] = []
    for candidate in dataset_dir.glob(f"*{suffix}"):
        if candidate.is_dir() and candidate.name.endswith(suffix):
            trainers.append(candidate.name[: -len(suffix)])
    return sorted(set(trainers))


def resolve_prediction_trainer(
    nnunet_root: Path,
    dataset_id: int,
    dataset_name: str,
    configuration: str,
    plans_identifier: str,
    requested_trainer: str | None,
) -> str:
    if requested_trainer:
        return requested_trainer

    available = available_prediction_trainers(
        nnunet_root=nnunet_root,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        configuration=configuration,
        plans_identifier=plans_identifier,
    )
    if len(available) == 1:
        trainer = available[0]
        log(f"Auto-detected trainer from results: {trainer}")
        return trainer

    if len(available) > 1:
        for preferred in (
            "nnUNetTrainer",
            "nnUNetTrainerRareClassBoostWandb",
            "nnUNetTrainerLungPretrainedWandb",
            "nnUNetTrainerLungPretrained",
        ):
            if preferred in available:
                log(f"Auto-selected trainer from available models: {preferred}")
                return preferred
        raise RuntimeError(
            "Multiple model trainers found for this configuration. "
            f"Please pass --trainer explicitly. Available: {', '.join(available)}"
        )

    # Keep legacy fallback for fresh environments without local results.
    return "nnUNetTrainerRareClassBoostWandb"


def prediction_folds(fold_value: str, model_dir: Path) -> List[str]:
    if fold_value == "all":
        return sorted(path.name for path in model_dir.glob("fold_*") if path.is_dir())
    return [f"fold_{fold_value}"]


def resolve_prediction_checkpoint(
    nnunet_root: Path,
    dataset_id: int,
    dataset_name: str,
    configuration: str,
    plans_identifier: str,
    fold: str,
    trainer: str = "nnUNetTrainer",
) -> str | None:
    model_dir = model_output_dir(
        nnunet_root=nnunet_root,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        configuration=configuration,
        plans_identifier=plans_identifier,
        trainer=trainer,
    )
    folds = prediction_folds(fold, model_dir)
    if not folds:
        return None

    for checkpoint_name in CHECKPOINT_CANDIDATES:
        if all((model_dir / fold_name / checkpoint_name).exists() for fold_name in folds):
            return checkpoint_name
    return None


def detect_gpu_vram_gb() -> float | None:
    if shutil.which("nvidia-smi") is None:
        return None

    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=3,
        )
        values = [line.strip() for line in output.splitlines() if line.strip()]
        if not values:
            return None
        memory_mib = max(int(value) for value in values)
        return memory_mib / 1024.0
    except Exception:
        return None


def _normalize_case_id_from_filename(name: str) -> str:
    if name.endswith(".nii.gz"):
        stem = name[:-7]
    else:
        stem = Path(name).stem
    return re.sub(r"_0000(?:_\d+)?$", "", stem)


def _labels_by_id_from_dataset_json(dataset_json_path: Path) -> dict[int, str]:
    with dataset_json_path.open("r", encoding="utf-8") as f:
        dataset_json = json.load(f)

    labels = dataset_json.get("labels", {})
    labels_by_id: dict[int, str] = {}
    for key, value in labels.items():
        if isinstance(value, (int, str)):
            try:
                labels_by_id[int(value)] = str(key)
                continue
            except (TypeError, ValueError):
                pass
        try:
            labels_by_id[int(key)] = str(value)
        except (TypeError, ValueError):
            continue

    if not labels_by_id:
        raise RuntimeError(f"Failed to parse labels from dataset JSON: {dataset_json_path}")
    return dict(sorted(labels_by_id.items()))


def evaluate_prediction_folder(
    prediction_dir: Path,
    labels_dir: Path,
    dataset_json_path: Path,
    metrics_json_path: Path | None = None,
) -> dict:
    """Evaluate predicted labels against references using nnU-Net-style Dice and IoU.

    Dice: 2TP / (2TP + FP + FN)
    IoU : TP / (TP + FP + FN)
    """
    if not prediction_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {prediction_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    labels_by_id = _labels_by_id_from_dataset_json(dataset_json_path)
    class_ids = sorted(labels_by_id.keys())

    gt_files = sorted(labels_dir.glob("*.nii.gz"))
    if not gt_files:
        raise RuntimeError(f"No ground-truth NIfTI files found in {labels_dir}")

    per_class_counts: dict[int, dict[str, int]] = {
        c: {"tp": 0, "fp": 0, "fn": 0} for c in class_ids
    }
    matched_cases = 0

    for gt_file in gt_files:
        case_id = _normalize_case_id_from_filename(gt_file.name)
        pred_file = prediction_dir / f"{case_id}.nii.gz"
        if not pred_file.exists():
            raise FileNotFoundError(
                f"Missing prediction for case '{case_id}': expected {pred_file}"
            )

        gt_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_file))).astype(np.int16)
        pred_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_file))).astype(np.int16)
        if gt_arr.shape != pred_arr.shape:
            raise RuntimeError(
                f"Shape mismatch for {case_id}: gt={gt_arr.shape}, pred={pred_arr.shape}"
            )

        matched_cases += 1
        for class_id in class_ids:
            gt_mask = gt_arr == class_id
            pred_mask = pred_arr == class_id
            tp = int(np.logical_and(gt_mask, pred_mask).sum())
            fp = int(np.logical_and(~gt_mask, pred_mask).sum())
            fn = int(np.logical_and(gt_mask, ~pred_mask).sum())
            per_class_counts[class_id]["tp"] += tp
            per_class_counts[class_id]["fp"] += fp
            per_class_counts[class_id]["fn"] += fn

    per_class_metrics: list[dict] = []
    for class_id in class_ids:
        counts = per_class_counts[class_id]
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        dice_den = 2 * tp + fp + fn
        iou_den = tp + fp + fn
        dice = (2.0 * tp / dice_den) if dice_den > 0 else None
        iou = (tp / iou_den) if iou_den > 0 else None
        per_class_metrics.append(
            {
                "class_id": class_id,
                "class_name": labels_by_id[class_id],
                "dice": dice,
                "iou": iou,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

    foreground = [m for m in per_class_metrics if m["class_id"] != 0]
    all_dice_values = [m["dice"] for m in per_class_metrics if m["dice"] is not None]
    all_iou_values = [m["iou"] for m in per_class_metrics if m["iou"] is not None]
    fg_dice_values = [m["dice"] for m in foreground if m["dice"] is not None]
    fg_iou_values = [m["iou"] for m in foreground if m["iou"] is not None]

    overall = {
        "mean_dice_all_classes": float(np.mean(all_dice_values)) if all_dice_values else None,
        "miou_all_classes": float(np.mean(all_iou_values)) if all_iou_values else None,
        "mean_dice_foreground": float(np.mean(fg_dice_values)) if fg_dice_values else None,
        "miou_foreground": float(np.mean(fg_iou_values)) if fg_iou_values else None,
        "evaluated_cases": matched_cases,
    }

    result = {
        "prediction_dir": str(prediction_dir),
        "labels_dir": str(labels_dir),
        "dataset_json": str(dataset_json_path),
        "overall": overall,
        "per_class": per_class_metrics,
    }

    log(
        "Evaluation summary: "
        f"mean_dice_all={overall['mean_dice_all_classes']}, "
        f"miou_all={overall['miou_all_classes']}, "
        f"mean_fg_dice={overall['mean_dice_foreground']}, "
        f"miou_fg={overall['miou_foreground']}, cases={matched_cases}"
    )
    for metric in per_class_metrics:
        log(
            f"  class {metric['class_id']} ({metric['class_name']}): "
            f"dice={metric['dice']}, iou={metric['iou']}"
        )

    if metrics_json_path is not None:
        metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_json_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        log(f"Saved evaluation metrics to {metrics_json_path}")

    return result


def _install_trainer_file(filename: str) -> bool:
    """Copy a custom trainer file into nnunetv2's trainer variants directory."""
    try:
        import nnunetv2
    except ImportError:
        log("Warning: nnunetv2 not importable — cannot install custom trainer.")
        return False

    trainer_source = Path(__file__).parent / filename
    if not trainer_source.exists():
        raise FileNotFoundError(f"Custom trainer source not found: {trainer_source}")

    variants_dir = Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer" / "variants"
    variants_dir.mkdir(exist_ok=True)
    dest = variants_dir / trainer_source.name

    if not dest.exists() or dest.stat().st_mtime < trainer_source.stat().st_mtime:
        shutil.copy2(trainer_source, dest)
        log(f"Installed custom trainer to: {dest}")
    return True


def install_custom_trainer_to_nnunetv2() -> bool:
    """Install nnUNetTrainerLungPretrained into nnunetv2."""
    return _install_trainer_file("nnunet_trainer_pretrained.py")


def install_wandb_trainer_to_nnunetv2() -> bool:
    """Install nnUNetTrainerWandb / nnUNetTrainerLungPretrainedWandb into nnunetv2."""
    return _install_trainer_file("nnunet_trainer_wandb.py")


def install_cediceskel_trainer_to_nnunetv2() -> bool:
    """Install nnUNetTrainerCeDiceSkel into nnunetv2."""
    return _install_trainer_file("nnunet_trainer_cediceskel.py")


def prediction_worker_profile(vram_gb: float | None) -> tuple[int, int]:
    """Choose npp/nps to maximize speed while keeping VRAM usage reasonable."""
    import os
    if os.name == 'nt':
        return 0, 0
        
    if vram_gb is None:
        return 2, 2
    if vram_gb >= 16:
        return 6, 6
    if vram_gb >= 12:
        return 4, 4
    if vram_gb >= 8:
        return 3, 3
    if vram_gb >= 6:
        return 2, 2
    return 1, 1

def build_parser() -> argparse.ArgumentParser:
    _, add_clusterfit_arguments, _ = import_clusterfit_helpers()

    def add_hidden_legacy_planner_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--resenc-preset", choices=["M", "L", "XL"], default=None, help=argparse.SUPPRESS)
        subparser.add_argument("--planner", default=None, help=argparse.SUPPRESS)

    def add_dataset_args(subparser: argparse.ArgumentParser) -> None:
        # Allow --dataset-id / --dataset-name after the subcommand as well as before.
        # Using SUPPRESS means these only override the root-parser value when explicitly
        # provided by the user; they don't clobber the root default when omitted.
        subparser.add_argument("--dataset-id", type=int, default=argparse.SUPPRESS)
        subparser.add_argument("--dataset-name", default=argparse.SUPPRESS)

    parser = argparse.ArgumentParser(description="nnU-Net v2 pipeline for Dataset001")
    parser.add_argument("--nnunet-root", type=Path, default=DEFAULT_NNUNET_ROOT)
    parser.add_argument("--dataset-id", type=int, default=1)
    parser.add_argument("--dataset-name", default="BPWoodDefects")

    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare", help="Process ZIPs and CVAT masks into nnU-Net raw format")
    prep.add_argument("--source", type=Path, default=Path("src/ground_truth"), help="Directory containing DICOM .zip files")
    prep.add_argument("--cvat-exports", type=Path, default=Path("src/cvat_exports"), help="Directory containing CVAT exported folders")
    prep.add_argument("--overwrite", action="store_true")
    add_clusterfit_arguments(prep)

    plan = subparsers.add_parser("plan", help="Run nnU-Net planning and preprocessing")
    add_dataset_args(plan)
    plan.add_argument("--verify-dataset-integrity", action="store_true")
    plan.add_argument("--resenc-preset", choices=["M", "L", "XL"], default="L")
    plan.add_argument("--planner", default=None)
    plan.add_argument("--plans-identifier", default=None)
    plan.add_argument("--configurations", nargs="+", default=None)
    plan.add_argument("--num-processes", type=int, default=None)
    add_clusterfit_arguments(plan)

    train = subparsers.add_parser("train", help="Train nnU-Net model")
    add_dataset_args(train)
    train.add_argument("--configuration", default="3d_fullres")
    train.add_argument("--fold", default="0", help="Fold index or 'all'")
    add_hidden_legacy_planner_args(train)
    train.add_argument("--plans-identifier", default=None)
    train.add_argument("--trainer", default=None, help="Optional custom trainer class name")
    train.add_argument("--save-every", type=int, default=10)
    train.add_argument("--skip-arch-plot", action="store_true")
    train.add_argument("--initial-lr", type=float, default=1e-3)
    train.add_argument(
        "--optimizer",
        choices=["sgd", "adam", "adamw"],
        default=None,
        help="Optimizer override (default: sgd, nnUNet built-in). Use adam/adamw with --initial-lr 1e-4.",
    )
    train.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay override for nnU-Net trainer optimizer (default: nnU-Net built-in).",
    )
    train.add_argument(
        "--regularize-arch",
        action="store_true",
        help="Patch the selected plans JSON before training (dropout + smaller model).",
    )
    train.add_argument(
        "--model-dropout-p",
        type=float,
        default=0.2,
        help="Dropout probability used when --regularize-arch is enabled.",
    )
    train.add_argument(
        "--model-features",
        type=int,
        nargs="+",
        default=None,
        help="features_per_stage override when --regularize-arch is enabled.",
    )
    train.add_argument(
        "--model-n-stages",
        type=int,
        choices=[5, 6],
        default=6,
        help="Encoder depth override when --regularize-arch is enabled.",
    )
    train.add_argument(
        "--model-batch-size",
        type=int,
        default=None,
        help="Patch plans batch_size when --regularize-arch is enabled.",
    )
    train.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        default=None,
        metavar=("D", "H", "W"),
        help="Override patch_size in the plans JSON (three ints: D H W). "
             "Applied before training without requiring --regularize-arch.",
    )
    train.add_argument(
        "--compile",
        choices=["auto", "on", "off"],
        default="auto",
        help="Control torch.compile usage (default: auto). Use 'off' if startup hangs.",
    )
    train.add_argument(
        "--n-proc-da",
        type=int,
        default=4,
        help="Number of nnU-Net data augmentation worker processes (default: 4).",
    )
    train.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
        help="Threads per process for BLAS/OpenMP libs (default: 1).",
    )
    train.add_argument(
        "--continue-training",
        action="store_true",
        help="Resume training from checkpoint_latest.pth (nnUNetv2_train --c)",
    )
    train.add_argument(
        "--pretrained-weights",
        type=Path,
        default=None,
        metavar="CHECKPOINT",
        help=(
            "Path to a pretrained nnUNet checkpoint (.pth or .model). "
            "Weights are loaded with strict=False so the output head trains from scratch. "
            "Install the Lung zip first: nnUNetv2_install_pretrained_model_from_zip Task006_Lung.zip"
        ),
    )
    train.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    train.add_argument("--wandb-project", default="nnunet-training", metavar="PROJECT", help="W&B project name (default: nnunet-training).")
    train.add_argument("--wandb-entity", default=None, metavar="ENTITY", help="W&B entity / team name.")
    train.add_argument("--wandb-run-name", default=None, metavar="NAME", help="Display name for this W&B run.")
    train.add_argument(
        "--test",
        action="store_true",
        help="Holdout test mode: exclude --test-tree cases from training and use them as the validation split.",
    )
    train.add_argument(
        "--test-tree",
        default="dub_2",
        metavar="TREE",
        help="Tree name to hold out as the test set in --test mode (default: dub_2).",
    )
    add_clusterfit_arguments(train)

    predict = subparsers.add_parser("predict", help="Run nnU-Net inference")
    add_dataset_args(predict)
    predict.add_argument("--input", type=Path, required=True, help="Folder with *_0000.nii.gz inputs")
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Optional folder with reference labels (*.nii.gz) for post-predict evaluation.",
    )
    predict.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Optional output JSON path for evaluation metrics (Dice and mIoU).",
    )
    predict.add_argument("--configuration", default="3d_fullres")
    predict.add_argument("--fold", default="0", help="Fold index or 'all'")
    predict.add_argument("--trainer", default=None,
                         help="Trainer class name used during training. "
                              "If omitted, auto-detected from existing model folders when possible.")
    add_hidden_legacy_planner_args(predict)
    predict.add_argument("--plans-identifier", default=None)
    predict.add_argument(
        "--save-probabilities",
        action="store_true",
        help="Save softmax probabilities (.b2nd) alongside predictions. "
             "Required for cascade: use this when predicting imagesTr with 3d_lowres "
             "and set --output to the predicted_next_stage/3d_cascade_fullres/ folder.",
    )
    add_clusterfit_arguments(predict)

    predict_tree = subparsers.add_parser("predict-tree", help="Run whole-tree inference and export to Datumaro")
    add_dataset_args(predict_tree)
    predict_tree.add_argument("--tree", required=True, help="Tree name (e.g., DUB_4)")
    predict_tree.add_argument(
        "--ground-truth-root",
        type=Path,
        default=PROJECT_ROOT / "src/ground_truth",
        help="Directory containing tree folders or zip files (default: project-root/src/ground_truth).",
    )
    predict_tree.add_argument("--segmentation-output-root", type=Path, required=True)
    predict_tree.add_argument(
        "--prepared-volume",
        type=Path,
        default=None,
        help=(
            "Path to a prebuilt NIfTI test volume (.nii or .nii.gz). "
            "When set, predict-tree skips ground-truth PNG preparation and uses this volume directly."
        ),
    )
    predict_tree.add_argument("--configuration", default="3d_fullres")
    predict_tree.add_argument("--fold", default="0", help="Fold index")
    predict_tree.add_argument("--make-datumaro", action="store_true", help="Convert NIfTI outputs to Datumaro format")
    predict_tree.add_argument("--trainer", default=None,
                              help="Trainer class name used during training. "
                                   "If omitted, auto-detected from existing model folders when possible.")
    predict_tree.add_argument("--npp", type=int, default=None, help="Number of preprocessing workers (overrides auto-detection).")
    predict_tree.add_argument("--nps", type=int, default=None, help="Number of segmentation export workers (overrides auto-detection).")
    predict_tree.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Split the 3D volume into chunks of N slices before prediction to reduce peak RAM. "
            "Predictions are merged automatically. Use ~350 for large trees (>1000 slices) on GPU nodes with 60G RAM."
        ),
    )
    add_hidden_legacy_planner_args(predict_tree)
    predict_tree.add_argument("--plans-identifier", default=None)
    add_clusterfit_arguments(predict_tree)

    all_cmd = subparsers.add_parser("all", help="Prepare + plan + train")
    add_dataset_args(all_cmd)
    all_cmd.add_argument("--source", type=Path, default=Path("src/ground_truth"))
    all_cmd.add_argument("--cvat-exports", type=Path, default=Path("src/cvat_exports"))
    all_cmd.add_argument("--overwrite", action="store_true")
    all_cmd.add_argument("--skip-prepare", action="store_true")
    add_hidden_legacy_planner_args(all_cmd)
    all_cmd.add_argument("--plans-identifier", default=None)
    all_cmd.add_argument("--configuration", default=None)
    all_cmd.add_argument("--fold", default="0")
    all_cmd.add_argument("--save-every", type=int, default=10)
    all_cmd.add_argument("--skip-arch-plot", action="store_true")
    all_cmd.add_argument("--initial-lr", type=float, default=1e-3)
    all_cmd.add_argument(
        "--optimizer",
        choices=["sgd", "adam", "adamw"],
        default=None,
        help="Optimizer override (default: sgd). Use adam/adamw with --initial-lr 1e-4.",
    )
    all_cmd.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay override for nnU-Net trainer optimizer (default: nnU-Net built-in).",
    )
    all_cmd.add_argument(
        "--regularize-arch",
        action="store_true",
        help="Patch the selected plans JSON before training (dropout + smaller model).",
    )
    all_cmd.add_argument(
        "--model-dropout-p",
        type=float,
        default=0.2,
        help="Dropout probability used when --regularize-arch is enabled.",
    )
    all_cmd.add_argument(
        "--model-features",
        type=int,
        nargs="+",
        default=None,
        help="features_per_stage override when --regularize-arch is enabled.",
    )
    all_cmd.add_argument(
        "--model-n-stages",
        type=int,
        choices=[5, 6],
        default=6,
        help="Encoder depth override when --regularize-arch is enabled.",
    )
    all_cmd.add_argument(
        "--model-batch-size",
        type=int,
        default=None,
        help="Patch plans batch_size when --regularize-arch is enabled.",
    )
    all_cmd.add_argument("--compile", choices=["auto", "on", "off"], default="auto")
    all_cmd.add_argument("--n-proc-da", type=int, default=4)
    all_cmd.add_argument("--cpu-threads", type=int, default=1)
    all_cmd.add_argument("--continue-training", action="store_true")
    all_cmd.add_argument(
        "--pretrained-weights",
        type=Path,
        default=None,
        metavar="CHECKPOINT",
        help="Path to pretrained checkpoint; enables nnUNetTrainerLungPretrained.",
    )
    all_cmd.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    all_cmd.add_argument("--wandb-project", default="nnunet-training", metavar="PROJECT")
    all_cmd.add_argument("--wandb-entity", default=None, metavar="ENTITY")
    all_cmd.add_argument("--wandb-run-name", default=None, metavar="NAME")
    all_cmd.add_argument("--plan-configurations", nargs="+", default=["3d_fullres"])
    all_cmd.add_argument("--plan-num-processes", type=int, default=1)
    add_clusterfit_arguments(all_cmd)

    custom_train_p = subparsers.add_parser(
        "custom-train",
        help="Train the custom 3D SwinUNETR wood-defect model (supports ClusterFIT)",
    )
    custom_train_p.add_argument("--dataset-id", type=int, default=2)
    custom_train_p.add_argument("--dataset-name", default="BPWoodDefectsSplit")
    custom_train_p.add_argument(
        "--image-dir", type=Path, default=None,
        help="Directory with input NIfTI volumes "
             "(default: nnunet_root/nnUNet_raw/DatasetXXX_<name>/imagesTr)",
    )
    custom_train_p.add_argument(
        "--label-dir", type=Path, default=None,
        help="Directory with label NIfTI volumes "
             "(default: nnunet_root/nnUNet_raw/DatasetXXX_<name>/labelsTr)",
    )
    custom_train_p.add_argument("--output-dir", type=Path, default=Path("./output/custom_model"))
    custom_train_p.add_argument("--epochs", type=int, default=1000)
    custom_train_p.add_argument("--batch-size", type=int, default=2)
    custom_train_p.add_argument("--patch-size", type=int, nargs=3, default=[128, 384, 128])
    custom_train_p.add_argument("--learning-rate", type=float, default=1e-3)
    custom_train_p.add_argument("--weight-decay", type=float, default=1e-4)
    custom_train_p.add_argument("--num-classes", type=int, default=7)
    custom_train_p.add_argument("--val-fraction", type=float, default=0.25)
    custom_train_p.add_argument("--num-workers", type=int, default=4)
    custom_train_p.add_argument("--seed", type=int, default=42)
    custom_train_p.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    custom_train_p.add_argument("--sliding-window-overlap", type=float, default=0.5)
    custom_train_p.add_argument(
        "--rare-label-idx", type=int, default=6,
        help="Label index to boost (default: 6 = Poškození hmyzem).",
    )
    custom_train_p.add_argument(
        "--rare-class-weight", type=float, default=15.0,
        help="CE loss weight multiplier for the rare class (default: 15).",
    )
    custom_train_p.add_argument(
        "--oversample-factor", type=int, default=8,
        help="Extra case copies per epoch for the rare class (default: 8).",
    )
    custom_train_p.add_argument(
        "--early-stopping-patience", type=int, default=50,
        help="Stop if val_dice does not improve for this many epochs (0 disables early stopping).",
    )
    custom_train_p.add_argument(
        "--early-stopping-min-delta", type=float, default=1e-4,
        help="Minimum val_dice increase to count as an improvement.",
    )
    custom_train_p.add_argument(
        "--early-stopping-min-epochs", type=int, default=50,
        help="Do not early-stop before this epoch count.",
    )
    custom_train_p.add_argument(
        "--grad-accumulation-steps",
        type=int,
        default=4,
        help="Accumulate gradients over N batches before each optimizer step (default: 4).",
    )
    custom_train_p.add_argument(
        "--warmup-epochs",
        type=int,
        default=20,
        help="Linear LR warmup duration in epochs before cosine decay (default: 20).",
    )
    custom_train_p.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping; 0 disables clipping (default: 1.0).",
    )
    custom_train_p.add_argument(
        "--dropout-path-rate",
        type=float,
        default=0.1,
        help="Stochastic depth drop-path rate for SwinUNETR (default: 0.1).",
    )
    custom_train_p.add_argument(
        "--model-name",
        choices=["swinunetr", "swinunetr_v2", "unetr", "basicunetplusplus", "mednext", "segmamba"],
        default="swinunetr",
        help="Custom model architecture to train (default: swinunetr).",
    )
    custom_train_p.add_argument(
        "--model-feature-size",
        type=int,
        default=48,
        help="Feature size for SwinUNETR/UNETR (default: 48).",
    )
    custom_train_p.add_argument(
        "--unetr-hidden-size",
        type=int,
        default=768,
        help="UNETR hidden transformer size (default: 768).",
    )
    custom_train_p.add_argument(
        "--unetr-mlp-dim",
        type=int,
        default=3072,
        help="UNETR MLP dimension (default: 3072).",
    )
    custom_train_p.add_argument(
        "--unetr-num-heads",
        type=int,
        default=12,
        help="UNETR number of attention heads (default: 12).",
    )
    custom_train_p.add_argument(
        "--basicunet-features",
        type=int,
        nargs=6,
        default=[32, 32, 64, 128, 256, 32],
        metavar=("F0", "F1", "F2", "F3", "F4", "F5"),
        help="Six channel sizes for BasicUNetPlusPlus (default: 32 32 64 128 256 32).",
    )
    custom_train_p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    custom_train_p.add_argument("--wandb-project", default="bp-custom-model", metavar="PROJECT", help="W&B project name.")
    custom_train_p.add_argument("--wandb-entity", default=None, metavar="ENTITY", help="W&B entity / team name.")
    custom_train_p.add_argument("--wandb-run-name", default=None, metavar="NAME", help="Display name for this W&B run.")
    custom_train_p.add_argument(
        "--cache-rate",
        type=float,
        default=0.0,
        help="Fraction of dataset to cache in RAM between epochs (0=off, 1.0=full cache). "
             "Set to 1.0 on A100 cluster nodes to eliminate per-epoch disk I/O.",
    )
    custom_train_p.add_argument(
        "--normalization",
        choices=["range", "zscore"],
        default="zscore",
        help="Intensity normalization mode for custom model training (default: zscore).",
    )
    custom_train_p.add_argument(
        "--norm-clip-min",
        type=float,
        default=-1000.0,
        help="Lower intensity bound used before normalization (default: -1000).",
    )
    custom_train_p.add_argument(
        "--norm-clip-max",
        type=float,
        default=500.0,
        help="Upper intensity bound used before normalization (default: 500).",
    )
    custom_train_p.add_argument(
        "--pretrained-weights",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to SSL pretrained SwinUNETR backbone (model_swinvit.pt). "
             "Only used for swinunetr / swinunetr_v2 models.",
    )
    custom_train_p.add_argument(
        "--loss-type",
        choices=["combined", "dice_focal"],
        default="combined",
        help="Loss function: 'combined' = CE+SoftDice+SkeletonRecall (default), "
             "'dice_focal' = legacy DiceFocal.",
    )
    custom_train_p.add_argument("--debug-data", action="store_true", help="Print dataset split and label diagnostics.")
    custom_train_p.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Fold index from splits_final.json (0-based). If not specified, uses stratified random split.",
    )
    custom_train_p.add_argument(
        "--splits-json",
        type=Path,
        default="splits_final.json",
        help="Path to splits JSON file for fold-based training (default: splits_final.json).",
    )
    custom_train_p.add_argument(
        "--test",
        action="store_true",
        help="Holdout test mode: exclude --test-tree cases from training and use them as the validation split.",
    )
    custom_train_p.add_argument(
        "--test-tree",
        default="dub_2",
        metavar="TREE",
        help="Tree name to hold out as the test set in --test mode (default: dub_2).",
    )
    custom_train_p.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to last_model.pth checkpoint to resume training from.",
    )
    add_clusterfit_arguments(custom_train_p)

    custom_predict_p = subparsers.add_parser(
        "custom-predict",
        help="Run inference with a trained custom model (MedNeXt, SwinUNETR V2, etc.)",
    )
    custom_predict_p.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing config.json and best_model.pth",
    )
    custom_predict_p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing *_0000.nii.gz input volumes",
    )
    custom_predict_p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory to write prediction NIfTI files",
    )
    add_clusterfit_arguments(custom_predict_p)

    custom_eval_p = subparsers.add_parser(
        "custom-evaluate",
        help="Evaluate custom model predictions against ground truth (nnUNet summary.json format)",
    )
    custom_eval_p.add_argument(
        "--pred-dir", type=Path, required=True,
        help="Directory with predicted NIfTI files",
    )
    custom_eval_p.add_argument(
        "--gt-dir", type=Path, required=True,
        help="Directory with ground-truth NIfTI files",
    )
    custom_eval_p.add_argument(
        "--num-classes", type=int, default=7,
        help="Total number of classes including background (default: 7)",
    )
    custom_eval_p.add_argument(
        "--output", type=Path, default=None,
        help="Output summary.json path (default: <pred-dir>/summary.json)",
    )
    add_clusterfit_arguments(custom_eval_p)

    cascade_prepare_p = subparsers.add_parser(
        "cascade-prepare",
        help=(
            "Convert lowres predictions to .b2nd NDArray for cascade fullres training. "
            "Run this after 'predict --save-probabilities' on imagesTr to fix the format "
            "before launching 3d_cascade_fullres training."
        ),
    )
    add_dataset_args(cascade_prepare_p)
    cascade_prepare_p.add_argument(
        "--pred-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing .npz (from --save-probabilities) or .nii.gz "
            "lowres predictions. .b2nd files are written to the same directory."
        ),
    )
    add_hidden_legacy_planner_args(cascade_prepare_p)
    cascade_prepare_p.add_argument("--plans-identifier", default=None)

    return parser


def run_cascade_prepare(args: argparse.Namespace) -> None:
    """Convert lowres predictions to NDArray .b2nd for cascade fullres training.

    nnUNet cascade expects predicted_next_stage/*.b2nd to contain 3D integer argmax
    arrays [D, H, W] in preprocessed lowres space.

    This function handles two input formats:
      .npz  — softmax [C, D, H, W] saved by nnUNetv2_predict --save_probabilities.
              Already in preprocessed lowres space; only argmax + format conversion needed.
      .nii.gz — argmax [D, H, W] in patient space (nnUNetv2_predict without the flag).
                Resampled to preprocessed lowres shape using nearest-neighbour zoom.
    """
    import blosc2
    from scipy.ndimage import zoom
    import nibabel as nib

    plans_identifier = resolve_plans_identifier(args)

    preprocessed_dir = (
        args.nnunet_root / "nnUNet_preprocessed"
        / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
        / f"{plans_identifier}_3d_lowres"
    )
    if not preprocessed_dir.exists():
        raise FileNotFoundError(f"Preprocessed lowres dir not found: {preprocessed_dir}")

    pred_dir = args.pred_dir
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction dir not found: {pred_dir}")

    # Build map: case_id -> preprocessed lowres shape [D, H, W]
    prep_shapes: dict[str, tuple] = {}
    for f in sorted(preprocessed_dir.glob("*.b2nd")):
        if f.stem.endswith("_seg"):
            continue
        data = blosc2.open(str(f))
        prep_shapes[f.stem] = data.shape[1:]  # skip channel dim
    for f in sorted(preprocessed_dir.glob("*.npz")):
        if f.stem.endswith("_seg") or f.stem in prep_shapes:
            continue
        d = np.load(str(f))
        key = list(d.keys())[0]
        prep_shapes[f.stem] = d[key].shape[1:]

    if not prep_shapes:
        raise RuntimeError(f"No preprocessed lowres cases found in {preprocessed_dir}")
    log(f"Found {len(prep_shapes)} preprocessed lowres shapes in {preprocessed_dir}")

    converted = skipped = 0
    for case_id, target_shape in sorted(prep_shapes.items()):
        out_b2nd = pred_dir / f"{case_id}.b2nd"
        npz_file = pred_dir / f"{case_id}.npz"
        nii_file = pred_dir / f"{case_id}.nii.gz"

        if npz_file.exists():
            # Softmax from --save_probabilities — already in preprocessed lowres space.
            # Just take argmax; no resampling needed.
            d = np.load(str(npz_file))
            key = list(d.keys())[0]
            softmax = d[key]          # [C, D, H, W]
            arr = np.argmax(softmax, axis=0).astype(np.int16)  # [D, H, W]
            if arr.shape != tuple(target_shape):
                # Defensive: resample if somehow dimensions differ
                factors = [t / s for t, s in zip(target_shape, arr.shape)]
                arr = zoom(arr.astype(float), factors, order=0).astype(np.int16)
            log(f"{case_id}: npz softmax{softmax.shape} -> argmax{arr.shape}")
        elif nii_file.exists():
            # Argmax in patient space — must resample to preprocessed lowres shape.
            import nibabel as nib  # noqa: F811
            pred_img = nib.load(str(nii_file))
            arr = np.asarray(pred_img.dataobj).astype(np.int16)
            if arr.shape != tuple(target_shape):
                factors = [t / s for t, s in zip(target_shape, arr.shape)]
                arr = zoom(arr.astype(float), factors, order=0).astype(np.int16)
            log(f"{case_id}: nii.gz argmax{arr.shape}")
        else:
            log(f"WARNING: no prediction found for {case_id} — skipping")
            skipped += 1
            continue

        if out_b2nd.exists():
            out_b2nd.unlink()
        blosc2.asarray(arr).save(str(out_b2nd))
        converted += 1

    log(f"cascade-prepare done: {converted} converted, {skipped} skipped. Output: {pred_dir}")


def run_prepare(args: argparse.Namespace) -> None:
    """Uses the new segmask2ima pipeline to automatically process all logs in ground_truth."""
    from src.preprocessing.conversion.segmask2ima import process_tree
    
    zip_files = list(args.source.glob("*.zip"))
    if not zip_files:
        log(f"No DICOM zip files found in {args.source}. Make sure your raw zips are there.")
        return
        
    for zip_file in zip_files:
        tree_name = zip_file.stem  # e.g., 'DUB_5' -> 'dub_5'
        log(f"Auto-processing dataset for: {tree_name}")
        process_tree(tree_name)


def build_holdout_split(
    nnunet_root: Path,
    dataset_id: int,
    dataset_name: str,
    test_tree: str,
) -> tuple:
    """Build a fold-0 holdout split that holds out all cases belonging to test_tree.

    Cases are discovered from imagesTr (strips the _XXXX.nii.gz modality suffix).
    Matching logic: exact name OR <test_tree>_partN variants (case-insensitive).
    Returns (train_ids, val_ids, split_path).
    """
    raw_dir = nnunet_root / "nnUNet_raw" / f"Dataset{dataset_id:03d}_{dataset_name}" / "imagesTr"
    if not raw_dir.exists():
        raise FileNotFoundError(f"imagesTr directory not found: {raw_dir}")

    all_ids = sorted({
        re.sub(r"_\d{4}\.nii\.gz$", "", f.name)
        for f in raw_dir.glob("*.nii.gz")
    })
    if not all_ids:
        raise RuntimeError(f"No NIfTI cases found in {raw_dir}")

    test_prefix = test_tree.lower().replace("-", "_")

    def _is_test(case_id: str) -> bool:
        norm = case_id.lower()
        return norm == test_prefix or bool(
            re.match(rf"^{re.escape(test_prefix)}_part\d+$", norm)
        )

    val_ids = [c for c in all_ids if _is_test(c)]
    train_ids = [c for c in all_ids if not _is_test(c)]

    if not val_ids:
        raise RuntimeError(
            f"No test cases matched '{test_tree}' in {raw_dir}. "
            f"Available (first 10): {all_ids[:10]}"
        )
    if not train_ids:
        raise RuntimeError(f"No training cases remain after excluding '{test_tree}'")

    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise RuntimeError(f"Unexpected overlap between train and val sets: {overlap}")

    split = [{"train": train_ids, "val": val_ids}]
    split_path = PROJECT_ROOT / f"splits_test_{test_prefix}.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

    log(f"Test holdout split: {len(train_ids)} train | {len(val_ids)} val (tree={test_tree})")
    log(f"  Val cases: {val_ids}")
    log(f"  Split file: {split_path}")
    return train_ids, val_ids, split_path


def run_plan(args: argparse.Namespace, env: Dict[str, str]) -> None:
    planner = planner_from_args(args)
    plans_identifier = args.plans_identifier or default_plans_for_planner(planner)
    cmd = build_plan_command(args, planner, plans_identifier)
    run_cmd(cmd, env, "plan + preprocess")


def run_train(args: argparse.Namespace, env: Dict[str, str]) -> None:
    plans_identifier = resolve_plans_identifier(args)
    configuration = resolve_train_configuration(args)

    apply_plan_regularization_overrides(args, plans_identifier, configuration)

    ensure_crossval_splits(
        nnunet_root=args.nnunet_root,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        configuration=configuration,
        plans_identifier=plans_identifier,
    )

    splits_file: Path | None = None
    splits_backup: Path | None = None

    if getattr(args, "test", False):
        test_tree = getattr(args, "test_tree", "dub_2")
        _, _, split_path = build_holdout_split(
            nnunet_root=args.nnunet_root,
            dataset_id=args.dataset_id,
            dataset_name=args.dataset_name,
            test_tree=test_tree,
        )
        dataset_dir = (
            args.nnunet_root / "nnUNet_preprocessed"
            / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
        )
        splits_file = dataset_dir / "splits_final.json"
        if splits_file.exists():
            splits_backup = splits_file.with_name("splits_final.json.testmode_bak")
            shutil.copy2(splits_file, splits_backup)
            log(f"Backed up {splits_file.name} → {splits_backup.name}")
        splits_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(split_path, splits_file)
        args.fold = "0"
        log(f"Test mode: fold forced to 0, holdout split written to {splits_file}")

    # Determine trainer based on --pretrained-weights and --wandb flags.
    pretrained_weights = getattr(args, "pretrained_weights", None)
    use_wandb = getattr(args, "wandb", False)
    trainer = getattr(args, "trainer", None)
    if not trainer:
        if pretrained_weights and use_wandb:
            trainer = "nnUNetTrainerRareClassBoostLungPretrainedWandb"
            if not install_wandb_trainer_to_nnunetv2():
                log("Warning: wandb trainer installation failed — falling back to LungPretrained.")
                trainer = "nnUNetTrainerLungPretrained"
                install_custom_trainer_to_nnunetv2()
        elif pretrained_weights:
            trainer = "nnUNetTrainerLungPretrained"
            if not install_custom_trainer_to_nnunetv2():
                log(
                    "Warning: custom trainer installation failed. "
                    "Falling back to base nnUNetTrainer with built-in --pretrained_weights."
                )
                trainer = None
        elif use_wandb:
            trainer = "nnUNetTrainerRareClassBoostWandb"
            if not install_wandb_trainer_to_nnunetv2():
                log("Warning: wandb trainer installation failed — training without W&B.")
                trainer = None

    if trainer == "nnUNetTrainerCeDiceSkel":
        if not install_cediceskel_trainer_to_nnunetv2():
            log("Warning: CeDiceSkel trainer installation failed — falling back to default trainer.")
            trainer = None

    cmd = [
        "nnUNetv2_train",
        str(args.dataset_id),
        configuration,
        str(args.fold),
        "-p",
        plans_identifier,
    ]
    if trainer:
        cmd.extend(["-tr", trainer])
    elif pretrained_weights:
        # Fallback: use nnUNetv2's native partial-weight loading
        cmd.extend(["--pretrained_weights", str(Path(pretrained_weights).resolve())])
    if args.continue_training:
        cmd.append("--c")

    try:
        run_cmd(cmd, env, "train")
    finally:
        if splits_file is not None:
            if splits_backup is not None and splits_backup.exists():
                shutil.move(str(splits_backup), str(splits_file))
                log("Restored original splits_final.json")
            elif splits_file.exists():
                splits_file.unlink()
                log("Removed temporary test-mode splits_final.json")


def convert_dicom_zip_to_nifti(zip_path: Path, output_dir: Path) -> None:
    log(f"Extracting and converting DICOM zip to NIfTI: {zip_path.name}")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(temp_path)

        dicom_files: List[Path] = []
        for ext in ("*.IMA", "*.dcm", "*.dicom"):
            dicom_files.extend(temp_path.rglob(ext))
        
        if not dicom_files:
            dicom_files.extend([p for p in temp_path.rglob("*") if p.is_file() and not p.suffix])

        if not dicom_files:
            raise FileNotFoundError(f"No DICOM/IMA files found in {zip_path}")

        dicom_dir = dicom_files[0].parent

        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))

        if not dicom_names:
            dicom_names = [str(p) for p in sorted(dicom_files)]

        reader.SetFileNames(dicom_names)
        try:
            image = reader.Execute()
        except Exception as e:
            raise RuntimeError(f"Failed to read DICOM series from {zip_path}: {e}")

        out_name = f"{zip_path.stem}_0000.nii.gz"
        sitk.WriteImage(image, str(output_dir / out_name), useCompression=True)
        log(f"Successfully converted to {out_name}")


def run_predict(args: argparse.Namespace, env: Dict[str, str]) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    
    input_path = args.input
    temp_input_dir = None

    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        temp_input_dir = Path(tempfile.mkdtemp(prefix="nnunet_pred_in_"))
        convert_dicom_zip_to_nifti(input_path, temp_input_dir)
        active_input = temp_input_dir
    elif input_path.is_dir():
        zip_files = list(input_path.glob("*.zip"))
        nii_files = list(input_path.glob("*_0000.nii.gz"))
        
        if zip_files and not nii_files:
            temp_input_dir = Path(tempfile.mkdtemp(prefix="nnunet_pred_in_"))
            for zf in zip_files:
                convert_dicom_zip_to_nifti(zf, temp_input_dir)
            active_input = temp_input_dir
        else:
            active_input = input_path
    else:
        active_input = input_path

    plans_identifier = resolve_plans_identifier(args)
    trainer = resolve_prediction_trainer(
        nnunet_root=args.nnunet_root,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        configuration=args.configuration,
        plans_identifier=plans_identifier,
        requested_trainer=getattr(args, "trainer", None),
    )
    checkpoint_name = resolve_prediction_checkpoint(
        nnunet_root=args.nnunet_root,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        configuration=args.configuration,
        plans_identifier=plans_identifier,
        fold=str(args.fold),
        trainer=trainer,
    )

    cmd = [
        "nnUNetv2_predict",
        "-i", str(active_input),
        "-o", str(args.output),
        "-d", str(args.dataset_id),
        "-c", args.configuration,
        "-f", str(args.fold),
        "-p", plans_identifier,
        "-tr", trainer,
    ]
    if checkpoint_name is not None:
        cmd.extend(["-chk", checkpoint_name])
        log(f"Using prediction checkpoint: {checkpoint_name}")
    if getattr(args, "save_probabilities", False):
        cmd.append("--save_probabilities")
        log("Saving softmax probabilities (cascade mode).")
    log(f"Using trainer: {trainer}")

    vram_gb = detect_gpu_vram_gb()
    npp, nps = prediction_worker_profile(vram_gb)
    cmd.extend(["-npp", str(npp), "-nps", str(nps)])
    
    if vram_gb is not None:
        log(f"Fast predict profile: disable_tta, npp={npp}, nps={nps} (GPU VRAM ~{vram_gb:.1f} GB)")
    else:
        log(f"Fast predict profile: disable_tta, npp={npp}, nps={nps} (GPU VRAM unknown)")
    
    try:
        run_cmd(cmd, env, "predict")

        labels_dir = getattr(args, "labels_dir", None)
        if labels_dir is not None:
            labels_dir = Path(labels_dir).expanduser()
            if not labels_dir.is_absolute():
                labels_dir = (PROJECT_ROOT / labels_dir).resolve()
            else:
                labels_dir = labels_dir.resolve()

            dataset_json_path = (
                args.nnunet_root
                / "nnUNet_raw"
                / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
                / "dataset.json"
            )
            metrics_json_path = getattr(args, "metrics_json", None)
            if metrics_json_path is None:
                metrics_json_path = args.output / "evaluation_metrics.json"
            else:
                metrics_json_path = Path(metrics_json_path).expanduser()
                if not metrics_json_path.is_absolute():
                    metrics_json_path = (PROJECT_ROOT / metrics_json_path).resolve()
                else:
                    metrics_json_path = metrics_json_path.resolve()

            evaluate_prediction_folder(
                prediction_dir=args.output,
                labels_dir=labels_dir,
                dataset_json_path=dataset_json_path,
                metrics_json_path=metrics_json_path,
            )
    finally:
        if temp_input_dir and temp_input_dir.exists():
            shutil.rmtree(temp_input_dir, ignore_errors=True)
            log("Cleaned up temporary NIfTI inputs.")


def run_predict_tree(args: argparse.Namespace, env: Dict[str, str]) -> None:
    from src.nn_UNet.tree_inference_helpers import (
        prepare_png_tree_from_ground_truth,
        write_tree_inference_nifti,
        export_prediction_masks,
        export_datumaro_for_tree,
    )

    tree_name = args.tree
    tree_slug = tree_name.lower().replace(" ", "_")
    ground_truth_root = Path(args.ground_truth_root).expanduser()
    if not ground_truth_root.is_absolute():
        ground_truth_root = (PROJECT_ROOT / ground_truth_root).resolve()
    else:
        ground_truth_root = ground_truth_root.resolve()

    tree_output_root = args.segmentation_output_root / tree_slug
    tree_segmentation_output = tree_output_root / "segmentation_style"
    tree_nifti_output = tree_output_root / "nnunet_nifti_predictions"
    tree_output_root.mkdir(parents=True, exist_ok=True)

    dataset_json_path = args.nnunet_root / "nnUNet_raw" / f"Dataset{args.dataset_id:03d}_{args.dataset_name}" / "dataset.json"

    temp_dir = Path(tempfile.mkdtemp(prefix=f"nnunet_tree_{tree_name}_"))
    png_root = temp_dir / "pngs"
    nifti_in_dir = temp_dir / "nifti_in"
    nifti_out_dir = temp_dir / "nifti_out"
    nifti_in_dir.mkdir(parents=True, exist_ok=True)
    nifti_out_dir.mkdir(parents=True, exist_ok=True)

    try:
        is_3d = "3d" in args.configuration.lower()
        prepared_volume = getattr(args, "prepared_volume", None)
        if prepared_volume is not None:
            prepared_volume = Path(prepared_volume).expanduser()
            if not prepared_volume.is_absolute():
                prepared_volume = (PROJECT_ROOT / prepared_volume).resolve()
            else:
                prepared_volume = prepared_volume.resolve()
        elif tree_slug == "dub_2":
            candidate_paths = [
                PROJECT_ROOT / "BPWoodSlices2" / "dub_2.nii.gz",
                Path.home() / "BPWoosCLices2" / "dub_2.nii.gz",
                Path.home() / "BPWoodSlices2" / "dub_2.nii.gz",
            ]
            for candidate in candidate_paths:
                if candidate.exists():
                    prepared_volume = candidate.resolve()
                    break

        tree_dir = None
        written_niftis: list = []
        chunk_size = getattr(args, "chunk_size", None)
        if prepared_volume is not None:
            if not prepared_volume.exists():
                raise FileNotFoundError(f"Prepared test volume not found: {prepared_volume}")
            out_name = f"{tree_name}_0000.nii.gz"
            shutil.copy2(prepared_volume, nifti_in_dir / out_name)
            log(f"Using prepared test volume: {prepared_volume}")
        else:
            log(f"Preparing PNGs for {tree_name}...")
            tree_dir = prepare_png_tree_from_ground_truth(
                tree_name=tree_name,
                png_root=png_root,
                ground_truth_root=ground_truth_root,
                temp_root=temp_dir
            )

            log("Converting slices to 3D NIfTI for nnU-Net inference...")
            written_niftis = write_tree_inference_nifti(tree_dir, nifti_in_dir, tree_name, is_3d, chunk_size=chunk_size)
            if len(written_niftis) > 1:
                log(f"Volume split into {len(written_niftis)} chunks of up to {chunk_size} slices each.")

        # Build the standard nnUNet predict command
        plans_identifier = resolve_plans_identifier(args)
        trainer = resolve_prediction_trainer(
            nnunet_root=args.nnunet_root,
            dataset_id=args.dataset_id,
            dataset_name=args.dataset_name,
            configuration=args.configuration,
            plans_identifier=plans_identifier,
            requested_trainer=getattr(args, "trainer", None),
        )
        checkpoint_name = resolve_prediction_checkpoint(
            nnunet_root=args.nnunet_root, dataset_id=args.dataset_id,
            dataset_name=args.dataset_name, configuration=args.configuration,
            plans_identifier=plans_identifier, fold=str(args.fold), trainer=trainer,
        )

        cmd = [
            "nnUNetv2_predict",
            "-i", str(nifti_in_dir), "-o", str(nifti_out_dir),
            "-d", str(args.dataset_id), "-c", args.configuration,
            "-f", str(args.fold), "-p", plans_identifier,
            "-tr", trainer,
        ]
        if checkpoint_name:
            cmd.extend(["-chk", checkpoint_name])

        vram_gb = detect_gpu_vram_gb()
        npp_auto, nps_auto = prediction_worker_profile(vram_gb)
        npp = getattr(args, "npp", None) if getattr(args, "npp", None) is not None else npp_auto
        nps = getattr(args, "nps", None) if getattr(args, "nps", None) is not None else nps_auto
        cmd.extend(["-npp", str(npp), "-nps", str(nps)])
        log(f"Worker profile: npp={npp}, nps={nps} (auto={npp_auto}/{nps_auto}, VRAM ~{vram_gb:.1f} GB)" if vram_gb else f"Worker profile: npp={npp}, nps={nps}")

        log("Running nnU-Net prediction...")
        if len(written_niftis) > 1:
            # Run one nnUNet call per chunk so each call's background export
            # workers finish and release RAM before the next chunk starts.
            # Running all chunks in a single call causes export workers to
            # accumulate across chunks and OOM on the third chunk.
            for chunk_nifti in written_niftis:
                chunk_in_dir = temp_dir / f"in_{chunk_nifti.stem}"
                chunk_in_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(chunk_nifti, chunk_in_dir / chunk_nifti.name)
                chunk_cmd = list(cmd)
                i_idx = chunk_cmd.index("-i")
                chunk_cmd[i_idx + 1] = str(chunk_in_dir)
                log(f"Predicting chunk: {chunk_nifti.stem.replace('_0000', '')}")
                run_cmd(chunk_cmd, env, f"predict-tree:{chunk_nifti.stem}")
                shutil.rmtree(chunk_in_dir)
            from src.nn_UNet.tree_inference_helpers import merge_prediction_chunks
            log(f"Merging {len(written_niftis)} prediction chunks...")
            merge_prediction_chunks(nifti_out_dir, tree_name)
            log("Prediction chunks merged.")
        else:
            run_cmd(cmd, env, "predict-tree")

        # Datumaro Export Phase
        if args.make_datumaro:
            if tree_dir is None:
                raise RuntimeError(
                    "--make-datumaro requires PNG tree geometry. "
                    "Run without --prepared-volume, or disable --make-datumaro."
                )
            log("Slicing NIfTI into PNG masks and formatting for Datumaro...")
            export_prediction_masks(
                prediction_dir=nifti_out_dir,
                tree_dir=tree_dir,
                segmentation_output_dir=tree_segmentation_output,
                dataset_json_path=dataset_json_path,
                tree_name=tree_name,
                is_3d=is_3d
            )
            datumaro_zip = tree_output_root / f"datumaro_{tree_name}.zip"
            export_datumaro_for_tree(tree_segmentation_output, datumaro_zip, tree_name)
            log(f"Success! Datumaro dataset zipped at: {datumaro_zip}")
        else:
            shutil.copytree(nifti_out_dir, tree_nifti_output, dirs_exist_ok=True)
            log(f"Saved standard NIfTI predictions to {tree_nifti_output}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        log("Cleaned up temporary prediction files.")
# ---------------------------------------------------

def run_custom_train(args: argparse.Namespace, env: Dict[str, str]) -> None:
    """Run `python -m src.custom_model.train` with the args from the custom-train subcommand."""
    raw_root = args.nnunet_root / "nnUNet_raw" / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    image_dir = args.image_dir or (raw_root / "imagesTr")
    label_dir = args.label_dir or (raw_root / "labelsTr")

    if getattr(args, "test", False):
        test_tree = getattr(args, "test_tree", "dub_2")
        _, _, split_path = build_holdout_split(
            nnunet_root=args.nnunet_root,
            dataset_id=args.dataset_id,
            dataset_name=args.dataset_name,
            test_tree=test_tree,
        )
        args.fold = 0
        args.splits_json = split_path
        log(f"Test mode: fold=0, splits_json={split_path}")

    cmd = [
        sys.executable, "-m", "src.custom_model.train",
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
        cmd.extend(["--fold", str(args.fold)])
        cmd.extend(["--splits-json", str(args.splits_json)])
    if getattr(args, "test", False):
        cmd.append("--test-mode")
        cmd.extend(["--test-tree", str(getattr(args, "test_tree", "dub_2"))])

    log(f"Custom model training: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError:
        log("Failed custom-train")
        raise


def _compute_nnunet_summary(
    pred_dir: Path,
    gt_dir: Path,
    num_classes: int = 7,
    output_json: Path | None = None,
) -> dict:
    """Compute per-case and mean metrics in nnUNet summary.json format."""
    import nibabel as nib
    from nibabel.orientations import io_orientation, ornt_transform, apply_orientation

    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    if not pred_files:
        raise RuntimeError(f"No prediction NIfTI files found in {pred_dir}")

    metric_per_case = []
    for pred_file in pred_files:
        gt_file = gt_dir / pred_file.name
        if not gt_file.exists():
            raise FileNotFoundError(
                f"No ground-truth file for '{pred_file.name}' in {gt_dir}"
            )

        gt_nib = nib.load(str(gt_file))
        pred_nib = nib.load(str(pred_file))

        gt_arr = np.asarray(gt_nib.dataobj).astype(np.int16)
        pred_arr = np.asarray(pred_nib.dataobj).astype(np.int16)

        # Reorient prediction voxel axes to match GT so the comparison is voxel-accurate
        # even when the prediction was saved in a different orientation (e.g. RAS vs LPS).
        gt_ornt = io_orientation(gt_nib.affine)
        pred_ornt = io_orientation(pred_nib.affine)
        if not np.array_equal(gt_ornt, pred_ornt):
            pred_arr = apply_orientation(
                pred_arr, ornt_transform(pred_ornt, gt_ornt)
            ).astype(np.int16)
            log(f"  Reoriented prediction to match GT axes for {pred_file.name}")

        if pred_arr.shape != gt_arr.shape:
            raise RuntimeError(
                f"Shape mismatch for {pred_file.name}: pred={pred_arr.shape}, gt={gt_arr.shape}"
            )

        total = int(pred_arr.size)
        case_metrics: dict[str, dict] = {}
        for c in range(1, num_classes):
            pm = pred_arr == c
            gm = gt_arr == c
            tp = int(np.logical_and(pm, gm).sum())
            fp = int(np.logical_and(pm, ~gm).sum())
            fn = int(np.logical_and(~pm, gm).sum())
            tn = total - tp - fp - fn
            dice_den = 2 * tp + fp + fn
            iou_den = tp + fp + fn
            case_metrics[str(c)] = {
                "Dice": (2.0 * tp / dice_den) if dice_den > 0 else 0.0,
                "FN": fn,
                "FP": fp,
                "IoU": (float(tp) / iou_den) if iou_den > 0 else 0.0,
                "TN": tn,
                "TP": tp,
                "n_pred": tp + fp,
                "n_ref": tp + fn,
            }

        metric_per_case.append({
            "metrics": case_metrics,
            "prediction_file": str(pred_file),
            "reference_file": str(gt_file),
        })

    class_keys = [str(c) for c in range(1, num_classes)]
    metric_names = ["Dice", "FN", "FP", "IoU", "TN", "TP", "n_pred", "n_ref"]
    mean = {
        c: {k: float(np.mean([case["metrics"][c][k] for case in metric_per_case])) for k in metric_names}
        for c in class_keys
    }
    foreground_mean = {
        k: float(np.mean([mean[c][k] for c in class_keys]))
        for k in metric_names
    }

    result = {"foreground_mean": foreground_mean, "mean": mean, "metric_per_case": metric_per_case}

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
        log(f"Saved summary to {output_json}")

    log(f"Foreground mean  Dice={foreground_mean['Dice']:.4f}  IoU={foreground_mean['IoU']:.4f}")
    for c in class_keys:
        log(f"  class {c}  Dice={mean[c]['Dice']:.4f}  IoU={mean[c]['IoU']:.4f}")

    return result


def run_custom_evaluate(args: argparse.Namespace, _env: Dict[str, str]) -> None:
    output = args.output or (Path(args.pred_dir) / "summary.json")
    _compute_nnunet_summary(
        pred_dir=Path(args.pred_dir),
        gt_dir=Path(args.gt_dir),
        num_classes=args.num_classes,
        output_json=output,
    )


def run_custom_predict(args: argparse.Namespace, env: Dict[str, str]) -> None:
    import nibabel as nib
    from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation
    import torch
    from monai.inferers import sliding_window_inference
    from src.custom_model.model import get_model
    from src.custom_model.transforms import get_inference_transforms

    model_dir = Path(args.model_dir)
    config_path = model_dir / "config.json"
    checkpoint_path = model_dir / "best_model.pth"

    log(f"Loading config from {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_name = cfg.get("model_name", "swinunetr")
    num_classes = cfg.get("num_classes", 7)
    patch_size = tuple(cfg.get("patch_size", [128, 384, 128]))
    dropout_path_rate = cfg.get("dropout_path_rate", 0.1)
    feature_size = cfg.get("model_feature_size", 48)
    sliding_window_overlap = cfg.get("sliding_window_overlap", 0.5)
    normalization = cfg.get("normalization", "zscore")
    clip_min = cfg.get("norm_clip_min", -1000.0)
    clip_max = cfg.get("norm_clip_max", 500.0)
    zscore_mean = cfg.get("normalization_mean")
    zscore_std = cfg.get("normalization_std")

    log(f"Model: {model_name}, classes={num_classes}, patch_size={patch_size}")

    model = get_model(
        model_name=model_name,
        num_classes=num_classes,
        img_size=patch_size,
        dropout_path_rate=dropout_path_rate,
        feature_size=feature_size,
    )

    log(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    log(f"Device: {device}")

    transforms = get_inference_transforms(
        normalization=normalization,
        clip_min=clip_min,
        clip_max=clip_max,
        zscore_mean=zscore_mean,
        zscore_std=zscore_std,
    )

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*_0000.nii.gz")) or sorted(input_dir.glob("*.nii.gz"))
    if not input_files:
        raise RuntimeError(f"No NIfTI input files found in {input_dir}")
    log(f"Found {len(input_files)} input file(s)")

    for nii_file in input_files:
        stem = nii_file.name[:-7] if nii_file.name.endswith(".nii.gz") else Path(nii_file.name).stem
        case_name = re.sub(r"_0000(?:_\d+)?$", "", stem)
        log(f"Predicting: {nii_file.name} → {case_name}.nii.gz")

        data = transforms({"image": str(nii_file)})
        img_tensor = data["image"]
        inputs = img_tensor.unsqueeze(0).to(device)

        with torch.inference_mode():
            outputs = sliding_window_inference(
                inputs,
                roi_size=patch_size,
                sw_batch_size=1,
                predictor=model,
                overlap=sliding_window_overlap,
            )

        pred_ras = outputs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Inference ran in RAS space (Orientationd); reorient back to original image space
        # so the saved prediction aligns voxel-for-voxel with the source file.
        orig_nib = nib.load(str(nii_file))
        orig_ornt = io_orientation(orig_nib.affine)
        ras_to_orig = ornt_transform(axcodes2ornt(("R", "A", "S")), orig_ornt)
        pred = apply_orientation(pred_ras, ras_to_orig).astype(np.uint8)

        nib.save(nib.Nifti1Image(pred, orig_nib.affine), str(output_dir / f"{case_name}.nii.gz"))
        log(f"Saved: {output_dir / f'{case_name}.nii.gz'}")


def submit_to_clusterfit(args: argparse.Namespace, env: Dict[str, str]) -> None:
    SlurmJobSubmitter, _, build_slurm_config_from_args = import_clusterfit_helpers()
    
    if SlurmJobSubmitter is None:
        raise RuntimeError("ClusterFIT utilities are missing. Cannot submit to Slurm from this environment.")

    slurm_env: Dict[str, str] = {}
    for key in (
        "PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "VIRTUAL_ENV",
        "PYTHONPATH", "PYTHONUNBUFFERED", "nnUNet_raw", "nnUNet_preprocessed",
        "nnUNet_results", "NNUNET_SAVE_EVERY", "NNUNET_INITIAL_LR", "NNUNET_OPTIMIZER",
        "NNUNET_WEIGHT_DECAY",
        "NNUNET_PRETRAINED_WEIGHTS", "NNUNET_SKIP_ARCH_PLOT", "nnUNet_compile", "nnUNet_n_proc_DA",
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_RUN_NAME", "WANDB_API_KEY",
        "PYTORCH_CUDA_ALLOC_CONF",
    ):
        value = env.get(key)
        if value:
            slurm_env[key] = value

    slurm_config = build_slurm_config_from_args(args, args.command)
    
    original_args = sys.argv[1:]
    safe_args = [arg for arg in original_args if arg != "--clusterfit"]
    cmd = [sys.executable, str(Path(__file__).resolve())] + safe_args

    script_content = SlurmJobSubmitter.build_slurm_script(
        job_command=cmd,
        slurm_config=slurm_config,
        env_vars=slurm_env,
        modules_load=["cray-ccdb", "cray-mvapich2_pmix_nogpu"] if getattr(args, "arm_hpe_cpe", False) else None,
    )
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script_content)
        script_path = Path(f.name)
    
    log(f"Slurm script written to: {script_path}")
    log(f"Partition: {slurm_config.partition}, Time: {slurm_config.time}")
    if slurm_config.gres:
        log(f"GPU: {slurm_config.gres}")
    
    script_path.chmod(0o755)
    
    try:
        job_id = SlurmJobSubmitter.submit_job(
            script_path,
            dry_run=args.slurm_dry_run,
            wait=args.slurm_wait,
        )
        if job_id:
            log(f"Job submitted successfully with ID: {job_id}")
            if args.slurm_wait:
                log("Job completed (--slurm-wait was set)")
        else:
            log("Job submission completed")
    finally:
        if not args.slurm_dry_run:
            log(f"Slurm script saved at: {script_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    env = ensure_env(args.nnunet_root)
    apply_runtime_env_overrides(env, args)

    started = time.perf_counter()
    log(f"Running command: {args.command}")
    
    if getattr(args, "clusterfit", False):
        log("Submitting to ClusterFIT...")
        submit_to_clusterfit(args, env)
        log(f"Submission completed in {int(time.perf_counter() - started)}s")
        return

    if args.command in {"prepare", "all"}:
        if args.command == "all" and args.skip_prepare:
            if not has_prepared_dataset(args.nnunet_root, args.dataset_id, args.dataset_name):
                dataset_root = prepared_dataset_root(args.nnunet_root, args.dataset_id, args.dataset_name)
                raise RuntimeError(
                    f"--skip-prepare was set, but prepared dataset is missing or incomplete at {dataset_root}."
                )
            log("Skipping prepare step (reusing existing prepared dataset).")
        else:
            run_prepare(args)

    if args.command in {"plan", "all"}:
        run_plan(args, env)

    if args.command in {"train", "all"}:
        run_train(args, env)

    if args.command == "cascade-prepare":
        run_cascade_prepare(args)

    if args.command == "predict":
        run_predict(args, env)

    if args.command == "predict-tree":
        run_predict_tree(args, env)

    if args.command == "custom-train":
        run_custom_train(args, env)

    if args.command == "custom-predict":
        run_custom_predict(args, env)

    if args.command == "custom-evaluate":
        run_custom_evaluate(args, env)

    log(f"Done in {int(time.perf_counter() - started)}s")


if __name__ == "__main__":
    main()