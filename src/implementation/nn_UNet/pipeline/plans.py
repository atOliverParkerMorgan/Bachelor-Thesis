from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .env import (
    log,
    DEFAULT_PLANNER,
    PLANNER_FOR_PRESET,
    DEFAULT_PLANS_FOR_PLANNER,
    RESENC_PLAN_NAMES,
    CHECKPOINT_CANDIDATES,
)
from .utils import _fit_int_list, decoder_len_for_stages


def planner_from_args(args) -> str:
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
    return sorted({p.stem for p in dataset_dir.glob("*Plans.json") if p.is_file()})


def resolve_plans_identifier(args) -> str:
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
            log(f"Falling back to 'nnUNetPlans' (preferred '{preferred}' missing).")
        return "nnUNetPlans"
    if available:
        log(f"Using '{available[0]}' (preferred '{preferred}' not found).")
        return available[0]
    return preferred


def prepared_dataset_root(nnunet_root: Path, dataset_id: int, dataset_name: str) -> Path:
    return nnunet_root / "nnUNet_raw" / f"Dataset{dataset_id:03d}_{dataset_name}"


def has_prepared_dataset(nnunet_root: Path, dataset_id: int, dataset_name: str) -> bool:
    root = prepared_dataset_root(nnunet_root, dataset_id, dataset_name)
    images_tr = root / "imagesTr"
    labels_tr = root / "labelsTr"
    return (
        (root / "dataset.json").exists()
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
        fallback = dataset_dir / f"nnUNetPlans_{configuration}"
        if not fallback.exists():
            return
        config_dir = fallback

    case_ids = sorted({p.stem for p in config_dir.glob("*.b2nd") if not p.stem.endswith("_seg")})
    if not case_ids:
        case_ids = sorted({p.stem for p in config_dir.glob("*.npz") if not p.stem.endswith("_seg")})

    num_cases = len(case_ids)
    if num_cases == 0 or num_cases >= 5:
        return
    if num_cases < 2:
        raise RuntimeError(f"Training requires at least 2 cases. Found {num_cases} in {config_dir}.")

    splits = [{"train": [c for c in case_ids if c != case_ids[i]], "val": [case_ids[i]]} for i in range(num_cases)]
    with open(splits_file, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    log(f"Created splits_final.json with {num_cases} folds for tiny dataset.")


def build_plan_command(args, planner: str, plans_identifier: str) -> List[str]:
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


def resolve_train_configuration(args) -> str:
    if args.command != "all":
        return args.configuration
    if args.configuration:
        return args.configuration
    if args.plan_configurations and len(args.plan_configurations) == 1:
        return args.plan_configurations[0]
    return "3d_fullres"


def apply_plan_regularization_overrides(args, plans_identifier: str, configuration: str) -> None:
    regularize = getattr(args, "regularize_arch", False)
    patch_size = getattr(args, "patch_size", None)
    model_batch_size = getattr(args, "model_batch_size", None)

    if not regularize and patch_size is None and model_batch_size is None:
        return

    plans_file = (
        args.nnunet_root / "nnUNet_preprocessed"
        / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
        / f"{plans_identifier}.json"
    )
    if not plans_file.exists():
        log(f"Plan override skipped: {plans_file} not found. Run plan/preprocess first.")
        return

    with open(plans_file, "r", encoding="utf-8") as f:
        plans = json.load(f)

    cfg = plans.get("configurations", {}).get(configuration)
    if not isinstance(cfg, dict):
        log(f"Plan override skipped: configuration '{configuration}' not found in {plans_file.name}.")
        return

    if regularize:
        arch_kwargs = cfg.get("architecture", {}).get("arch_kwargs")
        if not isinstance(arch_kwargs, dict):
            log("Plan override skipped: arch_kwargs missing.")
            return

        n_stages = int(getattr(args, "model_n_stages", 6) or 6)
        user_features = getattr(args, "model_features", None)
        features = [int(v) for v in user_features] if user_features else (
            [32, 64, 128, 256, 256, 256] if n_stages == 6 else [32, 64, 128, 256, 256]
        )
        features = _fit_int_list(features, n_stages)

        arch_kwargs["n_stages"] = n_stages
        arch_kwargs["features_per_stage"] = features
        arch_kwargs["dropout_op"] = "torch.nn.Dropout3d"
        arch_kwargs["dropout_op_kwargs"] = {"p": float(getattr(args, "model_dropout_p", 0.2)), "inplace": True}

        for key in ("kernel_sizes", "strides", "n_blocks_per_stage"):
            if isinstance(arch_kwargs.get(key), list):
                arch_kwargs[key] = _fit_int_list(arch_kwargs[key], n_stages)

        decoder_key = "n_conv_per_stage_decoder"
        if isinstance(arch_kwargs.get(decoder_key), list):
            arch_kwargs[decoder_key] = _fit_int_list(arch_kwargs[decoder_key], decoder_len_for_stages(n_stages))

    if patch_size is not None:
        cfg["patch_size"] = [int(v) for v in patch_size]

    if model_batch_size is not None:
        cfg["batch_size"] = int(model_batch_size)

    with open(plans_file, "w", encoding="utf-8") as f:
        json.dump(plans, f, indent=2)

    parts = []
    if regularize:
        ak = cfg.get("architecture", {}).get("arch_kwargs", {})
        parts += [
            f"dropout={ak.get('dropout_op_kwargs', {}).get('p')}",
            f"n_stages={getattr(args, 'model_n_stages', 6)}",
            f"features={ak.get('features_per_stage')}",
        ]
    if patch_size is not None:
        parts.append(f"patch_size={cfg['patch_size']}")
    if model_batch_size is not None:
        parts.append(f"batch_size={cfg.get('batch_size')}")
    log("Applied plans override: " + ", ".join(parts))


def model_output_dir(
    nnunet_root: Path,
    dataset_id: int,
    dataset_name: str,
    configuration: str,
    plans_identifier: str,
    trainer: str = "nnUNetTrainer",
) -> Path:
    return (
        nnunet_root / "nnUNet_results"
        / f"Dataset{dataset_id:03d}_{dataset_name}"
        / f"{trainer}__{plans_identifier}__{configuration}"
    )


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
    return sorted({d.name[: -len(suffix)] for d in dataset_dir.glob(f"*{suffix}") if d.is_dir()})


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
        log(f"Auto-detected trainer: {available[0]}")
        return available[0]
    if len(available) > 1:
        for preferred in ("nnUNetTrainer", "nnUNetTrainerRareClassBoostWandb", "nnUNetTrainerLungPretrained"):
            if preferred in available:
                log(f"Auto-selected trainer: {preferred}")
                return preferred
        raise RuntimeError(
            f"Multiple trainers found. Pass --trainer explicitly. Available: {', '.join(available)}"
        )
    return "nnUNetTrainerRareClassBoostWandb"


def prediction_folds(fold_value: str, model_dir: Path) -> List[str]:
    if fold_value == "all":
        return sorted(p.name for p in model_dir.glob("fold_*") if p.is_dir())
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
    for name in CHECKPOINT_CANDIDATES:
        if all((model_dir / fn / name).exists() for fn in folds):
            return name
    return None
