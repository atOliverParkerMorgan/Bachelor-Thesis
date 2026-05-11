from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict

# env.py is at src/implementation/nn_UNet/pipeline/env.py; parents[4] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_PLANNER = "nnUNetPlannerResEncL"

PLANNER_FOR_PRESET: dict[str, str] = {
    "M": "nnUNetPlannerResEncM",
    "L": "nnUNetPlannerResEncL",
    "XL": "nnUNetPlannerResEncXL",
}

DEFAULT_PLANS_FOR_PLANNER: dict[str, str] = {
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

DEFAULT_NNUNET_ROOT = Path("src/implementation/nn_UNet/nnunet_data")


def import_clusterfit_helpers():
    try:
        from src.implementation.nn_UNet.clusterfit_utils import (
            SlurmJobSubmitter,
            add_clusterfit_arguments,
            build_slurm_config_from_args,
        )
        return SlurmJobSubmitter, add_clusterfit_arguments, build_slurm_config_from_args
    except ImportError:
        def _noop(*args, **kwargs):
            pass
        return None, _noop, None


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}")


def ensure_env(nnunet_root: Path) -> Dict[str, str]:
    env = os.environ.copy()
    raw = nnunet_root / "nnUNet_raw"
    preprocessed = nnunet_root / "nnUNet_preprocessed"
    results = nnunet_root / "nnUNet_results"

    raw.mkdir(parents=True, exist_ok=True)
    preprocessed.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    env["nnUNet_raw"] = str(raw.resolve())
    env["nnUNet_preprocessed"] = str(preprocessed.resolve())
    env["nnUNet_results"] = str(results.resolve())

    project_root_str = str(PROJECT_ROOT)
    existing = env.get("PYTHONPATH", "")
    if existing:
        parts = existing.split(os.pathsep)
        if project_root_str not in parts:
            env["PYTHONPATH"] = os.pathsep.join([project_root_str, existing])
    else:
        env["PYTHONPATH"] = project_root_str

    return env


def apply_runtime_env_overrides(env: Dict[str, str], args) -> None:
    env["PYTHONUNBUFFERED"] = "1"

    # Reduces CUDA allocator fragmentation for large 3D patch training
    if getattr(args, "command", None) == "custom-train":
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    _maybe_set(env, "NNUNET_SAVE_EVERY", getattr(args, "save_every", None))
    _maybe_set(env, "NNUNET_INITIAL_LR", getattr(args, "initial_lr", None))
    _maybe_set(env, "NNUNET_OPTIMIZER", getattr(args, "optimizer", None))
    _maybe_set(env, "NNUNET_WEIGHT_DECAY", getattr(args, "weight_decay", None))

    if getattr(args, "skip_arch_plot", False):
        env["NNUNET_SKIP_ARCH_PLOT"] = "1"

    compile_mode = getattr(args, "compile", None)
    if compile_mode == "off":
        env["nnUNet_compile"] = "0"
    elif compile_mode == "on":
        env["nnUNet_compile"] = "1"

    pretrained = getattr(args, "pretrained_weights", None)
    if pretrained is not None:
        env["NNUNET_PRETRAINED_WEIGHTS"] = str(Path(pretrained).resolve())

    _maybe_set(env, "WANDB_PROJECT", getattr(args, "wandb_project", None))
    _maybe_set(env, "WANDB_ENTITY", getattr(args, "wandb_entity", None))
    _maybe_set(env, "WANDB_RUN_NAME", getattr(args, "wandb_run_name", None))
    _maybe_set(env, "WANDB_API_KEY", getattr(args, "wandb_api_key", None))
    _maybe_set(env, "nnUNet_n_proc_DA", getattr(args, "n_proc_da", None))

    cpu_threads = getattr(args, "cpu_threads", None)
    if cpu_threads is not None:
        t = str(cpu_threads)
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[var] = t


def _maybe_set(env: Dict[str, str], key: str, value) -> None:
    if value is not None:
        env[key] = str(value)
