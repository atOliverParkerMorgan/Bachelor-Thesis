#!/usr/bin/env python3
"""Command wrappers for nnU-Net v2 using Dataset001 with 3D defaults."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List


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
    return env


def _fmt_elapsed(seconds: float) -> str:
    total = int(round(seconds))
    mins, secs = divmod(total, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours}h {mins}m {secs}s"
    if mins:
        return f"{mins}m {secs}s"
    return f"{secs}s"


def log_status(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def log_loading_bar(completed: int, total: int, label: str) -> None:
    total = max(total, 1)
    completed = min(max(completed, 0), total)
    width = 30
    filled = int((completed / total) * width)
    bar = "#" * filled + "-" * (width - filled)
    percent = int((completed / total) * 100)
    log_status(f"[{bar}] {percent:3d}% {label}")


def run_cmd(command: List[str], env: Dict[str, str], label: str) -> None:
    executable = command[0]
    if shutil.which(executable) is None:
        raise RuntimeError(
            f"Required executable not found: {executable}. Install nnunetv2 and run through poetry."
        )
    print("$", " ".join(command))
    try:
        subprocess.run(command, check=True, env=env)
    except subprocess.CalledProcessError:
        log_status(f"Failed {label}")
        raise


def ensure_crossval_splits(
    nnunet_root: Path,
    dataset_id: int,
    dataset_name: str,
    configuration: str,
    plans_identifier: str,
) -> None:
    """Create splits_final.json for tiny datasets so nnU-Net does not force 5-fold CV."""
    dataset_dir = nnunet_root / "nnUNet_preprocessed" / f"Dataset{dataset_id:03d}_{dataset_name}"
    splits_file = dataset_dir / "splits_final.json"
    if splits_file.exists():
        return

    config_dir = dataset_dir / f"{plans_identifier}_{configuration}"
    if not config_dir.exists():
        fallback_dir = dataset_dir / f"nnUNetPlans_{configuration}"
        if fallback_dir.exists():
            config_dir = fallback_dir
        else:
            # If preprocessing for this config is missing, nnU-Net will emit the canonical error later.
            return

    case_ids = sorted(
        {
            path.stem
            for path in config_dir.glob("*.b2nd")
            if not path.stem.endswith("_seg")
        }
    )
    if not case_ids:
        case_ids = sorted(
            {
                path.stem
                for path in config_dir.glob("*.npz")
                if not path.stem.endswith("_seg")
            }
        )

    num_cases = len(case_ids)
    if num_cases == 0:
        return
    if num_cases < 2:
        raise RuntimeError(
            "Training requires at least 2 cases. Found only "
            f"{num_cases} case in {config_dir}."
        )
    if num_cases >= 5:
        return

    n_splits = num_cases
    splits = []
    for idx in range(n_splits):
        val = [case_ids[idx]]
        train = [case_id for case_id in case_ids if case_id not in val]
        splits.append({"train": train, "val": val})

    with open(splits_file, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    log_status(
        "Created splits_final.json with "
        f"{n_splits} folds for tiny dataset ({num_cases} cases)."
    )


def prepared_dataset_root(nnunet_root: Path, dataset_id: int, dataset_name: str) -> Path:
    dataset_dirname = f"Dataset{dataset_id:03d}_{dataset_name}"
    return nnunet_root / "nnUNet_raw" / dataset_dirname


def has_prepared_dataset(nnunet_root: Path, dataset_id: int, dataset_name: str) -> bool:
    dataset_root = prepared_dataset_root(nnunet_root, dataset_id, dataset_name)
    images_tr = dataset_root / "imagesTr"
    labels_tr = dataset_root / "labelsTr"
    dataset_json = dataset_root / "dataset.json"

    if not dataset_json.exists() or not images_tr.exists() or not labels_tr.exists():
        return False

    has_images = any(images_tr.glob("*_0000.nii.gz"))
    has_labels = any(labels_tr.glob("*.nii.gz"))
    return has_images and has_labels


def default_plans_identifier_for_planner(planner: str) -> str:
    mapping = {
        "ExperimentPlanner": "nnUNetPlans",
        "nnUNetPlannerResEncM": "nnUNetResEncUNetMPlans",
        "nnUNetPlannerResEncL": "nnUNetResEncUNetLPlans",
        "nnUNetPlannerResEncXL": "nnUNetResEncUNetXLPlans",
    }
    return mapping.get(planner, "nnUNetPlans")


def resolve_plans_identifier(explicit_value: str | None, planner: str) -> str:
    if explicit_value:
        return explicit_value
    return default_plans_identifier_for_planner(planner)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="nnU-Net v2 3D pipeline for Dataset001")
    parser.add_argument("--nnunet-root", type=Path, default=Path("src/nn_unet/nnunet_data"))
    parser.add_argument("--dataset-id", type=int, default=1)
    parser.add_argument("--dataset-name", default="BPWoodDefects")

    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare", help="Convert Dataset001 to nnU-Net raw format")
    prep.add_argument("--source", type=Path, default=Path("datasets/Dataset001"))
    prep.add_argument("--geometry-root", type=Path, default=Path("src/png"))
    prep.add_argument("--overwrite", action="store_true")

    plan = subparsers.add_parser("plan", help="Run nnU-Net planning and preprocessing")
    plan.add_argument("--verify-dataset-integrity", action="store_true")
    plan.add_argument(
        "--planner",
        default="nnUNetPlannerResEncL",
        help="Experiment planner class (recommended: nnUNetPlannerResEncL)",
    )
    plan.add_argument(
        "--plans-identifier",
        default=None,
        help="Custom plans identifier; defaults based on selected planner",
    )
    plan.add_argument(
        "--configurations",
        nargs="+",
        default=None,
        help="Optional nnU-Net configs to preprocess (for example: 3d_lowres 3d_fullres)",
    )
    plan.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Optional preprocessing worker count (-np). Use 1 on low-memory machines.",
    )

    train = subparsers.add_parser("train", help="Train nnU-Net model")
    train.add_argument("--configuration", default="3d_fullres")
    train.add_argument("--fold", default="0", help="Fold index or 'all'")
    train.add_argument(
        "--plans-identifier",
        default="nnUNetResEncUNetLPlans",
        help="Plans identifier to use during training (use nnUNetPlans for old planner)",
    )
    train.add_argument("--trainer", default=None, help="Optional custom trainer class name")

    predict = subparsers.add_parser("predict", help="Run nnU-Net inference")
    predict.add_argument("--input", type=Path, required=True, help="Folder with *_0000.nii.gz inputs")
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--configuration", default="3d_fullres")
    predict.add_argument("--fold", default="0", help="Fold index or 'all'")

    all_cmd = subparsers.add_parser("all", help="Prepare + plan + train in one command")
    all_cmd.add_argument("--source", type=Path, default=Path("datasets/Dataset001"))
    all_cmd.add_argument("--geometry-root", type=Path, default=Path("src/png"))
    all_cmd.add_argument("--overwrite", action="store_true")
    all_cmd.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip prepare step and reuse existing nnUNet_raw dataset",
    )
    all_cmd.add_argument(
        "--planner",
        default="nnUNetPlannerResEncL",
        help="Experiment planner class used in all command",
    )
    all_cmd.add_argument(
        "--plans-identifier",
        default=None,
        help="Plans identifier for all command; defaults based on selected planner",
    )
    all_cmd.add_argument(
        "--configuration",
        default=None,
        help=(
            "Training configuration for the all command. If omitted and exactly one "
            "--plan-configurations value is given, that value is used. "
            "Otherwise defaults to 3d_fullres."
        ),
    )
    all_cmd.add_argument("--fold", default="0")
    all_cmd.add_argument(
        "--plan-configurations",
        nargs="+",
        default=None,
        help="Optional nnU-Net configs to preprocess before training",
    )
    all_cmd.add_argument(
        "--plan-num-processes",
        type=int,
        default=None,
        help="Optional preprocessing worker count for plan step in all command",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    env = ensure_env(args.nnunet_root)

    step_labels = {
        "prepare": ["prepare dataset"],
        "plan": ["plan + preprocess"],
        "train": ["train model"],
        "predict": ["predict"],
        "all": ["prepare dataset", "plan + preprocess", "train model"],
    }
    selected_steps = step_labels[args.command]
    log_status(
        f"nnU-Net command '{args.command}' for Dataset{args.dataset_id:03d}_{args.dataset_name}"
    )
    log_status(f"Planned steps: {', '.join(selected_steps)}")
    pipeline_started = time.perf_counter()
    log_loading_bar(0, len(selected_steps), "Queued")

    step_idx = 0
    completed_steps = 0

    def next_label(base: str) -> str:
        nonlocal step_idx
        step_idx += 1
        return f"[{step_idx}/{len(selected_steps)}] {base}"

    def start_step(label: str) -> float:
        log_loading_bar(completed_steps, len(selected_steps), f"Running {label}")
        log_status(f"Starting {label}")
        return time.perf_counter()

    def finish_step(label: str, started: float) -> None:
        nonlocal completed_steps
        completed_steps += 1
        log_status(f"Finished {label} in {_fmt_elapsed(time.perf_counter() - started)}")
        log_loading_bar(completed_steps, len(selected_steps), f"Completed {label}")

    if args.command in {"prepare", "all"}:
        label = next_label("prepare dataset")
        should_skip_prepare = args.command == "all" and getattr(args, "skip_prepare", False)
        if should_skip_prepare:
            if not has_prepared_dataset(args.nnunet_root, args.dataset_id, args.dataset_name):
                dataset_root = prepared_dataset_root(args.nnunet_root, args.dataset_id, args.dataset_name)
                raise RuntimeError(
                    "--skip-prepare was set, but prepared dataset is missing or incomplete at "
                    f"{dataset_root}. Run prepare first."
                )
            log_loading_bar(completed_steps, len(selected_steps), f"Skipping {label}")
            log_status(f"Skipped {label} (reusing existing prepared dataset)")
            completed_steps += 1
            log_loading_bar(completed_steps, len(selected_steps), f"Completed {label} (skipped)")
        else:
            started = start_step(label)
            from prepare_dataset001 import prepare_dataset

            prepare_dataset(
                source_root=args.source,
                nnunet_root=args.nnunet_root,
                geometry_root=args.geometry_root,
                dataset_id=args.dataset_id,
                dataset_name=args.dataset_name,
                overwrite=args.overwrite,
            )
            finish_step(label, started)

    if args.command in {"plan", "all"}:
        cmd = ["nnUNetv2_plan_and_preprocess", "-d", str(args.dataset_id)]
        if getattr(args, "verify_dataset_integrity", False):
            cmd.append("--verify_dataset_integrity")
        planner = getattr(args, "planner", "nnUNetPlannerResEncL")
        plans_identifier = resolve_plans_identifier(getattr(args, "plans_identifier", None), planner)
        cmd.extend(["-pl", planner])
        if plans_identifier != "nnUNetPlans":
            cmd.extend(["-overwrite_plans_name", plans_identifier])
        if args.command == "plan":
            configs = args.configurations
            num_processes = args.num_processes
        else:
            configs = args.plan_configurations
            num_processes = args.plan_num_processes

        if configs:
            cmd.extend(["-c", *configs])
        if num_processes is not None:
            cmd.extend(["-np", str(num_processes)])
        label = next_label("plan + preprocess")
        started = start_step(label)
        run_cmd(cmd, env, label)
        finish_step(label, started)

    if args.command in {"train", "all"}:
        planner = getattr(args, "planner", "nnUNetPlannerResEncL")
        plans_identifier = resolve_plans_identifier(getattr(args, "plans_identifier", None), planner)
        configuration = args.configuration
        if args.command == "all" and configuration is None:
            if args.plan_configurations and len(args.plan_configurations) == 1:
                configuration = args.plan_configurations[0]
            else:
                configuration = "3d_fullres"

        ensure_crossval_splits(
            nnunet_root=args.nnunet_root,
            dataset_id=args.dataset_id,
            dataset_name=args.dataset_name,
            configuration=configuration,
            plans_identifier=plans_identifier,
        )

        trainer_args: List[str] = []
        trainer = getattr(args, "trainer", None)
        if trainer:
            trainer_args = ["-tr", trainer]
        cmd = [
            "nnUNetv2_train",
            str(args.dataset_id),
            configuration,
            str(args.fold),
            "-p",
            plans_identifier,
            *trainer_args,
        ]
        label = next_label("train model")
        started = start_step(label)
        run_cmd(cmd, env, label)
        finish_step(label, started)

    if args.command == "predict":
        args.output.mkdir(parents=True, exist_ok=True)
        cmd = [
            "nnUNetv2_predict",
            "-i",
            str(args.input),
            "-o",
            str(args.output),
            "-d",
            str(args.dataset_id),
            "-c",
            args.configuration,
            "-f",
            str(args.fold),
        ]
        label = next_label("predict")
        started = start_step(label)
        run_cmd(cmd, env, label)
        finish_step(label, started)

    log_status(f"All done in {_fmt_elapsed(time.perf_counter() - pipeline_started)}")


if __name__ == "__main__":
    main()
