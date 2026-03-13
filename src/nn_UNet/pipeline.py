#!/usr/bin/env python3
"""Minimal command wrappers for nnU-Net v2 on Dataset001."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

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
    save_every = getattr(args, "save_every", None)
    if save_every is not None:
        env["NNUNET_SAVE_EVERY"] = str(save_every)

    initial_lr = getattr(args, "initial_lr", None)
    if initial_lr is not None:
        env["NNUNET_INITIAL_LR"] = str(initial_lr)

    if getattr(args, "skip_arch_plot", False):
        env["NNUNET_SKIP_ARCH_PLOT"] = "1"


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
            log(
                "Using detected legacy plans 'nnUNetPlans' because preferred "
                f"'{preferred}' is missing."
            )
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
    """Create splits_final.json for tiny datasets so nnU-Net does not force 5-fold CV."""
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
    if num_cases == 0 or num_cases >= 5:
        return
    if num_cases < 2:
        raise RuntimeError(
            "Training requires at least 2 cases. Found only "
            f"{num_cases} case in {config_dir}."
        )

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


def build_parser() -> argparse.ArgumentParser:
    def add_hidden_legacy_planner_args(subparser: argparse.ArgumentParser) -> None:
        # Keep backward compatibility for scripts that still pass these options,
        # but remove them from normal help output to keep CLI lean.
        subparser.add_argument("--resenc-preset", choices=["M", "L", "XL"], default=None, help=argparse.SUPPRESS)
        subparser.add_argument("--planner", default=None, help=argparse.SUPPRESS)

    parser = argparse.ArgumentParser(description="nnU-Net v2 pipeline for Dataset001")
    parser.add_argument("--nnunet-root", type=Path, default=Path("src/nn_UNet/nnunet_data"))
    parser.add_argument("--dataset-id", type=int, default=1)
    parser.add_argument("--dataset-name", default="BPWoodDefects")

    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare", help="Convert Dataset001 to nnU-Net raw format")
    prep.add_argument("--source", type=Path, default=Path("datasets/Dataset001"))
    prep.add_argument("--geometry-root", type=Path, default=Path("src/png"))
    prep.add_argument("--overwrite", action="store_true")

    plan = subparsers.add_parser("plan", help="Run nnU-Net planning and preprocessing")
    plan.add_argument("--verify-dataset-integrity", action="store_true")
    plan.add_argument("--resenc-preset", choices=["M", "L", "XL"], default="L")
    plan.add_argument("--planner", default=None)
    plan.add_argument("--plans-identifier", default=None)
    plan.add_argument("--configurations", nargs="+", default=None)
    plan.add_argument("--num-processes", type=int, default=None)

    train = subparsers.add_parser("train", help="Train nnU-Net model")
    train.add_argument("--configuration", default="3d_fullres")
    train.add_argument("--fold", default="0", help="Fold index or 'all'")
    add_hidden_legacy_planner_args(train)
    train.add_argument("--plans-identifier", default=None)
    train.add_argument("--trainer", default=None, help="Optional custom trainer class name")
    train.add_argument("--save-every", type=int, default=10)
    train.add_argument("--skip-arch-plot", action="store_true")
    train.add_argument("--initial-lr", type=float, default=None)
    train.add_argument(
        "--continue-training",
        action="store_true",
        help="Resume training from checkpoint_latest.pth (nnUNetv2_train --c)",
    )

    predict = subparsers.add_parser("predict", help="Run nnU-Net inference")
    predict.add_argument("--input", type=Path, required=True, help="Folder with *_0000.nii.gz inputs")
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--configuration", default="3d_fullres")
    predict.add_argument("--fold", default="0", help="Fold index or 'all'")
    add_hidden_legacy_planner_args(predict)
    predict.add_argument("--plans-identifier", default=None)

    predict_tree = subparsers.add_parser(
        "predict-tree",
        help="Predict one PNG tree, convert masks to Datumaro, and optionally upload to CVAT",
    )
    predict_tree.add_argument("--tree", required=True, help="Tree folder name under src/png, for example dub5")
    predict_tree.add_argument("--input-root", type=Path, default=Path("src/png"))
    predict_tree.add_argument("--ground-truth-root", type=Path, default=Path("src/ground_truth"))
    predict_tree.add_argument("--segmentation-output-root", type=Path, default=Path("src/output"))
    predict_tree.add_argument("--temp-root", type=Path, default=Path("src/nn_UNet/nnunet_data/tree_inference"))
    predict_tree.add_argument("--configuration", default="2d")
    predict_tree.add_argument("--fold", default="0", help="Fold index or 'all'")
    predict_tree.add_argument("--plans-identifier", default=None)
    predict_tree.add_argument("--make-datumaro", action="store_true", help="Export the predicted masks to Datumaro zip")
    predict_tree.add_argument("--datumaro-output", type=Path, default=None)
    predict_tree.add_argument("--upload-cvat", action="store_true", help="Upload the Datumaro zip to CVAT after export")
    predict_tree.add_argument("--organization", default=None, help="Optional CVAT organization slug override")
    predict_tree.add_argument("--keep-temp", action="store_true", help="Keep temporary NIfTI inputs and predictions")
    add_hidden_legacy_planner_args(predict_tree)

    all_cmd = subparsers.add_parser("all", help="Prepare + plan + train")
    all_cmd.add_argument("--source", type=Path, default=Path("datasets/Dataset001"))
    all_cmd.add_argument("--geometry-root", type=Path, default=Path("src/png"))
    all_cmd.add_argument("--overwrite", action="store_true")
    all_cmd.add_argument("--skip-prepare", action="store_true")
    add_hidden_legacy_planner_args(all_cmd)
    all_cmd.add_argument("--plans-identifier", default=None)
    all_cmd.add_argument("--configuration", default=None)
    all_cmd.add_argument("--fold", default="0")
    all_cmd.add_argument("--save-every", type=int, default=10)
    all_cmd.add_argument("--skip-arch-plot", action="store_true")
    all_cmd.add_argument("--initial-lr", type=float, default=None)
    all_cmd.add_argument("--continue-training", action="store_true")
    all_cmd.add_argument("--plan-configurations", nargs="+", default=["3d_fullres"])
    all_cmd.add_argument("--plan-num-processes", type=int, default=1)

    return parser


def run_prepare(args: argparse.Namespace) -> None:
    from src.preprocessing.conversion.segmentmask2nnunetformat import prepare_dataset

    prepare_dataset(
        source_root=args.source,
        nnunet_root=args.nnunet_root,
        geometry_root=args.geometry_root,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        overwrite=args.overwrite,
    )


def run_plan(args: argparse.Namespace, env: Dict[str, str]) -> None:
    planner = planner_from_args(args)
    plans_identifier = args.plans_identifier or default_plans_for_planner(planner)
    cmd = build_plan_command(args, planner, plans_identifier)
    run_cmd(cmd, env, "plan + preprocess")


def run_train(args: argparse.Namespace, env: Dict[str, str]) -> None:
    plans_identifier = resolve_plans_identifier(args)
    configuration = resolve_train_configuration(args)

    ensure_crossval_splits(
        nnunet_root=args.nnunet_root,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        configuration=configuration,
        plans_identifier=plans_identifier,
    )

    cmd = [
        "nnUNetv2_train",
        str(args.dataset_id),
        configuration,
        str(args.fold),
        "-p",
        plans_identifier,
    ]
    if args.trainer:
        cmd.extend(["-tr", args.trainer])
    if args.continue_training:
        cmd.append("--c")

    run_cmd(cmd, env, "train")


def run_predict(args: argparse.Namespace, env: Dict[str, str]) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    plans_identifier = resolve_plans_identifier(args)
    checkpoint_name = resolve_prediction_checkpoint(
        nnunet_root=args.nnunet_root,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        configuration=args.configuration,
        plans_identifier=plans_identifier,
        fold=str(args.fold),
    )

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
        "-p",
        plans_identifier,
    ]
    if checkpoint_name is not None:
        cmd.extend(["-chk", checkpoint_name])
        log(f"Using prediction checkpoint: {checkpoint_name}")
    run_cmd(cmd, env, "predict")


def run_predict_tree(args: argparse.Namespace, env: Dict[str, str]) -> None:
    from src.nn_UNet.prepare_dataset_to_nnunet_format import (
        default_datumaro_output,
        export_datumaro_for_tree,
        export_prediction_masks,
        prepare_png_tree_from_ground_truth,
        upload_tree_datumaro,
        write_tree_slices_nifti,
    )

    tree_dir = prepare_png_tree_from_ground_truth(
        tree_name=args.tree,
        png_root=args.input_root,
        ground_truth_root=args.ground_truth_root,
        temp_root=args.temp_root,
    )

    temp_case_dir = args.temp_root / args.tree / "input"
    temp_prediction_dir = args.temp_root / args.tree / "prediction"
    segmentation_output_dir = args.segmentation_output_root / args.tree
    dataset_json_path = prepared_dataset_root(args.nnunet_root, args.dataset_id, args.dataset_name) / "dataset.json"

    if not dataset_json_path.exists():
        raise FileNotFoundError(
            "dataset.json for label mapping was not found at "
            f"{dataset_json_path}. Run prepare first or point --nnunet-root to the prepared dataset."
        )

    write_tree_slices_nifti(tree_dir, temp_case_dir)

    temp_prediction_dir.mkdir(parents=True, exist_ok=True)
    plans_identifier = resolve_plans_identifier(args)
    checkpoint_name = resolve_prediction_checkpoint(
        nnunet_root=args.nnunet_root,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        configuration=args.configuration,
        plans_identifier=plans_identifier,
        fold=str(args.fold),
    )
    cmd = [
        "nnUNetv2_predict",
        "-i",
        str(temp_case_dir),
        "-o",
        str(temp_prediction_dir),
        "-d",
        str(args.dataset_id),
        "-c",
        args.configuration,
        "-f",
        str(args.fold),
        "-p",
        plans_identifier,
    ]
    if checkpoint_name is not None:
        cmd.extend(["-chk", checkpoint_name])
        log(f"Using prediction checkpoint: {checkpoint_name}")
    run_cmd(cmd, env, "predict-tree")

    export_prediction_masks(
        prediction_dir=temp_prediction_dir,
        tree_dir=tree_dir,
        segmentation_output_dir=segmentation_output_dir,
        dataset_json_path=dataset_json_path,
    )

    should_export_datumaro = args.make_datumaro or args.upload_cvat
    datumaro_zip = args.datumaro_output or default_datumaro_output(segmentation_output_dir, args.tree)
    if should_export_datumaro:
        export_datumaro_for_tree(segmentation_output_dir, datumaro_zip, args.tree)
    if args.upload_cvat:
        upload_tree_datumaro(datumaro_zip, organization=args.organization)

    if not args.keep_temp and args.temp_root.exists():
        shutil.rmtree(args.temp_root / args.tree, ignore_errors=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    env = ensure_env(args.nnunet_root)
    apply_runtime_env_overrides(env, args)

    started = time.perf_counter()
    log(f"Running command: {args.command}")

    if args.command in {"prepare", "all"}:
        if args.command == "all" and args.skip_prepare:
            if not has_prepared_dataset(args.nnunet_root, args.dataset_id, args.dataset_name):
                dataset_root = prepared_dataset_root(args.nnunet_root, args.dataset_id, args.dataset_name)
                raise RuntimeError(
                    "--skip-prepare was set, but prepared dataset is missing or incomplete at "
                    f"{dataset_root}."
                )
            log("Skipping prepare step (reusing existing prepared dataset).")
        else:
            run_prepare(args)

    if args.command in {"plan", "all"}:
        run_plan(args, env)

    if args.command in {"train", "all"}:
        run_train(args, env)

    if args.command == "predict":
        run_predict(args, env)
    if args.command == "predict-tree":
        run_predict_tree(args, env)

    log(f"Done in {int(time.perf_counter() - started)}s")


if __name__ == "__main__":
    main()
