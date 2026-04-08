#!/usr/bin/env python3
"""Minimal command wrappers for nnU-Net v2 on Dataset001."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Dict, List

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
# Aligned to your newly established project root
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

    save_every = getattr(args, "save_every", None)
    if save_every is not None:
        env["NNUNET_SAVE_EVERY"] = str(save_every)

    initial_lr = getattr(args, "initial_lr", None)
    if initial_lr is not None:
        env["NNUNET_INITIAL_LR"] = str(initial_lr)

    if getattr(args, "skip_arch_plot", False):
        env["NNUNET_SKIP_ARCH_PLOT"] = "1"

    compile_mode = getattr(args, "compile", None)
    if compile_mode == "off":
        env["nnUNet_compile"] = "0"
    elif compile_mode == "on":
        env["nnUNet_compile"] = "1"

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
    plan.add_argument("--verify-dataset-integrity", action="store_true")
    plan.add_argument("--resenc-preset", choices=["M", "L", "XL"], default="L")
    plan.add_argument("--planner", default=None)
    plan.add_argument("--plans-identifier", default=None)
    plan.add_argument("--configurations", nargs="+", default=None)
    plan.add_argument("--num-processes", type=int, default=None)
    add_clusterfit_arguments(plan)

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
    add_clusterfit_arguments(train)

    predict = subparsers.add_parser("predict", help="Run nnU-Net inference")
    predict.add_argument("--input", type=Path, required=True, help="Folder with *_0000.nii.gz inputs")
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--configuration", default="3d_fullres")
    predict.add_argument("--fold", default="0", help="Fold index or 'all'")
    add_hidden_legacy_planner_args(predict)
    predict.add_argument("--plans-identifier", default=None)
    add_clusterfit_arguments(predict)

    # --- NEW: predict-tree command added here ---
    predict_tree = subparsers.add_parser("predict-tree", help="Run whole-tree inference and export to Datumaro")
    predict_tree.add_argument("--tree", required=True, help="Tree name (e.g., DUB_4)")
    predict_tree.add_argument("--ground-truth-root", type=Path, default=Path("src/ground_truth"))
    predict_tree.add_argument("--segmentation-output-root", type=Path, required=True)
    predict_tree.add_argument("--configuration", default="3d_fullres")
    predict_tree.add_argument("--fold", default="0", help="Fold index")
    predict_tree.add_argument("--make-datumaro", action="store_true", help="Convert NIfTI outputs to Datumaro format")
    add_hidden_legacy_planner_args(predict_tree)
    predict_tree.add_argument("--plans-identifier", default=None)
    add_clusterfit_arguments(predict_tree)
    # --------------------------------------------

    all_cmd = subparsers.add_parser("all", help="Prepare + plan + train")
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
    all_cmd.add_argument("--initial-lr", type=float, default=None)
    all_cmd.add_argument("--compile", choices=["auto", "on", "off"], default="auto")
    all_cmd.add_argument("--n-proc-da", type=int, default=4)
    all_cmd.add_argument("--cpu-threads", type=int, default=1)
    all_cmd.add_argument("--continue-training", action="store_true")
    all_cmd.add_argument("--plan-configurations", nargs="+", default=["3d_fullres"])
    all_cmd.add_argument("--plan-num-processes", type=int, default=1)
    add_clusterfit_arguments(all_cmd)

    return parser


def run_prepare(args: argparse.Namespace) -> None:
    """Uses the new segmask2ima pipeline to automatically process all logs in ground_truth."""
    from src.processing.conversion.segmask2ima import process_tree
    
    zip_files = list(args.source.glob("*.zip"))
    if not zip_files:
        log(f"No DICOM zip files found in {args.source}. Make sure your raw zips are there.")
        return
        
    for zip_file in zip_files:
        tree_name = zip_file.stem  # e.g., 'DUB_5' -> 'dub_5'
        log(f"Auto-processing dataset for: {tree_name}")
        process_tree(tree_name)


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
        "-i", str(active_input),
        "-o", str(args.output),
        "-d", str(args.dataset_id),
        "-c", args.configuration,
        "-f", str(args.fold),
        "-p", plans_identifier,
    ]
    if checkpoint_name is not None:
        cmd.extend(["-chk", checkpoint_name])
        log(f"Using prediction checkpoint: {checkpoint_name}")

    vram_gb = detect_gpu_vram_gb()
    npp, nps = prediction_worker_profile(vram_gb)
    cmd.extend(["-npp", str(npp), "-nps", str(nps)])
    
    if vram_gb is not None:
        log(f"Fast predict profile: disable_tta, npp={npp}, nps={nps} (GPU VRAM ~{vram_gb:.1f} GB)")
    else:
        log(f"Fast predict profile: disable_tta, npp={npp}, nps={nps} (GPU VRAM unknown)")
    
    try:
        run_cmd(cmd, env, "predict")
    finally:
        if temp_input_dir and temp_input_dir.exists():
            shutil.rmtree(temp_input_dir, ignore_errors=True)
            log("Cleaned up temporary NIfTI inputs.")


# --- NEW: Function to handle the full tree logic ---
def run_predict_tree(args: argparse.Namespace, env: Dict[str, str]) -> None:
    from src.nn_UNet.tree_inference_helpers import (
        prepare_png_tree_from_ground_truth,
        write_tree_inference_nifti,
        export_prediction_masks,
        export_datumaro_for_tree,
    )

    tree_name = args.tree
    tree_slug = tree_name.lower().replace(" ", "_")
    tree_output_root = args.segmentation_output_root / tree_slug
    tree_segmentation_output = tree_output_root / "segmentation_style"
    tree_nifti_output = tree_output_root / "nnunet_nifti_predictions"
    tree_output_root.mkdir(parents=True, exist_ok=True)

    dataset_json_path = args.nnunet_root / "nnUNet_raw" / f"Dataset{args.dataset_id:03d}_{args.dataset_name}" / "dataset.json"

    # Create temporary directories for the conversion pipeline
    temp_dir = Path(tempfile.mkdtemp(prefix=f"nnunet_tree_{tree_name}_"))
    png_root = temp_dir / "pngs"
    nifti_in_dir = temp_dir / "nifti_in"
    nifti_out_dir = temp_dir / "nifti_out"
    nifti_in_dir.mkdir(parents=True, exist_ok=True)
    nifti_out_dir.mkdir(parents=True, exist_ok=True)

    try:
        log(f"Preparing PNGs for {tree_name}...")
        tree_dir = prepare_png_tree_from_ground_truth(
            tree_name=tree_name,
            png_root=png_root,
            ground_truth_root=args.ground_truth_root,
            temp_root=temp_dir
        )

        log("Converting slices to 3D NIfTI for nnU-Net inference...")
        is_3d = "3d" in args.configuration.lower()
        write_tree_inference_nifti(tree_dir, nifti_in_dir, tree_name, is_3d)

        # Build the standard nnUNet predict command
        plans_identifier = resolve_plans_identifier(args)
        checkpoint_name = resolve_prediction_checkpoint(
            nnunet_root=args.nnunet_root, dataset_id=args.dataset_id,
            dataset_name=args.dataset_name, configuration=args.configuration,
            plans_identifier=plans_identifier, fold=str(args.fold)
        )

        cmd = [
            "nnUNetv2_predict",
            "-i", str(nifti_in_dir), "-o", str(nifti_out_dir),
            "-d", str(args.dataset_id), "-c", args.configuration,
            "-f", str(args.fold), "-p", plans_identifier,
        ]
        if checkpoint_name:
            cmd.extend(["-chk", checkpoint_name])

        log("Running nnU-Net prediction...")
        run_cmd(cmd, env, "predict-tree")

        # Datumaro Export Phase
        if args.make_datumaro:
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

def submit_to_clusterfit(args: argparse.Namespace, env: Dict[str, str]) -> None:
    SlurmJobSubmitter, _, build_slurm_config_from_args = import_clusterfit_helpers()
    
    if SlurmJobSubmitter is None:
        raise RuntimeError("ClusterFIT utilities are missing. Cannot submit to Slurm from this environment.")

    slurm_env: Dict[str, str] = {}
    for key in (
        "PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "VIRTUAL_ENV", 
        "PYTHONPATH", "PYTHONUNBUFFERED", "nnUNet_raw", "nnUNet_preprocessed", 
        "nnUNet_results", "NNUNET_SAVE_EVERY", "NNUNET_INITIAL_LR", 
        "NNUNET_SKIP_ARCH_PLOT", "nnUNet_compile", "nnUNet_n_proc_DA", 
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
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

    if args.command == "predict":
        run_predict(args, env)

    # --- NEW: predict-tree trigger ---
    if args.command == "predict-tree":
        run_predict_tree(args, env)
    # ---------------------------------

    log(f"Done in {int(time.perf_counter() - started)}s")


if __name__ == "__main__":
    main()