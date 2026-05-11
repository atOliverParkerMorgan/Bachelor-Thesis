from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np

from .env import log, PROJECT_ROOT
from .utils import run_cmd, detect_gpu_vram_gb, prediction_worker_profile, convert_dicom_zip_to_nifti, normalize_case_id
from .plans import (
    resolve_plans_identifier,
    resolve_train_configuration,
    apply_plan_regularization_overrides,
    ensure_crossval_splits,
    has_prepared_dataset,
    prepared_dataset_root,
    build_plan_command,
    resolve_prediction_trainer,
    resolve_prediction_checkpoint,
    planner_from_args,
    default_plans_for_planner,
)
from .trainers import install_pretrained_trainer, install_wandb_trainer, install_cediceskel_trainer
from .evaluate import evaluate_prediction_folder


def build_holdout_split(
    nnunet_root: Path,
    dataset_id: int,
    dataset_name: str,
    test_tree: str,
) -> tuple:
    raw_dir = nnunet_root / "nnUNet_raw" / f"Dataset{dataset_id:03d}_{dataset_name}" / "imagesTr"
    if not raw_dir.exists():
        raise FileNotFoundError(f"imagesTr not found: {raw_dir}")

    all_ids = sorted({re.sub(r"_\d{4}\.nii\.gz$", "", f.name) for f in raw_dir.glob("*.nii.gz")})
    if not all_ids:
        raise RuntimeError(f"No NIfTI cases found in {raw_dir}")

    prefix = test_tree.lower().replace("-", "_")

    def _is_test(case_id: str) -> bool:
        norm = case_id.lower()
        return norm == prefix or bool(re.match(rf"^{re.escape(prefix)}_part\d+$", norm))

    val_ids = [c for c in all_ids if _is_test(c)]
    train_ids = [c for c in all_ids if not _is_test(c)]

    if not val_ids:
        raise RuntimeError(f"No cases matched '{test_tree}'. Available (first 10): {all_ids[:10]}")
    if not train_ids:
        raise RuntimeError(f"No training cases remain after excluding '{test_tree}'.")

    split = [{"train": train_ids, "val": val_ids}]
    split_path = PROJECT_ROOT / f"splits_test_{prefix}.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

    log(f"Test holdout: {len(train_ids)} train | {len(val_ids)} val (tree={test_tree})")
    log(f"  Val cases: {val_ids}")
    return train_ids, val_ids, split_path


def run_prepare(args) -> None:
    from src.implementation.preprocessing.conversion.segmask2ima import process_tree

    zip_files = list(args.source.glob("*.zip"))
    if not zip_files:
        log(f"No DICOM zip files found in {args.source}.")
        return
    for zip_file in zip_files:
        log(f"Auto-processing: {zip_file.stem}")
        process_tree(zip_file.stem)


def run_plan(args, env: Dict[str, str]) -> None:
    planner = planner_from_args(args)
    plans_identifier = args.plans_identifier or default_plans_for_planner(planner)
    run_cmd(build_plan_command(args, planner, plans_identifier), env, "plan + preprocess")


def run_train(args, env: Dict[str, str]) -> None:
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
        _, _, split_path = build_holdout_split(
            nnunet_root=args.nnunet_root,
            dataset_id=args.dataset_id,
            dataset_name=args.dataset_name,
            test_tree=getattr(args, "test_tree", "dub_2"),
        )
        dataset_dir = args.nnunet_root / "nnUNet_preprocessed" / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
        splits_file = dataset_dir / "splits_final.json"
        if splits_file.exists():
            splits_backup = splits_file.with_name("splits_final.json.testmode_bak")
            shutil.copy2(splits_file, splits_backup)
        splits_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(split_path, splits_file)
        args.fold = "0"
        log(f"Test mode: fold=0, holdout split written to {splits_file}")

    trainer = _resolve_trainer(args, env)

    cmd = [
        "nnUNetv2_train",
        str(args.dataset_id), configuration, str(args.fold),
        "-p", plans_identifier,
    ]
    if trainer:
        cmd.extend(["-tr", trainer])
    elif getattr(args, "pretrained_weights", None):
        cmd.extend(["--pretrained_weights", str(Path(args.pretrained_weights).resolve())])
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


def _resolve_trainer(args, env: Dict[str, str]) -> str | None:
    pretrained = getattr(args, "pretrained_weights", None)
    use_wandb = getattr(args, "wandb", False)
    trainer = getattr(args, "trainer", None)

    if not trainer:
        if pretrained and use_wandb:
            trainer = "nnUNetTrainerRareClassBoostLungPretrainedWandb"
            if not install_wandb_trainer():
                trainer = "nnUNetTrainerLungPretrained"
                install_pretrained_trainer()
        elif pretrained:
            trainer = "nnUNetTrainerLungPretrained"
            if not install_pretrained_trainer():
                trainer = None
        elif use_wandb:
            trainer = "nnUNetTrainerRareClassBoostWandb"
            if not install_wandb_trainer():
                trainer = None

    if trainer == "nnUNetTrainerCeDiceSkel":
        if not install_cediceskel_trainer():
            log("CeDiceSkel trainer install failed — falling back to default.")
            trainer = None

    return trainer


def run_predict(args, env: Dict[str, str]) -> None:
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

    vram_gb = detect_gpu_vram_gb()
    npp, nps = prediction_worker_profile(vram_gb)

    cmd = [
        "nnUNetv2_predict",
        "-i", str(active_input), "-o", str(args.output),
        "-d", str(args.dataset_id), "-c", args.configuration,
        "-f", str(args.fold), "-p", plans_identifier,
        "-tr", trainer, "-npp", str(npp), "-nps", str(nps),
    ]
    if checkpoint_name is not None:
        cmd.extend(["-chk", checkpoint_name])
        log(f"Checkpoint: {checkpoint_name}")
    if getattr(args, "save_probabilities", False):
        cmd.append("--save_probabilities")
        log("Saving softmax probabilities (cascade mode).")
    log(f"Trainer: {trainer}")

    try:
        run_cmd(cmd, env, "predict")

        labels_dir = getattr(args, "labels_dir", None)
        if labels_dir is not None:
            labels_dir = Path(labels_dir).expanduser().resolve()
            dataset_json_path = (
                args.nnunet_root / "nnUNet_raw"
                / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
                / "dataset.json"
            )
            metrics_json = getattr(args, "metrics_json", None)
            metrics_json = (
                Path(metrics_json).expanduser().resolve()
                if metrics_json is not None
                else args.output / "evaluation_metrics.json"
            )
            evaluate_prediction_folder(
                prediction_dir=args.output,
                labels_dir=labels_dir,
                dataset_json_path=dataset_json_path,
                metrics_json_path=metrics_json,
            )
    finally:
        if temp_input_dir and temp_input_dir.exists():
            shutil.rmtree(temp_input_dir, ignore_errors=True)


def run_predict_tree(args, env: Dict[str, str]) -> None:
    from src.implementation.preprocessing.utils.tree_inference_helpers import (
        prepare_png_tree_from_ground_truth,
        write_tree_inference_nifti,
        export_prediction_masks,
        export_datumaro_for_tree,
    )

    tree_name = args.tree
    tree_slug = tree_name.lower().replace(" ", "_")
    ground_truth_root = Path(args.ground_truth_root).expanduser().resolve()
    tree_output_root = args.segmentation_output_root / tree_slug
    tree_segmentation_output = tree_output_root / "segmentation_style"
    tree_nifti_output = tree_output_root / "nnunet_nifti_predictions"
    tree_output_root.mkdir(parents=True, exist_ok=True)

    dataset_json_path = (
        args.nnunet_root / "nnUNet_raw"
        / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
        / "dataset.json"
    )

    temp_dir = Path(tempfile.mkdtemp(prefix=f"nnunet_tree_{tree_name}_"))
    nifti_in_dir = temp_dir / "nifti_in"
    nifti_out_dir = temp_dir / "nifti_out"
    nifti_in_dir.mkdir(parents=True, exist_ok=True)
    nifti_out_dir.mkdir(parents=True, exist_ok=True)

    try:
        is_3d = "3d" in args.configuration.lower()
        prepared_volume = _resolve_prepared_volume(args, tree_slug)
        tree_dir = None
        written_niftis: list = []
        chunk_size = getattr(args, "chunk_size", None)

        if prepared_volume is not None:
            if not prepared_volume.exists():
                raise FileNotFoundError(f"Prepared test volume not found: {prepared_volume}")
            shutil.copy2(prepared_volume, nifti_in_dir / f"{tree_name}_0000.nii.gz")
            log(f"Using prepared volume: {prepared_volume}")
        else:
            log(f"Preparing PNGs for {tree_name}...")
            tree_dir = prepare_png_tree_from_ground_truth(
                tree_name=tree_name, png_root=temp_dir / "pngs",
                ground_truth_root=ground_truth_root, temp_root=temp_dir,
            )
            written_niftis = write_tree_inference_nifti(tree_dir, nifti_in_dir, tree_name, is_3d, chunk_size=chunk_size)
            if len(written_niftis) > 1:
                log(f"Volume split into {len(written_niftis)} chunks of up to {chunk_size} slices.")

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

        vram_gb = detect_gpu_vram_gb()
        npp_auto, nps_auto = prediction_worker_profile(vram_gb)
        npp = getattr(args, "npp", None) or npp_auto
        nps = getattr(args, "nps", None) or nps_auto

        cmd = [
            "nnUNetv2_predict",
            "-i", str(nifti_in_dir), "-o", str(nifti_out_dir),
            "-d", str(args.dataset_id), "-c", args.configuration,
            "-f", str(args.fold), "-p", plans_identifier, "-tr", trainer,
            "-npp", str(npp), "-nps", str(nps),
        ]
        if checkpoint_name:
            cmd.extend(["-chk", checkpoint_name])

        if len(written_niftis) > 1:
            # Run one nnUNet call per chunk so export workers finish and release RAM
            # before the next chunk starts; running all chunks together causes OOM.
            for chunk_nifti in written_niftis:
                chunk_in_dir = temp_dir / f"in_{chunk_nifti.stem}"
                chunk_in_dir.mkdir(parents=True, exist_ok=True)
                _link_or_copy(chunk_nifti, chunk_in_dir / chunk_nifti.name)
                chunk_cmd = list(cmd)
                chunk_cmd[chunk_cmd.index("-i") + 1] = str(chunk_in_dir)
                log(f"Predicting chunk: {chunk_nifti.stem.replace('_0000', '')}")
                run_cmd(chunk_cmd, env, f"predict-tree:{chunk_nifti.stem}")
                shutil.rmtree(chunk_in_dir)
            from src.implementation.preprocessing.utils.tree_inference_helpers import merge_prediction_chunks
            merge_prediction_chunks(nifti_out_dir, tree_name)
            log("Prediction chunks merged.")
        else:
            run_cmd(cmd, env, "predict-tree")

        if args.make_datumaro:
            if tree_dir is None:
                raise RuntimeError(
                    "--make-datumaro requires PNG tree geometry. "
                    "Run without --prepared-volume, or disable --make-datumaro."
                )
            export_prediction_masks(
                prediction_dir=nifti_out_dir, tree_dir=tree_dir,
                segmentation_output_dir=tree_segmentation_output,
                dataset_json_path=dataset_json_path, tree_name=tree_name, is_3d=is_3d,
            )
            datumaro_zip = tree_output_root / f"datumaro_{tree_name}.zip"
            export_datumaro_for_tree(tree_segmentation_output, datumaro_zip, tree_name)
            log(f"Datumaro dataset: {datumaro_zip}")
        else:
            shutil.copytree(nifti_out_dir, tree_nifti_output, dirs_exist_ok=True)
            log(f"NIfTI predictions saved to {tree_nifti_output}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _resolve_prepared_volume(args, tree_slug: str) -> Path | None:
    vol = getattr(args, "prepared_volume", None)
    if vol is not None:
        p = Path(vol).expanduser()
        return (PROJECT_ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    if tree_slug == "dub_2":
        for candidate in [
            PROJECT_ROOT / "BPWoosCLices2" / "dub_2.nii.gz",
            PROJECT_ROOT / "BPWoodSlices2" / "dub_2.nii.gz",
            Path.home() / "BPWoosCLices2" / "dub_2.nii.gz",
            Path.home() / "BPWoodSlices2" / "dub_2.nii.gz",
        ]:
            if candidate.exists():
                return candidate.resolve()
    return None


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.symlink(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def run_cascade_prepare(args) -> None:
    import blosc2
    from scipy.ndimage import zoom as nd_zoom
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

    # Build case_id → target shape from preprocessed lowres files.
    # Prefer .b2nd over .npz; both store [C, D, H, W] so shape[1:] gives spatial dims.
    prep_shapes: dict[str, tuple] = {}
    for f in sorted(preprocessed_dir.iterdir()):
        if f.stem.endswith("_seg"):
            continue
        if f.suffix == ".b2nd" and f.stem not in prep_shapes:
            prep_shapes[f.stem] = blosc2.open(str(f)).shape[1:]
        elif f.suffix == ".npz" and f.stem not in prep_shapes:
            d = np.load(str(f))
            prep_shapes[f.stem] = d[list(d.keys())[0]].shape[1:]

    if not prep_shapes:
        raise RuntimeError(f"No preprocessed lowres cases found in {preprocessed_dir}")
    log(f"Found {len(prep_shapes)} lowres shapes in {preprocessed_dir}")

    converted = skipped = 0
    for case_id, target_shape in sorted(prep_shapes.items()):
        out_b2nd = pred_dir / f"{case_id}.b2nd"
        npz_file = pred_dir / f"{case_id}.npz"
        nii_file = pred_dir / f"{case_id}.nii.gz"

        if npz_file.exists():
            # Softmax from --save_probabilities is already in preprocessed lowres space.
            d = np.load(str(npz_file))
            softmax = d[list(d.keys())[0]]      # [C, D, H, W]
            arr = np.argmax(softmax, axis=0).astype(np.int16)  # [D, H, W]
            if arr.shape != tuple(target_shape):
                factors = [t / s for t, s in zip(target_shape, arr.shape)]
                arr = nd_zoom(arr.astype(float), factors, order=0).astype(np.int16)
            log(f"{case_id}: softmax{softmax.shape} → argmax{arr.shape}")
        elif nii_file.exists():
            # Patient-space argmax — resample to preprocessed lowres shape.
            arr = np.asarray(nib.load(str(nii_file)).dataobj).astype(np.int16)
            if arr.shape != tuple(target_shape):
                factors = [t / s for t, s in zip(target_shape, arr.shape)]
                arr = nd_zoom(arr.astype(float), factors, order=0).astype(np.int16)
            log(f"{case_id}: nii.gz argmax{arr.shape}")
        else:
            log(f"WARNING: no prediction found for {case_id} — skipping")
            skipped += 1
            continue

        if out_b2nd.exists():
            out_b2nd.unlink()
        blosc2.asarray(arr).save(str(out_b2nd))
        converted += 1

    log(f"cascade-prepare: {converted} converted, {skipped} skipped → {pred_dir}")
