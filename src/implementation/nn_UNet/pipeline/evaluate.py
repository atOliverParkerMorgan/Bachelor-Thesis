from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import SimpleITK as sitk

from .env import log
from .utils import normalize_case_id


def _labels_by_id(dataset_json_path: Path) -> dict[int, str]:
    with dataset_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    labels = data.get("labels", {})
    result: dict[int, str] = {}
    for key, value in labels.items():
        if isinstance(value, (int, str)):
            try:
                result[int(value)] = str(key)
                continue
            except (TypeError, ValueError):
                pass
        try:
            result[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    if not result:
        raise RuntimeError(f"Failed to parse labels from {dataset_json_path}")
    return dict(sorted(result.items()))


def evaluate_prediction_folder(
    prediction_dir: Path,
    labels_dir: Path,
    dataset_json_path: Path,
    metrics_json_path: Path | None = None,
) -> dict:
    if not prediction_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {prediction_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    labels_by_id = _labels_by_id(dataset_json_path)
    class_ids = sorted(labels_by_id)

    gt_files = sorted(labels_dir.glob("*.nii.gz"))
    if not gt_files:
        raise RuntimeError(f"No ground-truth NIfTI files found in {labels_dir}")

    counts: dict[int, dict[str, int]] = {c: {"tp": 0, "fp": 0, "fn": 0} for c in class_ids}
    matched = 0

    for gt_file in gt_files:
        case_id = normalize_case_id(gt_file.name)
        pred_file = prediction_dir / f"{case_id}.nii.gz"
        if not pred_file.exists():
            raise FileNotFoundError(f"Missing prediction for '{case_id}': expected {pred_file}")

        gt_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_file))).astype(np.int16)
        pred_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_file))).astype(np.int16)
        if gt_arr.shape != pred_arr.shape:
            raise RuntimeError(f"Shape mismatch for {case_id}: gt={gt_arr.shape}, pred={pred_arr.shape}")

        matched += 1
        for c in class_ids:
            gm = gt_arr == c
            pm = pred_arr == c
            counts[c]["tp"] += int(np.logical_and(gm, pm).sum())
            counts[c]["fp"] += int(np.logical_and(~gm, pm).sum())
            counts[c]["fn"] += int(np.logical_and(gm, ~pm).sum())

    per_class = []
    for c in class_ids:
        tp, fp, fn = counts[c]["tp"], counts[c]["fp"], counts[c]["fn"]
        dice_den, iou_den = 2 * tp + fp + fn, tp + fp + fn
        per_class.append({
            "class_id": c,
            "class_name": labels_by_id[c],
            "dice": (2.0 * tp / dice_den) if dice_den > 0 else None,
            "iou": (tp / iou_den) if iou_den > 0 else None,
            "tp": tp, "fp": fp, "fn": fn,
        })

    fg = [m for m in per_class if m["class_id"] != 0]
    overall = {
        "mean_dice_all_classes": _mean_or_none([m["dice"] for m in per_class]),
        "miou_all_classes": _mean_or_none([m["iou"] for m in per_class]),
        "mean_dice_foreground": _mean_or_none([m["dice"] for m in fg]),
        "miou_foreground": _mean_or_none([m["iou"] for m in fg]),
        "evaluated_cases": matched,
    }

    result = {
        "prediction_dir": str(prediction_dir),
        "labels_dir": str(labels_dir),
        "dataset_json": str(dataset_json_path),
        "overall": overall,
        "per_class": per_class,
    }

    log(f"Evaluation: fg_dice={overall['mean_dice_foreground']}, fg_miou={overall['miou_foreground']}, cases={matched}")
    for m in per_class:
        log(f"  class {m['class_id']} ({m['class_name']}): dice={m['dice']}, iou={m['iou']}")

    if metrics_json_path is not None:
        metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_json_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        log(f"Saved metrics to {metrics_json_path}")

    return result


def compute_nnunet_summary(
    pred_dir: Path,
    gt_dir: Path,
    num_classes: int = 7,
    output_json: Path | None = None,
) -> dict:
    import nibabel as nib
    from nibabel.orientations import io_orientation, ornt_transform, apply_orientation

    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    if not pred_files:
        raise RuntimeError(f"No prediction NIfTI files found in {pred_dir}")

    metric_names = ["Dice", "FN", "FP", "IoU", "TN", "TP", "n_pred", "n_ref"]
    class_keys = [str(c) for c in range(1, num_classes)]
    metric_per_case = []

    for pred_file in pred_files:
        gt_file = gt_dir / pred_file.name
        if not gt_file.exists():
            raise FileNotFoundError(f"No ground-truth for '{pred_file.name}' in {gt_dir}")

        gt_nib = nib.load(str(gt_file))
        pred_nib = nib.load(str(pred_file))
        gt_arr = np.asarray(gt_nib.dataobj).astype(np.int16)
        pred_arr = np.asarray(pred_nib.dataobj).astype(np.int16)

        # Predictions are saved in RAS; GT may be in a different orientation.
        gt_ornt = io_orientation(gt_nib.affine)
        pred_ornt = io_orientation(pred_nib.affine)
        if not np.array_equal(gt_ornt, pred_ornt):
            pred_arr = apply_orientation(pred_arr, ornt_transform(pred_ornt, gt_ornt)).astype(np.int16)
            log(f"  Reoriented prediction: {pred_file.name}")

        if pred_arr.shape != gt_arr.shape:
            raise RuntimeError(f"Shape mismatch for {pred_file.name}: pred={pred_arr.shape}, gt={gt_arr.shape}")

        total = int(pred_arr.size)
        case_metrics: dict[str, dict] = {}
        for c in range(1, num_classes):
            pm, gm = pred_arr == c, gt_arr == c
            tp = int(np.logical_and(pm, gm).sum())
            fp = int(np.logical_and(pm, ~gm).sum())
            fn = int(np.logical_and(~pm, gm).sum())
            tn = total - tp - fp - fn
            dice_den, iou_den = 2 * tp + fp + fn, tp + fp + fn
            case_metrics[str(c)] = {
                "Dice": (2.0 * tp / dice_den) if dice_den > 0 else 0.0,
                "FN": fn, "FP": fp,
                "IoU": (float(tp) / iou_den) if iou_den > 0 else 0.0,
                "TN": tn, "TP": tp,
                "n_pred": tp + fp, "n_ref": tp + fn,
            }
        metric_per_case.append({
            "metrics": case_metrics,
            "prediction_file": str(pred_file),
            "reference_file": str(gt_file),
        })

    mean = {
        c: {k: float(np.mean([case["metrics"][c][k] for case in metric_per_case])) for k in metric_names}
        for c in class_keys
    }
    foreground_mean = {k: float(np.mean([mean[c][k] for c in class_keys])) for k in metric_names}

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


def run_custom_evaluate(args, _env: Dict) -> None:
    output = args.output or (Path(args.pred_dir) / "summary.json")
    compute_nnunet_summary(
        pred_dir=Path(args.pred_dir),
        gt_dir=Path(args.gt_dir),
        num_classes=args.num_classes,
        output_json=output,
    )


def _mean_or_none(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None
