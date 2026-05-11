from __future__ import annotations

import argparse
from pathlib import Path

from .env import DEFAULT_NNUNET_ROOT, PROJECT_ROOT, import_clusterfit_helpers


def _args_hidden_planner(p: argparse.ArgumentParser) -> None:
    p.add_argument("--resenc-preset", choices=["M", "L", "XL"], default=None, help=argparse.SUPPRESS)
    p.add_argument("--planner", default=None, help=argparse.SUPPRESS)


def _args_dataset(p: argparse.ArgumentParser) -> None:
    # SUPPRESS means subcommand values only override the root parser default when explicit.
    p.add_argument("--dataset-id", type=int, default=argparse.SUPPRESS)
    p.add_argument("--dataset-name", default=argparse.SUPPRESS)


def _args_wandb(p: argparse.ArgumentParser, default_project: str = "nnunet-training") -> None:
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default=default_project, metavar="PROJECT")
    p.add_argument("--wandb-entity", default=None, metavar="ENTITY")
    p.add_argument("--wandb-run-name", default=None, metavar="NAME")


def _args_regularize_arch(p: argparse.ArgumentParser) -> None:
    p.add_argument("--regularize-arch", action="store_true")
    p.add_argument("--model-dropout-p", type=float, default=0.2)
    p.add_argument("--model-features", type=int, nargs="+", default=None)
    p.add_argument("--model-n-stages", type=int, choices=[5, 6], default=6)
    p.add_argument("--model-batch-size", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    _, add_clusterfit_arguments, _ = import_clusterfit_helpers()

    parser = argparse.ArgumentParser(description="nnU-Net v2 pipeline")
    parser.add_argument("--nnunet-root", type=Path, default=DEFAULT_NNUNET_ROOT)
    parser.add_argument("--dataset-id", type=int, default=1)
    parser.add_argument("--dataset-name", default="BPWoodDefects")

    subs = parser.add_subparsers(dest="command", required=True)

    _build_prepare(subs, add_clusterfit_arguments)
    _build_plan(subs, add_clusterfit_arguments)
    _build_train(subs, add_clusterfit_arguments)
    _build_predict(subs, add_clusterfit_arguments)
    _build_predict_tree(subs, add_clusterfit_arguments)
    _build_cascade_prepare(subs)
    _build_all(subs, add_clusterfit_arguments)
    _build_custom_train(subs, add_clusterfit_arguments)
    _build_custom_predict(subs, add_clusterfit_arguments)
    _build_custom_evaluate(subs, add_clusterfit_arguments)

    return parser


def _build_prepare(subs, add_clusterfit_arguments) -> None:
    p = subs.add_parser("prepare", help="Process ZIPs and CVAT masks into nnU-Net raw format")
    p.add_argument("--source", type=Path, default=Path("src/ground_truth"))
    p.add_argument("--cvat-exports", type=Path, default=Path("src/cvat_exports"))
    p.add_argument("--overwrite", action="store_true")
    add_clusterfit_arguments(p)


def _build_plan(subs, add_clusterfit_arguments) -> None:
    p = subs.add_parser("plan", help="Run nnU-Net planning and preprocessing")
    _args_dataset(p)
    p.add_argument("--verify-dataset-integrity", action="store_true")
    p.add_argument("--resenc-preset", choices=["M", "L", "XL"], default="L")
    p.add_argument("--planner", default=None)
    p.add_argument("--plans-identifier", default=None)
    p.add_argument("--configurations", nargs="+", default=None)
    p.add_argument("--num-processes", type=int, default=None)
    add_clusterfit_arguments(p)


def _build_train(subs, add_clusterfit_arguments) -> None:
    p = subs.add_parser("train", help="Train nnU-Net model")
    _args_dataset(p)
    _args_hidden_planner(p)
    p.add_argument("--configuration", default="3d_fullres")
    p.add_argument("--fold", default="0")
    p.add_argument("--plans-identifier", default=None)
    p.add_argument("--trainer", default=None)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--skip-arch-plot", action="store_true")
    p.add_argument("--initial-lr", type=float, default=1e-3)
    p.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    _args_regularize_arch(p)
    p.add_argument("--patch-size", type=int, nargs=3, default=None, metavar=("D", "H", "W"))
    p.add_argument("--compile", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--n-proc-da", type=int, default=4)
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--continue-training", action="store_true")
    p.add_argument("--pretrained-weights", type=Path, default=None, metavar="CHECKPOINT")
    p.add_argument("--test", action="store_true")
    p.add_argument("--test-tree", default="dub_2", metavar="TREE")
    _args_wandb(p)
    add_clusterfit_arguments(p)


def _build_predict(subs, add_clusterfit_arguments) -> None:
    p = subs.add_parser("predict", help="Run nnU-Net inference")
    _args_dataset(p)
    _args_hidden_planner(p)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--labels-dir", type=Path, default=None)
    p.add_argument("--metrics-json", type=Path, default=None)
    p.add_argument("--configuration", default="3d_fullres")
    p.add_argument("--fold", default="0")
    p.add_argument("--trainer", default=None)
    p.add_argument("--plans-identifier", default=None)
    p.add_argument(
        "--save-probabilities", action="store_true",
        help="Save softmax probabilities (.b2nd) for cascade training.",
    )
    add_clusterfit_arguments(p)


def _build_predict_tree(subs, add_clusterfit_arguments) -> None:
    p = subs.add_parser("predict-tree", help="Run whole-tree inference and export to Datumaro")
    _args_dataset(p)
    _args_hidden_planner(p)
    p.add_argument("--tree", required=True)
    p.add_argument("--ground-truth-root", type=Path, default=PROJECT_ROOT / "src/ground_truth")
    p.add_argument("--segmentation-output-root", type=Path, required=True)
    p.add_argument("--prepared-volume", type=Path, default=None)
    p.add_argument("--configuration", default="3d_fullres")
    p.add_argument("--fold", default="0")
    p.add_argument("--make-datumaro", action="store_true")
    p.add_argument("--trainer", default=None)
    p.add_argument("--npp", type=int, default=None)
    p.add_argument("--nps", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=None, metavar="N")
    p.add_argument("--plans-identifier", default=None)
    add_clusterfit_arguments(p)


def _build_cascade_prepare(subs) -> None:
    p = subs.add_parser(
        "cascade-prepare",
        help="Convert lowres predictions to .b2nd NDArray for cascade fullres training.",
    )
    _args_dataset(p)
    _args_hidden_planner(p)
    p.add_argument("--pred-dir", type=Path, required=True)
    p.add_argument("--plans-identifier", default=None)


def _build_all(subs, add_clusterfit_arguments) -> None:
    p = subs.add_parser("all", help="Prepare + plan + train")
    _args_dataset(p)
    _args_hidden_planner(p)
    p.add_argument("--source", type=Path, default=Path("src/ground_truth"))
    p.add_argument("--cvat-exports", type=Path, default=Path("src/cvat_exports"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--plans-identifier", default=None)
    p.add_argument("--configuration", default=None)
    p.add_argument("--fold", default="0")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--skip-arch-plot", action="store_true")
    p.add_argument("--initial-lr", type=float, default=1e-3)
    p.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    _args_regularize_arch(p)
    p.add_argument("--compile", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--n-proc-da", type=int, default=4)
    p.add_argument("--cpu-threads", type=int, default=1)
    p.add_argument("--continue-training", action="store_true")
    p.add_argument("--pretrained-weights", type=Path, default=None, metavar="CHECKPOINT")
    p.add_argument("--plan-configurations", nargs="+", default=["3d_fullres"])
    p.add_argument("--plan-num-processes", type=int, default=1)
    _args_wandb(p)
    add_clusterfit_arguments(p)


def _build_custom_train(subs, add_clusterfit_arguments) -> None:
    p = subs.add_parser("custom-train", help="Train the custom 3D model (SwinUNETR, MedNeXt, etc.)")
    p.add_argument("--dataset-id", type=int, default=2)
    p.add_argument("--dataset-name", default="BPWoodDefectsSplit")
    p.add_argument("--image-dir", type=Path, default=None)
    p.add_argument("--label-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("./output/custom_model"))
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--patch-size", type=int, nargs=3, default=[128, 384, 128])
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-classes", type=int, default=7)
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--sliding-window-overlap", type=float, default=0.5)
    p.add_argument("--rare-label-idx", type=int, default=6)
    p.add_argument("--rare-class-weight", type=float, default=15.0)
    p.add_argument("--oversample-factor", type=int, default=8)
    p.add_argument("--early-stopping-patience", type=int, default=50)
    p.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    p.add_argument("--early-stopping-min-epochs", type=int, default=50)
    p.add_argument("--grad-accumulation-steps", type=int, default=4)
    p.add_argument("--warmup-epochs", type=int, default=20)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--dropout-path-rate", type=float, default=0.1)
    p.add_argument("--model-name",
                   choices=["swinunetr", "swinunetr_v2", "unetr", "basicunetplusplus", "mednext", "segmamba"],
                   default="swinunetr")
    p.add_argument("--model-feature-size", type=int, default=48)
    p.add_argument("--unetr-hidden-size", type=int, default=768)
    p.add_argument("--unetr-mlp-dim", type=int, default=3072)
    p.add_argument("--unetr-num-heads", type=int, default=12)
    p.add_argument("--basicunet-features", type=int, nargs=6,
                   default=[32, 32, 64, 128, 256, 32], metavar=("F0", "F1", "F2", "F3", "F4", "F5"))
    p.add_argument("--cache-rate", type=float, default=0.0)
    p.add_argument("--normalization", choices=["range", "zscore"], default="zscore")
    p.add_argument("--norm-clip-min", type=float, default=-1000.0)
    p.add_argument("--norm-clip-max", type=float, default=500.0)
    p.add_argument("--pretrained-weights", type=Path, default=None, metavar="PATH")
    p.add_argument("--loss-type", choices=["combined", "dice_focal"], default="combined")
    p.add_argument("--debug-data", action="store_true")
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--splits-json", type=Path, default="splits_final.json")
    p.add_argument("--test", action="store_true")
    p.add_argument("--test-tree", default="dub_2", metavar="TREE")
    p.add_argument("--resume-checkpoint", type=Path, default=None, metavar="PATH")
    _args_wandb(p, default_project="bp-custom-model")
    add_clusterfit_arguments(p)


def _build_custom_predict(subs, add_clusterfit_arguments) -> None:
    p = subs.add_parser("custom-predict", help="Run inference with a trained custom model")
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    add_clusterfit_arguments(p)


def _build_custom_evaluate(subs, add_clusterfit_arguments) -> None:
    p = subs.add_parser("custom-evaluate", help="Evaluate custom model predictions")
    p.add_argument("--pred-dir", type=Path, required=True)
    p.add_argument("--gt-dir", type=Path, required=True)
    p.add_argument("--num-classes", type=int, default=7)
    p.add_argument("--output", type=Path, default=None)
    add_clusterfit_arguments(p)
