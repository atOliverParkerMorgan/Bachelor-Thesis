from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path
from typing import Dict

from .env import log, ensure_env, apply_runtime_env_overrides, import_clusterfit_helpers
from .plans import has_prepared_dataset, prepared_dataset_root
from .parser import build_parser
from .nnunet import run_prepare, run_plan, run_train, run_predict, run_predict_tree, run_cascade_prepare
from .custom import run_custom_train, run_custom_predict
from .evaluate import run_custom_evaluate

_SLURM_ENV_KEYS = (
    "PATH", "HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "VIRTUAL_ENV",
    "PYTHONPATH", "PYTHONUNBUFFERED",
    "nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results",
    "NNUNET_SAVE_EVERY", "NNUNET_INITIAL_LR", "NNUNET_OPTIMIZER", "NNUNET_WEIGHT_DECAY",
    "NNUNET_PRETRAINED_WEIGHTS", "NNUNET_SKIP_ARCH_PLOT", "nnUNet_compile", "nnUNet_n_proc_DA",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_RUN_NAME", "WANDB_API_KEY",
    "PYTORCH_CUDA_ALLOC_CONF",
)


def submit_to_clusterfit(args, env: Dict[str, str]) -> None:
    SlurmJobSubmitter, _, build_slurm_config_from_args = import_clusterfit_helpers()
    if SlurmJobSubmitter is None:
        raise RuntimeError("ClusterFIT utilities are missing. Cannot submit to Slurm.")

    slurm_env = {k: v for k in _SLURM_ENV_KEYS if (v := env.get(k))}
    slurm_config = build_slurm_config_from_args(args, args.command)

    # Use -m so the command works from a package directory, not a single file path.
    safe_args = [a for a in sys.argv[1:] if a != "--clusterfit"]
    cmd = [sys.executable, "-m", "src.implementation.nn_UNet.pipeline"] + safe_args

    script_content = SlurmJobSubmitter.build_slurm_script(
        job_command=cmd,
        slurm_config=slurm_config,
        env_vars=slurm_env,
        modules_load=["cray-ccdb", "cray-mvapich2_pmix_nogpu"] if getattr(args, "arm_hpe_cpe", False) else None,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script_content)
        script_path = Path(f.name)

    script_path.chmod(0o755)
    log(f"Partition: {slurm_config.partition}, Time: {slurm_config.time}")
    if slurm_config.gres:
        log(f"GPU: {slurm_config.gres}")

    job_id = SlurmJobSubmitter.submit_job(
        script_path,
        dry_run=args.slurm_dry_run,
        wait=args.slurm_wait,
    )
    if job_id:
        log(f"Job submitted: {job_id}")
        if args.slurm_wait:
            log("Job completed.")
    if not args.slurm_dry_run:
        log(f"Slurm script: {script_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    env = ensure_env(args.nnunet_root)
    apply_runtime_env_overrides(env, args)

    started = time.perf_counter()
    log(f"Running command: {args.command}")

    if getattr(args, "clusterfit", False):
        submit_to_clusterfit(args, env)
        log(f"Submission completed in {int(time.perf_counter() - started)}s")
        return

    if args.command in {"prepare", "all"}:
        if args.command == "all" and args.skip_prepare:
            if not has_prepared_dataset(args.nnunet_root, args.dataset_id, args.dataset_name):
                root = prepared_dataset_root(args.nnunet_root, args.dataset_id, args.dataset_name)
                raise RuntimeError(f"--skip-prepare set but dataset missing at {root}.")
            log("Skipping prepare (existing dataset).")
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
