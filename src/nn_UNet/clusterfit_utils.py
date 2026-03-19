"""ClusterFIT (Slurm) submission utilities for nn_UNet pipeline."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict


@dataclass
class SlurmConfig:
    """Configuration for Slurm job submission."""
    partition: str = "fast"  # fast, gpu, arm_fast
    nodes: int = 1
    time: str = "00:30:00"  # HH:MM:SS
    job_name: str = "nnunet"
    gres: Optional[str] = None  # "gpu:1", "gpu:p100_16:1", "gpu:a100_40:1"
    mem: Optional[str] = None  # e.g., "32G"
    ntasks: int = 1
    cpus_per_task: int = 1
    output: Optional[Path] = None
    error: Optional[Path] = None
    email: Optional[str] = None
    email_type: str = "END,FAIL"

    def to_sbatch_args(self) -> list[str]:
        """Convert config to sbatch command-line arguments."""
        args = [
            f"--partition={self.partition}",
            f"--nodes={self.nodes}",
            f"--time={self.time}",
            f"--job-name={self.job_name}",
            f"--ntasks={self.ntasks}",
            f"--cpus-per-task={self.cpus_per_task}",
        ]
        if self.gres:
            args.append(f"--gres={self.gres}")
        if self.mem:
            args.append(f"--mem={self.mem}")
        if self.output:
            args.append(f"--output={self.output}")
        if self.error:
            args.append(f"--error={self.error}")
        if self.email:
            args.append(f"--mail-user={self.email}")
            args.append(f"--mail-type={self.email_type}")
        return args


class SlurmJobSubmitter:
    """Helper for submitting jobs to ClusterFIT via Slurm."""

    GPU_MODELS = {
        "p100": "p100_16",
        "v100": "v100_32",
        "a100_40": "a100_40",
        "a100_80": "a100_80",
        "mi210": "mi210",
    }

    CPU_PARTITIONS = {
        "fast": "fast CPU partition (amd64/x86_64, 64GB RAM)",
        "gpu": "GPU partition (amd64/x86_64)",
        "arm_fast": "ARM partition (aarch64, 32GB RAM)",
        "amd": "AMD GPU partition (amd64/x86_64, 512GB RAM)",
    }

    @staticmethod
    def validate_gpu_model(model: str) -> str:
        """Validate and normalize GPU model name."""
        if model.lower() in SlurmJobSubmitter.GPU_MODELS:
            return SlurmJobSubmitter.GPU_MODELS[model.lower()]
        # Accept normalized names directly
        if model in SlurmJobSubmitter.GPU_MODELS.values():
            return model
        raise ValueError(
            f"Unknown GPU model: {model}. Available: "
            f"{', '.join(SlurmJobSubmitter.GPU_MODELS.keys())}"
        )

    @staticmethod
    def build_slurm_script(
        job_command: list[str],
        slurm_config: SlurmConfig,
        env_vars: Dict[str, str] | None = None,
        modules_load: list[str] | None = None,
    ) -> str:
        """Build a Slurm batch script."""
        script_lines = ["#!/bin/bash"]
        
        # Add SBATCH directives
        for arg in slurm_config.to_sbatch_args():
            script_lines.append(f"#SBATCH {arg}")
        
        script_lines.append("")
        script_lines.append("# Job information")
        script_lines.append("echo 'Job started at ' $(date)")
        script_lines.append("echo 'Hostname: ' $(hostname)")
        script_lines.append("")
        
        # Load modules if specified (for ARM/HPE CPE environment)
        if modules_load:
            script_lines.append("# Load environment modules")
            for module in modules_load:
                script_lines.append(f"module load {module}")
            script_lines.append("")
        
        # Set environment variables
        if env_vars:
            script_lines.append("# Set environment variables")
            for key, value in env_vars.items():
                script_lines.append(f"export {key}=\"{value}\"")
            script_lines.append("")
        
        # Run the job
        script_lines.append("# Run job")
        script_lines.append(" ".join(job_command))
        
        script_lines.append("")
        script_lines.append("echo 'Job finished at ' $(date)")
        
        return "\n".join(script_lines) + "\n"

    @staticmethod
    def submit_job(
        script_path: Path,
        dry_run: bool = False,
        wait: bool = False,
    ) -> str | None:
        """Submit job to Slurm and return job ID."""
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        
        cmd = ["sbatch"]
        if dry_run:
            cmd.append("--test-only")
        if wait:
            cmd.append("--wait")
        
        cmd.append(str(script_path))
        
        print(f"$ {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse job ID from output
        output = result.stdout.strip()
        if "Submitted batch job" in output:
            job_id = output.split()[-1]
            print(f"Job submitted with ID: {job_id}")
            return job_id
        
        print(output)
        return None

    @staticmethod
    def cancel_job(job_id: str) -> None:
        """Cancel a submitted job."""
        cmd = ["scancel", job_id]
        print(f"$ {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"Job {job_id} cancelled")

    @staticmethod
    def get_job_status(job_id: str) -> None:
        """Check job status."""
        cmd = ["squeue", "-j", job_id]
        print(f"$ {' '.join(cmd)}")
        subprocess.run(cmd, check=False)


def add_clusterfit_arguments(subparser: argparse.ArgumentParser) -> None:
    """Add ClusterFIT/Slurm arguments to a subparser."""
    group = subparser.add_argument_group("ClusterFIT (Slurm)")
    group.add_argument(
        "--clusterfit",
        action="store_true",
        help="Submit job to ClusterFIT via Slurm instead of running locally",
    )
    group.add_argument(
        "--slurm-partition",
        choices=["fast", "gpu", "arm_fast", "amd"],
        default="fast",
        help="ClusterFIT partition to use (default: fast)",
    )
    group.add_argument(
        "--slurm-time",
        default=None,
        help="Time limit in HH:MM:SS format (e.g., 02:00:00)",
    )
    group.add_argument(
        "--slurm-gpu",
        default=None,
        help="GPU model for GPU partition (p100, v100, a100_40, a100_80, mi210)",
    )
    group.add_argument(
        "--slurm-mem",
        default=None,
        help="Memory per node (e.g., 32G, 256G)",
    )
    group.add_argument(
        "--slurm-nodes",
        type=int,
        default=1,
        help="Number of nodes for multi-node jobs (default: 1)",
    )
    group.add_argument(
        "--slurm-cpus-per-task",
        type=int,
        default=1,
        help="CPUs per task (default: 1)",
    )
    group.add_argument(
        "--slurm-job-name",
        default=None,
        help="Slurm job name (default: auto-generated from command)",
    )
    group.add_argument(
        "--slurm-output",
        type=Path,
        default=None,
        help="Output log file path for Slurm",
    )
    group.add_argument(
        "--slurm-email",
        default=None,
        help="Email address for job notifications",
    )
    group.add_argument(
        "--slurm-dry-run",
        action="store_true",
        help="Test Slurm submission without actually submitting",
    )
    group.add_argument(
        "--slurm-wait",
        action="store_true",
        help="Wait for job to complete before returning",
    )
    group.add_argument(
        "--arm-hpe-cpe",
        action="store_true",
        help="Use HPE Cray Programming Environment for ARM jobs (automatic if partition is arm_fast)",
    )


def build_slurm_config_from_args(
    args: argparse.Namespace,
    command_name: str,
) -> SlurmConfig:
    """Build SlurmConfig from parsed arguments."""
    # Determine time limits based on command
    time_limits = {
        "prepare": "01:00:00",
        "plan": "04:00:00",
        "train": "12:00:00",
        "predict": "02:00:00",
        "predict-tree": "02:00:00",
    }
    default_time = time_limits.get(command_name, "02:00:00")
    time_limit = args.slurm_time or default_time

    # Determine job name
    job_name = args.slurm_job_name or f"nnunet-{command_name}"

    # Determine partition and GPU selection
    partition = args.slurm_partition
    gres = None
    if partition == "gpu" and args.slurm_gpu:
        gpu_model = SlurmJobSubmitter.validate_gpu_model(args.slurm_gpu)
        gres = f"gpu:{gpu_model}:1"
    elif partition == "gpu":
        gres = "gpu:1"

    # Set output/error log files if not specified
    output = args.slurm_output
    if not output:
        output = Path(f"slurm_logs/{job_name}_%j.log")
        output.parent.mkdir(parents=True, exist_ok=True)

    return SlurmConfig(
        partition=partition,
        nodes=args.slurm_nodes,
        time=time_limit,
        job_name=job_name,
        gres=gres,
        mem=args.slurm_mem,
        ntasks=1,
        cpus_per_task=args.slurm_cpus_per_task,
        output=output,
        email=args.slurm_email,
    )
