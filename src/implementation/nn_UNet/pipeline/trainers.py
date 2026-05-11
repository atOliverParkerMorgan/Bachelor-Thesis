from __future__ import annotations

import shutil
from pathlib import Path

from .env import log


def _install_trainer_file(filename: str) -> bool:
    try:
        import nnunetv2
    except ImportError:
        log("Warning: nnunetv2 not importable — cannot install custom trainer.")
        return False

    src = Path(__file__).parent.parent / filename
    if not src.exists():
        raise FileNotFoundError(f"Trainer source not found: {src}")

    dest_dir = Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer" / "variants"
    dest_dir.mkdir(exist_ok=True)
    dest = dest_dir / src.name

    if not dest.exists() or dest.stat().st_mtime < src.stat().st_mtime:
        shutil.copy2(src, dest)
        log(f"Installed custom trainer: {dest}")
    return True


def install_pretrained_trainer() -> bool:
    return _install_trainer_file("nnunet_trainer_pretrained.py")


def install_wandb_trainer() -> bool:
    return _install_trainer_file("nnunet_trainer_wandb.py")


def install_cediceskel_trainer() -> bool:
    return _install_trainer_file("nnunet_trainer_cediceskel.py")
