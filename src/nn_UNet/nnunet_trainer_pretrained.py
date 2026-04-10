#!/usr/bin/env python3
"""
nnUNetTrainerLungPretrained — partial-weight transfer from a pretrained Lung CT checkpoint.

pipeline.py copies this file into the nnunetv2 trainer variants directory automatically
when --pretrained-weights is passed, making the class discoverable by nnUNetv2_train.

Checkpoint path is read from the environment variable NNUNET_PRETRAINED_WEIGHTS.
Uses strict=False so the final segmentation head (different class count) is always
reinitialised from scratch.  Supports both:
  - nnUNetv2 .pth checkpoints  (key: 'network_weights')
  - nnUNetv1 .model checkpoints (key: 'state_dict')
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerLungPretrained(nnUNetTrainer):
    """Transfer-learning trainer: initialises encoder from a pretrained Lung CT model.

    The final segmentation head is always trained from scratch because the number of
    output classes differs between Lung (2) and the wood-defect dataset.
    """

    def initialize(self) -> None:
        super().initialize()
        self._load_pretrained_weights_partial()

    def _load_pretrained_weights_partial(self) -> None:
        pretrained_path_str = os.environ.get("NNUNET_PRETRAINED_WEIGHTS")
        if not pretrained_path_str:
            self.print_to_log_file(
                "WARNING: NNUNET_PRETRAINED_WEIGHTS env var not set — training from scratch."
            )
            return

        pretrained_path = Path(pretrained_path_str)
        if not pretrained_path.exists():
            raise FileNotFoundError(
                f"Pretrained checkpoint not found: {pretrained_path}\n"
                "Install the Lung model first (run on cluster):\n"
                "  nnUNetv2_install_pretrained_model_from_zip Task006_Lung.zip\n"
                "Then pass the installed .pth path to --pretrained-weights."
            )

        self.print_to_log_file(f"Loading pretrained weights from: {pretrained_path}")

        try:
            checkpoint = torch.load(str(pretrained_path), map_location="cpu", weights_only=False)
        except TypeError:
            # PyTorch < 2.0 does not have weights_only parameter
            checkpoint = torch.load(str(pretrained_path), map_location="cpu")

        if "network_weights" in checkpoint:
            state_dict = checkpoint["network_weights"]
            fmt = "nnUNetv2"
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            fmt = "nnUNetv1"
        else:
            raise KeyError(
                f"Unrecognised checkpoint format at {pretrained_path}. "
                f"Keys found: {list(checkpoint.keys())}. "
                "Expected 'network_weights' (v2) or 'state_dict' (v1)."
            )

        incompatible = self.network.load_state_dict(state_dict, strict=False)
        n_missing = len(incompatible.missing_keys)
        n_unexpected = len(incompatible.unexpected_keys)
        n_transferred = len(state_dict) - n_unexpected

        self.print_to_log_file(
            f"Pretrained weights loaded (format: {fmt}).\n"
            f"  Transferred layers : {n_transferred}\n"
            f"  Init from scratch  : {n_missing} keys (expected — output head differs)\n"
            f"  Ignored/unexpected : {n_unexpected} keys"
        )
        if incompatible.missing_keys:
            shown = incompatible.missing_keys[:5]
            more = f" … (+{n_missing - 5} more)" if n_missing > 5 else ""
            self.print_to_log_file(f"  First missing keys : {shown}{more}")
