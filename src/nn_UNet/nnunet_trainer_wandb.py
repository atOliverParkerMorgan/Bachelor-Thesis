#!/usr/bin/env python3
"""
nnUNetTrainerWandb — nnU-Net v2 trainer with Weights & Biases logging.

pipeline.py copies this file into the nnunetv2 trainer variants directory
automatically when --wandb is passed, making the classes discoverable by
nnUNetv2_train.

Configuration via environment variables (all optional):
  WANDB_PROJECT       W&B project name           (default: nnunet-training)
  WANDB_ENTITY        W&B entity / team name      (default: None)
  WANDB_RUN_NAME      Display name for this run   (default: auto-generated)
  WANDB_API_KEY       API key if not already logged in

Also includes nnUNetTrainerLungPretrainedWandb which combines transfer-learning
from a pretrained Lung CT checkpoint with W&B logging.  Checkpoint path is read
from NNUNET_PRETRAINED_WEIGHTS.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss


# ── helpers used by nnUNetTrainerRareClassBoostWandb ────────────────────────


class _RareClassFocusedDataset:
    """
    Thin proxy around any nnUNetBaseDataset subclass
    (nnUNetDatasetBlosc2, nnUNetDatasetNumpy, …).

    nnUNetDataLoader reads case identifiers from data.identifiers (a plain list,
    set at DataLoader.__init__ line 46: self.indices = data.identifiers) and
    loads cases via data.load_case(identifier).  This wrapper:

      - Exposes an identifiers list that includes '__rare_boost_N' suffixed
        duplicates of rare-class cases (already appended to the underlying
        dataset's identifiers by _duplicate_rare_class_cases before wrapping).
      - Overrides load_case so that for any boost key the suffix is stripped
        to load the real data, and class_locations in the returned properties
        is narrowed to only the rare label — forcing the DataLoader's foreground
        oversampling (get_bbox) to centre every boost-case patch on a rare-class
        voxel instead of a random foreground class.
    """

    def __init__(self, dataset, rare_label_idx: int) -> None:
        self._dataset = dataset
        self.rare_label_idx = rare_label_idx
        # Copy identifiers so we own the list; boost keys were already appended
        # to dataset.identifiers before this wrapper was created.
        self.identifiers = list(dataset.identifiers)

    def load_case(self, identifier: str):
        is_boost = "__rare_boost_" in str(identifier)
        base_id  = identifier.split("__rare_boost_")[0] if is_boost else identifier

        data, seg, seg_prev, props = self._dataset.load_case(base_id)

        if is_boost:
            locs = props.get("class_locations", {})
            if self.rare_label_idx in locs and len(locs[self.rare_label_idx]) > 0:
                props = dict(props)
                props["class_locations"] = {
                    self.rare_label_idx: locs[self.rare_label_idx]
                }

        return data, seg, seg_prev, props

    def __getitem__(self, identifier):
        return self.load_case(identifier)

    def __getattr__(self, name):
        # Delegate source_folder, folder_with_segs_from_previous_stage, etc.
        return getattr(self._dataset, name)


class _RareClassBinaryDice(nn.Module):
    """
    Auxiliary binary Dice loss focused on a single rare class.

    Computes Dice only between the rare-class softmax probability channel and
    the binary rare-class mask.  Added on top of the standard DC+CE loss to
    give an extra gradient signal specifically for the rare class.

    Handles both plain tensors (validation / no deep supervision) and
    deep-supervision lists (training): in the list case only the
    full-resolution head (index 0) is used.
    """

    def __init__(self, rare_label_idx: int, smooth: float = 1.0) -> None:
        super().__init__()
        self.rare_label_idx = rare_label_idx
        self.smooth = smooth

    def _binary_dice(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        logits : [B, C, ...]  — raw network logits
        target : [B, 1, ...]  — integer segmentation map
        """
        probs = torch.softmax(logits.float(), dim=1)[:, self.rare_label_idx]
        mask  = (target.squeeze(1) == self.rare_label_idx).float()
        tp = (probs * mask).sum()
        fp = (probs * (1.0 - mask)).sum()
        fn = ((1.0 - probs) * mask).sum()
        return 1.0 - (2.0 * tp + self.smooth) / (
            2.0 * tp + fp + fn + self.smooth
        )

    def forward(self, net_output, target):
        # Deep supervision: both inputs are lists — use full-resolution head
        if isinstance(net_output, (list, tuple)):
            lo = net_output[0]
            tg = target[0] if isinstance(target, (list, tuple)) else target
        else:
            lo, tg = net_output, target
        return self._binary_dice(lo, tg)


class _CompoundLoss(nn.Module):
    """Adds a weighted auxiliary loss on top of a main loss."""

    def __init__(
        self, main: nn.Module, aux: nn.Module, aux_weight: float
    ) -> None:
        super().__init__()
        self.main = main
        self.aux  = aux
        self.aux_weight = aux_weight

    def forward(self, net_output, target):
        return self.main(net_output, target) + self.aux_weight * self.aux(
            net_output, target
        )


# ── trainers ────────────────────────────────────────────────────────────────


class nnUNetTrainerWandb(nnUNetTrainer):
    """nnU-Net trainer that logs metrics to Weights & Biases after every epoch."""

    def initialize(self) -> None:
        env_lr = os.environ.get("NNUNET_INITIAL_LR")
        if env_lr is not None:
            self.initial_lr = float(env_lr)
        super().initialize()

    def on_train_start(self) -> None:
        super().on_train_start()
        try:
            import wandb
        except ImportError:
            self.print_to_log_file(
                "WARNING: wandb not installed — W&B logging disabled. "
                "Install with: pip install wandb"
            )
            self._wandb_enabled = False
            return

        project = os.environ.get("WANDB_PROJECT", "nnunet-training")
        entity = os.environ.get("WANDB_ENTITY") or None
        run_name = os.environ.get("WANDB_RUN_NAME") or None

        config = {
            "configuration": self.configuration_name,
            "fold": self.fold,
            "initial_lr": self.initial_lr,
            "weight_decay": self.weight_decay,
            "num_epochs": self.num_epochs,
        }
        try:
            config["plans_identifier"] = self.plans_manager.plans.get("experiment_planner_used", "unknown")
            config["dataset_name"] = self.plans_manager.dataset_name
        except Exception:
            pass
        pretrained = os.environ.get("NNUNET_PRETRAINED_WEIGHTS")
        if pretrained:
            config["pretrained_weights"] = pretrained

        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=config,
            resume="allow",
        )
        self._wandb_enabled = True
        self.print_to_log_file(
            f"W&B run initialised — project: {project}, entity: {entity}, name: {run_name}"
        )

    def on_epoch_end(self) -> None:
        super().on_epoch_end()

        if not getattr(self, "_wandb_enabled", False):
            return

        try:
            import wandb
            if wandb.run is None:
                return

            log = self.logger.my_fantastic_logging
            metrics: dict = {"epoch": self.current_epoch}

            def _last(key: str):
                vals = log.get(key)
                return vals[-1] if vals else None

            train_loss = _last("train_losses")
            val_loss   = _last("val_losses")
            mean_dice  = _last("mean_fg_dice")
            ema_dice   = _last("ema_fg_dice")
            lr         = _last("lrs")

            if train_loss is not None:
                metrics["train/loss"] = float(train_loss)
            if val_loss is not None:
                metrics["val/loss"] = float(val_loss)
            if mean_dice is not None:
                metrics["val/mean_fg_dice"] = float(mean_dice)
            if ema_dice is not None:
                metrics["val/ema_fg_dice"] = float(ema_dice)
            if lr is not None:
                metrics["train/lr"] = float(lr)

            # Per-class dice if available
            dice_per_class = _last("dice_per_class_or_region")
            if dice_per_class is not None:
                try:
                    for i, d in enumerate(dice_per_class):
                        metrics[f"val/dice_class_{i}"] = float(d)
                except (TypeError, ValueError):
                    pass

            wandb.log(metrics, step=self.current_epoch)
        except Exception as e:
            self.print_to_log_file(f"WARNING: W&B logging failed this epoch: {e}")

    def on_train_end(self) -> None:
        super().on_train_end()
        if not getattr(self, "_wandb_enabled", False):
            return
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass


class nnUNetTrainerLungPretrainedWandb(nnUNetTrainerWandb):
    """Transfer-learning trainer (Lung CT pretrained weights) with W&B logging.

    Combines nnUNetTrainerLungPretrained and nnUNetTrainerWandb in one class.
    Pretrained checkpoint path is read from NNUNET_PRETRAINED_WEIGHTS.
    Uses strict=False so the final segmentation head is always reinitialised.
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
        n_missing    = len(incompatible.missing_keys)
        n_unexpected = len(incompatible.unexpected_keys)
        n_transferred = len(state_dict) - n_unexpected

        self.print_to_log_file(
            f"Pretrained weights loaded (format: {fmt}).\n"
            f"  Transferred layers : {n_transferred}\n"
            f"  Init from scratch  : {n_missing} keys (expected — output head differs)\n"
            f"  Ignored/unexpected : {n_unexpected} keys"
        )


class nnUNetTrainerRareClassBoostWandb(nnUNetTrainerWandb):
    """
    Trainer that maximises learning for rare / under-represented classes.

    Combines four complementary techniques:

      1. Higher foreground-oversample rate (0.67 vs default 0.33) so the
         DataLoader more aggressively picks foreground-containing patches.

      2. Case-level oversampling: training cases that contain the rare class
         are duplicated so they appear proportionally more often per epoch.

      3. Patch-level rare-class focus: the duplicated cases are served through
         a _RareClassFocusedDataset proxy that narrows class_locations to only
         the rare label, forcing every foreground-oversampled patch from a
         boost case to be centred on a rare-class voxel.

      4. Loss amplification for the rare class:
           a. Cross-entropy class weight (30×) — large gradient signal on
              each mis-classified rare-class voxel.
           b. Auxiliary binary Dice loss (weight 5×) — directly optimises
              the Dice score for the rare class on top of the standard
              DC+CE loss.

    Configuration (override as class attributes in a subclass if needed)
    -----------------------------------------------------------------------
    RARE_LABEL_IDX         int   label index to boost (6 = Poškození hmyzem)
      CASE_OVERSAMPLE_FACTOR int   extra copies of rare cases per epoch (8)
      CE_RARE_CLASS_WEIGHT   float CE loss weight multiplier for rare class (30.0)
      RARE_DICE_AUX_WEIGHT   float weight of auxiliary binary Dice term (5.0)

    Usage
    -----
      nnUNetv2_train 1 3d_fullres 0 -p nnUNetResEncUNetLPlans \\
          --trainer nnUNetTrainerRareClassBoostWandb

    (pipeline.py will copy this file to the nnunetv2 trainers directory
     automatically when --wandb is passed, or you can copy it manually.)
    """

    # ── tuneable knobs ─────────────────────────────────────────────────────────
    RARE_LABEL_IDX:         int   = 6     # Poškození hmyzem (dataset.json index)
    CASE_OVERSAMPLE_FACTOR: int   = 8     # extra copies of rare cases per epoch
    CE_RARE_CLASS_WEIGHT:   float = 8.0   # CE loss weight for the rare class
    RARE_DICE_AUX_WEIGHT:   float = 1.0   # weight of auxiliary rare-class Dice term

    # More aggressively sample foreground-containing patches (nnUNet built-in)
    oversample_foreground_percent: float = 0.67  # default is 0.33

    # ── dataset setup (overrides nnUNetTrainer.get_tr_and_val_datasets) ────────

    def get_tr_and_val_datasets(self):
        """
        Build the training/validation datasets via super(), then apply
        case-level oversampling and the rare-class patch proxy on dataset_tr
        before the DataLoader is constructed.

        dataset_tr does not exist during initialize() — it is created here
        inside get_dataloaders() → get_tr_and_val_datasets().
        """
        dataset_tr, dataset_val = super().get_tr_and_val_datasets()
        self.dataset_tr = dataset_tr
        self._duplicate_rare_class_cases()
        self._wrap_dataset_for_rare_class_focus()
        return self.dataset_tr, dataset_val

    def _wrap_dataset_for_rare_class_focus(self) -> None:
        """
        Replace self.dataset_tr with a _RareClassFocusedDataset proxy.

        Must be called AFTER _duplicate_rare_class_cases so the boost keys
        already exist inside the underlying dataset before wrapping.
        """
        if not hasattr(self, "dataset_tr") or self.dataset_tr is None:
            self.print_to_log_file(
                "RareClassBoost: dataset_tr not available — "
                "skipping rare-class patch focus."
            )
            return
        self.dataset_tr = _RareClassFocusedDataset(
            self.dataset_tr, self.RARE_LABEL_IDX
        )
        self.print_to_log_file(
            f"RareClassBoost: dataset_tr wrapped with _RareClassFocusedDataset "
            f"— boost-case patches will be centred on label {self.RARE_LABEL_IDX}."
        )

    # ── case oversampling ──────────────────────────────────────────────────────

    def _duplicate_rare_class_cases(self) -> None:
        """
        Scan the preprocessed property files to find training cases that
        contain RARE_LABEL_IDX, then add CASE_OVERSAMPLE_FACTOR duplicate
        entries for each such case in self.dataset_tr.

        nnUNet's DataLoader samples uniformly from dataset_tr.identifiers, so
        adding duplicates increases the chance those cases appear in a batch.
        Each duplicate value points to the SAME preprocessed files, so no
        disk space is wasted.
        """
        if not hasattr(self, "dataset_tr") or self.dataset_tr is None:
            self.print_to_log_file(
                "RareClassBoost: dataset_tr not initialised yet — skipping case oversample."
            )
            return

        rare_ids = []
        for identifier in list(self.dataset_tr.identifiers):
            try:
                # Each case has a .pkl with 'class_locations': {label_idx: coords, ...}
                # nnUNet builds this during fingerprinting / preprocessing.
                props = self._load_case_properties(identifier)
                if props is None:
                    continue
                locs = props.get("class_locations", {})
                if self.RARE_LABEL_IDX in locs and len(locs[self.RARE_LABEL_IDX]) > 0:
                    rare_ids.append(identifier)
            except Exception as e:
                self.print_to_log_file(
                    f"RareClassBoost: could not read properties for {identifier}: {e}"
                )

        if not rare_ids:
            self.print_to_log_file(
                f"RareClassBoost: no training cases found with label {self.RARE_LABEL_IDX}. "
                "Falling back to loss-weight and oversample-percent only."
            )
            return

        n_added = 0
        for identifier in rare_ids:
            for i in range(self.CASE_OVERSAMPLE_FACTOR):
                self.dataset_tr.identifiers.append(f"{identifier}__rare_boost_{i}")
                n_added += 1

        self.print_to_log_file(
            f"RareClassBoost: found {len(rare_ids)} case(s) with "
            f"label {self.RARE_LABEL_IDX} (Poškození hmyzem). "
            f"Added {n_added} duplicate entries → "
            f"total training cases: {len(self.dataset_tr.identifiers)}"
        )

    def _load_case_properties(self, identifier: str) -> dict | None:
        """Load the .pkl properties file for a training case."""
        if not hasattr(self.dataset_tr, 'source_folder'):
            return None
        pkl_path = os.path.join(self.dataset_tr.source_folder, identifier + '.pkl')
        if not os.path.exists(pkl_path):
            return None
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    # ── loss with rare-class amplification ────────────────────────────────────

    def _build_loss(self):
        """
        Build standard DC+CE loss, then apply two rare-class amplifications:

          1. Replace the CE module with a class-weighted version (30×) so
             mis-classifying a rare-class voxel costs proportionally more.

          2. Wrap in _CompoundLoss, adding an auxiliary _RareClassBinaryDice
             term (weight 5×) that directly optimises the per-class Dice score
               for Poškození hmyzem on top of the standard DC+CE signal.
        """
        loss = super()._build_loss()

        # ── inject CE class weight ─────────────────────────────────────────────
        # super() may return a DeepSupervisionWrapper; the inner DC_and_CE_loss
        # is stored at loss.loss by DeepSupervisionWrapper.
        dc_ce = getattr(loss, "loss", loss)

        if not hasattr(dc_ce, "ce"):
            self.print_to_log_file(
                "RareClassBoost: unexpected loss structure — "
                "could not inject CE class weight."
            )
        else:
            n_classes = self.label_manager.num_segmentation_heads
            weights   = torch.ones(n_classes, dtype=torch.float32)
            if self.RARE_LABEL_IDX < n_classes:
                weights[self.RARE_LABEL_IDX] = self.CE_RARE_CLASS_WEIGHT
                # Move to the trainer's device (GPU) so CrossEntropyLoss.weight
                # is on the same device as the network output tensors.
                weights    = weights.to(self.device)
                ignore_idx = getattr(dc_ce.ce, "ignore_index", -100)
                dc_ce.ce   = RobustCrossEntropyLoss(
                    weight=weights, ignore_index=ignore_idx
                )
                self.print_to_log_file(
                    f"RareClassBoost: CE weight vector = {weights.tolist()}  "
                    f"(class {self.RARE_LABEL_IDX} × {self.CE_RARE_CLASS_WEIGHT})"
                )
            else:
                self.print_to_log_file(
                    f"RareClassBoost: RARE_LABEL_IDX={self.RARE_LABEL_IDX} is out "
                    f"of range (n_classes={n_classes}) — CE weight not applied."
                )

        # ── auxiliary binary Dice for the rare class ───────────────────────────
        aux_dice = _RareClassBinaryDice(self.RARE_LABEL_IDX)
        loss     = _CompoundLoss(loss, aux_dice, self.RARE_DICE_AUX_WEIGHT)
        self.print_to_log_file(
            f"RareClassBoost: auxiliary rare-class Dice added "
            f"(weight {self.RARE_DICE_AUX_WEIGHT})."
        )
        return loss


class nnUNetTrainerRareClassBoostLungPretrainedWandb(nnUNetTrainerRareClassBoostWandb):
    """
    Combines all three capabilities in one trainer:
      - Rare-class boost (case oversampling, patch forcing, weighted loss)
      - Transfer learning from a pretrained Lung CT checkpoint
      - Weights & Biases logging

    Pretrained checkpoint path is read from NNUNET_PRETRAINED_WEIGHTS.
    Selected automatically by pipeline.py when both --pretrained-weights
    and --wandb are passed together.
    """

    def initialize(self) -> None:
        super().initialize()
        self._load_pretrained_weights_partial()

    def _load_pretrained_weights_partial(self) -> None:
        pretrained_path_str = os.environ.get("NNUNET_PRETRAINED_WEIGHTS")
        if not pretrained_path_str:
            self.print_to_log_file(
                "WARNING: NNUNET_PRETRAINED_WEIGHTS env var not set — "
                "training from scratch."
            )
            return

        pretrained_path = Path(pretrained_path_str)
        if not pretrained_path.exists():
            raise FileNotFoundError(
                f"Pretrained checkpoint not found: {pretrained_path}\n"
                "Install the Lung model first:\n"
                "  nnUNetv2_install_pretrained_model_from_zip Task006_Lung.zip\n"
                "Then pass the installed .pth path to --pretrained-weights."
            )

        self.print_to_log_file(
            f"Loading pretrained weights from: {pretrained_path}"
        )

        try:
            checkpoint = torch.load(
                str(pretrained_path), map_location="cpu", weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(str(pretrained_path), map_location="cpu")

        if "network_weights" in checkpoint:
            state_dict, fmt = checkpoint["network_weights"], "nnUNetv2"
        elif "state_dict" in checkpoint:
            state_dict, fmt = checkpoint["state_dict"], "nnUNetv1"
        else:
            raise KeyError(
                f"Unrecognised checkpoint format at {pretrained_path}. "
                f"Keys: {list(checkpoint.keys())}. "
                "Expected 'network_weights' (v2) or 'state_dict' (v1)."
            )

        incompatible  = self.network.load_state_dict(state_dict, strict=False)
        n_transferred = len(state_dict) - len(incompatible.unexpected_keys)
        self.print_to_log_file(
            f"Pretrained weights loaded (format: {fmt}).\n"
            f"  Transferred layers : {n_transferred}\n"
            f"  Init from scratch  : {len(incompatible.missing_keys)} keys\n"
            f"  Ignored/unexpected : {len(incompatible.unexpected_keys)} keys"
        )
