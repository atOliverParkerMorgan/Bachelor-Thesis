#!/usr/bin/env python3
"""nnUNetTrainerWandb — nnU-Net trainer variants with optional W&B logging."""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import torch
import torch.nn as nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss


class _RareClassFocusedDataset:
    """Proxy dataset that narrows boost patches to the rare label."""

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
    """Dice loss for a single rare class, treating it as a binary problem."""
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



class nnUNetTrainerWandb(nnUNetTrainer):
    """nnU-Net trainer that logs metrics to Weights & Biases after every epoch."""

    def initialize(self) -> None:
        env_lr = os.environ.get("NNUNET_INITIAL_LR")
        if env_lr is not None:
            self.initial_lr = float(env_lr)
        env_wd = os.environ.get("NNUNET_WEIGHT_DECAY")
        if env_wd is not None:
            self.weight_decay = float(env_wd)
        super().initialize()

    def configure_optimizers(self):
        optimizer_name = os.environ.get("NNUNET_OPTIMIZER", "sgd").lower()
        if optimizer_name in ("adam", "adamw"):
            cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
            optimizer = cls(
                self.network.parameters(),
                lr=self.initial_lr,
                weight_decay=self.weight_decay,
            )
            from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
            lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
            self.print_to_log_file(
                f"Optimizer: {cls.__name__}  lr={self.initial_lr}  wd={self.weight_decay}"
            )
            return optimizer, lr_scheduler
        return super().configure_optimizers()

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
            "optimizer": os.environ.get("NNUNET_OPTIMIZER", "sgd"),
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
    """W&B trainer that also loads partial Lung CT pretrained weights."""

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
    """Trainer variant that increases sampling and loss emphasis for rare classes."""

    RARE_LABEL_IDX:         int   = 6     # Poškození hmyzem (dataset.json index)
    CASE_OVERSAMPLE_FACTOR: int   = 8     # extra copies of rare cases per epoch
    CE_RARE_CLASS_WEIGHT:   float = 8.0   # CE loss weight for the rare class
    RARE_DICE_AUX_WEIGHT:   float = 1.0   # weight of auxiliary rare-class Dice term

    # More aggressively sample foreground-containing patches (nnUNet built-in)
    oversample_foreground_percent: float = 0.67  # default is 0.33

    def get_tr_and_val_datasets(self):
        """dataset_tr is created here (not in initialize()) — apply boost before DataLoader."""
        dataset_tr, dataset_val = super().get_tr_and_val_datasets()
        self.dataset_tr = dataset_tr
        self._duplicate_rare_class_cases()
        self._wrap_dataset_for_rare_class_focus()
        return self.dataset_tr, dataset_val

    def _wrap_dataset_for_rare_class_focus(self) -> None:
        """Must be called after _duplicate_rare_class_cases — boost keys must exist before wrapping."""
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

    def _duplicate_rare_class_cases(self) -> None:

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

    def _build_loss(self):
        """Add rare-class CE weight and auxiliary binary Dice term to the base DC+CE loss."""
        loss = super()._build_loss()

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

        aux_dice = _RareClassBinaryDice(self.RARE_LABEL_IDX)
        loss     = _CompoundLoss(loss, aux_dice, self.RARE_DICE_AUX_WEIGHT)
        self.print_to_log_file(
            f"RareClassBoost: auxiliary rare-class Dice added "
            f"(weight {self.RARE_DICE_AUX_WEIGHT})."
        )
        return loss


class nnUNetTrainerRareClassBoostLungPretrainedWandb(nnUNetTrainerRareClassBoostWandb):
    """Rare-class boost trainer with Lung CT pretraining and W&B logging."""

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
