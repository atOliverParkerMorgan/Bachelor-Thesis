#!/usr/bin/env python3
"""nnUNetTrainerCeDiceSkel — CE + Dice + skeleton-recall loss for nnU-Net v2."""
from __future__ import annotations

import concurrent.futures
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import zoom as nd_zoom

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class _SkeletonRecallLoss(nn.Module):
    """1 - mean skeleton recall over foreground classes."""

    def __init__(
        self,
        include_background: bool = False,
        epsilon: float = 1e-5,
        downsample_factor: int = 2,
        max_workers: int = 4,
    ):
        super().__init__()
        self.include_background = include_background
        self.epsilon = epsilon
        self.downsample_factor = downsample_factor
        self.max_workers = max_workers

    @staticmethod
    def _skeletonize_one(mask: np.ndarray, factor: int) -> np.ndarray:
        """Compute skeleton of a single binary 3D mask, with optional downsampling."""
        from skimage.morphology import skeletonize

        if not mask.any():
            return np.zeros_like(mask, dtype=np.float32)

        if factor > 1:
            small = nd_zoom(mask.astype(np.float32), 1.0 / factor, order=0).astype(bool)
            skel_small = skeletonize(small).astype(np.float32)
            out = nd_zoom(skel_small, factor, order=0)
            # Trim/pad to original shape (zoom rounding can differ by 1 voxel)
            slices = tuple(slice(0, s) for s in mask.shape)
            out = out[slices]
            if out.shape != mask.shape:
                padded = np.zeros(mask.shape, dtype=np.float32)
                insert = tuple(slice(0, min(o, m)) for o, m in zip(out.shape, mask.shape))
                padded[insert] = out[insert]
                out = padded
        else:
            out = skeletonize(mask).astype(np.float32)

        return out

    @torch.no_grad()
    def _compute_skeleton(self, onehot: torch.Tensor) -> torch.Tensor:
        B, C = onehot.shape[:2]
        arr = onehot.detach().cpu().numpy().astype(bool)
        out = np.zeros_like(arr, dtype=np.float32)
        factor = self.downsample_factor

        jobs: list[tuple[int, int]] = [
            (b, c) for b in range(B) for c in range(C)
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {
                ex.submit(self._skeletonize_one, arr[b, c], factor): (b, c)
                for b, c in jobs
            }
            for fut in concurrent.futures.as_completed(futures):
                b, c = futures[fut]
                out[b, c] = fut.result()

        return torch.from_numpy(out).to(onehot.device)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)

        tgt = targets.long()
        if tgt.shape[1] == 1:
            tgt = tgt.squeeze(1)

        onehot = torch.zeros_like(probs)
        onehot.scatter_(1, tgt.unsqueeze(1), 1.0)

        start = 0 if self.include_background else 1
        probs_fg = probs[:, start:]
        onehot_fg = onehot[:, start:]

        skel = self._compute_skeleton(onehot_fg)

        spatial = list(range(2, skel.ndim))
        numer = (probs_fg * skel).sum(dim=spatial)
        denom = skel.sum(dim=spatial) + self.epsilon

        recall = numer / denom
        return 1.0 - recall.mean()


class _SkelAugmentedLoss(nn.Module):
    """Wrapper that adds skeleton recall loss to any base loss."""

    def __init__(
        self,
        base_loss: nn.Module,
        lambda_skel: float = 0.5,
        skel_every_n: int = 4,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.skel = _SkeletonRecallLoss(
            include_background=False,
            downsample_factor=2,
            max_workers=4,
        )
        self.lambda_skel = lambda_skel
        self.skel_every_n = skel_every_n
        self._step = 0

    def forward(self, net_output, target):
        base = self.base_loss(net_output, target)

        self._step += 1
        if self._step % self.skel_every_n != 0:
            return base

        # Deep supervision: net_output and target are lists ordered coarse→fine.
        # Index 0 is always the full-resolution output.
        primary_out = net_output[0] if isinstance(net_output, (list, tuple)) else net_output
        primary_tgt = target[0] if isinstance(target, (list, tuple)) else target

        skel = self.skel(primary_out, primary_tgt)
        return base + self.lambda_skel * skel


class nnUNetTrainerCeDiceSkel(nnUNetTrainer):
    """nnU-Net trainer with CE, Dice, and skeleton recall loss."""

    def _build_loss(self) -> nn.Module:
        base = super()._build_loss()
        return _SkelAugmentedLoss(base, lambda_skel=0.5, skel_every_n=4)
