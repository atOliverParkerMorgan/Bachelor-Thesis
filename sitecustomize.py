"""Runtime shims loaded automatically by Python at startup.

This module keeps nnU-Net plotting compatibility with newer PyTorch builds,
avoids noisy Graphviz failures when `dot` is not installed, and allows
checkpoint save cadence to be configured via environment variable.
"""

from __future__ import annotations

import os
import shutil
import inspect

try:
    import torch

    # hiddenlayer calls these legacy private symbols.
    if not hasattr(torch.onnx, "_optimize_graph") and hasattr(torch.onnx, "utils"):
        optimize_graph = getattr(torch.onnx.utils, "_optimize_graph", None)
        if optimize_graph is not None:
            torch.onnx._optimize_graph = optimize_graph  # type: ignore[attr-defined]

    if not hasattr(torch.onnx, "_optimize_trace") and hasattr(torch.onnx, "utils"):
        optimize_graph = getattr(torch.onnx.utils, "_optimize_graph", None)
        if optimize_graph is not None:
            torch.onnx._optimize_trace = optimize_graph  # type: ignore[attr-defined]
except Exception:
    # Training must continue even if plotting compatibility cannot be patched.
    pass


try:
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    _original_plot_network_architecture = nnUNetTrainer.plot_network_architecture
    _original_init = nnUNetTrainer.__init__

    def _patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        configured = os.environ.get("NNUNET_SAVE_EVERY", "").strip()
        if not configured:
            configured = ""
        if configured:
            try:
                interval = int(configured)
                if interval > 0:
                    self.save_every = interval
            except ValueError:
                # Ignore invalid values and keep nnU-Net defaults.
                pass

        initial_lr = os.environ.get("NNUNET_INITIAL_LR", "").strip()
        if initial_lr:
            try:
                parsed_lr = float(initial_lr)
                if parsed_lr > 0:
                    self.initial_lr = parsed_lr
            except ValueError:
                # Ignore invalid values and keep nnU-Net defaults.
                pass

    def _patched_plot_network_architecture(self):
        if os.environ.get("NNUNET_SKIP_ARCH_PLOT", "").strip() == "1":
            return
        if shutil.which("dot") is None:
            return
        return _original_plot_network_architecture(self)

    nnUNetTrainer.__init__ = _patched_init
    # nnU-Net inspects self.__init__ parameters; preserve original signature.
    nnUNetTrainer.__init__.__signature__ = inspect.signature(_original_init)
    nnUNetTrainer.plot_network_architecture = _patched_plot_network_architecture
except Exception:
    # Keep startup robust even if nnU-Net internals change.
    pass
