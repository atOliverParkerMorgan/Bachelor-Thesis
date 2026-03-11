"""Compatibility shims loaded automatically by Python at startup.

This bridges nnU-Net's optional hiddenlayer plotting with newer PyTorch
versions where torch.onnx private helpers moved under torch.onnx.utils.
"""

from __future__ import annotations

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
