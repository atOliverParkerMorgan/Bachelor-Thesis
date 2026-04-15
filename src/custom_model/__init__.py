from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from src.custom_model.train import TrainConfig

__all__ = ["TrainConfig", "train"]


def __getattr__(name: str):
	if name in {"TrainConfig", "train"}:
		from src.custom_model.train import TrainConfig, train

		return {"TrainConfig": TrainConfig, "train": train}[name]
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
