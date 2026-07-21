from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class PreparedFeatures:
    names: tuple[str, ...]
    values: np.ndarray[Any, Any]


@dataclass(frozen=True)
class ModelPrediction:
    score: float
    threshold: float
    threshold_version: str
    model_version: str


class TabularModelPort(Protocol):
    def predict(self, features: PreparedFeatures) -> ModelPrediction:
        ...


class ExplanationPort(Protocol):
    def explain(self, features: PreparedFeatures) -> tuple[str, str, list[dict[str, float | str]]]:
        ...
