from __future__ import annotations

from typing import Any

import numpy as np


MONEY_SCALE = 100_000_000.0
COUNT_SCALE = 120.0
MONTH_SCALE = 240.0


def preprocess_feature_vector(values: list[float]) -> np.ndarray[Any, Any]:
    transformed = np.asarray(values, dtype=np.float32)
    log_scaled_indices = (0, 1, 4, 7)
    for index in log_scaled_indices:
        transformed[index] = np.log1p(max(float(transformed[index]), 0.0)) / np.log1p(MONEY_SCALE)
    transformed[2] = _clip_unit(float(transformed[2]) / 10.0)
    transformed[3] = _clip_unit(float(transformed[3]) / 5.0)
    transformed[5] = _clip_unit(float(transformed[5]) / COUNT_SCALE)
    transformed[6] = _clip_unit(float(transformed[6]) / MONTH_SCALE)
    transformed[8] = _clip_unit(float(transformed[8]) / 10.0)
    transformed[9] = _clip_unit(float(transformed[9]) / MONTH_SCALE)
    transformed[10] = 1.0 if transformed[10] >= 0.5 else 0.0
    transformed[11] = _clip_unit(float(transformed[11]) / COUNT_SCALE)
    return transformed.reshape(1, -1)


def _clip_unit(value: float) -> float:
    return min(max(value, 0.0), 1.0)
