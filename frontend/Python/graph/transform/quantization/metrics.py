from __future__ import annotations

import numpy as np


def nmse(ref: np.ndarray, hat: np.ndarray) -> float:
    r = np.asarray(ref, dtype=np.float32).reshape(-1)
    h = np.asarray(hat, dtype=np.float32).reshape(-1)
    if r.shape != h.shape:
        raise ValueError("nmse shape mismatch")
    den = float(np.mean(r * r))
    if den == 0.0:
        raise ValueError("nmse denom 0")
    return float(np.mean((r - h) ** 2) / den)
