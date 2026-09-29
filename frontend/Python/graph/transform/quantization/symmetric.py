from __future__ import annotations

import numpy as np


def quantize_symmetric(
    w: np.ndarray, axes: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    if w.dtype != np.float32:
        raise ValueError(f"expected float32, got {w.dtype}")
    if len(axes) != len(set(axes)):
        raise ValueError("duplicate axes")
    if any(a < 0 or a >= w.ndim for a in axes):
        raise ValueError("axes out of range")
    abs_w = np.abs(w)
    if not axes:
        max_abs = float(abs_w.max())
        if max_abs == 0.0:
            raise ValueError("maxAbs is 0")
        dw = np.float32(max_abs / 127.0)
        q = np.clip(np.rint(w / dw), -128, 127).astype(np.int8)
        return q, np.array(dw)
    reduce_axes = tuple(i for i in range(w.ndim) if i not in axes)
    max_abs = abs_w.max(axis=reduce_axes, keepdims=True)
    if np.any(max_abs == 0):
        raise ValueError("maxAbs is 0")
    dw_keep = (max_abs / 127.0).astype(np.float32)
    q = np.clip(np.rint(w / dw_keep), -128, 127).astype(np.int8)
    dw = np.squeeze(dw_keep, axis=reduce_axes)
    return q, dw.astype(np.float32)
