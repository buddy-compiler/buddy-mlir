"""Independent host arithmetic with explicit FP32 rounding and reduction order.

This module does not load generated kernels or model graph objects. Its host
profile follows emitted scalar LLVM operations. The NR profile emulates the
observed RVV fused attention order using independent scalar fmaf loops and
reuses the public scalar NR math runtime under renamed symbols. The latter
therefore validates graph/quantized-kernel behavior, not that shared math code.
"""
import ctypes
import ctypes.util
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np


_libm = ctypes.CDLL(ctypes.util.find_library("m"))
_unary = {}
_nr_library = None
for _name in ("expf", "sinf", "cosf"):
    _function = getattr(_libm, _name)
    _function.argtypes = [ctypes.c_float]
    _function.restype = ctypes.c_float
    _unary[_name] = _function


def nr_library():
    """Build a host-only arithmetic oracle; never use a generated kernel."""
    global _nr_library
    if _nr_library is not None:
        return _nr_library
    source = Path(__file__).with_suffix(".c")
    model = source.parents[1]
    runtime = model.parents[1] / "common/nr/nr_math.c"
    fingerprint = hashlib.sha256(source.read_bytes() + runtime.read_bytes()).hexdigest()
    directory = model / "build/reference-support" / fingerprint[:16]
    directory.mkdir(parents=True, exist_ok=True)
    library = directory / "libreference_fp32.so"
    compiler = os.environ.get("HOST_CC", "cc")
    command = [compiler, "-shared", "-fPIC", "-O2", "-fno-builtin", "-ffp-contract=off",
               *[f"-D{name}=qwen_ref_nr_{name}" for name in
                 ("expf", "sinf", "cosf", "logf", "powf", "tanhf", "erff", "sqrtf")],
               str(source), str(runtime), "-lm", "-o", str(library)]
    if not library.exists():
        temporary = library.with_name(f"{library.name}.{os.getpid()}.tmp")
        subprocess.run(command[:-1] + [str(temporary)], check=True)
        temporary.replace(library)
    (directory / "build.json").write_text(json.dumps({
        "command": command, "source_sha256": fingerprint,
        "scope": "independent host FMA dot; shared NR scalar libm reused with renamed symbols; no generated kernels or model graph"}, indent=2) + "\n")
    _nr_library = ctypes.CDLL(str(library))
    pointer = ctypes.POINTER(ctypes.c_float)
    _nr_library.qwen_ref_nr_unary.argtypes = [pointer, pointer, ctypes.c_size_t, ctypes.c_int]
    _nr_library.qwen_ref_nr_dot.argtypes = [pointer, pointer, pointer] + [ctypes.c_size_t] * 5
    return _nr_library


def unary(name, value, profile="triton-host"):
    value = np.asarray(value, dtype=np.float32)
    if profile == "nr-fpga":
        value = np.ascontiguousarray(value)
        result = np.empty_like(value)
        pointer = ctypes.POINTER(ctypes.c_float)
        nr_library().qwen_ref_nr_unary(value.ctypes.data_as(pointer), result.ctypes.data_as(pointer),
                                      value.size, {"expf": 0, "sinf": 1, "cosf": 2}[name])
        return result
    fn = _unary[name]
    return np.fromiter((fn(float(x)) for x in value.flat), dtype=np.float32,
                       count=value.size).reshape(value.shape)


def ordered_sum(value, *, keepdims=False):
    value = np.asarray(value, dtype=np.float32)
    # A leading +0 records the reduction's initializer, including signed zero.
    initial = np.zeros(value.shape[:-1] + (1,), dtype=np.float32)
    result = np.add.accumulate(np.concatenate((initial, value), axis=-1),
                               axis=-1, dtype=np.float32)[..., -1:]
    return result if keepdims else result[..., 0]


def blocked_dot(left, right, block_k=64, profile="triton-host"):
    """Batch A[...,M,K] @ B[...,K,N], scalar FP32 dot partials of BK=64."""
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if left.shape[:-2] != right.shape[:-2] or left.shape[-1] != right.shape[-2]:
        raise ValueError("incompatible batched dot shapes")
    result = np.zeros(left.shape[:-1] + (right.shape[-1],), dtype=np.float32)
    if profile == "nr-fpga":
        left, right = np.ascontiguousarray(left), np.ascontiguousarray(right)
        pointer = ctypes.POINTER(ctypes.c_float)
        nr_library().qwen_ref_nr_dot(left.ctypes.data_as(pointer), right.ctypes.data_as(pointer),
                                    result.ctypes.data_as(pointer), int(np.prod(left.shape[:-2])),
                                    left.shape[-2], right.shape[-1], left.shape[-1], block_k)
        return result
    for start in range(0, left.shape[-1], block_k):
        a = left[..., :, start:start + block_k]
        b = right[..., start:start + block_k, :].swapaxes(-1, -2)
        products = a[..., :, None, :] * b[..., None, :, :]
        result = result + ordered_sum(products)
    return result


def softmax(score, mask, profile="triton-host"):
    masked = np.asarray(score + mask, dtype=np.float32)
    maximum = np.max(masked, axis=-1, keepdims=True)
    exp = unary("expf", masked - maximum, profile)
    return exp / ordered_sum(exp, keepdims=True)


def silu(value, profile="triton-host"):
    value = np.asarray(value, dtype=np.float32)
    return value / (np.float32(1) + unary("expf", -value, profile))
