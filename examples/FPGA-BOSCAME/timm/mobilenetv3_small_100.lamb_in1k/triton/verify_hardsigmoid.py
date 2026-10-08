"""Compare the compiled Host LLVM program with aten.hardsigmoid.default.

This supplements, and never replaces, each launcher's independent C oracle.
"""

import ctypes
import json
import shlex

from cases import MODEL_ROOT


def verify(case, out, config, command):
    import torch

    library = out / "pytorch-check.so"
    command([*shlex.split(config["HOST_CC"]), "-O2", "-ffp-contract=off",
             "-fno-vectorize", "-fno-slp-vectorize", "-shared", "-fPIC",
             "-DHOST_TEST", "-I", MODEL_ROOT, out / "kernel.ll",
             out.parent / "adapter.c", MODEL_ROOT / "support.c", "-o", library],
            log=out / "pytorch-compile.log")
    module = ctypes.CDLL(str(library))
    kernel = module.run_hardsigmoid
    kernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel.restype = None
    count = case["constexprs"]["COUNT"]
    shape = case["shapes"][0]
    edge_bits = torch.tensor([
        0xff800000, 0xff7fffff, 0xc0400001, 0xc0400000,
        0xc03fffff, 0xbf800000, 0x80000000, 0x00000000,
        0x3f800000, 0x403fffff, 0x40400000, 0x40400001,
        0x7f7fffff, 0x7f800000, 0x7fc00001, 0xffc00001,
    ], dtype=torch.int64).to(torch.int32)
    generator = torch.Generator().manual_seed(0)
    maximum = 0.0
    nan_count = 0
    with torch.inference_mode():
        for trial in range(35):
            if trial < 3:
                indices = (torch.arange(count) + trial * 5) % edge_bits.numel()
                x = edge_bits[indices].view(torch.float32).clone()
            elif trial % 2:
                x = torch.rand(count, generator=generator, dtype=torch.float32) * 16 - 8
            else:
                x = torch.randint(0, 2**32, (count,), generator=generator,
                                  dtype=torch.int64).to(torch.int32).view(torch.float32)
            x = x.reshape(shape)
            original = x.view(torch.int32).clone()
            expected = torch.ops.aten.hardsigmoid.default(x)
            actual = torch.full_like(x, -9999.0)
            kernel(x.data_ptr(), actual.data_ptr())
            nan = torch.isnan(expected)
            if not torch.equal(torch.isnan(actual), nan):
                raise RuntimeError(f"{case['name']} trial {trial}: aten NaN mismatch")
            error = (actual[~nan] - expected[~nan]).abs()
            if error.numel():
                maximum = max(maximum, error.max().item())
            if not torch.equal(actual[~nan], expected[~nan]):
                raise RuntimeError(f"{case['name']} trial {trial}: aten mismatch, max_abs_error={maximum}")
            if not torch.equal(x.view(torch.int32), original):
                raise RuntimeError(f"{case['name']} trial {trial}: modified input")
            nan_count += int(nan.sum())
    report = {
        "case": case["name"], "status": "PASS", "torch_version": torch.__version__,
        "reference": "torch.ops.aten.hardsigmoid.default", "shape": shape,
        "dtype": "float32", "device": "cpu", "seed": 0, "trials": 35,
        "samples": count * 35, "nan_samples": nan_count, "max_abs_error": maximum,
        "comparison": "exact FP32 numerical equality; NaNs compared by classification",
        "inputs": "3 edge patterns, 16 uniform [-8,8) patterns, 16 random FP32 bit patterns",
        "compiled_program": str(out / "kernel.ll"), "library": str(library),
    }
    (out / "pytorch.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"aten check PASS {case['name']} max_abs_error={maximum}", flush=True)
    return report
