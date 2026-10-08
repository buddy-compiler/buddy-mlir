"""Compare the actual compiled program with PyTorch NCHW broadcasting.

Scale is always allocated as [1,C,1,1], never materialized as [1,C,H,W].
This supplements the independent C oracle and its guarded memory checks.
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
    kernel = module.run_se_mul
    kernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    kernel.restype = None
    shape = case["shapes"][0]
    scale_shape = case["scale_shape"]
    channels = shape[1]
    generator = torch.Generator().manual_seed(0)
    maximum = 0.0
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.inference_mode():
            for trial in range(16):
                x = torch.rand(shape, generator=generator, dtype=torch.float32) * 16 - 8
                if trial == 0:
                    x.fill_(1.0)
                    scale = (torch.arange(channels, dtype=torch.float32) + 1) / 1024
                elif trial == 1:
                    scale = torch.arange(channels, 0, -1, dtype=torch.float32) / 1024
                elif trial == 2:
                    scale = torch.arange(1, channels + 1, dtype=torch.float32) / 256
                    scale[1::2] *= -1
                    scale[0::8] = 0.0
                    scale[1::8] = -0.0
                    scale[2::8] = 1.0
                    scale[3::8] = -1.0
                    x[:, 0::2, 0, 0] = 0.0
                    x[:, 1::2, 0, 0] = -0.0
                elif trial == 3:
                    scale = torch.zeros(channels, dtype=torch.float32)
                    scale[-1] = 1.0
                else:
                    scale = torch.rand(channels, generator=generator, dtype=torch.float32)
                    if trial % 2:
                        scale = scale * 4 - 2
                scale = scale.reshape(scale_shape)
                assert scale.numel() == channels and scale.is_contiguous()
                original_x = x.view(torch.int32).clone()
                original_scale = scale.view(torch.int32).clone()
                expected = torch.ops.aten.mul.Tensor(x, scale)
                actual = torch.full_like(x, -9999.0)
                kernel(x.data_ptr(), scale.data_ptr(), actual.data_ptr())
                if not torch.isfinite(actual).all():
                    raise RuntimeError(f"{case['name']} trial {trial}: nonfinite result")
                maximum = max(maximum, (actual - expected).abs().max().item())
                if not torch.equal(actual, expected):
                    raise RuntimeError(f"{case['name']} trial {trial}: "
                                       f"NCHW broadcasting mismatch, max_abs_error={maximum}")
                if (not torch.equal(x.view(torch.int32), original_x)
                        or not torch.equal(scale.view(torch.int32), original_scale)):
                    raise RuntimeError(f"{case['name']} trial {trial}: modified input")
    finally:
        torch.set_num_threads(previous_threads)
    report = {
        "case": case["name"], "status": "PASS", "torch_version": torch.__version__,
        "reference": "torch.ops.aten.mul.Tensor(X, Scale)", "shape": shape,
        "scale_shape": scale_shape, "scale_elements": channels,
        "dtype": "float32", "device": "cpu", "seed": 0, "trials": 16,
        "samples": case["constexprs"]["COUNT"] * 16, "max_abs_error": maximum,
        "comparison": "exact finite FP32 numerical equality; both zero signs accepted",
        "inputs": "unique/reversed channel scales; signed scales and zeros; last-channel-only; "
                  "12 random patterns with Scale in [0,1) or [-2,2)",
        "compiled_program": str(out / "kernel.ll"), "library": str(library),
    }
    (out / "pytorch.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"aten broadcasting PASS {case['name']} max_abs_error={maximum}", flush=True)
    return report
