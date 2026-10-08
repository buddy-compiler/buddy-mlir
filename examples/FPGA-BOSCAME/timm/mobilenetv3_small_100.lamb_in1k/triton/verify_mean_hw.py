"""Check the compiled FP32 reduction against FP64 and aten.mean.dim.

The per-channel error bound scales with mean(abs(X)), including cancellation.
The separate C oracle is mandatory and uses independent double c/h/w loops.
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
    kernel = module.run_mean_hw
    kernel.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel.restype = None
    shape = case["shapes"][0]
    output_shape = case["output_shape"]
    channels = shape[1]
    spatial = case["constexprs"]["SPATIAL"]
    generator = torch.Generator().manual_seed(0)
    maximum = maximum_aten = maximum_ratio = 0.0
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.inference_mode():
            for trial in range(20):
                channel_values = torch.arange(1, channels + 1, dtype=torch.float32)
                channel_values[1::2] *= -1
                if trial == 0:
                    x = torch.zeros(shape, dtype=torch.float32)
                    x[:, 1::2] = -0.0
                elif trial == 1:
                    x = (channel_values / 8).reshape(output_shape).expand(shape).clone()
                elif trial in (2, 3):
                    x = torch.zeros(shape, dtype=torch.float32)
                    if trial == 2:
                        x[0, :, -1, -1] = channel_values / 4
                    else:
                        x[0, :, 0, 0] = channel_values / 4
                elif trial == 4:
                    plane = torch.where(torch.arange(spatial) % 2 == 0, 32.0, -32.0)
                    x = plane.repeat(channels).reshape(shape)
                    x += (torch.rand(shape, generator=generator) - 0.5) * 0.001
                elif trial == 5:
                    plane = 1.0 + (torch.arange(spatial, dtype=torch.float32) % 31) * 0.0001
                    x = plane.repeat(channels).reshape(shape)
                    x += (channel_values % 7).reshape(output_shape) * 0.1
                else:
                    x = torch.rand(shape, generator=generator, dtype=torch.float32) * 16 - 8
                    x *= (2.0**-10, 1.0, 2.0**10)[trial % 3]
                original = x.view(torch.int32).clone()
                reference = x.double().mean(dim=(2, 3), keepdim=True)
                magnitude = x.double().abs().mean(dim=(2, 3), keepdim=True)
                bound = case["tolerance"]["gamma"] * magnitude + case["tolerance"]["absolute_floor"]
                expected = torch.ops.aten.mean.dim(x, [2, 3], True)
                actual = torch.full(output_shape, -9999.0, dtype=torch.float32)
                kernel(x.data_ptr(), actual.data_ptr())
                if not torch.isfinite(actual).all():
                    raise RuntimeError(f"{case['name']} trial {trial}: nonfinite result")
                error = (actual.double() - reference).abs()
                error_aten = (expected.double() - reference).abs()
                delta_aten = (actual.double() - expected.double()).abs()
                ratio = error / bound
                maximum = max(maximum, error.max().item())
                maximum_aten = max(maximum_aten, delta_aten.max().item())
                maximum_ratio = max(maximum_ratio, ratio.max().item())
                if (torch.any(error > bound) or torch.any(error_aten > bound)
                        or torch.any(delta_aten > 2 * bound)):
                    raise RuntimeError(f"{case['name']} trial {trial}: reduction error exceeds bound; "
                                       f"max_error_over_bound={maximum_ratio}")
                if not torch.equal(x.view(torch.int32), original):
                    raise RuntimeError(f"{case['name']} trial {trial}: modified input")
    finally:
        torch.set_num_threads(previous_threads)
    report = {
        "case": case["name"], "status": "PASS", "torch_version": torch.__version__,
        "reference": "FP64 mean plus torch.ops.aten.mean.dim(X,[2,3],True)",
        "shape": shape, "output_shape": output_shape, "dtype": "float32", "device": "cpu",
        "seed": 0, "trials": 20, "samples": channels * 20,
        "input_samples": case["constexprs"]["COUNT"] * 20,
        "max_abs_error": maximum, "max_abs_error_vs_aten": maximum_aten,
        "max_error_over_bound": maximum_ratio, "tolerance": case["tolerance"],
        "aten_comparison": "both FP32 programs within bound of FP64; pairwise difference <= 2*bound",
        "inputs": "signed zeros, distinct channel constants, first/last impulses, cancellation, "
                  "positive small increments, random inputs at 2^-10/1/2^10 scales",
        "compiled_program": str(out / "kernel.ll"), "library": str(library),
    }
    (out / "pytorch.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"aten mean PASS {case['name']} max_abs_error={maximum:.9g} "
          f"max_error_over_bound={maximum_ratio:.9g}", flush=True)
    return report
