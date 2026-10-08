"""Check compiled pointwise LLVM against FP64/NCHW ATen convolution."""

import ctypes
import json
import shlex

from cases import MODEL_ROOT


def verify(case, out, config, command):
    import torch
    from torch.nn.functional import conv2d

    library = out / "pytorch-check.so"
    command([*shlex.split(config["HOST_CC"]), "-O2", "-ffp-contract=off",
             "-fno-vectorize", "-fno-slp-vectorize", "-shared", "-fPIC",
             "-DHOST_TEST", "-I", MODEL_ROOT, out / "kernel.ll",
             out.parent / "adapter.c", MODEL_ROOT / "support.c", "-o", library],
            log=out / "pytorch-compile.log")
    module = ctypes.CDLL(str(library))
    kernel = module.run_pointwise_conv2d
    kernel.argtypes = [ctypes.c_void_p] * 4
    kernel.restype = None
    shape, weight_shape, output_shape = case["shapes"][0], case["weight_shape"], case["output_shape"]
    constants = case["constexprs"]
    cin, cout, h, w, bm, bn = (constants[k] for k in ("CIN", "COUT", "H", "W", "BM", "BN"))
    generator = torch.Generator().manual_seed(0)
    maximum = maximum_aten = maximum_ratio = 0.0
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    trials, program_checks = 12, 0
    try:
        with torch.inference_mode():
            for trial in range(trials):
                x = torch.randn(shape, generator=generator, dtype=torch.float32)
                weight = torch.randn(weight_shape, generator=generator, dtype=torch.float32)
                bias = torch.randn(cout, generator=generator, dtype=torch.float32)
                if trial == 0:
                    x.zero_()
                    x[:, 1::2] = -0.0
                elif trial == 1:
                    x[:] = ((torch.arange(cin, dtype=torch.float32) + 1).reshape(1, cin, 1, 1) * 0.006
                            + torch.arange(h, dtype=torch.float32).reshape(1, 1, h, 1) * 0.031
                            - torch.arange(w, dtype=torch.float32).reshape(1, 1, 1, w) * 0.019)
                elif trial == 2:
                    x[:, 1:].zero_()
                elif trial == 3:
                    x[:, :-1].zero_()
                elif trial == 4:
                    weight.zero_()
                    channels = torch.arange(cout)
                    chosen = (channels * 7 + 3) % cin
                    weight[channels, chosen, 0, 0] = (channels % 17 - 8).float() / 32
                elif trial == 5:
                    weight[:-1].zero_()
                    bias[:-1].zero_()
                elif trial == 6:
                    x[:] = torch.where(torch.arange(cin) % 2 == 0, 0.25, -0.25).reshape(1, cin, 1, 1)
                    weight[:] = 0.75 + weight * 0.0001
                elif trial == 7:
                    x.zero_()
                    channels = torch.arange(1, cin + 1, dtype=torch.float32)
                    x[0, :, 0, 0] = channels / 32
                    x[0, :, 0, -1] = -channels / 16
                    x[0, :, -1, 0] = channels * 3 / 32
                    x[0, :, -1, -1] = -channels / 8
                else:
                    scale = (2.0**-4, 1.0, 2.0**4)[trial % 3]
                    x *= scale
                    weight *= scale
                    bias *= scale * scale
                originals = [t.view(torch.int32).clone() for t in (x, weight, bias)]
                reference = conv2d(x.double(), weight.double(), bias.double())
                magnitude = conv2d(x.double().abs(), weight.double().abs(), bias.double().abs())
                bound = case["tolerance"]["gamma"] * magnitude + case["tolerance"]["absolute_floor"]
                expected = torch.ops.aten.convolution.default(x, weight, bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
                actual = torch.full(output_shape, -9999.0, dtype=torch.float32)
                kernel(x.data_ptr(), weight.data_ptr(), bias.data_ptr(), actual.data_ptr())
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
                    raise RuntimeError(f"{case['name']} trial {trial}: convolution error exceeds bound; "
                                       f"max_error_over_bound={maximum_ratio}")
                if not all(torch.equal(t.view(torch.int32), original)
                           for t, original in zip((x, weight, bias), originals, strict=True)):
                    raise RuntimeError(f"{case['name']} trial {trial}: modified input/weight/bias")

            # Execute boundary and interior tile pairs independently; check
            # physical NCHW stores, including both M and N tails, with sentinels.
            class MemRef0(ctypes.Structure):
                _fields_ = [("allocated", ctypes.c_void_p), ("aligned", ctypes.c_void_p),
                            ("offset", ctypes.c_int64)]

            raw = getattr(module, case["symbol"])
            raw.argtypes = [ctypes.c_int64, ctypes.c_void_p] * 4 + [ctypes.c_int32] * 6
            raw.restype = None
            descriptors = [MemRef0(t.data_ptr(), t.data_ptr(), 0) for t in (x, weight, bias, actual)]
            arguments = [arg for descriptor in descriptors for arg in (0, ctypes.byref(descriptor))]
            gx, gy, _ = case["grid"]
            for py in sorted({0, gy // 2, gy - 1}, reverse=True):
                for px in sorted({0, gx // 2, gx - 1}, reverse=True):
                    actual.fill_(-9999.0)
                    raw(*arguments, *case["grid"], px, py, 0)
                    owned = torch.zeros(output_shape, dtype=torch.bool)
                    owned.view(cout, h * w)[py * bn:min((py + 1) * bn, cout),
                                             px * bm:min((px + 1) * bm, h * w)] = True
                    if (not torch.all(actual[~owned] == -9999.0)
                            or not torch.isfinite(actual[owned]).all()
                            or torch.any((actual.double() - reference).abs()[owned] > bound[owned])):
                        raise RuntimeError(f"{case['name']} program ({px},{py}): wrong NCHW store range/value")
                    program_checks += 1
    finally:
        torch.set_num_threads(previous_threads)
    report = {
        "case": case["name"], "status": "PASS", "torch_version": torch.__version__,
        "reference": "FP64 NCHW conv2d plus torch.ops.aten.convolution.default; groups=1, kernel=1x1",
        "shape": shape, "weight_shape": weight_shape, "output_shape": output_shape, "mnk": case["mnk"],
        "dtype": "float32", "device": "cpu", "seed": 0, "trials": trials,
        "samples": trials * cout * h * w, "per_program_store_checks": program_checks,
        "max_abs_error": maximum, "max_abs_error_vs_aten": maximum_aten,
        "max_error_over_bound": maximum_ratio, "tolerance": case["tolerance"],
        "aten_comparison": "each FP32 result within bound of FP64; pairwise difference <= 2*bound",
        "inputs": "bias-only, NCHW channel/spatial ramps, first/last input channel, one-hot channel routing, "
                  "last output only, cancellation, corner impulses, seeded random inputs/weights/bias",
        "compiled_program": str(out / "kernel.ll"), "library": str(library),
    }
    (out / "pytorch.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"aten pointwise PASS {case['name']} ({out.name}) max_abs_error={maximum:.9g} "
          f"max_error_over_bound={maximum_ratio:.9g}", flush=True)
    return report
