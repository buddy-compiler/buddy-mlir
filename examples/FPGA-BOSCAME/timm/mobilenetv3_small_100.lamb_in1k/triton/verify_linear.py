"""Run the actual scalar/RVV-lowered LLVM against FP64 and ATen Linear."""

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
    kernel = module.run_linear
    kernel.argtypes = [ctypes.c_void_p] * 4
    kernel.restype = None
    m, n, k = (case["constexprs"][name] for name in ("M", "N", "K"))
    generator = torch.Generator().manual_seed(0)
    maximum = maximum_aten = maximum_ratio = 0.0
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    trials = 16
    checked_programs = 0
    try:
        with torch.inference_mode():
            for trial in range(trials):
                x = torch.randn(m, k, generator=generator, dtype=torch.float32)
                weight = torch.randn(n, k, generator=generator, dtype=torch.float32)
                bias = torch.randn(n, generator=generator, dtype=torch.float32)
                if trial == 0:
                    x.zero_()
                    x[:, 1::2] = -0.0
                elif trial in (1, 2, 3):
                    x.zero_()
                    x[:, (0, k // 2, k - 1)[trial - 1]] = -0.5
                elif trial == 4:
                    x[:] = torch.where(torch.arange(k) % 2 == 0, 1.0, -1.0)
                    weight[:] = 4.0 + weight * 0.0001
                elif trial == 5:
                    weight[:-1].zero_()
                    bias[:-1].zero_()
                else:
                    scale = (2.0**-4, 1.0, 2.0**4)[trial % 3]
                    x *= scale
                    weight *= scale
                    bias *= scale * scale
                originals = [t.view(torch.int32).clone() for t in (x, weight, bias)]
                reference = torch.nn.functional.linear(x.double(), weight.double(), bias.double())
                magnitude = torch.nn.functional.linear(x.double().abs(), weight.double().abs(), bias.double().abs())
                bound = case["tolerance"]["gamma"] * magnitude + case["tolerance"]["absolute_floor"]
                expected = torch.ops.aten.linear.default(x, weight, bias)
                actual = torch.full((m, n), -9999.0, dtype=torch.float32)
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
                    raise RuntimeError(f"{case['name']} trial {trial}: dot error exceeds bound; "
                                       f"max_error_over_bound={maximum_ratio}")
                if not all(torch.equal(t.view(torch.int32), original)
                           for t, original in zip((x, weight, bias), originals, strict=True)):
                    raise RuntimeError(f"{case['name']} trial {trial}: modified input/weight/bias")
            # Check each program in reverse launch order with a fresh sentinel
            # output. Final tile reads overlap, but no output may have two
            # owners. This checks the actual compiled masks, not just a model
            # of the indexing formula or the default adapter's launch order.
            class MemRef0(ctypes.Structure):
                _fields_ = [("allocated", ctypes.c_void_p), ("aligned", ctypes.c_void_p),
                            ("offset", ctypes.c_int64)]

            raw = getattr(module, case["symbol"])
            raw.argtypes = [ctypes.c_int64, ctypes.c_void_p] * 4 + [ctypes.c_int32] * 6
            raw.restype = None
            descriptors = [MemRef0(t.data_ptr(), t.data_ptr(), 0) for t in (x, weight, bias, actual)]
            arguments = [arg for descriptor in descriptors for arg in (0, ctypes.byref(descriptor))]
            bn = case["constexprs"]["BN"]
            owners = torch.zeros((m, n), dtype=torch.int32)
            for pid in reversed(range(case["grid"][1])):
                actual.fill_(-9999.0)
                raw(*arguments, *case["grid"], 0, pid, 0)
                start = min(pid * bn, n - bn)
                end = n if pid == case["grid"][1] - 1 else min(start + bn, n - bn)
                owned = torch.zeros((m, n), dtype=torch.bool)
                owned[:, start:end] = True
                owners += owned.to(torch.int32)
                if (not torch.all(actual[~owned] == -9999.0)
                        or not torch.isfinite(actual[owned]).all()
                        or torch.any((actual.double() - reference).abs()[owned] > bound[owned])):
                    raise RuntimeError(f"{case['name']} program {pid}: wrong store ownership/value")
                checked_programs += 1
            if not torch.all(owners == 1):
                raise RuntimeError("Linear programs do not partition the output exactly once")
    finally:
        torch.set_num_threads(previous_threads)
    report = {
        "case": case["name"], "status": "PASS", "torch_version": torch.__version__,
        "reference": "FP64 torch.nn.functional.linear plus torch.ops.aten.linear.default",
        "shape": [m, k], "weight_shape": [n, k], "bias_shape": [n], "output_shape": [m, n],
        "dtype": "float32", "device": "cpu", "seed": 0, "trials": trials, "samples": trials * m * n,
        "per_program_store_checks": checked_programs, "output_owners_per_element": 1,
        "max_abs_error": maximum, "max_abs_error_vs_aten": maximum_aten,
        "max_error_over_bound": maximum_ratio, "tolerance": case["tolerance"],
        "aten_comparison": "each FP32 program within bound of FP64; pairwise difference <= 2*bound",
        "inputs": "bias-only, first/middle/last K basis, cancellation, last-output-channel-only, "
                  "seeded normal inputs/weights/bias at 2^-4/1/2^4 scales",
        "compiled_program": str(out / "kernel.ll"), "library": str(library),
    }
    (out / "pytorch.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"aten Linear PASS {case['name']} ({out.name}) max_abs_error={maximum:.9g} "
          f"max_error_over_bound={maximum_ratio:.9g}", flush=True)
    return report
