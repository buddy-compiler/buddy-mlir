#!/usr/bin/env python3
"""Real Triton AST -> TTIR -> triton-riscv Linalg, without a GPU runtime."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cases import FAMILIES, ROOT, select


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def adapter(case):
    gx, gy, gz = case["grid"]
    buffers = [name.lower() for name in case["signature"]]
    entry = "run_add" if case["family"] == "residual_add" else "run_" + case["family"]
    declaration = ", ".join(["int64_t, MemRef0 *"] * len(buffers) + ["int32_t"] * 6)
    parameters = ", ".join("float *" + name for name in buffers)
    descriptors = ", ".join(f"m{i} = {{{name}, {name}, 0}}" for i, name in enumerate(buffers))
    arguments = ", ".join(f"0, &m{i}" for i in range(len(buffers)))
    grid_loop = (f"  for (int32_t pid = 0; pid < {gx}; ++pid)\n"
                 f"    {case['symbol']}({arguments}, {gx}, {gy}, {gz}, pid, 0, 0);\n")
    if gy != 1 or gz != 1:
        grid_loop = (f"  for (int32_t z = 0; z < {gz}; ++z)\n"
                     f"    for (int32_t y = 0; y < {gy}; ++y)\n"
                     f"      for (int32_t x = 0; x < {gx}; ++x)\n"
                     f"        {case['symbol']}({arguments}, {gx}, {gy}, {gz}, x, y, z);\n")
    return f'''/* Generated ABI/grid glue; no tensor arithmetic. */
#include <stdint.h>
typedef struct {{ void *allocated, *aligned; int64_t offset; }} MemRef0;
extern void {case["symbol"]}({declaration});
void {entry}({parameters}) {{
  MemRef0 {descriptors};
{grid_loop.rstrip()}
}}
'''


def compile_case(case, build_root):
    import triton
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.backends.triton_shared.compiler import CPUBackend
    from triton.backends.triton_shared.paths import _get_triton_shared_opt_path
    from triton.compiler import ASTSource
    import kernels

    kernel = getattr(kernels, case["kernel"])

    backend = CPUBackend(GPUTarget("cpu", 0, 0))
    options = backend.parse_options({"enable_fp_fusion": False})
    source = ASTSource(fn=kernel, signature=case["signature"],
                       constexprs=case["constexprs"])
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = source.make_ir(backend.target, options,
                            backend.get_codegen_implementation(options),
                            backend.get_module_map(), context)
    module = backend.make_ttir(module, {}, options)
    ttir = str(module)
    pattern = "@" + re.escape(kernel.__name__) + r"(?=[\s(])"
    if not re.search(pattern, ttir):
        raise RuntimeError("Missing compiled Triton entry symbol")
    out = build_root / case["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "kernel.ttir").write_text(re.sub(pattern, "@" + case["symbol"], ttir))
    # Use the converter's standard Linalg pipeline explicitly. Some installed
    # backends select an already vectorized CPU pipeline in their Python helper.
    converter = _get_triton_shared_opt_path()
    pipeline = "--triton-to-linalg-experimental"
    # Linear uses the same tile materialization as pointwise Conv: loads still
    # address physical Weight[N,K], but each program gets a local [BK,BN] tile.
    # Buddy can then vectorize N without an RVV horizontal FP reduction.
    conversion = [converter, str(out / "kernel.ttir"),
                  pipeline, "-o", str(out / "kernel.linalg.mlir")]
    subprocess.run(conversion, check=True)
    expected_op = {"mean_hw": "linalg.reduce", "linear": "linalg.matmul",
                   "pointwise_conv2d": "linalg.matmul", "conv_stem": "linalg.matmul"}.get(case["family"], "linalg.generic")
    if case["family"] == "linear" and ("tt.dot" not in ttir or "tt.trans" in ttir):
        raise RuntimeError("Linear must contain a real Triton dot without a transpose")
    if expected_op not in (out / "kernel.linalg.mlir").read_text():
        raise RuntimeError(f"Expected actual {expected_op} computation from triton-riscv")
    if case["family"] in ("pointwise_conv2d", "conv_stem") and "tt.dot" not in ttir:
        raise RuntimeError("Convolution must contain a genuine Triton dot")
    if case["family"] == "depthwise_conv2d":
        linalg = (out / "kernel.linalg.mlir").read_text()
        if "tt.dot" in ttir or re.search(r"linalg\.(?:batch_)?matmul", linalg):
            raise RuntimeError("Depthwise must use direct per-channel convolution, not dense GEMM")
        if "arith.mulf" not in ttir or "arith.addf" not in ttir:
            raise RuntimeError("Missing genuine depthwise FP32 multiply/accumulate")
    (out / "adapter.c").write_text(adapter(case))
    manifest = {
        "case": case, "frontend": "ASTSource -> CPUBackend.make_ttir -> triton-riscv triton-to-linalg-experimental",
        "triton_version": triton.__version__,
        "backend_source": str(Path(sys.modules[CPUBackend.__module__].__file__).resolve()),
        "backend_sha256": sha(sys.modules[CPUBackend.__module__].__file__),
        "triton_shared_opt": converter, "triton_shared_opt_sha256": sha(converter),
        "conversion_command": conversion,
        "sources": {name: sha(ROOT / name) for name in ("kernels.py", "cases.py", "export.py")},
        "artifacts": {name: sha(out / name) for name in ("kernel.ttir", "kernel.linalg.mlir", "adapter.c")},
    }
    (out / "frontend.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("exported " + case["name"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append")
    parser.add_argument("--family", choices=FAMILIES, default="residual_add")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "build")
    args = parser.parse_args()
    cases = select(args.case, args.family)
    if args.list:
        print(json.dumps(cases, indent=2))
        return
    for case in cases:
        compile_case(case, args.output)


if __name__ == "__main__":
    main()
