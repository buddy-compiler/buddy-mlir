#!/usr/bin/env python3
"""Compile real Triton Python ASTs into TTIR and triton-riscv Linalg IR."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cases import ROOT, describe, inventory

REPO = ROOT.parents[3]


def configure_environment():
    plugin = REPO / "thirdparty/triton-riscv"
    os.environ.setdefault("TRITON_RISCV_DIR", str(plugin))
    os.environ.setdefault("TRITON_PLUGIN_DIRS", str(plugin))
    if not os.environ.get("TRITON_SHARED_OPT_PATH"):
        for checkout in (plugin / "triton", REPO / "thirdparty/triton"):
            candidates = sorted((checkout / "build").glob(
                "*/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt"))
            if candidates:
                os.environ["TRITON_SHARED_OPT_PATH"] = str(candidates[-1])
                break


def make_adapter(case):
    """Descriptor/grid ABI glue only; all numerical work is in the JIT kernel."""
    arguments = case["arguments"]
    signature = ", ".join(f"MemRef{arg['rank']} *a{i}" for i, arg in enumerate(arguments))
    raw_signature = ", ".join(["int64_t, MemRef0 *"] * len(arguments) + ["int32_t"] * 6)
    lines = ['#include "support.h"',
             "typedef struct { void *allocated, *aligned; int64_t offset; } MemRef0;",
             f"extern void {case['symbol']}({raw_signature});",
             f"void _mlir_ciface_kernel_{case['name']}({signature}) {{"]
    item_sizes = {"f32": 4, "i8": 1, "i32": 4, "i64": 8}
    for i, arg in enumerate(arguments):
        lines += [f"  void *p{i} = (unsigned char *)a{i}->aligned + a{i}->offset * {item_sizes[arg['dtype']]};",
                  f"  MemRef0 m{i} = {{p{i}, p{i}, 0}};"]
    gx, gy, gz = case["grid"]
    params = ", ".join(f"0, &m{i}" for i in range(len(arguments)))
    lines += [f"  for (int32_t x=0; x<{gx}; ++x)",
              f"    for (int32_t y=0; y<{gy}; ++y)",
              f"      for (int32_t z=0; z<{gz}; ++z)",
              f"        {case['symbol']}({params}, {gx}, {gy}, {gz}, x, y, z);",
              "}", ""]
    return "\n".join(lines)


def compile_case(case, output_root):
    configure_environment()
    import triton
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.backends.triton_shared.compiler import CPUBackend, _ttir_to_ttsharedir
    from triton.compiler import ASTSource
    import importlib

    kernel_module = importlib.import_module(case.get("kernel_module", "kernels"))
    fn = getattr(kernel_module, case["kernel"])
    target = GPUTarget("cpu", 0, 0)
    backend = CPUBackend(target)
    options = backend.parse_options({"enable_fp_fusion": False})
    source = ASTSource(fn=fn, signature=case["signature"], constexprs=case["constexprs"])
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = source.make_ir(target, options, backend.get_codegen_implementation(options),
                            backend.get_module_map(), context)
    module = backend.make_ttir(module, {}, options)
    ttir = str(module)
    linalg = _ttir_to_ttsharedir(module)
    # This is only symbol namespacing, after genuine Triton/triton-riscv passes.
    # No computations, loop bodies or linalg operations are synthesized here.
    old_symbol = fn.__name__
    pattern = r"@" + re.escape(old_symbol) + r"(?=[\s(])"
    if not re.search(pattern, ttir) or not re.search(pattern, linalg):
        raise RuntimeError(f"Cannot identify compiled entry symbol {old_symbol!r}")
    ttir = re.sub(pattern, "@" + case["symbol"], ttir)
    linalg = re.sub(pattern, "@" + case["symbol"], linalg)
    output = Path(output_root) / case["name"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "kernel.ttir").write_text(ttir)
    (output / "kernel.linalg.mlir").write_text(linalg)
    (output / "adapter.c").write_text(make_adapter(case))
    manifest = dict(case)
    manifest.update(frontend="triton.compiler.ASTSource -> CPUBackend.make_ttir -> triton-riscv _ttir_to_ttsharedir",
                    triton_version=triton.__version__,
                    kernels_sha256=hashlib.sha256((ROOT / "kernels.py").read_bytes()).hexdigest(),
                    kernel_module_sha256=hashlib.sha256(Path(kernel_module.__file__).read_bytes()).hexdigest(),
                    exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    cases_sha256=hashlib.sha256((ROOT / "cases.py").read_bytes()).hexdigest(),
                    ttir_sha256=hashlib.sha256(ttir.encode()).hexdigest(),
                    linalg_sha256=hashlib.sha256(linalg.encode()).hexdigest(),
                    abi="unranked memref: (i64 rank=0, rank-0 descriptor pointer) per tensor, then six i32 grid/pid values")
    (output / "frontend.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"exported {case['name']}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "build")
    args = parser.parse_args()
    cases = [describe(name) for name in args.cases] if args.cases else inventory()
    if args.list:
        print(json.dumps(cases, indent=2))
        return
    for case in cases:
        compile_case(case, args.output)


if __name__ == "__main__":
    main()
