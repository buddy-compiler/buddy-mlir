#!/usr/bin/env python3
"""Link existing baseline/RVV Triton builds into one independently checked image.

Inputs are case directories containing frontend.json, adapter.c, and nr/kernel.o
and nr/adapter.o. No kernel numerical computation is synthesized here. Original
objects are copied and hashed before symbol renaming; the source inputs remain
untouched. --host instead compiles the same lowered LLVM IR for host validation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent
QWEN = MODEL.parent
REPO = MODEL.parents[3]
COMMON = QWEN.parent / "common"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(command, log=None):
    result = subprocess.run(list(map(str, command)), text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if log:
        Path(log).write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"command failed: {command!r}\n{result.stdout}")
    return result.stdout


def tool(name, llvm_bin):
    path = llvm_bin / name
    if not path.is_file():
        found = shutil.which(name)
        if not found:
            raise ValueError(f"tool unavailable: {name}; set --llvm-bin")
        path = Path(found)
    # Keep argv[0] spelling: resolving clang symlinks can select another driver.
    return str(path.absolute())


def inspect_case(directory, benchmark="dequant"):
    manifest = json.loads((directory / "frontend.json").read_text())
    constants = manifest["constexprs"]
    arguments = [(a["rank"], a["dtype"]) for a in manifest["arguments"]]
    if benchmark == "dequant":
        if manifest.get("family") != "dequantization":
            raise ValueError(f"not a dequantization case: {directory}")
        shape = (constants["ROWS"], constants["COLS"])
        if arguments != [(2, "i32"), (1, "f32"), (1, "f32"), (2, "f32")]:
            raise ValueError(f"unexpected input/output ABI: {arguments}")
        rows, cols = shape
        block = constants["BLOCK"]
        if block < 1 or block & (block - 1):
            raise ValueError("expected a positive power-of-two dequantization block")
        if manifest["kernel"] == "dequantize" and manifest.get("kernel_module", "kernels") == "kernels":
            expected_grid = [(rows * cols + block - 1) // block, 1, 1]
        elif manifest["kernel"] == "dequantize_rows" and manifest.get("kernel_module") == "kernels_dequant":
            expected_grid = [(cols + block - 1) // block, rows, 1]
        else:
            raise ValueError("unsupported kernel/module contract")
    elif benchmark == "quant":
        if (manifest.get("family") != "per_token_quantization" or
                manifest["kernel"] != "quantize" or manifest.get("kernel_module", "kernels") != "kernels"):
            raise ValueError("quant benchmark requires the production Triton quantize")
        if arguments != [(2, "f32"), (2, "i8"), (1, "f32")]:
            raise ValueError("unexpected quantization descriptor ABI")
        shape = (manifest["grid"][0], constants["WIDTH"])
        if constants["BLOCK"] < shape[1] or constants["BLOCK"] & (constants["BLOCK"] - 1):
            raise ValueError("quantization block must be a power of two covering WIDTH")
        expected_grid = [shape[0], 1, 1]
    elif benchmark == "ame":
        if manifest.get("family") != "matmul_i8" or manifest["kernel"] != "linear" or manifest.get("kernel_module", "kernels") != "kernels":
            raise ValueError("AME benchmark requires the production INT8 Triton linear")
        shape = (constants["M"], constants["N"], constants["K"])
        if arguments != [(2, "i8"), (2, "i8"), (2, "i32")] or constants["INTEGER"] is not True:
            raise ValueError("wrong INT8 linear dtype/descriptor contract")
        bm, bn = constants["BM"], constants["BN"]
        if min(bm, bn, constants["BK"]) < 1:
            raise ValueError("invalid INT8 linear tiling")
        expected_grid = [(shape[0] + bm - 1) // bm, (shape[1] + bn - 1) // bn, 1]
        if shape[2] > 65535:
            raise ValueError("benchmark oracle bound requires K <=65535")
    else:
        raise ValueError(f"unsupported benchmark: {benchmark}")
    if manifest["grid"] != expected_grid:
        raise ValueError(f"unexpected grid: {manifest['grid']} != {expected_grid}")
    build_manifest = json.loads((directory / "nr/manifest.json").read_text())
    for key in ("name", "family", "kernel", "symbol", "arguments", "signature", "constexprs", "grid", "quantization_lowering"):
        if build_manifest["case"].get(key) != manifest.get(key):
            raise ValueError(f"frontend/native case mismatch: {key}")
    for key, relative in (("frontend_manifest_sha256", "frontend.json"),
                          ("ttir_sha256", "kernel.ttir"),
                          ("linalg_sha256", "kernel.linalg.mlir"),
                          ("llvm_sha256", "nr/kernel.ll"),
                          ("adapter_sha256", "adapter.c")):
        if build_manifest.get(key) != sha(directory / relative):
            raise ValueError(f"stale native build: {relative} does not match {key}")
    for key, relative in (("ttir_sha256", "kernel.ttir"), ("linalg_sha256", "kernel.linalg.mlir")):
        if manifest.get(key) != sha(directory / relative):
            raise ValueError(f"stale frontend: {relative}")
    # Verify the original exporter produced exactly the adapter being linked;
    # this preserves rank/offset/grid conversion without adding another adapter.
    sys.path.insert(0, str(QWEN / "triton"))
    from export import make_adapter
    if (directory / "adapter.c").read_text() != make_adapter(manifest):
        raise ValueError("adapter differs from the declared descriptor/grid ABI")
    llvm = (directory / "nr/kernel.ll").read_text()
    entry = re.search(r"define\s+void\s+@" + re.escape(manifest["symbol"]) + r"\((.*?)\)", llvm, re.S)
    actual = [part.strip().split()[0] for part in entry.group(1).split(",")] if entry else []
    if actual != ["i64", "ptr"] * len(arguments) + ["i32"] * 6:
        raise ValueError("lowered raw entry ABI differs from unranked descriptors plus grid")
    return manifest, shape


def main(benchmark="dequant"):
    description = (__doc__ if benchmark in ("dequant", "quant") else
                   "Link existing INT8 Triton linear kernel/adapter variants with the shared NR runtime; "
                   "check accumulation using the independent AME A/B oracle.")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--optimized", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3 if benchmark == "dequant" else 2)
    if benchmark == "ame":
        parser.add_argument("--full-input-scan", action="store_true")
    parser.add_argument("--llvm-bin", type=Path,
                        default=REPO / "llvm/build-2d26/bin")
    parser.add_argument("--linker", type=Path, default=Path("/usr/bin/ld.lld-20"))
    parser.add_argument("--host", action="store_true")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("at least two alternating repetitions required")
    baseline, shape = inspect_case(args.baseline, benchmark)
    optimized, other_shape = inspect_case(args.optimized, benchmark)
    if shape != other_shape:
        parser.error(f"shape mismatch: {shape} != {other_shape}")
    if min(shape) < 1:
        parser.error("empty shapes unsupported")
    if benchmark in ("dequant", "quant"):
        prefix = "DQ" if benchmark == "dequant" else "Q"
        definitions = [f"-D{prefix}_ROWS={shape[0]}", f"-D{prefix}_COLS={shape[1]}", f"-D{prefix}_REPEATS={args.repeats}"]
    else:
        definitions = [f"-DAM_M={shape[0]}", f"-DAM_N={shape[1]}", f"-DAM_K={shape[2]}",
                       f"-DAM_REPEATS={args.repeats}", f"-DAM_FULL_INPUT_SCAN={int(args.full_input_scan)}"]
    stem = f"{benchmark}-ab"
    shape_name = "x".join(map(str, shape))
    build = (args.output or MODEL / "build/ame-v05" /
             f"{stem}-{shape_name}{'-host' if args.host else ''}").resolve()
    if build.exists() and any(build.iterdir()):
        parser.error(f"output is not empty: {build}; use a new --output to preserve prior evidence")
    build.mkdir(parents=True, exist_ok=True)
    clang = tool("clang", args.llvm_bin)
    objcopy = tool("llvm-objcopy", args.llvm_bin)
    objdump = tool("llvm-objdump", args.llvm_bin)
    nm = tool("llvm-nm", args.llvm_bin)
    harness = HERE / f"{benchmark}_ab_launch.c"
    provenance = {"benchmark": benchmark, "shape": list(shape), "repeats": args.repeats,
                  "host": args.host, "variants": {},
                  "oracle": "independent C int32->FP32, multiply row then column; exact bits",
                  "timing": "existing descriptor adapter including unchanged grid loop",
                  "fence_policy": "no added hardware fence; matches shipped RVV-only adapter",
                  "harness_sha256": sha(harness),
                  "builder_sha256": sha(Path(__file__)),
                  "source_hashes_and_abi_checked": True}
    if benchmark == "quant":
        provenance.update(oracle="independent FP32 absmax/divide/signed half/truncate/clamp; exact scale bits and all int8",
                          input_integrity="full", semantics="overwrite")
    if benchmark == "ame":
        provenance.update(oracle="independent int64 dots for32 B templates, exact all-output C+=A@B check",
                          timing="adapter and kernel plus post ame_fence; pre/post sync separately reported",
                          fence_policy="common ame_fence before and after each adapter call; kernel unchanged",
                          input_integrity="full" if args.full_input_scan else "full_A_sampled_B",
                          semantics="accumulate", b_layout="logical[K,N], physical[N,K], strides[1,K]")
    runtime_sources = [QWEN / "support.h", COMMON / "toolchain.mk", HERE / "dequant_ab.mk"]
    runtime_sources.extend(path for path in (COMMON / "nr").iterdir()
                           if path.suffix in (".c", ".h", ".S", ".ld", ".mk"))
    runtime_sources.append(COMMON / "uart/uart.h")
    provenance["runtime_sources_sha256"] = {str(p.relative_to(REPO)): sha(p)
                                            for p in sorted(runtime_sources)}
    objects = []
    for label, directory, manifest in (("baseline", args.baseline, baseline),
                                       ("optimized", args.optimized, optimized)):
        snapshot = build / "inputs" / label
        snapshot.mkdir(parents=True, exist_ok=True)
        host_llvm = "host/kernel.ll" if benchmark == "ame" else "nr/kernel.ll"
        if args.host and benchmark == "ame":
            host_manifest = json.loads((directory / "host/manifest.json").read_text())
            for key, relative in (("frontend_manifest_sha256", "frontend.json"),
                                  ("ttir_sha256", "kernel.ttir"), ("linalg_sha256", "kernel.linalg.mlir"),
                                  ("llvm_sha256", host_llvm), ("adapter_sha256", "adapter.c")):
                if host_manifest.get(key) != sha(directory / relative):
                    raise ValueError(f"stale host control: {label}/{relative}")
            provenance["host_control_scope"] = "same Triton TTIR lowered through host linalg; does not execute AME"
        files = {}
        for relative in ("frontend.json", "adapter.c", "kernel.ttir", "kernel.linalg.mlir",
                         "nr/kernel.o", "nr/adapter.o", "nr/kernel.ll",
                         "nr/kernel.llvm.mlir", "nr/kernel.nr.S", "nr/manifest.json",
                         "host/kernel.ll", "host/manifest.json"):
            original = directory / relative
            if original.is_file():
                destination = snapshot / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, destination)
                files[relative] = sha(destination)
        wrapper = "_mlir_ciface_kernel_" + manifest["name"]
        raw = manifest["symbol"]
        new_wrapper, new_raw = f"qwen_{benchmark}_" + label, f"qwen_{benchmark}_raw_" + label
        pair = []
        for kind in ("kernel", "adapter"):
            destination = build / f"{label}.{kind}.o"
            if args.host:
                source = snapshot / (host_llvm if kind == "kernel" else "adapter.c")
                temporary = build / f"{label}.{kind}.original.o"
                run([clang, "-O2", "-fno-vectorize", "-fno-slp-vectorize", "-ffp-contract=off",
                     "-DHOST_TEST", "-I", QWEN, "-c", source, "-o", temporary],
                    build / f"{label}.{kind}.compile.log")
                source_object = temporary
            else:
                source_object = snapshot / f"nr/{kind}.o"
            run([objcopy, "--redefine-sym", f"{wrapper}={new_wrapper}",
                 "--redefine-sym", f"{raw}={new_raw}", source_object, destination])
            symbols = run([nm, destination])
            (build / f"{label}.{kind}.symbols.txt").write_text(symbols)
            definition = new_raw if kind == "kernel" else new_wrapper
            if not re.search(r"\bT\s+" + re.escape(definition) + r"$", symbols, re.M):
                raise ValueError(f"renamed entry not defined in {destination}: {definition}")
            objects.append(destination)
            pair.append({"object": destination.name, "sha256": sha(destination)})
        provenance["variants"][label] = {
            "source": str(directory.resolve()), "manifest": manifest, "inputs_sha256": files,
            "renamed_symbols": {wrapper: new_wrapper, raw: new_raw}, "objects": pair}
    if args.host:
        image = build / "check"
        run([clang, "-O2", "-fno-vectorize", "-fno-slp-vectorize", "-ffp-contract=off",
             "-DHOST_TEST", "-I", QWEN, *definitions, harness,
             *objects, QWEN / "support.c", QWEN / "tools/host_main.c", "-lm", "-o", image],
            build / "link.log")
        run([image], build / "output.log")
    else:
        linker_version = run([args.linker, "--version"]).strip()
        match = re.search(r"LLD (\d+)", linker_version)
        if not match or int(match.group(1)) < 20:
            raise ValueError(f"LLD >=20 required, found: {linker_version}")
        provenance["linker"] = {"path": str(args.linker.absolute()),
                                "version": linker_version, "sha256": sha(args.linker)}
        run(["make", "--no-print-directory", "-f", HERE / "dequant_ab.mk",
             f"BUILD={build}", f"BENCHMARK={benchmark}", f"HARNESS_DEFINES={' '.join(definitions)}",
             f"RISCV_CC={clang}", f"RISCV_LD={args.linker} -m elf64lriscv",
             f"RISCV_OBJCOPY={objcopy}", f"RISCV_OBJDUMP={objdump}",
             f"PYTHON={os.sys.executable}"], build / "build.log")
        image = build / f"{stem}.bin"
    provenance["image"] = {"path": image.name, "sha256": sha(image)}
    provenance["clang"] = {"path": clang, "version": run([clang, "--version"]).strip()}
    (build / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(image)


if __name__ == "__main__":
    main()
