#!/usr/bin/env python3
# ===- compile_pipeline.py - MLIR → .o compilation pipeline ---------------===//
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===----------------------------------------------------------------------===//
#
# Replaces the CMake macros dsr1_mlir_to_obj / dsr1_subgraph_to_obj /
# dsr1_subgraph_decode_to_obj with a single config-driven Python script.
#
# Single file:
#   python compile_pipeline.py --config config.json \
#       --input forward_prefill.mlir --output forward_prefill.o \
#       --pipeline standard \
#       --buddy-opt /path/to/buddy-opt --llvm-tools-dir /path/to/llvm/bin
#
# All files at once:
#   python compile_pipeline.py --config config.json \
#       --compile-all --mlir-dir ./mlir --output-dir ./obj \
#       --buddy-opt /path/to/buddy-opt --llvm-tools-dir /path/to/llvm/bin
#
# ===----------------------------------------------------------------------===//

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# ──────────────────────────────────────────────────────────────────────────────
# Pass pipeline definitions
# ──────────────────────────────────────────────────────────────────────────────

TOSA_PIPELINE = (
    "builtin.module(func.func(tosa-to-linalg-named),"
    "func.func(tosa-to-linalg),"
    "func.func(tosa-to-tensor),"
    "func.func(tosa-to-arith))"
)

LOWER_TO_LLVM = [
    "-memref-expand",
    "-arith-expand",
    "-convert-vector-to-llvm",
    "-convert-arith-to-llvm",
    "-finalize-memref-to-llvm",
    "-convert-scf-to-cf",
    "-convert-cf-to-llvm",
    "-llvm-request-c-wrappers",
    "-convert-openmp-to-llvm",
    "-convert-arith-to-llvm",
    "-convert-math-to-llvm",
    "-convert-math-to-libm",
    "-convert-func-to-llvm",
    "-reconcile-unrealized-casts",
]


def build_stages(
    pipeline_type: str, num_threads: int, llc_attrs: str, variant: str = "f32"
):
    """
    Build the list of (tool_name, [args]) stages for a given pipeline type.

    Pipeline types mirror the three CMake macros:
      - "standard":        forward_prefill / forward_decode
      - "subgraph":        subgraph_prefill
      - "subgraph_decode": subgraph_decode
    """
    is_quantized = variant.startswith("w")
    stages = []

    # ── Stage 1: buddy-opt (initial simplification) ──────────────────────────
    init_opts = ["-simplify-tosa-reshape"]
    if pipeline_type == "subgraph_decode":
        init_opts.append("-simplify-tosa-matmul-scalar")
    stages.append(("buddy-opt", init_opts))

    # ── Stage 2: mlir-opt (TOSA lowering) ────────────────────────────────────
    stages.append(("mlir-opt", [f"-pass-pipeline={TOSA_PIPELINE}"]))

    # ── Stage 3: buddy-opt (bufferize → vectorize → lower) ──────────────────
    opts = [
        "-eliminate-empty-tensors",
        "-empty-tensor-to-alloc-tensor",
    ]
    if pipeline_type in ("subgraph", "subgraph_decode"):
        opts.append("-convert-elementwise-to-linalg")

    opts.extend(
        [
            "-one-shot-bufferize=bufferize-function-boundaries",
            "-expand-strided-metadata",
            "-ownership-based-buffer-deallocation",
            "-canonicalize",
            "-buffer-deallocation-simplification",
            "-bufferization-lower-deallocations",
            "-cse",
            "-canonicalize",
            "-optimize-allocation-liveness",
        ]
    )

    if pipeline_type == "subgraph_decode":
        opts.extend(
            [
                "-eliminate-memref-copy",
                "-assume-tight-memref-layout",
                "-staticize-memref-layout",
            ]
        )
        if variant in ("w8a32", "w8a16"):
            opts.append("-dequant-matmul-vectorization-decode=vector-size=32")
            opts.append("-matmul-vectorization-decode=vector-size=32")
        elif variant == "w4a16":
            opts.append(
                "-int4-dequant-matmul-vectorization-decode=vector-size=32"
            )
            opts.append("-matmul-vectorization-decode=vector-size=32")
        elif variant != "w8a8":
            opts.append("-matmul-vectorization-decode=vector-size=32")
        opts.extend(
            [
                "-batch-matmul-vectorization-decode=vector-size=128",
                "-batchmatmul-transpose-b-vectorization=vector-size=16",
                "-convert-linalg-to-affine-loops",
                "-convert-vector-to-scf",
                "-lower-affine",
                f"-convert-scf-to-openmp=num-threads={num_threads}",
                "-cse",
            ]
        )
    elif pipeline_type == "subgraph":
        opts.extend(
            [
                "-matmul-vectorization-blis",
                "-batchmatmul-optimize",
                "-batchmatmul-transpose-b-vectorization",
                "-convert-linalg-to-affine-loops",
                "-affine-parallelize",
                "-convert-vector-to-scf",
                "-lower-affine",
                f"-convert-scf-to-openmp=num-threads={num_threads}",
                "-cse",
            ]
        )
    else:  # standard
        opts.extend(
            [
                "-matmul-vectorization-blis",
                "-batchmatmul-optimize",
                "-convert-linalg-to-affine-loops",
                "-affine-parallelize",
                "-convert-vector-to-scf",
                "-lower-affine",
                f"-convert-scf-to-openmp=num-threads={num_threads}",
            ]
        )

    opts.extend(LOWER_TO_LLVM)
    stages.append(("buddy-opt", opts))

    # ── Stage 4–6: LLVM backend ─────────────────────────────────────────────
    stages.append(("mlir-translate", ["-mlir-to-llvmir"]))
    stages.append(("llvm-as", []))

    llc_args = llc_attrs.split() + [
        "-filetype=obj",
        "-relocation-model=pic",
        "-O3",
    ]
    stages.append(("llc", llc_args))

    return stages


# ──────────────────────────────────────────────────────────────────────────────
# Execution
# ──────────────────────────────────────────────────────────────────────────────


def _resolve_tool(name: str, buddy_opt: str, llvm_dir: str) -> str:
    if name == "buddy-opt":
        return buddy_opt
    return os.path.join(llvm_dir, name)


def run_pipeline(
    stages: list,
    input_file: str,
    output_file: str,
    buddy_opt: str,
    llvm_dir: str,
) -> None:
    """Execute a multi-stage piped compilation pipeline."""
    procs = []

    for i, (tool_name, args) in enumerate(stages):
        tool = _resolve_tool(tool_name, buddy_opt, llvm_dir)
        is_first = i == 0
        is_last = i == len(stages) - 1

        cmd = [tool]
        if is_first:
            cmd.append(input_file)
        cmd.extend(args)
        if is_last:
            cmd.extend(["-o", output_file])

        stdin = None if is_first else procs[-1].stdout
        stdout = None if is_last else subprocess.PIPE

        p = subprocess.Popen(
            cmd, stdin=stdin, stdout=stdout, stderr=subprocess.PIPE
        )
        procs.append(p)

        if not is_first:
            procs[-2].stdout.close()

    # Collect results
    errors = []
    for i, p in enumerate(procs):
        p.wait()
        if p.returncode != 0:
            stderr = (
                p.stderr.read().decode(errors="replace") if p.stderr else ""
            )
            errors.append(
                f"Stage {i} ({stages[i][0]}) exit {p.returncode}:\n{stderr[:2000]}"
            )

    if errors:
        raise RuntimeError(
            f"Pipeline failed for {input_file}:\n" + "\n".join(errors)
        )


# ──────────────────────────────────────────────────────────────────────────────
# Compile-all: process all 4 MLIR files according to config
# ──────────────────────────────────────────────────────────────────────────────

# Mapping from pipeline config key → (input MLIR basename, output .o basename)
MLIR_FILE_MAP = {
    "forward_prefill": ("forward_prefill.mlir", "forward_prefill.o"),
    "subgraph_prefill": ("subgraph0_prefill.mlir", "subgraph_prefill.o"),
    "forward_decode": ("forward_decode.mlir", "forward_decode.o"),
    "subgraph_decode": ("subgraph0_decode.mlir", "subgraph_decode.o"),
}


def _compile_one(task: dict) -> str:
    """Compile a single MLIR file. Returns a status message."""
    name = task["name"]
    stages = build_stages(
        task["pipeline_type"],
        task["num_threads"],
        task["llc_attrs"],
        task.get("variant", "f32"),
    )
    run_pipeline(
        stages,
        task["input"],
        task["output"],
        task["buddy_opt"],
        task["llvm_dir"],
    )
    return f"  {name}: {os.path.basename(task['output'])}"


def compile_all(
    config: dict,
    mlir_dir: str,
    output_dir: str,
    buddy_opt: str,
    llvm_dir: str,
    llc_attrs: str,
    jobs: int = 1,
) -> None:
    """Compile all MLIR files defined in config['compilation']['pipelines']."""
    pipelines = config["compilation"]["pipelines"]
    num_threads = config["compilation"]["num_threads"]
    variant = config.get("variant", "")

    os.makedirs(output_dir, exist_ok=True)

    # Build suffix for variant-specific MLIR files (e.g., "-w8a16")
    suffix = f"-{variant}" if variant not in ("f32", "") else ""

    tasks = []
    for key, pipeline_type in pipelines.items():
        base_mlir, base_obj = MLIR_FILE_MAP[key]
        # MLIR inputs may be variant-suffixed (e.g. forward_prefill-w8a16.mlir).
        # Object files keep stable names (forward_prefill.o): one build dir = one variant.
        mlir_name = base_mlir.replace(".mlir", f"{suffix}.mlir")
        obj_name = base_obj

        input_path = os.path.join(mlir_dir, mlir_name)
        output_path = os.path.join(output_dir, obj_name)

        if not os.path.exists(input_path):
            # Fall back to name without suffix
            input_path = os.path.join(mlir_dir, base_mlir)

        tasks.append(
            {
                "name": key,
                "pipeline_type": pipeline_type,
                "num_threads": num_threads,
                "llc_attrs": llc_attrs,
                "variant": variant,
                "input": input_path,
                "output": output_path,
                "buddy_opt": buddy_opt,
                "llvm_dir": llvm_dir,
            }
        )

    print(
        f"[compile] Compiling {len(tasks)} MLIR files (jobs={jobs})...",
        file=sys.stderr,
    )

    if jobs <= 1:
        for t in tasks:
            msg = _compile_one(t)
            print(msg, file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_compile_one, t): t["name"] for t in tasks}
            for fut in as_completed(futures):
                msg = fut.result()
                print(msg, file=sys.stderr)

    print("[compile] All done.", file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────────
# Link .o files → shared library
# ──────────────────────────────────────────────────────────────────────────────


def link_shared_lib(
    obj_files: list[str],
    output_so: str,
    cxx: str = "c++",
    llvm_lib_dir: str = "",
) -> None:
    """Link object files into a shared library."""
    cmd = [
        cxx,
        "-shared",
        "-fPIC",
        f"-Wl,-soname,{os.path.basename(output_so)}",
        "-Wl,--allow-multiple-definition",
        "-o",
        output_so,
    ] + obj_files

    if llvm_lib_dir:
        cmd.extend(
            [
                f"-L{llvm_lib_dir}",
                f"-Wl,-rpath,{llvm_lib_dir}",
            ]
        )
    cmd.extend(["-lomp", "-lmlir_c_runner_utils", "-lm"])

    print(f"[link] {os.path.basename(output_so)}", file=sys.stderr)
    subprocess.check_call(cmd)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="MLIR → .o compilation pipeline (replaces CMake macros)."
    )
    parser.add_argument(
        "--config", required=True, help="Full model config JSON"
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--compile-all",
        action="store_true",
        help="Compile all 4 MLIR files from config",
    )
    mode.add_argument("--input", help="Single MLIR file to compile")

    parser.add_argument("--output", help="Output .o file (single-file mode)")
    parser.add_argument(
        "--pipeline",
        choices=["standard", "subgraph", "subgraph_decode"],
        help="Pipeline type (single-file mode)",
    )
    parser.add_argument(
        "--mlir-dir", help="Directory with MLIR files (compile-all)"
    )
    parser.add_argument(
        "--output-dir", help="Output directory for .o (compile-all)"
    )
    parser.add_argument(
        "--buddy-opt", required=True, help="Path to buddy-opt binary"
    )
    parser.add_argument(
        "--llvm-tools-dir",
        required=True,
        help="Directory containing mlir-opt, llc, etc.",
    )
    parser.add_argument(
        "--llc-attrs", default="-mcpu=native", help="LLC attributes string"
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="Parallel jobs (compile-all mode)",
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help="Also link .o → .so after compilation (compile-all mode)",
    )
    parser.add_argument("--cxx", default="c++", help="C++ compiler for linking")
    parser.add_argument(
        "--llvm-lib-dir", default="", help="LLVM library directory"
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    if args.compile_all:
        if not args.mlir_dir or not args.output_dir:
            parser.error("--compile-all requires --mlir-dir and --output-dir")

        compile_all(
            config=config,
            mlir_dir=args.mlir_dir,
            output_dir=args.output_dir,
            buddy_opt=args.buddy_opt,
            llvm_dir=args.llvm_tools_dir,
            llc_attrs=args.llc_attrs,
            jobs=args.jobs,
        )

        if args.link:
            so_name = config["compilation"]["so_name"]

            obj_files = []
            for key in config["compilation"]["pipelines"]:
                _, base_obj = MLIR_FILE_MAP[key]
                obj_files.append(os.path.join(args.output_dir, base_obj))

            link_shared_lib(
                obj_files=obj_files,
                output_so=os.path.join(args.output_dir, so_name),
                cxx=args.cxx,
                llvm_lib_dir=args.llvm_lib_dir,
            )
    else:
        if not args.output or not args.pipeline:
            parser.error("Single-file mode requires --output and --pipeline")

        num_threads = config["compilation"]["num_threads"]
        variant = config.get("variant", "f32")
        stages = build_stages(
            args.pipeline, num_threads, args.llc_attrs, variant
        )

        print(
            f"[compile] {os.path.basename(args.input)} → {os.path.basename(args.output)} "
            f"(pipeline={args.pipeline})",
            file=sys.stderr,
        )
        run_pipeline(
            stages, args.input, args.output, args.buddy_opt, args.llvm_tools_dir
        )
        print("[compile] Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
