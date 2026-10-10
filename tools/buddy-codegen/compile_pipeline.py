#!/usr/bin/env python3
# ruff: noqa: E501
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
# Subgraph, whole-graph, and partitioned forward dispatcher passes are selected
# here. Partitioned forward dispatchers contain calls and memref view plumbing,
# so they use a lighter lowering pipeline than compute-heavy subgraphs.
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
import math
import os
import re
import subprocess
import sys
import time
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
    "-finalize-memref-to-llvm",  # see lower_to_llvm()
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


# The buffer deallocation passes, left out with the arena.
BUFFER_DEALLOCATION = [
    "-ownership-based-buffer-deallocation",
    "-canonicalize",
    "-buffer-deallocation-simplification",
    "-bufferization-lower-deallocations",
]


# The matrix-engine code of the SpacemiT K3 A100 cores (pipeline
# "kernels_ime") is written for exactly this VLEN: vscale 16, at which
# vector<[8]xi8> is one 8 x 16 int8 tile of smt.vmadot.
A100_VLEN = 1024


def a100_llc_args(llc_base_args: list[str]) -> list[str]:
    """llc options of the A100 code, on top of the build's: the IME
    extension, the exact VLEN (whatever BUDDY_RISCV_VLEN gives the other
    kernels; an option llc accepts once is not repeated) and the A100
    scheduling model, top down (it issues the weight load of an IME step
    first and unpacks it while the activations load)."""
    vmax = f"-riscv-v-vector-bits-max={A100_VLEN}"
    given = [
        a for a in llc_base_args if a.startswith("-riscv-v-vector-bits-max=")
    ]
    if any(a != vmax for a in given):
        raise ValueError(
            f"the A100 IME kernels need VLEN {A100_VLEN}, the build gives "
            f"{given[-1]}"
        )
    return [
        f"-mattr=+xsmtvdotii,+zvl{A100_VLEN}b",
        "-mcpu=spacemit-a100",
        "-misched-prera-direction=topdown",
        *([] if given else [vmax]),
    ]


def lower_to_llvm(arena: bool) -> list[str]:
    """LOWER_TO_LLVM. With the arena (gen_config.derive_memory_options),
    allocations and frees call _mlir_memref_to_llvm_alloc / _aligned_alloc /
    _free, which runtime/arena/BuddyArena.c implements, instead of malloc /
    aligned_alloc / free."""
    if not arena:
        return list(LOWER_TO_LLVM)
    return [
        (
            "-finalize-memref-to-llvm=use-generic-functions=true"
            if p == "-finalize-memref-to-llvm"
            else p
        )
        for p in LOWER_TO_LLVM
    ]


def build_stages(
    pipeline_type: str,
    num_threads: int,
    llc_attrs: str,
    variant: str = "f32",
    tiered: bool = False,
    decode_pack: dict | None = None,
    arena: bool = False,
    chunked_prefill: bool = False,
    batch_vector_size: int | None = None,
):
    """
    Build the list of (tool_name, [args]) stages for a given pipeline type.

    Pipeline types mirror the three CMake macros:
      - "standard":        forward_prefill / forward_decode
      - "forward":         partitioned forward dispatcher wrappers
      - "subgraph":        subgraph_prefill
      - "subgraph_decode": subgraph_decode
    """
    stages = []
    llc_base_args = llc_attrs.split()
    if variant.startswith("w") and not any(
        arg.startswith(("-code-model", "--code-model")) for arg in llc_base_args
    ):
        llc_base_args.append("-code-model=large")

    if pipeline_type in ("kernels", "kernels_a100", "kernels_ime"):
        # Generated kernels (graph/transform/k3_w4.py): scf / vector /
        # memref, scf.parallel for the threads. "kernels_ime": the prefill
        # tiles on the matrix engine of the SpacemiT K3 A100 cores (IME
        # dialect), for the A100 only (a100_llc_args). "kernels_a100": the
        # other kernels when the model runs on the A100 cores only, with
        # the same llc options: scheduled for the in-order A100, which
        # issues independent work in the latency of a reduction (decode
        # attention at position 900: 332 -> 284 us); the same results.
        ime = pipeline_type == "kernels_ime"
        a100 = pipeline_type != "kernels"
        lower = lower_to_llvm(arena)
        i = lower.index("-convert-vector-to-llvm") + 1
        stages.append(
            (
                "buddy-opt",
                (["-lower-ime=target=k3"] if ime else [])
                + [
                    f"-convert-scf-to-openmp=num-threads={num_threads}",
                    "-expand-strided-metadata",
                    "-convert-vector-to-scf",
                    "-lower-affine",
                    "-expand-strided-metadata",
                ]
                + lower[:i]
                + ["-convert-ub-to-llvm"]
                + lower[i:],
            )
        )
        # buddy-translate also translates the IME intrinsics.
        if ime:
            stages.append(("buddy-translate", ["--buddy-to-llvmir"]))
        else:
            stages.append(("mlir-translate", ["-mlir-to-llvmir"]))
        stages.append(("llvm-as", []))
        stages.append(
            (
                "llc",
                llc_base_args
                + (a100_llc_args(llc_base_args) if a100 else [])
                + ["-filetype=obj", "-relocation-model=pic", "-O3"],
            )
        )
        return stages

    if pipeline_type == "forward":
        stages.append(
            (
                "buddy-opt",
                [
                    "-expand-strided-metadata",
                    "-canonicalize",
                    "-cse",
                ]
                + lower_to_llvm(arena),
            )
        )
        stages.append(("mlir-translate", ["-mlir-to-llvmir"]))
        stages.append(("llvm-as", []))
        stages.append(
            (
                "llc",
                llc_base_args
                + [
                    "-filetype=obj",
                    "-relocation-model=pic",
                    "-O3",
                ],
            )
        )
        return stages

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
            # With the arena, nothing is freed before the session resets it.
            *([] if arena else BUFFER_DEALLOCATION),
            "-convert-bufferization-to-memref",
            "-cse",
            "-canonicalize",
            "-optimize-allocation-liveness",
        ]
    )

    # The KV caches are function arguments that the graph updates: write them
    # in place instead of copying each into a new buffer first. Decode, and
    # prefill in chunks (gen_config.derive_prefill_chunk), which has the
    # decode ABI; the session handles results that alias its inputs.
    if pipeline_type == "subgraph_decode" or (
        pipeline_type == "subgraph" and chunked_prefill
    ):
        opts.append("-eliminate-memref-copy")
    if pipeline_type == "subgraph_decode":
        opts.extend(
            [
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
            vector_size = (
                decode_pack["vector_size"]
                if decode_pack and decode_pack.get("enabled")
                else (128 if tiered else 32)
            )
            if decode_pack and decode_pack.get("enabled"):
                # No packed-shapes: pack_decode_matmul_weights packed *every*
                # matmul weight in the decode graph -- and refuses to run at all
                # if it cannot -- so there is no list of exceptions to keep in
                # step, and hence no way for one to drift. A drifted list is
                # silent corruption: the plain kernel would read panel-packed
                # bytes as row-major and answer fluently and wrongly.
                opts.append(
                    "-matmul-vectorization-decode-packed="
                    f"vector-size={vector_size}"
                )
            opts.append(
                f"-matmul-vectorization-decode=vector-size={vector_size}"
            )
        opts.extend(
            [
                "-batch-matmul-vectorization-decode="
                f"vector-size={batch_vector_size or (32 if tiered else 128)}",
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
                # -affine-parallelize makes every parallel loop of a nest an
                # scf.parallel dimension, the unit batch dimension too, and
                # -convert-scf-to-openmp forks the threads over the outermost
                # scf.parallel only: with one iteration, one thread runs the
                # whole nest. -canonicalize drops single-iteration dimensions.
                "-canonicalize",
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
                "-canonicalize",  # as for "subgraph" above
                f"-convert-scf-to-openmp=num-threads={num_threads}",
            ]
        )
        if tiered:
            opts.append("-cse")

    opts.extend(lower_to_llvm(arena))
    stages.append(("buddy-opt", opts))

    # ── Stage 4–6: LLVM backend ─────────────────────────────────────────────
    stages.append(("mlir-translate", ["-mlir-to-llvmir"]))
    stages.append(("llvm-as", []))

    llc_args = llc_base_args + [
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
    if name == "buddy-translate":
        return os.path.join(os.path.dirname(buddy_opt), name)
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
    "k3_kernels": ("k3_kernels.mlir", "k3_kernels.o"),
    "k3_kernels_ime": ("k3_kernels_ime.mlir", "k3_kernels_ime.o"),
    "subgraph_decode": ("subgraph0_decode.mlir", "subgraph_decode.o"),
}


def is_tiered_kv_cache(config: dict) -> bool:
    return bool(config.get("tiered_kv_cache", {}).get("enabled", False))


def tiered_cache_sizes(config: dict) -> list[int]:
    return [
        int(x) for x in config.get("tiered_kv_cache", {}).get("cache_sizes", [])
    ]


def decode_batch_vector_size(config: dict) -> int:
    """Unmasked attention vectors must fit both the cache and head dimensions."""
    if not is_tiered_kv_cache(config):
        return 128
    sizes = tiered_cache_sizes(config)
    return math.gcd(128, config["shape"]["hidden_size"], *sizes)


def compile_entries(config: dict) -> list[tuple[str, str, str]]:
    """Return (pipeline_key, input_mlir_name, output_obj_name) entries."""
    pipelines = config["compilation"]["pipelines"]
    if is_tiered_kv_cache(config):
        entries = []
        for cache_size in tiered_cache_sizes(config):
            for key in pipelines:
                if key == "forward_prefill":
                    entries.append(
                        (
                            key,
                            f"forward_prefill_{cache_size}.mlir",
                            f"forward_prefill_{cache_size}.o",
                        )
                    )
                elif key == "subgraph_prefill":
                    entries.append(
                        (
                            key,
                            f"subgraph0_prefill_{cache_size}.mlir",
                            f"subgraph_prefill_{cache_size}.o",
                        )
                    )
                elif key == "forward_decode":
                    entries.append(
                        (
                            key,
                            f"forward_decode_{cache_size}.mlir",
                            f"forward_decode_{cache_size}.o",
                        )
                    )
                elif key == "subgraph_decode":
                    entries.append(
                        (
                            key,
                            f"subgraph0_decode_{cache_size}.mlir",
                            f"subgraph_decode_{cache_size}.o",
                        )
                    )
                else:
                    raise KeyError(
                        f"Unknown pipeline key for tiered build: {key}"
                    )
        return entries

    variant = config.get("variant", "")
    suffix = f"-{variant}" if variant not in ("f32", "") else ""
    entries = []
    for key in pipelines:
        base_mlir, base_obj = MLIR_FILE_MAP[key]
        mlir_name = base_mlir.replace(".mlir", f"{suffix}.mlir")
        entries.append((key, mlir_name, base_obj))
    return entries


def _compile_one(task: dict) -> str:
    """Compile a single MLIR file. Returns a status message."""
    name = task["name"]
    started = time.perf_counter()
    input_file = task["input"]
    stages = build_stages(
        task["pipeline_type"],
        task["num_threads"],
        task["llc_attrs"],
        task.get("variant", "f32"),
        task.get("tiered", False),
        task.get("decode_pack"),
        task.get("arena", False),
        task.get("chunked_prefill", False),
        task.get("batch_vector_size"),
    )
    run_pipeline(
        stages,
        input_file,
        task["output"],
        task["buddy_opt"],
        task["llvm_dir"],
    )
    elapsed = time.perf_counter() - started
    return f"  {name}: {os.path.basename(task['output'])} ({elapsed:.2f}s)"


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
    decode_pack = config.get("decode_pack", {"enabled": False})

    os.makedirs(output_dir, exist_ok=True)

    tasks = []
    for key, mlir_name, obj_name in compile_entries(config):
        pipeline_type = pipelines[key]
        input_path = os.path.join(mlir_dir, mlir_name)
        output_path = os.path.join(output_dir, obj_name)

        if not os.path.exists(input_path) and not is_tiered_kv_cache(config):
            # Fall back to name without suffix
            base_mlir, _ = MLIR_FILE_MAP[key]
            input_path = os.path.join(mlir_dir, base_mlir)

        tasks.append(
            {
                "name": key,
                "pipeline_type": pipeline_type,
                "num_threads": num_threads,
                "llc_attrs": llc_attrs,
                "variant": variant,
                "tiered": is_tiered_kv_cache(config),
                "decode_pack": decode_pack,
                "batch_vector_size": decode_batch_vector_size(config),
                "arena": bool(config.get("arena", False)),
                "chunked_prefill": bool(config.get("prefill_chunk")),
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


def partitioned_compile_entries(
    mlir_dir: str,
    prefill_only: bool = False,
    full_mlir_dir: str | None = None,
    tiered: bool = False,
) -> list[tuple[str, str, str, str]]:
    """Discover per-layer MLIR files emitted by import_model.py."""
    if tiered:
        patterns = [
            (
                re.compile(r"^forward_prefill_(\d+)\.mlir$"),
                "forward_prefill",
                "forward",
            ),
            (
                re.compile(r"^forward_decode_(\d+)\.mlir$"),
                "forward_decode",
                "forward",
            ),
            (
                re.compile(r"^subgraph0_prefill_(\d+)_(\d+)\.mlir$"),
                "subgraph_prefill",
                "subgraph",
            ),
            (
                re.compile(r"^subgraph0_decode_(\d+)_(\d+)\.mlir$"),
                "subgraph_decode",
                "subgraph_decode",
            ),
            (
                re.compile(r"^subgraph0_decode_(\d+)\.mlir$"),
                "subgraph_decode",
                "subgraph_decode",
            ),
        ]
    else:
        patterns = [
            (
                re.compile(r"^forward_prefill\.mlir$"),
                "forward_prefill",
                "forward",
            ),
            (
                re.compile(r"^forward_decode\.mlir$"),
                "forward_decode",
                "forward",
            ),
            (
                re.compile(r"^subgraph0_prefill(\d+)\.mlir$"),
                "subgraph_prefill",
                "subgraph",
            ),
            (
                re.compile(r"^subgraph0_decode(\d+)\.mlir$"),
                "subgraph_decode",
                "subgraph_decode",
            ),
            (
                re.compile(r"^forward_prefill(\d+)\.mlir$"),
                "forward_prefill",
                "standard",
            ),
            (
                re.compile(r"^forward_decode(\d+)\.mlir$"),
                "forward_decode",
                "standard",
            ),
        ]

    entries = []
    for filename in os.listdir(mlir_dir):
        is_decode_file = (
            filename == "forward_decode.mlir"
            or re.match(r"^forward_decode\d+\.mlir$", filename)
            or re.match(r"^forward_decode_\d+\.mlir$", filename)
            or re.match(r"^subgraph0_decode\d+\.mlir$", filename)
            or re.match(r"^subgraph0_decode_\d+\.mlir$", filename)
            or re.match(r"^subgraph0_decode_\d+_\d+\.mlir$", filename)
        )
        if prefill_only and is_decode_file:
            continue
        for regex, key_prefix, pipeline_type in patterns:
            match = regex.match(filename)
            if not match:
                continue
            if len(match.groups()) >= 2:
                index = int(match.group(1)) * 10000 + int(match.group(2))
            else:
                index = int(match.group(1)) if match.groups() else -1
            stem = filename.removesuffix(".mlir")
            entries.append(
                (
                    f"{key_prefix}_{index}",
                    filename,
                    f"{stem}.o",
                    pipeline_type,
                    index,
                )
            )
            break

    if prefill_only:
        if not full_mlir_dir:
            raise RuntimeError(
                "--partitioned-prefill-only requires --full-mlir-dir"
            )
        entries.extend(
            [
                (
                    "forward_decode_-1",
                    os.path.join(full_mlir_dir, "forward_decode.mlir"),
                    "forward_decode.o",
                    "standard",
                    -1,
                ),
                (
                    "subgraph_decode_-1",
                    os.path.join(full_mlir_dir, "subgraph0_decode.mlir"),
                    "subgraph_decode.o",
                    "subgraph_decode",
                    -1,
                ),
            ]
        )

    entries.sort(key=lambda item: (item[3], item[4], item[0]))
    return [
        (name, mlir_name, obj_name, pipeline)
        for name, mlir_name, obj_name, pipeline, _ in entries
    ]


def compile_partitioned(
    config: dict,
    mlir_dir: str,
    output_dir: str,
    buddy_opt: str,
    llvm_dir: str,
    llc_attrs: str,
    jobs: int = 1,
    prefill_only: bool = False,
    full_mlir_dir: str | None = None,
) -> None:
    """Compile all per-layer MLIR files from a layer_partitioned directory."""
    entries = partitioned_compile_entries(
        mlir_dir,
        prefill_only=prefill_only,
        full_mlir_dir=full_mlir_dir,
        tiered=is_tiered_kv_cache(config),
    )
    if not entries:
        raise RuntimeError(f"No partitioned MLIR files found in {mlir_dir}")

    num_threads = config["compilation"]["num_threads"]
    variant = config.get("variant", "")
    decode_pack = config.get("decode_pack", {"enabled": False})
    os.makedirs(output_dir, exist_ok=True)

    tasks = []
    for key, mlir_name, obj_name, pipeline_type in entries:
        tasks.append(
            {
                "name": key,
                "pipeline_type": pipeline_type,
                "num_threads": num_threads,
                "llc_attrs": llc_attrs,
                "variant": variant,
                "decode_pack": decode_pack,
                "batch_vector_size": decode_batch_vector_size(config),
                "tiered": is_tiered_kv_cache(config),
                "arena": bool(config.get("arena", False)),
                "chunked_prefill": bool(config.get("prefill_chunk")),
                "input": mlir_name
                if os.path.isabs(mlir_name)
                else os.path.join(mlir_dir, mlir_name),
                "output": os.path.join(output_dir, obj_name),
                "buddy_opt": buddy_opt,
                "llvm_dir": llvm_dir,
            }
        )

    started = time.perf_counter()
    print(
        f"[compile] Compiling {len(tasks)} partitioned MLIR files "
        f"(jobs={jobs})...",
        file=sys.stderr,
    )

    if jobs <= 1:
        for task in tasks:
            print(_compile_one(task), file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_compile_one, t): t["name"] for t in tasks}
            for fut in as_completed(futures):
                print(fut.result(), file=sys.stderr)

    elapsed = time.perf_counter() - started
    print(f"[compile] All done in {elapsed:.2f}s.", file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────────
# Link .o files → shared library
# ──────────────────────────────────────────────────────────────────────────────


def link_shared_lib(
    obj_files: list[str],
    output_so: str,
    cxx: str = "c++",
    llvm_lib_dir: str = "",
    openmp_runtime_lib: str = "",
) -> None:
    """Link object files into a shared library."""
    lib_dirs = []
    if llvm_lib_dir:
        llvm_lib_dir = os.path.normpath(llvm_lib_dir)
        lib_dirs.append(llvm_lib_dir)

    if openmp_runtime_lib:
        openmp_runtime_lib = os.path.normpath(openmp_runtime_lib)
        openmp_runtime_dir = os.path.dirname(openmp_runtime_lib)
        if openmp_runtime_dir:
            lib_dirs.append(openmp_runtime_dir)

    deduped_lib_dirs = []
    seen_lib_dirs = set()
    for lib_dir in lib_dirs:
        if lib_dir in seen_lib_dirs:
            continue
        seen_lib_dirs.add(lib_dir)
        deduped_lib_dirs.append(lib_dir)

    cmd = [
        cxx,
        "-shared",
        "-fPIC",
        "-o",
        output_so,
    ] + obj_files

    if sys.platform == "darwin":
        cmd.insert(3, f"-Wl,-install_name,@rpath/{os.path.basename(output_so)}")
    else:
        cmd.insert(3, f"-Wl,-soname,{os.path.basename(output_so)}")
        cmd.insert(4, "-Wl,--allow-multiple-definition")

    for lib_dir in deduped_lib_dirs:
        cmd.extend(
            [
                f"-L{lib_dir}",
                f"-Wl,-rpath,{lib_dir}",
            ]
        )
    omp_link_arg = openmp_runtime_lib if openmp_runtime_lib else "-lomp"
    cmd.extend([omp_link_arg, "-lmlir_c_runner_utils", "-lm"])

    print(f"[link] {os.path.basename(output_so)}", file=sys.stderr)
    subprocess.check_call(cmd)


def partitioned_runtime_objects(
    output_dir: str,
    config: dict,
    prefill_only: bool = False,
) -> list[str]:
    """Return the object files needed by the runtime partitioned .so."""
    if is_tiered_kv_cache(config):
        patterns = [
            r"^subgraph0_prefill_\d+_\d+\.o$",
            r"^subgraph0_decode_\d+_\d+\.o$",
            r"^subgraph0_decode_\d+\.o$",
            r"^forward_prefill_\d+\.o$",
            r"^forward_decode_\d+\.o$",
        ]
        if prefill_only:
            patterns = [
                r"^subgraph0_prefill_\d+_\d+\.o$",
                r"^forward_prefill_\d+\.o$",
                r"^subgraph_decode\.o$",
                r"^forward_decode\.o$",
            ]
        obj_files = [
            os.path.join(output_dir, name)
            for name in os.listdir(output_dir)
            if any(re.match(pattern, name) for pattern in patterns)
        ]
        obj_files.sort()
        return obj_files

    patterns = [
        r"^subgraph0_prefill\d+\.o$",
        r"^subgraph0_decode\d+\.o$",
    ]
    if prefill_only:
        patterns = [
            r"^subgraph0_prefill\d+\.o$",
            r"^subgraph_decode\.o$",
        ]

    obj_files = []
    for name in os.listdir(output_dir):
        if any(re.match(pattern, name) for pattern in patterns):
            obj_files.append(os.path.join(output_dir, name))
    obj_files.sort()

    for name in ("forward_decode.o", "forward_prefill.o"):
        path = os.path.join(output_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing partitioned runtime object: {path}"
            )
        obj_files.append(path)

    return obj_files


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
    mode.add_argument(
        "--compile-partitioned",
        action="store_true",
        help="Compile per-layer MLIR files emitted under layer_partitioned",
    )
    mode.add_argument("--input", help="Single MLIR file to compile")

    parser.add_argument("--output", help="Output .o file (single-file mode)")
    parser.add_argument(
        "--pipeline",
        choices=["standard", "forward", "subgraph", "subgraph_decode"],
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
        help="Also link .o → .so after compilation",
    )
    parser.add_argument("--cxx", default="c++", help="C++ compiler for linking")
    parser.add_argument(
        "--llvm-lib-dir", default="", help="LLVM library directory"
    )
    parser.add_argument(
        "--openmp-runtime-lib",
        default="",
        help="Full path to the OpenMP runtime library used for linking",
    )
    parser.add_argument(
        "--output-so",
        help="Output shared library path when --link is used",
    )
    parser.add_argument(
        "--partitioned-prefill-only",
        action="store_true",
        help=(
            "With --compile-partitioned, compile partitioned prefill but keep "
            "decode from --full-mlir-dir."
        ),
    )
    parser.add_argument(
        "--full-mlir-dir",
        help="Directory containing whole-graph MLIR files for mixed partitioning",
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
            output_so = args.output_so or os.path.join(
                args.output_dir, config["compilation"]["so_name"]
            )

            obj_files = []
            for _key, _mlir_name, obj_name in compile_entries(config):
                obj_files.append(os.path.join(args.output_dir, obj_name))

            link_shared_lib(
                obj_files=obj_files,
                output_so=output_so,
                cxx=args.cxx,
                llvm_lib_dir=args.llvm_lib_dir,
                openmp_runtime_lib=args.openmp_runtime_lib,
            )
    elif args.compile_partitioned:
        if not args.mlir_dir or not args.output_dir:
            parser.error(
                "--compile-partitioned requires --mlir-dir and --output-dir"
            )

        compile_partitioned(
            config=config,
            mlir_dir=args.mlir_dir,
            output_dir=args.output_dir,
            buddy_opt=args.buddy_opt,
            llvm_dir=args.llvm_tools_dir,
            llc_attrs=args.llc_attrs,
            jobs=args.jobs,
            prefill_only=args.partitioned_prefill_only,
            full_mlir_dir=args.full_mlir_dir,
        )
        if args.link:
            output_so = args.output_so or os.path.join(
                args.output_dir, config["compilation"]["so_name"]
            )
            link_shared_lib(
                obj_files=partitioned_runtime_objects(
                    args.output_dir,
                    config,
                    prefill_only=args.partitioned_prefill_only,
                ),
                output_so=output_so,
                cxx=args.cxx,
                llvm_lib_dir=args.llvm_lib_dir,
                openmp_runtime_lib=args.openmp_runtime_lib,
            )
    else:
        if not args.output or not args.pipeline:
            parser.error("Single-file mode requires --output and --pipeline")

        num_threads = config["compilation"]["num_threads"]
        variant = config.get("variant", "f32")
        stages = build_stages(
            args.pipeline,
            num_threads,
            args.llc_attrs,
            variant,
            is_tiered_kv_cache(config),
            config.get("decode_pack"),
            bool(config.get("arena", False)),
            bool(config.get("prefill_chunk")),
            decode_batch_vector_size(config),
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
