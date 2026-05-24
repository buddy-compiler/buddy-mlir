#!/usr/bin/env python3
# ===- run_model_bench.py -------------------------------------------------===//
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

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from subprocess import CompletedProcess

try:
    from buddy.utils.terminal import TerminalUI, run_with_status
except ModuleNotFoundError as err:
    raise SystemExit(
        "Cannot import buddy.utils.terminal. From the build directory, export:\n"
        "  export BUDDY_MLIR_BUILD_DIR=$PWD\n"
        "  export LLVM_MLIR_BUILD_DIR=$PWD/../llvm/build\n"
        "  export PYTHONPATH=${BUDDY_MLIR_BUILD_DIR}/python_packages:"
        "${PYTHONPATH}"
    ) from err

ROOT = Path(__file__).resolve().parents[2]

MODEL_BENCH = ROOT / "benchmarks" / "ModelBench"
DEEPSEEK_EXAMPLE = ROOT / "examples" / "BuddyDeepSeekR1"
DEEPSEEK_BENCH = MODEL_BENCH / "DeepSeekR1"
OUT_DIR = Path(
    os.environ.get(
        "MODEL_BENCH_OUT_DIR",
        ROOT / "build" / "benchmarks" / "ModelBench",
    )
)
TOSA_PIPELINE = (
    "builtin.module(func.func(tosa-to-linalg-named),func.func(tosa-to-linalg),"
    "func.func(tosa-to-tensor),func.func(tosa-to-arith))"
)


def default_tool(path: str, env_name: str) -> str:
    return os.environ.get(env_name, str(ROOT / path))


UI = TerminalUI()


def section(title: str) -> None:
    UI.section(title)


def format_seconds(seconds: float) -> str:
    return UI.format_seconds(seconds)


def run(
    cmd: list[str],
    *,
    stdout: Path | None = None,
    label: str | None = None,
    phase: str = "build",
    capture_stdout: bool = False,
    verbose: bool = False,
) -> CompletedProcess[bytes]:
    return run_with_status(
        cmd,
        cwd=ROOT,
        ui=UI,
        stdout=stdout,
        label=label,
        phase=phase,
        capture_stdout=capture_stdout,
        verbose=verbose,
    )


def common_pipeline(openmp_threads: int, kind: str) -> list[str]:
    pipeline = [
        "-eliminate-empty-tensors",
        "-empty-tensor-to-alloc-tensor",
    ]
    if kind in {"subgraph", "decode-subgraph"}:
        pipeline.append("-convert-elementwise-to-linalg")
    pipeline.extend(
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
            "-eliminate-memref-copy",
            "-assume-tight-memref-layout",
            "-staticize-memref-layout",
        ]
    )
    if kind == "decode-subgraph":
        pipeline.extend(
            [
                "-matmul-vectorization-decode=vector-size=32",
                "-batch-matmul-vectorization-decode=vector-size=128",
                "-batchmatmul-transpose-b-vectorization=vector-size=16",
                "-convert-linalg-to-affine-loops",
            ]
        )
    else:
        pipeline.extend(["-matmul-vectorization-blis", "-batchmatmul-optimize"])
        if kind == "subgraph":
            pipeline.append("-batchmatmul-transpose-b-vectorization")
        pipeline.extend(
            ["-convert-linalg-to-affine-loops", "-affine-parallelize"]
        )
    pipeline.extend(
        [
            "-convert-vector-to-scf",
            "-lower-affine",
            f"-convert-scf-to-openmp=num-threads={openmp_threads}",
            "-cse",
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
    )
    return pipeline


def compile_mlir_to_object(
    args: argparse.Namespace, mlir: Path, obj: Path, kind: str
) -> None:
    stage_dir = obj.parent / "stages" / obj.stem
    frontend_mlir = stage_dir / "frontend.mlir"
    tosa_mlir = stage_dir / "tosa.mlir"
    llvm_mlir = stage_dir / "llvm.mlir"
    llvm_ir = stage_dir / "llvm.ll"
    bitcode = stage_dir / "llvm.bc"

    frontend_passes = ["-simplify-tosa-reshape"]
    if kind == "decode-subgraph":
        frontend_passes.append("-simplify-tosa-matmul-scalar")

    run(
        [args.buddy_opt, str(mlir), *frontend_passes],
        stdout=frontend_mlir,
        label=f"{obj.stem}: frontend simplification",
        verbose=args.verbose_commands,
    )
    run(
        [args.mlir_opt, str(frontend_mlir), "-pass-pipeline", TOSA_PIPELINE],
        stdout=tosa_mlir,
        label=f"{obj.stem}: TOSA lowering",
        verbose=args.verbose_commands,
    )
    run(
        [
            args.buddy_opt,
            str(tosa_mlir),
            *common_pipeline(args.openmp_threads, kind),
        ],
        stdout=llvm_mlir,
        label=f"{obj.stem}: bufferize/vectorize/lower",
        verbose=args.verbose_commands,
    )
    run(
        [args.mlir_translate, "-mlir-to-llvmir", str(llvm_mlir)],
        stdout=llvm_ir,
        label=f"{obj.stem}: translate to LLVM IR",
        verbose=args.verbose_commands,
    )
    run(
        [args.llvm_as, str(llvm_ir), "-o", str(bitcode)],
        label=f"{obj.stem}: assemble bitcode",
        verbose=args.verbose_commands,
    )
    run(
        [
            args.llc,
            str(bitcode),
            *args.llc_attrs,
            "-filetype=obj",
            "-relocation-model=pic",
            "-O3",
            "-o",
            str(obj),
        ],
        label=f"{obj.stem}: emit object",
        verbose=args.verbose_commands,
    )


def import_deepseek(args: argparse.Namespace, case_dir: Path) -> None:
    outputs = [
        case_dir / "forward_prefill.mlir",
        case_dir / "subgraph0_prefill.mlir",
        case_dir / "forward_decode.mlir",
        case_dir / "subgraph0_decode.mlir",
    ]
    if args.skip_import and all(path.exists() for path in outputs):
        print("[build] import DeepSeek-R1 F32 MLIR skipped")
        return
    run(
        [
            sys.executable,
            str(DEEPSEEK_EXAMPLE / "import-deepseek-r1.py"),
            "--output-dir",
            str(case_dir),
            "--precision",
            "f32",
        ],
        label="import DeepSeek-R1 F32 MLIR",
        verbose=args.verbose_commands,
    )


def build_deepseek(args: argparse.Namespace, case_dir: Path) -> Path:
    objects_dir = case_dir / "objects"
    objects_dir.mkdir(parents=True, exist_ok=True)
    objects = [
        (
            case_dir / "forward_prefill.mlir",
            objects_dir / "forward_prefill.o",
            "forward",
        ),
        (
            case_dir / "subgraph0_prefill.mlir",
            objects_dir / "subgraph_prefill.o",
            "subgraph",
        ),
        (
            case_dir / "forward_decode.mlir",
            objects_dir / "forward_decode.o",
            "forward",
        ),
        (
            case_dir / "subgraph0_decode.mlir",
            objects_dir / "subgraph_decode.o",
            "decode-subgraph",
        ),
    ]
    if not args.skip_compile:
        for mlir, obj, kind in objects:
            compile_mlir_to_object(args, mlir, obj, kind)

    runner = case_dir / "modelbench-deepseek-r1-run"
    run(
        [
            args.clangxx,
            "-O3",
            "-std=c++17",
            str(DEEPSEEK_BENCH / "deepseek-r1-bench.cpp"),
            *(str(obj) for _, obj, _ in objects),
            "-I",
            str(ROOT / "frontend" / "Interfaces"),
            "-I",
            str(ROOT / "llvm" / "mlir" / "include"),
            "-I",
            str(ROOT / "llvm" / "build" / "tools" / "mlir" / "include"),
            "-L",
            str(args.llvm_lib_dir),
            "-lmlir_c_runner_utils",
            "-lomp",
            "-lm",
            "-Wl,--allow-multiple-definition",
            f"-Wl,-rpath,{args.llvm_lib_dir}",
            "-o",
            str(runner),
        ],
        label="link benchmark runner",
        verbose=args.verbose_commands,
    )
    return runner


def parse_result_csv(output: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    header_idx = next(
        (
            idx
            for idx, line in enumerate(lines)
            if line.startswith("model,precision,prefill_s,decode_s")
        ),
        None,
    )
    if header_idx is None:
        return []
    header = lines[header_idx].split(",")
    rows = []
    for line in lines[header_idx + 1 :]:
        values = line.split(",")
        if len(values) == len(header):
            rows.append(dict(zip(header, values)))
    return rows


def print_result_table(rows: list[dict[str, str]], csv_path: Path) -> None:
    section("ModelBench Result")
    if not rows:
        print("No parseable benchmark rows were produced.")
        return
    print(
        f"{'model':<16} {'precision':<10} {'prefill':>14} "
        f"{'decode(avg)':>14} {'decode iters':>12}"
    )
    print(UI.rule(72))
    for row in rows:
        prefill = format_seconds(float(row["prefill_s"]))
        decode = format_seconds(float(row["decode_s"]))
        decode_iters = row.get("decode_iters", "1")
        print(
            f"{row['model']:<16} {row['precision']:<10} "
            f"{prefill:>14} {decode:>14} {decode_iters:>12}"
        )
    print()
    print(f"csv: {csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ModelBench cases.")
    parser.add_argument(
        "cases",
        nargs="*",
        default=["deepseek-r1"],
        choices=["deepseek-r1"],
        help="model benchmark cases to run",
    )
    parser.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help="directory for generated MLIR, objects, binaries, and results",
    )
    parser.add_argument(
        "--skip-import",
        action="store_true",
        help="reuse imported MLIR files if they already exist",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="reuse compiled object files and relink the runner",
    )
    parser.add_argument(
        "--run-only",
        action="store_true",
        help="run the existing benchmark binary without import or compile",
    )
    parser.add_argument(
        "--verbose-commands",
        action="store_true",
        help="print full compiler commands during build",
    )
    parser.add_argument(
        "--openmp-threads",
        type=int,
        default=int(os.environ.get("MODEL_BENCH_OPENMP_THREADS", "48")),
    )
    parser.add_argument(
        "--llc-attrs",
        default=os.environ.get("MODEL_BENCH_LLC_ATTRS", "-mcpu=native"),
        help='quoted attributes passed to llc, e.g. --llc-attrs="-mcpu=native"',
    )
    parser.add_argument(
        "--buddy-opt", default=default_tool("build/bin/buddy-opt", "BUDDY_OPT")
    )
    parser.add_argument(
        "--mlir-opt",
        default=default_tool("llvm/build/bin/mlir-opt", "MLIR_OPT"),
    )
    parser.add_argument(
        "--mlir-translate",
        default=default_tool("llvm/build/bin/mlir-translate", "MLIR_TRANSLATE"),
    )
    parser.add_argument(
        "--llvm-as", default=default_tool("llvm/build/bin/llvm-as", "LLVM_AS")
    )
    parser.add_argument(
        "--llc", default=default_tool("llvm/build/bin/llc", "LLC")
    )
    parser.add_argument(
        "--clangxx", default=default_tool("llvm/build/bin/clang++", "CLANGXX")
    )
    parser.add_argument(
        "--llvm-lib-dir",
        type=Path,
        default=Path(
            os.environ.get("LLVM_LIB_DIR", ROOT / "llvm" / "build" / "lib")
        ),
    )
    args = parser.parse_args()
    args.llc_attrs = shlex.split(args.llc_attrs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for case in args.cases:
        case_dir = out_dir / case
        case_dir.mkdir(parents=True, exist_ok=True)
        runner = case_dir / "modelbench-deepseek-r1-run"
        if not args.run_only:
            section(f"ModelBench Build: {case}")
            import_deepseek(args, case_dir)
            runner = build_deepseek(args, case_dir)
        else:
            print(f"[build] skipped; using {runner}")
        section(f"ModelBench Run: {case}")
        result = run(
            [str(runner)],
            label="execute benchmark runner",
            phase="run",
            capture_stdout=True,
            verbose=args.verbose_commands,
        )
        output = result.stdout.decode(errors="replace")
        csv_path = case_dir / "results.csv"
        csv_path.write_text(output)
        rows = parse_result_csv(output)
        print_result_table(rows, csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
