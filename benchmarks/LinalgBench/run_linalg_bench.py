# ===- run_linalg_bench.py -----------------------------------------------------
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
# ===---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OPS = ROOT / "ops"
OUT_DIR = Path(
    os.environ.get(
        "LINALG_BENCH_OUT_DIR", REPO / "build" / "benchmarks" / "LinalgBench"
    )
)
GENERATED = OUT_DIR / "generated"
RESULTS = OUT_DIR / "results"
USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

FUNC_RE = re.compile(
    r"func\.func\s+@(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?P<args>.*?)\)\s*->\s*(?P<ret>tensor<[^>]+>)",
    re.S,
)
ARG_RE = re.compile(r"%[A-Za-z0-9_]+:\s*(tensor<[^>]+>)")


def default_tool(path: str, env_name: str) -> str:
    return os.environ.get(env_name, str(REPO / path))


def run(
    cmd: list[str], *, stdin: str | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
    )


def tensor_shape(tensor_type: str) -> list[int]:
    body = tensor_type.removeprefix("tensor<").removesuffix(">")
    parts = body.split("x")
    if len(parts) == 1:
        return []
    return [int(p) for p in parts[:-1]]


def tensor_element_type(tensor_type: str) -> str:
    return tensor_type.removeprefix("tensor<").removesuffix(">").split("x")[-1]


def num_elements(tensor_type: str) -> int:
    elements = 1
    for dim in tensor_shape(tensor_type):
        elements *= dim
    return elements


def parse_kernel(path: Path) -> tuple[str, list[str], str]:
    source = path.read_text()
    match = FUNC_RE.search(source)
    if not match:
        raise ValueError(
            f"cannot find a single tensor-returning func.func in {path}"
        )
    name = match.group("name")
    arg_types = ARG_RE.findall(match.group("args"))
    ret_type = match.group("ret")
    if not arg_types:
        raise ValueError(f"{path} has no tensor arguments")
    return name, arg_types, ret_type


def fill_value(element_type: str, idx: int) -> str:
    if element_type in {"f16", "bf16", "f32", "f64"}:
        return f"{1.0 + idx * 0.125:.6e}"
    if element_type == "i1":
        return "true" if idx % 2 == 0 else "false"
    if element_type.startswith("i"):
        return str(1 + idx)
    raise ValueError(f"unsupported tensor element type: {element_type}")


def linalg_to_vir_pipeline() -> list[str]:
    return [
        "-one-shot-bufferize=bufferize-function-boundaries",
        "-buffer-deallocation-pipeline",
        "-convert-bufferization-to-memref",
        "-lower-linalg-to-vir",
        "-linalg-generalize-named-ops",
        "-lower-linalg-to-vir",
    ]


def vir_to_llvm_pipeline(vector_width: int, scalable: bool) -> list[str]:
    vir_to_vector = f"-lower-vir-to-vector=vector-width={vector_width}"
    if scalable:
        vir_to_vector = f"-lower-vir-to-vector=vector-width={vector_width} use-scalable=true"
    return [
        vir_to_vector,
        "-cse",
        "-convert-vector-to-scf",
        "-lower-affine",
        "-convert-scf-to-cf",
        "-convert-cf-to-llvm",
        "-convert-vector-to-llvm",
        "-expand-strided-metadata",
        "-lower-affine",
        "-finalize-memref-to-llvm",
        "-llvm-request-c-wrappers",
        "-convert-math-to-libm",
        "-convert-vector-to-llvm",
        "-convert-math-to-llvm",
        "-convert-arith-to-llvm",
        "-convert-func-to-llvm",
        "-reconcile-unrealized-casts",
    ]


def rank(tensor_type: str) -> int:
    return len(tensor_shape(tensor_type))


def cpp_memref_type(tensor_type: str) -> str:
    elem = tensor_element_type(tensor_type)
    cpp_elem = cpp_element_type(elem)
    r = rank(tensor_type)
    if r == 0:
        return f"StridedMemRefType<{cpp_elem}, 0>"
    return f"MemRef<{cpp_elem}, {r}>"


def cpp_element_type(element_type: str) -> str:
    mapping = {
        "f32": "float",
        "f64": "double",
        "i1": "bool",
        "i8": "int8_t",
        "i32": "int32_t",
        "i64": "int64_t",
    }
    if element_type not in mapping:
        raise ValueError(
            f"AOT runtime runner currently does not support {element_type}"
        )
    return mapping[element_type]


def cpp_literal(tensor_type: str, idx: int) -> str:
    elem = tensor_element_type(tensor_type)
    value = fill_value(elem, idx)
    if elem == "f32":
        return f"{value}f"
    return value


def cpp_shape(tensor_type: str) -> str:
    shape = tensor_shape(tensor_type)
    return "{" + ", ".join(str(dim) for dim in shape) + "}"


def cpp_decl(name: str, arg_types: list[str], ret_type: str) -> str:
    params = [f"{cpp_memref_type(ret_type)} *result"]
    params += [f"{cpp_memref_type(t)} *arg{i}" for i, t in enumerate(arg_types)]
    return f'extern "C" void _mlir_ciface_{name}({", ".join(params)});'


def cpp_arg_def(i: int, tensor_type: str, suffix: str = "") -> str:
    typ = cpp_memref_type(tensor_type)
    elem = cpp_element_type(tensor_element_type(tensor_type))
    value = cpp_literal(tensor_type, i)
    storage = f"arg{i}{suffix}Storage"
    arg = f"arg{i}{suffix}"
    if rank(tensor_type) == 0:
        return f"""  {elem} {storage}[1] = {{{value}}};
  {typ} {arg};
  {arg}.basePtr = {storage};
  {arg}.data = {storage};
  {arg}.offset = 0;"""
    return (
        f"  {typ} {arg}(std::vector<size_t>{cpp_shape(tensor_type)}, {value});"
    )


def cpp_result_def(ret_type: str) -> str:
    typ = cpp_memref_type(ret_type)
    if rank(ret_type) == 0:
        return f"    {typ} result{{}};"
    return (
        f"    {typ} result(std::vector<size_t>{cpp_shape(ret_type)}, false, 0);"
    )


def cpp_first_expr(ret_type: str) -> str:
    if rank(ret_type) == 0:
        return "static_cast<double>(result.data[result.offset])"
    return "static_cast<double>(result.getData()[0])"


def cpp_release_result(ret_type: str) -> str:
    if rank(ret_type) == 0:
        return "    std::free(result.basePtr);"
    # MemRef owns the malloc'd result descriptor storage after the C interface
    # writes into it, so the MemRef destructor releases it at end of scope.
    return ""


def build_runtime_runner(
    name: str, arg_types: list[str], ret_type: str, iterations: int
) -> str:
    arg_defs = "\n".join(cpp_arg_def(i, t) for i, t in enumerate(arg_types))
    fresh_arg_defs = "\n".join(
        cpp_arg_def(i, t, "Fresh") for i, t in enumerate(arg_types)
    )
    call_args = ", ".join(
        ["&result"] + [f"&arg{i}" for i in range(len(arg_types))]
    )
    fresh_call_args = ", ".join(
        ["&freshResult"] + [f"&arg{i}Fresh" for i in range(len(arg_types))]
    )
    release = cpp_release_result(ret_type)
    if release:
        release = "\n" + release
    fresh_release = cpp_release_result(ret_type).replace(
        "result", "freshResult"
    )
    if fresh_release:
        fresh_release = "\n" + fresh_release
    return f"""#include "buddy/Core/Container.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

{cpp_decl(name, arg_types, ret_type)}

int main(int argc, char **argv) {{
  int iterations = {iterations};
  if (argc > 1)
    iterations = std::max(1, std::atoi(argv[1]));

{arg_defs}

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {{
{cpp_result_def(ret_type)}
    _mlir_ciface_{name}({call_args});
    (void){cpp_first_expr(ret_type)};{release}
  }}
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = end - start;

{fresh_arg_defs}
{cpp_result_def(ret_type).replace("result", "freshResult")}
  _mlir_ciface_{name}({fresh_call_args});
  double result0 = {cpp_first_expr(ret_type).replace("result", "freshResult")};{fresh_release}

  std::cout << "elapsed_s\\n" << elapsed.count() << "\\n" << result0 << "\\n";
  return 0;
}}
"""


def row(
    name: str,
    op_file: Path,
    status: str,
    compile_s: float,
    run_s: str = "",
    stdout: str = "",
    stderr: str = "",
) -> dict[str, str]:
    return {
        "name": name,
        "file": str(op_file),
        "status": status,
        "compile_s": f"{compile_s:.6f}",
        "run_s": run_s,
        "stdout": stdout.strip(),
        "stderr": stderr.strip(),
    }


def parse_runner_stdout(stdout: str) -> tuple[str, str]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if len(lines) >= 3 and lines[0] == "elapsed_s":
        return lines[1], lines[2]
    return "", ""


def color(text: str, code: str) -> str:
    if not USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def status_label(status: str) -> str:
    padded = f"{status:<10}"
    if status == "success":
        return color(padded, "32;1")
    return color(padded, "31;1")


def verify_vir_only(ir: str) -> tuple[bool, str]:
    if not re.search(r"\bvir\.", ir):
        return False, "lower-linalg-to-vir did not produce any vir.* operations"
    residual = sorted(set(re.findall(r"\blinalg\.[A-Za-z0-9_]+", ir)))
    if residual:
        return False, "residual Linalg ops after VIR lowering: " + ", ".join(
            residual
        )
    return True, ""


def format_seconds(value: str) -> str:
    if not value:
        return "-"
    try:
        seconds = float(value)
    except ValueError:
        return value
    if seconds < 1e-6:
        return f"{seconds * 1e9:.3f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.3f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.3f} ms"
    return f"{seconds:.3f} s"


def average_seconds(value: str, iterations: int) -> str:
    if not value:
        return ""
    try:
        seconds = float(value) / max(1, iterations)
    except ValueError:
        return value
    return f"{seconds:.9f}"


def format_number(value: str) -> str:
    if not value:
        return "-"
    try:
        number = float(value)
    except ValueError:
        return value
    return f"{number:.6g}"


def fit(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def format_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3g}G"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3g}M"
    if value >= 1_000:
        return f"{value / 1_000:.3g}K"
    return str(value)


def compact_tensor_type(tensor_type: str) -> str:
    return tensor_type.removeprefix("tensor<").removesuffix(">")


def io_shapes(result: dict[str, str]) -> tuple[str, str]:
    op_file = result.get("file", "")
    try:
        _, arg_types, ret_type = parse_kernel(Path(op_file))
    except Exception:
        return "-", "-"
    return ",".join(
        compact_tensor_type(t) for t in arg_types
    ), compact_tensor_type(ret_type)


def print_table_header(total: int, args: argparse.Namespace) -> None:
    vector_kind = "scalable" if args.scalable else "fixed"
    print(
        color(
            f"LinalgBench AOT runtime ({total} case{'s' if total != 1 else ''})",
            "36;1",
        )
    )
    print(f"  iterations   : {args.iterations}")
    print(f"  vector       : {vector_kind}, width={args.vector_width}")
    print(f"  output       : {OUT_DIR}")
    print()
    print(
        f"{'case':<32} {'status':<10} {'input':<48} {'output':<20} {'runner(avg.)':>13} "
        f"{'kernel(avg.)':>13} {'result[0]':>12}"
    )
    print(color("-" * 155, "2"))


def print_table_row(result: dict[str, str], iterations: int) -> None:
    kernel_s, result0 = parse_runner_stdout(result.get("stdout", ""))
    status = status_label(result["status"])
    input_shape, output_shape = io_shapes(result)
    print(
        f"{result['name']:<32} {status} "
        f"{fit(input_shape, 48):<48} "
        f"{fit(output_shape, 20):<20} "
        f"{format_seconds(average_seconds(result['run_s'], iterations)):>13} "
        f"{format_seconds(average_seconds(kernel_s, iterations)):>13} "
        f"{format_number(result0):>12}"
    )


def print_summary(
    rows: list[dict[str, str]], csv_path: Path, iterations: int
) -> None:
    success = sum(1 for r in rows if r["status"] == "success")
    failed = len(rows) - success
    total_kernel = 0.0
    for result in rows:
        kernel_s, _ = parse_runner_stdout(result.get("stdout", ""))
        if kernel_s:
            total_kernel += float(kernel_s)
    print(color("-" * 155, "2"))
    summary_status = color(
        f"{success}/{len(rows)} succeeded", "32;1" if failed == 0 else "31;1"
    )
    total_iterations = len(rows) * max(1, iterations)
    avg_kernel = total_kernel / total_iterations if total_iterations else 0.0
    print(
        f"summary: {summary_status}"
        f"{f', {failed} failed' if failed else ''}; "
        f"avg kernel/iter {format_seconds(f'{avg_kernel:.9f}')}"
    )
    print(f"csv: {color(str(csv_path), '36')}")
    failures = [r for r in rows if r["status"] != "success"]
    if failures:
        print()
        print(color("failures:", "31;1"))
        for failure in failures:
            detail = failure.get("stderr", "").splitlines()
            message = detail[0] if detail else failure["status"]
            print(f"  {failure['name']}: {message}")


def run_case(args: argparse.Namespace, op_file: Path) -> dict[str, str]:
    name, arg_types, ret_type = parse_kernel(op_file)
    out_dir = GENERATED / name
    out_dir.mkdir(parents=True, exist_ok=True)
    llvm_ir = out_dir / f"{name}.ll"
    so_file = out_dir / f"lib{name}.so"
    runner_cpp = out_dir / f"{name}_runner.cpp"
    runner_bin = out_dir / f"{name}_runner"

    lower_cmd = [args.buddy_opt, str(op_file), *linalg_to_vir_pipeline()]
    t0 = time.perf_counter()
    lowered = run(lower_cmd)
    if lowered.returncode != 0:
        return row(
            name,
            op_file,
            "lower-failed",
            time.perf_counter() - t0,
            stdout=lowered.stdout,
            stderr=lowered.stderr,
        )
    vir_ok, vir_error = verify_vir_only(lowered.stdout)
    if not vir_ok:
        return row(
            name,
            op_file,
            "no-vir",
            time.perf_counter() - t0,
            stdout=lowered.stdout,
            stderr=vir_error,
        )

    llvm_dialect = run(
        [
            args.buddy_opt,
            *vir_to_llvm_pipeline(args.vector_width, args.scalable),
        ],
        stdin=lowered.stdout,
    )
    if llvm_dialect.returncode != 0:
        return row(
            name,
            op_file,
            "llvm-lower-failed",
            time.perf_counter() - t0,
            stdout=llvm_dialect.stdout,
            stderr=llvm_dialect.stderr,
        )

    translated = run(
        [args.mlir_translate, "-mlir-to-llvmir"], stdin=llvm_dialect.stdout
    )
    if translated.returncode != 0:
        return row(
            name,
            op_file,
            "translate-failed",
            time.perf_counter() - t0,
            stdout=translated.stdout,
            stderr=translated.stderr,
        )
    llvm_ir.write_text(translated.stdout)

    lib_dir = str(Path(args.mlir_c_runner_utils).resolve().parent)
    compile_so = run(
        [
            args.clangxx,
            "-shared",
            "-fPIC",
            "-O3",
            str(llvm_ir),
            "-L",
            lib_dir,
            "-lmlir_c_runner_utils",
            "-lmlir_runner_utils",
            "-lm",
            f"-Wl,-rpath,{lib_dir}",
            "-o",
            str(so_file),
        ]
    )
    if compile_so.returncode != 0:
        return row(
            name,
            op_file,
            "compile-so-failed",
            time.perf_counter() - t0,
            stdout=compile_so.stdout,
            stderr=compile_so.stderr,
        )

    runner_cpp.write_text(
        build_runtime_runner(name, arg_types, ret_type, args.iterations)
    )
    compile_runner = run(
        [
            args.clangxx,
            "-O3",
            "-std=c++17",
            str(runner_cpp),
            str(so_file),
            "-I",
            str(REPO / "frontend" / "Interfaces"),
            "-I",
            str(REPO / "llvm" / "mlir" / "include"),
            "-I",
            str(REPO / "llvm" / "build" / "tools" / "mlir" / "include"),
            f"-Wl,-rpath,{out_dir.resolve()}",
            "-o",
            str(runner_bin),
        ]
    )
    compile_s = time.perf_counter() - t0
    if compile_runner.returncode != 0:
        return row(
            name,
            op_file,
            "compile-runner-failed",
            compile_s,
            stdout=compile_runner.stdout,
            stderr=compile_runner.stderr,
        )

    t1 = time.perf_counter()
    executed = run([str(runner_bin), str(args.iterations)])
    run_s = time.perf_counter() - t1
    status = "success" if executed.returncode == 0 else "run-failed"
    return row(
        name,
        op_file,
        status,
        compile_s,
        f"{run_s:.6f}",
        executed.stdout,
        executed.stderr,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cases",
        nargs="*",
        help="case names or .mlir files; default: all ops/*.mlir",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=int(os.environ.get("LINALG_BENCH_ITERS", "10")),
    )
    parser.add_argument(
        "--vector-width",
        type=int,
        default=int(os.environ.get("LINALG_BENCH_VECTOR_WIDTH", "4")),
    )
    parser.add_argument(
        "--scalable",
        action="store_true",
        help="use scalable vector lowering in lower-vir-to-vector",
    )
    parser.add_argument(
        "--buddy-opt", default=default_tool("build/bin/buddy-opt", "BUDDY_OPT")
    )
    parser.add_argument(
        "--mlir-translate",
        default=default_tool("llvm/build/bin/mlir-translate", "MLIR_TRANSLATE"),
    )
    parser.add_argument(
        "--clangxx", default=default_tool("llvm/build/bin/clang++", "CLANGXX")
    )
    parser.add_argument(
        "--mlir-c-runner-utils",
        default=default_tool(
            "llvm/build/lib/libmlir_c_runner_utils.so", "MLIR_C_RUNNER_UTILS"
        ),
    )
    parser.add_argument("--csv", default=str(RESULTS / "linalg_bench.csv"))
    args = parser.parse_args()

    files = []
    if args.cases:
        for case in args.cases:
            path = Path(case)
            files.append(path if path.suffix else OPS / f"{case}.mlir")
    else:
        files = sorted(OPS.glob("*.mlir"))

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    print_table_header(len(files), args)
    for op_file in files:
        result = run_case(args, op_file)
        result["mode"] = "aot-runtime"
        rows.append(result)
        print_table_row(result, args.iterations)

    with Path(args.csv).open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "name",
                "file",
                "status",
                "compile_s",
                "run_s",
                "stdout",
                "stderr",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print_summary(rows, Path(args.csv), args.iterations)
    return 0 if all(r["status"] == "success" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
