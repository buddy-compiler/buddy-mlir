#!/usr/bin/env python3
# ===- build_model.py - One entry: variant spec → full CMake build --------===//
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
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
# You maintain **one** JSON file under models/<family>/specs/ (e.g. w8a16.json).
# This script configures the Buddy build and builds the model artifacts.
#
# Usage: relative paths
# (--spec, --build-dir, --hf-config, --local-model, --source-dir) are
# resolved from the **current working directory** (where you run the command),
# not from the source tree root.
#
#   cd buddy-mlir
#   python3 tools/buddy-codegen/build_model.py \\
#       --spec models/deepseek_r1/specs/w8a16.json \\
#       --build-dir build \\
#       --hf-config ~/.cache/huggingface/hub/.../config.json
#
#   # Local HF snapshot (offline import): directory with config.json + weights
#   python3 tools/buddy-codegen/build_model.py \\
#       --spec models/deepseek_r1/specs/f32.json \\
#       --build-dir build \\
#       --local-model /path/to/DeepSeek-R1-Distill-Qwen-1.5B
#
#   # From another build directory (e.g. build-review/):
#   cd build-review
#   python3 ../tools/buddy-codegen/build_model.py \\
#       --spec ../models/deepseek_r1/specs/w8a16.json \\
#       --build-dir .
#
#   # optional: skip Python import (use pre-generated MLIR)
#   python3 tools/buddy-codegen/build_model.py --spec ... \\
#       --cmake-args=-DDEEPSEEKR1_MLIR_DIR=/path/to/mlir
#
# Full import (Mode C, no DEEPSEEKR1_MLIR_DIR) needs
# torch + transformers + Buddy Python frontend.
# Use --cmake-args=-DDEEPSEEKR1_MLIR_DIR=... to skip import.
#
# ===----------------------------------------------------------------------===//

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="One-command build: single variant spec JSON → codegen + model artifacts"
    )
    ap.add_argument(
        "--spec",
        required=True,
        type=Path,
        help="Variant spec JSON path (relative paths: from current working directory)",
    )
    ap.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build"),
        help="CMake -B build directory (relative paths: from current working directory; default: build)",
    )
    ap.add_argument(
        "--hf-config",
        type=Path,
        default=None,
        help="HuggingFace config.json (passed to gen_config via CMake)",
    )
    ap.add_argument(
        "--local-model",
        type=Path,
        default=None,
        help="Local HuggingFace-format model directory for PyTorch import. "
        "For deepseek_r1, this sets BUDDY_DSR1_LOCAL_MODEL and may provide "
        "config.json. For qwen3_vl, this sets BUDDY_QWEN3_VL_MODEL_PATH.",
    )
    ap.add_argument(
        "--no-configure",
        action="store_true",
        help="Skip cmake -S/-B (only run cmake --build); use when CMakeCache is already correct",
    )
    ap.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Buddy-mlir source root for cmake -S (default: inferred from this script; relative paths: cwd)",
    )
    ap.add_argument(
        "--target",
        default=None,
        help="Semicolon-separated CMake targets (default: <model_family>_rax)",
    )
    ap.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=0,
        help="Parallel build jobs (0 = default)",
    )
    ap.add_argument(
        "--cmake-args",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra CMake -D options, e.g. --cmake-args=-DDEEPSEEKR1_MLIR_DIR=/tmp/mlir",
    )
    ap.add_argument(
        "--is-rvv-crosscompile",
        action="store_true",
        help="Enable RVV cross-compilation for model.so (sets IS_RVV_CROSSCOMPILE=ON)",
    )
    ap.add_argument(
        "--riscv-gnu-toolchain",
        type=Path,
        default=None,
        help="Path to RISCV GNU toolchain root (used for --sysroot/--gcc-toolchain)",
    )
    ap.add_argument(
        "--riscv-omp-shared",
        type=Path,
        default=None,
        help="Path to target OpenMP shared library used for RVV Stage 3 linking",
    )
    ap.add_argument(
        "--riscv-mlir-c-runner-utils",
        type=Path,
        default=None,
        help="Path to target mlir_c_runner_utils shared library used for RVV Stage 3 linking",
    )
    ap.add_argument(
        "--buddy-mlir-build-dir",
        type=Path,
        default=None,
        help="Path used as BUDDY_MLIR_BUILD_DIR to derive ../llvm/build/bin/clang(++)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only",
    )
    args = ap.parse_args()

    here = Path.cwd()

    def resolve_from_cwd(p: Path) -> Path:
        return p.resolve() if p.is_absolute() else (here / p).resolve()

    root = (
        resolve_from_cwd(args.source_dir)
        if args.source_dir is not None
        else _repo_root()
    )

    spec = resolve_from_cwd(args.spec)
    if not spec.is_file():
        print(f"error: spec not found: {spec}", file=sys.stderr)
        return 1
    try:
        spec_data = json.loads(spec.read_text())
    except Exception as e:
        print(f"error: failed to read spec JSON {spec}: {e}", file=sys.stderr)
        return 1
    model_family = spec_data.get("model_family")
    if model_family not in {"deepseek_r1", "whisper", "qwen3_vl"}:
        print(
            "error: unsupported model_family in spec: "
            f"{model_family!r} (supported: deepseek_r1, whisper, qwen3_vl)",
            file=sys.stderr,
        )
        return 1

    build_dir = resolve_from_cwd(args.build_dir)

    rvv_toolchain: Path | None = None
    rvv_omp_shared: Path | None = None
    rvv_mlir_runner_utils: Path | None = None
    rvv_build_dir: Path | None = None
    if args.is_rvv_crosscompile:
        if args.riscv_gnu_toolchain is None:
            print(
                "error: --is-rvv-crosscompile requires --riscv-gnu-toolchain",
                file=sys.stderr,
            )
            return 1
        if args.riscv_omp_shared is None:
            print(
                "error: --is-rvv-crosscompile requires --riscv-omp-shared",
                file=sys.stderr,
            )
            return 1
        if args.riscv_mlir_c_runner_utils is None:
            print(
                "error: --is-rvv-crosscompile requires --riscv-mlir-c-runner-utils",
                file=sys.stderr,
            )
            return 1

        rvv_toolchain = resolve_from_cwd(args.riscv_gnu_toolchain)
        if not rvv_toolchain.is_dir():
            print(
                f"error: --riscv-gnu-toolchain is not a directory: {rvv_toolchain}",
                file=sys.stderr,
            )
            return 1

        rvv_omp_shared = resolve_from_cwd(args.riscv_omp_shared)
        if not rvv_omp_shared.is_file():
            print(
                f"error: --riscv-omp-shared is not a file: {rvv_omp_shared}",
                file=sys.stderr,
            )
            return 1

        rvv_mlir_runner_utils = resolve_from_cwd(args.riscv_mlir_c_runner_utils)
        if not rvv_mlir_runner_utils.is_file():
            print(
                f"error: --riscv-mlir-c-runner-utils is not a file: {rvv_mlir_runner_utils}",
                file=sys.stderr,
            )
            return 1

        rvv_build_dir = (
            resolve_from_cwd(args.buddy_mlir_build_dir)
            if args.buddy_mlir_build_dir
            else build_dir
        )

    local_model: Path | None = None
    if args.local_model is not None:
        local_model = resolve_from_cwd(args.local_model)
        if not local_model.is_dir():
            print(
                f"error: --local-model is not a directory: {local_model}",
                file=sys.stderr,
            )
            return 1

    # Syncs frontend/Python → build/python_packages/buddy/compiler for import.
    # Without this, import scripts see an empty PYTHONPATH tree.
    cmake_args = ["-DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON"]
    if model_family == "deepseek_r1":
        cmake_args.extend(
            [
                f"-DBUDDY_DSR1_SPEC={spec}",
                "-DBUDDY_BUILD_DEEPSEEK_R1_MODEL=ON",
            ]
        )
        if local_model is not None:
            cmake_args.append(f"-DBUDDY_DSR1_LOCAL_MODEL={local_model}")

        if args.hf_config is not None:
            hf = resolve_from_cwd(args.hf_config)
            cmake_args.append(f"-DBUDDY_DSR1_HF_CONFIG={hf}")
        elif local_model is not None:
            auto_cfg = local_model / "config.json"
            if auto_cfg.is_file():
                cmake_args.append(f"-DBUDDY_DSR1_HF_CONFIG={auto_cfg}")
    elif model_family == "whisper":
        cmake_args.extend(
            [
                f"-DBUDDY_WHISPER_SPEC={spec}",
                "-DBUDDY_BUILD_WHISPER_MODEL=ON",
            ]
        )
        if local_model is not None:
            cmake_args.append(f"-DBUDDY_WHISPER_MODEL_PATH={local_model}")
        if args.hf_config is not None:
            print(
                "warning: --hf-config is ignored for whisper; the Whisper "
                "importer uses --spec and optional --local-model.",
                file=sys.stderr,
            )
    elif model_family == "qwen3_vl":
        cmake_args.extend(
            [
                f"-DBUDDY_QWEN3_VL_SPEC={spec}",
                "-DBUDDY_BUILD_QWEN3_VL_MODEL=ON",
            ]
        )
        if local_model is not None:
            cmake_args.append(f"-DBUDDY_QWEN3_VL_MODEL_PATH={local_model}")
        if args.hf_config is not None:
            print(
                "warning: --hf-config is ignored for qwen3_vl; the Qwen3-VL "
                "importer uses --spec and optional --local-model.",
                file=sys.stderr,
            )

    if args.is_rvv_crosscompile:
        cmake_args.extend(
            [
                "-DIS_RVV_CROSSCOMPILE=ON",
                f"-DRISCV_GNU_TOOLCHAIN={rvv_toolchain}",
                f"-DRISCV_OMP_SHARED={rvv_omp_shared}",
                f"-DRISCV_MLIR_C_RUNNER_UTILS={rvv_mlir_runner_utils}",
                f"-DBUDDY_MLIR_BUILD_DIR={rvv_build_dir}",
            ]
        )

    for extra in args.cmake_args:
        if extra.startswith("-D"):
            cmake_args.append(extra)
        else:
            cmake_args.append(f"-D{extra}")

    def run(cmd: list[str]) -> int:
        print("[build_model]", " ".join(cmd), file=sys.stderr)
        if args.dry_run:
            return 0
        r = subprocess.run(cmd, cwd=root)
        return r.returncode

    if not args.no_configure:
        cmd = ["cmake", "-S", str(root), "-B", str(build_dir)] + cmake_args
        rc = run(cmd)
        if rc != 0:
            return rc

    target = args.target or f"{model_family}_rax"
    build_cmd = ["cmake", "--build", str(build_dir), "--target"]
    build_cmd.extend(target.split(";"))
    if args.jobs > 0:
        build_cmd.extend(["-j", str(args.jobs)])
    rc = run(build_cmd)
    return rc


if __name__ == "__main__":
    sys.exit(main())
