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
# This script configures the Buddy build (sets BUDDY_BUILD_DEEPSEEK_R1_MODEL=ON;
# DeepSeek R1 uses buddy-codegen only) and builds the model + CLI in one shot.
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
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description="One-command build: single variant spec JSON → codegen + buddy-cli + .rax")
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
        help="Local HuggingFace-format model directory for PyTorch import (sets "
        "BUDDY_DSR1_LOCAL_MODEL). If --hf-config is omitted and <dir>/config.json "
        "exists, it is used for gen_config.",
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
        default="deepseek_r1_rax;buddy-cli",
        help="Semicolon-separated CMake targets (default: deepseek_r1_rax;buddy-cli)",
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
        "--dry-run",
        action="store_true",
        help="Print commands only",
    )
    args = ap.parse_args()

    here = Path.cwd()

    def resolve_from_cwd(p: Path) -> Path:
        return p.resolve() if p.is_absolute() else (here / p).resolve()

    root = resolve_from_cwd(args.source_dir) if args.source_dir is not None else _repo_root()

    spec = resolve_from_cwd(args.spec)
    if not spec.is_file():
        print(f"error: spec not found: {spec}", file=sys.stderr)
        return 1

    build_dir = resolve_from_cwd(args.build_dir)

    local_model: Path | None = None
    if args.local_model is not None:
        local_model = resolve_from_cwd(args.local_model)
        if not local_model.is_dir():
            print(
                f"error: --local-model is not a directory: {local_model}",
                file=sys.stderr,
            )
            return 1

    cmake_args = [
        f"-DBUDDY_DSR1_SPEC={spec}",
        "-DBUDDY_BUILD_DEEPSEEK_R1_MODEL=ON",
        # Syncs frontend/Python → build/python_packages/buddy/compiler (import_model.py).
        # Without this, buddy_add_model sets PYTHONPATH but the tree is empty → No module named 'buddy'.
        "-DBUDDY_MLIR_ENABLE_PYTHON_PACKAGES=ON",
    ]
    if local_model is not None:
        cmake_args.append(f"-DBUDDY_DSR1_LOCAL_MODEL={local_model}")

    if args.hf_config is not None:
        hf = resolve_from_cwd(args.hf_config)
        cmake_args.append(f"-DBUDDY_DSR1_HF_CONFIG={hf}")
    elif local_model is not None:
        auto_cfg = local_model / "config.json"
        if auto_cfg.is_file():
            cmake_args.append(f"-DBUDDY_DSR1_HF_CONFIG={auto_cfg}")

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

    build_cmd = ["cmake", "--build", str(build_dir), "--target"]
    build_cmd.extend(args.target.split(";"))
    if args.jobs > 0:
        build_cmd.extend(["-j", str(args.jobs)])
    rc = run(build_cmd)
    return rc


if __name__ == "__main__":
    sys.exit(main())
