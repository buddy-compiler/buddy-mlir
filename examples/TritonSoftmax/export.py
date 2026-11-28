#!/usr/bin/env python3
"""复用既有 Triton 后端 AOT 导出 RISC-V 内核；不执行目标代码。"""

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import triton
import triton.backends.triton_shared.compiler as cpu_compiler
import triton.backends.triton_shared.driver as cpu_driver
from softmax_metadata import EXPORT_PROFILE, target_header, validate_export
from triton.backends.compiler import GPUTarget
from triton.backends.triton_shared.driver import CPUDriver
from triton.compiler import ASTSource


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triton-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("TRITON_RISCV_USE_IME") == "1":
        raise RuntimeError("本 profile 不支持 IME；请取消 TRITON_RISCV_USE_IME")
    for module, name in [
        (cpu_compiler, "compiler.py"),
        (cpu_driver, "driver.py"),
    ]:
        if (
            Path(module.__file__).read_bytes()
            != (args.triton_repo / "backend" / name).read_bytes()
        ):
            raise RuntimeError(f"已安装的 {name} 与指定 checkout 不一致")
    triton.runtime.driver.set_active(CPUDriver())
    examples = args.triton_repo / "python/examples"
    sys.path.insert(0, str(examples))
    spec = importlib.util.spec_from_file_location(
        "upstream_softmax", examples / "test_softmax.py"
    )
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    source = ASTSource(
        fn=example.softmax_kernel,
        signature={
            "output_ptr": "*fp32",
            "input_ptr": "*fp32",
            "input_row_stride": "i32",
            "output_row_stride": "i32",
            "n_cols": "i32",
            "BLOCK_SIZE": "constexpr",
        },
        constexprs={"BLOCK_SIZE": EXPORT_PROFILE["block_size"]},
    )
    kernel = triton.compile(
        source,
        target=GPUTarget("cpu", 0, 0),
        options={
            "target_triple": EXPORT_PROFILE["target"],
            "target_features": EXPORT_PROFILE["features"],
            "openmp_num_threads": 1,
        },
    )
    signature = re.search(
        r"^define void @softmax_kernel\(([^\n]+)\)", kernel.asm["llir"], re.M
    )
    if signature is None:
        raise RuntimeError("找不到 softmax_kernel；请检查后端 ABI")
    types = [arg.strip().split()[0] for arg in signature[1].split(",")]
    if types != EXPORT_PROFILE["llvm_parameter_types"]:
        raise RuntimeError(f"Triton ABI 已改变：{types}；必须更新 Adapter")
    if (
        f'target triple = "{EXPORT_PROFILE["target"]}"'
        not in kernel.asm["llir"]
    ):
        raise RuntimeError("导出目标架构不符")
    args.output.mkdir(parents=True, exist_ok=True)
    for stage, artifact in kernel.asm.items():
        path = args.output / f"softmax.{stage}"
        if isinstance(artifact, bytes):
            path.write_bytes(artifact)
        else:
            path.write_text(artifact)
    metadata = {
        **copy.deepcopy(EXPORT_PROFILE),
        "entry": kernel.metadata.name,
        "triton_version": triton.__version__,
        "backend_commit": subprocess.check_output(
            ["git", "-C", str(args.triton_repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "backend_dirty": bool(
            subprocess.check_output(
                ["git", "-C", str(args.triton_repo), "status", "--porcelain"],
                text=True,
            )
        ),
        "backend_files_sha256": {
            name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module, name in [
                (cpu_compiler, "compiler.py"),
                (cpu_driver, "driver.py"),
            ]
        },
        "object_sha256": hashlib.sha256(kernel.asm["obj"]).hexdigest(),
        "llvm_parameter_types": types,
    }
    validate_export(metadata, args.output / "softmax.obj")
    (args.output / "export.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    (args.output / "SoftmaxTarget.h").write_text(target_header(metadata))
    print("AOT export complete (not executed):", json.dumps(metadata))


if __name__ == "__main__":
    main()
