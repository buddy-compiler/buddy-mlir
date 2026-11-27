#!/usr/bin/env python3
"""导入现有 Buddy Softmax，并在构建机交叉生成 RISC-V 对象。"""

import argparse
import subprocess
from pathlib import Path

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import linalg


def softmax(x):
    return torch.softmax(x, dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--buddy-opt", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    args = parser.parse_args()
    compiler = DynamoCompiler(primary_registry=linalg.ops_registry)
    graphs = compiler.importer(softmax, torch.ones((2, 4), dtype=torch.float32))
    if len(graphs) != 1:
        raise RuntimeError("expected one Buddy graph")
    graphs[0].lower_to_top_level_ir()
    text = str(graphs[0]._imported_module)
    args.output.write_text(text.replace("@forward(", "@buddy_softmax("))
    print("PASS imported Buddy softmax_op:", args.output)
    llvm_mlir = args.output.with_name("buddy-llvm.mlir")
    llvm_ir = args.output.with_suffix(".ll")
    obj = args.output.with_suffix(".o")
    commands = [
        [
            str(args.buddy_opt),
            str(args.output),
            "--tosa-to-linalg-pipeline=validation=false",
            "--tosa-to-arith",
            "--tosa-to-tensor",
            "--empty-tensor-to-alloc-tensor",
            "--one-shot-bufferize=bufferize-function-boundaries "
            "function-boundary-type-conversion=identity-layout-map",
            "--buffer-deallocation-pipeline",
            "--convert-linalg-to-loops",
            "--lower-affine",
            "--expand-strided-metadata",
            "--convert-scf-to-cf",
            "--convert-math-to-llvm",
            "--convert-math-to-libm",
            "--convert-arith-to-llvm",
            "--convert-index-to-llvm",
            "--finalize-memref-to-llvm",
            "--convert-cf-to-llvm",
            "--llvm-request-c-wrappers",
            "--convert-func-to-llvm",
            "--reconcile-unrealized-casts",
            "-o",
            str(llvm_mlir),
        ],
        [
            str(args.llvm_bin / "mlir-translate"),
            "--mlir-to-llvmir",
            str(llvm_mlir),
            "-o",
            str(llvm_ir),
        ],
        [
            str(args.llvm_bin / "llc"),
            "-O3",
            "-filetype=obj",
            "-mtriple=riscv64-unknown-linux-gnu",
            "-mattr=+m,+a,+f,+d,+c",
            "-target-abi=lp64d",
            "-relocation-model=pic",
            str(llvm_ir),
            "-o",
            str(obj),
        ],
    ]
    for command in commands:
        print("RUN:", command, flush=True)
        subprocess.run(command, check=True)
    print("PASS compiled Buddy fallback:", obj)
    computation = Path(__file__).with_name("computation.mlir")
    dispatch_mlir = args.output.with_name("computation-llvm.mlir")
    dispatch_ir = args.output.with_name("computation.ll")
    dispatch_obj = args.output.with_name("computation.o")
    subprocess.run(
        [
            str(args.buddy_opt),
            str(computation),
            "--convert-func-to-llvm",
            "--finalize-memref-to-llvm",
            "--reconcile-unrealized-casts",
            "-o",
            str(dispatch_mlir),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(args.llvm_bin / "mlir-translate"),
            "--mlir-to-llvmir",
            str(dispatch_mlir),
            "-o",
            str(dispatch_ir),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(args.llvm_bin / "llc"),
            "-O3",
            "-filetype=obj",
            "-mtriple=riscv64-unknown-linux-gnu",
            "-mattr=+m,+a,+f,+d,+c",
            "-target-abi=lp64d",
            "-relocation-model=pic",
            str(dispatch_ir),
            "-o",
            str(dispatch_obj),
        ],
        check=True,
    )
    print("PASS compiled Buddy dispatch computation:", dispatch_obj)


if __name__ == "__main__":
    main()
