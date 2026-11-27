#!/usr/bin/env python3
"""检查交叉产物并生成自包含板端测试包，不执行目标程序。"""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

TARGETS = [
    "standalone",
    "run",
    "buddy-cli",
    "softmax_runner.so",
    "libtriton_softmax.so",
    "runtime-contract",
    "graph-contract",
    "rax-malformed",
    "abi-layout",
    "metadata-contract",
    "dependencies-contract",
    "manifest-probe",
    "capability-probe",
    "host-preparation",
    "vector-policy-exec",
    "measurement_kernel.so",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--llvm-bin", type=Path, required=True)
    args = parser.parse_args()
    out, build, llvm = (
        args.artifact_dir.resolve(),
        args.build_dir.resolve(),
        args.llvm_bin.resolve(),
    )
    repo = Path(__file__).resolve().parents[2]
    fixtures = out / "metadata-fixtures"
    fixtures.mkdir(exist_ok=True)
    subprocess.run(
        [
            str(llvm / "split-file"),
            str(repo / "tests/Runtime/core/rax-kernel-metadata.mlir"),
            str(fixtures),
        ],
        check=True,
    )
    for name in ["primary.so", "dependency.so", "optional.so"]:
        shutil.copyfile(out / "fixtures/dependency.so", fixtures / name)
    for source, name, embedded in [
        ("metadata", "plain", False),
        ("metadata", "embedded", True),
        ("legacy", "legacy", False),
    ]:
        command = [
            str(build / "bin/rax-pack"),
            str(fixtures / (source + ".mlir")),
            "-o",
            str(fixtures / (name + ".rax")),
        ]
        if embedded:
            command.append("--embed-payload")
        subprocess.run(command, check=True)
        if name in {"plain", "embedded"}:
            inspection = subprocess.check_output(
                [
                    str(build / "bin/rax-inspect"),
                    str(fixtures / (name + ".rax")),
                ],
                text=True,
            )
            (out / ("inspect-" + name + ".log")).write_text(inspection)
            subprocess.run(
                [
                    str(llvm / "FileCheck"),
                    str(repo / "tests/Runtime/core/rax-kernel-metadata.mlir"),
                    "--check-prefix=" + ("EMBED" if embedded else "INSPECT"),
                ],
                input=inspection,
                text=True,
                check=True,
            )
    writer_checks = []
    for name in ["bad-optional", "bad-type", "bad-strides", "bad-layout"]:
        command = [
            str(build / "bin/rax-pack"),
            str(fixtures / (name + ".mlir")),
            "-o",
            str(out / (name + ".rax")),
        ]
        result = subprocess.run(command, text=True, capture_output=True)
        (out / ("pack-" + name + ".log")).write_text(
            result.stdout + result.stderr
        )
        if result.returncode <= 0:
            raise RuntimeError(f"writer 应正常拒绝 {name}，不能成功或崩溃")
        prefix = "BAD-" + name.removeprefix("bad-").upper()
        subprocess.run(
            [
                str(llvm / "FileCheck"),
                str(repo / "tests/Runtime/core/rax-kernel-metadata.mlir"),
                "--check-prefix=" + prefix,
            ],
            input=result.stdout + result.stderr,
            text=True,
            check=True,
        )
        writer_checks.append(
            {
                "name": name,
                "returncode": result.returncode,
                "diagnostic_filecheck": prefix,
            }
        )
    (out / "writer-checks.json").write_text(
        json.dumps(writer_checks, indent=2) + "\n"
    )
    elfs = (
        [out / name for name in TARGETS]
        + list((out / "fixtures").glob("*.so"))
        + [out / name for name in ["softmax.obj", "buddy.o", "computation.o"]]
    )
    for path in elfs:
        headers = subprocess.check_output(
            [str(llvm / "llvm-readobj"), "--file-headers", str(path)], text=True
        )
        if (
            "elf64-littleriscv" not in headers
            or "EF_RISCV_FLOAT_ABI_DOUBLE" not in headers
        ):
            raise RuntimeError(f"不是 RISC-V LP64D 产物: {path.name}")
        (out / (path.name + ".headers.txt")).write_text(headers)
    exports = subprocess.check_output(
        [
            str(llvm / "llvm-nm"),
            "-D",
            "--defined-only",
            "-j",
            str(out / "libtriton_softmax.so"),
        ],
        text=True,
    ).strip()
    if exports != "buddy_external_kernel_query_v1":
        raise RuntimeError(f"错误的内核导出符号: {exports}")
    needed = subprocess.check_output(
        [
            str(llvm / "llvm-readobj"),
            "--needed-libs",
            str(out / "softmax_runner.so"),
        ],
        text=True,
    )
    (out / "runner-dependencies.txt").write_text(needed)
    if "libtriton_softmax" in needed:
        raise RuntimeError("可选内核成为了强制依赖")
    for name in [
        "run_rax.py",
        "verify_contracts.py",
        "verify_board.py",
        "softmax_metadata.py",
        "benchmark.py",
        "compare_host_modes.py",
    ]:
        shutil.copyfile(Path(__file__).with_name(name), out / name)
    sources = [p for p in Path(__file__).parent.iterdir() if p.is_file()]
    sources += [
        p
        for p in (Path(__file__).parent / "diagnostics").rglob("*")
        if p.is_file()
    ]
    sources += [
        repo / name
        for name in [
            "runtime/core/ExternalKernel.cpp",
            "runtime/include/buddy/runtime/core/ModelManifest.h",
            "runtime/include/buddy/runtime/core/ExternalKernel.h",
            "runtime/include/buddy/runtime/core/ExternalKernelABI.h",
            "runtime/include/buddy/runtime/core/ExternalKernelValidation.h",
        ]
    ]
    (out / "source-sha256.json").write_text(
        json.dumps(
            {
                str(p.relative_to(repo)): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in sources
            },
            indent=2,
        )
        + "\n"
    )
    files = [
        out / name
        for name in TARGETS
        + [
            "run_rax.py",
            "verify_contracts.py",
            "verify_board.py",
            "softmax_metadata.py",
            "benchmark.py",
            "compare_host_modes.py",
            "rax-cases.json",
            "export.json",
            "source-sha256.json",
        ]
    ]
    files += (
        list(out.glob("*.rax"))
        + list((out / "fixtures").glob("*.so"))
        + [p for p in fixtures.iterdir() if p.is_file()]
    )
    checksums = {
        str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    (out / "bundle-sha256.json").write_text(
        json.dumps(checksums, indent=2) + "\n"
    )
    with tarfile.open(out / "test-bundle.tar.gz", "w:gz") as archive:
        for path in files + [out / "bundle-sha256.json"]:
            archive.add(path, arcname=str(path.relative_to(out)))
    (out / "artifact-checks.json").write_text(
        json.dumps(
            {
                "riscv_lp64d_elfs": len(elfs),
                "bundle_files": len(checksums),
                "query_only": True,
                "optional_not_needed": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"PASS packaged {len(elfs)} RISC-V ELF artifacts; no target executed")


if __name__ == "__main__":
    main()
