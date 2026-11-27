#!/usr/bin/env python3
"""在构建机交叉构建契约测试；不执行目标程序。"""

import argparse
import os
import shlex
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    out = args.artifact_dir.resolve()
    source = repo / "examples/TritonSoftmax"
    tests = repo / "tests/Runtime/core"
    runtime = repo / "runtime/core"
    compiler = shlex.split(os.environ["CROSS_CXX"])
    common = compiler + [
        "-std=c++17",
        "-O2",
        "-march=rv64gc",
        "-mabi=lp64d",
        "-fPIC",
        "-pthread",
        "-I" + str(repo / "runtime/include"),
        "-I" + str(out),
        "-I" + str(out / "include"),
        "-I" + str(args.build_dir.resolve() / "runtime/include"),
    ]
    (out / "fixtures").mkdir(exist_ok=True)

    def build(name, inputs, flags=()):
        command = (
            common
            + list(flags)
            + [str(p) for p in inputs]
            + ["-ldl", "-lm", "-o", str(out / name)]
        )
        print("COMMAND:", shlex.join(command), flush=True)
        subprocess.run(command, check=True)

    mapping = {
        "VALID": 0,
        "BAD_ABI": 1,
        "BAD_TARGET": 2,
        "BAD_HOST": 3,
        "FAILED": 4,
        "NO_SYMBOL": 5,
        "FEATURE_PLUGIN": 6,
        "NULL_QUERY": 7,
        "BAD_SIZE": 8,
        "NO_CALLBACK": 9,
        "NO_INVOKE": 10,
        "SHAPE": 11,
        "UNSUPPORTED": 12,
        "INVALID": 13,
    }
    for mode in range(14):
        build(
            f"fixtures/kernel-{mode}.so",
            [tests / "ExternalKernelTestPlugin.cpp"],
            ["-shared", f"-DTEST_MODE={mode}"],
        )
    defines = [
        f'-DTEST_{key}="./fixtures/kernel-{mode}.so"'
        for key, mode in mapping.items()
    ]
    build(
        "runtime-contract",
        [tests / "ExternalKernelTest.cpp", runtime / "ExternalKernel.cpp"],
        defines,
    )
    build(
        "graph-contract",
        [
            source / "ContractTest.cpp",
            source / "SoftmaxSession.cpp",
            runtime / "ExternalKernel.cpp",
            out / "buddy.o",
            out / "computation.o",
        ],
    )
    build("rax-malformed", [source / "RaxMalformed.cpp"])
    build("manifest-probe", [source / "ManifestProbe.cpp"])
    build("capability-probe", [source / "CapabilityProbe.cpp"])
    build(
        "host-preparation",
        [source / "HostPreparationTest.cpp", runtime / "ExternalKernel.cpp"],
    )
    policy_command = compiler + [
        "-march=rv64gc",
        "-mabi=lp64d",
        "-nostdlib",
        "-static",
        "-Wl,-e,_start",
        str(source / "VectorPolicyExec.S"),
        "-o",
        str(out / "vector-policy-exec"),
    ]
    print("COMMAND:", shlex.join(policy_command), flush=True)
    subprocess.run(policy_command, check=True)
    build(
        "fixtures/host-preparation-spy.so",
        [source / "HostPreparationOverride.cpp"],
        ["-shared"],
    )
    for mode in [1, 2]:
        build(
            f"fixtures/capability-{mode}.so",
            [source / "CapabilityOverride.cpp"],
            ["-shared", f"-DTEST_CAP_MODE={mode}"],
        )
    vlen = out / "fixtures/vlen"
    vlen.mkdir(exist_ok=True)
    for name in ["Adapter.cpp", "RiscvCapabilities.h", "HostPreparation.h"]:
        shutil.copyfile(source / name, vlen / name)
    (vlen / "SoftmaxTarget.h").write_text(
        (out / "SoftmaxTarget.h").read_text().replace("256", "128")
    )
    build(
        "fixtures/vlen128.so",
        [vlen / "Adapter.cpp", out / "softmax.obj"],
        ["-shared"],
    )
    build("abi-layout", [source / "abi-layout.c"], ["-x", "c", "-std=c11"])
    build(
        "fixtures/dependency.so",
        [tests / "RaxDependenciesTestPlugin.cpp"],
        ["-shared"],
    )
    build(
        "metadata-contract",
        [
            tests / "Inputs/RaxKernelMetadataCheck.cpp",
            runtime / "ExternalKernel.cpp",
            runtime / "RaxDependenciesHandle.cpp",
        ],
        ['-DBUDDY_RAX_METADATA_TEST_PLUGIN="./fixtures/dependency.so"'],
    )
    build(
        "dependencies-contract",
        [
            tests / "RaxDependenciesHandleTest.cpp",
            runtime / "RaxDependenciesHandle.cpp",
        ],
        ['-DBUDDY_RAX_DEPENDENCY_TEST_PLUGIN="./fixtures/dependency.so"'],
    )
    print("Cross-compiled all contract tests; nothing executed on build host.")


if __name__ == "__main__":
    main()
