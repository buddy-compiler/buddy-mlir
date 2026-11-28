#!/usr/bin/env python3
"""批准后仅在 RISC-V 板端执行 CMake 消费者，不编译或测量性能。"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=int, default=0)
    args = parser.parse_args()
    if platform.machine() != "riscv64" or args.cpu not in os.sched_getaffinity(
        0
    ):
        raise RuntimeError("必须在 RISC-V 板端选择允许的 CPU")
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "sha256.json").read_text())
    for name, expected in manifest.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"产物哈希不符: {name}")
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    env.pop("LD_PRELOAD", None)
    env.pop("BUDDY_EXTERNAL_KERNEL_HOSTCHECK", None)
    results = []
    kernel = root / "libtriton_softmax.so"
    for case in ["build-tree", "installed", "relocated-consumer"]:
        for shared in [False, True]:
            name = case + ("-shared" if shared else "-executable")
            command = ["taskset", "-c", str(args.cpu)]
            if shared:
                command += [
                    str(root / "shared-loader"),
                    str(root / case / "libexternal-kernel-shared-consumer.so"),
                    str(kernel),
                ]
            else:
                command += [
                    str(root / case / "external-kernel-consumer"),
                    str(kernel),
                ]
            print("COMMAND:", command, flush=True)
            result = subprocess.run(
                command,
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                timeout=60,
            )
            text = result.stdout + result.stderr
            (root / (name + ".log")).write_text(text)
            if (
                result.returncode != 0
                or "external=1 fallback=1 invalid=1" not in text
            ):
                raise RuntimeError(f"{name}: exit={result.returncode}\n{text}")
            results.append({"name": name, "returncode": 0, "passed": True})
            print("PASS", name, flush=True)
    (root / "consumer-results.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print(
        "PASS CMake consumer cases=6 actual_triton=1 performance=not_run",
        flush=True,
    )


if __name__ == "__main__":
    main()
