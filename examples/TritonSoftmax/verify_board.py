#!/usr/bin/env python3
"""板端空目录重放：核对包、执行正确性和回归，不运行开销测量。"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=Path.cwd())
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    if platform.machine() != "riscv64":
        raise RuntimeError("仅在 RISC-V 板端执行")
    root = args.artifact_dir.resolve()
    # A new extraction directory must start without runtime caches.
    for name in [
        "rax-cache",
        "contract-cache",
        "metadata-cache",
        "contract-results.json",
        "rax-results.json",
    ]:
        if (root / name).exists():
            raise RuntimeError(f"干净重放不允许残留 {name}")
    checksums = json.loads((root / "bundle-sha256.json").read_text())
    for name, digest in checksums.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"文件哈希不符: {name}")
    print(f"PASS clean bundle hashes={len(checksums)} caches=empty", flush=True)
    cpu = args.cpu if args.cpu is not None else min(os.sched_getaffinity(0))
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    env.pop("BUDDY_SOFTMAX_MEASURE_PREPARATION", None)
    commands = [
        ["taskset", "-c", str(cpu), str(root / "standalone")],
        [
            "taskset",
            "-c",
            str(cpu),
            str(root / "run"),
            str(root / "libtriton_softmax.so"),
        ],
        [
            sys.executable,
            str(root / "run_rax.py"),
            "verify",
            "--artifact-dir",
            str(root),
            "--cpu",
            str(cpu),
        ],
        [
            sys.executable,
            str(root / "verify_contracts.py"),
            "--artifact-dir",
            str(root),
            "--cpu",
            str(cpu),
        ],
    ]
    for command in commands:
        print("COMMAND:", command, flush=True)
        subprocess.run(
            command,
            cwd=root,
            env=env,
            stdin=subprocess.DEVNULL,
            check=True,
            timeout=180,
        )
    summary = {
        "rax_variants": len(
            json.loads((root / "rax-results.json").read_text())
        ),
        "contract_groups": len(
            json.loads((root / "contract-results.json").read_text())
        ),
        "bundle_files": len(checksums),
        "performance": "paused",
        "clean_replay": True,
    }
    (root / "board-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("PASS clean replay", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
