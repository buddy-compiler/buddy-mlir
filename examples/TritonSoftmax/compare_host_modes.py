#!/usr/bin/env python3
"""批准后在板端交替测严格/准备期模式；不编译、不调整系统权限。"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    args = parser.parse_args()
    root = args.artifact_dir.resolve()
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    # Copy immutable bundle inputs, never reuse mutable correctness fixtures or
    # extraction caches. Each mode runs identical binaries and kernel objects.
    names = {
        "run",
        "buddy-cli",
        "softmax_runner.so",
        "libtriton_softmax.so",
        "measurement_kernel.so",
        "benchmark.py",
        "softmax_metadata.py",
        "export.json",
        "source-sha256.json",
        "bundle-sha256.json",
    }
    names.update(
        name + ".rax"
        for name in [
            "native",
            "missing-kernel",
            "no-kernel",
            "embedded",
            "no-kernel-embedded",
        ]
    )
    runs = []
    output_hashes = set()
    for index, mode in enumerate(["strict", "prepared", "strict", "prepared"]):
        target = output / f"{index}-{mode}"
        target.mkdir()
        for name in sorted(names):
            source = root / name
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            destination.chmod(source.stat().st_mode)
        env = dict(os.environ)
        env.pop("LD_PRELOAD", None)
        if mode == "strict":
            env["BUDDY_EXTERNAL_KERNEL_HOSTCHECK"] = "strict"
        else:
            env.pop("BUDDY_EXTERNAL_KERNEL_HOSTCHECK", None)
        command = [
            sys.executable,
            str(target / "benchmark.py"),
            "run",
            "--artifact-dir",
            str(target),
            "--cpu",
            str(args.cpu),
            "--processes",
            "3",
            "--iterations",
            "200000",
        ]
        print("COMMAND:", command, "hostcheck=", mode, flush=True)
        subprocess.run(command, env=env, check=True)
        results = json.loads(
            (target / "benchmark-results/results.json").read_text()
        )
        output_hashes.update(
            p["output_bits_fnv1a"] for p in results["process_resources"]
        )
        if len(output_hashes) != 1:
            raise RuntimeError("严格基线与准备期版本的输出位模式不一致")
        summary = target / "benchmark-results/summary.json"
        subprocess.run(
            [
                sys.executable,
                str(target / "benchmark.py"),
                "summarize",
                "--results",
                str(target / "benchmark-results/results.json"),
                "--output",
                str(summary),
            ],
            check=True,
        )
        runs.append(
            {
                "index": index,
                "mode": mode,
                "summary": str(summary.relative_to(output)),
                "data": json.loads(summary.read_text()),
            }
        )
    (output / "comparison.json").write_text(json.dumps(runs, indent=2) + "\n")
    print("PASS comparison order=strict,prepared,strict,prepared", flush=True)


if __name__ == "__main__":
    main()
