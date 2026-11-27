#!/usr/bin/env python3
"""仅在 RISC-V 板端执行正确性、错误处理和缓存测试，不测量开销。"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    if platform.machine() != "riscv64":
        raise RuntimeError("必须在 RISC-V 板端执行")
    root = args.artifact_dir.resolve()
    cpu = args.cpu if args.cpu is not None else min(os.sched_getaffinity(0))
    logs = root / "contract-logs"
    logs.mkdir(exist_ok=True)
    results = []
    base_env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    base_env.pop("BUDDY_SOFTMAX_MEASURE_PREPARATION", None)
    # An inherited diagnostic mode must not change candidate acceptance tests.
    base_env.pop("BUDDY_EXTERNAL_KERNEL_HOSTCHECK", None)

    def invoke(
        name, command, *, ok=True, contains=None, env=None, affinity=None
    ):
        mask = str(cpu) if affinity is None else affinity
        command = ["taskset", "-c", mask] + [str(arg) for arg in command]
        print("COMMAND:", name, command, flush=True)
        result = subprocess.run(
            command,
            cwd=root,
            env=env or base_env,
            text=True,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=60,
        )
        text = result.stdout + result.stderr
        (logs / (name + ".log")).write_text(text)
        if ok != (result.returncode == 0) or (
            contains and contains not in text
        ):
            raise RuntimeError(f"{name}: exit={result.returncode}\n{text}")
        if not ok and (result.returncode < 0 or "PASS RAX Softmax" in text):
            raise RuntimeError(f"{name} 崩溃或错误路径仍成功")
        if "manifest_load_ns=" in text or "round,order," in text:
            raise RuntimeError("本轮禁止开销测试")
        results.append(
            {"name": name, "returncode": result.returncode, "passed": True}
        )
        print("PASS", name, flush=True)
        return text

    cli = root / "buddy-cli"

    def rax(name, package, **kwargs):
        return invoke(
            name,
            [cli, "--model", package, "--prompt", "softmax-contract"],
            **kwargs,
        )

    invoke("abi-c", [root / "abi-layout"], contains="PASS C ABI")
    invoke(
        "runtime",
        [root / "runtime-contract"],
        contains="External kernel dispatch tests passed",
    )
    invoke(
        "compiled-graph",
        [root / "graph-contract", root / "libtriton_softmax.so"],
        contains="PASS ISA records=",
    )
    invoke(
        "dependencies",
        [root / "dependencies-contract"],
        contains="RAX dependency handle tests passed",
    )
    invoke(
        "capability-positive",
        [root / "capability-probe", root / "libtriton_softmax.so", "1"],
    )
    for mode, name in [(1, "no-os-v"), (2, "thread-v-off")]:
        env = dict(
            base_env, LD_PRELOAD=str(root / f"fixtures/capability-{mode}.so")
        )
        invoke(
            name,
            [root / "capability-probe", root / "libtriton_softmax.so", "0"],
            env=env,
            contains="compatible=0 status=1",
        )
    invoke(
        "wrong-vlen",
        [root / "capability-probe", root / "fixtures/vlen128.so", "0"],
        contains="compatible=0 status=1",
    )
    others = sorted(os.sched_getaffinity(0) - {cpu})
    if not others:
        raise RuntimeError("不能验证非单CPU亲和性：当前只允许一个CPU")
    invoke(
        "multi-cpu",
        [root / "capability-probe", root / "libtriton_softmax.so", "1"],
        affinity=f"{cpu},{others[0]}",
        contains="compatible=1 status=0",
    )
    spy_env = dict(
        base_env, LD_PRELOAD=str(root / "fixtures/host-preparation-spy.so")
    )
    for mode, expected in [
        ("positive", "steady_syscalls=0 fork_exec=1"),
        ("deny-domain", "PASS rejected domain fallback=100"),
        ("concurrent", "PASS concurrent first-use workers=8"),
    ]:
        invoke(
            "host-preparation-" + mode,
            [root / "host-preparation", root / "libtriton_softmax.so", mode],
            env=spy_env,
            contains=expected,
        )

    invoke("generate-raw-rax", [root / "rax-malformed", root])
    for package in sorted(root.glob("raw-*.rax")):
        name = package.stem.removeprefix("raw-")
        if name == "valid":
            rax("raw-valid", package, contains="PASS RAX Softmax path=external")
            continue
        if name == "unknown-code":
            diagnostic = "unknown Softmax dispatch code object id"
        elif name == "duplicate-buffer":
            diagnostic = "duplicate Softmax buffer id"
        elif name == "duplicate-code":
            diagnostic = "duplicate Softmax code object id"
        elif name == "multi-function":
            diagnostic = "accepts exactly one function"
        elif name in ["multi-dispatch", "non-dispatch"] or name.startswith(
            "barrier-"
        ):
            diagnostic = "requires one dispatch only"
        else:
            diagnostic = "invalid Softmax dispatch bindings"
        rax("raw-" + name, package, ok=False, contains=diagnostic)

    broken = root / "broken-flatbuffer.rax"
    content = bytearray((root / "native.rax").read_bytes())
    content[4:8] = b"BAD!"
    broken.write_bytes(content)
    rax("broken-flatbuffer", broken, ok=False, contains="not a valid RAX0")
    broken_payload = root / "broken-payload.rax"
    broken_payload.write_bytes((root / "embedded.rax").read_bytes()[:-16])
    rax("broken-payload", broken_payload, ok=False, contains="payload")

    cache = root / "contract-cache"
    env = dict(base_env, BUDDY_RAX_PAYLOAD_DIR=str(cache))
    probe = [root / "manifest-probe", root / "embedded.rax"]
    extracted = Path(invoke("cache-extract", probe, env=env).strip())
    expected = hashlib.sha256(extracted.read_bytes()).hexdigest()
    initial_mtime = extracted.stat().st_mtime_ns
    invoke("cache-reuse", probe, env=env)
    if extracted.stat().st_mtime_ns != initial_mtime:
        raise RuntimeError("合法缓存未被复用")
    damaged = bytearray(extracted.read_bytes())
    damaged[len(damaged) // 2] ^= 1
    extracted.write_bytes(damaged)
    invoke("cache-repair", probe, env=env)
    if hashlib.sha256(extracted.read_bytes()).hexdigest() != expected:
        raise RuntimeError("同大小损坏缓存没有修复")
    rax(
        "repaired-payload-exec",
        root / "embedded.rax",
        env=env,
        contains="path=external",
    )
    blocked = root / "not-a-directory"
    blocked.write_text("fixture")
    rax(
        "cache-unwritable",
        root / "embedded.rax",
        env=dict(base_env, BUDDY_RAX_PAYLOAD_DIR=str(blocked)),
        ok=False,
        contains="Error",
    )
    denied = root / "permission-denied-cache"
    denied.mkdir(exist_ok=True)
    denied.chmod(0o500)
    try:
        rax(
            "cache-permission-denied",
            root / "embedded.rax",
            env=dict(base_env, BUDDY_RAX_PAYLOAD_DIR=str(denied)),
            ok=False,
            contains="Permission denied",
        )
    finally:
        denied.chmod(0o700)

    relocated = root / "relocated"
    relocated.mkdir(exist_ok=True)
    for name in [
        "native.rax",
        "embedded.rax",
        "no-kernel.rax",
        "no-kernel-embedded.rax",
        "libtriton_softmax.so",
        "softmax_runner.so",
    ]:
        shutil.copyfile(root / name, relocated / name)
    rax("relocated-file", relocated / "native.rax", contains="path=external")
    (relocated / "libtriton_softmax.so").unlink()
    # The embedded package must work with both adjacent libraries absent.
    (relocated / "softmax_runner.so").unlink()
    rax(
        "embedded-self-contained",
        relocated / "embedded.rax",
        env=env,
        contains="path=external",
    )
    rax(
        "embedded-no-kernel",
        relocated / "no-kernel-embedded.rax",
        env=env,
        contains="path=fallback",
    )
    metadata = root / "metadata-fixtures"
    helper = root / "metadata-contract"
    invoke("metadata-prepare", [helper, "prepare", metadata])
    (metadata / "optional.so").unlink()
    invoke(
        "metadata-plain",
        [helper, "plain", metadata / "plain.rax"],
        contains="PASS plain RAX",
    )
    invoke(
        "metadata-embedded",
        [helper, "embedded", metadata / "embedded.rax"],
        env=dict(base_env, BUDDY_RAX_PAYLOAD_DIR=str(root / "metadata-cache")),
        contains="PASS embedded RAX",
    )
    invoke(
        "metadata-legacy",
        [helper, "legacy", metadata / "legacy.rax"],
        contains="PASS legacy",
    )
    invoke(
        "metadata-invalid",
        [helper, "invalid", metadata],
        contains="PASS malformed",
    )

    (root / "contract-results.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print(f"PASS contract groups={len(results)} performance=paused", flush=True)


if __name__ == "__main__":
    main()
