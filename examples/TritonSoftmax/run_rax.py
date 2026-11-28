#!/usr/bin/env python3
"""构建机打包（prepare）；RISC-V 板端执行与结果检查（verify）。"""

import argparse
import json
import math
import os
import platform
import shlex
import subprocess
from pathlib import Path

from softmax_metadata import kernel_attributes, validate_export


def run(command, *, env=None):
    print("COMMAND:", shlex.join(str(arg) for arg in command), flush=True)
    return subprocess.run(
        command, text=True, capture_output=True, env=env, check=True
    )


def manifest(
    *,
    metadata,
    kernel=True,
    kernel_uri="file:libtriton_softmax.so",
    target=None,
    entry=None,
    isa,
    strides="4, 1",
    primary="file:softmax_runner.so",
    scalars="-1",
    buffer_args='"input", "output"',
    second=False,
    second_isa=None,
    second_uri="file:libtriton_softmax.so",
    dispatch_target="triton_softmax",
    kernel_attrs=None,
):
    target = target if target is not None else metadata["target"]
    entry = entry if entry is not None else metadata["query_entry"]
    optional = ""
    if kernel:
        attrs = kernel_attributes(metadata)
        # Deliberate fixture overrides happen only after canonical validation.
        attrs.update(target=target, required_isa=isa)
        for key, value in (kernel_attrs or {}).items():
            if value is None:
                attrs.pop(key, None)
            else:
                attrs[key] = value
        rendered_attrs = ", ".join(
            f"{key} = {json.dumps(value)}" for key, value in attrs.items()
        )
        optional = f"""rhal.codeobj @triton_softmax {{
          id = 2 : i32, kind = "host_shared_lib", backend = "triton_cpu",
          uri = {json.dumps(kernel_uri)}, entry_symbol = {json.dumps(entry)},
          kernel_attrs = {{{rendered_attrs}}}
        }}"""
    if second:
        extra = optional.replace("@triton_softmax", "@triton_softmax_second")
        extra = extra.replace("id = 2 : i32", "id = 3 : i32")
        extra = extra.replace(json.dumps(kernel_uri), json.dumps(second_uri))
        extra = extra.replace(
            json.dumps(target), json.dumps(metadata["target"])
        )
        extra = extra.replace(
            f"required_isa = {json.dumps(isa)}",
            f"required_isa = {json.dumps(second_isa or isa)}",
        )
        optional += "\n" + extra
    # The function entry binds buffers/scalars to the kernel in ABI order and
    # names the dispatch target. It is omitted for the no-kernel variants: a
    # package without the optional kernel has no dispatch to declare.
    dispatch = ""
    if kernel:
        dispatch = (
            '        rhal.func @softmax {inputs = ["input"], '
            f'outputs = ["output"], dispatch = {json.dumps(dispatch_target)}, '
            f"args = [{buffer_args}], scalars = array<i64: {scalars}>}}\n"
        )
    return f"""module {{
      rhal.module @softmax attributes {{version = "0.1.0",
          model_name = "triton_softmax_demo", runner_library = "file:softmax_runner.so"}} {{
        {optional}
        rhal.codeobj @buddy_computation {{id = 1 : i32,
          kind = "host_shared_lib", backend = "cpu", uri = {json.dumps(primary)}}}
        rhal.buffer @input {{space = "host", type = tensor<2x4xf32>,
          layout = "row_major", strides = array<i64: {strides}>}}
        rhal.buffer @output {{space = "host", type = tensor<2x4xf32>,
          layout = "row_major", strides = array<i64: 4, 1>}}
        {dispatch}      }}
    }}
"""


# 名称、manifest 改动、嵌入与否、预期路径、诊断。
VARIANTS = [
    ("native", {}, False, "external", None),
    ("embedded", {}, True, "external", None),
    ("no-kernel", {"kernel": False}, False, "fallback", None),
    ("no-kernel-embedded", {"kernel": False}, True, "fallback", None),
    (
        "missing-kernel",
        {"kernel_uri": "file:missing_triton_softmax.so"},
        False,
        "fallback",
        "rejected=load_or_host",
    ),
    (
        "wrong-target",
        {"target": "x86_64-unknown-linux-gnu"},
        False,
        "fallback",
        "rejected=metadata",
    ),
    (
        "wrong-entry",
        {"entry": "not_the_v1_query"},
        False,
        "fallback",
        "rejected=metadata",
    ),
    (
        "descriptor-mismatch",
        {"isa": "not-the-library-profile"},
        False,
        "fallback",
        "rejected=requirements",
    ),
    ("empty-isa", {"isa": " , ; "}, False, "fallback", "rejected=metadata"),
    (
        "missing-first-valid-second",
        {"kernel_uri": "file:missing_triton_softmax.so", "second": True},
        False,
        "external",
        "rejected=load_or_host",
    ),
    (
        "wrong-first-valid-second",
        {"target": "x86_64-unknown-linux-gnu", "second": True},
        False,
        "external",
        "rejected=metadata",
    ),
    ("axis-one", {"scalars": "1"}, False, "external", None),
    (
        "axis-zero",
        {"scalars": "0"},
        False,
        None,
        "invalid Softmax buffer reference or axis",
    ),
    (
        "extra-scalar",
        {"scalars": "-1, 7"},
        False,
        None,
        "invalid Softmax dispatch bindings",
    ),
    (
        "duplicate-binding",
        {"buffer_args": '"input", "input"'},
        False,
        None,
        "invalid Softmax dispatch bindings",
    ),
    (
        "swapped-binding",
        {"buffer_args": '"output", "input"'},
        False,
        None,
        "invalid Softmax dispatch bindings",
    ),
    (
        "bad-input",
        {"strides": "8, 1"},
        False,
        None,
        "requires host row-major 2x4 FP32",
    ),
    (
        "missing-primary",
        {"primary": "file:missing_buddy_computation.so"},
        False,
        None,
        "dlopen RAX dependency failed",
    ),
]

VARIANTS.extend(
    [
        (
            "all-candidates-unavailable",
            {
                "kernel_uri": "file:missing_triton_softmax.so",
                "second": True,
                "second_uri": "file:also_missing_triton_softmax.so",
            },
            False,
            "fallback",
            [
                "candidate=triton_softmax rejected=load_or_host",
                "candidate=triton_softmax_second rejected=load_or_host",
            ],
        ),
        (
            "both-compatible-first-selected",
            {"second": True},
            False,
            "external",
            None,
        ),
        (
            "both-compatible-explicit-second",
            {"second": True, "dispatch_target": "triton_softmax_second"},
            False,
            "external",
            None,
        ),
        (
            "first-specialization-second-valid",
            {
                "kernel_uri": "file:fixtures/kernel-11.so",
                "isa": "baseline",
                "second": True,
            },
            False,
            "external",
            "rejected=specialization",
        ),
        (
            "first-isa-second-valid",
            {"isa": "not-the-library-profile", "second": True},
            False,
            "external",
            "rejected=requirements",
        ),
        (
            "repacked",
            {"kernel_attrs": {"build_revision": "second"}},
            True,
            "external",
            None,
        ),
    ]
)

for key in (
    "kernel_abi",
    "operation",
    "dtype",
    "layout",
    "shape",
    "strides",
    "axis",
    "target",
    "required_isa",
    "profile",
    "required_rvv_version",
    "rvv_vlen_bits",
    "required_matrix_features",
):
    VARIANTS.append(
        (
            "missing-" + key,
            {"kernel_attrs": {key: None}},
            False,
            "fallback",
            "rejected=metadata",
        )
    )
for key, value in {
    "kernel_abi": "unknown",
    "shape": "1,4",
    "dtype": "f16",
    "layout": "col_major",
    "required_rvv_version": "0.7",
    "rvv_vlen_bits": "128",
    "required_matrix_features": "ime",
}.items():
    VARIANTS.append(
        (
            "bad-" + key,
            {"kernel_attrs": {key: value}},
            False,
            "fallback",
            "rejected=metadata",
        )
    )
for mode, name, diagnostic in [
    (1, "abi", "rejected=load_or_host"),
    (3, "host", "rejected=load_or_host"),
    (5, "symbol", "rejected=load_or_host"),
    (7, "null-query", "rejected=load_or_host"),
    (8, "size", "rejected=load_or_host"),
    (9, "callback", "rejected=load_or_host"),
    (10, "invoke", "rejected=load_or_host"),
    (11, "specialization", "rejected=specialization"),
    (4, "failed", "external_status=3"),
    (12, "unsupported", "external_status=1"),
    (13, "invalid", "external_status=2"),
]:
    VARIANTS.append(
        (
            "fixture-" + name,
            {
                "kernel_uri": f"file:fixtures/kernel-{mode}.so",
                "isa": "baseline",
            },
            False,
            "fallback",
            diagnostic,
        )
    )


def prepare(args):
    build = args.build_dir.resolve()
    artifacts = args.artifact_dir.resolve()
    provenance = json.loads((artifacts / "export.json").read_text())
    validate_export(provenance, artifacts / "softmax.obj")
    for name in (
        "missing_triton_softmax.so",
        "also_missing_triton_softmax.so",
        "missing_buddy_computation.so",
    ):
        if (artifacts / name).exists():
            raise RuntimeError(f"缺库 fixture 已存在：{name}")
    cases = []
    variants = list(VARIANTS)
    variants.append(
        (
            "normalized-isa",
            {
                "isa": " ; ".join(
                    reversed(provenance["required_isa"].upper().split(","))
                )
            },
            False,
            "external",
            None,
        )
    )
    for name, options, embedded, expected, diagnostic in variants:
        options = {"isa": provenance["required_isa"], **options}
        if options.get("second"):
            options["second_isa"] = provenance["required_isa"]
        src = artifacts / f"{name}.mlir"
        package = artifacts / f"{name}.rax"
        src.write_text(manifest(metadata=provenance, **options))
        command = [str(build / "bin/rax-pack"), str(src), "-o", str(package)]
        if embedded:
            command.append("--embed-payload")
        run(command)
        inspection = run([str(build / "bin/rax-inspect"), str(package)]).stdout
        (artifacts / f"{name}-inspect.txt").write_text(inspection)
        selected = {
            "both-compatible-first-selected": "selected_kernel=triton_softmax code_object_id=2",
            "both-compatible-explicit-second": "selected_kernel=triton_softmax_second code_object_id=3",
        }.get(name)
        cases.append(
            {
                "name": name,
                "path": expected,
                "diagnostic": diagnostic,
                "selected": selected,
            }
        )
    (artifacts / "rax-cases.json").write_text(
        json.dumps(cases, indent=2) + "\n"
    )
    print("RAX packaging complete; target programs were not executed.")


def verify(args):
    if platform.machine() != "riscv64":
        raise RuntimeError("verify 只能在 RISC-V 板端执行")
    artifacts = args.artifact_dir.resolve()
    cpu = args.cpu if args.cpu is not None else min(os.sched_getaffinity(0))
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        BUDDY_RAX_PAYLOAD_DIR=str(artifacts / "rax-cache"),
    )
    env.pop("BUDDY_SOFTMAX_MEASURE_PREPARATION", None)
    results = []
    for case in json.loads((artifacts / "rax-cases.json").read_text()):
        command = [
            "taskset",
            "-c",
            str(cpu),
            str(artifacts / "buddy-cli"),
            "--model",
            str(artifacts / (case["name"] + ".rax")),
            "--prompt",
            "softmax-contract",
        ]
        print("COMMAND:", shlex.join(command), flush=True)
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            env=env,
            timeout=60,
        )
        log = result.stdout + result.stderr
        (artifacts / ("rax-" + case["name"] + ".log")).write_text(log)
        expected = case["path"]
        if expected:
            marker = f"PASS RAX Softmax path={expected} fallback_calls={int(expected == 'fallback')}"
            if result.returncode or marker not in log:
                raise RuntimeError(f"{case['name']} 路径错误：\n{log}")
        elif result.returncode <= 0 or "PASS RAX Softmax" in log:
            raise RuntimeError(f"{case['name']} 应明确报错，不能成功或崩溃")
        diagnostics = case["diagnostic"]
        if isinstance(diagnostics, str):
            diagnostics = [diagnostics]
        if any(item not in log for item in diagnostics or []):
            raise RuntimeError(f"{case['name']} 缺少预期诊断：\n{log}")
        if case.get("selected") and (
            case["selected"] not in log or log.count("selected_kernel=") != 1
        ):
            raise RuntimeError(
                f"{case['name']} 没有选择显式指定的候选：\n{log}"
            )
        if (
            case["name"] == "all-candidates-unavailable"
            and "selected_kernel=" in log
        ):
            raise RuntimeError("不可用候选不应被选中")
        values = []
        if expected:
            for line in result.stdout.splitlines():
                try:
                    row = [float(item) for item in line.split()]
                except ValueError:
                    continue
                if len(row) == 4:
                    values.extend(row)
            if len(values) != 8 or any(not math.isfinite(x) for x in values):
                raise RuntimeError("输出必须包含八个有限 FP32 值")
        results.append(
            {**case, "returncode": result.returncode, "output": values}
        )
        print("PASS", case["name"], flush=True)
    baseline = next(
        row["output"] for row in results if row["name"] == "no-kernel"
    )
    for row in results:
        if row["path"] != "external":
            continue
        errors = [abs(a - b) for a, b in zip(row["output"], baseline)]
        if any(e > 1e-6 + 1e-5 * abs(b) for e, b in zip(errors, baseline)):
            raise RuntimeError("Triton 输出与 Buddy 原实现不一致")
        row["max_abs_error_vs_buddy"] = max(errors)
    (artifacts / "rax-results.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "verify"])
    parser.add_argument("--build-dir", type=Path, default=repo / "build")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    (prepare if args.stage == "prepare" else verify)(args)


if __name__ == "__main__":
    main()
