#!/usr/bin/env python3
"""板端多进程测量及可移植结果汇总；不编译、不改变系统配置。"""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import resource
import statistics
import subprocess
import time
from pathlib import Path


def read(path):
    try:
        return Path(path).read_text().strip().replace("\x00", "")
    except OSError:
        return None


def describe(cpu):
    return {
        "machine": platform.machine(),
        "kernel": platform.release(),
        "libc": platform.libc_ver(),
        "cpu": cpu,
        "allowed_cpus": sorted(os.sched_getaffinity(0)),
        "cpuinfo_first": (read("/proc/cpuinfo") or "").split("\n\n")[0],
        "governor": read(
            f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
        ),
        "frequency_khz": read(
            f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"
        ),
        "loadavg": read("/proc/loadavg"),
    }


def run(args):
    if platform.machine() != "riscv64" or args.processes < 3:
        raise RuntimeError("仅在riscv64板端运行，至少3个独立进程")
    root = args.artifact_dir.resolve()
    output = root / "benchmark-results"
    output.mkdir(exist_ok=False)
    hashes = json.loads((root / "bundle-sha256.json").read_text())
    variants = [
        "native",
        "missing-kernel",
        "no-kernel",
        "embedded",
        "no-kernel-embedded",
    ]
    # The correctness suite deliberately removes/corrupts negative fixtures.
    # Verify every measurement input, rather than treating those mutable test
    # fixtures as required runtime dependencies. verify_board checked the full
    # bundle before running the destructive fixture tests.
    measurement_names = {
        "run",
        "buddy-cli",
        "softmax_runner.so",
        "libtriton_softmax.so",
        "measurement_kernel.so",
        "benchmark.py",
        "softmax_metadata.py",
        "export.json",
        "source-sha256.json",
    }
    measurement_names.update(name + ".rax" for name in variants)
    for name in sorted(measurement_names):
        expected = hashes[name]
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"测试包哈希不符: {name}")
    if args.cpu not in os.sched_getaffinity(0):
        raise RuntimeError("CPU不在允许集合")
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        BUDDY_SOFTMAX_BENCH_ITERATIONS=str(args.iterations),
    )
    env.pop("BUDDY_SOFTMAX_MEASURE_PREPARATION", None)
    result = {
        "hostcheck_mode": env.get(
            "BUDDY_EXTERNAL_KERNEL_HOSTCHECK", "prepared"
        ),
        "platform": describe(args.cpu),
        "processes": args.processes,
        "rounds": 10,
        "iterations": args.iterations,
        "threads": 1,
        "input": [[1, 2, 3, 4], [0, 0, 0, 0]],
        "layout_conversion": "not_applicable",
        "fallback_return_copy_bytes": 32,
        "steady": [],
        "preparation": [],
        "process_resources": [],
        "commands": [],
        "bundle_hashes": hashes,
        "verified_measurement_files": sorted(measurement_names),
    }

    def execute(name, command, current_env):
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        start = time.perf_counter_ns()
        completed = subprocess.run(
            command,
            cwd=root,
            env=current_env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=240,
        )
        elapsed = time.perf_counter_ns() - start
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        text = completed.stdout + completed.stderr
        (output / (name + ".log")).write_text(text)
        result["commands"].append(
            {"name": name, "argv": command, "returncode": completed.returncode}
        )
        if completed.returncode:
            raise RuntimeError(f"{name}失败，整组无效: {text[-2000:]}")
        return (
            text,
            elapsed,
            {
                "name": name,
                "wall_ns": elapsed,
                "cpu_seconds": after.ru_utime
                + after.ru_stime
                - before.ru_utime
                - before.ru_stime,
                "involuntary_context_switches": after.ru_nivcsw
                - before.ru_nivcsw,
                "voluntary_context_switches": after.ru_nvcsw - before.ru_nvcsw,
            },
        )

    for process in range(args.processes):
        print(f"steady process={process} start", flush=True)
        text, _, usage = execute(
            f"steady-{process}",
            [
                "taskset",
                "-c",
                str(args.cpu),
                "./run",
                "./measurement_kernel.so",
                "--benchmark",
            ],
            env,
        )
        if (
            "PASS benchmark status_paths_outputs=valid" not in text
            or "PASS finite_corpus=1000" not in text
        ):
            raise RuntimeError("计时状态/数值验证缺失")
        usage["first_session_prepare_ns"] = float(
            re.search(r"first_session_prepare_ns=([\d.eE+-]+)", text)[1]
        )
        usage["output_bits_fnv1a"] = re.search(
            r"output_bits_fnv1a=(\d+)", text
        )[1]
        for key in ["first_external_call_ns", "prepare_plus_first_call_ns"]:
            usage[key] = float(re.search(key + r"=([\d.eE+-]+)", text)[1])
        lines = text.splitlines()
        index = next(
            i for i, line in enumerate(lines) if line.startswith("round,order,")
        )
        rows = list(
            csv.DictReader(io.StringIO("\n".join(lines[index : index + 11])))
        )
        if len(rows) != 10:
            raise RuntimeError("缺少10轮计时")
        for expected_round, row in enumerate(rows):
            if int(row["round"]) != expected_round or row["order"] != (
                "AB" if expected_round % 2 == 0 else "BA"
            ):
                raise RuntimeError("计时顺序不符")
            entry = {
                "process": process,
                "round": expected_round,
                "order": row["order"],
            }
            for key, value in row.items():
                if key.endswith("_ns"):
                    entry[key] = float(value)
                    if not math.isfinite(entry[key]) or entry[key] <= 0:
                        raise RuntimeError("无效计时值")
            result["steady"].append(entry)
        result["process_resources"].append(usage)
        print(f"steady process={process} PASS", flush=True)

    prep_keys = {
        "manifest_load_ns",
        "selection_and_load_ns",
        "bindings_ns",
        "mandatory_dependencies_ns",
        "candidate_lookup_ns",
        "candidate_load_match_ns",
    }
    for process in range(args.processes):
        for round_id in range(10):
            shift = (process + round_id) % len(variants)
            order = variants[shift:] + variants[:shift]
            if round_id % 2:
                order.reverse()
            for variant in order:
                cache = (
                    output / "prep-cache" / f"p{process}-r{round_id}-{variant}"
                )
                states = (
                    ["empty", "existing"]
                    if "embedded" in variant
                    else ["not_applicable"]
                )
                for state in states:
                    current_env = dict(
                        env,
                        BUDDY_SOFTMAX_MEASURE_PREPARATION="1",
                        BUDDY_RAX_PAYLOAD_DIR=str(cache),
                    )
                    name = f"prep-{process}-{round_id}-{variant}-{state}"
                    text, elapsed, _ = execute(
                        name,
                        [
                            "taskset",
                            "-c",
                            str(args.cpu),
                            "./buddy-cli",
                            "--model",
                            f"./{variant}.rax",
                            "--prompt",
                            "softmax-contract",
                        ],
                        current_env,
                    )
                    expected_path = (
                        "external"
                        if variant in {"native", "embedded"}
                        else "fallback"
                    )
                    if (
                        f"PASS RAX Softmax path={expected_path} fallback_calls={int(expected_path == 'fallback')}"
                        not in text
                    ):
                        raise RuntimeError("准备期路径不符")
                    metrics = {
                        key: float(value)
                        for key, value in re.findall(
                            r"(\w+_ns)=([\d.eE+-]+)", text
                        )
                    }
                    if not prep_keys.issubset(metrics) or any(
                        not math.isfinite(v) or v < 0 for v in metrics.values()
                    ):
                        raise RuntimeError("准备期计时缺失或无效")
                    result["preparation"].append(
                        {
                            "process": process,
                            "round": round_id,
                            "variant": variant,
                            "payload_cache": state,
                            "cli_process_wall_ns": elapsed,
                            **metrics,
                        }
                    )
            print(
                f"preparation process={process} round={round_id} PASS",
                flush=True,
            )
    result["platform_after"] = describe(args.cpu)
    # Profiling is supplementary. Permission denial is recorded, never worked
    # around by changing kernel settings or requesting credentials.
    profile_command = [
        "perf",
        "record",
        "-e",
        "cpu-clock:u",
        "-F",
        "999",
        "-o",
        str(output / "perf.data"),
        "--",
        "taskset",
        "-c",
        str(args.cpu),
        "./run",
        "./measurement_kernel.so",
        "--benchmark",
    ]
    try:
        profile = subprocess.run(
            profile_command,
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=240,
        )
        (output / "perf.log").write_text(profile.stdout + profile.stderr)
        result["profile"] = {
            "returncode": profile.returncode,
            "available": profile.returncode == 0,
        }
        if profile.returncode == 0:
            report = subprocess.run(
                [
                    "perf",
                    "report",
                    "--stdio",
                    "--no-children",
                    "--sort",
                    "symbol",
                    "-g",
                    "none",
                    "-n",
                    "-i",
                    str(output / "perf.data"),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            (output / "perf-report.txt").write_text(
                report.stdout + report.stderr
            )
    except (OSError, subprocess.TimeoutExpired) as error:
        result["profile"] = {"available": False, "reason": type(error).__name__}
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    for key in ["steady", "preparation"]:
        with (output / (key + ".csv")).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result[key][0]))
            writer.writeheader()
            writer.writerows(result[key])
    print(
        f"PASS measurement steady={len(result['steady'])} preparation={len(result['preparation'])}",
        flush=True,
    )


def distribution(values):
    values = sorted(values)

    def percentile(fraction):
        point = (len(values) - 1) * fraction
        lower = int(point)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (point - lower)

    return {
        "n": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p25": percentile(0.25),
        "p75": percentile(0.75),
        "p95": percentile(0.95),
    }


def summarize(args):
    data = json.loads(args.results.read_text())
    metrics = [key for key in data["steady"][0] if key.endswith("_ns")]
    summary = {
        "first_call": {
            key: distribution([p[key] for p in data["process_resources"]])
            for key in [
                "first_session_prepare_ns",
                "first_external_call_ns",
                "prepare_plus_first_call_ns",
            ]
        },
        "steady": {
            key: distribution([row[key] for row in data["steady"]])
            for key in metrics
        },
        "per_process_median": [],
        "preparation": {},
        "paired_runtime_minus_direct_ns": distribution(
            [
                row["runtime_dispatch_ns"] - row["triton_direct_ns"]
                for row in data["steady"]
            ]
        ),
        "paired_graph_minus_direct_ns": distribution(
            [
                row["triton_graph_ns"] - row["triton_direct_ns"]
                for row in data["steady"]
            ]
        ),
        "paired_graph_ratio": distribution(
            [row["triton_graph_ns"] / row["buddy_ns"] for row in data["steady"]]
        ),
        # §6 metric 1: complete integration path (A) minus raw kernel entry (B).
        "paired_full_graph_minus_raw_ns": distribution(
            [
                row["triton_graph_ns"] - row["raw_kernel_ns"]
                for row in data["steady"]
            ]
        ),
        "paired_full_graph_over_raw_ratio": distribution(
            [
                row["triton_graph_ns"] / row["raw_kernel_ns"]
                for row in data["steady"]
            ]
        ),
    }
    for process in range(data["processes"]):
        rows = [row for row in data["steady"] if row["process"] == process]
        summary["per_process_median"].append(
            {
                "process": process,
                **{
                    key: statistics.median(row[key] for row in rows)
                    for key in metrics
                },
            }
        )
    groups = sorted(
        {(row["variant"], row["payload_cache"]) for row in data["preparation"]}
    )
    for variant, state in groups:
        rows = [
            row
            for row in data["preparation"]
            if row["variant"] == variant and row["payload_cache"] == state
        ]
        summary["preparation"][variant + "/" + state] = {
            key: distribution([row[key] for row in rows])
            for key in rows[0]
            if key.endswith("_ns")
        }
    summary["graph_ratio_of_medians"] = (
        summary["steady"]["triton_graph_ns"]["median"]
        / summary["steady"]["buddy_ns"]["median"]
    )
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["steady"], indent=2))
    print("graph_ratio_of_medians=", summary["graph_ratio_of_medians"])
    print(
        "metric1_full_graph_minus_raw_median_ns=",
        summary["paired_full_graph_minus_raw_ns"]["median"],
    )
    print(
        "metric1_full_graph_over_raw_ratio_median=",
        summary["paired_full_graph_over_raw_ratio"]["median"],
    )


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    measure = commands.add_parser("run")
    measure.add_argument("--artifact-dir", type=Path, default=Path.cwd())
    measure.add_argument("--cpu", type=int, default=0)
    measure.add_argument("--processes", type=int, default=3)
    measure.add_argument("--iterations", type=int, default=200000)
    report = commands.add_parser("summarize")
    report.add_argument("--results", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (run if args.command == "run" else summarize)(args)


if __name__ == "__main__":
    main()
