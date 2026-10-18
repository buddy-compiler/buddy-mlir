#!/usr/bin/env python3
"""Reproducible source, export and end-to-end CPU coverage measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from classify import STAGES, classify_static
from parse_frontend import (
    default_frontend_py,
    default_ops_dir,
    flatten_target_ops,
    load_target_op_set,
    merge_registry_keys,
    parse_all_registries,
    parse_ops_map,
)
from probes import OPERATORS, PROFILES, WORKLOADS
from report import build_report_payload, write_json, write_markdown

HERE = Path(__file__).resolve().parent


def run_static(repo, target):
    ops_map = parse_ops_map(default_frontend_py(repo))
    lowering = merge_registry_keys(parse_all_registries(default_ops_dir(repo)))
    records = [
        classify_static(row, ops_map, lowering, target)
        for row in flatten_target_ops(target)
    ]
    for record in records:
        if record.operator in OPERATORS:
            record.required_cases = list(PROFILES)
    return records


def worker_environment(repo):
    env = os.environ.copy()
    paths = [
        repo / "build/python_packages",
        repo / "llvm/build/tools/mlir/python_packages/mlir_core",
    ]
    env["PYTHONPATH"] = os.pathsep.join(
        [str(p) for p in paths] + [env.get("PYTHONPATH", "")]
    )
    env["PYTHONHASHSEED"] = "0"
    return env


def isolated_worker(repo, request, timeout):
    # Each native execution has its own process, result file and timeout.
    with tempfile.TemporaryDirectory(prefix="buddy-coverage-") as directory:
        root = Path(directory)
        request_path, result_path = root / "request.json", root / "result.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        command = [
            sys.executable,
            "-B",
            str(HERE / "worker.py"),
            "--repo-root",
            str(repo),
            "--request",
            str(request_path),
            "--result",
            str(result_path),
        ]
        expired = False
        with (
            (root / "stdout.log").open("w", encoding="utf-8") as stdout,
            (root / "stderr.log").open("w", encoding="utf-8") as stderr,
        ):
            try:
                process = subprocess.run(
                    command,
                    cwd=repo,
                    env=worker_environment(repo),
                    stdout=stdout,
                    stderr=stderr,
                    timeout=timeout,
                    check=False,
                )
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                expired, returncode = True, None
        result = {}
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                pass
        valid = isinstance(result, dict) and (
            "torch" in result
            if request["mode"] == "environment"
            else result.get("status") in ("passed", "failed", "skipped")
        )
        if not isinstance(result, dict):
            result = {}
        if expired or returncode != 0 or not valid:
            result.update(
                status="timeout" if expired else "failed",
                reason="Worker timeout"
                if expired
                else f"Worker exited {returncode} without a valid completion",
            )
        result["worker_returncode"] = returncode
        if result.get("status") in ("failed", "timeout"):
            with (root / "stderr.log").open("rb") as log:
                log.seek(0, 2)
                log.seek(max(0, log.tell() - 4000))
                result["stderr_tail"] = log.read().decode(
                    "utf-8", errors="replace"
                )
        return result


def not_run_case(case_id, status, reason):
    return {
        "case_id": case_id,
        "status": status,
        "reason": reason,
        **dict.fromkeys(STAGES, "not_run"),
    }


def run_measurements(repo, records, mode, timeout, include_workloads):
    env = isolated_worker(
        repo,
        {"mode": "environment", "operators": [r.operator for r in records]},
        timeout,
    )
    if mode == "live" and env.get("buddy_source_sha256"):
        local_sources = {
            p.relative_to(repo / "frontend/Python").as_posix(): hashlib.sha256(
                p.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            for p in (repo / "frontend/Python").rglob("*.py")
        }
        loaded_sources = dict(env["buddy_source_sha256"])
        # The wheel adds a package initializer absent from frontend/Python.
        # Keep its hash in the report, but compare only source-owned files.
        if "__init__.py" not in local_sources:
            loaded_sources.pop("__init__.py", None)
        if local_sources != loaded_sources:
            env.update(
                status="failed",
                error="Loaded Buddy Python sources differ from the inspected repository",
            )
    ready = (
        env.get("status") not in ("failed", "timeout")
        and bool(env.get("torch"))
        and (mode == "trace" or env.get("buddy"))
    )
    for record in records:
        actual = env.get("schemas", {}).get(record.operator)
        if actual and record.pytorch_schema and actual != record.pytorch_schema:
            env.setdefault("schema_errors", {})[record.operator] = (
                "Installed PyTorch schema differs from the target snapshot"
            )
    for record in records:
        record.cases.clear()
        if record.operator in OPERATORS:
            record.required_cases = list(PROFILES)
        if not record.required_cases:
            record.cases = [
                not_run_case(
                    "unconfigured", "skipped", "No explicit input contract yet"
                )
            ]
            continue
        error = env.get("schema_errors", {}).get(record.operator)
        actual_schema = env.get("schemas", {}).get(record.operator)
        if actual_schema:
            record.pytorch_schema = actual_schema
        for profile in record.required_cases:
            if error:
                result = not_run_case(profile, "failed", error)
            elif not ready:
                result = not_run_case(
                    profile,
                    "blocked",
                    env.get("error", env.get("reason", "Stack unavailable")),
                )
            else:
                result = isolated_worker(
                    repo,
                    {
                        "mode": mode,
                        "operator": record.operator,
                        "profile": profile,
                    },
                    timeout,
                )
                result["case_id"] = profile
            record.cases.append(result)
    workloads = []
    if include_workloads:
        for name in WORKLOADS:
            cases = []
            for profile in PROFILES:
                if ready:
                    result = isolated_worker(
                        repo,
                        {"mode": mode, "operator": name, "profile": profile},
                        timeout,
                    )
                    result["case_id"] = profile
                else:
                    result = not_run_case(
                        profile,
                        "blocked",
                        env.get(
                            "error", env.get("reason", "Stack unavailable")
                        ),
                    )
                cases.append(result)
            workloads.append({"name": name, "cases": cases})
    return env, workloads


def provenance(repo, target_path):
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    files = [
        target_path,
        *HERE.glob("*.py"),
        *repo.glob("frontend/Python/**/*.py"),
        repo / "tests/Python/AtenOpsCoverage/aten_coverage_runner.py",
    ]
    hashes = {}
    for path in sorted(set(files)):
        key = (
            path.relative_to(repo).as_posix()
            if path.is_relative_to(repo)
            else path.name
        )
        hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode()
    ).hexdigest()
    return {
        "git_revision": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=normal")),
        "source_sha256": digest,
        "files_sha256": hashes,
        "python": sys.version.split()[0],
    }


def exit_status(mode, records, workloads, environment, threshold=None):
    if mode == "static":
        return 0
    if environment.get("status") in ("failed", "timeout"):
        return 2
    if not environment.get("torch") or (
        mode == "live" and not environment.get("buddy")
    ):
        return 2
    if environment.get("schema_errors"):
        return 1
    cases = [c for r in records for c in r.cases] + [
        c for w in workloads for c in w["cases"]
    ]
    if any(c["status"] in ("failed", "blocked", "timeout") for c in cases):
        return 1
    if threshold is not None:
        pct = 100 * sum(r.validated() for r in records) / len(records)
        if pct < threshold:
            return 1
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=HERE.parents[1])
    parser.add_argument(
        "--target-set", type=Path, default=HERE / "data/target_ops_v1.json"
    )
    parser.add_argument("--out-dir", type=Path, default=HERE / "out")
    parser.add_argument(
        "--mode", choices=("static", "trace", "live"), default="static"
    )
    parser.add_argument(
        "--workloads",
        action="store_true",
        help="Also measure small Transformer and MoE blocks",
    )
    parser.add_argument(
        "--timeout", type=float, default=120, help="Seconds per isolated worker"
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        help="Live validated percentage required for exit 0",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a positive finite number")
    if args.min_coverage is not None and (
        args.mode != "live" or not 0 <= args.min_coverage <= 100
    ):
        parser.error(
            "--min-coverage requires live mode and a value in [0, 100]"
        )
    repo, target_path = args.repo_root.resolve(), args.target_set.resolve()
    try:
        target = load_target_op_set(target_path)
        records = run_static(repo, target)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    environment, workloads = {}, []
    source = provenance(repo, target_path)
    if args.mode != "static":
        environment, workloads = run_measurements(
            repo, records, args.mode, args.timeout, args.workloads
        )
    code = exit_status(
        args.mode, records, workloads, environment, args.min_coverage
    )
    source["changed_during_run"] = (
        source["source_sha256"]
        != provenance(repo, target_path)["source_sha256"]
    )
    if source["changed_during_run"]:
        code = 1
    payload = build_report_payload(
        records,
        target=target,
        mode=args.mode,
        provenance=source,
        environment=environment,
        workloads=workloads,
        exit_code=code,
        threshold=args.min_coverage,
    )
    write_json(payload, args.out_dir / "pytorch_op_coverage.json")
    write_markdown(payload, args.out_dir / "pytorch_op_coverage.md")
    print(
        f"mode={args.mode} exit={code} target={target['name']} ops={len(records)} "
        f"validated={payload['summary']['validated_for_profile']}"
    )
    if environment.get("error"):
        print(environment["error"])
    print(f"Report: {args.out_dir / 'pytorch_op_coverage.md'}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
