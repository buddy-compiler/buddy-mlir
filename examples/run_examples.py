#!/usr/bin/env python3
from __future__ import annotations
import itertools

import argparse
from dataclasses import dataclass, field
import fnmatch
from pathlib import Path
import subprocess
from typing import Any, Callable


@dataclass
class AssertSpec:
    cmd: str
    expect_contains: list[str] = field(default_factory=list)


@dataclass
class CaseSpec:
    name: str
    cmd: list[str]
    asserts: list[AssertSpec] = field(default_factory=list)
    expected_fail: str = ""


@dataclass
class FlatCase:
    id: str
    group: str
    name: str
    cmd: list[str]
    asserts: list[AssertSpec]
    expected_fail: str


@dataclass
class RunResult:
    id: str
    group: str
    name: str
    cmd: list[str]
    exit_code: int
    dry_run: bool
    failed_cmd: str = ""
    failed_assert: str = ""
    expected_fail: str = ""


@dataclass
class RunConfig:
    source_dir: str
    build_dir: str
    mlir_dir: str
    llvm_dir: str
    buddy_dir: str
    case_filter: str = "*"
    dry_run: bool = False


def get_regression_spec(cfg: RunConfig) -> dict[str, list[CaseSpec]]:
    def external_config_cmd(*example_flags: str) -> str:
        flags = " ".join(example_flags)
        return (
            f"cmake -S {cfg.source_dir} -B {cfg.build_dir} "
            "-DBUDDY_EXAMPLES_USE_EXTERNAL_TOOLS=ON "
            "-DBUDDY_MLIR_ENABLE_CORE=OFF "
            "-DBUDDY_MLIR_ENABLE_TESTS=OFF "
            "-DBUDDY_MLIR_ENABLE_EXAMPLES=ON "
            f"{flags} "
            f"-DMLIR_DIR={cfg.mlir_dir} "
            f"-DLLVM_DIR={cfg.llvm_dir} "
            f"-DCMAKE_PREFIX_PATH={cfg.buddy_dir} "
            "-DPython3_EXECUTABLE=$(which python)"
        )

    return {
        "BuddyDeepSeekR1": [
            CaseSpec(
                name="build-buddy-deepseek-r1-run-f32",
                cmd=[
                    external_config_cmd("-DBUDDY_DEEPSEEKR1_EXAMPLES=ON"),
                    f"cmake --build {cfg.build_dir} --target buddy-deepseek-r1-run -j",
                    external_config_cmd("-DBUDDY_DEEPSEEKR1_EXAMPLES=OFF"),
                ],
            ),
            CaseSpec(
                name="build-buddy-deepseek-r1-run-f16",
                cmd=[
                    external_config_cmd("-DBUDDY_DEEPSEEKR1_EXAMPLES=ON"),
                    f"cmake --build {cfg.build_dir} --target buddy-deepseek-r1-f16-run -j",
                    external_config_cmd("-DBUDDY_DEEPSEEKR1_EXAMPLES=OFF"),
                ],
            ),
            CaseSpec(
                name="build-buddy-deepseek-r1-run-bf16",
                expected_fail="Compile failed",
                cmd=[
                    external_config_cmd("-DBUDDY_DEEPSEEKR1_EXAMPLES=ON"),
                    f"cmake --build {cfg.build_dir} --target buddy-deepseek-r1-bf16-run -j",
                    external_config_cmd("-DBUDDY_DEEPSEEKR1_EXAMPLES=OFF"),
                ],
            ),
            CaseSpec(
                name="build-buddy-deepseek-r1-cli",
                cmd=[
                    external_config_cmd("-DBUDDY_DEEPSEEKR1_EXAMPLES=ON"),
                    f"cmake --build {cfg.build_dir} --target buddy-deepseek-r1-cli -j",
                    external_config_cmd("-DBUDDY_DEEPSEEKR1_EXAMPLES=OFF"),
                ],
                asserts=[
                    AssertSpec(
                        cmd=(
                            f'echo "Hello." | {cfg.build_dir}/bin/buddy-deepseek-r1-cli '
                            "--max-tokens=128 --no-stats"
                        ),
                        expect_contains=["Hello! How can I assist you today?"],
                    )
                ],
            ),
        ],
        "BuddyQwen3": [
            CaseSpec(
                name="build-buddy-qwen3-0.6b-run-f32",
                cmd=[
                    external_config_cmd("-DBUDDY_QWEN3_EXAMPLES=ON"),
                    f"cmake --build {cfg.build_dir} --target buddy-qwen3-0.6b-run -j",
                    external_config_cmd("-DBUDDY_QWEN3_EXAMPLES=OFF"),
                ],
            ),
        ],
        "BuddyLlama": [
            CaseSpec(
                name="build-buddy-llama-run-f32",
                expected_fail="Need a default model download path",
                cmd=[
                    external_config_cmd("-DBUDDY_LLAMA_EXAMPLES=ON"),
                    f"cmake --build {cfg.build_dir} --target buddy-llama-run -j",
                    external_config_cmd("-DBUDDY_LLAMA_EXAMPLES=OFF"),
                ],
            )
        ],
        "BuddyBert": [
            CaseSpec(
                name="build-buddy-bert-run-f32",
                cmd=[
                    external_config_cmd("-DBUDDY_BERT_EXAMPLES=ON"),
                    f"cmake --build {cfg.build_dir} --target buddy-bert-run -j",
                    external_config_cmd("-DBUDDY_BERT_EXAMPLES=OFF"),
                ],
            )
        ],
        "BuddyStableDiffusion": [
            CaseSpec(
                name="build-buddy-stable-diffusion-run-f32",
                expected_fail="Default model download path is not accessible",
                cmd=[
                    external_config_cmd("-DBUDDY_STABLE_DIFFUSION_EXAMPLES=ON"),
                    f"cmake --build {cfg.build_dir} --target buddy-stable-diffusion-run -j",
                    external_config_cmd(
                        "-DBUDDY_STABLE_DIFFUSION_EXAMPLES=OFF"
                    ),
                ],
            )
        ],
        "BuddyWhisper": [
            CaseSpec(
                name="build-buddy-whisper-run-f32",
                expected_fail="Segmentfault",
                cmd=[
                    external_config_cmd("-DBUDDY_WHISPER_EXAMPLES=ON"),
                    f"cmake --build {cfg.build_dir} --target buddy-whisper-run -j",
                    external_config_cmd("-DBUDDY_WHISPER_EXAMPLES=OFF"),
                ],
            )
        ],
        "BuddyLeNet": [
            CaseSpec(
                name="build-buddy-lenet-run-f32",
                expected_fail="Require train model, we should cache it somewhere",
                cmd=[
                    external_config_cmd(
                        "-DBUDDY_LENET_EXAMPLES=ON -DBUDDY_ENABLE_PNG=ON"
                    ),
                    f"cmake --build {cfg.build_dir} --target buddy-lenet-run -j",
                    external_config_cmd(
                        "-DBUDDY_LENET_EXAMPLES=OFF -DBUDDY_ENABLE_PNG=OFF"
                    ),
                ],
            )
        ],
        "BuddyMobileNetV3": [
            CaseSpec(
                name="build-buddy-mobilenetv3-run-f32",
                cmd=[
                    external_config_cmd("-DBUDDY_MOBILENETV3_EXAMPLES=ON"),
                    f"cmake --build {cfg.build_dir} --target buddy-mobilenetv3-run -j",
                    external_config_cmd("-DBUDDY_MOBILENETV3_EXAMPLES=OFF"),
                ],
            )
        ],
        "BuddyResNet18": [
            CaseSpec(
                name="build-buddy-resnet-run-f32",
                cmd=[
                    external_config_cmd("-DBUDDY_RESNET_EXAMPLES=ON"),
                    f"cmake --build {cfg.build_dir} --target buddy-resnet-run -j",
                    external_config_cmd("-DBUDDY_RESNET_EXAMPLES=OFF"),
                ],
            )
        ],
        "BuddyTransformer": [
            CaseSpec(
                name="build-buddy-transformer-runner-staged",
                cmd=[
                    external_config_cmd("-DBUDDY_TRANSFORMER_EXAMPLES=ON"),
                    f"cmake --build {cfg.build_dir} --target transformer-runner-staged -j",
                    external_config_cmd("-DBUDDY_TRANSFORMER_EXAMPLES=OFF"),
                ],
            )
        ],
        "BuddyOneDNN": [
            CaseSpec(
                name="build-buddy-onednn-run",
                expected_fail="Need install oneDNN",
                cmd=[
                    external_config_cmd("-DBUDDY_ONEDNN_EXAMPLES=ON"),
                    f"cmake --build {cfg.build_dir} --target buddy-onednn-run -j",
                    external_config_cmd("-DBUDDY_ONEDNN_EXAMPLES=OFF"),
                ],
            )
        ],
    }


def run_cases(
    cases: list[FlatCase],
    *,
    source_root: Path,
    dry_run: bool = False,
    popen_kwargs: dict[str, Any] | None = None,
    on_case_done: Callable[[FlatCase, RunResult, str, str], None] | None = None,
) -> list[RunResult]:
    popen_kwargs = popen_kwargs or {}
    results: list[RunResult] = []
    for case in cases:
        case_out: list[str] = []
        case_err: list[str] = []
        exit_code = 0
        failed_cmd = ""
        failed_assert = ""

        for cmd in case.cmd:
            if dry_run:
                case_out.append(f"DRY RUN: {cmd}\n")
                continue

            proc = subprocess.Popen(
                cmd,
                cwd=str(source_root),
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                **popen_kwargs,
            )
            out, err = proc.communicate()

            case_out.append(out or "")
            case_err.append(err or "")
            if proc.returncode != 0:
                exit_code = proc.returncode
                failed_cmd = cmd
                break

        if exit_code == 0:
            for assertion in case.asserts:
                if dry_run:
                    case_out.append(f"DRY RUN ASSERT: {assertion.cmd}\n")
                    continue
                proc = subprocess.Popen(
                    assertion.cmd,
                    cwd=str(source_root),
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    **popen_kwargs,
                )
                out, err = proc.communicate()
                case_out.append(out or "")
                case_err.append(err or "")
                if proc.returncode != 0:
                    exit_code = proc.returncode
                    failed_cmd = assertion.cmd
                    failed_assert = "assert command failed"
                    break
                combined = f"{out or ''}\n{err or ''}"
                missing = [
                    s for s in assertion.expect_contains if s not in combined
                ]
                if missing:
                    exit_code = 3
                    failed_cmd = assertion.cmd
                    failed_assert = (
                        f"missing expected output: {', '.join(missing)}"
                    )
                    case_err.append(f"[ASSERT] {failed_assert}\n")
                    break

        out = "".join(case_out)
        err = "".join(case_err)
        result = RunResult(
            id=case.id,
            group=case.group,
            name=case.name,
            cmd=case.cmd,
            exit_code=exit_code,
            dry_run=dry_run,
            failed_cmd=failed_cmd,
            failed_assert=failed_assert,
            expected_fail=case.expected_fail,
        )
        results.append(result)
        if on_case_done is not None:
            on_case_done(case, result, out, err)

    return results


def _match_case(group: str, name: str, pattern: str) -> bool:
    return fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(
        f"{group}/{name}", pattern
    )


def run_with_config(
    cfg: RunConfig, selected_cases: list[FlatCase]
) -> list[RunResult]:
    emitted_groups: set[str] = set()

    def on_case_done(
        case: FlatCase, result: RunResult, out: str, err: str
    ) -> None:
        if case.group not in emitted_groups:
            print(case.group)
            emitted_groups.add(case.group)
        if result.expected_fail:
            if result.exit_code != 0:
                print(f"  - XFAIL {case.name} ({result.expected_fail})")
                return
            print(f"  - XPASS {case.name} ({result.expected_fail})")
            return
        if result.exit_code == 0:
            print(f"  - PASS {case.name}")
            return
        print(f"  - FAIL {case.name}")
        print(f"    - cmd: {result.failed_cmd or '<unknown>'}")
        print(f"    - exit: {result.exit_code}")
        if result.failed_assert:
            print(f"    - assert: {result.failed_assert}")
        if out.strip():
            print("    - stdout:")
            print(out.rstrip("\n"))
        if err.strip():
            print("    - stderr:")
            print(err.rstrip("\n"))

    return run_cases(
        selected_cases,
        source_root=Path(cfg.source_dir),
        dry_run=cfg.dry_run,
        on_case_done=on_case_done,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Buddy-MLIR example runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run regression cases.")
    run_parser.add_argument("--source-dir", required=True)
    run_parser.add_argument("--build-dir", required=True)
    run_parser.add_argument("--mlir-dir", required=True)
    run_parser.add_argument("--llvm-dir", required=True)
    run_parser.add_argument("--buddy-dir", required=True)
    run_parser.add_argument("--filter", default="*")
    run_parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "run":
        cfg = RunConfig(
            source_dir=args.source_dir,
            build_dir=args.build_dir,
            mlir_dir=args.mlir_dir,
            llvm_dir=args.llvm_dir,
            buddy_dir=args.buddy_dir,
            case_filter=args.filter,
            dry_run=args.dry_run,
        )

        selected_cases: list[FlatCase] = []
        for group, items in get_regression_spec(cfg).items():
            for item in items:
                if not _match_case(group, item.name, cfg.case_filter):
                    continue
                selected_cases.append(
                    FlatCase(
                        id=f"{group}:{item.name}",
                        group=group,
                        name=item.name,
                        cmd=item.cmd,
                        asserts=item.asserts,
                        expected_fail=item.expected_fail,
                    )
                )

        if not selected_cases:
            print(f"error: no cases matched --filter '{cfg.case_filter}'")
            return 2

        return sum(
            (r.exit_code == 0 and r.expected_fail != "")  # FAIL
            or (r.exit_code != 0 and r.expected_fail == "")  # XPASS
            for r in run_with_config(cfg, selected_cases)
        )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
