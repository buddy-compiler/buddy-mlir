#!/usr/bin/env python3
# ===- build_and_publish.py - Build riscv64 .rax packages and push to HF --===//
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===----------------------------------------------------------------------===//
#
# Orchestrates the publish-models workflow:
#   1. make riscv            (host LLVM + Buddy + RISC-V toolchain/runtimes)
#   2. build_model.py        (one .rax per spec, cross-compiled for riscv64)
#   3. stage + publish_hf.py (upload to Hugging Face and tag the repo)
#
# The matrix JSON comes from scripts/model_release/prepare_matrix.py and looks
# like:
#   {"version": "0.0.6", "hf_tag": "v0.0.6",
#    "include": [{"family": "whisper", "repo": "RuyiAI/buddy-whisper",
#                 "arch": "riscv64", "snapshot": "openai/whisper-base",
#                 "variants": [{"variant": "base", "spec": "..."}]}]}
#
# Build outputs live in --cache-root so repeated runs (self-hosted runners)
# reuse the LLVM / Buddy / toolchain / model build trees.
#
# ===----------------------------------------------------------------------===//

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(cmd: list, cwd: Path, dry_run: bool, env: dict | None = None) -> None:
    printable = " ".join(str(c) for c in cmd)
    print(f"[run] {printable}", flush=True)
    if dry_run:
        return
    subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env, check=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_paths(cache_root: Path) -> dict:
    # `buddy_add_model` derives clang from
    # `${BUDDY_MLIR_BUILD_DIR}/../llvm/build/bin/clang`, so `buddy` and `llvm`
    # must share a parent directory.
    return {
        "llvm_build": cache_root / "llvm" / "build",
        "buddy_build": cache_root / "buddy",
        "riscv_build": cache_root / "riscv-gnu-toolchain" / "build",
        "riscv_install": cache_root / "riscv",
        "riscv_mlir_build": cache_root / "build-cross-mlir-rv",
        "riscv_omp_build": cache_root / "build-omp-shared-rv",
        "model_build_root": cache_root / "models",
    }


def build_deps(
    repo: Path, paths: dict, python: str, jobs: int, dry_run: bool
) -> None:
    _run(
        [
            "make",
            "riscv",
            f"LLVM_BUILD={paths['llvm_build']}",
            f"BUDDY_BUILD={paths['buddy_build']}",
            f"RISCV_BUILD={paths['riscv_build']}",
            f"RISCV_INSTALL={paths['riscv_install']}",
            f"RISCV_MLIR_BUILD={paths['riscv_mlir_build']}",
            f"RISCV_OMP_BUILD={paths['riscv_omp_build']}",
            f"PYTHON={python}",
            f"PARALLEL={jobs}",
        ],
        cwd=repo,
        dry_run=dry_run,
    )


def download_snapshot(
    repo_id: str, cache_root: Path, dry_run: bool
) -> Path | None:
    if not repo_id:
        return None
    dest = cache_root / "snapshots" / repo_id.replace("/", "--")
    if dest.is_dir() and any(dest.iterdir()):
        print(f"[snapshot] Reusing {dest}")
        return dest
    if dry_run:
        print(f"[snapshot] Would download {repo_id} -> {dest}")
        return dest
    from huggingface_hub import snapshot_download

    # Hugging Face mirrors are read-only; only downloads use the mirror
    # endpoint, never the upload performed by publish_hf.py.
    print(f"[snapshot] Downloading {repo_id} -> {dest}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        endpoint=os.environ.get("HF_DOWNLOAD_ENDPOINT") or None,
    )
    return dest


def build_variant(
    repo: Path,
    paths: dict,
    python: str,
    jobs: int,
    family: str,
    variant: str,
    spec: str,
    version: str,
    local_model: Path | None,
    dry_run: bool,
    cmake_generator: str = "Ninja",
) -> Path:
    model_build = paths["model_build_root"] / f"{family}-{variant}"

    # CMake fixes the generator in CMakeCache.txt; wipe a stale tree built with
    # a different generator (Makefiles vs Ninja) before re-configuring.
    cache_file = model_build / "CMakeCache.txt"
    if cache_file.is_file():
        current = ""
        for line in cache_file.read_text().splitlines():
            if line.startswith("CMAKE_GENERATOR:INTERNAL="):
                current = line.split("=", 1)[1]
                break
        if current and current != cmake_generator:
            print(
                f"[build] Generator changed ({current!r} -> "
                f"{cmake_generator!r}); removing {model_build}"
            )
            if not dry_run:
                shutil.rmtree(model_build)

    llvm_root = paths["llvm_build"]
    cmd = [
        python,
        "tools/buddy-codegen/build_model.py",
        "--spec",
        spec,
        "--build-dir",
        model_build,
        f"--cmake-args=-DLLVM_DIR={llvm_root}/lib/cmake/llvm",
        f"--cmake-args=-DMLIR_DIR={llvm_root}/lib/cmake/mlir",
        f"--cmake-args=-DPython3_EXECUTABLE={python}",
        f"--cmake-args=-DPython_EXECUTABLE={python}",
        f"--cmake-args=-DBUDDY_PACKAGE_VERSION={version}",
        "--cmake-args=-DBUDDY_ENABLE_TESTS=OFF",
        "-j",
        str(jobs),
    ]
    cmd += [
        "--is-rvv-crosscompile",
        "--riscv-gnu-toolchain",
        paths["riscv_install"],
        "--riscv-omp-shared",
        paths["riscv_install"] / "lib" / "libomp.so",
        "--riscv-mlir-c-runner-utils",
        paths["riscv_install"] / "lib" / "libmlir_c_runner_utils.so",
        "--buddy-mlir-build-dir",
        paths["buddy_build"],
    ]
    if local_model is not None:
        cmd += ["--local-model", local_model]
    build_env = os.environ.copy()
    build_env["CMAKE_GENERATOR"] = cmake_generator
    _run(cmd, cwd=repo, dry_run=dry_run, env=build_env)

    rax = model_build / "models" / family / f"{family}.rax"
    if not dry_run and not rax.is_file():
        raise SystemExit(f"error: expected .rax not found: {rax}")
    return rax


def stage(
    dist_root: Path,
    arch: str,
    family: str,
    variant: str,
    rax: Path,
    meta: dict,
    dry_run: bool,
) -> Path:
    out_dir = dist_root / arch / variant
    rax_name = f"{family}-{variant}.rax"
    if dry_run:
        print(f"[stage] Would stage {rax} -> {out_dir / rax_name}")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rax, out_dir / rax_name)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out_dir


def write_repo_readme(
    dist_root: Path, family: str, arch: str, entries: list
) -> None:
    lines = [
        f"# {family}",
        "",
        "Cross-compiled RISC-V (riscv64) `.rax` packages for the buddy-mlir "
        "`buddy-cli` runtime.",
        "",
        "## Variants",
        "",
        "| Variant | File |",
        "| --- | --- |",
    ]
    for e in entries:
        lines.append(
            f"| `{e['variant']}` | `{arch}/{e['variant']}/{family}-"
            f"{e['variant']}.rax` |"
        )
    lines += [
        "",
        "## Run",
        "",
        "```bash",
        f"./buddy-cli --model {arch}/<variant>/{family}-<variant>.rax",
        "```",
        "",
    ]
    (dist_root / "README.md").write_text("\n".join(lines))


def publish(
    repo: Path,
    repo_id: str,
    dist_dir: Path,
    hf_tag: str,
    version: str,
    dry_run: bool,
) -> None:
    # Resolve the helper next to this file so an older build source that does
    # not ship scripts/model_release/ still works.
    publish_script = Path(__file__).resolve().with_name("publish_hf.py")
    _run(
        [
            sys.executable,
            publish_script,
            "--repo",
            repo_id,
            "--folder",
            dist_dir,
            "--tag",
            hf_tag,
            "--commit-message",
            f"Publish model package {hf_tag}",
        ],
        cwd=repo,
        dry_run=dry_run,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--matrix-json", help="Publish matrix JSON string")
    src.add_argument(
        "--matrix-file", type=Path, help="Publish matrix JSON file"
    )
    ap.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="buddy-mlir source root (default: inferred)",
    )
    ap.add_argument(
        "--cache-root",
        type=Path,
        default=Path.home() / "buddy-rax-cache",
        help="Persistent build cache root (default: ~/buddy-rax-cache)",
    )
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument(
        "--cmake-generator",
        default="Ninja",
        help="CMake generator for model builds (default: Ninja; the Makefiles "
        "generator mishandles the arg0.data byproduct on clean builds)",
    )
    ap.add_argument("--jobs", type=int, default=0, help="0 = os.cpu_count()")
    ap.add_argument("--dist-dir", type=Path, default=Path("dist"))
    ap.add_argument(
        "--skip-deps",
        action="store_true",
        help="Skip `make riscv` (assume the cache is already populated)",
    )
    ap.add_argument(
        "--deps-stamp",
        type=Path,
        default=None,
        help="Write this file after `make riscv` succeeds (cache marker).",
    )
    ap.add_argument(
        "--skip-publish",
        action="store_true",
        help="Build and stage only; do not upload to Hugging Face",
    )
    ap.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep building the other models when one fails; exit non-zero at the end.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    payload = (
        json.loads(args.matrix_json)
        if args.matrix_json
        else json.loads(args.matrix_file.read_text())
    )
    version = payload["version"]
    hf_tag = payload["hf_tag"]

    repo = (
        args.source_dir.resolve()
        if args.source_dir is not None
        else _repo_root()
    )
    cache_root = args.cache_root.expanduser().resolve()
    jobs = args.jobs or (os.cpu_count() or 1)
    dist_root = (
        (repo / args.dist_dir).resolve()
        if not args.dist_dir.is_absolute()
        else args.dist_dir
    )
    paths = _make_paths(cache_root)
    commit = os.environ.get("GITHUB_SHA", "")

    if not args.skip_deps:
        build_deps(repo, paths, args.python, jobs, args.dry_run)
        if args.deps_stamp is not None and not args.dry_run:
            args.deps_stamp.parent.mkdir(parents=True, exist_ok=True)
            args.deps_stamp.write_text("ready\n")

    failures: list[str] = []
    for target in payload["include"]:
        family = target["family"]
        repo_id = target["repo"]
        try:
            arch = target.get("arch", "riscv64")
            local_model = download_snapshot(
                target.get("snapshot"), cache_root, args.dry_run
            )

            family_dist = dist_root / family
            entries = []
            for spec_entry in target["variants"]:
                variant = spec_entry["variant"]
                spec = spec_entry["spec"]
                print(f"[build] {family}/{variant} ({arch}) version={version}")
                rax = build_variant(
                    repo=repo,
                    paths=paths,
                    python=args.python,
                    jobs=jobs,
                    family=family,
                    variant=variant,
                    spec=spec,
                    version=version,
                    local_model=local_model,
                    dry_run=args.dry_run,
                    cmake_generator=args.cmake_generator,
                )
                meta = {
                    "family": family,
                    "variant": variant,
                    "arch": arch,
                    "version": version,
                    "hf_tag": hf_tag,
                    "spec": spec,
                    "source_commit": commit,
                    "file": f"{arch}/{variant}/{family}-{variant}.rax",
                    "built_at": dt.datetime.now(dt.UTC).isoformat(),
                }
                if not args.dry_run:
                    meta["size_bytes"] = rax.stat().st_size
                    meta["sha256"] = _sha256(rax)
                stage(
                    family_dist, arch, family, variant, rax, meta, args.dry_run
                )
                entries.append({"variant": variant})

            if not args.dry_run:
                write_repo_readme(family_dist, family, arch, entries)

            if args.skip_publish:
                print(f"[publish] Skipped {repo_id} (--skip-publish)")
            else:
                publish(
                    repo, repo_id, family_dist, hf_tag, version, args.dry_run
                )
        except Exception as exc:  # noqa: BLE001
            if not args.continue_on_error:
                raise
            failures.append(family)
            print(f"[error] {family}: {exc}", file=sys.stderr)

    if failures:
        print(
            f"[error] failed targets: {', '.join(failures)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
