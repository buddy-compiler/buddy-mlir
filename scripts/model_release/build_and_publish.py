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
# Reads models/publish.json, cross-compiles one riscv64 .rax per spec and
# uploads them to Hugging Face (one repo per model family), tagging each repo
# with the CLI version derived from --tag.
#
#   1. make riscv        host LLVM + Buddy + RISC-V toolchain/runtimes
#   2. build_model.py    one .rax per spec, cross-compiled for riscv64
#   3. upload + tag      push to Hugging Face
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
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(cmd: list, cwd: Path, env: dict | None = None) -> None:
    print(f"[run] {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env, check=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tag_to_version(tag: str) -> str:
    """`release/v0.0.6` / a commit SHA / a branch -> a usable version."""
    tag = tag.strip()
    for prefix in ("refs/tags/", "release/", "nightly/"):
        if tag.startswith(prefix):
            tag = tag[len(prefix) :]
            break
    if tag.startswith("v"):
        tag = tag[1:]
    if re.fullmatch(r"[0-9a-f]{40}", tag):
        tag = tag[:7]
    tag = re.sub(r"[^A-Za-z0-9._-]+", "-", tag).strip("-.")
    if not tag:
        raise SystemExit(f"error: could not derive a version from {tag!r}")
    return tag


def _load_targets(registry_path: Path) -> list:
    """Flatten the enabled entries of models/publish.json into build targets."""
    registry = json.loads(registry_path.read_text())
    namespace = registry["namespace"]
    arch = registry.get("arch", "riscv64")
    targets = []
    for entry in registry["targets"]:
        if not entry.get("enabled") or not entry.get("variants"):
            continue
        targets.append(
            {
                "family": entry["family"],
                "repo": f"{namespace}/{entry['repo']}",
                "arch": arch,
                "snapshot": entry.get("snapshot"),
                "variants": [
                    {"variant": v, "spec": s}
                    for v, s in entry["variants"].items()
                ],
            }
        )
    if not targets:
        raise SystemExit("error: no enabled model targets in the registry")
    return targets


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


def _build_deps(repo: Path, paths: dict, python: str, jobs: int) -> None:
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
    )


def _download_snapshot(repo_id: str, cache_root: Path) -> Path | None:
    if not repo_id:
        return None
    dest = cache_root / "snapshots" / repo_id.replace("/", "--")
    if dest.is_dir() and any(dest.iterdir()):
        print(f"[snapshot] Reusing {dest}")
        return dest
    from huggingface_hub import snapshot_download

    print(f"[snapshot] Downloading {repo_id} -> {dest}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        endpoint=os.environ.get("HF_DOWNLOAD_ENDPOINT") or None,
    )
    return dest


def _build_variant(
    repo: Path,
    paths: dict,
    python: str,
    jobs: int,
    family: str,
    variant: str,
    version: str,
    local_model: Path | None,
) -> Path:
    model_build = paths["model_build_root"] / f"{family}-{variant}"

    # Always build with Ninja: the Makefiles generator mishandles the arg0.data
    # byproduct. CMake fixes the generator in CMakeCache.txt, so drop a stale
    # tree built with a different one.
    cache_file = model_build / "CMakeCache.txt"
    if cache_file.is_file():
        for line in cache_file.read_text().splitlines():
            if line.startswith("CMAKE_GENERATOR:INTERNAL="):
                if line.split("=", 1)[1] != "Ninja":
                    print(f"[build] Removing {model_build} (generator changed)")
                    shutil.rmtree(model_build)
                break

    # Reuse the Makefile's per-spec model target. `-o riscv` skips its `riscv`
    # dependency, which `_build_deps` already built.
    cmd = [
        "make",
        "-o",
        "riscv",
        f"riscv-model-{family}-{variant}",
        f"LLVM_BUILD={paths['llvm_build']}",
        f"BUDDY_BUILD={paths['buddy_build']}",
        f"RISCV_BUILD={paths['riscv_build']}",
        f"RISCV_INSTALL={paths['riscv_install']}",
        f"RISCV_MLIR_BUILD={paths['riscv_mlir_build']}",
        f"RISCV_OMP_BUILD={paths['riscv_omp_build']}",
        f"RISCV_MODEL_BUILD={paths['model_build_root']}",
        f"BUDDY_PACKAGE_VERSION={version}",
        f"PYTHON={python}",
        f"PARALLEL={jobs}",
    ]
    if local_model is not None:
        cmd.append(f"RISCV_MODEL_LOCAL={local_model}")
    env = os.environ.copy()
    env["CMAKE_GENERATOR"] = "Ninja"
    _run(cmd, cwd=repo, env=env)

    rax = model_build / "models" / family / f"{family}.rax"
    if not rax.is_file():
        raise SystemExit(f"error: expected .rax not found: {rax}")
    return rax


def _stage(
    dist_root: Path,
    arch: str,
    family: str,
    variant: str,
    rax: Path,
    meta: dict,
) -> None:
    out_dir = dist_root / arch / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rax, out_dir / f"{family}-{variant}.rax")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def _write_readme(dist_root: Path, family: str, arch: str, entries: list):
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


def _retry(label: str, fn, attempts: int = 5):
    """Retry transient Hugging Face 5xx errors with exponential backoff."""
    delay = 5
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as err:  # noqa: BLE001
            status = getattr(
                getattr(err, "response", None), "status_code", None
            )
            if status is None or not 500 <= status < 600 or attempt == attempts:
                raise
            print(
                f"[publish] {label}: transient HTTP {status}; "
                f"retry {attempt}/{attempts - 1} in {delay}s",
                file=sys.stderr,
            )
            time.sleep(delay)
            delay *= 2


def _publish_hf(repo_id: str, folder: Path, tag: str) -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    )
    if not token:
        raise SystemExit("error: HF_TOKEN is not set")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    _retry(
        "create_repo",
        lambda: api.create_repo(repo_id=repo_id, exist_ok=True),
    )
    refs = _retry("list_repo_refs", lambda: api.list_repo_refs(repo_id=repo_id))
    if tag in {ref.name for ref in refs.tags}:
        print(f"[publish] Tag {tag} already exists on {repo_id}; skipping")
        return
    _retry(
        "upload_folder",
        lambda: api.upload_folder(
            repo_id=repo_id,
            folder_path=str(folder),
            commit_message=f"Publish {tag}",
        ),
    )
    _retry(
        "create_tag",
        lambda: api.create_tag(repo_id=repo_id, tag=tag, exist_ok=True),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tag",
        required=True,
        help="Release tag/ref that drives the version (e.g. release/v0.0.6)",
    )
    ap.add_argument(
        "--registry", type=Path, default=Path("models/publish.json")
    )
    ap.add_argument("--source-dir", type=Path, default=None)
    ap.add_argument(
        "--cache-root",
        type=Path,
        default=Path.home() / "buddy-rax-cache",
        help="Persistent build cache root (default: ~/buddy-rax-cache)",
    )
    ap.add_argument("--python", default=sys.executable)
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
        "--continue-on-error",
        action="store_true",
        help="Keep building the other models when one fails; exit non-zero at the end.",
    )
    args = ap.parse_args()

    version = _tag_to_version(args.tag)
    hf_tag = f"v{version}"

    repo = (
        args.source_dir.resolve()
        if args.source_dir is not None
        else _repo_root()
    )
    registry = (
        args.registry if args.registry.is_absolute() else repo / args.registry
    )
    targets = _load_targets(registry)

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
        _build_deps(repo, paths, args.python, jobs)
        if args.deps_stamp is not None:
            args.deps_stamp.parent.mkdir(parents=True, exist_ok=True)
            args.deps_stamp.write_text("ready\n")

    failures: list[str] = []
    for target in targets:
        family = target["family"]
        try:
            arch = target["arch"]
            local_model = _download_snapshot(target.get("snapshot"), cache_root)
            family_dist = dist_root / family
            entries = []
            for spec_entry in target["variants"]:
                variant = spec_entry["variant"]
                spec = spec_entry["spec"]
                print(f"[build] {family}/{variant} ({arch}) version={version}")
                rax = _build_variant(
                    repo,
                    paths,
                    args.python,
                    jobs,
                    family,
                    variant,
                    version,
                    local_model,
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
                    "size_bytes": rax.stat().st_size,
                    "sha256": _sha256(rax),
                }
                _stage(family_dist, arch, family, variant, rax, meta)
                entries.append({"variant": variant})

            _write_readme(family_dist, family, arch, entries)
            _publish_hf(target["repo"], family_dist, hf_tag)
        except Exception as exc:  # noqa: BLE001
            if not args.continue_on_error:
                raise
            failures.append(family)
            print(f"[error] {family}: {exc}", file=sys.stderr)

    if failures:
        print(f"[error] failed targets: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
