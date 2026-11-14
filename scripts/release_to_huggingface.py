#!/usr/bin/env python3
# ===- release_to_huggingface.py - Build riscv64 .rax and push to HF --===//
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
# Cross-compiles one riscv64 .rax per --spec and uploads each to its own
# Hugging Face repo, tagged with the version derived from --tag.
#
# The specs are the only input; everything else is derived from them:
#
#   make target     riscv-model-<family>-<variant>
#   upstream model  <spec>.hf_model_path
#   HF repo         <namespace>/buddy-<family>-<variant>
#
# All build trees live under --cache-root (the Makefile's CACHE_ROOT), so
# repeated runs on self-hosted runners reuse them.
#
# ===----------------------------------------------------------------------===//

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import HfHubHTTPError

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Target:
    family: str
    variant: str
    snapshot: str
    repo: str

    @property
    def stem(self):
        return f"{self.family}-{self.variant}"


def _make(cache_root, *args, **env):
    cmd = [
        "make",
        *args,
        f"CACHE_ROOT={cache_root}",
        f"PYTHON={sys.executable}",
        f"PARALLEL={os.cpu_count() or 1}",
    ]
    print("[run]", *cmd, flush=True)
    subprocess.run(cmd, cwd=ROOT, env=os.environ | env, check=True)


def _version(tag):
    """`release/v0.0.6`, `nightly/v0.0.9`, a SHA or a branch -> a version."""
    tag = tag.strip()
    for prefix in ("refs/tags/", "release/", "nightly/", "v"):
        tag = tag.removeprefix(prefix)
    if re.fullmatch(r"[0-9a-f]{40}", tag):
        tag = tag[:7]
    tag = re.sub(r"[^A-Za-z0-9._-]+", "-", tag).strip("-.")
    if not tag:
        raise SystemExit(f"error: cannot derive a version from {tag!r}")
    return tag


def _snapshot(repo_id, cache_root):
    local = cache_root / "snapshots" / repo_id.replace("/", "--")
    if not local.exists():
        print(f"[snapshot] {repo_id} -> {local}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local),
            endpoint=os.environ.get("HF_DOWNLOAD_ENDPOINT"),
        )
    return local


def _build(target, cache_root, version, local_model):
    # RISCV_MODEL_BUILD defaults to $(CACHE_ROOT)/build-riscv in the Makefile.
    build_dir = cache_root / "build-riscv" / target.stem

    # The Makefiles generator mishandles the arg0.data byproduct, so always use
    # Ninja and drop a stale tree configured with another generator.
    cache = build_dir / "CMakeCache.txt"
    if (
        cache.exists()
        and "CMAKE_GENERATOR:INTERNAL=Ninja" not in cache.read_text()
    ):
        shutil.rmtree(build_dir)

    args = [
        "-o",
        "riscv",
        f"riscv-model-{target.stem}",
        f"BUDDY_PACKAGE_VERSION={version}",
    ]
    if local_model:
        args.append(f"RISCV_MODEL_LOCAL={local_model}")
    _make(cache_root, *args, CMAKE_GENERATOR="Ninja")

    rax = build_dir / "models" / target.family / f"{target.family}.rax"
    if not rax.is_file():
        raise SystemExit(f"error: {target.stem}: expected {rax}")
    return rax


def _board(variant):
    """Human description of the target the Makefile builds this variant for."""
    if variant.startswith("k3_"):
        return "SpacemiT K3 A100 (VLEN=1024, Zfh/Zvfh)"
    if variant.startswith("a210_"):
        return "Zhihe A210 (VLEN=128, Zfh/Zvfh)"
    return "generic riscv64 (any VLEN, no Zfh/Zvfh)"


def _stage(target, rax, version, commit):
    folder = ROOT / "dist" / target.stem
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rax, folder / f"{target.stem}.rax")
    source = commit or "unknown"
    # install_buddy_cli.sh only resolves release/v* tags, so pin the matching
    # version for a release; commit and nightly builds have no such binary.
    release = ".dev" not in version and not re.fullmatch(
        r"[0-9a-f]{7}", version
    )
    cli = f"bash -s -- {version}" if release else "bash"
    model = f"{target.stem}/{target.stem}.rax"
    if target.variant.startswith("k3_"):
        # The K3 puts a process' threads on the A100 cores only after this.
        run = (
            "sh -c 'echo 0 > /proc/set_ai_thread && exec "
            f'buddy-cli --model {model} --prompt "hello"\''
        )
    else:
        run = f'buddy-cli --model {model} --prompt "hello"'
    (folder / "README.md").write_text(
        f"""# {target.repo}

Cross-compiled RISC-V (riscv64) `.rax` package for the
[`buddy-cli`](https://github.com/buddy-compiler/buddy-mlir) runtime.

- version: `{version}`
- target: {_board(target.variant)}
- upstream model: [{target.snapshot}](https://huggingface.co/{target.snapshot})
- source: `{source}`

## 1. Download the model

```bash
pip install -U "huggingface_hub[cli]"
hf download {target.repo} --local-dir {target.stem}
```

## 2. Install buddy-cli

```bash
curl -fsSL https://raw.githubusercontent.com/buddy-compiler/buddy-mlir/main/scripts/install_buddy_cli.sh | {cli}
export PATH="$HOME/.local/bin:$PATH"
```

## 3. Run

```bash
{run}
```
"""
    )
    return folder


def _retry(fn, *args, attempts=5, **kwargs):
    """Retry transient Hugging Face 5xx errors with exponential backoff."""
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except HfHubHTTPError as err:
            if err.response.status_code < 500 or attempt == attempts - 1:
                raise
            delay = 5 * 2**attempt
            print(
                f"[publish] transient HTTP {err.response.status_code}; "
                f"retry in {delay}s",
                file=sys.stderr,
            )
            time.sleep(delay)


def _publish(repo_id, folder, tag):
    # HfApi reads the token from HF_TOKEN / HUGGING_FACE_HUB_TOKEN.
    api = HfApi()
    _retry(api.create_repo, repo_id=repo_id, exist_ok=True)
    refs = _retry(api.list_repo_refs, repo_id=repo_id)
    if tag in {ref.name for ref in refs.tags}:
        print(f"[publish] {repo_id}@{tag} already exists; skipping")
        return
    _retry(
        api.upload_folder,
        repo_id=repo_id,
        folder_path=str(folder),
        commit_message=f"Publish {tag}",
    )
    _retry(api.create_tag, repo_id=repo_id, tag=tag, exist_ok=True)


def _resolve_specs(raw_specs, namespace):
    """Derive the build target, upstream model and HF repo from each spec."""
    targets = []
    for raw in raw_specs:
        spec = raw if raw.is_absolute() else ROOT / raw
        if not spec.is_file():
            raise SystemExit(f"error: spec not found: {raw}")
        snapshot = json.loads(spec.read_text()).get("hf_model_path", "")
        if not snapshot or os.path.isabs(snapshot):
            raise SystemExit(
                f"error: {raw} needs a public Hugging Face hf_model_path"
            )
        family, variant = spec.parent.parent.name, spec.stem
        targets.append(
            Target(
                family=family,
                variant=variant,
                snapshot=snapshot,
                repo=(
                    f"{namespace}/buddy-{family.replace('_', '-')}"
                    f"-{variant.replace('_', '-')}"
                ),
            )
        )
    return targets


def main():
    ap = argparse.ArgumentParser(
        description="Cross-compile riscv64 .rax packages and publish them to Hugging Face."
    )
    ap.add_argument("--tag", required=True, help="Revision to package")
    ap.add_argument(
        "--namespace", required=True, help="Hugging Face namespace/org"
    )
    ap.add_argument(
        "--spec",
        required=True,
        action="append",
        type=Path,
        metavar="PATH",
        help="models/<family>/specs/<variant>.json (repeatable)",
    )
    ap.add_argument(
        "--cache-root",
        type=Path,
        default=Path.home() / "buddy-rax-cache",
        help="Makefile CACHE_ROOT (default: ~/buddy-rax-cache)",
    )
    ap.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep going when a spec fails; exit non-zero at the end",
    )
    args = ap.parse_args()

    version = _version(args.tag)
    cache_root = args.cache_root.expanduser().resolve()
    commit = os.environ.get("GITHUB_SHA", "")

    _make(cache_root, "riscv")

    failed = []
    for target in _resolve_specs(args.spec, args.namespace):
        try:
            print(f"[build] {target.stem} -> {target.repo} (v{version})")
            rax = _build(
                target,
                cache_root,
                version,
                _snapshot(target.snapshot, cache_root),
            )
            _publish(
                target.repo, _stage(target, rax, version, commit), f"v{version}"
            )
        except Exception as err:  # noqa: BLE001
            if not args.continue_on_error:
                raise
            failed.append(target.stem)
            print(f"[error] {target.stem}: {err}", file=sys.stderr)

    if failed:
        print(f"[error] failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
