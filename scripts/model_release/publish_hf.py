#!/usr/bin/env python3
# ===- publish_hf.py - Upload a staged model folder and tag the HF repo ---===//
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
# Usage:
#   HF_TOKEN=... python scripts/model_release/publish_hf.py \
#       --repo RuyiAI/buddy-whisper \
#       --folder dist/riscv64/whisper \
#       --path-in-repo riscv64 \
#       --tag v0.0.6 \
#       --commit-message "Publish whisper for v0.0.6"
#
# The upload is additive (existing files are overwritten in place) and the
# tag is created only if it does not already exist, so re-running the workflow
# is safe.
#
# ===----------------------------------------------------------------------===//

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _retry(label: str, fn, attempts: int = 5):
    """Retry transient Hugging Face 5xx errors with exponential backoff."""
    delay = 5
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as err:
            status = getattr(
                getattr(err, "response", None), "status_code", None
            )
            transient = status is not None and 500 <= status < 600
            if not transient or attempt == attempts:
                raise
            print(
                f"[publish_hf] {label}: transient HTTP {status}; "
                f"retry {attempt}/{attempts - 1} in {delay}s",
                file=sys.stderr,
            )
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"{label}: unreachable")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="HF repo id, e.g. org/name")
    ap.add_argument("--folder", required=True, type=Path, help="Local folder")
    ap.add_argument(
        "--path-in-repo",
        default="",
        help="Destination directory inside the HF repo (default: repo root)",
    )
    ap.add_argument("--tag", required=True, help="Version tag, e.g. v0.0.6")
    ap.add_argument("--commit-message", default="")
    ap.add_argument(
        "--repo-type",
        default="model",
        choices=("model", "dataset", "space"),
    )
    ap.add_argument(
        "--revision",
        default="main",
        help="Branch to upload to (default: main)",
    )
    ap.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private if it does not exist",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if not args.folder.is_dir():
        print(f"error: folder not found: {args.folder}", file=sys.stderr)
        return 1

    token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    )
    if not token:
        print("error: HF_TOKEN is not set", file=sys.stderr)
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print(
            "error: huggingface_hub is required (pip install huggingface_hub)",
            file=sys.stderr,
        )
        return 1

    api = HfApi(token=token)

    print(f"[publish_hf] Ensuring repo exists: {args.repo}")
    _retry(
        "create_repo",
        lambda: api.create_repo(
            repo_id=args.repo,
            repo_type=args.repo_type,
            private=args.private,
            exist_ok=True,
        ),
    )

    refs = _retry(
        "list_repo_refs",
        lambda: api.list_repo_refs(repo_id=args.repo, repo_type=args.repo_type),
    )
    if args.tag in {ref.name for ref in refs.tags}:
        print(
            f"[publish_hf] Tag {args.tag} already exists; nothing to do "
            f"(re-run with a new version to republish)."
        )
        return 0

    commit_message = args.commit_message or f"Publish {args.tag}"
    print(
        f"[publish_hf] Uploading {args.folder} -> "
        f"{args.repo}:{args.path_in_repo or '/'}"
    )
    _retry(
        "upload_folder",
        lambda: api.upload_folder(
            repo_id=args.repo,
            repo_type=args.repo_type,
            folder_path=str(args.folder),
            path_in_repo=args.path_in_repo or None,
            revision=args.revision,
            commit_message=commit_message,
        ),
    )

    print(f"[publish_hf] Creating tag {args.tag} on {args.repo}")
    _retry(
        "create_tag",
        lambda: api.create_tag(
            repo_id=args.repo,
            repo_type=args.repo_type,
            tag=args.tag,
            revision=args.revision,
            exist_ok=True,
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
