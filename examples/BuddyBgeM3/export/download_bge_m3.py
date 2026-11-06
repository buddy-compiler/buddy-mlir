#!/usr/bin/env python3
# ===- download_bge_m3.py - Download a BAAI/bge-m3 snapshot ---------------===//
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
# Downloads the Hugging Face BAAI/bge-m3 snapshot to a local directory.
# HF_ENDPOINT may be set to use a mirror (e.g. https://hf-mirror.com).
#
# Usage:
#   python3 download_bge_m3.py [--out-dir /path/to/bge-m3]
import argparse
import os

from huggingface_hub import snapshot_download


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        default=os.getenv("LOCAL_BGE_M3", "./bge-m3"),
        help="destination directory (default: $LOCAL_BGE_M3 or ./bge-m3)",
    )
    args = ap.parse_args()
    snapshot_download("BAAI/bge-m3", local_dir=args.out_dir)
    print(f"[download] BAAI/bge-m3 -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

