#!/usr/bin/env python3
# ===- BuddyServerStartupFailureTest.py - Startup failure checks --------===//
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

import shutil
import subprocess
import sys
from pathlib import Path


def expect_failure(server: str, rax: Path, needle: str) -> None:
    result = subprocess.run(
        [server, "--model", str(rax), "--port", "18993"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode != 0, f"server unexpectedly started for {rax}"
    assert needle in result.stderr, result.stderr


def plugin_case(
    server: str,
    source_rax: Path,
    plugin: Path,
    work_dir: Path,
    name: str,
    needle: str,
) -> None:
    case_dir = work_dir / name
    if case_dir.exists():
        shutil.rmtree(case_dir)
    shutil.copytree(source_rax.parent, case_dir)
    shutil.copy2(plugin, case_dir / "transcription.so")
    expect_failure(server, case_dir / source_rax.name, needle)


def main() -> int:
    if len(sys.argv) != 7:
        raise SystemExit(
            "usage: test.py SERVER FILE_RAX NO_MODEL_RAX "
            "MISMATCH_PLUGIN MISSING_DESCRIPTOR_PLUGIN WORK_DIR"
        )
    server, file_rax, no_model_rax, mismatch, missing, work_dir = sys.argv[1:]
    source_rax = Path(file_rax)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    expect_failure(server, Path(no_model_rax), "missing model_library")
    plugin_case(
        server,
        source_rax,
        Path(missing),
        work,
        "missing-descriptor",
        "is missing buddy_server_model_descriptor_v1",
    )
    plugin_case(
        server,
        source_rax,
        Path(mismatch),
        work,
        "factory-mismatch",
        "legacy server ABI; please use a .rax package produced with the latest",
    )

    print("buddy-server startup failure tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
