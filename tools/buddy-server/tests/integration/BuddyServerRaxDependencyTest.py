#!/usr/bin/env python3
# ===- BuddyServerRaxDependencyTest.py - RAX dependency test ------------===//
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

import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: test.py SERVER MODEL_RAX")

    server, model_rax = sys.argv[1:]
    result = subprocess.run(
        [server, "--model", str(Path(model_rax)), "--port", "18994"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode != 0, "server unexpectedly started"
    assert "server model descriptor has null operations" in result.stderr, (
        result.stderr
    )
    assert "returned a null descriptor" not in result.stderr, result.stderr

    print("buddy-server RAX dependency test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
