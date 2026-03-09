# ===- __init__.py -------------------------------------------------------------
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
# ===---------------------------------------------------------------------------
#
# Console entry points that call the bundled Buddy executables.
#
# ===---------------------------------------------------------------------------

from __future__ import annotations

import subprocess
import sys
from importlib import resources


def _exe_resource(exe_name: str):
    return resources.files(__package__) / "bin" / exe_name


def _run(exe_name: str) -> int:
    exe_resource = _exe_resource(exe_name)
    with resources.as_file(exe_resource) as exe:
        if not exe.exists():
            raise SystemExit(
                f"Bundled executable '{exe_name}' is missing in the wheel."
            )
        return subprocess.call([str(exe), *sys.argv[1:]])


def buddy_opt() -> int:
    return _run("buddy-opt")


def buddy_translate() -> int:
    return _run("buddy-translate")


def buddy_llc() -> int:
    return _run("buddy-llc")


def buddy_lsp_server() -> int:
    return _run("buddy-lsp-server")


def buddy_frontendgen() -> int:
    return _run("buddy-frontendgen")


def buddy_audio_container_test() -> int:
    return _run("buddy-audio-container-test")


def buddy_text_container_test() -> int:
    return _run("buddy-text-container-test")


def buddy_container_test() -> int:
    return _run("buddy-container-test")
