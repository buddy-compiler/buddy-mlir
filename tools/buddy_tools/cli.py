"""Console entry points that call the bundled Buddy executables."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from . import bin_dir


def _run(exe_name: str) -> int:
    exe = Path(bin_dir() / exe_name)
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
