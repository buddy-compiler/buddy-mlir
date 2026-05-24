# ===- terminal.py -------------------------------------------------------------
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
# Shared terminal UI helpers for Buddy Python tools.
#
# ===---------------------------------------------------------------------------

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ORANGE = "1;38;5;214"
GRAY = "90"
RED = "31;1"


class TerminalUI:
    def __init__(self, *, stream=sys.stdout):
        self.stream = stream
        self.use_color = stream.isatty() and os.environ.get("NO_COLOR") is None
        self.use_animation = (
            stream.isatty() and os.environ.get("NO_ANIMATION") is None
        )

    def color(self, text: str, code: str) -> str:
        if not self.use_color:
            return text
        return f"\033[{code}m{text}\033[0m"

    def section(self, title: str) -> None:
        print(file=self.stream)
        print(self.color(f"== {title} ==", ORANGE), file=self.stream)

    def rule(self, width: int) -> str:
        return self.color("-" * width, "2")

    def format_seconds(self, seconds: float) -> str:
        if seconds < 1e-3:
            return f"{seconds * 1e6:.3f} us"
        if seconds < 1:
            return f"{seconds * 1e3:.3f} ms"
        return f"{seconds:.3f} s"

    def spinner(self, phase: str, label: str) -> Spinner:
        return Spinner(self, phase, label)


class Spinner:
    def __init__(self, ui: TerminalUI, phase: str, label: str):
        self.ui = ui
        self.phase = phase
        self.label = label
        self.start = time.perf_counter()
        self.done = threading.Event()
        self.thread: threading.Thread | None = None

    def start_animation(self) -> None:
        if not self.ui.use_animation:
            print(f"[{self.phase}] {self.label} ...", flush=True)
            return
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()

    def stop(self, status: str, elapsed: float) -> None:
        if not self.ui.use_animation:
            print(
                f"[{self.phase}] {self.label} {status} in "
                f"{self.ui.format_seconds(elapsed)}"
            )
            return
        self.done.set()
        if self.thread:
            self.thread.join()
        self.ui.stream.write("\r\033[K")
        status_code = ORANGE if status == "done" else RED
        print(
            f"{self.ui.color('>', ORANGE)} "
            f"{self.ui.color(f'[{self.phase}]', GRAY)} {self.label} "
            f"{self.ui.color(status, status_code)} "
            f"{self.ui.color(f'in {self.ui.format_seconds(elapsed)}', GRAY)}",
            file=self.ui.stream,
        )

    def _animate(self) -> None:
        frames = ["-", "\\", "|", "/"]
        idx = 0
        while not self.done.wait(0.12):
            elapsed = self.ui.format_seconds(time.perf_counter() - self.start)
            frame = self.ui.color(frames[idx % len(frames)], ORANGE)
            status = self.ui.color(f"[{self.phase}] {self.label}", GRAY)
            self.ui.stream.write(
                f"\r\033[K{frame} {status} {self.ui.color(elapsed, GRAY)}"
            )
            self.ui.stream.flush()
            idx += 1


def run_with_status(
    cmd: list[str],
    *,
    cwd: Path,
    ui: TerminalUI,
    stdout: Path | None = None,
    label: str | None = None,
    phase: str = "build",
    capture_stdout: bool = False,
    verbose: bool = False,
) -> subprocess.CompletedProcess[bytes]:
    if verbose:
        print("+ " + " ".join(cmd))
    spinner = ui.spinner(phase, label) if label else None
    if spinner:
        spinner.start_animation()

    start = time.perf_counter()
    stdout_file = None
    if stdout:
        stdout.parent.mkdir(parents=True, exist_ok=True)
        stdout_file = stdout.open("wb")
        stdout_target = stdout_file
    else:
        stdout_target = subprocess.PIPE if capture_stdout or label else None

    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=stdout_target,
        stderr=subprocess.PIPE,
    )
    captured_stdout, captured_stderr = process.communicate()
    if stdout_file:
        stdout_file.close()

    elapsed = time.perf_counter() - start
    if process.returncode != 0:
        if spinner:
            spinner.stop("failed", elapsed)
        stdout_text = (
            captured_stdout.decode(errors="replace")
            if captured_stdout is not None
            else ""
        )
        stderr = (
            captured_stderr.decode(errors="replace")
            if captured_stderr is not None
            else ""
        )
        output = f"{stdout_text}\n{stderr}".strip()
        raise RuntimeError(
            "command failed with exit code "
            f"{process.returncode}: {' '.join(cmd)}\n{output}"
        )

    if spinner:
        spinner.stop("done", elapsed)
    return subprocess.CompletedProcess(
        cmd,
        process.returncode,
        stdout=captured_stdout or b"",
        stderr=captured_stderr or b"",
    )
