#!/usr/bin/env python3
"""Remote half of fpga_run.py for the NR FPGA server.

Runs inside an approved workdir: claims the board, opens UART, loads the
image via make uv_runN, captures serial output, verifies DDR readback, and
writes result.json. Supports --detach (start once), --background (worker),
and --relay (resume UART from a byte offset without reloading hardware).
"""

# ===- fpga_remote.py ----------------------------------------------------------
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

import argparse
import errno
import fcntl
import hashlib
import json
import os
import pty
import re
import select
import shutil
import signal
import subprocess
import sys
import termios
import time
from pathlib import Path

# Match UVHS / make fatal lines in the platform log stream.
ERROR = re.compile(
    rb"(?:^|\n)(?:[^\r\n]*\]\s*(?:ERROR|FATAL):|(?:ERROR|FATAL):|"
    rb"Error: in running command:|make(?:\[\d+\])?: \*\*\*|"
    rb"[ \t]*Total (?:ERROR|FATAL):[ \t]*[1-9])",
    re.MULTILINE,
)
# Match bare-metal UART failure markers from hello / kernel dumps.
UART_FAILURE = re.compile(
    rb"TRAP mcause=|"
    rb"verify[^\r\n]*:\s*(?:FAIL|mismatches=[1-9])|"
    rb"\[nr\][^\r\n]*FAIL"
)


def log(message):
    """Write a status line to stderr (UART data stays on stdout)."""
    print("[fpga_run] " + message, file=sys.stderr, flush=True)


def digest(path):
    """Return the SHA-256 hex digest of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def within(path, root):
    """True if path resolves to root or a descendant of root."""
    return path.resolve() == root or root in path.resolve().parents


def check_paths(root, run_dir):
    """Refuse run directories or platform outputs that escape the workdir."""
    if (
        root.resolve() != root
        or not within(run_dir, root)
        or run_dir.is_symlink()
    ):
        raise RuntimeError("run directory must be inside the approved workdir")
    for name in (
        "cmd",
        "test",
        "apsRun",
        "uv_run.log",
        "uv_stacktrace.txt",
        ".socket_uv_shell",
        ".uvshell_workdir_lock",
        ".fpga_run.lock",
    ):
        if not within(root / name, root):
            raise RuntimeError(
                f"refusing output path that points outside workdir: {name}"
            )


def check_processes(root, fpga, device):
    """Refuse to start if another UVHS session or UART client owns the board.

    Scans /proc for uvhs2_shell / uv_shell_exec targeting this workdir or
    hw_runN.tcl, and for serial tools holding /dev/FPGAN.
    """
    actual = os.path.realpath(device)
    for proc in Path("/proc").glob("[0-9]*"):
        if int(proc.name) == os.getpid():
            continue
        try:
            argv = proc.joinpath("cmdline").read_bytes().split(b"\0")
            executable = os.path.basename(os.fsdecode(argv[0]))
            args = [os.fsdecode(a) for a in argv if a]
            if executable in ("uvhs2_shell", "uv_shell_exec") or any(
                os.path.basename(a) == "uvhs2_shell" for a in args[:2]
            ):
                same_dir = proc.joinpath("cwd").resolve() == root
                same_board = any(
                    Path(a).name == f"hw_run{fpga}.tcl" for a in args
                )
                if same_dir or same_board:
                    raise RuntimeError(
                        f"existing UVHS session PID {proc.name}; close it first"
                    )
            if executable in (
                "minicom",
                "picocom",
                "screen",
                "cat",
                "socat",
            ) and (device in args or actual in args):
                raise RuntimeError(f"{device} is in use by PID {proc.name}")
            for fd in proc.joinpath("fd").iterdir():
                try:
                    if os.readlink(fd) == actual:
                        raise RuntimeError(
                            f"{device} is open in PID {proc.name}"
                        )
                except (FileNotFoundError, PermissionError):
                    pass
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue


def board_available(output, fpga):
    """Parse UVHS 'query -fpgas -all'; raise if FPGAN is booked or owned."""
    board, slot = divmod(fpga, 4)
    pattern = (
        rf"^\S+\s+B{board}\s+F{slot}\s+\S+\s+link\s+(?:up|down)\s+"
        r"(?:(\S+)\s+)?(true|false)\s*$"
    )
    row = re.search(pattern, output.decode(errors="replace"), re.MULTILINE)
    if not row:
        raise RuntimeError(
            f"could not establish FPGA{fpga} availability from UVHS query"
        )
    if row[1] or row[2] == "true":
        raise RuntimeError(
            f"FPGA{fpga} is occupied/booked (owner: {row[1] or 'reserved'})"
        )


def shutdown(process, master=None):
    """Stop a PTY-backed UVHS/make session within the process group we created.

    Sends 'exit' on the master PTY when possible, then SIGTERM/SIGKILL to the
    process group. Never signals unrelated processes.
    """
    if process is None:
        return
    if process.poll() is None and master is not None:
        try:
            os.write(master, b"exit\n")
        except OSError:
            pass
    until = time.monotonic() + 12
    while process.poll() is None and time.monotonic() < until:
        if master is not None:
            try:
                if select.select([master], [], [], 0.1)[0]:
                    os.read(master, 65536)
            except OSError:
                pass
        else:
            time.sleep(0.1)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            continue
        time.sleep(0.1)
    process.wait()


def preflight(root, run_dir, fpga, stopped):
    """Validate hw_runN.tcl mapping and confirm FPGAN is free via UVHS query.

    Writes query.tcl under run_dir, runs uvhs2_shell, then board_available.
    stopped() is polled so Ctrl-C / remote stop can abort the wait.
    """
    executable = shutil.which("uvhs2_shell")
    if not executable:
        raise RuntimeError("uvhs2_shell is not in the SSH environment PATH")
    script = root / "user_script" / "hw_run" / f"hw_run{fpga}.tcl"
    text = script.read_text()
    targets = re.findall(
        r"^\s*load_db\s+-db\s+\S+\s+-to\s+\{?(b\d+\.f\d+)\}?", text, re.M | re.I
    )
    if targets != [f"b{fpga // 4}.f{fpga % 4}"]:
        raise RuntimeError(f"unexpected board mapping in {script}")
    query = run_dir / "query.tcl"
    query.write_text("query -fpgas -all\nexit\n")
    process = None
    with (run_dir / "preflight.log").open("wb") as output:
        try:
            process = subprocess.Popen(
                [
                    executable,
                    "-t",
                    "runtime",
                    "-d",
                    "V1",
                    "-workdir",
                    "./",
                    "-script",
                    str(query),
                    "-bypass_vivado_version_check",
                ],
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + 45
            while process.poll() is None:
                if stopped():
                    raise InterruptedError("resource query interrupted")
                if time.monotonic() >= deadline:
                    raise RuntimeError("resource query timed out")
                time.sleep(0.1)
            if process.returncode:
                raise RuntimeError(
                    f"UVHS resource query exited {process.returncode}"
                )
        finally:
            shutdown(process)
    output = (run_dir / "preflight.log").read_bytes()
    if ERROR.search(output):
        raise RuntimeError("UVHS resource query failed; see preflight.log")
    board_available(output, fpga)


def verify_readback(run_dir, runtime_output, expected_sha):
    """Confirm the padded image and DDR .readback match the uploaded SHA-256.

    Checks UV_RUN_IMAGE is under run_dir, padding is zero after the raw bytes,
    and digest(image) == digest(readback). Returns the padded-image digest.
    """
    match = re.search(rb"UV_RUN_IMAGE=([^\r\n]+)", runtime_output)
    if not match:
        raise RuntimeError("platform did not report UV_RUN_IMAGE")
    image = Path(os.fsdecode(match[1])).resolve()
    if image.parent != run_dir:
        raise RuntimeError(
            "platform selected an image outside this run directory"
        )
    readback = Path(str(image) + ".readback")
    if readback.is_symlink() or not readback.is_file():
        raise RuntimeError("missing DDR readback")
    raw = run_dir / "image.bin"
    if digest(raw) != expected_sha:
        raise RuntimeError("uploaded image changed during this run")
    if image.stat().st_size % 64 or image.stat().st_size < raw.stat().st_size:
        raise RuntimeError("invalid padded image size")
    with raw.open("rb") as a, image.open("rb") as b:
        for block in iter(lambda: a.read(1024 * 1024), b""):
            if b.read(len(block)) != block:
                raise RuntimeError(
                    "padded image does not contain the uploaded binary"
                )
        for block in iter(lambda: b.read(1024 * 1024), b""):
            if any(block):
                raise RuntimeError("nonzero padding after uploaded binary")
    sha = digest(image)
    if digest(readback) != sha:
        raise RuntimeError("DDR readback differs from the loaded image")
    return sha


def run(args, root=None, run_dir=None, device=None, control_fd=0):
    """Load FPGAN, capture UART, verify DDR, write result.json.

    Acquires .fpga_run.lock, opens /dev/FPGAN at the requested baud (8N1),
    runs make uv_runN under a PTY, streams UART to stdout and uart.raw.log,
    then verifies readback. control_fd (default stdin) or a stop file can
    interrupt. Returns 0 / 1 / 130.
    """
    root = Path(root) if root is not None else Path.cwd()
    run_dir = Path(run_dir or Path(__file__).resolve().parent)
    device = device or f"/dev/FPGA{args.fpga}"
    os.chdir(root)
    check_paths(root, run_dir)
    result = {
        "status": "ERROR",
        "fpga": args.fpga,
        "sha256": args.sha256,
        "capture_seconds": args.capture_seconds,
        "baud": args.baud,
    }
    stopped_flag = [False]
    old_signals = {}

    def stop(signum, frame):
        # SIGINT/SIGTERM/SIGHUP: request a clean teardown via stopped().
        stopped_flag[0] = True

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        old_signals[sig] = signal.signal(sig, stop)

    def stopped():
        """True if a signal, stop file, or control_fd asked us to abort."""
        if getattr(args, "background", False) and (run_dir / "stop").exists():
            stopped_flag[0] = True
        if stopped_flag[0]:
            return True
        if control_fd is not None and select.select([control_fd], [], [], 0)[0]:
            os.read(control_fd, 4096)
            stopped_flag[0] = True
        return stopped_flag[0]

    lock = serial = master = None
    original = process = None
    status = 1
    try:
        # Exclusive workdir lock so two fpga_run sessions cannot collide.
        lock = os.open(
            root / ".fpga_run.lock",
            os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
            0o600,
        )
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as err:
            raise RuntimeError(
                "another fpga_run is using this platform workdir"
            ) from err
        raw = run_dir / "image.bin"
        if (
            raw.is_symlink()
            or not raw.is_file()
            or raw.stat().st_size == 0
            or digest(raw) != args.sha256
        ):
            raise RuntimeError("uploaded binary SHA256 mismatch or empty input")
        raw_size = raw.stat().st_size
        readback_budget = 2 * max(
            1048576, ((raw_size + 1048575) // 1048576) * 1048576
        )
        available = shutil.disk_usage(str(root)).free
        if available < readback_budget + 64 * 1024 * 1024:
            raise RuntimeError(
                "insufficient remote disk space for DDR readbacks: "
                "required=%d available=%d"
                % (readback_budget + 64 * 1024 * 1024, available)
            )
        check_processes(root, args.fpga, device)
        preflight(root, run_dir, args.fpga, stopped)
        check_processes(root, args.fpga, device)
        if stopped():
            raise InterruptedError("stopped before loading")
        serial = os.open(device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        fcntl.ioctl(serial, termios.TIOCEXCL)
        original = termios.tcgetattr(serial)
        attrs = termios.tcgetattr(serial)
        attrs[0] = attrs[1] = attrs[3] = 0
        attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        attrs[4] = attrs[5] = getattr(termios, "B" + str(args.baud))
        attrs[6][termios.VMIN] = attrs[6][termios.VTIME] = 0
        termios.tcsetattr(serial, termios.TCSANOW, attrs)
        termios.tcflush(serial, termios.TCIFLUSH)
        log(f"UART ready: {device}, {args.baud}, 8N1, no flow control")
        master, slave = pty.openpty()
        env = os.environ.copy()
        for key in ("UV_RUN_PLAN_TCL", "UV_RUN_PREP_ONLY"):
            env.pop(key, None)
        env.update(
            UV_RUN_READBACK="1",
            UV_RUN_SYS_CLK_HZ="14745600",
            UV_RUN_ALIGN_BYTES="1048576",
            UV_RUN_MIN_BYTES="1048576",
            TMPDIR=str(run_dir),
        )
        try:
            process = subprocess.Popen(
                ["make", f"uv_run{args.fpga}", f"test={raw}"],
                cwd=root,
                env=env,
                stdin=slave,
                stdout=slave,
                stderr=slave,
                start_new_session=True,
            )
        finally:
            os.close(slave)
        output = bytearray()
        uart_total = 0
        started = None
        exit_sent = None
        deadline = time.monotonic() + args.startup_timeout
        uart_tail = b""
        failed_uart = False
        heartbeat = time.monotonic()
        # Parenthesized multi-with is 3.10+; FPGA servers often ship 3.8/3.9.
        with (run_dir / "uvhs.log").open("wb") as uvlog, (
            run_dir / "uart.raw.log"
        ).open("wb") as uart:
            while True:
                if stopped():
                    raise InterruptedError(
                        "interrupted; releasing this session"
                    )
                now = time.monotonic()
                if now - heartbeat >= 30:
                    elapsed = int(now - started) if started is not None else 0
                    log(
                        f"FPGA{args.fpga} session active; elapsed={elapsed}s, "
                        f"UART={uart_total} bytes"
                    )
                    heartbeat = now
                if started is None and now >= deadline:
                    raise RuntimeError("FPGA startup timed out; see uvhs.log")
                if (
                    started is not None
                    and exit_sent is None
                    and now - started >= args.capture_seconds
                ):
                    os.write(master, b"exit\n")
                    exit_sent = now
                if exit_sent is not None and now - exit_sent > 20:
                    raise RuntimeError("UVHS did not exit after capture")
                ready, _, _ = select.select([master, serial], [], [], 0.1)
                for fd in ready:
                    try:
                        data = os.read(fd, 65536)
                    except OSError as e:
                        if fd == master and e.errno == errno.EIO:
                            data = b""
                        else:
                            raise
                    if fd == serial:
                        if not data:
                            raise RuntimeError("UART disconnected")
                        uart.write(data)
                        uart.flush()
                        uart_total += len(data)
                        uart_tail = (uart_tail + data)[-8192:]
                        failed_uart |= bool(UART_FAILURE.search(uart_tail))
                        sys.stdout.buffer.write(data)
                        sys.stdout.buffer.flush()
                    else:
                        uvlog.write(data)
                        uvlog.flush()
                        output.extend(data)
                        if ERROR.search(output):
                            raise RuntimeError(
                                "platform reported ERROR/FATAL; see uvhs.log"
                            )
                        if started is None and b"hspRun>" in output:
                            if (
                                b"UV_RUN_IMAGE=" not in output
                                or b"reset -name cpu_reset -value 0 success"
                                not in output
                            ):
                                raise RuntimeError(
                                    "UVHS reached its prompt without "
                                    "loading/starting the image"
                                )
                            started = time.monotonic()
                            log(
                                f"FPGA{args.fpga} started; capturing for "
                                f"{args.capture_seconds} seconds"
                            )
                if process.poll() is not None:
                    if (
                        process.returncode
                        or started is None
                        or exit_sent is None
                    ):
                        raise RuntimeError(
                            f"platform exited unexpectedly "
                            f"({process.returncode}); see uvhs.log"
                        )
                    break
        result["padded_sha256"] = verify_readback(
            run_dir, bytes(output), args.sha256
        )
        result["ddr_readback_matches"] = True
        result["uart_bytes"] = uart_total
        if failed_uart:
            raise RuntimeError(
                "program reported a verification failure or trap"
            )
        if not uart_total:
            raise RuntimeError("no UART output during the capture window")
        result["status"] = "OK"
        status = 0
        log(f"Finished: {uart_total} UART bytes; DDR readback matches")
    except InterruptedError as e:
        result.update(status="INTERRUPTED", error=str(e))
        status = 130
        log(str(e))
    except (OSError, RuntimeError, ValueError) as e:
        result["error"] = str(e)
        log("ERROR: " + str(e))
    finally:
        cleanup_errors = []

        def cleanup(action):
            try:
                action()
            except (OSError, subprocess.SubprocessError) as error:
                cleanup_errors.append(str(error))

        cleanup(lambda: shutdown(process, master))
        if master is not None:
            cleanup(lambda: os.close(master))
        if serial is not None:
            if original is not None:
                cleanup(
                    lambda: termios.tcsetattr(serial, termios.TCSANOW, original)
                )
            cleanup(lambda: fcntl.ioctl(serial, termios.TIOCNXCL))
            cleanup(lambda: os.close(serial))
        if lock is not None:
            cleanup(lambda: os.close(lock))
        for sig, handler in old_signals.items():
            signal.signal(sig, handler)
        if cleanup_errors:
            result["cleanup_errors"] = cleanup_errors
            result.update(
                status="ERROR",
                error=result.get("error", "session cleanup failed"),
            )
            status = 1
        pending = run_dir / "result.json.tmp"
        pending.write_text(json.dumps(result, indent=2) + "\n")
        pending.replace(run_dir / "result.json")
    return status


def detached_start(args):
    """Start the background worker at most once for this run directory.

    Uses start.lock + started.json so a lost SSH acknowledgement does not
    launch a second hardware session. Returns 0 if already started or newly
    spawned.
    """
    root = Path.cwd()
    directory = Path(__file__).resolve().parent
    check_paths(root, directory)
    fd = os.open(
        directory / "start.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        state = directory / "started.json"
        if state.exists():
            recorded = json.loads(state.read_text())
            if (
                recorded["sha256"] != args.sha256
                or recorded["fpga"] != args.fpga
            ):
                raise RuntimeError(
                    "run identity mismatch; refusing another hardware launch"
                )
            return 0
        pending = directory / "started.json.tmp"
        pending.write_text(
            json.dumps({"sha256": args.sha256, "fpga": args.fpga, "pid": None})
        )
        pending.replace(state)
        command = [
            sys.executable,
            "-u",
            "-B",
            str(Path(__file__).resolve()),
            "--background",
            "--fpga",
            str(args.fpga),
            "--sha256",
            args.sha256,
            "--capture-seconds",
            str(args.capture_seconds),
            "--startup-timeout",
            str(args.startup_timeout),
            "--baud",
            str(args.baud),
        ]
        with (directory / "worker.log").open("ab", buffering=0) as output:
            process = subprocess.Popen(
                command,
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=output,
                start_new_session=True,
            )
        pending.write_text(
            json.dumps(
                {"sha256": args.sha256, "fpga": args.fpga, "pid": process.pid}
            )
        )
        pending.replace(state)
        return 0
    finally:
        os.close(fd)


def relay(offset):
    """Stream uart.raw.log from byte offset without touching FPGA hardware.

    Used by the local client after SSH drops. Exits when result.json appears,
    or raises if the worker died without writing a result.
    """
    directory = Path(__file__).resolve().parent
    uart = directory / "uart.raw.log"
    result_path = directory / "result.json"
    heartbeat = time.monotonic()
    empty_deadline = heartbeat + 30
    while True:
        if uart.exists():
            with uart.open("rb") as stream:
                stream.seek(offset)
                data = stream.read(65536)
            if data:
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
                offset += len(data)
                continue
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result["status"] != "OK":
                log(result.get("error", result["status"]))
            return (
                0
                if result["status"] == "OK"
                else 130
                if result["status"] == "INTERRUPTED"
                else 1
            )
        now = time.monotonic()
        state_path = directory / "started.json"
        if now > empty_deadline:
            state = (
                json.loads(state_path.read_text())
                if state_path.exists()
                else {}
            )
            pid_file = directory / "worker.pid"
            pid = (
                int(pid_file.read_text())
                if pid_file.exists()
                else state.get("pid")
            )
            proc = Path("/proc", str(pid), "cmdline")
            if (
                not pid
                or not proc.exists()
                or str(Path(__file__).resolve()).encode()
                not in proc.read_bytes()
            ):
                raise RuntimeError(
                    "worker exited without a result; run was not restarted"
                )
        if now - heartbeat >= 15:
            log(f"UART relay active; received={offset} bytes")
            heartbeat = now
        time.sleep(0.1)


def main():
    """CLI entry: --relay, --detach, or --background / foreground run()."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fpga", type=int, choices=range(8))
    parser.add_argument("--sha256")
    parser.add_argument("--capture-seconds", type=int)
    parser.add_argument("--startup-timeout", type=int)
    parser.add_argument("--baud", type=int)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--relay", type=int, metavar="UART_OFFSET")
    args = parser.parse_args()
    try:
        if args.relay is not None:
            if args.relay < 0:
                parser.error("UART offset cannot be negative")
            return relay(args.relay)
        if any(
            getattr(args, name) is None
            for name in (
                "fpga",
                "sha256",
                "capture_seconds",
                "startup_timeout",
                "baud",
            )
        ):
            parser.error("run options are required")
        if args.detach:
            return detached_start(args)
        if args.background:
            (Path(__file__).resolve().parent / "worker.pid").write_text(
                str(os.getpid())
            )
        return run(args, control_fd=None if args.background else 0)
    except (OSError, RuntimeError, ValueError) as error:
        log("ERROR: " + str(error))
        return 1


if __name__ == "__main__":
    sys.exit(main())
