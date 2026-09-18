#!/usr/bin/env python3
# ===- fpga_run.py -------------------------------------------------------------
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
# Upload an NR image and relay its UART through one managed SSH session.
#
# ===---------------------------------------------------------------------------

import argparse
import hashlib
import json
import os
import select
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path

DEFAULT_REMOTE_DIR = "Desktop/fpga-tester-ISCAS"
SSH_OPTIONS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=2",
]
DISK_RESERVE_BYTES = 64 * 1024 * 1024


def upload_disk_budget(args):
    """New uploads plus full DDR readbacks and small platform/log scratch."""
    size = args.image.stat().st_size
    padded = max(1048576, (size + 1048575) // 1048576 * 1048576)
    return size + 2 * padded + DISK_RESERVE_BYTES


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="fpga_run.sh",
        description="Upload an NR .bin, run make uv_runN on ssh fpga, and stream UART.",
        epilog="UART goes to stdout; status goes to stderr. Ctrl-C closes this run. "
        "No local compilation is performed. Close minicom before running.",
    )
    p.add_argument("image", type=Path, help="local binary, padded or unpadded")
    p.add_argument("--fpga", type=int, choices=range(8), required=True)
    p.add_argument(
        "--capture-seconds",
        type=positive,
        default=10,
        help="capture duration after startup (default: 10)",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=5,
        help="SSH reconnection attempts (default: 5)",
    )
    p.add_argument(
        "--retry-delay",
        type=positive,
        default=3,
        help="seconds between SSH retries",
    )
    p.add_argument(
        "--startup-timeout",
        type=positive,
        default=180,
        help="maximum loading/startup time in seconds (default: 180)",
    )
    p.add_argument(
        "--baud",
        type=int,
        choices=(9600, 19200, 38400, 57600, 115200, 230400, 460800),
        default=115200,
    )
    p.add_argument(
        "--ssh-host", default=os.environ.get("FPGA_SSH_HOST", "fpga")
    )
    p.add_argument(
        "--remote-dir",
        default=os.environ.get("FPGA_REMOTE_DIR", DEFAULT_REMOTE_DIR),
        help="server workdir, relative to SSH login directory unless absolute "
        "(default: Desktop/fpga-tester-ISCAS)",
    )
    args = p.parse_args(argv)
    if args.retries < 0:
        p.error("retries cannot be negative")
    args.image = args.image.expanduser().resolve()
    if not args.image.is_file() or args.image.stat().st_size == 0:
        p.error("image must be an existing, nonempty file")
    if not args.ssh_host or args.ssh_host.startswith("-"):
        p.error("invalid SSH host")
    if (
        not args.remote_dir
        or "\x00" in args.remote_dir
        or "\n" in args.remote_dir
    ):
        p.error("invalid remote directory")
    if args.remote_dir.startswith("~"):
        p.error(
            "use a login-relative remote directory without ~/ or an absolute path"
        )
    return args


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    args = parse_args(argv)
    run_id = "run-" + uuid.uuid4().hex[:16]
    remote_dir = "fpga-runs/" + run_id
    remote_display = (
        args.ssh_host + ":" + args.remote_dir.rstrip("/") + "/" + remote_dir
    )
    local_dir = (
        Path(__file__).resolve().parents[1] / "build" / "fpga-runs" / run_id
    )
    local_dir.mkdir(parents=True, exist_ok=False)
    ssh = ["ssh", *SSH_OPTIONS, args.ssh_host]
    ssh_env = dict(os.environ, LC_ALL="C", LANG="C")
    sha = digest(args.image)
    uploaded = False
    hardware_requested = False

    def command(words):
        return (
            "cd -- " + shlex.quote(args.remote_dir) + " && " + shlex.join(words)
        )

    def log(message):
        print("[fpga_run] " + message, file=sys.stderr, flush=True)

    def remote(words, **kwargs):
        for attempt in range(args.retries + 1):
            for stream_name in ("stdin", "stdout"):
                stream = kwargs.get(stream_name)
                if hasattr(stream, "seek"):
                    stream.seek(0)
                    if stream_name == "stdout":
                        stream.truncate()
            result = subprocess.run(
                [*ssh, command(words)], env=ssh_env, **kwargs
            )
            if result.returncode == 0:
                return result
            if result.returncode != 255 or attempt == args.retries:
                raise subprocess.CalledProcessError(result.returncode, words)
            log(
                f"SSH unavailable; reconnecting {attempt + 1}/{args.retries} in {args.retry_delay}s"
            )
            time.sleep(args.retry_delay)

    prepare = """from pathlib import Path
import sys, shutil
root=Path.cwd()
required=int(sys.argv[2]); available=shutil.disk_usage(str(root)).free
if available < required:
    raise SystemExit('insufficient remote disk space for uploads and DDR readbacks: required=%d available=%d; existing files were not removed' % (required, available))
runs=root/'fpga-runs'
runs.mkdir(exist_ok=True)
if runs.is_symlink() or runs.resolve().parent != root:
    raise SystemExit('fpga-runs must be a real directory inside the workdir')
directory=runs/sys.argv[1]
directory.mkdir(mode=0o700,exist_ok=True)
if directory.is_symlink() or directory.resolve().parent != runs:
    raise SystemExit('invalid private run directory')
"""
    try:
        log(
            f"FPGA{args.fpga}: {args.image} ({args.image.stat().st_size} bytes)"
        )
        log(f"SHA256 {sha}")
        log(f"Server workdir: {args.ssh_host}:{args.remote_dir}")
        remote(
            [
                "python3",
                "-B",
                "-c",
                prepare,
                run_id,
                str(upload_disk_budget(args)),
            ]
        )
        upload = """from pathlib import Path
import hashlib, os, sys
path=Path(sys.argv[1]); temporary=path.with_name(path.name+'.upload')
fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_TRUNC|os.O_NOFOLLOW,0o600)
sha=hashlib.sha256()
with os.fdopen(fd,'wb') as output:
    for block in iter(lambda:sys.stdin.buffer.read(1048576),b''):
        output.write(block);sha.update(block)
if sha.hexdigest()!=sys.argv[2]:raise SystemExit('incomplete upload')
if path.is_symlink():raise SystemExit('upload destination is a symlink')
temporary.replace(path)
"""
        for source, name in (
            (Path(__file__).with_name("fpga_remote.py"), "runner.py"),
            (args.image, "image.bin"),
        ):
            with source.open("rb") as stream:
                remote(
                    [
                        "python3",
                        "-B",
                        "-c",
                        upload,
                        remote_dir + "/" + name,
                        digest(source),
                    ],
                    stdin=stream,
                )
        uploaded = True
        log("Upload complete. Opening UART and loading the image.")
        words = [
            "python3",
            "-u",
            "-B",
            remote_dir + "/runner.py",
            "--fpga",
            str(args.fpga),
            "--sha256",
            sha,
            "--capture-seconds",
            str(args.capture_seconds),
            "--startup-timeout",
            str(args.startup_timeout),
            "--baud",
            str(args.baud),
        ]
        hardware_requested = True
        remote([*words, "--detach"])
        status = 1
        process = None
        try:
            with (local_dir / "uart.raw.log").open("wb") as uart:
                for attempt in range(args.retries + 1):
                    relay = [
                        "python3",
                        "-u",
                        "-B",
                        remote_dir + "/runner.py",
                        "--relay",
                        str(uart.tell()),
                    ]
                    process = subprocess.Popen(
                        [*ssh, command(relay)],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        env=ssh_env,
                        start_new_session=True,
                    )
                    try:
                        while True:
                            ready, _, _ = select.select(
                                [process.stdout.fileno()], [], [], 0.1
                            )
                            if process.stdout.fileno() in ready:
                                block = os.read(process.stdout.fileno(), 4096)
                                if not block:
                                    break
                                uart.write(block)
                                uart.flush()
                                sys.stdout.buffer.write(block)
                                sys.stdout.buffer.flush()
                        status = process.wait()
                    finally:
                        process.stdout.close()
                    if (
                        status != 255 and status >= 0
                    ) or attempt == args.retries:
                        break
                    log(
                        f"SSH disconnected; resuming UART at byte {uart.tell()} ({attempt + 1}/{args.retries})"
                    )
                    time.sleep(args.retry_delay)
        except (KeyboardInterrupt, BrokenPipeError):
            log("Stopping this run and releasing UART/FPGA...")
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            try:
                remote(
                    [
                        "python3",
                        "-B",
                        "-c",
                        "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
                        remote_dir + "/stop",
                    ]
                )
            except subprocess.CalledProcessError:
                log(
                    "Stop request could not reach server; worker remains bounded by its configured timeouts"
                )
            status = 130
        if status == 255 or status < 0:
            log(
                "Reconnection limit reached; the bounded worker retains server logs and may still be running"
            )
        if status == 130:
            wait_result = """from pathlib import Path
import sys,time
p=Path(sys.argv[1]);deadline=time.monotonic()+25
while not p.exists() and time.monotonic()<deadline:time.sleep(0.1)
if not p.exists():raise SystemExit('worker cleanup has not finished; logs remain on server')
"""
            try:
                remote(
                    [
                        "python3",
                        "-B",
                        "-c",
                        wait_result,
                        remote_dir + "/result.json",
                    ]
                )
            except subprocess.CalledProcessError:
                log(
                    "Worker cleanup is still pending; inspect the remote run directory"
                )
        for name in ("result.json", "uvhs.log", "worker.log"):
            try:
                with (local_dir / name).open("wb") as f:
                    remote(["cat", remote_dir + "/" + name], stdout=f)
            except subprocess.CalledProcessError:
                log(f"Could not retrieve {name}; inspect {remote_display}")
        result_file = local_dir / "result.json"
        if status == 0:
            try:
                result = json.loads(result_file.read_text())
                if result.get("status") != "OK" or result.get("sha256") != sha:
                    status = 1
            except (OSError, ValueError):
                status = 1
        log(f"Local logs: {local_dir}")
        log(f"Remote logs: {remote_display}")
        return status
    except KeyboardInterrupt:
        log("Interrupted.")
        if hardware_requested:
            try:
                remote(
                    [
                        "python3",
                        "-B",
                        "-c",
                        "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
                        remote_dir + "/stop",
                    ]
                )
            except (subprocess.CalledProcessError, KeyboardInterrupt):
                log(
                    "Stop request could not reach server; worker remains bounded by its configured timeouts"
                )
        return 130
    except (OSError, ValueError, subprocess.CalledProcessError) as e:
        log(str(e))
        if uploaded:
            log(f"Remote files: {remote_display}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
