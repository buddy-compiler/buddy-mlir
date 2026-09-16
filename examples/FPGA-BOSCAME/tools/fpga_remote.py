#!/usr/bin/env python3
"""Remote half of fpga_run.sh; all generated files stay in the approved root."""
import argparse
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pty
import re
import select
import shutil
import signal
import subprocess
import sys
import termios
import time

ERROR = re.compile(
    rb'(?:^|\n)(?:[^\r\n]*\]\s*(?:ERROR|FATAL):|(?:ERROR|FATAL):|'
    rb'Error: in running command:|make(?:\[\d+\])?: \*\*\*|'
    rb'[ \t]*Total (?:ERROR|FATAL):[ \t]*[1-9])', re.MULTILINE)
UART_FAILURE = re.compile(rb'TRAP mcause=|verify[^\r\n]*:\s*(?:FAIL|mismatches=[1-9])')


def log(message):
    print('[fpga_run] ' + message, file=sys.stderr, flush=True)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def within(path, root):
    return path.resolve() == root or root in path.resolve().parents


def check_paths(root, run_dir):
    if root.resolve() != root or not within(run_dir, root) or run_dir.is_symlink():
        raise RuntimeError('run directory must be inside the approved workdir')
    for name in ('cmd', 'test', 'apsRun', 'uv_run.log', 'uv_stacktrace.txt',
                 '.socket_uv_shell', '.uvshell_workdir_lock', '.fpga_run.lock'):
        if not within(root / name, root):
            raise RuntimeError(f'refusing output path that points outside workdir: {name}')


def check_processes(root, fpga, device):
    """Do not take over existing sessions or read from someone else's UART."""
    actual = os.path.realpath(device)
    for proc in Path('/proc').glob('[0-9]*'):
        if int(proc.name) == os.getpid():
            continue
        try:
            argv = proc.joinpath('cmdline').read_bytes().split(b'\0')
            executable = os.path.basename(os.fsdecode(argv[0]))
            args = [os.fsdecode(a) for a in argv if a]
            if executable in ('uvhs2_shell', 'uv_shell_exec') or any(
                    os.path.basename(a) == 'uvhs2_shell' for a in args[:2]):
                same_dir = proc.joinpath('cwd').resolve() == root
                same_board = any(Path(a).name == f'hw_run{fpga}.tcl' for a in args)
                if same_dir or same_board:
                    raise RuntimeError(f'existing UVHS session PID {proc.name}; close it first')
            if executable in ('minicom', 'picocom', 'screen', 'cat', 'socat') and (
                    device in args or actual in args):
                raise RuntimeError(f'{device} is in use by PID {proc.name}')
            for fd in proc.joinpath('fd').iterdir():
                try:
                    if os.readlink(fd) == actual:
                        raise RuntimeError(f'{device} is open in PID {proc.name}')
                except (FileNotFoundError, PermissionError):
                    pass
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue


def board_available(output, fpga):
    board, slot = divmod(fpga, 4)
    pattern = (rf'^\S+\s+B{board}\s+F{slot}\s+\S+\s+link\s+(?:up|down)\s+'
               r'(?:(\S+)\s+)?(true|false)\s*$')
    row = re.search(pattern, output.decode(errors='replace'), re.MULTILINE)
    if not row:
        raise RuntimeError(f'could not establish FPGA{fpga} availability from UVHS query')
    if row[1] or row[2] == 'true':
        raise RuntimeError(f'FPGA{fpga} is occupied/booked (owner: {row[1] or "reserved"})')


def shutdown(process, master=None):
    """Exit the session, escalating only within the process group we created."""
    if process is None:
        return
    if process.poll() is None and master is not None:
        try:
            os.write(master, b'exit\n')
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
    # Even if a wrapper exited, UVHS descendants may still hold its group.
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
    executable = shutil.which('uvhs2_shell')
    if not executable:
        raise RuntimeError('uvhs2_shell is not in the SSH environment PATH')
    script = root / 'user_script' / 'hw_run' / f'hw_run{fpga}.tcl'
    text = script.read_text()
    targets = re.findall(r'^\s*load_db\s+-db\s+\S+\s+-to\s+\{?(b\d+\.f\d+)\}?', text, re.M | re.I)
    if targets != [f'b{fpga // 4}.f{fpga % 4}']:
        raise RuntimeError(f'unexpected board mapping in {script}')
    query = run_dir / 'query.tcl'
    query.write_text('query -fpgas -all\nexit\n')
    process = None
    with (run_dir / 'preflight.log').open('wb') as output:
        try:
            process = subprocess.Popen([executable, '-t', 'runtime', '-d', 'V1',
                                        '-workdir', './', '-script', str(query),
                                        '-bypass_vivado_version_check'], cwd=root,
                                       stdin=subprocess.DEVNULL, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 45
            while process.poll() is None:
                if stopped():
                    raise InterruptedError('resource query interrupted')
                if time.monotonic() >= deadline:
                    raise RuntimeError('resource query timed out')
                time.sleep(0.1)
            if process.returncode:
                raise RuntimeError(f'UVHS resource query exited {process.returncode}')
        finally:
            shutdown(process)
    output = (run_dir / 'preflight.log').read_bytes()
    if ERROR.search(output):
        raise RuntimeError('UVHS resource query failed; see preflight.log')
    board_available(output, fpga)


def verify_readback(run_dir, runtime_output, expected_sha):
    match = re.search(rb'UV_RUN_IMAGE=([^\r\n]+)', runtime_output)
    if not match:
        raise RuntimeError('platform did not report UV_RUN_IMAGE')
    image = Path(os.fsdecode(match[1])).resolve()
    if image.parent != run_dir:
        raise RuntimeError('platform selected an image outside this run directory')
    readback = Path(str(image) + '.readback')
    if readback.is_symlink() or not readback.is_file():
        raise RuntimeError('missing DDR readback')
    raw = run_dir / 'image.bin'
    if digest(raw) != expected_sha:
        raise RuntimeError('uploaded image changed during this run')
    if image.stat().st_size % 64 or image.stat().st_size < raw.stat().st_size:
        raise RuntimeError('invalid padded image size')
    with raw.open('rb') as a, image.open('rb') as b:
        for block in iter(lambda: a.read(1024 * 1024), b''):
            if b.read(len(block)) != block:
                raise RuntimeError('padded image does not contain the uploaded binary')
        for block in iter(lambda: b.read(1024 * 1024), b''):
            if any(block):
                raise RuntimeError('nonzero padding after uploaded binary')
    sha = digest(image)
    if digest(readback) != sha:
        raise RuntimeError('DDR readback differs from the loaded image')
    return sha


def run(args, root=None, run_dir=None, device=None, control_fd=0):
    # The SSH launcher enters the configured platform workdir before starting
    # this worker. Derive the allowed root from that directory, not a username.
    root = Path(root) if root is not None else Path.cwd()
    run_dir = Path(run_dir or Path(__file__).resolve().parent)
    device = device or f'/dev/FPGA{args.fpga}'
    os.chdir(root)
    check_paths(root, run_dir)
    result = {'status': 'ERROR', 'fpga': args.fpga, 'sha256': args.sha256,
              'capture_seconds': args.capture_seconds, 'baud': args.baud}
    stopped_flag = [False]
    old_signals = {}
    def stop(signum, frame):
        stopped_flag[0] = True
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        old_signals[sig] = signal.signal(sig, stop)
    def stopped():
        if stopped_flag[0]:
            return True
        if control_fd is not None and select.select([control_fd], [], [], 0)[0]:
            # STOP or SSH stdin EOF both request graceful cleanup.
            os.read(control_fd, 4096)
            stopped_flag[0] = True
        return stopped_flag[0]
    lock = serial = master = None
    original = process = None
    status = 1
    try:
        lock = os.open(root / '.fpga_run.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another fpga_run is using this platform workdir')
        raw = run_dir / 'image.bin'
        if raw.is_symlink() or not raw.is_file() or raw.stat().st_size == 0 or digest(raw) != args.sha256:
            raise RuntimeError('uploaded binary SHA256 mismatch or empty input')
        check_processes(root, args.fpga, device)
        preflight(root, run_dir, args.fpga, stopped)
        check_processes(root, args.fpga, device)
        if stopped():
            raise InterruptedError('stopped before loading')
        serial = os.open(device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        fcntl.ioctl(serial, termios.TIOCEXCL)
        original = termios.tcgetattr(serial)
        attrs = termios.tcgetattr(serial)
        attrs[0] = attrs[1] = attrs[3] = 0
        attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        attrs[4] = attrs[5] = getattr(termios, 'B' + str(args.baud))
        attrs[6][termios.VMIN] = attrs[6][termios.VTIME] = 0
        termios.tcsetattr(serial, termios.TCSANOW, attrs)
        termios.tcflush(serial, termios.TCIFLUSH)
        log(f'UART ready: {device}, {args.baud}, 8N1, no flow control')
        master, slave = pty.openpty()
        env = os.environ.copy()
        # Use the known platform defaults, regardless of the login environment.
        for key in ('UV_RUN_PLAN_TCL', 'UV_RUN_PREP_ONLY'):
            env.pop(key, None)
        env.update(UV_RUN_READBACK='1', UV_RUN_SYS_CLK_HZ='14745600',
                   UV_RUN_ALIGN_BYTES='1048576', UV_RUN_MIN_BYTES='1048576',
                   TMPDIR=str(run_dir))
        try:
            process = subprocess.Popen(['make', f'uv_run{args.fpga}', f'test={raw}'],
                                       cwd=root, env=env, stdin=slave, stdout=slave,
                                       stderr=slave, start_new_session=True)
        finally:
            os.close(slave)
        output = bytearray()
        uart_total = 0
        started = None
        exit_sent = None
        deadline = time.monotonic() + args.startup_timeout
        uart_tail = b''
        failed_uart = False
        with (run_dir / 'uvhs.log').open('wb') as uvlog, (run_dir / 'uart.raw.log').open('wb') as uart:
            while True:
                if stopped():
                    raise InterruptedError('interrupted; releasing this session')
                now = time.monotonic()
                if started is None and now >= deadline:
                    raise RuntimeError('FPGA startup timed out; see uvhs.log')
                if started is not None and exit_sent is None and now - started >= args.capture_seconds:
                    os.write(master, b'exit\n')
                    exit_sent = now
                if exit_sent is not None and now - exit_sent > 20:
                    raise RuntimeError('UVHS did not exit after capture')
                ready, _, _ = select.select([master, serial], [], [], 0.1)
                for fd in ready:
                    try:
                        data = os.read(fd, 65536)
                    except OSError as e:
                        if fd == master and e.errno == errno.EIO:
                            data = b''
                        else:
                            raise
                    if fd == serial:
                        if not data:
                            raise RuntimeError('UART disconnected')
                        uart.write(data); uart.flush()
                        uart_total += len(data)
                        uart_tail = (uart_tail + data)[-8192:]
                        failed_uart |= bool(UART_FAILURE.search(uart_tail))
                        sys.stdout.buffer.write(data); sys.stdout.buffer.flush()
                    else:
                        uvlog.write(data); uvlog.flush()
                        output.extend(data)
                        if ERROR.search(output):
                            raise RuntimeError('platform reported ERROR/FATAL; see uvhs.log')
                        if started is None and b'hspRun>' in output:
                            if b'UV_RUN_IMAGE=' not in output or b'reset -name cpu_reset -value 0 success' not in output:
                                raise RuntimeError('UVHS reached its prompt without loading/starting the image')
                            started = time.monotonic()
                            log(f'FPGA{args.fpga} started; capturing for {args.capture_seconds} seconds')
                if process.poll() is not None:
                    if process.returncode or started is None or exit_sent is None:
                        raise RuntimeError(f'platform exited unexpectedly ({process.returncode}); see uvhs.log')
                    break
        result['padded_sha256'] = verify_readback(run_dir, bytes(output), args.sha256)
        result['ddr_readback_matches'] = True
        result['uart_bytes'] = uart_total
        if failed_uart:
            raise RuntimeError('program reported a verification failure or trap')
        if not uart_total:
            raise RuntimeError('no UART output during the capture window')
        result['status'] = 'OK'
        status = 0
        log(f'Finished: {uart_total} UART bytes; DDR readback matches')
    except InterruptedError as e:
        result.update(status='INTERRUPTED', error=str(e))
        status = 130
        log(str(e))
    except (OSError, RuntimeError) as e:
        result['error'] = str(e)
        log('ERROR: ' + str(e))
    finally:
        shutdown(process, master)
        if master is not None:
            os.close(master)
        if serial is not None:
            try:
                if original is not None:
                    termios.tcsetattr(serial, termios.TCSANOW, original)
                fcntl.ioctl(serial, termios.TIOCNXCL)
            finally:
                os.close(serial)
        if lock is not None:
            os.close(lock)
        for sig, handler in old_signals.items():
            signal.signal(sig, handler)
        (run_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpga', type=int, choices=range(8), required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--capture-seconds', type=int, required=True)
    parser.add_argument('--startup-timeout', type=int, required=True)
    parser.add_argument('--baud', type=int, required=True)
    sys.exit(run(parser.parse_args()))
