#!/usr/bin/env python3
"""Remote half of fpga_run.sh; all generated files stay in the approved root."""
import argparse
from contextlib import contextmanager
import errno
import fcntl
import hashlib
import json
import math
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
import tempfile
import time

ERROR = re.compile(
    rb'(?:^|\n)(?:[^\r\n]*\]\s*(?:ERROR|FATAL):|(?:ERROR|FATAL):|'
    rb'Error: in running command:|make(?:\[\d+\])?: \*\*\*|'
    rb'[ \t]*Total (?:ERROR|FATAL):[ \t]*[1-9])', re.MULTILINE)
UART_FAILURE = re.compile(rb'TRAP mcause=|verify[^\r\n]*:\s*(?:FAIL|mismatches=[1-9])|\[nr\][^\r\n]*FAIL')


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


def layout_segments(run_dir, plan_name, expected_sha):
    """Check the upload manifest before letting the platform load any memory."""
    if not plan_name or not re.fullmatch(r'[A-Za-z0-9_.-]+', plan_name):
        raise RuntimeError('layout plan must be a basename inside this run directory')
    plan = run_dir / plan_name
    manifest_path = run_dir / 'run-manifest.json'
    if plan.is_symlink() or manifest_path.is_symlink():
        raise RuntimeError('layout inputs cannot be symlinks')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('plan') != plan_name or manifest.get('plan_sha256') != digest(plan):
        raise RuntimeError('layout plan SHA256 mismatch')
    segments = manifest.get('segments', [])
    seen = set()
    for segment in segments:
        name = segment.get('file')
        if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or
                name in ('.', '..') or name in seen):
            raise RuntimeError('invalid layout segment name')
        source = run_dir / name
        readback = run_dir / (name + '.readback')
        if source.is_symlink() or readback.is_symlink() or not source.is_file():
            raise RuntimeError('layout segment/readback must stay in this run directory')
        if (source.stat().st_size <= 0 or source.stat().st_size != segment.get('size') or
                digest(source) != segment.get('sha256')):
            raise RuntimeError('layout segment size/SHA256 mismatch: ' + name)
        if name == 'image.bin' and segment['sha256'] != expected_sha:
            raise RuntimeError('layout boot image SHA256 mismatch')
        seen.add(name)
    if 'image.bin' not in seen:
        raise RuntimeError('layout must contain the uploaded image.bin')
    return segments


def verify_layout_readbacks(run_dir, segments):
    results = {}
    for segment in segments:
        name = segment['file']
        source, readback = run_dir / name, run_dir / (name + '.readback')
        if (source.is_symlink() or readback.is_symlink() or not readback.is_file() or
                readback.stat().st_size != segment['size'] or
                digest(source) != segment['sha256'] or
                digest(readback) != segment['sha256']):
            size = readback.stat().st_size if readback.is_file() else -1
            raise RuntimeError('missing or mismatched segment readback: ' + name +
                               ' expected_bytes=%d actual_bytes=%d' % (segment['size'], size))
        results[name] = True
    return results


class UartInput:
    """Retain unsent UTF-8 bytes across nonblocking partial writes/EAGAIN."""
    def __init__(self, payloads, delay, gap):
        self.pending = [payload.encode('utf-8') for payload in payloads if payload]
        self.delay, self.gap = delay, gap
        self.next_time = None
        self.sent_bytes = 0

    def start(self, now):
        self.next_time = now + self.delay

    def pump(self, serial, now):
        if not self.pending or self.next_time is None or now < self.next_time:
            return
        try:
            count = os.write(serial, self.pending[0])
        except BlockingIOError:
            return
        if count <= 0:
            raise RuntimeError('UART input write made no progress')
        self.sent_bytes += count
        self.pending[0] = self.pending[0][count:]
        if not self.pending[0]:
            self.pending.pop(0)
            self.next_time = now + self.gap
            log('UART payload written; total={} bytes (board receipt unconfirmed)'.format(self.sent_bytes))


class InputQueue:
    """A single UART owner consumes immutable, deduplicated host requests.

    An ack means the OS accepted all bytes, not that firmware received them.
    A crashed worker is never restarted: absence of its ack must not cause a
    second worker to replay an ambiguously partially written UART payload.
    """
    def __init__(self, directory):
        self.directory = directory / 'input'
        self.active = None
        self.remaining = b''
        self.sent_bytes = 0

    @contextmanager
    def locked(self):
        self.directory.mkdir(mode=0o700, exist_ok=True)
        if self.directory.is_symlink() or self.directory.resolve().parent != self.directory.parent.resolve():
            raise RuntimeError('input queue must be a real directory inside this run')
        fd = os.open(self.directory/'queue.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def read(self, path):
        if path.is_symlink():
            raise RuntimeError('input queue records cannot be symlinks')
        return json.loads(path.read_text())

    def write(self, path, value):
        if path.is_symlink():
            raise RuntimeError('input queue records cannot be symlinks')
        fd, temporary = tempfile.mkstemp(prefix='.pending-', dir=str(self.directory))
        try:
            with os.fdopen(fd, 'w') as output:
                json.dump(value, output)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)

    def requests(self):
        jobs = [self.read(path) for path in self.directory.glob('*.request.json')]
        for job in jobs:
            if not re.fullmatch(r'[0-9a-f]{32}', job.get('id', '')):
                raise RuntimeError('invalid UART request id')
            data = bytes.fromhex(job['hex'])
            if not data or len(data) > 4096 or hashlib.sha256(data).hexdigest() != job.get('sha256'):
                raise RuntimeError('invalid UART request payload')
        return sorted(jobs, key=lambda job: job['sequence'])

    def enqueue(self, request_id, data):
        if not re.fullmatch(r'[0-9a-f]{32}', request_id) or not data or len(data) > 4096:
            raise RuntimeError('UART request needs a UUID and 1..4096 payload bytes')
        sha = hashlib.sha256(data).hexdigest()
        with self.locked():
            path = self.directory/(request_id+'.request.json')
            if path.exists() or path.is_symlink():
                existing = self.read(path)
                if existing.get('sha256') != sha or existing.get('hex') != data.hex():
                    raise RuntimeError('UART request id reused for different input')
                return existing
            if (self.directory/'closed.json').exists() or (self.directory.parent/'result.json').exists():
                raise RuntimeError('UART worker has finished accepting input')
            jobs = self.requests()
            sequence = jobs[-1]['sequence'] + 1 if jobs else 0
            job = {'id': request_id, 'sequence': sequence, 'sha256': sha, 'hex': data.hex()}
            self.write(path, job)
            return job

    def pending(self):
        pending = []
        for job in self.requests():
            path = self.directory/(job['id']+'.ack.json')
            if not path.exists() and not path.is_symlink():
                pending.append(job)
                continue
            ack = self.read(path)
            if (ack.get('id') != job['id'] or ack.get('sha256') != job['sha256'] or
                    ack.get('bytes_written') != len(bytes.fromhex(job['hex']))):
                raise RuntimeError('UART acknowledgement does not match its request')
        return pending

    def pump(self, serial):
        if self.active is None:
            with self.locked():
                pending = self.pending()
                if not pending: return
                self.active = pending[0]
                self.remaining = bytes.fromhex(self.active['hex'])
        try:
            count = os.write(serial, self.remaining)
        except BlockingIOError:
            return
        if count <= 0: raise RuntimeError('UART queue write made no progress')
        self.sent_bytes += count
        self.remaining = self.remaining[count:]
        if not self.remaining:
            with self.locked():
                job = self.active
                self.write(self.directory/(job['id']+'.ack.json'), {
                    'id': job['id'], 'sequence': job['sequence'], 'sha256': job['sha256'],
                    'bytes_written': len(bytes.fromhex(job['hex'])),
                    'meaning': 'OS write completed; FPGA receipt unconfirmed'})
            log('UART input request {} written (FPGA receipt unconfirmed)'.format(job['id']))
            self.active = None

    def close(self):
        with self.locked():
            self.write(self.directory/'closed.json', {'closed': True})
            return len(self.pending())


def enqueue_input(request_id, payload_hex):
    root, directory = Path.cwd(), Path(__file__).resolve().parent
    check_paths(root, directory)
    if not (directory/'started.json').is_file():
        raise RuntimeError('no existing worker reservation for this input queue')
    job = InputQueue(directory).enqueue(request_id, bytes.fromhex(payload_hex))
    print(json.dumps({'accepted': True, 'id': job['id'], 'sequence': job['sequence'],
                      'sha256': job['sha256']}), flush=True)
    return 0


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
        if getattr(args, 'background', False) and (run_dir / 'stop').exists():
            stopped_flag[0] = True
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
    layout = getattr(args, 'layout', None)
    uart_input = UartInput(getattr(args, 'uart_input', []),
                           getattr(args, 'uart_input_delay', 60.0),
                           getattr(args, 'uart_input_gap', 5.0))
    input_queue = InputQueue(run_dir)
    try:
        lock = os.open(root / '.fpga_run.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another fpga_run is using this platform workdir')
        raw = run_dir / 'image.bin'
        if raw.is_symlink() or not raw.is_file() or raw.stat().st_size == 0 or digest(raw) != args.sha256:
            raise RuntimeError('uploaded binary SHA256 mismatch or empty input')
        segments = layout_segments(run_dir, layout, args.sha256) if layout else None
        # Upload and execution are separate steps; another process can consume
        # free space between them. Check again under the platform lock.
        raw_size = raw.stat().st_size
        readback_budget = (sum(s['size'] for s in segments) if segments is not None else
                           2 * max(1048576, ((raw_size + 1048575) // 1048576) * 1048576))
        available = shutil.disk_usage(str(root)).free
        if available < readback_budget + 64 * 1024 * 1024:
            raise RuntimeError('insufficient remote disk space for DDR readbacks: required=%d available=%d' %
                               (readback_budget + 64 * 1024 * 1024, available))
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
            if layout is not None:
                # run_uv_layout.sh resolves the plan relative to the platform
                # root, while the upload lands in this run's directory.
                plan = Path(run_dir) / layout
                if not plan.is_file():
                    raise RuntimeError(f'load plan not found: {plan}')
                generated = root / 'cmd' / (plan.stem + '_uv_run' + str(args.fpga) + '.tcl')
                if generated.is_symlink() or not within(generated, root):
                    raise RuntimeError('platform Tcl output points outside the workdir')
                make_target = ['make', f'uv_run{args.fpga}',
                               f'layout={os.path.relpath(plan, root)}']
            else:
                make_target = ['make', f'uv_run{args.fpga}', f'test={raw}']
            process = subprocess.Popen(make_target,
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
        marker_text = getattr(args, 'completion_marker', None)
        marker = marker_text.encode('utf-8') if marker_text else None
        marker_time = None
        heartbeat = time.monotonic()
        with (run_dir / 'uvhs.log').open('wb') as uvlog, (run_dir / 'uart.raw.log').open('wb') as uart:
            while True:
                if stopped():
                    raise InterruptedError('interrupted; releasing this session')
                now = time.monotonic()
                if now - heartbeat >= 30:
                    elapsed = int(now - started) if started is not None else 0
                    log(f'FPGA{args.fpga} session active; elapsed={elapsed}s, UART={uart_total} bytes')
                    heartbeat = now
                if started is None and now >= deadline:
                    raise RuntimeError('FPGA startup timed out; see uvhs.log')
                if started is not None and exit_sent is None and (now - started >= args.capture_seconds or (marker_time is not None and now - marker_time >= 0.5)):
                    if input_queue.close() or uart_input.pending:
                        raise RuntimeError('UART input was not completely sent before capture ended')
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
                        if marker and marker in uart_tail and marker_time is None:
                            marker_time = time.monotonic()
                        failed_uart |= bool(UART_FAILURE.search(uart_tail))
                        sys.stdout.buffer.write(data); sys.stdout.buffer.flush()
                    else:
                        uvlog.write(data); uvlog.flush()
                        output.extend(data)
                        if ERROR.search(output):
                            raise RuntimeError('platform reported ERROR/FATAL; see uvhs.log')
                        if started is None and b'hspRun>' in output:
                            loaded = (b'UV_RUN_IMAGE=' in output if layout is None
                                      else b'segment ' in output)
                            if not loaded or b'reset -name cpu_reset -value 0 success' not in output:
                                raise RuntimeError('UVHS reached its prompt without loading/starting the image')
                            started = time.monotonic()
                            if segments is not None:
                                # Catch truncated readmem output immediately,
                                # before spending a full model run on bad load evidence.
                                result['segment_readbacks'] = verify_layout_readbacks(run_dir, segments)
                                result['ddr_readback_matches'] = True
                            uart_input.start(started)
                            log(f'FPGA{args.fpga} started; capturing for {args.capture_seconds} seconds')
                if exit_sent is None:
                    uart_input.pump(serial, time.monotonic())
                    if started is not None and not uart_input.pending:
                        input_queue.pump(serial)
                if process.poll() is not None:
                    if process.returncode or started is None or exit_sent is None:
                        raise RuntimeError(f'platform exited unexpectedly ({process.returncode}); see uvhs.log')
                    break
        if layout is None:
            result['padded_sha256'] = verify_readback(run_dir, bytes(output), args.sha256)
            result['ddr_readback_matches'] = True
        else:
            result['segment_readbacks'] = verify_layout_readbacks(run_dir, segments)
            result['ddr_readback_matches'] = True
        result['uart_bytes'] = uart_total
        if failed_uart:
            raise RuntimeError('program reported a verification failure or trap')
        if not uart_total:
            raise RuntimeError('no UART output during the capture window')
        if marker:
            result['completion_marker'] = marker_text
            result['completion_marker_seen'] = marker_time is not None
            if marker_time is None:
                raise RuntimeError('UART completion marker missing at capture timeout')
        result['status'] = 'OK'
        status = 0
        log(f'Finished: {uart_total} UART bytes; DDR readback matches')
    except InterruptedError as e:
        result.update(status='INTERRUPTED', error=str(e))
        status = 130
        log(str(e))
    except (OSError, RuntimeError, ValueError) as e:
        result['error'] = str(e)
        log('ERROR: ' + str(e))
    finally:
        result['uart_input_bytes_written'] = uart_input.sent_bytes
        result['uart_interactive_bytes_written'] = input_queue.sent_bytes
        try:
            result['uart_input_unacknowledged_requests'] = input_queue.close()
        except (OSError, RuntimeError, ValueError) as error:
            result.update(status='ERROR', error='input queue cleanup: ' + str(error))
            status = 1
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
                cleanup(lambda: termios.tcsetattr(serial, termios.TCSANOW, original))
            cleanup(lambda: fcntl.ioctl(serial, termios.TIOCNXCL))
            cleanup(lambda: os.close(serial))
        if lock is not None:
            cleanup(lambda: os.close(lock))
        for sig, handler in old_signals.items():
            signal.signal(sig, handler)
        if cleanup_errors:
            result['cleanup_errors'] = cleanup_errors
            result.update(status='ERROR', error=result.get('error', 'session cleanup failed'))
            status = 1
        pending = run_dir / 'result.json.tmp'
        pending.write_text(json.dumps(result, indent=2) + '\n')
        pending.replace(run_dir / 'result.json')
    return status


def detached_start(args):
    """Start at most once, even when SSH loses the acknowledgement."""
    root = Path.cwd()
    directory = Path(__file__).resolve().parent
    check_paths(root, directory)
    fd = os.open(directory / 'start.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        state = directory / 'started.json'
        if state.exists():
            recorded = json.loads(state.read_text())
            if recorded['sha256'] != args.sha256 or recorded['fpga'] != args.fpga:
                raise RuntimeError('run identity mismatch; refusing another hardware launch')
            return 0
        # Reserve the launch before spawning. A failed/uncertain spawn must never
        # cause a second FPGA reset; the relay will report a missing worker.
        pending=directory/'started.json.tmp'
        pending.write_text(json.dumps({'sha256':args.sha256, 'fpga':args.fpga, 'pid':None}))
        pending.replace(state)
        command = [sys.executable, '-u', '-B', str(Path(__file__).resolve()),
                   '--background', '--fpga', str(args.fpga), '--sha256', args.sha256,
                   '--capture-seconds', str(args.capture_seconds),
                   '--startup-timeout', str(args.startup_timeout), '--baud', str(args.baud)]
        if args.completion_marker:
            command += ['--completion-marker', args.completion_marker]
        # Every option the detached worker needs must be forwarded explicitly:
        # this command line is rebuilt, not reused, and dropping one silently
        # falls back to the single-image path.
        if getattr(args, 'layout', None) is not None:
            command += ['--layout', args.layout]
        # Same trap as --layout: this command line is rebuilt, so every option
        # the worker needs has to be forwarded explicitly.
        for payload in getattr(args, 'uart_input', []) or []:
            command += ['--uart-input='+payload]
        command += ['--uart-input-delay', str(getattr(args, 'uart_input_delay',
                                                      60.0))]
        command += ['--uart-input-gap', str(getattr(args, 'uart_input_gap', 5.0))]
        with (directory / 'worker.log').open('ab', buffering=0) as output:
            process = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=output, start_new_session=True)
        pending.write_text(json.dumps({'sha256':args.sha256, 'fpga':args.fpga, 'pid':process.pid}))
        pending.replace(state)
        return 0
    finally:
        os.close(fd)


def relay(offset):
    """Read only: reconnect at the exact UART byte offset, without reloading."""
    directory = Path(__file__).resolve().parent
    uart = directory / 'uart.raw.log'
    result_path = directory / 'result.json'
    heartbeat = time.monotonic()
    empty_deadline = heartbeat + 30
    while True:
        if uart.exists():
            with uart.open('rb') as stream:
                stream.seek(offset)
                data = stream.read(65536)
            if data:
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
                offset += len(data)
                continue
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result['status'] != 'OK':
                log(result.get('error', result['status']))
            return 0 if result['status']=='OK' else 130 if result['status']=='INTERRUPTED' else 1
        now = time.monotonic()
        state_path = directory / 'started.json'
        if now > empty_deadline:
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            pid_file = directory / 'worker.pid'
            pid = int(pid_file.read_text()) if pid_file.exists() else state.get('pid')
            proc = Path('/proc', str(pid), 'cmdline')
            if not pid or not proc.exists() or str(Path(__file__).resolve()).encode() not in proc.read_bytes():
                raise RuntimeError('worker exited without a result; run was not restarted')
        if now - heartbeat >= 15:
            log(f'UART relay active; received={offset} bytes')
            heartbeat = now
        time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpga', type=int, choices=range(8))
    parser.add_argument('--sha256')
    parser.add_argument('--completion-marker')
    parser.add_argument('--capture-seconds', type=int)
    parser.add_argument('--startup-timeout', type=int)
    parser.add_argument('--baud', type=int)
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--background', action='store_true')
    parser.add_argument('--relay', type=int, metavar='UART_OFFSET')
    parser.add_argument('--enqueue', metavar='REQUEST_UUID')
    parser.add_argument('--input-hex', default='')
    # Model runs need more than the single 1 MiB boot image: the parameters
    # arrive as their own DDR segment. With --layout the platform's own
    # run_uv_layout.sh loads and readback-verifies every segment of the plan
    # instead of the single image, so the image-specific readback check below is
    # replaced by the platform's per-segment one. Unset means the single-image
    # behaviour, unchanged.
    parser.add_argument('--layout', metavar='PLAN')
    # Interactive firmware needs bytes sent *to* the board. The worker holds the
    # UART device exclusively (it restores the tty with TIOCNXCL only on exit), so
    # a second process cannot open it -- the input has to be written here, by the
    # same process that owns the capture.
    parser.add_argument('--uart-input', action='append', default=[],
                        help='text to write to the board UART after startup')
    parser.add_argument('--uart-input-delay', type=float, default=60.0,
                        help='seconds after startup before the first write')
    parser.add_argument('--uart-input-gap', type=float, default=5.0,
                        help='seconds between successive writes')
    args = parser.parse_args()
    if any(not math.isfinite(value) or value < 0 for value in
           (args.uart_input_delay, args.uart_input_gap)):
        parser.error('UART input timing must be finite and nonnegative')
    try:
        if args.enqueue is not None:
            return enqueue_input(args.enqueue, args.input_hex)
        if args.relay is not None:
            if args.relay < 0: parser.error('UART offset cannot be negative')
            return relay(args.relay)
        if any(getattr(args, name) is None for name in
               ('fpga','sha256','capture_seconds','startup_timeout','baud')):
            parser.error('run options are required')
        if args.detach:
            return detached_start(args)
        if args.background:
            (Path(__file__).resolve().parent / 'worker.pid').write_text(str(os.getpid()))
        return run(args, control_fd=None if args.background else 0)
    except (OSError, RuntimeError, ValueError) as error:
        log('ERROR: '+str(error))
        return 1


if __name__ == '__main__':
    sys.exit(main())
