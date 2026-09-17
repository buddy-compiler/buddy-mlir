#!/usr/bin/env python3
"""Upload an NR image and relay its UART through one managed SSH session."""
import argparse
import hashlib
import json
import math
import os
import re
import select
from pathlib import Path
import shlex
import subprocess
import sys
import time
import tomllib
import uuid

DEFAULT_REMOTE_DIR = 'Desktop/fpga-tester-ISCAS'
SSH_OPTIONS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
               '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2']
DISK_RESERVE_BYTES = 64 * 1024 * 1024


def upload_disk_budget(args):
    """New uploads plus full DDR readbacks and small platform/log scratch."""
    size = args.image.stat().st_size
    if args.layout_plan is not None:
        return 2 * (size + sum(p.stat().st_size for p in args.segment)) + DISK_RESERVE_BYTES
    # The single-image platform path also creates a padded image copy.
    padded = max(1048576, (size + 1048575) // 1048576 * 1048576)
    return size + 2 * padded + DISK_RESERVE_BYTES


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def nonnegative_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError('must be finite and nonnegative')
    return number


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog='fpga_run.sh',
        description='Upload an NR .bin, run make uv_runN on ssh fpga, and stream UART.',
        epilog='UART goes to stdout; status goes to stderr. Ctrl-C closes this run. '
               'No local compilation is performed. Close minicom before running.')
    p.add_argument('image', type=Path, help='local binary, padded or unpadded')
    p.add_argument('--fpga', type=int, choices=range(8), required=True)
    p.add_argument('--capture-seconds', type=positive, default=10,
                   help='capture duration after startup (default: 10)')
    p.add_argument('--resume-run', help='reattach to an existing run-<16 hex> without uploading or reloading')
    # Model runs load their parameters as a second DDR segment, which the
    # single-image path cannot express. These options hand the platform's own
    # multi-segment loader a plan plus the extra files it names; the image
    # argument stays the boot segment, so the single-image flow is untouched.
    p.add_argument('--layout-plan', type=Path,
                   help='DDR load plan (TOML) to upload and run instead of a single image')
    # Interactive firmware needs bytes sent *to* the board, which the capture
    # path never does. Only the worker writes the tty it owns. Successful host
    # writes alone do not establish that the FPGA received the bytes.
    p.add_argument('--uart-send', action='append', default=[],
                   help='text to write to the board UART after startup (repeatable)')
    p.add_argument('--interactive', action='store_true',
                   help='forward stdin bytes through the existing worker UART; EOF ends input only')
    p.add_argument('--uart-send-delay', type=nonnegative_seconds, default=20.0,
                   help='seconds to wait after startup before sending')
    p.add_argument('--uart-send-gap', type=nonnegative_seconds, default=5.0,
                   help='seconds between successive UART payloads')
    p.add_argument('--segment', action='append', type=Path, default=[],
                   help='extra file the plan references (repeatable)')
    p.add_argument('--retries', type=int, default=5, help='SSH reconnection attempts (default: 5)')
    p.add_argument('--retry-delay', type=positive, default=3, help='seconds between SSH retries')
    p.add_argument('--completion-marker', help='finish after this UART text; fail if absent at capture timeout')
    p.add_argument('--startup-timeout', type=positive, default=180,
                   help='maximum loading/startup time in seconds (default: 180)')
    p.add_argument('--baud', type=int, choices=(9600, 19200, 38400, 57600, 115200, 230400, 460800),
                   default=115200)
    p.add_argument('--ssh-host', default=os.environ.get('FPGA_SSH_HOST', 'fpga'))
    p.add_argument('--remote-dir', default=os.environ.get('FPGA_REMOTE_DIR', DEFAULT_REMOTE_DIR),
                   help='server workdir, relative to SSH login directory unless absolute '
                        '(default: Desktop/fpga-tester-ISCAS)')
    args = p.parse_args(argv)
    if args.resume_run and not re.fullmatch(r'run-[0-9a-f]{16}', args.resume_run): p.error('invalid run id')
    if args.retries < 0: p.error('retries cannot be negative')
    args.image = args.image.expanduser().resolve()
    if not args.image.is_file() or args.image.stat().st_size == 0:
        p.error('image must be an existing, nonempty file')
    if args.segment and args.layout_plan is None:
        p.error('--segment requires --layout-plan')
    if args.resume_run and (args.layout_plan or args.segment or args.uart_send):
        p.error('--resume-run only reattaches; omit upload and UART send options')
    if any('\x00' in value for value in args.uart_send):
        p.error('UART payloads cannot contain NUL command-line bytes')
    if args.uart_send and (args.uart_send_delay +
                          (len(args.uart_send)-1)*args.uart_send_gap >= args.capture_seconds):
        p.error('UART send schedule must finish before --capture-seconds')
    if args.layout_plan:
        args.layout_plan = args.layout_plan.expanduser().resolve()
        args.segment = [path.expanduser().resolve() for path in args.segment]
        try:
            args.layout_manifest = layout_manifest(args)
        except (OSError, ValueError) as error:
            p.error(str(error))
    if args.completion_marker is not None and (not args.completion_marker or len(args.completion_marker) > 256 or any(c in args.completion_marker for c in '\r\n\x00')):
        p.error('completion marker must contain 1..256 characters on one line')
    if not args.ssh_host or args.ssh_host.startswith('-'):
        p.error('invalid SSH host')
    if not args.remote_dir or '\x00' in args.remote_dir or '\n' in args.remote_dir:
        p.error('invalid remote directory')
    if args.remote_dir.startswith('~'):
        p.error('use a login-relative remote directory without ~/ or an absolute path')
    return args


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def layout_manifest(args):
    """Validate every upload name before the platform can create readbacks.

    The platform loader accepts arbitrary paths; this launcher deliberately
    confines each segment and its readback to the newly allocated run directory.
    """
    plan = tomllib.loads(args.layout_plan.read_text())
    reserved = {'runner.py', 'run-manifest.json', 'image.bin', 'result.json',
                'uart.raw.log', 'worker.log', 'uvhs.log', 'started.json',
                'worker.pid', 'stop', 'start.lock', 'query.tcl', 'preflight.log', 'input'}
    if (not re.fullmatch(r'[A-Za-z0-9_.-]+', args.layout_plan.name) or
            args.layout_plan.name in reserved or
            args.layout_plan.name.endswith(('.readback', '.upload', '.tmp'))):
        raise ValueError('layout plan name collides with a runner output')
    files = {'image.bin': args.image}
    names = reserved | {args.layout_plan.name}
    for path in args.segment:
        if (not re.fullmatch(r'[A-Za-z0-9_.-]+', path.name) or path.name in names or
                path.name.endswith(('.readback', '.upload', '.tmp'))):
            raise ValueError(f'segment name collision: {path.name}')
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f'segment must be an existing, nonempty file: {path}')
        files[path.name] = path
        names.add(path.name)
    if plan.get('version') != 1 or not isinstance(plan.get('segments'), list):
        raise ValueError('expected a version 1 DDR plan with segments')
    sections = []
    seen = set()
    for segment in plan['segments']:
        name = segment.get('file')
        if name not in files or name in seen:
            raise ValueError(f'plan segment must name one distinct uploaded basename: {name!r}')
        source = files[name]
        sha = digest(source)
        if segment.get('size') != source.stat().st_size or segment.get('sha256') != sha:
            raise ValueError(f'plan size/SHA256 mismatch: {name}')
        seen.add(name)
        sections.append({'file': name, 'size': source.stat().st_size, 'sha256': sha})
    if seen != set(files):
        raise ValueError('plan must reference image.bin and every uploaded segment exactly once')
    return {'plan': args.layout_plan.name, 'plan_sha256': digest(args.layout_plan),
            'segments': sections}


class PendingInput:
    """Persist UUIDs before SSH, so a lost acknowledgement is safe to retry."""
    def __init__(self, path):
        self.path = path
        self.jobs = json.loads(path.read_text()) if path.exists() else []

    def save(self):
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.jobs, indent=2)+'\n')
        temporary.replace(self.path)

    def append(self, data):
        self.jobs.append({'id': uuid.uuid4().hex, 'hex': data.hex()})
        self.save()

    def accept(self, answer):
        job = self.jobs[0]
        sha = hashlib.sha256(bytes.fromhex(job['hex'])).hexdigest()
        if not answer.get('accepted') or answer.get('id') != job['id'] or answer.get('sha256') != sha:
            raise ValueError('UART queue acknowledgement identity mismatch')
        self.jobs.pop(0)
        self.save()


def main(argv=None):
    args = parse_args(argv)
    run_id = args.resume_run or 'run-' + uuid.uuid4().hex[:16]
    # Commands first cd to the configured server workdir. Keep run paths
    # relative to it: no server username or machine-specific root is embedded.
    remote_dir = 'fpga-runs/' + run_id
    remote_display = args.ssh_host + ':' + args.remote_dir.rstrip('/') + '/' + remote_dir
    local_dir = Path(__file__).resolve().parents[1] / 'build' / 'fpga-runs' / run_id
    local_dir.mkdir(parents=True,exist_ok=bool(args.resume_run))
    ssh = ['ssh', *SSH_OPTIONS, args.ssh_host]
    ssh_env = dict(os.environ, LC_ALL='C', LANG='C')
    sha = digest(args.image)
    uploaded = False
    hardware_requested = False

    def command(words):
        return 'cd -- ' + shlex.quote(args.remote_dir) + ' && ' + shlex.join(words)

    def log(message):
        print('[fpga_run] ' + message, file=sys.stderr, flush=True)

    def remote(words, **kwargs):
        for attempt in range(args.retries + 1):
            # Upload/read retries always start from byte zero. Runs themselves
            # are started once by a server-side reservation and never retried.
            for stream_name in ('stdin', 'stdout'):
                stream = kwargs.get(stream_name)
                if hasattr(stream, 'seek'):
                    stream.seek(0)
                    if stream_name == 'stdout': stream.truncate()
            result = subprocess.run([*ssh, command(words)], env=ssh_env, **kwargs)
            if result.returncode == 0:
                return result
            if result.returncode != 255 or attempt == args.retries:
                raise subprocess.CalledProcessError(result.returncode, words)
            log(f'SSH unavailable; reconnecting {attempt+1}/{args.retries} in {args.retry_delay}s')
            time.sleep(args.retry_delay)

    # Create only real directories below the approved workdir. Existing
    # Makefile/user_script/hw.dat symlinks are only read by the platform tools.
    prepare = '''from pathlib import Path
import sys, shutil
import time
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
'''
    try:
        log(f'FPGA{args.fpga}: {args.image} ({args.image.stat().st_size} bytes)')
        log(f'SHA256 {sha}')
        log(f'Server workdir: {args.ssh_host}:{args.remote_dir}')
        if args.resume_run:
            verify = """from pathlib import Path
import hashlib,json,sys
p=Path(sys.argv[1]);state=json.loads((p/'started.json').read_text())
if state['sha256']!=sys.argv[2] or state['fpga']!=int(sys.argv[3]):raise SystemExit('resume identity mismatch')
if hashlib.sha256((p/'image.bin').read_bytes()).hexdigest()!=sys.argv[2]:raise SystemExit('resume image mismatch')
"""
            remote(['python3','-B','-c',verify,remote_dir,sha,str(args.fpga)])
            log('Resuming the existing run; no upload or hardware restart.')
        else:
            remote(['python3', '-B', '-c', prepare, run_id, str(upload_disk_budget(args))])
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
            extra = []
            if args.layout_plan is not None:
                extra.append((args.layout_plan, args.layout_plan.name))
                for segment in args.segment:
                    extra.append((segment, segment.name))
            for source, name in ((Path(__file__).with_name('fpga_remote.py'), 'runner.py'),
                                 (args.image, 'image.bin'), *extra):
                with source.open('rb') as stream:
                    remote(['python3', '-B', '-c', upload, remote_dir+'/'+name, digest(source)], stdin=stream)
            if args.layout_plan is not None:
                manifest = local_dir / 'run-manifest.json'
                manifest.write_text(json.dumps(args.layout_manifest, indent=2)+'\n')
                with manifest.open('rb') as stream:
                    remote(['python3', '-B', '-c', upload,
                            remote_dir+'/run-manifest.json', digest(manifest)], stdin=stream)
        uploaded = True
        if not args.resume_run: log('Upload complete. Opening UART and loading the image.')
        words = ['python3', '-u', '-B', remote_dir + '/runner.py',
                 '--fpga', str(args.fpga), '--sha256', sha,
                 '--capture-seconds', str(args.capture_seconds),
                 '--startup-timeout', str(args.startup_timeout), '--baud', str(args.baud)]
        if args.completion_marker:
            words += ['--completion-marker', args.completion_marker]
        if args.layout_plan is not None:
            words += ['--layout', args.layout_plan.name]
        # A detached worker owns UART/UVHS for at most startup+capture time.
        # SSH relays are read-only; reconnecting cannot reload/reset the FPGA.
        hardware_requested = True
        # The payloads have to be appended BEFORE the worker is launched. They were
        # appended after the detach call at first, so the worker never received
        # them and the board sat at its prompt for the whole capture window.
        #
        # They travel with the worker invocation because the worker owns the UART
        # exclusively (TIOCEXCL); a second process cannot open the device.
        if args.uart_send:
            for payload in args.uart_send:
                words += ['--uart-input='+payload]
            words += ['--uart-input-delay', str(args.uart_send_delay)]
            words += ['--uart-input-gap', str(args.uart_send_gap)]
        if not args.resume_run: remote([*words, '--detach'])
        status = 1
        process = None
        pending_input = PendingInput(local_dir/'input-pending.json')
        stdin_fd = sys.stdin.fileno() if args.interactive else None
        next_input_attempt = 0.0
        if args.interactive:
            log('Interactive stdin enabled. Wait for the FPGA prompt; EOF stops input, Ctrl-C stops this run.')
        elif pending_input.jobs:
            log('Pending UART input retained; use --resume-run with --interactive to retry the same request IDs.')
        try:
            with (local_dir / 'uart.raw.log').open('ab' if args.resume_run else 'wb') as uart:
                for attempt in range(args.retries + 1):
                    relay = ['python3', '-u', '-B', remote_dir+'/runner.py', '--relay', str(uart.tell())]
                    process = subprocess.Popen([*ssh, command(relay)], stdin=subprocess.DEVNULL,
                                               stdout=subprocess.PIPE, env=ssh_env, start_new_session=True)
                    try:
                        while True:
                            watched = [process.stdout.fileno()]
                            if stdin_fd is not None: watched.append(stdin_fd)
                            ready, _, _ = select.select(watched, [], [], 0.1)
                            if stdin_fd is not None and stdin_fd in ready:
                                data = os.read(stdin_fd, 4096)
                                if data: pending_input.append(data)
                                else: stdin_fd = None
                            if (args.interactive and pending_input.jobs and
                                    time.monotonic() >= next_input_attempt):
                                job = pending_input.jobs[0]
                                try:
                                    response = remote(['python3', '-u', '-B', remote_dir+'/runner.py',
                                                       '--enqueue', job['id'], '--input-hex', job['hex']],
                                                      capture_output=True, timeout=45)
                                    pending_input.accept(json.loads(response.stdout))
                                    log('UART input queued; worker acknowledgement remains separate from FPGA receipt.')
                                except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                                        ValueError) as error:
                                    log('UART input delivery unconfirmed; keeping request UUID for retry: '+str(error))
                                    next_input_attempt = time.monotonic() + args.retry_delay
                            if process.stdout.fileno() in ready:
                                block = os.read(process.stdout.fileno(), 4096)
                                if not block: break
                                uart.write(block); uart.flush()
                                sys.stdout.buffer.write(block); sys.stdout.buffer.flush()
                        status = process.wait()
                    finally:
                        process.stdout.close()
                    if (status != 255 and status >= 0) or attempt == args.retries: break
                    log(f'SSH disconnected; resuming UART at byte {uart.tell()} ({attempt+1}/{args.retries})')
                    time.sleep(args.retry_delay)
        except (KeyboardInterrupt, BrokenPipeError):
            log('Stopping this run and releasing UART/FPGA...')
            if process is not None and process.poll() is None:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
            # STOP is persistent and independent of a broken relay connection.
            try:
                remote(['python3', '-B', '-c',
                        "from pathlib import Path; import sys; Path(sys.argv[1]).touch()",
                        remote_dir+'/stop'])
            except subprocess.CalledProcessError:
                log('Stop request could not reach server; worker remains bounded by its configured timeouts')
            status = 130
        if status == 255 or status < 0:
            log('Reconnection limit reached; the bounded worker retains server logs and may still be running')
            log(f'Reattach with the same image and --fpga={args.fpga} --resume-run={run_id}')
        if status == 130:
            # STOP is asynchronous: allow the worker to release UVHS/UART and
            # atomically publish its result before retrieving the final logs.
            wait_result = """from pathlib import Path
import sys,time
p=Path(sys.argv[1]);deadline=time.monotonic()+25
while not p.exists() and time.monotonic()<deadline:time.sleep(0.1)
if not p.exists():raise SystemExit('worker cleanup has not finished; logs remain on server')
"""
            try:
                remote(['python3','-B','-c',wait_result,remote_dir+'/result.json'])
            except subprocess.CalledProcessError:
                log('Worker cleanup is still pending; inspect the remote run directory')
        # Logs remain available remotely even if SSH was interrupted. Fetching
        # logs does not retry the upload or execute the FPGA program again.
        for name in ('result.json', 'uvhs.log', 'worker.log'):
            try:
                with (local_dir / name).open('wb') as f:
                    remote(['cat', remote_dir + '/' + name], stdout=f)
            except subprocess.CalledProcessError:
                log(f'Could not retrieve {name}; inspect {remote_display}')
        result_file=local_dir/'result.json'
        if status == 0:
            try:
                result=json.loads(result_file.read_text())
                if result.get('status')!='OK' or result.get('sha256')!=sha:
                    status=1
            except (OSError,ValueError): status=1
        if args.interactive and pending_input.jobs:
            log('Unconfirmed input requests saved in '+str(pending_input.path))
            if status == 0: status = 1
        log(f'Local logs: {local_dir}')
        log(f'Remote logs: {remote_display}')
        return status
    except KeyboardInterrupt:
        log('Interrupted.')
        if hardware_requested:
            try:
                remote(['python3', '-B', '-c', "from pathlib import Path; import sys; Path(sys.argv[1]).touch()", remote_dir+'/stop'])
            except (subprocess.CalledProcessError, KeyboardInterrupt):
                log('Stop request could not reach server; worker remains bounded by its configured timeouts')
        return 130
    except (OSError, ValueError, subprocess.CalledProcessError) as e:
        log(str(e))
        if uploaded:
            log(f'Remote files: {remote_display}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
