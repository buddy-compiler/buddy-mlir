#!/usr/bin/env python3
"""Upload an NR image and relay its UART through one managed SSH session."""
import argparse
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

DEFAULT_REMOTE_DIR = 'Desktop/fpga-tester-ISCAS'
SSH_OPTIONS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
               '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2']


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
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
    p.add_argument('--startup-timeout', type=positive, default=180,
                   help='maximum loading/startup time in seconds (default: 180)')
    p.add_argument('--baud', type=int, choices=(9600, 19200, 38400, 57600, 115200, 230400, 460800),
                   default=115200)
    p.add_argument('--ssh-host', default=os.environ.get('FPGA_SSH_HOST', 'fpga'))
    p.add_argument('--remote-dir', default=os.environ.get('FPGA_REMOTE_DIR', DEFAULT_REMOTE_DIR),
                   help='server workdir, relative to SSH login directory unless absolute '
                        '(default: Desktop/fpga-tester-ISCAS)')
    args = p.parse_args(argv)
    args.image = args.image.expanduser().resolve()
    if not args.image.is_file() or args.image.stat().st_size == 0:
        p.error('image must be an existing, nonempty file')
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


def main(argv=None):
    args = parse_args(argv)
    run_id = 'run-' + uuid.uuid4().hex[:16]
    # Commands first cd to the configured server workdir. Keep run paths
    # relative to it: no server username or machine-specific root is embedded.
    remote_dir = 'fpga-runs/' + run_id
    remote_display = args.ssh_host + ':' + args.remote_dir.rstrip('/') + '/' + remote_dir
    local_dir = Path(__file__).resolve().parents[1] / 'build' / 'fpga-runs' / run_id
    local_dir.mkdir(parents=True)
    ssh = ['ssh', *SSH_OPTIONS, args.ssh_host]
    ssh_env = dict(os.environ, LC_ALL='C', LANG='C')
    sha = digest(args.image)
    uploaded = False

    def command(words):
        return 'cd -- ' + shlex.quote(args.remote_dir) + ' && ' + shlex.join(words)

    def log(message):
        print('[fpga_run] ' + message, file=sys.stderr, flush=True)

    def remote(words, **kwargs):
        return subprocess.run([*ssh, command(words)], check=True, env=ssh_env, **kwargs)

    # Create only real directories below the approved workdir. Existing
    # Makefile/user_script/hw.dat symlinks are only read by the platform tools.
    prepare = '''from pathlib import Path
import sys
root=Path.cwd()
runs=root/'fpga-runs'
runs.mkdir(exist_ok=True)
if runs.is_symlink() or runs.resolve().parent != root:
    raise SystemExit('fpga-runs must be a real directory inside the workdir')
(runs/sys.argv[1]).mkdir(mode=0o700)
'''
    try:
        log(f'FPGA{args.fpga}: {args.image} ({args.image.stat().st_size} bytes)')
        log(f'SHA256 {sha}')
        log(f'Server workdir: {args.ssh_host}:{args.remote_dir}')
        remote(['python3', '-B', '-c', prepare, run_id])
        for source, name in ((Path(__file__).with_name('fpga_remote.py'), 'runner.py'),
                             (args.image, 'image.bin')):
            # The private, freshly created directory and noclobber avoid
            # overwriting existing files or following destination symlinks.
            destination = remote_dir + '/' + name
            with source.open('rb') as f:
                remote(['sh', '-c', 'set -C; umask 077; cat > ' + shlex.quote(destination)], stdin=f)
        uploaded = True
        log('Upload complete. Opening UART and loading the image.')
        words = ['python3', '-u', '-B', remote_dir + '/runner.py',
                 '--fpga', str(args.fpga), '--sha256', sha,
                 '--capture-seconds', str(args.capture_seconds),
                 '--startup-timeout', str(args.startup_timeout), '--baud', str(args.baud)]
        # Keep SSH out of the terminal's foreground process group so Ctrl-C
        # reaches us first. STOP/EOF tells the remote runner to exit gracefully.
        process = subprocess.Popen([*ssh, command(words)], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, env=ssh_env, start_new_session=True)
        status = 1
        try:
            with (local_dir / 'uart.raw.log').open('wb') as uart:
                while True:
                    block = os.read(process.stdout.fileno(), 4096)
                    if not block:
                        break
                    uart.write(block)
                    uart.flush()
                    sys.stdout.buffer.write(block)
                    sys.stdout.buffer.flush()
            status = process.wait()
        except (KeyboardInterrupt, BrokenPipeError):
            log('Stopping this run and releasing UART/FPGA...')
            try:
                process.stdin.write(b'STOP\n')
                process.stdin.flush()
                # Closing our stdout reader also makes remote output fail fast.
                process.stdout.close()
                process.wait(timeout=25)
            except (BrokenPipeError, subprocess.TimeoutExpired, KeyboardInterrupt):
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            status = 130
        finally:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
            process.stdout.close()
        # Logs remain available remotely even if SSH was interrupted. Fetching
        # logs does not retry the upload or execute the FPGA program again.
        for name in ('result.json', 'uvhs.log'):
            try:
                with (local_dir / name).open('wb') as f:
                    remote(['cat', remote_dir + '/' + name], stdout=f)
            except subprocess.CalledProcessError:
                log(f'Could not retrieve {name}; inspect {remote_display}')
        log(f'Local logs: {local_dir}')
        log(f'Remote logs: {remote_display}')
        return status
    except KeyboardInterrupt:
        log('Interrupted.')
        return 130
    except (OSError, subprocess.CalledProcessError) as e:
        log(str(e))
        if uploaded:
            log(f'Remote files: {remote_display}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
