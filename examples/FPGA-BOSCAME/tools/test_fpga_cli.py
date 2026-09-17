"""Configuration and relocation tests; fake SSH executes only in temp folders."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import fpga_run as launcher


FAKE_SSH = '''#!/usr/bin/env python3
import hashlib, json, os, shlex, subprocess, sys, time
from pathlib import Path
command=sys.argv[-1]
words=shlex.split(command)
assert words[:2]==['cd','--'] and words[3]=='&&',words
home=Path(os.environ['FAKE_SSH_HOME'])
root=(home/words[2]).resolve()
argv=words[4:]
if argv[:3]==['python3','-u','-B']:
    runner=root/argv[3]
    assert runner.is_file()
    image=runner.parent/'image.bin'
    if '--enqueue' in argv:
        rid=argv[argv.index('--enqueue')+1]
        data=bytes.fromhex(argv[argv.index('--input-hex')+1])
        request=runner.parent/('input-'+rid)
        first=not request.exists()
        if first:
            request.write_bytes(data)
            with (runner.parent/'uart.raw.log').open('ab') as stream:stream.write(data)
            state=json.loads((runner.parent/'started.json').read_text())
            (runner.parent/'result.json').write_text(json.dumps({'status':'OK','sha256':state['sha256']}))
        else: assert request.read_bytes()==data
        if first and os.environ.get('FAKE_DROP_ENQUEUE_ACK'):sys.exit(255)
        print(json.dumps({'accepted':True,'id':rid,'sha256':hashlib.sha256(data).hexdigest()}))
    elif '--detach' in argv:
        expected=argv[argv.index('--sha256')+1]
        assert hashlib.sha256(image.read_bytes()).hexdigest()==expected
        assert argv[argv.index('--fpga')+1]=='5'
        if not os.environ.get('FAKE_INTERACTIVE'):
            (runner.parent/'result.json').write_text(json.dumps({'status':'OK','sha256':expected}))
        (runner.parent/'started.json').write_text(json.dumps({'sha256':expected,'fpga':5}))
        (runner.parent/'uvhs.log').write_text('fake load completed\\n')
        (runner.parent/'worker.log').write_text('worker done\\n')
        (runner.parent/'uart.raw.log').write_bytes(b'Hello from fake UART!\\r\\n')
        counter=runner.parent/'launches'
        counter.write_text(str(int(counter.read_text())+1) if counter.exists() else '1')
    else:
        assert '--relay' in argv
        offset=int(argv[argv.index('--relay')+1])
        if os.environ.get('FAKE_INTERACTIVE'):
            deadline=time.monotonic()+12
            while not (runner.parent/'result.json').exists() and time.monotonic()<deadline:time.sleep(.05)
        payload=(runner.parent/'uart.raw.log').read_bytes()
        if os.environ.get('FAKE_DISCONNECT') and offset==0:
            sys.stdout.buffer.write(payload[:7]);sys.stdout.buffer.flush();sys.exit(255)
        sys.stdout.buffer.write(payload[offset:])

else:
    sys.exit(subprocess.call(command,shell=True,executable='/bin/bash',cwd=home))
'''


class ConfigurationTests(unittest.TestCase):
    def test_disk_budget_includes_upload_padding_and_all_segment_readbacks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            boot, weight = root/'image.bin', root/'weights.bin'
            boot.write_bytes(b'x' * 65)
            weight.write_bytes(b'y' * 128)
            args = SimpleNamespace(image=boot, layout_plan=None, segment=[])
            self.assertEqual(launcher.upload_disk_budget(args), 65 + 2*1048576 + launcher.DISK_RESERVE_BYTES)
            args.layout_plan, args.segment = root/'plan', [weight]
            self.assertEqual(launcher.upload_disk_budget(args), 2*(65+128) + launcher.DISK_RESERVE_BYTES)

    def test_interactive_stdin_enqueue_ack_loss_is_not_duplicate_input(self):
        package = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); moved=root/'package'; (moved/'tools').mkdir(parents=True)
            shutil.copy2(package/'fpga_run.sh', moved/'fpga_run.sh')
            for name in ('fpga_run.py', 'fpga_remote.py'):
                shutil.copy2(package/'tools'/name, moved/'tools'/name)
            work=root/'platform'; work.mkdir()
            tools=root/'bin'; tools.mkdir()
            ssh=tools/'ssh'; ssh.write_text(FAKE_SSH); ssh.chmod(0o755)
            image=root/'image.bin'; image.write_bytes(b'image')
            env=dict(os.environ, FAKE_SSH_HOME=str(root), FAKE_INTERACTIVE='1',
                     FAKE_DROP_ENQUEUE_ACK='1', PATH=str(tools)+':'+os.environ['PATH'])
            payload='你好\n'.encode()
            result=subprocess.run([str(moved/'fpga_run.sh'),str(image),'--fpga=5',
                                   '--interactive','--remote-dir='+str(work),'--retry-delay=1'],
                                  input=payload,capture_output=True,env=env,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr.decode())
            self.assertEqual(result.stdout,b'Hello from fake UART!\r\n'+payload)
            run, = (work/'fpga-runs').glob('run-*')
            request, = run.glob('input-*')
            self.assertEqual(request.read_bytes(),payload)
            self.assertEqual((run/'launches').read_text(),'1')
            pending=moved/'build/fpga-runs'/run.name/'input-pending.json'
            self.assertEqual(json.loads(pending.read_text()),[])

    def test_layout_manifest_rejects_escaping_or_overwriting_uploads(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root/'boot.bin'; image.write_bytes(b'boot')
            weights = root/'weights.bin'; weights.write_bytes(b'weights')
            plan = root/'model.plan'
            def text(weight_name='weights.bin'):
                return ('version = 1\n' + ''.join(
                    '\n[[segments]]\nfile = '+json.dumps(name)+'\nsize = '+str(path.stat().st_size)+
                    '\nsha256 = "'+hashlib.sha256(path.read_bytes()).hexdigest()+'"\n'
                    for name, path in [('image.bin', image), (weight_name, weights)]))
            plan.write_text(text())
            argv = [str(image), '--fpga=5', '--layout-plan='+str(plan), '--segment='+str(weights)]
            args = launcher.parse_args(argv)
            self.assertEqual(len(args.layout_manifest['segments']), 2)
            plan.write_text(text('../weights.bin'))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                launcher.parse_args(argv)
            plan.write_text(text())
            weights.write_bytes(b'changed')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                launcher.parse_args(argv)
            collision = root/'runner.py'; collision.write_bytes(b'oops')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                launcher.parse_args(argv+['--segment='+str(collision)])

    def test_invalid_uart_schedule_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder)/'image.bin'; image.write_bytes(b'image')
            for value in ('nan', 'inf', '-1', '11'):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    launcher.parse_args([str(image), '--fpga=5', '--uart-send=hello',
                                         '--uart-send-delay='+value])

    def test_server_directory_defaults_and_overrides(self):
        with tempfile.TemporaryDirectory() as folder:
            image=Path(folder)/'image.bin'; image.write_bytes(b'image')
            with patch.dict(os.environ, {}, clear=True):
                args=launcher.parse_args([str(image),'--fpga=5'])
                self.assertEqual(args.remote_dir,'Desktop/fpga-tester-ISCAS')
                self.assertEqual(args.ssh_host,'fpga')
            with patch.dict(os.environ, FPGA_REMOTE_DIR='custom/platform', FPGA_SSH_HOST='board'):
                args=launcher.parse_args([str(image),'--fpga=5'])
                self.assertEqual(args.remote_dir,'custom/platform')
                args=launcher.parse_args([str(image),'--fpga=5','--remote-dir=another/platform',
                                          '--ssh-host=other-board'])
                self.assertEqual(args.remote_dir,'another/platform')
                self.assertEqual(args.ssh_host,'other-board')
            for directory in ('','~/platform','line\nbreak'):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    launcher.parse_args([str(image),'--fpga=5','--remote-dir='+directory])

    def test_relocated_launcher_and_shell_quoting(self):
        package=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            moved=root/'relocated package'; (moved/'tools').mkdir(parents=True)
            shutil.copy2(package/'fpga_run.sh',moved/'fpga_run.sh')
            for name in ('fpga_run.py','fpga_remote.py'):
                shutil.copy2(package/'tools'/name,moved/'tools'/name)
            home=root/'remote login'; home.mkdir()
            work=home/"platform ' $(touch BAD) ;"; work.mkdir()
            tools=root/'fake bin';tools.mkdir()
            ssh=tools/'ssh';ssh.write_text(FAKE_SSH);ssh.chmod(0o755)
            image=root/"image ' $(touch BAD).bin";image.write_bytes(b'fresh binary\x00')
            env=dict(os.environ,FAKE_SSH_HOME=str(home),PATH=str(tools)+':'+os.environ['PATH'])
            env.pop('FPGA_SSH_HOST',None)
            # Both login-relative and user-supplied absolute paths must work.
            for directory in (work.name,str(work)):
                env["FAKE_DISCONNECT"]="1"
                with self.subTest(directory=directory):
                    result=subprocess.run([str(moved/'fpga_run.sh'),image.name,'--fpga=5',
                                           '--remote-dir='+directory,'--retry-delay=1'],cwd=root,env=env,
                                          capture_output=True,timeout=15)
                    self.assertEqual(result.returncode,0,result.stderr.decode())
                    self.assertEqual(result.stdout,b'Hello from fake UART!\r\n')
            runs=list((work/'fpga-runs').glob('run-*'))
            self.assertEqual(len(runs),2)
            for run in runs:
                self.assertEqual((run/'image.bin').read_bytes(),image.read_bytes())
                self.assertEqual((run/'launches').read_text(),'1')
                local=moved/'build/fpga-runs'/run.name
                self.assertEqual((local/'uart.raw.log').read_bytes(),b'Hello from fake UART!\r\n')
                self.assertEqual(json.loads((local/'result.json').read_text())['status'],'OK')
                # A separate invocation can recover an interrupted client's log
                # without uploading or launching a second hardware execution.
                (local/'uart.raw.log').write_bytes(b'Hello f')
                resumed=subprocess.run([str(moved/'fpga_run.sh'),str(image),'--fpga=5',
                                        '--remote-dir='+str(work),'--resume-run='+run.name],
                                       cwd=root,env=env,capture_output=True,timeout=15)
                self.assertEqual(resumed.returncode,0,resumed.stderr.decode())
                self.assertEqual(resumed.stdout,b'rom fake UART!\r\n')
                self.assertEqual((local/'uart.raw.log').read_bytes(),b'Hello from fake UART!\r\n')
                self.assertEqual((run/'launches').read_text(),'1')
            self.assertFalse(list(root.rglob('BAD')))


if __name__=='__main__':
    unittest.main()
