"""Configuration and relocation tests; fake SSH executes only in temp folders."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import fpga_run as launcher


FAKE_SSH = '''#!/usr/bin/env python3
import hashlib, json, os, shlex, subprocess, sys
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
    expected=argv[argv.index('--sha256')+1]
    assert hashlib.sha256(image.read_bytes()).hexdigest()==expected
    assert argv[argv.index('--fpga')+1]=='5'
    # Exercise stdout streaming and local retrieval without opening hardware.
    (runner.parent/'result.json').write_text(json.dumps({'status':'OK','sha256':expected}))
    (runner.parent/'uvhs.log').write_text('fake load completed\\n')
    sys.stdout.buffer.write(b'Hello from fake UART!\\r\\n')
else:
    sys.exit(subprocess.call(command,shell=True,executable='/bin/bash',cwd=home))
'''


class ConfigurationTests(unittest.TestCase):
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
                with self.subTest(directory=directory):
                    result=subprocess.run([str(moved/'fpga_run.sh'),image.name,'--fpga=5',
                                           '--remote-dir='+directory],cwd=root,env=env,
                                          capture_output=True,timeout=15)
                    self.assertEqual(result.returncode,0,result.stderr.decode())
                    self.assertEqual(result.stdout,b'Hello from fake UART!\r\n')
            runs=list((work/'fpga-runs').glob('run-*'))
            self.assertEqual(len(runs),2)
            for run in runs:
                self.assertEqual((run/'image.bin').read_bytes(),image.read_bytes())
                local=moved/'build/fpga-runs'/run.name
                self.assertEqual((local/'uart.raw.log').read_bytes(),b'Hello from fake UART!\r\n')
                self.assertEqual(json.loads((local/'result.json').read_text())['status'],'OK')
            self.assertFalse(list(root.rglob('BAD')))


if __name__=='__main__':
    unittest.main()
