"""No hardware: exercise UART/PTY lifecycle, image provenance and failures."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import pty
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import fpga_remote as remote


FAKE_MAKE = '''#!/usr/bin/env python3
import os, pathlib, sys, time
image=pathlib.Path(next(a[5:] for a in sys.argv if a.startswith('test=')))
padded=image.with_name('image_padded.bin')
data=image.read_bytes(); data+=bytes((-len(data))%64)
padded.write_bytes(data)
readback=pathlib.Path(str(padded)+'.readback')
mode=os.environ['FPGA_TEST_MODE']
readback.write_bytes(data if mode!='bad_readback' else b'corrupt')
print('UV_RUN_IMAGE='+str(padded),flush=True)
if mode=='error':
    print('[12:00:00] [RTM-1] ERROR: simulated loader failure',flush=True)
print('reset -name cpu_reset -value 0 success',flush=True)
image.with_name('trigger').touch()
time.sleep(.05)
if mode!='timeout': print('hspRun>',flush=True)
for line in sys.stdin:
    if line.strip()=='exit': break
print('    Total ERROR:  0  suppressed:  0',flush=True)
'''


class RunnerTests(unittest.TestCase):
    def test_resource_table(self):
        remote.board_available(b'uvhs-0 B1 F1 on link down false\n', 5)
        for row in (b'uvhs-0 B1 F1 on link up hjuser false\n',
                    b'uvhs-0 B1 F1 on link down true\n', b'no table'):
            with self.assertRaises(RuntimeError):
                remote.board_available(row, 5)

    def test_error_detection(self):
        self.assertFalse(remote.ERROR.search(b'    Total ERROR:  0 suppressed: 0\n'))
        self.assertTrue(remote.ERROR.search(b'[12:00] [CMN-1] ERROR: failed\n'))
        self.assertTrue(remote.ERROR.search(b'    Total FATAL:  1 suppressed: 0\n'))

    def test_symlink_output_rejected(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            root=Path(a); run=root/'run'; run.mkdir()
            (root/'cmd').symlink_to(b)
            with self.assertRaises(RuntimeError):
                remote.check_paths(root,run)

    def simulate(self, mode, payload=b'Hello, World!\r\nverify hello: PASS\r\n'):
        cwd=Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); run=root/'run'; run.mkdir()
            image=run/'image.bin'; image.write_bytes(b'new image\x00'*9)
            bindir=root/'bin'; bindir.mkdir()
            make=bindir/'make'; make.write_text(FAKE_MAKE); make.chmod(0o755)
            serial_master,serial_slave=pty.openpty()
            device=os.ttyname(serial_slave)
            control_r,control_w=os.pipe()
            args=SimpleNamespace(fpga=5,sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                                 capture_seconds=.15,startup_timeout=.3,baud=115200)
            if mode=='hash_mismatch': args.sha256='0'*64
            producer_error=[]
            def producer():
                deadline=time.monotonic()+2
                while not (run/'trigger').exists() and time.monotonic()<deadline:
                    time.sleep(.005)
                if (run/'trigger').exists():
                    try:
                        if mode=='interrupt': os.write(control_w,b'STOP\n')
                        elif mode!='empty': os.write(serial_master,payload)
                    except OSError as error:
                        producer_error.append(error)
            thread=threading.Thread(target=producer)
            thread.start()
            stdout=io.TextIOWrapper(io.BytesIO(),encoding='utf-8')
            try:
                with patch.dict(os.environ,PATH=str(bindir)+':'+os.environ['PATH'],FPGA_TEST_MODE=mode), \
                     patch.object(remote,'check_processes'),patch.object(remote,'preflight') as preflight, \
                     contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(io.StringIO()):
                    status=remote.run(args,root=root,run_dir=run,device=device,control_fd=control_r)
                result=json.loads((run/'result.json').read_text())
                stdout.flush(); output=stdout.buffer.getvalue()
                # TIOCEXCL must be released even on errors or interruption.
                reopened=os.open(device,os.O_RDWR|os.O_NOCTTY); os.close(reopened)
                if mode=='hash_mismatch': preflight.assert_not_called()
                return status,result,output
            finally:
                os.chdir(cwd)
                thread.join(timeout=3)
                for fd in (serial_master,serial_slave,control_r,control_w): os.close(fd)
                stdout.close()
                self.assertFalse(thread.is_alive())
                self.assertFalse(producer_error)

    def test_first_uart_bytes_and_readback(self):
        status,result,output=self.simulate('ok')
        self.assertEqual(status,0)
        self.assertTrue(result['ddr_readback_matches'])
        self.assertEqual(output,b'Hello, World!\r\nverify hello: PASS\r\n')

    def test_corrupt_readback(self):
        status,result,_=self.simulate('bad_readback')
        self.assertNotEqual(status,0)
        self.assertIn('DDR readback differs',result['error'])

    def test_upload_mismatch_prevents_hardware_access(self):
        status,result,_=self.simulate('hash_mismatch')
        self.assertNotEqual(status,0)
        self.assertIn('SHA256 mismatch',result['error'])

    def test_no_uart_is_not_success(self):
        status,result,_=self.simulate('empty')
        self.assertNotEqual(status,0)
        self.assertIn('no UART output',result['error'])

    def test_loader_error_even_if_make_exits_zero(self):
        status,result,_=self.simulate('error')
        self.assertNotEqual(status,0)
        self.assertIn('ERROR/FATAL',result['error'])

    def test_program_failure(self):
        status,result,_=self.simulate('ok',b'verify hello: FAIL\r\n')
        self.assertNotEqual(status,0)
        self.assertIn('verification failure',result['error'])

    def test_timeout(self):
        status,result,_=self.simulate('timeout')
        self.assertNotEqual(status,0)
        self.assertIn('startup timed out',result['error'])

    def test_interrupt_does_not_kill_unrelated_process(self):
        unrelated=subprocess.Popen(['sleep','30'],start_new_session=True)
        try:
            status,result,_=self.simulate('interrupt')
            self.assertEqual(status,130)
            self.assertEqual(result['status'],'INTERRUPTED')
            self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate();unrelated.wait()


if __name__=='__main__':
    unittest.main()
