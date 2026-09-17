"""Reconnect reads UART by offset and never launches a second FPGA worker."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import fpga_remote as remote

class ReconnectTests(unittest.TestCase):
    def test_detached_start_is_idempotent(self):
        cwd=Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'run';run.mkdir()
            runner=run/'runner.py';runner.write_text('')
            args=SimpleNamespace(fpga=5,sha256='1'*64,capture_seconds=10,startup_timeout=10,baud=115200,completion_marker='DONE',
                                 layout='model.plan', uart_input=['-hello\n', '你好\n'],
                                 uart_input_delay=2.0, uart_input_gap=0.5)
            try:
                os.chdir(root)
                with patch.object(remote,'__file__',str(runner)),patch.object(remote.subprocess,'Popen') as spawn:
                    spawn.return_value.pid=123
                    self.assertEqual(remote.detached_start(args),0)
                    self.assertEqual(remote.detached_start(args),0)
                    spawn.assert_called_once()
                    command = spawn.call_args.args[0]
                    self.assertEqual(command[command.index('--layout')+1], 'model.plan')
                    self.assertIn('--uart-input=-hello\n', command)
                    self.assertIn('--uart-input=你好\n', command)
                    self.assertEqual(command[command.index('--uart-input-delay')+1], '2.0')
                    self.assertEqual(command[command.index('--uart-input-gap')+1], '0.5')
                    args.sha256='2'*64
                    with self.assertRaises(RuntimeError):remote.detached_start(args)
            finally:os.chdir(cwd)

    def test_relay_resumes_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);runner=root/'runner.py';runner.write_text('')
            data=b'first line\r\nlast line\r\n'
            (root/'uart.raw.log').write_bytes(data)
            (root/'result.json').write_text(json.dumps({'status':'OK'}))
            stream=io.TextIOWrapper(io.BytesIO(),encoding='utf-8')
            with patch.object(remote,'__file__',str(runner)),contextlib.redirect_stdout(stream):
                status=remote.relay(7)
            stream.flush()
            self.assertEqual(status,0)
            self.assertEqual(stream.buffer.getvalue(),data[7:])
            stream.close()

    def test_relay_retains_hardware_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);runner=root/'runner.py';runner.write_text('')
            (root/'result.json').write_text(json.dumps({'status':'ERROR','error':'numeric failure'}))
            with patch.object(remote,'__file__',str(runner)),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(remote.relay(0),1)
if __name__=='__main__':unittest.main()
