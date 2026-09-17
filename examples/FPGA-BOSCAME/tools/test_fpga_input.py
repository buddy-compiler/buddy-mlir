"""UART input survives SSH retries; only one existing worker owns the device."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import pty
import tempfile
import unittest
from unittest.mock import patch

import fpga_remote as remote
import fpga_run as launcher


class InputQueueTests(unittest.TestCase):
    def test_deduplicate_lost_enqueue_ack_and_preserve_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sender = remote.InputQueue(root)
            first = sender.enqueue('a'*32, '你好\n'.encode())
            self.assertEqual(first, remote.InputQueue(root).enqueue('a'*32, '你好\n'.encode()))
            second = sender.enqueue('b'*32, b'next\n')
            self.assertLess(first['sequence'], second['sequence'])
            self.assertEqual(len(sender.requests()), 2)
            with self.assertRaisesRegex(RuntimeError, 'different input'):
                sender.enqueue('a'*32, b'changed')
            with patch.object(remote.os, 'write', side_effect=[2, BlockingIOError(), 5, 5]) as write:
                sender.pump(10)
                sender.pump(10)
                sender.pump(10)
                sender.pump(10)
                self.assertEqual(write.call_args_list[1].args[1], '你好\n'.encode()[2:])
            self.assertFalse(sender.pending())
            self.assertEqual(sender.sent_bytes, 12)
            # A client reconnect only retries enqueue; an acknowledged job never
            # returns to the worker's pending set, even after object recreation.
            again = remote.InputQueue(root)
            self.assertEqual(again.enqueue('a'*32, '你好\n'.encode()), first)
            with patch.object(remote.os, 'write') as write:
                again.pump(10)
                write.assert_not_called()
            ack = json.loads((root/'input'/('a'*32+'.ack.json')).read_text())
            self.assertEqual(ack['bytes_written'], 7)
            self.assertIn('FPGA receipt unconfirmed', ack['meaning'])

    def test_real_pty_byte_transfer_with_embedded_nul(self):
        with tempfile.TemporaryDirectory() as temporary:
            master, slave = pty.openpty()
            try:
                queue = remote.InputQueue(Path(temporary))
                data = b'\x00UTF8:\xe4\xbd\xa0\xe5\xa5\xbd'
                queue.enqueue('c'*32, data)
                queue.pump(slave)
                self.assertEqual(os.read(master, 4096), data)
                self.assertFalse(queue.pending())
            finally:
                os.close(master); os.close(slave)

    def test_closed_session_rejects_new_input_but_allows_idempotent_reply(self):
        with tempfile.TemporaryDirectory() as temporary:
            queue = remote.InputQueue(Path(temporary))
            job = queue.enqueue('d'*32, b'hello')
            self.assertEqual(queue.close(), 1)
            self.assertEqual(queue.enqueue('d'*32, b'hello'), job)
            with self.assertRaisesRegex(RuntimeError, 'finished'):
                queue.enqueue('e'*32, b'new')

    def test_symlinks_cannot_redirect_queue_or_ack_files(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            root, outside = Path(a), Path(b)
            (root/'input').symlink_to(outside)
            with self.assertRaisesRegex(RuntimeError, 'real directory'):
                remote.InputQueue(root).enqueue('a'*32, b'input')
            self.assertFalse(list(outside.iterdir()))
            (root/'input').unlink()
            queue = remote.InputQueue(root)
            queue.enqueue('a'*32, b'input')
            (root/'input'/('a'*32+'.ack.json')).symlink_to(outside/'fake')
            with self.assertRaisesRegex(RuntimeError, 'symlinks'):
                queue.pending()

    def test_local_uuid_persists_until_valid_ack(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'pending.json'
            pending = launcher.PendingInput(path)
            pending.append('你好\n'.encode())
            job = pending.jobs[0]
            restored = launcher.PendingInput(path)
            self.assertEqual(restored.jobs[0], job)
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                restored.accept({'accepted': True, 'id': 'bad'})
            restored.accept({'accepted': True, 'id': job['id'],
                             'sha256': hashlib.sha256(bytes.fromhex(job['hex'])).hexdigest()})
            self.assertFalse(launcher.PendingInput(path).jobs)


if __name__ == '__main__':
    unittest.main()
