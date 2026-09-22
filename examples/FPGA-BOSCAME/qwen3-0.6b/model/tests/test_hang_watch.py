"""Exercise generated --wrap glue against independent host kernel/runtime stubs."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


MODEL = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('hang_watch', MODEL / 'tools/hang_watch.py')
watch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watch)
SYMBOL = '_mlir_ciface_kernel_matmul_1x1024x1024'
PROTOTYPE = f'extern void {SYMBOL}(MemRef2 *, MemRef2 *, MemRef2 *);\n'


class HangWatchTests(unittest.TestCase):
    def test_probe_parser_rejects_unsafe_and_out_of_range_values(self):
        self.assertEqual(watch.parse_watch(SYMBOL + ':0x1A'),
                         {'symbol': SYMBOL, 'call_index': 26})
        self.assertEqual(watch.parse_watch(SYMBOL + ':0002')['call_index'], 2)
        self.assertEqual(watch.parse_watch(SYMBOL + ':18446744073709551615')['call_index'],
                         0xffffffffffffffff)
        for value in (None, '', SYMBOL, SYMBOL + ':-1', SYMBOL + ':1;bad',
                      'other:0', SYMBOL + ':1.0', SYMBOL + ':18446744073709551616'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                watch.parse_watch(value)

    def test_rejects_unsupported_or_conflicting_selected_abi(self):
        invalid = [
            PROTOTYPE.replace('MemRef2 *', 'MemRef1 *', 1),
            PROTOTYPE.replace(', MemRef2 *', '', 1),
            PROTOTYPE.replace('extern void', 'extern int'),
            PROTOTYPE.replace('MemRef2 *', 'MemRef2', 1),
            PROTOTYPE.replace('MemRef2 *', 'void *', 1),
            PROTOTYPE.replace('MemRef2 *', 'MemRef2 **', 1),
            PROTOTYPE + PROTOTYPE.replace('extern void', 'extern int'),
            '/* ' + PROTOTYPE + ' */',
            '// ' + PROTOTYPE,
            PROTOTYPE.replace(SYMBOL, SYMBOL + '_different'),
        ]
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            for prototype in invalid:
                with self.subTest(prototype=prototype):
                    (d / 'adapters.c').write_text(prototype)
                    with self.assertRaises(ValueError):
                        watch.validate_watch(d / 'adapters.c', SYMBOL + ':0')
                    with self.assertRaises(ValueError):
                        watch.generate_watch(d / 'adapters.c', d, SYMBOL + ':0')
                    self.assertFalse((d / 'hang-watch.c').exists())

    def test_metadata_preserves_unknown_dtype_and_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            adapters = d / 'adapters.c'
            adapters.write_text(PROTOTYPE + PROTOTYPE
                                + 'extern void _mlir_ciface_kernel_other(MemRef1 *);\n')
            self.assertEqual(watch.validate_watch(adapters, SYMBOL + ':2'),
                             {'symbol': SYMBOL, 'call_index': 2})
            self.assertEqual(list(d.iterdir()), [adapters])
            source, flags = watch.generate_watch(adapters, d / 'out', SYMBOL + ':2')
            metadata = json.loads((source.parent / 'hang-watch.json').read_text())
            self.assertEqual(flags, ['--wrap=' + SYMBOL])
            self.assertEqual(metadata['watch'], {'symbol': SYMBOL, 'call_index': 2})
            self.assertEqual(metadata['source_sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(metadata['adapter_sha256'], hashlib.sha256(adapters.read_bytes()).hexdigest())
            for operand in metadata['operands']:
                self.assertIsNone(operand['dtype'])
                self.assertEqual(operand['offset_unit'], 'elements')
                self.assertEqual(operand['stride_unit'], 'elements')
            self.assertFalse(metadata['progress_uart_in_wrapper'])
            self.assertFalse(metadata['extra_ame_sync_in_wrapper'])
            self.assertNotIn('nr_puts', source.read_text())
            self.assertNotIn('ame_fence', source.read_text())
            self.assertNotIn('__asm__', source.read_text())
            self.assertEqual(source.read_text().count('void __wrap_'), 1)
            self.assertIn('cannot establish cache incoherence', ' '.join(metadata['limits']))

    @unittest.skipUnless(shutil.which('cc'), 'host C compiler required')
    def test_real_linker_wrap_preserves_calls_descriptors_and_per_graph_selection(self):
        """Use separate TUs so the host linker, not the test, resolves __real_."""
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            (d / 'adapters.c').write_text(PROTOTYPE)
            source, flags = watch.generate_watch(d / 'adapters.c', d, SYMBOL + ':1')
            (d / 'nr_runtime.h').write_text('''#ifndef TEST_NR_RUNTIME_H
#define TEST_NR_RUNTIME_H
#include <stdint.h>
enum { NR_DIAG_GRAPH_BEGIN=1, NR_DIAG_KERNEL_ENTER=2, NR_DIAG_KERNEL_RETURN=3,
       NR_DIAG_GRAPH_RETURN=4, NR_DIAG_SYNC_DONE=5,
       NR_DIAG_COLLECT_BEGIN=6, NR_DIAG_COLLECT_DONE=7 };
void nr_diag_reset(uint64_t position);
void nr_diag_mark(uint64_t stage, uint64_t detail);
void nr_diag_memref(unsigned operand, uintptr_t descriptor, uintptr_t aligned,
                    int64_t offset, int64_t rows, int64_t cols,
                    int64_t stride0, int64_t stride1);
#endif
''')
            # Use the real MemRef declarations from support.h, but a runtime stub
            # so this is purely a C ABI/ordering test without hardware assumptions.
            (d / 'kernel.c').write_text('''#include "support.h"
#include <assert.h>
extern unsigned real_calls, events_count;
extern int expect_observed;
extern MemRef2 *expected[3];
extern void event(unsigned kind);
void ''' + SYMBOL + '''(MemRef2 *a, MemRef2 *b, MemRef2 *c) {
  assert(a == expected[0] && b == expected[1] && c == expected[2]);
  ++real_calls;
  if (expect_observed) event(40);
  /* Mutate an output descriptor to ensure snapshots precede the real call. */
  if (c) c->offset += 11;
}
''')
            (d / 'harness.c').write_text('''#include "support.h"
#include "nr_runtime.h"
#include <assert.h>
extern void ''' + SYMBOL + '''(MemRef2 *, MemRef2 *, MemRef2 *);
extern void qwen_hang_reset(unsigned);
unsigned real_calls, events_count;
int expect_observed;
MemRef2 *expected[3];
static uint64_t position_seen;
static unsigned events[16], snapshots;
static MemRef2 before[3];
void event(unsigned kind) { assert(events_count < 16); events[events_count++] = kind; }
void nr_diag_reset(uint64_t position) { position_seen=position; event(1); }
void nr_diag_mark(uint64_t stage, uint64_t detail) {
  assert(expect_observed && detail == 1);
  if (stage == NR_DIAG_KERNEL_ENTER) {
    assert(snapshots == 3); event(30);
  } else {
    assert(stage == NR_DIAG_KERNEL_RETURN);
    assert(events[events_count-1] == 40); event(50);
  }
}
void nr_diag_memref(unsigned operand, uintptr_t descriptor, uintptr_t aligned,
                    int64_t offset, int64_t rows, int64_t cols,
                    int64_t stride0, int64_t stride1) {
  assert(expect_observed && operand == snapshots && operand < 3);
  assert(descriptor == (uintptr_t)expected[operand]);
  if (descriptor) {
    MemRef2 *m = &before[operand];
    assert(aligned == (uintptr_t)m->aligned && offset == m->offset);
    assert(rows == m->sizes[0] && cols == m->sizes[1]);
    assert(stride0 == m->strides[0] && stride1 == m->strides[1]);
  } else {
    assert(!aligned && !offset && !rows && !cols && !stride0 && !stride1);
  }
  ++snapshots; event(10 + operand);
}
static void invoke(int selected) {
  expect_observed = selected;
  snapshots = 0;
  for (unsigned i=0; i<3; ++i) if (expected[i]) before[i] = *expected[i];
  ''' + SYMBOL + '''(expected[0], expected[1], expected[2]);
}
int main(void) {
  int buffer[12];
  MemRef2 a = {buffer, buffer+1, -3, {2, 3}, {9, -2}};
  MemRef2 b = {buffer, buffer+2, 7, {0, 19}, {0, 3}};
  MemRef2 c = {buffer, buffer+3, 5, {4, 2}, {17, 4}};
  expected[0]=&a; expected[1]=&b; expected[2]=&c;
  qwen_hang_reset(16);
  assert(position_seen == 16 && events_count == 1);
  invoke(0); assert(real_calls == 1 && events_count == 1);
  invoke(1); assert(real_calls == 2 && events_count == 7);
  unsigned sequence[] = {1, 10, 11, 12, 30, 40, 50};
  for (unsigned i=0; i<7; ++i) assert(events[i] == sequence[i]);
  invoke(0); assert(real_calls == 3 && events_count == 7);
  events_count = 0;
  qwen_hang_reset(17);
  assert(position_seen == 17);
  expected[1] = 0; /* Null is observed without dereference, forwarded unchanged. */
  invoke(0); assert(real_calls == 4 && events_count == 1);
  invoke(1); assert(real_calls == 5 && events_count == 7);
  for (unsigned i=0; i<7; ++i) assert(events[i] == sequence[i]);
  invoke(0); assert(real_calls == 6 && events_count == 7);
  return 0;
}
''')
            command = ['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-O2',
                       '-DHOST_TEST', '-DNR_HANG_DIAGNOSTICS',
                       '-I' + str(d), '-I' + str(MODEL.parent), str(source),
                       str(d / 'kernel.c'), str(d / 'harness.c'),
                       *['-Wl,' + flag for flag in flags], '-o', str(d / 'test')]
            subprocess.run(command, check=True, capture_output=True, text=True)
            subprocess.run([str(d / 'test')], check=True, capture_output=True, text=True)


if __name__ == '__main__':
    unittest.main()
