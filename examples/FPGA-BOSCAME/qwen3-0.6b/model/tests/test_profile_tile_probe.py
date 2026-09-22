"""Check raw tile probe ABI forwarding with real host linker --wrap calls."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import test_image_abi as image_fixture

MODEL = image_fixture.MODEL
sys.path.insert(0, str(MODEL / 'tools'))
import kernel_profile as generator

HIGH = '_mlir_ciface_kernel_matmul_16x3072x1024'
RAW = 'triton_matmul_16x3072x1024'
RAW_TYPES = ', '.join(['int64_t, MemRef0 *'] * 3 + ['int32_t'] * 6)
RAW_ADAPTER = f'''#include "support.h"
typedef struct {{ void *allocated, *aligned; int64_t offset; }} MemRef0;
extern void {RAW}({RAW_TYPES});
void {HIGH}(MemRef2 *a0, MemRef2 *a1, MemRef2 *a2) {{}}
'''

HEADER = r'''
#include <stdint.h>
#include "support.h"
typedef struct { void *allocated, *aligned; int64_t offset; } MemRef0;
extern MemRef0 operands[3];
extern unsigned raw_calls;
extern char events[1024];
extern unsigned event_count;
void record(char);
void invoke_raw(int32_t);
extern void triton_matmul_16x3072x1024(int64_t, MemRef0 *, int64_t, MemRef0 *,
  int64_t, MemRef0 *, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);
'''

RAW_IMPLEMENTATION = r'''
#include <assert.h>
#include "raw.h"
void triton_matmul_16x3072x1024(int64_t ra, MemRef0 *a, int64_t rb, MemRef0 *b,
  int64_t rc, MemRef0 *c, int32_t gx, int32_t gy, int32_t gz,
  int32_t x, int32_t y, int32_t z) {
  assert(ra == -3 && rb == 7 && rc == 19);
  assert(a == &operands[0] && b == &operands[1] && c == &operands[2]);
  for (unsigned i = 0; i < 3; ++i) {
    assert(operands[i].allocated == &operands[(i + 1) % 3]);
    assert(operands[i].aligned == &operands[(i + 2) % 3]);
    assert(operands[i].offset == (int64_t)i - 9);
  }
  assert(gx == 2 && gy == 5 && gz == 11);
  assert((x == 0 || x == 1) && y == 3 && z == 7);
  record('R'); raw_calls++;
}
'''

ADAPTER_IMPLEMENTATION = r'''
#include <assert.h>
#include "raw.h"
void invoke_raw(int32_t x) {
  triton_matmul_16x3072x1024(-3, &operands[0], 7, &operands[1], 19, &operands[2],
                           2, 5, 11, x, 3, 7);
}
void _mlir_ciface_kernel_matmul_16x3072x1024(MemRef2 *a, MemRef2 *b, MemRef2 *c) {
  assert(a->offset == 2 && b->offset == 4 && c->offset == 6);
  record('a'); invoke_raw(0); invoke_raw(1); record('b');
}
'''

HARNESS = r'''
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "raw.h"
#include "nr_runtime.h"
MemRef0 operands[3];
unsigned raw_calls, event_count;
char events[1024];
static char log_text[32768];
static unsigned log_size;
static MemRef2 *expected_high[3];
static unsigned invocation_index, descriptors, high_marks, tile_marks;
static int high_active, tile_active;
void qwen_profile_reset(void);
void _mlir_ciface_kernel_matmul_16x3072x1024(MemRef2 *, MemRef2 *, MemRef2 *);
void record(char c) { assert(event_count < sizeof(events) - 1); events[event_count++] = c; }
void nr_puts(const char *s) {
  size_t n = strlen(s); assert(log_size + n < sizeof(log_text));
  memcpy(log_text + log_size, s, n + 1); log_size += n;
  if (!strncmp(s, "[tile-probe] begin", 18)) record('B');
  if (!strncmp(s, "[tile-probe] returned", 21)) record('T');
  if (!strncmp(s, "[kernel] begin", 14)) record('H');
  if (!strncmp(s, "[kernel-phase] returned", 23)) record('U');
  if (!strncmp(s, "[kernel] end", 12)) record('E');
}
void nr_hex64(uint64_t x) { char b[17]; snprintf(b, sizeof(b), "%016llX", (unsigned long long)x); nr_puts(b); }
void nr_hex32(uint32_t x) { char b[9]; snprintf(b, sizeof(b), "%08X", x); nr_puts(b); }
uint64_t nr_cycles(void) { return raw_calls * 100; }
void ame_fence(void) { record('F'); invoke_raw(0); }
void nr_diag_memref(unsigned operand, uintptr_t descriptor, uintptr_t aligned,
  int64_t offset, int64_t rows, int64_t cols, int64_t stride0, int64_t stride1) {
  assert(EXPECT_WATCH && invocation_index == 1 && !high_active && !tile_active);
  assert(operand < 3 && operand == descriptors % 3);
  const MemRef2 *want = expected_high[operand];
  assert(descriptor == (uintptr_t)want && aligned == (uintptr_t)want->aligned);
  assert(offset == want->offset && rows == want->sizes[0] && cols == want->sizes[1]);
  assert(stride0 == want->strides[0] && stride1 == want->strides[1]);
  assert(events[event_count - 1] == (operand ? '0' + operand - 1 : 'H'));
  record('0' + operand); descriptors++;
}
void nr_diag_mark(uint64_t stage, uint64_t detail) {
  assert(EXPECT_WATCH && invocation_index == 1);
  switch (stage) {
  case NR_DIAG_KERNEL_ENTER:
    assert(detail == 1 && !high_active && !tile_active);
    assert(descriptors % 3 == 0 && events[event_count - 1] == '2');
    high_active = 1; high_marks++; record('K'); break;
  case NR_DIAG_KERNEL_RETURN:
    assert(detail == 1 && high_active && !tile_active);
    assert(events[event_count - 1] == 'b');
    high_active = 0; high_marks++; record('L'); break;
  case 8:
    assert(detail == 3 && high_active && !tile_active);
    assert(events[event_count - 1] == 'B');
    tile_active = 1; tile_marks++; record('I'); break;
  case 9:
    assert(detail == 3 && high_active && tile_active);
    assert(events[event_count - 1] == 'R');
    tile_active = 0; tile_marks++; record('J'); break;
  default: assert(0);
  }
}
int main(void) {
  for (unsigned i = 0; i < 3; ++i)
    operands[i] = (MemRef0){&operands[(i+1)%3], &operands[(i+2)%3], (int64_t)i-9};
  MemRef2 a = make_2(&operands[0], 16, 1024), b = make_2(&operands[1], 3072, 1024),
          c = make_2(&operands[2], 16, 3072);
  a.offset=2; b.offset=4; c.offset=6;
  expected_high[0]=&a; expected_high[1]=&b; expected_high[2]=&c;
  invoke_raw(0);
  for (unsigned graph = 0; graph < 2; ++graph) {
    qwen_profile_reset();
    for (unsigned i = 0; i < 3; ++i) {
      invocation_index = i;
      _mlir_ciface_kernel_matmul_16x3072x1024(&a, &b, &c);
      assert(!high_active && !tile_active);
    }
  }
  invoke_raw(1);
  assert(raw_calls == 20);
#if EXPECT_WATCH
  assert(descriptors == 6 && high_marks == 4 && tile_marks == 8);
  assert(!strcmp(events, "RHaRRbUFREH012KaBIRJTBIRJTbLUFREHaRRbUFREHaRRbUFREH012KaBIRJTBIRJTbLUFREHaRRbUFRER"));
#else
  assert(descriptors == 0 && high_marks == 0 && tile_marks == 0);
  assert(!strcmp(events, "RHaRRbUFREHaBRTBRTbUFREHaRRbUFREHaRRbUFREHaBRTBRTbUFREHaRRbUFRER"));
#endif
  const char *record = log_text;
  unsigned tile_records = 0;
  while ((record = strstr(record, "[tile-probe]"))) { tile_records++; record++; }
  assert(tile_records == 8);
  assert(strstr(log_text, "[tile-probe] begin triton_matmul_16x3072x1024 call=0000000000000001 grid_x=00000002 grid_y=00000005 grid_z=0000000B x=00000001 y=00000003 z=00000007\r\n"));
  return 0;
}
'''


class TileProbeTests(unittest.TestCase):
    def test_nh_watch_records_bracket_real_call_before_return_uart(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            adapters, raw = self.fixture(d)
            source, _ = generator.generate_profile(adapters, d, progress=True,
                probe=HIGH + ':38', tile_probe=True, raw_adapter=raw, watch=True)
            text = source.read_text()
            high = text.split('void __wrap_' + HIGH, 1)[1]
            order = ['nr_diag_memref(0', 'nr_diag_memref(1', 'nr_diag_memref(2',
                     'nr_diag_mark(NR_DIAG_KERNEL_ENTER', '__real_' + HIGH,
                     'nr_diag_mark(NR_DIAG_KERNEL_RETURN', '[kernel-phase] returned']
            self.assertEqual([high.index(x) for x in order], sorted(high.index(x) for x in order))
            tile = text.split('void __wrap_' + RAW, 1)[1].split('void __wrap_' + HIGH, 1)[0]
            order = ['[tile-probe] begin', 'nr_diag_mark(8', '__real_' + RAW,
                     'nr_diag_mark(9', '[tile-probe] returned']
            self.assertEqual([tile.index(x) for x in order], sorted(tile.index(x) for x in order))
            self.assertEqual(json.loads((d / 'kernel-profile.json').read_text())['nh_watch']['selection'],
                             {'symbol': HIGH, 'call_index': 38})
            with self.assertRaisesRegex(ValueError, '--profile-watch requires'):
                generator.generate_profile(adapters, d, watch=True)

    def fixture(self, root):
        root = Path(root)
        adapters = root / 'adapters.c'
        adapters.write_text(f'extern void {HIGH}(MemRef2 *, MemRef2 *, MemRef2 *);\n')
        raw = root / 'evidence/matmul_16x3072x1024/adapter.c'
        raw.parent.mkdir(parents=True)
        raw.write_text(RAW_ADAPTER)
        return adapters, raw

    def test_default_source_and_manifest_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            adapters, raw = self.fixture(d)
            for progress, probe in ((False, None), (True, None), (True, HIGH + ':1')):
                source, flags = generator.generate_profile(adapters, d, progress=progress, probe=probe)
                original = source.read_bytes()
                manifest = (d / 'kernel-profile.json').read_bytes()
                source, explicit_flags = generator.generate_profile(adapters, d, progress=progress,
                    probe=probe, tile_probe=False, raw_adapter=raw)
                self.assertEqual(source.read_bytes(), original)
                self.assertEqual((d / 'kernel-profile.json').read_bytes(), manifest)
                self.assertEqual(flags, explicit_flags)
                self.assertNotIn('tile_probe', source.read_text())

    def test_manifest_and_scope_order(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            adapters, raw = self.fixture(d)
            source, flags = generator.generate_profile(adapters, d, progress=True,
                probe=HIGH + ':0x26', tile_probe=True, raw_adapter=raw)
            meta = json.loads((d / 'kernel-profile.json').read_text())
            tile = meta['tile_probe']
            self.assertEqual(tile['raw_symbol'], RAW)
            self.assertEqual(tile['call_index'], 38)
            self.assertEqual(len(tile['raw_argument_types']), 12)
            self.assertEqual(tile['raw_adapter_sha256'], hashlib.sha256(raw.read_bytes()).hexdigest())
            self.assertEqual(flags, ['--wrap=' + HIGH, '--wrap=' + RAW])
            self.assertEqual(meta['linker_flags'], flags)
            wrapper = source.read_text().split('void __wrap_' + HIGH, 1)[1]
            self.assertIn('  tile_probe_selected = probe_selected;\n  __real_' + HIGH +
                          '(a0, a1, a2);\n  tile_probe_selected = 0;', wrapper)
            self.assertIn('not a throughput', ' '.join(tile['limits']))

    def test_unsupported_configurations_fail_before_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            adapters, raw = self.fixture(d)
            for probe, progress, evidence, pattern in (
                (None, True, raw, 'requires --profile-probe'),
                ('_mlir_ciface_kernel_norm:0', True, raw, 'only matmul'),
                (HIGH + ':0', False, raw, 'requires --profile-progress'),
                (HIGH + ':0', True, None, 'requires archive evidence'),
            ):
                with self.subTest(probe=probe, progress=progress), self.assertRaisesRegex(ValueError, pattern):
                    generator.generate_profile(adapters, d, progress=progress, probe=probe,
                                               tile_probe=True, raw_adapter=evidence)
                self.assertFalse((d / 'kernel-profile.c').exists())
            raw.write_text(RAW_ADAPTER.replace('int32_t', 'int64_t', 1))
            with self.assertRaisesRegex(ValueError, '12-argument ABI'):
                generator.validate_tile_probe(adapters, HIGH + ':0', raw)
            raw.write_text(RAW_ADAPTER.replace('int64_t offset;', 'int32_t offset;'))
            with self.assertRaisesRegex(ValueError, 'MemRef0 layout'):
                generator.validate_tile_probe(adapters, HIGH + ':0', raw)
            raw.write_text(RAW_ADAPTER)
            adapters.write_text(f'extern void {HIGH}(MemRef2 *, MemRef2 *);\n')
            with self.assertRaisesRegex(ValueError, 'three MemRef2'):
                generator.validate_tile_probe(adapters, HIGH + ':0', raw)

    def test_builder_preflight_and_default_source(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            args, report, segment = image_fixture.ImageABI().setup_case(d, 1)
            args.adapters, raw = self.fixture(d)
            args.archive = d / 'libqwen_triton.a'
            args.profile_kernels = args.profile_progress = True
            args.profile_probe = HIGH + ':1'
            baseline, _ = image_fixture.image.generate(report, segment, args)
            args.profile_tile_probe = True
            source, plan = image_fixture.image.generate(report, segment, args)
            self.assertEqual(source, baseline)
            self.assertTrue(plan['profile_tile_probe'])
            args.profile_probe = None
            with self.assertRaisesRegex(ValueError, '--profile-tile-probe requires'):
                image_fixture.image.generate(report, segment, args)

    def test_host_linker_forwarding_every_argument_selection_reset_and_order(self):
        cc = shutil.which('clang') or shutil.which('cc')
        if not cc:
            self.skipTest('host C compiler unavailable')
        for watch in (False, True):
            with self.subTest(watch=watch), tempfile.TemporaryDirectory() as directory:
                d = Path(directory)
                adapters, raw = self.fixture(d)
                source, flags = generator.generate_profile(adapters, d, progress=True,
                    probe=HIGH + ':1', tile_probe=True, raw_adapter=raw, watch=watch)
                for name, code in (('raw.h', HEADER), ('raw.c', RAW_IMPLEMENTATION),
                                   ('adapter.c', ADAPTER_IMPLEMENTATION), ('harness.c', HARNESS)):
                    (d / name).write_text(code)
                result = subprocess.run([cc, '-O1', '-g', '-fsanitize=address,undefined',
                    '-DHOST_TEST', '-DNR_HANG_DIAGNOSTICS', f'-DEXPECT_WATCH={int(watch)}',
                    '-I' + str(MODEL.parent), '-I' + str(MODEL.parents[1] / 'common/nr'),
                    str(source), str(d/'adapter.c'), str(d/'raw.c'), str(d/'harness.c'),
                    *['-Wl,' + flag for flag in flags], '-o', str(d/'check')],
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                result = subprocess.run([str(d/'check')], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
