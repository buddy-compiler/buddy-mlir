"""Host semantics of the real NR watch record; no cache/hardware simulation.

The C harness includes the production implementation and mocks only its platform
hooks. Flush/invalidate assertions check that requests are made, not that another
core would observe their effects on FPGA.
"""

from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


NR = Path(__file__).resolve().parents[3] / 'common' / 'nr'
BEGIN = b'\x1eNRWATCH1\n'
END = b'\x1f'

HARNESS = r'''
#include <assert.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "nr_runtime.h"

static struct {
  volatile uint32_t count, consumed;
} console;
static uint64_t now;
static unsigned fences, invalidations, flushes;
static const volatile void *flushed[16];
static int tear_on_second_header_read;
static unsigned header_reads;
static char output[16384];
static size_t output_length;

static void fence(void) { assert(++fences < 1000); }
static void flush(const volatile void *address) {
  assert(flushes < sizeof(flushed) / sizeof(flushed[0]));
  flushed[flushes++] = address;
}
static void invalidate(const volatile void *address);
uint64_t nr_cycles(void) { return now; }
static void host_puts(const char *text) {
  size_t length = strlen(text);
  assert(output_length + length < sizeof(output));
  memcpy(output + output_length, text, length + 1);
  output_length += length;
}
static void host_hex(uint64_t value) {
  char text[17];
  snprintf(text, sizeof(text), "%016" PRIx64, value);
  host_puts(text);
}

#include "nr_hang_watch.inc"

static void invalidate(const volatile void *address) {
  assert(++invalidations < 1000);
  if (address == nr_diag_record && tear_on_second_header_read &&
      ++header_reads == 2) {
    /* Another publication completes between the first read and recheck. */
    nr_diag_record[0] += 2;
  }
}

static void initialize_case(void) {
  for (unsigned i = 0; i < 32; ++i) nr_diag_record[i] = UINT64_MAX;
  for (unsigned i = 0; i < 8; ++i) nr_diag_console[i] = UINT64_MAX;
  nr_diag_init();
  for (unsigned i = 0; i < 32; ++i) assert(nr_diag_record[i] == 0);
  for (unsigned i = 0; i < 8; ++i) assert(nr_diag_console[i] == 0);
  assert(flushes == 5 && fences > 0);
  /* Every separate 64-byte RA-owned record line was handed to flush. */
  for (unsigned line = 0; line < 5; ++line) {
    const volatile void *expected = line < 4
        ? (const volatile void *)&nr_diag_record[line * 8]
        : (const volatile void *)nr_diag_console;
    unsigned seen = 0;
    for (unsigned i = 0; i < flushes; ++i) seen += flushed[i] == expected;
    assert(seen == 1);
  }
  assert(output_length == 0);
}

static void lifecycle_case(void) {
  uint64_t last = 0, sample = 0;
  nr_diag_init();
  nr_diag_reset(18);
  nr_diag_memref(0, 0x1000, 0x80000100, -3, 1, 1024, 1024, 1);
  nr_diag_memref(1, 0x2000, 0x80002000, 7, 1024, 1024, 2048, 2);
  nr_diag_memref(2, 0x3000, 0x80004000, 0, 1, 1024, 1024, 1);
  nr_diag_mark(NR_DIAG_KERNEL_ENTER, 7);
  console.count = 1;
  console.consumed = UINT32_MAX;
  nr_diag_console[0] = 1;
  nr_diag_console[1] = 11;
  nr_diag_console[2] = 3;
  nr_diag_poll(&last, &sample, 1);
  nr_diag_mark(NR_DIAG_KERNEL_RETURN, 7);
  nr_diag_poll(&last, &sample, 1);
  nr_diag_mark(NR_DIAG_GRAPH_RETURN, 0);
  nr_diag_poll(&last, &sample, 1);
  nr_diag_mark(NR_DIAG_COLLECT_BEGIN, 99);
  nr_diag_poll(&last, &sample, 1);
  nr_diag_reset(19);
  nr_diag_poll(&last, &sample, 1);
  assert(sample == 5 && !(nr_diag_record[0] & 1));
  /* A new graph must not inherit any previous operand's valid marker. */
  for (unsigned i = 8; i < 32; ++i) assert(nr_diag_record[i] == 0);
}

static void throttle_case(void) {
  uint64_t last = 0, sample = 0;
  nr_diag_init();
  nr_diag_reset(16);
  now = NR_HANG_DIAG_PERIOD_CYCLES - 1;
  unsigned before = invalidations;
  nr_diag_poll(&last, &sample, 0);
  assert(last == 0 && sample == 0 && invalidations == before);
  assert(output_length == 0);
  now = NR_HANG_DIAG_PERIOD_CYCLES;
  nr_diag_poll(&last, &sample, 0);
  assert(last == now && sample == 1);
  size_t bytes = output_length;
  before = invalidations;
  nr_diag_poll(&last, &sample, 0);
  assert(output_length == bytes && sample == 1 && invalidations == before);
  nr_diag_poll(&last, &sample, 1);
  assert(output_length > bytes && sample == 2 && last == now);
  /* The unsigned cycle counter can wrap without freezing the sampler. */
  last = UINT64_MAX - NR_HANG_DIAG_PERIOD_CYCLES + 1;
  now = 0;
  nr_diag_poll(&last, &sample, 0);
  assert(last == 0 && sample == 3);
}

static void unstable_case(int torn) {
  uint64_t last = 0, sample = 0;
  nr_diag_init();
  nr_diag_reset(18);
  nr_diag_memref(0, 0x1000, 0x80000100, 0, 1, 1024, 1024, 1);
  nr_diag_mark(NR_DIAG_KERNEL_ENTER, 7);
  if (torn) tear_on_second_header_read = 1;
  else ++nr_diag_record[0]; /* Writer remains mid-publication. */
  nr_diag_poll(&last, &sample, 1);
  assert(sample == 1);
  /* No retry until the writer quiesces: NH's poll must remain bounded. */
  assert(invalidations <= 16 && fences < 100);
}

static void invalid_operand_case(void) {
  uint64_t saved[32];
  nr_diag_init();
  nr_diag_reset(18);
  nr_diag_memref(0, 0x1000, 0x80000100, 3, 1, 1024, 1024, 1);
  for (unsigned i = 0; i < 32; ++i) saved[i] = nr_diag_record[i];
  unsigned before = fences;
  nr_diag_memref(3, 123, 456, 7, 8, 9, 10, 11);
  nr_diag_memref(UINT_MAX, 123, 456, 7, 8, 9, 10, 11);
  assert(fences == before);
  for (unsigned i = 0; i < 32; ++i) assert(saved[i] == nr_diag_record[i]);
}

int main(int argc, char **argv) {
  assert(argc == 2);
  if (!strcmp(argv[1], "initialize")) initialize_case();
  else if (!strcmp(argv[1], "lifecycle")) lifecycle_case();
  else if (!strcmp(argv[1], "throttle")) throttle_case();
  else if (!strcmp(argv[1], "odd")) unstable_case(0);
  else if (!strcmp(argv[1], "torn")) unstable_case(1);
  else if (!strcmp(argv[1], "invalid")) invalid_operand_case();
  else assert(0);
  assert(fwrite(output, 1, output_length, stdout) == output_length);
  return 0;
}
'''


@unittest.skipUnless(shutil.which('cc'), 'host C compiler required')
class NrHangWatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        directory = Path(cls.temporary.name)
        source = directory / 'harness.c'
        source.write_text(HARNESS)
        cls.binary = directory / 'harness'
        subprocess.run(
            ['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-O2',
             '-DNR_HANG_DIAGNOSTICS', '-I' + str(NR), str(source),
             '-o', str(cls.binary)], check=True, capture_output=True, timeout=30)

    def run_case(self, mode):
        result = subprocess.run([str(self.binary), mode], check=True,
                                capture_output=True, timeout=5)
        self.assertEqual(result.stderr, b'')
        return result.stdout

    def frames(self, output):
        frames = []
        while output:
            self.assertTrue(output.startswith(BEGIN), repr(output[:64]))
            payload, separator, output = output[len(BEGIN):].partition(END)
            self.assertEqual(separator, END)
            self.assertNotIn(BEGIN, payload)
            lines = payload.decode('ascii').splitlines()
            records = []
            for line in lines:
                if not line:
                    continue
                self.assertRegex(line, r'^\[nh-watch(?:-memref)?\] ')
                records.append(dict((key, int(value, 16)) for key, value in
                                    re.findall(r'(\w+)=([0-9a-f]+)', line)))
            self.assertTrue(records)
            frames.append(records)
        return frames

    def test_nh_initializes_and_requests_flush_for_all_owned_lines(self):
        self.assertEqual(self.run_case('initialize'), b'')

    def test_lifecycle_retains_selected_kernel_through_graph_and_collect(self):
        frames = self.frames(self.run_case('lifecycle'))
        self.assertEqual(len(frames), 5)
        for index, (phase, target) in enumerate(((2, 2), (3, 3), (4, 3), (6, 3))):
            header, *operands = frames[index]
            self.assertEqual(header['sample'], index + 1)
            self.assertEqual(header['stable'], 1)
            self.assertEqual(header['position'], 18)
            self.assertEqual(header['phase'], phase)
            self.assertEqual(header['target_phase'], target)
            self.assertEqual(header['target_call'], 7)
            self.assertEqual(header['pending'], 2)
            self.assertEqual(header['ra_console_wait'], 1)
            self.assertEqual(header['dropped'], 11)
            self.assertEqual(header['waits'], 3)
            self.assertEqual([o['arg'] for o in operands], [0, 1, 2])
            self.assertEqual([o['valid'] for o in operands], [1, 1, 1])
            self.assertEqual([o['descriptor'] for o in operands], [0x1000, 0x2000, 0x3000])
            self.assertEqual([o['aligned'] for o in operands],
                             [0x80000100, 0x80002000, 0x80004000])
            self.assertEqual([o['offset'] for o in operands], [(1 << 64) - 3, 7, 0])
            self.assertEqual([o['rows'] for o in operands], [1, 1024, 1])
            self.assertEqual([o['cols'] for o in operands], [1024, 1024, 1024])
            self.assertEqual([o['stride0'] for o in operands], [1024, 2048, 1024])
            self.assertEqual([o['stride1'] for o in operands], [1, 2, 1])
        self.assertEqual(frames[3][0]['detail'], 99)
        self.assertEqual(len(frames[4]), 1)
        reset = frames[4][0]
        self.assertEqual(reset['phase'], 1)
        self.assertEqual(reset['position'], 19)
        self.assertEqual(reset['target_phase'], 0)
        self.assertEqual(reset['target_call'], 0)

    def test_throttling_force_and_cycle_counter_wrap(self):
        frames = self.frames(self.run_case('throttle'))
        self.assertEqual([f[0]['sample'] for f in frames], [1, 2, 3])
        self.assertEqual(frames[0][0]['nh_cycles'], frames[1][0]['nh_cycles'])
        self.assertEqual(frames[2][0]['nh_cycles'], 0)

    def test_unstable_publications_are_bounded_and_hide_descriptors(self):
        for mode in ('odd', 'torn'):
            with self.subTest(mode=mode):
                frames = self.frames(self.run_case(mode))
                self.assertEqual(len(frames), 1)
                self.assertEqual(len(frames[0]), 1)
                header = frames[0][0]
                self.assertEqual(header['stable'], 0)
                for field in ('phase', 'position', 'detail', 'target_phase', 'target_call'):
                    self.assertNotIn(field, header)
                if mode == 'odd':
                    self.assertEqual(header['seq'] & 1, 1)
                else:
                    self.assertNotEqual(header['seq'], header['seq_after'])

    def test_invalid_operands_do_not_mutate_publication(self):
        self.assertEqual(self.run_case('invalid'), b'')


if __name__ == '__main__':
    unittest.main()
