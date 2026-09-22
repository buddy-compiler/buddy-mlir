"""Exercise production console logic with mocked UART/cache/fence hooks.

These tests check protocol ownership and ordering requests, not FPGA cache
coherence. In particular, a host fence callback cannot model DDR visibility.
"""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


NR = Path(__file__).resolve().parents[3] / 'common' / 'nr'

HARNESS = r'''
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static unsigned fences, flushes, invalidations;
static unsigned input_polls;
static const volatile void *flushed[8192];
static const volatile void *invalidated[8192];
static unsigned wait_entries, wait_leaves, dropped;
static void (*fence_action)(void);
static char output[128 * 1024];
static size_t output_length;

static void fence(void) {
  assert(++fences < 1024 * 1024);
  if (fence_action) fence_action();
}
static void flush(const volatile void *address) {
  assert(flushes < sizeof(flushed) / sizeof(flushed[0]));
  flushed[flushes++] = address;
}
static void invalidate(const volatile void *address) {
  assert(invalidations < sizeof(invalidated) / sizeof(invalidated[0]));
  invalidated[invalidations++] = address;
}
static void uart_putc(char value) {
  assert(output_length + 1 < sizeof(output));
  output[output_length++] = value;
  output[output_length] = 0;
}
static void host_puts(const char *text) {
  while (*text) uart_putc(*text++);
}
static inline void pump_input(void) { ++input_polls; }

#include "nr_console.inc"

#if defined(NR_HANG_DIAGNOSTICS) && !NR_CONSOLE_APPEND_ONLY
static void nr_console_wait_enter(void) { ++wait_entries; }
static void nr_console_wait_leave(void) { ++wait_leaves; }
#ifdef NR_HANG_CONSOLE_BOUNDED
static void nr_console_drop(void) { ++dropped; }
#endif
#endif

static char pattern(uint32_t index) { return (char)('!' + index % 90); }

static void initialize_case(void) {
  console.count = UINT32_MAX;
  console.overflow = 1;
  console.consumed = UINT32_MAX;
  for (unsigned i = 0; i < NR_CONSOLE_CAPACITY; ++i)
    console.data[i] = pattern(i);
  nr_console_init();
  assert(console.count == 0 && console.overflow == 0 && console.consumed == 0);
  assert(flushes == 2 && fences == 1 && !invalidations);
  assert(flushed[0] == &console.count && flushed[1] == &console.consumed);
  assert((uintptr_t)&console % 64 == 0);
  assert((uintptr_t)&console.consumed - (uintptr_t)&console.count == 64);
  assert((uintptr_t)&console.data - (uintptr_t)&console.count == 128);
  /* Initialization owns control words; stale data is hidden by count = 0. */
  for (unsigned i = 0; i < NR_CONSOLE_CAPACITY; ++i)
    assert(console.data[i] == pattern(i));
  assert(output_length == 0);
}

static unsigned publish_fences;
static void observe_publication(void) {
  ++publish_fences;
  assert(console.data[0] == 'a');
  assert(console.count == (publish_fences == 1 ? 0u : 1u));
  assert(console.consumed == 0);
}
static void publish_case(void) {
  nr_console_init();
  unsigned before = flushes;
  fence_action = observe_publication;
  write_serial('a');
  fence_action = NULL;
  assert(publish_fences == 2 && flushes == before && !invalidations);
  assert(console.count == 1 && console.overflow == 0);
  assert(drain(0) == 1 && console.consumed == 1);
  assert(output_length == 1 && output[0] == 'a');
  assert(console.count == 1 && console.overflow == 0 && console.data[0] == 'a');
  assert(invalidated[0] == &console.count);
  assert(invalidated[1] == &console.data[0]);
  assert(flushes == before + (NR_CONSOLE_APPEND_ONLY ? 0 : 1));
}

static void terminal_case(void) {
  nr_console_init();
  nr_console_report(1);
  assert(strstr(output, "verify NR runtime: PASS\r\n"));
  output_length = 0; output[0] = 0;
  nr_console_report(2);
  assert(strstr(output, "verify NR runtime: FAIL\r\n"));
  assert(!strstr(output, "PASS"));
}

static void running_case(void) {
  nr_console_init();
  write_serial('a');
  write_serial('b');
  uint32_t consumed = nr_console_poll_running(0);
#if NR_CONSOLE_DRAIN_AFTER_COMPLETION
  assert(consumed == 0 && output_length == 0 && input_polls == 0);
  assert(invalidations == 0 && flushes == 2);
  assert(console.count == 2 && console.consumed == 0);
  /* Completion uses the same consumer as live mode, after RA stops writing. */
  consumed = drain(consumed);
#else
  assert(input_polls == 1);
#endif
  assert(consumed == 2 && !strcmp(output, "ab"));
  assert(console.consumed == 2);
  nr_console_report(1);
  assert(strstr(output, "verify NR runtime: PASS\r\n"));
}

#if NR_CONSOLE_APPEND_ONLY
static void append_case(void) {
  nr_console_init();
  /* This value would make a ring producer wait before writing its first byte. */
  console.consumed = UINT32_MAX - NR_CONSOLE_CAPACITY;
  unsigned before = fences;
  for (unsigned i = 0; i < NR_CONSOLE_CAPACITY; ++i) write_serial(pattern(i));
  assert(fences - before == NR_CONSOLE_CAPACITY * 2);
  assert(console.count == NR_CONSOLE_CAPACITY && console.overflow == 0);
  assert(console.consumed == UINT32_MAX - NR_CONSOLE_CAPACITY);
  for (unsigned i = 0; i < 17; ++i) write_serial('?');
  assert(console.count == NR_CONSOLE_CAPACITY && console.overflow == 1);
  assert(fences - before == NR_CONSOLE_CAPACITY * 2 + 17);
  for (unsigned i = 0; i < NR_CONSOLE_CAPACITY; ++i)
    assert(console.data[i] == pattern(i));
  assert(!wait_entries && !wait_leaves && !dropped);
  uint32_t consumed = 0;
  unsigned before_flush = flushes;
  while (consumed != console.count) {
    uint32_t next = drain(consumed);
    assert(next > consumed && next - consumed <= 256);
    consumed = next;
  }
  assert(flushes == before_flush);
  assert(output_length == NR_CONSOLE_CAPACITY);
  for (unsigned i = 0; i < output_length; ++i) assert(output[i] == pattern(i));
  /* Draining must not reopen append capacity or reset the sticky failure. */
  write_serial('?');
  assert(console.count == NR_CONSOLE_CAPACITY && console.overflow == 1);
  nr_console_report(1);
  assert(strstr(output, "[nr] console overflow: FAIL (output truncated)\r\n"));
  assert(strstr(output, "verify NR runtime: FAIL\r\n"));
  assert(!strstr(output, "PASS"));
}
#else
static void wrap_case(void) {
  nr_console_init();
  uint32_t begin = UINT32_MAX - 31;
  console.count = console.consumed = begin;
  for (unsigned i = 0; i < NR_CONSOLE_CAPACITY; ++i) write_serial(pattern(i));
  assert(console.count == (uint32_t)(begin + NR_CONSOLE_CAPACITY));
  assert(console.consumed == begin && !console.overflow);
  uint32_t consumed = begin;
  while (consumed != console.count) {
    uint32_t next = drain(consumed);
    assert((uint32_t)(next - consumed) <= 256);
    consumed = next;
  }
  assert(output_length == NR_CONSOLE_CAPACITY);
  for (unsigned i = 0; i < output_length; ++i) assert(output[i] == pattern(i));
  assert(console.consumed == console.count && !console.overflow);
  for (unsigned i = 2; i < flushes; ++i) assert(flushed[i] == &console.consumed);
}

static unsigned wait_fences;
static void acknowledge(void) {
  /* Confirm the full buffer is intact until NH acknowledges consumption. */
  if (++wait_fences <= 4) {
    assert(console.count == NR_CONSOLE_CAPACITY);
    assert(console.data[0] == pattern(0));
  }
  if (wait_fences == 4) {
    fence_action = NULL;
    assert(drain(0) > 0);
  }
}
static void wait_case(void) {
  nr_console_init();
  for (unsigned i = 0; i < NR_CONSOLE_CAPACITY; ++i) write_serial(pattern(i));
  fence_action = acknowledge;
  write_serial('?');
  fence_action = NULL;
  /* A full ring is an explicit transport failure. The RA side must not issue
   * an unverified CBO invalidate or wait forever for a possibly stale cursor.
   */
  assert(console.count == NR_CONSOLE_CAPACITY && console.consumed == 0);
  assert(console.overflow == 1 && !invalidations);
  assert(wait_fences == 1);
  for (unsigned i = 0; i < NR_CONSOLE_CAPACITY; ++i)
    assert(console.data[i] == pattern(i));
}
#endif

int main(int argc, char **argv) {
  assert(argc == 2);
  if (!strcmp(argv[1], "init")) initialize_case();
  else if (!strcmp(argv[1], "publish")) publish_case();
  else if (!strcmp(argv[1], "terminal")) terminal_case();
  else if (!strcmp(argv[1], "running")) running_case();
#if NR_CONSOLE_APPEND_ONLY
  else if (!strcmp(argv[1], "append")) append_case();
#else
  else if (!strcmp(argv[1], "wrap")) wrap_case();
  else if (!strcmp(argv[1], "wait")) wait_case();
#endif
  else assert(0);
  /* Make diagnostic counter references independent of the selected mode. */
  assert(wait_entries + wait_leaves + dropped + input_polls < 1024 * 1024);
  return 0;
}
'''


@unittest.skipUnless(shutil.which('cc'), 'host C compiler required')
class NrConsoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.source = cls.directory / 'harness.c'
        cls.source.write_text(HARNESS)
        cls.variants = {
            'default': [],
            'ring': ['-DNR_CONSOLE_CAPACITY=128'],
            'append': ['-DNR_CONSOLE_CAPACITY=128', '-DNR_CONSOLE_APPEND_ONLY=1'],
            'append_large': ['-DNR_CONSOLE_CAPACITY=1024', '-DNR_CONSOLE_APPEND_ONLY=1'],
            'append_deferred': ['-DNR_CONSOLE_CAPACITY=1024', '-DNR_CONSOLE_APPEND_ONLY=1',
                                '-DNR_CONSOLE_DRAIN_AFTER_COMPLETION=1'],
            'diagnostic': ['-DNR_CONSOLE_CAPACITY=128', '-DNR_HANG_DIAGNOSTICS'],
            'bounded': ['-DNR_CONSOLE_CAPACITY=128', '-DNR_HANG_DIAGNOSTICS',
                        '-DNR_HANG_CONSOLE_BOUNDED', '-DNR_HANG_CONSOLE_SPINS=3'],
        }
        for name, flags in cls.variants.items():
            result = cls.compile(flags, cls.directory / name)
            if result.returncode:
                raise RuntimeError(result.stderr)

    @classmethod
    def compile(cls, flags, output):
        return subprocess.run(
            ['cc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-O2', *flags,
             '-I' + str(NR), str(cls.source), '-o', str(output)],
            capture_output=True, text=True, timeout=30)

    def run_case(self, variant, case):
        result = subprocess.run([str(self.directory / variant), case],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '')

    def test_initialization_and_ownership(self):
        for variant in self.variants:
            with self.subTest(variant=variant):
                self.run_case(variant, 'init')

    def test_publish_before_count_and_consumer_ownership(self):
        for variant in self.variants:
            with self.subTest(variant=variant):
                self.run_case(variant, 'publish')

    def test_terminal_status(self):
        for variant in self.variants:
            with self.subTest(variant=variant):
                self.run_case(variant, 'terminal')

    def test_deferred_drain_skips_shared_reads_uart_and_input_until_completion(self):
        for variant in self.variants:
            with self.subTest(variant=variant):
                self.run_case(variant, 'running')

    def test_append_never_waits_and_overflow_cannot_report_pass(self):
        for variant in ('append', 'append_large', 'append_deferred'):
            with self.subTest(variant=variant):
                self.run_case(variant, 'append')

    def test_ring_crosses_buffer_and_uint32_wrap(self):
        for variant in ('default', 'ring', 'diagnostic', 'bounded'):
            with self.subTest(variant=variant):
                self.run_case(variant, 'wrap')

    def test_ring_waits_for_ack_and_bounded_diagnostic_drops(self):
        for variant in ('default', 'ring', 'diagnostic', 'bounded'):
            with self.subTest(variant=variant):
                self.run_case(variant, 'wait')

    def test_invalid_configuration_is_rejected(self):
        for flags in (
                ['-DNR_CONSOLE_CAPACITY=0'], ['-DNR_CONSOLE_CAPACITY=32'],
                ['-DNR_CONSOLE_CAPACITY=96'], ['-DNR_CONSOLE_CAPACITY=2147483648u'],
                ['-DNR_CONSOLE_APPEND_ONLY=2'],
                ['-DNR_CONSOLE_DRAIN_AFTER_COMPLETION=2'],
                ['-DNR_CONSOLE_DRAIN_AFTER_COMPLETION=1'],
                ['-DNR_CONSOLE_APPEND_ONLY=1', '-DNR_CONSOLE_DRAIN_AFTER_COMPLETION=1',
                 '-DNR_HANG_DIAGNOSTICS'],
                ['-DNR_CONSOLE_APPEND_ONLY=1', '-DNR_CONSOLE_DRAIN_AFTER_COMPLETION=1',
                 '-DNR_UART_DEBUG'],
                ['-DNR_CONSOLE_APPEND_ONLY=1', '-DNR_HANG_CONSOLE_BOUNDED']):
            with self.subTest(flags=flags):
                result = self.compile(flags, self.directory / 'invalid')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('error:', result.stderr)


if __name__ == '__main__':
    unittest.main()
