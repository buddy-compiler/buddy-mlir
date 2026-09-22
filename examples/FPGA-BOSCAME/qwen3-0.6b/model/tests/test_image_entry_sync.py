"""Execute optional entry synchronization against graph stubs, not hardware.

HOST_TEST replaces only the entry fence with an observable host hook. These
tests verify preparation/order and session lifetime, not AME/cache completion.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import test_image_abi as fixture

image = fixture.image
MODEL = fixture.MODEL


HARNESS = r'''
#include <assert.h>
#include <string.h>
#include "generated.c"
static unsigned calls, primes, completions, entries, pending, event;
static float logits[VOCAB];
static unsigned local_call(void) { return calls % (MODE == 2 ? 3 : 9); }
static unsigned expected_position(void) {
  unsigned local = local_call();
  return MODE == 2 ? local : (local ? 15 + local : 0);
}
static void assert_prepared(void) {
  unsigned position = expected_position();
  assert(*cache_position == position);
  unsigned prefill = MODE != 2 && !local_call();
  const int32_t *bounds = (const int32_t *)(prefill ? ws_prefill_raw : ws_decode_raw);
  for (unsigned i = 0; i < (prefill ? 16 : 1); ++i)
    assert(bounds[i] == position + i);
  assert(input_ids[0] == (MODE == 2 ? (position == 0 ? 7 : position == 1 ? 9 : 5)
                                  : (prefill ? 0 : 5)));
}
void nr_puts(const char *s) { (void)s; event = 1; }
void nr_write(const void *s, size_t n) { (void)s; (void)n; event = 1; }
void nr_hex32(uint32_t x) { (void)x; event = 1; }
void nr_hex64(uint64_t x) { (void)x; event = 1; }
uint64_t nr_cycles(void) { event = 1; return 100; }
uintptr_t nr_heap_mark(void) { event = 1; return 64; }
void nr_heap_reset(uintptr_t mark) { assert(mark == 64); event = 1; }
void nr_copy_bytes(void *d, const void *s, size_t n) { memcpy(d, s, n); event = 1; }
void ame_fence(void) {
  if (pending) { completions++; pending = 0; }
  else { assert(calls == 0 && primes == 0); assert_prepared(); primes++; }
  event = 2;
}
void qwen_graph_entry_fence(void) {
  assert_prepared();
  assert(primes == EXPECT_PRIME && !pending && completions == calls);
  entries++; event = 3;
}
static void graph(GraphResults *r, MemRef2 *ids, MemRef1 *pos,
                  MemRef4 *key, MemRef4 *value, MemRef1 *bounds) {
  if (EXPECT_ENTRY) assert(event == 3 && entries == calls + 1);
  else assert(entries == 0);
  assert(primes == EXPECT_PRIME && !pending && completions == calls);
  assert_prepared();
  assert(*(int64_t *)pos->aligned == expected_position());
  assert(((int32_t *)bounds->aligned)[0] == expected_position());
  assert(ids->aligned == input_ids);
  r->cache[0] = (CacheResult){*pos, *key, *value};
  logits[5] = 1;
  r->logits = make_3(logits, 1, 1, VOCAB);
  calls++; pending = 1; event = 4;
}
void _mlir_ciface_forward_prefill(GraphResults *r, MemRef2 *ids, MemRef1 *pos,
  MemRef4 *key, MemRef4 *value, MemRef1 *bounds) {
  assert(MODE != 2 && local_call() == 0);
  graph(r, ids, pos, key, value, bounds);
}
void _mlir_ciface_forward_decode(GraphResults *r, MemRef2 *ids, MemRef1 *pos,
  MemRef4 *key, MemRef4 *value, MemRef1 *bounds) {
  assert(MODE == 2 || local_call() != 0);
  graph(r, ids, pos, key, value, bounds);
}
#if MODE != 0
int nr_getchar(void) {
  static unsigned cursor;
  const char input[] = "x\nx\n/quit\nx\nx\n/quit\n";
  assert(cursor < sizeof(input) - 1);
  return input[cursor++];
}
int qwen_tokenizer_open(QwenTokenizerResource *r, const void *b, size_t n) {
  (void)r; (void)b; (void)n; return 0;
}
int qwen_chat_single_turn(uint8_t *out, size_t cap, size_t *n, const uint8_t *s,
  size_t sn, int hs, const uint8_t *u, size_t un, int t) {
  (void)cap; (void)s; (void)sn; (void)hs; (void)t;
  assert(un == 1 && u[0] == 'x'); out[0] = 'x'; *n = 1; return 0;
}
int qwen_encode(const QwenTokenizerResource *r, const uint8_t *s, size_t n,
  uint32_t *out, size_t cap, size_t *written) {
  (void)r; (void)s; (void)n; (void)cap;
  if (MODE == 1) { for (unsigned i = 0; i < 16; ++i) out[i] = i; *written = 16; }
  else { out[0] = 7; out[1] = 9; *written = 2; }
  return 0;
}
int qwen_decode_token(const QwenTokenizerResource *r, QwenUtf8Decoder *s,
  uint32_t token, int skip, QwenEmit emit, void *ctx) {
  (void)r; (void)s; (void)skip; (void)emit; (void)ctx;
  assert(token == 5); return 0;
}
void qwen_decode_finish(QwenUtf8Decoder *s, QwenEmit emit, void *ctx) {
  (void)s; (void)emit; (void)ctx;
}
#endif
int main(void) {
  assert(launch() == 0);
  assert(launch() == 0);
  assert(calls == (MODE == 2 ? 12 : 18));
  assert(primes == EXPECT_PRIME && completions == calls && !pending);
  assert(entries == (EXPECT_ENTRY ? calls : 0));
  return 0;
}
'''


class ImageEntrySyncTests(unittest.TestCase):
    def setup_mode(self, directory, mode):
        args, report, segment = fixture.ImageABI().setup_case(directory, 1)
        if mode != 'numeric':
            args.tokenizer_blob = Path(directory) / 'tokenizer.bin'
            args.tokenizer_blob.write_bytes(b'fixture')
        args.prompt_text = 'x' if mode == 'fixed' else None
        args.interactive = mode == 'interactive'
        args.max_new_tokens = 2
        return args, report, segment

    def test_explicit_defaults_preserve_source_in_all_modes(self):
        for mode in ('numeric', 'fixed', 'interactive'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                args, report, segment = self.setup_mode(directory, mode)
                original, original_plan = image.generate(report, segment, args)
                args.ame_startup = 'none'
                args.graph_entry_fence = False
                source, plan = image.generate(report, segment, args)
                self.assertEqual(source, original)
                self.assertEqual(plan, original_plan)
                self.assertNotIn('prime_ame_once', source)
                self.assertNotIn('qwen_graph_entry_fence', source)
                self.assertEqual(plan['ame_startup'], 'none')
                self.assertFalse(plan['graph_entry_fence'])

    def test_invalid_startup_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args, report, segment = self.setup_mode(directory, 'numeric')
            args.ame_startup = 'self-test'
            with self.assertRaisesRegex(ValueError, '--ame-startup'):
                image.generate(report, segment, args)

    def test_entry_fence_follows_preparation_immediately_precedes_call(self):
        with tempfile.TemporaryDirectory() as directory:
            args, report, segment = self.setup_mode(directory, 'numeric')
            args.graph_entry_fence = True
            args.ame_startup = 'prime'
            source, plan = image.generate(report, segment, args)
            for kind in ('prefill', 'decode'):
                start = source.index(f'static int run_{kind}(')
                call = source.index(f'  _mlir_ciface_forward_{kind}(&result,', start)
                before_call = source[start:call]
                self.assertTrue(before_call.endswith('  qwen_graph_entry_fence();\n'))
                self.assertLess(before_call.index('position + j;'),
                                before_call.index('  prime_ame_once();'))
                self.assertLess(before_call.index('  prime_ame_once();'),
                                before_call.index('  uint64_t begin = nr_cycles();'))
            self.assertIn('__asm__ volatile ("fence rw, rw" ::: "memory");', source)
            self.assertEqual(source.count('    ame_fence();'), 1)
            self.assertEqual(source.count('  ame_fence();\n  graph_phase'), 2)
            self.assertEqual(plan['graph_completion_sync'], 'ame-resync')
            self.assertIsNone(plan['completion_sync'])
            self.assertIn('no AME correctness self-test', ' '.join(plan['entry_sync_limits']))

    def test_execution_order_and_once_per_boot_across_repeated_launches(self):
        cc = shutil.which('clang') or shutil.which('cc')
        if not cc:
            self.skipTest('host C compiler unavailable')
        for index, mode in enumerate(('numeric', 'fixed', 'interactive')):
            for prime, entry in ((False, False), (True, False), (False, True), (True, True)):
                with self.subTest(mode=mode, prime=prime, entry=entry), tempfile.TemporaryDirectory() as directory:
                    p = Path(directory)
                    args, report, segment = self.setup_mode(directory, mode)
                    args.ame_startup = 'prime' if prime else 'none'
                    args.graph_entry_fence = entry
                    source, _ = image.generate(report, segment, args)
                    (p / 'generated.c').write_text(source)
                    (p / 'harness.c').write_text(HARNESS)
                    command = [cc, '-O1', '-g', '-fsanitize=address,undefined', '-DHOST_TEST',
                               f'-DMODE={index}', f'-DEXPECT_PRIME={int(prime)}',
                               f'-DEXPECT_ENTRY={int(entry)}', '-I' + str(MODEL.parent),
                               '-I' + str(MODEL.parents[1] / 'common/nr'),
                               '-I' + str(MODEL / 'text'), str(p / 'harness.c'),
                               '-o', str(p / 'check')]
                    result = subprocess.run(command, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    result = subprocess.run([str(p / 'check')], capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
