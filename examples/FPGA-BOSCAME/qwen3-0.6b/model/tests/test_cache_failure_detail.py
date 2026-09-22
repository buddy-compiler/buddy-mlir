"""Execute the generated comparator with old/current/future cache errors."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import test_image_abi as fixture


HARNESS = r'''
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "generated.c"
const float model_reference_raw[REF_LOGITS_ROWS * VOCAB +
                               2 * LAYERS * 8 * REF_LENGTH * 128] = {0};
static float scores[VOCAB];
void nr_puts(const char *s) { fputs(s, stdout); }
void nr_hex32(uint32_t x) { printf("%08X", x); }
void nr_hex64(uint64_t x) { printf("%016llX", (unsigned long long)x); }
int main(int argc, char **argv) {
  assert(argc == 2);
  unsigned t = (unsigned)strtoul(argv[1], NULL, 10);
  unsigned index = (1 * 8 + 3) * CAPACITY * 128 + t * 128 + 7;
  k_cache_f[index] = 2;
  v_cache_f[index] = -3;
  int result = compare_reference(NULL, 17, scores, 1);
  assert((result != 0) == (t <= 17));
  return 0;
}
'''


class CacheFailureDetailTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('cc'), 'host C compiler required')
    def test_old_and_current_errors_are_located_future_slots_are_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            args, report, segment = fixture.ImageABI().setup_case(directory, 2)
            args.reference_arrays = p / 'reference.npz'
            args.reference_atol = 1e-4
            args.reference_mean_atol = 1e-6
            source, _ = fixture.image.generate(report, segment, args)
            (p / 'generated.c').write_text(source)
            (p / 'harness.c').write_text(HARNESS)
            common = fixture.MODEL.parents[1] / 'common'
            compiled = subprocess.run(['cc', '-std=c11', '-O1', '-DHOST_TEST',
                            '-ffunction-sections', '-fdata-sections',
                            '-I', str(fixture.MODEL.parent), '-I', str(common / 'nr'),
                            str(p / 'harness.c'), '-Wl,--gc-sections',
                            '-o', str(p / 'probe')], capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            for position in (4, 17, 18):
                with self.subTest(position=position):
                    run = subprocess.run([str(p / 'probe'), str(position)],
                                         check=True, capture_output=True, text=True)
                    details = [line for line in run.stdout.splitlines()
                               if line.startswith('[compare-detail]')]
                    self.assertEqual(len(details), 2 if position <= 17 else 0)
                    for row, name, bits in zip(details, ('key_cache', 'value_cache'),
                                               ('40000000', 'C0400000')):
                        self.assertIn(name + ' position=00000011', row)
                        self.assertIn('layer=00000001 head=00000003', row)
                        self.assertIn(f'cache_position={position:08X} dimension=00000007', row)
                        self.assertIn(f'actual_bits={bits} expected_bits=00000000', row)
                        self.assertTrue(row.endswith('first_unequal'))


if __name__ == '__main__':
    unittest.main()
