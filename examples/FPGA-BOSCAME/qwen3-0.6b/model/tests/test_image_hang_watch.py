"""Validate opt-in wiring into the imported model launch and reject mixed probes."""
from pathlib import Path
import sys
import tempfile
import unittest

import test_image_abi as fixture

image = fixture.image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
SYMBOL = '_mlir_ciface_kernel_matmul_1x1024x2048'


class ImageHangWatch(unittest.TestCase):
    def fixture(self, directory):
        args, report, segment = fixture.ImageABI().setup_case(directory, 1)
        args.adapters = Path(directory) / 'adapters.c'
        args.adapters.write_text(f'extern void {SYMBOL}(MemRef2 *, MemRef2 *, MemRef2 *);\n')
        return args, report, segment

    def test_boundaries_and_modes_preserve_graph_and_sync(self):
        with tempfile.TemporaryDirectory() as directory:
            args, report, segment = self.fixture(directory)
            baseline, _ = image.generate(report, segment, args)
            self.assertNotIn('nr_diag_mark', baseline)
            args.hang_watch = SYMBOL + ':14'
            blocking, plan = image.generate(report, segment, args)
            self.assertEqual(plan['hang_console'], 'blocking')
            self.assertEqual(blocking.count('  qwen_hang_reset(position);'), 2)
            self.assertEqual(blocking.count('  ame_fence();'), 2)
            for kind in ('prefill', 'decode'):
                call = blocking.index(f'  _mlir_ciface_forward_{kind}(&result,')
                stages = [blocking.index(item, call) for item in (
                    'nr_diag_mark(NR_DIAG_GRAPH_RETURN', '  ame_fence();',
                    'nr_diag_mark(NR_DIAG_SYNC_DONE', 'nr_diag_mark(NR_DIAG_COLLECT_BEGIN',
                    'int status = collect(', 'nr_diag_mark(NR_DIAG_COLLECT_DONE')]
                self.assertEqual(stages, sorted(stages))
                self.assertLess(blocking.rfind('qwen_hang_reset(position)', 0, call), call)
            args.hang_console = 'bounded'
            bounded, plan = image.generate(report, segment, args)
            self.assertEqual(bounded, blocking)  # Only runtime compilation changes.
            self.assertEqual(plan['hang_console'], 'bounded')
            self.assertIn('invalidating', ' '.join(plan['hang_watch_limits']))

    def test_rejects_unobservable_and_conflicting_configurations(self):
        with tempfile.TemporaryDirectory() as directory:
            args, report, segment = self.fixture(directory)
            args.hang_console = 'bounded'
            with self.assertRaisesRegex(ValueError, 'requires --hang-watch'):
                image.generate(report, segment, args)
            args.hang_watch = SYMBOL + ':14'
            for field, value in (('profile_kernels', True), ('intermediate_arrays', Path('probe'))):
                setattr(args, field, value)
                with self.assertRaisesRegex(ValueError, 'cannot be combined'):
                    image.generate(report, segment, args)
                setattr(args, field, None)
            args.adapters.write_text(f'extern void {SYMBOL}(MemRef1 *, MemRef2 *, MemRef2 *);\n')
            with self.assertRaisesRegex(ValueError, 'unsupported --hang-watch ABI'):
                image.generate(report, segment, args)

    def test_graph_fence_changes_only_post_graph_sync(self):
        with tempfile.TemporaryDirectory() as directory:
            args, report, segment = self.fixture(directory)
            args.hang_watch = SYMBOL + ':14'
            baseline, baseline_plan = image.generate(report, segment, args)
            args.graph_sync = 'fence'
            source, plan = image.generate(report, segment, args)
            fence = '  __asm__ volatile ("fence rw, rw" ::: "memory");'
            self.assertEqual(source, baseline.replace('  ame_fence();', fence))
            self.assertEqual(source.count(fence), 2)
            self.assertIsNone(plan['completion_sync'])  # No profiler resync remains.
            self.assertEqual(baseline_plan['graph_completion_sync'], 'ame-resync')
            self.assertEqual(plan['graph_completion_sync'], 'fence')
            for kind in ('prefill', 'decode'):
                call = source.index(f'  _mlir_ciface_forward_{kind}(&result,')
                stages = [source.index(item, call) for item in (
                    'nr_diag_mark(NR_DIAG_GRAPH_RETURN', fence,
                    'nr_diag_mark(NR_DIAG_SYNC_DONE', 'int status = collect(',
                    'nr_heap_reset(mark);')]
                self.assertEqual(stages, sorted(stages))
            args.graph_sync = 'none'
            with self.assertRaisesRegex(ValueError, '--graph-sync'):
                image.generate(report, segment, args)


if __name__ == '__main__':
    unittest.main()
