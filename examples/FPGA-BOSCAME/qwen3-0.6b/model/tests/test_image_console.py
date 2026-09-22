"""Console transport changes must not change the generated model program."""
import tempfile
import unittest

import test_image_abi as fixture

image = fixture.image


class ImageConsoleTests(unittest.TestCase):
    def test_same_capacity_transport_comparison_preserves_model_source(self):
        with tempfile.TemporaryDirectory() as directory:
            args, report, segment = fixture.ImageABI().setup_case(directory, 1)
            args.console_capacity = 524288
            ring, ring_plan = image.generate(report, segment, args)
            args.console_mode = 'append-only'
            append, append_plan = image.generate(report, segment, args)
            self.assertEqual(ring, append)
            self.assertTrue(ring_plan['console']['ra_waits_for_nh_consumed'])
            self.assertFalse(append_plan['console']['ra_waits_for_nh_consumed'])
            self.assertEqual(append_plan['console']['capacity_bytes'], 524288)
            self.assertIn('FAIL', append_plan['console']['overflow_policy'])
            del ring_plan['console'], append_plan['console']
            self.assertEqual(ring_plan, append_plan)

    def test_existing_default_and_append_default(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _ = fixture.ImageABI().setup_case(directory, 1)
            self.assertEqual(image.console_config(args)['capacity_bytes'], 65536)
            self.assertEqual(image.console_config(args)['drain'], 'live')
            args.console_mode = 'append-only'
            self.assertEqual(image.console_config(args)['capacity_bytes'], 524288)

    def test_deferred_drain_preserves_graph_and_records_observation_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            args, report, segment = fixture.ImageABI().setup_case(directory, 1)
            args.console_mode = 'append-only'
            args.console_capacity = 4194304
            live, live_plan = image.generate(report, segment, args)
            args.console_drain = 'after-completion'
            deferred, deferred_plan = image.generate(report, segment, args)
            self.assertEqual(live, deferred)
            console = deferred_plan['console']
            self.assertEqual(console['drain'], 'after-completion')
            self.assertFalse(console['nh_console_and_input_service_during_compute'])
            self.assertIn('only after RA_SIGNAL', console['drain_limits'])
            self.assertIn('stalled run', console['drain_limits'])
            del live_plan['console'], deferred_plan['console']
            self.assertEqual(live_plan, deferred_plan)

    def test_deferred_drain_rejects_incompatible_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            args, report, segment = fixture.ImageABI().setup_case(directory, 1)
            args.console_drain = 'invalid'
            with self.assertRaisesRegex(ValueError, 'console-drain'):
                image.generate(report, segment, args)
            args.console_drain = 'after-completion'
            with self.assertRaisesRegex(ValueError, 'requires finite'):
                image.generate(report, segment, args)
            args.console_mode = 'append-only'
            for attribute, value in (('interactive', True), ('hang_watch', 'kernel:0'),
                                     ('profile_watch', True), ('uart_probe', True)):
                previous = getattr(args, attribute, None)
                setattr(args, attribute, value)
                with self.subTest(attribute=attribute), self.assertRaisesRegex(ValueError, 'console-drain'):
                    image.generate(report, segment, args)
                setattr(args, attribute, previous)

    def test_invalid_capacity_and_incompatible_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            args, report, segment = fixture.ImageABI().setup_case(directory, 1)
            for capacity in (0, -64, 32, 65, (1 << 30) + 1, True):
                args.console_capacity = capacity
                with self.subTest(capacity=capacity), self.assertRaisesRegex(ValueError, 'console-capacity'):
                    image.generate(report, segment, args)
            args.console_capacity = 524288
            args.console_mode = 'append-only'
            args.interactive = True
            with self.assertRaisesRegex(ValueError, 'finite validation'):
                image.generate(report, segment, args)
            args.interactive = False
            args.hang_console = 'bounded'
            with self.assertRaisesRegex(ValueError, 'requires --console-mode=ring'):
                image.generate(report, segment, args)


if __name__ == '__main__':
    unittest.main()
