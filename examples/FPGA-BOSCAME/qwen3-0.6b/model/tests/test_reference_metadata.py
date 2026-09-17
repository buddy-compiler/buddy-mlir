"""Reject a numerical oracle for a different deployment or generation trajectory."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import test_image_abi as fixtures
from test_numeric_oracle import fixture


class ReferenceMetadataTests(unittest.TestCase):
    def test_oracle_identity_and_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            args, _, _ = fixtures.ImageABI().setup_case(directory, 1)
            args.reference_arrays = p / 'arrays.npz'
            args.reference_atol, args.reference_mean_atol = .001, .0001
            _, arrays, _ = fixture()
            np.savez(args.reference_arrays, **arrays)
            metadata = dict(prompt_ids=args.prompt_ids, layers=1, capacity=32,
                            arithmetic_profile='nr-fpga', prefill_argmax_last=42,
                            decode_steps_recorded=[dict(step=i, cache_position=16+i,
                                                       input_token=42, generated_token=42)
                                                   for i in range(8)])
            path = p / 'quant-reference.json'
            path.write_text(json.dumps(metadata))
            result = fixtures.image.prepare_reference(args, p)
            self.assertTrue(result['trajectory_verified'])
            self.assertEqual(result['prompt_ids'], args.prompt_ids)
            mutations = []
            for name, value in [('prompt_ids', [1]*16), ('layers', 4), ('capacity', 512),
                                ('arithmetic_profile', 'triton-host'),
                                ('prefill_argmax_last', 43), ('decode_steps_recorded', [])]:
                mutations.append(dict(metadata, **{name: value}))
            for name in ('step', 'cache_position', 'input_token', 'generated_token'):
                changed = copy.deepcopy(metadata)
                changed['decode_steps_recorded'][7][name] += 1
                mutations.append(changed)
            for changed in mutations:
                with self.subTest(metadata=changed):
                    path.write_text(json.dumps(changed))
                    with self.assertRaisesRegex(ValueError, 'reference|FPGA oracle'):
                        fixtures.image.prepare_reference(args, p)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                fixtures.image.prepare_reference(args, p)


if __name__ == '__main__':
    unittest.main()
