"""Timing totals, weighted per-call means and explicit graph remainder."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from summarize_kernel_timing import aggregate, PREFIX


def fixture():
    cases = {name: {"family": "matmul_i8"} for name in ("prefill", "decode", "shared")}
    stages = []
    for i in range(9):
        kernels = [{"symbol": PREFIX + ("prefill" if i == 0 else "decode"),
                    "calls": 2, "cycles": 20 + i},
                   {"symbol": PREFIX + "shared", "calls": 1, "cycles": 10}]
        stages.append({"kind": "prefill" if i == 0 else "decode",
                       "graph_start_position": 0 if i == 0 else 15 + i,
                       "position": 15 + i, "compute_cycles": 40 + i,
                       "kernel_cycles": 30 + i, "graph_cycles_outside_kernel_measurements": 10,
                       "kernels": kernels})
    return stages, cases


class TimingSummary(unittest.TestCase):
    def test_complete_totals_and_phase_weighting(self):
        stages, cases = fixture()
        result = aggregate(stages, cases, 1000)
        family = result["families"][0]
        self.assertEqual(family["prefill_milliseconds"], 30)
        self.assertEqual(family["decode_mean_milliseconds"], 34.5)
        self.assertEqual(family["decode_calls_per_step"], 3)
        self.assertEqual(family["decode_mean_call_milliseconds"], 11.5)
        self.assertEqual(result["graph_outside_kernels"]["decode_mean_milliseconds"], 10)
        self.assertAlmostEqual(family["decode_percent_graph"] +
                               result["graph_outside_kernels"]["decode_percent_graph"], 100)
        self.assertEqual(sum(k["total_cycles"] for k in result["kernels"]), family["total_cycles"])
        self.assertEqual(len(result["per_stage_kernel"]), 18)
        prefill = next(k for k in result["kernels"] if k["name"] == "prefill")
        self.assertIsNone(prefill["decode_mean_call_milliseconds"])

    def test_reject_partial_inconsistent_and_unmatched_evidence(self):
        stages, cases = fixture()
        bad_total = deepcopy(stages)
        bad_total[0]["kernel_cycles"] += 1
        for invalid in (stages[:-1], stages[::-1], bad_total):
            with self.assertRaises(ValueError):
                aggregate(invalid, cases, 1000)
        with self.assertRaises(ValueError):
            aggregate(stages, {}, 1000)
        with self.assertRaises(ValueError):
            aggregate(stages, cases, 0)


if __name__ == "__main__":
    unittest.main()
