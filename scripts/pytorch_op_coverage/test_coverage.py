"""Regression tests for denominator integrity, evidence and worker failures."""

import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
classify = importlib.import_module("classify")
parse = importlib.import_module("parse_frontend")
runner = importlib.import_module("run_coverage")
report = importlib.import_module("report")
REPO = HERE.parents[1]


def passed(case_id):
    return {
        "case_id": case_id,
        "status": "passed",
        **dict.fromkeys(classify.STAGES, "passed"),
    }


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.target = parse.load_target_op_set(HERE / "data/target_ops_v1.json")
        self.records = runner.run_static(REPO, self.target)

    def test_alias_is_not_direct_mapping(self):
        item = next(
            r for r in self.records if r.operator == "aten::reshape.default"
        )
        self.assertFalse(item.frontend_recognized)
        self.assertEqual(item.static_status, "alias_candidate")
        self.assertFalse(item.lowering_dialects)
        self.assertTrue(item.alias_candidates)

    def test_live_pass_and_failure_do_not_change_source_counts(self):
        before = classify.summarize(self.records)
        for r in self.records:
            r.cases = [passed(case_id) for case_id in r.required_cases]
        self.records[0].cases[0]["status"] = "failed"
        after = classify.summarize(self.records)
        for key in (
            "frontend_recognized",
            "registered_lowering",
            "alias_candidate",
            "known_limited",
            "unmapped",
            "total_ops",
        ):
            self.assertEqual(before[key], after[key])
        limited = next(
            r for r in self.records if r.operator == "aten::topk.default"
        )
        self.assertFalse(limited.validated())
        self.assertTrue(limited.known_limitations)

    def test_every_stage_and_required_case_is_needed(self):
        item = classify.OpCoverageRecord(
            "aten::mm.default", [], required_cases=["a", "b"]
        )
        item.cases = [passed("a")]
        self.assertFalse(item.validated())
        item.cases.append(passed("b"))
        self.assertTrue(item.validated())
        for stage in classify.STAGES:
            item.cases[1][stage] = "not_run"
            self.assertFalse(item.validated())
            item.cases[1][stage] = "passed"
        item.cases.append(passed("b"))
        self.assertFalse(item.validated())

    def test_trace_never_counts_as_end_to_end(self):
        for r in self.records:
            r.cases = [
                dict(passed(case_id), compiled="not_run", correctness="not_run")
                for case_id in r.required_cases
            ]
        self.assertEqual(
            classify.summarize(self.records)["validated_for_profile"], 0
        )

    def test_skip_keeps_denominator_and_fails_threshold(self):
        self.assertEqual(len(self.records), 106)
        for r in self.records:
            r.cases = [
                runner.not_run_case("unconfigured", "skipped", "no input")
            ]
        env = {"torch": "fixture", "buddy": True}
        self.assertEqual(
            runner.exit_status("live", self.records, [], env, 90), 1
        )
        self.assertEqual(runner.exit_status("live", self.records, [], env), 0)

    def test_stack_failure_exit_two_and_visible_markdown(self):
        with patch.object(
            runner,
            "isolated_worker",
            return_value={
                "torch": None,
                "buddy": False,
                "error": "SENTINEL_MISSING_STACK",
            },
        ):
            env, workloads = runner.run_measurements(
                REPO, self.records, "live", 1, True
            )
        code = runner.exit_status("live", self.records, workloads, env)
        self.assertEqual(code, 2)
        payload = report.build_report_payload(
            self.records,
            target=self.target,
            mode="live",
            provenance={
                "git_revision": "fixture",
                "dirty": True,
                "source_sha256": "fixture",
                "python": "fixture",
            },
            environment=env,
            workloads=workloads,
            exit_code=code,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            report.write_markdown(payload, path)
            content = path.read_text()
        self.assertIn("SENTINEL_MISSING_STACK", content)
        self.assertIn("**blocked**", content)
        self.assertIn("moe_block", content)
        self.assertEqual(payload["summary"]["validated_for_profile"], 0)

    def test_failure_reason_is_visible_in_markdown(self):
        self.records[0].cases = [
            runner.not_run_case("a", "failed", "sentinel | failure\nnext line")
        ]
        payload = report.build_report_payload(
            self.records,
            target=self.target,
            mode="live",
            provenance={
                "git_revision": "fixture",
                "dirty": True,
                "source_sha256": "fixture",
                "python": "fixture",
            },
            environment={},
            workloads=[],
            exit_code=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            report.write_markdown(payload, path)
            self.assertIn("sentinel \\| failure<br>next line", path.read_text())

    def test_stale_installed_frontend_blocks_live_measurement(self):
        with patch.object(
            runner,
            "isolated_worker",
            return_value={
                "torch": "fixture",
                "buddy": True,
                "buddy_source_sha256": {"frontend.py": "stale"},
            },
        ) as worker:
            env, workloads = runner.run_measurements(
                REPO, self.records, "live", 1, True
            )
        self.assertEqual(worker.call_count, 1)
        self.assertEqual(
            runner.exit_status("live", self.records, workloads, env), 2
        )
        self.assertIn("sources differ", env["error"])

    def test_moe_membership_is_not_only_first_family(self):
        mm = next(r for r in self.records if r.operator == "aten::mm.default")
        self.assertIn("moe_critical", mm.families)
        self.assertEqual(
            classify.summarize(self.records)["moe_critical"]["total_ops"], 47
        )

    def test_repeat_measurements_replace_case_results(self):
        subset = [self.records[0]]
        with patch.object(
            runner,
            "isolated_worker",
            return_value={"torch": None, "error": "missing"},
        ):
            runner.run_measurements(REPO, subset, "live", 1, False)
            runner.run_measurements(REPO, subset, "live", 1, False)
        self.assertEqual(len(subset[0].cases), 3)

    def test_schema_drift_even_on_untested_operator_fails(self):
        row = next(r for r in self.records if not r.required_cases)
        env = {
            "torch": "fixture",
            "buddy": True,
            "schemas": {row.operator: "different schema"},
        }
        with patch.object(runner, "isolated_worker", return_value=env):
            actual, _ = runner.run_measurements(REPO, [row], "live", 1, False)
        self.assertEqual(runner.exit_status("live", [row], [], actual), 1)

    def test_workload_inventory_preserves_outside_target(self):
        case = {"observed_ops": {"aten::outside.default": 2}}
        workloads = [{"name": "moe_block", "cases": [case, case]}]
        gap = report.workload_gaps(workloads, self.records)[0]
        self.assertFalse(gap["in_target"])
        self.assertEqual(gap["workloads"]["moe_block"], 2)
        self.assertEqual(len(self.records), 106)


class TargetTests(unittest.TestCase):
    def load(self, target):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "target.json"
            path.write_text(json.dumps(target))
            return parse.load_target_op_set(path)

    def test_empty_and_unqualified_targets_rejected(self):
        for target in (
            {},
            {"name": "x", "version": "1", "families": {"x": {"ops": []}}},
            {
                "name": "x",
                "version": "1",
                "families": {"x": {"ops": ["mm.default"]}},
            },
        ):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.load(target)

    def test_deduplication_preserves_families(self):
        rows = parse.flatten_target_ops(
            {
                "families": {
                    "a": {"ops": ["aten::mm.default"]},
                    "b": {"ops": ["aten::mm.default"]},
                }
            }
        )
        self.assertEqual(
            rows, [{"operator": "aten::mm.default", "families": ["a", "b"]}]
        )

    def test_unknown_metadata_key_rejected(self):
        target = {
            "name": "x",
            "version": "1",
            "families": {"x": {"ops": ["aten::mm.default"]}},
            "schemas": {"aten::outside.default": "x"},
        }
        with self.assertRaises(ValueError):
            self.load(target)

    def test_v1_schema_snapshot_and_migration(self):
        target = parse.load_target_op_set(HERE / "data/target_ops_v1.json")
        keys = {r["operator"] for r in parse.flatten_target_ops(target)}
        self.assertEqual(keys, set(target["schemas"]))
        self.assertIn("prims::convert_element_type.default", keys)
        self.assertNotIn("aten::fill_cache.default", keys)
        self.assertIn("Tensor", target["schemas"]["aten::mm.default"])
        self.assertEqual(target["migration_from_v0"]["original_count"], 108)


class IsolationTests(unittest.TestCase):
    def test_timeout_preserves_last_completed_stage(self):
        def timeout(command, **kwargs):
            path = Path(command[command.index("--result") + 1])
            path.write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "imported": "passed",
                        "active_stage": "lowered",
                    }
                )
            )
            raise subprocess.TimeoutExpired(command, 1)

        with patch.object(runner.subprocess, "run", side_effect=timeout):
            result = runner.isolated_worker(REPO, {"mode": "live"}, 1)
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["imported"], "passed")
        self.assertEqual(result["active_stage"], "lowered")

    def test_native_crash_is_failure(self):
        with patch.object(
            runner.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=-11),
        ):
            result = runner.isolated_worker(REPO, {"mode": "live"}, 1)
        self.assertEqual(result["status"], "failed")
        self.assertIn("-11", result["reason"])

    def test_malformed_or_missing_worker_payload_is_failure(self):
        for payload in ("[]", "{}", '{"status":"nonsense"}', "not json"):

            def complete(command, payload=payload, **kwargs):
                Path(command[command.index("--result") + 1]).write_text(payload)
                return SimpleNamespace(returncode=0)

            with (
                self.subTest(payload=payload),
                patch.object(runner.subprocess, "run", side_effect=complete),
            ):
                self.assertEqual(
                    runner.isolated_worker(REPO, {"mode": "live"}, 1)["status"],
                    "failed",
                )


if __name__ == "__main__":
    unittest.main()
