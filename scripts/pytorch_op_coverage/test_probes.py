"""Optional PyTorch-only checks for fixtures and the JIT adapter contract."""

import ast
import importlib
import importlib.util
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
probes = importlib.import_module("probes")
worker = importlib.import_module("worker")


@contextmanager
def mock_modules(modules):
    # Restore only injected names; torch lazily loads native registrations.
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


@unittest.skipUnless(
    importlib.util.find_spec("torch"), "requires optional torch"
)
class TorchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.torch = torch
        torch.set_num_threads(1)
        # Exercise the real existing flatten helper without importing its MLIR
        # dependencies. The production adapter imports the original module.
        source = (
            HERE.parents[1]
            / "tests/Python/AtenOpsCoverage/aten_coverage_runner.py"
        ).read_text()
        fn = next(
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef) and n.name == "_flatten_outputs"
        )
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__",
                    names=[ast.alias(name="annotations")],
                    level=0,
                ),
                fn,
            ],
            type_ignores=[],
        )
        namespace = {"torch": torch}
        exec(
            compile(
                ast.fix_missing_locations(module), "flatten-helper", "exec"
            ),
            namespace,
        )
        cls.helper = SimpleNamespace(
            _flatten_outputs=namespace["_flatten_outputs"]
        )

    def test_all_fixtures_eager_and_export(self):
        torch = self.torch
        from torch.utils._pytree import tree_leaves

        for name in (*probes.OPERATORS, *probes.WORKLOADS):
            for profile in probes.PROFILES:
                with self.subTest(name=name, profile=profile), torch.no_grad():
                    module, args = probes.build_case(name, profile)
                    expected = module(*args)
                    exported = torch.export.export(module, args, strict=True)
                    actual = exported.module()(*args)
                    expected_leaves, actual_leaves = (
                        tree_leaves(expected),
                        tree_leaves(actual),
                    )
                    self.assertEqual(len(expected_leaves), len(actual_leaves))
                    for ref, out in zip(expected_leaves, actual_leaves):
                        torch.testing.assert_close(out, ref, equal_nan=False)

    def test_moe_matches_explicit_token_expert_reference(self):
        torch = self.torch
        for profile in probes.PROFILES:
            module, args = probes.build_case("moe_block", profile)
            x, gate, up, down = args
            weights, experts = torch.softmax(x @ gate, -1).topk(2, -1)
            weights = weights / weights.sum(-1, keepdim=True)
            reference = torch.zeros_like(x)
            for token in range(x.shape[0]):
                for slot in range(2):
                    expert = int(experts[token, slot])
                    hidden = torch.nn.functional.silu(x[token] @ up[expert])
                    reference[token] += weights[token, slot] * (
                        hidden @ down[expert]
                    )
            torch.testing.assert_close(module(*args), reference)

    def test_compare_rejects_arity_dtype_order_and_values(self):
        torch = self.torch
        expected = (torch.tensor([1.0]), torch.tensor([2.0]))
        bad_outputs = [
            expected[:1],
            expected + (expected[0],),
            tuple(reversed(expected)),
            (expected[0].double(), expected[1]),
            (torch.tensor([10.0]), expected[1]),
        ]
        with mock_modules({"aten_coverage_runner": self.helper}):
            worker.compare_outputs(expected, expected, torch)
            for actual in bad_outputs:
                with (
                    self.subTest(actual=actual),
                    self.assertRaises(AssertionError),
                ):
                    worker.compare_outputs(expected, actual, torch)
            with self.assertRaises(AssertionError):
                worker.compare_outputs(
                    torch.tensor([100000]), torch.tensor([100001]), torch
                )

    def test_adapter_stage_boundaries_with_fake_backend(self):
        # This validates orchestration only, and is never a coverage measurement.
        for failure in (
            None,
            "imported",
            "lowered",
            "compiled",
            "executed",
            "correctness",
            "multiple_graphs",
        ):

            class Compiler:
                failure_mode = failure

                def __init__(self, **kwargs):
                    self.imported_graphs = []

                def _compile_fx(self, gm, args):
                    if self.failure_mode == "imported":
                        raise RuntimeError("import fixture failure")
                    self.gm = gm

                    def lower():
                        if self.failure_mode == "lowered":
                            raise RuntimeError("lower fixture failure")

                    self.imported_graphs = [
                        SimpleNamespace(lower_to_top_level_ir=lower, body=[])
                    ]
                    if self.failure_mode == "multiple_graphs":
                        self.imported_graphs *= 2

                def dynamo_run(self):
                    if self.failure_mode == "compiled":
                        raise RuntimeError("compile fixture failure")

                    def execute(*args):
                        if self.failure_mode == "executed":
                            raise RuntimeError("execution fixture failure")
                        if self.failure_mode == "correctness":
                            return []
                        return self.gm(*args)

                    return execute

            modules = {
                "buddy.compiler.frontend": SimpleNamespace(
                    DynamoCompiler=Compiler
                ),
                "buddy.compiler.ops": SimpleNamespace(
                    tosa=SimpleNamespace(ops_registry={})
                ),
                "aten_coverage_runner": self.helper,
            }
            with (
                self.subTest(failure=failure),
                mock_modules(modules),
                tempfile.TemporaryDirectory() as directory,
            ):
                result = worker.run_case(
                    "aten::add.Tensor",
                    "small-f32",
                    "live",
                    Path(directory) / "result.json",
                )
                if failure is None:
                    self.assertEqual(result["status"], "passed")
                    self.assertEqual(result["correctness"], "passed")
                else:
                    self.assertEqual(result["status"], "failed")
                    stage = (
                        "imported" if failure == "multiple_graphs" else failure
                    )
                    self.assertEqual(result[stage], "failed")
                    self.assertNotEqual(result["correctness"], "passed")


if __name__ == "__main__":
    unittest.main()
