"""Export signature binding tests without model downloads or a Buddy build."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_model_validation import lifted_inputs


def exported(specs, state=None, constants=None):
    return SimpleNamespace(
        graph_signature=SimpleNamespace(
            input_specs=[
                SimpleNamespace(kind=SimpleNamespace(name=kind), target=target)
                for kind, target in specs
            ]
        ),
        state_dict=state or {},
        constants=constants or {},
    )


class SignatureTests(unittest.TestCase):
    def test_lifted_state_and_nonpersistent_buffers_keep_signature_order(self):
        graph = exported(
            [
                ("PARAMETER", "weight"),
                ("USER_INPUT", None),
                ("BUFFER", "running"),
                ("BUFFER", "positions"),
                ("CONSTANT_TENSOR", "constant"),
                ("USER_INPUT", None),
            ],
            {"weight": "w", "running": "b"},
            {"positions": "p", "constant": "c"},
        )
        self.assertEqual(
            lifted_inputs(graph, ["x", "y"]), ["w", "x", "b", "p", "c", "y"]
        )

    def test_missing_state_is_not_replaced_by_an_input(self):
        with self.assertRaises(KeyError):
            lifted_inputs(exported([("BUFFER", "missing")]), ["x"])

    def test_input_count_must_match(self):
        graph = exported([("USER_INPUT", None)])
        with self.assertRaises(StopIteration):
            lifted_inputs(graph, [])
        with self.assertRaisesRegex(ValueError, "Unused model inputs"):
            lifted_inputs(graph, ["x", "extra"])

    def test_unknown_signature_kind_fails_explicitly(self):
        with self.assertRaisesRegex(
            ValueError, "Unsupported exported input kind"
        ):
            lifted_inputs(exported([("TOKEN", None)]), [])


if __name__ == "__main__":
    unittest.main()
