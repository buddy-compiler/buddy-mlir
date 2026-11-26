# RUN: %PYTHON %s
"""Reuse the fixed-shape MoE graph across changed routing and expert weights.

This tests the coverage workload, not dynamic-size dispatch in model libraries.
"""

import sys
from pathlib import Path

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[3] / "scripts/pytorch_op_coverage"),
)
from probes import build_workload


def reference(args):
    x, gate, up, down = args
    weights, experts = torch.softmax(x @ gate, -1).topk(2, -1)
    weights = weights / weights.sum(-1, keepdim=True)
    result = torch.zeros_like(x)
    for token in range(x.shape[0]):
        for slot in range(2):
            expert = int(experts[token, slot])
            hidden = torch.nn.functional.silu(x[token] @ up[expert])
            result[token] += weights[token, slot] * (hidden @ down[expert])
    return result, experts


torch.set_num_threads(1)
torch.manual_seed(0)
count = 0
for dtype in (torch.float32, torch.float64):
    for tokens in (1, 5):
        for hidden in (4, 8):
            x = torch.zeros(tokens, hidden, dtype=dtype)
            x[:, :4] = torch.tensor([4, 3, 2, 1], dtype=dtype)
            gate = torch.zeros(hidden, 4, dtype=dtype)
            gate[:4] = torch.eye(4, dtype=dtype)
            up = torch.randn(4, hidden, 2 * hidden, dtype=dtype) * 0.1
            down = torch.randn(4, 2 * hidden, hidden, dtype=dtype) * 0.1
            model, _ = build_workload(
                "moe_block",
                x,
                lambda *shape, dtype=dtype: torch.randn(*shape, dtype=dtype),
            )
            first = (x, gate, up, down)
            exported = torch.export.export(model, first, strict=True)
            compiler = DynamoCompiler(
                primary_registry=tosa.ops_registry, enable_external_calls=False
            )
            compiler._compile_fx(exported.graph_module, list(first))
            execute = compiler.dynamo_run()

            changed = x.clone()
            changed[:, :4] = x[:, :4].flip(-1)
            distributed = x.clone()
            for token in range(tokens):
                distributed[token, :4] = x[token, :4].roll(token % 4)
            batches = (
                first,
                (changed, gate, up, down),
                (distributed, gate, up, down),
                (x, gate.roll(2, dims=1), up * 0.5, down * 1.5),
            )
            initial_experts = reference(first)[1]
            assert initial_experts.unique().numel() == 2
            assert not torch.equal(initial_experts, reference(batches[1])[1])
            assert not torch.equal(initial_experts, reference(batches[3])[1])
            if tokens == 5:
                assert reference(batches[2])[1].unique().numel() == 4
            for args in batches:
                original = tuple(a.clone() for a in args)
                expected, _ = reference(args)
                actual = execute(*args)
                assert len(actual) == 1
                torch.testing.assert_close(
                    actual[0], expected, rtol=1e-4, atol=1e-5
                )
                for a, before in zip(args, original):
                    torch.testing.assert_close(a, before, rtol=0, atol=0)
                count += 1
print(f"MoE routing reuse: {count} cases passed across 8 compiled graphs")
