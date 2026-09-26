# RUN: %PYTHON %s
"""Compare CPU attention output and log-sum-exp with PyTorch."""

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


class Attention(torch.nn.Module):
    def __init__(self, causal, scale):
        super().__init__()
        self.causal = causal
        self.scale = scale

    def forward(self, q, k, v):
        return (
            torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(
                q, k, v, 0.0, self.causal, scale=self.scale
            )
        )


torch.set_num_threads(1)
torch.manual_seed(0)
count = 0
for dtype in (torch.float32, torch.float64):
    for length, source in ((3, 3), (2, 5), (5, 2), (1, 7)):
        for causal in (False, True):
            for scale in (None, 0.25):
                args = (
                    torch.randn(2, 2, length, 4, dtype=dtype),
                    torch.randn(2, 2, source, 4, dtype=dtype),
                    torch.randn(2, 2, source, 4, dtype=dtype),
                )
                originals = [x.clone() for x in args]
                model = Attention(causal, scale)
                expected = model(*args)
                exported = torch.export.export(model, args, strict=True)
                compiler = DynamoCompiler(
                    primary_registry=tosa.ops_registry,
                    enable_external_calls=False,
                )
                compiler._compile_fx(exported.graph_module, list(args))
                actual = compiler.dynamo_run()(*args)
                assert len(actual) == len(expected) == 2
                for result, reference in zip(actual, expected):
                    torch.testing.assert_close(
                        result, reference, rtol=1e-4, atol=1e-5
                    )
                for arg, original in zip(args, originals):
                    torch.testing.assert_close(arg, original, rtol=0, atol=0)
                count += 1
print(f"CPU attention: {count} cases passed")
