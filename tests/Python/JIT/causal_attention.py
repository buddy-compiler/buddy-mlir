# RUN: %PYTHON %s
"""Compare CPU attention output and log-sum-exp with PyTorch."""

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


class Attention(torch.nn.Module):
    def __init__(self, causal, scale, sliced_mask=False):
        super().__init__()
        self.causal = causal
        self.scale = scale
        self.sliced_mask = sliced_mask

    def forward(self, q, k, v, mask=None):
        if self.sliced_mask:
            mask = mask[..., 1::2]
        return (
            torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(
                q, k, v, 0.0, self.causal, attn_mask=mask, scale=self.scale
            )
        )


def check(model, args):
    originals = [x.clone() for x in args]
    expected = model(*args)
    exported = torch.export.export(model, args, strict=True)
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry, enable_external_calls=False
    )
    compiler._compile_fx(exported.graph_module, list(args))
    actual = compiler.dynamo_run()(*args)
    assert len(actual) == len(expected) == 2
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=1e-4, atol=1e-5)
    for arg, original in zip(args, originals):
        torch.testing.assert_close(arg, original, rtol=0, atol=0)


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
                check(Attention(causal, scale), args)
                count += 1
    for length, source in ((2, 5), (5, 2)):
        args = (
            torch.randn(2, 3, length, 4, dtype=dtype),
            torch.randn(2, 3, source, 4, dtype=dtype),
            torch.randn(2, 3, source, 4, dtype=dtype),
        )
        for shape in (
            (length, source),
            (1, source),
            (length, 1),
            (1, 1),
            (2, 1, length, source),
            (1, 3, 1, source),
            (2, 3, length, 1),
            (2, 3, length, source),
        ):
            for pattern in ("finite", "mixed", "fully-masked"):
                sliced = pattern == "mixed"
                storage_shape = list(shape)
                if sliced:
                    storage_shape[-1] = storage_shape[-1] * 2 + 1
                mask = torch.randn(storage_shape, dtype=dtype)
                if pattern == "mixed":
                    mask[..., 0, :] = float("-inf")
                elif pattern == "fully-masked":
                    mask.fill_(float("-inf"))
                for causal in (False, True):
                    check(Attention(causal, 0.25, sliced), (*args, mask))
                    count += 1
                    if dtype == torch.float64:
                        check(
                            Attention(causal, 0.25, sliced),
                            (*args, mask.to(torch.float32)),
                        )
                        count += 1
print(f"CPU attention: {count} cases passed")
