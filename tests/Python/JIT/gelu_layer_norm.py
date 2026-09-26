# RUN: %PYTHON %s

"""Validate direct CPU GELU/layer-norm lowering without a decomposition table."""

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


class Case(torch.nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, *args):
        return self.function(*args)


def check(function, args):
    model = Case(function)
    exported = torch.export.export(model, args, strict=True)
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry, enable_external_calls=False
    )
    compiler._compile_fx(exported.graph_module, list(args))
    assert len(compiler.imported_graphs) == 1
    actual = compiler.dynamo_run()(*args)
    expected = model(*args)
    expected = expected if isinstance(expected, tuple) else (expected,)
    assert len(actual) == len(expected)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=1e-4, atol=1e-5)


torch.manual_seed(0)
torch.set_num_threads(1)
count = 0
for dtype in (torch.float32, torch.float64):
    for approximate in ("none", "tanh"):
        for value in (
            torch.tensor(-0.5, dtype=dtype),
            torch.linspace(-8, 8, 33, dtype=dtype),
            torch.randn(3, 5, dtype=dtype),
        ):
            check(
                lambda x, approximate=approximate: torch.ops.aten.gelu.default(
                    x, approximate=approximate
                ),
                (value,),
            )
            count += 1
    for shape, normalized, constant in (
        ((3, 5), (5,), False),
        ((2, 3, 4), (3, 4), False),
        ((4,), (4,), False),
        ((2, 1), (1,), True),
    ):
        value = (
            torch.full(shape, 3.0, dtype=dtype)
            if constant
            else torch.randn(shape, dtype=dtype)
        )
        for has_weight, has_bias in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            args = [value]
            if has_weight:
                args.append(torch.randn(normalized, dtype=dtype))
            if has_bias:
                args.append(torch.randn(normalized, dtype=dtype))

            def layer_norm(
                x,
                *affine,
                has_weight=has_weight,
                has_bias=has_bias,
                normalized=normalized,
            ):
                weight = affine[0] if has_weight else None
                bias = affine[-1] if has_bias else None
                return torch.ops.aten.native_layer_norm.default(
                    x, normalized, weight, bias, 1e-5
                )

            check(layer_norm, tuple(args))
            count += 1
    check(
        lambda x: torch.ops.aten.native_layer_norm.default(
            x, [4], None, None, 0.25
        ),
        (torch.full((2, 4), 3.0, dtype=dtype),),
    )
    count += 1

for function in (
    torch.ops.aten.gelu.default,
    lambda x: torch.ops.aten.native_layer_norm.default(
        x, [4], None, None, 1e-5
    ),
):
    try:
        check(function, (torch.ones(2, 4, dtype=torch.float16),))
    except NotImplementedError as error:
        assert "requires f32 or f64" in str(error)
    else:
        raise AssertionError("Unsupported half precision must be rejected")
print(f"GELU and layer norm: {count} CPU JIT cases passed")
