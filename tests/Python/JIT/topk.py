# RUN: %PYTHON %s
"""Top-k selects valid distinct indices, including NaNs, ties and strided views."""

import math

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


class TopK(torch.nn.Module):
    def __init__(self, k, dim, largest, sorted_result, view="none"):
        super().__init__()
        self.k = k
        self.dim = dim
        self.largest = largest
        self.sorted_result = sorted_result
        self.view = view

    def source(self, x):
        if self.view == "slice":
            return x[..., 1::2]
        if self.view == "transpose":
            return x.transpose(-1, -2)
        return x

    def forward(self, x):
        return torch.topk(
            self.source(x), self.k, self.dim, self.largest, self.sorted_result
        )


def check(x, model):
    original = x.clone()
    reference_values, _ = model(x)
    exported = torch.export.export(model, (x,), strict=True)
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry, enable_external_calls=False
    )
    compiler._compile_fx(exported.graph_module, [x])
    actual = compiler.dynamo_run()(x)
    assert len(actual) == 2
    values, indices = actual
    source = model.source(x)
    assert values.shape == reference_values.shape
    assert values.dtype == source.dtype and indices.dtype == torch.int64
    assert indices.shape == values.shape
    if source.ndim == 0:
        assert indices.item() == 0
        selected = source
    else:
        axis = model.dim % source.ndim
        assert bool(((indices >= 0) & (indices < source.shape[axis])).all())
        ordered_indices = indices.sort(axis).values
        assert not bool((ordered_indices.diff(dim=axis) == 0).any())
        selected = torch.gather(source, axis, indices)
    torch.testing.assert_close(values, selected, rtol=0, atol=0, equal_nan=True)
    # Tie indices and the order for sorted=False are not specified by PyTorch.
    if not model.sorted_result and source.ndim:
        values = values.sort(model.dim).values
        reference_values = reference_values.sort(model.dim).values
    torch.testing.assert_close(
        values, reference_values, rtol=0, atol=0, equal_nan=True
    )
    torch.testing.assert_close(x, original, rtol=0, atol=0, equal_nan=True)


torch.set_num_threads(1)
count = 0
for dtype in (torch.float32, torch.float64, torch.int32, torch.int64):
    for shape in ((5,), (3, 5), (2, 3, 4), (), (0, 3), (2, 0)):
        x = (torch.arange(math.prod(shape)).reshape(shape) % 7 - 3).to(dtype)
        for axis in range(max(len(shape), 1)):
            extent = shape[axis] if shape else 1
            for k in sorted({0, min(1, extent), extent}):
                for largest in (False, True):
                    for ordered in (False, True):
                        check(
                            x,
                            TopK(
                                k, axis - max(len(shape), 1), largest, ordered
                            ),
                        )
                        count += 1
    if dtype.is_floating_point:
        patterns = (
            [float("nan"), 3, 1, float("nan"), -2],
            [float("-inf")] * 5,
            [float("inf")] * 5,
            [0.0, -0.0, 2, 2, 2],
        )
    else:
        limits = torch.iinfo(dtype)
        patterns = ([limits.min] * 5, [limits.max] * 5, [2, 2, -3, -3, 0])
    for pattern in patterns:
        for largest in (False, True):
            for k in (2, 5):
                check(
                    torch.tensor(pattern, dtype=dtype),
                    TopK(k, -1, largest, True),
                )
                count += 1
    x = torch.arange(2 * 3 * 7).reshape(2, 3, 7).to(dtype)
    for view in ("slice", "transpose"):
        for axis in range(3):
            for largest in (False, True):
                check(x, TopK(2, axis, largest, True, view))
                count += 1
print(f"Top-k: {count} cases passed")
