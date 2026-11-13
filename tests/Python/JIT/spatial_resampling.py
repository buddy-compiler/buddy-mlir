# RUN: %PYTHON %s
"""Reflection padding and interpolation across shapes and internal views."""

import torch
from buddy.compiler.frontend import DynamoCompiler
from buddy.compiler.ops import tosa


class Spatial(torch.nn.Module):
    def __init__(self, mode, view=False, **options):
        super().__init__()
        self.mode = mode
        self.view = view
        self.options = options

    def forward(self, x):
        if self.view:
            x = x[..., ::2]
        if self.mode in ("reflect", "replicate", "circular", "constant"):
            return torch.nn.functional.pad(x, mode=self.mode, **self.options)
        return torch.nn.functional.interpolate(
            x, mode=self.mode, **self.options
        )


def check(model, x):
    ep = torch.export.export(model, (x,), strict=True)
    compiler = DynamoCompiler(
        primary_registry=tosa.ops_registry, enable_external_calls=False
    )
    compiler._compile_fx(ep.graph_module, [x])
    run = compiler.dynamo_run()
    for values in (x, -x + 0.25):
        original = values.clone()
        actual = run(values)
        assert len(actual) == 1
        torch.testing.assert_close(
            actual[0],
            model(values),
            rtol=1e-5,
            atol=1e-5,
            msg=f"{model.mode} {model.options} shape={tuple(x.shape)} view={model.view}",
        )
        torch.testing.assert_close(values, original, rtol=0, atol=0)


torch.set_num_threads(1)
torch.manual_seed(0)
count = 0
for dtype in (torch.float32, torch.float64):
    for shape in ((2, 5), (2, 3, 5)):
        for pad in ((0, 0), (1, 2), (4, 4), (-1, 2), (2, -1), (-1, -2)):
            for view in (False, True):
                source_shape = shape[:-1] + (shape[-1] * (2 if view else 1),)
                check(
                    Spatial("reflect", view, pad=pad),
                    torch.randn(source_shape, dtype=dtype),
                )
                count += 2
print(f"Reflection padding: {count} cases passed", flush=True)
count = 0
for dtype in (torch.float32, torch.float64):
    for shape in ((1, 2, 3, 5), (2, 1, 1, 4)):
        for mode in ("nearest", "bilinear"):
            for options in (
                {"size": (1, 1)},
                {"size": (7, 8)},
                {"scale_factor": (1.5, 2.25)},
            ):
                for align in (False, True) if mode == "bilinear" else (None,):
                    for view in (False, True):
                        source_shape = shape[:-1] + (
                            shape[-1] * (2 if view else 1),
                        )
                        kwargs = dict(options)
                        if align is not None:
                            kwargs["align_corners"] = align
                        check(
                            Spatial(mode, view, **kwargs),
                            torch.randn(source_shape, dtype=dtype),
                        )
                        count += 2
print(f"Interpolation: {count} cases passed", flush=True)

count = 0
for dtype in (torch.float32, torch.float64):
    for dimensions in (1, 2, 3):
        for batched in (False, True):
            shape = ((2, 2) if batched else (2,)) + (4,) * dimensions
            for mode in ("reflect", "replicate", "circular", "constant"):
                for pads in (
                    (1, 2) * dimensions,
                    (-1, 1) * dimensions,
                    (0, 0) * dimensions,
                ):
                    for view in (False, True):
                        source_shape = shape[:-1] + (
                            shape[-1] * (2 if view else 1),
                        )
                        options = {"pad": pads}
                        if mode == "constant":
                            options["value"] = -1.25
                        check(
                            Spatial(mode, view, **options),
                            torch.randn(source_shape, dtype=dtype),
                        )
                        count += 2
print(f"General padding: {count} cases passed", flush=True)

for pads in ((-5, 6), (6, -5), (-3, -2)):
    model = Spatial("constant", pad=pads, value=1.0)
    x = torch.ones(2, 4)
    try:
        model(x)
    except RuntimeError:
        pass
    else:
        raise AssertionError("PyTorch accepted invalid cropping")
    try:
        ep = torch.export.export(model, (x,), strict=True)
        compiler = DynamoCompiler(
            primary_registry=tosa.ops_registry, enable_external_calls=False
        )
        compiler._compile_fx(ep.graph_module, [x])
        compiler.dynamo_run()
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError("Buddy accepted invalid cropping")
print("Constant padding: 3 invalid-cropping checks passed", flush=True)
