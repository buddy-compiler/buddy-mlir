"""Opt-in export preserving checked one-hot and empty pixel-unshuffle semantics.

The custom operations retain eager behavior through AOT functionalization.
Use ``export`` instead of ``torch.export.export`` when these operations occur.
Other operations follow the normal PyTorch export path.
"""

import torch
from torch.overrides import TorchFunctionMode


@torch.library.custom_op("buddy_export::checked_one_hot", mutates_args=())
def one_hot(x: torch.Tensor, classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(x, classes)


@one_hot.register_fake
def _one_hot_meta(x, classes):
    torch._check(x.dtype == torch.int64, lambda: "one_hot requires int64 input")
    torch._check(
        classes == -1 or classes > 0,
        lambda: "one_hot requires positive or inferred class count",
    )
    if classes == -1:
        torch._check(
            x.numel() > 0, lambda: "Cannot infer classes from empty input"
        )
        classes = torch.library.get_ctx().new_dynamic_size(min=1)
    return x.new_empty((*x.shape, classes), dtype=torch.int64)


@torch.library.custom_op(
    "buddy_export::checked_pixel_unshuffle", mutates_args=()
)
def pixel_unshuffle(x: torch.Tensor, factor: int) -> torch.Tensor:
    return torch.nn.functional.pixel_unshuffle(x, factor).clone()


@pixel_unshuffle.register_fake
def _pixel_meta(x, factor):
    torch._check(x.dim() >= 3, lambda: "pixel_unshuffle requires rank >= 3")
    torch._check(factor > 0, lambda: "pixel_unshuffle requires positive factor")
    torch._check(
        x.shape[-2] % factor == 0,
        lambda: "pixel_unshuffle height must divide by factor",
    )
    torch._check(
        x.shape[-1] % factor == 0,
        lambda: "pixel_unshuffle width must divide by factor",
    )
    if x.numel() == 0:
        return x.new_empty(x.shape)
    return x.new_empty(
        (
            *x.shape[:-3],
            x.shape[-3] * factor * factor,
            x.shape[-2] // factor,
            x.shape[-1] // factor,
        )
    )


class _ExportMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in (
            torch.nn.functional.one_hot,
            torch.ops.aten.one_hot.default,
            torch.ops.aten.one_hot,
        ):
            value = (
                args[0] if args else kwargs.get("tensor", kwargs.get("self"))
            )
            classes = (
                args[1] if len(args) > 1 else kwargs.get("num_classes", -1)
            )
            return one_hot(value, classes)
        if func in (
            torch.nn.functional.pixel_unshuffle,
            torch.ops.aten.pixel_unshuffle.default,
            torch.ops.aten.pixel_unshuffle,
        ):
            value = args[0] if args else kwargs.get("input", kwargs.get("self"))
            factor = args[1] if len(args) > 1 else kwargs["downscale_factor"]
            return pixel_unshuffle(value, factor)
        return func(*args, **kwargs)


ADAPTED_OPERATORS = {
    "aten::one_hot.default": "buddy_export::checked_one_hot.default",
    "aten::pixel_unshuffle.default": "buddy_export::checked_pixel_unshuffle.default",
}


def export(module, args, **options):
    """Export with checked operators; accepts torch.export.export options."""
    with _ExportMode():
        return torch.export.export(module, args, **options)
