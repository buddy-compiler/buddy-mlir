"""Deterministic CPU cases; importing this module does not require PyTorch."""

from __future__ import annotations

PROFILES = {
    "small-f32": {"rows": 4, "width": 8, "dtype": "float32"},
    "rect-f32": {"rows": 3, "width": 5, "dtype": "float32"},
    "small-f64": {"rows": 4, "width": 8, "dtype": "float64"},
}
OPERATORS = (
    "aten::add.Tensor",
    "aten::mul.Tensor",
    "aten::mm.default",
    "aten::bmm.default",
    "aten::addmm.default",
    "aten::silu.default",
    "aten::gelu.default",
    "aten::_softmax.default",
    "aten::softmax.int",
    "aten::topk.default",
    "aten::gather.default",
    "aten::scatter_add.default",
    "aten::index_select.default",
    "aten::index_add.default",
    "aten::index_copy.default",
    "aten::one_hot.default",
    "aten::bincount.default",
    "aten::sort.default",
    "aten::argsort.default",
    "aten::view.default",
    "aten::reshape.default",
    "aten::native_layer_norm.default",
    "aten::sum.dim_IntList",
    "prims::convert_element_type.default",
    "aten::baddbmm.default",
    "aten::div.Tensor",
    "aten::sub.Tensor",
    "aten::eq.Tensor",
    "aten::maximum.default",
    "aten::minimum.default",
    "aten::neg.default",
    "aten::rsqrt.default",
    "aten::sqrt.default",
    "aten::exp.default",
    "aten::relu.default",
    "aten::sigmoid.default",
    "aten::tanh.default",
    "aten::clone.default",
    "aten::mean.dim",
    "aten::amax.default",
    "aten::argmax.default",
    "aten::argmin.default",
    "aten::cumsum.default",
    "aten::cumprod.default",
    "aten::pow.Tensor_Scalar",
    "aten::clamp.default",
    "aten::eq.Scalar",
    "aten::ne.Scalar",
    "aten::gt.Scalar",
    "aten::lt.Scalar",
    "aten::le.Scalar",
    "aten::ge.Scalar",
    "aten::where.self",
    "aten::masked_fill.Scalar",
    "aten::transpose.int",
    "aten::permute.default",
    "aten::unsqueeze.default",
    "aten::squeeze.dim",
    "aten::expand.default",
    "aten::repeat.default",
    "aten::slice.Tensor",
    "aten::select.int",
    "aten::cat.default",
    "aten::stack.default",
    "aten::embedding.default",
    "aten::_to_copy.default",
    "aten::arange.start",
    "aten::arange.start_step",
    "aten::ones.default",
    "aten::zeros.default",
    "aten::full.default",
    "aten::scalar_tensor.default",
    "aten::slice_scatter.default",
    "aten::split.Tensor",
    "aten::split_with_sizes.default",
    "aten::unbind.int",
    "aten::copy.default",
    "aten::lift_fresh_copy.default",
    "aten::repeat_interleave.self_int",
    "aten::constant_pad_nd.default",
    "aten::convolution.default",
    "aten::avg_pool2d.default",
    "aten::_adaptive_avg_pool2d.default",
    "aten::max_pool2d_with_indices.default",
    "aten::upsample_bilinear2d.vec",
    "aten::upsample_nearest2d.vec",
    "aten::grid_sampler_2d.default",
    "aten::reflection_pad2d.default",
    "aten::pixel_shuffle.default",
    "aten::pixel_unshuffle.default",
    "aten::_scaled_dot_product_flash_attention_for_cpu.default",
    "aten::index.Tensor",
    "aten::contiguous.default",
    "aten::_unsafe_index.Tensor",
    "aten::index_put.default",
    "aten::scatter.src",
    "aten::scatter.value",
    "aten::scatter.reduce",
    "aten::scatter.value_reduce",
    "aten::scatter_reduce.two",
    "aten::masked_scatter.default",
    "aten::masked_select.default",
    "aten::nonzero.default",
    "aten::nonzero_static.default",
    "aten::repeat_interleave.Tensor",
    "aten::pad.default",
)
WORKLOADS = ("transformer_block", "moe_block")


def resolve_operator(name):
    import torch

    namespace, key = name.split("::")
    op, overload = key.rsplit(".", 1)
    return getattr(getattr(getattr(torch.ops, namespace), op), overload)


def build_case(name, profile):
    import torch

    config = PROFILES[profile]
    n, d = config["rows"], config["width"]
    dtype = getattr(torch, config["dtype"])
    torch.manual_seed(0)

    def rand(*shape):
        return torch.randn(*shape, dtype=dtype)

    x = rand(n, d)
    if name in WORKLOADS:
        return build_workload(name, x, rand)
    op = resolve_operator(name)
    fn = op
    if name == "aten::_scaled_dot_product_flash_attention_for_cpu.default":

        def fn(q, k, v):
            return op(q, k, v, 0.0, True)

        args = (rand(1, 2, n, 4), rand(1, 2, d, 4), rand(1, 2, d, 4))
    elif name in ("aten::index.Tensor", "aten::_unsafe_index.Tensor"):
        fn, args = lambda a, i: op(a, [i]), (x, torch.tensor([0, n - 1, 0]))
    elif name == "aten::contiguous.default":
        fn, args = lambda a: (op(a), op(a[:, 1::2])), (x[:, ::2],)
    elif name == "aten::index_put.default":
        fn, args = (
            lambda a, i, s: op(a, [i], s, True),
            (x, torch.tensor([0, 0]), rand(2, d)),
        )
    elif name in (
        "aten::scatter.src",
        "aten::scatter.value",
        "aten::scatter.reduce",
        "aten::scatter.value_reduce",
        "aten::scatter_reduce.two",
    ):
        index = torch.tensor([0, 1, d - 1]).repeat(n, 1)
        source = rand(n, 3)
        if name == "aten::scatter.src":
            fn, args = lambda a, i, s: op(a, 1, i, s), (x, index, source)
        elif name == "aten::scatter.value":
            fn, args = lambda a, i: op(a, 1, i, -2), (x, index)
        elif name == "aten::scatter.reduce":
            index[:, 1] = 0
            fn, args = (
                lambda a, i, s: op(a, 1, i, s, reduce="add"),
                (x, index, source),
            )
        elif name == "aten::scatter.value_reduce":
            index[:, 1] = 0
            fn, args = (
                lambda a, i: op(a, 1, i, 2, reduce="multiply"),
                (x, index),
            )
        else:
            index[:, 1] = 0

            def fn(a, i, s):
                return op(a, 1, i, s, "sum", include_self=False)

            args = (x, index, source)
    elif name == "aten::masked_scatter.default":
        args = (x, x > 0, rand(n * d))
    elif name == "aten::masked_select.default":
        args = (x, x > 0)
    elif name == "aten::nonzero.default":
        args = (x * (x > 0),)
    elif name == "aten::nonzero_static.default":
        fn, args = lambda a: op(a, size=d, fill_value=-1), (x * (x > 0),)
    elif name == "aten::repeat_interleave.Tensor":
        fn, args = lambda a: op(a, output_size=3), (torch.tensor([1, 0, 2]),)
    elif name == "aten::pad.default":
        fn, args = lambda a: op(a, [1, 2], "reflect"), (x,)
    elif name == "aten::embedding.default":
        args = (x, torch.tensor([0, n - 1, 1, 0]))
    elif name == "aten::_to_copy.default":
        target_dtype = (
            torch.float64 if dtype == torch.float32 else torch.float32
        )
        fn, args = lambda a: op(a, dtype=target_dtype), (x,)
    elif name in ("aten::arange.start", "aten::arange.start_step"):
        step = 2 if name.endswith("start_step") else None

        def fn():
            return op(
                2,
                2 + 2 * d,
                *(() if step is None else (step,)),
                dtype=dtype,
                device="cpu",
            )

        args = ()
    elif name in ("aten::ones.default", "aten::zeros.default"):
        fn, args = lambda: op([n, d], dtype=dtype, device="cpu"), ()
    elif name == "aten::full.default":
        fn, args = lambda: op([n, d], -2.5, dtype=dtype, device="cpu"), ()
    elif name == "aten::scalar_tensor.default":
        fn, args = lambda: op(2.5, dtype=dtype, device="cpu"), ()
    elif name == "aten::slice_scatter.default":
        fn, args = (
            lambda a, b: op(a, b, 1, 1, d, 2),
            (x, rand(n, len(range(1, d, 2)))),
        )
    elif name == "aten::split.Tensor":
        fn, args = lambda a: op(a, 2, 1), (x,)
    elif name == "aten::split_with_sizes.default":
        fn, args = lambda a: op(a, [1, 2, d - 3], 1), (x,)
    elif name == "aten::unbind.int":
        fn, args = lambda a: op(a, 0), (x,)
    elif name == "aten::copy.default":
        args = (x, rand(n, d))
    elif name == "aten::repeat_interleave.self_int":
        fn, args = lambda a: op(a, 2, 1), (x,)
    elif name == "aten::constant_pad_nd.default":
        fn, args = lambda a: op(a, [1, 2], 0.5), (x,)
    elif name == "aten::convolution.default":

        def fn(a, w, b):
            return op(a, w, b, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        args = (rand(1, 2, n + 1, d), rand(3, 2, 3, 3), rand(3))
    elif name == "aten::avg_pool2d.default":
        fn, args = (
            lambda a: op(a, [2, 2], [2, 2], [0, 0], False, True),
            (rand(1, 2, n + 1, d),),
        )
    elif name == "aten::_adaptive_avg_pool2d.default":
        fn, args = lambda a: op(a, [2, 2]), (rand(1, 2, n + 1, d),)
    elif name == "aten::max_pool2d_with_indices.default":
        fn, args = (
            lambda a: op(a, [2, 2], [2, 2], [0, 0], [1, 1], False),
            (rand(1, 2, n + 1, d),),
        )
    elif name == "aten::upsample_bilinear2d.vec":
        fn, args = (
            lambda a: op(a, [n + 3, d + 2], False, None),
            (rand(1, 2, n + 1, d),),
        )
    elif name == "aten::upsample_nearest2d.vec":
        fn, args = (
            lambda a: op(a, [n + 3, d + 2], None),
            (rand(1, 2, n + 1, d),),
        )
    elif name == "aten::grid_sampler_2d.default":
        fn, args = (
            lambda a, grid: op(a, grid, 0, 0, False),
            (rand(1, 2, n + 1, d), torch.rand(1, n, 3, 2, dtype=dtype) * 2 - 1),
        )
    elif name == "aten::reflection_pad2d.default":
        fn, args = lambda a: op(a, [1, 2, 1, 1]), (rand(1, 2, n + 1, d),)
    elif name == "aten::pixel_shuffle.default":
        fn, args = lambda a: op(a, 2), (rand(1, 8, n, d),)
    elif name == "aten::pixel_unshuffle.default":
        fn, args = lambda a: op(a, 2), (rand(1, 2, 2 * n, 2 * d),)
    elif name in ("aten::add.Tensor", "aten::mul.Tensor"):
        args = (x, rand(n, d))
    elif name == "aten::mm.default":
        args = (x, rand(d, n + 1))
    elif name == "aten::bmm.default":
        args = (rand(2, n, d), rand(2, d, n + 1))
    elif name == "aten::baddbmm.default":
        args = (rand(2, n, n + 1), rand(2, n, d), rand(2, d, n + 1))
    elif name in (
        "aten::div.Tensor",
        "aten::sub.Tensor",
        "aten::eq.Tensor",
        "aten::maximum.default",
        "aten::minimum.default",
    ):
        args = (x, rand(n, d).abs() + 0.5)
    elif name in ("aten::rsqrt.default", "aten::sqrt.default"):
        args = (x.abs() + 0.5,)
    elif name == "aten::pow.Tensor_Scalar":
        fn, args = lambda a: op(a, 2), (x,)
    elif name == "aten::clamp.default":
        fn, args = lambda a: op(a, -0.5, 0.5), (x,)
    elif name in (
        "aten::eq.Scalar",
        "aten::ne.Scalar",
        "aten::gt.Scalar",
        "aten::lt.Scalar",
        "aten::le.Scalar",
        "aten::ge.Scalar",
    ):
        x[0, 0] = 0
        fn, args = lambda a: op(a, 0), (x,)
    elif name == "aten::where.self":
        args = (x > 0, x, rand(n, d))
    elif name == "aten::masked_fill.Scalar":
        fn, args = lambda a, mask: op(a, mask, -2), (x, x > 0)
    elif name in ("aten::mean.dim", "aten::amax.default"):
        fn, args = lambda a: op(a, [-1], False), (x,)
    elif name in ("aten::argmax.default", "aten::argmin.default"):
        fn, args = lambda a: op(a, -1, False), (x,)
    elif name in ("aten::cumsum.default", "aten::cumprod.default"):
        fn, args = lambda a: op(a, -1), (x,)
    elif name == "aten::transpose.int":
        fn, args = lambda a: op(a, 0, 1), (x,)
    elif name == "aten::permute.default":
        fn, args = lambda a: op(a, [1, 0]), (x,)
    elif name == "aten::unsqueeze.default":
        fn, args = lambda a: op(a, 1), (x,)
    elif name == "aten::squeeze.dim":
        fn, args = lambda a: op(a, 1), (x.unsqueeze(1),)
    elif name == "aten::expand.default":
        fn, args = lambda a: op(a, [n, d]), (rand(n, 1),)
    elif name == "aten::repeat.default":
        fn, args = lambda a: op(a, [1, 2]), (x,)
    elif name == "aten::slice.Tensor":
        fn, args = lambda a: op(a, 1, 1, d, 2), (x,)
    elif name == "aten::select.int":
        fn, args = lambda a: op(a, 1, d - 1), (x,)
    elif name in ("aten::cat.default", "aten::stack.default"):
        fn, args = lambda a, b: op([a, b], 1), (x, rand(n, d))
    elif name == "aten::addmm.default":
        args = (rand(n, n + 1), x, rand(d, n + 1))
    elif name == "aten::_softmax.default":
        fn, args = lambda a: op(a, -1, False), (x,)
    elif name == "aten::softmax.int":
        target_dtype = (
            torch.float64 if dtype == torch.float32 else torch.float32
        )
        fn, args = lambda a: (op(a, -1), op(a.T, 0, dtype=target_dtype)), (x,)
    elif name == "aten::argsort.default":
        fn, args = lambda a: (op(a), op(a.T, 0, True)), (x,)
    elif name == "aten::topk.default":
        fn, args = lambda a: op(a, 2, -1, True, True), (x,)
    elif name == "aten::gather.default":

        def fn(a, b):
            return op(a, 1, b)

        args = (x, torch.randint(d, (n, 3)))
    elif name == "aten::scatter_add.default":

        def fn(a, b, c):
            return op(a, 1, b, c)

        # Repeated indices exercise accumulation rather than only assignment.
        args = (x, torch.zeros(n, 3, dtype=torch.int64), rand(n, 3))
    elif name in (
        "aten::index_select.default",
        "aten::index_add.default",
        "aten::index_copy.default",
    ):
        indices = torch.tensor([0, n - 1], dtype=torch.int64)
        if name == "aten::index_select.default":
            fn, args = lambda a, b: op(a, 0, b), (x, indices)
        else:
            fn, args = lambda a, b, c: op(a, 0, b, c), (x, indices, rand(2, d))
    elif name == "aten::one_hot.default":
        fn, args = lambda a: op(a, 4), (torch.tensor([0, 3, 1, 0]),)
    elif name == "aten::bincount.default":
        fn, args = lambda a: op(a, minlength=4), (torch.tensor([0, 3, 1, 0]),)
    elif name == "aten::view.default":
        fn, args = lambda a: op(a, [d, n]), (x,)
    elif name == "aten::reshape.default":
        fn, args = lambda a: (op(a, [d, n]), op(a[:, 1::2], [-1])), (x,)
    elif name == "aten::native_layer_norm.default":

        def fn(a, w, b):
            return op(a, [d], w, b, 1e-5)

        args = (x, rand(d), rand(d))
    elif name == "aten::sum.dim_IntList":
        fn, args = lambda a: op(a, [-1], False), (x,)
    elif name == "prims::convert_element_type.default":
        fn, args = lambda a: op(a, torch.float64), (x,)
    else:
        args = (x,)
    return as_module(fn), args


def as_module(fn):
    import torch

    class CaseModule(torch.nn.Module):
        def forward(self, *args):
            return fn(*args)

    return CaseModule().eval()


def build_workload(name, x, rand):
    import torch
    import torch.nn.functional as functional

    d = x.shape[-1]
    if name == "transformer_block":

        def forward(a, q, k, v, out, up, down, norm_w, norm_b):
            scores = (a @ q) @ (a @ k).transpose(-1, -2) / (d**0.5)
            attention = torch.softmax(scores, dim=-1) @ (a @ v)
            residual = a + attention @ out
            normalized = functional.layer_norm(residual, [d], norm_w, norm_b)
            return residual + functional.gelu(normalized @ up) @ down

        args = (
            x,
            rand(d, d),
            rand(d, d),
            rand(d, d),
            rand(d, d),
            rand(d, 2 * d),
            rand(2 * d, d),
            rand(d),
            rand(d),
        )
    else:

        def forward(a, gate, up, down):
            weights, experts = torch.topk(torch.softmax(a @ gate, -1), 2, -1)
            weights = weights / weights.sum(-1, keepdim=True)
            token_ids = (
                torch.arange(a.shape[0]).unsqueeze(1).expand(-1, 2).reshape(-1)
            )
            dispatched = a.index_select(0, token_ids)
            selected_up = up.index_select(0, experts.reshape(-1))
            selected_down = down.index_select(0, experts.reshape(-1))
            hidden = functional.silu(
                torch.bmm(dispatched.unsqueeze(1), selected_up)
            )
            expert_out = torch.bmm(hidden, selected_down).squeeze(1)
            weighted = expert_out * weights.reshape(-1, 1)
            return torch.zeros_like(a).scatter_add(
                0, token_ids[:, None].expand(-1, d), weighted
            )

        args = (x, rand(d, 4), rand(4, d, 2 * d), rand(4, 2 * d, d))
    return as_module(forward), args
