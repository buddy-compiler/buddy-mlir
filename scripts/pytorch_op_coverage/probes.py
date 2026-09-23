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
    if name in ("aten::add.Tensor", "aten::mul.Tensor"):
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
        fn, args = lambda a: op(a, -1), (x,)
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
    elif name in ("aten::view.default", "aten::reshape.default"):
        fn, args = lambda a: op(a, [d, n]), (x,)
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
