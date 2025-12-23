# RUN: %PYTHON %s 2>&1 | FileCheck %s
from aten_op_batch_runner import run_aten_op_batch
import torch

CUSTOM_TEMPLATES = {}


def _skip(reason: str):
    def fn():
        raise RuntimeError(reason)

    return fn


def _template_rot90_default():
    x = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    return [x], {}


def _template_rot90_out():
    args, _ = _template_rot90_default()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _rrelu_base():
    x = torch.tensor([[-1.0, 2.0], [0.5, -0.3]], dtype=torch.float32)
    noise = torch.full_like(x, 0.2)
    return x, noise, 0.1, 0.3


def _template_rrelu_with_noise():
    x, noise, lower, upper = _rrelu_base()
    return [x, noise.clone(), lower, upper, True, None], {}


def _template_rrelu_with_noise_out():
    args, _ = _template_rrelu_with_noise()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_rrelu_with_noise_inplace():
    x, noise, lower, upper = _rrelu_base()
    return [x, noise.clone(), lower, upper, True, None], {}


def _template_rrelu_with_noise_backward():
    x, noise, lower, upper = _rrelu_base()
    grad_output = torch.ones_like(x)
    return [grad_output, x, noise, lower, upper, True, False], {}


def _template_rrelu_with_noise_backward_out():
    args, _ = _template_rrelu_with_noise_backward()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_rrelu_with_noise_functional():
    x, noise, lower, upper = _rrelu_base()
    return [x, noise.clone(), lower, upper, True, None], {}


def _template_scalar_tensor():
    return [1.5], {"dtype": torch.float32}


def _template_scalar_tensor_out():
    return [1.5], {"out": torch.empty((), dtype=torch.float32)}


def _scatter_base(named: bool = False):
    base = torch.zeros((2, 3), dtype=torch.float32)
    index = torch.tensor([[0, 1, 1], [0, 0, 1]], dtype=torch.int64)
    src = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    if named:
        base = base.refine_names("N", "C")
        index = index.refine_names("N", "C")
        src = src.refine_names("N", "C")
    return base, index, src


def _template_scatter_value():
    base, index, _ = _scatter_base()
    return [base, 1, index, 2.5], {}


def _template_scatter_src():
    base, index, src = _scatter_base()
    return [base, 1, index, src], {}


def _template_scatter_reduce():
    base, index, src = _scatter_base()
    return [base, 1, index, src], {"reduce": "add"}


def _template_scatter_value_reduce():
    base, index, _ = _scatter_base()
    return [base, 1, index, 1.5], {"reduce": "add"}


def _template_scatter_value_out():
    args, _ = _template_scatter_value()
    return args, {"out": torch.empty_like(args[0])}


def _template_scatter_src_out():
    args, _ = _template_scatter_src()
    return args, {"out": torch.empty_like(args[0])}


def _template_scatter_reduce_out():
    args, _ = _template_scatter_reduce()
    return args[0:4], {"reduce": "add", "out": torch.empty_like(args[0])}


def _template_scatter_value_reduce_out():
    args, _ = _template_scatter_value_reduce()
    return args[0:4], {"reduce": "add", "out": torch.empty_like(args[0])}


def _template_scatter_dimname_src():
    base, index, src = _scatter_base(named=True)
    return [base, "names::C", index, src], {}


def _template_scatter_dimname_value():
    base, index, _ = _scatter_base(named=True)
    return [base, "names::C", index, 1.0], {}


def _template_scatter__src():
    base, index, src = _scatter_base()
    return [base, 1, index, src], {}


def _template_scatter__value():
    base, index, _ = _scatter_base()
    return [base, 1, index, 3.0], {}


def _template_scatter__reduce():
    base, index, src = _scatter_base()
    return [base, 1, index, src], {"reduce": "add"}


def _template_scatter__value_reduce():
    base, index, _ = _scatter_base()
    return [base, 1, index, 2.0], {"reduce": "add"}


def _template_scatter_add():
    base, index, src = _scatter_base()
    return [base, 1, index, src], {}


def _template_scatter_add_out():
    args, _ = _template_scatter_add()
    return args, {"out": torch.empty_like(args[0])}


def _template_scatter_add_dimname():
    base, index, src = _scatter_base(named=True)
    return [base, "names::C", index, src], {}


def _template_scatter_reduce_two():
    base, index, src = _scatter_base()
    return [base, 1, index, src, "sum"], {}


def _template_scatter_reduce_two_out():
    args, _ = _template_scatter_reduce_two()
    return args, {"out": torch.empty_like(args[0])}


def _template_searchsorted_tensor():
    sorted_seq = torch.tensor([1.0, 3.0, 5.0, 7.0])
    values = torch.tensor([0.0, 4.0, 6.0])
    return [sorted_seq, values], {}


def _template_searchsorted_tensor_out():
    sorted_seq = torch.tensor([1.0, 3.0, 5.0, 7.0])
    values = torch.tensor([0.0, 4.0, 6.0])
    out = torch.empty(values.shape, dtype=torch.int64)
    return [sorted_seq, values], {"out": out}


def _template_searchsorted_scalar():
    sorted_seq = torch.tensor([1.0, 3.0, 5.0, 7.0])
    return [sorted_seq, 4.0], {}


def _template_searchsorted_scalar_out():
    sorted_seq = torch.tensor([1.0, 3.0, 5.0, 7.0])
    out = torch.empty((), dtype=torch.int64)
    return [sorted_seq, 4.0], {"out": out}


def _template_segment_reduce():
    data = torch.tensor([1.0, 2.0, 3.0])
    lengths = torch.tensor([2, 1], dtype=torch.int64)
    return [data, "sum"], {"lengths": lengths}


def _template_segment_reduce_out():
    args, kwargs = _template_segment_reduce()
    out = torch.empty((2,), dtype=args[0].dtype)
    kwargs = dict(kwargs)
    kwargs["out"] = out
    return args, kwargs


def _template_select_int():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    return [x, 1, 1], {}


def _template_select_dimname():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).refine_names("N", "C")
    return [x, "C", 1], {}


def _template_select_t():
    return [[0, 1, 2, 3], 2], {}


def _template_select_backward():
    input_sizes = [2, 3]
    grad_output = torch.randn(2, dtype=torch.float32)
    return [grad_output, input_sizes, 1, 1], {}


def _template_select_backward_out():
    input_sizes = [2, 3]
    grad_output = torch.randn(2, dtype=torch.float32)
    grad_input = torch.empty(input_sizes, dtype=grad_output.dtype)
    return [grad_output, input_sizes, 1, 1], {"out": grad_input}


def _template_select_scatter():
    base = torch.zeros((2, 3), dtype=torch.float32)
    src = torch.tensor([7.0, 8.0], dtype=torch.float32)
    return [base, src, 1, 1], {}


def _template_select_scatter_out():
    args, _ = _template_select_scatter()
    return args, {"out": torch.empty_like(args[0])}


def _template_slice_t():
    return [[0, 1, 2, 3], 1, 3, 1], {}


def _template_slice_str():
    return ["abcdef", 1, 4, 1], {}


def _template_slice_tensor():
    x = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    return [x, 1, 1, 4, 1], {}


def _template_slice_backward():
    grad_output = torch.randn(2, 3, dtype=torch.float32)
    input_sizes = [2, 5]
    return [grad_output, input_sizes, 1, 1, 4, 1], {}


def _template_slice_backward_out():
    grad_output = torch.randn(2, 3, dtype=torch.float32)
    input_sizes = [2, 5]
    out = torch.empty(input_sizes, dtype=grad_output.dtype)
    return [grad_output, input_sizes, 1, 1, 4, 1], {"out": out}


def _template_slice_scatter():
    base = torch.zeros((2, 5), dtype=torch.float32)
    src = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    return [base, src, 1, 1, 4, 1], {}


def _template_slice_scatter_out():
    args, _ = _template_slice_scatter()
    return args, {"out": torch.empty_like(args[0])}


def _loss_pair():
    inp = torch.tensor([0.2, -0.3], dtype=torch.float32)
    target = torch.tensor([0.1, 0.0], dtype=torch.float32)
    return inp, target


def _template_smooth_l1_loss_backward_grad_input():
    inp, tgt = _loss_pair()
    grad_output = torch.ones_like(inp)
    grad_input = torch.empty_like(inp)
    return [grad_output, inp, tgt, 1, 1.0], {"grad_input": grad_input}


def _template_soft_margin_loss_backward_grad_input():
    inp, tgt = _loss_pair()
    grad_output = torch.ones_like(inp)
    grad_input = torch.empty_like(inp)
    return [grad_output, inp, tgt, 1], {"grad_input": grad_input}


def _template_sigmoid_backward_grad_input():
    output = torch.sigmoid(torch.tensor([0.5, -0.5], dtype=torch.float32))
    grad_output = torch.ones_like(output)
    grad_input = torch.empty_like(output)
    return [grad_output, output], {"grad_input": grad_input}


def _template_silu_backward_grad_input():
    x = torch.tensor([0.5, -1.0], dtype=torch.float32)
    grad_output = torch.ones_like(x)
    grad_input = torch.empty_like(x)
    return [grad_output, x], {"grad_input": grad_input}


def _template_softplus_backward_grad_input():
    x = torch.tensor([0.5, -0.5], dtype=torch.float32)
    grad_output = torch.ones_like(x)
    grad_input = torch.empty_like(x)
    return [grad_output, x, 1.0, 2.0], {"grad_input": grad_input}


def _sort_tensor_base():
    return torch.tensor(
        [[3.0, 1.0, 2.0], [4.0, 0.0, -1.0]], dtype=torch.float32
    )


def _template_sort_default():
    x = _sort_tensor_base()
    return [x, -1, False], {}


def _template_sort_stable():
    x = _sort_tensor_base()
    return [x], {"stable": True, "dim": -1, "descending": False}


def _template_sort_values():
    x = _sort_tensor_base()
    values = torch.empty_like(x)
    indices = torch.empty_like(x, dtype=torch.int64)
    return [x, -1, False], {"values": values, "indices": indices}


def _template_sort_values_stable():
    x = _sort_tensor_base()
    values = torch.empty_like(x)
    indices = torch.empty_like(x, dtype=torch.int64)
    return [x], {
        "stable": True,
        "dim": -1,
        "descending": False,
        "values": values,
        "indices": indices,
    }


def _sort_named_base():
    return torch.tensor(
        [[3.0, 1.0], [0.0, -1.0]], dtype=torch.float32
    ).refine_names("N", "C")


def _template_sort_dimname():
    x = _sort_named_base()
    return [x, "C", False], {}


def _template_sort_dimname_values():
    x = _sort_named_base()
    values = torch.empty_like(x)
    indices = torch.empty_like(x, dtype=torch.int64)
    return [x, "C", False], {"values": values, "indices": indices}


def _template_sort_dimname_stable():
    x = _sort_named_base()
    return [x], {"stable": True, "dim": "C", "descending": False}


def _template_sort_dimname_values_stable():
    x = _sort_named_base()
    values = torch.empty_like(x)
    indices = torch.empty_like(x, dtype=torch.int64)
    return [x], {
        "stable": True,
        "dim": "C",
        "descending": False,
        "values": values,
        "indices": indices,
    }


def _template_sort_str():
    return [["c", "a", "b"], False], {}


def _template_sort_any():
    return [[3, 1, 2], False], {}


def _poly_base():
    x = torch.tensor([0.2, 0.5], dtype=torch.float32)
    n_tensor = torch.tensor([2, 3], dtype=torch.int32)
    return x, n_tensor


def _poly_out_for(*tensors):
    shapes = [t.shape for t in tensors if isinstance(t, torch.Tensor)]
    shape = torch.broadcast_shapes(*shapes) if shapes else ()
    return torch.empty(shape, dtype=torch.float32)


def _template_poly_default():
    x, n = _poly_base()
    return [x, n], {}


def _template_poly_out():
    x, n = _poly_base()
    return [x, n], {"out": _poly_out_for(x, n)}


def _template_poly_x_scalar():
    _, n = _poly_base()
    return [0.3, n], {}


def _template_poly_x_scalar_out():
    _, n = _poly_base()
    return [0.3, n], {"out": _poly_out_for(n)}


def _template_poly_n_scalar():
    x, _ = _poly_base()
    return [x, 2], {}


def _template_poly_n_scalar_out():
    x, _ = _poly_base()
    return [x, 2], {"out": _poly_out_for(x)}


def _template_signbit_out():
    x = torch.tensor([1.0, -2.0], dtype=torch.float32)
    out = torch.empty_like(x, dtype=torch.bool)
    return [x], {"out": out}


# Edit the OPS list to add/remove ops in this batch (format: "op.overload").
OPS = [
    "rot90.default",
    "rot90.out",
    "round.default",
    "round.decimals",
    "round.out",
    "round.decimals_out",
    "round.int",
    "round.float",
    "round.Scalar",
    "round_.default",
    "round_.decimals",
    "rrelu_with_noise.default",
    "rrelu_with_noise.out",
    "rrelu_with_noise_.default",
    "rrelu_with_noise_backward.default",
    "rrelu_with_noise_backward.out",
    "rrelu_with_noise_functional.default",
    "rsqrt.default",
    "rsqrt.out",
    "rsqrt_.default",
    "rsub.Tensor",
    "rsub.Scalar",
    "rsub.Tensor_out",
    "rsub.Scalar_out",
    "scalar_tensor.default",
    "scalar_tensor.out",
    "scatter.value",
    "scatter.src",
    "scatter.reduce",
    "scatter.value_reduce",
    "scatter.src_out",
    "scatter.value_out",
    "scatter.reduce_out",
    "scatter.value_reduce_out",
    "scatter.dimname_src",
    "scatter.dimname_value",
    "scatter_.src",
    "scatter_.value",
    "scatter_.reduce",
    "scatter_.value_reduce",
    "scatter_add.default",
    "scatter_add.out",
    "scatter_add.dimname",
    "scatter_add_.default",
    "scatter_reduce.two",
    "scatter_reduce.two_out",
    "scatter_reduce_.two",
    "searchsorted.Tensor",
    "searchsorted.Tensor_out",
    "searchsorted.Scalar",
    "searchsorted.Scalar_out",
    "segment_reduce.default",
    "segment_reduce.out",
    "select.Dimname",
    "select.int",
    "select.t",
    "select_backward.default",
    "select_backward.out",
    "select_scatter.default",
    "select_scatter.out",
    "selu.default",
    "selu_.default",
    "set_.source_Storage_storage_offset",
    "set_.source_Tensor",
    "set_.default",
    "set_.source_Storage",
    "set_.source_Tensor_storage_offset",
    "sgn.default",
    "sgn.out",
    "sgn_.default",
    "sigmoid.default",
    "sigmoid.out",
    "sigmoid_.default",
    "sigmoid_backward.default",
    "sigmoid_backward.grad_input",
    "sign.default",
    "sign.out",
    "sign_.default",
    "signbit.default",
    "signbit.out",
    "silu.default",
    "silu.out",
    "silu_.default",
    "silu_backward.default",
    "silu_backward.grad_input",
    "sin.default",
    "sin.out",
    "sin.int",
    "sin.float",
    "sin.complex",
    "sin.Scalar",
    "sin_.default",
    "sinc.default",
    "sinc.out",
    "sinc_.default",
    "sinh.default",
    "sinh.out",
    "sinh.int",
    "sinh.float",
    "sinh.complex",
    "sinh.Scalar",
    "sinh_.default",
    "size.int",
    "size.Dimname",
    "size.default",
    "slice.Tensor",
    "slice.t",
    "slice.str",
    "slice_backward.default",
    "slice_backward.out",
    "slice_scatter.default",
    "slice_scatter.out",
    "smooth_l1_loss.default",
    "smooth_l1_loss.out",
    "smooth_l1_loss_backward.grad_input",
    "smooth_l1_loss_backward.default",
    "soft_margin_loss.default",
    "soft_margin_loss.out",
    "soft_margin_loss_backward.default",
    "soft_margin_loss_backward.grad_input",
    "softplus.default",
    "softplus.out",
    "softplus_backward.default",
    "softplus_backward.grad_input",
    "softshrink.default",
    "softshrink.out",
    "sort.default",
    "sort.stable",
    "sort.values_stable",
    "sort.values",
    "sort.dimname",
    "sort.dimname_values",
    "sort.dimname_stable",
    "sort.dimname_values_stable",
    "sort.int",
    "sort.float",
    "sort.Tensor",
    "sort.bool",
    "sort.str",
    "sort.any",
    "sparse_dim.default",
    "special_airy_ai.default",
    "special_airy_ai.out",
    "special_bessel_j0.default",
    "special_bessel_j0.out",
    "special_bessel_j1.default",
    "special_bessel_j1.out",
    "special_bessel_y0.default",
    "special_bessel_y0.out",
    "special_bessel_y1.default",
    "special_bessel_y1.out",
    "special_chebyshev_polynomial_t.default",
    "special_chebyshev_polynomial_t.out",
    "special_chebyshev_polynomial_t.x_scalar",
    "special_chebyshev_polynomial_t.x_scalar_out",
    "special_chebyshev_polynomial_t.n_scalar",
    "special_chebyshev_polynomial_t.n_scalar_out",
    "special_chebyshev_polynomial_u.default",
    "special_chebyshev_polynomial_u.out",
    "special_chebyshev_polynomial_u.x_scalar",
    "special_chebyshev_polynomial_u.x_scalar_out",
    "special_chebyshev_polynomial_u.n_scalar",
    "special_chebyshev_polynomial_u.n_scalar_out",
    "special_chebyshev_polynomial_v.default",
    "special_chebyshev_polynomial_v.out",
    "special_chebyshev_polynomial_v.x_scalar",
    "special_chebyshev_polynomial_v.x_scalar_out",
    "special_chebyshev_polynomial_v.n_scalar",
    "special_chebyshev_polynomial_v.n_scalar_out",
    "special_chebyshev_polynomial_w.default",
    "special_chebyshev_polynomial_w.out",
    "special_chebyshev_polynomial_w.x_scalar",
    "special_chebyshev_polynomial_w.x_scalar_out",
    "special_chebyshev_polynomial_w.n_scalar",
    "special_chebyshev_polynomial_w.n_scalar_out",
    "special_entr.default",
    "special_entr.out",
    "special_erfcx.default",
    "special_erfcx.out",
    "special_hermite_polynomial_h.default",
    "special_hermite_polynomial_h.out",
    "special_hermite_polynomial_h.x_scalar",
    "special_hermite_polynomial_h.x_scalar_out",
    "special_hermite_polynomial_h.n_scalar",
    "special_hermite_polynomial_h.n_scalar_out",
    "special_hermite_polynomial_he.default",
    "special_hermite_polynomial_he.out",
    "special_hermite_polynomial_he.x_scalar",
    "special_hermite_polynomial_he.x_scalar_out",
    "special_hermite_polynomial_he.n_scalar",
    "special_hermite_polynomial_he.n_scalar_out",
    "special_i0e.default",
    "special_i0e.out",
    "special_i1.default",
    "special_i1.out",
    "special_i1e.default",
    "special_i1e.out",
    "special_laguerre_polynomial_l.default",
    "special_laguerre_polynomial_l.out",
    "special_laguerre_polynomial_l.x_scalar",
]


CUSTOM_TEMPLATES.update(
    {
        "rot90.default": _template_rot90_default,
        "rot90.out": _template_rot90_out,
        "rrelu_with_noise.default": _skip("random_op_not_supported"),
        "rrelu_with_noise.out": _skip("random_op_not_supported"),
        "rrelu_with_noise_.default": _skip("random_op_not_supported"),
        "rrelu_with_noise_backward.default": _template_rrelu_with_noise_backward,
        "rrelu_with_noise_backward.out": _template_rrelu_with_noise_backward_out,
        "rrelu_with_noise_functional.default": _skip("random_op_not_supported"),
        "scalar_tensor.default": _template_scalar_tensor,
        "scalar_tensor.out": _template_scalar_tensor_out,
        "scatter.value": _template_scatter_value,
        "scatter.src": _template_scatter_src,
        "scatter.reduce": _template_scatter_reduce,
        "scatter.value_reduce": _template_scatter_value_reduce,
        "scatter.src_out": _template_scatter_src_out,
        "scatter.value_out": _template_scatter_value_out,
        "scatter.reduce_out": _template_scatter_reduce_out,
        "scatter.value_reduce_out": _template_scatter_value_reduce_out,
        "scatter.dimname_src": _skip("named_tensor_torchscript"),
        "scatter.dimname_value": _skip("named_tensor_torchscript"),
        "scatter_.src": _template_scatter__src,
        "scatter_.value": _template_scatter__value,
        "scatter_.reduce": _template_scatter__reduce,
        "scatter_.value_reduce": _template_scatter__value_reduce,
        "scatter_add.default": _template_scatter_add,
        "scatter_add.out": _template_scatter_add_out,
        "scatter_add.dimname": _skip("named_tensor_torchscript"),
        "scatter_add_.default": _template_scatter_add,
        "scatter_reduce.two": _template_scatter_reduce_two,
        "scatter_reduce.two_out": _template_scatter_reduce_two_out,
        "scatter_reduce_.two": _template_scatter_reduce_two,
        "searchsorted.Tensor": _template_searchsorted_tensor,
        "searchsorted.Tensor_out": _template_searchsorted_tensor_out,
        "searchsorted.Scalar": _template_searchsorted_scalar,
        "searchsorted.Scalar_out": _template_searchsorted_scalar_out,
        "segment_reduce.default": _skip("segment_reduce_not_ready"),
        "segment_reduce.out": _skip("segment_reduce_not_ready"),
        "select.Dimname": _skip("dynamo_dimname_fake_tensor"),
        "select.int": _template_select_int,
        "select.t": _skip("dynamo_immutable_list"),
        "select_backward.default": _template_select_backward,
        "select_backward.out": _template_select_backward_out,
        "select_scatter.default": _template_select_scatter,
        "select_scatter.out": _template_select_scatter_out,
        "size.Dimname": _skip("dynamo_dimname_fake_tensor"),
        "slice.t": _skip("dynamo_immutable_list"),
        "slice.Tensor": _template_slice_tensor,
        "slice_backward.default": _template_slice_backward,
        "slice_backward.out": _template_slice_backward_out,
        "slice_scatter.default": _template_slice_scatter,
        "slice_scatter.out": _template_slice_scatter_out,
        "set_.default": _skip("mutation_set_not_supported"),
        "set_.source_Storage": _skip("mutation_set_not_supported"),
        "set_.source_Storage_storage_offset": _skip(
            "mutation_set_not_supported"
        ),
        "set_.source_Tensor": _skip("mutation_set_not_supported"),
        "set_.source_Tensor_storage_offset": _skip(
            "mutation_set_not_supported"
        ),
        "sigmoid_backward.grad_input": _template_sigmoid_backward_grad_input,
        "silu_backward.grad_input": _template_silu_backward_grad_input,
        "smooth_l1_loss_backward.grad_input": _template_smooth_l1_loss_backward_grad_input,
        "soft_margin_loss_backward.grad_input": _template_soft_margin_loss_backward_grad_input,
        "softplus_backward.grad_input": _template_softplus_backward_grad_input,
        "sort.default": _template_sort_default,
        "sort.stable": _template_sort_stable,
        "sort.values": _template_sort_values,
        "sort.values_stable": _template_sort_values_stable,
        "sort.dimname": _skip("dynamo_dimname_fake_tensor"),
        "sort.dimname_values": _skip("dynamo_dimname_fake_tensor"),
        "sort.dimname_stable": _skip("dynamo_dimname_fake_tensor"),
        "sort.dimname_values_stable": _skip("dynamo_dimname_fake_tensor"),
        "sort.str": _skip("dynamo_list_mutation"),
        "sort.any": _skip("dynamo_immutable_list"),
        "special_airy_ai.default": _skip("special_function_not_supported"),
        "special_airy_ai.out": _skip("special_function_not_supported"),
        "special_bessel_j0.default": _skip("special_function_not_supported"),
        "special_bessel_j0.out": _skip("special_function_not_supported"),
        "special_bessel_j1.default": _skip("special_function_not_supported"),
        "special_bessel_j1.out": _skip("special_function_not_supported"),
        "special_bessel_y0.default": _skip("special_function_not_supported"),
        "special_bessel_y0.out": _skip("special_function_not_supported"),
        "special_bessel_y1.default": _skip("special_function_not_supported"),
        "special_bessel_y1.out": _skip("special_function_not_supported"),
        "special_i0e.default": _skip("special_function_not_supported"),
        "special_i0e.out": _skip("special_function_not_supported"),
        "special_i1.default": _skip("special_function_not_supported"),
        "special_i1.out": _skip("special_function_not_supported"),
        "special_i1e.default": _skip("special_function_not_supported"),
        "special_i1e.out": _skip("special_function_not_supported"),
        "special_chebyshev_polynomial_t.default": _skip(
            "special_polynomial_not_supported"
        ),
        "special_chebyshev_polynomial_t.out": _skip(
            "special_polynomial_not_supported"
        ),
        "special_chebyshev_polynomial_t.x_scalar": _skip(
            "special_polynomial_not_supported"
        ),
        "special_chebyshev_polynomial_t.x_scalar_out": _skip(
            "special_polynomial_not_supported"
        ),
        "special_chebyshev_polynomial_t.n_scalar": _skip(
            "special_polynomial_not_supported"
        ),
        "special_chebyshev_polynomial_t.n_scalar_out": _skip(
            "special_polynomial_not_supported"
        ),
        "special_chebyshev_polynomial_u.default": _skip(
            "special_polynomial_not_supported"
        ),
        "special_chebyshev_polynomial_u.out": _skip(
            "special_polynomial_not_supported"
        ),
        "special_chebyshev_polynomial_u.x_scalar": _skip(
            "special_polynomial_not_supported"
        ),
        "special_chebyshev_polynomial_u.x_scalar_out": _skip(
            "special_polynomial_not_supported"
        ),
        "special_chebyshev_polynomial_u.n_scalar": _skip(
            "special_polynomial_not_supported"
        ),
        "special_chebyshev_polynomial_u.n_scalar_out": _skip(
            "special_polynomial_not_supported"
        ),
        "special_chebyshev_polynomial_v.default": _skip(
            "special_polynomial_not_supported"
        ),
        "special_chebyshev_polynomial_v.out": _skip(
            "special_polynomial_not_supported"
        ),
        "special_chebyshev_polynomial_v.x_scalar": _skip(
            "special_polynomial_not_supported"
        ),
        "special_chebyshev_polynomial_v.x_scalar_out": _skip(
            "special_polynomial_not_supported"
        ),
        "special_chebyshev_polynomial_v.n_scalar": _skip(
            "special_polynomial_not_supported"
        ),
        "special_chebyshev_polynomial_v.n_scalar_out": _skip(
            "special_polynomial_not_supported"
        ),
        "special_chebyshev_polynomial_w.default": _skip(
            "special_polynomial_not_supported"
        ),
        "special_chebyshev_polynomial_w.out": _skip(
            "special_polynomial_not_supported"
        ),
        "special_chebyshev_polynomial_w.x_scalar": _skip(
            "special_polynomial_not_supported"
        ),
        "special_chebyshev_polynomial_w.x_scalar_out": _skip(
            "special_polynomial_not_supported"
        ),
        "special_chebyshev_polynomial_w.n_scalar": _skip(
            "special_polynomial_not_supported"
        ),
        "special_chebyshev_polynomial_w.n_scalar_out": _skip(
            "special_polynomial_not_supported"
        ),
        "special_hermite_polynomial_h.default": _skip(
            "special_polynomial_not_supported"
        ),
        "special_hermite_polynomial_h.out": _skip(
            "special_polynomial_not_supported"
        ),
        "special_hermite_polynomial_h.x_scalar": _skip(
            "special_polynomial_not_supported"
        ),
        "special_hermite_polynomial_h.x_scalar_out": _skip(
            "special_polynomial_not_supported"
        ),
        "special_hermite_polynomial_h.n_scalar": _skip(
            "special_polynomial_not_supported"
        ),
        "special_hermite_polynomial_h.n_scalar_out": _skip(
            "special_polynomial_not_supported"
        ),
        "special_hermite_polynomial_he.default": _skip(
            "special_polynomial_not_supported"
        ),
        "special_hermite_polynomial_he.out": _skip(
            "special_polynomial_not_supported"
        ),
        "special_hermite_polynomial_he.x_scalar": _skip(
            "special_polynomial_not_supported"
        ),
        "special_hermite_polynomial_he.x_scalar_out": _skip(
            "special_polynomial_not_supported"
        ),
        "special_hermite_polynomial_he.n_scalar": _skip(
            "special_polynomial_not_supported"
        ),
        "special_hermite_polynomial_he.n_scalar_out": _skip(
            "special_polynomial_not_supported"
        ),
        "special_laguerre_polynomial_l.default": _skip(
            "special_polynomial_not_supported"
        ),
        "special_laguerre_polynomial_l.out": _skip(
            "special_polynomial_not_supported"
        ),
        "special_laguerre_polynomial_l.x_scalar": _skip(
            "special_polynomial_not_supported"
        ),
        "signbit.out": _template_signbit_out,
    }
)

if __name__ == "__main__":
    run_aten_op_batch(
        OPS,
        batch_label="test_batch_7",
        max_fails=20,
        templates=CUSTOM_TEMPLATES,
    )
# CHECK: SUMMARY pass=
# CHECK-SAME: fail=0
