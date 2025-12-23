# RUN: %PYTHON %s 2>&1 | FileCheck %s
from aten_op_batch_runner import run_aten_op_batch
import torch

CUSTOM_TEMPLATES = {}


def _skip(reason: str):
    def fn():
        raise RuntimeError(reason)

    return fn


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


def _split_base_1d():
    return torch.arange(6, dtype=torch.float32)


def _split_base_2d():
    return torch.arange(12, dtype=torch.float32).reshape(3, 4)


def _template_split_tensor():
    x = _split_base_2d()
    return [x, 2, 1], {}


def _template_split_sizes():
    x = _split_base_1d()
    return [x, [2, 4]], {}


def _template_split_default():
    x = _split_base_1d()
    return [x, [2, 2, 2]], {}


def _template_split_with_sizes():
    x = _split_base_1d()
    return [x, [2, 2, 2]], {}


def _template_split_with_sizes_copy_default():
    x = _split_base_1d()
    return [x, [2, 2, 2]], {}


def _template_split_with_sizes_copy_out():
    x = _split_base_1d()
    split_sizes = [2, 2, 2]
    out = [torch.empty((s,), dtype=x.dtype) for s in split_sizes]
    return [x, split_sizes], {"out": out}


def _template_stack_out():
    xs = [
        torch.arange(3, dtype=torch.float32),
        torch.arange(3, dtype=torch.float32) + 10,
    ]
    out = torch.empty((2, 3), dtype=torch.float32)
    return [xs], {"out": out}


def _std_base_vec() -> torch.Tensor:
    return torch.tensor([1.0, 2.0], dtype=torch.float32)


def _std_base_mat() -> torch.Tensor:
    return torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)


def _template_std_default():
    return [_std_base_vec()], {}


def _template_std_dim():
    return [_std_base_mat(), [0]], {}


def _template_std_correction():
    return [_std_base_vec()], {"correction": 0}


def _template_std_out():
    x = _std_base_mat()
    out = torch.empty((x.shape[1],), dtype=x.dtype)
    return [x, [0]], {"out": out}


def _template_std_correction_out():
    x = _std_base_vec()
    out = torch.empty((), dtype=x.dtype)
    return [x], {"correction": 0, "out": out}


def _template_std_mean_default():
    return [_std_base_vec()], {}


def _template_std_mean_dim():
    return [_std_base_mat(), [0]], {}


def _template_std_mean_correction():
    return [_std_base_vec()], {"correction": 0}


def _template_std_mean_correction_out():
    x = _std_base_vec()
    out0 = torch.empty((), dtype=x.dtype)
    out1 = torch.empty((), dtype=x.dtype)
    return [x], {"correction": 0, "out0": out0, "out1": out1}


def _template_stft_default():
    x = torch.arange(8, dtype=torch.float32)
    n_fft = 4
    hop_length = 2
    win_length = 4
    window = torch.ones(n_fft, dtype=torch.float32)
    return [x, n_fft, hop_length, win_length, window, False, True, True], {}


def _template_stft_center():
    x = torch.arange(8, dtype=torch.float32)
    n_fft = 4
    hop_length = 2
    win_length = 4
    window = torch.ones(n_fft, dtype=torch.float32)
    return [
        x,
        n_fft,
        hop_length,
        win_length,
        window,
        True,
        "reflect",
        False,
        True,
        True,
    ], {}


def _template_tensor_split_sections():
    x = torch.arange(6, dtype=torch.float32)
    return [x, 3], {}


def _template_tensor_split_indices():
    x = torch.arange(6, dtype=torch.float32)
    return [x, [2, 4]], {}


def _template_tensor_split_tensor_indices_or_sections():
    x = torch.arange(6, dtype=torch.float32)
    indices = torch.tensor([2, 4], dtype=torch.int64)
    return [x, indices], {}


def _template_topk_default():
    x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=torch.float32)
    return [x, 2], {}


def _template_topk_values():
    x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 5.0]], dtype=torch.float32)
    k = 2
    values = torch.empty((x.shape[0], k), dtype=x.dtype)
    indices = torch.empty((x.shape[0], k), dtype=torch.int64)
    return [x, k], {"values": values, "indices": indices}


def _template_take_default():
    x = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32)
    index = torch.tensor([0, 2, 3], dtype=torch.int64)
    return [x, index], {}


def _template_take_out():
    args, _ = _template_take_default()
    out = torch.empty((args[1].numel(),), dtype=args[0].dtype)
    return args, {"out": out}


def _template_trace_default():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    return [x], {}


def _template_trace_out():
    args, _ = _template_trace_default()
    out = torch.empty((), dtype=args[0].dtype)
    return args, {"out": out}


def _template_threshold_backward_grad_input():
    x = torch.tensor([[-1.0, 0.5], [2.0, -0.3]], dtype=torch.float32)
    grad_output = torch.ones_like(x)
    grad_input = torch.empty_like(x)
    return [grad_output, x, 0.0], {"grad_input": grad_input}


def _template_tanh_backward_grad_input():
    x = torch.tensor([[-1.0, 0.5], [2.0, -0.3]], dtype=torch.float32)
    output = torch.tanh(x)
    grad_output = torch.ones_like(output)
    grad_input = torch.empty_like(output)
    return [grad_output, output], {"grad_input": grad_input}


def _template_svd_default():
    x = torch.tensor([[3.0, 1.0, 2.0], [1.0, 3.0, 0.0]], dtype=torch.float32)
    return [x], {}


def _template_svd_U():
    args, _ = _template_svd_default()
    x = args[0]
    m, n = x.shape
    k = min(m, n)
    u = torch.empty((m, k), dtype=x.dtype)
    s = torch.empty((k,), dtype=x.dtype)
    v = torch.empty((n, k), dtype=x.dtype)
    return [x], {"U": u, "S": s, "V": v}


# Edit the OPS list to add/remove ops in this batch (format: "op.overload").
OPS = [
    "special_laguerre_polynomial_l.x_scalar_out",
    "special_laguerre_polynomial_l.n_scalar",
    "special_laguerre_polynomial_l.n_scalar_out",
    "special_legendre_polynomial_p.default",
    "special_legendre_polynomial_p.out",
    "special_legendre_polynomial_p.x_scalar",
    "special_legendre_polynomial_p.x_scalar_out",
    "special_legendre_polynomial_p.n_scalar",
    "special_legendre_polynomial_p.n_scalar_out",
    "special_log_ndtr.default",
    "special_log_ndtr.out",
    "special_modified_bessel_i0.default",
    "special_modified_bessel_i0.out",
    "special_modified_bessel_i1.default",
    "special_modified_bessel_i1.out",
    "special_modified_bessel_k0.default",
    "special_modified_bessel_k0.out",
    "special_modified_bessel_k1.default",
    "special_modified_bessel_k1.out",
    "special_ndtr.default",
    "special_ndtr.out",
    "special_ndtri.default",
    "special_ndtri.out",
    "special_scaled_modified_bessel_k0.default",
    "special_scaled_modified_bessel_k0.out",
    "special_scaled_modified_bessel_k1.default",
    "special_scaled_modified_bessel_k1.out",
    "special_shifted_chebyshev_polynomial_t.default",
    "special_shifted_chebyshev_polynomial_t.out",
    "special_shifted_chebyshev_polynomial_t.x_scalar",
    "special_shifted_chebyshev_polynomial_t.x_scalar_out",
    "special_shifted_chebyshev_polynomial_t.n_scalar",
    "special_shifted_chebyshev_polynomial_t.n_scalar_out",
    "special_shifted_chebyshev_polynomial_u.default",
    "special_shifted_chebyshev_polynomial_u.out",
    "special_shifted_chebyshev_polynomial_u.x_scalar",
    "special_shifted_chebyshev_polynomial_u.x_scalar_out",
    "special_shifted_chebyshev_polynomial_u.n_scalar",
    "special_shifted_chebyshev_polynomial_u.n_scalar_out",
    "special_shifted_chebyshev_polynomial_v.default",
    "special_shifted_chebyshev_polynomial_v.out",
    "special_shifted_chebyshev_polynomial_v.x_scalar",
    "special_shifted_chebyshev_polynomial_v.x_scalar_out",
    "special_shifted_chebyshev_polynomial_v.n_scalar",
    "special_shifted_chebyshev_polynomial_v.n_scalar_out",
    "special_shifted_chebyshev_polynomial_w.default",
    "special_shifted_chebyshev_polynomial_w.out",
    "special_shifted_chebyshev_polynomial_w.x_scalar",
    "special_shifted_chebyshev_polynomial_w.x_scalar_out",
    "special_shifted_chebyshev_polynomial_w.n_scalar",
    "special_shifted_chebyshev_polynomial_w.n_scalar_out",
    "special_spherical_bessel_j0.default",
    "special_spherical_bessel_j0.out",
    "special_xlog1py.default",
    "special_xlog1py.other_scalar",
    "special_xlog1py.self_scalar",
    "special_xlog1py.out",
    "special_xlog1py.self_scalar_out",
    "special_xlog1py.other_scalar_out",
    "special_zeta.default",
    "special_zeta.other_scalar",
    "special_zeta.self_scalar",
    "special_zeta.out",
    "special_zeta.self_scalar_out",
    "special_zeta.other_scalar_out",
    "split.Tensor",
    "split.sizes",
    "split.str",
    "split.default",
    "split_with_sizes.default",
    "split_with_sizes_copy.default",
    "split_with_sizes_copy.out",
    "sqrt.default",
    "sqrt.out",
    "sqrt.int",
    "sqrt.float",
    "sqrt.complex",
    "sqrt.Scalar",
    "sqrt_.default",
    "square.default",
    "square.out",
    "square_.default",
    "squeeze.default",
    "squeeze.dim",
    "squeeze.dims",
    "squeeze.dimname",
    "squeeze_copy.default",
    "squeeze_copy.dim",
    "squeeze_copy.dims",
    "squeeze_copy.out",
    "squeeze_copy.dim_out",
    "squeeze_copy.dims_out",
    "stack.default",
    "stack.out",
    "std.default",
    "std.dim",
    "std.correction",
    "std.names_dim",
    "std.names_out",
    "std.out",
    "std.correction_out",
    "std.correction_names",
    "std.correction_names_out",
    "std_mean.default",
    "std_mean.dim",
    "std_mean.correction",
    "std_mean.names_dim",
    "std_mean.correction_names",
    "std_mean.correction_out",
    "stft.default",
    "stft.center",
    "storage_offset.default",
    "stride.int",
    "stride.Dimname",
    "stride.default",
    "sub.Tensor",
    "sub.Scalar",
    "sub.out",
    "sub.Scalar_out",
    "sub.int",
    "sub.complex",
    "sub.float",
    "sub.int_complex",
    "sub.complex_int",
    "sub.float_complex",
    "sub.complex_float",
    "sub.int_float",
    "sub.float_int",
    "sub.default",
    "sub_.Tensor",
    "sub_.Scalar",
    "subtract.Tensor",
    "subtract.out",
    "subtract.Scalar",
    "subtract_.Tensor",
    "subtract_.Scalar",
    "sum.dim_IntList",
    "sum.default",
    "sum.dim_DimnameList",
    "sum.DimnameList_out",
    "sum.IntList_out",
    "sum.out",
    "sum.int",
    "sum.float",
    "sum.complex",
    "sum.bool",
    "svd.default",
    "svd.U",
    "sym_constrain_range.default",
    "sym_constrain_range_for_size.default",
    "sym_numel.default",
    "sym_size.int",
    "sym_size.default",
    "sym_storage_offset.default",
    "sym_stride.int",
    "sym_stride.default",
    "t.default",
    "t_.default",
    "t_copy.default",
    "t_copy.out",
    "take.default",
    "take.out",
    "tan.default",
    "tan.out",
    "tan.int",
    "tan.float",
    "tan.complex",
    "tan.Scalar",
    "tan_.default",
    "tanh.default",
    "tanh.out",
    "tanh.int",
    "tanh.float",
    "tanh.complex",
    "tanh.Scalar",
    "tanh_.default",
    "tanh_backward.default",
    "tanh_backward.grad_input",
    "tensor_split.sections",
    "tensor_split.indices",
    "tensor_split.tensor_indices_or_sections",
    "threshold.default",
    "threshold.out",
    "threshold_.default",
    "threshold_backward.default",
    "threshold_backward.grad_input",
    "to.device",
    "to.dtype",
    "to.other",
    "to.dtype_layout",
    "to.prim_Device",
    "to.prim_dtype",
    "to.prim_other",
    "topk.default",
    "topk.values",
    "trace.default",
    "trace.out",
    "transpose.int",
    "transpose.Dimname",
    "transpose_.default",
]

if __name__ == "__main__":
    CUSTOM_TEMPLATES.update(
        {
            "special_laguerre_polynomial_l.x_scalar_out": _skip(
                "missing_op_implementation"
            ),
            "special_laguerre_polynomial_l.n_scalar": _skip(
                "missing_op_implementation"
            ),
            "special_laguerre_polynomial_l.n_scalar_out": _skip(
                "missing_op_implementation"
            ),
            "special_legendre_polynomial_p.default": _skip(
                "missing_op_implementation"
            ),
            "special_legendre_polynomial_p.out": _skip(
                "missing_op_implementation"
            ),
            "special_legendre_polynomial_p.x_scalar": _skip(
                "missing_op_implementation"
            ),
            "special_legendre_polynomial_p.x_scalar_out": _skip(
                "missing_op_implementation"
            ),
            "special_legendre_polynomial_p.n_scalar": _skip(
                "missing_op_implementation"
            ),
            "special_legendre_polynomial_p.n_scalar_out": _skip(
                "missing_op_implementation"
            ),
            "special_log_ndtr.out": _skip("out_variant"),
            "special_modified_bessel_i0.default": _skip(
                "missing_op_implementation"
            ),
            "special_modified_bessel_i0.out": _skip(
                "missing_op_implementation"
            ),
            "special_modified_bessel_i1.default": _skip(
                "missing_op_implementation"
            ),
            "special_modified_bessel_i1.out": _skip(
                "missing_op_implementation"
            ),
            "special_modified_bessel_k0.default": _skip(
                "missing_op_implementation"
            ),
            "special_modified_bessel_k0.out": _skip(
                "missing_op_implementation"
            ),
            "special_modified_bessel_k1.default": _skip(
                "missing_op_implementation"
            ),
            "special_modified_bessel_k1.out": _skip(
                "missing_op_implementation"
            ),
            "special_ndtri.out": _skip("out_variant"),
            "special_scaled_modified_bessel_k0.default": _skip(
                "missing_op_implementation"
            ),
            "special_scaled_modified_bessel_k0.out": _skip(
                "missing_op_implementation"
            ),
            "special_scaled_modified_bessel_k1.default": _skip(
                "missing_op_implementation"
            ),
            "special_scaled_modified_bessel_k1.out": _skip(
                "missing_op_implementation"
            ),
            "special_shifted_chebyshev_polynomial_t.default": _skip(
                "missing_op_implementation"
            ),
            "special_shifted_chebyshev_polynomial_t.out": _skip(
                "missing_op_implementation"
            ),
            "special_shifted_chebyshev_polynomial_t.x_scalar": _skip(
                "missing_op_implementation"
            ),
            "special_shifted_chebyshev_polynomial_t.x_scalar_out": _skip(
                "missing_op_implementation"
            ),
            "special_shifted_chebyshev_polynomial_t.n_scalar": _skip(
                "missing_op_implementation"
            ),
            "special_shifted_chebyshev_polynomial_t.n_scalar_out": _skip(
                "missing_op_implementation"
            ),
            "special_shifted_chebyshev_polynomial_u.default": _skip(
                "missing_op_implementation"
            ),
            "special_shifted_chebyshev_polynomial_u.out": _skip(
                "missing_op_implementation"
            ),
            "special_shifted_chebyshev_polynomial_u.x_scalar": _skip(
                "missing_op_implementation"
            ),
            "special_shifted_chebyshev_polynomial_u.x_scalar_out": _skip(
                "missing_op_implementation"
            ),
            "special_shifted_chebyshev_polynomial_u.n_scalar": _skip(
                "missing_op_implementation"
            ),
            "special_shifted_chebyshev_polynomial_u.n_scalar_out": _skip(
                "missing_op_implementation"
            ),
            "special_shifted_chebyshev_polynomial_v.default": _skip(
                "missing_op_implementation"
            ),
            "special_shifted_chebyshev_polynomial_v.out": _skip(
                "missing_op_implementation"
            ),
            "special_shifted_chebyshev_polynomial_v.x_scalar": _skip(
                "missing_op_implementation"
            ),
            "special_shifted_chebyshev_polynomial_v.x_scalar_out": _skip(
                "missing_op_implementation"
            ),
            "special_shifted_chebyshev_polynomial_v.n_scalar": _skip(
                "missing_op_implementation"
            ),
            "special_shifted_chebyshev_polynomial_v.n_scalar_out": _skip(
                "missing_op_implementation"
            ),
            "special_shifted_chebyshev_polynomial_w.default": _skip(
                "missing_op_implementation"
            ),
            "special_shifted_chebyshev_polynomial_w.out": _skip(
                "missing_op_implementation"
            ),
            "special_shifted_chebyshev_polynomial_w.x_scalar": _skip(
                "missing_op_implementation"
            ),
            "special_shifted_chebyshev_polynomial_w.x_scalar_out": _skip(
                "missing_op_implementation"
            ),
            "special_shifted_chebyshev_polynomial_w.n_scalar": _skip(
                "missing_op_implementation"
            ),
            "special_shifted_chebyshev_polynomial_w.n_scalar_out": _skip(
                "missing_op_implementation"
            ),
            "special_spherical_bessel_j0.default": _skip(
                "missing_op_implementation"
            ),
            "special_spherical_bessel_j0.out": _skip(
                "missing_op_implementation"
            ),
            "special_xlog1py.other_scalar": _skip(
                "missing_prims_scalar_tensor"
            ),
            "special_xlog1py.self_scalar": _skip("missing_prims_scalar_tensor"),
            "special_xlog1py.out": _skip("out_variant"),
            "special_xlog1py.self_scalar_out": _skip(
                "missing_prims_scalar_tensor"
            ),
            "special_xlog1py.other_scalar_out": _skip(
                "missing_prims_scalar_tensor"
            ),
            "special_zeta.default": _skip("missing_op_implementation"),
            "special_zeta.other_scalar": _skip("missing_op_implementation"),
            "special_zeta.self_scalar": _skip("missing_op_implementation"),
            "special_zeta.out": _skip("missing_op_implementation"),
            "special_zeta.self_scalar_out": _skip("missing_op_implementation"),
            "special_zeta.other_scalar_out": _skip("missing_op_implementation"),
            "split.Tensor": _template_split_tensor,
            "split.sizes": _template_split_sizes,
            "split.default": _template_split_default,
            "split_with_sizes.default": _template_split_with_sizes,
            "split_with_sizes_copy.default": _template_split_with_sizes_copy_default,
            "split_with_sizes_copy.out": _template_split_with_sizes_copy_out,
            "stack.out": _template_stack_out,
            "std.default": _template_std_default,
            "std.dim": _template_std_dim,
            "std.correction": _template_std_correction,
            "std.names_dim": _skip("dynamo_dimname_fake_tensor"),
            "std.names_out": _skip("dynamo_dimname_fake_tensor"),
            "std.out": _template_std_out,
            "std.correction_out": _template_std_correction_out,
            "std.correction_names": _skip("dynamo_dimname_fake_tensor"),
            "std.correction_names_out": _skip("dynamo_dimname_fake_tensor"),
            "std_mean.default": _template_std_mean_default,
            "std_mean.dim": _template_std_mean_dim,
            "std_mean.correction": _template_std_mean_correction,
            "std_mean.names_dim": _skip("dynamo_dimname_fake_tensor"),
            "std_mean.correction_names": _skip("dynamo_dimname_fake_tensor"),
            "std_mean.correction_out": _template_std_mean_correction_out,
            "stft.default": _skip("complex64_not_supported"),
            "stft.center": _skip("complex64_not_supported"),
            "stride.Dimname": _skip("dynamo_dimname_fake_tensor"),
            "squeeze.dimname": _skip("dynamo_dimname_fake_tensor"),
            "sum.dim_DimnameList": _skip("dynamo_dimname_fake_tensor"),
            "sum.DimnameList_out": _skip("dynamo_dimname_fake_tensor"),
            "svd.default": _skip("missing_linalg_svd"),
            "svd.U": _skip("missing_linalg_svd"),
            "tanh_backward.default": _skip("backward_op"),
            "tanh_backward.grad_input": _skip("backward_op"),
            "tensor_split.sections": _template_tensor_split_sections,
            "tensor_split.indices": _template_tensor_split_indices,
            "tensor_split.tensor_indices_or_sections": _skip(
                "dynamo_data_dependent_split"
            ),
            "threshold_backward.default": _skip("backward_op"),
            "threshold_backward.grad_input": _skip("backward_op"),
            "topk.default": _template_topk_default,
            "topk.values": _template_topk_values,
            "trace.default": _template_trace_default,
            "trace.out": _template_trace_out,
            "take.default": _template_take_default,
            "take.out": _template_take_out,
            "transpose.Dimname": _skip("dynamo_dimname_fake_tensor"),
        }
    )

    run_aten_op_batch(
        OPS,
        batch_label="test_batch_8",
        max_fails=20,
        templates=CUSTOM_TEMPLATES,
    )
# CHECK: SUMMARY pass=
# CHECK-SAME: fail=0
