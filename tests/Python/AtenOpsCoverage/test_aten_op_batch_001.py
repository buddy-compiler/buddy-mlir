# RUN: %PYTHON %s 2>&1 | FileCheck %s
from aten_op_batch_runner import run_aten_op_batch
import torch
import torch.nn.functional as F


CUSTOM_TEMPLATES = {}


def _skip(reason: str):
    def fn():
        raise RuntimeError(reason)

    return fn


def _bitwise_pair():
    a = torch.tensor([1, 2, 3], dtype=torch.int32)
    b = torch.tensor([3, 2, 1], dtype=torch.int32)
    return a, b


def _bitwise_tensor_tensor():
    a, b = _bitwise_pair()
    return [a, b], {}


def _bitwise_tensor_tensor_out():
    args, _ = _bitwise_tensor_tensor()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _bitwise_tensor_scalar():
    a, _ = _bitwise_pair()
    return [a, 1], {}


def _bitwise_tensor_scalar_out():
    args, _ = _bitwise_tensor_scalar()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _bitwise_scalar_tensor():
    _, b = _bitwise_pair()
    return [1, b], {}


def _bitwise_scalar_tensor_out():
    args, _ = _bitwise_scalar_tensor()
    out = torch.empty_like(args[1])
    return args, {"out": out}


def _bitwise_not():
    x, _ = _bitwise_pair()
    return [x], {}


def _bitwise_not_out():
    args, _ = _bitwise_not()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _bitwise_not_inplace():
    x, _ = _bitwise_pair()
    return [x.clone()], {}


def _bitwise_inplace_tensor_tensor():
    a, b = _bitwise_pair()
    return [a.clone(), b], {}


def _bitwise_inplace_tensor_scalar():
    a, _ = _bitwise_pair()
    return [a.clone(), 1], {}


def _template_bincount():
    x = torch.tensor([0, 1, 1, 2], dtype=torch.int64)
    return [x], {}


def _template_bincount_out():
    args, _ = _template_bincount()
    out = torch.empty(3, dtype=torch.int64)
    return args, {"out": out}


def _template_bernoulli_float():
    x = torch.tensor([0.4, 0.6], dtype=torch.float32)
    return [x.clone(), 0.5], {}


def _template_block_diag():
    a = torch.eye(2, dtype=torch.float32)
    b = torch.ones(1, 1, dtype=torch.float32)
    return [[a, b]], {}


def _template_block_diag_out():
    args, _ = _template_block_diag()
    out = torch.empty((3, 3), dtype=torch.float32)
    return args, {"out": out}


def _template_bmm():
    a = torch.randn(2, 3, 4, dtype=torch.float32)
    b = torch.randn(2, 4, 5, dtype=torch.float32)
    return [a, b], {}


def _template_bmm_out():
    args, _ = _template_bmm()
    out = torch.empty((2, 3, 5), dtype=torch.float32)
    return args, {"out": out}


def _template_bmm_dtype():
    args, _ = _template_bmm()
    return args + [torch.float32], {}


def _template_bmm_dtype_out():
    args, _ = _template_bmm()
    out_dtype = torch.float32
    out = torch.empty((2, 3, 5), dtype=out_dtype)
    return args + [out_dtype], {"out": out}


def _template_bucketize_tensor():
    boundaries = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float32)
    values = torch.tensor([0.5, 3.5], dtype=torch.float32)
    return [values, boundaries], {}


def _template_bucketize_tensor_out():
    args, _ = _template_bucketize_tensor()
    out = torch.empty_like(args[0], dtype=torch.int64)
    return args, {"out": out}


def _template_bucketize_scalar():
    boundaries = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float32)
    return [2.5, boundaries], {}


def _template_bucketize_scalar_out():
    args, _ = _template_bucketize_scalar()
    out = torch.empty((), dtype=torch.int64)
    return args, {"out": out}


def _template_broadcast_tensors():
    a = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    b = torch.tensor([1.0, 2.0], dtype=torch.float32)
    return [[a, b]], {}


def _named_pair():
    t1 = torch.randn(1, 2, dtype=torch.float32).refine_names("N", "C")
    t2 = torch.randn(1, 2, dtype=torch.float32).refine_names("N", "C")
    return t1, t2


def _template_cat_names():
    t1, t2 = _named_pair()
    return [[t1, t2], "C"], {}


def _template_cat_names_out():
    args, _ = _template_cat_names()
    out = torch.empty((1, 4), dtype=torch.float32).refine_names("N", "C")
    return args, {"out": out}


def _template_cat_default():
    t1 = torch.randn(1, 2, dtype=torch.float32)
    t2 = torch.randn(1, 2, dtype=torch.float32)
    return [[t1, t2], 0], {}


def _template_cat_out():
    args, _ = _template_cat_default()
    out = torch.empty((2, 2), dtype=torch.float32)
    return args, {"out": out}


def _template_channel_shuffle():
    x = torch.randn(1, 4, 2, 2, dtype=torch.float32)
    return [x, 2], {}


def _template_channel_shuffle_out():
    args, _ = _template_channel_shuffle()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _spd_matrix():
    return torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float32)


def _template_cholesky():
    a = _spd_matrix()
    return [a, False], {}


def _template_cholesky_out():
    args, _ = _template_cholesky()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_cholesky_inverse():
    a = torch.linalg.cholesky(_spd_matrix())
    return [a, False], {}


def _template_cholesky_inverse_out():
    args, _ = _template_cholesky_inverse()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_cholesky_solve():
    b = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    u = torch.linalg.cholesky(_spd_matrix())
    return [b, u, False], {}


def _template_cholesky_solve_out():
    args, _ = _template_cholesky_solve()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_col2im():
    cols = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    output_size = [4, 4]
    kernel_size = [2, 2]
    dilation = [1, 1]
    padding = [0, 0]
    stride = [2, 2]
    return [cols, output_size, kernel_size, dilation, padding, stride], {}


def _template_col2im_out():
    args, _ = _template_col2im()
    out = torch.empty((1, 1, 4, 4), dtype=torch.float32)
    return args, {"out": out}


def _template_constant_pad_nd():
    x = torch.ones(1, 2, dtype=torch.float32)
    pad = [1, 1, 1, 1]
    return [x, pad, 0.5], {}


def _template_constant_pad_nd_out():
    args, _ = _template_constant_pad_nd()
    x, pad, _ = args
    out_shape = (x.shape[0] + pad[2] + pad[3], x.shape[1] + pad[0] + pad[1])
    out = torch.empty(out_shape, dtype=x.dtype)
    return args, {"out": out}


def _template_cauchy():
    x = torch.ones(2, dtype=torch.float32)
    return [x, 0.0, 1e-3], {}


def _template_cauchy_out():
    args, _ = _template_cauchy()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_cauchy_inplace():
    args, _ = _template_cauchy()
    args[0] = args[0].clone()
    return args, {}


def _template_complex_out():
    real = torch.tensor([1.0], dtype=torch.float32)
    imag = torch.tensor([2.0], dtype=torch.float32)
    out = torch.empty_like(real, dtype=torch.complex64)
    return [real, imag], {"out": out}


def _conv2d_base():
    inp = torch.randn(1, 1, 4, 4, dtype=torch.float32)
    weight = torch.randn(1, 1, 3, 3, dtype=torch.float32)
    bias = torch.randn(1, dtype=torch.float32)
    stride = [1, 1]
    padding = [1, 1]
    dilation = [1, 1]
    groups = 1
    return inp, weight, bias, stride, padding, dilation, groups


def _template_conv2d_default():
    args = _conv2d_base()
    return list(args), {}


def _template_conv2d_padding():
    inp, weight, bias, stride, _, dilation, groups = _conv2d_base()
    return [inp, weight, bias, stride, "valid", dilation, groups], {}


def _convolution_base():
    inp = torch.randn(1, 1, 4, 4, dtype=torch.float32)
    weight = torch.randn(1, 1, 3, 3, dtype=torch.float32)
    bias = torch.randn(1, dtype=torch.float32)
    stride = [1, 1]
    padding = [1, 1]
    dilation = [1, 1]
    output_padding = [0, 0]
    groups = 1
    transposed = False
    out = F.conv2d(
        inp,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    return {
        "input": inp,
        "weight": weight,
        "bias": bias,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "output_padding": output_padding,
        "groups": groups,
        "transposed": transposed,
        "output": out,
    }


def _template_convolution():
    cfg = _convolution_base()
    return [
        cfg["input"],
        cfg["weight"],
        cfg["bias"],
        cfg["stride"],
        cfg["padding"],
        cfg["dilation"],
        cfg["transposed"],
        cfg["output_padding"],
        cfg["groups"],
    ], {}


def _template_convolution_out():
    cfg = _convolution_base()
    out = torch.empty_like(cfg["output"])
    return [
        cfg["input"],
        cfg["weight"],
        cfg["bias"],
        cfg["stride"],
        cfg["padding"],
        cfg["dilation"],
        cfg["transposed"],
        cfg["output_padding"],
        cfg["groups"],
    ], {"out": out}


def _template_convolution_backward():
    cfg = _convolution_base()
    grad_output = torch.randn_like(cfg["output"])
    bias_sizes = [cfg["bias"].shape[0]]
    output_mask = [True, True, True]
    return [
        grad_output,
        cfg["input"],
        cfg["weight"],
        bias_sizes,
        cfg["stride"],
        cfg["padding"],
        cfg["dilation"],
        cfg["transposed"],
        cfg["output_padding"],
        cfg["groups"],
        output_mask,
    ], {}


def _template_convolution_backward_out():
    cfg = _convolution_base()
    grad_output = torch.randn_like(cfg["output"])
    bias_sizes = [cfg["bias"].shape[0]]
    output_mask = [True, True, True]
    grad_input = torch.empty_like(cfg["input"])
    grad_weight = torch.empty_like(cfg["weight"])
    grad_bias = torch.empty_like(cfg["bias"])
    return [
        grad_output,
        cfg["input"],
        cfg["weight"],
        bias_sizes,
        cfg["stride"],
        cfg["padding"],
        cfg["dilation"],
        cfg["transposed"],
        cfg["output_padding"],
        cfg["groups"],
        output_mask,
    ], {"out0": grad_input, "out1": grad_weight, "out2": grad_bias}


def _template_bce_backward_grad_input():
    inp = torch.tensor([0.5, 0.2], dtype=torch.float32)
    target = torch.tensor([1.0, 0.0], dtype=torch.float32)
    grad_output = torch.ones_like(inp)
    grad_input = torch.empty_like(inp)
    return [grad_output, inp, target], {"grad_input": grad_input}


def _template_count_nonzero_out():
    x = torch.tensor([[0, 1], [1, 1]], dtype=torch.float32)
    out = torch.empty((), dtype=torch.int64)
    return [x], {"out": out}


def _template_count_nonzero_dimlist():
    x = torch.tensor([[0, 1], [2, 0]], dtype=torch.float32)
    return [x, [1]], {}


def _template_count_nonzero_dimlist_out():
    x = torch.tensor([[0, 1], [2, 0]], dtype=torch.float32)
    out = torch.empty((2,), dtype=torch.int64)
    return [x, [1]], {"out": out}


def _named_matrix():
    return torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32
    ).refine_names("N", "C")


def _template_cummax():
    x = torch.tensor([[1.0, 2.0], [0.5, 3.0]], dtype=torch.float32)
    return [x, 1], {}


def _template_cummax_out():
    args, _ = _template_cummax()
    values = torch.empty_like(args[0])
    indices = torch.empty_like(args[0], dtype=torch.int64)
    return args, {"values": values, "indices": indices}


def _template_cummin():
    x = torch.tensor([[1.0, 2.0], [0.5, 3.0]], dtype=torch.float32)
    return [x, 1], {}


def _template_cummin_out():
    args, _ = _template_cummin()
    values = torch.empty_like(args[0])
    indices = torch.empty_like(args[0], dtype=torch.int64)
    return args, {"values": values, "indices": indices}


def _template_cummax_dimname():
    x = _named_matrix()
    return [x, "C"], {}


def _template_cummax_dimname_out():
    x = _named_matrix()
    values = torch.empty_like(x)
    indices = torch.empty_like(x.rename(None), dtype=torch.int64).refine_names(
        *x.names
    )
    return [x, "C"], {"values": values, "indices": indices}


def _template_cummin_dimname():
    x = _named_matrix()
    return [x, "C"], {}


def _template_cummin_dimname_out():
    x = _named_matrix()
    values = torch.empty_like(x)
    indices = torch.empty_like(x.rename(None), dtype=torch.int64).refine_names(
        *x.names
    )
    return [x, "C"], {"values": values, "indices": indices}


def _template_cumprod_dimname():
    x = _named_matrix()
    return [x, "C"], {}


def _template_cumprod_dimname_out():
    x = _named_matrix()
    out = torch.empty_like(x)
    return [x, "C"], {"out": out}


def _template_cumprod_dimname_inplace():
    x = _named_matrix()
    return [x.clone(), "C"], {}


def _template_cumsum_dimname():
    x = _named_matrix()
    return [x, "C"], {}


def _template_cumsum_dimname_out():
    x = _named_matrix()
    out = torch.empty_like(x)
    return [x, "C"], {"out": out}


def _template_cumsum_dimname_inplace():
    x = _named_matrix()
    return [x.clone(), "C"], {}


def _template_cumprod_default():
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    return [x, 0], {}


def _template_cumprod_out():
    args, _ = _template_cumprod_default()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_cumprod_inplace():
    args, _ = _template_cumprod_default()
    args[0] = args[0].clone()
    return args, {}


def _template_cumsum_default():
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    return [x, 0], {}


def _template_cumsum_out():
    args, _ = _template_cumsum_default()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_cumsum_inplace():
    args, _ = _template_cumsum_default()
    args[0] = args[0].clone()
    return args, {}


def _template_ceil_scalar():
    return [0.5], {}


def _template_cos_scalar():
    return [0.5], {}


def _template_cosh_scalar():
    return [0.5], {}


def _template_copysign_default():
    return [1.0, -2.0], {}


def _template_copysign_scalar():
    x = torch.tensor([1.0, -2.0], dtype=torch.float32)
    return [x, -1.0], {}


def _template_copysign_scalar_out():
    args, _ = _template_copysign_scalar()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_copysign_scalar_inplace():
    args, _ = _template_copysign_scalar()
    args[0] = args[0].clone()
    return args, {}


def _template_clamp_min():
    x = torch.tensor([-1.0, 0.5, 2.0], dtype=torch.float32)
    return [x, 0.0], {}


def _template_clamp_min_out():
    args, _ = _template_clamp_min()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_clamp_min_inplace():
    args, _ = _template_clamp_min()
    args[0] = args[0].clone()
    return args, {}


def _template_clamp_max():
    x = torch.tensor([-1.0, 0.5, 2.0], dtype=torch.float32)
    return [x, 1.0], {}


def _template_clamp_max_out():
    args, _ = _template_clamp_max()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_clamp_max_inplace():
    args, _ = _template_clamp_max()
    args[0] = args[0].clone()
    return args, {}


def _template_clamp_scalar():
    x = torch.tensor([-1.0, 0.5, 2.0], dtype=torch.float32)
    return [x, 0.0, 1.0], {}


def _template_clamp_scalar_out():
    args, _ = _template_clamp_scalar()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_clamp_scalar_inplace():
    x = torch.tensor([-1.0, 0.5, 2.0], dtype=torch.float32)
    return [x, 0.0, 1.0], {}


def _template_clamp_tensor():
    x = torch.tensor([-1.0, 0.5, 2.0], dtype=torch.float32)
    min_t = torch.tensor(0.0)
    max_t = torch.tensor(1.0)
    return [x, min_t, max_t], {}


def _template_clamp_tensor_out():
    args, _ = _template_clamp_tensor()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_clamp_tensor_inplace():
    x = torch.tensor([-1.0, 0.5, 2.0], dtype=torch.float32)
    min_t = torch.tensor(0.0)
    max_t = torch.tensor(1.0)
    return [x, min_t, max_t], {}


def _template_copy_list():
    return [[1, 2, 3]], {}


def _template_copy_dict_str():
    return [{"a": 1, "b": 2}], {}


def _template_copy_dict_int():
    return [{1: "a", 2: "b"}], {}


def _template_copy_dict_bool():
    return [{True: 1, False: 0}], {}


def _template_copy_dict_float():
    return [{1.0: 2.0}], {}


def _template_copy_dict_complex():
    return [{1 + 0j: 2}], {}


CUSTOM_TEMPLATES.update(
    {
        "bernoulli_.float": _skip("missing_lowering"),
        "binary_cross_entropy_backward.grad_input": _template_bce_backward_grad_input,
        "bincount.default": _skip("missing_lowering"),
        "bincount.out": _skip("missing_lowering"),
        "block_diag.default": _template_block_diag,
        "block_diag.out": _template_block_diag_out,
        "bmm.default": _template_bmm,
        "bmm.out": _template_bmm_out,
        "bmm.dtype": _skip("unsupported_cpu_dtype"),
        "bmm.dtype_out": _skip("unsupported_cpu_dtype"),
        "broadcast_tensors.default": _template_broadcast_tensors,
        "bucketize.Tensor": _template_bucketize_tensor,
        "bucketize.Tensor_out": _template_bucketize_tensor_out,
        # FakeTensor/meta for bucketize.Scalar currently calls Tensor.item() on meta tensors and crashes.
        "bucketize.Scalar": _skip("dynamo_fake_tensor_bucketize_scalar_bug"),
        "bucketize.Scalar_out": _skip(
            "dynamo_fake_tensor_bucketize_scalar_bug"
        ),
        "cat.default": _template_cat_default,
        "cat.out": _template_cat_out,
        "cat.names": _skip("named_tensor_unsupported"),
        "cat.names_out": _skip("named_tensor_unsupported"),
        "channel_shuffle.default": _template_channel_shuffle,
        "channel_shuffle.out": _template_channel_shuffle_out,
        "cholesky.default": _skip("linalg_not_supported"),
        "cholesky.out": _skip("linalg_not_supported"),
        "cholesky_inverse.default": _skip("linalg_not_supported"),
        "cholesky_inverse.out": _skip("linalg_not_supported"),
        "cholesky_solve.default": _skip("linalg_not_supported"),
        "cholesky_solve.out": _skip("linalg_not_supported"),
        "col2im.default": _template_col2im,
        "col2im.out": _template_col2im_out,
        "constant_pad_nd.default": _template_constant_pad_nd,
        "constant_pad_nd.out": _template_constant_pad_nd_out,
        "complex.default": _skip("complex64_not_supported"),
        "complex.out": _skip("complex64_not_supported"),
        "conv2d.default": _template_conv2d_default,
        "conv2d.padding": _template_conv2d_padding,
        "convolution.default": _template_convolution,
        "convolution.out": _template_convolution_out,
        "convolution_backward.default": _skip("backward_not_supported"),
        "convolution_backward.out": _skip("backward_not_supported"),
        "cauchy.default": _skip("randop_not_supported"),
        "cauchy.out": _skip("randop_not_supported"),
        "cauchy_.default": _skip("randop_not_supported"),
        "count_nonzero.out": _template_count_nonzero_out,
        "count_nonzero.dim_IntList": _template_count_nonzero_dimlist,
        "count_nonzero.dim_IntList_out": _template_count_nonzero_dimlist_out,
        "copy.t": _skip("generic_list_not_supported"),
        "copy.Dict_str": _skip("generic_dict_not_supported"),
        "copy.Dict_int": _skip("generic_dict_not_supported"),
        "copy.Dict_bool": _skip("generic_dict_not_supported"),
        "copy.Dict_float": _skip("generic_dict_not_supported"),
        "copy.Dict_complex": _skip("generic_dict_not_supported"),
        "copy.Dict_Tensor": _skip("tensor_keys_not_hashable"),
        "cummax.default": _template_cummax,
        "cummax.out": _template_cummax_out,
        "cummax.dimname": _skip("named_tensor_unsupported"),
        "cummax.dimname_out": _skip("named_tensor_unsupported"),
        "cummin.default": _template_cummin,
        "cummin.out": _template_cummin_out,
        "cummin.dimname": _skip("named_tensor_unsupported"),
        "cummin.dimname_out": _skip("named_tensor_unsupported"),
        "cumprod.default": _template_cumprod_default,
        "cumprod.out": _template_cumprod_out,
        "cumprod_.default": _template_cumprod_inplace,
        "cumprod.dimname": _skip("named_tensor_unsupported"),
        "cumprod.dimname_out": _skip("named_tensor_unsupported"),
        "cumprod_.dimname": _skip("named_tensor_unsupported"),
        "cumsum.default": _template_cumsum_default,
        "cumsum.out": _template_cumsum_out,
        "cumsum_.default": _template_cumsum_inplace,
        "cumsum.dimname": _skip("named_tensor_unsupported"),
        "cumsum.dimname_out": _skip("named_tensor_unsupported"),
        "cumsum_.dimname": _skip("named_tensor_unsupported"),
        "clamp.default": _template_clamp_scalar,
        "clamp.out": _template_clamp_scalar_out,
        "clamp_.default": _template_clamp_scalar_inplace,
        "clamp.Tensor": _template_clamp_tensor,
        "clamp.Tensor_out": _template_clamp_tensor_out,
        "clamp_.Tensor": _template_clamp_tensor_inplace,
        "clamp_min.default": _template_clamp_min,
        "clamp_min.out": _template_clamp_min_out,
        "clamp_min_.default": _template_clamp_min_inplace,
        "clamp_max.default": _template_clamp_max,
        "clamp_max.out": _template_clamp_max_out,
        "clamp_max_.default": _template_clamp_max_inplace,
        "clip.default": _template_clamp_scalar,
        "clip.out": _template_clamp_scalar_out,
        "clip_.default": _template_clamp_scalar_inplace,
        "clip.Tensor": _template_clamp_tensor,
        "clip.Tensor_out": _template_clamp_tensor_out,
        "clip_.Tensor": _template_clamp_tensor_inplace,
        "ceil.Scalar": _template_ceil_scalar,
        "cos.Scalar": _template_cos_scalar,
        "cosh.Scalar": _template_cosh_scalar,
        "copysign.default": _template_copysign_default,
        "copysign.Scalar": _template_copysign_scalar,
        "copysign.Scalar_out": _template_copysign_scalar_out,
        "copysign_.Scalar": _template_copysign_scalar_inplace,
        # Complex types not supported
        "cos.complex": _skip("complex_not_supported"),
        "cosh.complex": _skip("complex_not_supported"),
        # Sparse tensor attribute query
        "dense_dim.default": _skip("sparse_not_supported"),
    }
)

for name in [
    "bitwise_and.Tensor",
    "bitwise_or.Tensor",
    "bitwise_xor.Tensor",
    "bitwise_left_shift.Tensor",
    "bitwise_right_shift.Tensor",
]:
    CUSTOM_TEMPLATES[name] = _bitwise_tensor_tensor

for name in [
    "bitwise_and.Tensor_out",
    "bitwise_or.Tensor_out",
    "bitwise_xor.Tensor_out",
    "bitwise_left_shift.Tensor_out",
    "bitwise_right_shift.Tensor_out",
]:
    CUSTOM_TEMPLATES[name] = _bitwise_tensor_tensor_out

for name in [
    "bitwise_and.Scalar",
    "bitwise_or.Scalar",
    "bitwise_xor.Scalar",
    "bitwise_left_shift.Tensor_Scalar",
    "bitwise_right_shift.Tensor_Scalar",
]:
    CUSTOM_TEMPLATES[name] = _bitwise_tensor_scalar

for name in [
    "bitwise_and.Scalar_out",
    "bitwise_or.Scalar_out",
    "bitwise_xor.Scalar_out",
    "bitwise_left_shift.Tensor_Scalar_out",
    "bitwise_right_shift.Tensor_Scalar_out",
]:
    CUSTOM_TEMPLATES[name] = _bitwise_tensor_scalar_out

for name in [
    "bitwise_and.Scalar_Tensor",
    "bitwise_or.Scalar_Tensor",
    "bitwise_xor.Scalar_Tensor",
    "bitwise_left_shift.Scalar_Tensor",
    "bitwise_right_shift.Scalar_Tensor",
]:
    CUSTOM_TEMPLATES[name] = _bitwise_scalar_tensor

for name in [
    "bitwise_and.Scalar_Tensor_out",
    "bitwise_or.Scalar_Tensor_out",
    "bitwise_xor.Scalar_Tensor_out",
    "bitwise_left_shift.Scalar_Tensor_out",
    "bitwise_right_shift.Scalar_Tensor_out",
]:
    CUSTOM_TEMPLATES[name] = _bitwise_scalar_tensor_out

for name in [
    "bitwise_and_.Tensor",
    "bitwise_or_.Tensor",
    "bitwise_xor_.Tensor",
    "bitwise_left_shift_.Tensor",
    "bitwise_right_shift_.Tensor",
]:
    CUSTOM_TEMPLATES[name] = _bitwise_inplace_tensor_tensor

for name in [
    "bitwise_and_.Scalar",
    "bitwise_or_.Scalar",
    "bitwise_xor_.Scalar",
    "bitwise_left_shift_.Tensor_Scalar",
    "bitwise_right_shift_.Tensor_Scalar",
]:
    CUSTOM_TEMPLATES[name] = _bitwise_inplace_tensor_scalar

CUSTOM_TEMPLATES.update(
    {
        "bitwise_not.default": _bitwise_not,
        "bitwise_not.out": _bitwise_not_out,
        "bitwise_not_.default": _bitwise_not_inplace,
    }
)

# Edit the OPS list to add/remove ops in this batch (format: "op.overload").
OPS = [
    "bernoulli_.float",
    "binary_cross_entropy.default",
    "binary_cross_entropy.out",
    "binary_cross_entropy_backward.default",
    "binary_cross_entropy_backward.grad_input",
    "binary_cross_entropy_with_logits.default",
    "binary_cross_entropy_with_logits.out",
    "bincount.default",
    "bincount.out",
    "bitwise_and.Tensor",
    "bitwise_and.Scalar",
    "bitwise_and.Scalar_Tensor",
    "bitwise_and.Tensor_out",
    "bitwise_and.Scalar_out",
    "bitwise_and.Scalar_Tensor_out",
    "bitwise_and_.Tensor",
    "bitwise_and_.Scalar",
    "bitwise_left_shift.Tensor",
    "bitwise_left_shift.Tensor_Scalar",
    "bitwise_left_shift.Scalar_Tensor",
    "bitwise_left_shift.Tensor_out",
    "bitwise_left_shift.Tensor_Scalar_out",
    "bitwise_left_shift.Scalar_Tensor_out",
    "bitwise_left_shift_.Tensor_Scalar",
    "bitwise_left_shift_.Tensor",
    "bitwise_not.default",
    "bitwise_not.out",
    "bitwise_not_.default",
    "bitwise_or.Tensor",
    "bitwise_or.Scalar",
    "bitwise_or.Scalar_Tensor",
    "bitwise_or.Tensor_out",
    "bitwise_or.Scalar_out",
    "bitwise_or.Scalar_Tensor_out",
    "bitwise_or_.Tensor",
    "bitwise_or_.Scalar",
    "bitwise_right_shift.Tensor",
    "bitwise_right_shift.Tensor_Scalar",
    "bitwise_right_shift.Scalar_Tensor",
    "bitwise_right_shift.Tensor_out",
    "bitwise_right_shift.Tensor_Scalar_out",
    "bitwise_right_shift.Scalar_Tensor_out",
    "bitwise_right_shift_.Tensor_Scalar",
    "bitwise_right_shift_.Tensor",
    "bitwise_xor.Tensor",
    "bitwise_xor.Scalar",
    "bitwise_xor.Scalar_Tensor",
    "bitwise_xor.Tensor_out",
    "bitwise_xor.Scalar_out",
    "bitwise_xor.Scalar_Tensor_out",
    "bitwise_xor_.Tensor",
    "bitwise_xor_.Scalar",
    "block_diag.default",
    "block_diag.out",
    "bmm.default",
    "bmm.out",
    "bmm.dtype_out",
    "bmm.dtype",
    "broadcast_tensors.default",
    "bucketize.Tensor",
    "bucketize.Scalar",
    "bucketize.Tensor_out",
    "bucketize.Scalar_out",
    "cat.default",
    "cat.names",
    "cat.names_out",
    "cat.out",
    "cauchy.default",
    "cauchy.out",
    "cauchy_.default",
    "ceil.default",
    "ceil.out",
    "ceil.int",
    "ceil.float",
    "ceil.Scalar",
    "ceil_.default",
    "celu.default",
    "celu.out",
    "celu_.default",
    "channel_shuffle.default",
    "channel_shuffle.out",
    "cholesky.default",
    "cholesky.out",
    "cholesky_inverse.default",
    "cholesky_inverse.out",
    "cholesky_solve.default",
    "cholesky_solve.out",
    "clamp.default",
    "clamp.Tensor",
    "clamp.out",
    "clamp.Tensor_out",
    "clamp_.default",
    "clamp_.Tensor",
    "clamp_max.default",
    "clamp_max.Tensor",
    "clamp_max.out",
    "clamp_max.Tensor_out",
    "clamp_max_.default",
    "clamp_max_.Tensor",
    "clamp_min.default",
    "clamp_min.Tensor",
    "clamp_min.out",
    "clamp_min.Tensor_out",
    "clamp_min_.default",
    "clamp_min_.Tensor",
    "clip.default",
    "clip.Tensor",
    "clip.out",
    "clip.Tensor_out",
    "clip_.default",
    "clip_.Tensor",
    "clone.default",
    "clone.out",
    "col2im.default",
    "col2im.out",
    "complex.default",
    "complex.out",
    "conj.default",
    "conj_physical.default",
    "conj_physical.out",
    "conj_physical_.default",
    "constant_pad_nd.default",
    "constant_pad_nd.out",
    "conv2d.default",
    "conv2d.padding",
    "convolution.default",
    "convolution.out",
    "convolution_backward.default",
    "convolution_backward.out",
    "copy.default",
    "copy.out",
    "copy.t",
    "copy.Dict_str",
    "copy.Dict_int",
    "copy.Dict_bool",
    "copy.Dict_float",
    "copy.Dict_complex",
    "copy.Dict_Tensor",
    "copy_.default",
    "copy_.Tensor",
    "copy_.int",
    "copy_.float",
    "copysign.Tensor",
    "copysign.Scalar",
    "copysign.out",
    "copysign.Scalar_out",
    "copysign.int",
    "copysign.float",
    "copysign.int_float",
    "copysign.float_int",
    "copysign.default",
    "copysign_.Tensor",
    "copysign_.Scalar",
    "cos.default",
    "cos.out",
    "cos.int",
    "cos.float",
    "cos.complex",
    "cos.Scalar",
    "cos_.default",
    "cosh.default",
    "cosh.out",
    "cosh.int",
    "cosh.float",
    "cosh.complex",
    "cosh.Scalar",
    "cosh_.default",
    "count_nonzero.dim_IntList",
    "count_nonzero.dim_IntList_out",
    "count_nonzero.default",
    "count_nonzero.out",
    "cudnn_batch_norm.default",
    "cudnn_batch_norm.out",
    "cudnn_batch_norm_backward.default",
    "cudnn_batch_norm_backward.out",
    "cummax.default",
    "cummax.dimname",
    "cummax.dimname_out",
    "cummax.out",
    "cummin.default",
    "cummin.dimname",
    "cummin.dimname_out",
    "cummin.out",
    "cumprod.default",
    "cumprod.dimname",
    "cumprod.dimname_out",
    "cumprod.out",
    "cumprod_.default",
    "cumprod_.dimname",
    "cumsum.default",
    "cumsum.dimname",
    "cumsum.dimname_out",
    "cumsum.out",
    "cumsum_.default",
    "cumsum_.dimname",
    "deg2rad.default",
    "deg2rad.out",
    "deg2rad_.default",
    "dense_dim.default",
    "detach.default",
]

if __name__ == "__main__":
    run_aten_op_batch(
        OPS,
        batch_label="test_batch_1",
        max_fails=20,
        templates=CUSTOM_TEMPLATES,
        show_skips=True,
    )
# CHECK: SUMMARY pass=
# CHECK-SAME: fail=0
