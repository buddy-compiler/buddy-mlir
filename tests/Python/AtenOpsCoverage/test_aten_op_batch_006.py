# RUN: %PYTHON %s 2>&1 | FileCheck %s
from aten_op_batch_runner import run_aten_op_batch
import torch


CUSTOM_TEMPLATES = {}


def _skip(reason: str):
    def fn():
        raise RuntimeError(reason)

    return fn


# Edit the OPS list to add/remove ops in this batch (format: "op.overload").
OPS = [
    "nonzero_static.out",
    "norm.Scalar",
    "norm.ScalarOpt_dim",
    "norm.names_ScalarOpt_dim",
    "norm.ScalarOpt_dim_dtype",
    "norm.dtype_out",
    "norm.out",
    "norm.ScalarOpt_dtype",
    "norm.ScalarOpt_dtype_out",
    "norm.Scalar_out",
    "norm.names_ScalarOpt_dim_dtype",
    "norm.names_dtype_out",
    "norm.names_out",
    "normal.Tensor_float",
    "normal.Tensor_float_out",
    "normal.float_Tensor_out",
    "normal.float_Tensor",
    "normal.Tensor_Tensor",
    "normal.Tensor_Tensor_out",
    "normal.float_float",
    "normal.float_float_out",
    "normal.out",
    "normal_.default",
    "not_equal.Tensor",
    "not_equal.Scalar",
    "not_equal.Scalar_out",
    "not_equal.Tensor_out",
    "not_equal_.Scalar",
    "not_equal_.Tensor",
    "numel.default",
    "ones.names",
    "ones.default",
    "ones.names_out",
    "ones.out",
    "ones_like.default",
    "ones_like.out",
    "ormqr.default",
    "ormqr.out",
    "pad_sequence.default",
    "pairwise_distance.default",
    "pdist.default",
    "permute.default",
    "permute_copy.default",
    "permute_copy.out",
    "pin_memory.default",
    "pixel_shuffle.default",
    "pixel_shuffle.out",
    "pixel_unshuffle.default",
    "pixel_unshuffle.out",
    "poisson.default",
    "poisson.out",
    "polar.default",
    "polar.out",
    "polar.int",
    "polar.float",
    "polar.int_float",
    "polar.float_int",
    "polar.Scalar_Scalar",
    "polygamma.default",
    "polygamma.out",
    "positive.default",
    "pow.Tensor_Tensor",
    "pow.Tensor_Scalar",
    "pow.Scalar",
    "pow.Scalar_out",
    "pow.Tensor_Scalar_out",
    "pow.Tensor_Tensor_out",
    "pow.int",
    "pow.complex",
    "pow.float",
    "pow.int_float",
    "pow.float_int",
    "pow.float_complex",
    "pow.complex_float",
    "pow.Scalar_Scalar",
    "pow.int_to_int",
    "pow_.Scalar",
    "pow_.Tensor",
    "prelu.default",
    "prod.default",
    "prod.dim_int",
    "prod.dim_Dimname",
    "prod.Dimname_out",
    "prod.int_out",
    "prod.out",
    "quantized_gru.input",
    "quantized_gru.data",
    "quantized_gru.input_legacy",
    "quantized_gru.data_legacy",
    "quantized_lstm.input",
    "quantized_lstm.data",
    "quantized_lstm.input_legacy",
    "quantized_lstm.data_legacy",
    "rad2deg.default",
    "rad2deg.out",
    "rad2deg_.default",
    "rand.default",
    "rand.generator",
    "rand.names",
    "rand.generator_with_names",
    "rand.out",
    "rand.generator_out",
    "rand.names_out",
    "rand.generator_with_names_out",
    "rand_like.default",
    "rand_like.out",
    "randint.default",
    "randint.generator",
    "randint.low",
    "randint.low_generator",
    "randint.out",
    "randint.generator_out",
    "randint.low_out",
    "randint.low_generator_out",
    "randint_like.default",
    "randint_like.low_dtype",
    "randint_like.out",
    "randint_like.Tensor",
    "randint_like.Tensor_out",
    "randint_like.low_dtype_out",
    "randn.default",
    "randn.generator",
    "randn.names",
    "randn.generator_with_names",
    "randn.out",
    "randn.generator_out",
    "randn.names_out",
    "randn.generator_with_names_out",
    "randn_like.default",
    "randn_like.out",
    "randperm.default",
    "randperm.generator",
    "randperm.out",
    "randperm.generator_out",
    "real.default",
    "reciprocal.default",
    "reciprocal.out",
    "reciprocal_.default",
    "reflection_pad1d.default",
    "reflection_pad1d.out",
    "reflection_pad1d_backward.default",
    "reflection_pad1d_backward.grad_input",
    "reflection_pad2d.default",
    "reflection_pad2d.out",
    "reflection_pad2d_backward.default",
    "reflection_pad2d_backward.grad_input",
    "reflection_pad3d.default",
    "reflection_pad3d.out",
    "reflection_pad3d_backward.default",
    "reflection_pad3d_backward.grad_input",
    "relu.default",
    "relu.out",
    "relu6.default",
    "relu_.default",
    "remainder.Tensor",
    "remainder.Scalar",
    "remainder.Scalar_Tensor",
    "remainder.Tensor_out",
    "remainder.Scalar_out",
    "remainder.Scalar_Tensor_out",
    "remainder.int",
    "remainder.float",
    "remainder.int_float",
    "remainder.float_int",
    "remainder.default",
    "remainder_.Tensor",
    "remainder_.Scalar",
    "renorm.default",
    "renorm.out",
    "renorm_.default",
    "repeat.default",
    "repeat.out",
    "repeat_interleave.Tensor",
    "repeat_interleave.self_Tensor",
    "repeat_interleave.self_int",
    "repeat_interleave.Tensor_out",
    "replication_pad1d.default",
    "replication_pad1d.out",
    "replication_pad1d_backward.default",
    "replication_pad1d_backward.grad_input",
    "replication_pad2d.default",
    "replication_pad2d.out",
    "replication_pad2d_backward.default",
    "replication_pad2d_backward.grad_input",
    "replication_pad3d.default",
    "replication_pad3d.out",
    "replication_pad3d_backward.default",
    "replication_pad3d_backward.grad_input",
    "reshape.default",
    "resize.default",
    "resize.out",
    "resize_as.default",
    "resize_as.out",
    "resize_as_.default",
    "rnn_relu.input",
    "rnn_relu.data",
    "rnn_tanh.input",
    "rnn_tanh.data",
    "roll.default",
    "roll.out",
]


def _template_ones_default():
    size = [2, 3]
    return [size], {"dtype": torch.float32}


def _template_ones_out():
    args, _ = _template_ones_default()
    out = torch.empty(args[0], dtype=torch.float32)
    return args, {"out": out}


def _template_norm_scalaropt_dim():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    p = 2.0
    dim = [1]
    keepdim = False
    return [x, p, dim, keepdim], {}


def _template_norm_scalaropt_dim_dtype():
    args, _ = _template_norm_scalaropt_dim()
    return args, {"dtype": torch.float32}


def _template_norm_out():
    args, _ = _template_norm_scalaropt_dim()
    out = torch.empty((args[0].shape[0],), dtype=torch.float32)
    return args, {"out": out}


def _template_norm_dtype_out():
    args, kwargs = _template_norm_out()
    kwargs["dtype"] = torch.float32
    return args, kwargs


def _template_norm_scalaropt_dtype():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    p = 2.0
    return [x, p], {"dtype": torch.float32}


def _template_norm_scalaropt_dtype_out():
    args, kwargs = _template_norm_scalaropt_dtype()
    out = torch.empty((), dtype=torch.float32)
    kwargs["out"] = out
    return args, kwargs


def _template_not_equal_scalar():
    x = torch.tensor([1.0, 2.0], dtype=torch.float32)
    other = 1.5
    return [x, other], {}


def _template_not_equal_scalar_out():
    args, _ = _template_not_equal_scalar()
    out = torch.empty_like(args[0], dtype=torch.bool)
    return args, {"out": out}


def _template_not_equal_inplace_scalar():
    args, _ = _template_not_equal_scalar()
    return args, {}


def _template_pow_tensor_scalar():
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    exponent = 2.0
    return [x, exponent], {}


def _template_pow_tensor_scalar_out():
    args, _ = _template_pow_tensor_scalar()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_pow_scalar():
    self_val = 2.0
    exponent = torch.tensor([1.0, 2.0], dtype=torch.float32)
    return [self_val, exponent], {}


def _template_pow_scalar_out():
    args, _ = _template_pow_scalar()
    out = torch.empty_like(args[1])
    return args, {"out": out}


def _template_pow_scalar_scalar():
    return [2.0, 3.0], {}


def _template_pow_inplace_scalar():
    x = torch.tensor([1.0, 2.0], dtype=torch.float32)
    exponent = 2.0
    return [x, exponent], {}


def _template_remainder_scalar():
    x = torch.tensor([5.0, 6.0, 7.0], dtype=torch.float32)
    other = 2.0
    return [x, other], {}


def _template_remainder_scalar_out():
    args, _ = _template_remainder_scalar()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_remainder_scalar_tensor():
    self_val = 5.0
    other = torch.tensor([2.0, 3.0], dtype=torch.float32)
    return [self_val, other], {}


def _template_remainder_scalar_tensor_out():
    args, _ = _template_remainder_scalar_tensor()
    out = torch.empty_like(args[1])
    return args, {"out": out}


def _template_remainder_default():
    return [5.5, 2.0], {}


def _template_remainder_inplace_scalar():
    args, _ = _template_remainder_scalar()
    return args, {}


def _template_renorm_default():
    x = torch.randn(2, 3, dtype=torch.float32)
    p = 2.0
    dim = 0
    maxnorm = 1.5
    return [x, p, dim, maxnorm], {}


def _template_renorm_out():
    args, _ = _template_renorm_default()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_renorm_inplace():
    args, _ = _template_renorm_default()
    return args, {}


def _template_permute_default():
    x = torch.randn(2, 3, 4, dtype=torch.float32)
    dims = [2, 0, 1]
    return [x, dims], {}


def _template_permute_copy_default():
    return _template_permute_default()


def _template_permute_copy_out():
    args, _ = _template_permute_default()
    x, dims = args
    out = torch.empty(x.permute(dims).shape, dtype=x.dtype)
    return args, {"out": out}


def _template_pixel_shuffle():
    x = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2)
    upscale_factor = 2
    return [x, upscale_factor], {}


def _template_pixel_shuffle_out():
    args, _ = _template_pixel_shuffle()
    out = torch.empty((1, 1, 4, 4), dtype=torch.float32)
    return args, {"out": out}


def _template_pixel_unshuffle():
    x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    downscale_factor = 2
    return [x, downscale_factor], {}


def _template_pixel_unshuffle_out():
    args, _ = _template_pixel_unshuffle()
    out = torch.empty((1, 4, 2, 2), dtype=torch.float32)
    return args, {"out": out}


def _template_pdist():
    x = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float32)
    p = 2.0
    return [x, p], {}


def _template_pairwise_distance():
    x1 = torch.randn(2, 3, dtype=torch.float32)
    x2 = torch.randn(2, 3, dtype=torch.float32)
    return [x1, x2, 2.0, 1e-6, False], {}


def _template_prod_int_out():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    dim = 1
    keepdim = False
    out = torch.empty((x.shape[0],), dtype=x.dtype)
    return [x, dim, keepdim], {"out": out}


def _template_prod_out():
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    out = torch.empty((), dtype=x.dtype)
    return [x], {"out": out}


def _template_repeat_default():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    repeats = [2, 3]
    return [x, repeats], {}


def _template_repeat_out():
    args, _ = _template_repeat_default()
    x, repeats = args
    out_shape = [x.shape[i] * repeats[i] for i in range(len(repeats))]
    out = torch.empty(out_shape, dtype=x.dtype)
    return args, {"out": out}


def _template_repeat_interleave_tensor():
    repeats = torch.tensor([1, 3, 2], dtype=torch.int64)
    output_size = 6
    return [repeats], {"output_size": output_size}


def _template_repeat_interleave_tensor_out():
    args, kwargs = _template_repeat_interleave_tensor()
    out = torch.empty((kwargs["output_size"],), dtype=torch.int64)
    return args, {**kwargs, "out": out}


def _template_repeat_interleave_self_tensor():
    self_tensor = torch.tensor([10, 20, 30], dtype=torch.int64)
    repeats = torch.tensor([1, 3, 2], dtype=torch.int64)
    dim = 0
    output_size = 6
    return [self_tensor, repeats, dim], {"output_size": output_size}


def _template_reshape_default():
    x = torch.arange(6, dtype=torch.float32)
    shape = [2, 3]
    return [x, shape], {}


def _template_resize_default():
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    size = [3, 2]
    return [x, size], {}


def _template_roll_default():
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    shifts = [1]
    dims = [1]
    return [x, shifts, dims], {}


def _template_roll_out():
    args, _ = _template_roll_default()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_remainder_float_int():
    return [5.5, 2], {}


def _template_randint_default():
    high = 5
    size = [2, 3]
    return [high, size], {"dtype": torch.int64}


def _template_randint_generator():
    args, _ = _template_randint_default()
    g = torch.Generator().manual_seed(0)
    return args, {"generator": g, "dtype": torch.int64}


def _template_randint_low():
    low, high = 0, 5
    size = [2, 3]
    return [low, high, size], {"dtype": torch.int64}


def _template_randint_low_generator():
    args, _ = _template_randint_low()
    g = torch.Generator().manual_seed(0)
    return args, {"generator": g, "dtype": torch.int64}


def _template_randint_out():
    args, _ = _template_randint_default()
    out = torch.empty(args[1], dtype=torch.int64)
    return args, {"out": out}


def _template_randint_generator_out():
    args, _ = _template_randint_default()
    g = torch.Generator().manual_seed(0)
    out = torch.empty(args[1], dtype=torch.int64)
    return args, {"generator": g, "out": out}


def _template_randint_low_out():
    args, _ = _template_randint_low()
    out = torch.empty(args[2], dtype=torch.int64)
    return args, {"out": out}


def _template_randint_low_generator_out():
    args, _ = _template_randint_low()
    g = torch.Generator().manual_seed(0)
    out = torch.empty(args[2], dtype=torch.int64)
    return args, {"generator": g, "out": out}


def _template_randint_like_default():
    self = torch.zeros((2, 3), dtype=torch.int64)
    high = 5
    return [self, high], {"dtype": torch.int64}


def _template_randint_like_out():
    args, _ = _template_randint_like_default()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_randint_like_low_dtype():
    self = torch.zeros((2, 3), dtype=torch.int64)
    low, high = 0, 5
    return [self, low, high], {"dtype": torch.int64}


def _template_randint_like_low_dtype_out():
    args, _ = _template_randint_like_low_dtype()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_randint_like_tensor():
    self = torch.zeros((2, 3), dtype=torch.int64)
    high = torch.tensor(5)
    return [self, high], {"dtype": torch.int64}


def _template_randint_like_tensor_out():
    args, _ = _template_randint_like_tensor()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_randperm_default():
    n = 6
    return [n], {"dtype": torch.int64}


def _template_randperm_generator():
    args, _ = _template_randperm_default()
    g = torch.Generator().manual_seed(0)
    return args, {"generator": g, "dtype": torch.int64}


def _template_randperm_out():
    args, _ = _template_randperm_default()
    n = args[0]
    out = torch.empty((n,), dtype=torch.int64)
    return args, {"out": out}


def _template_randperm_generator_out():
    args, _ = _template_randperm_default()
    g = torch.Generator().manual_seed(0)
    n = args[0]
    out = torch.empty((n,), dtype=torch.int64)
    return args, {"generator": g, "out": out}


def _template_nonzero_static_out():
    x = torch.tensor([[0.0, 1.0, 0.0], [2.0, 0.0, 3.0]], dtype=torch.float32)
    size = 5
    out = torch.empty((size, x.dim()), dtype=torch.int64)
    return [x], {"size": size, "fill_value": -1, "out": out}


def _template_ormqr_default():
    torch.manual_seed(0)
    inp = torch.randn(3, 2, dtype=torch.float32)
    tau = torch.randn(2, dtype=torch.float32)
    other = torch.randn(3, 2, dtype=torch.float32)
    return [inp, tau, other, True, False], {}


def _template_ormqr_out():
    args, _ = _template_ormqr_default()
    out = torch.empty_like(args[2])
    return args, {"out": out}


def _template_reflection_pad1d_default():
    x = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    padding = [2, 1]
    return [x, padding], {}


def _template_reflection_pad1d_out():
    args, _ = _template_reflection_pad1d_default()
    ref = torch.ops.aten.reflection_pad1d.default(*args)
    out = torch.empty_like(ref)
    return args, {"out": out}


def _template_reflection_pad2d_default():
    x = torch.arange(1 * 1 * 3 * 3, dtype=torch.float32).reshape(1, 1, 3, 3)
    padding = [1, 1, 1, 0]
    return [x, padding], {}


def _template_reflection_pad2d_out():
    args, _ = _template_reflection_pad2d_default()
    ref = torch.ops.aten.reflection_pad2d.default(*args)
    out = torch.empty_like(ref)
    return args, {"out": out}


def _template_reflection_pad3d_default():
    x = torch.arange(1 * 1 * 3 * 3 * 3, dtype=torch.float32).reshape(
        1, 1, 3, 3, 3
    )
    padding = [1, 0, 1, 0, 1, 0]
    return [x, padding], {}


def _template_reflection_pad3d_out():
    args, _ = _template_reflection_pad3d_default()
    ref = torch.ops.aten.reflection_pad3d.default(*args)
    out = torch.empty_like(ref)
    return args, {"out": out}


def _template_replication_pad1d_default():
    x = torch.arange(6, dtype=torch.float32).reshape(1, 1, 6)
    padding = [2, 3]
    return [x, padding], {}


def _template_replication_pad1d_out():
    args, _ = _template_replication_pad1d_default()
    ref = torch.ops.aten.replication_pad1d.default(*args)
    out = torch.empty_like(ref)
    return args, {"out": out}


def _template_replication_pad2d_default():
    x = torch.arange(1 * 1 * 3 * 4, dtype=torch.float32).reshape(1, 1, 3, 4)
    padding = [1, 2, 0, 1]
    return [x, padding], {}


def _template_replication_pad2d_out():
    args, _ = _template_replication_pad2d_default()
    ref = torch.ops.aten.replication_pad2d.default(*args)
    out = torch.empty_like(ref)
    return args, {"out": out}


def _template_replication_pad3d_default():
    x = torch.arange(1 * 1 * 2 * 2 * 2, dtype=torch.float32).reshape(
        1, 1, 2, 2, 2
    )
    padding = [1, 0, 1, 0, 1, 0]
    return [x, padding], {}


def _template_replication_pad3d_out():
    args, _ = _template_replication_pad3d_default()
    ref = torch.ops.aten.replication_pad3d.default(*args)
    out = torch.empty_like(ref)
    return args, {"out": out}


# Register custom templates or skips.
CUSTOM_TEMPLATES.update(
    {
        "nonzero_static.out": _template_nonzero_static_out,
        "norm.ScalarOpt_dim": _template_norm_scalaropt_dim,
        "norm.ScalarOpt_dim_dtype": _template_norm_scalaropt_dim_dtype,
        "norm.dtype_out": _template_norm_dtype_out,
        "norm.out": _template_norm_out,
        "norm.ScalarOpt_dtype": _template_norm_scalaropt_dtype,
        "norm.ScalarOpt_dtype_out": _template_norm_scalaropt_dtype_out,
        "norm.names_ScalarOpt_dim": _skip("named_tensor_torchscript"),
        "norm.names_ScalarOpt_dim_dtype": _skip("named_tensor_torchscript"),
        "norm.names_dtype_out": _skip("named_tensor_torchscript"),
        "norm.names_out": _skip("named_tensor_torchscript"),
        "normal.Tensor_float": _skip("random_op_not_supported"),
        "normal.Tensor_float_out": _skip("random_op_not_supported"),
        "normal.float_Tensor_out": _skip("random_op_not_supported"),
        "normal.float_Tensor": _skip("random_op_not_supported"),
        "normal.Tensor_Tensor": _skip("random_op_not_supported"),
        "normal.Tensor_Tensor_out": _skip("random_op_not_supported"),
        "normal.float_float": _skip("random_op_not_supported"),
        "normal.float_float_out": _skip("random_op_not_supported"),
        "normal.out": _skip("random_op_not_supported"),
        "normal_.default": _skip("random_op_not_supported"),
        "not_equal.Scalar": _template_not_equal_scalar,
        "not_equal.Scalar_out": _template_not_equal_scalar_out,
        "not_equal_.Scalar": _template_not_equal_inplace_scalar,
        "ones.names": _skip("named_tensor_torchscript"),
        "ones.default": _template_ones_default,
        "ones.names_out": _skip("named_tensor_torchscript"),
        "ones.out": _template_ones_out,
        "ones_like.out": _skip("dynamo_fake_tensor_out_variant_nyi"),
        # linalg/QR-related op: not supported by Buddy backend yet.
        "ormqr.default": _skip("linalg_not_supported"),
        "ormqr.out": _skip("linalg_not_supported"),
        "pad_sequence.default": _skip("backend_missing_pad_sequence"),
        "pairwise_distance.default": _template_pairwise_distance,
        "pdist.default": _template_pdist,
        "permute.default": _template_permute_default,
        "permute_copy.default": _template_permute_copy_default,
        "permute_copy.out": _template_permute_copy_out,
        "pin_memory.default": _skip("dynamo_fake_tensor_nyi_pin_memory"),
        "pixel_shuffle.default": _template_pixel_shuffle,
        "pixel_shuffle.out": _template_pixel_shuffle_out,
        "pixel_unshuffle.default": _template_pixel_unshuffle,
        "pixel_unshuffle.out": _template_pixel_unshuffle_out,
        "poisson.default": _skip("random_op_not_supported"),
        "poisson.out": _skip("random_op_not_supported"),
        "polar.default": _skip("complex_dtype_not_supported"),
        "polar.out": _skip("complex_dtype_not_supported"),
        "polar.int": _skip("complex_dtype_not_supported"),
        "polar.float": _skip("complex_dtype_not_supported"),
        "polar.int_float": _skip("complex_dtype_not_supported"),
        "polar.float_int": _skip("complex_dtype_not_supported"),
        "polar.Scalar_Scalar": _skip("complex_dtype_not_supported"),
        "polygamma.default": _skip("backend_missing_polygamma"),
        "polygamma.out": _skip("backend_missing_polygamma"),
        "pow.Tensor_Scalar": _template_pow_tensor_scalar,
        "pow.Tensor_Scalar_out": _template_pow_tensor_scalar_out,
        "pow.Scalar": _template_pow_scalar,
        "pow.Scalar_out": _template_pow_scalar_out,
        "pow.Scalar_Scalar": _template_pow_scalar_scalar,
        "pow_.Scalar": _template_pow_inplace_scalar,
        "prod.int_out": _template_prod_int_out,
        "prod.out": _template_prod_out,
        "prod.dim_Dimname": _skip("named_tensor_torchscript"),
        "prod.Dimname_out": _skip("named_tensor_torchscript"),
        "rand.default": _skip("backend_missing_rand"),
        "rand.generator": _skip("backend_missing_rand"),
        "rand.names": _skip("backend_missing_rand"),
        "rand.generator_with_names": _skip("backend_missing_rand"),
        "rand.out": _skip("backend_missing_rand"),
        "rand.generator_out": _skip("backend_missing_rand"),
        "rand.names_out": _skip("backend_missing_rand"),
        "rand.generator_with_names_out": _skip("backend_missing_rand"),
        "rand_like.default": _skip("backend_missing_rand"),
        "rand_like.out": _skip("backend_missing_rand"),
        "randint.default": _template_randint_default,
        "randint.generator": _skip("backend_ignores_generator_rng"),
        "randint.low": _template_randint_low,
        "randint.low_generator": _skip("backend_ignores_generator_rng"),
        "randint.out": _template_randint_out,
        "randint.generator_out": _skip("backend_ignores_generator_rng"),
        "randint.low_out": _template_randint_low_out,
        "randint.low_generator_out": _skip("backend_ignores_generator_rng"),
        "randint_like.default": _template_randint_like_default,
        "randint_like.low_dtype": _template_randint_like_low_dtype,
        "randint_like.out": _skip("dynamo_fake_tensor_out_variant_nyi"),
        "randint_like.Tensor": _skip("random_op_not_supported"),
        "randint_like.Tensor_out": _skip("dynamo_fake_tensor_out_variant_nyi"),
        "randint_like.low_dtype_out": _skip(
            "dynamo_fake_tensor_out_variant_nyi"
        ),
        "randn.default": _skip("backend_missing_randn"),
        "randn.generator": _skip("backend_missing_randn"),
        "randn.names": _skip("backend_missing_randn"),
        "randn.generator_with_names": _skip("backend_missing_randn"),
        "randn.out": _skip("backend_missing_randn"),
        "randn.generator_out": _skip("backend_missing_randn"),
        "randn.names_out": _skip("backend_missing_randn"),
        "randn.generator_with_names_out": _skip("backend_missing_randn"),
        "randn_like.default": _skip("backend_missing_randn"),
        "randn_like.out": _skip("backend_missing_randn"),
        "randperm.default": _template_randperm_default,
        "randperm.generator": _skip("backend_ignores_generator_rng"),
        "randperm.out": _template_randperm_out,
        "randperm.generator_out": _skip("backend_ignores_generator_rng"),
        "reflection_pad1d.default": _template_reflection_pad1d_default,
        "reflection_pad1d.out": _template_reflection_pad1d_out,
        "reflection_pad1d_backward.default": _skip(
            "backend_missing_reflection_pad"
        ),
        "reflection_pad1d_backward.grad_input": _skip(
            "backend_missing_reflection_pad"
        ),
        "reflection_pad2d.default": _template_reflection_pad2d_default,
        "reflection_pad2d.out": _template_reflection_pad2d_out,
        "reflection_pad2d_backward.default": _skip(
            "backend_missing_reflection_pad"
        ),
        "reflection_pad2d_backward.grad_input": _skip(
            "backend_missing_reflection_pad"
        ),
        "reflection_pad3d.default": _template_reflection_pad3d_default,
        "reflection_pad3d.out": _template_reflection_pad3d_out,
        "reflection_pad3d_backward.default": _skip(
            "backend_missing_reflection_pad"
        ),
        "reflection_pad3d_backward.grad_input": _skip(
            "backend_missing_reflection_pad"
        ),
        "remainder.Scalar": _template_remainder_scalar,
        "remainder.Scalar_Tensor": _template_remainder_scalar_tensor,
        "remainder.Scalar_out": _template_remainder_scalar_out,
        "remainder.Scalar_Tensor_out": _template_remainder_scalar_tensor_out,
        "remainder.int": _skip("backend_fpe_remainder_int"),
        "remainder.float_int": _template_remainder_float_int,
        "remainder.default": _template_remainder_default,
        "remainder_.Scalar": _template_remainder_inplace_scalar,
        "renorm.default": _template_renorm_default,
        "renorm.out": _template_renorm_out,
        "renorm_.default": _template_renorm_inplace,
        "repeat.default": _template_repeat_default,
        "repeat.out": _template_repeat_out,
        "repeat_interleave.Tensor": _template_repeat_interleave_tensor,
        "repeat_interleave.self_Tensor": _template_repeat_interleave_self_tensor,
        "repeat_interleave.Tensor_out": _template_repeat_interleave_tensor_out,
        "replication_pad1d.default": _template_replication_pad1d_default,
        "replication_pad1d.out": _template_replication_pad1d_out,
        "replication_pad1d_backward.default": _skip(
            "backend_missing_replication_pad"
        ),
        "replication_pad1d_backward.grad_input": _skip(
            "backend_missing_replication_pad"
        ),
        "replication_pad2d.default": _template_replication_pad2d_default,
        "replication_pad2d.out": _template_replication_pad2d_out,
        "replication_pad2d_backward.default": _skip(
            "backend_missing_replication_pad"
        ),
        "replication_pad2d_backward.grad_input": _skip(
            "backend_missing_replication_pad"
        ),
        "replication_pad3d.default": _template_replication_pad3d_default,
        "replication_pad3d.out": _template_replication_pad3d_out,
        "replication_pad3d_backward.default": _skip(
            "backend_missing_replication_pad"
        ),
        "replication_pad3d_backward.grad_input": _skip(
            "backend_missing_replication_pad"
        ),
        "reshape.default": _template_reshape_default,
        "resize.default": _template_resize_default,
        "resize.out": _skip("functionalization_alias_nyi_resize_out"),
        # FakeTensor decomposition for resize_as(out=...) forwards an unexpected
        # out kwarg and errors before compilation; skip to avoid noisy logs.
        "resize_as.out": _skip("dynamo_fake_tensor_resize_as_out_decomp_bug"),
        "rnn_relu.input": _skip("backend_missing_rnn"),
        "rnn_relu.data": _skip("backend_missing_rnn"),
        "rnn_tanh.input": _skip("backend_missing_rnn"),
        "rnn_tanh.data": _skip("backend_missing_rnn"),
        "roll.default": _template_roll_default,
        "roll.out": _template_roll_out,
    }
)

if __name__ == "__main__":
    run_aten_op_batch(
        OPS,
        batch_label="test_batch_6",
        max_fails=20,
        templates=CUSTOM_TEMPLATES,
    )
# CHECK: SUMMARY pass=
# CHECK-SAME: fail=0
