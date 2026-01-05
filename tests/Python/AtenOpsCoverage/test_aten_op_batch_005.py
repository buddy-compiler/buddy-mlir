# RUN: %PYTHON %s 2>&1 | FileCheck %s
from aten_op_batch_runner import run_aten_op_batch
import torch


CUSTOM_TEMPLATES = {}


def _skip(reason: str):
    def fn():
        raise RuntimeError(reason)

    return fn


def _template_max_dim_max():
    x = torch.tensor([[1.0, 2.0], [3.0, 0.5]], dtype=torch.float32)
    dim = 1
    values = torch.empty((x.shape[0],), dtype=x.dtype)
    indices = torch.empty((x.shape[0],), dtype=torch.int64)
    return [x, dim, False], {"max": values, "max_values": indices}


def _template_min_dim_min():
    x = torch.tensor([[1.0, 2.0], [3.0, 0.5]], dtype=torch.float32)
    dim = 1
    values = torch.empty((x.shape[0],), dtype=x.dtype)
    indices = torch.empty((x.shape[0],), dtype=torch.int64)
    return [x, dim, False], {"min": values, "min_indices": indices}


def _template_max_pool2d():
    x = torch.randn(1, 1, 4, 4, dtype=torch.float32)
    kernel = [2, 2]
    stride = [2, 2]
    padding = [0, 0]
    dilation = [1, 1]
    ceil_mode = False
    return [x, kernel, stride, padding, dilation, ceil_mode], {}


def _template_max_pool2d_out():
    args, _ = _template_max_pool2d()
    out = torch.empty((1, 1, 2, 2), dtype=torch.float32)
    indices = torch.empty((1, 1, 2, 2), dtype=torch.int64)
    return args, {"out": out, "indices": indices}


def _template_max_pool2d_backward():
    x = torch.randn(1, 1, 4, 4, dtype=torch.float32)
    kernel = [2, 2]
    stride = [2, 2]
    padding = [0, 0]
    dilation = [1, 1]
    ceil_mode = False
    out, indices = torch.ops.aten.max_pool2d_with_indices(
        x, kernel, stride, padding, dilation, ceil_mode
    )
    grad_out = torch.ones_like(out)
    return [
        grad_out,
        x,
        kernel,
        stride,
        padding,
        dilation,
        ceil_mode,
        indices,
    ], {}


def _template_max_pool2d_backward_out():
    args, _ = _template_max_pool2d_backward()
    grad_input = torch.empty_like(args[1])
    return args, {"grad_input": grad_input}


def _template_max_pool3d():
    x = torch.randn(1, 1, 4, 4, 4, dtype=torch.float32)
    kernel = [2, 2, 2]
    stride = [2, 2, 2]
    padding = [0, 0, 0]
    dilation = [1, 1, 1]
    ceil_mode = False
    return [x, kernel, stride, padding, dilation, ceil_mode], {}


def _template_max_pool3d_out():
    args, _ = _template_max_pool3d()
    out = torch.empty((1, 1, 2, 2, 2), dtype=torch.float32)
    indices = torch.empty((1, 1, 2, 2, 2), dtype=torch.int64)
    return args, {"out": out, "indices": indices}


def _template_max_pool3d_backward():
    x = torch.randn(1, 1, 4, 4, 4, dtype=torch.float32)
    kernel = [2, 2, 2]
    stride = [2, 2, 2]
    padding = [0, 0, 0]
    dilation = [1, 1, 1]
    ceil_mode = False
    out, indices = torch.ops.aten.max_pool3d_with_indices(
        x, kernel, stride, padding, dilation, ceil_mode
    )
    grad_out = torch.ones_like(out)
    return [
        grad_out,
        x,
        kernel,
        stride,
        padding,
        dilation,
        ceil_mode,
        indices,
    ], {}


def _template_max_pool3d_backward_out():
    args, _ = _template_max_pool3d_backward()
    grad_input = torch.empty_like(args[1])
    return args, {"grad_input": grad_input}


def _template_max_unpool2d():
    x = torch.randn(1, 1, 4, 4, dtype=torch.float32)
    kernel = [2, 2]
    stride = [2, 2]
    padding = [0, 0]
    dilation = [1, 1]
    ceil_mode = False
    pooled, indices = torch.ops.aten.max_pool2d_with_indices(
        x, kernel, stride, padding, dilation, ceil_mode
    )
    output_size = list(x.shape[-2:])
    return [pooled, indices, output_size], {}


def _template_max_unpool2d_out():
    args, _ = _template_max_unpool2d()
    out = torch.empty((1, 1, 4, 4), dtype=torch.float32)
    return args, {"out": out}


def _template_max_unpool3d():
    x = torch.randn(1, 1, 4, 4, 4, dtype=torch.float32)
    kernel = [2, 2, 2]
    stride = [2, 2, 2]
    padding = [0, 0, 0]
    dilation = [1, 1, 1]
    ceil_mode = False
    pooled, indices = torch.ops.aten.max_pool3d_with_indices(
        x, kernel, stride, padding, dilation, ceil_mode
    )
    output_size = list(x.shape[-3:])
    return [pooled, indices, output_size, stride, padding], {}


def _template_max_unpool3d_out():
    args, _ = _template_max_unpool3d()
    out = torch.empty((1, 1, 4, 4, 4), dtype=torch.float32)
    return args, {"out": out}


def _template_mean():
    x = torch.randn(2, 3, dtype=torch.float32)
    return [x], {}


def _template_mean_dim():
    x = torch.randn(2, 3, dtype=torch.float32)
    return [x, [1], True], {}


def _template_mean_out():
    args, _ = _template_mean_dim()
    out = torch.empty((args[0].shape[0], 1), dtype=torch.float32)
    return args, {"out": out}


def _template_mean_dtype_out():
    x = torch.randn(2, 3, dtype=torch.float32)
    out = torch.empty((), dtype=torch.float32)
    return [x], {"dtype": torch.float32, "out": out}


def _template_median_dim_values():
    x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    dim = 1
    values = torch.empty((x.shape[0],), dtype=x.dtype)
    indices = torch.empty((x.shape[0],), dtype=torch.int64)
    return [x, dim, False], {"values": values, "indices": indices}


def _template_meshgrid():
    a = torch.tensor([1.0, 2.0], dtype=torch.float32)
    b = torch.tensor([3.0, 4.0], dtype=torch.float32)
    return [[a, b]], {}


def _template_meshgrid_indexing():
    args, _ = _template_meshgrid()
    return args, {"indexing": "ij"}


def _template_mm():
    a = torch.randn(2, 3, dtype=torch.float32)
    b = torch.randn(3, 4, dtype=torch.float32)
    return [a, b], {}


def _template_mm_out():
    args, _ = _template_mm()
    out = torch.empty((args[0].shape[0], args[1].shape[1]), dtype=torch.float32)
    return args, {"out": out}


def _template_matmul_out():
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    b = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    out = torch.empty((2, 4), dtype=torch.float32)
    return [a, b], {"out": out}


def _template_mm_dtype():
    args, _ = _template_mm()
    return [args[0], args[1], torch.float32], {}


def _template_mm_dtype_out():
    args, _ = _template_mm()
    out = torch.empty((args[0].shape[0], args[1].shape[1]), dtype=torch.float32)
    return [args[0], args[1], torch.float32], {"out": out}


def _template_mode_values():
    x = torch.tensor([[1.0, 1.0, 2.0], [3.0, 4.0, 4.0]], dtype=torch.float32)
    dim = 1
    values = torch.empty((x.shape[0],), dtype=x.dtype)
    indices = torch.empty((x.shape[0],), dtype=torch.int64)
    return [x, dim, False], {"values": values, "indices": indices}


def _template_multi_margin_loss():
    self = torch.randn(2, 3, dtype=torch.float32)
    target = torch.tensor([0, 2], dtype=torch.int64)
    return [self, target, 1, 1, None, 1], {}


def _template_multi_margin_loss_out():
    args, _ = _template_multi_margin_loss()
    out = torch.empty((), dtype=torch.float32)
    return args, {"out": out}


def _template_multilabel_margin_loss_forward():
    # torch.compile/Inductor currently treats -1 padding in target differently from eager
    # semantics; it treats -1 as 0 and includes it in loss computation, so numerical
    # comparisons will fail. To avoid flagging upstream differences as Buddy backend issues,
    # use a target without -1 here.
    self = torch.tensor([[0.2, 0.1, 0.3]], dtype=torch.float32)
    target = torch.tensor([[0, 2, 0]], dtype=torch.int64)
    reduction = 1
    return [self, target, reduction], {}


def _template_multilabel_margin_loss_forward_out():
    args, _ = _template_multilabel_margin_loss_forward()
    output = torch.empty((), dtype=torch.float32)
    is_target = torch.empty_like(args[0], dtype=torch.float32)
    return args, {"output": output, "is_target": is_target}


def _template_multinomial():
    probs = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float32)
    return [probs, 2, False], {}


def _template_multinomial_out():
    args, _ = _template_multinomial()
    out = torch.empty((args[1],), dtype=torch.int64)
    return args, {"out": out}


def _template_mv():
    a = torch.randn(2, 3, dtype=torch.float32)
    v = torch.randn(3, dtype=torch.float32)
    return [a, v], {}


def _template_mv_out():
    args, _ = _template_mv()
    out = torch.empty((args[0].shape[0],), dtype=torch.float32)
    return args, {"out": out}


def _template_mvlgamma():
    x = torch.tensor([1.5, 2.0], dtype=torch.float32)
    return [x, 1], {}


def _template_mvlgamma_out():
    args, _ = _template_mvlgamma()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_nansum_dim_values():
    x = torch.tensor([[1.0, float("nan")], [2.0, 3.0]], dtype=torch.float32)
    return [x, 0, False], {}


def _template_nansum_out():
    args, _ = _template_nansum_dim_values()
    out = torch.empty((args[0].shape[1],), dtype=torch.float32)
    return args, {"out": out}


def _template_narrow_tensor():
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    start = torch.tensor(1)
    length = 2
    return [x, 1, start, length], {}


def _template_native_batch_norm():
    x = torch.randn(2, 3, 4, 4, dtype=torch.float32)
    weight = torch.ones(3, dtype=torch.float32)
    bias = torch.zeros(3, dtype=torch.float32)
    running_mean = torch.zeros(3, dtype=torch.float32)
    running_var = torch.ones(3, dtype=torch.float32)
    training = True
    momentum = 0.1
    eps = 1e-5
    return [
        x,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
    ], {}


def _template_native_batch_norm_out():
    args, _ = _template_native_batch_norm()
    c = args[0].shape[1]
    out = torch.empty_like(args[0])
    save_mean = torch.empty((c,), dtype=torch.float32)
    save_invstd = torch.empty((c,), dtype=torch.float32)
    return args, {
        "out": out,
        "save_mean": save_mean,
        "save_invstd": save_invstd,
    }


def _template_native_batch_norm_backward():
    args, _ = _template_native_batch_norm()
    x, weight, bias, running_mean, running_var, training, momentum, eps = args
    out, save_mean, save_invstd = torch.ops.aten.native_batch_norm(
        x, weight, bias, running_mean, running_var, training, momentum, eps
    )
    grad_out = torch.ones_like(out)
    output_mask = [True, True, True]
    return [
        grad_out,
        x,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_invstd,
        training,
        eps,
        output_mask,
    ], {}


def _template_native_batch_norm_backward_out():
    args, _ = _template_native_batch_norm_backward()
    x = args[1]
    weight = args[2]
    out0 = torch.empty_like(x)
    out1 = torch.empty_like(weight)
    out2 = torch.empty_like(weight)
    return args, {"out0": out0, "out1": out1, "out2": out2}


def _template_native_dropout_out():
    x = torch.randn(2, 3, dtype=torch.float32)
    p = 0.5
    train = True
    out0 = torch.empty_like(x)
    out1 = torch.empty_like(x, dtype=torch.bool)
    return [x, p, train], {"out0": out0, "out1": out1}


def _template_native_group_norm():
    x = torch.randn(2, 4, 4, 4, dtype=torch.float32)
    weight = torch.ones(4, dtype=torch.float32)
    bias = torch.zeros(4, dtype=torch.float32)
    N, C = x.shape[0], x.shape[1]
    HxW = x.shape[2] * x.shape[3]
    group = 2
    eps = 1e-5
    return [x, weight, bias, N, C, HxW, group, eps], {}


def _template_native_group_norm_out():
    args, _ = _template_native_group_norm()
    N = args[3]
    group = args[6]
    out0 = torch.empty_like(args[0])
    out1 = torch.empty((N, group), dtype=torch.float32)
    out2 = torch.empty((N, group), dtype=torch.float32)
    return args, {"out0": out0, "out1": out1, "out2": out2}


def _template_native_group_norm_backward():
    args, _ = _template_native_group_norm()
    x, weight, bias, N, C, HxW, group, eps = args
    out, mean, rstd = torch.ops.aten.native_group_norm(
        x, weight, bias, N, C, HxW, group, eps
    )
    grad_out = torch.ones_like(out)
    output_mask = [True, True, True]
    return [grad_out, x, mean, rstd, weight, N, C, HxW, group, output_mask], {}


def _template_native_group_norm_backward_out():
    args, _ = _template_native_group_norm_backward()
    x = args[1]
    weight = args[4]
    out0 = torch.empty_like(x)
    out1 = torch.empty_like(weight)
    out2 = torch.empty_like(weight)
    return args, {"out0": out0, "out1": out1, "out2": out2}


def _template_native_layer_norm():
    x = torch.randn(2, 3, 4, dtype=torch.float32)
    normalized_shape = [4]
    weight = torch.ones(4, dtype=torch.float32)
    bias = torch.zeros(4, dtype=torch.float32)
    eps = 1e-5
    return [x, normalized_shape, weight, bias, eps], {}


def _template_native_layer_norm_out():
    args, _ = _template_native_layer_norm()
    x = args[0]
    out0 = torch.empty_like(x)
    norm_shape = tuple(int(s) for s in args[1])
    stat_shape = x.shape[: -len(norm_shape)] + (1,) * len(norm_shape)
    out1 = torch.empty(stat_shape, dtype=torch.float32)
    out2 = torch.empty(stat_shape, dtype=torch.float32)
    return args, {"out0": out0, "out1": out1, "out2": out2}


def _template_native_layer_norm_backward():
    args, _ = _template_native_layer_norm()
    x, normalized_shape, weight, bias, eps = args
    out, mean, rstd = torch.ops.aten.native_layer_norm(
        x, normalized_shape, weight, bias, eps
    )
    grad_out = torch.ones_like(out)
    output_mask = [True, True, True]
    return [
        grad_out,
        x,
        normalized_shape,
        mean,
        rstd,
        weight,
        bias,
        output_mask,
    ], {}


def _template_native_layer_norm_backward_out():
    args, _ = _template_native_layer_norm_backward()
    x = args[1]
    weight = args[5]
    out0 = torch.empty_like(x)
    out1 = torch.empty_like(weight)
    out2 = torch.empty_like(weight)
    return args, {"out0": out0, "out1": out1, "out2": out2}


def _template_ne_int_list():
    return [[1, 2], [2, 3]], {}


def _template_ne_float_list():
    return [[1.0, 2.0], [2.0, 3.0]], {}


def _template_ne_bool_list():
    return [[True, False], [True, True]], {}


def _template_ne_tensor_list():
    a = [torch.tensor([1.0]), torch.tensor([2.0])]
    b = [torch.tensor([1.0]), torch.tensor([3.0])]
    return [a, b], {}


def _template_mul_scalar():
    x = torch.tensor([1.0, 2.0], dtype=torch.float32)
    return [x, 2.0], {}


def _template_mul_scalar_out():
    args, _ = _template_mul_scalar()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_mul_scalar_inplace():
    args, _ = _template_mul_scalar()
    return args, {}


def _template_mul_default_scalar_pair():
    return [2.0, 3.0], {}


def _template_neg_scalar():
    return [1.5], {}


def _template_ne_scalar():
    x = torch.tensor([1.0, 2.0], dtype=torch.float32)
    return [x, 1.5], {}


def _template_ne_scalar_out():
    args, _ = _template_ne_scalar()
    out = torch.empty_like(args[0], dtype=torch.bool)
    return args, {"out": out}


def _template_ne_scalar_inplace():
    args, _ = _template_ne_scalar()
    return args, {}


def _template_ne_default():
    return [1.0, 2.0], {}


def _template_ne_str():
    return ["foo", "bar"], {}


def _template_ne_str_list():
    return [["foo", "bar"], ["foo", "baz"]], {}


def _template_new_full():
    self = torch.tensor([1.0], dtype=torch.float32)
    size = [2, 3]
    fill_value = 1.5
    return [self, size, fill_value], {}


def _template_new_size_ops():
    self = torch.tensor([1.0], dtype=torch.float32)
    size = [2, 3]
    return self, size


def _template_new_empty():
    self, size = _template_new_size_ops()
    return [self, size], {}


def _template_new_empty_out():
    args, _ = _template_new_empty()
    out = torch.empty((2, 3), dtype=args[0].dtype)
    return args, {"out": out}


def _template_new_empty_strided():
    self, size = _template_new_size_ops()
    stride = [3, 1]
    return [self, size, stride], {}


def _template_new_empty_strided_out():
    args, _ = _template_new_empty_strided()
    out = torch.empty((2, 3), dtype=args[0].dtype)
    return args, {"out": out}


def _template_new_ones():
    self, size = _template_new_size_ops()
    return [self, size], {}


def _template_new_ones_out():
    args, _ = _template_new_ones()
    out = torch.empty((2, 3), dtype=args[0].dtype)
    return args, {"out": out}


def _template_new_zeros():
    self, size = _template_new_size_ops()
    return [self, size], {}


def _template_new_zeros_out():
    args, _ = _template_new_zeros()
    out = torch.empty((2, 3), dtype=args[0].dtype)
    return args, {"out": out}


def _template_nll_loss_base():
    self = torch.randn(2, 3, dtype=torch.float32)
    target = torch.tensor([0, 2], dtype=torch.int64)
    weight = None
    reduction = 1
    ignore_index = -100
    return self, target, weight, reduction, ignore_index


def _template_nll_loss():
    self, target, weight, reduction, ignore_index = _template_nll_loss_base()
    return [self, target, weight, reduction, ignore_index], {}


def _template_nll_loss_out():
    args, _ = _template_nll_loss()
    out = torch.empty((), dtype=torch.float32)
    return args, {"out": out}


def _template_nll_loss_forward():
    self, target, weight, reduction, ignore_index = _template_nll_loss_base()
    return [self, target, weight, reduction, ignore_index], {}


def _template_nll_loss_forward_out():
    args, _ = _template_nll_loss_forward()
    output = torch.empty((), dtype=torch.float32)
    total_weight = torch.empty((), dtype=torch.float32)
    return args, {"output": output, "total_weight": total_weight}


def _template_nll_loss_backward():
    self, target, weight, reduction, ignore_index = _template_nll_loss_base()
    output, total_weight = torch.ops.aten.nll_loss_forward(
        self, target, weight, reduction, ignore_index
    )
    grad_output = torch.ones_like(output)
    return [
        grad_output,
        self,
        target,
        weight,
        reduction,
        ignore_index,
        total_weight,
    ], {}


def _template_nll_loss_backward_out():
    args, _ = _template_nll_loss_backward()
    grad_input = torch.empty_like(args[1])
    return args, {"grad_input": grad_input}


def _template_nll_loss2d_base():
    self = torch.randn(1, 3, 2, 2, dtype=torch.float32)
    target = torch.tensor([[[0, 1], [2, 0]]], dtype=torch.int64)
    weight = None
    reduction = 1
    ignore_index = -100
    return self, target, weight, reduction, ignore_index


def _template_nll_loss2d_forward():
    self, target, weight, reduction, ignore_index = _template_nll_loss2d_base()
    return [self, target, weight, reduction, ignore_index], {}


def _template_nll_loss2d_forward_out():
    args, _ = _template_nll_loss2d_forward()
    output = torch.empty((), dtype=torch.float32)
    total_weight = torch.empty((), dtype=torch.float32)
    return args, {"output": output, "total_weight": total_weight}


def _template_nll_loss2d_backward():
    self, target, weight, reduction, ignore_index = _template_nll_loss2d_base()
    output, total_weight = torch.ops.aten.nll_loss2d_forward(
        self, target, weight, reduction, ignore_index
    )
    grad_output = torch.ones_like(output)
    return [
        grad_output,
        self,
        target,
        weight,
        reduction,
        ignore_index,
        total_weight,
    ], {}


def _template_nll_loss2d_backward_out():
    args, _ = _template_nll_loss2d_backward()
    grad_input = torch.empty_like(args[1])
    return args, {"grad_input": grad_input}


# Register custom templates or skips.
CUSTOM_TEMPLATES.update(
    {
        # Dynamic shape ops: output shape depends on input values, not just shapes.
        # TorchDynamo cannot compile these statically.
        "masked_select.default": _skip("dynamic_shape_op"),
        "masked_select.out": _skip("dynamic_shape_op"),
        "max.dim_max": _template_max_dim_max,
        "max.names_dim": _skip("named_tensor_torchscript"),
        "max.names_dim_max": _skip("named_tensor_torchscript"),
        "min.dim_min": _template_min_dim_min,
        "min.names_dim": _skip("named_tensor_torchscript"),
        "min.names_dim_min": _skip("named_tensor_torchscript"),
        "max_pool2d_with_indices.default": _template_max_pool2d,
        "max_pool2d_with_indices.out": _template_max_pool2d_out,
        "max_pool2d_with_indices_backward.default": _template_max_pool2d_backward,
        "max_pool2d_with_indices_backward.grad_input": _template_max_pool2d_backward_out,
        "max_pool3d_with_indices.default": _template_max_pool3d,
        "max_pool3d_with_indices.out": _template_max_pool3d_out,
        # Backward op: out of scope for this operator-coverage batch.
        "max_pool3d_with_indices_backward.default": _skip(
            "backward_not_supported"
        ),
        "max_pool3d_with_indices_backward.grad_input": _skip(
            "backward_not_supported"
        ),
        "max_unpool2d.default": _template_max_unpool2d,
        "max_unpool2d.out": _template_max_unpool2d_out,
        "max_unpool3d.default": _template_max_unpool3d,
        "max_unpool3d.out": _template_max_unpool3d_out,
        "mean.default": _template_mean,
        "mean.dim": _template_mean_dim,
        "mean.names_dim": _skip("named_tensor_torchscript"),
        "mean.names_out": _skip("named_tensor_torchscript"),
        "mean.out": _template_mean_out,
        "mean.dtype_out": _template_mean_dtype_out,
        "median.dim_values": _template_median_dim_values,
        "median.names_dim": _skip("named_tensor_torchscript"),
        "median.names_dim_values": _skip("named_tensor_torchscript"),
        "nanmedian.dim_values": _template_median_dim_values,
        "nanmedian.names_dim": _skip("named_tensor_torchscript"),
        "nanmedian.names_dim_values": _skip("named_tensor_torchscript"),
        "meshgrid.default": _template_meshgrid,
        "meshgrid.indexing": _template_meshgrid_indexing,
        "mm.default": _template_mm,
        "mm.out": _template_mm_out,
        "mm.dtype": _skip("cpu_backend_missing_mm_dtype"),
        "mm.dtype_out": _skip("cpu_backend_missing_mm_dtype"),
        "matmul.out": _template_matmul_out,
        "mode.values": _template_mode_values,
        "mode.dimname": _skip("named_tensor_torchscript"),
        "mode.dimname_out": _skip("named_tensor_torchscript"),
        "multi_margin_loss.default": _template_multi_margin_loss,
        "multi_margin_loss.out": _template_multi_margin_loss_out,
        "multilabel_margin_loss_forward.default": _template_multilabel_margin_loss_forward,
        "multilabel_margin_loss_forward.output": _template_multilabel_margin_loss_forward_out,
        # Random op: skip to avoid nondeterministic behavior.
        "multinomial.default": _skip("random_op_not_supported"),
        "multinomial.out": _skip("random_op_not_supported"),
        "mv.default": _template_mv,
        "mv.out": _template_mv_out,
        "mvlgamma.default": _template_mvlgamma,
        "mvlgamma.out": _template_mvlgamma_out,
        "mvlgamma_.default": _template_mvlgamma,
        "nansum.default": _template_nansum_dim_values,
        "nansum.out": _template_nansum_out,
        "narrow.Tensor": _template_narrow_tensor,
        "native_batch_norm.default": _template_native_batch_norm,
        "native_batch_norm.out": _template_native_batch_norm_out,
        "native_batch_norm_backward.default": _template_native_batch_norm_backward,
        "native_batch_norm_backward.out": _template_native_batch_norm_backward_out,
        "native_dropout.out": _template_native_dropout_out,
        "native_group_norm.default": _template_native_group_norm,
        "native_group_norm.out": _template_native_group_norm_out,
        "native_group_norm_backward.default": _template_native_group_norm_backward,
        "native_group_norm_backward.out": _template_native_group_norm_backward_out,
        "native_layer_norm.default": _template_native_layer_norm,
        "native_layer_norm.out": _template_native_layer_norm_out,
        "native_layer_norm_backward.default": _template_native_layer_norm_backward,
        "native_layer_norm_backward.out": _template_native_layer_norm_backward_out,
        "ne.int_list": _template_ne_int_list,
        "ne.float_list": _template_ne_float_list,
        "ne.Tensor_list": _template_ne_tensor_list,
        "ne.bool_list": _template_ne_bool_list,
        "ne.Scalar": _template_ne_scalar,
        "ne.Scalar_out": _template_ne_scalar_out,
        "ne_.Scalar": _template_ne_scalar_inplace,
        "ne.default": _template_ne_default,
        "ne.str": _template_ne_str,
        "ne.str_list": _template_ne_str_list,
        "ne.enum": _skip("any_enum_not_supported"),
        "mul.Scalar": _template_mul_scalar,
        "mul.Scalar_out": _template_mul_scalar_out,
        "mul_.Scalar": _template_mul_scalar_inplace,
        "mul.default": _template_mul_default_scalar_pair,
        "mul.left_t": _skip("generic_list_op_unsupported"),
        "mul.right_": _skip("generic_list_op_unsupported"),
        "mul_.t": _skip("generic_list_op_unsupported"),
        "multiply.Scalar": _template_mul_scalar,
        "multiply_.Scalar": _template_mul_scalar_inplace,
        "neg.Scalar": _template_neg_scalar,
        "new_full.default": _template_new_full,
        "new_full.out": _skip("dynamo_out_overload_bug"),
        "new_empty.default": _template_new_empty,
        "new_empty.out": _skip("dynamo_out_overload_bug"),
        "new_empty_strided.default": _template_new_empty_strided,
        "new_empty_strided.out": _skip("dynamo_out_overload_bug"),
        "new_ones.default": _template_new_ones,
        "new_ones.out": _skip("dynamo_out_overload_bug"),
        "new_zeros.default": _template_new_zeros,
        "new_zeros.out": _skip("dynamo_out_overload_bug"),
        "nll_loss.default": _template_nll_loss,
        "nll_loss.out": _template_nll_loss_out,
        "nll_loss_forward.default": _template_nll_loss_forward,
        "nll_loss_forward.output": _template_nll_loss_forward_out,
        "nll_loss_backward.default": _template_nll_loss_backward,
        "nll_loss_backward.grad_input": _template_nll_loss_backward_out,
        "nll_loss2d_forward.default": _template_nll_loss2d_forward,
        "nll_loss2d_forward.output": _template_nll_loss2d_forward_out,
        "nll_loss2d_backward.default": _template_nll_loss2d_backward,
        "nll_loss2d_backward.grad_input": _template_nll_loss2d_backward_out,
        # Dynamic shape ops: output shape depends on input values
        "nonzero.default": _skip("dynamic_shape_op"),
        "nonzero.out": _skip("dynamic_shape_op"),
        "nonzero_numpy.default": _skip("dynamic_shape_op"),
        "miopen_batch_norm.default": _skip("backend_specific_miopen"),
        "miopen_batch_norm.out": _skip("backend_specific_miopen"),
        "miopen_batch_norm_backward.default": _skip("backend_specific_miopen"),
        "miopen_batch_norm_backward.out": _skip("backend_specific_miopen"),
        "mkldnn_rnn_layer.default": _skip("backend_specific_mkldnn"),
        "mkldnn_rnn_layer.out": _skip("backend_specific_mkldnn"),
        "mkldnn_rnn_layer_backward.default": _skip("backend_specific_mkldnn"),
        "mkldnn_rnn_layer_backward.out": _skip("backend_specific_mkldnn"),
    }
)

# Edit the OPS list to add/remove ops in this batch (format: "op.overload").
OPS = [
    "masked_select.default",
    "masked_select.out",
    "matmul.default",
    "matmul.out",
    "max.other",
    "max.default",
    "max.dim",
    "max.dim_max",
    "max.names_dim",
    "max.names_dim_max",
    "max.unary_out",
    "max.out",
    "max_pool2d_with_indices.default",
    "max_pool2d_with_indices.out",
    "max_pool2d_with_indices_backward.default",
    "max_pool2d_with_indices_backward.grad_input",
    "max_pool3d_with_indices.default",
    "max_pool3d_with_indices.out",
    "max_pool3d_with_indices_backward.default",
    "max_pool3d_with_indices_backward.grad_input",
    "max_unpool2d.default",
    "max_unpool2d.out",
    "max_unpool3d.default",
    "max_unpool3d.out",
    "maximum.default",
    "maximum.out",
    "mean.default",
    "mean.dim",
    "mean.names_dim",
    "mean.names_out",
    "mean.out",
    "mean.dtype_out",
    "median.default",
    "median.dim",
    "median.dim_values",
    "median.names_dim",
    "median.names_dim_values",
    "median.out",
    "meshgrid.default",
    "meshgrid.indexing",
    "min.other",
    "min.default",
    "min.dim",
    "min.dim_min",
    "min.names_dim",
    "min.names_dim_min",
    "min.unary_out",
    "min.out",
    "minimum.default",
    "minimum.out",
    "miopen_batch_norm.default",
    "miopen_batch_norm.out",
    "miopen_batch_norm_backward.default",
    "miopen_batch_norm_backward.out",
    "mish.default",
    "mish.out",
    "mish_.default",
    "mish_backward.default",
    "mkldnn_rnn_layer.default",
    "mkldnn_rnn_layer.out",
    "mkldnn_rnn_layer_backward.default",
    "mkldnn_rnn_layer_backward.out",
    "mm.default",
    "mm.out",
    "mm.dtype_out",
    "mm.dtype",
    "mode.default",
    "mode.dimname",
    "mode.dimname_out",
    "mode.values",
    "mse_loss.default",
    "mse_loss.out",
    "mse_loss_backward.default",
    "mse_loss_backward.grad_input",
    "mul.Tensor",
    "mul.Scalar",
    "mul.out",
    "mul.Scalar_out",
    "mul.left_t",
    "mul.right_",
    "mul.int",
    "mul.complex",
    "mul.float",
    "mul.int_complex",
    "mul.complex_int",
    "mul.float_complex",
    "mul.complex_float",
    "mul.int_float",
    "mul.float_int",
    "mul.default",
    "mul_.Tensor",
    "mul_.Scalar",
    "mul_.t",
    "multi_margin_loss.default",
    "multi_margin_loss.out",
    "multilabel_margin_loss_forward.default",
    "multilabel_margin_loss_forward.output",
    "multinomial.default",
    "multinomial.out",
    "multiply.Tensor",
    "multiply.Scalar",
    "multiply.out",
    "multiply_.Tensor",
    "multiply_.Scalar",
    "mv.default",
    "mv.out",
    "mvlgamma.default",
    "mvlgamma.out",
    "mvlgamma_.default",
    "nan_to_num.default",
    "nan_to_num.out",
    "nan_to_num_.default",
    "nanmedian.default",
    "nanmedian.dim",
    "nanmedian.dim_values",
    "nanmedian.names_dim",
    "nanmedian.names_dim_values",
    "nanmedian.out",
    "nansum.default",
    "nansum.out",
    "narrow.default",
    "narrow.Tensor",
    "narrow_copy.default",
    "narrow_copy.out",
    "native_batch_norm.default",
    "native_batch_norm.out",
    "native_batch_norm_backward.default",
    "native_batch_norm_backward.out",
    "native_dropout.default",
    "native_dropout.out",
    "native_dropout_backward.default",
    "native_dropout_backward.out",
    "native_group_norm.default",
    "native_group_norm.out",
    "native_group_norm_backward.default",
    "native_group_norm_backward.out",
    "native_layer_norm.default",
    "native_layer_norm.out",
    "native_layer_norm_backward.default",
    "native_layer_norm_backward.out",
    "ne.Tensor",
    "ne.Scalar",
    "ne.Scalar_out",
    "ne.Tensor_out",
    "ne.int_list",
    "ne.device",
    "ne.bool",
    "ne.enum",
    "ne.int",
    "ne.complex",
    "ne.float",
    "ne.int_float",
    "ne.float_int",
    "ne.float_complex",
    "ne.complex_float",
    "ne.default",
    "ne.str",
    "ne.float_list",
    "ne.Tensor_list",
    "ne.bool_list",
    "ne.str_list",
    "ne_.Scalar",
    "ne_.Tensor",
    "neg.default",
    "neg.out",
    "neg.int",
    "neg.float",
    "neg.complex",
    "neg.Scalar",
    "neg_.default",
    "negative.default",
    "negative.out",
    "negative_.default",
    "new_empty.default",
    "new_empty.out",
    "new_empty_strided.default",
    "new_empty_strided.out",
    "new_full.default",
    "new_full.out",
    "new_ones.default",
    "new_ones.out",
    "new_zeros.default",
    "new_zeros.out",
    "nextafter.default",
    "nextafter.out",
    "nextafter_.default",
    "nll_loss.default",
    "nll_loss.out",
    "nll_loss2d_backward.default",
    "nll_loss2d_backward.grad_input",
    "nll_loss2d_forward.default",
    "nll_loss2d_forward.output",
    "nll_loss_backward.default",
    "nll_loss_backward.grad_input",
    "nll_loss_forward.default",
    "nll_loss_forward.output",
    "nonzero.default",
    "nonzero.out",
    "nonzero_numpy.default",
    "nonzero_static.default",
]

if __name__ == "__main__":
    run_aten_op_batch(
        OPS,
        batch_label="test_batch_5",
        max_fails=20,
        templates=CUSTOM_TEMPLATES,
    )
# CHECK: SUMMARY pass=
# CHECK-SAME: fail=0
