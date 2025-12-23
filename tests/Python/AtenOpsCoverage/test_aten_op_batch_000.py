# RUN: %PYTHON %s 2>&1 | FileCheck %s
from aten_op_batch_runner import run_aten_op_batch
import torch

# Define custom input templates for this batch to allow per-op tuning.
CUSTOM_TEMPLATES = {}


def _template_addmm():
    bias = torch.randn(2, 4, dtype=torch.float32)
    mat1 = torch.randn(2, 3, dtype=torch.float32)
    mat2 = torch.randn(3, 4, dtype=torch.float32)
    return [bias, mat1, mat2], {}


def _template_addmv():
    bias = torch.randn(2, dtype=torch.float32)
    mat = torch.randn(2, 3, dtype=torch.float32)
    vec = torch.randn(3, dtype=torch.float32)
    return [bias, mat, vec], {}


def _template_addbmm2d():
    bias = torch.randn(3, 5, dtype=torch.float32)
    batch1 = torch.randn(1, 3, 4, dtype=torch.float32)
    batch2 = torch.randn(1, 4, 5, dtype=torch.float32)
    return [bias, batch1, batch2], {}


def _template_baddbmm():
    bias = torch.randn(2, 3, 5, dtype=torch.float32)
    batch1 = torch.randn(2, 3, 4, dtype=torch.float32)
    batch2 = torch.randn(2, 4, 5, dtype=torch.float32)
    return [bias, batch1, batch2], {}


def _template_adaptive_max_pool2d_backward_grad_input():
    x = torch.randn(1, 1, 4, 4, dtype=torch.float32)
    grad_output = torch.randn(1, 1, 2, 2, dtype=torch.float32)
    indices = torch.zeros(1, 1, 2, 2, dtype=torch.int64)
    grad_input = torch.empty_like(x)
    return [grad_output, x, indices], {"grad_input": grad_input}


def _template_adaptive_max_pool3d_backward_grad_input():
    x = torch.randn(1, 1, 4, 4, 4, dtype=torch.float32)
    grad_output = torch.randn(1, 1, 2, 2, 2, dtype=torch.float32)
    indices = torch.zeros(1, 1, 2, 2, 2, dtype=torch.int64)
    grad_input = torch.empty_like(x)
    return [grad_output, x, indices], {"grad_input": grad_input}


def _template_adaptive_max_pool2d_default():
    x = torch.randn(1, 1, 4, 4, dtype=torch.float32)
    return [x, (2, 2)], {}


def _template_adaptive_max_pool2d_backward_default():
    x = torch.randn(1, 1, 4, 4, dtype=torch.float32)
    grad_output = torch.randn(1, 1, 2, 2, dtype=torch.float32)
    indices = torch.zeros(1, 1, 2, 2, dtype=torch.int64)
    return [grad_output, x, indices], {}


def _template_adaptive_max_pool3d_default():
    x = torch.randn(1, 1, 4, 4, 4, dtype=torch.float32)
    return [x, (2, 2, 2)], {}


def _template_adaptive_max_pool3d_backward_default():
    x = torch.randn(1, 1, 4, 4, 4, dtype=torch.float32)
    grad_output = torch.randn(1, 1, 2, 2, 2, dtype=torch.float32)
    indices = torch.zeros(1, 1, 2, 2, 2, dtype=torch.int64)
    return [grad_output, x, indices], {}


def _template_all_any_out_dim():
    inp = torch.tensor([[True, False], [True, True]])
    dim = 0
    out = torch.empty((2,), dtype=torch.bool)
    return [inp, dim, False], {"out": out}


def _template_all_any_out_nodim():
    inp = torch.tensor([[True, False], [True, True]])
    out = torch.empty((), dtype=torch.bool)
    return [inp], {"out": out}


def _template_aminmax_out_scalar():
    x = torch.tensor([1.0], dtype=torch.float32)
    min_out = torch.tensor(1.0, dtype=torch.float32)
    max_out = torch.tensor(1.0, dtype=torch.float32)
    return [x], {"min": min_out, "max": max_out}


def _wrap_out_dtype(base_template):
    def fn():
        args, kwargs = base_template()
        kwargs = dict(kwargs)
        kwargs.setdefault("out_dtype", torch.float32)
        return args, kwargs

    return fn


def _wrap_out_tensor(base_template):
    def fn():
        args, kwargs = base_template()
        kwargs = dict(kwargs)
        # Assume bias/self provides the base shape for out.
        out_shape = args[0].shape if args else ()
        kwargs["out"] = torch.empty(out_shape, dtype=torch.float32)
        return args, kwargs

    return fn


def _skip(reason: str):
    def fn():
        raise RuntimeError(reason)

    return fn


def _template_inplace_unary_safe(domain: str = "unit"):
    if domain == "asin":
        x = torch.tensor([0.5], dtype=torch.float32)
    elif domain == "acosh":
        x = torch.tensor([1.5], dtype=torch.float32)
    elif domain == "atanh":
        x = torch.tensor([0.2], dtype=torch.float32)
    else:
        x = torch.tensor([0.5], dtype=torch.float32)
    return [x], {}


def _template_affine_grid():
    theta = torch.eye(2, 3).unsqueeze(0)
    size = [1, 1, 4, 4]
    return [theta, size, False], {}


def _template_affine_grid_out():
    args, _ = _template_affine_grid()
    out = torch.empty((1, 4, 4, 2), dtype=torch.float32)
    return args, {"out": out}


def _template_arg_reduce_out(kind: str):
    x = torch.randn(2, 3)
    out = torch.empty(3 if kind == "argmax" else 3, dtype=torch.long)
    return [x, 0, False], {"out": out}


def _template_all_any_dims(is_all: bool):
    x = torch.tensor([[True, False], [True, True]])
    dim = [0]
    if is_all:
        return [x, dim, False], {}
    return [x, dim, False], {}


def _template_all_any_scalar_list(vtype: str):
    if vtype == "int":
        arr = [0, 1]
    elif vtype == "float":
        arr = [0.0, 1.0]
    else:
        arr = [False, True]
    return [arr], {}


def _template_as_strided():
    x = torch.arange(6, dtype=torch.float32)
    size = [2, 3]
    stride = [3, 1]
    return [x, size, stride, 0], {}


def _template_as_strided_inplace_safe():
    # torch.compile currently triggers an internal ShapeEnv guard IndexError for
    # as_strided_ with multi-dim size/stride.
    # Use a 1D identity shape to avoid a Dynamo entry crash while keeping
    # as_strided_ coverage.
    x = torch.arange(6, dtype=torch.float32)
    size = [6]
    stride = [1]
    return [x, size, stride, 0], {}


def _template_as_strided_out():
    args, _ = _template_as_strided()
    out = torch.empty((2, 3), dtype=torch.float32)
    return args, {"out": out}


def _template_pool_out(kind: str):
    if kind == "2d":
        x = torch.randn(1, 1, 4, 4)
        out = torch.empty((1, 1, 2, 2))
        indices = torch.empty((1, 1, 2, 2), dtype=torch.int64)
        return [x, (2, 2)], {"out": out, "indices": indices}
    x = torch.randn(1, 1, 4, 4, 4)
    out = torch.empty((1, 1, 2, 2, 2))
    indices = torch.empty((1, 1, 2, 2, 2), dtype=torch.int64)
    return [x, (2, 2, 2)], {"out": out, "indices": indices}


def _template_avg_pool2d():
    x = torch.randn(1, 1, 4, 4)
    return [x, (2, 2), (2, 2), (0, 0), False, True], {}


def _template_avg_pool2d_out():
    args, _ = _template_avg_pool2d()
    out = torch.empty((1, 1, 2, 2))
    return args, {"out": out}


def _template_avg_pool2d_backward():
    x = torch.randn(1, 1, 4, 4)
    grad_output = torch.randn(1, 1, 2, 2)
    return [grad_output, x, (2, 2), (2, 2), (0, 0), False, True, None], {}


def _template_avg_pool2d_backward_out():
    args, _ = _template_avg_pool2d_backward()
    grad_input = torch.empty((1, 1, 4, 4))
    return args, {"grad_input": grad_input}


def _template_avg_pool3d():
    x = torch.randn(1, 1, 4, 4, 4)
    return [x, (2, 2, 2), (2, 2, 2), (0, 0, 0), False, True], {}


def _template_avg_pool3d_out():
    args, _ = _template_avg_pool3d()
    out = torch.empty((1, 1, 2, 2, 2))
    return args, {"out": out}


def _template_avg_pool3d_backward():
    x = torch.randn(1, 1, 4, 4, 4)
    grad_output = torch.randn(1, 1, 2, 2, 2)
    return [
        grad_output,
        x,
        (2, 2, 2),
        (2, 2, 2),
        (0, 0, 0),
        False,
        True,
        None,
    ], {}


def _template_avg_pool3d_backward_out():
    args, _ = _template_avg_pool3d_backward()
    grad_input = torch.empty((1, 1, 4, 4, 4))
    return args, {"grad_input": grad_input}


def _template_batch_norm_backward():
    x = torch.randn(2, 3, 4, 4)
    grad_out = torch.randn_like(x)
    weight = torch.ones(3)
    running_mean = torch.zeros(3)
    running_var = torch.ones(3)
    save_mean = torch.zeros(3)
    save_var = torch.ones(3)
    reserve = torch.empty(0)
    output_mask = [True, True, True]
    return [
        grad_out,
        x,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_var,
        True,
        1e-5,
        output_mask,
        reserve,
    ], {}


def _template_add_scalar(out: bool = False, inplace: bool = False):
    self = torch.ones(2)
    other = 2.0
    kwargs = {}
    if out:
        kwargs["out"] = torch.empty_like(self)
    return [self, other], kwargs


def _template_add_tensor(out: bool = False, inplace: bool = False):
    a = torch.randn(2, 3)
    b = torch.randn(2, 3)
    kwargs = {}
    if out:
        kwargs["out"] = torch.empty_like(a)
    return [a, b], kwargs


def _template_add_str():
    return ["foo", "bar"], {}


def _template_add_list():
    return [[1, 2], [3, 4]], {}


def _template_add_t():
    a = [torch.tensor(1.0), torch.tensor(2.0)]
    b = [torch.tensor(3.0), torch.tensor(4.0)]
    return [a, b], {}


def _template_scalar_unary(name: str):
    val = 0.5
    if name in ("acosh",):
        val = 1.5
    return [val], {}


def _template_scalar_binary():
    return [1.0, 2.0], {}


def _template_any_str():
    return [["", "nonempty"]], {}


def _template_arange(kind: str):
    if kind == "default":
        return [5], {}
    if kind == "start":
        return [1, 5], {}
    if kind == "start_step":
        return [1, 5, 2], {}
    return [5], {}


def _template_arange_out(kind: str):
    args, _ = _template_arange(kind)
    out = torch.empty(
        (
            (
                (args[1] - args[0]) // (args[2] if len(args) > 2 else 1)
                if len(args) > 1
                else args[0]
            ),
        ),
        dtype=torch.int64,
    )
    return args, {"out": out}


def _template_dimname(is_all: bool):
    x = torch.randn(2, 3).refine_names("N", "C")
    dim = ["N"]
    return [x, dim, False], {}


def _template_dimname_out(is_all: bool):
    args, _ = _template_dimname(is_all)
    out = torch.empty((2,), dtype=torch.bool).refine_names("C")
    return args, {"out": out}


# Populate ops that need special inputs.
CUSTOM_TEMPLATES.update(
    {
        "addmm.default": _template_addmm,
        "addmm.out": _wrap_out_tensor(_template_addmm),
        "addmm.dtype": _skip("unsupported_cpu_dtype"),
        "addmm.dtype_out": _skip("unsupported_cpu_dtype"),
        "addmv.default": _template_addmv,
        "addmv.out": _wrap_out_tensor(_template_addmv),
        "addbmm.default": _template_addbmm2d,
        "addbmm.out": _wrap_out_tensor(_template_addbmm2d),
        "addbmm.dtype": _wrap_out_dtype(_template_addbmm2d),
        "addbmm.dtype_out": _wrap_out_dtype(
            _wrap_out_tensor(_template_addbmm2d)
        ),
        "addbmm_.default": _template_addbmm2d,
        "baddbmm.default": _template_baddbmm,
        "baddbmm.out": _wrap_out_tensor(_template_baddbmm),
        "baddbmm.dtype": _skip("unsupported_cpu_dtype"),
        "baddbmm.dtype_out": _skip("unsupported_cpu_dtype"),
        "baddbmm_.default": _template_baddbmm,
        "aminmax.out": _template_aminmax_out_scalar,
        "addmm_.default": _template_addmm,
        "addmv_.default": _template_addmv,
        "adaptive_max_pool2d_backward.grad_input": _template_adaptive_max_pool2d_backward_grad_input,
        "adaptive_max_pool3d_backward.grad_input": _template_adaptive_max_pool3d_backward_grad_input,
        "all.out": _template_all_any_out_dim,
        "all.all_out": _template_all_any_out_nodim,
        "any.out": _template_all_any_out_dim,
        "any.all_out": _template_all_any_out_nodim,
        "arcsin_.default": lambda: _template_inplace_unary_safe("asin"),
        "arctanh_.default": lambda: _template_inplace_unary_safe("atanh"),
        "asin_.default": lambda: _template_inplace_unary_safe("asin"),
        "atanh_.default": lambda: _template_inplace_unary_safe("atanh"),
        "acos_.default": lambda: _template_inplace_unary_safe("asin"),
        "acosh_.default": lambda: _template_inplace_unary_safe("acosh"),
        "arccos_.default": lambda: _template_inplace_unary_safe("asin"),
        "arccosh_.default": lambda: _template_inplace_unary_safe("acosh"),
        "acosh.int": lambda: ([2], {}),
        "acos.int": lambda: ([0], {}),
        "acosh.float": lambda: ([2.0], {}),
        "acos.float": lambda: ([0.5], {}),
        "affine_grid_generator.default": _template_affine_grid,
        "affine_grid_generator.out": _template_affine_grid_out,
        "all.dims": lambda: _template_all_any_dims(is_all=True),
        "all.dims_out": lambda: (
            [torch.tensor([[True, False], [True, True]]), [0], False],
            {"out": torch.empty((2,), dtype=torch.bool)},
        ),
        "all.int": lambda: _template_all_any_scalar_list("int"),
        "all.float": lambda: _template_all_any_scalar_list("float"),
        "all.bool": lambda: _template_all_any_scalar_list("bool"),
        "any.dims": lambda: _template_all_any_dims(is_all=False),
        "any.dims_out": lambda: (
            [torch.tensor([[True, False], [True, True]]), [0], False],
            {"out": torch.empty((2,), dtype=torch.bool)},
        ),
        "any.int": lambda: _template_all_any_scalar_list("int"),
        "any.float": lambda: _template_all_any_scalar_list("float"),
        "any.bool": lambda: _template_all_any_scalar_list("bool"),
        "argmax.out": lambda: (
            [torch.randn(2, 3), 0, False],
            {"out": torch.empty((3,), dtype=torch.long)},
        ),
        "argmin.out": lambda: (
            [torch.randn(2, 3), 0, False],
            {"out": torch.empty((3,), dtype=torch.long)},
        ),
        "as_strided.default": _template_as_strided,
        "as_strided_.default": _template_as_strided_inplace_safe,
        "as_strided_copy.default": _template_as_strided,
        "as_strided_copy.out": _template_as_strided_out,
        "as_strided_scatter.default": lambda: (
            [torch.zeros(6), torch.zeros(6), [6], [1], 0],
            {},
        ),
        "as_strided_scatter.out": lambda: (
            [torch.zeros(6), torch.zeros(6), [6], [1], 0],
            {"out": torch.zeros(6)},
        ),
        "adaptive_max_pool2d.default": _template_adaptive_max_pool2d_default,
        "adaptive_max_pool2d.out": lambda: _template_pool_out("2d"),
        "adaptive_max_pool2d_backward.default": _template_adaptive_max_pool2d_backward_default,
        "avg_pool2d.default": _template_avg_pool2d,
        "avg_pool2d.out": _template_avg_pool2d_out,
        "avg_pool2d_backward.default": _template_avg_pool2d_backward,
        "avg_pool2d_backward.grad_input": _template_avg_pool2d_backward_out,
        "adaptive_max_pool3d.default": _template_adaptive_max_pool3d_default,
        "adaptive_max_pool3d.out": lambda: _template_pool_out("3d"),
        "adaptive_max_pool3d_backward.default": _template_adaptive_max_pool3d_backward_default,
        "avg_pool3d.default": _template_avg_pool3d,
        "avg_pool3d.out": _template_avg_pool3d_out,
        "avg_pool3d_backward.default": _template_avg_pool3d_backward,
        "avg_pool3d_backward.grad_input": _template_avg_pool3d_backward_out,
        "batch_norm_backward.default": _template_batch_norm_backward,
        "acos.Scalar": lambda: _template_scalar_unary("acos"),
        "acosh.Scalar": lambda: _template_scalar_unary("acosh"),
        "add.Scalar": lambda: _template_add_scalar(out=False, inplace=False),
        "add.Scalar_out": lambda: _template_add_scalar(out=True, inplace=False),
        "add.default": lambda: ([1.0, 2.0], {}),
        "add.t": _skip("unsupported_add_t"),
        "add.str": _template_add_str,
        "add_.Scalar": lambda: _template_add_scalar(out=False, inplace=True),
        "add_.t": _skip("unsupported_add_t"),
        "any.str": _template_any_str,
        "arange.default": lambda: _template_arange("default"),
        "arange.start": lambda: _template_arange("start"),
        "arange.start_step": lambda: _template_arange("start_step"),
        "arange.start_out": lambda: _template_arange_out("start"),
        "arange.out": lambda: _template_arange_out("default"),
        "asin.Scalar": lambda: _template_scalar_unary("asin"),
        "asinh.Scalar": lambda: _template_scalar_unary("asinh"),
        "atan.Scalar": lambda: _template_scalar_unary("atan"),
        "atan2.Scalar_Scalar": _template_scalar_binary,
        "atanh.Scalar": lambda: _template_scalar_unary("atanh"),
        "all.dimname": _skip("dimname_not_supported"),
        "all.dimname_out": _skip("dimname_not_supported"),
        "any.dimname": _skip("dimname_not_supported"),
        "any.dimname_out": _skip("dimname_not_supported"),
        "angle.Scalar": lambda: ([0.5], {}),
        "bernoulli.default": _skip("randop_not_supported"),
        "bernoulli.out": _skip("randop_not_supported"),
        "bernoulli.p": _skip("randop_not_supported"),
        "bernoulli.Tensor": _skip("randop_not_supported"),
        "bernoulli.Tensor_out": _skip("randop_not_supported"),
        "bernoulli.float_out": _skip("randop_not_supported"),
        "bernoulli_.Tensor": _skip("randop_not_supported"),
        # Complex types not supported
        "acos.complex": _skip("complex_not_supported"),
        "acosh.complex": _skip("complex_not_supported"),
        "add.complex": _skip("complex_not_supported"),
        "add.int_complex": _skip("complex_not_supported"),
        "add.complex_int": _skip("complex_not_supported"),
        "add.float_complex": _skip("complex_not_supported"),
        "add.complex_float": _skip("complex_not_supported"),
        "asin.complex": _skip("complex_not_supported"),
        "asinh.complex": _skip("complex_not_supported"),
        "atan.complex": _skip("complex_not_supported"),
        "atanh.complex": _skip("complex_not_supported"),
    }
)

# Edit the OPS list to add/remove ops in this batch (format: "op.overload").
OPS = [
    "abs.default",
    "abs.out",
    "abs_.default",
    "absolute.default",
    "absolute.out",
    "absolute_.default",
    "acos.default",
    "acos.out",
    "acos.int",
    "acos.float",
    "acos.complex",
    "acos.Scalar",
    "acos_.default",
    "acosh.default",
    "acosh.out",
    "acosh.int",
    "acosh.float",
    "acosh.complex",
    "acosh.Scalar",
    "acosh_.default",
    "adaptive_max_pool2d.default",
    "adaptive_max_pool2d.out",
    "adaptive_max_pool2d_backward.default",
    "adaptive_max_pool2d_backward.grad_input",
    "adaptive_max_pool3d.default",
    "adaptive_max_pool3d.out",
    "adaptive_max_pool3d_backward.default",
    "adaptive_max_pool3d_backward.grad_input",
    "add.Tensor",
    "add.Scalar",
    "add.out",
    "add.Scalar_out",
    "add.t",
    "add.str",
    "add.int",
    "add.complex",
    "add.float",
    "add.int_complex",
    "add.complex_int",
    "add.float_complex",
    "add.complex_float",
    "add.int_float",
    "add.float_int",
    "add.default",
    "add_.Tensor",
    "add_.Scalar",
    "add_.t",
    "addbmm.default",
    "addbmm.out",
    "addbmm_.default",
    "addcdiv.default",
    "addcdiv.out",
    "addcdiv_.default",
    "addcmul.default",
    "addcmul.out",
    "addcmul_.default",
    "addmm.default",
    "addmm.out",
    "addmm.dtype_out",
    "addmm.dtype",
    "addmm_.default",
    "addmv.default",
    "addmv.out",
    "addmv_.default",
    "addr.default",
    "addr.out",
    "affine_grid_generator.default",
    "affine_grid_generator.out",
    "alias.default",
    "alias_copy.default",
    "alias_copy.out",
    "all.default",
    "all.dim",
    "all.dims",
    "all.out",
    "all.dims_out",
    "all.all_out",
    "all.dimname",
    "all.dimname_out",
    "all.int",
    "all.float",
    "all.bool",
    "alpha_dropout.default",
    "amax.default",
    "amax.out",
    "amin.default",
    "amin.out",
    "aminmax.default",
    "aminmax.out",
    "angle.default",
    "angle.out",
    "angle.int",
    "angle.float",
    "angle.complex",
    "angle.Scalar",
    "any.default",
    "any.dim",
    "any.dims",
    "any.out",
    "any.dims_out",
    "any.all_out",
    "any.dimname",
    "any.dimname_out",
    "any.str",
    "any.int",
    "any.float",
    "any.bool",
    "arange.default",
    "arange.start",
    "arange.start_step",
    "arange.start_out",
    "arange.out",
    "arccos.default",
    "arccos.out",
    "arccos_.default",
    "arccosh.default",
    "arccosh.out",
    "arccosh_.default",
    "arcsin.default",
    "arcsin.out",
    "arcsin_.default",
    "arcsinh.default",
    "arcsinh.out",
    "arcsinh_.default",
    "arctan.default",
    "arctan.out",
    "arctan2.default",
    "arctan2.out",
    "arctan2_.default",
    "arctan_.default",
    "arctanh.default",
    "arctanh.out",
    "arctanh_.default",
    "argmax.default",
    "argmax.out",
    "argmin.default",
    "argmin.out",
    "as_strided.default",
    "as_strided_.default",
    "as_strided_copy.default",
    "as_strided_copy.out",
    "as_strided_scatter.default",
    "as_strided_scatter.out",
    "asin.default",
    "asin.out",
    "asin.int",
    "asin.float",
    "asin.complex",
    "asin.Scalar",
    "asin_.default",
    "asinh.default",
    "asinh.out",
    "asinh.int",
    "asinh.float",
    "asinh.complex",
    "asinh.Scalar",
    "asinh_.default",
    "atan.default",
    "atan.out",
    "atan.int",
    "atan.float",
    "atan.complex",
    "atan.Scalar",
    "atan2.default",
    "atan2.out",
    "atan2.int",
    "atan2.float",
    "atan2.int_float",
    "atan2.float_int",
    "atan2.Scalar_Scalar",
    "atan2_.default",
    "atan_.default",
    "atanh.default",
    "atanh.out",
    "atanh.int",
    "atanh.float",
    "atanh.complex",
    "atanh.Scalar",
    "atanh_.default",
    "avg_pool2d.default",
    "avg_pool2d.out",
    "avg_pool2d_backward.default",
    "avg_pool2d_backward.grad_input",
    "avg_pool3d.default",
    "avg_pool3d.out",
    "avg_pool3d_backward.default",
    "avg_pool3d_backward.grad_input",
    "baddbmm.default",
    "baddbmm.out",
    "baddbmm.dtype_out",
    "baddbmm.dtype",
    "baddbmm_.default",
    "batch_norm_backward.default",
    "bernoulli.default",
    "bernoulli.out",
    "bernoulli.p",
    "bernoulli.Tensor",
    "bernoulli.Tensor_out",
    "bernoulli.float_out",
    "bernoulli_.Tensor",
]

if __name__ == "__main__":
    run_aten_op_batch(
        OPS,
        batch_label="test_batch_0",
        max_fails=20,
        templates=CUSTOM_TEMPLATES,
        show_skips=True,
    )
# CHECK: SUMMARY pass=
# CHECK-SAME: fail=0
