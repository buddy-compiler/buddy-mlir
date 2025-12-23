# RUN: %PYTHON %s 2>&1 | FileCheck %s

import torch
from aten_op_batch_runner import run_aten_op_batch

CUSTOM_TEMPLATES = {}


def _skip(reason: str):
    def fn():
        raise RuntimeError(reason)

    return fn


def _mat2x2():
    return torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float32)


def _template_item():
    return [torch.tensor(3.14)], {}


def _template_kthvalue():
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    return [x, 2], {}


def _template_kthvalue_out():
    args, _ = _template_kthvalue()
    # kthvalue on a 1D tensor returns scalar outputs when keepdim=False.
    values = torch.empty((), dtype=torch.float32)
    indices = torch.empty((), dtype=torch.int64)
    return args, {"values": values, "indices": indices}


def _template_lcm_tensor():
    a = torch.tensor([4], dtype=torch.int64)
    b = torch.tensor([6], dtype=torch.int64)
    return [a, b], {}


def _template_lcm_out():
    args, _ = _template_lcm_tensor()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_linalg_cross():
    a = torch.tensor([[1.0, 0.0, 0.0]])
    b = torch.tensor([[0.0, 1.0, 0.0]])
    return [a, b], {}


def _template_linalg_cross_out():
    args, _ = _template_linalg_cross()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_linalg_matrix():
    A = _mat2x2()
    return [A], {}


def _template_linalg_matrix_out():
    args, _ = _template_linalg_matrix()
    out = torch.empty_like(args[0])
    info = torch.empty((args[0].shape[0],), dtype=torch.int32)
    return args, {"inverse": out, "info": info}


def _template_linalg_cholesky_ex():
    A = _mat2x2()
    return [A], {"upper": False, "check_errors": False}


def _template_linalg_cholesky_ex_out():
    args, kwargs = _template_linalg_cholesky_ex()
    L = torch.empty_like(args[0])
    info = torch.empty((args[0].shape[0],), dtype=torch.int32)
    return args, {"L": L, "info": info, **kwargs}


def _template_linalg_eig_like():
    A = _mat2x2()
    return [A], {}


def _template_linalg_eig_like_out():
    args, _ = _template_linalg_eig_like()
    n = args[0].shape[-1]
    eigenvalues = torch.empty((n,), dtype=torch.complex64)
    eigenvectors = torch.empty((n, n), dtype=torch.complex64)
    return args, {"eigenvalues": eigenvalues, "eigenvectors": eigenvectors}


def _template_linalg_eigvals_out():
    args, _ = _template_linalg_eig_like()
    out = torch.empty((args[0].shape[-1],), dtype=torch.complex64)
    return args, {"out": out}


def _template_linalg_householder_product():
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    tau = torch.tensor([0.0, 0.0])
    return [a, tau], {}


def _template_linalg_householder_product_out():
    args, _ = _template_linalg_householder_product()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_linalg_ldl_factor():
    A = _mat2x2()
    return [A], {"hermitian": False, "check_errors": False}


def _template_linalg_ldl_factor_out():
    args, kwargs = _template_linalg_ldl_factor()
    LD = torch.empty_like(args[0])
    pivots = torch.empty((args[0].shape[0],), dtype=torch.int32)
    info = torch.empty((args[0].shape[0],), dtype=torch.int32)
    return args, {"LD": LD, "pivots": pivots, "info": info, **kwargs}


def _template_linalg_ldl_solve():
    LD, pivots, _ = torch.linalg.ldl_factor_ex(_mat2x2())
    B = torch.ones_like(_mat2x2())
    return [LD, pivots, B], {"hermitian": False}


def _template_linalg_ldl_solve_out():
    args, kwargs = _template_linalg_ldl_solve()
    out = torch.empty_like(args[2])
    return args, {"out": out, **kwargs}


def _template_linalg_lu():
    A = _mat2x2()
    return [A], {"pivot": True}


def _template_linalg_lu_out():
    args, kwargs = _template_linalg_lu()
    P = torch.empty_like(args[0])
    L = torch.empty_like(args[0])
    U = torch.empty_like(args[0])
    return args, {"P": P, "L": L, "U": U, **kwargs}


def _template_linalg_lu_factor_ex():
    A = _mat2x2()
    return [A], {"pivot": True, "check_errors": False}


def _template_linalg_lu_factor_ex_out():
    args, kwargs = _template_linalg_lu_factor_ex()
    LU = torch.empty_like(args[0])
    pivots = torch.empty((args[0].shape[0],), dtype=torch.int32)
    info = torch.empty((args[0].shape[0],), dtype=torch.int32)
    return args, {"LU": LU, "pivots": pivots, "info": info, **kwargs}


def _template_linalg_lu_solve():
    LU, pivots, _ = torch.linalg.lu_factor_ex(_mat2x2())
    B = torch.ones_like(_mat2x2())
    return [LU, pivots, B], {"left": True, "adjoint": False}


def _template_linalg_lu_solve_out():
    args, kwargs = _template_linalg_lu_solve()
    out = torch.empty_like(args[2])
    return args, {"out": out, **kwargs}


def _template_linalg_matrix_exp():
    A = _mat2x2()
    return [A], {}


def _template_linalg_matrix_exp_out():
    args, _ = _template_linalg_matrix_exp()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_linalg_qr():
    A = _mat2x2()
    return [A], {"mode": "reduced"}


def _template_linalg_qr_out():
    args, kwargs = _template_linalg_qr()
    Q = torch.empty_like(args[0])
    R = torch.empty_like(args[0])
    return args, {"Q": Q, "R": R, **kwargs}


def _template_linalg_solve_triangular():
    A = torch.triu(_mat2x2())
    B = torch.ones_like(_mat2x2())
    return [A, B], {"upper": True}


def _template_linalg_solve_triangular_out():
    args, kwargs = _template_linalg_solve_triangular()
    out = torch.empty_like(args[1])
    return args, {"out": out, **kwargs}


def _template_linalg_vector_norm():
    x = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    return [x, 2, None], {}


def _template_linalg_vector_norm_out():
    args, _ = _template_linalg_vector_norm()
    out = torch.empty((1,), dtype=torch.float32)
    return args, {"out": out}


def _template_linear():
    x = torch.randn(2, 4, dtype=torch.float32)
    weight = torch.randn(3, 4, dtype=torch.float32)
    bias = torch.randn(3, dtype=torch.float32)
    return [x, weight, bias], {}


def _template_linear_out():
    args, _ = _template_linear()
    out = torch.empty((args[0].shape[0], args[1].shape[0]), dtype=torch.float32)
    return args, {"out": out}


def _template_linear_backward_base():
    grad = torch.ones(2, 3, dtype=torch.float32)
    inp = torch.randn(2, 4, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
    args = [grad, inp, weight, [True, True, True]]
    return args


def _template_linear_backward():
    return _template_linear_backward_base(), {}


def _template_linear_backward_out():
    args = _template_linear_backward_base()
    grad_input = torch.empty_like(args[1])
    grad_weight = torch.empty_like(args[2])
    grad_bias = torch.empty((args[2].shape[0],), dtype=torch.float32)
    return args, {"out0": grad_input, "out1": grad_weight, "out2": grad_bias}


def _template_linspace():
    return [0.0, 1.0, 5], {}


def _template_linspace_out():
    args, _ = _template_linspace()
    out = torch.empty((args[2],), dtype=torch.float32)
    return args, {"out": out}


def _template_linspace_tensor():
    start = torch.tensor(0.0)
    end = torch.tensor(1.0)
    return [start, end, 5], {}


def _template_linspace_tensor_out():
    args, _ = _template_linspace_tensor()
    out = torch.empty((args[2],), dtype=torch.float32)
    return args, {"out": out}


def _template_linspace_tensor_scalar():
    start = torch.tensor(0.0)
    return [start, 1.0, 5], {}


def _template_linspace_tensor_scalar_out():
    args, _ = _template_linspace_tensor_scalar()
    out = torch.empty((args[2],), dtype=torch.float32)
    return args, {"out": out}


def _template_linspace_scalar_tensor():
    end = torch.tensor(1.0)
    return [0.0, end, 5], {}


def _template_linspace_scalar_tensor_out():
    args, _ = _template_linspace_scalar_tensor()
    out = torch.empty((args[2],), dtype=torch.float32)
    return args, {"out": out}


def _template_log_normal():
    base = torch.ones(2, dtype=torch.float32)
    return [base, 0.5, 1.0], {}


def _template_log_normal_out():
    args, _ = _template_log_normal()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_log_sigmoid_forward():
    x = torch.randn(2, dtype=torch.float32)
    return [x], {}


def _template_log_sigmoid_forward_out():
    args, _ = _template_log_sigmoid_forward()
    out = torch.empty_like(args[0])
    buffer = torch.empty_like(args[0])
    return args, {"output": out, "buffer": buffer}


def _template_log_sigmoid_backward():
    torch.manual_seed(0)
    self = torch.randn(2, 3, dtype=torch.float32)
    output, buffer = torch.ops.aten.log_sigmoid_forward.default(self)
    grad_output = torch.randn_like(output)
    return [grad_output, self, buffer], {}


def _template_log_sigmoid_backward_grad_input():
    args, _ = _template_log_sigmoid_backward()
    grad_input = torch.empty_like(args[1])
    return args, {"grad_input": grad_input}


def _template_logspace_tensor():
    start = torch.tensor(0.0)
    end = torch.tensor(1.0)
    return [start, end, 5], {}


def _template_logspace_tensor_out():
    args, _ = _template_logspace_tensor()
    out = torch.empty((args[2],), dtype=torch.float32)
    return args, {"out": out}


def _template_logspace_tensor_scalar():
    start = torch.tensor(0.0)
    return [start, 1.0, 5], {}


def _template_logspace_tensor_scalar_out():
    args, _ = _template_logspace_tensor_scalar()
    out = torch.empty((args[2],), dtype=torch.float32)
    return args, {"out": out}


def _template_logspace_scalar_tensor():
    end = torch.tensor(1.0)
    return [0.0, end, 5], {}


def _template_logspace_scalar_tensor_out():
    args, _ = _template_logspace_scalar_tensor()
    out = torch.empty((args[2],), dtype=torch.float32)
    return args, {"out": out}


def _template_logspace_scalar():
    return [0.0, 1.0, 5], {}


def _template_logspace_scalar_out():
    args, _ = _template_logspace_scalar()
    out = torch.empty((args[2],), dtype=torch.float32)
    return args, {"out": out}


def _template_scalar_input(a=1.0):
    return [a], {}


def _template_scalar_pair(a=1.0, b=2.0):
    return [a, b], {}


def _template_scalar_out_fn(a=1.0, b=2.0):
    args = [a, b]
    out = torch.empty(1, dtype=torch.float32)
    return args, {"out": out}


def _template_lu_unpack():
    A = _mat2x2()
    LU, pivots = torch.lu(A)
    return [LU, pivots], {"unpack_data": True, "unpack_pivots": True}


def _template_lu_unpack_out():
    args, kwargs = _template_lu_unpack()
    n = args[0].shape[0]
    P = torch.empty((n, n), dtype=args[0].dtype)
    L = torch.empty_like(args[0])
    U = torch.empty_like(args[0])
    return args, {"P": P, "L": L, "U": U, **kwargs}


def _template_masked_fill_tensor():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask = torch.tensor([[True, False], [False, True]])
    value = torch.tensor(0.0)
    return [x, mask, value], {}


def _template_masked_fill_tensor_out():
    args, _ = _template_masked_fill_tensor()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_masked_fill_tensor_inplace():
    args, _ = _template_masked_fill_tensor()
    return args, {}


def _template_masked_scatter():
    x = torch.zeros(4, dtype=torch.float32)
    mask = torch.tensor([True, False, True, False])
    source = torch.tensor([5.0, 6.0], dtype=torch.float32)
    return [x, mask, source], {}


def _template_masked_scatter_out():
    args, _ = _template_masked_scatter()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_masked_scatter_backward():
    grad = torch.ones(4, dtype=torch.float32)
    mask = torch.tensor([True, False, True, False])
    sizes = [4]
    return [grad, mask, sizes], {}


def _template_cmp_tensor_tensor():
    a = torch.tensor([1.0, 2.0], dtype=torch.float32)
    b = torch.tensor([0.5, 2.5], dtype=torch.float32)
    return [a, b], {}


def _template_cmp_tensor_scalar():
    a = torch.tensor([1.0, 2.0], dtype=torch.float32)
    return [a, 1.5], {}


def _template_cmp_tensor_scalar_out():
    args, _ = _template_cmp_tensor_scalar()
    out = torch.empty_like(args[0], dtype=torch.bool)
    return args, {"out": out}


def _template_cmp_tensor_tensor_out():
    args, _ = _template_cmp_tensor_tensor()
    out = torch.empty_like(args[0], dtype=torch.bool)
    return args, {"out": out}


def _template_cmp_scalar_pair(a, b):
    return [a, b], {}


def _template_lerp_scalar():
    a = torch.tensor([1.0, 2.0], dtype=torch.float32)
    b = torch.tensor([3.0, 4.0], dtype=torch.float32)
    return [a, b, 0.5], {}


def _template_lerp_scalar_out():
    args, _ = _template_lerp_scalar()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_leaky_relu_backward():
    grad = torch.ones(2, dtype=torch.float32)
    self = torch.tensor([1.0, -1.0], dtype=torch.float32)
    negative_slope = 0.1
    return [grad, self, negative_slope, True], {}


def _template_leaky_relu_backward_out():
    args, _ = _template_leaky_relu_backward()
    grad_input = torch.empty_like(args[1])
    return args, {"grad_input": grad_input}


def _template_masked_fill_scalar():
    x = torch.tensor([1.0, 2.0])
    mask = torch.tensor([True, False])
    value = 0.0
    return [x, mask, value], {}


def _template_masked_fill_scalar_out():
    args, _ = _template_masked_fill_scalar()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_log_inputs():
    return [2, 3], {}


def _template_log_float_inputs():
    return [2.0, 3.0], {}


def _template_lstm_input():
    torch.manual_seed(0)
    x = torch.randn(2, 1, 3)
    mod = torch.nn.LSTM(
        input_size=3, hidden_size=4, num_layers=1, batch_first=False
    )
    params = [p for p in mod.parameters()]
    h0 = torch.zeros(1, 1, 4)
    c0 = torch.zeros(1, 1, 4)
    return [x, [h0, c0], params, True, 1, 0.0, False, False, False], {}


# Register custom templates or skips.
CUSTOM_TEMPLATES.update(
    {
        "item.default": _template_item,
        "kthvalue.default": _template_kthvalue,
        "kthvalue.values": _template_kthvalue_out,
        "kthvalue.dimname": _skip("named_tensor_torchscript"),
        "kthvalue.dimname_out": _skip("named_tensor_torchscript"),
        "lcm.default": _template_lcm_tensor,
        "lcm.out": _template_lcm_out,
        "lcm_.default": _template_lcm_tensor,
        "linalg_cholesky_ex.default": _skip("linalg_not_supported"),
        "linalg_cholesky_ex.L": _skip("linalg_not_supported"),
        "linalg_cross.default": _template_linalg_cross,
        "linalg_cross.out": _template_linalg_cross_out,
        "linalg_eig.default": _skip("linalg_not_supported"),
        "linalg_eig.out": _skip("linalg_not_supported"),
        "linalg_eigvals.default": _skip("linalg_not_supported"),
        "linalg_eigvals.out": _skip("linalg_not_supported"),
        "linalg_householder_product.default": _skip("linalg_not_supported"),
        "linalg_householder_product.out": _skip("linalg_not_supported"),
        "linalg_inv_ex.default": _skip("linalg_not_supported"),
        "linalg_inv_ex.inverse": _skip("linalg_not_supported"),
        "linalg_ldl_factor_ex.default": _skip("linalg_not_supported"),
        "linalg_ldl_factor_ex.out": _skip("linalg_not_supported"),
        "linalg_ldl_solve.default": _skip("linalg_not_supported"),
        "linalg_ldl_solve.out": _skip("linalg_not_supported"),
        "linalg_lu.default": _skip("linalg_not_supported"),
        "linalg_lu.out": _skip("linalg_not_supported"),
        "linalg_lu_factor_ex.default": _skip("linalg_not_supported"),
        "linalg_lu_factor_ex.out": _skip("linalg_not_supported"),
        "linalg_lu_solve.default": _skip("linalg_not_supported"),
        "linalg_lu_solve.out": _skip("linalg_not_supported"),
        "linalg_matrix_exp.default": _skip("linalg_not_supported"),
        "linalg_matrix_exp.out": _skip("linalg_not_supported"),
        "linalg_qr.default": _skip("linalg_not_supported"),
        "linalg_qr.out": _skip("linalg_not_supported"),
        "linalg_solve_triangular.default": _skip("linalg_not_supported"),
        "linalg_solve_triangular.out": _skip("linalg_not_supported"),
        "linalg_vector_norm.default": _template_linalg_vector_norm,
        "linalg_vector_norm.out": _template_linalg_vector_norm_out,
        "linear.default": _template_linear,
        "linear.out": _template_linear_out,
        "linear_backward.default": _skip("linear_backward_backend_missing"),
        "linear_backward.out": _skip("linear_backward_backend_missing"),
        "linspace.default": _skip("backend_crash_linspace"),
        "linspace.out": _skip("backend_crash_linspace"),
        "linspace.Tensor_Tensor": _skip("backend_crash_linspace"),
        "linspace.Tensor_Scalar": _skip("backend_crash_linspace"),
        "linspace.Scalar_Tensor": _skip("backend_crash_linspace"),
        "linspace.Tensor_Tensor_out": _skip("backend_crash_linspace"),
        "linspace.Tensor_Scalar_out": _skip("backend_crash_linspace"),
        "linspace.Scalar_Tensor_out": _skip("backend_crash_linspace"),
        "log_normal.default": _skip("random_op_not_supported"),
        "log_normal.out": _skip("random_op_not_supported"),
        "log_sigmoid_backward.default": _template_log_sigmoid_backward,
        "log_sigmoid_backward.grad_input": _template_log_sigmoid_backward_grad_input,
        "log_sigmoid_forward.default": _template_log_sigmoid_forward,
        "log_sigmoid_forward.output": _template_log_sigmoid_forward_out,
        # torch.compile doesn't support named tensors / dimname-based overloads.
        "logsumexp.names": _skip("named_tensor_unsupported"),
        "logsumexp.names_out": _skip("named_tensor_unsupported"),
        "logspace.Tensor_Tensor": _skip("backend_crash_logspace"),
        "logspace.Tensor_Tensor_out": _skip("backend_crash_logspace"),
        "logspace.Tensor_Scalar": _skip("backend_crash_logspace"),
        "logspace.Scalar_Tensor": _skip("backend_crash_logspace"),
        "logspace.default": _skip("backend_crash_logspace"),
        "logspace.out": _skip("backend_crash_logspace"),
        "logspace.Tensor_Scalar_out": _skip("backend_crash_logspace"),
        "logspace.Scalar_Tensor_out": _skip("backend_crash_logspace"),
        "log.int_int": _template_log_inputs,
        "log.float_float": _template_log_float_inputs,
        "log.Scalar": lambda: _template_scalar_input(2.0),
        "log.Scalar_Scalar": lambda: _template_scalar_pair(2.0, 3.0),
        "log10.Scalar": lambda: _template_scalar_input(2.0),
        "log1p.Scalar": lambda: _template_scalar_input(2.0),
        "log_normal_.default": _skip("random_op_not_supported"),
        "lstm.input": _skip("rnn_not_supported"),
        "lstm.data": _skip("rnn_not_supported"),
        "lu_unpack.default": _skip("linalg_not_supported"),
        "lu_unpack.out": _skip("linalg_not_supported"),
        "masked_fill.Tensor": _template_masked_fill_tensor,
        "masked_fill.Tensor_out": _template_masked_fill_tensor_out,
        "masked_fill_.Tensor": _template_masked_fill_tensor_inplace,
        "masked_fill.Scalar": _template_masked_fill_scalar,
        "masked_fill.Scalar_out": _template_masked_fill_scalar_out,
        "masked_fill_.Scalar": _template_masked_fill_scalar,
        "masked_scatter.default": _template_masked_scatter,
        "masked_scatter.out": _template_masked_scatter_out,
        "masked_scatter_.default": _template_masked_scatter,
        "masked_scatter_backward.default": _skip("backward_not_supported"),
        "le.Tensor": _template_cmp_tensor_tensor,
        "le.Tensor_out": _template_cmp_tensor_tensor_out,
        "le.Scalar": _template_cmp_tensor_scalar,
        "le.int": lambda: _template_cmp_scalar_pair(1, 0),
        "le.float": lambda: _template_cmp_scalar_pair(1.0, 0.5),
        "le.int_float": lambda: _template_cmp_scalar_pair(1, 0.5),
        "le.float_int": lambda: _template_cmp_scalar_pair(1.0, 1),
        "le.default": lambda: _template_cmp_scalar_pair(1.0, 1.0),
        "le.str": lambda: _template_cmp_scalar_pair("a", "b"),
        "le_.Scalar": _template_cmp_tensor_scalar,
        "le_.Tensor": _template_cmp_tensor_tensor,
        "less.Tensor": _template_cmp_tensor_tensor,
        "less.Tensor_out": _template_cmp_tensor_tensor_out,
        "less.Scalar": _template_cmp_tensor_scalar,
        "less.Scalar_out": _template_cmp_tensor_scalar_out,
        "less_.Scalar": _template_cmp_tensor_scalar,
        "less_.Tensor": _template_cmp_tensor_tensor,
        "less_equal.Tensor": _template_cmp_tensor_tensor,
        "less_equal.Tensor_out": _template_cmp_tensor_tensor_out,
        "less_equal.Scalar": _template_cmp_tensor_scalar,
        "less_equal.Scalar_out": _template_cmp_tensor_scalar_out,
        "less_equal_.Scalar": _template_cmp_tensor_scalar,
        "less_equal_.Tensor": _template_cmp_tensor_tensor,
        "lt.Tensor": _template_cmp_tensor_tensor,
        "lt.Tensor_out": _template_cmp_tensor_tensor_out,
        "lt.Scalar": _template_cmp_tensor_scalar,
        "lt.Scalar_out": _template_cmp_tensor_scalar_out,
        "lt.int": lambda: _template_cmp_scalar_pair(1, 0),
        "lt.float": lambda: _template_cmp_scalar_pair(1.0, 0.5),
        "lt.int_float": lambda: _template_cmp_scalar_pair(1, 0.5),
        "lt.float_int": lambda: _template_cmp_scalar_pair(1.0, 1),
        "lt.default": lambda: _template_cmp_scalar_pair(1.0, 1.0),
        "lt.str": lambda: _template_cmp_scalar_pair("a", "b"),
        "lt_.Scalar": _template_cmp_tensor_scalar,
        "lt_.Tensor": _template_cmp_tensor_tensor,
        "lerp.Scalar": _template_lerp_scalar,
        "lerp.Scalar_out": _template_lerp_scalar_out,
        "lerp_.Scalar": _template_lerp_scalar,
        "leaky_relu_backward.default": _template_leaky_relu_backward,
        "leaky_relu_backward.grad_input": _template_leaky_relu_backward_out,
        "lgamma.Scalar": lambda: _template_scalar_input(2.0),
        "le.Scalar_out": _skip("meta_missing"),
    }
)

# Edit the OPS list to add/remove ops in this batch (format: "op.overload").
OPS = [
    "item.default",
    "kthvalue.default",
    "kthvalue.dimname",
    "kthvalue.dimname_out",
    "kthvalue.values",
    "lcm.default",
    "lcm.out",
    "lcm_.default",
    "le.Tensor",
    "le.Scalar",
    "le.Scalar_out",
    "le.Tensor_out",
    "le.int",
    "le.float",
    "le.int_float",
    "le.float_int",
    "le.default",
    "le.str",
    "le_.Scalar",
    "le_.Tensor",
    "leaky_relu.default",
    "leaky_relu.out",
    "leaky_relu_.default",
    "leaky_relu_backward.default",
    "leaky_relu_backward.grad_input",
    "lerp.Scalar",
    "lerp.Tensor",
    "lerp.Scalar_out",
    "lerp.Tensor_out",
    "lerp_.Scalar",
    "lerp_.Tensor",
    "less.Tensor",
    "less.Scalar",
    "less.Scalar_out",
    "less.Tensor_out",
    "less_.Scalar",
    "less_.Tensor",
    "less_equal.Tensor",
    "less_equal.Scalar",
    "less_equal.Scalar_out",
    "less_equal.Tensor_out",
    "less_equal_.Scalar",
    "less_equal_.Tensor",
    "lgamma.default",
    "lgamma.out",
    "lgamma.int",
    "lgamma.float",
    "lgamma.Scalar",
    "lgamma_.default",
    "lift.default",
    "lift.out",
    "lift_fresh.default",
    "lift_fresh_copy.default",
    "lift_fresh_copy.out",
    "linalg_cholesky_ex.default",
    "linalg_cholesky_ex.L",
    "linalg_cross.default",
    "linalg_cross.out",
    "linalg_eig.default",
    "linalg_eig.out",
    "linalg_eigvals.default",
    "linalg_eigvals.out",
    "linalg_householder_product.default",
    "linalg_householder_product.out",
    "linalg_inv_ex.default",
    "linalg_inv_ex.inverse",
    "linalg_ldl_factor_ex.default",
    "linalg_ldl_factor_ex.out",
    "linalg_ldl_solve.default",
    "linalg_ldl_solve.out",
    "linalg_lu.default",
    "linalg_lu.out",
    "linalg_lu_factor_ex.default",
    "linalg_lu_factor_ex.out",
    "linalg_lu_solve.default",
    "linalg_lu_solve.out",
    "linalg_matrix_exp.default",
    "linalg_matrix_exp.out",
    "linalg_qr.default",
    "linalg_qr.out",
    "linalg_solve_triangular.default",
    "linalg_solve_triangular.out",
    "linalg_vector_norm.default",
    "linalg_vector_norm.out",
    "linear.default",
    "linear.out",
    "linear_backward.out",
    "linear_backward.default",
    "linspace.Tensor_Tensor",
    "linspace.Tensor_Scalar",
    "linspace.Scalar_Tensor",
    "linspace.default",
    "linspace.out",
    "linspace.Tensor_Tensor_out",
    "linspace.Tensor_Scalar_out",
    "linspace.Scalar_Tensor_out",
    "log.default",
    "log.out",
    "log.int",
    "log.float",
    "log.complex",
    "log.Scalar",
    "log.int_int",
    "log.float_float",
    "log.complex_complex",
    "log.int_float",
    "log.float_int",
    "log.int_complex",
    "log.complex_int",
    "log.float_complex",
    "log.complex_float",
    "log.Scalar_Scalar",
    "log10.default",
    "log10.out",
    "log10.int",
    "log10.float",
    "log10.complex",
    "log10.Scalar",
    "log10_.default",
    "log1p.default",
    "log1p.out",
    "log1p.int",
    "log1p.float",
    "log1p.Scalar",
    "log1p_.default",
    "log2.default",
    "log2.out",
    "log2_.default",
    "log_.default",
    "log_normal.default",
    "log_normal.out",
    "log_normal_.default",
    "log_sigmoid_backward.default",
    "log_sigmoid_backward.grad_input",
    "log_sigmoid_forward.default",
    "log_sigmoid_forward.output",
    "logaddexp.default",
    "logaddexp.out",
    "logaddexp2.default",
    "logaddexp2.out",
    "logcumsumexp.default",
    "logcumsumexp.dimname",
    "logcumsumexp.dimname_out",
    "logcumsumexp.out",
    "logical_and.default",
    "logical_and.out",
    "logical_and_.default",
    "logical_not.default",
    "logical_not.out",
    "logical_not_.default",
    "logical_or.default",
    "logical_or.out",
    "logical_or_.default",
    "logical_xor.default",
    "logical_xor.out",
    "logical_xor_.default",
    "logit.default",
    "logit.out",
    "logit_.default",
    "logit_backward.default",
    "logit_backward.grad_input",
    "logspace.Tensor_Tensor",
    "logspace.Tensor_Scalar",
    "logspace.Scalar_Tensor",
    "logspace.default",
    "logspace.out",
    "logspace.Tensor_Tensor_out",
    "logspace.Tensor_Scalar_out",
    "logspace.Scalar_Tensor_out",
    "logsumexp.default",
    "logsumexp.names",
    "logsumexp.names_out",
    "logsumexp.out",
    "lstm.input",
    "lstm.data",
    "lt.Tensor",
    "lt.Scalar",
    "lt.Scalar_out",
    "lt.Tensor_out",
    "lt.int",
    "lt.float",
    "lt.int_float",
    "lt.float_int",
    "lt.default",
    "lt.str",
    "lt_.Scalar",
    "lt_.Tensor",
    "lu_unpack.default",
    "lu_unpack.out",
    "margin_ranking_loss.default",
    "masked_fill.Scalar",
    "masked_fill.Tensor",
    "masked_fill.Scalar_out",
    "masked_fill.Tensor_out",
    "masked_fill_.Scalar",
    "masked_fill_.Tensor",
    "masked_scatter.default",
    "masked_scatter.out",
    "masked_scatter_.default",
    "masked_scatter_backward.default",
]

if __name__ == "__main__":
    run_aten_op_batch(
        OPS,
        batch_label="test_batch_4",
        max_fails=20,
        templates=CUSTOM_TEMPLATES,
        show_skips=True,
    )
# CHECK: SUMMARY pass=
# CHECK-SAME: fail=0
