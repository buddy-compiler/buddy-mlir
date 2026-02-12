# RUN: %PYTHON %s 2>&1 | FileCheck %s
from aten_op_batch_runner import run_aten_op_batch
import torch


CUSTOM_TEMPLATES = {}


def _diag_base():
    return torch.randn(2, 3, dtype=torch.float32)


def _skip(reason: str):
    def fn():
        raise RuntimeError(reason)

    return fn


def _template_diagonal():
    x = _diag_base()
    return [x, 0, 0], {}


def _template_diagonal_out():
    args, _ = _template_diagonal()
    out = torch.empty(2, dtype=args[0].dtype)
    return args, {"out": out}


def _template_diagonal_backward():
    x = _diag_base()
    grad_output = torch.ones(2, dtype=torch.float32)
    return [grad_output, list(x.shape), 0, 0, 1], {}


def _template_diagonal_backward_out():
    args, _ = _template_diagonal_backward()
    out = torch.empty(args[1][0], args[1][1], dtype=torch.float32)
    return args, {"out": out}


def _template_diagonal_copy():
    args, _ = _template_diagonal()
    return args, {}


def _template_diagonal_copy_out():
    args, _ = _template_diagonal_copy()
    out = torch.empty(2, dtype=args[0].dtype)
    return args, {"out": out}


def _template_diagonal_scatter():
    x = _diag_base()
    src = torch.ones(min(x.shape), dtype=torch.float32)
    return [x, src, 0, 0, 1], {}


def _template_diagonal_scatter_out():
    args, _ = _template_diagonal_scatter()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_diagonal_dimname():
    x = _diag_base().refine_names("N", "C")
    return [x], {
        "outdim": "D",
        "dim1": "N",
        "dim2": "C",
        "offset": 0,
    }


def _template_div_int():
    return [4, 2], {}


def _template_div_scalar():
    x = torch.tensor([4.0, 2.0], dtype=torch.float32)
    return [x, 2.0], {}


def _template_div_scalar_out():
    args, _ = _template_div_scalar()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_div_scalar_mode():
    x = torch.tensor([4.0, 2.0], dtype=torch.float32)
    return [x, 2.0], {"rounding_mode": None}


def _template_div_tensor_mode():
    x = torch.tensor([4.0, 2.0], dtype=torch.float32)
    y = torch.tensor([2.0, 1.0], dtype=torch.float32)
    return [x, y], {"rounding_mode": None}


def _template_div_tensor_mode_out():
    args, _ = _template_div_tensor_mode()
    out = torch.empty_like(args[0])
    return args, {"out": out, "rounding_mode": None}


def _template_embedding():
    weight = torch.randn(4, 3, dtype=torch.float32)
    indices = torch.tensor([0, 1], dtype=torch.int64)
    return [weight, indices, -1, False, False], {}


def _template_embedding_out():
    args, _ = _template_embedding()
    out = torch.empty((2, 3), dtype=torch.float32)
    return args, {"out": out}


def _template_embedding_dense_backward():
    grad_output = torch.randn(2, 3, dtype=torch.float32)
    indices = torch.tensor([0, 1], dtype=torch.int64)
    num_weights = 4
    padding_idx = -1
    return [grad_output, indices, num_weights, padding_idx, False], {}


def _template_embedding_dense_backward_out():
    args, _ = _template_embedding_dense_backward()
    out = torch.empty((args[2], args[0].shape[1]), dtype=torch.float32)
    return args, {"out": out}


def _template_empty_sizes():
    return [[0]], {}


def _template_empty_out():
    args, _ = _template_empty_sizes()
    out = torch.empty(0)
    return args, {"out": out}


def _template_empty_permuted():
    size = [0]
    layout = [0]
    return [size, layout], {}


def _template_empty_permuted_out():
    args, _ = _template_empty_permuted()
    out = torch.empty(0, dtype=torch.float32)
    return args, {"out": out}


def _template_empty_strided():
    size = [0]
    stride = [1]
    return [size, stride], {}


def _template_empty_strided_out():
    args, _ = _template_empty_strided()
    out = torch.empty(0, dtype=torch.float32)
    return args, {"out": out}


def _template_empty_like():
    x = torch.empty(0, dtype=torch.float32)
    return [x], {}


def _template_empty_like_out():
    args, _ = _template_empty_like()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_eq_int_list():
    return [[0, 1], [1, 1]], {}


def _template_eq_float_list():
    return [[0.5, 1.0], [0.5, 1.0]], {}


def _template_eq_bool_list():
    return [[True, False], [True, False]], {}


def _template_eq_tensor_list():
    a = torch.tensor([1.0])
    b = torch.tensor([1.0])
    return [[a], [b]], {}


def _template_eq_scalar():
    x = torch.tensor([1.0], dtype=torch.float32)
    return [x, 1.0], {}


def _template_eq_scalar_out():
    args, _ = _template_eq_scalar()
    out = torch.empty_like(args[0], dtype=torch.bool)
    return args, {"out": out}


def _template_expand():
    x = torch.tensor([1.0])
    size = [2, 2]
    return [x, size], {}


def _template_expand_out():
    args, _ = _template_expand()
    out = torch.empty((2, 2), dtype=torch.float32)
    return args, {"out": out}


def _template_floor_scalar():
    return [1.5], {}


def _template_floor_divide_scalar():
    x = torch.tensor([4.0, 5.0], dtype=torch.float32)
    return [x, 2.0], {}


def _template_floor_divide_scalar_out():
    args, _ = _template_floor_divide_scalar()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_fmod_scalar():
    x = torch.tensor([5.0, 3.0], dtype=torch.float32)
    return [x, 2.0], {}


def _template_fill_scalar():
    x = torch.zeros(2, dtype=torch.float32)
    return [x, 1.0], {}


def _template_fill_scalar_out():
    args, _ = _template_fill_scalar()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_fill_scalar_inplace():
    args, _ = _template_fill_scalar()
    args[0] = args[0].clone()
    return args, {}


def _template_float_power_scalar():
    x = torch.tensor([2.0, 3.0], dtype=torch.float64)
    return [x, 2.0], {}


def _template_fft_1d():
    x = torch.randn(4, dtype=torch.complex64)
    return [x, None, -1, None], {}


def _template_fft_1d_out():
    args, _ = _template_fft_1d()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_fft_2d():
    x = torch.randn(2, 2, dtype=torch.complex64)
    return [x, None, [-2, -1], None], {}


def _template_fft_2d_out():
    args, _ = _template_fft_2d()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_fft_nd():
    x = torch.randn(2, 2, 2, dtype=torch.complex64)
    return [x, None, None, None], {}


def _template_fft_nd_out():
    args, _ = _template_fft_nd()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_fft_rfft():
    x = torch.randn(4, dtype=torch.float32)
    return [x, None, -1, None], {}


def _template_fft_rfft_out():
    args, _ = _template_fft_rfft()
    out = torch.empty(3, dtype=torch.complex64)
    return args, {"out": out}


def _template_fft_irfft():
    x = torch.randn(3, dtype=torch.complex64)
    return [x, None, -1, None], {}


def _template_fft_irfft_out():
    args, _ = _template_fft_irfft()
    out = torch.empty(4, dtype=torch.float32)
    return args, {"out": out}


def _template_fft_hfft():
    x = torch.randn(4, dtype=torch.complex64)
    return [x, None, -1, None], {}


def _template_fft_hfft_out():
    args, _ = _template_fft_hfft()
    out = torch.empty(4, dtype=torch.float32)
    return args, {"out": out}


def _template_fft_ihfft():
    x = torch.randn(4, dtype=torch.float32)
    return [x, None, -1, None], {}


def _template_fft_ihfft_out():
    args, _ = _template_fft_ihfft()
    out = torch.empty(4, dtype=torch.complex64)
    return args, {"out": out}


def _template_elu_backward():
    x = torch.tensor([0.5, -0.5], dtype=torch.float32)
    grad_output = torch.ones_like(x)
    alpha = 1.0
    scale = 1.0
    input_scale = 1.0
    is_result = False
    return [grad_output, alpha, scale, input_scale, is_result, x], {}


def _template_elu_backward_out():
    args, _ = _template_elu_backward()
    out = torch.empty_like(args[0])
    return args, {"grad_input": out}


def _template_exponential():
    x = torch.empty(0, dtype=torch.float32)
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    return [x, 1.0], {"generator": g}


def _template_exponential_out():
    args, kwargs = _template_exponential()
    out = torch.empty_like(args[0])
    kwargs["out"] = out
    return args, kwargs


def _template_exponential_inplace():
    args, kwargs = _template_exponential()
    args[0] = args[0].clone()
    return args, kwargs


def _template_fill_tensor():
    x = torch.tensor([1.0, 2.0], dtype=torch.float32)
    val = torch.tensor(3.0)
    return [x, val], {}


def _template_fill_tensor_out():
    args, _ = _template_fill_tensor()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_fill_tensor_inplace():
    args, _ = _template_fill_tensor()
    args[0] = args[0].clone()
    return args, {}


def _template_flip():
    x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    return [x, [1]], {}


def _template_flip_out():
    args, _ = _template_flip()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_float_power_inplace():
    x = torch.tensor([2.0, 3.0], dtype=torch.float64)
    exp = torch.tensor([2.0], dtype=torch.float64)
    return [x, exp], {}


CUSTOM_TEMPLATES.update(
    {
        "diagonal.default": _template_diagonal,
        "diagonal.out": _template_diagonal_out,
        # torch.compile/TorchScript doesn't support named tensors yet.
        "diagonal.Dimname": _skip("named_tensor_torchscript"),
        "diagonal_backward.default": _template_diagonal_backward,
        "diagonal_backward.out": _template_diagonal_backward_out,
        "diagonal_copy.default": _template_diagonal_copy,
        "diagonal_copy.out": _template_diagonal_copy_out,
        "diagonal_scatter.default": _template_diagonal_scatter,
        "diagonal_scatter.out": _template_diagonal_scatter_out,
        "div.int": _template_div_int,
        "div.Scalar": _template_div_scalar,
        "div.Scalar_out": _template_div_scalar_out,
        "div.Scalar_mode": _template_div_scalar_mode,
        "div.Scalar_mode_out": lambda: (
            _template_div_scalar_mode()[0],
            {
                "out": torch.empty_like(_template_div_scalar_mode()[0][0]),
                "rounding_mode": None,
            },
        ),
        "div.Tensor_mode": _template_div_tensor_mode,
        "div.out_mode": _template_div_tensor_mode_out,
        "div_.Tensor_mode": _template_div_tensor_mode,
        "div_.Scalar": _template_div_scalar,
        "div_.Scalar_mode": _template_div_scalar_mode,
        "divide.Scalar": _template_div_scalar,
        "divide.Tensor_mode": _template_div_tensor_mode,
        "divide.Scalar_mode": _template_div_scalar_mode,
        "divide.out_mode": _template_div_tensor_mode_out,
        "divide_.Tensor_mode": _template_div_tensor_mode,
        "divide_.Scalar_mode": _template_div_scalar_mode,
        "divide_.Scalar": _template_div_scalar,
        "div.default": lambda: ([1.0, 1.0], {}),
        "embedding.default": _template_embedding,
        "embedding.out": _template_embedding_out,
        "embedding_dense_backward.default": _template_embedding_dense_backward,
        "embedding_dense_backward.out": _template_embedding_dense_backward_out,
        "empty.memory_format": _template_empty_sizes,
        "empty.out": _template_empty_out,
        "empty_permuted.default": _template_empty_permuted,
        "empty_permuted.out": _template_empty_permuted_out,
        "empty_strided.default": _template_empty_strided,
        "empty_strided.out": _template_empty_strided_out,
        "empty_like.default": _template_empty_like,
        "empty_like.out": _skip("dynamo_fake_tensor_out_overload_bug"),
        "eq.int_list": _template_eq_int_list,
        "eq.float_list": _template_eq_float_list,
        "eq.bool_list": _template_eq_bool_list,
        "eq.Tensor_list": _template_eq_tensor_list,
        "eq.Scalar": _template_eq_scalar,
        "eq.Scalar_out": _template_eq_scalar_out,
        "eq.default": lambda: ([1.0, 1.0], {}),
        "eq.str": lambda: (["a", "a"], {}),
        "eq.str_list": lambda: ([["a"], ["a"]], {}),
        "eq.enum": _skip("qscheme_unavailable"),
        "eq_.Scalar": lambda: (
            [torch.tensor([1.0], dtype=torch.float32), 1.0],
            {},
        ),
        "expand.default": _template_expand,
        "expand_copy.default": _template_expand,
        "expand_copy.out": _template_expand_out,
        "erf.Scalar": lambda: ([0.5], {}),
        "erfc.Scalar": lambda: ([0.5], {}),
        "exp.Scalar": lambda: ([0.5], {}),
        "expm1.Scalar": lambda: ([0.5], {}),
        "floor.Scalar": _template_floor_scalar,
        "floor_divide.Scalar": _template_floor_divide_scalar,
        "floor_divide.Scalar_out": _template_floor_divide_scalar_out,
        "floor_divide_.Scalar": _template_floor_divide_scalar,
        "elu_backward.default": _template_elu_backward,
        "elu_backward.grad_input": _template_elu_backward_out,
        "exponential.default": _skip("randop_not_supported"),
        "exponential.out": _skip("randop_not_supported"),
        "exponential_.default": _skip("randop_not_supported"),
        "fill.Tensor": _template_fill_tensor,
        "fill.Tensor_out": _template_fill_tensor_out,
        "fill_.Tensor": _template_fill_tensor_inplace,
        "fill.Scalar": _template_fill_scalar,
        "fill.Scalar_out": _template_fill_scalar_out,
        "fill_.Scalar": _template_fill_scalar_inplace,
        "flip.default": _template_flip,
        "flip.out": _template_flip_out,
        "fmod.Scalar": _template_fmod_scalar,
        # float_power_ outputs require float64 dtype, which the backend doesn't support yet.
        "float_power_.Tensor": _skip("float64_dtype_not_supported"),
        "float_power_.Scalar": _skip("float64_dtype_not_supported"),
        # FFT ops require complex dtype, which the backend doesn't support yet.
        "fft_fft.default": _skip("complex_dtype_not_supported"),
        "fft_fft.out": _skip("complex_dtype_not_supported"),
        "fft_fft2.default": _skip("complex_dtype_not_supported"),
        "fft_fft2.out": _skip("complex_dtype_not_supported"),
        "fft_fftn.default": _skip("complex_dtype_not_supported"),
        "fft_fftn.out": _skip("complex_dtype_not_supported"),
        "fft_hfft.default": _skip("complex_dtype_not_supported"),
        "fft_hfft.out": _skip("complex_dtype_not_supported"),
        "fft_hfft2.default": _skip("complex_dtype_not_supported"),
        "fft_hfft2.out": _skip("complex_dtype_not_supported"),
        "fft_hfftn.default": _skip("complex_dtype_not_supported"),
        "fft_hfftn.out": _skip("complex_dtype_not_supported"),
        "fft_ifft.default": _skip("complex_dtype_not_supported"),
        "fft_ifft.out": _skip("complex_dtype_not_supported"),
        "fft_ifft2.default": _skip("complex_dtype_not_supported"),
        "fft_ifft2.out": _skip("complex_dtype_not_supported"),
        "fft_ifftn.default": _skip("complex_dtype_not_supported"),
        "fft_ifftn.out": _skip("complex_dtype_not_supported"),
        "fft_ihfft.default": _skip("complex_dtype_not_supported"),
        "fft_ihfft.out": _skip("complex_dtype_not_supported"),
        "fft_ihfft2.default": _skip("complex_dtype_not_supported"),
        "fft_ihfft2.out": _skip("complex_dtype_not_supported"),
        "fft_ihfftn.default": _skip("complex_dtype_not_supported"),
        "fft_ihfftn.out": _skip("complex_dtype_not_supported"),
        "fft_irfft.default": _skip("complex_dtype_not_supported"),
        "fft_irfft.out": _skip("complex_dtype_not_supported"),
        "fft_irfft2.default": _skip("complex_dtype_not_supported"),
        "fft_irfft2.out": _skip("complex_dtype_not_supported"),
        "fft_irfftn.default": _skip("complex_dtype_not_supported"),
        "fft_irfftn.out": _skip("complex_dtype_not_supported"),
        "fft_rfft.default": _skip("complex_dtype_not_supported"),
        "fft_rfft.out": _skip("complex_dtype_not_supported"),
        "fft_rfft2.default": _skip("complex_dtype_not_supported"),
        "fft_rfft2.out": _skip("complex_dtype_not_supported"),
        "fft_rfftn.default": _skip("complex_dtype_not_supported"),
        "fft_rfftn.out": _skip("complex_dtype_not_supported"),
        "empty.names": _skip("named_tensor_torchscript"),
        "empty.names_out": _skip("named_tensor_torchscript"),
    }
)

# Edit the OPS list to add/remove ops in this batch (format: "op.overload").
OPS = [
    "diag.default",
    "diag.out",
    "diag_embed.default",
    "diag_embed.out",
    "diagonal.default",
    "diagonal.Dimname",
    "diagonal_backward.default",
    "diagonal_backward.out",
    "diagonal_copy.default",
    "diagonal_copy.out",
    "diagonal_scatter.default",
    "diagonal_scatter.out",
    "digamma.default",
    "digamma.out",
    "digamma_.default",
    "dim.default",
    "dist.default",
    "dist.out",
    "div.Tensor",
    "div.Scalar",
    "div.Tensor_mode",
    "div.Scalar_mode",
    "div.out",
    "div.out_mode",
    "div.Scalar_out",
    "div.Scalar_mode_out",
    "div.int",
    "div.complex",
    "div.float",
    "div.default",
    "div_.Tensor",
    "div_.Tensor_mode",
    "div_.Scalar",
    "div_.Scalar_mode",
    "divide.Tensor",
    "divide.Scalar",
    "divide.Tensor_mode",
    "divide.Scalar_mode",
    "divide.out",
    "divide.out_mode",
    "divide_.Tensor",
    "divide_.Tensor_mode",
    "divide_.Scalar_mode",
    "divide_.Scalar",
    "dot.default",
    "dot.out",
    "dropout.default",
    "elu.default",
    "elu.out",
    "elu_.default",
    "elu_backward.default",
    "elu_backward.grad_input",
    "embedding.default",
    "embedding.out",
    "embedding_dense_backward.default",
    "embedding_dense_backward.out",
    "empty.memory_format",
    "empty.out",
    "empty.names",
    "empty.names_out",
    "empty_like.default",
    "empty_like.out",
    "empty_permuted.default",
    "empty_permuted.out",
    "empty_strided.default",
    "empty_strided.out",
    "eq.Tensor",
    "eq.Scalar",
    "eq.Scalar_out",
    "eq.Tensor_out",
    "eq.int_list",
    "eq.device",
    "eq.bool",
    "eq.enum",
    "eq.int",
    "eq.complex",
    "eq.float",
    "eq.int_float",
    "eq.float_int",
    "eq.float_complex",
    "eq.complex_float",
    "eq.default",
    "eq.str",
    "eq.float_list",
    "eq.Tensor_list",
    "eq.bool_list",
    "eq.str_list",
    "eq_.Scalar",
    "eq_.Tensor",
    "erf.default",
    "erf.out",
    "erf.int",
    "erf.float",
    "erf.Scalar",
    "erf_.default",
    "erfc.default",
    "erfc.out",
    "erfc.int",
    "erfc.float",
    "erfc.Scalar",
    "erfc_.default",
    "erfinv.default",
    "erfinv.out",
    "erfinv_.default",
    "exp.default",
    "exp.out",
    "exp.int",
    "exp.float",
    "exp.complex",
    "exp.Scalar",
    "exp2.default",
    "exp2.out",
    "exp2_.default",
    "exp_.default",
    "expand.default",
    "expand_copy.default",
    "expand_copy.out",
    "expm1.default",
    "expm1.out",
    "expm1.int",
    "expm1.float",
    "expm1.Scalar",
    "expm1_.default",
    "exponential.default",
    "exponential.out",
    "exponential_.default",
    "eye.default",
    "eye.m",
    "eye.out",
    "eye.m_out",
    "fft_fft.default",
    "fft_fft.out",
    "fft_fft2.default",
    "fft_fft2.out",
    "fft_fftn.default",
    "fft_fftn.out",
    "fft_fftshift.default",
    "fft_hfft.default",
    "fft_hfft.out",
    "fft_hfft2.default",
    "fft_hfft2.out",
    "fft_hfftn.default",
    "fft_hfftn.out",
    "fft_ifft.default",
    "fft_ifft.out",
    "fft_ifft2.default",
    "fft_ifft2.out",
    "fft_ifftn.default",
    "fft_ifftn.out",
    "fft_ifftshift.default",
    "fft_ihfft.default",
    "fft_ihfft.out",
    "fft_ihfft2.default",
    "fft_ihfft2.out",
    "fft_ihfftn.default",
    "fft_ihfftn.out",
    "fft_irfft.default",
    "fft_irfft.out",
    "fft_irfft2.default",
    "fft_irfft2.out",
    "fft_irfftn.default",
    "fft_irfftn.out",
    "fft_rfft.default",
    "fft_rfft.out",
    "fft_rfft2.default",
    "fft_rfft2.out",
    "fft_rfftn.default",
    "fft_rfftn.out",
    "fill.Scalar",
    "fill.Scalar_out",
    "fill.Tensor",
    "fill.Tensor_out",
    "fill_.Scalar",
    "fill_.Tensor",
    "fix.default",
    "fix.out",
    "fix_.default",
    "flip.default",
    "flip.out",
    "float_power_.Tensor",
    "float_power_.Scalar",
    "floor.default",
    "floor.out",
    "floor.int",
    "floor.float",
    "floor.Scalar",
    "floor_.default",
    "floor_divide.default",
    "floor_divide.Scalar",
    "floor_divide.out",
    "floor_divide.Scalar_out",
    "floor_divide_.Scalar",
    "floor_divide_.Tensor",
    "fmax.default",
    "fmax.out",
    "fmin.default",
    "fmin.out",
    "fmod.Tensor",
    "fmod.Scalar",
    "fmod.Tensor_out",
]

if __name__ == "__main__":
    run_aten_op_batch(
        OPS,
        batch_label="test_batch_2",
        max_fails=20,
        templates=CUSTOM_TEMPLATES,
        show_skips=True,
    )
# CHECK: SUMMARY pass=
# CHECK-SAME: fail=0
