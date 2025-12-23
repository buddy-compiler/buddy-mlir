# RUN: %PYTHON %s 2>&1 | FileCheck %s
from aten_op_batch_runner import run_aten_op_batch
import torch


CUSTOM_TEMPLATES = {}


def _skip(reason: str):
    def fn():
        raise RuntimeError(reason)

    return fn


def _template_fmod_scalar_pair(a, b):
    return [a, b], {}


def _template_fmod_scalar_out():
    x = torch.tensor([5.0, 3.0], dtype=torch.float32)
    out = torch.empty_like(x)
    return [x, 2.0], {"out": out}


def _template_fmod_tensor_inplace():
    x = torch.tensor([5.0, 3.0], dtype=torch.float32)
    other = torch.tensor([2.0, 2.0], dtype=torch.float32)
    return [x, other], {}


def _template_fmod_scalar_inplace():
    x = torch.tensor([5.0, 3.0], dtype=torch.float32)
    return [x, 2.0], {}


def _template_fractional_max_pool2d():
    x = torch.randn(1, 1, 4, 4, dtype=torch.float32)
    output_size = (2, 2)
    random_samples = torch.rand((1, 1, 2), dtype=torch.float32)
    return [x, (2, 2), output_size, random_samples], {}


def _template_fractional_max_pool2d_out():
    args, _ = _template_fractional_max_pool2d()
    out = torch.empty((1, 1, 2, 2), dtype=torch.float32)
    indices = torch.empty((1, 1, 2, 2), dtype=torch.int64)
    return args, {"output": out, "indices": indices}


def _template_frexp():
    x = torch.tensor([1.0, 2.0], dtype=torch.float32)
    return [x], {}


def _template_frexp_default():
    return [3.5], {}


def _template_frexp_out():
    args, _ = _template_frexp()
    mantissa = torch.empty_like(args[0])
    exponent = torch.empty_like(args[0], dtype=torch.int32)
    return args, {"mantissa": mantissa, "exponent": exponent}


def _template_full_names():
    size = [2]
    fill_value = 1.0
    # Use names=None to avoid the named tensor path.
    return [size, fill_value], {
        "names": None,
        "dtype": torch.float32,
        "device": torch.device("cpu"),
    }


def _template_full_names_out():
    args, _ = _template_full_names()
    out = torch.empty(args[0], dtype=torch.float32)
    return args, {"names": None, "out": out}


def _template_full():
    size = [2]
    fill_value = 1.0
    return [size, fill_value], {
        "dtype": torch.float32,
        "device": torch.device("cpu"),
    }


def _template_full_out():
    args, _ = _template_full()
    out = torch.empty(args[0], dtype=torch.float32)
    return args, {"out": out}


def _template_full_like():
    x = torch.empty(2, dtype=torch.float32)
    return [x, 1.0], {}


def _template_full_like_out():
    args, _ = _template_full_like()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_gather():
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    index = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    return [x, 1, index], {}


def _template_gather_out():
    args, _ = _template_gather()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_gather_dimname():
    return _skip("named_tensor_torchscript")()


def _template_gather_dimname_out():
    return _skip("named_tensor_torchscript")()


def _template_gcd_int():
    return [6, 4], {}


def _template_gcd_default():
    a = torch.tensor([6, 12], dtype=torch.int64)
    b = torch.tensor([4, 8], dtype=torch.int64)
    return [a, b], {}


def _template_gcd_inplace():
    a, b = _template_gcd_default()[0]
    return [a.clone(), b], {}


def _template_gcd_out():
    args, _ = _template_gcd_default()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_geometric():
    x = torch.empty(2, dtype=torch.float32)
    return [x, 0.5], {}


def _template_geometric_out():
    args, _ = _template_geometric()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_glu():
    x = torch.randn(2, 2, dtype=torch.float32)
    return [x, 1], {}


def _template_glu_out():
    args, _ = _template_glu()
    out = torch.empty((2, 1), dtype=torch.float32)
    return args, {"out": out}


def _template_glu_backward():
    x = torch.randn(2, 2, dtype=torch.float32)
    grad_output = torch.ones((2, 1), dtype=torch.float32)
    dim = 1
    return [grad_output, x, dim], {}


def _template_glu_backward_out():
    args, _ = _template_glu_backward()
    out = torch.empty_like(args[1])
    return args, {"grad_input": out}


def _template_grid_sampler_2d():
    input = torch.randn(1, 1, 2, 2, dtype=torch.float32)
    grid = torch.zeros(1, 2, 2, 2, dtype=torch.float32)
    return [input, grid, 0, 0, False], {}


def _template_grid_sampler_2d_out():
    args, _ = _template_grid_sampler_2d()
    out = torch.empty((1, 1, 2, 2), dtype=torch.float32)
    return args, {"out": out}


def _template_grid_sampler_2d_backward():
    input = torch.randn(1, 1, 2, 2, dtype=torch.float32)
    grid = torch.zeros(1, 2, 2, 2, dtype=torch.float32)
    grad_output = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    return [grad_output, input, grid, 0, 0, False, [True, True]], {}


def _template_grid_sampler_2d_backward_out():
    args, _ = _template_grid_sampler_2d_backward()
    grad_input = torch.empty_like(args[1])
    grad_grid = torch.empty_like(args[2])
    return args, {"out0": grad_input, "out1": grad_grid}


def _template_grid_sampler_3d():
    input = torch.randn(1, 1, 2, 2, 2, dtype=torch.float32)
    grid = torch.zeros(1, 2, 2, 2, 3, dtype=torch.float32)
    return [input, grid, 0, 0, False], {}


def _template_grid_sampler_3d_out():
    args, _ = _template_grid_sampler_3d()
    out = torch.empty((1, 1, 2, 2, 2), dtype=torch.float32)
    return args, {"out": out}


def _template_grid_sampler_3d_backward():
    input = torch.randn(1, 1, 2, 2, 2, dtype=torch.float32)
    grid = torch.zeros(1, 2, 2, 2, 3, dtype=torch.float32)
    grad_output = torch.ones((1, 1, 2, 2, 2), dtype=torch.float32)
    return [grad_output, input, grid, 0, 0, False, [True, True]], {}


def _template_grid_sampler_3d_backward_out():
    args, _ = _template_grid_sampler_3d_backward()
    grad_input = torch.empty_like(args[1])
    grad_grid = torch.empty_like(args[2])
    return args, {"out0": grad_input, "out1": grad_grid}


def _template_is_finite(kind: str):
    if kind == "float":
        return [torch.tensor([0.0, float("inf")], dtype=torch.float32)], {}
    if kind == "complex":
        return [
            torch.tensor([0 + 0j, float("inf") + 0j], dtype=torch.complex64)
        ], {}
    return [torch.tensor([1.0, 2.0], dtype=torch.float32)], {}


def _template_isfinite_scalar(kind: str):
    if kind == "float":
        return [0.0], {}
    if kind == "complex":
        return [0.0 + 0.0j], {}
    return [0.0], {}


def _template_isinf(kind: str):
    if kind == "float":
        return [torch.tensor([float("inf"), 0.0], dtype=torch.float32)], {}
    if kind == "complex":
        return [
            torch.tensor([float("inf") + 0j, 0 + 0j], dtype=torch.complex64)
        ], {}
    return [torch.tensor([1.0, 2.0], dtype=torch.float32)], {}


def _template_isinf_scalar(kind: str):
    if kind == "float":
        return [float("inf")], {}
    if kind == "complex":
        return [complex(float("inf"), 0.0)], {}
    return [float("inf")], {}


def _template_isnan(kind: str):
    if kind == "float":
        return [torch.tensor([float("nan"), 0.0], dtype=torch.float32)], {}
    if kind == "complex":
        return [
            torch.tensor([float("nan") + 0j, 0 + 0j], dtype=torch.complex64)
        ], {}
    return [torch.tensor([1.0, 2.0], dtype=torch.float32)], {}


def _template_isnan_scalar(kind: str):
    if kind == "float":
        return [float("nan")], {}
    if kind == "complex":
        return [complex(float("nan"), 0.0)], {}
    return [float("nan")], {}


def _template_bool_out(base_fn):
    args, kwargs = base_fn()
    out = torch.empty_like(args[0], dtype=torch.bool)
    kwargs = kwargs.copy()
    kwargs["out"] = out
    return args, kwargs


def _template_pos_neg_inf(kind: str):
    if kind == "pos":
        return [torch.tensor([float("inf"), 0.0], dtype=torch.float32)], {}
    return [torch.tensor([-float("inf"), 0.0], dtype=torch.float32)], {}


def _template_cmp_tensor_tensor():
    a = torch.tensor([1.0, 2.0], dtype=torch.float32)
    b = torch.tensor([0.5, 2.5], dtype=torch.float32)
    return [a, b], {}


def _template_cmp_tensor_tensor_out():
    args, _ = _template_cmp_tensor_tensor()
    out = torch.empty_like(args[0], dtype=torch.bool)
    return args, {"out": out}


def _template_cmp_tensor_scalar():
    a = torch.tensor([1.0, 2.0], dtype=torch.float32)
    return [a, 1.5], {}


def _template_cmp_tensor_scalar_out():
    args, _ = _template_cmp_tensor_scalar()
    out = torch.empty_like(args[0], dtype=torch.bool)
    return args, {"out": out}


def _template_cmp_tensor_scalar_inplace():
    a = torch.tensor([1.0, 2.0], dtype=torch.float32)
    return [a, 1.5], {}


def _template_cmp_scalar_pair(a, b):
    return [a, b], {}


def _template_cmp_str():
    return ["a", "b"], {}


def _template_hardtanh_backward():
    grad_output = torch.ones(2, dtype=torch.float32)
    self = torch.tensor([0.5, -1.0], dtype=torch.float32)
    return [grad_output, self, -1.0, 1.0], {}


def _template_hardtanh_backward_out():
    args, _ = _template_hardtanh_backward()
    grad_input = torch.empty_like(args[1])
    return args, {"grad_input": grad_input}


def _template_isin_tensor_tensor():
    elements = torch.tensor([1, 2, 3], dtype=torch.int64)
    test_elements = torch.tensor([2, 4], dtype=torch.int64)
    return [elements, test_elements], {}


def _template_isin_tensor_tensor_out():
    args, _ = _template_isin_tensor_tensor()
    out = torch.empty_like(args[0], dtype=torch.bool)
    return args, {"out": out}


def _template_isin_tensor_scalar():
    elements = torch.tensor([1, 2, 3], dtype=torch.int64)
    return [elements, 2], {}


def _template_isin_tensor_scalar_out():
    args, _ = _template_isin_tensor_scalar()
    out = torch.empty_like(args[0], dtype=torch.bool)
    return args, {"out": out}


def _template_isin_scalar_tensor():
    element = 2
    test_elements = torch.tensor([1, 2, 3], dtype=torch.int64)
    return [element, test_elements], {}


def _template_isin_scalar_tensor_out():
    args, _ = _template_isin_scalar_tensor()
    out = torch.empty((), dtype=torch.bool)
    return args, {"out": out}


def _template_gru_params():
    mod = torch.nn.GRU(
        input_size=2, hidden_size=2, num_layers=1, batch_first=True
    )
    params = [p for p in mod.parameters()]
    hx = torch.zeros(1, 1, 2, dtype=torch.float32)
    return params, hx


def _template_gru_input():
    torch.manual_seed(0)
    params, hx = _template_gru_params()
    inp = torch.randn(1, 1, 2, dtype=torch.float32)
    return [inp, hx, params, True, 1, 0.0, False, False, True], {}


def _template_gru_data():
    torch.manual_seed(0)
    params, hx = _template_gru_params()
    inp = torch.randn(1, 1, 2, dtype=torch.float32)
    packed = torch.nn.utils.rnn.pack_padded_sequence(
        inp, lengths=torch.tensor([1]), batch_first=True, enforce_sorted=False
    )
    return [
        packed.data,
        packed.batch_sizes,
        hx,
        params,
        True,
        1,
        0.0,
        False,
        False,
    ], {}


def _template_im2col():
    x = torch.arange(1.0, 10.0, dtype=torch.float32).reshape(1, 1, 3, 3)
    kernel_size = (2, 2)
    dilation = (1, 1)
    padding = (0, 0)
    stride = (1, 1)
    return [x, kernel_size, dilation, padding, stride], {}


def _template_im2col_out():
    args, _ = _template_im2col()
    ref = torch.nn.functional.unfold(
        args[0],
        kernel_size=args[1],
        dilation=args[2],
        padding=args[3],
        stride=args[4],
    )
    out = torch.empty_like(ref)
    return args, {"out": out}


def _template_imag():
    x = torch.tensor([1 + 2j], dtype=torch.complex64)
    return [x], {}


def _template_index_tensor():
    x = torch.arange(1, 5, dtype=torch.float32).reshape(2, 2)
    idx0 = torch.tensor([0, 1], dtype=torch.int64)
    idx1 = torch.tensor([1, 0], dtype=torch.int64)
    return [x, [idx0, idx1]], {}


def _template_index_tensor_out():
    args, _ = _template_index_tensor()
    result = args[0][(args[1][0], args[1][1])]
    out = torch.empty_like(result)
    return args, {"out": out}


def _template_index_tensor_hacked():
    return _template_index_tensor()


def _template_index_list_int():
    return [[0, 1, 2], 1], {}


def _template_index_list_float():
    return [[0.5, 1.5], 1.5], {}


def _template_index_list_bool():
    return [[True, False], False], {}


def _index_base_tensors():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    idx_rows = torch.tensor([0, 1], dtype=torch.int64)
    idx_cols = torch.tensor([1, 0], dtype=torch.int64)
    return x, idx_rows, idx_cols


def _named_base_tensors():
    x = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32
    ).refine_names("N", "C")
    idx_rows = torch.tensor([0, 1], dtype=torch.int64).refine_names("N")
    idx_cols = torch.tensor([1, 0], dtype=torch.int64).refine_names("N")
    return x, idx_rows, idx_cols


def _template_index_list_tensor():
    return [[torch.tensor([1.0]), torch.tensor([2.0])], torch.tensor([2.0])], {}


def _template_index_add():
    x, idx_rows, _ = _index_base_tensors()
    source = torch.ones_like(x)
    return [x, 0, idx_rows, source], {}


def _template_index_add_out():
    args, _ = _template_index_add()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_index_add_inplace():
    x, idx_rows, _ = _index_base_tensors()
    source = torch.ones_like(x)
    return [x, 0, idx_rows, source], {}


def _template_index_add_dimname():
    return _skip("named_tensor_torchscript")()


def _template_index_copy():
    x, idx_rows, _ = _index_base_tensors()
    source = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
    return [x, 0, idx_rows, source], {}


def _template_index_copy_out():
    args, _ = _template_index_copy()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_index_copy_dimname():
    return _skip("named_tensor_torchscript")()


def _template_index_copy_dimname_out():
    return _skip("named_tensor_torchscript")()


def _template_index_fill_tensor():
    x, idx_rows, _ = _index_base_tensors()
    value = torch.tensor(5.0)
    return [x, 0, idx_rows, value], {}


def _template_index_fill_tensor_out():
    args, _ = _template_index_fill_tensor()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_index_fill_scalar():
    x, idx_rows, _ = _index_base_tensors()
    value = 5.0
    return [x, 0, idx_rows, value], {}


def _template_index_fill_scalar_out():
    args, _ = _template_index_fill_scalar()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_index_fill_dimname_scalar():
    return _skip("named_tensor_torchscript")()


def _template_index_fill_dimname_tensor():
    return _skip("named_tensor_torchscript")()


def _template_index_put():
    x, idx_rows, idx_cols = _index_base_tensors()
    values = torch.tensor([9.0, 8.0], dtype=torch.float32)
    return [x, [idx_rows, idx_cols], values, False], {}


def _template_index_put_out():
    args, _ = _template_index_put()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_index_put_hacked():
    x, idx_rows, idx_cols = _index_base_tensors()
    values = torch.tensor([9.0, 8.0], dtype=torch.float32)
    return [x, [idx_rows, idx_cols], values, False], {}


def _template_index_select():
    x, idx_rows, _ = _index_base_tensors()
    return [x, 0, idx_rows], {}


def _template_index_select_out():
    args, _ = _template_index_select()
    out_shape = (args[2].numel(), args[0].shape[1])
    out = torch.empty(out_shape, dtype=args[0].dtype)
    return args, {"out": out}


def _template_index_select_dimname():
    return _skip("named_tensor_torchscript")()


def _template_index_select_dimname_out():
    return _skip("named_tensor_torchscript")()


def _template_is_coalesced():
    indices = torch.tensor([[0, 1], [0, 0]], dtype=torch.int64)
    values = torch.tensor([1.0, 2.0], dtype=torch.float32)
    sparse = torch.sparse_coo_tensor(indices, values, (2, 1))
    return [sparse.coalesce()], {}


def _template_index_str():
    return ["hello", "lo", 0, -1], {}


def _template_index_list_str():
    return [["a", "b", "c"], "b"], {}


def _template_index_reduce():
    x, idx_rows, _ = _index_base_tensors()
    source = torch.ones_like(x)
    return [x, 0, idx_rows, source, "prod"], {"include_self": True}


def _template_index_reduce_out():
    args, kwargs = _template_index_reduce()
    out = torch.empty_like(args[0])
    return args, {"out": out, **kwargs}


def _template_index_reduce_inplace():
    args, kwargs = _template_index_reduce()
    return args, kwargs


def _template_istft():
    torch.manual_seed(0)
    waveform = torch.randn(4, dtype=torch.float32)
    n_fft = 4
    hop_length = 2
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        return_complex=True,
        center=False,
    )
    return [
        stft,
        n_fft,
        hop_length,
        n_fft,
        None,
        False,
        False,
        True,
        waveform.numel(),
        False,
    ], {}


# Populate ops that need special inputs.
CUSTOM_TEMPLATES.update(
    {
        "fmod.Scalar_out": _template_fmod_scalar_out,
        "fmod.int": lambda: _template_fmod_scalar_pair(5, 2),
        "fmod.float": lambda: _template_fmod_scalar_pair(5.0, 2.0),
        "fmod.int_float": lambda: _template_fmod_scalar_pair(5, 2.0),
        "fmod.float_int": lambda: _template_fmod_scalar_pair(5.0, 2),
        "fmod.default": lambda: _template_fmod_scalar_pair(5, 2),
        "fmod_.Tensor": _template_fmod_tensor_inplace,
        "fmod_.Scalar": _template_fmod_scalar_inplace,
        "fractional_max_pool2d.default": _template_fractional_max_pool2d,
        "fractional_max_pool2d.output": _template_fractional_max_pool2d_out,
        "frexp.Tensor": _template_frexp,
        "frexp.Tensor_out": _template_frexp_out,
        "frexp.default": _template_frexp_default,
        "full.names": _skip("named_tensor_torchscript"),
        "full.names_out": _skip("named_tensor_torchscript"),
        "full.default": _template_full,
        "full.out": _template_full_out,
        "full_like.default": _template_full_like,
        "full_like.out": _skip("dynamo_fake_tensor_out_overload_bug"),
        "gather.default": _template_gather,
        "gather.out": _template_gather_out,
        "gather.dimname": _template_gather_dimname,
        "gather.dimname_out": _template_gather_dimname_out,
        "gcd.default": _template_gcd_default,
        "gcd.out": _template_gcd_out,
        "gcd.int": _template_gcd_int,
        "gcd_.default": _template_gcd_inplace,
        "geometric.default": _skip("random_op_not_supported"),
        "geometric.out": _skip("random_op_not_supported"),
        "geometric_.default": _skip("random_op_not_supported"),
        "glu.default": _template_glu,
        "glu.out": _template_glu_out,
        "glu_backward.default": _template_glu_backward,
        "glu_backward.grad_input": _template_glu_backward_out,
        "grid_sampler_2d.default": _template_grid_sampler_2d,
        "grid_sampler_2d.out": _template_grid_sampler_2d_out,
        "grid_sampler_2d_backward.default": _skip("backward_not_supported"),
        "grid_sampler_2d_backward.out": _skip("backward_not_supported"),
        "grid_sampler_3d.default": _template_grid_sampler_3d,
        "grid_sampler_3d.out": _template_grid_sampler_3d_out,
        "grid_sampler_3d_backward.default": _skip("backward_not_supported"),
        "grid_sampler_3d_backward.out": _skip("backward_not_supported"),
        "gru.input": _template_gru_input,
        "gru.data": _skip("dynamo_packed_sequence"),
        "isfinite.default": lambda: _template_is_finite("default"),
        "isfinite.float": lambda: _template_isfinite_scalar("float"),
        "isfinite.complex": lambda: _template_isfinite_scalar("complex"),
        "isinf.default": lambda: _template_isinf("default"),
        "isinf.out": lambda: _template_bool_out(
            lambda: _template_isinf("default")
        ),
        "isinf.float": lambda: _template_isinf_scalar("float"),
        "isinf.complex": lambda: _template_isinf_scalar("complex"),
        "isnan.default": lambda: _template_isnan("default"),
        "isnan.out": lambda: _template_bool_out(
            lambda: _template_isnan("default")
        ),
        "isnan.float": lambda: _template_isnan_scalar("float"),
        "isnan.complex": lambda: _template_isnan_scalar("complex"),
        "isneginf.out": lambda: _template_bool_out(
            lambda: _template_pos_neg_inf("neg")
        ),
        "isposinf.out": lambda: _template_bool_out(
            lambda: _template_pos_neg_inf("pos")
        ),
        "ge.Scalar": _template_cmp_tensor_scalar,
        "ge.Scalar_out": _template_cmp_tensor_scalar_out,
        "ge.Tensor": _template_cmp_tensor_tensor,
        "ge.Tensor_out": _template_cmp_tensor_tensor_out,
        "ge.int": lambda: _template_cmp_scalar_pair(1, 0),
        "ge.float": lambda: _template_cmp_scalar_pair(1.0, 0.5),
        "ge.int_float": lambda: _template_cmp_scalar_pair(1, 0.5),
        "ge.float_int": lambda: _template_cmp_scalar_pair(1.0, 1),
        "ge.default": lambda: _template_cmp_scalar_pair(1.0, 1.0),
        "ge.str": _template_cmp_str,
        "ge_.Scalar": _template_cmp_tensor_scalar_inplace,
        "ge_.Tensor": _template_cmp_tensor_tensor,
        "greater.Tensor": _template_cmp_tensor_tensor,
        "greater.Scalar": _template_cmp_tensor_scalar,
        "greater.Scalar_out": _template_cmp_tensor_scalar_out,
        "greater.Tensor_out": _template_cmp_tensor_tensor_out,
        "greater_.Scalar": _template_cmp_tensor_scalar_inplace,
        "greater_.Tensor": _template_cmp_tensor_tensor,
        "greater_equal.Tensor": _template_cmp_tensor_tensor,
        "greater_equal.Scalar": _template_cmp_tensor_scalar,
        "greater_equal.Scalar_out": _template_cmp_tensor_scalar_out,
        "greater_equal.Tensor_out": _template_cmp_tensor_tensor_out,
        "greater_equal_.Scalar": _template_cmp_tensor_scalar_inplace,
        "greater_equal_.Tensor": _template_cmp_tensor_tensor,
        "gt.Tensor": _template_cmp_tensor_tensor,
        "gt.Scalar": _template_cmp_tensor_scalar,
        "gt.Scalar_out": _template_cmp_tensor_scalar_out,
        "gt.Tensor_out": _template_cmp_tensor_tensor_out,
        "gt.int": lambda: _template_cmp_scalar_pair(1, 0),
        "gt.float": lambda: _template_cmp_scalar_pair(1.0, 0.5),
        "gt.int_float": lambda: _template_cmp_scalar_pair(1, 0.5),
        "gt.float_int": lambda: _template_cmp_scalar_pair(1.0, 1),
        "gt.default": lambda: _template_cmp_scalar_pair(1.0, 1.0),
        "gt.str": _template_cmp_str,
        "gt_.Scalar": _template_cmp_tensor_scalar_inplace,
        "gt_.Tensor": _template_cmp_tensor_tensor,
        "im2col.default": _template_im2col,
        "im2col.out": _template_im2col_out,
        "imag.default": _skip("complex_dtype_not_supported"),
        "hardtanh_backward.default": _template_hardtanh_backward,
        "hardtanh_backward.grad_input": _template_hardtanh_backward_out,
        "index.Tensor": _template_index_tensor,
        "index.Tensor_out": _template_index_tensor_out,
        "index.Tensor_hacked_twin": _template_index_tensor_hacked,
        "index.list_int": _template_index_list_int,
        "index.list_float": _template_index_list_float,
        "index.list_bool": _template_index_list_bool,
        "index.list_Tensor": _template_index_list_tensor,
        "index_add.default": _template_index_add,
        "index_add.out": _template_index_add_out,
        "index_add_.default": _template_index_add_inplace,
        "index_add.dimname": _template_index_add_dimname,
        "index_copy.default": _template_index_copy,
        "index_copy.out": _template_index_copy_out,
        "index_copy_.default": _template_index_copy,
        "index_copy.dimname": _template_index_copy_dimname,
        "index_copy_.dimname": _template_index_copy_dimname,
        "index_copy.dimname_out": _template_index_copy_dimname_out,
        "index_fill.int_Tensor": _template_index_fill_tensor,
        "index_fill.int_Tensor_out": _template_index_fill_tensor_out,
        "index_fill_.int_Tensor": _template_index_fill_tensor,
        "index_fill.int_Scalar": _template_index_fill_scalar,
        "index_fill.int_Scalar_out": _template_index_fill_scalar_out,
        "index_fill_.int_Scalar": _template_index_fill_scalar,
        "index_fill.Dimname_Scalar": _template_index_fill_dimname_scalar,
        "index_fill.Dimname_Tensor": _template_index_fill_dimname_tensor,
        "index_fill_.Dimname_Scalar": _template_index_fill_dimname_scalar,
        "index_fill_.Dimname_Tensor": _template_index_fill_dimname_tensor,
        "index_put.default": _template_index_put,
        "index_put.out": _template_index_put_out,
        "index_put.hacked_twin": _template_index_put_hacked,
        "index_put_.default": _template_index_put,
        "index_put_.hacked_twin": _template_index_put_hacked,
        "index_select.default": _template_index_select,
        "index_select.out": _template_index_select_out,
        "index_select.dimname": _template_index_select_dimname,
        "index_select.dimname_out": _template_index_select_dimname_out,
        "index_reduce.default": _template_index_reduce,
        "index_reduce.out": _template_index_reduce_out,
        "index_reduce_.default": _template_index_reduce_inplace,
        "is_coalesced.default": _skip("sparse_not_supported"),
        "index.str": _template_index_str,
        "index.list_str": _template_index_list_str,
        "istft.default": _skip("complex_fft_not_supported"),
        "isin.Tensor_Tensor": _template_isin_tensor_tensor,
        "isin.Tensor_Tensor_out": _template_isin_tensor_tensor_out,
        "isin.Tensor_Scalar": _template_isin_tensor_scalar,
        "isin.Tensor_Scalar_out": _template_isin_tensor_scalar_out,
        "isin.Scalar_Tensor": _template_isin_scalar_tensor,
        "isin.Scalar_Tensor_out": _template_isin_scalar_tensor_out,
        # Skip inplace heaviside due to PyTorch/Dynamo functionalization limitation
        "heaviside_.default": _skip("dynamo_inplace_shape_mismatch"),
    }
)

# Edit the OPS list to add/remove ops in this batch (format: "op.overload").
OPS = [
    "fmod.Scalar_out",
    "fmod.int",
    "fmod.float",
    "fmod.int_float",
    "fmod.float_int",
    "fmod.default",
    "fmod_.Tensor",
    "fmod_.Scalar",
    "frac.default",
    "frac.out",
    "frac_.default",
    "fractional_max_pool2d.default",
    "fractional_max_pool2d.output",
    "frexp.Tensor",
    "frexp.Tensor_out",
    "frexp.default",
    "full.names",
    "full.default",
    "full.names_out",
    "full.out",
    "full_like.default",
    "full_like.out",
    "gather.default",
    "gather.out",
    "gather.dimname",
    "gather.dimname_out",
    "gcd.default",
    "gcd.out",
    "gcd.int",
    "gcd_.default",
    "ge.Tensor",
    "ge.Scalar",
    "ge.Scalar_out",
    "ge.Tensor_out",
    "ge.int",
    "ge.float",
    "ge.int_float",
    "ge.float_int",
    "ge.default",
    "ge.str",
    "ge_.Scalar",
    "ge_.Tensor",
    "gelu.default",
    "gelu.out",
    "gelu_.default",
    "gelu_backward.default",
    "gelu_backward.grad_input",
    "geometric.default",
    "geometric.out",
    "geometric_.default",
    "glu.default",
    "glu.out",
    "glu_backward.default",
    "glu_backward.grad_input",
    "greater.Tensor",
    "greater.Scalar",
    "greater.Scalar_out",
    "greater.Tensor_out",
    "greater_.Scalar",
    "greater_.Tensor",
    "greater_equal.Tensor",
    "greater_equal.Scalar",
    "greater_equal.Scalar_out",
    "greater_equal.Tensor_out",
    "greater_equal_.Scalar",
    "greater_equal_.Tensor",
    "grid_sampler_2d.default",
    "grid_sampler_2d.out",
    "grid_sampler_2d_backward.default",
    "grid_sampler_2d_backward.out",
    "grid_sampler_3d.default",
    "grid_sampler_3d.out",
    "grid_sampler_3d_backward.default",
    "grid_sampler_3d_backward.out",
    "gru.input",
    "gru.data",
    "gt.Tensor",
    "gt.Scalar",
    "gt.Scalar_out",
    "gt.Tensor_out",
    "gt.int",
    "gt.float",
    "gt.int_float",
    "gt.float_int",
    "gt.default",
    "gt.str",
    "gt_.Scalar",
    "gt_.Tensor",
    "hardshrink.default",
    "hardshrink.out",
    "hardsigmoid.default",
    "hardsigmoid.out",
    "hardsigmoid_.default",
    "hardsigmoid_backward.default",
    "hardsigmoid_backward.grad_input",
    "hardswish.default",
    "hardswish.out",
    "hardswish_.default",
    "hardswish_backward.default",
    "hardswish_backward.out",
    "hardtanh.default",
    "hardtanh.out",
    "hardtanh_.default",
    "hardtanh_backward.default",
    "hardtanh_backward.grad_input",
    "heaviside.default",
    "heaviside.out",
    "heaviside_.default",
    "hinge_embedding_loss.default",
    "histc.default",
    "histc.out",
    "huber_loss.default",
    "huber_loss.out",
    "huber_loss_backward.out",
    "huber_loss_backward.default",
    "hypot.default",
    "hypot.out",
    "hypot_.default",
    "i0.default",
    "i0.out",
    "i0_.default",
    "igamma.default",
    "igamma.out",
    "igamma_.default",
    "igammac.default",
    "igammac.out",
    "igammac_.default",
    "im2col.default",
    "im2col.out",
    "imag.default",
    "index.Tensor",
    "index.Tensor_out",
    "index.Tensor_hacked_twin",
    "index.str",
    "index.list_int",
    "index.list_float",
    "index.list_bool",
    "index.list_Tensor",
    "index.list_str",
    "index_add.default",
    "index_add.out",
    "index_add.dimname",
    "index_add_.default",
    "index_copy.default",
    "index_copy.dimname",
    "index_copy.out",
    "index_copy_.default",
    "index_copy_.dimname",
    "index_fill.int_Tensor",
    "index_fill.int_Scalar",
    "index_fill.Dimname_Scalar",
    "index_fill.Dimname_Tensor",
    "index_fill.int_Scalar_out",
    "index_fill.int_Tensor_out",
    "index_fill_.int_Tensor",
    "index_fill_.int_Scalar",
    "index_fill_.Dimname_Scalar",
    "index_fill_.Dimname_Tensor",
    "index_put.default",
    "index_put.out",
    "index_put.hacked_twin",
    "index_put_.default",
    "index_put_.hacked_twin",
    "index_reduce.default",
    "index_reduce.out",
    "index_reduce_.default",
    "index_select.default",
    "index_select.out",
    "index_select.dimname",
    "index_select.dimname_out",
    "is_coalesced.default",
    "is_complex.default",
    "is_contiguous.default",
    "is_contiguous.memory_format",
    "is_non_overlapping_and_dense.default",
    "is_pinned.default",
    "is_same_size.default",
    "is_strides_like_format.default",
    "isfinite.default",
    "isfinite.float",
    "isfinite.complex",
    "isin.Tensor_Tensor",
    "isin.Tensor_Tensor_out",
    "isin.Tensor_Scalar",
    "isin.Tensor_Scalar_out",
    "isin.Scalar_Tensor",
    "isin.Scalar_Tensor_out",
    "isinf.default",
    "isinf.out",
    "isinf.float",
    "isinf.complex",
    "isnan.default",
    "isnan.out",
    "isnan.float",
    "isnan.complex",
    "isneginf.default",
    "isneginf.out",
    "isposinf.default",
    "isposinf.out",
    "istft.default",
]

if __name__ == "__main__":
    run_aten_op_batch(
        OPS,
        batch_label="test_batch_3",
        max_fails=20,
        templates=CUSTOM_TEMPLATES,
        show_skips=True,
    )
# CHECK: SUMMARY pass=
# CHECK-SAME: fail=0
