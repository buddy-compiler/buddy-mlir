# RUN: %PYTHON %s 2>&1 | FileCheck %s
from aten_op_batch_runner import run_aten_op_batch
import torch

CUSTOM_TEMPLATES = {}


def _skip(reason: str):
    def fn():
        raise RuntimeError(reason)

    return fn


def _base_vec() -> torch.Tensor:
    return torch.tensor([1.0, 2.0], dtype=torch.float32)


def _base_mat() -> torch.Tensor:
    return torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)


def _base_square_2d() -> torch.Tensor:
    return torch.arange(4, dtype=torch.float32).reshape(2, 2)


def _template_tril_default():
    x = _base_square_2d()
    return [x], {}


def _template_tril_out():
    x = _base_square_2d()
    return [x], {"out": torch.empty_like(x)}


def _template_tril_inplace():
    x = _base_square_2d().clone()
    return [x], {}


def _template_triangular_solve_default():
    b = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
    A = torch.tensor([[2.0, 1.0], [0.0, 3.0]], dtype=torch.float32)
    return [b, A], {}


def _template_triangular_solve_X():
    b, A = _template_triangular_solve_default()[0]
    X = torch.empty_like(b)
    M = torch.empty_like(A)
    return [b, A], {"X": X, "M": M}


def _num_tril_indices(row: int, col: int, offset: int) -> int:
    n = 0
    for i in range(row):
        upper = i + offset + 1
        if upper <= 0:
            continue
        n += min(col, upper)
    return n


def _num_triu_indices(row: int, col: int, offset: int) -> int:
    n = 0
    for i in range(row):
        start = i + offset
        if start < 0:
            start = 0
        if start < col:
            n += col - start
    return n


def _template_tril_indices_default():
    return [3, 4, 0], {}


def _template_tril_indices_out():
    row, col, offset = 3, 4, 0
    n = _num_tril_indices(row, col, offset)
    out = torch.empty((2, n), dtype=torch.int64)
    return [row, col, offset], {"out": out}


def _template_triu_indices_default():
    return [3, 4, 0], {}


def _template_triu_indices_out():
    row, col, offset = 3, 4, 0
    n = _num_triu_indices(row, col, offset)
    out = torch.empty((2, n), dtype=torch.int64)
    return [row, col, offset], {"out": out}


def _template_unbind_int():
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    return [x, 0], {}


def _template_unfold_default():
    x = torch.arange(6, dtype=torch.float32)
    return [x, 0, 3, 1], {}


def _template_unfold_copy_out():
    x = torch.arange(6, dtype=torch.float32)
    out = torch.empty((4, 3), dtype=x.dtype)
    return [x, 0, 3, 1], {"out": out}


def _template_unfold_backward_default():
    grad_in = torch.ones((4, 3), dtype=torch.float32)
    input_sizes = [6]
    return [grad_in, input_sizes, 0, 3, 1], {}


def _template_unfold_backward_out():
    args, _ = _template_unfold_backward_default()
    out = torch.empty((6,), dtype=torch.float32)
    return args, {"out": out}


def _template_unsafe_chunk_default():
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    return [x, 2, 1], {}


def _template_unsafe_split_tensor():
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    return [x, 2, 1], {}


def _template_unsafe_split_tensor_out():
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    out = [
        torch.empty((3, 2), dtype=x.dtype),
        torch.empty((3, 2), dtype=x.dtype),
    ]
    return [x, 2, 1], {"out": out}


def _template_unsafe_split_with_sizes_default():
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    return [x, [1, 3], 1], {}


def _template_unsafe_split_with_sizes_out():
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    out = [
        torch.empty((3, 1), dtype=x.dtype),
        torch.empty((3, 3), dtype=x.dtype),
    ]
    return [x, [1, 3], 1], {"out": out}


def _upsample_input_1d() -> torch.Tensor:
    return torch.arange(4, dtype=torch.float32).reshape(1, 1, 4)


def _upsample_input_2d() -> torch.Tensor:
    return torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)


def _upsample_input_3d() -> torch.Tensor:
    return torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 2, 2)


def _template_upsample_bicubic2d_default():
    x = _upsample_input_2d()
    return [x, [4, 4], False], {}


def _template_upsample_bicubic2d_vec():
    x = _upsample_input_2d()
    return [x, [4, 4], False, None], {}


def _template_upsample_bicubic2d_out():
    x = _upsample_input_2d()
    out = torch.empty((1, 1, 4, 4), dtype=x.dtype)
    return [x, [4, 4], False], {"out": out}


def _template_upsample_bilinear2d_default():
    x = _upsample_input_2d()
    return [x, [4, 4], False], {}


def _template_upsample_bilinear2d_vec():
    x = _upsample_input_2d()
    return [x, [4, 4], False, None], {}


def _template_upsample_bilinear2d_out():
    x = _upsample_input_2d()
    out = torch.empty((1, 1, 4, 4), dtype=x.dtype)
    return [x, [4, 4], False], {"out": out}


def _template_upsample_bilinear2d_vec_out():
    x = _upsample_input_2d()
    out = torch.empty((1, 1, 4, 4), dtype=x.dtype)
    return [x, [4, 4], False, None], {"out": out}


def _template_upsample_linear1d_default():
    x = _upsample_input_1d()
    return [x, [8], False], {}


def _template_upsample_linear1d_vec():
    x = _upsample_input_1d()
    return [x, [8], False, None], {}


def _template_upsample_linear1d_out():
    x = _upsample_input_1d()
    out = torch.empty((1, 1, 8), dtype=x.dtype)
    return [x, [8], False], {"out": out}


def _template_upsample_nearest1d_default():
    x = _upsample_input_1d()
    return [x, [8]], {}


def _template_upsample_nearest1d_vec():
    x = _upsample_input_1d()
    return [x, [8], None], {}


def _template_upsample_nearest1d_out():
    x = _upsample_input_1d()
    out = torch.empty((1, 1, 8), dtype=x.dtype)
    return [x, [8]], {"out": out}


def _template_upsample_nearest2d_default():
    x = _upsample_input_2d()
    return [x, [4, 4]], {}


def _template_upsample_nearest2d_vec():
    x = _upsample_input_2d()
    return [x, [4, 4], None], {}


def _template_upsample_nearest2d_out():
    x = _upsample_input_2d()
    out = torch.empty((1, 1, 4, 4), dtype=x.dtype)
    return [x, [4, 4]], {"out": out}


def _template_upsample_nearest2d_vec_out():
    x = _upsample_input_2d()
    out = torch.empty((1, 1, 4, 4), dtype=x.dtype)
    return [x, [4, 4], None], {"out": out}


def _template_upsample_nearest2d_backward_default():
    grad_out = torch.ones((1, 1, 4, 4), dtype=torch.float32)
    return [grad_out, [4, 4], [1, 1, 2, 2]], {}


def _template_upsample_nearest2d_backward_grad_input():
    args, _ = _template_upsample_nearest2d_backward_default()
    grad_input = torch.empty((1, 1, 2, 2), dtype=torch.float32)
    return args, {"grad_input": grad_input}


def _template_upsample_nearest3d_default():
    x = _upsample_input_3d()
    return [x, [4, 4, 4]], {}


def _template_upsample_nearest3d_vec():
    x = _upsample_input_3d()
    return [x, [4, 4, 4], None], {}


def _template_upsample_nearest3d_out():
    x = _upsample_input_3d()
    out = torch.empty((1, 1, 4, 4, 4), dtype=x.dtype)
    return [x, [4, 4, 4]], {"out": out}


def _template_upsample_trilinear3d_default():
    x = _upsample_input_3d()
    return [x, [4, 4, 4], False], {}


def _template_upsample_trilinear3d_vec():
    x = _upsample_input_3d()
    return [x, [4, 4, 4], False, None], {}


def _template_upsample_trilinear3d_out():
    x = _upsample_input_3d()
    out = torch.empty((1, 1, 4, 4, 4), dtype=x.dtype)
    return [x, [4, 4, 4], False], {"out": out}


def _template_var_default():
    x = _base_vec()
    return [x], {}


def _template_var_dim():
    x = _base_mat()
    return [x, [0]], {}


def _template_var_correction():
    x = _base_vec()
    return [x, None], {"correction": 0}


def _template_var_out():
    x = _base_mat()
    out = torch.empty((x.shape[1],), dtype=x.dtype)
    return [x, [0]], {"out": out}


def _template_var_correction_out():
    x = _base_vec()
    out = torch.empty((), dtype=x.dtype)
    return [x, None], {"correction": 0, "out": out}


def _template_var_mean_default():
    x = _base_vec()
    return [x], {}


def _template_var_mean_dim():
    x = _base_mat()
    return [x, [0]], {}


def _template_var_mean_correction():
    x = _base_vec()
    return [x, None], {"correction": 0}


def _template_var_mean_correction_out():
    x = _base_vec()
    out0 = torch.empty((), dtype=x.dtype)
    out1 = torch.empty((), dtype=x.dtype)
    return [x, None], {"correction": 0, "out0": out0, "out1": out1}


def _template_unique_consecutive_out():
    x = torch.tensor([1.0, 1.0, 2.0, 3.0], dtype=torch.float32)
    out0 = torch.empty((0,), dtype=x.dtype)
    out1 = torch.empty((0,), dtype=torch.int64)
    out2 = torch.empty((0,), dtype=torch.int64)
    return [x, True, True, None], {"out0": out0, "out1": out1, "out2": out2}


def _template_unique_dim_out():
    x = torch.tensor([[1.0, 1.0], [2.0, 3.0]], dtype=torch.float32)
    out0 = torch.empty((0,), dtype=x.dtype)
    out1 = torch.empty((0,), dtype=torch.int64)
    out2 = torch.empty((0,), dtype=torch.int64)
    return [x, 0, True, True, True], {"out0": out0, "out1": out1, "out2": out2}


def _template_view_default():
    x = torch.arange(6, dtype=torch.float32)
    return [x, [2, 3]], {}


def _template_view_dtype():
    x = torch.arange(6, dtype=torch.float32)
    return [x, torch.float32], {}


def _template_view_as_complex():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    return [x], {}


def _template_view_as_real():
    x = torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex64)
    return [x], {}


def _template_view_copy_default():
    x = torch.arange(6, dtype=torch.float32)
    return [x, [2, 3]], {}


def _template_view_copy_dtype():
    x = torch.arange(6, dtype=torch.float32)
    return [x, torch.float32], {}


def _template_view_copy_out():
    x = torch.arange(6, dtype=torch.float32)
    out = torch.empty((2, 3), dtype=x.dtype)
    return [x, [2, 3]], {"out": out}


def _template_view_copy_dtype_out():
    x = torch.arange(6, dtype=torch.float32)
    out = torch.empty_like(x, dtype=torch.float32)
    return [x, torch.float32], {"out": out}


def _template_where_self():
    cond = torch.tensor([True, False, True], dtype=torch.bool)
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    y = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
    return [cond, x, y], {}


def _template_where_scalar_other():
    cond = torch.tensor([True, False, True], dtype=torch.bool)
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    return [cond, x, 2.0], {}


def _template_where_scalar_self():
    cond = torch.tensor([True, False, True], dtype=torch.bool)
    y = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
    return [cond, 2.0, y], {}


def _template_where_scalar():
    cond = torch.tensor([True, False, True], dtype=torch.bool)
    return [cond, 2.0, 3.0], {}


def _template_where_self_out():
    args, _ = _template_where_self()
    out = torch.empty_like(args[1])
    return args, {"out": out}


def _template_xlogy_scalar_other():
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    return [x, 2.0], {}


def _template_xlogy_scalar_self():
    y = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    return [2.0, y], {}


def _template_xlogy_out_scalar_self():
    args, _ = _template_xlogy_scalar_self()
    out = torch.empty_like(args[1])
    return args, {"out": out}


def _template_xlogy_out_scalar_other():
    args, _ = _template_xlogy_scalar_other()
    out = torch.empty_like(args[0])
    return args, {"out": out}


def _template_xlogy_inplace_scalar_other():
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    return [x, 2.0], {}


# Edit the OPS list to add/remove ops in this batch (format: "op.overload").
OPS = [
    "transpose_copy.int",
    "transpose_copy.int_out",
    "triangular_solve.default",
    "triangular_solve.X",
    "tril.default",
    "tril.out",
    "tril_.default",
    "tril_indices.default",
    "tril_indices.out",
    "triu.default",
    "triu.out",
    "triu_.default",
    "triu_indices.default",
    "triu_indices.out",
    "true_divide.Tensor",
    "true_divide.Scalar",
    "true_divide.out",
    "true_divide_.Tensor",
    "true_divide_.Scalar",
    "trunc.default",
    "trunc.out",
    "trunc_.default",
    "unbind.int",
    "unbind.Dimname",
    "unbind_copy.int",
    "unbind_copy.int_out",
    "unfold.default",
    "unfold_backward.default",
    "unfold_backward.out",
    "unfold_copy.default",
    "unfold_copy.out",
    "uniform.default",
    "uniform.out",
    "uniform_.default",
    "unique_consecutive.default",
    "unique_consecutive.out",
    "unique_dim.default",
    "unique_dim.out",
    "unsafe_chunk.default",
    "unsafe_split.Tensor",
    "unsafe_split.Tensor_out",
    "unsafe_split_with_sizes.default",
    "unsafe_split_with_sizes.out",
    "unsqueeze.default",
    "unsqueeze_.default",
    "unsqueeze_copy.default",
    "unsqueeze_copy.out",
    "upsample_bicubic2d.default",
    "upsample_bicubic2d.vec",
    "upsample_bicubic2d.out",
    "upsample_bilinear2d.default",
    "upsample_bilinear2d.vec",
    "upsample_bilinear2d.out",
    "upsample_bilinear2d.vec_out",
    "upsample_linear1d.default",
    "upsample_linear1d.vec",
    "upsample_linear1d.out",
    "upsample_nearest1d.default",
    "upsample_nearest1d.vec",
    "upsample_nearest1d.out",
    "upsample_nearest2d.default",
    "upsample_nearest2d.vec",
    "upsample_nearest2d.out",
    "upsample_nearest2d.vec_out",
    "upsample_nearest2d_backward.default",
    "upsample_nearest2d_backward.grad_input",
    "upsample_nearest3d.default",
    "upsample_nearest3d.vec",
    "upsample_nearest3d.out",
    "upsample_trilinear3d.default",
    "upsample_trilinear3d.vec",
    "upsample_trilinear3d.out",
    "var.default",
    "var.dim",
    "var.correction",
    "var.names_dim",
    "var.names_out",
    "var.out",
    "var.correction_out",
    "var.correction_names",
    "var.correction_names_out",
    "var_mean.default",
    "var_mean.dim",
    "var_mean.correction",
    "var_mean.names_dim",
    "var_mean.correction_names",
    "var_mean.correction_out",
    "vdot.default",
    "vdot.out",
    "view.default",
    "view.dtype",
    "view_as_complex.default",
    "view_as_real.default",
    "view_copy.default",
    "view_copy.dtype",
    "view_copy.out",
    "view_copy.dtype_out",
    "where.self",
    "where.ScalarOther",
    "where.ScalarSelf",
    "where.Scalar",
    "where.default",
    "where.self_out",
    "xlogy.Tensor",
    "xlogy.Scalar_Other",
    "xlogy.Scalar_Self",
    "xlogy.OutTensor",
    "xlogy.OutScalar_Self",
    "xlogy.OutScalar_Other",
    "xlogy_.Tensor",
    "xlogy_.Scalar_Other",
    "zero.default",
    "zero.out",
    "zero_.default",
    "zeros.names",
    "zeros.default",
    "zeros.names_out",
    "zeros.out",
    "zeros_like.default",
    "zeros_like.out",
]

if __name__ == "__main__":
    CUSTOM_TEMPLATES.update(
        {
            "triangular_solve.default": _skip("multi_output_op"),
            "triangular_solve.X": _skip("multi_output_op"),
            "tril.default": _template_tril_default,
            "tril.out": _template_tril_out,
            "tril_.default": _template_tril_inplace,
            "tril_indices.default": _template_tril_indices_default,
            "tril_indices.out": _template_tril_indices_out,
            "triu.default": _template_tril_default,
            "triu.out": _template_tril_out,
            "triu_.default": _template_tril_inplace,
            "triu_indices.default": _template_triu_indices_default,
            "triu_indices.out": _template_triu_indices_out,
            "unbind.int": _template_unbind_int,
            "unbind.Dimname": _skip("dynamo_dimname_fake_tensor"),
            "unbind_copy.int": _template_unbind_int,
            "unbind_copy.int_out": _skip("dynamo_fake_tensor_list_out"),
            "unfold.default": _template_unfold_default,
            "unfold_copy.default": _template_unfold_default,
            "unfold_copy.out": _template_unfold_copy_out,
            "unfold_backward.default": _template_unfold_backward_default,
            "unfold_backward.out": _template_unfold_backward_out,
            "unique_consecutive.default": _skip("data_dependent_output"),
            "unique_consecutive.out": _skip("data_dependent_output"),
            "unique_dim.default": _skip("data_dependent_output"),
            "unique_dim.out": _skip("data_dependent_output"),
            "where.default": _skip("data_dependent_output"),
            "zeros.names": _skip("named_tensor_torchscript"),
            "zeros.names_out": _skip("named_tensor_torchscript"),
            "unsafe_chunk.default": _template_unsafe_chunk_default,
            "unsafe_split.Tensor": _template_unsafe_split_tensor,
            "unsafe_split.Tensor_out": _template_unsafe_split_tensor_out,
            "unsafe_split_with_sizes.default": _template_unsafe_split_with_sizes_default,
            "unsafe_split_with_sizes.out": _template_unsafe_split_with_sizes_out,
            "unsqueeze_.default": _skip("dynamo_inplace_view_op"),
            "upsample_bicubic2d.default": _template_upsample_bicubic2d_default,
            "upsample_bicubic2d.vec": _template_upsample_bicubic2d_vec,
            "upsample_bicubic2d.out": _skip("out_variant"),
            "upsample_bilinear2d.default": _template_upsample_bilinear2d_default,
            "upsample_bilinear2d.vec": _template_upsample_bilinear2d_vec,
            "upsample_bilinear2d.out": _skip("out_variant"),
            "upsample_bilinear2d.vec_out": _skip("out_variant"),
            "upsample_linear1d.default": _template_upsample_linear1d_default,
            "upsample_linear1d.vec": _template_upsample_linear1d_vec,
            "upsample_linear1d.out": _template_upsample_linear1d_out,
            "upsample_nearest1d.default": _template_upsample_nearest1d_default,
            "upsample_nearest1d.vec": _template_upsample_nearest1d_vec,
            "upsample_nearest1d.out": _template_upsample_nearest1d_out,
            "upsample_nearest2d.default": _template_upsample_nearest2d_default,
            "upsample_nearest2d.vec": _template_upsample_nearest2d_vec,
            "upsample_nearest2d.out": _template_upsample_nearest2d_out,
            "upsample_nearest2d.vec_out": _template_upsample_nearest2d_vec_out,
            "upsample_nearest2d_backward.default": _skip("backward_op"),
            "upsample_nearest2d_backward.grad_input": _skip("backward_op"),
            "upsample_nearest3d.default": _template_upsample_nearest3d_default,
            "upsample_nearest3d.vec": _template_upsample_nearest3d_vec,
            "upsample_nearest3d.out": _template_upsample_nearest3d_out,
            "upsample_trilinear3d.default": _skip(
                "tosa_resize_5d_not_supported"
            ),
            "upsample_trilinear3d.vec": _skip("tosa_resize_5d_not_supported"),
            "upsample_trilinear3d.out": _skip("tosa_resize_5d_not_supported"),
            "var.default": _template_var_default,
            "var.dim": _template_var_dim,
            "var.correction": _template_var_correction,
            "var.names_dim": _skip("dynamo_dimname_fake_tensor"),
            "var.names_out": _skip("dynamo_dimname_fake_tensor"),
            "var.out": _template_var_out,
            "var.correction_out": _template_var_correction_out,
            "var.correction_names": _skip("dynamo_dimname_fake_tensor"),
            "var.correction_names_out": _skip("dynamo_dimname_fake_tensor"),
            "var_mean.default": _template_var_mean_default,
            "var_mean.dim": _template_var_mean_dim,
            "var_mean.correction": _template_var_mean_correction,
            "var_mean.names_dim": _skip("dynamo_dimname_fake_tensor"),
            "var_mean.correction_names": _skip("dynamo_dimname_fake_tensor"),
            "var_mean.correction_out": _template_var_mean_correction_out,
            "view.default": _template_view_default,
            "view.dtype": _skip("dtype_view_not_supported"),
            "view_as_complex.default": _skip("complex64_not_supported"),
            "view_as_real.default": _skip("complex64_not_supported"),
            "view_copy.default": _template_view_copy_default,
            "view_copy.dtype": _template_view_copy_dtype,
            "view_copy.out": _template_view_copy_out,
            "view_copy.dtype_out": _template_view_copy_dtype_out,
            "where.self": _template_where_self,
            "where.ScalarOther": _template_where_scalar_other,
            "where.ScalarSelf": _template_where_scalar_self,
            "where.Scalar": _template_where_scalar,
            "where.self_out": _template_where_self_out,
            "xlogy.Scalar_Other": _template_xlogy_scalar_other,
            "xlogy.Scalar_Self": _template_xlogy_scalar_self,
            "xlogy.OutScalar_Self": _template_xlogy_out_scalar_self,
            "xlogy.OutScalar_Other": _template_xlogy_out_scalar_other,
            "xlogy_.Scalar_Other": _template_xlogy_inplace_scalar_other,
            "zeros_like.out": _skip("dynamo_fake_tensor_out_arg_mismatch"),
        }
    )

    run_aten_op_batch(
        OPS,
        batch_label="test_batch_9",
        max_fails=20,
        templates=CUSTOM_TEMPLATES,
    )
# CHECK: SUMMARY pass=
# CHECK-SAME: fail=0
