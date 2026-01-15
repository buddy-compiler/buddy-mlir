# ===- math.py -----------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===---------------------------------------------------------------------------
#
# The registry of mappings from Torch node to MLIR math dialect operations.
#
# ===---------------------------------------------------------------------------

from buddy_mlir.dialects import math


def erf_op(node, symbol_table):
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    op = math.ErfOp(input_tensor)
    return op


def sqrt_op(node, symbol_table):
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    op = math.SqrtOp(input_tensor)
    return op


def cos_op(node, symbol_table):
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    op = math.CosOp(input_tensor)
    return op


def sin_op(node, symbol_table):
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    op = math.SinOp(input_tensor)
    return op


def tan_op(node, symbol_table):
    """tan(x) using math.TanOp"""
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    op = math.TanOp(input_tensor)
    return op


def acos_op(node, symbol_table):
    """arccos(x) using math.AcosOp"""
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    op = math.AcosOp(input_tensor)
    return op


def asin_op(node, symbol_table):
    """arcsin(x) using math.AsinOp"""
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    op = math.AsinOp(input_tensor)
    return op


def atan_op(node, symbol_table):
    """arctan(x) using math.AtanOp"""
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    op = math.AtanOp(input_tensor)
    return op


def atan2_op(node, symbol_table):
    """atan2(y, x) using math.Atan2Op"""
    input1 = symbol_table.get((str(node.args[0]), 0))
    input2 = symbol_table.get((str(node.args[1]), 0))
    op = math.Atan2Op(input1, input2)
    return op


def sinh_op(node, symbol_table):
    """sinh(x) using math.SinhOp"""
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    op = math.SinhOp(input_tensor)
    return op


def cosh_op(node, symbol_table):
    """cosh(x) using math.CoshOp"""
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    op = math.CoshOp(input_tensor)
    return op


def tanh_op(node, symbol_table):
    """tanh(x) using math.TanhOp"""
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    op = math.TanhOp(input_tensor)
    return op


def acosh_op(node, symbol_table):
    """arccosh(x) using math.AcoshOp"""
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    op = math.AcoshOp(input_tensor)
    return op


def asinh_op(node, symbol_table):
    """arcsinh(x) using math.AsinhOp"""
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    op = math.AsinhOp(input_tensor)
    return op


def atanh_op(node, symbol_table):
    """arctanh(x) using math.AtanhOp"""
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    op = math.AtanhOp(input_tensor)
    return op


def exp_op(node, symbol_table):
    """exp(x) using math.ExpOp"""
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    op = math.ExpOp(input_tensor)
    return op


def exp2_op(node, symbol_table):
    """2^x using math.Exp2Op"""
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    op = math.Exp2Op(input_tensor)
    return op


def expm1_op(node, symbol_table):
    """exp(x) - 1 using math.ExpM1Op"""
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    op = math.ExpM1Op(input_tensor)
    return op


def log_op(node, symbol_table):
    """log(x) using math.LogOp"""
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    op = math.LogOp(input_tensor)
    return op


def log2_op(node, symbol_table):
    """log2(x) using math.Log2Op"""
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    op = math.Log2Op(input_tensor)
    return op


def log10_op(node, symbol_table):
    """log10(x) using math.Log10Op"""
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    op = math.Log10Op(input_tensor)
    return op


def log1p_op(node, symbol_table):
    """log(1 + x) using math.Log1pOp"""
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    op = math.Log1pOp(input_tensor)
    return op


def rsqrt_op(node, symbol_table):
    """1/sqrt(x) using math.RsqrtOp"""
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    op = math.RsqrtOp(input_tensor)
    return op


def ceil_op(node, symbol_table):
    """ceil(x) using math.CeilOp"""
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    op = math.CeilOp(input_tensor)
    return op


def floor_op(node, symbol_table):
    """floor(x) using math.FloorOp"""
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    op = math.FloorOp(input_tensor)
    return op


def round_op(node, symbol_table):
    """round(x) using math.RoundOp"""
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    op = math.RoundOp(input_tensor)
    return op


def trunc_op(node, symbol_table):
    """trunc(x) using math.TruncOp"""
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    op = math.TruncOp(input_tensor)
    return op


def abs_op(node, symbol_table):
    """abs(x) using math.AbsFOp for float, math.AbsIOp for int"""
    import buddy_mlir.ir as ir

    input_tensor = symbol_table.get((str(node.args[0]), 0))
    element_type = ir.RankedTensorType(input_tensor.type).element_type
    if str(element_type).startswith("f"):
        op = math.AbsFOp(input_tensor)
    else:
        op = math.AbsIOp(input_tensor)
    return op


def powf_op(node, symbol_table):
    """pow(x, y) for float tensors using math.PowFOp"""
    input1 = symbol_table.get((str(node.args[0]), 0))
    input2 = symbol_table.get((str(node.args[1]), 0))
    op = math.PowFOp(input1, input2)
    return op


ops_registry = {
    "ErfOp": erf_op,
    "SqrtOp": sqrt_op,
    "CosOp": cos_op,
    "SinOp": sin_op,
    "TanOp": tan_op,
    "AcosOp": acos_op,
    "AsinOp": asin_op,
    "AtanOp": atan_op,
    "Atan2Op": atan2_op,
    "SinhOp": sinh_op,
    "CoshOp": cosh_op,
    "TanhOp": tanh_op,
    "AcoshOp": acosh_op,
    "AsinhOp": asinh_op,
    "AtanhOp": atanh_op,
    "ExpOp": exp_op,
    "Exp2Op": exp2_op,
    "Expm1Op": expm1_op,
    "LogOp": log_op,
    "Log2Op": log2_op,
    "Log10Op": log10_op,
    "Log1pOp": log1p_op,
    "RsqrtOp": rsqrt_op,
    "CeilOp": ceil_op,
    "FloorOp": floor_op,
    "RoundOp": round_op,
    "TruncOp": trunc_op,
    "AbsOp": abs_op,
    "PowTensorTensorOp": powf_op,
}
