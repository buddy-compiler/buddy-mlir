# ===- ir_type_utils.py --------------------------------------------------------
#
# MLIR Python (nanobind) no longer exposes Type subclass staticmethods like
# FloatType.isinstance(). Use casting constructors and ValueError instead.
#
# ===---------------------------------------------------------------------------

from __future__ import annotations

import buddy_mlir.ir as ir


def is_float_mlir_type(element_type: ir.Type) -> bool:
    try:
        ir.FloatType(element_type)
        return True
    except ValueError:
        return False


def is_bf16_mlir_type(element_type: ir.Type) -> bool:
    try:
        ir.BF16Type(element_type)
        return True
    except ValueError:
        return False


def is_integer_mlir_type(element_type: ir.Type) -> bool:
    try:
        ir.IntegerType(element_type)
        return True
    except ValueError:
        return False


def is_complex_mlir_type(element_type: ir.Type) -> bool:
    try:
        ir.ComplexType(element_type)
        return True
    except ValueError:
        return False
