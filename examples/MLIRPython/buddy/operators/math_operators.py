from mlir.dialects import math


def erf_op(node, symbol_table):
    input_ = symbol_table.get((str(node.args[0]), 0))
    op = math.ErfOp(input_)
    return op


operators_registry = {
    "erf.default": erf_op,
}
