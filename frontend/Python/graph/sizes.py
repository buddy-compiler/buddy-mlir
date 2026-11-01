# ===- sizes.py ----------------------------------------------------------------
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
# Symbolic sizes. TorchDynamo traces a dynamic size as a torch.SymInt whose
# expression is over the symbols of the graph inputs (s0, 2*s0, s0 + 1, ...).
# In MLIR such a size is a `?` of the tensor type, and its value an SSA index:
# the importer binds every symbol to a `tensor.dim` of the first function
# argument that has it, and an operation lowering asks here for the values of
# the sizes it needs (tensor.empty, the output shape of a reshape, a loop
# bound), computed with `arith` from the bound symbols.
#
# ===---------------------------------------------------------------------------

import buddy_mlir.ir as ir
import sympy
import torch
from buddy_mlir.dialects import arith, tensor
from torch.utils._sympy.functions import FloorDiv

# The symbol table entry of the sizes: sympy expression -> index value.
sizes_key = ("__buddy_sizes__", 0)


def is_symbolic(dimension) -> bool:
    return isinstance(dimension, torch.SymInt)


def mlir_shape(shape) -> list[int]:
    """The sizes of `shape` in an MLIR tensor type: a symbolic size is `?`."""
    return [
        ir.ShapedType.get_dynamic_size() if is_symbolic(d) else d for d in shape
    ]


# The symbolic sizes of the values of the graph being imported: value ->
# the sympy expression of each size, None for a static one. Elementwise
# lowerings read them through same_sizes; bind_sizes starts a new graph.
symbols = {}
# The size expressions of the node being lowered and of its arguments: a
# value its lowering builds has its dynamic sizes among them.
node_sizes = set()


def bind_sizes(arguments, shapes, symbol_table):
    """Bind each symbol of the argument shapes to a `tensor.dim` of the first
    argument that has it as a size."""
    sizes = {}
    symbol_table[sizes_key] = sizes
    symbols.clear()
    for argument, shape in zip(arguments, shapes, strict=True):
        for axis, dimension in enumerate(shape):
            if not is_symbolic(dimension):
                continue
            expression = dimension.node.expr
            if not expression.is_Symbol or expression in sizes:
                continue
            sizes[expression] = dim(argument, axis)


def _value(expression, sizes):
    if expression in sizes:
        return sizes[expression]
    index = ir.IndexType.get()
    if expression.is_Integer:
        result = arith.ConstantOp(index, int(expression)).result
    elif expression.is_Add or expression.is_Mul:
        operation = arith.AddIOp if expression.is_Add else arith.MulIOp
        terms = [_value(term, sizes) for term in expression.args]
        result = terms[0]
        for term in terms[1:]:
            result = operation(result, term).result
    elif isinstance(expression, FloorDiv):
        numerator, denominator = (_value(e, sizes) for e in expression.args)
        result = arith.FloorDivSIOp(numerator, denominator).result
    else:
        raise ValueError(f"No value for the size {expression}")
    sizes[expression] = result
    return result


def size_value(dimension, symbol_table) -> ir.Value:
    """The index value of one size of a tensor, symbolic or not."""
    expression = (
        dimension.node.expr
        if is_symbolic(dimension)
        else sympy.Integer(dimension)
    )
    return _value(expression, symbol_table[sizes_key])


def dynamic_sizes(shape, symbol_table) -> list[ir.Value]:
    """The index values of the symbolic sizes of `shape`, in order: the
    operands a `tensor.empty` or an `expand_shape` takes for its `?` sizes."""
    return [size_value(d, symbol_table) for d in shape if is_symbolic(d)]


def dim(value: ir.Value, axis: int) -> ir.Value:
    """The index value of size `axis` of a tensor value."""
    return tensor.DimOp(
        value, arith.ConstantOp(ir.IndexType.get(), axis).result
    ).result


def sizes_of(value: ir.Value) -> list:
    """The sizes of a ranked tensor value as tensor.EmptyOp takes them: an int
    for a static size, its `tensor.dim` for a dynamic one."""
    shape = ir.RankedTensorType(value.type).shape
    return [
        dim(value, axis) if size == ir.ShapedType.get_dynamic_size() else size
        for axis, size in enumerate(shape)
    ]


def mixed_sizes(shape, symbol_table) -> list:
    """The sizes of `shape` as tensor.EmptyOp takes them: an int for a static
    size, the index value of a symbolic one."""
    return [size_value(d, symbol_table) if is_symbolic(d) else d for d in shape]


def broadcast_sizes(result_type, operands) -> list:
    """The sizes of an elementwise result over `operands` of its rank, as
    tensor.EmptyOp takes them: a dynamic size is that of the first operand
    dynamic along the axis."""
    sizes = []
    for axis, size in enumerate(ir.RankedTensorType(result_type).shape):
        if size == ir.ShapedType.get_dynamic_size():
            source = next(
                v
                for v in operands
                if ir.RankedTensorType(v.type).is_dynamic_dim(axis)
            )
            size = dim(source, axis)
        sizes.append(size)
    return sizes


def freeze_sizes(graph):
    """Make every later decision on a symbolic size of `graph` raise: a
    frontend that branches on one (k % 32) or a lowering that reads one as a
    number would hold for the traced size only, and the ShapeEnv would only
    guard it."""
    for node in graph.body:
        shapes = node.tensor_meta.get("shape", [])
        shapes = (
            shapes
            if shapes and isinstance(shapes[0], (list, tuple, torch.Size))
            else [shapes]
        )
        for shape in shapes:
            for dimension in shape:
                if is_symbolic(dimension):
                    environment = dimension.node.shape_env
                    environment.frozen = True
                    environment._error_on_new_guards = True
                    return


def note_symbols(value, shape):
    """Note the symbolic sizes of a value a lowering builds of `shape`."""
    symbols[value] = [d.node.expr if is_symbolic(d) else None for d in shape]


def record_symbols(node, symbol_table):
    """Note the symbolic sizes of the values `node` lowered to."""
    shape = node.tensor_meta.get("shape")
    if shape is None:
        return
    shapes = (
        shape
        if shape and isinstance(shape[0], (list, tuple, torch.Size))
        else [shape]
    )
    for index, sizes in enumerate(shapes):
        value = symbol_table.get((str(node.name), index))
        if isinstance(value, ir.Value):
            note_symbols(value, sizes)


def _expressions(shape):
    shapes = (
        shape
        if shape and isinstance(shape[0], (list, tuple, torch.Size))
        else [shape]
    )
    return {d.node.expr for sizes in shapes for d in sizes if is_symbolic(d)}


def enter_node(node, nodes):
    """Note the size expressions of `node` and its arguments (`nodes`: name ->
    node) before its lowering builds values of its own."""
    node_sizes.clear()
    for name in [node.name, *node.parents]:
        node_sizes.update(
            _expressions(nodes[name].tensor_meta.get("shape", []))
        )


def _symbols_of(value):
    """The symbolic sizes of `value`: those noted for it, or every dynamic
    size the one size expression of the node being lowered (a value its
    lowering builds); None when unknown."""
    if value in symbols:
        return symbols[value]
    if len(node_sizes) != 1:
        return None
    (expression,) = node_sizes
    return [
        expression if size == ir.ShapedType.get_dynamic_size() else None
        for size in ir.RankedTensorType(value.type).shape
    ]


def same_sizes(value1, value2) -> bool:
    """Whether every size two operands of an elementwise operation both have
    as dynamic is one symbol of the trace, aligned from the right; sizes that
    are not known to be one may differ, or be 1, at run time."""
    shape1 = list(ir.RankedTensorType(value1.type).shape)
    shape2 = list(ir.RankedTensorType(value2.type).shape)
    symbols1, symbols2 = _symbols_of(value1), _symbols_of(value2)
    dynamic = ir.ShapedType.get_dynamic_size()
    for axis in range(1, min(len(shape1), len(shape2)) + 1):
        if shape1[-axis] != dynamic or shape2[-axis] != dynamic:
            continue
        if symbols1 is None or symbols2 is None:
            return False
        if symbols1[-axis] is None or symbols1[-axis] != symbols2[-axis]:
            return False
    return True
