# ===- ttir_import.py ---------------------------------------------------------
#
# Part of the Buddy Compiler frontends. Lowers a Buddy Graph to a TTIR MLIR
# module using the ttmlir Python bindings (tt-mlir build output).
#
# ===---------------------------------------------------------------------------

from __future__ import annotations

import functools
import operator
from typing import List, Sequence

from ..ops.ttir import TTIRSandbox
from .operation import FuncOp, GetItemOp, OutputOp, PlaceholderOp
from .operation import Op
from .type import TensorDType, TensorMeta


def _shape_dtype_from_op_tensor_meta(op: Op):
    """Single consumer-facing shape/dtype from a node's ``tensor_meta``."""
    tm = op.tensor_meta
    if isinstance(tm, dict):
        sh = tm.get("shape")
        dt = tm.get("dtype")
        if isinstance(sh, tuple) and sh and hasattr(sh[0], "numel"):
            sh = sh[0]
        if isinstance(sh, tuple) and sh and not hasattr(sh, "numel"):
            sh = sh[0]
        if isinstance(dt, tuple) and dt:
            dt = dt[0]
        shape = list(sh) if not hasattr(sh, "shape") else list(sh)
        return shape, dt
    return list(tm.shape), tm.dtype


def _mlir_element_type_for_tensor_dtype(ctx, td, default_float_elt):
    """Map Buddy ``TensorDType`` / enum to MLIR element type (incl. integers)."""
    from ttmlir.ir import BF16Type, F16Type, F32Type, IntegerType

    if td is None:
        return default_float_elt
    if isinstance(td, TensorDType):
        name = td.value
    else:
        name = str(td)
    if name in ("bfloat16", "bf16"):
        return BF16Type.get()
    if name in ("float16", "f16"):
        return F16Type.get()
    if name in ("float32", "f32"):
        return F32Type.get()
    if name in ("int64", "i64"):
        return IntegerType.get_signless(64, ctx)
    if name in ("int32", "i32"):
        return IntegerType.get_signless(32, ctx)
    if name in ("bool", "i1"):
        return IntegerType.get_signless(1, ctx)
    return default_float_elt


def _infer_func_result_types(
    body: List[Op],
    elt_type: object,
    ranked_tensor_type,
    ctx,
):
    """Build result tensor types for ``func.func`` from the graph ``OutputOp``."""
    output_nodes = [n for n in body if isinstance(n, OutputOp)]
    if not output_nodes:
        raise NotImplementedError(
            "TTIR import requires at least one OutputOp in the graph body."
        )
    # ``GraphDriver`` may append duplicate ``OutputOp`` nodes; the body-walking
    # importer below uses the **last** OutputOp's args (loop overwrite). Pick
    # the same one here so the function signature matches ``func.return``.
    out_node = output_nodes[-1]
    types = []
    for arg_name in out_node.args:
        prod = next((n for n in body if str(n.name) == str(arg_name)), None)
        if prod is None:
            raise RuntimeError(
                f"Output refers to unknown node {arg_name!r}."
            )
        shape, dt = _shape_dtype_from_op_tensor_meta(prod)
        mel = _mlir_element_type_for_tensor_dtype(ctx, dt, elt_type)
        types.append(ranked_tensor_type.get(shape, mel))
    return types


def build_ttir_module_for_graph(
    body: List[Op],
    params_shapes: List[TensorMeta],
    inputs_shapes: List[TensorMeta],
    func_name: str,
    ops_registry: dict,
    *,
    verbose: bool,
    element_dtype: str = "bf16",
):
    """
    Build a ``ttmlir.ir.Module`` for one Buddy graph (single ``func.func``).

    Args:
        body: Graph operation list (same order as ``Graph._body``).
        params_shapes: Parameter tensor metas (block arguments first).
        inputs_shapes: User input tensor metas (block arguments after params).
        func_name: MLIR function name (e.g. ``forward`` / ``subgraph0``).
        ops_registry: Mapping ``BuddyOpClassName`` -> lower function.
        verbose: If True, print per-node debug like ``GraphImporter``.
        element_dtype: ``\"bf16\"`` (default) or ``\"f32\"`` for TTIR tensors.

    Returns:
        ``(module, context)`` — ``(ttmlir.ir.Module, ttmlir.ir.Context)``.

    Raises:
        ImportError: If ``ttmlir`` is not on ``PYTHONPATH``.
        KeyError: If an operation has no entry in ``ops_registry``.
    """
    try:
        from ttmlir.ir import (
            BF16Type,
            Context,
            F32Type,
            FunctionType,
            InsertionPoint,
            Location,
            Module,
            RankedTensorType,
        )
        from ttmlir.dialects import func as tt_func
    except ImportError as e:
        raise ImportError(
            "ttmlir is required for TTIR lowering. Add tt-mlir's python "
            "packages to PYTHONPATH (see tt-mlir / TTMLIR documentation)."
        ) from e

    d = (element_dtype or "bf16").lower().strip()
    if d in ("bf16", "bfloat16"):
        _dtype_key = "bf16"
    elif d in ("f32", "float32"):
        _dtype_key = "f32"
    else:
        raise ValueError(
            f"Unsupported element_dtype for TTIR: {element_dtype!r} "
            '(expected "bf16" or "f32").'
        )

    ctx = Context()
    loc = Location.unknown(ctx)

    symbol_table: dict = {}
    extern_func: List[Op] = []
    for node in body:
        if isinstance(node, FuncOp):
            extern_func.append(node)

    with ctx, loc:
        if _dtype_key == "bf16":
            elt_type = BF16Type.get()
        else:
            elt_type = F32Type.get()

        sb = TTIRSandbox(ctx=ctx, loc=loc, elt_type=elt_type)

        def _tensor_type_from_meta(tm: TensorMeta | dict):
            if isinstance(tm, TensorMeta):
                shape = list(tm.shape)
                dt = tm.dtype
            else:
                shape = list(tm["shape"])
                dt = tm.get("dtype")
            mel = _mlir_element_type_for_tensor_dtype(ctx, dt, elt_type)
            return RankedTensorType.get(shape, mel)

        arguments: List = []
        for arg in params_shapes + inputs_shapes:
            arguments.append(_tensor_type_from_meta(arg))
        if not arguments:
            raise ValueError("TTIR import expects at least one block argument.")

        result_types = _infer_func_result_types(
            body, elt_type, RankedTensorType, ctx
        )
        # Buddy may still label the last op as float32 while ``lower_to_ttir`` uses bf16
        # activations; align declared return types with the import element type.
        if _dtype_key == "bf16":
            from ttmlir.ir import F32Type

            aligned = []
            for rt in result_types:
                if rt.element_type == F32Type.get():
                    aligned.append(
                        RankedTensorType.get(list(rt.shape), elt_type)
                    )
                else:
                    aligned.append(rt)
            result_types = aligned
        ft = FunctionType.get(inputs=arguments, results=result_types)

        module = Module.create(loc=loc)
        with InsertionPoint(module.body):
            f = tt_func.FuncOp(func_name, ft)
            entry = f.add_entry_block()
            args_list = list(entry.arguments)

        num_input_visited = 0

        def _import_placeholder(node: PlaceholderOp):
            nonlocal num_input_visited
            symbol_table[(str(node.name), 0)] = args_list[num_input_visited]
            num_input_visited += 1

        def _import_op(node: Op):
            op_name = node.__class__.__name__
            if op_name not in ops_registry:
                raise KeyError(
                    f"No TTIR lower registered for {op_name}. "
                    "Extend buddy.compiler.ops.ttir.ops_registry."
                )
            op_ret = ops_registry[op_name](node, symbol_table, sb)
            if isinstance(op_ret, (list, tuple)):
                for i, val in enumerate(op_ret):
                    if hasattr(val, "result"):
                        symbol_table[(str(node.name), i)] = val.result
                    else:
                        symbol_table[(str(node.name), i)] = val
            else:
                if hasattr(op_ret, "result"):
                    symbol_table[(str(node.name), 0)] = op_ret.result
                else:
                    symbol_table[(str(node.name), 0)] = op_ret

        with InsertionPoint(entry):
            for node in body:
                if node in extern_func:
                    continue
                if isinstance(node, OutputOp):
                    output_node_args = node.args
                    returns = [
                        symbol_table.get((str(output_arg), 0))
                        for output_arg in output_node_args
                    ]
                    symbol_table[("output", 0)] = returns
                elif isinstance(node, PlaceholderOp):
                    old_ops = list(entry.operations)
                    _import_placeholder(node)
                    new_ops = list(entry.operations)
                    if verbose:
                        _print_verbose(node, old_ops, new_ops)
                elif isinstance(node, GetItemOp):
                    symbol_table[(str(node.name), 0)] = symbol_table[
                        (str(node.args[0]), node.args[1])
                    ]
                else:
                    old_ops = list(entry.operations)
                    _import_op(node)
                    new_ops = list(entry.operations)
                    if verbose:
                        _print_verbose(node, old_ops, new_ops)

            outs = symbol_table.get(("output", 0))
            if outs is None:
                raise RuntimeError(
                    "Graph has no OutputOp or output was not lowered."
                )
            if not isinstance(outs, list):
                outs = [outs]
            tt_func.ReturnOp(outs)

    return module, ctx


def _shape_numel(shape: Sequence[int]) -> int:
    return functools.reduce(operator.mul, (int(x) for x in shape), 1)


def append_ttir_forward_with_packed_weights(
    module,
    *,
    subgraph_func_name: str,
    forward_func_name: str = "forward",
):
    """
    Append a ``forward``-style function to an existing TTIR module that already
    defines ``subgraph_func_name`` (e.g. ``subgraph0``).

    This mirrors the native Buddy LeNet split: one 1-D packed weight buffer
    (same layout as ``arg0.data`` from ``buddy-lenet-import.py``) plus the
    user input tensor, unpack weights with ``ttir.slice_static`` / ``ttir.reshape``,
    then ``func.call`` the subgraph.

    Shapes and argument order are taken from the callee's ``function_type``:
    the **first** input must be the activations tensor (e.g. image); remaining
    inputs are unpacked from the 1-D buffer in order.

    Args:
        module: ``ttmlir.ir.Module`` produced by ``build_ttir_module_for_graph`` /
            ``Graph.lower_to_ttir()`` for the subgraph only.
        subgraph_func_name: Callee symbol (usually ``"subgraph0"``).
        forward_func_name: Name of the entry function to add (default ``forward``).

    Returns:
        The same ``module`` (mutated in place).
    """
    try:
        from ttmlir.dialects import func as tt_func
        from ttmlir.dialects import ttir
        from ttmlir.ir import (
            FunctionType,
            InsertionPoint,
            Location,
            RankedTensorType,
            TypeAttr,
        )
    except ImportError as e:
        raise ImportError(
            "ttmlir is required. Add tt-mlir python_packages to PYTHONPATH."
        ) from e

    ctx = module.operation.context
    loc = Location.unknown(ctx)

    def _mlir_attr_get(attrs, name: str):
        """mlir `OpAttributeMap` may not implement ``.get`` (ttmlir)."""
        getter = getattr(attrs, "get", None)
        if callable(getter):
            return getter(name)
        try:
            return attrs[name]
        except (KeyError, TypeError):
            return None

    def _iter_func_ops():
        for op in module.body.operations:
            if not hasattr(op, "operation"):
                continue
            if op.operation.name != "func.func":
                continue
            yield op

    def _func_signature(func_op):
        """Return ``FunctionType`` (some bindings expose ``function_type`` as ``TypeAttr``)."""
        ta = _mlir_attr_get(func_op.attributes, "function_type")
        if ta is None:
            raise RuntimeError("func.func missing function_type.")
        if hasattr(ta, "inputs"):
            return ta
        out = TypeAttr(ta).value
        if not hasattr(out, "inputs"):
            raise RuntimeError(
                f"Expected FunctionType from function_type, got {type(out)!r}."
            )
        return out

    callee = None
    for op in _iter_func_ops():
        sym = _mlir_attr_get(op.attributes, "sym_name")
        if sym is None:
            continue
        # StringAttr .value (mlir python)
        name = sym.value if hasattr(sym, "value") else str(sym)
        if name == subgraph_func_name:
            callee = op
            break
    if callee is None:
        raise ValueError(
            f"Module has no func.func named {subgraph_func_name!r}; "
            "lower the subgraph first."
        )

    for op in _iter_func_ops():
        sym = _mlir_attr_get(op.attributes, "sym_name")
        if sym is None:
            continue
        name = sym.value if hasattr(sym, "value") else str(sym)
        if name == forward_func_name:
            raise ValueError(
                f"Module already defines {forward_func_name!r}; "
                "refusing to append a duplicate."
            )

    fty = _func_signature(callee)
    result_types = list(fty.results)
    callee_input_types = list(fty.inputs)

    if len(callee_input_types) < 2:
        raise ValueError(
            "Packed forward expects subgraph type (image, weight...,) with at "
            f"least 2 inputs, got {len(callee_input_types)}."
        )

    image_ty = callee_input_types[0]
    weight_types = callee_input_types[1:]
    elt_type = image_ty.element_type

    with ctx, loc:
        total_elems = sum(_shape_numel(wit.shape) for wit in weight_types)
        packed_ty = RankedTensorType.get([total_elems], elt_type)
        forward_ft = FunctionType.get(
            inputs=[packed_ty, image_ty], results=result_types
        )

        def _unpack_one(slice_val, wit):
            sh = [int(d) for d in wit.shape]
            if len(sh) == 1:
                return slice_val
            out_ty = RankedTensorType.get(sh, elt_type)
            shape_i32 = [int(x) for x in sh]
            return ttir.reshape(out_ty, slice_val, shape_i32)

        with InsertionPoint(module.body):
            fwd = tt_func.FuncOp(forward_func_name, forward_ft)
            entry = fwd.add_entry_block()
            packed_arg, image_arg = entry.arguments

        with InsertionPoint(entry):
            offset = 0
            call_args = [image_arg]
            for wit in weight_types:
                sh = [int(d) for d in wit.shape]
                n = _shape_numel(sh)
                slice_ty = RankedTensorType.get([n], elt_type)
                beg = [offset]
                end = [offset + n]
                stp = [1]
                sliced = ttir.slice_static(
                    slice_ty, packed_arg, beg, end, stp, loc=loc
                )
                call_args.append(_unpack_one(sliced, wit))
                offset += n

            call_op = tt_func.CallOp(
                result_types, subgraph_func_name, call_args, loc=loc
            )
            tt_func.ReturnOp(list(call_op.results))

    return module


def append_ttir_forward_bf16_f32_packed_i64_runtime(
    module,
    *,
    subgraph_func_name: str = "subgraph0",
    forward_func_name: str = "forward",
):
    """
    Append ``@forward`` that packs all bf16 and all f32 callee tensors into two
    1-D buffers (same element type each), passes integer tensors through as
    separate arguments, unpacks with ``ttir.slice_static`` / ``ttir.reshape``,
    then ``func.call`` the subgraph.

    Typical for LLM TTIR where the only runtime input is ``i64`` token ids and
    all float weights are bf16/f32 (e.g. DeepSeek-R1 prefill subgraph).
    """
    try:
        from ttmlir.dialects import func as tt_func
        from ttmlir.dialects import ttir
        from ttmlir.ir import (
            BF16Type,
            F32Type,
            FunctionType,
            InsertionPoint,
            Location,
            RankedTensorType,
            TypeAttr,
            IntegerType,
        )
    except ImportError as e:
        raise ImportError(
            "ttmlir is required. Add tt-mlir python_packages to PYTHONPATH."
        ) from e

    ctx = module.operation.context
    loc = Location.unknown(ctx)

    def _mlir_attr_get(attrs, name: str):
        getter = getattr(attrs, "get", None)
        if callable(getter):
            return getter(name)
        try:
            return attrs[name]
        except (KeyError, TypeError):
            return None

    def _iter_func_ops():
        for op in module.body.operations:
            if not hasattr(op, "operation"):
                continue
            if op.operation.name != "func.func":
                continue
            yield op

    def _func_signature(func_op):
        ta = _mlir_attr_get(func_op.attributes, "function_type")
        if ta is None:
            raise RuntimeError("func.func missing function_type.")
        if hasattr(ta, "inputs"):
            return ta
        out = TypeAttr(ta).value
        if not hasattr(out, "inputs"):
            raise RuntimeError(
                f"Expected FunctionType from function_type, got {type(out)!r}."
            )
        return out

    callee = None
    for op in _iter_func_ops():
        sym = _mlir_attr_get(op.attributes, "sym_name")
        if sym is None:
            continue
        name = sym.value if hasattr(sym, "value") else str(sym)
        if name == subgraph_func_name:
            callee = op
            break
    if callee is None:
        raise ValueError(
            f"Module has no func.func named {subgraph_func_name!r}; "
            "lower the subgraph first."
        )

    for op in _iter_func_ops():
        sym = _mlir_attr_get(op.attributes, "sym_name")
        if sym is None:
            continue
        name = sym.value if hasattr(sym, "value") else str(sym)
        if name == forward_func_name:
            raise ValueError(
                f"Module already defines {forward_func_name!r}; "
                "refusing to append a duplicate."
            )

    fty = _func_signature(callee)
    result_types = list(fty.results)
    callee_input_types = list(fty.inputs)

    bf16_ts: List[RankedTensorType] = []
    f32_ts: List[RankedTensorType] = []
    int_ts: List[RankedTensorType] = []
    for i, rt in enumerate(callee_input_types):
        et = rt.element_type
        if BF16Type.isinstance(et):
            bf16_ts.append(rt)
        elif F32Type.isinstance(et):
            f32_ts.append(rt)
        elif IntegerType.isinstance(et):
            int_ts.append(rt)
        else:
            raise NotImplementedError(
                f"{forward_func_name}: unsupported subgraph arg {i} element type {et!r} "
                "(expected bf16, f32, or integer)."
            )

    if not int_ts:
        raise ValueError(
            f"{subgraph_func_name} has no integer (e.g. i64) inputs; "
            "use append_ttir_forward_with_packed_weights for image+weights."
        )

    def _numel(rt: RankedTensorType) -> int:
        return functools.reduce(
            operator.mul, (int(d) for d in rt.shape), 1
        )

    n_bf16 = sum(_numel(t) for t in bf16_ts)
    n_f32 = sum(_numel(t) for t in f32_ts)
    if n_bf16 == 0:
        raise ValueError("No bf16 tensors to pack (unexpected for LLM TTIR).")
    # TTNN/tt-metal stores 1-D tensors in TILE layout as a [1, N] matrix padded
    # to [32, ceil(N/32)*32] (tile = 32x32), which costs 32× the logical size.
    # For large LLM weights that is > 100 GB and OOMs. To avoid materializing a
    # 1-D packed view, we pack weights as **column blocks** inside a
    # ``[32, slab_cols]`` slab (slab_cols = sum(n_i/32)): weight ``i`` with
    # numel ``n_i`` occupies slab[:, col_i : col_i + n_i/32] and its data is
    # stored as ``w.reshape(32, n_i/32)`` (row-major). Each weight's original
    # ``torch.flatten()`` order is preserved via that reshape, so @forward can
    # simply 2-D slice the block and reshape to the callee's tensor shape with
    # no global 1-D reshape in sight.
    slab_rows = 32
    per_weight_nb: List[int] = []
    for t in bf16_ts:
        n = _numel(t)
        if n % slab_rows != 0:
            raise ValueError(
                f"{forward_func_name}: bf16 weight numel {n} is not divisible by "
                f"{slab_rows}; column-block packing requires per-weight numel %% 32 == 0."
            )
        per_weight_nb.append(n // slab_rows)
    slab_cols = sum(per_weight_nb)
    assert slab_cols * slab_rows == n_bf16

    with ctx, loc:
        elt_bf16 = BF16Type.get()
        elt_f32 = F32Type.get()
        packed_bf16_ty = RankedTensorType.get([slab_rows, slab_cols], elt_bf16)
        forward_input_tys = [packed_bf16_ty]
        if n_f32 > 0:
            forward_input_tys.append(RankedTensorType.get([n_f32], elt_f32))
        forward_input_tys.extend(int_ts)

        forward_ft = FunctionType.get(
            inputs=forward_input_tys, results=result_types
        )

        with InsertionPoint(module.body):
            fwd = tt_func.FuncOp(forward_func_name, forward_ft)
            entry = fwd.add_entry_block()
            args = list(entry.arguments)

        idx = 0
        packed_bf16_arg = args[idx]
        idx += 1
        packed_f32_arg = None
        if n_f32 > 0:
            packed_f32_arg = args[idx]
            idx += 1
        runtime_int_args = args[idx:]

        with InsertionPoint(entry):
            # 2-D column-block slicing on the [32, slab_cols] packed slab. No
            # 1-D reshape is emitted, so the TTNN runtime never sees a 1-D
            # tile-layout view (which would pad to 32× the logical size).
            col_off = 0  # in units of slab_cols (= n_i / 32)
            off_f32 = 0
            rt_i = 0
            call_args = []
            bi = 0
            fi = 0
            for i, rt in enumerate(callee_input_types):
                et = rt.element_type
                if BF16Type.isinstance(et):
                    wit = bf16_ts[bi]
                    nb = per_weight_nb[bi]
                    bi += 1
                    sh = [int(d) for d in wit.shape]
                    block_ty = RankedTensorType.get(
                        [slab_rows, nb], elt_bf16
                    )
                    sliced = ttir.slice_static(
                        block_ty,
                        packed_bf16_arg,
                        [0, col_off],
                        [slab_rows, col_off + nb],
                        [1, 1],
                        loc=loc,
                    )
                    col_off += nb
                    out_ty = RankedTensorType.get(sh, elt_bf16)
                    call_args.append(
                        ttir.reshape(out_ty, sliced, [int(x) for x in sh])
                    )
                elif F32Type.isinstance(et):
                    wit = f32_ts[fi]
                    fi += 1
                    sh = [int(d) for d in wit.shape]
                    n = _shape_numel(sh)
                    slice_ty = RankedTensorType.get([n], elt_f32)
                    beg = [off_f32]
                    end = [off_f32 + n]
                    stp = [1]
                    sliced = ttir.slice_static(
                        slice_ty, packed_f32_arg, beg, end, stp, loc=loc
                    )
                    off_f32 += n
                    if len(sh) == 1:
                        call_args.append(sliced)
                    else:
                        out_ty = RankedTensorType.get(sh, elt_f32)
                        call_args.append(
                            ttir.reshape(out_ty, sliced, [int(x) for x in sh])
                        )
                elif IntegerType.isinstance(et):
                    call_args.append(runtime_int_args[rt_i])
                    rt_i += 1
                else:
                    raise NotImplementedError(f"arg {i}")

            call_op = tt_func.CallOp(
                result_types, subgraph_func_name, call_args, loc=loc
            )
            tt_func.ReturnOp(list(call_op.results))

    return module


def _print_verbose(node, old_ops, new_ops):
    print("=" * 20 + "Graph Node" + "=" * 20)
    print("Node: " + node.name)
    print("Type: " + str(node._op_type))
    print("Arguments: " + str(node.args))
    print("Parents: " + str(node._parents))
    print("Children: " + str(node._children))
    print("-" * 20 + "MLIR OPS (TTIR)" + "-" * 20)
    for op in new_ops:
        if op not in old_ops:
            print(op)
    print("")
