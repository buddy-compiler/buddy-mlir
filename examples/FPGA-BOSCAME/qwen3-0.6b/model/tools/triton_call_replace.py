#!/usr/bin/env python3
"""Replace matched model subgraphs with calls into the compiled Triton library.

Matching is structural, on the *real* imported graph, and every replacement has
to satisfy four conditions before it is allowed:

  1. the pattern is recognised by following dataflow edges (an RMSNorm is
     Pow -> Mean -> Add -> Rsqrt -> Mul -> Mul(weight), a linear is a Matmul
     whose right operand is a permuted weight), never by node name;
  2. the exact (M, N, K) or (rows, width) exists as a *built* case in the Triton
     build tree -- the symbol table is read from the real ``frontend.json``
     files, so a missing specialisation is reported as uncovered instead of
     being silently replaced by something that does not match;
  3. the dtype and rank of every operand agree with the kernel's declared
     arguments;
  4. the replacement is single-output, so the result can be bound to the node's
     value without changing the graph's shape.

Anything that fails these checks is left alone and listed in the coverage
report. Nothing falls back to a scalar C implementation.

ABI note: tensor-returning external functions receive an uninitialized output
DESCRIPTOR slot, not caller-allocated tensor data. Returning adapters fill that
slot with a private per-callsite static buffer (single invocation only). W8A8,
attention and KV calls instead return void and take explicit caller-owned
workspace descriptors. The library's descriptor/grid adapters only convert ABI.
"""
import argparse
import json
import math
from pathlib import Path
import re

# Op type names are used only to *walk* the pattern; the decision to replace is
# made from shapes and dataflow, and every candidate is re-checked below.
VIEW_TYPES = ("PermuteOp",)


def load_kernel_index(triton_build):
    """Index every built Triton case by name, with its real ABI."""
    index = {}
    for frontend in sorted(Path(triton_build).glob("*/frontend.json")):
        data = json.loads(frontend.read_text())
        index[data["name"]] = {
            "name": data["name"],
            "symbol": data["symbol"],
            "adapter_entry": "_mlir_ciface_kernel_" + data["name"],
            "arguments": data["arguments"],
            "constexprs": data["constexprs"],
            "grid": data["grid"],
            "frontend": str(frontend),
            "kernel_module": data.get("kernel_module", "kernels"),
        }
    return index


def shape_of(op):
    meta = getattr(op, "tensor_meta", None)
    if not isinstance(meta, dict):
        return None
    shape = meta.get("shape")
    if shape is None:
        return None
    try:
        return [int(v) for v in shape]
    except (TypeError, ValueError):
        return None


def dtype_of(op):
    meta = getattr(op, "tensor_meta", None)
    if not isinstance(meta, dict):
        return None
    return str(meta.get("dtype"))


def workspace_record(node):
    record = {"name": node.name, "shape": list(shape_of(node)),
              "dtype": str(node.tensor_meta.get("dtype")),
              "role": getattr(node, "workspace_role", "kernel_output")}
    if record["role"] == "cache_positions":
        record["initialization"] = "cache_position + arange(sequence), every graph invocation"
    elif record["role"] == "accumulator":
        record["initialization"] = "zero, every graph invocation"
    return record


def parents_of(graph, op):
    return [graph.node_table[name] for name in (op._parents or [])
            if name in graph.node_table]


def children_of(graph, op):
    return [graph.node_table[name] for name in (op._children or [])
            if name in graph.node_table]


def is_float32(op):
    return "Float32" in (dtype_of(op) or "")


def _unsupported(op, reason):
    return {"reason": reason, "shape": shape_of(op) or []}


def returning_symbol(graph, op, kernel_symbol):
    """A returned static buffer belongs to one call site, never a shape.

    Q/K projections and residual paths can keep equal-shaped results alive
    simultaneously. Sharing their backing array silently overwrites an earlier
    SSA value. Include the graph entry too: prefill/decode may use different
    descriptor ranks even when the flattened kernel shape is identical.
    """
    site = f"{graph._func_name}_{op.name}"
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", site):
        raise ValueError(f"cannot form a stable C symbol for graph call site {site!r}")
    return kernel_symbol + "__" + site


def _unwrap_shape_views(graph, node):
    while type(node).__name__ in ("ViewOp", "UnsqueezeOp", "ExpandOp", "CloneOp", "AliasOp"):
        if not node.args or node.args[0] not in graph.node_table:
            break
        node = graph.node_table[node.args[0]]
    return node


def _is_iota(node, length):
    return (type(node).__name__ == "IotaOp" and list(node.args) == [length]
            and node.kwargs.get("start", 0) == 0 and node.kwargs.get("step", 1) == 1)


def _position_source(graph, node, sequence):
    node = _unwrap_shape_views(graph, node)
    if type(node).__name__ != "AddOp" or len(node.args) != 2:
        return None
    first = graph.node_table.get(node.args[0])
    source = graph.node_table.get(node.args[1])
    if (first is None or source is None or not _is_iota(first, sequence)
            or type(source).__name__ != "PlaceholderOp" or shape_of(source) != [1]
            or "Int64" not in (dtype_of(source) or "")):
        return None
    return source.name


def _causal_mask_source(graph, mask_name, sequence, total):
    """Prove mask == where(slot <= start + arange(S), 0, -inf)."""
    mask = graph.node_table.get(mask_name)
    if mask is None or type(mask).__name__ != "WhereOp" or len(mask.args) != 3:
        return None
    condition, yes, no = [graph.node_table.get(name) for name in mask.args]
    if (yes is None or no is None or type(yes).__name__ != "ScalarTensorOp"
            or type(no).__name__ != "ScalarTensorOp" or list(yes.args) != [0.0]
            or list(no.args) != [-float("inf")]):
        return None
    condition = _unwrap_shape_views(graph, condition)
    if type(condition).__name__ != "LeTensorOp" or len(condition.args) != 2:
        return None
    slots, positions = [graph.node_table.get(name) for name in condition.args]
    if slots is None or positions is None:
        return None
    if shape_of(slots) != [1, 1, 1, total] or shape_of(positions) != [1, 1, sequence, 1]:
        return None
    slots = _unwrap_shape_views(graph, slots)
    if type(slots).__name__ == "AddOp" and len(slots.args) == 2 and slots.args[1] == 0:
        slots = graph.node_table.get(slots.args[0])
    if slots is None or not _is_iota(slots, total):
        return None
    return _position_source(graph, positions, sequence)


def match_linear(graph, op, index, require_kernel=True):
    """Matmul(A[M,K], Permute(W[N,K])) -> [M,N], W a real weight placeholder."""
    if type(op).__name__ != "MatmulOp":
        return None
    parents = parents_of(graph, op)
    if len(parents) != 2:
        return None
    left, right = parents
    if type(right).__name__ not in VIEW_TYPES:
        return None
    if not is_float32(left) or not is_float32(op):
        return None
    weight = parents_of(graph, right)
    if len(weight) != 1:
        return None
    weight = weight[0]
    if type(weight).__name__ != "PlaceholderOp":
        return None

    a_shape, w_shape, out_shape = shape_of(left), shape_of(weight), shape_of(op)
    if not a_shape or not w_shape or not out_shape:
        return None
    if len(a_shape) != 2 or len(w_shape) != 2 or len(out_shape) != 2:
        return None
    m, k = a_shape
    n, k2 = w_shape
    if k != k2 or out_shape != [m, n]:
        return None
    if not is_float32(weight) or not is_float32(right):
        return _unsupported(op, "linear operands must all be f32")
    if (len(right.args) != 2 or not isinstance(right.args[1], (list, tuple))
            or list(right.args[1]) != [1, 0]):
        return _unsupported(op, "linear weight view must transpose axes [1, 0]")
    if shape_of(right) != [k, n]:
        return _unsupported(op, "linear transposed weight shape disagrees with [K,N]")

    case_name = f"matmul_{m}x{n}x{k}_f32"
    case = index.get(case_name)
    if case is None and require_kernel:
        return {"reason": f"no built f32 linear case for M={m} N={n} K={k}",
                "shape": [m, n, k]}
    return {
        "case": case,
        # The graph-facing wrapper must NOT reuse the raw kernel's symbol: the
        # compiled kernel is itself called ``triton_<case>``, so reusing that
        # name would define it twice and make the archive member a duplicate
        # that the linker silently drops instead of pulling the kernel in.
        "symbol": "qwen_graph_" + case_name,
        "kind": "linear",
        "shape": [m, n, k],
        "operands": [left, weight],
        "result": op,
        "interior": [right],
        # How to build the archive call from the adapter's parameters, where a0
        # is the result descriptor and a1.. are the graph operands in order.
        "entry_args": ["a1", "a2", "a0"],
        "adapter": "reorder (result, A, W) -> archive entry (A, W, result)",
    }


def find_rmsnorm(graph, op):
    """Mul(weight, Mul(x, Rsqrt(Add(Mean(Pow(x)))))).

    Walked backwards from the candidate so the match is anchored on the edges
    that actually carry the values.
    """
    if type(op).__name__ != "MulOp" or not is_float32(op):
        return None
    parents = parents_of(graph, op)
    if len(parents) != 2:
        return None
    weight, scaled = parents
    if shape_of(weight) is None or len(shape_of(weight)) != 1:
        return None
    if type(scaled).__name__ != "MulOp":
        return None
    scaled_parents = parents_of(graph, scaled)
    if len(scaled_parents) != 2:
        return None
    source, rsqrt = scaled_parents
    if type(rsqrt).__name__ != "RsqrtOp":
        return None
    add = parents_of(graph, rsqrt)
    if len(add) != 1 or type(add[0]).__name__ != "AddOp":
        return None
    mean = parents_of(graph, add[0])
    if len(mean) != 1 or type(mean[0]).__name__ != "MeanOp":
        return None
    power = parents_of(graph, mean[0])
    if len(power) != 1 or type(power[0]).__name__ != "PowOp":
        return None
    inner = parents_of(graph, power[0])
    if len(inner) != 1 or inner[0] is not source:
        return None
    return {"source": source, "weight": weight,
            "interior": [scaled, rsqrt, add[0], mean[0], power[0]]}


def match_rmsnorm(graph, op, index):
    found = find_rmsnorm(graph, op)
    if found is None:
        return None
    source, weight = found["source"], found["weight"]
    source_shape, out_shape = shape_of(source), shape_of(op)
    width = shape_of(weight)[0]
    if not source_shape or not out_shape or source_shape != out_shape:
        return None
    if source_shape[-1] != width:
        return None
    scaled, rsqrt, add, mean, power = found["interior"]
    if not all(is_float32(node) for node in (source, weight, scaled, rsqrt,
                                           add, mean, power)):
        return _unsupported(op, "RMSNorm requires f32 operands and intermediates")
    if len(power.args) != 2 or power.args[1] != 2:
        return _unsupported(op, "RMSNorm square exponent must equal 2")
    axes = mean.args[1] if len(mean.args) > 1 else None
    if (not isinstance(axes, (list, tuple)) or len(axes) != 1
            or axes[0] not in (-1, len(source_shape) - 1)
            or len(mean.args) < 3 or mean.args[2] is not True):
        return _unsupported(op, "RMSNorm must reduce the last axis with keepdim=True")
    if (len(add.args) != 2 or not isinstance(add.args[1], (int, float))
            or not math.isclose(add.args[1], 1e-6, rel_tol=1e-7, abs_tol=0)):
        return _unsupported(op, "RMSNorm kernel epsilon is fixed at 1e-6")
    rows = 1
    for dim in source_shape[:-1]:
        rows *= dim

    case = index.get(f"rmsnorm_{rows}x{width}")
    if case is None:
        return {"reason": f"no built rmsnorm case for rows={rows} width={width}",
                "shape": [rows, width]}
    return {
        "case": case,
        "symbol": "qwen_graph_" + case["name"],
        "kind": "rmsnorm",
        "shape": [rows, width],
        # The Triton rmsnorm writes a per-row sum-of-squares side output; it is
        # scratch with no consumer in the graph, so the adapter supplies it.
        "needs_scratch": {"name": "Sums", "elements": rows},
        "operands": [source, weight],
        "result": op,
        "interior": found["interior"],
        "entry_args": ["a1", "a2", "scratch", "a0"],
        "adapter": "reorder (result, X, weight) -> archive entry (X, weight, "
                   "scratch, result)",
    }


def retarget_call(graph, call_op, operands):
    """Point the external call at the operands the matcher chose.

    ``displace_node`` copies the *replaced node's* argument list onto the new
    node, which is exactly wrong here: an RMSNorm replacement must consume the
    pre-norm tensor, not the node that already produced the normalised one, and
    a linear replacement must consume the physical [N,K] weight, not its
    transposed view. Leaving the inherited list in place would silently feed the
    kernels the wrong tensors (and would keep the whole interior chain alive).
    """
    new_names = [operand.name for operand in operands]
    old_parents = list(call_op._parents or [])
    call_op._parents = []
    for name in new_names:
        call_op.add_parent(name)
        producer = graph.node_table[name]
        if call_op.name not in producer._children:
            producer.add_children(call_op.name)
    for name in old_parents:
        if name in new_names:
            continue
        producer = graph.node_table.get(name)
        if producer is not None and call_op.name in producer._children:
            producer._children.remove(call_op.name)
    call_op._arguments = new_names
    call_op._args_index = [0] * len(new_names)


def match_silu(graph, op, index):
    """Mul(x, Sigmoid(x)) -> the shared silu kernel.

    Both operands must be the *same* producer, otherwise this is a plain
    elementwise multiply and the kernel would silently compute something else.
    """
    if type(op).__name__ != "MulOp" or not is_float32(op):
        return None
    parents = parents_of(graph, op)
    if len(parents) != 2:
        return None
    first, second = parents
    sigmoid = second if type(second).__name__ == "SigmoidOp" else first
    if type(sigmoid).__name__ != "SigmoidOp":
        return None
    source = parents_of(graph, sigmoid)
    if len(source) != 1:
        return None
    source = source[0]
    if source.name not in (first.name, second.name):
        return None
    shape = shape_of(op)
    if not shape or shape_of(source) != shape:
        return None
    rows = 1
    for dim in shape[:-1]:
        rows *= dim
    width = shape[-1]
    case = index.get(f"silu_{rows}x{width}")
    if case is None:
        return {"reason": f"no built silu case for rows={rows} width={width}",
                "shape": [rows, width]}
    return {
        "case": case,
        "symbol": "qwen_graph_" + case["name"],
        "kind": "silu",
        "shape": [rows, width],
        "operands": [source],
        "result": op,
        "interior": [sigmoid],
        "entry_args": ["a1", "a0"],
        "adapter": "reorder (result, X) -> archive entry (X, result)",
    }


def match_embedding(graph, op, index, require_kernel=True):
    """Embedding(weight, ids) -> the shared gather kernel."""
    if type(op).__name__ != "EmbeddingOp":
        return None
    parents = parents_of(graph, op)
    if len(parents) != 2:
        return None
    weight, ids = parents
    weight_shape, result_shape = shape_of(weight), shape_of(op)
    if not weight_shape or not result_shape or len(weight_shape) != 2:
        return None
    rows = 1
    for dim in result_shape[:-1]:
        rows *= dim
    width = result_shape[-1]
    if weight_shape[1] != width:
        return None
    if (not is_float32(weight) or not is_float32(op)
            or "Int64" not in (dtype_of(ids) or "")
            or shape_of(ids) is None or result_shape != shape_of(ids) + [width]):
        return _unsupported(op, "embedding requires f32 [vocab,width], i64 IDs and result IDs-shape+[width]")
    case_name = f"embedding_{rows}x{width}"
    case = index.get(case_name)
    if case is None and require_kernel:
        return {"reason": f"no built embedding case for rows={rows} width={width}",
                "shape": [rows, width]}
    return {
        "case": case,
        "symbol": "qwen_graph_" + case_name,
        "kind": "embedding",
        "shape": [rows, width],
        "operands": [ids, weight],
        "result": op,
        "interior": [],
        # kernel order is (Ids, Out, Weight)
        "entry_args": ["a1", "a0", "a2"],
        "adapter": "reorder (result, ids, weight) -> archive entry (ids, result, weight)",
    }


def rewrite_lm_head_last_token(graph, report):
    """Slice the prefill hidden state to its last position before lm_head.

    Generation only ever reads the logits of the last position, but the imported
    prefill graph computes all of them. That has a cost far beyond the extra
    arithmetic: because the prefill lm_head is M=16 it is the one pattern the
    replacement cannot cover, so its weight keeps a ``PermuteOp`` operand and
    ``one-shot-bufferize`` materialises the transposed matrix out of place --
    measured at 622,329,920 bytes (1024 x 151936 x 4), once per graph call.
    Cutting the sequence down to one position makes it an ordinary M=1 linear,
    which the existing ``matmul_1x151936x1024_f32`` kernel covers with the
    physical ``[N,K]`` weight and no copy at all.

    The lm_head is identified semantically, not by shape: it is the matmul whose
    right operand is a view of the *same* weight placeholder the embedding reads
    (the checkpoint ties them). A shape rule would also match q_proj, whose
    output width happens to equal its weight's first dimension.

    This changes the prefill graph's logits from ``[1, S, V]`` to ``[1, 1, V]``;
    the report records that so comparisons use the last position.
    """
    import torch
    from buddy.compiler.graph.operation import SliceOp, ViewOp

    params_before = list(graph.params)
    inputs_before = list(graph.inputs)
    # The embedding takes (weight, ids); both are placeholders, so pick the
    # 2-D one rather than whichever comes last in the parent list.
    embedding_weight = None
    for op in graph.body:
        if type(op).__name__ != "EmbeddingOp":
            continue
        result_shape = shape_of(op)
        for parent in parents_of(graph, op):
            if type(parent).__name__ != "PlaceholderOp":
                continue
            shape = shape_of(parent)
            if shape and len(shape) == 2 and result_shape \
                    and shape[1] == result_shape[-1]:
                embedding_weight = parent.name
    if embedding_weight is None:
        return 0

    changed = 0
    for op in list(graph.body):
        if type(op).__name__ != "MatmulOp":
            continue
        parents = parents_of(graph, op)
        if len(parents) != 2:
            continue
        left, right = parents
        right_source = parents_of(graph, right)
        if len(right_source) != 1 or right_source[0].name != embedding_weight:
            continue
        left_shape = shape_of(left)
        if not left_shape or len(left_shape) != 2 or left_shape[0] == 1:
            continue
        source = parents_of(graph, left)
        if len(source) != 1:
            continue
        source = source[0]
        source_shape = shape_of(source)
        if (not source_shape or len(source_shape) != 3
                or source_shape[0] != 1 or source_shape[1] <= 1):
            continue
        _, sequence, width = source_shape
        dtype = source.tensor_meta.get("dtype") if isinstance(
            source.tensor_meta, dict) else None

        slice_op = SliceOp()
        slice_op.name = f"{op.name}_last_slice"
        slice_op._arguments = [source.name, 1, sequence - 1, sequence]
        slice_op.tensor_meta = {"shape": torch.Size([1, 1, width]),
                                "dtype": dtype}
        view_op = ViewOp()
        view_op.name = f"{op.name}_last_view"
        view_op._arguments = [slice_op.name, [1, width]]
        view_op.tensor_meta = {"shape": torch.Size([1, width]), "dtype": dtype}

        index = graph.body.index(op)
        graph.body.insert(index, view_op)
        graph.body.insert(index, slice_op)
        graph.node_table[slice_op.name] = slice_op
        graph.node_table[view_op.name] = view_op
        slice_op.add_parent(source.name)
        source.add_children(slice_op.name)
        slice_op.add_children(view_op.name)
        view_op.add_parent(slice_op.name)

        # Repoint the matmul at the sliced view.
        op._arguments = [view_op.name if name == left.name else name
                         for name in op.args]
        op._parents[:] = [view_op.name if name == left.name else name
                          for name in op._parents]
        if op.name in left._children:
            left._children.remove(op.name)
        view_op.add_children(op.name)

        # The matmul and whatever views its result now describe one position,
        # not `sequence`. Leaving the old shape behind would (a) fail the
        # replacement's own shape check and (b) declare a graph output that does
        # not match what the kernel writes.
        result_shape = shape_of(op)
        out_width = result_shape[-1]
        op.tensor_meta["shape"] = torch.Size([1, out_width])
        reshaped_children = []
        for child in children_of(graph, op):
            if type(child).__name__ != "ViewOp":
                continue
            child_shape = shape_of(child)
            if (not child_shape or len(child_shape) != 3
                    or child_shape[1] != sequence or child_shape[2] != out_width):
                continue
            child.tensor_meta["shape"] = torch.Size([1, 1, out_width])
            # ViewOp lowers from args[1] (the shape list), so the arguments must
            # change too -- updating tensor_meta alone leaves the reshape asking
            # for the old element count and the pipeline fails with
            # "'tosa.reshape' op cannot reshape 151936 elements into 2430976".
            child._arguments = [child.args[0], [1, 1, out_width]]
            reshaped_children.append(child.name)

        report.setdefault("lm_head_last_token", []).append({
            "matmul": op.name,
            "source": source.name,
            "source_shape": list(source_shape),
            "sliced_to": [1, width],
            "inserted": [slice_op.name, view_op.name],
            "result_shape_now": [1, out_width],
            "views_reshaped": reshaped_children,
        })
        changed += 1
    return changed


def _reindex(graph, params, inputs):
    """Rebuild the positional parameter/input index lists from node identities.

    ``Graph._fake_params`` and ``Graph._inputs`` store *body indices*, so any
    ``body.insert`` shifts every entry after it and the lists silently point at
    the wrong nodes -- which surfaces much later as a KeyError('shape') while the
    main graph reads a compute node's tensor_meta as if it were a parameter.
    Snapshotting the nodes and recomputing their indices afterwards is immune to
    however many insertions or deletions happened in between.
    """
    index_of = {id(node): i for i, node in enumerate(graph.body)}
    # Nodes pruned since the snapshot are simply gone; keeping them would raise.
    #
    # Sorted ascending, because these lists must be in *body order*: the importer
    # binds placeholders to the signature (`params_shapes + inputs_shapes`) by
    # walking the body, so a param list in any other order silently hands one
    # placeholder another's tensor -- which showed up as a `func.call` receiving
    # the 151936x1024 lm_head weight where a 2048x1024 matrix was expected.
    graph._fake_params = sorted(index_of[id(node)] for node in params
                                if id(node) in index_of)
    graph._inputs = sorted(index_of[id(node)] for node in inputs
                           if id(node) in index_of)


def _placeholder_anchor(graph, parameters):
    """Where a newly created parameter must be inserted in the body.

    ``Graph.lower_to_top_level_ir`` builds the signature as
    ``params_shapes + inputs_shapes`` and then binds placeholders to those
    arguments **in body order**. So the body's placeholder order has to be
    "parameters, then inputs" -- a new parameter placed after the inputs is bound
    to an input's argument instead, which is how a quantised weight buffer ended
    up where the cache position belongs
    (``IndexPutOp: index shape [2048] is not broadcastable to [1]``).

    The anchor is therefore the first node that is not one of the existing
    parameters: the last parameter's successor, which is the first input when the
    inputs follow the parameters.
    """
    known = {id(node) for node in parameters}
    for node in graph.body:
        if id(node) not in known:
            return node
    return None


def _input_anchor(graph, inputs):
    """The first compute node, i.e. where the existing input run ends.

    Same reasoning as _placeholder_anchor: the signature is
    ``params_shapes + inputs_shapes`` and placeholders bind in body order, so a
    new *input* has to sit after every existing input and before every compute
    node.
    """
    for node in graph.body:
        if type(node).__name__ != "PlaceholderOp":
            return node
    return None


def prune_dead_nodes(graph, report):
    """Drop the nodes the rewrite made unreachable.

    Replacing a linear with quantize/matmul/dequantize leaves its f32 weight
    view and, once every linear is covered, the f32 weight placeholder itself
    with no consumers. Leaving them in the graph would defeat the purpose of the
    rewrite: they are parameters, so the deployment image would still have to
    carry the 2.22 GiB of FP32 that W8A8 exists to avoid.
    """
    removed = []
    views = ("PermuteOp", "ViewOp", "AliasOp", "ReshapeOp", "ExpandOp")
    changed = True
    while changed:
        changed = False
        for op in list(graph.body):
            if op._children or type(op).__name__ not in views:
                continue
            for parent in parents_of(graph, op):
                if op.name in parent._children:
                    parent._children.remove(op.name)
            graph.body.remove(op)
            graph.node_table.pop(op.name, None)
            removed.append(op.name)
            changed = True
    for op in list(graph.body):
        if type(op).__name__ == "PlaceholderOp" and not op._children:
            graph.body.remove(op)
            graph.node_table.pop(op.name, None)
            removed.append(op.name)
    if removed:
        report.setdefault("pruned_nodes", []).extend(removed)
    return removed


def _new_node(graph, cls, name, arguments, shape, dtype, before=None):
    """Insert a node into the graph body and register it.

    Body indices for parameters and inputs are positional, so anything inserted
    before an existing entry would need those lists rewritten; callers insert
    either at the very end or immediately before a node whose own index is not in
    either list (an ordinary compute node), which keeps that bookkeeping trivial.
    """
    import torch
    node = cls()
    node.name = name
    node._arguments = list(arguments)
    node.tensor_meta = {"shape": torch.Size(shape), "dtype": dtype}
    graph.node_table[name] = node
    if before is None:
        graph.body.append(node)
    else:
        graph.body.insert(graph.body.index(before), node)
    return node


def rewrite_embedding_to_w8a8(graph, index, call_types, report, tied, only=None):
    """Replace the gather with the int8 gather-and-rescale kernel.

    Quantising the linears alone leaves the embedding in FP32, which is 622 MB of
    the full model's parameters -- enough that the model does not fit the board's
    HIGH region once the KV cache is added. The int8 kernel reads int8 rows and
    multiplies by the token's stored scale, exactly matching
    tools/quant_reference.py's ``embed``.

    The int8 matrix and its scales are recorded in report["_tied_int8_weights"]
    keyed by the f32 placeholder they came from, so the lm_head linear -- which
    reads the same tied matrix -- reuses them instead of duplicating 155 MB.
    """
    import torch
    from buddy.compiler.graph import TensorDType
    from buddy.compiler.graph.operation import CallExternalOp, PlaceholderOp

    params_before = list(graph.params)
    inputs_before = list(graph.inputs)
    if "original_parameters" not in report:
        # Captured before ANY rewrite adds a parameter. This rewrite now runs
        # first, so if the linear rewrite captured it instead it would also
        # include the embedding's int8 matrix and scales, and the layout check in
        # run_graph_host.py would see 313 parameters against 311 sections.
        report["original_parameters"] = [p.name for p in params_before]
    added_params, added_inputs = [], []
    workspace_inputs = []
    matched = 0
    for op in list(graph.body):
        if type(op).__name__ != "EmbeddingOp":
            continue
        if only is not None and op.name not in only:
            continue
        found = match_embedding(graph, op, index, require_kernel=False)
        if found is None:
            continue
        if "reason" in found:
            report["uncovered"].append({"kind": "embedding-w8a8", "node": op.name,
                                        **found})
            continue
        ids, weight = found["operands"]
        rows, width = found["shape"]
        case = index.get(f"embedding_w8a8_{rows}x{width}")
        if case is None:
            report["uncovered"].append({
                "kind": "embedding-w8a8", "node": op.name,
                "shape": found["shape"],
                "reason": f"no built embedding_w8a8_{rows}x{width} case"})
            continue
        weight_shape = shape_of(weight)
        anchor = _placeholder_anchor(graph, params_before)
        w8 = _new_node(graph, PlaceholderOp, f"w8a8_{op.name}_weight_i8", [],
                       weight_shape, TensorDType.Int8, before=anchor)
        ws = _new_node(graph, PlaceholderOp, f"w8a8_{op.name}_weight_scale", [],
                       [weight_shape[0]], TensorDType.Float32, before=anchor)
        added_params.extend([w8, ws])
        tied[weight.name] = (w8.name, ws.name)

        # The destination is workspace, as for the linears: one-shot bufferize
        # treats an external call's operands as read-only, so a tensor.empty
        # destination would be copied and the writes lost.
        out = _new_node(graph, PlaceholderOp, f"w8a8_{op.name}_out", [],
                        list(shape_of(op)), TensorDType.Float32,
                        before=_input_anchor(graph, inputs_before))
        added_inputs.append(out)
        workspace_inputs.append(out)

        name = f"w8a8_{op.name}_{case['name']}"
        call = CallExternalOp(call_func_name="qwen_graph_" + name,
                              args=[ids.name, out.name, w8.name, ws.name],
                              args_index=[0] * 4, tensor_meta={}, name=name)
        graph.node_table[name] = call
        graph.body.insert(graph.body.index(op), call)
        for operand in (ids, out, w8, ws):
            call.add_parent(operand.name)
            operand.add_children(name)
        call_types["qwen_graph_" + name] = {
            "case": case, "kind": "w8a8-embedding", "shape": [rows, width],
            "operands": [ids, out, w8, ws],
            "graph_ranks": [len(shape_of(o)) for o in (ids, out, w8, ws)],
            "result_shape": None, "result_rank": 0, "void": True,
            "entry_args": ["a0", "a1", "a2", "a3"],
            "adapter": "void call: every destination is an argument",
        }
        for child in children_of(graph, op):
            child._arguments = [out.name if a == op.name else a
                                for a in child.args]
            child._parents[:] = [out.name if p == op.name else p
                                 for p in child._parents]
            out.add_children(child.name)
        graph.delete_node(op, parents_of(graph, op))
        matched += 1
        report.setdefault("w8a8_embeddings", []).append({
            "node": op.name, "rows": rows, "width": width,
            "case": case["name"], "weights_param": w8.name,
            "scale_param": ws.name, "out_input": out.name,
            # the f32 placeholder the int8 matrix replaces, so an offline builder
            # can quantise the right checkpoint tensor
            "source_param": weight.name})
    _reindex(graph, params_before + added_params, inputs_before + added_inputs)
    if workspace_inputs:
        report.setdefault("w8a8_workspace_inputs", []).extend(
            workspace_record(node)
            for node in workspace_inputs)
    return matched


def attention_position_case_error(case, heads, sequence, total, head_dim, qk,
                                  native_key=False):
    """Reject a dynamic-attention artifact unless its entire static ABI agrees."""
    m, n, k = sequence, (total if qk else head_dim), (head_dim if qk else total)
    expected = [("A", 3, "f32"), ("B", 3, "f32"),
                ("Position", 1, "i32"), ("C", 3, "f32")]
    actual = [(a.get("name"), a.get("rank"), a.get("dtype")) for a in case.get("arguments", [])]
    constants = dict(M=m, N=n, K=k, BM=min(m, 16), BN=16, BK=64, QK=qk)
    module = "kernels_position_native" if native_key and qk else "kernels_position"
    if case.get("kernel_module") != module:
        return "dynamic attention requires the position-aware Triton source module"
    if actual != expected or case.get("constexprs") != constants:
        return "dynamic attention argument dtype/rank or physical shape/tile disagrees"
    if case.get("grid") != [(m + 15) // 16, (n + 15) // 16, heads]:
        return "dynamic attention physical grid disagrees"
    return None


def rewrite_attention_to_triton(graph, index, call_types, report, shared=None,
                                only=None, runtime_length=False, native_key=False):
    """Replace the fused attention with the four kernels that already pass on FPGA5.

    The graph carries a single ``ScaledDotProductFlashAttentionForCpuOp``; the
    library has it split into QK, scale+mask+position, softmax and PV, at exactly
    this model's shapes (16 heads, 16 sequence, 512 context, 128 head_dim). This is
    what the operator inventory called the missing attention replacement.

    The graph already expands the 8 KV heads to 16 before the fused op, so the
    gqa_repeat kernel is not needed here.

    Semantics worth stating: the mask kernel applies the causal boundary from a
    run-time position, which is the same ``start + arange(S)`` expression the
    reference uses, so prefill and decode share one kernel each.

    Returns the number of fused attention ops replaced.
    """
    import torch
    from buddy.compiler.graph import TensorDType
    from buddy.compiler.graph.operation import CallExternalOp, PlaceholderOp

    if native_key and not runtime_length:
        raise ValueError("native key cache requires position-bounded attention")

    shared = {} if shared is None else shared
    params_before = list(graph.params)
    inputs_before = list(graph.inputs)
    added_params, added_inputs, workspace_inputs = [], [], []
    matched = 0
    for op in list(graph.body):
        if type(op).__name__ != "ScaledDotProductFlashAttentionForCpuOp":
            continue
        if only is not None and op.name not in only:
            continue
        operands = parents_of(graph, op)
        if len(operands) != 3:
            report["uncovered"].append({"kind": "attention", "node": op.name,
                                        "reason": f"{len(operands)} operands, expected 3"})
            continue
        query, key, value = operands
        q_shape, k_shape, v_shape = (shape_of(query), shape_of(key), shape_of(value))
        if (not q_shape or not k_shape or not v_shape or len(q_shape) != 4
                or len(k_shape) != 4 or len(v_shape) != 4):
            report["uncovered"].append({"kind": "attention", "node": op.name,
                                        "reason": "unexpected operand shapes"})
            continue
        heads, sequence, head_dim = q_shape[1], q_shape[2], q_shape[3]
        total = k_shape[2]
        if (q_shape[0] != 1 or k_shape != [1, heads, total, head_dim]
                or v_shape != k_shape or not all(is_float32(x) for x in operands)):
            report["uncovered"].append({"kind": "attention", "node": op.name,
                                        "reason": "requires batch-one f32 Q/K/V with matching head layout"})
            continue
        scale = op.kwargs.get("scale", head_dim ** -0.5)
        position_source = _causal_mask_source(graph, op.kwargs.get("attn_mask"), sequence, total)
        readers = children_of(graph, op)
        if (not isinstance(scale, (int, float)) or not math.isclose(scale, head_dim ** -0.5, rel_tol=1e-7)
                or op.kwargs.get("dropout_p", 0) != 0 or len(op.args) != 3
                or position_source is None
                or any(type(r).__name__ != "GetItemOp" or list(r.args) != [op.name, 0] for r in readers)):
            report["uncovered"].append({"kind": "attention", "node": op.name,
                                        "reason": "unproven scale/dropout/causal-mask semantics or a live auxiliary result"})
            continue
        # The library names these heads-first: attention_qk_16x1x512x128 is the
        # decode case with M=1, not a 16-token one.
        wanted = {
            # attention_dot computes A[M,K] @ B[K,N], so K has to arrive already
            # transposed to [heads, head_dim, total]; V is used as [heads, total,
            # head_dim] directly, which is the layout the graph already has.
            "k_layout": f"layout_k_{heads}x{total}x{head_dim}",
            "qk": f"attention_qk_{heads}x{sequence}x{total}x{head_dim}",
            "mask": f"attention_scale_mask_position_{heads}x{sequence}x{total}",
            "softmax": f"softmax_{heads}x{sequence}x{total}",
            "pv": f"attention_pv_{heads}x{sequence}x{head_dim}x{total}",
        }
        if runtime_length:
            wanted["qk"] = f"attention_qk_position_{heads}x{sequence}x{total}x{head_dim}"
            wanted["pv"] = f"attention_pv_position_{heads}x{sequence}x{head_dim}x{total}"
        if native_key:
            wanted.pop("k_layout")
            wanted["qk"] = f"attention_qk_position_native_{heads}x{sequence}x{total}x{head_dim}"
        cases = {name: index.get(case) for name, case in wanted.items()}
        missing = [wanted[n] for n, case in cases.items() if case is None]
        if missing:
            report["uncovered"].append({"kind": "attention", "node": op.name,
                                        "shape": [heads, sequence, total, head_dim],
                                        "reason": "no built case(s): " + ", ".join(missing)})
            continue

        if runtime_length:
            errors = [attention_position_case_error(cases[name], heads, sequence,
                                                   total, head_dim, name == "qk", native_key)
                      for name in ("qk", "pv")]
            position = graph.node_table.get(shared.get("position"))
            if position is not None and (shape_of(position) != [sequence]
                    or "Int32" not in (dtype_of(position) or "")
                    or getattr(position, "workspace_role", None) != "cache_positions"):
                errors.append("shared position must be i32[S] with cache_positions initialization")
            if any(errors):
                report["uncovered"].append({"kind": "attention", "node": op.name,
                                            "reason": "; ".join(e for e in errors if e)})
                continue

        prefix = f"attn_{op.name}"
        scored_shape = [q_shape[0], heads, sequence, total]
        pooled_shape = [q_shape[0], heads, sequence, head_dim]

        # Destinations again: function arguments, because an external call's
        # operands are read-only to one-shot bufferize.
        anchor = _input_anchor(graph, inputs_before)
        buffers = {}
        for suffix, shape, dtype in (
                ("k_t", [q_shape[0], heads, head_dim, total], TensorDType.Float32),
                ("qk", scored_shape, TensorDType.Float32),
                ("masked", scored_shape, TensorDType.Float32),
                ("maxima", [heads * sequence], TensorDType.Float32),
                ("sums", [heads * sequence], TensorDType.Float32),
                ("probs", scored_shape, TensorDType.Float32),
                ("pooled", pooled_shape, TensorDType.Float32),
                # The mask kernel does `tl.load(Position + query)`, so Position
                # is an array of one boundary per query row, not a scalar: for
                # prefill that is [0..S-1] and for a decode step [p]. The graph's
                # own cache position is an i64 scalar, so this is its own i32
                # argument.
                ("position", [sequence], TensorDType.Int32)):
            if native_key and suffix == "k_t":
                continue
            if suffix == "position" and "position" in shared:
                # the KV write needs the same absolute slot per token, so one
                # buffer serves both
                buffers[suffix] = graph.node_table[shared["position"]]
                continue
            node = _new_node(graph, PlaceholderOp, f"{prefix}_{suffix}", [], shape,
                             dtype, before=anchor)
            if suffix == "position":
                shared["position"] = node.name
                node.workspace_role = "cache_positions"
            buffers[suffix] = node
            added_inputs.append(node)
            workspace_inputs.append(node)

        def call(case, operands, suffix):
            name = f"{prefix}_{suffix}_{case['name']}"
            node = CallExternalOp(call_func_name="qwen_graph_" + name,
                                  args=[o.name for o in operands],
                                  args_index=[0] * len(operands),
                                  tensor_meta={}, name=name)
            graph.node_table[name] = node
            graph.body.insert(graph.body.index(op), node)
            for operand in operands:
                node.add_parent(operand.name)
                operand.add_children(name)
            call_types["qwen_graph_" + name] = {
                "case": case, "kind": "attention-" + suffix, "shape": [heads, sequence, total],
                "operands": list(operands),
                "graph_ranks": [len(shape_of(o)) for o in operands],
                "result_shape": None, "result_rank": 0, "void": True,
                "entry_args": [f"a{i}" for i in range(len(operands))],
                "adapter": "void call: every destination is an argument",
            }
            return node

        if not native_key:
            call(cases["k_layout"], [key, buffers["k_t"]], "k_layout")
        position_arg = [buffers["position"]] if runtime_length else []
        key_operand = key if native_key else buffers["k_t"]
        call(cases["qk"], [query, key_operand, *position_arg, buffers["qk"]], "qk")
        call(cases["mask"], [buffers["qk"], buffers["position"], buffers["masked"]],
             "mask")
        call(cases["softmax"], [buffers["masked"], buffers["maxima"],
                                buffers["sums"], buffers["probs"]], "softmax")
        call(cases["pv"], [buffers["probs"], value, *position_arg, buffers["pooled"]], "pv")

        # the fused op's result is read through a GetItemOp; repoint its readers
        readers = children_of(graph, op)
        for reader in readers:
            for child in children_of(graph, reader):
                child._arguments = [buffers["pooled"].name if a == reader.name else a
                                    for a in child.args]
                child._parents[:] = [buffers["pooled"].name if p == reader.name else p
                                     for p in child._parents]
                buffers["pooled"].add_children(child.name)
            graph.delete_node(reader, [op])
        graph.delete_node(op, operands)
        matched += 1
        report.setdefault("attention_replacements", []).append({
            "node": op.name, "heads": heads, "sequence": sequence,
            "total": total, "head_dim": head_dim,
            "position_source": position_source,
            "mask_contract": "where(slot <= scalar_position + arange(S), 0, -inf)",
            "runtime_length": runtime_length,
            "native_key_layout": native_key,
            "key_kernel_layout": {
                "shape": [heads, total, head_dim] if native_key else [heads, head_dim, total],
                "strides": [total * head_dim, head_dim, 1] if native_key else [head_dim * total, total, 1],
                "graph_shape": k_shape,
                "batch_dimension": "batch=1 stripped by raw-pointer kernel adapter",
                "copy": "only active kernel tile" if native_key else "full-capacity layout_k kernel",
                "removed_workspace_bytes": heads * total * head_dim * 4 if native_key else 0,
            },
            "position_contract": {
                "shape": [sequence], "dtype": "i32", "stride": 1,
                "initialization": "scalar_position + arange(S), before every invocation",
                "valid_length": "Position[S-1] + 1",
                "bounds": f"0 <= scalar_position; scalar_position + S <= {total}",
                "invalid_policy": "caller rejects out-of-context inputs; kernel clamp is memory safety only",
                "physical_capacity": total,
            },
            "cases": [cases[n]["name"] for n in
                      ("k_layout", "qk", "mask", "softmax", "pv") if n in cases],
            "workspace": [buffers[n].name for n in
                          ("k_t", "qk", "masked", "maxima", "sums", "probs",
                           "pooled", "position") if n in buffers],
        })
    _reindex(graph, params_before + added_params, inputs_before + added_inputs)
    if workspace_inputs:
        report.setdefault("w8a8_workspace_inputs", []).extend(
            workspace_record(node)
            for node in workspace_inputs)
    return matched


def rewrite_kv_cache_to_triton(graph, index, call_types, report, shared=None,
                               only=None):
    """Write K and V into the cache with the library kernel instead of IndexPut.

    The graph expresses the cache write as an IndexPut whose destination is a
    graph input, so the op's result is "the cache, updated". The kernel takes the
    cache as an argument and updates it in place, which is the out-parameter form
    the operator inventory said this replacement would need.

    Because the cache is a function argument -- not a tensor.empty -- one-shot
    bufferize treats it as writable, so the write reaches the buffer the attention
    later reads; that is the same reasoning that forced the W8A8 destinations to be
    arguments.

    Position is the absolute destination slot per token, which is exactly the
    array the attention mask already needs, so it is shared rather than duplicated.
    """
    import torch
    from buddy.compiler.graph import TensorDType
    from buddy.compiler.graph.operation import CallExternalOp, PlaceholderOp

    shared = {} if shared is None else shared
    params_before = list(graph.params)
    inputs_before = list(graph.inputs)
    added_params, added_inputs, workspace_inputs = [], [], []
    matched = 0
    for op in list(graph.body):
        if type(op).__name__ != "IndexPutOp":
            continue
        if only is not None and op.name not in only:
            continue
        parents = parents_of(graph, op)
        if len(parents) != 3:
            continue
        # named slots, not `index`: that name is the case table this function
        # searches, and shadowing it broke every lookup
        cache, slots, value = parents
        if type(cache).__name__ != "PlaceholderOp":
            continue   # only the cache writes target a graph input
        cache_shape, value_shape = shape_of(cache), shape_of(value)
        if (not cache_shape or not value_shape or len(cache_shape) != 4
                or len(value_shape) != 4):
            continue
        heads, capacity, head_dim = cache_shape[1], cache_shape[2], cache_shape[3]
        sequence = value_shape[2]
        indices = op.args[1] if len(op.args) > 1 else None
        position_source = _position_source(graph, slots, sequence)
        if (cache_shape[0] != 1 or value_shape != [1, heads, sequence, head_dim]
                or not all(is_float32(x) for x in (cache, value))
                or indices != [None, None, slots.name]
                or (len(op.args) > 3 and op.args[3] is not False)
                or op.kwargs.get("accumulate", False) or position_source is None):
            report["uncovered"].append({"kind": "kv-cache", "node": op.name,
                                        "reason": "requires overwrite into cache dimension 2 at scalar_position + arange(S)"})
            continue
        case = index.get(f"kv_cache_update_position_{sequence}x{heads}x{head_dim}"
                         f"_cap{capacity}")
        # the kernel indexes X as [S, heads, head_dim] but the graph carries the
        # projected value head-major, so it is transposed first
        layout_case = index.get(f"layout_context_{heads}x{sequence}x{head_dim}")
        missing = [n for n, c in ((f"kv_cache_update_position_{sequence}x{heads}x"
                                   f"{head_dim}_cap{capacity}", case),
                                  (f"layout_context_{heads}x{sequence}x{head_dim}",
                                   layout_case)) if c is None]
        if missing:
            report["uncovered"].append({
                "kind": "kv-cache", "node": op.name, "shape": list(cache_shape),
                "reason": "no built case(s): " + ", ".join(missing)})
            continue
        anchor = _input_anchor(graph, inputs_before)
        if "position" in shared:
            position = graph.node_table[shared["position"]]
        else:
            position = _new_node(graph, PlaceholderOp, "kv_cache_position", [],
                                 [sequence], TensorDType.Int32, before=anchor)
            position.workspace_role = "cache_positions"
            shared["position"] = position.name
            added_inputs.append(position)
            workspace_inputs.append(position)

        token_major = _new_node(graph, PlaceholderOp, f"kv_{op.name}_tm", [],
                                list(value_shape), TensorDType.Float32,
                                before=_input_anchor(graph, inputs_before))
        added_inputs.append(token_major)
        workspace_inputs.append(token_major)
        # The symbol must carry the case name. `call_types` is shared across
        # graphs and keyed by this symbol, so a name without the case let the
        # decode graph's entry overwrite the prefill one -- and the single wrapper
        # then called the DECODE layout kernel (8x1x128) for the prefill
        # (8x16x128), transposing only the first token.
        layout_name = f"kv_{op.name}_{layout_case['name']}"
        layout = CallExternalOp(call_func_name="qwen_graph_" + layout_name,
                                args=[value.name, token_major.name],
                                args_index=[0, 0], tensor_meta={},
                                name=layout_name)
        graph.node_table[layout_name] = layout
        graph.body.insert(graph.body.index(op), layout)
        layout.add_parent(value.name); value.add_children(layout_name)
        layout.add_parent(token_major.name); token_major.add_children(layout_name)
        call_types["qwen_graph_" + layout_name] = {
            "case": layout_case, "kind": "kv-cache-layout",
            "shape": [sequence, heads, head_dim],
            "operands": [value, token_major],
            "graph_ranks": [len(shape_of(o)) for o in (value, token_major)],
            "result_shape": None, "result_rank": 0, "void": True,
            "entry_args": ["a0", "a1"],
            "adapter": "void call: destination is an argument",
        }
        name = f"kv_{op.name}_{case['name']}"
        call = CallExternalOp(call_func_name="qwen_graph_" + name,
                              args=[token_major.name, position.name, cache.name],
                              args_index=[0, 0, 0], tensor_meta={}, name=name)
        graph.node_table[name] = call
        graph.body.insert(graph.body.index(op), call)
        for operand in (token_major, position, cache):
            call.add_parent(operand.name)
            operand.add_children(name)
        call_types["qwen_graph_" + name] = {
            "case": case, "kind": "kv-cache-update",
            "shape": [sequence, heads, head_dim],
            "operands": [token_major, position, cache],
            "graph_ranks": [len(shape_of(o)) for o in (token_major, position, cache)],
            "result_shape": None, "result_rank": 0, "void": True,
            "entry_args": ["a0", "a1", "a2"],
            "adapter": "void call: the cache is written through the argument",
        }
        # the consumers of the IndexPut read "the updated cache"; the call has
        # already written the input, so they read that instead
        for child in children_of(graph, op):
            child._arguments = [cache.name if a == op.name else a
                                for a in child.args]
            child._parents[:] = [cache.name if p == op.name else p
                                 for p in child._parents]
            cache.add_children(child.name)
        graph.delete_node(op, parents)
        matched += 1
        report.setdefault("kv_cache_replacements", []).append({
            "node": op.name, "cache": cache.name, "sequence": sequence,
            "heads": heads, "capacity": capacity, "case": case["name"],
            "layout_case": layout_case["name"],
            "position_source": position_source,
            "position_buffer": position.name})
    _reindex(graph, params_before + added_params, inputs_before + added_inputs)
    if workspace_inputs:
        report.setdefault("w8a8_workspace_inputs", []).extend(
            workspace_record(node)
            for node in workspace_inputs)
    return matched


def rewrite_linear_to_w8a8(graph, index, call_types, report, tied, only=None):
    """Express each linear as quantize -> int8 matmul -> dequantize.

    FP32 parameters are 2.22 GiB and cannot be placed on this board (see
    validation/board/fp32-28layer-does-not-fit.json), so the deployment form is
    W8A8. Rather than fuse the three steps into one kernel or hide them in an
    adapter, the graph itself is rewritten to call the three kernels that already
    exist and pass on FPGA5:

        quantize   X[rows,K] f32        -> Q[rows,K] i8, Scale[rows] f32
        matmul     Q[rows,K] i8, W[N,K] i8 -> Acc[rows,N] i32   (accumulates!)
        dequantize Acc[rows,N] i32, Scale[rows], WScale[N] -> Out[rows,N] f32

    Each call returns nothing: the destination is a graph-produced buffer, which
    is what keeps the arithmetic in the compiled kernels instead of in C. The
    ``matmul`` kernel adds into its output, so the accumulator is a zero-filled
    constant -- the same accumulator initialisation the FP32 graph performs.

    The int8 weight and its per-output-channel scales become new *parameters* of
    the graph, so the deployment image carries them instead of the f32 matrix.
    """
    import torch
    from buddy.compiler.graph import TensorDType
    from buddy.compiler.graph.operation import (CallExternalOp, CopyOp, EmptyOp,
                                                FullOp, PlaceholderOp)

    import os
    _debug = os.environ.get("W8A8_DEBUG") == "1"
    params_before = list(graph.params)
    inputs_before = list(graph.inputs)
    added_inputs = []
    if "original_parameters" not in report:
        # Names in the order the weight layout uses; a builder needs this to map
        # a surviving f32 parameter back to its checkpoint tensor.
        report["original_parameters"] = [p.name for p in params_before]
    added_params = []
    matched = 0
    _visited = 0
    for op in list(graph.body):
        if type(op).__name__ != "MatmulOp":
            continue
        _visited += 1
        if only is not None and op.name not in only:
            continue
        found = match_linear(graph, op, index, require_kernel=False)
        if _debug:
            print(f"[w8a8] {op.name} shape={shape_of(op)} match={'None' if found is None else found.get('reason', 'ok')}",
                  file=__import__("sys").stderr)
        if found is None:
            continue
        if "reason" in found:
            report["uncovered"].append({"kind": "linear-w8a8", "node": op.name,
                                        "shape": found["shape"],
                                        "reason": found["reason"]})
            continue
        activation, weight = found["operands"]
        a_shape = shape_of(activation)
        w_shape = shape_of(weight)
        rows, k = a_shape
        n = w_shape[0]

        quantise_case = index.get(f"quantize_{rows}x{k}")
        matmul_case = index.get(f"matmul_{rows}x{n}x{k}")
        dequant_case = index.get(f"dequantize_{rows}x{n}")
        missing = [name for name, case in (
            (f"quantize_{rows}x{k}", quantise_case),
            (f"matmul_{rows}x{n}x{k}", matmul_case),
            (f"dequantize_{rows}x{n}", dequant_case)) if case is None]
        if missing:
            report["uncovered"].append({"kind": "linear-w8a8", "node": op.name,
                                        "shape": found["shape"],
                                        "reason": "no built case(s): " + ", ".join(missing)})
            continue

        prefix = f"w8a8_{op.name}"
        new_weight_name = prefix + "_weight_i8"
        new_scale_name = prefix + "_weight_scale"
        # New parameters: the int8 matrix and its per-output-channel scales.
        # They are appended only after the insertions below, because
        # `_fake_params` stores *body indices* and inserting before `op` would
        # shift anything placed after it.
        # A placeholder must precede the call that reads it, AND every
        # placeholder must stay ahead of every compute node: Graph.lower_to_top_level_ir
        # binds function arguments in body order, so a placeholder dropped into
        # the middle silently shifts the whole signature. That is what produced
        # "IndexPutOp: index shape [2048] is not broadcastable to [1]" -- a
        # quantised weight buffer had been bound where the cache position
        # belongs. Inserting right after the last existing placeholder satisfies
        # both.
        # The checkpoint ties the embedding matrix to lm_head, so both read the
        # same values. Quantising it twice would put two identical int8 copies and
        # two scale vectors in the image -- 311 MB of duplication for the full
        # model. Reuse the placeholder the embedding rewrite already made.
        shared = tied.get(weight.name)
        if shared is not None:
            w8 = graph.node_table[shared[0]]
            ws = graph.node_table[shared[1]]
        else:
            anchor = _placeholder_anchor(graph, params_before)
            w8 = _new_node(graph, PlaceholderOp, new_weight_name, [], w_shape,
                           TensorDType.Int8, before=anchor)
            ws = _new_node(graph, PlaceholderOp, new_scale_name, [], [n],
                           TensorDType.Float32, before=anchor)
            added_params.extend([w8, ws])
            tied[weight.name] = (w8.name, ws.name)
        new_weight_name, new_scale_name = w8.name, ws.name

        # Buffers the kernels write into.
        #
        # NOT FullOp: that lowers to an `arith.constant`, and a constant used as a
        # writable operand makes one-shot bufferize hand the call a *copy*, so the
        # writes land in a buffer nobody reads and the consumer sees the original
        # zero -- which is exactly how the first W8A8 run produced all-zero
        # logits. `tensor.empty` is a real tensor that bufferize allocates and
        # shares between the writer and the reader.
        #
        # The accumulator additionally has to *start* at zero, because the int8
        # matmul kernel loads C and accumulates into it (`tt.load %address` then
        # `tt.dot(a, b, %previous)`), unlike a fresh-output matmul. Copying a zero
        # constant into an empty buffer is a linalg.generic: a computed value, so
        # it is neither folded back into a constant nor copied on write.
        # The destinations are WORKSPACE, supplied as graph inputs.
        #
        # They cannot be produced in-graph: one-shot bufferize treats the
        # operands of an unknown external function as read-only, so a call that
        # writes a `tensor.empty` result is given a *copy* and its writes are
        # discarded. Measured directly: with the dequantize wrapper replaced by
        # one that fills Out with a sentinel, the graph output was still exactly
        # zero even though the wrapper ran. Function arguments, by contrast, are
        # always treated as writable.
        #
        # The caller therefore owns these buffers, which is also what the
        # deployment needs: an explicit DDR workspace. The accumulator is
        # per-linear precisely so that the caller can zero them all once per
        # graph call -- the int8 matmul accumulates into C rather than writing it.
        anchor = _input_anchor(graph, inputs_before)
        workspace = []
        for suffix, shape, dtype in (
                ("_q", [rows, k], TensorDType.Int8),
                ("_a_scale", [rows], TensorDType.Float32),
                ("_acc", [rows, n], TensorDType.Int32),
                ("_out", [rows, n], TensorDType.Float32)):
            workspace.append(_new_node(graph, PlaceholderOp, prefix + suffix, [],
                                       shape, dtype, before=anchor))
        quantised, act_scale, accumulator, result = workspace
        accumulator.workspace_role = "accumulator"
        added_inputs.extend(workspace)

        def call(case, operands, position):
            name = f"{prefix}_{case['name']}"
            node = CallExternalOp(
                call_func_name="qwen_graph_" + name,
                args=[operand.name for operand in operands],
                args_index=[0] * len(operands),
                tensor_meta={},   # no results: every destination is an argument
                name=name)
            graph.node_table[name] = node
            graph.body.insert(graph.body.index(op), node)
            for operand in operands:
                node.add_parent(operand.name)
                operand.add_children(name)
            call_types["qwen_graph_" + name] = {
                "case": case,
                "kind": "w8a8-" + case["name"].split("_")[0],
                "shape": [len(operands)],
                "operands": list(operands),
                "graph_ranks": [len(shape_of(o)) for o in operands],
                "result_shape": None,
                "result_rank": 0,
                "void": True,
                "entry_args": [f"a{i}" for i in range(len(operands))],
                "adapter": "void call: every destination is an argument",
            }
            report.setdefault("w8a8_calls", []).append({
                "node": name, "case": case["name"],
                "operands": [o.name for o in operands],
                "operand_ranks": [len(shape_of(o)) for o in operands],
            })
            return node

        call(quantise_case, [activation, quantised, act_scale], 0)
        call(matmul_case, [quantised, w8, accumulator], 1)
        # dequantize's C signature is (X, Row, Column, Out); the per-output-channel
        # weight scales are the column scales.
        call(dequant_case, [accumulator, act_scale, ws, result], 2)

        # Consumers now read the dequantised result.
        for child in children_of(graph, op):
            child._arguments = [result.name if a == op.name else a
                                for a in child.args]
            child._parents[:] = [result.name if p == op.name else p
                                 for p in child._parents]
            result.add_children(child.name)
        if op.name in graph.node_table:
            graph.delete_node(op, parents_of(graph, op))
        matched += 1
        report.setdefault("w8a8_linears", []).append({
            "node": op.name, "rows": rows, "n": n, "k": k,
            "cases": [quantise_case["name"], matmul_case["name"],
                      dequant_case["name"]],
            # The placeholder the f32 matrix came from, so an offline builder can
            # quantise the right checkpoint tensor into these parameters.
            "weight_param": weight.name,
            "weight_param_name": new_weight_name,
            "scale_param_name": new_scale_name,
        })
    # The int8 weights and scales ARE parameters: reindexing from the pre-surgery
    # snapshot alone would drop them, and the graph would then be missing the
    # very operands the new calls read.
    _reindex(graph, params_before + added_params, inputs_before + added_inputs)
    if added_inputs:
        report.setdefault("w8a8_workspace_inputs", []).extend(
            workspace_record(node)
            for node in added_inputs)
    if _debug:
        print(f"[w8a8] visited {_visited} MatmulOps, replaced {matched}",
              file=__import__("sys").stderr)
    return matched


def replace(graph, matcher, index, call_types, report):
    """Apply one matcher across the graph, newest-first safe ordering."""
    from buddy.compiler.graph.operation import CallExternalOp, OpType

    matched = []
    for op in list(graph.body):
        found = matcher(graph, op, index)
        if found is None:
            continue
        matched.append((op, found))

    for op, found in matched:
        if "reason" in found:
            report["uncovered"].append({
                "kind": matcher.__name__.replace("match_", ""),
                "node": op.name,
                "shape": found["shape"],
                "reason": found["reason"],
            })
            continue

        # Returned descriptors reference private static backing storage in the
        # adapter. The same kernel may occur repeatedly with live results.
        found["symbol"] = returning_symbol(graph, op, found["symbol"])

        call_op = CallExternalOp(
            call_func_name=found["symbol"],
            args=[operand.name for operand in found["operands"]],
            args_index=[0] * len(found["operands"]),
            tensor_meta=dict(op.tensor_meta) if isinstance(op.tensor_meta, dict)
            else {"shape": shape_of(op), "dtype": dtype_of(op)},
            name=op.name,
        )
        graph.displace_node(op, call_op)
        call_op._op_type = OpType.Unfusable
        retarget_call(graph, call_op, found["operands"])

        # Drop the interior producers, but only when nothing else consumes them:
        # deleting a node that still has a live user would corrupt the graph.
        for interior in found.get("interior", []):
            if interior.name not in graph.node_table:
                continue
            remaining = [child for child in children_of(graph, interior)
                         if child.name in graph.node_table]
            if remaining:
                report["kept_interior"].append(
                    {"node": interior.name, "still_used_by":
                     [child.name for child in remaining]})
                continue
            graph.delete_node(interior, parents_of(graph, interior))

        found["graph_ranks"] = [len(shape_of(o)) for o in found["operands"]]
        found["result_rank"] = len(shape_of(op))
        found["result_shape"] = shape_of(op)
        call_types.setdefault(found["symbol"], found)
        report["replaced"].append({
            "node": op.name,
            "kind": found["kind"],
            "shape": found["shape"],
            "symbol": found["symbol"],
            "c_symbol": "_mlir_ciface_" + found["symbol"],
            "archive_entry": found["case"]["adapter_entry"],
            "operands": [o.name for o in found["operands"]],
            "operand_shapes": [shape_of(o) for o in found["operands"]],
            "graph_ranks": [len(shape_of(o)) for o in found["operands"]],
            "result_shape": shape_of(op),
            "result_rank": len(shape_of(op)),
            "kernel_arguments": found["case"]["arguments"],
            "adapter": found["adapter"],
            "needs_scratch": found.get("needs_scratch"),
        })
    return sum("reason" not in found for _, found in matched)


def generate_adapters(call_types, output):
    """Emit the ABI adapters between the graph and the compiled Triton kernels.

    Three facts, all verified in this checkout's sources, shape the generated C:

    1. For an *external* declaration, ``convert-func-to-llvm`` gives the raw
       symbol a private body that forwards to the ``_mlir_ciface_`` wrapper and
       leaves that wrapper as the external symbol to resolve
       (``llvm/mlir/lib/Conversion/FuncToLLVM/FuncToLLVM.cpp:188-200``,
       ``:425-435``). So the graph must export ``_mlir_ciface_<symbol>``.
    2. The wrapper's first pointer is an **uninitialised result descriptor slot**
       allocated on the caller's stack (``FuncToLLVM.cpp:205-216``); the callee
       owns filling it. Reading it -- as an earlier version of this adapter did
       -- dereferences garbage.
    3. The repository's own external kernels match that contract by allocating
       (``runtime/external_kernels/rng/RNGUtils.cpp:105-117``). We must not
       allocate, so each adapter points ``aligned`` at a **static buffer** --
       exactly the static workspace the board will reserve -- and leaves
       ``allocated`` null so the caller's deallocation pass does not free it.

    The Triton ``linear`` kernel accumulates into its output
    (``accumulator + previous``), so the buffer is zeroed before every call;
    the original graph zero-initialises the matmul destination too, so this is
    not extra work, it is the same accumulator initialisation.

    Each returned call site has its own symbol/backing buffer, including the
    graph entry name. Equal-shaped SSA results therefore cannot overwrite each
    other. These static adapters remain single-invocation/non-reentrant; separate
    model executions must not run concurrently against the same library.
    """
    lines = [
        "/* Generated by model/tools/triton_call_replace.py -- do not edit.",
        " *",
        " * ABI adapter only: result-descriptor fill, argument order, descriptor",
        " * rank and the RMSNorm scratch buffer. No numerical work happens here. */",
        '#include "support.h"',
        "",
    ]
    max_scratch = max([found["needs_scratch"]["elements"]
                       for found in call_types.values()
                       if found.get("needs_scratch")] or [0])
    if max_scratch:
        lines += [
            "/* RMSNorm sum-of-squares side output; scratch, no graph consumer. */",
            f"static float qwen_rmsnorm_scratch[{max_scratch}];",
            "",
        ]

    # One static output buffer per symbol, sized by its result shape.
    total = 0
    for symbol, found in sorted(call_types.items()):
        if found.get("void"):
            continue
        shape = found["result_shape"]
        count = 1
        for dim in shape:
            count *= dim
        lines.append(f"/* {symbol}: result {shape}, {count} f32 */")
        lines.append(f"static float qwen_out_{symbol}[{count}];")
        total += count
    if total:
        lines.append("")
        lines.append(f"/* Total static output workspace: {total} f32 "
                     f"({total * 4} bytes). */")
        lines.append("")

    declared = set()
    for symbol, found in sorted(call_types.items()):
        entry = found["case"]["adapter_entry"]
        if entry in declared:
            continue
        declared.add(entry)
        ranks = [a["rank"] for a in found["case"]["arguments"]]
        signature = ", ".join(f"MemRef{r} *" for r in ranks)
        lines.append(f"extern void {entry}({signature});")
    if declared:
        lines.append("")

    for symbol, found in sorted(call_types.items()):
        entry = found["case"]["adapter_entry"]
        case_ranks = [a["rank"] for a in found["case"]["arguments"]]
        graph_ranks = found["graph_ranks"]
        result_rank = found["result_rank"]
        shape = found["result_shape"]
        count = 1
        # void entries have no result, so no output buffer to size
        for dim in (shape or []):
            count *= dim
        # A void call has no result descriptor, so its parameter list is the
        # operands only; putting result_rank in front would declare a phantom a0
        # and shift every operand by one.
        if found.get("void"):
            parameter_ranks = list(graph_ranks)
        else:
            parameter_ranks = [result_rank] + list(graph_ranks)
        params = ", ".join(f"MemRef{r} *a{i}"
                           for i, r in enumerate(parameter_ranks))
        lines.append(f"/* {found['kind']} {found['shape']} -> {entry}")
        lines.append(f" * graph ranks {[result_rank] + list(graph_ranks)}, "
                     f"kernel ranks {case_ranks} */")
        if found.get("void"):
            # No result: the graph passes every destination as an argument, so
            # the wrapper is a rank cast and nothing else.
            lines.append(f"/* {found['kind']} void call -> {entry} */")
            lines.append(f"void _mlir_ciface_{symbol}({params}) {{")
            lines.append("  " + entry + "(" + ", ".join(
                f"(MemRef{case_ranks[i]} *)a{i}"
                for i in range(len(case_ranks))) + ");")
            lines.append("}")
            lines.append("")
            continue
        lines.append(f"void _mlir_ciface_{symbol}({params}) {{")
        lines.append(f"  float *out = qwen_out_{symbol};")
        if found["kind"] == "linear":
            lines.append(f"  for (int i = 0; i < {count}; ++i) out[i] = 0.0f;")
        # Fill the result descriptor the caller left uninitialised.
        sizes = ", ".join(str(d) for d in shape)
        strides = ", ".join(str(_row_major_stride(shape, i))
                            for i in range(len(shape)))
        # `allocated` carries ownership: the buffer deallocation pass inserts
        # `free(descriptor.allocated)` for call results, which would free a
        # static array and abort with "munmap_chunk(): invalid pointer".
        # Reporting it as NULL marks the result as a view into workspace the
        # caller does not own, which is what it is.
        lines.append(f"  a0->allocated = (void *)0;")
        lines.append(f"  a0->aligned = out;")
        lines.append(f"  a0->offset = 0;")
        for i, dim in enumerate(shape):
            lines.append(f"  a0->sizes[{i}] = {dim};")
            lines.append(f"  a0->strides[{i}] = {_row_major_stride(shape, i)};")
        if found.get("needs_scratch"):
            rows = found["needs_scratch"]["elements"]
            lines.append(f"  MemRef1 scratch = make_1(qwen_rmsnorm_scratch, {rows});")
        cast = {"scratch": "&scratch"}
        call_args = []
        for position, name in enumerate(found["entry_args"]):
            if name == "scratch":
                call_args.append("&scratch")
            else:
                call_args.append(f"(MemRef{case_ranks[position]} *){name}")
        lines.append(f"  {entry}({', '.join(call_args)});")
        lines.append("}")
        lines.append("")
    output.write_text("\n".join(lines) + "\n")
    return max_scratch, total


def _row_major_stride(shape, axis):
    stride = 1
    for dim in shape[axis + 1:]:
        stride *= dim
    return stride


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--triton-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--prefill-len", type=int, default=16)
    parser.add_argument("--max-cache-len", type=int, default=512)
    parser.add_argument("--kv-cache", action="store_true",
                        help="replace the cache writes only (diagnostic: isolates "
                             "them from the attention rewrite)")
    parser.add_argument("--attention", action="store_true",
                        help="replace the fused attention with the QK / mask / "
                             "softmax / PV kernels")
    parser.add_argument("--attention-position", action="store_true",
                        help="opt in to position-bounded QK/PV; requires --attention and separately built kernels")
    parser.add_argument("--attention-native-key", action="store_true",
                        help="read original K cache, omitting layout_k; requires --attention-position")
    parser.add_argument("--w8a8", action="store_true",
                        help="express each linear as quantize -> int8 matmul -> "
                             "dequantize instead of one f32 matmul")
    parser.add_argument("--w8a8-only", action="append", default=None,
                        help="restrict --w8a8 to these node names (diagnostic)")
    parser.add_argument("--no-last-token-slice", action="store_true",
                        help="keep the prefill lm_head at M=S instead of slicing to "
                             "the last position (costs a transposed-weight copy)")
    parser.add_argument("--pattern", action="append", default=None,
                        choices=["linear", "rmsnorm", "silu", "embedding"])
    args = parser.parse_args()
    if args.attention_position and not args.attention:
        parser.error("--attention-position requires --attention")
    if args.attention_native_key and not args.attention_position:
        parser.error("--attention-native-key requires --attention-position")

    import sys
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import import_model as im
    from buddy.compiler.frontend import DynamoCompiler
    from buddy.compiler.graph.type import DeviceType
    from buddy.compiler.graph.operation import PlaceholderOp
    from buddy.compiler.ops import tosa
    from torch._inductor.decomposition import decompositions as decomp
    from transformers import StaticCache

    patterns = args.pattern or ["linear", "rmsnorm", "silu", "embedding"]
    matchers = {"linear": match_linear, "rmsnorm": match_rmsnorm,
                "silu": match_silu, "embedding": match_embedding}

    index = load_kernel_index(args.triton_build)
    model, config, _ = im.build_model(args.assets, args.checkpoint,
                                      torch.float32, args.layers)

    cache = StaticCache(config=model.config, max_cache_len=args.max_cache_len,
                        batch_size=1)
    with torch.no_grad():
        model(input_ids=torch.zeros((1, 1), dtype=torch.int64),
              past_key_values=cache, use_cache=True,
              cache_implementation="static",
              cache_position=torch.tensor([0], dtype=torch.int64))
    for tensor in im.all_cache_tensors(cache):
        tensor.zero_()

    lm_head_rewrites = 0
    report = {
        "stage": "graph -> external Triton call replacement",
        "triton_build": str(args.triton_build),
        "kernel_index_size": len(index),
        "patterns": patterns,
        "graphs": {},
        "replaced": [],
        "uncovered": [],
        "kept_interior": [],
        "not_fpga": True,
    }
    call_types = {}

    def build(kind, ids):
        compiler = DynamoCompiler(
            primary_registry=tosa.ops_registry,
            aot_autograd_decomposition=decomp,
            func_name=f"forward_{kind}",
            enable_external_calls=True,
        )
        kwargs = dict(input_ids=ids, past_key_values=cache, use_cache=True,
                      cache_implementation="static")
        if kind == "decode":
            kwargs["cache_position"] = torch.tensor([args.prefill_len],
                                                    dtype=torch.int64)
        with torch.no_grad():
            graphs = compiler.importer(model, **kwargs)
        return graphs[0], compiler

    args.output.mkdir(parents=True, exist_ok=True)
    for kind, ids in (("prefill", torch.zeros((1, args.prefill_len),
                                              dtype=torch.int64)),
                      ("decode", torch.zeros((1, 1), dtype=torch.int64))):
        graph, compiler = build(kind, ids)
        graph._enable_external_calls = True
        sliced = 0
        if not args.no_last_token_slice:
            sliced = rewrite_lm_head_last_token(graph, report)
            lm_head_rewrites += sliced
        before = len(graph.body)
        counts = {}
        shared_attention = {}
        # attention first: it creates the shared position buffer the KV write
        # then reuses
        if args.attention:
            counts["attention"] = rewrite_attention_to_triton(
                graph, index, call_types, report, shared_attention,
                runtime_length=args.attention_position, native_key=args.attention_native_key)
        if args.attention or args.kv_cache:
            counts["kv_cache"] = rewrite_kv_cache_to_triton(
                graph, index, call_types, report, shared_attention)
        if args.w8a8:
            # The embedding goes first so it can create the tied int8 matrix that
            # the lm_head linear then reuses. Both consume MatmulOps/EmbeddingOps
            # the f32 matchers would otherwise take.
            # tied is per graph: it is keyed by node name, and every graph names
            # its own placeholders independently, so sharing the dict across
            # graphs made the decode lm_head reuse a name that only exists in the
            # prefill graph -- and the prefill lm_head silently made a second
            # 155 MB copy of the tied matrix.
            tied = {}
            counts["w8a8_embedding"] = rewrite_embedding_to_w8a8(
                graph, index, call_types, report, tied)
            counts["w8a8"] = rewrite_linear_to_w8a8(
                graph, index, call_types, report, tied, only=args.w8a8_only)
        for name in patterns:
            counts[name] = replace(graph, matchers[name], index, call_types,
                                   report)
        # NOTE: graph surgery and its reporting both have to happen *after*
        # report["graphs"][kind] is assigned below; writing to it before that
        # assignment is silently overwritten (this bit me once already).
        report["graphs"][kind] = {
            "lm_head_last_token_rewrites": sliced,
            "ops_before": before,
            "ops_after": len(graph.body),
            "pattern_matches": counts,
            "external_calls": sum(
                1 for op in graph.body
                if type(op).__name__ == "CallExternalOp"),
        }
        if args.attention or args.w8a8 or args.kv_cache:
            params_now = list(graph.params)
            inputs_now = list(graph.inputs)
            pruned = prune_dead_nodes(graph, report)
            _reindex(graph, params_now, inputs_now)
            report["graphs"][kind]["pruned_nodes"] = len(pruned)
            # The core claim of the rewrite: the FP32 matrices are gone from the
            # parameter list, so the image no longer has to carry 2.22 GiB.
            report["graphs"][kind]["parameters"] = [
                {"name": p.name, "shape": list(shape_of(p) or []),
                 "dtype": str(p.tensor_meta.get("dtype"))}
                for p in graph.params]
            # Per graph, not the merged list: the workspace buffers are sized by
            # the row count, so prefill's and decode's are different sets and an
            # image must declare exactly the ones its own entry point takes.
            report["graphs"][kind]["workspace_inputs"] = [
                workspace_record(p)
                for p in graph.inputs
                if p.name in {e["name"] for e in
                              (report.get("w8a8_workspace_inputs") or [])}]
        graph.op_groups = {"sg": [op for op in graph.body
                                  if not isinstance(op, PlaceholderOp)]}
        graph.group_map_device = {"sg": DeviceType.CPU}
        from graph_contract import describe_entry_abi, remaining_dense_work
        report["graphs"][kind]["dense_coverage"] = remaining_dense_work(graph)
        graph.lower_to_top_level_ir()
        report["graphs"][kind]["entry_abi"] = describe_entry_abi(graph)
        from buddy.compiler.graph import GraphDriver
        driver = GraphDriver(graph)
        driver.subgraphs[0].lower_to_top_level_ir()
        (args.output / f"subgraph0_{kind}.triton.mlir").write_text(
            str(driver.subgraphs[0]._imported_module))
        (args.output / f"forward_{kind}.triton.mlir").write_text(
            str(driver.construct_main_graph(True)))

    scratch, out_workspace = generate_adapters(
        call_types, args.output / "qwen_triton_adapters.c")
    report["adapter_output_workspace_floats"] = out_workspace
    report["lm_head_last_token_rewrites"] = lm_head_rewrites
    report["distinct_symbols"] = sorted(call_types)
    report["pattern_status"] = {
        name: "implemented" for name in patterns
    }
    report["patterns_not_implemented"] = [
        "RoPE: remains in Buddy graph lowering, including sin/cos and split-half rotation"]
    report["attention_enabled"] = args.attention
    report["kv_cache_enabled"] = args.attention or args.kv_cache
    report["declaration_naming"] = (
        "graph declares `func.func private @<symbol>`; the loaded library must "
        "export `_mlir_ciface_<symbol>` (MLIR emit_c_interface convention)")
    report["adapter_scratch_floats"] = scratch
    report["adapter_source"] = str(args.output / "qwen_triton_adapters.c")
    large_uncovered = sum(len(g["dense_coverage"]["large_uncovered"])
                          for g in report["graphs"].values())
    report["large_uncovered_count"] = large_uncovered
    report["status"] = ("PARTIAL" if report["uncovered"] or large_uncovered
                        else "PASS" if call_types else "NO_MATCHES")
    (args.output / "triton-call-replacement.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("replaced", "uncovered", "kept_interior")},
                     indent=2))
    print(f"\nreplaced {len(report['replaced'])} nodes, "
          f"{len(report['distinct_symbols'])} distinct symbols")
    for name in report["distinct_symbols"]:
        print("   ", name)
    if report["uncovered"]:
        print(f"uncovered {len(report['uncovered'])} (first 5):")
        for entry in report["uncovered"][:5]:
            print("   ", entry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
