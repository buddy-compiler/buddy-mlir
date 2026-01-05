from __future__ import annotations

from .. import Graph
from ..operation import (
    AddOp,
    BatchMatmulOp,
    CatOp,
    ConstantPadNdOp,
    MulOp,
    PermuteOp,
    SumDimOp,
    UnsqueezeOp,
    ViewOp,
)


def affine_grid_generator_simplify(graph: Graph):
    """
    Rewrite the affine_grid_generator decomposition pattern:

        mul(grid[..., 3, 1], unsqueeze(permute(theta))) -> sum(dim=-2)

    into a batch matmul (bmm) form to avoid broadcasted mul + reduce_sum, which
    currently produces incorrect numeric results in some execution paths.
    """
    i = 0
    while i < len(graph.body):
        node = graph.body[i]
        if not isinstance(node, SumDimOp):
            i += 1
            continue

        if len(node.args) < 2:
            i += 1
            continue
        mul_name = node.args[0]
        reduce_dims = node.args[1]
        if reduce_dims not in ([-2], (-2,)):
            i += 1
            continue
        if not isinstance(mul_name, str) or mul_name not in graph.node_table:
            i += 1
            continue

        mul_node = graph.node_table[mul_name]
        if not isinstance(mul_node, MulOp) or len(mul_node.args) != 2:
            i += 1
            continue

        grid_name, theta_name = mul_node.args
        if (
            not isinstance(grid_name, str)
            or not isinstance(theta_name, str)
            or grid_name not in graph.node_table
            or theta_name not in graph.node_table
        ):
            i += 1
            continue

        grid_node = graph.node_table[grid_name]
        theta_node = graph.node_table[theta_name]
        if not isinstance(grid_node, ViewOp):
            i += 1
            continue
        if not isinstance(theta_node, UnsqueezeOp) or len(theta_node.args) != 2:
            i += 1
            continue
        if theta_node.args[1] != 1:
            i += 1
            continue

        permute_name = theta_node.args[0]
        if (
            not isinstance(permute_name, str)
            or permute_name not in graph.node_table
        ):
            i += 1
            continue
        permute_node = graph.node_table[permute_name]
        if not isinstance(permute_node, PermuteOp):
            i += 1
            continue

        grid_shape = list(grid_node.tensor_meta.get("shape", []))
        permute_shape = list(permute_node.tensor_meta.get("shape", []))
        if len(grid_shape) != 3 or len(permute_shape) != 3:
            i += 1
            continue
        if int(grid_shape[2]) != 1:
            i += 1
            continue
        if int(permute_shape[1]) != int(grid_shape[1]):
            i += 1
            continue
        if int(permute_shape[2]) != 2:
            i += 1
            continue

        n = int(permute_shape[0])
        hw = int(grid_shape[0])
        k = int(grid_shape[1])

        insert_pos = i

        # Build grid view: (hw, k, 1) -> (1, hw, k)
        grid_view = ViewOp()
        grid_view.name = f"{node.name}_grid_view"
        grid_view.add_argument(grid_name)
        grid_view.add_argument([1, hw, k])
        grid_view.tensor_meta["shape"] = [1, hw, k]
        grid_view.tensor_meta["dtype"] = grid_node.tensor_meta.get("dtype")
        grid_view.add_parent(grid_name)
        grid_node.add_children(grid_view.name)

        graph.node_table[grid_view.name] = grid_view
        graph.body.insert(insert_pos, grid_view)
        insert_pos += 1

        grid_arg_for_bmm = grid_view.name

        # Expand to batch if needed: (1, hw, k) -> (n, hw, k)
        if n != 1:
            from ..operation import ExpandOp

            grid_expand = ExpandOp()
            grid_expand.name = f"{node.name}_grid_expand"
            grid_expand.add_argument(grid_view.name)
            grid_expand.add_argument([n, hw, k])
            grid_expand.tensor_meta["shape"] = [n, hw, k]
            grid_expand.tensor_meta["dtype"] = grid_node.tensor_meta.get(
                "dtype"
            )
            grid_expand.add_parent(grid_view.name)
            grid_view.add_children(grid_expand.name)
            grid_expand.add_children(node.name)
            grid_arg_for_bmm = grid_expand.name
            graph.node_table[grid_expand.name] = grid_expand
            graph.body.insert(insert_pos, grid_expand)
            insert_pos += 1

        # Replace the reduce-sum node with bmm(grid, permuted_theta).
        bmm = BatchMatmulOp()
        bmm.name = node.name
        bmm.add_argument(grid_arg_for_bmm)
        bmm.add_argument(permute_name)
        bmm.tensor_meta = node.tensor_meta
        bmm.add_parent(grid_arg_for_bmm)
        bmm.add_parent(permute_name)
        for child in node._children:
            bmm.add_children(child)
        permute_node.add_children(bmm.name)

        # Insert grid_view before the bmm.
        # Update node table and body for the replaced node.
        graph.node_table[bmm.name] = bmm
        graph.body[insert_pos] = bmm

        # Detach old dependency: SumDimOp no longer depends on mul_node.
        if bmm.name in mul_node._children:
            mul_node._children.remove(bmm.name)
        if mul_name in bmm._parents:
            bmm._parents.remove(mul_name)

        # Remove mul + unsqueeze nodes to avoid executing the problematic path.
        try:
            if graph.check_delete_node(mul_node):
                parents = [graph.node_table[p] for p in mul_node._parents]
                graph.delete_node(mul_node, parents)
        except Exception:
            pass
        try:
            if graph.check_delete_node(theta_node):
                parents = [graph.node_table[p] for p in theta_node._parents]
                graph.delete_node(theta_node, parents)
        except Exception:
            pass

        # Only expect one match per graph; stop once rewritten.
        return

    return


def affine_grid_generator_homogeneous_base_simplify(graph: Graph):
    _rewrite_affine_grid_homogeneous_base(graph)


def _rewrite_affine_grid_homogeneous_base(graph: Graph):
    # Find the ViewOp that reshapes the homogeneous grid to (hw, 3).
    target_view = None
    for node in graph.body:
        if not isinstance(node, ViewOp) or len(node.args) < 2:
            continue
        shape = node.args[1]
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            continue
        if int(shape[1]) != 3:
            continue
        target_view = node
        break
    if target_view is None:
        return
    if not isinstance(target_view.args[0], str):
        return
    add2_name = target_view.args[0]
    add2 = graph.node_table.get(add2_name)
    if not isinstance(add2, AddOp) or len(add2.args) != 2:
        return

    add1_name, pad_c_name = add2.args
    add1 = graph.node_table.get(add1_name)
    pad_c = graph.node_table.get(pad_c_name)
    if not isinstance(add1, AddOp) or len(add1.args) != 2:
        return
    if not isinstance(pad_c, ConstantPadNdOp):
        return
    pad_a = graph.node_table.get(add1.args[0])
    pad_b = graph.node_table.get(add1.args[1])
    if not isinstance(pad_a, ConstantPadNdOp) or not isinstance(
        pad_b, ConstantPadNdOp
    ):
        return

    pads = [pad_a, pad_b, pad_c]
    sources_by_pad = {}
    for pad in pads:
        if len(pad.args) < 2:
            return
        src, pad_list = pad.args[0], pad.args[1]
        if not isinstance(src, str) or not isinstance(pad_list, (list, tuple)):
            return
        sources_by_pad[tuple(int(x) for x in pad_list)] = src

    x_src = sources_by_pad.get((0, 2))
    y_src = sources_by_pad.get((1, 1))
    ones_src = sources_by_pad.get((2, 0))
    if not x_src or not y_src or not ones_src:
        return

    x_node = graph.node_table.get(x_src)
    y_node = graph.node_table.get(y_src)
    ones_node = graph.node_table.get(ones_src)
    if x_node is None or y_node is None or ones_node is None:
        return

    out_shape = list(add2.tensor_meta.get("shape", []))
    out_dtype = add2.tensor_meta.get("dtype")
    if len(out_shape) != 3 or int(out_shape[2]) != 3:
        return

    cat = CatOp()
    cat.name = f"{target_view.name}_cat"
    cat.add_argument([x_src, y_src, ones_src])
    cat.add_argument(2)
    cat.tensor_meta["shape"] = out_shape
    cat.tensor_meta["dtype"] = out_dtype
    cat.add_parent(x_src)
    cat.add_parent(y_src)
    cat.add_parent(ones_src)
    cat.add_children(target_view.name)

    x_node.add_children(cat.name)
    y_node.add_children(cat.name)
    ones_node.add_children(cat.name)

    graph.node_table[cat.name] = cat
    idx = graph.body.index(target_view)
    graph.body.insert(idx, cat)

    # Rewire the view to take cat output instead of the add chain.
    target_view.args[0] = cat.name
    if add2_name in target_view._parents:
        target_view._parents.remove(add2_name)
    if cat.name not in target_view._parents:
        target_view._parents.append(cat.name)
    if target_view.name in add2._children:
        add2._children.remove(target_view.name)

    # Delete the now-dead add/pad nodes (best-effort).
    for dead in (add2, add1, pad_a, pad_b, pad_c):
        try:
            if graph.check_delete_node(dead):
                parents = [graph.node_table[p] for p in dead._parents]
                graph.delete_node(dead, parents)
        except Exception:
            pass
