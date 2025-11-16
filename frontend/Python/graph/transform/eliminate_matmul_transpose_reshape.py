# ===- eliminate_matmul_transpose_reshape.py --------------------------------
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
# Eliminate transpose operations before reshape when the data layout doesn't
# actually change (data is contiguous). Instead of transpose -> reshape, we
# directly reshape the input to the final shape.
#
# ===---------------------------------------------------------------------------

from .. import Graph
from ..operation import *
from ..type import TensorDType
import torch


def eliminate_matmul_transpose_reshape(graph: Graph):
    """
    Eliminate transpose operations before reshape when the data layout doesn't
    actually change (data is contiguous). This optimization directly reshapes
    the input to the final shape, skipping the transpose operation.

    This handles patterns like:
    - AnyOp -> TransposeOp -> ReshapeOp/ViewOp/UnsqueezeOp
    - AnyOp -> ViewOp -> TransposeOp -> ReshapeOp

    Only applies to f32 dtype.

    Args:
        graph (Graph): The Graph to be optimized.
    """
    nodes_to_remove = []
    stats = {"checked": 0, "found_pattern": 0, "eliminated": 0}

    # Find all transpose/permute nodes and check if they can be eliminated
    for node in graph.body:
        # Check if this is a TransposeOp or PermuteOp
        is_transpose_op = isinstance(node, TransposeOp)
        is_permute_op = isinstance(node, PermuteOp)

        if not (is_transpose_op or is_permute_op):
            continue
        stats["checked"] += 1
        transpose_node = node
        transpose_node_name = node.name

        # Check dtype is f32
        node_dtype = transpose_node.tensor_meta.get("dtype")
        if node_dtype != TensorDType.Float32:
            continue

        # Check transpose has at least one child
        if len(transpose_node._children) == 0:
            continue

        # Check if all children are reshape-like ops
        all_children_are_reshape = True
        reshape_children = []

        for child_name in transpose_node._children:
            if child_name not in graph.node_table:
                all_children_are_reshape = False
                break
            child_node = graph.node_table[child_name]
            is_reshape_op = isinstance(child_node, ReshapeOp)
            is_view_op = isinstance(child_node, ViewOp)
            is_unsqueeze_op = isinstance(child_node, UnsqueezeOp)
            if is_reshape_op or is_view_op or is_unsqueeze_op:
                reshape_children.append((child_name, child_node))
            else:
                all_children_are_reshape = False

        # If single child and it's a reshape, use the existing logic
        # If multiple children or non-reshape children, we'll reshape directly to transpose output shape
        if len(transpose_node._children) == 1 and all_children_are_reshape:
            reshape_node_name = transpose_node._children[0]
            reshape_node = graph.node_table[reshape_node_name]
            is_reshape_op = isinstance(reshape_node, ReshapeOp)
            is_view_op = isinstance(reshape_node, ViewOp)
            is_unsqueeze_op = isinstance(reshape_node, UnsqueezeOp)
            use_final_shape = False
        else:
            # Multiple children or non-reshape children: reshape directly to transpose output shape
            reshape_node = None
            reshape_node_name = None
            is_reshape_op = False
            is_view_op = False
            is_unsqueeze_op = False
            use_final_shape = True

        # Get the input node (skip no-op reshapes before transpose)
        if len(transpose_node.args) < 1:
            continue

        input_arg = transpose_node.args[0]
        if str(input_arg) not in graph.node_table:
            continue

        current_input = graph.node_table[str(input_arg)]
        skipped_nodes = []

        # Skip up to 3 levels of no-op reshapes before transpose
        for _ in range(3):
            is_view = isinstance(current_input, ViewOp)
            is_reshape = isinstance(current_input, ReshapeOp)

            if is_view or is_reshape:
                # No-op reshape: same input/output shape
                if len(current_input.args) > 0:
                    parent_name = str(current_input.args[0])
                    if parent_name in graph.node_table:
                        parent_node = graph.node_table[parent_name]
                        input_shape = list(parent_node.tensor_meta["shape"])
                        output_shape = list(current_input.tensor_meta["shape"])
                        if input_shape == output_shape:
                            # It's a no-op, go to parent
                            skipped_nodes.append(
                                (current_input.name, current_input)
                            )
                            current_input = parent_node
                            continue
            break

        input_node = current_input
        input_node_name = input_node.name

        # Get shapes
        input_shape = list(input_node.tensor_meta["shape"])
        transpose_output_shape = list(transpose_node.tensor_meta["shape"])

        # For multiple children case, we'll reshape to transpose output shape
        if use_final_shape:
            reshape_output_shape = transpose_output_shape
        else:
            reshape_output_shape = list(reshape_node.tensor_meta["shape"])

        # Get transpose permutation
        perm = None
        if is_permute_op:
            # PermuteOp: args[1] is a full permutation list
            if len(transpose_node.args) < 2:
                continue
            perm = transpose_node.args[1]
            if not isinstance(perm, (list, tuple)):
                continue
            perm = list(perm)
        elif is_transpose_op:
            # TransposeOp has two possible conventions:
            # 1) TOSA-style: args[1] is a full permutation list (e.g. [0, 2, 1, 3])
            # 2) PyTorch-style: args[1], args[2] are two dimensions to swap
            if len(transpose_node.args) >= 2 and isinstance(
                transpose_node.args[1], (list, tuple)
            ):
                # Use the full permutation list directly
                perm = list(transpose_node.args[1])
            else:
                # Fallback to "swap two dimensions" convention
                if len(transpose_node.args) < 3:
                    continue
                dim1 = int(transpose_node.args[1])
                dim2 = int(transpose_node.args[2])
                if dim1 < 0:
                    dim1 += len(input_shape)
                if dim2 < 0:
                    dim2 += len(input_shape)
                if (
                    dim1 < 0
                    or dim1 >= len(input_shape)
                    or dim2 < 0
                    or dim2 >= len(input_shape)
                ):
                    continue
                # Create permutation list for transpose
                perm = list(range(len(input_shape)))
                perm[dim1], perm[dim2] = perm[dim2], perm[dim1]

        if perm is None:
            continue

        # Validate permutation
        if len(perm) != len(input_shape) or set(perm) != set(
            range(len(input_shape))
        ):
            continue

        # Calculate expected transpose output shape
        expected_transpose_shape = [input_shape[i] for i in perm]
        # Check if transpose output shape matches expected
        if transpose_output_shape != expected_transpose_shape:
            continue

        # Only eliminate transpose when it does not change the logical layout:
        # identity perm is always safe; otherwise, the relative order of dims with size > 1
        # must be preserved.
        identity_perm = list(range(len(input_shape)))
        if perm != identity_perm:
            # Collect indices of all non-unit dimensions
            non_unit_indices = [i for i, d in enumerate(input_shape) if d > 1]
            before = non_unit_indices
            after = [axis for axis in perm if axis in non_unit_indices]
            # If the relative order of non-unit dimensions changes, layout changes and we skip
            if before != after:
                continue
        # At this point, either perm is identity, or it only moves size-1 dimensions.
        # The logical layout is preserved, so transpose can be treated as a no-op.
        # Calculate total elements
        input_elements = 1
        for dim in input_shape:
            input_elements *= dim

        transpose_elements = 1
        for dim in transpose_output_shape:
            transpose_elements *= dim

        reshape_elements = 1
        for dim in reshape_output_shape:
            reshape_elements *= dim

        # All should have the same number of elements
        if input_elements != transpose_elements:
            continue

        # Get reshape new_shape
        if use_final_shape:
            # For multiple children case, reshape directly to transpose output shape
            new_shape = transpose_output_shape
        elif is_unsqueeze_op:
            # For UnsqueezeOp, use the output shape as the target shape
            new_shape = reshape_output_shape
        else:
            # For ReshapeOp/ViewOp, get new_shape from args[1]
            if len(reshape_node.args) < 2:
                continue

            new_shape = reshape_node.args[1]
            if not isinstance(new_shape, (list, tuple)):
                continue

            new_shape = list(new_shape)

            # Handle -1 in new_shape (inferred dimension)
            if -1 in new_shape:
                reshape_elements = 1
                for dim in reshape_output_shape:
                    reshape_elements *= dim
                total_elements = reshape_elements
                inferred_size = total_elements
                for dim in new_shape:
                    if dim != -1:
                        inferred_size //= dim
                new_shape = [
                    inferred_size if dim == -1 else dim for dim in new_shape
                ]

            # Verify reshape output shape matches new_shape
            if reshape_output_shape != new_shape:
                continue

        stats["found_pattern"] += 1

        # Now we can eliminate the transpose:
        stats["eliminated"] += 1

        # Handle multiple children case: create a new reshape node
        if use_final_shape:
            # Create a new ReshapeOp to replace the transpose
            new_reshape_node = ReshapeOp()
            # Ensure unique node name
            new_node_name = transpose_node_name + "_reshaped"
            counter = 0
            while new_node_name in graph.node_table:
                counter += 1
                new_node_name = (
                    transpose_node_name + "_reshaped_" + str(counter)
                )
            new_reshape_node.name = new_node_name
            new_reshape_node._arguments = [input_node_name, new_shape]
            # Set tensor_meta with correct output shape
            new_reshape_node.tensor_meta = transpose_node.tensor_meta.copy()
            new_reshape_node.tensor_meta["shape"] = tuple(new_shape)
            # _op_type is already set to OpType.ReshapeType by ReshapeOp constructor
            new_reshape_node._parents = [input_node_name]
            new_reshape_node._children = list(transpose_node._children)

            # Add the new reshape node to graph FIRST, before updating references
            # Insert the new node right after transpose_node in graph.body to maintain correct order
            # IMPORTANT: We need to ensure the new node is inserted BEFORE transpose_node in graph.body
            # so that when we process nodes in order, the new node's input (input_node_name)
            # has already been processed to symbol_table
            graph.node_table[new_reshape_node.name] = new_reshape_node
            if transpose_node in graph.body:
                transpose_idx = graph.body.index(transpose_node)
                # Insert right after transpose_node, but before any children of transpose_node
                # Actually, we should insert it right after the input_node to ensure input_node is processed first
                # But since input_node comes before transpose_node, inserting after transpose_node should be fine
                graph.body.insert(transpose_idx + 1, new_reshape_node)
            else:
                graph.body.append(new_reshape_node)

            # Update all children to use the new reshape node instead of transpose
            # Copy children list first in case it gets modified during iteration
            transpose_children = list(transpose_node._children)
            for child_name in transpose_children:
                if child_name in graph.node_table:
                    child_node = graph.node_table[child_name]
                    # Update all args that reference transpose_node
                    for i, arg in enumerate(child_node.args):
                        if str(arg) == transpose_node_name:
                            child_node.args[i] = new_reshape_node.name
                    # Update parents list
                    if transpose_node_name in child_node._parents:
                        idx = child_node._parents.index(transpose_node_name)
                        child_node._parents[idx] = new_reshape_node.name

            # Update input node's children
            for skipped_name, _ in skipped_nodes:
                if skipped_name in input_node._children:
                    input_node._children.remove(skipped_name)
            if transpose_node_name in input_node._children:
                input_node._children.remove(transpose_node_name)
            if new_reshape_node.name not in input_node._children:
                input_node._children.append(new_reshape_node.name)

            # IMPORTANT: Update transpose_node's args and parents to skip skipped_nodes
            # This is critical - transpose_node may still reference skipped_nodes in its args
            if len(transpose_node.args) > 0:
                if str(transpose_node.args[0]) in [
                    sn[0] for sn in skipped_nodes
                ]:
                    transpose_node.args[0] = input_node_name
                # Also update _arguments directly
                if len(transpose_node._arguments) > 0:
                    if str(transpose_node._arguments[0]) in [
                        sn[0] for sn in skipped_nodes
                    ]:
                        transpose_node._arguments[0] = input_node_name
            # Update transpose_node's parents
            for skipped_name, _ in skipped_nodes:
                if skipped_name in transpose_node._parents:
                    idx = transpose_node._parents.index(skipped_name)
                    transpose_node._parents[idx] = input_node_name

            # Update skipped nodes: update all nodes that reference skipped_nodes
            for skipped_name, skipped_node in skipped_nodes:
                # Update all nodes that reference this skipped node
                for node in graph.body:
                    if node.name == skipped_name:
                        continue
                    # Update args
                    for i, arg in enumerate(node.args):
                        if str(arg) == skipped_name:
                            node.args[i] = input_node_name
                    # Update parents
                    if skipped_name in node._parents:
                        idx = node._parents.index(skipped_name)
                        node._parents[idx] = input_node_name

                # Update input_node's children
                if skipped_name in input_node._children:
                    input_node._children.remove(skipped_name)
                skipped_node._children.clear()
                nodes_to_remove.append(skipped_node)

            # Clear transpose node's children and mark for removal
            transpose_node._children.clear()
            nodes_to_remove.append(transpose_node)
            continue  # Skip the rest of the logic for single reshape child case

        # Update reshape node to use input directly (skipping transpose and intermediate nodes)
        # For UnsqueezeOp, we need to convert it to ReshapeOp since we're doing a full reshape
        if is_unsqueeze_op:
            # Create a new ReshapeOp to replace UnsqueezeOp
            new_reshape_node = ReshapeOp()
            new_reshape_node.name = reshape_node.name
            new_reshape_node._arguments = [input_node_name, new_shape]
            new_reshape_node.tensor_meta = reshape_node.tensor_meta.copy()
            # _op_type is already set to OpType.ReshapeType by ReshapeOp constructor
            new_reshape_node._parents = [input_node_name]
            new_reshape_node._children = list(reshape_node._children)

            # Update children to point to new node
            for child_name in new_reshape_node._children:
                if child_name in graph.node_table:
                    child_node = graph.node_table[child_name]
                    for i, arg in enumerate(child_node.args):
                        if str(arg) == reshape_node_name:
                            child_node.args[i] = new_reshape_node.name
                    if reshape_node_name in child_node._parents:
                        idx = child_node._parents.index(reshape_node_name)
                        child_node._parents[idx] = new_reshape_node.name

            # Replace in graph
            graph.node_table[new_reshape_node.name] = new_reshape_node
            if reshape_node in graph.body:
                idx = graph.body.index(reshape_node)
                graph.body[idx] = new_reshape_node

            # Update reshape_node reference
            reshape_node = new_reshape_node
            reshape_node_name = new_reshape_node.name
        else:
            # For ReshapeOp/ViewOp, just update the input
            if len(reshape_node.args) > 0:
                # Check if args[0] matches transpose_node_name (could be string or node name)
                input_arg = reshape_node.args[0]
                if (
                    str(input_arg) == transpose_node_name
                    or input_arg == transpose_node_name
                ):
                    reshape_node.args[0] = input_node_name
                    # Also update new_shape if needed (for ReshapeOp)
                    if is_reshape_op and len(reshape_node.args) > 1:
                        reshape_node.args[1] = new_shape

            # Also ensure we update _arguments directly to be safe
            if len(reshape_node._arguments) > 0:
                if str(reshape_node._arguments[0]) == transpose_node_name:
                    reshape_node._arguments[0] = input_node_name

        # Update parent/child relationships for reshape_node
        if transpose_node_name in reshape_node._parents:
            idx = reshape_node._parents.index(transpose_node_name)
            reshape_node._parents[idx] = input_node_name

        # IMPORTANT: Update ALL children of transpose_node to reference input_node_name instead
        # This is critical - we need to update ALL children, not just reshape_node
        # Similar to eliminate_weight_transpose.py logic
        transpose_children = list(transpose_node._children)
        for child_name in transpose_children:
            if child_name not in graph.node_table:
                continue
            child_node = graph.node_table[child_name]
            # Update all args that reference transpose_node
            for i, arg in enumerate(child_node.args):
                if str(arg) == transpose_node_name:
                    # For reshape_node, we already updated it above to input_node_name
                    # For other children (shouldn't happen in use_final_shape=False case), update to input_node_name
                    if child_name == reshape_node_name:
                        # Already updated above, but double-check
                        if child_node.args[i] != input_node_name:
                            child_node.args[i] = input_node_name
                    else:
                        # This shouldn't happen in use_final_shape=False case, but handle it
                        child_node.args[i] = input_node_name
            # Update parents list
            if transpose_node_name in child_node._parents:
                idx = child_node._parents.index(transpose_node_name)
                child_node._parents[idx] = input_node_name

        # Update input node's children: remove skipped nodes and transpose, add reshape
        # Remove all skipped nodes from input node's children
        for skipped_name, _ in skipped_nodes:
            if skipped_name in input_node._children:
                input_node._children.remove(skipped_name)
        # Remove transpose from input node's children
        if transpose_node_name in input_node._children:
            input_node._children.remove(transpose_node_name)
        # Add reshape to input node's children
        if reshape_node_name not in input_node._children:
            input_node._children.append(reshape_node_name)

        # IMPORTANT: Update transpose_node's args and parents to skip skipped_nodes
        # This is critical - transpose_node may still reference skipped_nodes in its args
        if len(transpose_node.args) > 0:
            if str(transpose_node.args[0]) in [sn[0] for sn in skipped_nodes]:
                transpose_node.args[0] = input_node_name
            # Also update _arguments directly
            if len(transpose_node._arguments) > 0:
                if str(transpose_node._arguments[0]) in [
                    sn[0] for sn in skipped_nodes
                ]:
                    transpose_node._arguments[0] = input_node_name
        # Update transpose_node's parents
        for skipped_name, _ in skipped_nodes:
            if skipped_name in transpose_node._parents:
                idx = transpose_node._parents.index(skipped_name)
                transpose_node._parents[idx] = input_node_name

        # Update skipped nodes: update all nodes that reference skipped_nodes
        for skipped_name, skipped_node in skipped_nodes:
            # Update all nodes that reference this skipped node
            for node in graph.body:
                if node.name == skipped_name:
                    continue
                # Update args
                for i, arg in enumerate(node.args):
                    if str(arg) == skipped_name:
                        node.args[i] = input_node_name
                # Update parents
                if skipped_name in node._parents:
                    idx = node._parents.index(skipped_name)
                    node._parents[idx] = input_node_name

            # Update input_node's children
            if skipped_name in input_node._children:
                input_node._children.remove(skipped_name)
            skipped_node._children.clear()
            nodes_to_remove.append(skipped_node)

        # Clear transpose node's children
        transpose_node._children.clear()
        nodes_to_remove.append(transpose_node)

    # Remove transpose nodes from graph
    # Before removing, update all references to removed nodes
    for node_to_remove in nodes_to_remove:
        removed_node_name = node_to_remove.name
        # Find the replacement node name (if any)
        replacement_name = None
        # Check if this is a transpose node that was replaced by a new reshape node
        if removed_node_name.endswith("_reshaped") or any(
            n.name == removed_node_name + "_reshaped"
            for n in graph.body
            if hasattr(n, "name")
        ):
            # This shouldn't happen, but handle it
            continue
        # For transpose nodes, find what replaced them
        for node in graph.body:
            if node.name == removed_node_name:
                continue
            # Check if any args reference the removed node
            for i, arg in enumerate(node.args):
                if str(arg) == removed_node_name:
                    # Try to find replacement - check if there's a _reshaped node
                    possible_replacement = removed_node_name + "_reshaped"
                    if possible_replacement in graph.node_table:
                        node.args[i] = possible_replacement
                    else:
                        # This should have been updated earlier
                        pass
            # Update parents
            if removed_node_name in node._parents:
                idx = node._parents.index(removed_node_name)
                possible_replacement = removed_node_name + "_reshaped"
                if possible_replacement in graph.node_table:
                    node._parents[idx] = possible_replacement
                else:
                    # This should have been updated earlier
                    pass

    # Now remove the nodes
    # Only remove nodes that are NOT newly created nodes
    nodes_to_remove_filtered = []
    for node in nodes_to_remove:
        # Skip if this is a newly created node (should not happen, but be safe)
        if node.name.endswith("_reshaped"):
            continue
        nodes_to_remove_filtered.append(node)

    for node in nodes_to_remove_filtered:
        if node.name in graph.node_table:
            del graph.node_table[node.name]
        if node in graph.body:
            graph.body.remove(node)

    # Stats are collected but not printed; callers may use them if needed.
