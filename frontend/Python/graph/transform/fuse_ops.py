# ===- fuse_ops.py -------------------------------------------------------------
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
# Construct op fusion pattern.
#
# ===---------------------------------------------------------------------------

from .. import Graph
from ..operation import *
from .. import DeviceType
from torch.fx.immutable_collections import immutable_list

classicfuse_register = {
    "transpose_matmul_fusion": TransposeMatmulFusedOp,
    "qkv_fusion": QKVFusedOp,
}

# TODO: classify op type for op fusion
# OP_TYPE_FUSABLE = [OpType.BroadcastType, OpType.ElementwiseType, OpType.ReshapeType]
# OP_TYPE_UNFUSABLE = [OpType.Unfusable, OpType.ConcatType]
# OP_TYPE_FUSABLE_BY_SPECIFIC_PASS = []
# ANCHOR_OP_TYPE = []


def classic_fuse_check(graph: Graph):
    """
    Function to identifies and fuses PermuteOp operations with preceding
    MatmulOp operations in a computation graph to optimize performance.

    Args:
        graph (Graph): The computation graph to analyze and optimize.

    Returns:
        None
    """
    for op in graph.body:
        pattern = None
        if isinstance(op, MatmulOp):
            parentop = [graph.node_table[str(i)] for i in op._parents]
            for target in parentop:
                if isinstance(target, PermuteOp) and target.args[
                    1
                ] == immutable_list([1, 0]):
                    pattern = target, parentop, "transpose_matmul_fusion"
        if pattern:
            transpose_matmul_fusion(
                graph, op, pattern[0], pattern[1], pattern[2]
            )


def transpose_matmul_fusion(
    graph: Graph, node, target: Op, parents: List[Op], pattern: str
):
    """
    Function to fuse some typical operations into one operation.
    Such as transpose + matmul
    Args:
    - graph (Graph): The input graph to be simplified.
    - node (Op): The operation to be fused.
    - target (Op): The target operation to be fused.
    - parents (List[Op]): The parents of the node to be fused.
    - pattern (str): The pattern of the fusion.
    Returns:
    - None: Modifies the input graph in place.
    """
    fused_op = classicfuse_register.get(pattern)()
    # matmulop -> fusedmatmulopnode
    fused_op.name = "fused" + node.name
    graph.displace_node(node, fused_op)
    fused_op.args.pop(fused_op.args.index(target.name))
    fused_op._parents.pop(fused_op._parents.index(target.name))
    fused_op.args.extend(target.args)

    fused_op._parents.extend(target._parents)
    targets_parent = [graph.node_table[i] for i in target._parents]
    for i in targets_parent:
        i.add_children(fused_op.name)
    target._children.pop(target._children.index(fused_op.name))

    if graph.check_delete_node(target):
        graph.delete_node(target, targets_parent)


def qkv_fuse_check(graph: Graph):
    """
    Function to identify and fuse QKV linear transformation patterns.

    Identifies three AddMMOp operations that:
    1. Use the same input tensor (through different view operations)
    2. Have output dimensions matching Q(1536), K(256), V(256) pattern
    3. Are positioned close to each other in the computation graph

    Args:
        graph (Graph): The computation graph to analyze and optimize.

    Returns:
        None
    """
    fusion_count = 0
    max_fusions = 1000  # limit max fusion count to avoid infinite loop

    while fusion_count < max_fusions:
        nodes = graph.body
        addmm_nodes = [node for node in nodes if isinstance(node, AddMMOp)]

        found_pattern = False

        # Look for QKV patterns in groups of 3 AddMMOp nodes
        for i in range(len(addmm_nodes) - 2):
            q_node = addmm_nodes[i]
            k_node = addmm_nodes[i + 1]
            v_node = addmm_nodes[i + 2]

            # Extract shapes - handle both torch.Size and list formats
            q_shape = _extract_shape(q_node)
            k_shape = _extract_shape(k_node)
            v_shape = _extract_shape(v_node)

            # Check for QKV pattern: Q[40,1536], K[40,256], V[40,256]
            if (
                q_shape
                and k_shape
                and v_shape
                and len(q_shape) == 2
                and len(k_shape) == 2
                and len(v_shape) == 2
                and q_shape[1] == 1536
                and k_shape[1] == 256
                and v_shape[1] == 256
                and q_shape[0] == k_shape[0] == v_shape[0]
            ):  # Same batch/sequence dimension

                # Check if they use the same input (through view operations)
                shared_input = _find_shared_input(graph, q_node, k_node, v_node)
                if shared_input:
                    qkv_fusion(graph, q_node, k_node, v_node, shared_input)
                    fusion_count += 1
                    found_pattern = True
                    break  # restart search after fusion

        if not found_pattern:
            break  # found no more patterns, exit loop


def _extract_shape(node):
    """Extract shape from tensor_meta, handling different formats"""
    if not hasattr(node, "tensor_meta") or not node.tensor_meta:
        return None

    shape = node.tensor_meta.get("shape", None)
    if shape is None:
        return None

    # Handle torch.Size objects
    if hasattr(shape, "size"):
        return list(shape.size())
    elif hasattr(shape, "__iter__"):
        return list(shape)
    else:
        return None


def _find_shared_input(graph: Graph, q_node, k_node, v_node):
    """
    Find the shared input tensor used by Q, K, V computations.

    Args:
        graph: The computation graph
        q_node, k_node, v_node: The three AddMMOp nodes

    Returns:
        str: Name of the shared input tensor, or None if not found
    """
    # AddMMOp args format: [bias, input, weight]
    q_input = q_node.args[1] if len(q_node.args) > 1 else None
    k_input = k_node.args[1] if len(k_node.args) > 1 else None
    v_input = v_node.args[1] if len(v_node.args) > 1 else None

    if not (q_input and k_input and v_input):
        return None

    # Check if they are view operations of the same tensor
    q_parent = graph.node_table.get(q_input)
    k_parent = graph.node_table.get(k_input)
    v_parent = graph.node_table.get(v_input)

    if (
        q_parent
        and k_parent
        and v_parent
        and isinstance(q_parent, ViewOp)
        and isinstance(k_parent, ViewOp)
        and isinstance(v_parent, ViewOp)
    ):
        # Check if all view operations have the same input
        q_source = q_parent.args[0] if q_parent.args else None
        k_source = k_parent.args[0] if k_parent.args else None
        v_source = v_parent.args[0] if v_parent.args else None

        if q_source == k_source == v_source:
            return q_source

    # Alternative: check if they directly use the same input (without view operations)
    if q_input == k_input == v_input:
        return q_input

    return None


def qkv_fusion(graph: Graph, q_node, k_node, v_node, shared_input):
    """
    Fuse three QKV AddMMOp operations into a single QKVFusedOp.

    Args:
        graph: The computation graph
        q_node, k_node, v_node: The three AddMMOp nodes to fuse
        shared_input: The shared input tensor name
    """
    # Verify that all bias PlaceholderOp nodes exist
    bias_names = [q_node.args[0], k_node.args[0], v_node.args[0]]
    for bias_name in bias_names:
        if bias_name not in graph.node_table:
            raise ValueError(
                f"Bias PlaceholderOp {bias_name} not found in graph.node_table"
            )
        bias_node = graph.node_table[bias_name]
        if (
            not hasattr(bias_node, "__class__")
            or bias_node.__class__.__name__ != "PlaceholderOp"
        ):
            raise ValueError(
                f"Bias node {bias_name} is not a PlaceholderOp, got {bias_node.__class__.__name__}"
            )

    # Create the fused operation
    fused_op = QKVFusedOp()
    fused_op.name = f"qkv_fused_{q_node.name}_{k_node.name}_{v_node.name}"

    # Store original dimensions for later splitting
    q_shape = _extract_shape(q_node)
    k_shape = _extract_shape(k_node)
    v_shape = _extract_shape(v_node)

    fused_op.q_dim = q_shape[1] if q_shape and len(q_shape) > 1 else 1536
    fused_op.k_dim = k_shape[1] if k_shape and len(k_shape) > 1 else 256
    fused_op.v_dim = v_shape[1] if v_shape and len(v_shape) > 1 else 256

    # Extract original weights and biases from Q, K, V nodes
    # AddMMOp args: [bias, input, weight]
    q_bias = q_node.args[0] if len(q_node.args) > 0 else None
    q_weight = q_node.args[2] if len(q_node.args) > 2 else None

    k_bias = k_node.args[0] if len(k_node.args) > 0 else None
    k_weight = k_node.args[2] if len(k_node.args) > 2 else None

    v_bias = v_node.args[0] if len(v_node.args) > 0 else None
    v_weight = v_node.args[2] if len(v_node.args) > 2 else None

    # Verify that all required nodes exist
    for name, node_name in [
        ("Q weight", q_weight),
        ("K weight", k_weight),
        ("V weight", v_weight),
        ("Q bias", q_bias),
        ("K bias", k_bias),
        ("V bias", v_bias),
    ]:
        if node_name not in graph.node_table:
            raise ValueError(
                f"{name}: {node_name} not found in graph.node_table"
            )

    # Get original node dtype
    original_dtype = (
        q_node.tensor_meta.get("dtype")
        if hasattr(q_node, "tensor_meta") and q_node.tensor_meta
        else None
    )

    # Create concatenation operations for weights using 2 steps (Q+K, then +V)
    # Step 1: Concatenate Q and K weights
    qk_weight_concat_op = CatOp()
    qk_weight_concat_op.name = f"{fused_op.name}_qk_weight_concat"
    qk_weight_concat_op._arguments = [
        [q_weight, k_weight],
        1,
    ]  # concat along dim 1
    qk_weight_concat_op.tensor_meta = {
        "shape": [1536, fused_op.q_dim + fused_op.k_dim],
        "dtype": original_dtype,
    }
    qk_weight_concat_op._parents = [q_weight, k_weight]

    # Step 2: Concatenate QK result with V weight
    weight_concat_op = CatOp()
    weight_concat_op.name = f"{fused_op.name}_weight_concat"
    weight_concat_op._arguments = [
        [qk_weight_concat_op.name, v_weight],
        1,
    ]  # concat along dim 1
    weight_concat_op.tensor_meta = {
        "shape": [1536, fused_op.q_dim + fused_op.k_dim + fused_op.v_dim],
        "dtype": original_dtype,
    }
    weight_concat_op._parents = [qk_weight_concat_op.name, v_weight]

    # Create concatenation operations for biases using 2 steps (Q+K, then +V)
    # Step 1: Concatenate Q and K biases
    qk_bias_concat_op = CatOp()
    qk_bias_concat_op.name = f"{fused_op.name}_qk_bias_concat"
    qk_bias_concat_op._arguments = [[q_bias, k_bias], 0]  # concat along dim 0
    qk_bias_concat_op.tensor_meta = {
        "shape": [fused_op.q_dim + fused_op.k_dim],
        "dtype": original_dtype,
    }
    qk_bias_concat_op._parents = [q_bias, k_bias]

    # Step 2: Concatenate QK result with V bias
    bias_concat_op = CatOp()
    bias_concat_op.name = f"{fused_op.name}_bias_concat"
    bias_concat_op._arguments = [
        [qk_bias_concat_op.name, v_bias],
        0,
    ]  # concat along dim 0
    bias_concat_op.tensor_meta = {
        "shape": [fused_op.q_dim + fused_op.k_dim + fused_op.v_dim],
        "dtype": original_dtype,
    }
    bias_concat_op._parents = [qk_bias_concat_op.name, v_bias]

    # Find the earliest position where we can insert concat operations
    # This should be after all referenced PlaceholderOp and weight nodes
    min_insert_index = 0

    # Check PlaceholderOp positions - they should come first in the graph
    for bias_name in bias_names:
        if bias_name in graph.node_table:
            bias_node = graph.node_table[bias_name]
            if (
                hasattr(bias_node, "__class__")
                and bias_node.__class__.__name__ == "PlaceholderOp"
            ):
                if bias_node in graph.body:
                    placeholder_index = graph.body.index(bias_node)
                    min_insert_index = max(
                        min_insert_index, placeholder_index + 1
                    )

    # Check weight node positions
    weight_nodes = [q_weight, k_weight, v_weight]
    for weight_name in weight_nodes:
        if weight_name in graph.node_table:
            weight_node = graph.node_table[weight_name]
            if weight_node in graph.body:
                weight_index = graph.body.index(weight_node)
                min_insert_index = max(min_insert_index, weight_index + 1)

    # Insert concat operations at the safe position (4 operations total)
    if min_insert_index > 0:
        graph.body.insert(min_insert_index, qk_weight_concat_op)
        graph.body.insert(min_insert_index + 1, weight_concat_op)
        graph.body.insert(min_insert_index + 2, qk_bias_concat_op)
        graph.body.insert(min_insert_index + 3, bias_concat_op)
    else:
        # Fallback: insert before Q node, which is shouldn't happen
        q_node_index = graph.body.index(q_node)
        graph.body.insert(q_node_index, qk_weight_concat_op)
        graph.body.insert(q_node_index + 1, weight_concat_op)
        graph.body.insert(q_node_index + 2, qk_bias_concat_op)
        graph.body.insert(q_node_index + 3, bias_concat_op)

    graph.node_table[qk_weight_concat_op.name] = qk_weight_concat_op
    graph.node_table[weight_concat_op.name] = weight_concat_op
    graph.node_table[qk_bias_concat_op.name] = qk_bias_concat_op
    graph.node_table[bias_concat_op.name] = bias_concat_op

    # Set up fused operation arguments using concat operations
    fused_op._arguments = [
        bias_concat_op.name,  # Combined bias [2048]
        shared_input,  # Shared input [40, 1536]
        weight_concat_op.name,  # Combined weight [1536, 2048]
    ]

    # Set tensor metadata for the fused output
    # Use shared_input shape instead of q_node shape to determine output shape
    combined_dim = fused_op.q_dim + fused_op.k_dim + fused_op.v_dim

    # Get shared input shape from the graph
    shared_input_node = graph.node_table.get(shared_input)
    if (
        shared_input_node
        and hasattr(shared_input_node, "tensor_meta")
        and shared_input_node.tensor_meta
    ):
        input_shape = shared_input_node.tensor_meta.get("shape")
    else:
        # Fallback to q_shape
        input_shape = q_shape

    if input_shape and len(input_shape) == 2:
        # 2D input: [seq_len, hidden_dim] -> [seq_len, combined_dim]
        output_shape = [input_shape[0], combined_dim]
    elif input_shape and len(input_shape) == 3:
        # 3D input: [batch, seq_len, hidden_dim] -> [batch, seq_len, combined_dim]
        output_shape = [input_shape[0], input_shape[1], combined_dim]
    else:
        # default to 2D with batch size 40
        batch_size = (
            input_shape[0] if input_shape and len(input_shape) > 0 else 40
        )
        output_shape = [batch_size, combined_dim]

    # print(f"  QKVFused output shape: {output_shape} (input_shape: {input_shape})")

    fused_op.tensor_meta = {"shape": output_shape, "dtype": original_dtype}

    # Create slice operations to extract Q, K, V from the fused output
    # Determine slice dimension and shapes based on output shape
    if len(output_shape) == 2:
        # 2D: [seq_len, combined_dim] -> [seq_len, q_dim/k_dim/v_dim]
        q_slice_shape = [output_shape[0], fused_op.q_dim]
        k_slice_shape = [output_shape[0], fused_op.k_dim]
        v_slice_shape = [output_shape[0], fused_op.v_dim]
        slice_dim = 1  # slice in the first dimension
    else:
        # 3D: [batch, seq_len, combined_dim] -> [batch, seq_len, q_dim/k_dim/v_dim]
        q_slice_shape = [output_shape[0], output_shape[1], fused_op.q_dim]
        k_slice_shape = [output_shape[0], output_shape[1], fused_op.k_dim]
        v_slice_shape = [output_shape[0], output_shape[1], fused_op.v_dim]
        slice_dim = 2  # slice in the 2nd dimension

    q_slice_op = SliceOp()
    q_slice_op.name = f"{fused_op.name}_q_slice"
    q_slice_op._arguments = [
        fused_op.name,  # input tensor
        slice_dim,  # dimension to slice
        0,  # start index
        fused_op.q_dim,  # end index, Q dimension
    ]
    q_slice_op.tensor_meta = {"shape": q_slice_shape, "dtype": original_dtype}

    k_slice_op = SliceOp()
    k_slice_op.name = f"{fused_op.name}_k_slice"
    k_slice_op._arguments = [
        fused_op.name,  # input tensor
        slice_dim,  # dimension to slice
        fused_op.q_dim,  # start index (Q)
        fused_op.q_dim + fused_op.k_dim,  # end index (Q+K)
    ]
    k_slice_op.tensor_meta = {"shape": k_slice_shape, "dtype": original_dtype}

    v_slice_op = SliceOp()
    v_slice_op.name = f"{fused_op.name}_v_slice"
    v_slice_op._arguments = [
        fused_op.name,  # input tensor
        slice_dim,  # dimension to slice
        fused_op.q_dim + fused_op.k_dim,  # start index (Q+K)
        fused_op.q_dim + fused_op.k_dim + fused_op.v_dim,  # end index (Q+K+V)
    ]
    v_slice_op.tensor_meta = {"shape": v_slice_shape, "dtype": original_dtype}

    # Validate slice operations
    for slice_op in [q_slice_op, k_slice_op, v_slice_op]:
        if len(slice_op._arguments) != 4:
            raise ValueError(
                f"SliceOp {slice_op.name} should have 4 arguments, got {len(slice_op._arguments)}"
            )

        start_idx = slice_op._arguments[2]
        end_idx = slice_op._arguments[3]

        if start_idx < 0 or end_idx <= start_idx:
            raise ValueError(
                f"SliceOp {slice_op.name} has invalid slice range [{start_idx}, {end_idx}]"
            )

        if end_idx > combined_dim:
            raise ValueError(
                f"SliceOp {slice_op.name} end index {end_idx} exceeds combined dimension {combined_dim}"
            )

    # Ensure tensor_meta and arguments are properly set for QKVFused operation
    fused_op.tensor_meta = {"shape": output_shape, "dtype": original_dtype}
    fused_op._arguments = [
        bias_concat_op.name,  # Combined bias [2048]
        shared_input,  # Shared input [40, 1536]
        weight_concat_op.name,  # Combined weight [1536, 2048]
    ]

    # Add fused operation to node_table
    graph.node_table[fused_op.name] = fused_op

    # Correct insertion logic: QKVFused should come BEFORE slice operations
    # Find the correct insertion point: after the last concat operation
    bias_concat_index = graph.body.index(bias_concat_op)

    # Insert QKVFused operation right after bias_concat
    qkv_fused_insert_index = bias_concat_index + 1
    graph.body.insert(qkv_fused_insert_index, fused_op)

    # Insert slice operations right after QKVFused operation
    slice_insert_index = qkv_fused_insert_index + 1

    # print(f"QKV Fusion Debug:")
    # print(f"  bias_concat at index: {bias_concat_index}")
    # print(f"  Inserting QKVFused at index: {qkv_fused_insert_index}")
    # print(f"  Inserting slices starting at index: {slice_insert_index}")

    # Set up slice operations with proper attributes and relationships
    for slice_op in [q_slice_op, k_slice_op, v_slice_op]:
        slice_op._name = slice_op.name
        slice_op._op_type = OpType.ReshapeType
        slice_op._parents = [fused_op.name]
        fused_op.add_children(slice_op.name)

    # Add slice operations to node table
    graph.node_table[q_slice_op.name] = q_slice_op
    graph.node_table[k_slice_op.name] = k_slice_op
    graph.node_table[v_slice_op.name] = v_slice_op

    # Insert slice operations in the correct order after QKVFused
    graph.body.insert(slice_insert_index, q_slice_op)
    graph.body.insert(slice_insert_index + 1, k_slice_op)
    graph.body.insert(slice_insert_index + 2, v_slice_op)

    # print(f"  Inserted Q slice at index: {slice_insert_index}")
    # print(f"  Inserted K slice at index: {slice_insert_index + 1}")
    # print(f"  Inserted V slice at index: {slice_insert_index + 2}")

    # Update references to point to slice operations BEFORE removing original nodes
    _update_node_references(graph, q_node.name, q_slice_op.name)
    _update_node_references(graph, k_node.name, k_slice_op.name)
    _update_node_references(graph, v_node.name, v_slice_op.name)

    # Find all operations that depend on slice operations (recursively)
    # We need to move not just direct dependents, but also indirect dependents
    # BUT only those that are currently BEFORE the last slice operation
    dependent_operations = []
    slice_names = [q_slice_op.name, k_slice_op.name, v_slice_op.name]

    # Find the position of the last slice operation
    last_slice_index = graph.body.index(v_slice_op)

    # Build a set of all nodes that transitively depend on slices
    # but only consider nodes that are BEFORE the last slice
    dependent_node_names = set(slice_names)
    changed = True
    while changed:
        changed = False
        for i, node in enumerate(graph.body):
            # Skip nodes that are already after the last slice
            if i > last_slice_index:
                continue
            if node.name in dependent_node_names:
                continue
            if hasattr(node, "args") and node.args:
                for arg in node.args:
                    if isinstance(arg, str) and arg in dependent_node_names:
                        dependent_node_names.add(node.name)
                        changed = True
                        break

    # Collect nodes that need to be moved (excluding the slices themselves)
    # Only move nodes that are BEFORE the last slice
    for i, node in enumerate(graph.body):
        if i > last_slice_index:
            break
        if node.name in dependent_node_names and node.name not in slice_names:
            dependent_operations.append((i, node))

    # Sort dependent operations by their current index (descending order for safe removal)
    dependent_operations.sort(key=lambda x: x[0], reverse=True)

    # Remove dependent operations from their current positions
    removed_operations = []
    for idx, node in dependent_operations:
        if node in graph.body:
            graph.body.remove(node)
            removed_operations.append(node)
            # print(f"  Moved dependent operation: {node.name} from index {idx}")

    # Topologically sort the removed operations to maintain dependencies
    # Build dependency graph for removed operations
    removed_names = {node.name for node in removed_operations}
    sorted_operations = []
    remaining = list(removed_operations)

    while remaining:
        # Find nodes with no dependencies in the remaining set
        ready = []
        for node in remaining:
            has_dep_in_remaining = False
            if hasattr(node, "args") and node.args:
                for arg in node.args:
                    if isinstance(arg, str) and arg in removed_names:
                        # Check if this dependency is still in remaining
                        if any(n.name == arg for n in remaining):
                            has_dep_in_remaining = True
                            break
            if not has_dep_in_remaining:
                ready.append(node)

        if not ready:
            # Circular dependency or error - just append remaining in order
            sorted_operations.extend(remaining)
            break

        # Add ready nodes to sorted list and remove from remaining
        sorted_operations.extend(ready)
        for node in ready:
            remaining.remove(node)

    # Insert sorted operations after the last slice operation
    last_slice_index = graph.body.index(v_slice_op)
    for i, node in enumerate(sorted_operations):
        insert_pos = last_slice_index + 1 + i
        graph.body.insert(insert_pos, node)
        # print(f"  Inserted {node.name} at index {insert_pos}")

    # Set up children relationships for slice operations
    for node in graph.body:
        if hasattr(node, "args") and node.args:
            for arg in node.args:
                if isinstance(arg, str):
                    if arg == q_slice_op.name:
                        q_slice_op.add_children(node.name)
                    elif arg == k_slice_op.name:
                        k_slice_op.add_children(node.name)
                    elif arg == v_slice_op.name:
                        v_slice_op.add_children(node.name)

    # Set up parent-child relationships for concat operations
    # QK weight concat depends on Q and K weights
    q_weight_node = graph.node_table.get(q_weight)
    k_weight_node = graph.node_table.get(k_weight)
    v_weight_node = graph.node_table.get(v_weight)
    q_bias_node = graph.node_table.get(q_bias)
    k_bias_node = graph.node_table.get(k_bias)
    v_bias_node = graph.node_table.get(v_bias)

    if q_weight_node:
        q_weight_node.add_children(qk_weight_concat_op.name)
    if k_weight_node:
        k_weight_node.add_children(qk_weight_concat_op.name)
    if q_bias_node:
        q_bias_node.add_children(qk_bias_concat_op.name)
    if k_bias_node:
        k_bias_node.add_children(qk_bias_concat_op.name)

    # Final weight concat depends on QK concat and V weight
    qk_weight_concat_op.add_children(weight_concat_op.name)
    if v_weight_node:
        v_weight_node.add_children(weight_concat_op.name)

    # Final bias concat depends on QK concat and V bias
    qk_bias_concat_op.add_children(bias_concat_op.name)
    if v_bias_node:
        v_bias_node.add_children(bias_concat_op.name)

    # Set up parent-child relationships for fused operation
    shared_input_node = graph.node_table.get(shared_input)
    if shared_input_node:
        fused_op._parents = [
            shared_input,
            bias_concat_op.name,
            weight_concat_op.name,
        ]
        shared_input_node.add_children(fused_op.name)
        bias_concat_op.add_children(fused_op.name)
        weight_concat_op.add_children(fused_op.name)

        # Set up slice operation relationships
        q_slice_op._parents = [fused_op.name]
        k_slice_op._parents = [fused_op.name]
        v_slice_op._parents = [fused_op.name]
        fused_op.add_children(q_slice_op.name)
        fused_op.add_children(k_slice_op.name)
        fused_op.add_children(v_slice_op.name)

    # Remove original Q, K, V nodes from the graph AFTER updating references
    if q_node in graph.body:
        graph.body.remove(q_node)
    if q_node.name in graph.node_table:
        del graph.node_table[q_node.name]

    if k_node in graph.body:
        graph.body.remove(k_node)
    if k_node.name in graph.node_table:
        del graph.node_table[k_node.name]

    if v_node in graph.body:
        graph.body.remove(v_node)
    if v_node.name in graph.node_table:
        del graph.node_table[v_node.name]

    return [q_slice_op, k_slice_op, v_slice_op]


def _update_node_references(graph: Graph, old_name: str, new_name: str):
    """Update all references to old_name with new_name in the graph"""
    for node in graph.body:
        # Update arguments list
        if hasattr(node, "_arguments") and node._arguments:
            for i, arg in enumerate(node._arguments):
                if isinstance(arg, str) and arg == old_name:
                    node._arguments[i] = new_name
                elif isinstance(arg, list):
                    # Handle nested lists in arguments
                    for j, sub_arg in enumerate(arg):
                        if isinstance(sub_arg, str) and sub_arg == old_name:
                            arg[j] = new_name

        # Update args attribute (for compatibility)
        if hasattr(node, "args") and node.args:
            for i, arg in enumerate(node.args):
                if isinstance(arg, str) and arg == old_name:
                    node.args[i] = new_name
                elif isinstance(arg, list):
                    for j, sub_arg in enumerate(arg):
                        if isinstance(sub_arg, str) and sub_arg == old_name:
                            arg[j] = new_name

        # Update parent relationships
        if hasattr(node, "_parents") and old_name in node._parents:
            node._parents = [
                new_name if p == old_name else p for p in node._parents
            ]


def apply_classic_fusion(graph: Graph):
    """
    Function to fuse some typical operations into one operation and fuse
    all operations into one graph.

    Args:
    - graph (Graph): The input graph to be simplified.

    Returns:
    - None: Modifies the input graph in place.
    """
    new_op_group = []
    device = DeviceType.CPU
    # Run the first round of op fusion
    classic_fuse_check(graph)
    # Run QKV fusion
    qkv_fuse_check(graph)
    for op in graph.body:
        if isinstance(op, PlaceholderOp):
            continue
        new_op_group.append(op)
    graph.op_groups = {}
    graph.op_groups["subgraph0"] = new_op_group
    graph.group_map_device = {"subgraph0": device}

    # Ensure proper dependencies between PlaceholderOp and Cat operations
    cat_referenced_placeholders = set()
    for op in graph.body:
        if isinstance(op, CatOp):
            # CatOp args format: [[tensor1, tensor2], axis]
            if (
                hasattr(op, "_arguments")
                and op._arguments
                and len(op._arguments) > 0
            ):
                tensor_list = op._arguments[0]
                if isinstance(tensor_list, list):
                    for tensor_name in tensor_list:
                        node = graph.node_table.get(tensor_name, None)
                        if node and isinstance(node, PlaceholderOp):
                            cat_referenced_placeholders.add(node)

    # Set up parent-child relationships for proper dependency tracking
    for placeholder_node in cat_referenced_placeholders:
        for op in graph.op_groups["subgraph0"]:
            if (
                isinstance(op, CatOp)
                and hasattr(op, "_arguments")
                and op._arguments
            ):
                tensor_list = op._arguments[0] if len(op._arguments) > 0 else []
                if (
                    isinstance(tensor_list, list)
                    and placeholder_node.name in tensor_list
                ):
                    if (
                        hasattr(op, "_parents")
                        and placeholder_node.name not in op._parents
                    ):
                        op.add_parent(placeholder_node.name)
                    if (
                        hasattr(placeholder_node, "_children")
                        and op.name not in placeholder_node._children
                    ):
                        placeholder_node.add_children(op.name)


def simply_fuse(graph: Graph):
    """
    Function to fuse all operations into one graph.

    Args:
    - graph (Graph): The input graph to be simplified.

    Returns:
    - None: Modifies the input graph in place.
    """
    new_op_group = []
    device = DeviceType.CPU
    for op in graph.body:
        if isinstance(op, PlaceholderOp):
            continue
        new_op_group.append(op)
    graph.op_groups = {}
    graph.op_groups["subgraph0"] = new_op_group
    graph.group_map_device = {"subgraph0": device}
