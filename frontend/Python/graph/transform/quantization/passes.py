# ===- passes.py ---------------------------------------------------------------
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
# High-level quantization pass entry points that compose the low-level
# quantize_graph / sort_graph primitives with specific configurations.
#
# ===---------------------------------------------------------------------------

from .quantize import quantise_graph, sort_graph
from .weight_only_channel_wise import (
    WeightOnlyQuantization,
    WeightOnlyInt4F16Quantization,
)
from ...operation import (
    AddMMOp,
    EmbeddingOp,
    MatmulOp,
    MulOp,
    PlaceholderOp,
    QuantizedAddMMOp,
    QuantizedGroupAddMMOp,
    QuantizedGroupMatmulOp,
    QuantizedGroupEmbeddingOp,
    QuantizedMatmulOp,
)
from ... import NodeType
from ...type import TensorDType


def weight_only_channel_wise(graph):
    """Int8 weight-only quantization (w8a32 / w8a16)."""
    quantise_graph(
        graph=graph,
        quantization=WeightOnlyQuantization(),
    )


def weight_only_int4_f16_channel_wise(graph):
    """Int4 packed weight quantization with f16 activation (w4a16)."""
    quantise_graph(
        graph=graph,
        quantization=WeightOnlyInt4F16Quantization(),
        target_dtype=TensorDType.Int8,
    )


def w8a8_channel_wise(graph):
    """W8A8 quantization: weight-only quantization followed by converting
    dequant+matmul pairs to native int8 matmul with dynamic activation
    quantization."""
    quantise_graph(
        graph=graph,
        quantization=WeightOnlyQuantization(),
    )
    _convert_to_quantized_matmul(graph)
    sort_graph(graph)


def w8a8_per_group(graph, group_sizes_by_k):
    """Convert linear weights to native per-group W8A8 matmuls.

    ``group_sizes_by_k`` maps the linear input width K to its quantization
    group size.  This intentionally keeps policy out of the generic Buddy
    pass: model importers choose the mixed group sizes required by their
    checkpoint format.
    """
    quantise_graph(
        graph=graph,
        quantization=WeightOnlyQuantization(),
    )
    _convert_to_quantized_matmul(graph, group_sizes_by_k)
    sort_graph(graph)


def embedding_w8_per_group(graph, group_size):
    """Rewrite embedding lookup to int8 per-group dequantization."""
    group_size = int(group_size)
    if group_size <= 0:
        raise ValueError("embedding group size must be positive")

    replacements = []
    for node in graph._body:
        if not isinstance(node, EmbeddingOp):
            continue
        weight_name = str(node.args[0])
        token_name = str(node.args[1])
        weight_node = graph.node_table.get(weight_name)
        if weight_node is None or weight_node not in graph.params:
            raise ValueError(
                f"embedding weight {weight_name} must be a graph parameter"
            )
        weight_shape = list(
            weight_node.tensor_meta.get(
                "w8a8_logical_shape", weight_node.tensor_meta["shape"]
            )
        )
        if len(weight_shape) != 2 or weight_shape[1] % group_size != 0:
            raise ValueError(
                f"embedding weight shape {weight_shape} is incompatible "
                f"with group size {group_size}"
            )
        replacements.append(
            (node, weight_node, token_name, weight_shape)
        )

    for node, weight_node, token_name, weight_shape in replacements:
        scaler_name = "scaler_" + weight_node.name
        if scaler_name in graph.node_table:
            raise ValueError(
                f"embedding weight {weight_node.name} already has a scaler; "
                "untie embedding and LM-head weights before W8A8 rewriting"
            )
        scaler = PlaceholderOp()
        scaler._name = scaler_name
        scaler._tensor_meta["shape"] = (
            int(weight_shape[0]),
            int(weight_shape[1]) // group_size,
        )
        scaler._tensor_meta["dtype"] = weight_node.tensor_meta["dtype"]
        graph.add_node(scaler, node_type=NodeType.FakeNode)
        weight_node._tensor_meta["dtype"] = TensorDType.Int8

        replacement = QuantizedGroupEmbeddingOp()
        replacement._name = node.name
        replacement._arguments = [
            weight_node.name,
            token_name,
            scaler_name,
            group_size,
        ]
        replacement._parents = [weight_node.name, token_name, scaler_name]
        replacement._children = node._children[:]
        replacement._tensor_meta = node._tensor_meta.copy()

        weight_node._children = [
            replacement.name if child == node.name else child
            for child in weight_node._children
        ]
        token_node = graph.node_table[token_name]
        token_node._children = [
            replacement.name if child == node.name else child
            for child in token_node._children
        ]
        scaler._children.append(replacement.name)

        index = graph._body.index(node)
        graph._body[index] = replacement
        graph.node_table[replacement.name] = replacement

    sort_graph(graph)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _convert_to_quantized_matmul(graph, group_sizes_by_k=None):
    """Replace CastOp->MulOp(dequant)->AddMMOp/MatmulOp patterns with
    QuantizedAddMMOp/QuantizedMatmulOp that perform native i8 matmul."""

    dequant_muls = {}
    for node in graph._body:
        if isinstance(node, MulOp) and node.name.startswith("dequantized_"):
            weight_name = node.name[len("dequantized_") :]
            scaler_name = "scaler_" + weight_name
            if scaler_name in graph.node_table:
                dequant_muls[node.name] = (weight_name, scaler_name)

    def set_group_scaler_source_shape(scaler_node, groups, outputs):
        """Update the fake parameter feeding a transformed weight scaler."""
        current = scaler_node
        visited = set()
        while current.name not in visited:
            visited.add(current.name)
            scaler_parents = [
                graph.node_table[parent]
                for parent in current.parents
                if parent in graph.node_table and parent.startswith("scaler_")
            ]
            if not scaler_parents:
                break
            current = scaler_parents[0]

        source_shape = list(current.tensor_meta["shape"])
        if source_shape in ([groups, outputs], [outputs, groups]):
            # A tied/shared parameter may feed more than one matmul.  Its
            # scaler source was already expanded by the first replacement.
            return
        if len(source_shape) != 2 or outputs not in source_shape:
            raise ValueError(
                f"unsupported W8A8 scaler source {current.name} shape "
                f"{source_shape} for N={outputs}"
            )
        reduced_axes = [
            axis for axis, size in enumerate(source_shape) if size == 1
        ]
        if len(reduced_axes) != 1:
            raise ValueError(
                f"cannot identify reduced axis in W8A8 scaler source "
                f"{current.name} shape {source_shape}"
            )
        source_shape[reduced_axes[0]] = groups
        current._tensor_meta["shape"] = tuple(source_shape)

    replacements = []
    for node in graph._body:
        if isinstance(node, (AddMMOp, MatmulOp)):
            if isinstance(node, AddMMOp):
                weight_arg = node.args[2]
                if weight_arg in dequant_muls:
                    replacements.append(
                        (node, "addmm", weight_arg, dequant_muls[weight_arg])
                    )
            elif isinstance(node, MatmulOp):
                weight_arg = node.args[1]
                if weight_arg in dequant_muls:
                    replacements.append(
                        (node, "matmul", weight_arg, dequant_muls[weight_arg])
                    )

    for (
        matmul_node,
        op_type,
        dequant_name,
        (weight_name, scaler_name),
    ) in replacements:
        weight_node = graph.node_table[weight_name]
        scaler_node = graph.node_table[scaler_name]

        weight_shape = list(weight_node.tensor_meta["shape"])
        if len(weight_shape) != 2:
            raise ValueError(
                f"W8A8 matmul weight {weight_name} must be rank 2, "
                f"got {weight_shape}"
            )
        input_width = int(weight_shape[0])
        group_size = None
        if group_sizes_by_k is not None:
            try:
                group_size = int(group_sizes_by_k[input_width])
            except KeyError as exc:
                raise ValueError(
                    f"no W8A8 group size configured for K={input_width} "
                    f"({weight_name})"
                ) from exc
            if group_size <= 0 or input_width % group_size != 0:
                raise ValueError(
                    f"invalid W8A8 group size {group_size} for "
                    f"K={input_width} ({weight_name})"
                )
            output_width = int(weight_shape[1])
            if input_width % 64 != 0:
                raise ValueError(
                    f"Qwen3 W8A8 K={input_width} must be divisible by 64 "
                    f"({weight_name})"
                )
            if output_width % 64 != 0:
                raise ValueError(
                    f"Qwen3 W8A8 D={output_width} must be divisible by 64 "
                    f"({weight_name})"
                )
            if group_size > 1024:
                raise ValueError(
                    f"Qwen3 W8A8 group size {group_size} exceeds 1024 "
                    f"({weight_name})"
                )
            scaler_node._tensor_meta["shape"] = (
                input_width // group_size,
                output_width,
            )
            set_group_scaler_source_shape(
                scaler_node,
                input_width // group_size,
                output_width,
            )
            # Preserve the logical PyTorch [K,D] shape for checkpoint packing,
            # while exposing the physical AME blob layout to MLIR function
            # arguments.  This metadata also makes repeated/shared uses
            # idempotent.
            weight_node._tensor_meta["w8a8_logical_shape"] = (
                input_width,
                output_width,
            )
            weight_node._tensor_meta["w8a8_layout"] = "ame_outblk64"
            weight_node._tensor_meta["shape"] = (
                output_width // 64,
                input_width // group_size,
                64,
                group_size,
            )

        if op_type == "addmm":
            new_op = (
                QuantizedGroupAddMMOp()
                if group_size is not None
                else QuantizedAddMMOp()
            )
            new_op._name = matmul_node.name
            bias_name = matmul_node.args[0]
            act_name = matmul_node.args[1]
            new_op._arguments = [bias_name, act_name, weight_name, scaler_name]
            new_op._parents = [bias_name, act_name, weight_name, scaler_name]
        else:
            new_op = (
                QuantizedGroupMatmulOp()
                if group_size is not None
                else QuantizedMatmulOp()
            )
            new_op._name = matmul_node.name
            act_name = matmul_node.args[0]
            new_op._arguments = [act_name, weight_name, scaler_name]
            new_op._parents = [act_name, weight_name, scaler_name]

        if group_size is not None:
            new_op._arguments.append(group_size)

        new_op._tensor_meta = matmul_node._tensor_meta.copy()
        new_op._children = matmul_node._children[:]

        weight_node._children = [
            c for c in weight_node._children if c != dequant_name
        ]
        if new_op.name not in weight_node._children:
            weight_node._children.append(new_op.name)

        scaler_node._children = [
            c for c in scaler_node._children if c != dequant_name
        ]
        if new_op.name not in scaler_node._children:
            scaler_node._children.append(new_op.name)

        dequant_node = graph.node_table.get(dequant_name)
        if dequant_node:
            dequant_node._children = [
                c for c in dequant_node._children if c != matmul_node.name
            ]

        idx = graph._body.index(matmul_node)
        graph._body[idx] = new_op
        graph.node_table[new_op.name] = new_op

    # Remove orphaned dequant MulOp nodes
    nodes_to_remove = set()
    for node in graph._body:
        if isinstance(node, MulOp) and node.name.startswith("dequantized_"):
            if len(node._children) == 0:
                nodes_to_remove.add(node.name)
                for p_name in node._parents:
                    p_node = graph.node_table.get(p_name)
                    if p_node:
                        p_node._children = [
                            c for c in p_node._children if c != node.name
                        ]

    param_objs = [graph._body[idx] for idx in graph._fake_params]
    input_objs = [graph._body[idx] for idx in graph._inputs]

    graph._body = [n for n in graph._body if n.name not in nodes_to_remove]
    for name in nodes_to_remove:
        graph.node_table.pop(name, None)

    body_index = {id(n): i for i, n in enumerate(graph._body)}
    graph._fake_params = [body_index[id(n)] for n in param_objs]
    graph._inputs = [body_index[id(n)] for n in input_objs]
