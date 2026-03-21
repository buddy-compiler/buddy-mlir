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
    MatmulOp,
    MulOp,
    QuantizedAddMMOp,
    QuantizedMatmulOp,
)
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _convert_to_quantized_matmul(graph):
    """Replace CastOp->MulOp(dequant)->AddMMOp/MatmulOp patterns with
    QuantizedAddMMOp/QuantizedMatmulOp that perform native i8 matmul."""

    dequant_muls = {}
    for node in graph._body:
        if isinstance(node, MulOp) and node.name.startswith("dequantized_"):
            weight_name = node.name[len("dequantized_") :]
            scaler_name = "scaler_" + weight_name
            if scaler_name in graph.node_table:
                dequant_muls[node.name] = (weight_name, scaler_name)

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

        if op_type == "addmm":
            new_op = QuantizedAddMMOp()
            new_op._name = matmul_node.name
            bias_name = matmul_node.args[0]
            act_name = matmul_node.args[1]
            new_op._arguments = [bias_name, act_name, weight_name, scaler_name]
            new_op._parents = [bias_name, act_name, weight_name, scaler_name]
        else:
            new_op = QuantizedMatmulOp()
            new_op._name = matmul_node.name
            act_name = matmul_node.args[0]
            new_op._arguments = [act_name, weight_name, scaler_name]
            new_op._parents = [act_name, weight_name, scaler_name]

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
