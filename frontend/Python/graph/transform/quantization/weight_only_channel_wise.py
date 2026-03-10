# ===- weight_only_channel_wise.py ---------------------------------------------
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
# Unified weight-only channel-wise quantization pass.
# Supports multiple quantization variants via configuration:
#   - int8 weights (w8a32, w8a16): direct dequant via MulOp
#   - int4 packed weights (w4a16): Int4UnpackOp -> MulOp dequant
#
# ===---------------------------------------------------------------------------

from .quantize import (
    QuantizationContext,
    QuantizationState,
    QuantizationConstraint,
    Quantization,
    register_quantizer,
    register_parameterized,
    register_dequantizer,
    OpType,
    Quantizable,
    Unquantized,
    Quantized,
    ToQuantize,
    Consumer,
    QuantizationMethod,
    requires_parents,
    AnyQuantizable,
    AnyQuantized,
    AnyUnquantized,
)
from ... import Graph, NodeType
from ...operation import *
from ...type import TensorDType

import torch
from typing import Any, Callable, Dict


# ---------------------------------------------------------------------------
# Shared constraint
# ---------------------------------------------------------------------------


class ChannelWiseQuantizationConstraint(QuantizationConstraint):
    axis: int

    def __init__(self, axis: int, gain: int = 1):
        self.axis = axis
        self.gain = gain

    def check(self, other: "ChannelWiseQuantizationConstraint") -> bool:
        return other.axis == self.axis

    def hash(self):
        return self.axis


Constraint = ChannelWiseQuantizationConstraint


# ---------------------------------------------------------------------------
# Quantization class identities (needed by the registration system)
# ---------------------------------------------------------------------------


class WeightOnlyQuantization(Quantization):
    """Int8 weight-only quantization (w8a32, w8a16)."""

    pass


class WeightOnlyInt4F16Quantization(Quantization):
    """Int4 packed weight quantization with f16 activation (w4a16)."""

    pass


# ---------------------------------------------------------------------------
# Per-variant configuration
# ---------------------------------------------------------------------------

_VARIANT_CONFIG = {
    WeightOnlyQuantization: {
        "int4_packed": False,
        "scaler_dtype": None,  # keep node's original dtype
    },
    WeightOnlyInt4F16Quantization: {
        "int4_packed": True,
        "scaler_dtype": TensorDType.Float16,
    },
}


# ---------------------------------------------------------------------------
# Method registration factory
# ---------------------------------------------------------------------------


def _register_methods(quantization_cls):
    """Register all quantization methods for *quantization_cls* using the
    variant configuration stored in _VARIANT_CONFIG."""

    cfg = _VARIANT_CONFIG[quantization_cls]
    int4_packed = cfg["int4_packed"]
    scaler_dtype_override = cfg["scaler_dtype"]

    def _quantizer(op_type):
        return register_quantizer(
            quantization=quantization_cls,
            op_type=op_type,
        )

    def _parameterized(params):
        return register_parameterized(
            quantization=quantization_cls,
            params=params,
        )

    # --- Transparent unary ops (Permute, Transpose, …) -------------------

    @_parameterized(
        [
            (PermuteOp, {"torch_op": torch.permute}),
            (TransposeOp, {"torch_op": torch.transpose}),
            (UnsqueezeOp, {"torch_op": torch.unsqueeze}),
            (SqueezeOp, {"torch_op": torch.squeeze}),
            (CloneOp, {"torch_op": torch.clone}),
        ]
    )
    @requires_parents(AnyQuantizable)
    class TransparentUnaryQuantizationMethod(QuantizationMethod):
        torch_op: Callable[[Any], torch.Tensor] = lambda: None
        buddy_op: type = None

        def rewriter(self, node, context):
            assert isinstance(
                (state := context.quantization_table[node.name]), ToQuantize
            )
            parent_name = node._parents[0]
            args = node.args[1:]
            parent_scaler_name = "scaler_" + parent_name
            parent_scaler_op = context.graph.node_table[parent_scaler_name]
            parent_scaler_shape = parent_scaler_op.tensor_meta["shape"]
            parent_node = context.graph.node_table[parent_name]
            node.tensor_meta["dtype"] = parent_node.tensor_meta["dtype"]
            op_scaler_name = "scaler_" + node._name
            scaler: Op = self.buddy_op()
            scaler._name = op_scaler_name
            scaler._tensor_meta["dtype"] = parent_scaler_op.tensor_meta["dtype"]
            probe = torch.empty(parent_scaler_shape)
            result = self.torch_op(probe, *args)
            scaler._tensor_meta["shape"] = result.shape
            scaler._arguments = [parent_scaler_name, *args]
            scaler._parents = [parent_scaler_name]
            parent_scaler_op._children.append(op_scaler_name)
            context.graph.add_node(scaler)
            context.quantization_table[node.name] = Quantized(
                constraint=state.constraint
            )

        def callback(self, constraint, node, context):
            assert isinstance(constraint, Constraint)
            state = context.quantization_table[node.name]
            assert isinstance(state, Quantizable)
            probe_shape = list(range(len(node.tensor_meta["shape"])))
            probe = torch.empty(probe_shape)
            args = node.args[1:]
            result = self.torch_op(probe, *args)
            new_constraint = Constraint(
                axis=result.shape[constraint.axis], gain=constraint.gain
            )
            parent_name = node._parents[0]
            if context.quantization_table[parent_name].propagate_constraint(
                new_constraint
            ):
                return constraint

        def _forward(self, node, context):
            parent_name = node._parents[0]
            state = context.quantization_table[parent_name]
            if not isinstance(state, Quantized):
                return Unquantized()
            axis = state.constraint.axis
            fake_shape = [
                1 if i == axis else 0
                for i in range(len(node.tensor_meta["shape"]))
            ]
            probe = torch.empty(fake_shape)
            args = node.args[1:]
            result = self.torch_op(probe, *args)
            return Quantizable([Constraint(axis=result.shape.index(1), gain=0)])

    # --- View / Reshape --------------------------------------------------

    @_parameterized(
        [
            (ViewOp, {"torch_op": torch.Tensor.view}),
            (ReshapeOp, {"torch_op": torch.Tensor.reshape}),
        ]
    )
    @requires_parents(AnyQuantizable)
    class ViewReshapeQuantizationMethod(QuantizationMethod):
        torch_op: Callable[[Any], torch.Tensor] = lambda: None
        buddy_op: type = None

        @classmethod
        def compute_strides(cls, shape: list[int]) -> list[int]:
            strides = []
            stride = 1
            for axis_size in reversed(shape):
                strides.append(stride)
                stride *= axis_size
            strides.reverse()
            return strides

        @classmethod
        def get_new_quantization_axis_or_none(
            cls, old_shape, new_shape, axis
        ) -> int | None:
            old_strides = cls.compute_strides(old_shape)
            quant_stride = old_strides[axis]
            new_strides = cls.compute_strides(new_shape)
            for ax, stride in enumerate(new_strides):
                if quant_stride == stride:
                    return ax
            return None

        def rewriter(self, node, context):
            parent_name = node._parents[0]
            args = node.args[1:]
            parent_scaler_name = "scaler_" + parent_name
            parent_scaler_op = context.graph.node_table[parent_scaler_name]
            parent_scaler_shape = parent_scaler_op.tensor_meta["shape"]
            op_scaler_name = "scaler_" + node._name
            scaler: Op = self.buddy_op()
            scaler._name = op_scaler_name
            scaler._tensor_meta["dtype"] = parent_scaler_op.tensor_meta["dtype"]
            probe = torch.empty(parent_scaler_shape)
            result = self.torch_op(probe, *args)
            scaler._tensor_meta["shape"] = result.shape
            scaler._arguments = [parent_scaler_name, args]
            scaler._parents = [parent_scaler_name]
            parent_scaler_op._children.append(op_scaler_name)
            context.graph.add_node(scaler)

        def callback(self, constraint, node, context):
            assert isinstance(constraint, Constraint)
            state = context.quantization_table[node.name]
            assert isinstance(state, Quantizable)
            probe_shape = list(range(len(node.tensor_meta["shape"])))
            probe = torch.empty(probe_shape)
            args = node.args[1:]
            result = self.torch_op(probe, *args)
            constraint = Constraint(axis=result.shape[constraint.axis])
            parent_name = node._parents[0]
            if context.quantization_table[parent_name].propagate_constraint(
                constraint
            ):
                return constraint

        def _forward(self, node, context):
            parent_name = node._parents[0]
            state = context.quantization_table[parent_name]
            if not isinstance(state, Quantized):
                return Unquantized()
            parent_axis = state.constraint.axis
            parent_node = context.graph.node_table[parent_name]
            old_shape = parent_node.tensor_meta["shape"]
            new_shape = node.tensor_meta["shape"]
            axis = self.get_new_quantization_axis_or_none(
                old_shape, new_shape, parent_axis
            )
            if axis is None:
                return Unquantized()
            return Quantizable(constraints=[Constraint(axis=axis)])

    # --- Dequantizer -----------------------------------------------------

    if int4_packed:

        @register_dequantizer(quantization_method=quantization_cls)
        def _dequantizer(node: Op, context: QuantizationContext) -> Op:
            """Int4-packed dequant: Int4UnpackOp -> Mul(unpacked_i8, scale)."""
            node_name = node.name
            dequantized_name = "dequantized_" + node_name
            try:
                return context.graph.node_table[dequantized_name]
            except KeyError:
                pass

            scaler_name = "scaler_" + node_name
            assert (
                scaler_name in context.graph.node_table
            ), f"Missing scaler for {node_name}."
            scaler_node = context.graph.node_table[scaler_name]

            packed_shape = list(node.tensor_meta["shape"])
            unpacked_shape = list(packed_shape)
            unpacked_shape[-1] = unpacked_shape[-1] * 2

            unpack_name = "unpack_" + node_name
            unpack_op = Int4UnpackOp()
            unpack_op._name = unpack_name
            unpack_op._arguments = [node_name]
            unpack_op._parents = [node_name]
            unpack_op._tensor_meta["shape"] = tuple(unpacked_shape)
            unpack_op._tensor_meta["dtype"] = node.tensor_meta["dtype"]
            node._children.append(unpack_name)
            context.graph.add_node(unpack_op)

            mul_op = MulOp()
            mul_op._name = dequantized_name
            mul_op._arguments = [unpack_name, scaler_name]
            mul_op._parents = [unpack_name, scaler_name]
            mul_op._tensor_meta["shape"] = tuple(unpacked_shape)
            mul_op._tensor_meta["dtype"] = scaler_node.tensor_meta["dtype"]
            unpack_op._children.append(dequantized_name)
            scaler_node._children.append(dequantized_name)
            context.graph.add_node(mul_op)
            return mul_op

    else:

        @register_dequantizer(quantization_method=quantization_cls)
        def _dequantizer(node: Op, context: QuantizationContext) -> MulOp:
            """Standard dequant: Mul(weight_i8, scale)."""
            node_name = node.name
            dequantized_name = "dequantized_" + node_name
            try:
                return context.graph.node_table[dequantized_name]
            except KeyError:
                pass

            scaler_name = "scaler_" + node_name
            assert (
                scaler_name in context.graph.node_table
            ), f"Missing scaler for {node_name}."
            scaler_node = context.graph.node_table[scaler_name]

            mul_op = MulOp()
            mul_op._name = dequantized_name
            mul_op._arguments = [node_name, scaler_name]
            mul_op._parents = [node_name, scaler_name]
            mul_op._tensor_meta["shape"] = node.tensor_meta["shape"]
            mul_op._tensor_meta["dtype"] = scaler_node.tensor_meta["dtype"]
            node._children.append(dequantized_name)
            scaler_node._children.append(dequantized_name)
            context.graph.add_node(mul_op)
            return mul_op

    # --- Placeholder (root params) ---------------------------------------

    @_quantizer(op_type=PlaceholderOp)
    class PlaceholderQuantizationMethod(QuantizationMethod):

        def _check_eligibility(self, node: Op, context: QuantizationContext):
            input_names = [inp.name for inp in context.graph.inputs]
            if node.name in input_names:
                return Unquantized()
            param_names = [param.name for param in context.graph.params]
            if node.name in param_names:
                return Quantizable()
            return Unquantized()

        def rewriter(self, node: Op, context: QuantizationContext):
            assert isinstance(
                (state := context.quantization_table[node.name]), ToQuantize
            )
            node_name = node.name
            node_shape = list(node.tensor_meta["shape"])
            quantization_state: Quantizable = context.quantization_table[
                node_name
            ]
            quantization_axis = quantization_state.constraint.axis

            scaler_node = PlaceholderOp()
            scaler_node._name = "scaler_" + node_name
            scaler_node._tensor_meta["shape"] = [
                1 if i != quantization_axis else axis_dim
                for i, axis_dim in enumerate(node_shape)
            ]
            scaler_node._tensor_meta["dtype"] = (
                scaler_dtype_override
                if scaler_dtype_override is not None
                else node._tensor_meta["dtype"]
            )

            node._tensor_meta["dtype"] = context.target_dtype

            if int4_packed:
                packed_shape = list(node_shape)
                packed_shape[-1] = packed_shape[-1] // 2
                node._tensor_meta["shape"] = tuple(packed_shape)

            context.graph.add_node(scaler_node, node_type=NodeType.FakeNode)
            context.quantization_table[node.name] = Quantized(
                constraint=state.constraint
            )

        def callback(self, constraint, node, context):
            return constraint

        def _forward(self, op: Op, context: QuantizationContext):
            return Quantizable(
                constraints=[
                    Constraint(axis=i, gain=0)
                    for i in range(len(op.tensor_meta["shape"]))
                ]
            )

    # --- Matmul -----------------------------------------------------------

    @_quantizer(op_type=MatmulOp)
    class MatmulQuantizeMethod(QuantizationMethod):

        def _check_eligibility(self, node, context):
            parent_op1_name, parent_op2_name = node._parents
            op1_axis = max(
                0,
                len(
                    context.graph.node_table[parent_op1_name].tensor_meta[
                        "shape"
                    ]
                )
                - 2,
            )
            context.quantization_table[node._parents[0]].propagate_constraint(
                Constraint(axis=op1_axis)
            )
            op2_axis = max(
                0,
                len(
                    context.graph.node_table[parent_op2_name].tensor_meta[
                        "shape"
                    ]
                )
                - 1,
            )
            context.quantization_table[node._parents[1]].propagate_constraint(
                Constraint(axis=op2_axis)
            )
            return Unquantized()

        def rewriter(self, node: Op, context: QuantizationContext):
            pass

        def _forward(self, node: Op, context: QuantizationContext):
            return Unquantized()


# ---------------------------------------------------------------------------
# Execute registrations
# ---------------------------------------------------------------------------

_register_methods(WeightOnlyQuantization)
_register_methods(WeightOnlyInt4F16Quantization)
