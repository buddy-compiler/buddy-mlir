from .. import Graph, NodeType
from ..operation import *
from .. import DeviceType
from torch.fx.immutable_collections import immutable_list
from ..type import TensorDType

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

import torch

class QuantizationState:
    pass

class Quantized(QuantizationState):
    axis: int

    def __init__(self, axis):
        self.axis = axis

class Quantizable(QuantizationState):
    axis: int = None
    callback: Callable | None = None

    def __init__(
            self,
            rewrite: Callable[[], None],
            axis: int | None = None,
            callback: Callable[[int], None] | None = None,
        ):
        if axis:
            self.axis = axis
            if callback:
                callback(self.axis)
        
        self.callback = callback
        self.rewrite = rewrite

    def set_axis(self, axis: int):
        assert self.axis is None, "Axis cannot be set twice on the same Quantized."

        self.axis = axis

        if self.callback:
            self.callback(self.axis)

    def check_axis(self, axis: int) -> bool:
        """
        Check if the quantization of the tensor is compatible
        with the given axis.

        When the quantization of the tensor is not set apriori,
        then it is supposed to be determined based on context,
        so if `axis is None`, then it is compatible.

        Args:
            axis (int): axis to determine compatibility with.

        Returns:
            bool: Whether the quantization is compatible.
        """

        if self.axis is None:
            self.set_axis(axis=axis)
            return True

        return self.axis == axis
    
class Unquantized(QuantizationState):
    pass

# TODO: this might need a better name.
class Consumer(Unquantized):
    rewriter: Callable[[], None] | None = None

    def __init__(self, rewrite: Callable[[], None]):
        self.rewrite = rewrite


class QuantizationMode(Enum):
    WOTensorWise = auto() # adds a single scaler per tensor
    WOChannelWise = auto() # adds a scaler each channel of the tensor.

class QuantizationContext:
    """
    Quantization context to store relevant information.

    graph (Graph): graph to quantize
    quantization_table (dict[str, QuantizationState]): table storing whether each
        node is quantized or not. Also acts as a way of determining if a node has been
        evaluated or not.
    quantization_queue (list[str]): List of nodes, which have not yet been evaluated, 
        but are ready for evaluation (i.e. all their parents have been evaluated)
    """
    graph: Graph
    quantization_table: dict[str, QuantizationState]
    quantization_mode: QuantizationMode
    target_dtype: TensorDType

    def __init__(
            self,
            graph: Graph,
            quantization_mode: QuantizationMode,
            target_dtype: TensorDType,
):
        self.graph = graph
        self.quantization_mode = quantization_mode
        self.target_dtype = target_dtype
        self.quantization_table = {}

# TODO: maybe this should be called "GraphWalker" or smtn
class Pass:

    def __init__(
            self,
            processor: Callable[[Op, QuantizationContext], None],
            context: QuantizationContext
    ):
        self.processor = processor
        self.context = context
        self.queue = []
        self.processed = []

    def check_ready_children(self, node: Op):
        for child_name in node._children:
            child_node = self.context.graph.node_table[child_name]

            for parent_name in child_node._parents:
                if parent_name not in self.processed:
                    break
            else:
                self.queue.append(child_name)

    def run_pass(self):
        while len(self.queue) > 0:
            node_name = self.queue.pop()
            node = self.context.graph.node_table[node_name]

            self.processor(node, self.context)
            self.processed.append(node_name)
            self.check_ready_children(node)


QuantizationFunctionRegistry: dict[type, dict[str, Callable[[Op, QuantizationContext], None | QuantizationState]]] = {}

# ----
# Quantization methods.
# ----

def mask_guard_factory(
        op_type: type,
        quantization_mask: list[type],
        #quant_state_on_success: QuantizationState
):
    """
    This factory creates a custom decorator for each node quantization function based on the op type
    they process, and the quantization mask they accept.

    It also adds the function the `QuantizationFunctionRegistry`, which automatically dispatches
    op quantizations to supported node types and quantitation masks.

    TODO: This should also allow declaring the QuantizationState of the result
    """
    def mask_guard(fn: Callable[[Op, QuantizationContext], None | QuantizationState]):
        def guarded_fn(op: Op, context: QuantizationContext) -> None | QuantizationState:
            assert isinstance(op, op_type)
            assert len(quantization_mask) == len(op._parents), "Op should have as many arguments as quantization mask entries."
            assert all([
                    isinstance(context.quantization_table[parent], quantization_state)
                    for (parent, quantization_state) in zip(op._parents, quantization_mask)
                ]), \
                "Op argument quantization should match quantization mask."
            return fn(op, context)

        quantizer_mask_str = ''.join(['q' if mask == Quantizable else 'u' for mask in quantization_mask])

        # Register function in `QuantizationFunctionRegistry`
        QuantizationFunctionRegistry.setdefault(op_type, {})[quantizer_mask_str] = guarded_fn

        # TODO: Maybe we should delete this function from user space by not returning
        return guarded_fn
    
    return mask_guard

# All quantization functions take their target op, and the QuantizationContext
# as their inputs, and return True or False based on the success of the operation.
# ** if False is returned, the method SHOULD NOT MODIFY the graph.
#
# These methods should only be called by `dispatch_op_quantization`, and thus their signature
# should be fixed to Callable[[Op, QuantizationContext], bool]. The enforcement of the input op
# type and the quantization mask happens in the `mask_guard_factory`.

@mask_guard_factory(op_type=ViewOp, quantization_mask=[
    Quantizable,
])
def _quantize_view_q(op: Op, context: QuantizationContext) -> None | QuantizationState:
    """
    Quantizing the view op of a quantized input.

    - In WOTensorWise mode, we reuse the scaler from the parent node.
    - In WOChannelWise in general there is no way to quantize the result. In can be done
        if the view is only eliminating axes, or other similar operations, but for now 
        we don't support it.
    """
    return None

def _rewrite_permute_q(node: Op, context: QuantizationContext) -> None:
    parent_name = node._parents[0]
    permute_dims: list[int] = node.args[1]

    parent_scaler_name = "scaler_" + parent_name
    parent_scaler_op = context.graph.node_table[parent_scaler_name]

    op_scaler_name = "scaler_" + node._name
    permute_scaler = PermuteOp()
    permute_scaler._name = op_scaler_name
    permute_scaler._tensor_meta["dtype"] = parent_scaler_op.tensor_meta["dtype"]
    permute_scaler._tensor_meta["shape"] = [parent_scaler_op._tensor_meta["shape"][i] for i in permute_dims]
    permute_scaler._arguments = [parent_scaler_name, permute_dims]
    permute_scaler._parents = [parent_scaler_name]
    parent_scaler_op._children.append(op_scaler_name)
    
    context.graph.add_node(permute_scaler)

@mask_guard_factory(op_type=PermuteOp, quantization_mask=[
    Quantizable
])
def _quantize_permute_q(node: Op, context: QuantizationContext) -> None | QuantizationState:
    """
    Quantizing the permute op of a quantized tensor.

    - In WOTensorWise mode, we reuse the scaler from the parent node.
    - In WOChannelWise mode, the result will have a new quantization axis.
    """
    
    parent_name = node._parents[0]
    permute_dims: list[int] = node.args[1]

    # If the previous node had its quantization axis set, we can define quantization by axis.
    if (axis := context.quantization_table[parent_name].axis):
        return Quantizable(axis=permute_dims.index(axis), rewrite=_rewrite_permute_q)
    # If not, we define quantization with callback, so that if an axis is determined, we can propagate that back
    return Quantizable(callback=lambda a: context.quantization_table[parent_name].set_axis(permute_dims[a]), rewrite=_rewrite_permute_q)

def _rewrite_matmul(node: Op, context: QuantizationContext) -> None:

    node_name = node.name
    op_chain = [node]

    for i, parent_name in enumerate(node._parents, 1):
        if isinstance(context.quantization_table[parent_name], Quantized):
            
            prev_node = op_chain[-1]

            parent_scaler_name = "scaler_" + parent_name

            # check that this actually exists.

            mul_op = MulOp()
            mul_op_name = f"scale{i}_" + node_name
            mul_op._name = mul_op_name
            mul_op._arguments  = mul_op._parents = ["mm", parent_scaler_name]
            mul_op.tensor_meta["shape"] = prev_node.tensor_meta["shape"]
            mul_op.tensor_meta["dtype"] = prev_node.tensor_meta["dtype"]
            if len(op_chain) > 1:
                prev_node._children.append(mul_op_name)

            context.graph.add_node(mul_op)
            op_chain.append(mul_op)

    context.graph.replace_as_parent(
        parent_op=node,
        child_ops=node._children,
        new_op=op_chain[-1],
    )

    node._children = [op_chain[1].name]

# TODO: this is actually `aa` but dispatch system does not support it yet.
@mask_guard_factory(op_type=MatmulOp, quantization_mask=[
    Unquantized, Quantizable
])
def _quantize_matmul_aa(node: Op, context: QuantizationContext) -> None | QuantizationState:
    """
        This method replaces the pattern
        (mm arg1:unquantized, arg2:quantized)
        with
        (mul (mm arg1, arg2), scaler_arg2)
    """

    quantizable = False

    if isinstance((state := context.quantization_table[node._parents[0]]), Quantizable):
        if state.check_axis(0):
            quantizable = True

    if isinstance((state := context.quantization_table[node._parents[1]]), Quantizable):
        if state.check_axis(1):
            quantizable = True

    if quantizable:
        return Consumer(rewrite=_rewrite_matmul)

    return Unquantized()

def _rewrite_addmm_quq(node: Op, context: QuantizationContext) -> None:
    node_name = node.name

    mm_op = MatmulOp()
    mm_op_name = "pre_scaled_" + node_name
    mm_op._name = mm_op_name
    mm_op._parents = node._parents[1:]
    mm_op._arguments = node._parents[1:]
    mm_op._tensor_meta = node.tensor_meta

    context.graph.replace_as_child(node._parents[1:], node, mm_op)

    context.graph.add_node(mm_op)

    scaling_op = MulOp()
    scaling_op_name = "scaled_" + node_name
    scaling_op._name = scaling_op_name
    scaler_op_name = "scaler_" + node._parents[2]
    scaling_op._tensor_meta = node.tensor_meta

    assert scaler_op_name in context.graph.node_table.keys(), "Scalar op missing."

    scaling_op._parents = [mm_op_name, scaler_op_name]
    scaling_op._arguments = [mm_op_name, scaler_op_name]
    mm_op._children = [scaling_op_name]

    context.graph.add_node(scaling_op)

    add_op = AddOp()
    add_op_name = "biased_" + node_name
    add_op._name = add_op_name
    add_op._parents = [scaling_op_name, node._parents[0]]
    add_op._arguments = [scaling_op_name, node._parents[0]]
    add_op._tensor_meta = node.tensor_meta

    context.graph.replace_as_child([node._parents[0]], node, add_op)

    add_op._children = node._children
    scaling_op._children = [add_op_name]

    context.graph.replace_as_parent(node, node._children, add_op)

    context.graph.add_node(add_op)

    node._parents = []
    node._children = []
    context.graph.delete_node(node, [])

@mask_guard_factory(op_type=AddMMOp, quantization_mask=[
    Quantizable, Unquantized, Quantizable
])
def _quantize_addmm_quq(node: Op, context: QuantizationContext) -> None | QuantizationState:
    """
    Convert the dag
    (addmm arg1:u, arg2:u, arg3:q)
    to
    (add (mul (mm arg2, arg3), scalar_arg3), arg1)
    """

    # If the matrix is quantized along the wrong axis, return None.
    # Note: `check_axis` is defined on `Quantized` only, but the 
    #   mask guard guarantees that this argument is of the correct type.
    if not context.quantization_table[node._parents[2]].check_axis(1):
        return None

    return Consumer(rewrite=_rewrite_addmm_quq)

def get_op_mask(op: Op, context: QuantizationContext) -> str:
    """
    Docstring for get_op_mask
    
    Args:
        op (Op): The op to compute the mask for.
        context (QuantizationContext): The context.

    Returns:
        str: Quantization mask for the op.
    """
    states = []
    # TODO: add error handling when for some reason a parent is missing.
    for parent in op._parents:
        states.append('u' if isinstance(context.quantization_table[parent], Unquantized) else 'q')

    return ''.join(states)

def dispatch_op_quantization(op: Op, context: QuantizationContext) -> None | QuantizationState:
    """
    Function for finding a `_quantize` function for `op`. If a suitable
    function is found, it is called, otherwise `False` is returned to
    signal the failure of quantizing the operation.
    
    Args:
        op (Op): Op to find quantization for.
        context (Context): The quantization context.

    Returns:
        bool: whether quantizing the op was successful.
    """
    mask = get_op_mask(op, context)
    try:
        return QuantizationFunctionRegistry[type(op)][mask](op, context)
    except KeyError:
        return None

def quantize_node(node: Op, context: QuantizationContext):
    """
    This method takes a node, all of whose parents have been analyzed.

    1. We start by checking, if there is a quantizer function for the current op
        with the given quantization mask.
        Note that the final state of the node can be `Unquantized` even if a quantizer
        function applies (e.g. matmul)
    2. If none applies, we fall back, and for each quantized parent we insert an 
        dequantization op.

    Args:
        op (Op): Op to find quantization for.
        context (Context): The quantization context.
    """

    quantized_parent_nodes: list[tuple[int, str]] = []

    # Check of we can quantize the op with the given mask.
    if (state := dispatch_op_quantization(node, context)):
        context.quantization_table[node._name] = state
        return

    # If not, we fall back to dequantizing all quantized parents,
    # and performing the original op.

    # Record all quantized parents
    for idx, parent_name in enumerate(node._parents):
        if isinstance(context.quantization_table[parent_name], Quantizable):
            quantized_parent_nodes.append((idx, parent_name))

    for idx, parent_name in quantized_parent_nodes:
        parent_op = context.graph.node_table[parent_name]
        dequantize_op = MulOp()
        dequantize_op_name = "dequantize_" + parent_op._name
        dequantize_op._name = dequantize_op_name

        parent_scaler_name = "scaler_" + parent_name
        parent_scaler_node = context.graph.node_table[parent_scaler_name]

        dequantize_op._tensor_meta = {
            "shape" : parent_op._tensor_meta["shape"],
            "dtype" : parent_scaler_node._tensor_meta["dtype"]
        }

        dequantize_op._parents = [parent_name, parent_scaler_name]
        dequantize_op._children = [node._name]

        parent_op._children.remove(node._name)
        parent_op._children.append(dequantize_op)
        node._parents[idx] = dequantize_op_name

        context.graph.add_node(dequantize_op)
    
    context.quantization_table[node._name] = Unquantized()

def rewrite_placeholder(node: Op, context: QuantizationContext) -> None:
    node_name = node.name
    scaler_node = PlaceholderOp()
    scaler_node._name = "scaler_" + node_name
    
    # We apply channel(row)-wise normalization.
    # This tensor stores the column normalization constants
    node_shape = node.tensor_meta["shape"]
    scaler_node._tensor_meta["shape"] = (node_shape[0], 1)
    scaler_node._tensor_meta["dtype"] = node._tensor_meta["dtype"]

    node._tensor_meta["dtype"] = context.target_dtype
    context.graph.add_node(scaler_node, node_type=NodeType.FakeNode)


def rewrite_node(node: Op, context: QuantizationContext):
    node_name = node.name

    node_status = context.quantization_table[node_name]
    
    if isinstance(node_status, Quantizable):
        if node_status.axis is not None:
            context.quantization_table[node_name] = Quantized(axis=node_status.axis)
            node_status.rewrite(node, context)
        else:
            context.quantization_table[node_name] = Unquantized()

    elif isinstance(node_status, Consumer):
        node_status.rewrite(node, context)

def quantise_graph(
        graph: Graph,
        target_dtype: TensorDType = TensorDType.Int8,
        quantization_mode: QuantizationMode = QuantizationMode.WOChannelWise
):
    """
    Weight only quantization.

    The following algorithm is used:
    1. Preparation stage.
        Each input node is marked as `Unquantized` and each
        param node is marked `Quantized`. For each param node
        we also insert a scaler param, which is its weight only
        scaler. TODO: add reference.

        For each node processed, we check if any children can already be
        quantized (i.e. all their parents are marked with either `Quantized`
        or `Unquantized`).
        We add these to a queue (list).

    2. Graph quantization
        For each element of the queue, we call the `quantize_node` function
        to attempt quantizing it.

    Args:
        graph (Graph): Graph to quantize
        target_dtype (TensorDType): dtype to quantize the model weights to
    """

    context = QuantizationContext(
        graph=graph,
        quantization_mode=quantization_mode,
        target_dtype=target_dtype,
    )

    evaluator_pass = Pass(
        processor=quantize_node,
        context=context,
    )

    for in_node in graph.inputs:
        node_name = in_node.name

        context.quantization_table[node_name] = Unquantized()
        evaluator_pass.processed.append(node_name)
        node = context.graph.node_table[node_name]
        evaluator_pass.check_ready_children(node)
    
                
    # Insert quantization scaler weights for each weight and mark them quantized.
    for node in graph.params:
        node_name = node.name

        context.quantization_table[node_name] = Quantizable(rewrite=rewrite_placeholder)
        evaluator_pass.processed.append(node_name)

        node = context.graph.node_table[node_name]
        evaluator_pass.check_ready_children(node)

    evaluator_pass.run_pass()

    rewriter_pass = Pass(
        processor=rewrite_node,
        context=context,
    )

    for node in graph.params + graph.inputs:
        rewriter_pass.queue.append(node._name)

    rewriter_pass.run_pass()
        