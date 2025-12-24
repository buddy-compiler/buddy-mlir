from .. import Graph, NodeType
from ..operation import *
from .. import DeviceType
from torch.fx.immutable_collections import immutable_list
from ..type import TensorDType

from enum import Enum, auto
from typing import Callable

class QuantizationState(Enum):
    Quantized = auto()
    Unquantized = auto()

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
    quantization_queue: list[str]

    def __init__(self, graph: Graph):
        self.graph = graph
        self.quantization_table = {}
        self.quantization_queue = []

def check_ready_children(node: Op, context: QuantizationContext):
    """
    Evaluate whether any child of `node` is ready for quantization (i.e.
    all its parents appear in `quantization_table`).

    Args:
        node (Op): Node to check the children of
        context (QuantizationContext): The context.

    """
    for child_name in node._children:
            child_node = context.graph.node_table[child_name]

            for parent_name in child_node._parents:
                if parent_name not in context.quantization_table.keys():
                    break
            else:
                context.quantization_queue.append(child_name)

QuantizationFunctionRegistry: dict[type, dict[str, Callable[[Op, QuantizationContext], bool]]] = {}

# ----
# Quantization methods.
# ----

def mask_guard_factory(
        op_type: type,
        quantization_mask: list[QuantizationState],
):
    """
    This factory creates a custom decorator for each node quantization function based on the op type
    they process, and the quantization mask they accept.

    It also adds the function the `QuantizationFunctionRegistry`, which automatically dispatches
    op quantizations to supported node types and quantitation masks.
    """
    def mask_guard(fn: Callable[[Op, QuantizationContext], bool]):
        def guarded_fn(op: Op, context: QuantizationContext) -> bool:
            assert isinstance(op, op_type)
            assert len(quantization_mask) == len(op._parents), "Op should have as many arguments as quantization mask entries."
            assert all([context.quantization_table[parent] == quantization_state for (parent, quantization_state) in zip(op._parents, quantization_mask)]), \
                "Op argument quantization should match quantization mask."
            return fn(op, context)

        quantizer_mask_str = ''.join(['q' if mask == QuantizationState.Quantized else 'u' for mask in quantization_mask])

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
    QuantizationState.Quantized
])
def _quantize_view_q(op: Op, context: QuantizationContext) -> bool:
    """
    Docstring for quantize_view_q
    
    Quantizer method for ViewOp, with a single quantized input node.

    Inserts a view op that acts on the quantization scaler of the scaler of the
    input node.
    """
    parent_name = op._parents[0]
    #parent_node = context.graph.node_table[parent_name]
    parent_scaler_name = "scaler_" + parent_name

    assert parent_scaler_name in context.graph.node_table.keys(), "Parent of ViewOp should have quantization scaler."

    scaler_view_op = ViewOp()
    scaler_view_op._name = "scaler_" + op._name
    scaler_view_op._parents = [parent_scaler_name]
    # TODO: set _tensor_meta

    context.graph.add_node(scaler_view_op)

    return True

@mask_guard_factory(op_type=PermuteOp, quantization_mask=[
    QuantizationState.Quantized
])
def _quantize_permute_q(op: Op, context: QuantizationContext) -> bool:
    """

    """
    parent_name = op._parents[0]

    parent_node = context.graph.node_table[parent_name]
    op._tensor_meta["dtype"] = parent_node.tensor_meta["dtype"]

    parent_scaler_name = "scaler_" + parent_name

    assert parent_scaler_name in context.graph.node_table.keys(), "Parent of PermuteOp should have quantization scaler."

    scaler_op = PermuteOp()
    scaler_op._name = "scaler_" + op._name
    scaler_op._parents = [parent_scaler_name]
    # FIXME: this should be actually dynamic
    #scaler_op._tensor_meta["shape"] = (scaler_op._tensor_meta["shape"][1], scaler_op._tensor_meta["shape"][0])
    scaler_op._tensor_meta["dtype"] = op.tensor_meta["dtype"]
    
    context.graph.add_node(scaler_op)

    return True

@mask_guard_factory(op_type=MatmulOp, quantization_mask=[
    QuantizationState.Unquantized, QuantizationState.Unquantized, QuantizationState.Quantized
])
def _quantize_matmul_uuq(node: MatmulOp, context: QuantizationContext) -> bool:
    
    # TODO: add overload support
    return False

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
        states.append('u' if context.quantization_table[parent] == QuantizationState.Unquantized else 'q')

    return ''.join(states)

def dispatch_op_quantization(op: Op, context: QuantizationContext) -> bool:
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
        return False

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

    #quantization_state: QuantizationState = QuantizationState.AttemptQuantization
    quantized_parent_nodes: list[tuple[int, str]] = []

    # record all 
    for idx, parent_name in enumerate(node._parents):
        # If any of the parents is not quantized, we can't quantize the current node.
            quantized_parent_nodes.append((idx, parent_name))

    # Check of we can quantize the op with the given mask.
    if dispatch_op_quantization(node, context):
        # FIXME: This here is incorrect.
        context.quantization_table[node._name] = QuantizationState.Quantized
        return

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
    
    context.quantization_table[node._name] = QuantizationState.Unquantized

def quantise_graph(graph: Graph, target_dtype: TensorDType = TensorDType.Int8):
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
    )

    for in_node in graph.inputs:
        node_name = in_node.name
        context.quantization_table[node_name] = QuantizationState.Unquantized
    
        # maybe rewrite this ool.
        context.check_ready_children(node=in_node)
    
                
    # Insert quantization scaler weights for each weight and mark them quantized.
    for node in graph.params:
        node_name = node.name
        scaler_node = PlaceholderOp()
        scaler_node._name = "scaler_" + node_name
        
        # We apply channel(row)-wise normalization.
        # This tensor stores the column normalization constants
        node_shape = node.tensor_meta["shape"]
        scaler_node._tensor_meta["shape"] = (node_shape[0], 1)
        scaler_node._tensor_meta["dtype"] = node._tensor_meta["dtype"]

        node._tensor_meta["dtype"] = target_dtype

        graph.add_node(scaler_node, node_type=NodeType.FakeNode)
        context.quantization_table[node_name] = QuantizationState.Quantized
        
        context.check_ready_children(node=node)

    while len(context.quantization_queue) > 0:
        node_name = context.quantization_queue.pop()
        node = context.graph.node_table[node_name]

        quantize_node(node, context)
        context.check_ready_children(node)