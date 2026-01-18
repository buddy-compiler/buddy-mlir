from .. import Graph, NodeType
from ..operation import *
from .. import DeviceType
from torch.fx.immutable_collections import immutable_list
from ..type import TensorDType

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, TypeVar
from itertools import product
from abc import ABC, abstractmethod

import torch

class QuantizationState:
    pass

class QuantizationConstraint(ABC):
    
    @abstractmethod
    def check(self, other: "QuantizationConstraint") -> bool:
        """
        Check if `other` is compatible with `self`.
        """
        pass

class Rewritable:
    rewrite: Callable[[Op, "QuantizationContext"], None]

    def set_rewrite(self, rewrite: Callable[[Op, "QuantizationContext"], None]):
        self.rewrite = rewrite


class Quantized(QuantizationState):
    constraint: QuantizationConstraint

    def __init__(self, constraint: QuantizationConstraint):
        self.constraint = constraint

class Quantizable(QuantizationState, Rewritable):
    callback: Callable | None = None
    constraint: QuantizationConstraint | None

    def __init__(
            self,
            constraint: QuantizationConstraint | None = None,
            callback: Callable[[int], None] | None = None,
        ):
        if constraint:
            self.constraint = [constraint]
            if callback:
                callback(self.constraint)
        
        self.callback = callback
        self.constraint = constraint

    def set_constraint(self, constraint: QuantizationConstraint):
        assert self.constraint is None, "Constraint cannot be set twice on the same Quantized."

        self.constraint = constraint

        if self.callback:
            self.callback(self.constraint)

    def check_constraint(self, constraint: QuantizationConstraint) -> bool:
        """
        Check if the quantization of the tensor is compatible
        with the given constraint.

        When the quantization of the tensor is not set apriori,
        then it is supposed to be determined based on context,
        so if `constraint is None`, then it is compatible.

        Args:
            constraint (QuantizationConstraint): constraint to determine compatibility with.

        Returns:
            bool: Whether the quantization is compatible.
        """

        if self.constraint is None:
            self.set_constraint(constraint=constraint)
            return True

        return self.constraint.check(constraint)
    
class Unquantized(QuantizationState):
    pass

# TODO: this might need a better name.
class Consumer(Unquantized, Rewritable):
    """
    Nodes marked consumer are operations, that have `Unquantized` output,
    but do need to be changed, as one or more of their inputs is quantized.
    """

class Quantization(ABC):
    pass

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
    quantization: Quantization
    target_dtype: TensorDType

    def __init__(
            self,
            graph: Graph,
            quantization: Quantization,
            target_dtype: TensorDType,
):
        self.graph = graph
        self.quantization = quantization
        self.target_dtype = target_dtype
        self.quantization_table = {}

class Pass:
    """
    Class for dependently walking a Graph.
    Only processes on a node if all its parents have already 
    been processed.

    Args:
        processor (Callable[[Op, Context], None]): Callback to process
            the nodes of the graph
        context (QuantizationContext): Quantization context.
    """

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
        """
        Check the children of `node` if they have all their
        parents already processed, and add them to the queue.

        Args:
            node (Op): Node to check the children of.
        """
        for child_name in node._children:
            child_node = self.context.graph.node_table[child_name]

            for parent_name in child_node._parents:
                if parent_name not in self.processed:
                    break
            else:
                self.queue.append(child_name)

    def run_pass(self):
        """
        Run quantization pass.    
        """
        while len(self.queue) > 0:
            node_name = self.queue.pop()
            node = self.context.graph.node_table[node_name]

            self.processor(node, self.context)
            self.processed.append(node_name)
            self.check_ready_children(node)

class QuantizationMethod(ABC):
    """
    Function object for installing 
    """
    @staticmethod
    @abstractmethod
    def rewriter(node: Op, context: QuantizationContext):
        """
        Rewrite node in graph.

        (Must be implemented by any quantization method)
        """
        pass
    
    @staticmethod
    def callback(constraint: QuantizationConstraint, node: Op, context: QuantizationContext):
        """
        Determine the quantization state from the quantization of downstream ops.

        E.g. parameter ops are quantizable along any axis apriori, but given the 
        quantization of a child, we can determine the quantization of the placeholder.

        (Optionally implementable by quantization methods)
        """
        return None


    @staticmethod
    @abstractmethod
    def forward(node: Op, context: QuantizationState) -> QuantizationState | None:
        """
        Determine the quantization state from the quantization state of the parents. (E.g. 
        given that a parent op is quantized along axis 1, the result of a transpose op can
        only be quantized along a specific axis)

        Can still leave a callback, if necessary (e.g. multiple different quantization options
        available, and can only decide based on consumer information.

        (Must be implemented by quantization methods)
        """
        pass

    def __call__(self, node: Op, context: QuantizationContext) -> None | QuantizationState:
        quantization = self.__class__.forward(node, context)

        if isinstance(quantization, Quantizable):
            if quantization.constraint is None:
                # load the node and context into the callback before passing it.
                quantization.callback = lambda constr: self.__class__.callback(constr, node, context)
        
        if isinstance(quantization, Rewritable):
            quantization.set_rewrite(self.__class__.rewriter)
        
        return quantization

QuantizationMethodType = TypeVar("QuantizationMethodType", bound=QuantizationMethod)

class MethodRegistry(dict[OpType, QuantizationMethod]):
    dequantizer: Callable[[Op, QuantizationContext], Op] = None

# quantizationmethod -> op
QuantizationType = TypeVar("Method", bound=Quantization)
OpType = TypeVar("OpType", bound=Op)
QuantizationFunctionRegistry: dict[QuantizationType, MethodRegistry] = {}

def validate_registry():
    for method, method_registry in QuantizationFunctionRegistry.items():
        assert method_registry.dequantizer is not None, f"Each quantization method must register a dequantization function ({method})"

def register_quantizer(quantization: QuantizationType, op_type: OpType):
    def guard(cls: QuantizationMethodType):
        fn_obj = cls()
        #def guarded_fn(op: Op, context: QuantizationContext) -> None | QuantizationState:
        #    assert isinstance(op, op_type)
        #    assert isinstance(context.quantization_method, quantization)
        #    return fn(op, context)


        assert op_type not in QuantizationFunctionRegistry.keys(), f"Only one quantization function should be declared per operation ({op_type} has two)"
        QuantizationFunctionRegistry.setdefault(quantization, MethodRegistry())[op_type] = fn_obj
        # Delete this function from user name space by not returning it.
    
    return guard

def register_dequantizer(quantization_method: QuantizationMethodType):
    def guard(fn: Callable[[Op, QuantizationContext], None | QuantizationState]):
        def guarded_fn(op: Op, context: QuantizationContext) -> None | QuantizationState:
            assert isinstance(context.quantization_method, quantization_method)
            return fn(op, context)


        assert QuantizationFunctionRegistry.setdefault(quantization_method, MethodRegistry()).dequantizer is None, f"Only one dequantizer is allowed per quantization method ({quantization_method})"
        QuantizationFunctionRegistry.setdefault(quantization_method, MethodRegistry()).dequantizer = guarded_fn
        # Delete this function from user name space by not returning it.
    return guard

"""
All op should have 2 quantization functions:
1. _quantize_{op}: This is used to determine IF the operation CAN be quantized
    That is, they take an op, with state markers like `Quantizable` or `Unquantized`
    and decide if the op can be quantized. They might also enforce that the operations they
    need quantized WILL be quantized.

2. _rewrite_{op}: A callback, that `_quantize_op` wraps up with the chosen quantization state.
    This op is is purely responsible for rewriting the operation
"""


def dispatch_op_quantization(op: Op, context: QuantizationContext) -> None | QuantizationState:
    """
    Function for finding a `_quantize` function for `op`. If a suitable
    function is found, it is called, otherwise `None` is returned to
    signal that the operation will not be changed by the quantization pass
    
    Args:
        op (Op): Op to find quantization for.
        context (Context): The quantization context.

    Returns:
        bool: whether quantizing the op was successful.
    """
    try:
        return QuantizationFunctionRegistry[type(context.quantization)][type(op)](op, context)
    except KeyError:
        return None

def quantize_node(node: Op, context: QuantizationContext):
    """
    This method takes a node, all of whose parents have been analyzed, and 
    tries to apply a quantization pattern from the defined patterns, and specify
    the state of the node using the pattern. If not pattern is found, we return 
    by specifying the op as unquantized.

    Args:
        op (Op): Op to find quantization for.
        context (Context): The quantization context.
    """

    if (state := dispatch_op_quantization(node, context)):
        context.quantization_table[node._name] = state
        return
    
    context.quantization_table[node._name] = Unquantized()

def get_dequantized(op: Op, context: QuantizationContext) -> Op:
    try:
        return QuantizationFunctionRegistry[type(context.quantization_method)].dequantizer(op, context)
    except KeyError:
        print("Unreachable")
        exit(1)

def rewrite_node(node: Op, context: QuantizationContext):
    node_name = node.name

    # We check if needs to be rewritten with a specified pattern
    node_status = context.quantization_table[node_name]
    if isinstance(node_status, Quantizable):
        if node_status.constraint is not None:
            context.quantization_table[node_name] = Quantized(constraint=node_status.constraint)
            node_status.rewrite(node, context)
        else:
            context.quantization_table[node_name] = Unquantized()
        return

    # Consumer nodes force quantization, so they are also quantised.
    elif isinstance(node_status, Consumer):
        node_status.rewrite(node, context)
        return

    # Otherwise, we default to dequantizing all quantized parents,
    # and leaving the original op intact.
    quantized_parent_nodes: list[tuple[int, str]] = []

    # Record all quantized parents
    for idx, parent_name in enumerate(node._parents):
        if isinstance(context.quantization_table[parent_name], Quantizable):
            quantized_parent_nodes.append((idx, parent_name))

    for idx, parent_name in quantized_parent_nodes:
        parent_op = context.graph.node_table[parent_name]
        dequantize_op = get_dequantized(parent_op, context)
        
        dequantize_op._children.append(node._name)
        parent_op._children.remove(node._name)
        parent_op._children.append(dequantize_op)
        node._parents[idx] = dequantize_op.name

def sort_graph(graph: Graph):
    """
    Sorts all the PlaceholderOp's to the 
    front of graph._body. See Issue #667
    """

    pass

def quantise_graph(
        graph: Graph,
        quantization: Quantization,
        target_dtype: TensorDType = TensorDType.Int8,
):
    """
    TODO

    Args:
        graph (Graph): Graph to quantize
        target_dtype (TensorDType): dtype to quantize the model weights to
    """

    validate_registry()

    context = QuantizationContext(
        graph=graph,
        quantization=quantization,
        target_dtype=target_dtype,
    )

    # This pass is responsible for determining
    # a) which operations CAN be quantized,
    # b) which operations SHOULD be quantized.
    evaluator_pass = Pass(
        processor=quantize_node,
        context=context,
    )

    for placeholder_op in graph.inputs + graph.params:
        evaluator_pass.queue.append(placeholder_op.name)

    evaluator_pass.run_pass()

    rewriter_pass = Pass(
        processor=rewrite_node,
        context=context,
    )

    for node in graph.params + graph.inputs:
        rewriter_pass.queue.append(node._name)

    rewriter_pass.run_pass()
        
    sort_graph(context.graph)