# ===- quantize.py -------------------------------------------------------------
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
# Core quantization framework for Buddy graph IR. Provides the pass
# infrastructure (eligibility analysis, constraint propagation, graph
# rewriting) and a stable topological sort that preserves the original
# operation order after quantization node insertion.
#
# ===---------------------------------------------------------------------------

from ... import Graph, NodeType
from ...operation import *
from ... import DeviceType
from torch.fx.immutable_collections import immutable_list
from ...type import TensorDType

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, TypeVar, Any
from itertools import product
from abc import ABC, abstractmethod


class QuantizationConstraint(ABC):

    gain: int = 1

    @abstractmethod
    def check(self, other: "QuantizationConstraint") -> bool:
        """
        Check if `other` is the same as self
        """
        pass

    @abstractmethod
    def hash(self) -> Any:
        pass


class QuantizationState:
    def check_constraint(self, constraint: QuantizationConstraint) -> bool:
        return False

    def add_constraint(self, constraint: QuantizationConstraint):
        pass

    def propagate_constraint(self, constraint: QuantizationConstraint) -> bool:
        return False


class Rewritable:
    rewrite: Callable[[Op, "QuantizationContext"], None]

    def set_rewrite(self, rewrite: Callable[[Op, "QuantizationContext"], None]):
        self.rewrite = rewrite


class Quantizable(QuantizationState, Rewritable):
    callback: Callable[[QuantizationConstraint], None] | None = None
    constraints: list[QuantizationConstraint]

    def __init__(
        self,
        constraints: list[QuantizationConstraint] | None = None,
    ):
        if constraints is None:
            self.constraints = []
        else:
            self.constraints = constraints

    def backprop_constraint(
        self, constraint: QuantizationConstraint, add_on_success: bool
    ) -> bool:
        if (backprop_constraint := self.callback(constraint)) is None:
            return False

        if add_on_success:
            self.add_constraint(backprop_constraint)

        return True

    def add_constraint(self, constraint):
        for i in range(len(self.constraints)):
            if self.constraints[i].check(constraint):
                self.constraints[i].gain += constraint.gain
                return

        self.constraints.append(constraint)

    def check_constraint(self, constraint: QuantizationConstraint) -> bool:
        return self.backprop_constraint(
            constraint=constraint, add_on_success=False
        )

    def propagate_constraint(self, constraint: QuantizationConstraint) -> bool:
        return self.backprop_constraint(
            constraint=constraint, add_on_success=True
        )


class ToQuantize(Rewritable):
    constraint: QuantizationConstraint

    def __init__(
        self,
        constraint: QuantizationConstraint,
        rewrite: Callable,
    ):
        self.constraint = constraint
        self.rewrite = rewrite

    @classmethod
    def from_quantizable(
        cls, state: Quantizable, constraint: QuantizationConstraint
    ) -> "ToQuantize":
        return cls(
            constraint=constraint,
            rewrite=state.rewrite,
        )


class Quantized(Quantizable):
    constraint: QuantizationConstraint

    def __init__(self, constraint: QuantizationConstraint = None):
        self.constraint = constraint


class Unquantized(QuantizationState):
    pass


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


def max_gain_quantization(
    backward_state: Quantizable, forward_state: Quantizable
) -> Unquantized | ToQuantize:
    possible_quantizations = set(
        [constraint.hash() for constraint in forward_state.constraints]
    )
    max_gain_constraint: QuantizationConstraint = None
    for constraint in backward_state.constraints:
        if not (constraint.hash() in possible_quantizations):
            continue

        if (max_gain_constraint is None) or (
            constraint.gain > max_gain_constraint.gain
        ):
            max_gain_constraint = constraint

    if max_gain_constraint is None:
        return Unquantized()

    return ToQuantize.from_quantizable(
        state=backward_state, constraint=max_gain_constraint
    )


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
        context: QuantizationContext,
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
            if child_name not in self.context.graph.node_table:
                continue
            child_node = self.context.graph.node_table[child_name]

            for parent_name in child_node._parents:
                if parent_name not in self.processed:
                    break
            else:
                if child_name not in self.processed:
                    self.queue.append(child_name)

    def run_pass(self):
        """
        Run quantization pass.
        """
        while len(self.queue) > 0:
            node_name = self.queue.pop()
            if node_name in self.processed:
                continue
            node = self.context.graph.node_table[node_name]

            self.processor(node, self.context)
            self.processed.append(node_name)
            self.check_ready_children(node)


class EligibilityPattern:
    """
    Base class for eligibility pattern matching.
    Allows composable patterns using | operator.
    """

    def __init__(self, *state_types):
        self.state_types = state_types

    def matches(self, state: QuantizationState) -> bool:
        """Check if state matches this pattern."""
        return isinstance(state, self.state_types)

    def __or__(self, other: "EligibilityPattern") -> "EligibilityPattern":
        """Allow pattern1 | pattern2 syntax."""
        return EligibilityPattern(*(self.state_types + other.state_types))


AnyQuantizable = EligibilityPattern(Quantizable)
AnyQuantized = EligibilityPattern(Quantized)
AnyUnquantized = EligibilityPattern(Unquantized)
AnyConsumer = EligibilityPattern(Consumer)


def requires_parents(*patterns: EligibilityPattern):
    """
    Decorator to declaratively specify parent eligibility patterns for quantization.
    """

    def decorator(cls: "QuantizationMethod"):
        if cls._eligibility_patterns is None:
            cls._eligibility_patterns = []

        cls._eligibility_patterns.append(patterns)
        return cls

    return decorator


class QuantizationMethod(ABC):
    """
    Function object for installing quantization behavior.
    """

    _eligibility_patterns: list[EligibilityPattern] | None = None

    def _populate_info(
        self, node: Op, context: QuantizationContext, state: QuantizationState
    ) -> QuantizationState:

        if isinstance(state, Quantizable):

            def callback_wrapper(constr):
                return self.callback(constr, node, context)

            state.callback = callback_wrapper

        if isinstance(state, Rewritable):
            state.rewrite = self.rewriter

        return state

    def _check_eligibility(
        self, node: Op, context: QuantizationContext
    ) -> QuantizationState:
        if self._eligibility_patterns is None:
            return Quantizable()

        for patterns in self._eligibility_patterns:
            if len(node._parents) != len(patterns):
                continue

            all_match = True
            for parent_name, pattern in zip(node._parents, patterns):
                pattern: EligibilityPattern
                parent_state = context.quantization_table.get(parent_name)
                if parent_state is None or not pattern.matches(parent_state):
                    all_match = False
                    break

            if all_match:
                return Quantizable()

        return Unquantized()

    def check_eligibility(self, node: Op, context: QuantizationContext):
        state = self._check_eligibility(node, context)
        return self._populate_info(node, context, state)

    @abstractmethod
    def rewriter(self, node: Op, context: QuantizationContext):
        pass

    def callback(
        self,
        constraint: QuantizationConstraint,
        node: Op,
        context: QuantizationContext,
    ) -> QuantizationConstraint | None:
        return None

    @abstractmethod
    def _forward(
        self, node: Op, context: QuantizationContext
    ) -> QuantizationState | None:
        pass

    def forward(
        self, node: Op, context: QuantizationContext
    ) -> None | QuantizationState:
        state = self._forward(node, context)
        return self._populate_info(node, context, state)


QuantizationMethodType = TypeVar(
    "QuantizationMethodType", bound=QuantizationMethod
)


class MethodRegistry(dict):
    dequantizer: Callable[[Op, QuantizationContext], Op] = None


QuantizationType = TypeVar("Method", bound=Quantization)
OpType = TypeVar("OpType", bound=Op)
QuantizationFunctionRegistry: dict = {}


def validate_registry():
    for method, method_registry in QuantizationFunctionRegistry.items():
        assert (
            method_registry.dequantizer is not None
        ), f"Each quantization method must register a dequantization function ({method})"


def register_quantizer(quantization: QuantizationType, op_type: OpType):
    def guard(cls: QuantizationMethodType):
        fn_obj = cls()

        assert (
            op_type not in QuantizationFunctionRegistry.keys()
        ), f"Only one quantization function should be declared per operation ({op_type} has two)"
        QuantizationFunctionRegistry.setdefault(quantization, MethodRegistry())[
            op_type
        ] = fn_obj

    return guard


def register_parameterized(quantization: QuantizationType, params: list[tuple]):
    def parameterized(cls: QuantizationMethodType):
        for op_type, param_dict in params:

            param_dict["buddy_op"] = op_type

            subclass_name = f"{cls.__name__}_{op_type.__name__}"
            new_cls = type(subclass_name, (cls,), param_dict)

            guard = register_quantizer(
                quantization=quantization, op_type=op_type
            )
            guard(new_cls)

    return parameterized


def register_dequantizer(quantization_method: QuantizationMethodType):
    def guard(
        fn: Callable[[Op, QuantizationContext], None | QuantizationState]
    ):
        def guarded_fn(
            op: Op, context: QuantizationContext
        ) -> None | QuantizationState:
            assert isinstance(context.quantization, quantization_method)
            return fn(op, context)

        assert (
            QuantizationFunctionRegistry.setdefault(
                quantization_method, MethodRegistry()
            ).dequantizer
            is None
        ), f"Only one dequantizer is allowed per quantization method ({quantization_method})"
        QuantizationFunctionRegistry.setdefault(
            quantization_method, MethodRegistry()
        ).dequantizer = guarded_fn

    return guard


def get_quantization_method(
    op: Op, context: QuantizationContext
) -> QuantizationMethod | None:
    try:
        method = QuantizationFunctionRegistry[type(context.quantization)][
            type(op)
        ]
        return method
    except KeyError:
        return None


def check_op_eligibility(
    op: Op, context: QuantizationContext
) -> QuantizationMethod | None:
    try:
        method = QuantizationFunctionRegistry[type(context.quantization)][
            type(op)
        ]
        if method.check_eligibility(op, context):
            return method
        return None
    except KeyError:
        return None


def dispatch_op_quantization(
    op: Op, context: QuantizationContext, populate_constraint: bool = False
) -> None | QuantizationState:
    try:
        return QuantizationFunctionRegistry[type(context.quantization)][
            type(op)
        ](op, context, populate_constraints=populate_constraint)
    except KeyError:
        return None


def quantize_node(node: Op, context: QuantizationContext):
    if method := get_quantization_method(node, context):
        context.quantization_table[node._name] = method.check_eligibility(
            node, context
        )
        return

    context.quantization_table[node._name] = Unquantized()


def get_dequantized(op: Op, context: QuantizationContext) -> Op:
    try:
        return QuantizationFunctionRegistry[
            type(context.quantization)
        ].dequantizer(op, context)
    except KeyError:
        raise RuntimeError(
            f"No dequantizer registered for {type(context.quantization)}"
        )


def rewrite_node(node: Op, context: QuantizationContext):
    node_name = node.name

    node_requests = context.quantization_table[node_name]

    if isinstance(node_requests, Quantizable):
        if len(node_requests.constraints) != 0:
            if method := get_quantization_method(node, context):
                if isinstance(
                    forward_quantizability := method.forward(node, context),
                    Quantizable,
                ):

                    context.quantization_table[node_name] = (
                        reeval_state := max_gain_quantization(
                            node_requests, forward_quantizability
                        )
                    )
                    if isinstance(reeval_state, ToQuantize):
                        reeval_state.rewrite(node, context)
                        return

    elif isinstance(node_requests, Consumer):
        node_requests.rewrite(node, context)
        return

    # Default: dequantize all actually quantized parents
    quantized_parent_nodes: list[tuple[int, str]] = []

    for idx, parent_name in enumerate(node._parents):
        if isinstance(context.quantization_table.get(parent_name), Quantized):
            quantized_parent_nodes.append((idx, parent_name))

    for idx, parent_name in quantized_parent_nodes:
        parent_op = context.graph.node_table[parent_name]
        dequantize_op = get_dequantized(parent_op, context)

        dequantize_op._children.append(node._name)
        if node._name in parent_op._children:
            parent_op._children.remove(node._name)
        parent_op._children.append(dequantize_op._name)
        node._parents[idx] = dequantize_op.name
        for i, arg in enumerate(node._arguments):
            if arg == parent_name:
                node._arguments[i] = dequantize_op.name


def sort_graph(graph: Graph):
    """
    Sorts all the PlaceholderOp's to the
    front of graph._body. See Issue #667

    After quantization, new PlaceholderOp (scaler) and MulOp (dequantize) nodes
    are appended at the end of _body. This function restores topological order:
      1. Params (in _fake_params order)
      2. Inputs (in _inputs order)
      3. Remaining ops in topological order
      4. OutputOps last
    It also rebuilds _fake_params and _inputs index lists.
    """
    param_nodes = [graph._body[idx] for idx in graph._fake_params]
    input_nodes = [graph._body[idx] for idx in graph._inputs]

    # Deterministic param ordering: original params first (in trace order),
    # then scaler params sorted by their corresponding weight's position.
    # This is critical because prefill and decode traces may produce scalers
    # in different orders (due to DFS traversal differences), but the packed
    # parameter data is serialized from prefill's ordering—decode must match.
    orig_params = [n for n in param_nodes if not n.name.startswith("scaler_")]
    scaler_params = [n for n in param_nodes if n.name.startswith("scaler_")]
    if scaler_params:
        weight_pos = {n.name: i for i, n in enumerate(orig_params)}
        scaler_params.sort(
            key=lambda s: weight_pos.get(s.name[len("scaler_") :], float("inf"))
        )
    param_nodes = orig_params + scaler_params

    placeholder_ids = set(id(n) for n in param_nodes + input_nodes)

    other_nodes = []
    output_nodes = []
    for n in graph._body:
        if id(n) in placeholder_ids:
            continue
        if isinstance(n, OutputOp):
            output_nodes.append(n)
        else:
            other_nodes.append(n)

    import heapq
    from collections import defaultdict

    original_order = {n.name: i for i, n in enumerate(other_nodes)}
    node_map = {n.name: n for n in other_nodes}
    remaining_set = set(node_map.keys())

    # Build adjacency from _parents (authoritative) instead of _children
    # to avoid inconsistencies between _parents and _children.
    children_from_parents = defaultdict(list)
    in_degree = {}
    for n in other_nodes:
        deg = 0
        for p in n._parents:
            if p in remaining_set:
                deg += 1
                children_from_parents[p].append(n.name)
        in_degree[n.name] = deg

    heap = []
    for n in other_nodes:
        if in_degree[n.name] == 0:
            heapq.heappush(heap, (original_order[n.name], n.name))

    sorted_other = []
    while heap:
        _, name = heapq.heappop(heap)
        sorted_other.append(node_map[name])
        remaining_set.discard(name)
        for child_name in children_from_parents[name]:
            if child_name in remaining_set:
                in_degree[child_name] -= 1
                if in_degree[child_name] == 0:
                    heapq.heappush(
                        heap,
                        (
                            original_order.get(child_name, len(other_nodes)),
                            child_name,
                        ),
                    )

    if remaining_set:
        for n in other_nodes:
            if n.name in remaining_set:
                sorted_other.append(n)

    new_body = param_nodes + input_nodes + sorted_other + output_nodes
    graph._body = new_body
    graph._fake_params = list(range(len(param_nodes)))
    graph._inputs = list(
        range(len(param_nodes), len(param_nodes) + len(input_nodes))
    )


def quantise_graph(
    graph: Graph,
    quantization: Quantization,
    target_dtype: TensorDType = TensorDType.Int8,
):
    """
    Driver method for quantizing a Buddy Graph.

    Args:
        graph (Graph): Graph to quantize
        quantization (Quantization): The quantization method to use
        target_dtype (TensorDType): dtype to quantize the model weights to
    """

    validate_registry()

    context = QuantizationContext(
        graph=graph,
        quantization=quantization,
        target_dtype=target_dtype,
    )

    # Collect all root nodes (no parents) to ensure full graph traversal.
    all_root_names = []
    seen = set()
    for node in graph.body:
        if len(node._parents) == 0 and node.name not in seen:
            all_root_names.append(node.name)
            seen.add(node.name)

    evaluator_pass = Pass(
        processor=quantize_node,
        context=context,
    )

    for name in all_root_names:
        evaluator_pass.queue.append(name)

    evaluator_pass.run_pass()

    rewriter_pass = Pass(
        processor=rewrite_node,
        context=context,
    )

    for name in all_root_names:
        rewriter_pass.queue.append(name)

    rewriter_pass.run_pass()

    sort_graph(context.graph)
