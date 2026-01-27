# Quantization Infrastructure

## Overview

The quantization framework provides a generic infrastructure for implementing different quantization methods. The framework separates concerns through a declarative interface, allowing implementers to focus on the specific behavior of each operation under quantization.

## The Four-Phase Quantization Pipeline

The framework performs quantization in four distinct phases:

1. **Eligibility Detection** - Determine which nodes _could_ be quantized given their parent states
2. **Gain Computation** - Compute and propagate quantization gains backward through the graph
3. **Quantization Selection** - Decide which nodes will be quantized based on gains and exact parent states
4. **Graph Rewriting** - Transform the graph to use the selected quantizations

### Phase 1: Eligibility Detection

In the first pass, we traverse the graph and mark each node's eligibility for quantization based on its parent states.

- A node is marked `Quantizable` if there exists _some_ quantization of its parents that would allow the node to be quantized
- This is a **possibility check**, not a commitment - we're determining potential, not making decisions
- Eligibility is declared via the `@requires_parents` decorator (see QuantizationMethod section below)

Example: A `permute` node with a `Quantizable` parent is itself `Quantizable`, since quantization can propagate through permute operations.

### Phase 2: Gain Computation (Backward Pass)

During the same forward traversal, we also propagate quantization constraints backward:

- Operations with inherent gain (e.g., matmul benefits from quantized inputs) propagate constraints upstream
- Each constraint carries its cumulative gain value
- Constraints are stored in each node's `QuantizationState.constraints` list
- Operations without inherent gain can still propagate constraints to enable downstream gains

Example: A matmul operation requests that its weight input be quantized along a specific axis, propagating this constraint (with gain) to the weight parameter node.

### Phase 3: Quantization Selection

In the second pass, we make final decisions about which nodes to quantize:

- For each node previously marked `Quantizable`, we recompute available quantizations based on parents' **exact** states
- We compare requested quantizations (from Phase 2) with available quantizations (from exact parent states)
- The quantization with maximum gain is selected, or the node is marked `Unquantized` if no valid quantization exists

This phase uses the `forward()` method of `QuantizationMethod` to compute exact quantization options.

### Phase 4: Graph Rewriting

Once quantization decisions are made, we rewrite the graph:

- Nodes marked `ToQuantize` have their `rewriter()` method called
- `Consumer` nodes (benefit from quantized inputs but have unquantized outputs) are rewritten
- Dequantization operations are inserted where quantized values flow into unquantized operations

## Core Components

### QuantizationState Types

The pipeline uses several state types to track quantization status:

- **`Quantizable`** - Node can potentially be quantized; stores downstream quantization requests (Phases 1-2)
- **`ToQuantize`** - Transient state between selection and rewriting (Phase 3 → 4)
- **`Quantized`** - Node has been successfully quantized and rewritten (Phase 4)
- **`Unquantized`** - Node has non-quantized output throughout the pipeline
- **`Consumer`** - Node benefits from quantized inputs but produces unquantized output (e.g., matmul)

### QuantizationMethod Interface

`QuantizationMethod` is where all operation-specific quantization logic lives. The responsibilities are separated as follows.

#### 1. Eligibility Declaration (Phase 1)

Eligibility is declared using the `@requires_parents` decorator with `EligibilityPattern` instances:

```python
@requires_parents(AnyQuantizable)
class PermuteQuantizationMethod(QuantizationMethod):
    # Single parent must be Quantizable
    ...

@requires_parents(AnyQuantizable, AnyQuantizable)
class SomeOpQuantizationMethod(QuantizationMethod):
    # Both parents must be Quantizable
    ...
```

**Multiple decorators provide OR logic:**

```python
@requires_parents(AnyQuantizable, AnyQuantizable)
@requires_parents(AnyQuantizable, AnyUnquantized)
@requires_parents(AnyUnquantized, AnyQuantizable)
class MatmulQuantizationMethod(QuantizationMethod):
    # At least one parent must be Quantizable
    # Matches: (Q, Q), (Q, U), or (U, Q)
    ...
```

**Pattern composition with | operator:**

```python
@requires_parents(AnyQuantizable | AnyConsumer, AnyQuantizable)
class ExampleMethod(QuantizationMethod):
    # First parent can be Quantizable OR Consumer
    # Second parent must be Quantizable
    ...
```

**Special handling.**

In cases where the declarative syntax is not enough to express the intent, one can manually implement the `check_eligibility` method. This is the same method that is populated by the decorators

```python
class PlaceholderQuantizationMethod(QuantizationMethod):
    def check_eligibility(self, node, context):
        # Custom logic for nodes without parents
        ...
```

**Available patterns:**
- `AnyQuantizable` - Matches `Quantizable` state
- `AnyQuantized` - Matches `Quantized` state
- `AnyUnquantized` - Matches `Unquantized` state
- `AnyConsumer` - Matches `Consumer` state

#### 2. Backward Propagation (Phase 2)

The `callback()` method propagates quantization constraints upstream:

```python
def callback(self, constraint, node, context):
    """
    Given a downstream quantization request, determine if this node
    can satisfy it and propagate appropriate constraints to parents.

    Returns:
        QuantizationConstraint to add to this node, or None if unsupported
    """
    # Transform constraint for parent node structure
    parent_constraint = transform_constraint(constraint, node)

    # Propagate to parent
    if context.quantization_table[parent_name].propagate_constraint(parent_constraint):
        return constraint

    return None
```

#### 3. Forward Quantization (Phase 3)

The `forward()` method computes exact quantization options given exact parent states:

```python
def forward(self, node, context):
    """
    Compute exact quantization possibilities given parents' exact states.

    Eligibility is already confirmed - focus on computing specific
    quantization constraints available.

    Returns:
        QuantizationState with concrete constraints
    """
    parent_state = context.quantization_table[node._parents[0]]

    if not isinstance(parent_state, Quantized):
        # Parent not not quantized, so we cannot proceed
        return Unquantized()

    # Compute exact quantization based on parent's exact state
    return Quantizable([specific_constraint])
```

**Key distinction:** Eligibility (Phase 1) asks "could this be quantized?", while `forward()` (Phase 3) asks "what exact quantizations are available?"

#### 4. Graph Rewriting (Phase 4)

The `rewriter()` method performs the actual graph transformation:

```python
def rewriter(self, node, context):
    """
    Rewrite the graph to implement the selected quantization.

    This might involve:
    - Creating scaling nodes
    - Updating tensor dtypes
    - Inserting dequantization operations
    """
    # Create quantization artifacts (scalers, etc.)
    # Update node metadata
    # Mark node as Quantized
    context.quantization_table[node.name] = Quantized(constraint=...)
```

## Implementing a New Quantization Method

To add a new quantization method to the framework:

### 1. Define the Quantization Type

Inherit from the `Quantization` abstract class:

```python
class MyQuantization(Quantization):
    pass
```

### 2. Define Quantization Constraints

Create a constraint class that inherits from `QuantizationConstraint`:

```python
class MyQuantizationConstraint(QuantizationConstraint):
    # Fields specific to your quantization
    param: int

    def __init__(self, param: int, gain: int = 1):
        self.param = param
        self.gain = gain

    def check(self, other) -> bool:
        """Check if two constraints are equivalent"""
        return other.param == self.param

    def hash(self):
        """
        Return hashable representation for deduplication
        and fast search.
        """
        return self.param
```

### 3. Implement the Dequantizer

Register a function to dequantize nodes:

```python
@register_dequantizer(quantization_method=MyQuantization)
def my_dequantizer(node: Op, context: QuantizationContext) -> Op:
    """
    Return an operation that dequantizes the given quantized node.
    """
    # Create and return dequantization operation
    return dequantize_op
```

### 4. Implement QuantizationMethods for Each Operation

For each operation type that participates in your quantization:

```python
@register_quantizer(quantization=MyQuantization, op_type=SomeOp)
@requires_parents(AnyQuantizable)  # Declare eligibility
class SomeOpQuantizationMethod(QuantizationMethod):

    def callback(self, constraint, node, context):
        # Backward propagation logic
        ...

    def forward(self, node, context):
        # Forward quantization computation
        ...

    def rewriter(self, node, context):
        # Graph rewriting logic
        ...
```

**For parameterized operations with small variations:**

```python
@register_parameterized(
    quantization=MyQuantization,
    params=[
        (Op1, {"param1": value1}),
        (Op2, {"param1": value2}),
        (Op3, {"param1": value3}),
    ]
)
@requires_parents(AnyQuantizable)
class ParameterizedQuantizationMethod(QuantizationMethod):
    param1: int  # Injected from params
    buddy_op: type  # Automatically set to op type

    # Methods can access self.param1 and self.buddy_op
    ...
```

### 5. Expose a User-Facing API

Create a convenience function for users:

```python
def quantize_with_my_method(graph: Graph, **kwargs):
    return quantise_graph(
        graph=graph,
        quantization=MyQuantization(),
        **kwargs
    )
```

## Example: Weight-Only Channel-Wise Quantization

See `weight_only_channel_wise.py` for a complete implementation example demonstrating:
- Custom constraint types (`ChannelWiseQuantizationConstraint`)
- Transparent operations (permute, transpose, etc.) that propagate quantization
- View/reshape operations with stride-based quantization axis tracking
- Consumer operations (matmul) that benefit from but don't output quantized values
- Root node handling (placeholder parameters vs inputs)
- Parameterized method registration for similar operations

## Design Principles

1. **Separation of Concerns** - Eligibility, gain computation, selection, and rewriting are separate phases
2. **Declarative Eligibility** - Parent state requirements are declared via decorators, not buried in code
3. **Composability** - Pattern matching supports OR logic (multiple decorators) and AND logic (pattern combinations)
4. **Extensibility** - New quantization methods plug in without modifying the framework
5. **Type Safety** - Each quantization method defines its own constraint types
6. **Lazy Evaluation** - Eligibility is checked before expensive computations
