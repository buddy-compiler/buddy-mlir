from ..quantise import (
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
  QuantizationMethod
)
from ... import Graph, NodeType
from ...operation import *

import torch
from typing import Any, Callable

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

class WeightOnlyQuantization(Quantization):
    pass

def register_wo_quantizer(op_type: OpType):
    return register_quantizer(quantization=WeightOnlyQuantization, op_type=op_type)

def register_wo_parameterized(params: list[tuple[OpType, Dict[str, Any]]]):
    return register_parameterized(quantization=WeightOnlyQuantization, params=params)

def propagate_scaler(node: Op, context: QuantizationContext):
    """
    Given that
    - node has a single parent,
    - this parent has a scaler,
    this function adds a scaler node for `node`, that performs the exact same
    operation in the parent's scaler to obtain a scaler for node.
    """
    pass

@register_wo_parameterized([
    (PermuteOp, {"torch_op" : torch.permute}),
    (TransposeOp, {"torch_op" : torch.transpose}),
    (UnsqueezeOp, {"torch_op" : torch.unsqueeze}),
    (SqueezeOp, {"torch_op" : torch.squeeze}),
    (CloneOp, {"torch_op" : torch.clone}),
])
class TransparentUnaryQuantizationMethod(QuantizationMethod):
    
    torch_op: Callable[[Any], torch.Tensor] = lambda : print("Undefined field: torch_op")
    buddy_op: type = None
    
    def rewriter(self, node, context):
        assert isinstance((state := context.quantization_table[node.name]), ToQuantize)

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
        scaler._arguments = [parent_scaler_name, *args]
        scaler._parents = [parent_scaler_name]
        parent_scaler_op._children.append(op_scaler_name)
        
        context.graph.add_node(scaler)

        context.quantization_table[node.name] = Quantized(constraint=state.constraint)
    
    def callback(self, constraint, node, context):
        assert isinstance(constraint, Constraint)
        
        state = context.quantization_table[node.name]
        assert isinstance(state, Quantizable)

        probe_shape = [i for i in range(len(node.tensor_meta["shape"]))]
        probe = torch.empty(probe_shape)

        args = node.args[1:]
        result = self.torch_op(probe, *args)

        new_constraint = Constraint(axis=result.shape[constraint.axis], gain=constraint.gain)

        parent_name = node._parents[0]
        if context.quantization_table[parent_name].propagate_constraint(new_constraint):
            return constraint

    def forward(self, node, context):
        
        parent_name = node._parents[0]

        if not isinstance((state := context.quantization_table[parent_name]), Quantizable):
            return Unquantized()
        
        # Since we can compute the axes backwards, we leave the callback of
        # the __parent node__ in the state.
        if not isinstance(state, Quantized):
            return Quantizable()

        axis = state.constraint.axis

        fake_shape = [1 if i == axis else 0 for i in range(len(node.tensor_meta["shape"]))]
        probe = torch.empty(fake_shape)
        args = node.args[1:]

        result = self.torch_op(probe, *args)

        return Quantizable([Constraint(axis=result.shape.index(1), gain=0)])
  
@register_wo_parameterized([
    (ViewOp, {"torch_op" : torch.Tensor.view}),
    (ReshapeOp, {"torch_op" : torch.Tensor.reshape}),
])
class ViewReshapeQuantizationMethod(QuantizationMethod):
    """

    NOTE: These ops do not have a callback. We only leave callbacks, 
    if we can _guarantee_ that any quantization requirements can be met upon
    request. Since in view and reshape ops this is violated, we conservatively
    assume that views cannot be quantized, if there is not enough information 
    in the forward pass.
    """

    torch_op: Callable[[Any], torch.Tensor] = lambda : print("Undefined field: torch_op")
    buddy_op: type = None
  
    @classmethod
    def compute_strides(cls, shape: list[int]) -> list[int]:
        """
        Compute the strides of each dimension of a tensor based on its shape

        Args:
          cls: This class
          shape (list[int]): shape to compute strides for
        
        Returns:
          list[int]: the strides of the tensor corresponding to the dimensions
        """
        strides = []
        stride = 1
        for axis_size in shape.reverse():
            strides.append(stride)
            stride *= axis_size
        
        return strides

    @classmethod
    def get_new_quantization_axis_or_none(cls, old_shape: list[int], new_shape: list[int], axis: int) -> int | None:
        """
        Check if there is an axis of the tensor with new_shape, so that
        the quantization axis of old_shape can be reused.

        Args:
          cls: This class
          old_shape (list[int]): the shape of the tensor before reshaping
          new_shape (list[int]): the shape of the tensor after reshaping
          axis (int): original quantization axis
        
        Returns:
          int | None: the axis along which new can be quantized using the
            old weights, or None if quantization is not possible.
        """
        
        old_strides = cls.compute_strides(old_shape)
        quant_stride = old_strides[axis]
        new_strides = cls.compute_strides(new_shape)

        for axis, stride in enumerate(new_strides):
            # Can only be subsumed if stride | quant_stride, or the other way around
            if (quant_stride == stride):
                return axis
          
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
    
    def forward(self, node, context):
        
        parent_name = node._parents[0]

        if not isinstance((state := context.quantization_table[parent_name]), Quantizable):
            return Unquantized()
        
        if not isinstance(state, Quantized):
            return Quantizable()
        
        parent_axis = state.constraint.axis

        parent_node = context.graph.node_table[parent_name]
        old_shape = parent_node.tensor_meta["shape"]
        new_shape = node.tensor_meta["shape"]

        axis = ViewReshapeQuantizationMethod.get_new_quantization_axis_or_none(
            old_shape=old_shape,
            new_shape=new_shape,
            axis=parent_axis)
        
        if axis is None:
            return Unquantized()
        
        return Quantizable(constraint=Constraint(axis=axis))


@register_dequantizer(quantization_method=WeightOnlyQuantization)
def dequantizer(node: Op, context: QuantizationContext) -> MulOp:
    """
    Return a dequantized version of `node`.

    Args:
        node (Op): Node to dequantize
        context (QuantizationContext): Quantization context.
    
    Returns:
        Op: dequantized node.
    """

    node_name = node.name
    dequantized_name = "dequantized_" + node_name 

    try:
        return context.graph.node_table[dequantized_name]
    except KeyError:
        pass

    scaler_name = "scaler_" + node_name

    assert scaler_name in context.graph.node_table.keys(), f"Dequantization constant should exist for {node_name}."

    scaler_node = context.graph.node_table[scaler_name]

    mul_op = MulOp()
    mul_op._name = dequantized_name
    mul_op._arguments = \
    mul_op._parents = [node_name, scaler_name]
    mul_op._tensor_meta["shape"] = node.tensor_meta["shape"]
    mul_op._tensor_meta["dtype"] = scaler_node.tensor_meta["dtype"]
    
    node._children.append(mul_op)

    return mul_op

@register_wo_quantizer(op_type=PlaceholderOp)
class PlaceholderQuantizationMethod(QuantizationMethod):

  def rewriter(self, node: Op, context: QuantizationContext):
      assert isinstance((state := context.quantization_table[node.name]), ToQuantize)

      node_name = node.name
      scaler_node = PlaceholderOp()
      scaler_node._name = "scaler_" + node_name
      
      # We apply channel(row)-wise normalization.
      # This tensor stores the column normalization constants
      node_shape = node.tensor_meta["shape"]
      quantization_state: Quantizable = context.quantization_table[node_name]
      quantization_axis = quantization_state.constraint.axis
      scaler_node._tensor_meta["shape"] = [1 if i != quantization_axis else axis_dim for i, axis_dim in enumerate(node_shape)]
      scaler_node._tensor_meta["dtype"] = node._tensor_meta["dtype"]

      node._tensor_meta["dtype"] = context.target_dtype
      context.graph.add_node(scaler_node, node_type=NodeType.FakeNode)

      context.quantization_table[node.name] = Quantized(constraint=state.constraint)
      
  def callback(self, constraint, node, context):
      return constraint
      
  def forward(self, op: Op, context: QuantizationContext) -> None | QuantizationState:
      
      input_names = [inp.name for inp in context.graph.inputs]
      if op.name in input_names:
          return Unquantized()
      
      param_names = [param.name for param in context.graph.params]
      if op.name in param_names:
          return Quantizable(constraints=[Constraint(axis=i, gain=0) for i in range(len(op.tensor_meta["shape"]))])

@register_wo_quantizer(op_type=MatmulOp)
class MatmulQuantizeMethod(QuantizationMethod):
    """
    Matmul ops are consumers, as they benefit from quantization,
    but the output is not quantizable in any situation.

    Thus, we perform the mm on the possibly quantized inputs,
    and dequantize along all quantized axes.


    TODO: mark all produced nodes in the quantization_table.
    """
    
    def rewriter(self, node: Op, context: QuantizationContext):
        node_name = node.name
        op_chain = [node]

        for i, parent_name in enumerate(node._parents, 1):
            if isinstance(context.quantization_table[parent_name], Quantized):
                
                prev_node = op_chain[-1]

                parent_scaler_name = "scaler_" + parent_name

                assert parent_scaler_name in context.graph.node_table.keys()

                mul_op = MulOp()
                mul_op_name = f"scale{i}_" + node_name
                mul_op._name = mul_op_name
                mul_op._arguments  = mul_op._parents = [parent_name, parent_scaler_name]
                mul_op.tensor_meta["shape"] = prev_node.tensor_meta["shape"]
                mul_op.tensor_meta["dtype"] = prev_node.tensor_meta["dtype"]
                if len(op_chain) > 1:
                    prev_node._children.append(mul_op_name)

                context.graph.add_node(mul_op)
                op_chain.append(mul_op)

        if len(op_chain) > 1:
            context.graph.replace_as_parent(
                parent_op=node,
                child_ops=node._children,
                new_op=op_chain[-1],
            )

            node._children = [op_chain[1].name]

    def forward(self, node: Op, context: QuantizationContext) -> None | QuantizationState:
        quantizable = False
        
        parent_op1_name, parent_op2_name = node._parents

        op1_axis = max(0, len(context.graph.node_table[parent_op1_name].tensor_meta["shape"]) - 2)
        if context.quantization_table[node._parents[0]].propagate_constraint(Constraint(axis=op1_axis)):
            quantizable = True

        op2_axis = max(0, len(context.graph.node_table[parent_op2_name].tensor_meta["shape"]) - 1)
        if context.quantization_table[node._parents[1]].propagate_constraint(Constraint(axis=op2_axis)):
            quantizable = True

        if quantizable:
            return Consumer()

        return Unquantized()

# AddMMOp, BatchMatmulOp, BaddbmmOp