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
  Consumer,
  QuantizationMethod
)
from ... import Graph, NodeType
from ...operation import *

import torch
from typing import Any, Callable

class ChannelWiseQuantizationConstraint(QuantizationConstraint):
    axis: int

    def __init__(self, axis: int):
        self.axis = axis

    def check(self, other: "ChannelWiseQuantizationConstraint") -> bool:
        return other.axis == self.axis
    
Constraint = ChannelWiseQuantizationConstraint

class WeightOnlyQuantization(Quantization):
    pass

def register_wo_quantizer(op_type: OpType):
    return register_quantizer(quantization=WeightOnlyQuantization, op_type=op_type)

def register_wo_parameterized(params: list[tuple[OpType, Dict[str, Any]]]):
    register_parameterized(quantization=WeightOnlyQuantization, params=params)

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

        probe_shape = [i for i in range(len(node.tensor_meta["shape"]))]
        probe = torch.empty(probe_shape)

        args = node.args[1:]
        result = self.torch_op(probe, *args)

        state.check_constraint(constraint=Constraint(axis=result.shape[constraint.axis]))
    
    def forward(self, node, context):
        
        parent_name = node._parents[0]

        if not isinstance((state := context.quantization_table[parent_name]), Quantizable):
            return Unquantized()
        
        if (constr := state.constraint) is None:
            return Quantized()

        axis = constr.axis

        fake_shape = [1 if i == axis else 0 for i in range(len(node.tensor_meta["shape"]))]
        probe = torch.empty(fake_shape)
        args = node.args[1:]

        result = self.torch_op(probe, *args)

        Quantizable(Constraint(axis=torch.argmax(result.shape.index(1)))) 
  
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
        
        if (constr := state.constraint) is None:
            return Unquantized()
        
        constr: Constraint

        parent_axis = constr.axis

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

  def forward(self, op: Op, context: QuantizationContext) -> None | QuantizationState:
      
      input_names = [inp.name for inp in context.graph.inputs]
      if op.name in input_names:
          return Unquantized()
      
      param_names = [param.name for param in context.graph.params]
      if op.name in param_names:
          return Quantizable()

@register_wo_quantizer(op_type=PermuteOp)
class PermuteQuantizationMethod(QuantizationMethod):

  def rewriter(self, node: Op, context: QuantizationContext) -> None:
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

  def callback(self, constraint: Constraint, node: Op, context: QuantizationContext) -> None:
      node_name = node.name
      permute_dims = node.args[1]

      state: QuantizationState = context.quantization_table[node_name]

      assert isinstance(state, Quantizable)

      state.set_constraint(Constraint(axis=permute_dims[constraint.axis]))

  def forward(self, node: Op, context: QuantizationContext) -> None | QuantizationState:
      """
      Quantizing the permute op of a quantized tensor.

      We track where the original quantized axis ends up.
      """
      
      parent_name = node._parents[0]
      permute_dims: list[int] = node.args[1]

      if not isinstance(context.quantization_table[parent_name], Quantizable):
          return Unquantized()

      # If the previous node had its quantization axis set, we can define quantization by axis.
      if (axis := context.quantization_table[parent_name].axis):
          return Quantizable(constraint=Constraint(axis=permute_dims.index(axis)))
      # If not, we define quantization with callback, so that if an axis is determined, we can propagate that back
      # FIXME: this is incorrect:
      return Quantizable()


@register_wo_quantizer(op_type=MatmulOp)
class MatmulQuantizeMethod(QuantizationMethod):
    
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

        context.graph.replace_as_parent(
            parent_op=node,
            child_ops=node._children,
            new_op=op_chain[-1],
        )

        node._children = [op_chain[1].name]

    def forward(self, node: Op, context: QuantizationContext) -> None | QuantizationState:
        quantizable = False

        if isinstance((state := context.quantization_table[node._parents[0]]), Quantizable):
            if state.check_constraint(Constraint(axis=0)):
                quantizable = True

        if isinstance((state := context.quantization_table[node._parents[1]]), Quantizable):
            if state.check_constraint(Constraint(axis=1)):
                quantizable = True

        if quantizable:
            return Consumer()

        return Unquantized()

@register_wo_quantizer(op_type=AddMMOp)
class AddMMQuantizationMethod(QuantizationMethod):

  def rewriter(self, node: Op, context: QuantizationContext) -> None:
      node_name = node.name

      mm_op = MatmulOp()
      mm_op_name = "pre_scaled_" + node_name
      mm_op._name = mm_op_name
      mm_op._parents = node._parents[1:]
      mm_op._arguments = node._parents[1:]
      mm_op._tensor_meta = node.tensor_meta

      context.graph.replace_as_child(node._parents[1:], node, mm_op)

      context.graph.add_node(mm_op)

      op_chain = [mm_op]

      for i, parent_name in enumerate(mm_op._parents, 1):
          if isinstance(context.quantization_table[parent_name], Quantized):
              
              prev_node = op_chain[-1]

              parent_scaler_name = "scaler_" + parent_name

              assert parent_scaler_name in context.graph.node_table.keys(), f"Scaler {parent_scaler_name} should exist."

              mul_op = MulOp()
              mul_op_name = f"scale{i}_" + node_name
              mul_op._name = mul_op_name
              mul_op._arguments = \
              mul_op._parents = [parent_name, parent_scaler_name]
              mul_op.tensor_meta["shape"] = prev_node.tensor_meta["shape"]
              mul_op.tensor_meta["dtype"] = prev_node.tensor_meta["dtype"]
              if len(op_chain) > 1:
                  prev_node._children.append(mul_op_name)

              context.graph.add_node(mul_op)
              op_chain.append(mul_op)

      # TODO: this should check the quantization state of the bias.

      add_op = AddOp()
      add_op_name = "biased_" + node_name
      add_op._name = add_op_name
      add_op._arguments = \
      add_op._parents = [op_chain[-1].name, node._parents[0]]
      add_op._tensor_meta = node.tensor_meta

      context.graph.replace_as_child([node._parents[0]], node, add_op)

      add_op._children = node._children
      op_chain[-1]._children = [add_op_name]

      context.graph.replace_as_parent(node, node._children, add_op)

      context.graph.add_node(add_op)

      node._parents = []
      node._children = []
      context.graph.delete_node(node, [])

  def forward(self, node: Op, context: QuantizationContext) -> None | QuantizationState:
      """
      Convert the dag
      (addmm arg1:u, arg2:u, arg3:q)
      to
      (add (mul (mm arg2, arg3), scalar_arg3), arg1)
      """

      quantizable = False

      if isinstance((state := context.quantization_table[node._parents[1]]), Quantizable):
          if state.check_constraint(Constraint(axis=0)):
              quantizable = True

      if isinstance((state := context.quantization_table[node._parents[2]]), Quantizable):
          if state.check_constraint(Constraint(axis=1)):
              quantizable = True

      if quantizable:
          return Consumer()

      return None

@register_wo_quantizer(op_type=BatchMatmulOp)
class BatchMatmulQunatizationMethod(QuantizationMethod):

    def rewrite(self, node: Op, context: QuantizationContext):
      MatmulQuantizeMethod.rewriter(node, context)

    def forward(self, node: Op, context: QuantizationContext) -> None | QuantizationState:

      quantizable = False

      if isinstance((state := context.quantization_table[node._parents[0]]), Quantizable):
          if state.check_constraint(Constraint(axis=1)):
              quantizable = True

      if isinstance((state := context.quantization_table[node._parents[1]]), Quantizable):
          if state.check_constraint(Constraint(axis=2)):
              quantizable = True

      if quantizable:
          return Consumer()

      return Unquantized()

        
@register_wo_quantizer(op_type=BaddbmmOp)
class BatchAddMMQuantizationMethod(QuantizationMethod):
    
    def rewriter(self, node: Op, context: QuantizationContext):

        node_name = node.name

        batchmatmul_op = BatchMatmulOp()
        batchmatmul_op_name = "pre_scaled_" + node_name

        batchmatmul_op._name = batchmatmul_op_name
        batchmatmul_op._parents = node._parents[1:]
        batchmatmul_op._tensor_meta = node.tensor_meta

        context.graph.replace_as_child(node._parents[1:], node, batchmatmul_op)

        op_chain = [batchmatmul_op]

        for i, parent_name in enumerate(batchmatmul_op._parents, 1):
            if isinstance(context.quantization_table[parent_name], Quantized):

                prev_op = op_chain[-1]
                prev_op_name = prev_op.name

                mul_op = MulOp()
                mul_op._name = f"scale{i}_" + node_name

                scaler_name = "scaler_" + parent_name
                assert scaler_name in context.graph.node_table.keys(), f"Scaler node `{scaler_name}` should exist."
                scaler_node = context.graph.node_table[scaler_name]
                mul_op._tensor_meta["shape"] = node.tensor_meta["shape"]
                mul_op._tensor_meta["dtype"] = scaler_node["dtype"]

                mul_op._parents = \
                mul_op._arguments = [prev_op_name, scaler_name]

                if len(op_chain) > 0:
                    prev_op._children = [mul_op]

                op_chain.append(mul_op)
                context.graph.add_node(mul_op)

        add_op = AddOp()
        add_op._name = "biased_" + node_name

        bias_node_name = node._parents[0]
        bias_node = context.graph.node_table[bias_node_name]

        if isinstance(context.quantization_table[bias_node_name], Quantized):
          bias_node = dequantizer(bias_node)
          bias_node_name = bias_node.name

        add_op._parents = [op_chain[-1].name, bias_node_name]
        add_op.tensor_meta["shape"] = node.tensor_meta["shape"]
        add_op.tensor_meta["dtype"] = context.graph.node_table[bias_node_name].tensor_meta["dtype"]

        context.graph.add_node(add_op)

        context.graph.replace_as_child(node._parents[0], node, add_op)
        context.graph.replace_as_parent(node._children, node, add_op)
            
    def forward(self, node: Op, context: QuantizationContext):
    
        quantizable = False

        if isinstance((state := context.quantization_table[node._parents[1]]), Quantizable):
            if state.check_constraint(Constraint(axis=1)):
                quantizable = True

        if isinstance((state := context.quantization_table[node._parents[2]]), Quantizable):
            if state.check_constraint(Constraint(axis=2)):
                quantizable = True

        if quantizable:
            return Consumer()

        return Unquantized()
