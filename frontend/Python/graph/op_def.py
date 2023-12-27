from enum import Enum
import torch


class OpType(Enum):
    """
    Enum class for declare op's type.
    """

    BroadcastType = 0
    ElementwiseType = 1
    ReshapeType = 2
    ReduceType = 3
    ConcatType = 4
    PlaceholderType = 5
    GetItemType = 6


class Op:
    """
    Base class for all ops.
    Attributes:
        _name: The unique name of op node.
        _arguments: The op node's input.
        _children: The op node's successor nodes.
        _parent: The op node's predecessor nodes.
        _tensor_meta: The op node's output tensor shape and dtype.
        _op_type: The op node's type in class OpType.
        _device: The device for the op node to run.
    """

    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = None
        self._device = "cpu"

    def add_parent(self, parent):
        self._parent.append(parent)

    def add_argument(self, arg):
        self._arguments.append(arg)

    def add_children(self, child):
        self._children.append(child)

    def set_device(self, device):
        self._device = device

    @property
    def args(self):
        return self._arguments

    @property
    def name(self):
        return self._name

    @property
    def tensor_meta(self):
        return self._tensor_meta


class PlaceholderOp(Op):
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = OpType.PlaceholderType
        self._device = "cpu"

    @staticmethod
    def fx_create_node(
        node_name, node_input, node_users, node_output_shape, node_output_dtype
    ):
        """
        Create placeholder node.
        Args:
            node_name: The unique name of op node.
            node_input: Placeholder node should have only one input.
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = PlaceholderOp()
        buddy_node._name = node_name
        for input_arg in node_input:
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
                buddy_node.add_parent(str(input_arg))
            else:
                buddy_node.add_argument(input_arg)
        for child in node_users:
            buddy_node.add_children(child)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node


class MatmulOp(Op):
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = OpType.ReduceType
        self._device = "cpu"

    @staticmethod
    def fx_create_node(
        node_name,
        node_input,
        node_users,
        node_output_shape,
        node_output_dtype,
    ):
        """
        Create matmul node.
        Args:
            node_name: The unique name of op node.
            node_input: Matmul node should have only two input. A and B in
            formula: C = A · B
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = MatmulOp()
        buddy_node._name = node_name
        for input_arg in node_input:
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
                buddy_node.add_parent(str(input_arg))
            else:
                buddy_node.add_argument(input_arg)
        for child in node_users:
            buddy_node.add_children(child)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node


class GetItemOp(Op):
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = OpType.GetItemType
        self._lower_strategy = []
        self._device = "cpu"

    @staticmethod
    def fx_create_node(
        node_name, node_input, node_users, node_output_shape, node_output_dtype
    ):
        """
        Create getitem node.
        Args:
            node_name: The unique name of op node.
            node_input: GetItem node should have only two input, input node and
            select index.
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = GetItemOp()
        buddy_node._name = node_name
        for input_arg in node_input:
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
                buddy_node.add_parent(str(input_arg))
            else:
                buddy_node.add_argument(input_arg)
        for child in node_users:
            buddy_node.add_children(child)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node


class OutputOp(Op):
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = OpType.GetItemType
        self._lower_strategy = []
        self._device = "cpu"

    @staticmethod
    def fx_create_node(node_name, node_input):
        """
        Create output node.
        Args:
            node_name: The unique name of op node.
            node_input: Output nodes.
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = OutputOp()
        buddy_node._name = node_name
        for input_arg in node_input[0]:
            # only support fx node as output
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
                buddy_node.add_parent(str(input_arg))
        return buddy_node


class ArangeOp(Op):
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = OpType.PlaceholderType
        self._device = "cpu"

    @staticmethod
    def fx_create_node(
        node_name,
        node_input,
        node_users,
        node_output_shape,
        node_output_dtype,
    ):
        """
        Create matmul node.
        Args:
            node_name: The unique name of op node.
            node_input: Matmul node should have only two input. A and B in
            formula: C = A · B
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = ArangeOp()
        buddy_node._name = node_name
        for input_arg in node_input:
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
                buddy_node.add_parent(str(input_arg))
            else:
                buddy_node.add_argument(input_arg)
        for child in node_users:
            buddy_node.add_children(child)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node

class UnsqueezeOp(Op):
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = OpType.ReshapeType
        self._device = "cpu"

    @staticmethod
    def fx_create_node(
        node_name,
        node_input,
        node_users,
        node_output_shape,
        node_output_dtype,
    ):
        """
        Create unsqueeze node.
        Args:
            node_name: The unique name of op node.
            node_input: Unsqueeze node should have only two input, input node
            and axis for unsqueeze.
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = UnsqueezeOp()
        buddy_node._name = node_name
        for input_arg in node_input:
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
                buddy_node.add_parent(str(input_arg))
            else:
                buddy_node.add_argument(input_arg)
        for child in node_users:
            buddy_node.add_children(child)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node

class ViewOp(Op):
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = OpType.ReshapeType
        self._device = "cpu"

    @staticmethod
    def fx_create_node(
        node_name,
        node_input,
        node_users,
        node_output_shape,
        node_output_dtype,
    ):
        """
        Create view node.
        Args:
            node_name: The unique name of op node.
            node_input: View node should have only two input. A and B in
            formula: C = A · B
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = ViewOp()
        buddy_node._name = node_name
        for input_arg in node_input:
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
                buddy_node.add_parent(str(input_arg))
            else:
                buddy_node.add_argument(input_arg)
        for child in node_users:
            buddy_node.add_children(child)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node

class EmbeddingOp(Op):
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = OpType.ReshapeType
        self._device = "cpu"

    @staticmethod
    def fx_create_node(
        node_name,
        node_input,
        node_users,
        node_output_shape,
        node_output_dtype,
    ):
        """
        Create embedding node.
        Args:
            node_name: The unique name of op node.
            node_input: Embedding node should have only two input node, A and B.
            A should be index tensor. B should be word vector matrix.
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = EmbeddingOp()
        buddy_node._name = node_name
        for input_arg in node_input:
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
                buddy_node.add_parent(str(input_arg))
            else:
                buddy_node.add_argument(input_arg)
        for child in node_users:
            buddy_node.add_children(child)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node

class OnesOp(Op):
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = OpType.PlaceholderType
        self._device = "cpu"

    @staticmethod
    def fx_create_node(
        node_name,
        node_input,
        node_users,
        node_output_shape,
        node_output_dtype,
    ):
        """
        Create ones node.
        Args:
            node_name: The unique name of op node.
            node_input: Ones node should have only one input to express output 
            shape.
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = OnesOp()
        buddy_node._name = node_name
        for input_arg in node_input:
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
                buddy_node.add_parent(str(input_arg))
            else:
                buddy_node.add_argument(input_arg)
        for child in node_users:
            buddy_node.add_children(child)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node

class FullOp(Op):
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = OpType.PlaceholderType
        self._device = "cpu"

    @staticmethod
    def fx_create_node(
        node_name,
        node_input,
        node_users,
        node_output_shape,
        node_output_dtype,
    ):
        """
        Create full node.
        Args:
            node_name: The unique name of op node.
            node_input: Full node should have two input to express output shape
            and element value.
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = FullOp()
        buddy_node._name = node_name
        for input_arg in node_input:
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
                buddy_node.add_parent(str(input_arg))
            else:
                buddy_node.add_argument(input_arg)
        for child in node_users:
            buddy_node.add_children(child)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node

class LessthanOp(Op):
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = OpType.BroadcastType
        self._device = "cpu"

    @staticmethod
    def fx_create_node(
        node_name,
        node_input,
        node_users,
        node_output_shape,
        node_output_dtype,
    ):
        """
        Create lessthan node.
        Args:
            node_name: The unique name of op node.
            node_input: Lessthan node should have two input node to compare.
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = LessthanOp()
        buddy_node._name = node_name
        for input_arg in node_input:
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
                buddy_node.add_parent(str(input_arg))
            else:
                buddy_node.add_argument(input_arg)
        for child in node_users:
            buddy_node.add_children(child)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node

class MaskedFillOp(Op):
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = OpType.ElementwiseType
        self._device = "cpu"

    @staticmethod
    def fx_create_node(
        node_name,
        node_input,
        node_users,
        node_output_shape,
        node_output_dtype,
    ):
        """
        Create maskedfill node.
        Args:
            node_name: The unique name of op node.
            node_input: Maskedfill node should have Three input, two input node,
            the first is masked node, the second is bool node, and one value for
            fill.
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = MaskedFillOp()
        buddy_node._name = node_name
        for input_arg in node_input:
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
                buddy_node.add_parent(str(input_arg))
            else:
                buddy_node.add_argument(input_arg)
        for child in node_users:
            buddy_node.add_children(child)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node

class SliceOp(Op):
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = OpType.ReshapeType
        self._device = "cpu"

    @staticmethod
    def fx_create_node(
        node_name,
        node_input,
        node_users,
        node_output_shape,
        node_output_dtype,
    ):
        """
        Create slice node.
        Args:
            node_name: The unique name of op node.
            node_input: Slice node should have five input, the first is input
            node, the second is dim for slice, the third is slice start
            position, the forth is slice end position, the fifth is slice
            stride.
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = SliceOp()
        buddy_node._name = node_name
        for input_arg in node_input:
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
                buddy_node.add_parent(str(input_arg))
            else:
                buddy_node.add_argument(input_arg)
        for child in node_users:
            buddy_node.add_children(child)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node

class ExpandOp(Op):
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = OpType.ReshapeType
        self._device = "cpu"

    @staticmethod
    def fx_create_node(
        node_name,
        node_input,
        node_users,
        node_output_shape,
        node_output_dtype,
    ):
        """
        Create expand node.
        Args:
            node_name: The unique name of op node.
            node_input: Expand node should have two input, the first is input
            node, the second is output shape.
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = ExpandOp()
        buddy_node._name = node_name
        for input_arg in node_input:
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
                buddy_node.add_parent(str(input_arg))
            else:
                buddy_node.add_argument(input_arg)
        for child in node_users:
            buddy_node.add_children(child)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node

class ToCopyOp(Op):
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = OpType.ElementwiseType
        self._device = "cpu"

    @staticmethod
    def fx_create_node(
        node_name,
        node_input,
        node_users,
        node_output_shape,
        node_output_dtype,
    ):
        """
        Create tocopy node.
        Args:
            node_name: The unique name of op node.
            node_input: ToCopy node should have one input node to copy.
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = ToCopyOp()
        buddy_node._name = node_name
        for input_arg in node_input:
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
                buddy_node.add_parent(str(input_arg))
            else:
                buddy_node.add_argument(input_arg)
        for child in node_users:
            buddy_node.add_children(child)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node

class RsubOp(Op):
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = OpType.BroadcastType
        self._device = "cpu"

    @staticmethod
    def fx_create_node(
        node_name,
        node_input,
        node_users,
        node_output_shape,
        node_output_dtype,
    ):
        """
        Create rsub node.
        Args:
            node_name: The unique name of op node.
            node_input: Rsub node should have two input, the second sub the
            first.
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = RsubOp()
        buddy_node._name = node_name
        for input_arg in node_input:
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
                buddy_node.add_parent(str(input_arg))
            else:
                buddy_node.add_argument(input_arg)
        for child in node_users:
            buddy_node.add_children(child)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node

class PowOp(Op):
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = OpType.BroadcastType
        self._device = "cpu"

    @staticmethod
    def fx_create_node(
        node_name,
        node_input,
        node_users,
        node_output_shape,
        node_output_dtype,
    ):
        """
        Create pow node.
        Args:
            node_name: The unique name of op node.
            node_input: Pow node should have two input, the first pow the
            second.
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = PowOp()
        buddy_node._name = node_name
        for input_arg in node_input:
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
                buddy_node.add_parent(str(input_arg))
            else:
                buddy_node.add_argument(input_arg)
        for child in node_users:
            buddy_node.add_children(child)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node

class MeanOp(Op):
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = OpType.ReduceType
        self._device = "cpu"

    @staticmethod
    def fx_create_node(
        node_name,
        node_input,
        node_users,
        node_output_shape,
        node_output_dtype,
    ):
        """
        Create mean node.
        Args:
            node_name: The unique name of op node.
            node_input: Mean node should have three input, the first is input
            node, the second is the dim for compute mean, the third is a bool
            value determine if keep dim as origin.
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = MeanOp()
        buddy_node._name = node_name
        for input_arg in node_input:
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
                buddy_node.add_parent(str(input_arg))
            else:
                buddy_node.add_argument(input_arg)
        for child in node_users:
            buddy_node.add_children(child)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node

class RsqrtOp(Op):
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = OpType.ElementwiseType
        self._device = "cpu"

    @staticmethod
    def fx_create_node(
        node_name,
        node_input,
        node_users,
        node_output_shape,
        node_output_dtype,
    ):
        """
        Create rsqrt node.
        Args:
            node_name: The unique name of op node.
            node_input: Rsqrt node should have one input node to compute rsqrt.
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = RsqrtOp()
        buddy_node._name = node_name
        for input_arg in node_input:
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
                buddy_node.add_parent(str(input_arg))
            else:
                buddy_node.add_argument(input_arg)
        for child in node_users:
            buddy_node.add_children(child)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node

class MulOp(Op):
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = OpType.BroadcastType
        self._device = "cpu"

    @staticmethod
    def fx_create_node(
        node_name,
        node_input,
        node_users,
        node_output_shape,
        node_output_dtype,
    ):
        """
        Create mul node.
        Args:
            node_name: The unique name of op node.
            node_input: Mul node should have two input node to compute
            elementwise mul.
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = MulOp()
        buddy_node._name = node_name
        for input_arg in node_input:
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
                buddy_node.add_parent(str(input_arg))
            else:
                buddy_node.add_argument(input_arg)
        for child in node_users:
            buddy_node.add_children(child)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node

class TransposeOp(Op):
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = OpType.ReshapeType
        self._device = "cpu"

    @staticmethod
    def fx_create_node(
        node_name,
        node_input,
        node_users,
        node_output_shape,
        node_output_dtype,
    ):
        """
        Create transpose node.
        Args:
            node_name: The unique name of op node.
            node_input: Transpose node should have one input node to transpose.
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = TransposeOp()
        buddy_node._name = node_name
        for input_arg in node_input:
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
                buddy_node.add_parent(str(input_arg))
            else:
                buddy_node.add_argument(input_arg)
        for child in node_users:
            buddy_node.add_children(child)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node

class TransposeSpecificDimOp(Op):
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = OpType.ReshapeType
        self._device = "cpu"

    @staticmethod
    def fx_create_node(
        node_name,
        node_input,
        node_users,
        node_output_shape,
        node_output_dtype,
    ):
        """
        Create transpose specific dim node.
        Args:
            node_name: The unique name of op node.
            node_input: TransposeSpecificDim node should have three input, the
            first is input node, the second and the third is specific dims to
            transpose.
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = TransposeSpecificDimOp()
        buddy_node._name = node_name
        for input_arg in node_input:
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
                buddy_node.add_parent(str(input_arg))
            else:
                buddy_node.add_argument(input_arg)
        for child in node_users:
            buddy_node.add_children(child)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node

class IndexOp(Op):
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = OpType.ReshapeType
        self._device = "cpu"

    @staticmethod
    def fx_create_node(
        node_name,
        node_input,
        node_users,
        node_output_shape,
        node_output_dtype,
    ):
        """
        Create index node.
        Args:
            node_name: The unique name of op node.
            node_input: Index node should have two input, the input node and
            index node list, such as node_input(arg0_node, [arg1_node]).
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = IndexOp()
        buddy_node._name = node_name
        for input_arg in node_input:
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
                buddy_node.add_parent(str(input_arg))
            else:
                buddy_node.add_argument(input_arg)
        for child in node_users:
            buddy_node.add_children(child)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node

class NegOp(Op):
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = OpType.ElementwiseType
        self._device = "cpu"

    @staticmethod
    def fx_create_node(
        node_name,
        node_input,
        node_users,
        node_output_shape,
        node_output_dtype,
    ):
        """
        Create neg node.
        Args:
            node_name: The unique name of op node.
            node_input: Neg node should have one input node to compute neg.
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = NegOp()
        buddy_node._name = node_name
        for input_arg in node_input:
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
                buddy_node.add_parent(str(input_arg))
            else:
                buddy_node.add_argument(input_arg)
        for child in node_users:
            buddy_node.add_children(child)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node

class CatOp(Op):
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = OpType.ConcatType
        self._device = "cpu"

    @staticmethod
    def fx_create_node(
        node_name,
        node_input,
        node_users,
        node_output_shape,
        node_output_dtype,
    ):
        """
        Create cat node.
        Args:
            node_name: The unique name of op node.
            node_input: Cat node should have two input, the first is a list of
            nodes to concat, the second is the dim for concat.
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = CatOp()
        buddy_node._name = node_name
        for input_arg in node_input:
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
                buddy_node.add_parent(str(input_arg))
            else:
                buddy_node.add_argument(input_arg)
        for child in node_users:
            buddy_node.add_children(child)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node

class SqueezeOp(Op):
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = OpType.ReshapeType
        self._device = "cpu"

    @staticmethod
    def fx_create_node(
        node_name,
        node_input,
        node_users,
        node_output_shape,
        node_output_dtype,
    ):
        """
        Create suqeeze node.
        Args:
            node_name: The unique name of op node.
            node_input: Squeeze node should have two input, the first is the
            input node, the second is the dim for squeeze.
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = SqueezeOp()
        buddy_node._name = node_name
        for input_arg in node_input:
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
                buddy_node.add_parent(str(input_arg))
            else:
                buddy_node.add_argument(input_arg)
        for child in node_users:
            buddy_node.add_children(child)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node

class BatchMatmulOp(Op):
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = OpType.ReduceType
        self._device = "cpu"

    @staticmethod
    def fx_create_node(
        node_name,
        node_input,
        node_users,
        node_output_shape,
        node_output_dtype,
    ):
        """
        Create batchmatmul node.
        Args:
            node_name: The unique name of op node.
            node_input: BatchMatmul node should have two input to compute
            batchmatmul.
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = BatchMatmulOp()
        buddy_node._name = node_name
        for input_arg in node_input:
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
                buddy_node.add_parent(str(input_arg))
            else:
                buddy_node.add_argument(input_arg)
        for child in node_users:
            buddy_node.add_children(child)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node

class DivOp(Op):
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = OpType.BroadcastType
        self._device = "cpu"

    @staticmethod
    def fx_create_node(
        node_name,
        node_input,
        node_users,
        node_output_shape,
        node_output_dtype,
    ):
        """
        Create div node.
        Args:
            node_name: The unique name of op node.
            node_input: Div node should have two input to compute div.
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = DivOp()
        buddy_node._name = node_name
        for input_arg in node_input:
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
                buddy_node.add_parent(str(input_arg))
            else:
                buddy_node.add_argument(input_arg)
        for child in node_users:
            buddy_node.add_children(child)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node

class SoftmaxOp(Op):
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = OpType.ReduceType
        self._device = "cpu"

    @staticmethod
    def fx_create_node(
        node_name,
        node_input,
        node_users,
        node_output_shape,
        node_output_dtype,
    ):
        """
        Create softmax node.
        Args:
            node_name: The unique name of op node.
            node_input: Softmax node should have three input, the first is input
            node, the second is dim to compute, the third is a bool value to
            determine if half to float.
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = SoftmaxOp()
        buddy_node._name = node_name
        for input_arg in node_input:
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
                buddy_node.add_parent(str(input_arg))
            else:
                buddy_node.add_argument(input_arg)
        for child in node_users:
            buddy_node.add_children(child)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node

class CloneOp(Op):
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = OpType.ReduceType
        self._device = "cpu"

    @staticmethod
    def fx_create_node(
        node_name,
        node_input,
        node_users,
        node_output_shape,
        node_output_dtype,
    ):
        """
        Create clone node.
        Args:
            node_name: The unique name of op node.
            node_input: Clone node should have one input node to clone.
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = CloneOp()
        buddy_node._name = node_name
        for input_arg in node_input:
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
                buddy_node.add_parent(str(input_arg))
            else:
                buddy_node.add_argument(input_arg)
        for child in node_users:
            buddy_node.add_children(child)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node

class SiluOp(Op):
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._children = []
        self._parent = []
        self._tensor_meta = {}
        self._op_type = OpType.ElementwiseType
        self._device = "cpu"

    @staticmethod
    def fx_create_node(
        node_name,
        node_input,
        node_users,
        node_output_shape,
        node_output_dtype,
    ):
        """
        Create silu node.
        Args:
            node_name: The unique name of op node.
            node_input: Silu node should have one input node to compute silu
            activation.
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = SiluOp()
        buddy_node._name = node_name
        for input_arg in node_input:
            if isinstance(input_arg, torch.fx.Node):
                buddy_node.add_argument(str(input_arg))
                buddy_node.add_parent(str(input_arg))
            else:
                buddy_node.add_argument(input_arg)
        for child in node_users:
            buddy_node.add_children(child)
        buddy_node._tensor_meta["shape"] = node_output_shape
        buddy_node._tensor_meta["dtype"] = node_output_dtype
        return buddy_node