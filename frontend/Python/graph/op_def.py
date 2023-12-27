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
    PlaceholderType = 4
    GetItemType = 5


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

