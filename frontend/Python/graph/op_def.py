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

    @classmethod
    def fx_create_node(
        cls,
        node_name: str,
        node_input: torch.fx.Node | str,
        node_users: str,
        node_output_shape: torch.Size,
        node_output_dtype: torch.dtype,
    ):
        """
        Create an op node.
        Args:
            node_name: The unique name of op node.
            node_input: The op node's input.
            node_users: The op node's successor nodes.
            node_output_shape: The op node's output tensor shape.
            node_output_dtype: The op node's output tensor dtype.
        """
        buddy_node = cls()
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
        super().__init__()
        self._op_type = OpType.PlaceholderType


class MatmulOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class GetItemOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.GetItemType
        self._lower_strategy = []


class OutputOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.GetItemType

    @classmethod
    def fx_create_node(
        cls,
        node_name: str,
        node_input: torch.fx.Node | str,
    ):
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
        super().__init__()
        self._op_type = OpType.PlaceholderType


class UnsqueezeOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class ViewOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class EmbeddingOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class OnesOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.PlaceholderType


class FullOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.PlaceholderType


class LessthanOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.BroadcastType


class MaskedFillOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class SliceOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class ExpandOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class ToCopyOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class RsubOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.BroadcastType


class PowOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.BroadcastType


class MeanOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class RsqrtOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class MulOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.BroadcastType


class TransposeOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class TransposeSpecificDimOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class IndexOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class NegOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class CatOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ConcatType


class SqueezeOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class BatchMatmulOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class DivOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.BroadcastType


class SoftmaxOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class CloneOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class SiluOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class AddOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.BroadcastType
