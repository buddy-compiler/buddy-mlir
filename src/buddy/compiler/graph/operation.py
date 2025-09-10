# ===- operation.py ------------------------------------------------------------
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
# This is the operation structure of Buddy Compiler graph representation.
#
# ===---------------------------------------------------------------------------

from enum import Enum
from typing import Dict, Optional, List, Tuple

from .type import TensorDType, TensorMeta


class OpType(Enum):
    """
    Enum class for declaring operation types.

    Members:
    - BroadcastType: int
        Represents a broadcast operation.
    - ElementwiseType: int
        Represents an elementwise operation.
    - ReshapeType: int
        Represents a reshape operation.
    - ReduceType: int
        Represents a reduction operation.
    - ConcatType: int
        Represents a concatenation operation.
    - PlaceholderType: int
        Represents a placeholder operation.
    - GetItemType: int
        Represents an operation to retrieve an item.

    Note: The underlying values are integers for these operation types.
    """

    BroadcastType = 0
    ElementwiseType = 1
    ReshapeType = 2
    SliceLikeType = 3
    ReduceType = 4
    ConcatType = 5
    PlaceholderType = 6
    GetItemType = 7
    Unfusable = 8


class Op:
    """
    Base class for all operations in a computational graph.

    Attributes:
    - _name: str
        The unique name of the operation node.
    - _arguments: list
        The input arguments of the operation node.
    - _keyword_arguments: dict
        The keyword arguments of the operation node.
    - _tensor_meta: dict
        The metadata of the output tensor, including shape and data type.
    - _op_type: OpType
        The type of the operation node, as defined in the OpType enum.
    """

    def __init__(self) -> None:
        """
        Initialize a new instance of the Op class.
        """
        self._name = None
        self._arguments = []
        self._keyword_arguments = {}
        self._tensor_meta: Dict = {}
        self._op_type: OpType = None
        self._children: List[str] = []
        self._parents: List[str] = []
        self._args_index = []

    def add_argument(self, arg, arg_index=0):
        """
        Add an input argument to the operation node.

        Parameters:
        - arg: Any
            The input argument to be added.
        """
        self._arguments.append(arg)
        self._args_index.append(arg_index)

    def add_parent(self, parent: str):
        """
        Add an parent node's name to the operation node.

        Parameters:
        - parent: str
            The parent node's name to be added.
        """
        self._parents.append(parent)

    def add_children(self, child):
        """
        Add an user node's name to the operation node.

        Parameters:
        - user: str
            The user node's name to be added.
        """
        self._children.append(child)

    @property
    def args(self):
        return self._arguments

    @property
    def kwargs(self):
        return self._keyword_arguments

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        self._name = new_name

    @property
    def tensor_meta(self):
        return self._tensor_meta

    @tensor_meta.setter
    def tensor_meta(self, new_tensor_meta):
        self._tensor_meta = new_tensor_meta


class PlaceholderOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.PlaceholderType


class MatmulOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class TransposeMatmulFusedOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class GetItemOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.GetItemType


class OutputOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.GetItemType


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


class LessThanOp(Op):
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


class AddMMOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class AmaxOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class SubOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.BroadcastType


class ConvertElementTypeOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class ExpOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class ExpandOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class PermuteOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class ReshapeOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class SelectOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class SumDimOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class TanhOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class VarMeanOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class TOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class ErfOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class Conv2dOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType
        self._layout = "NCHW_FCHW"


class ReluOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class SigmoidOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class IotaOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.PlaceholderType


class ScalarTensorOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.PlaceholderType


class WhereOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class MaxPool2dWithIndicesOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType
        self._layout = "NCHW"


class MaxPool2dOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType
        self._layout = "NCHW"


class CallOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self.call_func_name = ""
        self._op_type = OpType.Unfusable


class FuncOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.Unfusable


class ReciprocalOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class SqrtOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class ClampMinOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class ClampMaxOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class RandIntLowOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.PlaceholderType


class CosOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class SinOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class ArgMaxOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class SplitOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class MaxOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class GtOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class ScaledDotProductFlashAttentionForCpuOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class GeOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class GreaterThanOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.BroadcastType


class UnsafeIndexOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class EqualOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class SliceScatterOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class CopyOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType

