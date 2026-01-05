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


class ArangeStartStepOp(Op):
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


class IndexSelectOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class CatOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ConcatType


class BatchMatmulOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class DivOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.BroadcastType


class DivTensorModeOp(Op):
    """
    Division operation with rounding mode.
    Implements aten.div.Tensor_mode: x / y with specified rounding mode.
    rounding_mode can be 'floor', 'trunc', or None.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.BroadcastType


class SoftmaxOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class LogSoftmaxOp(Op):
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


class ErfinvOp(Op):
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


class GluOp(Op):
    """
    Gated linear unit operation.
    Implements aten.glu.default: Splits tensor and applies sigmoid gate.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class GruOp(Op):
    """
    Gated recurrent unit operation.
    Implements aten.gru.input.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.Unfusable


class IotaOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.PlaceholderType


class ScalarTensorOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.PlaceholderType


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


class MaxPool1dOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType
        self._layout = "NCW"


class MaxPool3dOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType
        self._layout = "NCDHW"


class AvgPool3dOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType
        self._layout = "NCDHW"


class AdaptiveMaxPool1dOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType
        self._layout = "NCW"


class AdaptiveMaxPool2dOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType
        self._layout = "NCHW"


class AdaptiveAvgPool1dOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType
        self._layout = "NCW"


class AdaptiveAvgPool2dOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType
        self._layout = "NCHW"


class AdaptiveAvgPool3dOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType
        self._layout = "NCDHW"


class AvgPool1dOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType
        self._layout = "NCW"


class CallOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self.call_func_name = ""
        self._op_type = OpType.Unfusable


class CallExternalOp(Op):
    """
    Operation for calling external library functions (e.g., oneDNN).
    This is separate from CallOp to avoid breaking existing functionality.
    Uses RankedTensorType for TOSA dialect compatibility.
    """

    def __init__(
        self,
        call_func_name: str,
        args: List[str],
        args_index: List[int],
        tensor_meta: dict,
        name: str = None,
    ) -> None:
        super().__init__()
        if name is not None:
            self._name = name
        self.call_func_name = call_func_name
        self._arguments = list(args)
        self._args_index = list(args_index)
        self.tensor_meta = tensor_meta
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


class AbsOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class LogOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class CeilOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class FloorOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class MaximumOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.BroadcastType


class MinimumOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.BroadcastType


class BitwiseNotOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class LogicalNotOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class ClampOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class LogicalAndOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.BroadcastType


class LogicalOrOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.BroadcastType


class BitwiseOrOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.BroadcastType


class BitwiseXorOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.BroadcastType


class AminOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class MinDimOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class AvgPool2dOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType
        self._layout = "NCHW"


class LogicalXorOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.BroadcastType


class ProdOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class NegOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class WhereOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.BroadcastType


class EqTensorOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.BroadcastType


class NeTensorOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.BroadcastType


class GtTensorOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.BroadcastType


class GeTensorOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.BroadcastType


class LtTensorOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.BroadcastType


class LeTensorOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.BroadcastType


class Log10Op(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class Log2Op(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class Log1pOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class Expm1Op(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class ConstantPadNdOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class MaskedFillOp(Op):
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


class ArgMinOp(Op):
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


class FlashAttentionForCpuPrefillOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class GeOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class GcdOp(Op):
    """
    Greatest common divisor operation.
    Implements aten.gcd.default.
    """

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


class LeOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class LeScalarOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class LtScalarOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class BitwiseAndTensorOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class IndexPutOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class LiftFreshCopyOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class NeScalarOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class CumsumOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class TensorConstantOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.PlaceholderType


class RepeatOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class RepeatInterleaveOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class AsStridedOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class ScatterSrcOp(Op):
    """
    Scatter operation with source tensor.
    Implements aten.scatter.src: self[index] = src
    Scatters values from src tensor into self tensor at positions specified by index.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class ScatterValueOp(Op):
    """
    Scatter operation with scalar value.
    Implements aten.scatter.value: self[index] = value
    Scatters a scalar value into self tensor at positions specified by index.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class ScatterReduceOp(Op):
    """
    Scatter operation with reduce.
    Implements aten.scatter_reduce: performs scatter with reduction operation.
    Supports reduce operations like 'sum', 'prod', 'mean', 'amax', 'amin'.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class ScatterAddOp(Op):
    """
    Scatter add operation.
    Implements aten.scatter_add: adds all values from the tensor src into self
    at the indices specified in the index tensor.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class SqueezeOp(Op):
    """
    Squeeze operation.
    Implements aten.squeeze: removes dimensions of size 1 from the tensor.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class SqueezeDimOp(Op):
    """
    Squeeze dimension operation.
    Implements aten.squeeze.dim: removes a specific dimension of size 1.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class SortOp(Op):
    """
    Sort operation.
    Implements aten.sort: sorts a tensor along a dimension.
    Returns (sorted_values, indices).
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class TopkOp(Op):
    """
    Top-k operation.
    Implements aten.topk: returns the k largest elements along a dimension.
    Returns (values, indices).
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class CumSumOp(Op):
    """
    Cumulative sum operation.
    Implements aten.cumsum: returns cumulative sum along a dimension.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class CumProdOp(Op):
    """
    Cumulative product operation.
    Implements aten.cumprod: returns cumulative product along a dimension.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class LogCumsumExpOp(Op):
    """
    Log cumulative sum exp operation.
    Implements aten.logcumsumexp.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class UnbindOp(Op):
    """
    Unbind operation.
    Implements aten.unbind: removes a dimension and returns a tuple of tensors.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class GatherOp(Op):
    """
    Gather operation.
    Implements aten.gather: Gathers values along an axis specified by dim.
    For a 3-D tensor the output is specified by:
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class NativeLayerNormOp(Op):
    """
    Native layer normalization operation.
    Implements aten.native_layer_norm: Applies Layer Normalization over a mini-batch.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class NativeGroupNormOp(Op):
    """
    Native group normalization operation.
    Implements aten.native_group_norm: Applies Group Normalization over a mini-batch.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class NativeBatchNormLegitOp(Op):
    """
    Native batch normalization operation (legit version).
    Implements aten._native_batch_norm_legit: Batch normalization with running stats.
    Args: input, weight, bias, running_mean, running_var, training, momentum, eps
    Returns: (output, save_mean, save_invstd)
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class NativeBatchNormLegitNoStatsOp(Op):
    """
    Native batch normalization operation without running stats.
    Implements aten._native_batch_norm_legit.no_stats: Batch normalization without running stats.
    Args: input, weight, bias, training, momentum, eps
    Returns: (output, save_mean, save_invstd)
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class NativeBatchNormLegitNoTrainingOp(Op):
    """
    Native batch normalization operation without training.
    Implements aten._native_batch_norm_legit_no_training: Batch normalization inference only.
    Args: input, weight, bias, running_mean, running_var, momentum, eps
    Returns: (output, save_mean, save_invstd)
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class NativeDropoutOp(Op):
    """
    Native dropout operation.
    Implements aten.native_dropout: Applies dropout to the input tensor.
    Returns (output, mask) where mask is the dropout mask.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class GridSampler2dOp(Op):
    """
    Grid sampler 2D operation.
    Implements aten.grid_sampler_2d: Sample from input using grid coordinates.
    Args: input, grid, interpolation_mode, padding_mode, align_corners
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class Col2imOp(Op):
    """
    Col2im operation (column to image).
    Implements aten.col2im: Rearranges columns back into image blocks.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class UpsampleBilinear2dVecOp(Op):
    """
    Upsample bilinear 2D with vec interface.
    Implements aten.upsample_bilinear2d.vec: Bilinear upsampling.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class UpsampleNearest2dVecOp(Op):
    """
    Upsample nearest 2D with vec interface.
    Implements aten.upsample_nearest2d.vec: Nearest neighbor upsampling.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class SymSizeOp(Op):
    """
    Symbolic size operation.
    Implements aten.sym_size: Get symbolic size of a tensor dimension.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ShapeType


class SymStrideOp(Op):
    """
    Symbolic stride operation.
    Implements aten.sym_stride: Get symbolic stride of a tensor dimension.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ShapeType


class SymNumelOp(Op):
    """
    Symbolic numel operation.
    Implements aten.sym_numel: Get symbolic number of elements.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ShapeType


class SymStorageOffsetOp(Op):
    """
    Symbolic storage offset operation.
    Implements aten.sym_storage_offset: Get symbolic storage offset.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ShapeType


class UnfoldOp(Op):
    """
    Unfold operation.
    Implements aten.unfold: Extract sliding local blocks from a batched input tensor.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class SqueezeDimsOp(Op):
    """
    Squeeze with multiple dimensions operation.
    Implements aten.squeeze.dims: Remove specified dimensions of size 1.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class BaddbmmOp(Op):
    """
    Batched matrix-matrix product with addition.
    Implements aten.baddbmm: batch1 @ batch2 + input * beta + alpha * batch1 @ batch2
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.Unfusable


class LgammaOp(Op):
    """
    Log gamma function.
    Implements aten.lgamma: Natural logarithm of the absolute value of the gamma function.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class DigammaOp(Op):
    """
    Digamma function.
    Implements aten.digamma: Logarithmic derivative of the gamma function.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class I0Op(Op):
    """
    Modified Bessel function of the first kind, order 0.
    Implements aten.i0: I0(x).
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class ErfcOp(Op):
    """
    Complementary error function.
    Implements aten.erfc: 1 - erf(x).
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class CummaxOp(Op):
    """
    Cumulative maximum operation.
    Implements aten.cummax: Returns cumulative maximum and indices along a dimension.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class CumminOp(Op):
    """
    Cumulative minimum operation.
    Implements aten.cummin: Returns cumulative minimum and indices along a dimension.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class ClampMinTensorOp(Op):
    """
    Clamp minimum with tensor operation.
    Implements aten.clamp_min.Tensor: Clamps input to be >= min tensor.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class ClampMaxTensorOp(Op):
    """
    Clamp maximum with tensor operation.
    Implements aten.clamp_max.Tensor: Clamps input to be <= max tensor.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class HypotOp(Op):
    """
    Hypotenuse operation.
    Implements aten.hypot: sqrt(x^2 + y^2).
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class CopysignOp(Op):
    """
    Copy sign operation.
    Implements aten.copysign: Returns x with the sign of y.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class NextafterOp(Op):
    """
    Next after operation.
    Implements aten.nextafter: Returns the next floating-point value after x towards y.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class MaskedScatterOp(Op):
    """
    Masked scatter operation.
    Implements aten.masked_scatter: Copies elements from source to input at positions where mask is True.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class RevOp(Op):
    """
    Reverse operation.
    Implements aten.rev: Reverses a tensor along specified dimensions.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class GeluOp(Op):
    """
    GELU activation function.
    Implements aten.gelu: Gaussian Error Linear Unit activation.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class FlipOp(Op):
    """
    Flip operation.
    Implements aten.flip: Reverse the order of elements along given dimensions.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class LeakyReluOp(Op):
    """
    Leaky ReLU activation function.
    Implements aten.leaky_relu: LeakyReLU(x) = max(0, x) + negative_slope * min(0, x)
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class HardtanhOp(Op):
    """
    Hardtanh activation function.
    Implements aten.hardtanh: Hardtanh(x) = max(min_val, min(max_val, x))
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class EluOp(Op):
    """
    ELU activation function.
    Implements aten.elu: ELU(x) = max(0, x) + min(0, alpha * (exp(x) - 1))
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class SignOp(Op):
    """
    Sign operation.
    Implements aten.sign: Returns a tensor with the signs of the elements of input.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class RoundOp(Op):
    """
    Round operation.
    Implements aten.round: Rounds elements to the nearest integer.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class TruncOp(Op):
    """
    Truncation operation.
    Implements aten.trunc: Returns a tensor with truncated integer values of elements.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class SinhOp(Op):
    """
    Hyperbolic sine operation.
    Implements aten.sinh: sinh(x) = (exp(x) - exp(-x)) / 2
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class CoshOp(Op):
    """
    Hyperbolic cosine operation.
    Implements aten.cosh: cosh(x) = (exp(x) + exp(-x)) / 2
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class TanOp(Op):
    """
    Tangent operation.
    Implements aten.tan: tan(x) = sin(x) / cos(x)
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class Exp2Op(Op):
    """
    Base 2 exponential operation.
    Implements aten.exp2: exp2(x) = 2^x = exp(x * ln(2))
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class ZerosOp(Op):
    """
    Create tensor of zeros.
    Implements aten.zeros: Returns a tensor filled with zeros.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.PlaceholderType


class ZerosLikeOp(Op):
    """
    Create tensor of zeros with same shape.
    Implements aten.zeros_like: Returns a tensor of zeros with the same shape as input.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class OnesLikeOp(Op):
    """
    Create tensor of ones with same shape.
    Implements aten.ones_like: Returns a tensor of ones with the same shape as input.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class FullLikeOp(Op):
    """
    Create tensor filled with a value with same shape.
    Implements aten.full_like: Returns a tensor filled with value with the same shape as input.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class AllOp(Op):
    """
    Reduce all operation.
    Implements aten.all: Returns True if all elements are True.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class AnyOp(Op):
    """
    Reduce any operation.
    Implements aten.any: Returns True if any element is True.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class IsInfOp(Op):
    """
    Check for infinity.
    Implements aten.isinf: Returns a boolean tensor indicating if each element is infinite.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class IsNanOp(Op):
    """
    Check for NaN.
    Implements aten.isnan: Returns a boolean tensor indicating if each element is NaN.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class FloorDivideOp(Op):
    """
    Floor division operation.
    Implements aten.floor_divide: Returns floor of x / y.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class FmodOp(Op):
    """
    Float modulo operation.
    Implements aten.fmod: Returns element-wise remainder of division.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class RemainderOp(Op):
    """
    Remainder operation.
    Implements aten.remainder: Returns element-wise remainder (Python-style).
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class PowTensorTensorOp(Op):
    """
    Tensor-tensor power operation.
    Implements aten.pow.Tensor_Tensor: Returns x^y element-wise.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class SoftplusOp(Op):
    """
    Softplus activation function.
    Implements aten.softplus: softplus(x) = log(1 + exp(x))
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class HardswishOp(Op):
    """
    Hardswish activation function.
    Implements aten.hardswish: x * relu6(x + 3) / 6
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class TileOp(Op):
    """
    Tile operation.
    Implements aten.tile: Constructs a tensor by tiling input.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class StackOp(Op):
    """
    Stack operation.
    Implements aten.stack: Concatenates tensors along a new dimension.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class LerpOp(Op):
    """
    Linear interpolation operation.
    Implements aten.lerp: start + weight * (end - start)
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class ClampTensorOp(Op):
    """
    Clamp operation with tensor bounds.
    Implements aten.clamp.Tensor: Clamp values between min and max tensors.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class AddScalarOp(Op):
    """
    Add scalar operation.
    Implements aten.add.Scalar: Adds a scalar to a tensor.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class SubScalarOp(Op):
    """
    Subtract scalar operation.
    Implements aten.sub.Scalar: Subtracts a scalar from a tensor.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class DivScalarOp(Op):
    """
    Divide by scalar operation.
    Implements aten.div.Scalar: Divides a tensor by a scalar.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class DivScalarModeOp(Op):
    """
    Divide by scalar with rounding mode operation.
    Implements aten.div.Scalar_mode: Divides a tensor by a scalar with rounding.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class PowScalarOp(Op):
    """
    Power with scalar base operation.
    Implements aten.pow.Scalar: Raises a scalar to the power of tensor elements.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class MeanDefaultOp(Op):
    """
    Mean reduction operation (all elements).
    Implements aten.mean.default: Computes the mean of all elements.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class VarCorrectionOp(Op):
    """
    Variance with correction operation.
    Implements aten.var.correction: Computes variance with Bessel correction.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class VarDimOp(Op):
    """
    Variance along dimension operation.
    Implements aten.var.dim: Computes variance along specified dimension.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class AnyDimsOp(Op):
    """
    Any reduction along multiple dimensions.
    Implements aten.any.dims: Tests if any element is True along dims.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class FillScalarOp(Op):
    """
    Fill with scalar operation.
    Implements aten.fill.Scalar: Fills a tensor with a scalar value.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class AcosOp(Op):
    """
    Arccosine operation.
    Implements aten.acos.default: Computes element-wise arccosine.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class AsinOp(Op):
    """
    Arcsine operation.
    Implements aten.asin.default: Computes element-wise arcsine.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class AtanOp(Op):
    """
    Arctangent operation.
    Implements aten.atan.default: Computes element-wise arctangent.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class Atan2Op(Op):
    """
    Two-argument arctangent operation.
    Implements aten.atan2.default: Computes element-wise arctangent of y/x.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class AcoshOp(Op):
    """
    Inverse hyperbolic cosine operation.
    Implements aten.acosh.default: Computes element-wise inverse hyperbolic cosine.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class AsinhOp(Op):
    """
    Inverse hyperbolic sine operation.
    Implements aten.asinh.default: Computes element-wise inverse hyperbolic sine.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class AtanhOp(Op):
    """
    Inverse hyperbolic tangent operation.
    Implements aten.atanh.default: Computes element-wise inverse hyperbolic tangent.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class DiagonalOp(Op):
    """
    Diagonal extraction operation.
    Implements aten.diagonal.default: Extracts diagonal elements.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class AliasOp(Op):
    """
    Alias operation (view with same data).
    Implements aten.alias.default: Creates an alias of the input tensor.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class EmptyOp(Op):
    """
    Empty tensor creation operation.
    Implements aten.empty.memory_format: Creates an uninitialized tensor.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class RandOp(Op):
    """
    Random uniform tensor operation.
    Implements aten.rand.default: Creates a tensor with uniform random values.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class RandnOp(Op):
    """
    Random normal tensor operation.
    Implements aten.randn.default: Creates a tensor with normal random values.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class SelectScatterOp(Op):
    """
    Select scatter operation.
    Implements aten.select_scatter.default: Scatters values at selected indices.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class DiagonalScatterOp(Op):
    """
    Diagonal scatter operation.
    Implements aten.diagonal_scatter.default: Scatters values along a diagonal.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class SplitWithSizesOp(Op):
    """
    Split with specified sizes operation.
    Implements aten.split_with_sizes.default: Splits tensor by given sizes.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class MaxDimOp(Op):
    """
    Maximum along dimension operation.
    Implements aten.max.dim: Returns max values and indices along a dimension.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class NonzeroOp(Op):
    """
    Nonzero elements operation.
    Implements aten.nonzero.default: Returns indices of nonzero elements.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class StdDefaultOp(Op):
    """
    Standard deviation over all elements.
    Implements aten.std.default: Computes std of all elements.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class StdDimOp(Op):
    """
    Standard deviation along dimension.
    Implements aten.std.dim: Computes std along specified dimensions.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class StdCorrectionOp(Op):
    """
    Standard deviation with Bessel correction.
    Implements aten.std.correction: Computes std with correction factor.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class SumDefaultOp(Op):
    """
    Sum over all elements.
    Implements aten.sum.default: Computes sum of all elements.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class AllDimsOp(Op):
    """
    Logical AND reduction over multiple dimensions.
    Implements aten.all.dims: Reduces bool tensor over dims.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class NormScalarOp(Op):
    """
    Norm operation with scalar p.
    Implements aten.norm.Scalar: Computes p-norm over all elements.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class NormScalarOptDimOp(Op):
    """
    Norm operation with optional dimension.
    Implements aten.norm.ScalarOpt_dim: Computes p-norm along dimensions.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class VarDefaultOp(Op):
    """
    Variance over all elements.
    Implements aten.var.default: Computes variance of all elements.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


# =============================================================================
# Backward Operations (Gradient Computation)
# =============================================================================


class AdaptiveAvgPool2dBackwardOp(Op):
    """
    Backward pass for adaptive average pooling 2D.
    Implements aten._adaptive_avg_pool2d_backward:
    Distributes gradients uniformly over pooling regions.

    Args:
        grad_output: Gradient from upstream (N, C, out_h, out_w)
        self: Original input tensor (N, C, H, W)

    Returns:
        grad_input: Gradient w.r.t. input (N, C, H, W)
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.Unfusable
        self._layout = "NCHW"


class AvgPool2dBackwardOp(Op):
    """
    Backward pass for average pooling 2D.
    Implements aten.avg_pool2d_backward:
    Distributes gradients uniformly over pooling regions.

    Args:
        grad_output: Gradient from upstream
        self: Original input tensor
        kernel_size: Pooling kernel size [kH, kW]
        stride: Pooling stride [sH, sW]
        padding: Padding [pH, pW]
        ceil_mode: Whether to use ceil for output size
        count_include_pad: Whether to include padding in average
        divisor_override: Override divisor in average computation

    Returns:
        grad_input: Gradient w.r.t. input
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.Unfusable
        self._layout = "NCHW"


class ConvolutionBackwardOp(Op):
    """
    Backward pass for convolution.
    Implements aten.convolution_backward:
    Computes gradients for input, weight, and bias.

    Args:
        grad_output: Gradient from upstream
        input: Original input tensor
        weight: Convolution weight
        bias_sizes: Size of bias tensor (optional)
        stride: Convolution stride
        padding: Convolution padding
        dilation: Convolution dilation
        transposed: Whether transposed convolution
        output_padding: Output padding for transposed conv
        groups: Number of groups
        output_mask: [3] bool mask for which gradients to compute

    Returns:
        (grad_input, grad_weight, grad_bias): Tuple of gradients
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.Unfusable
        self._layout = "NCHW"


class EmbeddingDenseBackwardOp(Op):
    """
    Backward pass for embedding.
    Implements aten.embedding_dense_backward:
    Accumulates gradients into the embedding weight matrix.

    Args:
        grad_output: Gradient from upstream (*, embedding_dim)
        indices: Indices into embedding table (*)
        num_weights: Number of embeddings in table
        padding_idx: Index to ignore (-1 for none)
        scale_grad_by_freq: Whether to scale by frequency

    Returns:
        grad_weight: Gradient w.r.t. embedding weight (num_weights, embedding_dim)
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.Unfusable


class MaxPool2dWithIndicesBackwardOp(Op):
    """
    Backward pass for max pooling 2D with indices.
    Implements aten.max_pool2d_with_indices_backward:
    Scatters gradients to positions specified by indices.

    Args:
        grad_output: Gradient from upstream
        self: Original input tensor
        kernel_size: Pooling kernel size [kH, kW]
        stride: Pooling stride [sH, sW]
        padding: Padding [pH, pW]
        dilation: Dilation [dH, dW]
        ceil_mode: Whether to use ceil for output size
        indices: Indices from forward max_pool2d_with_indices

    Returns:
        grad_input: Gradient w.r.t. input
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.Unfusable
        self._layout = "NCHW"


class NativeGroupNormBackwardOp(Op):
    """
    Backward pass for group normalization.
    Implements aten.native_group_norm_backward:
    Computes gradients for input, weight, and bias.

    Args:
        grad_out: Gradient from upstream (N, C, *)
        input: Original input tensor (N, C, *)
        mean: Mean from forward pass (N, num_groups)
        rstd: Reciprocal std from forward pass (N, num_groups)
        weight: Affine weight (C,) or None
        N: Batch size
        C: Number of channels
        HxW: Spatial size (H * W)
        group: Number of groups
        output_mask: [3] bool mask for which gradients to compute

    Returns:
        (grad_input, grad_weight, grad_bias): Tuple of gradients
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.Unfusable


class NativeLayerNormBackwardOp(Op):
    """
    Backward pass for layer normalization.
    Implements aten.native_layer_norm_backward:
    Computes gradients for input, weight, and bias.

    Args:
        grad_out: Gradient from upstream
        input: Original input tensor
        normalized_shape: Shape over which to normalize
        mean: Mean from forward pass
        rstd: Reciprocal std from forward pass
        weight: Affine weight or None
        bias: Affine bias or None
        output_mask: [3] bool mask for which gradients to compute

    Returns:
        (grad_input, grad_weight, grad_bias): Tuple of gradients
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.Unfusable


# === Bitwise Scalar Operations ===


class BitwiseAndScalarOp(Op):
    """Bitwise AND with scalar: tensor & scalar"""

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class BitwiseOrScalarOp(Op):
    """Bitwise OR with scalar: tensor | scalar"""

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class BitwiseXorScalarOp(Op):
    """Bitwise XOR with scalar: tensor ^ scalar"""

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


# === Padding Operations ===


class ReflectionPad1dOp(Op):
    """1D reflection padding"""

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class ReflectionPad2dOp(Op):
    """2D reflection padding"""

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class ReflectionPad3dOp(Op):
    """3D reflection padding"""

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class ReplicationPad2dOp(Op):
    """2D replication padding"""

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class ReplicationPad3dOp(Op):
    """3D replication padding"""

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


# === Other Operations ===


class EmptyStridedOp(Op):
    """Create empty tensor with specified strides"""

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.PlaceholderType


class RandpermOp(Op):
    """Random permutation of integers 0 to n-1"""

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.PlaceholderType


# === Core Aten Remaining Operations ===


class EmbeddingBagOp(Op):
    """Embedding bag operation with aggregation (sum, mean, max)"""

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class CdistForwardOp(Op):
    """Compute pairwise distance between two sets of vectors"""

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class PdistForwardOp(Op):
    """Compute pairwise distance within a set of vectors"""

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class FftR2cOp(Op):
    """Real-to-complex FFT transform"""

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class LocalScalarDenseOp(Op):
    """Convert single-element tensor to scalar"""

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class ResizeOp(Op):
    """Resize tensor in-place"""

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class GQAAttentionFusedOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class AsStridedScatterOp(Op):
    """
    Scatter into a base tensor via as_strided view.
    Implements aten.as_strided_scatter.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class BitwiseLeftShiftOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.BroadcastType


class BitwiseRightShiftOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.BroadcastType


class FractionalMaxPool2dOp(Op):
    """
    Fractional max pool 2D operation.
    Implements aten.fractional_max_pool2d.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType
        self._layout = "NCHW"


class FrexpOp(Op):
    """
    Frexp decomposition.
    Implements aten.frexp: Returns mantissa and exponent.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class GridSampler3dOp(Op):
    """
    Grid sampler 3D operation.
    Implements aten.grid_sampler_3d: Sample from input using grid coordinates.
    Args: input, grid, interpolation_mode, padding_mode, align_corners
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class HistcOp(Op):
    """
    Histogram count operation.
    Implements aten.histc.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class IgammaOp(Op):
    """
    Lower regularized incomplete gamma.
    Implements aten.igamma: igamma(a, x).
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class IgammacOp(Op):
    """
    Upper regularized incomplete gamma.
    Implements aten.igammac: igammac(a, x).
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class KthValueOp(Op):
    """
    Kth value operation.
    Implements aten.kthvalue: returns k-th smallest values and indices.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class MedianOp(Op):
    """
    Median reduction operation.
    Implements aten.median (default and dim variants).
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class ModeOp(Op):
    """
    Mode reduction operation.
    Implements aten.mode.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class NanMedianOp(Op):
    """
    NaN-aware median reduction operation.
    Implements aten.nanmedian (default and dim variants).
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class NewEmptyStridedOp(Op):
    """
    Empty tensor creation with explicit stride.
    Implements aten.new_empty_strided.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class NonzeroStaticOp(Op):
    """
    Nonzero elements with fixed output size.
    Implements aten.nonzero_static.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class SearchSortedOp(Op):
    """
    Searchsorted operation.
    Implements aten.searchsorted on tensors.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReduceType


class SignbitOp(Op):
    """
    Signbit operation.
    Implements aten.signbit: Returns a boolean tensor indicating negative values.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class SpecialEntrOp(Op):
    """
    Elementwise entropy.
    Implements aten.special_entr.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class SpecialI0eOp(Op):
    """
    Scaled modified Bessel function of the first kind, order 0.
    Implements aten.special_i0e.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class SpecialI1Op(Op):
    """
    Modified Bessel function of the first kind, order 1.
    Implements aten.special_i1.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class SpecialI1eOp(Op):
    """
    Scaled modified Bessel function of the first kind, order 1.
    Implements aten.special_i1e.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class SpecialErfcxOp(Op):
    """
    Scaled complementary error function.
    Implements aten.special_erfcx: exp(x^2) * erfc(x).
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class SpecialNdtrOp(Op):
    """
    Standard normal CDF.
    Implements aten.special_ndtr.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class SpecialLogNdtrOp(Op):
    """
    Log CDF of the standard normal distribution.
    Implements aten.special_log_ndtr.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class SpecialNdtriOp(Op):
    """
    Inverse of the standard normal CDF.
    Implements aten.special_ndtri.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class SpecialSphericalBesselJ0Op(Op):
    """
    Spherical Bessel function of the first kind, order 0.
    Implements aten.special_spherical_bessel_j0.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType


class UniformOp(Op):
    """
    Uniform random fill operation.
    Implements aten.uniform / aten.uniform_.
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ReshapeType


class BernoulliOp(Op):
    """
    Bernoulli sampling operation.
    Implements aten.bernoulli.* variants (functional forms).
    """

    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.ElementwiseType
