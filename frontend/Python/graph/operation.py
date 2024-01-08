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
        self._keyword_arguments = {}
        self._tensor_meta = {}
        self._op_type = None
        self._device = "cpu"

    def add_argument(self, arg):
        self._arguments.append(arg)

    def set_device(self, device):
        self._device = device

    @property
    def args(self):
        return self._arguments

    @property
    def kwargs(self):
        return self._keyword_arguments

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
