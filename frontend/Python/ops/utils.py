# ===- utils.py ----------------------------------------------------------------
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
# The element utils of mlir element type and attribute.
#
# ===---------------------------------------------------------------------------

from typing import Dict
import mlir.ir as ir

from ..graph import TensorDType


def mlir_element_type_get(type_name):
    """
    Get the mlir element type base on TensorDType's enum type.
    Args:
        type_name: The TensorDType's enum type.
    """
    match type_name:
        case TensorDType.Float32:
            return ir.F32Type.get()
        case TensorDType.Int64:
            return ir.IntegerType.get_signless(64)
        case TensorDType.Bool:
            return ir.IntegerType.get_signless(1)


def mlir_element_attr_get(type_name, value):
    """
    Get the mlir element attribute base on TensorDType's enum type and value.
    Args:
        type_name: The TensorDType's enum type.
        value: The real value for mlir element attribute.
    """
    match type_name:
        case TensorDType.Float32:
            return ir.FloatAttr.get(ir.F32Type.get(), value)
        case TensorDType.Int64:
            return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), value)
        case TensorDType.Bool:
            return ir.IntegerAttr.get(ir.IntegerType.get_signless(1), value)


def tensor_shape_size(shape):
    """
    Calculate the product of all dimensions in the given shape list, 
    which represents the size of the tensor.
    Args:
        shape: A list containing the sizes of each dimension of the tensor.
    """
    size = 1
    for dim in shape:
        size *= dim
    return size

def generate_strides(shape):
    """
    Generate strides based on the input matrix shape.
    
    Args:
        shape (list[int]): The shape of the input matrix, e.g., [2, 3, 4].

    Returns:
        list[int]: The corresponding strides, e.g., [12, 4, 1].
    """
    strides = []
    stride = 1
    for dim in reversed(shape):
        strides.insert(0, stride)
        stride *= dim
    return strides

def transpose_strides(strides, permutation):
    """
    Reorder strides based on the input permutation.
    
    Args:
        strides (list[int]): The original strides list, e.g., [12, 4, 1].
        permutation (list[int]): The permutation order, e.g., [1, 2, 0].

    Returns:
        list[int]: The reordered strides list, e.g., [4, 1, 12].
    """
    transposed_strides = [strides[i] for i in permutation]
    return transposed_strides
