# ===- type.py -----------------------------------------------------------------
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
# This is the tensor type of the Buddy Compiler frontend.
#
# ===---------------------------------------------------------------------------

from enum import Enum


class TensorDType(Enum):
    """
    Enum class for declaring tensor data types.

    Members:
    - Int8: str
        Represents the 8-bit integer data type.
    - Int32: str
        Represents the 32-bit integer data type.
    - Int64: str
        Represents the 64-bit integer data type.
    - Float16: str
        Represents the 16-bit floating-point data type.
    - BFloat16: str
        Represents the 16-bit brain floating-point data type.
    - Float32: str
        Represents the 32-bit floating-point data type.
    - Float64: str
        Represents the 64-bit floating-point data type.
    - Bool: str
        Represents the boolean data type.
    """

    Int8 = "int8"
    Int32 = "int32"
    Int64 = "int64"
    Float16 = "float16"
    BFloat16 = "bfloat16"
    Float32 = "float32"
    Float64 = "float64"
    Bool = "bool"


class TensorMeta:
    """
    Store tensor metadata, including shape and data type.
    Use None or -1 for dynamic dimensions (e.g., shape=(1, None) for variable sequence length).
    """

    def __init__(self, shape, dtype) -> None:
        self.shape = shape
        self.dtype = dtype

    def is_dynamic(self) -> bool:
        """Check if any dimension is dynamic (None or -1)."""
        if self.shape is None:
            return False
        return any(dim is None or dim == -1 for dim in self.shape)

    def get_mlir_shape_string(self) -> str:
        """Convert shape to MLIR format: (1, None, 128) -> '1x?x128'."""
        if self.shape is None or len(self.shape) == 0:
            return ""
        shape_parts = []
        for dim in self.shape:
            if dim is None or dim == -1:
                shape_parts.append("?")
            else:
                shape_parts.append(str(dim))
        return "x".join(shape_parts)


class DeviceType(Enum):
    """
    Enumeration class representing different types of devices.

    Attributes:
    - CPU: Central Processing Unit.
    - GPU: Graphics Processing Unit.
    - UNKNOW: Unknown device type.

    Each attribute represents a specific device type and is associated with a
    string value.
    """

    CPU = "cpu"
    GPU = "gpu"
    UNKNOW = "unknow"
