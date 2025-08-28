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
    - Int32: str
        Represents the 32-bit integer data type.
    - Int64: str
        Represents the 64-bit integer data type.
    - Float32: str
        Represents the 32-bit floating-point data type.
    - Bool: str
        Represents the boolean data type.
    """
    
    Int8 = "int8"
    Int32 = "int32"
    Int64 = "int64"
    Float16 = "float16"
    Float32 = "float32"
    Float64 = "float64"
    Bool = "bool"


class TensorMeta:
    """
    Store tensor metadata, including shape and data type, while overlooking raw 
    data.

    Attributes:
    - shape: tuple
        Represents the shape of the tensor.
    - dtype: str
        Represents the data type of the tensor.

    Methods:
    - __init__(shape: tuple, dtype: str) -> None:
        Initializes a new instance of the TensorMeta class with the specified 
        shape and data type.

    Example:
    meta = TensorMeta(shape=(3, 4), dtype='float32')
    # Access metadata attributes: meta.shape, meta.dtype
    """

    def __init__(self, shape, dtype) -> None:
        """
        Initialize a new instance of the TensorMeta class.

        Parameters:
        - shape: tuple
            Represents the shape of the tensor.
        - dtype: str
            Represents the data type of the tensor.
        """
        self.shape = shape
        self.dtype = dtype

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
    CPU = 'cpu'
    GPU = 'gpu'
    UNKNOW = 'unknow'
