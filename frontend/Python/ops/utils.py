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

import torch
import array
from typing import Dict, List, Tuple, Union

import mlir.ir as ir

from ..graph import Tensordtype

def mlir_element_type_get(type_name):
    """
    Get the element type base on type_name.
    """
    match type_name:
        case Tensordtype.Float32:
            return ir.F32Type.get()
        case Tensordtype.Int64:
            return ir.IntegerType.get_signless(64)
        case Tensordtype.Bool:
            return ir.IntegerType.get_signless(1)

def mlir_element_attr_get(type_name, value):
    """
    Get the element attribute base on type_name and value.
    """
    match type_name:
        case Tensordtype.Float32:
            return ir.FloatAttr.get(ir.F32Type.get(), value)
        case Tensordtype.Int64:
            return ir.IntegerAttr.get(ir.IntegerType.get_signless(64), value)
        case Tensordtype.Bool:
            return ir.IntegerAttr.get(ir.IntegerType.get_signless(1), value)
